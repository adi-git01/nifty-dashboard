import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os

warnings.filterwarnings('ignore')

# ----------------- Configuration -----------------
from utils.nifty500_list import TICKERS

# We'll test on the top 150 liquid stocks for speed over 5 years
UNIVERSE = TICKERS[:150]
MAX_YEARS_DATA = 6 # Need 6 years of data to test a 5-year horizon (1 yr lookback for MA200)

MAX_POSITIONS = 10
CASH_DEFAULT = 0.05
INITIAL_CAP = 1000000
TRAILING_STOP_PCT = 0.12

# Composite RS Weights
W_1W = 0.30
W_1M = 0.30
W_3M = 0.40

def calculate_timeframe_rs(stock_df, nifty_df, days):
    """Calculates RS for a specific trading day lookback"""
    if len(stock_df) < days:
        return 0.0
    try:
        s_close_now = float(stock_df['Close'].iloc[-1])
        s_close_past = float(stock_df['Close'].iloc[-days])
        n_close_now = float(nifty_df['Close'].iloc[-1])
        n_close_past = float(nifty_df['Close'].iloc[-days])
        
        s_ret = (s_close_now - s_close_past) / s_close_past
        n_ret = (n_close_now - n_close_past) / n_close_past
        return float((s_ret - n_ret) * 100)
    except:
        return 0.0

def simulate_portfolio(tickers, start_date, end_date):
    """Runs a duel simulation between pure 63-day RS and a Composite RS"""
    print(f"\n[{start_date} -> {end_date}] Initializing Universe Download...")
    
    # We need a 1-year buffer before the start_date to calculate 200DMA and 3-month RS on day 1
    sim_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
    
    data = yf.download(tickers, start=sim_start, end=end_date, group_by='ticker', threads=True, progress=False, auto_adjust=True)
    nifty = yf.download('^CRSLDX', start=sim_start, end=end_date, threads=True, progress=False, auto_adjust=True)
    
    dates = nifty.loc[start_date:end_date].index
    if len(dates) < 5:
        return None
        
    base_pos = {} # Dict of ticker: {entry_price, max_price}
    comp_pos = {}
    
    base_cash = INITIAL_CAP
    comp_cash = INITIAL_CAP
    
    base_final = INITIAL_CAP
    comp_final = INITIAL_CAP
    
    rebelance_freq = 15 # Check for new entries every 15 trading days
    day_count = 0
    
    for current_date in dates:
        # 1. Update trailing stops & exits
        # BASE PORTFOLIO EXIT
        for t in list(base_pos.keys()):
            if t in data and current_date in data[t].index:
                row = data[t].loc[current_date]
                if pd.isna(row['Close']): continue
                
                # Update max price
                if row['Close'] > base_pos[t]['max_price']:
                    base_pos[t]['max_price'] = row['Close']
                
                # Check trailing stop
                stop_price = base_pos[t]['max_price'] * (1 - TRAILING_STOP_PCT)
                if row['Close'] < stop_price:
                    # Exit
                    ret = (row['Close'] / base_pos[t]['entry_price']) - 1
                    base_cash += (base_pos[t]['shares'] * row['Close'])
                    del base_pos[t]
                    
        # COMPOSITE PORTFOLIO EXIT
        for t in list(comp_pos.keys()):
            if t in data and current_date in data[t].index:
                row = data[t].loc[current_date]
                if pd.isna(row['Close']): continue
                
                # Update max price
                if row['Close'] > comp_pos[t]['max_price']:
                    comp_pos[t]['max_price'] = row['Close']
                
                # Check trailing stop
                stop_price = comp_pos[t]['max_price'] * (1 - TRAILING_STOP_PCT)
                if row['Close'] < stop_price:
                    # Exit
                    comp_cash += (comp_pos[t]['shares'] * row['Close'])
                    del comp_pos[t]
                    
        # 2. Rebalance (Scan for new entries)
        if day_count % rebelance_freq == 0:
            # Need Nifty context
            n_idx = nifty.index.get_loc(current_date)
            if n_idx < 65: # Need at least 63 days of history
                day_count += 1
                continue
                
            n_window = nifty.iloc[:n_idx+1]
            cands = []
            
            for t in tickers:
                if t in data:
                    df = data[t]
                    if current_date not in df.index: continue
                    t_idx = df.index.get_loc(current_date)
                    if t_idx < 205: continue # Need 200DMA 
                    
                    window = df.iloc[:t_idx+1]
                    price = window['Close'].iloc[-1]
                    ma50 = window['Close'].rolling(50).mean().iloc[-1]
                    ma200 = window['Close'].rolling(200).mean().iloc[-1]
                    
                    if pd.isna(ma50) or pd.isna(ma200) or price < ma50 or price < ma200:
                        continue # Trend filter
                        
                    # Calculate multi-timeframe RS
                    rs_1w = calculate_timeframe_rs(window, n_window, 5)   # 1 Week ~ 5 trading days
                    rs_1m = calculate_timeframe_rs(window, n_window, 21)  # 1 Month ~ 21 trading days
                    rs_3m = calculate_timeframe_rs(window, n_window, 63)  # 3 Months ~ 63 trading days
                    
                    # Define the two scores
                    baseline_score = rs_3m
                    composite_score = (rs_1w * W_1W) + (rs_1m * W_1M) + (rs_3m * W_3M)
                    
                    cands.append({
                        'ticker': t, 
                        'price': price,
                        'base_score': baseline_score,
                        'comp_score': composite_score
                    })
            
            # --- EXECUTE BASELINE BUYS ---
            avail_slots = MAX_POSITIONS - len(base_pos)
            if avail_slots > 0 and cands:
                # Require absolute minimum RS for baseline
                valid_cands = [x for x in cands if x['base_score'] >= 10.0 and x['ticker'] not in base_pos]
                valid_cands.sort(key=lambda x: x['base_score'], reverse=True)
                
                target_alloc = (base_cash * (1 - CASH_DEFAULT)) / dict(base=avail_slots).get('base', avail_slots) if avail_slots else 0
                for c in valid_cands[:avail_slots]:
                    alloc = min(target_alloc, base_cash)
                    if alloc < 5000: break
                    shares = alloc / c['price']
                    base_pos[c['ticker']] = {
                        'entry_price': c['price'],
                        'max_price': c['price'],
                        'shares': shares
                    }
                    base_cash -= alloc
                    
            # --- EXECUTE COMPOSITE BUYS ---
            avail_slots_comp = MAX_POSITIONS - len(comp_pos)
            if avail_slots_comp > 0 and cands:
                # Require absolute minimum composite score
                valid_cands_comp = [x for x in cands if x['comp_score'] >= 3.0 and x['ticker'] not in comp_pos]
                valid_cands_comp.sort(key=lambda x: x['comp_score'], reverse=True)
                
                target_alloc_comp = (comp_cash * (1 - CASH_DEFAULT)) / dict(comp=avail_slots_comp).get('comp', avail_slots_comp) if avail_slots_comp else 0
                for c in valid_cands_comp[:avail_slots_comp]:
                    alloc = min(target_alloc_comp, comp_cash)
                    if alloc < 5000: break
                    shares = alloc / c['price']
                    comp_pos[c['ticker']] = {
                        'entry_price': c['price'],
                        'max_price': c['price'],
                        'shares': shares
                    }
                    comp_cash -= alloc

        # End of day equity calculation
        current_base_eq = base_cash
        for t, p in base_pos.items():
            if t in data and current_date in data[t].index:
                row = data[t].loc[current_date]
                if not pd.isna(row['Close']):
                    current_base_eq += (p['shares'] * row['Close'])
        base_final = current_base_eq
        
        current_comp_eq = comp_cash
        for t, p in comp_pos.items():
            if t in data and current_date in data[t].index:
                row = data[t].loc[current_date]
                if not pd.isna(row['Close']):
                    current_comp_eq += (p['shares'] * row['Close'])
        comp_final = current_comp_eq

        day_count += 1
        
    # Calculate final stats
    days_total = (dates[-1] - dates[0]).days
    years = max(1, days_total / 365.25)
    
    base_cagr = ((base_final / INITIAL_CAP) ** (1/years) - 1) * 100
    comp_cagr = ((comp_final / INITIAL_CAP) ** (1/years) - 1) * 100
    
    n_start = float(nifty.loc[dates[0]]['Close'].iloc[0] if isinstance(nifty.loc[dates[0]]['Close'], pd.Series) else nifty.loc[dates[0]]['Close']) if len(nifty) > 0 else 1.0
    n_end = float(nifty.loc[dates[-1]]['Close'].iloc[0] if isinstance(nifty.loc[dates[-1]]['Close'], pd.Series) else nifty.loc[dates[-1]]['Close']) if len(nifty) > 0 else 1.0
    n_cagr = float(((n_end / n_start) ** (1/years) - 1) * 100)
    
    return {
        'years': years,
        'base_cagr': base_cagr,
        'comp_cagr': comp_cagr,
        'nifty_cagr': n_cagr
    }

if __name__ == "__main__":
    print("==========================================================")
    print("THE ARENA: DNA3-V3.1 (63-Day RS) vs MULTI-TIMEFRAME RS")
    print("==========================================================")
    
    today = datetime.now()
    horizons = {
        "6 Mo (Current Regime)": (today - relativedelta(months=6)).strftime('%Y-%m-%d'),
        "1 Yr (Recent Bull)": (today - relativedelta(years=1)).strftime('%Y-%m-%d'),
        "3 Yr (Post-Covid)": (today - relativedelta(years=3)).strftime('%Y-%m-%d'),
        "5 Yr (Full Cycle)": (today - relativedelta(years=5)).strftime('%Y-%m-%d')
    }
    
    end_date = today.strftime('%Y-%m-%d')
    
    results = []
    
    for label, start_date in horizons.items():
        metrics = simulate_portfolio(UNIVERSE, start_date, end_date)
        if metrics:
            results.append({
                "Horizon": label,
                "Baseline (63-Day RS)": metrics['base_cagr'],
                "Composite RS": metrics['comp_cagr'],
                "Nifty Benchmark": metrics['nifty_cagr']
            })
            
    print("\n\n" + "="*80)
    print("FINAL CAGE MATCH RESULTS: COMPOUNDED ANNUAL GROWTH RATE (CAGR)")
    print("="*80)
    
    df = pd.DataFrame(results)
    
    for _, row in df.iterrows():
        base = row['Baseline (63-Day RS)']
        comp = row['Composite RS']
        nifty = row['Nifty Benchmark']
        
        winner = "COMPOSITE" if comp > base else "BASELINE (63-Day)"
        diff = comp - base
        
        print(f"\n{row['Horizon'].upper()} ({nifty:.1f}% Benchmark):")
        print(f"   Baseline (63-Day RS): {base:+.1f}%")
        print(f"   Composite (MWQ) RS:   {comp:+.1f}%")
        print(f"   Winner: {winner} (Delta: {diff:+.1f}%)")
        
    print("\n" + "="*80)
