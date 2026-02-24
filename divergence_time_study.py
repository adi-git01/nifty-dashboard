import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import sys
import os

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath('main.py')))
from utils.nifty500_list import TICKERS

OUTPUT_DIR = "analysis_2026/divergence"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Horizons to test
HORIZONS = {'6mo': 0.5, '1y': 1, '3y': 3, '5y': 5, '10y': 10, '15y': 15}

def get_trend(series, window=20):
    if len(series) < window: return "FLAT"
    pct_change = (series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100
    if pct_change > 5: return "UP"
    elif pct_change < -5: return "DOWN"
    return "FLAT"

def fetch_data(years=15):
    start = (datetime.now() - timedelta(days=365 * years + 100)).strftime('%Y-%m-%d')
    print("Fetching Nifty...")
    nifty = yf.Ticker('^NSEI').history(start=start)
    nifty.index = nifty.index.tz_localize(None)
    
    print(f"Bulk downloading 300 stocks for {years} Years...")
    # Using 300 to balance speed and statistical size across 15 years
    bulk = yf.download(TICKERS[:300], start=start, progress=False, threads=True)
    
    cache = {}
    for t in TICKERS[:300]:
        try:
            if isinstance(bulk.columns, pd.MultiIndex):
                if t in bulk.columns.get_level_values(1):
                    df = bulk.xs(t, axis=1, level=1).dropna(how='all')
                    if len(df) > 200:
                        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                        cache[t] = df
            else:
                if t in bulk.columns:
                    df = bulk[t].dropna()
                    if len(df) > 200:
                        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                        cache[t] = df
        except: pass
    print(f"Loaded {len(cache)} valid stocks.")
    return nifty, cache

def run_backtest(nifty, cache, start_date, variant_name):
    capital = 1_000_000
    positions = {}
    trades = []
    equity_curve = []
    
    dates = nifty.index[nifty.index >= start_date]
    day_counter = 0
    
    for date in dates:
        # 1. Processing Exits
        to_remove = []
        for t, pos in positions.items():
            if t in cache:
                df = cache[t]
                idx = df.index.searchsorted(date)
                if idx > 0 and idx < len(df):
                    price = df['Close'].iloc[idx]
                    ma50 = df['Close'].iloc[:idx+1].rolling(50).mean().iloc[-1]
                    
                    if price > pos['peak']: pos['peak'] = price
                    
                    sell_reason = None
                    # Base Exits
                    if price < pos['peak'] * 0.85: sell_reason = 'TSL_15'
                    elif price < ma50: sell_reason = 'MA50'
                    
                    # Variant Explicit Exits
                    if 'Exhaustion Exit' in variant_name:
                        cur_ret = (price - pos['entry']) / pos['entry'] * 100
                        if cur_ret > 10: # Only look for exhaustion if we are in profit
                            window = df.iloc[max(0, idx-20):idx+1]
                            p_trend = get_trend(window['Close'], 20)
                            v_trend = get_trend(window['Volume'].rolling(5).mean().dropna(), 15)
                            if p_trend == 'UP' and v_trend == 'DOWN':
                                sell_reason = 'EXHAUSTION (Vol Drop)'
                    
                    if sell_reason:
                        ret = (price - pos['entry']) / pos['entry']
                        capital += pos['shares'] * price
                        trades.append({
                            'Ticker': t, 'PnL%': ret * 100, 'Reason': sell_reason,
                            'Entry_Date': pos['entry_date'], 'Exit_Date': date
                        })
                        to_remove.append(t)
        for t in to_remove: del positions[t]
        
        # 2. Processing Entries (Rebalance every 13 days - OptComp-V21)
        if day_counter % 13 == 0 and len(positions) < 10:
            candidates = []
            for t, df in cache.items():
                if t in positions: continue
                idx = df.index.searchsorted(date)
                if idx < 100 or idx >= len(df): continue
                
                window = df.iloc[:idx+1]
                price = window['Close'].iloc[-1]
                ma50 = window['Close'].rolling(50).mean().iloc[-1]
                if price < ma50: continue
                
                # Liquidity approx
                val = window['Volume'].iloc[-5:].mean() * price
                if val < 10_000_000: continue
                
                nifty_idx = nifty.index.searchsorted(date)
                if nifty_idx < 63: continue
                
                # OptComp RS Calculation (10% 1W, 50% 1M, 40% 3M)
                def get_rs(days):
                    t_ret = (price - window['Close'].iloc[-days]) / window['Close'].iloc[-days]
                    n_ret = (nifty['Close'].iloc[nifty_idx] - nifty['Close'].iloc[nifty_idx-days]) / nifty['Close'].iloc[nifty_idx-days]
                    return (t_ret - n_ret) * 100
                
                rs_1w = get_rs(5)
                rs_1m = get_rs(21)
                rs_3m = get_rs(63)
                comp_rs = (rs_1w * 0.10) + (rs_1m * 0.50) + (rs_3m * 0.40)
                
                if comp_rs < 0: continue
                
                # Divergence / Volume Logic
                valid = True
                score_boost = 0
                
                if 'UD_Vol>1.2' in variant_name:
                    price_change = window['Close'].diff()
                    up_vol = window['Volume'].where(price_change > 0, 0).rolling(50).sum()
                    down_vol = window['Volume'].where(price_change < 0, 0).rolling(50).sum()
                    ud_ratio = float(up_vol.iloc[-1] / down_vol.iloc[-1]) if down_vol.iloc[-1] > 0 else 1.0
                    if ud_ratio < 1.2: valid = False
                        
                if 'Springboard' in variant_name:
                    # Look for Price DOWN, RS UP divergence to boost score
                    p_trend = get_trend(window['Close'][-20:], 20)
                    rs_start = get_rs(20) # Simplification: use current 20d RS vs what it was 20d ago
                    # Rough proxy:
                    rs_is_strong = comp_rs > 10
                    if p_trend == 'DOWN' and rs_is_strong:
                        score_boost = 50 # Massive boost to rank it first
                
                if valid:
                    candidates.append((t, comp_rs + score_boost, price))
            
            # Rank and execute
            candidates.sort(key=lambda x: -x[1])
            free_slots = 10 - len(positions)
            for t, score, price in candidates[:free_slots]:
                size = capital / (free_slots + 1)
                shares = int(size / price)
                if shares > 0:
                    capital -= (shares * price)
                    positions[t] = {'entry': price, 'peak': price, 'shares': shares, 'entry_date': date}
        
        # Track Equity
        eq = capital
        for t, pos in positions.items():
            df = cache[t]
            idx = df.index.searchsorted(date)
            if idx < len(df): eq += pos['shares'] * df['Close'].iloc[idx]
        equity_curve.append({'Date': date, 'Equity': eq})
        day_counter += 1
        
    return pd.DataFrame(trades), pd.DataFrame(equity_curve)

def run_suite():
    nifty, cache = fetch_data(15)
    
    variants = [
        'Baseline (OptComp-V21)',
        'OptComp + UD_Vol>1.2',
        'OptComp + Exhaustion Exit',
        'OptComp + Springboard Boost'
    ]
    
    all_results = []
    yearly_results = []
    
    # Run across horizons
    for label, years in HORIZONS.items():
        print(f"\n--- Testing Horizon: {label} ({years} Years) ---")
        start_date = nifty.index[-1] - timedelta(days=int(365.25 * years))
        if start_date < nifty.index[0]: start_date = nifty.index[0]
        
        # Nifty Return for Benchmarking
        base_nifty = nifty.loc[start_date:]['Close']
        nifty_cagr = ((base_nifty.iloc[-1] / base_nifty.iloc[0]) ** (1/years) - 1) * 100 if len(base_nifty)>0 else 0
        
        for name in variants:
            trades, eq_curve = run_backtest(nifty, cache, start_date, name)
            
            if eq_curve.empty: continue
            
            eq = eq_curve['Equity'].values
            if len(eq) < 20: continue
            
            cagr = ((eq[-1] / eq[0]) ** (1/max(1, years)) - 1) * 100
            
            peak = pd.Series(eq).cummax()
            dd = (pd.Series(eq) - peak) / peak * 100
            max_dd = dd.min()
            
            win_rate = (trades['PnL%'] > 0).mean() * 100 if not trades.empty else 0
            
            all_results.append({
                'Horizon': label,
                'Variant': name,
                'CAGR%': cagr,
                'Alpha%': cagr - nifty_cagr,
                'MaxDD%': max_dd,
                'WinRate%': win_rate,
                'Trades': len(trades)
            })
            
            # --- YEARLY BREAKDOWN (Only for the 15Y full run to avoid spamming) ---
            if years == 15 and name in ['Baseline (OptComp-V21)', 'OptComp + UD_Vol>1.2']:
                eq_curve['Year'] = eq_curve['Date'].dt.year
                yearly = eq_curve.groupby('Year').agg(
                    Start=('Equity', 'first'),
                    End=('Equity', 'last')
                )
                yearly['Ret%'] = (yearly['End'] - yearly['Start']) / yearly['Start'] * 100
                for year, row in yearly.iterrows():
                    yearly_results.append({'Variant': name, 'Year': year, 'Return%': row['Ret%']})

    # Output Results
    res_df = pd.DataFrame(all_results)
    print("\n" + "="*80)
    print("OPTCOMP-V21 VOLUME/DIVERGENCE THESIS BACKTEST (HORIZON COMPARISON)")
    print("="*80)
    for horizon in ['6mo', '1y', '3y', '5y', '10y', '15y']:
        print(f"\nHorizon: {horizon}")
        sub = res_df[res_df['Horizon'] == horizon]
        if not sub.empty:
            print(sub[['Variant', 'CAGR%', 'Alpha%', 'MaxDD%', 'WinRate%', 'Trades']].to_string(index=False, float_format="%.1f"))
            
    print("\n" + "="*80)
    print("YEARLY BREAKDOWN (15-Year Run): Baseline vs Up/Down Volume")
    print("="*80)
    yr_df = pd.DataFrame(yearly_results).pivot(index='Year', columns='Variant', values='Return%')
    if not yr_df.empty:
        yr_df['Edge% (Vol vs Base)'] = yr_df['OptComp + UD_Vol>1.2'] - yr_df['Baseline (OptComp-V21)']
        print(yr_df.round(1).to_string())

if __name__ == '__main__':
    run_suite()
