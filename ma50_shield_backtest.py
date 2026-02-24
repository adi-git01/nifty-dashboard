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

OUTPUT_DIR = "analysis_2026/ma50_filter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HORIZONS = {'1y': 1, '3y': 3, '5y': 5, '10y': 10, '15y': 15}

def fetch_data(years=15):
    start = (datetime.now() - timedelta(days=365 * years + 100)).strftime('%Y-%m-%d')
    print("Fetching Nifty...")
    nifty = yf.Ticker('^NSEI').history(start=start)
    nifty.index = nifty.index.tz_localize(None)
    nifty['MA50'] = nifty['Close'].rolling(50).mean()
    nifty['Dist_MA50'] = (nifty['Close'] - nifty['MA50']) / nifty['MA50'] * 100
    
    print(f"Bulk downloading 300 stocks...")
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

def run_backtest(nifty, cache, start_date, apply_ma50_filter=False):
    capital = 1_000_000
    positions = {}
    trades = []
    equity_curve = []
    
    dates = nifty.index[nifty.index >= start_date]
    day_counter = 0
    
    for date in dates:
        # Exits
        to_remove = []
        for t, pos in positions.items():
            if t in cache:
                df = cache[t]
                idx = df.index.searchsorted(date)
                if idx > 0 and idx < len(df):
                    price = df['Close'].iloc[idx]
                    ma50 = df['Close'].iloc[:idx+1].rolling(50).mean().iloc[-1]
                    
                    if price > pos['peak']: pos['peak'] = price
                    
                    if price < pos['peak'] * 0.85 or price < ma50:
                        ret = (price - pos['entry']) / pos['entry'] * 100
                        capital += pos['shares'] * price
                        trades.append({'Ticker': t, 'PnL%': ret, 'Entry': pos['entry_date'], 'Exit': date})
                        to_remove.append(t)
        for t in to_remove: del positions[t]
        
        # Entries
        if day_counter % 13 == 0 and len(positions) < 10:
            nifty_idx = nifty.index.searchsorted(date)
            if nifty_idx < 63: continue
            
            # --- THE NIFTY MA50 FILTER RULE ---
            current_nifty_dist = nifty['Dist_MA50'].iloc[nifty_idx]
            if pd.isna(current_nifty_dist): current_nifty_dist = 0
            
            # If the filter is ON, and Nifty is in the danger zone (-2% to 0%), SKIP ALL BUYS.
            skip_buys = False
            if apply_ma50_filter and (-2.0 <= current_nifty_dist <= 0.0):
                skip_buys = True
            
            if not skip_buys:
                candidates = []
                for t, df in cache.items():
                    if t in positions: continue
                    idx = df.index.searchsorted(date)
                    if idx < 100 or idx >= len(df): continue
                    
                    window = df.iloc[:idx+1]
                    price = window['Close'].iloc[-1]
                    ma50 = window['Close'].rolling(50).mean().iloc[-1]
                    if price < ma50: continue
                    
                    val = window['Volume'].iloc[-5:].mean() * price
                    if val < 10_000_000: continue
                    
                    def get_rs(days):
                        t_ret = (price - window['Close'].iloc[-days]) / window['Close'].iloc[-days]
                        n_ret = (nifty['Close'].iloc[nifty_idx] - nifty['Close'].iloc[nifty_idx-days]) / nifty['Close'].iloc[nifty_idx-days]
                        return (t_ret - n_ret) * 100
                    
                    comp_rs = (get_rs(5) * 0.10) + (get_rs(21) * 0.50) + (get_rs(63) * 0.40)
                    if comp_rs > 0:
                        candidates.append((t, comp_rs, price))
                
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
        equity_curve.append(eq)
        day_counter += 1
        
    return pd.DataFrame(trades), pd.DataFrame({'Equity': equity_curve})

def run_suite():
    nifty, cache = fetch_data(15)
    
    variants = [
        ('Baseline (OptComp-V21)', False),
        ('OptComp + Nifty MA50 Shield', True)
    ]
    
    all_results = []
    
    for label, years in HORIZONS.items():
        print(f"\n--- Testing Horizon: {label} ({years} Years) ---")
        start_date = nifty.index[-1] - timedelta(days=int(365.25 * years))
        if start_date < nifty.index[0]: start_date = nifty.index[0]
        
        for name, use_filter in variants:
            trades, eq_curve = run_backtest(nifty, cache, start_date, apply_ma50_filter=use_filter)
            
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
                'MaxDD%': max_dd,
                'WinRate%': win_rate,
                'Trades': len(trades)
            })

    res_df = pd.DataFrame(all_results)
    print("\n" + "="*80)
    print("OPTCOMP-V21: THE NIFTY MA50 SHIELD RULE IMPACT")
    print("="*80)
    for horizon in ['1y', '3y', '5y', '10y', '15y']:
        print(f"\nHorizon: {horizon}")
        sub = res_df[res_df['Horizon'] == horizon]
        if not sub.empty:
            print(sub[['Variant', 'CAGR%', 'MaxDD%', 'WinRate%', 'Trades']].to_string(index=False, float_format="%.1f"))

if __name__ == '__main__':
    run_suite()
