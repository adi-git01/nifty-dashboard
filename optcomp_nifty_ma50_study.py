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

def run_optcomp_ma50_study(years=5):
    print(f"Fetching Nifty Data for last {years} years...")
    start = (datetime.now() - timedelta(days=365 * years + 100)).strftime('%Y-%m-%d')
    nifty = yf.Ticker('^NSEI').history(start=start)
    nifty.index = nifty.index.tz_localize(None)
    
    # Calculate Nifty MA50 and Distance
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

    # OptComp-V21 Backtest Engine
    capital = 1_000_000
    positions = {}
    trades = []
    
    dates = nifty.index[nifty.index > (datetime.now() - timedelta(days=365 * years))]
    day_counter = 0
    
    print("Running OptComp-V21 and logging Nifty MA50 proximity...")
    for date in dates:
        # Exits: 15% Trailing or MA50
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
                        trades.append({
                            'Ticker': t, 
                            'PnL%': ret, 
                            'Entry_Date': pos['entry_date'],
                            'Exit_Date': date,
                            'Nifty_Dist_MA50': pos['nifty_dist_ma50'] # Logged at entry
                        })
                        to_remove.append(t)
        for t in to_remove: del positions[t]
        
        # Entries: Every 13 days
        if day_counter % 13 == 0 and len(positions) < 10:
            nifty_idx = nifty.index.searchsorted(date)
            if nifty_idx < 63: continue
            
            current_nifty_dist = nifty['Dist_MA50'].iloc[nifty_idx]
            if pd.isna(current_nifty_dist): current_nifty_dist = 0
            
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
                
                # OptComp RS (10% 1W, 50% 1M, 40% 3M)
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
                    positions[t] = {
                        'entry': price, 
                        'peak': price, 
                        'shares': shares, 
                        'entry_date': date,
                        'nifty_dist_ma50': current_nifty_dist
                    }
        day_counter += 1

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        print("No trades generated.")
        return
        
    print("\n" + "="*80)
    print("OPTCOMP-V21: NIFTY 50-DAY MA PROXIMITY VS TRADE PERFORMANCE (5 YEARS)")
    print("="*80)
    
    bins = [-100, -5, -2, 0, 2, 5, 100]
    labels = ['Crash (<-5%)', 'Oversold (-5% to -2%)', 'Support (-2% to 0%)', 'Early Trend (0% to +2%)', 'Strong (+2% to +5%)', 'Extended (>+5%)']
    trades_df['Nifty_MA50_Bucket'] = pd.cut(trades_df['Nifty_Dist_MA50'], bins=bins, labels=labels)
    
    summary = trades_df.groupby('Nifty_MA50_Bucket', observed=False).agg(
        Trades=('PnL%', 'count'),
        Win_Rate=('PnL%', lambda x: (x > 0).mean() * 100),
        Avg_PnL=('PnL%', 'mean'),
        Avg_Win=('PnL%', lambda x: x[x>0].mean()),
        Avg_Loss=('PnL%', lambda x: x[x<=0].mean())
    ).dropna()
    
    print(summary.round(1).to_string())

if __name__ == "__main__":
    run_optcomp_ma50_study(years=5)
