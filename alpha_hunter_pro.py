"""
Alpha Hunter PRO: Full Market Analysis
--------------------------------------
Analyzes 500 stocks across Indices (Nifty 50, Next 50, Mid, Small) and Sectors.
Buckets scores into Quartiles (4 zones) to find granular Alpha.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
import sys
import warnings
import concurrent.futures
import time

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.nifty500_list import TICKERS, SECTOR_MAP
from utils.score_history import calculate_historical_scores

CACHE_FILE = "nifty500_pro_cache.csv"

def fetch_market_cap(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            'ticker': ticker,
            'marketCap': info.get('marketCap', 0),
            'sector': info.get('sector', SECTOR_MAP.get(ticker, 'Unknown'))
        }
    except:
        return {'ticker': ticker, 'marketCap': 0, 'sector': SECTOR_MAP.get(ticker, 'Unknown')}

def get_universe_df():
    """
    Loads all tickers and assigns Index Groups based on Market Cap.
    """
    print("Loading Universe Data...")
    
    if os.path.exists(CACHE_FILE):
        print("Loading from cache...")
        df = pd.read_csv(CACHE_FILE)
    else:
        print("Fetching Market Caps (Threaded)...")
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            future_to_ticker = {executor.submit(fetch_market_cap, t): t for t in TICKERS}
            count = 0
            for future in concurrent.futures.as_completed(future_to_ticker):
                res = future.result()
                results.append(res)
                count += 1
                if count % 50 == 0:
                    print(f"Fetched {count}/{len(TICKERS)}...")
                    
        df = pd.DataFrame(results)
        df.to_csv(CACHE_FILE, index=False)
        
    # Sort and Rank
    df = df.sort_values('marketCap', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    
    def get_index(rank):
        if rank <= 50: return "Nifty 50"
        if rank <= 100: return "Nifty Next 50"
        if rank <= 250: return "Nifty Midcap 150"
        return "Nifty Smallcap 250"
        
    df['Index_Group'] = df['Rank'].apply(get_index)
    return df

def bucket_score(score, type='trend'):
    """Assigns score to one of 4 buckets."""
    if pd.isna(score): return np.nan
    
    if type == 'trend': # 0-100
        if score <= 25: return "0-25 (Low)"
        if score <= 50: return "25-50 (Mid-Low)"
        if score <= 75: return "50-75 (Mid-High)"
        return "75-100 (High)"
    elif type == 'mom': # 0-10
        if score <= 2.5: return "0-2.5 (Bear)"
        if score <= 5.0: return "2.5-5.0 (Neut)"
        if score <= 7.5: return "5.0-7.5 (Bull)"
        return "7.5-10 (Strong)"
    else: # Volume
        if score <= 2.5: return "0-2.5 (Low)"
        if score <= 5.0: return "2.5-5.0 (Mid)"
        if score <= 7.5: return "5.0-7.5 (High)"
        return "7.5-10 (Spike)"

def analyze_stock_history(ticker, df_hist, meta_info):
    if len(df_hist) < 200: return []
    
    scores = calculate_historical_scores(df_hist)
    # Drop rows where scores calc failed (warmup)
    scores = scores.dropna(subset=['trend_score_hist', 'momentum_score_hist'])
    
    # Forward Returns (15, 30, 45, 60, 90 Days)
    horizons = [15, 30, 45, 60, 90]
    for d in horizons:
        scores[f'ret_{d}d'] = scores['Close'].pct_change(d).shift(-d) * 100
        
    results = []
    subset = scores.iloc[::5] # Resample
    
    idx_grp = meta_info['Index_Group']
    sector = meta_info['sector']
    
    for date, row in subset.iterrows():
        if pd.isna(row['ret_90d']): continue # Ensure maximum horizon exists (or use partial?)
        # If we enforce 90d, we lose recent 3 months of data. 
        # Better: Check individual availability? 
        # User wants valid comparison so we should likely enforce it or handle NaNs in aggregation.
        # Let's enforce it to ensure "apple to apple" comparison for the row.
        
        entry = {
            'Index': idx_grp,
            'Sector': sector,
            'Trend_Bucket': bucket_score(row['trend_score_hist'], 'trend'),
            'Mom_Bucket': bucket_score(row['momentum_score_hist'], 'mom'),
            'Vol_Bucket': bucket_score(row['volume_score_hist'], 'vol'),
            'Ret_15d': row['ret_15d'],
            'Ret_30d': row['ret_30d'],
            'Ret_45d': row['ret_45d'],
            'Ret_60d': row.get('ret_60d', np.nan),
            'Ret_90d': row.get('ret_90d', np.nan)
        }
        results.append(entry)
        
    return results

def main():
    print("Starting Main Loop...")
    universe = get_universe_df()
    print(f"Universe Size: {len(universe)}")
    
    tickers = universe['ticker'].tolist()
    
    # Batch Download History (Chunked)
    chunk_size = 50
    all_results = []
    
    print("Downloading Score History...")
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        print(f"Processing chunk {i}-{i+len(chunk)}...")
        
        try:
            data = yf.download(chunk, period="2y", group_by='ticker', progress=False, auto_adjust=True, threads=True)
            
            for t in chunk:
                try:
                    meta = universe[universe['ticker'] == t].iloc[0]
                    
                    if len(chunk) > 1:
                        if t not in data.columns.get_level_values(0): continue
                        df = data[t].dropna()
                    else:
                        df = data.dropna()
                        
                    res = analyze_stock_history(t, df, meta)
                    all_results.extend(res)
                except: continue
                
        except Exception as e:
            print(f"Chunk failed: {e}")
            
    df_res = pd.DataFrame(all_results)
    if df_res.empty: return
    
    # REPORTING
    print("\n" + "="*80)
    print("ALPHA PRO REPORT (Sector Deep Dive: 15-90 Days)")
    print("="*80)
    
    # 1. Best Sector X Strategy (Deep Value)
    print("\n--- SECTOR PERFORMANCE: Deep Value Strategy (Trend 0-25) ---")
    deep_val = df_res[df_res['Trend_Bucket'] == '0-25 (Low)']
    
    # Aggregate Mean for all horizons
    sector_perf = deep_val.groupby('Sector')[['Ret_15d', 'Ret_30d', 'Ret_45d', 'Ret_60d', 'Ret_90d']].mean()
    # Sort by 45d to keep consistent ranking
    print(sector_perf.sort_values('Ret_45d', ascending=False).round(2))
    
    # 2. Best Sector X Momentum
    print("\n--- SECTOR PERFORMANCE: Momentum Strategy (Mom 7.5-10) ---")
    mom_play = df_res[df_res['Mom_Bucket'] == '7.5-10 (High)']
    sector_mom = mom_play.groupby('Sector')[['Ret_15d', 'Ret_30d', 'Ret_45d', 'Ret_60d', 'Ret_90d']].mean()
    print(sector_mom.sort_values('Ret_45d', ascending=False).round(2))
    
    df_res.to_csv("alpha_hunter_sector_results.csv", index=False)
    print("\nDetailed CSV saved to alpha_hunter_sector_results.csv")

if __name__ == "__main__":
    main()
