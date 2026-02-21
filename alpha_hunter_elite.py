"""
Alpha Hunter ELITE: Exhaustive Strategy Search
----------------------------------------------
"Juicing every ounce of Alpha"
Features:
1. Quartile Bucketing for Trend (0-100) and Decile/Quartile for Mom/Vol (0-10).
2. SIGNAL DELTA: Analyzing 'Jumps' in scores (e.g. Volume 4->7).
3. CONFLUENCE: Trend Bucket + Volume Change interaction.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
import sys
import warnings
import concurrent.futures

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.nifty500_list import TICKERS, SECTOR_MAP
from utils.score_history import calculate_historical_scores

CACHE_FILE = "nifty500_pro_cache.csv"

def get_universe_df():
    # Reuse cache logic
    if os.path.exists(CACHE_FILE):
        df = pd.read_csv(CACHE_FILE)
    else:
        # Just stick to what we have or empty
        print("Cache not found. Please run alpha_hunter_pro.py first to seed cache.")
        return pd.DataFrame({'ticker': TICKERS, 'marketCap': 0, 'sector': 'Unknown'})
    
    # Sort/Rank logic
    if 'marketCap' in df.columns:
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
    if pd.isna(score): return np.nan
    
    if type == 'trend': # 0-100 (Quartiles)
        if score <= 25: return "Q1 (0-25) Oversold"
        if score <= 50: return "Q2 (25-50) Weak"
        if score <= 75: return "Q3 (50-75) Building"
        return "Q4 (75-100) Strong"
        
    elif type == 'mom': # 0-10 (Quartiles approx)
        if score <= 2.5: return "Q1 (0-2.5) Bear"
        if score <= 5.0: return "Q2 (2.5-5) Neut-"
        if score <= 7.5: return "Q3 (5-7.5) Neut+"
        return "Q4 (7.5-10) Bull"
        
    elif type == 'vol': # 0-10
        if score <= 2.5: return "Q1 (0-2.5) Quiet"
        if score <= 5.0: return "Q2 (2.5-5) Normal"
        if score <= 7.5: return "Q3 (5-7.5) Active"
        return "Q4 (7.5-10) High"

def bucket_delta(delta):
    """Classify the CHANGE in score."""
    if pd.isna(delta): return "N/A"
    if delta >= 3: return "Big Jump (+3)"
    if delta >= 1: return "Jump (+1 to +3)"
    if delta <= -3: return "Big Drop (-3)"
    if delta <= -1: return "Drop (-1 to -3)"
    return "Flat (-1 to +1)"

def analyze_stock_elite(ticker, df_hist):
    if len(df_hist) < 200: return []
    
    scores = calculate_historical_scores(df_hist)
    scores = scores.dropna(subset=['trend_score_hist', 'momentum_score_hist', 'volume_score_hist'])
    
    # === CRITICAL: CALCULATE DELTAS ===
    # Using 5-day delta (1 week change) to capture "Jumps"
    scores['vol_delta'] = scores['volume_score_hist'].diff(5)
    
    # Forward Returns
    horizons = [15, 30, 45, 60, 90]
    for d in horizons:
        scores[f'ret_{d}d'] = scores['Close'].pct_change(d).shift(-d) * 100
        
    results = []
    subset = scores.iloc[::5]
    
    for date, row in subset.iterrows():
        if pd.isna(row['ret_45d']): continue
        
        entry = {
            'Ticker': ticker,
            # Base Buckets
            'Trend_Bucket': bucket_score(row['trend_score_hist'], 'trend'),
            'Vol_Bucket': bucket_score(row['volume_score_hist'], 'vol'),
            'Mom_Bucket': bucket_score(row['momentum_score_hist'], 'mom'),
            # Delta Buckets
            'Vol_Change': bucket_delta(row['vol_delta']),
            # Returns
            'Ret_15d': row['ret_15d'],
            'Ret_30d': row['ret_30d'],
            'Ret_45d': row['ret_45d'],
            'Ret_60d': row.get('ret_60d', np.nan),
            'Ret_90d': row.get('ret_90d', np.nan)
        }
        results.append(entry)
        
    return results

def main():
    print("Starting ELITE Alpha Hunt...")
    universe = get_universe_df()
    tickers = universe['ticker'].tolist()
    
    chunk_size = 50
    all_results = []
    
    print(f"Scanning {len(tickers)} stocks for Exhaustive Alpha...")
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        print(f"  Chunk {i}-{i+len(chunk)}...")
        try:
            data = yf.download(chunk, period="2y", group_by='ticker', progress=False, auto_adjust=True, threads=True)
            for t in chunk:
                try:
                    if len(chunk) > 1:
                        if t not in data.columns.get_level_values(0): continue
                        df = data[t].dropna()
                    else:
                        df = data.dropna()
                    
                    res = analyze_stock_elite(t, df)
                    all_results.extend(res)
                except: continue
        except: pass
            
    df = pd.DataFrame(all_results)
    if df.empty:
        print("No data.")
        return
        
    print("\n" + "="*80)
    print("ELITE ALPHA REPORT: 'Juicing the Alpha'")
    print("="*80)
    
    # 1. Volume Change Analysis (Does a JUMP help?)
    print("\n--- DOES VOLUME JUMPING GENERATE ALPHA? ---")
    # Group by Trend Bucket + Vol Change
    # We want to know: If Trend is Low (oversold), and Volume JUMPS, is it a buy?
    vol_matrix = df.groupby(['Trend_Bucket', 'Vol_Change'])[['Ret_30d', 'Ret_60d']].mean()
    print(vol_matrix.round(2))
    
    # 2. The "Perfect Setup" Search
    # Flatten the grouping to find the single best combination
    print("\n--- THE HOLY GRAIL COMBINATIONS (Sorted by 60d Return) ---")
    combo = df.groupby(['Trend_Bucket', 'Vol_Change'])[['Ret_60d']].mean()
    print(combo.sort_values('Ret_60d', ascending=False).head(10).round(2))
    
    # 3. Momentum Buckets
    print("\n--- MOMENTUM BUCKETS (0-10 split 4 ways) ---")
    mom_perf = df.groupby('Mom_Bucket')[['Ret_30d', 'Ret_60d']].mean()
    print(mom_perf.round(2))
    
    df.to_csv("alpha_hunter_elite_results.csv", index=False)
    print("\nSaved to alpha_hunter_elite_results.csv")

if __name__ == "__main__":
    main()
