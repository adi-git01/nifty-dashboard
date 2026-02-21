"""
Alpha Hunter: Score Band Efficacy Analysis
------------------------------------------
Analyzes which "Score Zones" (0-10, 10-20...) historically deliver the best forward returns.
Helps identify "Sweet Spots" for entry and "Danger Zones" for exit.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import random
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.nifty500_list import TICKERS, SECTOR_MAP
from utils.score_history import calculate_historical_scores

def get_stratified_sample(n=30, exclude=[]):
    """Selects n stocks ensuring sector diversity, excluding specified list."""
    # Group by sector
    sector_groups = {}
    for t in TICKERS:
        if t in exclude: continue
        s = SECTOR_MAP.get(t, "Unknown")
        if s not in sector_groups:
            sector_groups[s] = []
        sector_groups[s].append(t)
    
    selected = []
    sectors = list(sector_groups.keys())
    
    # Shuffle
    random.shuffle(sectors)
    for k in sector_groups:
        random.shuffle(sector_groups[k])
    
    # Round robin
    while len(selected) < n and any(sector_groups.values()):
        for s in sectors:
            if len(selected) >= n: break
            if sector_groups[s]:
                stock = sector_groups[s].pop(0)
                selected.append(stock)
    
    return selected

def analyze_buckets(ticker, df):
    scores = calculate_historical_scores(df)
    
    # Drop rows where scores couldn't be calculated (initial warmup period)
    scores = scores.dropna(subset=['trend_score_hist', 'momentum_score_hist', 'volume_score_hist'])

    # Forward Returns (15, 20, 30, 45 Days)
    horizons = [15, 30, 45]
    for d in horizons:
        scores[f'ret_{d}d'] = scores['Close'].pct_change(d).shift(-d) * 100
    
    results = []
    
    # Sample every 5th day to reduce autocorrelation
    subset = scores.iloc[::5]
    
    for date, row in subset.iterrows():
        # Check if we have valid returns for at least the shortest period
        if pd.isna(row['ret_15d']): continue
        
        entry = {
            'Ticker': ticker,
            'Trend_Bin': int((row['trend_score_hist'] // 10) * 10), 
            'Momentum_Bin': int(round(row['momentum_score_hist'])), 
            'Volume_Bin': int(round(row['volume_score_hist'])), 
        }
        for d in horizons:
            entry[f'Ret_{d}d'] = row.get(f'ret_{d}d', np.nan)
            
        results.append(entry)
        
    return results

def main():
    print("Loading Sample B (New Random Set)...")
    
    # Sample A (Hardcoded from previous run to exclude)
    sample_a = [
        'ADANIENT.NS', 'ABB.NS', 'VBL.NS', 'POLYCAB.NS', 'ASHOKLEY.NS', 'UPL.NS', 'HINDZINC.NS',
        'TATATECH.NS', 'PAYTM.NS', 'ABCAPITAL.NS', 'CASTROLIND.NS', 'BDL.NS', 'CLEAN.NS',
        'AIAENG.NS', 'HINDCOPPER.NS', 'KAJARIACER.NS',
        'ACMESOLAR.NS', 'ABREL.NS', 'FACT.NS', 'JBMA.NS', 'CREDITACC.NS', 'GESHIP.NS',
        'TEJASNET.NS', 'HEG.NS', 'GRAVITA.NS', 'SAPPHIRE.NS', 'ECLERX.NS', 'EIDPARRY.NS',
        'ABLBL.NS', 'ETERNAL.NS'
    ]
    
    sample = get_stratified_sample(30, exclude=sample_a)
    print(f"Sample B Selected: {sample[:5]}...")
    
    print("Fetching History...")
    try:
        data = yf.download(sample, period="2y", group_by='ticker', progress=True, auto_adjust=True, threads=True)
    except:
        return

    all_data = []
    
    for ticker in sample:
        try:
            if len(sample) > 1:
                if ticker not in data.columns.get_level_values(0): continue
                df = data[ticker].dropna()
            else:
                df = data.dropna()
                
            if len(df) < 200: continue
            
            pts = analyze_buckets(ticker, df)
            all_data.extend(pts)
            
        except Exception as e:
            print(f"Error {ticker}: {e}")
            continue
        
    res = pd.DataFrame(all_data)
    if res.empty:
        print("No data.")
        return
        
    print("\n" + "="*60)
    print("SAMPLE B RESULTS: (15/30/45 Days)")
    print("="*60)
    
    ret_cols = [c for c in res.columns if 'Ret_' in c]
    
    # 1. Trend Score Analysis
    trend_stats = res.groupby('Trend_Bin')[ret_cols].mean()
    print("\nTrend Score Performance:")
    print(trend_stats.round(2))
    
    # 2. Momentum Score Analysis
    mom_stats = res.groupby('Momentum_Bin')[ret_cols].mean()
    print("\nMomentum Score Performance:")
    print(mom_stats.round(2))

    res.to_csv("alpha_hunter_sample_b.csv", index=False)

if __name__ == "__main__":
    main()
