"""
DEEP INDUSTRY CYCLE & LAG ANALYSIS
==================================
Uses Screener granular industries to answer:
1. Cycle Length & Amplitude: How long does a typical full cycle last per industry?
2. Earnings vs Price Lag: Does price lead earnings? By how many quarters?
3. Regime Performance: Bull/Bear alpha for these granular industries.
4. Deep Seasonality: The 10Y robust seasonal pattern check for granular industries.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os
import sys
import time

warnings.filterwarnings('ignore')

OUTPUT_DIR = "analysis_2026/industry_cycles"
SCREENER_FILE = r"c:\Users\adity\.gemini\antigravity\playground\perihelion-lunar\screener_sectors_constituents.csv"

# Minimum stocks required to constitute a valid industry index
MIN_STOCKS = 5

def map_ticker(name):
    """Maps screener company name to Yahoo Finance ticker (best effort rule-based)."""
    # This is a simplification. We will use a fuzzy match against Nifty 500 later,
    # or just use the exact names from a mapping we generate.
    # For now, we will construct a mapping from the nse symbol list if possible.
    return None # We will do this differently


def run():
    print("=" * 120)
    print("DEEP INDUSTRY CYCLE EXPLORATION")
    print("=" * 120)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Screener Data
    df_scr = pd.read_csv(SCREENER_FILE)
    print(f"Loaded {len(df_scr)} companies from Screener file")
    
    # Analyze Sector composition
    sec_counts = df_scr['Sector'].value_counts()
    valid_sectors = sec_counts[sec_counts >= MIN_STOCKS].index.tolist()
    print(f"\nFound {len(valid_sectors)} valid granular industries (>= {MIN_STOCKS} stocks):")
    for s in valid_sectors[:10]:
        print(f"  - {s} ({sec_counts[s]} stocks)")
    if len(valid_sectors) > 10:
        print(f"  ... and {len(valid_sectors)-10} more.")
    
    # We need a robust Name -> Ticker mapping to get historical data
    # We will use our Nifty 500 list from previous scripts as a bridge
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from utils.nifty500_list import TICKERS
    except Exception as e:
        print(f"Error loading Nifty 500 list: {e}")
        return
        
    print(f"\nMapping {len(TICKERS)} Nifty 500 tickers to granular industries...")
    
    # Let's download a fresh symbol mapping from NSE if needed, or just map by matching names
    # Simplified approach for script: We will use yfinance to fetch info for a sample to map
    # But since we have 500 tickers, we can just map based on the first word or two.
    
    # Create name tokens for Nifty 500
    ticker_tokens = {}
    for t in TICKERS:
        name_part = t.replace('.NS', '').replace('-', ' ').lower()
        ticker_tokens[t] = set(name_part.split())
        
    # Map screener names to tickers
    mapped_tickers = {}
    industry_map = {}
    
    matched = 0
    for _, row in df_scr.iterrows():
        scr_name = str(row['Name']).lower().replace('.', ' ').replace('-', ' ').replace('ltd', '')
        scr_tokens = set(scr_name.split())
        
        # Super simple Jaccard match
        best_t = None
        best_score = 0
        for t, t_tokens in ticker_tokens.items():
            intersection = len(scr_tokens.intersection(t_tokens))
            union = len(scr_tokens.union(t_tokens))
            score = intersection / union if union > 0 else 0
            if score > best_score and score > 0.3:
                best_score = score
                best_t = t
                
        if best_t:
            mapped_tickers[best_t] = row['Sector']
            if row['Sector'] not in industry_map:
                industry_map[row['Sector']] = []
            industry_map[row['Sector']].append(best_t)
            matched += 1
            
    print(f"Successfully mapped {matched} / 500 Nifty 500 tickers to granular industries.")
    
    # Keep only industries with enough mapped stocks
    final_industries = {k: v for k, v in industry_map.items() if len(v) >= 3}
    print(f"\nProceeding with {len(final_industries)} industries for deep analysis.")
    for ind, t_list in final_industries.items():
        print(f"  {ind[:30]:<30}: {len(t_list)} stocks (e.g. {t_list[0]}, {t_list[min(1, len(t_list)-1)]})")


if __name__ == "__main__":
    run()

