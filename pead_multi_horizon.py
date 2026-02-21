"""
MULTI-HORIZON PEAD CORRELATION STUDY
====================================
Tests whether Post-Earnings Announcement Drift (PEAD) has experienced
structural alpha decay or regime change over time.

Scans all 500 Nifty stocks for "Earnings Shock Footprints"
(>5% daily jump on >300% relative volume).

Groups the analysis by these historical horizons:
['6mo', '1y', '3y', '5y', '10y', '15y']

Calculates:
- T-20 (Lead)
- T0 (Shock)
- T+20 (Drift)
"""

import pandas as pd
import numpy as np
import yfinance as yf
import json
import warnings
import os
import sys
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nifty500_list import TICKERS

OUTPUT_DIR = "analysis_2026/earnings_shocks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HORIZONS = {
    '6mo': 180,
    '1y': 365,
    '3y': 365 * 3,
    '5y': 365 * 5,
    '10y': 365 * 10,
    '15y': 365 * 15
}

def analyze_multi_horizon_pead():
    print("======================================================================")
    print("MULTI-HORIZON PEAD STRUCTURAL SHIFT ANALYSIS")
    print("======================================================================")
    
    print("[1/3] Loading Industry map...")
    info_cache = {}
    try:
        with open("analysis_2026/encyclopedia/yfinance_info_cache.json", "r") as f:
            info_cache = json.load(f)
    except:
        print("Error: Industry map not found.")
        return
        
    print(f"[2/3] Fetching 15 years of daily data for 500 stocks. Please wait...")
    data = yf.download(TICKERS[:500], period="15y", group_by='ticker', threads=True, progress=False, auto_adjust=True)
    
    print("[3/3] Scanning for Shocks across multiple horizons...")
    
    all_shocks = []
    
    # Calculate cutoff dates for each horizon relative to today
    today = datetime.now()
    cutoffs = {h: today - timedelta(days=d) for h, d in HORIZONS.items()}
    
    for t in TICKERS[:500]:
        if t not in data.columns.get_level_values(0): continue
        df = data[t].dropna(how='all')
        if len(df) < 100: continue # Need history
            
        ind = info_cache.get(t, {}).get('industry', 'Unknown')
        if ind == 'Unknown': continue
            
        closes = df['Close'].values
        volumes = df['Volume'].values
        dates = df.index
        
        # Vectorized volume moving average (20 day)
        vol_df = pd.Series(volumes)
        vol_ma = vol_df.rolling(20).mean().shift(1).values
        
        # Calculate daily returns
        c_series = pd.Series(closes)
        pct_change = c_series.pct_change(fill_method=None).values * 100
        
        # Avoid division by zero
        safe_vol_ma = np.where(vol_ma == 0, 1, vol_ma)
        vol_ratio = volumes / safe_vol_ma
        
        # Boolean mask for shocks ( >5% jump, >3x volume )
        shock_mask = (pct_change > 5.0) & (vol_ratio > 3.0)
        
        shock_indices = np.where(shock_mask)[0]
        # Valid means we have 20 days before and 20 days after
        valid_shocks = [idx for idx in shock_indices if idx >= 20 and (len(closes) - idx) > 20]
        
        for idx in valid_shocks:
            shock_date = dates[idx].to_pydatetime()
            
            p_T_minus_20 = closes[idx - 20]
            p_T_minus_1 = closes[idx - 1]
            p_T0 = closes[idx]
            p_T_plus_1 = closes[idx + 1]
            p_T_plus_20 = closes[idx + 20]
            
            lead_ret = (p_T_minus_1 - p_T_minus_20) / p_T_minus_20 * 100
            shock_ret = (p_T_plus_1 - p_T_minus_1) / p_T_minus_1 * 100
            drift_20_ret = (p_T_plus_20 - p_T_plus_1) / p_T_plus_1 * 100
            
            # Determine which horizons this shock falls into
            for h, cutoff in cutoffs.items():
                # Note: timezone unaware comparison
                # if shock_date >= cutoff:
                # Need to strip tzinfo for comparison if dates have it
                if shock_date.replace(tzinfo=None) >= cutoff.replace(tzinfo=None):
                    all_shocks.append({
                        'Horizon': h,
                        'Ticker': t,
                        'Industry': ind,
                        'Date': shock_date.strftime('%Y-%m-%d'),
                        'Lead_20': lead_ret,
                        'Shock': shock_ret,
                        'Drift_20': drift_20_ret
                    })
            
    df_res = pd.DataFrame(all_shocks)
    
    if df_res.empty:
        print("No shocks found.")
        return
        
    # Process each horizon
    print("\n____________________________________________________________________________________________________")
    print("  PEAD MULTI-HORIZON INDUSTRY AVERAGES (Validating structural alpha decay)")
    print("____________________________________________________________________________________________________\n")
    
    # We will pick a few dominant industries to display the structural shift cleanly
    key_industries = [
        'Capital Markets', 'Specialty Chemicals', 'Aerospace & Defense', 
        'Information Technology Ser', 'Auto Parts', 'Oil & Gas Integrated'
    ]
    
    horizon_order = ['15y', '10y', '5y', '3y', '1y', '6mo']
    
    for ind in key_industries:
        print(f"[{ind.upper()}] Post-Earnings Behavior Over Time:")
        print(f"  {'Horizon':<8} | {'Events':<6} | {'Lead (T-20)%-Front-run':<22} | {'Shock (T0)':<12} | {'Drift (T+20)%-PEAD'}")
        print(f"  {'-'*85}")
        
        ind_df = df_res[df_res['Industry'].str.contains(ind, case=False, na=False)]
        
        for h in horizon_order:
            h_df = ind_df[ind_df['Horizon'] == h]
            if h_df.empty: continue
            
            count = len(h_df)
            lead = h_df['Lead_20'].mean()
            shock = h_df['Shock'].mean()
            drift = h_df['Drift_20'].mean()
            
            print(f"  {h:<8} | {count:<6} | {lead:>8.1f}%{' ' * 13} | {shock:>8.1f}%   | {drift:>8.1f}%")
            
        print("\n")
        
    # Export full dataset
    agg_full = df_res.groupby(['Horizon', 'Industry']).agg({
        'Ticker': 'count',
        'Lead_20': 'mean',
        'Shock': 'mean',
        'Drift_20': 'mean'
    }).rename(columns={'Ticker': 'Events'}).reset_index()
    
    out_path = f"{OUTPUT_DIR}/multi_horizon_pead.csv"
    agg_full.to_csv(out_path, index=False)
    print(f"Full multi-horizon statistical cross-tab saved to: {out_path}")


if __name__ == "__main__":
    analyze_multi_horizon_pead()
