"""
EARNINGS SHOCK & DRIFT INDUSTRY ANALYSIS
========================================
Historically scans all 500 Nifty stocks over 5 years for "Earnings Shock Footprints"
(>5% daily jump on >300% relative volume).

It then measures the price action around that node:
- T-20 to T-1 (The "Lead" / Front-running)
- T0 to T+1 (The "Concurrent" / The Shock Reaction)
- T+1 to T+60 (The "Lag" / Post-Earnings Announcement Drift - PEAD)

Aggregates the averages by Industry to tell us which industries front-run, 
which ones price it in perfectly on the day, and which ones slowly digest it.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import json
import warnings
import os
import sys

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nifty500_list import TICKERS

OUTPUT_DIR = "analysis_2026/earnings_shocks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_earnings_drifts():
    print("======================================================================")
    print("EARNINGS SHOCK & DRIFT INDUSTRY ANALYSIS (5-Year Historical Study)")
    print("======================================================================")
    
    print("[1/3] Loading Industry map...")
    info_cache = {}
    try:
        with open("analysis_2026/encyclopedia/yfinance_info_cache.json", "r") as f:
            info_cache = json.load(f)
    except:
        print("Error: Industry map not found.")
        return
        
    print(f"[2/3] Fetching 5 years of daily data for 500 stocks. This takes 15-20s...")
    data = yf.download(TICKERS[:500], period="5y", group_by='ticker', threads=True, progress=False, auto_adjust=True)
    
    print("[3/3] Scanning for Shock footprints and calculating PEAD (Drift)...")
    
    results = []
    
    for t in TICKERS[:500]:
        if t not in data.columns.get_level_values(0): continue
        df = data[t].dropna(how='all')
        if len(df) < 100: continue # Need history
            
        ind = info_cache.get(t, {}).get('industry', 'Unknown')
        if ind == 'Unknown': continue
            
        closes = df['Close'].values
        volumes = df['Volume'].values
        
        # Vectorized volume moving average (20 day)
        vol_df = pd.Series(volumes)
        vol_ma = vol_df.rolling(20).mean().shift(1).values
        
        # Calculate daily returns
        c_series = pd.Series(closes)
        pct_change = c_series.pct_change(fill_method=None).values * 100
        
        # Find Shock Indices:
        # 1. Daily return > 5%
        # 2. Volume > 3x the 20-day average volume
        
        # Avoid division by zero
        safe_vol_ma = np.where(vol_ma == 0, 1, vol_ma)
        vol_ratio = volumes / safe_vol_ma
        
        # Boolean mask for shocks
        shock_mask = (pct_change > 5.0) & (vol_ratio > 3.0)
        
        # Get indices where shock occurred. We need padding on both sides to calculate T-20 and T+60.
        shock_indices = np.where(shock_mask)[0]
        valid_shocks = [idx for idx in shock_indices if idx >= 20 and (len(closes) - idx) > 60]
        
        for idx in valid_shocks:
            p_T_minus_20 = closes[idx - 20]
            p_T_minus_1 = closes[idx - 1]
            p_T0 = closes[idx]
            p_T_plus_1 = closes[idx + 1]
            p_T_plus_20 = closes[idx + 20]
            p_T_plus_60 = closes[idx + 60]
            
            # Sub-period Returns
            lead_ret = (p_T_minus_1 - p_T_minus_20) / p_T_minus_20 * 100
            shock_ret = (p_T_plus_1 - p_T_minus_1) / p_T_minus_1 * 100
            drift_20_ret = (p_T_plus_20 - p_T_plus_1) / p_T_plus_1 * 100
            drift_60_ret = (p_T_plus_60 - p_T_plus_1) / p_T_plus_1 * 100
            
            results.append({
                'Ticker': t,
                'Industry': ind,
                'Shock_Date': df.index[idx].strftime('%Y-%m-%d'),
                'Volume_Spike': np.round(vol_ratio[idx], 1),
                'Lead_Ret_T20': lead_ret,
                'Shock_Ret_T0': shock_ret,
                'Drift_Ret_Tpos20': drift_20_ret,
                'Drift_Ret_Tpos60': drift_60_ret
            })
            
    df_res = pd.DataFrame(results)
    
    if df_res.empty:
        print("No shocks found in the historical data.")
        return
        
    print(f"\nTotal Individual Era Shocks Analyzed: {len(df_res)}")
    
    # ---------------------------------------------------------
    # AGGREGATE BY INDUSTRY
    # ---------------------------------------------------------
    agg = df_res.groupby('Industry').agg({
        'Ticker': 'count',
        'Lead_Ret_T20': 'mean',
        'Shock_Ret_T0': 'mean',
        'Drift_Ret_Tpos20': 'mean',
        'Drift_Ret_Tpos60': 'mean',
        'Volume_Spike': 'mean'
    }).rename(columns={'Ticker': 'Shock_Events'})
    
    # Filter out statistically insignificant industries (< 10 shocks over 5 years)
    agg = agg[agg['Shock_Events'] >= 10].copy()
    
    # Classification Logic
    def classify_behavior(row):
        lead = row['Lead_Ret_T20']
        drift = row['Drift_Ret_Tpos60']
        
        if lead > 8 and drift < 3:
            return "FRONT-RUNNERS (Priced in early. Fade the shock.)"
        elif drift > 10:
            return "DRIFTERS (Slow digesters. Buy the shock.)"
        elif 2 < lead < 8 and 2 < drift < 8:
            return "BALANCED (Typical momentum continuation.)"
        else:
            return "CONCURRENT (Priced perfectly on the day. No drift.)"
            
    agg['Classification'] = agg.apply(classify_behavior, axis=1)
    
    # Sort by 60D Drift (to find the best PEAD industries to trade)
    agg = agg.sort_values('Drift_Ret_Tpos60', ascending=False)
    
    print("\n___________________________________________________________________________________________________________________")
    print("  POST-EARNINGS SHOCK DRIFT (PEAD) BY INDUSTRY - 5 YEAR AVERAGES")
    print("___________________________________________________________________________________________________________________")
    
    headers = ["Industry", "Events", "Lead (T-20)%", "Shock (T0)%", "Drift 20D%", "Drift 60D%", "Behavior Matrix"]
    print(f"  {headers[0]:<28} | {headers[1]:<6} | {headers[2]:<12} | {headers[3]:<11} | {headers[4]:<10} | {headers[5]:<10} | {headers[6]}")
    print(f"  {'-'*110}")
    
    for ind, row in agg.iterrows():
        lead = f"{row['Lead_Ret_T20']:.1f}%"
        shock = f"{row['Shock_Ret_T0']:.1f}%"
        drift20 = f"{row['Drift_Ret_Tpos20']:.1f}%"
        drift60 = f"{row['Drift_Ret_Tpos60']:.1f}%"
        
        print(f"  {ind[:26]:<28} | {int(row['Shock_Events']):<6} | {lead:<12} | {shock:<11} | {drift20:<10} | {drift60:<10} | {row['Classification']}")
        
    print("\n[Insights]")
    print("- Front-Runners: The market knows the good news 20 days early. Do not buy the print.")
    print("- Drifters: Institutional investors slowly accumulate these over 60 days following a shock. (BUY THESE).")
    
    # Export
    agg.to_csv(f"{OUTPUT_DIR}/industry_drift_analysis.csv")
    df_res.to_csv(f"{OUTPUT_DIR}/raw_shock_events_log.csv", index=False)
    print(f"\nSaved raw log and aggregated summary to {OUTPUT_DIR}/")

if __name__ == "__main__":
    analyze_earnings_drifts()
