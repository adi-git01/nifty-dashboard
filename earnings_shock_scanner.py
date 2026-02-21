"""
EARNINGS SHOCK SCANNER
======================
Hunts for "Concurrent" industry stocks (e.g., Diagnostics, Defense, Hotels)
that just printed a massive EPS surprise, triggering a zero-lag breakout.
Since price does not lead earnings in these sectors, we must react instantly
to the fundamental catalyst.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nifty500_list import TICKERS, SECTOR_MAP

warnings.filterwarnings('ignore')

OUTPUT_DIR = "analysis_2026/earnings_shocks"

# Industries where Price = Earnings (Zero Lag). Found in our Encyclopedia.
CONCURRENT_INDUSTRIES = [
    'Diagnostics', 'Defense', 'Aerospace', 'Hotels', 'Lodging',
    'Broadcasting', 'Asset Management', 'Life Sciences'
]

def is_concurrent_industry(industry_str):
    if not industry_str: return False
    s = str(industry_str).lower()
    for kw in CONCURRENT_INDUSTRIES:
        if kw.lower() in s: return True
    return False

def scan_earnings_shocks():
    """
    Since actual live EPS surprise data usually requires a paid API (like Bloomberg/Refinitiv),
    this functional scanner looks for the "Price Action Footprint" of an earnings shock:
    A massive volume gap-up (or strong thrust) in a Concurrent industry stock that was
    previously consolidating.
    """
    print("=" * 80)
    print("EARNINGS SHOCK SCANNER (Zero-Lag Industry Hunting)")
    print("=" * 80)
    
    start = (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d')
    
    print("[1/2] Loading yfinance industry cache...")
    info_cache = {}
    try:
        with open("analysis_2026/encyclopedia/yfinance_info_cache.json", "r") as f:
            info_cache = json.load(f)
    except:
        print("Warning: info cache not found. Sector mapping will be degraded.")
        
    print(f"[2/2] Scanning fast price action footprints for {len(TICKERS[:500])} stocks...")
    
    try:
        data = yf.download(TICKERS[:500], start=start, group_by='ticker', threads=True, progress=False, auto_adjust=True)
    except Exception as e:
        print(f"Failed to fetch market data: {e}")
        return
        
    shocks = []
    
    for t in TICKERS[:500]:
        try:
            if t not in data.columns.get_level_values(0): continue
            df = data[t].dropna(how='all')
            if len(df) < 50: continue
            
            ind = info_cache.get(t, {}).get('industry', '')
            
            # RULE 1: Must be a Concurrent Industry
            if not is_concurrent_industry(ind): continue
            
            last_close = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2]
            
            # RULE 2: Must be a massive 1-day thrust or gap up (> 4%)
            daily_jump = (last_close - prev_close) / prev_close * 100
            if daily_jump < 4.0: continue
                
            # RULE 3: Volume must be an extreme shock (e.g. 300% of 20-day average)
            vol_20d = df['Volume'].iloc[-21:-1].mean()
            last_vol = df['Volume'].iloc[-1]
            if vol_20d == 0 or last_vol < (vol_20d * 3): continue
                
            # RULE 4: Price > MA50 (Trend confirmation)
            ma50 = df['Close'].rolling(50).mean().iloc[-1]
            if last_close < ma50: continue
                
            shocks.append({
                'Ticker': t,
                'Industry': ind,
                'Daily_Jump%': np.round(daily_jump, 2),
                'Vol_Spike_Ratio': np.round(last_vol / vol_20d, 1),
                'Close_Price': np.round(last_close, 2)
            })
            
        except Exception:
            pass
            
    # Display Results
    print("\n" + "_" * 75)
    print("  TIER 1 EARNINGS SHOCK BREAKOUTS DETECTED TODAY")
    print("  These 'Concurrent' industries require immediate action.")
    print("_" * 75)
    
    if not shocks:
        print("\n  [NO SHOCKS DETECTED TODAY] Markets are quiet in Concurrent Sectors.\n")
    else:
        df_shocks = pd.DataFrame(shocks).sort_values('Daily_Jump%', ascending=False)
        print(f"\n  {'Ticker':<15} | {'Industry':<25} | {'Jump%':<8} | {'Vol Spike':<12} | {'Close'}")
        print(f"  {'-'*72}")
        for _, r in df_shocks.iterrows():
            print(f"  {r['Ticker']:<15} | {r['Industry'][:23]:<25} | +{r['Daily_Jump%']:<7.1f} | {r['Vol_Spike_Ratio']:<10.1f}x | ₹{r['Close_Price']:<6.0f}")
            
    # Save log
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if shocks:
        pd.DataFrame(shocks).to_csv(f"{OUTPUT_DIR}/latest_shocks.csv", index=False)
        print(f"\n[Saved to {OUTPUT_DIR}/latest_shocks.csv]")

if __name__ == "__main__":
    scan_earnings_shocks()
