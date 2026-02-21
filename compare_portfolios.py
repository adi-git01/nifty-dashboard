"""
PORTFOLIO COMPARISON: EARLY MOMENTUM vs DNA3-V2.1
=================================================
Runs both strategy engines and compares their current picks.

1. DNA3-V2.1: Price > MA50, RS > 0 (Established Leaders)
2. Early Momentum: Breaking out from base, fresh RS > 0 (New Leaders)
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nifty500_list import TICKERS, SECTOR_MAP
from dna3_current_portfolio import DNA3LiveEngine

def get_early_momentum_picks():
    # Logic from early_momentum.py (Simulated here for speed)
    # 1. Price > MA20
    # 2. RS Crossing 0 recently ( Fresh Momentum)
    # 3. Volume Spikes
    pass 
    # To ensure accuracy, let's load the actual Early Momentum logic if available.
    # Checking file system...
    
    # Simple simulation of "Early Momentum" for comparison:
    # - Price > MA20 but < MA50 (Turnaround) OR
    # - Price > MA50 AND RS crossed 0 in last 20 days.
    
    picks = []
    # Using DNA3 Engine's cache to save time
    engine = DNA3LiveEngine()
    engine.fetch_data()
    
    print("\nScanning for Early Momentum (Fresh Breakouts)...")
    for t in engine.data_cache:
        if t == 'NIFTY': continue
        df = engine.data_cache[t]
        
        price = df['Close'].iloc[-1]
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        ma50 = df['Close'].rolling(50).mean().iloc[-1]
        
        # RS Logic
        if len(df) < 65: continue
        rs_series = (df['Close'].pct_change(63) - engine.data_cache['NIFTY']['Close'].pct_change(63)) * 100
        rs_now = rs_series.iloc[-1]
        rs_20d_ago = rs_series.iloc[-20]
        
        # Early Mob Rule:
        # 1. RS Just turned positive ( < 0 to > 0)
        # 2. OR Small Cap breaking out on Volume
        
        is_early = False
        if rs_20d_ago < 0 and rs_now > 0 and price > ma20: is_early = True
        
        if is_early:
            picks.append({
                'Ticker': t,
                'Price': price,
                'RS_Score': rs_now,
                'Setup': 'Fresh RS Crossover'
            })
            
    picks.sort(key=lambda x: -x['RS_Score'])
    return picks, engine

def main():
    print("Running DNA3 Scan...")
    engine = DNA3LiveEngine()
    dna3_picks = engine.run_scan() # This fetches data too
    dna3_tickers = set(p['Ticker'] for p in dna3_picks)
    
    print("\nRunning Early Momentum Scan...")
    # Re-use data from engine
    early_picks = []
    
    # Nifty data for RS calculation
    nifty = engine.data_cache['NIFTY']
    
    for t in engine.data_cache:
        if t == 'NIFTY': continue
        df = engine.data_cache[t]
        if len(df) < 100: continue
        
        price = df['Close'].iloc[-1]
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        ma50 = df['Close'].rolling(50).mean().iloc[-1]
        
        # RS Calculation
        try:
            p_63 = df['Close'].iloc[-63]
            n_63 = nifty['Close'].iloc[-63]
            rs = ((price - p_63)/p_63 - (nifty['Close'].iloc[-1] - n_63)/n_63) * 100
            
            p_20_ago = df['Close'].iloc[-20]
            n_20_ago = nifty['Close'].iloc[-20]
            p_83_ago = df['Close'].iloc[-83]
            n_83_ago = nifty['Close'].iloc[-83]
            
            rs_prev = ((p_20_ago - p_83_ago)/p_83_ago - (n_20_ago - n_83_ago)/n_83_ago) * 100
            
            # EARLY MOMENTUM CRITERIA:
            # 1. RS was negative 1 month ago, now Positive (Fresh Leader)
            # 2. Price > MA20 (Short term trend up)
            if rs_prev < 0 and rs > 0 and price > ma20:
                early_picks.append({
                    'Ticker': t,
                    'Price': price,
                    'RS_Score': rs
                })
        except: pass
        
    early_picks.sort(key=lambda x: -x['RS_Score'])
    early_top = early_picks[:15]
    early_tickers = set(p['Ticker'] for p in early_top)
    
    # COMPARISON
    common = dna3_tickers.intersection(early_tickers)
    only_dna3 = dna3_tickers - early_tickers
    only_early = early_tickers - dna3_tickers
    
    print("\n" + "="*60)
    print("PORTFOLIO COMPARISON")
    print("="*60)
    
    print(f"\nCOMMON PICKS (Strong & Getting Stronger): {len(common)}")
    for t in common: print(f"  - {t}")
    
    print(f"\nDNA3-V2.1 ONLY (Established Leaders > MA50): {len(only_dna3)}")
    for t in list(only_dna3)[:15]: print(f"  - {t}")
    
    print(f"\nEARLY MOMENTUM ONLY (Fresh Breakouts): {len(only_early)}")
    for t in list(only_early)[:15]: print(f"  - {t}")

    # Create detailed DataFrame for csv
    res = []
    for t in dna3_tickers: res.append({'Ticker': t, 'Strategy': 'DNA3-V2.1'})
    for t in early_tickers: res.append({'Ticker': t, 'Strategy': 'Early Momentum'})
    
    pd.DataFrame(res).to_csv('portfolio_comparison.csv', index=False)
    print("\nSaved to portfolio_comparison.csv")

if __name__ == "__main__":
    main()
