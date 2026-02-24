"""
ANALYZE PGEL DIVERGENCE
=======================
Why does PGEL have Low Trend Score but High RS/DNA Buy?
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

STOCK = "PGEL.NS"
NIFTY = "^NSEI"

def analyze():
    print(f"Fetching data for {STOCK}...")
    df = yf.Ticker(STOCK).history(period="1y")
    nifty = yf.Ticker(NIFTY).history(period="1y")
    
    if df.empty or nifty.empty:
        print("Data fetch failed.")
        return

    # 1. METRICS
    price = df['Close'].iloc[-1]
    ma50 = df['Close'].rolling(50).mean().iloc[-1]
    ma200 = df['Close'].rolling(200).mean().iloc[-1]
    high_52 = df['High'].max()
    low_52 = df['Low'].min()
    
    # 2. RS SCORE (3 Month)
    # Align dates
    df.index = df.index.tz_localize(None)
    nifty.index = nifty.index.tz_localize(None)
    
    p_63 = df['Close'].iloc[-63]
    n_63 = nifty['Close'].iloc[-63]
    rs_stock = (price - p_63)/p_63
    rs_nifty = (nifty['Close'].iloc[-1] - n_63)/n_63
    rs_score = (rs_stock - rs_nifty) * 100
    
    # 3. TREND SCORE CALCULATION (Replicating logic)
    score = 50
    print(f"\n--- TREND SCORE BREAKDOWN ---")
    print(f"Base Score: 50")
    
    # MA50
    if price > ma50:
        score += 15
        print(f"Price > MA50: +15 ({price:.2f} > {ma50:.2f})")
    else:
        score -= 10
        print(f"Price < MA50: -10 ({price:.2f} < {ma50:.2f})")
        
    # MA200
    if price > ma200:
        score += 15
        print(f"Price > MA200: +15 ({price:.2f} > {ma200:.2f})")
    else:
        score -= 15
        print(f"Price < MA200: -15 ({price:.2f} < {ma200:.2f})")
        
    # Cross
    if ma50 > ma200:
        score += 10
        print(f"MA50 > MA200 (Golden Cross): +10")
    else:
        score -= 5
        print(f"MA50 < MA200 (Death Cross): -5")
        
    # 52W Position
    range_52 = high_52 - low_52
    pos = (price - low_52) / range_52
    range_score = int((pos - 0.5) * 30)
    score += range_score
    print(f"52-Week Range Position ({pos:.2f}): {range_score:+d}")
    
    # Dist from High
    dist = (price - high_52)/high_52 * 100
    if dist > -5:
        score += 10
        print(f"Near 52W High (>-5%): +10")
    elif dist < -30:
        score -= 10
        print(f"Deep Drawdown (<-30%): -10")
    else:
        print(f"Drawdown ({dist:.1f}%): No Bonus/Penalty")
        
    score = max(0, min(100, score))
    print(f"FINAL TREND SCORE: {score}")
    
    print(f"\n--- DNA3 STATUS ---")
    print(f"Price > MA50: {price > ma50}")
    print(f"RS Score > 0: {rs_score > 0} ({rs_score:.1f})")
    
    if price > ma50 and rs_score > 0:
        print("VERDICT: DNA3 BUY SIGNAL")
    else:
        print("VERDICT: NO SIGNAL ❌")

if __name__ == "__main__":
    analyze()
