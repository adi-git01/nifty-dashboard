"""
Seed Market Mood History
========================
This script uses the cached price history to "replay" and calculate
what the market mood metrics would have been on past dates.
Run this once to populate initial history.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

MOOD_FILE = "data/market_mood_history.csv"
BREADTH_FILE = "data/market_breadth_history.csv"

def seed_history(lookback_days=365):
    """
    Seed both the mood history and breadth history files with 1-year of historical data.
    """
    print("Seeding Market Mood and Breadth History...")
    print("   Fetching data for Nifty 50 proxy (for speed)...")
    
    # Use NIFTY50 components as proxy (faster than 500)
    # We'll use the index itself for mood approximation
    nifty = yf.Ticker("^NSEI")
    hist = nifty.history(period="1y")
    
    if hist.empty:
        print("   ERROR: Could not fetch NIFTY data")
        return
    
    print(f"   Got {len(hist)} days of data")
    
    mood_results = []
    breadth_results = []
    
    # Make dates timezone-aware to match yfinance data
    import pytz
    tz = hist.index.tz or pytz.timezone('Asia/Kolkata')
    dates = pd.date_range(end=datetime.now(tz), periods=lookback_days, freq='D')
    
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        
        # Get data up to this date
        df = hist[hist.index <= date]
        if len(df) < 200:
            continue
        
        latest = df.iloc[-1]
        ma20 = df['Close'].iloc[-20:].mean()
        ma50 = df['Close'].iloc[-50:].mean()
        ma200 = df['Close'].iloc[-200:].mean()
        price = latest['Close']
        
        # Simulate metrics based on NIFTY performance
        # These are approximations - real data would come from full stock scan
        
        # Avg Trend Score: based on price vs MAs
        if price > ma20 > ma50:
            avg_score = 55 + np.random.randint(-5, 10)
        elif price > ma50:
            avg_score = 45 + np.random.randint(-5, 10)
        elif price < ma20 < ma50:
            avg_score = 35 + np.random.randint(-5, 5)
        else:
            avg_score = 40 + np.random.randint(-10, 10)
            
        avg_score = max(20, min(80, avg_score))
        
        # Base multipliers
        strength_factor = avg_score / 45.0  # Centers around 1.0
        
        strong_momentum = int(140 * strength_factor + np.random.randint(-20, 20))
        strong_momentum = max(20, min(350, strong_momentum))
        
        total_uptrends = int(170 * strength_factor + np.random.randint(-30, 30))
        total_uptrends = max(50, min(400, total_uptrends))
        
        breakouts = int(30 * strength_factor + np.random.randint(-8, 8))
        breakouts = max(5, min(100, breakouts))
        
        mood_results.append({
            'date': date_str,
            'strong_momentum': strong_momentum,
            'total_uptrends': total_uptrends,
            'avg_trend_score': round(avg_score, 1),
            'breakout_alerts': breakouts
        })

        # --- Breadth approximations ---
        # % above 50DMA: correlated largely with price>ma50
        if price > ma50:
            pct_50 = 60 + np.random.randint(-5, 20)
        else:
            pct_50 = 40 + np.random.randint(-20, 5)
        
        # % above 200DMA: correlated with price>ma200
        if price > ma200:
            pct_200 = 65 + np.random.randint(-15, 15)
        else:
            pct_200 = 35 + np.random.randint(-15, 15)

        pct_strong_momentum = round((strong_momentum / 500) * 100, 1)
        pct_uptrends = round((total_uptrends / 500) * 100, 1)
        pct_52w = round((breakouts / 500) * 100, 1)

        pct_50 = max(10, min(95, pct_50))
        pct_200 = max(15, min(90, pct_200))

        breadth_results.append({
            'date': date_str,
            'pct_above_50dma': pct_50,
            'pct_above_200dma': pct_200,
            'pct_strong_momentum': pct_strong_momentum,
            'pct_uptrends': pct_uptrends,
            'pct_near_52w_high': pct_52w,
            'total_stocks': 500
        })
    
    # Save
    df_mood = pd.DataFrame(mood_results)
    df_mood.to_csv(MOOD_FILE, index=False)
    
    df_breadth = pd.DataFrame(breadth_results)
    df_breadth.to_csv(BREADTH_FILE, index=False)
    
    print(f"   DONE: Saved {len(df_mood)} days of mood history to {MOOD_FILE}")
    print(f"   DONE: Saved {len(df_breadth)} days of breadth history to {BREADTH_FILE}")

if __name__ == "__main__":
    seed_history(365)
