import yfinance as yf
import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path to import utils
sys.path.append(os.getcwd())

try:
    from utils.volume_analysis import get_combined_volume_signal
except ImportError:
    # Fallback if running from a different directory structure
    pass

def normalize(value, min_val, max_val):
    if value is None: return 5
    if value < min_val: return 0
    if value > max_val: return 10
    return ((value - min_val) / (max_val - min_val)) * 10

def calculate_method_1(info, hist):
    """Simple Volume Ratio (Current Implementation)"""
    avg_vol = info.get('averageVolume', 0)
    avg_vol_10d = info.get('averageVolume10days', 0)
    current_vol = info.get('volume', 0)
    
    # Fallback
    if avg_vol_10d == 0 and current_vol > 0:
        avg_vol_10d = current_vol
        
    if avg_vol > 0 and avg_vol_10d > 0:
        vol_ratio = avg_vol_10d / avg_vol
        
        # New Scoring Logic
        if vol_ratio <= 0.7:
            score = 0
        elif vol_ratio >= 1.5:
            score = 10
        elif vol_ratio < 1.0:
            score = ((vol_ratio - 0.7) / 0.3) * 5
        else:
            score = 5 + ((vol_ratio - 1.0) / 0.5) * 5
            
        # Price action bonus
        if not hist.empty:
            r1m = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
        else:
            r1m = 0
            
        if vol_ratio > 1.15 and r1m > 0:
            score = min(10, score + 1.0)
        elif vol_ratio > 1.3 and r1m < -5:
            score = max(0, score - 1.5)
            
        return round(score, 1), vol_ratio
    return 5.0, 0

def calculate_method_2(hist):
    """VPT + A/D (Advanced Method)"""
    if hist.empty:
        return 5.0, "NEUTRAL"
        
    # Uses the util function
    # get_combined_volume_signal(high, low, close, volume, lookback=20)
    result = get_combined_volume_signal(
        hist['High'], 
        hist['Low'], 
        hist['Close'], 
        hist['Volume'], 
        lookback=20
    )
    return result['combined_score'], result['combined_signal']

# 15 Diverse Stocks
tickers = [
    'HINDCOPPER.NS', 'TRENT.NS', 'ZOMATO.NS', 'IRFC.NS', 'RELIANCE.NS', 
    'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'TATAMOTORS.NS', 'ADANIENT.NS', 
    'SUNPHARMA.NS', 'ITC.NS', 'JIOFIN.NS', 'BEL.NS', 'COALINDIA.NS'
]

print(f"{'STOCK':<15} | {'M1 SCORE':<10} | {'M1 RATIO':<10} || {'M2 SCORE':<10} | {'M2 SIGNAL':<15} | {'DIFF':<5}")
print("-" * 85)

metrics = []

for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        # Fetch info for M1
        info = stock.info
        
        # Fetch history for M1 (price bonus) and M2 (VPT/AD)
        hist = stock.history(period="3mo")
        
        m1_score, m1_ratio = calculate_method_1(info, hist)
        m2_score, m2_signal = calculate_method_2(hist)
        
        diff = m2_score - m1_score
        
        print(f"{ticker.replace('.NS', ''):<15} | {m1_score:<10} | {m1_ratio:.2f}x      || {m2_score:<10} | {m2_signal:<15} | {diff:+.1f}")
        
        metrics.append({
            'ticker': ticker, 'm1': m1_score, 'm2': m2_score, 'diff': diff
        })
        
    except Exception as e:
        print(f"{ticker:<15} | ERROR: {str(e)}")

print("-" * 85)
print("\nANALYSIS:")
avg_diff = np.mean([abs(m['diff']) for m in metrics])
print(f"Average Absolute Difference: {avg_diff:.1f}")
print("Note: Method 1 is snapshot-based (recent vs avg). Method 2 is trend-based (cumulative buying/selling).")
