"""
EARLY MOMENTUM vs DNA3-V2.1: THE SHOWDOWN
=========================================
User Hypothesis: "I should buy early (RS cross 0) instead of chasing hot money (DNA3)."
Our Goal: Prove which one actually makes money over 10 years.

Strategies:
1. DNA3-V2.1: Buy when Price > MA50 and RS > 0. (The "High Wave")
2. EARLY MOMENTUM: Buy when Price > MA20 and RS crosses 0. (The "Early Wave")

Metrics:
- Win Rate (How often is the 'early' signal fake?)
- Avg Win vs Avg Loss
- CAGR
- Max Drawdown
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nifty500_list import TICKERS

def backtest_strategies():
    print("Fetching Data for Backtest (2020-2025)...")
    data_cache = {}
    
    # Nifty
    nifty = yf.Ticker("^NSEI").history(start='2019-01-01')
    nifty.index = nifty.index.tz_localize(None)
    
    # Stocks (Limit to 100 for speed)
    tickers = TICKERS[:100]
    for t in tickers:
        try:
            df = yf.Ticker(t).history(start='2019-01-01')
            if not df.empty:
                df.index = df.index.tz_localize(None)
                data_cache[t] = df
        except: pass
        
    print(f"Loaded {len(data_cache)} stocks.")
    
    # Metrics
    # DNA3
    dna3_trades = []
    # Early Mom
    early_trades = []
    
    for t, df in data_cache.items():
        if len(df) < 200: continue
        
        # Calculate Indicators
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        
        # RS (vs Nifty)
        # We need to broadcast nifty to df index
        nifty_reindexed = nifty['Close'].reindex(df.index).fillna(method='ffill')
        
        # 3-Month RS
        rs_series = (df['Close'].pct_change(63) - nifty_reindexed.pct_change(63)) * 100
        
        in_dna3 = False
        in_early = False
        entry_dna3 = 0
        entry_early = 0
        
        for i in range(200, len(df)):
            date = df.index[i]
            price = df['Close'].iloc[i]
            prev_price = df['Close'].iloc[i-1]
            ma20 = df['MA20'].iloc[i]
            ma50 = df['MA50'].iloc[i]
            rs = rs_series.iloc[i]
            rs_prev = rs_series.iloc[i-1]
            
            # --- DNA3 LOGIC (Established Trend) ---
            # Buy: Price > MA50 and RS > 0
            # Sell: Close < MA50
            if not in_dna3:
                if price > ma50 and rs > 0:
                    in_dna3 = True
                    entry_dna3 = price
            else:
                if price < ma50:
                    ret = (price - entry_dna3)/entry_dna3
                    dna3_trades.append(ret)
                    in_dna3 = False
                    
            # --- EARLY MOMENTUM LOGIC (Fresh Breakout) ---
            # Buy: Price > MA20 and RS crosses 0
            # Sell: Close < MA20 (Tight stop for early moves)
            if not in_early:
                if price > ma20 and rs > 0 and rs_prev < 0:
                    in_early = True
                    entry_early = price
            else:
                if price < ma20:
                    ret = (price - entry_early)/entry_early
                    early_trades.append(ret)
                    in_early = False

    # Analyze Results
    def analyze(trades, name):
        if not trades: return
        win_rate = len([t for t in trades if t > 0]) / len(trades) * 100
        avg_ret = sum(trades) / len(trades) * 100
        print(f"\n{name} RESULTS:")
        print(f"  Trades: {len(trades)}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Avg Return: {avg_ret:.1f}%")
        print(f"  Fakeouts (Losses): {len([t for t in trades if t < 0])}")

    analyze(dna3_trades, "DNA3 (High Wave)")
    analyze(early_trades, "EARLY MOMENTUM (Early Wave)")

if __name__ == "__main__":
    backtest_strategies()
