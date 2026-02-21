"""
DNA3 LOW TREND SCORE BACKTEST (2020-2025)
=========================================
Strategy:
1. DNA3 Entry: Price > MA50 AND RS > 0
2. FILTER: Trend Score < 40 (Turnaround Plays / Deep Value)
3. Exit: Close < MA50

Hypothesis: Does buying "battered" stocks that are just starting to recover (DNA3) outperform?
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

# CONFIG
START_DATE = "2020-01-01"
END_DATE = "2025-02-01"
# Top liquid stocks to ensure realistic execution
STOCKS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", 
          "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "TATAMOTORS.NS", "L&T.NS",
          "BAJFINANCE.NS", "MARUTI.NS", "AXISBANK.NS", "SUNPHARMA.NS", "TITAN.NS",
          "ULTRACEMCO.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "TATASTEEL.NS",
          "JSWSTEEL.NS", "M&M.NS", "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS",
          "HINDALCO.NS", "BAJAJFINSV.NS", "TECHM.NS", "WIPRO.NS", "HCLTECH.NS",
          "DRREDDY.NS", "CIPLA.NS", "APOLLOHOSP.NS", "DIVISLAB.NS", "GRASIM.NS",
          "HEROMOTOCO.NS", "EICHERMOT.NS", "BRITANNIA.NS", "NESTLEIND.NS", "TATACONSUM.NS",
          "INDUSINDBK.NS", "KOTAKBANK.NS", "SBILIFE.NS", "HDFCLIFE.NS", "BPCL.NS",
          "ASIANPAINT.NS", "UPL.NS", "PGEL.NS", "SUZLON.NS", "ZOMATO.NS", "PAYTM.NS", 
          "POLICYBZR.NS", "DMART.NS", "HAL.NS", "BEL.NS", "VBL.NS", "TRENT.NS", 
          "JIOFIN.NS", "IRFC.NS", "PFC.NS", "RECLTD.NS", "IOC.NS", "VEDL.NS", "GAIL.NS",
          "DLF.NS", "SIEMENS.NS", "ABB.NS", "HAVELLS.NS", "SRF.NS", "PIDILITIND.NS",
          "GODREJCP.NS", "DABUR.NS", "MARICO.NS", "AMBUJACEM.NS", "ACC.NS", "SHREECEM.NS"]

def calculate_trend_score(price, ma50, ma200, high52, low52):
    if pd.isna(ma200) or pd.isna(high52): return 50
    
    score = 50
    # MA50
    if price > ma50: score += 15
    else: score -= 10
    
    # MA200
    if price > ma200: score += 15
    else: score -= 15
    
    # Cross
    if ma50 > ma200: score += 10
    else: score -= 5
    
    # 52W Range
    range_52 = high52 - low52
    if range_52 > 0:
        pos = (price - low52)/range_52
        score += int((pos - 0.5) * 30)
    
    # Drawdown
    if high52 > 0:
        dd = (price - high52)/high52 * 100
        if dd > -5: score += 10
        elif dd < -30: score -= 10
        
    return max(0, min(100, score))

def run_backtest():
    print(f"Fetching Data for {len(STOCKS)} stocks (2019-2025)...")
    data_cache = {}
    
    # Download Nifty for RS
    nifty = yf.Ticker("^NSEI").history(start="2019-01-01")
    nifty['Close'] = nifty['Close'].fillna(method='ffill')
    
    # Download Stocks
    for s in STOCKS:
        try:
            df = yf.Ticker(s).history(start="2019-01-01")
            if df.empty: continue
            
            # Pre-calc Indicators
            df['MA50'] = df['Close'].rolling(50).mean()
            df['MA200'] = df['Close'].rolling(200).mean()
            df['High52'] = df['Close'].rolling(252).max()
            df['Low52'] = df['Close'].rolling(252).min()
            
            data_cache[s] = df
        except: pass

    print(f"Loaded {len(data_cache)} stocks.")
    
    trades = [] # {Ticker, EntryDate, EntryPrice, ExitDate, ExitPrice, Return, Duration}
    active_trades = {} # {Ticker: {EntryPrice, EntryDate}}
    
    # Date Range
    dates = nifty.index[nifty.index >= START_DATE]
    dates = dates[dates <= END_DATE]
    
    print("Running Simulation...")
    for d in dates:
        # 1. Check Exits
        for t in list(active_trades.keys()):
            df = data_cache[t]
            if d not in df.index: continue
            
            row = df.loc[d]
            entry_price = active_trades[t]['Price']
            
            # Exit: Price < MA50
            if row['Close'] < row['MA50']:
                ret = (row['Close'] - entry_price) / entry_price
                trades.append({
                    'Ticker': t,
                    'Entry_Date': active_trades[t]['Date'],
                    'Exit_Date': d,
                    'Return': ret
                })
                del active_trades[t]
        
        # 2. Check Entries
        for t in STOCKS:
            if t in active_trades or t not in data_cache: continue
            df = data_cache[t]
            if d not in df.index: continue
            
            row = df.loc[d]
            
            # DNA3 Criteria
            if pd.isna(row['MA50']): continue
            if row['Close'] <= row['MA50']: continue
            
            # RS Check
            idx_loc = df.index.get_loc(d)
            if idx_loc < 63: continue
            
            # Calculate RS
            p_today = row['Close']
            p_63 = df['Close'].iloc[idx_loc-63]
            
            # Align Nifty
            if d not in nifty.index: continue
            n_loc = nifty.index.get_loc(d)
            n_today = nifty['Close'].iloc[n_loc]
            n_63 = nifty['Close'].iloc[n_loc-63]
            
            rs_stock = (p_today - p_63)/p_63
            rs_nifty = (n_today - n_63)/n_63
            rs_score = (rs_stock - rs_nifty) * 100
            
            if rs_score <= 0: continue
            
            # === TREND SCORE FILTER ===
            ts = calculate_trend_score(row['Close'], row['MA50'], row['MA200'], row['High52'], row['Low52'])
            
            if ts < 40:
                # TURNAROUND BUY!
                active_trades[t] = {'Price': row['Close'], 'Date': d}
                
    # Results
    if not trades:
        print("No trades found.")
        return

    res_df = pd.DataFrame(trades)
    win_rate = len(res_df[res_df['Return'] > 0]) / len(res_df) * 100
    avg_ret = res_df['Return'].mean() * 100
    total_ret = res_df['Return'].sum() * 100 # Simple sum not compounded
    
    print("\n" + "="*50)
    print(f"DNA3 + LOW TREND SCORE (<40) BACKTEST ({START_DATE} to {END_DATE})")
    print("="*50)
    print(f"Total Trades: {len(res_df)}")
    print(f"Win Rate:     {win_rate:.1f}%")
    print(f"Avg Return:   {avg_ret:.1f}%")
    print("-" * 50)
    
    # Winners
    print("\nTop 5 Winners:")
    print(res_df.sort_values('Return', ascending=False).head(5)[['Ticker', 'Entry_Date', 'Return']])

    # Compare with Baseline DNA3 (Approx from previous runs)
    print("\nvs BASELINE DNA3 (All Trend Scores):")
    print("Avg Return: ~4.0% (Baseline) vs", f"{avg_ret:.1f}% (Low Trend)")
    
if __name__ == "__main__":
    run_backtest()
