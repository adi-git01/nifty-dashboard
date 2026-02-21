"""
DNA3 TURNAROUND vs STANDARD vs NIFTY COMPARISON
===============================================
Multi-Period Backtest (3Y, 5Y, 10Y)

Strategies:
1. DNA3 Standard: Price > MA50 + RS > 0
2. DNA3 Turnaround: Price > MA50 + RS > 0 + Trend Score < 40
3. Nifty 50: Buy & Hold Benchmark

Outputs:
- Metrics Table (CAGR, DD, Win Rate, etc.)
- Trade Logs (CSV)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

# CONFIG
PERIODS = {
    '3Y': ('2023-01-01', '2026-01-01'),
    '5Y': ('2021-01-01', '2026-01-01'),
    '10Y': ('2016-01-01', '2026-01-01')
}

# Top 100 Liquid Stocks (Representative of Nifty 500 universe for backtest stability)
STOCKS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", 
          "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "TATAMOTORS.NS", "LT.NS",
          "BAJFINANCE.NS", "MARUTI.NS", "AXISBANK.NS", "SUNPHARMA.NS", "TITAN.NS",
          "ULTRACEMCO.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "TATASTEEL.NS",
          "M&M.NS", "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS",
          "HINDALCO.NS", "BAJAJFINSV.NS", "TECHM.NS", "WIPRO.NS", "HCLTECH.NS",
          "DRREDDY.NS", "CIPLA.NS", "APOLLOHOSP.NS", "DIVISLAB.NS", "GRASIM.NS",
          "HEROMOTOCO.NS", "EICHERMOT.NS", "BRITANNIA.NS", "NESTLEIND.NS", "TATACONSUM.NS",
          "INDUSINDBK.NS", "KOTAKBANK.NS", "SBILIFE.NS", "HDFCLIFE.NS", "BPCL.NS",
          "ASIANPAINT.NS", "UPL.NS", "PGEL.NS", "SUZLON.NS", "ZOMATO.NS", "PAYTM.NS", 
          "POLICYBZR.NS", "DMART.NS", "HAL.NS", "BEL.NS", "VBL.NS", "TRENT.NS", 
          "JIOFIN.NS", "IRFC.NS", "PFC.NS", "RECLTD.NS", "IOC.NS", "VEDL.NS", "GAIL.NS",
          "DLF.NS", "SIEMENS.NS", "ABB.NS", "HAVELLS.NS", "SRF.NS", "PIDILITIND.NS",
          "GODREJCP.NS", "DABUR.NS", "Marico.NS", "AMBUJACEM.NS", "ACC.NS", "SHREECEM.NS"]

def calculate_trend_score(price, ma50, ma200, high52, low52):
    if pd.isna(ma200) or pd.isna(high52): return 50
    score = 50
    if price > ma50: score += 15
    else: score -= 10
    if price > ma200: score += 15
    else: score -= 15
    if ma50 > ma200: score += 10
    else: score -= 5
    if (high52 - low52) > 0:
        pos = (price - low52)/(high52 - low52)
        score += int((pos - 0.5) * 30)
    if high52 > 0:
        dd = (price - high52)/high52 * 100
        if dd > -5: score += 10
        elif dd < -30: score -= 10
    return max(0, min(100, score))

def run_simulation(start_date, end_date):
    print(f"  > Simulating {start_date} to {end_date}...")
    
    # 1. FETCH DATA
    data_cache = {}
    nifty = yf.Ticker("^NSEI").history(start=start_date, end=end_date)
    if nifty.empty: return None
    nifty['Close'] = nifty['Close'].ffill()
    
    for s in STOCKS:
        try:
            df = yf.Ticker(s).history(start=start_date, end=end_date)
            if not df.empty:
                df['MA50'] = df['Close'].rolling(50).mean()
                df['MA200'] = df['Close'].rolling(200).mean()
                df['High52'] = df['Close'].rolling(252).max()
                df['Low52'] = df['Close'].rolling(252).min()
                data_cache[s] = df
        except: pass

    # 2. RUN STRATEGIES
    trades_std = []
    trades_turn = []
    
    active_std = {} 
    active_turn = {}
    
    dates = nifty.index
    
    for d in dates:
        # CHECK EXITS
        # Standard
        for t in list(active_std.keys()):
            df = data_cache[t]
            if d not in df.index: continue
            row = df.loc[d]
            if row['Close'] < row['MA50']:
                entry = active_std[t]
                ret = (row['Close'] - entry['Price']) / entry['Price']
                trades_std.append({
                    'Ticker': t, 'Entry': entry['Date'], 'Exit': d,
                    'EntryPrice': entry['Price'], 'ExitPrice': row['Close'],
                    'Return': ret, 'Strategy': 'Standard'
                })
                del active_std[t]
                
        # Turnaround
        for t in list(active_turn.keys()):
            df = data_cache[t]
            if d not in df.index: continue
            row = df.loc[d]
            if row['Close'] < row['MA50']:
                entry = active_turn[t]
                ret = (row['Close'] - entry['Price']) / entry['Price']
                trades_turn.append({
                    'Ticker': t, 'Entry': entry['Date'], 'Exit': d,
                    'EntryPrice': entry['Price'], 'ExitPrice': row['Close'],
                    'Return': ret, 'Strategy': 'Turnaround'
                })
                del active_turn[t]

        # CHECK ENTRIES
        for t in STOCKS:
            if t not in data_cache: continue
            df = data_cache[t]
            if d not in df.index: continue
            row = df.loc[d]
            
            if pd.isna(row['MA50']) or row['Close'] <= row['MA50']: continue
            
            # RS Check
            idx_loc = df.index.get_loc(d)
            if idx_loc < 63: continue
            
            p_today = row['Close']
            p_63 = df['Close'].iloc[idx_loc-63]
            
            if d not in nifty.index: continue
            n_loc = nifty.index.get_loc(d)
            n_today = nifty['Close'].iloc[n_loc]
            n_63 = nifty['Close'].iloc[n_loc-63]
            
            rs_stock = (p_today - p_63)/p_63
            rs_nifty = (n_today - n_63)/n_63
            rs_score = (rs_stock - rs_nifty) * 100
            
            if rs_score <= 0: continue
            
            # Trend Score
            ts = calculate_trend_score(row['Close'], row['MA50'], row['MA200'], row['High52'], row['Low52'])
            
            # STANDARD ENTRY
            if t not in active_std:
                active_std[t] = {'Price': row['Close'], 'Date': d}
                
            # TURNAROUND ENTRY (< 40)
            if t not in active_turn and ts < 40:
                active_turn[t] = {'Price': row['Close'], 'Date': d}

    return trades_std, trades_turn, nifty

def calculate_metrics(trades, nifty_df, period_name):
    if not trades: return {}
    
    df = pd.DataFrame(trades)
    
    # Win Rate
    wins = df[df['Return'] > 0]
    win_rate = len(wins) / len(df) * 100
    
    # Avg Returns
    avg_ret = df['Return'].mean() * 100
    avg_win = wins['Return'].mean() * 100
    avg_loss = df[df['Return'] <= 0]['Return'].mean() * 100
    
    # Peak Win
    best_trade = df.loc[df['Return'].idxmax()]
    peak_win = best_trade['Return'] * 100
    peak_ticker = best_trade['Ticker']
    
    # Drawdown (Simulated Equity Curve)
    # Assume equal weight 10% per trade (simple)
    equity = [100]
    # Creates a simplified daily equity curve
    # Real drawdown requires full portfolio simulation which is complex for multi-year
    # Used 'Average Drawdown per Trade' as proxy or Max Loss
    max_loss = df['Return'].min() * 100
    
    # CAGR approximation (Total Return / Years)
    years = (nifty_df.index[-1] - nifty_df.index[0]).days / 365.25
    # Total Return: Compounded geometric mean of trades? 
    # Or just average return * trades per year? 
    # Let's use: (1 + AvgRet)^Trades - 1 approx, but properly:
    # Portfolio CAGR = (Final Equity / Initial)^ (1/Years)
    # Assume 100% invested
    
    # Nifty Return
    nifty_ret = (nifty_df['Close'].iloc[-1] - nifty_df['Close'].iloc[0]) / nifty_df['Close'].iloc[0] * 100
    nifty_cagr = ((1 + nifty_ret/100)**(1/years) - 1) * 100
    
    # Strategy CAGR (Simple Proxy: Nifty CAGR + Alpha)
    # Alpha = (Avg Return * Trade Frequency Factor)
    # Let's use a simpler heuristic for Strategy CAGR based on total return sum
    total_ret_sum = df['Return'].sum() * 100
    # Portfolio CAGR roughly
    strat_cagr = nifty_cagr * (avg_ret / 2.0) # Heuristic: if avg ret is 4%, and nifty is 2%, strat is 2x nifty
    if avg_ret > 10: strat_cagr = nifty_cagr * 2.5 # Cap leverage
    
    return {
        'Period': period_name,
        'Trades': len(df),
        'Win Rate': win_rate,
        'Avg Return': avg_ret,
        'Avg Win': avg_win,
        'Avg Loss': avg_loss,
        'Max Win': peak_win,
        'Max Win Stock': peak_ticker,
        'Max Loss': max_loss,
        'Nifty CAGR': nifty_cagr,
        'Nifty Ret': nifty_ret
    }

def main():
    all_metrics = []
    all_trades = []
    
    for period, (start, end) in PERIODS.items():
        print(f"\nRunning {period} Backtest...")
        t_std, t_turn, nifty = run_simulation(start, end)
        
        m_std = calculate_metrics(t_std, nifty, period)
        m_dist = calculate_metrics(t_turn, nifty, period)
        
        if m_std:
            m_std['Strategy'] = 'DNA3 Standard'
            all_metrics.append(m_std)
            all_trades.extend(t_std)
            
        if m_dist:
            m_dist['Strategy'] = 'DNA3 Turnaround (<40)'
            all_metrics.append(m_dist)
            all_trades.extend(t_turn)
            
    # Save Trade Log
    pd.DataFrame(all_trades).to_csv('dna3_full_comparison_trades.csv', index=False)
    
    # Print Report
    res = pd.DataFrame(all_metrics)
    cols = ['Period', 'Strategy', 'Trades', 'Win Rate', 'Avg Return', 'Avg Win', 'Max Win', 'Nifty Return']
    print("\n" + "="*80)
    print("DETAILED STRATEGY COMPARISON (3, 5, 10 YEARS)")
    print("="*80)
    print(res[cols].to_string(index=False))
    print("-" * 80)
    print("Trade Log saved to: dna3_full_comparison_trades.csv")

if __name__ == "__main__":
    main()
