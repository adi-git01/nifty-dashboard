"""
ANALYZE DNA3 BACKTEST LOGS
==========================
Reads the generated trade log and produces the final comparison report.
"""

import pandas as pd
import yfinance as yf
import numpy as np

# CONFIG
PERIODS = {
    '3Y': ('2023-01-01', '2026-01-01'),
    '5Y': ('2021-01-01', '2026-01-01'),
    '10Y': ('2016-01-01', '2026-01-01')
}

def analyze():
    if not os.path.exists("dna3_full_comparison_trades.csv"):
        print("Trade log not found.")
        return

    df = pd.read_csv("dna3_full_comparison_trades.csv")
    df['Entry'] = pd.to_datetime(df['Entry'])
    df['Exit'] = pd.to_datetime(df['Exit'])
    
    nifty = yf.Ticker("^NSEI").history(period="max")
    
    results = []
    
    for p_name, (start, end) in PERIODS.items():
        # Filter trades for this period (Entry date within period)
        mask = (df['Entry'] >= start) & (df['Entry'] < end)
        p_df = df[mask]
        
        # Nifty Benchmark
        try:
            n_start_price = nifty.loc[nifty.index >= start]['Close'].iloc[0]
            n_end_price = nifty.loc[nifty.index <= end]['Close'].iloc[-1]
            n_ret = (n_end_price - n_start_price) / n_start_price * 100
            years = (pd.to_datetime(end) - pd.to_datetime(start)).days / 365.25
            n_cagr = ((1 + n_ret/100)**(1/years) - 1) * 100
        except:
            n_ret, n_cagr = 0, 0

        # Add Nifty Row
        results.append({
            'Strategy': 'Nifty 50 (Benchmark)',
            'Period': p_name,
            'Trades': 0, 'Win %': '-', 'Avg Win': '-', 'Max Win': '-', 
            'Avg Drawdown': '-', 'CAGR': f"{n_cagr:.1f}%", 'Total Return': f"{n_ret:.1f}%"
        })

        for strat in ['Standard', 'Turnaround']:
            s_df = p_df[p_df['Strategy'] == strat]
            
            if s_df.empty: continue
            
            wins = s_df[s_df['Return'] > 0]
            win_rate = len(wins) / len(s_df) * 100
            
            avg_ret = s_df['Return'].mean() * 100
            avg_win = wins['Return'].mean() * 100
            
            # Max Win
            max_r = s_df['Return'].max()
            max_row = s_df.loc[s_df['Return'].idxmax()]
            max_win_str = f"{max_r*100:.0f}% ({max_row['Ticker']})"
            
            # Est CAGR (Heuristic based on performance)
            # DNA3 Standard alpha ~ 2.0x Nifty (from 5Y test)
            # DNA3 Turnaround alpha ~ 3.4x Standard (from 5Y test)
            # Let's verify with Total Return Sum
            # Turnaround trade count is very low (selectivity).
            # So holding period matters.
            # Let's show Avg Return per Trade as key metric.
            
            total_ret_points = s_df['Return'].sum() * 100
            
            results.append({
                'Strategy': f"DNA3 {strat}",
                'Period': p_name,
                'Trades': len(s_df),
                'Win %': f"{win_rate:.1f}%",
                'Avg Win': f"{avg_win:.1f}%",
                'Avg Return': f"{avg_ret:.1f}%",
                'Max Win': max_win_str,
                'Total Ret Sum': f"{total_ret_points:.0f}%", # Proxy for activity
            })
            
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

import os
if __name__ == "__main__":
    analyze()
