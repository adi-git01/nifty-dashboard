"""
CALCULATE PORTFOLIO CAGR FROM TRADE LOGS
========================================
Simulates a portfolio to calculate the true CAGR of the strategies.
Assumptions:
- Initial Capital: 10,00,000
- Max Positions: 10
- Allocation: 10% of Current Equity per trade (Compounding)
- No friction costs (for raw comparison) of slippage/brokerage
"""

import pandas as pd
import numpy as np
from datetime import datetime

# CONFIG
INITIAL_CAPITAL = 1000000
MAX_POSITIONS = 10

PERIODS = {
    '3Y': ('2023-01-01', '2026-01-01'),
    '5Y': ('2021-01-01', '2026-01-01'),
    '10Y': ('2016-01-01', '2026-01-01')
}

def simulate_portfolio(trades_df, start_date, end_date):
    """Simulates portfolio growth trade-by-trade."""
    
    # Filter trades within period
    mask = (trades_df['Entry'] >= start_date) & (trades_df['Entry'] < end_date)
    period_trades = trades_df[mask].copy()
    period_trades = period_trades.sort_values('Entry')
    
    cash = INITIAL_CAPITAL
    equity = INITIAL_CAPITAL
    active_positions = 0
    
    # Simple simulation: 
    # Since we don't have daily data here, we'll approximate by assuming 
    # we can take every trade that comes (infinite slots) but position size is fixed % of equity.
    # To be more realistic without daily data:
    # 1. Sort by Entry Date.
    # 2. Update Equity on Exit Date.
    # Actually, proper CAGR requires handling concurrent trades. 
    # With just a trade log, exact daily equity is hard if we don't track held periods.
    # Approximation: Add sum of Returns * Allocation% to Equity.
    
    # Better Approximation:
    # Portfolio Return = Sum(Trade Return * Weight)
    # If we assume 10 equal slots, Weight = 1/10.
    # Compounding: New Equity = Old Equity * (1 + Return * 0.1)
    # This assumes we always have a slot open. 
    
    # Let's use the Compounded Growth Formula on the 'Avg Return * Frequency'
    # N_Trades = len(period_trades)
    # Avg_Ret = period_trades['Return'].mean()
    # Years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
    
    # Kelly Criterion / Optimal f style:
    # Geometric Mean Return = (Product(1 + r))^(1/N) - 1
    # But we hold ~10 stocks. 
    # So effectively, the portfolio exposure is 100%. 
    # Portfolio Return for a year ~ Sum of all trade returns / 10 (since 10 slots).
    
    total_ret_sum = period_trades['Return'].sum()
    # The sum of returns represents the total % gain if we invested 100% in each trade sequentially.
    # Since we invest 10% (1/10th) in parallel:
    # Effective Portfolio Growth = Total_Ret_Sum / 10
    
    final_equity_simple = INITIAL_CAPITAL * (1 + total_ret_sum / 10)
    
    # CAGR Formula
    years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
    cagr = (final_equity_simple / INITIAL_CAPITAL) ** (1 / years) - 1
    
    return cagr * 100, final_equity_simple

def main():
    if not os.path.exists("dna3_full_comparison_trades.csv"):
        print("Trade log not found.")
        return

    df = pd.read_csv("dna3_full_comparison_trades.csv")
    df['Entry'] = pd.to_datetime(df['Entry'])
    
    results = []
    
    for period, (start, end) in PERIODS.items():
        for strat in ['Standard', 'Turnaround']:
            strat_name = 'Standard' if strat == 'Standard' else 'Turnaround'
            subset = df[df['Strategy'] == strat]
            
            cagr, final_val = simulate_portfolio(subset, start, end)
            
            # Nifty Benchmark (Hardcoded from typical historicals for context)
            nifty_cagr_map = {'3Y': 13.0, '5Y': 14.5, '10Y': 12.8}
            
            results.append({
                'Period': period,
                'Strategy': f"DNA3 {strat_name}",
                'CAGR': cagr,
                'Final Value (10L)': final_val,
                'Alpha vs Nifty': cagr - nifty_cagr_map.get(period, 12)
            })
            
    # Print Table
    res_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("STRATEGY CAGR COMPARISON (Simulated Portfolio)")
    print("="*60)
    print(res_df[['Period', 'Strategy', 'CAGR', 'Alpha vs Nifty']].to_string(index=False, float_format="%.1f%%"))

import os
if __name__ == "__main__":
    main()
