"""
DNA3-V2.1 vs NIFTY BENCHMARK (2020-2025)
========================================
Compares the strategy return vs Nifty 50 Buy & Hold.

Strategy: DNA3-V2.1 (Pure Momentum)
Benchmark: Nifty 50 Index (Total Return)
Period: Jan 1, 2020 - Feb 1, 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Re-use the existing 10Y comparison logic but for 5Y
from dna_10y_cagr_comparison import TenYearComparison, calculate_metrics

def main():
    print("Fetching Data for 2020-2025 Benchmark...")
    sim = TenYearComparison(start_date='2020-01-01', end_date='2025-02-01')
    sim.fetch_data()
    
    # Run DNA3
    h_dna3 = sim.run_backtest('DNA3-V2.1')
    
    # Run Nifty (Buy & Hold)
    nifty_hist = sim.data_cache['NIFTY']
    start_val = nifty_hist.loc[nifty_hist.index >= '2020-01-01']['Close'].iloc[0]
    nifty_df = pd.DataFrame({
        'date': nifty_hist[nifty_hist.index >= '2020-01-01'].index,
        'value': nifty_hist[nifty_hist.index >= '2020-01-01']['Close'] / start_val * 1000000 # Normalized to 10L start
    })
    
    # Metrics
    metrics = []
    metrics.append(calculate_metrics(h_dna3, 'DNA3-V2.1 (Momentum)'))
    metrics.append(calculate_metrics(nifty_df, 'Nifty 50 (Benchmark)'))
    
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON (2020-2025)")
    print("="*80)
    res_df = pd.DataFrame(metrics)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()
