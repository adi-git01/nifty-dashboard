"""
Diagnose Composite RS vs 63-Day RS Gap
=======================================
Reads the generated equity curves from dna3_composite_rs_comparison.py
and calculates month-by-month and quarter-by-quarter differences to
isolate exactly WHEN pure 63-day RS outperformed the composite.

Usage:
  python diagnose_composite_gap.py
"""

import pandas as pd
import numpy as np
import warnings
import os
import sys

warnings.filterwarnings('ignore')

DIR = "analysis_2026"
# We need to re-run a fast subset to get the daily equity curves
# since the previous script didn't save the daily equity curves to CSV.

def diagnose_gap():
    print("=" * 100)
    print("DIAGNOSING THE 3Y-5Y COMPOSITE RS UNDERPERFORMANCE")
    print("=" * 100)
    
    # Check if files exist
    base_file = f"{DIR}/composite_rs_rolling_12m.csv"
    if not os.path.exists(base_file):
        print(f"ERROR: Could not find {base_file}. Cannot run diagnosis.")
        return
        
    df = pd.read_csv(base_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Pivot to get strategies as columns
    pivot = df.pivot(index='date', columns='Strategy', values='ret')
    
    if 'V2.1-63d' not in pivot.columns or 'V2.1-Composite' not in pivot.columns:
        print("Required strategies not found in data.")
        return
        
    # Calculate rolling 12M Delta (Composite - 63d)
    pivot['V2.1_Delta'] = pivot['V2.1-Composite'] - pivot['V2.1-63d']
    pivot['V3.1_Delta'] = pivot['V3.1-Composite'] - pivot['V3.1-63d']
    
    # Separate by Year
    pivot['Year'] = pivot.index.year
    
    yearly_delta = pivot.groupby('Year')[['V2.1_Delta', 'V3.1_Delta']].mean()
    
    print("\n[1] AVERAGE ROLLING 12-MONTH DELTA BY YEAR (Composite minus 63d)")
    print("-" * 80)
    print(f"{'Year':<10} {'V2.1 Delta':>15} {'V3.1 Delta':>15} {'Verdict (V2.1)':<20}")
    print("-" * 80)
    
    for year, row in yearly_delta.iterrows():
        d21 = row['V2.1_Delta']
        d31 = row['V3.1_Delta']
        if pd.isna(d21): continue
        
        if d21 > 5: verdict = "Composite crushed it"
        elif d21 > 0: verdict = "Composite won mildly"
        elif d21 > -5: verdict = "63d won mildly"
        else: verdict = "63d crushed it"
            
        print(f"{int(year):<10} {d21:>14.1f}% {d31:>14.1f}%   {verdict}")
        
    
    # Find the worst specific 12-month periods for Composite
    print("\n\n[2] WORST 5 12-MONTH PERIODS FOR COMPOSITE RS (V2.1)")
    print("-" * 80)
    worst = pivot.sort_values('V2.1_Delta').head(5)
    for date, row in worst.iterrows():
        print(f"Ending {date.strftime('%Y-%b')} | 63d: {row['V2.1-63d']:+5.1f}% | Comp: {row['V2.1-Composite']:+5.1f}% | Delta: {row['V2.1_Delta']:+6.1f}%")

    # Find the best specific 12-month periods for Composite
    print("\n\n[3] BEST 5 12-MONTH PERIODS FOR COMPOSITE RS (V2.1)")
    print("-" * 80)
    best = pivot.sort_values('V2.1_Delta', ascending=False).head(5)
    for date, row in best.iterrows():
        print(f"Ending {date.strftime('%Y-%b')} | 63d: {row['V2.1-63d']:+5.1f}% | Comp: {row['V2.1-Composite']:+5.1f}% | Delta: {row['V2.1_Delta']:+6.1f}%")

    
    # Save delta analysis to CSV
    out_file = f"{DIR}/composite_gap_diagnosis.csv"
    yearly_delta.to_csv(out_file)
    print(f"\nAnalysis saved to {out_file}")

if __name__ == "__main__":
    diagnose_gap()
