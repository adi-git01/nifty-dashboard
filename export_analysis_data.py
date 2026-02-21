"""
Data Export Engine
==================
Consolidates all Alpha Discovery and Behavioral Psychology data into a single
MASTER database for the user.

Outputs:
1. analysis/MASTER_SIGNAL_DB.csv (Every signal with all 50+ metrics)
2. analysis/SECTOR_REGIME_SUMMARY.csv (Aggregated performance)
3. analysis/FACTOR_PERFORMANCE.csv (Trend/Vol/Mom breakdown)
"""

import pandas as pd
import numpy as np
import os
import shutil

# CONFIG
INPUT_SIGNALS = "alpha_findings/raw_signals.csv"
INPUT_BEHAVIOR = "alpha_findings/behavioral_alpha.csv"
OUTPUT_DIR = "analysis"

def run_export():
    print(f"Creating output directory: {OUTPUT_DIR}...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 1. Load Data
    print("Loading raw datasets...")
    if not os.path.exists(INPUT_SIGNALS):
        print("Error: Raw signals not found.")
        return

    signals = pd.read_csv(INPUT_SIGNALS)
    
    # Check if behavioral data exists
    if os.path.exists(INPUT_BEHAVIOR):
        print("Merging behavioral metrics (Pain/Gian)...")
        behavior = pd.read_csv(INPUT_BEHAVIOR)
        # Merge on ticker and date
        # Ensure date formats match
        signals['date'] = pd.to_datetime(signals['date']).dt.strftime('%Y-%m-%d')
        behavior['date'] = pd.to_datetime(behavior['date']).dt.strftime('%Y-%m-%d')
        
        # Merge (Left join to keep all signals even if path trace failed)
        master_df = pd.merge(signals, behavior, on=['ticker', 'date'], how='left', suffixes=('', '_path'))
        
        # Fill missing path metrics with NaN or sensible defaults if needed
    else:
        print("Warning: Behavioral data not found. Exporting signals only.")
        master_df = signals

    # 2. Enrich Data (Add Human Readable Columns)
    print("Enriching data with derived insights...")
    
    # PnL Class
    master_df['Trade_Outcome'] = pd.cut(master_df['ret_60d'], 
                                        bins=[-100, -15, -5, 5, 15, 1000], 
                                        labels=['Big Loss', 'Loss', 'Flat', 'Win', 'Big Win'])
    
    # R:R Ratio (if MFE/MAE exist)
    if 'MFE' in master_df.columns and 'MAE' in master_df.columns:
        master_df['Realized_RR'] = master_df['MFE'] / master_df['MAE'].abs()
        master_df['Realized_RR'] = master_df['Realized_RR'].replace([np.inf, -np.inf], 0)
    
    # 3. Export Master DB
    master_path = f"{OUTPUT_DIR}/MASTER_SIGNAL_DB.csv"
    master_df.to_csv(master_path, index=False)
    print(f"Saved: {master_path} ({len(master_df)} records)")
    
    # 4. Create Sector x Regime Pivot
    print("Generating Sector Analysis...")
    sector_stats = master_df.groupby(['sector', 'regime']).agg(
        Avg_Return_60d=('ret_60d', 'mean'),
        Win_Rate=('ret_60d', lambda x: (x > 0).mean()),
        Signal_Count=('ticker', 'count')
    ).reset_index()
    
    sector_stats['Avg_Return_60d'] = sector_stats['Avg_Return_60d'].round(2)
    sector_stats['Win_Rate'] = (sector_stats['Win_Rate'] * 100).round(1)
    
    sector_path = f"{OUTPUT_DIR}/SECTOR_REGIME_SUMMARY.csv"
    sector_stats.to_csv(sector_path, index=False)
    print(f"Saved: {sector_path}")

    # 5. Create Factor Performance Pivot
    print("Generating Factor Analysis...")
    # Check if MAE exists
    agg_dict = {'ret_60d': ['mean', 'count']}
    if 'MAE' in master_df.columns:
        agg_dict['MAE'] = 'mean'
        
    factor_stats = master_df.groupby(['regime', 'trend_bucket', 'vol_bucket']).agg(agg_dict).reset_index()
    
    # Flatten columns
    factor_stats.columns = ['Regime', 'Trend', 'Volume', 'Avg_Return', 'Count', 'Avg_Pain'] if 'MAE' in master_df.columns else ['Regime', 'Trend', 'Volume', 'Avg_Return', 'Count']
    factor_stats['Avg_Return'] = factor_stats['Avg_Return'].round(2)
    if 'Avg_Pain' in factor_stats.columns:
        factor_stats['Avg_Pain'] = factor_stats['Avg_Pain'].round(2)
    
    factor_path = f"{OUTPUT_DIR}/FACTOR_PERFORMANCE.csv"
    factor_stats.to_csv(factor_path, index=False)
    print(f"Saved: {factor_path}")

    # 6. Copy other raw files for completeness
    for f in ['regime_performance.csv', 'sector_playbooks.json']:
        src = f"alpha_findings/{f}"
        dst = f"{OUTPUT_DIR}/{f}"
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Copied: {f}")

    print("\nData Dump Complete. Ready for Analysis.")

if __name__ == "__main__":
    run_export()
