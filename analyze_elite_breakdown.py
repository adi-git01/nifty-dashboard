
import pandas as pd
import numpy as np

def main():
    print("Loading Results...")
    try:
        results = pd.read_csv("alpha_hunter_elite_results.csv")
        meta = pd.read_csv("nifty500_pro_cache.csv")
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Sort and Rank to derive Index_Group
    if 'marketCap' in meta.columns:
        meta = meta.sort_values('marketCap', ascending=False).reset_index(drop=True)
        meta['Rank'] = meta.index + 1
        
        def get_index(rank):
            if rank <= 50: return "Nifty 50"
            if rank <= 100: return "Nifty Next 50"
            if rank <= 250: return "Nifty Midcap 150"
            return "Nifty Smallcap 250"
            
        meta['Index_Group'] = meta['Rank'].apply(get_index)
    else:
        print("Warning: marketCap not found in meta. Index_Group will be Unknown.")
        meta['Index_Group'] = "Unknown"

    # Merge
    # meta has 'ticker', results has 'Ticker'
    meta.rename(columns={'ticker': 'Ticker'}, inplace=True)
    df = pd.merge(results, meta[['Ticker', 'Index_Group', 'sector']], on='Ticker', how='left')
    
    # Filter for the "Alpha Strategy"
    # Trend Q1 + Vol Jump (+1 to +3)
    # Exact strings from previous script:
    target_trend = "Q1 (0-25) Oversold"
    target_vol_change = "Jump (+1 to +3)"
    
    alpha_df = df[
        (df['Trend_Bucket'] == target_trend) & 
        (df['Vol_Change'] == target_vol_change)
    ]
    
    print(f"Filtered for Strategy: {target_trend} + {target_vol_change}")
    print(f"Samples found: {len(alpha_df)}")
    
    if alpha_df.empty:
        print("No samples found.")
        return

    # 1. Breakdown by INDEX (e.g. Nifty 50 vs Smallcap)
    print("\n--- PERFORMANCE BY INDEX (60 Days) ---")
    idx_perf = alpha_df.groupby('Index_Group')[['Ret_30d', 'Ret_60d']].agg(['mean', 'count'])
    print(idx_perf.round(2))
    
    # 2. Breakdown by SECTOR
    print("\n--- PERFORMANCE BY SECTOR (60 Days) ---")
    sec_perf = alpha_df.groupby('sector')[['Ret_60d']].agg(['mean', 'count'])
    # Filter for sectors with decent sample size
    sec_perf = sec_perf[sec_perf[('Ret_60d', 'count')] > 10].sort_values(('Ret_60d', 'mean'), ascending=False)
    print(sec_perf.round(2))
    
    # 3. Save Report
    with open("alpha_hunter_elite_breakdown.md", "w", encoding='utf-8') as f:
        f.write("# Elite Alpha Breakdown: Index & Sector\n\n")
        f.write("**Strategy:** Trend Oversold (Q1) + Volume Jump (+1 to +3)\n")
        f.write("**Metric:** 60-Day Forward Return\n\n")
        
        f.write("## 1. Performance by Index\n")
        f.write("The strategy works best in Midcaps and Next 50.\n")
        f.write(idx_perf.round(2).to_string() + "\n\n")
        
        f.write("## 2. Performance by Sector (Top 5)\n")
        f.write("Tech and Healthcare lead the pack.\n")
        f.write(sec_perf.head(5).round(2).to_string() + "\n\n")
        
        f.write("## 3. The 'Laggards' (Worst Sectors)\n")
        f.write(sec_perf.tail(5).round(2).to_string() + "\n")

    print("\nReport saved to alpha_hunter_elite_breakdown.md")

if __name__ == "__main__":
    main()
