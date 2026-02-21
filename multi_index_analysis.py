"""
Multi-Index Leading Indicator Analysis
=======================================
Tests mood indicators against multiple market segments:
- Nifty 50 (Large Cap)
- Nifty Next 50 (Large-Mid Cap)
- Nifty Midcap 100
- Nifty Smallcap 100
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Constants
MOOD_FILE = "data/market_mood_history.csv"
OUTPUT_FILE = "multi_index_timing_results.csv"

# Index tickers - comprehensive sector coverage
INDICES = {
    # Broad Market
    'Nifty 50': '^NSEI',
    'Nifty Midcap 50': '^NSMIDCP',
    'Nifty Bank': '^NSEBANK',
    
    # Sector Indices
    'Nifty IT': '^CNXIT',
    'Nifty Pharma': '^CNXPHARMA',
    'Nifty Auto': '^CNXAUTO',
    'Nifty FMCG': '^CNXFMCG',
    'Nifty Metal': '^CNXMETAL',
    'Nifty Realty': '^CNXREALTY',
    'Nifty Energy': '^CNXENERGY',
    'Nifty Infra': '^CNXINFRA',
    'Nifty PSE': '^CNXPSE',
    'Nifty Financial': '^CNXFIN',
}

# Alternative tickers if above don't work
BACKUP_INDICES = {
    'Nifty 50': '^NSEI',
    'Nifty Bank': '^NSEBANK',
    'Nifty IT': '^NSETECH',
}


def load_mood_data():
    """Load mood history."""
    print("\n" + "="*70)
    print("  MULTI-INDEX LEADING INDICATOR ANALYSIS")
    print("  Testing mood indicators across market segments")
    print("="*70)
    
    print("\n1. Loading market mood history...")
    mood_df = pd.read_csv(MOOD_FILE, parse_dates=['date'])
    print(f"   Found {len(mood_df)} days of mood data")
    print(f"   Date range: {mood_df['date'].min().date()} to {mood_df['date'].max().date()}")
    
    return mood_df


def fetch_index_data(mood_df):
    """Fetch data for all indices."""
    print("\n2. Fetching index data...")
    
    start_date = mood_df['date'].min() - timedelta(days=10)
    end_date = mood_df['date'].max() + timedelta(days=300)  # Extra for 9-month forward returns
    
    index_data = {}
    
    for name, ticker in INDICES.items():
        try:
            print(f"   Fetching {name} ({ticker})...")
            data = yf.Ticker(ticker).history(start=start_date, end=end_date)
            
            if not data.empty and len(data) > 50:
                data.index = data.index.tz_localize(None)
                index_data[name] = data['Close'].to_dict()
                print(f"      Got {len(data)} days")
            else:
                print(f"      Insufficient data, skipping")
        except Exception as e:
            print(f"      Error: {e}")
    
    # Try backup indices if we have less than 2
    if len(index_data) < 2:
        print("\n   Trying backup indices...")
        for name, ticker in BACKUP_INDICES.items():
            if name not in index_data:
                try:
                    data = yf.Ticker(ticker).history(start=start_date, end=end_date)
                    if not data.empty:
                        data.index = data.index.tz_localize(None)
                        index_data[name] = data['Close'].to_dict()
                        print(f"      Got {name}: {len(data)} days")
                except:
                    pass
    
    return index_data


def calculate_forward_returns(mood_df, index_data):
    """Calculate forward returns for each index."""
    print("\n3. Calculating forward returns...")
    
    results = []
    
    for _, row in mood_df.iterrows():
        base_date = pd.Timestamp(row['date']).normalize()
        
        result = {
            'date': base_date,
            'avg_trend_score': row['avg_trend_score'],
            'strong_momentum': row['strong_momentum'],
            'total_uptrends': row['total_uptrends'],
            'breakout_alerts': row['breakout_alerts'],
        }
        
        # For each index, calculate forward returns
        for index_name, prices in index_data.items():
            # Get base price
            base_price = None
            for offset in range(0, 5):
                check_date = base_date + timedelta(days=offset)
                if check_date in prices:
                    base_price = prices[check_date]
                    break
            
            if base_price is None:
                continue
            
            # Calculate 30, 60, 90, 180, 270 day returns
            for days in [30, 60, 90, 180, 270]:
                future_price = None
                for offset in range(days, days + 5):
                    check_date = base_date + timedelta(days=offset)
                    if check_date in prices:
                        future_price = prices[check_date]
                        break
                
                col_name = f"{index_name}_{days}d"
                if future_price:
                    result[col_name] = round(((future_price / base_price) - 1) * 100, 2)
                else:
                    result[col_name] = np.nan
        
        results.append(result)
    
    return pd.DataFrame(results)


def analyze_correlations(df, index_data):
    """Analyze correlations between mood metrics and each index."""
    print("\n4. Correlation Analysis by Index")
    print("-" * 80)
    
    metrics = ['avg_trend_score', 'strong_momentum', 'total_uptrends', 'breakout_alerts']
    horizons = [30, 60, 90, 180, 270]
    
    all_correlations = []
    
    for index_name in index_data.keys():
        print(f"\n  {index_name.upper()}")
        print(f"  {'Metric':<20} {'30D':>10} {'60D':>10} {'90D':>10} {'6MO':>10} {'9MO':>10}")
        print("  " + "-" * 70)
        
        for metric in metrics:
            row = {'index': index_name, 'metric': metric}
            line = f"  {metric:<20}"
            
            for days in horizons:
                col = f"{index_name}_{days}d"
                if col in df.columns:
                    valid_data = df[[metric, col]].dropna()
                    if len(valid_data) >= 10:  # Need at least 10 samples
                        corr = valid_data.corr().iloc[0, 1]
                        row[f'{days}d'] = round(corr, 3)
                        line += f" {corr:>+10.3f}"
                    else:
                        row[f'{days}d'] = None
                        line += f" {'N/A':>10}"
                else:
                    row[f'{days}d'] = None
                    line += f" {'N/A':>10}"
            
            all_correlations.append(row)
            print(line)
    
    return pd.DataFrame(all_correlations)


def find_best_indicator(corr_df):
    """Find the best indicator for each index/horizon combination."""
    print("\n\n5. BEST INDICATOR BY INDEX & HORIZON")
    print("-" * 80)
    
    # Pivot for easier analysis
    print("\n  Which metric best predicts each index?")
    print("\n  (More negative = better contrarian predictor)\n")
    
    indices = corr_df['index'].unique()
    metrics = corr_df['metric'].unique()
    horizons = ['30d', '60d', '90d']
    
    best_combinations = []
    
    for index in indices:
        idx_data = corr_df[corr_df['index'] == index]
        print(f"  {index}:")
        
        for horizon in horizons:
            if horizon in idx_data.columns:
                best_row = idx_data.loc[idx_data[horizon].idxmin()]
                corr_val = best_row[horizon]
                best_metric = best_row['metric']
                
                if not pd.isna(corr_val):
                    print(f"    {horizon}: {best_metric} ({corr_val:+.3f})")
                    best_combinations.append({
                        'index': index,
                        'horizon': horizon,
                        'best_metric': best_metric,
                        'correlation': corr_val
                    })
        print()
    
    return pd.DataFrame(best_combinations)


def score_bin_analysis(df, index_data):
    """Analyze returns by score bins for each index."""
    print("\n6. RETURNS BY SCORE BIN (90-Day Horizon)")
    print("-" * 80)
    
    bins = [0, 40, 55, 70, 100]
    labels = ['<40 (Bearish)', '40-55 (Weak)', '55-70 (Bullish)', '>70 (Strong)']
    
    df['score_bin'] = pd.cut(df['avg_trend_score'], bins=bins, labels=labels)
    
    for index_name in index_data.keys():
        col = f"{index_name}_90d"
        if col not in df.columns:
            continue
            
        print(f"\n  {index_name}")
        print(f"  {'Score Bin':<20} {'Avg 90D Ret':>15} {'Samples':>10}")
        print("  " + "-" * 45)
        
        for bin_label in labels:
            bin_data = df[df['score_bin'] == bin_label][col].dropna()
            if len(bin_data) > 0:
                avg_ret = bin_data.mean()
                count = len(bin_data)
                print(f"  {bin_label:<20} {avg_ret:>+14.2f}% {count:>10}")


def actionable_summary(corr_df, best_df):
    """Print actionable insights."""
    print("\n\n" + "="*80)
    print("  ACTIONABLE INSIGHTS")
    print("="*80)
    
    # Find the strongest overall correlations
    all_corrs = []
    for _, row in corr_df.iterrows():
        for h in ['30d', '60d', '90d']:
            if h in row and not pd.isna(row[h]):
                all_corrs.append({
                    'index': row['index'],
                    'metric': row['metric'],
                    'horizon': h,
                    'correlation': row[h]
                })
    
    if all_corrs:
        all_corrs_df = pd.DataFrame(all_corrs)
        strongest = all_corrs_df.loc[all_corrs_df['correlation'].idxmin()]
        
        print(f"\n  STRONGEST SIGNAL FOUND:")
        print(f"    Index: {strongest['index']}")
        print(f"    Metric: {strongest['metric']}")
        print(f"    Horizon: {strongest['horizon']}")
        print(f"    Correlation: {strongest['correlation']:+.3f}")
        
        print("\n  INTERPRETATION:")
        print(f"    When {strongest['metric']} is LOW, expect {strongest['index']} to")
        print(f"    outperform over the next {strongest['horizon']}.")
        
        print("\n  RECOMMENDED ACTION:")
        print("    1. Set alert for when Avg Trend Score drops below 40")
        print("    2. Use as signal to increase allocation to mid/small caps")
        print("    3. Best for 60-90 day investment horizon")


def main():
    # Load mood data
    mood_df = load_mood_data()
    
    # Fetch index data
    index_data = fetch_index_data(mood_df)
    
    if not index_data:
        print("\nERROR: Could not fetch any index data!")
        return
    
    # Calculate forward returns
    results_df = calculate_forward_returns(mood_df, index_data)
    
    # Correlation analysis
    corr_df = analyze_correlations(results_df, index_data)
    
    # Find best indicators
    best_df = find_best_indicator(corr_df)
    
    # Score bin analysis
    score_bin_analysis(results_df, index_data)
    
    # Actionable summary
    actionable_summary(corr_df, best_df)
    
    # Save results
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n\nDetailed results saved to: {OUTPUT_FILE}")
    print("="*80)


if __name__ == "__main__":
    main()
