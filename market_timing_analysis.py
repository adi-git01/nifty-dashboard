"""
Market Timing Analysis
======================
Analyzes whether market mood indicators can serve as leading indicators
for timing market entry/exit.

Key Analysis:
1. Forward return calculation at different horizons
2. Score-binned return analysis
3. Correlation between mood metrics and forward returns
4. What-if investment scenarios
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Constants
MOOD_FILE = "data/market_mood_history.csv"
OUTPUT_FILE = "market_timing_results.csv"

def load_data():
    """Load mood history and Nifty index data."""
    print("\n" + "="*60)
    print("  MARKET TIMING ANALYSIS")
    print("  Can mood indicators predict forward returns?")
    print("="*60)
    
    # Load mood history
    print("\n1. Loading market mood history...")
    mood_df = pd.read_csv(MOOD_FILE, parse_dates=['date'])
    print(f"   Found {len(mood_df)} days of mood data")
    print(f"   Date range: {mood_df['date'].min().date()} to {mood_df['date'].max().date()}")
    
    # Fetch Nifty 50 index data (proxy for market returns)
    print("\n2. Fetching Nifty 50 index data...")
    nifty = yf.Ticker("^NSEI")
    
    # Get data for the same period + extra for forward returns
    start_date = mood_df['date'].min() - timedelta(days=10)
    end_date = mood_df['date'].max() + timedelta(days=100)  # Extra for forward returns
    
    nifty_hist = nifty.history(start=start_date, end=end_date)
    
    if nifty_hist.empty:
        print("   ERROR: Could not fetch Nifty data!")
        return None, None
    
    print(f"   Got {len(nifty_hist)} days of Nifty data")
    
    # Make timezone-naive for merging
    nifty_hist.index = nifty_hist.index.tz_localize(None)
    nifty_hist.reset_index(inplace=True)
    nifty_hist.rename(columns={'Date': 'date', 'Close': 'nifty_close'}, inplace=True)
    
    return mood_df, nifty_hist


def calculate_forward_returns(mood_df, nifty_df):
    """Calculate forward returns for each date in mood history."""
    print("\n3. Calculating forward returns...")
    
    # Create lookup for Nifty prices by date
    nifty_prices = nifty_df.set_index('date')['nifty_close'].to_dict()
    
    # Convert mood dates to match nifty date format
    mood_df['date'] = pd.to_datetime(mood_df['date']).dt.normalize()
    
    results = []
    
    for _, row in mood_df.iterrows():
        base_date = row['date']
        
        # Find closest trading day price
        base_price = None
        for offset in range(0, 5):  # Look up to 5 days ahead for trading day
            check_date = base_date + timedelta(days=offset)
            if check_date in nifty_prices:
                base_price = nifty_prices[check_date]
                break
        
        if base_price is None:
            continue
        
        result = {
            'date': base_date,
            'avg_trend_score': row['avg_trend_score'],
            'strong_momentum': row['strong_momentum'],
            'total_uptrends': row['total_uptrends'],
            'breakout_alerts': row['breakout_alerts'],
            'nifty_price': base_price
        }
        
        # Calculate forward returns at different horizons (longer for leading indicator test)
        for days, label in [(21, '21d'), (30, '30d'), (60, '60d'), (90, '90d')]:
            future_price = None
            for offset in range(days, days + 5):  # Allow some slack for non-trading days
                check_date = base_date + timedelta(days=offset)
                if check_date in nifty_prices:
                    future_price = nifty_prices[check_date]
                    break
            
            if future_price:
                fwd_return = ((future_price / base_price) - 1) * 100
                result[f'fwd_return_{label}'] = round(fwd_return, 2)
            else:
                result[f'fwd_return_{label}'] = np.nan
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    print(f"   Calculated returns for {len(results_df)} dates")
    
    return results_df


def analyze_score_bins(df):
    """Analyze returns by score bins."""
    print("\n4. Analyzing returns by Avg Trend Score bins...")
    print("-" * 70)
    
    # Define score bins
    bins = [0, 35, 45, 55, 65, 100]
    labels = ['0-35 (Bearish)', '35-45 (Weak)', '45-55 (Neutral)', '55-65 (Bullish)', '65+ (Strong Bull)']
    
    df['score_bin'] = pd.cut(df['avg_trend_score'], bins=bins, labels=labels)
    
    # Group by bin and calculate stats
    bin_stats = df.groupby('score_bin', observed=True).agg({
        'fwd_return_21d': ['mean', 'median', 'std', 'count'],
        'fwd_return_30d': ['mean', 'median', 'std'],
        'fwd_return_60d': ['mean', 'median', 'std'],
        'fwd_return_90d': ['mean', 'median', 'std']
    }).round(2)
    
    print("\n  RETURNS BY AVG TREND SCORE BIN")
    print("  (What happens AFTER investing at each score level?)\n")
    
    # Pretty print
    print(f"  {'Score Range':<20} {'21D Ret%':>10} {'30D Ret%':>10} {'60D Ret%':>10} {'90D Ret%':>10} {'Samples':>8}")
    print("  " + "-" * 68)
    
    for bin_label in labels:
        if bin_label in bin_stats.index:
            row = bin_stats.loc[bin_label]
            ret_21d = row[('fwd_return_21d', 'mean')]
            ret_30d = row[('fwd_return_30d', 'mean')]
            ret_60d = row[('fwd_return_60d', 'mean')]
            ret_90d = row[('fwd_return_90d', 'mean')]
            count = int(row[('fwd_return_21d', 'count')])
            
            # Color coding (conceptual)
            print(f"  {bin_label:<20} {ret_21d:>+10.2f} {ret_30d:>+10.2f} {ret_60d:>+10.2f} {ret_90d:>+10.2f} {count:>8}")
    
    return bin_stats


def calculate_correlations(df):
    """Calculate correlations between mood metrics and forward returns."""
    print("\n\n5. Correlation Analysis")
    print("-" * 70)
    print("  (Positive = higher score -> higher future returns)")
    print("  (Negative = higher score -> lower future returns -> contrarian signal)\n")
    
    metrics = ['avg_trend_score', 'strong_momentum', 'total_uptrends', 'breakout_alerts']
    returns = ['fwd_return_21d', 'fwd_return_30d', 'fwd_return_60d', 'fwd_return_90d']
    
    print(f"  {'Metric':<20} {'vs 21D':>10} {'vs 30D':>10} {'vs 60D':>10} {'vs 90D':>10}")
    print("  " + "-" * 60)
    
    correlations = {}
    for metric in metrics:
        corr_row = {}
        for ret in returns:
            corr = df[[metric, ret]].dropna().corr().iloc[0, 1]
            corr_row[ret] = round(corr, 3)
        correlations[metric] = corr_row
        
        print(f"  {metric:<20} {corr_row['fwd_return_21d']:>+10.3f} {corr_row['fwd_return_30d']:>+10.3f} {corr_row['fwd_return_60d']:>+10.3f} {corr_row['fwd_return_90d']:>+10.3f}")
    
    return pd.DataFrame(correlations).T


def run_what_if_scenarios(df):
    """Simulate different investment strategies."""
    print("\n\n6. What-If Investment Scenarios")
    print("-" * 70)
    print("  Starting Capital: Rs 1,00,000")
    print("  Period: Full mood history\n")
    
    capital = 100000
    
    # Strategy 1: Buy when score < 40 (contrarian buy low)
    low_score_entries = df[df['avg_trend_score'] < 40].copy()
    if len(low_score_entries) > 0 and 'fwd_return_21d' in low_score_entries.columns:
        avg_return_low = low_score_entries['fwd_return_21d'].dropna().mean()
        final_low = capital * (1 + avg_return_low / 100)
        print(f"  STRATEGY 1: Buy when Score < 40 (Contrarian)")
        print(f"    Entry days found: {len(low_score_entries)}")
        print(f"    Avg 21-day return: {avg_return_low:+.2f}%")
        print(f"    Simulated single trade: Rs {capital:,.0f} -> Rs {final_low:,.0f}")
    else:
        print("  STRATEGY 1: Not enough data for score < 40")
    
    # Strategy 2: Buy when score > 60 (momentum following)
    high_score_entries = df[df['avg_trend_score'] > 60].copy()
    if len(high_score_entries) > 0 and 'fwd_return_21d' in high_score_entries.columns:
        avg_return_high = high_score_entries['fwd_return_21d'].dropna().mean()
        final_high = capital * (1 + avg_return_high / 100)
        print(f"\n  STRATEGY 2: Buy when Score > 60 (Momentum)")
        print(f"    Entry days found: {len(high_score_entries)}")
        print(f"    Avg 21-day return: {avg_return_high:+.2f}%")
        print(f"    Simulated single trade: Rs {capital:,.0f} -> Rs {final_high:,.0f}")
    else:
        print("\n  STRATEGY 2: Not enough data for score > 60")
    
    # Strategy 3: Buy and hold
    if len(df) > 0 and 'nifty_price' in df.columns:
        first_price = df.iloc[0]['nifty_price']
        last_price = df.iloc[-1]['nifty_price']
        bh_return = ((last_price / first_price) - 1) * 100
        final_bh = capital * (1 + bh_return / 100)
        print(f"\n  STRATEGY 3: Buy & Hold (Benchmark)")
        print(f"    Period return: {bh_return:+.2f}%")
        print(f"    Result: Rs {capital:,.0f} -> Rs {final_bh:,.0f}")
    
    # Strategy 4: Score-based position sizing
    print(f"\n  STRATEGY 4: Score-Based Position Sizing")
    print("    (Invest more when score is low, less when high)")
    
    total_invested = 0
    total_value = 0
    trades = 0
    
    for _, row in df.iterrows():
        score = row['avg_trend_score']
        fwd_ret = row.get('fwd_return_21d', np.nan)
        
        if pd.isna(fwd_ret):
            continue
        
        # Position size inversely proportional to score
        # Score 30 -> invest 100%, Score 70 -> invest 30%
        position_pct = max(0.3, 1 - (score - 30) / 60)
        investment = capital * position_pct
        
        result = investment * (1 + fwd_ret / 100)
        total_invested += investment
        total_value += result
        trades += 1
    
    if trades > 0:
        overall_return = ((total_value / total_invested) - 1) * 100
        print(f"    Trades simulated: {trades}")
        print(f"    Avg position-weighted return: {overall_return:+.2f}%")


def identify_leading_indicator_potential(df):
    """Identify if score changes lead price changes."""
    print("\n\n7. Leading Indicator Analysis")
    print("-" * 70)
    
    # Calculate score change (momentum of the indicator)
    df['score_change'] = df['avg_trend_score'].diff()
    
    # When score is rising from low levels - is this predictive?
    rising_from_low = df[(df['avg_trend_score'] < 50) & (df['score_change'] > 3)]
    if len(rising_from_low) > 0:
        avg_ret = rising_from_low['fwd_return_21d'].dropna().mean()
        print(f"\n  SIGNAL: Score rising from below 50 (momentum turning)")
        print(f"    Occurrences: {len(rising_from_low)}")
        print(f"    Avg 21-day forward return: {avg_ret:+.2f}%")
    
    # When score is falling from high levels
    falling_from_high = df[(df['avg_trend_score'] > 60) & (df['score_change'] < -3)]
    if len(falling_from_high) > 0:
        avg_ret = falling_from_high['fwd_return_21d'].dropna().mean()
        print(f"\n  SIGNAL: Score falling from above 60 (momentum fading)")
        print(f"    Occurrences: {len(falling_from_high)}")
        print(f"    Avg 21-day forward return: {avg_ret:+.2f}%")
    
    # Extreme readings
    very_low = df[df['avg_trend_score'] < 35]
    very_high = df[df['avg_trend_score'] > 70]
    
    if len(very_low) > 0:
        avg_ret = very_low['fwd_return_21d'].dropna().mean()
        print(f"\n  EXTREME LOW (Score < 35):")
        print(f"    Occurrences: {len(very_low)}")
        print(f"    Avg 21-day forward return: {avg_ret:+.2f}%")
    
    if len(very_high) > 0:
        avg_ret = very_high['fwd_return_21d'].dropna().mean()
        print(f"\n  EXTREME HIGH (Score > 70):")
        print(f"    Occurrences: {len(very_high)}")
        print(f"    Avg 21-day forward return: {avg_ret:+.2f}%")


def main():
    """Run the full analysis."""
    # Load data
    mood_df, nifty_df = load_data()
    if mood_df is None:
        return
    
    # Calculate forward returns
    results_df = calculate_forward_returns(mood_df, nifty_df)
    
    # Score-binned analysis
    bin_stats = analyze_score_bins(results_df)
    
    # Correlations
    corr_df = calculate_correlations(results_df)
    
    # What-if scenarios
    run_what_if_scenarios(results_df)
    
    # Leading indicator analysis
    identify_leading_indicator_potential(results_df)
    
    # Save results
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n\nDetailed results saved to: {OUTPUT_FILE}")
    
    # Summary
    print("\n" + "="*60)
    print("  SUMMARY & KEY FINDINGS")
    print("="*60)
    
    # Check if contrarian or momentum works better
    low_score_ret = results_df[results_df['avg_trend_score'] < 40]['fwd_return_21d'].dropna().mean()
    high_score_ret = results_df[results_df['avg_trend_score'] > 60]['fwd_return_21d'].dropna().mean()
    
    print("\n  Key Question: Is Score a CONTRARIAN or MOMENTUM signal?")
    print(f"    Low Score (<40) -> 21D Avg Return: {low_score_ret:+.2f}%" if not pd.isna(low_score_ret) else "    Low Score: Insufficient data")
    print(f"    High Score (>60) -> 21D Avg Return: {high_score_ret:+.2f}%" if not pd.isna(high_score_ret) else "    High Score: Insufficient data")
    
    if not pd.isna(low_score_ret) and not pd.isna(high_score_ret):
        if low_score_ret > high_score_ret:
            print("\n  => CONTRARIAN signal detected! Buy when mood is LOW.")
        else:
            print("\n  => MOMENTUM signal detected! Buy when mood is HIGH.")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
