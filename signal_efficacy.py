"""
Signal Efficacy Analysis (Predictive Power Test)
-----------------------------------------------
Randomly samples 30 stocks from Nifty 500 (stratified by Sector).
Analyzes historical performance of:
1. Trend Score > 70 (Breakout)
2. Momentum Score > 8 (High Momentum)
3. Volume Score > 8 (High Volume/Accumulation)
4. Golden Cross (50 > 200 EMA)
5. Bullish Divergence

Calculates Forward Returns for 10, 20, 30, 45, 60, 90 days.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import random
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.nifty500_list import TICKERS, SECTOR_MAP
from utils.score_history import calculate_historical_scores, detect_divergences

def get_stratified_sample(n=30):
    """Selects n stocks ensuring sector diversity."""
    # Group by sector
    sector_groups = {}
    for t in TICKERS:
        s = SECTOR_MAP.get(t, "Unknown")
        if s not in sector_groups:
            sector_groups[s] = []
        sector_groups[s].append(t)
    
    selected = []
    sectors = list(sector_groups.keys())
    
    # Shuffle for randomness
    random.shuffle(sectors)
    for k in sector_groups:
        random.shuffle(sector_groups[k])
    
    # Round robin selection
    while len(selected) < n and any(sector_groups.values()):
        for s in sectors:
            if len(selected) >= n: break
            if sector_groups[s]:
                stock = sector_groups[s].pop(0)
                selected.append(stock)
    
    return selected

def analyze_signals(ticker, history_df):
    """Detects signals and calculates forward returns."""
    if history_df.empty or len(history_df) < 200:
        return []
        
    # Calculate Scores
    scores = calculate_historical_scores(history_df)
    
    # Detect Divergences
    divs = detect_divergences(scores)
    # Join on index (Date)
    scores = scores.join(divs[['div_bull', 'div_bear']])
    
    signals = []
    
    # Pre-calculate forward returns
    # Horizons: 10, 20, 30, 45, 60, 90 days
    # Note: These are trading days (approx 2 weeks, 4 weeks, etc.)
    horizons = [10, 20, 30, 45, 60, 90]
    for days in horizons:
        scores[f'ret_{days}d'] = scores['Close'].pct_change(days).shift(-days) * 100

    # Define Signal Conditions (Entry based)
    
    # 1. Trend Return to Strength (>70)
    # Signal when it crosses above 70
    trend_entry = (scores['trend_score_hist'] >= 70) & (scores['trend_score_hist'].shift(1) < 70)
    
    # 2. High Momentum Entry (>8)
    mom_entry = (scores['momentum_score_hist'] >= 8) & (scores['momentum_score_hist'].shift(1) < 8)
    
    # 3. High Volume Entry (>8)
    vol_entry = (scores['volume_score_hist'] >= 8) & (scores['volume_score_hist'].shift(1) < 8)
    
    # 4. Golden Cross
    if 'ma50' in scores.columns and 'ma200' in scores.columns:
        golden_cross = (scores['ma50'] > scores['ma200']) & (scores['ma50'].shift(1) <= scores['ma200'].shift(1))
    else:
        golden_cross = pd.Series(False, index=scores.index)
        
    # 5. Bullish Divergence
    # div_bull contains Price at divergence point, or NaN
    bull_div = scores['div_bull'].notna()
    
    # Combine
    event_types = {
        'Trend Breakout (>70)': trend_entry,
        'High Momentum (>8)': mom_entry,
        'Volume Spike (>8)': vol_entry,
        'Golden Cross': golden_cross,
        'Bullish Divergence': bull_div
    }
    
    for name, series in event_types.items():
        # Get dates where signal is True
        events = scores[series]
        
        for date, row in events.iterrows():
            # Skip if forward returns are NaN (end of data)
            if pd.isna(row.get('ret_10d')):
                continue
                
            signal_data = {
                'Ticker': ticker,
                'Date': date.strftime('%Y-%m-%d'),
                'Signal': name,
                'Close_Price': row['Close']
            }
            # Add returns
            for days in horizons:
                signal_data[f'{days}d_Ret'] = row.get(f'ret_{days}d')
            
            signals.append(signal_data)
            
    return signals

def main():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting Signal Efficacy Analysis...")
    print("-" * 60)
    
    print("Sampling 30 stocks (Secor Stratified)...")
    sample = get_stratified_sample(30)
    print(f"Selected: {sample[:5]} ... and {len(sample)-5} more")
    
    print("\nFetching 2-Year Historical Data...")
    try:
        # Download in bulk for speed
        data = yf.download(sample, period="2y", group_by='ticker', progress=True, auto_adjust=True, threads=True) 
    except Exception as e:
        print(f"Error downloading data: {e}")
        return

    all_signals = []
    
    for i, ticker in enumerate(sample):
        try:
            # Handle MultiIndex if >1 ticker, else standardized
            if len(sample) > 1:
                if ticker not in data.columns.get_level_values(0):
                    print(f"No data for {ticker}")
                    continue
                df = data[ticker].dropna()
            else:
                df = data.dropna()
                
            if df.empty: continue
            
            # Progress indicator
            print(f"[{i+1}/{len(sample)}] Analyzing {ticker}...", end='\r')
            
            sigs = analyze_signals(ticker, df)
            all_signals.extend(sigs)
            
        except Exception as e:
            print(f"\nError analyzing {ticker}: {e}")
            
    print(f"\n\nAnalysis Complete. Found {len(all_signals)} total signal events.")
    
    # Compile Report
    res_df = pd.DataFrame(all_signals)
    if res_df.empty:
        print("No signals found to analyze.")
        return
        
    # Aggegration
    # 1. Win Rate (>0% Return)
    win_rates = res_df.groupby('Signal')[[c for c in res_df.columns if 'd_Ret' in c]].apply(lambda x: (x>0).mean() * 100)
    
    # 2. Average Return
    avg_returns = res_df.groupby('Signal')[[c for c in res_df.columns if 'd_Ret' in c]].mean()
    
    # 3. Counts
    counts = res_df['Signal'].value_counts()
    
    print("\n" + "="*80)
    print("WIN RATE (%) - Likelihood of Positive Return")
    print("="*80)
    print(win_rates.round(1))
    
    print("\n" + "="*80)
    print("AVERAGE RETURN (%) - Expected Magnitude")
    print("="*80)
    print(avg_returns.round(1))
    
    print("\n" + "="*80)
    print("SIGNAL FREQUENCY")
    print("="*80)
    print(counts)
    
    # Save results
    res_df.to_csv("signal_efficacy_raw.csv", index=False)
    win_rates.to_csv("signal_efficacy_winrates.csv")
    avg_returns.to_csv("signal_efficacy_avg_returns.csv")
    print("\nResults saved to CSV files.")

if __name__ == "__main__":
    main()
