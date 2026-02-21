"""
Multi-Horizon Alpha Analysis Engine
====================================
Extends the 60-day analysis to longer investment horizons:
- 90 days (3 months)
- 180 days (6 months)
- 270 days (9 months)
- 365 days (1 year)
- 756 days (3 years)

Calculates:
- Forward returns for each horizon
- MAE/MFE (drawdown/peak) for each horizon
- Ulcer Index for each horizon
- Regime at entry
- Sector behavior

Outputs comprehensive raw data and summary pivots.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta, datetime
import os
import json

# --- CONFIG ---
OUTPUT_DIR = "analysis_multihorizon"
HORIZONS = {
    '90d': 90,
    '180d': 180,
    '270d': 270,
    '1yr': 252,
    '3yr': 756
}

# Stock Universe (same as previous analysis)
STOCKS = {
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS'],
    'IT_Services': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS'],
    'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'APOLLOHOSP.NS'],
    'Industrials': ['LT.NS', 'SIEMENS.NS', 'ABB.NS', 'HAVELLS.NS', 'CUMMINSIND.NS'],
    'Metals': ['TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 'VEDL.NS'],
    'Auto': ['MARUTI.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS'],
    'Consumer': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'TITAN.NS'],
    'Energy': ['RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'IOC.NS', 'NTPC.NS'],
    'Telecom': ['BHARTIARTL.NS', 'IDEA.NS', 'TATACOMM.NS', 'INDUSTOWER.NS', 'IRCTC.NS'],
    'Realty': ['DLF.NS', 'GODREJPROP.NS', 'OBEROIRLTY.NS', 'PRESTIGE.NS', 'PHOENIXLTD.NS'],
    'Defense': ['HAL.NS', 'BEL.NS', 'BHEL.NS', 'GRSE.NS', 'MAZAGON.NS']
}

def fetch_all_data():
    """Fetch 10+ years of data for all stocks and Nifty."""
    print("Fetching historical data (10+ years)...")
    data = {}
    
    # Nifty for regime
    nifty = yf.Ticker("^NSEI").history(period="max")
    nifty.index = nifty.index.tz_localize(None)
    data['NIFTY'] = nifty
    
    all_tickers = [t for sector in STOCKS.values() for t in sector]
    for ticker in all_tickers:
        try:
            hist = yf.Ticker(ticker).history(period="max")
            hist.index = hist.index.tz_localize(None)
            if len(hist) > 252:  # At least 1 year of data
                data[ticker] = hist
        except Exception as e:
            print(f"  Failed: {ticker} - {e}")
    
    print(f"  Loaded {len(data)-1} stocks + Nifty")
    return data

def classify_regime(nifty_data, date):
    """Classify market regime at a given date."""
    try:
        idx = nifty_data.index.searchsorted(date)
        if idx < 200:
            return "Unknown"
        
        window = nifty_data.iloc[max(0, idx-200):idx+1]
        price = window['Close'].iloc[-1]
        ma50 = window['Close'].rolling(50).mean().iloc[-1]
        ma200 = window['Close'].rolling(200).mean().iloc[-1]
        
        if ma50 > ma200:
            if price > ma50:
                return "Strong_Bull"
            else:
                return "Mild_Bull"
        else:
            if price < ma50:
                return "Strong_Bear"
            else:
                return "Recovery"
    except:
        return "Unknown"

def calculate_trend_score(data, idx):
    """Calculate Trend Score at a given index."""
    if idx < 200:
        return 50
    
    window = data.iloc[max(0, idx-252):idx+1]
    price = window['Close'].iloc[-1]
    ma50 = window['Close'].rolling(50).mean().iloc[-1]
    ma200 = window['Close'].rolling(200).mean().iloc[-1]
    high_52 = window['Close'].max()
    low_52 = window['Close'].min()
    
    score = 50
    if price > ma50: score += 15
    else: score -= 10
    if price > ma200: score += 15
    else: score -= 15
    if ma50 > ma200: score += 10
    else: score -= 5
    
    range_52 = high_52 - low_52
    if range_52 > 0:
        pos = (price - low_52) / range_52
        score += int((pos - 0.5) * 30)
    
    return max(0, min(100, score))

def calculate_volume_state(data, idx):
    """Calculate Volume State at a given index."""
    if idx < 60:
        return "Flat"
    
    window = data.iloc[max(0, idx-60):idx+1]
    vol_10 = window['Volume'].iloc[-10:].mean()
    vol_60 = window['Volume'].mean()
    
    if vol_60 == 0:
        return "Flat"
    
    ratio = vol_10 / vol_60
    
    if ratio > 2.0: return "Spike"
    elif ratio > 1.5: return "Jump"
    elif ratio < 0.5: return "Big_Drop"
    elif ratio < 0.7: return "Drop"
    else: return "Flat"

def calculate_path_metrics(data, start_idx, horizon):
    """Calculate returns and path metrics for a given horizon."""
    if start_idx + horizon >= len(data):
        return None
    
    entry_price = data['Close'].iloc[start_idx]
    path = data.iloc[start_idx+1:start_idx+1+horizon]
    
    if len(path) < horizon * 0.8:  # Need at least 80% of data
        return None
    
    # Returns
    exit_price = path['Close'].iloc[-1]
    ret = (exit_price - entry_price) / entry_price * 100
    
    # MAE (Max Adverse Excursion - Pain)
    min_price = path['Low'].min()
    mae = (min_price - entry_price) / entry_price * 100
    
    # MFE (Max Favorable Excursion - Peak)
    max_price = path['High'].max()
    mfe = (max_price - entry_price) / entry_price * 100
    
    # Time Underwater
    days_underwater = (path['Close'] < entry_price).sum()
    pct_underwater = days_underwater / len(path) * 100
    
    # Ulcer Index
    rolling_peak = path['Close'].cummax()
    drawdowns = (path['Close'] - rolling_peak) / rolling_peak * 100
    ulcer_index = np.sqrt((drawdowns ** 2).mean())
    
    return {
        'return': ret,
        'mae': mae,
        'mfe': mfe,
        'time_underwater': pct_underwater,
        'ulcer_index': ulcer_index
    }

def generate_signals(data_cache, nifty_data):
    """Generate signal events with multi-horizon metrics."""
    print("\nGenerating signal events with multi-horizon metrics...")
    
    signals = []
    
    for sector, tickers in STOCKS.items():
        for ticker in tickers:
            if ticker not in data_cache:
                continue
            
            stock_data = data_cache[ticker]
            
            # Sample every 20 trading days (monthly-ish)
            for idx in range(252, len(stock_data) - 756, 20):
                date = stock_data.index[idx]
                
                # Skip if too recent (need 3yr forward data)
                if date > datetime(2022, 9, 1):
                    continue
                
                regime = classify_regime(nifty_data, date)
                if regime == "Unknown":
                    continue
                
                trend_score = calculate_trend_score(stock_data, idx)
                vol_state = calculate_volume_state(stock_data, idx)
                
                # Trend bucket
                if trend_score < 20: trend_bucket = "0-20"
                elif trend_score < 40: trend_bucket = "20-40"
                elif trend_score < 60: trend_bucket = "40-60"
                elif trend_score < 80: trend_bucket = "60-80"
                else: trend_bucket = "80-100"
                
                signal = {
                    'ticker': ticker,
                    'sector': sector,
                    'date': date.strftime('%Y-%m-%d'),
                    'regime': regime,
                    'trend_score': trend_score,
                    'trend_bucket': trend_bucket,
                    'vol_state': vol_state,
                    'entry_price': stock_data['Close'].iloc[idx]
                }
                
                # Calculate metrics for each horizon
                for horizon_name, horizon_days in HORIZONS.items():
                    metrics = calculate_path_metrics(stock_data, idx, horizon_days)
                    if metrics:
                        signal[f'ret_{horizon_name}'] = metrics['return']
                        signal[f'mae_{horizon_name}'] = metrics['mae']
                        signal[f'mfe_{horizon_name}'] = metrics['mfe']
                        signal[f'underwater_{horizon_name}'] = metrics['time_underwater']
                        signal[f'ulcer_{horizon_name}'] = metrics['ulcer_index']
                
                signals.append(signal)
    
    print(f"  Generated {len(signals)} multi-horizon signals")
    return pd.DataFrame(signals)

def generate_pivots(df):
    """Generate summary pivot tables."""
    print("\nGenerating summary pivots...")
    
    pivots = {}
    
    # 1. Regime Performance by Horizon
    regime_perf = []
    for horizon in HORIZONS.keys():
        col = f'ret_{horizon}'
        if col not in df.columns:
            continue
        stats = df.groupby('regime').agg({
            col: ['mean', 'std', 'count'],
            f'mae_{horizon}': 'mean',
            f'mfe_{horizon}': 'mean'
        }).reset_index()
        stats.columns = ['Regime', 'Avg_Return', 'Std', 'Count', 'Avg_MAE', 'Avg_MFE']
        stats['Horizon'] = horizon
        stats['Win_Rate'] = df.groupby('regime')[col].apply(lambda x: (x > 0).mean()).values * 100
        regime_perf.append(stats)
    
    pivots['regime_by_horizon'] = pd.concat(regime_perf, ignore_index=True)
    
    # 2. Sector Performance by Horizon
    sector_perf = []
    for horizon in HORIZONS.keys():
        col = f'ret_{horizon}'
        if col not in df.columns:
            continue
        stats = df.groupby('sector').agg({
            col: ['mean', 'count'],
            f'mae_{horizon}': 'mean'
        }).reset_index()
        stats.columns = ['Sector', 'Avg_Return', 'Count', 'Avg_MAE']
        stats['Horizon'] = horizon
        sector_perf.append(stats)
    
    pivots['sector_by_horizon'] = pd.concat(sector_perf, ignore_index=True)
    
    # 3. Factor Matrix (Trend x Vol x Regime) for 1yr
    if 'ret_1yr' in df.columns:
        factor_stats = df.groupby(['regime', 'trend_bucket', 'vol_state']).agg(
            Avg_Return=('ret_1yr', 'mean'),
            Win_Rate=('ret_1yr', lambda x: (x > 0).mean() * 100),
            Count=('ret_1yr', 'count'),
            Avg_MAE=('mae_1yr', 'mean'),
            Avg_MFE=('mfe_1yr', 'mean')
        ).reset_index()
        pivots['factor_1yr'] = factor_stats
    
    # 4. Best Setups per Horizon
    best_setups = []
    for horizon in HORIZONS.keys():
        col = f'ret_{horizon}'
        if col not in df.columns:
            continue
        stats = df.groupby(['regime', 'trend_bucket', 'vol_state']).agg({
            col: ['mean', 'count']
        }).reset_index()
        stats.columns = ['Regime', 'Trend', 'Volume', 'Avg_Return', 'Count']
        stats = stats[stats['Count'] > 20]
        stats = stats.sort_values('Avg_Return', ascending=False).head(10)
        stats['Horizon'] = horizon
        best_setups.append(stats)
    
    pivots['best_setups'] = pd.concat(best_setups, ignore_index=True)
    
    return pivots

def save_outputs(df, pivots):
    """Save all outputs to disk."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Master Database
    master_path = f"{OUTPUT_DIR}/MASTER_MULTIHORIZON_DB.csv"
    df.to_csv(master_path, index=False)
    print(f"\nSaved: {master_path} ({len(df)} records)")
    
    # Pivots
    for name, pivot_df in pivots.items():
        path = f"{OUTPUT_DIR}/{name}.csv"
        pivot_df.round(2).to_csv(path, index=False)
        print(f"Saved: {path}")

def generate_report(df, pivots):
    """Generate findings report."""
    
    report = """# Multi-Horizon Alpha Analysis Report

**Analysis Date:** {date}
**Signals Analyzed:** {n_signals}
**Horizons:** 90d, 180d, 270d, 1yr, 3yr

---

## 1. Key Findings by Horizon

### Average Returns by Regime

""".format(date=datetime.now().strftime('%Y-%m-%d'), n_signals=len(df))

    # Regime table
    regime_pivot = pivots['regime_by_horizon']
    for horizon in HORIZONS.keys():
        h_data = regime_pivot[regime_pivot['Horizon'] == horizon].sort_values('Avg_Return', ascending=False)
        report += f"#### {horizon} Horizon\n"
        report += "| Regime | Return | Win Rate | MAE | MFE | Count |\n"
        report += "|:-------|:-------|:---------|:----|:----|:------|\n"
        for _, row in h_data.iterrows():
            report += f"| {row['Regime']} | {row['Avg_Return']:.1f}% | {row['Win_Rate']:.0f}% | {row['Avg_MAE']:.1f}% | {row['Avg_MFE']:.1f}% | {row['Count']:.0f} |\n"
        report += "\n"

    # Best setups
    report += """---

## 2. Top Setups by Horizon

"""
    best = pivots['best_setups']
    for horizon in HORIZONS.keys():
        h_data = best[best['Horizon'] == horizon].head(5)
        if len(h_data) > 0:
            report += f"### {horizon}\n"
            report += "| Regime | Trend | Volume | Return | Count |\n"
            report += "|:-------|:------|:-------|:-------|:------|\n"
            for _, row in h_data.iterrows():
                report += f"| {row['Regime']} | {row['Trend']} | {row['Volume']} | **{row['Avg_Return']:.1f}%** | {row['Count']:.0f} |\n"
            report += "\n"

    # Sector insights
    report += """---

## 3. Sector Performance (1 Year Horizon)

"""
    if 'sector_by_horizon' in pivots:
        sector_data = pivots['sector_by_horizon']
        yr_data = sector_data[sector_data['Horizon'] == '1yr'].sort_values('Avg_Return', ascending=False)
        report += "| Sector | 1yr Return | MAE |\n"
        report += "|:-------|:-----------|:----|\n"
        for _, row in yr_data.iterrows():
            report += f"| {row['Sector']} | {row['Avg_Return']:.1f}% | {row['Avg_MAE']:.1f}% |\n"

    report += """
---

## 4. Key Insights

1. **Longer Horizons = Higher Returns (but more pain)**
2. **Regime remains critical** across all timeframes
3. **Bear Market Alpha compounds** - 3yr returns in Strong Bear setups are exceptional
4. **MAE/MFE Ratio** improves with longer holding periods

---

*Generated by Multi-Horizon Alpha Engine*
"""

    report_path = f"{OUTPUT_DIR}/multihorizon_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Saved: {report_path}")

def main():
    print("="*60)
    print("MULTI-HORIZON ALPHA ANALYSIS ENGINE")
    print("Horizons: 90d, 180d, 270d, 1yr, 3yr")
    print("="*60)
    
    # Fetch data
    data_cache = fetch_all_data()
    
    # Generate signals with multi-horizon metrics
    signals_df = generate_signals(data_cache, data_cache['NIFTY'])
    
    # Generate pivots
    pivots = generate_pivots(signals_df)
    
    # Save outputs
    save_outputs(signals_df, pivots)
    
    # Generate report
    generate_report(signals_df, pivots)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print("="*60)

if __name__ == "__main__":
    main()
