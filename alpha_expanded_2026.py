"""
Expanded Multi-Horizon Alpha Analysis v2.0
==========================================
- Extended to 100+ stocks
- Data through February 2026
- Compare 2008-2022 vs 2022-2026 periods

Goal: Validate if old assumptions still hold in the new market regime.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

# --- CONFIG ---
OUTPUT_DIR = "analysis_2026"
HORIZONS = {
    '90d': 90,
    '180d': 180,
    '1yr': 252,
}

# Expanded Stock Universe (100+ stocks)
STOCKS = {
    'Banking': [
        'HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS',
        'INDUSINDBK.NS', 'BANDHANBNK.NS', 'FEDERALBNK.NS', 'IDFCFIRSTB.NS', 'PNB.NS'
    ],
    'IT_Services': [
        'TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS',
        'LTIM.NS', 'MPHASIS.NS', 'COFORGE.NS', 'PERSISTENT.NS', 'LTTS.NS'
    ],
    'Pharma': [
        'SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'APOLLOHOSP.NS',
        'BIOCON.NS', 'LUPIN.NS', 'AUROPHARMA.NS', 'ALKEM.NS', 'TORNTPHARM.NS'
    ],
    'Industrials': [
        'LT.NS', 'SIEMENS.NS', 'ABB.NS', 'HAVELLS.NS', 'CUMMINSIND.NS',
        'BHARATFORG.NS', 'THERMAX.NS', 'VOLTAS.NS', 'CROMPTON.NS', 'POLYCAB.NS'
    ],
    'Metals': [
        'TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 'VEDL.NS',
        'NMDC.NS', 'JINDALSTEL.NS', 'NATIONALUM.NS', 'SAIL.NS', 'HINDZINC.NS'
    ],
    'Auto': [
        'MARUTI.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS',
        'TATAMOTORS.NS', 'ASHOKLEY.NS', 'TVSMOTOR.NS', 'BALKRISIND.NS', 'MRF.NS'
    ],
    'Consumer': [
        'HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'TITAN.NS',
        'DABUR.NS', 'MARICO.NS', 'GODREJCP.NS', 'COLPAL.NS', 'TRENT.NS'
    ],
    'Energy': [
        'RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'IOC.NS', 'NTPC.NS',
        'POWERGRID.NS', 'ADANIGREEN.NS', 'TATAPOWER.NS', 'GAIL.NS', 'PETRONET.NS'
    ],
    'Telecom': [
        'BHARTIARTL.NS', 'TATACOMM.NS', 'INDUSTOWER.NS', 'IRCTC.NS', 'ZOMATO.NS'
    ],
    'Realty': [
        'DLF.NS', 'GODREJPROP.NS', 'OBEROIRLTY.NS', 'PRESTIGE.NS', 'PHOENIXLTD.NS',
        'BRIGADE.NS', 'SOBHA.NS', 'LODHA.NS'
    ],
    'Defense': [
        'HAL.NS', 'BEL.NS', 'BHEL.NS'
    ],
    'Financials_NBFC': [
        'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'CHOLAFIN.NS', 'MUTHOOTFIN.NS', 'SHRIRAMFIN.NS',
        'M&MFIN.NS', 'LICHSGFIN.NS', 'PFC.NS', 'RECLTD.NS', 'SBICARD.NS'
    ]
}

def fetch_all_data():
    """Fetch full history for all stocks."""
    print("Fetching data for 100+ stocks (through 2026)...")
    data = {}
    
    # Nifty for regime
    nifty = yf.Ticker("^NSEI").history(period="max")
    if not nifty.empty:
        nifty.index = nifty.index.tz_localize(None)
        data['NIFTY'] = nifty
        print(f"  Nifty: {nifty.index.min().strftime('%Y-%m-%d')} to {nifty.index.max().strftime('%Y-%m-%d')}")
    
    all_tickers = [t for sector in STOCKS.values() for t in sector]
    success = 0
    failed = []
    
    for ticker in all_tickers:
        try:
            hist = yf.Ticker(ticker).history(period="max")
            if not hist.empty and len(hist) > 252:
                hist.index = hist.index.tz_localize(None)
                data[ticker] = hist
                success += 1
        except Exception as e:
            failed.append(ticker)
    
    print(f"  Successfully loaded: {success} stocks")
    if failed:
        print(f"  Failed: {len(failed)} stocks")
    
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
            return "Strong_Bull" if price > ma50 else "Mild_Bull"
        else:
            return "Strong_Bear" if price < ma50 else "Recovery"
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

def calculate_metrics(data, start_idx, horizon):
    """Calculate returns and path metrics."""
    if start_idx + horizon >= len(data):
        return None
    
    entry_price = data['Close'].iloc[start_idx]
    path = data.iloc[start_idx+1:start_idx+1+horizon]
    
    if len(path) < horizon * 0.8:
        return None
    
    exit_price = path['Close'].iloc[-1]
    ret = (exit_price - entry_price) / entry_price * 100
    
    min_price = path['Low'].min()
    mae = (min_price - entry_price) / entry_price * 100
    
    max_price = path['High'].max()
    mfe = (max_price - entry_price) / entry_price * 100
    
    days_underwater = (path['Close'] < entry_price).sum()
    pct_underwater = days_underwater / len(path) * 100
    
    return {
        'return': ret,
        'mae': mae,
        'mfe': mfe,
        'underwater': pct_underwater
    }

def generate_signals(data_cache, nifty_data):
    """Generate signals through 2026."""
    print("\nGenerating signals (extending to 2026)...")
    
    signals = []
    
    for sector, tickers in STOCKS.items():
        for ticker in tickers:
            if ticker not in data_cache:
                continue
            
            stock_data = data_cache[ticker]
            
            # Sample every 20 trading days
            for idx in range(252, len(stock_data) - 252, 20):
                date = stock_data.index[idx]
                
                regime = classify_regime(nifty_data, date)
                if regime == "Unknown":
                    continue
                
                trend_score = calculate_trend_score(stock_data, idx)
                vol_state = calculate_volume_state(stock_data, idx)
                
                # Determine if this is "new era" (2022+) or "old era"
                era = "2022_2026" if date >= datetime(2022, 1, 1) else "2008_2022"
                
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
                    'era': era 
                }
                
                # Calculate metrics for each horizon
                for horizon_name, horizon_days in HORIZONS.items():
                    metrics = calculate_metrics(stock_data, idx, horizon_days)
                    if metrics:
                        signal[f'ret_{horizon_name}'] = metrics['return']
                        signal[f'mae_{horizon_name}'] = metrics['mae']
                        signal[f'mfe_{horizon_name}'] = metrics['mfe']
                        signal[f'underwater_{horizon_name}'] = metrics['underwater']
                
                signals.append(signal)
    
    print(f"  Generated {len(signals)} signals")
    return pd.DataFrame(signals)

def compare_eras(df):
    """Compare 2008-2022 vs 2022-2026."""
    print("\nComparing Eras...")
    
    comparison = []
    
    for era in ['2008_2022', '2022_2026']:
        era_df = df[df['era'] == era]
        
        if 'ret_90d' not in era_df.columns or era_df.empty:
            continue
            
        for regime in era_df['regime'].unique():
            regime_df = era_df[era_df['regime'] == regime]
            if len(regime_df) < 20:
                continue
                
            comparison.append({
                'Era': era,
                'Regime': regime,
                'Count': len(regime_df),
                'Ret_90d': regime_df['ret_90d'].mean(),
                'Ret_180d': regime_df['ret_180d'].mean() if 'ret_180d' in regime_df else 0,
                'Ret_1yr': regime_df['ret_1yr'].mean() if 'ret_1yr' in regime_df else 0,
                'MAE_90d': regime_df['mae_90d'].mean(),
                'Win_Rate_90d': (regime_df['ret_90d'] > 0).mean() * 100
            })
    
    return pd.DataFrame(comparison)

def save_outputs(df, comparison_df):
    """Save all outputs."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Master DB
    df.to_csv(f"{OUTPUT_DIR}/MASTER_2026_DB.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR}/MASTER_2026_DB.csv ({len(df)} records)")
    
    # Era Comparison
    comparison_df.round(2).to_csv(f"{OUTPUT_DIR}/era_comparison.csv", index=False)
    print(f"Saved: {OUTPUT_DIR}/era_comparison.csv")
    
    # Regime Performance
    regime_stats = df.groupby(['regime', 'era']).agg({
        'ret_90d': ['mean', 'count'],
        'mae_90d': 'mean'
    }).reset_index()
    regime_stats.columns = ['Regime', 'Era', 'Ret_90d', 'Count', 'MAE_90d']
    regime_stats.to_csv(f"{OUTPUT_DIR}/regime_by_era.csv", index=False)
    print(f"Saved: {OUTPUT_DIR}/regime_by_era.csv")

def generate_report(df, comparison_df):
    """Generate comparison report."""
    
    old_era = comparison_df[comparison_df['Era'] == '2008_2022']
    new_era = comparison_df[comparison_df['Era'] == '2022_2026']
    
    report = f"""# Analysis Update: 2008-2026 (Expanded)

**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Stocks Analyzed:** {df['ticker'].nunique()}
**Sectors:** {df['sector'].nunique()}
**Total Signals:** {len(df):,}

---

## 📊 Data Coverage

| Era | Signals | Date Range |
|:----|:--------|:-----------|
| 2008-2022 | {len(df[df['era']=='2008_2022']):,} | Old Analysis |
| 2022-2026 | {len(df[df['era']=='2022_2026']):,} | **NEW DATA** |

---

## 🔄 Era Comparison: Did Assumptions Change?

### Returns by Regime (90 Day)

| Regime | 2008-2022 | 2022-2026 | Change |
|:-------|:----------|:----------|:-------|
"""
    
    for regime in ['Strong_Bull', 'Mild_Bull', 'Strong_Bear', 'Recovery']:
        old_ret = old_era[old_era['Regime'] == regime]['Ret_90d'].values
        new_ret = new_era[new_era['Regime'] == regime]['Ret_90d'].values
        
        old_val = f"{old_ret[0]:.1f}%" if len(old_ret) > 0 else "N/A"
        new_val = f"{new_ret[0]:.1f}%" if len(new_ret) > 0 else "N/A"
        
        if len(old_ret) > 0 and len(new_ret) > 0:
            change = new_ret[0] - old_ret[0]
            change_str = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"
        else:
            change_str = "N/A"
        
        report += f"| {regime} | {old_val} | {new_val} | {change_str} |\n"

    report += """
---

## ⚠️ Key Changes Detected

### 1. Bull Market Returns
- **Old Era:** Strong Bull ~7% (90d)
- **New Era:** Check if higher/lower due to 2023-2024 rally

### 2. Bear Market Opportunities
- 2022 Bear was brief but sharp
- Check if reversion alpha still works

### 3. Sector Rotation
- Defense/PSU stocks exploded 2023-2025
- Tech had a flat period

---

## 📝 Recommendation

Based on the comparison, update the playbook if:
1. Returns differ by >3% in same regime
2. Win rates differ by >10%
3. New patterns emerge in 2022-2026 data

---

*Analysis extended through 2026*
"""
    
    with open(f"{OUTPUT_DIR}/analysis_update_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Saved: {OUTPUT_DIR}/analysis_update_report.md")

def main():
    print("="*60)
    print("EXPANDED ANALYSIS: 100+ Stocks through 2026")
    print("="*60)
    
    # Fetch data
    data_cache = fetch_all_data()
    
    if 'NIFTY' not in data_cache:
        print("ERROR: Could not fetch Nifty data")
        return
    
    # Generate signals
    signals_df = generate_signals(data_cache, data_cache['NIFTY'])
    
    if signals_df.empty:
        print("ERROR: No signals generated")
        return
    
    # Compare eras
    comparison_df = compare_eras(signals_df)
    
    # Save outputs
    save_outputs(signals_df, comparison_df)
    
    # Generate report
    generate_report(signals_df, comparison_df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print(f"Output folder: {OUTPUT_DIR}/")
    print("="*60)

if __name__ == "__main__":
    main()
