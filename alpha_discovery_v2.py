"""
Alpha Discovery Framework v2.0 - COMPREHENSIVE
===============================================
Unbiased deep analysis across 10 years of data.

Dimensions:
- Trend (5 buckets)
- Momentum (5 buckets)
- Volume Delta (5 buckets)
- Volatility (3 buckets)
- Regime (6 types)
- Sector (11 sectors)

Outputs to: alpha_findings/
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import warnings
import os
warnings.filterwarnings('ignore')

# --- OUTPUT DIRECTORY ---
OUTPUT_DIR = "alpha_findings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- SECTOR-WISE TICKERS ---
SECTOR_TICKERS = {
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"],
    "IT_Services": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
    "Pharma": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "LUPIN.NS", "AUROPHARMA.NS"],
    "Industrials": ["LT.NS", "CUMMINSIND.NS", "ABB.NS", "SIEMENS.NS", "HAVELLS.NS"],
    "Metals": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "VEDL.NS", "COALINDIA.NS"],
    "Auto": ["TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "MARUTI.NS", "EICHERMOT.NS"],
    "Consumer": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS"],
    "Telecom": ["BHARTIARTL.NS", "TATACOMM.NS", "TATAPOWER.NS", "POWERGRID.NS", "NTPC.NS"],
    "Realty": ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "PHOENIXLTD.NS"],
    "Defense": ["BEL.NS", "HAL.NS", "BHEL.NS", "IRCTC.NS", "IRFC.NS"],
}

NIFTY50_TICKER = "^NSEI"
HORIZONS = [10, 20, 40, 60, 90]
RF_RATE = 0.06  # 6% annual

# Bucket definitions
TREND_BUCKETS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
MOMENTUM_BUCKETS = [(-100, -5), (-5, 0), (0, 5), (5, 15), (15, 100)]  # 1M return %
VOLUME_DELTA_BUCKETS = [(-10, -2), (-2, 0), (0, 2), (2, 4), (4, 10)]
VOLATILITY_BUCKETS = [(0, 15), (15, 30), (30, 100)]  # ATR %


def fetch_data(ticker: str, period: str = "10y") -> pd.DataFrame:
    """Fetch OHLCV data."""
    try:
        data = yf.Ticker(ticker).history(period=period)
        if data.empty or len(data) < 100:
            return pd.DataFrame()
        return data
    except Exception as e:
        return pd.DataFrame()


def calculate_trend_score(price: float, ma50: float, ma200: float, high_52w: float, low_52w: float) -> float:
    """Calculate Trend Score 0-100."""
    score = 50
    
    if ma50 > 0:
        score += 15 if price > ma50 else -10
    if ma200 > 0:
        score += 15 if price > ma200 else -15
    if ma50 > 0 and ma200 > 0:
        score += 10 if ma50 > ma200 else -5
    if high_52w > low_52w:
        range_52w = high_52w - low_52w
        position = (price - low_52w) / range_52w
        score += int((position - 0.5) * 30)
    
    return max(0, min(100, score))


def calculate_volume_score(volumes: pd.Series) -> float:
    """Calculate Volume Score 0-10."""
    if len(volumes) < 60:
        return 5.0
    
    avg_10d = volumes.iloc[-10:].mean()
    avg_60d = volumes.mean()
    
    if avg_60d == 0:
        return 5.0
    
    ratio = avg_10d / avg_60d
    
    if ratio <= 0.7:
        return 0.0
    elif ratio >= 1.5:
        return 10.0
    elif ratio < 1.0:
        return ((ratio - 0.7) / 0.3) * 5
    else:
        return 5 + ((ratio - 1.0) / 0.5) * 5


def calculate_atr_pct(hist: pd.DataFrame, period: int = 14) -> float:
    """Calculate ATR as % of price."""
    if len(hist) < period + 1:
        return 0
    
    high = hist['High'].iloc[-period:]
    low = hist['Low'].iloc[-period:]
    close = hist['Close'].iloc[-period-1:-1]
    
    tr = pd.concat([
        high - low,
        abs(high - close),
        abs(low - close)
    ], axis=1).max(axis=1)
    
    atr = tr.mean()
    current_price = hist['Close'].iloc[-1]
    
    return (atr / current_price * 100) if current_price > 0 else 0


def detect_bullish_divergence(hist: pd.DataFrame, lookback: int = 14) -> bool:
    """Detect bullish divergence (price down, RSI up)."""
    if len(hist) < lookback + 5:
        return False
    
    # Simple RSI calculation
    delta = hist['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 0.0001)
    rsi = 100 - (100 / (1 + rs))
    
    # Compare last 5 days
    price_trend = hist['Close'].iloc[-5:].mean() < hist['Close'].iloc[-10:-5].mean()
    rsi_trend = rsi.iloc[-5:].mean() > rsi.iloc[-10:-5].mean()
    
    return price_trend and rsi_trend


def detect_golden_cross(hist: pd.DataFrame) -> bool:
    """Detect MA50 crossing above MA200."""
    if len(hist) < 210:
        return False
    
    ma50 = hist['Close'].rolling(50).mean()
    ma200 = hist['Close'].rolling(200).mean()
    
    # Cross in last 5 days
    prev_below = ma50.iloc[-10] < ma200.iloc[-10]
    now_above = ma50.iloc[-1] > ma200.iloc[-1]
    
    return prev_below and now_above


def classify_regime(nifty_data: pd.DataFrame, date: pd.Timestamp) -> str:
    """Enhanced regime classification."""
    data_slice = nifty_data.loc[:date]
    if len(data_slice) < 210:
        return 'Choppy'
    
    ma50 = data_slice['Close'].rolling(50).mean()
    ma200 = data_slice['Close'].rolling(200).mean()
    
    ma50_curr = ma50.iloc[-1]
    ma200_curr = ma200.iloc[-1]
    ma50_prev = ma50.iloc[-20] if len(ma50) > 20 else ma50_curr
    
    ma50_trend = (ma50_curr - ma50_prev) / ma50_prev * 100 if ma50_prev > 0 else 0
    
    # Check for recent crossover (Recovery)
    ma50_10ago = ma50.iloc[-10] if len(ma50) > 10 else ma50_curr
    ma200_10ago = ma200.iloc[-10] if len(ma200) > 10 else ma200_curr
    
    if ma50_10ago < ma200_10ago and ma50_curr > ma200_curr:
        return 'Recovery'
    
    if ma50_curr > ma200_curr:
        if ma50_trend > 0.5:
            return 'Strong_Bull'
        else:
            return 'Mild_Bull'
    elif ma50_curr < ma200_curr:
        if ma50_trend < -0.5:
            return 'Strong_Bear'
        else:
            return 'Mild_Bear'
    else:
        return 'Choppy'


def get_bucket_label(value: float, buckets: List[Tuple[float, float]], labels: List[str]) -> str:
    """Map value to bucket label."""
    for i, (low, high) in enumerate(buckets):
        if low <= value < high:
            return labels[i]
    return labels[-1]


def calculate_forward_metrics(hist: pd.DataFrame, entry_idx: int, horizon: int) -> Dict:
    """Calculate forward return and drawdown metrics."""
    if entry_idx + horizon >= len(hist):
        return {'return': np.nan, 'max_dd': np.nan, 'dd_duration': np.nan}
    
    entry_price = hist['Close'].iloc[entry_idx]
    future_prices = hist['Close'].iloc[entry_idx:entry_idx + horizon + 1]
    
    exit_price = future_prices.iloc[-1]
    ret = ((exit_price - entry_price) / entry_price) * 100
    
    # Max drawdown
    running_max = entry_price
    max_dd = 0
    dd_days = 0
    max_dd_duration = 0
    current_dd_duration = 0
    
    for price in future_prices:
        if price >= running_max:
            running_max = price
            if current_dd_duration > max_dd_duration:
                max_dd_duration = current_dd_duration
            current_dd_duration = 0
        else:
            dd = (price - running_max) / running_max * 100
            max_dd = min(max_dd, dd)
            current_dd_duration += 1
    
    return {
        'return': ret,
        'max_dd': abs(max_dd),
        'dd_duration': max_dd_duration
    }


def run_comprehensive_analysis():
    """Main analysis runner."""
    print("=" * 70)
    print("ALPHA DISCOVERY FRAMEWORK v2.0 - COMPREHENSIVE")
    print("=" * 70)
    print(f"Output Directory: {OUTPUT_DIR}/")
    print()
    
    # --- PHASE 1: Fetch Nifty 50 for regime ---
    print("[1/7] Fetching Nifty 50 index data...")
    nifty_data = fetch_data(NIFTY50_TICKER, period="10y")
    if nifty_data.empty:
        print("[ERROR] Could not fetch Nifty 50 data.")
        return
    print(f"      Nifty 50: {len(nifty_data)} days ({nifty_data.index.min().date()} to {nifty_data.index.max().date()})")
    
    # --- PHASE 2: Fetch all sector stocks ---
    print("\n[2/7] Fetching sector-wise stock data...")
    all_stock_data = {}
    sector_mapping = {}
    
    for sector, tickers in SECTOR_TICKERS.items():
        print(f"      {sector}:", end=" ")
        fetched = 0
        for ticker in tickers:
            data = fetch_data(ticker, period="10y")
            if not data.empty and len(data) > 250:
                all_stock_data[ticker] = data
                sector_mapping[ticker] = sector
                fetched += 1
        print(f"{fetched}/{len(tickers)} stocks")
    
    total_stocks = len(all_stock_data)
    print(f"\n      Total: {total_stocks} stocks loaded")
    
    if total_stocks < 20:
        print("[ERROR] Insufficient stocks. Exiting.")
        return
    
    # --- PHASE 3: Generate signals ---
    print("\n[3/7] Generating signal events...")
    
    all_signals = []
    
    for ticker, hist in all_stock_data.items():
        sector = sector_mapping[ticker]
        
        # Pre-calculate MAs
        hist['MA50'] = hist['Close'].rolling(50).mean()
        hist['MA200'] = hist['Close'].rolling(200).mean()
        hist['High52W'] = hist['Close'].rolling(252).max()
        hist['Low52W'] = hist['Close'].rolling(252).min()
        hist['Return1M'] = hist['Close'].pct_change(20) * 100
        
        # Sample every 5 trading days after warmup period
        sample_indices = list(range(260, len(hist) - 100, 5))
        
        for idx in sample_indices:
            date = hist.index[idx]
            row = hist.iloc[idx]
            hist_slice = hist.iloc[max(0, idx-260):idx+1]
            
            # Calculate all metrics
            trend_score = calculate_trend_score(
                row['Close'], row['MA50'], row['MA200'], row['High52W'], row['Low52W']
            )
            
            vol_score_now = calculate_volume_score(hist_slice['Volume'])
            vol_score_prev = calculate_volume_score(hist.iloc[max(0, idx-5-260):idx-5]['Volume']) if idx > 5 else vol_score_now
            vol_delta = vol_score_now - vol_score_prev
            
            momentum_1m = row['Return1M'] if not pd.isna(row['Return1M']) else 0
            atr_pct = calculate_atr_pct(hist_slice)
            
            regime = classify_regime(nifty_data, date)
            
            # Signals
            bullish_div = detect_bullish_divergence(hist_slice)
            golden_cross = detect_golden_cross(hist_slice)
            
            # Bucket labels
            trend_bucket = get_bucket_label(trend_score, TREND_BUCKETS, ['0-20', '20-40', '40-60', '60-80', '80-100'])
            mom_bucket = get_bucket_label(momentum_1m, MOMENTUM_BUCKETS, ['Strong_Down', 'Weak_Down', 'Flat', 'Weak_Up', 'Strong_Up'])
            vol_bucket = get_bucket_label(vol_delta, VOLUME_DELTA_BUCKETS, ['Big_Drop', 'Drop', 'Flat', 'Jump', 'Spike'])
            volatility_bucket = get_bucket_label(atr_pct, VOLATILITY_BUCKETS, ['Low', 'Med', 'High'])
            
            # Calculate forward returns for all horizons
            signal = {
                'ticker': ticker,
                'sector': sector,
                'date': date,
                'trend_score': trend_score,
                'trend_bucket': trend_bucket,
                'momentum_1m': momentum_1m,
                'mom_bucket': mom_bucket,
                'vol_delta': vol_delta,
                'vol_bucket': vol_bucket,
                'atr_pct': atr_pct,
                'volatility_bucket': volatility_bucket,
                'regime': regime,
                'bullish_div': bullish_div,
                'golden_cross': golden_cross,
            }
            
            for h in HORIZONS:
                metrics = calculate_forward_metrics(hist, idx, h)
                signal[f'ret_{h}d'] = metrics['return']
                signal[f'dd_{h}d'] = metrics['max_dd']
            
            all_signals.append(signal)
    
    df = pd.DataFrame(all_signals)
    df = df.dropna(subset=['ret_60d'])
    print(f"      Generated {len(df)} signal events")
    
    # Save raw signals
    df.to_csv(f"{OUTPUT_DIR}/raw_signals.csv", index=False)
    print(f"      Saved: {OUTPUT_DIR}/raw_signals.csv")
    
    # --- PHASE 4: Regime Analysis ---
    print("\n[4/7] Analyzing by Market Regime...")
    
    regime_stats = []
    for regime in df['regime'].unique():
        regime_df = df[df['regime'] == regime]
        for horizon in HORIZONS:
            ret_col = f'ret_{horizon}d'
            returns = regime_df[ret_col].dropna()
            if len(returns) >= 20:
                regime_stats.append({
                    'regime': regime,
                    'horizon': horizon,
                    'n': len(returns),
                    'avg_ret': returns.mean(),
                    'median_ret': returns.median(),
                    'win_rate': (returns > 0).mean() * 100,
                    'std': returns.std(),
                    'sharpe': (returns.mean() - RF_RATE/365*horizon) / returns.std() * np.sqrt(252/horizon) if returns.std() > 0 else 0,
                    'worst': returns.min(),
                    'best': returns.max(),
                })
    
    regime_df_out = pd.DataFrame(regime_stats)
    regime_df_out.to_csv(f"{OUTPUT_DIR}/regime_performance.csv", index=False)
    print(f"      Saved: {OUTPUT_DIR}/regime_performance.csv")
    
    # --- PHASE 5: Sector Playbooks ---
    print("\n[5/7] Building Sector Playbooks...")
    
    sector_playbooks = {}
    sector_stats = []
    
    for sector in df['sector'].unique():
        sector_df = df[df['sector'] == sector]
        
        best_setup = None
        best_sharpe = -999
        
        for regime in sector_df['regime'].unique():
            for trend in sector_df['trend_bucket'].unique():
                for vol in sector_df['vol_bucket'].unique():
                    cell = sector_df[
                        (sector_df['regime'] == regime) &
                        (sector_df['trend_bucket'] == trend) &
                        (sector_df['vol_bucket'] == vol)
                    ]
                    
                    if len(cell) >= 15:
                        for horizon in [60, 90]:
                            ret_col = f'ret_{horizon}d'
                            returns = cell[ret_col].dropna()
                            if len(returns) >= 10:
                                avg_ret = returns.mean()
                                std_ret = returns.std()
                                sharpe = (avg_ret - RF_RATE/365*horizon) / std_ret * np.sqrt(252/horizon) if std_ret > 0 else 0
                                
                                sector_stats.append({
                                    'sector': sector,
                                    'regime': regime,
                                    'trend': trend,
                                    'volume': vol,
                                    'horizon': horizon,
                                    'n': len(returns),
                                    'avg_ret': avg_ret,
                                    'win_rate': (returns > 0).mean() * 100,
                                    'sharpe': sharpe,
                                })
                                
                                if sharpe > best_sharpe:
                                    best_sharpe = sharpe
                                    best_setup = {
                                        'best_regime': regime,
                                        'best_setup': f"{trend}_{vol}",
                                        'optimal_horizon': horizon,
                                        'expected_return': round(avg_ret, 2),
                                        'sharpe': round(sharpe, 2),
                                        'win_rate': round((returns > 0).mean() * 100, 1),
                                        'n_signals': len(returns)
                                    }
        
        if best_setup:
            sector_playbooks[sector] = best_setup
    
    pd.DataFrame(sector_stats).to_csv(f"{OUTPUT_DIR}/sector_analysis.csv", index=False)
    with open(f"{OUTPUT_DIR}/sector_playbooks.json", 'w') as f:
        json.dump(sector_playbooks, f, indent=2)
    print(f"      Saved: {OUTPUT_DIR}/sector_playbooks.json")
    
    # --- PHASE 6: Signal Effectiveness ---
    print("\n[6/7] Testing Signal Effectiveness...")
    
    signal_stats = []
    for regime in df['regime'].unique():
        for horizon in HORIZONS:
            regime_df = df[df['regime'] == regime]
            ret_col = f'ret_{horizon}d'
            
            # Baseline (no signal)
            baseline = regime_df[ret_col].dropna()
            if len(baseline) >= 20:
                signal_stats.append({
                    'regime': regime,
                    'signal': 'Baseline',
                    'horizon': horizon,
                    'n': len(baseline),
                    'avg_ret': baseline.mean(),
                    'win_rate': (baseline > 0).mean() * 100,
                })
            
            # Bullish Divergence
            div_df = regime_df[regime_df['bullish_div'] == True]
            if len(div_df) >= 10:
                returns = div_df[ret_col].dropna()
                if len(returns) >= 5:
                    signal_stats.append({
                        'regime': regime,
                        'signal': 'Bullish_Div',
                        'horizon': horizon,
                        'n': len(returns),
                        'avg_ret': returns.mean(),
                        'win_rate': (returns > 0).mean() * 100,
                    })
            
            # Golden Cross
            gc_df = regime_df[regime_df['golden_cross'] == True]
            if len(gc_df) >= 5:
                returns = gc_df[ret_col].dropna()
                if len(returns) >= 3:
                    signal_stats.append({
                        'regime': regime,
                        'signal': 'Golden_Cross',
                        'horizon': horizon,
                        'n': len(returns),
                        'avg_ret': returns.mean(),
                        'win_rate': (returns > 0).mean() * 100,
                    })
    
    pd.DataFrame(signal_stats).to_csv(f"{OUTPUT_DIR}/signal_effectiveness.csv", index=False)
    print(f"      Saved: {OUTPUT_DIR}/signal_effectiveness.csv")
    
    # --- PHASE 7: Full Matrix with Risk Metrics ---
    print("\n[7/7] Building Full Risk-Adjusted Matrix...")
    
    matrix_results = []
    
    for regime in df['regime'].unique():
        for trend in df['trend_bucket'].unique():
            for vol in df['vol_bucket'].unique():
                cell = df[
                    (df['regime'] == regime) &
                    (df['trend_bucket'] == trend) &
                    (df['vol_bucket'] == vol)
                ]
                
                if len(cell) >= 20:
                    for horizon in [60, 90]:
                        ret_col = f'ret_{horizon}d'
                        dd_col = f'dd_{horizon}d'
                        
                        returns = cell[ret_col].dropna()
                        drawdowns = cell[dd_col].dropna()
                        
                        if len(returns) >= 15:
                            avg_ret = returns.mean()
                            std_ret = returns.std()
                            win_rate = (returns > 0).mean() * 100
                            
                            # Winners and losers
                            winners = returns[returns > 0]
                            losers = returns[returns < 0]
                            avg_win = winners.mean() if len(winners) > 0 else 0
                            avg_loss = losers.mean() if len(losers) > 0 else 0
                            payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0
                            
                            # Risk metrics
                            sharpe = (avg_ret - RF_RATE/365*horizon) / std_ret * np.sqrt(252/horizon) if std_ret > 0 else 0
                            sortino_denom = returns[returns < 0].std() if len(returns[returns < 0]) > 0 else std_ret
                            sortino = (avg_ret - RF_RATE/365*horizon) / sortino_denom * np.sqrt(252/horizon) if sortino_denom > 0 else 0
                            max_dd = drawdowns.max()
                            calmar = avg_ret / max_dd if max_dd > 0 else 0
                            
                            # Tail risk
                            var_95 = np.percentile(returns, 5)
                            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
                            
                            matrix_results.append({
                                'regime': regime,
                                'trend': trend,
                                'volume': vol,
                                'horizon': horizon,
                                'n': len(returns),
                                'avg_ret': round(avg_ret, 2),
                                'median_ret': round(returns.median(), 2),
                                'win_rate': round(win_rate, 1),
                                'avg_winner': round(avg_win, 2),
                                'avg_loser': round(avg_loss, 2),
                                'payoff_ratio': round(payoff, 2),
                                'max_dd': round(max_dd, 2),
                                'sharpe': round(sharpe, 2),
                                'sortino': round(sortino, 2),
                                'calmar': round(calmar, 2),
                                'var_95': round(var_95, 2),
                                'cvar_95': round(cvar_95, 2),
                                'worst_trade': round(returns.min(), 2),
                                'best_trade': round(returns.max(), 2),
                            })
    
    matrix_df = pd.DataFrame(matrix_results)
    matrix_df.to_csv(f"{OUTPUT_DIR}/full_matrix.csv", index=False)
    print(f"      Saved: {OUTPUT_DIR}/full_matrix.csv")
    
    # --- GENERATE SUMMARY REPORT ---
    print("\n" + "=" * 70)
    print("GENERATING SUMMARY REPORT...")
    print("=" * 70)
    
    # Find top performers
    top_by_sharpe = matrix_df.nlargest(10, 'sharpe')
    top_by_return = matrix_df.nlargest(10, 'avg_ret')
    top_by_winrate = matrix_df.nlargest(10, 'win_rate')
    
    # Generate report
    report = f"""# Alpha Discovery Report v2.0

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Data Period:** {df['date'].min().date()} to {df['date'].max().date()}
**Stocks Analyzed:** {total_stocks} across 11 sectors
**Total Signals:** {len(df):,}

---

## Executive Summary

### Data Coverage by Regime
| Regime | Signals | % of Total |
|:-------|:--------|:-----------|
"""
    
    for regime in df['regime'].value_counts().index:
        count = df['regime'].value_counts()[regime]
        pct = count / len(df) * 100
        report += f"| {regime} | {count:,} | {pct:.1f}% |\n"
    
    report += """
---

## Top 10 Setups by Sharpe Ratio (60-90 Day Horizon)

| Rank | Regime | Trend | Volume | Horizon | Return | Win% | Sharpe | Max DD |
|:-----|:-------|:------|:-------|:--------|:-------|:-----|:-------|:-------|
"""
    
    for i, row in top_by_sharpe.iterrows():
        rank = top_by_sharpe.index.get_loc(i) + 1
        report += f"| {rank} | {row['regime']} | {row['trend']} | {row['volume']} | {row['horizon']}d | {row['avg_ret']:.1f}% | {row['win_rate']:.0f}% | {row['sharpe']:.2f} | {row['max_dd']:.1f}% |\n"
    
    report += """
---

## Sector Playbooks (Best Setup per Sector)

| Sector | Best Regime | Setup | Horizon | Return | Win% | Sharpe |
|:-------|:------------|:------|:--------|:-------|:-----|:-------|
"""
    
    for sector, playbook in sector_playbooks.items():
        report += f"| {sector} | {playbook['best_regime']} | {playbook['best_setup']} | {playbook['optimal_horizon']}d | {playbook['expected_return']:.1f}% | {playbook['win_rate']:.0f}% | {playbook['sharpe']:.2f} |\n"
    
    report += """
---

## Regime-Specific Performance (60-Day Horizon)

"""
    
    regime_60d = regime_df_out[regime_df_out['horizon'] == 60].sort_values('sharpe', ascending=False)
    
    report += """| Regime | N | Avg Ret | Win% | Sharpe | Worst | Best |
|:-------|:--|:--------|:-----|:-------|:------|:-----|
"""
    
    for _, row in regime_60d.iterrows():
        report += f"| {row['regime']} | {row['n']} | {row['avg_ret']:.1f}% | {row['win_rate']:.0f}% | {row['sharpe']:.2f} | {row['worst']:.1f}% | {row['best']:.1f}% |\n"
    
    report += """
---

## Key Findings

### 1. Regime Distribution
"""
    
    # Calculate regime insights
    most_common = df['regime'].mode()[0]
    regime_returns = regime_df_out[regime_df_out['horizon'] == 60].set_index('regime')['avg_ret'].to_dict()
    best_regime = max(regime_returns, key=regime_returns.get)
    worst_regime = min(regime_returns, key=regime_returns.get)
    
    report += f"""- Most common regime: **{most_common}** ({df['regime'].value_counts()[most_common]/len(df)*100:.1f}% of signals)
- Best performing regime: **{best_regime}** ({regime_returns.get(best_regime, 0):.1f}% avg return)
- Worst performing regime: **{worst_regime}** ({regime_returns.get(worst_regime, 0):.1f}% avg return)

### 2. Best Overall Setup
"""
    
    if len(top_by_sharpe) > 0:
        best = top_by_sharpe.iloc[0]
        report += f"""- **{best['regime']} + {best['trend']} Trend + {best['volume']} Volume**
- Sharpe: {best['sharpe']:.2f}
- Win Rate: {best['win_rate']:.0f}%
- Avg Return: {best['avg_ret']:.1f}%
- Optimal Horizon: {best['horizon']} days

### 3. Sector Leaders
"""
        
        # Find best sector
        best_sector = max(sector_playbooks, key=lambda x: sector_playbooks[x].get('sharpe', 0))
        bp = sector_playbooks[best_sector]
        report += f"""- Best sector: **{best_sector}**
- Setup: {bp['best_setup']} in {bp['best_regime']}
- Sharpe: {bp['sharpe']:.2f}
- Expected Return: {bp['expected_return']:.1f}%
"""
    
    report += """
---

## Output Files Generated

1. `raw_signals.csv` - All signal events with metrics
2. `regime_performance.csv` - Performance by market regime
3. `sector_analysis.csv` - Detailed sector × regime × setup analysis
4. `sector_playbooks.json` - Best setup per sector
5. `signal_effectiveness.csv` - Bullish Div & Golden Cross performance
6. `full_matrix.csv` - Complete risk-adjusted matrix

---

*Generated by Alpha Discovery Framework v2.0*
"""
    
    with open(f"{OUTPUT_DIR}/alpha_discovery_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"      Saved: {OUTPUT_DIR}/alpha_discovery_report.md")
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nKey files:")
    print(f"  - {OUTPUT_DIR}/alpha_discovery_report.md (Main Report)")
    print(f"  - {OUTPUT_DIR}/sector_playbooks.json (Per-Sector Strategies)")
    print(f"  - {OUTPUT_DIR}/full_matrix.csv (Complete Analysis)")


if __name__ == "__main__":
    run_comprehensive_analysis()
