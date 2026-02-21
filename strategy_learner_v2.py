"""
Strategy Learner Analysis v2 - Deep Analysis
=============================================
Enhanced analysis with:
- Volume patterns during rally
- Max drawdown / stop loss analysis
- Consolidation periods (no movement)
- Trailing stop behavior
- Phase-based metrics

Run: python strategy_learner_v2.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

# Import from existing modules
from utils.nifty500_list import TICKERS
from utils.data_engine import batch_fetch_tickers
from utils.scoring import calculate_scores, calculate_trend_metrics

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "top_n": 25,
    "lookback_months": 12,
    "output_file": "strategy_learner_deep_results.csv",
    "summary_file": "strategy_learner_deep_summary.csv",
}


# ==========================================
# ENHANCED ANALYSIS FUNCTIONS
# ==========================================

def calculate_max_drawdown(prices: pd.Series) -> Tuple[float, datetime, datetime]:
    """
    Calculate maximum drawdown during a period.
    Returns: (max_dd_pct, peak_date, trough_date)
    """
    if len(prices) < 2:
        return 0.0, None, None
    
    # Calculate running maximum
    running_max = prices.expanding().max()
    drawdown = (prices - running_max) / running_max * 100
    
    max_dd = drawdown.min()
    trough_idx = drawdown.idxmin()
    
    # Find the peak before the trough
    peak_idx = prices[:trough_idx].idxmax() if trough_idx else prices.index[0]
    
    return round(max_dd, 2), peak_idx, trough_idx


def calculate_volume_metrics(ohlcv: pd.DataFrame, rally_start: datetime) -> Dict:
    """
    Calculate volume-related metrics during the rally.
    """
    if 'Volume' not in ohlcv.columns or len(ohlcv) < 50:
        return {
            "avg_volume_before": None,
            "avg_volume_during": None,
            "volume_increase_pct": None,
            "volume_spike_days": None,
            "max_volume_spike": None
        }
    
    # Split into before and during rally
    before_rally = ohlcv[ohlcv.index < rally_start]
    during_rally = ohlcv[ohlcv.index >= rally_start]
    
    if len(before_rally) < 20 or len(during_rally) < 5:
        return {
            "avg_volume_before": None,
            "avg_volume_during": None,
            "volume_increase_pct": None,
            "volume_spike_days": None,
            "max_volume_spike": None
        }
    
    avg_before = before_rally['Volume'].mean()
    avg_during = during_rally['Volume'].mean()
    
    volume_increase = ((avg_during - avg_before) / avg_before * 100) if avg_before > 0 else 0
    
    # Count days with volume > 2x average (spike days)
    spike_threshold = avg_before * 2
    spike_days = len(during_rally[during_rally['Volume'] > spike_threshold])
    
    # Max single day volume spike
    max_spike = (during_rally['Volume'].max() / avg_before) if avg_before > 0 else 0
    
    return {
        "avg_volume_before": int(avg_before),
        "avg_volume_during": int(avg_during),
        "volume_increase_pct": round(volume_increase, 1),
        "volume_spike_days": spike_days,
        "max_volume_spike": round(max_spike, 1)
    }


def calculate_consolidation_periods(ohlcv: pd.DataFrame, rally_start: datetime) -> Dict:
    """
    Identify consolidation periods (low movement days) during the rally.
    Consolidation = periods where daily range is < 2% for 5+ consecutive days
    """
    if len(ohlcv) < 20:
        return {
            "consolidation_periods": 0,
            "max_consolidation_days": 0,
            "total_consolidation_days": 0,
            "consolidation_pct_of_rally": 0
        }
    
    during_rally = ohlcv[ohlcv.index >= rally_start].copy()
    
    if len(during_rally) < 5:
        return {
            "consolidation_periods": 0,
            "max_consolidation_days": 0,
            "total_consolidation_days": 0,
            "consolidation_pct_of_rally": 0
        }
    
    # Calculate daily range as percentage
    during_rally['daily_range'] = abs(during_rally['High'] - during_rally['Low']) / during_rally['Close'] * 100
    
    # Mark low movement days (< 2% range)
    during_rally['low_movement'] = during_rally['daily_range'] < 2.0
    
    # Find consecutive low movement periods
    consolidation_periods = []
    current_streak = 0
    
    for is_low in during_rally['low_movement']:
        if is_low:
            current_streak += 1
        else:
            if current_streak >= 5:  # 5+ days = consolidation
                consolidation_periods.append(current_streak)
            current_streak = 0
    
    if current_streak >= 5:
        consolidation_periods.append(current_streak)
    
    total_consolidation = sum(consolidation_periods)
    max_consolidation = max(consolidation_periods) if consolidation_periods else 0
    consolidation_pct = (total_consolidation / len(during_rally) * 100) if len(during_rally) > 0 else 0
    
    return {
        "consolidation_periods": len(consolidation_periods),
        "max_consolidation_days": max_consolidation,
        "total_consolidation_days": total_consolidation,
        "consolidation_pct_of_rally": round(consolidation_pct, 1)
    }


def analyze_trailing_stop(ohlcv: pd.DataFrame, rally_start: datetime, trailing_pcts: List[float] = [-10, -15, -20]) -> Dict:
    """
    Simulate trailing stop behavior during the rally.
    Returns how many times each trailing stop would have been triggered.
    """
    during_rally = ohlcv[ohlcv.index >= rally_start].copy()
    
    if len(during_rally) < 10:
        return {f"trailing_{abs(int(p))}_triggers": None for p in trailing_pcts}
    
    results = {}
    
    for pct in trailing_pcts:
        # Simulate trailing stop
        close = during_rally['Close']
        running_high = close.expanding().max()
        drawdown_from_high = (close - running_high) / running_high * 100
        
        # Count how many times we would have been stopped out
        stopped = drawdown_from_high < pct
        trigger_count = stopped.sum()
        
        # Find first trigger date
        first_trigger = stopped.idxmax() if stopped.any() else None
        
        results[f"trailing_{abs(int(pct))}_triggers"] = trigger_count
        if first_trigger:
            # Calculate % of rally captured before stop
            if len(during_rally) > 0:
                first_price = during_rally['Close'].iloc[0]
                stop_price = during_rally.loc[first_trigger, 'Close']
                pct_captured = ((stop_price - first_price) / first_price * 100)
                results[f"trailing_{abs(int(pct))}_captured"] = round(pct_captured, 1)
            else:
                results[f"trailing_{abs(int(pct))}_captured"] = None
        else:
            results[f"trailing_{abs(int(pct))}_captured"] = "never_triggered"
    
    return results


def calculate_phase_metrics(ohlcv: pd.DataFrame, rally_start: datetime) -> Dict:
    """
    Divide rally into phases and calculate metrics for each.
    Phases: Early (first 25%), Mid (25-75%), Late (75-100%)
    """
    during_rally = ohlcv[ohlcv.index >= rally_start].copy()
    
    if len(during_rally) < 20:
        return {}
    
    n = len(during_rally)
    early_end = n // 4
    mid_end = 3 * n // 4
    
    phases = {
        "early": during_rally.iloc[:early_end],
        "mid": during_rally.iloc[early_end:mid_end],
        "late": during_rally.iloc[mid_end:]
    }
    
    results = {}
    
    for phase_name, phase_data in phases.items():
        if len(phase_data) < 3:
            continue
            
        # Calculate metrics for this phase
        close = phase_data['Close']
        phase_return = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100) if len(close) > 1 else 0
        
        # Volatility (std of daily returns)
        daily_returns = close.pct_change().dropna()
        volatility = daily_returns.std() * 100 * np.sqrt(252)  # Annualized
        
        # Max drawdown in this phase
        max_dd, _, _ = calculate_max_drawdown(close)
        
        # Average daily range
        if 'High' in phase_data.columns and 'Low' in phase_data.columns:
            avg_range = ((phase_data['High'] - phase_data['Low']) / phase_data['Close'] * 100).mean()
        else:
            avg_range = 0
        
        results[f"{phase_name}_return_pct"] = round(phase_return, 1)
        results[f"{phase_name}_volatility"] = round(volatility, 1)
        results[f"{phase_name}_max_drawdown"] = round(max_dd, 1)
        results[f"{phase_name}_avg_daily_range"] = round(avg_range, 2)
        results[f"{phase_name}_days"] = len(phase_data)
    
    return results


def calculate_trend_score_at_date(ohlcv: pd.DataFrame, target_date: datetime) -> Dict:
    """Calculate trend score at a specific historical date."""
    data = ohlcv[ohlcv.index <= target_date].copy()
    
    if len(data) < 200:
        return {"trend_score": None, "dist_52w": None, "dist_200dma": None}
    
    try:
        close = data['Close']
        current_price = float(close.iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
        
        high_52 = data['High'].iloc[-252:].max() if 'High' in data.columns else close.iloc[-252:].max()
        low_52 = data['Low'].iloc[-252:].min() if 'Low' in data.columns else close.iloc[-252:].min()
        
        score = 50
        
        if current_price > ma50:
            score += 15
        else:
            score -= 10
            
        if current_price > ma200:
            score += 15
        else:
            score -= 15
            
        if ma50 > ma200:
            score += 10
        else:
            score -= 5
        
        if high_52 > low_52:
            pos = (current_price - low_52) / (high_52 - low_52)
            score += int((pos - 0.5) * 30)
            
            dist_52w = ((current_price - high_52) / high_52) * 100
            if dist_52w > -5:
                score += 10
            elif dist_52w < -30:
                score -= 10
        else:
            dist_52w = 0
        
        score = max(0, min(100, score))
        dist_200dma = ((current_price - ma200) / ma200) * 100 if ma200 > 0 else 0
        
        return {
            "trend_score": int(score),
            "dist_52w": round(dist_52w, 2),
            "dist_200dma": round(dist_200dma, 2),
            "price": round(current_price, 2)
        }
    except:
        return {"trend_score": None, "dist_52w": None, "dist_200dma": None}


def identify_rally_start(ohlcv: pd.DataFrame) -> Optional[datetime]:
    """Find the date when the major uptrend began."""
    if len(ohlcv) < 60:
        return None
    
    close = ohlcv['Close']
    ma50 = close.rolling(50).mean()
    above_ma = close > ma50
    
    for i in range(50, len(above_ma) - 10):
        if not above_ma.iloc[i-1] and above_ma.iloc[i]:
            if above_ma.iloc[i:i+5].all():
                return ohlcv.index[i]
    
    return ohlcv.index[len(ohlcv) // 5] if len(ohlcv) > 5 else ohlcv.index[0]


def get_top_performers(n: int = 25, months: int = 12) -> pd.DataFrame:
    """Identify top N performers by actual price returns."""
    print(f"[DATA] Fetching top {n} performers over {months} months...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)
    
    try:
        prices = yf.download(TICKERS, start=start_date, end=end_date, 
                            interval="1d", progress=True, auto_adjust=True)['Close']
    except Exception as e:
        print(f"   Error: {e}")
        return pd.DataFrame()
    
    returns = []
    for ticker in TICKERS:
        try:
            if ticker in prices.columns:
                col = prices[ticker].dropna()
                if len(col) >= 50:
                    start_price = col.iloc[0]
                    end_price = col.iloc[-1]
                    ret = ((end_price - start_price) / start_price) * 100
                    returns.append({
                        "ticker": ticker,
                        "start_price": start_price,
                        "end_price": end_price,
                        "return_pct": ret
                    })
        except:
            continue
    
    df = pd.DataFrame(returns).sort_values("return_pct", ascending=False).head(n)
    print(f"   [OK] Found {len(df)} top performers")
    return df


def get_current_fundamentals(ticker: str, market_df: pd.DataFrame) -> Dict:
    """Get current fundamental scores for a stock."""
    if market_df.empty:
        return {}
    
    row = market_df[market_df['ticker'] == ticker]
    if row.empty:
        return {}
    
    row = row.iloc[0]
    return {
        "name": row.get("name", ticker),
        "sector": row.get("sector", "Unknown"),
        "overall": row.get("overall", 5),
        "quality": row.get("quality", 5),
        "value": row.get("value", 5),
        "growth": row.get("growth", 5),
        "momentum": row.get("momentum", 5),
    }


def run_deep_analysis():
    """Main deep analysis function."""
    print("\n" + "="*70)
    print("STRATEGY LEARNER - DEEP ANALYSIS v2")
    print("="*70)
    print(f"Analyzing top {CONFIG['top_n']} performers with enhanced metrics")
    print()
    
    # Step 1: Get top performers
    top_df = get_top_performers(n=CONFIG['top_n'], months=CONFIG['lookback_months'])
    if top_df.empty:
        print("[ERROR] Failed to get top performers")
        return
    
    # Step 2: Get current market data for fundamentals
    print("\n[DATA] Loading market data for fundamentals...")
    market_df = batch_fetch_tickers(TICKERS)
    
    from utils.historical_tracker import calculate_multi_period_returns
    scored_data = []
    for _, row in market_df.iterrows():
        data = row.to_dict()
        returns = calculate_multi_period_returns(row['ticker'])
        data.update(returns)
        trends = calculate_trend_metrics(data)
        data.update(trends)
        scores = calculate_scores(data)
        data.update(scores)
        scored_data.append(data)
    market_df = pd.DataFrame(scored_data)
    
    # Step 3: Fetch detailed historical data
    print("\n[DATA] Fetching historical OHLCV data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=CONFIG['lookback_months'] * 30 + 250)
    
    top_tickers = top_df['ticker'].tolist()
    
    try:
        historical = yf.download(
            top_tickers, 
            start=start_date, 
            end=end_date,
            interval="1d",
            progress=True,
            auto_adjust=True,
            group_by='ticker'
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        return
    
    # Step 4: Deep analysis for each winner
    print("\n[ANALYSIS] Running deep analysis on each winner...")
    results = []
    
    for idx, row in top_df.iterrows():
        ticker = row['ticker']
        print(f"   [{len(results)+1}/{len(top_df)}] {ticker}...")
        
        try:
            if len(top_tickers) == 1:
                ohlcv = historical.copy()
            else:
                ohlcv = historical[ticker].copy() if ticker in historical.columns.get_level_values(0) else pd.DataFrame()
            
            if ohlcv.empty or len(ohlcv) < 200:
                print(f"      [SKIP] Insufficient data")
                continue
            
            fundamentals = get_current_fundamentals(ticker, market_df)
            rally_start = identify_rally_start(ohlcv)
            
            # Basic metrics
            result = {
                "ticker": ticker,
                "name": fundamentals.get("name", ticker),
                "sector": fundamentals.get("sector", "Unknown"),
                "return_12mo": round(row['return_pct'], 1),
                "rally_start_date": rally_start.strftime("%Y-%m-%d") if rally_start else None,
            }
            
            # Current fundamentals
            result["quality"] = fundamentals.get("quality")
            result["value"] = fundamentals.get("value")
            result["growth"] = fundamentals.get("growth")
            result["momentum"] = fundamentals.get("momentum")
            result["overall"] = fundamentals.get("overall")
            
            # Calculate trend score at rally start
            rally_score = calculate_trend_score_at_date(ohlcv, rally_start)
            result["score_at_rally_start"] = rally_score.get("trend_score")
            
            # Max drawdown during rally
            during_rally = ohlcv[ohlcv.index >= rally_start]
            if len(during_rally) > 10:
                max_dd, dd_peak, dd_trough = calculate_max_drawdown(during_rally['Close'])
                result["max_drawdown_pct"] = max_dd
                result["drawdown_peak_date"] = dd_peak.strftime("%Y-%m-%d") if dd_peak else None
                result["drawdown_trough_date"] = dd_trough.strftime("%Y-%m-%d") if dd_trough else None
            else:
                result["max_drawdown_pct"] = None
            
            # Volume metrics
            volume_metrics = calculate_volume_metrics(ohlcv, rally_start)
            result.update(volume_metrics)
            
            # Consolidation periods
            consolidation = calculate_consolidation_periods(ohlcv, rally_start)
            result.update(consolidation)
            
            # Trailing stop analysis
            trailing = analyze_trailing_stop(ohlcv, rally_start)
            result.update(trailing)
            
            # Phase metrics
            phases = calculate_phase_metrics(ohlcv, rally_start)
            result.update(phases)
            
            results.append(result)
            
        except Exception as e:
            print(f"      [ERROR] {e}")
            continue
    
    # Step 5: Create output DataFrame
    results_df = pd.DataFrame(results)
    
    # Step 6: Calculate deep summary statistics
    print("\n[SUMMARY] Computing aggregate insights...")
    
    summary = {"metric": [], "value": [], "insight": []}
    
    # Drawdown insights
    avg_max_dd = results_df["max_drawdown_pct"].mean()
    summary["metric"].append("Avg Max Drawdown")
    summary["value"].append(f"{avg_max_dd:.1f}%")
    summary["insight"].append("Average pain during rallies - set stops accordingly")
    
    min_dd = results_df["max_drawdown_pct"].max()  # Least negative = best
    max_dd = results_df["max_drawdown_pct"].min()  # Most negative = worst
    summary["metric"].append("Drawdown Range")
    summary["value"].append(f"{max_dd:.1f}% to {min_dd:.1f}%")
    summary["insight"].append("Wide range suggests stock-specific risk management needed")
    
    # Volume insights
    avg_vol_increase = results_df["volume_increase_pct"].mean()
    summary["metric"].append("Avg Volume Increase")
    summary["value"].append(f"{avg_vol_increase:.0f}%")
    summary["insight"].append("Winners show volume confirmation during rally")
    
    avg_spike_days = results_df["volume_spike_days"].mean()
    summary["metric"].append("Avg Volume Spike Days")
    summary["value"].append(f"{avg_spike_days:.0f} days")
    summary["insight"].append("Number of 2x volume days during rally")
    
    # Consolidation insights
    avg_consol_pct = results_df["consolidation_pct_of_rally"].mean()
    summary["metric"].append("Avg Consolidation %")
    summary["value"].append(f"{avg_consol_pct:.0f}%")
    summary["insight"].append("Portion of rally spent in sideways movement")
    
    # Trailing stop insights
    for pct in [10, 15, 20]:
        col = f"trailing_{pct}_triggers"
        if col in results_df.columns:
            avg_triggers = results_df[col].mean()
            summary["metric"].append(f"-{pct}% Trailing Stop")
            summary["value"].append(f"{avg_triggers:.0f} triggers avg")
            summary["insight"].append(f"How often a -{pct}% trailing stop hits during rally")
    
    # Phase insights
    for phase in ["early", "mid", "late"]:
        col = f"{phase}_return_pct"
        if col in results_df.columns:
            avg_ret = results_df[col].mean()
            summary["metric"].append(f"{phase.title()} Phase Return")
            summary["value"].append(f"{avg_ret:.1f}%")
            summary["insight"].append(f"Average return in {phase} phase of rally")
    
    # Fundamental ranges
    for fund in ["quality", "value", "growth", "momentum"]:
        col_min = results_df[fund].min()
        col_max = results_df[fund].max()
        col_avg = results_df[fund].mean()
        summary["metric"].append(f"{fund.title()} Range")
        summary["value"].append(f"{col_min:.1f} - {col_max:.1f} (avg: {col_avg:.1f})")
        summary["insight"].append(f"Current {fund} scores among winners")
    
    summary_df = pd.DataFrame(summary)
    
    # Step 7: Save outputs
    results_df.to_csv(CONFIG["output_file"], index=False)
    summary_df.to_csv(CONFIG["summary_file"], index=False)
    
    # Print insights
    print("\n" + "="*70)
    print("[OK] DEEP ANALYSIS COMPLETE")
    print("="*70)
    
    print(f"\nResults: {CONFIG['output_file']}")
    print(f"Summary: {CONFIG['summary_file']}")
    
    print("\n" + "-"*50)
    print("KEY DEEP INSIGHTS")
    print("-"*50)
    
    print(f"\n[DRAWDOWN ANALYSIS]")
    print(f"   * Average max drawdown during rally: {avg_max_dd:.1f}%")
    print(f"   * Worst drawdown among winners: {max_dd:.1f}%")
    print(f"   * Best drawdown (least pain): {min_dd:.1f}%")
    
    print(f"\n[VOLUME PATTERNS]")
    print(f"   * Avg volume increase during rally: {avg_vol_increase:.0f}%")
    print(f"   * Avg high-volume days (2x normal): {avg_spike_days:.0f}")
    
    print(f"\n[CONSOLIDATION]")
    print(f"   * % of rally in consolidation: {avg_consol_pct:.0f}%")
    
    print(f"\n[TRAILING STOP BEHAVIOR]")
    for pct in [10, 15, 20]:
        col = f"trailing_{pct}_triggers"
        cap_col = f"trailing_{pct}_captured"
        if col in results_df.columns:
            avg_triggers = results_df[col].mean()
            print(f"   * -{pct}% trailing: {avg_triggers:.0f} triggers on average")
    
    print(f"\n[PHASE RETURNS (Early/Mid/Late)]")
    for phase in ["early", "mid", "late"]:
        col = f"{phase}_return_pct"
        if col in results_df.columns:
            avg_ret = results_df[col].mean()
            print(f"   * {phase.title()}: {avg_ret:.1f}% avg return")
    
    print(f"\n[FUNDAMENTAL RANGES OF WINNERS (Current)]")
    for fund in ["quality", "value", "growth", "momentum"]:
        print(f"   * {fund.title()}: {results_df[fund].min():.1f} - {results_df[fund].max():.1f} (avg: {results_df[fund].mean():.1f})")
    
    print("\n")
    return results_df, summary_df


if __name__ == "__main__":
    run_deep_analysis()
