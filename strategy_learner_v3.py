"""
Strategy Learner Analysis v3 - Multi-Period Deep Analysis
==========================================================
Enhanced analysis with:
- Top 50 stocks analysis (up from 25)
- Multiple time periods: 3 months, 6 months, 12 months
- Trend transitions (entries/exits from trend zones)
- Volume breakout accuracy
- Sideways periods (< 5% movement)
- Drawdown recovery time
- Comparative summary across periods

Run: python strategy_learner_v3.py
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
    "top_n": 50,
    "time_periods": [3, 6, 12],  # months
    "output_prefix": "strategy_learner_v3",
    "trend_threshold": 50,       # Score > 50 = in trend
    "super_uptrend_threshold": 80,  # Score > 80 = super uptrend
    "sideways_threshold": 5,     # < 5% move = sideways
    "sideways_min_days": 5,      # Minimum days for sideways period
}


# ==========================================
# ENHANCED ANALYSIS FUNCTIONS
# ==========================================

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


def calculate_trend_transitions(ohlcv: pd.DataFrame, rally_start: datetime) -> Dict:
    """
    Track when stock enters/exits trend zones.
    
    Returns:
    - Number of times entering trend (>50 score)
    - Number of times exiting trend
    - Number of times entering super uptrend (>80)
    - Average duration in trend
    """
    during_rally = ohlcv[ohlcv.index >= rally_start].copy()
    
    if len(during_rally) < 30:
        return {
            "trend_entries": None,
            "trend_exits": None,
            "super_uptrend_entries": None,
            "avg_trend_duration_days": None,
            "max_trend_duration_days": None,
            "time_in_trend_pct": None
        }
    
    # Calculate trend score for each week (sampling to reduce computation)
    sample_dates = during_rally.index[::5]  # Every 5 days
    scores = []
    
    for date in sample_dates:
        score_data = calculate_trend_score_at_date(ohlcv, date)
        if score_data["trend_score"] is not None:
            scores.append({
                "date": date,
                "score": score_data["trend_score"]
            })
    
    if len(scores) < 5:
        return {
            "trend_entries": None,
            "trend_exits": None,
            "super_uptrend_entries": None,
            "avg_trend_duration_days": None,
            "max_trend_duration_days": None,
            "time_in_trend_pct": None
        }
    
    scores_df = pd.DataFrame(scores)
    
    # Track transitions
    trend_threshold = CONFIG["trend_threshold"]
    super_threshold = CONFIG["super_uptrend_threshold"]
    
    in_trend = scores_df["score"] > trend_threshold
    in_super = scores_df["score"] > super_threshold
    
    # Count entries (False -> True transitions)
    trend_entries = ((~in_trend.shift(1).fillna(False)) & in_trend).sum()
    trend_exits = (in_trend.shift(1).fillna(False) & (~in_trend)).sum()
    super_uptrend_entries = ((~in_super.shift(1).fillna(False)) & in_super).sum()
    
    # Calculate time in trend
    time_in_trend_pct = (in_trend.sum() / len(in_trend)) * 100
    
    # Calculate trend durations
    trend_durations = []
    current_duration = 0
    for is_in_trend in in_trend:
        if is_in_trend:
            current_duration += 5  # 5 days per sample
        else:
            if current_duration > 0:
                trend_durations.append(current_duration)
            current_duration = 0
    if current_duration > 0:
        trend_durations.append(current_duration)
    
    avg_duration = np.mean(trend_durations) if trend_durations else 0
    max_duration = max(trend_durations) if trend_durations else 0
    
    return {
        "trend_entries": int(trend_entries),
        "trend_exits": int(trend_exits),
        "super_uptrend_entries": int(super_uptrend_entries),
        "avg_trend_duration_days": round(avg_duration, 0),
        "max_trend_duration_days": int(max_duration),
        "time_in_trend_pct": round(time_in_trend_pct, 1)
    }


def calculate_max_drawdown_with_recovery(prices: pd.Series) -> Dict:
    """
    Calculate maximum drawdown and time to recover.
    
    Returns: max_dd_pct, peak_date, trough_date, recovery_days
    """
    if len(prices) < 10:
        return {
            "max_drawdown_pct": None,
            "drawdown_peak_date": None,
            "drawdown_trough_date": None,
            "recovery_days": None,
            "recovered": None
        }
    
    # Calculate running maximum
    running_max = prices.expanding().max()
    drawdown = (prices - running_max) / running_max * 100
    
    max_dd = drawdown.min()
    trough_idx = drawdown.idxmin()
    
    # Find the peak before the trough
    peak_idx = prices[:trough_idx].idxmax() if trough_idx else prices.index[0]
    peak_price = prices[peak_idx]
    
    # Find recovery date (when price returns to peak)
    after_trough = prices[trough_idx:]
    recovery_mask = after_trough >= peak_price
    
    if recovery_mask.any():
        recovery_idx = recovery_mask.idxmax()
        recovery_days = (recovery_idx - trough_idx).days
        recovered = True
    else:
        recovery_days = None
        recovered = False
    
    return {
        "max_drawdown_pct": round(max_dd, 2),
        "drawdown_peak_date": peak_idx.strftime("%Y-%m-%d") if peak_idx else None,
        "drawdown_trough_date": trough_idx.strftime("%Y-%m-%d") if trough_idx else None,
        "recovery_days": recovery_days,
        "recovered": recovered
    }


def calculate_volume_breakout_metrics(ohlcv: pd.DataFrame, rally_start: datetime) -> Dict:
    """
    Calculate volume breakout accuracy.
    
    Volume breakout = day with > 2x average volume
    Accuracy = % of volume breakouts that led to gains in next 5 days
    """
    if 'Volume' not in ohlcv.columns or len(ohlcv) < 50:
        return {
            "volume_breakout_days": None,
            "volume_breakout_accuracy": None,
            "avg_gain_after_breakout": None
        }
    
    # Split into before and during rally
    before_rally = ohlcv[ohlcv.index < rally_start]
    during_rally = ohlcv[ohlcv.index >= rally_start].copy()
    
    if len(before_rally) < 20 or len(during_rally) < 10:
        return {
            "volume_breakout_days": None,
            "volume_breakout_accuracy": None,
            "avg_gain_after_breakout": None
        }
    
    avg_volume = before_rally['Volume'].mean()
    breakout_threshold = avg_volume * 2
    
    # Find volume breakout days
    during_rally['is_breakout'] = during_rally['Volume'] > breakout_threshold
    during_rally['daily_return'] = during_rally['Close'].pct_change() * 100
    
    # Calculate 5-day forward return for each breakout
    during_rally['fwd_5d_return'] = during_rally['Close'].pct_change(5).shift(-5) * 100
    
    breakout_days = during_rally[during_rally['is_breakout']]
    
    if len(breakout_days) == 0:
        return {
            "volume_breakout_days": 0,
            "volume_breakout_accuracy": None,
            "avg_gain_after_breakout": None
        }
    
    # Calculate accuracy (% of breakouts with positive 5d return)
    valid_breakouts = breakout_days.dropna(subset=['fwd_5d_return'])
    if len(valid_breakouts) > 0:
        accuracy = (valid_breakouts['fwd_5d_return'] > 0).mean() * 100
        avg_gain = valid_breakouts['fwd_5d_return'].mean()
    else:
        accuracy = None
        avg_gain = None
    
    return {
        "volume_breakout_days": int(len(breakout_days)),
        "volume_breakout_accuracy": round(accuracy, 1) if accuracy else None,
        "avg_gain_after_breakout": round(avg_gain, 2) if avg_gain else None
    }


def calculate_sideways_periods(ohlcv: pd.DataFrame, rally_start: datetime) -> Dict:
    """
    Identify periods with < 5% movement over rolling windows.
    
    Sideways = 5+ consecutive days where max high-low range is < 5%
    """
    during_rally = ohlcv[ohlcv.index >= rally_start].copy()
    
    if len(during_rally) < 10:
        return {
            "sideways_periods": 0,
            "total_sideways_days": 0,
            "sideways_pct_of_period": 0,
            "max_sideways_streak": 0,
            "avg_sideways_length": 0
        }
    
    threshold = CONFIG["sideways_threshold"]
    min_days = CONFIG["sideways_min_days"]
    
    # Calculate rolling range as percentage
    during_rally['rolling_high'] = during_rally['High'].rolling(min_days).max()
    during_rally['rolling_low'] = during_rally['Low'].rolling(min_days).min()
    during_rally['rolling_range_pct'] = (
        (during_rally['rolling_high'] - during_rally['rolling_low']) / 
        during_rally['rolling_low'] * 100
    )
    
    # Mark sideways days
    during_rally['is_sideways'] = during_rally['rolling_range_pct'] < threshold
    
    # Find consecutive sideways periods
    sideways_periods = []
    current_streak = 0
    
    for is_sideways in during_rally['is_sideways'].fillna(False):
        if is_sideways:
            current_streak += 1
        else:
            if current_streak >= min_days:
                sideways_periods.append(current_streak)
            current_streak = 0
    
    if current_streak >= min_days:
        sideways_periods.append(current_streak)
    
    total_sideways = sum(sideways_periods)
    sideways_pct = (total_sideways / len(during_rally) * 100) if len(during_rally) > 0 else 0
    max_streak = max(sideways_periods) if sideways_periods else 0
    avg_length = np.mean(sideways_periods) if sideways_periods else 0
    
    return {
        "sideways_periods": len(sideways_periods),
        "total_sideways_days": total_sideways,
        "sideways_pct_of_period": round(sideways_pct, 1),
        "max_sideways_streak": max_streak,
        "avg_sideways_length": round(avg_length, 1)
    }


def calculate_volume_metrics(ohlcv: pd.DataFrame, rally_start: datetime) -> Dict:
    """Calculate volume-related metrics during the rally."""
    if 'Volume' not in ohlcv.columns or len(ohlcv) < 50:
        return {
            "avg_volume_before": None,
            "avg_volume_during": None,
            "volume_increase_pct": None,
            "volume_spike_days": None,
            "max_volume_spike": None
        }
    
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
    
    spike_threshold = avg_before * 2
    spike_days = len(during_rally[during_rally['Volume'] > spike_threshold])
    max_spike = (during_rally['Volume'].max() / avg_before) if avg_before > 0 else 0
    
    return {
        "avg_volume_before": int(avg_before),
        "avg_volume_during": int(avg_during),
        "volume_increase_pct": round(volume_increase, 1),
        "volume_spike_days": spike_days,
        "max_volume_spike": round(max_spike, 1)
    }


def analyze_trailing_stop(ohlcv: pd.DataFrame, rally_start: datetime, trailing_pcts: List[float] = [-10, -15, -20]) -> Dict:
    """Simulate trailing stop behavior during the rally."""
    during_rally = ohlcv[ohlcv.index >= rally_start].copy()
    
    if len(during_rally) < 10:
        return {f"trailing_{abs(int(p))}_triggers": None for p in trailing_pcts}
    
    results = {}
    
    for pct in trailing_pcts:
        close = during_rally['Close']
        running_high = close.expanding().max()
        drawdown_from_high = (close - running_high) / running_high * 100
        
        stopped = drawdown_from_high < pct
        trigger_count = stopped.sum()
        
        first_trigger = stopped.idxmax() if stopped.any() else None
        
        results[f"trailing_{abs(int(pct))}_triggers"] = trigger_count
        if first_trigger:
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
    """Divide rally into phases and calculate metrics for each."""
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
            
        close = phase_data['Close']
        phase_return = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100) if len(close) > 1 else 0
        
        daily_returns = close.pct_change().dropna()
        volatility = daily_returns.std() * 100 * np.sqrt(252)
        
        running_max = close.expanding().max()
        drawdown = ((close - running_max) / running_max * 100).min()
        
        if 'High' in phase_data.columns and 'Low' in phase_data.columns:
            avg_range = ((phase_data['High'] - phase_data['Low']) / phase_data['Close'] * 100).mean()
        else:
            avg_range = 0
        
        results[f"{phase_name}_return_pct"] = round(phase_return, 1)
        results[f"{phase_name}_volatility"] = round(volatility, 1)
        results[f"{phase_name}_max_drawdown"] = round(drawdown, 1)
        results[f"{phase_name}_avg_daily_range"] = round(avg_range, 2)
        results[f"{phase_name}_days"] = len(phase_data)
    
    return results


def calculate_trend_trajectory(ohlcv: pd.DataFrame, rally_start: datetime) -> Dict:
    """Calculate trend score trajectory at key points."""
    results = {}
    
    # Score at rally start
    score_start = calculate_trend_score_at_date(ohlcv, rally_start)
    results["score_at_rally_start"] = score_start.get("trend_score")
    
    # Score at +1 month
    one_month = rally_start + timedelta(days=30)
    if one_month <= ohlcv.index[-1]:
        score_1mo = calculate_trend_score_at_date(ohlcv, one_month)
        results["score_at_1mo"] = score_1mo.get("trend_score")
    else:
        results["score_at_1mo"] = None
    
    # Score at +3 months
    three_months = rally_start + timedelta(days=90)
    if three_months <= ohlcv.index[-1]:
        score_3mo = calculate_trend_score_at_date(ohlcv, three_months)
        results["score_at_3mo"] = score_3mo.get("trend_score")
    else:
        results["score_at_3mo"] = None
    
    # Current score
    score_current = calculate_trend_score_at_date(ohlcv, ohlcv.index[-1])
    results["score_current"] = score_current.get("trend_score")
    
    return results


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


def get_top_performers(n: int = 50, months: int = 12) -> pd.DataFrame:
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


def run_analysis_for_period(months: int, market_df: pd.DataFrame) -> pd.DataFrame:
    """Run deep analysis for a specific time period."""
    print(f"\n{'='*70}")
    print(f"ANALYZING {months}-MONTH PERIOD")
    print(f"{'='*70}")
    
    # Get top performers for this period
    top_df = get_top_performers(n=CONFIG['top_n'], months=months)
    if top_df.empty:
        print(f"[ERROR] Failed to get top performers for {months}mo")
        return pd.DataFrame()
    
    # Fetch detailed historical data
    print(f"\n[DATA] Fetching historical OHLCV data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30 + 300)  # Extra for indicators
    
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
        return pd.DataFrame()
    
    # Deep analysis for each winner
    print(f"\n[ANALYSIS] Running deep analysis on {len(top_df)} stocks...")
    results = []
    
    for idx, row in top_df.iterrows():
        ticker = row['ticker']
        print(f"   [{len(results)+1}/{len(top_df)}] {ticker}...")
        
        try:
            if len(top_tickers) == 1:
                ohlcv = historical.copy()
            else:
                ohlcv = historical[ticker].copy() if ticker in historical.columns.get_level_values(0) else pd.DataFrame()
            
            if ohlcv.empty or len(ohlcv) < 100:
                print(f"      [SKIP] Insufficient data")
                continue
            
            fundamentals = get_current_fundamentals(ticker, market_df)
            rally_start = identify_rally_start(ohlcv)
            
            # Basic metrics
            result = {
                "ticker": ticker,
                "name": fundamentals.get("name", ticker),
                "sector": fundamentals.get("sector", "Unknown"),
                f"return_{months}mo": round(row['return_pct'], 1),
                "rally_start_date": rally_start.strftime("%Y-%m-%d") if rally_start else None,
            }
            
            # Current fundamentals
            result["quality"] = fundamentals.get("quality")
            result["value"] = fundamentals.get("value")
            result["growth"] = fundamentals.get("growth")
            result["momentum"] = fundamentals.get("momentum")
            result["overall"] = fundamentals.get("overall")
            
            # Trend trajectory
            trajectory = calculate_trend_trajectory(ohlcv, rally_start)
            result.update(trajectory)
            
            # Trend transitions (NEW)
            transitions = calculate_trend_transitions(ohlcv, rally_start)
            result.update(transitions)
            
            # Max drawdown with recovery (ENHANCED)
            during_rally = ohlcv[ohlcv.index >= rally_start]
            if len(during_rally) > 10:
                dd_metrics = calculate_max_drawdown_with_recovery(during_rally['Close'])
                result.update(dd_metrics)
            
            # Volume metrics
            volume_metrics = calculate_volume_metrics(ohlcv, rally_start)
            result.update(volume_metrics)
            
            # Standardized Volume Signal (VPT + A/D) (M2)
            try:
                from utils.volume_analysis import get_combined_volume_signal
                # Use data from rally start onwards to check signal strength during rally
                rally_data = ohlcv[ohlcv.index >= rally_start]
                if len(rally_data) > 20:
                    vol_sig = get_combined_volume_signal(
                        rally_data['High'], rally_data['Low'], rally_data['Close'], rally_data['Volume']
                    )
                    result["volume_score_avg"] = vol_sig['combined_score']
                    result["volume_signal_type"] = vol_sig['combined_signal']
            except ImportError:
                pass
            
            # Volume breakout accuracy (NEW)
            breakout_metrics = calculate_volume_breakout_metrics(ohlcv, rally_start)
            result.update(breakout_metrics)
            
            # Sideways periods (NEW - <5% movement)
            sideways = calculate_sideways_periods(ohlcv, rally_start)
            result.update(sideways)
            
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
    
    return pd.DataFrame(results)


def generate_comparative_summary(period_results: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Generate comparative summary across all time periods."""
    summary_data = []
    
    metrics_to_compare = [
        ("Avg Return", lambda df, m: df[f"return_{m}mo"].mean(), "%"),
        ("Avg Max Drawdown", lambda df, m: df["max_drawdown_pct"].mean(), "%"),
        ("Avg Recovery Days", lambda df, m: df["recovery_days"].dropna().mean(), " days"),
        ("Avg Volume Increase", lambda df, m: df["volume_increase_pct"].mean(), "%"),
        ("Avg Volume Breakout Days", lambda df, m: df["volume_breakout_days"].mean(), ""),
        ("Vol Breakout Accuracy", lambda df, m: df["volume_breakout_accuracy"].dropna().mean(), "%"),
        ("Avg Sideways %", lambda df, m: df["sideways_pct_of_period"].mean(), "%"),
        ("Avg Trend Entries", lambda df, m: df["trend_entries"].mean(), ""),
        ("Avg Trend Exits", lambda df, m: df["trend_exits"].mean(), ""),
        ("Super Uptrend Entries", lambda df, m: df["super_uptrend_entries"].mean(), ""),
        ("Time in Trend", lambda df, m: df["time_in_trend_pct"].mean(), "%"),
        ("Avg Trend Duration", lambda df, m: df["avg_trend_duration_days"].mean(), " days"),
        ("Early Phase Return", lambda df, m: df["early_return_pct"].mean(), "%"),
        ("Mid Phase Return", lambda df, m: df["mid_return_pct"].mean(), "%"),
        ("Late Phase Return", lambda df, m: df["late_return_pct"].mean(), "%"),
        ("Trailing 10% Triggers", lambda df, m: df["trailing_10_triggers"].mean(), ""),
        ("Trailing 15% Triggers", lambda df, m: df["trailing_15_triggers"].mean(), ""),
        ("Trailing 20% Triggers", lambda df, m: df["trailing_20_triggers"].mean(), ""),
        ("Avg Quality Score", lambda df, m: df["quality"].mean(), ""),
        ("Avg Value Score", lambda df, m: df["value"].mean(), ""),
        ("Avg Growth Score", lambda df, m: df["growth"].mean(), ""),
        ("Avg Momentum Score", lambda df, m: df["momentum"].mean(), ""),
    ]
    
    for metric_name, calc_func, suffix in metrics_to_compare:
        row = {"Metric": metric_name}
        for months, df in period_results.items():
            if df.empty:
                row[f"{months}mo"] = "N/A"
            else:
                try:
                    value = calc_func(df, months)
                    if pd.isna(value):
                        row[f"{months}mo"] = "N/A"
                    else:
                        row[f"{months}mo"] = f"{value:.1f}{suffix}"
                except:
                    row[f"{months}mo"] = "N/A"
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)


def run_multi_period_analysis():
    """Main function to run analysis across all time periods."""
    print("\n" + "="*70)
    print("STRATEGY LEARNER v3 - MULTI-PERIOD DEEP ANALYSIS")
    print("="*70)
    print(f"Analyzing top {CONFIG['top_n']} performers across {CONFIG['time_periods']} month periods")
    print()
    
    # Load market data once for fundamentals
    print("[DATA] Loading market data for fundamentals...")
    market_df = batch_fetch_tickers(TICKERS)
    
    from utils.historical_tracker import calculate_multi_period_returns
    scored_data = []
    for _, row in market_df.iterrows():
        data = row.to_dict()
        returns, _ = calculate_multi_period_returns(row['ticker'])
        data.update(returns)
        trends = calculate_trend_metrics(data)
        data.update(trends)
        scores = calculate_scores(data)
        data.update(scores)
        scored_data.append(data)
    market_df = pd.DataFrame(scored_data)
    
    # Run analysis for each time period
    period_results = {}
    
    for months in CONFIG['time_periods']:
        results_df = run_analysis_for_period(months, market_df)
        if not results_df.empty:
            output_file = f"{CONFIG['output_prefix']}_{months}mo_results.csv"
            results_df.to_csv(output_file, index=False)
            print(f"\n[SAVED] {output_file}")
            period_results[months] = results_df
    
    # Generate comparative summary
    print("\n" + "="*70)
    print("GENERATING COMPARATIVE SUMMARY")
    print("="*70)
    
    summary_df = generate_comparative_summary(period_results)
    summary_file = f"{CONFIG['output_prefix']}_comparative_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\n[SAVED] {summary_file}")
    
    # Print key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS ACROSS TIME PERIODS")
    print("="*70)
    
    print("\n" + summary_df.to_string(index=False))
    
    # Analysis insights
    print("\n" + "-"*50)
    print("OBSERVATIONS")
    print("-"*50)
    
    for metric_name in ["Avg Max Drawdown", "Avg Sideways %", "Time in Trend", "Vol Breakout Accuracy"]:
        row = summary_df[summary_df["Metric"] == metric_name]
        if not row.empty:
            print(f"\n{metric_name}:")
            for months in CONFIG['time_periods']:
                val = row[f"{months}mo"].values[0]
                print(f"   {months}mo: {val}")
    
    print("\n")
    return period_results, summary_df


if __name__ == "__main__":
    run_multi_period_analysis()
