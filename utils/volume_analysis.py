"""
Volume Analysis Module
======================
Provides volume confirmation signals for momentum and S/R strategies.

Features:
- Volume Ratio (current vs 20-day average)
- OBV (On-Balance Volume) trend detection
- Volume confirmation for breakouts/bounces
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


def calculate_volume_ratio(volume: pd.Series, lookback: int = 20) -> float:
    """
    Calculate current volume relative to average.
    
    Args:
        volume: Volume series
        lookback: Days for average calculation
        
    Returns:
        Volume ratio (>1 means above average, <1 means below)
    """
    if len(volume) < lookback:
        return 1.0
    
    current_vol = volume.iloc[-1]
    avg_vol = volume.iloc[-lookback:].mean()
    
    if pd.isna(current_vol) or pd.isna(avg_vol) or avg_vol == 0:
        return 1.0
    
    return current_vol / avg_vol


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).
    
    OBV adds volume on up days and subtracts on down days.
    Rising OBV = Accumulation, Falling OBV = Distribution
    """
    if len(close) < 2 or len(volume) < 2:
        return pd.Series([0], index=close.index[-1:])
    
    # Calculate price direction
    direction = np.sign(close.diff())
    
    # OBV = cumulative sum of (direction * volume)
    obv = (direction * volume).cumsum()
    
    return obv


def calculate_vpt(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate Volume Price Trend (VPT).
    
    VPT weights volume by percentage price change, making it more
    sensitive to actual price momentum than OBV.
    
    Formula: VPT = cumsum(((close - prev_close) / prev_close) * volume)
    """
    if len(close) < 2 or len(volume) < 2:
        return pd.Series([0], index=close.index[-1:])
    
    try:
        # Calculate percentage change
        pct_change = close.pct_change()
        
        # VPT = cumulative sum of (pct_change * volume)
        vpt = (pct_change * volume).cumsum()
        
        return vpt.fillna(0)
    except:
        return pd.Series([0] * len(close), index=close.index)


def calculate_ad_line(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate Accumulation/Distribution Line (A/D).
    
    A/D uses the full price range (high-low-close) to determine
    money flow direction. More accurate than OBV for detecting
    institutional accumulation/distribution.
    
    Formula: 
    - Money Flow Multiplier = ((close - low) - (high - close)) / (high - low)
    - Money Flow Volume = MFM * volume
    - A/D = cumsum(Money Flow Volume)
    """
    if len(close) < 2 or len(volume) < 2:
        return pd.Series([0], index=close.index[-1:])
    
    try:
        # Calculate Money Flow Multiplier
        high_low_range = high - low
        
        # Avoid division by zero
        high_low_range = high_low_range.replace(0, np.nan)
        
        mf_multiplier = ((close - low) - (high - close)) / high_low_range
        mf_multiplier = mf_multiplier.fillna(0)
        
        # Money Flow Volume
        mf_volume = mf_multiplier * volume
        
        # A/D Line = cumulative sum
        ad_line = mf_volume.cumsum()
        
        return ad_line.fillna(0)
    except:
        return pd.Series([0] * len(close), index=close.index)


def get_combined_volume_signal(
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series, 
    volume: pd.Series,
    lookback: int = 20,
    vpt_weight: float = 0.70,
    ad_weight: float = 0.30
) -> Dict[str, any]:
    """
    Calculate combined VPT + A/D volume signal.
    
    Args:
        high, low, close: Price series
        volume: Volume series
        lookback: Days for trend calculation
        vpt_weight: Weight for VPT (default 70%)
        ad_weight: Weight for A/D (default 30%)
        
    Returns:
        Dict with combined signal and individual components
    """
    if len(close) < lookback or len(volume) < lookback:
        return {
            "combined_signal": "NEUTRAL",
            "combined_score": 5,
            "vpt_trend": "NEUTRAL",
            "vpt_slope": 0,
            "ad_trend": "NEUTRAL",
            "ad_slope": 0
        }
    
    try:
        # Calculate VPT trend
        vpt = calculate_vpt(close, volume)
        vpt_recent = vpt.iloc[-lookback:]
        vpt_start = float(vpt_recent.iloc[0])
        vpt_end = float(vpt_recent.iloc[-1])
        
        vpt_range = float(vpt_recent.max() - vpt_recent.min())
        if vpt_range > 0 and not pd.isna(vpt_start) and not pd.isna(vpt_end):
            vpt_slope = (vpt_end - vpt_start) / vpt_range
        else:
            vpt_slope = 0
        
        # Calculate A/D trend
        ad = calculate_ad_line(high, low, close, volume)
        ad_recent = ad.iloc[-lookback:]
        ad_start = float(ad_recent.iloc[0])
        ad_end = float(ad_recent.iloc[-1])
        
        ad_range = float(ad_recent.max() - ad_recent.min())
        if ad_range > 0 and not pd.isna(ad_start) and not pd.isna(ad_end):
            ad_slope = (ad_end - ad_start) / ad_range
        else:
            ad_slope = 0
        
        # Determine individual trends
        vpt_trend = "ACCUMULATION" if vpt_slope > 0.1 else ("DISTRIBUTION" if vpt_slope < -0.1 else "NEUTRAL")
        ad_trend = "ACCUMULATION" if ad_slope > 0.1 else ("DISTRIBUTION" if ad_slope < -0.1 else "NEUTRAL")
        
        # Combine signals with weights (70% VPT, 30% A/D)
        combined_slope = (vpt_weight * vpt_slope) + (ad_weight * ad_slope)
        
        if combined_slope > 0.1:
            combined_signal = "ACCUMULATION"
        elif combined_slope < -0.1:
            combined_signal = "DISTRIBUTION"
        else:
            combined_signal = "NEUTRAL"
        
        # Calculate combined score (0-10)
        # Base of 5, +/- based on combined slope
        combined_score = 5 + int(combined_slope * 5)
        combined_score = max(0, min(10, combined_score))
        
        return {
            "combined_signal": combined_signal,
            "combined_score": combined_score,
            "combined_slope": round(combined_slope, 3),
            "vpt_trend": vpt_trend,
            "vpt_slope": round(vpt_slope, 3),
            "ad_trend": ad_trend,
            "ad_slope": round(ad_slope, 3)
        }
    except Exception as e:
        return {
            "combined_signal": "NEUTRAL",
            "combined_score": 5,
            "vpt_trend": "NEUTRAL",
            "vpt_slope": 0,
            "ad_trend": "NEUTRAL",
            "ad_slope": 0
        }


def detect_obv_trend(close: pd.Series, volume: pd.Series, lookback: int = 20) -> Dict[str, any]:
    """
    Detect OBV trend to identify accumulation/distribution.
    
    Returns:
        Dict with:
        - obv_trend: "ACCUMULATION", "DISTRIBUTION", or "NEUTRAL"
        - obv_slope: Rate of change
        - obv_divergence: True if price and OBV diverge
    """
    if len(close) < lookback or len(volume) < lookback:
        return {
            "obv_trend": "NEUTRAL",
            "obv_slope": 0,
            "obv_divergence": False
        }
    
    obv = calculate_obv(close, volume)
    
    # Check for NaN
    if obv.isna().all():
        return {
            "obv_trend": "NEUTRAL",
            "obv_slope": 0,
            "obv_divergence": False
        }
    
    # Calculate OBV slope (normalized)
    obv_recent = obv.iloc[-lookback:]
    obv_start = obv_recent.iloc[0]
    obv_end = obv_recent.iloc[-1]
    
    if pd.isna(obv_start) or pd.isna(obv_end):
        obv_slope = 0
    else:
        obv_range = obv_recent.max() - obv_recent.min()
        if obv_range > 0:
            obv_slope = (obv_end - obv_start) / obv_range
        else:
            obv_slope = 0
    
    # Calculate price slope for divergence detection
    price_start = close.iloc[-lookback]
    price_end = close.iloc[-1]
    
    if pd.isna(price_start) or pd.isna(price_end) or price_start == 0:
        price_slope = 0
    else:
        price_slope = (price_end - price_start) / price_start
    
    # Determine trend
    if obv_slope > 0.1:
        obv_trend = "ACCUMULATION"
    elif obv_slope < -0.1:
        obv_trend = "DISTRIBUTION"
    else:
        obv_trend = "NEUTRAL"
    
    # Detect divergence (price up but OBV down, or vice versa)
    obv_divergence = (
        (price_slope > 0.05 and obv_slope < -0.1) or  # Bearish divergence
        (price_slope < -0.05 and obv_slope > 0.1)     # Bullish divergence
    )
    
    return {
        "obv_trend": obv_trend,
        "obv_slope": round(obv_slope, 3),
        "obv_divergence": obv_divergence,
        "divergence_type": "BEARISH" if (price_slope > 0 and obv_slope < 0) else 
                          "BULLISH" if (price_slope < 0 and obv_slope > 0) else "NONE"
    }


def calculate_volume_trend_score(
    close: pd.Series, 
    volume: pd.Series,
    lookback: int = 20
) -> Dict[str, any]:
    """
    Calculate volume-based trend score component.
    
    Returns:
        Dict with volume metrics and score contribution
    """
    volume_ratio = calculate_volume_ratio(volume, lookback)
    obv_data = detect_obv_trend(close, volume, lookback)
    
    # Volume score contribution (0-10)
    volume_score = 5  # Neutral baseline
    
    # Volume ratio impact
    if volume_ratio > 1.5:
        volume_score += 2  # High volume confirmation
    elif volume_ratio > 1.2:
        volume_score += 1
    elif volume_ratio < 0.7:
        volume_score -= 1  # Low volume = weak move
    
    # OBV trend impact
    if obv_data["obv_trend"] == "ACCUMULATION":
        volume_score += 2
    elif obv_data["obv_trend"] == "DISTRIBUTION":
        volume_score -= 2
    
    # Divergence penalty
    if obv_data["obv_divergence"]:
        if obv_data["divergence_type"] == "BEARISH":
            volume_score -= 2  # Price up but OBV down = warning
        elif obv_data["divergence_type"] == "BULLISH":
            volume_score += 1  # Price down but OBV up = potential reversal
    
    volume_score = max(0, min(10, volume_score))
    
    return {
        "volume_ratio": round(volume_ratio, 2),
        "volume_score": volume_score,
        **obv_data
    }


def check_volume_confirmation(
    close: pd.Series,
    volume: pd.Series,
    signal_type: str,
    min_ratio: float = 1.3
) -> Dict[str, any]:
    """
    Check if volume confirms a breakout or bounce signal.
    
    Args:
        close: Close prices
        volume: Volume data
        signal_type: "breakout" or "bounce"
        min_ratio: Minimum volume ratio for confirmation
        
    Returns:
        Dict with confirmation status and metrics
    """
    volume_ratio = calculate_volume_ratio(volume, 20)
    obv_data = detect_obv_trend(close, volume, 10)  # Shorter lookback for signals
    
    confirmed = False
    confidence = "LOW"
    
    if signal_type == "breakout":
        # Breakout needs high volume
        if volume_ratio >= min_ratio:
            if obv_data["obv_trend"] == "ACCUMULATION":
                confirmed = True
                confidence = "HIGH"
            else:
                confirmed = True
                confidence = "MEDIUM"
        elif volume_ratio >= 1.0 and obv_data["obv_trend"] == "ACCUMULATION":
            confirmed = True
            confidence = "MEDIUM"
    
    elif signal_type == "bounce":
        # Bounce needs volume pickup from support
        if volume_ratio >= min_ratio * 0.8:  # Slightly lower threshold
            if obv_data["obv_trend"] != "DISTRIBUTION":
                confirmed = True
                confidence = "MEDIUM" if volume_ratio < min_ratio else "HIGH"
    
    return {
        "confirmed": confirmed,
        "confidence": confidence,
        "volume_ratio": round(volume_ratio, 2),
        "obv_trend": obv_data["obv_trend"],
        "reason": f"Vol: {volume_ratio:.1f}x avg, OBV: {obv_data['obv_trend']}"
    }


def get_volume_signals(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series
) -> Dict[str, any]:
    """
    Get all volume-related signals for a stock.
    
    Returns comprehensive volume analysis for strategy use.
    """
    if len(close) < 20 or len(volume) < 20:
        return {
            "volume_ratio": 1.0,
            "volume_score": 5,
            "obv_trend": "NEUTRAL",
            "obv_slope": 0,
            "obv_divergence": False,
            "breakout_confirmed": False,
            "bounce_confirmed": False
        }
    
    # Basic volume analysis
    trend_data = calculate_volume_trend_score(close, volume)
    
    # Check for breakout/bounce confirmation
    # (Would be used in conjunction with S/R signals)
    breakout_check = check_volume_confirmation(close, volume, "breakout")
    bounce_check = check_volume_confirmation(close, volume, "bounce")
    
    return {
        **trend_data,
        "breakout_confirmed": breakout_check["confirmed"],
        "breakout_confidence": breakout_check["confidence"],
        "bounce_confirmed": bounce_check["confirmed"],
        "bounce_confidence": bounce_check["confidence"]
    }
