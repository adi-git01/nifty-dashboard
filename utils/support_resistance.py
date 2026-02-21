"""
Support & Resistance Calculator
================================
Calculates dynamic S/R levels using 20-day swing highs/lows.
Used by Proposal 5 (S/R Enhanced Momentum).
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


def find_swing_highs(prices: pd.Series, window: int = 5) -> pd.Series:
    """
    Find swing highs - local maxima with 'window' bars on each side.
    
    Args:
        prices: Series of prices (typically 'High' column)
        window: Number of bars on each side to confirm swing
        
    Returns:
        Series with NaN except at swing high points
    """
    swing_highs = pd.Series(index=prices.index, dtype=float)
    
    for i in range(window, len(prices) - window):
        is_swing = True
        center = prices.iloc[i]
        
        # Check if center is higher than all surrounding bars
        for j in range(1, window + 1):
            if prices.iloc[i - j] >= center or prices.iloc[i + j] >= center:
                is_swing = False
                break
        
        if is_swing:
            swing_highs.iloc[i] = center
    
    return swing_highs


def find_swing_lows(prices: pd.Series, window: int = 5) -> pd.Series:
    """
    Find swing lows - local minima with 'window' bars on each side.
    """
    swing_lows = pd.Series(index=prices.index, dtype=float)
    
    for i in range(window, len(prices) - window):
        is_swing = True
        center = prices.iloc[i]
        
        # Check if center is lower than all surrounding bars
        for j in range(1, window + 1):
            if prices.iloc[i - j] <= center or prices.iloc[i + j] <= center:
                is_swing = False
                break
        
        if is_swing:
            swing_lows.iloc[i] = center
    
    return swing_lows


def calculate_support_resistance(
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series,
    lookback: int = 20,
    swing_window: int = 5
) -> Dict[str, float]:
    """
    Calculate nearest support and resistance levels using 20-day swing points.
    
    Args:
        high: High prices series
        low: Low prices series  
        close: Close prices series
        lookback: Number of days to look back for swing points
        swing_window: Window for swing point detection
        
    Returns:
        Dict with:
        - nearest_support: Closest support level
        - nearest_resistance: Closest resistance level
        - dist_to_support_pct: % distance to support
        - dist_to_resistance_pct: % distance to resistance
        - sr_position: 'near_support', 'near_resistance', 'middle'
    """
    if len(close) < lookback:
        return {
            "nearest_support": None,
            "nearest_resistance": None,
            "dist_to_support_pct": None,
            "dist_to_resistance_pct": None,
            "sr_position": "unknown"
        }
    
    current_price = close.iloc[-1]
    
    # Get recent segment for swing analysis
    recent_high = high.iloc[-lookback:]
    recent_low = low.iloc[-lookback:]
    
    # Find swing points
    swing_highs = find_swing_highs(recent_high, window=swing_window)
    swing_lows = find_swing_lows(recent_low, window=swing_window)
    
    # Get valid swing levels (non-NaN)
    resistance_levels = swing_highs.dropna().tolist()
    support_levels = swing_lows.dropna().tolist()
    
    # Find nearest resistance (above current price)
    resistance_above = [r for r in resistance_levels if r > current_price]
    nearest_resistance = min(resistance_above) if resistance_above else (
        max(resistance_levels) if resistance_levels else current_price * 1.1
    )
    
    # Find nearest support (below current price)
    support_below = [s for s in support_levels if s < current_price]
    nearest_support = max(support_below) if support_below else (
        min(support_levels) if support_levels else current_price * 0.9
    )
    
    # Calculate distances
    dist_to_support = ((current_price - nearest_support) / current_price) * 100
    dist_to_resistance = ((nearest_resistance - current_price) / current_price) * 100
    
    # Determine position
    if dist_to_support < 3:
        position = "near_support"
    elif dist_to_resistance < 3:
        position = "near_resistance"
    else:
        position = "middle"
    
    return {
        "nearest_support": round(nearest_support, 2),
        "nearest_resistance": round(nearest_resistance, 2),
        "dist_to_support_pct": round(dist_to_support, 2),
        "dist_to_resistance_pct": round(dist_to_resistance, 2),
        "sr_position": position
    }


def is_breakout_above_resistance(
    close: pd.Series,
    resistance: float,
    lookback_for_confirm: int = 3
) -> bool:
    """
    Check if price has broken above resistance.
    Confirms with lookback_for_confirm days of closes above resistance.
    """
    if resistance is None or len(close) < lookback_for_confirm:
        return False
    
    recent_closes = close.iloc[-lookback_for_confirm:]
    return all(c > resistance for c in recent_closes)


def is_bounce_from_support(
    close: pd.Series,
    low: pd.Series,
    support: float,
    bounce_threshold_pct: float = 2.0
) -> bool:
    """
    Check if price has bounced from support.
    - Recent low touched support (within threshold)
    - Current close is above the low
    """
    if support is None or len(close) < 5:
        return False
    
    recent_low = low.iloc[-5:].min()
    current_close = close.iloc[-1]
    
    # Low touched support (within threshold)
    touched_support = abs((recent_low - support) / support * 100) < bounce_threshold_pct
    
    # Price bounced up from that low
    bounced = current_close > recent_low * 1.01  # At least 1% above low
    
    return touched_support and bounced


def calculate_sr_signals(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    lookback: int = 20,
    volume: pd.Series = None
) -> Dict[str, any]:
    """
    Calculate S/R levels and trading signals.
    Now with optional volume confirmation.
    
    Args:
        high, low, close: Price series
        lookback: Days for S/R calculation
        volume: Optional volume series for confirmation
    
    Returns:
        Dict with S/R levels, signal flags, and volume confirmation
    """
    sr_levels = calculate_support_resistance(high, low, close, lookback)
    
    # Check for breakout
    breakout = is_breakout_above_resistance(
        close, 
        sr_levels.get("nearest_resistance")
    )
    
    # Check for bounce
    bounce = is_bounce_from_support(
        close,
        low,
        sr_levels.get("nearest_support")
    )
    
    sr_levels["breakout_signal"] = breakout
    sr_levels["bounce_signal"] = bounce
    sr_levels["entry_signal"] = breakout or bounce
    
    # Volume confirmation (if volume provided)
    if volume is not None and len(volume) >= 20:
        try:
            from utils.volume_analysis import check_volume_confirmation, calculate_volume_trend_score
            
            # Get volume metrics
            vol_trend = calculate_volume_trend_score(close, volume, lookback)
            sr_levels["volume_ratio"] = vol_trend["volume_ratio"]
            sr_levels["volume_score"] = vol_trend["volume_score"]
            sr_levels["obv_trend"] = vol_trend["obv_trend"]
            sr_levels["obv_divergence"] = vol_trend["obv_divergence"]
            
            # Check volume confirmation for signals
            if breakout:
                breakout_vol = check_volume_confirmation(close, volume, "breakout")
                sr_levels["breakout_confirmed"] = breakout_vol["confirmed"]
                sr_levels["breakout_confidence"] = breakout_vol["confidence"]
                sr_levels["entry_signal"] = breakout_vol["confirmed"]  # Only valid if volume confirms
            
            if bounce:
                bounce_vol = check_volume_confirmation(close, volume, "bounce")
                sr_levels["bounce_confirmed"] = bounce_vol["confirmed"]
                sr_levels["bounce_confidence"] = bounce_vol["confidence"]
                sr_levels["entry_signal"] = bounce_vol["confirmed"] or sr_levels.get("entry_signal", False)
            
        except ImportError:
            # Volume module not available, skip confirmation
            sr_levels["volume_ratio"] = 1.0
            sr_levels["volume_score"] = 5
            sr_levels["obv_trend"] = "UNKNOWN"
    else:
        sr_levels["volume_ratio"] = 1.0
        sr_levels["volume_score"] = 5
        sr_levels["obv_trend"] = "UNKNOWN"
    
    return sr_levels
