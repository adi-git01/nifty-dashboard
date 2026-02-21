
import pandas as pd
import numpy as np

def normalize(value, min_val, max_val):
    """Vectorized normalization to 0-10 scale."""
    if value is None:
        return 5
    
    # Handle scalar or series
    if isinstance(value, (int, float)):
        if value < min_val: return 0
        if value > max_val: return 10
        return ((value - min_val) / (max_val - min_val)) * 10
    
    # Handle pandas Series
    result = ((value - min_val) / (max_val - min_val)) * 10
    return result.clip(0, 10)

def calculate_historical_scores(df):
    """
    Calculates historical Trend, Momentum, and Volume scores for the entire dataframe.
    
    Args:
        df: DataFrame with 'Close', 'High', 'Low', 'Volume' columns (datetime index)
        
    Returns:
        DataFrame with added score columns:
        - trend_score_hist (0-100)
        - momentum_score_hist (0-10)
        - volume_score_hist (0-10)
    """
    if df.empty or len(df) < 200:
        return df
    
    # Work on a copy
    d = df.copy()
    
    # ==========================================
    # 1. HISTORICAL TREND SCORE (0-100)
    # ==========================================
    # Moving Averages
    d['ma50'] = d['Close'].ewm(span=50, adjust=False).mean()
    d['ma200'] = d['Close'].ewm(span=200, adjust=False).mean()
    
    # 52-Week High/Low (Rolling 252 days)
    d['high52'] = d['Close'].rolling(window=252, min_periods=50).max()
    d['low52'] = d['Close'].rolling(window=252, min_periods=50).min()
    
    # Base Score
    trend = pd.Series(50, index=d.index)
    
    # MA Logic
    trend += np.where(d['Close'] > d['ma50'], 15, -10)
    trend += np.where(d['Close'] > d['ma200'], 15, -15)
    # Golden Cross / Death Cross
    trend += np.where(d['ma50'] > d['ma200'], 10, -5)
    
    # 52W Position Logic
    range_52 = d['high52'] - d['low52']
    # Avoid division by zero
    range_52 = range_52.replace(0, np.nan)
    pos_52 = (d['Close'] - d['low52']) / range_52
    
    # Position score (-15 to +15)
    pos_score = (pos_52 - 0.5) * 30
    trend += pos_score.fillna(0)
    
    # Near High Bonus
    dist_high = (d['Close'] - d['high52']) / d['high52'] * 100
    trend += np.where(dist_high > -5, 10, 0)
    trend += np.where(dist_high < -30, -10, 0)
    
    d['trend_score_hist'] = trend.clip(0, 100)
    
    
    # ==========================================
    # 2. HISTORICAL MOMENTUM SCORE (0-10)
    # ==========================================
    # Returns
    r1w = d['Close'].pct_change(5) * 100
    r1m = d['Close'].pct_change(21) * 100
    r3m = d['Close'].pct_change(63) * 100
    
    # Normalized Scores
    s_1w = normalize(r1w, -5, 10)
    s_1m = normalize(r1m, -8, 20)
    s_3m = normalize(r3m, -10, 40)
    
    # Trend Score contribution (normalized 15-90 -> 0-10)
    s_trend_pos = normalize(d['trend_score_hist'], 15, 90)
    
    # Weighted Sum
    mom = (s_1w * 0.15) + (s_1m * 0.30) + (s_3m * 0.30) + (s_trend_pos * 0.25)
    
    # Consistency Bonus
    bonus = ((r1w > 0) & (r1m > 0) & (r3m > 0)).astype(int) * 1.5
    mom += bonus
    
    # High Trend Bonus
    high_trend_bonus = (d['trend_score_hist'] >= 80).astype(int) * 0.5
    mom += high_trend_bonus
    
    d['momentum_score_hist'] = mom.clip(0, 10)
    
    
    # ==========================================
    # 3. HISTORICAL VOLUME SCORE (0-10)
    # ==========================================
    # Volume Ratio
    avg_vol_20 = d['Volume'].rolling(window=20).mean()
    vol_ratio = d['Volume'] / avg_vol_20
    
    # Base Volume Score
    vol_score = pd.Series(5.0, index=d.index)
    
    # Ratio Scoring
    vol_score += np.where(vol_ratio > 1.5, 2, 0)
    vol_score += np.where((vol_ratio > 1.2) & (vol_ratio <= 1.5), 1, 0)
    vol_score += np.where(vol_ratio < 0.7, -1, 0)
    
    # OBV Trend Detection (Rolling Slope)
    # Simple approximation: 10-day OBV slope
    direction = np.sign(d['Close'].diff())
    obv = (direction * d['Volume']).cumsum()
    
    # Normalized OBV Slope (Rolling 10d)
    obv_min = obv.rolling(10).min()
    obv_max = obv.rolling(10).max()
    obv_range = obv_max - obv_min
    obv_change = obv.diff(10)
    
    obv_slope = obv_change / obv_range.replace(0, np.nan)
    
    # Map slope to score
    vol_score += np.where(obv_slope > 0.2, 2, 0) # Accumulation
    vol_score += np.where(obv_slope < -0.2, -2, 0) # Distribution
    
    # Divergence (simplified for vectorized)
    # Price Up + OBV Down = Bearish (-2)
    price_slope = d['Close'].pct_change(10)
    bearish_div = (price_slope > 0.05) & (obv_slope < -0.2)
    bullish_div = (price_slope < -0.05) & (obv_slope > 0.2)
    
    vol_score += np.where(bearish_div, -2, 0)
    vol_score += np.where(bullish_div, 1, 0)
    
    d['volume_score_hist'] = vol_score.clip(0, 10)
    
    return d


def detect_divergences(df, price_col='Close', indicator_col='momentum_score_hist', window=5):
    """
    Detects bullish and bearish divergences.
    Returns Series with divergence price points (NaN where no divergence).
    """
    if df.empty or len(df) < window * 2:
        return pd.DataFrame({'div_bull': pd.Series(dtype=float), 'div_bear': pd.Series(dtype=float)})

    # Work on a view/copy to detect extrema
    # Simple rolling min/max detection
    # Using window=5 (approx 1 week) to find local pivot points
    
    # 1. Identify Local Extrema
    # Center-aligned rolling window to find peaks/valleys
    d = df.copy()
    d['min_local'] = d[price_col].rolling(window=window, center=True).min()
    d['max_local'] = d[price_col].rolling(window=window, center=True).max()
    
    # Indices where local extrema occur
    is_min = (d[price_col] == d['min_local'])
    is_max = (d[price_col] == d['max_local'])
    
    min_indices = d.index[is_min]
    max_indices = d.index[is_max]
    
    d['div_bull'] = np.nan
    d['div_bear'] = np.nan
    
    # 2. Bullish Divergence (Price Lower Low, Indicator Higher Low)
    # We iterate through consecutive local lows
    # Vectorization is hard for "consecutive" logic without strict indexing, loop is safer for small datasets
    for i in range(1, len(min_indices)):
        curr_idx = min_indices[i]
        prev_idx = min_indices[i-1]
        
        # Ensure points are not too far apart (e.g., within 6 months)
        # if (curr_idx - prev_idx).days > 180: continue
        
        p_curr = d.loc[curr_idx, price_col]
        p_prev = d.loc[prev_idx, price_col]
        
        # Check Indicator (Momentum Score)
        i_curr = d.loc[curr_idx, indicator_col]
        i_prev = d.loc[prev_idx, indicator_col]
        
        # Condition: Price Lower, Indicator Higher
        # Add buffer: Price needs to be significantly lower? No, strict LD is usually fine.
        if p_curr < p_prev and i_curr > i_prev:
             d.loc[curr_idx, 'div_bull'] = p_curr
             
    # 3. Bearish Divergence (Price Higher High, Indicator Lower High)
    for i in range(1, len(max_indices)):
        curr_idx = max_indices[i]
        prev_idx = max_indices[i-1]
        
        p_curr = d.loc[curr_idx, price_col]
        p_prev = d.loc[prev_idx, price_col]
        
        i_curr = d.loc[curr_idx, indicator_col]
        i_prev = d.loc[prev_idx, indicator_col]
        
        if p_curr > p_prev and i_curr < i_prev:
            d.loc[curr_idx, 'div_bear'] = p_curr
            
    return d[['div_bull', 'div_bear']]
