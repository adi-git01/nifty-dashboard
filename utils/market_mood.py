"""
Market Mood Indicator
=====================
Tracks market-wide health metrics over time:
1. Strong Momentum (stocks with trend_score >= 80)
2. Total Uptrends (stocks with trend_signal containing 'UPTREND')
3. Avg Trend Score (mean of all trend scores)
4. Breakout Alerts (stocks near 52-week high)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

MOOD_FILE = "data/market_mood_history.csv"

def ensure_data_dir():
    os.makedirs("data", exist_ok=True)

def calculate_mood_metrics(df):
    """
    Calculate the 4 market mood metrics from current market data.
    Returns dict with date and 4 metrics.
    """
    if df.empty:
        return None
    
    # 1. Strong Momentum: trend_score >= 80
    strong_momentum = len(df[df.get('trend_score', pd.Series([0]*len(df))) >= 80])
    
    # 2. Total Uptrends: trend_signal contains 'UPTREND'
    if 'trend_signal' in df.columns:
        total_uptrends = len(df[df['trend_signal'].str.contains('UPTREND', na=False)])
    else:
        total_uptrends = 0
    
    # 3. Avg Trend Score
    if 'trend_score' in df.columns:
        avg_trend_score = df['trend_score'].mean()
    else:
        avg_trend_score = 50
    
    # 4. Breakout Alerts: within 5% of 52-week high
    if 'dist_52w' in df.columns:
        breakout_alerts = len(df[df['dist_52w'] >= -5])
    elif 'fiftyTwoWeekHigh' in df.columns and 'price' in df.columns:
        df['_dist'] = ((df['price'] - df['fiftyTwoWeekHigh']) / df['fiftyTwoWeekHigh']) * 100
        breakout_alerts = len(df[df['_dist'] >= -5])
    else:
        breakout_alerts = 0
    
    return {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'strong_momentum': strong_momentum,
        'total_uptrends': total_uptrends,
        'avg_trend_score': round(avg_trend_score, 1),
        'breakout_alerts': breakout_alerts
    }

def save_mood_snapshot(metrics):
    """
    Appends today's metrics to the history file.
    Avoids duplicate entries for the same date.
    """
    if not metrics:
        return
    
    ensure_data_dir()
    
    # Load existing
    if os.path.exists(MOOD_FILE):
        history = pd.read_csv(MOOD_FILE)
        # Remove today if exists (we'll re-add with fresh data)
        history = history[history['date'] != metrics['date']]
    else:
        history = pd.DataFrame()
    
    # Append new
    new_row = pd.DataFrame([metrics])
    history = pd.concat([history, new_row], ignore_index=True)
    
    # Keep only last 365 days (1 year)
    history['date'] = pd.to_datetime(history['date'])
    cutoff = datetime.now() - timedelta(days=365)
    history = history[history['date'] >= cutoff]
    history['date'] = history['date'].dt.strftime('%Y-%m-%d')
    
    history.to_csv(MOOD_FILE, index=False)

def load_mood_history():
    """
    Loads the mood history for charting.
    Returns DataFrame with date and 4 metrics.
    """
    if not os.path.exists(MOOD_FILE):
        return pd.DataFrame()
    
    df = pd.read_csv(MOOD_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df

def replay_historical_mood(price_history_df, lookback_days=365):
    """
    'Time Travel' - Replay historical price data to reconstruct mood metrics.
    This uses the stored 1-year price history to calculate what the metrics
    would have been on each past date.
    
    Args:
        price_history_df: MultiIndex DataFrame from yf.download (ticker, OHLCV)
        lookback_days: How many days to replay
    
    Returns:
        DataFrame with historical mood metrics
    """
    # This is a complex operation. For MVP, we'll just use saved snapshots.
    # Full replay would require recalculating trend scores for each day.
    # TODO: Implement full replay logic
    return load_mood_history()

def chart_market_mood(history_df):
    """
    Creates a Plotly chart with 4 mood metrics over time.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    if history_df.empty:
        return None
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=history_df['date'], y=history_df['strong_momentum'], 
                   name="🚀 Strong Momentum", line=dict(color="#00d4ff", width=2)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=history_df['date'], y=history_df['total_uptrends'], 
                   name="📈 Total Uptrends", line=dict(color="#00ff88", width=2)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=history_df['date'], y=history_df['breakout_alerts'], 
                   name="🔥 Breakout Alerts", line=dict(color="#ff6b6b", width=2)),
        secondary_y=False
    )
    
    # Avg Trend Score on secondary axis (different scale: 0-100)
    fig.add_trace(
        go.Scatter(x=history_df['date'], y=history_df['avg_trend_score'], 
                   name="📊 Avg Trend Score", line=dict(color="#ffd93d", width=3, dash='dot')),
        secondary_y=True
    )
    
    # Styling
    fig.update_layout(
        title="🌡️ Market Mood History",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=350,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    fig.update_yaxes(title_text="Stock Count", secondary_y=False)
    fig.update_yaxes(title_text="Avg Score (0-100)", secondary_y=True, range=[0, 100])
    
    return fig
