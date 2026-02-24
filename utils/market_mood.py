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
        
    total_stocks = len(df)
    if total_stocks == 0:
        return None
    
    # 1. Strong Momentum: trend_score >= 80 (as % of universe)
    strong_momentum_count = len(df[df.get('trend_score', pd.Series([0]*len(df))) >= 80])
    strong_momentum = (strong_momentum_count / total_stocks) * 100
    
    # 2. Total Uptrends: trend_signal contains 'UPTREND' (as % of universe)
    if 'trend_signal' in df.columns:
        total_uptrends_count = len(df[df['trend_signal'].str.contains('UPTREND', na=False)])
        total_uptrends = (total_uptrends_count / total_stocks) * 100
    else:
        total_uptrends = 0
    
    # 3. Avg Trend Score (already naturally scaled 0-100)
    if 'trend_score' in df.columns:
        avg_trend_score = df['trend_score'].mean()
    else:
        avg_trend_score = 50
    
    # 4. Breakout Alerts: within 5% of 52-week high (as % of universe)
    if 'dist_52w' in df.columns:
        breakout_alerts_count = len(df[df['dist_52w'] >= -5])
        breakout_alerts = (breakout_alerts_count / total_stocks) * 100
    elif 'fiftyTwoWeekHigh' in df.columns and 'price' in df.columns:
        df['_dist'] = ((df['price'] - df['fiftyTwoWeekHigh']) / df['fiftyTwoWeekHigh']) * 100
        breakout_alerts_count = len(df[df['_dist'] >= -5])
        breakout_alerts = (breakout_alerts_count / total_stocks) * 100
    else:
        breakout_alerts = 0
    
    return {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'strong_momentum': round(strong_momentum, 1),
        'total_uptrends': round(total_uptrends, 1),
        'avg_trend_score': round(avg_trend_score, 1),
        'breakout_alerts': round(breakout_alerts, 1)
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

def load_mood_history(days=365):
    """
    Loads the mood history for charting.
    Returns DataFrame with date and 4 metrics.
    """
    if not os.path.exists(MOOD_FILE):
        return pd.DataFrame()
    
    df = pd.read_csv(MOOD_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    if days:
        cutoff = pd.to_datetime('today') - pd.Timedelta(days=days)
        df = df[df['date'] >= cutoff]
        
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
    All metrics are now normalized as percentages (0-100%).
    """
    import plotly.graph_objects as go
    
    if history_df.empty:
        return None
    
    # Create simple figure (no secondary axis needed since all are 0-100%)
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=history_df['date'], y=history_df['strong_momentum'], 
                   name="🚀 Strong Momentum", mode='lines', 
                   line=dict(color="#00C853", width=2.5, shape='spline', smoothing=0.8))
    )
    
    fig.add_trace(
        go.Scatter(x=history_df['date'], y=history_df['total_uptrends'], 
                   name="📈 Total Uptrends", mode='lines', 
                   fill='tozeroy', fillcolor='rgba(99,91,255,0.05)',
                   line=dict(color="#635BFF", width=3, shape='spline', smoothing=0.8))
    )
    
    fig.add_trace(
        go.Scatter(x=history_df['date'], y=history_df['breakout_alerts'], 
                   name="🔥 Breakout Alerts", mode='lines',
                   line=dict(color="#FF3366", width=2.5, dash='dot', shape='spline', smoothing=0.8))
    )
    
    # Avg Trend Score
    fig.add_trace(
        go.Scatter(x=history_df['date'], y=history_df['avg_trend_score'], 
                   name="📊 Avg Trend Score", mode='lines',
                   line=dict(color="#FFC107", width=2.5, shape='spline', smoothing=0.8))
    )
    
    # Styling
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1),
        height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(
            title_text="Percentage / Score (0-100%)", 
            range=[0, 100], 
            gridcolor='rgba(0,0,0,0.05)'
        )
    )
    
    fig.update_xaxes(showgrid=False)
    
    return fig
