"""
DNA-3 V2 METRICS ENGINE
========================
Calculates Relative Strength (RS) vs Nifty and Volatility for dashboard display.

This module adds the DNA-3 V2 alpha signals to the dashboard:
1. RS vs Nifty (3-Month): How much the stock outperformed Nifty
2. Volatility (Annualized): Price movement magnitude
3. DNA-3 Signal: PASS/FAIL based on filters
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_nifty_returns():
    """Fetch Nifty 50 returns for RS calculation."""
    try:
        nifty = yf.Ticker("^NSEI").history(period="1y")
        if nifty.empty: return {}
        
        nifty.index = nifty.index.tz_localize(None)
        
        # Calculate returns for different periods
        current = nifty['Close'].iloc[-1]
        ret_1m = (current - nifty['Close'].iloc[-21]) / nifty['Close'].iloc[-21] * 100 if len(nifty) > 21 else 0
        ret_3m = (current - nifty['Close'].iloc[-63]) / nifty['Close'].iloc[-63] * 100 if len(nifty) > 63 else 0
        
        return {'1m': ret_1m, '3m': ret_3m, 'data': nifty}
    except:
        return {'1m': 0, '3m': 0, 'data': None}

def calculate_dna_metrics(ticker: str, hist_df: pd.DataFrame = None) -> dict:
    """
    Calculate DNA-3 V2 metrics for a single stock.
    
    Args:
        ticker: Stock ticker
        hist_df: Optional pre-fetched history DataFrame
        
    Returns:
        dict with rs_3m, volatility, dna_signal, dna_score
    """
    try:
        if hist_df is None or hist_df.empty:
            hist_df = yf.Ticker(ticker).history(period="1y")
            if hist_df.empty:
                return {'rs_3m': None, 'volatility': None, 'dna_signal': 'NO DATA', 'dna_score': 0}
            hist_df.index = hist_df.index.tz_localize(None)
        
        nifty_data = get_nifty_returns()
        
        current = hist_df['Close'].iloc[-1]
        
        # === RELATIVE STRENGTH (3-Month vs Nifty) ===
        if len(hist_df) > 63:
            ret_3m = (current - hist_df['Close'].iloc[-63]) / hist_df['Close'].iloc[-63] * 100
            rs_3m = ret_3m - nifty_data.get('3m', 0)
        else:
            rs_3m = 0
        
        # === VOLATILITY (Annualized) ===
        if len(hist_df) > 60:
            daily_returns = hist_df['Close'].pct_change().dropna()[-60:]
            volatility = daily_returns.std() * np.sqrt(252) * 100
        else:
            volatility = 0
        
        # === MA50 CHECK ===
        if len(hist_df) > 50:
            ma50 = hist_df['Close'].rolling(50).mean().iloc[-1]
            above_ma50 = current > ma50
        else:
            above_ma50 = True  # Assume pass if not enough data
        
        # === DNA-3 V2 SIGNAL ===
        passes_rs = rs_3m >= 2.0
        passes_vol = volatility >= 30
        passes_trend = above_ma50
        
        if passes_rs and passes_vol and passes_trend:
            dna_signal = 'BUY'
            dna_score = int(min(100, 50 + rs_3m + (volatility - 30)))
        elif passes_rs and passes_trend:
            dna_signal = 'WATCH'
            dna_score = int(min(70, 30 + rs_3m))
        else:
            dna_signal = 'HOLD'
            dna_score = 0
        
        return {
            'rs_3m': round(rs_3m, 2),
            'volatility': round(volatility, 1),
            'dna_signal': dna_signal,
            'dna_score': dna_score
        }
        
    except Exception as e:
        return {'rs_3m': None, 'volatility': None, 'dna_signal': 'ERROR', 'dna_score': 0}

def add_dna_columns_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add DNA-3 V2 columns to the main market data DataFrame.
    
    Args:
        df: DataFrame with 'ticker' column
        
    Returns:
        DataFrame with added rs_3m, volatility, dna_signal, dna_score columns
    """
    if 'ticker' not in df.columns:
        return df
    
    # Initialize columns
    df['rs_3m'] = None
    df['volatility'] = None
    df['dna_signal'] = 'HOLD'
    df['dna_score'] = 0
    
    # Batch calculate for all tickers
    for idx, row in df.iterrows():
        ticker = row['ticker']
        metrics = calculate_dna_metrics(ticker)
        df.at[idx, 'rs_3m'] = metrics['rs_3m']
        df.at[idx, 'volatility'] = metrics['volatility']
        df.at[idx, 'dna_signal'] = metrics['dna_signal']
        df.at[idx, 'dna_score'] = metrics['dna_score']
    
    return df

def get_dna_styled_value(value, metric_type='rs'):
    """Return styled HTML for DNA metrics display."""
    if value is None:
        return '<span style="color: gray;">N/A</span>'
    
    if metric_type == 'rs':
        if value >= 5:
            color = '#00C853'  # Green
            icon = '🚀'
        elif value >= 2:
            color = '#69F0AE'  # Light Green
            icon = '📈'
        elif value >= 0:
            color = '#FFD600'  # Yellow
            icon = '➖'
        else:
            color = '#FF5252'  # Red
            icon = '📉'
        return f'<span style="color: {color}; font-weight: bold;">{icon} {value:+.1f}%</span>'
    
    elif metric_type == 'vol':
        if value >= 45:
            color = '#FF6D00'  # Orange (High)
            icon = '⚡'
        elif value >= 30:
            color = '#00C853'  # Green (Goldilocks)
            icon = '✅'
        else:
            color = '#FFD600'  # Yellow (Low)
            icon = '🐢'
        return f'<span style="color: {color}; font-weight: bold;">{icon} {value:.1f}%</span>'
    
    elif metric_type == 'signal':
        if value == 'BUY':
            return '<span style="background: #00C853; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold;">BUY</span>'
        elif value == 'WATCH':
            return '<span style="background: #FFD600; color: black; padding: 2px 8px; border-radius: 4px; font-weight: bold;">WATCH</span>'
        else:
            return '<span style="color: gray;">HOLD</span>'
    
    return str(value)
