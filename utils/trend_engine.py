import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import streamlit as st
from utils.data_engine import get_stock_history

def calculate_stock_trend_history(ticker, hist_df=None):
    """
    Reconstructs historical trend scores (0-100) and signals for a stock
    over the entire history provided.
    
    Args:
        ticker: Ticker symbol
        hist_df: DataFrame with Date, Open, High, Low, Close (from yfinance)
        
    Returns:
        DataFrame with added 'trend_score', 'trend_signal', 'signal_color' columns
    """
    if hist_df is None or hist_df.empty:
        # Fallback fetch if not provided (Use Shared Cache)
        hist_df = get_stock_history(ticker, period="2y")

    if hist_df.empty:
        return pd.DataFrame()

    df = hist_df.copy()
    
    # Calculate MAs
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate rolling 52-week High/Low (approx 252 trading days)
    df['52W_High'] = df['High'].rolling(window=252, min_periods=50).max()
    df['52W_Low'] = df['Low'].rolling(window=252, min_periods=50).min()
    
    # Initialize Score
    df['trend_score'] = 50.0  # Default neutral
    
    # --- Vectorized Trend Calculation (faster than iterrows) ---
    
    # 1. MA Alignment
    # Price > MA50 (+15), Price > MA200 (+15), Golden Cross (+10)
    # Penalties: Price < MA50 (-10), Price < MA200 (-15), Death Cross (-5)
    
    ma_score = np.zeros(len(df))
    
    # Price vs MA50
    ma_score += np.where(df['Close'] > df['MA50'], 15, -10)
    
    # Price vs MA200
    ma_score += np.where(df['Close'] > df['MA200'], 15, -15)
    
    # Crosses
    ma_score += np.where((df['MA50'] > 0) & (df['MA200'] > 0) & (df['MA50'] > df['MA200']), 10, -5)
    
    # 2. 52-Week Position (30 pts max)
    # Range Position (-15 to +15) + Near High Bonus (+10) or Low Penalty (-10)
    
    range_52 = df['52W_High'] - df['52W_Low']
    range_52 = range_52.replace(0, np.nan) # Avoid div by zero
    
    # Position: -0.5 to 0.5 range scaled to -15 to +15
    pct_pos = (df['Close'] - df['52W_Low']) / range_52
    pos_score = (pct_pos - 0.5) * 30
    pos_score = pos_score.fillna(0)
    
    # Near High Bonus (> -5% from high)
    dist_from_high = (df['Close'] - df['52W_High']) / df['52W_High']
    high_bonus = np.where(dist_from_high > -0.05, 10, 
                          np.where(dist_from_high < -0.30, -10, 0))
    
    # Total Score
    total_score = 50 + ma_score + pos_score + high_bonus
    df['trend_score'] = total_score.clip(0, 100).fillna(50)
    
    # Determine Signals
    conditions = [
        (df['trend_score'] >= 75),
        (df['trend_score'] >= 60),
        (df['trend_score'] >= 40),
        (df['trend_score'] >= 25)
    ]
    choices = ['STRONG UPTREND', 'UPTREND', 'NEUTRAL', 'DOWNTREND']
    df['trend_signal'] = np.select(conditions, choices, default='STRONG DOWNTREND')
    
    # Color mapping for chart
    color_map = {
        'STRONG UPTREND': '#00C853',
        'UPTREND': '#69F0AE',
        'NEUTRAL': '#FFD600',
        'DOWNTREND': '#FF6D00',
        'STRONG DOWNTREND': '#D50000'
    }
    df['signal_color'] = df['trend_signal'].map(color_map)
    
    return df


@st.cache_data(ttl=3600*4)
def calculate_sector_history(nifty500_df):
    """
    Constructs equal-weighted historical indices for each sector using cached price history.
    
    Args:
        nifty500_df: DataFrame containing at least 'ticker' and 'sector' columns
        
    Returns:
        DataFrame index=Date, columns=Sectors (values are normalized 100 start)
    """
    try:
        # Get unique tickers
        tickers = nifty500_df['ticker'].unique().tolist()
        
        # Batch fetch history (optimized)
        # We'll valid tickers only
        valid_tickers = [t for t in tickers if t.endswith(('.NS', '.BO'))]
        
        # Download in one massive batch for speed
        data = yf.download(valid_tickers, period="1y", interval="1d", progress=False)['Close']
        
        if data.empty:
            return pd.DataFrame()
            
        # Group tickers by sector
        sector_map = nifty500_df.set_index('ticker')['sector'].to_dict()
        
        # Calculate daily returns
        returns = data.pct_change()
        
        # Aggregated Sector Returns
        sector_indices = pd.DataFrame()
        
        unique_sectors = nifty500_df['sector'].unique()
        
        for sector in unique_sectors:
            if not sector: continue
                
            # Find tickers in this sector
            sector_tickers = [t for t in valid_tickers if sector_map.get(t) == sector]
            
            # Filter returns for these tickers
            # Handle potential missing columns if fetch failed for some
            avail_tickers = [t for t in sector_tickers if t in returns.columns]
            
            if not avail_tickers:
                continue
                
            # Average daily return of the sector (Equal Weight)
            sector_daily_ret = returns[avail_tickers].mean(axis=1)
            
            # Construct Index (Base 100)
            sector_index = (1 + sector_daily_ret).cumprod() * 100
            sector_indices[sector] = sector_index
            
        # Also add Nifty 500 Benchmark (Approximate using mean of all)
        market_ret = returns.mean(axis=1)
        sector_indices['Nifty 500 (Eq Wt)'] = (1 + market_ret).cumprod() * 100
        
        return sector_indices.dropna()
        
    except Exception as e:
        print(f"Error calculating sector history: {e}")
        return pd.DataFrame()
