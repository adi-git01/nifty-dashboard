"""
Market Regime Detector
======================
Detects bull/bear/neutral market regime using Nifty 50 vs 200DMA.
Used by Proposal 4 (Active Allocation).
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple
import streamlit as st


# Nifty 50 ticker on Yahoo Finance
NIFTY50_TICKER = "^NSEI"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nifty50_data(days: int = 250) -> pd.DataFrame:
    """
    Fetch Nifty 50 historical data.
    
    Args:
        days: Number of days to fetch (need ~250 for 200DMA)
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        nifty = yf.download(
            NIFTY50_TICKER,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True
        )
        
        return nifty
    except Exception as e:
        print(f"Error fetching Nifty 50 data: {e}")
        return pd.DataFrame()


def calculate_nifty_200dma(nifty_data: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate current Nifty 50 price and 200DMA.
    
    Returns:
        Tuple of (current_price, 200dma)
    """
    if nifty_data.empty or len(nifty_data) < 200:
        return 0, 0
    
    current_price = nifty_data['Close'].iloc[-1]
    dma_200 = nifty_data['Close'].rolling(window=200).mean().iloc[-1]
    
    return float(current_price), float(dma_200)


def detect_market_regime(nifty_data: pd.DataFrame = None) -> Dict[str, any]:
    """
    Detect current market regime based on Nifty 50 position vs 200DMA.
    
    Regimes:
    - "bull": Nifty 50 > 5% above 200DMA
    - "bear": Nifty 50 > 5% below 200DMA  
    - "neutral": Nifty 50 within ±5% of 200DMA
    
    Returns:
        Dict with regime info
    """
    if nifty_data is None or nifty_data.empty:
        nifty_data = fetch_nifty50_data()
    
    if nifty_data.empty:
        return {
            "regime": "neutral",
            "nifty_price": 0,
            "nifty_200dma": 0,
            "dist_from_200dma_pct": 0,
            "description": "Unable to fetch data, defaulting to neutral"
        }
    
    current_price, dma_200 = calculate_nifty_200dma(nifty_data)
    
    if dma_200 == 0:
        return {
            "regime": "neutral",
            "nifty_price": current_price,
            "nifty_200dma": 0,
            "dist_from_200dma_pct": 0,
            "description": "Insufficient data for 200DMA"
        }
    
    dist_pct = ((current_price - dma_200) / dma_200) * 100
    
    if dist_pct > 5:
        regime = "bull"
        description = f"Nifty 50 is {dist_pct:.1f}% above 200DMA - BULL market"
    elif dist_pct < -5:
        regime = "bear"
        description = f"Nifty 50 is {abs(dist_pct):.1f}% below 200DMA - BEAR market"
    else:
        regime = "neutral"
        description = f"Nifty 50 is {dist_pct:+.1f}% from 200DMA - NEUTRAL market"
    
    return {
        "regime": regime,
        "nifty_price": round(current_price, 2),
        "nifty_200dma": round(dma_200, 2),
        "dist_from_200dma_pct": round(dist_pct, 2),
        "description": description
    }


def detect_regime_for_date(nifty_data: pd.DataFrame, date: datetime) -> str:
    """
    Detect market regime for a specific historical date.
    Used in backtesting to get regime at each rebalance.
    
    Args:
        nifty_data: Full Nifty 50 history
        date: Date to check
        
    Returns:
        Regime string: "bull", "bear", or "neutral"
    """
    if nifty_data is None or nifty_data.empty:
        return "neutral"
    
    try:
        # Handle timezone-aware dates
        if hasattr(date, 'tz') and date.tz is not None:
            date = date.tz_localize(None)
        
        # Get data up to and including date
        data_to_date = nifty_data[nifty_data.index <= date]
        
        if len(data_to_date) < 200:
            return "neutral"
        
        # Handle MultiIndex columns from yfinance
        if isinstance(data_to_date.columns, pd.MultiIndex):
            # Try to get Close column - could be ('Close', ticker) format
            close_cols = [c for c in data_to_date.columns if 'Close' in str(c)]
            if close_cols:
                close_series = data_to_date[close_cols[0]]
            else:
                return "neutral"
        else:
            close_series = data_to_date['Close']
        
        # Flatten if still multi-dimensional
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]
        
        # Extract scalar values using .item() for safety
        last_val = close_series.iloc[-1]
        if hasattr(last_val, 'item'):
            current_price = float(last_val.item())
        else:
            current_price = float(last_val)
        
        ma_200_series = close_series.rolling(window=200).mean()
        ma_val = ma_200_series.iloc[-1]
        if hasattr(ma_val, 'item'):
            dma_200 = float(ma_val.item())
        else:
            dma_200 = float(ma_val)
        
        # Check for NaN or zero
        if pd.isna(current_price) or pd.isna(dma_200) or dma_200 == 0:
            return "neutral"
        
        dist_pct = ((current_price - dma_200) / dma_200) * 100
        
        if dist_pct > 5:
            return "bull"
        elif dist_pct < -5:
            return "bear"
        else:
            return "neutral"
    except Exception as e:
        return "neutral"


def get_regime_allocations(regime: str, proposal_config: Dict) -> Dict[str, float]:
    """
    Get strategy allocations based on current regime.
    
    Args:
        regime: "bull", "bear", or "neutral"
        proposal_config: The proposal config dict with 'regimes' key
        
    Returns:
        Dict of strategy_name -> allocation percentage
    """
    regimes = proposal_config.get("regimes", {})
    
    if regime in regimes:
        return regimes[regime]
    
    # Default to neutral if regime not found
    return regimes.get("neutral", {
        "GARP": 0.40,
        "MomentumBreakout": 0.35,
        "TrueValue": 0.25,
    })
