"""
Historical Performance & Sector Analytics Module
Calculates multi-period returns and sector performance metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from utils.data_engine import get_stock_history

def calculate_multi_period_returns(ticker):
    """
    Fetches historical data and calculates returns over multiple periods.
    Returns: dict with 1D, 1W, 1M, 3M, 6M, 1Y returns
    """
    try:
        # Use shared cache instead of new fetch
        hist = get_stock_history(ticker, period="1y")
        
        if hist.empty or len(hist) < 5:
            return {
                'return_1d': 0, 'return_1w': 0, 'return_1m': 0,
                'return_3m': 0, 'return_6m': 0, 'return_1y': 0
            }
        
        current_price = hist['Close'].iloc[-1]
        
        # Calculate returns for different periods
        returns = {}
        
        # 1 Day
        if len(hist) >= 2:
            returns['return_1d'] = ((current_price / hist['Close'].iloc[-2]) - 1) * 100
        else:
            returns['return_1d'] = 0
            
        # 1 Week (5 trading days)
        if len(hist) >= 5:
            returns['return_1w'] = ((current_price / hist['Close'].iloc[-5]) - 1) * 100
        else:
            returns['return_1w'] = 0
            
        # 1 Month (~21 trading days)
        if len(hist) >= 21:
            returns['return_1m'] = ((current_price / hist['Close'].iloc[-21]) - 1) * 100
        else:
            returns['return_1m'] = returns.get('return_1w', 0)
            
        # 3 Months (~63 trading days)
        if len(hist) >= 63:
            returns['return_3m'] = ((current_price / hist['Close'].iloc[-63]) - 1) * 100
        else:
            returns['return_3m'] = returns.get('return_1m', 0)
            
        # 6 Months (~126 trading days)
        if len(hist) >= 126:
            returns['return_6m'] = ((current_price / hist['Close'].iloc[-126]) - 1) * 100
        else:
            returns['return_6m'] = returns.get('return_3m', 0)
            
        # 1 Year (~252 trading days)
        if len(hist) >= 252:
            returns['return_1y'] = ((current_price / hist['Close'].iloc[0]) - 1) * 100
        else:
            returns['return_1y'] = ((current_price / hist['Close'].iloc[0]) - 1) * 100
            
        return returns
        
    except Exception as e:
        print(f"Error calculating returns for {ticker}: {e}")
        return {
            'return_1d': 0, 'return_1w': 0, 'return_1m': 0,
            'return_3m': 0, 'return_6m': 0, 'return_1y': 0
        }


def calculate_sector_breadth(df, sector):
    """
    Calculates advance/decline ratios for a sector.
    """
    sector_df = df[df['sector'] == sector].copy()
    
    if sector_df.empty:
        return {}
    
    total = len(sector_df)
    
    # Day
    advancing_1d = len(sector_df[sector_df['return_1d'] > 0])
    declining_1d = len(sector_df[sector_df['return_1d'] < 0])
    
    # Week
    advancing_1w = len(sector_df[sector_df['return_1w'] > 0])
    declining_1w = len(sector_df[sector_df['return_1w'] < 0])
    
    # Month
    advancing_1m = len(sector_df[sector_df['return_1m'] > 0])
    declining_1m = len(sector_df[sector_df['return_1m'] < 0])
    
    # Quarter
    advancing_3m = len(sector_df[sector_df['return_3m'] > 0])
    declining_3m = len(sector_df[sector_df['return_3m'] < 0])
    
    return {
        'sector': sector,
        'total_stocks': total,
        'adv_1d': advancing_1d,
        'dec_1d': declining_1d,
        'adv_1w': advancing_1w,
        'dec_1w': declining_1w,
        'adv_1m': advancing_1m,
        'dec_1m': declining_1m,
        'adv_3m': advancing_3m,
        'dec_3m': declining_3m,
        'breadth_1d': (advancing_1d - declining_1d) / total * 100 if total > 0 else 0,
        'breadth_1w': (advancing_1w - declining_1w) / total * 100 if total > 0 else 0,
        'breadth_1m': (advancing_1m - declining_1m) / total * 100 if total > 0 else 0,
        'breadth_3m': (advancing_3m - declining_3m) / total * 100 if total > 0 else 0,
        # Composite Breadth Score: Weighted (1D: 10%, 1W: 30%, 1M: 60%)
        'composite_breadth': (
            ((advancing_1d - declining_1d) / total * 100 * 0.1) +
            ((advancing_1w - declining_1w) / total * 100 * 0.3) +
            ((advancing_1m - declining_1m) / total * 100 * 0.6)
        ) if total > 0 else 0,
        'avg_return_1d': sector_df['return_1d'].mean(),
        'avg_return_1w': sector_df['return_1w'].mean(),
        'avg_return_1m': sector_df['return_1m'].mean(),
        'avg_return_3m': sector_df['return_3m'].mean(),
    }


def get_quarterly_performance(ticker):
    """
    Gets quarterly performance snapshots for a stock.
    Returns last 4 quarters' performance.
    """
    try:
        # Use shared cache
        hist = get_stock_history(ticker, period="1y")
        
        if hist.empty or len(hist) < 60:
            return []
        
        quarters = []
        today = datetime.now()
        
        # Calculate performance for each quarter
        for q in range(4):
            end_idx = len(hist) - (q * 63)  # ~63 trading days per quarter
            start_idx = end_idx - 63
            
            if start_idx < 0:
                break
                
            if end_idx > len(hist):
                end_idx = len(hist)
            if start_idx < 0:
                start_idx = 0
                
            q_start = hist['Close'].iloc[start_idx]
            q_end = hist['Close'].iloc[end_idx - 1]
            q_return = ((q_end / q_start) - 1) * 100
            
            quarter_date = today - timedelta(days=q * 90)
            quarter_name = f"Q{4 - q} {quarter_date.year}"
            
            quarters.append({
                'quarter': quarter_name,
                'return': round(q_return, 2)
            })
        
        return quarters[::-1]  # Oldest first
        
    except Exception as e:
        print(f"Error getting quarterly performance for {ticker}: {e}")
        return []


def calculate_peer_relative_performance(ticker, df, lookback_quarters=4):
    """
    Compares stock performance vs sector peers over time.
    """
    try:
        # Get this stock's sector
        stock_row = df[df['ticker'] == ticker]
        if stock_row.empty:
            return None
            
        sector = stock_row['sector'].iloc[0]
        peers = df[df['sector'] == sector]
        
        if len(peers) < 3:
            return None
        
        peer_returns = peers[['ticker', 'return_3m']].copy()
        peer_returns['rank'] = peer_returns['return_3m'].rank(ascending=False, pct=True)
        
        stock_rank = peer_returns[peer_returns['ticker'] == ticker]['rank'].iloc[0]
        
        return {
            'sector': sector,
            'peer_count': len(peers),
            'percentile': round((1 - stock_rank) * 100, 1),
            'stock_return': stock_row['return_3m'].iloc[0] if 'return_3m' in stock_row.columns else 0,
            'sector_avg': peers['return_3m'].mean() if 'return_3m' in peers.columns else 0,
            'outperformance': stock_row['return_3m'].iloc[0] - peers['return_3m'].mean() if 'return_3m' in stock_row.columns else 0
        }
        
    except Exception as e:
        print(f"Error calculating relative performance: {e}")
        return None


def analyze_all_sectors_breadth(df):
    """
    Analyzes breadth for all sectors in the dataframe.
    """
    sectors = df['sector'].unique().tolist()
    breadth_data = []
    
    for sector in sectors:
        if sector and sector != 'Unknown':
            breadth = calculate_sector_breadth(df, sector)
            if breadth:
                breadth_data.append(breadth)
    
    if not breadth_data:
        return pd.DataFrame()
        
    return pd.DataFrame(breadth_data).sort_values('avg_return_1m', ascending=False)
