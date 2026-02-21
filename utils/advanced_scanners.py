import pandas as pd
import numpy as np

def calculate_vcp_compression(df, window=10):
    """
    Measures the price compression (True Range) over the last N days.
    Lower compression score = tighter price action.
    """
    if len(df) < window + 1:
        return float('inf')
        
    recent = df.iloc[-window:]
    highs = recent['High'].values
    lows = recent['Low'].values
    
    # Calculate ATR %
    atr_pct = ((highs - lows) / lows).mean() * 100
    return atr_pct

def find_vcp_setups(market_df, full_history_dict):
    """
    Finds Volatility Contraction Patterns (VCP).
    Requirements:
    1. Trend > 60 (Stock must be in a general uptrend, not breaking down)
    2. Price > MA50 (Above structural support)
    3. Last 10 days ATR % < 3.5% (Extremely tight daily ranges)
    4. Today's volume < 50% of 60-day average volume (Severe dry-up / supply removed)
    """
    if market_df is None or market_df.empty:
        return []

    vcp_candidates = []
    
    for _, row in market_df.iterrows():
        ticker = row['ticker']
        score = row.get('trend_score', 0)
        p = row.get('price', 0)
        
        # Fast veto
        if score < 50 or p < 20: 
            continue
            
        # Get history
        if ticker not in full_history_dict:
            continue
            
        hist = full_history_dict[ticker]
        if len(hist) < 65:
            continue
            
        ma50 = hist['Close'].rolling(50).mean().iloc[-1]
        if p < ma50:
            continue # Needs to consolidate above key support
            
        # 1. Volume Dry-up Check
        vol_today = hist['Volume'].iloc[-1]
        vol_60d_avg = hist['Volume'].rolling(60).mean().iloc[-1]
        
        if vol_60d_avg == 0: continue
        vol_ratio = vol_today / vol_60d_avg
        
        if vol_ratio > 0.5: 
            continue # Volume has not dried up enough
            
        # 2. Price Contraction Check
        compression_pct = calculate_vcp_compression(hist, 10)
        if compression_pct > 3.5: 
            continue # Action is too loose
            
        vcp_candidates.append({
            'Ticker': ticker,
            'Name': row.get('name', ticker),
            'Sector': row.get('sector', 'Unknown'),
            'Price': p,
            'Score': score,
            'Compression': compression_pct,
            'Vol_Ratio': vol_ratio * 100
        })
        
    return sorted(vcp_candidates, key=lambda x: x['Compression'])


def find_rs_divergence(market_df, nifty_df):
    """
    Finds "Green in a Sea of Red" (Relative Strength Divergence).
    Requirements:
    1. Nifty must be down > 0.5% today (A red market day)
    2. Stock must close POSITIVE (> 0.5%)
    3. Stock must be near 52-week highs (Distance > -15%)
    """
    if nifty_df is None or len(nifty_df) < 2 or market_df is None or market_df.empty:
        return []
        
    nifty_close = nifty_df['Close'].iloc[-1]
    nifty_prev = nifty_df['Close'].iloc[-2]
    nifty_return = (nifty_close - nifty_prev) / nifty_prev * 100
    
    # If Nifty is not falling significantly, RS Divergence is useless
    if nifty_return > -0.5:
        return []
        
    rs_candidates = []
    for _, row in market_df.iterrows():
        ticker = row['ticker']
        
        # Require positive daily return vs negative market
        day_chg = row.get('return_1d', 0)
        if day_chg < 0.5:
            continue
            
        # Require stock to not be garbage (near highs)
        dist_hi = row.get('dist_52w', -100)
        if dist_hi < -15:
            continue
            
        # Calculate Delta RS
        delta_rs = day_chg - nifty_return
        
        rs_candidates.append({
            'Ticker': ticker,
            'Name': row.get('name', ticker),
            'Sector': row.get('sector', 'Unknown'),
            'Price': row.get('price', 0),
            'Stock_Ret': day_chg,
            'Nifty_Ret': nifty_return,
            'Delta_RS': delta_rs,
            'Dist_52W': dist_hi
        })
        
    return sorted(rs_candidates, key=lambda x: x['Delta_RS'], reverse=True)


def find_live_earnings_shocks(market_df, full_history_dict):
    """
    Finds Day-0 Earnings Gaps/Shocks.
    Requirements:
    1. Stock jumps > 5% today
    2. Volume jumps > 300% of its 20-day average
    """
    if market_df is None or market_df.empty:
        return []
        
    shock_candidates = []
    
    from utils.live_desk import get_pead_edge # Import specifically for the profile
    
    for _, row in market_df.iterrows():
        ticker = row['ticker']
        p = row.get('price', 0)
        day_chg = row.get('return_1d', 0)
        
        # 1. Price Jump
        if day_chg < 5.0:
            continue
            
        # Get history for Volume
        if ticker not in full_history_dict:
            continue
        hist = full_history_dict[ticker]
        if len(hist) < 25:
            continue
            
        vol_today = hist['Volume'].iloc[-1]
        vol_20d = hist['Volume'].rolling(20).mean().shift(1).iloc[-1]
        
        if vol_20d == 0: continue
        vol_ratio = vol_today / vol_20d
        
        # 2. Volume Jump > 3x (300%)
        if vol_ratio < 3.0:
            continue
            
        sector = row.get('sector', 'Unknown')
        profile = get_pead_edge(sector)
            
        shock_candidates.append({
            'Ticker': ticker,
            'Name': row.get('name', ticker),
            'Sector': sector,
            'Price': p,
            'Jump_Pct': day_chg,
            'Vol_Mult': vol_ratio,
            'PEAD_Action': profile
        })
        
    return sorted(shock_candidates, key=lambda x: x['Jump_Pct'], reverse=True)
