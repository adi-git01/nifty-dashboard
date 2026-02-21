"""
LIVE TRADING DESK MODULE
========================
Provides live integration of the DNA3-V4 Seasonal Momentum engine and Macro
Regime Detector into the Streamlit dashboard for daily actionable scanning.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

# ======================================================================
# 1. MACRO REGIME DETECTOR
# ======================================================================
def get_live_regime(nifty_df):
    """
    Calculates the exact current macro regime based on Nifty 50 close,
    MA50, MA200, 3-month return, and peak drawdown.
    Expected to run on real-time / EOD Nifty data.
    """
    if nifty_df is None or nifty_df.empty or len(nifty_df) < 63:
        return {
            'regime': 'UNKNOWN',
            'cash': 0.20,
            'max_pos': 8,
            'color': 'gray',
            'description': 'Insufficient Nifty data to determine regime.'
        }
        
    p = nifty_df['Close'].iloc[-1]
    ma50 = nifty_df['Close'].rolling(min_periods=1, window=50).mean().iloc[-1]
    ma200 = nifty_df['Close'].rolling(min_periods=1, window=200).mean().iloc[-1]
    ret_63 = (p - nifty_df['Close'].iloc[-63]) / nifty_df['Close'].iloc[-63] * 100
    
    pk = nifty_df['Close'].cummax().iloc[-1]
    dd = (p - pk) / pk * 100 if pk > 0 else 0
    
    if p > ma50 and ma50 > ma200 and ret_63 > 5:
        return {
            'regime': 'STRONG BULL',
            'cash': 0.05,
            'max_pos': 12,
            'color': '#34C759',
            'description': 'Aggressive uptrend. Maximum capital deployment.'
        }
    elif p > ma50 and ret_63 > 0:
        return {
            'regime': 'MILD BULL',
            'cash': 0.10,
            'max_pos': 10,
            'color': '#32D74B',
            'description': 'Uptrend intact but momentum is standard. Normal deployment.'
        }
    elif p < ma50 and (ret_63 < -5 or dd < -10):
        return {
            'regime': 'BEAR / CORRECTION',
            'cash': 0.40,
            'max_pos': 6,
            'color': '#FF3B30',
            'description': 'Market is in a structural correction. Hold high cash, strict stops.'
        }
    else:
        return {
            'regime': 'SIDEWAYS / CHOP',
            'cash': 0.20,
            'max_pos': 8,
            'color': '#FFD60A',
            'description': 'Choppy consolidation. Veto marginal setups to avoid whipsaws.'
        }


# ======================================================================
# 2. V4 SEASONAL OVERLAY DICTIONARY
# ======================================================================
SEASONAL_RULES = {
    # TRAPS (Months where the sector bleeds out - VETO ALL BUYS)
    'Financial': {1: 'TRAP', 2: 'TRAP', 5: 'TRAP'},          
    'Industrial': {1: 'TRAP'},                               
    'IT': {2: 'TRAP', 3: 'TRAP'},                            
    'Auto': {2: 'TRAP'},                                     
    'Real Estate': {2: 'TRAP'},                              
    
    # EDGES (Golden months - ACCELERATE BUYS)
    'Industrial': {3: 'EDGE', 4: 'EDGE', 5: 'EDGE'},         
    'Auto': {4: 'EDGE', 5: 'EDGE'},                          
    'Financial': {10: 'EDGE', 11: 'EDGE', 12: 'EDGE'},       
    'IT': {6: 'EDGE', 7: 'EDGE'},                            
    'Consumer': {9: 'EDGE', 10: 'EDGE'},                     
    'Healthcare': {7: 'EDGE', 8: 'EDGE'},                    
}

def classify_sector(sector_str):
    s = str(sector_str).upper()
    if any(x in s for x in ['BANK', 'FINANC', 'INSUR', 'BROKER']): return 'Financial'
    if any(x in s for x in ['INDUST', 'CAPITAL GOODS', 'ENGINEERING', 'METAL']): return 'Industrial'
    if any(x in s for x in ['IT ', 'SOFTWARE', 'TECH']): return 'IT'
    if any(x in s for x in ['AUTO']): return 'Auto'
    if any(x in s for x in ['CONSUMER', 'FMCG', 'RETAIL', 'FOOD']): return 'Consumer'
    if any(x in s for x in ['HEALTH', 'PHARMA', 'HOSPITAL']): return 'Healthcare'
    if any(x in s for x in ['REALTY', 'REAL ESTATE', 'BUILDING']): return 'Real Estate'
    return 'Other'

def get_seasonal_overlay(sector_str, month_num):
    """Returns 'TRAP', 'EDGE', or 'NEUTRAL' based on mathematical calendar odds."""
    cat = classify_sector(sector_str)
    if cat in SEASONAL_RULES and month_num in SEASONAL_RULES[cat]:
        return SEASONAL_RULES[cat][month_num]
    return 'NEUTRAL'

# ======================================================================
# 3. PLAYBOOK: CYCLICITY & PEAD PROFILING
# ======================================================================
LONG_CYCLE_KWS = ['auto', 'farm', 'heavy', 'metal', 'medical', 'conglomerates', 'machinery', 'agricultural', 'electric', 'software', 'aluminum', 'coal', 'defense', 'aerospace', 'cement', 'power', 'infrastructure', 'equipment', 'engineering', 'construction', 'minerals']
SHORT_CYCLE_KWS = ['insurance', 'capital market', 'chemical', 'tobacco', 'power producer', 'gas', 'tools', 'real estate', 'realty', 'lodging', 'diagnostic', 'broker', 'fmcg', 'consumer', 'retail', 'textile', 'apparel', 'food', 'beverage', 'leisure', 'media', 'entertainment', 'finance', 'banks']

def get_cyclicity(sector_str):
    s = str(sector_str).lower()
    for kw in LONG_CYCLE_KWS:
        if kw in s: return "🕰️ Long (-20%)"
    for kw in SHORT_CYCLE_KWS:
        if kw in s: return "⚡ Short (-8%)"
    return "⚖️ Mid (-12%)"

_pead_df = None
def get_pead_edge(sector_str):
    global _pead_df
    if _pead_df is None:
        path = "analysis_2026/earnings_shocks/industry_drift_analysis.csv"
        if os.path.exists(path):
            _pead_df = pd.read_csv(path)
        else:
            return "Unknown"
            
    if _pead_df is None or _pead_df.empty: return "Unknown"
    
    s = str(sector_str).lower()
    matches = _pead_df[_pead_df['Industry'].str.lower().str.contains(s.split()[0][:4], case=False, na=False)] 
    
    if "bank" in s: matches = _pead_df[_pead_df['Industry'].str.contains("Bank", case=False, na=False)]
    elif "auto" in s: matches = _pead_df[_pead_df['Industry'].str.contains("Auto", case=False, na=False)]
    elif "pharma" in s or "biotech" in s: matches = _pead_df[_pead_df['Industry'].str.contains("Drug|Biotech", case=False, na=False)]
    elif "it -" in s: matches = _pead_df[_pead_df['Industry'].str.contains("Software|Information Tech", case=False, na=False)]
    elif "fmcg" in s or "consumer durables" in s: matches = _pead_df[_pead_df['Industry'].str.contains("Household|Packaged Food", case=False, na=False)]
    elif "chemical" in s: matches = _pead_df[_pead_df['Industry'].str.contains("Chemical", case=False, na=False)]
    elif "telecom" in s: matches = _pead_df[_pead_df['Industry'].str.contains("Telecom", case=False, na=False)]
    elif "construction" in s or "engineering" in s: matches = _pead_df[_pead_df['Industry'].str.contains("Infrastructure|Engineering", case=False, na=False)]
    elif "metal" in s: matches = _pead_df[_pead_df['Industry'].str.contains("Steel|Aluminum|Metal", case=False, na=False)]
    elif "healthcare" in s: matches = _pead_df[_pead_df['Industry'].str.contains("Medical|Diagnostics", case=False, na=False)]
    
    if not matches.empty:
        c = matches.iloc[0]['Classification']
        behavior = c.split('(')[0].strip()
        if "FRONT" in behavior: return "🔴 Fade (Front-Run)"
        elif "DRIFT" in behavior: return "🟢 Buy (Drifter)"
        else: return "🟡 Neutral (Priced-In)"
        
    return "Unknown"


# ======================================================================
# 4. LIVE DNA3-V3.1 SCANNER (With Seasonal Indicators)
# ======================================================================
def generate_v3_watchlist(market_df, max_results=15):
    """
    Applies pure V3.1 relative strength criteria. 
    Appends the V4 Seasonal Array as an INFORMATIONAL INDICATOR only,
    never as a hard veto, because recent correlations have broken down.
    """
    if market_df is None or market_df.empty:
        return pd.DataFrame()
        
    current_month = datetime.now().month
    month_name = datetime.now().strftime("%B")
    
    results = []
    
    for _, row in market_df.iterrows():
        score = row.get('trend_score', 0)
        # We only want legit momentum setups (V3.1 baseline)
        if score < 70: 
            continue
            
        sector = str(row.get('sector', 'Unknown'))
        price = row.get('price', 0)
        volume_status = row.get('volume_signal_score', 0)
        
        # Must have adequate liquidity & accumulation
        if volume_status < 4 or price < 20:
            continue
            
        seasonal_action = get_seasonal_overlay(sector, current_month)
        
        # UI Badging for Seasonality
        if seasonal_action == 'TRAP':
            season_display = f"⚠️ HISTORICAL TRAP ({month_name})"
        elif seasonal_action == 'EDGE':
            season_display = f"🔥 GOLDEN MONTH ({month_name})"
        else:
            season_display = "Neutral"
            
        # Identify Sub-Industry metrics
        risk_profile = get_cyclicity(sector)
        pead_edge = get_pead_edge(sector)
            
        results.append({
            'Ticker': row['ticker'],
            'Target': row.get('name', row['ticker']),
            'Sector': sector,
            'Price': price,
            'V3_Score': score,
            'Cyclicity': risk_profile,
            'Seasonality': season_display,
            'PEAD_Edge': pead_edge,
            'Volume_Rating': volume_status / 10.0
        })
        
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values('V3_Score', ascending=False).head(max_results)
        
    return df_results
