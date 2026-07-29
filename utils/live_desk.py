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

def get_seasonal_guideline(sector_str):
    """Returns a string summarizing the month-by-month seasonality."""
    cat = classify_sector(sector_str)
    if cat not in SEASONAL_RULES:
        return "Seasonality Neutral"
        
    edges = []
    traps = []
    
    # Month abbreviation map
    m_abbr = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    
    for m, rule in SEASONAL_RULES[cat].items():
        if rule == 'EDGE':
            edges.append(m_abbr[m])
        elif rule == 'TRAP':
            traps.append(m_abbr[m])
            
    parts = []
    if edges:
        parts.append(f"🟢 EDGE: {','.join(edges)}")
    if traps:
        parts.append(f"🔴 TRAP: {','.join(traps)}")
        
    if not parts:
        return "Seasonality Neutral"
        
    return " | ".join(parts)

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

# Dashboard sector -> the yfinance industry names used in
# industry_drift_analysis.csv. The PEAD table is keyed by RAW yfinance
# industries ("Capital Markets", "Thermal Coal"), but this function is
# called with the dashboard's consolidated sector ("Capital Goods",
# "Pharma & Healthcare"). The old code bridged that gap by matching the
# first 4 characters of the first word, which silently produced nonsense:
# "Capital Goods" -> "Capital Markets" (capi), "Commercial Services &
# Supplies" -> "Communication Equipment" (comm), "Business Equipment &
# Supplies" -> "Specialty Business Services" (busi), "Apparel
# Manufacturing" -> "Apparel Retail". It then took .iloc[0] of an
# arbitrarily-ordered match set, so near-identical sectors ("Consumer
# Durables" vs "Consumer Goods") landed on opposite labels. Every PEAD
# label downstream was therefore close to noise.
#
# Keys cover all three vocabularies present in the logs (raw yfinance,
# Playbook-58, and the broad consolidated names), since the sector column
# changed meaning over time as the mapping was corrected.
_PEAD_SECTOR_MAP = {
    # --- Financials ---
    "banking": ["Banks - Regional"],
    "banks": ["Banks - Regional"],
    "financial services": ["Asset Management", "Credit Services", "Financial Conglomerates", "Mortgage Finance"],
    "finance": ["Credit Services", "Financial Conglomerates", "Mortgage Finance"],
    "capital markets": ["Capital Markets", "Financial Data & Stock Exchanges", "Asset Management"],
    "financial technology (fintech)": ["Financial Data & Stock Exchanges", "Credit Services"],
    "insurance": ["Insurance - Life", "Insurance - Diversified", "Insurance Brokers", "Insurance - Reinsurance"],
    # --- Technology / Telecom ---
    "it & technology": ["Software - Application", "Software - Infrastructure", "Information Technology Services"],
    "it - software": ["Software - Application", "Software - Infrastructure"],
    "it - services": ["Information Technology Services", "Specialty Business Services"],
    "it - hardware": ["Computer Hardware", "Electronic Components"],
    "telecom": ["Telecom Services", "Communication Equipment"],
    "telecom - services": ["Telecom Services"],
    "telecom - equipment & accessories": ["Communication Equipment"],
    # --- Healthcare ---
    "pharma & healthcare": ["Drug Manufacturers - Specialty & Generic", "Drug Manufacturers - General", "Biotechnology"],
    "pharmaceuticals & biotechnology": ["Drug Manufacturers - Specialty & Generic", "Drug Manufacturers - General", "Biotechnology"],
    "healthcare services": ["Medical Care Facilities", "Diagnostics & Research", "Health Information Services"],
    "healthcare equipment & supplies": ["Medical Instruments & Supplies"],
    # --- Consumer ---
    "consumer goods": ["Packaged Foods", "Household & Personal Products", "Confectioners"],
    "diversified fmcg": ["Packaged Foods", "Household & Personal Products"],
    "food products": ["Packaged Foods", "Confectioners"],
    "agricultural food & other products": ["Packaged Foods", "Agricultural Inputs"],
    "personal products": ["Household & Personal Products"],
    "household products": ["Household & Personal Products"],
    "beverages": ["Beverages - Wineries & Distilleries"],
    "cigarettes & tobacco products": ["Tobacco"],
    "consumer durables": ["Consumer Electronics", "Furnishings, Fixtures & Appliances", "Luxury Goods"],
    "retail": ["Internet Retail", "Apparel Retail", "Restaurants"],
    "retailing": ["Internet Retail", "Apparel Retail"],
    "textiles": ["Textile Manufacturing", "Apparel Manufacturing", "Footwear & Accessories"],
    "textiles & apparels": ["Textile Manufacturing", "Apparel Manufacturing", "Footwear & Accessories"],
    "hospitality": ["Lodging", "Restaurants", "Travel Services"],
    "leisure services": ["Lodging", "Restaurants", "Travel Services"],
    "other consumer services": ["Internet Content & Information", "Travel Services"],
    "media": ["Broadcasting", "Advertising Agencies", "Entertainment"],
    "entertainment": ["Entertainment", "Broadcasting"],
    "printing & publication": ["Advertising Agencies", "Broadcasting"],
    # --- Autos ---
    "auto": ["Auto Manufacturers", "Auto Parts"],
    "automobiles": ["Auto Manufacturers"],
    "auto components": ["Auto Parts"],
    "agricultural, commercial & construction vehicles": ["Farm & Heavy Construction Machinery"],
    # --- Industrials / Capital Goods ---
    "capital goods": ["Specialty Industrial Machinery", "Electrical Equipment & Parts", "Metal Fabrication", "Tools & Accessories"],
    "industrial manufacturing": ["Specialty Industrial Machinery", "Metal Fabrication"],
    "industrial products": ["Specialty Industrial Machinery", "Tools & Accessories", "Metal Fabrication"],
    "electrical equipment": ["Electrical Equipment & Parts"],
    "engineering services": ["Engineering & Construction"],
    "commercial services & supplies": ["Specialty Business Services", "Consulting Services"],
    "business equipment & supplies": ["Specialty Business Services"],
    "aerospace & defense": ["Aerospace & Defense"],
    # --- Infra / Realty / Materials ---
    "infrastructure": ["Engineering & Construction", "Infrastructure Operations"],
    "construction": ["Engineering & Construction", "Infrastructure Operations"],
    "transport infrastructure": ["Airports & Air Services", "Infrastructure Operations"],
    "transport services": ["Integrated Freight & Logistics", "Marine Shipping", "Railroads"],
    "real estate": ["Real Estate Services", "Real Estate - Development", "Real Estate - Diversified"],
    "realty": ["Real Estate - Development", "Real Estate Services", "Real Estate - Diversified"],
    "cement & materials": ["Building Materials", "Building Products & Equipment"],
    "cement & cement products": ["Building Materials"],
    "other construction materials": ["Building Materials", "Building Products & Equipment"],
    "paper, forest & jute products": ["Lumber & Wood Production"],
    # --- Chemicals / Energy / Metals / Utilities ---
    "chemicals": ["Specialty Chemicals", "Chemicals"],
    "chemicals & petrochemicals": ["Specialty Chemicals", "Chemicals"],
    "fertilizers & agrochemicals": ["Agricultural Inputs"],
    "oil & gas": ["Oil & Gas Integrated", "Oil & Gas Refining & Marketing"],
    "oil": ["Oil & Gas Integrated"],
    "petroleum products": ["Oil & Gas Refining & Marketing"],
    "gas": ["Utilities - Regulated Gas"],
    "consumable fuels": ["Thermal Coal"],
    "metals & mining": ["Steel", "Aluminum", "Copper", "Other Industrial Metals & Mining"],
    "ferrous metals": ["Steel"],
    "non - ferrous metals": ["Aluminum", "Copper"],
    "diversified metals": ["Other Industrial Metals & Mining"],
    "minerals & mining": ["Other Industrial Metals & Mining"],
    "metals & minerals trading": ["Other Industrial Metals & Mining"],
    "power & utilities": ["Utilities - Regulated Electric", "Utilities - Renewable", "Utilities - Independent Power Producers"],
    "power": ["Utilities - Regulated Electric", "Utilities - Renewable", "Utilities - Independent Power Producers"],
    "other utilities": ["Utilities - Regulated Electric"],
    # --- Misc ---
    "diversified": ["Conglomerates", "Financial Conglomerates"],
    # Raw yfinance labels that appear in older log rows but have no exact row
    # in the PEAD table; mapped to their closest analogue.
    "farm products": ["Agricultural Inputs", "Packaged Foods"],
    "publishing": ["Advertising Agencies", "Broadcasting"],
    "resorts & casinos": ["Lodging", "Restaurants"],
    # "Packaging & Containers" is deliberately left unmapped — there is no
    # honest analogue in the table, and Unknown is better than a guess.
}

_pead_df = None


def _pead_label(classification: str) -> str:
    b = str(classification).split('(')[0].strip().upper()
    if "FRONT" in b:
        return "🔴 Fade (Front-Run)"
    if "DRIFT" in b:
        return "🟢 Buy (Drifter)"
    return "🟡 Neutral (Priced-In)"


def get_pead_edge(sector_str):
    """
    Post-Earnings-Announcement-Drift behaviour for a sector.

    Resolution order: exact industry name -> curated sector map -> Unknown.
    When a sector maps to several industries the classification is decided by
    Shock_Events-weighted vote (the old code took an arbitrary .iloc[0]).

    NOTE: the shipped table contains only CONCURRENT / BALANCED / DRIFTERS —
    there is no FRONT-run class — so "🔴 Fade" is currently unreachable. The
    branch is kept for forward-compatibility if the analysis is regenerated
    with that class, rather than implying a Fade signal exists today.
    """
    global _pead_df
    if _pead_df is None:
        path = "analysis_2026/earnings_shocks/industry_drift_analysis.csv"
        if not os.path.exists(path):
            return "Unknown"
        try:
            _pead_df = pd.read_csv(path)
        except Exception:
            return "Unknown"

    if _pead_df is None or _pead_df.empty:
        return "Unknown"

    s = str(sector_str).strip().lower()
    if not s or s in ("unknown", "nan", "none"):
        return "Unknown"

    ind_lower = _pead_df['Industry'].astype(str).str.strip().str.lower()

    # 1. Exact industry match (log rows written when 'sector' held raw
    #    yfinance industry names).
    exact = _pead_df[ind_lower == s]
    if not exact.empty:
        return _pead_label(exact.iloc[0]['Classification'])

    # 2. Curated sector -> industries mapping.
    targets = _PEAD_SECTOR_MAP.get(s)
    if not targets:
        return "Unknown"
    sel = _pead_df[ind_lower.isin([t.strip().lower() for t in targets])]
    if sel.empty:
        return "Unknown"

    # 3. Shock_Events-weighted vote across the matched industries.
    w = pd.to_numeric(sel.get('Shock_Events'), errors='coerce').fillna(1.0).clip(lower=1.0)
    tally = {}
    for cls, weight in zip(sel['Classification'], w):
        tally[_pead_label(cls)] = tally.get(_pead_label(cls), 0.0) + float(weight)
    return max(tally.items(), key=lambda kv: kv[1])[0]


# ======================================================================
# 4. LIVE DNA3-V2.2 SCANNER (With Seasonal Indicators)
# ======================================================================
def generate_v3_watchlist(market_df, max_results=15):
    """
    Applies pure V2.2 relative strength criteria. 
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
        # We only want legit momentum setups (V2.2 baseline)
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
            'Volume_Rating': volume_status / 10.0,
            'Vol_Badge': row.get('vol_badge', '')
        })
        
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values('V3_Score', ascending=False).head(max_results)
        
    return df_results
