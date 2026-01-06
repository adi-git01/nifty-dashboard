"""
4-Pillar Scoring System v3 - Sector-Aware
==========================================
1. QUALITY (25%)  - Business fundamentals, sector-adjusted weights
2. VALUE (25%)    - Sector-relative valuation (P/B for banks, P/E for others)
3. GROWTH (25%)   - Revenue & earnings momentum
4. MOMENTUM (25%) - Price momentum across timeframes
"""

import numpy as np
from utils.sector_profiles import get_sector_profile, get_quality_config, get_value_config, QUALITY_PROFILES

def normalize(value, min_val, max_val):
    """Normalizes a value to 0-10 scale."""
    if value is None:
        return 5  # Neutral if missing
    try:
        value = float(value)
    except:
        return 5
    if value < min_val:
        return 0
    if value > max_val:
        return 10
    return ((value - min_val) / (max_val - min_val)) * 10


def safe_num(val, default=0):
    """Safely convert to float."""
    if val is None:
        return default
    try:
        return float(val)
    except:
        return default


def calculate_scores(data, sector_pe_median=None, sector=None):
    """
    Calculates 4-pillar scores for a stock with sector-aware adjustments.
    
    Args:
        data: Dict with stock metrics
        sector_pe_median: Median PE for the stock's sector (for relative valuation)
        sector: Sector name (e.g., "Private Sector Bank", "Gems Jewellery And Watches")
    
    Returns:
        Dict with pillar scores, overall score, and sector_profile
    """
    if not data:
        return {"quality": 5, "value": 5, "growth": 5, "momentum": 5, "overall": 5, "sector_profile": "DEFAULT"}
    
    # Get sector from data if not provided
    if not sector:
        sector = data.get("sector", "")
    
    # Get sector profile and config
    profile = get_sector_profile(sector)
    quality_config = get_quality_config(sector)
    value_config = get_value_config(sector)


    # =========================================
    # PILLAR 1: QUALITY (Sector-Aware Scoring)
    # =========================================
    weights = quality_config["weights"]
    ranges = quality_config.get("ranges", {})
    bonuses = quality_config.get("bonuses", {})
    
    # Extract raw metrics
    roe = safe_num(data.get("roe"), 0)
    roe_pct = roe * 100 if roe < 1 else roe
    
    roa = safe_num(data.get("roa"), 0)
    roa_pct = roa * 100 if roa < 1 else roa
    
    npm = safe_num(data.get("profitMargins"), 0)
    npm_pct = npm * 100 if npm < 1 else npm
    
    gpm = safe_num(data.get("grossMargins"), 0)
    gpm_pct = gpm * 100 if gpm < 1 else gpm
    
    de_raw = data.get("debtToEquity")
    
    if de_raw is None:
        s_debt = 5  # Neutral penalty for missing data
        de = 0 # Default for other calculations if needed, but score is fixed
    else:
        de = safe_num(de_raw, 0)
        de = de / 100 if de > 10 else de
        s_debt = 10 - normalize(de, 0, 2.0)
    
    eq = safe_num(data.get("earningsQuality"), 1.0)
    pb = safe_num(data.get("pb"), 3.0)
    
    # Calculate component scores using profile-specific ranges
    roe_range = ranges.get("roe", (5, 25))
    s_roe = normalize(roe_pct, roe_range[0], roe_range[1])
    
    roa_range = ranges.get("roa", (2, 12))
    s_roa = normalize(roa_pct, roa_range[0], roa_range[1])
    
    npm_range = ranges.get("npm", (0, 18))
    s_npm = normalize(npm_pct, npm_range[0], npm_range[1])
    
    gpm_range = ranges.get("gpm", (15, 50))
    s_gpm = normalize(gpm_pct, gpm_range[0], gpm_range[1])
    
    # s_debt is already calculated above
    s_eq = normalize(eq, 0.5, 1.5)
    
    # P/B bonus for banks (low P/B = value)
    s_pb_bonus = 0
    if weights.get("pb_bonus", 0) > 0:
        pb_threshold = bonuses.get("low_pb", {}).get("threshold", 1.5)
        if pb < pb_threshold and pb > 0:
            s_pb_bonus = 10
        elif pb < 3:
            s_pb_bonus = 5
        else:
            s_pb_bonus = 0
    
    # Quality Score: Weighted average using profile weights
    quality = (
        s_roe * weights.get("roe", 0.30) +
        s_roa * weights.get("roa", 0.15) +
        s_npm * weights.get("npm", 0.18) +
        s_gpm * weights.get("gpm", 0.10) +
        s_debt * weights.get("debt", 0.15) +
        s_eq * weights.get("eq", 0.12) +
        s_pb_bonus * weights.get("pb_bonus", 0)
    )
    
    # Apply profile-specific bonuses
    if "roe_excellent" in bonuses:
        if roe_pct > bonuses["roe_excellent"]["threshold"]:
            quality = min(10, quality + bonuses["roe_excellent"]["bonus"])
    
    if "npm_excellent" in bonuses:
        if npm_pct > bonuses["npm_excellent"]["threshold"]:
            quality = min(10, quality + bonuses["npm_excellent"]["bonus"])
    
    if "npm_decent" in bonuses:
        if npm_pct > bonuses["npm_decent"]["threshold"]:
            quality = min(10, quality + bonuses["npm_decent"]["bonus"])
    
    if "roa_excellent" in bonuses:
        if roa_pct > bonuses["roa_excellent"]["threshold"]:
            quality = min(10, quality + bonuses["roa_excellent"]["bonus"])
    
    if "gpm_excellent" in bonuses:
        if gpm_pct > bonuses["gpm_excellent"]["threshold"]:
            quality = min(10, quality + bonuses["gpm_excellent"]["bonus"])
    
    if "low_debt" in bonuses:
        if de * 100 < bonuses["low_debt"]["threshold"]:
            quality = min(10, quality + bonuses["low_debt"]["bonus"])
    
    if "profitability_combo" in bonuses:
        b = bonuses["profitability_combo"]
        if roe_pct > b["roe_threshold"] and npm_pct > b["npm_threshold"]:
            quality = min(10, quality + b["bonus"])


    # ==========================================
    # PILLAR 2: VALUE (Sector-Aware Valuation)
    # ==========================================
    pe = safe_num(data.get("pe"), 25)
    forward_pe = safe_num(data.get("forwardPE"), pe)
    peg = safe_num(data.get("pegRatio"), 1.5)
    pb = safe_num(data.get("pb"), 3)
    
    # Get sector-specific value weights
    v_pb_weight = value_config.get("pb_weight", 0.15)
    v_pe_weight = value_config.get("pe_weight", 0.40)
    v_peg_weight = value_config.get("peg_weight", 0.25)
    v_fwd_weight = value_config.get("fwd_pe_weight", 0.20)
    pb_range = value_config.get("pb_range", (0.5, 5.0))
    use_pb_primary = value_config.get("use_pb_primary", False)
    
    # Handle negative PE (loss-making companies) - they get 0 for PE-based metrics
    if pe <= 0:
        s_rel_pe = 0  # Loss-making = no value score from PE
        s_fwd = 0
        s_peg = 0
    else:
        # Sector-Relative PE (if available)
        if sector_pe_median and sector_pe_median > 0:
            relative_pe = pe / sector_pe_median
            s_rel_pe = 10 - normalize(relative_pe, 0.5, 1.8) 
        else:
            # Absolute PE scoring: <12 cheap, >50 expensive
            s_rel_pe = 10 - normalize(pe, 10, 60)
        
        # Forward PE discount
        if forward_pe > 0:
            pe_discount = (pe - forward_pe) / pe
            s_fwd = normalize(pe_discount, -0.2, 0.3)
        else:
            s_fwd = 5
        
        # PEG scoring
        if peg <= 0 or peg > 5:
            s_peg = 0
        else:
            s_peg = 10 - normalize(peg, 0.5, 3.0)
    
    # P/B scoring with sector-specific range
    if pb <= 0:
        s_pb = 5
    else:
        s_pb = 10 - normalize(pb, pb_range[0], pb_range[1])
    
    # Value Score: Use sector-specific weights
    # Banks: P/B = 50%, PE = 20%, PEG = 15%, Fwd = 15%
    # Others: PE = 40%, PEG = 25%, Fwd = 20%, P/B = 15%
    value = (s_rel_pe * v_pe_weight) + (s_fwd * v_fwd_weight) + (s_peg * v_peg_weight) + (s_pb * v_pb_weight)


    # ===============================
    # PILLAR 3: GROWTH (Earnings Trend)
    # ===============================
    # Revenue Growth (YoY)
    rev_g = safe_num(data.get("revenueGrowth"), 0) * 100
    s_rev = normalize(rev_g, -5, 25)
    
    # Earnings Growth (from API)
    earn_g = safe_num(data.get("earningsGrowth"), 0) * 100
    s_earn = normalize(earn_g, -10, 40)
    
    # Earnings Trend (from financials - YoY Net Income)
    earn_trend = safe_num(data.get("earningsTrend"), 0) * 100
    s_trend = normalize(earn_trend, -20, 50)
    
    # Quarterly Earnings Growth
    qtr_g = safe_num(data.get("earningsQuarterlyGrowth"), 0) * 100
    s_qtr = normalize(qtr_g, -15, 30)
    
    # Growth Score: Weighted average
    growth = (s_rev * 0.30) + (s_earn * 0.25) + (s_trend * 0.25) + (s_qtr * 0.20)
    
    # Bonus for consistent growth (revenue + earnings both positive)
    if rev_g > 5 and earn_g > 10:
        growth = min(10, growth + 1.0)

    # ================================
    # PILLAR 4: MOMENTUM (Price Trend)
    # ================================
    r1w = safe_num(data.get("return_1w"), 0)
    r1m = safe_num(data.get("return_1m"), 0)
    r3m = safe_num(data.get("return_3m"), 0)
    
    # Multi-period returns - CALIBRATED RANGES (More intuitive scoring)
    # 1 week: -5% to +10% (5% 1-week move is excellent, deserves high score)
    s_1w = normalize(r1w, -5, 10)
    # 1 month: -8% to +20% (5% monthly is solid/good, 10% is excellent)
    s_1m = normalize(r1m, -8, 20)
    # 3 months: -10% to +40% (15% quarterly is solid, 20%+ is strong)
    s_3m = normalize(r3m, -10, 40)
    
    # 52-week position (from trend score) - WIDENED for better granularity
    trend_score = safe_num(data.get("trend_score"), 50)
    s_trend_pos = normalize(trend_score, 15, 90)
    
    # Momentum Score: Weighted average
    momentum = (s_1w * 0.15) + (s_1m * 0.30) + (s_3m * 0.30) + (s_trend_pos * 0.25)
    
    # Bonus for consistency (all positive across timeframes)
    if r1w > 0 and r1m > 0 and r3m > 0:
        momentum = min(10, momentum + 1.5)
    
    # Bonus for stocks near 52-week highs (strong technical position)
    if trend_score >= 80:
        momentum = min(10, momentum + 0.5)
    
    # Penalty for SEVERE reversal only (major short-term breakdown, not minor pullbacks)
    # Only penalize if 1-week is down significantly (-7%+) while 3M is strong
    elif r1w < -7 and r3m > 20:
        momentum = max(0, momentum - 1.0)

    # ===============
    # OVERALL SCORE
    # ===============
    overall = (quality * 0.25) + (value * 0.25) + (growth * 0.25) + (momentum * 0.25)

    return {
        "quality": round(quality, 1),
        "value": round(value, 1),
        "growth": round(growth, 1),
        "momentum": round(momentum, 1),
        "overall": round(overall, 1),
        "sector_profile": profile
    }


def calculate_trend_metrics(data):
    """
    Calculates a technical Trend Score (0-100) and Signal.
    Based on price position relative to MAs and 52-week range.
    """
    if not data:
        return {"trend_score": 50, "trend_signal": "NEUTRAL", "dist_52w": 0}
    
    price = safe_num(data.get("currentPrice") or data.get("price"), 0)
    ma50 = safe_num(data.get("fiftyDayAverage"), 0)
    ma200 = safe_num(data.get("twoHundredDayAverage"), 0)
    high_52 = safe_num(data.get("fiftyTwoWeekHigh"), 0)
    low_52 = safe_num(data.get("fiftyTwoWeekLow"), 0)
    
    if price == 0 or high_52 == 0:
        return {"trend_score": 50, "trend_signal": "NEUTRAL", "dist_52w": 0}
    
    score = 50  # Start neutral
    
    # MA Position (40 pts max)
    if ma50 > 0:
        if price > ma50:
            score += 15
        else:
            score -= 10
    
    if ma200 > 0:
        if price > ma200:
            score += 15
        else:
            score -= 15
    
    if ma50 > 0 and ma200 > 0:
        if ma50 > ma200:  # Golden cross setup
            score += 10
        else:  # Death cross setup
            score -= 5
    
    # 52-Week Position (30 pts max)
    if high_52 > low_52:
        range_52 = high_52 - low_52
        position = (price - low_52) / range_52  # 0 to 1
        score += int((position - 0.5) * 30)  # -15 to +15
        
        # Near 52-week high bonus
        dist_from_high = (price - high_52) / high_52 * 100
        if dist_from_high > -5:  # Within 5% of high
            score += 10
        elif dist_from_high < -30:  # More than 30% from high
            score -= 10
    else:
        dist_from_high = 0
    
    # Clamp score
    score = max(0, min(100, score))
    
    # Determine signal
    if score >= 75:
        signal = "STRONG UPTREND"
    elif score >= 60:
        signal = "UPTREND"
    elif score >= 40:
        signal = "NEUTRAL"
    elif score >= 25:
        signal = "DOWNTREND"
    else:
        signal = "STRONG DOWNTREND"
    
    # Calculate 200DMA distance
    dist_200dma = ((price - ma200) / ma200 * 100) if ma200 > 0 else 0
    
    return {
        "trend_score": int(score),
        "trend_signal": signal,
        "dist_52w": round(dist_from_high, 2),
        "dist_200dma": round(dist_200dma, 2)
    }
