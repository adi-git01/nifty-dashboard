"""
Professional Equity Research Report Generator
Inspired by institutional research format - 2 Page Summary
"""

import pandas as pd
from datetime import datetime

def safe_get(d, key, default=0):
    """Safely get a value from a dict, returning default if None or missing."""
    if d is None:
        return default
    val = d.get(key, default)
    return val if val is not None else default


def fmt(val, fmt_str="{:.2f}"):
    """Format a number safely."""
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "N/A"
        return fmt_str.format(val)
    except:
        return "N/A"


def fmt_cr(val):
    """Format value in Crores."""
    if val is None or val == 0:
        return "N/A"
    try:
        return f"₹{val/10000000:,.0f} Cr"
    except:
        return "N/A"


def fmt_pct(val):
    """Format as percentage."""
    if val is None:
        return "N/A"
    try:
        return f"{val*100:.1f}%"
    except:
        return "N/A"


def get_verdict(overall_score):
    """Determine investment verdict based on score."""
    if overall_score >= 7.5:
        return "🟢 STRONG BUY", "strong_buy"
    elif overall_score >= 6.5:
        return "🟢 BUY", "buy"
    elif overall_score >= 5.0:
        return "🟡 HOLD", "hold"
    elif overall_score >= 3.5:
        return "🟠 REDUCE", "reduce"
    else:
        return "🔴 AVOID", "avoid"


def get_score_bar(score, max_score=10):
    """Create a visual score bar."""
    filled = int(score)
    empty = max_score - filled
    return "█" * filled + "░" * empty


def generate_equity_report(ticker, info, scores, news_items, hist_data):
    """
    Generates a professional 2-page equity research report.
    Handles all edge cases where data might be missing.
    """
    
    # Safe data extraction
    if info is None:
        info = {}
    if scores is None:
        scores = {'quality': 5, 'value': 5, 'growth': 5, 
                  'momentum': 5, 'volume_signal_score': 5, 'overall': 5}
    
    # Extract key data
    company_name = info.get('longName', ticker.replace('.NS', ''))
    sector = info.get('sector', 'N/A')
    industry = info.get('industry', 'N/A')
    cmp = safe_get(info, 'currentPrice', safe_get(info, 'regularMarketPrice', 0))
    market_cap = info.get('marketCap', 0)
    
    # Financial metrics
    pe = info.get('pe') or info.get('trailingPE')
    forward_pe = info.get('forwardPE')
    peg = info.get('pegRatio')
    pb = info.get('pb') or info.get('priceToBook')
    roe = safe_get(info, 'roe', 0)  # Our pre-fetched ROE
    roa = safe_get(info, 'roa', 0)  # Our pre-fetched ROA
    npm = safe_get(info, 'profitMargins', 0)
    gpm = safe_get(info, 'grossMargins', 0)
    debt_eq = safe_get(info, 'debtToEquity', 0)
    current_ratio = info.get('currentRatio')
    
    # Price data
    high_52w = info.get('fiftyTwoWeekHigh', 0)
    low_52w = info.get('fiftyTwoWeekLow', 0)
    avg_50d = info.get('fiftyDayAverage', 0)
    avg_200d = info.get('twoHundredDayAverage', 0)
    change_52w = safe_get(info, '52WeekChange', 0)
    
    # Scores
    overall = scores.get('overall', 5)
    verdict_text, verdict_class = get_verdict(overall)
    
    # Generate dynamic analysis based on 4-pillar scores
    strengths = []
    concerns = []
    
    # Quality Pillar
    if scores.get('quality', 5) >= 7:
        strengths.append("High-quality business with strong ROE, margins, and cash generation")
    elif scores.get('quality', 5) <= 4:
        concerns.append("Weak fundamentals - low profitability or high leverage")
    
    # Value Pillar
    if scores.get('value', 5) >= 7:
        strengths.append("Attractively valued relative to sector peers and growth")
    elif scores.get('value', 5) <= 4:
        concerns.append("Expensive valuation compared to sector - limited margin of safety")
    
    # Growth Pillar
    if scores.get('growth', 5) >= 7:
        strengths.append("Strong revenue and earnings growth trajectory")
    elif scores.get('growth', 5) <= 4:
        concerns.append("Stagnant or declining growth - earnings trend concerning")
    
    # Momentum Pillar
    if scores.get('momentum', 5) >= 7:
        strengths.append("Strong price momentum with positive trend across timeframes")
    elif scores.get('momentum', 5) <= 4:
        concerns.append("Weak price action - technical setup is negative")

    # Volume Pillar
    vol_score = scores.get('volume_signal_score', 5)
    if vol_score >= 7:
        strengths.append("High volume score indicates strong institutional accumulation")
    elif vol_score <= 3:
        concerns.append("Low volume/distribution signals - institutional participation is weak")
    
    # Growth metrics for report
    rev_growth = safe_get(info, 'revenueGrowth', 0) * 100
    earn_growth = safe_get(info, 'earningsGrowth', 0) * 100
    earn_qoq = safe_get(info, 'earningsQuarterlyGrowth', 0) * 100
    opm = safe_get(info, 'operatingMargins', 0) * 100
    eq = safe_get(info, 'earningsQuality', 1.0)
    
    # Calculate proximity to 52W high/low
    if cmp and high_52w:
        pct_from_high = ((cmp / high_52w) - 1) * 100
    else:
        pct_from_high = 0
    
    if cmp and low_52w and low_52w > 0:
        pct_from_low = ((cmp / low_52w) - 1) * 100
    else:
        pct_from_low = 0

    # Build the report
    report = f"""
# 📊 EQUITY RESEARCH REPORT

---

## **{company_name}**
**{ticker}** | {sector} | {industry}

---

### 📅 Report Date: {datetime.now().strftime('%B %d, %Y')}

| Metric | Value |
|--------|-------|
| **Current Price** | ₹{fmt(cmp, "{:,.2f}")} |
| **Market Cap** | {fmt_cr(market_cap)} |
| **52W High / Low** | ₹{fmt(high_52w, "{:,.0f}")} / ₹{fmt(low_52w, "{:,.0f}")} |
| **Distance from 52W High** | {fmt(pct_from_high, "{:.1f}")}% |

---

## 📈 4-PILLAR INVESTMENT SCORECARD

| Pillar | Score | Rating | What It Measures |
|--------|-------|--------|------------------|
| **🔵 Quality** | {scores.get('quality', 5):.1f}/10 | {get_score_bar(scores.get('quality', 5))} | ROE, Margins, Debt, Earnings Quality |
| **💰 Value** | {scores.get('value', 5):.1f}/10 | {get_score_bar(scores.get('value', 5))} | PE vs Sector, PEG, Forward PE |
| **📈 Growth** | {scores.get('growth', 5):.1f}/10 | {get_score_bar(scores.get('growth', 5))} | Revenue & Earnings Trend |
| **🚀 Momentum** | {scores.get('momentum', 5):.1f}/10 | {get_score_bar(scores.get('momentum', 5))} | Price Action (1W/1M/3M) |
| **📊 Volume** | {scores.get('volume_signal_score', 5):.1f}/10 | {get_score_bar(scores.get('volume_signal_score', 5))} | Accumulation vs Distribution |
| **═══════════════════** | **═══════** | **═══════════════════** | |
| **OVERALL SCORE** | **{overall:.1f}/10** | **{get_score_bar(overall)}** | Equal-weighted average |

---

## 🎯 INVESTMENT VERDICT

# {verdict_text}

---

## 💪 KEY STRENGTHS

"""
    
    if strengths:
        for s in strengths:
            report += f"✅ {s}\n\n"
    else:
        report += "⚪ No exceptional strengths identified at current levels\n\n"
    
    report += """
## ⚠️ KEY CONCERNS

"""
    
    if concerns:
        for c in concerns:
            report += f"🔸 {c}\n\n"
    else:
        report += "⚪ No major concerns at current levels\n\n"

    report += f"""
---

## 📊 FINANCIAL SNAPSHOT

### Valuation Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| **P/E Ratio** | {fmt(pe)}x | {"Cheap (<15x)" if pe and pe < 15 else "Fair (15-25x)" if pe and pe < 25 else "Expensive (>25x)" if pe else "N/A"} |
| **Forward P/E** | {fmt(forward_pe)}x | {"Discounting growth" if forward_pe and pe and forward_pe < pe else "Stable expectations" if forward_pe else "N/A"} |
| **PEG Ratio** | {fmt(peg)} | {"Undervalued (<1)" if peg and peg < 1 else "Fair (1-2)" if peg and peg < 2 else "Overvalued (>2)" if peg else "N/A"} |
| **P/B Ratio** | {fmt(pb)}x | {"Asset play" if pb and pb < 1 else "Fair" if pb and pb < 3 else "Premium valuation" if pb else "N/A"} |

### Profitability Metrics

| Metric | Value | Benchmark |
|--------|-------|-----------|
| **Return on Equity (ROE)** | {fmt_pct(roe)} | Strong if >15% |
| **Return on Assets (ROA)** | {fmt_pct(roa)} | Good if >5% |
| **Net Profit Margin** | {fmt_pct(npm)} | Healthy if >10% |
| **Gross Margin** | {fmt_pct(gpm)} | Pricing power if >40% |

### Balance Sheet Health

| Metric | Value | Risk Level |
|--------|-------|------------|
| **Debt/Equity** | {fmt(debt_eq/100 if debt_eq else 0)}x | {"Low Risk" if debt_eq and debt_eq < 50 else "Moderate" if debt_eq and debt_eq < 100 else "High Risk" if debt_eq else "N/A"} |
| **Current Ratio** | {fmt(current_ratio)} | {"Strong" if current_ratio and current_ratio > 1.5 else "Adequate" if current_ratio and current_ratio > 1 else "Tight" if current_ratio else "N/A"} |

### 📈 Growth Analysis

| Metric | Value | Signal |
|--------|-------|--------|
| **Revenue Growth (YoY)** | {fmt(rev_growth, "{:.1f}")}% | {"🟢 Strong (>15%)" if rev_growth > 15 else "🟡 Moderate (5-15%)" if rev_growth > 5 else "🔴 Weak (<5%)"} |
| **Earnings Growth (YoY)** | {fmt(earn_growth, "{:.1f}")}% | {"🟢 Strong (>20%)" if earn_growth > 20 else "🟡 Moderate (0-20%)" if earn_growth > 0 else "🔴 Declining"} |
| **Earnings Growth (QoQ)** | {fmt(earn_qoq, "{:.1f}")}% | {"🟢 Accelerating" if earn_qoq > 10 else "🟡 Stable" if earn_qoq > -5 else "🔴 Decelerating"} |
| **Operating Margin** | {fmt(opm, "{:.1f}")}% | {"🟢 High (>20%)" if opm > 20 else "🟡 Moderate (10-20%)" if opm > 10 else "🔴 Low (<10%)"} |

---

## 📉 TECHNICAL POSITION

| Indicator | Value | Signal |
|-----------|-------|--------|
| **Price vs 50 DMA** | ₹{fmt(avg_50d, "{:,.0f}")} | {"🟢 Bullish" if cmp and avg_50d and cmp > avg_50d else "🔴 Bearish"} |
| **Price vs 200 DMA** | ₹{fmt(avg_200d, "{:,.0f}")} | {"🟢 Bullish" if cmp and avg_200d and cmp > avg_200d else "🔴 Bearish"} |
| **52 Week Change** | {fmt(change_52w*100 if change_52w else 0, "{:.1f}")}% | {"🟢 Outperformer" if change_52w and change_52w > 0.15 else "🟡 In-line" if change_52w and change_52w > -0.1 else "🔴 Underperformer"} |
| **From 52W High** | {fmt(pct_from_high, "{:.1f}")}% | {"🟢 Near High" if pct_from_high > -10 else "🟡 Correction" if pct_from_high > -25 else "🔴 Deep Correction"} |

---

## 📰 RECENT NEWS & DEVELOPMENTS

"""
    
    if news_items:
        for i, item in enumerate(news_items[:5]):
            title = item.get('title', 'News')
            link = item.get('link', '#')
            snippet = item.get('snippet', '')[:150]
            date = item.get('date', '')
            report += f"**{i+1}. [{title}]({link})**\n"
            report += f"   _{snippet}..._ ({date})\n\n"
    else:
        report += "*No recent news available for this company.*\n\n"
    
    report += f"""
---

## 🎯 INVESTMENT THESIS

Based on our multi-dimensional analysis, **{company_name}** receives an overall score of **{overall:.1f}/10**.

### Recommendation: {verdict_text}

**Key Takeaways:**
- The company operates in the **{sector}** sector with {"strong" if scores.get('quality', 5) >= 6 else "moderate" if scores.get('quality', 5) >= 4 else "weak"} fundamentals.
- Valuation is {"attractive" if scores.get('value', 5) >= 7 else "fair" if scores.get('value', 5) >= 5 else "stretched"} at current levels.
- Momentum is {"positive" if scores.get('momentum', 5) >= 6 else "neutral" if scores.get('momentum', 5) >= 4 else "negative"} with {"upside" if pct_from_high < -10 else "limited room"} to 52-week highs.

---

### ⚙️ Entry & Exit Strategy

| Action | Price Level | Rationale |
|--------|-------------|-----------|
| **Buy Zone** | ₹{fmt(cmp * 0.95 if cmp else 0, "{:,.0f}")} - ₹{fmt(cmp * 1.02 if cmp else 0, "{:,.0f}")} | Current levels ±5% |
| **Target Price** | ₹{fmt(high_52w * 0.95 if high_52w else 0, "{:,.0f}")} | Near 52W high |
| **Stop Loss** | ₹{fmt(cmp * 0.92 if cmp else 0, "{:,.0f}")} | 8% below entry |

---

*⚠️ Disclaimer: This report is auto-generated by Alpha Trend Engine for informational purposes only. Not financial advice. Please conduct your own research before investing.*

*Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M')}*
"""
    
    return report
