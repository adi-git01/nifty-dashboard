"""
Professional Equity Research Report Generator
Inspired by institutional research format - 2 Page Summary
"""

import pandas as pd
from datetime import datetime
import markdown
import re
from io import BytesIO
from fpdf import FPDF

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
    Generates a professional Balanced Equity Research Report - V2.1
    Optimized for LLM ingestion via PDF export constraint.
    """
    if info is None: info = {}
    if scores is None: scores = {'quality': 5, 'value': 5, 'growth': 5, 'momentum': 5, 'volume_signal_score': 5, 'overall': 5}
    
    # 1. CORE DATA EXTRACTION
    company_name = info.get('longName', ticker.replace('.NS', ''))
    sector = str(info.get('sector', 'N/A'))
    industry = str(info.get('industry', 'N/A'))
    cmp = safe_get(info, 'currentPrice', safe_get(info, 'regularMarketPrice', 0))
    mcap = info.get('marketCap', 0)
    
    # Financials
    pe = info.get('pe') or info.get('trailingPE')
    f_pe = info.get('forwardPE')
    peg = info.get('pegRatio')
    pb = info.get('pb') or info.get('priceToBook')
    roe_pct = safe_get(info, 'roe', 0) * 100
    roa_pct = safe_get(info, 'roa', 0) * 100
    npm_pct = safe_get(info, 'profitMargins', 0) * 100
    gpm_pct = safe_get(info, 'grossMargins', 0) * 100
    opm_pct = safe_get(info, 'operatingMargins', 0) * 100
    de = safe_get(info, 'debtToEquity', 0)
    
    # Growth
    rev_g = safe_get(info, 'revenueGrowth', 0) * 100
    earn_g = safe_get(info, 'earningsGrowth', 0) * 100
    qoq_g = safe_get(info, 'earningsQuarterlyGrowth', 0) * 100
    
    # Price
    high_52w = info.get('fiftyTwoWeekHigh', 0)
    low_52w = info.get('fiftyTwoWeekLow', 0)
    pct_from_high = ((cmp / high_52w) - 1) * 100 if cmp and high_52w else 0
    pct_from_low = ((cmp / low_52w) - 1) * 100 if cmp and low_52w else 0
    
    overall = scores.get('overall', 5)
    verdict, _ = get_verdict(overall)
    
    # Generate contextual placeholders based on known data
    promoter_holding = "High" if "PSU" in industry or "Bank" in industry else "Stable"
    
    # --- REPORT CONSTRUCTION ---
    
    report = f"""
# BALANCED EQUITY RESEARCH REPORT — V2.1

**{company_name} ({ticker})** | Sector: {sector} | Date: {datetime.now().strftime('%Y-%m-%d')}
**CMP**: ₹{fmt(cmp)} | **Mcap**: {fmt_cr(mcap)} | **52W H/L**: ₹{fmt(high_52w, "{:,.0f}")} / ₹{fmt(low_52w, "{:,.0f}")} (% from High: {fmt(pct_from_high)}%)

---

### 1. EXECUTIVE SUMMARY

**Scorecard**
| Dimension | Score | Evidence |
|-----------|-------|----------|
| Business Quality | {scores.get('quality', 5):.1f}/10 | ROE {fmt(roe_pct)}%, NPM {fmt(npm_pct)}% |
| Competitive Position | {scores.get('value', 5):.1f}/10 | PE {fmt(pe)}x, PEG {fmt(peg)} |
| Momentum | {scores.get('momentum', 5):.1f}/10 | Price trend vs 50DMA/200DMA |
| Volume/Institutional | {scores.get('volume_signal_score', 5):.1f}/10 | Recent accumulation signals |

**Snapshot**
| Metric | Current | YoY Δ | Status |
|--------|---------|--------|--------|
| Revenue Growth | {fmt(rev_g)}% | - | {"Strong" if rev_g > 15 else "Weak"} |
| Earnings Growth | {fmt(earn_g)}% | - | {"Strong" if earn_g > 20 else "Weak"} |
| Qtly Earnings Gr. | {fmt(qoq_g)}% | - | {"Accelerating" if qoq_g > 10 else "Decelerating"} |
| Operating Margin | {fmt(opm_pct)}% | - | {"High" if opm_pct > 15 else "Low"} |

**Verdict**: {verdict.split(' ')[1]} — Overall Score: {overall:.1f}/10.

---

### 2. BUSINESS

**How They Make Money**
| Revenue Layer | Core Segment | Unit Economics | Scalability |
|---------------|--------------|----------------|-------------|
| Primary | {industry} operations | {"High" if gpm_pct > 40 else "Medium"} Margins | M |
| Secondary | Ancillary services | Variable | H |

**Moat Assessment**
| Factor | Rating | Evidence |
|--------|--------|----------|
| Pricing Power | {"S" if gpm_pct > 40 else "M"} | Gross Margins at {fmt(gpm_pct)}% |
| Capital Efficiency | {"S" if roe_pct > 15 else "W"} | ROE at {fmt(roe_pct)}% |

**Management & Ownership**
| Factor | Data | Flag |
|--------|------|------|
| Promoter Holding | ~50% (Est) | {promoter_holding} |
| Debt/Equity | {fmt(de/100 if de else 0)}x | {"Green" if de < 50 else "Red"} |

---

### 3. FINANCIALS

**Dashboard**
| Metric | Current | Benchmark |
|--------|---------|-----------|
| P/E Ratio | {fmt(pe)}x | {"Cheap" if pe and pe < 15 else "Expensive" if pe and pe > 30 else "Fair"} |
| Forward P/E | {fmt(f_pe)}x | {"Discounting growth" if f_pe and pe and f_pe < pe else "Stable expectations"} |
| P/B Ratio | {fmt(pb)}x | {"Asset play" if pb and pb < 1 else "Premium" if pb and pb > 3 else "Fair"} |
| PEG Ratio | {fmt(peg)} | {"Undervalued (<1)" if peg and peg < 1 else "Fair/Overvalued"} |
| ROA | {fmt(roa_pct)}% | Good if >5% |

---

### 4. LENS 1: CURRENT VIEW

**Multiples**
| Multiple | Current | Note |
|----------|---------|------|
| P/E | {fmt(pe)} | vs Forward PE {fmt(f_pe)} |
| EV/EBITDA | N/A | Available via deeper filings |
| P/B | {fmt(pb)} | |

**FV (Current Lens)**: Fair value proxy aligns with PEG {fmt(peg)}.

---

### 5. LENS 2: HISTORICAL VIEW

**Cycle Position**
| Indicator | Current | Zone |
|-----------|---------|------|
| OPM Margin | {fmt(opm_pct)}% | {"Peak" if opm_pct > 20 else "Mid/Trough"} |
| Valuation (P/E) | {fmt(pe)} | {"Premium" if pe and pe > 25 else "Discount"} |

**Position**: Current momentum score of {scores.get('momentum', 5):.1f}/10 implies {"Late Stage Rally" if pct_from_high > -5 else "Early Recovery" if pct_from_low < 15 else "Mid-Cycle"}.

---

### 6. LENS 3: FORWARD VIEW

**Industry Shifts**
| Factor | Status | Impact |
|--------|--------|--------|
| Sector Momentum | {sector} | {"Tailwind" if scores.get('momentum', 5) > 6 else "Headwind"} |
| Earnings Trajectory | {fmt(f_pe)}x Fwd PE | {"Earnings Expansion" if f_pe and pe and f_pe < pe else "Earnings Contraction"} |

---

### 7. SYNTHESIS

**Synthesized View**
| Metric | Value |
|--------|-------|
| Overall Alpha Score | {overall:.1f}/10 |
| CMP | ₹{fmt(cmp)} |
| Distance to 52W High | {fmt(pct_from_high)}% |

---

### 8. RISKS
| Risk | Impact | Monitor |
|------|--------|---------|
| Valuation Risk | {"High" if pe and pe > 35 else "Low"} | Forward P/E expansion |
| Leverage Risk | {"High" if de and de > 100 else "Low"} | Debt/Equity ratio |
| Momentum Break | Mod | 50DMA support levels |

---

### 9. VERDICT

**Recommendation**: {verdict}

**Thesis**: Company scores {overall:.1f}/10 on the Alpha Engine. Quality pillar is at {scores.get('quality', 5):.1f}, powered by ROE of {fmt(roe_pct)}% and Net Margins of {fmt(npm_pct)}%. Growth trajectory is currently evaluated at {scores.get('growth', 5):.1f}/10.

**Monitor**
| Signal | Bull Confirms | Bear Warns |
|--------|---------------|------------|
| Price Action | Breakout above 52W High | Breakdown below 200DMA |
| Earnings | Growth > 15% YoY | Margin contraction |
"""
    return report


def generate_pdf_from_md(md_content):
    """
    Converts the generated Markdown report into a PDF byte stream
    using fpdf2 and the markdown library, stripping out emojis 
    which fpdf struggles to render without custom fonts.
    """
    # 1. Strip emojis and some difficult markdown elements for PDF safety
    # FPDF default fonts only support latin-1
    clean_md = md_content
    # Remove emojis (simple regex for most common emojis used in the report)
    clean_md = re.sub(r'[🟢🟡🟠🔴📊💰🚀✅💪⚠️📉📰🎯⚙️📈]', '', clean_md)
    clean_md = clean_md.replace('═══════════════════', '-------------------')
    clean_md = clean_md.replace('═══════', '-------')
    clean_md = clean_md.replace('█', '#')
    clean_md = clean_md.replace('░', '-')
    clean_md = clean_md.replace('₹', 'Rs.')
    
    # 2. Convert stripped Markdown to HTML
    html_content = markdown.markdown(clean_md, extensions=['tables'])
    
    # 3. Use FPDF to convert HTML to PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=10)
    pdf.write_html(html_content)
    
    # Return as bytes
    return pdf.output(dest='S')
