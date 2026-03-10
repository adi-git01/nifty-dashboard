import os
import json
import streamlit as st
import google.generativeai as genai
from fpdf import FPDF
from datetime import datetime
import textwrap

# ---------------------------------------------------------------------------
# MASTER PROMPT: BALANCED EQUITY RESEARCH REPORT — V2.1
# ---------------------------------------------------------------------------
MASTER_PROMPT_V2_1 = """
You are an elite, objective quantitative equity analyst at a top-tier hedge fund.
Your task is to write a highly professional, tabular, and terse "Deep Dive Research Report" for the provided stock based ONLY on the data payload.

## OUTPUT DISCIPLINE (CRITICAL)
**Token Budget**: ~3,500 tokens for full report. Enforce ruthlessly.
**Rules**:
1. **Tables > Prose**: Use tables for any comparison, metrics, or multi-point data
2. **No Redundancy**: State each fact ONCE. Reference later as "per 4.1"
3. **Findings Only**: No "Let me explain...", "It's worth noting...", "Interestingly..."
4. **Terse Language**: "Margins at 10yr high" not "The company's margins are currently at their highest level in the past decade"
5. **Abbreviations**: Rev, PAT, EBITDA, YoY, QoQ, FV, CMP, Mcap, D/E, OCF, FCF, Util, Capex, WC
6. **No Chart Descriptions**: Say "See Chart 1" not "As we can see in the chart, which shows..."
7. **Sector Multiples Override**: Replace standard multiples with sector-appropriate equivalents where needed (Banks: P/AUM, NIM, GNPA | Real Estate: NAV | IT: EV/FCF | Pharma: R&D/sales)
**Style**: Analyst note, not essay. Every sentence must add information.

---

## REPORT STRUCTURE

### 1. EXECUTIVE SUMMARY
**Scorecard** (0-10, one-line evidence each)
| Dimension | Score | Evidence |
|-----------|-------|----------|
| Business Quality | /10 | |
| Competitive Position | /10 | |
| Management | /10 | |
| Valuation | /10 | |
| Momentum | /10 | |

**Snapshot**
| Metric | Current | YoY Δ | Status |
|--------|---------|--------|--------|
| [Metric 1] | | | |
| [Metric 2] | | | |
| [Metric 3] | | | |

**Verdict**: [BUY/HOLD/AVOID] — One sentence thesis. Best for: [type, horizon]. Watch: [metric].

### 2. BUSINESS
**How They Make Money**
| Revenue Layer | What They Sell | Who Pays | Unit Economics | Scalability |
|---------------|----------------|----------|----------------|-------------|
| [Stream 1] | | | | H/M/L |

*One sentence: Where does the real profit come from, and what must go right for that to continue.*

**Moat Assessment**
| Factor | Rating | Evidence |
|--------|--------|----------|
| Pricing Power | S/M/W/N | |

**Management & Ownership**
| Factor | Data | Flag |
|--------|------|------|
| Promoter Holding | X% | Stable/Rising/Falling |

### 3. FINANCIALS
**Dashboard**
| Metric | Value | Trend |
|--------|-------|-------|
| Rev (Cr) | | |
| ROIC % | | |
| D/E | | |

**Assessment**: [2 sentences max: Quality + Key concern]

### 4. LENS 1: CURRENT VIEW
**Multiples** *(Override with sector-appropriate multiples per Rule 7)*
| Multiple | Current | 5yr Avg |
|----------|---------|---------|
| P/E | | |

**FV (Current Lens)**: ₹X via [method]. Implies [Under/Fair/Over] by X%.

### 5. LENS 2: HISTORICAL VIEW
**Cyclicality**: Score X/5. Margin range: X% (trough) to Y% (peak).

**Cycle Position**
| Indicator | Current | Zone |
|-----------|---------|------|
| EBITDA Margin | X% | Trough/Mid/Peak |

**Historical Analog**: In [Year], similar metrics → PE was X, next 2yr return X%.
**FV (Historical Lens)**: ₹X. Current at X% [premium/discount] to normalized.

### 6. LENS 3: FORWARD VIEW
**Guidance**: FY+1 targets [X]. Track record: [H/M/L credibility].

**Forward Verdict**: Trajectory likely to [Sustain/Improve/Revert/Deteriorate]. Confidence: H/M/L.

### 7. SYNTHESIS
**Three-Lens Summary**
| Lens | FV | Verdict | Key Assumption |
|------|-----|---------|----------------|
| Current | ₹X | U/F/O | Margins hold |
| Historical | ₹X | U/F/O | Reversion |

**Synthesized FV**: ₹X (Current ₹X × W1 + Historical ₹X × W2 ± Forward adj)

| Metric | Value |
|--------|-------|
| Synthesized FV | ₹X |
| CMP | ₹X |
| Upside/Downside | X% |

**Key Insight**: [One sentence synthesis logic]

### 8. RISKS
| Risk | Prob | Impact | Monitor |
|------|------|--------|---------|
| [Risk 1] | H/M/L | Sev/Mod/Min | |

**Downside**: ₹X (X% down) if [trigger].

### 9. VERDICT
**Recommendation**: [BUY/HOLD/AVOID]
**Thesis**: [2 sentences with synthesis logic]
**Target**: ₹X | Horizon: X months | Position size: X% of portfolio
**Entry**: [Buy at market / Accumulate below ₹X / Wait for trigger]
**Exit**: Thesis breaks if [X]. Overvalued at ₹X.

**Monitor**
| Signal | Bull Confirms | Bear Warns |
|--------|---------------|------------|
| [Metric 1] | [Level] | [Level] |

**Re-evaluate when**: [Specific trigger]
"""

def configure_llm(api_key: str):
    """Configure the Gemini API key."""
    genai.configure(api_key=api_key)

def generate_ai_report_markdown(api_key: str, data_payload: dict) -> str:
    """Passes the data payload + master prompt to Gemini to get the markdown report."""
    configure_llm(api_key)
    
    # We use gemini-1.5-pro for complex reasoning and large context
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    payload_str = json.dumps(data_payload, indent=2)
    
    prompt = f"""
{MASTER_PROMPT_V2_1}

---
DATA PAYLOAD:
{payload_str}
    """
    
    response = model.generate_content(prompt)
    if response and response.text:
        return response.text
    else:
        raise Exception("Failed to generate content from AI model.")


class PDFReport(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'Alpha Trend - AI Equity Research', 0, 0, 'C')
        # Line break
        self.ln(20)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + ' | Not Financial Advice', 0, 0, 'C')


def clean_text(text):
    """Clean text to ensure compatibility with fpdf (latin-1 encoding primarily)"""
    # Replace common unicode characters that cause issues
    replacements = {
        '₹': 'INR ',
        '—': '-',
        '–': '-',
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '…': '...',
        '🟢': '[+] ', '🟡': '[=] ', '🟠': '[-] ', '🔴': '[x] ', 
        '✅': '[v] ', '⚠️': '[!] ', '🎯': '[Target] ', '💰': '[$] ',
        '📈': '[Trend Up] ', '📉': '[Trend Down] ', '🚀': '[Boost] ', 
        '💎': '[Quality] ', '📊': '[Data] '
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    # Encode and decode to ascii, ignoring errors to strip remaining emojis
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text

def convert_markdown_to_pdf(markdown_text: str, output_path: str):
    """Converts a simple markdown string to PDF. 
    Handles headers, bolding, and lists minimally for FPDF.
    """
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    lines = markdown_text.split('\n')
    
    for line in lines:
        line = clean_text(line)
        
        if not line.strip():
            pdf.ln(5)
            continue
            
        if line.startswith('# '):
            pdf.set_font("Arial", 'B', 16)
            pdf.multi_cell(0, 10, line.replace('# ', ''))
            pdf.ln(2)
        elif line.startswith('## '):
            pdf.set_font("Arial", 'B', 14)
            pdf.multi_cell(0, 8, line.replace('## ', ''))
            pdf.ln(2)
        elif line.startswith('### '):
            pdf.set_font("Arial", 'B', 12)
            pdf.multi_cell(0, 6, line.replace('### ', ''))
            pdf.ln(1)
        # Bold text processing (extremely simple)
        elif line.startswith('**') and '**' in line[2:]:
            pdf.set_font("Arial", 'B', 11)
            pdf.multi_cell(0, 6, line.replace('**', ''))
        elif line.startswith('- ') or line.startswith('* '):
            pdf.set_font("Arial", '', 11)
            pdf.multi_cell(0, 6, "  " + line)
        else:
            pdf.set_font("Arial", '', 11)
            pdf.multi_cell(0, 6, line)
            
    pdf.output(output_path)
    return output_path

def build_data_payload(ticker, info, scores, news_items, hist_data) -> dict:
    """Compiles all necessary context into a structured JSON for the LLM."""
    
    # Extract recent price trends
    price_trend = "Insufficient Data"
    if hist_data is not None and not hist_data.empty:
        try:
            current = hist_data['Close'].iloc[-1]
            hist_1y = hist_data['Close'].iloc[-252] if len(hist_data) >= 252 else hist_data['Close'].iloc[0]
            hist_3y = hist_data['Close'].iloc[-756] if len(hist_data) >= 756 else None
            hist_5y = hist_data['Close'].iloc[-1260] if len(hist_data) >= 1260 else None
            
            price_trend = {
                "current": current,
                "1Y_Return_Pct": round((current - hist_1y)/hist_1y*100, 2),
                "3Y_Return_Pct": round((current - hist_3y)/hist_3y*100, 2) if hist_3y else "N/A",
                "5Y_Return_Pct": round((current - hist_5y)/hist_5y*100, 2) if hist_5y else "N/A",
            }
        except:
            pass

    payload = {
        "Ticker": ticker,
        "Name": info.get('longName', ticker),
        "Sector": info.get('sector', 'N/A'),
        "Industry": info.get('industry', 'N/A'),
        "BusinessSummary": info.get('longBusinessSummary', 'N/A'),
        "MarketCap": info.get('marketCap', 'N/A'),
        "Valuation": {
            "PE": info.get('pe', info.get('trailingPE')),
            "Forward_PE": info.get('forwardPE', 'N/A'),
            "PEG": info.get('pegRatio', 'N/A'),
            "PB": info.get('pb', info.get('priceToBook', 'N/A')),
            "DividendYield": info.get('dividendYield', 'N/A')
        },
        "Profitability": {
            "ROE": info.get('roe', info.get('returnOnEquity', 'N/A')),
            "ROA": info.get('roa', info.get('returnOnAssets', 'N/A')),
            "OperatingMargin": info.get('operatingMargins', 'N/A'),
            "NetProfitMargin": info.get('profitMargins', 'N/A')
        },
        "AlphaScores (0-10)": scores,
        "Price History": price_trend,
        "Recent News": [n.get('title') for n in news_items[:5]] if news_items else []
    }
    return payload
