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
Your task is to write a highly professional, tabular, and terse "Deep Dive Research Report" for the provided stock.
You must be aggressively balanced: for every bullish point, find a bearish counter-weight. 

Do NOT use fluff, generic advice, or disclaimers. 
Do NOT output anything except the exact Markdown sections requested below.

DATA PAYLOAD EXPERT:
You will be provided with a JSON data payload containing the stock's 5-Year price trend, 6-quarter financials, DNA3 alpha scores, and recent news.

OUTPUT FORMAT INSTRUCTIONS (Markdown Only):

# [COMPANY NAME] ([TICKER]) - Equity Research Report
**Date:** [Current Date]
**Sector:** [Sector] | **Industry:** [Industry]
**Current Price:** ₹[Price] | **Market Cap:** [Market Cap] Cr

## 1. Executive Summary
Provide a 3-sentence, hard-hitting summary of the investment thesis. Be objective. Why is this stock interesting right now, and what is the glaring risk?

## 2. Quantitative Alpha Scorecard
Provide a markdown table summarizing the 5 pillars.
| Pillar | Score (out of 10) | Observation |
|---|---|---|
| Quality | ... | ... |
| Value | ... | ... |
| Growth | ... | ... |
| Momentum | ... | ... |
| Volume/Liquidity | ... | ... |

## 3. Financial & Fundamental Trajectory
Analyze the 6-quarter financial trajectory. Is the company expanding margins or bleeding cash? 
Use bullet points. Keep it extremely brief and data-rich.

## 4. The Bull vs Bear Case
Create a balanced 2-column table or two distinct sub-sections.
**Bull Case:** 3 strong catalysts/strengths.
**Bear Case:** 3 severe risks/weaknesses.

## 5. Technical & Cycle Setup
Analyze the 5-Year price history, distance from 52-week highs, moving averages, and any cyclical edge or Post-Earnings Announcement Drift (PEAD) behavior described in the payload.

## 6. Verdict & Action Plan
State an objective verdict (STRONG BUY, BUY, HOLD, REDUCE, AVOID). 
Define a strict Entry Zone, Target Price, and Stop Loss based on the data.
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
