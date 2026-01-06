
import streamlit as st
import pandas as pd
import yfinance as yf
from utils.data_engine import get_stock_info
from utils.scoring import calculate_scores, calculate_trend_metrics
from utils.analytics_engine import calculate_cycle_position, analyze_stock_health
from utils.visuals import chart_score_radar, chart_gauge
from utils.ui_components import card_metric, card_verdict
import time

def render_market_explorer():
    """
    Renders the Universal Market Explorer tab.
    Allows searching for any ticker and generating a full report.
    """
    st.markdown("## 🔍 Market Explorer")
    st.markdown("Analyze any stock (NSE/BSE/US) with the full Alpha Engine.")
    
    # 1. Search Bar
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Enter Ticker Symbol", placeholder="e.g. ZOMATO, TATASTEEL, TSLA, 500325.BO")
    with col2:
        analyze_btn = st.button("🚀 Analyze Strategy", type="primary", use_container_width=True)
        
    st.markdown("---")
    
    if not query:
        st.info("💡 **Tip**: For NSE stocks, you can just type the name (e.g. `PAYTM`). For BSE coverage, add `.BO`.")
        return

    # 2. Logic execution
    if query:
        ticker = query.strip().upper()
        # Smart suffix logic
        if not ticker.endswith(".NS") and not ticker.endswith(".BO") and not any(c.isdigit() for c in ticker) and "." not in ticker:
             # Assume NSE by default for simple alpha strings
             ticker += ".NS"
        
        with st.spinner(f"🔍 Deep Diving into {ticker}..."):
            # A. Fetch Data
            try:
                info = get_stock_info(ticker)
            except Exception as e:
                info = None
            
            if not info or not info.get("currentPrice"):
                st.error(f"❌ Could not fetch data for **{ticker}**. Please check the symbol.")
                return

            # B. Calculate Metrics & Scores
            # 1. Trend Metrics
            trend_data = calculate_trend_metrics(info)
            info.update(trend_data)
            
            # 2. Cycle Position
            cycle = calculate_cycle_position(ticker)
            
            # 3. Scores
            # For sector relative scoring, we might not have the precise sector median cached.
            # We use a conservative fallback of PE=30 if unknown.
            sector_median_pe = st.session_state.get('sector_pe_cache', {}).get(info.get('sector'), 30.0)
            
            scores = calculate_scores(info, sector_pe_median=sector_median_pe)
            
            # 4. Qualitative Analysis
            analysis = analyze_stock_health(info, scores)
            
            # === RENDER REPORT ===
            
            # Hero Section
            c1, c2, c3 = st.columns([2, 2, 2])
            with c1:
                st.markdown(f"### {info.get('name', ticker)}")
                st.caption(f"{info.get('sector', 'Unknown Sector')} • {info.get('industry', 'Unknown Industry')}")
            with c2:
                price = info.get('currentPrice', 0)
                st.markdown(f"### ₹{price:,.2f}")
            with c3:
                score = scores['overall']
                color = "green" if score >= 7 else "orange" if score >= 4 else "red"
                st.markdown(f"### Score: :{color}[{score}/10]")
            
            st.markdown("---")
            
            # Main Grid
            col_score, col_cycle = st.columns([1, 1])
            
            with col_score:
                st.markdown("#### 🎯 4-Pillar Scorecard")
                st.plotly_chart(chart_score_radar(scores), use_container_width=True)
                
                # Rationales
                with st.expander("📊 Score Analysis", expanded=True):
                    st.write(f"**Quality ({scores['quality']}/10)**: {analysis['rationales']['quality']}")
                    st.write(f"**Value ({scores['value']}/10)**: {analysis['rationales']['value']}")
                    st.write(f"**Growth ({scores['growth']}/10)**: {analysis['rationales']['growth']}")
                    st.write(f"**Momentum ({scores['momentum']}/10)**: {analysis['rationales']['momentum']}")

            with col_cycle:
                st.markdown("#### 🔄 Market Cycle Position")
                
                phase = cycle.get('phase', 'UNKNOWN')
                phase_map = {
                    "EARLY_RECOVERY": ("Early Recovery", "green"),
                    "MID_CYCLE": ("Mid Cycle", "blue"),
                    "LATE_CYCLE": ("Late Cycle", "orange"),
                    "DOWNTURN": ("Downturn", "red"),
                    "UNKNOWN": ("Unknown", "gray")
                }
                p_text, p_color = phase_map.get(phase, ("Unknown", "gray"))
                
                st.info(f"**Current Stage**: {p_text}")
                st.caption("Based on Operating Margin trends and deviation from historical average.")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Current OPM", f"{cycle.get('current_margin',0)}%")
                m2.metric("Avg OPM (5Y)", f"{cycle.get('avg_margin',0)}%")
                m3.metric("Trend", cycle.get('trend', 'Flat').title())

            # Key Financials
            st.markdown("#### 🏗️ Key Fundamentals")
            f1, f2, f3, f4, f5 = st.columns(5)
            f1.metric("P/E Ratio", f"{info.get('pe', 0):.1f}x")
            f2.metric("ROE", f"{info.get('roe', 0)*100:.1f}%")
            f3.metric("Earnings Growth", f"{info.get('earningsGrowth', 0)*100:.1f}%")
            f4.metric("Debt/Equity", f"{info.get('debtToEquity', 0):.2f}")
            f5.metric("PEG Ratio", f"{info.get('pegRatio', 0):.2f}")
            
