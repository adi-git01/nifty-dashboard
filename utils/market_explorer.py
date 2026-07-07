
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from utils.data_engine import get_stock_info
from utils.scoring import calculate_scores, calculate_trend_metrics
from utils.analytics_engine import calculate_cycle_position, analyze_stock_health
from utils.visuals import chart_score_radar, chart_gauge
from utils.ui_components import card_metric, card_verdict
import time
import os

# Tickers that no longer trade standalone after a merger/renaming, mapped to
# the entity investors actually mean today. HDFC Ltd merged into HDFC Bank
# in July 2023 and delisted — "HDFC" is the single most common gotcha here.
TICKER_ALIASES = {
    "HDFC": "HDFCBANK",
}


def _suggest_tickers(query, limit=5):
    """Best-effort ticker/company-name suggestions when a lookup fails."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "nifty1000_list.csv")
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    q = query.upper()
    by_ticker = df[df["Ticker"].str.upper().str.startswith(q)]
    by_name = df[df["Company_Name"].str.upper().str.contains(q, na=False)]
    hits = pd.concat([by_ticker, by_name]).drop_duplicates(subset="Ticker")
    return list(hits["Ticker"].str.replace(".NS", "", regex=False).head(limit))


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
        raw = query.strip().upper()
        # Redirect known merged/renamed tickers (e.g. HDFC -> HDFCBANK) before
        # appending the exchange suffix, so a delisted-entity typo doesn't
        # just dead-end with "check the symbol".
        base = raw.split(".")[0]
        redirected = TICKER_ALIASES.get(base)
        ticker = redirected if redirected else raw
        # Smart suffix logic
        if not ticker.endswith(".NS") and not ticker.endswith(".BO") and not any(c.isdigit() for c in ticker) and "." not in ticker:
             # Assume NSE by default for simple alpha strings
             ticker += ".NS"

        if redirected:
            st.info(f"ℹ️ **{base}** no longer trades standalone (merged) — showing **{ticker}** instead.")

        with st.spinner(f"🔍 Deep Diving into {ticker}..."):
            # A. Fetch Data
            try:
                info = get_stock_info(ticker)
            except Exception as e:
                info = None

            if not info or not info.get("currentPrice"):
                st.error(f"❌ Could not fetch data for **{ticker}**. Please check the symbol.")
                suggestions = _suggest_tickers(base)
                if suggestions:
                    st.caption(f"Did you mean: {', '.join(suggestions)}?")
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
            
            # ── 6-Month CompRS History ────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 📈 6-Month Composite RS vs Nifty")
            st.caption("Daily CompRS = RS5×30% + RS21×50% + RS63×20% vs ^NSEI | Green = outperforming, Red = underperforming")

            @st.cache_data(ttl=3600, show_spinner=False)
            def _fetch_comprs_history(tkr):
                try:
                    raw = yf.download(
                        [tkr, "^NSEI"], period="400d", interval="1d",
                        group_by="ticker", threads=False, progress=False, auto_adjust=True
                    )
                    if isinstance(raw.columns, pd.MultiIndex):
                        s_close = raw[tkr]["Close"].dropna()
                        n_close = raw["^NSEI"]["Close"].dropna()
                    else:
                        return pd.DataFrame()
                    if s_close.index.tz is not None:
                        s_close.index = s_close.index.tz_localize(None)
                    if n_close.index.tz is not None:
                        n_close.index = n_close.index.tz_localize(None)
                    common = s_close.index.intersection(n_close.index)
                    s = s_close.reindex(common)
                    n = n_close.reindex(common)
                    rs5  = (s.pct_change(5)  - n.pct_change(5))  * 100
                    rs21 = (s.pct_change(21) - n.pct_change(21)) * 100
                    rs63 = (s.pct_change(63) - n.pct_change(63)) * 100
                    comp = rs5 * 0.30 + rs21 * 0.50 + rs63 * 0.20
                    out = pd.DataFrame({
                        "date":    common,
                        "CompRS":  comp.values,
                        "RS5":     rs5.values,
                        "RS21":    rs21.values,
                        "RS63":    rs63.values,
                        "price":   s.values,
                    }).dropna(subset=["CompRS"])
                    return out.tail(126)   # last ~6 months of trading days
                except Exception:
                    return pd.DataFrame()

            _crs_df = _fetch_comprs_history(ticker)

            if _crs_df.empty:
                st.info("CompRS history unavailable for this ticker.")
            else:
                _latest_crs = float(_crs_df["CompRS"].iloc[-1])
                _avg_crs    = float(_crs_df["CompRS"].mean())
                _pct_above  = (_crs_df["CompRS"] > 0).mean() * 100

                _m1, _m2, _m3 = st.columns(3)
                _m1.metric("Current CompRS", f"{_latest_crs:+.2f}%",
                           delta=f"{_latest_crs - float(_crs_df['CompRS'].iloc[-6]):+.2f}% vs 5d ago" if len(_crs_df) >= 6 else None)
                _m2.metric("6M Avg CompRS", f"{_avg_crs:+.2f}%")
                _m3.metric("Days Outperforming", f"{_pct_above:.0f}%",
                           help="% of last 6 months where CompRS > 0")

                # Build Plotly area chart
                _dates = _crs_df["date"]
                _vals  = _crs_df["CompRS"]
                _pos   = _vals.clip(lower=0)
                _neg   = _vals.clip(upper=0)

                _fig = go.Figure()
                _fig.add_trace(go.Scatter(
                    x=_dates, y=_pos, fill="tozeroy", mode="none",
                    fillcolor="rgba(0,200,83,0.25)", name="Above 0"
                ))
                _fig.add_trace(go.Scatter(
                    x=_dates, y=_neg, fill="tozeroy", mode="none",
                    fillcolor="rgba(213,0,0,0.25)", name="Below 0"
                ))
                _fig.add_trace(go.Scatter(
                    x=_dates, y=_vals, mode="lines",
                    line=dict(color="#42A5F5", width=1.5), name="CompRS"
                ))
                _fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)", line_width=1)
                _fig.update_layout(
                    height=240,
                    margin=dict(l=0, r=0, t=10, b=10),
                    showlegend=False,
                    yaxis=dict(title="CompRS %", zeroline=False, tickformat="+.1f"),
                    xaxis=dict(showgrid=False),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(_fig, use_container_width=True)

                # Mini breakdown table
                with st.expander("RS component breakdown", expanded=False):
                    _breakdown = _crs_df[["date", "RS5", "RS21", "RS63", "CompRS"]].tail(10).copy()
                    _breakdown["date"] = pd.to_datetime(_breakdown["date"]).dt.strftime("%d %b")
                    st.dataframe(
                        _breakdown.rename(columns={
                            "date": "Date", "RS5": "RS5 (1W)", "RS21": "RS21 (1M)",
                            "RS63": "RS63 (3M)", "CompRS": "CompRS"
                        }),
                        column_config={
                            "RS5 (1W)":  st.column_config.NumberColumn(format="%+.2f%%"),
                            "RS21 (1M)": st.column_config.NumberColumn(format="%+.2f%%"),
                            "RS63 (3M)": st.column_config.NumberColumn(format="%+.2f%%"),
                            "CompRS":    st.column_config.NumberColumn(format="%+.2f%%"),
                        },
                        hide_index=True, use_container_width=True
                    )

            st.markdown("---")
            if st.button("📄 Generate Structural Report", use_container_width=True):
                with st.spinner("Generating Institutional-Grade Report..."):
                    from utils.report_generator import generate_equity_report, generate_pdf_from_md
                    from utils.news_engine import fetch_latest_news
                    news = fetch_latest_news(ticker)
                    hist = yf.Ticker(ticker).history(period="1y")
                    report_md = generate_equity_report(ticker, info, scores, news, hist)
                    report_pdf = generate_pdf_from_md(report_md)
                    
                    st.success("Report Generated!")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button(
                            label="📥 Download Markdown Report",
                            data=report_md,
                            file_name=f"{ticker}_Research_Report.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                    with c2:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=report_pdf,
                            file_name=f"{ticker}_Research_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            type="primary"
                        )
                    
                    with st.expander("Preview Report", expanded=True):
                        st.markdown(report_md)
