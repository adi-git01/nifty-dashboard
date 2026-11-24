"""Compare Stocks, Deep Dive, Notes, and AI Capex page views (extracted from main.py)."""
from __future__ import annotations

from datetime import datetime, timedelta
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from utils.data_engine import get_stock_history, get_stock_info
from utils.news_engine import fetch_latest_news
from utils.nifty500_list import TICKERS
from utils.scoring import calculate_scores, calculate_trend_metrics
from utils.ui_components import COLORS, page_header
from utils.visuals import chart_gauge, chart_price_history, chart_score_radar


@st.cache_data(ttl=3600)
def _nifty_rs_refs(_nifty_recent_fn):
    try:
        close = _nifty_recent_fn("100d")['Close'].dropna()
        if len(close) < 65:
            return 0.0, 0.0, 0.0
        n_1w = (close.iloc[-1] - close.iloc[-6])  / close.iloc[-6]  * 100
        n_1m = (close.iloc[-1] - close.iloc[-22]) / close.iloc[-22] * 100
        n_3m = (close.iloc[-1] - close.iloc[-64]) / close.iloc[-64] * 100
        return round(n_1w, 2), round(n_1m, 2), round(n_3m, 2)
    except Exception:
        return 0.0, 0.0, 0.0


def render_compare_stocks(df: pd.DataFrame) -> None:
    from utils import funnel, funnel_view, stock_compare
    from utils.industry_board_view import get_board
    from utils.yf_safe import safe_history

    st.markdown(
        page_header(
            "⚖️ Stock Comparison & Decision Matrix",
            "Compare 2–4 stocks head-to-head across Funnel Stages, Scanner RS, Technicals v6, Quarterly Earnings, and Valuation v4.1",
        ),
        unsafe_allow_html=True,
    )

    if df is None or df.empty or "ticker" not in df.columns:
        st.warning("No market snapshot loaded.")
        return

    available_tickers = sorted(df["ticker"].dropna().astype(str).unique().tolist())
    if len(available_tickers) < 2:
        st.info("Need at least 2 tickers in the universe to compare.")
        return

    # Quick presets
    p1, p2, _ = st.columns([1.4, 1.8, 2.8])
    if p1.button("🏆 Top 3 Funnel Picks", key="cmp_preset_top3", use_container_width=True):
        try:
            T_all = funnel_view.get_table(df)
            picks = funnel.run(T_all)["picks"]["ticker"].tolist()[:3]
            if len(picks) >= 2:
                st.session_state["compare_tickers"] = picks
                st.session_state["cmp_multi_tickers"] = picks
        except Exception:
            pass
    auto_pair = [t for t in ("MINDACORP.NS", "ASKAUTOLTD.NS") if t in available_tickers]
    if len(auto_pair) == 2 and p2.button("🚗 MINDACORP vs ASKAUTOLTD", key="cmp_preset_auto", use_container_width=True):
        st.session_state["compare_tickers"] = auto_pair
        st.session_state["cmp_multi_tickers"] = auto_pair

    seed = stock_compare.normalize_tickers(
        st.session_state.get("compare_tickers") or auto_pair or available_tickers[:2],
        available=available_tickers,
        max_items=4,
    )
    if len(seed) < 2:
        seed = available_tickers[:2]
    if "cmp_multi_tickers" in st.session_state:
        cleaned = stock_compare.normalize_tickers(st.session_state["cmp_multi_tickers"], available=available_tickers, max_items=4)
        st.session_state["cmp_multi_tickers"] = cleaned if len(cleaned) >= 2 else seed
    else:
        st.session_state["cmp_multi_tickers"] = seed

    selected = st.multiselect(
        "📊 Select 2 to 4 stocks to compare head-to-head",
        options=available_tickers,
        max_selections=4,
        format_func=lambda x: x.replace(".NS", ""),
        key="cmp_multi_tickers",
    )
    st.session_state["compare_tickers"] = selected

    if len(selected) < 2:
        st.info("Select at least 2 stocks above to build the side-by-side Decision Matrix.")
        return

    board = get_board(df)
    val = funnel_view.valuation()
    res = stock_compare.build_comparison(selected, df, board=board, val=val)
    matrix = res["matrix"]
    sw = res.get("swing_winner", {})
    gw = res.get("garp_winner", {})

    c_sw, c_gw = st.columns(2)
    if sw.get("bare"):
        c_sw.info(f"**🏆 3–6m Swing / Momentum Advantage: `{sw['bare']}`**  \n{sw.get('why', '')}")
    if gw.get("bare"):
        c_gw.success(f"**🛡️ 2–3y Core / GARP Advantage: `{gw['bare']}`**  \n{gw.get('why', '')}")

    st.markdown("### 🧭 Unified 5-Pillar Decision Matrix")
    st.dataframe(matrix, use_container_width=True, hide_index=True, height=470)

    act_c1, act_c2, act_c3 = st.columns([2.2, 1.2, 1.6])
    rpt_t = act_c1.selectbox(
        "Open full Stock Report for",
        options=res["tickers"],
        format_func=lambda x: x.replace(".NS", ""),
        key="cmp_open_rpt_sel",
        label_visibility="collapsed",
    )
    if act_c2.button("📋 Open Stock Report", key="cmp_open_rpt_btn", use_container_width=True):
        st.session_state["report_ticker"] = rpt_t
        st.session_state["nav_page"] = "📋 Stock Report"
        st.rerun()
    act_c3.download_button(
        "⬇️ Download Matrix CSV",
        data=matrix.to_csv(index=False).encode("utf-8"),
        file_name=f"comparison_{'_vs_'.join(res['bares'])}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # 6-Month Relative Price Performance (Base 100) using local-first safe_history
    st.markdown("---")
    st.markdown("### 📈 Price Performance (Last 6 Months · Base 100)")
    palette = ["#00d4ff", "#ff6b6b", "#10b981", "#f59e0b"]
    fig2 = go.Figure()
    perfs: list[tuple[str, float]] = []
    for idx, (t, bare) in enumerate(zip(res["tickers"], res["bares"])):
        hist = safe_history(t, period="6mo", timeout=10)
        if hist is None or hist.empty or "Close" not in hist.columns:
            continue
        s_close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
        if len(s_close) < 2 or float(s_close.iloc[0]) <= 0:
            continue
        norm = (s_close / float(s_close.iloc[0])) * 100.0
        ret_6m = float((s_close.iloc[-1] / s_close.iloc[0] - 1.0) * 100.0)
        perfs.append((bare, ret_6m))
        fig2.add_trace(
            go.Scatter(
                x=norm.index,
                y=norm.values,
                name=f"{bare} ({ret_6m:+.1f}%)",
                line=dict(color=palette[idx % len(palette)], width=2.2),
            )
        )
    if perfs:
        fig2.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="Base 100")
        fig2.update_layout(
            height=360,
            template="plotly_white",
            yaxis_title="Performance (Base 100)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig2, use_container_width=True)
        p_cols = st.columns(len(perfs) + 1)
        for i, (bare, ret_6m) in enumerate(perfs):
            p_cols[i].metric(f"{bare} (6M)", f"{ret_6m:+.1f}%")
        best_6m = max(perfs, key=lambda x: x[1])
        p_cols[-1].metric("🏆 6M Price Leader", best_6m[0], f"{best_6m[1]:+.1f}%")

    # Collapsible Legacy 0-10 Pillar Radar
    with st.expander("📊 Legacy 0–10 Pillar Radar & Scores", expanded=False):
        categories = ["Trend", "Quality", "Value", "Growth", "Momentum", "Volume"]
        fig = go.Figure()
        for idx, (t, bare) in enumerate(zip(res["tickers"], res["bares"])):
            sub_row = df[df["ticker"] == t]
            if sub_row.empty:
                continue
            r0 = sub_row.iloc[0]
            vals = [
                float(r0.get("trend_score", 0) or 0) / 10.0,
                float(r0.get("quality", 0) or 0),
                float(r0.get("value", 0) or 0),
                float(r0.get("growth", 0) or 0),
                float(r0.get("momentum", 0) or 0),
                float(r0.get("volume_signal_score", 0) or 0),
            ]
            fig.add_trace(
                go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=bare,
                    line_color=palette[idx % len(palette)],
                )
            )
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=True,
            template="plotly_white",
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_deep_dive() -> None:

    st.sidebar.markdown("---")

    # Helper for safe number formatting to prevent "cannot convert float NaN to integer"
    def safe_int(val, default=0):
        try:
            if pd.isna(val) or val is None or val == "NaN":
                return default
            return int(float(val))
        except:
            return default

    def safe_float(val, default=0.0):
        try:
            if pd.isna(val) or val is None or val == "NaN":
                return default
            return float(val)
        except:
            return default

    # Check if coming from Trend Scanner
    default_ticker = st.session_state.get('quick_dive_ticker', TICKERS[0])
    try:
        default_index = TICKERS.index(default_ticker)
    except ValueError:
        default_index = 0

    # Selection Logic
    selection_mode = st.sidebar.radio("Selection Mode", ["List Selection", "Custom Ticker"], horizontal=True, label_visibility="collapsed")

    target_ticker = None
    is_custom = False

    if selection_mode == "Custom Ticker":
        custom_input = st.sidebar.text_input("Enter Ticker Symbol (e.g. RELIANCE)")
        if custom_input:
            target_ticker = custom_input.upper()
            if not target_ticker.endswith(".NS") and not target_ticker.endswith(".BO"):
                target_ticker += ".NS"
            is_custom = True
    else:
        target_ticker = st.sidebar.selectbox("Select Company", TICKERS, index=default_index)
        is_custom = False

    st.sidebar.markdown("---")
    gemini_api_key = st.sidebar.text_input("Gemini API Key (For AI Reports)", type="password")
    if not gemini_api_key:
        st.sidebar.caption("🔑 Enter API key to unlock the 🧠 Generate AI Report feature.")

    # Main Content
    if target_ticker:
        try:
            info = None
            scores = None
        
            # =========================================================
            # PATH A: CACHED NIFTY 500 STOCK (Single Source of Truth)
            # =========================================================
            if not is_custom and 'market_data' in st.session_state:
                df_cache = st.session_state['market_data']
                # Fast lookup
                row = df_cache[df_cache['ticker'] == target_ticker]
            
                if not row.empty:
                    # Use the EXACT data record from memory
                    info = row.iloc[0].to_dict()
                
                    # Ensure numeric types for scores to prevent UI crashes
                    scores = {
                        'quality': safe_float(info.get('quality'), 5.0),
                        'value': safe_float(info.get('value'), 5.0),
                        'growth': safe_float(info.get('growth'), 5.0),
                        'momentum': safe_float(info.get('momentum'), 5.0),
                        'overall': safe_float(info.get('overall'), 5.0),
                        'sector_profile': info.get('sector_profile', 'DEFAULT')
                    }
                
                    # Ensure trend metrics exist from cache (or recalculate safely)
                    if 'trend_score' not in info or pd.isna(info.get('trend_score')):
                         trend_res = calculate_trend_metrics(info)
                         info.update(trend_res)

            # =========================================================
            # PATH B: CUSTOM TICKER OR CACHE MISS (Live Fetch)
            # =========================================================
            if info is None:
                with st.spinner(f"Fetching data for {target_ticker}..."):
                    info = get_stock_info(target_ticker)
                
                    if info:
                        # 1. Trend Metrics
                        info.update(calculate_trend_metrics(info))
                    
                        # 2. Score Calculation
                        sector_pe_cache = st.session_state.get('sector_pe_cache', {})
                        sector = info.get('sector', 'Unknown')
                        sector_pe = sector_pe_cache.get(sector, 25.0) # Default PE 25 if unknown
                    
                        scores_raw = calculate_scores(info, sector_pe_median=sector_pe)
                        scores = {k: safe_float(v) if isinstance(v, (int, float)) else v for k,v in scores_raw.items()}

            # =========================================================
            # DISPLAY LOGIC
            # =========================================================
            if not info:
                st.error(f"Could not load data for {target_ticker}. Please check the ticker symbol.")
            else:
                # History is needed for charts - this is always live/cached separately
                hist = get_stock_history(target_ticker)
            
                # News
                news_items = []
                try:
                    news_items = fetch_latest_news(info.get('name', target_ticker))
                except: 
                    pass
            
                view_mode = st.radio("Display Mode", ["Research Report", "Interactive Dashboard"], horizontal=True, key="dd_view_mode")

                # --- HEADER METRICS ---
                st.markdown(f"## {info.get('name', target_ticker)}")
            
                h_col1, h_col2, h_col3, h_col4 = st.columns(4)
            
                price = safe_float(info.get('currentPrice') or info.get('price'))
                chg_52w = safe_float(info.get('52WeekChange')) * 100
                mcap = safe_float(info.get('marketCap')) / 10000000
                pe_ratio = safe_float(info.get('pe'))
            
                h_col1.metric("Price", f"₹{price:,.2f}")
                h_col2.metric("1Y Return", f"{chg_52w:+.1f}%")
                h_col3.metric("Market Cap", f"₹{mcap:,.0f} Cr")
                h_col4.metric("P/E Ratio", f"{pe_ratio:.1f}x")
            
                st.markdown("---")

                if view_mode == "Research Report":
                    from utils.ai_report_engine import build_data_payload, generate_ai_report_markdown, convert_markdown_to_pdf

                
                    rep_col1, rep_col2 = st.columns([3, 1])
                    with rep_col1:
                        st.markdown("### 📊 Standard Report")
                    with rep_col2:
                        # Clear AI session state if ticker changes
                        if st.session_state.get('ai_current_ticker') != target_ticker:
                            st.session_state['ai_report_md'] = None
                            st.session_state['ai_report_pdf'] = None
                            st.session_state['ai_current_ticker'] = target_ticker
                        
                        if st.button("🧠 Generate AI PDF Report", type="primary", use_container_width=True):
                            if 'gemini_api_key' not in locals() or not gemini_api_key:
                                st.error("Please enter a Gemini API Key in the sidebar.")
                            else:
                                with st.spinner("🤖 AI Analyst is writing your report... This takes ~15 seconds."):
                                    try:
                                        payload = build_data_payload(target_ticker, info, scores, news_items, hist)
                                        md_report = generate_ai_report_markdown(gemini_api_key, payload)
                                    
                                        # Save PDF
                                        os.makedirs("analysis_2026/reports", exist_ok=True)
                                        pdf_path = f"analysis_2026/reports/{target_ticker.replace('.NS', '')}_AI_Report.pdf"
                                        convert_markdown_to_pdf(md_report, pdf_path)
                                    
                                        st.session_state['ai_report_md'] = md_report
                                        st.session_state['ai_report_pdf'] = pdf_path
                                        st.success("Report generated successfully!")
                                    except Exception as e:
                                        st.error(f"Generation failed: {e}")

                    if st.session_state.get('ai_report_md') and st.session_state.get('ai_report_pdf'):
                        st.markdown("---")
                        st.markdown("### 🤖 Institutional AI Research")
                    
                        pdf_file_path = st.session_state['ai_report_pdf']
                        if os.path.exists(pdf_file_path):
                            with open(pdf_file_path, "rb") as pdf_file:
                                st.download_button(
                                    label="📥 Download PDF Report",
                                    data=pdf_file,
                                    file_name=f"{target_ticker.replace('.NS', '')}_Research_Report.pdf",
                                    mime="application/pdf",
                                    type="primary"
                                )
                        with st.expander("👁️ Preview AI Report (Markdown)", expanded=True):
                            st.markdown(st.session_state['ai_report_md'])
                        st.markdown("---")
                
                    if scores:
                        from utils.report_generator import generate_equity_report, generate_pdf_from_md
                        report_content = generate_equity_report(target_ticker, info, scores, news_items, hist)
                        report_pdf = generate_pdf_from_md(report_content)
                    
                        st.download_button(
                            label="📥 Download PDF Report (For LLM Analysis)",
                            data=report_pdf,
                            file_name=f"{target_ticker}_Research_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            type="primary"
                        )
                        st.markdown(report_content)
                
                    if hist is not None and not hist.empty:
                        st.markdown("### Price Trend")
                        st.plotly_chart(chart_price_history(hist), use_container_width=True)
            
                else:
                    # --- DASHBOARD VIEW ---
                    st.subheader("📊 Investment Scorecard")
                
                    if scores:
                        col_radar, col_gauge, col_details = st.columns([1.5, 1, 1.5])

                        with col_radar:
                            # Assuming sector_history is available for chart_sector_rotation
                            # This line was part of the provided snippet, but its context (sector_history) is missing.
                            # If this is intended for Deep Dive, ensure 'sector_history' is defined.
                            # For now, commenting it out or assuming it's a placeholder.
                            # st.plotly_chart(chart_sector_rotation(sector_history), use_container_width=True)
                            st.plotly_chart(chart_score_radar(scores), use_container_width=True) # Keeping original radar chart
                    
                        with col_gauge:
                            overall = scores.get('overall', 5.0)
                            st.plotly_chart(chart_gauge(overall), use_container_width=True)
                        
                            rec = "BUY" if overall >= 7.5 else "AVOID" if overall < 5.0 else "HOLD"
                            rec_color = "green" if rec == "BUY" else "red" if rec == "AVOID" else "orange"
                            st.markdown(f"<h3 style='text-align: center; color: {rec_color}; margin-top: -20px;'>{rec}</h3>", unsafe_allow_html=True)
                    
                        with col_details:
                            st.markdown("#### Pillar Breakdown")
                            def score_bar(label, val, icon):
                                val = safe_float(val, 5.0)
                                st.progress(val/10, text=f"{icon} {label}: **{val:.1f}**/10")
                            
                            score_bar("Quality", scores.get('quality'), "💎")
                            score_bar("Value", scores.get('value'), "💰")
                            score_bar("Growth", scores.get('growth'), "📈")
                            score_bar("Momentum", scores.get('momentum'), "🚀")
                
                    # --- FINANCIALS & CHARTS ---
                    st.markdown("---")
                    t1, t2 = st.tabs(["📈 Charts & Technicals", "📋 Key Financials"])
                
                    with t1:
                        if hist is not None and not hist.empty:
                            st.plotly_chart(chart_price_history(hist), use_container_width=True, key="dd_price_chart")
                        else:
                            st.info("Price history unavailable.")
                        
                    with t2:
                        f1, f2, f3 = st.columns(3)
                        f1.metric("ROE", f"{safe_float(info.get('roe'))*100:.1f}%")
                        f2.metric("Profit Margin", f"{safe_float(info.get('profitMargins'))*100:.1f}%")
                        f3.metric("Debt/Equity", f"{safe_float(info.get('debtToEquity')):.2f}")
                    
                        f4, f5, f6 = st.columns(3)
                        f4.metric("PEG Ratio", f"{safe_float(info.get('pegRatio')):.2f}")
                        f5.metric("P/B Ratio", f"{safe_float(info.get('pb')):.2f}")
                        f6.metric("Div Yield", f"{safe_float(info.get('dividendYield'))*100:.2f}%")
                
                    # --- SCORE HISTORY TAB ---
                    st.markdown("---")
                    st.subheader("📈 Score History (Backtesting)")
                    st.caption("See how this stock's scores have evolved over the past 4 quarters.")
                
                    from utils.backtesting import calculate_historical_scores, get_score_trend_insight
                    from utils.visuals import chart_score_history
                
                    with st.spinner("Calculating historical scores..."):
                        hist_scores = calculate_historical_scores(target_ticker, lookback_quarters=4)
                
                    if not hist_scores.empty:
                        # Insight
                        insight = get_score_trend_insight(hist_scores, scores)
                        st.info(insight)
                    
                        # Chart
                        fig_hist = chart_score_history(hist_scores, current_scores=scores)
                        if fig_hist:
                            st.plotly_chart(fig_hist, use_container_width=True)
                    
                        # Data table
                        with st.expander("📊 Raw Historical Scores"):
                            st.dataframe(hist_scores, hide_index=True)
                    else:
                        st.warning("Could not calculate historical scores. Quarterly financial data may be unavailable.")

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.code(f"Error details: {e}")


def render_notes() -> None:
    st.title("📝 Trading Journal")
    st.markdown("Use this space to log your daily observations and strategy ideas.")

    if 'trading_notes' not in st.session_state:
        st.session_state['trading_notes'] = ""
    
    notes = st.text_area("Daily Log", value=st.session_state['trading_notes'], height=400, placeholder="- Observed strong volume in IT sector...\n- Planning to enter RELIANCE above 2500...")

    if st.button("💾 Save Journal"):
        st.session_state['trading_notes'] = notes
        st.success("Notes saved successfully! (Session storage)")


def render_ai_capex(*, nifty_recent) -> None:
    st.markdown(page_header(
        "🖥️ AI Capex Play",
        "US AI datacenter buildout — Indian power infra, cables, equipment & software beneficiaries"
    ), unsafe_allow_html=True)

    # ── Universe by sub-theme ─────────────────────────────────────────────────
    AI_CAPEX_UNIVERSE = {
        "⚡ Power Infra":           ["ADANIGREEN.NS", "ADANIENSOL.NS", "NTPC.NS", "POWERGRID.NS",
                                     "TATAPOWER.NS", "TORNTPOWER.NS", "CESC.NS"],
        "🔌 Cables & Connectivity": ["POLYCAB.NS", "KEI.NS", "APARINDS.NS", "STLTECH.NS", "KECL.NS"],
        "⚙️ Electrical Equipment":  ["ABB.NS", "SIEMENS.NS", "HAVELLS.NS", "THERMAX.NS", "BHEL.NS"],
        "❄️ Cooling & HVAC":        ["VOLTAS.NS", "BLUESTAR.NS"],
        "🔧 Electronics Mfg (EMS)": ["DIXON.NS", "KAYNES.NS", "SYRMA.NS"],
        "💻 IT / AI Services":      ["PERSISTENT.NS", "COFORGE.NS", "LTTS.NS", "LTIM.NS",
                                     "MPHASIS.NS", "TATAELXSI.NS", "KPITTECH.NS"],
        "🔋 Storage & Power Elec":  ["AMARARAJA.NS", "EXIDEIND.NS"],
    }
    ALL_AI_TICKERS   = [t for grp in AI_CAPEX_UNIVERSE.values() for t in grp]
    TICKER_THEME_MAP = {t: theme for theme, tickers in AI_CAPEX_UNIVERSE.items() for t in tickers}

    # ── RS weight config ──────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("**⚖️ AI Capex RS Weights**")
    st.sidebar.caption("RS63 stays below RS5 & RS21 to front-load theme momentum")
    w5  = st.sidebar.slider("RS5  (1-week)",  0.10, 0.50, 0.30, 0.05, key="ai_w5")
    w21 = st.sidebar.slider("RS21 (1-month)", 0.20, 0.60, 0.50, 0.05, key="ai_w21")
    raw_w63 = round(1.0 - w5 - w21, 2)
    w63 = max(0.05, raw_w63)
    total_w = w5 + w21 + w63
    w5_n, w21_n, w63_n = w5 / total_w, w21 / total_w, w63 / total_w
    st.sidebar.caption(f"RS63 auto-set to **{w63_n:.0%}**")
    if w63_n >= w5_n:
        st.sidebar.warning("⚠️ RS63 ≥ RS5 — slide RS5 up to prioritise recent momentum")

    # ── Nifty period returns (cached 1 h) ─────────────────────────────────────
    n_1w_ref, n_1m_ref, n_3m_ref = _nifty_rs_refs(nifty_recent)

    # ── Load & filter market data ─────────────────────────────────────────────
    md = st.session_state.get('market_data', pd.DataFrame())
    if md is None or (isinstance(md, pd.DataFrame) and md.empty):
        st.warning("Market data not yet loaded. Visit the Trend Scanner first to trigger a data refresh.")
        st.stop()

    ai_df = md[md['ticker'].isin(ALL_AI_TICKERS)].copy()
    if ai_df.empty:
        st.warning("No AI Capex stocks found in current market data. Ensure Nifty 1000 universe is loaded.")
        st.stop()

    # ── Compute AI Comp RS with custom weights ────────────────────────────────
    # Prefer stored rs_1w / rs_1m from data engine (available after cache refresh);
    # fall back to stock absolute return minus Nifty period return.
    if 'rs_1w' in ai_df.columns:
        ai_df['_rs_1w'] = ai_df['rs_1w'].fillna(ai_df['return_1w'].fillna(0) - n_1w_ref)
    else:
        ai_df['_rs_1w'] = ai_df['return_1w'].fillna(0) - n_1w_ref

    if 'rs_1m' in ai_df.columns:
        ai_df['_rs_1m'] = ai_df['rs_1m'].fillna(ai_df['return_1m'].fillna(0) - n_1m_ref)
    else:
        ai_df['_rs_1m'] = ai_df['return_1m'].fillna(0) - n_1m_ref

    if 'rs_3m' in ai_df.columns:
        ai_df['_rs_3m'] = ai_df['rs_3m'].fillna(ai_df['return_3m'].fillna(0) - n_3m_ref)
    else:
        ai_df['_rs_3m'] = ai_df['return_3m'].fillna(0) - n_3m_ref

    ai_df['ai_comp_rs'] = (
        ai_df['_rs_1w'] * w5_n +
        ai_df['_rs_1m'] * w21_n +
        ai_df['_rs_3m'] * w63_n
    ).round(2)
    ai_df['sub_theme'] = ai_df['ticker'].map(TICKER_THEME_MAP)

    # ── Theme-level Pulse metrics ─────────────────────────────────────────────
    n_stocks  = len(ai_df)
    avg_rs    = round(float(ai_df['ai_comp_rs'].mean()), 2)
    pct_above = 0.0
    if 'fiftyDayAverage' in ai_df.columns:
        pct_above = float((ai_df['price'] > ai_df['fiftyDayAverage']).mean() * 100)
    buy_watch = 0
    if 'dna_signal' in ai_df.columns:
        buy_watch = int(ai_df['dna_signal'].isin(['BUY', 'WATCH']).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Theme Avg RS vs Nifty", f"{avg_rs:+.2f}%",
        help=f"Average AI Comp RS — weights RS5={w5_n:.0%} / RS21={w21_n:.0%} / RS63={w63_n:.0%}"
    )
    c2.metric("Above MA50", f"{pct_above:.0f}%",
              help="% of AI Capex stocks trading above their 50-day MA")
    c3.metric("BUY / WATCH Signals", f"{buy_watch} / {n_stocks}",
              help="Stocks meeting BUY or WATCH criteria (AI Comp RS > 0 + above MA50)")
    c4.metric("Universe Coverage", f"{n_stocks} / {len(ALL_AI_TICKERS)}",
              help="Stocks from the AI Capex basket found in current market data")

    theme_color = COLORS['positive'] if avg_rs > 0 else COLORS['negative']
    theme_msg   = ("Theme outperforming Nifty — momentum is on." if avg_rs > 0
                   else "Theme underperforming Nifty — wait for RS turn before adding.")
    st.markdown(
        f'<div style="background:{theme_color}18;border-left:4px solid {theme_color};'
        f'padding:10px 16px;border-radius:6px;margin:4px 0 16px 0;'
        f'color:{theme_color};font-weight:500">'
        f'{"🟢" if avg_rs > 0 else "🔴"} {theme_msg}</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ── Sub-theme Momentum — OptComp theme-rotation detector ─────────────────
    st.markdown("### Sub-theme Momentum")
    st.caption(
        "Which cluster is leading? Identifies intra-theme rotation via AI Comp RS. "
        "Sorted best → worst so you can spot which sub-theme is currently 'in play'."
    )

    _agg: dict = dict(
        avg_rs    = ('ai_comp_rs', 'mean'),
        avg_rs_1w = ('_rs_1w',     'mean'),
        avg_rs_1m = ('_rs_1m',     'mean'),
        avg_rs_3m = ('_rs_3m',     'mean'),
        n         = ('ticker',     'count'),
    )
    if 'dna_signal' in ai_df.columns:
        _agg['pct_sig'] = ('dna_signal', lambda x: x.isin(['BUY', 'WATCH']).mean() * 100)

    theme_stats = (
        ai_df.groupby('sub_theme')
             .agg(**_agg)
             .reset_index()
             .sort_values('avg_rs', ascending=False)
    )
    if 'pct_sig' not in theme_stats.columns:
        theme_stats['pct_sig'] = 0.0

    max_abs = max(float(theme_stats['avg_rs'].abs().max()), 0.01)

    for _, row in theme_stats.iterrows():
        rs_val   = float(row['avg_rs'])
        bar_pct  = abs(rs_val) / max_abs * 160   # px width, max 160px
        color    = COLORS['positive'] if rs_val >= 0 else COLORS['negative']
        arrow    = "▲" if rs_val >= 0 else "▼"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin:5px 0;font-size:13px">'
            f'<span style="min-width:210px;color:{COLORS["text_primary"]}">{row["sub_theme"]}</span>'
            f'<div style="width:{bar_pct:.0f}px;min-width:4px;height:14px;'
            f'background:{color}55;border-radius:3px;border-right:2px solid {color}"></div>'
            f'<span style="color:{color};font-weight:700;min-width:72px">{arrow} {abs(rs_val):.2f}%</span>'
            f'<span style="color:{COLORS["text_muted"]};font-size:11px">'
            f'RS5:{row["avg_rs_1w"]:+.1f} &nbsp;RS21:{row["avg_rs_1m"]:+.1f} &nbsp;RS63:{row["avg_rs_3m"]:+.1f}'
            f'&nbsp;|&nbsp;{row["pct_sig"]:.0f}% signals &nbsp;({int(row["n"])} stocks)'
            f'</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Stock Screener ────────────────────────────────────────────────────────
    st.markdown("### Stock Screener")

    fc1, fc2, fc3 = st.columns([2, 1, 1])
    theme_filter = fc1.multiselect(
        "Sub-Theme", list(AI_CAPEX_UNIVERSE.keys()),
        default=list(AI_CAPEX_UNIVERSE.keys()), key="ai_theme_filter"
    )
    if 'dna_signal' in ai_df.columns:
        signal_filter = fc2.multiselect(
            "Signal", ["BUY", "WATCH", "HOLD"],
            default=["BUY", "WATCH", "HOLD"], key="ai_sig_filter"
        )
    else:
        signal_filter = ["BUY", "WATCH", "HOLD"]
        fc2.caption("Signal filter n/a")
    min_rs_val = fc3.slider("Min AI Comp RS", -15.0, 10.0, -15.0, 0.5, key="ai_min_rs")

    fdf = ai_df[ai_df['sub_theme'].isin(theme_filter)].copy()
    if 'dna_signal' in fdf.columns:
        fdf = fdf[fdf['dna_signal'].isin(signal_filter)]
    fdf = fdf[fdf['ai_comp_rs'] >= min_rs_val].sort_values('ai_comp_rs', ascending=False)

    disp = fdf.copy()
    disp['screener_link'] = (
        "https://www.screener.in/company/"
        + disp['ticker'].str.replace(r'\.(NS|BO)$', '', regex=True)
        + "/"
    )

    _col_map = [
        ('screener_link', 'Stock'),
        ('name',          'Name'),
        ('sub_theme',     'Sub-Theme'),
        ('price',         'CMP'),
        ('ai_comp_rs',    'AI Comp RS'),
        ('_rs_1w',        'RS5 (1W)'),
        ('_rs_1m',        'RS21 (1M)'),
        ('_rs_3m',        'RS63 (3M)'),
        ('comp_rs',       'Std RS (V22)'),
        ('dna_signal',    'Signal'),
        ('trend_score',   'Trend'),
        ('dist_52w',      '52W Off%'),
    ]
    avail_cols = [(src, dst) for src, dst in _col_map if src in disp.columns]
    disp_show  = disp[[src for src, _ in avail_cols]].rename(columns={src: dst for src, dst in avail_cols})

    _col_cfg = {
        'Stock':        st.column_config.LinkColumn("Stock", display_text="🔗 Screener"),
        'CMP':          st.column_config.NumberColumn("CMP ₹", format="₹%.1f"),
        'AI Comp RS':   st.column_config.NumberColumn(
                            f"AI RS  ({w5_n:.0%}/{w21_n:.0%}/{w63_n:.0%})",
                            format="%+.2f%%",
                            help="Composite RS with AI-Capex weights vs standard OptComp-V22"
                        ),
        'RS5 (1W)':     st.column_config.NumberColumn("RS5",  format="%+.1f%%"),
        'RS21 (1M)':    st.column_config.NumberColumn("RS21", format="%+.1f%%"),
        'RS63 (3M)':    st.column_config.NumberColumn("RS63", format="%+.1f%%"),
        'Std RS (V22)': st.column_config.NumberColumn("Std RS", format="%+.2f%%",
                            help="Standard OptComp-V22: RS5=10% RS21=50% RS63=40%"),
        'Trend':        st.column_config.ProgressColumn("Trend", min_value=0, max_value=100, format="%d"),
        '52W Off%':     st.column_config.NumberColumn("52W Off%", format="%.1f%%"),
    }

    st.dataframe(
        disp_show.reset_index(drop=True),
        column_config={k: v for k, v in _col_cfg.items() if k in disp_show.columns},
        use_container_width=True,
        hide_index=True,
        height=min(len(disp_show) * 38 + 42, 640),
    )

    # ── Why different weights? ────────────────────────────────────────────────
    with st.expander("ℹ️ Why different RS weights for AI Capex?"):
        st.markdown(f"""
    **Standard OptComp-V22:** RS5 = 10%, RS21 = 50%, RS63 = 40%

    **AI Capex (current):** RS5 = {w5_n:.0%}, RS21 = {w21_n:.0%}, RS63 = {w63_n:.0%}

    AI Datacenter is a **fast-rotating, news-driven theme**. The 3-month (RS63) lookback often
    includes time *before* the theme ignited, diluting the breakout signal. Elevating RS5
    catches fresh breakouts earlier; RS21 stays dominant as the medium-term confirmation window.

    **Sub-theme rotation:** Within AI Capex, leadership rotates — Power leads one week,
    Cables the next, then IT Services. The Sub-theme Momentum panel uses AI Comp RS per cluster
    to show which sub-theme is currently "in play", so you can size into the right pocket.

    Use the sidebar sliders to tune weights for current market conditions.
        """)
