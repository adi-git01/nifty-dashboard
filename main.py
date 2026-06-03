import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import time
import subprocess
from datetime import datetime, timedelta
import sys
import io

# Force UTF-8 encoding for standard output/error to prevent cp1252 crashes on Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.nifty500_list import TICKERS
from utils.data_engine import get_stock_info, get_stock_history, batch_fetch_tickers
from utils.scoring import calculate_scores, calculate_trend_metrics
from utils.visuals import chart_score_radar, chart_price_history, chart_gauge, chart_market_heatmap, chart_relative_performance
from utils.news_engine import fetch_latest_news
from utils.report_generator import generate_equity_report
from utils.ui_components import css_styles, card_metric, card_verdict, page_header, COLORS, hero_pnl_card, sidebar_alert_panel
from utils.analytics_engine import analyze_sectors, calculate_cycle_position, get_monthly_alpha_calendar
from utils.market_explorer import render_market_explorer
from utils.positions import (
    get_all_positions, get_positions_with_pnl, get_position,
    add_position, update_position, remove_position, close_position,
    check_position_alerts, get_summary, migrate_from_legacy, is_position_exists,
    add_to_watchlist
)
from utils.email_notifier import is_email_configured, configure_email, send_weekly_summary, send_trend_change_alert, test_email_connection, get_email_address
from utils.return_tracker import export_weekly_summary
from utils.telegram_notifier import send_telegram_message, is_telegram_configured
from utils.trend_engine import calculate_sector_history, calculate_stock_trend_history
from utils.market_mood import calculate_mood_metrics, save_mood_snapshot, load_mood_history, chart_market_mood
from utils.market_breadth import render_breadth_widget
from utils.fast_data_engine import load_base_fundamentals, fetch_and_process_market_data
from utils.live_desk import get_cyclicity, get_seasonal_guideline
from utils.theme_engine import ThemeEngine, AI_CAPEX_THEME
from utils.us_data_engine import fetch_us_market_data, load_sp500_universe
from utils.us_rotation_tracker import (
    save_us_rotation_snapshot, load_us_rotation_history,
    build_rotation_pivot, render_us_rotation_table, render_us_rotation_heatmap,
    backfill_us_rotation_if_needed,
)
from utils.advanced_scanners import (
    find_vcp_setups, find_rs_divergence, find_live_earnings_shocks,
    save_rs_divergence_signals, save_earnings_shock_signals,
    refresh_signal_log_prices, load_signal_log,
    RS_LOG_FILE, RS_LOG_COLS, SHOCK_LOG_FILE, SHOCK_LOG_COLS,
    find_turnaround_catalysts, save_turnaround_catalyst_signals,
    TC_LOG_FILE, US_TC_LOG_FILE, TC_LOG_COLS,
)

import re as _re

def _google_finance_url(ticker: str) -> str:
    """Google Finance URL for a US stock ticker."""
    return f"https://www.google.com/finance/quote/{ticker}"

_GF_DISPLAY_RE = r"https://www\.google\.com/finance/quote/(.+)$"

# Debug mode: set DASH_DEBUG=1 to show debug panel
DASH_DEBUG = os.environ.get('DASH_DEBUG', '0') == '1'

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Nifty 500 Research Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PREMIUM DASHBOARD CSS ---
st.markdown(css_styles(), unsafe_allow_html=True)

# --- SIDEBAR & NAV ---
st.sidebar.title("🚀 Alpha Trend")

if st.sidebar.button("🔄 Hard Reset Cache", type="primary"):
    if os.path.exists("nifty500_cache.csv"):
        os.remove("nifty500_cache.csv")
    if os.path.exists("market_data.parquet"):
        os.remove("market_data.parquet")
    # Clear ALL caches including @st.cache_data decorated functions
    st.cache_data.clear()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

live_mode_toggle = st.sidebar.checkbox("🚀 Bypass Master Cache (Live Mode)", value=False, help="Forces a fresh pull of all 1000 stocks from Yahoo Finance instead of using the daily Parquet cache.")

st.sidebar.markdown("---")

# === SCREENING PRESETS ===
st.sidebar.markdown("### 🎯 Quick Presets")
preset = st.sidebar.selectbox(
    "Strategy Filter",
    [
        "All Stocks",
        "🚀 Strong Momentum (Top 20%)",
        "💎 Quality at Reasonable Price",
        "📈 Breakout Candidates",
        "🔥 Turnaround Plays",
        "🧬 DNA-3 V2 Picks"
    ]
)
st.session_state['preset'] = preset

# === WATCHLIST (Persistent via positions.json) ===
st.sidebar.markdown("---")
st.sidebar.markdown("### ⭐ Watchlist")
watchlist_positions = get_all_positions('watching')
watchlist_count = len(watchlist_positions)
st.sidebar.write(f"Watching: {watchlist_count} stocks")

if watchlist_count > 0:
    with st.sidebar.expander("View Watchlist"):
        for wp in watchlist_positions:
            ticker = wp.get('ticker', '')
            display_name = ticker.replace('.NS', '').replace('.BO', '')
            wl_c1, wl_c2 = st.columns([3, 1])
            wl_c1.write(f"📌 {display_name}")
            if wl_c2.button("❌", key=f"wl_rm_{ticker}"):
                remove_position(ticker)
                st.rerun()

# === POSITIONS & ALERTS (Unified) ===
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Positions")
active_sidebar = get_all_positions('active')
active_count = len(active_sidebar)

if active_count > 0:
    st.sidebar.caption(f"Active: {active_count} positions")
    for pos in active_sidebar[:5]:
        ticker = pos.get('ticker', '').replace('.NS', '').replace('.BO', '')
        sl = f"SL:{pos['stop_loss']:.0f}" if pos.get('stop_loss') else ""
        tgt = f"T:{pos['target']:.0f}" if pos.get('target') else ""
        st.sidebar.caption(f"• {ticker} {sl} {tgt}")
    if active_count > 5:
        st.sidebar.caption(f"  +{active_count - 5} more...")
else:
    st.sidebar.caption("No active positions")

# Link to full manager
if st.sidebar.button("📋 Open Position Manager", use_container_width=True, type="secondary"):
    st.session_state['nav_page'] = "📊 Return Tracker"
    st.rerun()

# Persistent Alert Panel in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔔 Alerts")

# ============================================================
# CENTRALIZED DATA LOADING + AUTO-REFRESH
# ============================================================
AUTO_REFRESH_MINUTES = 30  # Auto-refresh data after 30 minutes

# Check if data needs refresh (stale data)
_needs_refresh = 'market_data' not in st.session_state
if not _needs_refresh and 'data_loaded_at' in st.session_state:
    _minutes_old = (datetime.now() - st.session_state['data_loaded_at']).total_seconds() / 60
    if _minutes_old > AUTO_REFRESH_MINUTES:
        _needs_refresh = True
        st.toast("🔄 Data is stale, refreshing...", icon="⏰")

if _needs_refresh:
    
    load_status = st.empty()
    progress_bar = st.progress(0, text="Initializing...")
    
    # Step 1: Load fundamentals
    progress_bar.progress(10, text="Step 1/4: Loading fundamentals...")
    fundamentals = load_base_fundamentals(live_mode=live_mode_toggle)
    
    # Step 2: Fetch live stock data
    progress_bar.progress(30, text=f"Step 2/4: Fetching live prices for {len(fundamentals)} stocks...")
    df = fetch_and_process_market_data(fundamentals['ticker'].tolist(), fundamentals, live_mode=live_mode_toggle)
    
    if df.empty:
        progress_bar.empty()
        st.error("⚠️ Data Fetch Failed! Check internet.")
        st.stop()
    
    # Step 3: Pre-fetch Nifty index data (shared across all pages)
    progress_bar.progress(80, text="Step 3/4: Fetching Nifty index data...")
    try:
        _nifty_start = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
        _nifty_data = yf.Ticker("^NSEI").history(start=_nifty_start)
        if not _nifty_data.empty:
            st.session_state['nifty_data'] = _nifty_data
            
            # --- V2.2 Regime Manager ---
            from utils.regime_manager import classify_regime
            _nd = _nifty_data.copy()
            if _nd.index.tz is not None:
                _nd.index = _nd.index.tz_localize(None)
            _ma200 = _nd['Close'].rolling(200).mean().iloc[-1]
            _h52w  = _nd['High'].rolling(252).max().iloc[-1]
            _close = _nd['Close'].iloc[-1]
            current_regime = classify_regime(float(_close), float(_ma200), float(_h52w))
            st.session_state['market_regime'] = current_regime
    except Exception:
        pass  # Non-fatal — individual pages can fallback
    
    # Step 4: Finalize
    progress_bar.progress(95, text="Step 4/4: Calculating scores...")
    st.session_state['market_data'] = df
    st.session_state['data_loaded_at'] = datetime.now()
    
    # Cache sector PE for scoring consistency
    if 'pe' in df.columns and 'sector' in df.columns:
         st.session_state['sector_pe_cache'] = df.groupby('sector')['pe'].median().to_dict()
    
    progress_bar.progress(100, text="✅ Ready!")
    time.sleep(0.5)
    progress_bar.empty()
    load_status.empty()


df = st.session_state['market_data']

# Data freshness indicator
if 'data_loaded_at' in st.session_state:
    loaded_at = st.session_state['data_loaded_at']
    minutes_ago = int((datetime.now() - loaded_at).total_seconds() / 60)
    if minutes_ago < 1:
        freshness_text = "🟢 Just now"
    elif minutes_ago < 60:
        freshness_text = f"🟢 {minutes_ago} min ago"
    elif minutes_ago < 240:
        freshness_text = f"🟡 {minutes_ago // 60}h ago"
    else:
        freshness_text = f"🔴 {minutes_ago // 60}h ago (stale)"
    st.sidebar.caption(f"📊 {len(df)} stocks • Updated: {freshness_text}")
    
    regime = st.session_state.get('market_regime', 'UNKNOWN')
    regime_color = {"BULL": "🟢", "CAUTION": "🟡", "BEAR": "🟠", "CRISIS": "🔴"}.get(regime, "⚪")
    st.sidebar.markdown(f"**Market Regime:** {regime_color} **{regime}**")
else:
    st.sidebar.success(f"Loaded {len(df)} Tickers")

# Check positions for SL/Target hits and notify via Telegram + Email
triggered_positions = check_position_alerts(df)
if triggered_positions and 'pos_alerts_notified' not in st.session_state:
    alert_messages = []
    for pos in triggered_positions:
        ticker_display = pos.get('ticker', '').replace('.NS', '').replace('.BO', '')
        alert_type = pos.get('alert_type', 'ALERT')
        current_price = pos.get('current_price', 0)
        
        if alert_type == 'STOP_LOSS':
            alert_messages.append(f"🔴 {ticker_display} SL HIT ₹{current_price:.0f}")
        else:
            alert_messages.append(f"🎯 {ticker_display} TARGET HIT ₹{current_price:.0f}")
        
        # Send Telegram
        try:
            if is_telegram_configured():
                entry = pos.get('entry_price', 0)
                pnl = pos.get('pnl_pct', 0)
                msg = (f"{'SL' if alert_type == 'STOP_LOSS' else 'TARGET'} HIT: {ticker_display}\n"
                       f"Price: ₹{current_price:.2f} | Entry: ₹{entry:.2f} | P&L: {pnl:+.1f}%\n"
                       f"Action Required: Review position")
                send_telegram_message(msg)
        except:
            pass
        
        # Send Email  
        try:
            if is_email_configured():
                send_trend_change_alert([{
                    'ticker': pos.get('ticker'),
                    'entry_trend_signal': 'ACTIVE',
                    'current_signal': alert_type.replace('_', ' '),
                    'return_pct': pos.get('pnl_pct', 0),
                    'days_tracked': 0
                }])
        except:
            pass
    
    # Single batched toast instead of one per alert
    if alert_messages:
        st.toast(f"🚨 {len(alert_messages)} Alert(s): {' | '.join(alert_messages)}", icon="🔔")
    
    st.session_state['pos_alerts_notified'] = True

# Render persistent alert panel in sidebar (regardless of notification state)
if triggered_positions:
    st.sidebar.markdown(sidebar_alert_panel(triggered_positions), unsafe_allow_html=True)
else:
    st.sidebar.caption("✅ No alerts")

# Debug: Show data verification (only if DASH_DEBUG=1)
if DASH_DEBUG:
    with st.sidebar.expander("🔍 Debug Data"):
        st.write(f"Columns: {len(df.columns)}")
        if 'trend_signal' in df.columns:
            st.write("Signals:", df['trend_signal'].value_counts().to_dict())
        else:
            st.error("trend_signal column MISSING!")
        if 'trend_score' in df.columns:
            st.write(f"Trend Score Range: {df['trend_score'].min()} - {df['trend_score'].max()}")
        else:
            st.error("trend_score column MISSING!")
        sample = df.iloc[0].dropna().to_dict()
        price_keys = [k for k in sample.keys() if any(x in k.lower() for x in ['price', 'average', 'high', 'low', '52', 'week'])]
        st.write(price_keys[:15])


# trend_engine imports moved to top

# (Auto-alert check is now handled by the unified position alert system above)

# === DNA3 MORNING BRIEF AUTO-TRIGGER (Once per day on dashboard load) ===
try:
    _json = json
    _config_path = "config.json"
    _config = {}
    if os.path.exists(_config_path):
        with open(_config_path, 'r') as _f:
            _config = _json.load(_f)
    
    dna3_alert_config = _config.get("dna3_morning_alert", {})
    dna3_alert_enabled = dna3_alert_config.get("enabled", False)
    last_sent_date = dna3_alert_config.get("last_sent_date", "")
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    if dna3_alert_enabled and last_sent_date != today_str and 'dna3_alert_sent' not in st.session_state:
        # Auto-send DNA3 morning brief (first load of the day)
        from dna3_morning_alert import send_morning_alert
        results = send_morning_alert()
        if results:
            # Record sent date
            _config["dna3_morning_alert"]["last_sent_date"] = today_str
            with open(_config_path, 'w') as _f:
                _json.dump(_config, _f, indent=2)
            
            channels = []
            if results.get('telegram'): channels.append("Telegram")
            if results.get('email'): channels.append("Email")
            if channels:
                st.toast(f"📤 DNA3 Morning Brief sent via {' + '.join(channels)}!", icon="🧬")
        st.session_state['dna3_alert_sent'] = True
except Exception as e:
    pass  # Silently fail

# ... imports ...

# --- NAVIGATION CONTROLLER ---
st.sidebar.markdown("---")

# Initialize workspace in session state if not present
if 'active_workspace' not in st.session_state:
    st.session_state['active_workspace'] = "🔍 Market Specs"

active_workspace = st.sidebar.selectbox("📂 Workspace", [
    "🔍 Market Specs", 
    "📋 Portfolio Manager", 
    "⚖️ Analysis Lab"
], key="active_workspace")

page = "🌊 Trend Scanner" # Default

if active_workspace == "🔍 Market Specs":
    page = st.sidebar.radio("View", ["🌊 Trend Scanner", "🚀 Live Trading Desk", "🔍 Market Explorer", "📊 Sector Pulse", "🎯 Turnaround Radar", "🖥️ AI Capex", "🇺🇸 US AI Play", "🇺🇸 US Scanner"], key="page_market_specs")
    
elif active_workspace == "📋 Portfolio Manager":
    page = st.sidebar.radio("Tools", ["📊 Return Tracker", "📝 Notes"], key="page_portfolio")
    
elif active_workspace == "⚖️ Analysis Lab":
    page = st.sidebar.radio("Tools", ["⚖️ Compare Stocks", "📉 Deep Dive", "⏳ Time Trends"], key="page_analysis")

# Handle auto-navigation override (e.g. from Deep Dive buttons)
if 'nav_page' in st.session_state:
    page = st.session_state['nav_page']
    # Clear it so it doesn't persist
    del st.session_state['nav_page']


# --- VIEW: COMPARE STOCKS ---
if page == "⚖️ Compare Stocks":
    st.markdown(page_header("⚖️ Stock Comparison Tool", "Compare two stocks side-by-side on all metrics"), unsafe_allow_html=True)
    
    # Stock selectors
    available_tickers = sorted(df['ticker'].tolist())
    
    col1, col2 = st.columns(2)
    with col1:
        stock_a = st.selectbox("📊 Stock A", available_tickers, index=0, key="comp_stock_a")
    with col2:
        stock_b = st.selectbox("📊 Stock B", available_tickers, index=min(1, len(available_tickers)-1), key="comp_stock_b")
    
    if stock_a and stock_b:
        # Get stock data
        data_a = df[df['ticker'] == stock_a].iloc[0] if not df[df['ticker'] == stock_a].empty else None
        data_b = df[df['ticker'] == stock_b].iloc[0] if not df[df['ticker'] == stock_b].empty else None
        
        if data_a is not None and data_b is not None:
            st.markdown("---")
            st.markdown("### 📈 Key Metrics Comparison")
            
            # Metrics comparison table
            metrics_cols = st.columns(3)
            
            def format_metric(val, is_pct=False, higher_better=True):
                if pd.isna(val):
                    return "N/A"
                if is_pct:
                    return f"{val:+.1f}%" if val >= 0 else f"{val:.1f}%"
                return f"{val:.1f}"
            
            with metrics_cols[0]:
                st.markdown(f"**Metric**")
                for m in ["Trend Score", "Quality", "Value", "Growth", "Momentum", "Volume", "Overall"]:
                    st.write(f"📊 {m}")
            
            with metrics_cols[1]:
                st.markdown(f"**{stock_a.replace('.NS', '')}**")
                st.write(f"🎯 {data_a.get('trend_score', 0):.0f}/100")
                st.write(f"⭐ {data_a.get('quality', 0):.1f}/10")
                st.write(f"💰 {data_a.get('value', 0):.1f}/10")
                st.write(f"📈 {data_a.get('growth', 0):.1f}/10")
                st.write(f"⚡ {data_a.get('momentum', 0):.1f}/10")
                st.write(f"📊 {data_a.get('volume_signal_score', 0):.1f}/10")
                st.write(f"🏆 {data_a.get('overall', 0):.1f}/10")
            
            with metrics_cols[2]:
                st.markdown(f"**{stock_b.replace('.NS', '')}**")
                st.write(f"🎯 {data_b.get('trend_score', 0):.0f}/100")
                st.write(f"⭐ {data_b.get('quality', 0):.1f}/10")
                st.write(f"💰 {data_b.get('value', 0):.1f}/10")
                st.write(f"📈 {data_b.get('growth', 0):.1f}/10")
                st.write(f"⚡ {data_b.get('momentum', 0):.1f}/10")
                st.write(f"📊 {data_b.get('volume_signal_score', 0):.1f}/10")
                st.write(f"🏆 {data_b.get('overall', 0):.1f}/10")
            
            st.markdown("---")
            st.markdown("### 📊 Score Comparison")
            
            # Radar chart comparison
            
            categories = ['Trend', 'Quality', 'Value', 'Growth', 'Momentum', 'Volume']
            
            # Normalize trend score to 0-10 scale for comparison
            values_a = [
                data_a.get('trend_score', 0) / 10,
                data_a.get('quality', 0),
                data_a.get('value', 0),
                data_a.get('growth', 0),
                data_a.get('momentum', 0),
                data_a.get('volume_signal_score', 0)
            ]
            values_b = [
                data_b.get('trend_score', 0) / 10,
                data_b.get('quality', 0),
                data_b.get('value', 0),
                data_b.get('growth', 0),
                data_b.get('momentum', 0),
                data_b.get('volume_signal_score', 0)
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values_a + [values_a[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=stock_a.replace('.NS', ''),
                line_color='#00d4ff'
            ))
            fig.add_trace(go.Scatterpolar(
                r=values_b + [values_b[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=stock_b.replace('.NS', ''),
                line_color='#ff6b6b'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                showlegend=True,
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Price comparison chart
            st.markdown("---")
            st.markdown("### 📈 Price Performance (Last 6 Months)")
            

            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            
            with st.spinner("Fetching price data..."):
                prices_a = yf.download(stock_a, start=start_date, end=end_date, progress=False)['Close']
                prices_b = yf.download(stock_b, start=start_date, end=end_date, progress=False)['Close']
            
            if not prices_a.empty and not prices_b.empty:
                # Normalize to base 100
                norm_a = (prices_a / prices_a.iloc[0]) * 100
                norm_b = (prices_b / prices_b.iloc[0]) * 100
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=norm_a.index, y=norm_a.values, name=stock_a.replace('.NS', ''), line=dict(color='#00d4ff', width=2)))
                fig2.add_trace(go.Scatter(x=norm_b.index, y=norm_b.values, name=stock_b.replace('.NS', ''), line=dict(color='#ff6b6b', width=2)))
                fig2.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="Base")
                fig2.update_layout(
                    height=350,
                    template='plotly_white',
                    yaxis_title='Performance (Base 100)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # Performance summary - extract scalar values from Series
                perf_a = float(((prices_a.iloc[-1] / prices_a.iloc[0]) - 1) * 100)
                perf_b = float(((prices_b.iloc[-1] / prices_b.iloc[0]) - 1) * 100)
                
                winner = stock_a if perf_a > perf_b else stock_b
                
                p_cols = st.columns(3)
                p_cols[0].metric(stock_a.replace('.NS', ''), f"{perf_a:+.1f}%")
                p_cols[1].metric(stock_b.replace('.NS', ''), f"{perf_b:+.1f}%")
                p_cols[2].metric("🏆 Winner", winner.replace('.NS', ''))

# --- VIEW: MARKET EXPLORER ---
elif page == "🔍 Market Explorer":
    render_market_explorer()

# --- VIEW: LIVE TRADING DESK (DNA3-V2.2 + REGIME) ---
elif page == "🚀 Live Trading Desk":
    st.markdown(page_header("🚀 V2.2 Momentum Engine: The Live Trading Desk", "Pure mathematical momentum deployment guided by 15-Year out-of-sample Regime & Seasonality analytics."), unsafe_allow_html=True)
    
    from utils.live_desk import get_live_regime, generate_v3_watchlist
    
    # 1. Use centralized Nifty data (pre-fetched at startup)
    with st.spinner("Calculating Live Macro Regime..."):
        if 'nifty_data' in st.session_state:
            nifty_live = st.session_state['nifty_data']
        else:
            start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
            nifty_live = yf.Ticker("^NSEI").history(start=start_date)
        if not nifty_live.empty:
            regime_data = get_live_regime(nifty_live)
            
            # Show Auto-Regime detection Results
            st.markdown("### 🧭 MACRO REGIME DETECTOR")
            
            rg_col1, rg_col2, rg_col3 = st.columns([1.5, 1, 1])
            with rg_col1:
                st.markdown(f"""
                <div style="background: {regime_data['color']}22; border-left: 5px solid {regime_data['color']}; padding: 15px; border-radius: 4px;">
                    <h3 style="margin:0; color: {regime_data['color']};">Current Trend: {regime_data['regime']}</h3>
                    <p style="margin-top: 5px;">{regime_data['description']}</p>
                </div>
                """, unsafe_allow_html=True)
            with rg_col2:
                st.metric("Optimal Cash Level", f"{regime_data['cash']*100:.0f}%", help="Based on 15Y Regime Backtests")
            with rg_col3:
                st.metric("Max Positions Allowed", f"{regime_data['max_pos']}", help="Fractional Kelly Simulation constraint")
                
            st.markdown("---")
            
            # 2. V2.2 Momentum Engine + Seasonal Indicators
            st.markdown("### 📅 V2.2 Momentum Engine: THE WATCHLIST")
            st.caption(f"*Seasonality Warning:* Recent structural regime changes have broken historical calendar correlations. Seasonality is now shown as an **indicator only**, rather than a strict filter.")
            
            with st.spinner("Running V2.2 Engine + Seasonality checks..."):
                v3_watchlist = generate_v3_watchlist(df)
                
                if not v3_watchlist.empty:
                    v3_watchlist['screener_link'] = "https://www.screener.in/company/" + v3_watchlist['Ticker'].str.replace('.NS', '', regex=False) + "/"
                    st.dataframe(
                        v3_watchlist[['screener_link', 'Target', 'Sector', 'Price', 'V3_Score', 'Cyclicity', 'Seasonality', 'PEAD_Edge', 'Vol_Badge']],
                        column_config={
                            "screener_link": st.column_config.LinkColumn("Ticker", width="small", display_text=r"https://www\.screener\.in/company/(.*?)/"),
                            "Target": "Company Name",
                            "Sector": "Industry Theme",
                            "Price": st.column_config.NumberColumn("CMP", format="₹%.0f"),
                            "V3_Score": st.column_config.ProgressColumn("Trend Conviction", format="%.0f", min_value=0, max_value=100),
                            "Cyclicity": st.column_config.TextColumn("Risk Policy", help="Determines wide (-20%) vs tight (-8%) trailing stops"),
                            "Seasonality": st.column_config.TextColumn("Seasonality", help="Warning indicator only based on 15Y odds"),
                            "PEAD_Edge": st.column_config.TextColumn("Earnings Edge", help="Historical Post-Earnings Reaction"),
                            "Vol_Badge": st.column_config.TextColumn("Inst. Flow", help="Up/Down Volume > 1.2 indicates institutional accumulation")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.info("⚠️ Scanner found absolutely zero safe momentum setups today. Stay in Cash.")
                    
            # 3. Advanced Alpha Scanners (VCP, RS Divergence, Day-0 Shocks)
            st.markdown("---")
            st.markdown("### 🔬 ADVANCED QUANTITATIVE SCANNERS")
            
            @st.cache_data(ttl=3600)
            def get_fast_histories(tickers_tuple):
                d = yf.download(list(tickers_tuple), period="5mo", group_by='ticker', threads=False, progress=False, auto_adjust=True)
                hists = {}
                for t in tickers_tuple:
                    if t in d.columns.get_level_values(0):
                        sub_df = d[t].dropna(how='all')
                        if not sub_df.empty:
                            hists[t] = sub_df
                return hists

            @st.cache_data(ttl=3600)
            def get_us_fast_histories(tickers_tuple):
                d = yf.download(list(tickers_tuple), period="5mo", group_by='ticker', threads=False, progress=False, auto_adjust=True)
                hists = {}
                for t in tickers_tuple:
                    if t in d.columns.get_level_values(0):
                        sub_df = d[t].dropna(how='all')
                        if not sub_df.empty:
                            hists[t] = sub_df
                return hists

            @st.cache_data(ttl=3600)
            def _get_spy_history():
                return yf.Ticker("SPY").history(period="5d")

            @st.cache_data(ttl=3600, show_spinner=False)
            def _get_us_mkt_for_desk():
                return fetch_us_market_data(benchmark="SPY", rs_weights=[(5, 0.30), (21, 0.50), (63, 0.20)], live_mode=False)

            @st.cache_data(ttl=3600, show_spinner=False)
            def _get_prev_day_maps():
                import glob
                files = sorted(glob.glob("data/cache/market_master_*.parquet"))
                if len(files) < 2:
                    return {}, {}
                prev_df = pd.read_parquet(files[-2])
                scores  = prev_df.set_index("ticker")["trend_score"].dropna().to_dict()
                rs_col  = "rs_1m" if "rs_1m" in prev_df.columns else "return_1m"
                rs21    = prev_df.set_index("ticker")[rs_col].dropna().to_dict() if rs_col in prev_df.columns else {}
                return scores, rs21

            _scan_tab_in, _scan_tab_us = st.tabs(["🇮🇳 Nifty Universe", "🇺🇸 S&P 500"])

            with _scan_tab_in:
                with st.spinner("Running Advanced Pattern Recognition..."):
                    top_300_tickers = df.nlargest(300, 'trend_score')['ticker'].tolist()
                    hist_dict = get_fast_histories(tuple(top_300_tickers))

                    vcp_list   = find_vcp_setups(df[df['ticker'].isin(top_300_tickers)], hist_dict)
                    rs_list    = find_rs_divergence(df, nifty_live)
                    shock_list = find_live_earnings_shocks(df[df['ticker'].isin(top_300_tickers)], hist_dict)

                    _prev_scores, _prev_rs21 = _get_prev_day_maps()
                    _in_df = df.copy()
                    if "rs_1m" in _in_df.columns and "return_1m" not in _in_df.columns:
                        _in_df["return_1m"] = _in_df["rs_1m"]
                    tc_list = find_turnaround_catalysts(_in_df, _prev_scores, _prev_rs21)

                    _price_cols = [c for c in ("price", "currentPrice") if c in df.columns]
                    if _price_cols:
                        price_map = df.set_index("ticker")[_price_cols[0]].dropna().to_dict()
                    else:
                        price_map = {}

                    save_rs_divergence_signals(rs_list)
                    save_earnings_shock_signals(shock_list)
                    save_turnaround_catalyst_signals(tc_list)

                    rs_log_df    = refresh_signal_log_prices(RS_LOG_FILE,    RS_LOG_COLS,    price_map)
                    shock_log_df = refresh_signal_log_prices(SHOCK_LOG_FILE, SHOCK_LOG_COLS, price_map)
                    tc_log_df    = refresh_signal_log_prices(TC_LOG_FILE,    TC_LOG_COLS,    price_map)

                col_adv1, col_adv2, col_adv3 = st.columns(3)

                with col_adv1:
                    with st.expander(f"🗜️ Volatility Contraction (VCP) [{len(vcp_list)}]", expanded=True):
                        st.caption("Price tightening (10D ATR < 5.5%) + Volume < 75% of 60D avg. Supply exhaustion before breakout.")
                        if vcp_list:
                            vcp_df = pd.DataFrame(vcp_list)
                            vcp_df['screener_link'] = "https://www.screener.in/company/" + vcp_df['Ticker'].str.replace('.NS', '', regex=False) + "/"
                            st.dataframe(
                                vcp_df[['screener_link', 'Price', 'Score', 'Compression', 'Vol_Ratio', 'MA50_Dist', 'Dist_52W']],
                                column_config={
                                    "screener_link": st.column_config.LinkColumn("Ticker", display_text=r"https://www\.screener\.in/company/(.*?)/"),
                                    "Price":       st.column_config.NumberColumn("Price",      format="₹%.2f"),
                                    "Score":       st.column_config.NumberColumn("Trend Score", format="%.0f"),
                                    "Compression": st.column_config.NumberColumn("10D ATR%",   format="%.1f%%"),
                                    "Vol_Ratio":   st.column_config.NumberColumn("Vol vs 60D", format="%.0f%%"),
                                    "MA50_Dist":   st.column_config.NumberColumn("vs MA50",    format="%+.1f%%"),
                                    "Dist_52W":    st.column_config.NumberColumn("Dist 52W",   format="%.1f%%"),
                                },
                                hide_index=True, use_container_width=True
                            )
                        else:
                            st.info("No VCP setups found today.")

                with col_adv2:
                    with st.expander(f"🟢 RS Divergence — Today [{len(rs_list)}]", expanded=True):
                        st.caption("Green in a sea of Red. Stock closed > +0.3% while Nifty fell > -0.3%.")
                        if rs_list:
                            rs_df = pd.DataFrame(rs_list)
                            rs_df['screener_link'] = "https://www.screener.in/company/" + rs_df['Ticker'].str.replace('.NS', '', regex=False) + "/"
                            st.dataframe(
                                rs_df[['screener_link', 'Stock_Ret', 'Nifty_Ret', 'Delta_RS', 'Dist_52W']],
                                column_config={
                                    "screener_link": st.column_config.LinkColumn("Ticker", display_text=r"https://www\.screener\.in/company/(.*?)/"),
                                    "Stock_Ret": st.column_config.NumberColumn("Stock %",    format="%+.1f%%"),
                                    "Nifty_Ret": st.column_config.NumberColumn("Nifty %",    format="%+.1f%%"),
                                    "Delta_RS":  st.column_config.NumberColumn("Delta RS",   format="%+.1f%%"),
                                    "Dist_52W":  st.column_config.NumberColumn("Dist 52W",   format="%.1f%%"),
                                },
                                hide_index=True, use_container_width=True
                            )
                        else:
                            st.info("Nifty is not falling today, or no divergences found.")

                with col_adv3:
                    with st.expander(f"⚡ Earnings Shocks — Today [{len(shock_list)}]", expanded=True):
                        st.caption("Day-0 gap > 4% on > 2.5× volume. Use PEAD Edge to Buy vs Fade.")
                        if shock_list:
                            shk_df = pd.DataFrame(shock_list)
                            shk_df['screener_link'] = "https://www.screener.in/company/" + shk_df['Ticker'].str.replace('.NS', '', regex=False) + "/"
                            st.dataframe(
                                shk_df[['screener_link', 'Jump_Pct', 'Vol_Mult', 'PEAD_Action']],
                                column_config={
                                    "screener_link": st.column_config.LinkColumn("Ticker", display_text=r"https://www\.screener\.in/company/(.*?)/"),
                                    "Jump_Pct":    st.column_config.NumberColumn("Price Jump",  format="%+.1f%%"),
                                    "Vol_Mult":    st.column_config.NumberColumn("Vol Ratio",   format="%.1fx"),
                                    "PEAD_Action": "Playbook",
                                },
                                hide_index=True, use_container_width=True
                            )
                        else:
                            st.info("No earnings shocks detected today.")

                st.markdown("---")
                with st.expander(f"🔄 Turnaround Catalysts — Today [{len(tc_list)}]", expanded=bool(tc_list)):
                    st.caption("Beaten-down stocks (>20% off 52W high) showing a big reversal day. Pattern A = price gap ≥5% + vol score ≥7. Pattern B = TS jump ≥20 pts in one day. RS21 Vel ≥5 = 🚀")
                    if tc_list:
                        tc_df = pd.DataFrame(tc_list)
                        tc_df['screener_link'] = "https://www.screener.in/company/" + tc_df['Ticker'].str.replace('.NS', '', regex=False) + "/"
                        tc_df['RS21_Flag'] = tc_df['RS21_Vel'].apply(lambda v: "🚀" if v >= 5 else "")
                        st.dataframe(
                            tc_df[['screener_link', 'Name', 'Sector', 'Price', 'Pattern', 'Jump%',
                                   'Vol_Score', 'Dist_52W', 'TS_Pre', 'TS_Now', 'TS_Gain', 'RS21_Vel', 'RS21_Flag']],
                            column_config={
                                "screener_link": st.column_config.LinkColumn("Ticker", display_text=r"https://www\.screener\.in/company/(.*?)/"),
                                "Name":      st.column_config.TextColumn("Name"),
                                "Sector":    st.column_config.TextColumn("Sector"),
                                "Price":     st.column_config.NumberColumn("Price",     format="₹%.1f"),
                                "Pattern":   st.column_config.TextColumn("Pattern"),
                                "Jump%":     st.column_config.NumberColumn("Day Jump",  format="%+.1f%%"),
                                "Vol_Score": st.column_config.NumberColumn("Vol Score", format="%.1f /10"),
                                "Dist_52W":  st.column_config.NumberColumn("Dist 52W",  format="%.1f%%"),
                                "TS_Pre":    st.column_config.NumberColumn("TS Prev",   format="%.0f"),
                                "TS_Now":    st.column_config.NumberColumn("TS Now",    format="%.0f"),
                                "TS_Gain":   st.column_config.NumberColumn("TS Gain",   format="+%.0f"),
                                "RS21_Vel":  st.column_config.NumberColumn("RS21 Vel",  format="%+.1f"),
                                "RS21_Flag": st.column_config.TextColumn("Vel"),
                            },
                            hide_index=True, use_container_width=True
                        )
                    else:
                        st.info("No turnaround catalyst events detected today.")

                st.markdown("### 📋 SIGNAL LOGS — RS Divergence, Earnings Shock & Turnaround Catalysts")

                log_tab1, log_tab2, log_tab3 = st.tabs(["🟢 RS Divergence Log", "⚡ Earnings Shock Log", "🔄 Turnaround Catalyst Log"])

                with log_tab1:
                    st.caption("All RS Divergence signals fired in the last 365 days, with live return tracking. Auto-closed after 21 days.")
                    if not rs_log_df.empty:
                        _rs_disp = rs_log_df.copy()
                        _rs_disp['screener_link'] = "https://www.screener.in/company/" + _rs_disp['ticker'].str.replace('.NS', '', regex=False) + "/"
                        _rs_disp['signal_date'] = pd.to_datetime(_rs_disp['signal_date']).dt.strftime('%Y-%m-%d')
                        st.dataframe(
                            _rs_disp[['signal_date', 'screener_link', 'name', 'sector',
                                      'signal_price', 'current_price', 'return_since_signal',
                                      'stock_ret_on_day', 'nifty_ret_on_day', 'delta_rs',
                                      'dist_52w', 'days_held', 'status']],
                            column_config={
                                "signal_date":          st.column_config.TextColumn("Signal Date"),
                                "screener_link":        st.column_config.LinkColumn("Ticker", display_text=r"https://www\.screener\.in/company/(.*?)/"),
                                "name":                 st.column_config.TextColumn("Name"),
                                "sector":               st.column_config.TextColumn("Sector"),
                                "signal_price":         st.column_config.NumberColumn("Signal ₹",   format="₹%.2f"),
                                "current_price":        st.column_config.NumberColumn("Current ₹",  format="₹%.2f"),
                                "return_since_signal":  st.column_config.NumberColumn("Return %",   format="%+.2f%%"),
                                "stock_ret_on_day":     st.column_config.NumberColumn("Stock Day%", format="%+.2f%%"),
                                "nifty_ret_on_day":     st.column_config.NumberColumn("Nifty Day%", format="%+.2f%%"),
                                "delta_rs":             st.column_config.NumberColumn("Delta RS",   format="%+.2f%%"),
                                "dist_52w":             st.column_config.NumberColumn("Dist 52W",   format="%.1f%%"),
                                "days_held":            st.column_config.NumberColumn("Days"),
                                "status":               st.column_config.TextColumn("Status"),
                            },
                            hide_index=True, use_container_width=True
                        )
                        _rs_csv = _rs_disp.to_csv(index=False).encode()
                        st.download_button("⬇️ Download RS Log CSV", _rs_csv, "rs_divergence_log.csv", "text/csv", key="dl_rs_log")
                    else:
                        st.info("No RS Divergence signals logged yet. The log populates on red-market days when stocks show relative strength.")

                with log_tab2:
                    st.caption("All Earnings Shock signals fired in the last 365 days. PEAD drift tracked at 5D and 21D.")
                    if not shock_log_df.empty:
                        _shk_disp = shock_log_df.copy()
                        _shk_disp['screener_link'] = "https://www.screener.in/company/" + _shk_disp['ticker'].str.replace('.NS', '', regex=False) + "/"
                        _shk_disp['signal_date'] = pd.to_datetime(_shk_disp['signal_date']).dt.strftime('%Y-%m-%d')
                        st.dataframe(
                            _shk_disp[['signal_date', 'screener_link', 'name', 'sector',
                                       'signal_price', 'current_price', 'return_since_signal',
                                       'jump_pct', 'vol_mult', 'pead_action',
                                       'return_5d', 'return_21d', 'days_held', 'status']],
                            column_config={
                                "signal_date":         st.column_config.TextColumn("Signal Date"),
                                "screener_link":       st.column_config.LinkColumn("Ticker", display_text=r"https://www\.screener\.in/company/(.*?)/"),
                                "name":                st.column_config.TextColumn("Name"),
                                "sector":              st.column_config.TextColumn("Sector"),
                                "signal_price":        st.column_config.NumberColumn("Signal ₹",   format="₹%.2f"),
                                "current_price":       st.column_config.NumberColumn("Current ₹",  format="₹%.2f"),
                                "return_since_signal": st.column_config.NumberColumn("Return %",   format="%+.2f%%"),
                                "jump_pct":            st.column_config.NumberColumn("Gap %",      format="%+.2f%%"),
                                "vol_mult":            st.column_config.NumberColumn("Vol Ratio",  format="%.1fx"),
                                "pead_action":         st.column_config.TextColumn("PEAD Playbook"),
                                "return_5d":           st.column_config.NumberColumn("5D Ret %",   format="%+.2f%%"),
                                "return_21d":          st.column_config.NumberColumn("21D Ret %",  format="%+.2f%%"),
                                "days_held":           st.column_config.NumberColumn("Days"),
                                "status":              st.column_config.TextColumn("Status"),
                            },
                            hide_index=True, use_container_width=True
                        )
                        _shk_csv = _shk_disp.to_csv(index=False).encode()
                        st.download_button("⬇️ Download Shock Log CSV", _shk_csv, "earnings_shock_log.csv", "text/csv", key="dl_shock_log")
                    else:
                        st.info("No Earnings Shock signals logged yet. The log populates on days when stocks gap > 4% on > 2.5× volume.")

                with log_tab3:
                    st.caption("All Turnaround Catalyst signals. Pattern A = big gap day. Pattern B = TS jump ≥20 pts. 🚀 = RS21 velocity ≥5.")
                    if not tc_log_df.empty:
                        _tc_disp = tc_log_df.copy()
                        _tc_disp['screener_link'] = "https://www.screener.in/company/" + _tc_disp['ticker'].str.replace('.NS', '', regex=False) + "/"
                        _tc_disp['signal_date'] = pd.to_datetime(_tc_disp['signal_date']).dt.strftime('%Y-%m-%d')
                        _tc_disp['vel_flag'] = _tc_disp['rs21_vel'].apply(lambda v: "🚀" if float(v or 0) >= 5 else "")
                        _tc_cols = [c for c in ['signal_date', 'screener_link', 'name', 'sector',
                                                 'signal_price', 'current_price', 'return_since_signal',
                                                 'pattern', 'jump_pct', 'vol_score', 'dist_52w',
                                                 'ts_pre', 'ts_now', 'ts_gain', 'rs21_vel', 'vel_flag',
                                                 'days_held', 'status'] if c in _tc_disp.columns or c in ('screener_link','vel_flag')]
                        st.dataframe(
                            _tc_disp[_tc_cols],
                            column_config={
                                "signal_date":         st.column_config.TextColumn("Signal Date"),
                                "screener_link":       st.column_config.LinkColumn("Ticker", display_text=r"https://www\.screener\.in/company/(.*?)/"),
                                "name":                st.column_config.TextColumn("Name"),
                                "sector":              st.column_config.TextColumn("Sector"),
                                "signal_price":        st.column_config.NumberColumn("Signal ₹",  format="₹%.2f"),
                                "current_price":       st.column_config.NumberColumn("Current ₹", format="₹%.2f"),
                                "return_since_signal": st.column_config.NumberColumn("Return %",  format="%+.2f%%"),
                                "pattern":             st.column_config.TextColumn("Pattern"),
                                "jump_pct":            st.column_config.NumberColumn("Day Jump",  format="%+.1f%%"),
                                "vol_score":           st.column_config.NumberColumn("Vol Score", format="%.1f /10"),
                                "dist_52w":            st.column_config.NumberColumn("Dist 52W",  format="%.1f%%"),
                                "ts_pre":              st.column_config.NumberColumn("TS Prev",   format="%.0f"),
                                "ts_now":              st.column_config.NumberColumn("TS Now",    format="%.0f"),
                                "ts_gain":             st.column_config.NumberColumn("TS Gain",   format="+%.0f"),
                                "rs21_vel":            st.column_config.NumberColumn("RS21 Vel",  format="%+.1f"),
                                "vel_flag":            st.column_config.TextColumn("Vel"),
                                "days_held":           st.column_config.NumberColumn("Days"),
                                "status":              st.column_config.TextColumn("Status"),
                            },
                            hide_index=True, use_container_width=True
                        )
                        st.download_button("⬇️ Download TC Log CSV", _tc_disp.to_csv(index=False).encode(), "tc_log.csv", "text/csv", key="dl_tc_log")
                    else:
                        st.info("No Turnaround Catalyst signals logged yet. The log populates on days with big reversal moves in beaten-down stocks.")

            with _scan_tab_us:
                st.caption("RS Divergence vs SPY | VCP setups | Earnings Gaps — across S&P 500 universe.")

                with st.spinner("Running US Alpha Scanners..."):
                    us_mkt_df = _get_us_mkt_for_desk()
                    spy_live  = _get_spy_history()

                    from utils.advanced_scanners import (
                        save_us_rs_divergence_signals, save_us_earnings_shock_signals,
                        US_RS_LOG_FILE, US_SHOCK_LOG_FILE,
                    )

                    if not us_mkt_df.empty:
                        _us_top = us_mkt_df.nlargest(150, 'trend_score')['ticker'].tolist()
                        us_hist_dict = get_us_fast_histories(tuple(_us_top))

                        us_vcp_list   = find_vcp_setups(us_mkt_df[us_mkt_df['ticker'].isin(_us_top)], us_hist_dict)
                        us_rs_list    = find_rs_divergence(us_mkt_df, spy_live)
                        us_shock_list = find_live_earnings_shocks(us_mkt_df[us_mkt_df['ticker'].isin(_us_top)], us_hist_dict)

                        _us_df_tc = us_mkt_df.copy()
                        if "rs_1m" in _us_df_tc.columns and "return_1m" not in _us_df_tc.columns:
                            _us_df_tc["return_1m"] = _us_df_tc["rs_1m"]
                        us_tc_list = find_turnaround_catalysts(_us_df_tc)

                        _us_price_map = us_mkt_df.set_index('ticker')['price'].dropna().to_dict()
                        save_us_rs_divergence_signals(us_rs_list)
                        save_us_earnings_shock_signals(us_shock_list)
                        save_turnaround_catalyst_signals(us_tc_list, log_file=US_TC_LOG_FILE)
                        us_rs_log_df    = refresh_signal_log_prices(US_RS_LOG_FILE,    RS_LOG_COLS,    _us_price_map)
                        us_shock_log_df = refresh_signal_log_prices(US_SHOCK_LOG_FILE, SHOCK_LOG_COLS, _us_price_map)
                        us_tc_log_df    = refresh_signal_log_prices(US_TC_LOG_FILE,    TC_LOG_COLS,    _us_price_map)
                    else:
                        us_vcp_list = []; us_rs_list = []; us_shock_list = []; us_tc_list = []
                        us_rs_log_df = pd.DataFrame(); us_shock_log_df = pd.DataFrame(); us_tc_log_df = pd.DataFrame()

                us_adv1, us_adv2, us_adv3 = st.columns(3)

                with us_adv1:
                    with st.expander(f"🗜️ VCP Setups [{len(us_vcp_list)}]", expanded=True):
                        st.caption("Price tightening + volume dry-up across top S&P 500 momentum names.")
                        if us_vcp_list:
                            _uvcp = pd.DataFrame(us_vcp_list)
                            _uvcp['yf_link'] = _uvcp.apply(lambda r: _google_finance_url(r['Ticker']), axis=1)
                            st.dataframe(
                                _uvcp[['yf_link', 'Price', 'Score', 'Compression', 'Vol_Ratio', 'MA50_Dist', 'Dist_52W']],
                                column_config={
                                    "yf_link":     st.column_config.LinkColumn("Ticker", display_text=_GF_DISPLAY_RE),
                                    "Price":       st.column_config.NumberColumn("Price",       format="$%.2f"),
                                    "Score":       st.column_config.NumberColumn("Trend Score", format="%.0f"),
                                    "Compression": st.column_config.NumberColumn("10D ATR%",    format="%.1f%%"),
                                    "Vol_Ratio":   st.column_config.NumberColumn("Vol vs 60D",  format="%.0f%%"),
                                    "MA50_Dist":   st.column_config.NumberColumn("vs MA50",     format="%+.1f%%"),
                                    "Dist_52W":    st.column_config.NumberColumn("Dist 52W",    format="%.1f%%"),
                                },
                                hide_index=True, use_container_width=True
                            )
                        else:
                            st.info("No US VCP setups found today.")

                with us_adv2:
                    with st.expander(f"🟢 RS vs SPY — Today [{len(us_rs_list)}]", expanded=True):
                        st.caption("US stocks closing positive while SPY falls > -0.3%.")
                        if us_rs_list:
                            _urs = pd.DataFrame(us_rs_list)
                            _urs['yf_link'] = _urs.apply(lambda r: _google_finance_url(r['Ticker']), axis=1)
                            st.dataframe(
                                _urs[['yf_link', 'Stock_Ret', 'Nifty_Ret', 'Delta_RS', 'Dist_52W']],
                                column_config={
                                    "yf_link":   st.column_config.LinkColumn("Ticker", display_text=_GF_DISPLAY_RE),
                                    "Stock_Ret": st.column_config.NumberColumn("Stock %",  format="%+.1f%%"),
                                    "Nifty_Ret": st.column_config.NumberColumn("SPY %",    format="%+.1f%%"),
                                    "Delta_RS":  st.column_config.NumberColumn("Delta RS", format="%+.1f%%"),
                                    "Dist_52W":  st.column_config.NumberColumn("Dist 52W", format="%.1f%%"),
                                },
                                hide_index=True, use_container_width=True
                            )
                        else:
                            st.info("SPY is not falling today, or no US divergences found.")

                with us_adv3:
                    with st.expander(f"⚡ US Earnings Gaps [{len(us_shock_list)}]", expanded=True):
                        st.caption("Day-0 gap > 4% on > 2.5× volume across S&P 500.")
                        if us_shock_list:
                            _ushk = pd.DataFrame(us_shock_list)
                            _ushk['yf_link'] = _ushk.apply(lambda r: _google_finance_url(r['Ticker']), axis=1)
                            st.dataframe(
                                _ushk[['yf_link', 'Jump_Pct', 'Vol_Mult', 'PEAD_Action']],
                                column_config={
                                    "yf_link":     st.column_config.LinkColumn("Ticker", display_text=_GF_DISPLAY_RE),
                                    "Jump_Pct":    st.column_config.NumberColumn("Price Jump", format="%+.1f%%"),
                                    "Vol_Mult":    st.column_config.NumberColumn("Vol Ratio",  format="%.1fx"),
                                    "PEAD_Action": "Playbook",
                                },
                                hide_index=True, use_container_width=True
                            )
                        else:
                            st.info("No US earnings gaps detected today.")

                st.markdown("---")
                with st.expander(f"🔄 US Turnaround Catalysts — Today [{len(us_tc_list)}]", expanded=bool(us_tc_list)):
                    st.caption("Beaten-down S&P 500 stocks showing an initial reversal catalyst. Pattern A = price gap ≥5% + vol score ≥7. Pattern B = TS jump ≥20 pts.")
                    if us_tc_list:
                        _utc = pd.DataFrame(us_tc_list)
                        _utc['yf_link'] = _utc.apply(lambda r: _google_finance_url(r['Ticker']), axis=1)
                        _utc['RS21_Flag'] = _utc['RS21_Vel'].apply(lambda v: "🚀" if v >= 5 else "")
                        st.dataframe(
                            _utc[['yf_link', 'Name', 'Sector', 'Price', 'Pattern', 'Jump%',
                                  'Vol_Score', 'Dist_52W', 'TS_Pre', 'TS_Now', 'TS_Gain', 'RS21_Vel', 'RS21_Flag']],
                            column_config={
                                "yf_link":   st.column_config.LinkColumn("Ticker", display_text=_GF_DISPLAY_RE),
                                "Name":      st.column_config.TextColumn("Name"),
                                "Sector":    st.column_config.TextColumn("Sector"),
                                "Price":     st.column_config.NumberColumn("Price",     format="$%.2f"),
                                "Pattern":   st.column_config.TextColumn("Pattern"),
                                "Jump%":     st.column_config.NumberColumn("Day Jump",  format="%+.1f%%"),
                                "Vol_Score": st.column_config.NumberColumn("Vol Score", format="%.1f /10"),
                                "Dist_52W":  st.column_config.NumberColumn("Dist 52W",  format="%.1f%%"),
                                "TS_Pre":    st.column_config.NumberColumn("TS Prev",   format="%.0f"),
                                "TS_Now":    st.column_config.NumberColumn("TS Now",    format="%.0f"),
                                "TS_Gain":   st.column_config.NumberColumn("TS Gain",   format="+%.0f"),
                                "RS21_Vel":  st.column_config.NumberColumn("RS21 Vel",  format="%+.1f"),
                                "RS21_Flag": st.column_config.TextColumn("Vel"),
                            },
                            hide_index=True, use_container_width=True
                        )
                    else:
                        st.info("No US turnaround catalyst events detected today.")

                st.markdown("### 📋 US SIGNAL LOGS — RS Divergence, Earnings Shock & Turnaround Catalysts")

                us_log_tab1, us_log_tab2, us_log_tab3 = st.tabs(["🟢 US RS Divergence Log", "⚡ US Earnings Shock Log", "🔄 US Turnaround Catalyst Log"])

                with us_log_tab1:
                    st.caption("US stocks showing RS vs SPY in the last 365 days. Auto-closed after 21 days.")
                    if not us_rs_log_df.empty:
                        _urs_disp = us_rs_log_df.copy()
                        _urs_disp['yf_link'] = _urs_disp.apply(lambda r: _google_finance_url(r['ticker']), axis=1)
                        _urs_disp['signal_date'] = pd.to_datetime(_urs_disp['signal_date']).dt.strftime('%Y-%m-%d')
                        st.dataframe(
                            _urs_disp[['signal_date', 'yf_link', 'name', 'sector',
                                       'signal_price', 'current_price', 'return_since_signal',
                                       'stock_ret_on_day', 'nifty_ret_on_day', 'delta_rs',
                                       'dist_52w', 'days_held', 'status']],
                            column_config={
                                "signal_date":          st.column_config.TextColumn("Signal Date"),
                                "yf_link":              st.column_config.LinkColumn("Ticker", display_text=_GF_DISPLAY_RE),
                                "name":                 st.column_config.TextColumn("Name"),
                                "sector":               st.column_config.TextColumn("Sector"),
                                "signal_price":         st.column_config.NumberColumn("Signal $",   format="$%.2f"),
                                "current_price":        st.column_config.NumberColumn("Current $",  format="$%.2f"),
                                "return_since_signal":  st.column_config.NumberColumn("Return %",   format="%+.2f%%"),
                                "stock_ret_on_day":     st.column_config.NumberColumn("Stock Day%", format="%+.2f%%"),
                                "nifty_ret_on_day":     st.column_config.NumberColumn("SPY Day%",   format="%+.2f%%"),
                                "delta_rs":             st.column_config.NumberColumn("Delta RS",   format="%+.2f%%"),
                                "dist_52w":             st.column_config.NumberColumn("Dist 52W",   format="%.1f%%"),
                                "days_held":            st.column_config.NumberColumn("Days"),
                                "status":               st.column_config.TextColumn("Status"),
                            },
                            hide_index=True, use_container_width=True
                        )
                        st.download_button("⬇️ Download US RS Log CSV", _urs_disp.to_csv(index=False).encode(), "us_rs_divergence_log.csv", "text/csv", key="dl_us_rs_log")
                    else:
                        st.info("No US RS Divergence signals logged yet.")

                with us_log_tab2:
                    st.caption("US earnings gap signals from the last 365 days. PEAD drift tracked at 5D and 21D.")
                    if not us_shock_log_df.empty:
                        _ushk_disp = us_shock_log_df.copy()
                        _ushk_disp['yf_link'] = _ushk_disp.apply(lambda r: _google_finance_url(r['ticker']), axis=1)
                        _ushk_disp['signal_date'] = pd.to_datetime(_ushk_disp['signal_date']).dt.strftime('%Y-%m-%d')
                        st.dataframe(
                            _ushk_disp[['signal_date', 'yf_link', 'name', 'sector',
                                        'signal_price', 'current_price', 'return_since_signal',
                                        'jump_pct', 'vol_mult', 'pead_action',
                                        'return_5d', 'return_21d', 'days_held', 'status']],
                            column_config={
                                "signal_date":         st.column_config.TextColumn("Signal Date"),
                                "yf_link":             st.column_config.LinkColumn("Ticker", display_text=_GF_DISPLAY_RE),
                                "name":                st.column_config.TextColumn("Name"),
                                "sector":              st.column_config.TextColumn("Sector"),
                                "signal_price":        st.column_config.NumberColumn("Signal $",   format="$%.2f"),
                                "current_price":       st.column_config.NumberColumn("Current $",  format="$%.2f"),
                                "return_since_signal": st.column_config.NumberColumn("Return %",   format="%+.2f%%"),
                                "jump_pct":            st.column_config.NumberColumn("Gap %",      format="%+.2f%%"),
                                "vol_mult":            st.column_config.NumberColumn("Vol Ratio",  format="%.1fx"),
                                "pead_action":         st.column_config.TextColumn("PEAD Playbook"),
                                "return_5d":           st.column_config.NumberColumn("5D Ret %",   format="%+.2f%%"),
                                "return_21d":          st.column_config.NumberColumn("21D Ret %",  format="%+.2f%%"),
                                "days_held":           st.column_config.NumberColumn("Days"),
                                "status":              st.column_config.TextColumn("Status"),
                            },
                            hide_index=True, use_container_width=True
                        )
                        st.download_button("⬇️ Download US Shock Log CSV", _ushk_disp.to_csv(index=False).encode(), "us_earnings_shock_log.csv", "text/csv", key="dl_us_shock_log")
                    else:
                        st.info("No US Earnings Shock signals logged yet.")

                with us_log_tab3:
                    st.caption("US turnaround catalyst signals. Pattern A = big gap day. Pattern B = TS jump ≥20 pts. 🚀 = RS21 velocity ≥5.")
                    if not us_tc_log_df.empty:
                        _utc_disp = us_tc_log_df.copy()
                        _utc_disp['yf_link'] = _utc_disp.apply(lambda r: _google_finance_url(r['ticker']), axis=1)
                        _utc_disp['signal_date'] = pd.to_datetime(_utc_disp['signal_date']).dt.strftime('%Y-%m-%d')
                        _utc_disp['vel_flag'] = _utc_disp['rs21_vel'].apply(lambda v: "🚀" if float(v or 0) >= 5 else "")
                        _utc_cols = [c for c in ['signal_date', 'yf_link', 'name', 'sector',
                                                  'signal_price', 'current_price', 'return_since_signal',
                                                  'pattern', 'jump_pct', 'vol_score', 'dist_52w',
                                                  'ts_pre', 'ts_now', 'ts_gain', 'rs21_vel', 'vel_flag',
                                                  'days_held', 'status'] if c in _utc_disp.columns or c in ('yf_link','vel_flag')]
                        st.dataframe(
                            _utc_disp[_utc_cols],
                            column_config={
                                "signal_date":         st.column_config.TextColumn("Signal Date"),
                                "yf_link":             st.column_config.LinkColumn("Ticker", display_text=_GF_DISPLAY_RE),
                                "name":                st.column_config.TextColumn("Name"),
                                "sector":              st.column_config.TextColumn("Sector"),
                                "signal_price":        st.column_config.NumberColumn("Signal $",  format="$%.2f"),
                                "current_price":       st.column_config.NumberColumn("Current $", format="$%.2f"),
                                "return_since_signal": st.column_config.NumberColumn("Return %",  format="%+.2f%%"),
                                "pattern":             st.column_config.TextColumn("Pattern"),
                                "jump_pct":            st.column_config.NumberColumn("Day Jump",  format="%+.1f%%"),
                                "vol_score":           st.column_config.NumberColumn("Vol Score", format="%.1f /10"),
                                "dist_52w":            st.column_config.NumberColumn("Dist 52W",  format="%.1f%%"),
                                "ts_pre":              st.column_config.NumberColumn("TS Prev",   format="%.0f"),
                                "ts_now":              st.column_config.NumberColumn("TS Now",    format="%.0f"),
                                "ts_gain":             st.column_config.NumberColumn("TS Gain",   format="+%.0f"),
                                "rs21_vel":            st.column_config.NumberColumn("RS21 Vel",  format="%+.1f"),
                                "vel_flag":            st.column_config.TextColumn("Vel"),
                                "days_held":           st.column_config.NumberColumn("Days"),
                                "status":              st.column_config.TextColumn("Status"),
                            },
                            hide_index=True, use_container_width=True
                        )
                        st.download_button("⬇️ Download US TC Log CSV", _utc_disp.to_csv(index=False).encode(), "us_tc_log.csv", "text/csv", key="dl_us_tc_log")
                    else:
                        st.info("No US Turnaround Catalyst signals logged yet.")

        else:
            st.error("Failed to connect to NSE index to calculate regime.")

elif page == "🌊 Trend Scanner":
    
    # === HERO SECTION ===
    st.markdown(page_header("🌊 Alpha Trend Scanner", "Real-time momentum intelligence for Nifty 500 | Powered by AI"), unsafe_allow_html=True)
    
    # === DNA3-V2.2 MODEL PORTFOLIO SECTION (LIVE TRACKING) ===
    
    DNA3_SNAPSHOT = "data/dna3_portfolio_snapshot.json"
    DNA3_EQUITY = "data/dna3_equity_curve.csv"
    DNA3_LOG = "data/dna3_trade_log.csv"
    
    if os.path.exists(DNA3_SNAPSHOT):
        try:
            with open(DNA3_SNAPSHOT, 'r') as f:
                dna3_data = json.load(f)
            
            # Load Equity Curve for returns
            live_return_pct = 0.0
            equity_val = 1000000

            # Read initial capital from snapshot config so it stays in sync
            # with INITIAL_CAPITAL in dna3_current_portfolio.py
            _dna3_cfg = dna3_data.get('config', {})
            start_eq = float(_dna3_cfg.get('initial_capital', 1_000_000))

            if os.path.exists(DNA3_EQUITY):
                eq_df = pd.read_csv(DNA3_EQUITY)
                if not eq_df.empty:
                    last_eq = eq_df['Equity'].iloc[-1]
                    live_return_pct = (last_eq - start_eq) / start_eq * 100
                    equity_val = last_eq

            # === HERO P&L CARD (Robinhood-style) ===
            st.markdown(hero_pnl_card(
                portfolio_value=equity_val,
                return_pct=live_return_pct,
                holdings_count=dna3_data.get('count', 0)
            ), unsafe_allow_html=True)
            
            # === EQUITY VS NIFTY CHART ===
            if os.path.exists(DNA3_EQUITY):
                eq_chart_df = pd.read_csv(DNA3_EQUITY)
                eq_chart_df['Date'] = pd.to_datetime(eq_chart_df['Date'])
                eq_chart_df = eq_chart_df.drop_duplicates(subset='Date', keep='last').sort_values('Date')
                
                if len(eq_chart_df) >= 2:
                    import plotly.graph_objects as go
                    
                    # Fetch Nifty data for same period
                    start_str = eq_chart_df['Date'].iloc[0].strftime('%Y-%m-%d')
                    try:
                        nifty_eq = yf.download("^NSEI", start=start_str, progress=False,
                                               threads=False, auto_adjust=True)
                        if isinstance(nifty_eq.columns, pd.MultiIndex):
                            nifty_eq.columns = nifty_eq.columns.get_level_values(0)
                        if nifty_eq.index.tz is not None:
                            nifty_eq.index = nifty_eq.index.tz_localize(None)
                        
                        # Normalize both to 100 at inception (same base as hero card)
                        port_norm = eq_chart_df['Equity'] / start_eq * 100
                        
                        # Match Nifty to portfolio dates
                        nifty_close = nifty_eq['Close'].reindex(eq_chart_df['Date'].values, method='ffill')
                        nifty_base = nifty_close.iloc[0] if not pd.isna(nifty_close.iloc[0]) else nifty_close.dropna().iloc[0]
                        nifty_norm = nifty_close / nifty_base * 100
                        
                        fig = go.Figure()
                        
                        # Portfolio line
                        fig.add_trace(go.Scatter(
                            x=eq_chart_df['Date'], y=port_norm,
                            name='Portfolio', mode='lines+markers',
                            line=dict(color='#00d4aa', width=3),
                            marker=dict(size=6),
                            hovertemplate='Portfolio: %{y:.1f}<br>Value: ₹%{customdata:,.0f}<extra></extra>',
                            customdata=eq_chart_df['Equity']
                        ))
                        
                        # Nifty line
                        fig.add_trace(go.Scatter(
                            x=eq_chart_df['Date'], y=nifty_norm,
                            name='Nifty 50', mode='lines',
                            line=dict(color='#666', width=2, dash='dot'),
                            hovertemplate='Nifty: %{y:.1f}<extra></extra>'
                        ))
                        
                        # 100 baseline
                        fig.add_hline(y=100, line_dash="dash", line_color="rgba(255,255,255,0.2)", line_width=1)
                        
                        fig.update_layout(
                            title=None,
                            template='plotly_dark',
                            height=280,
                            margin=dict(l=10, r=10, t=10, b=10),
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                            xaxis=dict(showgrid=False),
                            yaxis=dict(title='Normalized (100 = Start)', showgrid=True,
                                       gridcolor='rgba(255,255,255,0.05)'),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key="equity_vs_nifty_chart")
                    except Exception:
                        pass  # Silently skip chart if Nifty fetch fails

            # === TRADE SUMMARY STATS ===
            if os.path.exists(DNA3_LOG):
                tlog = pd.read_csv(DNA3_LOG)
                buys = tlog[tlog['Action'] == 'BUY']
                sells = tlog[tlog['Action'] == 'SELL']
                
                if not sells.empty:
                    wins = sells[sells['PnL%'] > 0]
                    losses = sells[sells['PnL%'] <= 0]
                    wr = len(wins) / len(sells) * 100 if len(sells) > 0 else 0
                    avg_win = wins['PnL%'].mean() if not wins.empty else 0
                    avg_loss = losses['PnL%'].mean() if not losses.empty else 0
                    best = sells['PnL%'].max()
                    worst = sells['PnL%'].min()
                    
                    ts1, ts2, ts3, ts4, ts5 = st.columns(5)
                    ts1.metric("Total Buys", len(buys))
                    ts2.metric("Total Sells", len(sells))
                    ts3.metric("Win Rate", f"{wr:.0f}%")
                    ts4.metric("Avg Win", f"{avg_win:+.1f}%")
                    ts5.metric("Avg Loss", f"{avg_loss:+.1f}%")
                
                # Trade History in expander
                with st.expander(f"📋 **Trade History** ({len(tlog)} records)", expanded=False):
                    display_tlog = tlog.copy()
                    display_tlog['Ticker'] = display_tlog['Ticker'].str.replace('.NS', '', regex=False)
                    st.dataframe(
                        display_tlog,
                        column_config={
                            "Price": st.column_config.NumberColumn("Price", format="₹%.2f"),
                            "PnL": st.column_config.NumberColumn("P&L", format="₹%.2f"),
                            "PnL%": st.column_config.NumberColumn("P&L%", format="%+.2f%%"),
                        },
                        hide_index=True,
                        use_container_width=True,
                    )
            
            # Portfolio details + controls row
            hero_left, hero_right = st.columns([3, 1])
            
            with hero_left:
                if dna3_data.get('portfolio'):
                    p_df = pd.DataFrame(dna3_data['portfolio'])
                    p_df['screener_link'] = "https://www.screener.in/company/" + p_df['Ticker'].str.replace('.NS', '', regex=False) + "/"
                    st.dataframe(
                        p_df[['screener_link', 'Sector', 'Price', 'Entry', 'PnL%', 'RS_Score', 'Dist_MA50']],
                        column_config={
                            "screener_link": st.column_config.LinkColumn("Stock", display_text=r"https://www\.screener\.in/company/(.*?)/"),
                            "Price": st.column_config.NumberColumn("Current Price", format="₹%.2f"),
                            "Entry": st.column_config.NumberColumn("Entry Price", format="₹%.2f"),
                            "PnL%": st.column_config.NumberColumn("Unrealized P&L", format="%+.1f%%"),
                            "RS_Score": st.column_config.ProgressColumn("RS Score", min_value=0, max_value=100, format="%.1f"),
                            "Dist_MA50": st.column_config.NumberColumn("% vs MA50", format="%.1f%%")
                        },
                        hide_index=True,
                        use_container_width=True,
                        height=250
                    )
                else:
                    st.info("No stocks meet DNA3 criteria currently (Cash Mode).")
            
            with hero_right:
                if st.button("🔄 Refresh Portfolio", use_container_width=True):
                    with st.spinner("Fetching live prices for portfolio..."):
                        try:
                            # Fast inline refresh: only fetch held tickers (~10 stocks)
                            held_tickers = [p['Ticker'] for p in dna3_data.get('portfolio', [])]
                            if held_tickers:
                                live_prices = yf.download(held_tickers, period="100d", group_by='ticker',
                                                         threads=False, progress=False, auto_adjust=True)
                                nifty_hist = yf.Ticker("^NSEI").history(period="100d")
                                if nifty_hist.index.tz is not None:
                                    nifty_hist.index = nifty_hist.index.tz_localize(None)
                                nifty_price = float(nifty_hist['Close'].iloc[-1])
                                
                                # RS weight config (same as OptComp-V22)
                                rs_weights = [(5, 0.10), (21, 0.50), (63, 0.40)]
                                
                                updated_portfolio = []
                                for p in dna3_data['portfolio']:
                                    t = p['Ticker']
                                    try:
                                        if len(held_tickers) == 1:
                                            stock_df = live_prices.dropna(how='all')
                                        else:
                                            stock_df = live_prices[t].dropna(how='all') if t in live_prices.columns.get_level_values(0) else None
                                        if stock_df is not None and not stock_df.empty and len(stock_df) > 50:
                                            if stock_df.index.tz is not None:
                                                stock_df.index = stock_df.index.tz_localize(None)
                                            curr_price = float(stock_df['Close'].iloc[-1])
                                            ma50 = float(stock_df['Close'].rolling(50).mean().iloc[-1])
                                            dist_ma50 = (curr_price - ma50) / ma50 * 100
                                            
                                            # Composite RS: iloc[-(period+1)] gives exactly
                                            # `period` trading-day intervals (same convention
                                            # as fast_data_engine searchsorted approach)
                                            rs_total = 0.0
                                            for period, weight in rs_weights:
                                                if len(stock_df) >= period + 2 and len(nifty_hist) >= period + 2:
                                                    rs_stock = (curr_price / float(stock_df['Close'].iloc[-(period + 1)]) - 1)
                                                    rs_nifty = (nifty_price / float(nifty_hist['Close'].iloc[-(period + 1)]) - 1)
                                                    rs_total += (rs_stock - rs_nifty) * 100 * weight
                                            
                                            entry_price = dna3_data['holdings'].get(t, {}).get('entry_price', p.get('Entry', curr_price))
                                            pnl_pct = (curr_price - entry_price) / entry_price * 100
                                            
                                            p['Price'] = round(curr_price, 2)
                                            p['PnL%'] = round(pnl_pct, 2)
                                            p['RS_Score'] = round(rs_total, 1)
                                            p['Dist_MA50'] = round(dist_ma50, 1)
                                    except Exception:
                                        pass  # Keep original values if fetch fails
                                    updated_portfolio.append(p)
                                
                                # Update equity
                                total_equity = dna3_data.get('cash', 0)
                                for p in updated_portfolio:
                                    t = p['Ticker']
                                    shares = dna3_data['holdings'].get(t, {}).get('shares', 0)
                                    total_equity += shares * p['Price']
                                
                                dna3_data['portfolio'] = updated_portfolio
                                dna3_data['equity'] = total_equity
                                dna3_data['date'] = datetime.now().strftime('%Y-%m-%d')
                                
                                with open(DNA3_SNAPSHOT, 'w') as f:
                                    json.dump(dna3_data, f, indent=4)
                                
                                st.toast("✅ Portfolio prices updated!", icon="🔄")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Refresh failed: {e}")
                            st.rerun()
                if os.path.exists(DNA3_LOG):
                    with open(DNA3_LOG, "rb") as file:
                        st.download_button(
                            label="📥 Trade Log",
                            data=file,
                            file_name="dna3_trade_log.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

        except Exception as e:
            st.error(f"Error loading DNA3 Portfolio: {e}")
    
    # === DNA3 ALERT CONTROLS ===
    with st.expander("🔔 **DNA3 Alert Settings** (Telegram + Email)", expanded=False):
        _json = json
        _config_path = "config.json"
        _config = {}
        if os.path.exists(_config_path):
            with open(_config_path, 'r') as _f:
                _config = _json.load(_f)
        
        dna3_cfg = _config.get("dna3_morning_alert", {"enabled": False, "last_sent_date": ""})
        
        a_col1, a_col2, a_col3 = st.columns([2, 2, 2])
        
        with a_col1:
            auto_enabled = st.toggle("Daily Auto-Send", value=dna3_cfg.get("enabled", False), 
                                      help="Auto-sends DNA3 portfolio brief via Telegram + Email once per day when dashboard loads")
            # Save toggle state
            if auto_enabled != dna3_cfg.get("enabled", False):
                _config["dna3_morning_alert"] = {**dna3_cfg, "enabled": auto_enabled}
                with open(_config_path, 'w') as _f:
                    _json.dump(_config, _f, indent=2)
                st.rerun()
        
        with a_col2:
            if st.button("📤 Send Now", use_container_width=True, help="Send DNA3 brief right now"):
                with st.spinner("Sending..."):
                    from dna3_morning_alert import send_morning_alert
                    results = send_morning_alert()
                    if results:
                        channels = []
                        if results.get('telegram'): channels.append("✅ Telegram")
                        if results.get('email'): channels.append("✅ Email")
                        failed = []
                        if not results.get('telegram'): failed.append("❌ Telegram")
                        if not results.get('email'): failed.append("❌ Email")
                        st.success(" | ".join(channels + failed))
                    else:
                        st.error("No portfolio data. Run Refresh first.")
        
        with a_col3:
            last_sent = dna3_cfg.get("last_sent_date", "Never")
            st.caption(f"Last sent: **{last_sent}**")
            
            # Check config status
            from utils.telegram_notifier import is_telegram_configured
            from utils.email_notifier import is_email_configured
            tg_ok = "✅" if is_telegram_configured() else "❌"
            em_ok = "✅" if is_email_configured() else "❌"
            st.caption(f"Telegram: {tg_ok} | Email: {em_ok}")
    
    # === END DNA3 SECTION ===
    
    # === QUICK STATS ===
    col1, col2, col3, col4 = st.columns(4)
    
    strong_uptrends = len(df[df['trend_signal'] == "STRONG UPTREND"])
    total_uptrends = len(df[df['trend_signal'].isin(["STRONG UPTREND", "UPTREND"])])
    avg_trend = df['trend_score'].mean()
    breakout_count = len(df[df['dist_52w'] > -2.0])
    
    col1.metric("🚀 Strong Momentum", f"{strong_uptrends}", help="Stocks with Trend Score > 80")
    col2.metric("📈 Total Uptrends", f"{total_uptrends}", help="Stocks in upward trajectory")
    col3.metric("📊 Avg Trend Score", f"{avg_trend:.0f}/100", help="Market-wide momentum gauge")
    col4.metric("🔥 Breakout Alerts", f"{breakout_count}", help="Near 52-week highs")
    
    # === MARKET MOOD HISTORY CHART ===
    # market_mood imports at top
    
    # Save today's snapshot
    mood_metrics = calculate_mood_metrics(df)
    if mood_metrics:
        save_mood_snapshot(mood_metrics)
    
    # Check and send mood-based alerts (if threshold crossed)
    try:
        from utils.telegram_notifier import check_and_send_mood_alerts, is_telegram_configured
        mood_history_temp = load_mood_history()
        if is_telegram_configured() and len(mood_history_temp) > 1:
            alert_result = check_and_send_mood_alerts(df, mood_history_temp)
            if alert_result.get('sent'):
                st.toast("📢 Market Timing Alert sent!", icon="🔔")
    except Exception as e:
        pass  # Silently fail if alert check fails
    
    # Display chart
    mood_history = load_mood_history()
    if not mood_history.empty and len(mood_history) > 1:
        with st.expander("🌡️ **Market Mood History** (Last 1 Year)", expanded=True):
            mood_chart = chart_market_mood(mood_history)
            if mood_chart:
                st.plotly_chart(mood_chart, use_container_width=True)
            else:
                st.caption("Chart will appear after 2+ days of data collection.")
    else:
        st.caption("📊 Market Mood chart will appear after 2+ days of tracking.")
    
    # === MARKET BREADTH MONITOR (Narrow Market Detector) ===
    # market_breadth import at top
    render_breadth_widget(df)
    
    # === MARKET TIMING SIGNALS (Based on Analysis) ===
    current_score = avg_trend
    previous_score = mood_history['avg_trend_score'].iloc[-2] if len(mood_history) > 1 else current_score
    
    # Mood Alert Widget
    mood_cols = st.columns([1, 2])
    
    with mood_cols[0]:
        # Determine mood zone
        if current_score < 40:
            zone_color = "#ff6b6b"
            zone_text = "🔴 BEARISH ENTRY"
            zone_desc = "Buy Signal - Accumulate"
        elif current_score > 65:
            zone_color = "#4ecdc4"
            zone_text = "🟡 CAUTION"
            zone_desc = "High Mood - Be Selective"
        else:
            zone_color = "#69db7c"
            zone_text = "🟢 NEUTRAL"
            zone_desc = "Hold Current Positions"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {zone_color}22 0%, {zone_color}44 100%);
                    border: 2px solid {zone_color}; border-radius: 12px; padding: 15px; text-align: center;">
            <div style="font-size: 1.2em; font-weight: bold; color: {zone_color};">{zone_text}</div>
            <div style="font-size: 2em; font-weight: bold; color: white;">{current_score:.0f}</div>
            <div style="font-size: 0.9em; color: rgba(255,255,255,0.7);">{zone_desc}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with mood_cols[1]:
        # Score change alert
        score_change = current_score - previous_score
        if current_score < 40 and previous_score >= 40:
            st.warning("🚨 **ALERT**: Score dropped below 40 - Potential BUY signal for Midcap/Bank!")
        elif current_score > 65 and previous_score <= 65:
            st.info("📢 **ALERT**: Score crossed above 65 - Consider IT sector, reduce Midcap exposure")
        else:
            st.info("📊 Stable market mood. Maintain current strategy.")
    
    st.markdown("---")

    
    # === TOP MOVERS TICKER TAPE ===
    top_movers = df.nlargest(8, 'trend_score')[['ticker', 'trend_score', 'price']]
    ticker_html = " &nbsp;•&nbsp; ".join([
        f"<span style='color: #34C759; font-weight: 600;'>{row['ticker']}</span> <span style='color: #888;'>₹{row['price']:.0f}</span> <span style='background: rgba(52,199,89,0.2); padding: 2px 8px; border-radius: 10px; color: #34C759;'>{row['trend_score']}</span>"
        for _, row in top_movers.iterrows()
    ])
    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.03); padding: 12px 20px; border-radius: 8px; 
                overflow-x: auto; white-space: nowrap; border: 1px solid rgba(255,255,255,0.1);">
        <span style="color: #FFD700; margin-right: 10px;">🔥 TOP MOVERS:</span> {ticker_html}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # -- BREAKOUT ALERTS --
    breakouts = df[df['dist_52w'] > -2.0].copy()  # Within 2% of High
    if not breakouts.empty:
        with st.expander(f"🚨 **{len(breakouts)} BREAKOUT ALERTS** (Within 2% of 52W High)", expanded=False):
            # Sort by closest to high
            breakouts_sorted = breakouts.nsmallest(20, 'dist_52w').copy()
            breakouts_sorted['screener_link'] = "https://www.screener.in/company/" + breakouts_sorted['ticker'].str.replace('.NS', '', regex=False) + "/"
            st.dataframe(
                breakouts_sorted[['screener_link', 'name', 'price', 'dist_52w', 'trend_score', 'overall']],
                column_config={
                    "screener_link": st.column_config.LinkColumn("Ticker", display_text=r"https://www\.screener\.in/company/(.*?)/"),
                    "name": "Company", 
                    "price": st.column_config.NumberColumn("Price", format="₹%.2f"),
                    "dist_52w": st.column_config.NumberColumn("% from 52W High", format="%.1f%%"),
                    "trend_score": st.column_config.ProgressColumn("Trend", min_value=0, max_value=100),
                    "overall": st.column_config.NumberColumn("Score", format="%.1f"),
                },
                hide_index=True,
                height=300
            )
    
    # -- FILTERS --
    # -- FILTERS --
    with st.expander("⚡ Trend Filter", expanded=True):
        # Row 1: Search & Sector
        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            search_query = st.text_input("🔍 Search Stock", placeholder="e.g. RELIANCE, TATA...")
        with f_col2:
            # Safely get unique sectors
            df['sector'] = df['sector'].fillna("Unknown")
            all_sectors = sorted(df['sector'].astype(str).unique().tolist())
            sel_sector = st.multiselect("Sector (Empty = All)", all_sectors, default=[]) 
            
        # Row 2: Trend & Signal
        f_col3, f_col4 = st.columns(2)
        with f_col3:
            min_score = st.slider("Min Trend Score", 0, 100, 0)
        with f_col4:
            # Signal filter - match exact values from scoring.py
            signal_options = ["STRONG UPTREND", "UPTREND", "NEUTRAL", "DOWNTREND", "STRONG DOWNTREND"]
            sig_filter = st.multiselect("Signal", signal_options, default=signal_options)
        
        # Fundamental Filters
        st.markdown("---")
        st.markdown("**🔬 Fundamental Quality Filters**")
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            min_quality = st.slider("Min Quality Score", 0, 10, 0, help="Filter for high ROE/Margins")
        with fc2:
            min_value = st.slider("Min Value Score", 0, 10, 0, help="Filter for low PE/PB")
        with fc3:
            min_growth = st.slider("Min Growth Score", 0, 10, 0, help="Filter for high earnings growth")
        with fc4:
            min_volume = st.slider("Min Volume Score", 0, 10, 0, help="Filter for Accumulation (>7)")
    
    # Apply Filters
    filtered_df = df.copy()
    
    # 1. Search Filter
    if search_query:
        query = search_query.lower()
        filtered_df = filtered_df[
            filtered_df['ticker'].str.lower().str.contains(query) | 
            filtered_df['name'].str.lower().str.contains(query)
        ]
        
    if sel_sector:
        filtered_df = filtered_df[filtered_df['sector'].isin(sel_sector)]
    filtered_df = filtered_df[filtered_df['trend_score'] >= min_score]
    if sig_filter:
        filtered_df = filtered_df[filtered_df['trend_signal'].isin(sig_filter)]
        
    # Apply Fundamental Filters
    if min_quality > 0:
        filtered_df = filtered_df[filtered_df['quality'] >= min_quality]
    if min_value > 0:
        filtered_df = filtered_df[filtered_df['value'] >= min_value]
    if min_growth > 0:
        filtered_df = filtered_df[filtered_df['growth'] >= min_growth]
    if min_volume > 0:
        filtered_df = filtered_df[filtered_df['volume_signal_score'] >= min_volume]
    
    # Apply Presets from Sidebar
    preset = st.session_state.get('preset', 'All Stocks')
    if preset == "🚀 Strong Momentum (Top 20%)":
        threshold = filtered_df['trend_score'].quantile(0.8)
        filtered_df = filtered_df[filtered_df['trend_score'] >= threshold]
        st.info(f"Preset: Showing top 20% by Trend Score (>= {threshold:.0f})")
    elif preset == "💎 Quality at Reasonable Price":
        filtered_df = filtered_df[(filtered_df['overall'] >= 6) & (filtered_df['value'] >= 6)]
        st.info("Preset: Quality (Score >= 6) + Reasonable Value (Value >= 6)")
    elif preset == "📈 Breakout Candidates":
        filtered_df = filtered_df[filtered_df['dist_52w'] > -5]
        st.info("Preset: Within 5% of 52-Week High")
    elif preset == "🔥 Turnaround Plays":
        filtered_df = filtered_df[(filtered_df['momentum'] >= 5) & (filtered_df['overall'] < 5)]
        st.info("Preset: Improving Momentum (>= 5) but Low Overall Score (< 5)")
    
    elif preset == "🧬 DNA-3 V2 Picks":
        # Apply DNA-3 Filter: RS > 2%, Vol > 30%, Price > MA50
        # data is now pre-calculated in fast_data_engine
        if 'dna_signal' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['dna_signal'] == 'BUY']
        else:
            st.warning("DNA-3 Metrics not found. Please click 'Force Refresh Data'.")
            
        st.info("Preset: DNA-3 V2 Filter (RS > 2% vs Nifty, Vol > 30%, Above MA50)")
    
    if filtered_df.empty:
        st.warning("No stocks found matching these filters.")
        if st.button("🔄 Force Refresh Data"):

            # Remove both caches to force full rebuild
            if os.path.exists("nifty500_cache.csv"): os.remove("nifty500_cache.csv")
            if os.path.exists("market_data.parquet"): os.remove("market_data.parquet")
            
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    _nifty_hdr, _nifty_dl = st.columns([3, 1])
    with _nifty_hdr:
        st.subheader(f"Found {len(filtered_df)} Momentum Stocks")
    
    # Ensure columns exist (handle legacy cache)
    for col in ['comp_rs', 'volatility', 'dna_signal']:
        if col not in filtered_df.columns:
            filtered_df[col] = None
    
    
    # === DYNAMIC COLUMN FILTER (USER REQUEST) - MOVED HERE ===
    with st.expander("🌪️ **Add Custom Column Filter**", expanded=False):
        c_col1, c_col2, c_col3 = st.columns([2, 1, 2])
        with c_col1:
            filter_col = st.selectbox("Filter Column", 
                ["RS Score (vs Nifty)", "Volatility", "Trend Score", "Distance from 52W High", "Price"],
                index=0
            )
        
        # Map friendly name to df column
        col_map = {
            "RS Score (vs Nifty)": "comp_rs",
            "Volatility": "volatility",
            "Trend Score": "trend_score",
            "Distance from 52W High": "dist_52w",
            "Price": "price"
        }
        target_col = col_map[filter_col]
        
        with c_col3:
            # Determine range for slider based on data (safely handle NaNs)
            try:
                min_val = float(filtered_df[target_col].min()) if not filtered_df.empty and target_col in filtered_df and filtered_df[target_col].notna().any() else 0.0
                max_val = float(filtered_df[target_col].max()) if not filtered_df.empty and target_col in filtered_df and filtered_df[target_col].notna().any() else 100.0
                if min_val == max_val: max_val += 1.0 # Avoid slider error
            except:
                min_val, max_val = 0.0, 100.0
            
            filter_val = st.slider(f"{filter_col} Range", min_val, max_val, (min_val, max_val))
            
        # Apply Filter (only if user moved the slider from defaults)
        if target_col in filtered_df.columns and (filter_val[0] > min_val or filter_val[1] < max_val):
            # Keep rows within range OR value is NaN (no data yet)
            filtered_df = filtered_df[
                ((filtered_df[target_col] >= filter_val[0]) & (filtered_df[target_col] <= filter_val[1])) | (filtered_df[target_col].isna())
            ]
            st.caption(f"Showing {len(filtered_df)} stocks with {filter_col} between {filter_val[0]:.1f} and {filter_val[1]:.1f}")

    _SIGNAL_EMOJI = {
        'STRONG UPTREND': '🟢',
        'UPTREND': '🔵',
        'NEUTRAL': '🟡',
        'DOWNTREND': '🟠',
        'STRONG DOWNTREND': '🔴',
    }
    filtered_df = filtered_df.copy()
    filtered_df['signal_display'] = filtered_df['trend_signal'].map(
        lambda s: f"{_SIGNAL_EMOJI.get(s, '')} {s}" if s else s
    )

    display_cols = ['screener_link', 'name', 'sector', 'price', 'signal_display', 'trend_score', 'comp_rs', 'volatility', 'dna_signal', 'dist_52w', 'dist_200dma']
    # Add 5-pillar fundamental columns + RS Score for user request
    display_cols.extend(['quality', 'value', 'growth', 'momentum', 'volume_signal_score'])

    filtered_df['screener_link'] = "https://www.screener.in/company/" + filtered_df['ticker'].str.replace('.NS', '', regex=False) + "/"

    st.dataframe(
        filtered_df[display_cols].sort_values(by='trend_score', ascending=False),
        column_config={
            "screener_link": st.column_config.LinkColumn("Ticker", display_text=r"https://www\.screener\.in/company/(.*?)/"),
            "trend_score": st.column_config.ProgressColumn("Trend Score", format="%d", min_value=0, max_value=100),
            "price": st.column_config.NumberColumn("Price", format="₹ %.2f"),
            "dist_52w": st.column_config.NumberColumn("% from 52W High", format="%.1f%%"),
            "dist_200dma": st.column_config.NumberColumn("% vs 200DMA", format="%.1f%%"),
            "signal_display": st.column_config.TextColumn("Signal"),
            "quality": st.column_config.ProgressColumn("Quality", min_value=0, max_value=10, format="%.1f"),
            "value": st.column_config.ProgressColumn("Value", min_value=0, max_value=10, format="%.1f"),
            "growth": st.column_config.ProgressColumn("Growth", min_value=0, max_value=10, format="%.1f"),
            "momentum": st.column_config.ProgressColumn("Momentum", min_value=0, max_value=10, format="%.1f"),
            "volume_signal_score": st.column_config.ProgressColumn("Volume", min_value=0, max_value=10, format="%.1f"),
            "comp_rs": st.column_config.NumberColumn("RS vs Nifty", format="%+.1f%%", help="Composite Relative Strength vs Nifty (1W+1M+3M)"),
            "volatility": st.column_config.NumberColumn("Volatility", format="%.0f%%", help="Annualized Price Volatility"),
            "dna_signal": st.column_config.TextColumn("DNA Signal", help="BUY = All DNA-3 filters pass"),
        },
        height=500,
        use_container_width=True,
        hide_index=True
    )
    with _nifty_dl:
        st.markdown("<div style='padding-top:10px'></div>", unsafe_allow_html=True)
        _dl_cols = [c for c in display_cols if c != 'screener_link']
        _export = filtered_df[_dl_cols].sort_values('trend_score', ascending=False).copy()
        _export['signal'] = _export.get('trend_signal', _export.get('signal_display', ''))
        st.download_button(
            label="⬇️ Download CSV",
            data=_export.to_csv(index=False),
            file_name=f"nifty_momentum_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="nifty_scanner_download",
        )

    # Show filters active message
    active_filters = []
    if min_quality > 0: active_filters.append(f"Quality > {min_quality}")
    if min_value > 0: active_filters.append(f"Value > {min_value}")
    if min_growth > 0: active_filters.append(f"Growth > {min_growth}")
    
    if active_filters:
        st.caption(f"ℹ️ Active Fundamental Filters: {', '.join(active_filters)}")

    # === QUICK DIVE SECTION ===
    st.markdown("---")
    st.markdown("### 🔍 Quick Deep Dive")
    
    dive_col1, dive_col2 = st.columns([3, 1])
    with dive_col1:
        if not filtered_df.empty:
            selected_for_dive = st.selectbox(
                "Select a stock to analyze",
                options=filtered_df['ticker'].tolist(),
                format_func=lambda x: f"{x} - {filtered_df[filtered_df['ticker']==x]['name'].values[0]}" if len(filtered_df[filtered_df['ticker']==x]['name'].values) > 0 else x
            )
        else:
            selected_for_dive = None
            st.info("No stocks in current filter to analyze")
    
    with dive_col2:
        if selected_for_dive:
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                if st.button("📉 Report", type="primary", use_container_width=True):
                    st.session_state['quick_dive_ticker'] = selected_for_dive
                    st.session_state['nav_page'] = "📉 Deep Dive"  # Signal to switch
                    st.rerun()  # Rerun to trigger navigation
            with btn_col2:
                is_in_watchlist = selected_for_dive in st.session_state.get('watchlist', [])
                if is_in_watchlist:
                    if st.button("⭐ Remove", use_container_width=True):
                        st.session_state['watchlist'].remove(selected_for_dive)
                        st.rerun()
                else:
                    if st.button("➕ Watch", use_container_width=True):
                        add_to_watchlist(selected_for_dive)
                        st.toast(f"Added {selected_for_dive} to Watchlist!", icon="⭐")
                        st.rerun()
            with btn_col3:
                # Track button for Position Manager
                stock_row = filtered_df[filtered_df['ticker'] == selected_for_dive].iloc[0]
                if is_position_exists(selected_for_dive):
                    if st.button("📊 Untrack", use_container_width=True):
                        remove_position(selected_for_dive)
                        st.toast(f"Removed {selected_for_dive} from tracking!", icon="📊")
                        st.rerun()
                else:
                    if st.button("📊 Track + Alert", use_container_width=True, help="Add to tracking with Stop Loss & Target"):
                        add_position(
                            selected_for_dive,
                            name=stock_row.get('name', selected_for_dive),
                            sector=stock_row.get('sector', 'Unknown'),
                            entry_price=stock_row.get('price', 0),
                            entry_signal=stock_row.get('trend_signal', 'N/A'),
                            entry_score=stock_row.get('trend_score', 0)
                        )
                        # Prefill alert ticker and redirect to Position Manager
                        st.session_state['prefill_alert_ticker'] = selected_for_dive
                        st.session_state['nav_page'] = "📊 Return Tracker"
                        st.toast(f"Tracking {selected_for_dive}! Set your Stop Loss & Target in Alerts tab.", icon="📊")
                        st.rerun()

# --- VIEW: RETURN TRACKER ---
elif page == "📊 Return Tracker":
    
    # Show toast for recently deleted items (after rerun)
    if 'position_removed' in st.session_state:
        st.toast(f"✅ Removed {st.session_state['position_removed']}!", icon="🗑️")
        del st.session_state['position_removed']
    if 'alert_deleted' in st.session_state:
        st.toast(f"✅ Alert deleted for {st.session_state['alert_deleted']}!", icon="🗑️")
        del st.session_state['alert_deleted']
    
    st.markdown(page_header("📊 Position Manager", "Unified tracking: Positions • Alerts • Notes | Your complete portfolio command center"), unsafe_allow_html=True)
    
    # Auto-migrate from legacy systems (runs once if positions.json is empty)
    if not os.path.exists('positions.json'):
        migrated = migrate_from_legacy()
        if migrated > 0:
            st.toast(f"Migrated {migrated} positions from legacy system!", icon="🔄")
    
    # Get positions with P&L
    positions = get_positions_with_pnl(df)
    active_positions = [p for p in positions if p.get('status') == 'active']
    watching_positions = [p for p in positions if p.get('status') == 'watching']
    triggered_alerts = check_position_alerts(df)
    summary = get_summary(df)
    
    # === SUMMARY METRICS (3 cols x 2 rows for readability) ===
    row1_c1, row1_c2, row1_c3 = st.columns(3)
    row1_c1.metric("📈 Active", summary.get('total_active', 0))
    row1_c2.metric("👀 Watching", summary.get('total_watching', 0))
    avg_return = summary.get('avg_pnl_pct', 0)
    return_color = "normal" if avg_return >= 0 else "inverse"
    row1_c3.metric("💰 Avg Return", f"{avg_return:+.1f}%", delta_color=return_color)
    
    row2_c1, row2_c2, row2_c3 = st.columns(3)
    row2_c1.metric("✅ Win/Loss", f"{summary.get('winners', 0)}/{summary.get('losers', 0)}")
    row2_c2.metric("🎯 Win Rate", f"{summary.get('win_rate', 0):.0f}%")
    row2_c3.metric("⚠️ Triggered", len(triggered_alerts))
    
    st.markdown("---")
    
    # === MAIN TABS (Simplified) ===
    tab_active, tab_watchlist, tab_add, tab_settings = st.tabs(["📋 Active Positions", "👀 Watchlist", "➕ Add New", "⚙️ Settings"])
    
    # ========================================
    # TAB 1: ACTIVE POSITIONS (Using unified positions.json)
    # ========================================
    with tab_active:
        
        if active_positions:
            # Enhanced positions table with SL/Target
            st.markdown("### 📈 Your Positions")
            
            # Build portfolio data from positions.json
            portfolio_data = []
            for pos in active_positions:
                ticker = pos.get('ticker', '')
                entry_price = pos.get('entry_price') or 0
                current_price = pos.get('current_price') or 0
                pnl_pct = pos.get('pnl_pct') or 0
                stop_loss = pos.get('stop_loss')
                target = pos.get('target')
                
                # Calculate distances
                dist_to_sl = ((current_price - stop_loss) / current_price * 100) if stop_loss and current_price else None
                dist_to_target = ((target - current_price) / current_price * 100) if target and current_price else None
                
                # Determine status
                status = "✅"
                if stop_loss and current_price and current_price <= stop_loss:
                    status = "🔴 SL HIT"
                elif target and current_price and current_price >= target:
                    status = "🎯 TARGET"
                elif dist_to_sl and dist_to_sl < 5:
                    status = "⚠️ Near SL"
                
                # Calculate days
                entry_date = pos.get('entry_date')
                days = 0
                if entry_date:
                    try:
                        entry_dt = datetime.fromisoformat(entry_date) if isinstance(entry_date, str) else entry_date
                        days = (datetime.now() - entry_dt).days
                    except:
                        pass
                
                portfolio_data.append({
                    'Ticker': ticker.replace('.NS', '').replace('.BO', ''),
                    'Entry': entry_price,
                    'Current': current_price,
                    'P&L %': pnl_pct,
                    'Days': days,
                    'Stop Loss': stop_loss,
                    'Target': target,
                    'SL Dist': f"{dist_to_sl:.1f}%" if dist_to_sl else "—",
                    'Status': status,
                    'ticker_full': ticker
                })
            
            portfolio_df = pd.DataFrame(portfolio_data)
            
            st.dataframe(
                portfolio_df[['Ticker', 'Entry', 'Current', 'P&L %', 'Days', 'Stop Loss', 'Target', 'SL Dist', 'Status']],
                column_config={
                    "Ticker": st.column_config.TextColumn("Stock"),
                    "Entry": st.column_config.NumberColumn("Entry ₹", format="%.2f"),
                    "Current": st.column_config.NumberColumn("Current ₹", format="%.2f"),
                    "P&L %": st.column_config.NumberColumn("Return", format="%+.1f%%"),
                    "Days": st.column_config.NumberColumn("Days", format="%d"),
                    "Stop Loss": st.column_config.NumberColumn("SL ₹", format="%.0f"),
                    "Target": st.column_config.NumberColumn("Target ₹", format="%.0f"),
                    "SL Dist": st.column_config.TextColumn("To SL"),
                    "Status": st.column_config.TextColumn("Status"),
                },
                hide_index=True,
                use_container_width=True
            )
            
            # === MANAGE POSITIONS ===
            st.markdown("---")
            with st.expander("🔧 Manage Positions"):
                mgmt_col1, mgmt_col2, mgmt_col3 = st.columns([2, 1, 1])
                
                with mgmt_col1:
                    ticker_options = [p.get('ticker', '') for p in active_positions]
                    selected_ticker = st.selectbox(
                        "Select position", 
                        ticker_options,
                        format_func=lambda x: x.replace('.NS', '').replace('.BO', ''),
                        key="pos_select"
                    )
                
                with mgmt_col2:
                    if st.button("❌ Remove Position", use_container_width=True, key="btn_remove_pos"):
                        if selected_ticker:
                            remove_position(selected_ticker)
                            st.session_state['position_removed'] = selected_ticker
                            st.rerun()
                
                with mgmt_col3:
                    if st.button("🔔 Add Alert", use_container_width=True):
                        st.session_state['prefill_alert_ticker'] = selected_ticker
            
            # === TREND & ALERT CHANGES (Unified) ===
            # Detect positions where trend signal changed from entry
            trend_changed_positions = [
                p for p in active_positions 
                if p.get('entry_signal') and p.get('current_signal') 
                and p.get('entry_signal') != p.get('current_signal', '')
                and p.get('entry_signal') != 'N/A'
            ]
            
            if trend_changed_positions:
                st.markdown("---")
                st.markdown("### ⚠️ Trend Changes")
                
                for pos in trend_changed_positions:
                    entry_signal = pos.get('entry_signal', 'N/A')
                    current_signal = pos.get('current_signal', 'N/A')
                    pnl = pos.get('pnl_pct', 0) or 0
                    
                    if 'UPTREND' in str(entry_signal) and 'DOWNTREND' in str(current_signal):
                        alert_color = COLORS['negative']
                    elif 'DOWNTREND' in str(entry_signal) and 'UPTREND' in str(current_signal):
                        alert_color = COLORS['positive']
                    else:
                        alert_color = COLORS['warning']
                    
                    ticker_display = pos.get('ticker', '').replace('.NS', '').replace('.BO', '')
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px; 
                                margin-bottom: 8px; border-left: 4px solid {alert_color};">
                        <strong>{ticker_display}</strong> 
                        <span style="color: #888;">|</span> 
                        {entry_signal} → <span style="color: {alert_color};">{current_signal}</span>
                        <span style="color: #888;">|</span>
                        <span style="color: {COLORS['positive'] if pnl >= 0 else COLORS['negative']};">{pnl:+.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # === SL/TARGET ALERTS PANEL ===
            if triggered_alerts:
                st.markdown("---")
                st.markdown("### 🚨 Triggered Alerts")
                for ta in triggered_alerts:
                    ta_ticker = ta.get('ticker', '').replace('.NS', '').replace('.BO', '')
                    ta_type = ta.get('alert_type', 'ALERT')
                    ta_price = ta.get('current_price', 0)
                    ta_pnl = ta.get('pnl_pct', 0) or 0
                    
                    if ta_type == 'STOP_LOSS':
                        ta_color = COLORS['negative']
                        ta_icon = "🔴"
                        ta_label = f"SL HIT at ₹{ta_price:.2f}"
                    else:
                        ta_color = COLORS['positive']
                        ta_icon = "🎯"
                        ta_label = f"TARGET HIT at ₹{ta_price:.2f}"
                    
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.03); padding: 14px; border-radius: 10px;
                                margin-bottom: 8px; border-left: 4px solid {ta_color};">
                        <span style="font-size: 16px;">{ta_icon}</span>
                        <strong style="font-size: 15px;">{ta_ticker}</strong>
                        <span style="color: {ta_color}; font-weight: bold;"> {ta_label}</span>
                        <span style="color: #888;"> | P&L: </span>
                        <span style="color: {COLORS['positive'] if ta_pnl >= 0 else COLORS['negative']};">{ta_pnl:+.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
        else:
            st.info("📭 No positions yet. Add stocks from **Trend Scanner** → **📊 Track** button")
            
            # Quick add section
            st.markdown("---")
            st.markdown("### 🚀 Quick Add: Top Stocks")
            
            top_stocks = df.nlargest(6, 'trend_score')[['ticker', 'name', 'price', 'trend_score', 'trend_signal', 'sector']]
            
            cols = st.columns(3)
            for idx, (_, stock) in enumerate(top_stocks.iterrows()):
                with cols[idx % 3]:
                    ticker = stock['ticker']
                    if st.button(
                        f"📊 {ticker.replace('.NS', '')} ({stock['trend_score']:.0f})",
                        key=f"quick_track_{ticker}",
                        use_container_width=True
                    ):
                        add_position(
                            ticker, name=stock['name'], sector=stock['sector'],
                            entry_price=stock['price'], entry_signal=stock['trend_signal'],
                            entry_score=stock['trend_score']
                        )
                        st.toast(f"Now tracking {ticker}! Refresh to see changes.", icon="📊")
    
    # ========================================
    # TAB 2: WATCHLIST
    # ========================================
    with tab_watchlist:
        
        st.markdown("### 👀 Stocks to Watch")
        
        # Show triggered alerts prominently
        if triggered_alerts:
            st.error(f"⚡ {len(triggered_alerts)} alerts triggered!")
            for alert in triggered_alerts:
                trigger_type = alert.get('trigger_type', 'METRIC')
                icon = "🔴" if trigger_type == "STOP_LOSS" else "🎯" if trigger_type == "TARGET" else "📊"
                st.warning(f"{icon} **{alert['ticker'].replace('.NS', '')}**: {alert.get('alert_message', alert.get('alert_type', 'Alert triggered'))}")
            st.markdown("---")
        
        # Two columns: Active Alerts + Add New
        alert_col1, alert_col2 = st.columns([2, 1])
        
        with alert_col1:
            st.markdown("#### Active Alerts")
            
            # Fetch alerts using the utility function from alerts.py
            from utils.alerts import get_alerts_with_pnl
            all_alerts = get_alerts_with_pnl(df)
            
            if all_alerts:
                for alert in all_alerts:
                    ticker = alert.get('ticker', '').replace('.NS', '')
                    
                    # Build status line
                    parts = [f"**{ticker}**"]
                    
                    metric = alert.get('metric', '')
                    if metric and metric != 'price':
                        parts.append(f"{metric.title()} {alert.get('condition', '')} {alert.get('threshold', '')}")
                    
                    if alert.get('stop_loss'):
                        parts.append(f"SL: ₹{alert['stop_loss']:.0f}")
                    if alert.get('target'):
                        parts.append(f"T: ₹{alert['target']:.0f}")
                    
                    # Current P&L if available
                    pnl = alert.get('pnl_pct')
                    if pnl is not None:
                        pnl_color = "green" if pnl >= 0 else "red"
                        parts.append(f":{pnl_color}[{pnl:+.1f}%]")
                    
                    col_info, col_del = st.columns([5, 1])
                    with col_info:
                        st.write(" | ".join(parts))
                        if alert.get('notes'):
                            st.caption(f"📝 {alert['notes'][:80]}...")
                    with col_del:
                        if st.button("🗑️", key=f"del_alert_{alert.get('id')}", help="Delete this alert"):
                            remove_alert(alert.get('id'))
                            st.session_state['alert_deleted'] = alert.get('ticker', '')
                            st.rerun()
                    st.markdown("<hr style='margin: 5px 0; opacity: 0.2;'>", unsafe_allow_html=True)
            else:
                st.info("No active alerts. Create one →")
        
        with alert_col2:
            st.markdown("#### ➕ New Alert")
            
            # Use form to prevent reload on each input change
            with st.form("new_alert_form", clear_on_submit=True):
                # Prefill ticker if coming from position management
                prefill = st.session_state.get('prefill_alert_ticker', '')
                if prefill:
                    del st.session_state['prefill_alert_ticker']
                
                new_ticker = st.text_input("Ticker", value=prefill.replace('.NS', ''), placeholder="HDFCBANK")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    new_entry = st.number_input("Entry ₹", value=0.0, step=10.0, format="%.2f")
                    new_sl = st.number_input("Stop Loss ₹", value=0.0, step=10.0, format="%.2f")
                with col_b:
                    new_target = st.number_input("Target ₹", value=0.0, step=10.0, format="%.2f")
                    
                new_notes = st.text_area("Notes", placeholder="Investment thesis...", height=80)
                
                submitted = st.form_submit_button("✅ Create Alert", use_container_width=True, type="primary")
                
                if submitted and new_ticker:
                    ticker = new_ticker.upper()
                    if not ticker.endswith(".NS"):
                        ticker += ".NS"
                    add_price_alert(
                        ticker,
                        entry_price=new_entry if new_entry > 0 else None,
                        stop_loss=new_sl if new_sl > 0 else None,
                        target=new_target if new_target > 0 else None,
                        notes=new_notes
                    )
                    st.toast(f"Alert created for {ticker}!", icon="✅")
    
    # ========================================
    # TAB 3: ADD NEW POSITION
    # ========================================
    with tab_add:
        
        st.markdown("### ➕ Add New Position")
        st.caption("Add a stock to your portfolio or watchlist")
        
        # Use unified form
        with st.form("add_position_form", clear_on_submit=True):
            add_col1, add_col2 = st.columns(2)
            
            # Build ticker options from full NIFTY 500 list (not just loaded data)
            from utils.nifty500_list import TICKERS
            ticker_options = [""] + sorted(TICKERS)
            # Build display names from df where available
            ticker_display = {}
            for t in ticker_options:
                if t and not df[df['ticker']==t].empty:
                    name = df[df['ticker']==t]['name'].values[0]
                    ticker_display[t] = f"{t.replace('.NS', '')} - {name}"
                elif t:
                    ticker_display[t] = t.replace('.NS', '')
            
            with add_col1:
                new_ticker = st.selectbox(
                    "Ticker (type to search)", 
                    options=ticker_options,
                    format_func=lambda x: ticker_display.get(x, x) if x else "Select a stock...",
                    index=0
                )
                new_entry = st.number_input("Entry Price ₹", value=0.0, step=10.0, format="%.2f")
                new_quantity = st.number_input("Quantity", value=1, min_value=1, step=1)
                
            with add_col2:
                new_status = st.selectbox("Status", ["active", "watching"], format_func=lambda x: "📈 Active Position" if x == "active" else "👀 Watchlist")
                new_sl = st.number_input("Stop Loss ₹", value=0.0, step=10.0, format="%.2f")
                new_target = st.number_input("Target ₹", value=0.0, step=10.0, format="%.2f")
            
            new_notes = st.text_area("Notes / Investment Thesis", placeholder="Why are you buying this? Key catalysts, risks...", height=100)
            
            submitted = st.form_submit_button("✅ Add Position", use_container_width=True, type="primary")
            
            if submitted and new_ticker:
                ticker = new_ticker.upper()
                if not ticker.endswith(".NS"):
                    ticker += ".NS"
                
                # Get stock info from market data
                stock_row = df[df['ticker'] == ticker]
                stock_name = stock_row.iloc[0].get('name', ticker) if not stock_row.empty else ticker.replace('.NS', '')
                stock_sector = stock_row.iloc[0].get('sector', 'Unknown') if not stock_row.empty else 'Unknown'
                stock_signal = stock_row.iloc[0].get('trend_signal', 'N/A') if not stock_row.empty else 'N/A'
                stock_score = stock_row.iloc[0].get('trend_score', 0) if not stock_row.empty else 0
                
                add_position(
                    ticker=ticker,
                    name=stock_name,
                    sector=stock_sector,
                    status=new_status,
                    entry_price=new_entry if new_entry > 0 else None,
                    stop_loss=new_sl if new_sl > 0 else None,
                    target=new_target if new_target > 0 else None,
                    quantity=new_quantity,
                    notes=new_notes,
                    entry_signal=stock_signal,
                    entry_score=stock_score
                )
                st.toast(f"{'Position' if new_status == 'active' else 'Watchlist item'} added for {ticker}! Refresh to see.", icon="✅")
        
        # Quick add from top stocks
        st.markdown("---")
        st.markdown("### 🔥 Quick Add: Top Trending Stocks")
        top_stocks = df.nlargest(6, 'trend_score')[['ticker', 'name', 'price', 'trend_score', 'trend_signal', 'sector']]
        
        cols = st.columns(3)
        for idx, (_, stock) in enumerate(top_stocks.iterrows()):
            with cols[idx % 3]:
                ticker = stock['ticker']
                is_tracked = is_position_exists(ticker)
                
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px; margin-bottom: 8px;">
                    <strong>{stock['name'][:20]}</strong><br>
                    <span style="color: #888;">₹{stock['price']:.0f}</span> • 
                    <span style="color: #00C853;">Score: {stock['trend_score']:.0f}</span>
                </div>
                """, unsafe_allow_html=True)
                
                if is_tracked:
                    st.success("✅ Tracking")
                else:
                    if st.button(f"➕ Watch", key=f"quick_add_{ticker}", use_container_width=True):
                        add_position(
                            ticker=ticker,
                            name=stock['name'],
                            sector=stock['sector'],
                            status='watching',
                            entry_signal=stock['trend_signal'],
                            entry_score=stock['trend_score']
                        )
                        st.toast(f"Added {ticker} to watchlist! Refresh to see.", icon="✅")
    
    # ========================================
    # TAB 4: SETTINGS
    # ========================================
    with tab_settings:
        st.markdown("### 📬 Notification Settings")
        
        # Import telegram notifier
        from utils.telegram_notifier import (
            configure_telegram, is_telegram_configured, test_telegram,
            send_triggered_alerts, configure_auto_alert, is_auto_alert_enabled,
            run_auto_alert_check, send_daily_summary as tg_send_summary
        )
        
        notif_tab1, notif_tab2, notif_tab3 = st.tabs(["📧 Email", "📱 Telegram", "⚡ Auto-Alert"])
        
        # === EMAIL TAB ===
        with notif_tab1:
            email_col1, email_col2 = st.columns([2, 1])
            
            with email_col1:
                if is_email_configured():
                    st.success(f"✅ Email: {get_email_address()}")
                else:
                    st.warning("Email not configured")
                    gmail_input = st.text_input("Gmail Address", placeholder="your.email@gmail.com", key="notif_gmail")
                    app_pass = st.text_input("App Password", type="password", key="notif_app_pass")
                    if st.button("💾 Save Email Config", key="save_email_btn"):
                        if gmail_input and app_pass:
                            configure_email(gmail_input, app_pass)
                            st.toast("Email configured! Refresh to see changes.", icon="✅")
            
            with email_col2:
                if is_email_configured():
                    if st.button("📤 Send Report", use_container_width=True, key="send_email_report"):
                        with st.spinner("Generating email report..."):
                            summary_data = export_weekly_summary(df)
                            success, msg = send_weekly_summary(summary_data)
                            if success:
                                st.toast("✅ Email report sent successfully!", icon="📧")
                            else:
                                st.toast(f"❌ Failed to send email: {msg}", icon="⚠️")
        
        # === TELEGRAM TAB ===
        with notif_tab2:
            st.markdown("**Setup Instructions:**")
            st.caption("""
            1. Search @BotFather on Telegram
            2. Send /newbot and follow instructions
            3. Copy the bot token
            4. Start chat with your bot and send /start
            5. Get your chat_id from @userinfobot
            """)
            
            tg_col1, tg_col2 = st.columns([2, 1])
            
            with tg_col1:
                if is_telegram_configured():
                    st.success("✅ Telegram configured")
                else:
                    with st.form("telegram_config_form"):
                        st.write("Enter credentials below:")
                        tg_token = st.text_input("Bot Token", type="password", 
                                                placeholder="123456:ABC...", key="form_tg_token")
                        tg_chat_id = st.text_input("Chat ID",
                                                  placeholder="123456789", key="form_tg_chat_id")
                        
                        submitted = st.form_submit_button("💾 Save Telegram Config", use_container_width=True, type="primary")
                        
                        if submitted:
                            if tg_token and tg_chat_id:
                                try:
                                    configure_telegram(tg_token, tg_chat_id)
                                    st.toast("Telegram configured! Refresh to see changes.", icon="✅")
                                except Exception as e:
                                    st.error(f"Error saving config: {e}")
                            else:
                                st.error("Please enter both Bot Token and Chat ID")
            
            with tg_col2:
                if is_telegram_configured():
                    if st.button("📱 Test Telegram", use_container_width=True, key="test_tg"):
                        with st.spinner("Sending..."):
                            success, msg = test_telegram()
                            if success:
                                st.toast("✅ Test message sent to your phone!", icon="📱")
                            else:
                                st.toast(f"❌ Failed: {msg}", icon="⚠️")
                    
                    # Weekly summary button enabled unconditionally
                    if st.button("📊 Send Summary", use_container_width=True, key="send_tg_summary"):
                            with st.spinner("Generating summary..."):
                                summary_data = export_weekly_summary(df)
                                success, msg = tg_send_summary(summary_data)
                                if success:
                                    st.toast("✅ Portfolio summary sent to Telegram!", icon="📊")
                                else:
                                    st.toast(f"❌ Failed: {msg}", icon="⚠️")
        
        # === AUTO-ALERT TAB ===
        with notif_tab3:
            st.markdown("**Automatic Alert Notifications**")
            st.caption("Automatically send notifications when alerts are triggered")
            
            auto_enabled = is_auto_alert_enabled()
            
            aa_col1, aa_col2 = st.columns([2, 1])
            
            with aa_col1:
                new_auto_state = st.toggle("Enable Auto-Alerts", value=auto_enabled, key="auto_alert_toggle")
                
                if new_auto_state != auto_enabled:
                    configure_auto_alert(new_auto_state)
                    st.toast(f"Auto-alerts {'enabled' if new_auto_state else 'disabled'}!", icon="🔔")
                
                if auto_enabled:
                    st.info("🔔 Alerts will be sent automatically when triggered")
                    st.caption("• Checks run on dashboard load (max once per 30 min)")
                    st.caption("• Uses Telegram if configured, otherwise Email")
            
            with aa_col2:
                if triggered_alerts:
                    st.warning(f"⚡ {len(triggered_alerts)} alerts triggered!")
                    if st.button("📤 Send Now", use_container_width=True, key="send_triggered"):
                        results = send_triggered_alerts(triggered_alerts)
                        st.success(f"Sent: TG={results.get('telegram', 0)}, Email={results.get('email', 0)}")


# --- VIEW: PORTFOLIO BACKTEST ---
elif page == "📈 Portfolio Backtest":
    
    from utils.portfolio_backtest import (
        run_backtest, get_current_portfolio_from_scores, BACKTEST_CONFIG
    )
    
    st.markdown(page_header("📈 Portfolio Backtest", "Validate trend scores with real historical returns | 6-month backtest • Bi-weekly rebalancing • ₹2L capital"), unsafe_allow_html=True)
    
    # === CONFIG DISPLAY ===
    with st.expander("⚙️ Backtest Configuration", expanded=False):
        cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)
        cfg_col1.metric("Initial Capital", f"₹{BACKTEST_CONFIG['initial_capital']:,}")
        cfg_col2.metric("Portfolio Size", f"{BACKTEST_CONFIG['portfolio_size']} stocks")
        cfg_col3.metric("Rebalance", f"Every {BACKTEST_CONFIG['rebalance_freq_days']} days")
        cfg_col4.metric("Lookback", f"{BACKTEST_CONFIG['lookback_months']} months")
        
        st.markdown("**Entry Criteria:** Trend Score ∈ [70, 90] (Hot Zone), Price > 200 DMA")
        st.markdown("**Exit Criteria:** Trend Score < 40, Stop Loss -15%, Trailing Stop -8%")
    
    st.markdown("---")
    
    # === CURRENT RECOMMENDED PORTFOLIO ===
    st.markdown("### 💼 Current Recommended Portfolio")
    st.caption("These are the top 20 stocks by trend score that meet ALL entry criteria TODAY")
    
    current_portfolio = get_current_portfolio_from_scores(df)
    
    if current_portfolio.empty:
        st.warning("⚠️ No stocks currently meet all entry criteria (Trend 70-90, Price > 200 DMA)")
    else:
        # Summary row
        port_col1, port_col2, port_col3 = st.columns(3)
        port_col1.metric("Eligible Stocks", len(current_portfolio))
        port_col2.metric("Avg Trend Score", f"{current_portfolio['trend_score'].mean():.0f}")
        port_col3.metric("Avg Overall Score", f"{current_portfolio['overall'].mean():.1f}")
        
        # Portfolio table
        st.dataframe(
            current_portfolio,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker"),
                "name": st.column_config.TextColumn("Company"),
                "sector": st.column_config.TextColumn("Sector"),
                "price": st.column_config.NumberColumn("Price", format="₹%.2f"),
                "trend_score": st.column_config.ProgressColumn("Trend Score", min_value=0, max_value=100),
                "trend_signal": st.column_config.TextColumn("Signal"),
                "overall": st.column_config.NumberColumn("Overall", format="%.1f"),
                "comp_rs": st.column_config.NumberColumn("OptComp RS", format="%.2f", help="Composite RS vs Nifty (1W/1M/3M)"),
                "target_allocation": st.column_config.NumberColumn("Allocation ₹", format="%.0f"),
                "target_shares": st.column_config.NumberColumn("Target Shares", format="%d"),
            },
            hide_index=True,
            use_container_width=True,
            height=400
        )
        
        # Sector breakdown
        st.markdown("#### 🏭 Sector Allocation")
        sector_counts = current_portfolio['sector'].value_counts()
        fig_sector = px.pie(
            values=sector_counts.values, 
            names=sector_counts.index,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_sector.update_layout(
            template='plotly_white',
            paper_bgcolor='rgba(0,0,0,0)',
            height=300
        )
        st.plotly_chart(fig_sector, use_container_width=True)
    
    st.markdown("---")
    
    # === RUN BACKTEST ===
    st.markdown("### 📊 Historical Backtest Results")
    
    # Check if backtest already run
    if 'backtest_results' not in st.session_state:
        if st.button("🚀 Run 6-Month Backtest", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status = st.empty()
            
            def update_progress(pct):
                progress_bar.progress(pct)
                status.text(f"Backtesting... {pct*100:.0f}%")
            
            with st.spinner("Running historical backtest... This may take a minute."):
                results = run_backtest(df, progress_callback=update_progress)
                st.session_state['backtest_results'] = results
            
            progress_bar.empty()
            status.empty()
            st.rerun()
        else:
            st.info("👆 Click the button above to run the historical backtest and see how the trend-score strategy would have performed.")
    else:
        results = st.session_state['backtest_results']
        
        if 'error' in results:
            st.error(f"Backtest Error: {results['error']}")
        else:
            metrics = results['metrics']
            equity_df = results['equity_curve']
            trades = results['trades']
            metrics = results['metrics']
            equity_df = results['equity_curve']
            trades = results['trades']
            analytics = results.get('analytics', {})
            factor_analysis = analytics.get('factor_perf', {})
            heatmaps = analytics.get('heatmaps', {})
            exit_analysis = analytics.get('exit_analysis', pd.DataFrame())
            
            # === PERFORMANCE METRICS ===
            st.markdown("#### 📈 Performance Summary")
            
            m1, m2, m3, m4, m5 = st.columns(5)
            
            total_return = metrics['total_return_pct']
            return_color = "#00C853" if total_return > 0 else "#FF5252"
            
            m1.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 10px; text-align: center;">
                <div style="color: #888; font-size: 12px;">Total Return</div>
                <div style="color: {return_color}; font-size: 24px; font-weight: 700;">{total_return:+.1f}%</div>
                <div style="color: #888; font-size: 10px;">₹{metrics['final_value']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            alpha = metrics['alpha']
            alpha_color = "#00C853" if alpha > 0 else "#FF5252"
            
            m2.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 10px; text-align: center;">
                <div style="color: #888; font-size: 12px;">Alpha vs Benchmark</div>
                <div style="color: {alpha_color}; font-size: 24px; font-weight: 700;">{alpha:+.1f}%</div>
                <div style="color: #888; font-size: 10px;">Benchmark: {metrics['benchmark_return_pct']:+.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            m3.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 10px; text-align: center;">
                <div style="color: #888; font-size: 12px;">Sharpe Ratio</div>
                <div style="color: #667eea; font-size: 24px; font-weight: 700;">{metrics['sharpe_ratio']:.2f}</div>
                <div style="color: #888; font-size: 10px;">Risk-Adjusted</div>
            </div>
            """, unsafe_allow_html=True)
            
            dd_color = "#FF5252" if metrics['max_drawdown_pct'] < -10 else "#FF9800"
            
            m4.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 10px; text-align: center;">
                <div style="color: #888; font-size: 12px;">Max Drawdown</div>
                <div style="color: {dd_color}; font-size: 24px; font-weight: 700;">{metrics['max_drawdown_pct']:.1f}%</div>
                <div style="color: #888; font-size: 10px;">Worst Peak-to-Trough</div>
            </div>
            """, unsafe_allow_html=True)
            
            win_color = "#00C853" if metrics['win_rate_pct'] > 50 else "#FF9800"
            
            m5.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 10px; text-align: center;">
                <div style="color: #888; font-size: 12px;">Win Rate</div>
                <div style="color: {win_color}; font-size: 24px; font-weight: 700;">{metrics['win_rate_pct']:.0f}%</div>
                <div style="color: #888; font-size: 10px;">{metrics['total_trades']} trades</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # === EQUITY CURVE ===
            st.markdown("#### 📈 Equity Curve vs Benchmark")
            
            if not equity_df.empty:
                # Normalize to base 100 for comparison
                equity_df['portfolio_normalized'] = equity_df['equity'] / equity_df['equity'].iloc[0] * 100
                
                fig_equity = go.Figure()
                
                fig_equity.add_trace(go.Scatter(
                    x=equity_df['date'],
                    y=equity_df['portfolio_normalized'],
                    name='Trend Strategy',
                    line=dict(color='#667eea', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.1)'
                ))
                
                # Add benchmark line (base 100)
                benchmark_return = metrics['benchmark_return_pct']
                start_date = equity_df['date'].iloc[0]
                end_date = equity_df['date'].iloc[-1]
                days = len(equity_df)
                
                benchmark_line = pd.Series(
                    [100 + (i / days) * benchmark_return for i in range(days)],
                    index=equity_df['date']
                )
                
                fig_equity.add_trace(go.Scatter(
                    x=equity_df['date'],
                    y=benchmark_line,
                    name='Nifty 500 (approx)',
                    line=dict(color='#888888', width=2, dash='dash')
                ))
                
                fig_equity.update_layout(
                    template='plotly_white',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02),
                    yaxis_title='Portfolio Value (Base 100)',
                    xaxis_title='Date'
                )
                
                st.plotly_chart(fig_equity, use_container_width=True)
            
            # === FACTOR ANALYSIS ===
            st.markdown("---")
            st.markdown("### 🔬 Factor Analysis: Which Scores Predict Returns?")
            st.caption("Analyzing if higher entry trend scores correlate with better returns")
            
            if factor_analysis:
                # Isolate Trend Score performance for this specific section
                trend_data = factor_analysis.get('Trend Score', {})
                
                if trend_data:
                    # Sort buckets for display (descending ideal)
                    sorted_buckets = sorted(trend_data.items(), key=lambda x: x[0], reverse=True)
                    factor_cols = st.columns(len(sorted_buckets))
                    
                    for i, (bucket, data) in enumerate(sorted_buckets):
                        with factor_cols[i]:
                            avg_ret = data.get('avg_return', 0)
                            color = "#00C853" if avg_ret > 0 else "#FF5252"
                            
                            st.markdown(f"""
                            <div style="background: rgba(255,255,255,0.03); padding: 20px; border-radius: 12px; text-align: center;">
                                <div style="font-size: 18px; font-weight: 600;">Trend {bucket}</div>
                                <div style="color: {color}; font-size: 28px; font-weight: 700; margin: 10px 0;">{avg_ret:+.1f}%</div>
                                <div style="color: #888;">Win Rate: {data.get('win_rate', 0):.0f}%</div>
                                <div style="color: #666; font-size: 12px;">{data.get('count', 0)} trades</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Insight
                if trend_data:
                    buckets_sorted = sorted(trend_data.items(), key=lambda x: x[1].get('avg_return', 0), reverse=True)
                    best_bucket = buckets_sorted[0][0] if buckets_sorted else None
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    if best_bucket == '90-100':
                        st.success("✅ **Insight**: Highest trend scores (90-100) generate the best returns. The scoring engine has strong predictive power!")
                    elif best_bucket == '80-89':
                        st.info("📊 **Insight**: The 80-89 range performs best. Consider raising the entry threshold from 70 to 80 for better results.")
                    elif best_bucket == '70-79':
                        st.warning("⚠️ **Insight**: Lower scores (70-79) outperform higher ones. The trend score may be missing some predictive factors - consider analyzing individual score components (quality, value, etc.)")
            else:
                st.info("Not enough completed trades to analyze factor performance yet.")

            # === HEATMAPS ===
            if heatmaps:
                st.markdown("#### 🔥 Performance Heatmaps")
                st.caption("How Strategy Returns vary by Trend Score vs Fundamental Factors")
                
                # Show tabs for different factors
                hm_tabs = st.tabs(list(heatmaps.keys()))
                
                for i, (factor_name, hm_data) in enumerate(heatmaps.items()):
                    with hm_tabs[i]:
                        if not hm_data.empty:
                            fig_hm = go.Figure(data=go.Heatmap(
                                z=hm_data.values,
                                x=hm_data.columns,
                                y=hm_data.index,
                                colorscale='RdYlGn',
                                texttemplate="%{z:.1f}%",
                                textfont={"size": 12},
                                hoverongaps=False
                            ))
                            fig_hm.update_layout(
                                title=f"Trend vs {factor_name} (Avg Return %)",
                                height=350,
                                template='plotly_white',
                                xaxis_title='Trend Bucket',
                                yaxis_title=f'{factor_name} Bucket',
                                yaxis={'categoryorder':'category descending'}
                            )
                            st.plotly_chart(fig_hm, use_container_width=True)
                        else:
                            st.info(f"Not enough data to generate {factor_name} heatmap.")
            
            # === EXIT ANALYSIS ===
            if not exit_analysis.empty:
                st.markdown("#### 🚪 Exit Analysis")
                st.caption("Why are trades being closed?")
                
                ea_col1, ea_col2 = st.columns([1, 1])
                
                with ea_col1:
                    fig_pie = px.pie(
                        exit_analysis, 
                        values='Count', 
                        names=exit_analysis.index,
                        title='Exit Reasons Breakdown',
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_pie.update_layout(height=300, showlegend=True, template='plotly_white')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with ea_col2:
                    st.dataframe(
                        exit_analysis,
                        use_container_width=True
                    )
            
            # === TRADE LOG ===
            st.markdown("---")
            with st.expander("📋 Trade Log", expanded=False):
                if trades:
                    trades_df = pd.DataFrame(trades)
                    st.dataframe(
                        trades_df,
                        column_config={
                            "date": st.column_config.DateColumn("Date"),
                            "ticker": st.column_config.TextColumn("Ticker"),
                            "action": st.column_config.TextColumn("Action"),
                            "price": st.column_config.NumberColumn("Price", format="₹%.2f"),
                            "shares": st.column_config.NumberColumn("Shares"),
                            "value": st.column_config.NumberColumn("Value", format="₹%.0f"),
                            "reason": st.column_config.TextColumn("Reason"),
                            "return_pct": st.column_config.NumberColumn("Return %", format="%+.1f%%"),
                        },
                        hide_index=True,
                        height=400
                    )
            
            # === SCORE IMPROVEMENT RECOMMENDATIONS ===
            st.markdown("---")
            st.markdown("### 💡 Score Engine Improvement Recommendations")
            
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                if metrics['win_rate_pct'] < 50:
                    st.markdown("""
                    <div style="background: rgba(255,82,82,0.1); padding: 20px; border-radius: 12px; border-left: 4px solid #FF5252;">
                        <h4 style="color: #FF5252; margin: 0;">⚠️ Win Rate Below 50%</h4>
                        <p style="color: #888; margin-top: 10px;">
                            <strong>Problem:</strong> More losing trades than winning trades.<br>
                            <strong>Suggestion:</strong> The trend score entry threshold (70) may be too low. Consider:
                            <ul>
                                <li>Raising entry threshold to 75 or 80</li>
                                <li>Adding a Quality score filter (≥ 6.0)</li>
                                <li>Requiring 3-month positive return before entry</li>
                            </ul>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: rgba(0,200,83,0.1); padding: 20px; border-radius: 12px; border-left: 4px solid #00C853;">
                        <h4 style="color: #00C853; margin: 0;">✅ Good Win Rate</h4>
                        <p style="color: #888; margin-top: 10px;">
                            The trend score is identifying winners more often than losers. 
                            Focus on improving the <strong>magnitude</strong> of wins.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with rec_col2:
                if metrics['alpha'] < 0:
                    st.markdown(f"""
                    <div style="background: rgba(255,152,0,0.1); padding: 20px; border-radius: 12px; border-left: 4px solid #FF9800;">
                        <h4 style="color: #FF9800; margin: 0;">📉 Negative Alpha</h4>
                        <p style="color: #888; margin-top: 10px;">
                            <strong>Problem:</strong> Strategy underperformed the benchmark by {abs(metrics['alpha']):.1f}%.<br>
                            <strong>Suggestion:</strong> The momentum-focused approach may need balancing:
                            <ul>
                                <li>Add a Value score filter to avoid overpriced momentum</li>
                                <li>Consider sector diversification limits</li>
                                <li>Tighten stop losses from -15% to -10%</li>
                            </ul>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: rgba(0,200,83,0.1); padding: 20px; border-radius: 12px; border-left: 4px solid #00C853;">
                        <h4 style="color: #00C853; margin: 0;">✅ Positive Alpha: +{metrics['alpha']:.1f}%</h4>
                        <p style="color: #888; margin-top: 10px;">
                            The trend score strategy is beating the market! Consider:
                            <ul>
                                <li>Increasing position sizes for high-conviction picks</li>
                                <li>Extending the holding period for winners</li>
                            </ul>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Reset button
            st.markdown("---")
            if st.button("🔄 Re-run Backtest", use_container_width=True):
                del st.session_state['backtest_results']
                st.rerun()
    
    # === MULTI-STRATEGY COMPARISON SECTION ===
    st.markdown("---")
    st.markdown("## 🔬 Multi-Strategy Comparison")
    st.caption("Compare 5 different strategy proposals with bi-weekly and monthly rebalancing")
    
    from utils.multi_strategy_backtest import (
        run_all_proposals_comparison, get_best_proposal,
        run_proposal_backtest
    )
    from utils.strategy_definitions import get_all_proposal_keys, get_proposal_description, ALL_PROPOSALS
    
    # Show available proposals
    with st.expander("📋 Available Strategies", expanded=False):
        for key in get_all_proposal_keys():
            proposal = ALL_PROPOSALS.get(key, {})
            st.markdown(f"**{key}**: {proposal.get('name', 'Unknown')} - {proposal.get('description', '')}")
    
    # Run comparison
    if 'multi_strategy_results' not in st.session_state:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_proposals = st.multiselect(
                "Select proposals to test",
                get_all_proposal_keys(),
                default=get_all_proposal_keys()[:3]  # Default: first 3
            )
        
        with col2:
            selected_freq = st.selectbox(
                "Rebalancing Frequency", 
                ["bi-weekly", "monthly", "both"],
                index=2  # Default: both
            )
        
        if st.button("🚀 Run Strategy Comparison", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = {}
            frequencies = ["bi-weekly", "monthly"] if selected_freq == "both" else [selected_freq]
            
            total_runs = len(selected_proposals) * len(frequencies)
            current_run = 0
            
            for proposal_key in selected_proposals:
                for freq in frequencies:
                    status_text.text(f"Testing {proposal_key} ({freq})...")
                    
                    result = run_proposal_backtest(
                        proposal_key=proposal_key,
                        market_df=df,
                        rebalance_freq=freq
                    )
                    
                    results[f"{proposal_key}_{freq}"] = result
                    current_run += 1
                    progress_bar.progress(current_run / total_runs)
            
            st.session_state['multi_strategy_results'] = results
            progress_bar.empty()
            status_text.empty()
            st.rerun()
    else:
        results = st.session_state['multi_strategy_results']
        
        # Find best performer
        best_key, best_result = get_best_proposal(results)
        
        if best_key and 'error' not in best_result:
            st.success(f"🏆 **Best Performer**: {best_key} with {best_result['metrics']['alpha']:+.1f}% alpha")
        
        # Comparison table
        st.markdown("### 📊 Results Comparison")
        
        comparison_data = []
        for key, result in results.items():
            if 'error' in result:
                continue
            
            m = result.get('metrics', {})
            comparison_data.append({
                "Strategy": key,
                "Total Return %": m.get('total_return_pct', 0),
                "Alpha %": m.get('alpha', 0),
                "Sharpe": m.get('sharpe_ratio', 0),
                "Max DD %": m.get('max_drawdown_pct', 0),
                "Win Rate %": m.get('win_rate_pct', 0),
                "Trades": m.get('total_trades', 0),
                "Final Value": m.get('final_value', 0),
            })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values("Alpha %", ascending=False)
            
            st.dataframe(
                comparison_df,
                column_config={
                    "Strategy": st.column_config.TextColumn("Strategy"),
                    "Total Return %": st.column_config.NumberColumn("Return", format="%+.1f%%"),
                    "Alpha %": st.column_config.NumberColumn("Alpha", format="%+.1f%%"),
                    "Sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                    "Max DD %": st.column_config.NumberColumn("Max DD", format="%.1f%%"),
                    "Win Rate %": st.column_config.NumberColumn("Win Rate", format="%.0f%%"),
                    "Trades": st.column_config.NumberColumn("Trades"),
                    "Final Value": st.column_config.NumberColumn("Final ₹", format="₹%.0f"),
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Visual comparison
            st.markdown("### 📈 Alpha Comparison")
            
            alpha_df = comparison_df[['Strategy', 'Alpha %']].copy()
            alpha_df['Color'] = alpha_df['Alpha %'].apply(lambda x: '#00C853' if x > 0 else '#FF5252')
            
            fig_alpha = px.bar(
                alpha_df, 
                x='Strategy', 
                y='Alpha %',
                color='Alpha %',
                color_continuous_scale=['#FF5252', '#FF9800', '#00C853'],
                title='Alpha vs Benchmark by Strategy'
            )
            fig_alpha.update_layout(
                template='plotly_white',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350
            )
            st.plotly_chart(fig_alpha, use_container_width=True)
            
            # Risk-adjusted view
            st.markdown("### ⚖️ Risk-Adjusted Performance")
            
            risk_df = comparison_df[['Strategy', 'Sharpe', 'Max DD %', 'Win Rate %']].copy()
            
            fig_risk = go.Figure()
            fig_risk.add_trace(go.Bar(
                name='Sharpe Ratio',
                x=risk_df['Strategy'],
                y=risk_df['Sharpe'],
                marker_color='#667eea'
            ))
            fig_risk.add_trace(go.Bar(
                name='Max Drawdown',
                x=risk_df['Strategy'],
                y=risk_df['Max DD %'].abs(),
                marker_color='#FF5252'
            ))
            fig_risk.update_layout(
                barmode='group',
                template='plotly_white',
                paper_bgcolor='rgba(0,0,0,0)',
                height=300,
                legend=dict(orientation='h')
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # === DEEP DIVE ANALYTICS FOR BEST STRATEGY ===
            if best_key and best_result:
                st.markdown("---")
                st.markdown(f"### 🔬 Deep Dive Analytics: {best_key}")
                
                best_analytics = best_result.get('analytics', {})
                best_heatmaps = best_analytics.get('heatmaps', {})
                best_exit = best_analytics.get('exit_analysis', pd.DataFrame())
                
                # Show Heatmap
                if best_heatmaps and "Quality" in best_heatmaps:
                    st.markdown("#### 🔥 Trend vs Quality Heatmap")
                    hm_data = best_heatmaps["Quality"]
                    if not hm_data.empty:
                        fig_hm = go.Figure(data=go.Heatmap(
                            z=hm_data.values,
                            x=hm_data.columns,
                            y=hm_data.index,
                            colorscale='RdYlGn',
                            texttemplate="%{z:.1f}%",
                            textfont={"size": 10}
                        ))
                        fig_hm.update_layout(
                            height=350,
                            template='plotly_white',
                            xaxis_title='Trend Score', 
                            yaxis_title='Quality Score',
                            title=f"Return Heatmap ({best_key})"
                        )
                        st.plotly_chart(fig_hm, use_container_width=True)
                
                # Show Exit Analysis
                if not best_exit.empty:
                    st.markdown("#### 🚪 Exit Analysis")
                    ea_col1, ea_col2 = st.columns([1, 1])
                    with ea_col1:
                        if isinstance(best_exit, pd.DataFrame) and 'Count' in best_exit.columns:
                            fig_pie = px.pie(
                                best_exit, 
                                values='Count', 
                                names=best_exit.index,
                                hole=0.4,
                                color_discrete_sequence=px.colors.qualitative.Set3,
                                title="Exit Reasons"
                            )
                            fig_pie.update_layout(height=300, template='plotly_white')
                            st.plotly_chart(fig_pie, use_container_width=True)
                    with ea_col2:
                        st.dataframe(best_exit, use_container_width=True)
        
        # Reset button
        if st.button("🔄 Re-run Comparison", use_container_width=True):
            del st.session_state['multi_strategy_results']
            st.rerun()

# --- VIEW: STRATEGY LAB ---
elif page == "🔬 Strategy Lab":
    from utils.strategy_optimizer import (
        run_optimization, get_top_strategies, generate_heatmap_data,
        DEFAULT_GRID, FIXED_PARAMS, run_quick_backtest
    )
    
    st.markdown(page_header("🔬 Strategy Lab", "Test parameter combinations to find high-alpha strategies"), unsafe_allow_html=True)
    
    # Load market data
    if 'market_df' not in st.session_state or st.session_state.get('market_df') is None:
        with st.spinner("Loading market data..."):
            from utils.nifty500_list import TICKERS
            from utils.data_engine import batch_fetch_tickers
            from utils.scoring import calculate_scores, calculate_trend_metrics
            
            st.session_state['market_df'] = batch_fetch_tickers(TICKERS)
    
    df = st.session_state.get('market_df')
    
    if df is None or df.empty:
        st.error("Failed to load market data. Please refresh.")
    else:
        # === TAB LAYOUT ===
        tab1, tab2, tab3, tab4 = st.tabs(["🎛️ Custom Strategy", "🎯 Strategy Presets", "⚡ Optimizer Grid", "📊 My Portfolio"])
        
        # === TAB 1: CUSTOM STRATEGY ===
        with tab1:
            st.markdown("### 🎛️ Customize Entry & Exit Parameters")
            st.caption("Adjust parameters and run a single backtest")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Entry Rules")
                trend_min = st.slider("Trend Score Min", 30, 80, 40, key="custom_trend_min")
                trend_max = st.slider("Trend Score Max", 50, 100, 80, key="custom_trend_max")
                quality_min = st.slider("Quality Min", 0.0, 10.0, 5.0, 0.5, key="custom_quality")
                value_min = st.slider("Value Min", 0.0, 10.0, 0.0, 0.5, key="custom_value")
                growth_min = st.slider("Growth Min", 0.0, 10.0, 0.0, 0.5, key="custom_growth")
                volume_min = st.slider("Volume Signal Min", 0, 10, 5, key="custom_volume")
            
            with col2:
                st.markdown("#### Exit Rules")
                stop_loss = st.slider("Stop Loss %", -25, -5, -15, key="custom_stop")
                trailing_stop = st.slider("Trailing Stop %", -20, -3, -10, key="custom_trail")
                time_exit_days = st.slider("Time Exit (days)", 14, 60, 45, key="custom_days")
                time_exit_return = st.slider("Time Exit Min Return %", -5.0, 10.0, 0.0, 0.5, key="custom_ret")
                partial_pct = st.slider("Partial Profit Trigger %", 10, 50, 20, key="custom_partial")
                lookback = st.slider("Backtest Months", 6, 24, 12, key="custom_lookback")
            
            if st.button("▶️ Run Custom Backtest", type="primary", use_container_width=True):
                config = {
                    **FIXED_PARAMS,
                    "entry_trend_score": trend_min,
                    "entry_trend_score_max": trend_max,
                    "quality_min": quality_min,
                    "value_min": value_min,
                    "growth_min": growth_min,
                    "volume_combined_min": volume_min,
                    "stop_loss_pct": stop_loss,
                    "trailing_stop_pct": trailing_stop,
                    "time_exit_days": time_exit_days,
                    "time_exit_min_return": time_exit_return,
                    "partial_profit_pct": partial_pct,
                    "lookback_months": lookback
                }
                
                with st.spinner("Running backtest..."):
                    result = run_quick_backtest(df, config)
                
                if "error" not in result:
                    metrics = result['metrics']
                    
                    st.markdown("### 📊 Results")
                    m_cols = st.columns(5)
                    m_cols[0].metric("Return", f"{metrics.get('total_return_pct', 0):+.1f}%")
                    m_cols[1].metric("Alpha", f"{metrics.get('alpha', 0):+.1f}%")
                    m_cols[2].metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")
                    m_cols[3].metric("Win Rate", f"{metrics.get('win_rate_pct', 0):.0f}%")
                    m_cols[4].metric("Trades", f"{metrics.get('total_trades', 0)}")
                    
                    # Equity curve
                    eq_df = result.get('equity_curve')
                    if eq_df is not None and not eq_df.empty:
                        fig = px.line(eq_df, x='date', y='equity', title='Equity Curve')
                        fig.update_layout(height=300, template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Trade log
                    trades_list = result.get('trades', [])
                    if trades_list:
                        trade_df = pd.DataFrame(trades_list)
                        with st.expander(f"📋 Trade Log ({len(trade_df)} trades)", expanded=False):
                            st.dataframe(trade_df, use_container_width=True, height=300)
                            
                            # CSV download
                            csv_data = trade_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Trade Log (CSV)",
                                data=csv_data,
                                file_name="custom_strategy_trades.csv",
                                mime="text/csv",
                                key="custom_trades_download"
                            )
                else:
                    st.error(f"Error: {result.get('error', 'Unknown')}")
        
        # === TAB 2: PRESETS ===
        with tab2:
            st.markdown("### 🎯 Strategy Presets")
            st.caption("One-click strategies for different market conditions")
            
            preset_cols = st.columns(5)
            
            presets = {
                "🔥 Hot Zone (Best)": {
                    "trend_min": 35, "trend_max": 60, "quality_min": 0, "value_min": 0, "growth_min": 0,
                    "volume_combined_min": 6, "time_exit_days": 45, "trailing_stop_pct": -8
                },
                "🚀 Momentum": {
                    "trend_min": 70, "trend_max": 90, "quality_min": 5, "value_min": 0,
                    "time_exit_days": 20, "trailing_stop_pct": -10
                },
                "💎 Deep Value": {
                    "trend_min": 30, "trend_max": 55, "quality_min": 5, "value_min": 5,
                    "time_exit_days": 60, "trailing_stop_pct": -15, "stop_loss_pct": -20
                },
                "⚖️ GARP": {
                    "trend_min": 50, "trend_max": 75, "quality_min": 5, "value_min": 5, "growth_min": 5,
                    "time_exit_days": 30, "trailing_stop_pct": -12
                },
                "📊 Breakout": {
                    "trend_min": 60, "trend_max": 80, "quality_min": 0, "value_min": 0,
                    "volume_combined_min": 7, "time_exit_days": 25
                }
            }
            
            for i, (name, params) in enumerate(presets.items()):
                with preset_cols[i]:
                    st.markdown(f"**{name}**")
                    st.caption(f"Trend: {params['trend_min']}-{params['trend_max']}")
                    st.caption(f"Q:{params.get('quality_min', 0)} V:{params.get('value_min', 0)}")
                    
                    if st.button(f"Test {name.split()[1]}", key=f"preset_{i}"):
                        config = {**FIXED_PARAMS, **params}
                        config["entry_trend_score"] = params["trend_min"]
                        config["entry_trend_score_max"] = params["trend_max"]
                        
                        with st.spinner(f"Testing {name}..."):
                            result = run_quick_backtest(df, config)
                        
                        if "error" not in result:
                            st.session_state['preset_result'] = result
                            st.session_state['preset_name'] = name
            
            # Show preset result
            if 'preset_result' in st.session_state:
                result = st.session_state['preset_result']
                name = st.session_state.get('preset_name', 'Preset')
                metrics = result.get('metrics', {})
                trades = result.get('trades', [])
                
                st.markdown(f"### {name} Results")
                m_cols = st.columns(5)
                m_cols[0].metric("Return", f"{metrics.get('total_return_pct', 0):+.1f}%")
                m_cols[1].metric("Alpha", f"{metrics.get('alpha', 0):+.1f}%")
                m_cols[2].metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")
                m_cols[3].metric("Win Rate", f"{metrics.get('win_rate_pct', 0):.0f}%")
                m_cols[4].metric("Trades", f"{metrics.get('total_trades', 0)}")
                
                # Equity curve with benchmark overlay
                eq_df = result.get('equity_curve')
                if eq_df is not None and not eq_df.empty:

                    
                    fig = go.Figure()
                    
                    # Normalize to base 100
                    initial_val = eq_df['equity'].iloc[0]
                    eq_df['normalized'] = (eq_df['equity'] / initial_val) * 100
                    
                    # Add strategy line
                    fig.add_trace(go.Scatter(
                        x=eq_df['date'], y=eq_df['normalized'],
                        mode='lines', name='Strategy',
                        line=dict(color='#00C853', width=2)
                    ))
                    
                    # Fetch benchmark (Nifty 500)
                    try:
                        start = eq_df['date'].min()
                        end = eq_df['date'].max()
                        nifty = yf.download("^CRSLDX", start=start, end=end, progress=False)
                        if not nifty.empty:
                            nifty_start = nifty['Close'].iloc[0]
                            nifty['normalized'] = (nifty['Close'] / nifty_start) * 100
                            fig.add_trace(go.Scatter(
                                x=nifty.index, y=nifty['normalized'],
                                mode='lines', name='Nifty 500',
                                line=dict(color='#888', width=1, dash='dash')
                            ))
                    except:
                        pass
                    
                    fig.update_layout(
                        height=280, template='plotly_white',
                        title='Equity Curve vs Benchmark (Base 100)',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        xaxis_title='', yaxis_title='Value'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Trade Log
                if trades:
                    with st.expander(f"📋 Trade Log ({len(trades)} trades)", expanded=False):
                        trades_df = pd.DataFrame(trades)
                        
                        # Display table
                        st.dataframe(
                            trades_df,
                            column_config={
                                "date": st.column_config.DateColumn("Date"),
                                "ticker": st.column_config.TextColumn("Ticker"),
                                "action": st.column_config.TextColumn("Action"),
                                "price": st.column_config.NumberColumn("Price", format="₹%.2f"),
                                "shares": st.column_config.NumberColumn("Shares"),
                                "reason": st.column_config.TextColumn("Reason"),
                                "return_pct": st.column_config.NumberColumn("Return %", format="%+.1f%%"),
                                "days_held": st.column_config.NumberColumn("Days"),
                                "entry_trend_score": st.column_config.NumberColumn("Entry Trend"),
                                "quality": st.column_config.NumberColumn("Quality"),
                                "value": st.column_config.NumberColumn("Value"),
                            },
                            hide_index=True,
                            height=400
                        )
                        
                        # Export button
                        csv = trades_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Trade Log CSV",
                            data=csv,
                            file_name=f"{name.replace(' ', '_').replace('🚀', '').replace('💎', '').replace('⚖️', '').replace('🔥', '').strip()}_trades.csv",
                            mime="text/csv"
                        )
        
        # === TAB 3: OPTIMIZER GRID ===
        with tab3:
            st.markdown("### ⚡ Two-Stage Optimization")
            st.caption("Stage 1: Find winning entry params → Stage 2: Refine exit params")
            
            # Two-stage sub-tabs
            stage1_tab, stage2_tab = st.tabs(["📊 Stage 1: Entry Params", "🔧 Stage 2: Exit Params"])
            
            # === STAGE 1: ENTRY OPTIMIZATION ===
            with stage1_tab:
                st.markdown("#### Find the best Trend × Quality × Value × Growth combination")
                st.info("Tests ~162 entry combos with fixed exit params (30 days, -10% trail, -15% stop)")
                
                # Live results container
                live_results_container = st.empty()
                
                col_run, col_stop = st.columns([3, 1])
                with col_run:
                    run_btn = st.button("🚀 Run Entry Optimizer", type="primary", use_container_width=True, key="stage1_run")
                with col_stop:
                    stop_btn = st.button("⏹️ Stop", use_container_width=True, key="stage1_stop")
                
                if stop_btn:
                    st.session_state['optimizer_stop'] = True
                
                if run_btn:
                    st.session_state['optimizer_stop'] = False
                    st.session_state['live_results'] = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Import components
                    from utils.strategy_optimizer import (
                        generate_param_grid, run_quick_backtest, calculate_composite_score,
                        ENTRY_GRID, STAGE1_FIXED_EXITS, FIXED_PARAMS
                    )

                    from utils.portfolio_backtest import fetch_historical_prices
                
                    # Generate configs using ENTRY_GRID with fixed exit params
                    configs = generate_param_grid(ENTRY_GRID, STAGE1_FIXED_EXITS)
                    total = len(configs)
                    
                    # Fetch data once
                    status_text.text("Fetching historical data...")
                    tickers = df['ticker'].tolist()
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=365)
                    fetch_start = start_date - timedelta(days=400)
                    historical_data = fetch_historical_prices(tickers, fetch_start, end_date)
                    
                    # Run each config and stream results
                    for i, config in enumerate(configs):
                        # Check stop flag
                        if st.session_state.get('optimizer_stop', False):
                            status_text.text(f"⏹️ Stopped at {i}/{total}")
                            break
                        
                        progress_bar.progress((i + 1) / total)
                        status_text.text(f"Testing config {i+1}/{total}...")
                        
                        try:
                            result = run_quick_backtest(df, config, historical_data, start_date, end_date)
                            
                            if "error" not in result:
                                metrics = result.get('metrics', {})
                                trades = result.get('trades', [])
                                score = calculate_composite_score(metrics, trades)
                                
                                row = {
                                    'archetype': config.get('archetype', 'Unknown'),
                                    'trend_min': config['entry_trend_score'],
                                    'trend_max': config['entry_trend_score_max'],
                                    'quality_min': config.get('quality_min', 0),
                                    'value_min': config.get('value_min', 0),
                                    'growth_min': config.get('growth_min', 0),
                                    'coverage': config.get('coverage_pct', 0),
                                    'return_pct': metrics.get('total_return_pct', 0),
                                    'alpha': metrics.get('alpha', 0),
                                    'sharpe': metrics.get('sharpe_ratio', 0),
                                    'win_rate': metrics.get('win_rate_pct', 0),
                                    'max_dd': metrics.get('max_drawdown_pct', 0),
                                    'trades': metrics.get('total_trades', 0),
                                    'score': score
                                }
                                st.session_state['live_results'].append(row)
                                
                                # Update live results table (sorted by score)
                                if st.session_state['live_results']:
                                    live_df = pd.DataFrame(st.session_state['live_results'])
                                    live_df = live_df.sort_values('score', ascending=False).reset_index(drop=True)
                                    live_df['rank'] = range(1, len(live_df) + 1)
                                    
                                    with live_results_container.container():
                                        st.markdown(f"### 📊 Results So Far ({len(live_df)} tested)")
                                        display_cols = ['rank', 'archetype', 'trend_min', 'trend_max', 'quality_min', 'value_min', 'growth_min', 'return_pct', 'alpha', 'score']
                                        display_cols = [c for c in display_cols if c in live_df.columns]
                                        st.dataframe(
                                            live_df.head(25)[display_cols],
                                            hide_index=True,
                                            use_container_width=True,
                                            height=400
                                        )
                        except Exception as e:
                            # Track errors silently
                            if 'error_count' not in st.session_state:
                                st.session_state['error_count'] = 0
                            st.session_state['error_count'] += 1
                            continue
                    
                    # Final results
                    if st.session_state.get('live_results'):
                        final_df = pd.DataFrame(st.session_state['live_results'])
                        
                        # Remove duplicates (entry configs with same results)
                        dedup_cols = ['trend_min', 'trend_max', 'quality_min', 'value_min', 
                                      'growth_min', 'return_pct', 'alpha', 'sharpe']
                        dedup_cols = [c for c in dedup_cols if c in final_df.columns]
                        final_df = final_df.drop_duplicates(subset=dedup_cols, keep='first')
                        
                        final_df = final_df.sort_values('score', ascending=False).reset_index(drop=True)
                        final_df['rank'] = range(1, len(final_df) + 1)
                        st.session_state['stage1_results'] = final_df
                        
                        progress_bar.progress(100)
                        error_count = st.session_state.get('error_count', 0)
                        success_count = len(st.session_state.get('live_results', []))
                        status_text.text(f"✅ Stage 1 complete! {success_count} success, {error_count} errors")
                    else:
                        progress_bar.progress(100)
                        error_count = st.session_state.get('error_count', 0)
                        st.error(f"⚠️ No successful configs! {error_count} errors. Check that stocks have quality/value/growth data.")
                
                # Show Stage 1 results
                if 'stage1_results' in st.session_state:
                    results_df = st.session_state['stage1_results']
                    
                    # Stats
                    tested_count = len(st.session_state.get('live_results', []))
                    unique_count = len(results_df)
                    st.success(f"📊 Tested **{tested_count}** → **{unique_count}** unique entry strategies")
                    
                    st.markdown("### 🏆 Top 10 Entry Strategies")
                    
                    display_cols = ['rank', 'trend_min', 'trend_max', 'quality_min', 'value_min', 'growth_min',
                                   'return_pct', 'alpha', 'sharpe', 'win_rate', 'trades', 'score']
                    display_cols = [c for c in display_cols if c in results_df.columns]
                    
                    st.dataframe(
                        results_df.head(10)[display_cols],
                        column_config={
                            "rank": st.column_config.NumberColumn("Rank"),
                            "trend_min": st.column_config.NumberColumn("Trend Min"),
                            "trend_max": st.column_config.NumberColumn("Trend Max"),
                            "quality_min": st.column_config.NumberColumn("Quality"),
                            "value_min": st.column_config.NumberColumn("Value"),
                            "growth_min": st.column_config.NumberColumn("Growth"),
                            "return_pct": st.column_config.NumberColumn("Return %", format="%+.1f%%"),
                            "alpha": st.column_config.NumberColumn("Alpha", format="%+.1f%%"),
                            "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                            "win_rate": st.column_config.NumberColumn("Win %", format="%.0f%%"),
                            "trades": st.column_config.NumberColumn("Trades"),
                            "score": st.column_config.NumberColumn("Score", format="%.3f"),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Download button
                    csv_data = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Entry Results (CSV)",
                        data=csv_data,
                        file_name="stage1_entry_results.csv",
                        mime="text/csv",
                        key="stage1_download"
                    )
                    
                    # Select winner for Stage 2
                    st.markdown("---")
                    st.markdown("### ➡️ Select Winner for Stage 2")
                    winner_idx = st.selectbox(
                        "Choose entry strategy to optimize exits",
                        options=range(min(10, len(results_df))),
                        format_func=lambda x: f"#{x+1}: Trend {results_df.iloc[x]['trend_min']}-{results_df.iloc[x]['trend_max']}, Q:{results_df.iloc[x]['quality_min']}, V:{results_df.iloc[x]['value_min']}, G:{results_df.iloc[x].get('growth_min', 0)} → Alpha {results_df.iloc[x]['alpha']:+.1f}%",
                        key="stage2_winner_select"
                    )
                    
                    if st.button("✅ Use for Stage 2", key="use_for_stage2"):
                        winner = results_df.iloc[winner_idx]
                        st.session_state['stage2_entry_params'] = {
                            'entry_trend_score': int(winner['trend_min']),
                            'entry_trend_score_max': int(winner['trend_max']),
                            'quality_min': float(winner['quality_min']),
                            'value_min': float(winner['value_min']),
                            'growth_min': float(winner.get('growth_min', 0))
                        }
                        st.success(f"✅ Selected: Trend {winner['trend_min']}-{winner['trend_max']}, Q:{winner['quality_min']}, V:{winner['value_min']} → Go to Stage 2 tab!")
            
            # === STAGE 2: EXIT OPTIMIZATION ===
            with stage2_tab:
                st.markdown("#### Refine exit params for your winning entry strategy")
                
                # Show selected entry params from Stage 1
                if 'stage2_entry_params' in st.session_state:
                    entry = st.session_state['stage2_entry_params']
                    st.success(f"✅ Entry: Trend {entry['entry_trend_score']}-{entry['entry_trend_score_max']}, "
                              f"Q:{entry['quality_min']}, V:{entry['value_min']}, G:{entry['growth_min']}")
                    
                    st.info("Tests ~36 exit combos (Time × Trailing × Stop × Volume)")
                    
                    # Stage 2 run button
                    if st.button("🚀 Run Exit Optimizer", type="primary", use_container_width=True, key="stage2_run"):
                        st.session_state['stage2_live_results'] = []
                        
                        progress_bar2 = st.progress(0)
                        status_text2 = st.empty()
                        live_results_container2 = st.empty()
                        
                        # Import components
                        from utils.strategy_optimizer import (
                            generate_param_grid, run_quick_backtest, calculate_composite_score,
                            EXIT_GRID, FIXED_PARAMS
                        )

                        from utils.portfolio_backtest import fetch_historical_prices
                        
                        # Generate exit configs with fixed entry from Stage 1
                        configs = generate_param_grid(EXIT_GRID, entry)
                        total = len(configs)
                        
                        # Fetch data
                        status_text2.text("Fetching historical data...")
                        tickers = df['ticker'].tolist()
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=365)
                        fetch_start = start_date - timedelta(days=400)
                        historical_data = fetch_historical_prices(tickers, fetch_start, end_date)
                        
                        # Run each config
                        for i, config in enumerate(configs):
                            progress_bar2.progress((i + 1) / total)
                            status_text2.text(f"Testing exit config {i+1}/{total}...")
                            
                            try:
                                result = run_quick_backtest(df, config, historical_data, start_date, end_date)
                                
                                if "error" not in result:
                                    metrics = result.get('metrics', {})
                                    trades = result.get('trades', [])
                                    score = calculate_composite_score(metrics, trades)
                                    
                                    row = {
                                        'time_exit': config.get('time_exit_days', 20),
                                        'trail_stop': config.get('trailing_stop_pct', -10),
                                        'stop_loss': config.get('stop_loss_pct', -15),
                                        'volume_min': config.get('volume_combined_min', 5),
                                        'return_pct': metrics.get('total_return_pct', 0),
                                        'alpha': metrics.get('alpha', 0),
                                        'sharpe': metrics.get('sharpe_ratio', 0),
                                        'win_rate': metrics.get('win_rate_pct', 0),
                                        'max_dd': metrics.get('max_drawdown_pct', 0),
                                        'trades': metrics.get('total_trades', 0),
                                        'score': score
                                    }
                                    st.session_state['stage2_live_results'].append(row)
                                    
                                    # Update live display
                                    if st.session_state['stage2_live_results']:
                                        live_df = pd.DataFrame(st.session_state['stage2_live_results'])
                                        live_df = live_df.sort_values('score', ascending=False).reset_index(drop=True)
                                        live_df['rank'] = range(1, len(live_df) + 1)
                                        
                                        with live_results_container2.container():
                                            st.markdown(f"### 📊 Exit Results ({len(live_df)} tested)")
                                            st.dataframe(
                                                live_df.head(10)[['rank', 'time_exit', 'trail_stop', 'stop_loss', 'volume_min', 'return_pct', 'alpha', 'sharpe', 'score']],
                                                hide_index=True,
                                                use_container_width=True
                                            )
                            except:
                                continue
                        
                        # Final results
                        if st.session_state.get('stage2_live_results'):
                            final_df = pd.DataFrame(st.session_state['stage2_live_results'])
                            final_df = final_df.sort_values('score', ascending=False).reset_index(drop=True)
                            final_df['rank'] = range(1, len(final_df) + 1)
                            st.session_state['stage2_results'] = final_df
                        
                        progress_bar2.progress(100)
                        status_text2.text("✅ Stage 2 complete!")
                    
                    # Show Stage 2 results
                    if 'stage2_results' in st.session_state:
                        results_df = st.session_state['stage2_results']
                        
                        st.markdown("### 🏆 Top Exit Strategies")
                        st.dataframe(
                            results_df.head(10),
                            column_config={
                                "rank": st.column_config.NumberColumn("Rank"),
                                "time_exit": st.column_config.NumberColumn("Time Exit"),
                                "trail_stop": st.column_config.NumberColumn("Trail %"),
                                "stop_loss": st.column_config.NumberColumn("Stop %"),
                                "volume_min": st.column_config.NumberColumn("Vol Min"),
                                "return_pct": st.column_config.NumberColumn("Return %", format="%+.1f%%"),
                                "alpha": st.column_config.NumberColumn("Alpha", format="%+.1f%%"),
                                "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                                "win_rate": st.column_config.NumberColumn("Win %", format="%.0f%%"),
                                "max_dd": st.column_config.NumberColumn("Max DD", format="%.1f%%"),
                                "trades": st.column_config.NumberColumn("Trades"),
                                "score": st.column_config.NumberColumn("Score", format="%.3f"),
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Best combined config
                        if len(results_df) > 0:
                            best = results_df.iloc[0]
                            entry = st.session_state['stage2_entry_params']
                            st.markdown("### 🥇 Best Complete Configuration")
                            st.json({
                                "Entry": {
                                    "Trend Range": f"{entry['entry_trend_score']}-{entry['entry_trend_score_max']}",
                                    "Quality Min": entry['quality_min'],
                                    "Value Min": entry['value_min'],
                                    "Growth Min": entry['growth_min']
                                },
                                "Exit": {
                                    "Time Exit Days": best['time_exit'],
                                    "Trailing Stop": f"{best['trail_stop']}%",
                                    "Stop Loss": f"{best['stop_loss']}%",
                                    "Volume Min": best['volume_min']
                                },
                                "Performance": {
                                    "Return": f"{best['return_pct']:+.1f}%",
                                    "Alpha": f"{best['alpha']:+.1f}%",
                                    "Sharpe": f"{best['sharpe']:.2f}",
                                    "Score": f"{best['score']:.3f}"
                                }
                            })
                        
                        # Download
                        csv_data = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Exit Results (CSV)",
                            data=csv_data,
                            file_name="stage2_exit_results.csv",
                            mime="text/csv",
                            key="stage2_download"
                        )
                else:
                    st.warning("⚠️ First run Stage 1 and select a winning entry strategy")
        
        # === TAB 4: MY PORTFOLIO ===
        with tab4:
            st.markdown("### 📊 Custom Portfolio Backtest")
            st.caption("Build your own portfolio and test how it would have performed")
            
            # Get available tickers
            available_tickers = sorted(df['ticker'].tolist()) if 'ticker' in df.columns else []
            
            # Portfolio selection
            st.markdown("#### 1️⃣ Select Your Stocks")
            
            col_select, col_upload = st.columns([2, 1])
            
            with col_select:
                selected_stocks = st.multiselect(
                    "Pick stocks for your portfolio",
                    options=available_tickers,
                    default=st.session_state.get('my_portfolio', []),
                    help="Start typing to search",
                    key="portfolio_select"
                )
                st.session_state['my_portfolio'] = selected_stocks
            
            with col_upload:
                uploaded_file = st.file_uploader("Or upload CSV", type=['csv'], key="portfolio_csv")
                if uploaded_file:
                    try:
                        upload_df = pd.read_csv(uploaded_file)
                        # Try common column name patterns
                        ticker_col = None
                        for col_name in ['ticker', 'Ticker', 'Symbol', 'symbol', 'Stock Name', 'stock_name', 'Name', 'name', 'Company', 'company']:
                            if col_name in upload_df.columns:
                                ticker_col = col_name
                                break
                        
                        if ticker_col:
                            tickers = upload_df[ticker_col].tolist()
                        else:
                            # Default to first column, but skip if it looks numeric
                            first_col = upload_df.columns[0]
                            if upload_df[first_col].dtype in ['int64', 'float64']:
                                # Likely a rank/index column, try second column
                                if len(upload_df.columns) > 1:
                                    tickers = upload_df.iloc[:, 1].tolist()
                                else:
                                    tickers = upload_df.iloc[:, 0].tolist()
                            else:
                                tickers = upload_df.iloc[:, 0].tolist()
                        
                        # Convert to strings and add .NS suffix if missing
                        tickers = [str(t).strip() for t in tickers if pd.notna(t)]
                        
                        # Build name-to-ticker mapping for fuzzy matching
                        name_to_ticker = {}
                        for _, row in df.iterrows():
                            ticker = row.get('ticker', '')
                            name = str(row.get('name', '')).lower().strip()
                            company = str(row.get('company', '')).lower().strip()
                            # Add both name and company as keys
                            if name:
                                name_to_ticker[name] = ticker
                            if company:
                                name_to_ticker[company] = ticker
                        
                        resolved_tickers = []
                        unmatched = []
                        
                        for t in tickers:
                            t_lower = t.lower().strip()
                            t_with_suffix = t if t.endswith('.NS') or t.endswith('.BO') else f"{t}.NS"
                            
                            # Try 1: Direct ticker match
                            if t_with_suffix in available_tickers:
                                resolved_tickers.append(t_with_suffix)
                            # Try 2: Exact name match
                            elif t_lower in name_to_ticker:
                                resolved_tickers.append(name_to_ticker[t_lower])
                            # Try 3: Partial name match (company name contains the input)
                            else:
                                matched = False
                                for name, ticker in name_to_ticker.items():
                                    # Check if input is substantially contained in name or vice versa
                                    if t_lower in name or name in t_lower:
                                        resolved_tickers.append(ticker)
                                        matched = True
                                        break
                                    # Check for word overlap (at least 2 words match)
                                    t_words = set(t_lower.replace('(', ' ').replace(')', ' ').split())
                                    name_words = set(name.replace('(', ' ').replace(')', ' ').split())
                                    if len(t_words & name_words) >= 1 and len(t_words) > 0:
                                        # Prefer matches where first word matches
                                        t_first = list(t_words)[0] if t_words else ''
                                        if t_first in name_words and len(t_first) > 3:
                                            resolved_tickers.append(ticker)
                                            matched = True
                                            break
                                if not matched:
                                    unmatched.append(t)
                        
                        valid_tickers = list(dict.fromkeys(resolved_tickers))  # Remove duplicates, preserve order
                        
                        if valid_tickers:
                            st.session_state['my_portfolio'] = valid_tickers
                            msg = f"✅ Loaded {len(valid_tickers)} stocks from CSV"
                            if unmatched:
                                msg += f" ({len(unmatched)} not found: {', '.join(unmatched[:5])}{'...' if len(unmatched) > 5 else ''})"
                            st.success(msg)
                            st.rerun()
                        else:
                            st.warning(f"Could not match any stocks. Tried: {tickers[:5]}...")
                    except Exception as e:
                        st.error(f"Error reading CSV: {e}")
            
            if selected_stocks:
                st.info(f"📋 Portfolio: {len(selected_stocks)} stocks selected")
                
                # Display selected portfolio
                with st.expander("View Selected Stocks", expanded=False):
                    portfolio_df = df[df['ticker'].isin(selected_stocks)][['ticker', 'company', 'sector', 'trend_score', 'quality', 'value', 'growth']].copy()
                    st.dataframe(portfolio_df, hide_index=True, use_container_width=True)
                
                st.markdown("---")
                st.markdown("#### 2️⃣ Configure Strategy Parameters")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Entry Rules**")
                    p_trend_min = st.slider("Trend Score Min", 30, 80, 40, key="p_trend_min")
                    p_trend_max = st.slider("Trend Score Max", 50, 100, 80, key="p_trend_max")
                    p_quality_min = st.slider("Quality Min", 0.0, 10.0, 0.0, 0.5, key="p_quality")
                    p_value_min = st.slider("Value Min", 0.0, 10.0, 0.0, 0.5, key="p_value")
                    p_growth_min = st.slider("Growth Min", 0.0, 10.0, 0.0, 0.5, key="p_growth")
                    p_volume_min = st.slider("Volume Signal Min", 0, 10, 5, key="p_volume")
                
                with col2:
                    st.markdown("**Exit Rules**")
                    p_stop_loss = st.slider("Stop Loss %", -25, -5, -15, key="p_stop")
                    p_trailing = st.slider("Trailing Stop %", -20, -3, -10, key="p_trail")
                    p_time_exit = st.slider("Time Exit (days)", 14, 60, 30, key="p_days")
                    p_time_return = st.slider("Time Exit Min Return %", -5.0, 10.0, 0.0, 0.5, key="p_ret")
                    p_partial = st.slider("Partial Profit Trigger %", 10, 50, 20, key="p_partial")
                    p_lookback = st.slider("Backtest Months", 6, 24, 12, key="p_lookback")
                
                st.markdown("---")
                
                if st.button("▶️ Run Portfolio Backtest", type="primary", use_container_width=True, key="run_portfolio_bt"):
                    # Build config
                    from utils.strategy_optimizer import FIXED_PARAMS, run_quick_backtest

                    from utils.portfolio_backtest import fetch_historical_prices
                    
                    config = {
                        **FIXED_PARAMS,
                        "entry_trend_score": p_trend_min,
                        "entry_trend_score_max": p_trend_max,
                        "quality_min": p_quality_min,
                        "value_min": p_value_min,
                        "growth_min": p_growth_min,
                        "volume_combined_min": p_volume_min,
                        "stop_loss_pct": p_stop_loss,
                        "trailing_stop_pct": p_trailing,
                        "time_exit_days": p_time_exit,
                        "time_exit_min_return": p_time_return,
                        "partial_profit_pct": p_partial,
                        "lookback_months": p_lookback
                    }
                    
                    # Filter market data to only selected stocks
                    portfolio_df = df[df['ticker'].isin(selected_stocks)].copy()
                    
                    with st.spinner(f"Running backtest on {len(portfolio_df)} stocks..."):
                        # Fetch historical data
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=p_lookback * 30)
                        fetch_start = start_date - timedelta(days=400)
                        historical_data = fetch_historical_prices(selected_stocks, fetch_start, end_date)
                        
                        # Run backtest
                        result = run_quick_backtest(portfolio_df, config, historical_data, start_date, end_date)
                    
                    if "error" not in result:
                        metrics = result.get('metrics', {})
                        
                        st.markdown("### 📊 Portfolio Performance")
                        
                        # Metrics row
                        m_cols = st.columns(6)
                        m_cols[0].metric("Return", f"{metrics.get('total_return_pct', 0):+.1f}%")
                        m_cols[1].metric("Alpha", f"{metrics.get('alpha', 0):+.1f}%")
                        m_cols[2].metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")
                        m_cols[3].metric("Win Rate", f"{metrics.get('win_rate_pct', 0):.0f}%")
                        m_cols[4].metric("Max DD", f"{metrics.get('max_drawdown_pct', 0):.1f}%")
                        m_cols[5].metric("Trades", f"{metrics.get('total_trades', 0)}")
                        
                        # Equity curve
                        eq_df = result.get('equity_curve')
                        if eq_df is not None and not eq_df.empty:

                            
                            # Fetch benchmark
                            bench = yf.download("^CRSLDX", start=eq_df['date'].min(), end=eq_df['date'].max(), progress=False)
                            if not bench.empty:
                                bench_prices = bench['Close'].reset_index()
                                bench_prices.columns = ['date', 'benchmark']
                                bench_prices['benchmark'] = bench_prices['benchmark'] / bench_prices['benchmark'].iloc[0] * 100
                                
                                eq_df['portfolio'] = eq_df['equity'] / eq_df['equity'].iloc[0] * 100
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=eq_df['date'], y=eq_df['portfolio'], name='Portfolio', line=dict(color='#00d4ff', width=2)))
                                fig.add_trace(go.Scatter(x=bench_prices['date'], y=bench_prices['benchmark'], name='Nifty 500', line=dict(color='#ff6b6b', width=2, dash='dot')))
                                fig.update_layout(height=350, template='plotly_white', title='Portfolio vs Nifty 500 (Base 100)')
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                fig = px.line(eq_df, x='date', y='equity', title='Equity Curve')
                                fig.update_layout(height=300, template='plotly_white')
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Trade log
                        trades_list = result.get('trades', [])
                        if trades_list:
                            trade_df = pd.DataFrame(trades_list)
                            with st.expander(f"📋 Trade Log ({len(trade_df)} trades)", expanded=False):
                                st.dataframe(trade_df, use_container_width=True, height=300)
                                
                                csv_data = trade_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Trade Log (CSV)",
                                    data=csv_data,
                                    file_name="my_portfolio_trades.csv",
                                    mime="text/csv",
                                    key="portfolio_trades_download"
                                )
                    else:
                        st.error(f"Backtest failed: {result.get('error', 'Unknown error')}")
            else:
                st.info("👆 Select at least one stock to build your portfolio")
# --- VIEW 2: SECTOR PULSE ---
elif page == "📊 Sector Pulse":
    
    st.markdown(page_header("📊 Sector Pulse", "Deep dive into sector performance and stock rotation dynamics"), unsafe_allow_html=True)
    
    # === SECTOR TIMING SIGNALS ===
    st.markdown("### 🎯 Sector Rotation & Timing Signals")
    
    # Calculate current market mood score for signals
    try:
        from utils.trend_engine import load_mood_history
        mood_history = load_mood_history()
        current_score = mood_history['avg_trend_score'].iloc[-1] if not mood_history.empty else 50
    except Exception:
        current_score = 50
        
    signal_cols = st.columns([1, 1])
    
    with signal_cols[0]:
        # Sector Rotation Signal Panel - Based on Correlation Analysis
        st.markdown("**📊 Current Rotation Signal**")
        
        if current_score < 40:
            st.success("🔥 **STRONG BUY**: Auto, Energy, Midcap")
            st.info("✅ **BUY**: Realty, PSE, Infra")
            st.warning("⏸️ **HOLD**: IT (wait for high mood)")
            st.caption("Expected 90D: Auto +10%, Energy +4%, Midcap +4%")
        elif current_score > 65:
            st.success("✅ **BUY**: IT sector (+8.7% expected)")
            st.error("❌ **AVOID**: Realty (-5%), Energy (-3%), Midcap (-1%)")
            st.warning("💰 Book profits in cyclicals")
        else:
            # 40-65 range
            st.info("⏸️ **HOLD**: Current positions")
            if current_score < 50:
                st.caption("Approaching BUY zone - watch Auto, Energy")
            else:
                st.caption("Approaching CAUTION zone - watch IT")
        st.caption(f"Market Mood Score: {current_score:.0f}/100 | Optimal Horizon: 60-90 days")

    with signal_cols[1]:
        # Enhanced Sector Signal Table based on analysis
        st.markdown("**📈 Sector Correlation Matrix**")
        
        # Tiered sector data from analysis
        sector_data = {
            "Sector": ["Auto 🏆", "Energy 🏆", "Midcap", "Realty", "IT 🔄"],
            "Corr": ["-0.72", "-0.78", "-0.81", "-0.64", "+0.26"],
            "@ <40": ["+10%", "+4%", "+4%", "+4%", "+5%"],
            "@ >70": ["+3%", "-3%", "-1%", "-5%", "+9%"],
        }
        st.dataframe(
            pd.DataFrame(sector_data),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Sector": st.column_config.TextColumn("Sector"),
                "Corr": st.column_config.TextColumn("Signal"),
                "@ <40": st.column_config.TextColumn("Low Mood"),
                "@ >70": st.column_config.TextColumn("High Mood"),
            }
        )
        
    st.markdown("---")
    
    # === SUB-INDUSTRY ROTATION HEATMAP ===
    st.markdown("### 🧬 Sub-Industry Rotation Matrix")
    st.caption("Granular tracking of capital flow across the 58 Sub-Industries of the Nifty 1000. Score 0–100 (percentile rank). 🟢 Leaders → 🟡 Mid → 🔴 Laggards.")
    
    sub_ind_file = "data/sub_industry_rotation.csv"
    if os.path.exists(sub_ind_file):
        try:
            sub_df = pd.read_csv(sub_ind_file)
            if not sub_df.empty:
                # Ensure score_0_100 exists (backward compat with old CSV)
                if 'score_0_100' not in sub_df.columns:
                    for date_val in sub_df['record_date'].unique():
                        mask = sub_df['record_date'] == date_val
                        rs_vals = sub_df.loc[mask, 'rs_momentum']
                        rs_min, rs_max = rs_vals.min(), rs_vals.max()
                        if rs_max > rs_min:
                            sub_df.loc[mask, 'score_0_100'] = ((rs_vals - rs_min) / (rs_max - rs_min) * 100).round(0).astype(int)
                        else:
                            sub_df.loc[mask, 'score_0_100'] = 50
                
                # Create month label for grouping (e.g., "Mar-26")
                sub_df['record_date'] = pd.to_datetime(sub_df['record_date'])
                sub_df['month_label'] = sub_df['record_date'].dt.strftime('%b-%y')
                
                # For each month, take the latest snapshot (in case multiple dates per month)
                sub_df = sub_df.sort_values('record_date')
                sub_df['month_key'] = sub_df['record_date'].dt.to_period('M')
                latest_per_month = sub_df.groupby(['month_key', 'sub_industry']).last().reset_index()
                
                # Pivot: rows = sub-industries, columns = months (use Period month_key for correct chronological ordering)
                # Last 12 months
                unique_months = sorted(latest_per_month['month_key'].unique())[-12:]
                latest_per_month = latest_per_month[latest_per_month['month_key'].isin(unique_months)]

                # Filter to only sub-industries present in the MOST RECENT month.
                # This drops ghost rows from old naming conventions (e.g. "Metals & Mining"
                # was later renamed to "Non-Ferrous Metals"), which would otherwise show as
                # mostly-white rows with data only on the left side of the heatmap.
                most_recent_month = max(unique_months)
                current_industries = set(
                    latest_per_month[latest_per_month['month_key'] == most_recent_month]['sub_industry'].unique()
                )
                latest_per_month = latest_per_month[latest_per_month['sub_industry'].isin(current_industries)]

                # Map each month_key Period to its month_label string (e.g. 'Jan-26')
                period_to_label = (
                    latest_per_month[['month_key', 'month_label']]
                    .drop_duplicates()
                    .set_index('month_key')['month_label']
                    .to_dict()
                )

                # Pivot on Period (guarantees chronological column order after sort)
                score_pivot = latest_per_month.pivot_table(
                    index='sub_industry', columns='month_key', values='score_0_100', aggfunc='last'
                )
                hover_pivot = latest_per_month.pivot_table(
                    index='sub_industry', columns='month_key', values='top_components', aggfunc='last'
                )

                # Sort columns chronologically (Periods sort naturally)
                score_pivot = score_pivot.sort_index(axis=1)
                hover_pivot = hover_pivot.sort_index(axis=1)

                # Forward-fill gaps so months with partial EOD data (e.g. Jan-26)
                # show the last known score instead of blank white cells
                score_pivot = score_pivot.ffill(axis=1)
                hover_pivot = hover_pivot.ffill(axis=1)

                # Rename Period columns -> readable month labels
                score_pivot.columns = [period_to_label.get(p, str(p)) for p in score_pivot.columns]
                hover_pivot.columns = [period_to_label.get(p, str(p)) for p in hover_pivot.columns]

                # Readable ordered list of month labels
                month_order = list(score_pivot.columns)

                # Sort rows by latest month score (best at top)
                if month_order:
                    last_month_col = month_order[-1]
                    score_pivot = score_pivot.sort_values(by=last_month_col, ascending=False, na_position='last')
                    hover_pivot = hover_pivot.reindex(score_pivot.index)

                # ── Two-tab view: sparkline table (primary) + heatmap (fallback) ──
                rot_tab1, rot_tab2 = st.tabs(["📋 Rotation Table", "🌡️ Heatmap"])

                with rot_tab1:
                    # Build one row per sub-industry
                    table_rows = []
                    for industry in score_pivot.index:
                        row_scores = score_pivot.loc[industry]
                        trend_vals = [int(round(float(v))) for v in row_scores.values if pd.notna(v)]

                        current_score = float(row_scores.iloc[-1]) if pd.notna(row_scores.iloc[-1]) else 0.0
                        prev_score    = float(row_scores.iloc[-2]) if len(row_scores) >= 2 and pd.notna(row_scores.iloc[-2]) else current_score
                        mom_change    = round(current_score - prev_score, 1)

                        leaders_raw = hover_pivot.loc[industry].iloc[-1]
                        leaders = leaders_raw if pd.notna(leaders_raw) else '—'

                        if current_score >= 70:
                            signal = '🟢 Leader'
                        elif current_score >= 40:
                            signal = '🟡 Mid'
                        else:
                            signal = '🔴 Laggard'

                        table_rows.append({
                            'Sub-Industry': industry,
                            'Signal':       signal,
                            'Score':        round(current_score),
                            'MoM Δ':        mom_change,
                            'Trend (12M)':  trend_vals,
                            'Leaders':      leaders,
                        })

                    rot_table_df = (pd.DataFrame(table_rows)
                                    .sort_values('Score', ascending=False)
                                    .reset_index(drop=True))

                    st.dataframe(
                        rot_table_df,
                        column_config={
                            'Sub-Industry': st.column_config.TextColumn("Sub-Industry", width="medium"),
                            'Signal':       st.column_config.TextColumn("Signal",       width="small"),
                            'Score':        st.column_config.ProgressColumn(
                                                "Score", min_value=0, max_value=100, format="%d"),
                            'MoM Δ':        st.column_config.NumberColumn(
                                                "MoM Δ", format="%+.0f",
                                                help="Month-over-month percentile rank change"),
                            'Trend (12M)':  st.column_config.LineChartColumn(
                                                "12M Trend", y_min=0, y_max=100),
                            'Leaders':      st.column_config.TextColumn("Leaders", width="medium"),
                        },
                        hide_index=True,
                        use_container_width=True,
                        height=min(len(rot_table_df) * 38 + 42, 950),
                    )

                    # Quick summary below table
                    if table_rows:
                        top3_t = rot_table_df.head(3)
                        bot3_t = rot_table_df.tail(3)
                        sc1, sc2 = st.columns(2)
                        with sc1:
                            st.markdown("**🟢 Current Leaders**")
                            for _, r in top3_t.iterrows():
                                st.caption(f"**{r['Sub-Industry']}** — {r['Score']}/100 | {r['Leaders']}")
                        with sc2:
                            st.markdown("**🔴 Current Laggards**")
                            for _, r in bot3_t.iterrows():
                                st.caption(f"**{r['Sub-Industry']}** — {r['Score']}/100 | {r['Leaders']}")

                with rot_tab2:
                    # ── Original heatmap (fallback / detail view) ──
                    hover_text = []
                    for idx in score_pivot.index:
                        row_text = []
                        for col in score_pivot.columns:
                            score_val = score_pivot.loc[idx, col]
                            leaders_h = hover_pivot.loc[idx, col] if pd.notna(hover_pivot.loc[idx, col]) else "N/A"
                            if pd.notna(score_val):
                                row_text.append(f"<b>{idx}</b><br>Score: {score_val:.0f}/100<br>Month: {col}<br>Leaders: {leaders_h}")
                            else:
                                row_text.append(f"<b>{idx}</b><br>No data<br>Month: {col}")
                        hover_text.append(row_text)

                    fig = go.Figure(data=go.Heatmap(
                        z=score_pivot.values,
                        x=score_pivot.columns.tolist(),
                        y=[s.replace('.NS', '') for s in score_pivot.index.tolist()],
                        colorscale=[
                            [0.0,  '#d32f2f'],
                            [0.25, '#ff7043'],
                            [0.5,  '#fdd835'],
                            [0.75, '#66bb6a'],
                            [1.0,  '#1b5e20'],
                        ],
                        zmin=0, zmax=100,
                        text=hover_text,
                        hovertemplate='%{text}<extra></extra>',
                        colorbar=dict(
                            title="Score",
                            tickvals=[0, 25, 50, 75, 100],
                            ticktext=["0 \U0001f534", "25", "50 \U0001f7e1", "75", "100 \U0001f7e2"],
                        ),
                        xgap=2, ygap=1,
                    ))
                    fig.update_layout(
                        height=max(600, len(score_pivot) * 22),
                        xaxis=dict(title="Month", side="top", tickangle=0),
                        yaxis=dict(title="", autorange="reversed", tickfont=dict(size=11)),
                        margin=dict(l=200, r=40, t=60, b=20),
                        template='plotly_dark',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sub-Industry data is empty. Run trading_engine.py first.")
        except Exception as e:
            st.error(f"Error loading Sub-Industry rotation: {e}")
    else:
        st.info("Sub-Industry rotation engine has not run yet. (Run trading_engine.py locally or via GitHub Action to generate backend matrix)")
    
    st.markdown("---")
    
    # === GRANULAR INDUSTRY SEASONALITY CALENDAR ===
    st.markdown("### 📅 Monthly Alpha Calendar (12-Month Lookahead)")
    st.caption("Which granular industries historically peak or bottom out in each calendar month? Price Lead/Lag delays are explicitly calculated (e.g. Lead: 3mo means buy 3 months *before* this peak).")
    
    alpha_cal_df = get_monthly_alpha_calendar()
    
    if not alpha_cal_df.empty:
        # Style the dataframe to highlight the current month row
        current_month_abbr = datetime.now().strftime("%b")
        
        def highlight_current_month(row):
            if row['📅 Month'] == current_month_abbr:
                return ['background-color: rgba(99, 91, 255, 0.15); font-weight: bold'] * len(row)
            return [''] * len(row)
            
        styled_cal = alpha_cal_df.style.apply(highlight_current_month, axis=1)
        
        st.dataframe(
            styled_cal,
            column_config={
                "📅 Month": st.column_config.TextColumn("Month", width="small"),
                "🟢 Historical Best (Accumulate)": st.column_config.TextColumn("🟢 Historical Best (Accumulate)", width="large"),
                "🔴 Historical Worst (Avoid/Lighten)": st.column_config.TextColumn("🔴 Historical Worst (Avoid/Lighten)", width="large"),
            },
            hide_index=True,
            use_container_width=True,
            height=450  # Enough height to show all 12 months without scrolling
        )
    else:
        st.info("Seasonality Engine offline. Missing granular_industry_analysis.csv")

    st.markdown("---")
    
    # === SECTOR-LEVEL CYCLE POSITION ===
    st.markdown("### 🔄 Sector Margin Cycle")
    st.caption("Which sectors are at which stage of their profitability cycle? Based on Operating Margin vs historical average.")
    
    # Sample 1 stock from each major cyclical sector
    cyclical_sectors = {
        "Iron & Steel": df[df['sector'].str.contains('Iron', case=False, na=False)]['ticker'].head(1).tolist(),
        "Cement": df[df['sector'].str.contains('Cement', case=False, na=False)]['ticker'].head(1).tolist(),
        "Power": df[df['sector'].str.contains('Power', case=False, na=False)]['ticker'].head(1).tolist(),
        "Oil & Gas": df[df['sector'].str.contains('Oil|Refin', case=False, na=False)]['ticker'].head(1).tolist(),
        "Auto": df[df['sector'].str.contains('Auto|Vehicle', case=False, na=False)]['ticker'].head(1).tolist(),
        "IT Services": df[df['sector'].str.contains('Software|IT', case=False, na=False)]['ticker'].head(1).tolist(),
    }
    
    sector_cycles = []
    with st.spinner("Analyzing sector margin cycles..."):
        for sector_name, tickers in cyclical_sectors.items():
            if tickers:
                cycle = calculate_cycle_position(tickers[0])
                if "error" not in cycle:
                    phase_emoji = {"EARLY_RECOVERY": "🟢", "MID_CYCLE": "🟡", "LATE_CYCLE": "🟠", "DOWNTURN": "🔴"}.get(cycle["phase"], "⚪")
                    sector_cycles.append({
                        "Sector": sector_name,
                        "Phase": f"{phase_emoji} {cycle['phase'].replace('_', ' ')}",
                        "OPM %": f"{cycle['current_margin']:.1f}%",
                        "vs Avg": f"{cycle['margin_vs_avg']:.2f}x",
                        "Trend": "↗" if cycle["trend"] == "rising" else "↘" if cycle["trend"] == "falling" else "→"
                    })
    
    if sector_cycles:
        sector_cycle_df = pd.DataFrame(sector_cycles)
        
        # Group by phase
        col_early, col_mid, col_late, col_down = st.columns(4)
        early = [s["Sector"] for s in sector_cycles if "RECOVERY" in s["Phase"]]
        mid = [s["Sector"] for s in sector_cycles if "MID" in s["Phase"]]
        late = [s["Sector"] for s in sector_cycles if "LATE" in s["Phase"]]
        down = [s["Sector"] for s in sector_cycles if "DOWNTURN" in s["Phase"]]
        
        with col_early:
            st.markdown("**🟢 Early Recovery**")
            st.caption("Buy opportunity")
            for s in early: st.write(f"• {s}")
            if not early: st.write("*None*")
        
        with col_mid:
            st.markdown("**🟡 Mid-Cycle**")
            st.caption("Hold/Ride")
            for s in mid: st.write(f"• {s}")
            if not mid: st.write("*None*")
        
        with col_late:
            st.markdown("**🟠 Late Cycle**")
            st.caption("Be cautious")
            for s in late: st.write(f"• {s}")
            if not late: st.write("*None*")
        
        with col_down:
            st.markdown("**🔴 Downturn**")
            st.caption("Avoid")
            for s in down: st.write(f"• {s}")
            if not down: st.write("*None*")
        
        with st.expander("📊 Detailed Sector Cycle Data"):
            st.dataframe(sector_cycle_df, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # === SECTOR DRILL-DOWN ===
    st.markdown("### 🔍 Sector Deep Dive")
    
    all_sectors = sorted(df['sector'].dropna().unique().tolist())
    selected_sector = st.selectbox("Select a Sector to Analyze", all_sectors, key="sector_pulse_select")
    
    # Filter stocks for selected sector
    sector_stocks = df[df['sector'] == selected_sector].copy()
    
    if sector_stocks.empty:
        st.warning(f"No stocks found in {selected_sector}")
    else:
        # Sector stats
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        s_col1.metric("📊 Stocks", len(sector_stocks))
        s_col2.metric("📈 Avg Score", f"{sector_stocks['overall'].mean():.1f}")
        s_col3.metric("🚀 Avg Trend", f"{sector_stocks['trend_score'].mean():.0f}")
        uptrend_pct = len(sector_stocks[sector_stocks['trend_signal'].isin(['STRONG UPTREND', 'UPTREND'])]) / len(sector_stocks) * 100
        s_col4.metric("💹 % Uptrending", f"{uptrend_pct:.0f}%")
        
        # Top performers in sector
        st.markdown(f"#### 🏆 Top Performers in {selected_sector}")
        top_in_sector = sector_stocks.nlargest(10, 'overall')[['ticker', 'name', 'price', 'overall', 'trend_score', 'trend_signal', 'return_1m', 'return_3m']].copy()
        top_in_sector['screener_link'] = "https://www.screener.in/company/" + top_in_sector['ticker'].str.replace('.NS', '', regex=False) + "/"
        st.dataframe(
            top_in_sector[['screener_link', 'name', 'price', 'overall', 'trend_score', 'trend_signal', 'return_1m', 'return_3m']],
            column_config={
                "screener_link": st.column_config.LinkColumn("Ticker", display_text=r"https://www\.screener\.in/company/(.*?)/"),
                "name": "Company",
                "price": st.column_config.NumberColumn("Price", format="₹%.2f"),
                "overall": st.column_config.ProgressColumn("Score", min_value=0, max_value=10, format="%.1f"),
                "trend_score": st.column_config.ProgressColumn("Trend", min_value=0, max_value=100),
                "trend_signal": "Signal",
                "return_1m": st.column_config.NumberColumn("1M Return", format="%.1f%%"),
                "return_3m": st.column_config.NumberColumn("3M Return", format="%.1f%%"),
            },
            hide_index=True
        )
        
        # === MARGIN CYCLE POSITION ===
        st.markdown("---")
        st.markdown("### 🔄 Margin Cycle Position")
        st.caption("Where are stocks in their profitability cycle? Based on Operating Margin vs historical average.")
        
        with st.expander("📖 Cycle Phases Explained", expanded=False):
            st.markdown("""
            | Phase | Meaning | Action |
            |-------|---------|--------|
            | 🟢 **EARLY_RECOVERY** | Margins below avg but improving | 💡 Potential BUY |
            | 🟡 **MID_CYCLE** | Margins at/above avg, stable | ✅ HOLD/RIDE |
            | 🟠 **LATE_CYCLE** | Margins above avg but falling | ⚠️ BE CAUTIOUS |
            | 🔴 **DOWNTURN** | Margins below avg, still falling | ❌ AVOID |
            """)
        
        # Calculate cycle for each stock (limited to first 15 to avoid API limits)
        cycle_data = []
        sample_tickers = sector_stocks.nlargest(15, 'overall')['ticker'].tolist()
        
        with st.spinner(f"Analyzing margin cycles for {len(sample_tickers)} stocks..."):
            for ticker in sample_tickers:
                cycle = calculate_cycle_position(ticker)
                if "error" not in cycle:
                    cycle_data.append({
                        "Ticker": ticker.replace(".NS", ""),
                        "Phase": cycle["phase"],
                        "OPM %": cycle["current_margin"],
                        "vs Avg": f"{cycle['margin_vs_avg']:.2f}x",
                        "Trend": "↗" if cycle["trend"] == "rising" else "↘" if cycle["trend"] == "falling" else "→",
                        "Cyclical?": "Yes" if cycle.get("is_cyclical", False) else "No"
                    })
        
        if cycle_data:
            cycle_df = pd.DataFrame(cycle_data)
            
            # Add phase emojis
            phase_emoji = {
                "EARLY_RECOVERY": "🟢",
                "MID_CYCLE": "🟡",
                "LATE_CYCLE": "🟠",
                "DOWNTURN": "🔴",
            }
            cycle_df["Phase"] = cycle_df["Phase"].apply(lambda x: f"{phase_emoji.get(x, '⚪')} {x}")
            
            # Show phase distribution
            phase_counts = cycle_df["Phase"].value_counts()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🟢 Recovery", len([p for p in cycle_df["Phase"] if "RECOVERY" in p]))
            c2.metric("🟡 Mid-Cycle", len([p for p in cycle_df["Phase"] if "MID_CYCLE" in p]))
            c3.metric("🟠 Late Cycle", len([p for p in cycle_df["Phase"] if "LATE" in p]))
            c4.metric("🔴 Downturn", len([p for p in cycle_df["Phase"] if "DOWNTURN" in p]))
            
            st.dataframe(cycle_df, hide_index=True, use_container_width=True)
        else:
            st.info("Cycle data not available for this sector (may lack operating income data)")
        
        st.markdown("---")
        
        # === STOCK ROTATION CHART ===
        st.markdown("### 🔄 Stock Rotation (Within Sector)")
        st.caption("X: 3-Month Return | Y: 1-Month Return | Color: Trend Signal | Size: Market Cap")
        

        
        # Ensure return columns exist, fill missing with 0
        if 'return_1m' not in sector_stocks.columns:
            sector_stocks['return_1m'] = 0
        if 'return_3m' not in sector_stocks.columns:
            sector_stocks['return_3m'] = 0
        
        # Fill NaN values with 0 for returns and min market cap for size
        sector_stocks['return_1m'] = sector_stocks['return_1m'].fillna(0)
        sector_stocks['return_3m'] = sector_stocks['return_3m'].fillna(0)
        sector_stocks['marketCap'] = sector_stocks['marketCap'].fillna(1000000000)  # Default 1000 Cr
        
        # Filter stocks with valid data
        chart_df = sector_stocks[
            (sector_stocks['marketCap'] > 0)
        ].copy()
        
        if len(chart_df) >= 3:
            # Map trend signals to colors
            color_map = {
                'STRONG UPTREND': '#00C853',  # Green
                'UPTREND': '#69F0AE',         # Light green
                'NEUTRAL': '#FFD600',         # Yellow
                'DOWNTREND': '#FF6D00',       # Orange
                'STRONG DOWNTREND': '#D50000' # Red
            }
            
            fig = px.scatter(
                chart_df,
                x='return_3m',
                y='return_1m',
                size='marketCap',
                color='trend_signal',
                hover_name='ticker',
                hover_data={'name': True, 'overall': ':.1f', 'return_1m': ':.1f', 'return_3m': ':.1f'},
                color_discrete_map=color_map,
                labels={
                    'return_3m': '3-Month Return (%)',
                    'return_1m': '1-Month Return (%)',
                    'trend_signal': 'Trend'
                },
            )
            
            # Add quadrant lines
            fig.add_hline(y=0, line_dash="solid", line_width=2, line_color="#E0E0E0", opacity=1.0)
            fig.add_vline(x=0, line_dash="solid", line_width=2, line_color="#E0E0E0", opacity=1.0)
            
            # Dynamically calculate label positions based on max axis bounds
            x_max = max(abs(chart_df['return_3m'].min()), abs(chart_df['return_3m'].max()), 10)
            y_max = max(abs(chart_df['return_1m'].min()), abs(chart_df['return_1m'].max()), 5)
            
            # Update axes to fit data tightly with padding
            fig.update_xaxes(range=[-x_max*1.15, x_max*1.15])
            fig.update_yaxes(range=[-y_max*1.15, y_max*1.15])
            
            # Add light background quadrant colors using dynamic extents
            fig.add_shape(type="rect", x0=0, y0=0, x1=x_max*2, y1=y_max*2, fillcolor="#00C853", opacity=0.03, line_width=0, layer="below")
            fig.add_shape(type="rect", x0=-x_max*2, y0=0, x1=0, y1=y_max*2, fillcolor="#2196F3", opacity=0.03, line_width=0, layer="below")
            fig.add_shape(type="rect", x0=0, y0=-y_max*2, x1=x_max*2, y1=0, fillcolor="#FF9800", opacity=0.03, line_width=0, layer="below")
            fig.add_shape(type="rect", x0=-x_max*2, y0=-y_max*2, x1=0, y1=0, fillcolor="#F44336", opacity=0.03, line_width=0, layer="below")
            
            fig.add_annotation(x=x_max*0.7, y=y_max*0.7, text="🚀 Winners", showarrow=False, font=dict(color="green", size=12))
            fig.add_annotation(x=-x_max*0.7, y=y_max*0.7, text="📈 Turnarounds", showarrow=False, font=dict(color="blue", size=12))
            fig.add_annotation(x=x_max*0.7, y=-y_max*0.7, text="⚠️ Fading", showarrow=False, font=dict(color="orange", size=12))
            fig.add_annotation(x=-x_max*0.7, y=-y_max*0.7, text="🔴 Laggards", showarrow=False, font=dict(color="red", size=12))
            
            fig.update_layout(
                height=450,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                template='plotly_white',
            )
            
            st.plotly_chart(fig, use_container_width=True, key="stock_rotation_chart")
        else:
            st.info(f"Not enough stocks with valid data in this sector for rotation chart. Found {len(chart_df)} stocks.")

# --- VIEW 3: TIME TRENDS ---
elif page == "⏳ Time Trends":
    from utils.trend_engine import calculate_sector_history, calculate_stock_trend_history
    from utils.visuals import chart_sector_rotation, chart_stock_cycle
    
    st.markdown(page_header("⏳ Time Travel Trends", "Visualize Market Cycles and Historical Momentum"), unsafe_allow_html=True)
    
    tab_sector, tab_stock = st.tabs(["🔄 Sector Cycles", "📈 Stock Trend History"])
    
    # 1. Sector Rotation Tab
    with tab_sector:
        st.markdown("### Sector Rotation: Relative Strength vs Nifty 500")
        st.caption("Tracks how sectors have performed relative to the market over the last year. Lines going UP are outperforming.")
        
        with st.spinner("Calculating Sector Histories... (This may take a moment)"):
            sector_hist = calculate_sector_history(df)
            
        if not sector_hist.empty:
            fig_sector = chart_sector_rotation(sector_hist)
            if fig_sector:
                st.plotly_chart(fig_sector, use_container_width=True)
            else:
                st.warning("Insufficient data to chart sector rotation.")
        else:
            st.error("Failed to load historical sector data.")

        st.markdown("---")
        st.subheader("🔄 Sector Stock Leadership")
        st.caption("Compare performance of specific stocks within a sector to spot rotation.")
        
        # Sector Stock Comparison UI
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            # Get list of sectors
            all_sectors_tt = sorted(df['sector'].unique().tolist())
            compare_sector = st.selectbox("Select Sector", all_sectors_tt, key="comp_sector_sel")
            
        with sc2:
            # Get stocks in that sector
            sector_stocks = df[df['sector'] == compare_sector]['ticker'].tolist()
            # Default to top 5 by trend score if available, else first 5
            default_stocks = sector_stocks[:5]
            compare_tickers = st.multiselect("Select Stocks", sector_stocks, default=default_stocks, key="comp_ticker_sel")
            
        with sc3:
            compare_period = st.selectbox("Timeframe", ["1y", "6mo", "3mo", "1mo"], index=0, key="comp_period_sel")
        
        if compare_tickers:
            with st.spinner("Fetching stock histories..."):
                comp_data = {}
                for t in compare_tickers:
                    # Use cache if possible, or fetch
                    h = get_stock_history(t, period=compare_period)
                    if not h.empty:
                        comp_data[t] = h['Close']
                
                if comp_data:
                    st.plotly_chart(chart_relative_performance(comp_data), use_container_width=True)
                else:
                    st.warning("No historical data available for selected stocks.")

    # 2. Stock Cycle Tab
    with tab_stock:
        st.markdown("### Historical Momentum Analysis")
        st.caption("Reconstructs the 'Trend Score' for the past 2 years to visualize signals and cycles.")
        
        # === SECTOR-BASED STOCK SELECTION ===
        filter_col1, filter_col2 = st.columns([1, 2])
        
        with filter_col1:
            # Sector filter
            all_sectors = sorted(df['sector'].dropna().unique().tolist())
            sector_options = ["All Sectors"] + all_sectors
            selected_sector_for_trend = st.selectbox(
                "Filter by Sector", 
                sector_options, 
                key="trend_sector_filter"
            )
        
        with filter_col2:
            # Filter tickers based on sector
            if selected_sector_for_trend == "All Sectors":
                filtered_tickers = TICKERS
                filtered_df_for_trend = df
            else:
                filtered_df_for_trend = df[df['sector'] == selected_sector_for_trend]
                filtered_tickers = filtered_df_for_trend['ticker'].tolist()
            
            if not filtered_tickers:
                st.warning(f"No stocks found in {selected_sector_for_trend}")
                trend_ticker = None
            else:
                # Try to grab from session state if available
                quick_ticker = st.session_state.get('quick_dive_ticker')
                default_index = 0
                if quick_ticker and quick_ticker in filtered_tickers:
                    default_index = filtered_tickers.index(quick_ticker)
                
                # Create display options with name
                def format_ticker(ticker):
                    stock_row = df[df['ticker'] == ticker]
                    if not stock_row.empty:
                        name = stock_row['name'].values[0]
                        trend_score = stock_row['trend_score'].values[0]
                        return f"{ticker} - {name} (Trend: {trend_score:.0f})"
                    return ticker
                
                trend_ticker = st.selectbox(
                    "Select Stock for Cycle Analysis", 
                    filtered_tickers, 
                    index=default_index, 
                    format_func=format_ticker,
                    key="trend_ticker_select"
                )
        
        # Show sector stats if a specific sector is selected
        if selected_sector_for_trend != "All Sectors" and not filtered_df_for_trend.empty:
            st.markdown(f"#### 📊 {selected_sector_for_trend} - Quick Stats")
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            stat_col1.metric("Stocks in Sector", len(filtered_df_for_trend))
            stat_col2.metric("Avg Trend Score", f"{filtered_df_for_trend['trend_score'].mean():.0f}")
            
            # Top and bottom performers
            if len(filtered_df_for_trend) > 1:
                top_stock = filtered_df_for_trend.loc[filtered_df_for_trend['trend_score'].idxmax()]
                worst_stock = filtered_df_for_trend.loc[filtered_df_for_trend['trend_score'].idxmin()]
                stat_col3.metric("🏆 Strongest", top_stock['ticker'], f"{top_stock['trend_score']:.0f}")
                stat_col4.metric("📉 Weakest", worst_stock['ticker'], f"{worst_stock['trend_score']:.0f}")
        
        st.markdown("---")
            
        if trend_ticker:
            # View Mode Toggle
            view_mode = st.radio("Analysis Mode", ["Standard Trend Cycle", "Multi-Factor Analysis"], horizontal=True, label_visibility="collapsed")
            
            if view_mode == "Standard Trend Cycle":
                with st.spinner(f"Reconstructing Trend History for {trend_ticker}..."):
                    trend_df = calculate_stock_trend_history(trend_ticker)
                
                if not trend_df.empty:
                    # Stats
                    last_row = trend_df.iloc[-1]
                    
                    # Show current status with color coding
                    signal = last_row['trend_signal']
                    score = last_row['trend_score']
                    
                    if signal in ['STRONG UPTREND', 'UPTREND']:
                        signal_color = '#00C853'
                    elif signal == 'NEUTRAL':
                        signal_color = '#FFD600'
                    else:
                        signal_color = '#D50000'
                    
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid {signal_color};">
                        <h4 style="margin: 0;">Current Status: <span style="color: {signal_color};">{signal}</span></h4>
                        <p style="margin: 5px 0 0 0; color: #888;">Trend Score: {score:.0f}/100</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    fig_cycle = chart_stock_cycle(trend_df)
                    if fig_cycle:
                        st.plotly_chart(fig_cycle, use_container_width=True)
                else:
                    st.error(f"Could not load history for {trend_ticker}")
            
            else: # Multi-Factor Analysis
                from utils.score_history import calculate_historical_scores
                from utils.scoring import calculate_scores

                
                # Defined locally to avoid Streamlit caching issues with utils module
                def detect_divergences(df, price_col='Close', indicator_col='momentum_score_hist', window=5):
                    if df.empty or len(df) < window * 2:
                        return pd.DataFrame({'div_bull': pd.Series(dtype=float), 'div_bear': pd.Series(dtype=float)})

                    d = df.copy()
                    d['min_local'] = d[price_col].rolling(window=window, center=True).min()
                    d['max_local'] = d[price_col].rolling(window=window, center=True).max()
                    
                    is_min = (d[price_col] == d['min_local'])
                    is_max = (d[price_col] == d['max_local'])
                    
                    min_indices = d.index[is_min]
                    max_indices = d.index[is_max]
                    
                    d['div_bull'] = np.nan
                    d['div_bear'] = np.nan
                    
                    for i in range(1, len(min_indices)):
                        curr_idx = min_indices[i]
                        prev_idx = min_indices[i-1]
                        
                        p_curr = d.loc[curr_idx, price_col]
                        p_prev = d.loc[prev_idx, price_col]
                        i_curr = d.loc[curr_idx, indicator_col]
                        i_prev = d.loc[prev_idx, indicator_col]
                        
                        if p_curr < p_prev and i_curr > i_prev:
                             d.loc[curr_idx, 'div_bull'] = p_curr
                             
                    for i in range(1, len(max_indices)):
                        curr_idx = max_indices[i]
                        prev_idx = max_indices[i-1]
                        
                        p_curr = d.loc[curr_idx, price_col]
                        p_prev = d.loc[prev_idx, price_col]
                        i_curr = d.loc[curr_idx, indicator_col]
                        i_prev = d.loc[prev_idx, indicator_col]
                        
                        if p_curr > p_prev and i_curr < i_prev:
                            d.loc[curr_idx, 'div_bear'] = p_curr
                            
                    return d[['div_bull', 'div_bear']]

                
                # Fetch history for calculation
                hist_data = yf.Ticker(trend_ticker).history(period="2y")
                
                if not hist_data.empty:
                    with st.spinner("Calculating multi-factor score history..."):
                        # 1. Calculate Historical Technicals
                        full_hist = calculate_historical_scores(hist_data)
                        
                        # 2. Get Current Fundamental Scores (Context)
                        # Try to get from cache first
                        stock_data = df[df['ticker'] == trend_ticker]
                        if not stock_data.empty:
                             stock_data = stock_data.iloc[0].to_dict()
                             scores = calculate_scores(stock_data)
                        else:
                             # Fallback defaults
                             scores = {'quality': 5, 'value': 5, 'growth': 5, 'momentum': 5, 'volume_signal_score': 5}
                        
                        # --- ROW 1: COMPONENT BREAKDOWN (Current) ---
                        st.markdown("### 🧩 Current Score Breakdown")
                        c1, c2, c3, c4, c5 = st.columns(5)
                        
                        c1.metric("Quality", f"{scores['quality']}/10", 
                                 help="Profitability, ROE, Margins")
                        c2.metric("Value", f"{scores['value']}/10",
                                 help="PE, PB, PEG vs Sector")
                        c3.metric("Growth", f"{scores['growth']}/10",
                                 help="Rev & Earnings Growth")
                        c4.metric("Momentum", f"{scores['momentum']}/10",
                                 help="Price Strength (1W/1M/3M)")
                        c5.metric("Volume", f"{scores['volume_signal_score']}/10",
                                 help="Accumulation vs Distribution")
                        
                        st.markdown("---")
                        
                        # --- ROW 2: HISTORICAL TRENDS ---
                        st.markdown("### 📈 Price vs Volume & Momentum History")
                        st.caption("Compare Price action with Momentum and Volume health to spot divergences.")
                        
                        # Tech Chart: Momentum vs Volume vs Price
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                           vertical_spacing=0.05, row_heights=[0.7, 0.3])
                        
                        # Main: Price & Trend
                        # Price Line
                        fig.add_trace(go.Scatter(x=full_hist.index, y=full_hist['Close'], name="Price", 
                                                line=dict(color='white', width=1)), row=1, col=1)
                        
                        # Divergence Signals
                        div_data = detect_divergences(full_hist)
                        if not div_data['div_bull'].isnull().all():
                            fig.add_trace(go.Scatter(
                                x=div_data.index, y=div_data['div_bull'],
                                mode='markers', marker=dict(color='#00E676', size=10, symbol='circle'),
                                name="Bullish Div"
                            ), row=1, col=1)
                            
                        if not div_data['div_bear'].isnull().all():
                            fig.add_trace(go.Scatter(
                                x=div_data.index, y=div_data['div_bear'],
                                mode='markers', marker=dict(color='#FF1744', size=10, symbol='circle'),
                                name="Bearish Div"
                            ), row=1, col=1)

                        # --- EMA Crossovers ---
                        # Use MA columns from history (Pre-calculated in score_history.py)
                        if 'ma50' in full_hist.columns and 'ma200' in full_hist.columns:
                            ma50 = full_hist['ma50']
                            ma200 = full_hist['ma200']
                            
                            # Golden Cross: 50 cross > 200
                            golden_cross = (ma50 > ma200) & (ma50.shift(1) <= ma200.shift(1))
                            # Death Cross: 50 cross < 200 
                            death_cross = (ma50 < ma200) & (ma50.shift(1) >= ma200.shift(1))
                            
                            golden_pts = full_hist[golden_cross]
                            death_pts = full_hist[death_cross]
                            
                            if not golden_pts.empty:
                                fig.add_trace(go.Scatter(
                                    x=golden_pts.index, y=golden_pts['Close'],
                                    mode='markers', marker=dict(color='#FFD700', size=14, symbol='star'),
                                    name="Golden Cross (50>200)"
                                ), row=1, col=1)
                                
                            if not death_pts.empty:
                                fig.add_trace(go.Scatter(
                                    x=death_pts.index, y=death_pts['Close'],
                                    mode='markers', marker=dict(color='black', line=dict(color='white', width=1), size=12, symbol='x'),
                                    name="Death Cross (50<200)"
                                ), row=1, col=1)
                        
                        # Trend Score Overlay (Secondary Axis logic needed? Or just normalize/color background?)
                        # Let's use background color for Trend Score zones or just add it as a line on secondary axis?
                        # Using simple price for now to keep it clean, maybe color price line by trend?
                        
                        # Sub: Momentum & Volume Scores
                        fig.add_trace(go.Scatter(x=full_hist.index, y=full_hist['momentum_score_hist'], name="Momentum (0-10)",
                                                line=dict(color='#2962FF', width=2)), row=2, col=1)
                        fig.add_trace(go.Scatter(x=full_hist.index, y=full_hist['volume_score_hist'], name="Volume (0-10)",
                                                line=dict(color='#FFD600', width=2)), row=2, col=1)
                        
                        # Reference lines for subchart
                        fig.add_hline(y=5, line_dash="dot", line_color="gray", row=2, col=1)
                        fig.add_hline(y=8, line_dash="dot", line_color="green", row=2, col=1, opacity=0.3)
                        fig.add_hline(y=2, line_dash="dot", line_color="red", row=2, col=1, opacity=0.3)
                        
                        fig.update_layout(height=600, template="plotly_white", hovermode="x unified")
                        fig.update_yaxes(title_text="Price", row=1, col=1)
                        fig.update_yaxes(title_text="Score (0-10)", range=[0, 11], row=2, col=1)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # --- NEW: SMART VOLUME ANALYSIS ---
                        st.markdown("### 📊 Smart Volume Analysis")
                        st.caption("Detailed volume flow analysis with accumulation/distribution signals.")
                        
                        from utils.visuals import chart_volume_analysis
                        vol_fig = chart_volume_analysis(full_hist)
                        if vol_fig:
                             st.plotly_chart(vol_fig, use_container_width=True)

                        # --- ROW 3: INSIGHTS ---
                        with st.expander("🔎 Analysis Guide", expanded=False):
                            st.markdown("""
                            **Divergence Signals:**
                            1. **Bullish Divergence**: Price makes new Low, but Momentum/Volume Score makes Higher Low.
                            2. **Bearish Divergence**: Price makes new High, but Momentum/Volume Score makes Lower High.
                            3. **Volume Climax**: Extremely high volume score (9-10) often marks a turning point or breakout.
                            """)
                else:
                    st.error(f"No history found for {trend_ticker}")

# --- VIEW 4: DEEP DIVE ---
elif page == "📉 Deep Dive":

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

# --- VIEW: ALERTS CONFIGURATION ---
if page == "⚠️ Alerts Configuration":
    st.title("⚠️ Alerts Configuration")
    st.info("🚧 Granular Alert Management is under development.")
    st.markdown("""
    Currently, alerts are managed via the **Return Tracker**.
    
    **Planned Features:**
    - [ ] Global Stop Loss Policies
    - [ ] Telegram/Email Integration Settings
    - [ ] Volatility Alerts
    """)

# --- VIEW: NOTES ---
if page == "📝 Notes":
    st.title("📝 Trading Journal")
    st.markdown("Use this space to log your daily observations and strategy ideas.")
    
    if 'trading_notes' not in st.session_state:
        st.session_state['trading_notes'] = ""
        
    notes = st.text_area("Daily Log", value=st.session_state['trading_notes'], height=400, placeholder="- Observed strong volume in IT sector...\n- Planning to enter RELIANCE above 2500...")
    
    if st.button("💾 Save Journal"):
        st.session_state['trading_notes'] = notes
        st.success("Notes saved successfully! (Session storage)")




# --- VIEW: TURNAROUND RADAR ---
elif page == "\U0001f3af Turnaround Radar":
    import plotly.graph_objects as go

    st.markdown(page_header(
        "\U0001f3af Institutional Turnaround Radar",
        "Beaten-down Nifty 1000 stocks showing early institutional accumulation \u2014 before V21 qualification."
    ), unsafe_allow_html=True)

    WATCHLIST_CSV = "data/turnaround_watchlist.csv"

    @st.cache_data(ttl=3600)
    def load_turnaround_watchlist():
        if not os.path.exists(WATCHLIST_CSV):
            return pd.DataFrame()
        return pd.read_csv(WATCHLIST_CSV)

    wdf = load_turnaround_watchlist()

    # Merge sector from main market df (not in watchlist CSV)
    if not wdf.empty and "Sector" not in wdf.columns and "sector" in df.columns:
        _sec_map = df.set_index("ticker")["sector"].to_dict()
        wdf["Sector"] = wdf["Ticker"].map(_sec_map).fillna("Unknown")
    elif not wdf.empty and "Sector" not in wdf.columns:
        wdf["Sector"] = "Unknown"

    regime = st.session_state.get('market_regime', 'UNKNOWN')
    if regime in ['BEAR', 'CRISIS']:
        st.error(f"🚨 **{regime} REGIME ACTIVE** 🚨\n\nTurnaround Module Suspended. The system is structurally blocking knife-catches during severe market drawdowns to preserve capital.")
        st.stop()

    if wdf.empty:
        st.warning("Watchlist not generated yet. Run `python turnaround_screener.py` locally or wait for the next GitHub Actions daily run.")
        st.code("python turnaround_screener.py", language="bash")
        st.stop()

    alert_n = int((wdf["Tier"] == "ALERT").sum())
    ready_n = int((wdf["Tier"] == "READY").sum())
    watch_n = int((wdf["Tier"] == "WATCH").sum())

    # Colour language: green = ready to fire, gold = warming up, blue = early radar ping
    TIER_COLORS = {
        "ALERT": "#00C853",  # Emerald green  — imminent V21, ready to fire
        "READY": "#FFB300",  # Amber gold     — RS velocity confirmed, getting warm
        "WATCH": "#42A5F5",  # Sky blue       — early accumulation ping
    }

    # --- Tier cards ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("\U0001f7e2 ALERT", alert_n, help="IAS 80+: RS21+RS63 turning. Strong liq floor. Imminent V21 — ready to fire.")
    c2.metric("\U0001f7e1 READY", ready_n, help="IAS 60-79: Liq stable, RS velocity confirmed. Getting warm.")
    c3.metric("\U0001f535 WATCH", watch_n, help="IAS 35-59: Early institutional ping. Set MA50 alert.")
    c4.metric("Total Pool", len(wdf))

    st.info("**How to use:** These are BELOW MA50 \u2014 do NOT buy. Set a price alert at each stock's MA50 level. When crossed with CompRS > 0, re-check the V21 scanner for fast-track to portfolio.")

    st.markdown("---")

    # --- Filters ---
    fcols = st.columns([1, 1, 1, 1.5, 1.5])
    tier_filter    = fcols[0].multiselect("Tier",  ["ALERT","READY","WATCH"], default=["ALERT","READY","WATCH"])
    cycle_filter   = fcols[1].multiselect("Cycle", ["LONG","MID","SHORT"],    default=["LONG","MID","SHORT"])
    min_ias        = fcols[2].slider("Min IAS", 35, 90, 35)
    _all_sectors   = sorted(wdf["Sector"].dropna().unique().tolist()) if "Sector" in wdf.columns else []
    sector_filter  = fcols[3].multiselect("Sector", _all_sectors, default=_all_sectors, placeholder="All sectors")
    search         = fcols[4].text_input("Search ticker or sub-industry")

    # --- Range filters (numeric columns) ---
    _WL_RF_DEFS = [
        # (column,          label,              step)
        ("IAS",            "IAS Score",         1.0),
        ("CompRS",         "CompRS",            0.01),
        ("RS21",           "RS21 %",            1.0),
        ("RS63",           "RS63 %",            1.0),
        ("Off_MA50",       "Off MA50 %",        1.0),
        ("Off_52W_High",   "Off 52W High %",    1.0),
        ("V21_CRS_Gap",    "V21 CRS Gap",       0.01),
        ("V21_MA50_Gap",   "V21 MA50 Gap %",    1.0),
        ("Liq5Cr",         "Liq 5Cr ₹Cr",      5.0),
        ("LiqFromLow",     "Liq From Low ×",    0.5),
        ("VolQuality",     "Vol Quality",       0.05),
    ]

    with st.expander("🎚️ Range Filters", expanded=False):
        import math as _wl_math
        _wl_range_vals: dict = {}
        _wl_rf_avail = [(c, lbl, step) for c, lbl, step in _WL_RF_DEFS if c in wdf.columns]
        _WL_N_COLS = 4
        for _ri in range(0, len(_wl_rf_avail), _WL_N_COLS):
            _row = _wl_rf_avail[_ri: _ri + _WL_N_COLS]
            _rcols = st.columns(_WL_N_COLS)
            for _j, (col, lbl, step) in enumerate(_row):
                _s = pd.to_numeric(wdf[col], errors="coerce").dropna()
                if len(_s) < 1:
                    continue
                _lo = _wl_math.floor(float(_s.min()) / step) * step
                _hi = _wl_math.ceil( float(_s.max()) / step) * step
                if _hi <= _lo:
                    _hi = _lo + step
                with _rcols[_j]:
                    _sel = st.slider(lbl, min_value=_lo, max_value=_hi,
                                     value=(_lo, _hi), step=step, key=f"wl_rf_{col}")
                    _wl_range_vals[col] = _sel

    fdf = wdf[
        wdf["Tier"].isin(tier_filter) &
        wdf["Cycle"].isin(cycle_filter) &
        (wdf["IAS"] >= min_ias) &
        (wdf["Sector"].isin(sector_filter) if sector_filter else pd.Series(True, index=wdf.index))
    ].copy()

    # Apply numeric range filters
    for _col, (_lo_sel, _hi_sel) in _wl_range_vals.items():
        _num = pd.to_numeric(fdf[_col], errors="coerce")
        fdf = fdf[_num.between(_lo_sel, _hi_sel, inclusive="both") | _num.isna()]

    if search:
        fdf = fdf[
            fdf["Ticker"].str.contains(search.upper(), na=False) |
            fdf["Sub_Industry"].str.contains(search, case=False, na=False)
        ]

    # --- Lifecycle explainer ---
    with st.expander("ℹ️ How stocks enter, graduate & drop from the IAS watchlist"):
        st.markdown("""
**Entry — IAS ≥ 35, price below MA50**

A stock enters the watchlist when `IAS ≥ 35`. IAS (Institutional Accumulation Score) combines
RS velocity, liquidity surge from the low, and volume quality — all must confirm simultaneous
early buying *while the stock is still below MA50* (not yet a V21 breakout, just radar ping).

| Tier | IAS range | Meaning |
|---|---|---|
| 🔵 WATCH | 35–59 | Early institutional ping — set a price alert at MA50 |
| 🟡 READY | 60–79 | RS velocity confirmed, liquidity floor stable — getting warm |
| 🟢 ALERT | 80+ | RS21 + RS63 both turning, strong liq floor — imminent V21, ready to fire |

**Graduation → GRADUATED (locked permanently)**

When **both** conditions are met on any screener run:
1. `Off_MA50 ≥ 0` — price has crossed *above* the 50-day MA
2. `CompRS > 0` — stock is outperforming the benchmark

This is the V21 breakout trigger. The stock is logged as GRADUATED in the IAS Signal Log
and promoted to the main portfolio scanner. Status can **never** be reverted — once a grad, always a grad.

**V21 CRS Gap & V21 MA50 Gap** tell you exactly how far each stock is from graduating:
- `V21 CRS Gap = 0` → CompRS is already positive (half-done)
- `V21 MA50 Gap % = 0` → price is at MA50 (half-done)
- Both zero → next run may graduate it

**Drop → DROPPED**

When the stock disappears from the daily scan (IAS falls below 35 — accumulation signal evaporated)
**and** it hasn't already GRADUATED. Typically means the smart-money move reversed or was a false signal.
DROPPED stocks are tracked in the Signal Log for post-mortem analysis.

**Typical holding time:** 2–12 weeks from ALERT to graduation. WATCH stocks can take months.
        """)

    # --- Watchlist table ---
    st.markdown(f"### Watchlist ({len(fdf)} stocks)")

    # Build screener.in links using the same pattern as other tabs
    display_df = fdf.copy().sort_values("IAS", ascending=False).reset_index(drop=True)
    display_df["screener_link"] = "https://www.screener.in/company/" + display_df["Ticker"].str.replace(r"\.(NS|BO)$", "", regex=True) + "/"

    display_cols = ["screener_link","Sub_Industry","Cycle","CMP","Off_52W_High","RS21","RS63",
                    "CompRS","Liq5Cr","LiqFromLow","VolQuality","IAS","Tier","Off_MA50","V21_CRS_Gap","V21_MA50_Gap"]
    available = [c for c in display_cols if c in display_df.columns]

    st.dataframe(
        display_df[available],
        column_config={
            "screener_link": st.column_config.LinkColumn("Ticker", display_text=r"https://www\.screener\.in/company/(.*?)/", width=110),
            "Sub_Industry": st.column_config.TextColumn("Sub-Industry", width=160),
            "Cycle":        st.column_config.TextColumn("Cycle", width=55),
            "CMP":          st.column_config.NumberColumn("CMP", format="%.1f"),
            "Off_52W_High": st.column_config.NumberColumn("Off52W%", format="%.1f"),
            "Off_MA50":     st.column_config.NumberColumn("OffMA50%", format="%.1f"),
            "RS21":         st.column_config.NumberColumn("RS21", format="%.1f"),
            "RS63":         st.column_config.NumberColumn("RS63", format="%.1f"),
            "CompRS":       st.column_config.NumberColumn("CompRS", format="%.3f"),
            "Liq5Cr":       st.column_config.NumberColumn("Liq5Cr", format="%.0f"),
            "LiqFromLow":   st.column_config.NumberColumn("LiqFromLow", format="%.1fx"),
            "VolQuality":   st.column_config.NumberColumn("Vol Quality", format="%.2f", help=">0.55 = Valid Accumulation"),
            "IAS":          st.column_config.ProgressColumn("IAS", format="%.0f", max_value=100),
            "Tier":         st.column_config.TextColumn("Tier", width=60),
            "V21_CRS_Gap":  st.column_config.NumberColumn("V21 CRS Gap", format="%.2f"),
            "V21_MA50_Gap": st.column_config.NumberColumn("V21 MA50 Gap%", format="%.1f"),
        },
        use_container_width=True,
        hide_index=True,
        height=600,
    )

    # --- IAS distribution + Sub-industry breakdown ---
    st.markdown("---")
    st.markdown("### Signal Analytics")
    dcols = st.columns([2, 1])

    with dcols[0]:
        st.markdown("**IAS Score Distribution**")
        fig_hist = go.Figure()
        for tier, color in TIER_COLORS.items():
            sub = wdf[wdf["Tier"] == tier]["IAS"]
            if not sub.empty:
                fig_hist.add_trace(go.Histogram(
                    x=sub, name=tier, marker_color=color, opacity=0.8,
                    xbins=dict(size=5)
                ))
        fig_hist.update_layout(
            barmode="stack", xaxis_title="IAS Score", yaxis_title="Stocks",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"), legend=dict(orientation="h"),
            height=280, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_hist, use_container_width=True, key="ias_hist_radar")

    with dcols[1]:
        st.markdown("**Top Sub-Industries**")
        si_counts = wdf["Sub_Industry"].value_counts().head(12).reset_index()
        si_counts.columns = ["Sub-Industry", "Count"]
        st.dataframe(si_counts, hide_index=True, use_container_width=True, height=280)

    # --- V21 Pipeline ---
    st.markdown("---")
    st.markdown("### V21 Pipeline (ALERT + READY tier)")
    st.caption("% below MA50 each stock must close to trigger V21 entry. Smaller bar = closer to qualification.")

    pipe_df = wdf[wdf["Tier"].isin(["ALERT","READY"])].copy()
    if not pipe_df.empty and "V21_MA50_Gap" in pipe_df.columns:
        pipe_df = pipe_df.sort_values("V21_MA50_Gap").head(30)
        fig_pipe = go.Figure()
        fig_pipe.add_trace(go.Bar(
            x=pipe_df["Ticker"].str.replace(".NS","", regex=False),
            y=pipe_df["V21_MA50_Gap"],
            marker_color=[TIER_COLORS.get(t, "#ccc") for t in pipe_df["Tier"]],
            text=[f"{t} | IAS {v}" for t, v in zip(pipe_df["Tier"], pipe_df["IAS"])],
            textposition="outside",
        ))
        fig_pipe.update_layout(
            xaxis_title="Stock", yaxis_title="% Below MA50",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"), height=340,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_pipe, use_container_width=True, key="pipeline_bar_radar")
    else:
        st.info("No ALERT or READY stocks today.")

    if "Date" in wdf.columns and len(wdf):
        st.caption(f"Last run: {wdf['Date'].iloc[0]} | Run turnaround_screener.py or wait for GitHub Actions.")

    # ── IAS Signal Log ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📓 IAS Signal Log — Multibagger Tracker")

    LOG_CSV_PATH = "data/ias_signal_log.csv"
    if not os.path.exists(LOG_CSV_PATH):
        st.info("No signal log yet. The log is created automatically after the next screener run.")
    else:
        log_df = pd.read_csv(LOG_CSV_PATH)

        # ── Live price refresh ────────────────────────────────────────────────
        # The screener only runs daily via GitHub Actions; refresh current_price
        # here so the tracker always shows today's prices, not the stale CSV value.

        # Fast path: use prices already loaded in the Trend Scanner session state
        _md_sess = st.session_state.get('market_data', pd.DataFrame())
        _md_prices: dict = {}
        if (isinstance(_md_sess, pd.DataFrame) and not _md_sess.empty
                and 'ticker' in _md_sess.columns and 'price' in _md_sess.columns):
            _md_prices = (
                _md_sess.dropna(subset=['price'])
                        .set_index('ticker')['price']
                        .to_dict()
            )

        @st.cache_data(ttl=1800, show_spinner=False)
        def _fetch_turnaround_live_prices(tickers_t: tuple) -> dict:
            """Batch-download last-close for log tickers not already in market_data."""
            if not tickers_t:
                return {}
            try:
                raw = yf.download(
                    list(tickers_t), period="5d",
                    threads=False, progress=False, auto_adjust=True
                )
                if raw is None or raw.empty:
                    return {}
                close = (raw["Close"] if isinstance(raw.columns, pd.MultiIndex)
                         else raw.rename(columns={raw.columns[0]: tickers_t[0]
                                                  if len(tickers_t) == 1 else raw.columns[0]}))
                result = {}
                for t in tickers_t:
                    try:
                        s = close[t].dropna() if t in close.columns else pd.Series(dtype=float)
                        if not s.empty:
                            result[t] = round(float(s.iloc[-1]), 2)
                    except Exception:
                        pass
                return result
            except Exception:
                return {}

        _log_tickers_all = log_df["ticker"].dropna().unique().tolist()
        _need_yf = tuple(t for t in _log_tickers_all if t not in _md_prices)

        _col_btn, _col_info = st.columns([1, 4])
        with _col_btn:
            if st.button("🔄 Refresh Prices", key="turnaround_refresh_prices"):
                st.cache_data.clear()
                st.rerun()
        with _col_info:
            _src_note = (
                f"Prices from session data ({len(_md_prices)} tickers) "
                + (f"+ live fetch ({len(_need_yf)} tickers)" if _need_yf else "")
            )
            st.caption(_src_note)

        with st.spinner("Fetching live prices for signal log…"):
            _yf_prices = _fetch_turnaround_live_prices(_need_yf)

        _all_live: dict = {**_yf_prices, **_md_prices}  # md_prices wins (fresher)

        # Vectorised update of current_price, return_since_signal, max_gain, xirr
        log_df["_live"] = log_df["ticker"].map(_all_live)
        _sig_p = pd.to_numeric(log_df["signal_price"], errors="coerce")
        _live_p = pd.to_numeric(log_df["_live"], errors="coerce")
        _has = _live_p.notna() & (_live_p > 0) & _sig_p.notna() & (_sig_p > 0)

        if _has.any():
            log_df.loc[_has, "current_price"] = _live_p[_has].round(1)
            _ret = ((_live_p[_has] / _sig_p[_has]) - 1) * 100
            log_df.loc[_has, "return_since_signal"] = _ret.round(1)
            _prev_max = pd.to_numeric(log_df.loc[_has, "max_gain"], errors="coerce").fillna(0)
            log_df.loc[_has, "max_gain"] = _ret.where(_ret > _prev_max, _prev_max).round(1)

            # XIRR (row-by-row: depends on days elapsed per signal_date)
            _today = datetime.now()
            for _idx in log_df.index[_has]:
                try:
                    _days = (_today - datetime.strptime(
                        str(log_df.at[_idx, "signal_date"])[:10], "%Y-%m-%d")).days
                    _sp = float(log_df.at[_idx, "signal_price"])
                    _cp = float(log_df.at[_idx, "_live"])
                    if _days >= 1 and _sp > 0 and _cp > 0:
                        log_df.at[_idx, "xirr"] = round(
                            ((_cp / _sp) ** (365.0 / _days) - 1) * 100, 1)
                except Exception:
                    pass

        log_df = log_df.drop(columns=["_live"])

        # ── Categorical filters + sort ────────────────────────────────────────
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            status_filter = st.multiselect(
                "Status", ["ACTIVE", "GRADUATED", "DROPPED"],
                default=["ACTIVE", "GRADUATED"],
                key="log_status_filter"
            )
        with col_f2:
            tier_filter = st.multiselect(
                "Peak Tier", ["ALERT", "READY", "WATCH"],
                default=["ALERT", "READY", "WATCH"],
                key="log_tier_filter"
            )
        with col_f3:
            sort_by = st.selectbox(
                "Sort by",
                ["xirr", "return_since_signal", "max_gain", "peak_ias", "signal_date"],
                index=0, key="log_sort_by"
            )

        # ── Range filters ─────────────────────────────────────────────────────
        _RF_DEFS = [
            # (column,                label,                  step)
            ("signal_ias",          "IAS (signal)",           1.0),
            ("signal_rs21",         "RS21 (signal)",          1.0),
            ("signal_rs63",         "RS63 (signal)",          1.0),
            ("signal_comp_rs",      "CompRS (signal)",        0.01),
            ("signal_off_52w_high", "Off 52W High %",         1.0),
            ("signal_off_ma50",     "Off MA50 %",             1.0),
            ("signal_off_ma200",    "Off MA200 %",            1.0),
            ("signal_liq5cr",       "Liq5Cr ₹Cr (signal)",   5.0),
            ("signal_liq_from_low", "Liq From Low (signal)",  0.5),
            ("signal_vol_quality",  "Vol Quality (signal)",   0.05),
            ("signal_shock_ratio",  "Shock Ratio (signal)",   0.05),
            ("peak_ias",            "Peak IAS",               1.0),
            ("days_on_watchlist",   "Days on Watchlist",      1.0),
            ("return_since_signal", "Return Since Signal %",  5.0),
            ("max_gain",            "Max Gain %",             5.0),
            ("xirr",                "XIRR / CAGR %",         5.0),
            ("return_5d",           "Return 5d %",            1.0),
            ("return_21d",          "Return 21d %",           1.0),
            ("return_63d",          "Return 63d %",           1.0),
        ]

        with st.expander("🎚️ Range Filters", expanded=False):
            import math as _math
            _range_vals: dict = {}
            rf_avail = [(c, lbl, step) for c, lbl, step in _RF_DEFS if c in log_df.columns]
            _N_RF_COLS = 4
            for _ri in range(0, len(rf_avail), _N_RF_COLS):
                _row = rf_avail[_ri: _ri + _N_RF_COLS]
                _rcols = st.columns(_N_RF_COLS)
                for _j, (col, lbl, step) in enumerate(_row):
                    _s = pd.to_numeric(log_df[col], errors="coerce").dropna()
                    if len(_s) < 1:
                        continue
                    _lo = _math.floor(float(_s.min()) / step) * step
                    _hi = _math.ceil( float(_s.max()) / step) * step
                    if _hi <= _lo:
                        _hi = _lo + step
                    with _rcols[_j]:
                        _sel = st.slider(
                            lbl, min_value=_lo, max_value=_hi,
                            value=(_lo, _hi), step=step, key=f"rf_{col}"
                        )
                        _range_vals[col] = _sel

        # ── Build combined filter mask ────────────────────────────────────────
        mask = pd.Series([True] * len(log_df))
        if status_filter:
            mask &= log_df["status"].isin(status_filter)
        if tier_filter:
            mask &= log_df["peak_tier"].isin(tier_filter)
        for _col, (_lo_sel, _hi_sel) in _range_vals.items():
            _num = pd.to_numeric(log_df[_col], errors="coerce")
            # rows with NaN are kept (they haven't been measured yet)
            mask &= (_num.between(_lo_sel, _hi_sel, inclusive="both") | _num.isna())

        filtered = log_df[mask].copy()

        if sort_by in filtered.columns:
            ascending = sort_by == "signal_date"
            filtered = filtered.sort_values(sort_by, ascending=ascending, na_position="last")

        # ── Summary KPIs ──────────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Logged", len(log_df))
        k2.metric("Active", int((log_df["status"] == "ACTIVE").sum()))
        k3.metric("Graduated", int((log_df["status"] == "GRADUATED").sum()))
        k4.metric("Dropped", int((log_df["status"] == "DROPPED").sum()))
        multibaggers = int((pd.to_numeric(log_df["return_since_signal"], errors="coerce") >= 100).sum())
        k5.metric("Multibaggers (2×+)", multibaggers)

        # ── Main table ────────────────────────────────────────────────────────
        DISPLAY_COLS = [
            "ticker", "name", "sub_industry", "cycle", "status",
            "signal_date", "signal_tier", "signal_price",
            "signal_ias", "signal_rs21", "signal_rs63", "signal_comp_rs",
            "signal_off_52w_high", "signal_off_ma50",
            "peak_tier", "peak_ias", "days_on_watchlist",
            "return_5d", "return_21d", "return_63d",
            "current_price", "return_since_signal", "max_gain", "xirr",
        ]
        show_cols = [c for c in DISPLAY_COLS if c in filtered.columns]
        st.dataframe(
            filtered[show_cols].reset_index(drop=True),
            use_container_width=True,
            height=420,
        )

        # ── Multibagger spotlight ──────────────────────────────────────────────
        mb_df = log_df[pd.to_numeric(log_df["return_since_signal"], errors="coerce") >= 100].copy()
        if not mb_df.empty:
            st.markdown("#### 🚀 Multibagger Board (2×+ returns)")
            mb_show = [c for c in ["ticker", "name", "signal_date", "signal_price",
                                    "current_price", "return_since_signal", "max_gain",
                                    "xirr", "status", "days_on_watchlist"] if c in mb_df.columns]
            st.dataframe(
                mb_df[mb_show].sort_values("return_since_signal", ascending=False).reset_index(drop=True),
                use_container_width=True,
            )

        # ── Forward return distribution ────────────────────────────────────────
        with st.expander("📊 Forward Return Distribution (5d / 21d / 63d)"):
            ret_cols = ["return_5d", "return_21d", "return_63d"]
            avail = [c for c in ret_cols if c in log_df.columns and log_df[c].notna().any()]
            if avail:
                import plotly.graph_objects as go
                fig_ret = go.Figure()
                colours = {"return_5d": "#4fc3f7", "return_21d": "#81c784", "return_63d": "#ffb74d"}
                labels  = {"return_5d": "5-day", "return_21d": "21-day", "return_63d": "63-day"}
                for col in avail:
                    vals = pd.to_numeric(log_df[col], errors="coerce").dropna()
                    fig_ret.add_trace(go.Histogram(
                        x=vals, name=labels[col],
                        marker_color=colours[col], opacity=0.7, nbinsx=20,
                    ))
                fig_ret.update_layout(
                    barmode="overlay", height=280,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e0e0e0"),
                    xaxis_title="Return (%)", yaxis_title="Count",
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                fig_ret.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.4)
                st.plotly_chart(fig_ret, use_container_width=True, key="log_fwd_ret_hist")

                # Summary table
                summary_rows = []
                for col in avail:
                    vals = pd.to_numeric(log_df[col], errors="coerce").dropna()
                    summary_rows.append({
                        "Period": labels[col], "N": len(vals),
                        "Avg %": round(vals.mean(), 1),
                        "Median %": round(vals.median(), 1),
                        "Win Rate %": round((vals > 0).mean() * 100, 1),
                        "Max %": round(vals.max(), 1),
                        "Min %": round(vals.min(), 1),
                    })
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            else:
                st.info("Not enough data yet — forward returns will populate once signals age past 5/21/63 trading days.")

        # ── Full detail expander ───────────────────────────────────────────────
        with st.expander("🔬 Full Signal Detail (all columns)"):
            st.dataframe(filtered.reset_index(drop=True), use_container_width=True, height=360)


# ─────────────────────────────────────────────────────────────────────────────
# VIEW: AI CAPEX PLAY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🖥️ AI Capex":
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
    @st.cache_data(ttl=3600)
    def _nifty_rs_refs():
        try:
            close = yf.Ticker("^NSEI").history(period="100d")['Close'].dropna()
            if len(close) < 65:
                return 0.0, 0.0, 0.0
            n_1w = (close.iloc[-1] - close.iloc[-6])  / close.iloc[-6]  * 100
            n_1m = (close.iloc[-1] - close.iloc[-22]) / close.iloc[-22] * 100
            n_3m = (close.iloc[-1] - close.iloc[-64]) / close.iloc[-64] * 100
            return round(n_1w, 2), round(n_1m, 2), round(n_3m, 2)
        except Exception:
            return 0.0, 0.0, 0.0

    n_1w_ref, n_1m_ref, n_3m_ref = _nifty_rs_refs()

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



# ─────────────────────────────────────────────────────────────────────────────
# VIEW: US AI PLAY  (rotation-first, ThemeEngine-powered)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🇺🇸 US AI Play":
    st.markdown(page_header(
        "🇺🇸 US AI Supply Chain",
        "Full 12-layer datacenter supply chain | Rotation detection vs SMH/SPY | Config D weights"
    ), unsafe_allow_html=True)

    # ── Sidebar controls ──────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🇺🇸 US AI Play Settings**")

    us_benchmark = st.sidebar.selectbox(
        "Benchmark", ["SMH", "SPY", "QQQ"], index=0, key="us_benchmark",
        help="SMH = Semis ETF (sector-relative alpha). SPY = broad market."
    )
    st.sidebar.markdown("**RS Weights (Config D)**")
    us_w5  = st.sidebar.slider("RS5  (1-week)",  0.10, 0.50, 0.30, 0.05, key="us_w5")
    us_w21 = st.sidebar.slider("RS21 (1-month)", 0.20, 0.60, 0.50, 0.05, key="us_w21")
    raw_us_w63 = round(1.0 - us_w5 - us_w21, 2)
    us_w63 = max(0.05, raw_us_w63)
    us_total_w = us_w5 + us_w21 + us_w63
    us_w5_n, us_w21_n, us_w63_n = us_w5 / us_total_w, us_w21 / us_total_w, us_w63 / us_total_w
    st.sidebar.caption(f"RS63 auto-set to **{us_w63_n:.0%}** | RS5={us_w5_n:.0%} RS21={us_w21_n:.0%}")
    if us_w63_n >= us_w5_n:
        st.sidebar.warning("⚠️ RS63 ≥ RS5 — raise RS5 to front-load theme momentum")

    # ── Cached ThemeEngine scan ───────────────────────────────────────────────
    @st.cache_data(ttl=3600, show_spinner=False)
    def _run_us_ai_scan(rs_weights_t: tuple, benchmark: str) -> pd.DataFrame:
        cfg = dict(AI_CAPEX_THEME)
        cfg['benchmark'] = benchmark
        cfg['rs_weights'] = list(rs_weights_t)
        engine = ThemeEngine(cfg)
        return engine.scan()

    rs_weights_t = ((5, round(us_w5_n, 4)), (21, round(us_w21_n, 4)), (63, round(us_w63_n, 4)))

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh Data", key="us_ai_refresh"):
            st.cache_data.clear()
            st.rerun()

    with st.spinner(f"Loading AI supply chain data vs {us_benchmark}…"):
        scan_df = _run_us_ai_scan(rs_weights_t, us_benchmark)

    if scan_df.empty:
        st.error("⚠️ Data fetch failed — check your connection. Click **Refresh Data** to retry.")
        st.stop()

    # ── 4 Pulse Metrics ───────────────────────────────────────────────────────
    pct_up    = scan_df['Signal'].isin(['STRONG UP', 'UPTREND']).mean() * 100
    avg_rs    = scan_df['CompRS'].mean()
    vel_vals  = scan_df['RS_Vel'].dropna()
    avg_vel   = vel_vals.mean() if not vel_vals.empty else 0.0
    top_layer_full = scan_df.groupby('Layer')['CompRS'].mean().idxmax() if not scan_df.empty else "—"
    top_layer_label = top_layer_full.split(': ', 1)[-1] if ': ' in top_layer_full else top_layer_full

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Stocks in Uptrend",            f"{pct_up:.0f}%")
    mc2.metric(f"Avg Comp RS vs {us_benchmark}", f"{avg_rs:+.1f}%")
    mc3.metric("Avg RS Velocity (5D)",           f"{avg_vel:+.2f}%",
               delta="↑ Accelerating" if avg_vel > 0 else "↓ Decelerating")
    mc4.metric("Leading Layer",                  top_layer_label)

    st.markdown("---")

    # ── Layer Rotation Matrix ─────────────────────────────────────────────────
    st.markdown("### 🔄 Layer Rotation Matrix")
    st.caption(
        f"Avg per supply-chain layer vs **{us_benchmark}** | "
        f"RS5={us_w5_n:.0%} / RS21={us_w21_n:.0%} / RS63={us_w63_n:.0%} | "
        "Sorted best→worst Composite RS"
    )

    layer_agg = (
        scan_df.groupby('Layer')
        .agg(
            CompRS=('CompRS', 'mean'),
            RS_Vel=('RS_Vel', 'mean'),
            RS5vs=('RS5vs', 'mean'),
            RS21vs=('RS21vs', 'mean'),
            RS63vs=('RS63vs', 'mean'),
            Count=('Ticker', 'count'),
            Pct_Above_MA=('Above_MA', lambda x: round(x.mean() * 100, 0)),
        )
        .round(2)
        .reset_index()
        .sort_values('CompRS', ascending=False)
    )

    # ── Persist daily layer snapshot ─────────────────────────────────────────
    _AI_LAYER_FILE = "data/ai_layer_rotation.csv"
    _today_str = datetime.now().strftime("%Y-%m-%d")

    def _save_ai_layer_snapshot(agg_df):
        os.makedirs("data", exist_ok=True)
        rows = agg_df[["Layer", "CompRS", "RS_Vel", "Pct_Above_MA"]].copy()
        rows["date"] = _today_str
        if os.path.exists(_AI_LAYER_FILE):
            existing = pd.read_csv(_AI_LAYER_FILE)
            existing = existing[existing["date"] != _today_str]  # replace today
        else:
            existing = pd.DataFrame()
        combined = pd.concat([existing, rows], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=120)
        combined = combined[combined["date"] >= cutoff]
        combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
        combined.to_csv(_AI_LAYER_FILE, index=False)
        return combined

    def _load_ai_layer_history():
        if not os.path.exists(_AI_LAYER_FILE):
            return pd.DataFrame()
        df = pd.read_csv(_AI_LAYER_FILE)
        df["date"] = pd.to_datetime(df["date"])
        return df

    _save_ai_layer_snapshot(layer_agg)
    _layer_hist = _load_ai_layer_history()

    # Backfill 12 months of history on first visit (< 2 distinct dates in file)
    if _layer_hist["date"].nunique() < 2 if not _layer_hist.empty else True:
        @st.cache_data(ttl=86400, show_spinner=False)
        def _backfill_ai_layers(rs_weights_t, benchmark):
            cfg = dict(AI_CAPEX_THEME)
            cfg['benchmark'] = benchmark
            cfg['rs_weights'] = list(rs_weights_t)
            eng = ThemeEngine(cfg)
            return eng.backfill_layer_rotation(n_months=12)

        with st.spinner("Building 12-month layer history (first visit only)…"):
            _bf_df = _backfill_ai_layers(rs_weights_t, us_benchmark)

        if not _bf_df.empty:
            _existing_dates = set(
                _layer_hist["date"].dt.strftime("%Y-%m-%d").tolist()
                if not _layer_hist.empty else []
            )
            _bf_new = _bf_df[~_bf_df["date"].isin(_existing_dates)]
            if not _bf_new.empty:
                os.makedirs("data", exist_ok=True)
                if os.path.exists(_AI_LAYER_FILE):
                    _curr_csv = pd.read_csv(_AI_LAYER_FILE)
                    _combined_bf = pd.concat([_bf_new, _curr_csv], ignore_index=True)
                else:
                    _combined_bf = _bf_new
                _combined_bf.to_csv(_AI_LAYER_FILE, index=False)
                _layer_hist = _load_ai_layer_history()

    # Build pivot: Layer × date (sorted chronologically)
    if not _layer_hist.empty:
        _hist_pivot = _layer_hist.pivot_table(
            index="Layer", columns="date", values="CompRS", aggfunc="mean"
        )
        _hist_pivot = _hist_pivot.sort_index(axis=1)  # dates ascending
        _n_days = len(_hist_pivot.columns)
    else:
        _hist_pivot = pd.DataFrame()
        _n_days = 0

    # ── Tabs: Trend Table | Heatmap ──────────────────────────────────────────
    rot_tab1, rot_tab2 = st.tabs(["📈 Trend Table (day-by-day)", "🌡️ Current Heatmap"])

    with rot_tab1:
        if _n_days < 2:
            st.info("Trend data accumulates from the second visit onwards — come back tomorrow for sparklines.")
        else:
            _trend_rows = []
            for _, r in layer_agg.iterrows():
                lyr = r["Layer"]
                if lyr in _hist_pivot.index:
                    series = _hist_pivot.loc[lyr].dropna().tolist()
                else:
                    series = [r["CompRS"]]

                # 5D delta: today vs ~5 sessions ago
                _delta_5d = None
                if lyr in _hist_pivot.index:
                    _valid = _hist_pivot.loc[lyr].dropna()
                    if len(_valid) >= 6:
                        _delta_5d = round(float(_valid.iloc[-1]) - float(_valid.iloc[-6]), 2)
                    elif len(_valid) >= 2:
                        _delta_5d = round(float(_valid.iloc[-1]) - float(_valid.iloc[0]), 2)

                # Momentum badge
                if _delta_5d is not None:
                    if _delta_5d > 1:
                        badge = "🟢 Rising"
                    elif _delta_5d < -1:
                        badge = "🔴 Falling"
                    else:
                        badge = "🟡 Flat"
                else:
                    badge = "—"

                _trend_rows.append({
                    "Layer":         lyr,
                    "Comp RS":       r["CompRS"],
                    "5D Δ":          _delta_5d,
                    "Momentum":      badge,
                    "RS Vel":        r["RS_Vel"],
                    "% > MA":        r["Pct_Above_MA"],
                    "Stocks":        int(r["Count"]),
                    f"Trend ({_n_days}d)": series,
                })

            _trend_df = pd.DataFrame(_trend_rows)
            _trend_col_cfg = {
                "Layer":    st.column_config.TextColumn("Layer"),
                "Comp RS":  st.column_config.NumberColumn("Comp RS", format="%+.1f%%"),
                "5D Δ":     st.column_config.NumberColumn("5D Δ", format="%+.1f%%"),
                "Momentum": st.column_config.TextColumn("Momentum"),
                "RS Vel":   st.column_config.NumberColumn("RS Velocity", format="%+.2f%%"),
                "% > MA":   st.column_config.NumberColumn("% > MA", format="%.0f%%"),
                "Stocks":   st.column_config.NumberColumn("Stocks", format="%d"),
                f"Trend ({_n_days}d)": st.column_config.LineChartColumn(
                    f"Trend ({_n_days}d)", y_min=-30, y_max=30
                ),
            }
            st.dataframe(_trend_df, column_config=_trend_col_cfg,
                         hide_index=True, use_container_width=True)

            _csv_bytes = _trend_df.drop(columns=[f"Trend ({_n_days}d)"]).to_csv(index=False).encode()
            st.download_button("⬇️ Download Layer Rotation CSV", _csv_bytes,
                               "ai_layer_rotation.csv", "text/csv", key="dl_layer_rot")

    with rot_tab2:
        _metrics  = ['RS_Vel', 'RS5vs', 'RS21vs', 'RS63vs', 'CompRS']
        _m_labels = ['RS Velocity (5D)', f'RS5 vs {us_benchmark}', f'RS21 vs {us_benchmark}',
                     f'RS63 vs {us_benchmark}', 'Composite RS']
        _layers   = layer_agg['Layer'].tolist()
        _z        = layer_agg[_metrics].values.tolist()

        _text = [
            [f"{v:+.1f}" if (v is not None and not (isinstance(v, float) and np.isnan(v))) else "—"
             for v in row]
            for row in _z
        ]

        fig_heat = go.Figure(go.Heatmap(
            z=_z,
            x=_m_labels,
            y=_layers,
            colorscale='RdYlGn',
            zmid=0,
            text=_text,
            texttemplate="%{text}",
            textfont={"size": 11},
            showscale=True,
            colorbar=dict(title="% vs Bench", thickness=14),
        ))
        fig_heat.update_layout(
            height=max(380, len(_layers) * 44),
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(side='top'),
            yaxis=dict(autorange='reversed'),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # ── Capex Cycle Position ───────────────────────────────────────────────────
    st.markdown("### 📍 Capex Cycle Position")

    _early_layers = [
        'L1: InP Substrate', 'L1: SiC/GaN',
        'L0: EDA Software', 'L0: IP Cores',
        'L3: Equipment', 'L3: Photomasks', 'L3: Ion Implant', 'L3: Metrology',
    ]
    _mid_layers = [
        'L4: Foundry', 'L5: HBM Memory',
        'L6: Chip Design', 'L6: Custom ASIC', 'L6: Power Semi', 'L6: VRM',
    ]
    _late_layers = [
        'L7: Packaging',
        'L8: Networking', 'L8: Optics', 'L8: Connectors',
        'L9: Power Dist', 'L9A: Power Gen', 'L9A: Nuclear', 'L9A: Fuel Cell', 'L9A: Solar',
        'L10: Cooling', 'L10A: Water',
        'L11: Construction', 'L11: DC REIT',
        'L12: Hyperscaler',
    ]

    def _stage_rs(layer_list):
        sub = scan_df[scan_df['Layer'].isin(layer_list)]
        return sub['CompRS'].mean() if not sub.empty else 0.0

    rs_early = _stage_rs(_early_layers)
    rs_mid   = _stage_rs(_mid_layers)
    rs_late  = _stage_rs(_late_layers)

    _max_rs = max(rs_early, rs_mid, rs_late)
    if _max_rs == rs_early:
        _stage_label = "🌱 Early Capex Cycle"
        _stage_desc  = "Equipment & EDA tools lead — foundries ramping capacity. **Long L3: ASML, AMAT, LRCX, KLAC.**"
        _stage_color = "#2196F3"
    elif _max_rs == rs_mid:
        _stage_label = "⚡ Mid Capex Cycle"
        _stage_desc  = "Chip production peaks — memory & custom ASIC demand surges. **Long L5-L6: NVDA, MU, AVGO, MRVL.**"
        _stage_color = "#FF9800"
    else:
        _stage_label = "🏗️ Late Capex Cycle"
        _stage_desc  = "Infrastructure buildout dominant: power, cooling, DC construction. **Long L9-L11: ETN, VRT, PWR, EQIX.**"
        _stage_color = "#4CAF50"

    st.markdown(
        f"""<div style="background:{_stage_color}20; border-left:4px solid {_stage_color};
        padding:14px 18px; border-radius:6px; margin-bottom:14px;">
        <span style="font-size:1.1em; font-weight:600">{_stage_label}</span><br>
        <span style="color:#bbb; font-size:0.95em">{_stage_desc}</span><br><br>
        <span style="font-size:0.85em; color:#aaa">
        🌱 Early RS: <b style="color:#ddd">{rs_early:+.1f}%</b> &nbsp;|&nbsp;
        ⚡ Mid RS: <b style="color:#ddd">{rs_mid:+.1f}%</b> &nbsp;|&nbsp;
        🏗️ Late RS: <b style="color:#ddd">{rs_late:+.1f}%</b>
        </span></div>""",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Entry Signals (top 3 leading layers) ─────────────────────────────────
    st.markdown("### 🎯 Entry Signals — Leading Layer Picks")

    _top3_layers = layer_agg.head(3)['Layer'].tolist()
    entry_df = scan_df[scan_df['Layer'].isin(_top3_layers)].copy()
    entry_df = entry_df[entry_df['RS_Vel'].notna()].copy()

    def _grade(r):
        if r['CompRS'] > 5 and r['RS_Vel'] > 0 and r['Above_MA']:
            return "🔥 BUY"
        if r['CompRS'] > 0 and r['Above_MA']:
            return "👀 WATCH"
        return "⚠️ WEAK"

    entry_df['Grade'] = entry_df.apply(_grade, axis=1)
    entry_df = entry_df.sort_values(['Grade', 'CompRS'], ascending=[True, False])

    _entry_display = entry_df[
        ['Ticker', 'Layer', 'Grade', 'CompRS', 'RS_Vel', 'RS5vs', 'RS21vs', 'Price', 'Bottleneck']
    ].copy()
    _entry_display.columns = ['Ticker', 'Layer', 'Signal', 'Comp RS', 'RS Vel', 'RS5 vs', 'RS21 vs', 'Price $', 'Chokepoint']

    st.dataframe(
        _entry_display.reset_index(drop=True),
        column_config={
            'Comp RS':   st.column_config.NumberColumn("Comp RS",        format="%+.2f%%"),
            'RS Vel':    st.column_config.NumberColumn("RS Vel (5D)",    format="%+.2f%%"),
            'RS5 vs':    st.column_config.NumberColumn(f"RS5 vs {us_benchmark}", format="%+.1f%%"),
            'RS21 vs':   st.column_config.NumberColumn(f"RS21 vs {us_benchmark}", format="%+.1f%%"),
            'Price $':   st.column_config.NumberColumn("Price $",        format="$%.2f"),
            'Chokepoint': st.column_config.TextColumn("Chokepoint"),
        },
        use_container_width=True,
        hide_index=True,
        height=min(len(_entry_display) * 38 + 42, 520),
    )
    st.caption(f"Showing top 3 layers by Composite RS | Refresh hourly | Benchmark: {us_benchmark}")

    st.markdown("---")

    # ── Full Universe Screener (expander) ────────────────────────────────────
    with st.expander("📋 Full Universe Screener", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            _sig_opts   = sorted(scan_df['Signal'].unique().tolist())
            _sig_def    = [s for s in ['STRONG UP', 'UPTREND'] if s in _sig_opts]
            sig_filter  = st.multiselect("Signal", _sig_opts, default=_sig_def, key="us_sig_filter")
        with fc2:
            layer_opts   = sorted(scan_df['Layer'].unique().tolist())
            layer_filter = st.multiselect("Layer", layer_opts, default=[], key="us_layer_filter")
        with fc3:
            btn_filter   = st.selectbox("Chokepoint", ["All", "CRITICAL", "TIGHT"], key="us_btn_filter")

        _filtered = scan_df.copy()
        if sig_filter:
            _filtered = _filtered[_filtered['Signal'].isin(sig_filter)]
        if layer_filter:
            _filtered = _filtered[_filtered['Layer'].isin(layer_filter)]
        if btn_filter != "All":
            _filtered = _filtered[_filtered['Bottleneck'] == btn_filter]

        _disp_cols = ['Ticker', 'Layer', 'Bottleneck', 'Signal', 'Price',
                      'CompRS', 'RS_Vel', 'RS5vs', 'RS21vs', 'RS63vs', 'Dist_52W']
        _disp = _filtered[_disp_cols].copy()
        _disp.columns = ['Ticker', 'Layer', 'Chokepoint', 'Signal', 'Price $',
                         'Comp RS', 'RS Vel', 'RS5 vs', 'RS21 vs', 'RS63 vs', '52W Off%']

        st.dataframe(
            _disp.reset_index(drop=True),
            column_config={
                'Price $':  st.column_config.NumberColumn("Price $",       format="$%.2f"),
                'Comp RS':  st.column_config.NumberColumn("Comp RS",       format="%+.2f%%"),
                'RS Vel':   st.column_config.NumberColumn("RS Vel (5D)",   format="%+.2f%%"),
                'RS5 vs':   st.column_config.NumberColumn(f"RS5 vs {us_benchmark}",  format="%+.1f%%"),
                'RS21 vs':  st.column_config.NumberColumn(f"RS21 vs {us_benchmark}", format="%+.1f%%"),
                'RS63 vs':  st.column_config.NumberColumn(f"RS63 vs {us_benchmark}", format="%+.1f%%"),
                '52W Off%': st.column_config.NumberColumn("52W Off%",      format="%.1f%%"),
            },
            use_container_width=True,
            hide_index=True,
            height=min(len(_disp) * 38 + 42, 700),
        )
        st.caption(
            f"Benchmark: {us_benchmark} | Weights: RS5={us_w5_n:.0%} / RS21={us_w21_n:.0%} / "
            f"RS63={us_w63_n:.0%} | {len(_disp)} of {len(scan_df)} stocks shown"
        )

    # ── Supply Chain Layer Map ────────────────────────────────────────────────
    with st.expander("🗺️ Supply Chain Layer Map"):
        st.markdown("""
| Layer | Category | Key Companies | Feeds Into |
|---|---|---|---|
| **L0** | EDA & IP Cores | SNPS, CDNS (duopoly), ARM | L4, L6 |
| **L1-L2** | Raw materials & substrates | AXTI (InP), WOLF (SiC), ENTG | L3, L8 |
| **L3** | Semiconductor equipment | ASML (EUV monopoly), AMAT, LRCX, KLAC, ACLS | L4, L5 |
| **L4-L5** | Foundry & memory | TSM (INTC, GFS), MU (HBM) | L6, L7 |
| **L6** | Chip design | NVDA, AMD, AVGO, MRVL, MPWR, ON | L7, L8 |
| **L7** | Advanced packaging | AMKR, KLIC | L8, L12 |
| **L8** | Networking & optics | ANET, COHR, LITE, MTSI, FN, APH | L12 |
| **L9-L10** | Power & cooling | ETN, VRT, GEV, PWR, NVT, ECL | L11, L12 |
| **L11** | DC construction & REITs | EQIX, DLR, EME, FIX, CEG, CCJ | L12 |
| **L12** | Hyperscalers (demand signal) | MSFT, AMZN, GOOGL, META, ORCL | Revenue |

**Rotation insight:** L3 Equipment leads at the start of each capex wave (new fab orders).  
L6 Chips peak mid-cycle. L9-L11 Power/Infra run late as sites come online.  
ASML→TSM is the deepest serial chokepoint — 2 CRITICAL monopolies in series.  
Power (L9) is the #1 near-term rate limiter for AI scaling.
        """)

    with st.expander("ℹ️ RS Weight Tuning Guide"):
        st.markdown(f"""
**Standard OptComp-V22:** RS5 = 10%, RS21 = 50%, RS63 = 40%

**AI Supply Chain (current):** RS5 = {us_w5_n:.0%}, RS21 = {us_w21_n:.0%}, RS63 = {us_w63_n:.0%}

AI datacenter is a **fast-rotating, news-driven theme.** The 3-month (RS63) lookback often
includes time *before* the theme ignited, diluting the signal. Elevating RS5 catches
fresh rotations earlier; RS21 stays dominant as the medium-term confirmation window.

**Config D (backtest-optimal):** RS5=30% / RS21=50% / RS63=20% — +70.3% CAGR, +25.5% alpha vs SMH, Sharpe 4.48

**Rotation Matrix** shows which supply-chain layer cluster is currently "in play" so you
can size into the right pocket before the rotation broadens.
        """)


# ---------------------------------------------------------------------------
# US SCANNER PAGE
# ---------------------------------------------------------------------------
elif page == "🇺🇸 US Scanner":

    st.markdown(page_header("🇺🇸 S&P 500 Momentum Scanner", "OptComp RS • Sector Heatmap • Sub-Industry Matrix | Powered by SPY Benchmark"), unsafe_allow_html=True)

    # --- Sidebar controls ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🇺🇸 US Scanner Settings**")
    us_scan_benchmark = st.sidebar.selectbox(
        "Benchmark",
        ["SPY", "QQQ", "IWM"],
        index=0,
        key="us_scan_bm",
        help="SPY = S&P 500 | QQQ = Nasdaq 100 | IWM = Russell 2000"
    )
    st.sidebar.markdown("**RS Weights (Config D)**")
    us_s_w5  = st.sidebar.slider("RS5  (1-week)",  0.10, 0.50, 0.30, 0.05, key="us_s_w5")
    us_s_w21 = st.sidebar.slider("RS21 (1-month)", 0.20, 0.60, 0.50, 0.05, key="us_s_w21")
    us_s_w63_n = round(1.0 - us_s_w5 - us_s_w21, 4)
    st.sidebar.caption(f"RS63 auto-set to **{us_s_w63_n:.0%}**")
    if us_s_w63_n < 0:
        st.sidebar.error("Weights exceed 100% — reduce RS5 or RS21")
        us_s_w63_n = 0.0

    us_live_mode = st.sidebar.checkbox("Live Mode (bypass cache)", value=False, key="us_scan_live")
    if st.sidebar.button("🗑️ Clear US Cache", key="us_clear_cache", help="Force fresh download (needed after ticker list changes)"):
        from utils.us_data_engine import clear_us_cache
        n = clear_us_cache()
        st.cache_data.clear()
        st.sidebar.success(f"Cleared {n} cache file(s). Reloading…")
        st.rerun()

    # --- Cached fetch wrapper ---
    @st.cache_data(ttl=3600, show_spinner=False)
    def _run_us_scan(benchmark: str, w5: float, w21: float, w63: float, live: bool) -> pd.DataFrame:
        weights = [(5, w5), (21, w21), (63, w63)]
        return fetch_us_market_data(benchmark=benchmark, rs_weights=weights, live_mode=live)

    with st.spinner("Downloading S&P 500 data…  (first load ~60 s, then cached)"):
        us_df = _run_us_scan(us_scan_benchmark, us_s_w5, us_s_w21, us_s_w63_n, us_live_mode)

    if us_df.empty:
        st.error("Failed to load US market data. Check internet connection or try Live Mode.")
        st.stop()

    # Ensure required columns exist
    for col in ["comp_rs", "volatility", "dna_signal", "sub_industry", "dist_200dma",
                "rs_5d", "rs_21d", "rs_63d"]:
        if col not in us_df.columns:
            us_df[col] = None

    # ── QUICK STATS ──────────────────────────────────────────────────────────
    qs1, qs2, qs3, qs4 = st.columns(4)
    _strong = len(us_df[us_df["trend_signal"] == "STRONG UPTREND"])
    _uptrend = len(us_df[us_df["trend_signal"].isin(["STRONG UPTREND", "UPTREND"])])
    _avg_ts = us_df["trend_score"].mean()
    _breakouts = len(us_df[us_df["dist_52w"] > -2.0])
    qs1.metric("🚀 Strong Momentum", f"{_strong}", help="Trend Score ≥ 75")
    qs2.metric("📈 Total Uptrends", f"{_uptrend}")
    qs3.metric("📊 Avg Trend Score", f"{_avg_ts:.0f}/100")
    qs4.metric("🔥 Breakout Alerts", f"{_breakouts}", help="Within 2% of 52W High")

    st.markdown("---")

    # ── SECTOR RS HEATMAP ────────────────────────────────────────────────────
    with st.expander("🗺️ **Sector RS Heatmap**", expanded=True):
        _sec_grp = (
            us_df.groupby("sector")[["rs_5d", "rs_21d", "rs_63d", "comp_rs", "trend_score"]]
            .mean()
            .round(2)
            .reset_index()
        )
        _sec_grp = _sec_grp.sort_values("comp_rs", ascending=False)

        if not _sec_grp.empty:
            _heat_z  = _sec_grp[["rs_5d", "rs_21d", "rs_63d", "comp_rs"]].values.tolist()
            _heat_y  = _sec_grp["sector"].tolist()
            _heat_x  = ["RS5 (1W)", "RS21 (1M)", "RS63 (3M)", "CompRS"]
            _heat_txt = [[f"{v:+.1f}" for v in row] for row in _heat_z]

            fig_heat = go.Figure(go.Heatmap(
                z=_heat_z,
                x=_heat_x,
                y=_heat_y,
                text=_heat_txt,
                texttemplate="%{text}",
                colorscale="RdYlGn",
                zmid=0,
                showscale=True,
                colorbar=dict(title="RS %", thickness=12),
            ))
            fig_heat.update_layout(
                template="plotly_dark",
                height=420,
                margin=dict(l=200, r=20, t=30, b=30),
                xaxis=dict(side="top"),
            )
            st.plotly_chart(fig_heat, use_container_width=True, key="us_sector_heatmap")

            # Sector table
            st.dataframe(
                _sec_grp.rename(columns={
                    "sector": "Sector", "rs_5d": "RS5 (1W)", "rs_21d": "RS21 (1M)",
                    "rs_63d": "RS63 (3M)", "comp_rs": "CompRS", "trend_score": "Avg Trend"
                }),
                column_config={
                    "RS5 (1W)":  st.column_config.NumberColumn(format="%+.2f%%"),
                    "RS21 (1M)": st.column_config.NumberColumn(format="%+.2f%%"),
                    "RS63 (3M)": st.column_config.NumberColumn(format="%+.2f%%"),
                    "CompRS":    st.column_config.NumberColumn(format="%+.2f%%"),
                    "Avg Trend": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                },
                hide_index=True,
                use_container_width=True,
            )

    # ── TOP MOVERS TAPE ──────────────────────────────────────────────────────
    _top8 = us_df.nlargest(8, "trend_score")[["ticker", "trend_score", "currentPrice"]]
    _tape_html = " &nbsp;•&nbsp; ".join([
        f"<span style='color:#34C759;font-weight:600'>{r['ticker']}</span>"
        f" <span style='color:#888'>${r['currentPrice']:.1f}</span>"
        f" <span style='background:rgba(52,199,89,0.2);padding:2px 8px;border-radius:10px;color:#34C759'>{r['trend_score']}</span>"
        for _, r in _top8.iterrows()
    ])
    st.markdown(
        f"""<div style="background:rgba(255,255,255,0.03);padding:12px 20px;border-radius:8px;
                        overflow-x:auto;white-space:nowrap;border:1px solid rgba(255,255,255,0.1);">
            <span style='color:#FFD700;margin-right:10px'>🔥 TOP MOVERS:</span> {_tape_html}
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── BREAKOUT ALERTS ───────────────────────────────────────────────────────
    _bkouts = us_df[us_df["dist_52w"] > -2.0].copy()
    if not _bkouts.empty:
        with st.expander(f"🚨 **{len(_bkouts)} BREAKOUT ALERTS** (Within 2% of 52W High)", expanded=False):
            _bkouts_s = _bkouts.nsmallest(20, "dist_52w").copy()
            _bkouts_s["yf_link"] = _bkouts_s.apply(lambda r: _google_finance_url(r['ticker']), axis=1)
            st.dataframe(
                _bkouts_s[["yf_link", "name", "currentPrice", "dist_52w", "trend_score", "overall"]],
                column_config={
                    "yf_link":      st.column_config.LinkColumn("Ticker", display_text=_GF_DISPLAY_RE),
                    "name":         "Company",
                    "currentPrice": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "dist_52w":     st.column_config.NumberColumn("% from 52W High", format="%.1f%%"),
                    "trend_score":  st.column_config.ProgressColumn("Trend", min_value=0, max_value=100),
                    "overall":      st.column_config.NumberColumn("Score", format="%.1f"),
                },
                hide_index=True,
                height=300,
            )

    # ── FILTERS ───────────────────────────────────────────────────────────────
    with st.expander("⚡ Filter", expanded=True):
        ff1, ff2 = st.columns([1, 2])
        with ff1:
            us_search = st.text_input("🔍 Search", placeholder="AAPL, Apple…", key="us_search")
        with ff2:
            _us_sectors = sorted(us_df["sector"].fillna("Unknown").unique().tolist())
            us_sel_sector = st.multiselect("Sector (Empty = All)", _us_sectors, default=[], key="us_sectors")

        ff3, ff4 = st.columns(2)
        with ff3:
            us_min_score = st.slider("Min Trend Score", 0, 100, 0, key="us_min_score")
        with ff4:
            _sig_opts = ["STRONG UPTREND", "UPTREND", "NEUTRAL", "DOWNTREND", "STRONG DOWNTREND"]
            us_sig_filter = st.multiselect("Signal", _sig_opts, default=_sig_opts, key="us_sig_filter")

        st.markdown("---")
        st.markdown("**🔬 Fundamental Filters**")
        uf1, uf2, uf3, uf4 = st.columns(4)
        with uf1:
            us_min_quality = st.slider("Min Quality", 0, 10, 0, key="us_min_qual")
        with uf2:
            us_min_value = st.slider("Min Value", 0, 10, 0, key="us_min_val")
        with uf3:
            us_min_growth = st.slider("Min Growth", 0, 10, 0, key="us_min_growth")
        with uf4:
            us_min_vol = st.slider("Min Volume Score", 0, 10, 0, key="us_min_vol")

    # Apply filters
    us_fdf = us_df.copy()
    if us_search:
        _q = us_search.lower()
        us_fdf = us_fdf[
            us_fdf["ticker"].str.lower().str.contains(_q) |
            us_fdf["name"].str.lower().str.contains(_q)
        ]
    if us_sel_sector:
        us_fdf = us_fdf[us_fdf["sector"].isin(us_sel_sector)]
    us_fdf = us_fdf[us_fdf["trend_score"] >= us_min_score]
    if us_sig_filter:
        us_fdf = us_fdf[us_fdf["trend_signal"].isin(us_sig_filter)]
    if us_min_quality > 0 and "quality" in us_fdf.columns:
        us_fdf = us_fdf[us_fdf["quality"] >= us_min_quality]
    if us_min_value > 0 and "value" in us_fdf.columns:
        us_fdf = us_fdf[us_fdf["value"] >= us_min_value]
    if us_min_growth > 0 and "growth" in us_fdf.columns:
        us_fdf = us_fdf[us_fdf["growth"] >= us_min_growth]
    if us_min_vol > 0 and "volume_signal_score" in us_fdf.columns:
        us_fdf = us_fdf[us_fdf["volume_signal_score"] >= us_min_vol]

    # Dynamic column filter
    with st.expander("🌪️ **Custom Column Filter**", expanded=False):
        dc1, dc2, dc3 = st.columns([2, 1, 2])
        with dc1:
            us_filter_col = st.selectbox(
                "Filter Column",
                ["RS Score (vs SPY)", "Volatility", "Trend Score", "Distance from 52W High", "Price"],
                key="us_dyn_col",
            )
        _us_col_map = {
            "RS Score (vs SPY)": "comp_rs",
            "Volatility": "volatility",
            "Trend Score": "trend_score",
            "Distance from 52W High": "dist_52w",
            "Price": "currentPrice",
        }
        _us_tcol = _us_col_map[us_filter_col]
        with dc3:
            try:
                _dmin = float(us_fdf[_us_tcol].min()) if _us_tcol in us_fdf and us_fdf[_us_tcol].notna().any() else 0.0
                _dmax = float(us_fdf[_us_tcol].max()) if _us_tcol in us_fdf and us_fdf[_us_tcol].notna().any() else 100.0
                if _dmin == _dmax:
                    _dmax += 1.0
            except Exception:
                _dmin, _dmax = 0.0, 100.0
            us_dyn_range = st.slider(f"{us_filter_col} Range", _dmin, _dmax, (_dmin, _dmax), key="us_dyn_range")
        if _us_tcol in us_fdf.columns and (us_dyn_range[0] > _dmin or us_dyn_range[1] < _dmax):
            us_fdf = us_fdf[
                ((us_fdf[_us_tcol] >= us_dyn_range[0]) & (us_fdf[_us_tcol] <= us_dyn_range[1])) |
                us_fdf[_us_tcol].isna()
            ]
            st.caption(f"Showing {len(us_fdf)} stocks with {us_filter_col} between {us_dyn_range[0]:.1f} and {us_dyn_range[1]:.1f}")

    # ── MAIN TABLE ────────────────────────────────────────────────────────────
    if us_fdf.empty:
        st.warning("No stocks match the current filters.")
    else:
        _us_hdr, _us_dl = st.columns([3, 1])
        with _us_hdr:
            st.subheader(f"Found {len(us_fdf)} US Momentum Stocks")

        _EMOJI = {
            "STRONG UPTREND": "🟢", "UPTREND": "🔵",
            "NEUTRAL": "🟡", "DOWNTREND": "🟠", "STRONG DOWNTREND": "🔴",
        }
        us_fdf = us_fdf.copy()
        us_fdf["signal_display"] = us_fdf["trend_signal"].map(
            lambda s: f"{_EMOJI.get(s, '')} {s}" if s else s
        )
        us_fdf["yf_link"] = us_fdf.apply(lambda r: _google_finance_url(r['ticker']), axis=1)

        _us_disp = [
            "yf_link", "name", "sector", "sub_industry", "currentPrice",
            "signal_display", "trend_score", "comp_rs", "rs_5d", "rs_21d", "rs_63d",
            "volatility", "dna_signal", "dist_52w", "dist_200dma",
            "quality", "value", "growth", "momentum", "volume_signal_score",
        ]
        _us_disp = [c for c in _us_disp if c in us_fdf.columns]

        st.dataframe(
            us_fdf[_us_disp].sort_values("trend_score", ascending=False),
            column_config={
                "yf_link":           st.column_config.LinkColumn("Ticker", display_text=_GF_DISPLAY_RE),
                "name":              "Company",
                "sector":            "Sector",
                "sub_industry":      "Sub-Industry",
                "currentPrice":      st.column_config.NumberColumn("Price", format="$%.2f"),
                "signal_display":    st.column_config.TextColumn("Signal"),
                "trend_score":       st.column_config.ProgressColumn("Trend", format="%d", min_value=0, max_value=100),
                "comp_rs":           st.column_config.NumberColumn("CompRS", format="%+.1f%%", help="Composite RS vs benchmark"),
                "rs_5d":             st.column_config.NumberColumn("RS5 (1W)", format="%+.1f%%"),
                "rs_21d":            st.column_config.NumberColumn("RS21 (1M)", format="%+.1f%%"),
                "rs_63d":            st.column_config.NumberColumn("RS63 (3M)", format="%+.1f%%"),
                "volatility":        st.column_config.NumberColumn("Volatility", format="%.0f%%"),
                "dna_signal":        st.column_config.TextColumn("Signal"),
                "dist_52w":          st.column_config.NumberColumn("% from 52W High", format="%.1f%%"),
                "dist_200dma":       st.column_config.NumberColumn("% vs 200DMA", format="%.1f%%"),
                "quality":           st.column_config.ProgressColumn("Quality", min_value=0, max_value=10, format="%.1f"),
                "value":             st.column_config.ProgressColumn("Value",   min_value=0, max_value=10, format="%.1f"),
                "growth":            st.column_config.ProgressColumn("Growth",  min_value=0, max_value=10, format="%.1f"),
                "momentum":          st.column_config.ProgressColumn("Momentum",min_value=0, max_value=10, format="%.1f"),
                "volume_signal_score": st.column_config.ProgressColumn("Volume", min_value=0, max_value=10, format="%.1f"),
            },
            height=520,
            use_container_width=True,
            hide_index=True,
        )
        with _us_dl:
            st.markdown("<div style='padding-top:10px'></div>", unsafe_allow_html=True)
            _us_export_cols = [c for c in _us_disp if c != 'yf_link']
            _us_export = us_fdf[_us_export_cols].sort_values('trend_score', ascending=False).copy()
            st.download_button(
                label="⬇️ Download CSV",
                data=_us_export.to_csv(index=False),
                file_name=f"us_momentum_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="us_scanner_download",
            )

    # ── SUB-INDUSTRY ROTATION MATRIX (day-by-day accumulating) ─────────────
    st.markdown("---")
    st.markdown("### 🧬 Sub-Industry Rotation Matrix")
    st.caption(
        "Capital flow across S&P 500 sub-industries. "
        "Score 0–100 (percentile rank of CompRS within each day). "
        "🟢 Leaders → 🟡 Mid → 🔴 Laggards.  Accumulates daily each time you visit this page."
    )

    # Save today's snapshot (idempotent — safe to call on every page load)
    try:
        save_us_rotation_snapshot(us_df)
    except Exception as _e:
        st.caption(f"⚠️ Could not save rotation snapshot: {_e}")

    # One-time historical backfill: pre-populate 12 months of monthly RS
    # so sparklines are meaningful from day 1 (runs ~30s, skipped thereafter)
    _us_rot_hist_check = load_us_rotation_history(days=365)
    _needs_backfill = (
        _us_rot_hist_check.empty or
        _us_rot_hist_check["date"].dt.to_period("M").nunique() < 2
    )
    if _needs_backfill:
        with st.spinner("📅 Building 12-month rotation history (one-time, ~30 s)…"):
            try:
                _ran = backfill_us_rotation_if_needed(
                    benchmark=us_scan_benchmark,
                    rs_weights=[(5, us_s_w5), (21, us_s_w21), (63, us_s_w63_n)],
                )
                if _ran:
                    st.success("✅ Historical rotation data ready!")
            except Exception as _be:
                st.caption(f"⚠️ Backfill error: {_be}")

    _us_rot_hist = load_us_rotation_history(days=365)

    if _us_rot_hist.empty:
        st.info("📊 Rotation history will appear after the first page load with live data.")
    else:
        _spiv, _hpiv, _morder = build_rotation_pivot(_us_rot_hist)

        if _spiv.empty:
            st.info("Building rotation history… visit again tomorrow for trend data.")
        else:
            rot_tab1, rot_tab2 = st.tabs(["📋 Rotation Table", "🌡️ Heatmap"])
            with rot_tab1:
                render_us_rotation_table(_spiv, _hpiv, _morder)
            with rot_tab2:
                render_us_rotation_heatmap(_spiv, _hpiv)

    # ── TODAY'S SNAPSHOT: point-in-time RS cross-section heatmap ─────────
    st.markdown("---")
    with st.expander("📸 **Today's Sub-Industry RS Snapshot** (RS5 / RS21 / RS63 / CompRS)", expanded=False):
        if "sub_industry" in us_df.columns:
            _sub_grp = (
                us_df.groupby(["sector", "sub_industry"])[["rs_5d", "rs_21d", "rs_63d", "comp_rs", "trend_score"]]
                .mean()
                .round(2)
                .reset_index()
                .sort_values("comp_rs", ascending=False)
            )
            if not _sub_grp.empty:
                _sub_z   = _sub_grp[["rs_5d", "rs_21d", "rs_63d", "comp_rs"]].values.tolist()
                _sub_y   = [f"{r['sector']} — {r['sub_industry']}" for _, r in _sub_grp.iterrows()]
                _sub_x   = ["RS5 (1W)", "RS21 (1M)", "RS63 (3M)", "CompRS"]
                _sub_txt = [[f"{v:+.1f}" for v in row] for row in _sub_z]

                fig_sub = go.Figure(go.Heatmap(
                    z=_sub_z, x=_sub_x, y=_sub_y,
                    text=_sub_txt, texttemplate="%{text}",
                    colorscale="RdYlGn", zmid=0, showscale=True,
                    colorbar=dict(title="RS %", thickness=12),
                ))
                fig_sub.update_layout(
                    template="plotly_dark",
                    height=max(500, len(_sub_y) * 22),
                    margin=dict(l=280, r=20, t=30, b=30),
                    xaxis=dict(side="top"),
                )
                st.plotly_chart(fig_sub, use_container_width=True, key="us_sub_industry_snapshot")

                st.dataframe(
                    _sub_grp.rename(columns={
                        "sector": "Sector", "sub_industry": "Sub-Industry",
                        "rs_5d": "RS5", "rs_21d": "RS21", "rs_63d": "RS63",
                        "comp_rs": "CompRS", "trend_score": "Avg Trend",
                    }),
                    column_config={
                        "RS5":       st.column_config.NumberColumn(format="%+.2f%%"),
                        "RS21":      st.column_config.NumberColumn(format="%+.2f%%"),
                        "RS63":      st.column_config.NumberColumn(format="%+.2f%%"),
                        "CompRS":    st.column_config.NumberColumn(format="%+.2f%%"),
                        "Avg Trend": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                    },
                    hide_index=True, use_container_width=True, height=400,
                )
        else:
            st.caption("Sub-industry data not available.")

    # ── VCP BACKTEST ─────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📊 **VCP Backtest — Last 6 Months (S&P 500)**", expanded=False):
        st.caption(
            "Walk-forward scan: weekly steps over the last 6 months. "
            "Applies all 5 VCP gates at each point using only data available on that day. "
            "Shows top 15 signals by 10D forward return + summary stats."
        )
        if st.button("▶️ Run VCP Backtest", key="run_vcp_backtest"):
            from utils.advanced_scanners import backtest_vcp_us
            _bt_tickers = us_df.nlargest(150, "trend_score")["ticker"].dropna().tolist()
            with st.spinner(f"Running walk-forward VCP backtest on {len(_bt_tickers)} S&P 500 stocks… (~30–60 s)"):
                _bt_top, _bt_summary = backtest_vcp_us(_bt_tickers, n_months=6, top_n=15)
            if _bt_summary:
                _bm1, _bm2, _bm3, _bm4, _bm5 = st.columns(5)
                _bm1.metric("Total Signals", f"{_bt_summary['total_signals']}")
                _bm2.metric("Hit Rate (10D)", f"{_bt_summary['hit_rate_pct']}%")
                _bm3.metric("Avg Return (10D)", f"{_bt_summary['avg_10d_ret']:+.2f}%")
                _bm4.metric("Best (10D)", f"{_bt_summary['best_10d_ret']:+.2f}%")
                _bm5.metric("Worst (10D)", f"{_bt_summary['worst_10d_ret']:+.2f}%")
            if not _bt_top.empty:
                st.markdown("#### Top 15 Signals by 10D Return")
                _bt_top["INDmoney"] = _bt_top["Ticker"].apply(_google_finance_url)
                st.dataframe(
                    _bt_top[["INDmoney", "Signal Date", "Signal Price",
                              "Compression", "Vol Ratio %", "Dist 52W %",
                              "Return 5D %", "Return 10D %", "Return 21D %"]],
                    column_config={
                        "INDmoney":      st.column_config.LinkColumn("Ticker", display_text=_GF_DISPLAY_RE),
                        "Signal Price":  st.column_config.NumberColumn(format="$%.2f"),
                        "Compression":   st.column_config.NumberColumn("ATR% (10D)", format="%.2f%%"),
                        "Vol Ratio %":   st.column_config.NumberColumn("Vol %", format="%.1f%%"),
                        "Dist 52W %":    st.column_config.NumberColumn("Dist 52W", format="%.1f%%"),
                        "Return 5D %":   st.column_config.NumberColumn("5D Ret", format="%+.2f%%"),
                        "Return 10D %":  st.column_config.NumberColumn("10D Ret", format="%+.2f%%"),
                        "Return 21D %":  st.column_config.NumberColumn("21D Ret", format="%+.2f%%"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )
            elif _bt_summary:
                st.info("No VCP signals found in the last 6 months with the current criteria.")
            else:
                st.warning("Backtest returned no data — check internet connection or reduce ticker count.")

