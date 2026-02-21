import streamlit as st
import pandas as pd
import os
from datetime import datetime
from utils.nifty500_list import TICKERS
from utils.data_engine import get_stock_info, get_stock_history, batch_fetch_tickers
from utils.scoring import calculate_scores, calculate_trend_metrics
from utils.visuals import chart_score_radar, chart_price_history, chart_gauge, chart_market_heatmap, chart_relative_performance
from utils.news_engine import fetch_latest_news
from utils.report_generator import generate_equity_report
from utils.ui_components import css_styles, card_metric, card_verdict, page_header, COLORS
from utils.analytics_engine import analyze_sectors, calculate_cycle_position
from utils.market_explorer import render_market_explorer
from utils.positions import (
    get_all_positions, get_positions_with_pnl, get_position,
    add_position, update_position, remove_position, close_position,
    check_position_alerts, get_summary, migrate_from_legacy, is_position_exists,
    add_to_watchlist
)
from utils.email_notifier import is_email_configured, configure_email, send_weekly_summary, send_trend_change_alert, test_email_connection, get_email_address
from utils.telegram_notifier import send_telegram_message, is_telegram_configured

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
    import os
    if os.path.exists("nifty500_cache.csv"):
        os.remove("nifty500_cache.csv")
    if os.path.exists("market_data.parquet"):
        os.remove("market_data.parquet")
    # Clear ALL caches including @st.cache_data decorated functions
    st.cache_data.clear()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

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

# Data loading (moved to top to ensure availability for all views)
if 'market_data' not in st.session_state:
    from utils.fast_data_engine import load_base_fundamentals, fetch_and_process_market_data
    
    load_status = st.empty()
    progress_bar = st.progress(0, text="Initializing...")
    
    # Step 1: Load fundamentals
    progress_bar.progress(10, text="Step 1/3: Loading fundamentals...")
    fundamentals = load_base_fundamentals()
    
    # Step 2: Fetch live data
    progress_bar.progress(30, text=f"Step 2/3: Fetching live prices for {len(fundamentals)} stocks...")
    df = fetch_and_process_market_data(fundamentals['ticker'].tolist(), fundamentals)
    
    if df.empty:
        progress_bar.empty()
        st.error("⚠️ Data Fetch Failed! Check internet.")
        st.stop()
    
    # Step 3: Finalize
    progress_bar.progress(90, text="Step 3/3: Calculating scores...")
    st.session_state['market_data'] = df
    st.session_state['data_loaded_at'] = datetime.now()
    
    # Cache sector PE for scoring consistency
    if 'pe' in df.columns and 'sector' in df.columns:
         st.session_state['sector_pe_cache'] = df.groupby('sector')['pe'].median().to_dict()
    
    progress_bar.progress(100, text="✅ Ready!")
    import time; time.sleep(0.5)
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

# Debug: Show data verification
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
    
    # Show available price fields from first row
    st.write("**Sample Row Keys (first stock):**")
    sample = df.iloc[0].dropna().to_dict()
    price_keys = [k for k in sample.keys() if any(x in k.lower() for x in ['price', 'average', 'high', 'low', '52', 'week'])]
    st.write(price_keys[:15])


from utils.trend_engine import calculate_sector_history, calculate_stock_trend_history

# (Auto-alert check is now handled by the unified position alert system above)

# === DNA3 MORNING BRIEF AUTO-TRIGGER (Once per day on dashboard load) ===
try:
    import json as _json
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
    "⚖️ Analysis Lab", 
    "⚙️ Strategy & Backtest"
], key="active_workspace")

page = "🌊 Trend Scanner" # Default

if active_workspace == "🔍 Market Specs":
    page = st.sidebar.radio("View", ["🌊 Trend Scanner", "🚀 Live Trading Desk", "🔍 Market Explorer", "📊 Sector Pulse"], key="page_market_specs")
    
elif active_workspace == "📋 Portfolio Manager":
    page = st.sidebar.radio("Tools", ["📊 Return Tracker", "⚠️ Alerts Configuration", "📝 Notes"], key="page_portfolio")
    
elif active_workspace == "⚖️ Analysis Lab":
    page = st.sidebar.radio("Tools", ["⚖️ Compare Stocks", "📉 Deep Dive", "⏳ Time Trends"], key="page_analysis")
    
elif active_workspace == "⚙️ Strategy & Backtest":
    page = st.sidebar.radio("Tools", ["🔬 Strategy Lab", "📈 Portfolio Backtest"], key="page_strategy")

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
            import plotly.graph_objects as go
            
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
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Price comparison chart
            st.markdown("---")
            st.markdown("### 📈 Price Performance (Last 6 Months)")
            
            import yfinance as yf
            from datetime import datetime, timedelta
            
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
                    template='plotly_dark',
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

# --- VIEW: LIVE TRADING DESK (DNA3-V4 + REGIME) ---
elif page == "🚀 Live Trading Desk":
    st.markdown(page_header("🚀 DNA3-V3.1: The Live Trading Desk", "Pure mathematical momentum deployment guided by 15-Year out-of-sample Regime & Seasonality analytics."), unsafe_allow_html=True)
    
    from utils.live_desk import get_live_regime, generate_v3_watchlist
    import yfinance as yf
    from datetime import datetime, timedelta
    
    # 1. Fetch Nifty for Regime calculation
    with st.spinner("Calculating Live Macro Regime..."):
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
            
            # 2. V3.1 Momentum Engine + Seasonal Indicators
            st.markdown("### 📅 DNA3-V3.1: THE WATCHLIST")
            st.caption(f"*Seasonality Warning:* Recent structural regime changes have broken historical calendar correlations. Seasonality is now shown as an **indicator only**, rather than a strict filter.")
            
            with st.spinner("Running V3.1 Engine + Seasonality checks..."):
                v3_watchlist = generate_v3_watchlist(df)
                
                if not v3_watchlist.empty:
                    st.dataframe(
                        v3_watchlist[['Ticker', 'Target', 'Sector', 'Price', 'V3_Score', 'Cyclicity', 'Seasonality', 'PEAD_Edge']],
                        column_config={
                            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                            "Target": "Company Name",
                            "Sector": "Industry Theme",
                            "Price": st.column_config.NumberColumn("CMP", format="₹%.0f"),
                            "V3_Score": st.column_config.ProgressColumn("Trend Conviction", format="%.0f", min_value=0, max_value=100),
                            "Cyclicity": st.column_config.TextColumn("Risk Policy", help="Determines wide (-20%) vs tight (-8%) trailing stops"),
                            "Seasonality": st.column_config.TextColumn("Seasonality", help="Warning indicator only based on 15Y odds"),
                            "PEAD_Edge": st.column_config.TextColumn("Earnings Edge", help="Historical Post-Earnings Reaction")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.info("⚠️ Scanner found absolutely zero safe momentum setups today. Stay in Cash.")
                    
            # 3. Advanced Alpha Scanners (VCP, RS Divergence, Day-0 Shocks)
            st.markdown("---")
            st.markdown("### 🔬 ADVANCED QUANTITATIVE SCANNERS")
            
            from utils.advanced_scanners import find_vcp_setups, find_rs_divergence, find_live_earnings_shocks
            
            # For VCP and Shocks we need full history, which is slow to fetch individually. 
            # We will use the cached market_data if it has valid columns, else we just show a placeholder warning.
            
            # In a production environment with a persistent DB, we would pass the full history dict here.
            # For this Streamlit demo, we'll try to fetch a localized batch for the top 100 stocks to keep it fast.
            # We will use a cached helper to get histories quickly.
            
            @st.cache_data(ttl=3600)
            def get_fast_histories(tickers):
                d = yf.download(tickers, period="3mo", group_by='ticker', threads=True, progress=False)
                hists = {}
                for t in tickers:
                    if t in d.columns.get_level_values(0):
                        sub_df = d[t].dropna(how='all')
                        if not sub_df.empty:
                            hists[t] = sub_df
                return hists
            
            with st.spinner("Running Advanced Pattern Recognition..."):
                top_150_tickers = df.nlargest(150, 'trend_score')['ticker'].tolist()
                hist_dict = get_fast_histories(top_150_tickers)
                
                vcp_list = find_vcp_setups(df[df['ticker'].isin(top_150_tickers)], hist_dict)
                rs_list = find_rs_divergence(df, nifty_live)
                shock_list = find_live_earnings_shocks(df[df['ticker'].isin(top_150_tickers)], hist_dict)
            
            col_adv1, col_adv2, col_adv3 = st.columns(3)
            
            with col_adv1:
                with st.expander(f"🗜️ Volatility Contraction (VCP) [{len(vcp_list)}]", expanded=True):
                    st.caption("Extremely tight 10D range + Volume dried up < 50% of 60D Avg. Indicates supply exhaustion before breakout.")
                    if vcp_list:
                        vcp_df = pd.DataFrame(vcp_list)
                        st.dataframe(
                            vcp_df[['Ticker', 'Price', 'Compression', 'Vol_Ratio']],
                            column_config={
                                "Compression": st.column_config.NumberColumn("10D ATR%", format="%.1f%%"),
                                "Vol_Ratio": st.column_config.NumberColumn("Vol vs 60D", format="%.0f%%")
                            },
                            hide_index=True, use_container_width=True
                        )
                    else:
                        st.info("No pure VCP setups found today.")
                        
            with col_adv2:
                with st.expander(f"🟢 RS Divergence [{len(rs_list)}]", expanded=True):
                    st.caption("Green in a sea of Red. Stocks closing positive > 0.5% while Nifty fell > -0.5% today.")
                    if rs_list:
                        rs_df = pd.DataFrame(rs_list)
                        st.dataframe(
                            rs_df[['Ticker', 'Stock_Ret', 'Delta_RS', 'Dist_52W']],
                            column_config={
                                "Stock_Ret": st.column_config.NumberColumn("Today %", format="%+.1f%%"),
                                "Delta_RS": st.column_config.NumberColumn("vs Nifty %", format="%+.1f%%"),
                                "Dist_52W": st.column_config.NumberColumn("Dist to 52W", format="%.1f%%")
                            },
                            hide_index=True, use_container_width=True
                        )
                    else:
                        st.info("Nifty is not falling today, or no divergences found.")
                        
            with col_adv3:
                with st.expander(f"⚡ Zero-Lag Earnings Shocks [{len(shock_list)}]", expanded=True):
                    st.caption("Day-0 >5% Gap on >300% Volume. Use PEAD Edge to Buy vs Fade.")
                    if shock_list:
                        shk_df = pd.DataFrame(shock_list)
                        st.dataframe(
                            shk_df[['Ticker', 'Jump_Pct', 'Vol_Mult', 'PEAD_Action']],
                            column_config={
                                "Jump_Pct": st.column_config.NumberColumn("Price Jump", format="%+.1f%%"),
                                "Vol_Mult": st.column_config.NumberColumn("Vol Ratio", format="%.1fx average"),
                                "PEAD_Action": "Playbook Action"
                            },
                            hide_index=True, use_container_width=True
                        )
                    else:
                        st.info("No massive >5% on >3x volume earnings shocks detected today.")
                        
        else:
            st.error("Failed to connect to NSE index to calculate regime.")

elif page == "🌊 Trend Scanner":
    
    # === HERO SECTION ===
    st.markdown(page_header("🌊 Alpha Trend Scanner", "Real-time momentum intelligence for Nifty 500 | Powered by AI"), unsafe_allow_html=True)
    
    # === DNA3-V2.1 MODEL PORTFOLIO SECTION (LIVE TRACKING) ===
    import json
    import os
    
    DNA3_SNAPSHOT = "data/dna3_portfolio_snapshot.json"
    DNA3_EQUITY = "data/dna3_equity_curve.csv"
    DNA3_LOG = "data/dna3_trade_log.csv"
    
    if os.path.exists(DNA3_SNAPSHOT):
        try:
            with open(DNA3_SNAPSHOT, 'r') as f:
                dna3_data = json.load(f)
            
            # Load Equity Curve for returns
            live_return = "0.0%"
            equity_val = 1000000
            
            if os.path.exists(DNA3_EQUITY):
                eq_df = pd.read_csv(DNA3_EQUITY)
                if not eq_df.empty:
                    last_eq = eq_df['Equity'].iloc[-1]
                    start_eq = 1000000 # 10L Start
                    ret_abs = (last_eq - start_eq) / start_eq * 100
                    live_return = f"{ret_abs:+.1f}%"
                    equity_val = last_eq

            with st.expander("🧬 **DNA3-V2.1 Model Portfolio (Live Tracking)**", expanded=True):
                d_col1, d_col2 = st.columns([1, 3], gap="medium")
                
                with d_col1:
                    # Live Metrics
                    st.metric("Live Return (Since Feb '26)", live_return, delta="Real Money Performance")
                    st.metric("Net Asset Value", f"₹{equity_val:,.0f}")
                    st.metric("Current Holdings", f"{dna3_data.get('count', 0)} Stocks")
                    
                    st.caption(f"Strategy: Price > MA50 + RS > 0")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("🔄 Refresh"):
                            with st.spinner("Updating Live Portfolio..."):
                                import subprocess
                                subprocess.run(["python", "dna3_current_portfolio.py"])
                                st.rerun()
                    with c2:
                        # Export Trade Log
                        if os.path.exists(DNA3_LOG):
                            with open(DNA3_LOG, "rb") as file:
                                st.download_button(
                                    label="📥 Log",
                                    data=file,
                                    file_name="dna3_trade_log.csv",
                                    mime="text/csv"
                                )
                
                with d_col2:
                    if dna3_data.get('portfolio'):
                        p_df = pd.DataFrame(dna3_data['portfolio'])
                        st.dataframe(
                            p_df[['Ticker', 'Sector', 'Price', 'Entry', 'PnL%', 'RS_Score', 'Dist_MA50']],
                            column_config={
                                "Ticker": "Stock",
                                "Price": st.column_config.NumberColumn("Current Price", format="₹%.2f"),
                                "Entry": st.column_config.NumberColumn("Entry Price", format="₹%.2f"),
                                "PnL%": st.column_config.NumberColumn("Unrealized P&L", format="%+.1f%%"),
                                "RS_Score": st.column_config.ProgressColumn("RS Score", min_value=0, max_value=100, format="%.1f"),
                                "Dist_MA50": st.column_config.NumberColumn("% vs MA50", format="%.1f%%")
                            },
                            hide_index=True,
                            use_container_width=True,
                            height=280
                        )
                    else:
                        st.info("No stocks meet DNA3 criteria currently (Cash Mode).")
        except Exception as e:
            st.error(f"Error loading DNA3 Portfolio: {e}")
    
    # === DNA3 ALERT CONTROLS ===
    with st.expander("🔔 **DNA3 Alert Settings** (Telegram + Email)", expanded=False):
        import json as _json
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
    from utils.market_mood import calculate_mood_metrics, save_mood_snapshot, load_mood_history, chart_market_mood
    
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
    
    # === MARKET TIMING SIGNALS (Based on Analysis) ===
    current_score = avg_trend
    previous_score = mood_history['avg_trend_score'].iloc[-2] if len(mood_history) > 1 else current_score
    
    # Mood Alert Widget
    mood_cols = st.columns([2, 3, 3])
    
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
        # Sector Rotation Signal Panel - Based on Correlation Analysis
        st.markdown("**📊 Sector Rotation Signal**")
        
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
        
        st.caption(f"Optimal Horizon: 60-90 days")
    
    with mood_cols[2]:
        # Enhanced Sector Signal Table based on analysis
        st.markdown("**📈 Sector Signal Strength**")
        
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
    
    # Score change alert
    score_change = current_score - previous_score
    if current_score < 40 and previous_score >= 40:
        st.warning("🚨 **ALERT**: Score dropped below 40 - Potential BUY signal for Midcap/Bank!")
    elif current_score > 65 and previous_score <= 65:
        st.info("📢 **ALERT**: Score crossed above 65 - Consider IT sector, reduce Midcap exposure")
    
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
            breakouts_sorted = breakouts.nsmallest(20, 'dist_52w')
            st.dataframe(
                breakouts_sorted[['ticker', 'name', 'price', 'dist_52w', 'trend_score', 'overall']],
                column_config={
                    "ticker": "Ticker",
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
            import os
            # Remove both caches to force full rebuild
            if os.path.exists("nifty500_cache.csv"): os.remove("nifty500_cache.csv")
            if os.path.exists("market_data.parquet"): os.remove("market_data.parquet")
            
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.subheader(f"Found {len(filtered_df)} Momentum Stocks")
    
    # Ensure columns exist (handle legacy cache)
    for col in ['rs_3m', 'volatility', 'dna_signal']:
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
            "RS Score (vs Nifty)": "rs_3m",
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
            
            filter_val = st.slider(f"Minimum {filter_col}", min_val, max_val, min_val)
            
        # Apply Filter (only if user moved the slider from minimum)
        if target_col in filtered_df.columns and filter_val > min_val:
            # Keep rows where value >= filter OR value is NaN (no data yet)
            filtered_df = filtered_df[
                (filtered_df[target_col] >= filter_val) | (filtered_df[target_col].isna())
            ]
            st.caption(f"Showing {len(filtered_df)} stocks with {filter_col} >= {filter_val:.1f}")

    display_cols = ['ticker', 'name', 'sector', 'price', 'trend_signal', 'trend_score', 'rs_3m', 'volatility', 'dna_signal', 'dist_52w', 'dist_200dma']
    # Add 5-pillar fundamental columns + RS Score for user request
    display_cols.extend(['quality', 'value', 'growth', 'momentum', 'volume_signal_score'])

    st.dataframe(
        filtered_df[display_cols].sort_values(by='trend_score', ascending=False),
        column_config={
            "trend_score": st.column_config.ProgressColumn("Trend Score", format="%d", min_value=0, max_value=100),
            "price": st.column_config.NumberColumn("Price", format="₹ %.2f"),
            "dist_52w": st.column_config.NumberColumn("% from 52W High", format="%.1f%%"),
            "dist_200dma": st.column_config.NumberColumn("% vs 200DMA", format="%.1f%%"),
            "trend_signal": st.column_config.TextColumn("Signal"),
            "quality": st.column_config.ProgressColumn("Quality", min_value=0, max_value=10, format="%.1f"),
            "value": st.column_config.ProgressColumn("Value", min_value=0, max_value=10, format="%.1f"),
            "growth": st.column_config.ProgressColumn("Growth", min_value=0, max_value=10, format="%.1f"),
            "momentum": st.column_config.ProgressColumn("Momentum", min_value=0, max_value=10, format="%.1f"),
            "volume_signal_score": st.column_config.ProgressColumn("Volume", min_value=0, max_value=10, format="%.1f"),
            "rs_3m": st.column_config.NumberColumn("RS vs Nifty", format="%+.1f%%", help="3-Month Relative Strength vs Nifty"),
            "volatility": st.column_config.NumberColumn("Volatility", format="%.0f%%", help="Annualized Price Volatility"),
            "dna_signal": st.column_config.TextColumn("DNA Signal", help="BUY = All DNA-3 filters pass"),
        },
        height=500,
        use_container_width=True,
        hide_index=True
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
                from datetime import datetime
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
                if is_email_configured() and tracked_returns:
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
                    
                    if tracked_returns:
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
    import plotly.express as px
    import plotly.graph_objects as go
    
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
            template='plotly_dark',
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
                    template='plotly_dark',
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
                                template='plotly_dark',
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
                    fig_pie.update_layout(height=300, showlegend=True, template='plotly_dark')
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
                template='plotly_dark',
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
                template='plotly_dark',
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
                            template='plotly_dark',
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
                            fig_pie.update_layout(height=300, template='plotly_dark')
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
    import plotly.express as px
    import plotly.graph_objects as go
    
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
                        fig.update_layout(height=300, template='plotly_dark')
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
                    import plotly.graph_objects as go
                    import yfinance as yf
                    
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
                        height=280, template='plotly_dark',
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
                    from datetime import datetime, timedelta
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
                        from datetime import datetime, timedelta
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
                    from datetime import datetime, timedelta
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
                            import yfinance as yf
                            
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
                                fig.update_layout(height=350, template='plotly_dark', title='Portfolio vs Nifty 500 (Base 100)')
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                fig = px.line(eq_df, x='date', y='equity', title='Equity Curve')
                                fig.update_layout(height=300, template='plotly_dark')
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
    
    # === SECTOR OVERVIEW ===
    sector_analysis = analyze_sectors(df)
    
    if sector_analysis.empty:
        st.error("Could not analyze sectors. Check data quality.")
        st.stop()
    
    # Display sector summary metrics
    st.markdown("### 🏭 Sector Performance Overview")
    
    col1, col2, col3 = st.columns(3)
    top_sector = sector_analysis.loc[sector_analysis['avg_overall'].idxmax()]
    worst_sector = sector_analysis.loc[sector_analysis['avg_overall'].idxmin()]
    top_momentum_sector = sector_analysis.loc[sector_analysis['avg_trend_score'].idxmax()]
    
    col1.metric("🏆 Top Rated Sector", top_sector.name, f"Score: {top_sector['avg_overall']:.1f}")
    col2.metric("📉 Weakest Sector", worst_sector.name, f"Score: {worst_sector['avg_overall']:.1f}")
    col3.metric("🚀 Best Momentum", top_momentum_sector.name, f"Trend: {top_momentum_sector['avg_trend_score']:.0f}")
    
    st.markdown("---")
    
    # Sector table
    st.markdown("### 📋 Sector Breakdown")
    st.dataframe(
        sector_analysis.reset_index().rename(columns={'index': 'Sector'}),
        column_config={
            "Sector": st.column_config.TextColumn("Sector"),
            "count": st.column_config.NumberColumn("Stocks", format="%d"),
            "avg_overall": st.column_config.ProgressColumn("Avg Score", min_value=0, max_value=10, format="%.1f"),
            "avg_trend_score": st.column_config.ProgressColumn("Avg Trend", min_value=0, max_value=100, format="%.0f"),
            "avg_momentum": st.column_config.NumberColumn("Momentum", format="%.1f"),
        },
        hide_index=True,
        height=400
    )
    
    st.markdown("---")
    
    # === SECTOR ROTATION CHART (3M vs 1M Returns) ===
    st.markdown("### 🔄 Sector Rotation Map")
    st.caption("X: 3-Month Return | Y: 1-Month Return | Size: Avg Trend Score | Identifies sector rotation themes")
    
    import plotly.express as px
    import plotly.graph_objects as go
    
    # Calculate sector-level returns
    sector_returns = df.groupby('sector').agg({
        'return_1m': 'mean',
        'return_3m': 'mean',
        'trend_score': 'mean',
        'overall': 'mean',
        'ticker': 'count'
    }).reset_index()
    sector_returns.columns = ['Sector', 'Return_1M', 'Return_3M', 'Avg_Trend', 'Avg_Score', 'Count']
    
    # Fill any NaN values
    sector_returns['Return_1M'] = sector_returns['Return_1M'].fillna(0)
    sector_returns['Return_3M'] = sector_returns['Return_3M'].fillna(0)
    sector_returns['Avg_Trend'] = sector_returns['Avg_Trend'].fillna(50)
    
    # Create the sector rotation scatter plot
    fig_rotation = px.scatter(
        sector_returns,
        x='Return_3M',
        y='Return_1M',
        size='Avg_Trend',
        color='Avg_Score',
        hover_name='Sector',
        hover_data={'Count': True, 'Avg_Score': ':.1f', 'Avg_Trend': ':.0f', 'Return_1M': ':.1f', 'Return_3M': ':.1f'},
        color_continuous_scale='RdYlGn',
        range_color=[3, 7],
        labels={
            'Return_3M': '3-Month Return (%)',
            'Return_1M': '1-Month Return (%)',
            'Avg_Score': 'Quality Score'
        },
        text='Sector'
    )
    
    # Add quadrant lines at 0
    fig_rotation.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    fig_rotation.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
    
    # Quadrant labels
    x_max = max(abs(sector_returns['Return_3M'].min()), abs(sector_returns['Return_3M'].max()), 5)
    y_max = max(abs(sector_returns['Return_1M'].min()), abs(sector_returns['Return_1M'].max()), 3)
    
    fig_rotation.add_annotation(x=x_max*0.8, y=y_max*0.8, text="🚀 LEADING", showarrow=False, 
                                font=dict(color="#00C853", size=14, family="Inter"))
    fig_rotation.add_annotation(x=-x_max*0.8, y=y_max*0.8, text="📈 IMPROVING", showarrow=False, 
                                font=dict(color="#2196F3", size=14, family="Inter"))
    fig_rotation.add_annotation(x=x_max*0.8, y=-y_max*0.8, text="⚠️ WEAKENING", showarrow=False, 
                                font=dict(color="#FF9800", size=14, family="Inter"))
    fig_rotation.add_annotation(x=-x_max*0.8, y=-y_max*0.8, text="🔴 LAGGING", showarrow=False, 
                                font=dict(color="#F44336", size=14, family="Inter"))
    
    fig_rotation.update_traces(textposition='top center', textfont=dict(size=10, color='white'))
    
    fig_rotation.update_layout(
        height=500,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        showlegend=False
    )
    
    st.plotly_chart(fig_rotation, use_container_width=True, key="sector_rotation_map")
    
    # Sector rotation interpretation
    with st.expander("📖 How to Read the Sector Rotation Map"):
        st.markdown("""
        - **🚀 LEADING (Top-Right)**: Sectors with positive 3M AND 1M returns - Strong uptrend, market leaders
        - **📈 IMPROVING (Top-Left)**: Negative 3M but positive 1M - Potential turnaround, momentum building
        - **⚠️ WEAKENING (Bottom-Right)**: Positive 3M but negative 1M - Losing momentum, potential rotation out
        - **🔴 LAGGING (Bottom-Left)**: Negative 3M AND 1M - Avoid, underperforming market
        
        **Size** = Average Trend Score of stocks in sector  
        **Color** = Average Quality Score (Green = High Quality, Red = Low Quality)
        """)
    
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
        top_in_sector = sector_stocks.nlargest(10, 'overall')[['ticker', 'name', 'price', 'overall', 'trend_score', 'trend_signal', 'return_1m', 'return_3m']]
        st.dataframe(
            top_in_sector,
            column_config={
                "ticker": "Ticker",
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
        
        import plotly.express as px
        
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
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            # Quadrant labels
            x_range = chart_df['return_3m'].max() - chart_df['return_3m'].min()
            y_range = chart_df['return_1m'].max() - chart_df['return_1m'].min()
            x_max = max(abs(chart_df['return_3m'].min()), abs(chart_df['return_3m'].max()), 10)
            y_max = max(abs(chart_df['return_1m'].min()), abs(chart_df['return_1m'].max()), 5)
            
            fig.add_annotation(x=x_max*0.7, y=y_max*0.7, text="🚀 Winners", showarrow=False, font=dict(color="green", size=12))
            fig.add_annotation(x=-x_max*0.7, y=y_max*0.7, text="📈 Turnarounds", showarrow=False, font=dict(color="blue", size=12))
            fig.add_annotation(x=x_max*0.7, y=-y_max*0.7, text="⚠️ Fading", showarrow=False, font=dict(color="orange", size=12))
            fig.add_annotation(x=-x_max*0.7, y=-y_max*0.7, text="🔴 Laggards", showarrow=False, font=dict(color="red", size=12))
            
            fig.update_layout(
                height=450,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                template='plotly_dark',
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
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                import yfinance as yf
                import numpy as np
                
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
                import yfinance as yf
                
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
                        
                        fig.update_layout(height=600, template="plotly_dark", hovermode="x unified")
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
                    if scores:
                        report_content = generate_equity_report(target_ticker, info, scores, news_items, hist)
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


