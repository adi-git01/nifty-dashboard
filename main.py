import streamlit as st
import pandas as pd
from utils.nifty500_list import TICKERS
from utils.data_engine import get_stock_info, get_stock_history, batch_fetch_tickers
from utils.scoring import calculate_scores, calculate_trend_metrics
from utils.visuals import chart_score_radar, chart_price_history, chart_gauge, chart_market_heatmap, chart_relative_performance
from utils.news_engine import fetch_latest_news
from utils.report_generator import generate_equity_report
from utils.ui_components import css_styles, card_metric, card_verdict
from utils.analytics_engine import analyze_sectors, calculate_cycle_position
from utils.market_explorer import render_market_explorer
from utils.return_tracker import add_to_tracker, remove_from_tracker, get_all_tracked, calculate_current_returns, get_tracker_summary, detect_trend_changes, export_weekly_summary, is_tracked
from utils.email_notifier import is_email_configured, configure_email, send_weekly_summary, send_trend_change_alert, test_email_connection, get_email_address

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
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Initialize Watchlist
if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = []

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
        "🔥 Turnaround Plays"
    ]
)
st.session_state['preset'] = preset

# === WATCHLIST ===
st.sidebar.markdown("---")
st.sidebar.markdown("### ⭐ Watchlist")
watchlist_count = len(st.session_state['watchlist'])
st.sidebar.write(f"Saved: {watchlist_count} stocks")

if watchlist_count > 0:
    with st.sidebar.expander("View Watchlist"):
        for ticker in st.session_state['watchlist']:
            col1, col2 = st.columns([3, 1])
            col1.write(ticker)
            if col2.button("❌", key=f"rm_{ticker}"):
                st.session_state['watchlist'].remove(ticker)
                st.rerun()
    
    if st.sidebar.button("Clear All", use_container_width=True):
        st.session_state['watchlist'] = []
        st.rerun()

# === ALERT SYSTEM ===
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔔 Alerts")

from utils.alerts import load_alerts, add_alert, remove_alert, check_alerts, get_alert_display_text, get_triggered_alert_text

alerts = load_alerts()
st.sidebar.write(f"Active: {len(alerts)} rules")

# Add new alert
with st.sidebar.expander("➕ Add Alert"):
    alert_ticker = st.text_input("Ticker", placeholder="e.g. HDFCBANK", key="alert_ticker_input")
    alert_metric = st.selectbox("Metric", ["trend_score", "overall", "quality", "value", "growth", "momentum"], key="alert_metric_sel")
    alert_condition = st.selectbox("Condition", [">", "<", ">=", "<="], key="alert_cond_sel")
    alert_threshold = st.number_input("Threshold", value=7.0, step=0.5, key="alert_thresh_input")
    
    if st.button("Create Alert", use_container_width=True, key="create_alert_btn"):
        if alert_ticker:
            ticker = alert_ticker.upper()
            if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
                ticker += ".NS"
            add_alert(ticker, alert_metric, alert_condition, alert_threshold)
            st.success(f"Alert created for {ticker}")
            st.rerun()

# Show existing alerts
if alerts:
    with st.sidebar.expander("View Alerts"):
        for alert in alerts:
            col1, col2 = st.columns([3, 1])
            col1.write(get_alert_display_text(alert))
            if col2.button("❌", key=f"rm_alert_{alert.get('id')}"):
                remove_alert(alert.get('id'))
                st.rerun()

# Data loading (moved to top to ensure availability for all views)
if 'market_data' not in st.session_state:
    with st.spinner("Initializing Alpha Engine... Scanning Market..."):
        raw_df = batch_fetch_tickers(TICKERS)
        
        if raw_df.empty:
            st.error("⚠️ Data Fetch Failed! No data returned. Check internet connection or API limits.")
            st.stop()
        
        # Import historical tracker for multi-period returns
        from utils.historical_tracker import calculate_multi_period_returns
        
        # === PHASE 1: Calculate Sector PE Medians for Relative Valuation ===
        sector_pe = raw_df.groupby('sector')['pe'].median().to_dict()
        st.session_state['sector_pe_cache'] = sector_pe
        
        scored_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        total = len(raw_df)
        
        for idx, (_, row) in enumerate(raw_df.iterrows()):
            # 1. Get base data
            start_data = row.to_dict()
            
            # 2. Multi-period Returns FIRST (needed for momentum)
            returns = calculate_multi_period_returns(row['ticker'])
            start_data.update(returns)
            
            # 3. Tech Trend Scores (before 4-pillar scoring, momentum uses trend_score)
            trends = calculate_trend_metrics(start_data)
            start_data.update(trends)
            
            # 4. 4-Pillar Scoring with Sector-Relative PE + Sector-Specific Weighting
            sector = start_data.get('sector', 'Unknown')
            sector_pe_median = sector_pe.get(sector, None)
            scores = calculate_scores(start_data, sector_pe_median=sector_pe_median, sector=sector)
            start_data.update(scores)
            
            # Recommendation based on overall score
            if scores['overall'] >= 7.5:
                start_data['recommendation'] = "BUY"
            elif scores['overall'] >= 5.0:
                start_data['recommendation'] = "HOLD"
            else:
                start_data['recommendation'] = "AVOID"
            
            scored_data.append(start_data)
            
            # Update progress every 10 stocks
            if idx % 10 == 0:
                progress_bar.progress(min((idx + 1) / total, 1.0))
                status_text.text(f"Calculating metrics... {idx + 1}/{total}")
        
        progress_bar.empty()
        status_text.empty()
        
        st.session_state['market_data'] = pd.DataFrame(scored_data)

df = st.session_state['market_data']
st.sidebar.success(f"Loaded {len(df)} Tickers")

# Check and notify triggered alerts
triggered_alerts = check_alerts(df)
if triggered_alerts:
    for alert in triggered_alerts:
        st.toast(get_triggered_alert_text(alert), icon="🔔")

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

from utils.analytics_engine import analyze_stock_health, safe_get, analyze_sectors

# ... imports ...

from utils.trend_engine import calculate_sector_history, calculate_stock_trend_history

# ... imports ...

# Navigation
page = st.sidebar.radio("Navigation", ["🌊 Trend Scanner", "📊 Return Tracker", "🔍 Market Explorer", "📊 Sector Pulse", "⏳ Time Trends", "📉 Deep Dive"])

# --- VIEW 1: TREND SCANNER ---
if page == "🔍 Market Explorer":
    render_market_explorer()

elif page == "🌊 Trend Scanner":
    
    # === HERO SECTION ===
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 16px; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0; font-size: 2.5em;">🌊 Alpha Trend Scanner</h1>
        <p style="color: rgba(255,255,255,0.8); margin-top: 10px;">
            Real-time momentum intelligence for Nifty 500 | Powered by AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
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
    with st.expander("⚡ Trend Filter", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            # Safely get unique sectors
            df['sector'] = df['sector'].fillna("Unknown")
            all_sectors = sorted(df['sector'].astype(str).unique().tolist())
            sel_sector = st.multiselect("Sector (Empty = All)", all_sectors, default=[]) 
        with c2:
            min_score = st.slider("Min Trend Score", 0, 100, 0)
        with c3:
            # Signal filter - match exact values from scoring.py
            signal_options = ["STRONG UPTREND", "UPTREND", "NEUTRAL", "DOWNTREND", "STRONG DOWNTREND"]
            sig_filter = st.multiselect("Signal", signal_options, default=signal_options)
        
        # Fundamental Filters
        st.markdown("---")
        st.markdown("**🔬 Fundamental Quality Filters**")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            min_quality = st.slider("Min Quality Score", 0, 10, 0, help="Filter for high ROE/Margins")
        with fc2:
            min_value = st.slider("Min Value Score", 0, 10, 0, help="Filter for low PE/PB")
        with fc3:
            min_growth = st.slider("Min Growth Score", 0, 10, 0, help="Filter for high earnings growth")
    
    # Apply Filters
    filtered_df = df.copy()
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
    
    if filtered_df.empty:
        st.warning("No stocks found matching these filters.")
        if st.button("🔄 Force Refresh Data"):
            import os
            if os.path.exists("nifty500_cache.csv"):
                os.remove("nifty500_cache.csv")
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.subheader(f"Found {len(filtered_df)} Momentum Stocks")
    
    display_cols = ['ticker', 'name', 'sector', 'price', 'trend_signal', 'trend_score', 'dist_52w', 'dist_200dma']
    # Add fundamental columns if filters are active or just generally useful
    display_cols.extend(['quality', 'value', 'growth'])

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
                    st.info(f"Switch to 'Deep Dive' tab")
            with btn_col2:
                is_in_watchlist = selected_for_dive in st.session_state.get('watchlist', [])
                if is_in_watchlist:
                    if st.button("⭐ Remove", use_container_width=True):
                        st.session_state['watchlist'].remove(selected_for_dive)
                        st.rerun()
                else:
                    if st.button("➕ Watch", use_container_width=True):
                        st.session_state['watchlist'].append(selected_for_dive)
                        st.toast(f"Added {selected_for_dive} to Watchlist!", icon="⭐")
                        st.rerun()
            with btn_col3:
                # Track button for Return Tracker
                stock_row = filtered_df[filtered_df['ticker'] == selected_for_dive].iloc[0]
                if is_tracked(selected_for_dive):
                    if st.button("📊 Untrack", use_container_width=True):
                        remove_from_tracker(selected_for_dive, stock_row.get('price', 0))
                        st.toast(f"Removed {selected_for_dive} from tracking!", icon="📊")
                        st.rerun()
                else:
                    if st.button("📊 Track", use_container_width=True):
                        add_to_tracker(
                            selected_for_dive,
                            stock_row.get('price', 0),
                            stock_row.get('trend_signal', 'N/A'),
                            stock_row.get('trend_score', 0),
                            stock_row.get('sector', 'Unknown'),
                            stock_row.get('name', selected_for_dive)
                        )
                        st.toast(f"Tracking {selected_for_dive}!", icon="📊")
                        st.rerun()

# --- VIEW: RETURN TRACKER ---
elif page == "📊 Return Tracker":
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #00C853 0%, #1DE9B6 100%); 
                padding: 30px; border-radius: 16px; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0;">📊 Return Tracker</h1>
        <p style="color: rgba(255,255,255,0.9); margin-top: 10px;">
            Track real-world returns from Trend Scanner picks | Validate your alpha
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get current tracking data
    tracker_summary = get_tracker_summary(df)
    tracked_returns = calculate_current_returns(df)
    trend_changes = detect_trend_changes(df)
    
    # === SUMMARY METRICS ===
    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
    
    total_tracked = tracker_summary.get('total_tracked', 0)
    avg_return = tracker_summary.get('total_return_pct', 0)
    positive_count = tracker_summary.get('positive_count', 0)
    negative_count = tracker_summary.get('negative_count', 0)
    change_count = tracker_summary.get('trend_changes', 0)
    
    sum_col1.metric("📈 Tracked Stocks", total_tracked)
    
    return_color = "normal" if avg_return >= 0 else "inverse"
    sum_col2.metric("💰 Avg Return", f"{avg_return:+.1f}%", delta_color=return_color)
    
    sum_col3.metric("✅ Win/Loss", f"{positive_count}/{negative_count}")
    
    sum_col4.metric("⚠️ Trend Changes", change_count, help="Stocks whose trend signal changed since entry")
    
    st.markdown("---")
    
    # === TRACKED PORTFOLIO TABLE ===
    if tracked_returns:
        st.markdown("### 📈 Your Tracked Portfolio")
        
        # Convert to DataFrame for display
        portfolio_data = []
        for stock in tracked_returns:
            entry_date = stock.get('entry_date', '')[:10] if stock.get('entry_date') else 'N/A'
            
            portfolio_data.append({
                'Ticker': stock.get('ticker', '').replace('.NS', ''),
                'Name': stock.get('name', 'N/A')[:25],
                'Sector': stock.get('sector', 'N/A')[:20],
                'Entry Price': stock.get('entry_price', 0),
                'Current Price': stock.get('current_price', 0) or 0,
                'Return %': stock.get('return_pct', 0) or 0,
                'Days': stock.get('days_tracked', 0),
                'Entry Signal': stock.get('entry_trend_signal', 'N/A'),
                'Current Signal': stock.get('current_signal', 'N/A'),
                'Changed': '⚠️' if stock.get('signal_changed') else '✓',
                'ticker_full': stock.get('ticker', '')
            })
        
        portfolio_df = pd.DataFrame(portfolio_data)
        
        # Styled display
        st.dataframe(
            portfolio_df[['Ticker', 'Name', 'Entry Price', 'Current Price', 'Return %', 'Days', 'Entry Signal', 'Current Signal', 'Changed']],
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Name": st.column_config.TextColumn("Company"),
                "Entry Price": st.column_config.NumberColumn("Entry ₹", format="%.2f"),
                "Current Price": st.column_config.NumberColumn("Current ₹", format="%.2f"),
                "Return %": st.column_config.NumberColumn("Return", format="%+.1f%%"),
                "Days": st.column_config.NumberColumn("Days", format="%d"),
                "Entry Signal": st.column_config.TextColumn("Entry Signal"),
                "Current Signal": st.column_config.TextColumn("Now"),
                "Changed": st.column_config.TextColumn("Status", help="⚠️ = Trend changed from entry"),
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Quick Actions for removal
        st.markdown("---")
        st.markdown("### 🔧 Manage Tracked Stocks")
        
        mgmt_col1, mgmt_col2 = st.columns([3, 1])
        
        with mgmt_col1:
            ticker_options = [s.get('ticker', '') for s in tracked_returns]
            selected_to_remove = st.selectbox(
                "Select stock to remove", 
                ticker_options,
                format_func=lambda x: x.replace('.NS', '').replace('.BO', '')
            )
        
        with mgmt_col2:
            if st.button("❌ Remove from Tracker", use_container_width=True, type="secondary"):
                if selected_to_remove:
                    current_price = df[df['ticker'] == selected_to_remove]['price'].values
                    price = current_price[0] if len(current_price) > 0 else None
                    remove_from_tracker(selected_to_remove, price)
                    st.toast(f"Removed {selected_to_remove} from tracker!", icon="✅")
                    st.rerun()
        
    else:
        st.info("📭 No stocks being tracked yet. Go to **Trend Scanner** and click **📊 Track** on any stock to start tracking returns!")
        
        # Quick add from top performers
        st.markdown("---")
        st.markdown("### 🚀 Quick Add: Top Trending Stocks")
        st.caption("One-click add from top performers")
        
        top_stocks = df.nlargest(6, 'trend_score')[['ticker', 'name', 'price', 'trend_score', 'trend_signal', 'sector']]
        
        cols = st.columns(3)
        for idx, (_, stock) in enumerate(top_stocks.iterrows()):
            with cols[idx % 3]:
                ticker = stock['ticker']
                if st.button(
                    f"📊 {ticker.replace('.NS', '')}\n{stock['trend_score']:.0f}",
                    key=f"quick_track_{ticker}",
                    use_container_width=True
                ):
                    add_to_tracker(
                        ticker,
                        stock['price'],
                        stock['trend_signal'],
                        stock['trend_score'],
                        stock['sector'],
                        stock['name']
                    )
                    st.toast(f"Now tracking {ticker}!", icon="📊")
                    st.rerun()
    
    # === TREND CHANGES SECTION ===
    if trend_changes:
        st.markdown("---")
        st.markdown("### ⚠️ Trend Changes Detected")
        st.caption("These stocks have changed their trend signal since you started tracking them")
        
        for change in trend_changes:
            entry_signal = change.get('entry_trend_signal', 'N/A')
            current_signal = change.get('current_signal', 'N/A')
            return_pct = change.get('return_pct', 0) or 0
            
            # Determine color based on change direction
            if 'UPTREND' in entry_signal and 'DOWNTREND' in current_signal:
                alert_color = "#FF5252"  # Red - bearish change
            elif 'DOWNTREND' in entry_signal and 'UPTREND' in current_signal:
                alert_color = "#00C853"  # Green - bullish change
            else:
                alert_color = "#FF9800"  # Orange - neutral change
            
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 10px; 
                        margin-bottom: 10px; border-left: 4px solid {alert_color};">
                <strong>{change.get('ticker', '').replace('.NS', '')}</strong> - {change.get('name', 'N/A')}<br>
                <span style="color: #888;">Trend:</span> {entry_signal} → <span style="color: {alert_color};">{current_signal}</span><br>
                <span style="color: #888;">Return since entry:</span> <span style="color: {'#00C853' if return_pct >= 0 else '#FF5252'};">{return_pct:+.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
    
    # === EMAIL SETTINGS SECTION ===
    st.markdown("---")
    st.markdown("### 📧 Email Notifications")
    
    email_col1, email_col2 = st.columns([2, 1])
    
    with email_col1:
        if is_email_configured():
            email_addr = get_email_address()
            st.success(f"✅ Email configured: {email_addr}")
        else:
            st.warning("⚠️ Email not configured. Set up Gmail to receive weekly reports and trend alerts.")
            
            with st.expander("📧 Configure Email"):
                gmail_input = st.text_input("Gmail Address", placeholder="your.email@gmail.com")
                app_pass_input = st.text_input("App Password", type="password", placeholder="xxxx xxxx xxxx xxxx", 
                                               help="Generate at myaccount.google.com/apppasswords")
                
                if st.button("💾 Save Email Settings", use_container_width=True):
                    if gmail_input and app_pass_input:
                        configure_email(gmail_input, app_pass_input)
                        st.success("Email configured! Testing connection...")
                        success, msg = test_email_connection()
                        if success:
                            st.success("✅ Test email sent! Check your inbox.")
                        else:
                            st.error(f"❌ {msg}")
                        st.rerun()
                    else:
                        st.error("Please fill in both fields")
    
    with email_col2:
        if is_email_configured() and tracked_returns:
            if st.button("📤 Send Report Now", type="primary", use_container_width=True):
                with st.spinner("Generating and sending report..."):
                    summary_data = export_weekly_summary(df)
                    success, msg = send_weekly_summary(summary_data)
                    if success:
                        st.success("✅ Weekly report sent!")
                    else:
                        st.error(f"❌ {msg}")
            
            # Trend change alert button
            if trend_changes:
                if st.button("🔔 Send Trend Alert", use_container_width=True):
                    success, msg = send_trend_change_alert(trend_changes)
                    if success:
                        st.success("✅ Trend change alert sent!")
                    else:
                        st.error(f"❌ {msg}")
        elif is_email_configured():
            st.info("Add stocks to send reports")
    
    # === PERFORMANCE INSIGHTS ===
    if tracked_returns and len(tracked_returns) >= 2:
        st.markdown("---")
        st.markdown("### 📈 Performance Insights")
        
        best = tracker_summary.get('best_performer')
        worst = tracker_summary.get('worst_performer')
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            if best:
                st.markdown(f"""
                <div style="background: rgba(0,200,83,0.1); padding: 20px; border-radius: 12px; border: 1px solid rgba(0,200,83,0.3);">
                    <h4 style="color: #00C853; margin: 0;">🏆 Best Performer</h4>
                    <h2 style="margin: 10px 0;">{best.get('ticker', '').replace('.NS', '')}</h2>
                    <p style="font-size: 24px; color: #00C853; margin: 0;">{best.get('return_pct', 0):+.1f}%</p>
                    <p style="color: #888; margin: 5px 0 0 0;">Tracked for {best.get('days_tracked', 0)} days</p>
                </div>
                """, unsafe_allow_html=True)
        
        with insight_col2:
            if worst and worst.get('return_pct', 0) < 0:
                st.markdown(f"""
                <div style="background: rgba(255,82,82,0.1); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,82,82,0.3);">
                    <h4 style="color: #FF5252; margin: 0;">📉 Needs Attention</h4>
                    <h2 style="margin: 10px 0;">{worst.get('ticker', '').replace('.NS', '')}</h2>
                    <p style="font-size: 24px; color: #FF5252; margin: 0;">{worst.get('return_pct', 0):+.1f}%</p>
                    <p style="color: #888; margin: 5px 0 0 0;">Tracked for {worst.get('days_tracked', 0)} days</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(102,126,234,0.1); padding: 20px; border-radius: 12px; border: 1px solid rgba(102,126,234,0.3);">
                    <h4 style="color: #667eea; margin: 0;">🎯 All Green!</h4>
                    <p style="color: #888; margin: 10px 0 0 0;">All tracked stocks are in profit! Nice picks!</p>
                </div>
                """, unsafe_allow_html=True)

# --- VIEW 2: SECTOR PULSE ---
elif page == "📊 Sector Pulse":
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                padding: 30px; border-radius: 16px; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0;">📊 Sector Pulse</h1>
        <p style="color: rgba(255,255,255,0.8); margin-top: 10px;">
            Deep dive into sector performance and stock rotation dynamics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #FF6B6B 0%, #556270 100%); 
                padding: 30px; border-radius: 16px; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0;">⏳ Time Travel Trends</h1>
        <p style="color: rgba(255,255,255,0.8); margin-top: 10px;">
            Visualize Market Cycles and Historical Momentum
        </p>
    </div>
    """, unsafe_allow_html=True)
    
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


