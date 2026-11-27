"""Trend Scanner page view (extracted from main.py)."""
from __future__ import annotations

from datetime import datetime
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from utils.market_breadth import render_breadth_widget
from utils.market_mood import calculate_mood_metrics, chart_market_mood, load_mood_history
from utils.positions import add_position, add_to_watchlist, get_position, is_position_exists, remove_position
from utils.ui_components import hero_pnl_card, page_header


def render(df: pd.DataFrame, *, yf_download_cached, nifty_recent) -> None:
    df = df[df['extended'] != True].copy() if (df is not None and not df.empty and 'extended' in df.columns) else (df.copy() if df is not None else pd.DataFrame())

    # === HERO SECTION ===
    st.markdown(page_header("🌊 Alpha Trend Scanner", "Real-time momentum intelligence for Liquid NSE Equities (≥ ₹2 Cr/d) | Powered by AI"), unsafe_allow_html=True)

    # SIMBA sits beside the scanner; key + on_change keeps the chosen tab across reruns, and
    # .open lets the hidden tab skip its work (the scanner body is heavy).
    _ts_main, _ts_simba, _ts_agree = st.tabs(["🌊 Scanner", "🦁 SIMBA", "🤝 Agreement"], key="ts_tab",
                                             on_change="rerun")
    with _ts_simba:
        if _ts_simba.open:
            from utils.simba_view import render as _simba_render
            _simba_render(df)
    with _ts_agree:
        if _ts_agree.open:
            from utils.consensus_view import render as _agree_render
            _agree_render(df)
    with _ts_main:
        if _ts_main.open is not False:

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
                        holdings_count=dna3_data.get('count', 0),
                        as_of=(pd.Timestamp(dna3_data['date']).strftime('%d %b') if dna3_data.get('date') else None)
                    ), unsafe_allow_html=True)
                    # the figure moves once a day, after the evening run: say so when that run is behind
                    try:
                        _age = int(np.busday_count(pd.Timestamp(dna3_data['date']).date(), pd.Timestamp.now(tz='Asia/Kolkata').date()))
                        _late = _age >= 2 or (_age == 1 and pd.Timestamp.now(tz='Asia/Kolkata').hour >= 19)
                        if _late:
                            st.caption(f"⏳ Last updated after the close on {pd.Timestamp(dna3_data['date']):%a %d %b}. The evening run for "
                                       "the latest session has not landed yet; the figure changes when it does.")
                    except Exception:
                        pass
        
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
                                nifty_eq = yf_download_cached("^NSEI", start=start_str, progress=False,
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
                                fig.add_hline(y=100, line_dash="dash", line_color="rgba(128,128,128,0.5)", line_width=1)
                    
                                fig.update_layout(
                                    title=None,
                                    template='plotly_white',
                                    height=280,
                                    margin=dict(l=10, r=10, t=10, b=10),
                                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                                    xaxis=dict(showgrid=False),
                                    yaxis=dict(title='Normalized (100 = Start)', showgrid=True,
                                               gridcolor='rgba(0,0,0,0.06)'),
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
                                    "RS_Score": st.column_config.NumberColumn("RS Score", format="%+.1f%%"),
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
                                        live_prices = yf_download_cached(tuple(held_tickers), period="100d", group_by='ticker',
                                                                 threads=False, progress=False, auto_adjust=True)
                                        nifty_hist = nifty_recent("100d").copy()
                                        if nifty_hist.index.tz is not None:
                                            nifty_hist.index = nifty_hist.index.tz_localize(None)
                                        nifty_price = float(nifty_hist['Close'].iloc[-1])
                            
                                        # RS weight config (same as OptComp-V22)
                                        rs_weights = [(5, 0.10), (21, 0.50), (63, 0.40)]
                            
                                        updated_portfolio = []
                                        for p in dna3_data['portfolio']:
                                            t = p['Ticker']
                                            try:
                                                if isinstance(live_prices.columns, pd.MultiIndex):
                                                    _lv0 = live_prices.columns.get_level_values(0)
                                                    stock_df = live_prices[t].dropna(how='all') if t in _lv0 else None
                                                else:
                                                    stock_df = live_prices.dropna(how='all')
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

                                        # Atomic write — this is the same live
                                        # portfolio state file the CI engine writes;
                                        # a truncated write here is just as capable
                                        # of wiping the portfolio on next load.
                                        from utils.atomic_io import atomic_json_dump, atomic_to_csv
                                        atomic_json_dump(dna3_data, DNA3_SNAPSHOT, indent=4)
                                        if os.path.exists(DNA3_EQUITY):
                                            try:
                                                _eq_live = pd.read_csv(DNA3_EQUITY)
                                                _today_s = dna3_data['date']
                                                _new_row = pd.DataFrame([{
                                                    'Date': _today_s, 'Equity': round(float(total_equity), 2),
                                                    'Cash': round(float(dna3_data.get('cash', 0)), 2),
                                                    'Holdings': len(updated_portfolio),
                                                    'Regime': dna3_data.get('regime', ''),
                                                }])
                                                _eq_live = _eq_live[_eq_live['Date'].astype(str) != _today_s]
                                                _eq_live = pd.concat([_eq_live, _new_row], ignore_index=True)
                                                atomic_to_csv(_eq_live, DNA3_EQUITY, index=False)
                                            except Exception:
                                                pass

                                        st.toast("✅ Portfolio prices updated!", icon="🔄")
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Refresh failed: {e}")
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

            col1.metric("🚀 Strong Momentum", f"{strong_uptrends}", help="Stocks rated STRONG UPTREND (Trend Score >= 75)")
            col2.metric("📈 Total Uptrends", f"{total_uptrends}", help="Stocks in upward trajectory")
            col3.metric("📊 Avg Trend Score", f"{avg_trend:.0f}/100", help="Market-wide momentum gauge")
            col4.metric("🔥 Breakout Alerts", f"{breakout_count}", help="Near 52-week highs")

            # === MARKET MOOD HISTORY CHART ===
            # market_mood imports at top

            # Save today's snapshot
            mood_metrics = calculate_mood_metrics(df)
            # (not saved here: the EOD run records the day's mood; saving on page views is what
            # put weekend and holiday rows into the history)

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
                    # Closest to 52W high first (0.0% at top, -1.9% below)
                    breakouts_sorted = breakouts.nlargest(20, 'dist_52w').copy()
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
                    # Industry (23 broad groups) -> sub-industry (the 58 groups the Industry board ranks).
                    # The data has no finer level than these two.
                    from utils.nifty1000_list import SUB_INDUSTRY_MAP as _SUBMAP
                    df['sector'] = df['sector'].fillna("Unknown")
                    df['sub_industry'] = df['ticker'].map(_SUBMAP).fillna(df.get('sector_granular', df['sector'])).fillna("Unknown")
                    sel_sector = st.multiselect("Industry (Empty = All)", sorted(df['sector'].astype(str).unique().tolist()),
                                                default=[], key="scan_industry")
                    _sub_pool = df[df['sector'].isin(sel_sector)] if sel_sector else df
                    sel_sub = st.multiselect("Sub-industry (Empty = All)", sorted(_sub_pool['sub_industry'].astype(str).unique().tolist()),
                                             default=[], key="scan_sub",
                                             help="Narrowed to the industries picked above. These are the groups ranked on the Industry board.")
                    try:
                        from utils.industry_board_view import get_board as _gb
                        _B = _gb(df)
                        _grp = dict(zip(_B['industry'], _B['group'])) if len(_B) else {}
                    except Exception:
                        _grp = {}
                    df['ind_status'] = df['sub_industry'].map(_grp).fillna("not ranked")
                    sel_status = st.multiselect("Sub-industry status (Empty = All)", ["LEADING", "IMPROVING", "WEAKENING", "LAGGING"],
                                                default=[], key="scan_status",
                                                help="From the Industry board: e.g. pick LEADING and IMPROVING to keep only stocks in strong groups")
        
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

            # Pre-compute cross-sectional entry/breakout/momentum ranks BEFORE filtering so microcaps/filtered subsets
            # never get re-ranked against only the filtered slice.
            from utils.entry_timing import add_entry_freshness, add_position_sizing
            from utils.breakout_tags import (add_breakout_tags, add_momentum_tags, BREAKOUT_OPTIONS, BREAKOUT_HELP,
                                             MOMO_HELP, SIZE_HELP, FLAG_HELP)
            df = add_entry_freshness(df)
            df = add_momentum_tags(add_breakout_tags(df))

            # Apply Filters
            filtered_df = df.copy()

            # 1. Search Filter
            if search_query:
                query = search_query.lower().strip()
                filtered_df = filtered_df[
                    filtered_df['ticker'].fillna('').astype(str).str.lower().str.contains(query, regex=False) |
                    filtered_df['name'].fillna('').astype(str).str.lower().str.contains(query, regex=False)
                ]
    
            if sel_sector:
                filtered_df = filtered_df[filtered_df['sector'].isin(sel_sector)]
            if sel_sub:
                filtered_df = filtered_df[filtered_df['sub_industry'].isin(sel_sub)]
            if sel_status:
                filtered_df = filtered_df[filtered_df['ind_status'].isin(sel_status)]
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
                filtered_df = filtered_df[(filtered_df['quality'] >= 6) & (filtered_df['value'] >= 6)]
                st.info("Preset: Quality (Quality >= 6) + Reasonable Value (Value >= 6)")
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
                    st.cache_data.clear()
                    for key in ('market_data', '_loaded_live_mode', 'data_loaded_at', 'nifty_data', 'nifty_fetched_at'):
                        st.session_state.pop(key, None)
                    st.rerun()

            from utils.ui_search import search_box as _search_box
            filtered_df = _search_box(filtered_df, "scan_search",
                                      label="🔎 Search name or ticker in the results below")
            _nifty_hdr, _nifty_dl = st.columns([3, 1])
            with _nifty_hdr:
                st.subheader(f"Found {len(filtered_df)} Momentum Stocks")

            # Ensure columns exist (handle legacy cache)
            for col in ['comp_rs', 'volatility', 'dna_signal']:
                if col not in filtered_df.columns:
                    filtered_df[col] = None

            # === ADJUSTABLE RS WEIGHTS (default = OptComp-V22: 10 / 50 / 40) ===
            # comp_rs ("RS vs Nifty") is a weighted blend of RS5 (1W), RS21 (1M) and
            # RS63 (3M) relative strength. Those components are stored per-stock in the
            # cache, so the blend can be re-weighted live here without re-fetching.
            # Defaults reproduce the engine's stored value exactly.
            _DEFAULT_RS_W = (10.0, 50.0, 40.0)
            if all(c in filtered_df.columns for c in ('rs_1w', 'rs_1m', 'rs_3m')):
                with st.expander("⚖️ **Adjust RS Weights** — RS5 / RS21 / RS63 (default 10 / 50 / 40)", expanded=False):
                    st.caption("Re-weight the Composite RS vs Nifty blend. Weights are "
                               "normalized, so they need not sum to 100. Defaults match "
                               "the OptComp-V22 engine.")
                    # A form, so stepping a weight does not rerun the whole page on every
                    # click; the table updates once, on Apply.
                    with st.form("rs_weights_form", border=False):
                        _wc1, _wc2, _wc3, _wc4 = st.columns([1, 1, 1, 1])
                        with _wc1:
                            st.number_input("RS5 (1W) %", min_value=0.0, max_value=100.0,
                                            value=_DEFAULT_RS_W[0], step=5.0, key="rs_w5")
                        with _wc2:
                            st.number_input("RS21 (1M) %", min_value=0.0, max_value=100.0,
                                            value=_DEFAULT_RS_W[1], step=5.0, key="rs_w21")
                        with _wc3:
                            st.number_input("RS63 (3M) %", min_value=0.0, max_value=100.0,
                                            value=_DEFAULT_RS_W[2], step=5.0, key="rs_w63")
                        with _wc4:
                            st.write("")  # vertical spacer to align buttons with inputs
                            st.form_submit_button("✓ Apply weights", use_container_width=True, type="primary")
                    if st.button("↺ Reset to default (10 / 50 / 40)", key="rs_w_reset"):
                        for _k in ('rs_w5', 'rs_w21', 'rs_w63'):
                            st.session_state.pop(_k, None)
                        st.rerun()
                    _w5 = st.session_state.get("rs_w5", _DEFAULT_RS_W[0])
                    _w21 = st.session_state.get("rs_w21", _DEFAULT_RS_W[1])
                    _w63 = st.session_state.get("rs_w63", _DEFAULT_RS_W[2])

                    _wsum = _w5 + _w21 + _w63
                    if _wsum <= 0:
                        st.warning("Weights sum to 0 — falling back to defaults (10 / 50 / 40).")
                        _w5, _w21, _w63 = _DEFAULT_RS_W
                        _wsum = sum(_DEFAULT_RS_W)

                    _n5, _n21, _n63 = _w5 / _wsum, _w21 / _wsum, _w63 / _wsum
                    filtered_df = filtered_df.copy()
                    # NaN components (e.g. rows scrubbed for a corrupt bar) stay NaN.
                    filtered_df['comp_rs'] = (
                        filtered_df['rs_1w'] * _n5
                        + filtered_df['rs_1m'] * _n21
                        + filtered_df['rs_3m'] * _n63
                    ).round(2)

                    if (_w5, _w21, _w63) != _DEFAULT_RS_W:
                        st.caption(f"🔧 Custom weights active — RS5 **{_n5*100:.0f}%** · "
                                   f"RS21 **{_n21*100:.0f}%** · RS63 **{_n63*100:.0f}%**. "
                                   f"The RS vs Nifty column and sorting below reflect these.")

            # === DISPLAY-TIME RS SANITY GUARD (defense in depth) ===
            # The engine sanitizes corrupt price bars at source and the cache is
            # patched, but a stale browser session or a future missed bad bar could
            # still surface a physically-impossible RS (e.g. ARIHANT +759%). A real
            # 3-month relative strength tops out near +150pp (CEMPRO ~147), so
            # |comp_rs| > 200 or |rs_3m| > 300 is always a bad-bar artifact — blank it
            # rather than ever render garbage in the "RS vs Nifty" column.
            if 'comp_rs' in filtered_df.columns:
                _rs_bad = pd.to_numeric(filtered_df['comp_rs'], errors='coerce').abs() > 200
                if 'rs_3m' in filtered_df.columns:
                    _rs_bad = _rs_bad | (pd.to_numeric(filtered_df['rs_3m'], errors='coerce').abs() > 300)
                if _rs_bad.any():
                    filtered_df = filtered_df.copy()
                    filtered_df.loc[_rs_bad, ['comp_rs', 'rs_3m']] = np.nan
                    st.caption(f"⚠️ {int(_rs_bad.sum())} stock(s) hidden from RS ranking due to a "
                               f"suspected bad price bar (blanked, not deleted).")


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
        
                # Apply Filter (only if user moved the slider from defaults; exclude NaN when narrowed)
                if target_col in filtered_df.columns and (filter_val[0] > min_val or filter_val[1] < max_val):
                    _vals = pd.to_numeric(filtered_df[target_col], errors='coerce')
                    filtered_df = filtered_df[(_vals >= filter_val[0]) & (_vals <= filter_val[1])]
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

            # === ENTRY FRESHNESS FILTER ===
            # (computed on full universe `df` above before filtering; ensure columns exist if custom path skipped)
            if 'entry_label' not in filtered_df.columns:
                filtered_df = add_entry_freshness(filtered_df)
            if 'momo_entry' not in filtered_df.columns:
                filtered_df = add_momentum_tags(add_breakout_tags(filtered_df))

            # === ENTRY STATUS FILTER ===
            _ENTRY_ORDER = ['🔵 Pullback Buy', '🟢 Actionable', '🟡 Extended', '🔴 Late/Fading', '⚪ Weak']
            # Backtest 2016-2026, median vs the typical stock over 3 / 6 months (analysis/scanner_tags_study.csv)
            _ENTRY_HELP = ("Backtest 2016-26, vs the typical stock over 3 / 6 months: 🟢 Actionable +0.6 / +1.1 pp "
                           "(most consistent, 11 of 13 years) · 🔵 Pullback +0.3 / +0.8 · 🟡 Extended +1.0 / +2.4 but weak "
                           "in the first 2 weeks, so size smaller rather than skip · 🔴 Late/Fading +0.2 / +0.7 · "
                           "⚪ Weak -0.9 / -1.9 (the only tag to avoid). All tags do better in a leader industry.")
            _present = [s for s in _ENTRY_ORDER if s in set(filtered_df['entry_label'])]
            if _present:
                _sel_entry = st.multiselect(
                    "🎯 Filter by Entry Status",
                    options=_present, default=_present,
                    help=_ENTRY_HELP + " Deselect to hide; empty = show all.",
                )
                # Filter only when the user narrows the set (empty = show all, avoids a blank table).
                if _sel_entry and len(_sel_entry) < len(_present):
                    filtered_df = filtered_df[filtered_df['entry_label'].isin(_sel_entry)]
                    st.caption(f"Showing **{len(filtered_df)}** stocks with entry status: {' · '.join(_sel_entry)}")

            # === BREAKOUT / INDUSTRY FILTERS ===
            _mf1, _mf2 = st.columns([1, 2])
            with _mf1:
                _momo_only = st.toggle("🚀 Momentum entry (breakout / squeeze)", value=False, help=MOMO_HELP)
            with _mf2:
                _tiers = st.multiselect("📏 Size tier", options=["Large", "Mid", "Small", "Micro"], default=[], help=SIZE_HELP)
            if _momo_only:
                filtered_df = filtered_df[filtered_df['momo_entry'] != ""]
            if _tiers:
                filtered_df = filtered_df[filtered_df['size_tier'].isin(_tiers)]
            if _momo_only or _tiers:
                st.caption(f"Showing **{len(filtered_df)}** stocks"
                           + ("" if 'sq_days' in filtered_df.columns else
                              " -- squeeze releases (🎯) arrive with the next daily engine run"))
            _bf1, _bf2, _bf3 = st.columns([2, 1, 1])
            with _bf1:
                _sel_bo = st.multiselect("⚡ Fresh breakout", options=BREAKOUT_OPTIONS, default=[], help=BREAKOUT_HELP)
            with _bf2:
                _lead_only = st.toggle("🟢 Leader industries only", value=False,
                                       help="Sub-industry rotation score >= 70 (the heatmap's green band). Every entry tag "
                                            "did better there; the same breakout in a laggard industry was worth ~1/3.")
            with _bf3:
                _anchor_only = st.toggle("🔊 Volume anchor only", value=False,
                                         help="Breakout day volume >= 3x its 50-day median, turnover >= Rs 5 cr, CLV >= 0.5, "
                                              "and no close since below that day's low -- added ~+1 pp over 3-6 months. "
                                              "Implies a fresh breakout. Not the same as the Technicals tab's anchor, which is "
                                              "the biggest-volume day of the last 60 sessions (it can be an older spike that "
                                              "closed mid-range -> 'MIXED anchor', or one whose low has broken -> VETO).")
            if 'hi52_bo_days' not in filtered_df.columns:
                st.info("⚡ Breakout data arrives with the next daily engine run -- the loaded snapshot predates it. "
                           "Industry and % from ATH already work.")
            _bo_mask = pd.Series(True, index=filtered_df.index)
            if _sel_bo:
                _types = {"ATH breakout": "ATH", "52W breakout": "52W"}
                _bo_mask &= filtered_df['bo_type'].isin([_types[x] for x in _sel_bo])
            if _anchor_only:
                _bo_mask &= filtered_df['bo_anchor']
            if _lead_only:
                _bo_mask &= filtered_df['ind_score'] >= 70
            if _sel_bo or _anchor_only or _lead_only:
                filtered_df = filtered_df[_bo_mask]
                st.caption(f"Showing **{len(filtered_df)}** stocks")

            # === BREAKOUTS THIS WEEK + LOG (utils/breakout_log.py) ===
            try:
                from utils.breakout_log import load_log as _bo_load, this_week as _bo_week
                _bo_log = _bo_load()
            except Exception:
                _bo_log = pd.DataFrame()
            if not _bo_log.empty:
                _now = df.drop_duplicates(subset=['ticker'], keep='last').set_index('ticker')
                _g1_now = (_now['fx_gate1'].fillna(False).astype(bool) if 'fx_gate1' in _now.columns else pd.Series(False, index=_now.index))
                _mom_now_s = _now['momo_entry'].ne("") | ((_now['mom_rank'] >= 90) & _g1_now)
                _bo_log = _bo_log.assign(
                    momentum_now=_bo_log.ticker.map(_mom_now_s).fillna(False),
                    flags_now=_bo_log.ticker.map(_now['watch_flags']).fillna(""),
                    screener_link="https://www.screener.in/company/" + _bo_log.ticker.str.replace('.NS', '', regex=False) + "/")
                _bo_cfg = {
                    "breakout_date": st.column_config.DateColumn("Breakout"),
                    "screener_link": st.column_config.LinkColumn("Ticker", display_text=r"https://www\.screener\.in/company/(.*?)/"),
                    "kind": st.column_config.TextColumn("Type"),
                    "momentum": st.column_config.CheckboxColumn("Momentum then", help="Top-10% 6-month return + Gate 1 on the breakout day -- the entry that passed in the study"),
                    "momentum_now": st.column_config.CheckboxColumn("Momentum now"),
                    "size_tier": st.column_config.TextColumn("Size", help=SIZE_HELP),
                    "vol_x": st.column_config.NumberColumn("Vol ×", format="%.1f×", help="Breakout-day volume vs its 50-day median"),
                    "held": st.column_config.CheckboxColumn("Held", help="No close since below the breakout-day low"),
                    "ind_score": st.column_config.ProgressColumn("Industry", min_value=0, max_value=100, format="%d"),
                    "signal_price": st.column_config.NumberColumn("Price then", format="₹%.2f"),
                    "current_price": st.column_config.NumberColumn("Now", format="₹%.2f"),
                    "return_pct": st.column_config.NumberColumn("Return", format="%+.1f%%"),
                    "sessions_since": st.column_config.NumberColumn("Sessions"),
                    "flags_now": st.column_config.TextColumn("Flags", help=FLAG_HELP),
                }
                _bo_show = ['breakout_date', 'screener_link', 'name', 'kind', 'momentum', 'momentum_now', 'size_tier',
                            'vol_x', 'held', 'ind_score', 'signal_price', 'current_price', 'return_pct', 'sessions_since',
                            'flags_now']
                _wk, _wk_label = _bo_week(_bo_log), "this week"
                if _wk.empty:                               # Monday before the EOD run: show the week just ended
                    _wk, _wk_label = _bo_week(_bo_log, pd.Timestamp.today().normalize() - pd.Timedelta(days=7)), "last week (none yet this week)"
                with st.expander(f"📅 **Breakouts {_wk_label}** — {len(_wk)} (all sizes, momentum or not; "
                                 f"{int(_wk.momentum.sum()) if len(_wk) else 0} momentum entries)", expanded=True):
                    if _wk.empty:
                        st.caption("No 52-week / all-time-high breakout or squeeze release yet this week.")
                    else:
                        st.dataframe(_wk.sort_values(['momentum', 'breakout_date'], ascending=[False, False])[_bo_show],
                                     column_config=_bo_cfg, hide_index=True, use_container_width=True)
                    st.caption("Every fresh 52w / ATH closing-high breakout and squeeze release in the week, from the daily "
                               "log. 'Momentum then' = the study's entry rule on the breakout day; " + MOMO_HELP.split(". ")[1] + ".")
                with st.expander(f"📒 Breakout log — {len(_bo_log)} events since {pd.to_datetime(_bo_log.breakout_date).min():%d %b %Y}"):
                    _g = _bo_log[_bo_log.sessions_since >= 5].groupby('momentum').return_pct
                    if len(_g.size()):
                        st.caption("Average return since the breakout (events at least 5 sessions old): " + " · ".join(
                            f"{'momentum entries' if k else 'other breakouts'} {v.mean():+.1f}% (n={len(v)})" for k, v in _g))
                    st.dataframe(_bo_log[_bo_show], column_config=_bo_cfg, hide_index=True, use_container_width=True, height=400)
                    st.download_button("⬇️ Download breakout log", _bo_log.to_csv(index=False).encode(), "breakout_log.csv",
                                       "text/csv", key="dl_bo_log")

            # === TRADE PLAN: EDGE, STOP, SIZE ===
            # The tag backtest overturned the old "fresh first / Extended = wait" advice
            # (see _ENTRY_HELP). What this panel is for: ranking the filtered names by
            # the historical edge of their tag x industry x breakout cell, and turning
            # each into a concrete order -- a stop level and a share count such that
            # being stopped out costs a fixed % of capital (risk control, not a return
            # signal).
            with st.expander("📋 **Trade Plan** — historical edge, stop level and position size for the names above", expanded=False):
                st.caption("**Edge** = median 3-month return vs the typical stock for this entry tag, industry band and "
                           "breakout status, 2016-26 (years = how often it was positive). **Stop** = the wider of 2.5 daily "
                           "volatilities and just below the 50-day average. **Shares** are sized so a stop-out loses your "
                           "risk per trade, capped at an equal-weight slot -- a wide stop means a smaller position, not a skip.")
                _ps1, _ps2, _ps3, _ps4 = st.columns(4)
                with _ps1:
                    _cap = st.number_input("Capital (₹)", min_value=10000.0, value=1_000_000.0, step=50000.0, key="et_cap")
                with _ps2:
                    _riskpct = st.number_input("Risk / trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.25, key="et_risk")
                with _ps3:
                    _maxpos = st.number_input("Max positions", min_value=1, max_value=50, value=15, step=1, key="et_maxpos")
                with _ps4:
                    _show = st.selectbox("Show", ["Top N by edge (N = max positions)", "Breakouts only", "All except ⚪ Weak"],
                                         index=0, key="et_show",
                                         help="Top N = the best-edge names that fit your max positions, with the totals to deploy")

                _et = add_position_sizing(filtered_df, capital=_cap, risk_pct=_riskpct, max_positions=int(_maxpos))
                _et = _et[_et['entry_label'] != '⚪ Weak']
                if _show == "Breakouts only":
                    _et = _et[_et['bo_type'] != ""]
                _et = _et.sort_values('edge_3m', ascending=False, na_position='last').copy()
                if _show.startswith("Top N"):
                    _et = _et.head(int(_maxpos))
                _slot = _cap / int(_maxpos)
                _et['sized_by'] = np.where(_et['suggested_value'] >= _slot - 1, "slot cap", "stop risk")
                _et['stop_price'] = (pd.to_numeric(_et['price'], errors='coerce') * (1 - _et['stop_pct'] / 100)).round(2)
                _et['risk_rs'] = (_et['suggested_shares'] * pd.to_numeric(_et['price'], errors='coerce') * _et['stop_pct'] / 100).round(0)
                _et['screener_link'] = "https://www.screener.in/company/" + _et['ticker'].str.replace('.NS', '', regex=False) + "/"
                _etc = ['screener_link', 'entry_label', 'breakout_label', 'ind_score', 'edge_3m', 'edge_6m',
                        'edge_years', 'price', 'stop_price', 'stop_pct', 'suggested_shares', 'suggested_value', 'risk_rs',
                        'sized_by']
                st.dataframe(
                    _et[_etc],
                    column_config={
                        "screener_link": st.column_config.LinkColumn("Ticker", display_text=r"https://www\.screener\.in/company/(.*?)/"),
                        "entry_label": st.column_config.TextColumn("Entry", help=_ENTRY_HELP),
                        "breakout_label": st.column_config.TextColumn("Breakout", help=BREAKOUT_HELP),
                        "ind_score": st.column_config.ProgressColumn("Industry", min_value=0, max_value=100, format="%d"),
                        "edge_3m": st.column_config.NumberColumn("Edge 3m", format="%+.1f pp", help="Median 3-month return vs the typical stock for this tag x industry x breakout cell, 2016-26"),
                        "edge_6m": st.column_config.NumberColumn("Edge 6m", format="%+.1f pp"),
                        "edge_years": st.column_config.TextColumn("Years +", help="Calendar years in which the 3-month edge was positive"),
                        "price": st.column_config.NumberColumn("Price", format="₹ %.2f"),
                        "stop_price": st.column_config.NumberColumn("Stop at", format="₹ %.2f", help="Price - Stop %"),
                        "stop_pct": st.column_config.NumberColumn("Stop %", format="%.1f%%", help="Wider of 2.5 daily volatilities and just below the 50-day average (6-30%)"),
                        "suggested_shares": st.column_config.NumberColumn("Shares", format="%d"),
                        "suggested_value": st.column_config.NumberColumn("Buy ₹", format="₹ %.0f"),
                        "risk_rs": st.column_config.NumberColumn("Loss if stopped", format="₹ %.0f"),
                        "sized_by": st.column_config.TextColumn("Sized by", help="slot cap = capital / max positions was the "
                                                                "smaller limit, so Risk / trade does not change this row; "
                                                                "stop risk = sized so a stop-out loses your risk per trade"),
                    },
                    height=360, use_container_width=True, hide_index=True
                )
                # Each row is the SMALLER of (risk / stop %) and the equal-weight slot. With typical stops of
                # 6-10%, 1% risk sizes above a 1/15 slot, so most rows sit at the cap and the risk input is
                # inert until it drops below ~stop% / max positions. Say so instead of looking broken.
                if len(_et):
                    _ncap = int((_et['sized_by'] == "slot cap").sum())
                    _med_stop = float(_et['stop_pct'].median())
                    _bind = _med_stop / int(_maxpos)
                    st.caption(f"Each row buys the **smaller** of ₹{_cap*_riskpct/100:,.0f} ÷ stop % (your risk per trade) "
                               f"and one slot of **₹{_slot:,.0f}** (capital ÷ max positions). **{_ncap} of {len(_et)}** rows "
                               f"hit the slot cap, so Risk / trade does not change them -- with a median stop of "
                               f"{_med_stop:.1f}%, risk only binds below ~{_bind:.2f}%. Capital and Max positions always "
                               "change the size; the edge, stop and order never do (they come from the stock, not your "
                               "inputs).")
                    if _show.startswith("Top N"):
                        st.caption(f"Top {len(_et)} together: buy **₹{_et['suggested_value'].sum():,.0f}** "
                                   f"({_et['suggested_value'].sum()/_cap:.0%} of capital), worst case if every stop hits "
                                   f"**₹{_et['risk_rs'].sum():,.0f}** ({_et['risk_rs'].sum()/_cap:.1%}).")
                st.caption("The edge is a historical median, not a forecast; in the live-portfolio backtest, preferring "
                           "breakouts did not beat the engine's 6-month RS ranking, so use this to choose among names "
                           "you already like.")

            display_cols = ['screener_link', 'name', 'sub_industry', 'ind_status', 'price', 'signal_display', 'trend_score', 'comp_rs',
                            'momo_entry', 'size_tier', 'watch_flags',
                            'entry_label', 'edge_3m', 'breakout_label', 'ind_score',
                            'dist_ma50', 'volatility', 'dna_signal', 'dist_52w', 'dist_ath', 'dist_200dma']
            for _nc_col in ('nse_flags', 'deliv_avg20', 'deliv_pct'):        # NSE daily data, when present (inserted in reverse)
                if _nc_col in filtered_df.columns:
                    display_cols.insert(display_cols.index('watch_flags') + 1, _nc_col)
            # Add 5-pillar fundamental columns + RS Score for user request
            display_cols.extend(['quality', 'value', 'growth', 'momentum', 'volume_signal_score'])

            filtered_df['screener_link'] = "https://www.screener.in/company/" + filtered_df['ticker'].str.replace('.NS', '', regex=False) + "/"

            st.dataframe(
                filtered_df[display_cols].sort_values(by=['trend_score', 'comp_rs'], ascending=False, na_position='last'),   # ties (many score 100) by RS
                column_config={
                    "screener_link": st.column_config.LinkColumn("Ticker", display_text=r"https://www\.screener\.in/company/(.*?)/"),
                    "trend_score": st.column_config.ProgressColumn("Trend Score", format="%d", min_value=0, max_value=100),
                    "price": st.column_config.NumberColumn("Price", format="₹ %.2f"),
                    "dist_52w": st.column_config.NumberColumn("% from 52W High", format="%.1f%%"),
                    "dist_200dma": st.column_config.NumberColumn("% vs 200DMA", format="%.1f%%"),
                    "signal_display": st.column_config.TextColumn("Signal"),
                    "sub_industry": st.column_config.TextColumn("Sub-industry"),
                    "ind_status": st.column_config.TextColumn("Group", help="The sub-industry's place on the Industry board"),
                    "quality": st.column_config.ProgressColumn("Quality", min_value=0, max_value=10, format="%.1f"),
                    "value": st.column_config.ProgressColumn("Value", min_value=0, max_value=10, format="%.1f"),
                    "growth": st.column_config.ProgressColumn("Growth", min_value=0, max_value=10, format="%.1f"),
                    "momentum": st.column_config.ProgressColumn("Momentum", min_value=0, max_value=10, format="%.1f"),
                    "volume_signal_score": st.column_config.ProgressColumn("Volume", min_value=0, max_value=10, format="%.1f"),
                    "comp_rs": st.column_config.NumberColumn("RS vs Nifty", format="%+.1f%%", help="Composite Relative Strength vs Nifty (1W+1M+3M)"),
                    "entry_label": st.column_config.TextColumn("Entry", help=_ENTRY_HELP),
                    "edge_3m": st.column_config.NumberColumn("Edge 3m", format="%+.1f pp", help="Median 3-month return vs the typical stock for this entry tag x industry band x breakout status, 2016-26 backtest"),
                    "breakout_label": st.column_config.TextColumn("Breakout", help=BREAKOUT_HELP),
                    "momo_entry": st.column_config.TextColumn("Momentum entry", help=MOMO_HELP),
                    "size_tier": st.column_config.TextColumn("Size", help=SIZE_HELP),
                    "watch_flags": st.column_config.TextColumn("Flags", help=FLAG_HELP),
                    "dist_ath": st.column_config.NumberColumn("% from ATH", format="%.1f%%", help="Distance from the all-time closing high (data/ath_levels.csv)"),
                    "ind_score": st.column_config.ProgressColumn("Industry", min_value=0, max_value=100, format="%d", help="Sub-industry rotation score (0-100): >= 70 leader, < 40 laggard"),
                    "dist_ma50": st.column_config.NumberColumn("% vs MA50", format="%+.1f%%", help="Distance above the 50-day MA. High = extended / already run up."),
                    "volatility": st.column_config.NumberColumn("Volatility", format="%.0f%%", help="Annualized Price Volatility"),
                    "dna_signal": st.column_config.TextColumn("DNA Signal", help="BUY = All DNA-3 filters pass"),
                    "deliv_pct": st.column_config.NumberColumn("Delivery %", format="%.0f%%", help="Share of the day's traded shares actually delivered (NSE). High = real buying or selling, low = intraday churn"),
                    "deliv_avg20": st.column_config.NumberColumn("Deliv 20d", format="%.0f%%", help="Average delivery % over the last 20 sessions"),
                    "nse_flags": st.column_config.TextColumn("NSE", help="Exchange surveillance (ASM / GSM) and the next results date or split / bonus / dividend ex-date"),
                },
                height=500,
                use_container_width=True,
                hide_index=True
            )
            with _nifty_dl:
                st.markdown("<div style='padding-top:10px'></div>", unsafe_allow_html=True)
                _dl_cols = ['ticker'] + [c for c in display_cols if c not in ('screener_link', 'ticker')]
                _export = filtered_df[_dl_cols].sort_values(['trend_score', 'comp_rs'] if 'comp_rs' in _dl_cols else 'trend_score', ascending=False).copy()
                _export['signal'] = filtered_df.loc[_export.index, 'trend_signal'] if 'trend_signal' in filtered_df.columns else _export.get('signal_display', '')
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
                        # one dict lookup per option; filtering the frame per option cost ~5 s a rerun
                        format_func=lambda x, _nm=dict(zip(filtered_df['ticker'], filtered_df['name'])): (
                            f"{x} - {_nm[x]}" if x in _nm else x)
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
                        _pos_existing = get_position(selected_for_dive)
                        _pos_status = _pos_existing.get('status') if _pos_existing else None
                        if _pos_status == 'active':
                            st.button("💼 Holding", disabled=True, use_container_width=True,
                                      help="Already an active holding in My Book — use Untrack on the right to remove.")
                        elif _pos_status == 'watching':
                            if st.button("⭐ Unwatch", use_container_width=True):
                                remove_position(selected_for_dive)
                                st.toast(f"Removed {selected_for_dive} from Watchlist!", icon="⭐")
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
