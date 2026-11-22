"""Return Tracker / Position Manager page view (extracted from main.py)."""
from __future__ import annotations

from datetime import datetime
import os

import pandas as pd
import streamlit as st

from utils import industry_alerts as ind_alerts
from utils.email_notifier import configure_email, get_email_address, is_email_configured, send_weekly_summary
from utils.positions import (
    add_position,
    check_position_alerts,
    get_positions_with_pnl,
    get_summary,
    is_position_exists,
    migrate_from_legacy,
    remove_position,
)
from utils.return_tracker import export_weekly_summary
from utils.rs_alerts import (
    DIRECTIONS,
    RS_METRICS,
    add_rs_alert,
    describe_arm_state,
    get_rs_alerts_with_status,
    normalize_ticker,
    read_metric,
    rearm_rs_alert,
    remove_rs_alert,
    storage_status as rs_storage_status,
    sync_rs_alerts,
)
from utils.ui_components import COLORS, page_header


def render(df: pd.DataFrame, *, alert_sync_banner, rerun_fragment, rs_fired) -> None:

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
    # A position whose symbol is not in the market data gets no price, so its stop-loss and
    # target are never checked -- say so instead of showing it as quietly fine.
    # only what you still hold or watch: a sold stock outside the universe has no price, and needs none
    _live_pos = [p for p in positions if p.get('status') in ('active', 'watching')]
    _orphans = [p.get('ticker', '') for p in _live_pos if not p.get('has_data')]
    if _orphans:
        st.error("⚠️ No market data for " + ", ".join(f"**{t.replace('.NS', '')}**" for t in _orphans)
                 + ". These positions have no live price, so their stop-loss and target alerts cannot fire. "
                 "Delete and re-add them with the correct NSE symbol.")
    _aliased = [f"{p['ticker'].replace('.NS', '')} → {p['resolved_ticker'].replace('.NS', '')}"
                for p in _live_pos if p.get('resolved_ticker')]
    if _aliased:
        st.info("Symbol corrected for pricing: " + ", ".join(_aliased) + ". Stop-loss and target are now checked.")
    for _p in _live_pos:
        if _p.get('adj_note'):
            st.info(f"**{_p['ticker'].replace('.NS', '')}**: {_p['adj_note']}.")
    from utils.positions import storage_status as _pos_status, sync_positions as _pos_sync
    alert_sync_banner(_pos_status(), _pos_sync, "pos")
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
    # key + on_change: the selected tab is kept in session state, so saving or
    # deleting an alert (which reruns the script) no longer drops you back on
    # Active Positions.
    tab_book, tab_import, tab_journal, tab_orders, tab_active, tab_watchlist, tab_add, tab_settings = st.tabs(
        ["📒 My Book", "📥 Import", "📓 Journal", "🎫 Model Orders", "📋 Positions (classic)", "🔔 Alerts", "➕ Add (classic)",
         "⚙️ Settings"], key="desk_tab", on_change="rerun")

    # Real-money copy of the model portfolio: order ticket, confirmation, fills (utils/model_orders_view.py)
    with tab_orders:
        if getattr(tab_orders, "open", True):
            try:
                from utils import model_orders_view as _mo
                _mo.render(df)
            except Exception as _e:
                st.error(f"Model Orders failed to load: {_e}")

    # Trade journal: every buy and sell with that day's readings, and what has worked (utils/journal_view.py)
    with tab_journal:
        if getattr(tab_journal, "open", True):
            try:
                from utils import journal_view as _jv
                _jv.render(df)
            except Exception as _e:
                st.error(f"Journal failed to load: {_e}")

    # Broker file import (utils/portfolio_import.py): its own tab so it is easy to find on a phone
    # Only the open tab is drawn (stateful tabs rerun on a switch): the page used to compute all six on every click.
    with tab_import:
        if getattr(tab_import, "open", True):
            try:
                from utils import book_view as _bv_import
                _bv_import.render_import(df)
            except Exception as _e:
                st.error(f"Import failed to load: {_e}")

    # TAB 0: holdings + watchlist as editable tables, one Save (utils/book_view.py)
    with tab_book:
        if getattr(tab_book, "open", True):
            try:
                from utils import book_view
                book_view.render(df)
            except Exception as _e:
                st.error(f"My Book failed to load: {_e}")

    # ========================================
    # TAB 1: ACTIVE POSITIONS (Using unified positions.json)
    # ========================================
    with tab_active:
        if getattr(tab_active, "open", True):

            if active_positions:
                # Enhanced positions table with SL/Target
                st.markdown("### 📈 Your Positions")

                # Build portfolio data from positions.json
                from utils import stock_picker as _sp_pos
                _pos_names = _sp_pos.names(df)
                _q_pos = _sp_pos.name_filter("🔎 Filter positions", "pos_filter") if len(active_positions) > 5 else ""
                portfolio_data = []
                for pos in [p for p in active_positions
                            if _sp_pos.matches(_q_pos, p.get('ticker', ''), p.get('name') or _pos_names.get(p.get('ticker', ''), ''))]:
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
        # One fragment for the whole tab: saving, deleting or re-arming an alert
        # reruns only this tab, not the 8,000-line page, and GitHub writes happen
        # in the background (utils/alert_store.py), so entries take no wait.
        @st.fragment
        def _watchlist_tab():
    
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
                from utils.alerts import get_alerts_with_pnl, remove_alert, add_price_alert, update_alert_ticker
                from utils import stock_picker as _sp
                all_alerts = get_alerts_with_pnl(df)
                _px_names = _sp.names(df)
                _q_alert = _sp.name_filter("🔎 Filter alerts", "px_alert_filter") if len(all_alerts) > 3 else ""
                _priced = set(df['ticker'])
                _shown = [a for a in all_alerts if _sp.matches(_q_alert, a.get('ticker', ''), _px_names.get(a.get('ticker', ''), ''))]
                _unpriced = [a for a in all_alerts if a.get('ticker') not in _priced]
                if _unpriced:
                    st.warning(f"⚠️ {len(_unpriced)} alert(s) have no price, so their stop and target are not checked: "
                               + ", ".join(str(a.get('ticker', '')).replace('.NS', '') for a in _unpriced)
                               + ". See the note under each one.")
                if all_alerts and not _shown:
                    st.caption("No alert matches the filter.")
        
                if all_alerts:
                    for alert in _shown:
                        ticker = alert.get('ticker', '').replace('.NS', '')
                        _cname = _px_names.get(alert.get('ticker', ''), '')
                
                        # Build status line
                        parts = [f"**{ticker}**" + (f" :grey[{_cname[:28]}]" if _cname and _cname.upper() != ticker else "")]
                
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
                
                        if alert.get('current_price') is not None and pnl is None:
                            parts.append(f"now ₹{float(alert['current_price']):,.0f}")
                        col_info, col_del = st.columns([5, 1])
                        with col_info:
                            st.write(" | ".join(parts))
                            if alert.get('notes'):
                                st.caption(f"📝 {alert['notes'][:80]}...")
                            if 'nse_flags' in df.columns:
                                _fl = df.loc[df['ticker'] == alert.get('ticker'), 'nse_flags']
                                if len(_fl) and str(_fl.iloc[0]):
                                    st.caption(str(_fl.iloc[0]))
                            if alert.get('ticker') not in _priced:
                                _t_raw = str(alert.get('ticker', ''))
                                _sug = _sp.suggest(_t_raw, df)
                                if _t_raw in _px_names or _t_raw.endswith('.BO'):
                                    st.caption("⏳ Outside the ranked universe: it gets a price from the next evening run.")
                                elif _sug and _sug[0] != _t_raw:
                                    st.caption(f"⚠️ No stock is listed as **{ticker}**. Did you mean **{_sug[0].replace('.NS', '')}** "
                                               f"({_sug[1]})?")
                                    if st.button(f"Use {_sug[0].replace('.NS', '')}", key=f"fix_alert_{alert.get('id')}"):
                                        update_alert_ticker(alert.get('id'), _sug[0])
                                        st.rerun()
                                else:
                                    st.caption(f"⚠️ No NSE stock is listed as **{ticker}**. If it trades only on BSE, delete this "
                                               "alert and add it again with its 6-digit BSE code.")
                        with col_del:
                            if st.button("🗑️", key=f"del_alert_{alert.get('id')}", help="Delete this alert"):
                                remove_alert(alert.get('id'))
                                st.session_state['alert_deleted'] = alert.get('ticker', '')
                                rerun_fragment()
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
            
                    from utils.stock_picker import pick as _pick_stock
                    new_ticker = _pick_stock("Stock", "new_alert_stock", df, default=prefill or None)
            
                    col_a, col_b = st.columns(2)
                    with col_a:
                        new_entry = st.number_input("Entry ₹", value=0.0, step=10.0, format="%.2f")
                        new_sl = st.number_input("Stop Loss ₹", value=0.0, step=10.0, format="%.2f")
                    with col_b:
                        new_target = st.number_input("Target ₹", value=0.0, step=10.0, format="%.2f")
                
                    new_notes = st.text_area("Notes", placeholder="Investment thesis...", height=80)
            
                    submitted = st.form_submit_button("✅ Create Alert", use_container_width=True, type="primary")
            
                    if submitted and not new_ticker:
                        st.error("Pick a stock from the list (or type a valid 6-digit BSE code).")
                    if submitted and new_ticker:
                        ticker = new_ticker.upper()
                        if not ticker.endswith((".NS", ".BO")):
                            ticker += ".NS"
                        add_price_alert(
                            ticker,
                            entry_price=new_entry if new_entry > 0 else None,
                            stop_loss=new_sl if new_sl > 0 else None,
                            target=new_target if new_target > 0 else None,
                            notes=new_notes
                        )
                        st.toast(f"Alert created for {ticker}!", icon="✅")
                        rerun_fragment()

            # ============================================================
            # COMPOSITE-RS CROSSING ALERTS
            # ============================================================
            st.markdown("---")
            st.markdown("### 📈 RS Alerts")
            st.caption("Fire when a stock's Composite RS **crosses** a level, up or "
                       "down. Crossing — not level — so an alert set above a stock "
                       "that is already there waits for a real move instead of "
                       "firing on every refresh.")

            alert_sync_banner(rs_storage_status(), sync_rs_alerts, "rs")

            if rs_fired:
                st.error(f"⚡ {len(rs_fired)} RS crossing(s) since the last EOD check "
                         f"(Telegram goes out from the EOD run)")
                for _a in rs_fired:
                    st.warning(_a['alert_message'])

            rs_col1, rs_col2 = st.columns([2, 1])

            with rs_col2:
                st.markdown("#### ➕ New RS Alert")
                with st.form("new_rs_alert_form", clear_on_submit=True):
                    from utils.stock_picker import pick as _pick_stock_rs
                    rs_ticker = _pick_stock_rs("Stock", "rs_alert_ticker", df, universe_only=True,
                                               help="RS is computed for the loaded universe only")
                    rs_metric = st.selectbox(
                        "Metric", options=list(RS_METRICS.keys()),
                        format_func=lambda k: RS_METRICS[k][0], index=0,
                        key="rs_alert_metric")
                    rs_direction = st.selectbox(
                        "Trigger when it", options=list(DIRECTIONS.keys()),
                        format_func=lambda k: DIRECTIONS[k], index=0,
                        key="rs_alert_dir")
                    rs_threshold = st.number_input(
                        "RS level", value=17.0, step=1.0, format="%.1f",
                        help="CompRS is in percentage points vs Nifty. The engine's "
                             "entry floor is 17 in BULL/CAUTION and 22 in BEAR/CRISIS.",
                        key="rs_alert_thr")
                    rs_repeat = st.checkbox(
                        "Re-fire on every future crossing", value=True,
                        help="Off = one-shot; it goes quiet after firing until you re-arm it.",
                        key="rs_alert_repeat")
                    rs_notes = st.text_area("Notes", placeholder="Why this level...",
                                            height=70, key="rs_alert_notes")

                    rs_submitted = st.form_submit_button(
                        "🔔 Create RS Alert", use_container_width=True, type="primary")

                    if rs_submitted and rs_ticker:
                        _t = normalize_ticker(rs_ticker)
                        if df[df['ticker'] == _t].empty:
                            st.error(f"{_t} is not in the loaded universe — "
                                     f"check the symbol.")
                        else:
                            _new = add_rs_alert(_t, rs_threshold, direction=rs_direction,
                                                metric=rs_metric, notes=rs_notes,
                                                repeat=rs_repeat, df=df, background=True)
                            _now = read_metric(df, _t, rs_metric)
                            _state = describe_arm_state(_now, rs_threshold, rs_direction)
                            if _new.get("duplicate"):
                                st.info(f"That alert on {_t.replace('.NS','')} already exists.")
                            else:
                                st.success(f"Alert set on {_t.replace('.NS','')} — {_state}. "
                                           f"Syncing to GitHub in the background.")

            with rs_col1:
                st.markdown("#### Active RS Alerts")
                _rs_all = get_rs_alerts_with_status(df)
                if not _rs_all:
                    st.info("No RS alerts yet. Create one →")
                else:
                    from utils import stock_picker as _sp_rs
                    _rs_names = _sp_rs.names(df)
                    _q_rs = _sp_rs.name_filter("🔎 Filter RS alerts", "rs_alert_filter") if len(_rs_all) > 3 else ""
                    for _a in [x for x in _rs_all if _sp_rs.matches(_q_rs, x['ticker'], _rs_names.get(x['ticker'], ''))]:
                        _name = _a['ticker'].replace('.NS', '').replace('.BO', '')
                        _label = RS_METRICS.get(_a['metric'], (_a['metric'], None))[0]
                        _cur = _a.get('current_value')
                        _dist = _a.get('distance')

                        _c_info, _c_act = st.columns([6, 1])
                        with _c_info:
                            _line = (f"**{_name}** · {_label} "
                                     f"{DIRECTIONS[_a['direction']].lower()} "
                                     f"**{_a['threshold']:+.1f}**")
                            if _cur is None:
                                _line += " · now :grey[no clean reading]"
                            else:
                                _col = "green" if (_dist or 0) >= 0 else "red"
                                _line += (f" · now :{_col}[{_cur:+.1f}] "
                                          f"({_dist:+.1f} away)")
                            if not _a.get('repeat', True):
                                _line += " · :grey[one-shot]"
                            st.write(_line)

                            _sub = _a.get('state', '')
                            if _a.get('trigger_count', 0):
                                _when = (_a.get('last_triggered') or '')[:16].replace('T', ' ')
                                _sub = (f"fired {_a['trigger_count']}× · last "
                                        f"{_a.get('last_trigger_direction','')} on {_when}")
                            st.caption(_sub + (f" · 📝 {_a['notes'][:60]}"
                                               if _a.get('notes') else ""))
                        with _c_act:
                            if st.button("🗑️", key=f"del_rs_{_a['id']}",
                                         help="Delete this RS alert"):
                                remove_rs_alert(_a['id'], _name, background=True)
                                rerun_fragment()
                            if (not _a.get('repeat', True)) and _a.get('trigger_count', 0):
                                if st.button("↺", key=f"rearm_rs_{_a['id']}",
                                             help="Re-arm this one-shot alert"):
                                    rearm_rs_alert(_a['id'], background=True)
                                    rerun_fragment()
                        st.markdown("<hr style='margin: 5px 0; opacity: 0.2;'>",
                                    unsafe_allow_html=True)

            # ============================================================
            # INDUSTRY / BASKET STATUS ALERTS
            # ============================================================
            st.markdown("---")
            st.markdown("### 🏭 Industry Alerts")
            st.caption(
                "Telegram after the close when an industry (or your own basket, e.g. Hotels) changes "
                "status, or its trend score crosses your level. Status from member breadth: 🔴 DOWNTREND → "
                "🟠 BASING → 🟡 TURNING UP → 🟢 UPTREND (🟣 WEAKENING = long trend intact, short-term lost). "
                "Trend score 0-100 = 40% share above 50-day + 40% share above 200-day + 20% median RS; "
                "< 30 downtrend, 30-50 basing, 50-70 improving, > 70 strong. Readings are 5-day averages and a "
                "new status must hold 3 days. Tested 2021-26: an early TURNING UP was not a buy on its own; "
                "UPTREND was the better-evidenced step — treat these as a watch tool.")
            alert_sync_banner(ind_alerts.storage_status(), ind_alerts.sync_industry_alerts, "ind")

            _ind_c1, _ind_c2 = st.columns([2, 1])
            with _ind_c2:
                st.markdown("#### ➕ New Industry Alert")
                _ind_kind = st.radio("Watch", ["Preset basket", "Sub-industry", "Custom tickers"],
                                     horizontal=True, key="ind_kind")
                with st.form("new_ind_alert_form", clear_on_submit=False):
                    _ind_sub, _ind_tk, _ind_name = None, None, ""
                    if _ind_kind == "Preset basket":
                        _ind_name = st.selectbox("Basket", list(ind_alerts.PRESET_BASKETS), key="ind_preset")
                        _ind_tk = ind_alerts.PRESET_BASKETS[_ind_name]
                        st.caption(", ".join(_ind_tk))
                    elif _ind_kind == "Sub-industry":
                        from utils.nifty1000_list import SUB_INDUSTRY_MAP as _SIM
                        _ind_sub = st.selectbox("Sub-industry", sorted({g for g in _SIM.values()
                                                                         if g and g != "Unknown"}),
                                                key="ind_sub")
                        _ind_name = _ind_sub
                    else:
                        _ind_name = st.text_input("Basket name", placeholder="Hotels", key="ind_name")
                        _ind_raw = st.text_area("Tickers (comma or space separated)",
                                                placeholder="INDHOTEL, CHALET, LEMONTREE", key="ind_tickers")
                        _ind_tk = [t for t in _ind_raw.replace(",", " ").split() if t]
                    _ind_status = st.checkbox("Alert on status changes", value=True, key="ind_on_status")
                    _ind_notify = st.selectbox("Which changes", list(ind_alerts.NOTIFY),
                                               format_func=lambda k: ind_alerts.NOTIFY[k], key="ind_notify")
                    _ind_use_score = st.checkbox("Also alert on a trend-score level", value=True,
                                                 key="ind_use_score")
                    _ind_dir = st.selectbox("Trigger when the score", list(ind_alerts.DIRECTIONS),
                                            format_func=lambda k: ind_alerts.DIRECTIONS[k], key="ind_dir")
                    _ind_thr = st.slider("Trend score level", 0, 100, 50, 5, key="ind_thr",
                                         help="Fires when the 5-day score moves from 2+ points one side "
                                              "of the level to 2+ points the other side.")
                    _ind_notes = st.text_input("Notes", placeholder="Why I'm watching...", key="ind_notes")
                    if st.form_submit_button("🔔 Create Industry Alert", use_container_width=True,
                                             type="primary"):
                        _mem = ([ind_alerts.normalize_ticker(t) for t in _ind_tk] if _ind_tk
                                else ind_alerts.basket_members({"sub_industry": _ind_sub}))
                        _rd = ind_alerts.basket_reading(df, _mem)
                        if not _ind_name:
                            st.error("Give the basket a name.")
                        elif _rd is None:
                            st.error(f"Fewer than {ind_alerts.MIN_VALID} of these stocks have data "
                                     f"in the loaded universe — check the symbols.")
                        else:
                            _new = ind_alerts.add_industry_alert(
                                _ind_name, sub_industry=_ind_sub, tickers=_ind_tk,
                                on_status=_ind_status, notify=_ind_notify,
                                score_threshold=float(_ind_thr) if _ind_use_score else None,
                                direction=_ind_dir, notes=_ind_notes, df=df, background=True)
                            _msg = (f"Watching {_ind_name}: now {ind_alerts.ICONS[_rd['status']]} "
                                    f"{_rd['status']}, score {_rd['score']:.0f}")
                            if _new.get("duplicate"):
                                st.info(f"You are already watching {_ind_name} with these settings.")
                            else:
                                st.success(_msg + ". Syncing to GitHub in the background.")

            with _ind_c1:
                st.markdown("#### Active Industry Alerts")
                _ind_all = ind_alerts.load_industry_alerts()
                if not _ind_all:
                    st.info("No industry alerts yet. Create one →")
                for _a in _ind_all:
                    _rd = ind_alerts.basket_reading(df, ind_alerts.basket_members(_a))
                    _stat = _a.get("last_status") or (_rd or {}).get("status")
                    _c_info, _c_act = st.columns([6, 1])
                    with _c_info:
                        _line = f"**{_a['name']}** · {ind_alerts.ICONS.get(_stat, '')} {_stat}"
                        if _a.get("pending_status"):
                            _line += (f" · :orange[→ {_a['pending_status']} pending "
                                      f"{_a.get('pending_days', 0)}/{ind_alerts.CONFIRM_DAYS}d]")
                        if _rd:
                            _line += (f" · score today **{_rd['score']:.0f}** "
                                      f"({_rd['above_50']:.0f}% > 50d, {_rd['above_200']:.0f}% > 200d)")
                        st.write(_line)
                        _bits = []
                        if _a.get("on_status", True):
                            _bits.append("status: " + ("upgrades" if _a.get("notify") == "up" else "all changes"))
                        if _a.get("score_threshold") is not None:
                            _bits.append(f"score {ind_alerts.DIRECTIONS[_a.get('direction', 'above')].lower()} "
                                         f"{_a['score_threshold']:.0f}")
                        _hist = _a.get("history") or []
                        if _hist:
                            _h = _hist[-1]
                            _bits.append(f"last change {_h['from']} → {_h['to']} on {_h['date']}")
                        if _a.get("notes"):
                            _bits.append(f"📝 {_a['notes'][:60]}")
                        st.caption(" · ".join(_bits))
                    with _c_act:
                        if st.button("🗑️", key=f"del_ind_{_a['id']}", help="Delete this industry alert"):
                            ind_alerts.remove_industry_alert(_a["id"], _a.get("name", ""), background=True)
                            rerun_fragment()
                    st.markdown("<hr style='margin: 5px 0; opacity: 0.2;'>", unsafe_allow_html=True)

                with st.expander("All sub-industries — status and trend score today"):
                    _ov = ind_alerts.all_sub_industry_readings(df)
                    if _ov.empty:
                        st.caption("No data.")
                    else:
                        _ov["status"] = _ov["status"].map(lambda x: f"{ind_alerts.ICONS.get(x, '')} {x}")
                        st.dataframe(
                            _ov.sort_values("score", ascending=False)[
                                ["industry", "status", "score", "above_50", "above_200", "med_rs", "n_valid"]],
                            hide_index=True, use_container_width=True,
                            column_config={"score": st.column_config.ProgressColumn(
                                               "Trend score", min_value=0, max_value=100, format="%.0f"),
                                           "above_50": st.column_config.NumberColumn("% > 50d", format="%.0f"),
                                           "above_200": st.column_config.NumberColumn("% > 200d", format="%.0f"),
                                           "med_rs": st.column_config.NumberColumn("Median RS", format="%+.1f"),
                                           "n_valid": "Stocks"})
                        st.caption("Today's single-day reading; alerts use the 5-day average.")

        if getattr(tab_watchlist, "open", True):
            _watchlist_tab()

    # ========================================
    # TAB 3: ADD NEW POSITION
    # ========================================
    with tab_add:
        if getattr(tab_add, "open", True):

            st.markdown("### ➕ Add New Position")
            st.caption("Add a stock to your portfolio or watchlist")

            # Use unified form
            with st.form("add_position_form", clear_on_submit=True):
                add_col1, add_col2 = st.columns(2)

                # every NSE equity plus the universe, searchable by company name (was the Nifty 500 list only)
                from utils.stock_picker import pick as _pick_stock_add, names as _stock_names
                with add_col1:
                    new_ticker = _pick_stock_add("Stock (type a name or symbol)", "add_pos_stock", df)
                    new_entry = st.number_input("Entry Price ₹", value=0.0, step=10.0, format="%.2f")
                    new_quantity = st.number_input("Quantity", value=1, min_value=1, step=1)

                with add_col2:
                    new_status = st.selectbox("Status", ["active", "watching"], format_func=lambda x: "📈 Active Position" if x == "active" else "👀 Watchlist")
                    new_sl = st.number_input("Stop Loss ₹", value=0.0, step=10.0, format="%.2f")
                    new_target = st.number_input("Target ₹", value=0.0, step=10.0, format="%.2f")

                new_notes = st.text_area("Notes / Investment Thesis", placeholder="Why are you buying this? Key catalysts, risks...", height=100)

                submitted = st.form_submit_button("✅ Add Position", use_container_width=True, type="primary")

                if submitted and not new_ticker:
                    st.error("Pick a stock from the list (or type a valid 6-digit BSE code).")
                if submitted and new_ticker:
                    ticker = new_ticker.upper()
                    if not ticker.endswith((".NS", ".BO")):
                        ticker += ".NS"

                    # Get stock info from market data
                    stock_row = df[df['ticker'] == ticker]
                    stock_name = (stock_row.iloc[0].get('name', ticker) if not stock_row.empty
                                  else _stock_names(df).get(ticker) or ticker.replace('.NS', ''))
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
        if getattr(tab_settings, "open", True):
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
