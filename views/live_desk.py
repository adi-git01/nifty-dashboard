"""Live Trading Desk page view (extracted from main.py)."""
from __future__ import annotations

from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import yfinance as yf

from utils.advanced_scanners import (
    RS_LOG_COLS,
    RS_LOG_FILE,
    SHOCK_LOG_COLS,
    SHOCK_LOG_FILE,
    TC_LOG_COLS,
    TC_LOG_FILE,
    US_TC_LOG_FILE,
    find_live_earnings_shocks,
    find_rs_divergence,
    find_turnaround_catalysts,
    find_vcp_setups,
    refresh_signal_log_prices,
    save_turnaround_catalyst_signals,
)
from utils.ui_components import page_header
from utils.us_data_engine import fetch_us_market_data


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
    return fetch_us_market_data(benchmark="SPY", rs_weights=[(252, 1.0)], live_mode=False)

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


def render(df: pd.DataFrame, *, load_nifty, us_quote_url, us_display_re, auto_refresh_minutes: int) -> None:
    st.markdown(page_header("🚀 V2.2 Momentum Engine: The Live Trading Desk", "Pure mathematical momentum deployment guided by 15-Year out-of-sample Regime & Seasonality analytics."), unsafe_allow_html=True)

    from utils.live_desk import generate_v3_watchlist

    # 1. Use centralized Nifty data (pre-fetched at startup)
    with st.spinner("Calculating Live Macro Regime..."):
        if 'nifty_data' in st.session_state:
            nifty_live = st.session_state['nifty_data']
        else:
            start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
            try:
                nifty_live = load_nifty(start_date)
            except Exception:
                nifty_live = pd.DataFrame()

        # The divergence scan pairs today's stock change against the index's
        # last-vs-previous bar, so both sides must describe the same session.
        # Calendar age is the wrong measure — a Friday bar read on Monday is
        # 3 days old and perfectly valid, while a Tuesday bar read on Thursday
        # is only 2 days old and means Wednesday's session is missing. Count
        # weekday sessions that elapsed with no bar instead.
        _nifty_bar_date, _nifty_bar_age, _sessions_missed = None, None, 0
        try:
            _lb = nifty_live.index[-1]
            _lb = _lb.tz_localize(None) if getattr(_lb, "tzinfo", None) else _lb
            _nifty_bar_date = _lb.to_pydatetime().date()
            _today = datetime.now().date()
            _nifty_bar_age = (_today - _nifty_bar_date).days
            _d = _nifty_bar_date + timedelta(days=1)
            while _d < _today:
                if _d.weekday() < 5:
                    _sessions_missed += 1
                _d += timedelta(days=1)
        except Exception:
            _nifty_bar_date = None
        # A market holiday can produce one missed weekday, so the message says
        # so — but the scan still stops. A day without this niche scanner costs
        # nothing; a scan pairing mismatched sessions produces wrong signals.
        _nifty_stale = _sessions_missed >= 1

        if nifty_live is not None and not nifty_live.empty:
            # Market regime: the same read as the sidebar and Morning Pulse (utils/regime_badge.py). The old
            # STRONG BULL / MILD BULL / ... card and its "optimal cash" / "max positions" came from a separate,
            # untested classifier that the engine never used -- removed.
            st.markdown("### 🧭 MARKET REGIME")
            _regime_asof = nifty_live.index[-1]
            _regime_asof_str = _regime_asof.strftime('%Y-%m-%d') if hasattr(_regime_asof, 'strftime') else str(_regime_asof)
            st.caption(f"Nifty data as of **{_regime_asof_str}** — refreshes every {auto_refresh_minutes} min")
            from utils import regime_badge as _rb
            _rb.render(_rb.compute(nifty_live, df), st)

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
        
            _scan_tab_in, _scan_tab_us = st.tabs(["🇮🇳 Nifty Universe", "🇺🇸 S&P 500"], key="scan_tab", on_change="rerun")

            with _scan_tab_in:
                with st.spinner("Running Advanced Pattern Recognition..."):
                    top_300_tickers = df.nlargest(300, 'trend_score')['ticker'].tolist()
                    hist_dict = get_fast_histories(tuple(top_300_tickers))

                    vcp_list   = find_vcp_setups(df[df['ticker'].isin(top_300_tickers)], hist_dict)
                    # A stale index series would pair today's stock move with a
                    # days-old Nifty move and report nonsense; skip instead.
                    rs_list    = [] if _nifty_stale else find_rs_divergence(df, nifty_live)
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

                    # The logs belong to the EOD run (run_daily_signals.py), which dates rows by the
                    # trading session. The dashboard used to append to them on every page view under
                    # the wall-clock date -- with end-of-day data that files yesterday's signals as
                    # today's -- and rewrote three CSVs per rerun. It only reads them now.
                    from utils.advanced_scanners import load_signal_log as _load_log
                    rs_log_df    = _load_log(RS_LOG_FILE,    RS_LOG_COLS)
                    shock_log_df = _load_log(SHOCK_LOG_FILE, SHOCK_LOG_COLS)
                    tc_log_df    = _load_log(TC_LOG_FILE,    TC_LOG_COLS)

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
                    # "Today" is misleading here: this is the last COMPLETED
                    # daily Nifty bar in the cached session data (can lag the
                    # live intraday tape by up to auto_refresh_minutes), not
                    # the live running session. Put the actual bar date in
                    # the expander title itself — a caption underneath wasn't
                    # prominent enough and this got reported twice.
                    if nifty_live is not None and not nifty_live.empty:
                        _rs_asof = nifty_live.index[-1]
                        _rs_asof_str = _rs_asof.strftime('%Y-%m-%d') if hasattr(_rs_asof, 'strftime') else str(_rs_asof)
                    else:
                        _rs_asof_str = "unknown"
                    _rs_title = (f"🟢 RS Divergence — as of {_rs_asof_str} [{len(rs_list)}]"
                                 if not _nifty_stale else
                                 f"🔴 RS Divergence — STALE INDEX DATA ({_rs_asof_str})")
                    with st.expander(_rs_title, expanded=True):
                        if _nifty_stale:
                            st.error(
                                f"Index data ends at **{_rs_asof_str}** "
                                f"({_sessions_missed} trading session(s) missing) while stock "
                                f"prices are current. Scan suspended — pairing today's stock "
                                f"move with a stale index move would report false divergences. "
                                f"If those sessions were market holidays this is harmless; "
                                f"otherwise use 🔄 Hard Reset Cache to force a re-fetch.")
                        st.caption(f"Green in a sea of Red. Stock closed > +0.3% while Nifty fell > -0.3% "
                                   f"on the bar dated **{_rs_asof_str}** (refreshes every {auto_refresh_minutes} min — "
                                   f"may lag the live intraday tape).")
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
                            _uvcp['yf_link'] = _uvcp.apply(lambda r: us_quote_url(r['Ticker']), axis=1)
                            st.dataframe(
                                _uvcp[['yf_link', 'Price', 'Score', 'Compression', 'Vol_Ratio', 'MA50_Dist', 'Dist_52W']],
                                column_config={
                                    "yf_link":     st.column_config.LinkColumn("Ticker", display_text=us_display_re),
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
                            _urs['yf_link'] = _urs.apply(lambda r: us_quote_url(r['Ticker']), axis=1)
                            st.dataframe(
                                _urs[['yf_link', 'Stock_Ret', 'Nifty_Ret', 'Delta_RS', 'Dist_52W']],
                                column_config={
                                    "yf_link":   st.column_config.LinkColumn("Ticker", display_text=us_display_re),
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
                            _ushk['yf_link'] = _ushk.apply(lambda r: us_quote_url(r['Ticker']), axis=1)
                            st.dataframe(
                                _ushk[['yf_link', 'Jump_Pct', 'Vol_Mult', 'PEAD_Action']],
                                column_config={
                                    "yf_link":     st.column_config.LinkColumn("Ticker", display_text=us_display_re),
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
                        _utc['yf_link'] = _utc.apply(lambda r: us_quote_url(r['Ticker']), axis=1)
                        _utc['RS21_Flag'] = _utc['RS21_Vel'].apply(lambda v: "🚀" if v >= 5 else "")
                        st.dataframe(
                            _utc[['yf_link', 'Name', 'Sector', 'Price', 'Pattern', 'Jump%',
                                  'Vol_Score', 'Dist_52W', 'TS_Pre', 'TS_Now', 'TS_Gain', 'RS21_Vel', 'RS21_Flag']],
                            column_config={
                                "yf_link":   st.column_config.LinkColumn("Ticker", display_text=us_display_re),
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
                        _urs_disp['yf_link'] = _urs_disp.apply(lambda r: us_quote_url(r['ticker']), axis=1)
                        _urs_disp['signal_date'] = pd.to_datetime(_urs_disp['signal_date']).dt.strftime('%Y-%m-%d')
                        st.dataframe(
                            _urs_disp[['signal_date', 'yf_link', 'name', 'sector',
                                       'signal_price', 'current_price', 'return_since_signal',
                                       'stock_ret_on_day', 'nifty_ret_on_day', 'delta_rs',
                                       'dist_52w', 'days_held', 'status']],
                            column_config={
                                "signal_date":          st.column_config.TextColumn("Signal Date"),
                                "yf_link":              st.column_config.LinkColumn("Ticker", display_text=us_display_re),
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
                        _ushk_disp['yf_link'] = _ushk_disp.apply(lambda r: us_quote_url(r['ticker']), axis=1)
                        _ushk_disp['signal_date'] = pd.to_datetime(_ushk_disp['signal_date']).dt.strftime('%Y-%m-%d')
                        st.dataframe(
                            _ushk_disp[['signal_date', 'yf_link', 'name', 'sector',
                                        'signal_price', 'current_price', 'return_since_signal',
                                        'jump_pct', 'vol_mult', 'pead_action',
                                        'return_5d', 'return_21d', 'days_held', 'status']],
                            column_config={
                                "signal_date":         st.column_config.TextColumn("Signal Date"),
                                "yf_link":             st.column_config.LinkColumn("Ticker", display_text=us_display_re),
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
                        _utc_disp['yf_link'] = _utc_disp.apply(lambda r: us_quote_url(r['ticker']), axis=1)
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
                                "yf_link":             st.column_config.LinkColumn("Ticker", display_text=us_display_re),
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
