"""
🧪 Signal Lab: every scanner's signals in one place, under a scorecard of how each has done.

Replaces the scanner sections of the Live Trading Desk, Morning Pulse and Turnaround Radar.
The scorecard (utils/signal_scorecard.py) is rebuilt by each end-of-day run.
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from utils import signal_scorecard as sc
from utils.ui_search import search_box

ICON = {"BEATS MOMENTUM": "🟢 Beats momentum", "LIKE MOMENTUM": "🟡 Like plain momentum", "NO EDGE": "⚪ No edge",
        "LAGS MARKET": "🔴 Lags the market", "TOO NEW": "🕐 Too new", "YARDSTICK": "📏 Yardstick"}
# label -> (file, date column, columns to show)
BROWSE = {
    "Turnaround Radar — today's watchlist": ("data/turnaround_watchlist.csv", "Date",
                                             ["Ticker", "Name", "Sub_Industry", "Tier", "IAS", "CMP", "Off_52W_High", "Off_MA50", "CompRS", "RS21"]),
    "Turnaround Radar — signal log": ("data/ias_signal_log.csv", "signal_date",
                                      ["signal_date", "ticker", "name", "sub_industry", "signal_tier", "peak_tier", "status", "signal_price",
                                       "current_price", "return_since_signal", "return_21d", "return_63d", "max_gain"]),
    "Turnaround Catalyst": ("data/tc_log.csv", "signal_date",
                            ["signal_date", "ticker", "name", "sector", "pattern", "jump_pct", "signal_price", "current_price",
                             "return_since_signal", "days_held", "status"]),
    "Earnings Shock": ("data/earnings_shock_log.csv", "signal_date",
                       ["signal_date", "ticker", "name", "sector", "jump_pct", "vol_mult", "signal_price", "current_price",
                        "return_5d", "return_21d", "return_since_signal", "status"]),
    "RS Divergence": ("data/rs_divergence_log.csv", "signal_date",
                      ["signal_date", "ticker", "name", "sector", "stock_ret_on_day", "nifty_ret_on_day", "dist_52w", "signal_price",
                       "current_price", "return_since_signal", "status"]),
    "Breakouts": ("data/breakout_log.csv", "breakout_date",
                  ["breakout_date", "ticker", "name", "kind", "vol_x", "ind_score", "signal_price", "current_price", "return_pct",
                   "sessions_since"]),
    "V2.2 watchlist BUY alerts": ("data/watchlist_history.csv", "added_date", ["added_date", "ticker", "sector", "v3_score", "rs_score", "status"]),
}


@st.cache_data(ttl=3600, show_spinner=False)
def _csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def _load(path: str) -> pd.DataFrame:
    return _csv(path, os.path.getmtime(path)) if os.path.exists(path) else pd.DataFrame()


def _scorecard():
    S = _load(sc.OUT)
    if S.empty:
        st.info("The scorecard is built by the end-of-day run and will appear after the next one.")
        return
    st.markdown(f"#### Scorecard — {int(S.sessions.iloc[0])} sessions to {pd.Timestamp(S.as_of.iloc[0]):%d %b %Y}")
    st.caption("Average return of the signalled stocks **minus the median stock** over the same days, by sessions after the "
               "signal. The yardstick is every stock on the Trend Scanner BUY list, every day: a scanner is worth watching only "
               "if it beats that.")
    V = pd.DataFrame({
        "Signal": S.signal, "Verdict": S.verdict.map(ICON).fillna(S.verdict), "Signals": S.signals,
        "5 sessions": S.avg_5d, "21 sessions": S.avg_21d, "Winners 21": S.hit_21d, "63 sessions": S.avg_63d,
        "Winners 63": S.hit_63d, "Outcomes (21 / 63)": S.n_21d.astype(int).astype(str) + " / " + S.n_63d.astype(int).astype(str),
        "Shown on": S["where"], "Idea": S.claim})
    pp = lambda t: st.column_config.NumberColumn(t, format="%+.1f pp")
    st.dataframe(V, hide_index=True, use_container_width=True, height=len(V) * 35 + 40, column_config={
        "5 sessions": pp("5 sessions"), "21 sessions": pp("21 sessions"), "63 sessions": pp("63 sessions"),
        "Winners 21": st.column_config.NumberColumn("Beat median, 21", format="%.0f%%", help="Share of signals that beat the median stock"),
        "Winners 63": st.column_config.NumberColumn("Beat median, 63", format="%.0f%%"),
        "Outcomes (21 / 63)": st.column_config.TextColumn(help="Signals old enough to have a 21- / 63-session outcome")})
    lag = S[S.verdict == "LAGS MARKET"].signal.str.strip(" …").tolist()
    beat = S[S.verdict == "BEATS MOMENTUM"].signal.tolist()
    st.markdown(
        (f"- 🟢 **Beating plain momentum:** {', '.join(beat)}.\n" if beat else "- **Nothing here beats plain momentum so far.** "
         "Earnings Shock and the V2.2 alerts are positive only because they are momentum lists; they add nothing to the Scanner BUY list.\n")
        + (f"- 🔴 **Behind the market:** {', '.join(lag)}.\n" if lag else "")
        + "- **RS Divergence** has clean data only since September 2026: too short to judge either way.\n"
        + "- One market regime and under a year of data. Treat these lists as ideas to check in the **Stock Report**, not as buy signals.")


def _strong(df: pd.DataFrame, kind: str):
    from utils import funnel_view as fv, strong_names as sn
    T = fv.get_table(df)
    pc = lambda f="%+.1f%%": st.column_config.NumberColumn(format=f)
    if kind == "rs":
        st.caption("Stocks that **rose on days the Nifty fell**. One row per stock; more red days held = more persistent "
                   "strength. The Funnel column says whether it is buyable today.")
        a, b, c = st.columns([1, 1, 2])
        days = a.selectbox("Window", [10, 30, 60], index=1, format_func=lambda d: f"Last {d} days", key="sn_rs_days")
        mn = b.selectbox("Held up on at least", [1, 2, 3, 4], index=1, format_func=lambda n: f"{n} red day(s)", key="sn_rs_min")
        only = c.checkbox("Only stocks in an uptrend (Scanner BUY)", value=False, key="sn_rs_buy")
        X = sn.rs_divergence(T, days)
        if X.empty:
            st.info("No RS divergence signals in this window.")
            return
        X = X[X.red_days >= mn]
        X = X[X.s_trend.eq(True)] if only else X
        X = search_box(X, "sn_rs_q")
        V = pd.DataFrame({"": X.today.map({True: "🆕", False: ""}), "Stock": X.ticker.str.replace(".NS", "", regex=False), "Name": X.name,
                          "Red days held": X.red_days, "Avg beat": X.avg_gap, "Latest": X.latest, "Since first": X.since_first,
                          "Funnel": X.funnel, "Technicals": X.technicals, "From high": X.dist_52w, "RS": X.comp_rs,
                          "Industry": X.industry, "Ind.": X.ind_group.astype(str).str.title()})
        st.dataframe(V, hide_index=True, use_container_width=True, height=min(len(V) * 35 + 40, 620), column_config={
            "": st.column_config.TextColumn("", width="small", help="🆕 = flagged on the latest red day"),
            "Avg beat": st.column_config.NumberColumn(format="%+.1f pp", help="Average of (stock move − Nifty move) on those days"),
            "Latest": st.column_config.DateColumn(format="DD MMM"), "Since first": pc(), "From high": pc("%.0f%%"),
            "RS": st.column_config.NumberColumn(format="%+.1f")})
        st.caption(f"{len(V)} stock(s). The log has clean data only since September 2026, so its track record is too short to "
                   "judge (see Scorecard). Use it to find names, then check them in **Stock Report**.")
    else:
        st.caption("Stocks that **jumped 4%+ on 2.5× volume**, usually on results. One row per stock, from its first jump in "
                   "the window. So far the drift has continued only when the stock was already in an uptrend "
                   "(+5.6 pp over 63 sessions v +0.8 pp when not); a bigger jump did not help.")
        a, b, c = st.columns([1, 1.6, 1.6])
        days = a.selectbox("Window", [15, 45, 90], index=1, format_func=lambda d: f"Last {d} days", key="sn_es_days")
        only = b.checkbox("Only stocks in an uptrend (Scanner BUY)", value=True, key="sn_es_buy")
        hold = c.checkbox("Only still above the jump-day close", value=False, key="sn_es_hold")
        X = sn.earnings_drift(T, days)
        if X.empty:
            st.info("No earnings jumps in this window.")
            return
        X = X[X.s_trend.eq(True)] if only else X
        X = X[X.holds] if hold else X
        X = search_box(X, "sn_es_q")
        V = pd.DataFrame({"Stock": X.ticker.str.replace(".NS", "", regex=False), "Name": X.name, "Jump day": X.jump_date,
                          "Jump": X.jump_pct, "Volume": X.vol_mult, "Since": X.since, "Holds jump": X.holds.map({True: "✅", False: "❌"}),
                          "Sessions": X.sessions, "Funnel": X.funnel, "Technicals": X.technicals, "From high": X.dist_52w,
                          "Industry": X.industry, "Ind.": X.ind_group.astype(str).str.title()})
        st.dataframe(V, hide_index=True, use_container_width=True, height=min(len(V) * 35 + 40, 620), column_config={
            "Jump day": st.column_config.DateColumn(format="DD MMM"), "Jump": pc(), "Since": pc(), "From high": pc("%.0f%%"),
            "Volume": st.column_config.NumberColumn(format="%.1f×"),
            "Holds jump": st.column_config.TextColumn(help="Price still at or above the close of the jump day")})
        st.caption(f"{len(V)} stock(s). Check the quarter itself under **Stock Report → Business trends** before acting.")


def render(df: pd.DataFrame):
    t1, t2, t3, t4 = st.tabs(["💪 RS Divergence", "⚡ Earnings Drift", "📏 Scorecard", "🗂 All signal logs"], key="sl_tab", on_change="rerun")
    with t1:
        _strong(df, "rs")
    with t2:
        _strong(df, "es")
    with t3:
        _scorecard()
    with t4:
        _logs()


def _logs():
    a, b = st.columns([2, 1])
    which = a.selectbox("List", list(BROWSE), key="sl_which")
    path, dcol, cols = BROWSE[which]
    L = _load(path)
    if L.empty:
        st.info("No rows in this log yet.")
        return
    L = L.rename(columns={"Ticker": "ticker", "Name": "name"}) if "Ticker" in L.columns else L
    cols = [{"Ticker": "ticker", "Name": "name"}.get(c, c) for c in cols]
    L[dcol] = pd.to_datetime(L[dcol], errors="coerce")
    last = L[dcol].max()
    span = b.radio("Show", ["Latest day", "Last 30 days", "All"], horizontal=True, key="sl_span")
    if span == "Latest day":
        L = L[L[dcol] == last]
    elif span == "Last 30 days":
        L = L[L[dcol] >= last - pd.Timedelta(days=30)]
    L = search_box(L, "sl_q", cols=tuple(c for c in ("name", "ticker") if c in L.columns))
    V = L[[c for c in cols if c in L.columns]].sort_values(dcol, ascending=False) if dcol in cols else L[[c for c in cols if c in L.columns]]
    if "ticker" in V:
        V = V.assign(ticker=V.ticker.astype(str).str.replace(".NS", "", regex=False))
    fmt = {c: st.column_config.NumberColumn(format="%+.1f%%") for c in V.columns if c.startswith("return") or c in ("max_gain", "jump_pct")}
    fmt[dcol] = st.column_config.DateColumn(format="DD MMM YYYY")
    st.dataframe(V, hide_index=True, use_container_width=True, height=min(len(V) * 35 + 40, 600), column_config=fmt)
    st.caption(f"{len(V)} row(s); latest signal day {last:%d %b %Y}. The original pages with their charts and filters are under "
               "the **🗄️ Archive** workspace.")
