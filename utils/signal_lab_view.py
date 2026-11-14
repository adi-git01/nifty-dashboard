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
        + "- One market regime and under a year of data. Treat these lists as ideas to check in the **Stock Report**, not as buy signals.")


def render(df: pd.DataFrame):
    _scorecard()
    st.markdown("---")
    st.markdown("#### Signals")
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
