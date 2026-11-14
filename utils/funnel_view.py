"""Investment Funnel page and the shared cached inputs (board, valuation, regime, funnel table)."""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from utils import funnel, simba
from utils.industry_board_view import get_board
from utils.ui_search import search_box

REG_ICON = {"WASHOUT": "🟣", "BEAR CONTRACTION": "🔴", "CORRECTION": "🟠", "NORMAL BULL": "🟢", "FROTH": "🟡", "UNKNOWN": "⚪"}
VAL_FILE = "data/valuation_snapshot.csv"
FUND_FILES = ("data/fundamentals_quarterly.csv.gz", "data/fundamentals_annual.csv.gz")


def snap_key(df: pd.DataFrame) -> str:
    bar = df["bar_date"].iloc[0] if "bar_date" in df.columns and len(df) else "?"
    return f"{len(df)}|{bar}|{float(pd.to_numeric(df.get('comp_rs'), errors='coerce').sum()):.2f}"


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _valuation(mtime: float) -> pd.DataFrame:
    return funnel.load_valuation(VAL_FILE)


def valuation() -> pd.DataFrame:
    return _valuation(os.path.getmtime(VAL_FILE) if os.path.exists(VAL_FILE) else 0.0)


def regime(df: pd.DataFrame) -> dict:
    px = pd.to_numeric(df.get("currentPrice", df.get("price")), errors="coerce")
    m200 = pd.to_numeric(df.get("twoHundredDayAverage"), errors="coerce")
    ok = px.notna() & m200.notna()
    breadth = float((px[ok] > m200[ok]).mean() * 100) if ok.sum() >= 50 else None
    try:
        return simba.market_regime(st.session_state.get("nifty_data"), breadth)
    except Exception:
        return {"regime": "UNKNOWN", "breadth": breadth}


@st.cache_data(ttl=6 * 3600, show_spinner="Running the funnel over the universe…")
def _table(key: str, fund_key: tuple, val_key: float, _df: pd.DataFrame) -> pd.DataFrame:
    return funnel.table(_df, get_board(_df), valuation())


def get_table(df: pd.DataFrame) -> pd.DataFrame:
    fk = tuple(os.path.getmtime(f) if os.path.exists(f) else 0 for f in FUND_FILES)
    return _table(snap_key(df), fk, os.path.getmtime(VAL_FILE) if os.path.exists(VAL_FILE) else 0.0, df)


def _show(P: pd.DataFrame, key: str, extra: tuple = ()):
    P = search_box(P, key, cols=("name", "ticker", "industry"))
    if P is None or P.empty:
        st.caption("Nothing here.")
        return
    V = pd.DataFrame({
        "Stock": P.ticker.str.replace(".NS", "", regex=False), "Name": P.name, **{c: P[c] for c in extra},
        "Industry": P.industry, "Ind.": P.ind_group.str.title(), "Price": P.price, "RS": P.comp_rs,
        "Technicals": P.technicals, "Stop": P.stop, "Risk %": P.risk_pct, "Target 1": P.t1,
        "Business": P.biz_headline, "Latest qtr profit YoY": P.np_yoy, "Flags": P.biz_flags,
        "Value pctile": P.val_pct, "To mid value": P.val_upside})
    st.dataframe(V, hide_index=True, use_container_width=True, height=min(len(V) * 35 + 40, 560), column_config={
        "Missed": st.column_config.TextColumn("Missed stage", help="The one stage this stock fails"),
        "Price": st.column_config.NumberColumn(format="₹%.1f"), "RS": st.column_config.NumberColumn(format="%+.1f"),
        "Stop": st.column_config.NumberColumn(format="₹%.1f", help="Technicals stop from the last close"),
        "Risk %": st.column_config.NumberColumn(format="%.1f%%", help="Distance from price to stop"),
        "Target 1": st.column_config.NumberColumn(format="₹%.1f"),
        "Latest qtr profit YoY": st.column_config.NumberColumn(format="%+.0f%%"),
        "Value pctile": st.column_config.NumberColumn(format="%.0f", help="Where the price sits in the valuation range (0 cheap, 100 expensive). Blank = not valued."),
        "To mid value": st.column_config.NumberColumn(format="%+.0f%%", help="Mid (P50) value vs price")})


def render(df: pd.DataFrame):
    reg = regime(df)
    name = reg.get("regime", "UNKNOWN")
    c = st.columns([2, 1, 1, 2])
    c[0].metric("0 · Market regime (SIMBA)", f"{REG_ICON.get(name, '⚪')} {name.title()}")
    c[1].metric("Risk scale", f"{reg['risk_scale']}/10" if reg.get("risk_scale") else "—")
    c[2].metric("Above 200-day", f"{reg['breadth']:.0f}%" if reg.get("breadth") is not None else "—")
    h = reg.get("holdings")
    c[3].metric("Suggested positions", f"{h[0]}–{h[1]}" if h else "—")
    if "BEAR" in name or name == "WASHOUT":
        st.warning("Bear regime: momentum entries lagged the typical stock here in the 10-year regime study. "
                   "Treat the list as a watchlist and keep new positions small.")

    with st.expander("How the funnel works, and what is and is not tested"):
        st.markdown(
            "| Stage | Question | Test | Tool | Evidence |\n|---|---|---|---|---|\n"
            "| 0 Market | How much risk? | Regime and risk scale | SIMBA regime | Momentum failed only in BEAR (10 yrs) |\n"
            f"| 1 Industry | Is money going here? | {funnel.WHAT['industry']} | Sector Pulse → Leadership Board | Small edge, 9 months only |\n"
            f"| 2 Trend | Is the stock leading? | {funnel.WHAT['trend']} | Trend Scanner | RS + Gate 1 beat the typical stock over 10 yrs |\n"
            f"| 3 Business | Is the business sound? | {funnel.WHAT['business']} | Business trends (Stock Report) | Judgement; untested |\n"
            f"| 4 Setup | Is there an entry now? | {funnel.WHAT['setup']} | Technicals, SIMBA | Stops and state tested; SIMBA untested |\n"
            f"| 5 Value | Am I overpaying? | {funnel.WHAT['value']} | Valuation | Off by default: expensive momentum beat cheap |\n"
            "| 6 Plan | Where am I wrong? | Stop, targets, size from the stop | Stock Report → Entry plan | — |\n"
            "| 7 Track | Is it working? | Position, alerts, Agreement log | Return Tracker, RS alerts | — |\n\n"
            "The funnel **as a whole has not been backtested**. It is a checklist that applies the dashboard's tools in a fixed "
            "order so nothing is skipped. Stocks that pass are logged in the Agreement tab's record over time.")

    T = get_table(df)
    st.markdown("#### Stages")
    cols = st.columns(len(funnel.STAGES))
    on = [k for (k, lab), col in zip(funnel.STAGES, cols)
          if col.checkbox(lab, value=k in funnel.DEFAULT_ON, key=f"fn_{k}", help=funnel.WHAT[k])]
    R = funnel.run(T, on)
    C = R["counts"]
    cc = st.columns(len(C))
    for col, (_, r), prev in zip(cc, C.iterrows(), [None] + C.left.tolist()[:-1]):
        col.metric(r.stage, int(r.left), None if prev is None else f"-{int(prev - r.left)}", delta_color="off")

    st.markdown(f"#### ✅ Passing every selected stage ({len(R['picks'])})")
    st.caption("Sorted by industry score, then RS. Open **Analysis Lab → 📋 Stock Report** for the full report on any of them.")
    _show(R["picks"], "fn_q")
    pk = R["picks"].ticker.tolist()
    if pk:
        a, b = st.columns([3, 1])
        t = a.selectbox("Open the one-shot report for", pk, format_func=lambda x: x.replace(".NS", ""), key="fn_open",
                        label_visibility="collapsed")
        if b.button("📋 Open report", key="fn_go", use_container_width=True):
            st.session_state["report_ticker"] = t
            st.session_state["nav_page"] = "📋 Stock Report"
            st.rerun()
    with st.expander(f"🟡 Near misses — fail exactly one stage ({len(R['near'])})"):
        st.caption("Most are waiting for an entry: put them on RS or price alerts.")
        N = R["near"]
        if len(N):
            _show(N.rename(columns={"missed": "Missed"}), "fn_nq", extra=("Missed",))
