"""Investment Funnel page and the shared cached inputs (board, valuation, regime, funnel table)."""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from utils import funnel, simba
from utils.industry_board_view import get_board
from utils.ui_search import search_box

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


LADDER = ["BEAR", "MILD BEAR", "CHOPPY", "MILD BULL", "BULL"]
GUIDE = {"BEAR": "No new positions; protect what you hold", "MILD BEAR": "Half-size positions, only the best setups",
         "CHOPPY": "Normal size, be selective", "MILD BULL": "Full size", "BULL": "Full size", "UNKNOWN": "—"}
ICON = {"BEAR": "🔴", "MILD BEAR": "🟠", "CHOPPY": "🟡", "MILD BULL": "🟢", "BULL": "🟢", "UNKNOWN": "⚪"}


def regime(df: pd.DataFrame) -> dict:
    """
    The dashboard's one market read: the same Nifty and breadth regimes as the sidebar
    (utils/regime_badge.py, tested in analysis/REGIME_VERDICT.md). Risk guidance follows the weaker
    of the two. SIMBA keeps its own regime only for its own score (simba_regime).
    """
    from utils import regime_badge as rb
    info = st.session_state.get("regime_info") or rb.compute(st.session_state.get("nifty_data"), df)
    n, b = info["nifty"].get("regime", "UNKNOWN"), info["breadth"].get("regime", "UNKNOWN")
    known = [x for x in (n, b) if x in LADDER]
    weak = min(known, key=LADDER.index) if known else "UNKNOWN"
    return dict(regime=n, breadth_regime=b, breadth=info["breadth"].get("pct"), engine=info["engine"].get("mode", "UNKNOWN"),
                risk=weak, guidance=GUIDE[weak], info=info)


def simba_regime(df: pd.DataFrame) -> dict:
    """SIMBA's own regime, used only to compute the SIMBA score."""
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
        "Technicals": P.technicals, "Stop": P.stop, "Risk %": P.risk_pct, "2× risk ref": P.t1,
        "Business": P.biz_headline, "Latest qtr profit YoY": P.np_yoy, "Flags": P.biz_flags,
        "Value pctile": P.val_pct, "To mid value": P.val_upside})
    st.dataframe(V, hide_index=True, use_container_width=True, height=min(len(V) * 35 + 40, 560), column_config={
        "Missed": st.column_config.TextColumn("Missed stage", help="The one stage this stock fails"),
        "Price": st.column_config.NumberColumn(format="₹%.1f"), "RS": st.column_config.NumberColumn(format="%+.1f"),
        "Stop": st.column_config.NumberColumn(format="₹%.1f", help="Technicals stop from the last close"),
        "Risk %": st.column_config.NumberColumn(format="%.1f%%", help="Distance from price to stop"),
        "2× risk ref": st.column_config.NumberColumn(format="₹%.1f", help="A yardstick, not an exit: booking at 2× risk halved returns in the backtest"),
        "Latest qtr profit YoY": st.column_config.NumberColumn(format="%+.0f%%"),
        "Value pctile": st.column_config.NumberColumn(format="%.0f", help="Where the price sits in the valuation range (0 cheap, 100 expensive). Blank = not valued."),
        "To mid value": st.column_config.NumberColumn(format="%+.0f%%", help="Mid (P50) value vs price")})


def render(df: pd.DataFrame):
    reg = regime(df)
    c = st.columns([1.2, 1.2, 1, 2.2])
    c[0].metric("0 · Nifty regime", f"{ICON.get(reg['regime'], '⚪')} {reg['regime'].title()}")
    c[1].metric("Breadth regime", f"{ICON.get(reg['breadth_regime'], '⚪')} {reg['breadth_regime'].title()}",
                f"{reg['breadth']:.0f}% above 200-day" if reg.get("breadth") is not None else None, delta_color="off")
    c[2].metric("Engine mode", reg["engine"].title())
    c[3].metric("Risk guidance", reg["guidance"])
    st.caption("One market read for the whole dashboard, the same as the sidebar. Guidance follows the weaker of the Nifty and "
               "breadth regimes; in the 10-year regime study momentum entries lagged the typical stock only in BEAR.")
    if reg["risk"] == "BEAR":
        st.warning("Bear regime: treat the list as a watchlist.")

    with st.expander("How the funnel works, and what is and is not tested"):
        st.markdown(
            "| Stage | Question | Test | Tool | Evidence |\n|---|---|---|---|---|\n"
            "| 0 Market | How much risk? | Nifty and breadth regimes | Regime badge (sidebar) | Momentum failed only in BEAR (10 yrs) |\n"
            f"| 1 Industry | Is money going here? | {funnel.WHAT['industry']} | Sector Pulse → Leadership Board | Scanner BUY in a leading industry beat plain Scanner BUY (Signal Lab scorecard, 9 months) |\n"
            f"| 2 Trend | Is the stock leading? | {funnel.WHAT['trend']} | Trend Scanner | RS + Gate 1 beat the typical stock over 10 yrs |\n"
            f"| 3 Business | Is the business sound? | {funnel.WHAT['business']} | Business score (Stock Report) | A quality screen: no return edge in the 2021–25 test |\n"
            f"| 4 Setup | Is there an entry now? | {funnel.WHAT['setup']} | Technicals, SIMBA | Stops and state tested; SIMBA untested |\n"
            f"| 5 Value | Am I overpaying? | {funnel.WHAT['value']} | Valuation | Off by default: expensive momentum beat cheap |\n"
            "| 6 Plan | Where am I wrong? | Stop, targets, size from the stop | Stock Report → Entry plan | — |\n"
            "| 7 Track | Is it working? | Position, alerts, Agreement log | Return Tracker, RS alerts | — |\n\n"
            "The funnel **as a whole has not been backtested**. Every day's passes and near misses are now logged "
            "(`data/funnel_log.csv`) and scored against the market on **Signal Lab → Scorecard**, so this will be answered "
            "with evidence over the coming months.")

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
