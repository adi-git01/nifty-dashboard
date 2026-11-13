"""Trend Scanner -> Agreement tab (logic in utils/consensus.py)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import consensus, simba
from utils.ui_search import search_box

TICK = {True: "✅", False: "—"}


def render(df: pd.DataFrame) -> None:
    st.markdown("### 🤝 Agreement — SIMBA · Trend Scanner · Technicals")
    st.caption(
        "Stocks that more than one system likes today, and a running log of every agreement with its "
        "return since. **SIMBA** = grade A/A+ with no gate hit, or an SI Opportunity in the last 5 days. "
        "**Scanner** = the engine's BUY signal (beating Nifty, above the 50-day average, liquid). "
        "**Technicals** = BUY NOW. This is a record to learn from — the combination has not been backtested.")

    px = pd.to_numeric(df.get("currentPrice", df.get("price")), errors="coerce")
    m200 = pd.to_numeric(df.get("twoHundredDayAverage"), errors="coerce")
    ok = px.notna() & m200.notna()
    breadth = float((px[ok] > m200[ok]).mean() * 100) if ok.sum() >= 50 else None
    reg = simba.market_regime(st.session_state.get("nifty_data"), breadth)
    try:
        from utils.simba_view import _fundamentals      # cached: reading the fundamentals files takes ~5 s
        v = consensus.votes(df, reg, fund=_fundamentals())
    except Exception as e:
        st.error(f"Could not compute today's votes: {e}")
        return
    has_simba = "simba_stage" in df.columns and df["simba_stage"].notna().any()
    if not has_simba:
        st.warning("SIMBA signals are not in today's data yet, so only Scanner and Technicals can vote "
                   "(2 of 3 at most) until the next EOD run.")

    k = st.columns(4)
    k[0].metric("All three agree", int((v.votes == 3).sum()))
    k[1].metric("Two agree", int((v.votes == 2).sum()))
    k[2].metric("Scanner BUY", int(v.scanner.sum()))
    k[3].metric("Technicals BUY NOW", int(v.technicals.sum()))

    st.markdown("#### Today")
    c1, c2 = st.columns([1, 2])
    need = c1.radio("Show", ["All three", "Two or more"], horizontal=True, key="agree_need")
    t = v[v.votes >= (3 if need == "All three" else 2)].copy()
    t = search_box(t, "agree_today_q", container=c2)
    for c in ("simba", "scanner", "technicals"):
        t[c] = t[c].map(TICK)
    cols = ["name", "ticker", "votes", "simba", "scanner", "technicals", "simba_grade", "simba_score", "currentPrice",
            "comp_rs", "return_1m", "sector"]
    cols = [c for c in cols if c in t.columns]
    if len(t):
        st.dataframe(t.sort_values(["votes", "simba_score", "comp_rs"], ascending=False)[cols], hide_index=True,
                     use_container_width=True,
                     column_config={"name": "Company", "ticker": "Ticker", "votes": "Votes", "simba": "SIMBA",
                                    "scanner": "Scanner", "technicals": "Technicals", "simba_grade": "SIMBA grade",
                                    "simba_score": st.column_config.NumberColumn("SIMBA score", format="%d"),
                                    "currentPrice": st.column_config.NumberColumn("Price", format="₹%.2f"),
                                    "comp_rs": st.column_config.NumberColumn("RS vs Nifty", format="%+.1f"),
                                    "return_1m": st.column_config.NumberColumn("1-month", format="%+.1f%%"),
                                    "sector": "Sector"})
    else:
        st.info("No stock meets that today.")

    st.markdown("#### Log and returns")
    log = consensus.load_log()
    if log.empty:
        st.info("The log starts with the next EOD run (3:45 pm IST). Each stock is added the first day it has "
                "two or three votes, at that day's price, and its return is updated daily.")
        return
    sm = consensus.summary(log)
    if not sm.empty:
        sm["max_votes"] = sm["max_votes"].map({3: "All three", 2: "Two"}).fillna(sm["max_votes"].astype(str))
        st.dataframe(sm, hide_index=True, use_container_width=True,
                     column_config={"max_votes": "Agreement reached", "stocks": "Stocks",
                                    "avg_return": st.column_config.NumberColumn("Average return", format="%+.1f%%"),
                                    "median_return": st.column_config.NumberColumn("Median return", format="%+.1f%%"),
                                    "winners_pct": st.column_config.NumberColumn("Winners", format="%d%%"),
                                    "avg_vs_nifty": st.column_config.NumberColumn("Average vs Nifty", format="%+.1f%%")})
        st.caption(f"Since {pd.to_datetime(log.first_seen).min():%d %b %Y}. Small samples and short holding "
                   "periods at first — treat early numbers as noise.")
    a, b, c = st.columns([1, 1, 2])
    only3 = a.toggle("Only all-three", value=False, key="agree_only3")
    only_active = b.toggle("Still agreeing", value=False, key="agree_active")
    L = log.copy()
    if only3:
        L = L[L.max_votes >= 3]
    if only_active:
        L = L[L.active == True]
    L = search_box(L, "agree_log_q", container=c)
    for col in ("simba", "scanner", "technicals"):
        L[col] = L[col].astype(bool).map(TICK)
    L["status"] = L["active"].map({True: "🟢 agreeing", False: "⚪ dropped out"})
    L["first_seen"] = pd.to_datetime(L.first_seen).dt.strftime("%d %b %Y")
    L["last_agree"] = pd.to_datetime(L.last_agree).dt.strftime("%d %b %Y")
    st.dataframe(L[["first_seen", "name", "ticker", "max_votes", "status", "simba", "scanner", "technicals",
                    "simba_grade", "signal_price", "current_price", "return_pct", "excess_pct", "sessions",
                    "last_agree"]], hide_index=True, use_container_width=True,
                 column_config={"first_seen": "Added", "name": "Company", "ticker": "Ticker", "max_votes": "Best votes",
                                "status": "Status", "simba": "SIMBA", "scanner": "Scanner", "technicals": "Technicals",
                                "simba_grade": "SIMBA grade",
                                "signal_price": st.column_config.NumberColumn("Price then", format="₹%.2f"),
                                "current_price": st.column_config.NumberColumn("Price now", format="₹%.2f"),
                                "return_pct": st.column_config.NumberColumn("Return", format="%+.1f%%"),
                                "excess_pct": st.column_config.NumberColumn("vs Nifty", format="%+.1f%%"),
                                "sessions": "Days held", "last_agree": "Last agreed"})
    st.download_button("⬇️ Download log (CSV)", log.to_csv(index=False).encode(), "consensus_log.csv", "text/csv",
                       key="agree_dl")
