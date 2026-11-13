"""Sector Pulse: the industry leadership board (utils/industry_board.py)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import industry_board as ib
from utils.ui_search import search_box

STATUS_ICON = {"UPTREND": "🟢 Uptrend", "TURNING UP": "🟡 Turning up", "BASING": "⚪ Basing",
               "WEAKENING": "🟣 Weakening", "DOWNTREND": "🔴 Downtrend"}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _board(key: str, _df: pd.DataFrame) -> pd.DataFrame:
    return ib.board(_df, ib.month_ago_snapshot())


def get_board(df: pd.DataFrame) -> pd.DataFrame:
    """The board for the loaded snapshot, cached per snapshot."""
    key = f"{len(df)}|{df.get('bar_date', pd.Series(['?'])).iloc[0]}|{float(pd.to_numeric(df.get('comp_rs'), errors='coerce').sum()):.2f}"
    return _board(key, df)


def render(df: pd.DataFrame):
    st.markdown("### 🏁 Industry Leadership Board")
    B = get_board(df)
    if B.empty:
        st.info("No industry data in the loaded snapshot.")
        return
    st.caption("Which industries lead **and** whether the strength is broad and holding. Score 0–100 ranks each industry on "
               "3-month strength vs Nifty (35%), share of stocks above the 200-day (25%) and 50-day (20%) averages, and share "
               "near 52-week highs (20%). Change is against about a month ago.")
    n = B.group.value_counts()
    c = st.columns(4)
    for col, g in zip(c, ("LEADING", "IMPROVING", "WEAKENING", "LAGGING")):
        names = B[B.group == g].industry.head(4).tolist()
        col.metric(ib.GROUP_ICON[g], int(n.get(g, 0)))
        col.caption(", ".join(names) if names else "—")

    f1, f2 = st.columns([2, 3])
    pick = f1.multiselect("Show", list(ib.GROUP_ICON), default=[], placeholder="All groups", key="ib_groups",
                          format_func=lambda g: ib.GROUP_ICON[g])
    hide_small = f2.checkbox(f"Hide industries with fewer than {ib.SMALL_GROUP} stocks (one stock can swing them)",
                             value=False, key="ib_small")
    V = B[B.group.isin(pick)] if pick else B
    if hide_small:
        V = V[~V.small]
    V = search_box(V, "ib_q", cols=("industry", "leaders"), label="🔎 Search industry or stock")
    T = pd.DataFrame({
        "Industry": V.industry + V.small.map({True: " ⚠︎", False: ""}), "Group": V.group.map(ib.GROUP_ICON),
        "Score": V.score, "1M Δ": V.change, "Trend": V.status.map(STATUS_ICON).fillna(V.status),
        "3M vs Nifty": V.rs_3m, "> 50D": V.above_50, "> 200D": V.above_200, "Near high": V.near_high,
        "On BUY": V.buy_pct, "Stocks": V.members, "Strongest": V.leaders})
    pc = lambda t, h: st.column_config.NumberColumn(t, format="%.0f%%", help=h)
    st.dataframe(T, hide_index=True, use_container_width=True, height=min(len(T) * 35 + 40, 720), column_config={
        "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
        "1M Δ": st.column_config.NumberColumn("1M Δ", format="%+.0f", help="Score now minus score ~21 sessions ago"),
        "Trend": st.column_config.TextColumn("Trend", help="Uptrend: most members above both averages with positive RS. "
                                             "Turning up: breadth above the 50-day recovering first. Weakening: above the "
                                             "200-day but losing the 50-day."),
        "3M vs Nifty": st.column_config.NumberColumn("3M vs Nifty", format="%+.1f", help="Median member 3-month return minus Nifty, pp"),
        "> 50D": pc("> 50D", "Share of members above their 50-day average"),
        "> 200D": pc("> 200D", "Share of members above their 200-day average"),
        "Near high": pc("Near high", "Share of members within 10% of the 52-week high"),
        "On BUY": pc("On BUY", "Share of members with the Trend Scanner BUY signal"),
        "Strongest": st.column_config.TextColumn("Strongest", width="medium", help="Top members by RS that are above the 50-day")})
    st.caption("⚠︎ = small group. Leading = score 60+ with at least half the stocks above the 50-day, and not down 10+ in a "
               "month · Weakening = the same but down 10+ · Improving = not a leader but up 10+ · Lagging = the rest. On Feb–Oct 2026 snapshots the top fifth by this score "
               "beat the average stock by ~0.6 pp over the next month and the rest were flat: a map of where strength is, "
               "not a tested edge.")

    with st.expander("🔎 Stocks inside an industry"):
        g = st.selectbox("Industry", B.industry.tolist(), key="ib_pick")
        from utils.nifty1000_list import SUB_INDUSTRY_MAP
        m = df[df.ticker.map(SUB_INDUSTRY_MAP) == g].copy()
        px = pd.to_numeric(m.get("currentPrice"), errors="coerce")
        M = pd.DataFrame({
            "Stock": m.ticker.str.replace(".NS", "", regex=False), "Name": m.get("name"),
            "RS": pd.to_numeric(m.get("comp_rs"), errors="coerce"), "3M vs Nifty": pd.to_numeric(m.get("rs_3m"), errors="coerce"),
            "> 50D": (px > pd.to_numeric(m.get("fiftyDayAverage"), errors="coerce")).map({True: "✅", False: "—"}),
            "> 200D": (px > pd.to_numeric(m.get("twoHundredDayAverage"), errors="coerce")).map({True: "✅", False: "—"}),
            "From high": pd.to_numeric(m.get("dist_52w"), errors="coerce"), "Scanner": m.get("dna_signal"),
            "Technicals": m.get("fx_action")}).sort_values("RS", ascending=False)
        st.dataframe(M, hide_index=True, use_container_width=True, column_config={
            "RS": st.column_config.NumberColumn(format="%+.1f"), "3M vs Nifty": st.column_config.NumberColumn(format="%+.1f"),
            "From high": st.column_config.NumberColumn(format="%.0f%%")})
