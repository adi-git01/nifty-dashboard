"""
☀️ Daily Brief: the one page to read each day.

Replaces Morning Pulse, the Live Trading Desk and Turnaround Radar as the landing page. It keeps
only what either has evidence behind it or is needed to act: the market regime and breadth, which
industries lead, what passes the funnel today, the state of your own positions, and the model
portfolio against the market. The scanners those pages listed are in 🧪 Signal Lab with a scorecard.
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import funnel, industry_board as ib
from utils import funnel_view as fv
from utils.industry_board_view import get_board

BENCH = "data/model_portfolio_benchmark.csv"
SNAP = "data/dna3_portfolio_snapshot.json"
NEAR_STOP = 3.0


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _prev_pass(key: str, _board: pd.DataFrame) -> set:
    """Tickers that passed Trend and Setup in the previous snapshot (to mark today's new entries)."""
    files = sorted(glob.glob("data/cache/market_master_*.parquet"))
    if len(files) < 2:
        return set()
    try:
        p = pd.read_parquet(files[-2])
        T = funnel.table(p, _board, None, business_for=[])
        return set(T.ticker[T.s_trend & T.s_setup])
    except Exception:
        return set()


def _market(df: pd.DataFrame):
    from utils import regime_badge as rb
    st.markdown("### 1 · Market")
    rb.render(st.session_state.get("regime_info") or rb.compute(st.session_state.get("nifty_data"), df), st)
    f = "data/market_breadth_history.csv"
    if not os.path.exists(f):
        return
    h = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
    if len(h) < 6:
        return
    now, wk = h.iloc[-1], h.iloc[-6]
    c = st.columns(4)
    for col, k, lab in zip(c, ("pct_above_50dma", "pct_above_200dma", "pct_near_52w_high", "pct_strong_momentum"),
                           ("Above 50-day", "Above 200-day", "Near 52-week high", "Strong momentum")):
        col.metric(lab, f"{now[k]:.0f}%", f"{now[k] - wk[k]:+.1f} pp in a week")
    t = h.tail(120)
    fig = go.Figure()
    fig.add_scatter(x=t.date, y=t.pct_above_50dma, name="% above 50-day", line=dict(color="#4C78A8"))
    fig.add_scatter(x=t.date, y=t.pct_above_200dma, name="% above 200-day", line=dict(color="#F58518"))
    fig.add_hline(y=50, line_dash="dot", line_color="gray")
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.15),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Breadth is the share of the ~1,170 stocks above their own averages. Falling breadth while the index holds "
               "means fewer stocks are carrying the market: be pickier.")


def _industries(B: pd.DataFrame):
    st.markdown("### 2 · Industries")
    if B.empty:
        st.info("No industry data.")
        return
    c = st.columns(3)
    for col, g in zip(c, ("LEADING", "IMPROVING", "WEAKENING")):
        x = B[(B.group == g) & ~B.small].head(6)
        col.markdown(f"**{ib.GROUP_ICON[g]}** ({int((B.group == g).sum())})")
        col.markdown("\n".join(f"- {r.industry} · {r.score:.0f}" + (f" ({r.change:+.0f})" if pd.notna(r.change) else "")
                               for r in x.itertuples()) or "—")
    st.caption("Score 0–100 and its one-month change. Full table and the stocks inside each: **Sector Pulse**.")


def _funnel(df: pd.DataFrame, B: pd.DataFrame):
    st.markdown("### 3 · What passes the funnel today")
    R = funnel.run(fv.get_table(df))
    C = R["counts"]
    st.markdown(" → ".join(f"{r.stage} **{int(r.left)}**" for r in C.itertuples()))
    P = R["picks"]
    if P.empty:
        st.info("Nothing passes every stage today. Near misses are on the Investment Funnel page.")
        return
    old = _prev_pass(fv.snap_key(df), B)
    V = pd.DataFrame({
        "": ["🆕" if t not in old else "" for t in P.ticker], "Stock": P.ticker.str.replace(".NS", "", regex=False),
        "Name": P.name, "Industry": P.industry, "Price": P.price, "RS": P.comp_rs, "Technicals": P.technicals,
        "Stop": P.stop, "Risk %": P.risk_pct, "Target 1": P.t1, "Business": P.biz_headline})
    st.dataframe(V, hide_index=True, use_container_width=True, height=min(len(V) * 35 + 40, 420), column_config={
        "": st.column_config.TextColumn("", width="small", help="🆕 = did not have both an uptrend and an entry yesterday"),
        "Price": st.column_config.NumberColumn(format="₹%.1f"), "RS": st.column_config.NumberColumn(format="%+.1f"),
        "Stop": st.column_config.NumberColumn(format="₹%.1f"), "Risk %": st.column_config.NumberColumn(format="%.1f%%"),
        "Target 1": st.column_config.NumberColumn(format="₹%.1f")})
    st.caption(f"{int((V[''] == '🆕').sum())} new today. Stage switches and near misses: **Investment Funnel**. "
               "Full write-up on any stock: **Analysis Lab → Stock Report**.")


def _positions(df: pd.DataFrame, B: pd.DataFrame):
    st.markdown("### 4 · Your positions")
    try:
        from utils.positions import get_positions_with_pnl
        pos = [p for p in get_positions_with_pnl(df) if p.get("status", "active") == "active"]
    except Exception as e:
        st.warning(f"Positions unavailable: {e}")
        return
    if not pos:
        st.caption("No open positions. Add them in **Portfolio Manager → Return Tracker**.")
        return
    from utils.nifty1000_list import SUB_INDUSTRY_MAP
    by = df.drop_duplicates("ticker").set_index("ticker")
    grp = B.set_index("industry")["group"] if not B.empty else pd.Series(dtype=str)
    rows = []
    for p in pos:
        t = p.get("resolved_ticker") or p.get("ticker")
        px, sl, tg = p.get("current_price"), p.get("stop_loss_adj", p.get("stop_loss")), p.get("target_adj", p.get("target"))
        r = by.loc[t] if t in by.index else None
        gap = (px / sl - 1) * 100 if px and sl else None
        scan = str(r.get("dna_signal")) if r is not None else "—"
        below50 = bool(r is not None and pd.notna(r.get("fiftyDayAverage")) and px and px < r.get("fiftyDayAverage"))
        caution = float(r.get("simba_caution") or 0) if r is not None and pd.notna(r.get("simba_caution")) else 0
        if not p.get("has_data"):
            flag = "❔ no price"
        elif p.get("sl_triggered"):
            flag = "🔴 stop hit"
        elif p.get("target_triggered"):
            flag = "🎯 target hit"
        elif gap is not None and gap < NEAR_STOP:
            flag = f"⚠️ {gap:.1f}% above stop"
        elif caution >= 2 or (below50 and scan != "BUY"):
            flag = "🟣 trend weakening"
        else:
            flag = "✅ on track"
        rows.append({"Stock": str(p.get("ticker", "")).replace(".NS", ""), "Status": flag, "P&L %": p.get("pnl_pct"), "Price": px,
                     "Stop": sl, "Above stop %": gap, "Target": tg, "Scanner": scan,
                     "Industry": ib.GROUP_ICON.get(grp.get(SUB_INDUSTRY_MAP.get(t, ""), ""), "—")})
    V = pd.DataFrame(rows)
    need = V[~V.Status.str.startswith("✅")]
    (st.warning if len(need) else st.success)(
        f"{len(need)} of {len(V)} position(s) need a look: " + ", ".join(f"{r.Stock} ({r.Status})" for r in need.itertuples())
        if len(need) else f"All {len(V)} position(s) on track.")
    st.dataframe(V, hide_index=True, use_container_width=True, column_config={
        "P&L %": st.column_config.NumberColumn(format="%+.1f%%"), "Price": st.column_config.NumberColumn(format="₹%.1f"),
        "Stop": st.column_config.NumberColumn(format="₹%.1f"), "Target": st.column_config.NumberColumn(format="₹%.1f"),
        "Above stop %": st.column_config.NumberColumn(format="%.1f%%"),
        "Industry": st.column_config.TextColumn(help="Where the stock's industry sits on the leadership board")})
    st.caption("Trend weakening = below the 50-day with no Scanner BUY, or SIMBA caution level 2+. Edit stops and targets in "
               "**Portfolio Manager → Return Tracker**.")


def _model_portfolio():
    st.markdown("### 5 · Model portfolio (paper)")
    if not (os.path.exists(BENCH) and os.path.exists(SNAP)):
        st.caption("Benchmark file not built yet; it appears after the next end-of-day run.")
        return
    b = pd.read_csv(BENCH, parse_dates=["date"])
    try:
        snap = json.load(open(SNAP))
    except Exception:
        snap = {}
    last = b.iloc[-1]
    c = st.columns(5)
    c[0].metric("Model", f"{last.model - 100:+.1f}%", help=f"Since {b.date.iloc[0]:%d %b %Y}")
    c[1].metric("Equal-weight market", f"{last.equal_weight - 100:+.1f}%", f"{last.model - last.equal_weight:+.1f} pp model vs this",
                delta_color="normal")
    if "top50" in b:
        c[2].metric("50 largest stocks", f"{last.top50 - 100:+.1f}%", f"{last.model - last.top50:+.1f} pp model vs this")
    c[3].metric("Holdings", int(last.holdings))
    c[4].metric("Cash", f"{last.cash_pct:.0f}%", f"avg {b.cash_pct.mean():.0f}%", delta_color="off")
    fig = go.Figure()
    for col, lab, colr in (("model", "Model portfolio", "#54A24B"), ("equal_weight", "Equal-weight market", "#9D9D9D"),
                           ("top50", "50 largest stocks", "#4C78A8")):
        if col in b:
            fig.add_scatter(x=b.date, y=b[col], name=lab, line=dict(color=colr, width=2.5 if col == "model" else 1.5))
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.15),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The momentum model (OptComp V2.2), tracked daily with paper money. Benchmarks are built from the same "
               "snapshots: every stock equally weighted, and the 50 largest by market value. Holdings, trades and the "
               "refresh button are on **Trend Scanner**.")
    pf = pd.DataFrame(snap.get("portfolio") or [])
    if len(pf):
        with st.expander(f"Holdings ({len(pf)})"):
            keep = [c for c in ("Ticker", "Sector", "Entry", "Price", "PnL%", "Trail_Stop", "Danger") if c in pf.columns]
            st.dataframe(pf[keep], hide_index=True, use_container_width=True)


def render(df: pd.DataFrame):
    B = get_board(df)
    _market(df)
    _industries(B)
    _funnel(df, B)
    _positions(df, B)
    _model_portfolio()
    st.caption("Scanner lists that used to be on Morning Pulse, the Live Trading Desk and Turnaround Radar are in "
               "**🧪 Signal Lab**, with a scorecard of how each has done. The old pages are under the **🗄️ Archive** workspace.")
