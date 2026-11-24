"""
☀️ Daily Brief: the one page to read each day.

Replaces Morning Pulse, the Live Trading Desk and Turnaround Radar as the landing page. It keeps
only what either has evidence behind it or is needed to act: the market regime and breadth, which
industries lead, what passes the funnel today, the state of your own positions, and the model
portfolio against the market. The scanners those pages listed are in 🧪 Signal Lab with a scorecard.
"""
from __future__ import annotations

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


@st.cache_data(ttl=1800, show_spinner=False)
def _read_csv_cached(path: str, mtime: float, parse_dates: tuple = ()) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=list(parse_dates) if parse_dates else False)


@st.cache_data(ttl=1800, show_spinner=False)
def _read_json_cached(path: str, mtime: float) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _prev_pass(key: str, _df: pd.DataFrame) -> set:
    """Tickers that passed Trend and Setup in the previous unique trading session."""
    from utils import snapshots
    prev_path = snapshots.snapshot_n_sessions_ago(1, current_df=_df)
    if not prev_path:
        return set()
    try:
        p = pd.read_parquet(prev_path)
        T = funnel.table(p, None, None, business_for=[])
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
    h = _read_csv_cached(f, os.path.getmtime(f), ("date",)).sort_values("date")
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
    _nse_context()


# reference palette (dataviz skill), slots 1-3: validated for colour-blind separation in every chart form
_SERIES = {"Nifty 50": "#2a78d6", "Midcap 150": "#eb6834", "Smallcap 250": "#1baf7a"}
_FII, _DII = "#2a78d6", "#eb6834"
_INK, _MUTED, _GRID = "#0b0b0b", "#52514e", "rgba(82,81,78,0.15)"
_rs = lambda v: f"{'+' if v >= 0 else '−'}₹{abs(v):,.0f}"          # +₹10,308 / −₹4,860
_rs_k = lambda v: f"{'+' if v >= 0 else '−'}₹{abs(v) / 1000:.1f}k" if abs(v) >= 1000 else _rs(v)   # end labels on a phone


@st.cache_data(ttl=3600, show_spinner=False)
def _ctx_history(_stamp: str) -> pd.DataFrame:
    from utils import nse_data as nd
    return nd.context_history()


def _ctx_stamp() -> str:
    import glob
    files = sorted(glob.glob("data/nse/indices/*.parquet")) + sorted(glob.glob("data/nse/indices/d/*.parquet")) + ["data/nse/market_context.csv"]
    return "|".join(f"{f}:{os.path.getmtime(f):.0f}" for f in files[-3:] if os.path.exists(f))


def _ctx_layout(fig: go.Figure, title: str, ysuffix: str = "", height: int = 300) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=8, r=8, t=56, b=8),
                      title=dict(text=title, x=0, font=dict(size=14, color=_INK)),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(color=_MUTED)),
                      hovermode="x unified", font=dict(color=_MUTED))
    fig.update_xaxes(showgrid=False, linecolor=_GRID)
    fig.update_yaxes(gridcolor=_GRID, zeroline=False, ticksuffix=ysuffix)
    return fig


def _flow_x_labels(idx: pd.DatetimeIndex) -> list[str]:
    """Category axis labels for flows chart: add 2-digit year when range spans multiple years."""
    if len(idx) == 0:
        return []
    fmt = "%d %b %y" if idx.year.nunique() > 1 else "%d %b"
    return [d.strftime(fmt) for d in idx]


def _last_valid_xy(xs: list[str], y: pd.Series):
    """Return `(x_label, float_val)` for the last finite value in `y` aligned with `xs`, else None."""
    arr = pd.to_numeric(y, errors="coerce").to_numpy()
    ok = [i for i, v in enumerate(arr) if pd.notna(v)]
    if not ok or ok[-1] >= len(xs):
        return None
    j = ok[-1]
    return xs[j], float(arr[j])


def _context_charts():
    """P/E of Nifty / Midcap 150 / Smallcap 250, FII index-futures long %, and FII vs DII net cash, from NSE."""
    H = _ctx_history(_ctx_stamp())
    if H is None or H.empty:
        st.caption("No NSE history yet.")
        return
    span = st.radio("Range", ["6M", "1Y", "2Y"], index=1, horizontal=True, key="ctx_range", label_visibility="collapsed")
    days = {"6M": 183, "1Y": 365, "2Y": 730}[span]
    W = H[H.index >= H.index.max() - pd.Timedelta(days=days)]

    # 1. valuation: three P/E lines on one axis (same unit), legend + end labels
    fig = go.Figure()
    notes = []
    for name, col in (("Nifty 50", "nifty_pe"), ("Midcap 150", "midcap_pe"), ("Smallcap 250", "smallcap_pe")):
        if col not in W or W[col].dropna().empty:
            continue
        y = W[col].dropna()
        fig.add_trace(go.Scatter(x=y.index, y=y, name=name, mode="lines", line=dict(color=_SERIES[name], width=2),
                                 hovertemplate=f"{name}: %{{y:.1f}}×<extra></extra>"))
        fig.add_annotation(x=y.index[-1], y=y.iloc[-1], text=f"{y.iloc[-1]:.1f}", showarrow=False, xanchor="left", xshift=4,
                           font=dict(size=11, color=_INK))
        pct = (y < y.iloc[-1]).mean() * 100
        notes.append(f"{name} {y.iloc[-1]:.1f} (range {y.min():.1f}–{y.max():.1f}, {pct:.0f}th percentile)")
    if fig.data:
        st.plotly_chart(_ctx_layout(fig, f"P/E · last {span}", "×"), use_container_width=True, config={"displayModeBar": False})
        st.caption("Today vs the period shown: " + " · ".join(notes) + ". Higher = richer; your 10-year Nifty range is roughly 18–28.")

    # 2. FII index-futures positioning: one series against its own median (no fixed threshold: FIIs sat under 30%
    #    long on 433 of the 503 sessions from Nov 2024, so a static "heavily short" line would fire almost every day)
    if "fii_idx_long_pct" in W and W["fii_idx_long_pct"].notna().sum() >= 2:
        y = W["fii_idx_long_pct"].dropna()
        med, pct = float(y.median()), float((y < y.iloc[-1]).mean() * 100)
        fig = go.Figure(go.Scatter(x=y.index, y=y, mode="lines", line=dict(color=_FII, width=2), name="FII long %",
                                   hovertemplate="FII long: %{y:.0f}%<extra></extra>"))
        fig.add_hline(y=med, line=dict(color=_MUTED, width=1, dash="dash"),
                      annotation_text=f"median {med:.0f}%", annotation_position="top left", annotation_font_color=_MUTED)
        fig.add_annotation(x=y.index[-1], y=y.iloc[-1], text=f"{y.iloc[-1]:.0f}%", showarrow=False, xanchor="left", xshift=4,
                           font=dict(size=11, color=_INK))
        fig.update_yaxes(range=[0, max(60.0, float(y.max()) + 5)])
        fig.update_layout(showlegend=False)
        st.plotly_chart(_ctx_layout(fig, "FII index futures · long contracts as % of long + short", "%"), use_container_width=True,
                        config={"displayModeBar": False})
        st.caption(f"Today {y.iloc[-1]:.0f}% long: higher than {pct:.0f}% of sessions in the period shown "
                   f"(range {y.min():.0f}–{y.max():.0f}%, median {med:.0f}%). FIIs usually run net short index futures, "
                   "so read the level against its own history and watch the direction: rising = short covering / turning bullish.")
    elif "fii_idx_long_pct" in W:
        st.caption("FII positioning history is being back-filled from NSE's participant files.")

    # 3. FII vs DII net cash: daily bars, daily lines, or cumulative lines (one axis, one zero baseline)
    F = W[[c for c in ("fii_net", "dii_net") if c in W]].dropna(how="all")
    if len(F):
        view = st.radio("FII / DII view", ["Daily bars", "Daily lines", "Cumulative"], index=0, horizontal=True,
                        key="ctx_flow_view", label_visibility="collapsed")
        fig = go.Figure()
        xs = _flow_x_labels(F.index)                 # category axis: one slot per session, no weekend gaps or 00:00 ticks
        n = len(F)
        series = (("fii_net", "FII", _FII), ("dii_net", "DII", _DII))
        if view == "Daily bars":
            for col, name, color in series:
                if col in F:
                    fig.add_trace(go.Bar(x=xs, y=F[col], name=name, marker=dict(color=color, line=dict(width=0)),
                                         hovertemplate=f"{name}: ₹%{{y:,.0f}} cr<extra></extra>"))
            fig.update_layout(barmode="group", bargap=max(0.2, 0.75 - 0.05 * n), bargroupgap=0.08, barcornerradius=4)
            title = "FII vs DII net cash buying · ₹ crore per session"
        else:
            cum = view == "Cumulative"
            mode = "lines+markers" if n < 30 else "lines"   # markers keep a short history visible
            for col, name, color in series:
                if col not in F:
                    continue
                y = F[col].fillna(0).cumsum() if cum else F[col]
                fig.add_trace(go.Scatter(x=xs, y=y, name=name, mode=mode, line=dict(color=color, width=2),
                                         marker=dict(size=8, color=color, line=dict(width=2, color="#ffffff")),
                                         hovertemplate=f"{name}: ₹%{{y:,.0f}} cr<extra></extra>"))
                end_xy = _last_valid_xy(xs, y)
                if end_xy is not None:
                    fig.add_annotation(x=end_xy[0], y=end_xy[1], text=_rs_k(end_xy[1]),
                                       showarrow=False, xanchor="left", xshift=8, font=dict(size=11, color=_INK))
            title = ("FII vs DII cumulative net cash · ₹ crore since " + f"{F.index.min():%d %b %Y}") if cum \
                else "FII vs DII net cash buying · ₹ crore per session"
        fig.add_hline(y=0, line=dict(color=_MUTED, width=1))
        # a spare slot on the right keeps the end labels on screen; tick density capped for phones
        fig.update_xaxes(type="category", range=[-0.5, max(n, 6) - 0.5 + (2.2 if view != "Daily bars" else 0)], nticks=8, tickangle=0)
        st.plotly_chart(_ctx_layout(fig, title, ""), use_container_width=True, config={"displayModeBar": False})
        tot = F.sum()
        st.caption(f"{n} session(s) since NSE flows were first saved ({F.index.min():%d %b %Y}): "
                   + " · ".join(f"{k.split('_')[0].upper()} {_rs(v)} cr" for k, v in tot.items())
                   + ". NSE keeps no archive of this, so the history grows by one session each evening. "
                   "Cumulative shows who has been the net buyer over the period.")
    with st.popover("Table view") if hasattr(st, "popover") else st.container():
        cols = [c for c in ("nifty_close", "nifty_pe", "midcap_pe", "smallcap_pe", "nifty_pb", "nifty_dy", "fii_idx_long_pct", "fii_net", "dii_net") if c in W]
        st.dataframe(W[cols].sort_index(ascending=False).round(2), use_container_width=True, height=280)


def _nse_context():
    """NSE's daily valuation and flows (scripts/nse_daily.py): Nifty P/E, FII / DII cash, FII index-futures positioning."""
    try:
        from utils import nse_data as nd
        H = _ctx_history(_ctx_stamp())
        C = nd.market_context()
    except Exception:
        return
    if (H is None or H.empty) and (C is None or C.empty):
        return
    if H is not None and not H.empty:
        S = H.sort_index()
        asof = S.index.max()
    else:
        S = C.copy()
        S["date"] = pd.to_datetime(S["date"], errors="coerce")
        S = S.dropna(subset=["date"]).sort_values("date").set_index("date")
        asof = S.index.max()
    last2 = lambda k: (pd.to_numeric(S[k], errors="coerce").dropna() if k in S else pd.Series(dtype=float))
    lp_val = lambda k: (float(last2(k).iloc[-1]) if len(last2(k)) else None,
                        float(last2(k).iloc[-2]) if len(last2(k)) >= 2 else None)
    st.markdown(f"**Valuation & flows** · NSE, {pd.Timestamp(asof):%d %b}")
    c = st.columns(5)
    pe, pe_p = lp_val("nifty_pe")
    pb, _ = lp_val("nifty_pb")
    dy, _ = lp_val("nifty_dy")
    if pe is not None:
        c[0].metric("Nifty P/E", f"{pe:.1f}", f"{pe - pe_p:+.2f}" if pe_p is not None else None, delta_color="off",
                    help="Trailing P/E of the Nifty 50 from NSE. Its 10-year range is roughly 18-28")
    if pb is not None:
        c[1].metric("Nifty P/B", f"{pb:.2f}", f"yield {dy or 0:.2f}%", delta_color="off")
    for col, k, lab in ((c[2], "fii_net", "FII cash"), (c[3], "dii_net", "DII cash")):
        v, _ = lp_val(k)
        if v is not None:
            col.metric(lab, f"₹{v:+,.0f} cr", "+ net bought" if v >= 0 else "- net sold", delta_color="normal")
    lp, lp_p = lp_val("fii_idx_long_pct")
    if lp is not None:
        hist = last2("fii_idx_long_pct")
        ctx = (f" Over the last {len(hist)} sessions it ranged {hist.min():.0f}–{hist.max():.0f}% (median {hist.median():.0f}%); "
               f"today is above {(hist < lp).mean() * 100:.0f}% of them.") if len(hist) >= 20 else ""
        c[4].metric("FII index futures", f"{lp:.0f}% long", f"{lp - lp_p:+.1f} pp" if lp_p is not None else None,
                    help="FIIs' long contracts as a share of their long + short index futures. FIIs usually hold more shorts "
                         "than longs, so judge the level against its own history; a rising reading often means short covering." + ctx)
    mid_s, small_s = last2("midcap_pe"), last2("smallcap_pe")
    cap_parts = []
    for lab, s_pe in (("Midcap 150", mid_s), ("Smallcap 250", small_s)):
        if len(s_pe):
            cur = float(s_pe.iloc[-1])
            w1y = s_pe[s_pe.index >= s_pe.index.max() - pd.Timedelta(days=365)]
            p1y = float((w1y < cur).mean() * 100) if len(w1y) >= 20 else None
            cap_parts.append(f"{lab} P/E {cur:.1f}" + (f" ({p1y:.0f}th pctile 1Y)" if p1y is not None else ""))
    if cap_parts:
        st.caption(" · ".join(cap_parts))
    with st.expander("📈 Valuation & flows history (P/E, FII positioning, FII / DII cash)"):
        try:
            _context_charts()
        except Exception as e:
            st.caption(f"History unavailable: {e}")
    try:
        D = nd.deals_for(None, days=3)
    except Exception:
        D = pd.DataFrame()
    if len(D):
        D = D.assign(bucket=D.client.map(nd.deal_kind))
        with st.expander(f"🐋 Bulk & block deals, last 3 sessions ({len(D)})"):
            st.dataframe(D[["date", "symbol", "side", "bucket", "client", "qty", "price", "value_cr", "kind"]], hide_index=True, use_container_width=True,
                         column_config={"date": st.column_config.DateColumn("Date", format="DD MMM"), "price": st.column_config.NumberColumn(format="₹%.2f"),
                                        "qty": st.column_config.NumberColumn("Qty", format="%,d"),
                                        "value_cr": st.column_config.NumberColumn("Value", format="₹%.1f cr")})
            st.caption("A large disclosed trade (over 0.5% of a company's shares). Who is buying matters more than that someone bought: "
                       "promoters and long-only funds mean more than arbitrage desks.")


def _industries(B: pd.DataFrame):
    st.markdown("### 2 · Industries")
    if B.empty:
        st.info("No industry data.")
        return
    c = st.columns(3)
    for col, g in zip(c, ("LEADING", "IMPROVING", "WEAKENING")):
        ns = B[(B.group == g) & ~B.small]
        sm = int(((B.group == g) & B.small).sum())
        hdr = f"**{ib.GROUP_ICON[g]}** ({len(ns)}" + (f" · +{sm} small)" if sm else ")")
        col.markdown(hdr)
        col.markdown("\n".join(f"- {r.industry} · {r.score:.0f}" + (f" ({r.change:+.0f})" if pd.notna(r.change) else "")
                               for r in ns.head(6).itertuples()) or "—")
    st.caption("Score 0–100 and its one-month change. Leading requires score ≥ 60 & ≥ 50% above 50-day; "
               "Improving is up ≥ 10 & ≥ 40% above 50-day. Full table and the stocks inside each: **Sector Pulse**.")


def _val_compact(P: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for r in P.itertuples():
        rat = str(getattr(r, "val_rating", "") or "").strip()
        vp = getattr(r, "val_pct", None)
        if rat and pd.notna(vp):
            out.append(f"{rat} (P{float(vp):.0f})")
        elif rat:
            out.append(rat)
        elif pd.notna(vp):
            out.append(f"P{float(vp):.0f}")
        else:
            out.append("—")
    return out


def _inst_compact(P: pd.DataFrame) -> list[str]:
    icon = {"ACCUMULATING": "🟢", "DISTRIBUTING": "🔴", "NEUTRAL": "⚪"}
    out: list[str] = []
    for r in P.itertuples():
        lab = str(getattr(r, "inst_label", "") or "")
        y = getattr(r, "inst_y", None)
        if not lab:
            out.append("—")
        elif pd.notna(y):
            out.append(f"{icon.get(lab, '')} {float(y):+.1f} pp")
        else:
            out.append(icon.get(lab, "") or "—")
    return out


def _funnel(df: pd.DataFrame, B: pd.DataFrame, R: dict | None = None):
    st.markdown("### 3 · What passes the funnel today")
    if R is None:
        R = funnel.run(fv.get_table(df))
    C = R["counts"]
    st.markdown(" → ".join(f"{r.stage} **{int(r.left)}**" for r in C.itertuples()))
    P = R["picks"]
    if P.empty:
        st.info("Nothing passes every stage today. Near misses are on the Investment Funnel page.")
        return
    old = _prev_pass(fv.snap_key(df), df)
    col_s = lambda c, default=None: P[c] if c in P.columns else pd.Series(default, index=P.index)
    V = pd.DataFrame({
        "": ["🆕" if t not in old else "" for t in P.ticker], "Stock": P.ticker.str.replace(".NS", "", regex=False),
        "Name": P.name, "Industry": P.industry, "Price": P.price,
        "vs Stop": col_s("dist_stop_pct"), "vs 52W High": col_s("dist_52w"),
        "RS": P.comp_rs, "Technicals": fv.setup_labels(P),
        "Stop": P.stop, "Risk %": P.risk_pct, "2× risk ref": P.t1, "Business": P.biz_headline,
        "Inst.": _inst_compact(P), "Valuation": _val_compact(P),
        "Liq ₹cr/d": col_s("liq_cr"), "NSE": col_s("nse_flags", "")})
    st.dataframe(V, hide_index=True, use_container_width=True, height=min(len(V) * 35 + 40, 420), column_config={
        "": st.column_config.TextColumn("", width="small", help="🆕 = did not have both an uptrend and an entry yesterday"),
        "Price": st.column_config.NumberColumn(format="₹%.1f"),
        "vs Stop": st.column_config.NumberColumn(format="%+.1f%%"),
        "vs 52W High": st.column_config.NumberColumn(format="%+.1f%%"),
        "RS": st.column_config.NumberColumn(format="%+.1f"),
        "Technicals": st.column_config.TextColumn("Setup", help="Technicals action, plus recent SIMBA signal or trigger price"),
        "Stop": st.column_config.NumberColumn(format="₹%.1f"),
        "Risk %": st.column_config.NumberColumn(format="%.1f%%", help="Distance from entry (or trigger) to stop"),
        "2× risk ref": st.column_config.NumberColumn(format="₹%.1f", help="A yardstick, not an exit: booking at 2× risk halved returns in the backtest"),
        "Inst.": st.column_config.TextColumn("Inst.", help="FII + DII 1Y change & accumulation status"),
        "Liq ₹cr/d": st.column_config.NumberColumn(format="₹%.1f"),
        "NSE": st.column_config.TextColumn("NSE", help="Surveillance warnings and upcoming corporate events")})
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
                     "NSE": str(r.get("nse_flags") or "") if r is not None and "nse_flags" in r else "",
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


def _nifty_since(dates: pd.Series):
    """Nifty 50 rebased to 100 on the first of `dates` (None when the index series is not loaded)."""
    try:
        n = st.session_state.get("nifty_data")
        if n is None or n.empty:
            return None
        cl = n["Close"].copy()
        if cl.index.tz is not None:
            cl.index = cl.index.tz_localize(None)
        cl.index = cl.index.normalize()
        x = cl[~cl.index.duplicated(keep="last")].reindex(pd.DatetimeIndex(dates), method="ffill")
        return None if x.isna().any() else (x / x.iloc[0] * 100).round(2)
    except Exception:
        return None


def _model_portfolio():
    st.markdown("### 5 · Model portfolio (paper)")
    if not (os.path.exists(BENCH) and os.path.exists(SNAP)):
        st.caption("Benchmark file not built yet; it appears after the next end-of-day run.")
        return
    b = _read_csv_cached(BENCH, os.path.getmtime(BENCH), ("date",))
    try:
        snap = _read_json_cached(SNAP, os.path.getmtime(SNAP))
    except Exception:
        snap = {}
    last = b.iloc[-1]
    since = f"since {b.date.iloc[0]:%d %b}"
    c = st.columns(5)
    # the headline is the same figure as the Trend Scanner card: equity against the starting capital
    start = float((snap.get("config") or {}).get("initial_capital") or 0)
    total = (float(snap["equity"]) / start - 1) * 100 if start and snap.get("equity") else None
    c[0].metric("Model, since inception" if total is not None else "Model", f"{total if total is not None else last.model - 100:+.1f}%",
                f"{last.model - 100:+.1f}% {since}", delta_color="off")
    nifty = _nifty_since(b.date)
    if nifty is not None:
        b["nifty50"] = nifty.values
        c[1].metric("Nifty 50", f"{b.nifty50.iloc[-1] - 100:+.1f}%", f"{last.model - b.nifty50.iloc[-1]:+.1f} pp model vs this")
    elif "top50" in b:
        c[1].metric("50 largest stocks", f"{last.top50 - 100:+.1f}%", f"{last.model - last.top50:+.1f} pp model vs this")
    c[2].metric("Average stock (equal-weight)", f"{last.equal_weight - 100:+.1f}%", f"{last.model - last.equal_weight:+.1f} pp model vs this",
                help="Every stock in the universe, equally weighted: the fair yardstick for a model that buys mid and small caps")
    c[3].metric("Holdings", int(last.holdings))
    c[4].metric("Cash", f"{last.cash_pct:.0f}%", f"avg {b.cash_pct.mean():.0f}%", delta_color="off")
    fig = go.Figure()
    if "nifty50" in b:
        fig.add_scatter(x=b.date, y=b.nifty50, name="Nifty 50", line=dict(color="#E45756", width=1.5, dash="dot"))
    for col, lab, colr in (("model", "Model portfolio", "#54A24B"), ("equal_weight", "Equal-weight market", "#9D9D9D"),
                           ("top50", "50 largest stocks", "#4C78A8")):
        if col in b:
            fig.add_scatter(x=b.date, y=b[col], name=lab, line=dict(color=colr, width=2.5 if col == "model" else 1.5))
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.15),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"The momentum model (OptComp V2.2), paper money, the same equity curve as the card on **Trend Scanner**. The lines "
               f"and the comparisons run {since}, the first day all series exist. Two yardsticks, two answers: against the Nifty 50 "
               "the model is well ahead; against the average stock (which is what it buys from) it is about level.")
    pf = pd.DataFrame(snap.get("portfolio") or [])
    if len(pf):
        with st.expander(f"Holdings ({len(pf)})"):
            keep = [c for c in ("Ticker", "Sector", "Entry", "Price", "PnL%", "Trail_Stop", "Danger") if c in pf.columns]
            st.dataframe(pf[keep], hide_index=True, use_container_width=True)


def _system_pulse(df: pd.DataFrame, B: pd.DataFrame, R: dict) -> None:
    """Compact 4-column one-glance summary strip of market posture, sector leadership, funnel, and holdings."""
    reg = fv.regime(df)
    top_leads = B[(B.group == "LEADING") & ~B.small].head(2)["industry"].tolist() if not B.empty else []
    n_lead = int(((B.group == "LEADING") & ~B.small).sum()) if not B.empty else 0
    n_imp = int(((B.group == "IMPROVING") & ~B.small).sum()) if not B.empty else 0
    picks = R.get("picks", pd.DataFrame())
    near = R.get("near", pd.DataFrame())
    n_buy_now = int(picks["technicals"].eq("BUY NOW").sum()) if len(picks) and "technicals" in picks else 0
    book_lbl, book_sub = "No open book", "Track in Return Tracker"
    try:
        from utils.positions import get_positions_with_pnl
        pos = [p for p in get_positions_with_pnl(df) if p.get("status", "active") == "active"]
        if pos:
            sl_hits = sum(1 for p in pos if p.get("sl_triggered"))
            near_sl = sum(
                1 for p in pos
                if not p.get("sl_triggered") and p.get("current_price") and (p.get("stop_loss_adj") or p.get("stop_loss"))
                and ((p["current_price"] / (p.get("stop_loss_adj") or p.get("stop_loss")) - 1) * 100) < NEAR_STOP
            )
            att = sl_hits + near_sl
            book_lbl = f"{len(pos)} active"
            book_sub = f"{att} need stop review" if att else "Stops clear"
    except Exception:
        pass
    c = st.columns(4)
    c[0].metric("Market Posture", f"{fv.ICON.get(reg['regime'], '⚪')} {reg['regime'].title()}", reg["guidance"], delta_color="off")
    c[1].metric("Sector Breadth", f"{n_lead} Lead · {n_imp} Imp", ", ".join(top_leads) if top_leads else "No leading groups", delta_color="off")
    c[2].metric("Funnel Today", f"{len(picks)} PASS · {n_buy_now} BUY NOW", f"{len(near)} near misses", delta_color="off")
    c[3].metric("Your Book", book_lbl, book_sub, delta_color="off")


def render(df: pd.DataFrame):
    B = get_board(df)
    R = funnel.run(fv.get_table(df))
    _system_pulse(df, B, R)
    _market(df)
    _industries(B)
    _funnel(df, B, R=R)
    _positions(df, B)
    _model_portfolio()
    st.caption("Scanner lists that used to be on Morning Pulse, the Live Trading Desk and Turnaround Radar are in "
               "**🧪 Signal Lab**, with a scorecard of how each has done. The old pages are under the **🗄️ Archive** workspace.")
