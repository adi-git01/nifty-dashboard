"""📋 Stock Report page: the one-shot report (utils/stock_report.py)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils import fund_trends, stock_report as sr
from utils import funnel_view as fv
from utils.industry_board_view import get_board


@st.cache_data(ttl=6 * 3600, show_spinner="Scoring SIMBA…")
def _simba(key: str, reg_name: str, _df: pd.DataFrame) -> pd.DataFrame:
    from utils import simba
    from utils.simba_view import _fundamentals
    try:
        ind = simba.industry_stages(_df)
    except Exception:
        ind = {}
    S = simba.score_table(_df, {"regime": reg_name}, _fundamentals(), ind)
    keep = [c for c in ("ticker", "score", "grade", "gates", "rs_rating", "industry_stage", "pts_topdown", "pts_ghoomega",
                        "pts_machayega", "pts_quality", "pts_float", "pts_mtf") if c in S.columns]
    return S[keep].set_index("ticker") if len(S) else pd.DataFrame()


def _bars(t: pd.DataFrame, bar: str, line: str, bar_name: str, line_name: str, title: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = [d.strftime("FY%y") if title.startswith("Year") and d.month == 3 else d.strftime("%b %y") for d in t.index]
    fig.add_bar(x=x, y=t[bar], name=bar_name, marker_color="#4C78A8", opacity=0.75)
    if line in t and t[line].notna().any():
        fig.add_scatter(x=x, y=t[line], name=line_name, mode="lines+markers", line=dict(color="#F58518", width=2), secondary_y=True)
    fig.update_layout(title=title, height=270, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2),
                      template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def _lines(t: pd.DataFrame, cols: dict, title: str) -> go.Figure:
    fig = go.Figure()
    x = [d.strftime("FY%y") if d.month == 3 and title.startswith("Year") else d.strftime("%b %y") for d in t.index]
    for c, lab in cols.items():
        if c in t and t[c].notna().any():
            fig.add_scatter(x=x, y=t[c], name=lab, mode="lines+markers")
    fig.update_layout(title=title, height=270, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2),
                      template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def render(df: pd.DataFrame):
    tickers = df["ticker"].dropna().tolist()
    names = dict(zip(df["ticker"], df.get("name", df["ticker"]).astype(str)))
    want = st.session_state.pop("report_ticker", None) or st.session_state.get("quick_dive_ticker")
    if want in tickers and st.session_state.get("sr_pick") != want and "sr_seeded" not in st.session_state:
        st.session_state["sr_pick"] = want
    st.session_state["sr_seeded"] = True
    t = st.selectbox("Stock", tickers, key="sr_pick", format_func=lambda x: f"{x.replace('.NS', '')} — {names.get(x, '')}",
                     help="Type part of the name or ticker.")
    reg = fv.regime(df)
    with_simba = "simba_stage" in df.columns and df["simba_stage"].notna().any()
    srow = None
    if with_simba:
        try:
            S = _simba(fv.snap_key(df), reg.get("regime", "UNKNOWN"), df)
            srow = S.loc[t].to_dict() if t in S.index else None
        except Exception:
            srow = None
    R = sr.gather(t, df, get_board(df), fv.valuation(), reg, srow)
    if not R["ok"]:
        st.error(R["error"])
        return
    p, tech, tr, v, ind = R["plan"], R["tech"], R["trends"], R["val"], R["ind"]
    f = sr._f

    h = st.columns([2.4, 1, 1, 1, 1.4])
    h[0].markdown(f"### {R['name']}\n{R['industry'] or R['sector']}")
    h[1].metric("Price", f"₹{R['price']:,.1f}", f"{f(tech['ret']['1m'], '{:+.1f}')}% 1M")
    h[2].metric("RS vs Nifty", f(tech["comp_rs"], "{:+.1f}"))
    h[3].metric("SIMBA", f"{f(srow.get('score'), '{:.0f}')} · {srow.get('grade', '—')}" if srow else "—")
    h[4].metric("Business", tr["verdict"].title())
    getattr(st, R["verdict"]["tone"])(f"**{R['verdict']['tag']}** — {R['verdict']['why']}")
    st.download_button("⬇️ Download this report (Markdown)", sr.markdown(R), file_name=f"{t.replace('.NS', '')}_report.md",
                       mime="text/markdown", key="sr_dl")

    st.markdown("#### 1 · Funnel checklist")
    icon = {True: "✅", False: "❌", None: "➖"}
    for c in R["checks"]:
        st.markdown(f"{icon[c['ok']]} **{c['stage']}** — {c['why']}")

    st.markdown("#### 2 · Market and industry")
    g = R["regime"]
    line = f"Market regime **{g.get('regime', 'UNKNOWN').title()}**" + (f", risk scale {g['risk_scale']}/10" if g.get("risk_scale") else "") + "."
    if ind:
        line += (f" **{ind['industry']}** ranks **{ind['rank']} of {ind['of']}** industries (score {f(ind['score'], '{:.0f}')}, "
                 f"{f(ind['change'], '{:+.0f}')} in a month) — {ind['group'].title()}, {ind['status'].title()}.")
        st.markdown(line)
        m = st.columns(5)
        m[0].metric("Above 50-day", f"{f(ind['above_50'], '{:.0f}')}%")
        m[1].metric("Above 200-day", f"{f(ind['above_200'], '{:.0f}')}%")
        m[2].metric("Near 52w high", f"{f(ind['near_high'], '{:.0f}')}%")
        m[3].metric("3M vs Nifty (median)", f"{f(ind['rs_3m'], '{:+.1f}')} pp")
        m[4].metric("Stocks", int(ind["members"]))
        st.caption(f"Strongest in the industry: {ind['leaders'] or '—'}")
    else:
        st.markdown(line + " No industry classification for this stock.")

    st.markdown("#### 3 · Trend and chart")
    m = st.columns(6)
    for col, (k, x) in zip(m, tech["ret"].items()):
        col.metric(k.upper(), f"{f(x, '{:+.1f}')}%")
    m[5].metric("From 52w high", f"{f(tech['dist_52w'], '{:.0f}')}%")
    stage = {1: "1 · base", 2: "2 · advancing", 3: "3 · topping", 4: "4 · declining"}.get(
        int(tech["stage"]) if pd.notna(tech["stage"]) else 0, "—")
    st.markdown(
        f"- **Chart stage** {stage}; price vs 50-day ₹{f(tech['ma50'], '{:,.0f}')} / 200-day ₹{f(tech['ma200'], '{:,.0f}')}\n"
        f"- **Trend Scanner** {tech['scanner']} · trend score {f(tech['trend_score'], '{:.0f}')} · 3-month RS {f(tech['rs_3m'], '{:+.1f}')}\n"
        f"- **Technicals** {p['action']} (state: {p['state']}; overhead supply: {p['supply']}) · RSI {f(tech['rsi'], '{:.0f}')} · ADX {f(tech['adx'], '{:.0f}')}\n"
        f"- **SIMBA** Ghoomega {tech['g_bits'].count('1')}/7 · Machayega {tech['m_bits'].count('1')}/5 · timeframes aligned {f(tech['mtf'], '{:.0f}')}/3"
        + (f" · caution level {f(tech['caution'], '{:.0f}')}" if pd.notna(tech["caution"]) and tech["caution"] > 0 else "")
        + (f"\n- **SIMBA score** {f(srow.get('score'), '{:.0f}')}/100, grade {srow.get('grade')}" + (f" — gates hit: {srow['gates']}" if srow.get("gates") else " — no gates hit") if srow else ""))

    st.markdown("#### 4 · Entry plan")
    if pd.notna(p["risk_pct"]):
        m = st.columns(5)
        m[0].metric("Entry", f"₹{p['entry']:,.1f}")
        m[1].metric("Stop", f"₹{p['stop']:,.1f}", f"-{p['risk_pct']:.1f}%", delta_color="inverse")
        m[2].metric("Target 1", f"₹{f(p['t1'], '{:,.1f}')}", f"{(p['t1'] / p['entry'] - 1) * 100:+.1f}%" if pd.notna(p["t1"]) else None)
        m[3].metric("Target 2", f"₹{f(p['t2'], '{:,.1f}')}" if pd.notna(p["t2"]) else "—")
        m[4].metric(f"Position size", f"{f(p['size_pct'], '{:.0f}')}%", f"of capital, risking {sr.RISK_PCT:.0f}%", delta_color="off")
        notes = []
        if p["risk_pct"] > 15:
            notes.append("the stop is more than 15% away — wait for a tighter setup or take a small position")
        if p["half"]:
            notes.append("Technicals flags half size")
        if p["action"] not in ("BUY NOW", "BUY ON TRIGGER"):
            notes.append(f"Technicals says {p['action']}: the levels are for reference, not a signal")
        st.caption("Stop and targets from the last close (Technicals). " + ("Note: " + "; ".join(notes) + "." if notes else "")
                   + " For the chart, anchors and held-position exits open **Technicals → Stock report**.")
    else:
        st.info("No valid stop below the price in today's data, so no entry plan.")

    st.markdown(f"#### 5 · Business trends — {tr['verdict'].title()} ({tr['up']} improving, {tr['down']} worsening)")
    if not tr["findings"]:
        st.info("No fundamentals on file for this stock.")
    else:
        lq = tr["latest"]
        if lq:
            st.markdown(f"**Latest quarter ({lq['quarter']}):** sales ₹{f(lq['sales'], '{:,.0f}')} cr ({f(lq['sales_yoy'], '{:+.0f}')}% YoY), "
                        f"net profit ₹{f(lq['np'], '{:,.0f}')} cr ({f(lq['np_yoy'], '{:+.0f}')}% YoY), net margin {f(lq['npm'])}%"
                        + (f", OPM {f(lq['opm'], '{:.0f}')}%" if pd.notna(lq.get("opm")) else ""))
        a, b = st.columns(2)
        half = (len(tr["findings"]) + 1) // 2
        for col, part in ((a, tr["findings"][:half]), (b, tr["findings"][half:])):
            col.markdown("\n".join(f"- {fund_trends.ICON[x['way']]} **{x['metric']}** — {x['text']}" for x in part))
        A, Q = R["annual"].tail(8), R["quarterly"].tail(12)
        if not A.empty:
            a, b = st.columns(2)
            a.plotly_chart(_bars(A, "sales", "opm", "Sales ₹cr", "OPM %", "Yearly sales and operating margin"), use_container_width=True)
            b.plotly_chart(_lines(A, {"roce": "ROCE %", "roe": "ROE %", "cash_conv_3y": "Cash conversion (3-yr) %"},
                                  "Yearly returns and cash conversion"), use_container_width=True)
            if any(c in A and A[c].notna().any() for c in ("ccc_days", "inventory_days", "debtor_days")):
                st.plotly_chart(_lines(A, {"inventory_days": "Inventory days", "debtor_days": "Debtor days", "payable_days": "Payable days",
                                           "ccc_days": "Cash conversion cycle"}, "Yearly working-capital days"), use_container_width=True)
        if not Q.empty:
            a, b = st.columns(2)
            a.plotly_chart(_bars(Q, "sales", "opm" if Q["opm"].notna().any() else "npm", "Sales ₹cr",
                                 "OPM %" if Q["opm"].notna().any() else "Net margin %", "Quarterly sales and margin"), use_container_width=True)
            b.plotly_chart(_lines(Q, {"promoters": "Promoters %", "fii": "FII %", "dii": "DII %"}, "Ownership by quarter"),
                           use_container_width=True)
        with st.expander("Yearly and quarterly tables"):
            st.dataframe(sr.annual_view(R["annual"]), use_container_width=True)
            st.dataframe(sr.quarterly_view(R["quarterly"]), use_container_width=True)
        missing = [x for x, ok in (("inventory / debtor / working-capital days", tr["has_wc_days"] or tr["financial"]),
                                   ("quarterly operating margin", tr["has_q_opm"])) if not ok]
        if missing:
            st.caption("Not on file yet: " + " and ".join(missing) + ". The weekly fundamentals refresh (Sunday) now collects them; "
                       "they appear here after its next run.")

    st.markdown("#### 6 · Valuation")
    if v and v.get("status") == "OK":
        m = st.columns(5)
        m[0].metric("Low (P10)", f"₹{f(v['p10'], '{:,.0f}')}")
        m[1].metric("Mid (P50)", f"₹{f(v['p50'], '{:,.0f}')}", f"{(float(v['p50']) / R['price'] - 1) * 100:+.0f}% vs price")
        m[2].metric("High (P90)", f"₹{f(v['p90'], '{:,.0f}')}")
        m[3].metric("Price percentile", f(v["cmp_pct"], "{:.0f}"))
        m[4].metric("Rating", str(v.get("rating") or "—").split(" (")[0])
        st.caption(f"Confidence {v.get('confidence')}; quality {v.get('q_grade')}. {v.get('q_note') or ''} "
                   f"Valued {str(v.get('as_of'))[:10]}. Full workings: **Technicals → Valuation**.")
    else:
        st.info(f"The valuation model does not cover this stock ({(v or {}).get('status', 'no row')}"
                + (f": {v['reason']}" if v and isinstance(v.get("reason"), str) and v["reason"] else "") + ").")
    st.caption("Built from the stored daily snapshot, weekly Screener files and the valuation snapshot. "
               "The funnel is a checklist, not a backtested system. Not investment advice.")
