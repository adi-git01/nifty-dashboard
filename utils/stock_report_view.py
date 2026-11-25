"""📋 Stock Report page: the one-shot report (utils/stock_report.py)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils import fund_trends, stock_report as sr
from utils import funnel_view as fv
from utils.industry_board_view import get_board


@st.cache_data(ttl=6 * 3600, show_spinner="Scoring SIMBA…")
def _simba(key: str, fund_stamp: tuple, reg_name: str, _df: pd.DataFrame) -> pd.DataFrame:
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


def _simba_fund_stamp() -> tuple:
    import os
    from utils import fund_paths
    files = (fund_paths.QUARTERLY, fund_paths.ANNUAL, fund_paths.EXT_QUARTERLY, fund_paths.EXT_ANNUAL, fund_paths.EXTRA_ANNUAL)
    return tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in (fund_paths.resolve(f) for f in files))


def _bars(t: pd.DataFrame, bar: str, line: str, bar_name: str, line_name: str, title: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = [d.strftime("FY%y") if title.startswith("Year") and d.month == 3 else d.strftime("%b %y") for d in t.index]
    fig.add_bar(x=x, y=t[bar], name=bar_name, marker_color="#4C78A8", opacity=0.75)
    if line in t and t[line].notna().any():
        fig.add_scatter(x=x, y=t[line], name=line_name, mode="lines+markers", line=dict(color="#F58518", width=2), secondary_y=True)
    fig.update_layout(title=title, height=270, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2),
                      template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def _lines(t: pd.DataFrame, cols: dict, title: str) -> go.Figure:
    fig = go.Figure()
    x = [d.strftime("FY%y") if d.month == 3 and title.startswith("Year") else d.strftime("%b %y") for d in t.index]
    for c, lab in cols.items():
        if c in t and t[c].notna().any():
            fig.add_scatter(x=x, y=t[c], name=lab, mode="lines+markers")
    fig.update_layout(title=title, height=270, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2),
                      template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching 14 months of prices and computing signals…")
def _extended_rows(key: str, ticker: str) -> pd.DataFrame:
    """Snapshot row for a stock outside the daily snapshot (utils/extended.py). Cached for the session's data day."""
    from utils import extended
    rows = extended.rows_for([ticker])
    if rows.empty:
        raise LookupError(ticker)               # not cached, so a temporary data failure is retried next time
    return rows


@st.cache_resource(show_spinner=False)
def _ext_val_ctx(stamp: float):
    from utils import valuation as V
    return V.load_context(extended=True)


def _extended_valuation(ticker: str, row: pd.Series, val: pd.DataFrame) -> pd.DataFrame:
    """Value an extended stock on request and return `val` with its row added (best effort)."""
    import os
    from utils import fund_paths, valuation as V
    try:
        paths = [fund_paths.resolve(p) for p in (fund_paths.EXT_ANNUAL, fund_paths.EXT_QUARTERLY, fund_paths.EXTRA_ANNUAL)]
        stamp = max([os.path.getmtime(f) for f in paths if os.path.exists(f)] or [0.0])
        o = V.value_stock(ticker.replace(".NS", ""), _ext_val_ctx(stamp), price=float(row["currentPrice"]),
                          hi52=float(row.get("fiftyTwoWeekHigh")) if pd.notna(row.get("fiftyTwoWeekHigh")) else None,
                          lo52=float(row.get("fiftyTwoWeekLow")) if pd.notna(row.get("fiftyTwoWeekLow")) else None)
        r = V.summary_row(o)
        r["ticker"] = ticker
        return pd.concat([val, pd.DataFrame([r]).set_index("ticker").reindex(columns=val.columns)]) if len(val) else pd.DataFrame([r]).set_index("ticker")
    except Exception:
        return val


@st.cache_data(ttl=3600, show_spinner=False)
def _deal_tally(_mtime: float) -> dict:
    from utils import nse_data
    return nse_data.deal_tally(90)


def _deals_90d() -> dict:
    from utils import nse_data
    p = nse_data.F["deals"]
    return _deal_tally(os.path.getmtime(p) if os.path.exists(p) else 0.0)


def _chart_analysis(t: str):
    """Technicals' structural chart read (forensics.plan, chart-prompt v6) for the selected stock, inline."""
    try:
        from utils import technicals_view as tv
        from utils.forensics import SUPPLY_LABEL, _p, plan
        from utils.forensics_chart import levels_ladder, price_chart
        h = tv._history(t)
        P = plan(h, None, None, 10_000.0) if h is not None and len(h) else None
    except Exception as e:
        st.caption(f"Chart analysis unavailable ({type(e).__name__}).")
        return
    if not P:
        st.caption("Chart analysis needs at least 60 closed sessions.")
        return
    r, cfg = P["r"], {"displayModeBar": False}
    st.markdown("**Chart analysis** (Technicals structural read: event, supply shelves, anchored levels)")
    k = st.columns(4)
    k[0].metric("Action", P["action"])
    k[1].metric("Event", r["state"], f"day {r['days']}" if "days" in r else None, delta_color="off")
    k[2].metric("Supply", SUPPLY_LABEL.get(r["supply"], r["supply"]))
    if P["trigger"]:
        k[3].metric("Trigger (close >)", f"₹{P['trigger']:,.2f}", f"{(P['trigger'] / r['close'] - 1) * 100:+.1f}%", delta_color="off")
    else:
        k[3].metric("Trend (Gate 1)", "PASS" if r["gate1"] else "FAIL", f"{r['dist_ma50']:+.1f}% vs MA50", delta_color="off")
    st.plotly_chart(price_chart(P, t), use_container_width=True, config=cfg)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(levels_ladder(P), use_container_width=True, config=cfg)
    with c2:
        notes = []
        if P.get("trig_a"):
            notes.append(f"**Trigger A** · {P['trig_a']}")
        if P.get("trig_b"):
            notes.append(f"**Trigger B** · {P['trig_b']}")
        sig = r.get("signature")
        if sig:
            notes.append(f"**Pullback** · {sig.lower()}: down days {_p(r.get('dn_vs_med'), 2)}× median volume"
                         + (" ⭐ best backtested cell with GO" if sig == "ABSORPTION" else ""))
        if r.get("shakeout"):
            _sd = pd.to_datetime(r.get("shakeout_date"), errors="coerce")
            _sd_txt = f"{_sd:%d %b}" if pd.notna(_sd) else "recent"
            notes.append(f"**🪤 Shakeout** · {_sd_txt}: broke ₹{_p(r['shakeout_level'])} intraday, "
                         f"closed back above on {_p(r['shakeout_volx'], 2)}× volume")
        if np.isfinite(r.get("stop", np.nan)) and 0 < r["stop"] < r["close"]:
            notes.append(f"**Stop** · ₹{r['stop']:,.2f} ({-r['stop_pct']:.1f}%)" + (" · half size" if r.get("half_size") else ""))
        notes.append(f"**Lens** · RSI {_p(r.get('rsi'), 0)} · ADX {_p(r.get('adx'), 0)} (+DI {_p(r.get('pdi'), 0)} / −DI {_p(r.get('ndi'), 0)})")
        st.markdown("\n\n".join(notes))
        src = h.attrs.get("source", "")
        st.caption((f"{src} to {pd.Timestamp(h.index[-1]):%d %b %Y}. " if src else "")
                   + "Held-position exits, valuation and the full report: **Technicals → Stock report**.")


def render(df: pd.DataFrame):
    from utils import extended
    from utils.nifty1000_list import EXTENDED
    core = df["ticker"].dropna().tolist()
    have = set(core)
    ext = [t for t in EXTENDED if t not in have]
    tickers = core + ext
    names = dict(zip(df["ticker"], df.get("name", df["ticker"]).astype(str)))
    names.update({t: f"{EXTENDED[t]['name']}  · extended" for t in ext})
    explicit = st.session_state.pop("report_ticker", None)
    if explicit in tickers:
        st.session_state["sr_pick"] = explicit
    elif "sr_seeded" not in st.session_state:
        want = st.session_state.get("quick_dive_ticker")
        if want in tickers:
            st.session_state["sr_pick"] = want
    st.session_state["sr_seeded"] = True
    a, b = st.columns([3, 1.4])
    t = a.selectbox("Stock", tickers, key="sr_pick", format_func=lambda x: f"{x.replace('.NS', '')} — {names.get(x, '')}",
                    help=f"Type part of the name or ticker. {len(core):,} ranked stocks plus {len(ext):,} extended ones "
                         f"(market cap ₹{extended.MIN_MCAP_CR:,} cr and above, analysed on request).")
    other = b.text_input("…or any other NSE symbol", key="sr_any", placeholder="e.g. TATACAP",
                         help="For a stock in neither list. Type the NSE symbol and press Enter. Clear this box to go back to the list.")
    if other and other.strip():
        t = extended.norm(other)
    is_ext = t not in have
    val = fv.valuation()
    if is_ext:
        try:
            rows = _extended_rows(fv.snap_key(df), t)
        except Exception:
            rows = pd.DataFrame()
        if rows.empty:
            st.error(f"No price data found for {t.replace('.NS', '')}. Check the NSE symbol; newly listed stocks need about "
                     "eight months of history.")
            return
        rows = rows.copy()
        for c in ("bar_date", "nifty_close", "nifty_prev_close"):
            if c in df.columns and c not in rows.columns:
                rows[c] = df[c].iloc[0]
        df = pd.concat([df, rows.reindex(columns=list(dict.fromkeys(list(df.columns) + ["extended"])))], ignore_index=True)
        val = _extended_valuation(t, rows.iloc[0], val)
        c1, c2 = st.columns([4, 1.3])
        c1.info("🔭 **Extended coverage.** This stock is outside the ranked universe, so its report was computed just now "
                "from 14 months of prices and the weekly fundamentals. It is not ranked against other stocks and does not "
                "appear in the funnel list, industry board or scanners. Add it to your watchlist and it joins the nightly "
                "run, so alerts, exit levels and the daily brief cover it.")
        from utils import positions as _pos
        if _pos.get_position(t) is None:
            if c2.button("👀 Add to watchlist", key="sr_watch", use_container_width=True):
                _pos.add_position(t, name=(EXTENDED.get(t) or {}).get("name") or t.replace(".NS", ""), status="watching",
                                  entry_price=float(rows["currentPrice"].iloc[0]))
                st.toast(f"{t.replace('.NS', '')} added to your watchlist; tracked nightly from the next run.", icon="👀")
                st.rerun()
        else:
            c2.success("In your book")
    reg = fv.regime(df)
    with_simba = "simba_stage" in df.columns and df["simba_stage"].notna().any()
    srow = None
    if with_simba:
        try:
            S = _simba(fv.snap_key(df) + (f"|{t}" if is_ext else ""), _simba_fund_stamp(),
                       fv.simba_regime(df).get("regime", "UNKNOWN"), df)
            srow = S.loc[t].to_dict() if t in S.index else None
        except Exception:
            srow = None
    R = sr.gather(t, df, get_board(df[df["ticker"].isin(have)] if is_ext else df), val, reg, srow)
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
    h[4].metric("Business", f"{tr['verdict'].title()} {tr['score']}" if tr.get("score") is not None else "—",
                tr["direction"].lower() + (" · investing" if tr.get("investing") else "") if tr.get("direction") else None, delta_color="off")
    getattr(st, R["verdict"]["tone"])(f"**{R['verdict']['tag']}** — {R['verdict']['why']}")
    _sr_b1, _sr_b2, _ = st.columns([1.4, 1.3, 2.3])
    _sr_b1.download_button("⬇️ Download this report (Markdown)", sr.markdown(R), file_name=f"{t.replace('.NS', '')}_report.md",
                           mime="text/markdown", key="sr_dl", use_container_width=True)
    if _sr_b2.button("🏛️ Convene Expert Council", key="sr_to_ec", use_container_width=True):
        st.session_state["council_ticker"] = t
        st.session_state["nav_page"] = "🏛️ Expert Council"
        st.rerun()

    st.markdown("#### 1 · Funnel checklist")
    icon = {True: "✅", False: "❌", None: "➖"}
    for c in R["checks"]:
        st.markdown(f"{icon[c['ok']]} **{c['stage']}** — {c['why']}")

    st.markdown("#### 2 · Market and industry")
    g = R["regime"]
    line = (f"Market: Nifty **{g.get('regime', 'UNKNOWN').title()}**, breadth **{g.get('breadth_regime', 'UNKNOWN').title()}**"
            + (f" — {g['guidance'].lower()}" if g.get("guidance") else "") + ".")
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

    _chart_analysis(t)

    st.markdown("#### 4 · Entry plan")
    if pd.notna(p["risk_pct"]):
        m = st.columns(5)
        m[0].metric("Entry", f"₹{p['entry']:,.1f}")
        m[1].metric("Stop", f"₹{p['stop']:,.1f}", f"-{p['risk_pct']:.1f}%", delta_color="inverse")
        m[2].metric("2× risk (reference)", f"₹{f(p['t1'], '{:,.1f}')}", f"{(p['t1'] / p['entry'] - 1) * 100:+.1f}%" if pd.notna(p["t1"]) else None)
        m[3].metric("Target 2", f"₹{f(p['t2'], '{:,.1f}')}" if pd.notna(p["t2"]) else "—")
        m[4].metric(f"Position size", f"{f(p['size_pct'], '{:.0f}')}%", f"of capital, risking {sr.RISK_PCT / (2 if p['half'] else 1):.1f}%" + (" (halved)" if p["half"] else ""), delta_color="off")
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

    st.markdown(f"#### 5 · Business — {fund_trends.headline(tr)}")
    if not tr["findings"]:
        st.info("No fundamentals on file for this stock.")
    else:
        lq = tr["latest"]
        if lq:
            st.markdown(f"**Latest quarter ({lq['quarter']}):** sales ₹{f(lq['sales'], '{:,.0f}')} cr ({f(lq['sales_yoy'], '{:+.0f}')}% YoY), "
                        f"net profit ₹{f(lq['np'], '{:,.0f}')} cr ({f(lq['np_yoy'], '{:+.0f}')}% YoY), net margin {f(lq['npm'])}%"
                        + (f", OPM {f(lq['opm'], '{:.0f}')}%" if pd.notna(lq.get("opm")) else ""))
        if tr.get("investing"):
            st.info(f"🏗️ **Investment phase** — {tr['investing']['text']}. The fall in ROCE, net margin and asset turnover that a "
                    "larger capital base causes is shown but not counted against the business; what to watch is whether returns "
                    "recover as the new assets start earning. A caution from the 2021–25 test: stocks in this state lagged the "
                    "median stock over the following year in each of the five yearly cohorts (about 5 pp on average).")
        cols = st.columns(len(tr["pillars"]) or 1)
        arrow = lambda d: "improving" if d > 0.25 else "slipping" if d < -0.25 else "steady"
        for col, (k, pv) in zip(cols, tr["pillars"].items()):
            col.metric(f"{k} ({pv.get('weight', 0) * 100:.0f}%)", f"{pv['score']}/100", arrow(pv["dir"]), delta_color="off")
            col.progress(int(pv["score"]))
            col.markdown("\n".join(f"- {fund_trends.ICON[x['way']]} **{x['metric']}** — {x['text']}" + ("" if x["scored"] else " *(not scored)*")
                                   for x in tr["findings"] if x["pillar"] == k))
        if tr.get("ownership"):
            st.markdown("**Ownership** (shown for context, not part of the score): " + " · ".join(
                f"{fund_trends.ICON[x['way']]} {x['metric']} {x['text']}" for x in tr["ownership"]))
        try:
            im = fund_trends.inst_map().get(t)
        except Exception:
            im = None
        if im and im.get("label"):
            fq = lambda v: "—" if pd.isna(v) else f"{v:+.1f} pp"
            badge = {"ACCUMULATING": "🟢 **Institutions accumulating**", "DISTRIBUTING": "🔴 **Institutions distributing**",
                     "NEUTRAL": "⚪ **Institutions steady**"}[im["label"]]
            st.markdown(f"{badge} ({im['period']} filing): FII + DII {fq(im['d_inst_q'])} on the quarter "
                        f"(FII {fq(im['d_fii_q'])}, DII {fq(im['d_dii_q'])}) · {fq(im['d_inst_y'])} on the year")
            st.caption("Accumulating = FII + DII up ≥ 1 pp in a quarter or ≥ 2 pp in a year. Nov 2023 – Nov 2026: such stocks beat "
                       "the median stock over the next 6 months in 9 of 9 filing quarters (median +2.9 pp, 56% hit). Distributing was mixed, "
                       "so treat it as context. Only 12 quarters of filings: a tilt, not proof.")
        try:
            from utils import nse_data as _nd
            dl = _deals_90d().get(t)
        except Exception:
            dl = None
        if dl:
            fc = lambda v: f"{v:+,.1f}"
            parts = " · ".join(f"{k} ₹{fc(dl[k])} cr" for k in ("FII", "MF", "Insurance", "Other") if abs(dl[k]) >= 0.05)
            st.markdown(f"**Bulk / block deals** since {dl['since']:%d %b} ({dl['n']} prints, net buy − sell): {parts or 'flat'}")
            with st.expander("Deal prints"):
                D = _nd.deals_for([t], days=90)
                if not D.empty:
                    D = D.assign(bucket=D.client.map(_nd.deal_kind))
                    st.dataframe(D[["date", "kind", "side", "client", "bucket", "qty", "price", "value_cr"]], hide_index=True,
                                 use_container_width=True, column_config={
                                     "date": st.column_config.DateColumn("Date", format="DD MMM YY"),
                                     "value_cr": st.column_config.NumberColumn("₹ cr", format="%.2f"),
                                     "price": st.column_config.NumberColumn("Price", format="%.2f")})
            st.caption("Bucket is read from the client name (FII = foreign names / Singapore, Mauritius, PTE, LLC…; MF = AMCs and trustees; "
                       "Insurance = insurers; Other = brokers, prop desks, HNIs, promoters). The deal file keeps 120 days and fills up day by day.")
        if tr.get("lender_kind"):
            st.caption(f"Scored as a **{tr['lender_kind'].lower()}** on its ROA tree: ROA, spread and leverage are judged against "
                       "break-points for that type of lender (a bank earning 1% on assets is good; a gold-loan company should earn 3.5%+).")
        st.caption(("Score = growth 25%, spread and costs 25%, ROA and ROE 30%, leverage and asset quality 20%; " if tr.get("lender_kind")
                    else "Score = growth 30%, margins 25%, cash and balance sheet 25%, returns on capital 20%; ") + "each mixes the level with "
                   "its direction. Strong 70+, Healthy 55+, Mixed 40+, Weak below. The weights are judgement. Tested on 2021–25 the score did not predict "
                   "the next year's return (it worked in 2024–25 and was inverted in 2021–23): read it as business quality, not a buy signal.")
        A, Q = R["annual"].tail(8), R["quarterly"].tail(12)
        RT = R.get("roa_tree")
        if RT is not None and not RT.empty:
            st.markdown(f"**ROA tree** — every line as % of average assets ({tr.get('lender_kind', 'lender').title()})")
            st.dataframe(sr.roa_tree_view(RT), use_container_width=True)
            a, b = st.columns(2)
            a.plotly_chart(_lines(RT.tail(8), {"nii": "Net interest income", "costs": "Costs (opex + provisions)", "other_income": "Other income",
                                               "roa": "ROA"}, "Yearly ROA tree, % of assets"), use_container_width=True)
            b.plotly_chart(_bars(RT.tail(8), "roe", "leverage", "ROE %", "Leverage x", "Yearly ROE and leverage"), use_container_width=True)
            st.caption("Net interest income + other income − costs − tax = ROA; ROA × leverage = ROE. Screener reports operating "
                       "expenses and provisions as one line, so they are one line here.")
        elif not A.empty:
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
