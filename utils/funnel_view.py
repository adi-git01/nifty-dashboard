"""Investment Funnel page and the shared cached inputs (board, valuation, regime, funnel table)."""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from utils import funnel, simba
from utils.industry_board_view import get_board
from utils.ui_search import search_box

VAL_FILE = "data/valuation_snapshot.csv"


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
                engine_status=rb.engine_status(info["engine"]), risk=weak, guidance=GUIDE[weak], info=info)


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
    from utils import fund_paths, fund_trends, nse_data as _nd
    paths = [fund_paths.resolve(f) for f in fund_trends.FILES + fund_trends.QFILES] + [fund_trends.BRIEF_CACHE, _nd.F["deals"]]
    fk = tuple(os.path.getmtime(p) if os.path.exists(p) else 0.0 for p in paths)
    return _table(snap_key(df), fk, os.path.getmtime(VAL_FILE) if os.path.exists(VAL_FILE) else 0.0, df)


def setup_labels(P: pd.DataFrame) -> list[str]:
    """Human-readable setup label per row (surfaces SIMBA-only setups and trigger prices)."""
    out: list[str] = []
    for r in P.itertuples():
        tech = str(getattr(r, "technicals", "") or "—")
        sd = getattr(r, "simba_days", None)
        ent = getattr(r, "entry", None)
        px = getattr(r, "price", None)
        if tech not in ("BUY NOW", "BUY ON TRIGGER") and pd.notna(sd) and 0 <= float(sd) <= funnel.SETUP_DAYS:
            out.append(f"{tech} · SIMBA ({int(sd)}d)")
        elif tech == "BUY ON TRIGGER" and pd.notna(ent) and pd.notna(px) and float(ent) > float(px):
            out.append(f"BUY ON TRIGGER (₹{float(ent):,.1f})")
        else:
            out.append(tech)
    return out


def _ind_labels(P: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for r in P.itertuples():
        grp = str(getattr(r, "ind_group", "") or "").strip().title()
        sc = getattr(r, "ind_score", None)
        if grp and pd.notna(sc):
            out.append(f"{grp} {float(sc):.0f}")
        else:
            out.append(grp or "—")
    return out


def _supply_labels(P: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for r in P.itertuples():
        sup = str(getattr(r, "fx_supply", "") or "").strip()
        sig = str(getattr(r, "fx_signature", "") or "").strip()
        shk = getattr(r, "fx_shakeout", False)
        bits = [sup] if sup else []
        if sig == "ABSORPTION":
            bits.append("⭐ ABSORPTION")
        if shk is True or str(shk).strip().lower() in ("true", "1"):
            bits.append("🪤 SHAKEOUT")
        out.append(" · ".join(bits) if bits else "—")
    return out


def _trend_labels(P: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for r in P.itertuples():
        stg = getattr(r, "stage", None)
        mtf = getattr(r, "simba_mtf", None)
        adx = getattr(r, "fx_adx", None)
        bits: list[str] = []
        if pd.notna(stg) or pd.notna(mtf):
            s_txt = f"S{int(float(stg))}" if pd.notna(stg) else "S?"
            m_txt = f"M{int(float(mtf))}/3" if pd.notna(mtf) else ""
            bits.append(f"{s_txt}{('·' + m_txt) if m_txt else ''}")
        if pd.notna(adx):
            bits.append(f"ADX {float(adx):.0f}")
        out.append(" · ".join(bits) if bits else "—")
    return out


def _inst_labels(P: pd.DataFrame) -> list[str]:
    icon = {"ACCUMULATING": "🟢", "DISTRIBUTING": "🔴", "NEUTRAL": "⚪"}
    out: list[str] = []
    for r in P.itertuples():
        lab = str(getattr(r, "inst_label", "") or "")
        q, y = getattr(r, "inst_q", None), getattr(r, "inst_y", None)
        if not lab:
            out.append("—")
            continue
        bits = [f"{float(q):+.1f} QoQ" if pd.notna(q) else "", f"{float(y):+.1f} YoY" if pd.notna(y) else ""]
        out.append(f"{icon.get(lab, '')} " + " · ".join(b for b in bits if b))
    return out


def _deliv_labels(P: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for r in P.itertuples():
        dp = getattr(r, "deliv_pct", None)
        da = getattr(r, "deliv_avg20", None)
        if pd.isna(dp):
            out.append("—")
        elif pd.notna(da) and float(da) > 0:
            out.append(f"{float(dp):.0f}% ({float(dp) / float(da):.1f}×)")
        else:
            out.append(f"{float(dp):.0f}%")
    return out


def _show(P: pd.DataFrame, key: str, extra: tuple = ()):
    P = search_box(P, key, cols=("name", "ticker", "industry"))
    if P is None or P.empty:
        st.caption("Nothing here.")
        return
    c1, c2 = st.columns([1.8, 1.4])
    if "inst_label" in P.columns and c1.toggle(
            "🏦 Institutions accumulating only", key=f"{key}_inst",
            help="FII + DII holding up ≥ 1 pp last quarter or ≥ 2 pp over a year (quarterly filings). "
                 "Nov 2023 - Nov 2026: beat the median stock over 6 months in 9 of 9 quarters (median +2.9 pp). Context, not a gate."):
        P = P[P["inst_label"].eq("ACCUMULATING")]
    view_mode = c2.radio("Columns", ["⚡ Quick Glance", "🔬 Full (all pillars)"], horizontal=True,
                         key=f"{key}_cols", label_visibility="collapsed")
    if P is None or P.empty:
        st.caption("Nothing here.")
        return
    col_s = lambda c, default=None: P[c] if c in P.columns else pd.Series(default, index=P.index)
    V = pd.DataFrame({
        "Stock": P.ticker.str.replace(".NS", "", regex=False), "Name": P.name, **{c: P[c] for c in extra},
        "Score": col_s("funnel_score"), "Grade": col_s("funnel_grade", ""),
        "Industry": P.industry, "Ind.": _ind_labels(P), "Price": P.price,
        "vs Stop": col_s("dist_stop_pct"), "vs MA50": col_s("dist_ma50"), "vs 52W High": col_s("dist_52w"),
        "Scanner": col_s("scanner", ""), "RS": P.comp_rs, "RS 3M": col_s("rs_3m"),
        "Technicals": setup_labels(P), "Supply": _supply_labels(P),
        "Stop": P.stop, "Risk %": P.risk_pct, "2× risk ref": P.t1, "Trend": _trend_labels(P),
        "Business": P.biz_headline, "Q Sales YoY": col_s("sales_yoy"),
        "Latest qtr profit YoY": P.np_yoy, "OPM Δ": col_s("opm_chg"), "Flags": P.biz_flags,
        "P/E": col_s("pe"), "Valuation": col_s("val_rating", ""),
        "Value pctile": P.val_pct, "To mid value": P.val_upside,
        "Institutions": _inst_labels(P), "Inst deals": col_s("deal_inst_cr"),
        "Deliv %": _deliv_labels(P), "Liq ₹cr/d": col_s("liq_cr"), "NSE Flags": col_s("nse_flags", ""),
        "Score Split": col_s("score_note", "")})
    if view_mode == "⚡ Quick Glance":
        q_cols = [c for c in (
            "Stock", "Name", *extra, "Score", "Grade", "Industry", "Ind.", "Price", "vs Stop", "vs 52W High",
            "RS", "Technicals", "Risk %", "Business", "Institutions", "Valuation", "NSE Flags",
        ) if c in V.columns]
        V = V[q_cols]
    st.dataframe(V, hide_index=True, use_container_width=True, height=min(len(V) * 35 + 40, 560), column_config={
        "Missed": st.column_config.TextColumn("Missed stage", help="The one stage this stock fails"),
        "Gap to pass": st.column_config.TextColumn("Gap to pass", help="How close this near miss is to flipping into a full PASS"),
        "Score": st.column_config.ProgressColumn(
            "Score", min_value=0, max_value=100, format="%.0f",
            help="Base (0–80): Mom(28) · Set/Extension(22) · Str ADX+MTF(12) · Business/Val-tail/Hygiene(18) + Alpha Catalysts (+0..22: "
                 "Absorption +13, Shakeout +9, Inst Accum +8, Deals/Deliv). Half-size stops (3–15%) unpenalized; noise stops (<2%), "
                 "loss-makers, quarterly drops, extreme valuation tails, extension, and ⚠️ surveillance penalized; ranking aid (not backtested as a composite)."),
        "Grade": st.column_config.TextColumn("Grade", width="small", help="A ≥ 74 · B 60–73 · C < 60"),
        "Score Split": st.column_config.TextColumn("Score Split", help="Points by pillar: Mom(28) · Set(22) · Str(12) · Qual(18) · Cat(+22)"),
        "Price": st.column_config.NumberColumn(format="₹%.1f"),
        "vs Stop": st.column_config.NumberColumn(format="%+.1f%%", help="(Price − Stop) / Price %"),
        "vs MA50": st.column_config.NumberColumn(format="%+.1f%%", help="(Price / 50-DMA − 1) %"),
        "vs 52W High": st.column_config.NumberColumn(format="%+.1f%%", help="Distance from 52-week high %"),
        "Scanner": st.column_config.TextColumn("Scanner", help="Trend Scanner signal (BUY / HOLD / AVOID)"),
        "RS": st.column_config.NumberColumn(format="%+.1f", help="Composite RS vs Nifty"),
        "RS 3M": st.column_config.NumberColumn(format="%+.1f", help="3-month relative strength vs Nifty"),
        "Technicals": st.column_config.TextColumn("Setup", help="Technicals action, plus recent SIMBA signal or trigger price"),
        "Supply": st.column_config.TextColumn("Supply & Vol", help="Overhead supply zone + absorption / shakeout signatures"),
        "Stop": st.column_config.NumberColumn(format="₹%.1f", help="Technicals stop from the last close"),
        "Risk %": st.column_config.NumberColumn(format="%.1f%%", help="Distance from entry (or trigger) to stop"),
        "2× risk ref": st.column_config.NumberColumn(format="₹%.1f", help="A yardstick, not an exit: booking at 2× risk halved returns in the backtest"),
        "Trend": st.column_config.TextColumn("Stage · MTF · ADX", help="SIMBA Stage, Multi-Timeframe score (/3), and ADX trend strength"),
        "Q Sales YoY": st.column_config.NumberColumn(format="%+.0f%%", help="Latest quarter revenue YoY %"),
        "Latest qtr profit YoY": st.column_config.NumberColumn(format="%+.0f%%"),
        "OPM Δ": st.column_config.NumberColumn(format="%+.1f pp", help="Latest quarter operating margin YoY change (percentage points)"),
        "P/E": st.column_config.NumberColumn(format="%.1f×", help="Trailing P/E"),
        "Valuation": st.column_config.TextColumn("Val Rating", help="Valuation v4.1 action rating"),
        "Value pctile": st.column_config.NumberColumn(format="%.0f", help="Where the price sits in the valuation range (0 cheap, 100 expensive). Blank = not valued."),
        "To mid value": st.column_config.NumberColumn(format="%+.0f%%", help="Mid (P50) value vs price"),
        "Institutions": st.column_config.TextColumn("FII+DII Δ pp", help="Change in FII + DII holding from quarterly filings. "
                                                    "🟢 accumulating (≥ +1 pp QoQ without heavy YoY exit, or ≥ +2 pp YoY without current selling) · 🔴 distributing · ⚪ neutral"),
        "Inst deals": st.column_config.NumberColumn("Inst deals ₹cr", format="%+.1f", help="Net FII + MF + insurer buying in NSE bulk/block "
                                                   "deals over 90 days (₹ crore; blank = no deals). Bucketed from client names."),
        "Deliv %": st.column_config.TextColumn("Deliv % (vs 20d)", help="Latest session delivery % and ratio vs 20-day average"),
        "Liq ₹cr/d": st.column_config.NumberColumn("Liq ₹cr/d", format="₹%.1f", help="Average daily turnover (₹ crore/day)"),
        "NSE Flags": st.column_config.TextColumn("NSE Flags", help="Surveillance (ASM/GSM) and upcoming corporate events")})


def render(df: pd.DataFrame):
    from utils import regime_badge as rb
    reg = regime(df)
    c = st.columns([1.2, 1.2, 1.4, 2.0])
    c[0].metric("0 · Nifty regime", f"{ICON.get(reg['regime'], '⚪')} {reg['regime'].title()}")
    c[1].metric("Breadth regime", f"{ICON.get(reg['breadth_regime'], '⚪')} {reg['breadth_regime'].title()}",
                f"{reg['breadth']:.0f}% above 200-day" if reg.get("breadth") is not None else None, delta_color="off")
    eng = reg["info"]["engine"]
    trail_lbl = f"{eng['trail_stop'] * 100:.0f}% trail" if eng.get("trail_stop") else None
    c[2].metric("Model rules", rb.engine_short(eng), trail_lbl, delta_color="off")
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

    n_a = int(R["picks"]["funnel_grade"].eq("A").sum()) if len(R["picks"]) and "funnel_grade" in R["picks"] else 0
    st.markdown(f"#### ✅ Passing every selected stage ({len(R['picks'])}" + (f" · {n_a} Grade A" if n_a else "") + ")")
    st.caption("Sorted by **Funnel Score (0–100)** — Base(80): Mom+Ind(28), Setup/Extension(22), ADX+MTF Structure(12), Business & Hygiene(18) + Catalysts(+0..22: Absorption, Shakeout, Inst Accum); ranking aid, not backtested as a composite. Open **Analysis Lab → 📋 Stock Report** for full details.")
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
        st.caption("Sorted by Funnel Score; **Gap to pass** shows how close each name is to flipping into a full PASS.")
        N = R["near"]
        if len(N):
            ren = {"missed": "Missed", "miss_gap": "Gap to pass"}
            extra_cols = ("Missed", "Gap to pass") if "miss_gap" in N.columns else ("Missed",)
            _show(N.rename(columns=ren), "fn_nq", extra=extra_cols)

    cand_pool = list(dict.fromkeys(pk + R["near"].ticker.tolist() + T.ticker.dropna().tolist()))
    if len(cand_pool) >= 2:
        with st.expander("⚖️ Compare 2–4 candidates head-to-head (All 5 Pillars + Advantage)", expanded=False):
            from utils import stock_compare
            default_cmp = pk[:2] if len(pk) >= 2 else cand_pool[:2]
            c_sel, c_btn = st.columns([3.5, 1.2])
            sel_cmp = c_sel.multiselect(
                "Select 2 to 4 stocks to compare",
                options=cand_pool,
                default=default_cmp,
                max_selections=4,
                format_func=lambda x: x.replace(".NS", ""),
                key="fn_cmp_sel",
            )
            if c_btn.button("⚖️ Open in Compare Stocks", key="fn_cmp_go", use_container_width=True) and len(sel_cmp) >= 2:
                st.session_state["compare_tickers"] = sel_cmp
                st.session_state["cmp_multi_tickers"] = list(sel_cmp)   # the Compare page's widget keeps its own state
                st.session_state["nav_page"] = "⚖️ Compare Stocks"
                st.rerun()
            if len(sel_cmp) >= 2:
                cmp_res = stock_compare.build_comparison(sel_cmp, df, board=get_board(df), val=valuation())
                sw, gw = cmp_res.get("swing_winner", {}), cmp_res.get("garp_winner", {})
                w1, w2 = st.columns(2)
                if sw.get("bare"):
                    w1.info(f"**🏆 3–6m Swing / Momentum Pick: `{sw['bare']}`**  \n{sw.get('why', '')}")
                if gw.get("bare"):
                    w2.success(f"**🛡️ 2–3y Core / GARP Pick: `{gw['bare']}`**  \n{gw.get('why', '')}")
                st.dataframe(cmp_res["matrix"], hide_index=True, use_container_width=True, height=390)
            else:
                st.caption("Pick at least 2 stocks above to view the side-by-side Decision Matrix.")
