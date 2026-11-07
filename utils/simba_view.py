"""Trend Scanner -> SIMBA tab (logic in utils/simba.py)."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from utils import simba

FUND_FILES = ("data/fundamentals_quarterly.csv.gz", "data/fundamentals_annual.csv.gz")
STAGE_ICON = {1: "1 · base", 2: "2 · advancing", 3: "3 · top", 4: "4 · declining"}
CAUTION = {0: "—", 1: "🟡 L1 watch", 2: "🟠 L2 trim", 3: "🔴 L3 exit"}
IND_ICON = {"LEADING": "🟢 Leading", "IMPROVING": "🟡 Improving", "NEUTRAL": "⚪ Neutral", "STAGE 4": "🔴 Stage 4"}
REG_ICON = {"WASHOUT": "🟣", "BEAR CONTRACTION": "🔴", "CORRECTION": "🟠", "NORMAL BULL": "🟢", "FROTH": "🟡",
            "UNKNOWN": "⚪"}
G_LABELS = ["G1 was dormant (150-DMA flat ~5 months)", "G2 150-DMA curving up", "G3 close > 150-DMA",
            "G4 50-DMA > 150-DMA and rising", "G5 higher low (no new low in 40 bars)",
            "G6 Mansfield RS > 0 and rising", "G7 not extended (≤ 25% above 150-DMA)"]
M_LABELS = ["M1 close > 50-day pivot high", "M2 tight base (15-bar range ≤ 15%)", "M3 volume ≥ 2× 50-day average",
            "M4 range ≥ 1.2× ATR", "M5 close in top 30% of range, up day"]


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _fund(key: tuple) -> pd.DataFrame:
    return simba.fundamentals()


def _fundamentals() -> pd.DataFrame:
    return _fund(tuple(os.path.getmtime(f) if os.path.exists(f) else 0 for f in FUND_FILES))


def _held() -> set:
    try:
        with open("positions.json") as f:
            pos = json.load(f)
        items = pos if isinstance(pos, list) else pos.get("positions", [])
        return {p.get("ticker") for p in items if p.get("status") == "active"}
    except Exception:
        return set()


def _ok(b) -> str:
    return "✅" if b else "❌"


def render(df: pd.DataFrame) -> None:
    st.markdown("### 🦁 SIMBA — Ghoomega · Machayega · Bachayega")
    st.caption(
        "A rule-based reconstruction of Kumar Saurabh's (Scientific Investing) Techno-Funda process. "
        "The original SIMBA engine and GMB indicators are closed-source; every threshold here is a "
        "reconstruction (desktop-tutorial / trading-system). Our own studies disagree with two parts: "
        "early turns lagged confirmed uptrends, and expensive momentum beat 'no valuation froth'. "
        "This is SIMBA as written — not the live engine.")

    # ---------------------------------------------------------------- top-down
    px = pd.to_numeric(df.get("currentPrice", df.get("price")), errors="coerce")
    m200 = pd.to_numeric(df.get("twoHundredDayAverage"), errors="coerce")
    ok = px.notna() & m200.notna()
    breadth = float((px[ok] > m200[ok]).mean() * 100) if ok.sum() >= 50 else None
    reg = simba.market_regime(st.session_state.get("nifty_data"), breadth)
    try:
        ind = simba.industry_stages(df)
    except Exception:            # degraded snapshot (e.g. a failed load): score industries as Neutral
        ind = {}
    rg = reg["regime"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SIMBA regime", f"{REG_ICON.get(rg, '')} {rg.title()}")
    c2.metric("Breadth (> 200-DMA)", "n/a" if breadth is None else f"{breadth:.0f}%")
    if reg.get("risk_scale"):
        lo, hi = reg["holdings"]
        c3.metric("Risk Scale", f"{reg['risk_scale']} / 10", help=(
            "Sets how many stocks to hold. The document uses 5 inputs; 2 exist here (breadth, Nifty's "
            "distance above its 10-month average). Valuation velocity, index P/E and froth flags are missing."))
        c4.metric("Hold", f"{lo}–{hi} stocks", f"full size ≈ {100 / hi:.0f}–{100 / lo:.0f}% · starter 2.5%",
                  delta_color="off")
    st.info(f"**{rg.title()}** — {simba.REGIME_ACTION.get(rg, '')}"
            + (f"  \nNifty {reg['dist_sma']:+.1f}% vs its 10-month average "
               f"({'rising' if reg['sma_rising'] else 'falling'}); 12-month return {reg['ret12']:+.0f}%. "
               "The document gauges this on the Smallcap 250; Nifty 50 is used here."
               if reg.get("dist_sma") is not None and reg.get("ret12") is not None else ""))

    n_lead = sum(v == "LEADING" for v in ind.values())
    n_imp = sum(v == "IMPROVING" for v in ind.values())
    with st.expander(f"🏭 Industries — {n_lead} stocks in Leading, {n_imp} in Improving sub-industries"):
        from utils.industry_alerts import all_sub_industry_readings
        try:
            rd = all_sub_industry_readings(df)
        except Exception:
            rd = pd.DataFrame()
        if not rd.empty:
            rd["SIMBA stage"] = rd["status"].map(lambda s: IND_ICON[simba.INDUSTRY_MAP.get(s, "NEUTRAL")])
            st.dataframe(rd.sort_values("score", ascending=False)[["industry", "SIMBA stage", "score", "above_50",
                                                                   "above_200", "med_rs", "n_valid"]],
                         hide_index=True, use_container_width=True,
                         column_config={"score": st.column_config.ProgressColumn("Trend score", min_value=0,
                                                                                 max_value=100, format="%.0f")})
            st.caption("Leading = UPTREND (≥ 50% of members above both averages); Improving = TURNING UP — "
                       "the 'better player in a sector entering Stage 2' filter.")

    fund = _fundamentals()
    if "simba_stage" not in df.columns or df["simba_stage"].notna().sum() == 0:
        st.warning("⏳ SIMBA chart signals are computed by the EOD engine from daily bars and appear after the "
                   "next run (3:45 pm IST), or now with **Bypass Master Cache (Live Mode)** in the sidebar. "
                   "Until then, the fundamental leg alone:")
        x = df[["ticker", "name", "currentPrice"]].merge(fund, on="ticker", how="left")
        x = x[x["er"] == True].assign(industry=lambda d: d.ticker.map(lambda t: IND_ICON[ind.get(t, "NEUTRAL")]))
        st.dataframe(x[["name", "ticker", "industry", "quarter", "sales_g", "np_g", "er_oplev", "er_accel",
                        "er_margin", "er_turn", "roce", "promoters"]],
                     hide_index=True, use_container_width=True)
        return

    S = simba.score_table(df, reg, fund, ind)
    if S.empty:
        st.info("No SIMBA readings in the loaded data.")
        return
    S["name"] = S.get("name", S["ticker"]).fillna(S["ticker"])
    S["stage"] = S["simba_stage"].map(STAGE_ICON)
    S["industry"] = S["industry_stage"].map(IND_ICON)
    S["caution"] = S["simba_caution"].map(CAUTION)
    S["mtf"] = S.apply(lambda r: f"{r.simba_rsi_d:.0f} / {r.simba_rsi_w:.0f} / {r.simba_rsi_m:.0f}"
                       if pd.notna(r.simba_rsi_m) else "—", axis=1)
    pxs = pd.to_numeric(S.get("currentPrice"), errors="coerce")
    S["chase_pct"] = (pxs / pd.to_numeric(S["simba_m_pivot"], errors="coerce") - 1) * 100
    S["er_legs"] = S.apply(lambda r: ", ".join(k for k, f in [("op-lev", "er_oplev"), ("accel", "er_accel"),
                                                              ("margin", "er_margin"), ("turn", "er_turn")]
                                               if r.get(f) is True) or "—", axis=1)
    if not S["simba_full"].fillna(False).any():
        st.caption("⚠️ Fewer than ~290 sessions of history in this snapshot: G1 (dormancy) and Mansfield RS "
                   "fill in from the next EOD run.")

    n_si = int((S.simba_si_since <= 5).sum())
    n_g = int(S.simba_g.fillna(False).sum())
    n_m = int((S.simba_m_since <= 5).sum())
    n_a = int(S.grade.isin(["A+", "A"]).sum())
    k = st.columns(4)
    k[0].metric("🎯 SI Opportunity (5 days)", n_si)
    k[1].metric("🌀 Ghoomega active", n_g)
    k[2].metric("🔥 Machayega (5 days)", n_m)
    k[3].metric("🏆 Grade A / A+ (no gate hit)", n_a)

    cols = ["name", "ticker", "grade", "score", "gates", "stage", "industry", "simba_g_bits", "simba_m_bits",
            "simba_g_since", "simba_m_since", "simba_vol_x_m", "simba_base_range", "chase_pct", "stop_pct", "mtf",
            "simba_mrs", "rs_rating", "er_legs", "roce", "promoters", "caution", "simba_trail"]
    cfg = {
        "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d",
                                                 help="Out of 100; 90 is the most measurable here (insider, "
                                                      "delivery and volume-vs-price growth are not in our data)"),
        "simba_g_bits": st.column_config.TextColumn("G1-G7", help="\n".join(G_LABELS)),
        "simba_m_bits": st.column_config.TextColumn("M1-M5 today", help="\n".join(M_LABELS)),
        "simba_g_since": st.column_config.NumberColumn("Ghoomega (days ago)", format="%d"),
        "simba_m_since": st.column_config.NumberColumn("Machayega (days ago)", format="%d"),
        "simba_vol_x_m": st.column_config.NumberColumn("Breakout vol ×", format="%.1f×"),
        "simba_base_range": st.column_config.NumberColumn("Base range", format="%.1f%%"),
        "chase_pct": st.column_config.NumberColumn("Above pivot", format="%+.1f%%",
                                                   help="Don't chase more than 5–7% above the breakout pivot"),
        "stop_pct": st.column_config.NumberColumn("Stop distance", format="%.1f%%",
                                                  help="To the 15-bar base low; must be ≤ 15%"),
        "mtf": st.column_config.TextColumn("RSI D / W / M", help="Aligned: > 55 / > 55 / > 50"),
        "simba_mrs": st.column_config.NumberColumn("Mansfield RS", format="%+.1f"),
        "rs_rating": st.column_config.NumberColumn("RS rating", format="%d", help="6-month return percentile, 1–99"),
        "er_legs": st.column_config.TextColumn("Earnings reversal"),
        "roce": st.column_config.NumberColumn("ROCE", format="%.0f%%"),
        "promoters": st.column_config.NumberColumn("Promoter %", format="%.0f%%"),
        "simba_trail": st.column_config.NumberColumn("Trail", format="%d%%",
                                                     help="Bachayega peak-drawdown stop by volatility"),
    }
    t_si, t_g, t_m, t_score, t_bach, t_check = st.tabs(
        ["🎯 SI Opportunity", "🌀 Ghoomega", "🔥 Machayega", "🏆 Scorecard", "🛡️ Bachayega", "📋 Checklist"])

    with t_si:
        st.caption("Machayega fired within 30 sessions of an active Ghoomega, with MTF RSI aligned and the stock "
                   "not extended — the document's core entry. Starter 2.5%, stop ≤ 15% below entry.")
        v = S[S.simba_si_since <= 5].sort_values(["simba_si_since", "score"], ascending=[True, False])
        st.dataframe(v[cols], hide_index=True, use_container_width=True, column_config=cfg) if len(v) else \
            st.info("No SI Opportunity in the last 5 sessions.")

    with t_g:
        st.caption("Stage-2 turn: all of G1–G7 true. 'Fresh' = active for ≤ 10 sessions. Our industry study found "
                   "early turns lagged confirmed uptrends — treat this as a watchlist for the Machayega spark.")
        v = S[S.simba_g.fillna(False)].sort_values(["simba_g_streak", "score"], ascending=[True, False])
        st.dataframe(v[cols + ["simba_g_streak"]], hide_index=True, use_container_width=True,
                     column_config={**cfg, "simba_g_streak": st.column_config.NumberColumn("Active (days)")}) \
            if len(v) else st.info("No stock passes all of G1–G7 today.")

    with t_m:
        st.caption("Price-volume breakout from a tight base in the last 5 sessions. A good weekly close above the "
                   "pivot confirms it; more than 5–7% above the pivot is a chase.")
        v = S[S.simba_m_since <= 5].sort_values(["simba_m_since", "score"], ascending=[True, False])
        v = v.assign(weekly=v["simba_good_wc"].map({True: "✅ good", False: "❌ weak"}).fillna("pending"))
        st.dataframe(v[cols + ["weekly"]], hide_index=True, use_container_width=True,
                     column_config={**cfg, "weekly": "Weekly close"}) if len(v) else \
            st.info("No Machayega breakout in the last 5 sessions.")

    with t_score:
        a, b, c = st.columns(3)
        g_sel = a.multiselect("Grade", ["A+", "A", "B"], default=["A+", "A"], key="simba_grades")
        show_gated = b.toggle("Include gated", value=False, key="simba_gated")
        ind_sel = c.multiselect("Industry", list(IND_ICON.values()), default=[], key="simba_ind")
        v = S[S.grade.str.replace(" (gated)", "", regex=False).isin(g_sel)]
        if not show_gated:
            v = v[v.gates.fillna("") == ""]
        if ind_sel:
            v = v[v.industry.isin(ind_sel)]
        st.dataframe(v[cols + ["pts_topdown", "pts_ghoomega", "pts_machayega", "pts_quality", "pts_float", "pts_mtf"]],
                     hide_index=True, use_container_width=True,
                     column_config={**cfg, "pts_topdown": "Top-down /20", "pts_ghoomega": "Ghoomega /20",
                                    "pts_machayega": "Machayega /20", "pts_quality": "Earnings+quality /25",
                                    "pts_float": "Float /10", "pts_mtf": "MTF /5"})
        st.caption("A+ ≥ 80: starter now, scale on confirmation · A 65–79: starter, scale after the next result · "
                   "B 50–64: watchlist. Hard gates: liquidity ≥ ₹5 cr/day, price ≥ ₹20, D/E ≤ 1 and CFO/EBITDA "
                   "≥ 60%, not bear contraction (unless RS ≥ 90 + earnings reversal), not extended, stop ≤ 15%.")

    with t_bach:
        held = _held()
        st.caption("Hard exits: 15% below entry, the volatility trail (20/25/30% from the peak close), or a close "
                   "below a flat/falling 150-DMA. Caution: L1 ≥ 4 distribution days or weekly RSI < 50 (tighten, "
                   "no adds); L2 close < falling 50-DMA, Mansfield < 0, 10-day ROC ≤ −12% or a climactic gap-down "
                   "(trim 33–50%); L3 stage breach (exit).")
        hv = S[S.ticker.isin(held)]
        st.markdown(f"**Your active positions** ({len(hv)})")
        if len(hv):
            st.dataframe(hv[["name", "ticker", "caution", "stage", "simba_dist_days", "simba_rsi_w", "simba_mrs",
                             "simba_trail"]], hide_index=True, use_container_width=True,
                         column_config={"simba_dist_days": "Distribution days (25)",
                                        "simba_rsi_w": st.column_config.NumberColumn("Weekly RSI", format="%.0f"),
                                        "simba_mrs": st.column_config.NumberColumn("Mansfield RS", format="%+.1f"),
                                        "simba_trail": st.column_config.NumberColumn("Trail", format="%d%%")})
        else:
            st.caption("No active positions in the Position Manager.")
        st.markdown(f"**Stage breach across the universe** ({int((S.simba_caution == 3).sum())})")
        st.dataframe(S[S.simba_caution == 3][["name", "ticker", "stage", "simba_ext_ma150", "simba_mrs"]].head(200),
                     hide_index=True, use_container_width=True,
                     column_config={"simba_ext_ma150": st.column_config.NumberColumn("vs 150-DMA", format="%+.1f%%")})

    with t_check:
        opts = S.ticker.tolist()
        t = st.selectbox("Stock", opts, index=0, key="simba_check_ticker",
                         format_func=lambda x: f"{x.replace('.NS', '')} — {S.set_index('ticker').at[x, 'name']}")
        r = S.set_index("ticker").loc[t]
        st.markdown(f"#### {r['name']} · score **{int(r['score'])}** · grade **{r['grade']}**"
                    + (f"  \n⛔ {r['gates']}" if r['gates'] else ""))
        a, b = st.columns(2)
        with a:
            st.markdown("**Top-down**")
            st.write(f"{_ok(rg != 'BEAR CONTRACTION')} Regime not bear contraction ({rg.title()})")
            st.write(f"{_ok(r['industry_stage'] in ('LEADING', 'IMPROVING'))} Industry Leading/Improving "
                     f"({IND_ICON[r['industry_stage']]})")
            st.write(f"{_ok(r['rs_rating'] >= 80 or bool(r['simba_mrs_cross']))} RS rating ≥ 80 or Mansfield "
                     f"crossing 0 (rating {r['rs_rating']:.0f})")
            st.markdown("**Ghoomega**")
            for lab, bit in zip(G_LABELS, str(r["simba_g_bits"])):
                st.write(f"{_ok(bit == '1')} {lab}")
            st.markdown("**Machayega (today)**")
            for lab, bit in zip(M_LABELS, str(r["simba_m_bits"])):
                st.write(f"{_ok(bit == '1')} {lab}")
            st.write(f"{'✅' if r['simba_good_wc'] is True else '❌' if r['simba_good_wc'] is False else '⏳'} "
                     "Good weekly close above the pivot")
        with b:
            st.markdown("**Confirmation**")
            st.write(f"{_ok(r['simba_mtf'] == 3)} MTF RSI aligned (D/W/M {r['mtf']})")
            st.write(f"{_ok(not r['simba_ext'])} Not extended (daily RSI ≤ 80 or ≤ 25% above 50-DMA)")
            st.write(f"{_ok(r.get('promoters', 0) >= 60)} Low float bonus (promoters ≥ 60%: "
                     f"{r.get('promoters', float('nan')):.0f}%)")
            st.write("➖ Insider cluster buying — not in our data")
            st.markdown("**Fundamentals**")
            st.write(f"{_ok(r.get('er') is True)} Earnings reversal ({r['er_legs']}; {r.get('quarter', '—')}: sales "
                     f"{r.get('sales_g', float('nan')):+.0f}%, profit {r.get('np_g', float('nan')):+.0f}% YoY)")
            st.write(f"{_ok(r.get('roce', 0) >= 15 or bool(r.get('roce_rising')))} ROCE ≥ 15% or rising "
                     f"({r.get('roce', float('nan')):.0f}%)")
            st.write(f"{_ok(r.get('de', 9) <= 1)} Debt/equity ≤ 1 ({r.get('de', float('nan')):.2f})")
            st.write(f"{_ok(r.get('cfo_ebitda', 0) >= 0.6)} 5-year CFO/EBITDA ≥ 60% "
                     f"({r.get('cfo_ebitda', float('nan')):.0%})")
            st.write(f"{_ok(r.get('opm_cv', 9) <= 0.25)} Stable margins (annual OPM CV ≤ 0.25: "
                     f"{r.get('opm_cv', float('nan')):.2f})")
            st.write(f"{_ok(not (r.get('cwip_fa', 0) > 0.3))} Capex strain (CWIP ≤ 30% of fixed assets)")
            st.markdown("**Execution**")
            st.write(f"{_ok(r['stop_pct'] <= 15)} Stop ≤ 15% below (base low {r['simba_base_low']:.2f}, "
                     f"{r['stop_pct']:.1f}% away)")
            if pd.notna(r["chase_pct"]):
                st.write(f"{_ok(r['chase_pct'] <= 7)} Within 7% of the breakout pivot ({r['chase_pct']:+.1f}%)")
            if reg.get("holdings"):
                lo, hi = reg["holdings"]
                st.write(f"Size: starter 2.5% → full ≈ {100 / hi:.0f}–{100 / lo:.0f}% (Risk Scale "
                         f"{reg['risk_scale']}); trail {r['simba_trail']:.0f}% from the peak close")
