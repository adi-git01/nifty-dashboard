"""
Investment funnel: one path from the whole universe to a short list with a trade plan.

Each stage reuses a tool the dashboard already has, in the order a top-down investor works:

  0 MARKET     regime + risk scale (utils/simba.market_regime). Not a stock filter: it sets how
               many positions and how much risk. Momentum entries lost to the typical stock only
               in the BEAR regime (analysis/REGIME_VERDICT.md).
  1 INDUSTRY   the stock's sub-industry is LEADING or IMPROVING on the leadership board, or its
               trend status is UPTREND / TURNING UP (excluding WEAKENING) (utils/industry_board.py)
  2 TREND      Trend Scanner BUY (RS vs Nifty > 0, above the 50-day, liquid) and not in a
               Stage 3 / 4 chart (SIMBA stage)
  3 BUSINESS   business score >= 45 of 100 (growth, margins, cash and balance sheet, returns on
               capital), 3-year cash conversion >= 50%, debt / equity <= 1.5
               (utils/fund_trends.py; lenders skip the cash and debt tests)
  4 SETUP      an entry exists now: Technicals BUY NOW / BUY ON TRIGGER, or a SIMBA Machayega
               breakout / SI Opportunity in the last 5 sessions, AND the stop is within 15% of the
               price (a wider stop is not a tradable entry)
  5 VALUE      price not in the top 30% of the valuation range (cmp_pct <= 70). Shown always,
               OFF as a gate by default: the entry/exit study found expensive momentum beat cheap
               (analysis/ENTRY_EXIT_VERDICT.md). Stocks the model cannot value are "n/a" and pass.

Stages 1, 2, 4 and 5 are vectorised over the snapshot. Stage 3 reads the Screener files per stock
(about 30 ms each), so it is evaluated only for stocks that pass TREND.

The funnel as a whole has NOT been backtested. Its parts have mixed evidence (see each verdict
file); the combination is a disciplined checklist, not a proven system.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

STAGES = [("industry", "🏭 Industry"), ("trend", "🌊 Trend"), ("business", "🏢 Business"),
          ("setup", "🎯 Setup"), ("value", "💰 Value")]
DEFAULT_ON = ("industry", "trend", "business", "setup")
VALUE_MAX_PCT = 70
SETUP_DAYS = 5
MAX_RISK_PCT = 15         # a stop further away than this is not a tradable entry (SIMBA hard gate)
LENDER_INDUSTRIES = {"Banks", "Finance"}  # operating margin from sales is meaningless for lenders
WHAT = {
    "industry": "Sub-industry is Leading / Improving on the leadership board, or in an Uptrend / Turning up (excluding Weakening groups)",
    "trend": "Trend Scanner BUY (RS > 0, above 50-day, liquid) and not a Stage 3 / 4 chart",
    "business": "Business score ≥ 45/100 (growth, margins, cash, returns); cash conversion ≥ 50%; debt/equity ≤ 1.5",
    "setup": f"Technicals BUY NOW / BUY ON TRIGGER, or a SIMBA breakout / SI Opportunity within {SETUP_DAYS} sessions, with the stop within {MAX_RISK_PCT}%",
    "value": f"Price in the lower {VALUE_MAX_PCT}% of the valuation range (unvalued stocks pass as n/a)",
}


def _num(df: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)


def load_valuation(path: str = "data/valuation_snapshot.csv") -> pd.DataFrame:
    try:
        v = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    v["ticker"] = v["ticker"].astype(str).map(lambda t: t if t.endswith(".NS") else t + ".NS")
    return v.drop_duplicates("ticker").set_index("ticker")


def table(df: pd.DataFrame, board: Optional[pd.DataFrame] = None, val: Optional[pd.DataFrame] = None,
          business_for: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    One row per stock with a pass flag per stage (s_industry ... s_value) and the facts behind it.
    business_for: tickers to evaluate the business stage for; None = those passing TREND.
    s_business is NaN where it was not evaluated.
    """
    from utils.nifty1000_list import SUB_INDUSTRY_MAP
    d = df.drop_duplicates("ticker").reset_index(drop=True)
    x = pd.DataFrame({"ticker": d["ticker"].values})
    x["name"] = d.get("name", d["ticker"]).values
    x["price"] = _num(d, "currentPrice").fillna(_num(d, "price")).values
    x["extended"] = d["extended"].eq(True).values if "extended" in d.columns else False
    x["industry"] = x.ticker.map(SUB_INDUSTRY_MAP)

    # 1 industry
    x["ind_group"], x["ind_status"], x["ind_score"] = "", "", np.nan
    if board is not None and not board.empty:
        b = board.set_index("industry")
        x["ind_group"] = x.industry.map(b["group"]).fillna("")
        x["ind_status"] = x.industry.map(b["status"]).fillna("")
        x["ind_score"] = x.industry.map(b["score"])
    x["s_industry"] = (x.ind_group.isin(["LEADING", "IMPROVING"]) | x.ind_status.isin(["UPTREND", "TURNING UP"])) & ~x.ind_group.eq("WEAKENING")

    # 2 trend
    x["scanner"] = d["dna_signal"].fillna("").astype(str) if "dna_signal" in d.columns else ""
    x["stage"] = _num(d, "simba_stage")
    x["simba_mtf"] = _num(d, "simba_mtf")
    x["comp_rs"], x["rs_1m"], x["rs_3m"], x["dist_52w"] = _num(d, "comp_rs"), _num(d, "rs_1m"), _num(d, "rs_3m"), _num(d, "dist_52w")
    ma50, ma200 = _num(d, "fiftyDayAverage"), _num(d, "twoHundredDayAverage")
    x["ma50"], x["ma200"] = ma50, ma200
    x["dist_ma50"] = ((x.price / ma50 - 1.0) * 100.0).where((x.price > 0) & (ma50 > 0)).fillna(_num(d, "dist_ma50"))
    x["dist_200dma"] = _num(d, "dist_200dma").fillna(((x.price / ma200 - 1.0) * 100.0).where((x.price > 0) & (ma200 > 0)))
    x["pe"] = _num(d, "pe")
    x["s_trend"] = x.scanner.eq("BUY") & ~x.stage.isin([3, 4])

    # 4 setup
    x["technicals"] = d["fx_action"].fillna("").astype(str) if "fx_action" in d.columns else ""
    x["fx_state"] = d["fx_state"].fillna("").astype(str) if "fx_state" in d.columns else ""
    x["fx_supply"] = d["fx_supply"].fillna("").astype(str) if "fx_supply" in d.columns else ""
    x["fx_signature"] = d["fx_signature"].fillna("").astype(str) if "fx_signature" in d.columns else ""
    _as_b = lambda v: v is True or str(v).strip().lower() in ("true", "1", "yes")
    x["fx_shakeout"] = d["fx_shakeout"].map(_as_b) if "fx_shakeout" in d.columns else False
    x["fx_adx"], x["fx_rsi"] = _num(d, "fx_adx"), _num(d, "fx_rsi")
    x["fx_pdi"], x["fx_ndi"] = _num(d, "fx_pdi"), _num(d, "fx_ndi")
    x["fx_bbp"] = _num(d, "fx_bbp")
    x["fx_gap"] = d["fx_gap"].map(_as_b) if "fx_gap" in d.columns else False
    x["hi52_bo_days"], x["ath_bo_days"] = _num(d, "hi52_bo_days"), _num(d, "ath_bo_days")
    m, si = _num(d, "simba_m_since"), _num(d, "simba_si_since")
    x["simba_days"] = pd.concat([m, si], axis=1).min(axis=1)
    x["stop"], x["stop_pct"], x["t1"], x["t2"] = _num(d, "fx_stop"), _num(d, "fx_stop_pct"), _num(d, "fx_t1"), _num(d, "fx_t2")
    x["entry"] = _num(d, "fx_entry").fillna(x.price) if "fx_entry" in d else x.price      # the trigger for BUY ON TRIGGER
    x["dist_stop_pct"] = ((x.price - x.stop) / x.price * 100.0).where((x.price > 0) & (x.stop > 0))
    x["trigger_gap_pct"] = ((x.entry / x.price - 1.0) * 100.0).where((x.price > 0) & (x.entry > 0))
    x["risk_pct"] = ((x.entry - x.stop) / x.entry * 100).where((x.stop > 0) & (x.entry > 0) & (x.price > 0) & (x.stop < x.entry) & (x.stop < x.price))
    x["fx_half_size"] = d["fx_half_size"].map(_as_b) if "fx_half_size" in d.columns else (x["risk_pct"] > 10.0)
    x["t1"] = x["t1"].where(x["t1"] > x.entry).fillna(x.entry + 2 * (x.entry - x.stop)).where(x.risk_pct.notna())
    x["signal"] = x.technicals.isin(["BUY NOW", "BUY ON TRIGGER"]) | ((x.simba_days >= 0) & (x.simba_days <= SETUP_DAYS))
    x["s_setup"] = x.signal & (x.risk_pct <= MAX_RISK_PCT)

    # 5 value
    x["val_pct"], x["val_p50"], x["val_rating"], x["val_loss"] = np.nan, np.nan, "", False
    if val is not None and not val.empty:
        if "reason" in val.columns:
            loss_idx = val.index[val["reason"].fillna("").astype(str).str.contains("not positive|loss|negative", case=False)]
            x["val_loss"] = x.ticker.isin(loss_idx)
        ok = val[val.get("status") == "OK"] if "status" in val else val
        ok = ok[~ok.index.duplicated(keep="first")]
        x["val_pct"] = x.ticker.map(pd.to_numeric(ok["cmp_pct"], errors="coerce"))
        x["val_p50"] = x.ticker.map(pd.to_numeric(ok["p50"], errors="coerce"))
        x["val_rating"] = x.ticker.map(ok["rating"]).fillna("")
        if "grid" in ok.columns:
            from utils import valuation as V
            snap_px_col = pd.to_numeric(ok["price"], errors="coerce") if "price" in ok.columns else None
            has_meta = all(c in ok.columns for c in ("gates_ok", "confidence", "identity"))
            for i, t, px in zip(x.index, x.ticker, x.price):
                if t not in ok.index or pd.isna(px) or px <= 0:
                    continue
                r = ok.loc[t]
                raw_grid = r.get("grid")
                if not raw_grid or pd.isna(raw_grid):
                    continue
                if snap_px_col is not None:
                    sp = snap_px_col.get(t)
                    if pd.notna(sp) and sp > 0 and abs(float(px) / float(sp) - 1.0) <= 1e-4:
                        continue
                try:
                    g = json.loads(raw_grid) if isinstance(raw_grid, str) else list(raw_grid)
                    if not g:
                        continue
                    if has_meta and len(g) == 99:
                        pct, rat = V.rating_from_grid(g, float(px), bool(r["gates_ok"]), str(r["confidence"]), str(r["identity"]))
                        x.at[i, "val_rating"] = rat
                    else:
                        ga = np.asarray(g, float)
                        pct = float(np.interp(float(px), ga, np.linspace(1, 99, len(ga)), left=0.5, right=99.5))
                    x.at[i, "val_pct"] = round(pct, 1)
                except Exception:
                    continue
    x["val_upside"] = ((x.val_p50 / x.price - 1) * 100).where(x.price > 0)
    x["s_value"] = x.val_pct.isna() | (x.val_pct <= VALUE_MAX_PCT)

    # 3 business: every stock (the scores are precomputed, ~0.05 s for all). Checking only trend passers left the rest
    # blank, which counted as a second failed stage: trend near-misses vanished, and switching Trend off kept 24 picks, not 42
    from utils import fund_trends
    todo = set(x.ticker) if business_for is None else set(business_for)
    x["s_business"], x["biz_verdict"], x["biz_flags"], x["quarter"] = np.nan, "", "", ""
    x["biz_score"], x["biz_headline"] = np.nan, ""
    x["sales_yoy"], x["np_yoy"], x["roce"] = np.nan, np.nan, np.nan
    x["opm"], x["opm_chg"] = np.nan, np.nan
    x["s_business"] = x["s_business"].astype(object)
    try:
        opm_lookup = fund_trends.q_opm_map() if todo else {}
    except Exception:
        opm_lookup = {}
    # institutional holding change (FII + DII, quarterly filings): context and a filter, never a stage
    try:
        inst = fund_trends.inst_map()
    except Exception:
        inst = {}
    g_inst = lambda t, k: inst.get(t, {}).get(k, np.nan)
    x["inst_label"] = [inst.get(t, {}).get("label", "") for t in x.ticker]
    x["inst_q"] = [g_inst(t, "d_inst_q") for t in x.ticker]
    x["inst_y"] = [g_inst(t, "d_inst_y") for t in x.ticker]
    x["fii_pct"] = [g_inst(t, "fii") for t in x.ticker]
    x["dii_pct"] = [g_inst(t, "dii") for t in x.ticker]
    x["prom_pct"] = [g_inst(t, "promoters") for t in x.ticker]
    x["prom_q"] = [g_inst(t, "d_prom_q") for t in x.ticker]
    av = _num(d, "averageVolume")
    x["liq_cr"] = (av * x.price / 1e7).where((av > 0) & (x.price > 0))
    try:
        from utils import nse_data
        deals = nse_data.deal_tally(90)
    except Exception:
        deals = {}
    x["deal_inst_cr"] = [deals[t]["inst"] if t in deals else np.nan for t in x.ticker]
    x["deliv_pct"] = _num(d, "deliv_pct") if "deliv_pct" in d else np.nan
    x["deliv_avg20"] = _num(d, "deliv_avg20") if "deliv_avg20" in d else np.nan
    x["nse_flags"] = d["nse_flags"].fillna("").astype(str) if "nse_flags" in d else ""
    if ("deliv_pct" not in d or "nse_flags" not in d) and len(x):
        try:
            from utils import nse_data
            att = nse_data.attach(x[["ticker"]])
            if "deliv_pct" not in d and "deliv_pct" in att:
                x["deliv_pct"] = pd.to_numeric(att["deliv_pct"], errors="coerce")
            if "deliv_avg20" not in d and "deliv_avg20" in att:
                x["deliv_avg20"] = pd.to_numeric(att["deliv_avg20"], errors="coerce")
            if "nse_flags" not in d and "nse_flags" in att:
                x["nse_flags"] = att["nse_flags"].fillna("").astype(str)
        except Exception:
            pass
    for i, t in zip(x.index, x.ticker):
        if t not in todo:
            continue
        if t in opm_lookup and str(x.at[i, "industry"]) not in LENDER_INDUSTRIES:
            x.at[i, "opm"], x.at[i, "opm_chg"] = opm_lookup[t]
        try:
            b = fund_trends.brief(t)
        except Exception:
            continue
        x.at[i, "s_business"] = bool(b["ok"])
        x.at[i, "biz_verdict"], x.at[i, "biz_flags"] = b["verdict"], "; ".join(b["flags"])
        x.at[i, "biz_score"], x.at[i, "biz_headline"] = b.get("score"), b.get("headline", b["verdict"].title())
        x.at[i, "quarter"] = b["quarter"] or ""
        x.at[i, "sales_yoy"], x.at[i, "np_yoy"], x.at[i, "roce"] = b["sales_yoy"], b["np_yoy"], b["roce"]
    return score_table(x)


def score_table(T: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the calibrated v3 0-100 Funnel ranking score (`funnel_score`), letter grade (`funnel_grade`: A/B/C),
    and breakdown (`score_note` = `Mom · Set · Str · Qual · Cat`):
      - Base (0-80):
          * Momentum & Industry (Mom, 0-28): CompRS (14) + Industry Leader (10) + Near 52wH (4)
          * Setup, Timing & Extension (Set, 0-22): Action + fresh SIMBA lift (capped at 8 for WAIT, 4 for WATCH)
            + gap/squeeze/breakout bonus (+4 max) - overhead supply (NEAR OVERHANG -6, INSIDE SHELF -2.5)
            - extended entry trim (vs MA50 / 200DMA)
          * Trend Structure (Str, 0-12): ADX>=25 & +DI>=-DI (7; 20-25: 3.5) + SIMBA MTF (5/3/1) kept separate
          * Business, Valuation Tail & Execution Hygiene (Qual, 0-18 base; red flags are un-floored):
            Business score (0-8) - qtr deterioration (-4.5) + valid stop 3-15% (5; 2-3%: 2; <2% noise stop: -3)
            + liquidity (3) + clean earnings/valuation floor (+2; REDUCE -2; STRONG SELL/PE>=100 -4; loss-maker -12)
            - surveillance ⚠️ (-14)
      - Proven Alpha Catalysts (Cat, +0..22 bonus over Base, total capped at 100):
          * ABSORPTION (+13), Shakeout (+9), FII+DII ACCUMULATING (+8 with YoY > -2 pp sanity check),
            net inst deals (+2), high delivery (+2)
          * DISTRIBUTION and DISTRIBUTING carry zero penalty.
    """
    x = T.copy()
    if x.empty:
        x["funnel_score"] = np.nan
        x["funnel_grade"] = ""
        x["score_note"] = ""
        return x
    num = lambda c, d=np.nan: pd.to_numeric(x[c], errors="coerce").fillna(d) if c in x.columns else pd.Series(d, index=x.index, dtype=float)
    txt = lambda c: x[c].fillna("").astype(str) if c in x.columns else pd.Series("", index=x.index)

    # 1. Momentum & Industry (0-28)
    rs_pts = ((num("comp_rs", -5.0) + 5.0) / 25.0).clip(0.0, 1.0) * 14.0
    ind_pts = (num("ind_score", 0.0).clip(0.0, 100.0) / 100.0) * 8.0 + txt("ind_group").str.upper().map({"LEADING": 2.0, "IMPROVING": 1.0}).fillna(0.0)
    d52_pts = ((num("dist_52w", -20.0) + 20.0) / 20.0).clip(0.0, 1.0) * 4.0
    p_mom = np.clip(rs_pts + ind_pts + d52_pts, 0.0, 28.0)

    # 2. Setup, Timing & Extension (0-22) — CLEAN AIR gives 0 extra (= Gate 1); overhang & extension deducted
    act = txt("technicals").str.upper()
    sd = num("simba_days", 99.0)
    simba_fresh = (sd >= 0.0) & (sd <= SETUP_DAYS)
    act_pts = act.map({"BUY NOW": 18.0, "BUY ON TRIGGER": 12.0, "WAIT": 5.0, "WATCH": 0.0}).fillna(0.0)
    act_pts = np.where(
        (act == "WAIT") & simba_fresh,
        8.0,
        np.where(
            (act == "WATCH") & simba_fresh,
            4.0,
            np.where((act_pts == 0.0) & (act != "WATCH") & simba_fresh, 10.0, act_pts),
        ),
    )
    bo_hit = num("hi52_bo_days", 99.0).between(0.0, 10.0) | num("ath_bo_days", 99.0).between(0.0, 10.0)
    gap_s = x["fx_gap"].eq(True) if "fx_gap" in x.columns else pd.Series(False, index=x.index)
    gap_b = np.where(gap_s, 2.0, 0.0)
    sq_b = np.where(num("fx_bbp", 50.0) < 20.0, 2.0, 0.0)
    bo_b = np.where(bo_hit, 2.0, 0.0)
    ovh_pen = txt("fx_supply").str.upper().map({"NEAR OVERHANG": -6.0, "INSIDE SHELF": -2.5}).fillna(0.0)
    dma50, dma200 = num("dist_ma50", 0.0), num("dist_200dma", 0.0)
    ext_pen = (
        np.where(dma50 > 20.0, -3.0, np.where(dma50 > 16.0, -1.5, 0.0))
        + np.where(dma200 > 60.0, -5.0, np.where(dma200 > 38.0, -3.0, 0.0))
    )
    p_set = np.clip(act_pts + np.minimum(gap_b + sq_b + bo_b, 4.0) + ovh_pen + ext_pen, 0.0, 22.0)

    # 3. Trend Structure — ADX & SIMBA MTF (0-12, kept separate from Momentum)
    adx = num("fx_adx", 0.0)
    pdi = num("fx_pdi", np.nan)
    ndi = num("fx_ndi", np.nan)
    pdi_ok = pdi.isna() | ndi.isna() | (pdi >= ndi)
    adx_pts = np.where((adx >= 25.0) & pdi_ok, 7.0, np.where((adx >= 20.0) & pdi_ok, 3.5, 0.0))
    mtf_pts = num("simba_mtf", 0.0).map({3.0: 5.0, 2.0: 3.0, 1.0: 1.0}).fillna(0.0)
    p_str = adx_pts + mtf_pts

    # 4. Business, Valuation Tail & Execution Hygiene (0-18 base; red flags are un-floored to pull total down)
    biz_pts = ((num("biz_score", 40.0) - 40.0) / 45.0).clip(0.0, 1.0) * 8.0
    bflg = txt("biz_flags").str.strip().ne("")
    q_deter = bflg | ((num("sales_yoy", 0.0) < 0.0) & (num("opm_chg", 0.0) <= -1.5))
    biz_pen = np.where(q_deter, -4.5, 0.0)

    rk = num("risk_pct", np.nan)
    rk_pts = np.where(
        rk.notna() & (rk >= 3.0) & (rk <= MAX_RISK_PCT),
        5.0,
        np.where(
            rk.notna() & (rk >= 2.0) & (rk < 3.0),
            2.0,
            np.where(rk.notna() & (rk > 0.0) & (rk < 2.0), -3.0, 0.0),
        ),
    )
    liq = num("liq_cr", 0.0)
    liq_pts = np.where(liq >= 20.0, 3.0, np.where(liq >= 5.0, 2.0, np.where(liq >= 2.0, 1.0, 0.0)))

    pe = num("pe", np.nan)
    vloss = x["val_loss"].eq(True) if "val_loss" in x.columns else pd.Series(False, index=x.index)
    is_loss = vloss | (pe <= 0.0) | (pe >= 200.0)
    vrat = txt("val_rating").str.upper()
    is_rich_tail = vrat.str.contains("STRONG SELL", na=False) | (pe >= 100.0)
    is_reduce = vrat.eq("REDUCE")
    val_pts = np.where(is_loss, -12.0, np.where(is_rich_tail, -4.0, np.where(is_reduce, -2.0, 2.0)))

    surv = txt("nse_flags").str.contains("⚠️", na=False)
    surv_pen = np.where(surv, -14.0, 0.0)
    p_qual = biz_pts + biz_pen + rk_pts + liq_pts + val_pts + surv_pen

    # 5. Proven Alpha Catalysts (+0..22 bonus over Base): Absorption, Shakeout & Inst Accumulation
    abs_pts = np.where(txt("fx_signature").str.upper() == "ABSORPTION", 13.0, 0.0)
    shk_s = x["fx_shakeout"].eq(True) if "fx_shakeout" in x.columns else pd.Series(False, index=x.index)
    shk_pts = np.where(shk_s, 9.0, 0.0)
    iy = num("inst_y", np.nan)
    inst_acc = np.where(txt("inst_label").str.upper().eq("ACCUMULATING") & (iy.isna() | (iy > -2.0)), 8.0, 0.0)
    deal_b = np.where(num("deal_inst_cr", 0.0) > 0.0, 2.0, 0.0)
    dp, da = num("deliv_pct", np.nan), num("deliv_avg20", np.nan)
    deliv_b = np.where((dp > 0.0) & (da > 0.0) & (dp / da >= 1.25), 2.0, 0.0)
    p_cat = np.minimum(abs_pts + shk_pts + inst_acc + deal_b + deliv_b, 22.0)

    tot = np.clip(p_mom + p_set + p_str + p_qual + p_cat, 0.0, 100.0).round(1)
    x["funnel_score"] = tot
    x["funnel_grade"] = np.where(tot >= 74.0, "A", np.where(tot >= 60.0, "B", "C"))
    x["score_note"] = [
        f"Mom {a:.0f} · Set {b:.0f} · Str {c:.0f} · Qual {d:.0f} · Cat {e:.0f}"
        for a, b, c, d, e in zip(p_mom, p_set, p_str, p_qual, p_cat)
    ]
    return x


def _miss_gap(row: pd.Series, missed_key: str) -> str:
    """Short human-readable distance to passing the one failed stage for a near miss."""
    s_txt = lambda k: "" if row.get(k) is None or pd.isna(row.get(k)) or str(row.get(k)).strip() in ("", "nan", "None") else str(row.get(k)).strip()
    if missed_key == "industry":
        grp = s_txt("ind_group").upper()
        sc = row.get("ind_score")
        if grp == "WEAKENING":
            return f"Weakening industry ({sc:.0f})" if pd.notna(sc) else "Weakening industry"
        if pd.notna(sc):
            gap = max(0.0, 60.0 - float(sc))
            return f"Ind {float(sc):.0f} ({gap:.0f} pts to Leading)"
        return "Lagging industry"
    if missed_key == "trend":
        stg = row.get("stage")
        crs = row.get("comp_rs")
        scan = s_txt("scanner")
        if pd.notna(stg) and int(float(stg)) in (3, 4):
            return f"Stage {int(float(stg))} (downtrend)"
        if pd.notna(crs) and float(crs) < 0:
            return f"RS {float(crs):+.1f} ({abs(float(crs)):.1f} to BUY)"
        return f"Scanner {scan or 'HOLD'}"
    if missed_key == "business":
        flags = s_txt("biz_flags")
        bsc = row.get("biz_score")
        if flags:
            return f"Flags: {flags}"
        if pd.notna(bsc):
            return f"Biz {float(bsc):.0f} ({max(0.0, 45.0 - float(bsc)):.0f} pts to pass)"
        return "No business score"
    if missed_key == "setup":
        tech = s_txt("technicals")
        rk = row.get("risk_pct")
        sig = bool(row.get("signal")) if pd.notna(row.get("signal")) else False
        if sig and pd.notna(rk) and float(rk) > MAX_RISK_PCT:
            return f"Stop {float(rk):.1f}% (+{float(rk) - MAX_RISK_PCT:.1f}% over {MAX_RISK_PCT}% cap)"
        if sig:
            return "No valid stop"
        if tech.upper() == "WAIT":
            return f"WAIT · {float(rk):.1f}% stop (needs trigger)" if pd.notna(rk) else "WAIT (needs trigger)"
        return f"{tech or 'WATCH'} (no setup)"
    if missed_key == "value":
        vp = row.get("val_pct")
        return f"P{float(vp):.0f} (+{float(vp) - VALUE_MAX_PCT:.0f} over P{VALUE_MAX_PCT})" if pd.notna(vp) else "Unvalued"
    return "—"


def run(T: pd.DataFrame, on: Iterable[str] = DEFAULT_ON) -> Dict[str, Any]:
    """Apply the stages in order. Returns counts after each stage, the survivors and near misses."""
    on = [k for k, _ in STAGES if k in set(on)]
    ext = T["extended"].eq(True) if "extended" in T.columns else pd.Series(False, index=T.index)
    live = T[T.price.notna() & (T.price > 0) & ~ext].copy()
    if "funnel_score" not in live.columns:
        live = score_table(live)
    counts: List[Dict[str, Any]] = [dict(stage="Universe", left=len(live))]
    mask = pd.Series(True, index=live.index)
    passes = {k: live[f"s_{k}"].eq(True) for k, _ in STAGES}
    for k in on:
        mask &= passes[k]
        counts.append(dict(stage=dict(STAGES)[k], left=int(mask.sum())))
    fails = sum((~passes[k]).astype(int) for k in on) if on else pd.Series(0, index=live.index)
    near = live[fails == 1].copy()
    if len(near):
        missed_keys = [next(k for k in on if not passes[k].loc[i]) for i in near.index]
        near["missed"] = [dict(STAGES)[k] for k in missed_keys]
        near["miss_gap"] = [_miss_gap(live.loc[i], mk) for i, mk in zip(near.index, missed_keys)]
    rank = lambda z: z.sort_values(["funnel_score", "ind_score", "comp_rs"], ascending=False, na_position="last")
    return dict(counts=pd.DataFrame(counts), picks=rank(live[mask]), near=rank(near), on=on)


def checklist(row: pd.Series) -> List[Dict[str, Any]]:
    """The funnel as a per-stock checklist (for the stock report)."""
    f = lambda v, fmt="{:.0f}": "—" if pd.isna(v) else fmt.format(v)
    s_txt = lambda v: "" if v is None or pd.isna(v) or str(v).strip() in ("", "nan", "None") else str(v).strip()
    ind_g = s_txt(row.get("ind_group")).title()
    ind_st = s_txt(row.get("ind_status")).title()
    biz_v = s_txt(row.get("biz_verdict"))
    biz_h = s_txt(row.get("biz_headline"))
    biz_f = s_txt(row.get("biz_flags"))
    qtr = s_txt(row.get("quarter"))
    scan = s_txt(row.get("scanner"))
    tech = s_txt(row.get("technicals"))
    out = [
        dict(stage="🏭 Industry", ok=bool(row.s_industry),
             why=f"{s_txt(row.get('industry')) or 'Unclassified'}: {ind_g or '—'}, {ind_st or '—'}, score {f(row.ind_score)}"),
        dict(stage="🌊 Trend", ok=bool(row.s_trend),
             why=f"Scanner {scan or '—'}, chart stage {f(row.stage)}, RS {f(row.comp_rs, '{:+.1f}')}, {f(row.dist_52w)}% from 52-week high"),
        dict(stage="🏢 Business", ok=None if pd.isna(row.s_business) else bool(row.s_business),
             why=(f"{biz_h or biz_v.title()}" + (f"; flags: {biz_f}" if biz_f else "")
                  + (f"; {qtr}: sales {f(row.sales_yoy, '{:+.0f}')}%, profit {f(row.np_yoy, '{:+.0f}')}% YoY" if qtr else ""))
             if biz_v else "Not evaluated"),
        dict(stage="🎯 Setup", ok=bool(row.s_setup),
             why=f"Technicals {tech or '—'}" + (f"; SIMBA signal {f(row.simba_days)} session(s) ago" if pd.notna(row.simba_days) else "")
             + (f"; stop {f(row.risk_pct, '{:.1f}')}% away" + (" — too wide" if row.signal and row.risk_pct > MAX_RISK_PCT else "")
                if pd.notna(row.risk_pct) else "; no valid stop")),
        dict(stage="💰 Value", ok=None if pd.isna(row.val_pct) else bool(row.s_value),
             why="Not valued by the model" if pd.isna(row.val_pct) else
             f"Price at the {f(row.val_pct)}th percentile of the value range; mid value ₹{f(row.val_p50, '{:,.0f}')} ({f(row.val_upside, '{:+.0f}')}%); {s_txt(row.get('val_rating'))}"),
    ]
    return out
