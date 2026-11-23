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
    x["val_pct"], x["val_p50"], x["val_rating"] = np.nan, np.nan, ""
    if val is not None and not val.empty:
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
    return x


def run(T: pd.DataFrame, on: Iterable[str] = DEFAULT_ON) -> Dict[str, Any]:
    """Apply the stages in order. Returns counts after each stage, the survivors and near misses."""
    on = [k for k, _ in STAGES if k in set(on)]
    ext = T["extended"].eq(True) if "extended" in T.columns else pd.Series(False, index=T.index)
    live = T[T.price.notna() & (T.price > 0) & ~ext]
    counts: List[Dict[str, Any]] = [dict(stage="Universe", left=len(live))]
    mask = pd.Series(True, index=live.index)
    passes = {k: live[f"s_{k}"].eq(True) for k, _ in STAGES}
    for k in on:
        mask &= passes[k]
        counts.append(dict(stage=dict(STAGES)[k], left=int(mask.sum())))
    fails = sum((~passes[k]).astype(int) for k in on) if on else pd.Series(0, index=live.index)
    near = live[fails == 1].copy()
    if len(near):
        near["missed"] = [next(dict(STAGES)[k] for k in on if not passes[k].loc[i]) for i in near.index]
    rank = lambda z: z.sort_values(["ind_score", "comp_rs"], ascending=False)
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
