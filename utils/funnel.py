"""
Investment funnel: one path from the whole universe to a short list with a trade plan.

Each stage reuses a tool the dashboard already has, in the order a top-down investor works:

  0 MARKET     regime + risk scale (utils/simba.market_regime). Not a stock filter: it sets how
               many positions and how much risk. Momentum entries lost to the typical stock only
               in the BEAR regime (analysis/REGIME_VERDICT.md).
  1 INDUSTRY   the stock's sub-industry is LEADING or IMPROVING on the leadership board, or its
               trend status is UPTREND / TURNING UP (utils/industry_board.py)
  2 TREND      Trend Scanner BUY (RS vs Nifty > 0, above the 50-day, liquid) and not in a
               Stage 3 / 4 chart (SIMBA stage)
  3 BUSINESS   business trends not WORSENING, 3-year cash conversion >= 50%, debt / equity <= 1.5
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

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

STAGES = [("industry", "🏭 Industry"), ("trend", "🌊 Trend"), ("business", "🏢 Business"),
          ("setup", "🎯 Setup"), ("value", "💰 Value")]
DEFAULT_ON = ("industry", "trend", "business", "setup")
VALUE_MAX_PCT = 70
SETUP_DAYS = 5
MAX_RISK_PCT = 15         # a stop further away than this is not a tradable entry (SIMBA hard gate)
WHAT = {
    "industry": "Sub-industry is Leading / Improving on the leadership board, or in an Uptrend / Turning up",
    "trend": "Trend Scanner BUY (RS > 0, above 50-day, liquid) and not a Stage 3 / 4 chart",
    "business": "Business trends not worsening; cash conversion ≥ 50%; debt/equity ≤ 1.5",
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
    x = pd.DataFrame({"ticker": df["ticker"].values})
    x["name"] = df.get("name", df["ticker"]).values
    x["price"] = _num(df, "currentPrice").fillna(_num(df, "price")).values
    x["industry"] = x.ticker.map(SUB_INDUSTRY_MAP)
    d = df.reset_index(drop=True)

    # 1 industry
    x["ind_group"], x["ind_status"], x["ind_score"] = "", "", np.nan
    if board is not None and not board.empty:
        b = board.set_index("industry")
        x["ind_group"] = x.industry.map(b["group"]).fillna("")
        x["ind_status"] = x.industry.map(b["status"]).fillna("")
        x["ind_score"] = x.industry.map(b["score"])
    x["s_industry"] = x.ind_group.isin(["LEADING", "IMPROVING"]) | x.ind_status.isin(["UPTREND", "TURNING UP"])

    # 2 trend
    x["scanner"] = d.get("dna_signal", pd.Series("", index=d.index)).astype(str)
    x["stage"] = _num(d, "simba_stage")
    x["comp_rs"], x["rs_3m"], x["dist_52w"] = _num(d, "comp_rs"), _num(d, "rs_3m"), _num(d, "dist_52w")
    x["s_trend"] = x.scanner.eq("BUY") & ~x.stage.isin([3, 4])

    # 4 setup
    x["technicals"] = d.get("fx_action", pd.Series("", index=d.index)).astype(str)
    m, si = _num(d, "simba_m_since"), _num(d, "simba_si_since")
    x["simba_days"] = pd.concat([m, si], axis=1).min(axis=1)
    x["stop"], x["stop_pct"], x["t1"], x["t2"] = _num(d, "fx_stop"), _num(d, "fx_stop_pct"), _num(d, "fx_t1"), _num(d, "fx_t2")
    x["risk_pct"] = ((x.price - x.stop) / x.price * 100).where(x.stop < x.price)
    x["signal"] = x.technicals.isin(["BUY NOW", "BUY ON TRIGGER"]) | (x.simba_days <= SETUP_DAYS)
    x["s_setup"] = x.signal & (x.risk_pct <= MAX_RISK_PCT)

    # 5 value
    x["val_pct"], x["val_p50"], x["val_rating"] = np.nan, np.nan, ""
    if val is not None and not val.empty:
        ok = val[val.get("status") == "OK"] if "status" in val else val
        x["val_pct"] = x.ticker.map(pd.to_numeric(ok["cmp_pct"], errors="coerce"))
        x["val_p50"] = x.ticker.map(pd.to_numeric(ok["p50"], errors="coerce"))
        x["val_rating"] = x.ticker.map(ok["rating"]).fillna("")
    x["val_upside"] = (x.val_p50 / x.price - 1) * 100
    x["s_value"] = x.val_pct.isna() | (x.val_pct <= VALUE_MAX_PCT)

    # 3 business (lazy)
    from utils import fund_trends
    todo = set(x.ticker[x.s_trend]) if business_for is None else set(business_for)
    x["s_business"], x["biz_verdict"], x["biz_flags"], x["quarter"] = np.nan, "", "", ""
    x["sales_yoy"], x["np_yoy"], x["roce"] = np.nan, np.nan, np.nan
    x["s_business"] = x["s_business"].astype(object)
    for i, t in zip(x.index, x.ticker):
        if t not in todo:
            continue
        try:
            b = fund_trends.brief(t)
        except Exception:
            continue
        x.at[i, "s_business"] = bool(b["ok"])
        x.at[i, "biz_verdict"], x.at[i, "biz_flags"] = b["verdict"], "; ".join(b["flags"])
        x.at[i, "quarter"] = b["quarter"] or ""
        x.at[i, "sales_yoy"], x.at[i, "np_yoy"], x.at[i, "roce"] = b["sales_yoy"], b["np_yoy"], b["roce"]
    return x


def run(T: pd.DataFrame, on: Iterable[str] = DEFAULT_ON) -> Dict[str, Any]:
    """Apply the stages in order. Returns counts after each stage, the survivors and near misses."""
    on = [k for k, _ in STAGES if k in set(on)]
    live = T[T.price.notna()]
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
    out = [
        dict(stage="🏭 Industry", ok=bool(row.s_industry),
             why=f"{row.industry or 'Unclassified'}: {row.ind_group.title() or '—'}, {row.ind_status.title() or '—'}, score {f(row.ind_score)}"),
        dict(stage="🌊 Trend", ok=bool(row.s_trend),
             why=f"Scanner {row.scanner or '—'}, chart stage {f(row.stage)}, RS {f(row.comp_rs, '{:+.1f}')}, {f(row.dist_52w)}% from 52-week high"),
        dict(stage="🏢 Business", ok=None if pd.isna(row.s_business) else bool(row.s_business),
             why=(f"Trends {row.biz_verdict.title()}" + (f"; flags: {row.biz_flags}" if row.biz_flags else "")
                  + (f"; {row.quarter}: sales {f(row.sales_yoy, '{:+.0f}')}%, profit {f(row.np_yoy, '{:+.0f}')}% YoY" if row.quarter else ""))
             if row.biz_verdict else "Not evaluated"),
        dict(stage="🎯 Setup", ok=bool(row.s_setup),
             why=f"Technicals {row.technicals or '—'}" + (f"; SIMBA signal {f(row.simba_days)} session(s) ago" if pd.notna(row.simba_days) else "")
             + (f"; stop {f(row.risk_pct, '{:.1f}')}% away" + (" — too wide" if row.signal and row.risk_pct > MAX_RISK_PCT else "")
                if pd.notna(row.risk_pct) else "; no valid stop")),
        dict(stage="💰 Value", ok=None if pd.isna(row.val_pct) else bool(row.s_value),
             why="Not valued by the model" if pd.isna(row.val_pct) else
             f"Price at the {f(row.val_pct)}th percentile of the value range; mid value ₹{f(row.val_p50, '{:,.0f}')} ({f(row.val_upside, '{:+.0f}')}%); {row.val_rating}"),
    ]
    return out
