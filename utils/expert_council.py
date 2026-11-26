"""🏛️ Expert Council Engine — Grounded Dossier + 2-Round Adversarial 3-Expert Debate.

Codifies the empirical playbooks discovered across the UltraTech/Kalyan and
11-Multibagger time-marches into a reusable, stock-agnostic investment committee:

  1. Momentum & Microstructure Trader (`mom_trader`)
  2. GARP & Valuation-Ladder Investor (`garp_investor`)
  3. Macro-Regime & Hybrid Contrarian PM (`hybrid_pm`)

Supports two execution modes:
  - `run_quant_council(dossier)`: Instant, deterministic 2-round adversarial
    engine (zero API key required; never unavailable on Streamlit Cloud).
  - `run_llm_council(dossier, ...)`: Live 2-round parallel LLM debate via
    `utils.llm_client` (OpenAI-compatible) or `google-generativeai` (Gemini),
    strictly grounded in the same factual dossier JSON.
"""
from __future__ import annotations

import concurrent.futures
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, (np.floating, float)):
        return round(float(v), 2) if np.isfinite(v) else None
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


def _num(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _fmt_inr(v: Any, decimals: int = 0) -> str:
    x = _num(v)
    if x is None:
        return "—"
    return f"₹{x:,.{decimals}f}"


def _parse_period_end(label: Optional[str]) -> Optional[pd.Timestamp]:
    """Parse quarterly SHP labels like 'Sep 2026' -> 2026-09-30."""
    if not label:
        return None
    try:
        ts = pd.to_datetime(f"01 {label}", format="%d %b %Y", errors="coerce")
        if pd.notna(ts):
            return ts + pd.offsets.MonthEnd(0)
        ts2 = pd.to_datetime(label, errors="coerce")
        return pd.Timestamp(ts2) if pd.notna(ts2) else None
    except Exception:
        return None


def resolve_ticker(
    query: str,
    df: Optional[pd.DataFrame] = None,
    repo: str = ".",
) -> Tuple[str, str]:
    """Resolve a company name fragment or NSE ticker -> ('TICKER.NS', 'Company Name')."""
    raw = str(query or "").strip()
    if not raw:
        raise ValueError("Empty stock query")
    q = raw.upper()

    mm = df
    if mm is None or mm.empty:
        mm_files = sorted(glob.glob(os.path.join(repo, "data", "cache", "market_master_*.parquet")))
        if mm_files:
            mm = pd.read_parquet(mm_files[-1])

    if mm is not None and not mm.empty and "ticker" in mm.columns:
        t_up = mm["ticker"].fillna("").astype(str).str.upper()
        # 1. Exact ticker match
        hit = mm[t_up.isin([q, f"{q}.NS"])]
        if not hit.empty:
            return str(hit.iloc[0]["ticker"]), str(hit.iloc[0].get("name") or hit.iloc[0]["ticker"])

        # 2. Multi-token substring match on name / ticker
        q_toks = [tok for tok in re.split(r"[^A-Z0-9]+", q) if len(tok) >= 3]
        if q_toks:
            for col in ("name", "ticker"):
                if col not in mm.columns:
                    continue
                s = mm[col].fillna("").astype(str).str.upper()
                hit_all = mm[s.apply(lambda x: all(tok in x for tok in q_toks))]
                if not hit_all.empty:
                    return str(hit_all.iloc[0]["ticker"]), str(hit_all.iloc[0].get("name") or hit_all.iloc[0]["ticker"])
                hit_first = mm[s.str.contains(re.escape(q_toks[0]), na=False)]
                if not hit_first.empty:
                    return str(hit_first.iloc[0]["ticker"]), str(hit_first.iloc[0].get("name") or hit_first.iloc[0]["ticker"])

    # 3. Check extended universe dictionary if present
    try:
        from utils.nifty1000_list import EXTENDED
        cand = q if q.endswith(".NS") else f"{q}.NS"
        if cand in EXTENDED:
            return cand, str(EXTENDED[cand].get("name") or cand)
        q_toks = [tok for tok in re.split(r"[^A-Z0-9]+", q) if len(tok) >= 3]
        if q_toks:
            for tk, meta in EXTENDED.items():
                nm = str(meta.get("name", "")).upper()
                if all(tok in nm for tok in q_toks):
                    return tk, str(meta.get("name") or tk)
    except Exception:
        pass

    sym = q.replace(".NS", "").replace(".BO", "").split()[0]
    return f"{sym}.NS", raw


# ---------------------------------------------------------------------------
# 1. Grounded Dossier Builder (Works both inside Streamlit and standalone CLI)
# ---------------------------------------------------------------------------
def _load_indices(repo: str) -> Tuple[pd.DataFrame, pd.Series]:
    idx_monthly = sorted(glob.glob(os.path.join(repo, "data", "nse", "indices", "*.parquet")))
    idx_daily = sorted(glob.glob(os.path.join(repo, "data", "nse", "indices", "d", "*.parquet")))
    idx_files = idx_monthly[-15:] + idx_daily[-30:]
    parts = []
    for f in idx_files:
        try:
            parts.append(pd.read_parquet(f, columns=["date", "index", "open", "high", "low", "close"]))
        except Exception:
            continue
    if not parts:
        return pd.DataFrame(), pd.Series(dtype=float)
    ix = pd.concat(parts, ignore_index=True)
    ix["date"] = pd.to_datetime(ix["date"], errors="coerce")
    nifty_df = (
        ix[ix["index"] == "Nifty 50"]
        .dropna(subset=["date", "close"])
        .drop_duplicates("date", keep="last")
        .set_index("date")
        .sort_index()
        .rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    )
    vix_s = (
        ix[ix["index"] == "India VIX"]
        .dropna(subset=["date", "close"])
        .drop_duplicates("date", keep="last")
        .set_index("date")
        .sort_index()["close"]
    )
    return nifty_df, vix_s


def build_dossier(
    query: str,
    *,
    df: Optional[pd.DataFrame] = None,
    nifty_df: Optional[pd.DataFrame] = None,
    repo: str = ".",
) -> Dict[str, Any]:
    """Assemble the complete, strictly grounded telemetry dossier for a stock."""
    from utils import (
        forensics as fx,
        fund_trends as ft,
        funnel,
        funnel_ui,
        industry_board as ib,
        nse_data as nd,
        nse_prices,
        regime_badge as rb,
        regime_manager as rm,
        valuation as V,
    )

    repo = os.path.abspath(repo)
    mm = df
    if mm is None or mm.empty:
        mm_files = sorted(glob.glob(os.path.join(repo, "data", "cache", "market_master_*.parquet")))
        mm = pd.read_parquet(mm_files[-1]) if mm_files else pd.DataFrame()

    ticker, name = resolve_ticker(query, df=mm, repo=repo)
    sym = ticker.replace(".NS", "").replace(".BO", "")

    mm_row: Dict[str, Any] = {}
    if mm is not None and not mm.empty and "ticker" in mm.columns and (mm["ticker"] == ticker).any():
        mm_row = mm[mm["ticker"] == ticker].iloc[0].to_dict()

    board = ib.board(mm) if (mm is not None and not mm.empty) else pd.DataFrame()
    val_path = os.path.join(repo, "data", "valuation_snapshot.csv")
    val_df = funnel.load_valuation(val_path) if os.path.exists(val_path) else pd.DataFrame()
    fn_tbl = funnel.table(mm, board, val_df) if (mm is not None and not mm.empty) else pd.DataFrame()
    fn_row = fn_tbl[fn_tbl["ticker"] == ticker].iloc[0] if (not fn_tbl.empty and (fn_tbl["ticker"] == ticker).any()) else None
    ind_view = ib.stock_industry_view(board, ticker) if not board.empty else {}

    # 2. Price history (NSE adjusted, fallback to Yahoo via yf_safe if absent)
    hist = nse_prices.history([ticker])
    px_df = hist.get(ticker)
    if px_df is None or px_df.empty:
        try:
            from utils.yf_safe import safe_history
            px_df = safe_history(ticker, period="2y")
        except Exception:
            px_df = None
    if px_df is None or px_df.empty or len(px_df) < 30:
        raise ValueError(f"Insufficient price history found for {ticker}")
    px_df = px_df.sort_index()

    r_fx = fx.analyse(px_df)
    ev_df = fx.events(px_df, look=60)
    plan = fx.plan(px_df)

    c = px_df["Close"].astype(float)
    px = float(c.iloc[-1])
    last_dt = pd.Timestamp(px_df.index[-1])
    ret = lambda n: round((px / float(c.iloc[-1 - n]) - 1.0) * 100.0, 1) if len(c) > n else None
    hi52 = float(px_df["High"].tail(252).max())
    lo52 = float(px_df["Low"].tail(252).min())
    tmed20_cr = round(float((px_df["Close"] * px_df["Volume"]).tail(20).median() / 1e7), 2)
    ma20 = float(c.rolling(20).mean().iloc[-1])
    ma50 = float(c.rolling(50).mean().iloc[-1])
    ma200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else np.nan
    stage = (
        2 if (np.isfinite(ma200) and px > ma50 > ma200)
        else 4 if (np.isfinite(ma200) and px < ma50 < ma200)
        else 1 if px > ma50
        else 3
    )

    # 3. Market Regime + Live VIX + RECOVERY + Market Context
    idx_nifty, vix_s = _load_indices(repo)
    use_nifty = idx_nifty if (idx_nifty is not None and len(idx_nifty) >= 200) else nifty_df
    if use_nifty is not None and not use_nifty.empty and getattr(use_nifty.index, "tz", None) is not None:
        use_nifty = use_nifty.copy()
        use_nifty.index = use_nifty.index.tz_localize(None)
    reg_info = rb.compute(use_nifty, mm, vix_df=(vix_s if not vix_s.empty else None), root=repo)
    rec_sig = rm.load_recovery_signals(root=repo)
    mc = nd.market_context()
    mc_last = mc.iloc[-1].to_dict() if (mc is not None and not mc.empty) else {}

    rs_1m = rs_3m = rs_6m = None
    if use_nifty is not None and not use_nifty.empty and "Close" in use_nifty.columns:
        c_naive = c.copy()
        if getattr(c_naive.index, "tz", None) is not None:
            c_naive.index = c_naive.index.tz_localize(None)
        rs_df = pd.concat([c_naive.rename("s"), use_nifty["Close"].astype(float).rename("n")], axis=1).dropna()

        def _rs(w: int) -> Optional[float]:
            if len(rs_df) <= w:
                return None
            return round(((rs_df["s"].iloc[-1] / rs_df["s"].iloc[-1 - w]) / (rs_df["n"].iloc[-1] / rs_df["n"].iloc[-1 - w]) - 1.0) * 100.0, 1)

        rs_1m, rs_3m, rs_6m = _rs(21), _rs(63), _rs(126)

    # 4. Fundamentals + Annual 5Y Cash History + Quarterly 6Q + Split SHP
    biz = ft.read(ticker, root=repo)
    brief = ft.brief(ticker, root=repo)
    q_df = ft.quarterly(ticker, root=repo)
    im = ft.inst_map(root=repo).get(ticker, {})

    ann_raw, _ = ft._data(repo)
    cash_hist: List[Dict[str, Any]] = []
    k_ann = ft._key(ticker, ann_raw)
    if k_ann is not None:
        w = ann_raw[k_ann].pivot_table(index="period", columns="field", values="value", aggfunc="last").sort_index()
        w = w[w.index.month == 3]
        if not w.empty:
            g = lambda col: w[col] if col in w.columns else pd.Series(np.nan, index=w.index)
            gross = g("fixed_assets") + g("cwip").fillna(0.0)
            capex = gross.diff() + g("dep").fillna(0.0)
            for dt_i in w.index[-5:]:
                sales_v = g("sales").loc[dt_i]
                op_v = g("op").loc[dt_i]
                np_v = g("np").loc[dt_i]
                cfo_v = g("cfo").loc[dt_i]
                cap_v = capex.loc[dt_i]
                roce_v = g("roce").loc[dt_i]
                borr_v = g("borrowings").loc[dt_i]
                eq_cap = g("eq_cap").loc[dt_i]
                resv = g("reserves").loc[dt_i]
                eq_v = (0.0 if pd.isna(eq_cap) else float(eq_cap)) + (0.0 if pd.isna(resv) else float(resv))
                cfo_p = round(float(cfo_v / np_v), 2) if (pd.notna(cfo_v) and pd.notna(np_v) and np_v > 0) else None
                fcf_p = round(float((cfo_v - max(0.0, cap_v)) / np_v), 2) if (pd.notna(cfo_v) and pd.notna(cap_v) and pd.notna(np_v) and np_v > 0) else None
                cash_hist.append({
                    "fy": pd.Timestamp(dt_i).strftime("FY%y"),
                    "sales_cr": _safe(sales_v),
                    "opm_pct": round(float(op_v / sales_v * 100.0), 1) if (pd.notna(op_v) and pd.notna(sales_v) and sales_v > 0) else None,
                    "np_cr": _safe(np_v),
                    "cfo_cr": _safe(cfo_v),
                    "capex_cr": _safe(cap_v),
                    "cfo_pat": cfo_p,
                    "fcf_pat": fcf_p,
                    "roce_pct": _safe(roce_v),
                    "de_ratio": round(float(borr_v / eq_v), 2) if (pd.notna(borr_v) and eq_v > 0) else None,
                })

    q_hist: List[Dict[str, Any]] = []
    if q_df is not None and not q_df.empty:
        for dt_i, row in q_df.tail(6).iterrows():
            q_hist.append({
                "qtr": pd.Timestamp(dt_i).strftime("%b %Y"),
                "sales_cr": _safe(row.get("sales")),
                "sales_yoy": _safe(row.get("sales_yoy")),
                "opm_pct": _safe(row.get("opm")),
                "np_cr": _safe(row.get("np")),
                "np_yoy": _safe(row.get("np_yoy")),
                "prom_pct": _safe(row.get("promoters")),
                "fii_pct": _safe(row.get("fii")),
                "dii_pct": _safe(row.get("dii")),
            })

    # 5. Live Valuation v4.1 (handles both core and extended stocks cleanly)
    val_summary: Dict[str, Any] = {}
    try:
        val_ctx = V.load_context(extended=(not bool(mm_row)))
        val_res = V.value_stock(sym, val_ctx, price=px, hi52=hi52, lo52=lo52, knobs=True)
        if val_res.get("status") != "OK" and bool(mm_row):
            val_ctx_ext = V.load_context(extended=True)
            val_res = V.value_stock(sym, val_ctx_ext, price=px, hi52=hi52, lo52=lo52, knobs=True)
        if val_res.get("status") == "OK":
            bl = val_res.get("blends") or {}
            d_ = val_res["dist"]
            m_ = val_res["multiple"]
            val_summary = {
                "status": "OK",
                "rating": val_res.get("rating"),
                "verdict_raw": val_res.get("verdict_raw"),
                "provisional": val_res.get("provisional"),
                "identity": val_res.get("identity"),
                "confidence": val_res.get("confidence"),
                "gates_ok": bool(val_res.get("gates_ok", False)),
                "failed_gates": [k for k in "ABCEF" if not val_res["gates"][k]["ok"]] + ([] if val_res["sanity_1f"]["ok"] else ["1F"]),
                "cmp_pct": round(float(d_["cmp_pct"]), 1),
                "p10": _safe(d_["P10"]),
                "p50": _safe(d_["P50"]),
                "p90": _safe(d_["P90"]),
                "upside_p50_pct": round((float(d_["P50"]) / px - 1.0) * 100.0, 1),
                "bear_blend": _safe(bl.get("bear")),
                "base_blend": _safe(bl.get("base")),
                "bull_blend": _safe(bl.get("bull")),
                "cmp_vs_bear_pct": round((px / float(bl["bear"]) - 1.0) * 100.0, 1) if np.isfinite(bl.get("bear", np.nan)) else None,
                "mult_kind": m_.get("kind"),
                "spot_mult": _safe(m_.get("spot")),
                "base_mult": _safe(m_.get("base")),
                "peer_median_mult": _safe(m_.get("sector_median")),
                "cagr_bear": _safe(val_res["cagr"]["bear"]),
                "cagr_base": _safe(val_res["cagr"]["base"]),
                "cagr_bull": _safe(val_res["cagr"]["bull"]),
                "band_firmness": (val_res.get("band") or {}).get("firmness"),
                "quality_grade": val_res["quality"]["grade"],
                "quality_note": val_res["quality"]["note"],
                "quality_lean": val_res["quality"]["lean"],
                "derating_triggered": bool(val_res["gates"]["D"]["triggered"]),
                "counterfactual_px": _safe(val_res["gates"]["D"].get("counterfactual")),
            }
        else:
            val_summary = {"status": val_res.get("status", "UNAVAILABLE"), "reason": val_res.get("reason", "")}
    except Exception as exc:
        val_summary = {"status": "ERROR", "reason": str(exc)}

    # 6. NSE Delivery + Bulk/Block Deals
    deliv_info: Dict[str, Any] = {}
    try:
        att = nd.attach(pd.DataFrame({"ticker": [ticker]}))
        if att is not None and not att.empty:
            dp = _safe(att["deliv_pct"].iloc[0]) if "deliv_pct" in att else None
            da = _safe(att["deliv_avg20"].iloc[0]) if "deliv_avg20" in att else None
            deliv_info = {
                "deliv_pct": dp,
                "deliv_avg20": da,
                "deliv_ratio": round(dp / da, 2) if (dp and da and da > 0) else None,
                "abs_deliv_cr": round(tmed20_cr * dp / 100.0, 2) if (dp and tmed20_cr) else None,
                "nse_flags": str(att["nse_flags"].iloc[0]) if ("nse_flags" in att and pd.notna(att["nse_flags"].iloc[0])) else "",
            }
    except Exception:
        pass
    deals_90 = nd.deal_tally(90).get(ticker, {}) if hasattr(nd, "deal_tally") else {}
    deals_raw = nd.deals_for([ticker], days=90) if hasattr(nd, "deals_for") else pd.DataFrame()
    deals_list = deals_raw.tail(6).to_dict(orient="records") if (deals_raw is not None and not deals_raw.empty) else []
    for dl in deals_list:
        for k, v in list(dl.items()):
            dl[k] = _safe(v)

    # 7. Recent 60d Anchors + Post-SHP Liquidation Audit + Funnel Chips
    chips = funnel_ui.build_signal_chips(fn_row) if fn_row is not None else {}
    recent_anchors: List[Dict[str, Any]] = []
    if ev_df is not None and not ev_df.empty:
        for _, er in ev_df.tail(8).iterrows():
            recent_anchors.append({
                "date": pd.Timestamp(er["date"]).strftime("%Y-%m-%d"),
                "volx": round(float(er["vol_ratio"]), 2),
                "clv": round(float(er["clv"]), 2),
                "dir": str(er["direction"]),
                "close": round(float(er["close"]), 2),
                "turn_cr": round(float(er["turnover_cr"]), 1),
                "state": str(er["state"]),
                "is_anchor": bool(er["anchor"]),
            })

    shp_end = _parse_period_end(im.get("period"))
    recent_bear_bars = [
        a for a in recent_anchors
        if ("-1" in a["dir"] or "BEAR" in a["state"].upper())
        and a["clv"] < 0.35
        and (pd.Timestamp(last_dt) - pd.Timestamp(a["date"])).days <= 30
    ]
    post_shp_bear_bars = [
        a for a in recent_bear_bars
        if shp_end is None or pd.Timestamp(a["date"]) > shp_end
    ]
    bear_turn_cr = round(sum(float(a["turn_cr"]) for a in post_shp_bear_bars), 1)
    bear_turn_mult = round(bear_turn_cr / tmed20_cr, 1) if (tmed20_cr and tmed20_cr > 0 and bear_turn_cr > 0) else 0.0

    atr_val = _num(r_fx.get("atr"))
    atr_pct = round(atr_val / px * 100.0, 2) if (atr_val and px > 0) else None
    gap_guard_pct = round(max(5.0, 1.5 * atr_pct), 1) if atr_pct is not None else 5.0

    return {
        "ticker": ticker,
        "name": name,
        "as_of": last_dt.strftime("%Y-%m-%d"),
        "market_regime": {
            "nifty_regime": reg_info["nifty"],
            "breadth_regime": reg_info["breadth"],
            "engine_mode": reg_info["engine"].get("mode"),
            "engine_status": rb.engine_status(reg_info["engine"]),
            "vix": reg_info.get("vix"),
            "recovery_signals": rec_sig,
            "nifty_pe": _safe(mc_last.get("nifty_pe")),
            "fii_idx_long_pct": _safe(mc_last.get("fii_idx_long_pct")),
            "fii_net_cash": _safe(mc_last.get("fii_net")),
            "dii_net_cash": _safe(mc_last.get("dii_net")),
        },
        "price_and_trend": {
            "close": round(px, 2),
            "stage": stage,
            "ma20": round(ma20, 2),
            "ma50": round(ma50, 2),
            "ma200": _safe(ma200),
            "dist_ma20_pct": round((px / ma20 - 1.0) * 100.0, 1),
            "dist_ma50_pct": round((px / ma50 - 1.0) * 100.0, 1),
            "dist_ma200_pct": round((px / ma200 - 1.0) * 100.0, 1) if np.isfinite(ma200) else None,
            "hi52": round(hi52, 2),
            "lo52": round(lo52, 2),
            "dist_52w_hi_pct": round((px / hi52 - 1.0) * 100.0, 1),
            "from_52w_lo_pct": round((px / lo52 - 1.0) * 100.0, 1),
            "ret_1w": ret(5),
            "ret_1m": ret(21),
            "ret_3m": ret(63),
            "ret_6m": ret(126),
            "ret_1y": ret(250),
            "rs_1m": rs_1m,
            "rs_3m": rs_3m,
            "rs_6m": rs_6m,
            "comp_rs": _safe(mm_row.get("comp_rs")),
            "scanner_signal": mm_row.get("dna_signal"),
            "liq_20d_med_cr": tmed20_cr,
            "mcap_cr": _safe(mm_row.get("marketCap") / 1e7) if pd.notna(mm_row.get("marketCap", np.nan)) else None,
            "pe_trailing": _safe(mm_row.get("pe", mm_row.get("trailingPE"))),
        },
        "forensics_v6": {
            "gate1": bool(r_fx.get("gate1", False)),
            "state": r_fx.get("state"),
            "action": r_fx.get("action"),
            "entry_guard": r_fx.get("entry_guard", ""),
            "anchor_date": _safe(r_fx.get("anchor_date")),
            "anchor_volx": _safe(r_fx.get("anchor_volx")),
            "anchor_clv": _safe(r_fx.get("anchor_clv")),
            "anchor_low": _safe(r_fx.get("anchor_low")),
            "anchor_close": _safe(r_fx.get("anchor_close")),
            "days_since_anchor": _safe(r_fx.get("days")),
            "retention": _safe(r_fx.get("retention")),
            "signature": r_fx.get("signature"),
            "shakeout": bool(r_fx.get("shakeout", False)),
            "gap": bool(r_fx.get("gap", False)),
            "supply": r_fx.get("supply"),
            "shelf_low": _safe(r_fx.get("shelf_low")),
            "shelf_top": _safe(r_fx.get("shelf_top")),
            "entry": _safe(r_fx.get("entry")),
            "stop": _safe(r_fx.get("stop")),
            "stop_pct": _safe(r_fx.get("stop_pct")),
            "half_size": bool(r_fx.get("half_size", False)),
            "atr": _safe(atr_val),
            "atr_pct": atr_pct,
            "ext_anc_atr": _safe(r_fx.get("ext_atr")),
            "ext_ma50_atr": _safe(r_fx.get("ext_ma50_atr")),
            "atr_gap_guard_pct": gap_guard_pct,
            "rsi": _safe(r_fx.get("rsi")),
            "adx": _safe(r_fx.get("adx")),
            "pdi": _safe(r_fx.get("pdi")),
            "ndi": _safe(r_fx.get("ndi")),
            "bbp": _safe(r_fx.get("bbp")),
            "recent_anchors_60d": recent_anchors,
            "post_shp_bear_bars": post_shp_bear_bars,
            "post_shp_bear_turn_cr": bear_turn_cr,
            "post_shp_bear_turn_mult": bear_turn_mult,
            "trig_a": plan.get("trig_a") if plan else None,
            "trig_b": plan.get("trig_b") if plan else None,
        },
        "industry_and_funnel": {
            "industry": mm_row.get("industry", ind_view.get("industry")),
            "ind_group": ind_view.get("group"),
            "ind_status": ind_view.get("status"),
            "ind_score": _safe(ind_view.get("score")),
            "ind_rank": f"{ind_view.get('rank')}/{ind_view.get('of')}" if ind_view.get("rank") else None,
            "funnel_score": _safe(fn_row.get("funnel_score")) if fn_row is not None else None,
            "funnel_grade": fn_row.get("funnel_grade") if fn_row is not None else None,
            "score_note": fn_row.get("score_note") if fn_row is not None else None,
            "funnel_stages": {k: _safe(fn_row.get(f"s_{k}")) for k in ("industry", "trend", "business", "setup", "value")} if fn_row is not None else {},
            "chips_pros": [ch["text"] for ch in chips.get("pros", [])],
            "chips_cons": [ch["text"] for ch in chips.get("cons", [])],
        },
        "fundamentals_and_flows": {
            "biz_verdict": brief.get("verdict"),
            "biz_score": _safe(brief.get("score")),
            "biz_headline": brief.get("headline"),
            "biz_flags": list(brief.get("flags") or []),
            "investing_phase": biz.get("investing"),
            "pillars": {k: {"score": _safe(v.get("score")), "dir": _safe(v.get("dir"))} for k, v in (biz.get("pillars") or {}).items()},
            "annual_cash_history": cash_hist,
            "quarterly_history": q_hist,
            "institutional_shp": {k: _safe(v) for k, v in im.items()},
            "delivery_and_flags": deliv_info,
            "deals_90d_tally": {k: _safe(v) for k, v in deals_90.items()},
            "deals_90d_recent": deals_list,
        },
        "valuation_v41": val_summary,
    }


# ---------------------------------------------------------------------------
# 2. Deterministic Quant Expert Council (2-Round Adversarial Engine, 0 API Key)
# ---------------------------------------------------------------------------
def _cash_profile(cash_hist: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Audit FY CFO/PAT inflection vs multi-year cash consistency."""
    valid = [r for r in cash_hist if _num(r.get("cfo_pat")) is not None]
    if not valid:
        return {
            "has_data": False, "latest_fy": None, "latest_cfo_pat": None, "latest_fcf_pat": None,
            "latest_roce": None, "latest_de": None, "neg_prior_years": 0, "inflection_only_1y": False,
            "passes_primary_gate": False, "proven_multi_year": False,
        }
    latest = valid[-1]
    prior = valid[:-1]
    cfo_p = _num(latest.get("cfo_pat"), 0.0)
    neg_prior = sum(1 for r in prior if (_num(r.get("cfo_cr"), 0.0) or 0.0) < 0)
    passes = cfo_p is not None and cfo_p >= 0.80
    prev_ok = bool(prior and (_num(prior[-1].get("cfo_pat"), -9.0) or -9.0) >= 0.80)
    return {
        "has_data": True,
        "latest_fy": latest.get("fy"),
        "latest_cfo_pat": cfo_p,
        "latest_fcf_pat": _num(latest.get("fcf_pat")),
        "latest_cfo_cr": _num(latest.get("cfo_cr")),
        "latest_np_cr": _num(latest.get("np_cr")),
        "latest_roce": _num(latest.get("roce_pct")),
        "latest_de": _num(latest.get("de_ratio")),
        "prev_de": _num(prior[-1].get("de_ratio")) if prior else None,
        "neg_prior_years": neg_prior,
        "prior_count": len(prior),
        "passes_primary_gate": passes,
        "inflection_only_1y": bool(passes and not prev_ok and neg_prior >= 1),
        "proven_multi_year": bool(passes and prev_ok),
    }


def _vote_summary(verdicts: List[str]) -> Tuple[str, str]:
    """Classify a 3-member vote into a chair verdict and tally label."""
    buy_like = {"BUY NOW", "STAGED BUY / ACCUMULATE"}
    n_buy = sum(1 for v in verdicts if v in buy_like)
    n_wait = sum(1 for v in verdicts if v == "WAIT FOR TRIGGER (WATCHLIST)")
    n_avoid = sum(1 for v in verdicts if v == "AVOID / SELL")
    if n_buy == 3:
        top = "BUY NOW" if verdicts.count("BUY NOW") >= 2 else "STAGED BUY / ACCUMULATE"
        return top, f"3–0 {top}"
    if n_buy == 2:
        top = "BUY NOW" if verdicts.count("BUY NOW") >= 2 else "STAGED BUY / ACCUMULATE"
        return top, f"2–1 {top}"
    if n_avoid >= 2:
        return "AVOID / SELL", f"{n_avoid}–{3 - n_avoid} AVOID / SELL"
    if n_wait >= 2:
        return "WAIT FOR TRIGGER (WATCHLIST)", f"{n_wait}–{3 - n_wait} WAIT FOR TRIGGER"
    return "WAIT FOR TRIGGER (WATCHLIST)", "1–1–1 SPLIT (DEFAULT WAIT)"


def run_quant_council(dossier: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the deterministic 2-round adversarial 3-expert council on `dossier`."""
    pt = dossier["price_and_trend"]
    fx = dossier["forensics_v6"]
    mr = dossier["market_regime"]
    ind = dossier["industry_and_funnel"]
    ff = dossier["fundamentals_and_flows"]
    val = dossier.get("valuation_v41") or {}

    px = float(pt["close"])
    stage = int(pt.get("stage") or 0)
    ma20 = _num(pt.get("ma20"), px) or px
    ma50 = _num(pt.get("ma50"), px) or px
    ma200 = _num(pt.get("ma200"))
    hi52 = _num(pt.get("hi52"), px) or px
    atr = _num(fx.get("atr"), px * 0.03) or (px * 0.03)
    atr_pct = _num(fx.get("atr_pct"), 3.0) or 3.0
    gap_guard = _num(fx.get("atr_gap_guard_pct"), max(5.0, round(1.5 * atr_pct, 1))) or 5.0
    liq = _num(pt.get("liq_20d_med_cr"), 0.0) or 0.0
    comp_rs = _num(pt.get("comp_rs"))
    rs_1m = _num(pt.get("rs_1m"))
    rs_6m = _num(pt.get("rs_6m"))
    adx = _num(fx.get("adx"))
    pdi = _num(fx.get("pdi"))
    ndi = _num(fx.get("ndi"))
    rsi = _num(fx.get("rsi"))
    gate1 = bool(fx.get("gate1"))
    fx_state = str(fx.get("state") or "UNKNOWN")
    fx_action = str(fx.get("action") or "WATCH")
    anchor_low = _num(fx.get("anchor_low"))
    anchor_close = _num(fx.get("anchor_close"))
    bear_bars = fx.get("post_shp_bear_bars") or []
    bear_turn_cr = _num(fx.get("post_shp_bear_turn_cr"), 0.0) or 0.0
    bear_turn_mult = _num(fx.get("post_shp_bear_turn_mult"), 0.0) or 0.0
    has_post_shp_dump = len(bear_bars) >= 1 and bear_turn_mult >= 2.5
    max_bear_close = max((_num(b.get("close"), 0.0) or 0.0) for b in bear_bars) if bear_bars else 0.0

    engine_mode = str(mr.get("engine_mode") or "UNKNOWN").upper()
    trail_pct = 15 if engine_mode == "BULL" else (10 if engine_mode == "DEFENSIVE" else 12)
    vix_info = mr.get("vix") or {}
    vix_close = _num(vix_info.get("close"))
    vix_pctile = _num(vix_info.get("pctile_252"))
    vix_high = (vix_pctile is not None and vix_pctile >= 80.0) or (vix_close is not None and vix_close >= 17.0)
    breadth = mr.get("breadth_regime") or {}
    breadth_pct = _num(breadth.get("pct"))
    nifty_reg = mr.get("nifty_regime") or {}
    rec = mr.get("recovery_signals") or {}
    rec_active = bool(rec.get("active"))
    fii_net = _num(mr.get("fii_net_cash"), 0.0) or 0.0
    dii_net = _num(mr.get("dii_net_cash"), 0.0) or 0.0
    inst_cash_cr = round(fii_net + dii_net, 0)
    fii_long = _num(mr.get("fii_idx_long_pct"))

    cash = _cash_profile(ff.get("annual_cash_history") or [])
    q_hist = ff.get("quarterly_history") or []
    latest_q = q_hist[-1] if q_hist else {}
    prev_q = q_hist[-2] if len(q_hist) >= 2 else {}
    q_np_yoy = _num(latest_q.get("np_yoy"))
    q_sales_yoy = _num(latest_q.get("sales_yoy"))
    q_opm = _num(latest_q.get("opm_pct"))
    prev_opm = _num(prev_q.get("opm_pct"))
    opm_dip_bps = round((prev_opm - q_opm) * 100) if (q_opm is not None and prev_opm is not None and prev_opm > q_opm) else 0
    shp = ff.get("institutional_shp") or {}
    d_inst_q = _num(shp.get("d_inst_q"))
    d_prom_q = _num(shp.get("d_prom_q"))
    deliv = ff.get("delivery_and_flags") or {}
    deliv_pct = _num(deliv.get("deliv_pct"))
    deliv_ratio = _num(deliv.get("deliv_ratio"))
    nse_flags = str(deliv.get("nse_flags") or "")
    biz_score = _num(ff.get("biz_score"))
    biz_flags = ff.get("biz_flags") or []

    val_ok = val.get("status") == "OK"
    val_rating = str(val.get("rating") or "N/A")
    val_raw = str(val.get("verdict_raw") or val_rating)
    bear_blend = _num(val.get("bear_blend"))
    p10 = _num(val.get("p10"))
    p50 = _num(val.get("p50"))
    cf_px = _num(val.get("counterfactual_px"))
    cmp_pct = _num(val.get("cmp_pct"))
    cmp_vs_bear = _num(val.get("cmp_vs_bear_pct"))
    q_grade = str(val.get("quality_grade") or "")
    q_note = str(val.get("quality_note") or "")
    spot_mult = _num(val.get("spot_mult"))
    base_mult = _num(val.get("base_mult"))
    peer_mult = _num(val.get("peer_median_mult"))
    mult_caution = q_grade.lower() == "caution" or "multiple recovery" in q_note.lower()

    # Key price landmarks
    ma50_reclaim_px = round(max(ma50, ma20) * 1.002, 1)
    bear_clear_px = round(max(max_bear_close, ma50_reclaim_px) + 1.0, 1) if max_bear_close > 0 else round(ma50_reclaim_px * 1.02, 1)
    vol_req = "1.8×" if liq >= 150.0 else "3.0×"
    gap_cap_px = round(bear_clear_px * (1.0 + gap_guard / 100.0), 1)
    cliff_dist_pct = round((px / anchor_low - 1.0) * 100.0, 1) if (anchor_low and anchor_low > 0) else None

    # =======================================================================
    # ROUND 1: INDEPENDENT SPECIALIST STANCES
    # =======================================================================
    # --- 1. Momentum & Microstructure Trader (Round 1) ---
    mom_ev: List[str] = []
    mom_rf: List[str] = []
    if gate1 and fx_action == "BUY NOW" and stage == 2 and not has_post_shp_dump and not nse_flags:
        mom_r1_verdict = "BUY NOW"
        mom_r1_conv = 8 if (comp_rs is not None and comp_rs > 5) else 7
        mom_r1_w = 3.0 if engine_mode in ("CAUTION", "DEFENSIVE") else 5.0
        mom_r1_max = 5.0 if engine_mode in ("CAUTION", "DEFENSIVE") else 7.0
        mom_r1_stop = round(_num(fx.get("stop"), px - 1.8 * atr) or (px - 1.8 * atr), 1)
        mom_r1_action = f"Initiate {mom_r1_w:.1f}% Stage-2 breakout entry at {_fmt_inr(px, 1)} (Forensics GO / BUY NOW, Gate-1=True) with hard stop at {_fmt_inr(mom_r1_stop, 1)}."
        mom_r1_trig = f"Active entry inside {gap_guard:.1f}% ATR gap guard; scale to {mom_r1_max:.1f}% on tight continuation above {_fmt_inr(px * 1.03, 1)}."
    elif stage == 4 or (not gate1 and ma200 and px < ma200 and (comp_rs or 0) < -10):
        mom_r1_verdict = "AVOID / SELL"
        mom_r1_conv = 9
        mom_r1_w = 0.0
        mom_r1_max = 0.0
        mom_r1_stop = round(ma50, 1)
        mom_r1_action = f"Hold 0% (Avoid); Stage-{stage} downtrend below MA200 ({_fmt_inr(ma200, 1)}) with weak RS ({comp_rs:+.1f} if comp_rs is not None else 'N/A')."
        if comp_rs is None:
            mom_r1_action = f"Hold 0% (Avoid); Stage-{stage} breakdown below MA200 ({_fmt_inr(ma200, 1)}) and Gate-1=False."
        mom_r1_trig = f"No long setup until stock reclaims MA200 ({_fmt_inr(ma200, 1)}) and MA50 ({_fmt_inr(ma50, 1)}) with a clean {vol_req} bull anchor."
    else:
        mom_r1_verdict = "WAIT FOR TRIGGER (WATCHLIST)"
        mom_r1_conv = 8 if (fx_state == "VETO" or has_post_shp_dump or not gate1) else 6
        mom_r1_w = 0.0
        mom_r1_max = 4.0 if engine_mode in ("CAUTION", "DEFENSIVE") else 6.0
        trig_px = bear_clear_px if has_post_shp_dump else max(ma50_reclaim_px, round(px * 1.02, 1))
        mom_r1_stop = round(trig_px - 1.5 * atr, 1)
        mom_r1_action = f"Hold 0% position; keep on watchlist under a strict buy-stop above {_fmt_inr(trig_px, 1)} while Stage-{stage} / Forensics {fx_state} (Gate-1={gate1})."
        mom_r1_trig = (
            f"Require daily close > {_fmt_inr(trig_px, 1)} on {vol_req}+ volume & CLV >= 0.50 (restoring Stage-2 / Gate-1=True); "
            f"enforce {gap_guard:.1f}% ATR gap guard (do not chase > {_fmt_inr(round(trig_px * (1 + gap_guard / 100.0), 1), 1)})."
        )

    mom_ev.append(
        f"Tape & Forensics v6: Stage {stage}, CMP {_fmt_inr(px, 1)} ({pt.get('dist_ma50_pct'):+.1f}% vs MA50 {_fmt_inr(ma50, 1)}), "
        f"Gate-1={gate1}, State={fx_state}, Action={fx_action}, Funnel={ind.get('funnel_score')}/{ind.get('funnel_grade')}."
    )
    if has_post_shp_dump:
        mom_ev.append(
            f"Fresh post-SHP institutional liquidation: {len(bear_bars)} heavy bear bar(s) totaling ₹{bear_turn_cr:.1f} Cr "
            f"({bear_turn_mult:.1f}× normal ₹{liq:.1f} Cr median turnover) with CLV < 0.35 right overhead up to {_fmt_inr(max_bear_close, 1)}."
        )
    mom_ev.append(
        f"Momentum internals: CompRS {comp_rs if comp_rs is not None else '—'}, RS 1M {rs_1m if rs_1m is not None else '—'}%, "
        f"ADX {adx if adx is not None else '—'} (+DI {pdi if pdi is not None else '—'} vs -DI {ndi if ndi is not None else '—'}), "
        f"RSI {rsi if rsi is not None else '—'}, Industry {ind.get('industry')} ({ind.get('ind_group')}/{ind.get('ind_status')}, {ind.get('ind_rank')})."
    )
    if cliff_dist_pct is not None and 0 <= cliff_dist_pct <= 3.0 and not gate1:
        mom_rf.append(f"Knife-edge base floor: CMP sits only +{cliff_dist_pct:.1f}% above the {fx.get('anchor_date')} anchor low ({_fmt_inr(anchor_low, 1)}); a break voids the base.")
    if has_post_shp_dump:
        mom_rf.append(f"Post-SHP supply mismatch: ₹{bear_turn_cr:.1f} Cr of heavy selling occurred AFTER the {shp.get('period') or 'latest'} SHP filing, overriding stale institutional ownership counts.")
    if liq < 30:
        mom_rf.append(f"Thin liquidity (₹{liq:.1f} Cr/d, ATR {atr_pct:.1f}%): institutional exits create sharp air-pockets in a {engine_mode} tape.")
    if nse_flags:
        mom_rf.append(f"NSE surveillance flag active: {nse_flags}.")
    if not mom_rf:
        mom_rf.append(f"Regime overhead: Engine={engine_mode}, VIX={vix_close} (P{vix_pctile}), Breadth={breadth.get('regime')} ({breadth_pct:.1f}% >200D)." if breadth_pct is not None else f"Regime overhead: Engine={engine_mode}.")

    mom_r1 = {
        "member": "mom_trader",
        "verdict": mom_r1_verdict,
        "conviction_0_10": mom_r1_conv,
        "target_weight_now_pct": mom_r1_w,
        "max_weight_on_trigger_pct": mom_r1_max,
        "action_now": mom_r1_action,
        "entry_or_trigger_plan": mom_r1_trig,
        "stop_and_invalidation": (
            f"Hard stop {_fmt_inr(mom_r1_stop, 1)} upon trigger; trail via {engine_mode} {trail_pct}% trail once profitable"
            + (f"; close < {_fmt_inr(anchor_low, 1)} drops from active watchlist." if anchor_low else ".")
        ),
        "key_evidence": mom_ev,
        "key_red_flags": mom_rf,
        "preemptive_challenge_to_peers": (
            f"To GARP: Don't treat a Valuation {val_rating} chip as a floor when Gate-1={gate1} and ₹{bear_turn_cr:.1f} Cr of fresh selling sits overhead. "
            f"To Hybrid: Curing {engine_mode} cash drag means rotating to Gate-1=True Stage-2 RS leaders, not forcing capital into a {fx_state} chart."
            if has_post_shp_dump else
            f"To GARP & Hybrid: Price confirms fundamentals—respect Gate-1={gate1}, Forensics {fx_state}, and the {gap_guard:.1f}% ATR gap guard before sizing up."
        ),
    }

    # --- 2. GARP & Valuation-Ladder Investor (Round 1) ---
    garp_ev: List[str] = []
    garp_rf: List[str] = []
    is_val_buy = val_ok and ("BUY" in val_rating.upper() or "BUY" in val_raw.upper())
    near_bear = cmp_vs_bear is not None and cmp_vs_bear <= 15.0
    below_bear = cmp_vs_bear is not None and cmp_vs_bear <= 0.0
    earnings_shrinking = q_np_yoy is not None and q_np_yoy <= 0.0
    strong_growth = (not earnings_shrinking) and ((q_np_yoy is not None and q_np_yoy >= 25.0) or ((biz_score or 0) >= 70))
    is_stage4_knife = stage == 4
    unconfirmed_cap_pct = 10.0
    cf_upside = round((cf_px / px - 1.0) * 100.0, 1) if (cf_px and px > 0) else None
    cf_upside_from_bear = round((cf_px / bear_blend - 1.0) * 100.0, 1) if (cf_px and bear_blend and bear_blend > 0) else None

    if not val_ok:
        garp_r1_verdict = "WAIT FOR TRIGGER (WATCHLIST)"
        garp_r1_conv = 4
        garp_r1_w = 0.0
        garp_r1_max = 3.0
        garp_r1_action = f"Hold 0%; Valuation v4.1 coverage unavailable ({val.get('reason') or val.get('status')})—cannot establish Bear-Blend margin of safety."
        garp_r1_trig = "Wait for Valuation v4.1 coverage or manual intrinsic floor verification with CFO/PAT >= 0.80."
    elif is_val_buy and earnings_shrinking:
        garp_r1_verdict = "WAIT FOR TRIGGER (WATCHLIST)"
        garp_r1_conv = 7
        garp_r1_w = 0.0
        garp_r1_max = 4.0
        garp_r1_action = f"Hold 0%; blocked by GARP earnings-growth floor (latest Q NP YoY = {q_np_yoy:+.1f}% <= 0% — value-trap guardrail)."
        garp_r1_trig = f"Wait for quarterly PAT YoY to turn positive (> 0%) while holding CFO/PAT >= 0.80 before bidding near Bear Blend {_fmt_inr(bear_blend, 0)}."
    elif is_val_buy and not cash["passes_primary_gate"] and cash["has_data"]:
        garp_r1_verdict = "AVOID / SELL" if (cash["latest_cfo_pat"] is not None and cash["latest_cfo_pat"] < 0) else "WAIT FOR TRIGGER (WATCHLIST)"
        garp_r1_conv = 7
        garp_r1_w = 0.0
        garp_r1_max = 4.0 if garp_r1_verdict != "AVOID / SELL" else 0.0
        garp_r1_action = f"Hold 0%; fails primary GARP cash-flow gate ({cash['latest_fy']} CFO/PAT = {cash['latest_cfo_pat']} < 0.80 — HFCL rule)."
        garp_r1_trig = f"Require verified CFO/PAT >= 0.80 and price <= Bear Blend {_fmt_inr(bear_blend, 0)} before initiating."
    elif below_bear and cash["passes_primary_gate"] and strong_growth and is_stage4_knife:
        garp_r1_verdict = "WAIT FOR TRIGGER (WATCHLIST)"
        garp_r1_conv = 6
        garp_r1_w = 0.0
        garp_r1_max = min(6.0, unconfirmed_cap_pct)
        garp_r1_action = (
            f"Hold 0% today despite CMP < Bear Blend ({_fmt_inr(bear_blend, 1)}): Stage-4 falling-knife guardrail blocks blind averaging below MA50 & MA200."
        )
        garp_r1_trig = (
            f"Require Stage-1 base stabilization (close > MA20 {_fmt_inr(ma20, 0)} or high-CLV absorption bar) before entering max 1 pilot tranche (<= 3.0%, total Stage-3/4 cap <= 10%); "
            "enforce 60-day cooldown after any -25% stop-out."
        )
    elif below_bear and cash["passes_primary_gate"] and strong_growth:
        garp_r1_verdict = "BUY NOW"
        garp_r1_conv = 9 if cash["proven_multi_year"] else 8
        garp_r1_w = 4.0 if cash["proven_multi_year"] else 3.0
        garp_r1_max = min(8.0, unconfirmed_cap_pct)
        garp_r1_action = f"Deploy {garp_r1_w:.1f}% immediately below Bear Blend ({_fmt_inr(bear_blend, 1)}) with verified {cash['latest_fy']} CFO/PAT {cash['latest_cfo_pat']:.2f}× and positive PAT growth."
        garp_r1_trig = f"Tranche 1: {garp_r1_w:.1f}% at CMP {_fmt_inr(px, 1)} (<Bear Blend); max 1 add tranche on intact earnings (total Stage-3/4 cap <= 10%, 60-day post-stop cooldown)."
    elif is_val_buy and near_bear and cash["passes_primary_gate"] and strong_growth and not is_stage4_knife:
        garp_r1_verdict = "STAGED BUY / ACCUMULATE"
        garp_r1_conv = 8 if cash["proven_multi_year"] else 7
        garp_r1_w = 2.0
        garp_r1_max = 6.0
        garp_r1_action = (
            f"Initiate {garp_r1_w:.1f}% starter at CMP {_fmt_inr(px, 1)} (P{cmp_pct:.0f}, +{cmp_vs_bear:.1f}% vs Bear Blend) "
            f"and stage +4.0% across P10 {_fmt_inr(p10, 0)} and Bear Blend {_fmt_inr(bear_blend, 0)} (Stage-3/4 cap <= 10%)."
        )
        garp_r1_trig = (
            f"3-tranche Valuation Ladder: 2.0% CMP {_fmt_inr(px, 1)} + 2.0% limit at P10 {_fmt_inr(p10, 0)} + "
            f"2.0% limit at Bear Blend {_fmt_inr(bear_blend, 0)}. Pause adds if stock enters Stage-4 breakdown; do not chase above Counterfactual {_fmt_inr(cf_px, 0)}."
        )
    elif val_ok and ("SELL" in val_rating.upper() or "OVERVALUED" in val_rating.upper() or (cmp_pct is not None and cmp_pct >= 75)):
        garp_r1_verdict = "AVOID / SELL"
        garp_r1_conv = 8
        garp_r1_w = 0.0
        garp_r1_max = 0.0
        garp_r1_action = f"Avoid / trim; Valuation v4.1 is {val_rating} (P{cmp_pct:.0f}, CMP {_fmt_inr(px, 1)} vs P50 {_fmt_inr(p50, 0)}) with zero margin of safety."
        garp_r1_trig = f"Revisit only on deep multiple reset toward P10 {_fmt_inr(p10, 0)} / Bear Blend {_fmt_inr(bear_blend, 0)}."
    else:
        garp_r1_verdict = "WAIT FOR TRIGGER (WATCHLIST)"
        garp_r1_conv = 6
        garp_r1_w = 0.0
        garp_r1_max = 5.0
        garp_r1_action = f"Hold 0% at CMP {_fmt_inr(px, 1)}; wait for Valuation-Ladder pullback toward P10 {_fmt_inr(p10, 0)} / Bear Blend {_fmt_inr(bear_blend, 0)} with Stage-1/2 base."
        garp_r1_trig = f"Stage 2.5% at P10 {_fmt_inr(p10, 0)} and +2.5% at Bear Blend {_fmt_inr(bear_blend, 0)} while CFO/PAT >= 0.80 and Q_NP_YoY > 0%."

    if cash["has_data"]:
        if cash["inflection_only_1y"]:
            garp_ev.append(
                f"Cash-Flow Inflection Nuance: Automated funnel flags '{', '.join(biz_flags) or 'cash conversion'}' from {cash['neg_prior_years']} prior negative-CFO year(s), "
                f"BUT {cash['latest_fy']} inflected sharply to CFO {_fmt_inr(cash['latest_cfo_cr'])} Cr (CFO/PAT {cash['latest_cfo_pat']:.2f}×, FCF/PAT {cash['latest_fcf_pat']:.2f}×, ROCE {cash['latest_roce']:.0f}%, D/E {cash['latest_de']:.2f})."
            )
            garp_rf.append(f"Single-year cash proof only: {cash['latest_fy']} CFO/PAT {cash['latest_cfo_pat']:.2f}× followed {cash['neg_prior_years']} negative-CFO years—H1/FY27 working-capital verification is mandatory.")
        else:
            garp_ev.append(
                f"Cash-Flow Gate ({cash['latest_fy']}): CFO {_fmt_inr(cash['latest_cfo_cr'])} Cr, CFO/PAT {cash['latest_cfo_pat']}×, "
                f"FCF/PAT {cash['latest_fcf_pat']}×, ROCE {cash['latest_roce']}%, D/E {cash['latest_de']}."
            )
            if not cash["passes_primary_gate"]:
                garp_rf.append(f"Primary cash-flow gate FAILED: {cash['latest_fy']} CFO/PAT ({cash['latest_cfo_pat']}) < 0.80 threshold (HFCL exit rule).")
    if val_ok:
        garp_ev.append(
            f"Valuation v4.1 ({val_rating}, P{cmp_pct:.1f}): Bear {_fmt_inr(bear_blend, 0)} ({cmp_vs_bear:+.1f}% vs CMP), "
            f"P10 {_fmt_inr(p10, 0)}, P50 {_fmt_inr(p50, 0)} ({val.get('upside_p50_pct'):+.1f}%), zero-rerating Counterfactual {_fmt_inr(cf_px, 0)} ({cf_upside:+.1f}% vs CMP)."
            if (cmp_vs_bear is not None and cf_upside is not None)
            else f"Valuation v4.1 ({val_rating}, P{cmp_pct}): Bear {_fmt_inr(bear_blend, 0)}, P50 {_fmt_inr(p50, 0)}."
        )
    if q_np_yoy is not None or q_sales_yoy is not None:
        garp_ev.append(
            f"Earnings Velocity ({latest_q.get('qtr')}): Sales {_fmt_inr(latest_q.get('sales_cr'))} Cr ({q_sales_yoy:+.1f}% YoY), "
            f"NP {_fmt_inr(latest_q.get('np_cr'))} Cr ({q_np_yoy:+.1f}% YoY), OPM {q_opm}%, Biz Score {biz_score} ({ff.get('biz_verdict')})."
            if (q_sales_yoy is not None and q_np_yoy is not None)
            else f"Latest quarter ({latest_q.get('qtr')}): Biz Score {biz_score} ({ff.get('biz_verdict')})."
        )
    if d_inst_q is not None:
        garp_ev.append(
            f"Ownership ({shp.get('period')}): Inst {d_inst_q:+.2f}pp QoQ (FII {shp.get('d_fii_q'):+.2f}pp, DII {shp.get('d_dii_q'):+.2f}pp) "
            f"vs Promoters {d_prom_q:+.2f}pp ({shp.get('promoters')}%)."
        )
    if mult_caution:
        garp_rf.append(
            f"Valuation Quality '{q_grade}': '{q_note}' (spot {spot_mult}× vs base {base_mult}× / peer {peer_mult}×)—strip out re-rating and anchor on Counterfactual {_fmt_inr(cf_px, 0)}."
        )
    if earnings_shrinking:
        garp_rf.append(f"Shrinking-earnings value trap blocked: latest Q NP YoY is {q_np_yoy:+.1f}% (<= 0% floor — Poisoned-29 HAPPSTMNDS rule).")
    if is_stage4_knife:
        garp_rf.append("Stage-4 falling-knife guardrail active: no repeated averaging below MA50 & MA200; enforce 60-day post-stop cooldown (Poisoned-29 LATENTVIEW rule).")
    if d_prom_q is not None and d_prom_q <= -1.0:
        garp_rf.append(f"Promoter stake trim: {d_prom_q:+.2f}pp QoQ in {shp.get('period')} (to {shp.get('promoters')}%); verify block rationale.")
    if opm_dip_bps >= 80:
        garp_rf.append(f"Early margin tick-down: OPM slipped -{opm_dip_bps} bps QoQ ({prev_opm}% → {q_opm}%).")
    if not garp_rf:
        garp_rf.append("Monitor quarterly OPM stability and working-capital conversion on next filing.")

    garp_r1 = {
        "member": "garp_investor",
        "verdict": garp_r1_verdict,
        "conviction_0_10": garp_r1_conv,
        "target_weight_now_pct": garp_r1_w,
        "max_weight_on_trigger_pct": garp_r1_max,
        "action_now": garp_r1_action,
        "entry_or_trigger_plan": garp_r1_trig,
        "stop_and_invalidation": (
            f"No tight technical stop; -25% catastrophe circuit breaker (~{_fmt_inr(px * 0.75, 0)}) with mandatory 60-day post-stop cooldown & max <=10% Stage-3/4 cap. "
            "Fundamental exit (HFCL rule): CFO/PAT < 0.80, Q_NP_YoY <= 0%, D/E > 0.80, 2Q OPM erosion > 300 bps, or unabsorbed promoter selling."
        ),
        "key_evidence": garp_ev,
        "key_red_flags": garp_rf,
        "preemptive_challenge_to_peers": (
            f"To Momentum: Waiting for a break above {_fmt_inr(bear_clear_px, 0)} surrenders margin of safety right underneath Counterfactual {_fmt_inr(cf_px, 0)}. "
            "To Hybrid: Don't let lagging funnel flags or high VIX freeze you if Bear-Blend asymmetry is building."
            if (cf_px and is_val_buy) else
            "To Momentum & Hybrid: Never overpay for multiple-expansion narratives—demand verified CFO/PAT >= 0.80 and Bear-Blend downside protection."
        ),
    }

    # --- 3. Macro-Regime & Hybrid Contrarian PM (Round 1) ---
    hyb_ev: List[str] = []
    hyb_rf: List[str] = []
    coiled_macro = (inst_cash_cr >= 2000) or rec_active or (fii_long is not None and fii_long <= 28.0 and inst_cash_cr > 0)
    strong_deliv = deliv_ratio is not None and deliv_ratio >= 1.05

    if gate1 and fx_action == "BUY NOW" and (is_val_buy or strong_growth) and not has_post_shp_dump:
        hyb_r1_verdict = "BUY NOW"
        hyb_r1_conv = 8
        hyb_r1_w = 4.0 if engine_mode in ("CAUTION", "DEFENSIVE") else 6.0
        hyb_r1_max = 8.0
        hyb_r1_stop = round(max(_num(fx.get("stop"), px - 2.0 * atr) or (px - 2.0 * atr), px * 0.88), 1)
        hyb_r1_action = f"Deploy {hyb_r1_w:.1f}% Core+Tactical entry at {_fmt_inr(px, 1)} (Stage-2 + fundamentals aligned) with {engine_mode} stop at {_fmt_inr(hyb_r1_stop, 1)}."
        hyb_r1_trig = f"Scale to {hyb_r1_max:.1f}% on confirmation above {_fmt_inr(px * 1.04, 1)} or VIX cooling < 15."
    elif (
        not earnings_shrinking
        and not is_stage4_knife
        and ((is_val_buy and near_bear and cash["passes_primary_gate"] and coiled_macro) or (is_val_buy and below_bear and cash["passes_primary_gate"]))
    ):
        hyb_r1_verdict = "STAGED BUY / ACCUMULATE"
        hyb_r1_conv = 6
        hyb_r1_w = 2.5
        hyb_r1_max = 7.5  # <= 10% cap on unconfirmed Stage-3/4 value entries
        hyb_r1_stop = round(ma200, 1) if (ma200 and ma200 < px) else round(px * 0.88, 1)
        hyb_r1_action = (
            f"Deploy regime-capped {hyb_r1_w:.1f}% pilot at {_fmt_inr(px, 1)} into coiled macro flows (FII+DII {inst_cash_cr:+,.0f} Cr), "
            f"holding 5.0% dry powder for Bear-Blend flush ({_fmt_inr(bear_blend, 0)}) or MA50/bear-supply reclaim (Stage-3/4 cap <= 10%, 20% profit trail)."
        )
        hyb_r1_trig = (
            f"Path A: +2.5% on flush to {_fmt_inr(bear_blend, 0)}–{_fmt_inr(px - 2 * atr, 0)} above MA200 ({_fmt_inr(ma200, 0)}). "
            f"Path B: scale to 7.5% on close > {_fmt_inr(bear_clear_px, 0)} with Gate-1=True & VIX < 15 / RECOVERY."
        )
    elif (not is_val_buy and not gate1) or is_stage4_knife or (is_val_buy and earnings_shrinking and not gate1):
        hyb_r1_verdict = "AVOID / SELL" if (is_stage4_knife and (not is_val_buy or earnings_shrinking)) else "WAIT FOR TRIGGER (WATCHLIST)"
        hyb_r1_conv = 7
        hyb_r1_w = 0.0
        hyb_r1_max = 0.0 if hyb_r1_verdict == "AVOID / SELL" else 5.0
        hyb_r1_stop = round(px * 0.88, 1)
        hyb_r1_action = (
            f"Hold 0% ({hyb_r1_verdict}); blocked by anti-trap guardrails (Stage {stage}, Q_NP_YoY {q_np_yoy}%, Gate-1={gate1})—no blind Stage-4 value knife-catching."
        )
        hyb_r1_trig = f"Wait for Stage-2 reclaim > {_fmt_inr(ma50_reclaim_px, 1)} (with RS > +5 if BBP < 20) OR Stage-1 stabilization with Q_NP_YoY > 0%."
    else:
        hyb_r1_verdict = "WAIT FOR TRIGGER (WATCHLIST)"
        hyb_r1_conv = 6
        hyb_r1_w = 0.0
        hyb_r1_max = 6.0
        hyb_r1_stop = round(bear_blend - 2.0 * atr, 1) if bear_blend else round(px * 0.88, 1)
        hyb_r1_action = f"Hold 0% today in {engine_mode} regime; keep dual triggers ready at Bear Blend ({_fmt_inr(bear_blend, 0)}) and Stage-2 reclaim ({_fmt_inr(ma50_reclaim_px, 0)})."
        hyb_r1_trig = f"Path A: 3.0% on Bear-Blend absorption ({_fmt_inr(bear_blend, 0)}); Path B: 3.0% on close > {_fmt_inr(ma50_reclaim_px, 0)} with VIX < 16."

    hyb_ev.append(
        f"Macro Regime: Nifty {nifty_reg.get('regime')} ({nifty_reg.get('dd'):+.1f}% DD, MA200 rising={nifty_reg.get('ma200_rising')}), "
        f"Breadth {breadth.get('regime')} ({breadth_pct:.1f}% >200D), Engine {engine_mode}, VIX {vix_close} (P{vix_pctile}), "
        f"FII+DII cash {inst_cash_cr:+,.0f} Cr, FII Long {fii_long}%, RECOVERY={rec_active}."
        if (nifty_reg.get("dd") is not None and breadth_pct is not None)
        else f"Macro Regime: Engine {engine_mode}, VIX {vix_close} (P{vix_pctile}), FII+DII cash {inst_cash_cr:+,.0f} Cr, RECOVERY={rec_active}."
    )
    if deliv_pct is not None:
        hyb_ev.append(
            f"Tape & Delivery: Delivery {deliv_pct:.1f}% ({deliv_ratio:.2f}× 20d avg, ₹{deliv.get('abs_deliv_cr')} Cr/d), "
            f"Supply={fx.get('supply')}, Sector {ind.get('industry')}={ind.get('ind_group')} ({ind.get('ind_rank')})."
            if deliv_ratio is not None else f"Delivery {deliv_pct:.1f}%, Supply={fx.get('supply')}."
        )
    if val_ok:
        hyb_ev.append(
            f"Asymmetry check: CMP {_fmt_inr(px, 1)} vs Bear {_fmt_inr(bear_blend, 0)} ({cmp_vs_bear:+.1f}%), "
            f"Counterfactual {_fmt_inr(cf_px, 0)} ({cf_upside:+.1f}%), P50 {_fmt_inr(p50, 0)} ({val.get('upside_p50_pct'):+.1f}%)."
            if (cmp_vs_bear is not None and cf_upside is not None) else f"Valuation rating {val_rating}."
        )
    if has_post_shp_dump:
        hyb_rf.append(f"Overhead supply conflict: ₹{bear_turn_cr:.1f} Cr ({bear_turn_mult:.1f}× daily turnover) of recent bear bars in a {ind.get('ind_group')} sector.")
    if vix_high and not rec_active:
        hyb_rf.append(f"VIX stress (P{vix_pctile}) with RECOVERY=False: institutional cash bursts favor liquid large-caps before midcaps.")
    if mult_caution:
        hyb_rf.append(f"Multiple-recovery trap: P50 relies on {base_mult}× PE vs {spot_mult}× spot; Counterfactual {_fmt_inr(cf_px, 0)} caps zero-rerating upside.")
    if not hyb_rf:
        hyb_rf.append(f"Monitor {engine_mode} breadth & VIX before exceeding 5% single-stock weight.")

    hyb_r1 = {
        "member": "hybrid_pm",
        "verdict": hyb_r1_verdict,
        "conviction_0_10": hyb_r1_conv,
        "target_weight_now_pct": hyb_r1_w,
        "max_weight_on_trigger_pct": hyb_r1_max,
        "action_now": hyb_r1_action,
        "entry_or_trigger_plan": hyb_r1_trig,
        "stop_and_invalidation": f"Initial structural stop {_fmt_inr(hyb_r1_stop, 1)}; enforce {engine_mode} {trail_pct}% trail and fundamental CFO/PAT >= 0.80 gate.",
        "key_evidence": hyb_ev,
        "key_red_flags": hyb_rf,
        "preemptive_challenge_to_peers": (
            "To Momentum: Sitting at 0% into washed-out FII positioning risks getting gapped past the ATR guard on short-covering. "
            "To GARP: Don't anchor on multiple-expansion P50 targets or ignore fresh overhead supply."
        ),
    }

    r1_verdicts = [mom_r1["verdict"], garp_r1["verdict"], hyb_r1["verdict"]]
    _, r1_tally = _vote_summary(r1_verdicts)

    # =======================================================================
    # ROUND 2: LIVE CROSS-EXAMINATION, BELIEF UPDATES & CONTRA STATEMENTS
    # =======================================================================
    # 1. Momentum Round 2 adaptation (GARP's Counterfactual R:R squeeze check)
    mom_r2_verdict = mom_r1_verdict
    mom_r2_conv = mom_r1_conv
    mom_r2_w = mom_r1_w
    mom_r2_max = mom_r1_max
    rr_squeezed = (
        mom_r1_verdict == "WAIT FOR TRIGGER (WATCHLIST)"
        and cf_px is not None
        and bear_clear_px > ma50_reclaim_px
        and ((cf_px / bear_clear_px - 1.0) * 100.0) < 14.0
        and cf_px > ma50_reclaim_px
    )
    if rr_squeezed:
        step1_stop = round(anchor_low if (anchor_low and anchor_low < ma50_reclaim_px and (ma50_reclaim_px - anchor_low) <= 1.8 * atr) else (ma50_reclaim_px - 1.2 * atr), 1)
        step1_risk_pct = round((1.0 - step1_stop / ma50_reclaim_px) * 100.0, 1)
        step1_upside_pct = round((cf_px / ma50_reclaim_px - 1.0) * 100.0, 1)
        rr_ratio = round(step1_upside_pct / max(step1_risk_pct, 0.5), 1)
        mom_delta = (
            f"Conceded GARP's {_fmt_inr(bear_clear_px, 0)} → {_fmt_inr(cf_px, 0)} R:R squeeze while holding 0% today. "
            f"Split post-trigger entry into Tranche 1 (1.5%) on confirmed MA50/Stage-2 reclaim > {_fmt_inr(ma50_reclaim_px, 0)} "
            f"({vol_req}+ vol, CLV >= 0.60, stop {_fmt_inr(step1_stop, 0)} -> -{step1_risk_pct:.1f}% risk vs +{step1_upside_pct:.1f}% to Counterfactual = {rr_ratio:.1f}:1 R:R) "
            f"and Tranche 2 (+2.5% to {mom_r2_max:.1f}% max) on clearing {_fmt_inr(bear_clear_px, 0)} (ratcheting stop to {_fmt_inr(round(ma50, 0), 0)})."
        )
    else:
        mom_delta = f"Held firm on {mom_r2_verdict} ({mom_r2_w:.1f}% now / {mom_r2_max:.1f}% max); microstructure rules remain non-negotiable."

    mom_r2 = {
        "member": "mom_trader",
        "rebuttal_to_garp": (
            f"GARP's R:R math on {_fmt_inr(bear_clear_px, 0)} vs Counterfactual {_fmt_inr(cf_px, 0)} is fair—resolved via a two-step >{_fmt_inr(ma50_reclaim_px, 0)} / >{_fmt_inr(bear_clear_px, 0)} trigger—but buying lower is NOT safer if ₹{bear_turn_cr:.1f} Cr of post-SHP liquidation was smart money exiting ahead of margin/cash-flow normalization."
            if rr_squeezed else
            f"Valuation floors don't stop Stage-{stage} distribution; wait for institutional demand to prove itself with Gate-1=True and {vol_req}+ volume."
        ),
        "rebuttal_to_hybrid": (
            f"Holding 0% in {dossier['ticker'].replace('.NS', '')} is NOT 100% cash drag—it means rotating {engine_mode} capital into Gate-1=True Stage-2 RS leaders rather than forcing a pilot into a {fx_state} chart."
            + (f" Also, bidding near {_fmt_inr(bear_blend, 0)} right above MA200 ({_fmt_inr(ma200, 0)}) leaves <1.5×ATR buffer—guaranteed whipsaw." if (bear_blend and ma200 and abs(bear_blend - ma200) < 1.5 * atr) else "")
        ),
        "strongest_contra_statement": (
            f"Anchoring to lagging quarterly filings while ignoring ₹{bear_turn_cr:.1f} Cr ({bear_turn_mult:.1f}× daily turnover) of fresh post-SHP liquidation (CLV < 0.35) risks tying up capital in Stage-{stage} dead money while true Stage-2 leaders compound."
            if has_post_shp_dump else
            f"Entering without Stage-2 alignment (Gate-1={gate1}, CompRS {comp_rs}) in a {engine_mode} / VIX P{vix_pctile} tape exposes capital to extended base churn or stop-outs."
        ),
        "final_verdict": mom_r2_verdict,
        "final_conviction_0_10": mom_r2_conv,
        "final_weight_now_pct": mom_r2_w,
        "final_max_weight_pct": mom_r2_max,
        "revised_plan_delta": mom_delta,
        "what_flips_my_call": (
            f"Daily close > {_fmt_inr(ma50_reclaim_px, 0)} (reclaiming MA50 {_fmt_inr(ma50, 0)}, Stage-2 / Gate-1=True) on {vol_req}+ volume and CLV >= 0.60 within the {gap_guard:.1f}% ATR gap guard."
            if mom_r2_verdict != "BUY NOW" else
            f"High-volume close below stop {_fmt_inr(mom_r1_stop, 0)} or MA50 ({_fmt_inr(ma50, 0)})."
        ),
    }

    # 2. GARP Round 2 adaptation (Momentum's Post-SHP liquidation warning)
    garp_r2_verdict = garp_r1_verdict
    garp_r2_conv = garp_r1_conv
    garp_r2_w = garp_r1_w
    garp_r2_max = garp_r1_max
    if garp_r1_verdict == "STAGED BUY / ACCUMULATE" and (has_post_shp_dump or (vix_high and not gate1)):
        garp_r2_conv = max(5, garp_r1_conv - 1)
        garp_r2_w = round(garp_r1_w * 0.5, 1)
        blended_est = round((garp_r2_w * px + 2.0 * (p10 or px * 0.96) + 3.0 * (bear_blend or px * 0.90)) / max(garp_r2_w + 5.0, 1.0), 0)
        garp_delta = (
            f"Conceded Momentum's post-SHP selling risk (₹{bear_turn_cr:.1f} Cr) & single-year cash proof: conviction {garp_r1_conv} → {garp_r2_conv}, "
            f"halved immediate starter {garp_r1_w:.1f}% → {garp_r2_w:.1f}% tracking foothold at {_fmt_inr(px, 1)}, and back-loaded +2.0% at P10 {_fmt_inr(p10, 0)} "
            f"and +3.0% at Bear Blend {_fmt_inr(bear_blend, 0)} (blended cost ~{_fmt_inr(blended_est, 0)})."
        )
    else:
        garp_delta = f"Maintained {garp_r2_verdict} ({garp_r2_w:.1f}% now / {garp_r2_max:.1f}% max) anchored to Bear Blend {_fmt_inr(bear_blend, 0)} and CFO/PAT >= 0.80 discipline."

    garp_r2 = {
        "member": "garp_investor",
        "rebuttal_to_mom": (
            f"Conceded that post-SHP selling (₹{bear_turn_cr:.1f} Cr) warrants halving immediate size to {garp_r2_w:.1f}%, but waiting for >{_fmt_inr(bear_clear_px, 0)} surrenders GARP asymmetry right beneath zero-rerating Counterfactual {_fmt_inr(cf_px, 0)} while delivery ({deliv_pct}%) shows absorption."
            if (has_post_shp_dump and is_val_buy) else
            "Pure technical triggers systematically enter late on high-ROCE compounders; Valuation-Ladder limit bids near Bear Blend + CFO/PAT >= 0.80 delivered 100% win rate and -0.94% average MAE across our prior marches."
        ),
        "rebuttal_to_hybrid": (
            f"Placing a tight MA200 stop ({_fmt_inr(ma200, 0)}) directly beneath a Bear-Blend ({_fmt_inr(bear_blend, 0)}) add bid leaves <1.5×ATR buffer in a {atr_pct:.1f}% ATR stock—stop-hunting suicide. Small starter sizing + fundamental HFCL invalidation + -25% catastrophe breaker caps portfolio NAV risk without whipsaw."
            if (bear_blend and ma200 and abs(bear_blend - ma200) < 1.5 * atr) else
            "Fundamental cash-flow invalidation (CFO/PAT < 0.80) outperforms arbitrary percentage stops for core compounders."
        ),
        "strongest_contra_statement": (
            f"Against our own GARP case: {cash['latest_fy']} CFO/PAT ({cash['latest_cfo_pat']:.2f}×) follows {cash['neg_prior_years']} negative-CFO years with zero H1/FY27 cash verification yet; if working capital re-expands while OPM mean-reverts ({opm_dip_bps} bps slip already visible) and spot PE ({spot_mult}×) de-rates toward 25×, price can slice straight through Bear Blend {_fmt_inr(bear_blend, 0)} into a -35% air-pocket."
            if (cash["inflection_only_1y"] and val_ok) else
            f"Against GARP value: Valuation quality is '{q_grade}' ({q_note or 'standard'}); if earnings growth decelerates below 20% or CFO/PAT slips < 0.80, even Bear Blend ({_fmt_inr(bear_blend, 0)}) will re-rate lower."
        ),
        "final_verdict": garp_r2_verdict,
        "final_conviction_0_10": garp_r2_conv,
        "final_weight_now_pct": garp_r2_w,
        "final_max_weight_pct": garp_r2_max,
        "revised_plan_delta": garp_delta,
        "what_flips_my_call": (
            f"UPGRADE to BUY NOW (8/10): verified H1/FY27 CFO/PAT >= 0.80 with stable OPM, OR panic flush into Bear Blend {_fmt_inr(bear_blend, 0)}. "
            "DOWNGRADE to AVOID: CFO/PAT < 0.80, D/E > 0.80, 2Q OPM drop > 300 bps, or -25% catastrophe breaker."
        ),
    }

    # 3. Macro-Hybrid PM Round 2 adaptation (Vote Flip on Post-SHP VETO + Stop-Collision Fix!)
    hyb_r2_verdict = hyb_r1_verdict
    hyb_r2_conv = hyb_r1_conv
    hyb_r2_w = hyb_r1_w
    hyb_r2_max = hyb_r1_max
    hyb_flipped = False
    path_a_stop = round((bear_blend - 2.0 * atr), 0) if bear_blend else round(px * 0.82, 0)
    path_a_risk_pct = round((1.0 - path_a_stop / bear_blend) * 100.0, 1) if (bear_blend and bear_blend > 0) else 10.0
    path_a_rr = round((cf_upside_from_bear or 25.0) / max(path_a_risk_pct, 1.0), 1)

    if hyb_r1_verdict == "STAGED BUY / ACCUMULATE" and (has_post_shp_dump or (not gate1 and vix_high and not rec_active)):
        hyb_r2_verdict = "WAIT FOR TRIGGER (WATCHLIST)"
        hyb_r2_conv = 7
        hyb_r2_w = 0.0
        hyb_r2_max = 6.0
        hyb_flipped = True
        hyb_delta = (
            f"FLIPPED from STAGED BUY ({hyb_r1_w:.1f}% now) → WAIT FOR TRIGGER (0% now, conviction {hyb_r1_conv} → {hyb_r2_conv}). "
            f"Reallocated immediate {engine_mode} cash-drag defense to Gate-1=True Stage-2 leaders and fixed the Bear-Blend/MA200 stop collision: "
            f"Path A (3.0% sniper) on flush to {_fmt_inr(bear_blend, 0)} with CLV >= 0.55 absorption bar & widened 2×ATR stop at {_fmt_inr(path_a_stop, 0)} "
            f"(-{path_a_risk_pct:.1f}% risk vs +{cf_upside_from_bear or 0:.1f}% Counterfactual = {path_a_rr:.1f}:1 R:R); "
            f"Path B (3.0%) on close > {_fmt_inr(ma50_reclaim_px, 0)} (CLV >= 0.60 + VIX < 16 or RECOVERY), scaling +3.0% > {_fmt_inr(bear_clear_px, 0)}."
        )
    else:
        hyb_delta = (
            f"Retained {hyb_r2_verdict} ({hyb_r2_w:.1f}% now / {hyb_r2_max:.1f}% max) with non-overlapping 2×ATR stop geometry ({_fmt_inr(path_a_stop, 0)} on Bear-Blend path)."
        )

    hyb_r2 = {
        "member": "hybrid_pm",
        "rebuttal_to_mom": (
            f"Conceded portfolio allocation: curing {engine_mode} cash drag belongs in Gate-1=True Stage-2 RS leaders, not forcing {hyb_r1_w:.1f}% into a {fx_state} chart right above {_fmt_inr(anchor_low, 0)}—dropped immediate pilot to 0%. "
            f"However, waiting solely for >{_fmt_inr(bear_clear_px, 0)} ignores two cleaner edges: MA50 reclaim >{_fmt_inr(ma50_reclaim_px, 0)} once VIX cools AND a high-asymmetry Bear-Blend sniper at {_fmt_inr(bear_blend, 0)}."
            if hyb_flipped else
            "Agree on respecting Gate-1 and ATR gap guards, while keeping a contrarian Bear-Blend sniper ready for VIX capitulation."
        ),
        "rebuttal_to_garp": (
            f"Owned my Round 1 stop-collision bug and widened the Bear-Blend stop to 2×ATR ({_fmt_inr(path_a_stop, 0)}, -{path_a_risk_pct:.1f}% risk vs +{cf_upside_from_bear or 0:.1f}% Counterfactual = {path_a_rr:.1f}:1 R:R). "
            f"Still push back on buying at CMP {_fmt_inr(px, 1)} into fresh selling: CMP-to-Counterfactual ({cf_upside:+.1f}%) vs -25% stop is sub-1:1 without multiple re-rating."
            if (val_ok and cf_upside is not None) else
            "Core GARP staging must still respect macro liquidity regimes and avoid compressing tranches within <1×ATR."
        ),
        "strongest_contra_statement": (
            f"Macro Contra: VIX {vix_close} (P{vix_pctile}) + Nifty MA200 rising={nifty_reg.get('ma200_rising')} + Breadth {breadth.get('regime')} ({breadth_pct:.1f}% >200D, RECOVERY={rec_active}) may be the OPENING of a midcap liquidity squeeze, not a late shakeout. Institutional inflows ({inst_cash_cr:+,.0f} Cr) bid liquid Nifty-50 heavyweights first—leaving ₹{liq:.1f} Cr/d midcaps in {ind.get('ind_group')} sectors exposed to PE compression."
            if breadth_pct is not None else
            f"Macro Contra: Elevated volatility ({engine_mode}) can compress midcap multiples far faster than quarterly earnings compound."
        ),
        "final_verdict": hyb_r2_verdict,
        "final_conviction_0_10": hyb_r2_conv,
        "final_weight_now_pct": hyb_r2_w,
        "final_max_weight_pct": hyb_r2_max,
        "revised_plan_delta": hyb_delta,
        "what_flips_my_call": (
            f"Flips to ACTIVE BUY (3.0%) on EITHER (1) daily close > {_fmt_inr(ma50_reclaim_px, 0)} (CLV >= 0.60) with VIX < 16 or RECOVERY=True, "
            f"OR (2) capitulation sweep into Bear Blend {_fmt_inr(bear_blend, 0)} with CLV >= 0.55 absorption bar. Flips to AVOID below {_fmt_inr(path_a_stop, 0)} (2 closes) or CFO/PAT < 0.80."
        ),
    }

    r2_verdicts = [mom_r2["final_verdict"], garp_r2["final_verdict"], hyb_r2["final_verdict"]]
    chair_verdict, r2_tally = _vote_summary(r2_verdicts)
    vote_flipped = r1_tally != r2_tally

    # Synthesize Bear & Bull Contra Statements + Split Playbooks + External Diligence
    bear_contra = " ".join([
        mom_r2["strongest_contra_statement"],
        garp_r2["strongest_contra_statement"],
    ])
    bull_contra = (
        f"Opportunity-Cost Contra (Against staying 0%): {dossier['name']} holds Biz Score {biz_score} ({ff.get('biz_verdict')}), "
        f"latest quarterly PAT {q_np_yoy:+.1f}% YoY, {cash['latest_fy'] or 'latest'} ROCE {cash['latest_roce']}%, "
        f"and {shp.get('period') or 'recent'} institutional flows {d_inst_q:+.2f}pp QoQ with delivery at {deliv_pct}% ({deliv_ratio}× avg). "
        f"In a thin ₹{liq:.1f} Cr/d tape with FII index longs washed out at {fii_long}%, a single macro relief rally can gap price past the {gap_guard:.1f}% ATR gap guard ({_fmt_inr(gap_cap_px, 0)}), stranding 0%-weight traders."
        if (q_np_yoy is not None and d_inst_q is not None and deliv_pct is not None)
        else f"Opportunity-Cost Contra: If macro breadth turns into RECOVERY and Forensics flips to GO, thin liquidity (₹{liq:.1f} Cr/d) can gap past the {gap_guard:.1f}% ATR gap guard before strict buy-stops fill cleanly."
    )

    playbooks = [
        {
            "mandate": "Swing / Active Momentum",
            "action_today": f"{mom_r2['final_weight_now_pct']:.1f}% ({mom_r2['final_verdict']})",
            "triggers": (
                f"Tranche 1 (1.5%): Close > {_fmt_inr(ma50_reclaim_px, 0)} ({vol_req}+ vol, CLV >= 0.60, Gate-1=True). "
                f"Tranche 2 (+2.5%): Close > {_fmt_inr(bear_clear_px, 0)}. Gap guard <= {_fmt_inr(gap_cap_px, 0)} ({gap_guard:.1f}%)."
                if rr_squeezed else mom_r1["entry_or_trigger_plan"]
            ),
            "stop_rule": mom_r1["stop_and_invalidation"],
            "payoff_anchor": f"Counterfactual {_fmt_inr(cf_px, 0)} / 52W High {_fmt_inr(hi52, 0)}" if cf_px else f"52W High {_fmt_inr(hi52, 0)}",
        },
        {
            "mandate": "Contrarian Macro-Hybrid Sniper",
            "action_today": f"{hyb_r2['final_weight_now_pct']:.1f}% ({hyb_r2['final_verdict']})",
            "triggers": (
                f"Path A (3.0%): Flush into Bear Blend {_fmt_inr(bear_blend, 0)} with CLV >= 0.55 absorption bar. "
                f"Path B (3.0%): Close > {_fmt_inr(ma50_reclaim_px, 0)} + VIX < 16 / RECOVERY."
            ),
            "stop_rule": f"Path A: 2 closes < {_fmt_inr(path_a_stop, 0)} (2×ATR below Bear). Path B: close < {_fmt_inr(round((anchor_low or ma50 * 0.96) * 0.99, 0), 0)}; {engine_mode} {trail_pct}% trail.",
            "payoff_anchor": f"Counterfactual {_fmt_inr(cf_px, 0)} (+{cf_upside_from_bear:.1f}% from Bear = {path_a_rr:.1f}:1 R:R)" if (cf_px and cf_upside_from_bear is not None) else f"P50 {_fmt_inr(p50, 0)}",
        },
        {
            "mandate": "Core GARP Compounder",
            "action_today": f"{garp_r2['final_weight_now_pct']:.1f}% ({garp_r2['final_verdict']})",
            "triggers": (
                f"Starter {garp_r2['final_weight_now_pct']:.1f}% CMP {_fmt_inr(px, 1)}; +2.0% limit at P10 {_fmt_inr(p10, 0)}; +3.0% limit at Bear Blend {_fmt_inr(bear_blend, 0)}."
                if garp_r2["final_weight_now_pct"] > 0 else garp_r1["entry_or_trigger_plan"]
            ),
            "stop_rule": "No tight technical stop. Exit on CFO/PAT < 0.80, D/E > 0.80, 2Q OPM erosion > 300 bps, or -25% catastrophe breaker.",
            "payoff_anchor": f"Zero-rerating {_fmt_inr(cf_px, 0)}; P50 {_fmt_inr(p50, 0)} only if CFO proves multi-year" if cf_px else f"P50 {_fmt_inr(p50, 0)}",
        },
    ]

    diligence: List[str] = []
    if cash["inflection_only_1y"] or not cash["passes_primary_gate"]:
        diligence.append(
            f"Verify latest half-year / interim operating cash flow & working-capital days (did {cash['latest_fy']}'s CFO/PAT {cash['latest_cfo_pat']}× sustain, or did inventory/receivables re-expand?)."
        )
    if has_post_shp_dump:
        diligence.append(
            f"Investigate the catalyst & counterparty behind the ₹{bear_turn_cr:.1f} Cr post-SHP high-volume selloff ({', '.join(b['date'] for b in bear_bars)})—concall margin guidance vs institutional block overhang."
        )
    if d_prom_q is not None and d_prom_q <= -1.0:
        diligence.append(
            f"Confirm the rationale & lock-in behind the {shp.get('period')} {d_prom_q:+.2f}pp promoter stake reduction (to {shp.get('promoters')}%)."
        )
    if opm_dip_bps >= 80:
        diligence.append(f"Check management commentary on the -{opm_dip_bps} bps QoQ OPM dip ({prev_opm}% → {q_opm}%): seasonal raw-material mix vs structural pricing pressure.")
    if not diligence:
        diligence.append("Verify next quarterly earnings trajectory, order book / capacity utilization, and whether institutional accumulation persists.")

    return {
        "mode": "quant_deterministic",
        "ticker": dossier["ticker"],
        "name": dossier["name"],
        "as_of": dossier["as_of"],
        "round1_vote": r1_tally,
        "round2_final_vote": r2_tally,
        "vote_flipped": vote_flipped,
        "chair_verdict": chair_verdict,
        "chair_headline": (
            f"Round 2 flipped the Council ({r1_tally} → {r2_tally}): {hyb_r2['revised_plan_delta'].split('. ')[0]}."
            if vote_flipped else
            f"Council consensus held at {r2_tally} (CMP {_fmt_inr(px, 1)}, Stage {stage}, Forensics {fx_state}, Valuation {val_rating})."
        ),
        "round1": {"mom_trader": mom_r1, "garp_investor": garp_r1, "hybrid_pm": hyb_r1},
        "round2": {"mom_trader": mom_r2, "garp_investor": garp_r2, "hybrid_pm": hyb_r2},
        "bear_contra_statement": bear_contra,
        "bull_contra_statement": bull_contra,
        "playbooks": playbooks,
        "diligence_questions": diligence,
        "dossier": dossier,
    }


# ---------------------------------------------------------------------------
# 3. Live Multi-Round LLM Council (OpenCode, DeepSeek, Gemini, or Custom /v1)
# ---------------------------------------------------------------------------
LLM_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "opencode": {
        "label": "OpenCode (Zen API)",
        "base_url": "https://opencode.ai/zen/v1",
        "default_model": "gpt-5.6-luna",
        "models": ["gpt-5.6-luna", "deepseek-v4-flash", "deepseek-v4-flash-free", "kimi-k3"],
        "env_key": "OPENCODE_API_KEY",
    },
    "deepseek": {
        "label": "DeepSeek API",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "env_key": "DEEPSEEK_API_KEY",
    },
    "gemini": {
        "label": "Gemini API",
        "base_url": "",
        "default_model": "gemini-2.5-flash",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro"],
        "env_key": "GEMINI_API_KEY",
    },
    "custom": {
        "label": "Custom / Local OpenCode",
        "base_url": "http://localhost:4096/v1",
        "default_model": "deepseek-v4-flash",
        "models": ["deepseek-v4-flash", "gpt-5.6-luna", "deepseek-chat"],
        "env_key": "LLM_API_KEY",
    },
}


def _read_secret_or_env(key_name: str, provider_section: Optional[str] = None, field: str = "api_key") -> str:
    val = (os.environ.get(key_name) or "").strip()
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if key_name in st.secrets and str(st.secrets[key_name]).strip():
                return str(st.secrets[key_name]).strip()
            if provider_section and "llm" in st.secrets:
                provs = st.secrets["llm"].get("providers") or {}
                psec = provs.get(provider_section) or {}
                if psec.get(field):
                    return str(psec[field]).strip()
    except Exception:
        pass
    return ""


def resolve_llm_config(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    gemini_key: Optional[str] = None,
) -> Dict[str, str]:
    """Resolve provider -> concrete {provider, api_key, model, base_url, label}.

    Auto-detection priority when `provider` is None:
      1. OpenCode (`OPENCODE_API_KEY` or `[llm.providers.opencode]`)
      2. DeepSeek (`DEEPSEEK_API_KEY` or `[llm.providers.deepseek]`)
      3. Gemini (`GEMINI_API_KEY` or `gemini_key`)
      4. Generic `LLM_API_KEY` / `LLM_BASE_URL`
    """
    if gemini_key and not api_key and not provider:
        provider = "gemini"
        api_key = gemini_key

    # Auto-select configured provider when none explicitly requested
    if not provider:
        if _read_secret_or_env("OPENCODE_API_KEY", "opencode", "api_key") not in ("", "not-needed"):
            provider = "opencode"
        elif _read_secret_or_env("DEEPSEEK_API_KEY", "deepseek", "api_key"):
            provider = "deepseek"
        elif _read_secret_or_env("GEMINI_API_KEY"):
            provider = "gemini"
        elif os.environ.get("LLM_BASE_URL") or os.environ.get("LLM_API_KEY"):
            provider = "custom"
        else:
            return {"provider": "none", "api_key": "", "model": "", "base_url": "", "label": "Not configured"}

    prov = provider.lower().strip()
    spec = LLM_PROVIDERS.get(prov) or LLM_PROVIDERS["custom"]

    key = (api_key or "").strip()
    if not key:
        key = _read_secret_or_env(spec["env_key"], prov if prov in ("opencode", "deepseek") else None, "api_key")
        if not key and prov in ("opencode", "custom"):
            key = (os.environ.get("LLM_API_KEY") or "").strip()

    url = (base_url or "").strip()
    if not url and prov != "gemini":
        env_url_name = {"opencode": "OPENCODE_BASE_URL", "deepseek": "DEEPSEEK_BASE_URL"}.get(prov, "LLM_BASE_URL")
        url = (
            _read_secret_or_env(env_url_name, prov if prov in ("opencode", "deepseek") else None, "base_url")
            or (os.environ.get("LLM_BASE_URL") or "").strip()
            or spec["base_url"]
        )

    mdl = (model or "").strip()
    if not mdl:
        if prov == "gemini":
            mdl = os.environ.get("COUNCIL_GEMINI_MODEL", spec["default_model"])
        else:
            mdl = os.environ.get(f"COUNCIL_{prov.upper()}_MODEL") or os.environ.get("LLM_MODEL_JUDGMENT") or spec["default_model"]

    return {
        "provider": prov,
        "api_key": key,
        "model": mdl,
        "base_url": url.rstrip("/"),
        "label": spec["label"],
    }


def llm_backend_status(
    gemini_key: Optional[str] = None,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Tuple[str, str]:
    """Return `(provider_id, human_status)` for the active/selected LLM backend."""
    cfg = resolve_llm_config(provider=provider, api_key=api_key, model=model, base_url=base_url, gemini_key=gemini_key)
    p = cfg["provider"]
    if p == "none":
        return "none", "No LLM API key configured (supports OpenCode, DeepSeek, Gemini — using instant Quant Council)"
    if p == "custom" and cfg["base_url"] and "localhost" in cfg["base_url"]:
        return p, f"{cfg['label']}: {cfg['model']} ({cfg['base_url']})"
    if not cfg["api_key"] or cfg["api_key"] == "not-needed":
        if p == "opencode" and "localhost" in cfg["base_url"]:
            return p, f"OpenCode Local: {cfg['model']} ({cfg['base_url']})"
        return "none", f"{cfg['label']} selected — paste API key or set {LLM_PROVIDERS[p]['env_key']} in secrets/env"
    via = f" @ {cfg['base_url']}" if cfg["base_url"] else ""
    return p, f"{cfg['label']}: {cfg['model']}{via}"


def _strip_fences(raw: str) -> str:
    t = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL | re.IGNORECASE).strip()
    if t.startswith("```"):
        parts = t.split("```")
        t = parts[1] if len(parts) > 1 else t
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    i, j = t.find("{"), t.rfind("}")
    return t[i:j + 1] if (i != -1 and j > i) else t


def _call_llm_json(prompt: str, system: str, cfg: Dict[str, str]) -> Dict[str, Any]:
    prov = cfg["provider"]
    if prov == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=cfg["api_key"])
        m = genai.GenerativeModel(
            model_name=cfg["model"],
            system_instruction=system + "\nReturn strictly valid JSON only (no markdown fences).",
            generation_config={"temperature": 0.2, "response_mime_type": "application/json"},
        )
        raw = (m.generate_content(prompt).text or "").strip()
        return json.loads(_strip_fences(raw))

    # OpenAI-compatible path: OpenCode Zen, DeepSeek Official API, Custom / Local OpenCode
    from openai import OpenAI
    client = OpenAI(base_url=cfg["base_url"], api_key=(cfg["api_key"] or "not-needed"), timeout=180)
    sys_msg = system + "\n\nReturn a single strictly valid JSON object only. No prose, no markdown fences."
    msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}]

    is_reasoner = "reasoner" in cfg["model"].lower() or cfg["model"].lower().startswith("o1") or cfg["model"].lower().startswith("o3")
    formats = [None] if is_reasoner else [{"type": "json_object"}, None]
    last_exc: Optional[Exception] = None
    for fmt in formats:
        try:
            kw: Dict[str, Any] = {"model": cfg["model"], "messages": msgs}
            if not is_reasoner:
                kw["temperature"] = 0.2
            if fmt is not None:
                kw["response_format"] = fmt
            resp = client.chat.completions.create(**kw)
            raw = (resp.choices[0].message.content or "").strip()
            return json.loads(_strip_fences(raw))
        except Exception as exc:
            last_exc = exc
            if fmt is not None and any(s in str(exc).lower() for s in ("response_format", "json_object", "unsupported", "not supported", "invalid")):
                continue
            if fmt is not None:
                continue
    raise RuntimeError(f"{cfg['label']} ({cfg['model']}) call failed: {last_exc}")


_ROLE_SYSTEMS = {
    "mom_trader": (
        "You are Council Member 1: Momentum & Microstructure Trader on the Nifty Dashboard Expert Council. "
        "Enforce validated Forensics v6 Dual-ATR rules (do not relax entries beyond Dual-ATR): Stage-2 (Close>MA50>MA200), Gate-1=True, CompRS>0, clean bull anchors (CLV>=0.50), "
        "post-SHP heavy BEAR liquidation bars (CLV<0.35), ATR-scaled gap guard max(5%,1.5*ATR%), and rotating CAUTION cash into Stage-2 RS leaders. "
        "Stay 100% grounded in the provided dossier JSON; never invent external facts."
    ),
    "garp_investor": (
        "You are Council Member 2: GARP & Valuation-Ladder Investor on the Nifty Dashboard Expert Council. "
        "Enforce Poisoned-Universe validated rules: primary CFO/PAT>=0.80 gate (audit multi-year vs single-year inflection), MANDATORY earnings-growth floor (Q_NP_YoY > 0% — block shrinking-earnings value traps), "
        "NO repeated averaging into Stage-4 falling knives (require Stage-1/2 base stabilization, cap unconfirmed Stage-3/4 positions <= 10%, enforce 60-day post-stop cooldown after any -25% catastrophe stop-out), "
        "strip out multiple-expansion optimism using zero-rerating Counterfactual price when Quality=Caution, and exit on HFCL cash-flow break (CFO/PAT < 0.80). "
        "Stay 100% grounded in the provided dossier JSON; never invent external facts."
    ),
    "hybrid_pm": (
        "You are Council Member 3: Macro-Regime & Hybrid Contrarian PM on the Nifty Dashboard Expert Council. "
        "Synthesize India VIX percentile, Breadth, RECOVERY detector, FII/DII cash & FII Long%, sector rank, delivery, and Valuation Bear Blend into non-overlapping Core+Tactical triggers (>=2xATR stop buffer). "
        "Enforce Poisoned-Universe anti-trap guardrails: (1) BBP<20 squeezes ONLY in Stage-2 with CompRS>+5 (0% trap rate), (2) GARP/Bear-Blend leg requires CFO/PAT>=0.80 AND Q_NP_YoY>0% with NO Stage-4 knife-catching, "
        "(3) cap unconfirmed Stage-3/4 value positions <= 10% and pair with a 20% trailing stop once profitable. "
        "Stay 100% grounded in the provided dossier JSON; never invent external facts."
    ),
}


def run_llm_council(
    dossier: Dict[str, Any],
    *,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    gemini_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Run live 2-round parallel LLM debate across all 3 experts (OpenCode, DeepSeek, Gemini, or Custom)."""
    backend, desc = llm_backend_status(
        gemini_key=gemini_key, provider=provider, api_key=api_key, model=model, base_url=base_url
    )
    if backend == "none":
        raise RuntimeError(desc)
    cfg = resolve_llm_config(provider=backend, api_key=api_key, model=model, base_url=base_url, gemini_key=gemini_key)

    compact = json.dumps(dossier, default=str)

    # Round 1 (3 parallel calls)
    r1_prompt = (
        f"Dossier JSON (strictly grounded, as of {dossier['as_of']}):\n{compact}\n\n"
        "Return JSON with keys: member, verdict (one of 'BUY NOW', 'STAGED BUY / ACCUMULATE', 'WAIT FOR TRIGGER (WATCHLIST)', 'AVOID / SELL'), "
        "conviction_0_10 (int), target_weight_now_pct (float), max_weight_on_trigger_pct (float), action_now (str), "
        "entry_or_trigger_plan (str), stop_and_invalidation (str), key_evidence (list[str]), key_red_flags (list[str]), "
        "preemptive_challenge_to_peers (str), diligence_questions (list[str])."
    )
    r1: Dict[str, Dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futs = {m: pool.submit(_call_llm_json, r1_prompt, sys_p, cfg) for m, sys_p in _ROLE_SYSTEMS.items()}
        for m, f in futs.items():
            obj = f.result()
            obj["member"] = m
            r1[m] = obj

    # Round 2 (3 parallel cross-examination calls)
    r1_summary = json.dumps({
        m: {
            "verdict": o.get("verdict"), "conviction": o.get("conviction_0_10"),
            "weight_now": o.get("target_weight_now_pct"), "max_weight": o.get("max_weight_on_trigger_pct"),
            "action_now": o.get("action_now"), "entry_or_trigger_plan": o.get("entry_or_trigger_plan"),
            "stop_and_invalidation": o.get("stop_and_invalidation"),
            "challenge": o.get("preemptive_challenge_to_peers"),
        } for m, o in r1.items()
    }, default=str)

    r2: Dict[str, Dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futs = {}
        for m, sys_p in _ROLE_SYSTEMS.items():
            p2 = (
                f"Dossier:\n{compact}\n\nRound 1 stances of all 3 Council members:\n{r1_summary}\n\n"
                f"You are {m}. Cross-examine your two peers directly, adapt your verdict/sizing if a peer exposed a real flaw, "
                "and write your sharpest Contra Statement. Return JSON with keys: member, rebuttal_to_peer_1 (str), rebuttal_to_peer_2 (str), "
                "strongest_contra_statement (str), final_verdict (one of 'BUY NOW', 'STAGED BUY / ACCUMULATE', 'WAIT FOR TRIGGER (WATCHLIST)', 'AVOID / SELL'), "
                "final_conviction_0_10 (int), final_weight_now_pct (float), final_max_weight_pct (float), revised_plan_delta (str), what_flips_my_call (str)."
            )
            futs[m] = pool.submit(_call_llm_json, p2, sys_p, cfg)
        for m, f in futs.items():
            obj = f.result()
            obj["member"] = m
            r2[m] = obj

    # Use quant scaffold for clean playbooks/structure, overlay live LLM debate outputs
    base = run_quant_council(dossier)
    r1_verdicts = [str(r1[m].get("verdict", "WAIT FOR TRIGGER (WATCHLIST)")) for m in ("mom_trader", "garp_investor", "hybrid_pm")]
    r2_verdicts = [str(r2[m].get("final_verdict", r1_verdicts[i])) for i, m in enumerate(("mom_trader", "garp_investor", "hybrid_pm"))]
    _, r1_tally = _vote_summary(r1_verdicts)
    chair_verdict, r2_tally = _vote_summary(r2_verdicts)

    diligence = []
    for m in ("mom_trader", "garp_investor", "hybrid_pm"):
        for q in (r1[m].get("diligence_questions") or []):
            if q and q not in diligence:
                diligence.append(str(q))

    base.update({
        "mode": f"live_llm ({desc})",
        "round1_vote": r1_tally,
        "round2_final_vote": r2_tally,
        "vote_flipped": r1_tally != r2_tally,
        "chair_verdict": chair_verdict,
        "chair_headline": (
            f"Live 2-Round LLM Council shifted {r1_tally} → {r2_tally}."
            if r1_tally != r2_tally else
            f"Live 2-Round LLM Council consensus: {r2_tally}."
        ),
        "round1": r1,
        "round2": r2,
        "bear_contra_statement": " | ".join(
            str(r2[m].get("strongest_contra_statement", "")) for m in ("mom_trader", "garp_investor", "hybrid_pm") if r2[m].get("strongest_contra_statement")
        ),
        "diligence_questions": diligence[:5] or base["diligence_questions"],
    })
    return base


# ---------------------------------------------------------------------------
# 4. Markdown Exporter
# ---------------------------------------------------------------------------
def to_markdown(res: Dict[str, Any]) -> str:
    d = res["dossier"]
    pt, fx, mr, val = d["price_and_trend"], d["forensics_v6"], d["market_regime"], d["valuation_v41"]
    names = {
        "mom_trader": "Member 1: Momentum & Microstructure",
        "garp_investor": "Member 2: GARP & Valuation-Ladder",
        "hybrid_pm": "Member 3: Macro-Regime Hybrid PM",
    }
    lines = [
        f"# 🏛️ Expert Council Verdict: {res['name']} (`{res['ticker']}`)",
        f"**As of:** `{res['as_of']}` · **Mode:** `{res['mode']}`",
        "",
        f"## Chair Verdict: `{res['chair_verdict']}` ({res['round2_final_vote']})",
        f"- **Vote Shift:** `{res['round1_vote']}` → `{res['round2_final_vote']}`" + (" **(FLIP!)**" if res["vote_flipped"] else ""),
        f"- **Summary:** {res['chair_headline']}",
        "",
        "## Council Scorecard",
        "| Member | Round 1 (Wt) | Round 2 Final (Wt) | Max Wt | Conviction | Adaptation |",
        "|---|---|---|---:|---:|---|",
    ]
    for m in ("mom_trader", "garp_investor", "hybrid_pm"):
        o1, o2 = res["round1"][m], res["round2"][m]
        lines.append(
            f"| **{names[m]}** | {o1.get('verdict')} ({o1.get('target_weight_now_pct')}%) | "
            f"**{o2.get('final_verdict')}** ({o2.get('final_weight_now_pct')}%) | {o2.get('final_max_weight_pct')}% | "
            f"{o1.get('conviction_0_10')} → {o2.get('final_conviction_0_10')} | {o2.get('revised_plan_delta')} |"
        )
    lines += [
        "",
        "## Key Telemetry Snapshot",
        f"- **Price / Stage:** {_fmt_inr(pt.get('close'), 1)} · Stage {pt.get('stage')} · vs MA50 {pt.get('dist_ma50_pct')}% · vs MA200 {pt.get('dist_ma200_pct')}% · Liq ₹{pt.get('liq_20d_med_cr')} Cr/d",
        f"- **Forensics v6:** Gate-1={fx.get('gate1')} · State={fx.get('state')} · Action={fx.get('action')} · ATR {fx.get('atr_pct')}% (Gap Guard {fx.get('atr_gap_guard_pct')}%)",
        f"- **Market Regime:** Engine={mr.get('engine_mode')} · VIX={(mr.get('vix') or {}).get('close')} (P{(mr.get('vix') or {}).get('pctile_252')}) · Breadth={(mr.get('breadth_regime') or {}).get('regime')}",
        f"- **Valuation v4.1:** {val.get('rating', val.get('status'))} · Bear {_fmt_inr(val.get('bear_blend'), 0)} · P10 {_fmt_inr(val.get('p10'), 0)} · P50 {_fmt_inr(val.get('p50'), 0)} · Counterfactual {_fmt_inr(val.get('counterfactual_px'), 0)}",
        "",
        "## Explicit Contra Statements",
        f"- **🔴 Bear / Thesis-Killer Contra:** {res['bear_contra_statement']}",
        f"- **🟢 Bull / Opportunity-Cost Contra:** {res['bull_contra_statement']}",
        "",
        "## Split Actionable Playbooks",
        "| Mandate | Today | Triggers / Ladder | Stop & Invalidation | Payoff Anchor |",
        "|---|---|---|---|---|",
    ]
    for pb in res["playbooks"]:
        lines.append(f"| **{pb['mandate']}** | {pb['action_today']} | {pb['triggers']} | {pb['stop_rule']} | {pb['payoff_anchor']} |")
    lines += ["", "## External Diligence Checks"]
    for q in res["diligence_questions"]:
        lines.append(f"- {q}")
    return "\n".join(lines) + "\n"
