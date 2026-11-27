"""Experiments 1 & 2:
  - Experiment 1: 29-Stock "Poisoned Universe" Out-of-Sample Stress Test
      (11 Multibagger Winners + 18 Real March-2026 False-Positive Traps)
  - Experiment 2: Controlled Ablation of ONLY the Already-Merged forensics.py Dual-ATR Fix
      (Pre-Dual-ATR vs Post-Dual-ATR ONLY vs +ATR Gap Guard) on both the 11-Winner & 29-Poisoned universes.

100% strictly causal (<= t) point-in-time data, deterministic daily execution, 20 bps/side cost,
₹10,00,000 initial capital, full Cash Drag %, Max DD, Sharpe, Winner-vs-Trap attribution, and GARP MAE audit.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\repo_live"
OUT_DIR = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\poisoned_stress"
os.chdir(REPO)
sys.path.insert(0, REPO)

from utils import forensics as fx, fund_trends as ft, nse_prices, regime_manager as rm, valuation as V

WINNERS_11 = [
    "SIGMAADV.NS", "CUPID.NS", "STLTECH.NS", "TBZ.NS", "MONOLITH.NS",
    "SPECTRUM.NS", "HFCL.NS", "FILATEX.NS", "SSWL.NS", "LAURUSLABS.NS", "MACPOWER.NS",
]
START_DATE = "2026-02-27"
END_DATE = "2026-11-20"
COST_SIDE = 0.0020
INIT_CAP = 1_000_000.0

EVENT_DATES = [
    "2026-02-27", "2026-03-30", "2026-04-08", "2026-05-18",
    "2026-06-22", "2026-07-20", "2026-09-15", "2026-11-06",
]
MONTHLY_GATES = [
    "2026-02-27", "2026-03-30", "2026-04-30", "2026-05-29",
    "2026-06-30", "2026-07-31", "2026-08-31", "2026-09-30", "2026-10-30",
]


def _load_indices() -> Tuple[pd.DataFrame, pd.Series]:
    files = sorted(glob.glob(os.path.join(REPO, "data", "nse", "indices", "*.parquet"))
                   + glob.glob(os.path.join(REPO, "data", "nse", "indices", "d", "*.parquet")))
    parts = []
    for f in files[-36:]:
        try:
            parts.append(pd.read_parquet(f, columns=["date", "index", "open", "high", "low", "close"]))
        except Exception:
            continue
    ix = pd.concat(parts, ignore_index=True)
    ix["date"] = pd.to_datetime(ix["date"], errors="coerce")
    nifty = ix[ix["index"] == "Nifty 50"].dropna(subset=["date", "close"]).drop_duplicates("date", keep="last").set_index("date").sort_index()
    nifty = nifty.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    vix = ix[ix["index"] == "India VIX"].dropna(subset=["date", "close"]).drop_duplicates("date", keep="last").set_index("date").sort_index()["close"]
    return nifty, vix


def _load_breadth() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(REPO, "data", "nse", "prices", "2025_*.parquet"))
                   + glob.glob(os.path.join(REPO, "data", "nse", "prices", "2026_*.parquet"))
                   + glob.glob(os.path.join(REPO, "data", "nse", "prices", "d", "*.parquet")))
    parts = []
    for f in files:
        try:
            parts.append(pd.read_parquet(f, columns=["date", "symbol", "close", "low"]))
        except Exception:
            continue
    px = pd.concat(parts, ignore_index=True)
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px = px.dropna(subset=["date", "symbol", "close"])
    C = px.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    L = px.pivot_table(index="date", columns="symbol", values="low", aggfunc="last").sort_index()
    keep = C.notna().sum() >= 180
    C, L = C.loc[:, keep], L.loc[:, keep]
    ma50 = C.rolling(50, min_periods=40).mean()
    ma200 = C.rolling(200, min_periods=150).mean()
    lo252 = L.rolling(252, min_periods=180).min()
    pct50 = (C > ma50).sum(axis=1) / ma50.notna().sum(axis=1).replace(0, np.nan) * 100.0
    pct200 = (C > ma200).sum(axis=1) / ma200.notna().sum(axis=1).replace(0, np.nan) * 100.0
    pctlo = (L <= lo252 * 1.01).sum(axis=1) / lo252.notna().sum(axis=1).replace(0, np.nan) * 100.0
    return pd.DataFrame({"pct50": pct50, "pct200": pct200, "pctlo": pctlo})


def _pit_fund(tk: str, as_of: str, ann_raw: dict, qtr_raw: dict) -> dict:
    as_ts = pd.Timestamp(as_of)
    cfo_pat = fcf_pat = roce = np.nan
    k_a = ft._key(tk, ann_raw)
    if k_a is not None:
        w = ann_raw[k_a].pivot_table(index="period", columns="field", values="value", aggfunc="last").sort_index()
        w = w[w.index.month == 3]
        w = w[w.index + pd.Timedelta(days=60) <= as_ts]
        if not w.empty:
            g = lambda c: w[c] if c in w.columns else pd.Series(np.nan, index=w.index)
            gross = g("fixed_assets") + g("cwip").fillna(0.0)
            capex = gross.diff() + g("dep").fillna(0.0)
            np_v = float(g("np").iloc[-1]) if pd.notna(g("np").iloc[-1]) else np.nan
            cfo_v = float(g("cfo").iloc[-1]) if pd.notna(g("cfo").iloc[-1]) else np.nan
            cap_v = float(capex.iloc[-1]) if pd.notna(capex.iloc[-1]) else np.nan
            roce_v = g("roce").iloc[-1]
            if pd.notna(roce_v):
                roce = round(float(roce_v), 1)
            if np.isfinite(np_v) and np_v > 0 and np.isfinite(cfo_v):
                cfo_pat = round(cfo_v / np_v, 2)
                if np.isfinite(cap_v):
                    fcf_pat = round((cfo_v - max(0.0, cap_v)) / np_v, 2)
    q_np_yoy = np.nan
    k_q = ft._key(tk, qtr_raw)
    if k_q is not None:
        wq = qtr_raw[k_q].pivot_table(index="period", columns="field", values="value", aggfunc="last").sort_index()
        wq = wq[wq.index + pd.Timedelta(days=45) <= as_ts]
        if len(wq) >= 5 and "np" in wq.columns:
            cur_dt = wq.index[-1]
            prev_dt = cur_dt - pd.DateOffset(years=1)
            diffs = np.abs((wq.index - prev_dt).days)
            j = int(np.argmin(diffs))
            if diffs[j] <= 20:
                cnp, pnp = wq["np"].iloc[-1], wq["np"].iloc[j]
                if pd.notna(cnp) and pd.notna(pnp) and float(pnp) > 0:
                    q_np_yoy = round((float(cnp) / float(pnp) - 1.0) * 100.0, 1)
    return {
        "cfo_pat": None if not np.isfinite(cfo_pat) else cfo_pat,
        "fcf_pat": None if not np.isfinite(fcf_pat) else fcf_pat,
        "roce": None if not np.isfinite(roce) else roce,
        "q_np_yoy": None if not np.isfinite(q_np_yoy) else q_np_yoy,
    }


def _pre_dual_atr_action(r_post: dict, clv_last: float) -> str:
    """Reconstruct exact pre-Dual-ATR over-blocking entry guard on top of the same forensics state/supply:
    blocked BUY NOW whenever signature == 'DISTRIBUTION' OR clv < 0.35 OR stale-anchor ext_atr > 3.5.
    """
    ext_anc = r_post.get("ext_atr", np.nan)
    old_weak = bool(
        r_post.get("signature") == "DISTRIBUTION"
        or (np.isfinite(clv_last) and clv_last < 0.35)
        or (np.isfinite(ext_anc) and float(ext_anc) > fx.EXT_ATR_MAX)
    )
    return fx.action_for(bool(r_post.get("gate1")), str(r_post.get("state")), str(r_post.get("supply")), entry_weak=old_weak)


def build_daily_dataset(tickers: List[str]) -> dict:
    """Build strictly causal daily bars, forensics (PRE & POST Dual-ATR), regimes, and checkpoint valuations."""
    print(f"Building strictly causal daily dataset for {len(tickers)} stocks ({START_DATE} -> {END_DATE})...")
    hist = nse_prices.history(tickers)
    nifty_df, vix_s = _load_indices()
    breadth = _load_breadth()
    mc = pd.read_csv(os.path.join(REPO, "data", "nse", "market_context.csv"))
    mc["date"] = pd.to_datetime(mc["date"], errors="coerce")
    mc = mc.dropna(subset=["date"]).set_index("date").sort_index()
    ann_raw, qtr_raw = ft._data(REPO)
    val_ctx = V.load_context()

    ref_df = hist["LAURUSLABS.NS"].sort_index()
    dates = [d.strftime("%Y-%m-%d") for d in ref_df.loc[START_DATE:END_DATE].index]

    # 1. Daily regimes (strictly causal <= t)
    daily_regimes: Dict[str, dict] = {}
    for dt_str in dates:
        dt = pd.Timestamp(dt_str)
        nd = nifty_df.loc[:dt]
        c_last = float(nd["Close"].iloc[-1])
        m200 = float(nd["Close"].rolling(200).mean().iloc[-1])
        hi52 = float(nd["High"][nd.index > dt - pd.Timedelta(days=365)].max())
        base_mode = rm.classify_regime(c_last, m200, hi52)
        b_sub = breadth.loc[:dt]
        mc_sub = mc.loc[:dt]
        rec = rm.detect_capitulation_recovery(
            pct_above_50dma=b_sub["pct50"].dropna() if not b_sub.empty else None,
            pct_52w_lows=b_sub["pctlo"].dropna() if not b_sub.empty else None,
            fii_net=mc_sub["fii_net"].dropna() if ("fii_net" in mc_sub and not mc_sub.empty) else None,
            dii_net=mc_sub["dii_net"].dropna() if ("dii_net" in mc_sub and not mc_sub.empty) else None,
            fii_long_pct=mc_sub["fii_idx_long_pct"].dropna() if ("fii_idx_long_pct" in mc_sub and not mc_sub.empty) else None,
        )
        eff_mode = "RECOVERY" if (base_mode in ("BEAR", "DEFENSIVE", "CAUTION") and rec["active"]) else base_mode
        vix_sub = vix_s.loc[:dt]
        vix_v = round(float(vix_sub.iloc[-1]), 2) if not vix_sub.empty else None
        daily_regimes[dt_str] = {
            "base_mode": base_mode,
            "engine_mode": eff_mode,
            "recovery": bool(rec["active"]),
            "vix": vix_v,
            "pct50": round(float(b_sub["pct50"].iloc[-1]), 1) if not b_sub.empty else None,
            "pct200": round(float(b_sub["pct200"].iloc[-1]), 1) if not b_sub.empty else None,
            "nifty_close": round(c_last, 2),
        }

    # 2. Daily per-stock traces + PRE vs POST Dual-ATR actions + causal checkpoints fundamentals/valuation
    ckpt_dates = sorted(set(EVENT_DATES + MONTHLY_GATES))
    bars: Dict[str, Dict[str, dict]] = {}
    signal_counts: Dict[str, dict] = {}

    for tk in tickers:
        df = hist[tk].sort_index()
        c_s = df["Close"].astype(float)
        ma50_s = c_s.rolling(50).mean()
        ma200_s = c_s.rolling(200).mean()
        tmed20_s = (c_s * df["Volume"].astype(float)).rolling(20).median() / 1e7
        # Causal RS vs Nifty
        rs_join = pd.concat([c_s.rename("s"), nifty_df["Close"].astype(float).rename("n")], axis=1).ffill()
        rs_1m_s = ((rs_join["s"] / rs_join["s"].shift(21)) / (rs_join["n"] / rs_join["n"].shift(21)) - 1.0) * 100.0
        rs_3m_s = ((rs_join["s"] / rs_join["s"].shift(63)) / (rs_join["n"] / rs_join["n"].shift(63)) - 1.0) * 100.0
        rs_6m_s = ((rs_join["s"] / rs_join["s"].shift(126)) / (rs_join["n"] / rs_join["n"].shift(126)) - 1.0) * 100.0
        comp_rs_s = 0.25 * rs_1m_s.fillna(0) + 0.35 * rs_3m_s.fillna(0) + 0.40 * rs_6m_s.fillna(0)

        # Precompute checkpoint fundamentals + Valuation v4.1 (and forward-fill strictly causally!)
        fund_by_ckpt = {}
        val_by_ckpt = {}
        for cd in ckpt_dates:
            fund_by_ckpt[cd] = _pit_fund(tk, cd, ann_raw, qtr_raw)
            px_cd = float(df.loc[:cd, "Close"].iloc[-1])
            vb = {"bear_blend": None, "p50": None, "rating": "NO DATA", "cmp_lt_bear": False}
            try:
                v = V.value_stock(tk.replace(".NS", ""), val_ctx, price=px_cd, as_of=cd, knobs=False)
                if v.get("status") == "OK":
                    bl = v.get("blends") or {}
                    bb = bl.get("bear", np.nan)
                    if np.isfinite(bb):
                        vb["bear_blend"] = round(float(bb), 2)
                        vb["cmp_lt_bear"] = bool(px_cd <= float(bb))
                    vb["p50"] = round(float(v["dist"]["P50"]), 2)
                    vb["rating"] = str(v.get("rating") or "")
            except Exception:
                pass
            val_by_ckpt[cd] = vb

        bars[tk] = {}
        pre_bn = post_bn = go_days = 0
        last_ckpt = ckpt_dates[0]
        for dt_str in dates:
            if dt_str in fund_by_ckpt:
                last_ckpt = dt_str
            sub = df.loc[:dt_str]
            row = sub.iloc[-1]
            px = float(row["Close"])
            hi = float(row["High"])
            lo = float(row["Low"])
            op = float(row["Open"])
            clv_last = (px - lo) / (hi - lo) if hi > lo else 0.5
            r_post = fx.analyse(sub)
            act_post = str(r_post.get("action") or "WATCH")
            act_pre = _pre_dual_atr_action(r_post, clv_last)
            if r_post.get("state") == "GO":
                go_days += 1
            if act_pre == "BUY NOW":
                pre_bn += 1
            if act_post == "BUY NOW":
                post_bn += 1

            m50 = float(ma50_s.loc[sub.index[-1]]) if pd.notna(ma50_s.loc[sub.index[-1]]) else np.nan
            m200 = float(ma200_s.loc[sub.index[-1]]) if pd.notna(ma200_s.loc[sub.index[-1]]) else np.nan
            stage = 2 if (np.isfinite(m200) and px > m50 > m200) else (4 if (np.isfinite(m200) and px < m50 < m200) else (1 if (np.isfinite(m50) and px > m50) else 3))
            atr_v = float(r_post.get("atr", np.nan)) if np.isfinite(r_post.get("atr", np.nan)) else px * 0.035
            atr_pct = round(atr_v / px * 100.0, 2) if px > 0 else 3.5
            bbp_v = float(r_post.get("bbp", np.nan)) if np.isfinite(r_post.get("bbp", np.nan)) else None
            liq_v = float(tmed20_s.loc[sub.index[-1]]) if pd.notna(tmed20_s.loc[sub.index[-1]]) else 0.0
            crs = float(comp_rs_s.loc[sub.index[-1]]) if sub.index[-1] in comp_rs_s.index and pd.notna(comp_rs_s.loc[sub.index[-1]]) else 0.0
            f_pit = fund_by_ckpt[last_ckpt]
            v_pit = val_by_ckpt[last_ckpt]
            bb_val = v_pit["bear_blend"]

            bars[tk][dt_str] = {
                "open": round(op, 2), "high": round(hi, 2), "low": round(lo, 2), "close": round(px, 2),
                "ma50": round(m50, 2) if np.isfinite(m50) else None,
                "ma200": round(m200, 2) if np.isfinite(m200) else None,
                "stage": stage, "liq_cr": round(liq_v, 2), "comp_rs": round(crs, 1),
                "atr": round(atr_v, 2), "atr_pct": atr_pct, "bbp": round(bbp_v, 1) if bbp_v is not None else None,
                "gate1": bool(r_post.get("gate1", False)), "state": str(r_post.get("state") or ""),
                "action_pre": act_pre, "action_post": act_post,
                "stop_fx": round(float(r_post["stop"]), 2) if np.isfinite(r_post.get("stop", np.nan)) else None,
                "entry_fx": round(float(r_post["entry"]), 2) if np.isfinite(r_post.get("entry", np.nan)) else round(px, 2),
                "cfo_pat": f_pit["cfo_pat"], "fcf_pat": f_pit["fcf_pat"], "roce": f_pit["roce"], "q_np_yoy": f_pit["q_np_yoy"],
                "bear_blend": bb_val, "cmp_lt_bear": bool(bb_val is not None and px <= bb_val),
                "val_rating": v_pit["rating"],
            }
        signal_counts[tk] = {"go_days": go_days, "pre_buy_now": pre_bn, "post_buy_now": post_bn}
    return {"dates": dates, "tickers": tickers, "regimes": daily_regimes, "bars": bars, "signal_counts": signal_counts}


def simulate_strategy(
    data: dict,
    tickers: List[str],
    winners_set: set,
    strategy_name: str,
    *,
    decision_dates: Optional[set] = None,  # None => daily-reactive; set => event/monthly cadence
    max_pos_pct: float = 20.0,
    forensics_mode: str = "post",          # "pre" or "post"
    atr_gap_guard: bool = False,
    garp_cfo_gate: bool = True,
    garp_catastrophe_stop: Optional[float] = -25.0,
    hybrid_strict_stage2_bbp: bool = False,
) -> dict:
    """Strictly causal deterministic simulator with full attribution, Cash Drag %, and Tranche MAE."""
    dates = data["dates"]
    bars = data["bars"]
    regimes = data["regimes"]

    cash = INIT_CAP
    positions: Dict[str, dict] = {}
    pending_orders: Dict[str, dict] = {}  # tk -> next-open BUY/EXIT
    pending_triggers: Dict[str, dict] = {}
    trades: List[dict] = []
    tranches: List[dict] = []
    eq_curve: List[dict] = []
    gap_skips = 0

    def _nav(dt: str, field: str = "close") -> float:
        return cash + sum(p["shares"] * bars[tk][dt][field] for tk, p in positions.items())

    def _record_tranche(tk: str, dt: str, entry_px: float, sh: int, reason: str, sig_dt: str):
        b = bars[tk][sig_dt]
        tranches.append({
            "ticker": tk, "is_winner": tk in winners_set, "entry_date": dt, "entry_px": round(entry_px, 2),
            "shares": sh, "reason": reason, "bear_blend": b["bear_blend"],
            "cmp_lt_bear": bool(b["bear_blend"] is not None and entry_px <= b["bear_blend"]),
            "cfo_pat": b["cfo_pat"], "fcf_pat": b["fcf_pat"],
            "cfo_ge_08": bool(b["cfo_pat"] is not None and b["cfo_pat"] >= 0.80),
            "min_low": round(entry_px, 2), "min_close": round(entry_px, 2), "max_high": round(entry_px, 2),
            "mae_low_pct": 0.0, "mae_close_pct": 0.0, "mfe_high_pct": 0.0,
            "hit_minus25": False, "closed_date": None, "exit_px": None,
        })

    def _close_tranches(tk: str, dt: str, exit_px: float):
        for tr in tranches:
            if tr["ticker"] == tk and tr["closed_date"] is None:
                tr["closed_date"] = dt
                tr["exit_px"] = round(exit_px, 2)

    for idx, dt in enumerate(dates):
        # 1. Execute queued next-open exits first, then buys
        for tk, o in sorted(pending_orders.items(), key=lambda kv: 0 if kv[1]["side"] == "SELL" else 1):
            px_op = bars[tk][dt]["open"]
            if o["side"] == "SELL" and tk in positions:
                p = positions.pop(tk)
                net = p["shares"] * px_op * (1.0 - COST_SIDE)
                cash += net
                _close_tranches(tk, dt, px_op)
                trades.append({
                    "date": dt, "ticker": tk, "is_winner": tk in winners_set, "side": "SELL",
                    "reason": o["reason"], "px": round(px_op, 2), "shares": p["shares"],
                    "pnl_pct": round((px_op / p["avg_px"] - 1.0) * 100.0, 2),
                    "pnl_rs": round(net - p["cost_basis"], 0),
                })
            elif o["side"] == "BUY":
                gap_pct = (px_op / o["ref_px"] - 1.0) * 100.0 if o.get("ref_px") else 0.0
                if o.get("max_gap_pct") is not None and gap_pct > o["max_gap_pct"]:
                    gap_skips += 1
                    continue
                cur_eq = _nav(dt, "open")
                tgt_val = cur_eq * (min(max_pos_pct, o["target_pct"]) / 100.0)
                cur_val = positions[tk]["shares"] * px_op if tk in positions else 0.0
                buy_val = min(max(0.0, tgt_val - cur_val), cash / (1.0 + COST_SIDE))
                sh = int(buy_val / px_op)
                if sh > 0:
                    cost = sh * px_op * (1.0 + COST_SIDE)
                    cash -= cost
                    if tk in positions:
                        p = positions[tk]
                        p["shares"] += sh
                        p["cost_basis"] += cost
                        p["avg_px"] = p["cost_basis"] / p["shares"]
                        p["peak_close"] = max(p["peak_close"], px_op)
                    else:
                        positions[tk] = {
                            "entry_date": dt, "shares": sh, "cost_basis": cost, "avg_px": cost / sh,
                            "peak_close": px_op, "stop_px": o.get("stop_px"), "trail_pct": o.get("trail_pct"),
                            "use_ma50": bool(o.get("use_ma50", False)), "catastrophe_pct": o.get("catastrophe_pct"),
                        }
                    _record_tranche(tk, dt, px_op, sh, o["reason"], o["sig_dt"])
                    trades.append({
                        "date": dt, "ticker": tk, "is_winner": tk in winners_set, "side": "BUY",
                        "reason": o["reason"], "px": round(px_op, 2), "shares": sh, "value": round(sh * px_op, 0),
                    })
        pending_orders.clear()

        # 2. Execute armed trigger entries
        for tk, trig in list(pending_triggers.items()):
            if trig.get("armed") and tk not in positions:
                px_op = bars[tk][dt]["open"]
                gap_pct = (px_op / trig["trigger_px"] - 1.0) * 100.0
                if gap_pct <= trig["max_gap_pct"]:
                    cur_eq = _nav(dt, "open")
                    tgt_val = cur_eq * (min(max_pos_pct, trig["target_pct"]) / 100.0)
                    buy_val = min(tgt_val, cash / (1.0 + COST_SIDE))
                    sh = int(buy_val / px_op)
                    if sh > 0:
                        cost = sh * px_op * (1.0 + COST_SIDE)
                        cash -= cost
                        positions[tk] = {
                            "entry_date": dt, "shares": sh, "cost_basis": cost, "avg_px": cost / sh,
                            "peak_close": px_op, "stop_px": trig.get("stop_px"), "trail_pct": trig.get("trail_pct"),
                            "use_ma50": bool(trig.get("use_ma50", False)), "catastrophe_pct": None,
                        }
                        _record_tranche(tk, dt, px_op, sh, f"TRIGGER@{trig['trigger_px']}", trig["sig_dt"])
                        trades.append({
                            "date": dt, "ticker": tk, "is_winner": tk in winners_set, "side": "BUY",
                            "reason": "TRIGGER", "px": round(px_op, 2), "shares": sh, "value": round(sh * px_op, 0),
                        })
                else:
                    gap_skips += 1
                del pending_triggers[tk]

        # 3. Update active tranche MAE/MFE + Intraday hard/catastrophe stops + EOD trail/MA50 exits
        for tr in tranches:
            if tr["closed_date"] is None and tr["ticker"] in positions:
                b = bars[tr["ticker"]][dt]
                if b["low"] < tr["min_low"]:
                    tr["min_low"] = b["low"]
                    tr["mae_low_pct"] = round((b["low"] / tr["entry_px"] - 1.0) * 100.0, 2)
                if b["close"] < tr["min_close"]:
                    tr["min_close"] = b["close"]
                    tr["mae_close_pct"] = round((b["close"] / tr["entry_px"] - 1.0) * 100.0, 2)
                if b["high"] > tr["max_high"]:
                    tr["max_high"] = b["high"]
                    tr["mfe_high_pct"] = round((b["high"] / tr["entry_px"] - 1.0) * 100.0, 2)
                if tr["mae_low_pct"] <= -25.0:
                    tr["hit_minus25"] = True

        for tk in list(positions.keys()):
            p = positions[tk]
            b = bars[tk][dt]
            # Intraday hard stop OR catastrophe -25% circuit breaker
            cat_stop = p["avg_px"] * (1.0 + p["catastrophe_pct"] / 100.0) if p.get("catastrophe_pct") is not None else None
            eff_stop = max([x for x in (p.get("stop_px"), cat_stop) if x is not None], default=None)
            if eff_stop is not None and b["low"] <= eff_stop:
                fill_px = min(b["open"], eff_stop)
                net = p["shares"] * fill_px * (1.0 - COST_SIDE)
                cash += net
                _close_tranches(tk, dt, fill_px)
                rsn = "CATASTROPHE_-25%" if (cat_stop is not None and eff_stop == cat_stop) else f"HARD_STOP@{round(eff_stop, 1)}"
                trades.append({
                    "date": dt, "ticker": tk, "is_winner": tk in winners_set, "side": "SELL",
                    "reason": rsn, "px": round(fill_px, 2), "shares": p["shares"],
                    "pnl_pct": round((fill_px / p["avg_px"] - 1.0) * 100.0, 2),
                    "pnl_rs": round(net - p["cost_basis"], 0),
                })
                del positions[tk]
                continue

            if b["close"] > p["peak_close"]:
                p["peak_close"] = b["close"]
            if p.get("use_ma50") and b["ma50"] is not None and b["close"] < b["ma50"]:
                pending_orders[tk] = {"side": "SELL", "reason": "MA50_BREAK"}
            elif p.get("trail_pct") is not None and b["close"] < p["peak_close"] * (1.0 - p["trail_pct"]):
                pending_orders[tk] = {"side": "SELL", "reason": f"TRAIL_{int(p['trail_pct']*100)}%"}

        for tk, trig in pending_triggers.items():
            if tk not in positions and not trig.get("armed"):
                if bars[tk][dt]["close"] >= trig["trigger_px"]:
                    trig["armed"] = True

        # 4. EOD Strategy Decision (strictly causal <= dt; executes next open!)
        reg = regimes[dt]
        emode = reg["engine_mode"]
        is_decision_day = (decision_dates is None) or (dt in decision_dates)
        if is_decision_day and idx < len(dates) - 1:
            if decision_dates is not None:
                pending_triggers.clear()  # refresh triggers at each event/monthly checkpoint
            trail = 0.15 if emode in ("BULL", "RECOVERY") else 0.12
            act_field = "action_pre" if forensics_mode == "pre" else "action_post"

            if strategy_name in ("mom_trader", "monthly_quant"):
                # Pure Forensics BUY NOW / BUY ON TRIGGER + Stage 2 + RS > 0 + liq >= 3 Cr
                cands = []
                for tk in tickers:
                    b = bars[tk][dt]
                    if b["liq_cr"] < 3.0:
                        continue
                    act = b[act_field]
                    if act == "BUY NOW" and b["stage"] == 2 and b["comp_rs"] > 0:
                        cands.append((b["comp_rs"], tk, "BUY_NOW", b))
                    elif act == "BUY ON TRIGGER" and b["stage"] == 2 and b["comp_rs"] > 0:
                        cands.append((b["comp_rs"], tk, "TRIGGER", b))
                cands.sort(reverse=True)
                pos_sz = 14.0 if emode == "BEAR" else (18.0 if emode == "CAUTION" else max_pos_pct)
                for _, tk, kind, b in cands:
                    if tk in positions or tk in pending_orders:
                        continue
                    gap_lim = max(5.0, round(1.5 * b["atr_pct"], 1)) if atr_gap_guard else 5.0
                    stp = b["stop_fx"] if (b["stop_fx"] and b["stop_fx"] < b["close"]) else round(b["close"] - 2.0 * b["atr"], 2)
                    if kind == "BUY_NOW":
                        pending_orders[tk] = {
                            "side": "BUY", "reason": f"FX_{forensics_mode.upper()}_BUYNOW", "sig_dt": dt,
                            "target_pct": pos_sz, "stop_px": stp, "trail_pct": trail, "use_ma50": True,
                            "ref_px": b["close"], "max_gap_pct": gap_lim,
                        }
                    else:
                        pending_triggers[tk] = {
                            "trigger_px": b["entry_fx"], "target_pct": pos_sz, "stop_px": stp,
                            "trail_pct": trail, "use_ma50": True, "max_gap_pct": gap_lim, "sig_dt": dt, "armed": False,
                        }

            elif strategy_name == "garp_investor":
                # Exit existing GARP holdings if fundamental CFO/PAT turns negative (HFCL rule!)
                for tk in list(positions.keys()):
                    b = bars[tk][dt]
                    if garp_cfo_gate and b["cfo_pat"] is not None and b["cfo_pat"] < 0.0:
                        pending_orders[tk] = {"side": "SELL", "reason": f"FUND_CFO_NEG({b['cfo_pat']})"}
                # Buy CMP <= Bear Blend (or <= 1.03*Bear in RECOVERY) passing primary CFO/PAT >= 0.80 gate
                cands = []
                for tk in tickers:
                    b = bars[tk][dt]
                    if b["liq_cr"] < 3.0 or b["bear_blend"] is None:
                        continue
                    bear_ok = b["close"] <= b["bear_blend"] * (1.03 if emode == "RECOVERY" else 1.00)
                    if not bear_ok:
                        continue
                    if garp_cfo_gate:
                        if b["cfo_pat"] is None or b["cfo_pat"] < 0.80:
                            continue
                    disc = (1.0 - b["close"] / b["bear_blend"]) * 100.0
                    cands.append((disc, tk, b))
                cands.sort(reverse=True)
                for _, tk, b in cands:
                    if tk in pending_orders:
                        continue
                    cur_w = (positions[tk]["shares"] * b["close"] / max(_nav(dt), 1.0) * 100.0) if tk in positions else 0.0
                    if cur_w < max_pos_pct - 2.0:
                        pending_orders[tk] = {
                            "side": "BUY", "reason": "GARP_LT_BEAR", "sig_dt": dt,
                            "target_pct": min(max_pos_pct, cur_w + 10.0), "stop_px": None,
                            "trail_pct": None, "use_ma50": False, "catastrophe_pct": garp_catastrophe_stop,
                            "ref_px": b["close"], "max_gap_pct": None,
                        }

            elif strategy_name == "hybrid_pm":
                # Core GARP (<Bear + CFO/PAT >= 0.80) + Tactical Momentum (FX BUY NOW OR BBP < 20 squeeze)
                for tk in list(positions.keys()):
                    b = bars[tk][dt]
                    if positions[tk].get("catastrophe_pct") is not None and b["cfo_pat"] is not None and b["cfo_pat"] < 0.0:
                        pending_orders[tk] = {"side": "SELL", "reason": f"HYB_CFO_NEG({b['cfo_pat']})"}

                cands = []
                for tk in tickers:
                    b = bars[tk][dt]
                    if b["liq_cr"] < 3.0:
                        continue
                    # Leg 1: Core GARP Value Floor
                    if b["bear_blend"] is not None and b["close"] <= b["bear_blend"] and b["cfo_pat"] is not None and b["cfo_pat"] >= 0.80:
                        cands.append((100.0 + (b["roce"] or 0.0), tk, "HYB_GARP_BEAR", b))
                        continue
                    # Leg 2: Tactical Momentum Breakout (Forensics POST Dual-ATR BUY NOW)
                    if b["action_post"] == "BUY NOW" and b["stage"] == 2 and b["comp_rs"] > 0:
                        cands.append((50.0 + b["comp_rs"], tk, "HYB_FX_BREAKOUT", b))
                        continue
                    # Leg 3: BBP < 20 Tight Squeeze (Strict Stage-2+RS>0 vs Loose BEAR/RECOVERY contrarian)
                    if b["bbp"] is not None and b["bbp"] < 20.0:
                        if hybrid_strict_stage2_bbp:
                            if b["stage"] == 2 and b["comp_rs"] > 5.0 and emode in ("RECOVERY", "CAUTION", "BULL"):
                                cands.append((30.0 + b["comp_rs"], tk, "HYB_BBP_STRICT", b))
                        else:
                            # Loose contrarian BBP<20 + trailing NP YoY>=30% (tests trap exposure!)
                            if (b["stage"] in (1, 2) and b["comp_rs"] > 0) or (emode in ("BEAR", "RECOVERY") and (b["q_np_yoy"] or 0) >= 30.0):
                                cands.append((25.0 + (b["comp_rs"] or 0), tk, "HYB_BBP_LOOSE", b))
                cands.sort(reverse=True)
                for _, tk, kind, b in cands:
                    if tk in positions or tk in pending_orders:
                        continue
                    gap_lim = max(5.0, round(1.5 * b["atr_pct"], 1))
                    if kind == "HYB_GARP_BEAR":
                        pending_orders[tk] = {
                            "side": "BUY", "reason": kind, "sig_dt": dt, "target_pct": min(max_pos_pct, 15.0),
                            "stop_px": None, "trail_pct": 0.20, "use_ma50": False, "catastrophe_pct": -25.0,
                            "ref_px": b["close"], "max_gap_pct": None,
                        }
                    else:
                        stp = round(b["close"] - 2.0 * b["atr"], 2)
                        pending_orders[tk] = {
                            "side": "BUY", "reason": kind, "sig_dt": dt, "target_pct": min(max_pos_pct, 16.0),
                            "stop_px": stp, "trail_pct": trail, "use_ma50": True, "catastrophe_pct": None,
                            "ref_px": b["close"], "max_gap_pct": gap_lim,
                        }

        nav_dt = _nav(dt, "close")
        eq_curve.append({
            "date": dt, "nav": round(nav_dt, 2), "cash": round(cash, 2),
            "cash_pct": round(cash / nav_dt * 100.0, 2) if nav_dt > 0 else 100.0,
            "engine_mode": emode,
        })

    # Compute final summary metrics
    ec_df = pd.DataFrame(eq_curve).set_index("date")
    nav_s = ec_df["nav"]
    rets = nav_s.pct_change().dropna()
    tot_ret = round((float(nav_s.iloc[-1]) / INIT_CAP - 1.0) * 100.0, 2)
    mdd = round(float((nav_s / nav_s.cummax() - 1.0).min() * 100.0), 2)
    sd = float(rets.std(ddof=1)) if len(rets) > 2 else 0.0
    sharpe = round(float((rets.mean() / sd) * math.sqrt(252.0)), 3) if sd > 1e-12 else 0.0
    cash_mean = round(float(ec_df["cash_pct"].mean()), 1)
    cash_by_reg = {m: round(float(g["cash_pct"].mean()), 1) for m, g in ec_df.groupby("engine_mode")}

    # Winner vs Trap P&L attribution (realized + open MTM)
    end_dt = dates[-1]
    win_pnl = trap_pnl = 0.0
    win_buys = trap_buys = 0
    for t in trades:
        if t["side"] == "BUY":
            if t["is_winner"]:
                win_buys += 1
            else:
                trap_buys += 1
        elif t["side"] == "SELL":
            if t["is_winner"]:
                win_pnl += float(t["pnl_rs"])
            else:
                trap_pnl += float(t["pnl_rs"])
    for tk, p in positions.items():
        unrl = p["shares"] * bars[tk][end_dt]["close"] - p["cost_basis"]
        if tk in winners_set:
            win_pnl += unrl
        else:
            trap_pnl += unrl

    # Finalize tranche returns
    for tr in tranches:
        ex_px = bars[tr["ticker"]][end_dt]["close"] if tr["closed_date"] is None else tr["exit_px"]
        tr["final_ret_pct"] = round((ex_px / tr["entry_px"] - 1.0) * 100.0, 2)

    tot_buys = win_buys + trap_buys
    return {
        "final_nav": round(float(nav_s.iloc[-1]), 0),
        "total_ret_pct": tot_ret,
        "max_dd_pct": mdd,
        "sharpe": sharpe,
        "cash_drag_mean": cash_mean,
        "cash_by_regime": cash_by_reg,
        "total_buys": tot_buys,
        "win_buys": win_buys,
        "trap_buys": trap_buys,
        "trap_hit_rate_pct": round(trap_buys / tot_buys * 100.0, 1) if tot_buys > 0 else 0.0,
        "winner_pnl_rs": round(win_pnl, 0),
        "trap_pnl_rs": round(trap_pnl, 0),
        "gap_skips": gap_skips,
        "tranches": tranches,
    }


def _passive_ew(data: dict, tickers: List[str]) -> dict:
    dates = data["dates"]
    bars = data["bars"]
    alloc = INIT_CAP / len(tickers)
    shares = {tk: alloc / bars[tk][dates[0]]["close"] for tk in tickers}
    navs = [sum(shares[tk] * bars[tk][d]["close"] for tk in tickers) for d in dates]
    s = pd.Series(navs)
    r = s.pct_change().dropna()
    return {
        "final_nav": round(float(s.iloc[-1]), 0),
        "total_ret_pct": round((float(s.iloc[-1]) / INIT_CAP - 1.0) * 100.0, 2),
        "max_dd_pct": round(float((s / s.cummax() - 1.0).min() * 100.0), 2),
        "sharpe": round(float((r.mean() / r.std(ddof=1)) * math.sqrt(252.0)), 3) if r.std(ddof=1) > 1e-12 else 0.0,
    }


def main():
    with open(os.path.join(OUT_DIR, "traps_mar2026.json"), "r", encoding="utf-8") as f:
        traps_meta = json.load(f)
    traps_18 = [r["ticker"] for r in traps_meta["selected_18"]]
    poisoned_29 = WINNERS_11 + traps_18
    winners_set = set(WINNERS_11)

    data = build_daily_dataset(poisoned_29)

    # =====================================================================
    # EXPERIMENT 2: PURE forensics.py DUAL-ATR ABLATION (ZERO OTHER CHANGES)
    # =====================================================================
    exp2 = {}
    for u_name, u_tks in [("WINNERS_11", WINNERS_11), ("POISONED_29", poisoned_29)]:
        for cad_name, cad_dates in [("EVENT_8", set(EVENT_DATES)), ("DAILY", None)]:
            arm_pre = simulate_strategy(data, u_tks, winners_set, "mom_trader", decision_dates=cad_dates, forensics_mode="pre", atr_gap_guard=False)
            arm_post = simulate_strategy(data, u_tks, winners_set, "mom_trader", decision_dates=cad_dates, forensics_mode="post", atr_gap_guard=False)
            arm_post_gap = simulate_strategy(data, u_tks, winners_set, "mom_trader", decision_dates=cad_dates, forensics_mode="post", atr_gap_guard=True)
            exp2[f"{u_name}_{cad_name}"] = {
                "pre_dual_atr": {k: v for k, v in arm_pre.items() if k != "tranches"},
                "post_dual_atr_only": {k: v for k, v in arm_post.items() if k != "tranches"},
                "post_dual_atr_plus_atr_gap": {k: v for k, v in arm_post_gap.items() if k != "tranches"},
            }

    # Signal counts summary (Winners vs Traps)
    win_pre_bn = sum(data["signal_counts"][tk]["pre_buy_now"] for tk in WINNERS_11)
    win_post_bn = sum(data["signal_counts"][tk]["post_buy_now"] for tk in WINNERS_11)
    trap_pre_bn = sum(data["signal_counts"][tk]["pre_buy_now"] for tk in traps_18)
    trap_post_bn = sum(data["signal_counts"][tk]["post_buy_now"] for tk in traps_18)

    # =====================================================================
    # EXPERIMENT 1: FULL 29-STOCK POISONED UNIVERSE STRESS TEST
    # =====================================================================
    ev_set = set(EVENT_DATES)
    mo_set = set(MONTHLY_GATES)
    exp1 = {
        "mom_pre_event8": simulate_strategy(data, poisoned_29, winners_set, "mom_trader", decision_dates=ev_set, forensics_mode="pre", atr_gap_guard=False),
        "mom_post_only_event8": simulate_strategy(data, poisoned_29, winners_set, "mom_trader", decision_dates=ev_set, forensics_mode="post", atr_gap_guard=False),
        "mom_post_only_daily": simulate_strategy(data, poisoned_29, winners_set, "mom_trader", decision_dates=None, forensics_mode="post", atr_gap_guard=False),
        "garp_cfo_gate_minus25": simulate_strategy(data, poisoned_29, winners_set, "garp_investor", decision_dates=ev_set, garp_cfo_gate=True, garp_catastrophe_stop=-25.0),
        "garp_cfo_gate_no_stop": simulate_strategy(data, poisoned_29, winners_set, "garp_investor", decision_dates=ev_set, garp_cfo_gate=True, garp_catastrophe_stop=None),
        "garp_unfiltered_no_cfo_no_stop": simulate_strategy(data, poisoned_29, winners_set, "garp_investor", decision_dates=ev_set, garp_cfo_gate=False, garp_catastrophe_stop=None),
        "garp_unfiltered_no_cfo_minus25": simulate_strategy(data, poisoned_29, winners_set, "garp_investor", decision_dates=ev_set, garp_cfo_gate=False, garp_catastrophe_stop=-25.0),
        "hybrid_pm_strict_stage2_bbp": simulate_strategy(data, poisoned_29, winners_set, "hybrid_pm", decision_dates=ev_set, hybrid_strict_stage2_bbp=True),
        "hybrid_pm_loose_bbp": simulate_strategy(data, poisoned_29, winners_set, "hybrid_pm", decision_dates=ev_set, hybrid_strict_stage2_bbp=False),
        "hybrid_pm_strict_10pct_cap": simulate_strategy(data, poisoned_29, winners_set, "hybrid_pm", decision_dates=ev_set, max_pos_pct=10.0, hybrid_strict_stage2_bbp=True),
        "monthly_quant_post": simulate_strategy(data, poisoned_29, winners_set, "monthly_quant", decision_dates=mo_set, forensics_mode="post", atr_gap_guard=False),
    }

    benchmarks = {
        "BH_POISONED_29": _passive_ew(data, poisoned_29),
        "BH_TRAPS_18": _passive_ew(data, traps_18),
        "BH_WINNERS_11": _passive_ew(data, WINNERS_11),
    }

    out = {
        "signal_counts_summary": {
            "winners_11_pre_buy_now": win_pre_bn,
            "winners_11_post_buy_now": win_post_bn,
            "traps_18_pre_buy_now": trap_pre_bn,
            "traps_18_post_buy_now": trap_post_bn,
            "per_ticker": data["signal_counts"],
        },
        "benchmarks": benchmarks,
        "experiment_2_ablation": exp2,
        "experiment_1_poisoned_29": {k: {kk: vv for kk, vv in v.items() if kk != "tranches"} for k, v in exp1.items()},
        "garp_tranches_cfo_ok": exp1["garp_cfo_gate_minus25"]["tranches"],
        "garp_tranches_unfiltered": exp1["garp_unfiltered_no_cfo_no_stop"]["tranches"],
        "hybrid_loose_tranches": exp1["hybrid_pm_loose_bbp"]["tranches"],
        "hybrid_strict_tranches": exp1["hybrid_pm_strict_stage2_bbp"]["tranches"],
    }

    out_path = os.path.join(OUT_DIR, "experiments_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n================ SIGNAL COUNTS (PRE vs POST DUAL-ATR) ================")
    print(f"11 Winners BUY NOW days: {win_pre_bn} -> {win_post_bn} ({win_post_bn - win_pre_bn:+d}, {win_post_bn/max(win_pre_bn,1):.2f}x)")
    print(f"18 Traps   BUY NOW days: {trap_pre_bn} -> {trap_post_bn} ({trap_post_bn - trap_pre_bn:+d})")

    print("\n================ EXPERIMENT 2: DUAL-ATR ABLATION ================")
    for grp, arms in exp2.items():
        print(f"\n--- {grp} ---")
        for arm_k, m in arms.items():
            print(
                f"  {arm_k:28s} | Ret: {m['total_ret_pct']:+7.2f}% | MaxDD: {m['max_dd_pct']:6.2f}% | "
                f"Sharpe: {m['sharpe']:5.3f} | CashDrag: {m['cash_drag_mean']:5.1f}% | "
                f"Buys(W/T): {m['win_buys']:2d}/{m['trap_buys']:2d} | WinPnL: {m['winner_pnl_rs']:+9.0f} | TrapPnL: {m['trap_pnl_rs']:+8.0f} | GapSkip: {m['gap_skips']}"
            )

    print("\n================ EXPERIMENT 1: POISONED 29-STOCK STRESS TEST ================")
    for bk, bm in benchmarks.items():
        print(f"  BENCH {bk:24s} | Ret: {bm['total_ret_pct']:+7.2f}% | MaxDD: {bm['max_dd_pct']:6.2f}% | Sharpe: {bm['sharpe']:5.3f}")
    print("-" * 120)
    for sk, m in out["experiment_1_poisoned_29"].items():
        print(
            f"  {sk:30s} | Ret: {m['total_ret_pct']:+7.2f}% | MaxDD: {m['max_dd_pct']:6.2f}% | "
            f"Sharpe: {m['sharpe']:5.3f} | CashDrag: {m['cash_drag_mean']:5.1f}% | "
            f"TrapHit: {m['trap_hit_rate_pct']:4.1f}% ({m['trap_buys']:2d}/{m['total_buys']:2d}) | "
            f"WinPnL: {m['winner_pnl_rs']:+9.0f} | TrapPnL: {m['trap_pnl_rs']:+8.0f}"
        )


if __name__ == "__main__":
    main()
