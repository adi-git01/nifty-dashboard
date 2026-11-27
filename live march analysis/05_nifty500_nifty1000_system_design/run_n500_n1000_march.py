"""Live Strictly Causal Time-March on FULL Nifty 500 & FULL Nifty 1000 (2026-02-27 -> 2026-11-20).

Zero selection bias. Evaluates:
  - 11 strategies / 5 specialist expert voices (Momentum, GARP, Hybrid, Inst-Flow PEAD,
    Sector-Rotation Risk-Parity, and 5-Expert Consensus Ensemble + Pre/Post ablations)
  - 5 Control-Variable Sensitivity Sweeps (Liquidity floor, Position Cap / Inverse-Vol,
    Signal-Crowding Tie-Breaker, Industry Cap, Regime Gross Exposure Throttle)
on BOTH Nifty 500 (~490 stocks) and Nifty 1000 (~1,150 stocks).
"""
from __future__ import annotations

import json
import math
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

OUT_DIR = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\n500_n1000_march"
PKL_PATH = os.path.join(OUT_DIR, "panel_n1000.pkl")
COST_SIDE = 0.0020
INIT_CAP = 1_000_000.0


def load_panel() -> dict:
    with open(PKL_PATH, "rb") as f:
        return pickle.load(f)


def passive_ew(panel: dict, tickers: List[str]) -> dict:
    cols = [panel["ticker_to_col"][tk] for tk in tickers]
    C = panel["C"][:, cols]
    init_px = C[0, :]
    valid = np.isfinite(init_px) & (init_px > 0)
    C = C[:, valid]
    shares = (INIT_CAP / C.shape[1]) / C[0, :]
    nav = np.nansum(C * shares[None, :], axis=1)
    s = pd.Series(nav)
    r = s.pct_change().dropna()
    return {
        "final_nav": round(float(s.iloc[-1]), 0),
        "total_ret_pct": round((float(s.iloc[-1]) / INIT_CAP - 1.0) * 100.0, 2),
        "max_dd_pct": round(float((s / s.cummax() - 1.0).min() * 100.0), 2),
        "sharpe": round(float((r.mean() / r.std(ddof=1)) * math.sqrt(252.0)), 3) if r.std(ddof=1) > 1e-12 else 0.0,
        "n_stocks": int(C.shape[1]),
    }


def _gross_cap(emode: str, use_throttle: bool) -> float:
    if not use_throttle:
        return 1.00
    return {"BEAR": 0.40, "DEFENSIVE": 0.40, "CAUTION": 0.75, "RECOVERY": 0.90, "BULL": 1.00}.get(emode, 0.75)


def run_sim(
    panel: dict,
    tickers: List[str],
    strategy: str,
    *,
    decision_dates: set,
    liq_floor_cr: float = 5.0,
    max_pos_pct: float = 10.0,
    use_inv_vol: bool = False,
    max_per_industry: Optional[int] = None,
    use_regime_throttle: bool = False,
    tie_breaker: str = "comp_rs",  # comp_rs | q_np_yoy | ind_rank_pct | d_inst_q | abs_deliv_cr
    cooldown_days: int = 0,
) -> dict:
    dates = panel["dates"]
    ckpts = panel["ckpts"]
    regimes = panel["regimes"]
    t2c = panel["ticker_to_col"]
    O, H, L, C, MA50 = panel["O"], panel["H"], panel["L"], panel["C"], panel["MA50"]
    sub_ind = panel["sub_industry"]

    cash = INIT_CAP
    positions: Dict[str, dict] = {}
    pending_orders: Dict[str, dict] = {}
    cooldown_until: Dict[str, int] = {}  # tk -> day_idx until which re-entry is blocked
    trades: List[dict] = []
    tranches: List[dict] = []
    eq_rows: List[dict] = []
    crowding_counts: List[int] = []  # # of qualifying candidates per decision checkpoint

    def _nav_day(di: int, mat: np.ndarray) -> float:
        eq = cash
        for tk, p in positions.items():
            eq += p["shares"] * float(mat[di, t2c[tk]])
        return eq

    def _rank_val(s: dict, key: str) -> float:
        v = s.get(key)
        if v is None:
            return -999.0
        return float(v)

    def _slot_size(s: dict, default_pct: float) -> float:
        if not use_inv_vol:
            return min(max_pos_pct, default_pct)
        atr_p = float(s.get("atr_pct") or 3.5)
        # Target 0.30% daily portfolio vol contribution per stock => w = 0.30 / (atr_p/100) * 100 = 30 / atr_p
        w = 30.0 / max(atr_p, 1.5)
        return float(np.clip(w, 4.0, min(12.0, max_pos_pct * 1.2)))

    def _close_tranches(tk: str, dt: str, ex_px: float):
        for tr in tranches:
            if tr["ticker"] == tk and tr["closed_date"] is None:
                tr["closed_date"] = dt
                tr["exit_px"] = round(ex_px, 2)

    for di, dt in enumerate(dates):
        # 1. Execute queued next-open SELLS first, then BUYS
        for tk, o in sorted(pending_orders.items(), key=lambda kv: 0 if kv[1]["side"] == "SELL" else 1):
            col = t2c[tk]
            px_op = float(O[di, col])
            if not np.isfinite(px_op) or px_op <= 0:
                continue
            if o["side"] == "SELL" and tk in positions:
                p = positions.pop(tk)
                net = p["shares"] * px_op * (1.0 - COST_SIDE)
                cash += net
                _close_tranches(tk, dt, px_op)
                pnl_pct = round((px_op / p["avg_px"] - 1.0) * 100.0, 2)
                trades.append({"date": dt, "ticker": tk, "side": "SELL", "reason": o["reason"], "px": round(px_op, 2), "pnl_pct": pnl_pct, "pnl_rs": round(net - p["cost_basis"], 0)})
                if cooldown_days > 0 and pnl_pct < 0:
                    cooldown_until[tk] = di + cooldown_days
            elif o["side"] == "BUY":
                gap_pct = (px_op / o["ref_px"] - 1.0) * 100.0 if o.get("ref_px") else 0.0
                if o.get("max_gap_pct") is not None and gap_pct > o["max_gap_pct"]:
                    continue
                cur_eq = _nav_day(di, O)
                inv_now = cur_eq - cash
                g_cap = cur_eq * o.get("gross_cap", 1.0)
                room_gross = max(0.0, g_cap - inv_now)
                tgt_val = cur_eq * (o["target_pct"] / 100.0)
                cur_val = positions[tk]["shares"] * px_op if tk in positions else 0.0
                buy_val = min(max(0.0, tgt_val - cur_val), room_gross, cash / (1.0 + COST_SIDE))
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
                            "industry": sub_ind.get(tk, "Other"),
                        }
                    tranches.append({
                        "ticker": tk, "entry_date": dt, "entry_px": round(px_op, 2), "reason": o["reason"],
                        "min_low": round(px_op, 2), "mae_low_pct": 0.0, "closed_date": None, "exit_px": None,
                    })
                    trades.append({"date": dt, "ticker": tk, "side": "BUY", "reason": o["reason"], "px": round(px_op, 2), "value": round(sh * px_op, 0)})
        pending_orders.clear()

        # 2. Daily MAE update + Intraday hard/catastrophe stops + EOD MA50 / trailing stops
        for tr in tranches:
            if tr["closed_date"] is None and tr["ticker"] in positions:
                lo = float(L[di, t2c[tr["ticker"]]])
                if lo < tr["min_low"]:
                    tr["min_low"] = round(lo, 2)
                    tr["mae_low_pct"] = round((lo / tr["entry_px"] - 1.0) * 100.0, 2)

        for tk in list(positions.keys()):
            p = positions[tk]
            col = t2c[tk]
            lo, op, cl, m50 = float(L[di, col]), float(O[di, col]), float(C[di, col]), float(MA50[di, col])
            cat_stop = p["avg_px"] * (1.0 + p["catastrophe_pct"] / 100.0) if p.get("catastrophe_pct") is not None else None
            eff_stop = max([x for x in (p.get("stop_px"), cat_stop) if x is not None], default=None)
            if eff_stop is not None and lo <= eff_stop:
                fill_px = min(op, eff_stop)
                net = p["shares"] * fill_px * (1.0 - COST_SIDE)
                cash += net
                _close_tranches(tk, dt, fill_px)
                pnl_pct = round((fill_px / p["avg_px"] - 1.0) * 100.0, 2)
                rsn = "CATASTROPHE_-25%" if (cat_stop is not None and eff_stop == cat_stop) else "HARD_STOP"
                trades.append({"date": dt, "ticker": tk, "side": "SELL", "reason": rsn, "px": round(fill_px, 2), "pnl_pct": pnl_pct, "pnl_rs": round(net - p["cost_basis"], 0)})
                del positions[tk]
                if cooldown_days > 0:
                    cooldown_until[tk] = di + cooldown_days
                continue

            if cl > p["peak_close"]:
                p["peak_close"] = cl
            if p.get("use_ma50") and np.isfinite(m50) and cl < m50:
                pending_orders[tk] = {"side": "SELL", "reason": "MA50_BREAK"}
            elif p.get("trail_pct") is not None and cl < p["peak_close"] * (1.0 - p["trail_pct"]):
                # For GARP profit-trail legs, activate 20% trail once position has made any profit OR always if not profit_only
                pending_orders[tk] = {"side": "SELL", "reason": f"TRAIL_{int(p['trail_pct']*100)}%"}

        # 3. Checkpoint EOD Decisions (strictly causal <= dt; execute next open)
        emode = regimes[dt]["engine_mode"]
        if dt in decision_dates and di < len(dates) - 1 and dt in ckpts:
            snap = ckpts[dt]
            trail = 0.15 if emode in ("BULL", "RECOVERY") else 0.12
            g_cap = _gross_cap(emode, use_regime_throttle)
            ind_counts: Dict[str, int] = {}
            for tk_h, p_h in positions.items():
                if tk_h not in pending_orders:
                    ind_counts[p_h["industry"]] = ind_counts.get(p_h["industry"], 0) + 1

            # Fundamental exits for GARP / Hybrid / Consensus legs
            if strategy in ("garp_cfo_only", "garp_guarded", "hybrid_old", "hybrid_guarded", "council_5expert_consensus"):
                for tk_h in list(positions.keys()):
                    s = snap.get(tk_h)
                    if not s:
                        continue
                    if s["cfo_pat"] is not None and s["cfo_pat"] < 0.0 and positions[tk_h].get("catastrophe_pct") is not None:
                        pending_orders[tk_h] = {"side": "SELL", "reason": "FUND_CFO_NEG"}
                    elif strategy in ("garp_guarded", "hybrid_guarded") and s["q_np_yoy"] is not None and s["q_np_yoy"] <= -10.0 and positions[tk_h].get("catastrophe_pct") is not None:
                        pending_orders[tk_h] = {"side": "SELL", "reason": "FUND_NP_SHRINK"}

            # Candidate generation strictly <= dt
            cands = []
            for tk in tickers:
                if tk in positions or tk in pending_orders:
                    continue
                if cooldown_until.get(tk, -1) > di:
                    continue
                s = snap.get(tk)
                if not s or s["liq_cr"] < liq_floor_cr:
                    continue

                if strategy in ("mom_pre_dual_atr", "mom_post_dual_atr", "monthly_quant_post"):
                    act = s["action_pre"] if strategy == "mom_pre_dual_atr" else s["action_post"]
                    if act == "BUY NOW" and s["stage"] == 2 and s["comp_rs"] > 0:
                        cands.append((_rank_val(s, tie_breaker), tk, "MOM", s))

                elif strategy == "garp_unfiltered":
                    if s["cmp_lt_bear"] and s["bear_blend"] is not None:
                        disc = (1.0 - s["close"] / s["bear_blend"]) * 100.0
                        cands.append((disc, tk, "GARP", s))

                elif strategy == "garp_cfo_only":
                    if s["cmp_lt_bear"] and s["bear_blend"] is not None and s["cfo_pat"] is not None and s["cfo_pat"] >= 0.80:
                        disc = (1.0 - s["close"] / s["bear_blend"]) * 100.0
                        cands.append((disc, tk, "GARP", s))

                elif strategy == "garp_guarded":
                    # All 3 Poisoned-29 Guardrails: CFO/PAT>=0.80 + Q_NP_YoY>0% + NO Stage-4 knife!
                    if (
                        s["cmp_lt_bear"] and s["bear_blend"] is not None
                        and s["cfo_pat"] is not None and s["cfo_pat"] >= 0.80
                        and s["q_np_yoy"] is not None and s["q_np_yoy"] > 0.0
                        and s["stage"] != 4
                    ):
                        score = (1.0 - s["close"] / s["bear_blend"]) * 100.0 + 0.5 * (s["roce"] or 0.0)
                        cands.append((score, tk, "GARP_GUARDED", s))

                elif strategy == "hybrid_old":
                    if s["cmp_lt_bear"] and s["cfo_pat"] is not None and s["cfo_pat"] >= 0.80:
                        cands.append((80.0 + (s["roce"] or 0.0), tk, "HYB_GARP_OLD", s))
                    elif s["action_post"] == "BUY NOW" and s["stage"] == 2 and s["comp_rs"] > 0:
                        cands.append((50.0 + s["comp_rs"], tk, "MOM", s))
                    elif s["bbp"] is not None and s["bbp"] < 20.0 and ((s["stage"] in (1, 2) and s["comp_rs"] > 0) or (emode in ("BEAR", "RECOVERY") and (s["q_np_yoy"] or 0) >= 30.0)):
                        cands.append((30.0 + s["comp_rs"], tk, "BBP", s))

                elif strategy == "hybrid_guarded":
                    if (
                        s["cmp_lt_bear"] and s["cfo_pat"] is not None and s["cfo_pat"] >= 0.80
                        and s["q_np_yoy"] is not None and s["q_np_yoy"] > 0.0 and s["stage"] != 4
                    ):
                        cands.append((80.0 + (s["roce"] or 0.0), tk, "GARP_GUARDED", s))
                    elif s["action_post"] == "BUY NOW" and s["stage"] == 2 and s["comp_rs"] > 0:
                        cands.append((50.0 + s["comp_rs"], tk, "MOM", s))
                    elif s["bbp"] is not None and s["bbp"] < 20.0 and s["stage"] == 2 and s["comp_rs"] > 5.0 and emode in ("RECOVERY", "CAUTION", "BULL"):
                        cands.append((40.0 + s["comp_rs"], tk, "BBP", s))

                elif strategy == "inst_flow_pead":
                    # NEW EXPERT #4: Institutional Smart-Money + Post-Earnings Drift
                    if (
                        s["d_inst_q"] is not None and s["d_inst_q"] >= 1.0
                        and (s["d_prom_q"] is None or s["d_prom_q"] >= -1.0)
                        and s["q_np_yoy"] is not None and s["q_np_yoy"] >= 25.0
                        and (s["q_sales_yoy"] is None or s["q_sales_yoy"] >= 15.0)
                        and (s["opm_q"] is None or s["opm_prev"] is None or s["opm_q"] >= s["opm_prev"] - 1.5)
                        and s["abs_deliv_cr"] >= 5.0
                        and s["stage"] in (1, 2, 3) and s["comp_rs"] > 0
                    ):
                        score = s["d_inst_q"] * 10.0 + min(s["q_np_yoy"], 150.0) * 0.25 + s["comp_rs"] * 0.5
                        cands.append((score, tk, "PEAD", s))

                elif strategy == "sector_rotator_rp":
                    # NEW EXPERT #5: Top-30% Leading Industry + Stage-2 Dual-ATR + Q_NP_YoY>0 + Inv-Vol + Industry Cap
                    if (
                        s["ind_rank_pct"] >= 70.0
                        and s["stage"] == 2 and s["comp_rs"] > 0
                        and (s["q_np_yoy"] is None or s["q_np_yoy"] > 0.0)
                        and (s["action_post"] == "BUY NOW" or (s["bbp"] is not None and s["bbp"] < 20.0 and s["comp_rs"] > 5.0))
                    ):
                        score = 0.5 * s["ind_rank_pct"] + s["comp_rs"]
                        cands.append((score, tk, "SECTOR_RP", s))

                elif strategy == "council_5expert_consensus":
                    # 5-EXPERT CONSENSUS ENSEMBLE: Count orthogonal expert endorsements (0..5)
                    v_mom = s["action_post"] == "BUY NOW" and s["stage"] == 2 and s["comp_rs"] > 0
                    v_garp = (
                        s["cmp_lt_bear"] and s["cfo_pat"] is not None and s["cfo_pat"] >= 0.80
                        and s["q_np_yoy"] is not None and s["q_np_yoy"] > 0.0 and s["stage"] != 4
                    )
                    v_bbp = s["bbp"] is not None and s["bbp"] < 20.0 and s["stage"] == 2 and s["comp_rs"] > 5.0
                    v_pead = (
                        s["d_inst_q"] is not None and s["d_inst_q"] >= 1.0
                        and (s["d_prom_q"] is None or s["d_prom_q"] >= -1.0)
                        and s["q_np_yoy"] is not None and s["q_np_yoy"] >= 20.0
                        and s["stage"] != 4 and s["comp_rs"] > 0
                    )
                    v_sec = s["ind_rank_pct"] >= 70.0 and s["stage"] == 2 and s["comp_rs"] > 0
                    votes = int(v_mom) + int(v_garp) + int(v_bbp) + int(v_pead) + int(v_sec)
                    if votes >= 2 and (s["q_np_yoy"] is None or s["q_np_yoy"] > 0.0):
                        score = votes * 100.0 + _rank_val(s, tie_breaker) + 0.3 * s["ind_rank_pct"]
                        kind = "GARP_GUARDED" if (v_garp and not v_mom) else "CONSENSUS"
                        cands.append((score, tk, kind, s))

            crowding_counts.append(len(cands))
            cands.sort(key=lambda x: x[0], reverse=True)
            for _, tk, kind, s in cands:
                ind_k = s["industry"]
                if max_per_industry is not None and ind_counts.get(ind_k, 0) >= max_per_industry:
                    continue
                sz = _slot_size(s, max_pos_pct)
                if kind in ("GARP", "HYB_GARP_OLD"):
                    pending_orders[tk] = {
                        "side": "BUY", "reason": kind, "target_pct": sz, "stop_px": None,
                        "trail_pct": (0.20 if kind == "HYB_GARP_OLD" else None), "use_ma50": False,
                        "catastrophe_pct": (-25.0 if strategy != "garp_unfiltered" else None),
                        "ref_px": s["close"], "max_gap_pct": None, "gross_cap": g_cap,
                    }
                elif kind == "GARP_GUARDED":
                    pending_orders[tk] = {
                        "side": "BUY", "reason": kind, "target_pct": min(10.0, sz), "stop_px": None,
                        "trail_pct": 0.20, "use_ma50": False, "catastrophe_pct": -25.0,
                        "ref_px": s["close"], "max_gap_pct": None, "gross_cap": g_cap,
                    }
                else:
                    stp = s["stop_fx"] if (s["stop_fx"] and s["stop_fx"] < s["close"] and (1 - s["stop_fx"] / s["close"]) <= 0.15) else round(s["close"] - 2.0 * s["atr"], 2)
                    pending_orders[tk] = {
                        "side": "BUY", "reason": kind, "target_pct": sz, "stop_px": stp,
                        "trail_pct": trail, "use_ma50": True, "catastrophe_pct": None,
                        "ref_px": s["close"], "max_gap_pct": 5.0, "gross_cap": g_cap,
                    }
                ind_counts[ind_k] = ind_counts.get(ind_k, 0) + 1

        nav_dt = _nav_day(di, C)
        eq_rows.append({"date": dt, "nav": nav_dt, "cash_pct": cash / nav_dt * 100.0 if nav_dt > 0 else 100.0, "mode": emode, "n_pos": len(positions)})

    edf = pd.DataFrame(eq_rows).set_index("date")
    nav_s = edf["nav"]
    rets = nav_s.pct_change().dropna()
    sd = float(rets.std(ddof=1)) if len(rets) > 2 else 0.0
    sells = [t for t in trades if t["side"] == "SELL"]
    wins = sum(1 for t in sells if t["pnl_rs"] > 0)
    buys = [t for t in trades if t["side"] == "BUY"]
    return {
        "final_nav": round(float(nav_s.iloc[-1]), 0),
        "total_ret_pct": round((float(nav_s.iloc[-1]) / INIT_CAP - 1.0) * 100.0, 2),
        "max_dd_pct": round(float((nav_s / nav_s.cummax() - 1.0).min() * 100.0), 2),
        "sharpe": round(float((rets.mean() / sd) * math.sqrt(252.0)), 3) if sd > 1e-12 else 0.0,
        "cash_drag_mean": round(float(edf["cash_pct"].mean()), 1),
        "avg_positions": round(float(edf["n_pos"].mean()), 1),
        "buys": len(buys),
        "sells": len(sells),
        "win_rate_closed_pct": round(wins / len(sells) * 100.0, 1) if sells else 0.0,
        "avg_cands_per_event": round(float(np.mean(crowding_counts)), 1) if crowding_counts else 0.0,
        "max_cands_event": int(max(crowding_counts)) if crowding_counts else 0,
        "worst_tranche_mae_pct": round(min((tr["mae_low_pct"] for tr in tranches), default=0.0), 1),
    }


def main():
    panel = load_panel()
    ev_set = set(panel["event_dates"])
    mo_set = set(panel["monthly_gates"])
    u_map = {"NIFTY_500": panel["nifty500"], "NIFTY_1000": panel["nifty1000"]}

    all_results = {"benchmarks": {}, "experts": {}, "control_sweeps": {}}
    for u_name, tks in u_map.items():
        all_results["benchmarks"][u_name] = passive_ew(panel, tks)
        print(f"\n==================== {u_name} ({len(tks)} stocks) ====================")
        b = all_results["benchmarks"][u_name]
        print(f"  BENCH EW_{u_name:10s} | Ret: {b['total_ret_pct']:+7.2f}% | MaxDD: {b['max_dd_pct']:6.2f}% | Sharpe: {b['sharpe']:6.3f}")
        print("-" * 118)

        specs = [
            ("mom_pre_dual_atr", dict(decision_dates=ev_set, liq_floor_cr=5.0, max_pos_pct=10.0)),
            ("mom_post_dual_atr", dict(decision_dates=ev_set, liq_floor_cr=5.0, max_pos_pct=10.0)),
            ("garp_unfiltered", dict(decision_dates=ev_set, liq_floor_cr=5.0, max_pos_pct=10.0)),
            ("garp_cfo_only", dict(decision_dates=ev_set, liq_floor_cr=5.0, max_pos_pct=10.0)),
            ("garp_guarded", dict(decision_dates=ev_set, liq_floor_cr=5.0, max_pos_pct=10.0, cooldown_days=60)),
            ("hybrid_old", dict(decision_dates=ev_set, liq_floor_cr=5.0, max_pos_pct=20.0)),
            ("hybrid_guarded", dict(decision_dates=ev_set, liq_floor_cr=5.0, max_pos_pct=10.0, cooldown_days=60)),
            ("inst_flow_pead", dict(decision_dates=ev_set, liq_floor_cr=5.0, max_pos_pct=10.0, cooldown_days=30)),
            ("sector_rotator_rp", dict(decision_dates=ev_set, liq_floor_cr=5.0, max_pos_pct=10.0, use_inv_vol=True, max_per_industry=2, use_regime_throttle=True, cooldown_days=30)),
            ("council_5expert_consensus", dict(decision_dates=ev_set, liq_floor_cr=5.0, max_pos_pct=10.0, use_inv_vol=True, max_per_industry=2, use_regime_throttle=True, cooldown_days=60)),
            ("monthly_quant_post", dict(decision_dates=mo_set, liq_floor_cr=5.0, max_pos_pct=10.0)),
        ]
        all_results["experts"][u_name] = {}
        for s_name, kw in specs:
            m = run_sim(panel, tks, s_name, **kw)
            all_results["experts"][u_name][s_name] = m
            print(
                f"  {s_name:26s} | Ret: {m['total_ret_pct']:+7.2f}% | MaxDD: {m['max_dd_pct']:6.2f}% | "
                f"Sharpe: {m['sharpe']:6.3f} | Cash: {m['cash_drag_mean']:5.1f}% | Pos: {m['avg_positions']:4.1f} | "
                f"Win%: {m['win_rate_closed_pct']:5.1f}% ({m['buys']:3d}B) | Cands/Ev: {m['avg_cands_per_event']:4.1f} (max {m['max_cands_event']:3d})"
            )

        # Control-Variable Sensitivity Sweeps (on both Nifty 500 & Nifty 1000!)
        sweeps = {}
        # 1. Liquidity Floor Sweep (mom_post_dual_atr & consensus)
        for lf in (2.0, 5.0, 20.0, 50.0):
            sweeps[f"liq_{int(lf)}cr_mom"] = run_sim(panel, tks, "mom_post_dual_atr", decision_dates=ev_set, liq_floor_cr=lf, max_pos_pct=10.0)
            sweeps[f"liq_{int(lf)}cr_cons"] = run_sim(panel, tks, "council_5expert_consensus", decision_dates=ev_set, liq_floor_cr=lf, max_pos_pct=10.0, use_inv_vol=True, max_per_industry=2, use_regime_throttle=True, cooldown_days=60)
        # 2. Sizing / Concentration Sweep
        for cap_lbl, cap_v, iv in [("cap20_5slots", 20.0, False), ("cap10_10slots", 10.0, False), ("cap6.7_15slots", 6.7, False), ("inv_vol_atr", 10.0, True)]:
            sweeps[f"size_{cap_lbl}"] = run_sim(panel, tks, "council_5expert_consensus", decision_dates=ev_set, liq_floor_cr=5.0, max_pos_pct=cap_v, use_inv_vol=iv, max_per_industry=2, use_regime_throttle=True, cooldown_days=60)
        # 3. Signal-Crowding Tie-Breaker Sweep (critical on 500-1000 stocks!)
        for tb in ("comp_rs", "q_np_yoy", "ind_rank_pct", "d_inst_q", "abs_deliv_cr"):
            sweeps[f"rank_{tb}"] = run_sim(panel, tks, "mom_post_dual_atr", decision_dates=ev_set, liq_floor_cr=5.0, max_pos_pct=10.0, tie_breaker=tb)
        # 4. Industry Cap Sweep
        sweeps["ind_uncapped"] = run_sim(panel, tks, "council_5expert_consensus", decision_dates=ev_set, liq_floor_cr=5.0, max_pos_pct=10.0, use_inv_vol=True, max_per_industry=None, use_regime_throttle=True, cooldown_days=60)
        sweeps["ind_cap2"] = run_sim(panel, tks, "council_5expert_consensus", decision_dates=ev_set, liq_floor_cr=5.0, max_pos_pct=10.0, use_inv_vol=True, max_per_industry=2, use_regime_throttle=True, cooldown_days=60)
        # 5. Regime Gross Exposure Throttle Sweep
        sweeps["throttle_off"] = run_sim(panel, tks, "council_5expert_consensus", decision_dates=ev_set, liq_floor_cr=5.0, max_pos_pct=10.0, use_inv_vol=True, max_per_industry=2, use_regime_throttle=False, cooldown_days=60)
        sweeps["throttle_on"] = run_sim(panel, tks, "council_5expert_consensus", decision_dates=ev_set, liq_floor_cr=5.0, max_pos_pct=10.0, use_inv_vol=True, max_per_industry=2, use_regime_throttle=True, cooldown_days=60)
        all_results["control_sweeps"][u_name] = sweeps

        print(f"\n  --- {u_name} CONTROL-VARIABLE SWEEPS ---")
        for sw_k, m in sweeps.items():
            print(f"    {sw_k:22s} | Ret: {m['total_ret_pct']:+7.2f}% | MaxDD: {m['max_dd_pct']:6.2f}% | Sharpe: {m['sharpe']:6.3f} | Cash: {m['cash_drag_mean']:5.1f}%")

    out_json = os.path.join(OUT_DIR, "n500_n1000_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full results to {out_json}")


if __name__ == "__main__":
    main()
