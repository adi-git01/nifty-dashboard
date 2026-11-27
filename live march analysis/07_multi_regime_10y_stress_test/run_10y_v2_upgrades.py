"""Follow-up 10.6-Year All-Weather System Upgrades Test (2016-04-01 -> 2026-11-20).

Tests the exact structural fixes revealed by the 10.6Y multi-regime stress test:
  1. Why did GARP bleed -41% in 2018-19 Bear & -61.5% MaxDD?
     Test: (a) GARP + Regime Throttle, (b) GARP + Stage-2/Above-MA200 Confirmation,
           (c) GARP + Quality (ROCE>=12%, CFO>=0.70, NP_YoY>0%) + Wide Stop (25%) + Regime Throttle,
           (d) GARP Ranked by Quality+Momentum (not deepest falling knife bear_disc!)
  2. Dynamic Regime-Adaptive Liquidity Floor on Quality Momentum:
     >= ₹2-5 Cr/d in BULL/RECOVERY, automatically tightening to >= ₹20 Cr/d in CAUTION/BEAR!
  3. All-Weather 3-Sleeve Orthogonal Barbell combining the winning 10Y-validated sleeves
     across ALL 7 Indian market regimes (2016-17 Bull, 2018-19 Bear, 2020 COVID,
     2020-21 Post-COVID Bull, 2022-23 Chop, 2023-24 SuperBull, 2024-26 Post-24 Chop).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

OUT_DIR = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\multi_regime_10y"
sys.path.insert(0, OUT_DIR)
import run_10y_regimes as R  # noqa: E402

INIT_CAP = R.INIT_CAP
COST_SIDE = R.COST_SIDE


def run_v2_sim(
    panel: dict,
    strategy: str,
    *,
    liq_bull: float = 2.0,
    liq_bear: float = 20.0,
    max_pos_pct: float = 10.0,
    use_inv_vol: bool = False,
    max_per_ind: Optional[int] = 2,
    throttle_mode: str = "on",
    cooldown_days: int = 30,
) -> dict:
    dates = list(panel["sim_dates"])
    dec_dates = panel["ckpt_dates"]
    ckpts, regimes, t2c = panel["ckpts"], panel["regimes"], panel["t2c"]
    O, L, C, MA50 = panel["O"], panel["L"], panel["C"], panel["MA50"]

    cash = INIT_CAP
    positions: Dict[str, dict] = {}
    pending: Dict[str, dict] = {}
    cooldown: Dict[str, int] = {}
    nav_arr = np.empty(len(dates), dtype=np.float64)
    cash_arr = np.empty(len(dates), dtype=np.float64)
    n_buys = 0
    sells_pnl: List[float] = []

    for di, dt in enumerate(dates):
        if pending:
            for tk, o in sorted(pending.items(), key=lambda kv: 0 if kv[1]["side"] == "SELL" else 1):
                col = t2c[tk]
                px = float(O[di, col])
                if not np.isfinite(px) or px <= 0:
                    px = float(C[di, col])
                if not np.isfinite(px) or px <= 0:
                    continue
                if o["side"] == "SELL" and tk in positions:
                    p = positions.pop(tk)
                    net = p["shares"] * px * (1.0 - COST_SIDE)
                    cash += net
                    pnl_p = (px / p["avg_px"] - 1.0) * 100.0
                    sells_pnl.append(pnl_p)
                    if cooldown_days > 0 and pnl_p < 0:
                        cooldown[tk] = di + cooldown_days
                elif o["side"] == "BUY" and tk not in positions:
                    if o.get("max_gap") and (px / o["ref"] - 1.0) * 100.0 > o["max_gap"]:
                        continue
                    cur_eq = cash + sum(p["shares"] * (float(O[di, t2c[t]]) if np.isfinite(O[di, t2c[t]]) else p["last_px"]) for t, p in positions.items())
                    room = max(0.0, cur_eq * o.get("gcap", 1.0) - (cur_eq - cash))
                    bv = min(cur_eq * (o["sz"] / 100.0), room, cash / (1.0 + COST_SIDE))
                    sh = int(bv / px)
                    if sh > 0:
                        cost = sh * px * (1.0 + COST_SIDE)
                        cash -= cost
                        positions[tk] = dict(
                            shares=sh, cost_basis=cost, avg_px=cost / sh, peak=px, last_px=px,
                            stop=o.get("stop"), trail=o.get("trail"), ma50=o.get("ma50", False),
                            ind=o.get("ind", "Other"),
                        )
                        n_buys += 1
            pending.clear()

        eq = cash
        for tk in list(positions.keys()):
            p = positions[tk]
            col = t2c[tk]
            c_px = float(C[di, col])
            l_px = float(L[di, col])
            if not np.isfinite(c_px) or c_px <= 0:
                p["miss"] = p.get("miss", 0) + 1
                if p["miss"] >= 15:
                    net = p["shares"] * p["last_px"] * 0.85
                    cash += net
                    sells_pnl.append(-15.0)
                    positions.pop(tk)
                    continue
                c_px = p["last_px"]
                l_px = c_px
            else:
                p["last_px"] = c_px
                p["miss"] = 0
            eq += p["shares"] * c_px
            if c_px > p["peak"]:
                p["peak"] = c_px

            trig = False
            if p["stop"] is not None and (l_px / p["avg_px"] - 1.0) * 100.0 <= -p["stop"]:
                trig = True
            elif p["trail"] is not None and (c_px / p["peak"] - 1.0) * 100.0 <= -p["trail"]:
                trig = True
            elif p["ma50"]:
                m50_v = float(MA50[di, col])
                if np.isfinite(m50_v) and c_px < m50_v:
                    trig = True
            if trig:
                pending[tk] = dict(side="SELL", reason="STOP")

        nav_arr[di] = eq
        cash_arr[di] = (cash / eq * 100.0) if eq > 0 else 100.0

        if dt not in dec_dates or dt not in ckpts:
            continue
        snap = ckpts[dt]
        if not snap:
            continue
        reg = regimes[dt]
        mode = reg["mode"]
        gcap = R._gross_cap(reg, throttle_mode)
        # Dynamic Regime-Adaptive Liquidity Floor!
        dyn_liq = liq_bull if mode in ("BULL", "RECOVERY") else liq_bear

        for tk in list(positions.keys()):
            if tk in pending:
                continue
            s = snap.get(tk)
            if s is None:
                continue
            if strategy.startswith("garp") and s["fy_np_yoy"] is not None and s["fy_np_yoy"] < -15.0:
                pending[tk] = dict(side="SELL", reason="EARN_BREAK")

        cands = []
        for tk, s in snap.items():
            if tk in positions or tk in pending:
                continue
            if di < cooldown.get(tk, -1):
                continue
            if s["liq"] < dyn_liq:
                continue
            if np.isfinite(s["mcap"]) and s["mcap"] < 500.0:
                continue

            ok = False
            score = 0.0
            stop_p, trail_p, use_ma50, mgap = 12.0, 15.0, False, 4.0

            if strategy == "momq_dynamic_liq":
                if s["buy_now"] and (s["fy_np_yoy"] is None or s["fy_np_yoy"] > 0.0):
                    ok = True
                    np_sc = np.clip((s["fy_np_yoy"] if s["fy_np_yoy"] is not None else 15.0), 0.0, 100.0)
                    dcr_sc = min(100.0, math.log1p(max(0.0, s["dcr"])) * 22.0)
                    score = 0.40 * s["irank"] + 0.35 * dcr_sc + 0.25 * np_sc
                    stop_p, trail_p, use_ma50 = 10.0, 14.0, True

            elif strategy == "garp_throttled":
                # Same guarded GARP, but with Macro Regime Throttle + 25% wide stop (no -18% whipsaw!)
                cfo_ok = s["is_bfsi"] or (s["cfo_pat"] is not None and s["cfo_pat"] >= 0.70)
                earn_ok = (s["fy_np_yoy"] is not None and s["fy_np_yoy"] > 0.0)
                if s["cmp_lt_bear"] and cfo_ok and earn_ok and s["stage"] != 4:
                    ok = True
                    score = (s["bear_disc"] or 0.0) + 0.25 * min(s["fy_np_yoy"], 80.0)
                    stop_p, trail_p, use_ma50, mgap = 25.0, 25.0, False, 5.0

            elif strategy == "garp_stage2_trend":
                # Value-at-Inflection: CMP < Bear + CFO>=0.7 + NP_YoY>0 + Stage 2 (Close > MA50 & > MA200) + Ranked by RS+Quality!
                cfo_ok = s["is_bfsi"] or (s["cfo_pat"] is not None and s["cfo_pat"] >= 0.70)
                earn_ok = (s["fy_np_yoy"] is not None and s["fy_np_yoy"] > 0.0)
                if s["cmp_lt_bear"] and cfo_ok and earn_ok and s["stage"] == 2:
                    ok = True
                    score = 0.40 * s["crs"] + 0.30 * s["irank"] + 0.30 * min(s["fy_np_yoy"], 100.0)
                    stop_p, trail_p, use_ma50, mgap = 15.0, 20.0, False, 5.0

            elif strategy == "garp_stage12_quality":
                # Quality GARP: CMP < Bear + CFO>=0.7 + NP_YoY>0 + Stage 1/2 (NOT Stage 3/4 rolling over!) + IndRank+RS ranking
                cfo_ok = s["is_bfsi"] or (s["cfo_pat"] is not None and s["cfo_pat"] >= 0.70)
                earn_ok = (s["fy_np_yoy"] is not None and s["fy_np_yoy"] > 0.0)
                if s["cmp_lt_bear"] and cfo_ok and earn_ok and s["stage"] in (1, 2):
                    ok = True
                    score = 0.35 * s["crs"] + 0.35 * s["irank"] + 0.30 * min(s["fy_np_yoy"], 100.0)
                    stop_p, trail_p, use_ma50, mgap = 15.0, 20.0, False, 5.0

            elif strategy == "pead_dynamic_liq":
                if (s["stage"] == 2 and s["fy_np_yoy"] is not None and s["fy_np_yoy"] >= 20.0
                        and s["fy_sales_yoy"] is not None and s["fy_sales_yoy"] >= 12.0
                        and s["drat"] >= 1.10 and s["crs"] >= 60.0):
                    ok = True
                    score = min(s["fy_np_yoy"], 120.0) * 0.45 + s["drat"] * 20.0 + s["irank"] * 0.35
                    stop_p, trail_p, use_ma50 = 10.0, 15.0, True

            if ok:
                cands.append((score, tk, s, stop_p, trail_p, use_ma50, mgap))

        if not cands:
            continue
        cands.sort(key=lambda x: x[0], reverse=True)
        ind_counts: Dict[str, int] = {}
        for p in positions.values():
            ind_counts[p["ind"]] = ind_counts.get(p["ind"], 0) + 1
        max_slots = max(5, int(round(100.0 / max_pos_pct)) + 2)
        slots_left = max(0, max_slots - len(positions))
        for score, tk, s, stop_p, trail_p, use_ma50, mgap in cands:
            if slots_left <= 0:
                break
            if max_per_ind is not None and ind_counts.get(s["ind"], 0) >= max_per_ind:
                continue
            sz = float(np.clip(22.0 / max(s["atrp"], 1.5), 4.0, max_pos_pct)) if use_inv_vol else max_pos_pct
            pending[tk] = dict(side="BUY", ref=s["px"], sz=sz, gcap=gcap, stop=stop_p, trail=trail_p, ma50=use_ma50, max_gap=mgap, ind=s["ind"])
            ind_counts[s["ind"]] = ind_counts.get(s["ind"], 0) + 1
            slots_left -= 1

    return dict(
        curve=pd.Series(nav_arr, index=panel["sim_dates"]),
        cash_mean=round(float(np.mean(cash_arr)), 1),
        buys=n_buys,
        win_rate=round(sum(1 for x in sells_pnl if x > 0) / len(sells_pnl) * 100.0, 1) if sells_pnl else 0.0,
    )


def main() -> int:
    panel = R.load_universe_and_matrices()
    n50_c = panel["n50"] / float(panel["n50"].iloc[0]) * INIT_CAP
    ew_c = R.passive_ew_rebalanced(panel)

    runs = [
        ("momq_dyn_2_20_th",       "momq_dynamic_liq",    dict(liq_bull=2.0, liq_bear=20.0, use_inv_vol=True,  max_per_ind=2, throttle_mode="on",           cooldown_days=30)),
        ("momq_dyn_5_20_th",       "momq_dynamic_liq",    dict(liq_bull=5.0, liq_bear=20.0, use_inv_vol=True,  max_per_ind=2, throttle_mode="on",           cooldown_days=30)),
        ("momq_dyn_2_20_dualb",    "momq_dynamic_liq",    dict(liq_bull=2.0, liq_bear=20.0, use_inv_vol=True,  max_per_ind=2, throttle_mode="dual_breadth", cooldown_days=30)),
        ("garp_throttled_2cr",     "garp_throttled",      dict(liq_bull=2.0, liq_bear=5.0,  use_inv_vol=False, max_per_ind=2, throttle_mode="on",           cooldown_days=60)),
        ("garp_stage2_trend_th",   "garp_stage2_trend",   dict(liq_bull=2.0, liq_bear=5.0,  use_inv_vol=True,  max_per_ind=2, throttle_mode="on",           cooldown_days=45)),
        ("garp_stage2_trend_noth", "garp_stage2_trend",   dict(liq_bull=2.0, liq_bear=2.0,  use_inv_vol=True,  max_per_ind=2, throttle_mode="off",          cooldown_days=45)),
        ("garp_stage12_qual_th",   "garp_stage12_quality",dict(liq_bull=2.0, liq_bear=5.0,  use_inv_vol=True,  max_per_ind=2, throttle_mode="on",           cooldown_days=45)),
        ("pead_dyn_5_20_th",       "pead_dynamic_liq",    dict(liq_bull=5.0, liq_bear=20.0, use_inv_vol=True,  max_per_ind=2, throttle_mode="on",           cooldown_days=30)),
    ]

    curves = {"NIFTY_50": n50_c, "EW_TRADABLE_500CR": ew_c}
    meta = {}
    for lbl, st, kw in runs:
        res = run_v2_sim(panel, st, **kw)
        curves[lbl] = res.pop("curve")
        meta[lbl] = res

    def make_barbell(weights: Dict[str, float]) -> pd.Series:
        rets = pd.DataFrame({k: curves[k].pct_change().fillna(0.0) for k in weights})
        w_cur = dict(weights)
        val = INIT_CAP
        out = np.empty(len(rets))
        for i, dt in enumerate(rets.index):
            if i == 0:
                out[i] = val
                continue
            r_row = rets.iloc[i]
            port_r = sum(w_cur[k] * float(r_row[k]) for k in weights)
            val *= (1.0 + port_r)
            out[i] = val
            if dt in panel["monthly_dates"]:
                w_cur = dict(weights)
            else:
                denom = 1.0 + port_r
                if denom > 0:
                    w_cur = {k: w_cur[k] * (1.0 + float(r_row[k])) / denom for k in weights}
        return pd.Series(out, index=rets.index)

    curves["ALLWEATHER_40_40_20"] = make_barbell({"garp_stage2_trend_noth": 0.40, "momq_dyn_2_20_th": 0.40, "pead_dyn_5_20_th": 0.20})
    curves["ALLWEATHER_40_40_20_TH"] = make_barbell({"garp_stage2_trend_th": 0.40, "momq_dyn_2_20_th": 0.40, "pead_dyn_5_20_th": 0.20})
    curves["ALLWEATHER_50_35_15"] = make_barbell({"garp_stage2_trend_noth": 0.50, "momq_dyn_2_20_dualb": 0.35, "pead_dyn_5_20_th": 0.15})

    out = {"full_10y": {}, "regimes": {}, "corr": {}, "meta": meta}
    print("\n" + "=" * 120)
    print("ALL-WEATHER 10.6Y UPGRADES (2016-04-01 -> 2026-11-20)")
    print("=" * 120)
    for k, c in curves.items():
        m = R.perf_metrics(c)
        out["full_10y"][k] = m
        ex = meta.get(k, {})
        cs = f"Cash:{ex.get('cash_mean',0):4.1f}% Win:{ex.get('win_rate',0):4.1f}% ({ex.get('buys',0):4d}B)" if ex else ""
        print(f"  {k:25s} | TotRet: {m['ret']:+8.1f}% | CAGR: {m['cagr']:+6.2f}% | MaxDD: {m['mdd']:7.2f}% | Sharpe: {m['sharpe']:5.3f} | Calmar: {m['calmar']:4.2f} | {cs}")

    cdf = pd.DataFrame({
        "GARP_S2": curves["garp_stage2_trend_noth"].pct_change(),
        "MOMQ_Dyn": curves["momq_dyn_2_20_th"].pct_change(),
        "PEAD_Dyn": curves["pead_dyn_5_20_th"].pct_change(),
        "NIFTY_50": curves["NIFTY_50"].pct_change(),
    }).dropna()
    out["corr"] = cdf.corr().round(2).to_dict()
    print("\nAll-Weather 10.6Y Correlation Matrix:")
    print(cdf.corr().round(2).to_string())

    for r_name, r_s, r_e in R.REGIMES:
        out["regimes"][r_name] = {}
        print(f"\n--- {r_name} ({r_s} -> {r_e}) ---")
        for k in ("NIFTY_50", "EW_TRADABLE_500CR", "momq_dyn_2_20_th", "garp_throttled_2cr", "garp_stage2_trend_noth", "garp_stage2_trend_th", "pead_dyn_5_20_th", "ALLWEATHER_40_40_20", "ALLWEATHER_40_40_20_TH"):
            m = R.perf_metrics(curves[k].loc[r_s:r_e])
            out["regimes"][r_name][k] = m
            print(f"  {k:25s} | Ret: {m['ret']:+7.2f}% | CAGR: {m['cagr']:+7.2f}% | MaxDD: {m['mdd']:7.2f}% | Sharpe: {m['sharpe']:6.3f}")

    with open(os.path.join(OUT_DIR, "allweather_10y_upgrades.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nSaved allweather_10y_upgrades.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
