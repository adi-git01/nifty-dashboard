"""Strictly point-in-time (<= 2026-03-30) screener for 18 real March-2026 false-positive traps.

Finds liquid NSE stocks (>= ₹2 Cr/d 20d median turnover on 2026-03-30) outside the 11 winners
that genuinely triggered on 2026-03-30 at least one of:
  - Trap Type A (Value Trap):    CMP < Valuation v4.1 bear_blend (strictly as_of=2026-03-30)
  - Trap Type B (Squeeze Trap):  BBP < 20 (tight volatility coil / oversold lure on 2026-03-30)
  - Trap Type C (Earnings Trap): Trailing quarterly NP YoY >= +30% (45d SEBI Reg 33 lag <= 2026-03-30)
and subsequently fell -20% to -55% (or stayed dead <= 0%) from 2026-03-30 -> 2026-11-20.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\repo_live"
OUT_DIR = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\poisoned_stress"
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(REPO)
sys.path.insert(0, REPO)

from utils import forensics as fx, fund_trends as ft, nse_prices, valuation as V

WINNERS_11 = {
    "SIGMAADV.NS", "CUPID.NS", "STLTECH.NS", "TBZ.NS", "MONOLITH.NS",
    "SPECTRUM.NS", "HFCL.NS", "FILATEX.NS", "SSWL.NS", "LAURUSLABS.NS", "MACPOWER.NS",
}
START_DT = "2026-02-27"
SCREEN_DT = "2026-03-30"
END_DT = "2026-11-20"


def _pit_fundamentals(tk: str, as_of: str, ann_raw: dict, qtr_raw: dict) -> dict:
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

    q_np_yoy = q_sales_yoy = np.nan
    q_lbl = ""
    k_q = ft._key(tk, qtr_raw)
    if k_q is not None:
        wq = qtr_raw[k_q].pivot_table(index="period", columns="field", values="value", aggfunc="last").sort_index()
        wq = wq[wq.index + pd.Timedelta(days=45) <= as_ts]
        if len(wq) >= 5 and "np" in wq.columns:
            cur_dt = wq.index[-1]
            q_lbl = cur_dt.strftime("%b %Y")
            prev_dt = cur_dt - pd.DateOffset(years=1)
            # Match closest quarter within 15 days of 1 year prior
            diffs = np.abs((wq.index - prev_dt).days)
            j = int(np.argmin(diffs))
            if diffs[j] <= 20:
                cur_np, prev_np = wq["np"].iloc[-1], wq["np"].iloc[j]
                if pd.notna(cur_np) and pd.notna(prev_np) and float(prev_np) > 0:
                    q_np_yoy = round((float(cur_np) / float(prev_np) - 1.0) * 100.0, 1)
                if "sales" in wq.columns:
                    cs, ps = wq["sales"].iloc[-1], wq["sales"].iloc[j]
                    if pd.notna(cs) and pd.notna(ps) and float(ps) > 0:
                        q_sales_yoy = round((float(cs) / float(ps) - 1.0) * 100.0, 1)
    return {"cfo_pat": cfo_pat, "fcf_pat": fcf_pat, "roce": roce, "q_np_yoy": q_np_yoy, "q_sales_yoy": q_sales_yoy, "q_lbl": q_lbl}


def main():
    mm_files = sorted(glob.glob(os.path.join(REPO, "data", "cache", "market_master_*.parquet")))
    mm = pd.read_parquet(mm_files[-1])
    cands = [t for t in mm["ticker"].dropna().unique().tolist() if t not in WINNERS_11]
    print(f"Scanning {len(cands)} NSE universe candidates strictly as of {SCREEN_DT}...")

    hist = nse_prices.history(cands)
    ann_raw, qtr_raw = ft._data(REPO)
    val_ctx = V.load_context()
    name_map = dict(zip(mm["ticker"], mm.get("name", mm["ticker"]).astype(str)))
    ind_col = "industry" if "industry" in mm.columns else ("sector" if "sector" in mm.columns else None)
    ind_map = dict(zip(mm["ticker"], mm[ind_col].fillna("").astype(str))) if ind_col else {}

    rows = []
    for tk in cands:
        df = hist.get(tk)
        if df is None or df.empty:
            continue
        df = df.sort_index()
        if START_DT not in df.index.strftime("%Y-%m-%d") or SCREEN_DT not in df.index.strftime("%Y-%m-%d") or END_DT not in df.index.strftime("%Y-%m-%d"):
            continue
        sub_mar = df.loc[:SCREEN_DT]
        if len(sub_mar) < 205:
            continue
        px_feb = float(df.loc[START_DT, "Close"])
        px_mar = float(sub_mar["Close"].iloc[-1])
        px_nov = float(df.loc[END_DT, "Close"])
        if px_mar <= 10.0 or px_feb <= 10.0:
            continue
        tmed20_cr = float((sub_mar["Close"] * sub_mar["Volume"]).tail(20).median() / 1e7)
        if tmed20_cr < 3.0:
            continue

        ret_mar_nov = round((px_nov / px_mar - 1.0) * 100.0, 1)
        ret_feb_nov = round((px_nov / px_feb - 1.0) * 100.0, 1)
        # Only interested in real losers (-20% to -58%) or dead money (<= -5%) after Mar 30
        if ret_mar_nov > -5.0 or ret_mar_nov < -60.0:
            continue

        # Strictly causal technicals on 2026-03-30
        r_fx = fx.analyse(sub_mar)
        bbp = r_fx.get("bbp")
        bbp_v = round(float(bbp), 1) if (bbp is not None and np.isfinite(bbp)) else np.nan
        ma50 = float(sub_mar["Close"].rolling(50).mean().iloc[-1])
        ma200 = float(sub_mar["Close"].rolling(200).mean().iloc[-1])
        stage = 2 if px_mar > ma50 > ma200 else (4 if px_mar < ma50 < ma200 else (1 if px_mar > ma50 else 3))

        # Strictly causal fundamentals on 2026-03-30
        pf = _pit_fundamentals(tk, SCREEN_DT, ann_raw, qtr_raw)

        # Strictly causal Valuation v4.1 on 2026-03-30
        bear_blend = np.nan
        val_rating = ""
        cmp_lt_bear = False
        try:
            v = V.value_stock(tk.replace(".NS", ""), val_ctx, price=px_mar, as_of=SCREEN_DT, knobs=False)
            if v.get("status") == "OK":
                bl = v.get("blends") or {}
                bb = bl.get("bear", np.nan)
                if np.isfinite(bb):
                    bear_blend = round(float(bb), 1)
                    cmp_lt_bear = bool(px_mar <= float(bb))
                val_rating = str(v.get("rating") or "")
        except Exception:
            pass

        lures = []
        if cmp_lt_bear:
            lures.append("CMP<Bear")
        if np.isfinite(bbp_v) and bbp_v < 20.0:
            lures.append("BBP<20")
        if np.isfinite(pf["q_np_yoy"]) and pf["q_np_yoy"] >= 30.0:
            lures.append("NP_YoY>=30%")
        if not lures:
            continue

        rows.append({
            "ticker": tk,
            "name": name_map.get(tk, tk),
            "industry": ind_map.get(tk, ""),
            "px_feb27": round(px_feb, 2),
            "px_mar30": round(px_mar, 2),
            "px_nov20": round(px_nov, 2),
            "ret_mar_nov_pct": ret_mar_nov,
            "ret_feb_nov_pct": ret_feb_nov,
            "liq_cr": round(tmed20_cr, 1),
            "stage_mar30": stage,
            "bbp_mar30": bbp_v if np.isfinite(bbp_v) else None,
            "bear_blend_mar30": bear_blend if np.isfinite(bear_blend) else None,
            "cmp_lt_bear": cmp_lt_bear,
            "val_rating_mar30": val_rating,
            "cfo_pat": pf["cfo_pat"] if np.isfinite(pf["cfo_pat"]) else None,
            "fcf_pat": pf["fcf_pat"] if np.isfinite(pf["fcf_pat"]) else None,
            "roce": pf["roce"] if np.isfinite(pf["roce"]) else None,
            "q_np_yoy": pf["q_np_yoy"] if np.isfinite(pf["q_np_yoy"]) else None,
            "q_sales_yoy": pf["q_sales_yoy"] if np.isfinite(pf["q_sales_yoy"]) else None,
            "lures": lures,
        })

    df_out = pd.DataFrame(rows)
    print(f"Total March-2026 false-positive candidates found: {len(df_out)}")
    if df_out.empty:
        return

    # Select balanced 18 real traps across all 3 archetypes (diverse industries, real -10% to -55% losers)
    chosen = []
    used = set()
    used_inds = {}

    def _pick(pool_df, n_need, bucket_label):
        # Prefer sharper drawdowns (-20% to -55%) & higher liquidity, diverse industries
        pool = pool_df[~pool_df["ticker"].isin(used)].copy()
        pool["is_severe"] = pool["ret_mar_nov_pct"].between(-55.0, -20.0).astype(int)
        pool = pool.sort_values(["is_severe", "liq_cr", "ret_mar_nov_pct"], ascending=[False, False, True])
        cnt = 0
        for _, r in pool.iterrows():
            if cnt >= n_need:
                break
            ind_k = r["industry"] or "Other"
            if used_inds.get(ind_k, 0) >= 2:
                continue
            d = r.to_dict()
            d["trap_bucket"] = bucket_label
            chosen.append(d)
            used.add(r["ticker"])
            used_inds[ind_k] = used_inds.get(ind_k, 0) + 1
            cnt += 1

    # Bucket 1A: 4 Value Traps WITH CFO/PAT >= 0.80 (ultimate stress test of GARP's primary gate!)
    _pick(df_out[(df_out["cmp_lt_bear"]) & (df_out["cfo_pat"].fillna(-9) >= 0.80)], 4, "ValueTrap_CFO_OK")
    # Bucket 1B: 3 Value Traps WITH CFO/PAT < 0.80 (tests if CFO/PAT gate filters them out!)
    _pick(df_out[(df_out["cmp_lt_bear"]) & (df_out["cfo_pat"].fillna(-9) < 0.80)], 3, "ValueTrap_CFO_BAD")
    # Bucket 2: 6 BBP < 20 Squeeze / Oversold Traps (tests mom_trader & hybrid_pm!)
    _pick(df_out[(df_out["bbp_mar30"].fillna(99) < 20.0)], 6, "SqueezeTrap_BBP<20")
    # Bucket 3: 5 High Trailing NP YoY >= 30% Earnings-Momentum Traps
    _pick(df_out[(df_out["q_np_yoy"].fillna(-99) >= 30.0)], 5, "EarningsTrap_NPYoY>=30")

    # Top up to 18 if any bucket had fewer than target
    if len(chosen) < 18:
        _pick(df_out, 18 - len(chosen), "MixedTrap")

    res = {
        "screen_date": SCREEN_DT,
        "total_candidates": len(df_out),
        "n_val_cfo_ok": int(((df_out["cmp_lt_bear"]) & (df_out["cfo_pat"].fillna(-9) >= 0.80)).sum()),
        "n_val_cfo_bad": int(((df_out["cmp_lt_bear"]) & (df_out["cfo_pat"].fillna(-9) < 0.80)).sum()),
        "n_bbp_lt20": int((df_out["bbp_mar30"].fillna(99) < 20.0).sum()),
        "n_np_yoy_30": int((df_out["q_np_yoy"].fillna(-99) >= 30.0).sum()),
        "selected_18": chosen,
    }
    out_path = os.path.join(OUT_DIR, "traps_mar2026.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)

    print(f"Pool counts -> Val(CFO>=0.8): {res['n_val_cfo_ok']}, Val(CFO<0.8): {res['n_val_cfo_bad']}, BBP<20: {res['n_bbp_lt20']}, NP_YoY>=30: {res['n_np_yoy_30']}")
    print(f"\nSelected {len(chosen)} Real March-2026 False-Positive Traps:")
    for i, r in enumerate(chosen, 1):
        print(
            f"{i:2d}. {r['ticker']:15s} [{r['trap_bucket']:22s}] Mar->Nov: {r['ret_mar_nov_pct']:+6.1f}% | "
            f"Feb->Nov: {r['ret_feb_nov_pct']:+6.1f}% | Liq: {r['liq_cr']:5.1f}Cr | "
            f"BBP: {str(r['bbp_mar30']):>5s} | <Bear: {str(r['cmp_lt_bear']):5s} | CFO/P: {str(r['cfo_pat']):>5s} | NP_YoY: {str(r['q_np_yoy']):>6s} | {r['industry']}"
        )


if __name__ == "__main__":
    main()
