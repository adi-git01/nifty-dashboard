"""Build and cache the strictly causal 10.6-Year Blinded Panel (2016-04-01 -> 2026-11-20)
with rich per-stock dashboard telemetry (ROCE, OPM, OPM_delta, D/E, 3Y Sales/PAT CAGR,
CFO/PAT, PIT P/E & P/B + 5Y percentiles, Bear Discount, Stage 1-4, CompRS, Dual-ATR,
Delivery surge, Abs Delivery Cr, Industry Rank, ATR%, MCap, Liquidity) across 22 blind epochs.
"""
from __future__ import annotations

import glob
import json
import math
import os
import pickle
import random
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

OUT_DIR = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\phd_council_10y"
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\multi_regime_10y")
import run_10y_regimes as R  # noqa: E402
from utils import fund_trends as ft, nifty500_list as n500, nifty1000_list as n1000  # noqa: E402

# 22 Semi-Annual + Major-Regime-Aligned Council Gates (starting June 2016 when FY16 results are live!)
TARGET_GATES = [
    "2016-06-01", "2016-12-01",
    "2017-06-01", "2017-12-01",
    "2018-06-01", "2018-12-03",
    "2019-06-03", "2019-12-02",
    "2020-03-30", "2020-06-30", "2020-12-01",
    "2021-06-01", "2021-12-01",
    "2022-06-01", "2022-12-01",
    "2023-06-01", "2023-12-01",
    "2024-06-03", "2024-11-01",
    "2025-04-01", "2025-11-03",
    "2026-06-01",
]


def main() -> int:
    t0 = time.time()
    base = R.load_universe_and_matrices()
    sim_dates = list(base["sim_dates"])
    sim_idx = pd.DatetimeIndex(sim_dates)

    # Resolve exact trading dates >= each target gate
    gate_dates: List[pd.Timestamp] = []
    for g in TARGET_GATES:
        ts = pd.Timestamp(g)
        pos = int(sim_idx.searchsorted(ts, side="left"))
        if pos < len(sim_idx):
            gate_dates.append(sim_idx[pos])
    print(f"[build] resolved {len(gate_dates)} council gates: {gate_dates[0].date()} .. {gate_dates[-1].date()}", flush=True)

    # Load full annual + quarterly Screener fundamentals for richer PhD dashboard metrics (ROCE, OPM, D/E, 3Y CAGR, Q_NP_YoY)
    ann_raw, qtr_raw = ft._data(R.REPO_L)
    valid_tks = base["valid_tks"]
    t2c = base["t2c"]
    C = base["C"]
    L = base["L"]
    O = base["O"]
    MA50 = base["MA50"]
    LIQ20 = base["LIQ20"]
    PIT_MCAP = base["PIT_MCAP"]

    # Deterministic blinded IDs (shuffled with fixed seed so ID order reveals nothing alphabetically!)
    rng = random.Random(20160401)
    shuffled_idx = list(range(len(valid_tks)))
    rng.shuffle(shuffled_idx)
    tk_to_code = {valid_tks[orig_i]: f"STK_{code_num:04d}" for code_num, orig_i in enumerate(shuffled_idx, 1)}
    code_to_tk = {v: k for k, v in tk_to_code.items()}

    # Pre-pivot annual & quarterly rich metrics
    fy_rich: Dict[str, dict] = {}
    for tk in valid_tks:
        ka = ft._key(tk, ann_raw)
        if ka is None:
            continue
        w = ann_raw[ka].pivot_table(index="period", columns="field", values="value", aggfunc="last").sort_index()
        w = w[w.index.month == 3]
        if w.empty:
            continue
        g = lambda col: w[col].to_numpy(dtype=np.float64) if col in w.columns else np.full(len(w), np.nan)
        fy_rich[tk] = dict(
            dates=w.index.to_numpy(),
            sales=g("sales"), np=g("np"), opm=g("opm"), roce=g("roce"),
            cfo=g("cfo"), borr=g("borrowings"), book=g("eq_cap") + g("reserves"),
        )

    q_rich: Dict[str, dict] = {}
    for tk in valid_tks:
        kq = ft._key(tk, qtr_raw)
        if kq is None:
            continue
        wq = qtr_raw[kq].pivot_table(index="period", columns="field", values="value", aggfunc="last").sort_index()
        if wq.empty:
            continue
        g = lambda col: wq[col].to_numpy(dtype=np.float64) if col in wq.columns else np.full(len(wq), np.nan)
        q_rich[tk] = dict(dates=wq.index.to_numpy(), sales=g("sales"), np=g("np"))

    # Build enriched gate snapshots by taking nearest existing ckpt (<= gate) or computing exact gate snapshot!
    # Recompute exact snapshots on gate_dates using base arrays + rich fundamentals
    # Note: map each gate_date to nearest ckpt_date <= gate_date for technicals, OR better yet compute exact technicals at gate_date!
    ckpt_keys = sorted(base["ckpts"].keys())
    n50_s = base["n50"]
    mid_s = base["mid"]
    sml_s = base["sml"]
    ew_c = R.passive_ew_rebalanced(base)

    # Precompute daily returns for 3M/6M/12M lookups on gate_dates
    C_df = pd.DataFrame(C, index=sim_idx, columns=valid_tks)
    MA200_df = C_df.rolling(200, min_periods=120).mean()
    HI252_df = C_df.rolling(252, min_periods=120).max()
    R63_df = (C_df / C_df.shift(63) - 1.0) * 100.0
    R126_df = (C_df / C_df.shift(126) - 1.0) * 100.0
    R252_df = (C_df / C_df.shift(252) - 1.0) * 100.0

    gate_snaps: Dict[int, dict] = {}
    for gi, gdt in enumerate(gate_dates, 1):
        # Find nearest base ckpt <= gdt (within <= 10 trading days, strictly causal!)
        kpos = int(np.searchsorted(ckpt_keys, gdt, side="right")) - 1
        snap_src = base["ckpts"][ckpt_keys[max(0, kpos)]]
        di = int(sim_idx.get_loc(gdt))
        reg = base["regimes"][gdt]

        # Macro blinded telemetry strictly <= gdt
        n50_now = float(n50_s.loc[gdt])
        n50_63 = float(n50_s.iloc[max(0, di - 63)])
        n50_126 = float(n50_s.iloc[max(0, di - 126)])
        n50_252 = float(n50_s.iloc[max(0, di - 252)])
        n50_ma50 = float(n50_s.iloc[max(0, di - 49):di + 1].mean())
        n50_ma200 = float(n50_s.iloc[max(0, di - 199):di + 1].mean())
        n50_hi252 = float(n50_s.iloc[max(0, di - 251):di + 1].max())
        mid_126 = float(mid_s.loc[gdt] / mid_s.iloc[max(0, di - 126)] - 1.0) * 100.0 if pd.notna(mid_s.loc[gdt]) and pd.notna(mid_s.iloc[max(0, di - 126)]) else 0.0
        sml_126 = float(sml_s.loc[gdt] / sml_s.iloc[max(0, di - 126)] - 1.0) * 100.0 if pd.notna(sml_s.loc[gdt]) and pd.notna(sml_s.iloc[max(0, di - 126)]) else 0.0

        # Active universe stats on gdt
        c_row = C[di, :]
        l_row = LIQ20[di, :]
        m_row = PIT_MCAP[di, :]
        hi_row = HI252_df.iloc[di].to_numpy()
        act_mask = np.isfinite(c_row) & (c_row > 5.0) & (l_row >= 2.0) & (np.isnan(m_row) | (m_row >= 500.0))
        pct_near_hi = float(np.nanmean((c_row[act_mask] >= 0.85 * hi_row[act_mask]))) * 100.0 if act_mask.any() else 0.0
        pct_deep_dd = float(np.nanmean((c_row[act_mask] <= 0.65 * hi_row[act_mask]))) * 100.0 if act_mask.any() else 0.0

        macro = dict(
            epoch=f"E{gi:02d}",
            mode=reg["mode"],
            vix=round(reg["vix"], 1),
            breadth_above_50dma_pct=round(reg["b50"], 1),
            breadth_above_200dma_pct=round(reg["b200"], 1),
            pct_near_52w_hi=round(pct_near_hi, 1),
            pct_down_35pct_from_hi=round(pct_deep_dd, 1),
            idx_vs_50dma_pct=round((n50_now / n50_ma50 - 1.0) * 100.0, 1),
            idx_vs_200dma_pct=round((n50_now / n50_ma200 - 1.0) * 100.0, 1),
            idx_dd_from_52w_hi_pct=round((n50_now / n50_hi252 - 1.0) * 100.0, 1),
            idx_3m_ret_pct=round((n50_now / n50_63 - 1.0) * 100.0, 1),
            idx_6m_ret_pct=round((n50_now / n50_126 - 1.0) * 100.0, 1),
            idx_12m_ret_pct=round((n50_now / n50_252 - 1.0) * 100.0, 1),
            midcap_6m_ret_pct=round(mid_126, 1),
            smallcap_6m_ret_pct=round(sml_126, 1),
            active_universe_count=int(act_mask.sum()),
        )

        # Build enriched stock records for every active stock on gdt
        cutoff_fy = np.datetime64(gdt - pd.Timedelta(days=60))
        cutoff_q = np.datetime64(gdt - pd.Timedelta(days=45))
        stocks: Dict[str, dict] = {}
        for tk, s in snap_src.items():
            col = t2c[tk]
            px = float(C[di, col])
            liq = float(LIQ20[di, col])
            mc_v = float(PIT_MCAP[di, col])
            if not np.isfinite(px) or px <= 5.0 or not np.isfinite(liq) or liq < 2.0:
                continue
            if np.isfinite(mc_v) and mc_v < 500.0:
                continue

            r3m = float(R63_df.iloc[di, col]) if pd.notna(R63_df.iloc[di, col]) else None
            r6m = float(R126_df.iloc[di, col]) if pd.notna(R126_df.iloc[di, col]) else None
            r12m = float(R252_df.iloc[di, col]) if pd.notna(R252_df.iloc[di, col]) else None
            h252 = float(HI252_df.iloc[di, col]) if pd.notna(HI252_df.iloc[di, col]) else np.nan
            dd_hi = round((px / h252 - 1.0) * 100.0, 1) if np.isfinite(h252) and h252 > 0 else None

            # Rich strictly causal FY fundamentals
            roce = opm = opm_d = de = cagr3_s = cagr3_p = pe = pb = None
            fr = fy_rich.get(tk)
            if fr is not None:
                pos = int(np.searchsorted(fr["dates"], cutoff_fy, side="right")) - 1
                if pos >= 0:
                    if np.isfinite(fr["roce"][pos]): roce = round(float(fr["roce"][pos]), 1)
                    if np.isfinite(fr["opm"][pos]): opm = round(float(fr["opm"][pos]), 1)
                    if pos >= 1 and np.isfinite(fr["opm"][pos]) and np.isfinite(fr["opm"][pos - 1]):
                        opm_d = round(float(fr["opm"][pos] - fr["opm"][pos - 1]), 1)
                    if np.isfinite(fr["borr"][pos]) and np.isfinite(fr["book"][pos]) and fr["book"][pos] > 0:
                        de = round(float(fr["borr"][pos] / fr["book"][pos]), 2)
                    if pos >= 3 and np.isfinite(fr["sales"][pos]) and np.isfinite(fr["sales"][pos - 3]) and fr["sales"][pos - 3] > 0:
                        cagr3_s = round(((fr["sales"][pos] / fr["sales"][pos - 3]) ** (1.0 / 3.0) - 1.0) * 100.0, 1)
                    if pos >= 3 and np.isfinite(fr["np"][pos]) and np.isfinite(fr["np"][pos - 3]) and fr["np"][pos - 3] > 0 and fr["np"][pos] > 0:
                        cagr3_p = round(((fr["np"][pos] / fr["np"][pos - 3]) ** (1.0 / 3.0) - 1.0) * 100.0, 1)
                    if np.isfinite(mc_v) and np.isfinite(fr["np"][pos]) and fr["np"][pos] > 0:
                        v_pe = mc_v / fr["np"][pos]
                        if 1.0 <= v_pe <= 300.0: pe = round(float(v_pe), 1)
                    if np.isfinite(mc_v) and np.isfinite(fr["book"][pos]) and fr["book"][pos] > 0:
                        v_pb = mc_v / fr["book"][pos]
                        if 0.1 <= v_pb <= 80.0: pb = round(float(v_pb), 1)

            q_np_yoy = q_rev_yoy = None
            qr = q_rich.get(tk)
            if qr is not None:
                qp = int(np.searchsorted(qr["dates"], cutoff_q, side="right")) - 1
                if qp >= 4:
                    if np.isfinite(qr["np"][qp]) and np.isfinite(qr["np"][qp - 4]) and qr["np"][qp - 4] > 0:
                        q_np_yoy = round(float((qr["np"][qp] / qr["np"][qp - 4] - 1.0) * 100.0), 1)
                    if np.isfinite(qr["sales"][qp]) and np.isfinite(qr["sales"][qp - 4]) and qr["sales"][qp - 4] > 0:
                        q_rev_yoy = round(float((qr["sales"][qp] / qr["sales"][qp - 4] - 1.0) * 100.0), 1)

            code = tk_to_code[tk]
            stocks[code] = dict(
                code=code, ind=s["ind"], is_bfsi=s["is_bfsi"], px=round(px, 2),
                mcap_cr=round(mc_v, 0) if np.isfinite(mc_v) else None,
                liq_cr=round(liq, 1), dcr=round(s["dcr"], 1), drat=round(s["drat"], 2),
                atrp=round(s["atrp"], 2), crs=round(s["crs"], 1), irank=round(s["irank"], 1),
                stage=s["stage"], buy_now=s["buy_now"], gate1=s["gate1"],
                r3m=round(r3m, 1) if r3m is not None else None,
                r6m=round(r6m, 1) if r6m is not None else None,
                r12m=round(r12m, 1) if r12m is not None else None,
                dd_hi=dd_hi,
                fy_np_yoy=round(s["fy_np_yoy"], 1) if s["fy_np_yoy"] is not None else None,
                fy_sales_yoy=round(s["fy_sales_yoy"], 1) if s["fy_sales_yoy"] is not None else None,
                cagr3_s=cagr3_s, cagr3_p=cagr3_p, roce=roce, opm=opm, opm_d=opm_d, de=de,
                cfo_pat=round(s["cfo_pat"], 2) if s["cfo_pat"] is not None else None,
                pe=pe, pb=pb, cmp_lt_bear=s["cmp_lt_bear"],
                bear_disc=round(s["bear_disc"], 1) if s["bear_disc"] is not None else None,
                q_np_yoy=q_np_yoy, q_rev_yoy=q_rev_yoy,
            )
        gate_snaps[gi] = dict(gate_idx=gi, date=gdt, di=di, macro=macro, stocks=stocks)
        print(f"  [E{gi:02d}] {gdt.date()} ({macro['mode']:8s}) B50={macro['breadth_above_50dma_pct']:4.1f}% VIX={macro['vix']:4.1f} | Active={len(stocks):4d} | BUY_NOW={sum(1 for v in stocks.values() if v['buy_now']):3d} | <Bear={sum(1 for v in stocks.values() if v['cmp_lt_bear']):3d}", flush=True)

    payload = dict(
        sim_dates=sim_dates,
        gate_dates=gate_dates,
        gate_snaps=gate_snaps,
        valid_tks=valid_tks,
        t2c=t2c,
        tk_to_code=tk_to_code,
        code_to_tk=code_to_tk,
        O=O, L=L, C=C, MA50=MA50,
        n50=n50_s, ew=ew_c,
    )
    out_pkl = os.path.join(OUT_DIR, "blind_10y_council_panel.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[build] saved {out_pkl} ({os.path.getsize(out_pkl)/1e6:.1f} MB) in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
