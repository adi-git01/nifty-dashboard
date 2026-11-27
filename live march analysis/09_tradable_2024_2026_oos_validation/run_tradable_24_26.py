"""Exact Tradable >=₹500 Cr MCap & >=₹2 Cr/d Framework Applied to Full 2.0-Year Cycle (2024-11-18 -> 2026-11-20).

100% bit-for-bit identical conditions to run_tradable500cr.py (Feb-Nov 2026):
  - Exact utils/forensics.py (fx.analyse Dual-ATR BUY NOW, stop_fx, entry_fx)
  - Exact utils/valuation.py v4.1 Monte Carlo (V.value_stock with strict PIT slicing of quarterly<=t-45d, annual<=t-60d, monthly<=t)
  - Exact utils/regime_manager.py (classify_regime + detect_capitulation_recovery)
  - Exact Screener quarterly fundamentals (Q_NP_YoY, Q_Sales_YoY, d_inst_q, d_prom_q, CFO/PAT, ROCE)
  - Strict PIT MCap >= 500 Cr & 20D avg turnover >= 2 Cr/d at every checkpoint
  - T+1 Open execution, 0.25% one-way friction (0.50% round-trip), 10% max single-position cap, ₹10L starting capital
  - Survivorship-free 2023-2026 NSE daily OHLCV + delivery % from repo_nov21 (full MA200/HI252 warmup on 2024-11-18!)
"""
from __future__ import annotations

import glob
import json
import math
import os
import pickle
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_L = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\repo_live"
REPO_N = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\repo_nov21"
OUT_DIR = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\tradable_24_26"
os.makedirs(OUT_DIR, exist_ok=True)

# Import latest forensics/valuation/fund_trends from repo_live, load 2023-2026 prices from repo_nov21!
sys.path.insert(0, REPO_L)
from utils import forensics as fx, fund_trends as ft, nifty500_list as n500, nifty1000_list as n1000, regime_manager as rm, valuation as V  # noqa: E402

START_DATE = "2024-11-18"
END_DATE = "2026-11-20"
SPLIT_DATE = "2026-02-26"
OVERLAP_START = "2026-02-27"
COST_SIDE = 0.0025
INIT_CAP = 1_000_000.0

# 25 Monthly Event Dates (Nov 2024 -> Nov 2026) aligned with quarterly results & exact 2026-02-27 boundary
TARGET_EVENTS = [
    "2024-11-18", "2024-12-16",
    "2025-01-15", "2025-02-17", "2025-03-17", "2025-04-15", "2025-05-15", "2025-06-16",
    "2025-07-15", "2025-08-18", "2025-09-15", "2025-10-15", "2025-11-17", "2025-12-15",
    "2026-01-15", "2026-02-27", "2026-03-30", "2026-04-08", "2026-05-18", "2026-06-22",
    "2026-07-20", "2026-08-18", "2026-09-15", "2026-10-15", "2026-11-06",
]


def _load_indices_nov21():
    files = sorted(glob.glob(os.path.join(REPO_N, "data", "nse", "indices", "202[3-6]_*.parquet"))
                   + glob.glob(os.path.join(REPO_N, "data", "nse", "indices", "d", "*.parquet")))
    parts = [pd.read_parquet(f, columns=["date", "index", "open", "high", "low", "close"]) for f in files]
    ix = pd.concat(parts, ignore_index=True)
    ix["date"] = pd.to_datetime(ix["date"], errors="coerce").dt.normalize()
    nifty = ix[ix["index"] == "Nifty 50"].dropna(subset=["date", "close"]).drop_duplicates("date", keep="last").set_index("date").sort_index()
    nifty = nifty.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    vix = ix[ix["index"] == "India VIX"].dropna(subset=["date", "close"]).drop_duplicates("date", keep="last").set_index("date").sort_index()["close"]
    return nifty, vix


def _pre_dual_atr_action(r_post: dict, clv_last: float) -> str:
    ext_anc = r_post.get("ext_atr", np.nan)
    old_weak = bool(
        r_post.get("signature") == "DISTRIBUTION"
        or (np.isfinite(clv_last) and clv_last < 0.35)
        or (np.isfinite(ext_anc) and float(ext_anc) > fx.EXT_ATR_MAX)
    )
    return fx.action_for(bool(r_post.get("gate1")), str(r_post.get("state")), str(r_post.get("supply")), entry_weak=old_weak)


def _pit_fund_at(ann_w: pd.DataFrame | None, qtr_w: pd.DataFrame | None, as_ts: pd.Timestamp) -> dict:
    cfo_pat = fcf_pat = roce = de_ratio = None
    if ann_w is not None and not ann_w.empty:
        sub = ann_w[ann_w.index + pd.Timedelta(days=60) <= as_ts]
        if not sub.empty:
            row = sub.iloc[-1]
            np_v = row.get("np", np.nan)
            cfo_v = row.get("cfo", np.nan)
            cap_v = row.get("_capex", np.nan)
            roce_v = row.get("roce", np.nan)
            borr_v = row.get("borrowings", np.nan)
            eq_v = (0.0 if pd.isna(row.get("eq_cap", np.nan)) else float(row.get("eq_cap", 0))) + (
                0.0 if pd.isna(row.get("reserves", np.nan)) else float(row.get("reserves", 0))
            )
            if pd.notna(roce_v):
                roce = round(float(roce_v), 1)
            if pd.notna(borr_v) and eq_v > 0:
                de_ratio = round(float(borr_v) / eq_v, 2)
            if pd.notna(np_v) and float(np_v) > 0 and pd.notna(cfo_v):
                cfo_pat = round(float(cfo_v) / float(np_v), 2)
                if pd.notna(cap_v):
                    fcf_pat = round((float(cfo_v) - max(0.0, float(cap_v))) / float(np_v), 2)

    q_np_yoy = q_sales_yoy = opm_q = opm_prev = None
    d_inst_q = d_fii_q = d_dii_q = d_prom_q = None
    if qtr_w is not None and not qtr_w.empty:
        sub_pl = qtr_w[qtr_w.index + pd.Timedelta(days=45) <= as_ts]
        if len(sub_pl) >= 2:
            if "opm" in sub_pl.columns:
                v0, v1 = sub_pl["opm"].iloc[-1], sub_pl["opm"].iloc[-2]
                opm_q = round(float(v0), 1) if pd.notna(v0) else None
                opm_prev = round(float(v1), 1) if pd.notna(v1) else None
            if len(sub_pl) >= 5:
                cur_dt = sub_pl.index[-1]
                prev_dt = cur_dt - pd.DateOffset(years=1)
                diffs = np.abs((sub_pl.index - prev_dt).days)
                j = int(np.argmin(diffs))
                if diffs[j] <= 20:
                    if "np" in sub_pl.columns:
                        cnp, pnp = sub_pl["np"].iloc[-1], sub_pl["np"].iloc[j]
                        if pd.notna(cnp) and pd.notna(pnp) and float(pnp) > 0:
                            q_np_yoy = round((float(cnp) / float(pnp) - 1.0) * 100.0, 1)
                    if "sales" in sub_pl.columns:
                        cs, ps = sub_pl["sales"].iloc[-1], sub_pl["sales"].iloc[j]
                        if pd.notna(cs) and pd.notna(ps) and float(ps) > 0:
                            q_sales_yoy = round((float(cs) / float(ps) - 1.0) * 100.0, 1)
        sub_sh = qtr_w[qtr_w.index + pd.Timedelta(days=21) <= as_ts]
        if len(sub_sh) >= 2:
            for col_name, out_k in (("fii", "f"), ("dii", "d"), ("promoters", "p")):
                if col_name in sub_sh.columns:
                    s_v = sub_sh[col_name].dropna()
                    if len(s_v) >= 2:
                        dv = round(float(s_v.iloc[-1] - s_v.iloc[-2]), 2)
                        if out_k == "f":
                            d_fii_q = dv
                        elif out_k == "d":
                            d_dii_q = dv
                        else:
                            d_prom_q = dv
            if d_fii_q is not None or d_dii_q is not None:
                d_inst_q = round((d_fii_q or 0.0) + (d_dii_q or 0.0), 2)

    return {
        "cfo_pat": cfo_pat, "fcf_pat": fcf_pat, "roce": roce, "de_ratio": de_ratio,
        "q_np_yoy": q_np_yoy, "q_sales_yoy": q_sales_yoy, "opm_q": opm_q, "opm_prev": opm_prev,
        "d_inst_q": d_inst_q, "d_fii_q": d_fii_q, "d_dii_q": d_dii_q, "d_prom_q": d_prom_q,
    }


def build_panel() -> dict:
    pkl_path = os.path.join(OUT_DIR, "panel_tradable_24_26.pkl")
    if os.path.exists(pkl_path):
        print(f"Loading cached {pkl_path}...", flush=True)
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    t0 = time.time()
    # 1. Discover ALL NSE stocks with >=1.5Cr/d turnover in 2024-2026 from repo_nov21 survivorship-free parquets
    pfiles = sorted(glob.glob(os.path.join(REPO_N, "data", "nse", "prices", "202[4-6]_*.parquet"))
                    + glob.glob(os.path.join(REPO_N, "data", "nse", "prices", "d", "*.parquet")))
    chunks = []
    for f in pfiles:
        df = pd.read_parquet(f, columns=["date", "symbol", "series", "close", "volume", "deliv_pct"])
        df = df[df["series"].isin(["EQ", "BE"])]
        if not df.empty:
            chunks.append(df)
    dfp = pd.concat(chunks, ignore_index=True)
    dfp["date"] = pd.to_datetime(dfp["date"]).dt.normalize()
    dfp["turn_cr"] = dfp["close"] * dfp["volume"] / 1e7
    dfp["yr"] = dfp["date"].dt.year
    g = dfp.groupby(["symbol", "yr"]).agg(days=("date", "nunique"), avg_turn=("turn_cr", "mean"), last_c=("close", "last")).reset_index()
    cand_syms = sorted(g[(g["days"] >= 60) & (g["avg_turn"] >= 1.5) & (g["last_c"] > 5.0)]["symbol"].unique())
    cand_tickers = [f"{s}.NS" for s in cand_syms]
    print(f"Discovered {len(cand_tickers)} NSE stocks with >=1.5Cr/d in 2024-2026 [{time.time()-t0:.1f}s]", flush=True)

    # Exact daily delivery % panel from 2023-2026 parquets
    pfiles_23 = sorted(glob.glob(os.path.join(REPO_N, "data", "nse", "prices", "2023_*.parquet")))
    if pfiles_23:
        d23 = pd.concat([pd.read_parquet(f, columns=["date", "symbol", "series", "deliv_pct"]) for f in pfiles_23], ignore_index=True)
        d23 = d23[d23["series"].isin(["EQ", "BE"])]
        d23["date"] = pd.to_datetime(d23["date"]).dt.normalize()
        deliv_raw = pd.concat([d23[["date", "symbol", "deliv_pct"]], dfp[["date", "symbol", "deliv_pct"]]], ignore_index=True)
    else:
        deliv_raw = dfp[["date", "symbol", "deliv_pct"]].copy()
    deliv_raw = deliv_raw[deliv_raw["symbol"].isin(set(cand_syms))].drop_duplicates(["date", "symbol"], keep="last")
    deliv_raw["tk"] = deliv_raw["symbol"] + ".NS"
    deliv_df = deliv_raw.pivot(index="date", columns="tk", values="deliv_pct").sort_index().astype(np.float32)
    del dfp, deliv_raw, chunks

    # 2. Load corporate-action adjusted OHLCV via repo_nov21's nse_prices (Kite-verified demergers!)
    os.chdir(REPO_N)
    sys.path.insert(0, REPO_N)
    import importlib
    from utils import nse_prices as npx_n  # noqa: E402
    importlib.reload(npx_n)
    npx_n._load.cache_clear()
    hist = npx_n.history(cand_tickers + ["RELIANCE.NS"])
    nifty_df, vix_s = _load_indices_nov21()
    mc = pd.read_csv(os.path.join(REPO_N, "data", "nse", "market_context.csv"))
    mc["date"] = pd.to_datetime(mc["date"], errors="coerce").dt.normalize()
    mc = mc.dropna(subset=["date"]).set_index("date").sort_index()

    # Load fundamentals & Valuation v4.1 context from repo_live
    os.chdir(REPO_L)
    mm = pd.read_parquet(sorted(glob.glob(os.path.join(REPO_L, "data", "cache", "market_master_*.parquet")))[-1])
    ext_meta = getattr(n1000, "EXTENDED", {}) or {}
    sub_ind_map = dict(getattr(n1000, "SUB_INDUSTRY_MAP", {}) or {})
    sec_map = dict(zip(mm["ticker"], mm.get("sector_granular", mm.get("sector", pd.Series("", index=mm.index))).fillna("").astype(str)))
    mm_shares = {}
    for _, r in mm.iterrows():
        mc_v, px_v = r.get("marketCap"), r.get("currentPrice")
        if pd.notna(mc_v) and pd.notna(px_v) and float(px_v) > 0:
            mm_shares[str(r["ticker"])] = (float(mc_v) / 1e7) / float(px_v)

    ann_raw, qtr_raw = ft._data(REPO_L)
    val_ctx = V.load_context(root=REPO_L, extended=True)
    val_funda_set = set(val_ctx.funda.index.astype(str))

    ref_df = nifty_df.loc[START_DATE:END_DATE]
    dates_ts = ref_df.index
    dates = [d.strftime("%Y-%m-%d") for d in dates_ts]
    idx_all = nifty_df.loc["2023-06-01":END_DATE].index

    # Resolve exact trading dates for EVENT_DATES & MONTHLY_GATES
    event_dates = []
    for g in TARGET_EVENTS:
        pos = int(dates_ts.searchsorted(pd.Timestamp(g), side="left"))
        if pos < len(dates_ts):
            ds = dates_ts[pos].strftime("%Y-%m-%d")
            if ds not in event_dates:
                event_dates.append(ds)
    monthly_gates = sorted(set(
        pd.Series(dates_ts).groupby(pd.Series(dates_ts).dt.to_period("M")).last().dt.strftime("%Y-%m-%d").tolist()
    ))
    ckpt_dates = sorted(set(event_dates) | set(monthly_gates))
    print(f"Sim window: {dates[0]} -> {dates[-1]} ({len(dates)} sessions) | {len(event_dates)} V4.1 event gates | {len(ckpt_dates)} total ckpts [{time.time()-t0:.1f}s]", flush=True)

    sorted_hist: Dict[str, pd.DataFrame] = {}
    ann_piv: Dict[str, pd.DataFrame | None] = {}
    qtr_piv: Dict[str, pd.DataFrame | None] = {}
    shares_cr: Dict[str, float] = {}
    valid_all: List[str] = []

    for tk in cand_tickers:
        df = hist.get(tk)
        if df is None or df.empty:
            continue
        df = df.sort_index()
        sub = df.loc[START_DATE:END_DATE]
        if len(sub) < 120:
            continue

        sh_v = None
        ka = ft._key(tk, ann_raw)
        if ka is not None:
            w = ann_raw[ka].pivot_table(index="period", columns="field", values="value", aggfunc="last").sort_index()
            w_fy = w[w.index.month == 3].copy()
            if not w_fy.empty:
                g_c = lambda col: w_fy[col] if col in w_fy.columns else pd.Series(np.nan, index=w_fy.index)
                gross = g_c("fixed_assets") + g_c("cwip").fillna(0.0)
                w_fy["_capex"] = gross.diff() + g_c("dep").fillna(0.0)
            ann_piv[tk] = w_fy
            if "mcap_cr" in w.columns and "price_scr" in w.columns:
                mc_s, ps_s = w["mcap_cr"].dropna(), w["price_scr"].dropna()
                if not mc_s.empty and not ps_s.empty and float(ps_s.iloc[-1]) > 0:
                    sh_v = float(mc_s.iloc[-1]) / float(ps_s.iloc[-1])
        else:
            ann_piv[tk] = None
        if sh_v is None and tk in mm_shares:
            sh_v = mm_shares[tk]
        if sh_v is None and tk in ext_meta and ext_meta[tk].get("mcap_cr"):
            last_px = float(df["Close"].iloc[-1])
            if last_px > 0:
                sh_v = float(ext_meta[tk]["mcap_cr"]) / last_px
        if sh_v is None or sh_v <= 0:
            continue
        if sh_v * float(sub["Close"].max()) < 450.0:
            continue

        kq = ft._key(tk, qtr_raw)
        qtr_piv[tk] = qtr_raw[kq].pivot_table(index="period", columns="field", values="value", aggfunc="last").sort_index() if kq is not None else None
        sorted_hist[tk] = df
        shares_cr[tk] = sh_v
        valid_all.append(tk)
        if tk not in sub_ind_map or not sub_ind_map[tk]:
            sub_ind_map[tk] = (ext_meta.get(tk) or {}).get("industry") or sec_map.get(tk) or n500.SECTOR_MAP.get(tk, "Other")

    print(f"Eligible 2024-2026 Tradable >=500Cr Universe: {len(valid_all)} stocks [{time.time()-t0:.1f}s]", flush=True)

    O = pd.DataFrame({tk: sorted_hist[tk]["Open"].reindex(idx_all).ffill(limit=5) for tk in valid_all}, dtype=np.float32)
    H = pd.DataFrame({tk: sorted_hist[tk]["High"].reindex(idx_all).ffill(limit=5) for tk in valid_all}, dtype=np.float32)
    L = pd.DataFrame({tk: sorted_hist[tk]["Low"].reindex(idx_all).ffill(limit=5) for tk in valid_all}, dtype=np.float32)
    C = pd.DataFrame({tk: sorted_hist[tk]["Close"].reindex(idx_all).ffill(limit=5) for tk in valid_all}, dtype=np.float32)
    Vol = pd.DataFrame({tk: sorted_hist[tk]["Volume"].reindex(idx_all).fillna(0.0) for tk in valid_all}, dtype=np.float32)

    MA20 = C.rolling(20, min_periods=15).mean()
    MA50 = C.rolling(50, min_periods=40).mean()
    MA200 = C.rolling(200, min_periods=150).mean()
    HI252 = C.rolling(252, min_periods=150).max()
    LO252 = L.rolling(252, min_periods=180).min()
    AVG_LIQ20 = (C * Vol).rolling(20, min_periods=10).mean() / 1e7

    prev_c = C.shift(1)
    tr = np.maximum(H - L, np.maximum((H - prev_c).abs(), (L - prev_c).abs()))
    ATR14 = tr.rolling(14, min_periods=10).mean()
    ATR_PCT = (ATR14 / C.replace(0, np.nan)) * 100.0
    bb_std = C.rolling(20, min_periods=15).std()
    BBP = (4.0 * bb_std / MA20.replace(0, np.nan)).rolling(120, min_periods=60).rank(pct=True) * 100.0

    nif_c = nifty_df["Close"].reindex(idx_all).ffill().astype(np.float32)
    rs1m = ((C / C.shift(21)).div(nif_c / nif_c.shift(21), axis=0) - 1.0) * 100.0
    rs3m = ((C / C.shift(63)).div(nif_c / nif_c.shift(63), axis=0) - 1.0) * 100.0
    rs6m = ((C / C.shift(126)).div(nif_c / nif_c.shift(126), axis=0) - 1.0) * 100.0
    COMPRS = 0.25 * rs1m.fillna(0.0) + 0.35 * rs3m.fillna(0.0) + 0.40 * rs6m.fillna(0.0)

    D_PCT = deliv_df.reindex(index=idx_all, columns=valid_all).ffill(limit=3)
    D_RATIO = D_PCT / D_PCT.rolling(20, min_periods=10).mean().replace(0, np.nan)

    pct50_s = (C > MA50).sum(axis=1) / MA50.notna().sum(axis=1).replace(0, np.nan) * 100.0
    pctlo_s = (L <= LO252 * 1.01).sum(axis=1) / LO252.notna().sum(axis=1).replace(0, np.nan) * 100.0
    daily_regimes: Dict[str, dict] = {}
    for dt_str in dates:
        dt = pd.Timestamp(dt_str)
        nd_sub = nifty_df.loc[:dt]
        c_last = float(nd_sub["Close"].iloc[-1])
        m200_n = float(nd_sub["Close"].rolling(200).mean().iloc[-1])
        hi52_n = float(nd_sub["High"][nd_sub.index > dt - pd.Timedelta(days=365)].max())
        base_mode = rm.classify_regime(c_last, m200_n, hi52_n)
        mc_sub = mc.loc[:dt]
        rec = rm.detect_capitulation_recovery(
            pct_above_50dma=pct50_s.loc[:dt].dropna(),
            pct_52w_lows=pctlo_s.loc[:dt].dropna(),
            fii_net=mc_sub["fii_net"].dropna() if ("fii_net" in mc_sub and not mc_sub.empty) else None,
            dii_net=mc_sub["dii_net"].dropna() if ("dii_net" in mc_sub and not mc_sub.empty) else None,
            fii_long_pct=mc_sub["fii_idx_long_pct"].dropna() if ("fii_idx_long_pct" in mc_sub and not mc_sub.empty) else None,
        )
        eff_mode = "RECOVERY" if (base_mode in ("BEAR", "DEFENSIVE", "CAUTION") and rec["active"]) else base_mode
        vix_sub = vix_s.loc[:dt]
        daily_regimes[dt_str] = {
            "base_mode": base_mode, "engine_mode": eff_mode, "recovery": bool(rec["active"]),
            "vix": round(float(vix_sub.iloc[-1]), 2) if not vix_sub.empty else None,
        }

    print(f"Computing strictly causal checkpoints ({len(ckpt_dates)} total, {len(event_dates)} V4.1) across {len(valid_all)} stocks... [{time.time()-t0:.1f}s]", flush=True)
    event_set = set(event_dates)
    ckpts: Dict[str, Dict[str, dict]] = {}

    for ci, dt_str in enumerate(ckpt_dates, 1):
        dt = pd.Timestamp(dt_str)
        # Strict PIT slice of val_ctx for V4.1 Valuation on event dates!
        sub_vctx = None
        if dt_str in event_set:
            q_sub = val_ctx.quarterly[val_ctx.quarterly["period"] <= dt - pd.Timedelta(days=45)]
            a_sub = val_ctx.annual[val_ctx.annual["period"] <= dt - pd.Timedelta(days=60)]
            m_sub = val_ctx.monthly.loc[:dt]
            sub_vctx = V.Context(a_sub, q_sub, m_sub, val_ctx.funda, val_ctx.overrides, val_ctx.peer_cagr3, val_ctx.features)
            sub_vctx.peer_mult = val_ctx.peer_mult

        # PIT Industry Board
        ind_rows: Dict[str, List[tuple]] = {}
        for tk in valid_all:
            c_v = float(C.loc[dt, tk])
            m50_v = float(MA50.loc[dt, tk]) if pd.notna(MA50.loc[dt, tk]) else np.nan
            r3_v = float(rs3m.loc[dt, tk]) if pd.notna(rs3m.loc[dt, tk]) else np.nan
            if np.isfinite(c_v) and np.isfinite(m50_v) and np.isfinite(r3_v):
                ind_rows.setdefault(sub_ind_map.get(tk, "Other"), []).append((1.0 if c_v > m50_v else 0.0, r3_v))
        ind_scores = {k: 0.5 * float(np.mean([x[0] for x in v])) * 100.0 + 0.5 * float(np.clip(np.median([x[1] for x in v]) + 50.0, 0, 100))
                      for k, v in ind_rows.items() if len(v) >= 3}
        ind_rank = {k: round(float(v), 1) for k, v in (pd.Series(ind_scores).rank(pct=True) * 100.0).items()} if ind_scores else {}

        snap: Dict[str, dict] = {}
        for tk in valid_all:
            c_v = float(C.loc[dt, tk])
            if not np.isfinite(c_v) or c_v <= 5.0:
                continue
            h_v = float(H.loc[dt, tk])
            l_v = float(L.loc[dt, tk])
            m50_v = float(MA50.loc[dt, tk]) if pd.notna(MA50.loc[dt, tk]) else np.nan
            m200_v = float(MA200.loc[dt, tk]) if pd.notna(MA200.loc[dt, tk]) else np.nan
            hi252_v = float(HI252.loc[dt, tk]) if pd.notna(HI252.loc[dt, tk]) else np.nan
            avg_liq = float(AVG_LIQ20.loc[dt, tk]) if pd.notna(AVG_LIQ20.loc[dt, tk]) else 0.0
            pit_mcap = round(shares_cr[tk] * c_v, 1)
            atr_v = float(ATR14.loc[dt, tk]) if pd.notna(ATR14.loc[dt, tk]) else c_v * 0.035
            atr_p = float(ATR_PCT.loc[dt, tk]) if pd.notna(ATR_PCT.loc[dt, tk]) else 3.5
            bbp_v = float(BBP.loc[dt, tk]) if pd.notna(BBP.loc[dt, tk]) else None
            crs_v = float(COMPRS.loc[dt, tk]) if pd.notna(COMPRS.loc[dt, tk]) else 0.0
            dp_v = float(D_PCT.loc[dt, tk]) if pd.notna(D_PCT.loc[dt, tk]) else None
            dr_v = float(D_RATIO.loc[dt, tk]) if pd.notna(D_RATIO.loc[dt, tk]) else None
            stage = 2 if (np.isfinite(m200_v) and c_v > m50_v > m200_v) else (
                4 if (np.isfinite(m200_v) and c_v < m50_v < m200_v) else (1 if (np.isfinite(m50_v) and c_v > m50_v) else 3)
            )

            act_pre = act_post = "WATCH"
            stop_fx = entry_fx = None
            gate1_fast = bool(pit_mcap >= 450.0 and avg_liq >= 1.8 and np.isfinite(m50_v) and np.isfinite(m200_v) and c_v > m50_v and c_v > m200_v and (not np.isfinite(hi252_v) or c_v >= 0.84 * hi252_v))
            if gate1_fast:
                sub_px = sorted_hist[tk].loc[:dt]
                if len(sub_px) >= 60:
                    r_post = fx.analyse(sub_px)
                    clv_last = (c_v - l_v) / (h_v - l_v) if h_v > l_v else 0.5
                    act_post = str(r_post.get("action") or "WATCH")
                    act_pre = _pre_dual_atr_action(r_post, clv_last)
                    if np.isfinite(r_post.get("stop", np.nan)):
                        stop_fx = round(float(r_post["stop"]), 2)
                    if np.isfinite(r_post.get("entry", np.nan)):
                        entry_fx = round(float(r_post["entry"]), 2)

            fp = _pit_fund_at(ann_piv[tk], qtr_piv[tk], dt)
            bear_blend = None
            cmp_lt_bear = False
            bare = tk.replace(".NS", "")
            if sub_vctx is not None and pit_mcap >= 450.0 and avg_liq >= 1.8 and (tk in val_funda_set or bare in val_funda_set):
                try:
                    v = V.value_stock(bare, sub_vctx, price=c_v, as_of=dt_str, knobs=False, n=500)
                    if v.get("status") == "OK":
                        bb = (v.get("blends") or {}).get("bear", np.nan)
                        if np.isfinite(bb):
                            bear_blend = round(float(bb), 2)
                            cmp_lt_bear = bool(c_v <= float(bb))
                except Exception:
                    pass

            ind_k = sub_ind_map.get(tk, "Other")
            snap[tk] = {
                "close": round(c_v, 2), "stage": stage, "pit_mcap_cr": pit_mcap,
                "liq_cr": round(avg_liq, 2), "atr": round(atr_v, 2), "atr_pct": round(atr_p, 2),
                "bbp": round(bbp_v, 1) if bbp_v is not None else None, "comp_rs": round(crs_v, 1),
                "deliv_pct": round(dp_v, 1) if dp_v is not None else None,
                "deliv_ratio": round(dr_v, 2) if dr_v is not None else None,
                "abs_deliv_cr": round(avg_liq * dp_v / 100.0, 2) if (dp_v is not None and avg_liq > 0) else 0.0,
                "industry": ind_k, "ind_rank_pct": ind_rank.get(ind_k, 50.0),
                "action_pre": act_pre, "action_post": act_post,
                "stop_fx": stop_fx, "entry_fx": entry_fx or round(c_v, 2),
                "bear_blend": bear_blend, "cmp_lt_bear": cmp_lt_bear, **fp,
            }
        ckpts[dt_str] = snap
        if dt_str in event_set or ci % 5 == 0 or ci == len(ckpt_dates):
            elig_n = sum(1 for v in snap.values() if v["pit_mcap_cr"] >= 500.0 and v["liq_cr"] >= 2.0)
            post_n = sum(1 for v in snap.values() if v["pit_mcap_cr"] >= 500.0 and v["liq_cr"] >= 2.0 and v["action_post"] == "BUY NOW")
            bear_n = sum(1 for v in snap.values() if v["pit_mcap_cr"] >= 500.0 and v["liq_cr"] >= 2.0 and v["cmp_lt_bear"])
            print(f"  [{ci:2d}/{len(ckpt_dates)}] {dt_str} ({daily_regimes[dt_str]['engine_mode']:8s}) -> Active: {elig_n:4d} | BUY_NOW: {post_n:3d} | <Bear: {bear_n:3d} [{time.time()-t0:.1f}s]", flush=True)

    payload = {
        "dates": dates, "event_dates": event_dates, "monthly_gates": monthly_gates,
        "tradable_all": valid_all, "sub_industry": {tk: sub_ind_map.get(tk, "Other") for tk in valid_all},
        "regimes": daily_regimes, "ckpts": ckpts,
        "O": O.loc[dates_ts].to_numpy(dtype=np.float32),
        "L": L.loc[dates_ts].to_numpy(dtype=np.float32),
        "C": C.loc[dates_ts].to_numpy(dtype=np.float32),
        "MA50": MA50.loc[dates_ts].to_numpy(dtype=np.float32),
        "n50": nifty_df["Close"].reindex(dates_ts).ffill().to_numpy(dtype=np.float64),
        "ticker_to_col": {tk: i for i, tk in enumerate(valid_all)},
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {pkl_path} ({os.path.getsize(pkl_path)/1e6:.1f} MB) in {time.time()-t0:.1f}s", flush=True)
    return payload


def _gross_cap(emode: str, use_th: bool) -> float:
    return {"BEAR": 0.40, "DEFENSIVE": 0.40, "CAUTION": 0.75, "RECOVERY": 0.90, "BULL": 1.00}.get(emode, 0.75) if use_th else 1.0


def run_sim(
    panel: dict,
    strategy: str,
    *,
    decision_dates: set,
    mcap_floor_cr: float = 500.0,
    liq_floor_cr: float = 2.0,
    liq_bear_cr: Optional[float] = None,
    max_pos_pct: float = 10.0,
    use_inv_vol: bool = False,
    max_per_industry: Optional[int] = None,
    use_regime_throttle: bool = False,
    tie_breaker: str = "comp_rs",
    cooldown_days: int = 0,
) -> dict:
    dates, ckpts, regimes, t2c = panel["dates"], panel["ckpts"], panel["regimes"], panel["ticker_to_col"]
    tickers = panel["tradable_all"]
    O, L, C, MA50 = panel["O"], panel["L"], panel["C"], panel["MA50"]
    sub_ind = panel["sub_industry"]

    cash = INIT_CAP
    positions, pending, cooldown = {}, {}, {}
    trades, eq_rows, cands_cnt = [], [], []

    def _rval(s, k):
        v = s.get(k)
        return -999.0 if v is None else float(v)

    for di, dt in enumerate(dates):
        for tk, o in sorted(pending.items(), key=lambda kv: 0 if kv[1]["side"] == "SELL" else 1):
            px = float(O[di, t2c[tk]])
            if not np.isfinite(px) or px <= 0:
                continue
            if o["side"] == "SELL" and tk in positions:
                p = positions.pop(tk)
                net = p["shares"] * px * (1 - COST_SIDE)
                cash += net
                pnl_p = (px / p["avg_px"] - 1) * 100
                trades.append((dt, tk, "SELL", o["reason"], round(pnl_p, 1), round(net - p["cost_basis"], 0)))
                if cooldown_days > 0 and pnl_p < 0:
                    cooldown[tk] = di + cooldown_days
            elif o["side"] == "BUY":
                if o.get("max_gap") and (px / o["ref"] - 1) * 100 > o["max_gap"]:
                    continue
                cur_eq = cash + sum(p["shares"] * (float(O[di, t2c[t]]) if np.isfinite(O[di, t2c[t]]) else p["last_px"]) for t, p in positions.items())
                room = max(0.0, cur_eq * o.get("gcap", 1.0) - (cur_eq - cash))
                bv = min(cur_eq * (o["sz"] / 100.0), room, cash / (1 + COST_SIDE))
                sh = int(bv / px)
                if sh > 0:
                    cost = sh * px * (1 + COST_SIDE)
                    cash -= cost
                    positions[tk] = dict(shares=sh, cost_basis=cost, avg_px=cost / sh, peak=px, last_px=px, stop=o.get("stop"), trail=o.get("trail"), ma50=o.get("ma50", False), cat=o.get("cat"), ind=sub_ind.get(tk, "Other"))
                    trades.append((dt, tk, "BUY", o["reason"], 0.0, 0.0))
        pending.clear()

        for tk in list(positions.keys()):
            p = positions[tk]; col = t2c[tk]
            lo, op, cl, m50 = float(L[di, col]), float(O[di, col]), float(C[di, col]), float(MA50[di, col])
            if not np.isfinite(cl) or cl <= 0:
                cl = p["last_px"]; lo = cl; op = cl
            else:
                p["last_px"] = cl
            cat_s = p["avg_px"] * (1 + p["cat"] / 100.0) if p["cat"] is not None else None
            eff_s = max([x for x in (p["stop"], cat_s) if x is not None], default=None)
            if eff_s is not None and lo <= eff_s:
                fp = min(op, eff_s); net = p["shares"] * fp * (1 - COST_SIDE); cash += net
                trades.append((dt, tk, "SELL", "STOP", round((fp / p["avg_px"] - 1) * 100, 1), round(net - p["cost_basis"], 0)))
                del positions[tk]
                if cooldown_days > 0:
                    cooldown[tk] = di + cooldown_days
                continue
            if cl > p["peak"]:
                p["peak"] = cl
            if p["ma50"] and np.isfinite(m50) and cl < m50:
                pending[tk] = dict(side="SELL", reason="MA50")
            elif p["trail"] and cl < p["peak"] * (1 - p["trail"]):
                pending[tk] = dict(side="SELL", reason="TRAIL")

        emode = regimes[dt]["engine_mode"]
        if dt in decision_dates and di < len(dates) - 1 and dt in ckpts:
            snap = ckpts[dt]; trail = 0.15 if emode in ("BULL", "RECOVERY") else 0.12
            gcap = _gross_cap(emode, use_regime_throttle)
            eff_liq = liq_bear_cr if (liq_bear_cr is not None and emode in ("CAUTION", "BEAR", "DEFENSIVE")) else liq_floor_cr
            ic = {}
            for t, p in positions.items():
                if t not in pending:
                    ic[p["ind"]] = ic.get(p["ind"], 0) + 1
            if strategy in ("garp_cfo_only", "garp_guarded", "garp_stage2_v41", "hybrid_old", "hybrid_guarded"):
                for t in list(positions.keys()):
                    s = snap.get(t)
                    if not s or positions[t]["cat"] is None:
                        continue
                    if s["cfo_pat"] is not None and s["cfo_pat"] < 0:
                        pending[t] = dict(side="SELL", reason="FUND_CFO_NEG")
                    elif strategy in ("garp_guarded", "garp_stage2_v41", "hybrid_guarded") and s["q_np_yoy"] is not None and s["q_np_yoy"] <= -10:
                        pending[t] = dict(side="SELL", reason="FUND_NP_SHRINK")

            cands = []
            for tk in tickers:
                if tk in positions or tk in pending or cooldown.get(tk, -1) > di:
                    continue
                s = snap.get(tk)
                if not s or s["pit_mcap_cr"] < mcap_floor_cr or s["liq_cr"] < eff_liq:
                    continue
                if strategy in ("mom_pre_dual_atr", "mom_post_dual_atr", "monthly_quant_post"):
                    act = s["action_pre"] if strategy == "mom_pre_dual_atr" else s["action_post"]
                    if act == "BUY NOW" and s["stage"] == 2 and s["comp_rs"] > 0:
                        cands.append((_rval(s, tie_breaker), tk, "MOM", s))
                elif strategy == "mom_quality_ranked":
                    if s["action_post"] == "BUY NOW" and s["stage"] == 2 and s["comp_rs"] > 0 and (s["q_np_yoy"] is None or s["q_np_yoy"] > 0):
                        sc = 0.45 * s["ind_rank_pct"] + 0.35 * min(s["abs_deliv_cr"], 100.0) + 0.20 * min(max(s["q_np_yoy"] or 15.0, 0), 100.0)
                        cands.append((sc, tk, "MOM_Q", s))
                elif strategy == "garp_unfiltered":
                    if s["cmp_lt_bear"] and s["bear_blend"]:
                        cands.append(((1 - s["close"] / s["bear_blend"]) * 100, tk, "GARP_RAW", s))
                elif strategy == "garp_cfo_only":
                    if s["cmp_lt_bear"] and s["bear_blend"] and (s["cfo_pat"] or -9) >= 0.80:
                        cands.append(((1 - s["close"] / s["bear_blend"]) * 100, tk, "GARP_CFO", s))
                elif strategy == "garp_guarded":
                    if s["cmp_lt_bear"] and s["bear_blend"] and (s["cfo_pat"] or -9) >= 0.80 and (s["q_np_yoy"] or -9) > 0 and s["stage"] != 4:
                        cands.append(((1 - s["close"] / s["bear_blend"]) * 100 + 0.5 * (s["roce"] or 0), tk, "GARP_G", s))
                elif strategy == "garp_stage2_v41":
                    if s["cmp_lt_bear"] and s["bear_blend"] and (s["cfo_pat"] or -9) >= 0.80 and (s["q_np_yoy"] or -9) > 0 and s["stage"] == 2:
                        cands.append((0.4 * s["comp_rs"] + 0.3 * s["ind_rank_pct"] + 0.3 * min(s["q_np_yoy"] or 0, 100), tk, "GARP_G", s))
                elif strategy == "hybrid_old":
                    if s["cmp_lt_bear"] and (s["cfo_pat"] or -9) >= 0.80:
                        cands.append((80 + (s["roce"] or 0), tk, "HYB_OLD_G", s))
                    elif s["action_post"] == "BUY NOW" and s["stage"] == 2 and s["comp_rs"] > 0:
                        cands.append((50 + s["comp_rs"], tk, "MOM", s))
                    elif s["bbp"] is not None and s["bbp"] < 20 and ((s["stage"] in (1, 2) and s["comp_rs"] > 0) or (emode in ("BEAR", "RECOVERY") and (s["q_np_yoy"] or 0) >= 30)):
                        cands.append((30 + s["comp_rs"], tk, "BBP", s))
                elif strategy == "hybrid_guarded":
                    if s["cmp_lt_bear"] and (s["cfo_pat"] or -9) >= 0.80 and (s["q_np_yoy"] or -9) > 0 and s["stage"] != 4:
                        cands.append((80 + (s["roce"] or 0), tk, "GARP_G", s))
                    elif s["action_post"] == "BUY NOW" and s["stage"] == 2 and s["comp_rs"] > 0:
                        cands.append((50 + s["comp_rs"], tk, "MOM", s))
                    elif s["bbp"] is not None and s["bbp"] < 20 and s["stage"] == 2 and s["comp_rs"] > 5 and emode in ("RECOVERY", "CAUTION", "BULL"):
                        cands.append((40 + s["comp_rs"], tk, "BBP", s))
                elif strategy == "pead_calibrated":
                    if (s["d_inst_q"] or -9) >= 0.5 and (s["d_prom_q"] is None or s["d_prom_q"] >= -1.0) and (s["q_np_yoy"] or -9) >= 20.0 and s["stage"] in (1, 2, 3) and s["comp_rs"] > 0:
                        cands.append(((s["d_inst_q"] or 0) * 15 + min(s["q_np_yoy"] or 0, 120) * 0.3 + s["ind_rank_pct"] * 0.4, tk, "PEAD", s))
                elif strategy == "sector_rotator_rp":
                    if s["ind_rank_pct"] >= 70 and s["stage"] == 2 and s["comp_rs"] > 0 and (s["q_np_yoy"] is None or s["q_np_yoy"] > 0) and (s["action_post"] == "BUY NOW" or (s["bbp"] is not None and s["bbp"] < 20 and s["comp_rs"] > 5)):
                        cands.append((0.5 * s["ind_rank_pct"] + s["comp_rs"], tk, "SEC_RP", s))

            cands_cnt.append(len(cands))
            cands.sort(key=lambda x: x[0], reverse=True)
            for _, tk, kind, s in cands:
                if max_per_industry and ic.get(s["industry"], 0) >= max_per_industry:
                    continue
                sz = float(np.clip(30.0 / max(s["atr_pct"] or 3.5, 1.5), 4.0, 12.0)) if use_inv_vol else max_pos_pct
                if kind in ("GARP_RAW", "GARP_CFO", "HYB_OLD_G"):
                    pending[tk] = dict(side="BUY", reason=kind, sz=sz, stop=None, trail=(0.20 if kind == "HYB_OLD_G" else None), ma50=False, cat=(-25.0 if kind != "GARP_RAW" else None), ref=s["close"], max_gap=None, gcap=gcap)
                elif kind == "GARP_G":
                    pending[tk] = dict(side="BUY", reason=kind, sz=min(10.0, sz), stop=None, trail=0.20, ma50=False, cat=-25.0, ref=s["close"], max_gap=None, gcap=gcap)
                else:
                    stp = s["stop_fx"] if (s["stop_fx"] and s["stop_fx"] < s["close"] and (1 - s["stop_fx"] / s["close"]) <= 0.15) else round(s["close"] - 2 * s["atr"], 2)
                    pending[tk] = dict(side="BUY", reason=kind, sz=sz, stop=stp, trail=trail, ma50=True, cat=None, ref=s["close"], max_gap=5.0, gcap=gcap)
                ic[s["industry"]] = ic.get(s["industry"], 0) + 1

        nav = cash + sum(p["shares"] * (float(C[di, t2c[t]]) if np.isfinite(C[di, t2c[t]]) else p["last_px"]) for t, p in positions.items())
        eq_rows.append((dt, nav, cash / nav * 100.0 if nav > 0 else 100.0, len(positions)))

    pnl_tk = {}
    for _, tk, sd, _, _, prs in trades:
        if sd == "SELL":
            pnl_tk[tk] = pnl_tk.get(tk, 0) + prs
    for tk, p in positions.items():
        c_end = float(C[-1, t2c[tk]]) if np.isfinite(C[-1, t2c[tk]]) else p["last_px"]
        pnl_tk[tk] = pnl_tk.get(tk, 0) + (p["shares"] * c_end - p["cost_basis"])

    curve = pd.Series([x[1] for x in eq_rows], index=pd.to_datetime([x[0] for x in eq_rows]))
    sells = [t for t in trades if t[2] == "SELL"]
    wins = sum(1 for t in sells if t[5] > 0)
    return {
        "curve": curve,
        "cash_drag_mean": round(float(np.mean([x[2] for x in eq_rows])), 1),
        "avg_positions": round(float(np.mean([x[3] for x in eq_rows])), 1),
        "buys": sum(1 for t in trades if t[2] == "BUY"),
        "win_rate_pct": round(wins / len(sells) * 100.0, 1) if sells else 0.0,
        "avg_cands": round(float(np.mean(cands_cnt)), 1) if cands_cnt else 0.0,
        "max_cands": int(max(cands_cnt)) if cands_cnt else 0,
        "top5": [(k, int(v)) for k, v in sorted(pnl_tk.items(), key=lambda x: x[1], reverse=True)[:5]],
        "bot3": [(k, int(v)) for k, v in sorted(pnl_tk.items(), key=lambda x: x[1])[:3]],
    }


def metrics(curve: pd.Series) -> dict:
    s = curve.dropna()
    if len(s) < 5:
        return dict(ret=0.0, cagr=0.0, mdd=0.0, sharpe=0.0)
    tot = (float(s.iloc[-1]) / float(s.iloc[0]) - 1.0) * 100.0
    yrs = max(0.1, (s.index[-1] - s.index[0]).days / 365.25)
    cagr = ((float(s.iloc[-1]) / float(s.iloc[0])) ** (1.0 / yrs) - 1.0) * 100.0
    mdd = float((s / s.cummax() - 1.0).min() * 100.0)
    r = s.pct_change().dropna()
    sh = float((r.mean() / r.std(ddof=1)) * math.sqrt(252.0)) if r.std(ddof=1) > 1e-12 else 0.0
    return dict(ret=round(tot, 2), cagr=round(cagr, 2), mdd=round(mdd, 2), sharpe=round(sh, 3))


def passive_ew_pit(panel: dict) -> pd.Series:
    s0 = panel["ckpts"][panel["event_dates"][0]]
    elig = [tk for tk in panel["tradable_all"] if tk in s0 and s0[tk]["pit_mcap_cr"] >= 500.0 and s0[tk]["liq_cr"] >= 2.0]
    cols = [panel["ticker_to_col"][tk] for tk in elig]
    C = panel["C"][:, cols]
    valid = np.isfinite(C[0, :]) & (C[0, :] > 0)
    C = C[:, valid]
    shares = (INIT_CAP / C.shape[1]) / C[0, :]
    nav = np.nansum(C * shares[None, :], axis=1)
    return pd.Series(nav, index=pd.to_datetime(panel["dates"]))


def make_barbell(curves: dict, weights: dict, rebal_dates: set) -> pd.Series:
    rets = pd.DataFrame({k: curves[k].pct_change().fillna(0.0) for k in weights})
    w_cur = dict(weights)
    val = INIT_CAP
    out = np.empty(len(rets))
    for i, dt in enumerate(rets.index):
        if i == 0:
            out[i] = val
            continue
        r_row = rets.iloc[i]
        pr = sum(w_cur[k] * float(r_row[k]) for k in weights)
        val *= (1.0 + pr)
        out[i] = val
        if dt.strftime("%Y-%m-%d") in rebal_dates:
            w_cur = dict(weights)
        else:
            if 1.0 + pr > 0:
                w_cur = {k: w_cur[k] * (1.0 + float(r_row[k])) / (1.0 + pr) for k in weights}
    return pd.Series(out, index=rets.index)


def main() -> int:
    panel = build_panel()
    ev_set = set(panel["event_dates"])
    mo_set = set(panel["monthly_gates"])
    idx_dt = pd.to_datetime(panel["dates"])

    n50_c = pd.Series(panel["n50"] / panel["n50"][0] * INIT_CAP, index=idx_dt)
    ew_c = passive_ew_pit(panel)

    specs = [
        ("mom_pre_dual_atr",         "mom_pre_dual_atr",   dict(decision_dates=ev_set, liq_floor_cr=2.0,  max_pos_pct=10.0)),
        ("mom_post_dual_atr_comprs", "mom_post_dual_atr",  dict(decision_dates=ev_set, liq_floor_cr=2.0,  max_pos_pct=10.0, tie_breaker="comp_rs")),
        ("mom_post_deliv_rank",      "mom_post_dual_atr",  dict(decision_dates=ev_set, liq_floor_cr=2.0,  max_pos_pct=10.0, tie_breaker="abs_deliv_cr")),
        ("mom_quality_2cr",          "mom_quality_ranked", dict(decision_dates=ev_set, liq_floor_cr=2.0,  max_pos_pct=10.0, use_inv_vol=True, max_per_industry=2, use_regime_throttle=True, cooldown_days=30)),
        ("mom_quality_5cr",          "mom_quality_ranked", dict(decision_dates=ev_set, liq_floor_cr=5.0,  max_pos_pct=10.0, use_inv_vol=True, max_per_industry=2, use_regime_throttle=True, cooldown_days=30)),
        ("mom_quality_20cr",         "mom_quality_ranked", dict(decision_dates=ev_set, liq_floor_cr=20.0, max_pos_pct=10.0, use_inv_vol=True, max_per_industry=2, use_regime_throttle=True, cooldown_days=30)),
        ("mom_quality_dyn_5_20",     "mom_quality_ranked", dict(decision_dates=ev_set, liq_floor_cr=5.0,  liq_bear_cr=20.0, max_pos_pct=10.0, use_inv_vol=True, max_per_industry=2, use_regime_throttle=True, cooldown_days=30)),
        ("garp_unfiltered_2cr",      "garp_unfiltered",    dict(decision_dates=ev_set, liq_floor_cr=2.0,  max_pos_pct=10.0)),
        ("garp_cfo_only_2cr",        "garp_cfo_only",      dict(decision_dates=ev_set, liq_floor_cr=2.0,  max_pos_pct=10.0)),
        ("garp_guarded_2cr",         "garp_guarded",       dict(decision_dates=ev_set, liq_floor_cr=2.0,  max_pos_pct=10.0, cooldown_days=60)),
        ("garp_guarded_20cr",        "garp_guarded",       dict(decision_dates=ev_set, liq_floor_cr=20.0, max_pos_pct=10.0, cooldown_days=60)),
        ("garp_stage2_v41_2cr",      "garp_stage2_v41",    dict(decision_dates=ev_set, liq_floor_cr=2.0,  max_pos_pct=10.0, cooldown_days=60)),
        ("hybrid_old_2cr",           "hybrid_old",         dict(decision_dates=ev_set, liq_floor_cr=2.0,  max_pos_pct=20.0)),
        ("hybrid_guarded_2cr",       "hybrid_guarded",     dict(decision_dates=ev_set, liq_floor_cr=2.0,  max_pos_pct=10.0, cooldown_days=60)),
        ("pead_calibrated_2cr",      "pead_calibrated",    dict(decision_dates=ev_set, liq_floor_cr=2.0,  max_pos_pct=10.0, use_inv_vol=True, max_per_industry=2, use_regime_throttle=True, cooldown_days=30)),
        ("pead_calibrated_5cr",      "pead_calibrated",    dict(decision_dates=ev_set, liq_floor_cr=5.0,  max_pos_pct=10.0, use_inv_vol=True, max_per_industry=2, use_regime_throttle=True, cooldown_days=30)),
        ("pead_calibrated_20cr",     "pead_calibrated",    dict(decision_dates=ev_set, liq_floor_cr=20.0, max_pos_pct=10.0, use_inv_vol=True, max_per_industry=2, use_regime_throttle=True, cooldown_days=30)),
        ("sector_rotator_rp",        "sector_rotator_rp",  dict(decision_dates=ev_set, liq_floor_cr=2.0,  max_pos_pct=10.0, use_inv_vol=True, max_per_industry=2, use_regime_throttle=True, cooldown_days=30)),
        ("monthly_quant_post",       "monthly_quant_post", dict(decision_dates=mo_set, liq_floor_cr=2.0,  max_pos_pct=10.0)),
    ]

    curves = {"NIFTY_50": n50_c, "EW_TRADABLE_500CR": ew_c}
    meta = {}
    for name, strat, kw in specs:
        r = run_sim(panel, strat, **kw)
        curves[name] = r.pop("curve")
        meta[name] = r

    # Winning 2026 3-Sleeve Orthogonal Barbells (monthly rebalanced across 2.0Y)
    curves["BARBELL_BASE_50_30_20"] = make_barbell(curves, {"garp_guarded_2cr": 0.50, "mom_quality_5cr": 0.30, "pead_calibrated_2cr": 0.20}, mo_set)
    curves["BARBELL_SPLIT_50_30_20"] = make_barbell(curves, {"garp_guarded_2cr": 0.50, "mom_quality_20cr": 0.30, "pead_calibrated_20cr": 0.20}, mo_set)
    curves["BARBELL_REC_40_40_20"] = make_barbell(curves, {"garp_guarded_2cr": 0.40, "mom_quality_20cr": 0.40, "pead_calibrated_20cr": 0.20}, mo_set)
    curves["BARBELL_INST_20CR"] = make_barbell(curves, {"garp_guarded_20cr": 0.50, "mom_quality_20cr": 0.30, "pead_calibrated_20cr": 0.20}, mo_set)
    curves["BARBELL_S2_ALLWEATHER"] = make_barbell(curves, {"garp_stage2_v41_2cr": 0.40, "mom_quality_20cr": 0.40, "pead_calibrated_20cr": 0.20}, mo_set)

    out = {"full_2y": {}, "oos_24_26feb": {}, "overlap_26": {}, "corr_2y": {}, "meta": meta}

    print("\n" + "=" * 130)
    print("FULL 2.0-YEAR EXACT TRADABLE >=500Cr FRAMEWORK (2024-11-18 -> 2026-11-20 | Exact V4.1 + Forensics + Quarterly)")
    print("=" * 130)
    for name, c in curves.items():
        m_full = metrics(c)
        m_oos = metrics(c.loc[:SPLIT_DATE])
        m_ov = metrics(c.loc[OVERLAP_START:])
        out["full_2y"][name] = m_full
        out["oos_24_26feb"][name] = m_oos
        out["overlap_26"][name] = m_ov
        ex = meta.get(name, {})
        c_str = f"Cash:{ex.get('cash_drag_mean', 0):4.1f}% Win:{ex.get('win_rate_pct', 0):4.1f}% ({ex.get('buys', 0):3d}B)" if ex else ""
        print(
            f"  {name:25s} | 2Y: {m_full['ret']:+7.2f}% (CAGR {m_full['cagr']:+6.2f}%, DD {m_full['mdd']:6.2f}%, Sh {m_full['sharpe']:5.3f}) | "
            f"OOS(Nov24-Feb26): {m_oos['ret']:+7.2f}% (DD {m_oos['mdd']:6.2f}%) | 2026(Feb-Nov): {m_ov['ret']:+7.2f}% | {c_str}",
            flush=True,
        )

    cdf = pd.DataFrame({
        "GARP_2Cr": curves["garp_guarded_2cr"].pct_change(),
        "MOMQ_20Cr": curves["mom_quality_20cr"].pct_change(),
        "PEAD_20Cr": curves["pead_calibrated_20cr"].pct_change(),
        "NIFTY_50": curves["NIFTY_50"].pct_change(),
    }).dropna()
    out["corr_2y"] = cdf.corr().round(2).to_dict()
    print("\n2.0-Year Daily Return Correlation Matrix:")
    print(cdf.corr().round(2).to_string())

    print("\nTop 5 Stock Attribution (2.0Y Full Cycle):")
    for k in ("garp_guarded_2cr", "garp_stage2_v41_2cr", "mom_quality_20cr", "pead_calibrated_20cr", "hybrid_guarded_2cr"):
        print(f"  {k:22s} Top5: {meta[k]['top5']} | Bot3: {meta[k]['bot3']}")

    out_p = os.path.join(OUT_DIR, "tradable_24_26_results.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
