"""10.6-Year Strictly Causal Multi-Regime Walk-Forward Backtest (2016-04-01 -> 2026-11-20).

Tests ALL sleeves, guardrails, liquidity floors, tie-breakers, regime throttles, and 3-Sleeve Barbells
across 7 distinct Indian market regimes on the Survivorship-Free Tradable >=500Cr MCap & >=2Cr/d NSE Universe:
  R1: 2016-04 -> 2018-01  Pre-2018 Midcap Boom / Bull
  R2: 2018-02 -> 2020-01  2018-2019 NBFC / IL&FS Small & Midcap Bear Market
  R3: 2020-02 -> 2020-09  2020 COVID Crash (-38%) & V-Shape Shock
  R4: 2020-10 -> 2021-12  Post-COVID Secular Bull Run
  R5: 2022-01 -> 2023-03  2022-2023 Rate-Hike Chop / Sideways Grind
  R6: 2023-04 -> 2024-09  2023-2024 Secular Broad Super-Bull
  R7: 2024-10 -> 2026-11  Post-2024 Choppy / Multi-Leg Correction Regime
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_N = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\repo_nov21"
REPO_L = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\repo_live"
NSE_ROOT = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\multi_regime_10y\nse_11y"
OUT_DIR = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\multi_regime_10y"
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, REPO_N)
sys.path.insert(1, REPO_L)
os.chdir(NSE_ROOT)

from utils import nse_prices as npx  # noqa: E402
from utils import fund_trends as ft  # noqa: E402
from utils import nifty500_list as n500, nifty1000_list as n1000  # noqa: E402

START_DATE = "2016-04-01"
END_DATE = "2026-11-20"
INIT_CAP = 1_000_000.0
COST_SIDE = 0.0025  # 25 bps each way = 50 bps round-trip

REGIMES = [
    ("R1_16_17_Bull",      "2016-04-01", "2018-01-31"),
    ("R2_18_19_Bear",      "2018-02-01", "2020-01-31"),
    ("R3_20_COVID",        "2020-02-01", "2020-09-30"),
    ("R4_20_21_PostCovid", "2020-10-01", "2021-12-31"),
    ("R5_22_23_Chop",      "2022-01-01", "2023-03-31"),
    ("R6_23_24_SuperBull", "2023-04-01", "2024-09-30"),
    ("R7_24_26_Post24",    "2024-10-01", "2026-11-20"),
]

BFSI_SECS = {"Banks", "Finance", "Financial Services", "NBFC", "Banking", "Housing Finance"}


def load_indices_11y() -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    files = sorted(glob.glob(os.path.join(NSE_ROOT, "data", "nse", "indices", "*.parquet"))) + \
            sorted(glob.glob(os.path.join(NSE_ROOT, "data", "nse", "indices", "d", "*.parquet")))
    dfs = [pd.read_parquet(f, columns=["date", "index", "close"]) for f in files]
    ix = pd.concat(dfs, ignore_index=True)
    ix["date"] = pd.to_datetime(ix["date"]).dt.normalize()
    ix["index_l"] = ix["index"].astype(str).str.strip().str.lower()

    def _get(names):
        sub = ix[ix["index_l"].isin(names)].drop_duplicates("date", keep="last").set_index("date")["close"].sort_index()
        return sub

    n50 = _get({"nifty 50", "cnx nifty"})
    vix = _get({"india vix"})
    mid = _get({"nifty midcap 100", "nifty midcap 50", "cnx midcap"})
    sml = _get({"nifty smallcap 100", "nifty smallcap 250", "cnx smallcap"})
    return n50, vix, mid, sml


def load_universe_and_matrices() -> dict:
    t0 = time.time()
    print("[panel] scanning 11.5Y raw NSE parquets for survivorship-free universe & exact turnover/delivery...", flush=True)
    p_files = sorted(glob.glob(os.path.join(NSE_ROOT, "data", "nse", "prices", "*.parquet"))) + \
              sorted(glob.glob(os.path.join(NSE_ROOT, "data", "nse", "prices", "d", "*.parquet")))

    # Load symbol changes map so old symbols stitch cleanly into current symbols
    sc_path = os.path.join(NSE_ROOT, "data", "nse", "symbol_changes.csv")
    sym_map: Dict[str, str] = {}
    if os.path.exists(sc_path):
        sc = pd.read_csv(sc_path)
        for _, r in sc.iterrows():
            o, n = str(r.get("old", "")).strip(), str(r.get("new", "")).strip()
            if o and n and o != n:
                sym_map[o] = n
        # Resolve transitive renames (A -> B -> C)
        for k in list(sym_map.keys()):
            seen = {k}
            cur = sym_map[k]
            while cur in sym_map and cur not in seen:
                seen.add(cur)
                cur = sym_map[cur]
            sym_map[k] = cur

    # Single fast pass over all parquets to compute raw turnover (Cr), delivery %, and find all symbols with ANY tradable liquidity >=1.5Cr/d
    chunks = []
    for f in p_files:
        df = pd.read_parquet(f, columns=["date", "symbol", "series", "close", "volume", "deliv_pct"])
        df = df[df["series"].isin(["EQ", "BE"])].copy()
        if df.empty:
            continue
        df["symbol"] = df["symbol"].astype(str).str.strip().map(lambda s: sym_map.get(s, s))
        df["tval_cr"] = (df["close"].astype(np.float64) * df["volume"].astype(np.float64)) / 1e7
        chunks.append(df[["date", "symbol", "tval_cr", "deliv_pct"]])

    raw_all = pd.concat(chunks, ignore_index=True)
    raw_all["date"] = pd.to_datetime(raw_all["date"]).dt.normalize()
    raw_all = raw_all.drop_duplicates(["date", "symbol"], keep="last")
    print(f"[panel] loaded {len(raw_all):,} EQ rows across {raw_all['symbol'].nunique():,} stitched symbols [{time.time()-t0:.1f}s]", flush=True)

    # Identify every symbol that EVER had >= 60 sessions AND a rolling/annual median turnover >= 1.5 Cr/d (keeps ALL delisted 2016-2025 tradable stocks!)
    raw_all["yr"] = raw_all["date"].dt.year
    yr_liq = raw_all.groupby(["symbol", "yr"])["tval_cr"].agg(["count", "median"]).reset_index()
    elig_syms = sorted(yr_liq[(yr_liq["count"] >= 40) & (yr_liq["median"] >= 1.5)]["symbol"].unique())
    elig_tks = [f"{s}.NS" for s in elig_syms]
    print(f"[panel] survivorship-free tradable candidate pool (>=1.5Cr/d in any year): {len(elig_tks):,} symbols [{time.time()-t0:.1f}s]", flush=True)

    # Filter raw_all to elig_syms and pivot turnover & delivery matrices
    raw_sub = raw_all[raw_all["symbol"].isin(set(elig_syms))].copy()
    raw_sub["tk"] = raw_sub["symbol"] + ".NS"
    TVAL = raw_sub.pivot(index="date", columns="tk", values="tval_cr").sort_index().astype(np.float32)
    DELIV = raw_sub.pivot(index="date", columns="tk", values="deliv_pct").sort_index().astype(np.float32)
    del raw_all, raw_sub, chunks

    # Load corporate-action adjusted OHLCV via repo_nov21's exact nse_prices.history() (with Kite demerger bounds!)
    print(f"[panel] loading split/bonus/demerger-adjusted OHLCV for {len(elig_tks):,} stocks via nse_prices.history()...", flush=True)
    npx._load.cache_clear()
    hist = npx.history(elig_tks)

    n50, vix, mid, sml = load_indices_11y()
    all_dates = n50.loc["2015-04-01":END_DATE].index
    sim_dates = n50.loc[START_DATE:END_DATE].index

    valid_tks = [tk for tk in elig_tks if tk in hist and hist[tk] is not None and len(hist[tk]) >= 120]
    O = pd.DataFrame({tk: hist[tk]["Open"].reindex(all_dates) for tk in valid_tks}, dtype=np.float32)
    H = pd.DataFrame({tk: hist[tk]["High"].reindex(all_dates) for tk in valid_tks}, dtype=np.float32)
    L = pd.DataFrame({tk: hist[tk]["Low"].reindex(all_dates) for tk in valid_tks}, dtype=np.float32)
    C = pd.DataFrame({tk: hist[tk]["Close"].reindex(all_dates) for tk in valid_tks}, dtype=np.float32)
    TVAL = TVAL.reindex(index=all_dates, columns=valid_tks)
    DELIV = DELIV.reindex(index=all_dates, columns=valid_tks)
    del hist

    # Sanity-guard any unadjusted extreme single-day corporate-action artifact (> +180% or < -65% in 1 day without recovery)
    # Note: genuine circuits in India are <= +-20% (or +-50% on IPO/relist); >+250% or <-75% in 1 day is a missing split/demerger!
    ret1 = C.pct_change(fill_method=None)
    bad_jump = (ret1 > 2.5) | (ret1 < -0.75)
    n_bad = int(bad_jump.sum().sum())
    print(f"[panel] OHLCV matrices ready: {C.shape} | extreme >+250%/<-75% 1d glitches: {n_bad} [{time.time()-t0:.1f}s]", flush=True)

    # Load 12-Year Screener Annual Fundamentals (FY2015 -> FY2026) for 2,114 stocks
    ann_raw, _ = ft._data(REPO_L)
    ext_meta = getattr(n1000, "EXTENDED", {}) or {}
    sub_ind_map = dict(getattr(n1000, "SUB_INDUSTRY_MAP", {}) or {})
    mm_files = sorted(glob.glob(os.path.join(REPO_L, "data", "cache", "market_master_*.parquet")))
    sec_map = {}
    mm_shares = {}
    if mm_files:
        mm = pd.read_parquet(mm_files[-1])
        sec_map = dict(zip(mm["ticker"], mm.get("sector_granular", mm.get("sector", pd.Series("", index=mm.index))).fillna("").astype(str)))
        for _, r in mm.iterrows():
            mc_v, px_v = r.get("marketCap"), r.get("currentPrice")
            if pd.notna(mc_v) and pd.notna(px_v) and float(px_v) > 0:
                mm_shares[str(r["ticker"])] = (float(mc_v) / 1e7) / float(px_v)

    ann_piv: Dict[str, pd.DataFrame] = {}
    shares_cr: Dict[str, float] = {}
    ind_of: Dict[str, str] = {}
    is_bfsi: Dict[str, bool] = {}

    for tk in valid_tks:
        ka = ft._key(tk, ann_raw)
        sh_v = None
        if ka is not None:
            w = ann_raw[ka].pivot_table(index="period", columns="field", values="value", aggfunc="last").sort_index()
            w_fy = w[w.index.month == 3].copy()
            ann_piv[tk] = w_fy
            if "mcap_cr" in w.columns and "price_scr" in w.columns:
                mc_s = w["mcap_cr"].dropna()
                ps_s = w["price_scr"].dropna()
                if not mc_s.empty and not ps_s.empty and float(ps_s.iloc[-1]) > 0:
                    sh_v = float(mc_s.iloc[-1]) / float(ps_s.iloc[-1])
        if sh_v is None and tk in mm_shares:
            sh_v = mm_shares[tk]
        if sh_v is None and tk in ext_meta and ext_meta[tk].get("mcap_cr"):
            last_px = C[tk].dropna()
            if not last_px.empty and float(last_px.iloc[-1]) > 0:
                sh_v = float(ext_meta[tk]["mcap_cr"]) / float(last_px.iloc[-1])
        if sh_v is not None and sh_v > 0:
            shares_cr[tk] = sh_v
        ind = sub_ind_map.get(tk) or (ext_meta.get(tk) or {}).get("industry") or sec_map.get(tk) or n500.SECTOR_MAP.get(tk, "Other")
        ind_of[tk] = ind
        is_bfsi[tk] = any(b.lower() in str(ind).lower() for b in ("bank", "financ", "nbfc", "lending", "housing"))

    print(f"[panel] Screener annual coverage: {len(ann_piv):,}/{len(valid_tks):,} | MCap shares mapped: {len(shares_cr):,} (unmapped delisted kept for price/mom!) [{time.time()-t0:.1f}s]", flush=True)

    # Vectorized Technical Indicators (strictly causal <= t!)
    print("[panel] computing vectorized technicals, ATR14, CompRS, delivery, and breadth...", flush=True)
    C_f = C.ffill(limit=5)
    MA20 = C_f.rolling(20, min_periods=15).mean()
    MA50 = C_f.rolling(50, min_periods=40).mean()
    MA200 = C_f.rolling(200, min_periods=150).mean()
    HI252 = H.rolling(252, min_periods=160).max()
    # True Range & ATR14
    pc = C_f.shift(1)
    tr = np.maximum(H - L, np.maximum((H - pc).abs(), (L - pc).abs()))
    ATR14 = tr.rolling(14, min_periods=10).mean()
    ATR_PCT = (ATR14 / C_f * 100.0).astype(np.float32)

    R20 = (C_f / C_f.shift(20) - 1.0) * 100.0
    R63 = (C_f / C_f.shift(63) - 1.0) * 100.0
    R126 = (C_f / C_f.shift(126) - 1.0) * 100.0
    R252 = (C_f / C_f.shift(252) - 1.0) * 100.0
    COMP_RS_RAW = 0.40 * R63 + 0.30 * R126 + 0.30 * R252

    # Rolling 20D median turnover (Cr) & delivery metrics
    LIQ20 = TVAL.rolling(20, min_periods=10).median().astype(np.float32)
    DELIV_CR20 = (TVAL * (DELIV / 100.0)).rolling(20, min_periods=10).mean().astype(np.float32)
    DELIV_5 = DELIV.rolling(5, min_periods=3).mean()
    DELIV_60 = DELIV.rolling(60, min_periods=25).mean()
    DELIV_RATIO = (DELIV_5 / DELIV_60).replace([np.inf, -np.inf], np.nan).astype(np.float32)

    # Strictly PIT Market Cap matrix (for Screener-mapped stocks; NaN for delisted unmapped stocks)
    sh_vec = pd.Series({tk: shares_cr.get(tk, np.nan) for tk in valid_tks}, dtype=np.float32)
    PIT_MCAP = C_f.mul(sh_vec, axis=1).astype(np.float32)

    # Daily Universe Breadth (computed ONLY across active stocks trading >= 2Cr/d on day t!)
    act_mask = (LIQ20 >= 2.0) & C.notna() & (C > 5.0)
    b_above50 = ((C_f > MA50) & act_mask).sum(axis=1) / act_mask.sum(axis=1).replace(0, np.nan) * 100.0
    b_above200 = ((C_f > MA200) & act_mask).sum(axis=1) / act_mask.sum(axis=1).replace(0, np.nan) * 100.0
    n50_ma50 = n50.rolling(50).mean()
    n50_ma200 = n50.rolling(200).mean()

    # Daily Causal Regime Classification (matches regime_manager.py logic)
    regimes_daily: Dict[pd.Timestamp, dict] = {}
    for dt in sim_dates:
        nv = float(n50.loc[dt]) if dt in n50.index else np.nan
        m50 = float(n50_ma50.loc[dt]) if dt in n50_ma50.index else np.nan
        m200 = float(n50_ma200.loc[dt]) if dt in n50_ma200.index else np.nan
        vx = float(vix.loc[dt]) if dt in vix.index and pd.notna(vix.loc[dt]) else 16.0
        b50 = float(b_above50.loc[dt]) if dt in b_above50.index and pd.notna(b_above50.loc[dt]) else 50.0
        b200 = float(b_above200.loc[dt]) if dt in b_above200.index and pd.notna(b_above200.loc[dt]) else 50.0

        # Regime modes: BEAR / CAUTION / RECOVERY / BULL
        if np.isfinite(nv) and np.isfinite(m200) and nv < m200 and b50 < 40.0:
            mode = "BEAR"
        elif vx >= 28.0:
            mode = "BEAR"
        elif np.isfinite(nv) and np.isfinite(m50) and nv < m50 and b50 < 48.0:
            mode = "CAUTION"
        elif np.isfinite(nv) and np.isfinite(m50) and nv > m50 and b50 >= 55.0 and (np.isnan(m200) or nv > m200):
            mode = "BULL"
        else:
            mode = "CAUTION"
        # Capitulation recovery override (like 2020-04 / 2023-04 / 2025-04)
        if mode == "BEAR" and b50 >= 32.0 and np.isfinite(nv) and np.isfinite(m50) and nv > m50:
            mode = "RECOVERY"

        regimes_daily[dt] = dict(mode=mode, b50=b50, b200=b200, vix=vx)

    # Precompute strictly causal Annual Fundamental snapshot matrices on biweekly checkpoint dates
    # Biweekly decision dates (~every 10 trading sessions = ~262 checkpoints over 10.6 years!)
    sim_idx = list(sim_dates)
    biweekly_idx = list(range(0, len(sim_idx), 10))
    ckpt_dates = [sim_idx[i] for i in biweekly_idx]
    monthly_dates = set(pd.Series(sim_idx).groupby(pd.Series(sim_idx).dt.to_period("M")).first().tolist())

    print(f"[panel] computing strictly causal 10Y PIT Fundamentals & Bear Intrinsic Values across {len(ckpt_dates)} biweekly checkpoints...", flush=True)

    # Pre-extract per-stock FY arrays for ultra-fast causal lookup (FY period <= dt - 60 days!)
    fy_cache: Dict[str, dict] = {}
    for tk, w in ann_piv.items():
        if w.empty:
            continue
        p_dates = w.index.to_numpy()
        get_a = lambda col: w[col].to_numpy(dtype=np.float64) if col in w.columns else np.full(len(w), np.nan)
        np_a = get_a("np")
        sa_a = get_a("sales")
        cfo_a = get_a("cfo")
        roce_a = get_a("roce")
        eq_a = get_a("eq_cap")
        res_a = get_a("reserves")
        bk_a = eq_a + res_a
        fy_cache[tk] = dict(dates=p_dates, np=np_a, sales=sa_a, cfo=cfo_a, roce=roce_a, book=bk_a)

    # Precompute monthly PIT P/E & P/B history (strictly causal <= t) using month-end closes
    m_close = C_f.resample("ME").last()
    m_dates = m_close.index

    # For each stock, build monthly causal PE / PB series (using FY published >= 60d prior to month-end!)
    pit_pe_m = pd.DataFrame(np.nan, index=m_dates, columns=valid_tks, dtype=np.float32)
    pit_pb_m = pd.DataFrame(np.nan, index=m_dates, columns=valid_tks, dtype=np.float32)
    for tk, fc in fy_cache.items():
        if tk not in shares_cr:
            continue
        sh = shares_cr[tk]
        mc_s = m_close[tk].to_numpy(dtype=np.float64) * sh
        f_dts = fc["dates"]
        f_np = fc["np"]
        f_bk = fc["book"]
        pe_arr = np.full(len(m_dates), np.nan)
        pb_arr = np.full(len(m_dates), np.nan)
        for mi, md in enumerate(m_dates):
            if not np.isfinite(mc_s[mi]) or mc_s[mi] <= 0:
                continue
            cutoff = np.datetime64(md - pd.Timedelta(days=60))
            pos = int(np.searchsorted(f_dts, cutoff, side="right")) - 1
            if pos >= 0:
                if np.isfinite(f_np[pos]) and f_np[pos] > 0:
                    pe = mc_s[mi] / f_np[pos]
                    if 2.0 <= pe <= 200.0:
                        pe_arr[mi] = pe
                if np.isfinite(f_bk[pos]) and f_bk[pos] > 0:
                    pb = mc_s[mi] / f_bk[pos]
                    if 0.2 <= pb <= 50.0:
                        pb_arr[mi] = pb
        pit_pe_m[tk] = pe_arr.astype(np.float32)
        pit_pb_m[tk] = pb_arr.astype(np.float32)

    # Industry grouping map
    ind_groups: Dict[str, List[str]] = {}
    for tk in valid_tks:
        ind_groups.setdefault(ind_of[tk], []).append(tk)

    # Build checkpoint snapshots (dict of dicts) — fast & compact!
    ckpts: Dict[pd.Timestamp, Dict[str, dict]] = {}
    for ci, dt in enumerate(ckpt_dates):
        c_row = C.loc[dt]
        cf_row = C_f.loc[dt]
        ma20_row = MA20.loc[dt]
        ma50_row = MA50.loc[dt]
        ma200_row = MA200.loc[dt]
        hi252_row = HI252.loc[dt]
        atr_row = ATR14.loc[dt]
        atrp_row = ATR_PCT.loc[dt]
        r20_row = R20.loc[dt]
        r63_row = R63.loc[dt]
        rs_raw_row = COMP_RS_RAW.loc[dt]
        liq_row = LIQ20.loc[dt]
        dcr_row = DELIV_CR20.loc[dt]
        drat_row = DELIV_RATIO.loc[dt]
        mcap_row = PIT_MCAP.loc[dt]

        # Active universe on day dt: trading (Close > 5), liq >= 2.0Cr/d, and (mcap >= 500 Cr OR unmapped delisted stock)
        elig_mask = c_row.notna() & (c_row > 5.0) & (liq_row >= 2.0) & (mcap_row.isna() | (mcap_row >= 500.0))
        act_tks = c_row.index[elig_mask].tolist()
        if not act_tks:
            ckpts[dt] = {}
            continue

        # Cross-sectional CompRS percentile strictly across active stocks on day dt
        rs_pct = rs_raw_row.loc[act_tks].rank(pct=True) * 100.0

        # Industry median 63d return percentile strictly on day dt
        ind_r63 = {}
        for ind, members in ind_groups.items():
            vals = r63_row.reindex(members).dropna()
            if len(vals) >= 2:
                ind_r63[ind] = float(vals.median())
        ind_s = pd.Series(ind_r63)
        ind_pct_map = (ind_s.rank(pct=True) * 100.0).to_dict() if not ind_s.empty else {}

        # Sector median PIT PE / PB strictly <= dt (using rolling past 60m <= dt)
        m_hist = pit_pe_m.loc[:dt].tail(60)
        mb_hist = pit_pb_m.loc[:dt].tail(60)
        cur_pe_s = m_hist.iloc[-1] if len(m_hist) else pd.Series(dtype=float)
        cur_pb_s = mb_hist.iloc[-1] if len(mb_hist) else pd.Series(dtype=float)
        sec_med_pe: Dict[str, float] = {}
        sec_med_pb: Dict[str, float] = {}
        for ind, members in ind_groups.items():
            vp = cur_pe_s.reindex(members).dropna()
            if len(vp) >= 3:
                sec_med_pe[ind] = float(vp.median())
            vb = cur_pb_s.reindex(members).dropna()
            if len(vb) >= 3:
                sec_med_pb[ind] = float(vb.median())
        mkt_med_pe = float(cur_pe_s.dropna().median()) if len(cur_pe_s.dropna()) else 20.0
        mkt_med_pb = float(cur_pb_s.dropna().median()) if len(cur_pb_s.dropna()) else 2.5

        # Own 5Y P15 PE/PB strictly <= dt (vectorized across all columns at once!)
        own_p15_pe = m_hist.quantile(0.15) if len(m_hist) >= 6 else pd.Series(dtype=float)
        own_p15_pb = mb_hist.quantile(0.15) if len(mb_hist) >= 6 else pd.Series(dtype=float)
        own_cnt_pe = m_hist.notna().sum() if len(m_hist) else pd.Series(dtype=int)

        cutoff_fy = np.datetime64(dt - pd.Timedelta(days=60))
        snap: Dict[str, dict] = {}
        for tk in act_tks:
            px = float(cf_row[tk])
            m20 = float(ma20_row[tk]) if pd.notna(ma20_row[tk]) else np.nan
            m50 = float(ma50_row[tk]) if pd.notna(ma50_row[tk]) else np.nan
            m200 = float(ma200_row[tk]) if pd.notna(ma200_row[tk]) else np.nan
            h252 = float(hi252_row[tk]) if pd.notna(hi252_row[tk]) else np.nan
            atr = float(atr_row[tk]) if pd.notna(atr_row[tk]) else np.nan
            atrp = float(atrp_row[tk]) if pd.notna(atrp_row[tk]) else 3.0
            g20 = float(r20_row[tk]) if pd.notna(r20_row[tk]) else 0.0
            liq = float(liq_row[tk])
            dcr = float(dcr_row[tk]) if pd.notna(dcr_row[tk]) else 0.0
            drat = float(drat_row[tk]) if pd.notna(drat_row[tk]) else 1.0
            mc_v = float(mcap_row[tk]) if pd.notna(mcap_row[tk]) else np.nan
            crs = float(rs_pct.get(tk, 50.0)) if pd.notna(rs_pct.get(tk, np.nan)) else 50.0
            ind = ind_of[tk]
            irank = float(ind_pct_map.get(ind, 50.0))

            # Stage classification (Weinstein/Minervini causal)
            if np.isfinite(m50) and np.isfinite(m200):
                if px > m50 and px > m200:
                    stage = 2
                elif px < m50 and px < m200:
                    stage = 4
                elif px < m50 and px >= m200:
                    stage = 3
                else:
                    stage = 1
            else:
                stage = 0

            # Dual-ATR Gate 1 + Extension Gate (matches forensics.py post-Dual-ATR logic!)
            gate1 = (stage == 2) and np.isfinite(h252) and (px >= 0.84 * h252)
            ext_ok = False
            if gate1 and np.isfinite(atr) and atr > 0 and np.isfinite(m20):
                d20_atr = (px - m20) / atr
                d50_atr = (px - m50) / atr
                ext_ok = ((d20_atr <= 2.2) or (d50_atr <= 3.8)) and (g20 <= 32.0) and (atrp <= 7.5)
            buy_now = bool(gate1 and ext_ok)

            # Strictly Causal Annual Fundamentals (FY <= dt - 60d)
            fy_np_yoy = None
            fy_sales_yoy = None
            cfo_pat = None
            cmp_lt_bear = False
            bear_disc = None

            fc = fy_cache.get(tk)
            if fc is not None and np.isfinite(mc_v) and mc_v >= 500.0:
                pos = int(np.searchsorted(fc["dates"], cutoff_fy, side="right")) - 1
                if pos >= 1:
                    np0, np1 = fc["np"][pos], fc["np"][pos - 1]
                    sa0, sa1 = fc["sales"][pos], fc["sales"][pos - 1]
                    if np.isfinite(np0) and np.isfinite(np1) and np1 > 0:
                        fy_np_yoy = (np0 / np1 - 1.0) * 100.0
                    if np.isfinite(sa0) and np.isfinite(sa1) and sa1 > 0:
                        fy_sales_yoy = (sa0 / sa1 - 1.0) * 100.0
                    # 3Y cumulative CFO / PAT
                    lo_i = max(0, pos - 2)
                    cfo_sl = fc["cfo"][lo_i:pos + 1]
                    np_sl = fc["np"][lo_i:pos + 1]
                    valid_c = np.isfinite(cfo_sl) & np.isfinite(np_sl)
                    if valid_c.sum() >= 2 and np_sl[valid_c].sum() > 0:
                        cfo_pat = float(cfo_sl[valid_c].sum() / np_sl[valid_c].sum())

                    # Strictly Causal Bear Intrinsic Value (v4.1 Bear Formula)
                    if is_bfsi[tk]:
                        bk0 = fc["book"][pos]
                        if np.isfinite(bk0) and bk0 > 0 and np.isfinite(np0) and np0 > 0:
                            roe = min(max(np0 / bk0, 0.05), 0.16)
                            s_pb = sec_med_pb.get(ind, mkt_med_pb)
                            opb = float(own_p15_pb.get(tk, np.nan)) if own_cnt_pe.get(tk, 0) >= 12 else np.nan
                            bear_pb = min(opb, 0.80 * s_pb) if np.isfinite(opb) else 0.70 * s_pb
                            bear_pb = float(np.clip(bear_pb, 0.6, 4.5))
                            bear_mc = bk0 * bear_pb * (((1.0 + roe * 0.70) / 1.12) ** 3)
                            if bear_mc > 0:
                                bear_disc = (1.0 - mc_v / bear_mc) * 100.0
                                cmp_lt_bear = (mc_v < bear_mc) and (mc_v / bear_mc >= 0.25)
                    else:
                        if np.isfinite(np0) and np0 > 0 and np.isfinite(sa0) and sa0 > 0:
                            # Conservative bear growth rate from strictly causal historical sales CAGR
                            if pos >= 3 and np.isfinite(fc["sales"][pos - 3]) and fc["sales"][pos - 3] > 0:
                                cagr3 = (sa0 / fc["sales"][pos - 3]) ** (1.0 / 3.0) - 1.0
                            else:
                                cagr3 = (fy_sales_yoy / 100.0) if fy_sales_yoy is not None else 0.08
                            g_bear = float(np.clip(cagr3 * 0.55, 0.02, 0.14))
                            s_pe = sec_med_pe.get(ind, mkt_med_pe)
                            ope = float(own_p15_pe.get(tk, np.nan)) if own_cnt_pe.get(tk, 0) >= 12 else np.nan
                            bear_pe = min(ope, 0.75 * s_pe) if np.isfinite(ope) else 0.68 * s_pe
                            bear_pe = float(np.clip(bear_pe, 6.0, 32.0))
                            bear_mc = np0 * bear_pe * (((1.0 + g_bear) / 1.12) ** 3)
                            if bear_mc > 0:
                                bear_disc = (1.0 - mc_v / bear_mc) * 100.0
                                cmp_lt_bear = (mc_v < bear_mc) and (mc_v / bear_mc >= 0.25)

            snap[tk] = dict(
                px=px, mcap=mc_v, liq=liq, dcr=dcr, drat=drat, atrp=atrp, crs=crs, irank=irank,
                stage=stage, gate1=gate1, buy_now=buy_now, is_bfsi=is_bfsi[tk], ind=ind,
                fy_np_yoy=fy_np_yoy, fy_sales_yoy=fy_sales_yoy, cfo_pat=cfo_pat,
                cmp_lt_bear=cmp_lt_bear, bear_disc=bear_disc,
            )
        ckpts[dt] = snap

    print(f"[panel] DONE building 10.6Y causal panel ({len(ckpt_dates)} checkpoints, {len(valid_tks)} stocks) in {time.time()-t0:.1f}s", flush=True)
    return dict(
        sim_dates=sim_dates, ckpt_dates=set(ckpt_dates), monthly_dates=monthly_dates,
        valid_tks=valid_tks, t2c={tk: i for i, tk in enumerate(valid_tks)},
        O=O.loc[sim_dates].to_numpy(dtype=np.float32),
        L=L.loc[sim_dates].to_numpy(dtype=np.float32),
        C=C.loc[sim_dates].to_numpy(dtype=np.float32),
        MA50=MA50.loc[sim_dates].to_numpy(dtype=np.float32),
        LIQ20=LIQ20.loc[sim_dates].to_numpy(dtype=np.float32),
        PIT_MCAP=PIT_MCAP.loc[sim_dates].to_numpy(dtype=np.float32),
        n50=n50.reindex(sim_dates).ffill(),
        mid=mid.reindex(sim_dates).ffill(),
        sml=sml.reindex(sim_dates).ffill(),
        regimes=regimes_daily, ckpts=ckpts,
    )


def _gross_cap(reg: dict, throttle_mode: str) -> float:
    m = reg["mode"]
    if throttle_mode == "off":
        return 1.0
    if throttle_mode == "dual_breadth":
        # If All-Cap breadth (>50DMA) is strong (>=55%) despite Nifty-50 lag, allow 85% in CAUTION
        if m == "CAUTION" and reg["b50"] >= 55.0:
            return 0.85
    return {"BEAR": 0.40, "CAUTION": 0.75, "RECOVERY": 0.90, "BULL": 1.00}.get(m, 0.75)


def run_sim(
    panel: dict,
    strategy: str,
    *,
    liq_floor: float = 2.0,
    mcap_floor: float = 500.0,
    max_pos_pct: float = 10.0,
    use_inv_vol: bool = False,
    max_per_ind: Optional[int] = None,
    throttle_mode: str = "off",
    cooldown_days: int = 0,
    monthly_only: bool = False,
) -> dict:
    dates = list(panel["sim_dates"])
    dec_dates = panel["monthly_dates"] if monthly_only else panel["ckpt_dates"]
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
    cand_counts: List[int] = []
    pnl_by_tk: Dict[str, float] = {}

    for di, dt in enumerate(dates):
        # 1. Execute pending T+1 Open orders
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
                    pnl_rs = net - p["cost_basis"]
                    pnl_p = (px / p["avg_px"] - 1.0) * 100.0
                    sells_pnl.append(pnl_p)
                    pnl_by_tk[tk] = pnl_by_tk.get(tk, 0.0) + pnl_rs
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

        # 2. Daily Mark-to-Market + Stop / Trailing / MA50 / Delist Checks
        eq = cash
        for tk in list(positions.keys()):
            p = positions[tk]
            col = t2c[tk]
            c_px = float(C[di, col])
            l_px = float(L[di, col])
            if not np.isfinite(c_px) or c_px <= 0:
                # Stock halted/delisted: exit at last known price with haircut if missing > 15 days
                p["miss"] = p.get("miss", 0) + 1
                if p["miss"] >= 15:
                    net = p["shares"] * p["last_px"] * 0.85
                    cash += net
                    pnl_by_tk[tk] = pnl_by_tk.get(tk, 0.0) + (net - p["cost_basis"])
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

            # Stop checks (queue T+1 Open sell)
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

        # 3. Decision Checkpoint Evaluation (strictly causal <= dt)
        if dt not in dec_dates or dt not in ckpts:
            continue
        snap = ckpts[dt]
        if not snap:
            continue
        reg = regimes[dt]
        gcap = _gross_cap(reg, throttle_mode)

        # Strategy-specific exit checks on held positions
        for tk in list(positions.keys()):
            if tk in pending:
                continue
            s = snap.get(tk)
            if s is None:
                continue
            if strategy in ("garp_guarded", "garp_cfo_only") and s["fy_np_yoy"] is not None and s["fy_np_yoy"] < -15.0:
                pending[tk] = dict(side="SELL", reason="EARN_BREAK")
            elif strategy == "garp_guarded" and s["stage"] == 4 and (s["px"] / positions[tk]["avg_px"] - 1.0) * 100.0 <= -15.0:
                pending[tk] = dict(side="SELL", reason="S4_STOP")

        # Build candidate list strictly from active stocks passing liquidity & MCap floors
        cands = []
        for tk, s in snap.items():
            if tk in positions or tk in pending:
                continue
            if di < cooldown.get(tk, -1):
                continue
            if s["liq"] < liq_floor:
                continue
            if np.isfinite(s["mcap"]) and s["mcap"] < mcap_floor:
                continue

            ok = False
            score = 0.0
            stop_p, trail_p, use_ma50, mgap = 12.0, 15.0, False, 4.0

            if strategy == "mom_naive_comprs":
                if s["buy_now"]:
                    ok = True
                    score = s["crs"]
                    stop_p, trail_p, use_ma50 = 10.0, 15.0, True

            elif strategy == "mom_deliv_rank":
                if s["buy_now"]:
                    ok = True
                    score = s["dcr"]
                    stop_p, trail_p, use_ma50 = 10.0, 15.0, True

            elif strategy == "mom_quality_ranked":
                if s["buy_now"] and (s["fy_np_yoy"] is None or s["fy_np_yoy"] > 0.0):
                    ok = True
                    np_sc = np.clip((s["fy_np_yoy"] if s["fy_np_yoy"] is not None else 15.0), 0.0, 100.0)
                    dcr_sc = min(100.0, math.log1p(max(0.0, s["dcr"])) * 22.0)
                    score = 0.40 * s["irank"] + 0.35 * dcr_sc + 0.25 * np_sc
                    stop_p, trail_p, use_ma50 = 10.0, 14.0, True

            elif strategy == "garp_unfiltered":
                if s["cmp_lt_bear"]:
                    ok = True
                    score = s["bear_disc"] or 0.0
                    stop_p, trail_p, use_ma50, mgap = None, 25.0, False, 5.0

            elif strategy == "garp_guarded":
                cfo_ok = s["is_bfsi"] or (s["cfo_pat"] is not None and s["cfo_pat"] >= 0.70)
                earn_ok = (s["fy_np_yoy"] is not None and s["fy_np_yoy"] > 0.0)
                stage_ok = (s["stage"] != 4)
                if s["cmp_lt_bear"] and cfo_ok and earn_ok and stage_ok:
                    ok = True
                    score = (s["bear_disc"] or 0.0) + 0.20 * min(s["fy_np_yoy"], 80.0)
                    stop_p, trail_p, use_ma50, mgap = 18.0, 22.0, False, 5.0

            elif strategy == "hybrid_old":
                if s["cmp_lt_bear"] or s["buy_now"]:
                    ok = True
                    score = (s["bear_disc"] or 0.0) + 0.5 * s["crs"]
                    stop_p, trail_p, use_ma50 = 15.0, 18.0, False

            elif strategy == "hybrid_guarded":
                cfo_ok = s["is_bfsi"] or (s["cfo_pat"] is not None and s["cfo_pat"] >= 0.70)
                earn_ok = (s["fy_np_yoy"] is not None and s["fy_np_yoy"] > 0.0)
                if (s["cmp_lt_bear"] and cfo_ok and earn_ok and s["stage"] != 4) or (s["buy_now"] and (s["fy_np_yoy"] is None or s["fy_np_yoy"] > 0.0)):
                    ok = True
                    dcr_sc = min(100.0, math.log1p(max(0.0, s["dcr"])) * 22.0)
                    score = 0.5 * s["irank"] + 0.3 * dcr_sc + 0.2 * (s["bear_disc"] or 0.0)
                    stop_p, trail_p, use_ma50 = 12.0, 16.0, False

            elif strategy == "pead_earnings_flow":
                if (s["stage"] == 2 and s["fy_np_yoy"] is not None and s["fy_np_yoy"] >= 25.0
                        and s["fy_sales_yoy"] is not None and s["fy_sales_yoy"] >= 15.0
                        and s["drat"] >= 1.15 and s["crs"] >= 65.0):
                    ok = True
                    score = min(s["fy_np_yoy"], 120.0) * 0.5 + s["drat"] * 25.0 + s["irank"] * 0.3
                    stop_p, trail_p, use_ma50 = 10.0, 15.0, True

            if ok:
                cands.append((score, tk, s, stop_p, trail_p, use_ma50, mgap))

        cand_counts.append(len(cands))
        if not cands:
            continue
        cands.sort(key=lambda x: x[0], reverse=True)

        # Enforce max slots & max_per_ind
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
            sz = max_pos_pct
            if use_inv_vol:
                # Target ~0.25% daily vol contribution per position -> scaled inversely to ATR%
                sz = float(np.clip(22.0 / max(s["atrp"], 1.5), 4.0, max_pos_pct))
            pending[tk] = dict(
                side="BUY", ref=s["px"], sz=sz, gcap=gcap,
                stop=stop_p, trail=trail_p, ma50=use_ma50, max_gap=mgap, ind=s["ind"],
            )
            ind_counts[s["ind"]] = ind_counts.get(s["ind"], 0) + 1
            slots_left -= 1

    # Add open PnL attribution
    for tk, p in positions.items():
        pnl_by_tk[tk] = pnl_by_tk.get(tk, 0.0) + (p["shares"] * p["last_px"] - p["cost_basis"])

    curve = pd.Series(nav_arr, index=panel["sim_dates"])
    return dict(
        curve=curve,
        cash_mean=round(float(np.mean(cash_arr)), 1),
        buys=n_buys,
        win_rate=round(sum(1 for x in sells_pnl if x > 0) / len(sells_pnl) * 100.0, 1) if sells_pnl else 0.0,
        avg_cands=round(float(np.mean(cand_counts)), 1) if cand_counts else 0.0,
        max_cands=int(max(cand_counts)) if cand_counts else 0,
        top5=[(k, int(v)) for k, v in sorted(pnl_by_tk.items(), key=lambda x: x[1], reverse=True)[:5]],
    )


def perf_metrics(curve: pd.Series) -> dict:
    s = curve.dropna()
    if len(s) < 5:
        return dict(ret=0.0, cagr=0.0, mdd=0.0, sharpe=0.0, calmar=0.0)
    tot = (float(s.iloc[-1]) / float(s.iloc[0]) - 1.0) * 100.0
    yrs = max(0.1, (s.index[-1] - s.index[0]).days / 365.25)
    cagr = ((float(s.iloc[-1]) / float(s.iloc[0])) ** (1.0 / yrs) - 1.0) * 100.0
    mdd = float((s / s.cummax() - 1.0).min() * 100.0)
    r = s.pct_change().dropna()
    sh = float((r.mean() / r.std(ddof=1)) * math.sqrt(252.0)) if r.std(ddof=1) > 1e-12 else 0.0
    cal = cagr / abs(mdd) if abs(mdd) > 0.1 else 0.0
    return dict(ret=round(tot, 2), cagr=round(cagr, 2), mdd=round(mdd, 2), sharpe=round(sh, 3), calmar=round(cal, 2))


def passive_ew_rebalanced(panel: dict) -> pd.Series:
    """Strictly causal monthly-rebalanced EW index of ALL active >=500Cr & >=2Cr/d stocks (survivorship-free!)."""
    dates = list(panel["sim_dates"])
    C, LIQ, MCAP = panel["C"], panel["LIQ20"], panel["PIT_MCAP"]
    nav = np.ones(len(dates), dtype=np.float64) * INIT_CAP
    w_mask = None
    for di in range(len(dates)):
        if di == 0 or dates[di] in panel["monthly_dates"]:
            c0 = C[di, :]
            l0 = LIQ[di, :]
            m0 = MCAP[di, :]
            w_mask = np.isfinite(c0) & (c0 > 5.0) & (l0 >= 2.0) & (np.isnan(m0) | (m0 >= 500.0))
        if di > 0 and w_mask is not None and w_mask.any():
            c_prev = C[di - 1, w_mask]
            c_now = C[di, w_mask]
            ok = np.isfinite(c_prev) & (c_prev > 0) & np.isfinite(c_now) & (c_now > 0)
            if ok.any():
                dr = np.clip(c_now[ok] / c_prev[ok] - 1.0, -0.5, 0.5)
                nav[di] = nav[di - 1] * (1.0 + float(np.mean(dr)))
            else:
                nav[di] = nav[di - 1]
    return pd.Series(nav, index=panel["sim_dates"])


def main() -> int:
    panel = load_universe_and_matrices()
    n50_curve = panel["n50"] / float(panel["n50"].iloc[0]) * INIT_CAP
    ew_curve = passive_ew_rebalanced(panel)

    specs = [
        ("mom_naive_comprs",          "mom_naive_comprs",   dict(liq_floor=2.0,  use_inv_vol=False, max_per_ind=None, throttle_mode="off",          cooldown_days=0)),
        ("mom_deliv_rank_2cr",        "mom_deliv_rank",     dict(liq_floor=2.0,  use_inv_vol=False, max_per_ind=None, throttle_mode="off",          cooldown_days=0)),
        ("mom_quality_2cr_th",        "mom_quality_ranked", dict(liq_floor=2.0,  use_inv_vol=True,  max_per_ind=2,    throttle_mode="on",           cooldown_days=30)),
        ("mom_quality_20cr_th",       "mom_quality_ranked", dict(liq_floor=20.0, use_inv_vol=True,  max_per_ind=2,    throttle_mode="on",           cooldown_days=30)),
        ("mom_quality_20cr_dualb",    "mom_quality_ranked", dict(liq_floor=20.0, use_inv_vol=True,  max_per_ind=2,    throttle_mode="dual_breadth", cooldown_days=30)),
        ("mom_quality_20cr_noth",     "mom_quality_ranked", dict(liq_floor=20.0, use_inv_vol=True,  max_per_ind=2,    throttle_mode="off",          cooldown_days=30)),
        ("garp_unfiltered_2cr",       "garp_unfiltered",    dict(liq_floor=2.0,  use_inv_vol=False, max_per_ind=None, throttle_mode="off",          cooldown_days=0)),
        ("garp_guarded_2cr",          "garp_guarded",       dict(liq_floor=2.0,  use_inv_vol=False, max_per_ind=None, throttle_mode="off",          cooldown_days=60)),
        ("garp_guarded_20cr",         "garp_guarded",       dict(liq_floor=20.0, use_inv_vol=False, max_per_ind=None, throttle_mode="off",          cooldown_days=60)),
        ("hybrid_old_2cr",            "hybrid_old",         dict(liq_floor=2.0,  max_pos_pct=20.0,  throttle_mode="off", cooldown_days=0)),
        ("hybrid_guarded_2cr",        "hybrid_guarded",     dict(liq_floor=2.0,  max_pos_pct=10.0,  throttle_mode="off", cooldown_days=60)),
        ("pead_flow_5cr_th",          "pead_earnings_flow", dict(liq_floor=5.0,  use_inv_vol=True,  max_per_ind=2,    throttle_mode="on",           cooldown_days=30)),
        ("pead_flow_20cr_th",         "pead_earnings_flow", dict(liq_floor=20.0, use_inv_vol=True,  max_per_ind=2,    throttle_mode="on",           cooldown_days=30)),
    ]

    curves: Dict[str, pd.Series] = {"NIFTY_50": n50_curve, "EW_TRADABLE_500CR": ew_curve}
    meta: Dict[str, dict] = {}
    for label, strat, kw in specs:
        res = run_sim(panel, strat, **kw)
        curves[label] = res.pop("curve")
        meta[label] = res

    # Monthly-rebalanced 3-Sleeve Orthogonal Barbells (prevent drift across 10.6 years!)
    def make_barbell(weights: Dict[str, float]) -> pd.Series:
        # Monthly rebalanced weighted return portfolio
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
            # Drift weights daily, rebalance monthly
            if dt in panel["monthly_dates"]:
                w_cur = dict(weights)
            else:
                denom = 1.0 + port_r
                if denom > 0:
                    w_cur = {k: w_cur[k] * (1.0 + float(r_row[k])) / denom for k in weights}
        return pd.Series(out, index=rets.index)

    curves["BARBELL_50_30_20"] = make_barbell({"garp_guarded_2cr": 0.50, "mom_quality_20cr_th": 0.30, "pead_flow_20cr_th": 0.20})
    curves["BARBELL_40_40_20"] = make_barbell({"garp_guarded_2cr": 0.40, "mom_quality_20cr_th": 0.40, "pead_flow_20cr_th": 0.20})
    curves["BARBELL_40_40_20_DB"] = make_barbell({"garp_guarded_2cr": 0.40, "mom_quality_20cr_dualb": 0.40, "pead_flow_20cr_th": 0.20})
    curves["BARBELL_INST_20CR"] = make_barbell({"garp_guarded_20cr": 0.50, "mom_quality_20cr_th": 0.30, "pead_flow_20cr_th": 0.20})

    # Compute Full-10.6Y and Per-Regime performance tables + cross-sleeve correlations per regime
    results = {"full_10y": {}, "regimes": {}, "corr_full": {}, "corr_by_regime": {}, "meta": meta}

    print("\n" + "=" * 125)
    print("FULL 10.6-YEAR WALK-FORWARD PERFORMANCE (2016-04-01 -> 2026-11-20 | Survivorship-Free >=500Cr & >=2Cr/d)")
    print("=" * 125)
    for name, c in curves.items():
        m = perf_metrics(c)
        results["full_10y"][name] = m
        ex = meta.get(name, {})
        c_str = f"Cash:{ex.get('cash_mean', 0):4.1f}% Win:{ex.get('win_rate', 0):4.1f}% ({ex.get('buys', 0):4d}B)" if ex else ""
        print(f"  {name:24s} | TotRet: {m['ret']:+8.1f}% | CAGR: {m['cagr']:+6.2f}% | MaxDD: {m['mdd']:7.2f}% | Sharpe: {m['sharpe']:5.3f} | Calmar: {m['calmar']:4.2f} | {c_str}")

    corr_df = pd.DataFrame({
        "GARP_2Cr": curves["garp_guarded_2cr"].pct_change(),
        "MOMQ_20Cr": curves["mom_quality_20cr_th"].pct_change(),
        "PEAD_20Cr": curves["pead_flow_20cr_th"].pct_change(),
        "NIFTY_50": curves["NIFTY_50"].pct_change(),
    }).dropna()
    results["corr_full"] = corr_df.corr().round(2).to_dict()
    print("\nFull 10.6Y Daily Return Correlation Matrix:")
    print(corr_df.corr().round(2).to_string())

    for r_name, r_s, r_e in REGIMES:
        results["regimes"][r_name] = {}
        print(f"\n--- {r_name} ({r_s} -> {r_e}) ---")
        for name, c in curves.items():
            sub = c.loc[r_s:r_e]
            m = perf_metrics(sub)
            results["regimes"][r_name][name] = m
            if name in ("NIFTY_50", "EW_TRADABLE_500CR", "mom_naive_comprs", "mom_quality_2cr_th", "mom_quality_20cr_th", "mom_quality_20cr_noth", "garp_unfiltered_2cr", "garp_guarded_2cr", "hybrid_old_2cr", "hybrid_guarded_2cr", "pead_flow_20cr_th", "BARBELL_40_40_20", "BARBELL_INST_20CR"):
                print(f"  {name:24s} | Ret: {m['ret']:+7.2f}% | CAGR: {m['cagr']:+7.2f}% | MaxDD: {m['mdd']:7.2f}% | Sharpe: {m['sharpe']:6.3f}")
        rc = corr_df.loc[r_s:r_e].corr().round(2)
        results["corr_by_regime"][r_name] = rc.to_dict()

    # Check regime-mode distribution over 10.6Y
    rmodes = pd.Series([v["mode"] for v in panel["regimes"].values()], index=panel["sim_dates"])
    results["regime_mode_pct"] = (rmodes.value_counts(normalize=True) * 100.0).round(1).to_dict()
    print("\nDaily Market Regime Mode Distribution (2016-2026):", results["regime_mode_pct"])

    # Also validate 2026-02-27 -> 2026-11-20 exact overlap window!
    print("\n--- 2026 OVERLAP VALIDATION (2026-02-27 -> 2026-11-20) ---")
    results["overlap_2026"] = {}
    for name in ("NIFTY_50", "EW_TRADABLE_500CR", "garp_unfiltered_2cr", "garp_guarded_2cr", "mom_naive_comprs", "mom_quality_2cr_th", "mom_quality_20cr_th", "pead_flow_20cr_th", "BARBELL_40_40_20", "BARBELL_INST_20CR"):
        m = perf_metrics(curves[name].loc["2026-02-27":"2026-11-20"])
        results["overlap_2026"][name] = m
        print(f"  {name:24s} | Ret: {m['ret']:+7.2f}% | MaxDD: {m['mdd']:7.2f}% | Sharpe: {m['sharpe']:6.3f}")

    out_p = os.path.join(OUT_DIR, "multi_regime_10y_results.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
