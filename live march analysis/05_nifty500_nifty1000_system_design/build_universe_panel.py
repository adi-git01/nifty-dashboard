"""Build strictly point-in-time (<= t) causal dataset for FULL Nifty 500 (~497 stocks)
and FULL Nifty 1000 (~1,000+ stocks) from 2026-02-27 -> 2026-11-20 (180 NSE sessions).

Zero selection bias: includes EVERY universe constituent with valid NSE history.
Caches `panel_n1000.pkl` so multi-expert simulations and control-variable sweeps run in seconds.
"""
from __future__ import annotations

import glob
import os
import pickle
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

REPO = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\repo_live"
OUT_DIR = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\n500_n1000_march"
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(REPO)
sys.path.insert(0, REPO)

from utils import forensics as fx, fund_trends as ft, nifty500_list as n500, nifty1000_list as n1000, nse_data as nd, nse_prices, regime_manager as rm, valuation as V

START_DATE = "2026-02-27"
END_DATE = "2026-11-20"
EVENT_DATES = [
    "2026-02-27", "2026-03-30", "2026-04-08", "2026-05-18",
    "2026-06-22", "2026-07-20", "2026-09-15", "2026-11-06",
]
MONTHLY_GATES = [
    "2026-02-27", "2026-03-30", "2026-04-30", "2026-05-29",
    "2026-06-30", "2026-07-31", "2026-08-31", "2026-09-30", "2026-10-30",
]
CKPT_DATES = sorted(set(EVENT_DATES + MONTHLY_GATES))


def _load_indices():
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
        # P&L: 45-day SEBI Reg 33 filing lag
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

        # Shareholding: 21-day SEBI Reg 31 filing lag!
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


def main():
    t0 = time.time()
    mm = pd.read_parquet(sorted(glob.glob(os.path.join(REPO, "data", "cache", "market_master_*.parquet")))[-1])
    n500_set = set(n500.TICKERS)
    n1000_raw = list(dict.fromkeys(list(n1000.TICKERS_1000) + list(n500.TICKERS)))
    print(f"Raw universes: Nifty 500 = {len(n500_set)}, Nifty 1000 = {len(n1000_raw)}")

    # Sub-industry & sector lookup
    sub_ind_map = dict(getattr(n1000, "SUB_INDUSTRY_MAP", {}) or {})
    sec_map = dict(zip(mm["ticker"], mm.get("sector_granular", mm.get("sector", pd.Series("", index=mm.index))).fillna("").astype(str)))
    for tk in n1000_raw:
        if tk not in sub_ind_map or not sub_ind_map[tk]:
            sub_ind_map[tk] = sec_map.get(tk) or n500.SECTOR_MAP.get(tk, "Other")

    print("Loading NSE prices, indices, delivery, fundamentals, valuation context...")
    hist = nse_prices.history(n1000_raw)
    nifty_df, vix_s = _load_indices()
    deliv_df = nd.delivery_panel()
    if deliv_df is None or deliv_df.empty:
        deliv_df = pd.DataFrame()
    else:
        deliv_df = deliv_df.copy()
        deliv_df.columns = [c if str(c).endswith(".NS") else f"{c}.NS" for c in deliv_df.columns]
    mc = pd.read_csv(os.path.join(REPO, "data", "nse", "market_context.csv"))
    mc["date"] = pd.to_datetime(mc["date"], errors="coerce")
    mc = mc.dropna(subset=["date"]).set_index("date").sort_index()
    ann_raw, qtr_raw = ft._data(REPO)
    val_ctx = V.load_context()

    # Common trading calendar
    ref_df = hist["RELIANCE.NS"].sort_index()
    dates_ts = ref_df.loc[START_DATE:END_DATE].index
    dates = [d.strftime("%Y-%m-%d") for d in dates_ts]
    print(f"Trading window: {dates[0]} -> {dates[-1]} ({len(dates)} sessions) [{time.time()-t0:.1f}s]")

    # Keep constituents with valid OHLCV on START_DATE and >= 170/180 sessions in window (no missing-data ghosts)
    valid_n1000: List[str] = []
    valid_n500: List[str] = []
    sorted_hist: Dict[str, pd.DataFrame] = {}
    for tk in n1000_raw:
        df = hist.get(tk)
        if df is None or df.empty:
            continue
        df = df.sort_index()
        sub = df.loc[START_DATE:END_DATE]
        if len(sub) < 170 or len(df.loc[:START_DATE]) < 60:
            continue
        if float(sub["Close"].iloc[0]) <= 5.0:
            continue
        valid_n1000.append(tk)
        sorted_hist[tk] = df
        if tk in n500_set:
            valid_n500.append(tk)
    print(f"Valid continuous-history constituents: Nifty 500 = {len(valid_n500)}, Nifty 1000 = {len(valid_n1000)}", flush=True)

    # Vectorized daily matrices across valid_n1000 (super fast!)
    idx_all = ref_df.loc["2025-02-01":END_DATE].index
    O = pd.DataFrame({tk: sorted_hist[tk]["Open"].reindex(idx_all).ffill() for tk in valid_n1000}, dtype=np.float32)
    H = pd.DataFrame({tk: sorted_hist[tk]["High"].reindex(idx_all).ffill() for tk in valid_n1000}, dtype=np.float32)
    L = pd.DataFrame({tk: sorted_hist[tk]["Low"].reindex(idx_all).ffill() for tk in valid_n1000}, dtype=np.float32)
    C = pd.DataFrame({tk: sorted_hist[tk]["Close"].reindex(idx_all).ffill() for tk in valid_n1000}, dtype=np.float32)
    Vol = pd.DataFrame({tk: sorted_hist[tk]["Volume"].reindex(idx_all).fillna(0.0) for tk in valid_n1000}, dtype=np.float32)

    MA20 = C.rolling(20, min_periods=15).mean()
    MA50 = C.rolling(50, min_periods=40).mean()
    MA200 = C.rolling(200, min_periods=150).mean()
    HI252 = C.rolling(252, min_periods=150).max()
    LO252 = L.rolling(252, min_periods=180).min()
    LIQ20 = (C * Vol).rolling(20, min_periods=10).median() / 1e7

    # True Range & ATR14%
    prev_c = C.shift(1)
    tr = np.maximum(H - L, np.maximum((H - prev_c).abs(), (L - prev_c).abs()))
    ATR14 = tr.rolling(14, min_periods=10).mean()
    ATR_PCT = (ATR14 / C.replace(0, np.nan)) * 100.0

    # BBP (20d Bollinger BandWidth percentile over 120d)
    bb_std = C.rolling(20, min_periods=15).std()
    bbw = (4.0 * bb_std / MA20.replace(0, np.nan))
    BBP = bbw.rolling(120, min_periods=60).rank(pct=True) * 100.0

    # Causal RS vs Nifty
    nif_c = nifty_df["Close"].reindex(idx_all).ffill().astype(np.float32)
    rs1m = ((C / C.shift(21)).div(nif_c / nif_c.shift(21), axis=0) - 1.0) * 100.0
    rs3m = ((C / C.shift(63)).div(nif_c / nif_c.shift(63), axis=0) - 1.0) * 100.0
    rs6m = ((C / C.shift(126)).div(nif_c / nif_c.shift(126), axis=0) - 1.0) * 100.0
    COMPRS = 0.25 * rs1m.fillna(0.0) + 0.35 * rs3m.fillna(0.0) + 0.40 * rs6m.fillna(0.0)

    # Delivery % and 20d ratio
    if not deliv_df.empty:
        D_PCT = deliv_df.reindex(index=idx_all, columns=valid_n1000).ffill(limit=3)
        D_AVG20 = D_PCT.rolling(20, min_periods=10).mean()
        D_RATIO = D_PCT / D_AVG20.replace(0, np.nan)
    else:
        D_PCT = pd.DataFrame(np.nan, index=idx_all, columns=valid_n1000)
        D_RATIO = pd.DataFrame(np.nan, index=idx_all, columns=valid_n1000)

    # Market breadth from full N1000 matrix (strictly causal <= t!)
    pct50_s = (C > MA50).sum(axis=1) / MA50.notna().sum(axis=1).replace(0, np.nan) * 100.0
    pct200_s = (C > MA200).sum(axis=1) / MA200.notna().sum(axis=1).replace(0, np.nan) * 100.0
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
            "base_mode": base_mode,
            "engine_mode": eff_mode,
            "recovery": bool(rec["active"]),
            "vix": round(float(vix_sub.iloc[-1]), 2) if not vix_sub.empty else None,
            "pct50": round(float(pct50_s.loc[dt]), 1) if pd.notna(pct50_s.loc[dt]) else None,
            "pct200": round(float(pct200_s.loc[dt]), 1) if pd.notna(pct200_s.loc[dt]) else None,
            "nifty_close": round(c_last, 2),
        }

    # Pre-pivot annual & quarterly frames ONCE per ticker (massive speedup!)
    print(f"Pre-pivoting Screener fundamentals for {len(valid_n1000)} stocks... [{time.time()-t0:.1f}s]")
    ann_piv: Dict[str, pd.DataFrame | None] = {}
    qtr_piv: Dict[str, pd.DataFrame | None] = {}
    for tk in valid_n1000:
        ka = ft._key(tk, ann_raw)
        if ka is not None:
            w = ann_raw[ka].pivot_table(index="period", columns="field", values="value", aggfunc="last").sort_index()
            w = w[w.index.month == 3]
            if not w.empty:
                g = lambda col: w[col] if col in w.columns else pd.Series(np.nan, index=w.index)
                gross = g("fixed_assets") + g("cwip").fillna(0.0)
                w["_capex"] = gross.diff() + g("dep").fillna(0.0)
            ann_piv[tk] = w
        else:
            ann_piv[tk] = None
        kq = ft._key(tk, qtr_raw)
        if kq is not None:
            wq = qtr_raw[kq].pivot_table(index="period", columns="field", values="value", aggfunc="last").sort_index()
            qtr_piv[tk] = wq
        else:
            qtr_piv[tk] = None

    # Compute checkpoint signals strictly <= dt for all 14 unique checkpoints (8 events + 9 monthly gates)
    print(f"Computing strictly causal Forensics (PRE & POST Dual-ATR), Valuation v4.1, Fundamentals & Industry Rank across {len(CKPT_DATES)} checkpoints... [{time.time()-t0:.1f}s]")
    ckpts: Dict[str, Dict[str, dict]] = {}
    ind_board_hist: Dict[str, Dict[str, float]] = {}  # dt -> {industry: rank_pct 0..100 (100=best)}

    for ci, dt_str in enumerate(CKPT_DATES, 1):
        dt = pd.Timestamp(dt_str)
        # 1. Point-in-time Industry Board (strictly <= dt!)
        ind_rows: Dict[str, List[tuple]] = {}
        for tk in valid_n1000:
            c_v = float(C.loc[dt, tk])
            m50_v = float(MA50.loc[dt, tk]) if pd.notna(MA50.loc[dt, tk]) else np.nan
            r3_v = float(rs3m.loc[dt, tk]) if pd.notna(rs3m.loc[dt, tk]) else np.nan
            ind_k = sub_ind_map.get(tk, "Other")
            if np.isfinite(m50_v) and np.isfinite(r3_v):
                ind_rows.setdefault(ind_k, []).append((1.0 if c_v > m50_v else 0.0, r3_v))
        ind_scores = {}
        for ind_k, vals in ind_rows.items():
            if len(vals) >= 3:
                abv50 = float(np.mean([x[0] for x in vals])) * 100.0
                med_rs = float(np.median([x[1] for x in vals]))
                ind_scores[ind_k] = 0.5 * abv50 + 0.5 * np.clip(med_rs + 50.0, 0.0, 100.0)
        if ind_scores:
            s_ind = pd.Series(ind_scores).rank(pct=True) * 100.0
            ind_board_hist[dt_str] = {k: round(float(v), 1) for k, v in s_ind.items()}
        else:
            ind_board_hist[dt_str] = {}

        # 2. Per-stock causal Forensics (PRE & POST) + Fundamentals + Valuation v4.1
        snap: Dict[str, dict] = {}
        for tk in valid_n1000:
            c_v = float(C.loc[dt, tk])
            h_v = float(H.loc[dt, tk])
            l_v = float(L.loc[dt, tk])
            m50_v = float(MA50.loc[dt, tk]) if pd.notna(MA50.loc[dt, tk]) else np.nan
            m200_v = float(MA200.loc[dt, tk]) if pd.notna(MA200.loc[dt, tk]) else np.nan
            liq_v = float(LIQ20.loc[dt, tk]) if pd.notna(LIQ20.loc[dt, tk]) else 0.0
            atr_v = float(ATR14.loc[dt, tk]) if pd.notna(ATR14.loc[dt, tk]) else c_v * 0.035
            atr_p = float(ATR_PCT.loc[dt, tk]) if pd.notna(ATR_PCT.loc[dt, tk]) else 3.5
            bbp_v = float(BBP.loc[dt, tk]) if pd.notna(BBP.loc[dt, tk]) else None
            crs_v = float(COMPRS.loc[dt, tk]) if pd.notna(COMPRS.loc[dt, tk]) else 0.0
            dp_v = float(D_PCT.loc[dt, tk]) if pd.notna(D_PCT.loc[dt, tk]) else None
            dr_v = float(D_RATIO.loc[dt, tk]) if pd.notna(D_RATIO.loc[dt, tk]) else None
            stage = 2 if (np.isfinite(m200_v) and c_v > m50_v > m200_v) else (
                4 if (np.isfinite(m200_v) and c_v < m50_v < m200_v) else (1 if (np.isfinite(m50_v) and c_v > m50_v) else 3)
            )

            # Forensics PRE & POST Dual-ATR: mathematically BUY NOW requires Gate-1 (Close>MA50 & Close>MA200 & Close>=0.85*HI252)
            act_pre = act_post = "WATCH"
            fx_state = "NO ANCHOR"
            gate1 = False
            stop_fx = entry_fx = None
            hi252_v = float(HI252.loc[dt, tk]) if pd.notna(HI252.loc[dt, tk]) else np.nan
            gate1_fast = bool(liq_v >= 2.0 and np.isfinite(m50_v) and np.isfinite(m200_v) and c_v > m50_v and c_v > m200_v and (not np.isfinite(hi252_v) or c_v >= 0.84 * hi252_v))
            if gate1_fast:
                sub_px = sorted_hist[tk].loc[:dt]
                if len(sub_px) >= 60:
                    r_post = fx.analyse(sub_px)
                    clv_last = (c_v - l_v) / (h_v - l_v) if h_v > l_v else 0.5
                    act_post = str(r_post.get("action") or "WATCH")
                    act_pre = _pre_dual_atr_action(r_post, clv_last)
                    fx_state = str(r_post.get("state") or "NO ANCHOR")
                    gate1 = bool(r_post.get("gate1", False))
                    if np.isfinite(r_post.get("stop", np.nan)):
                        stop_fx = round(float(r_post["stop"]), 2)
                    if np.isfinite(r_post.get("entry", np.nan)):
                        entry_fx = round(float(r_post["entry"]), 2)

            # Point-in-time Fundamentals
            fp = _pit_fund_at(ann_piv[tk], qtr_piv[tk], dt)

            # Point-in-time Valuation v4.1 (run on the 8 Event Dates for all liquid stocks; fast & exact!)
            bear_blend = p50 = None
            val_rating = "NO DATA"
            cmp_lt_bear = False
            if dt_str in EVENT_DATES and liq_v >= 2.0:
                try:
                    v = V.value_stock(tk.replace(".NS", ""), val_ctx, price=c_v, as_of=dt_str, knobs=False)
                    if v.get("status") == "OK":
                        bl = v.get("blends") or {}
                        bb = bl.get("bear", np.nan)
                        if np.isfinite(bb):
                            bear_blend = round(float(bb), 2)
                            cmp_lt_bear = bool(c_v <= float(bb))
                        p50 = round(float(v["dist"]["P50"]), 2)
                        val_rating = str(v.get("rating") or "")
                except Exception:
                    pass

            ind_k = sub_ind_map.get(tk, "Other")
            snap[tk] = {
                "close": round(c_v, 2), "ma50": round(m50_v, 2) if np.isfinite(m50_v) else None,
                "ma200": round(m200_v, 2) if np.isfinite(m200_v) else None,
                "stage": stage, "liq_cr": round(liq_v, 2), "atr": round(atr_v, 2), "atr_pct": round(atr_p, 2),
                "bbp": round(bbp_v, 1) if bbp_v is not None else None, "comp_rs": round(crs_v, 1),
                "deliv_pct": round(dp_v, 1) if dp_v is not None else None,
                "deliv_ratio": round(dr_v, 2) if dr_v is not None else None,
                "abs_deliv_cr": round(liq_v * dp_v / 100.0, 2) if (dp_v is not None and liq_v > 0) else 0.0,
                "industry": ind_k, "ind_rank_pct": ind_board_hist[dt_str].get(ind_k, 50.0),
                "gate1": gate1, "fx_state": fx_state, "action_pre": act_pre, "action_post": act_post,
                "stop_fx": stop_fx, "entry_fx": entry_fx or round(c_v, 2),
                "bear_blend": bear_blend, "p50": p50, "val_rating": val_rating, "cmp_lt_bear": cmp_lt_bear,
                **fp,
            }
        ckpts[dt_str] = snap
        pre_n = sum(1 for v in snap.values() if v["action_pre"] == "BUY NOW")
        post_n = sum(1 for v in snap.values() if v["action_post"] == "BUY NOW")
        bear_n = sum(1 for v in snap.values() if v["cmp_lt_bear"])
        print(f"  [{ci:2d}/{len(CKPT_DATES)}] {dt_str} ({daily_regimes[dt_str]['engine_mode']:8s}, VIX={daily_regimes[dt_str]['vix']}) -> BUY_NOW pre={pre_n:3d} / post={post_n:3d} | <Bear={bear_n:3d} [{time.time()-t0:.1f}s]")

    # Slice daily execution arrays strictly to START_DATE..END_DATE for compact fast storage
    sl = dates_ts
    payload = {
        "dates": dates,
        "event_dates": EVENT_DATES,
        "monthly_gates": MONTHLY_GATES,
        "nifty500": valid_n500,
        "nifty1000": valid_n1000,
        "sub_industry": {tk: sub_ind_map.get(tk, "Other") for tk in valid_n1000},
        "regimes": daily_regimes,
        "ckpts": ckpts,
        "O": O.loc[sl].to_numpy(dtype=np.float32),
        "H": H.loc[sl].to_numpy(dtype=np.float32),
        "L": L.loc[sl].to_numpy(dtype=np.float32),
        "C": C.loc[sl].to_numpy(dtype=np.float32),
        "MA50": MA50.loc[sl].to_numpy(dtype=np.float32),
        "ticker_to_col": {tk: i for i, tk in enumerate(valid_n1000)},
    }
    out_pkl = os.path.join(OUT_DIR, "panel_n1000.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {out_pkl} ({os.path.getsize(out_pkl)/1e6:.1f} MB) in {time.time()-t0:.1f}s!")


if __name__ == "__main__":
    main()
