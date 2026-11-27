"""Build strict point-in-time (PIT) dashboard packets for ULTRACEMCO.NS and KALYANKJIL.NS (Feb 27 -> Nov 25, 2026)."""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

REPO = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\repo_nov21"
CACHE7Y = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\autoresearch_7y\panel_7y_cache.pkl"
OUT_DIR = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\time_march"
PKT_DIR = os.path.join(OUT_DIR, "packets")

if REPO not in sys.path:
    sys.path.insert(0, REPO)

from utils import (  # noqa: E402
    forensics,
    fund_trends as ft,
    funnel,
    funnel_ui,
    industry_board as ib,
    nse_data as nd,
    nse_prices as npx,
    regime_badge as rb,
    regime_manager as rm,
    simba,
    valuation as val_mod,
)
from utils.nifty1000_list import SUB_INDUSTRY_MAP  # noqa: E402

TARGETS = ["ULTRACEMCO.NS", "KALYANKJIL.NS"]
TARGET_GATES = [
    "2026-02-27",
    "2026-03-30",
    "2026-04-30",
    "2026-06-15",
    "2026-07-31",
    "2026-09-15",
    "2026-10-30",
    "2026-11-25",
]


def _f(v: Any, ndig: int = 2) -> Optional[float]:
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    return round(fv, ndig) if math.isfinite(fv) else None


def _fetch_extra_nse_days(extra_dates: List[str]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Fetch recent NSE stock & index bhavcopies in memory (without mutating repo_nov21)."""
    st_extra: Dict[str, pd.DataFrame] = {}
    ix_extra: Dict[str, pd.DataFrame] = {}
    try:
        S = nd.Session()
        for ds in extra_dates:
            day = pd.Timestamp(ds)
            try:
                st, ix, _ = npx._fetch_day_checked(S, day, tries=2)
                if st is not None and len(st):
                    st_extra[ds] = st
                if ix is not None and len(ix):
                    ix_extra[ds] = ix
            except Exception:
                continue
    except Exception:
        pass
    return st_extra, ix_extra


def _append_extra_bars(
    H: Dict[str, pd.DataFrame],
    st_extra: Dict[str, pd.DataFrame],
    ix_extra: Dict[str, pd.DataFrame],
    deliv_map: Dict[str, pd.Series],
) -> None:
    """Append fetched in-memory Nov 23-25 stock & index bars cleanly onto H and deliv_map."""
    for ds in sorted(st_extra.keys()):
        dt = pd.Timestamp(ds)
        df_s = st_extra[ds]
        eq = df_s[df_s["series"].astype(str).str.upper().isin(["EQ", "BE", "BZ"])] if "series" in df_s.columns else df_s
        for tk in TARGETS:
            sym = tk[:-3]
            sub = eq[eq["symbol"].astype(str).str.upper() == sym]
            if len(sub) and dt not in H[tk].index:
                r = sub.iloc[0]
                row = pd.DataFrame(
                    {"Open": [float(r["open"])], "High": [float(r["high"])], "Low": [float(r["low"])],
                     "Close": [float(r["close"])], "Volume": [float(r["volume"])]},
                    index=pd.DatetimeIndex([dt], name="Date"),
                )
                H[tk] = pd.concat([H[tk], row]).sort_index()
                if "deliv_pct" in r and pd.notna(r["deliv_pct"]):
                    deliv_map[tk].loc[dt] = float(r["deliv_pct"])

    ix_name_map = {"^NSEI": "NIFTY 50", "^INDIAVIX": "INDIA VIX"}
    for ds in sorted(ix_extra.keys()):
        dt = pd.Timestamp(ds)
        df_i = ix_extra[ds]
        ix_col = "index" if "index" in df_i.columns else ("symbol" if "symbol" in df_i.columns else None)
        if ix_col is None:
            continue
        for tk, nm in ix_name_map.items():
            sub = df_i[df_i[ix_col].astype(str).str.upper() == nm]
            if len(sub):
                r = sub.iloc[0]
                row = pd.DataFrame(
                    {"Open": [float(r["open"])], "High": [float(r["high"])], "Low": [float(r["low"])],
                     "Close": [float(r["close"])], "Volume": [float(r.get("volume", 0.0) or 0.0)]},
                    index=pd.DatetimeIndex([dt], name="Date"),
                )
                if tk not in H:
                    H[tk] = row
                elif dt not in H[tk].index:
                    H[tk] = pd.concat([H[tk], row]).sort_index()


def _pit_quarterly(tk: str, as_of: pd.Timestamp) -> dict:
    """Extract strictly PIT quarterly results & institutional holdings (45-day SEBI filing lag)."""
    q = ft.quarterly(tk)
    if q is None or q.empty:
        return dict(quarter="", sales_yoy=None, opm=None, opm_chg=None, np_yoy=None,
                    fii=None, dii=None, promoters=None, inst_q=None, inst_y=None, inst_label="NEUTRAL")
    cutoff = as_of - pd.Timedelta(days=45)
    q = q[q.index <= cutoff].sort_index()
    if q.empty:
        return dict(quarter="", sales_yoy=None, opm=None, opm_chg=None, np_yoy=None,
                    fii=None, dii=None, promoters=None, inst_q=None, inst_y=None, inst_label="NEUTRAL")
    latest_dt = q.index[-1]
    q_lbl = latest_dt.strftime("%b %Y")
    prev_y_dt = latest_dt - pd.DateOffset(years=1)
    # match closest quarter ~1 yr ago within 15 days
    y_matches = q.index[abs(q.index - prev_y_dt) <= pd.Timedelta(days=15)]
    y_row = q.loc[y_matches[-1]] if len(y_matches) else None
    cur = q.iloc[-1]

    def _yoy(col: str) -> Optional[float]:
        if y_row is None or col not in q.columns:
            return None
        c_v, y_v = _f(cur.get(col), 4), _f(y_row.get(col), 4)
        if c_v is None or y_v is None or abs(y_v) < 1e-6:
            return None
        if y_v < 0:
            return None
        return round((c_v / y_v - 1.0) * 100.0, 1)

    sales_yoy = _yoy("sales")
    np_yoy = _yoy("np")
    opm = _f(cur.get("opm"), 1) if "opm" in q.columns else None
    opm_prev = _f(y_row.get("opm"), 1) if (y_row is not None and "opm" in q.columns) else None
    opm_chg = round(opm - opm_prev, 1) if (opm is not None and opm_prev is not None) else None

    fii = _f(cur.get("fii"), 2) if "fii" in q.columns else None
    dii = _f(cur.get("dii"), 2) if "dii" in q.columns else None
    prom = _f(cur.get("promoters"), 2) if "promoters" in q.columns else None
    inst_now = (fii + dii) if (fii is not None and dii is not None) else None

    inst_q, inst_y = None, None
    if inst_now is not None and len(q) >= 2:
        pf = _f(q.iloc[-2].get("fii"), 2) if "fii" in q.columns else None
        pd_ = _f(q.iloc[-2].get("dii"), 2) if "dii" in q.columns else None
        if pf is not None and pd_ is not None:
            inst_q = round(inst_now - (pf + pd_), 2)
    if inst_now is not None and y_row is not None:
        yf = _f(y_row.get("fii"), 2) if "fii" in q.columns else None
        yd = _f(y_row.get("dii"), 2) if "dii" in q.columns else None
        if yf is not None and yd is not None:
            inst_y = round(inst_now - (yf + yd), 2)

    inst_label = "NEUTRAL"
    if (inst_q is not None and inst_q >= 1.0 and (inst_y is None or inst_y > -2.0)) or (
        inst_y is not None and inst_y >= 2.0 and (inst_q is None or inst_q > -0.5)
    ):
        inst_label = "ACCUMULATING"
    elif (inst_q is not None and inst_q <= -1.0) or (inst_y is not None and inst_y <= -2.0):
        inst_label = "DISTRIBUTING"

    return dict(
        quarter=q_lbl, sales_yoy=sales_yoy, opm=opm, opm_chg=opm_chg, np_yoy=np_yoy,
        fii=fii, dii=dii, promoters=prom, inst_q=inst_q, inst_y=inst_y, inst_label=inst_label,
    )


def _universe_snapshot_at(pb: dict, dt: pd.Timestamp, st_extra: Dict[str, pd.DataFrame], n50_s: pd.Series) -> pd.DataFrame:
    """Build a PIT Nifty-1000-style snapshot DataFrame at date `dt` for industry_board & breadth."""
    C = pb["close"]
    if dt in C.index:
        c_upto = C.loc[:dt]
    else:
        c_upto = C.loc[:dt]
        # Append extra fetched days up to dt for universe stocks
        for ds in sorted(st_extra.keys()):
            d_i = pd.Timestamp(ds)
            if d_i <= dt and d_i not in c_upto.index:
                df_s = st_extra[ds]
                eq = df_s[df_s["series"].astype(str).str.upper().isin(["EQ", "BE", "BZ"])] if "series" in df_s.columns else df_s
                mp = dict(zip(eq["symbol"].astype(str).str.upper() + ".NS", pd.to_numeric(eq["close"], errors="coerce")))
                row = pd.Series({t: mp.get(t, np.nan) for t in c_upto.columns}, name=d_i)
                c_upto = pd.concat([c_upto, row.to_frame().T])

    # Restrict to Nifty 1000 mapped symbols present in c_upto
    cols = [t for t in c_upto.columns if t in SUB_INDUSTRY_MAP]
    Cu = c_upto[cols].tail(260)
    cur = Cu.iloc[-1]
    ma50 = Cu.tail(50).mean()
    ma200 = Cu.tail(200).mean()
    hi252 = Cu.tail(252).max()
    dist_52w = (cur / hi252 - 1.0) * 100.0

    n_upto = n50_s.loc[:dt]
    def _rs(n_bars: int) -> pd.Series:
        if len(Cu) <= n_bars or len(n_upto) <= n_bars:
            return pd.Series(np.nan, index=cols)
        sr = (cur / Cu.iloc[-1 - n_bars] - 1.0) * 100.0
        nr = (float(n_upto.iloc[-1]) / float(n_upto.iloc[-1 - n_bars]) - 1.0) * 100.0
        return sr - nr

    rs5, rs21, rs63, rs126 = _rs(5), _rs(21), _rs(63), _rs(126)
    comp_rs = 0.10 * rs5 + 0.50 * rs21 + 0.40 * rs63
    dna_sig = np.where((comp_rs > 0) & (cur > ma50), "BUY", np.where(cur < ma50, "AVOID", "HOLD"))

    snap = pd.DataFrame({
        "ticker": cols,
        "currentPrice": cur.values,
        "fiftyDayAverage": ma50.values,
        "twoHundredDayAverage": ma200.values,
        "dist_52w": dist_52w.values,
        "comp_rs": comp_rs.values,
        "rs_1m": rs21.values,
        "rs_3m": rs63.values,
        "rs_6m": rs126.values,
        "dna_signal": dna_sig,
        "extended": False,
    })
    snap = snap[snap["currentPrice"].notna() & (snap["currentPrice"] > 0)].reset_index(drop=True)

    # Gate-1 median vol20 for InvVol20 reference
    log_r = np.log(Cu / Cu.shift(1))
    vol20 = log_r.tail(20).std() * math.sqrt(252.0)
    g1 = (cur > ma50) & (cur > ma200) & (cur >= 0.85 * hi252)
    g1_vmed = float(vol20[g1].dropna().median()) if g1.any() else float(vol20.dropna().median())
    return snap, g1_vmed


def main() -> int:
    t0 = time.time()
    os.makedirs(PKT_DIR, exist_ok=True)
    os.chdir(REPO)

    with open(CACHE7Y, "rb") as f:
        pb = pickle.load(f)

    st_extra, ix_extra = _fetch_extra_nse_days(["2026-11-23", "2026-11-24", "2026-11-25"])

    H = npx.history(TARGETS + ["^NSEI", "^INDIAVIX"], start="2024-01-01")
    deliv_map = {tk: pb["deliv_pct"][tk].dropna().copy() for tk in TARGETS if tk in pb["deliv_pct"].columns}
    _append_extra_bars(H, st_extra, ix_extra, deliv_map)

    n50_df = H["^NSEI"].sort_index()
    n50_s = n50_df["Close"]
    vix_s = pb["indices"]["vix"].dropna().sort_index().copy() if "vix" in pb.get("indices", {}) else pd.Series(dtype=float)
    if "^INDIAVIX" in H and len(H["^INDIAVIX"]):
        for dt_v, val_v in H["^INDIAVIX"]["Close"].items():
            vix_s.loc[dt_v] = float(val_v)
        vix_s = vix_s.sort_index()

    # Market context (Nifty PE, FII idx long %)
    npe_s = pb["nifty_pe"].sort_index().copy()
    for ds_x, df_ix in sorted(ix_extra.items()):
        if "index" in df_ix.columns and "pe" in df_ix.columns:
            m_n50 = df_ix[df_ix["index"].astype(str).str.upper() == "NIFTY 50"]
            if len(m_n50) and pd.notna(m_n50.iloc[0]["pe"]):
                npe_s.loc[pd.Timestamp(ds_x)] = float(m_n50.iloc[0]["pe"])
    npe_s = npe_s.sort_index()
    fii_s = pb["fii_idx_long_pct"].sort_index()

    # Snap checkpoints to exact available trading dates <= target
    all_dates = H[TARGETS[0]].index.intersection(H[TARGETS[1]].index).intersection(n50_s.index).sort_values()
    gate_dates: List[pd.Timestamp] = []
    for gd in TARGET_GATES:
        ts = pd.Timestamp(gd)
        avail = all_dates[all_dates <= ts]
        if len(avail):
            d_use = avail[-1]
            if not gate_dates or d_use > gate_dates[-1]:
                gate_dates.append(d_use)

    val_ctx = val_mod.load_context()

    # Precompute daily forensics + MA50 + Gate1 series over [2026-02-27 .. end] for sealed sim & inter-gate events
    daily_traces: Dict[str, List[dict]] = {tk: [] for tk in TARGETS}
    march_days = all_dates[all_dates >= gate_dates[0]]
    for tk in TARGETS:
        df_tk = H[tk]
        c_all = df_tk["Close"]
        ma50_all = c_all.rolling(50).mean()
        ma200_all = c_all.rolling(200).mean()
        hi252_all = c_all.rolling(252).max()
        for dt in march_days:
            sub = df_tk.loc[:dt]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fx = forensics.analyse(sub)
            px = float(sub["Close"].iloc[-1])
            m50 = float(ma50_all.loc[dt]) if pd.notna(ma50_all.loc[dt]) else np.nan
            m200 = float(ma200_all.loc[dt]) if pd.notna(ma200_all.loc[dt]) else np.nan
            h252 = float(hi252_all.loc[dt]) if pd.notna(hi252_all.loc[dt]) else np.nan
            g1 = bool(np.isfinite(m50) and np.isfinite(m200) and np.isfinite(h252) and px > m50 and px > m200 and px >= 0.85 * h252)
            daily_traces[tk].append(dict(
                date=str(dt.date()),
                open=round(float(sub["Open"].iloc[-1]), 2),
                high=round(float(sub["High"].iloc[-1]), 2),
                low=round(float(sub["Low"].iloc[-1]), 2),
                close=round(px, 2),
                volume=float(sub["Volume"].iloc[-1]),
                ma50=round(m50, 2) if np.isfinite(m50) else None,
                ma200=round(m200, 2) if np.isfinite(m200) else None,
                gate1=g1,
                fx_action=str(fx.get("action") or ""),
                fx_state=str(fx.get("state") or ""),
                fx_supply=str(fx.get("supply") or ""),
                fx_signature=str(fx.get("signature") or ""),
                fx_shakeout=bool(fx.get("shakeout", False)),
                fx_stop=_f(fx.get("stop"), 2),
                fx_stop_pct=_f(fx.get("stop_pct"), 2),
                fx_entry=_f(fx.get("entry"), 2),
            ))

    sealed = dict(
        gate_dates=[str(d.date()) for d in gate_dates],
        daily_traces=daily_traces,
        nifty={str(dt.date()): round(float(n50_s.loc[dt]), 2) for dt in march_days},
    )
    with open(os.path.join(OUT_DIR, "sealed_daily.pkl"), "wb") as f:
        pickle.dump(sealed, f)

    # Build strict PIT packets per gate
    prev_dt: Optional[pd.Timestamp] = None
    manifest = []
    for k_idx, dt in enumerate(gate_dates, 1):
        ds = str(dt.date())
        snap_df, g1_vmed = _universe_snapshot_at(pb, dt, st_extra, n50_s)
        board_df = ib.board(snap_df)
        n50_upto = n50_df.loc[:dt]
        reg_info = rb.compute(n50_upto, snap_df)

        # Macro values strictly <= dt
        vix_val = _f(vix_s.loc[:dt].iloc[-1], 2) if len(vix_s.loc[:dt]) else None
        npe_val = _f(npe_s.loc[:dt].dropna().iloc[-1], 2) if len(npe_s.loc[:dt].dropna()) else None
        fii_upto = fii_s.loc[:dt].dropna()
        fii_val = _f(fii_upto.iloc[-1], 1) if len(fii_upto) else None
        fii_p126 = _f(float((fii_upto.tail(126) <= fii_upto.iloc[-1]).mean() * 100.0), 0) if len(fii_upto) >= 20 else None

        pct_above_50 = _f(float((snap_df["currentPrice"] > snap_df["fiftyDayAverage"]).mean() * 100.0), 1)
        pct_above_200 = _f(reg_info["breadth"].get("pct"), 1)
        eng_mode = reg_info["engine"].get("mode", "UNKNOWN")
        eng_params = rm.get_regime_params(eng_mode) if eng_mode in rm.REGIME_PARAMS else {}

        market_ctx = dict(
            as_of=ds,
            nifty_close=_f(n50_upto["Close"].iloc[-1], 1),
            nifty_vs_200dma_pct=_f(reg_info["nifty"].get("vs_ma200"), 2),
            nifty_dd_from_52w_pct=_f(reg_info["nifty"].get("dd"), 2),
            nifty_regime=reg_info["nifty"].get("regime"),
            breadth_above_50dma_pct=pct_above_50,
            breadth_above_200dma_pct=pct_above_200,
            breadth_regime=reg_info["breadth"].get("regime"),
            engine_mode=eng_mode,
            engine_trail_stop_pct=_f(eng_params.get("trail_stop", 0.15) * 100.0, 0),
            india_vix=vix_val,
            nifty_pe=npe_val,
            fii_idx_futures_long_pct=fii_val,
            fii_idx_126d_pctile=fii_p126,
        )

        stocks_pkt = {}
        for tk in TARGETS:
            sub = H[tk].loc[:dt]
            px = float(sub["Close"].iloc[-1])
            ma50 = float(sub["Close"].tail(50).mean())
            ma200 = float(sub["Close"].tail(200).mean())
            hi52 = float(sub["High"].tail(252).max())
            lo52 = float(sub["Low"].tail(252).min())
            d_ma50 = (px / ma50 - 1.0) * 100.0
            d_ma200 = (px / ma200 - 1.0) * 100.0
            d_52w = (px / hi52 - 1.0) * 100.0

            def _tk_rs(n_b: int) -> Optional[float]:
                if len(sub) <= n_b or len(n50_upto) <= n_b:
                    return None
                return round(((px / float(sub["Close"].iloc[-1 - n_b]) - 1.0) - (float(n50_upto["Close"].iloc[-1]) / float(n50_upto["Close"].iloc[-1 - n_b]) - 1.0)) * 100.0, 2)

            rs1w, rs1m, rs3m, rs6m = _tk_rs(5), _tk_rs(21), _tk_rs(63), _tk_rs(126)
            comp_rs = round(0.10 * (rs1w or 0) + 0.50 * (rs1m or 0) + 0.40 * (rs3m or 0), 2)
            dna_sig = "BUY" if (comp_rs > 0 and px > ma50) else ("AVOID" if px < ma50 else "HOLD")
            gate1 = bool(px > ma50 and px > ma200 and px >= 0.85 * hi52)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fx = forensics.analyse(sub)
                sb = simba.chart_signals(sub, bench=n50_upto["Close"])

            ind_name = SUB_INDUSTRY_MAP.get(tk, "Unclassified")
            ind_row = board_df[board_df["industry"] == ind_name]
            if len(ind_row):
                ir = ind_row.iloc[0]
                ind_rank = int(board_df.index[board_df["industry"] == ind_name][0]) + 1
                ind_info = dict(
                    industry=ind_name, rank=ind_rank, of=len(board_df),
                    score=_f(ir["score"], 1), group=str(ir["group"]), status=str(ir["status"]),
                    above_50_pct=_f(ir["above_50"], 1), above_200_pct=_f(ir["above_200"], 1),
                )
            else:
                ind_info = dict(industry=ind_name, rank=None, of=len(board_df), score=None, group="", status="", above_50_pct=None, above_200_pct=None)

            # Microstructure
            log_r = np.log(sub["Close"] / sub["Close"].shift(1)).dropna()
            vol20 = float(log_r.tail(20).std() * math.sqrt(252.0)) if len(log_r) >= 15 else np.nan
            w_mult = float(np.clip(g1_vmed / vol20, 0.50, 1.50)) if (np.isfinite(vol20) and vol20 > 0 and np.isfinite(g1_vmed)) else 1.0
            liq_cr = float((sub["Close"] * sub["Volume"]).tail(20).median() / 1e7)
            dp_s = deliv_map.get(tk, pd.Series(dtype=float)).loc[:dt].dropna()
            dp_now = _f(dp_s.iloc[-1], 1) if len(dp_s) else None
            dp_avg20 = _f(dp_s.tail(20).mean(), 1) if len(dp_s) >= 5 else None
            dp_ratio = _f(dp_now / dp_avg20, 2) if (dp_now and dp_avg20 and dp_avg20 > 0) else None

            # PIT Fundamentals (60d annual lag + 45d quarterly lag)
            ann_asof = dt - pd.Timedelta(days=60)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fr = ft.read(tk, as_of=ann_asof)
            qp = _pit_quarterly(tk, dt)
            biz_ok = bool(fr.get("score") is not None and float(fr["score"]) >= 45.0 and not fr.get("hard_flags"))
            biz_flags = "; ".join(fr.get("hard_flags") or [])

            # PIT Valuation v4.1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vr = val_mod.value_stock(tk, val_ctx, price=px, n=3000, hi52=hi52, lo52=lo52, as_of=dt)
            v_ok = vr.get("status") == "OK"
            v_dist = vr.get("dist") or {}
            v_p50 = _f(v_dist.get("P50"), 1) if v_ok else None
            v_upside = _f((v_p50 / px - 1.0) * 100.0, 1) if (v_p50 and px > 0) else None
            v_pe = _f((vr.get("snapshot") or {}).get("spot_pe"), 1)
            v_lad = vr.get("ladder") or {}
            val_info = dict(
                status=vr.get("status"),
                rating=vr.get("rating", ""),
                cmp_pct=_f(v_dist.get("cmp_pct"), 1) if v_ok else None,
                p25=_f(v_dist.get("P25"), 1) if v_ok else None,
                p50=v_p50,
                p75=_f(v_dist.get("P75"), 1) if v_ok else None,
                to_p50_pct=v_upside,
                pe=v_pe,
                roce_pct=_f(((vr.get("snapshot") or {}).get("roce") or 0) * 100.0, 1) if (vr.get("snapshot") or {}).get("roce") is not None else None,
                de=_f((vr.get("snapshot") or {}).get("de"), 2),
                firmness=(vr.get("band") or {}).get("firmness", ""),
                ladder=dict(
                    accumulate=_f(v_lad.get("accumulate"), 1),
                    fair_hi=_f(v_lad.get("fair_hi"), 1),
                    trim=_f(v_lad.get("trim"), 1),
                ) if v_lad else {},
            )

            # PIT Funnel v3 score & Signal Strip chips
            sb_stage = sb.get("simba_stage")
            sb_mtf = sb.get("simba_mtf")
            sb_m_since = sb.get("simba_m_since")
            sb_si_since = sb.get("simba_si_since")
            s_ind = (ind_info["group"] in ("LEADING", "IMPROVING") or ind_info["status"] in ("UPTREND", "TURNING UP")) and ind_info["group"] != "WEAKENING"
            s_trd = (dna_sig == "BUY") and (sb_stage not in (3, 4))
            fx_stop = _f(fx.get("stop"), 2)
            fx_ent = _f(fx.get("entry"), 2) or px
            rk_pct = _f((fx_ent - fx_stop) / fx_ent * 100.0, 2) if (fx_stop and fx_ent and 0 < fx_stop < min(fx_ent, px)) else None
            simba_days = min([d for d in (sb_m_since, sb_si_since) if d is not None and np.isfinite(d)], default=np.nan)
            sig_ok = (str(fx.get("action") or "") in ("BUY NOW", "BUY ON TRIGGER")) or (np.isfinite(simba_days) and 0 <= simba_days <= 5)
            s_set = bool(sig_ok and rk_pct is not None and rk_pct <= 15.0)
            s_val = (val_info["cmp_pct"] is None) or (val_info["cmp_pct"] <= 70.0)

            f_row = pd.DataFrame([dict(
                ticker=tk, name=tk[:-3], industry=ind_name,
                ind_group=ind_info["group"], ind_score=ind_info["score"],
                price=px, comp_rs=comp_rs, dist_52w=d_52w, dist_ma50=d_ma50, dist_200dma=d_ma200,
                technicals=str(fx.get("action") or ""), fx_supply=str(fx.get("supply") or ""),
                fx_signature=str(fx.get("signature") or ""), fx_shakeout=bool(fx.get("shakeout", False)),
                fx_adx=_f(fx.get("adx"), 1), fx_pdi=_f(fx.get("pdi"), 1), fx_ndi=_f(fx.get("ndi"), 1),
                fx_bbp=_f(fx.get("bbp"), 1), fx_gap=bool(fx.get("gap", False)),
                simba_stage=sb_stage, simba_mtf=sb_mtf, simba_days=simba_days,
                biz_score=_f(fr.get("score"), 1), biz_flags=biz_flags,
                sales_yoy=qp["sales_yoy"], opm_chg=qp["opm_chg"], np_yoy=qp["np_yoy"],
                risk_pct=rk_pct, liq_cr=round(liq_cr, 1), pe=v_pe,
                val_rating=val_info["rating"], val_pct=val_info["cmp_pct"], val_loss=False,
                nse_flags="", inst_label=qp["inst_label"], inst_q=qp["inst_q"], inst_y=qp["inst_y"],
                deal_inst_cr=np.nan, deliv_pct=dp_now, deliv_avg20=dp_avg20,
                trigger_gap_pct=_f((fx_ent / px - 1.0) * 100.0, 2) if (fx_ent and px > 0) else 0.0,
            )])
            f_scored = funnel.score_table(f_row).iloc[0]
            chips = funnel_ui.build_signal_chips(f_scored)

            # Inter-gate causal events strictly inside (prev_dt, dt]
            events = []
            if prev_dt is not None:
                prev_ds = str(prev_dt.date())
                win = [x for x in daily_traces[tk] if prev_ds < x["date"] <= ds]
                prev_act = None
                prev_above50 = None
                for x in win:
                    if x["fx_signature"] == "ABSORPTION":
                        events.append(f"{x['date']}: ⭐ ABSORPTION (close {x['close']}, action={x['fx_action']})")
                    if x["fx_shakeout"]:
                        events.append(f"{x['date']}: 🪤 SHAKEOUT (close {x['close']})")
                    if x["fx_action"] in ("BUY NOW", "BUY ON TRIGGER") and x["fx_action"] != prev_act:
                        events.append(f"{x['date']}: Technicals -> {x['fx_action']} (close {x['close']}, stop {x['fx_stop']})")
                    if x["ma50"] is not None:
                        ab50 = x["close"] > x["ma50"]
                        if prev_above50 is not None and ab50 != prev_above50:
                            events.append(f"{x['date']}: {'Crossed ABOVE' if ab50 else 'Broke BELOW'} 50-DMA ({x['close']} vs {x['ma50']})")
                        prev_above50 = ab50
                    prev_act = x["fx_action"]
                # Cap verbose daily repeats cleanly
                events = events[-10:]

            stocks_pkt[tk] = dict(
                ticker=tk,
                price=round(px, 2),
                ma50=round(ma50, 2),
                ma200=round(ma200, 2),
                vs_50dma_pct=round(d_ma50, 2),
                vs_200dma_pct=round(d_ma200, 2),
                vs_52w_high_pct=round(d_52w, 2),
                rs=dict(rs1w=rs1w, rs1m=rs1m, rs3m=rs3m, rs6m=rs6m, comp_rs=comp_rs),
                dna_signal=dna_sig,
                gate1=gate1,
                industry=ind_info,
                forensics=dict(
                    action=str(fx.get("action") or ""),
                    state=str(fx.get("state") or ""),
                    supply=str(fx.get("supply") or ""),
                    signature=str(fx.get("signature") or ""),
                    shakeout=bool(fx.get("shakeout", False)),
                    adx=_f(fx.get("adx"), 1),
                    pdi=_f(fx.get("pdi"), 1),
                    ndi=_f(fx.get("ndi"), 1),
                    rsi=_f(fx.get("rsi"), 1),
                    bbp=_f(fx.get("bbp"), 1),
                    gap=bool(fx.get("gap", False)),
                    entry=fx_ent,
                    stop=fx_stop,
                    stop_pct=rk_pct,
                    half_size=bool(rk_pct is not None and rk_pct > 10.0),
                    t1_2r=_f(fx.get("t1"), 2),
                ),
                simba=dict(
                    stage=sb_stage,
                    mtf=sb_mtf,
                    g_active=bool(sb.get("simba_g", False)),
                    m_active=bool(sb.get("simba_m", False)),
                    m_since=_f(sb_m_since, 0),
                    si_since=_f(sb_si_since, 0),
                    mrs=_f(sb.get("simba_mrs"), 2),
                    vcp=bool(sb.get("simba_vcp", False)),
                    caution=sb.get("simba_caution"),
                    dist_days=sb.get("simba_dist_days"),
                    trail_ema=sb.get("simba_trail"),
                ),
                microstructure=dict(
                    deliv_pct=dp_now,
                    deliv_avg20=dp_avg20,
                    deliv_ratio=dp_ratio,
                    vol20_ann_pct=_f(vol20 * 100.0, 1) if np.isfinite(vol20) else None,
                    invvol20_w_mult=round(w_mult, 3),
                    liq_cr_day=round(liq_cr, 1),
                ),
                business_pit=dict(
                    verdict=fr.get("verdict"),
                    score=_f(fr.get("score"), 1),
                    headline=fr.get("headline", ""),
                    hard_flags=fr.get("hard_flags", []),
                    quarterly=qp,
                ),
                valuation_pit=val_info,
                funnel_v3=dict(
                    stages=dict(industry=s_ind, trend=s_trd, business=biz_ok, setup=s_set, value=s_val),
                    passes_default_4=bool(s_ind and s_trd and biz_ok and s_set),
                    score=_f(f_scored["funnel_score"], 1),
                    grade=str(f_scored["funnel_grade"]),
                    score_note=str(f_scored["score_note"]),
                    pros=[c["text"] for c in chips["pros"]],
                    cons=[c["text"] for c in chips["cons"]],
                ),
                inter_gate_events=events,
            )

        pkt = dict(gate_num=k_idx, gate_date=ds, market=market_ctx, stocks=stocks_pkt)
        pkt_path = os.path.join(PKT_DIR, f"gate_{k_idx:02d}_{ds}.json")
        with open(pkt_path, "w", encoding="utf-8") as f:
            json.dump(pkt, f, indent=2)
        manifest.append(dict(gate_num=k_idx, date=ds, path=pkt_path))
        prev_dt = dt

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(dict(gates=manifest, extra_nse_days=sorted(st_extra.keys()), elapsed_sec=round(time.time() - t0, 1)), f, indent=2)

    print(f"PIT Dossiers built in {time.time() - t0:.1f}s | gates={len(manifest)} ({manifest[0]['date']} .. {manifest[-1]['date']}) | extra_nse_days={sorted(st_extra.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
