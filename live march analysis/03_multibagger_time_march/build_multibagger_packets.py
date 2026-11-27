"""Build strictly point-in-time (<= t) dashboard packets & sealed daily traces for the 11 NSE 2026 multibaggers.

Runs on the upgraded repo_nov21 engine (P0/P1 fixes: forensics v6+fixes, RECOVERY breadth-thrust,
valuation <Bear provisional + FCF/PAT, live VIX, 21d/45d split SEBI lags, abs_deliv_cr) AND records
pre-fix forensics/regime flags side-by-side for before-vs-after attribution.
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

REPO = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\repo_nov21"
OUT_DIR = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\multibagger_march"
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(REPO)
sys.path.insert(0, REPO)

from utils import forensics as fx, fund_trends as ft, nse_prices, regime_badge as rb, regime_manager as rm, valuation as V

TARGETS = [
    "SIGMAADV.NS",
    "CUPID.NS",
    "STLTECH.NS",
    "TBZ.NS",
    "MONOLITH.NS",
    "SPECTRUM.NS",
    "HFCL.NS",
    "FILATEX.NS",
    "SSWL.NS",
    "LAURUSLABS.NS",
    "MACPOWER.NS",
]
START_DATE = "2026-02-27"
END_DATE = "2026-11-20"

MONTHLY_GATES = [
    "2026-02-27",
    "2026-03-30",
    "2026-04-30",
    "2026-05-29",
    "2026-06-30",
    "2026-07-31",
    "2026-08-31",
    "2026-09-30",
    "2026-10-30",
]

# Causal event dates (strictly triggered by real-time dashboard milestones <= t):
# E1: Launch | E2: Bear Capitulation (VIX 27.9, 52W lows 44%) | E3: First RECOVERY Breadth-Thrust Day (Apr 8, +33.6pp)
# E4: Post-Q4 FY26 Live (May 18) | E5: Mid-Year Breakout Surge (Jun 18) | E6: Jul Momentum Wave (Jul 20)
# E7: Post-Q1 / Sep Pullback (Sep 15) | E8: Oct Capitulation Rollover / Nov Breakout Thrust (Nov 6)
CAUSAL_EVENTS = [
    ("E1", "2026-02-27", "Launch"),
    ("E2", "2026-03-30", "Bear Capitulation Washout (52W lows 44%, VIX spike)"),
    ("E3", "2026-04-08", "RECOVERY Breadth-Thrust Trigger (+33.6pp >50DMA, lows -99%)"),
    ("E4", "2026-05-18", "Post-Q4 FY26 Filings Live (45d Reg 33)"),
    ("E5", "2026-06-22", "Summer Multi-Stock Breakout Cluster"),
    ("E6", "2026-07-20", "Mid-Year Trend Expansion & Q1 SHP Live (21d Reg 31)"),
    ("E7", "2026-09-15", "Autumn Pullback & Squeeze Setup"),
    ("E8", "2026-11-06", "Post-Oct Rollover & Nov Breakout Thrust (Sep SHP Live)"),
]


def _load_indices():
    files = sorted(glob.glob(os.path.join(REPO, "data", "nse", "indices", "*.parquet")))
    parts = [pd.read_parquet(f, columns=["date", "index", "open", "high", "low", "close", "pe"]) for f in files[-24:]]
    ix = pd.concat(parts, ignore_index=True)
    ix["date"] = pd.to_datetime(ix["date"])
    nifty = ix[ix["index"] == "Nifty 50"].drop_duplicates("date", keep="last").set_index("date").sort_index()
    nifty = nifty.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    vix = ix[ix["index"] == "India VIX"].drop_duplicates("date", keep="last").set_index("date").sort_index()["close"]
    return nifty, vix


def _load_breadth_panels():
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


def _load_delivery():
    try:
        from utils import nse_data as nd
        d = nd.delivery()
        if d is not None and not d.empty:
            cols = [t for t in TARGETS if t in d.columns]
            if cols:
                out = d[cols].copy()
                out.columns = [c.replace(".NS", "") for c in cols]
                return out.sort_index()
    except Exception:
        pass
    p = os.path.join(REPO, "data", "nse", "delivery.parquet")
    if os.path.exists(p):
        d = pd.read_parquet(p)
        if "symbol" in d.columns:
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            d = d[d["symbol"].isin([t.replace(".NS", "") for t in TARGETS])]
            return d.pivot_table(index="date", columns="symbol", values="deliv_pct", aggfunc="last").sort_index()
    return pd.DataFrame()


def _legacy_forensics_action(df_sub: pd.DataFrame) -> dict:
    """Pre-fix forensics (flat 3x vol, CLV-blind argmax pick_anchors, no ATR/CLV/DIST guard) for attribution."""
    if df_sub is None or len(df_sub) < 60:
        return {"state": "NO DATA", "action": "WATCH"}
    d = df_sub[["High", "Low", "Close", "Volume"]].astype(float).dropna(subset=["Close"])
    d = d[d["Close"] > 0]
    if len(d) < 60:
        return {"state": "NO DATA", "action": "WATCH"}
    c, h, l, v = (d[k].values for k in ("Close", "High", "Low", "Volume"))
    i = len(c) - 1
    ma50 = d.Close.rolling(50).mean().values
    ma200 = d.Close.rolling(200).mean().values
    atr = fx.indicators(d.Close, d.High, d.Low)[0].values
    vmed = d.Volume.rolling(50, min_periods=30).median().shift(1).values
    clv = fx._clv_arr(c, h, l)
    hi = np.nanmax(c[-252:])
    g1 = bool(c[i] > ma50[i] and np.isfinite(ma200[i]) and c[i] > ma200[i] and c[i] >= 0.85 * hi)
    qual = fx.qualifying(c, v, vmed, 5e7)  # legacy flat 3x
    a6, _ = fx.pick_anchors(qual, v, i)    # legacy CLV-blind
    if a6 is None:
        state = "NO ANCHOR"
    else:
        g, _ = fx.grade(c, h, l, v, vmed, ma50, atr, clv, a6, i)
        state = g["state"]
    sup = fx.supply_at(c, c * v, atr[i], i).get("supply", "NOT COMPUTED")
    return {"gate1": g1, "state": state, "action": fx.action_for(g1, state, sup)}


def _pit_valuation(tk: str, as_of: str, px: float, ctx: dict, ann_raw: dict) -> dict:
    """Strictly causal (<= as_of) valuation using repo_nov21 upgraded Valuation v4.1 + exact FY CFO/PAT & FCF/PAT."""
    sym = tk.replace(".NS", "")
    fcf_pat = cfo_pat = roce = np.nan
    fy_lbl = ""
    k = ft._key(tk, ann_raw)
    if k is not None:
        sub = ann_raw[k]
        w = sub.pivot_table(index="period", columns="field", values="value", aggfunc="last").sort_index()
        w = w[w.index.month == 3]  # full fiscal-year rows only
        w = w[w.index + pd.Timedelta(days=60) <= pd.Timestamp(as_of)]
        if not w.empty:
            fy_lbl = pd.Timestamp(w.index[-1]).strftime("FY%y")
            g = lambda c: w[c] if c in w.columns else pd.Series(np.nan, index=w.index)
            gross = g("fixed_assets") + g("cwip").fillna(0.0)
            capex_s = gross.diff() + g("dep").fillna(0.0)
            np_v = float(g("np").iloc[-1]) if pd.notna(g("np").iloc[-1]) else np.nan
            cfo_v = float(g("cfo").iloc[-1]) if pd.notna(g("cfo").iloc[-1]) else np.nan
            cap_v = float(capex_s.iloc[-1]) if pd.notna(capex_s.iloc[-1]) else np.nan
            roce_v = g("roce").iloc[-1]
            if pd.notna(roce_v):
                roce = round(float(roce_v), 1)
            if np.isfinite(np_v) and np_v > 0 and np.isfinite(cfo_v):
                cfo_pat = round(cfo_v / np_v, 2)
                if np.isfinite(cap_v):
                    fcf_pat = round((cfo_v - max(0.0, cap_v)) / np_v, 2)

    res = {"fy": fy_lbl, "roce": roce, "cfo_pat": cfo_pat, "fcf_pat": fcf_pat,
           "rating": "NO DATA", "verdict_raw": "", "cmp_pct": np.nan, "bear_blend": np.nan, "base_blend": np.nan, "p50": np.nan}
    try:
        v = V.value_stock(sym, ctx, price=px, as_of=as_of, knobs=False)
        if v.get("status") == "OK":
            bl = v.get("blends") or {}
            res.update(
                rating=v.get("rating", ""),
                verdict_raw=v.get("verdict_raw", ""),
                cmp_pct=round(float(v["dist"]["cmp_pct"]), 1),
                bear_blend=round(float(bl.get("bear", np.nan)), 1) if np.isfinite(bl.get("bear", np.nan)) else np.nan,
                base_blend=round(float(bl.get("base", np.nan)), 1) if np.isfinite(bl.get("base", np.nan)) else np.nan,
                p50=round(float(v["dist"]["P50"]), 1),
                cmp_lt_bear=bool(np.isfinite(bl.get("bear", np.nan)) and px <= float(bl["bear"])),
            )
    except Exception:
        pass
    return res


def _comp_rs(stock_c: pd.Series, nifty_c: pd.Series) -> tuple[float, float, float]:
    """Causal 1M / 3M / Composite RS vs Nifty."""
    df = pd.concat([stock_c.rename("s"), nifty_c.rename("n")], axis=1).dropna()
    if len(df) < 25:
        return np.nan, np.nan, np.nan
    def _rs(w):
        if len(df) <= w:
            return np.nan
        return ((df["s"].iloc[-1] / df["s"].iloc[-1 - w]) / (df["n"].iloc[-1] / df["n"].iloc[-1 - w]) - 1.0) * 100.0
    r1m, r3m, r6m = _rs(21), _rs(63), _rs(126)
    vals = [x for x in (r1m, r3m, r6m) if np.isfinite(x)]
    comp = float(0.25 * (r1m if np.isfinite(r1m) else 0) + 0.35 * (r3m if np.isfinite(r3m) else 0) + 0.40 * (r6m if np.isfinite(r6m) else 0)) if vals else np.nan
    return (round(r1m, 1) if np.isfinite(r1m) else np.nan,
            round(r3m, 1) if np.isfinite(r3m) else np.nan,
            round(comp, 1) if np.isfinite(comp) else np.nan)


def main():
    print("Loading NSE prices, indices, breadth, delivery, market context, valuation context...")
    hist = nse_prices.history(TARGETS)
    nifty_df, vix_s = _load_indices()
    breadth = _load_breadth_panels()
    deliv = _load_delivery()
    mc = pd.read_csv(os.path.join(REPO, "data", "nse", "market_context.csv"))
    mc["date"] = pd.to_datetime(mc["date"], errors="coerce")
    mc = mc.dropna(subset=["date"]).set_index("date").sort_index()
    val_ctx = V.load_context()
    ann_raw, _ = ft._data(REPO)

    # Common trading calendar from START_DATE..END_DATE
    ref_df = hist["LAURUSLABS.NS"]
    dates = [d.strftime("%Y-%m-%d") for d in ref_df.loc[START_DATE:END_DATE].index]
    print(f"Trading window: {dates[0]} -> {dates[-1]} ({len(dates)} sessions)")

    # 1. Build sealed daily traces + before-vs-after daily signal counts across all 11 multibaggers
    daily_traces: Dict[str, List[dict]] = {}
    daily_regimes: Dict[str, dict] = {}
    upgrade_stats = {tk: {"old_buy_now": 0, "new_buy_now": 0, "new_go": 0, "old_go": 0, "new_blocked_ext": 0, "new_first_go": None, "old_first_go": None} for tk in TARGETS}

    for dt_str in dates:
        dt = pd.Timestamp(dt_str)
        nd = nifty_df.loc[:dt]
        c_last = float(nd["Close"].iloc[-1])
        m200 = float(nd["Close"].rolling(200).mean().iloc[-1])
        hi52 = float(nd["High"][nd.index > dt - pd.Timedelta(days=365)].max())
        base_mode = rm.classify_regime(c_last, m200, hi52)
        b_sub = breadth.loc[:dt]
        mc_sub = mc.loc[:dt]
        fii_s = mc_sub["fii_net"].tail(3) if mc_sub["fii_net"].tail(3).notna().any() else None
        dii_s = mc_sub["dii_net"].tail(3) if mc_sub["dii_net"].tail(3).notna().any() else None
        rec = rm.detect_capitulation_recovery(
            b_sub["pct50"] if len(b_sub) else None,
            b_sub["pctlo"] if len(b_sub) else None,
            fii_s, dii_s,
            mc_sub["fii_idx_long_pct"] if len(mc_sub) else None,
        )
        new_mode = rm.classify_regime(c_last, m200, hi52, recovery=rec)
        v_sub = vix_s.loc[:dt]
        daily_regimes[dt_str] = {
            "date": dt_str,
            "nifty_close": round(c_last, 1),
            "nifty_pe": round(float(mc_sub["nifty_pe"].iloc[-1]), 1) if len(mc_sub) and pd.notna(mc_sub["nifty_pe"].iloc[-1]) else np.nan,
            "base_mode": base_mode,
            "engine_mode": new_mode,
            "recovery_active": rec["active"],
            "recovery_reasons": rec["reasons"],
            "breadth_50": round(float(b_sub["pct50"].iloc[-1]), 1) if len(b_sub) and pd.notna(b_sub["pct50"].iloc[-1]) else np.nan,
            "breadth_200": round(float(b_sub["pct200"].iloc[-1]), 1) if len(b_sub) and pd.notna(b_sub["pct200"].iloc[-1]) else np.nan,
            "lows_52w": round(float(b_sub["pctlo"].iloc[-1]), 1) if len(b_sub) and pd.notna(b_sub["pctlo"].iloc[-1]) else np.nan,
            "vix": round(float(v_sub.iloc[-1]), 2) if len(v_sub) else np.nan,
            "fii_long_pct": round(float(mc_sub["fii_idx_long_pct"].iloc[-1]), 1) if len(mc_sub) and pd.notna(mc_sub["fii_idx_long_pct"].iloc[-1]) else np.nan,
        }

    for tk in TARGETS:
        df = hist[tk]
        ma50_s = df["Close"].rolling(50).mean()
        ma200_s = df["Close"].rolling(200).mean()
        t_rows = []
        for dt_str in dates:
            if pd.Timestamp(dt_str) not in df.index:
                # Forward-fill last known bar if holiday/missing
                sub = df.loc[:dt_str]
                row = sub.iloc[-1]
                m50 = float(ma50_s.loc[:dt_str].iloc[-1]) if pd.notna(ma50_s.loc[:dt_str].iloc[-1]) else None
                m200 = float(ma200_s.loc[:dt_str].iloc[-1]) if pd.notna(ma200_s.loc[:dt_str].iloc[-1]) else None
            else:
                sub = df.loc[:dt_str]
                row = df.loc[dt_str]
                m50 = float(ma50_s.loc[dt_str]) if pd.notna(ma50_s.loc[dt_str]) else None
                m200 = float(ma200_s.loc[dt_str]) if pd.notna(ma200_s.loc[dt_str]) else None
            t_rows.append({
                "date": dt_str,
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": float(row["Volume"]),
                "ma50": round(m50, 2) if m50 is not None else None,
                "ma200": round(m200, 2) if m200 is not None else None,
            })
            # Check old vs new forensics on every session
            r_new = fx.analyse(sub)
            r_old = _legacy_forensics_action(sub)
            if r_old.get("state") == "GO":
                upgrade_stats[tk]["old_go"] += 1
                if upgrade_stats[tk]["old_first_go"] is None:
                    upgrade_stats[tk]["old_first_go"] = (dt_str, round(float(row["Close"]), 1))
            if r_old.get("action") == "BUY NOW":
                upgrade_stats[tk]["old_buy_now"] += 1
            if r_new.get("state") == "GO":
                upgrade_stats[tk]["new_go"] += 1
                if upgrade_stats[tk]["new_first_go"] is None:
                    upgrade_stats[tk]["new_first_go"] = (dt_str, round(float(row["Close"]), 1))
            if r_new.get("action") == "BUY NOW":
                upgrade_stats[tk]["new_buy_now"] += 1
            if r_new.get("state") == "GO" and r_new.get("entry_weak"):
                upgrade_stats[tk]["new_blocked_ext"] += 1
        daily_traces[tk] = t_rows

    # 2. Build causal point-in-time packets for ALL unique checkpoint dates (Monthly + Causal Events)
    all_checkpoints = sorted(set(MONTHLY_GATES + [d for _, d, _ in CAUSAL_EVENTS]))
    packets: Dict[str, dict] = {}
    for dt_str in all_checkpoints:
        # Snap to nearest valid trading date <= dt_str
        valid_dt = max(d for d in dates if d <= dt_str)
        dt = pd.Timestamp(valid_dt)
        im_pit = ft.inst_map(root=REPO, as_of=valid_dt)
        stocks_pkt = {}
        for tk in TARGETS:
            sym = tk.replace(".NS", "")
            sub = hist[tk].loc[:valid_dt]
            px = float(sub["Close"].iloc[-1])
            r_fx = fx.analyse(sub)
            r_old = _legacy_forensics_action(sub)
            r1m, r3m, crs = _comp_rs(sub["Close"], nifty_df.loc[:valid_dt, "Close"])
            tmed_cr = float((sub["Close"] * sub["Volume"]).tail(20).median() / 1e7)
            dp = da = abs_d = np.nan
            if sym in deliv.columns:
                ds = deliv[sym].loc[:valid_dt].dropna()
                if len(ds):
                    dp = round(float(ds.iloc[-1]), 1)
                    da = round(float(ds.tail(20).mean()), 1)
                    abs_d = round(tmed_cr * dp / 100.0, 1)
            q_pit = ft.quarterly(tk, root=REPO, as_of=valid_dt)
            q_sales_yoy = q_np_yoy = q_opm = np.nan
            q_per = ""
            if not q_pit.empty:
                pl_rows = q_pit.dropna(subset=["sales"]) if "sales" in q_pit else pd.DataFrame()
                if not pl_rows.empty:
                    ql = pl_rows.iloc[-1]
                    q_per = pd.Timestamp(pl_rows.index[-1]).strftime("%b %Y")
                    q_sales_yoy = round(float(ql["sales_yoy"]), 1) if pd.notna(ql.get("sales_yoy")) else np.nan
                    q_np_yoy = round(float(ql["np_yoy"]), 1) if pd.notna(ql.get("np_yoy")) else np.nan
                    q_opm = round(float(ql["opm"]), 1) if pd.notna(ql.get("opm")) else np.nan
            inst = im_pit.get(tk, {})
            val = _pit_valuation(tk, valid_dt, px, val_ctx, ann_raw)
            # Stage proxy
            m50, m200 = r_fx.get("ma50", np.nan), r_fx.get("ma200", np.nan)
            stage = 2 if (np.isfinite(m50) and np.isfinite(m200) and px > m50 > m200) else (
                4 if (np.isfinite(m50) and np.isfinite(m200) and px < m50 < m200) else (
                1 if (np.isfinite(m50) and px > m50) else 3))
            stocks_pkt[tk] = {
                "close": round(px, 2),
                "stage": stage,
                "dist_ma50": round(float(r_fx.get("dist_ma50", np.nan)), 1) if np.isfinite(r_fx.get("dist_ma50", np.nan)) else None,
                "dist_hi": round(float(r_fx.get("dist_hi", np.nan)), 1) if np.isfinite(r_fx.get("dist_hi", np.nan)) else None,
                "atr_pct": round(float(r_fx["atr"] / px * 100.0), 1) if np.isfinite(r_fx.get("atr", np.nan)) else None,
                "rs_1m": r1m if np.isfinite(r1m) else None,
                "rs_3m": r3m if np.isfinite(r3m) else None,
                "comp_rs": crs if np.isfinite(crs) else None,
                "gate1": bool(r_fx.get("gate1", False)),
                "fx_state": r_fx.get("state", "NO DATA"),
                "fx_action": r_fx.get("action", "WATCH"),
                "fx_legacy_action": r_old.get("action", "WATCH"),
                "fx_sig": r_fx.get("signature", ""),
                "fx_supply": r_fx.get("supply", ""),
                "fx_stop": round(float(r_fx["stop"]), 1) if np.isfinite(r_fx.get("stop", np.nan)) else None,
                "fx_stop_pct": round(float(r_fx["stop_pct"]), 1) if np.isfinite(r_fx.get("stop_pct", np.nan)) else None,
                "fx_ext_atr": round(float(r_fx["ext_atr"]), 2) if np.isfinite(r_fx.get("ext_atr", np.nan)) else None,
                "fx_guard": r_fx.get("entry_guard", ""),
                "fx_bbp": round(float(r_fx["bbp"]), 0) if np.isfinite(r_fx.get("bbp", np.nan)) else None,
                "fx_adx": round(float(r_fx["adx"]), 0) if np.isfinite(r_fx.get("adx", np.nan)) else None,
                "fx_rsi": round(float(r_fx["rsi"]), 0) if np.isfinite(r_fx.get("rsi", np.nan)) else None,
                "liq_cr": round(tmed_cr, 1),
                "deliv_pct": dp if np.isfinite(dp) else None,
                "deliv_ratio": round(dp / da, 2) if (np.isfinite(dp) and np.isfinite(da) and da > 0) else None,
                "abs_deliv_cr": abs_d if np.isfinite(abs_d) else None,
                "q_period": q_per,
                "q_sales_yoy": q_sales_yoy if np.isfinite(q_sales_yoy) else None,
                "q_np_yoy": q_np_yoy if np.isfinite(q_np_yoy) else None,
                "q_opm": q_opm if np.isfinite(q_opm) else None,
                "shp_period": inst.get("period", ""),
                "inst_label": inst.get("label", ""),
                "d_fii_q": round(float(inst["d_fii_q"]), 2) if pd.notna(inst.get("d_fii_q")) else None,
                "d_dii_q": round(float(inst["d_dii_q"]), 2) if pd.notna(inst.get("d_dii_q")) else None,
                "d_inst_q": round(float(inst["d_inst_q"]), 2) if pd.notna(inst.get("d_inst_q")) else None,
                "val_rating": val["rating"],
                "val_raw": val["verdict_raw"],
                "val_pct": val["cmp_pct"] if np.isfinite(val["cmp_pct"]) else None,
                "bear_blend": val["bear_blend"] if np.isfinite(val["bear_blend"]) else None,
                "p50": val["p50"] if np.isfinite(val["p50"]) else None,
                "cmp_lt_bear": val.get("cmp_lt_bear", False),
                "roce": val["roce"] if np.isfinite(val["roce"]) else None,
                "cfo_pat": val["cfo_pat"] if np.isfinite(val["cfo_pat"]) else None,
                "fcf_pat": val["fcf_pat"] if np.isfinite(val["fcf_pat"]) else None,
            }
        packets[dt_str] = {
            "as_of": valid_dt,
            "market": daily_regimes[valid_dt],
            "stocks": stocks_pkt,
        }

    sealed = {
        "targets": TARGETS,
        "dates": dates,
        "daily_traces": daily_traces,
        "daily_regimes": daily_regimes,
        "upgrade_stats": upgrade_stats,
        "monthly_gates": MONTHLY_GATES,
        "causal_events": CAUSAL_EVENTS,
        "packets": packets,
    }
    with open(os.path.join(OUT_DIR, "sealed_multibaggers.pkl"), "wb") as f:
        pickle.dump(sealed, f)

    # Save individual JSON packets (strictly point-in-time <= t)
    for dt_str, pkt in packets.items():
        with open(os.path.join(OUT_DIR, f"pkt_{dt_str}.json"), "w", encoding="utf-8") as f:
            json.dump(pkt, f, indent=2)

    print("\n=== UPGRADE ATTRIBUTION: OLD vs NEW FORENSICS ACROSS 11 MULTIBAGGERS (181 SESSIONS) ===")
    for tk in TARGETS:
        s = upgrade_stats[tk]
        ret = (daily_traces[tk][-1]["close"] / daily_traces[tk][0]["close"] - 1) * 100
        print(f"  {tk:15s} ret={ret:+6.1f}% | GO days: old={s['old_go']:3d} -> new={s['new_go']:3d} | "
              f"BUY_NOW: old={s['old_buy_now']:3d} -> new={s['new_buy_now']:3d} | guarded_WAIT={s['new_blocked_ext']:3d} | "
              f"1st GO: old={s['old_first_go']} new={s['new_first_go']}")

    rec_days = [d for d, r in daily_regimes.items() if r["engine_mode"] == "RECOVERY"]
    base_counts = pd.Series([r["base_mode"] for r in daily_regimes.values()]).value_counts().to_dict()
    new_counts = pd.Series([r["engine_mode"] for r in daily_regimes.values()]).value_counts().to_dict()
    print(f"\n=== REGIME COUNTS (181 sessions) ===")
    print(f"  Pre-fix base_mode : {base_counts}")
    print(f"  New engine_mode   : {new_counts}")
    print(f"  RECOVERY dates ({len(rec_days)}): {rec_days[:8]} ... {rec_days[-4:] if len(rec_days) > 8 else ''}")


if __name__ == "__main__":
    main()
