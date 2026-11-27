"""Build strictly point-in-time (PIT) event packets for the 9-event 3-Agent Blind Time-March (Feb 27 -> Nov 25, 2026)."""
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
OUT_DIR = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\time_march_3agents"
PKT_DIR = os.path.join(OUT_DIR, "packets")
SEALED_SRC = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\time_march\sealed_daily.pkl"

if REPO not in sys.path:
    sys.path.insert(0, REPO)
if r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\time_march" not in sys.path:
    sys.path.insert(0, r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\time_march")

import build_pit_dossiers as base  # noqa: E402
from utils import (  # noqa: E402
    forensics,
    fund_trends as ft,
    funnel,
    funnel_ui,
    industry_board as ib,
    nse_prices as npx,
    regime_badge as rb,
    regime_manager as rm,
    simba,
    valuation as val_mod,
)
from utils.nifty1000_list import SUB_INDUSTRY_MAP  # noqa: E402

TARGETS = ["ULTRACEMCO.NS", "KALYANKJIL.NS"]
EVENT_SCHEDULE = [
    ("2026-02-27", "E1: Launch (Feb 27) — Market CAUTION, UltraTech Gate 1 / Stage 3, Kalyan Stage 4"),
    ("2026-03-30", "E2: Bear Washout (Mar 30) — Nifty BEAR, Breadth 12.5%, UltraTech Valuation ACCUMULATE"),
    ("2026-04-22", "E3: Breadth Thrust & UltraTech 200-DMA / Gate 1 ON (Apr 22)"),
    ("2026-05-18", "E4: Post-Q4 FY26 Results Live (May 18) — UltraTech OPM 22%, Kalyan NP +118%"),
    ("2026-07-09", "E5: Kalyan 200-DMA High-Volume Breakout (Jul 9, Rs 443)"),
    ("2026-07-20", "E6: Dual Event (Jul 20) — UltraTech 200-DMA/Gate 1 ON + Kalyan First BUY NOW"),
    ("2026-09-15", "E7: Market BEAR + Kalyan Tight Squeeze BUY NOW + UltraTech ACCUMULATE P28 (Sep 15)"),
    ("2026-11-06", "E8: November Bull Breakout (Nov 6) — Both Stocks Gate 1 ON + Sep-2026 SHP Live"),
    ("2026-11-25", "E9: Final Checkpoint — Today (Nov 25, BULL Mode)"),
]


def _f(v: Any, ndig: int = 2) -> Optional[float]:
    return base._f(v, ndig)


def _pit_quarterly_split_lag(tk: str, as_of: pd.Timestamp) -> dict:
    """Strictly PIT quarterly P&L (45-day Reg 33 lag) & Shareholding (21-day Reg 31 lag)."""
    q = ft.quarterly(tk)
    if q is None or q.empty:
        return dict(quarter="", sales_yoy=None, opm=None, opm_chg=None, np_yoy=None,
                    shp_quarter="", fii=None, dii=None, promoters=None,
                    fii_q=None, dii_q=None, inst_q=None, inst_y=None, inst_label="NEUTRAL")

    # P&L: 45-day filing lag
    q_pl = q[q.index <= (as_of - pd.Timedelta(days=45))].sort_index()
    q_lbl, sales_yoy, opm, opm_chg, np_yoy = "", None, None, None, None
    if len(q_pl):
        latest_dt = q_pl.index[-1]
        q_lbl = latest_dt.strftime("%b %Y")
        prev_y_dt = latest_dt - pd.DateOffset(years=1)
        ym = q_pl.index[abs(q_pl.index - prev_y_dt) <= pd.Timedelta(days=15)]
        y_row = q_pl.loc[ym[-1]] if len(ym) else None
        cur = q_pl.iloc[-1]

        def _yoy(col: str) -> Optional[float]:
            if y_row is None or col not in q_pl.columns:
                return None
            c_v, y_v = _f(cur.get(col), 4), _f(y_row.get(col), 4)
            if c_v is None or y_v is None or y_v <= 1e-6:
                return None
            return round((c_v / y_v - 1.0) * 100.0, 1)

        sales_yoy = _yoy("sales")
        np_yoy = _yoy("np")
        opm = _f(cur.get("opm"), 1) if "opm" in q_pl.columns else None
        opm_prev = _f(y_row.get("opm"), 1) if (y_row is not None and "opm" in q_pl.columns) else None
        opm_chg = round(opm - opm_prev, 1) if (opm is not None and opm_prev is not None) else None

    # Shareholding: 21-day SEBI LODR Reg 31 lag
    q_sh = q[q.index <= (as_of - pd.Timedelta(days=21))].sort_index()
    shp_lbl, fii, dii, prom, fii_q, dii_q, inst_q, inst_y, inst_label = "", None, None, None, None, None, None, None, "NEUTRAL"
    if len(q_sh):
        cur_s = q_sh.iloc[-1]
        s_dt = q_sh.index[-1]
        shp_lbl = s_dt.strftime("%b %Y")
        fii = _f(cur_s.get("fii"), 2) if "fii" in q_sh.columns else None
        dii = _f(cur_s.get("dii"), 2) if "dii" in q_sh.columns else None
        prom = _f(cur_s.get("promoters"), 2) if "promoters" in q_sh.columns else None
        inst_now = (fii + dii) if (fii is not None and dii is not None) else None
        if len(q_sh) >= 2:
            pf = _f(q_sh.iloc[-2].get("fii"), 2) if "fii" in q_sh.columns else None
            pd_ = _f(q_sh.iloc[-2].get("dii"), 2) if "dii" in q_sh.columns else None
            if fii is not None and pf is not None:
                fii_q = round(fii - pf, 2)
            if dii is not None and pd_ is not None:
                dii_q = round(dii - pd_, 2)
            if inst_now is not None and pf is not None and pd_ is not None:
                inst_q = round(inst_now - (pf + pd_), 2)
        prev_sy = s_dt - pd.DateOffset(years=1)
        sym = q_sh.index[abs(q_sh.index - prev_sy) <= pd.Timedelta(days=15)]
        if inst_now is not None and len(sym):
            yr = q_sh.loc[sym[-1]]
            yf, yd = _f(yr.get("fii"), 2), _f(yr.get("dii"), 2)
            if yf is not None and yd is not None:
                inst_y = round(inst_now - (yf + yd), 2)
        if (inst_q is not None and inst_q >= 1.0 and (inst_y is None or inst_y > -2.0)) or (
            inst_y is not None and inst_y >= 2.0 and (inst_q is None or inst_q > -0.5)
        ):
            inst_label = "ACCUMULATING"
        elif (inst_q is not None and inst_q <= -1.0) or (inst_y is not None and inst_y <= -2.0):
            inst_label = "DISTRIBUTING"

    return dict(
        quarter=q_lbl, sales_yoy=sales_yoy, opm=opm, opm_chg=opm_chg, np_yoy=np_yoy,
        shp_quarter=shp_lbl, fii=fii, dii=dii, promoters=prom,
        fii_q=fii_q, dii_q=dii_q, inst_q=inst_q, inst_y=inst_y, inst_label=inst_label,
    )


def main() -> int:
    t0 = time.time()
    os.makedirs(PKT_DIR, exist_ok=True)
    os.chdir(REPO)

    with open(CACHE7Y, "rb") as f:
        pb = pickle.load(f)
    with open(SEALED_SRC, "rb") as f:
        sealed = pickle.load(f)

    st_extra, ix_extra = base._fetch_extra_nse_days(["2026-11-23", "2026-11-24", "2026-11-25"])
    H = npx.history(TARGETS + ["^NSEI"], start="2024-01-01")
    deliv_map = {tk: pb["deliv_pct"][tk].dropna().copy() for tk in TARGETS if tk in pb["deliv_pct"].columns}
    base._append_extra_bars(H, st_extra, ix_extra, deliv_map)

    n50_df = H["^NSEI"].sort_index()
    n50_s = n50_df["Close"]

    # Properly extract India VIX close series from pb["indices"]["vix"] DataFrame + Nov 23/25 extra
    vix_raw = pb["indices"]["vix"]
    vix_s = vix_raw["close"].dropna().sort_index().copy() if isinstance(vix_raw, pd.DataFrame) else vix_raw.dropna().sort_index().copy()
    if "^INDIAVIX" in H and len(H["^INDIAVIX"]):
        for dt_v, val_v in H["^INDIAVIX"]["Close"].items():
            vix_s.loc[dt_v] = float(val_v)
        vix_s = vix_s.sort_index()

    npe_s = pb["nifty_pe"].sort_index().copy()
    for ds_x, df_ix in sorted(ix_extra.items()):
        if "index" in df_ix.columns and "pe" in df_ix.columns:
            m_n50 = df_ix[df_ix["index"].astype(str).str.upper() == "NIFTY 50"]
            if len(m_n50) and pd.notna(m_n50.iloc[0]["pe"]):
                npe_s.loc[pd.Timestamp(ds_x)] = float(m_n50.iloc[0]["pe"])
    npe_s = npe_s.sort_index()
    fii_s = pb["fii_idx_long_pct"].sort_index()

    all_dates = H[TARGETS[0]].index.intersection(H[TARGETS[1]].index).intersection(n50_s.index).sort_values()
    val_ctx = val_mod.load_context()
    daily_traces = sealed["daily_traces"]

    manifest = []
    prev_dt: Optional[pd.Timestamp] = None
    for idx, (target_ds, label) in enumerate(EVENT_SCHEDULE, 1):
        avail = all_dates[all_dates <= pd.Timestamp(target_ds)]
        dt = avail[-1]
        ds = str(dt.date())

        snap_df, g1_vmed = base._universe_snapshot_at(pb, dt, st_extra, n50_s)
        board_df = ib.board(snap_df)
        n50_upto = n50_df.loc[:dt]
        reg_info = rb.compute(n50_upto, snap_df)

        vix_upto = vix_s.loc[:dt].dropna()
        vix_val = _f(vix_upto.iloc[-1], 2) if len(vix_upto) else None
        vix_5d_chg = _f(float(vix_upto.iloc[-1] - vix_upto.iloc[-6]), 2) if len(vix_upto) >= 6 else None
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
            india_vix_5d_chg=vix_5d_chg,
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

            # Microstructure + ATR14
            tr_h = sub["High"].tail(15).values
            tr_l = sub["Low"].tail(15).values
            tr_c = sub["Close"].tail(15).values
            atr14 = float(np.mean(np.maximum(tr_h[1:] - tr_l[1:], np.maximum(np.abs(tr_h[1:] - tr_c[:-1]), np.abs(tr_l[1:] - tr_c[:-1])))))
            atr14_pct = round(atr14 / px * 100.0, 2)

            d_s = deliv_map.get(tk, pd.Series(dtype=float)).loc[:dt].dropna()
            dp_now = _f(d_s.iloc[-1], 1) if len(d_s) else None
            dp_avg20 = _f(d_s.tail(20).mean(), 1) if len(d_s) >= 5 else None
            dp_ratio = _f(dp_now / dp_avg20, 2) if (dp_now and dp_avg20 and dp_avg20 > 0) else None
            lr = np.log(sub["Close"] / sub["Close"].shift(1)).dropna()
            vol20 = float(lr.tail(20).std() * math.sqrt(252.0)) if len(lr) >= 20 else np.nan
            w_mult = float(np.clip(g1_vmed / vol20, 0.50, 1.50)) if (np.isfinite(vol20) and vol20 > 1e-6) else 1.0
            liq_cr = float((sub["Close"] * sub["Volume"]).tail(20).median() / 1e7)
            abs_deliv_cr = round(liq_cr * (dp_now / 100.0), 1) if dp_now is not None else None

            # Business PIT + split-lag quarterly
            ann_asof = dt - pd.Timedelta(days=60)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fr = ft.read(tk, as_of=ann_asof)
            qp = _pit_quarterly_split_lag(tk, dt)
            biz_ok = bool(fr.get("score") is not None and float(fr["score"]) >= 45.0 and not fr.get("hard_flags"))
            biz_flags = "; ".join(fr.get("hard_flags") or [])

            # Valuation v4.1 PIT (including already-computed blends & CFO/FCF)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vr = val_mod.value_stock(tk, val_ctx, price=px, n=3000, hi52=hi52, lo52=lo52, as_of=dt)
            v_ok = vr.get("status") == "OK"
            v_dist = vr.get("dist") or {}
            v_snap = vr.get("snapshot") or {}
            v_bl = vr.get("blends") or {}
            v_lad = vr.get("ladder") or {}
            v_p50 = _f(v_dist.get("P50"), 1) if v_ok else None
            v_pe = _f(v_snap.get("spot_pe"), 1)
            val_info = dict(
                rating=vr.get("rating", ""),
                verdict_raw=vr.get("verdict_raw", ""),
                cmp_pct=_f(v_dist.get("cmp_pct"), 1) if v_ok else None,
                p50=v_p50,
                to_p50_pct=_f((v_p50 / px - 1.0) * 100.0, 1) if (v_p50 and px > 0) else None,
                bear_blend=_f(v_bl.get("bear"), 1) if v_bl else None,
                base_blend=_f(v_bl.get("base"), 1) if v_bl else None,
                pe=v_pe,
                roce_pct=_f((v_snap.get("roce") or 0) * 100.0, 1) if v_snap.get("roce") is not None else None,
                de=_f(v_snap.get("de"), 2),
                cfo_pat_3y=_f(v_snap.get("cfo_pat_3y"), 2),
                fcf_pat_3y=_f(v_snap.get("fcf_pat_3y"), 2),
                ladder=dict(accumulate=_f(v_lad.get("accumulate"), 1), fair_hi=_f(v_lad.get("fair_hi"), 1), trim=_f(v_lad.get("trim"), 1)) if v_lad else {},
            )

            # Funnel v3
            sb_stage, sb_mtf = sb.get("simba_stage"), sb.get("simba_mtf")
            sb_m_since, sb_si_since = sb.get("simba_m_since"), sb.get("simba_si_since")
            fx_stop = _f(fx.get("stop"), 2)
            fx_ent = _f(fx.get("entry"), 2) or px
            rk_pct = _f((fx_ent - fx_stop) / fx_ent * 100.0, 2) if (fx_stop and fx_ent and 0 < fx_stop < min(fx_ent, px)) else None
            simba_days = min([d for d in (sb_m_since, sb_si_since) if d is not None and np.isfinite(d)], default=np.nan)
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

            events = []
            if prev_dt is not None:
                prev_ds = str(prev_dt.date())
                win = [x for x in daily_traces[tk] if prev_ds < x["date"] <= ds]
                prev_act, prev_ab50, prev_ab200 = None, None, None
                for x in win:
                    if x["fx_action"] in ("BUY NOW", "BUY ON TRIGGER") and x["fx_action"] != prev_act:
                        events.append(f"{x['date']}: Technicals -> {x['fx_action']} (close {x['close']}, stop {x['fx_stop']})")
                    if x["ma50"] is not None:
                        ab50 = x["close"] > x["ma50"]
                        if prev_ab50 is not None and ab50 != prev_ab50:
                            events.append(f"{x['date']}: {'Above' if ab50 else 'Below'} 50-DMA ({x['close']} vs {x['ma50']})")
                        prev_ab50 = ab50
                    if x["ma200"] is not None:
                        ab200 = x["close"] > x["ma200"]
                        if prev_ab200 is not None and ab200 != prev_ab200:
                            events.append(f"{x['date']}: {'Above' if ab200 else 'Below'} 200-DMA ({x['close']} vs {x['ma200']})")
                        prev_ab200 = ab200
                    prev_act = x["fx_action"]
                events = events[-8:]

            stocks_pkt[tk] = dict(
                price=round(px, 2), ma50=round(ma50, 2), ma200=round(ma200, 2),
                vs_50dma_pct=round(d_ma50, 2), vs_200dma_pct=round(d_ma200, 2), vs_52w_high_pct=round(d_52w, 2),
                rs=dict(rs1w=rs1w, rs1m=rs1m, rs3m=rs3m, rs6m=rs6m, comp_rs=comp_rs),
                dna_signal=dna_sig, gate1=gate1, industry=ind_info,
                forensics=dict(
                    action=str(fx.get("action") or ""), state=str(fx.get("state") or ""),
                    supply=str(fx.get("supply") or ""), signature=str(fx.get("signature") or ""),
                    adx=_f(fx.get("adx"), 1), pdi=_f(fx.get("pdi"), 1), ndi=_f(fx.get("ndi"), 1),
                    rsi=_f(fx.get("rsi"), 1), bbp=_f(fx.get("bbp"), 1), gap=bool(fx.get("gap", False)),
                    entry=fx_ent, stop=fx_stop, stop_pct=rk_pct, half_size=bool(rk_pct is not None and rk_pct > 10.0),
                ),
                simba=dict(stage=sb_stage, mtf=sb_mtf, mrs=_f(sb.get("simba_mrs"), 2), caution=sb.get("simba_caution"), dist_days=sb.get("simba_dist_days")),
                microstructure=dict(
                    deliv_pct=dp_now, deliv_avg20=dp_avg20, deliv_ratio=dp_ratio, abs_deliv_cr=abs_deliv_cr,
                    vol20_ann_pct=_f(vol20 * 100.0, 1) if np.isfinite(vol20) else None,
                    atr14_pct=atr14_pct, invvol20_w_mult=round(w_mult, 3), liq_cr_day=round(liq_cr, 1),
                ),
                business_pit=dict(verdict=fr.get("verdict"), score=_f(fr.get("score"), 1), quarterly=qp),
                valuation_pit=val_info,
                funnel_v3=dict(score=_f(f_scored["funnel_score"], 1), grade=str(f_scored["funnel_grade"]),
                               pros=[c["text"] for c in chips["pros"]], cons=[c["text"] for c in chips["cons"]]),
                inter_event_log=events,
            )

        pkt = dict(event_num=idx, event_date=ds, event_label=label, market=market_ctx, stocks=stocks_pkt)
        pkt_path = os.path.join(PKT_DIR, f"event_{idx:02d}_{ds}.json")
        with open(pkt_path, "w", encoding="utf-8") as f:
            json.dump(pkt, f, indent=2)
        manifest.append(dict(event_num=idx, date=ds, label=label, path=pkt_path))
        prev_dt = dt

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(dict(events=manifest, elapsed_sec=round(time.time() - t0, 1)), f, indent=2)
    print(f"Built {len(manifest)} event packets ({manifest[0]['date']} .. {manifest[-1]['date']}) in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
