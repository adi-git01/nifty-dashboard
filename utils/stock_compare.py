"""Unified 2-4 stock head-to-head Decision Comparison engine across all 5 dashboard pillars."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from utils import funnel, industry_board

STAGE_META = (
    ("s_industry", "🏭 Industry"),
    ("s_trend", "🌊 Trend"),
    ("s_business", "🏢 Business"),
    ("s_setup", "🎯 Setup"),
    ("s_value", "💰 Value"),
)


def _bare(ticker: str) -> str:
    return str(ticker or "").strip().upper().replace(".NS", "")


def normalize_tickers(
    tickers: Iterable[str],
    available: Optional[Iterable[str]] = None,
    max_items: int = 4,
) -> List[str]:
    """Normalize bare or .NS tickers against available universe while preserving order."""
    avail_list = [str(t) for t in available] if available is not None else []
    avail_set = set(avail_list)
    bare_map = {_bare(t): t for t in avail_list}
    out: List[str] = []
    for raw in tickers or ():
        s = str(raw or "").strip().upper()
        if not s:
            continue
        b = _bare(s)
        cand = bare_map.get(b) if avail_set else (s if s.endswith(".NS") else f"{b}.NS")
        if cand is None and not avail_set:
            cand = s if s.endswith(".NS") else f"{b}.NS"
        if cand and (not avail_set or cand in avail_set) and cand not in out:
            out.append(cand)
        if len(out) >= max_items:
            break
    return out


def _txt(v: Any, default: str = "") -> str:
    if v is None or pd.isna(v):
        return default
    s = str(v).strip()
    return default if s in ("", "nan", "None") else s


def _num(v: Any) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except Exception:
        return np.nan


def _pick_advantage(bares: List[str], scores: List[float], min_margin: float = 1e-6) -> str:
    """Return the bare ticker with the highest finite score, or 'Tie' / '—'."""
    valid = [(b, s) for b, s in zip(bares, scores) if np.isfinite(s)]
    if not valid:
        return "—"
    valid.sort(key=lambda pair: pair[1], reverse=True)
    best_b, best_s = valid[0]
    ties = [b for b, s in valid if abs(best_s - s) <= min_margin]
    if len(ties) == len(bares) and len(bares) > 1:
        return "Tie"
    if len(ties) > 1:
        return " / ".join(ties)
    return best_b


def _fmt_funnel_stages(r: pd.Series) -> tuple[str, float]:
    passed = 0
    failed_labels: List[str] = []
    for col, label in STAGE_META:
        ok = r.get(col)
        if ok is True or (pd.notna(ok) and bool(ok)):
            passed += 1
        else:
            short = label.split(" ", 1)[-1]
            if col == "s_trend":
                sc = _txt(r.get("scanner"), "—")
                failed_labels.append(f"{short}: {sc}")
            elif col == "s_setup":
                tc = _txt(r.get("technicals"), "—")
                failed_labels.append(f"{short}: {tc}")
            else:
                failed_labels.append(short)
    core4 = sum(1 for col, _ in STAGE_META[:4] if r.get(col) is True or (pd.notna(r.get(col)) and bool(r.get(col))))
    if passed == 5:
        cell = "5 / 5 PASS ✅"
    elif core4 == 4:
        cell = f"4 / 4 Default PASS (Fails {', '.join(failed_labels)})"
    else:
        cell = f"{passed} / 5 (Fails {', '.join(failed_labels)})"
    comp_rs = _num(r.get("comp_rs"))
    tie = (comp_rs if np.isfinite(comp_rs) else -99.0) * 1e-3
    score = passed * 10.0 + core4 * 5.0 + (3.0 if bool(r.get("s_trend")) and bool(r.get("s_setup")) else 0.0) + tie
    return cell, score


def _fmt_industry(r: pd.Series) -> tuple[str, float]:
    ind = _txt(r.get("industry"), "Unclassified")
    grp = _txt(r.get("ind_group"), "—").title()
    st = _txt(r.get("ind_status"), "").title()
    sc = _num(r.get("ind_score"))
    sc_txt = f" {sc:.0f}" if np.isfinite(sc) else ""
    st_txt = f" ({st})" if st else ""
    cell = f"{ind} · {grp}{sc_txt}{st_txt}"
    score = (sc if np.isfinite(sc) else 0.0) + (5.0 if bool(r.get("s_industry")) else -10.0)
    return cell, score


def _fmt_scanner(r: pd.Series) -> tuple[str, float]:
    sig = _txt(r.get("scanner"), "—")
    crs = _num(r.get("comp_rs"))
    rs1 = _num(r.get("rs_1m"))
    rs3 = _num(r.get("rs_3m"))
    parts: List[str] = []
    if np.isfinite(crs):
        parts.append(f"CompRS {crs:+.1f}")
    if np.isfinite(rs1):
        parts.append(f"RS1m {rs1:+.1f}")
    if np.isfinite(rs3):
        parts.append(f"RS3m {rs3:+.1f}")
    cell = f"{sig} ({', '.join(parts)})" if parts else sig
    sig_pts = {"BUY": 100.0, "HOLD": 40.0, "AVOID": 0.0, "SELL": 0.0}.get(sig.upper(), 20.0)
    score = sig_pts + (crs if np.isfinite(crs) else -50.0) + 0.25 * (rs1 if np.isfinite(rs1) else 0.0)
    return cell, score


def _fmt_technicals(r: pd.Series) -> tuple[str, float]:
    act = _txt(r.get("technicals"), "—")
    st = _txt(r.get("fx_state"))
    sup = _txt(r.get("fx_supply"))
    sig = _txt(r.get("fx_signature"))
    shk = bool(r.get("fx_shakeout")) if pd.notna(r.get("fx_shakeout")) else False
    sd = _num(r.get("simba_days"))
    bits = [act]
    if st and st != act:
        bits.append(st)
    if sup:
        bits.append(sup)
    if sig == "ABSORPTION":
        bits.append("⭐ ABSORPTION")
    if shk:
        bits.append("🪤 SHAKEOUT")
    if np.isfinite(sd) and 0 <= sd <= funnel.SETUP_DAYS:
        bits.append(f"SIMBA {int(sd)}d")
    cell = " · ".join(bits)
    act_pts = {"BUY NOW": 100.0, "BUY ON TRIGGER": 70.0, "HOLD": 35.0, "WAIT": 20.0, "AVOID": 0.0}.get(act.upper(), 15.0)
    sup_pts = {"CLEAN AIR": 20.0, "NEAR OVERHANG": 5.0, "AT CEILING": -10.0, "OVERHEAD SUPPLY": -15.0}.get(sup.upper(), 0.0)
    sig_pts = 12.0 if sig == "ABSORPTION" else (-10.0 if sig == "DISTRIBUTION" else 0.0)
    shk_pts = 6.0 if shk else 0.0
    score = act_pts + sup_pts + sig_pts + shk_pts
    return cell, score


def _fmt_entry(r: pd.Series) -> tuple[str, float]:
    px = _num(r.get("price"))
    ent = _num(r.get("entry"))
    gap = _num(r.get("trigger_gap_pct"))
    act = _txt(r.get("technicals"), "—").upper()
    if not np.isfinite(ent) or ent <= 0:
        return "—", -999.0
    if not np.isfinite(gap) and np.isfinite(px) and px > 0:
        gap = (ent / px - 1.0) * 100.0
    if act == "BUY NOW" and (not np.isfinite(gap) or abs(gap) < 0.25):
        cell = f"₹{ent:,.1f} (0.0% gap, actionable now)"
        score = 100.0
    elif np.isfinite(gap) and gap > 0.05 and np.isfinite(px):
        cell = f"₹{ent:,.1f} ({gap:+.1f}% above current ₹{px:,.1f})"
        base = 75.0 if act in ("BUY NOW", "BUY ON TRIGGER") else 30.0
        score = base - min(abs(gap), 30.0)
    else:
        gap_txt = f"{gap:+.1f}% gap" if np.isfinite(gap) else "at market"
        cell = f"₹{ent:,.1f} ({gap_txt})"
        base = 85.0 if act == "BUY NOW" else (65.0 if act == "BUY ON TRIGGER" else 25.0)
        score = base - (abs(gap) if np.isfinite(gap) else 0.0)
    return cell, score


def _fmt_stop_risk(r: pd.Series) -> tuple[str, float]:
    stop = _num(r.get("stop"))
    vs_stop = _num(r.get("dist_stop_pct"))
    risk = _num(r.get("risk_pct"))
    half = bool(r.get("fx_half_size")) if pd.notna(r.get("fx_half_size")) else (np.isfinite(risk) and risk > 10.0)
    if not np.isfinite(stop) or stop <= 0:
        return "No valid stop", -999.0
    parts = [f"₹{stop:,.1f}"]
    sub: List[str] = []
    if np.isfinite(vs_stop):
        sub.append(f"vs Stop {vs_stop:+.1f}%")
    if np.isfinite(risk):
        sz = "HALF" if half else "FULL"
        sub.append(f"Risk {risk:.1f}% {sz}")
    else:
        sub.append("inverted/invalid risk")
    cell = f"{parts[0]} ({' · '.join(sub)})"
    if not np.isfinite(risk) or risk <= 0:
        score = -100.0
    elif risk <= 10.0:
        score = 100.0 - risk
    elif risk <= funnel.MAX_RISK_PCT:
        score = 70.0 - risk
    else:
        score = 20.0 - risk
    return cell, score


def _fmt_trend_strength(r: pd.Series) -> tuple[str, float]:
    adx = _num(r.get("fx_adx"))
    stg = _num(r.get("stage"))
    mtf = _num(r.get("simba_mtf"))
    d50 = _num(r.get("dist_ma50"))
    d52 = _num(r.get("dist_52w"))
    bits: List[str] = []
    if np.isfinite(adx):
        lbl = " (choppy)" if adx < 20 else (" (strong)" if adx >= 25 else "")
        bits.append(f"ADX {adx:.1f}{lbl}")
    if np.isfinite(stg) or np.isfinite(mtf):
        s_txt = f"S{int(stg)}" if np.isfinite(stg) else "S?"
        m_txt = f"MTF {int(mtf)}/3" if np.isfinite(mtf) else "MTF ?"
        bits.append(f"SIMBA {s_txt} · {m_txt}")
    if np.isfinite(d50):
        bits.append(f"vs MA50 {d50:+.1f}%")
    if np.isfinite(d52):
        bits.append(f"{d52:+.1f}% off 52wH")
    cell = " · ".join(bits) if bits else "—"
    stg_pts = 30.0 if (np.isfinite(stg) and int(stg) == 2) else (-25.0 if (np.isfinite(stg) and int(stg) in (3, 4)) else 0.0)
    mtf_pts = (mtf if np.isfinite(mtf) else 0.0) * 12.0
    adx_pts = min(adx, 50.0) * 0.8 if np.isfinite(adx) else 0.0
    d52_pts = max(d52, -40.0) * 0.5 if np.isfinite(d52) else -20.0
    score = stg_pts + mtf_pts + adx_pts + d52_pts
    return cell, score


def _fmt_quarter(r: pd.Series) -> tuple[str, float]:
    qtr = _txt(r.get("quarter"))
    sy = _num(r.get("sales_yoy"))
    ny = _num(r.get("np_yoy"))
    opm = _num(r.get("opm"))
    ochg = _num(r.get("opm_chg"))
    bits: List[str] = []
    if np.isfinite(sy):
        bits.append(f"Sales {sy:+.1f}%")
    if np.isfinite(ny):
        bits.append(f"NP {ny:+.1f}%")
    if np.isfinite(opm):
        d_txt = f" ({ochg:+.1f} pp)" if np.isfinite(ochg) else ""
        bits.append(f"OPM {opm:.1f}%{d_txt}")
    if not bits:
        return "—", -999.0
    prefix = f"{qtr}: " if qtr else ""
    cell = prefix + " · ".join(bits)
    ny_pts = np.clip(ny, -100.0, 200.0) * 0.5 if np.isfinite(ny) else 0.0
    sy_pts = np.clip(sy, -100.0, 150.0) * 0.4 if np.isfinite(sy) else 0.0
    opm_pts = np.clip(ochg, -15.0, 15.0) * 3.0 if np.isfinite(ochg) else 0.0
    score = ny_pts + sy_pts + opm_pts
    return cell, score


def _fmt_pe(r: pd.Series) -> tuple[str, float]:
    pe = _num(r.get("pe"))
    if not np.isfinite(pe) or pe <= 0:
        return "—", -999.0
    return f"{pe:.1f}×", -pe


def _fmt_biz_val(r: pd.Series) -> tuple[str, float]:
    bsc = _num(r.get("biz_score"))
    bverd = _txt(r.get("biz_verdict"), "—")
    vrat = _txt(r.get("val_rating"), "UNRATED")
    vpct = _num(r.get("val_pct"))
    vup = _num(r.get("val_upside"))
    b_part = f"Biz {bsc:.0f} {bverd}" if np.isfinite(bsc) else f"Biz {bverd}"
    p_bits: List[str] = []
    if np.isfinite(vpct):
        p_bits.append(f"P{vpct:.0f}")
    if np.isfinite(vup):
        p_bits.append(f"{vup:+.0f}% to mid")
    v_part = f"{vrat} ({', '.join(p_bits)})" if p_bits else vrat
    cell = f"{b_part} · {v_part}"
    b_pts = bsc if np.isfinite(bsc) else (65.0 if bverd == "STRONG" else (50.0 if bverd == "MIXED" else 25.0))
    rat_upper = vrat.upper()
    if "BUY" in rat_upper or "ACCUMULATE" in rat_upper:
        v_pts = 35.0
    elif "HOLD" in rat_upper:
        v_pts = 15.0
    elif "UNRATED" in rat_upper:
        v_pts = 5.0
    else:
        v_pts = -15.0
    pct_pts = (50.0 - vpct) * 0.4 if np.isfinite(vpct) else 0.0
    score = b_pts + v_pts + pct_pts
    return cell, score


ROW_BUILDERS = (
    ("Funnel Stages", _fmt_funnel_stages),
    ("Industry Leadership", _fmt_industry),
    ("Scanner Momentum", _fmt_scanner),
    ("Technicals v6 Action", _fmt_technicals),
    ("Entry & Trigger Gap", _fmt_entry),
    ("Stop & Risk", _fmt_stop_risk),
    ("Trend Strength", _fmt_trend_strength),
    ("Latest Quarter", _fmt_quarter),
    ("Trailing P/E", _fmt_pe),
    ("Business & Valuation v4.1", _fmt_biz_val),
)


def _compute_winners(T: pd.DataFrame, bares: List[str]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    swing_scores: List[tuple[str, str, float, str]] = []
    garp_scores: List[tuple[str, str, float, str]] = []
    for t, b in zip(T.index, bares):
        r = T.loc[t]
        _, s_fun = _fmt_funnel_stages(r)
        _, s_ind = _fmt_industry(r)
        _, s_scn = _fmt_scanner(r)
        _, s_tec = _fmt_technicals(r)
        _, s_ent = _fmt_entry(r)
        _, s_stp = _fmt_stop_risk(r)
        _, s_trd = _fmt_trend_strength(r)
        _, s_qtr = _fmt_quarter(r)
        _, s_pe = _fmt_pe(r)
        _, s_bvl = _fmt_biz_val(r)

        swing_total = (
            s_fun * 1.2
            + s_ind * 0.4
            + s_scn * 0.9
            + s_tec * 1.1
            + max(s_ent, -50.0) * 0.6
            + max(s_stp, -50.0) * 0.4
            + s_trd * 0.8
            + max(s_qtr, -50.0) * 0.35
        )
        act = _txt(r.get("technicals"), "—")
        sup = _txt(r.get("fx_supply"))
        crs = _num(r.get("comp_rs"))
        ny = _num(r.get("np_yoy"))
        s_reasons = [
            f"Scanner {_txt(r.get('scanner'), '—')}" + (f" (CompRS {crs:+.1f})" if np.isfinite(crs) else ""),
            f"{act}" + (f" · {sup}" if sup else ""),
        ]
        if np.isfinite(ny):
            s_reasons.append(f"Qtr NP {ny:+.0f}% YoY")
        swing_scores.append((t, b, float(swing_total), " · ".join(s_reasons)))

        pe = _num(r.get("pe"))
        pe_pts = (45.0 - min(pe, 80.0)) * 0.6 if (np.isfinite(pe) and pe > 0) else 0.0
        garp_total = (
            s_bvl * 1.6
            + max(s_qtr, -50.0) * 0.5
            + pe_pts
            + (15.0 if bool(r.get("s_business")) else -20.0)
            + (15.0 if bool(r.get("s_value")) else -10.0)
            + s_ind * 0.25
        )
        bsc = _num(r.get("biz_score"))
        vrat = _txt(r.get("val_rating"), "UNRATED")
        vpct = _num(r.get("val_pct"))
        g_reasons = [
            (f"Biz {bsc:.0f} {_txt(r.get('biz_verdict'))}" if np.isfinite(bsc) else f"Biz {_txt(r.get('biz_verdict'), '—')}"),
            f"{vrat}" + (f" (P{vpct:.0f})" if np.isfinite(vpct) else ""),
        ]
        if np.isfinite(pe) and pe > 0:
            g_reasons.append(f"P/E {pe:.1f}×")
        garp_scores.append((t, b, float(garp_total), " · ".join(g_reasons)))

    swing_scores.sort(key=lambda x: x[2], reverse=True)
    garp_scores.sort(key=lambda x: x[2], reverse=True)
    sw = swing_scores[0] if swing_scores else ("", "—", 0.0, "")
    gw = garp_scores[0] if garp_scores else ("", "—", 0.0, "")
    return (
        dict(ticker=sw[0], bare=sw[1], score=round(sw[2], 1), why=sw[3]),
        dict(ticker=gw[0], bare=gw[1], score=round(gw[2], 1), why=gw[3]),
    )


def build_comparison(
    tickers: Iterable[str],
    df: pd.DataFrame,
    board: Optional[pd.DataFrame] = None,
    val: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Build the 10-row side-by-head Decision Matrix across 2-4 stocks plus Swing & GARP winner cards.
    Uses the same underlying funnel.table() pipeline as Investment Funnel and Stock Report.
    """
    if df is None or df.empty or "ticker" not in df.columns:
        empty = pd.DataFrame(columns=["Metric", "Advantage"])
        return dict(tickers=[], bares=[], rows=pd.DataFrame(), matrix=empty, swing_winner={}, garp_winner={})

    valid = normalize_tickers(tickers, available=df["ticker"].tolist(), max_items=4)
    if not valid:
        empty = pd.DataFrame(columns=["Metric", "Advantage"])
        return dict(tickers=[], bares=[], rows=pd.DataFrame(), matrix=empty, swing_winner={}, garp_winner={})

    if board is None:
        board = industry_board.board(df, industry_board.month_ago_snapshot(current_df=df))
    if val is None:
        val = funnel.load_valuation()

    sub = df[df["ticker"].isin(valid)].copy()
    T = funnel.table(sub, board=board, val=val, business_for=valid)
    T = T.drop_duplicates(subset=["ticker"], keep="first").set_index("ticker").reindex(valid)
    bares = [_bare(t) for t in valid]

    matrix_rows: List[Dict[str, Any]] = []
    for metric_label, builder in ROW_BUILDERS:
        row_dict: Dict[str, Any] = {"Metric": metric_label}
        scores: List[float] = []
        for t, b in zip(valid, bares):
            cell, sc = builder(T.loc[t])
            row_dict[b] = cell
            scores.append(sc)
        row_dict["Advantage"] = _pick_advantage(bares, scores)
        matrix_rows.append(row_dict)

    matrix = pd.DataFrame(matrix_rows, columns=["Metric", *bares, "Advantage"])
    swing_winner, garp_winner = _compute_winners(T, bares)
    return dict(
        tickers=valid,
        bares=bares,
        rows=T.reset_index(),
        matrix=matrix,
        swing_winner=swing_winner,
        garp_winner=garp_winner,
    )
