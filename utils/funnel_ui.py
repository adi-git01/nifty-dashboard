"""Signal Strip & By Sector (Best-in-Theme) view models and HTML/CSS builders for Investment Funnel."""
from __future__ import annotations

import html
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

PILLAR_MAX = {"Mom": 28, "Set": 22, "Str": 12, "Qual": 18, "Cat": 22}
PILLAR_COLORS = {
    "Mom": "#635BFF",
    "Set": "#00A389",
    "Str": "#0054D2",
    "Qual": "#DC6803",
    "Cat": "#9333EA",
}
STRIP_FILTERS = [
    ("ALL", "All"),
    ("A", "🔥 Grade A"),
    ("CAT", "⭐ Catalysts"),
    ("CLEAN", "✅ Zero Cautions"),
    ("WARN", "⚠️ Has Cautions"),
]


def _get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    if isinstance(row, pd.Series):
        return row.get(key, default)
    return getattr(row, key, default)


def _num(row: Any, key: str) -> Optional[float]:
    v = _get(row, key, None)
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    return fv if math.isfinite(fv) else None


def _txt(row: Any, key: str) -> str:
    v = _get(row, key, "")
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none", "<na>") else s


def _flag(row: Any, key: str) -> bool:
    v = _get(row, key, False)
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in ("true", "1", "yes")


def parse_pillar_scores(score_note: Any) -> Dict[str, int]:
    """Parse 'Mom 17 · Set 18 · Str 10 · Qual -2 · Cat 8' into a 5-key dict."""
    out = {"Mom": 0, "Set": 0, "Str": 0, "Qual": 0, "Cat": 0}
    if score_note is None:
        return out
    if isinstance(score_note, float) and math.isnan(score_note):
        return out
    for k, val in re.findall(r"(Mom|Set|Str|Qual|Cat)\s+(-?\d+)", str(score_note)):
        try:
            out[k] = int(val)
        except ValueError:
            continue
    return out


def build_signal_chips(row: Any, pillars: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """
    Translate calibrated v3 scoring attributes into scannable pro/caution chips.
    Preserves v3 evidence discipline: DISTRIBUTION and DISTRIBUTING carry zero penalty.
    """
    pil = pillars if pillars is not None else parse_pillar_scores(_get(row, "score_note", ""))
    pros: List[Dict[str, str]] = []
    cons: List[Dict[str, str]] = []
    has_catalyst = False

    # 1. Near-miss indicator (if present on near-miss rows)
    missed = _txt(row, "Missed") or _txt(row, "missed")
    miss_gap = _txt(row, "Gap to pass") or _txt(row, "miss_gap")
    if missed:
        cons.append({
            "text": f"🎯 Missed {missed}: {miss_gap}" if miss_gap else f"🎯 Missed {missed}",
            "kind": "warn",
            "_lead": True,
        })

    # 2. Proven Alpha Catalysts (pros, catalysts listed first)
    sig = _txt(row, "fx_signature").upper()
    if sig == "ABSORPTION":
        pros.append({"text": "⭐ Absorption (+13)", "kind": "ok"})
        has_catalyst = True
    if _flag(row, "fx_shakeout"):
        pros.append({"text": "🪤 Shakeout (+9)", "kind": "ok"})
        has_catalyst = True

    inst_lab = _txt(row, "inst_label").upper()
    inst_q = _num(row, "inst_q")
    inst_y = _num(row, "inst_y")
    dii_q = _num(row, "dii_q")
    dii_y = _num(row, "dii_y")
    fii_q = _num(row, "fii_q")
    fii_y = _num(row, "fii_y")
    if inst_lab == "ACCUMULATING" and (inst_y is None or inst_y > -2.0):
        if inst_q is not None and inst_y is not None:
            txt = f"🟢 Inst {inst_q:+.1f}q/{inst_y:+.1f}y (+8)"
        elif inst_q is not None:
            txt = f"🟢 Inst {inst_q:+.1f}pp QoQ (+8)"
        else:
            txt = "🟢 Inst Accum (+8)"
        pros.append({"text": txt, "kind": "ok"})
        has_catalyst = True
    else:
        # Surface strong single-leg domestic/foreign accumulation even when netted FII+DII is NEUTRAL
        if (dii_q is not None and dii_q >= 0.75) or (dii_y is not None and dii_y >= 2.0 and (dii_q is None or dii_q > 0)):
            d_txt = f"🏦 DII {dii_q:+.1f}q" if dii_q is not None else f"🏦 DII {dii_y:+.1f}y"
            if dii_q is not None and dii_y is not None:
                d_txt += f"/{dii_y:+.1f}y"
            pros.append({"text": d_txt, "kind": "info"})
        elif (fii_q is not None and fii_q >= 0.75) or (fii_y is not None and fii_y >= 2.0 and (fii_q is None or fii_q > 0)):
            f_txt = f"🌍 FII {fii_q:+.1f}q" if fii_q is not None else f"🌍 FII {fii_y:+.1f}y"
            if fii_q is not None and fii_y is not None:
                f_txt += f"/{fii_y:+.1f}y"
            pros.append({"text": f_txt, "kind": "info"})

    deal_cr = _num(row, "deal_inst_cr")
    if deal_cr is not None and deal_cr > 0.0:
        pros.append({"text": f"🏦 Inst Deals +₹{deal_cr:.1f}cr", "kind": "ok"})
        has_catalyst = True

    dp = _num(row, "deliv_pct")
    da = _num(row, "deliv_avg20")
    abs_d = _num(row, "abs_deliv_cr")
    if abs_d is None:
        _liq = _num(row, "liq_cr")
        if _liq is not None and dp is not None and _liq > 0.0 and dp > 0.0:
            abs_d = _liq * dp / 100.0
    if dp is not None and da is not None and dp > 0.0 and da > 0.0 and (dp / da) >= 1.25:
        d_sfx = f" · ₹{abs_d:.0f}cr" if abs_d is not None and abs_d >= 25.0 else ""
        pros.append({"text": f"📦 Deliv {dp:.0f}% ({dp / da:.1f}×){d_sfx}", "kind": "ok"})
        has_catalyst = True
    elif abs_d is not None and abs_d >= 100.0 and dp is not None and dp > 0.0:
        pros.append({"text": f"📦 ₹{abs_d:.0f}cr/d Deliv ({dp:.0f}%)", "kind": "info"})

    if pil.get("Cat", 0) > 0:
        has_catalyst = True

    # 3. Other positive conviction chips
    crs = _num(row, "comp_rs")
    if crs is not None:
        if crs >= 8.0:
            pros.append({"text": f"🚀 RS {crs:+.1f} Leader", "kind": "ok"})
        elif crs >= 3.0:
            pros.append({"text": f"📈 RS {crs:+.1f}", "kind": "info"})

    hi52_bo = _num(row, "hi52_bo_days")
    ath_bo = _num(row, "ath_bo_days")
    if (hi52_bo is not None and 0.0 <= hi52_bo <= 10.0) or (ath_bo is not None and 0.0 <= ath_bo <= 10.0):
        pros.append({"text": "💥 Fresh Breakout", "kind": "ok"})

    bbp = _num(row, "fx_bbp")
    if bbp is not None and bbp < 20.0:
        pros.append({"text": "🌀 Tight Squeeze", "kind": "info"})

    if _flag(row, "fx_gap"):
        pros.append({"text": "⚡ Gap Up", "kind": "info"})

    tech = _txt(row, "technicals").upper()
    supply = _txt(row, "fx_supply").upper()
    if pil.get("Set", 0) >= 20 and tech == "BUY NOW" and supply == "CLEAN AIR":
        pros.append({"text": "🎯 Clean Setup", "kind": "ok"})

    if pil.get("Str", 0) >= 12:
        pros.append({"text": "📈 ADX+MTF 12/12", "kind": "info"})

    biz_sc = _num(row, "biz_score")
    if biz_sc is not None and biz_sc >= 70.0:
        pros.append({"text": f"🏢 Biz {biz_sc:.0f} Strong", "kind": "ok"})

    np_yoy = _num(row, "np_yoy")
    if np_yoy is not None and np_yoy >= 40.0:
        pros.append({"text": f"🔥 Qtr NP +{np_yoy:.0f}%", "kind": "ok"})

    rk = _num(row, "risk_pct")
    if rk is not None and 3.0 <= rk <= 10.0:
        pros.append({"text": f"📏 Tight {rk:.1f}% stop", "kind": "ok"})

    vrat = _txt(row, "val_rating").upper()
    vpct = _num(row, "val_pct")
    vp_sfx = f" (P{vpct:.0f})" if vpct is not None else ""
    if vrat in ("STRONG BUY", "BUY", "ACCUMULATE"):
        pros.append({"text": f"💰 Val {vrat}{vp_sfx}", "kind": "ok"})
    elif vrat.startswith("BUY*") or vrat.startswith("ACCUMULATE*"):
        base_r = vrat.split(" (", 1)[0]
        floor_tag = " <Bear" if "<BEAR" in vrat else ""
        pros.append({"text": f"💰 Val {base_r}{floor_tag}{vp_sfx}", "kind": "ok"})

    pe = _num(row, "pe")
    if pe is not None and 0.0 < pe <= 22.0:
        pros.append({"text": f"💵 P/E {pe:.1f}×", "kind": "info"})

    # 4. Hard penalty chips (kind="bad") — never penalize DISTRIBUTION or DISTRIBUTING
    nse_flags = _txt(row, "nse_flags")
    if "⚠️" in nse_flags:
        cons.append({"text": f"🔴 {nse_flags[:42]} (-14)", "kind": "bad"})

    is_loss = _flag(row, "val_loss") or (pe is not None and (pe <= 0.0 or pe >= 200.0))
    if is_loss:
        cons.append({"text": "🔴 Trailing Loss / Extreme P/E (-12)", "kind": "bad"})

    if rk is not None and 0.0 < rk < 2.0:
        cons.append({"text": f"🔴 {rk:.1f}% noise stop (-3)", "kind": "bad"})

    if supply == "NEAR OVERHANG":
        cons.append({"text": "🔴 Near Overhang (-6)", "kind": "bad"})

    if "STRONG SELL" in vrat:
        cons.append({"text": f"🔴 Val STRONG SELL{vp_sfx} (-4)", "kind": "bad"})

    biz_flags = _txt(row, "biz_flags")
    sales_yoy = _num(row, "sales_yoy")
    opm_chg = _num(row, "opm_chg")
    if biz_flags:
        cons.append({"text": f"🔴 {biz_flags[:48]}", "kind": "bad"})
    elif sales_yoy is not None and opm_chg is not None and sales_yoy < 0.0 and opm_chg <= -1.5:
        cons.append({"text": "🔴 Qtr Sales & Margin Drop (-4.5)", "kind": "bad"})

    if tech == "WATCH":
        cons.append({"text": "🔴 WATCH (No BUY setup)", "kind": "bad"})

    if rk is not None and rk > 15.0:
        cons.append({"text": f"🔴 Wide {rk:.1f}% stop (>15% cap)", "kind": "bad"})

    # 5. Soft caution chips (kind="warn") & informational trigger (kind="info")
    if supply == "INSIDE SHELF":
        cons.append({"text": "🟠 Inside Shelf (-2.5)", "kind": "warn"})

    dma50 = _num(row, "dist_ma50")
    if dma50 is not None:
        if dma50 > 20.0:
            cons.append({"text": f"⚠️ +{dma50:.1f}% vs 50DMA (-3)", "kind": "warn"})
        elif dma50 > 16.0:
            cons.append({"text": f"⚠️ +{dma50:.1f}% vs 50DMA (-1.5)", "kind": "warn"})

    dma200 = _num(row, "dist_200dma")
    if dma200 is not None:
        if dma200 > 60.0:
            cons.append({"text": f"⚠️ +{dma200:.0f}% vs 200DMA (-5)", "kind": "warn"})
        elif dma200 > 38.0:
            cons.append({"text": f"⚠️ +{dma200:.0f}% vs 200DMA (-3)", "kind": "warn"})

    if vrat == "REDUCE":
        cons.append({"text": "🟠 Val REDUCE (-2)", "kind": "warn"})

    if (not is_loss) and pe is not None and 100.0 <= pe < 200.0:
        cons.append({"text": f"🟠 P/E {pe:.0f}× (-4)", "kind": "warn"})

    if rk is not None and 10.0 < rk <= 15.0:
        cons.append({"text": f"⚖️ Half-size {rk:.1f}% stop", "kind": "warn"})

    if tech == "WAIT":
        cons.append({"text": "🟠 WAIT (Needs trigger)", "kind": "warn"})
    elif tech == "BUY ON TRIGGER":
        tgap = _num(row, "trigger_gap_pct")
        if tgap is not None and tgap > 0.0:
            cons.append({"text": f"⏳ Trigger +{tgap:.1f}%", "kind": "info"})

    liq = _num(row, "liq_cr")
    if liq is not None and 0.0 < liq < 5.0:
        cons.append({"text": f"⚠️ Thin Liq ₹{liq:.1f}cr/d", "kind": "warn"})

    # Keep leading missed chip first, then bad -> warn -> info
    kind_ord = {"bad": 0, "warn": 1, "info": 2}
    cons.sort(key=lambda c: (0 if c.get("_lead") else 1, kind_ord.get(c["kind"], 9)))
    for c in cons:
        c.pop("_lead", None)

    has_caution = any(c["kind"] in ("bad", "warn") for c in cons)
    return {
        "pros": pros[:5],
        "cons": cons[:4],
        "has_catalyst": has_catalyst,
        "has_caution": has_caution,
    }


def annotate_rows(P: pd.DataFrame, setup_lbls: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Convert a Funnel DataFrame into view-ready item dicts preserving input rank order."""
    if P is None or P.empty:
        return []
    out: List[Dict[str, Any]] = []
    for idx, (_, row) in enumerate(P.iterrows()):
        tk = _txt(row, "ticker")
        bare = tk[:-3] if tk.endswith(".NS") else tk
        sc = _num(row, "funnel_score")
        gr = _txt(row, "funnel_grade") or ("A" if (sc or 0) >= 74 else ("B" if (sc or 0) >= 60 else "C"))
        pil = parse_pillar_scores(_get(row, "score_note", ""))
        chips = build_signal_chips(row, pillars=pil)
        sec = _txt(row, "industry") or "Unclassified"
        setup = setup_lbls[idx] if (setup_lbls is not None and idx < len(setup_lbls)) else (_txt(row, "technicals") or "—")
        out.append({
            "rank": idx + 1,
            "ticker": tk,
            "bare": bare,
            "name": _txt(row, "name") or bare,
            "sector": sec,
            "ind_group": _txt(row, "ind_group").title(),
            "ind_score": _num(row, "ind_score"),
            "score": sc,
            "grade": gr,
            "price": _num(row, "price"),
            "rs": _num(row, "comp_rs"),
            "risk_pct": _num(row, "risk_pct"),
            "pe": _num(row, "pe"),
            "biz_score": _num(row, "biz_score"),
            "setup": setup,
            "pillars": pil,
            "chips": chips,
        })
    return out


def filter_annotated(items: List[Dict[str, Any]], filter_key: str = "ALL") -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Filter annotated rows by Signal Strip pill and return (filtered_items, counts_by_key)."""
    counts = {
        "ALL": len(items),
        "A": sum(1 for x in items if x["grade"] == "A"),
        "CAT": sum(1 for x in items if x["chips"]["has_catalyst"]),
        "CLEAN": sum(1 for x in items if not x["chips"]["has_caution"]),
        "WARN": sum(1 for x in items if x["chips"]["has_caution"]),
    }
    k = (filter_key or "ALL").upper()
    if k == "A":
        res = [x for x in items if x["grade"] == "A"]
    elif k == "CAT":
        res = [x for x in items if x["chips"]["has_catalyst"]]
    elif k == "CLEAN":
        res = [x for x in items if not x["chips"]["has_caution"]]
    elif k == "WARN":
        res = [x for x in items if x["chips"]["has_caution"]]
    else:
        res = list(items)
    return res, counts


def group_by_sector(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group annotated items by industry/sector sorted by Best-in-Theme score."""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for x in items:
        sec = x.get("sector") or "Unclassified"
        buckets.setdefault(sec, []).append(x)

    groups: List[Dict[str, Any]] = []
    for sec, members in buckets.items():
        ranked = sorted(
            members,
            key=lambda m: (
                -(m["score"] if m["score"] is not None else -1.0),
                -(m["rs"] if m["rs"] is not None else -999.0),
                m["bare"],
            ),
        )
        best = ranked[0]
        ind_groups = [m["ind_group"] for m in ranked if m["ind_group"]]
        ind_scores = [m["ind_score"] for m in ranked if m["ind_score"] is not None]
        groups.append({
            "sector": sec,
            "ind_group": ind_groups[0] if ind_groups else "",
            "ind_score": ind_scores[0] if ind_scores else None,
            "count": len(ranked),
            "grade_a_count": sum(1 for m in ranked if m["grade"] == "A"),
            "best": best,
            "top_score": best["score"],
            "items": ranked,
        })

    groups.sort(
        key=lambda g: (
            -(g["top_score"] if g["top_score"] is not None else -1.0),
            -g["count"],
            g["sector"],
        )
    )
    return groups


def css() -> str:
    """Scoped Stripe-light HTML/CSS styles matching utils/ui_components.py."""
    return (
        "<style>"
        "@keyframes fnEnter{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}"
        ".fn-strip-wrap{display:flex;flex-direction:column;gap:8px;margin-top:6px;margin-bottom:10px}"
        ".fn-strip-row{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:10px 14px;background:#FFFFFF;border:1px solid #E3E8EE;border-radius:10px;padding:10px 14px;box-shadow:0 1px 2px rgba(0,0,0,0.03);transition:border-color .15s ease,transform .15s ease,box-shadow .15s ease;animation:fnEnter .22s ease-out both}"
        ".fn-strip-row:hover{border-color:#635BFF;transform:translateY(-1px);box-shadow:0 4px 10px rgba(99,91,255,0.08)}"
        ".fn-strip-left{display:flex;align-items:center;gap:10px;min-width:225px;flex:0 0 auto}"
        ".fn-rank{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:#8792A2;width:24px;text-align:right}"
        ".fn-grade-badge{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;font-weight:700;padding:3px 8px;border-radius:6px;white-space:nowrap}"
        ".fn-grade-A{background:#E3FBEF;color:#008770;border:1px solid #A6EBD5}"
        ".fn-grade-B{background:#EEF4FF;color:#0054D2;border:1px solid #C6DAFC}"
        ".fn-grade-C{background:#FEE4E2;color:#D92D20;border:1px solid #FECDCA}"
        ".fn-tk-title{font-size:14px;font-weight:700;color:#1A1F36;line-height:1.25}"
        ".fn-tk-sub{font-size:12px;color:#4F566B;line-height:1.25;margin-top:1px}"
        ".fn-score-bar{width:130px;height:5px;background:#F3F4F6;border-radius:99px;overflow:hidden;margin-top:4px}"
        ".fn-score-fill{height:100%;border-radius:99px}"
        ".fn-chips{display:flex;flex-wrap:wrap;gap:5px;flex:1 1 280px;align-items:center}"
        ".fn-chip{display:inline-flex;align-items:center;padding:2px 8px;border-radius:6px;font-size:12px;font-weight:500;line-height:1.35;white-space:nowrap}"
        ".fn-chip-ok{background:#E3FBEF;color:#007A66;border:1px solid #BCEAD9}"
        ".fn-chip-info{background:#EEF4FF;color:#0054D2;border:1px solid #C6DAFC}"
        ".fn-chip-warn{background:#FFF6E5;color:#B54708;border:1px solid #FEDF89}"
        ".fn-chip-bad{background:#FEE4E2;color:#B42318;border:1px solid #FECDCA}"
        ".fn-strip-meta{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:#4F566B;text-align:right;white-space:nowrap;flex:0 0 auto}"
        ".fn-sector-card{background:#FFFFFF;border:1px solid #E3E8EE;border-radius:12px;padding:14px 16px;margin-bottom:8px;box-shadow:0 1px 2px rgba(0,0,0,0.03);transition:border-color .15s ease}"
        ".fn-sector-card-active{border:2px solid #635BFF;box-shadow:0 4px 12px rgba(99,91,255,0.12)}"
        ".fn-sec-hdr{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;border-bottom:1px solid #E3E8EE;padding-bottom:8px;margin-bottom:8px}"
        ".fn-sec-title{font-size:15px;font-weight:700;color:#1A1F36}"
        ".fn-sec-sub{font-size:12px;color:#4F566B;margin-top:2px}"
        ".fn-sec-list{display:flex;flex-direction:column;gap:5px}"
        ".fn-sec-row{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:5px 8px;border-radius:7px;font-size:12px}"
        ".fn-sec-row-best{background:#F7F9FC;border:1px solid #E3E8EE}"
        ".fn-sec-row-dim{opacity:0.68}"
        ".fn-sec-row-l{display:flex;align-items:center;gap:6px;min-width:0;flex:1}"
        ".fn-sec-snip{color:#4F566B;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}"
        ".fn-sec-row-r{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#4F566B;white-space:nowrap;font-size:12px}"
        ".fn-faceoff-wrap{background:#FFFFFF;border:1px solid #E3E8EE;border-radius:12px;padding:16px;margin-top:10px;margin-bottom:10px;box-shadow:0 1px 3px rgba(0,0,0,0.04)}"
        ".fn-faceoff-hdr{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:8px;border-bottom:1px solid #E3E8EE;padding-bottom:10px;margin-bottom:12px}"
        ".fn-faceoff-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}"
        ".fn-faceoff-card{background:#F7F9FC;border:1px solid #E3E8EE;border-radius:10px;padding:12px;display:flex;flex-direction:column;gap:8px}"
        ".fn-faceoff-top{display:flex;justify-content:space-between;align-items:center;gap:8px}"
        ".fn-faceoff-stats{display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:#4F566B}"
        ".fn-pillar-row{display:flex;align-items:center;gap:8px;font-size:12px}"
        ".fn-pillar-lbl{width:34px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#4F566B}"
        ".fn-pillar-track{flex:1;height:7px;background:#E3E8EE;border-radius:99px;overflow:hidden}"
        ".fn-pillar-fill{height:100%;border-radius:99px}"
        ".fn-pillar-val{width:42px;text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#1A1F36;font-weight:600}"
        "</style>"
    )


def _chip_html(chip: Dict[str, str]) -> str:
    cls_map = {
        "ok": "fn-chip-ok",
        "info": "fn-chip-info",
        "warn": "fn-chip-warn",
        "bad": "fn-chip-bad",
    }
    cls = cls_map.get(chip.get("kind", "info"), "fn-chip-info")
    return f'<span class="fn-chip {cls}">{html.escape(chip["text"], quote=True)}</span>'


def _grade_cls(gr: str) -> str:
    return f"fn-grade-{gr}" if gr in ("A", "B", "C") else "fn-grade-B"


def _bar_color(gr: str) -> str:
    return {"A": "#00A389", "B": "#635BFF", "C": "#D92D20"}.get(gr, "#635BFF")


def _fmt_price(px: Optional[float]) -> str:
    return f"₹{px:,.1f}" if px is not None else "—"


def _fmt_stop(rk: Optional[float]) -> str:
    return f"Stop {rk:.1f}%" if rk is not None else "Stop —"


def _fmt_rs(rs: Optional[float]) -> str:
    return f"RS {rs:+.1f}" if rs is not None else "RS —"


def _fmt_biz(bz: Optional[float]) -> str:
    return f"Biz {bz:.0f}" if bz is not None else "Biz —"


def signal_strip_html(items: List[Dict[str, Any]]) -> str:
    """Render Signal Strip rows with zero Markdown indentation issues."""
    if not items:
        return ""
    parts = ['<div class="fn-strip-wrap">']
    for item in items:
        sc = item["score"]
        sc_txt = f"{sc:.0f}" if sc is not None else "—"
        sc_pct = max(0.0, min(100.0, float(sc))) if sc is not None else 0.0
        gr = item["grade"]
        chips = item["chips"]["pros"] + item["chips"]["cons"]
        if not item["chips"]["has_caution"]:
            chips = list(chips) + [{"text": "✅ Zero Cautions", "kind": "ok"}]
        chips_str = "".join(_chip_html(c) for c in chips)
        meta_txt = (
            f"{_fmt_price(item['price'])} · {_fmt_rs(item['rs'])} · "
            f"{_fmt_stop(item['risk_pct'])} · {_fmt_biz(item['biz_score'])} · {item['setup']}"
        )
        parts.append(
            f'<div class="fn-strip-row">'
            f'<div class="fn-strip-left">'
            f'<span class="fn-rank">#{item["rank"]}</span>'
            f'<span class="fn-grade-badge {_grade_cls(gr)}">{html.escape(sc_txt)} {html.escape(gr)}</span>'
            f'<div>'
            f'<div class="fn-tk-title">{html.escape(item["bare"])} '
            f'<span style="font-weight:400;color:#4F566B;font-size:12px;">{html.escape(item["name"][:24])}</span></div>'
            f'<div class="fn-tk-sub">{html.escape(item["sector"])}</div>'
            f'<div class="fn-score-bar"><div class="fn-score-fill" style="width:{sc_pct:.1f}%;background:{_bar_color(gr)};"></div></div>'
            f'</div>'
            f'</div>'
            f'<div class="fn-chips">{chips_str}</div>'
            f'<div class="fn-strip-meta">{html.escape(meta_txt)}</div>'
            f'</div>'
        )
    parts.append("</div>")
    return "".join(parts)


def sector_card_html(group: Dict[str, Any], selected: bool = False) -> str:
    """Render a Best-in-Theme Sector Card with ranked tickers."""
    best = group["best"]
    best_sc = f"{best['score']:.0f}" if best["score"] is not None else "—"
    ind_bits: List[str] = []
    if group.get("ind_group"):
        ind_bits.append(group["ind_group"])
    if group.get("ind_score") is not None:
        ind_bits.append(f"{group['ind_score']:.0f}")
    ind_badge = (
        f'<span class="fn-chip fn-chip-info">{html.escape(" · ".join(ind_bits))}</span>'
        if ind_bits else ""
    )

    rows_html: List[str] = []
    for idx, m in enumerate(group["items"]):
        sc_txt = f"{m['score']:.0f}" if m["score"] is not None else "—"
        gr = m["grade"]
        icon = "🏆" if idx == 0 else ("⚠️" if gr == "C" else "•")
        row_cls = "fn-sec-row" + (" fn-sec-row-best" if idx == 0 else "") + (" fn-sec-row-dim" if gr == "C" else "")
        first_con = m["chips"]["cons"][0]["text"] if m["chips"]["cons"] else ""
        first_pro = m["chips"]["pros"][0]["text"] if m["chips"]["pros"] else ""
        snip = first_con or first_pro or m["setup"]
        r_meta = f"{_fmt_stop(m['risk_pct'])} · {_fmt_biz(m['biz_score'])}"
        rows_html.append(
            f'<div class="{row_cls}">'
            f'<div class="fn-sec-row-l">'
            f'<span>{icon}</span>'
            f'<strong style="color:#1A1F36;">{html.escape(m["bare"])}</strong>'
            f'<span class="fn-grade-badge {_grade_cls(gr)}" style="padding:1px 6px;">{html.escape(sc_txt)}</span>'
            f'<span class="fn-sec-snip">{html.escape(snip)}</span>'
            f'</div>'
            f'<span class="fn-sec-row-r">{html.escape(r_meta)}</span>'
            f'</div>'
        )

    card_cls = "fn-sector-card" + (" fn-sector-card-active" if selected else "")
    return (
        f'<div class="{card_cls}">'
        f'<div class="fn-sec-hdr">'
        f'<div>'
        f'<div class="fn-sec-title">{html.escape(group["sector"])} '
        f'<span style="font-weight:500;font-size:12px;color:#8792A2;">({group["count"]})</span></div>'
        f'<div class="fn-sec-sub">Best-in-Theme: <strong style="color:#1A1F36;">🏆 {html.escape(best["bare"])}</strong> '
        f'({html.escape(best_sc)} {html.escape(best["grade"])})</div>'
        f'</div>'
        f'{ind_badge}'
        f'</div>'
        f'<div class="fn-sec-list">{"".join(rows_html)}</div>'
        f'</div>'
    )


def _pillar_bars_html(pillars: Dict[str, int]) -> str:
    rows: List[str] = []
    for k in ("Mom", "Set", "Str", "Qual", "Cat"):
        v = int(pillars.get(k, 0))
        mx = PILLAR_MAX[k]
        pct = max(0.0, min(100.0, (max(0, v) / mx) * 100.0))
        col = "#D92D20" if v < 0 else PILLAR_COLORS[k]
        if v < 0:
            pct = 8.0
        rows.append(
            f'<div class="fn-pillar-row">'
            f'<span class="fn-pillar-lbl">{k}</span>'
            f'<div class="fn-pillar-track"><div class="fn-pillar-fill" style="width:{pct:.0f}%;background:{col};"></div></div>'
            f'<span class="fn-pillar-val">{v}/{mx}</span>'
            f'</div>'
        )
    return "".join(rows)


def sector_faceoff_html(group: Dict[str, Any], max_peers: int = 3) -> str:
    """Render side-by-side Instant Sector Face-Off comparison cards for top peers."""
    if not group or not group.get("items"):
        return ""
    peers = group["items"][:max_peers]
    cards: List[str] = []
    for idx, m in enumerate(peers):
        sc_txt = f"{m['score']:.0f}" if m["score"] is not None else "—"
        gr = m["grade"]
        pe_txt = f"P/E {m['pe']:.1f}×" if m["pe"] is not None else "P/E —"
        stats_line = f"{_fmt_price(m['price'])} · {_fmt_rs(m['rs'])} · {_fmt_stop(m['risk_pct'])} · {pe_txt}"
        chips_html = "".join(_chip_html(c) for c in (m["chips"]["pros"][:3] + m["chips"]["cons"][:2]))
        cards.append(
            f'<div class="fn-faceoff-card">'
            f'<div class="fn-faceoff-top">'
            f'<div><strong style="font-size:14px;color:#1A1F36;">{"🏆 " if idx == 0 else ""}{html.escape(m["bare"])}</strong>'
            f'<div style="font-size:12px;color:#4F566B;">{html.escape(m["setup"])}</div></div>'
            f'<span class="fn-grade-badge {_grade_cls(gr)}">{html.escape(sc_txt)} {html.escape(gr)}</span>'
            f'</div>'
            f'<div class="fn-faceoff-stats">{html.escape(stats_line)}</div>'
            f'<div>{_pillar_bars_html(m["pillars"])}</div>'
            f'<div class="fn-chips">{chips_html}</div>'
            f'</div>'
        )
    return (
        f'<div class="fn-faceoff-wrap">'
        f'<div class="fn-faceoff-hdr">'
        f'<span style="font-size:14px;font-weight:700;color:#1A1F36;">⚖️ Instant Sector Face-Off: {html.escape(group["sector"])} (Top {len(peers)})</span>'
        f'<span style="font-size:12px;color:#4F566B;">Side-by-side 5-pillar edge — pick the cleanest leader per theme</span>'
        f'</div>'
        f'<div class="fn-faceoff-grid">{"".join(cards)}</div>'
        f'</div>'
    )
