"""
Breakout / industry / edge columns for the Trend Scanner.

Evidence (2016-2026, point-in-time top 1000; analysis/BREAKOUT_SECTOR_VERDICT.md
and analysis/scanner_tags_study.csv):
  * a new 52-week or all-time closing high in a LEADER sub-industry (rotation
    score >= 70) beat the typical stock by ~+2 to +3 pp over 3 months and
    ~+3.5 to +4.5 pp over 6 months; a volume anchor added ~+1 pp;
  * the same breakout in a laggard industry was worth about a third of that;
  * the scanner's entry tags differ less than their names suggest: Actionable
    +0.6 pp (3m), Pullback +0.3, Extended +1.0 (weak first 2 weeks), Fading
    +0.2, Weak -0.9 -- and every tag does better in a leader industry.
  * Preferring breakouts inside the live portfolio engine did NOT beat its
    6-month RS ranking -- these columns are for discretionary screening.

Engine side  hi52_breakout(df)   fresh 52-week-high breakout from daily bars
             (ATH breakouts: utils/ath.py)
UI side      add_breakout_tags(df)  ind_score, bo_type, bo_days, bo_anchor,
             breakout_label (with the breakout day's volume), dist_ath,
             edge_3m / edge_6m / edge_years
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

FRESH_SESSIONS = 10          # "fresh" = broke out within the last 2 weeks
BASE_SESSIONS = 20           # sessions without a new high before a breakout
MIN_HISTORY = 200
LEADER, LAGGARD = 70, 40     # rotation heatmap bands (score_0_100)
TAG_STUDY = "analysis/scanner_tags_study.csv"

BREAKOUT_OPTIONS = ["ATH breakout", "52W breakout"]
BREAKOUT_HELP = ("Fresh = a new closing high in the last 2 weeks after >= 20 sessions without one. "
                 "ATH = all-time high (data/ath_levels.csv), 52W = 52-week high. In a leader industry these beat "
                 "the typical stock by ~+2 to +3 pp over 3 months and ~+4 pp over 6 months (2016-2026). "
                 "The label ends with the breakout day's volume vs its 50-day median; 🔊 = a volume anchor "
                 "(>= 3x, >= Rs 5 cr turnover, CLV >= 0.5), worth ~+1 pp more. A volume spike WITHOUT a breakout "
                 "has no edge of its own, so it is not flagged. Blank = no fresh breakout. "
                 "Nothing selected = no breakout filter.")


# ----------------------------------------------------------------------------
# engine side
# ----------------------------------------------------------------------------
def hi52_breakout(df: pd.DataFrame) -> dict:
    """
    Most recent 52-week closing-high breakout within FRESH_SESSIONS.
    Returns {'hi52_bo_days': sessions ago (0 = today) or NaN,
             'hi52_bo_anchor': True/False (NaN if no fresh breakout),
             'hi52_bo_volx': breakout-day volume / prior 50-session median}.
    """
    out = {"hi52_bo_days": np.nan, "hi52_bo_anchor": np.nan, "hi52_bo_volx": np.nan}
    if df is None or "Close" not in df or len(df) < MIN_HISTORY + 1:
        return out
    c = df["Close"].astype(float).values
    n = len(c)
    prior_max = pd.Series(c).shift(1).rolling(251, min_periods=MIN_HISTORY).max().values
    new_high = np.zeros(n, dtype=bool)
    ok = ~np.isnan(prior_max)
    new_high[ok] = c[ok] >= prior_max[ok]
    for i in range(n - 1, max(n - 1 - FRESH_SESSIONS, BASE_SESSIONS) - 1, -1):
        if new_high[i] and not new_high[i - BASE_SESSIONS:i].any() and ok[i - BASE_SESSIONS]:
            out["hi52_bo_days"] = n - 1 - i
            out["hi52_bo_anchor"], out["hi52_bo_volx"] = vol_stats(df, i)
            return out
    return out


def vol_stats(df: pd.DataFrame, i: int) -> tuple[bool, float]:
    """
    (volume anchor?, volume multiple) for bar i. Anchor: volume >= 3x the prior
    50-session median, turnover >= Rs 5 cr, CLV >= 0.5.
    """
    if not {"Volume", "High", "Low"} <= set(df.columns) or i < 30:
        return False, np.nan
    v = df["Volume"].astype(float).values
    med = np.nanmedian(v[max(0, i - 50):i])
    volx = v[i] / med if med > 0 else np.nan
    h, l, c = float(df["High"].iloc[i]), float(df["Low"].iloc[i]), float(df["Close"].iloc[i])
    clv = (c - l) / (h - l) if h > l else np.nan
    return bool(med > 0 and v[i] >= 3 * med and c * v[i] >= 5e7 and clv >= 0.5), round(float(volx), 1)


# ----------------------------------------------------------------------------
# UI side
# ----------------------------------------------------------------------------
def load_industry_scores(path: str = "data/sub_industry_rotation.csv") -> dict:
    """Latest score_0_100 per sub-industry from the rotation history."""
    try:
        r = pd.read_csv(path)
        r["record_date"] = pd.to_datetime(r["record_date"], format="mixed")
        last = r[r.record_date == r.record_date.max()]
        return dict(zip(last["sub_industry"], pd.to_numeric(last["score_0_100"], errors="coerce")))
    except Exception:
        return {}


def load_tag_edges(path: str = TAG_STUDY) -> dict:
    """{(view, tag): (3m pp, 6m pp, 'years')} from the scanner-tag backtest."""
    if not os.path.exists(path):
        return {}
    t = pd.read_csv(path)
    return {(r["view"], r["case"]): (r.get("3mo vsTypical"), r.get("6mo vsTypical"), r.get("yrs 3mo>0"))
            for r in t.to_dict("records")}


def _edge_view(band: str, fresh: bool) -> list[str]:
    """Most specific backtest cell first, broader fallbacks after."""
    views = []
    if fresh and band == "leader":
        views.append("52w/ATH breakout, leader industry")
    if fresh:
        views.append("52w/ATH breakout in last 2 weeks")
    if band:
        views.append(f"{band} industry")
    views.append("all stocks")
    return views


def add_breakout_tags(df: pd.DataFrame, scores: dict | None = None, sub_map: dict | None = None,
                      ath: dict | None = None, edges: dict | None = None) -> pd.DataFrame:
    """Adds the scanner's breakout / industry / edge columns. Safe on an older cache."""
    out = df.copy()
    if scores is None:
        scores = load_industry_scores()
    if sub_map is None:
        from utils.nifty1000_list import SUB_INDUSTRY_MAP as sub_map
    if ath is None:
        from utils.ath import load_ath
        ath = load_ath()
    if edges is None:
        edges = load_tag_edges()
    col = lambda c: out[c] if c in out else pd.Series(np.nan, index=out.index)

    tick = col("ticker")
    out["ind_score"] = pd.to_numeric(tick.map(sub_map).map(scores), errors="coerce")
    band = np.select([out.ind_score >= LEADER, out.ind_score < LAGGARD, out.ind_score.notna()],
                     ["leader", "laggard", "mid"], "")

    d52 = pd.to_numeric(col("hi52_bo_days"), errors="coerce")
    dath = pd.to_numeric(col("ath_bo_days"), errors="coerce")
    f52 = d52.notna() & (d52 <= FRESH_SESSIONS)
    fath = dath.notna() & (dath <= FRESH_SESSIONS)
    out["bo_type"] = np.select([fath, f52], ["ATH", "52W"], "")
    out["bo_days"] = np.where(fath, dath, np.where(f52, d52, np.nan))
    out["bo_anchor"] = (fath & col("ath_bo_anchor").eq(True)) | (~fath & f52 & col("hi52_bo_anchor").eq(True))
    out["bo_volx"] = pd.to_numeric(np.where(fath, col("ath_bo_volx"), np.where(f52, col("hi52_bo_volx"), np.nan)),
                                   errors="coerce")
    out["breakout_label"] = [
        "" if not t else f"⚡ {t} · {int(d)}d ago" + (
            f" · 🔊 {x:.1f}× vol" if a else f" · {x:.1f}× vol" if pd.notna(x) else "")
        for t, d, a, x in zip(out.bo_type, out.bo_days, out.bo_anchor, out.bo_volx)]

    # % from all-time high: engine value when present, else from the stored table
    price = pd.to_numeric(col("price").fillna(col("currentPrice")), errors="coerce")
    stored = pd.to_numeric(tick.map({t: r.get("ath_close") for t, r in ath.items()}), errors="coerce")
    athv = np.fmax(stored, price)
    out["dist_ath"] = pd.to_numeric(col("dist_ath"), errors="coerce").fillna((price / athv - 1) * 100)

    # historical edge of this tag x industry band x breakout cell
    tags = col("entry_label").astype(str).str.replace(r"^\S+\s", "", regex=True)
    e3, e6, ey = [], [], []
    for tg, b, fr in zip(tags, band, (fath | f52)):
        hit = next((edges[(v, tg)] for v in _edge_view(b, fr)
                    if (v, tg) in edges and pd.notna(edges[(v, tg)][0])), (np.nan, np.nan, ""))
        e3.append(hit[0]); e6.append(hit[1]); ey.append(hit[2] if isinstance(hit[2], str) else "")
    out["edge_3m"], out["edge_6m"], out["edge_years"] = e3, e6, ey
    return out
