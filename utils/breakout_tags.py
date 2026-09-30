"""
52-week-high breakout tags for the Trend Scanner.

Backtest (analysis/BREAKOUT_SECTOR_VERDICT.md, 2016-2026): a new 52-week
closing high in a LEADER sub-industry (rotation score >= 70) beat the typical
stock by +2.2 pp over 3 months and +3.5 pp over 6 months (11 of 12 years);
with a volume anchor +2.9 / +4.2 pp. The same breakout in a laggard industry
was worth a third as much. Among the scanner's entry tags, Actionable with a
fresh 52w/ATH breakout was the most consistent (+1.7 / +2.5 pp, 12 of 12
years) (analysis/scanner_tags_study.csv).

Two halves:
  hi52_breakout(df)        engine side -- run per stock on its daily history
  add_breakout_tags(df)    UI side -- combines it with the sub-industry score

Definitions match breakout_sector_study.py:
  breakout      close >= the highest close of the prior 251 sessions (needs
                >= 200 sessions of history), after >= 20 sessions without one
  fresh         the breakout was within the last FRESH_SESSIONS sessions
  volume anchor on the breakout day: volume >= 3x the median of the prior 50
                sessions, close x volume >= Rs 5 crore, CLV >= 0.5
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FRESH_SESSIONS = 10          # "fresh" = broke out within the last 2 weeks
BASE_SESSIONS = 20           # sessions without a new high before a breakout
MIN_HISTORY = 200
LEADER, LAGGARD = 70, 40     # rotation heatmap bands (score_0_100)

TAG_LEADER = "⚡ Leader Breakout"
TAG_MID = "⚡ 52W Breakout"
TAG_WEAK = "🔸 52W Breakout · weak industry"
TAG_LEADER_IND = "🟢 Leader Industry"
TAG_NONE = "—"
TAG_ORDER = [TAG_LEADER, TAG_MID, TAG_WEAK, TAG_LEADER_IND, TAG_NONE]
TAG_HELP = ("⚡ Leader Breakout = new 52-week closing high in the last 2 weeks in a leader sub-industry "
            "(score >= 70) -- beat the typical stock by ~+2 to +3 pp over 3 months in 2016-2026 · "
            "⚡ 52W Breakout = same, mid-ranked industry · 🔸 = breakout in a laggard industry (little edge) · "
            "🟢 Leader Industry = no fresh breakout, industry score >= 70")


def hi52_breakout(df: pd.DataFrame) -> dict:
    """
    Most recent 52-week closing-high breakout within FRESH_SESSIONS.
    Returns {'hi52_bo_days': sessions ago (0 = today) or NaN,
             'hi52_bo_anchor': True/False (NaN if no fresh breakout)}.
    df: daily bars with Close (and High / Low / Volume for the anchor).
    """
    out = {"hi52_bo_days": np.nan, "hi52_bo_anchor": np.nan}
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
            out["hi52_bo_anchor"] = _anchor(df, i)
            return out
    return out


def _anchor(df: pd.DataFrame, i: int) -> bool:
    if not {"Volume", "High", "Low"} <= set(df.columns) or i < 30:
        return False
    v = df["Volume"].astype(float).values
    med = np.nanmedian(v[max(0, i - 50):i])
    h, l, c = float(df["High"].iloc[i]), float(df["Low"].iloc[i]), float(df["Close"].iloc[i])
    clv = (c - l) / (h - l) if h > l else np.nan
    return bool(med > 0 and v[i] >= 3 * med and c * v[i] >= 5e7 and clv >= 0.5)


def load_industry_scores(path: str = "data/sub_industry_rotation.csv") -> dict:
    """Latest score_0_100 per sub-industry from the rotation history."""
    try:
        r = pd.read_csv(path)
        r["record_date"] = pd.to_datetime(r["record_date"], format="mixed")
        last = r[r.record_date == r.record_date.max()]
        return dict(zip(last["sub_industry"], pd.to_numeric(last["score_0_100"], errors="coerce")))
    except Exception:
        return {}


def add_breakout_tags(df: pd.DataFrame, scores: dict | None = None, sub_map: dict | None = None) -> pd.DataFrame:
    """Adds ind_score and breakout_tag. Safe when the engine columns are missing (older cache)."""
    out = df.copy()
    if scores is None:
        scores = load_industry_scores()
    if sub_map is None:
        from utils.nifty1000_list import SUB_INDUSTRY_MAP as sub_map
    ind = out["ticker"].map(sub_map) if "ticker" in out else pd.Series(np.nan, index=out.index)
    out["ind_score"] = pd.to_numeric(ind.map(scores), errors="coerce")
    days = pd.to_numeric(out.get("hi52_bo_days", pd.Series(np.nan, index=out.index)), errors="coerce")
    fresh = days.notna() & (days <= FRESH_SESSIONS)
    s = out["ind_score"]
    tag = np.select(
        [fresh & (s >= LEADER), fresh & (s < LAGGARD), fresh, s >= LEADER],
        [TAG_LEADER, TAG_WEAK, TAG_MID, TAG_LEADER_IND], TAG_NONE)
    out["breakout_tag"] = tag
    # True only for a fresh breakout on a volume anchor (engine writes True/False/NaN)
    out["hi52_bo_anchor"] = out.get("hi52_bo_anchor", pd.Series(np.nan, index=out.index)).eq(True) & fresh
    return out
