"""
All-time-high (ATH) breakouts for the Trend Scanner.

The scanner downloads one year of prices, which cannot tell a new all-time
high from a 52-week high. data/ath_levels.csv carries, per ticker, the highest
adjusted close up to a date ~32 sessions back (the "base"); seed_ath.py fills
it once from full history and the daily engine rolls it forward.

Why a base 32 sessions back instead of the running ATH: a breakout is "a new
all-time closing high after >= 20 sessions without one" (the backtest's
definition, breakout_sector_study.py), checked over the last 10 sessions. That
needs the true all-time max as of each of the last ~31 days:

    prior_max(day i) = max(base_max, closes after base_asof and before day i)

which is exact as long as the one-year window still reaches base_asof. A running
ATH would already include those days and could not be unwound.

Columns: ticker, base_max, base_asof, ath_close, ath_date, history_start, sessions
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ATH_FILE = "data/ath_levels.csv"
BASE_LAG = 32            # base date = 32 sessions back: covers 10 fresh + 20 base + 1
FRESH_SESSIONS = 10
BASE_SESSIONS = 20
MIN_HISTORY = 500        # ~2 years listed; before that "all-time" means little (as in the backtest)
COLS = ["ticker", "base_max", "base_asof", "ath_close", "ath_date", "history_start", "sessions"]


def load_ath(path: str = ATH_FILE) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        t = pd.read_csv(path, parse_dates=["base_asof", "ath_date", "history_start"])
        return {r["ticker"]: r for r in t.to_dict("records")}
    except Exception:
        return {}


def save_ath(rows: dict, path: str = ATH_FILE) -> None:
    if not rows:
        return
    from utils.atomic_io import atomic_to_csv
    t = pd.DataFrame(list(rows.values()))[COLS].sort_values("ticker")
    t[["base_max", "ath_close"]] = t[["base_max", "ath_close"]].astype(float).round(4)   # stable CSV round-trip
    for c in ("base_asof", "ath_date", "history_start"):
        t[c] = pd.to_datetime(t[c]).dt.strftime("%Y-%m-%d")
    atomic_to_csv(t, path, index=False)


def seed_row(ticker: str, close: pd.Series) -> dict | None:
    """Row from a full price history (seed_ath.py)."""
    c = close.dropna()
    if len(c) <= BASE_LAG:
        return None
    base = c.iloc[:-BASE_LAG]
    return dict(ticker=ticker, base_max=float(base.max()), base_asof=base.index[-1],
                ath_close=float(c.max()), ath_date=c.idxmax(),
                history_start=c.index[0], sessions=len(base))      # sessions up to base_asof


def ath_breakout(df: pd.DataFrame, row: dict | None) -> tuple[dict, dict | None]:
    """
    Fresh ATH breakout from the engine's 1-year df and the stored row.
    Returns (fields for the scanner, updated row to store or None to keep).
    Fields: ath_bo_days (sessions ago, NaN if none), ath_bo_anchor, ath_bo_volx, dist_ath (%).
    """
    out = {"ath_bo_days": np.nan, "ath_bo_anchor": np.nan, "ath_bo_volx": np.nan, "dist_ath": np.nan}
    if row is None or df is None or "Close" not in df or len(df) <= BASE_LAG:
        return out, None
    c = df["Close"].astype(float)
    base_asof = pd.Timestamp(row["base_asof"])
    if c.index[0] > base_asof:                  # window no longer reaches the base: cannot be exact
        return out, None
    base_max = float(row["base_max"])
    after = c[c.index > base_asof]
    ath_now = max(base_max, float(after.max()) if len(after) else base_max)
    out["dist_ath"] = (float(c.iloc[-1]) / ath_now - 1) * 100

    sessions = int(row.get("sessions", 0)) + len(after)
    if sessions >= MIN_HISTORY and len(after) >= BASE_SESSIONS + 1:
        vals = after.values
        prior = np.maximum.accumulate(np.r_[base_max, vals[:-1]])      # max before each day
        new = vals > prior
        n = len(vals)
        for k in range(n - 1, max(n - 1 - FRESH_SESSIONS, BASE_SESSIONS) - 1, -1):
            if new[k] and not new[k - BASE_SESSIONS:k].any():
                out["ath_bo_days"] = n - 1 - k
                from utils.breakout_tags import vol_stats
                out["ath_bo_anchor"], out["ath_bo_volx"] = vol_stats(df, len(c) - n + k)
                break

    # roll the base forward to BASE_LAG sessions back
    new_asof = c.index[-BASE_LAG]
    moved = c[(c.index > base_asof) & (c.index <= new_asof)]
    new_row = dict(row)
    new_row.update(base_max=max(base_max, float(moved.max())) if len(moved) else base_max,
                   base_asof=max(base_asof, new_asof), ath_close=ath_now,
                   ath_date=(after.idxmax() if len(after) and after.max() > base_max else row["ath_date"]),
                   sessions=int(row.get("sessions", 0)) + len(moved))
    return out, new_row
