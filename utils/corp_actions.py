"""
Corporate actions (splits, bonuses, consolidations) seen by the daily snapshots.

Signal logs and the position tracker store the price on the day of a signal or purchase. Yahoo
then back-adjusts its whole price series when a stock splits or issues a bonus, so the next
snapshot's price is on a new basis and "return since signal" becomes nonsense (BUILDPRO -82.8%
in three weeks, CCAVENUE +702%: neither happened).

Each snapshot carries currentPrice and previousClose from the SAME download, i.e. on the same
adjustment basis. So for consecutive snapshots A then B of one stock:
    factor = B.previousClose / A.currentPrice
is 1 when nothing happened (both are the close of A's day) and the adjustment factor when an
action took effect in between (0.5 for a 1:1 bonus or 2-for-1 split, 10 for a 10:1 consolidation).
A price stored before the action is put on today's basis by multiplying it by the factor.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd

CACHE_GLOB = "data/cache/market_master_*.parquet"
TABLE = "data/corporate_actions.csv"
COLS = ["ticker", "date", "factor", "prev_snapshot"]
MIN_MOVE = 0.20          # |factor - 1| below this is noise
MIN_MOVE_GAP = 0.45      # ... when sessions are missing between the two snapshots
MAX_GAP_DAYS = 5         # snapshots further apart than this are not compared


def _date(path: str) -> pd.Timestamp:
    m = re.search(r"(\d{4})_(\d{2})_(\d{2})", os.path.basename(path))
    return pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")


def factors(prev: pd.DataFrame, cur: pd.DataFrame, consecutive: bool = True) -> pd.Series:
    """
    ticker -> adjustment factor between two snapshots (only where an action happened).

    Not every mismatch is an action. If Yahoo had not yet published a stock's bar when the earlier
    snapshot was taken, the "factor" is that day's real move (ITI read 1.20, an upper circuit).
    So small upward factors are ignored, and when sessions are missing between the two snapshots
    only a large factor counts (a real two-day move can reach 30-40%).
    """
    if "previousClose" not in cur.columns or "currentPrice" not in prev.columns:
        return pd.Series(dtype=float)
    a = pd.to_numeric(prev.set_index("ticker")["currentPrice"], errors="coerce")
    b = pd.to_numeric(cur.set_index("ticker")["previousClose"], errors="coerce")
    f = (b / a.reindex(b.index)).replace([np.inf, -np.inf], np.nan).dropna()
    need = MIN_MOVE if consecutive else MIN_MOVE_GAP
    f = f[(f > 0) & ((f - 1).abs() >= need) & ~((f > 1) & (f <= 1.3))]
    return f.round(6)


def _consecutive(prev_date: pd.Timestamp, cur_date: pd.Timestamp) -> bool:
    return len(pd.bdate_range(prev_date, cur_date)) == 2


def build(save: bool = True) -> pd.DataFrame:
    """Scan every cached snapshot pair and (re)write the table of detected actions."""
    files = sorted(glob.glob(CACHE_GLOB))
    rows, prev, prev_f = [], None, None
    for f in files:
        try:
            cur = pd.read_parquet(f, columns=["ticker", "currentPrice", "previousClose"])
        except Exception:
            prev, prev_f = None, None
            continue
        if prev is not None and (_date(f) - _date(prev_f)).days <= MAX_GAP_DAYS:
            # skip duplicate snapshots (holiday / failed fetch): identical prices mean previousClose
            # in `cur` is not prev's close
            j = cur.merge(prev, on="ticker", suffixes=("", "_p"))
            if len(j) and (j.currentPrice == j.currentPrice_p).mean() < 0.9:
                for t, fac in factors(prev, cur, _consecutive(_date(prev_f), _date(f))).items():
                    rows.append(dict(ticker=t, date=_date(f).strftime("%Y-%m-%d"), factor=float(fac),
                                     prev_snapshot=_date(prev_f).strftime("%Y-%m-%d")))
        prev, prev_f = cur, f
    out = pd.DataFrame(rows, columns=COLS)
    if os.path.exists(TABLE):                      # keep events whose snapshots have since been pruned
        try:
            old = pd.read_csv(TABLE)
            out = pd.concat([old, out], ignore_index=True).drop_duplicates(["ticker", "date"], keep="last")
        except Exception:
            pass
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    if save:
        out.to_csv(TABLE, index=False)
    return out


def load() -> pd.DataFrame:
    if os.path.exists(TABLE):
        try:
            return pd.read_csv(TABLE)
        except Exception:
            pass
    return pd.DataFrame(columns=COLS)


def factor_map(table: pd.DataFrame, tickers, since, until=None) -> pd.Series:
    """Vectorised cumulative_factor: one factor per (ticker, since[, until]) row, 1.0 where none."""
    tickers, since = pd.Series(list(tickers)), pd.to_datetime(pd.Series(list(since)))
    until = pd.to_datetime(pd.Series(list(until))) if until is not None else None
    out = np.ones(len(tickers))
    if table is None or table.empty:
        return pd.Series(out)
    hit = tickers.isin(set(table.ticker)).values
    for i in np.flatnonzero(hit):
        out[i] = cumulative_factor(table, tickers.iloc[i], since.iloc[i], None if until is None else until.iloc[i])
    return pd.Series(out)


def cumulative_factor(table: pd.DataFrame, ticker: str, since, until=None) -> float:
    """Product of the factors of every action on `ticker` after `since` (exclusive) up to `until`."""
    if table is None or table.empty:
        return 1.0
    t = table[table.ticker == ticker]
    if t.empty:
        return 1.0
    d = pd.to_datetime(t.date)
    m = d > pd.Timestamp(since)
    if until is not None:
        m &= d <= pd.Timestamp(until)
    return float(np.prod(t.factor[m])) if m.any() else 1.0
