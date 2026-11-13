"""Helpers for the daily master snapshots (data/cache/market_master_YYYY_MM_DD.parquet)."""
from __future__ import annotations

import glob
import os
import re
from typing import Optional

import pandas as pd

CACHE_GLOB = "data/cache/market_master_*.parquet"


def file_date(path: str) -> Optional[pd.Timestamp]:
    m = re.search(r"(\d{4})_(\d{2})_(\d{2})", os.path.basename(path))
    return pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}") if m else None


def bar_date(df: pd.DataFrame, path: Optional[str] = None) -> Optional[pd.Timestamp]:
    """
    The trading session a snapshot's prices are from: its bar_date stamp (engine, Nov 2026 on),
    else the date in its filename. The filename is the run's wall-clock date, which is wrong on
    holidays and after late runs -- use the stamp whenever it exists.
    """
    if df is not None and "bar_date" in df.columns and df["bar_date"].notna().any():
        return pd.Timestamp(df["bar_date"].dropna().iloc[0]).normalize()
    return file_date(path) if path else None


def latest(n: int = 1) -> list:
    """Paths of the newest n snapshots, newest last."""
    return sorted(glob.glob(CACHE_GLOB))[-n:]


def is_new_session(today: pd.DataFrame, prev: Optional[pd.DataFrame]) -> bool:
    """False when `today` holds the same session as `prev` (market holiday or a re-run)."""
    if prev is None or prev.empty:
        return True
    a, b = bar_date(today), bar_date(prev)
    if a is not None and b is not None:
        return a > b
    j = today.merge(prev, on="ticker", suffixes=("", "_p"))
    col = "currentPrice" if "currentPrice" in j.columns else "price"
    return not (len(j) and (j[col] == j[f"{col}_p"]).mean() > 0.95)
