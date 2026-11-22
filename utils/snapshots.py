"""Helpers for the daily master snapshots (data/cache/market_master_YYYY_MM_DD.parquet)."""
from __future__ import annotations

from functools import lru_cache
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


def _session_probe(path: str) -> tuple[Optional[pd.Timestamp], Optional[pd.DataFrame]]:
    """Read minimal columns to identify a snapshot's trading session date and prices."""
    try:
        import pyarrow.parquet as pq
        names = pq.read_schema(path).names
        cols = [c for c in ("ticker", "currentPrice", "price", "bar_date") if c in names]
        df = pd.read_parquet(path, columns=cols) if cols else pd.read_parquet(path)
    except Exception:
        try:
            df = pd.read_parquet(path)
        except Exception:
            return None, None
    if df is None or df.empty:
        return None, None
    return bar_date(df, path), df


@lru_cache(maxsize=4)
def _unique_session_paths(file_sig: tuple[tuple[str, float], ...]) -> tuple[tuple[str, Optional[pd.Timestamp]], ...]:
    """Deduplicate consecutive holiday/re-run snapshot files into unique trading sessions."""
    uniq: list[tuple[str, Optional[pd.Timestamp]]] = []
    prev_df: Optional[pd.DataFrame] = None
    for path, _mtime in file_sig:
        dt, cur_df = _session_probe(path)
        if cur_df is None:
            continue
        if prev_df is None or is_new_session(cur_df, prev_df):
            uniq.append((path, dt))
            prev_df = cur_df
    return tuple(uniq)


def unique_session_files(files: Optional[list[str]] = None) -> list[tuple[str, Optional[pd.Timestamp]]]:
    """Chronological `(path, session_date)` pairs with holiday/duplicate snapshots removed."""
    paths = sorted(glob.glob(CACHE_GLOB)) if files is None else sorted(files)
    sig = tuple((p, os.path.getmtime(p)) for p in paths if os.path.exists(p))
    return list(_unique_session_paths(sig))


def snapshot_n_sessions_ago(
    sessions: int = 1,
    current_df: Optional[pd.DataFrame] = None,
    files: Optional[list[str]] = None,
) -> Optional[str]:
    """
    Path to the snapshot `sessions` unique trading sessions before `current_df` (or the newest file).
    Skips holiday/duplicate snapshot files so N-session lookbacks measure real trading days.
    """
    if sessions < 1:
        return None
    uniq = unique_session_files(files)
    if not uniq:
        return None
    if current_df is not None:
        cur_dt = bar_date(current_df)
        if cur_dt is not None:
            before = [(p, dt) for p, dt in uniq if dt is not None and dt < cur_dt]
        else:
            before = list(uniq)
            _, last_df = _session_probe(before[-1][0])
            if last_df is not None and not is_new_session(current_df, last_df):
                before = before[:-1]
        return before[-sessions][0] if len(before) >= sessions else None
    return uniq[-1 - sessions][0] if len(uniq) > sessions else None
