#!/usr/bin/env python3
"""
One-time (re-runnable) back-fill of the NSE price store (utils/nse_prices.py): every session's stock and index
file from `--years` back to today. Idempotent: sessions already in the store are skipped unless --redo.

    python scripts/nse_backfill.py --years 2
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from utils import nse_prices as npx  # noqa: E402
from utils.nse_data import Session  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    S = Session()
    start = pd.Timestamp.today().normalize() - pd.DateOffset(days=int(a.years * 365) + 10)
    have = set()
    if not a.redo and len(npx.raw("stocks")):
        have = set(pd.to_datetime(npx.raw("stocks")["date"].unique()))
    # fetch only the weekdays not in the store yet (resumes after an interruption)
    todo = [d for d in pd.bdate_range(start, pd.Timestamp.today().normalize()) if d not in have]
    print(f"[backfill] {len(todo)} weekdays to fetch from {start:%d %b %Y}", flush=True)
    got = dict(sessions=0, holidays=0, index_days=0)
    for i, day in enumerate(todo):
        r = npx.update(S, upto=day, start=day, sleep=0.2, log=lambda *x: None, finish=False)
        for k in got:
            got[k] += r[k]
        if (i + 1) % 25 == 0:
            npx.compact("stocks")
            npx.compact("indices")
            print(f"[backfill] {i + 1}/{len(todo)} · {day:%d %b %Y} · {got} · {time.time() - t0:.0f}s", flush=True)
    try:
        npx.fetch_symbol_changes(S)
    except Exception as e:
        print(f"[backfill] symbol changes: {e}")
    npx.compact("stocks")
    npx.compact("indices")
    npx._load.cache_clear()
    R = npx.raw("stocks")
    print(f"[backfill] done in {time.time() - t0:.0f}s: {got}; store {R['date'].min():%d %b %Y} -> {R['date'].max():%d %b %Y}, "
          f"{R['symbol'].nunique():,} symbols, {len(R):,} rows; "
          f"{sum(os.path.getsize(f) for f in glob.glob('data/nse/prices/**/*.parquet', recursive=True)) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
