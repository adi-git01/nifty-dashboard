"""
One-off repair of the daily signal logs
=======================================

1. BACKFILL the RS divergence log. It was only ever written by the dashboard,
   whose filesystem is ephemeral, so the committed file held 22 rows from
   2026-05-18 and nothing else. The EOD logger writes it now (run_daily_signals),
   but that cannot recover the past. The daily master snapshots can: every
   trading day since 2026-02-24 is in data/cache, and each carries the per-stock
   day change and 52-week distance the scanner uses. The one thing a snapshot
   lacks is the index's day return, so this fetches ^NSEI closes and pairs each
   snapshot with the index bar FOR THE SAME DATE -- a snapshot with no matching
   index bar is skipped, never paired with a neighbouring session.

   Backfilled rows get honest returns: a signal older than 21 days is CLOSED at
   the price on the last snapshot within 21 days of the signal, not at today's
   price, so return_since_signal means the same thing on every row.

2. REMOVE phantom rows. The engine runs on market holidays; Yahoo then returns
   the previous close, so the day's snapshot is a copy of the one before and
   every scanner re-fires on a move that already happened, logged under the
   holiday's date. Those rows are dropped from the shock, turnaround and RS
   logs. run_daily_signals.py now refuses to scan a duplicate snapshot, so this
   is needed once.

Run: python repair_signal_logs.py [--dry-run]
Needs network access to Yahoo for ^NSEI (the CI workflow has it).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.advanced_scanners import (
    RS_LOG_COLS, RS_LOG_FILE, SHOCK_LOG_FILE, TC_LOG_FILE,
    _append_to_log, find_rs_divergence, is_duplicate_snapshot,
)

CACHE_GLOB = "data/cache/market_master_*.parquet"
HOLD_DAYS = 21          # the logs' auto-close horizon, in calendar days


def load_snapshots():
    """{date: DataFrame} for every master snapshot, oldest first."""
    snaps = {}
    for f in sorted(glob.glob(CACHE_GLOB)):
        m = re.search(r"(\d{4})_(\d{2})_(\d{2})", os.path.basename(f))
        if m:
            snaps[pd.Timestamp("-".join(m.groups()))] = pd.read_parquet(f)
    return snaps


def duplicate_dates(snaps):
    """Dates whose snapshot is a copy of the previous one (market holidays)."""
    dates, dups = sorted(snaps), []
    for prev, cur in zip(dates, dates[1:]):
        if is_duplicate_snapshot(snaps[cur], snaps[prev]):
            dups.append(cur)
    return dups


def fetch_nifty(start):
    from utils.yf_safe import safe_history
    n = safe_history("^NSEI", start=start.strftime("%Y-%m-%d"))
    if n is None or n.empty:
        raise SystemExit("Could not fetch ^NSEI -- nothing to pair snapshots with. Aborting.")
    if n.index.tz is not None:
        n.index = n.index.tz_localize(None)
    n.index = n.index.normalize()
    return n[~n.index.duplicated(keep="last")]


def backfill_rs(snaps, dups, nifty, today):
    price = {d: s.set_index("ticker")["price"] for d, s in snaps.items() if d not in dups}
    usable = sorted(price)
    rows, skipped = [], []
    for d in usable:
        upto = nifty.loc[:d]
        if upto.empty or upto.index[-1] != d or len(upto) < 2:
            skipped.append(d)
            continue
        for r in find_rs_divergence(snaps[d], upto.iloc[-2:]):
            t = r["Ticker"]
            age = (today - d).days
            if age > HOLD_DAYS:
                horizon = [x for x in usable if d < x <= d + pd.Timedelta(days=HOLD_DAYS)]
                status = "CLOSED"
            else:
                horizon = [x for x in usable if x > d]
                status = "ACTIVE"
            cur = r["Price"]
            for x in reversed(horizon):                  # last snapshot that has the name
                v = price[x].get(t)
                if pd.notna(v) and v > 0:
                    cur = float(v)
                    break
            sp = float(r["Price"])
            rows.append({
                "signal_date": d.strftime("%Y-%m-%d"), "ticker": t, "name": r["Name"],
                "sector": r["Sector"], "signal_price": round(sp, 2),
                "current_price": round(cur, 2),
                "stock_ret_on_day": r["Stock_Ret"], "nifty_ret_on_day": r["Nifty_Ret"],
                "delta_rs": r["Delta_RS"], "dist_52w": r["Dist_52W"],
                "return_since_signal": round((cur / sp - 1) * 100, 2) if sp > 0 else 0.0,
                "days_held": age, "status": status,
            })
    return pd.DataFrame(rows, columns=RS_LOG_COLS), skipped


def drop_phantoms(path, dups, dry):
    if not os.path.exists(path):
        return 0
    df = pd.read_csv(path)
    bad = df["signal_date"].astype(str).isin({d.strftime("%Y-%m-%d") for d in dups})
    if bad.any() and not dry:
        df[~bad].to_csv(path, index=False)
    return int(bad.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    today = pd.Timestamp.now().normalize()

    snaps = load_snapshots()
    if not snaps:
        raise SystemExit("No snapshots found.")
    dups = duplicate_dates(snaps)
    first = min(snaps)
    print(f"[repair] {len(snaps)} snapshots {first.date()} -> {max(snaps).date()}")
    print(f"[repair] {len(dups)} duplicate (holiday) snapshots: "
          + ", ".join(str(d.date()) for d in dups))

    for path in (SHOCK_LOG_FILE, TC_LOG_FILE, RS_LOG_FILE):
        n = drop_phantoms(path, dups, args.dry_run)
        print(f"[repair] {path}: {n} phantom row(s) "
              f"{'would be ' if args.dry_run else ''}removed")

    nifty = fetch_nifty(first - pd.Timedelta(days=10))
    new, skipped = backfill_rs(snaps, dups, nifty, today)
    red = new["signal_date"].nunique() if len(new) else 0
    print(f"[repair] RS divergence: {len(new)} signal(s) on {red} red day(s); "
          f"{len(skipped)} snapshot(s) had no matching index bar and were skipped"
          + (f" ({', '.join(str(d.date()) for d in skipped[:6])}"
             f"{'...' if len(skipped) > 6 else ''})" if skipped else ""))

    before = len(pd.read_csv(RS_LOG_FILE)) if os.path.exists(RS_LOG_FILE) else 0
    if args.dry_run:
        print(f"[repair] --dry-run: RS log would grow from {before} rows (dedupe on date+ticker)")
        return
    after = _append_to_log(RS_LOG_FILE, new, ["signal_date", "ticker"], RS_LOG_COLS)
    print(f"[repair] RS log: {before} -> {len(after)} rows, "
          f"latest signal {after['signal_date'].max() if len(after) else 'none'}")


if __name__ == "__main__":
    main()
