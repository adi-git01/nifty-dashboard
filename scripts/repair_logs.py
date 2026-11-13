"""
Log hygiene: repair rows the pipeline wrote wrongly. Safe to run every day (each fix is a no-op
once applied); the EOD workflow runs it before the engine.

    python scripts/repair_logs.py [--dry-run]

1. Mis-dated sessions. A run on a market holiday whose previous run failed logs the last real
   session's signals under the holiday's date. Known case: Monday 9 Nov 2026 filed under Tuesday
   10 Nov (Diwali Balipratipada). Rows are re-dated in the earnings-shock, turnaround-catalyst,
   breakout and agreement logs.
2. Mood and breadth history: weekend rows (21 from Dec 2025 - Mar 2026, written when the
   dashboard itself saved snapshots), the holiday row, and any row that repeats the previous
   row's every metric.
3. Sub-industry rotation: one date format (older rows carry a time), the stale copy of Friday
   recorded for 9 Nov dropped, and the 10 Nov rows (Monday's data) re-dated.

Returns are NOT touched here: split / bonus adjustment happens where each log refreshes its
prices (utils/corp_actions.py).
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# holiday date under which a session was filed -> the session it really was
MISDATED = {"2026-11-10": "2026-11-09"}
DRY = "--dry-run" in sys.argv


def _save(df: pd.DataFrame, path: str) -> None:
    if not DRY:
        from utils.atomic_io import atomic_to_csv
        atomic_to_csv(df, path, index=False)


def redate(path: str, cols: list, key: list, shift_cols: tuple = (), drop_stale: bool = False) -> int:
    """
    Move rows dated on a holiday to the real session; `shift_cols` move back one business day.
    drop_stale: rows already filed under the real session came from the failed run re-scanning
    the session before it (9 Nov's run re-logged Friday's shocks) -- remove them first.
    """
    if not os.path.exists(path):
        return 0
    d = pd.read_csv(path)
    n = 0
    for bad, good in MISDATED.items():
        lead = cols[0]
        hit = d[lead].astype(str).str[:10] == bad
        if not hit.any():
            continue
        if drop_stale:
            stale = d[lead].astype(str).str[:10] == good
            n += int(stale.sum())
            d = d[~stale].reset_index(drop=True)
            hit = d[lead].astype(str).str[:10] == bad
        n += int(hit.sum())
        for c in shift_cols:                       # e.g. a breakout date counted back from the run date
            if c in d.columns:
                x = pd.to_datetime(d.loc[hit, c], errors="coerce") - pd.offsets.BDay(1)
                d.loc[hit, c] = x.dt.strftime("%Y-%m-%d")
        for c in cols:
            if c in d.columns:
                d.loc[d[c].astype(str).str[:10] == bad, c] = good
    if n:
        d = d.drop_duplicates(key, keep="first")
        _save(d, path)
    return n


def clean_history(path: str) -> dict:
    """Drop weekend rows, mis-dated holiday rows, and rows repeating the previous row exactly."""
    if not os.path.exists(path):
        return {}
    d = pd.read_csv(path).sort_values("date").reset_index(drop=True)
    dt = pd.to_datetime(d["date"], errors="coerce")
    weekend = dt.dt.weekday >= 5
    holiday = d["date"].isin(MISDATED) & d["date"].map(MISDATED).isin(set(d["date"]))
    vals = d.drop(columns=["date"])
    keep0 = d[~(weekend | holiday)].reset_index(drop=True)
    v = keep0.drop(columns=["date"])
    repeat = (v == v.shift(1)).all(axis=1)
    out = keep0[~repeat]
    stats = dict(weekend=int(weekend.sum()), holiday=int(holiday.sum()), repeat=int(repeat.sum()))
    if sum(stats.values()):
        _save(out, path)
    return stats


def fix_rotation(path: str = "data/sub_industry_rotation.csv") -> dict:
    if not os.path.exists(path):
        return {}
    d = pd.read_csv(path)
    raw = d["record_date"].astype(str)
    norm = pd.to_datetime(raw, format="mixed", errors="coerce").dt.strftime("%Y-%m-%d")
    stats = dict(reformatted=int((raw != norm).sum()), stale_dropped=0, redated=0)
    d["record_date"] = norm
    for bad, good in MISDATED.items():
        if not (d.record_date == bad).any():
            continue
        # rows recorded for the real session were a copy of the session before it (failed fetch)
        prior = d[d.record_date < good].record_date.max()
        g, p = d[d.record_date == good], d[d.record_date == prior]
        cols = [c for c in d.columns if c not in ("record_date",)]
        if len(g) and len(p) and g[cols].reset_index(drop=True).equals(p[cols].reset_index(drop=True)):
            stats["stale_dropped"] += len(g)
            d = d[d.record_date != good]
        if not (d.record_date == good).any():
            stats["redated"] += int((d.record_date == bad).sum())
            d.loc[d.record_date == bad, "record_date"] = good
        else:
            d = d[d.record_date != bad]
    if sum(stats.values()):
        _save(d, path)
    return stats


def main() -> int:
    print(f"[HYGIENE] {'dry run -- ' if DRY else ''}repairing logs")
    jobs = [
        ("data/earnings_shock_log.csv", ["signal_date"], ["ticker", "signal_date"], (), True),
        ("data/tc_log.csv", ["signal_date"], ["ticker", "signal_date"], (), True),
        ("data/breakout_log.csv", ["first_seen"], ["ticker", "kind", "breakout_date"], ("breakout_date",), False),
        ("data/consensus_log.csv", ["first_seen", "last_agree"], ["ticker", "first_seen"], (), False),
    ]
    for path, cols, key, shift, stale in jobs:
        try:
            n = redate(path, cols, key, shift, stale)
            print(f"   {os.path.basename(path)}: {n} row(s) re-dated or removed" if n else f"   {os.path.basename(path)}: ok")
        except Exception as e:
            print(f"   {os.path.basename(path)}: FAILED {e}")
    for path in ("data/market_breadth_history.csv", "data/market_mood_history.csv"):
        try:
            s = clean_history(path)
            print(f"   {os.path.basename(path)}: {s if sum(s.values()) else 'ok'}")
        except Exception as e:
            print(f"   {os.path.basename(path)}: FAILED {e}")
    try:
        s = fix_rotation()
        print(f"   sub_industry_rotation.csv: {s if sum(s.values()) else 'ok'}")
    except Exception as e:
        print(f"   sub_industry_rotation.csv: FAILED {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
