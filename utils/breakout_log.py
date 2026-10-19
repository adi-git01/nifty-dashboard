"""
Breakout / squeeze log
======================
Every fresh 52-week / all-time-high breakout and every Bollinger-squeeze release the daily engine sees,
one row per (ticker, kind, breakout date), with what was knowable that day (6-month momentum rank, Gate 1,
size tier, industry score, volume, loss-maker / absorption flags) and the return since.

Written by run_daily_signals.py after each EOD engine run (committed with data/*.csv), seeded from the
cached snapshots by `python -m utils.breakout_log --backfill`. Breakout fields exist in the snapshots
from 30 Sep 2026; squeeze releases from the first engine run that computes sq_days.

The momentum-entry rule (top-10% 6-month return + Gate 1 + a fresh breakout that held, or a squeeze
release) is the one that passed in entry_exit_study.py S5; the log keeps every breakout, momentum or
not, so the two can be compared as the log grows.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

LOG_FILE = "data/breakout_log.csv"
COLS = ["breakout_date", "first_seen", "ticker", "name", "kind", "momentum", "mom_rank", "gate1", "size_tier",
        "ind_score", "vol_x", "anchor", "held", "loss_maker", "absorption", "signal_price", "current_price",
        "return_pct", "last_seen", "sessions_since"]
CACHE_GLOB = "data/cache/market_master_*.parquet"


def _snap_date(path: str) -> pd.Timestamp:
    m = re.search(r"(\d{4})_(\d{2})_(\d{2})", os.path.basename(path))
    return pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")


def load_log() -> pd.DataFrame:
    if os.path.exists(LOG_FILE):
        try:
            d = pd.read_csv(LOG_FILE, parse_dates=["breakout_date", "first_seen", "last_seen"])
            return d.reindex(columns=COLS)
        except Exception:
            pass
    return pd.DataFrame(columns=COLS)


def events_from_snapshot(df: pd.DataFrame, snap: pd.Timestamp) -> pd.DataFrame:
    """One row per fresh breakout / squeeze release visible in this snapshot."""
    from utils.breakout_tags import add_breakout_tags, add_momentum_tags
    t = add_momentum_tags(add_breakout_tags(df))
    price = pd.to_numeric(t.get("price", t.get("currentPrice")), errors="coerce")
    gate1 = t["fx_gate1"].fillna(False).astype(bool) if "fx_gate1" in t else pd.Series(False, index=t.index)
    rows = []
    bdays = lambda n: (snap - pd.offsets.BDay(int(n))).normalize()
    for i, r in t.iterrows():
        base = dict(first_seen=snap, ticker=r["ticker"], name=r.get("name", ""), momentum=bool(r.mom_rank >= 90 and gate1[i]),
                    mom_rank=round(float(r.mom_rank), 1) if pd.notna(r.mom_rank) else np.nan, gate1=bool(gate1[i]),
                    size_tier=r.size_tier, ind_score=r.get("ind_score"), loss_maker=bool(r.loss_maker),
                    absorption=bool(r.absorption), signal_price=price[i], current_price=price[i], return_pct=0.0,
                    last_seen=snap, sessions_since=0)
        if r.get("bo_type", "") and pd.notna(r.get("bo_days")):
            rows.append({**base, "breakout_date": bdays(r.bo_days), "kind": f"{r.bo_type} breakout",
                         "vol_x": r.get("bo_volx"), "anchor": bool(r.get("bo_anchor", False)),
                         "held": bool(r.get("bo_held", True))})
        if pd.notna(r.get("sq_days", np.nan)):
            rows.append({**base, "breakout_date": bdays(r.sq_days), "kind": "squeeze release",
                         "vol_x": np.nan, "anchor": False, "held": True})
    return pd.DataFrame(rows, columns=COLS)


def update_log(df: pd.DataFrame, snap: pd.Timestamp, save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add new events from this snapshot, refresh prices/returns of logged ones. Returns (log, new rows)."""
    log = load_log()
    ev = events_from_snapshot(df, snap)
    key = ["ticker", "kind", "breakout_date"]
    # a breakout seen again on later days (bo_days counts up) keeps its first sighting; ±1 business-day
    # drift between snapshots is absorbed by matching on ticker + kind within 3 days
    new = []
    for _, e in ev.iterrows():
        same = log[(log.ticker == e.ticker) & (log.kind == e.kind) &
                   ((pd.to_datetime(log.breakout_date) - e.breakout_date).abs() <= pd.Timedelta(days=3))]
        if len(same):
            j = same.index[-1]
            log.loc[j, "held"] = bool(log.loc[j, "held"]) and bool(e.held)
        else:
            new.append(e)
    new = pd.DataFrame(new, columns=COLS)
    log = pd.concat([log, new], ignore_index=True) if len(new) else log
    # refresh current price / return for everything still in the universe
    px = pd.to_numeric(df.set_index("ticker").get("price", df.set_index("ticker").get("currentPrice")), errors="coerce")
    cur = log.ticker.map(px)
    ok = cur.notna()
    log.loc[ok, "current_price"] = cur[ok]
    log.loc[ok, "return_pct"] = (cur[ok] / pd.to_numeric(log.loc[ok, "signal_price"]) - 1) * 100
    log.loc[ok, "last_seen"] = snap
    log["sessions_since"] = [len(pd.bdate_range(b, snap)) - 1 for b in pd.to_datetime(log.breakout_date)]
    log = log.sort_values(["breakout_date", "momentum", "ticker"], ascending=[False, False, True]).drop_duplicates(key)
    if save:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        log.to_csv(LOG_FILE, index=False)
    return log, new


def backfill() -> pd.DataFrame:
    """Rebuild the log from every cached snapshot that carries breakout fields, oldest first."""
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    log = load_log()
    for f in sorted(glob.glob(CACHE_GLOB)):
        df = pd.read_parquet(f)
        if "hi52_bo_days" not in df.columns:
            continue
        log, new = update_log(df, _snap_date(f))
        print(f"  {os.path.basename(f)}: +{len(new)} (log {len(log)})")
    return log


def this_week(log: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Breakouts / squeezes dated Monday of the current week or later (all of them, no size or count limit)."""
    today = (today or pd.Timestamp.today()).normalize()
    monday = today - pd.Timedelta(days=today.weekday())   # on a weekend: the week that just ended
    d = pd.to_datetime(log.breakout_date)
    return log[(d >= monday) & (d < monday + pd.Timedelta(days=7))]


if __name__ == "__main__":
    import sys
    if "--backfill" in sys.argv:
        out = backfill()
        print(f"breakout log: {len(out)} rows, {out.ticker.nunique()} stocks -> {LOG_FILE}")
