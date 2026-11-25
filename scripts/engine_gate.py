#!/usr/bin/env python3
"""
When the evening/overnight EOD engine may run, and recording that it did (.github/workflows/engine.yml).

Why: NSE's daily Bhavcopy and indices (primary source in utils/nse_prices.py) are published ~18:30 IST
and remain available all night and pre-market. Previously, strict 19:00-23:50 IST gating caused every
run to skip silently whenever GitHub Actions delayed cron jobs by 8-12 hours into 02:00-08:59 IST.

So this gate allows a scheduled run:
  - in the evening window (19:00-23:59 IST Mon-Fri) targeting today's session, OR
  - in overnight / pre-market catch-up (00:00-08:59 IST Mon-Sat) targeting the previous weekday session,
  - until that target session is captured (`bar_date >= target`, or MAX_TRIES attempts on holidays).
Market/pre-publish hours (09:00-18:59 IST) and weekends (Sat >= 09:00 IST, all Sunday) are skipped.
A manual run (`workflow_dispatch`) always goes through.

  python scripts/engine_gate.py check   -> writes skip=true|false to $GITHUB_OUTPUT (and prints why)
  python scripts/engine_gate.py mark    -> records this run: date, target session, snapshot bar_date, tries
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone

MARK = "data/engine_last_run.json"
IST = timezone(timedelta(hours=5, minutes=30))
EVENING_FROM = (19, 0)       # after NSE publishes its day files (~18:30 IST)
CATCHUP_UNTIL = (9, 0)       # overnight / pre-market catch-up ends at market open (09:00 IST)
MAX_TRIES = 2


def _mark() -> dict:
    try:
        with open(MARK, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _prev_weekday(d: date) -> date:
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def target_session(now: datetime) -> str | None:
    """Return YYYY-MM-DD target NSE session for `now`, or None if outside scheduled EOD/catch-up windows."""
    wd = now.weekday()
    hm = (now.hour, now.minute)
    if wd == 6 or (wd == 5 and hm >= CATCHUP_UNTIL):
        return None
    if hm < CATCHUP_UNTIL:
        return _prev_weekday(now.date()).strftime("%Y-%m-%d")
    if hm < EVENING_FROM:
        return None
    return now.strftime("%Y-%m-%d")


def decide(now: datetime, event: str, mark: dict) -> tuple[bool, str]:
    """(skip, reason)."""
    if event == "workflow_dispatch":
        return False, "manual run"
    wd = now.weekday()
    hm = (now.hour, now.minute)
    if wd == 6 or (wd == 5 and hm >= CATCHUP_UNTIL):
        return True, "weekend"
    if CATCHUP_UNTIL <= hm < EVENING_FROM:
        return True, f"{now:%H:%M} IST is during market/pre-publish hours (09:00-19:00 IST)"
    target = target_session(now)
    bar_date = str(mark.get("bar_date") or "")[:10]
    if bar_date >= target:
        return True, f"session {target} is already captured"
    mark_target = str(mark.get("target") or mark.get("date") or "")[:10]
    if mark_target == target and int(mark.get("tries") or 0) >= MAX_TRIES:
        return True, f"{MAX_TRIES} attempts for {target} did not capture it (market holiday?)"
    phase = "overnight catch-up" if hm < CATCHUP_UNTIL else "evening"
    return False, f"session {target} not captured yet ({phase})"


def snapshot_bar_date() -> str:
    files = sorted(glob.glob("data/cache/market_master_*.parquet"))
    if not files:
        return ""
    try:
        import pandas as pd
        d = pd.read_parquet(files[-1], columns=["bar_date"])["bar_date"].dropna().astype(str)
        return d.mode().iloc[0][:10] if len(d) else ""
    except Exception:
        return ""


def main(argv) -> int:
    cmd = argv[1] if len(argv) > 1 else "check"
    now = datetime.now(IST)
    if cmd == "check":
        skip, why = decide(now, os.environ.get("GITHUB_EVENT_NAME", ""), _mark())
        print(f"[gate] {now:%Y-%m-%d %H:%M} IST · last run {_mark()} · {'SKIP' if skip else 'RUN'}: {why}")
        out = os.environ.get("GITHUB_OUTPUT")
        if out:
            with open(out, "a", encoding="utf-8") as f:
                f.write(f"skip={'true' if skip else 'false'}\n")
        return 0
    if cmd == "mark":
        today = now.strftime("%Y-%m-%d")
        target = target_session(now) or today
        m = _mark()
        prev_target = str(m.get("target") or m.get("date") or "")[:10]
        tries = (int(m.get("tries") or 0) if prev_target == target else 0) + 1
        bar = snapshot_bar_date()
        payload = {"date": today, "target": target, "bar_date": bar, "tries": tries, "at": now.strftime("%H:%M")}
        with open(MARK, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(f"[gate] marked: date {today}, target {target}, snapshot holds {bar or '?'}, attempt {tries}"
              + ("" if bar >= target else "  ⚠️ snapshot is behind target session"))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
