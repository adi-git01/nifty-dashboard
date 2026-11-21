#!/usr/bin/env python3
"""
When the evening engine may run, and recording that it did (.github/workflows/engine.yml).

Why: Yahoo serves a session's daily bar on the same evening, but a run that starts after midnight IST
gets data only up to the session before (seen 16-18 Nov 2026: every snapshot one session behind).
GitHub starts scheduled runs up to ~9 hours late, so a single 3:45 pm slot often began after midnight.

So the workflow fires often (morning slots that land in the evening when delayed, evening slots that
land on time), and this gate lets a run through only:
  - between 19:00 and 23:50 IST on a weekday (after NSE publishes its day files; before Yahoo's midnight gap, the fallback), and
  - until today's session is captured (or two attempts were made, e.g. on a market holiday).
A manual run always goes through.

  python scripts/engine_gate.py check   -> writes skip=true|false to $GITHUB_OUTPUT (and prints why)
  python scripts/engine_gate.py mark    -> records this run: today's date, the session the snapshot holds, tries
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, timedelta, timezone

MARK = "data/engine_last_run.json"
IST = timezone(timedelta(hours=5, minutes=30))
OPEN_FROM, OPEN_TO = (19, 0), (23, 50)       # after NSE publishes its day files (~18-19h)
MAX_TRIES = 2


def _mark() -> dict:
    try:
        return json.load(open(MARK))
    except Exception:
        return {}


def decide(now: datetime, event: str, mark: dict) -> tuple[bool, str]:
    """(skip, reason)."""
    today = now.strftime("%Y-%m-%d")
    if event == "workflow_dispatch":
        return False, "manual run"
    if now.weekday() >= 5:
        return True, "weekend"
    hm = (now.hour, now.minute)
    if hm < OPEN_FROM:
        return True, f"{now:%H:%M} IST is before 19:00, when NSE's day files are out"
    if hm >= OPEN_TO:
        return True, f"{now:%H:%M} IST is after 23:50: Yahoo would not have today's bar yet"
    if mark.get("date") == today and mark.get("bar_date") == today:
        return True, "today's session is already captured"
    if mark.get("date") == today and int(mark.get("tries") or 0) >= MAX_TRIES:
        return True, f"{MAX_TRIES} attempts today did not capture today's session (market holiday?)"
    return False, "today's session not captured yet"


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
            with open(out, "a") as f:
                f.write(f"skip={'true' if skip else 'false'}\n")
        return 0
    if cmd == "mark":
        today, m = now.strftime("%Y-%m-%d"), _mark()
        tries = (int(m.get("tries") or 0) if m.get("date") == today else 0) + 1
        bar = snapshot_bar_date()
        json.dump({"date": today, "bar_date": bar, "tries": tries, "at": now.strftime("%H:%M")}, open(MARK, "w"))
        print(f"[gate] marked: date {today}, snapshot holds {bar or '?'}, attempt {tries}"
              + ("" if bar == today else "  ⚠️ snapshot is not today's session"))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
