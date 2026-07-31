"""
RS Alert Daemon
===============
Headless evaluator for the Composite-RS crossing alerts created in the
dashboard. Runs inside the EOD workflow so alerts fire whether or not the
dashboard is open, and pushes to Telegram.

Data source
-----------
Reads the newest data/cache/market_master_*.parquet, which the trading engine
writes earlier in the same workflow run. That file already carries comp_rs and
its RS5/RS21/RS63 components, so no Yahoo call is needed here — this step
cannot fail on a rate limit, and it evaluates exactly the numbers the dashboard
would show.

If the trading engine failed and the newest snapshot is stale, nothing bad
happens: every alert's last_value already equals that snapshot's reading, so no
crossing is detected and no alert fires. A stale run is silent, not wrong. A
snapshot older than MAX_SNAPSHOT_AGE_DAYS is skipped outright and reported.

State
-----
check_rs_alerts() advances each alert's last_value and the workflow commits
data/rs_alerts.json. That write is the de-duplication: a fired alert cannot
fire again until the metric genuinely crosses back and over.

Run: python rs_alert_daemon.py [--dry-run]
"""

import argparse
import glob
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.rs_alerts import check_rs_alerts, load_rs_alerts
from utils.telegram_notifier import is_telegram_configured, send_telegram_message

IST = timezone(timedelta(hours=5, minutes=30))
CACHE_GLOB = "data/cache/market_master_*.parquet"
# Overridable for testing and for the odd long market holiday; the default is
# what the scheduled run uses.
MAX_SNAPSHOT_AGE_DAYS = int(os.environ.get("RS_MAX_SNAPSHOT_AGE_DAYS", "5"))


def latest_snapshot():
    """(DataFrame, date, path) for the newest master snapshot, or (None, None, None)."""
    dated = []
    for path in glob.glob(CACHE_GLOB):
        m = re.search(r"(\d{4}_\d{2}_\d{2})", path)
        if m:
            dated.append((pd.Timestamp(m.group(1).replace("_", "-")), path))
    if not dated:
        return None, None, None
    date, path = max(dated)
    try:
        return pd.read_parquet(path), date, path
    except Exception as e:
        print(f"[RS] Failed to read {path}: {e}")
        return None, None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="evaluate and report without advancing or saving state")
    args = ap.parse_args()

    print(f"[RS] RS alert check at {datetime.now(tz=IST):%Y-%m-%d %H:%M:%S} IST")

    alerts = load_rs_alerts()
    if not alerts:
        print("[RS] No RS alerts configured — nothing to do.")
        return 0
    print(f"[RS] {len(alerts)} alert(s) configured")

    df, date, path = latest_snapshot()
    if df is None:
        print("[RS] No usable market snapshot found — skipping.")
        return 0

    age = (pd.Timestamp.now().normalize() - date).days
    print(f"[RS] Snapshot {os.path.basename(path)} ({len(df)} tickers, {age}d old)")
    if age > MAX_SNAPSHOT_AGE_DAYS:
        print(f"[RS] Snapshot older than {MAX_SNAPSHOT_AGE_DAYS}d — skipping rather "
              f"than evaluating alerts against stale prices.")
        return 0

    triggered = check_rs_alerts(df, persist=not args.dry_run)

    if not triggered:
        print("[RS] No crossings.")
        return 0

    print(f"[RS] {len(triggered)} crossing(s):")
    for t in triggered:
        print(f"     {t['alert_message']}")
        if t.get("notes"):
            print(f"       note: {t['notes']}")

    if args.dry_run:
        print("[RS] --dry-run: state not saved, no notification sent.")
        return 0

    lines = [t["alert_message"] for t in triggered]
    notes = [f"  {t['ticker'].replace('.NS','')}: {t['notes']}"
             for t in triggered if t.get("notes")]
    message = ("📈 RS ALERT\n\n" + "\n".join(lines)
               + (("\n\nNotes:\n" + "\n".join(notes)) if notes else "")
               + f"\n\nSnapshot: {date:%d %b %Y}")

    if is_telegram_configured():
        ok, msg = send_telegram_message(message)
        print(f"[RS] Telegram: {'sent' if ok else 'failed — ' + str(msg)}")
    else:
        print("[RS] Telegram not configured — crossings logged only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
