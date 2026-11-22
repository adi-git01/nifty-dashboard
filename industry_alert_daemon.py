"""
Industry Alert Daemon
=====================
Evaluates the industry / basket status alerts (utils/industry_alerts.py) in the
EOD workflow and pushes them to Telegram, so they fire with the dashboard closed.

Reads the newest data/cache/market_master_*.parquet written by the trading
engine earlier in the same run (no Yahoo call). A snapshot older than
MAX_SNAPSHOT_AGE_DAYS is skipped, and re-running on the same snapshot date
neither confirms a status change nor re-fires a score crossing.

Run: python industry_alert_daemon.py [--dry-run]
"""
import argparse
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rs_alert_daemon import IST, MAX_SNAPSHOT_AGE_DAYS, latest_snapshot  # noqa: E402
from utils.industry_alerts import check_industry_alerts, load_industry_alerts, storage_status  # noqa: E402
from utils.telegram_notifier import is_telegram_configured, queue_commit, send_telegram_message  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="evaluate without saving or sending")
    args = ap.parse_args()

    alerts = load_industry_alerts()
    if not alerts:
        print("[IND] No industry alerts configured — nothing to do.")
        return 0
    df, date, path = latest_snapshot()
    if df is None:
        print("[IND] No usable market snapshot — skipping.")
        return 0
    age = (pd.Timestamp(datetime.now(tz=IST).date()) - date).days
    print(f"[IND] {len(alerts)} alert(s); snapshot {os.path.basename(path)} ({age}d old)")
    if age > MAX_SNAPSHOT_AGE_DAYS:
        print(f"[IND] Snapshot older than {MAX_SNAPSHOT_AGE_DAYS}d — skipping.")
        return 0

    # the session the prices are from, not the file's name (a holiday re-save must not count as a new day)
    sess = f"{date:%Y-%m-%d}"
    if "bar_date" in df.columns and df["bar_date"].notna().any():
        sess = str(df["bar_date"].dropna().astype(str).mode().iloc[0])[:10]
    msgs = check_industry_alerts(df, sess, persist=False)       # saved only after the message is out (below)
    for a in alerts:
        r = a.get("last_reading") or {}
        print(f"     {a.get('name')}: {a.get('last_status')} (pending {a.get('pending_status')}) "
              f"score {r.get('score')}")
    if not msgs:
        print("[IND] No status changes or score crossings.")
        if not args.dry_run:
            check_industry_alerts(df, sess, persist=True)       # still advance pending-status counters
        return 0
    print("\n".join(msgs))
    if args.dry_run:
        return 0
    text = "🏭 INDUSTRY ALERT\n\n" + "\n\n".join(msgs) + f"\n\nSnapshot: {date:%d %b %Y}"
    ok = True
    if is_telegram_configured():
        ok, info = send_telegram_message(text)
        print(f"[IND] Telegram: {'sent' if ok else 'failed — ' + str(info)}")
    else:
        print("[IND] Telegram not configured — logged only.")
    if ok:
        if bool(os.environ.get("TELEGRAM_BATCH", "").strip()) and path and queue_commit({"kind": "industry_persist", "snapshot": path, "session": sess}):
            print("[IND] persistence queued for daily brief delivery")
        else:
            check_industry_alerts(df, sess, persist=True)
            print(f"[IND] store: {storage_status()}")
    else:
        print("[IND] state not saved, so these changes are sent again on the next run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
