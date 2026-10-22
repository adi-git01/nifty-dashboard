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

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rs_alert_daemon import MAX_SNAPSHOT_AGE_DAYS, latest_snapshot  # noqa: E402
from utils.industry_alerts import check_industry_alerts, load_industry_alerts, storage_status  # noqa: E402
from utils.telegram_notifier import is_telegram_configured, send_telegram_message  # noqa: E402


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
    age = (pd.Timestamp.now().normalize() - date).days
    print(f"[IND] {len(alerts)} alert(s); snapshot {os.path.basename(path)} ({age}d old)")
    if age > MAX_SNAPSHOT_AGE_DAYS:
        print(f"[IND] Snapshot older than {MAX_SNAPSHOT_AGE_DAYS}d — skipping.")
        return 0

    msgs = check_industry_alerts(df, f"{date:%Y-%m-%d}", persist=not args.dry_run)
    if not args.dry_run:
        print(f"[IND] store: {storage_status()}")
    for a in load_industry_alerts() if not args.dry_run else alerts:
        r = a.get("last_reading") or {}
        print(f"     {a.get('name')}: {a.get('last_status')} (pending {a.get('pending_status')}) "
              f"score {r.get('score')}")
    if not msgs:
        print("[IND] No status changes or score crossings.")
        return 0
    print("\n".join(msgs))
    if args.dry_run:
        return 0
    text = "🏭 INDUSTRY ALERT\n\n" + "\n\n".join(msgs) + f"\n\nSnapshot: {date:%d %b %Y}"
    if is_telegram_configured():
        ok, info = send_telegram_message(text)
        print(f"[IND] Telegram: {'sent' if ok else 'failed — ' + str(info)}")
    else:
        print("[IND] Telegram not configured — logged only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
