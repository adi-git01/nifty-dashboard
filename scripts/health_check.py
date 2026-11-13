"""
End-of-run data health check for the EOD workflow.

Every engine step is continue-on-error, so a run that silently reused an old snapshot used to
finish green (9 Nov 2026: 3 of 1,174 stocks processed, Friday's data reused, nobody told).
This runs last, after the commit. It exits 1 -- turning the run red and firing the crash
notification -- when today's snapshot is missing or thin, and prints warnings for slower rot
(old fundamentals, old valuation run, missing SIMBA fields).

    python scripts/health_check.py            # exit 1 on a hard failure
    python scripts/health_check.py --no-fail  # report only
"""
import glob
import os
import re
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check(today: str | None = None) -> tuple[list, list]:
    fails, warns = [], []
    today = today or datetime.now().strftime("%Y_%m_%d")
    files = sorted(glob.glob("data/cache/market_master_*.parquet"))
    path = f"data/cache/market_master_{today}.parquet"
    if not os.path.exists(path):
        last = re.search(r"(\d{4}_\d{2}_\d{2})", files[-1]).group(1) if files else "none"
        fails.append(f"no snapshot for today ({today.replace('_', '-')}); the newest is {last.replace('_', '-')}. "
                     "Alerts, logs and the dashboard are running on old prices.")
        return fails, warns
    df = pd.read_parquet(path)
    try:
        universe = len(pd.read_csv("data/nifty1000_list.csv"))
    except Exception:
        universe = 0
    if universe and len(df) < 0.8 * universe:
        fails.append(f"today's snapshot has {len(df)} stocks, under 80% of the {universe}-stock universe")
    for col in ("currentPrice", "comp_rs", "fx_action", "trend_signal"):
        if col not in df.columns or df[col].isna().mean() > 0.2:
            fails.append(f"column {col} is missing or more than 20% blank in today's snapshot")
    if "simba_stage" not in df.columns or df["simba_stage"].notna().mean() < 0.5:
        warns.append("SIMBA chart fields are missing for most stocks")
    if len(files) >= 2 and files[-1] == path:
        prev = pd.read_parquet(files[-2]).set_index("ticker")["currentPrice"]
        cur = df.set_index("ticker")["currentPrice"]
        both = cur.index.intersection(prev.index)
        if len(both) and (cur[both] == prev[both]).mean() > 0.95:
            warns.append("today's prices equal the previous snapshot for over 95% of stocks "
                         "(a market holiday, or the fetch returned old bars)")
    try:
        from utils.positions import get_all_positions, resolve_ticker
        known = set(df["ticker"])
        lost = [p.get("ticker") for p in get_all_positions() if not resolve_ticker(p.get("ticker", ""), known)]
        if lost:
            warns.append(f"positions with no market data (stop-loss and target never checked): {', '.join(map(str, lost))}")
    except Exception:
        pass
    try:
        q = pd.read_csv("data/fundamentals_quarterly.csv.gz")
        q = q[q.field == "np"]
        last_q = pd.to_datetime(q.groupby("ticker").period.max())
        old = float((last_q < pd.Timestamp.today() - pd.Timedelta(days=200)).mean())
        if old > 0.3:
            warns.append(f"{old:.0%} of stocks have no quarterly result newer than 200 days -- the weekly "
                         "fundamentals refresh is not landing")
    except Exception as e:
        warns.append(f"quarterly fundamentals unreadable ({e})")
    try:
        v = pd.read_csv("data/valuation_snapshot.csv", usecols=["as_of"])
        age = (pd.Timestamp.today().normalize() - pd.to_datetime(v.as_of.dropna().iloc[0])).days
        if age > 5:
            warns.append(f"valuation snapshot is {age} days old")
    except Exception as e:
        warns.append(f"valuation snapshot unreadable ({e})")
    return fails, warns


def main() -> int:
    fails, warns = check()
    for w in warns:
        print(f"[HEALTH] warning: {w}")
    for f in fails:
        print(f"[HEALTH] FAIL: {f}")
    if not fails:
        print("[HEALTH] ok" + (f" ({len(warns)} warning(s))" if warns else ""))
        return 0
    try:
        from utils.telegram_notifier import is_telegram_configured, send_telegram_message
        if is_telegram_configured():
            send_telegram_message("🚨 DATA HEALTH CHECK FAILED\n\n" + "\n".join(f"• {x}" for x in fails)
                                  + ("\n\nWarnings:\n" + "\n".join(f"• {x}" for x in warns) if warns else ""))
    except Exception as e:
        print(f"[HEALTH] could not send Telegram: {e}")
    return 0 if "--no-fail" in sys.argv else 1


if __name__ == "__main__":
    raise SystemExit(main())
