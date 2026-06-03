"""
Daily Signal Logger
====================
Runs after trading_engine.py in the GitHub Actions pipeline.
Uses the parquet cache (already written today) to fire:
  1. Earnings Shock signals  → data/earnings_shock_log.csv
  2. Turnaround Catalyst signals → data/tc_log.csv

No OHLCV download needed — works entirely from cached market snapshots.
"""

import os
import sys
import glob
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.advanced_scanners import (
    find_turnaround_catalysts, save_turnaround_catalyst_signals,
    find_earnings_shocks_from_cache, save_earnings_shock_signals,
    TC_LOG_FILE, TC_LOG_COLS,
    SHOCK_LOG_FILE, SHOCK_LOG_COLS,
    refresh_signal_log_prices,
)

CACHE_DIR = "data/cache"


def load_latest_parquets():
    """Return (today_df, prev_scores_map, prev_rs21_map)."""
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "market_master_*.parquet")))
    if not files:
        print("❌ No parquet cache files found. Run trading_engine.py first.")
        return None, {}, {}

    today_df = pd.read_parquet(files[-1])
    print(f"  Today:     {os.path.basename(files[-1])}  ({len(today_df)} stocks)")

    prev_scores_map = {}
    prev_rs21_map   = {}
    if len(files) >= 2:
        prev_df = pd.read_parquet(files[-2])
        print(f"  Yesterday: {os.path.basename(files[-2])}  ({len(prev_df)} stocks)")
        prev_scores_map = prev_df.set_index("ticker")["trend_score"].dropna().to_dict()
        rs21_col = "rs_1m" if "rs_1m" in prev_df.columns else "return_1m"
        if rs21_col in prev_df.columns:
            prev_rs21_map = prev_df.set_index("ticker")[rs21_col].dropna().to_dict()

    # Normalise return_1m column name in today_df
    if "rs_1m" in today_df.columns and "return_1m" not in today_df.columns:
        today_df["return_1m"] = today_df["rs_1m"]

    return today_df, prev_scores_map, prev_rs21_map


def send_telegram(msg):
    import requests
    token   = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print(f"  Telegram error: {e}")


def main():
    print(f"\n[{datetime.now():%H:%M:%S}] Daily Signal Logger")
    print("=" * 50)

    today_df, prev_scores_map, prev_rs21_map = load_latest_parquets()
    if today_df is None:
        sys.exit(1)

    price_map = today_df.set_index("ticker")["price"].dropna().to_dict() if "price" in today_df.columns else {}

    # ── 1. Turnaround Catalyst ─────────────────────────────────────────────
    print("\n[TC] Running Turnaround Catalyst scanner...")
    tc_list = find_turnaround_catalysts(today_df, prev_scores_map, prev_rs21_map)
    save_turnaround_catalyst_signals(tc_list)
    refresh_signal_log_prices(TC_LOG_FILE, TC_LOG_COLS, price_map)
    print(f"  Found {len(tc_list)} TC signals today.")

    # ── 2. Earnings Shock (cache-based) ───────────────────────────────────
    print("\n[ES] Running Earnings Shock scanner (cache-based)...")
    shock_list = find_earnings_shocks_from_cache(today_df)
    save_earnings_shock_signals(shock_list)
    refresh_signal_log_prices(SHOCK_LOG_FILE, SHOCK_LOG_COLS, price_map)
    print(f"  Found {len(shock_list)} Earnings Shock signals today.")

    # ── 3. Telegram Summary ───────────────────────────────────────────────
    date_str = datetime.now().strftime("%d %b %Y")
    lines = [f"<b>📡 Daily Signal Logger — {date_str}</b>\n"]

    if tc_list:
        lines.append(f"<b>🔄 Turnaround Catalysts ({len(tc_list)})</b>")
        for r in tc_list[:8]:
            vel_flag = " 🚀" if r["RS21_Vel"] >= 5 else ""
            lines.append(
                f"  {r['Ticker'].replace('.NS','')} [{r['Pattern']}] "
                f"↑{r['Jump%']}% Vol:{r['Vol_Score']:.0f} "
                f"TS:{r['TS_Pre']}→{r['TS_Now']}(+{r['TS_Gain']}){vel_flag}"
            )
        if len(tc_list) > 8:
            lines.append(f"  ...+{len(tc_list)-8} more")
    else:
        lines.append("🔄 No Turnaround Catalysts today.")

    lines.append("")

    if shock_list:
        lines.append(f"<b>⚡ Earnings Shocks ({len(shock_list)})</b>")
        for r in shock_list[:5]:
            lines.append(
                f"  {r['Ticker'].replace('.NS','')} ↑{r['Jump_Pct']}% "
                f"Vol:{r['Vol_Mult']:.1f}x  {r['PEAD_Action']}"
            )
        if len(shock_list) > 5:
            lines.append(f"  ...+{len(shock_list)-5} more")
    else:
        lines.append("⚡ No Earnings Shocks today.")

    msg = "\n".join(lines)
    print(f"\nTelegram:\n{msg}\n")
    send_telegram(msg)

    print(f"[{datetime.now():%H:%M:%S}] Done.")


if __name__ == "__main__":
    main()
