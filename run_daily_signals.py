"""
Daily Signal Logger
====================
Runs after trading_engine.py in the GitHub Actions pipeline.
Uses the parquet cache (already written today) to fire:
  1. Earnings Shock signals  → data/earnings_shock_log.csv
  2. Turnaround Catalyst signals → data/tc_log.csv
  3. RS Divergence signals   → data/rs_divergence_log.csv

No per-stock OHLCV download — works from cached market snapshots. RS divergence
needs one ^NSEI call for the index's return on the snapshot's bar date, because
the snapshot stores each stock's day change but not the index's.

Before this logger wrote it, the RS divergence log was only ever appended by the
dashboard, whose filesystem is ephemeral and never committed — so the file in
git was seeded once on 2026-05-19 and never changed.
"""

import os
import re
import sys
import glob
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.advanced_scanners import (
    find_turnaround_catalysts, save_turnaround_catalyst_signals,
    find_earnings_shocks_from_cache, save_earnings_shock_signals,
    find_rs_divergence, save_rs_divergence_signals,
    TC_LOG_FILE, TC_LOG_COLS,
    SHOCK_LOG_FILE, SHOCK_LOG_COLS,
    RS_LOG_FILE, RS_LOG_COLS,
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


def snapshot_date():
    """Bar date of the newest master snapshot, parsed from its filename."""
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "market_master_*.parquet")))
    if not files:
        return None
    m = re.search(r"(\d{4})_(\d{2})_(\d{2})", os.path.basename(files[-1]))
    return pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}") if m else None


def nifty_bars_for(bar_date):
    """
    Two-row Nifty frame ending ON bar_date, or (None, reason).

    find_rs_divergence compares each stock's day change with the index's day
    change, so both must be for the same session. If Yahoo has not published
    the index bar for the snapshot's date, or has published a later one only,
    returning the latest two bars would pair a stock's move on one day with the
    index's move on another. Refuse instead: a skipped day is silent, a
    mismatched day writes false signals into a permanent log.
    """
    from utils.yf_safe import safe_history
    try:
        nifty = safe_history("^NSEI", period="1mo")
    except Exception as e:
        return None, f"index fetch raised {e}"
    if nifty is None or nifty.empty:
        return None, "index fetch returned nothing"
    if nifty.index.tz is not None:
        nifty.index = nifty.index.tz_localize(None)
    nifty.index = nifty.index.normalize()
    nifty = nifty[~nifty.index.duplicated(keep="last")]
    upto = nifty.loc[:bar_date]
    if upto.empty or upto.index[-1] != bar_date:
        last = upto.index[-1].date() if not upto.empty else "none"
        return None, f"no index bar for {bar_date.date()} (latest at or before: {last})"
    if len(upto) < 2:
        return None, "fewer than two index bars"
    return upto.iloc[-2:], None


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

    # ── 3. RS Divergence (green in a sea of red) ──────────────────────────
    print("\n[RS] Running RS Divergence scanner...")
    rs_list = []
    bar_date = snapshot_date()
    nifty2, why = (nifty_bars_for(bar_date) if bar_date is not None
                   else (None, "no snapshot date"))
    if nifty2 is None:
        print(f"  Skipped: {why}")
    else:
        rs_list = find_rs_divergence(today_df, nifty2)
        save_rs_divergence_signals(rs_list, signal_date=bar_date)
        n_ret = (nifty2["Close"].iloc[-1] / nifty2["Close"].iloc[-2] - 1) * 100
        print(f"  Nifty {bar_date.date()}: {n_ret:+.2f}% "
              f"({'red day, scanner armed' if n_ret <= -0.3 else 'not a red day, nothing to log'})")
    # Refresh open rows even on skipped days so returns keep moving.
    refresh_signal_log_prices(RS_LOG_FILE, RS_LOG_COLS, price_map)
    print(f"  Found {len(rs_list)} RS Divergence signals.")

    # ── 4. Telegram Summary ───────────────────────────────────────────────
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

    lines.append("")
    if rs_list:
        lines.append(f"<b>🟢 RS Divergence ({len(rs_list)})</b>")
        for r in rs_list[:5]:
            lines.append(
                f"  {r['Ticker'].replace('.NS','')} {r['Stock_Ret']:+.2f}% "
                f"vs Nifty {r['Nifty_Ret']:+.2f}%  ΔRS {r['Delta_RS']:+.2f}"
            )
        if len(rs_list) > 5:
            lines.append(f"  ...+{len(rs_list)-5} more")
    else:
        lines.append("🟢 No RS Divergence today.")

    msg = "\n".join(lines)
    print(f"\nTelegram:\n{msg}\n")
    send_telegram(msg)

    print(f"[{datetime.now():%H:%M:%S}] Done.")


if __name__ == "__main__":
    main()
