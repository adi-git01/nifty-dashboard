"""
Catalyst Detection Validation
==============================
Finds RAIN-like events across Nifty 500 (last 6 months):
  - Stock was in downtrend (price < MA50, dist_52w < -25%)
  - Massive volume day: price change > 5% AND volume > 3x 20D avg
  - Checks: would Earnings Shock gates + RS21 Velocity have caught it?
  - Shows 5D / 10D / 21D forward returns from the catalyst day

Run:  python validate_catalyst_detection.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
LOOKBACK_MONTHS  = 6        # how far back to scan
MIN_PRICE_JUMP   = 5.0      # % price move on catalyst day
VOL_MULT_THRESH  = 3.0      # volume vs 20D avg
DIST_52W_THRESH  = -25.0    # stock must be at least 25% below 52W high (was in pain)
MA50_GATE        = True     # require price < MA50 on day before catalyst
RS21_VEL_THRESH  = 5.0      # RS21 delta over 5D that constitutes "RS velocity spike"
TOP_N_RESULTS    = 8        # show top N catalyst events by 10D return

# ── Load tickers ─────────────────────────────────────────────────────────────
df_list = pd.read_csv("data/nifty500_list.csv")
# Add .NS suffix
tickers = [t.strip() + ".NS" for t in df_list["Ticker"].tolist() if str(t).strip()]
name_map = dict(zip(df_list["Ticker"].str.strip() + ".NS", df_list["Company Name"]))
sector_map = dict(zip(df_list["Ticker"].str.strip() + ".NS", df_list["Sector"]))

print(f"[{datetime.now():%H:%M:%S}] Downloading {len(tickers)} Nifty 500 stocks ({LOOKBACK_MONTHS+2}mo)...")

# Download in one batch — faster than per-ticker
raw = yf.download(
    tickers,
    period=f"{LOOKBACK_MONTHS + 2}mo",   # extra 2mo for MA50/vol warm-up
    group_by="ticker",
    threads=False,
    progress=True,
    auto_adjust=True,
)

print(f"[{datetime.now():%H:%M:%S}] Download complete. Scanning for catalyst events...")

# ── Download Nifty for RS calculation ─────────────────────────────────────
nifty = yf.Ticker("^NSEI").history(period=f"{LOOKBACK_MONTHS + 2}mo")[["Close"]]
nifty.index = nifty.index.tz_localize(None)

cutoff_start = datetime.now() - timedelta(days=LOOKBACK_MONTHS * 30)

events = []

for ticker in tickers:
    try:
        if ticker not in raw.columns.get_level_values(0):
            continue
        hist = raw[ticker].dropna(how="all")
        if len(hist) < 65:
            continue

        # Align Nifty to this ticker's dates
        nif_aligned = nifty["Close"].reindex(hist.index, method="ffill")

        # Precompute rolling series
        ma50    = hist["Close"].rolling(50).mean()
        hi52w   = hist["Close"].rolling(252).max()
        vol20   = hist["Volume"].rolling(20).mean().shift(1)  # shift to avoid look-ahead

        # RS21: 21-day relative return vs Nifty
        stock_ret21 = hist["Close"].pct_change(21) * 100
        nifty_ret21 = nif_aligned.pct_change(21) * 100
        rs21        = stock_ret21 - nifty_ret21

        # RS21 velocity: change in RS21 over last 5 days
        rs21_vel5 = rs21 - rs21.shift(5)

        # Scan within backtest window
        scan_idx = hist.index[hist.index >= pd.Timestamp(cutoff_start)]

        for date in scan_idx:
            i = hist.index.get_loc(date)
            if i < 55:   # need warm-up
                continue

            price_today = float(hist["Close"].iloc[i])
            price_prev  = float(hist["Close"].iloc[i - 1])
            vol_today   = float(hist["Volume"].iloc[i])
            vol_avg     = float(vol20.iloc[i]) if not pd.isna(vol20.iloc[i]) else 0

            if vol_avg == 0:
                continue

            # ── Gate 1: Price jump ─────────────────────────────────────────
            pct_chg = (price_today / price_prev - 1) * 100
            if pct_chg < MIN_PRICE_JUMP:
                continue

            # ── Gate 2: Volume surge ───────────────────────────────────────
            vol_mult = vol_today / vol_avg
            if vol_mult < VOL_MULT_THRESH:
                continue

            # ── Gate 3: Was in downtrend (dist from 52W high) ─────────────
            hi = float(hi52w.iloc[i - 1]) if not pd.isna(hi52w.iloc[i - 1]) else 0
            if hi == 0:
                continue
            dist_52w = (price_prev / hi - 1) * 100
            if dist_52w > DIST_52W_THRESH:   # already near highs = not a reversal
                continue

            # ── Gate 4 (optional): Price was below MA50 ────────────────────
            ma50_val = float(ma50.iloc[i - 1]) if not pd.isna(ma50.iloc[i - 1]) else 0
            was_below_ma50 = (price_prev < ma50_val) if ma50_val > 0 else None

            if MA50_GATE and not was_below_ma50:
                continue

            # ── RS21 Velocity on catalyst day ─────────────────────────────
            vel = float(rs21_vel5.iloc[i]) if not pd.isna(rs21_vel5.iloc[i]) else 0
            rs21_val = float(rs21.iloc[i]) if not pd.isna(rs21.iloc[i]) else 0

            # ── Forward returns ────────────────────────────────────────────
            future = hist.iloc[i:]
            ret_5d = ret_10d = ret_21d = None
            if len(future) > 5:
                ret_5d  = (float(future["Close"].iloc[5])  / price_today - 1) * 100
            if len(future) > 10:
                ret_10d = (float(future["Close"].iloc[10]) / price_today - 1) * 100
            if len(future) > 21:
                ret_21d = (float(future["Close"].iloc[21]) / price_today - 1) * 100

            # ── Would our new gates have fired? ───────────────────────────
            earnings_shock_gate = (pct_chg >= 4.0) and (vol_mult >= 2.5)   # current thresholds
            rs21_vel_gate       = (vel >= RS21_VEL_THRESH)

            events.append({
                "Date":            date.strftime("%Y-%m-%d"),
                "Ticker":          ticker,
                "Name":            name_map.get(ticker, ticker),
                "Sector":          sector_map.get(ticker, ""),
                "Price":           round(price_today, 1),
                "Jump%":           round(pct_chg, 1),
                "Vol_Mult":        round(vol_mult, 1),
                "Dist_52W%":       round(dist_52w, 1),
                "Below_MA50":      was_below_ma50,
                "RS21":            round(rs21_val, 1),
                "RS21_Vel5":       round(vel, 1),
                "Shock_Gate":      "✅" if earnings_shock_gate else "❌",
                "RS21Vel_Gate":    "✅" if rs21_vel_gate else "❌",
                "Ret_5D%":         round(ret_5d, 1)  if ret_5d  is not None else None,
                "Ret_10D%":        round(ret_10d, 1) if ret_10d is not None else None,
                "Ret_21D%":        round(ret_21d, 1) if ret_21d is not None else None,
            })

    except Exception as e:
        pass

print(f"\n[{datetime.now():%H:%M:%S}] Found {len(events)} RAIN-like catalyst events.\n")

if not events:
    print("No events found — try loosening thresholds.")
else:
    df = pd.DataFrame(events)

    # Sort by 10D return (descending) for top performers
    df_sorted = df.dropna(subset=["Ret_10D%"]).sort_values("Ret_10D%", ascending=False)

    print("=" * 100)
    print(f"TOP {TOP_N_RESULTS} CATALYST EVENTS BY 10D FORWARD RETURN")
    print("(Stock in downtrend >25% off 52W high, below MA50, then big gap+volume day)")
    print("=" * 100)

    cols = ["Date", "Ticker", "Name", "Sector", "Price", "Jump%", "Vol_Mult",
            "Dist_52W%", "RS21_Vel5", "Shock_Gate", "RS21Vel_Gate",
            "Ret_5D%", "Ret_10D%", "Ret_21D%"]
    print(df_sorted[cols].head(TOP_N_RESULTS).to_string(index=False))

    print("\n" + "=" * 60)
    print("SUMMARY STATS")
    print("=" * 60)
    n = len(df_sorted)
    pos_10d = (df_sorted["Ret_10D%"] > 0).sum()
    print(f"Total events:          {n}")
    print(f"Hit rate (10D > 0):    {pos_10d/n*100:.0f}%  ({pos_10d}/{n})")
    print(f"Avg 10D return:        {df_sorted['Ret_10D%'].mean():+.1f}%")
    print(f"Median 10D return:     {df_sorted['Ret_10D%'].median():+.1f}%")
    print(f"Avg 21D return:        {df_sorted.dropna(subset=['Ret_21D%'])['Ret_21D%'].mean():+.1f}%")

    # Gate effectiveness
    shock_hit  = df_sorted[df_sorted["Shock_Gate"]  == "✅"]
    vel_hit    = df_sorted[df_sorted["RS21Vel_Gate"] == "✅"]
    both_hit   = df_sorted[(df_sorted["Shock_Gate"] == "✅") & (df_sorted["RS21Vel_Gate"] == "✅")]

    print(f"\nEarnings Shock gate fires:  {len(shock_hit)}/{n} events ({len(shock_hit)/n*100:.0f}%)")
    if len(shock_hit):
        print(f"  → avg 10D ret:  {shock_hit['Ret_10D%'].mean():+.1f}%")
    print(f"RS21 Velocity gate fires:   {len(vel_hit)}/{n} events ({len(vel_hit)/n*100:.0f}%)")
    if len(vel_hit):
        print(f"  → avg 10D ret:  {vel_hit['Ret_10D%'].mean():+.1f}%")
    print(f"BOTH gates fire:            {len(both_hit)}/{n} events ({len(both_hit)/n*100:.0f}%)")
    if len(both_hit):
        print(f"  → avg 10D ret:  {both_hit['Ret_10D%'].mean():+.1f}%")

    # Deduplicate by ticker (keep best event per ticker for clean display)
    print("\n" + "=" * 60)
    print("DISTINCT TICKERS (best 10D return per ticker)")
    print("=" * 60)
    df_best = df_sorted.drop_duplicates(subset="Ticker").head(10)
    print(df_best[["Date", "Ticker", "Name", "Jump%", "Vol_Mult", "Dist_52W%",
                   "RS21_Vel5", "Shock_Gate", "RS21Vel_Gate",
                   "Ret_10D%", "Ret_21D%"]].to_string(index=False))

    # Save to CSV for inspection
    df_sorted.to_csv("data/catalyst_events_backtest.csv", index=False)
    print(f"\n✅ Full results saved to data/catalyst_events_backtest.csv")

    # Check if RAIN was found
    rain_events = df[df["Ticker"] == "RAIN.NS"]
    if not rain_events.empty:
        print(f"\n🌧️  RAIN.NS events found: {len(rain_events)}")
        print(rain_events[cols].to_string(index=False))
    else:
        print(f"\n⚠️  RAIN.NS not found — may have missed due to MA50 gate or thresholds")
