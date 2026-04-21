"""
Market Mood & Breadth History Backfill
=======================================
Downloads 2+ years of price history for the full Nifty 750 universe,
then reconstructs market mood and breadth metrics for every missing trading
day using the *exact same formula* as trend_engine.py.

Run once to fill any gaps:
    python backfill_mood_breadth.py

Safe to re-run — only inserts dates that are not already in the CSVs.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, datetime
import os
import sys
import time

MOOD_FILE    = "data/market_mood_history.csv"
BREADTH_FILE = "data/market_breadth_history.csv"

# Download start: 2 years back gives ~500 trading days — enough for MA200
DOWNLOAD_START = "2024-01-01"
DOWNLOAD_END   = date.today().strftime("%Y-%m-%d")

# Minimum number of stocks with valid data to record a day
MIN_STOCKS = 100


def load_tickers():
    df = pd.read_csv("data/nifty1000_list.csv")
    tickers = df["Ticker"].dropna().tolist()
    print(f"[BACKFILL] Universe: {len(tickers)} tickers")
    return tickers


def load_existing(path, date_col="date"):
    if os.path.exists(path):
        df = pd.read_csv(path)
        df[date_col] = df[date_col].astype(str)
        return df
    return pd.DataFrame()


def download_prices(tickers):
    """Bulk download in two chunks to avoid yfinance timeouts with 750 tickers."""
    print(f"[BACKFILL] Downloading price history {DOWNLOAD_START} → {DOWNLOAD_END}...")
    mid = len(tickers) // 2
    chunks = [tickers[:mid], tickers[mid:]]
    closes, highs, lows = [], [], []

    for i, chunk in enumerate(chunks):
        print(f"  chunk {i+1}/2 ({len(chunk)} tickers)...")
        for attempt in range(3):
            try:
                raw = yf.download(
                    chunk,
                    start=DOWNLOAD_START,
                    end=DOWNLOAD_END,
                    auto_adjust=True,
                    threads=True,
                    progress=False,
                )
                closes.append(raw["Close"])
                highs.append(raw["High"])
                lows.append(raw["Low"])
                break
            except Exception as e:
                print(f"  attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(5)
        else:
            print(f"  chunk {i+1} permanently failed — skipping")

    if not closes:
        raise RuntimeError("All download attempts failed.")

    close = pd.concat(closes, axis=1)
    high  = pd.concat(highs,  axis=1)
    low   = pd.concat(lows,   axis=1)

    # Remove duplicate columns (can happen if ticker appears in both chunks)
    close = close.loc[:, ~close.columns.duplicated()]
    high  = high.loc[:,  ~high.columns.duplicated()]
    low   = low.loc[:,   ~low.columns.duplicated()]

    print(f"[BACKFILL] Downloaded: {close.shape[1]} tickers × {len(close)} trading days")
    return close, high, low


def compute_indicators(close, high, low):
    """
    Vectorized computation of all indicators across all tickers and dates.
    Formula is identical to utils/trend_engine.py → calculate_stock_trend_history().
    """
    print("[BACKFILL] Computing rolling indicators (MA50, MA200, 52W high/low)...")

    ma50  = close.rolling(50,  min_periods=50).mean()
    ma200 = close.rolling(200, min_periods=200).mean()

    high_52w = high.rolling(252, min_periods=50).max()
    low_52w  = low.rolling(252,  min_periods=50).min()

    # ── MA score (same as trend_engine.py) ─────────────────────────────────
    # Price > MA50:   +15 / -10
    # Price > MA200:  +15 / -15
    # Golden cross:   +10 / -5
    above_50  = (close > ma50).fillna(False)
    above_200 = (close > ma200).fillna(False)
    golden    = (ma50  > ma200).fillna(False)

    ma_score = (
        above_50.astype(float)  * 25 - 10 +
        above_200.astype(float) * 30 - 15 +
        golden.astype(float)    * 15 - 5
    )

    # ── 52-week position score ──────────────────────────────────────────────
    range_52  = (high_52w - low_52w).replace(0, np.nan)
    pos_score = ((close - low_52w) / range_52 - 0.5) * 30
    pos_score = pos_score.fillna(0)

    # ── High bonus / penalty ────────────────────────────────────────────────
    dist_from_high = (close - high_52w) / high_52w.replace(0, np.nan)
    near_high_flag = dist_from_high >= -0.05   # within 5% of 52W high
    low_flag       = dist_from_high <  -0.30   # more than 30% below 52W high

    high_bonus = near_high_flag.astype(float) * 10 + low_flag.astype(float) * (-10)

    # ── Final trend score ────────────────────────────────────────────────────
    trend_scores = (50 + ma_score + pos_score + high_bonus).clip(0, 100)

    # Null out scores where MA50 not yet computed (insufficient history)
    trend_scores = trend_scores.where(ma50.notna(), np.nan)

    print("[BACKFILL] Indicators computed.")
    return trend_scores, above_50, above_200, near_high_flag


def aggregate_daily(trend_scores, above_50, above_200, near_high_flag):
    """
    Aggregate per-stock indicators to market-level percentages for each date.
    Matches the metrics saved by calculate_mood_metrics() and calculate_breadth_from_df().
    """
    valid_n = trend_scores.notna().sum(axis=1)

    # Mood metrics
    strong_mom_pct = (trend_scores >= 80).sum(axis=1) / valid_n * 100
    uptrends_pct   = (trend_scores >= 60).sum(axis=1) / valid_n * 100
    avg_score      = trend_scores.mean(axis=1)
    near_high_pct  = near_high_flag.sum(axis=1) / valid_n * 100

    # Breadth metrics (actual MA comparison, more accurate than the proxy)
    # Only count stocks where MA is defined
    valid_50  = above_50.astype(float).where(trend_scores.notna(), np.nan).notna().sum(axis=1)
    valid_200 = above_200.astype(float).where(trend_scores.notna(), np.nan).notna().sum(axis=1)
    above_50_pct  = above_50.sum(axis=1) / valid_50.replace(0, np.nan) * 100
    above_200_pct = above_200.sum(axis=1) / valid_200.replace(0, np.nan) * 100

    return pd.DataFrame({
        "valid_n":        valid_n,
        "strong_mom_pct": strong_mom_pct.round(1),
        "uptrends_pct":   uptrends_pct.round(1),
        "avg_score":      avg_score.round(1),
        "near_high_pct":  near_high_pct.round(1),
        "above_50_pct":   above_50_pct.round(1),
        "above_200_pct":  above_200_pct.round(1),
    }, index=trend_scores.index)


def build_mood_row(d_str, agg):
    return {
        "date":            d_str,
        "strong_momentum": float(agg["strong_mom_pct"]),
        "total_uptrends":  float(agg["uptrends_pct"]),
        "avg_trend_score": float(agg["avg_score"]),
        "breakout_alerts": float(agg["near_high_pct"]),
    }


def build_breadth_row(d_str, agg):
    return {
        "date":                 d_str,
        "pct_above_50dma":     float(agg["above_50_pct"]),
        "pct_above_200dma":    float(agg["above_200_pct"]),
        "pct_strong_momentum": float(agg["strong_mom_pct"]),
        "pct_uptrends":        float(agg["uptrends_pct"]),
        "pct_near_52w_high":   float(agg["near_high_pct"]),
        "total_stocks":        int(agg["valid_n"]),
    }


def main():
    os.makedirs("data", exist_ok=True)

    tickers    = load_tickers()
    mood_df    = load_existing(MOOD_FILE)
    breadth_df = load_existing(BREADTH_FILE)

    existing_mood_dates    = set(mood_df["date"].tolist()) if not mood_df.empty else set()
    existing_breadth_dates = set(breadth_df["date"].tolist()) if not breadth_df.empty else set()

    close, high, low = download_prices(tickers)

    trend_scores, above_50, above_200, near_high_flag = compute_indicators(
        close, high, low
    )
    daily = aggregate_daily(trend_scores, above_50, above_200, near_high_flag)

    # Normalize index to plain date strings for comparison
    daily.index = pd.to_datetime(daily.index).normalize()
    all_dates_str = daily.index.strftime("%Y-%m-%d").tolist()

    mood_new    = []
    breadth_new = []
    skipped     = 0

    for i, d_str in enumerate(all_dates_str):
        row = daily.iloc[i]

        if row["valid_n"] < MIN_STOCKS:
            skipped += 1
            continue

        if d_str not in existing_mood_dates:
            mood_new.append(build_mood_row(d_str, row))

        if d_str not in existing_breadth_dates:
            breadth_new.append(build_breadth_row(d_str, row))

    print(f"\n[BACKFILL] Skipped {skipped} dates with < {MIN_STOCKS} valid stocks")
    print(f"[BACKFILL] New mood rows:    {len(mood_new)}")
    print(f"[BACKFILL] New breadth rows: {len(breadth_new)}")

    if mood_new:
        combined = pd.concat(
            [mood_df, pd.DataFrame(mood_new)], ignore_index=True
        )
        combined["date"] = combined["date"].astype(str)
        combined = combined.sort_values("date").drop_duplicates("date")
        combined.to_csv(MOOD_FILE, index=False)
        print(f"[BACKFILL] Saved mood history:    {len(combined)} total rows → {MOOD_FILE}")

    if breadth_new:
        combined = pd.concat(
            [breadth_df, pd.DataFrame(breadth_new)], ignore_index=True
        )
        combined["date"] = combined["date"].astype(str)
        combined = combined.sort_values("date").drop_duplicates("date")
        combined.to_csv(BREADTH_FILE, index=False)
        print(f"[BACKFILL] Saved breadth history: {len(combined)} total rows → {BREADTH_FILE}")

    if not mood_new and not breadth_new:
        print("[BACKFILL] Nothing to add — history is already up to date.")
    else:
        print("[BACKFILL] Done.")


if __name__ == "__main__":
    main()
