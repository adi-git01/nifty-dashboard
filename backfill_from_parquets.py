"""
Backfill Market Mood & Breadth from Existing Parquet Cache
===========================================================
Reads every parquet in data/cache/ and computes the same mood + breadth
metrics that trading_engine.py / calculate_mood_metrics() /
calculate_breadth_from_df() would have produced on that day.

No yfinance required — runs in seconds using cached data.
Safe to re-run: only fills dates not already in the CSVs.
"""

import pandas as pd
import numpy as np
import os
import glob
import re
from datetime import datetime

CACHE_DIR    = "data/cache"
MOOD_FILE    = "data/market_mood_history.csv"
BREADTH_FILE = "data/market_breadth_history.csv"


def extract_date_from_filename(fname):
    """market_master_2026_04_17.parquet → '2026-04-17'"""
    m = re.search(r"(\d{4})_(\d{2})_(\d{2})", fname)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def compute_from_df(df, date_str):
    """
    Identical logic to calculate_mood_metrics() + calculate_breadth_from_df()
    in utils/market_mood.py and utils/market_breadth.py.
    """
    if df is None or df.empty:
        return None, None

    total = len(df)

    # ── Mood metrics ────────────────────────────────────────────────────────
    strong_momentum = 0.0
    total_uptrends  = 0.0
    avg_trend_score = 50.0
    breakout_alerts = 0.0

    if "trend_score" in df.columns:
        scores = df["trend_score"].dropna()
        n = len(scores)
        if n > 0:
            strong_momentum = (scores >= 80).sum() / n * 100
            avg_trend_score = float(scores.mean())

    if "trend_signal" in df.columns:
        total_uptrends = (
            df["trend_signal"].str.contains("UPTREND", na=False).sum() / total * 100
        )

    if "dist_52w" in df.columns:
        breakout_alerts = (df["dist_52w"] >= -5).sum() / total * 100

    mood = {
        "date":            date_str,
        "strong_momentum": round(strong_momentum, 1),
        "total_uptrends":  round(total_uptrends,  1),
        "avg_trend_score": round(avg_trend_score,  1),
        "breakout_alerts": round(breakout_alerts,  1),
    }

    # ── Breadth metrics ─────────────────────────────────────────────────────
    above_50dma  = 0
    above_200dma = 0

    if "dist_200dma" in df.columns:
        above_200dma = (df["dist_200dma"] > 0).sum()

    if "trend_score" in df.columns:
        # Proxy: trend_score >= 50 → stock likely above MA50
        above_50dma = (df["trend_score"] >= 50).sum()

    pct_above_50dma  = round(above_50dma  / total * 100, 1) if total else 0
    pct_above_200dma = round(above_200dma / total * 100, 1) if total else 0

    near_52w_high = 0
    if "dist_52w" in df.columns:
        near_52w_high = (df["dist_52w"] >= -5).sum()

    breadth = {
        "date":                 date_str,
        "pct_above_50dma":     pct_above_50dma,
        "pct_above_200dma":    pct_above_200dma,
        "pct_strong_momentum": round(strong_momentum, 1),
        "pct_uptrends":        round(total_uptrends,  1),
        "pct_near_52w_high":   round(near_52w_high / total * 100, 1) if total else 0,
        "total_stocks":        total,
    }

    return mood, breadth


def main():
    os.makedirs("data", exist_ok=True)

    # Load existing history
    mood_df    = pd.read_csv(MOOD_FILE)    if os.path.exists(MOOD_FILE)    else pd.DataFrame()
    breadth_df = pd.read_csv(BREADTH_FILE) if os.path.exists(BREADTH_FILE) else pd.DataFrame()

    existing_mood    = set(mood_df["date"].astype(str).tolist()) if not mood_df.empty else set()
    existing_breadth = set(breadth_df["date"].astype(str).tolist()) if not breadth_df.empty else set()

    print(f"[BACKFILL] Existing mood rows:    {len(existing_mood)}")
    print(f"[BACKFILL] Existing breadth rows: {len(existing_breadth)}")

    # Scan parquet cache
    parquets = sorted(glob.glob(os.path.join(CACHE_DIR, "market_master_*.parquet")))
    print(f"[BACKFILL] Found {len(parquets)} parquet files in {CACHE_DIR}")

    mood_new    = []
    breadth_new = []

    for path in parquets:
        date_str = extract_date_from_filename(os.path.basename(path))
        if not date_str:
            continue

        needs_mood    = date_str not in existing_mood
        needs_breadth = date_str not in existing_breadth

        if not needs_mood and not needs_breadth:
            continue  # already have both for this date

        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"  SKIP {date_str}: could not read parquet ({e})")
            continue

        mood, breadth = compute_from_df(df, date_str)

        if needs_mood and mood:
            mood_new.append(mood)
            print(f"  [MOOD]    {date_str}  avg_score={mood['avg_trend_score']}  uptrends={mood['total_uptrends']}%  breakouts={mood['breakout_alerts']}%")

        if needs_breadth and breadth:
            breadth_new.append(breadth)
            print(f"  [BREADTH] {date_str}  >50DMA={breadth['pct_above_50dma']}%  >200DMA={breadth['pct_above_200dma']}%  stocks={breadth['total_stocks']}")

    print(f"\n[BACKFILL] New mood rows:    {len(mood_new)}")
    print(f"[BACKFILL] New breadth rows: {len(breadth_new)}")

    if mood_new:
        combined = pd.concat([mood_df, pd.DataFrame(mood_new)], ignore_index=True)
        combined["date"] = combined["date"].astype(str)
        combined = combined.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        combined.to_csv(MOOD_FILE, index=False)
        print(f"[BACKFILL] ✓ Saved mood history: {len(combined)} total rows → {MOOD_FILE}")
        print(f"           Date range: {combined['date'].iloc[0]} → {combined['date'].iloc[-1]}")

    if breadth_new:
        combined = pd.concat([breadth_df, pd.DataFrame(breadth_new)], ignore_index=True)
        combined["date"] = combined["date"].astype(str)
        combined = combined.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        combined.to_csv(BREADTH_FILE, index=False)
        print(f"[BACKFILL] ✓ Saved breadth history: {len(combined)} total rows → {BREADTH_FILE}")
        print(f"           Date range: {combined['date'].iloc[0]} → {combined['date'].iloc[-1]}")

    if not mood_new and not breadth_new:
        print("[BACKFILL] Nothing to add — history is already up to date.")


if __name__ == "__main__":
    main()
