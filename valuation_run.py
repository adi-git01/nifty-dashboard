"""
Mechanical v4.1 valuation for every in-scope stock -> data/valuation_snapshot.csv
==================================================================================

One row per stock: rating, CMP percentile, P10/P50/P90, implied 3-year CAGR
per scenario, gate status, and the full 1-99 percentile grid of the fair-value
distribution. The dashboard re-percentiles that grid at today's price, so the
table stays current between runs; the single-stock view re-runs the engine
live (utils/valuation.value_stock).

Prices: the latest daily cache (data/cache/market_master_*.parquet), with its
52-week high/low for Gate E.

Run: python valuation_run.py   (also runs after fundamentals_annual.py in the
     Ranking / Rotation workflow, script=fundamentals)
"""
from __future__ import annotations

import glob
import sys
import time

import pandas as pd

from utils.valuation import load_context, summary_row, value_stock

OUT = "data/valuation_snapshot.csv"


def main():
    ctx = load_context()
    if ctx.annual.empty:
        sys.exit("[valuation] data/fundamentals_annual.csv.gz missing -- run fundamentals_annual.py first")
    files = sorted(glob.glob("data/cache/market_master_*.parquet"))
    snap = pd.read_parquet(files[-1]) if files else pd.DataFrame()
    px = snap.set_index("ticker") if not snap.empty else pd.DataFrame()
    tickers = sorted(set(ctx.annual.ticker) & set(px.index)) if not px.empty else sorted(set(ctx.annual.ticker))
    rows, t0 = [], time.time()
    for k, t in enumerate(tickers):
        price = float(px.at[t, "price"]) if t in px.index and pd.notna(px.at[t, "price"]) else None
        hi = float(px.at[t, "fiftyTwoWeekHigh"]) if t in px.index and "fiftyTwoWeekHigh" in px else None
        lo = float(px.at[t, "fiftyTwoWeekLow"]) if t in px.index and "fiftyTwoWeekLow" in px else None
        try:
            res = value_stock(t, ctx, price=price, knobs=True, hi52=hi, lo52=lo)
        except Exception as e:                       # one bad stock must not stop the run
            res = {"ticker": t.replace(".NS", ""), "status": "ERROR", "reason": str(e)[:120]}
        rows.append(summary_row(res))
        if (k + 1) % 200 == 0:
            print(f"[valuation] {k + 1}/{len(tickers)} ({time.time() - t0:.0f}s)", flush=True)
    T = pd.DataFrame(rows)
    T["as_of"] = files[-1].split("master_")[1][:10].replace("_", "-") if files else ""
    T.to_csv(OUT, index=False)
    ok = T[T.status == "OK"]
    print(f"[valuation] {len(T)} stocks in {time.time() - t0:.0f}s -> {OUT}")
    print(f"   status: {T.status.value_counts().to_dict()}")
    if len(ok):
        rat = ok.rating.str.replace(r" \(gate.*", " (gate fail)", regex=True)
        print(f"   ratings: {rat.value_counts().to_dict()}")
        print(f"   identity: {ok.identity.value_counts().to_dict()}; median CMP percentile {ok.cmp_pct.median():.0f}")
    if (T.status == "ERROR").any():
        print("   errors:", T[T.status == "ERROR"][["ticker", "reason"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
