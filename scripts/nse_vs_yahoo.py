#!/usr/bin/env python3
"""
NSE store vs Yahoo, side by side, before Yahoo is retired (analysis/NSE_VS_YAHOO.md).

  1. Prices: each stock's adjusted close history from both, over the same sessions. A gap > 2% on some day
     usually means a corporate action one side handled and the other did not (or Yahoo's dividend adjustment).
  2. The engine: the full snapshot (fast_data_engine.fetch_and_process_market_data) built once from NSE and
     once from Yahoo, then every key column and every signal compared. This is what the dashboard, alerts
     and the model portfolio see.
  3. Speed and coverage of each.

    python scripts/nse_vs_yahoo.py [--limit N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = "analysis/NSE_VS_YAHOO.md"
NUM = ["currentPrice", "previousClose", "fiftyDayAverage", "twoHundredDayAverage", "fiftyTwoWeekHigh", "comp_rs", "rs_6m",
       "trend_score", "volatility", "dist_52w", "dist_ma50", "overall", "volume_signal_score"]
CAT = ["dna_signal", "trend_signal", "fx_action", "fx_state", "simba_grade", "breakout_label", "entry_label"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from utils import nse_prices as npx
    from utils.fast_data_engine import fetch_and_process_market_data, load_base_fundamentals
    from utils.yf_safe import _safe_download_yahoo
    L = ["# NSE store vs Yahoo", "", f"Store: sessions to {npx.last_date():%d %b %Y}; current: {npx.is_current()}", ""]

    fund = load_base_fundamentals(live_mode=True)
    tickers = fund["ticker"].dropna().tolist()
    if a.limit:
        tickers, fund = tickers[:a.limit], fund[fund.ticker.isin(tickers[:a.limit])]

    # ---- 1. raw price histories
    t0 = time.time()
    nse = npx.history(tickers, start=pd.Timestamp.today() - pd.Timedelta(days=430))
    t_nse_hist = time.time() - t0
    t0 = time.time()
    ya = _safe_download_yahoo(tickers, start=(pd.Timestamp.today() - pd.Timedelta(days=430)).strftime("%Y-%m-%d"),
                              group_by="ticker", auto_adjust=False, threads=False, min_coverage=0.5)
    t_ya_hist = time.time() - t0
    rows = []
    for t in tickers:
        n = nse.get(t)
        try:
            y = ya[t]["Close"].dropna() if ya is not None and t in ya.columns.get_level_values(0) else None
        except Exception:
            y = None
        if y is not None and getattr(y.index, "tz", None) is not None:
            y.index = y.index.tz_localize(None)
        if n is None or y is None or y.empty:
            rows.append(dict(ticker=t, nse=n is not None, yahoo=y is not None and not (y is None or y.empty)))
            continue
        j = pd.concat([n["Close"].rename("n"), y.rename("y")], axis=1).dropna()
        if j.empty:
            rows.append(dict(ticker=t, nse=True, yahoo=True, overlap=0))
            continue
        d = (j.n / j.y - 1) * 100
        rows.append(dict(ticker=t, nse=True, yahoo=True, overlap=len(j), last_gap=float(d.iloc[-1]), med_gap=float(d.abs().median()),
                         max_gap=float(d.abs().max()), max_day=d.abs().idxmax().date(), n_last=n.index.max().date(),
                         y_last=y.index.max().date(), sessions_nse=len(n), sessions_y=len(y)))
    P = pd.DataFrame(rows)
    both = P[P.get("overlap", pd.Series(0, index=P.index)).fillna(0) > 0] if "overlap" in P else P.iloc[:0]
    L += ["## 1 · Price histories (split-adjusted close, last ~14 months)", "",
          f"- {len(tickers):,} tickers · NSE has {int(P.nse.sum()):,} · Yahoo has {int(P.yahoo.sum()):,} · both {len(both):,}",
          f"- NSE only: {', '.join(P[P.nse & ~P.yahoo].ticker.str[:-3].head(30))}",
          f"- Yahoo only: {', '.join(P[~P.nse & P.yahoo].ticker.str[:-3].head(30))}"]
    if len(both):
        L += [f"- last close within 0.1%: {(both.last_gap.abs() <= 0.1).mean():.1%} · median daily gap {both.med_gap.median():.3f}%",
              f"- stocks whose history differs by > 2% on some day: {(both.max_gap > 2).sum()} (corporate action handled "
              "differently, or Yahoo's dividend adjustment)", "",
              "| Stock | Max gap | On | Last gap | NSE last | Yahoo last |", "|---|---|---|---|---|---|"]
        for r in both.sort_values("max_gap", ascending=False).head(25).itertuples():
            L.append(f"| {r.ticker[:-3]} | {r.max_gap:.1f}% | {r.max_day} | {r.last_gap:+.2f}% | {r.n_last} | {r.y_last} |")
    L += ["", f"Fetch time: NSE store {t_nse_hist:.1f}s · Yahoo {t_ya_hist:.0f}s", ""]

    # ---- 2. the engine's snapshot both ways
    t0 = time.time()
    S_n = fetch_and_process_market_data(tickers, fund.copy(), live_mode=True, save=False)
    t_n = time.time() - t0
    t0 = time.time()
    S_y = fetch_and_process_market_data(tickers, fund.copy(), live_mode=True, save=False, price_source="yahoo")
    t_y = time.time() - t0
    from utils.yf_safe import source_counts
    L += ["## 2 · Engine snapshot built from each", "",
          f"- rows: NSE {len(S_n):,} · Yahoo {len(S_y):,} · build time NSE {t_n:.0f}s · Yahoo {t_y:.0f}s · sources used {source_counts()}"]
    if len(S_n) and len(S_y):
        J = S_n.set_index("ticker").join(S_y.set_index("ticker"), lsuffix="_n", rsuffix="_y", how="inner")
        L += [f"- compared on {len(J):,} stocks", "", "| Column | Within 1% (or 1 pt) | Median abs diff | Worst |", "|---|---|---|---|"]
        for c in NUM:
            if f"{c}_n" not in J or f"{c}_y" not in J:
                continue
            x, y = pd.to_numeric(J[f"{c}_n"], errors="coerce"), pd.to_numeric(J[f"{c}_y"], errors="coerce")
            ok = x.notna() & y.notna()
            if not ok.any():
                continue
            diff = (x - y)[ok]
            rel = (diff / y[ok].abs().replace(0, np.nan)).abs()
            close = ((rel <= 0.01) | (diff.abs() <= 1)).mean()
            worst = diff.abs().idxmax()
            L.append(f"| {c} | {close:.1%} | {diff.abs().median():.3g} | {worst[:-3]} ({x[worst]:.4g} vs {y[worst]:.4g}) |")
        L += ["", "| Signal | Same on | Different (NSE → Yahoo), examples |", "|---|---|---|"]
        for c in CAT:
            if f"{c}_n" not in J or f"{c}_y" not in J:
                continue
            a_, b_ = J[f"{c}_n"].astype(str), J[f"{c}_y"].astype(str)
            diff = J[a_ != b_]
            ex = ", ".join(f"{t[:-3]} {x}→{y}" for t, x, y in zip(diff.index[:8], a_[a_ != b_][:8], b_[a_ != b_][:8]))
            L.append(f"| {c} | {(a_ == b_).mean():.1%} | {len(diff)}: {ex} |")
        buy_n = set(J.index[J.get("fx_action_n", pd.Series(dtype=str)).astype(str) == "BUY NOW"])
        buy_y = set(J.index[J.get("fx_action_y", pd.Series(dtype=str)).astype(str) == "BUY NOW"])
        L += ["", f"Technicals BUY NOW: NSE {len(buy_n)} · Yahoo {len(buy_y)} · both {len(buy_n & buy_y)} · "
                  f"only NSE {sorted(t[:-3] for t in buy_n - buy_y)[:15]} · only Yahoo {sorted(t[:-3] for t in buy_y - buy_n)[:15]}"]
    os.makedirs("analysis", exist_ok=True)
    open(OUT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
