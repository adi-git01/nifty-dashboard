#!/usr/bin/env python3
"""
NSE price store vs TradingView (analysis/NSE_VS_TRADINGVIEW.md).

TradingView's public screener gives, per stock, the latest close, 50- and 200-day simple averages, 52-week high / low
and 3/6/12-month returns, all on its own split-adjusted daily series. The same numbers are computed here from the NSE
store (utils/nse_prices.history), so any split, bonus, demerger or missing session we handle differently shows up as
a gap in the averages, the 52-week levels or the returns.

    python scripts/nse_vs_tradingview.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = "analysis/NSE_VS_TRADINGVIEW.md"
URL = "https://scanner.tradingview.com/india/scan"
COLS = ["close", "SMA50", "SMA200", "price_52_week_high", "price_52_week_low", "Perf.3M", "Perf.6M", "Perf.Y"]


def tv_symbol(sym: str) -> str:
    return "NSE:" + sym.replace("&", "_").replace("-", "_")


def fetch_tv(symbols: list) -> pd.DataFrame:
    rows = []
    for i in range(0, len(symbols), 400):
        chunk = symbols[i:i + 400]
        body = json.dumps({"symbols": {"tickers": [tv_symbol(s) for s in chunk], "query": {"types": []}},
                           "columns": COLS}).encode()
        req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json",
                                                              "User-Agent": "Mozilla/5.0"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    j = json.load(r)
                break
            except Exception as e:
                print(f"[tv] batch {i}: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            continue
        back = {tv_symbol(s): s for s in chunk}
        for x in j.get("data", []):
            s = back.get(x.get("s"))
            if s:
                rows.append(dict(symbol=s, **dict(zip(COLS, x.get("d", [])))))
        time.sleep(1)
    return pd.DataFrame(rows)


def ours(h: pd.DataFrame) -> dict:
    c, hi, lo = h["Close"], h["High"], h["Low"]
    last = c.index[-1]
    yr = h[h.index > last - pd.Timedelta(days=365)]

    def perf(days):
        base = c[c.index <= last - pd.Timedelta(days=days)]
        return (c.iloc[-1] / base.iloc[-1] - 1) * 100 if len(base) else np.nan
    return {"close": c.iloc[-1], "SMA50": c.tail(50).mean() if len(c) >= 50 else np.nan,
            "SMA200": c.tail(200).mean() if len(c) >= 200 else np.nan,
            "price_52_week_high": yr["High"].max(), "price_52_week_low": yr["Low"].min(),
            "Perf.3M": perf(91), "Perf.6M": perf(182), "Perf.Y": perf(365), "last": last.date()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from utils import nse_prices as npx
    from utils.fast_data_engine import load_base_fundamentals
    tickers = load_base_fundamentals(live_mode=True)["ticker"].dropna()
    tickers = [t for t in tickers if str(t).endswith(".NS")]
    if a.limit:
        tickers = tickers[:a.limit]
    t0 = time.time()
    H = npx.history(tickers, start=pd.Timestamp.today() - pd.Timedelta(days=430))
    # a renamed stock is listed on TradingView under its new NSE symbol
    fwd = {o: n for n, olds in npx._symbol_map().items() for o, _ in olds}
    name = {t: fwd.get(t[:-3], t[:-3]) for t in tickers}
    N = pd.DataFrame({t: ours(h) for t, h in H.items() if len(h) > 20}).T
    N.index.name = "ticker"
    t_n = time.time() - t0
    t0 = time.time()
    T = fetch_tv(sorted(set(name.values())))
    t_tv = time.time() - t0
    L = ["# NSE store vs TradingView", "", f"Store to {npx.last_date():%d %b %Y} · {len(tickers):,} tickers · NSE has "
         f"{len(N):,} · TradingView returned {len(T):,} · NSE {t_n:.0f}s · TradingView {t_tv:.0f}s", ""]
    if T.empty:
        L.append("TradingView returned nothing (blocked or changed).")
    else:
        T["ticker"] = T["symbol"].map({v: k for k, v in name.items()})
        J = N.join(T.set_index("ticker"), lsuffix="_n", rsuffix="_tv", how="inner")
        L += [f"Compared on {len(J):,} stocks.", "", "| Measure | Within 0.5% | Within 2% | Median abs gap | Worst |",
              "|---|---|---|---|---|"]
        detail = []
        for c in COLS:
            x, y = pd.to_numeric(J[f"{c}_n"], errors="coerce"), pd.to_numeric(J[f"{c}_tv"], errors="coerce")
            ok = x.notna() & y.notna()
            if not ok.any():
                continue
            if c.startswith("Perf"):          # returns: gap in percentage points
                g = (x - y)[ok]
                tol1, tol2, unit = 0.5, 2.0, " pts"
            else:
                g = ((x / y - 1) * 100)[ok]
                tol1, tol2, unit = 0.5, 2.0, "%"
            w = g.abs().idxmax()
            L.append(f"| {c} | {(g.abs() <= tol1).mean():.1%} | {(g.abs() <= tol2).mean():.1%} | {g.abs().median():.3f}{unit} "
                     f"| {w[:-3]} ({x[w]:.4g} vs {y[w]:.4g}) |")
            detail.append(g.abs().rename(c))
        D = pd.concat(detail, axis=1)
        bad = D[(D[["SMA50", "SMA200", "price_52_week_high", "Perf.Y"]] > 2).any(axis=1)]
        L += ["", f"Stocks with a gap > 2% in the 50/200-day average, 52-week high or 1-year return: {len(bad)}", "",
              "| Stock | Close | SMA50 | SMA200 | 52w high | 1y return (pts) | NSE close / TV close |", "|---|---|---|---|---|---|---|"]
        for t in bad.max(axis=1).sort_values(ascending=False).head(40).index:
            r = J.loc[t]
            L.append(f"| {t[:-3]} | {D.loc[t, 'close']:.2f}% | {D.loc[t, 'SMA50']:.1f}% | {D.loc[t, 'SMA200']:.1f}% | "
                     f"{D.loc[t, 'price_52_week_high']:.1f}% | {D.loc[t, 'Perf.Y']:.1f} | {r['close_n']:.2f} / {r['close_tv']:.2f} |")
        J.to_csv("analysis/nse_vs_tradingview.csv")
    os.makedirs("analysis", exist_ok=True)
    open(OUT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
