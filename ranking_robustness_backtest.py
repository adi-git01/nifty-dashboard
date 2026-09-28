"""
Robustness check for ranking_rotation_backtest.py
=================================================

The first run (Actions run 36376061880) passed six variants under the
pre-registered rule. Two reasons not to act on that yet:

1. START-DATE SENSITIVITY. Its 10-year baseline returned +912.6% (Sharpe 1.35)
   while the H2 backtest's baseline, same rules and universe, returned +445%
   (Sharpe 1.03). The main difference is a start date two weeks later. A
   15-name book compounds its first few picks for a decade, so one start date
   is one draw.
2. THE RULE'S WINDOWS ARE NESTED. 1y sits inside 3y inside 5y inside 10y, so
   "4 of 4" is not four independent confirmations; and seven ranking keys were
   tried, so the best is partly selected by luck.

This script tests the finalists four ways:

  A  RECONCILE: rerun H2's exact 10-year window (2016-09-12 -> 2026-09-10,
     no history rule) to confirm the baseline gap is the start date.
  B  STAGGERED STARTS: 8 ten-year runs, start dates 16 sessions apart.
  C  NON-OVERLAPPING PERIODS: three separate books, ~2016-19, 2019-22, 2022-26.
  D  RANDOM-RANKING CONTROL: same eligible set, slots filled in random order.
     If random order also beats the baseline, a key's "improvement" is noise.

A key is credible only if it beats BASE in most staggered starts, in most
non-overlapping periods, AND beats the random-ranking distribution.

Run: python ranking_robustness_backtest.py --max-tickers 0
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from exit_rule_backtest import build_rs_panel, fetch, stats
from ranking_rotation_backtest import DEFAULTS, RS_CAP, build_keys, simulate

FINALISTS = {
    "BASE":               dict(key="base"),
    "K2 RS 3m":           dict(key="rs63"),
    "K3 RS 6m":           dict(key="rs126"),
    "K4 RS 1y":           dict(key="rs252"),
    "K6 blend 1m/3m/1y":  dict(key="blend_long"),
    "K7 dist 200dma":     dict(key="dist200"),
    "I2 IPOs hist>126":   dict(min_hist=126),
}
N_RANDOM = 8
N_STAGGER, STAGGER_STEP = 8, 16
OUT_CSV = "analysis/ranking_robustness.csv"


def run(name, over, close_df, vol_df, nifty, sub, base, keys, hist):
    cfg = {**DEFAULTS, **over}
    key = keys[cfg["key"]] if isinstance(cfg["key"], str) else cfg["key"]
    curve, trades = simulate(name, cfg, close_df, vol_df, nifty, sub, base, key, hist)
    return stats(curve, trades, name)


def main():
    from momentum_factor_backtest import build_pit_universe, load_candidates

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int, default=0)
    ap.add_argument("--top-n", type=int, default=1000)
    args = ap.parse_args()

    end = datetime.now()
    start = end - timedelta(days=365 * 10 + 450 + 200)   # room for the staggered starts
    tickers = load_candidates("all", args.max_tickers)
    close_df, vol_df, nifty = fetch(tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    dates = close_df.index[close_df.index.isin(nifty.index)]
    base = build_rs_panel(close_df, nifty, dates).where(lambda x: x.abs() <= RS_CAP)
    base = base.where(build_pit_universe(close_df, vol_df, args.top_n).reindex(dates))
    keys = build_keys(close_df, nifty, dates, base)
    hist = close_df.reindex(dates).notna().cumsum()
    rng = np.random.default_rng(20260928)
    for s in range(N_RANDOM):
        keys[f"rand{s}"] = pd.DataFrame(rng.random(base.shape), index=dates,
                                        columns=base.columns).where(base.notna())
    RANDOM = {f"RANDOM seed {s}": dict(key=f"rand{s}") for s in range(N_RANDOM)}
    print(f"[data] {len(dates)} days {dates[0].date()} -> {dates[-1].date()}, {close_df.shape[1]} tickers")
    rows = []

    # A. reconcile with H2
    sub = dates[(dates >= "2016-09-12") & (dates <= "2026-09-10")]
    s = run("BASE (H2 window, no hist rule)", dict(min_hist=0), close_df, vol_df, nifty, sub, base, keys, hist)
    print(f"\nA. RECONCILE  H2 window {sub[0].date()} -> {sub[-1].date()}, no history rule: "
          f"return {s['total_return_pct']}%, Sharpe {s['sharpe']}, maxDD {s['max_drawdown_pct']}%  "
          f"(H2 reported +445.0%, 1.03, -40.4%)")

    # B. staggered 10-year starts
    ten = dates[-1] - pd.Timedelta(days=3650)
    i0 = int(np.searchsorted(dates, ten))
    starts = [i0 - k * STAGGER_STEP for k in range(N_STAGGER)]
    for k, si in enumerate(starts):
        sub = dates[si: si + len(dates) - i0]            # every run is the same length
        for name, over in {**FINALISTS, **RANDOM}.items():
            r = run(name, over, close_df, vol_df, nifty, sub, base, keys, hist)
            r.update(test="B_stagger", period=f"start {sub[0].date()}")
            rows.append(r)
        print(f"   B start {sub[0].date()} done", flush=True)

    # C. non-overlapping periods (each a fresh book)
    edges = [dates[i0], pd.Timestamp("2019-12-31"), pd.Timestamp("2022-12-31"), dates[-1]]
    for a, b in zip(edges, edges[1:]):
        sub = dates[(dates > a) & (dates <= b)] if a != edges[0] else dates[(dates >= a) & (dates <= b)]
        for name, over in {**FINALISTS, **RANDOM}.items():
            r = run(name, over, close_df, vol_df, nifty, sub, base, keys, hist)
            r.update(test="C_period", period=f"{sub[0].date()} -> {sub[-1].date()}")
            rows.append(r)
        print(f"   C period {sub[0].date()} -> {sub[-1].date()} done", flush=True)

    T = pd.DataFrame(rows)
    os.makedirs("analysis", exist_ok=True)
    T.to_csv(OUT_CSV, index=False)

    def summarise(test):
        t = T[T.test == test]
        b = t[t.variant == "BASE"].set_index("period")
        rnd = t[t.variant.str.startswith("RANDOM")]
        out = []
        for v in list(FINALISTS) + ["RANDOM (all seeds)"]:
            x = rnd if v.startswith("RANDOM (") else t[t.variant == v]
            x = x.assign(d_sh=x.sharpe.values - b.loc[x.period, "sharpe"].values,
                         d_ret=x.cagr_pct.values - b.loc[x.period, "cagr_pct"].values,
                         d_dd=x.max_drawdown_pct.values - b.loc[x.period, "max_drawdown_pct"].values)
            out.append(dict(variant=v, runs=len(x), beats_base_sharpe=f"{(x.d_sh > 0).sum()}/{len(x)}",
                            median_d_sharpe=round(x.d_sh.median(), 2), median_d_cagr=round(x.d_ret.median(), 2),
                            median_d_maxdd=round(x.d_dd.median(), 1),
                            median_sharpe=round(x.sharpe.median(), 2)))
        return pd.DataFrame(out)

    for test, title in [("B_stagger", f"B. {N_STAGGER} STAGGERED 10-YEAR STARTS ({STAGGER_STEP} sessions apart)"),
                        ("C_period", "C. THREE NON-OVERLAPPING PERIODS (fresh book each)")]:
        print(f"\n{'=' * 100}\n{title}\n  d_ = variant minus BASE on the same start/period; "
              f"d_maxdd > 0 means a shallower drawdown\n{'=' * 100}")
        print(summarise(test).to_string(index=False))
        t = T[T.test == test]
        piv = t[~t.variant.str.startswith("RANDOM")].pivot_table(index="period", columns="variant",
                                                                  values="sharpe").round(2)
        print("\nSharpe by start/period:")
        print(piv.to_string())
        rnd = t[t.variant.str.startswith("RANDOM")].groupby("period").sharpe.agg(["min", "median", "max"]).round(2)
        print("\nRANDOM-ranking Sharpe range per start/period (min / median / max over seeds):")
        print(rnd.to_string())

    print(f"\nsaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
