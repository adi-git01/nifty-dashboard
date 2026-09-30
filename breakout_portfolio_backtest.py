"""
Live-portfolio test: should the engine prefer 52-week-high breakouts in
leader sub-industries when it fills free slots?
=========================================================================

The event studies found that a new 52-week / all-time closing high in a
leader sub-industry (rotation score >= 70) beat the typical stock by ~+3 pp
over 3 months and ~+4 pp over 6 months, and that stocks in laggard
sub-industries (< 40) lag. An event study is not a portfolio, and the live
engine already ranks eligible names by 6-month RS, which may pick most of these
names anyway. This runs the live rules (ranking_rotation_backtest.simulate:
eligibility by CompRS floor + MA50 + liquidity + breadth gate, MA50 exit,
regime trail, 13-session rebalance, 15 equal-weight slots) with:

  BASE   6-month RS ranking (live since PR #62)
  P1     leader breakout first: fresh (<= 10 sessions) 52w/ATH breakout in a
         leader sub-industry fills slots first, then the rest by 6-month RS
  P2     P1 but the breakout also needs a volume anchor (>= 3x 50-day median
         volume, >= Rs 5 cr turnover, CLV >= 0.5)
  P3     any fresh breakout first, whatever the industry
  L1     leader-industry names first (no breakout needed)
  X1     no new buys in laggard sub-industries (score < 40)
  P1+X1  both

Robustness exactly as for the 6-month decision (RANKING_BREADTH_VERDICT.md):
last 1/3/5/10 years, 8 ten-year starts 16 sessions apart, 3 separate periods,
and 8 random-order seeds on the same eligible set as a control. A variant is
credible only if it beats BASE at most starts AND in most periods, and beats
random ordering.

Run: python breakout_portfolio_backtest.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

FRESH = 10
N_RANDOM, N_STAGGER, STEP = 8, 8, 16
OUT_CSV = "analysis/breakout_portfolio_backtest.csv"


def main():
    from breakout_sector_study import fetch_ohlcv, since_last
    from exit_rule_backtest import build_rs_panel, stats
    from momentum_factor_backtest import build_pit_universe, load_candidates
    from ranking_rotation_backtest import DEFAULTS, RS_CAP, build_keys, simulate
    from transfer_backtest import group_panels
    from utils.nifty1000_list import SUB_INDUSTRY_MAP
    from utils.yf_safe import safe_history

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int, default=0)
    args = ap.parse_args()
    pd.set_option("display.width", 220)

    close, high, low, vol, _ = fetch_ohlcv(load_candidates("all", args.max_tickers), "2014-06-01")
    nifty = safe_history("^NSEI", start="2014-06-01")
    if nifty.index.tz is not None:
        nifty.index = nifty.index.tz_localize(None)
    nifty = nifty.reindex(close.index).ffill()
    dates = close.index

    pit = build_pit_universe(close, vol, 1000)
    base = build_rs_panel(close, nifty, dates).where(lambda x: x.abs() <= RS_CAP).where(pit)
    keys = build_keys(close, nifty, dates, base)
    hist = close.notna().cumsum()

    # sub-industry score per stock (the heatmap's score_0_100, live formula)
    _, gk, _ = group_panels(close, pit, SUB_INDUSTRY_MAP, nifty["Close"], [(5, .10), (21, .50), (63, .40)],
                            india_live=True)
    score = gk["live"].rank(axis=1, pct=True) * 100
    score = score.where(score.notna().sum(axis=1) >= 20)
    gscore = pd.DataFrame({t: score[g] if (g := SUB_INDUSTRY_MAP.get(t)) in score.columns else np.nan
                           for t in close.columns}, index=dates)
    leader, laggard = gscore >= 70, gscore < 40

    # fresh 52-week / all-time closing-high breakouts (breakout_sector_study definitions)
    hi52 = close.rolling(252, min_periods=200).max()
    new52 = close.ge(hi52) & hi52.notna()
    ath = (close > close.ffill().cummax().shift(1)) & (close.notna().cumsum() >= 500)
    ev = (new52 & (since_last(new52) >= 20)) | (ath & (since_last(ath) >= 20))
    clv = (close - low) / (high - low).where(lambda x: x > 0)
    vmed = vol.rolling(50, min_periods=30).median().shift(1)
    anchor = (vol >= 3 * vmed) & (close * vol >= 5e7) & (clv >= 0.5)
    fresh = ev.astype(float).rolling(FRESH, min_periods=1).max().astype(bool)
    fresh_anchor = (ev & anchor).astype(float).rolling(FRESH, min_periods=1).max().astype(bool)

    rs126 = keys["rs126"]
    first = lambda flag: (rs126 + 1e4 * flag.astype(float)).where(base.notna())
    keys.update({
        "P1": first(fresh & leader),
        "P2": first(fresh_anchor & leader),
        "P3": first(fresh),
        "L1": first(leader),
    })
    elig_x1 = base.where(~laggard)                      # NaN score (unmapped) stays eligible
    rng = np.random.default_rng(20260930)
    for s in range(N_RANDOM):
        keys[f"rand{s}"] = pd.DataFrame(rng.random(base.shape), index=dates, columns=base.columns).where(base.notna())

    V = {
        "BASE 6m RS":              (dict(key="rs126"), base),
        "P1 leader breakout 1st":  (dict(key="P1"), base),
        "P2 +vol anchor 1st":      (dict(key="P2"), base),
        "P3 any breakout 1st":     (dict(key="P3"), base),
        "L1 leader industry 1st":  (dict(key="L1"), base),
        "X1 skip laggard ind.":    (dict(key="rs126"), elig_x1),
        "P1+X1":                   (dict(key="P1"), elig_x1),
    }
    R = {f"RANDOM {s}": (dict(key=f"rand{s}"), base) for s in range(N_RANDOM)}
    flags = {"leader breakout": fresh & leader, "any breakout": fresh, "laggard industry": laggard}

    rows, share = [], []
    def run(name, over, el, sub, test, period):
        cfg = {**DEFAULTS, **over}
        curve, trades = simulate(name, cfg, close, vol, nifty, sub, el, keys[cfg["key"]], hist)
        r = stats(curve, trades, name)
        r.update(test=test, period=period)
        rows.append(r)
        if test == "window" and period == "10y" and len(trades):
            t = trades.drop_duplicates(["ticker", "entry_date"])
            ii, jj = dates.get_indexer(t.entry_date), close.columns.get_indexer(t.ticker)
            share.append(dict(variant=name, buys=len(t),
                              **{f"% buys {k}": round(v.values[ii, jj].mean() * 100, 1) for k, v in flags.items()}))

    end = dates[-1]
    for y in (1, 3, 5, 10):
        sub = dates[dates >= end - pd.Timedelta(days=365 * y)]
        for n, (o, el) in V.items():
            run(n, o, el, sub, "window", f"{y}y")
        print(f"   window {y}y done", flush=True)
    L = int(np.searchsorted(dates, end - pd.Timedelta(days=3650)))
    for k in range(N_STAGGER):
        i0 = L - k * STEP
        sub = dates[i0: i0 + len(dates) - L]
        for n, (o, el) in {**V, **R}.items():
            run(n, o, el, sub, "stagger", f"start {sub[0].date()}")
        print(f"   stagger {sub[0].date()} done", flush=True)
    edges = [dates[L], pd.Timestamp("2019-12-31"), pd.Timestamp("2022-12-31"), end]
    for a, b in zip(edges, edges[1:]):
        sub = dates[(dates >= a) & (dates <= b)] if a == edges[0] else dates[(dates > a) & (dates <= b)]
        for n, (o, el) in {**V, **R}.items():
            run(n, o, el, sub, "period", f"{sub[0].date()} -> {sub[-1].date()}")
        print(f"   period {sub[0].date()} -> {sub[-1].date()} done", flush=True)

    T = pd.DataFrame(rows)
    os.makedirs("analysis", exist_ok=True)
    T.to_csv(OUT_CSV, index=False)

    print(f"\n{'=' * 110}\nLAST 1/3/5/10 YEARS (nested -- not independent)\n{'=' * 110}")
    w = T[T.test == "window"]
    for m in ("total_return_pct", "sharpe", "max_drawdown_pct"):
        print(f"\n{m}:")
        print(w.pivot_table(index="variant", columns="period", values=m).reindex(list(V))[["1y", "3y", "5y", "10y"]].to_string())
    print("\nWhat each variant bought over 10 years:")
    print(pd.DataFrame(share).to_string(index=False))

    for test, title in (("stagger", f"{N_STAGGER} STAGGERED 10-YEAR STARTS"), ("period", "3 SEPARATE PERIODS")):
        t = T[T.test == test]
        b = t[t.variant == "BASE 6m RS"].set_index("period")
        rnd = t[t.variant.str.startswith("RANDOM")]
        best = rnd.groupby("period").sharpe.max()
        out = []
        for v in list(V)[1:]:
            x = t[t.variant == v].set_index("period")
            d = x.sharpe - b.sharpe.reindex(x.index)
            out.append(dict(variant=v, beats_base=f"{(d > 0).sum()}/{len(d)}", median_d_sharpe=round(d.median(), 2),
                            median_d_cagr=round((x.cagr_pct - b.cagr_pct.reindex(x.index)).median(), 2),
                            median_d_maxdd=round((x.max_drawdown_pct - b.max_drawdown_pct.reindex(x.index)).median(), 1),
                            beats_best_random=f"{(x.sharpe > best.reindex(x.index)).sum()}/{len(x)}",
                            median_sharpe=round(x.sharpe.median(), 2)))
        out.append(dict(variant="BASE 6m RS", beats_base="-", median_d_sharpe=0.0, median_d_cagr=0.0,
                        median_d_maxdd=0.0, beats_best_random=f"{(b.sharpe > best.reindex(b.index)).sum()}/{len(b)}",
                        median_sharpe=round(b.sharpe.median(), 2)))
        print(f"\n{'=' * 110}\n{title}  (d_ = variant minus BASE; d_maxdd > 0 = shallower drawdown)\n{'=' * 110}")
        print(pd.DataFrame(out).to_string(index=False))
        print("\nSharpe by start/period:")
        print(t[~t.variant.str.startswith("RANDOM")].pivot_table(index="period", columns="variant",
                                                                 values="sharpe").round(2)[list(V)].to_string())
        print("random-order Sharpe (min / median / max):")
        print(rnd.groupby("period").sharpe.agg(["min", "median", "max"]).round(2).to_string())
    print(f"\nsaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
