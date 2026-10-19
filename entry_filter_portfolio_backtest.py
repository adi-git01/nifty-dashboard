"""
Live-portfolio test of the entry_exit_study findings: does the engine do better if new buys need a
breakout / squeeze release, skip large caps, or skip loss-makers?
=================================================================================================

The live rules (ranking_rotation_backtest.simulate: CompRS floor + MA50 + liquidity + breadth gate,
6-month RS ranking, regime trail, MA50 exit, 13-session rebalance, 15 equal-weight slots) with only the
set of names allowed into a FREE slot changed:

  BASE        live engine (6-month RS ranking)
  E1 trigger  a new buy needs a fresh (<= 10 sessions) 52w/ATH breakout that has held, or a Bollinger
              squeeze release -- the two entries that passed H6 in entry_exit_study.py
  E2 trig 1st those names fill free slots first, the rest by 6-month RS (the earlier "breakout first"
              test, breakout_portfolio_backtest.py, made the book worse -- kept here for comparison)
  Z1 no large no new buys in the top 100 by market cap (large-cap momentum had no edge in 2021-26)
  X1 no loss  no new buys when the latest PUBLISHED annual EPS (period end + 92 days) is <= 0
  C1          E1 + Z1 + X1;   C2  E2 + Z1 + X1;   ZX  Z1 + X1

Robustness as for every engine change: last 1/3/5/10 years, 8 ten-year starts 16 sessions apart, 3
separate periods, and 8 random-order seeds on the same eligible set. A variant is credible only if it
beats BASE at most starts AND in most periods, and beats random ordering. Universe: point-in-time top
1000 by turnover from the full EQ list (no industry filter -- that filter brought survivorship back in
the earlier studies). Market cap = adjusted close x today's share count (Screener), ranked each day.
Costs as in simulate(); delisted companies are still missing (flatters every variant equally).

Run: python entry_filter_portfolio_backtest.py   (Actions: ranking_rotation_backtest.yml, script=entry_filter)
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
OUT_CSV = "analysis/entry_filter_portfolio_backtest.csv"
LAG_DAYS = 92


def eps_panel(dates, tickers):
    """Latest published annual EPS per day (period end + 92 days), from the Screener files."""
    files = ["data/fundamentals_annual.csv.gz", "data/fundamentals_annual_extra.csv.gz"]
    A = pd.concat([pd.read_csv(f, parse_dates=["period"]) for f in files if os.path.exists(f)])
    E = A[A.field == "eps"].drop_duplicates(["ticker", "period"], keep="last")
    cols = {}
    for t, g in E.groupby("ticker"):
        if t not in tickers:
            continue
        s = g.set_index(g.period + pd.Timedelta(days=LAG_DAYS)).value.sort_index()
        s = s[~s.index.duplicated(keep="last")]
        cols[t] = s.reindex(dates, method="ffill").values
    top = A[A.field.isin(["mcap_cr", "price_scr"])].pivot_table(index="ticker", columns="field", values="value",
                                                                 aggfunc="last")
    shares = (top.get("mcap_cr") * 1e7 / top.get("price_scr")).where(top.get("price_scr") > 0)
    return pd.DataFrame(cols, index=dates).reindex(columns=tickers), shares.reindex(tickers)


def main():
    from breakout_sector_study import fetch_ohlcv, since_last
    from exit_rule_backtest import build_rs_panel, stats
    from momentum_factor_backtest import build_pit_universe, load_candidates
    from ranking_rotation_backtest import DEFAULTS, RS_CAP, build_keys, simulate
    from utils.forensics import indicators
    from utils.yf_safe import safe_history

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int, default=0)
    args = ap.parse_args()
    pd.set_option("display.width", 240)

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

    # fresh 52w / ATH breakout that has held (no close below the breakout-day low since)
    hi52 = close.rolling(252, min_periods=200).max()
    new52 = close.ge(hi52) & hi52.notna()
    ath = (close > close.ffill().cummax().shift(1)) & (close.notna().cumsum() >= 500)
    ev = (new52 & (since_last(new52) >= 20)) | (ath & (since_last(ath) >= 20))
    lvl = low.where(ev).ffill(limit=FRESH)
    held = (close - lvl).rolling(FRESH, min_periods=1).min() >= 0
    fresh_bo = ev.astype(float).rolling(FRESH, min_periods=1).max().astype(bool) & held
    # Bollinger squeeze release (entry_exit_study.py S5 definition)
    bbp = indicators(close, high, low)[5]
    m20, s20 = close.rolling(20, min_periods=15).mean(), close.rolling(20, min_periods=15).std()
    sq = (bbp.rolling(5, min_periods=3).min().shift(1) < 20) & (close > m20 + 2 * s20)
    fresh_sq = sq.astype(float).rolling(FRESH, min_periods=1).max().astype(bool)
    trigger = fresh_bo | fresh_sq

    eps, shares = eps_panel(dates, close.columns)
    loss = eps.le(0)                                   # NaN EPS (no data) stays eligible
    mcap = close * shares
    large = mcap.where(pit).rank(axis=1, ascending=False) <= 100
    print(f"[data] EPS for {eps.notna().any().sum()} / {close.shape[1]} names; share count for "
          f"{shares.notna().sum()}; trigger share of eligible name-days {trigger.where(base.notna()).mean().mean():.1%}")

    rs126 = keys["rs126"]
    keys["E2"] = (rs126 + 1e4 * trigger.astype(float)).where(base.notna())
    rng = np.random.default_rng(20261019)
    for s in range(N_RANDOM):
        keys[f"rand{s}"] = pd.DataFrame(rng.random(base.shape), index=dates, columns=base.columns).where(base.notna())

    zx = ~large & ~loss
    V = {
        "BASE 6m RS":           (dict(key="rs126"), base),
        "E1 trigger required":  (dict(key="rs126"), base.where(trigger)),
        "E2 trigger first":     (dict(key="E2"), base),
        "Z1 no large caps":     (dict(key="rs126"), base.where(~large)),
        "X1 no loss-makers":    (dict(key="rs126"), base.where(~loss)),
        "ZX no large, no loss": (dict(key="rs126"), base.where(zx)),
        "C1 E1+Z1+X1":          (dict(key="rs126"), base.where(trigger & zx)),
        "C2 E2+Z1+X1":          (dict(key="E2"), base.where(zx)),
    }
    R = {f"RANDOM {s}": (dict(key=f"rand{s}"), base) for s in range(N_RANDOM)}
    flags = {"trigger": trigger, "large cap": large, "loss-maker": loss}

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
            share.append(dict(variant=name, buys=len(t), avg_hold_days=round((t.exit_date - t.entry_date).dt.days.mean()),
                              **{f"% buys {k}": round(np.nan_to_num(v.values[ii, jj].astype(float)).mean() * 100, 1)
                                 for k, v in flags.items()}))

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

    print(f"\n{'=' * 120}\nLAST 1/3/5/10 YEARS (nested -- not independent)\n{'=' * 120}")
    w = T[T.test == "window"]
    for m in ("cagr_pct", "total_return_pct", "sharpe", "max_drawdown_pct"):
        if m in w:
            print(f"\n{m}:")
            print(w.pivot_table(index="variant", columns="period", values=m).reindex(list(V))[["1y", "3y", "5y", "10y"]].round(2).to_string())
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
                            median_cagr=round(x.cagr_pct.median(), 1),
                            median_d_cagr=round((x.cagr_pct - b.cagr_pct.reindex(x.index)).median(), 2),
                            median_d_maxdd=round((x.max_drawdown_pct - b.max_drawdown_pct.reindex(x.index)).median(), 1),
                            beats_best_random=f"{(x.sharpe > best.reindex(x.index)).sum()}/{len(x)}",
                            median_sharpe=round(x.sharpe.median(), 2)))
        out.append(dict(variant="BASE 6m RS", beats_base="-", median_d_sharpe=0.0, median_cagr=round(b.cagr_pct.median(), 1),
                        median_d_cagr=0.0, median_d_maxdd=0.0,
                        beats_best_random=f"{(b.sharpe > best.reindex(b.index)).sum()}/{len(b)}",
                        median_sharpe=round(b.sharpe.median(), 2)))
        print(f"\n{'=' * 120}\n{title}  (d_ = variant minus BASE; d_maxdd > 0 = shallower drawdown)\n{'=' * 120}")
        print(pd.DataFrame(out).to_string(index=False))
        print("\nCAGR % by start/period:")
        print(t[~t.variant.str.startswith("RANDOM")].pivot_table(index="period", columns="variant",
                                                                 values="cagr_pct").round(1)[list(V)].to_string())
        print("random-order Sharpe (min / median / max):")
        print(rnd.groupby("period").sharpe.agg(["min", "median", "max"]).round(2).to_string())
    print(f"\nsaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
