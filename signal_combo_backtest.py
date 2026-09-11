"""
Signal combinations: CompRS x IAS(turnaround) x earnings buckets, 1/3/5/10y.

What is solid and what is not
-----------------------------
IAS is computed entirely from price and volume — velocity of RS21, liquidity
versus its own 10-day trough, distance off the 52-week low, RS63, and a shock
ratio. Nothing in it needs a fundamentals feed, so it can be rebuilt for the
full ten years and combined with CompRS honestly.

Earnings cannot. The parquet cache carries earningsQuarterlyGrowth on 107
snapshots (~6.5 months), and yfinance serves roughly five quarterly statements
per ticker. Year-on-year growth at a date needs that quarter and the one a year
earlier, so five quarters supports about one YoY observation — not a year of
them, and nowhere near three. --fetch-earnings therefore runs only if asked,
reports per-date coverage, and refuses to score a bucket whose coverage is too
thin to mean anything.

REPORTING LAG — the look-ahead that would otherwise be invisible
----------------------------------------------------------------
yfinance dates a statement by PERIOD END, but the market learns it at the
announcement, weeks later. SEBI allows 45 days for quarterly results. Treating
period-end as the knowledge date would let the backtest trade on numbers nobody
had. Every fundamental value is therefore lagged by REPORTING_LAG_DAYS (60,
deliberately conservative) before it can influence a decision.

    python signal_combo_backtest.py --years 10
    python signal_combo_backtest.py --years 1 --fetch-earnings
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from exit_rule_backtest import build_rs_panel, fetch, simulate, stats, VARIANTS
from momentum_factor_backtest import build_pit_universe, load_candidates, RS_CAP

OUT_DIR = "analysis"
REPORTING_LAG_DAYS = 60


# --------------------------------------------------------------------------
def build_ias_panel(close_df, vol_df, nifty):
    """
    Historical IAS, vectorised, mirroring turnaround_screener._calc_ias_full.
    Every input is price/volume derived, so this is available for the whole
    download rather than only for the period the signal log happens to cover.
    """
    n = nifty["Close"].reindex(close_df.index).ffill()

    def rs(period):
        return ((close_df / close_df.shift(period) - 1) * 100).sub(
            (n / n.shift(period) - 1) * 100, axis=0)

    rs21, rs63 = rs(21), rs(63)
    rs21_d5 = rs21.diff(5)

    liq = close_df * vol_df
    lfl = liq.rolling(5).mean() / liq.rolling(10).min().clip(lower=1.0)
    low252 = close_df.rolling(252, min_periods=50).min()
    off_low = (close_df / low252 - 1) * 100

    vel = (rs21_d5 / 5.0 * 35).clip(0, 35)
    lfls = ((lfl.clip(upper=50) - 1.0) * 15).clip(0, 30)
    price = (off_low / 15.0 * 20).clip(0, 20)
    r63 = pd.DataFrame(np.select(
        [rs63 > 0, rs63 > -5, rs63 > -15],
        [15.0, 10.0, 5.0], default=0.0), index=rs63.index, columns=rs63.columns)

    raw = vel.fillna(0) + lfls.fillna(0) + price.fillna(0) + r63
    max1d = rs21.diff(1).rolling(5).max()
    shock = (max1d / rs21_d5.where(rs21_d5 > 3.0))
    penalised = (raw * 0.60).clip(upper=60.0)
    ias = raw.where(~(shock > 0.75), penalised)
    return ias.where(close_df.notna())



def build_turnaround_mask(close_df, vol_df, nifty):
    """
    The screener's GATES, not just its score.

    The logged IAS values median ~60 while a rebuilt score over all names
    medians ~21, because the log only contains names that already cleared these
    gates and then scored >= 35. Scoring without gating would test "CompRS plus
    high RS velocity" — largely a restatement of CompRS — rather than the
    turnaround cohort the screener actually surfaces, which is defined by being
    20-60% off the 52-week high and building a base.

    vol_quality is omitted: it needs open prices and the bulk download carries
    closes only. That makes this mask slightly LOOSER than the live screener.
    """
    n = nifty["Close"].reindex(close_df.index).ffill()
    rs21 = ((close_df / close_df.shift(21) - 1) * 100).sub(
        (n / n.shift(21) - 1) * 100, axis=0)
    high252 = close_df.rolling(252, min_periods=50).max()
    off_high = (close_df / high252 - 1) * 100
    liq5_cr = (close_df * vol_df).rolling(5).mean() / 1e7
    low10 = close_df.rolling(10).min()

    return ((off_high <= -20.0) & (off_high >= -60.0)
            & (liq5_cr >= 30.0)
            & (rs21 >= -40.0)
            & (close_df > low10)
            & (close_df > close_df.shift(2)))


def fetch_earnings_growth(tickers, dates, max_tickers=500):
    """
    Point-in-time YoY earnings growth, lagged by REPORTING_LAG_DAYS.
    Returns (panel, coverage_series). Best-effort and usually thin — see module
    docstring. Coverage is returned so callers can refuse to score on it.
    """
    import yfinance as yf
    panel = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    got = 0
    for i, t in enumerate(tickers[:max_tickers]):
        try:
            q = yf.Ticker(t).quarterly_income_stmt
            if q is None or q.empty or "Net Income" not in q.index:
                continue
            ni = q.loc["Net Income"].dropna().sort_index()
            if len(ni) < 5:
                continue
            yoy = (ni / ni.shift(4) - 1).dropna()
            if yoy.empty:
                continue
            s = pd.Series(index=dates, dtype=float)
            for period_end, val in yoy.items():
                known = pd.Timestamp(period_end) + pd.Timedelta(days=REPORTING_LAG_DAYS)
                s.loc[s.index >= known] = float(val)
            panel[t] = s
            got += 1
        except Exception:
            continue
        if (i + 1) % 100 == 0:
            print(f"  [earnings] {i+1}/{min(len(tickers), max_tickers)} "
                  f"tried, {got} usable")
    cov = panel.notna().sum(axis=1)
    print(f"  [earnings] {got} tickers with usable YoY; median names covered "
          f"per date: {int(cov.median())}")
    return panel, cov


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--max-tickers", type=int, default=0)
    ap.add_argument("--top-n", type=int, default=1000)
    ap.add_argument("--fetch-earnings", action="store_true")
    ap.add_argument("--min-earnings-cov", type=int, default=150,
                    help="skip earnings variants below this many covered names/date")
    args = ap.parse_args()

    end = datetime.now()
    start = end - timedelta(days=365 * args.years + 400)
    tickers = load_candidates("all", args.max_tickers)
    close_df, vol_df, nifty = fetch(tickers, start.strftime("%Y-%m-%d"),
                                    end.strftime("%Y-%m-%d"))
    dates = close_df.index
    pit = build_pit_universe(close_df, vol_df, args.top_n)
    rs_panel = build_rs_panel(close_df, nifty, dates).where(
        lambda x: x.abs() <= RS_CAP).where(pit)
    ias = build_ias_panel(close_df, vol_df, nifty).where(pit)
    turn = build_turnaround_mask(close_df, vol_df, nifty) & pit
    print(f"[data] {len(dates)} days {dates[0].date()} -> {dates[-1].date()}")
    print(f"[IAS] median score {float(ias.stack().median()):.1f}, "
          f"names with IAS>=60 per day: {int((ias >= 60).sum(axis=1).median())}, "
          f">=80: {int((ias >= 80).sum(axis=1).median())}")
    print(f"[turnaround] names clearing the screener gates per day: "
          f"{int(turn.sum(axis=1).median())}")

    variants = {
        "1_CompRS only (baseline)":   lambda: rs_panel,
        "2_CompRS + IAS >= 35":       lambda: rs_panel.where(ias >= 35),
        "3_CompRS + IAS >= 60":       lambda: rs_panel.where(ias >= 60),
        "4_CompRS + IAS >= 80":       lambda: rs_panel.where(ias >= 80),
        "5_CompRS + IAS < 35 (inv)":  lambda: rs_panel.where(ias < 35),
        "6_CompRS + turnaround gates": lambda: rs_panel.where(turn),
        "7_CompRS + turnaround + IAS>=60": lambda: rs_panel.where(turn & (ias >= 60)),
    }

    eg = None
    if args.fetch_earnings:
        print("\n[earnings] fetching quarterly statements (slow)...")
        eg, cov = fetch_earnings_growth(list(close_df.columns), dates)
        if int(cov.median()) < args.min_earnings_cov:
            print(f"  [earnings] median coverage {int(cov.median())} < "
                  f"{args.min_earnings_cov}; earnings variants SKIPPED as "
                  f"too thin to be meaningful.")
            eg = None
        else:
            for lo in (0, 20, 40, 100):
                variants[f"8_CompRS + EPS growth > {lo}%"] = (
                    lambda l=lo: rs_panel.where(eg > l / 100.0))
            variants["9_CompRS + EPS>20% + IAS>=60"] = (
                lambda: rs_panel.where((eg > 0.20) & (ias >= 60)))
            variants["10_CompRS + EPS>20% + turnaround"] = (
                lambda: rs_panel.where((eg > 0.20) & turn))

    rows = []
    for y in (1, 3, 5, 10):
        cut = dates[-1] - pd.Timedelta(days=365 * y)
        sub = dates[dates >= cut]
        if len(sub) < 60:
            continue
        n0 = nifty["Close"].reindex(sub).ffill()
        nret = float(n0.iloc[-1] / n0.iloc[0] - 1) * 100
        print(f"\n{'=' * 100}\nlast {y}y   (Nifty {nret:+.1f}%)\n{'=' * 100}")
        wr = []
        for name, mk in variants.items():
            panel = mk()
            avg_names = int(panel.loc[sub].notna().sum(axis=1).median())
            curve, trades = simulate(name, VARIANTS["baseline"], close_df,
                                     vol_df, nifty, sub, rs_panel=panel)
            s = stats(curve, trades, name)
            s["window"] = f"last {y}y"
            s["eligible_names_median"] = avg_names
            s["excess_vs_nifty_pp"] = round(s["total_return_pct"] - nret, 1)
            wr.append(s)
        df = pd.DataFrame(wr)[["variant", "total_return_pct", "cagr_pct",
                               "max_drawdown_pct", "sharpe", "trades",
                               "win_rate_pct", "eligible_names_median",
                               "excess_vs_nifty_pp"]]
        print(df.to_string(index=False))
        rows += wr

    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/signal_combo.csv", index=False)
    print(f"\nsaved -> {OUT_DIR}/signal_combo.csv")
    print("\nIAS here is REBUILT from price/volume, so it is available for the whole\n"
          "window rather than only where the live signal log reaches. Earnings\n"
          "variants appear only if --fetch-earnings was passed AND coverage cleared\n"
          f"the {args.min_earnings_cov}-name floor; every fundamental is lagged "
          f"{REPORTING_LAG_DAYS}d behind period end.")


if __name__ == "__main__":
    main()
