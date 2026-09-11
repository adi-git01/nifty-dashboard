"""
Does CompRS actually predict? Nifty 500, over 1 / 3 / 5 / 10 years.
===================================================================

Every finding so far came from ~5 months of daily parquets sitting almost
entirely in one CAUTION regime, with 3 independent 21-day windows. That is
enough to raise a question and nowhere near enough to answer it. This runs the
same question over real multi-regime history.

Two measurements, deliberately separate:

  SIGNAL  — does CompRS rank forward returns? Spearman IC and quintile spreads,
            reported per calendar year so a dead factor in one regime cannot be
            hidden by a live one in another.
  MONEY   — what the strategy's own rules actually returned over each window,
            using the existing simulate() so entries, exits, sizing and costs
            match the live engine.

KNOWN BIAS — read before using any number here
----------------------------------------------
data/nifty500_list.csv holds TODAY's constituents. Running them back 10 years
is survivorship-biased: names that were delisted, demoted or collapsed are
absent, so the level of every return is flattered. This hits MONEY hardest.
It hits SIGNAL less, because a cross-sectional rank correlation is computed
among the survivors on each date, but it is not immune — the losers that would
have dragged the low-RS quintile are missing too. Treat year-by-year *changes*
in IC as the trustworthy part, and absolute CAGR as an upper bound.

Overlapping windows: consecutive daily observations of a 21-day forward return
share 20 of 21 days, so naive t-stats are inflated ~sqrt(21). Every IC is
therefore reported with a non-overlapping sample as well.

    python momentum_factor_backtest.py --years 10 --max-tickers 500
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from exit_rule_backtest import (build_rs_panel, fetch, simulate, stats,
                                VARIANTS, INITIAL_CAPITAL)

OUT_DIR = "analysis"
HORIZON = 21          # forward-return horizon for the signal test
RS_CAP = 200.0        # corrupt-bar guard, same as the dashboard


def load_candidates(source: str, max_tickers: int) -> list:
    """
    The pool we DOWNLOAD. Selection of which names are actually in the universe
    on a given date happens later, in build_pit_universe().

    Default is the full NSE EQ list rather than nifty500_list.csv for two
    reasons. First, the live strategy does not trade the Nifty 500 — it trades
    the top 1000 by market cap out of the whole EQ list, so backtesting the
    Nifty 500 measures a universe you do not trade. Second, today's index
    membership is itself a survivorship filter: a name that was in the Nifty
    500 in 2019 and got demoted for performing badly is absent from the file,
    which is exactly the cohort a momentum test must not drop.
    """
    if source == "nifty500":
        df = pd.read_csv("data/nifty500_list.csv")
        col = "Ticker" if "Ticker" in df.columns else df.columns[0]
        t = [str(x).strip() for x in df[col].dropna().unique()]
    elif source == "nifty1000":
        t = [str(x).strip() for x in pd.read_csv("data/nifty1000_list.csv")["Ticker"].dropna()]
    else:  # "all" — every live NSE EQ symbol
        e = pd.read_csv("EQUITY_L.csv")
        e.columns = [c.strip() for c in e.columns]
        e["SERIES"] = e["SERIES"].astype(str).str.strip()
        t = [str(x).strip() for x in e[e["SERIES"] == "EQ"]["SYMBOL"].dropna().unique()]
    t = [x if x.endswith((".NS", ".BO")) else x + ".NS" for x in t]
    return t[:max_tickers] if max_tickers else t


def build_pit_universe(close_df, vol_df, top_n: int, lookback: int = 60):
    """
    Point-in-time universe: on each date, the top `top_n` names by trailing
    median turnover. Returns a boolean mask (dates x tickers).

    This mirrors the live universe rule — build_nifty1000_ci.py ranks the whole
    EQ list and keeps the top 1000 — but applies it AS OF each date instead of
    once, today. A stock that was liquid in 2019 and is illiquid now is in the
    2019 universe and out of the current one, which is what the strategy would
    actually have faced. Turnover stands in for market cap because shares
    outstanding are not available historically, and it doubles as the same
    liquidity notion the live entry gate already uses.

    WHAT THIS DOES NOT FIX: delisting. Companies that went to zero or were
    taken off the exchange are absent from EQUITY_L and from every other file
    in this repo, so no reconstruction here can include them. The residual bias
    is therefore one-directional — returns are still flattered — and the only
    honest response is to say so rather than to imply the universe is clean.
    """
    turn = (close_df * vol_df).rolling(lookback, min_periods=lookback // 2).median()
    rank = turn.rank(axis=1, ascending=False, method="first")
    mask = rank <= top_n
    return mask & close_df.notna()


def ic_table(rs: pd.DataFrame, fwd: pd.DataFrame, label: str) -> dict:
    """Spearman IC of CompRS vs forward return, overlapping and not."""
    ics, dates = [], []
    for d in rs.index:
        a, b = rs.loc[d], fwd.loc[d]
        m = a.notna() & b.notna()
        if m.sum() > 50:
            ics.append(spearmanr(a[m], b[m])[0])
            dates.append(d)
    if not ics:
        return {}
    s = pd.Series(ics, index=dates)
    nono = s.iloc[::HORIZON]
    se = nono.std(ddof=1) / np.sqrt(len(nono)) if len(nono) > 1 else np.nan
    return {
        "window": label,
        "mean_IC": round(float(s.mean()), 4),
        "pct_positive": round(float((s > 0).mean()) * 100, 1),
        "n_days": int(len(s)),
        "IC_nonoverlap": round(float(nono.mean()), 4),
        "n_indep": int(len(nono)),
        "t_indep": round(float(nono.mean() / se), 2) if se and se == se else np.nan,
    }


def quintiles(rs: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    rows = []
    for d in rs.index:
        a, b = rs.loc[d], fwd.loc[d]
        m = a.notna() & b.notna()
        if m.sum() < 50:
            continue
        try:
            q = pd.qcut(a[m], 5, labels=["Q1 weak", "Q2", "Q3", "Q4", "Q5 strong"],
                        duplicates="drop")
        except ValueError:
            continue
        rows.append(b[m].groupby(q, observed=True).median() * 100)
    return pd.DataFrame(rows).mean() if rows else pd.Series(dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--max-tickers", type=int, default=0,
                    help="cap on names DOWNLOADED (0 = all)")
    ap.add_argument("--source", default="all",
                    choices=["all", "nifty1000", "nifty500"],
                    help="candidate pool to download (default: whole NSE EQ list)")
    ap.add_argument("--top-n", type=int, default=1000,
                    help="point-in-time universe size, by trailing turnover "
                         "(matches the live top-1000 rule)")
    args = ap.parse_args()

    end = datetime.now()
    start = end - timedelta(days=365 * args.years + 400)   # +400 to warm up RS63/MA200
    tickers = load_candidates(args.source, args.max_tickers)
    print(f"[universe] downloading {len(tickers)} candidates from '{args.source}'; "
          f"point-in-time universe = top {args.top_n} by trailing turnover")

    close_df, vol_df, nifty = fetch(tickers, start.strftime("%Y-%m-%d"),
                                    end.strftime("%Y-%m-%d"))
    dates = close_df.index
    rs_panel = build_rs_panel(close_df, nifty, dates).where(
        lambda x: x.abs() <= RS_CAP)

    # Point-in-time universe. simulate() gates eligibility on rs_panel.notna(),
    # so masking here excludes out-of-universe names from BOTH the signal test
    # and the money test without touching the simulator.
    pit = build_pit_universe(close_df, vol_df, args.top_n)
    rs_panel = rs_panel.where(pit)
    sizes = pit.sum(axis=1)
    churn = (pit.astype(int).diff().abs().sum(axis=1) / 2).rolling(252).sum()
    print(f"[universe] members/day: min {int(sizes.min())} median {int(sizes.median())} "
          f"max {int(sizes.max())} | median names entering-or-leaving per year: "
          f"{int(churn.median()) if churn.notna().any() else 0}")
    print(f"[universe] a static list would have frozen today's members across all "
          f"{len(dates)} days instead.")
    fwd = close_df.shift(-HORIZON) / close_df - 1
    print(f"[data] {len(dates)} trading days {dates[0].date()} -> {dates[-1].date()}")

    # ---------------- SIGNAL ----------------
    print("\n" + "=" * 78)
    print(f"SIGNAL — does CompRS rank forward {HORIZON}d returns?")
    print("=" * 78)
    rows, qrows = [], {}
    for y in (1, 3, 5, 10):
        cut = dates[-1] - pd.Timedelta(days=365 * y)
        m = (dates >= cut) & (dates <= dates[-HORIZON - 1])
        if m.sum() < 60:
            continue
        r = ic_table(rs_panel.loc[m], fwd.loc[m], f"last {y}y")
        if r:
            rows.append(r)
            qrows[f"last {y}y"] = quintiles(rs_panel.loc[m], fwd.loc[m])
    ic_df = pd.DataFrame(rows)
    print(ic_df.to_string(index=False))

    print(f"\nmedian {HORIZON}d forward return by CompRS quintile (%)")
    print(pd.DataFrame(qrows).round(2).to_string())
    qd = pd.DataFrame(qrows)
    if not qd.empty:
        print("\nQ5 - Q1 spread (pp): " +
              "  ".join(f"{k}: {qd[k].iloc[-1] - qd[k].iloc[0]:+.2f}" for k in qd.columns))

    print("\n" + "=" * 78)
    print("SIGNAL BY CALENDAR YEAR — the part survivorship bias distorts least")
    print("=" * 78)
    yrows = []
    for yr in sorted(set(dates.year)):
        m = (dates.year == yr) & (dates <= dates[-HORIZON - 1])
        if m.sum() < 60:
            continue
        r = ic_table(rs_panel.loc[m], fwd.loc[m], str(yr))
        if r:
            q = quintiles(rs_panel.loc[m], fwd.loc[m])
            r["Q5_minus_Q1_pp"] = (round(float(q.iloc[-1] - q.iloc[0]), 2)
                                   if len(q) >= 2 else np.nan)
            yrows.append(r)
    year_df = pd.DataFrame(yrows)
    print(year_df.to_string(index=False))

    # ---------------- MONEY ----------------
    print("\n" + "=" * 78)
    print("MONEY — the strategy's own rules (baseline variant) per window")
    print("=" * 78)
    mrows = []
    for y in (1, 3, 5, 10):
        cut = dates[-1] - pd.Timedelta(days=365 * y)
        sub = dates[dates >= cut]
        if len(sub) < 60:
            continue
        curve, trades = simulate("baseline", VARIANTS["baseline"],
                                 close_df, vol_df, nifty, sub, rs_panel=rs_panel)
        s = stats(curve, trades, f"last {y}y")
        n0 = nifty["Close"].reindex(sub).ffill()
        s["nifty_return_pct"] = round(float(n0.iloc[-1] / n0.iloc[0] - 1) * 100, 1)
        s["excess_vs_nifty_pp"] = round(s["total_return_pct"] - s["nifty_return_pct"], 1)
        mrows.append(s)
    money_df = pd.DataFrame(mrows)
    print(money_df.to_string(index=False))

    os.makedirs(OUT_DIR, exist_ok=True)
    ic_df.to_csv(f"{OUT_DIR}/factor_ic_by_window.csv", index=False)
    year_df.to_csv(f"{OUT_DIR}/factor_ic_by_year.csv", index=False)
    money_df.to_csv(f"{OUT_DIR}/factor_money_by_window.csv", index=False)
    qd.to_csv(f"{OUT_DIR}/factor_quintiles.csv")
    print(f"\nsaved -> {OUT_DIR}/factor_*.csv")
    print("\nREMINDER: the universe is point-in-time by turnover, so index-membership\n"
          "survivorship is handled — but DELISTED companies are absent from EQUITY_L and\n"
          "cannot be recovered from any file in this repo. Returns remain flattered by an\n"
          "unknown amount; year-over-year IC changes stay the most trustworthy output.")


if __name__ == "__main__":
    main()
