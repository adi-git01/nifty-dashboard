"""
Is cheapness mispricing, or information?
========================================

The claim under test (from a podcast, paraphrased): value investing worked
because information was scarce and asymmetric; now that information is
abundant, a stock that trades cheap trades cheap for a good reason.

WHY THIS IS A PRICE-ONLY TEST
-----------------------------
The obvious test is P/B or P/E deciles over ten years. We cannot run it.
data/fundamentals_cache.csv is a SINGLE CURRENT SNAPSHOT — one fund_last_updated
column, today's values — and yfinance serves roughly 5 quarterly and 4 annual
statements per ticker (this is what killed winner_eps_backtest.py at n=2).
Applying today's P/B to 2016 prices is look-ahead plus survivorship and would
produce a confident, meaningless number.

So we use the price-only proxy for cheapness: LONG-HORIZON REVERSAL, the
De Bondt-Thaler (1985) formulation. "Cheap" = has fallen a long way over a
multi-year formation window. This is not a perfect stand-in for book-to-market,
but it is the component of value that does not need a balance sheet, and the
two hypotheses make opposite predictions on it:

    classic value / reversal : long-term losers OUTPERFORM  -> IC negative
    the podcast's claim      : long-term losers KEEP LOSING -> IC positive

A null result (IC ~ 0) supports neither and is the most likely outcome given
everything else this repo has tested.

THE BIAS, AND WHY IT MAKES THE TEST READABLE ANYWAY
---------------------------------------------------
Delisting survivorship is unfixable here — companies that went to zero or left
the exchange are absent from EQUITY_L and from every file in this repo. That
bias falls HARDEST on exactly the bucket under test: long-term losers are the
names that delist. Excluding them flatters the loser bucket.

That asymmetry is useful. It means:
  - if losers still UNDERPERFORM, the result is strong (the bias worked against
    it and it survived anyway) -> evidence FOR the podcast's claim
  - if losers OUTPERFORM, the result is suspect and must not be believed
    without a delisting-complete dataset

THE INDIA SPLIT
---------------
The "information abundance" evidence is overwhelmingly US large cap. In India
analyst coverage is extremely uneven: Nifty 50 names carry 30+ analysts, names
below roughly Rs 5,000 cr often carry 0-2. If the podcast's mechanism is real,
the effect should be STRONGER where coverage is dense. So every test is also
run split by liquidity tier — top 200 by trailing turnover against 201-1000.
That split is the part of this test that actually discriminates between "value
is dead everywhere" and "value is dead where everyone is looking".

Run: python value_reversal_backtest.py --years 10 --formation 756
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from momentum_factor_backtest import build_pit_universe, load_candidates, shuffle_panel

SKIP = 21              # skip the most recent month so short-term reversal does
                       # not contaminate the multi-year formation signal
MIN_NAMES = 50         # minimum cross-section on a date to compute an IC


def fetch_prices(tickers, start, end):
    """
    Closes and volumes only.

    Deliberately does NOT fetch ^NSEI. exit_rule_backtest.fetch() aborts the
    whole run if the index download fails, and this test never touches the
    index — every measure here is cross-sectional (a name against the other
    names on the same date), so a benchmark series would be dead weight and one
    more thing to fail on.
    """
    from utils.yf_safe import safe_download

    print(f"[data] {len(tickers)} tickers (this is the slow part)...")
    bulk = safe_download(tickers, start=start, end=end, group_by="ticker",
                         threads=False, auto_adjust=True, min_coverage=0.5)
    if bulk is None or bulk.empty:
        raise SystemExit("Bulk download failed — aborting.")

    closes, vols = {}, {}
    multi = isinstance(bulk.columns, pd.MultiIndex)
    for t in tickers:
        try:
            sub = bulk[t] if multi else bulk
            c = sub["Close"].dropna()
            if len(c) > 250:
                closes[t] = c
                vols[t] = sub["Volume"].reindex(c.index)
        except Exception:
            continue
    close_df, vol_df = pd.DataFrame(closes), pd.DataFrame(vols)
    if close_df.index.tz is not None:
        close_df.index = close_df.index.tz_localize(None)
        vol_df.index = vol_df.index.tz_localize(None)
    print(f"[data] usable tickers: {close_df.shape[1]}")
    return close_df, vol_df


def formation_signal(close_df: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    Trailing return over `days`, ending SKIP days ago.

    Ranked ascending this is a cheapness proxy: the most negative value is the
    name that has fallen furthest and therefore screens cheapest on any
    price-anchored multiple.
    """
    past = close_df.shift(SKIP)
    return past / past.shift(days) - 1.0


def ic_table(sig: pd.DataFrame, fwd: pd.DataFrame, horizon: int, label: str) -> dict:
    """
    Spearman IC of the formation return against the forward return.

    Sign convention, stated once because it is the whole result:
        IC < 0  losers outperform    -> reversal, value alive
        IC > 0  losers keep losing   -> the podcast's claim
    """
    ics, dates = [], []
    for d in sig.index:
        a, b = sig.loc[d], fwd.loc[d]
        m = a.notna() & b.notna()
        if m.sum() > MIN_NAMES:
            ics.append(spearmanr(a[m], b[m])[0])
            dates.append(d)
    if not ics:
        return {}
    s = pd.Series(ics, index=dates)
    nono = s.iloc[::horizon]          # non-overlapping: consecutive forward
                                      # windows share horizon-1 days otherwise
    se = nono.std(ddof=1) / np.sqrt(len(nono)) if len(nono) > 1 else np.nan
    return {
        "window": label,
        "mean_IC": round(float(s.mean()), 4),
        "pct_neg": round(float((s < 0).mean()) * 100, 1),
        "IC_nonoverlap": round(float(nono.mean()), 4),
        "n_indep": int(len(nono)),
        "t_indep": round(float(nono.mean() / se), 2) if se and se == se else np.nan,
    }


def quintiles(sig: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Median forward return by formation quintile. Q1 = cheapest (worst past)."""
    rows = []
    labels = ["Q1 cheapest", "Q2", "Q3", "Q4", "Q5 dearest"]
    for d in sig.index:
        a, b = sig.loc[d], fwd.loc[d]
        m = a.notna() & b.notna()
        if m.sum() < MIN_NAMES:
            continue
        try:
            q = pd.qcut(a[m], 5, labels=labels, duplicates="drop")
        except ValueError:
            continue
        rows.append(b[m].groupby(q, observed=True).median() * 100)
    return pd.DataFrame(rows).mean() if rows else pd.Series(dtype=float)


def liquidity_tier(close_df, vol_df, lo: int, hi: int, lookback: int = 60):
    """Boolean mask: names ranked lo..hi by trailing median turnover, per date."""
    turn = (close_df * vol_df).rolling(lookback, min_periods=lookback // 2).median()
    rank = turn.rank(axis=1, ascending=False, method="first")
    return (rank > lo) & (rank <= hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=10, help="length of the TEST window")
    ap.add_argument("--formation", type=int, default=756,
                    help="formation window in trading days (756 = 3y, 1260 = 5y)")
    ap.add_argument("--max-tickers", type=int, default=0)
    ap.add_argument("--source", default="all", choices=["all", "nifty1000", "nifty500"])
    ap.add_argument("--top-n", type=int, default=1000)
    ap.add_argument("--random-seeds", type=int, default=20)
    args = ap.parse_args()

    # Need formation history BEFORE the test window opens, or the first years of
    # the test have no signal at all.
    warmup_days = args.formation + SKIP + 60
    end = datetime.now()
    start = end - timedelta(days=int(365 * args.years + warmup_days * 1.45) + 60)

    tickers = load_candidates(args.source, args.max_tickers)
    print(f"[universe] {len(tickers)} candidates; formation {args.formation}d "
          f"(~{args.formation/252:.1f}y), skip {SKIP}d, PIT top {args.top_n}")

    close_df, vol_df = fetch_prices(tickers, start.strftime("%Y-%m-%d"),
                                    end.strftime("%Y-%m-%d"))
    dates = close_df.index
    print(f"[data] {len(dates)} trading days {dates[0].date()} -> {dates[-1].date()}")

    sig = formation_signal(close_df, args.formation)
    pit = build_pit_universe(close_df, vol_df, args.top_n)
    sig = sig.where(pit)
    print(f"[universe] median members/day {int(pit.sum(axis=1).median())}; "
          f"names with a usable formation signal on the last date: "
          f"{int(sig.iloc[-1].notna().sum())}")

    tiers = {
        "ALL (top 1000)":      pit,
        "LARGE (top 200)":     liquidity_tier(close_df, vol_df, 0, 200),
        "SMALL/MID (201-1000)": liquidity_tier(close_df, vol_df, 200, 1000),
    }

    for horizon in (63, 252):
        fwd = close_df.shift(-horizon) / close_df - 1.0
        cut = dates[-1] - pd.Timedelta(days=365 * args.years)
        m = (dates >= cut) & (dates <= dates[-horizon - 1])
        if m.sum() < 60:
            print(f"\n[skip] {horizon}d horizon: only {m.sum()} usable dates")
            continue

        print("\n" + "=" * 92)
        print(f"FORWARD {horizon}d  —  IC<0 means losers outperform (value alive); "
              f"IC>0 means cheap stays cheap")
        print("=" * 92)

        rows, qrows = [], {}
        for name, mask in tiers.items():
            s = sig.where(mask).loc[m]
            r = ic_table(s, fwd.loc[m], horizon, name)
            if r:
                rows.append(r)
                qrows[name] = quintiles(s, fwd.loc[m])
        if rows:
            print(pd.DataFrame(rows).to_string(index=False))

        if qrows:
            print(f"\nMedian forward {horizon}d return by formation quintile (%), "
                  f"Q1 = fallen furthest:")
            print(pd.DataFrame(qrows).T.to_string())

        # Random control: same values, same NaN mask, reassigned across names.
        # Tells us what |IC| this cross-section produces by construction.
        if args.random_seeds:
            ctl = []
            for seed in range(args.random_seeds):
                r = ic_table(shuffle_panel(sig.loc[m], seed), fwd.loc[m],
                             horizon, f"seed{seed}")
                if r:
                    ctl.append(r["IC_nonoverlap"])
            if ctl:
                real = next((r["IC_nonoverlap"] for r in rows
                             if r["window"] == "ALL (top 1000)"), np.nan)
                beat = float(np.mean([abs(real) > abs(c) for c in ctl])) * 100
                print(f"\nrandom control ({len(ctl)} seeds): mean |IC| "
                      f"{np.mean(np.abs(ctl)):.4f}, real |IC| {abs(real):.4f} "
                      f"-> larger than {beat:.0f}% of random rankings")

    print("\n" + "=" * 92)
    print("READING THIS RESULT")
    print("=" * 92)
    print("Delisting survivorship flatters the LOSER bucket specifically — the "
          "names that\nfell furthest are the ones most likely to have left the "
          "exchange, and they are\nabsent from this data. So:")
    print("  losers UNDERPERFORM  -> strong, the bias worked against it "
          "(supports 'cheap for a reason')")
    print("  losers OUTPERFORM    -> suspect, do not believe without "
          "delisting-complete data")
    print("  IC ~ 0               -> neither claim supported; the most likely "
          "outcome")
    print("\nThis is a price-only proxy for cheapness. It cannot separate 'value "
          "is dead'\nfrom 'book value stopped measuring capital' — that needs "
          "point-in-time\nfundamentals this repo does not have.")


if __name__ == "__main__":
    main()
