"""
Do the big winners show earnings growth DURING the hold?

Takes every trade the 10-year simulation produced, buckets them by outcome, and
compares net income one quarter BEFORE entry against net income at exit. The
article's central claim is that durable momentum is an under-reaction to
improving fundamentals; if that is true, the doubles should show materially
stronger earnings growth across the hold than the losers do.

WHY A CONTROL GROUP IS THE WHOLE POINT
--------------------------------------
"Winners had rising EPS" means nothing on its own — in a decade where the index
tripled, most surviving companies grew earnings. The test is winners VERSUS
losers over the same period. Only the spread is evidence.

WHAT IS AND IS NOT TESTABLE
---------------------------
This needs ~2 earnings observations per traded ticker, not a full daily panel
for 1000 names — so it is far more tractable than the earlier bucket tests.
But yfinance still serves only ~5 quarterly statements and ~4 annual ones per
ticker, so:

  * trades that CLOSED within roughly the last year   -> quarterly granularity
  * trades that closed within roughly the last 4 years -> annual granularity
  * trades older than that                             -> NOT COVERED

The run reports coverage per bucket and refuses to compare buckets that fall
below a minimum count. A 10-year claim cannot be made from this; a 3-4 year one
can, with annual granularity.

Reporting lag: yfinance dates statements by PERIOD END while the market learns
them at announcement, so a statement is only treated as known
REPORTING_LAG_DAYS after its period end.

    python winner_eps_backtest.py --years 10
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from exit_rule_backtest import build_rs_panel, fetch, simulate, VARIANTS
from momentum_factor_backtest import build_pit_universe, load_candidates, RS_CAP

OUT_DIR = "analysis"
REPORTING_LAG_DAYS = 60
MIN_BUCKET = 8


def net_income_series(ticker):
    """(quarterly, annual) net-income Series indexed by period end, or (None, None)."""
    import yfinance as yf
    q = a = None
    try:
        t = yf.Ticker(ticker)
        qi = t.quarterly_income_stmt
        if qi is not None and not qi.empty and "Net Income" in qi.index:
            q = qi.loc["Net Income"].dropna().sort_index()
        ai = t.income_stmt
        if ai is not None and not ai.empty and "Net Income" in ai.index:
            a = ai.loc["Net Income"].dropna().sort_index()
    except Exception:
        pass
    return q, a


def value_known_at(series, when, lag_days=REPORTING_LAG_DAYS):
    """Most recent figure whose period end + lag is on or before `when`."""
    if series is None or series.empty:
        return None, None
    known = series[[pd.Timestamp(i) + pd.Timedelta(days=lag_days) <= pd.Timestamp(when)
                    for i in series.index]]
    if known.empty:
        return None, None
    return float(known.iloc[-1]), pd.Timestamp(known.index[-1])


def bucket_of(pnl_pct):
    if pnl_pct >= 100:
        return "A >=100% (doubled)"
    if pnl_pct >= 25:
        return "B 25-100%"
    if pnl_pct >= 0:
        return "C 0-25%"
    return "D negative"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--max-tickers", type=int, default=0)
    ap.add_argument("--top-n", type=int, default=1000)
    args = ap.parse_args()

    end = datetime.now()
    start = end - timedelta(days=365 * args.years + 400)
    tickers = load_candidates("all", args.max_tickers)
    close_df, vol_df, nifty = fetch(tickers, start.strftime("%Y-%m-%d"),
                                    end.strftime("%Y-%m-%d"))
    dates = close_df.index
    rs = build_rs_panel(close_df, nifty, dates).where(
        lambda x: x.abs() <= RS_CAP).where(build_pit_universe(close_df, vol_df, args.top_n))

    cut = dates[-1] - pd.Timedelta(days=365 * args.years)
    sub = dates[dates >= cut]
    _, trades = simulate("baseline", VARIANTS["baseline"], close_df, vol_df,
                         nifty, sub, rs_panel=rs)
    tr = pd.DataFrame(trades)
    if tr.empty:
        print("no trades"); return
    tr["bucket"] = tr["pnl_pct"].apply(bucket_of)
    print(f"[trades] {len(tr)} closed trades {tr.entry_date.min().date()} -> "
          f"{tr.exit_date.max().date()}")
    print(tr.groupby("bucket").agg(n=("pnl_pct", "size"),
                                   median_pnl=("pnl_pct", "median")).round(1).to_string())

    uniq = sorted(tr.ticker.unique())
    print(f"\n[earnings] fetching statements for {len(uniq)} traded tickers (slow)...")
    qmap, amap = {}, {}
    for i, t in enumerate(uniq):
        q, a = net_income_series(t)
        qmap[t], amap[t] = q, a
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(uniq)}")

    rows = []
    for r in tr.itertuples(index=False):
        pre_date = pd.Timestamp(r.entry_date) - pd.Timedelta(days=90)
        for gran, m in (("quarterly", qmap), ("annual", amap)):
            pre, pre_pe = value_known_at(m.get(r.ticker), pre_date)
            post, post_pe = value_known_at(m.get(r.ticker), r.exit_date)
            if pre is None or post is None or pre_pe == post_pe or pre == 0:
                continue
            rows.append({"ticker": r.ticker, "bucket": r.bucket,
                         "pnl_pct": r.pnl_pct, "granularity": gran,
                         "ni_pre": pre, "ni_post": post,
                         "ni_growth_pct": (post / abs(pre) - 1) * 100 if pre else np.nan,
                         "held_days": (pd.Timestamp(r.exit_date)
                                       - pd.Timestamp(r.entry_date)).days})
            break
    res = pd.DataFrame(rows)
    if res.empty:
        print("\nNO COVERAGE — yfinance returned no usable statement pairs. "
              "Nothing can be concluded.")
        return

    print(f"\n[coverage] {len(res)}/{len(tr)} trades have a usable before/after pair "
          f"({len(res)/len(tr)*100:.0f}%)")
    print(res.groupby(["bucket", "granularity"]).size().to_string())

    print("\n" + "=" * 80)
    print("NET INCOME GROWTH ACROSS THE HOLD, BY TRADE OUTCOME")
    print("=" * 80)
    g = res.groupby("bucket").agg(n=("ni_growth_pct", "size"),
                                  median_growth=("ni_growth_pct", "median"),
                                  mean_growth=("ni_growth_pct", "mean"),
                                  median_pnl=("pnl_pct", "median"),
                                  median_held=("held_days", "median"))
    thin = g[g.n < MIN_BUCKET].index.tolist()
    print(g.round(1).to_string())
    if thin:
        print(f"\nbuckets below {MIN_BUCKET} trades — NOT interpretable: {thin}")
    win, lose = "A >=100% (doubled)", "D negative"
    if win in g.index and lose in g.index and g.loc[win, "n"] >= MIN_BUCKET \
            and g.loc[lose, "n"] >= MIN_BUCKET:
        from scipy.stats import mannwhitneyu
        a = res[res.bucket == win]["ni_growth_pct"]
        b = res[res.bucket == lose]["ni_growth_pct"]
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        print(f"\ndoubles vs losers: median {a.median():+.1f}% vs {b.median():+.1f}%, "
              f"Mann-Whitney p={p:.3f}")
        print("  -> the article predicts doubles show materially stronger growth")
    os.makedirs(OUT_DIR, exist_ok=True)
    res.to_csv(f"{OUT_DIR}/winner_eps.csv", index=False)
    print(f"\nsaved -> {OUT_DIR}/winner_eps.csv")
    print(f"\nCOVERAGE CAVEAT: yfinance serves ~5 quarterly and ~4 annual statements "
          f"per ticker,\nso older trades are simply absent. Read the coverage table "
          f"before the result —\nif the doubles bucket is thin or skewed to recent "
          f"years, the comparison is not\nabout the decade, it is about whatever "
          f"period the data happens to reach.")


if __name__ == "__main__":
    main()
