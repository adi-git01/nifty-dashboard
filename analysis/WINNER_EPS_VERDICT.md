# Winner EPS Trajectory — NOT ANSWERABLE WITH THIS DATA

Run 34655152460 (winner_eps_backtest.py, 10y, full point-in-time universe).

Question: for trades that more than doubled, does net income one quarter
before entry differ from net income at exit, relative to losing trades?

## Verdict: no answer. The doubles bucket has n = 2.

1001 closed trades 2016-09-12 -> 2026-09-03. Only 144 (14%) have a usable
before/after statement pair, and they are concentrated in recent years.

| bucket             | trades | with EPS pair |
|--------------------|--------|---------------|
| A >=100% (doubled) | 24     | 2             |
| B 25-100%          | 90     | 10            |
| C 0-25%            | 245    | 38            |
| D negative         | 642    | 94            |

MIN_BUCKET = 8, so the script refused to interpret bucket A. Correct call.

## The readable buckets are confounded by holding period

Growth is measured ACROSS THE HOLD, and winners are held roughly 4x longer:

| bucket    | n  | median growth | median held | growth per day held |
|-----------|----|---------------|-------------|---------------------|
| B 25-100% | 10 | 66.7%         | 94.5d       | 0.71%/d             |
| C 0-25%   | 38 | 49.7%         | 38.5d       | 1.29%/d             |
| D negative| 94 | 22.3%         | 24.5d       | 0.91%/d             |

The apparent monotone gradient (67 > 50 > 22) disappears and partially
inverts once normalised for time. This is the same variable-horizon artifact
that invalidated the first IAS result (return_since_signal): the denominator,
not the signal, was doing the work. C's mean of 632.2% against a median of
49.7% is a separate small-denominator base-effect outlier.

## Why more runtime will not fix it

yfinance serves ~5 quarterly and ~4 annual statements per ticker. Trades
before roughly 2022 have no statements at all. The 14% coverage is structural.

To answer this properly:
  1. Fixed-window comparison — TTM net income at entry-60d vs the same four
     quarters later, identical horizon for every trade, so holding period
     cannot leak into the measurement.
  2. A statement source with a decade of history (Screener.in, exchange
     filings archive). yfinance cannot support the question.
