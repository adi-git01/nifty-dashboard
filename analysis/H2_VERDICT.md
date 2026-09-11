# H2 — regime-conditional exposure. Result: breadth works, trailing IC does not.

Run 34653237189, 2026-09-11. NSE point-in-time top-1000 by turnover, 2741 days,
2015-08 -> 2026-09. Gates block NEW ENTRIES only; open positions still exit on
the normal rules.

## Headline

`C_breadth >= 45%` is the only gate that beats the always-in baseline on
**Sharpe in all four windows and on drawdown in all four windows**, while also
improving the 10-year return.

| gate | Sharpe better | drawdown better | 10y return |
|---|---|---|---|
| B nifty > MA200      | 1/4 | 4/4 | 429.6% |
| **C breadth >= 45%** | **4/4** | **4/4** | **502.3%** |
| C2 breadth >= 55%    | 3/4 | 4/4 | 282.9% |
| D trailing IC > 0    | 1/4 | 3/4 | 129.1% |
| E MA200 AND IC > 0   | 0/4 | 4/4 | 116.7% |

baseline 10y return 445.0%

## Detail — baseline vs breadth 45%

| window | metric | baseline | breadth>=45% |
|---|---|---|---|
| 10y | return | 445.0% | **502.3%** |
| 10y | CAGR | 18.49% | **19.68%** |
| 10y | max DD | -40.4% | **-27.9%** |
| 10y | Sharpe | 1.03 | **1.18** |
| 10y | days invested | 100% | **56%** |
| 3y  | return | 13.1% | **24.1%** |
| 3y  | vs Nifty | -4.3pp | **+6.7pp** |
| 3y  | max DD | -29.6% | **-22.1%** |

Better return, 12.5pp less drawdown, higher Sharpe, and out of the market
nearly half the time.

## The trailing-IC gate failed

D cut the 10-year return from 445% to 129%. The factor's recent realised IC
does not predict its next IC. This was the hypothesis with the best story and
the most careful implementation (lagged, look-ahead tested by scrambling future
prices) — and it is simply wrong. Combining it with MA200 (E) was worse still.

## Threshold sensitivity is the live risk

45% wins the decade; 55% wins the 3-year drought (34.8% vs 13.1%, DD -13.3% vs
-29.6%) but gives up half the decade (282.9% vs 445.0%). Choosing 55% because
it wins the drought is fitting to the drought. 45% is the defensible pick: it
is the looser of the two AND wins the longest window.

Only two thresholds were tested. A finer sweep would raise the overfitting risk
rather than lower it.

## Open question before deploying

Breadth gating may be a general market-timing overlay rather than something
specific to this strategy — a buy-and-hold index position gated the same way
might improve too. That is not tested here. If it is generic, the gate is still
useful but it is not evidence about the momentum factor.

Survivorship still flatters all levels; delisted names remain absent.
