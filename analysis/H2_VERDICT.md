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

---

## Addendum — breadth control on plain Nifty (run 34655143574)

The gate was re-run with the same breadth rule and switching costs applied to
Nifty buy-and-hold, to separate "breadth helps the momentum book" from
"breadth is generic market timing that would help anything".

Pre-registered criterion: if gated Nifty improves as much as the book did,
the gate is generic. It does not — it moves the opposite way.

Total return, always-in vs breadth >= 45%:

| window | momentum A -> C | Nifty B&H -> same gate |
|--------|-----------------|------------------------|
| 1y     | 26.1  -> 23.6   | -6.1  -> -8.7          |
| 3y     | 13.1  -> 24.1   | 17.3  -> -0.1          |
| 5y     | 148.2 -> 142.7  | 35.1  -> 1.3           |
| 10y    | 445.0 -> 502.3  | 169.1 -> 113.7         |

Difference-in-differences in CAGR (gate's effect on the book minus its effect
on the index): +0.1pp (1y), +8.8pp (3y), +5.4pp (5y), +3.7pp (10y). Positive
in all four windows.

Mechanism: the gate is invested ~56% of days. An index compounds with
time-in-market, so sitting out costs it a third of its return. The book sits
out the same days and returns more, which means those days are ones where
momentum entries specifically lose beyond the market's average day.

What IS generic: drawdown. Nifty -38.4 -> -15.1, book -40.4 -> -27.9. Any
reduction in time-in-market buys this. Do not credit the factor for it.

CAUTION — the threshold is fitted. The return benefit is not monotone in
strictness: at 10y the 45% gate returns 502% while 55% returns 283% (worse
than always-in); at 3y the ranking flips (55% -> 34.8 vs 45% -> 24.1). The
best threshold is window-dependent.

Stable across both thresholds and all four windows: drawdown (better 4/4) and
Sharpe (C beats A 4/4 — 1.52->2.29, 0.33->0.60, 1.20->1.31, 1.03->1.18).

Adopt breadth as a risk overlay at a 45% floor. Expect the drawdown and
Sharpe benefit. Do not budget for the +57pp of extra 10-year return.
