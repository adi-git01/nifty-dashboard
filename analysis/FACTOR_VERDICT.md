# Does CompRS work? — Nifty/NSE point-in-time, 10 years

Run 34650219561, 2026-09-11. 2110 candidates downloaded, 2015 usable, universe =
top 1000 by trailing 60-day turnover recomputed daily (median 1000 members,
~536 names entering or leaving per year). 2741 trading days, 2015-08 -> 2026-09.

## Correction to the previous run

Run #1 reported IC **-0.028, t=-2.43** over 10 years and I called momentum
structurally broken. That run was capped at 500 downloads which, against the
alphabetically-sorted EQ list, took `20MICRONS..DIVISLAB` and excluded
RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, ITC and LT. The conclusion did
not survive a correct universe. It is withdrawn.

## SIGNAL — the factor is not negative, it is ~zero

| window | mean IC | % positive | indep windows | t | Q5-Q1 |
|---|---|---|---|---|---|
| 1y  | +0.0357 | 69.3% | 11  | +1.09 | +1.11pp |
| 3y  | +0.0057 | 58.0% | 35  | +0.11 | +0.31pp |
| 5y  | +0.0115 | 60.4% | 59  | +0.88 | +0.40pp |
| 10y | +0.0037 | 54.3% | 117 | +0.41 | **-0.00pp** |

All positive, all tiny, none significant. Over a decade the quintile spread is
exactly zero. Negative years: 2016, 2020 (-0.060), 2025 (-0.021). 2019 ~0.

## CONTROL — but the ranking still earns its keep, and recently stopped

Same rules, RS permuted across names each date (distribution and the count
clearing min_comp_rs preserved; only WHICH names changes), 20 seeds.

| window | real | random mean | p05 | p95 | real beats |
|---|---|---|---|---|---|
| 3y  | 13.1%  | 9.4%   | -11.5% | 34.0%  | 65% of seeds |
| 10y | 445.0% | 223.7% | 121.9% | **317.7%** | **100% of seeds** |

Over 10 years real sits ABOVE p95 and roughly doubles the random mean. Over
3 years it sits mid-band — indistinguishable from random.

Reconciling that with a ~0 IC: IC averages rank correlation across all ~1000
names, while the strategy holds the top 15. A factor can carry no monotone
cross-sectional ordering and still have a usable extreme right tail. The
quintile table cannot see the top 1.5%.

## MONEY

| window | CAGR | max DD | Sharpe | trades | win% | avg win | avg loss | vs Nifty |
|---|---|---|---|---|---|---|---|---|
| 1y  | 26.15% | -12.1% | 1.52 | 104  | 30.8% | +28.6% | -8.9%  | +32.1pp |
| 3y  | **4.20%** | **-29.6%** | **0.33** | 319 | 36.1% | +21.5% | -10.7% | **-4.3pp** |
| 5y  | 19.98% | -24.4% | 1.20 | 516  | 35.1% | +28.1% | -9.7%  | +112.9pp |
| 10y | 18.49% | -40.4% | 1.03 | 1001 | 35.8% | +29.4% | -10.0% | +275.6pp |

Survivorship still flatters the levels (delisted names absent). It cannot
explain the 3-year window trailing Nifty by 4.3pp.

## EARNINGS OVERLAY — not supported

Price-derived shock proxy (>4% day on >2.5x volume, within 63 days):

| window | all names | post-shock | no recent shock |
|---|---|---|---|
| 1y  | +0.0357 | +0.0420 | **+0.0448** |
| 3y  | +0.0057 | +0.0096 | **+0.0204** |
| 5y  | +0.0115 | **+0.0169** | +0.0134 |
| 10y | +0.0037 | +0.0090 | **+0.0108** |

"No recent shock" ranks better in 3 of 4 windows. The article's claim that
fundamentally-confirmed momentum ranks better is not supported by this proxy.
Differences are tiny and none are significant.

## Decision

The factor is not broken and the ranking is not decoration — it doubled returns
over the decade. But it has produced nothing for three years, and the
fundamental overlay we hoped would repair it does not. That argues for
regime-conditional exposure (H2) rather than further entry-signal work.
