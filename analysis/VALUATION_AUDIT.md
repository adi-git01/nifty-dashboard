# Mechanical v4.1 valuation — audit and engine-vs-judgment check (13 Oct 2026)

## Bugs found and fixed
| Finding | Scale | Fix |
|---|---|---|
| Consolidated net profit includes minority interests (EPS overstated: Vedanta, EID Parry, Grasim, Bajaj Finserv...) | 53 stocks with < 85% attributable | EPS and own-multiple history use the attributable share (latest FY EPS x shares / NP) |
| Holding companies valued on a multiple | 15 stocks < 60% attributable | Out of scope (v4.1: SOTP problem) |
| Stale quarterly data (latest quarter as old as 2009-2024) | 39 stocks | NO DATA when the latest quarter is > 270 days old |
| Own-history multiples on near-zero earnings (300x months) faked "cheap on recovery" | 89 stocks with own/spot > 3x | Own history trimmed above max(150x, 4x peers) |
| Book floor could invert bands (bull below base, negative after widening) | AKG, Atlas Cycles, Ashoka Metcast... | Floor capped below base; bear < base < bull enforced after every step |
| Missing margins left bear/bull blank (NaN) | 1 | +/-30% fallback, flagged |
| Gate C failed on Monte Carlo noise | 32 of 39 failures | Tolerance max(1%, 3 x MC standard error) |
| "Close to a verdict boundary" caution computed after the grade | all | Grade computed last |

## Business-model peers
Nearest neighbours within a sector family (chemicals + pharma share one; hospitals separate) on 5Y operating
margin, 5Y ROCE, capital intensity, 3Y growth and the multiple the market has paid for 5 years; lenders on ROE,
growth, own P/B and promoter holding (PSU vs private), banks with banks and NBFCs with NBFCs; commodity producers
against other producers; holding companies never used as peers. When even the nearest peers are far (mean rank
distance > 0.22), own history carries 75% of the anchor. The v4.1 cap now uses this peer median.

**Hybrid (tested before adopting).** Sub-industry is used as the peer pool when it is a tight, relevant group:
>= 6 members, P/E (P/B for lenders) interquartile spread / median <= 0.65, and the stock's own 5Y multiple inside
[P25 / 1.5, P75 x 1.5]. Otherwise the business-model neighbours above. Pre-registered criteria on the 11-stock
sample, all passed (narrowly): agreement with judgment no worse (same bucket 6/11, mean gap 11 pp, as before); peer P/E spread 0.59 vs
0.60; log-error of the peer median vs hand-picked peers 0.115 vs 0.127. 93 stocks now use their sub-industry;
12 ratings changed across the universe (JK Lakshmi fixed, Arvind slightly worse).

## Engine vs judgment v4.1 (random sample, seed 20261013, Rs 5,000 cr+)
14 drawn: 2 insurers (out of scope), 1 stale; 11 compared. Judgment runs set identity, hand-picked business-model
peers, normalized growth/margins and event risk, then used the same engine (analysis/valuation_judgment_vs_engine.csv).

| | before the fixes | after |
|---|---|---|
| same verdict bucket | 4 / 11 | 6 / 11 |
| mean absolute percentile gap | 32 | 11 |

Remaining misses are adjacent buckets except Redington (IT distribution: no listed same-model peer; judgment
anchors on own history alone). Caveat: the judgment runs were made by the same author as the engine -- the
reference log of your own v4.1 runs is the real test.

## Gate 1 "within 15% of the 52-week high" (approximate: monthly data 2020-26, current universe)
Stocks in an uptrend, median return vs the typical stock: 0-5% off the high +1.4 / +2.7 pp (3m / 6m), 5-10%
+0.5 / +1.1, 10-15% +0.1 / +0.1, 15-20% -1.2 / -2.5, 20-25% -2.2 / -3.4, > 25% -2.8 to -6.6. 15% sits where the edge
reaches zero; 10-15% is neutral, beyond 15% is negative. Keep 15% as the gate; a daily-data run should confirm.

## Technicals tab bugs found and fixed
| Finding | Effect | Fix |
|---|---|---|
| Position sizing when the stop is missing or above price | NaN risk fell through to a full slot | 0 shares, "no valid stop" |
| "R:R from current" was target-1-defined, so always 2.0 | No information | Reward to T2 (next supply shelf) / risk; blank when no resistance above |
| Gate 1 tests the 252-day closing high; the table showed the intraday high | 21 stocks looked > 15% off yet passed (or the reverse) | New "From high" column = the distance Gate 1 actually tests |

Checked and consistent: every BUY NOW has Gate 1 + GO; T1 above price; shelves ordered; anchor volume >= 3x;
retention in 0-1; edge cases (short history, zero volume, flat prices) do not crash.
