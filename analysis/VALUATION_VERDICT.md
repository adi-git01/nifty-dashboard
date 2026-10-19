# Where does valuation help, outside momentum? (valuation_study.py, 19 Oct 2026)

Monthly starts Jul 2017 - Oct 2026, 1,464 stocks, point-in-time top-1000 universe (past members included),
P/E = split-adjusted price / EPS public 92 days after the year end. Returns vs the median (typical) stock.
Halves: starts 2017-20 vs 2020-26. Delisted companies are missing, which flatters cheap stocks.

## Pre-registered: nothing supported in both halves
| | 2017-20 | 2020-26 | |
|---|---|---|---|
| V1 CHEAP beats RICH by >= 6 pp at 2y | +9.1 vs +13.7 | +15.6 vs -5.4 | fails 2017-20 |
| V2 RICH falls >= 3 pp more in crashes | gap -2.7 (2018-20), +1.4 (2022), -17 (2024-26) | | 1 of 3 windows |
| V3 P/E >= 100 / loss-makers trail by >= 5 pp at 2y | -3.9 / -15.3 | -14.3 / +2.1 | split |
| V4 70/30 or 50/50 engine/value blend beats engine | Sharpe 1.59 / 1.43 vs 1.64; max DD -37 / -45 vs -35 | | fails |
| V5a commodities: normalised P/E spread | too few stocks | +17.6 vs trailing +9.2 | too few |
| V5b lenders: P/B / ROE spread at 1y | -20.4 (dearest won) | +21.4 | split |
| V6 cheap + earnings recovery beats cheap | -2.5 vs +0.7 | +1.1 vs +2.0 | fails |

## What the data does say
**1. Valuation is a slow, regime-dependent signal.** At 3 years (median vs typical):
| | 2017-20 starts | 2020-26 starts |
|---|---|---|
| P/E 0-10 / P/E 100+ | +22.2 / -8.9 | +44.6 / -30.4 |
| P/E 60-100 | -7.4 | -31.1 |
| cheapest / dearest quintile vs peers | +11.8 / +0.1 | +28.1 / -22.8 |
| cheapest / dearest vs own history | +5.9 / +3.7 | +17.2 / -14.3 |
| zone CHEAP / RICH | +12.7 / +13.0 | +29.7 / -20.3 |
At 1 year almost nothing separates. The 2020-26 effect is very large (a value / re-rating era); in 2017-20
RICH did as well as CHEAP. The one pattern with the same sign in both halves at 3 years (seen after the run,
not pre-registered): **P/E >= 60 underperformed and the cheapest P/E band and cheapest-vs-peers
outperformed**, strongly in 2020-26, mildly in 2017-20.

**2. Valuation is not a crash shield; losses are.** Peak-to-trough by zone: 2018-20 RICH -65.6 vs CHEAP
-61.9; 2022 -23.1 vs -24.1; 2024-26 -33.0 vs -15.8. Loss-makers fell most in all three (-82.8, -27.9,
-42.0) and had the highest odds of a -50% fall within 2 years in both halves (65% vs 46%; 15% vs 9%).
RICH names actually had shallower one-year drawdowns in 2017-20 (-27 vs -38).

**3. A value sleeve does not improve the engine.** 30 cheapest of the 500 most liquid: 19.4% CAGR,
Sharpe 0.94, max DD -62% (lost 8.7% a year in 2017-19); the 30 DEAREST did slightly better (20.5%,
Sharpe 1.06). Correlation with the engine 0.45. Blends lowered Sharpe and deepened drawdowns over the full
period; only in 2022-26 did the 70/30 blend help (Sharpe 1.57 vs 1.40, max DD -16 vs -23).

**4. Lenders: price-to-book worked only in the recent half.** 2020-26 cheapest P/B / ROE +45.7 vs dearest
-22.6 at 2 years; 2017-20 the reverse — the 2018 NBFC credit crisis, when cheap lenders were cheap for a
reason. Commodities: normalised P/E beat trailing P/E in 2020-26 (spread +17.6 vs +9.2 at 1 year); too
few stocks before 2020 to judge.

**5. Turnarounds (cheap + earnings recovering on annual data) add nothing** over cheap alone.

## Use
- Do not add valuation to the momentum engine or a value sleeve beside it.
- For **multi-year holdings outside the engine**, treat P/E >= 60 as a headwind and cheapness vs peers as
  a mild tailwind — consistent in sign across both halves at 3 years but regime-dependent in size, and
  not pre-registered: a lean, not a rule.
- For **risk**, loss-makers are the consistent danger (worst in every crash, highest -50% odds); expensive
  profitable names were not.
- Lenders and commodities: use P/B / ROE and normalised P/E rather than trailing P/E, knowing a credit
  cycle can invert the lender signal.
