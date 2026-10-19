# Can charts and valuation improve momentum entries and exits? (run 1, 18 Oct 2026 — provisional)

Source: `entry_exit_study.py`, Actions `script=entry_exit`; output `entry_exit_output.txt`, cases
`ENTRY_EXIT_CASES.md`. Weekly stock-dates 2016-2026, returns vs the median (typical) stock.

## Two bugs found in run 1 — results below are provisional
1. **Survivorship came back.** The universe was "point-in-time top 1000 AND has a sub-industry"; the
   industry map only covers today's list, so the run used 1,085 of today's names and scraped 0 extra
   fundamentals. Stocks that collapsed out of the top 1000 are missing — that flatters "hold", micro caps
   and loss-makers, and undercounts "gave back". The same filter sits in `forensics_gate_study.real_data`,
   so earlier studies built on it share the bias. Fixed for run 2: universe is turnover-only; Screener
   fundamentals AND its "Broad Industry" (same 58-name vocabulary) are scraped for every past top-1000 name.
2. **Unadjusted bonuses.** Yahoo missed e.g. SHARDAMOTR's 1:1 bonus (Jan 2025: 1790 -> 861 in a day), which
   showed up as a "-61% gave back" case. 70 non-reversing one-day falls of 40%+ were flagged. Fixed for run 2:
   a fall within 2% of a bonus/split ratio, confirmed by equity capital rising ~1/ratio (bonus) or volume
   stepping up >= 0.7/ratio for 20 sessions (split), is repaired. Unit-tested: bonus and split repaired, a real
   crash and a crash landing exactly on -50% without confirmation left alone. Also affects earlier studies.
3. Run 2 also: P/E >= 200 joins LOSS (M&M 2020 read RICH at a capped P/E of 500 off one bad year), and the
   "gain left after the first trigger" statistic with a blind-entry baseline.

## What run 1 says (directionally)

**Pre-registered hypotheses: none supported (H1-H5 all fail in at least one era).**

| | 2016-20 | 2021-26 | Read |
|---|---|---|---|
| S1 80%-winner lift, 6m rank >= 90 | 1.55 (4/5 yrs) | 1.76 (6/6) | momentum is the best winner predictor |
| S1 lift, 1m ACCEL | 1.22 (5/5) | 1.29 (6/6) | consistent |
| S1 lift, Micro / Large tier | 1.81 / 0.29 | 1.68 / 0.28 | "80% winner" is largely a volatility label |
| S1 lift, valuation CHEAP / RICH | 1.15 / 0.90 | 0.88 / 0.94 | no consistent valuation edge |
| S1 lift, own P/E %ile <= 30 | 0.92 | 0.69 | cheap vs own history did NOT predict winners |
| S1 lift, breakout + anchor + leader | 0.99 | 1.30 | chart trigger helped only in 2021-26 |
| S1 lift, 6m rank >= 90 + RICH / + CHEAP-FAIR | 1.45 / 1.09 | 1.68 / 1.67 | being expensive did not hurt momentum |
| S2 momentum outcomes: continued / sideways / pullback / gave back | 42 / 36 / 10 / 13% | 39 / 38 / 11 / 13% | stable |
| S2 6m vs typical, momentum RICH / CHEAP-FAIR | +9.5 / +3.3 | +2.4 / +4.8 | sign flips: no rule |
| S2 6m vs typical, v6 ABSORPTION | +9.1 (62% beat) | +8.5 (62%, 6/6 yrs) | best sign found; small n (252 / 653) |
| S2 6m vs typical, near overhead supply | +2.2 | -1.9 | give-back lift 1.28 / 1.35; small n |
| S3 6m, FADING + RICH vs FADING + FAIR | +5.1 vs +2.5 | +3.5 vs +4.5 | velocity x valuation adds nothing |
| S3 6m, FADING vs STEADY (any valuation) | +1.0 vs +5.0 | +3.5 vs +3.1 | slowing hurt only in 2016-20 |

**Exits (2,520 momentum episodes).** Per trade, holding 6 months beat the live exit (+12.3 vs +4.3 pp vs Nifty
in 2021-26; +11.4 vs +4.5 in 2016-20): the live exit cut the continuers from +63% to +20% while saving the
give-backs (-36% -> -7%). Per day of capital the live exit still wins (~0.17 vs 0.13 %/day in 2021-26; 0.20
vs 0.16 in 2016-20), matching the 15-slot portfolio test that kept it. Adding a velocity or valuation exit
made both measures worse. Run 1's survivorship also flatters "hold".

**By size.** Large caps: momentum, v6 BUY NOW and breakouts all lost to the typical stock in 2021-26
(-3.8, -1.1, -3.0 pp at 6m). Mid caps: breakout + anchor + leader +4.9 / +4.5 in both eras. Small caps:
v6 BUY NOW +4.4 / +3.9 (5/5, 6/6 years). Micro: strongest in 2021-26 (+9 pp) but the most survivorship-prone.
**Commodity producers:** momentum + CHEAP/FAIR lost in both eras (-6.8 / -2.2 pp) while momentum + RICH won —
a low P/E at peak-cycle earnings is a trap, consistent with the engine's normalized-earnings commodity track
(small samples: 21 / 42 stocks).

## Cases (ENTRY_EXIT_CASES.md)
80% winners come in two kinds. **Crash rebounds** (M&M and IIFL 2020, SAMMAANCAP, PCJEWELLER 2018, FACT,
CUPID, ORCHPHARMA): far below their highs, CHEAP or FAIR, momentum fading, chart VETO/DISTRIBUTION at the
start — no momentum or chart rule should buy these, and none did. **Momentum continuations** (AURUM,
WEBELSOLAR, KITEX, EXCELINDUS): top-decile or accelerating, Gate 1, and a breakout/GO trigger within weeks
with most of the move still ahead. Valuation at the peak was usually RICH, but it was RICH for months before.

## Use (pending run 2)
Nothing to change yet. Run 2 decides whether ABSORPTION, overhead supply, the large-cap weakness and the
commodity cheap-at-peak trap survive once the collapsed names and the bonus errors are in.
