# Can charts and valuation improve momentum entries and exits? (runs 2-3, 19 Oct 2026)

Source: `entry_exit_study.py`, Actions `script=entry_exit`; output `entry_exit_output.txt`, cases
`ENTRY_EXIT_CASES.md`. Weekly stock-dates 2016-2026, point-in-time top-1000 by turnover, 1,855 stocks
(run 1: 1,085 survivors only). Returns vs the median (typical) stock; every rule judged in both eras.

## Data fixes between run 1 and run 2
- **Survivorship.** Run 1's universe was "top 1000 AND has a sub-industry", i.e. today's list only.
  Run 2: turnover-only universe; Screener fundamentals + Broad Industry scraped for all 766 past
  top-1000 names (766/766 parsed). The same filter sits in `forensics_gate_study.real_data`, so the
  earlier forensics / breakout studies were run on survivors.
- **Unadjusted bonuses** (SHARDAMOTR 1:1 bonus, Jan 2025, read as -52%): repaired when a one-day fall
  matches a bonus ratio AND equity capital rose by the same factor, or matches a split ratio AND
  volume stepped up. Run 2 also let volume confirm bonus-only ratios, which wrongly "repaired" real
  crashes (PCJEWELLER 2018, ANTGRAPHIC x25): fixed after run 2, affects < 100 events.
- **Demergers are not adjusted by Yahoo** (VEDL 2026: 655 -> 272) and cannot be caught by any ratio
  test. ~60 such unrepaired one-day falls of 40%+ remain flagged; next run lists all of them and
  repeats the exit table without the episodes that contain one. Against ~3,800 episodes this moves
  averages by < 1 pp and cannot flip the conclusions below.
- P/E >= 200 now counts with LOSS (M&M 2020 read RICH at a capped P/E of 500).

## Pre-registered hypotheses: none supported
| | 2016-20 | 2021-26 | Verdict |
|---|---|---|---|
| H1 winner lift CHEAP >= 1.2 x RICH | 1.08 vs 0.91 | 0.87 vs 0.91 | fail |
| H2 breakout + anchor + leader lift >= 1.5 | 1.04 | 1.15 | fail |
| H3 momentum RICH trails CHEAP/FAIR by >= 2 pp (6m) | +7.6 vs +4.1 | +0.9 vs +3.6 | fails 2016-20 |
| H4 FADING+RICH trails FADING+FAIR by >= 2 pp | +6.0 vs +3.3 | +2.6 vs +3.4 | fail |
| H5 live exit + velocity / valuation exit beats live | 2.1 / 2.8 vs 3.4 | 2.8 / 3.3 vs 3.8 | fail (both worse) |

## 1. 80%+ winners (4,872 runs)
- **Momentum predicts them; valuation does not.** 6m rank >= 90: lift 1.51 / 1.65 (5/5, 6/6 years);
  1m accelerating 1.18 / 1.24 (5/5, 6/6). Valuation CHEAP 1.08 / 0.87, RICH 0.91 / 0.91; cheap vs the
  stock's own history 0.89 / 0.72 (less likely to win, not more). Momentum + RICH 1.38 / 1.59 — being
  expensive did not stop momentum names becoming winners.
- **"80% winner" is mostly a volatility label**: Micro 1.39 / 1.40, Large 0.28 / 0.28; loss-makers 1.47 / 1.43.
- **Chart triggers confirm, they do not anticipate.** A breakout + volume anchor or v6 BUY NOW appeared
  in 57% / 86% of runs, after a median 14 / 9 weeks, with 36% / 48% of the run still ahead. Buying
  blind at the same delay left 44% (13 wk) / 63% (8 wk) ahead. The trigger is safer (it filters the
  non-starters) but not earlier or cheaper.
- Cases: winners are either crash rebounds (deep below highs, cheap, fading, chart VETO — no rule
  catches these) or momentum continuations (top decile / accelerating, Gate 1, trigger within weeks).

## 2. Momentum names (6m +50%, top 10%): what happened next
Continued 40 / 37%, sideways 35 / 37%, pullback 10 / 11%, gave back 15 / 15% (run 1 had 13%:
the collapsed names are now in).

Signs that held in BOTH eras (6m vs the typical stock):
- **Loss-making momentum** -4.8 / -1.4 pp; give-back odds 1.21 / 1.19x.
- **Already off the high (fails Gate 1)** -3.7 / -0.6 pp.
- **v6 ABSORPTION** (pullbacks on drying volume) +8.4 / +6.0 pp, 62% / 59% beat (n 310 / 821) — the
  strongest positive sign, but its give-back odds flip between eras (1.20 / 0.82).
- **Leader industry** +4.5 / +2.2 pp.
- Extended > 60% above 200d and RSI > 75 raise give-back odds (1.23 / 1.12, 1.21 / 1.15) but also
  keep continuing at the base rate — more volatility, not a sell signal (6m returns +0.5 / +1.8).
Not signs: RICH vs CHEAP/FAIR (+7.6 vs +4.1, then +0.9 vs +3.6), 1m FADING (-2.2 / +0.5),
DISTRIBUTION (+4.1 / +2.2), overhead supply (-2.7 / +0.6).

## 3. Momentum slowing x valuation (6m rank >= 80)
FADING alone: +1.1 / +1.8 pp vs STEADY +4.9 / +2.9 — slowing costs ~2-4 pp. Valuation does not sharpen
it: FADING+RICH +6.0 / +2.6, FADING+FAIR +3.3 / +3.4, FADING+CHEAP +7.7 / +1.3. LOSS is the only
valuation state that consistently hurts (FADING+LOSS -5.4 / +1.0; ACCEL+LOSS -3.4 / -0.3).

## 4. Exits (3,781 momentum episodes)
| | hold 6m | live exit (MA50 / 15% trail) | live + velocity | live + valuation |
|---|---|---|---|---|
| 2016-20 mean / vs Nifty / days | 17.0 / 8.2 / 126 | 5.0 / 3.4 / 28 | 3.2 / 2.1 / 23 | 4.2 / 2.8 / 26 |
| 2021-26 mean / vs Nifty / days | 14.2 / 9.7 / 126 | 4.7 / 3.8 / 29 | 3.6 / 2.8 / 25 | 4.3 / 3.3 / 25 |
Per trade, holding wins; per day of capital the live exit wins (0.18 vs 0.14 %/day 2016-20; 0.16 vs
0.11 2021-26) — the basis on which the 15-slot portfolio test kept it. Velocity and valuation exits
are worse on both measures. The live exit cuts the continuers from +62-70% to +19-21% and the
give-backs from -37 to -40% to -7 to -8%.

## 5. By size tier and family (6m vs typical, 2016-20 / 2021-26)
| | Large | Mid | Small | Micro |
|---|---|---|---|---|
| momentum (rank >= 90 + Gate 1) | +7.4 / **-4.0** | **+12.1 / +4.3** | +7.4 / +2.6 | +2.5 / +4.2 |
| v6 BUY NOW | +1.6 / +0.4 | +8.4 / +3.2 | **+4.6 / +3.6** (5/5, 6/6) | **+5.1 / +3.3** (5/5, 6/6) |
| breakout + anchor + leader | +0.5 / -1.9 | **+6.2 / +6.2** | +4.9 / +2.7 | -0.8 / +2.0 |
| momentum + CHEAP/FAIR vs + RICH | +5.9 vs +9.9 / -2.8 vs -5.5 | +8.7 vs +20.0 / +5.6 vs -2.2 | +7.9 vs +10.1 / +4.8 vs +3.0 | +1.6 vs +1.6 / +3.0 vs +2.9 |
- **Large caps: momentum and chart entries stopped working in 2021-26**; there the live exit beat
  holding even per trade (+1.2 vs +0.1 pp vs Nifty).
- **Mid caps are where charts add most**: breakout + volume anchor + leader industry +6.2 pp in both eras.
- **Small / micro: v6 BUY NOW is the steadiest entry** (positive every year in both eras).
- Valuation inside momentum flips by era in every tier: RICH did better in 2016-20, CHEAP/FAIR in
  2021-26. Not a rule.
- **Commodity producers: momentum while CHEAP/FAIR lost in both eras** (-4.8 / -0.8 pp) vs RICH
  +15.3 / +1.7 — a low P/E at peak-cycle earnings is a trap (small: 29 / 49 stocks).

## 6. Entry triggers inside momentum (run 3; H6, set before the run)
Every 6m-rank >= 90 + Gate 1 episode (2,097 / 2,303 — winners AND losers). Enter at once, or wait up to
3 months for a trigger. Judged against entering all episodes blind after the same delay (6m vs typical,
mean; 2016-20 / 2021-26):

| | fires | median wait | from trigger | blind, same wait | edge | median from trigger |
|---|---|---|---|---|---|---|
| enter at once | 100% | 0 | 11.2 / 10.3 | — | — | 3.0 / 2.8 |
| **52w / ATH breakout** | 45 / 47% | 26 / 27 d | 13.4 / 12.6 | 12.1 / 10.1 | **+1.2 / +2.5** | 6.0 / 4.1 |
| **squeeze release** | 52 / 49% | 29 / 27 d | 15.4 / 12.4 | 12.1 / 10.1 | **+3.3 / +2.2** | 7.2 / 4.3 |
| gap that held | 47 / 50% | 19 / 18 d | 14.5 / 10.3 | 12.0 / 10.1 | +2.5 / +0.2 | 3.6 / 0.8 |
| strong close (CLV >= 0.8, 1.5x vol) | 90 / 92% | 8 / 7 d | 13.0 / 10.5 | 11.9 / 10.1 | +1.1 / +0.4 | 4.3 / 2.8 |
| volume anchor | 70 / 86% | 6 / 5 d | 10.8 / 9.9 | 11.8 / 10.1 | -1.0 / -0.2 | 3.2 / 2.0 |
| pocket pivot, NR7, pullback-to-20d, RSI(2) dip | 93-100% | 2-13 d | | | -0.4 to +0.6 | |

- **Supported: 52w/ATH breakout and squeeze release.** A momentum name bought on its breakout (or on a
  Bollinger-squeeze release) did 1-3 pp better over 6 months than buying any momentum name after the
  same wait, and 2-5 pp better than buying at once — so waiting for the breakout did NOT give up gains.
  Both fire in only about half of episodes after ~5 weeks, so they select names rather than time them;
  year-by-year they beat the blind entry in 4/5 and 3/6 (breakout), 3/5 and 3/6 (squeeze).
- **Not supported:** CLV / strong close, volume anchor alone, gap, pocket pivot, NR7, pullback, RSI(2) —
  they fire on almost every momentum name within days, so they add little information.
- Demerger check: only 7 of ~3,780 exit episodes contained one of the 64 flagged one-day falls; dropping
  them changes nothing. Run 3's bonus repair still passed 3 COVID crash days on two banks (equity
  capital tolerance 15%); tightened to 5% after the run — too few rows to move any result.

## Use
1. Keep the live exit. Do not add velocity or valuation exits.
2. Do not use valuation (P/E vs history or peers) to accept or reject momentum entries; the one
   exception is **loss-making momentum** — flag it (worse returns, higher give-back odds, both eras).
3. In the scanner, show size tier next to the chart signal: breakout + anchor + leader is a mid-cap
   tool; v6 BUY NOW is the small/micro tool; large-cap momentum has had no edge since 2021.
4. Highlight v6 ABSORPTION on momentum names as the one positive chart sign.
5. Commodity names: treat a low P/E during a momentum run as a warning, not value.
6. For momentum names, prefer entries on a fresh 52w/ATH breakout or a squeeze release; do not wait
   for a CLV / volume-anchor / pullback / RSI(2) signal — they add nothing on top of momentum.

## 7. Live-portfolio test (entry_filter_portfolio_backtest.py, 19 Oct 2026): keep the engine
Live rules, only the names allowed into a free slot changed. PIT top-1000 by turnover, 2014-2026.

| variant | 10y CAGR | 10y Sharpe | 10y max DD | 8 starts beat BASE | 3 periods beat BASE | median d CAGR (starts) |
|---|---|---|---|---|---|---|
| **BASE live engine (6m RS)** | **27.5%** | **1.70** | -26.2% | — | — | — |
| E1 breakout / squeeze required | 22.5% | 1.43 | -26.0% | 1/8 | 0/3 | -2.7 pp |
| E2 breakout / squeeze first | 22.8% | 1.42 | -29.4% | 1/8 | 0/3 | -2.4 pp |
| Z1 no large caps | 27.0% | 1.68 | -30.3% | 4/8 | 2/3 | +0.2 pp |
| X1 no loss-makers | 24.1% | 1.52 | -32.4% | 0/8 | 0/3 | -2.1 pp |
| ZX no large, no loss | 22.5% | 1.44 | -35.4% | 0/8 | 0/3 | -3.2 pp |
| C1 E1 + Z1 + X1 | 18.8% | 1.22 | -24.1% | 0/8 | 0/3 | -6.3 pp |
| C2 E2 + Z1 + X1 | 18.1% | 1.17 | -29.6% | 0/8 | 0/3 | -6.3 pp |

- **Requiring or preferring a breakout/squeeze costs ~2.5 pp a year**, the same lesson as the 30 Sep
  breakout test. The event study measured holding the trigger name 6 months; the book exits on an MA50
  break and the filter removes ~13% of buys that the 6-month ranking would have made. The ranking already
  carries the momentum the trigger confirms.
- **Skipping loss-makers costs ~2 pp a year** even though loss-making momentum names do worse on average:
  17% of the engine's buys are loss-makers, and the turnarounds among them pay for the rest inside a book
  with an MA50 exit (the exit caps the bad ones).
- **No large caps is a tie** (4/8 starts, 2/3 periods, +0.2 / -0.3 pp) — only 4% of buys are large caps.
- Nothing beats BASE at most starts AND in most periods. **Engine unchanged.** The scanner columns
  (momentum entry, size tier, flags) stay as discretionary context, not engine rules.
