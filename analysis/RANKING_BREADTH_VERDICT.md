# Ranking, Rotation, IPO and Breadth — 10-Year Verdict

Runs (Actions): 36376061880 (ranking, 1/3/5/10y), 36424045400 (robustness),
36425438409 (breadth gates + sell-all switch). Point-in-time top-1000 universe,
2016-2026, live rules as baseline (CompRS 10/50/40 ranking, CompRS >= 17 floor,
MA50 entry/exit, regime trail, 13-session rebalance, breadth >= 30% gate).

Robustness design, because a two-week start shift and nested windows made the
first run unreliable on its own:
- 8 staggered 10-year starts, 16 sessions apart. These share ~95% of their data,
  so they test sensitivity to the START DATE, not to different market regimes.
- 3 non-overlapping periods (2016-19, 2020-22, 2023-26), fresh book each. This
  is the regime test.
- Random ranking (8 seeds): same eligible names, slots filled in random order.

## Adopt: rank eligible names by 6-month RS instead of CompRS

| Test | K3 6-month RS vs baseline |
|---|---|
| 1/3/5/10y windows | Sharpe better 4/4 |
| 8 staggered starts | better 8/8; median +0.31 Sharpe, +3.6 pts CAGR, drawdown 6.3 pts shallower |
| 3 separate periods | better 3/3: 0.63 -> 1.37, 2.25 -> 2.64, 1.16 -> 1.42 |
| Random-ranking control | beats the best random seed at every start and in every period |

The theme is consistent, not one lucky key: 1-year RS (8/8, 3/3), distance above
the 200dma (7/8, 2/3) and 3-month RS (5/8, 3/3) also beat CompRS. The 50%
1-month weight is what hurts. Simply reweighting CompRS to 10/25/65 did NOT help
(10y Sharpe 1.27 vs 1.35, deeper drawdown) — the gain comes from a longer
lookback, not from trimming one leg.

CompRS itself is real: it beat random ranking at 58 of 64 start/seed pairs.

Caveat: seven keys were tried, and 6-month RS was the best of them. The
consistency across all three tests and the whole family of long-lookback keys is
what makes it credible despite that.

## Reject

- Rotating out faded holdings (rank > 50 / > 100): Sharpe worse over 10y
  (-0.08 / -0.15), 20-37% more trades.
- Earlier IPO eligibility (126 bars): no effect (3/8 starts, median -0.01).
- Removing the stock-level MA50 exit: 10y +650% vs +848%, worse in 0/3 periods.
  Keep it. Removing the MA50 entry filter changes nothing (names with CompRS >= 17
  are almost always above MA50) — redundant but harmless.
- Higher MA50 breadth gates (40%, 50%): worse than 30%.

## Breadth gate on new entries: keep the live 30% MA50 rule

No gate (how the live engine ran until the fix) vs the 30% gate, 10y: +533% vs
+848%, max DD -41.4% vs -29.8%. The gate won 5/8 starts and 2/3 periods. This
supports the breadth-gate bug fix.

% trend score >= 60 as the gate (>= 40, >= 50, or strong-beats-weak): better
over 10 years (7-8/8 starts, drawdown 8-15 pts shallower) but WORSE in the
recent 1/3/5-year windows (e.g. >= 50: 1y Sharpe -0.32 vs 0.73, 3y 0.07 vs
0.53, 5y 1.10 vs 1.39) and better in only 2/3 periods. The 10-year edge is
earlier-decade; not established.

The rebuilt trend-score breadth matches the stored pct_uptrends history
(correlation 0.93, +4.2 pts).

## Sell-everything breadth switch: does not help

Hold only when % TS >= 60 is above an upper level; liquidate everything below a
lower level (40/30, 45/35, 50/30, 50/40; also MA50 40/30, 50/30).

- 10y return +579% to +774% vs +848%; Sharpe 1.21-1.37 vs 1.32; max drawdown
  -26.9% to -30.4% vs -29.8% — barely shallower for a large return cost.
- Worse in the recent 1/3/5y windows; better in at most 2/3 periods.
- Its one clear win is 2020-22 (the COVID crash), where plain Nifty improved
  by the same amount under the same switch — generic crash timing, not
  something specific to this strategy.

Why: the book already de-risks stock by stock as breadth falls (12% CAUTION
trail, MA50 exit). By the time the switch fires, what is left are the strongest
names, and re-entry waits for breadth to recover, which lags the rebound.

## Baseline gap with H2 — found: a harness bug

The two baselines disagreed by 2-3x (H2 today: +281%, Sharpe 0.94; ranking
harness on H2's exact settings: +848%, 1.52) with identical simulator code --
verified bit-for-bit on shared synthetic data. The cause was the input data:

- A bulk download returns the union of every ticker's dates, so a few stray
  rows carry prices for only one or two stocks.
- close_df.rolling(50) cannot span a gap, so each stray row blanked MA50 for
  ~96% of stocks for the next 50 rows: no entries, no MA50 exits.
- H2 iterated over every close_df row; the ranking harness over Nifty sessions.
  On synthetic data with 8 stray rows they returned +95% and +121% against
  +110% clean.

Fixed (exit_rule_backtest.py): fetch() drops rows outside the Nifty calendar,
and MA50 is computed over each stock's own valid closes, as the live engine
does. Both paths now reproduce the clean result exactly.

Consequence: the ABSOLUTE numbers in this file and in earlier backtests were
distorted. Every comparison within a run shared the same distortion, so the
direction of each finding should hold, but the ranking robustness and breadth
runs should be repeated on the fixed harness to confirm.

## Breadth rerun on the fixed harness (Actions run 36466112712)

2025 tickers, 13 stray rows dropped. Baseline (30% MA50 gate), 10y: +793%,
Sharpe 1.29, max DD -30.7%. Wins are counted against this baseline across 4
windows, 8 starts and 3 periods.

| Variant | Windows | Starts | Periods | Verdict |
|---|---|---|---|---|
| No gate (old live bug) | 0/4 | 2/8 | 0/3 | 10y +501%, DD -43%. Gate fix confirmed. |
| MA50 >= 40 / >= 50 | 1/4 | 2-3/8 | 1/3 | Reject |
| TS60 >= 40 | 1/4 | 8/8 | 2/3 | Better earlier; worse in 1/3/5y. Not adopted. |
| TS60 > TS30 | 1/4 | 8/8 | 2/3 | Same pattern as TS60 >= 40 |
| TS60 above its 20d avg | 3/4 | 4/8 | 2/3 | Drawdown 7-12 pts shallower everywhere; 10y return lower (+594%). Watch. |
| No MA50 exit | 2/4 | 1/8 | 0/3 | Keep the MA50 exit |
| Switch TS60 45/35 | 1/4 | 8/8 | 1/3 | 10y +798% with DD -21.3%; slightly lower Sharpe in recent windows |
| Other switches | 0-1/4 | 0-7/8 | 1-2/3 | Reject |

The findings from the old harness hold: keep the 30% MA50 gate, the MA50 exit
and no sell-all switch. What changed: the 45/35 switch no longer costs
return over 10 years. It matches the baseline (+798% vs +793%) and cuts the
worst drawdown by a third. On plain Nifty the same switch lowers Sharpe
(-0.06 over 10y), so the drawdown cut is specific to this book, not generic
market timing. It still fails the pre-registered rule (1/4 windows, 1/3
periods), and it was picked as the best of four thresholds. Treat it as
optional drawdown insurance, not an improvement.

The robustness rerun crashed: a held stock with a missing close made equity
NaN. The simulators now value a holding at its last valid close, and the
workflows fail on a crash instead of reporting success. 6-month RS ranking is
confirmed on the fixed harness only after that rerun.
