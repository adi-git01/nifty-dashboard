# Winners Post-mortem — 202 stocks up >= 50%, 25 Mar - 25 Sep 2026

Every stock in today's 1,168-name universe with a 6-month return >= 50% (202 —
not a hand-picked 25), traced through every tool and every portfolio decision
using the 151 daily snapshots, the trade log and the committed portfolio state.
Per-stock timelines: `analysis/winners_postmortem.csv`.

**The trap this avoids.** A winners-only study always concludes "hold longer,
widen stops, loosen gates, add slots" — the stocks went up by construction.
Every candidate improvement below was therefore re-tested on ALL stocks: the
whole eligible pool at every session, and a day-by-day simulator of the book.

Simulator: snapshot prices, real rebalance dates, regime from the equity curve,
exits (MA50 break, 12%/15% trail), 13-session re-entry cooldown, 0.2% per side.
Validated against the real book: +19.3% vs +14.3%, daily-return correlation
0.86, 51 of 66 names in common. It cannot replicate the CAUTION volume-quality
gate (needs open prices), which is the residual gap. One seven-month path, one
regime: treat small differences between variants as noise.

## Funnel

| Fate in the model portfolio | Stocks |
|---|---|
| Crowded out — eligible on a rebalance day, ranked below the free slots | 133 |
| Bought (10 still held; 30 exits, median P&L -4%, +17% further after exit) | 42 |
| Gated — never eligible on a rebalance day (mostly CompRS < 17) | 27 |

For the bought names, the book captured a median 3% of a median 100% move.

## Leak 1 — the breadth gate never blocked a buy (BUG, FIXED)

`dna3_current_portfolio.py` documents entry rule 4, "market breadth >= 30%".
The scan and buy block sat after the `if breadth < 30 / else` at the same
indentation, so it ran on narrow days too — printing "SKIPPING NEW BUYS" and
then buying. Breadth was 26.9% on 5 Mar and 18.1% on 24 Mar; the engine bought
15 and 10 names. Those 25 trades: median -7.8%, 32% winners, 16 ended in
trend breaks. They filled the book, leaving 2 free slots on each of 16 Apr and
6 May — the two rebalances where the April leaders ranked (WELCORP was #9 and
#5 on exactly those dates).

Simulated, same data: gate broken (as run) +19.3%, max DD -11.1%; gate working
+37.8%, max DD -5.1%, better in both halves. One episode — but the direction
matches the 10-year H2 test, and it is the documented rule.

Fix: `breadth_ok` now zeroes the free slots. End-to-end test on synthetic data:
narrow day, old engine 10 buys, fixed engine 0; healthy day, 15 and 15.

## Leak 2 — the universe

84 of the 202 were not in the universe until 2 Jul; a median 62% of their move
happened before they were visible. Estimating March market cap as today's /
(1 + 6m return): **64 of those 84 already ranked inside the top 1,000 in
March** — the curated list was ~750 names, not a true top 1,000, and missed
recent listings (CPPLUS ranked #287, CUPID #404, HAPPYFORGE #428). Only 19 were
genuinely too small and grew into the universe. A second hole: 2 Jul - 6 Aug the
damaged 777-name list dropped names that had been visible (OPTIEMUS,
LLOYDSENGG, UNIMECH, SKYGOLD, JINDWORLD...).

Status: the universe is now the 1,168-name repaired superset. Still open: the
quarterly market-cap rebuild has failed its last two runs, and the portfolio
ignores any stock with <= 200 bars of history (`len(df) > 200`), so a recent
IPO is unbuyable for ~10 months even when in the universe.

## Leak 3 — crowded out (largest count; mostly a consequence of Leak 1)

The ranking is not the problem in this window. Across all eligible names at
every session, top-K by CompRS minus eligible-pool mean, forward excess (pp):

| depth | 21d | 42d |
|---|---|---|
| top 3 | +13.5 | +28.5 |
| top 5 | +9.7 | +21.4 |
| top 15 | +5.3 | +11.3 |
| whole pool | +2.9 | +6.6 |

Robust to outliers (medians, winsorised): top-5 beat the pool on 76-84% of
days. The problem is too few free slots when the leaders rank. More slots do
NOT fix it: 20 / 25 slots held more winners (46 / 55 vs 34) but returned less
(+27.6% / +26.8% vs +37.8%) — dilution into weaker ranks.

## Leak 4 — exits (do not act on the winners-only view)

Among winners, exits looked costly: trailing-stop exits +16% after, MA50 trend
breaks +51% after. In the simulator a 15% trail or no MA50 exit also looked
better (+38.7% / +43.1%). This was a trending window, and the 10-year
multi-regime exit test (exit_rule_multiregime_result.md) found every loosening
loses once 2020 is excluded. No change.

## What does not help (tested on all stocks)

- Quality, value, ROE, low volatility, large size as ranking keys: negative
  selection edge at both horizons (e.g. low-vol top-5 -3.7 / -6.3 pp).
- Watchlist BUY alerts: 735 alerts on 642 names — over half the universe.
  They "caught" 86% of winners early by volume, not skill: forward excess
  +0.9 (21d) and +5.3 (42d) vs the eligible pool's +2.6 and +6.1.
- More slots (above). A 45% breadth gate here (+27.3%) — the threshold is
  fitted, as H2 found.

## Candidates for the 10-year test (not implemented)

1. RS weighting. The 1-month leg (50% of CompRS) is the weakest measure: its
   top-5 beat the pool on only 39-40% of days, vs 69-84% for 3-month RS,
   1-year return and distance above the 200dma. In the simulator, ranking by
   1-year return gave +46.9% but by 3-month RS +30.4% — mixed, so test the
   weights over a full cycle before changing anything.
2. Rotating out fading holdings on rebalance days (CompRS rank > 100): +41.4%
   vs +37.8% — within noise.
3. Lowering the 200-bar history requirement so recent IPOs can qualify.

## Top 25 by 6-month return

See `analysis/winners_postmortem.csv` for all 202 (visibility, first
eligibility and move left, each tool's first flag, every trade and what
happened after, the outcome at each of the 9 rebalances).
