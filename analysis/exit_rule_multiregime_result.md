# Multi-regime exit-rule backtest — result

Source: GitHub Actions run `30550106601` (attempt 2), `exit_rule_backtest.yml`,
`--start 2019-01-01 --max-tickers 400`.
Universe: 400 requested, 100% download coverage, **373 usable tickers**,
**1867 trading days** (2019-01-01 → 2026-07-30). Initial capital ₹10,00,000.

NOTE: that run was dispatched from `main` @ `c4eba89`, which carried only the
first **6** variants. B4 / B4b / B5 / B6 (18% trail, RS-deceleration) were added
later and still need a run from this branch.

## Verdict: keep the current exit rules. Change nothing.

## OVERALL

| variant | final equity | total % | CAGR % | max DD % | Sharpe | trades | win % | avg win | avg loss | vs baseline |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 75,26,105 | 652.6 | 30.55 | -32.6 | **1.55** | 580 | 41.4 | 32.12 | -8.93 | 0 |
| B1_caution_trail15 | 75,89,714 | 659.0 | 30.70 | -32.0 | 1.52 | 542 | 41.1 | 34.69 | -9.74 | +63,609 |
| B1b_trail_flat15 | 76,65,223 | 666.5 | 30.87 | -32.0 | 1.54 | 542 | 41.9 | 33.81 | -9.79 | +1,39,118 |
| B2_ma50_confirm2 | 68,01,718 | 580.2 | 28.82 | -35.3 | 1.47 | 569 | 40.2 | 32.85 | -9.31 | **-7,24,387** |
| B3_combined | 79,07,670 | 690.8 | 31.41 | -34.1 | 1.54 | 527 | 41.6 | 34.67 | -10.02 | +3,81,565 |
| B3b_flat15_confirm2 | 80,11,057 | 701.1 | 31.64 | -34.1 | 1.55 | 526 | 41.4 | 35.53 | -10.09 | +4,84,952 |

## The whole edge is one year (2020)

Return % by calendar year, gap vs baseline in pp:

| year | baseline | B1 | B1b | B2 | B3 | B3b |
|---|---|---|---|---|---|---|
| 2019 | 10.6 | -1.7 | -1.7 | +0.2 | -0.7 | -0.7 |
| 2020 | 105.8 | **+24.4** | **+17.5** | +2.5 | **+21.6** | **+28.4** |
| 2021 | 183.8 | -13.0 | -1.7 | -15.5 | -17.8 | -22.3 |
| 2022 | -1.0 | +4.5 | +4.5 | +2.0 | +12.6 | +12.6 |
| 2023 | 42.0 | -3.9 | -3.9 | -3.3 | -3.0 | -3.0 |
| 2024 | -12.2 | +1.0 | +1.0 | -5.7 | -1.5 | -1.5 |
| 2025 | -13.6 | -2.3 | -2.3 | +2.0 | -0.3 | -0.2 |
| 2026 YTD | 9.2 | -4.5 | -4.5 | -1.2 | -6.1 | -6.1 |

Compounded return **excluding 2020**:

| baseline | B1 | B1b | B2 | B3 | B3b |
|---|---|---|---|---|---|
| **+265.5%** | +229.6% | +243.3% | +226.4% | +247.4% | +241.9% |

Baseline wins ex-2020 against **every** variant. B3b's entire +₹4.85 lakh edge
is the single COVID V-recovery, where a tight trail whipsawed out of the March
2020 bottom and a wide one held. n=1 event. Baseline also wins 5–6 of 8 years
outright, including 2026 YTD.

## The stated rationale is falsified at trade level

Realised trade P&L % by regime at exit:

| regime | baseline | B1 | B1b | B3 | B3b |
|---|---|---|---|---|---|
| BEAR | **-2.46** | -6.18 | -14.38 | -6.18 | -14.38 |
| BULL | 15.35 | 16.80 | 16.49 | 16.30 | 16.72 |
| CAUTION | **3.03** | 1.63 | 1.46 | 1.90 | 1.73 |

The proposal was "wider stops survive the chop." Instead:

- In **BEAR**, loss per trade gets monotonically worse as the trail widens
  (-2.5% → -6.2% → -14.4%). Wider stops do not survive a bear; they bleed in it.
- In **CAUTION** — the regime B1 actually modifies, and where the portfolio
  spends most of its life — per-trade P&L *drops* (3.03% → 1.63%).
- The variants only earn in **BULL**, i.e. by riding winners longer, which is a
  different claim from the one being tested.

Caveat: exits are path-dependent, so a variant's regime buckets contain a
different trade set than baseline's. B1 and B3 show identical BEAR figures
(as do B1b and B3b), which implies the BEAR bucket is a small shared sample.

## Other findings

- **B2 (2-day MA50 confirm) is harmful alone**: -₹7.24 lakh, Sharpe 1.47,
  worst drawdown of the set at -35.3%. It only looks good *combined* with a
  wider trail (B3/B3b) — a textbook overfit interaction, and it too is
  2020/2022-driven.
- **No Sharpe improvement anywhere.** Baseline 1.55; best variant 1.55.
  The extra terminal wealth is bought entirely with extra risk.
- **Drawdown gets worse** for the two "winning" variants: -34.1% vs -32.6%.
- **Consistent with the offline run** (`exit_rule_local_summary.csv`,
  Feb–Jul 2026, CAUTION): baseline won there too, and 2026 YTD here repeats it
  (9.2% vs 3.1%).

## What this rules out

2024 (-12.2%) and 2025 (-13.6%) are two consecutive losing years, and **no exit
variant fixes either** — the best 2024 is -11.2%, the best 2025 is -11.6%. The
recent underperformance is not an exit-rule problem. The lever is upstream:
entry selection, universe, or regime gating.
