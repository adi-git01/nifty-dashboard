# Chart-prompt gates (v5 vs v6) — verdict

Run: ranking_rotation_backtest.yml `script=forensics`, 30 Sep 2026
(forensics_gate_study.py). 2,028 NSE tickers, point-in-time top 1000, weekly
samples 2016-2026: 393,176 stock-weeks, 134,245 pass Gate 1 (34%).
Numbers are median return vs the typical universe stock (pp) at 3 / 6 months
unless marked "mean" (= an equal-weight basket vs the average stock).
Files: forensics_gate_study.csv, forensics_gate_era.csv, forensics_exit_study.csv.

Noise floor (same code on random-walk data): cells of a few hundred rows
reached ±3 pp at 3 months and +8 pp at 6 months; v6 beat v5 by +0.7 pp at 3
months on pure noise. So a gate counts only if it beats "Gate 1 pass" in BOTH
eras and in most years.

## Gate 1 does most of the work

| | 3m | 6m | years 3m > 0 |
|---|---|---|---|
| every stock | 0.00 | 0.00 | — |
| Gate 1 pass | **+1.44** | **+2.45** | 10/12 |
| Gate 1 pass, leader industry | +1.94 | +3.31 | 11/12 |

Everything below is measured against the Gate 1 row.

## v5 vs v6 — v6 holds a modest edge, at 6 months

| | 3m | 6m | 6m mean | n |
|---|---|---|---|---|
| v6 GO | 1.73 | **3.43** | 2.90 | 45,916 |
| v5 GO | 1.92 | 2.82 | 1.66 | 22,110 |
| GO in v6 only | 1.57 | **3.42** | 3.20 | 30,850 |
| GO in v5 only | 1.60 | 1.88 | 0.63 | 7,473 |
| v6 BUY NOW, leader industry | 2.18 | 3.70 (12/12 yrs) | 2.83 | 19,567 |
| v5 BUY NOW, leader industry | 2.26 | 3.31 (10/12 yrs) | 2.00 | 9,218 |

- The two pick the same anchor day only 26% of the time.
- At 3 months they tie (v5 +0.2, inside the noise). At 6 months v6 leads in
  both eras (2016-20: 3.86 vs 3.35; 2021-26: 3.12 vs 2.52), and where they
  disagree the v6-only GOs beat the v5-only GOs by +1.5 pp (median) / +2.6 pp
  (mean). The highest-volume day is the better anchor than the latest one.
- v6 flags twice as many GOs at equal or better quality.
- Exits on BUY NOW entries (prompt rule): v6 +5.45% per trade, v5 +5.08%.

Size of the edge: v6 GO adds ~+0.3 pp (3m) / +1.0 pp (6m) to Gate 1 alone —
real and consistent, not large.

## What each gate adds (v6)

| Gate | Result | Verdict |
|---|---|---|
| GO | +1.73 / +3.43; both eras above Gate 1 | small add, keep |
| VETO | +1.11 / +2.32; below Gate 1 in both eras at 3m | mild caution, not an avoid |
| Bearish / mixed anchor (biggest day closed weak) | 6m +1.35 / +1.61, mean 6m +0.39 / +0.32 vs +1.96 | caution: weaker 6 months |
| NO ANCHOR | +2.40 (2016-20) but +0.39 (2021-26) | unstable, ignore |
| ABSORPTION (down days <= 0.8x median volume) | **+2.73 / +5.22**, mean +3.34 / +3.70, 11/12 yrs; but 2016-20 only +2.09 at 3m and mean -0.71 | best cell, recent-era driven |
| v5 absorption | +2.51 / +3.56; above Gate 1 in both eras (2.11, 2.85) | steadier, smaller |
| DISTRIBUTION | +1.34 / +3.01 (v5 +1.99 / +2.86) | NOT a warning: no worse than Gate 1 |
| Literal "down days 15-40% of anchor volume" | +1.70 / +2.55 | nothing — mechanical (anchor is >= 3x) |
| Shakeout probe | **+2.14 / +4.78**, mean +2.30 / +4.00; both eras above (2.75, 1.76) | promising, keep |
| CLV since anchor >= 0.5 | +1.87 / +2.88 vs < 0.5: +1.64 / +3.56 | nothing |
| Gap on anchor day | +1.95 / +3.36 vs +1.67 / +3.13 | mild +, as the prompt says |
| NEAR OVERHANG | 2wk -0.21, 1m -0.20, 3m +0.81 (both eras below Gate 1), 6m +2.52 | valid TIMING rule: 1-3 months of drag, then catches up |
| CLEAN AIR | = Gate 1 (it is 75% of rows) | no add by itself |
| Stop > 10% | +1.52 / +2.88, mean 6m +4.09 vs +0.81 for tight stops | keep them at half size — do not skip |
| RSI > 70 | +1.76 / +2.61, mean +2.46 / +3.40 | "overbought" is not a negative (as the prompt says) |
| ADX > 25, +DI > -DI | +1.59 / +2.90, 12/12 yrs; ADX < 20: +1.34 / +1.83 | mild +, consistent |
| Bollinger squeeze (< 20th pct) | +1.83 / +2.79 vs wide +1.10 / +2.15 | mild + |
| BUY NOW + absorption | **+3.24 / +5.40**, mean +3.47 / +3.93, 12/12 yrs, n = 2,018 | best combined cell |

## Exits (same entries, per trade, max 126 sessions)

| Rule | Gate 1 pass: mean / vs Nifty / win% | v6 BUY NOW: mean / vs Nifty |
|---|---|---|
| engine: 1 close < MA50 or 15% trail | 4.33 / 3.07 / 36% | 4.93 / 3.66 |
| prompt: 2 closes < MA50 or 15% trail | 4.94 / 3.47 / 38% | 5.45 / 3.99 |
| + initial stop | 4.72 / 3.31 / 37% | 5.26 / 3.85 |
| + 2R target | **2.56 / 1.67** / 40% | **2.86 / 1.95** |
| hold 126 sessions | 14.89 / 9.72 / 61% | 16.55 / 11.22 |

- The 2R target halves the result. Do not take profits at 2R; show T1/T2 as
  reference levels only.
- 2 closes < MA50 beats 1 close by ~+0.5-0.6% per trade here, but the
  portfolio test (exit_rule_multiregime_result.md, B2) found it worse once
  capital is recycled across 15 slots. Event-level and portfolio-level
  disagree, so the engine keeps its rule until a 10-year portfolio run says
  otherwise.
- Holding 126 sessions looks far better per trade but ties up capital ~4x
  longer (0.12%/day vs ~0.15%/day for the exit rules), carries -16% average
  losses, and is flattered most by missing delisted names.

## Use

- Build the Technicals tab on v6 logic.
- Show: Gate 1, v6 state (GO / WAIT / VETO), pullback signature with
  ABSORPTION highlighted, shakeout flag, supply status with the clean-air
  trigger, stop / stop % with "half size" above 10%, T1/T2 as reference, ADX
  and squeeze as context.
- Do not use as signals: DISTRIBUTION as a sell, the literal 15-40% rule, CLV
  since anchor, RSI > 70 as a negative, 2R profit-taking.
- These are event-study medians, not a portfolio. The live-portfolio test of
  breakouts (BREAKOUT_SECTOR_VERDICT.md) did not beat the engine's 6-month RS
  ranking; this tab is for discretionary picks and timing, not an engine rule.
