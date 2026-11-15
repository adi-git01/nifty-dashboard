# Does the business score predict returns? (15 Nov 2026)

`business_score_backtest.py`. Each 1 July 2021-2025, every stock scored with `fund_trends.read(as_of=31 March)`
(annual data only), then its next-12-month return against the median stock. 852-1,116 stocks a year;
a sixth cohort (July 2026) has 4 months. Rows: `analysis/business_score_backtest.csv`.

Limits: today's universe only (failed companies are missing, which flatters low scores); Screener
figures as restated today; monthly prices start Dec 2020, so five cohorts.

## Result: no. It describes the business; it does not pick winners.

Rank correlation of score with next-year excess return:

| Cohort | Score | Growth | Margins | Cash | Returns |
|---|---|---|---|---|---|
| 2021 | -0.11 | -0.10 | -0.02 | -0.11 | -0.12 |
| 2022 | -0.07 | -0.08 | -0.01 | 0.00 | -0.01 |
| 2023 | -0.09 | -0.07 | -0.03 | -0.04 | -0.10 |
| 2024 | +0.10 | +0.04 | +0.05 | +0.05 | +0.08 |
| 2025 | +0.14 | +0.09 | +0.09 | +0.04 | +0.14 |
| 2026 (4 months) | +0.09 | +0.05 | +0.06 | +0.06 | +0.07 |

Median excess by score quintile, average of the five full cohorts: lowest +1.5 pp, highest -1.6 pp.
2021-23 was a low-quality rally (Weak beat Strong by 6-20 pp); 2024-25 reversed (Strong beat Weak by
5-14 pp). The sign follows the market's appetite, not the score.

- Hard flags (cash conversion under 50%, debt/equity over 1.5) did not predict either: +1.5 pp average,
  positive in 2021-23, negative in 2024-25.
- Direction (improving / steady / slipping): +0.3 / -0.9 / -0.9 pp. Nothing.
- **Investment phase is the one consistent finding**: stocks with a recent large capital addition and
  falling ROCE lagged the median in all five cohorts (-1.2, -10.0, -9.3, -3.4, -1.4 pp; average -5.1 pp;
  44% beat the median). The score rightly does not call these businesses "worsening", but the
  stock has tended to wait.

## What this means for the dashboard

- No weights were re-fitted: there is nothing stable to fit to.
- The funnel's Business stage stays as a quality and risk screen, and is labelled as such.
- The Stock Report shows the investment-phase caution next to the tag.
- The score is most useful in the regime where quality is being rewarded (2024 to date).
