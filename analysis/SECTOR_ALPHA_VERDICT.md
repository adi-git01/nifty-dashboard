# Sub-industry ranking: is there alpha, and how long does a lead last?

Run: Sub-Industry Alpha Study, 29 Sep 2026 (sector_alpha_study.py). The heatmap's
score_0_100 rebuilt daily 2014-2026 for 58 sub-industries (rank corr 0.78 and
same colour 69% vs the stored live history since Apr 2025). Weekly checkpoints;
returns vs Nifty and vs the average sub-industry on the same day.

## 1. The edge is in avoiding laggards, not in chasing leaders

Median forward return vs the average sub-industry, by score:

| score | 1 month | 3 months | 6 months | beat avg group (3m) |
|---|---|---|---|---|
| 0-20 | -0.9 pp | -2.1 pp | -3.5 pp | 39% |
| 20-40 | -0.5 | -1.1 | -1.9 | 43% |
| 40-60 | -0.3 | -0.4 | -1.1 | 47% |
| 60-80 | +0.1 | -0.2 | -0.6 | 48% |
| 80-100 | +0.1 | +0.4 | +0.4 | 51% |

Same shape in 2016-20 and 2021-26. The bottom fifth lags about 60% of the
time; the top fifth is a coin flip. (Most medians are below zero because the
average is pulled up by a few big winners.)

## 2. How long a lead pays

Rank correlation between today's score and the 1-month return that starts
later: 0.077 now, 0.043 after 1 month, 0.02-0.04 from 3 to 9 months, -0.024
after 12 months. Most of the information is used up in the first month; after
a year leaders slightly lag.

Leadership (score >= 70) spells: median 2 weeks, 90% over within 9 weeks,
longest 41. Laggard spells: median 2 weeks, 90% within 11. Fresh and 3-month
leaders perform alike; the few leading 26+ weeks (n = 78) lag by 1.8 pp over
the next 3 months.

## 3. No turnarounds

Laggards keep lagging at every age: -1.3 pp (3m) after 1 week in red, -1.9 pp
after 13-26 weeks, -3.3 pp after 26+ weeks. No sub-industry rebounded from
red by > 2 pp in both eras (0 of 51; 0.9 expected by chance).

## 4. No reliable sub-industry character

Rank correlation of each group's statistics between 2016-20 and 2021-26 is
~0 (persistence +0.09, when leading +0.03, when lagging +0.03). Five groups
kept leads going in both eras -- Pharma & Biotech, Aerospace & Defense,
Diversified Metals, IT Software, Fertilizers & Agrochemicals -- against 2.5
expected by chance. Treat that as a watchlist hypothesis, not a pattern.

## 5. Colour changes add nothing beyond the level

Upgrades do not help: red -> green -1.5 pp over 3 months (no better than
staying red, -1.6); yellow -> green -0.1 pp. Only green -> green is positive
(+0.3 pp). The colour's level carries the information, not the change.

## Use

- As a filter: avoid new buys in bottom-fifth sub-industries (score < 20).
  Untested on the stock book -- a strong stock in a weak industry may be fine,
  so it needs the same backtest as any other rule before going live.
- Do not rotate into industries because they turned green, and do not buy
  laggard industries expecting a rebound.
