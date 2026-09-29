# Transfer of the stock-book learnings -- sub-industries, AI theme, US

Run: Transfer Backtest workflow, 29 Sep 2026 (transfer_backtest.py, all modes).
Each tool ranks its assets every rebalance and holds the top N equal weight.
Keys: the tool's live score, 3/6/12-month RS, random order (8 seeds). Tests: 8
staggered 10-year starts, 3 separate periods, last 1/3/5/10 years. "Gross" is
before costs; a key that wins net but not gross only trades less.

## India sub-industries: keep the live score

| key | 8 starts: beats live net / gross | median Sharpe | turnover |
|---|---|---|---|
| live (CompRS pctile + breadth) | - | 1.23 | 1011% |
| 6-month RS of the group | 2/8 / 0/8 | 1.20 | 558% |
| 12-month RS of the group | 5/8 / 0/8 | 1.21 | 428% |
| random order (median) | | 0.74 | |

The live group score beats the best random seed at 8/8 starts and holds
Sharpe 1.23 vs Nifty 0.74. No longer lookback beats it before costs. Unlike
single stocks, Indian industry momentum is best read short. No change.

## AI capex theme: 12-month RS ranks best, but the basket beats rotating

| key | 8 starts: beats live net / gross | median Sharpe |
|---|---|---|
| live (30% 1w / 50% 1m / 20% 3m) | - | 1.04 |
| 12-month RS | 8/8 / 8/8 | 1.20 |
| equal weight all 53 names | | 1.25 |
| SMH | | 1.08 |
| random order (median) | | 0.93 |

12-month RS beats the live weights at every start, before and after costs
(+0.09 gross, +0.17 net), and in 2 of 3 periods. But holding the whole list
equal weight beats every ranking, and the live weights beat the best random
seed at only 2/8 starts. The list itself -- picked today, with hindsight --
does the work, not the rotation.

## US sector ETFs and S&P 500 sub-industries: rotation loses to SPY

| | live | 12-month RS | equal weight | SPY | random |
|---|---|---|---|---|---|
| sectors, median Sharpe (8 starts) | 0.47 | 0.71 | 0.80 | 0.89 | 0.56 |
| sub-industries | 0.62 | 0.82 | 0.90 | 0.89 | 0.45 |

The live 1-week-heavy score is the worst key in both (for sectors it is below
random) and turns over 734-1237% a year. 12-month RS is the best key, 8/8
starts net, but part of that is lower turnover (gross 8/8 sectors, 5/8
sub-industries). Nothing beats holding SPY.

## What to change

- India sub-industry heatmap: nothing.
- AI theme scanner and US rotation tracker: rank by 12-month RS instead of
  the 30/50/20 blend -- it is the most reliable ordering in every US test.
- Treat US sector / sub-industry and AI rankings as a watchlist, not a
  trading rotation: in none of the US tests did rotating beat holding the
  index or the equal-weight basket.
