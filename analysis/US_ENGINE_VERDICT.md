# The India momentum engine on US stocks (us_engine_backtest.py, 20 Oct 2026): do not port it

Live engine rules unchanged, SPY in Nifty's place, point-in-time US top 1000 by dollar volume out of 5,465
NASDAQ/NYSE common stocks (2,044 were ever in it), 2014-2026. Delisted names missing.

**Pre-registered rule (Sharpe beats SPY at >= 6/8 starts and 2/3 periods, beats best random at >= 6/8):
every variant fails.**

| | 10y CAGR | 10y Sharpe | 10y max DD | beats SPY (8 starts / 3 periods) | beats best random |
|---|---|---|---|---|---|
| US base (6m RS) | 15.4% | 0.66 | -43% | 0/8, 1/3 | 3/8 |
| India costs | 13.4% | 0.59 | -46% | 0/8, 1/3 | 1/8 |
| no RS floor | 12.7% | 0.58 | -46% | 0/8, 1/3 | 5/8 |
| 12m RS | 9.1% | 0.50 | -50% | 0/8, 1/3 | 0/8 |
| top 500 | 18.5% | 0.75 | -33% | 1/8, 1/3 | 7/8 |
| **SPY buy & hold** | **15.4%** | **0.89** | **-34%** | — | — |
| equal-weight top 1000 | 13.0% | 0.69 | -41% | — | — |

- Matches SPY's return over 10 years with a deeper drawdown; last 5 years -1.6% a year vs SPY +12.6%.
- All of the edge is 2020-22 (+42% a year vs SPY +7%, the post-COVID momentum boom); 2016-19 +6% vs +16%,
  2023-26 +24% vs +22%.
- 37% win rate, exits mostly MA50 "trend breaks" — US large caps whipsaw through the 50-day average far more
  than Indian stocks, and a cap-weighted SPY led by a few mega-caps is a much harder benchmark than Nifty.
- The top-500 variant comes closest (more return than SPY at most starts, but lower Sharpe at 7/8) — not
  enough, and tuning exits on these results would be fitting.

## Use
Keep the US Scanner a watchlist (rank by 12-month RS per TRANSFER_VERDICT.md). For US exposure, an index
holding beat every version of the engine on a risk-adjusted basis. Do not build a US live engine.
