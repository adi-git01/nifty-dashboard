# Stock breakouts x sub-industry strength -- where the alpha is

Run 29 Sep 2026 (breakout_sector_study.py, via the Sub-Industry Alpha Study
workflow). 2,025 NSE stocks, point-in-time top 1000, 2014-2026. "vs typical" =
median breakout minus the median universe stock over the same window (pp).
Volume anchor = the chart tool's rule: volume >= 3x the 50-session median,
turnover >= Rs 5 cr, CLV >= 0.5.

## The alpha: 52-week / all-time highs in leader industries

| event | n | 2wk | 1mo | 3mo | 6mo | beat typical (3mo) | years 3mo > 0 |
|---|---|---|---|---|---|---|---|
| any stock, leader industry (baseline) | 125k | +0.2 | +0.4 | +0.9 | +1.5 | 52% | 12/13 |
| 52w high, leader industry | 4,317 | +0.9 | +1.4 | +2.2 | +3.5 | 56% | 11/12 |
| 52w high, leader + volume anchor | 2,174 | +1.2 | +1.8 | +2.9 | +4.2 | 56% | 11/12 |
| all-time high, leader + volume anchor | 1,229 | +1.2 | +2.0 | +3.2 | +4.4 | 58% | 10/11 |
| 52w high, laggard industry | 2,095 | +0.3 | +0.6 | +0.6 | +1.9 | 52% | 6/12 |

- The gain builds from 2 weeks to 6 months -- these do not fizzle. Only 8% fall
  back under the 50dma within 2 weeks (51% for any stock); 37% are below entry
  after 3 months (43%).
- Holds in both eras (6-month vs typical, leader + anchor): 52w high +3.4 pp
  in 2016-20 and +5.4 in 2021-26; all-time high +4.2 and +4.6.
- An equal-weight basket of 52w-high / ATH + anchor in leader industries beat
  the average stock by ~6 pp over 6 months.
- The volume anchor adds about +0.7 to +1 pp over 3-6 months. CLV >= 0.8 on top
  adds nothing consistent.
- The same breakout in a laggard industry is worth a third as much, and with
  the anchor + CLV it turns flat or negative.

## Avoid

- 50-dma reclaims: no edge in leader industries (+0.6 pp, below the baseline),
  and a volume anchor makes them WORSE -- leader + anchor -0.9 pp at 3 months,
  laggard + anchor -2.9 pp, 0/13 years positive. A heavy-volume bounce back
  over the 50dma is more often a failed rally than a new trend.
- Any volume anchor in a laggard industry: -1.1 pp at 3 months, 2/13 years.

## Not established

- Golden cross + anchor in leader industries looks strong (+2.3 / +3.5 pp) but
  n = 211 and the eras disagree (-1.0 vs +9.4 pp at 6 months).
- The volume anchor on its own adds nothing over simply being in a leader
  industry (+0.8 vs +0.9 pp at 3 months).

Retention caveat: retention of 80-90% after these events is partly mechanical
(an up-day entry); on random data a 50dma reclaim shows 73%.

Survivorship: delisted stocks are missing, which flatters every row alike; the
comparisons between rows are fair. Costs (~0.4% round trip) are not deducted.

## Live-portfolio test: do not change the engine (run 30 Sep 2026)

breakout_portfolio_backtest.py -- live rules, only the order free slots are
filled changed. Baseline = 6-month RS (live). 10y: +897%, Sharpe 1.63.

| variant | 8 starts beat BASE | 3 periods beat BASE | median d Sharpe (starts) | 10y return |
|---|---|---|---|---|
| leader breakout first | 3/8 | 1/3 | -0.06 | +473% |
| + volume anchor | 2/8 | 2/3 | -0.08 | +483% |
| any breakout first | 0/8 | 0/3 | -0.25 | +399% |
| leader industry first | 5/8 | 1/3 | +0.04 | +948% |
| skip laggard industries | 1/8 | 2/3 | -0.04 | +842% |
| breakout first + skip laggards | 3/8 | 1/3 | -0.04 | +414% |

Forcing breakouts in changes the book (63% of buys vs 7%) and makes it worse.
The event study measured holding a breakout for 3-6 months; the live book
exits on an MA50 break or trailing stop and gives up the slot of a higher
6-month-RS name. The 6-month ranking already captures the momentum these
breakouts carry. "Leader industry first" is the only one close (shallower
drawdowns, +0.04 Sharpe) but fails the periods test (1/3) -- not adopted.

Use the breakout tags as a discretionary watchlist in the Trend Scanner, not
as an engine rule.
