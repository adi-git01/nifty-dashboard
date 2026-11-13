# Investment funnel, industry leadership and the stock report (13 Nov 2026)

## 1. Sub-industry table: what the audit found

The stored rotation score (`trading_engine.generate_sub_industry_rotation`) is 70% average member
composite-RS percentile + 30% share of members with RS > 0, ranked 0-100. Findings on 196 dates:

- It measures short-term relative strength only. It does not say whether an industry is in an
  uptrend, how broad the move is, or whether it is gaining or losing ground.
- It is noisy: median daily change 4 points; 1.55 of the top 10 change every day.
- 8 of 53 industries have fewer than 6 members; one stock moves them.
- The `members` column is empty on 6,668 older rows (cosmetic; history charts do not use it).
- Forward 21-session median member return vs all stocks, 147 start dates, Feb-Oct 2026:

| Ranked by | Top 20% | Bottom 20% | Monthly IC (months positive) |
|---|---|---|---|
| Stored score | +0.53 pp | +0.26 | 0.038 (5/7) |
| Median 3-month RS | +0.68 | -0.01 | 0.068 (5/7) |
| Leadership blend (below) | +0.63 | — | 0.064 (5/7) |

All small, one regime, seven independent months: a map of strength, not a tested edge.

**Change.** Sector Pulse now opens with an Industry Leadership Board (`utils/industry_board.py`),
computed from the daily snapshot, so it needs no new pipeline:
score = rank of 35% median 3-month RS + 25% share above the 200-day + 20% share above the 50-day
+ 20% share within 10% of the 52-week high; change vs ~21 sessions ago; trend status (the
industry-alert ladder); share on BUY; strongest members; drill-down to the stocks.
Groups: Leading (score 60+, at least half the members above the 50-day, not down 10+),
Weakening (same but down 10+), Improving (not a leader, up 10+), Lagging. The absolute breadth
test matters: a relative score alone calls the least-bad industries "leaders" in a weak market.
The old matrix stays below it as the history view.

## 2. Business trends (`utils/fund_trends.py`)

Per stock, from the weekly Screener files: yearly sales growth, OPM, net margin, ROCE / ROE, cash
conversion (CFO / operating profit, 1-year and 3-year cumulative), debt / equity, interest cover,
fixed-asset turnover, capex; quarterly sales, profit, margin and ownership. Each metric is read as
IMPROVING / STABLE / WORSENING (last 3 years vs the 3 before) with a one-line reason and an
overall verdict. Lenders skip the cash, debt and working-capital tests.

The scraper now also collects debtor, inventory, payable, cash-conversion-cycle and
working-capital days, and quarterly operating profit / OPM. They appear after the next weekly
fundamentals run; until then the report says they are not on file.

## 3. The funnel (`utils/funnel.py`, page: Market Specs -> Investment Funnel)

| # | Stage | Pass test | Evidence |
|---|---|---|---|
| 0 | Market | Regime and risk scale set position count and size; not a stock filter | Momentum lagged only in BEAR (REGIME_VERDICT) |
| 1 | Industry | Leading / Improving, or Uptrend / Turning up | Small, 9 months |
| 2 | Trend | Scanner BUY and not a Stage 3 / 4 chart | RS + Gate 1: 10 years (FACTOR, FORENSICS verdicts) |
| 3 | Business | Trends not worsening; 3-year cash conversion >= 50%; debt / equity <= 1.5 | Judgement, untested |
| 4 | Setup | Technicals BUY NOW / ON TRIGGER or a SIMBA signal within 5 sessions, stop within 15% | Stops tested; SIMBA untested |
| 5 | Value | Price in the lower 70% of the value range; off by default | Expensive momentum beat cheap (ENTRY_EXIT_VERDICT) |
| 6 | Plan | Entry, stop, targets, size from the stop (Stock Report) | — |
| 7 | Track | Positions, RS / price alerts, Agreement log | — |

On 12 Nov 2026: 1,168 -> 520 -> 97 -> 57 -> 18 stocks; 11 with Value on.
Near misses (fail exactly one stage) are listed separately; most are waiting for an entry.

**The funnel as a whole is not backtested.** It orders tools whose individual evidence ranges from
ten years to none. To test it, log the daily pass list and measure forward returns against the
near-miss list; that needs a few months of snapshots.

## 4. Stock report (`utils/stock_report.py`, page: Analysis Lab -> Stock Report)

One screen in funnel order: verdict, checklist, market and industry, trend and chart, entry plan
(stop, targets, size at 1% risk), business trends with charts and tables, valuation range.
Built only from stored data, so it opens in about a second; downloadable as Markdown.
Verdicts: ACTIONABLE, ACTIONABLE — WEAK INDUSTRY, WATCHLIST, WAIT FOR A TIGHTER ENTRY, CAUTION,
NOT NOW, EXIT / AVOID.
