# Morning Pulse, Live Trading Desk, Turnaround Radar: what is worth keeping (14 Nov 2026)

Method: 169 real sessions of snapshots (24 Feb - 13 Nov 2026), split-adjusted. For each signal
list, forward return of the signalled stocks minus the median stock, averaged by signal date; a
re-signal within 21 days counts once (`utils/signal_scorecard.py`, rebuilt every EOD run).
Yardstick: every stock on the Trend Scanner BUY list, every day.

| List | Signals | 21 sessions | Beat median | 63 sessions | Verdict |
|---|---|---|---|---|---|
| Scanner BUY, any day (yardstick) | 57,455 | +1.2 pp | 50% | +4.2 pp | — |
| Turnaround Radar (IAS watchlist) | 564 | -0.3 | 39% | -1.5 | Lags the market |
| …READY tier | 163 | -1.1 | 35% | -1.0 | Lags the market |
| Turnaround Catalyst | 962 | +0.3 | 43% | -0.4 | No edge |
| Earnings Shock | 2,001 | +0.9 | 45% | +3.9 | Like plain momentum |
| RS Divergence | 719 | -1.6 | 44% | too few | Lags the market |
| V2.2 watchlist BUY alert | 810 | +0.6 | 45% | +4.1 | Like plain momentum |
| Morning Pulse top-10 trend-score gainers ("turnaround candidates") | 1,680 | +0.6 | 45% | +1.9 | Like plain momentum |
| Morning Pulse top-10 trend-score losers ("caution / exit watch") | 1,680 | +1.0 | 48% | +3.6 | Like plain momentum |
| Morning Pulse accumulation leaders | 1,690 | +0.1 | 45% | +3.8 | No edge |
| Breakouts, Agreement | 317 / 63 | — | — | — | Too new |

Reading:
- Nothing beats the plain Scanner BUY list. The positive rows are momentum lists; they add nothing.
- Turnaround Radar is the weakest: negative at every horizon, 39% winners, and its higher tiers
  are worse than its lowest. Its "multibagger board" holds 2 doubles out of 564 signals, and the
  largest (CCAVENUE +730%) looks like an unadjusted corporate action.
- Morning Pulse's two mover lists are mislabelled: the "exit watch" losers did better than the
  "turnaround candidate" gainers.
- The earnings-shock "Buy (Drifter)" tag did worse than "Neutral" (-0.7 v +1.4 pp at 21 sessions).

Model portfolio (OptComp V2.2, paper), 27 Feb - 13 Nov: +7.8%, the same as the equal-weight
universe (+7.8%) and ahead of the 50 largest stocks, cap-weighted (+3.4%). 71 closed trades,
41% winners, average win +19.9% v average loss -9.6%, median hold 30 days, average cash 24%.
It sat 93% in cash from late October and missed the November rebound. So far it has matched the
market with similar drawdown (-12.8% v -11.7%); worth tracking, not yet evidence of an edge.

What each page had that is needed daily: the regime and breadth read (Morning Pulse), and
nothing else that is not already better served by Sector Pulse, the Funnel or the Stock Report.

## Change

- **☀️ Daily Brief** (landing page): market regime and breadth, leading / improving / weakening
  industries, today's funnel picks with new entries marked, your positions with a status flag,
  model portfolio against the two benchmarks.
- **🧪 Signal Lab**: the scorecard above, live, and one searchable browser for all the signal logs.
- Morning Pulse, Live Trading Desk and Turnaround Radar are unchanged under a **🗄️ Archive**
  workspace. The pipelines that feed them still run, so the scorecard keeps measuring them.

Next step if the scorecard does not change in a few months: retire the IAS, Turnaround Catalyst
and RS Divergence jobs and their Telegram lines.
