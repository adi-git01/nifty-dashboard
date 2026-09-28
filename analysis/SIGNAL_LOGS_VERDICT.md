# Signal Logs — Alpha Verdict (Earnings Shock, Turnaround Catalyst, RS Divergence)

Data: 151 daily master snapshots, 2026-02-24 -> 2026-09-25, with 11 holiday
copies removed (140 real sessions). Forward returns are fixed-horizon (5/10/21
trading days) from snapshot prices, as EXCESS over the universe median on the
same date — no index series needed. Each signal set is compared with a CONTROL
that differs only in the signal's defining condition. Averages are clustered by
signal date. Returns winsorised at 1/99%.

Caveats that bound every number below:
- One seven-month window, one regime (CAUTION almost throughout).
- 21-day windows overlap heavily; ~6 independent 21d blocks exist. Read 5d and
  10d first; treat 21d t-stats as inflated.
- 2 Jul - 6 Aug the universe was the damaged 777-name list (no RELIANCE, TCS,
  WELCORP...). Names absent then simply drop out.
- The RS log itself had 22 rows, so RS divergence was reconstructed from
  snapshots with an index PROXY (the snapshots carry no index series). The
  proxy validated poorly (2026-05-18: -0.14 vs Nifty -0.95), so red-day
  classification is approximate. Re-run once repair_signal_logs.py has
  backfilled the log with real ^NSEI closes.

## Earnings Shock — the volume condition adds nothing

Date-clustered excess, pp:

| | 5d | 10d | 21d | hit |
|---|---|---|---|---|
| First shock per ticker per 21d (n=1,252) | +0.65 | +1.25 | +1.43 | 47-49% |
| CONTROL: >=4% up day WITHOUT 2.5x volume | +0.65 | +0.97 | +2.05 | 49-52% |

The drift is real but generic: big up days were followed by modest excess
return in this window whether or not volume confirmed them. Means are positive,
MEDIANS negative, hit rates under 50% — a minority of large continuations
carries the average.

The PEAD tag is uninformative-to-inverted: "Buy (Drifter)" 21d +1.14 vs
"Neutral (Priced-In)" +2.14.

## Turnaround Catalyst — no edge; underperforms its own control

| | 5d | 10d | 21d | hit |
|---|---|---|---|---|
| First TC per ticker per 21d (n=591) | -0.40 | -0.61 | +0.45 | 41% |
| CONTROL: >20% off high + >=2.5% up, not flagged | +0.09 | +0.32 | +1.56 | 43-45% |

TC's extra conditions (trend-score jump, volume score) select WORSE names than
the plain base rate at every horizon. No pattern (A, B, A+B, RS21 velocity)
is reliably positive. Consistent with the IAS finding: turnaround screens show
no forward edge in this data.

## RS Divergence — the red-day condition adds nothing

Restricted to names within 5% of the 52w high:

| | 5d | 10d | 21d |
|---|---|---|---|
| SIGNAL: red day, up on the day | +0.36 | +0.97 | +2.41 |
| CONTROL: red day, down with the market | +0.33 | +0.39 | +1.16 |
| CONTROL: GREEN day, up on the day | +1.01 | +1.71 | +2.33 |
| SIGNAL with red day = universe median <= -0.5% | -0.30 | -0.29 | +0.64 |

Being up on a green day did at least as well as being up on a red day, and the
"edge" changes sign under an alternative red-day definition. What survives is
"near the high and rising" — plain momentum / 52-week-high effect — and even
that is concentrated in the second half of the window (any-day near-high 10d:
-0.04 first half, +1.16 second half).

## Bottom line

None of the three logs shows alpha beyond what a simpler control captures. The
one recurring pattern — stocks that are up and near highs keep drifting up over
the following weeks — is this window's momentum regime, not a property of any
scanner, and the 10-year backtest (FACTOR_VERDICT.md) shows CompRS IC ~ 0 over
a full cycle. Do not trade any of the three as a standalone signal.

## Case study: WELCORP (+267%) and WELSPUNLIV (+107%), Feb-Sep 2026

| Tool | WELCORP | WELSPUNLIV |
|---|---|---|
| Trading-engine watchlist BUY alert | **2 Apr @ 846.85** (RS 14.7), 10 calendar days after the 23 Mar low of 771 | **2 Jun @ 143.65** (RS 15.6) |
| OptComp model portfolio | Eligible 16 Apr (#9) and 6 May (#5) but only 2 slots free each time; bought 26 Aug @ 2,306.7, stopped out 15 Sep @ 2,413.3 (+4.6%, 12% CAUTION trail); now 2,832 | Never: below the 17 CompRS floor in June; ranked #65 and #22 when eligible later |
| Earnings Shock | 6 signals from 21 Aug @ 2,311.9 — late | 6 signals from 15 Jun @ 144.9 |
| IAS / Turnaround Catalyst | Never — by design (both require >=20% off the 52w high; WELCORP rode near highs) | Never |
| RS Divergence log | Never — the log was not being written | Never |
| Universe | Absent 2 Jul - 6 Aug (damaged 777-name list) during a +24% leg | Absent same window |

The earliest and best-timed flag was the trading-engine watchlist alert, which
is not a traded strategy. The traded book missed the trend through slot
scarcity (too few free slots on the two rebalances where WELCORP ranked well),
then a universe defect, then caught only the tail and was shaken out by the
CAUTION-regime 12% trail. One name is an anecdote, not a test — but it shows
where the pipeline can lose a winner: slot count, universe integrity, and trail
width in CAUTION.

## Winner profiling — which parameters separate winners from losers?

Method: every parameter in the daily snapshot AS OF the signal date (tape:
CompRS and legs, trend/momentum scores, 52w and 200dma distances, volatility,
volume scores, liquidity, breadth; fundamentals: quality/value/growth/overall,
PE, PB, ROE, ROA, margins, growth, D/E, size, beta; plus each signal's own
inputs). Winners = top 20% by forward excess, losers = bottom 20%. A parameter
counts only if its rank correlation with forward excess clears |rho| >= 0.05
with a date-bootstrap CI excluding zero in the FIRST half of signal dates, then
keeps its sign with |rho| >= 0.03 in the SECOND half. The same test run 60
times on shuffled returns gives the chance baseline. Every surviving cluster is
then re-tested on the signal's control group.

| Signal | Horizon | Survivors | Chance mean / 95th pct |
|---|---|---|---|
| Earnings Shock | 10d / 21d | 2 / 1 | 1.1 / 5 ; 0.8 / 3 |
| Turnaround Catalyst | 10d / 21d | 0 / 6 | 1.7 / 6 ; 1.6 / 5 |
| RS Divergence | 10d / 21d | 2 / 9 | 0.6 / 3 ; 1.2 / 4 |

**RS Divergence — a real pattern, but it is momentum, not divergence.** At 21d
nine trend-strength parameters survive (trend_score, momentum score, 6m and 1y
return, dist to 52w high, dist above 200dma, off 52w low, overall score). The
same parameters predict equally in the control (near-high names up on ANY day):
dist_200dma rho +0.13/+0.09 in both. Filter "trend_score >= median AND
dist_200dma >= median", mean excess pp, pass vs rest:

| | 10d, 1st half | 10d, 2nd half | 21d, 1st half | 21d, 2nd half |
|---|---|---|---|---|
| RS signals | +1.42 vs +0.59 | +1.53 vs +0.60 | +2.30 vs +1.26 | +2.24 vs +1.44 |
| Control, any day | +1.05 vs +0.05 | +1.40 vs +0.79 | +2.06 vs +0.35 | +2.84 vs +1.76 |

Consistent in all eight cells. This is the only pattern in the logs that held
everywhere it was tested — and it is the premise OptComp already trades
(strongest established trends near highs). It says nothing new about red days,
and the 10-year backtest shows CompRS IC ~ 0 over a full cycle, so treat it as
this regime's momentum, not a durable edge.

**Turnaround Catalyst — the quality cluster reverses.** At 21d, growth /
earnings-growth / overall correlate strongly in the first half (+0.24 to +0.31)
and fade in the second (+0.03 to +0.10); a "low volatility AND high growth"
filter flips from +0.36 vs -0.48 to -1.46 vs +1.80. The control shows none of
it. Liquidity (+) and volatility (-) ARE consistent in both halves at both
horizons and absent in the control, but the best slice (liquid AND calm) still
does not beat the control: 10d +0.50 | -0.66 vs control +0.05 | +0.39; 21d
+0.14 | +0.90 vs control +2.37 | +1.12. The best TC names lose less; none win.

**Earnings Shock — no cluster beyond chance; one directional hint.** Winners had
SMALLER jumps (median 5.7% vs 6.6%), lower volume multiples (2.7x vs 3.0x) and
lower volatility than losers. Split at the median jump (~5.8%), moderate shocks
beat large ones in all four half x horizon cells (10d +0.90 vs -0.27 and +2.65
vs +1.32; 21d +1.10 vs -1.45 and +2.87 vs +2.64) — though the last gap is
small. The scanner and its Telegram summary rank shocks by jump size, largest
first, which this suggests is the wrong order. Suggestive, not established.

**Sectors.** Capital Goods, Pharma and Oil & Gas led in all three logs; Real
Estate and Infrastructure lagged in all three. Across three unrelated signals
this is sector leadership in the window, not a property of any signal.

No third holdout exists — both halves were used — so none of the above is
validated for trading. The clean next step is a forward test: log the filter
flags at signal time from now on and read them once enough new events accrue.
