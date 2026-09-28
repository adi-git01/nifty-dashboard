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
