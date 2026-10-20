# What works in which market regime (regime_study.py, 20 Oct 2026)

Weekly stock-dates 2016-2026, point-in-time top-1000 universe, returns vs the typical stock. Regimes
pre-defined two ways (Nifty: price vs MA200 and fall from the high; breadth: % of the universe above its own
200-day average). "Works" = 3m >= +1 pp AND 6m >= +2 pp AND positive in most episodes (>= 3) AND beats the
95th percentile of 200 random same-size baskets from the same regime (nothing passed on random data).

**Today (16 Oct 2026): Nifty MILD BEAR (below its 200-day average); breadth MILD BULL (54% above).** The two
views agree on only 48% of days — an index-weak, broad-market-OK tape, i.e. "choppy" in the everyday sense.

## 6-month median vs the typical stock (pp)
| Nifty regime | BEAR | MILD BEAR | CHOPPY | MILD BULL | BULL |
|---|---|---|---|---|---|
| momentum (6m rank >= 90 + Gate 1) | -7.1 | **+2.6** | **+3.6** | +2.3 | **+6.1** |
| momentum + fresh breakout | -5.1 | +1.4 | **+3.7** | +4.1 | **+6.0** |
| momentum + squeeze release | +9.6 | +4.7 | +2.3 | +3.5 | **+7.5** |
| v6 BUY NOW | -4.2 | +2.0 | **+3.0** | **+2.8** | **+4.1** |
| v6 ABSORPTION | -3.2 | **+6.5** | +4.7 | +3.5 | **+5.4** |
| within 5% of 52w high | -5.0 | **+2.8** | **+3.2** | **+2.7** | **+4.2** |
| > 40% below 52w high | +4.0 | -3.9 | -4.7 | -5.1 | -6.9 |
| 6m losers (rank <= 10) | **+9.6** | -4.2 | -4.5 | -4.6 | -6.6 |
| RSI(14) < 30 | +2.1 | +0.3 | -1.3 | -1.8 | -3.8 |
| valuation CHEAP | +5.7 | -0.6 | -0.5 | +2.3 | -0.8 |
| valuation RICH | -7.4 | **+2.0** | **+2.7** | +0.9 | **+2.6** |
| cheapest 20% vs peers | **+6.2** | -0.8 | -0.4 | +2.1 | +0.1 |
| loss-maker / P/E >= 200 | +2.4 | -1.2 | -2.3 | -3.1 | -2.3 |
| Mid cap | -3.8 | **+2.6** | **+3.3** | **+2.1** | **+2.7** |
| Micro cap | **+2.8** | -2.0 | -2.2 | -0.8 | -1.0 |
Bold = passes the pre-registered rule.

## Findings
1. **Strength works in every regime except a Nifty crash.** Gate 1 trend, near the 52-week high, v6 BUY NOW and
   6-month momentum pass in mild bear, choppy, mild bull and bull on the Nifty view, and in every breadth regime
   (+3 to +8 pp). The edge is largest in bulls (+4-6 pp) and roughly halves in choppy / mild-bear tapes.
2. **The exception is a Nifty BEAR (deep index fall below MA200 — mainly Mar 2020).** Momentum reverses (-7 pp,
   fading momentum -12, rich momentum -19) and losers, deep-drawdown, cheap and micro-cap stocks lead the
   rebound (+3 to +10 pp). Only 3 qualifying episodes — thin evidence, but the classic momentum crash at an
   index bottom. A *breadth* bear (broad small-cap grind, e.g. 2018-19) is different: momentum and BUY NOW led
   (+6.6 / +8.7 pp) — leaders held up while the rest fell.
3. **Valuation: cheap does not work outside the Nifty-bear rebound; expensive quality does.** RICH passes in
   choppy, mild bear and bull; momentum + RICH +5.8 / +7.0 / +6.5 pp in mild bear / choppy / bull. Loss-makers
   lag in every regime except the bear rebound.
4. **Chart signals by regime.** v6 ABSORPTION is the strongest single signal in a mild bear (+6.5 pp). Momentum +
   fresh breakout works in choppy and bull, not in mild bear (+1.4). Breakout + anchor + leader only in a Nifty
   bull. Squeeze release passes in bull (and breadth choppy). Oversold (RSI < 30) never works except at a crash
   bottom. DISTRIBUTION "works" like any GO-state name — confirms it is not a sell signal.
5. **Size: mid caps are the most consistent** (pass in 4 Nifty regimes); micro caps lag everywhere except the bear
   rebound; large caps help only in choppy / mild breadth tapes.
6. **Live engine by regime (annualised while in it):** BEAR -6.6% (Nifty -32%), MILD BEAR -5.3% (Nifty -22%),
   CHOPPY +19% (Nifty +6%), MILD BULL +26% (Nifty +3%), BULL +50% (Nifty +40%). It beats Nifty in every regime;
   in weak tapes it mostly preserves capital (breadth gate + MA50 exits) rather than making money.

## For the current tape (Nifty mild bear, breadth mild bull)
- Keep the engine as is: historically roughly flat in a Nifty mild bear while Nifty falls ~20% a year.
- Discretionary picks: Gate 1 names near their highs, ABSORPTION pullbacks, mid caps, momentum even when
  expensive. Avoid: bottom-fishing (> 40% below high, RSI < 30), cheap-for-its-own-sake, loss-makers, micro caps.
- Do not chase fresh breakouts while Nifty is below its 200-day average (+1.4 pp, fails); they work again in
  choppy / bull tapes.
- If Nifty falls into BEAR (>= 15% off the high, below MA200), the playbook flips toward beaten-down and cheap
  names for the rebound — on thin evidence (3 episodes).

Limits: delisted names missing; Nifty BEAR rests on ~3 episodes; regimes overlap (48% agreement), so treat
the two views as complementary, not as one truth.
