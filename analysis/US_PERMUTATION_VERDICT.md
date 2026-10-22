# US permutations: what works (us_permutation_study.py, run of 20 Oct 2026)

**Verdict: no momentum configuration beats SPY out of sample. Valuation is untested in this run:
EDGAR rejected the request (SEC needs a User-Agent with a contact email), so coverage was 0%.**

## Grid: 672 configs, chosen in-sample (Jul 2015 – Dec 2021), tested 2022 – Oct 2026
- In-sample, only 6% of configs beat SPY's Sharpe (0.90).
- Median in-sample Sharpe by setting:
  - key: 3m 0.73 > 6m 0.65 > 1m 0.63 > 12-1 0.55, 1w reversal 0.53, 12m 0.52
  - exit: trail 25% 0.70 ≈ live MA50 0.67 >> no exit 0.47
  - universe: top 200 0.67 > top 1000 0.53
  - gate: breadth 0.67 > SPY > 200d 0.57
  - entry: live filter 0.66 > none 0.56
  - rebalance: weekly 0.64 ≈ monthly 0.60
- Out of sample, SPY earned 11.8% CAGR with Sharpe 0.74. The live rules earned −0.03% with Sharpe 0.11 and a −31% max drawdown.
- **Artifact:** the top-5 in-sample configs used "6m + earnings yield". With no EDGAR data that key was all NaN, so these configs bought one basket and never traded (0 trades). Holding the top-200 names that passed the entry filter earned 19.4% / 0.88 out of sample. It still failed the pass rule, because its 2024-26 Sharpe (1.23) was below SPY's (1.28). The lesson is that rotation, not selection, is what costs money in the US. This is not a strategy.

## Signals (6m median vs the typical stock, 2015-20 / 2021-26)
- Nothing passes in both eras.
- US momentum vs the typical stock was flat to negative in 2021-26 at every lookback.
- Mega caps: 6m momentum +3.4 / +3.4 pp, momentum + squeeze +7.1 / +3.3 pp. Their 3m legs were negative in 2021-26.
- Short-term reversal was negative. "> 40% below the high" flipped from +2.9 to −7.3 pp.
- Near the 52-week high and Gate 1 were mildly positive in 2021-26 (+1.8 / +1.1 pp).

## Shorter frames?
- 3m is the best key in-sample. 1m and 1w reversal add turnover without edge.
- Shorter frames do not rescue the strategy out of sample.

## What to do
- Keep the US Scanner as a watchlist (see US_ENGINE_VERDICT.md), not a rotation book. For US exposure, SPY or holding mega-cap leaders beats rotating.
- To test valuation, add a repo secret `SEC_USER_AGENT` set to `Your Name you@domain`, then rerun `us_perm`.
  The script now drops the earnings-yield key and the valuation filter when EDGAR data is missing, so this artifact cannot recur.
