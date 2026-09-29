# 2025-26 doublers: setup, fundamentals, ownership

Run 1 of winners_setup_scan.py (29 Sep 2026). 47 stocks in the point-in-time
top 1000 in the first week of 2025 doubled by 17 Sep 2026; 21 of the 30 names on
the 1manfund list are among them. Of the other 9, 4 have no Yahoo data
(GUJGASLTD, SALSTEEL, ITDCEM, RELINFRA) and 5 were below the top 1000 in
January 2025 (WHEELS, INDOBORAX, HAPPYFORGE, UNIPARTS, CENTUM).

## The setup was momentum at a breakout -- and it holds out of period

Launch = the first new 52-week closing high in 2025 after >= 40 sessions without
one. Median winner: +146% overall, +90% of it still ahead at launch.

At launch, winners vs 668 other breakouts in the same window:

| | winners | other breakouts |
|---|---|---|
| 3-month RS vs Nifty | +30 pp | +20 pp |
| 6-month RS vs Nifty | +31 pp | +22 pp |
| % above 200dma | 30% | 21% |
| volume 10d / 120d | 1.8x | 1.4x (not scored) |
| new 3-year high | 60% | 50% |

Those three scored conditions, applied to all 6,708 breakouts 2016-2024:

| conditions met | breakouts | doubled within ~20 months |
|---|---|---|
| 0 | 1,743 | 19.5% |
| 1 | 1,253 | 29.1% |
| 2 | 2,204 | 31.9% |
| 3 | 1,508 | 42.6% |

It holds in both halves: 2+ conditions 38.5% vs 19.1% (2016-20), 34.7% vs
26.7% (2021-24). Median 12-month excess for the top group is only +6 pp (mean
+33 pp), so most still do not double; the payoff comes from the few that do.

This is the momentum the live engine already trades (6-month RS ranking, MA and
CompRS gates). It confirms the approach; it is not a new signal. Two other
tilts are risk rather than edge: the least liquid fifth doubled 37% of the time
vs 19% for the most liquid, and deeper prior falls doubled more often.

## Fundamentals and ownership: no trigger

The winners' last public quarter before launch was all over the place. 49% had
profit growth > 25% YoY, but RBLBANK, CHENNPETRO, ABCAPITAL and SBC launched on
falling profit. Median percentile vs the universe: profit growth 63, FII + DII
change 63, promoter change 29 (promoters trimmed).

Across ~1,000 stocks at four decision dates (Nov 2024 - Aug 2025), no
fundamental or ownership measure sorted the next 12 months: every quintile of
sales growth, profit growth, profit acceleration, FII, DII, promoter and
shareholder-count change sits between -3 and -12 pp excess. FII + DII rising two
quarters running: 4.4% doubled vs 5.8% for the rest. One weak regime (median
stock -6.9 pp vs Nifty), so this says "no edge here", not "never".

Read the FII + DII list (analysis/fii_dii_rising.csv) as information, not a
buy list. Large jumps are often supply, not buying: BAJAJHIND's DII +43 pp
came with a promoter -12 pp (debt converted to lenders); VMM and JSWINFRA are
promoter stake sales.

## Current lists (rerun with the data fix, as of 25 Sep 2026)

Yahoo had 28 Sep closes for only 657 of 2,025 stocks, so the scan uses 25 Sep.
Universe 969 stocks (the first run covered 497).

Setup scan (analysis/setup_scan_now.csv): 90 recent breakouts, 46 meeting all
three conditions -- e.g. WHEELS, OPTIEMUS, FCL, EBGNG, WABAG, VENUSPIPES,
GRANULES, GNA, AARTIPHARM, TEGA. Coiled within 5% of a 52-week high with 3/3:
SUNFLAG, EMUDHRA, TATVA. Historically about 4 in 10 of the 3/3 group doubled
within ~20 months; most did not.

FII + DII rose in each of the last two quarters (to Jun 2026) for 321 of 969;
26 of those are promoter stake moving to institutions (note column), and
BAJAJHIND's +42 pp is debt converted to lenders. Excluding those, FII and DII
both rose for 113, of which 102 also beat Nifty over 6 months (MARKSANS,
STLTECH, SHAILY, ANTHEM, UJJIVANSFB, PARAS, SKYGOLD, LAURUSLABS, TDPOWERSYS ...).
The test above found no edge in rising ownership on its own.
