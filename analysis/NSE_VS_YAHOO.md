# NSE store vs Yahoo

Store: sessions to 20 Nov 2026; current: True

## 1 · Price histories (split-adjusted close, last ~14 months)

- 1,170 tickers · NSE has 1,170 · Yahoo has 1,170 · both 1,170
- NSE only: 
- Yahoo only: 
- last close within 0.1%: 99.7% · median daily gap 0.000%
- stocks whose history differs by > 2% on some day: 29 (corporate action handled differently, or Yahoo's dividend adjustment)

| Stock | Max gap | On | Last gap | NSE last | Yahoo last |
|---|---|---|---|---|---|
| DELPHIFX | 222.4% | 2025-10-08 | +0.00% | 2026-11-20 | 2026-11-20 |
| DTIL | 109.5% | 2026-09-18 | +85.14% | 2026-11-20 | 2026-11-20 |
| AHCL | 100.0% | 2025-12-01 | -0.00% | 2026-11-20 | 2026-11-20 |
| NAZARA | 100.0% | 2025-09-23 | +0.00% | 2026-11-20 | 2026-11-20 |
| FCL | 100.0% | 2025-10-10 | +0.00% | 2026-11-20 | 2026-11-20 |
| BHARATRAS | 100.0% | 2025-09-18 | -0.00% | 2026-11-20 | 2026-11-20 |
| AQYLON | 90.0% | 2025-12-03 | +0.00% | 2026-11-20 | 2026-11-20 |
| BANSALWIRE | 80.0% | 2026-04-08 | -0.00% | 2026-11-20 | 2026-11-20 |
| INDIAGLYCO | 78.8% | 2026-07-01 | -0.00% | 2026-11-20 | 2026-11-20 |
| DCMSRIND | 68.3% | 2025-11-21 | +0.00% | 2026-11-20 | 2026-11-20 |
| ALLCARGO | 64.9% | 2025-10-27 | +0.00% | 2026-11-20 | 2026-11-20 |
| VEDL | 64.9% | 2025-12-09 | +0.00% | 2026-11-20 | 2026-11-20 |
| HEG | 62.6% | 2026-07-24 | -0.00% | 2026-11-20 | 2026-11-20 |
| SKFINDIA | 54.8% | 2025-10-01 | +0.00% | 2026-11-20 | 2026-11-20 |
| TRIVENI | 41.6% | 2026-01-20 | +0.00% | 2026-11-20 | 2026-11-20 |
| TMPV | 40.2% | 2025-09-26 | -0.00% | 2026-11-20 | 2026-11-20 |
| GOLDIAM | 33.3% | 2026-03-02 | +0.00% | 2026-11-20 | 2026-11-20 |
| TRENT | 33.3% | 2025-12-16 | -0.00% | 2026-11-20 | 2026-11-20 |
| GENESYS | 30.3% | 2026-04-17 | -0.00% | 2026-11-20 | 2026-11-20 |
| GANGAFORGE | 25.9% | 2026-06-25 | +0.00% | 2026-11-20 | 2026-11-20 |
| HCC | 23.3% | 2025-11-26 | +0.00% | 2026-11-20 | 2026-11-20 |
| ATL | 14.5% | 2025-10-21 | -0.00% | 2026-11-20 | 2026-11-20 |
| CENTEXT | 10.7% | 2026-03-02 | -0.00% | 2026-11-20 | 2026-11-20 |
| CCAVENUE | 5.3% | 2026-02-03 | +0.00% | 2026-11-20 | 2026-11-20 |
| PATELENG | 3.5% | 2025-11-28 | -0.00% | 2026-11-20 | 2026-11-20 |

Fetch time: NSE store 3.5s · Yahoo 171s

## 2 · Engine snapshot built from each

- rows: NSE 1,170 · Yahoo 1,168 · build time NSE 29s · Yahoo 117s · sources used {'nse': 1172, 'yahoo': 1172}
- compared on 1,168 stocks

| Column | Within 1% (or 1 pt) | Median abs diff | Worst |
|---|---|---|---|
| currentPrice | 99.9% | 3.05e-06 | DTIL (308.8 vs 166.8) |
| previousClose | 99.7% | 3.66e-06 | DTIL (317.1 vs 160) |
| fiftyDayAverage | 97.5% | 0.663 | PAGEIND (3.529e+04 vs 3.513e+04) |
| twoHundredDayAverage | 87.5% | 1.14 | INDIAGLYCO (237.4 vs 811.9) |
| fiftyTwoWeekHigh | 79.5% | 0.286 | BHARATRAS (5300 vs 2852) |
| comp_rs | 60.4% | 0.82 | INDIAGLYCO (-11.4 vs -48.11) |
| trend_score | 91.4% | 0 | FMGOETZE (3 vs 65) |
| volatility | 89.2% | 0.2 | BANSALWIRE (32.9 vs 166.3) |
| dist_52w | 83.2% | 0.06 | BANSALWIRE (-25.65 vs -85.13) |
| overall | 100.0% | 0 | INDIAGLYCO (6.6 vs 5.9) |
| volume_signal_score | 94.8% | 0 | EMUDHRA (9 vs 3) |

| Signal | Same on | Different (NSE → Yahoo), examples |
|---|---|---|
| dna_signal | 98.3% | 20: 360ONE HOLD→BUY, AEGISVOPAK BUY→HOLD, ASTERDM BUY→HOLD, BFINVEST HOLD→BUY, COFORGE HOLD→BUY, CPPLUS HOLD→BUY, CSLFINANCE HOLD→WATCH, DYNAMATECH BUY→HOLD |
| trend_signal | 95.6% | 51: AARVI STRONG DOWNTREND→DOWNTREND, ABBOTINDIA NEUTRAL→UPTREND, ABCOTS STRONG DOWNTREND→DOWNTREND, ALLDIGI UPTREND→STRONG UPTREND, AMBIKCO STRONG DOWNTREND→DOWNTREND, ANUHPHR DOWNTREND→UPTREND, ARIHANTCAP UPTREND→STRONG UPTREND, ARTNIRMAN DOWNTREND→NEUTRAL |
| fx_action | 99.3% | 8: ALLDIGI WATCH→WAIT, ASIANPAINT BUY NOW→BUY ON TRIGGER, BANCOINDIA BUY NOW→WATCH, BOROLTD BUY NOW→WATCH, HEG BUY NOW→WATCH, KFINTECH WATCH→BUY NOW, MRPL WATCH→BUY NOW, TATACONSUM BUY ON TRIGGER→BUY NOW |
| fx_state | 98.0% | 23: ADVENTHTL BEAR anchor→VETO, APOLLOPIPE WAIT (running)→VETO, ARCHIDPLY NO ANCHOR→BEAR anchor, AWL FADE→VETO, BAJAJHIND VETO→BEAR anchor, BANCOINDIA GO→MIXED anchor, BOROLTD GO→BEAR anchor, CHALET BEAR anchor→VETO |

Technicals BUY NOW: NSE 86 · Yahoo 85 · both 82 · only NSE ['ASIANPAINT', 'BANCOINDIA', 'BOROLTD', 'HEG'] · only Yahoo ['KFINTECH', 'MRPL', 'TATACONSUM']
