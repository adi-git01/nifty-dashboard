# NSE store vs Yahoo

Store: sessions to 20 Nov 2026; current: True

## 1 · Price histories (split-adjusted close, last ~14 months)

- 1,170 tickers · NSE has 1,170 · Yahoo has 1,170 · both 1,170
- NSE only: 
- Yahoo only: 
- last close within 0.1%: 99.7% · median daily gap 1.211%
- stocks whose history differs by > 2% on some day: 750 (corporate action handled differently, or Yahoo's dividend adjustment)

| Stock | Max gap | On | Last gap | NSE last | Yahoo last |
|---|---|---|---|---|---|
| DELPHIFX | 1454.6% | 2025-10-08 | +0.00% | 2026-11-20 | 2026-11-20 |
| AHCL | 930.7% | 2026-03-30 | -0.00% | 2026-11-20 | 2026-11-20 |
| V2RETAIL | 924.2% | 2026-01-16 | +0.00% | 2026-11-20 | 2026-11-20 |
| AQYLON | 920.4% | 2026-02-20 | +0.00% | 2026-11-20 | 2026-11-20 |
| FCL | 918.6% | 2025-10-10 | +0.00% | 2026-11-20 | 2026-11-20 |
| ANGELONE | 905.6% | 2026-02-11 | +0.00% | 2026-11-20 | 2026-11-20 |
| AVROIND | 883.5% | 2026-02-05 | -0.00% | 2026-11-20 | 2026-11-20 |
| TATAINVEST | 880.8% | 2025-09-22 | -0.00% | 2026-11-20 | 2026-11-20 |
| ABINFRA | 873.9% | 2025-09-30 | -0.00% | 2026-11-20 | 2026-11-20 |
| WEBELSOLAR | 823.5% | 2025-09-18 | +0.00% | 2026-11-20 | 2026-11-20 |
| ZFCVINDIA | 508.8% | 2025-09-29 | -0.00% | 2026-11-20 | 2026-11-20 |
| CUPID | 412.5% | 2025-11-25 | +0.00% | 2026-11-20 | 2026-11-20 |
| BANSALWIRE | 405.1% | 2026-10-15 | -0.00% | 2026-11-20 | 2026-11-20 |
| AIIL | 405.1% | 2025-12-01 | +0.00% | 2026-11-20 | 2026-11-20 |
| KOTAKBANK | 400.0% | 2026-01-09 | +0.00% | 2026-11-20 | 2026-11-20 |
| BECTORFOOD | 394.9% | 2025-11-12 | +0.00% | 2026-11-20 | 2026-11-20 |
| BUILDPRO | 392.1% | 2026-10-06 | +0.00% | 2026-11-20 | 2026-11-20 |
| ADANIPOWER | 390.7% | 2025-09-18 | +0.00% | 2026-11-20 | 2026-11-20 |
| CAMS | 389.1% | 2025-09-30 | -0.00% | 2026-11-20 | 2026-11-20 |
| AGIIL | 382.4% | 2025-10-06 | +0.00% | 2026-11-20 | 2026-11-20 |
| NUVAMA | 371.4% | 2025-09-24 | -0.00% | 2026-11-20 | 2026-11-20 |
| AJMERA | 354.1% | 2025-09-18 | +0.00% | 2026-11-20 | 2026-11-20 |
| MCX | 345.3% | 2025-12-19 | +0.00% | 2026-11-20 | 2026-11-20 |
| METROPOLIS | 311.3% | 2025-10-07 | +0.00% | 2026-11-20 | 2026-11-20 |
| BHARATRAS | 294.5% | 2025-09-18 | -0.00% | 2026-11-20 | 2026-11-20 |

Fetch time: NSE store 5.7s · Yahoo 175s

## 2 · Engine snapshot built from each

- rows: NSE 1,170 · Yahoo 1,168 · build time NSE 50s · Yahoo 145s · sources used {'nse': 1172, 'yahoo': 1172}
- compared on 1,168 stocks

| Column | Within 1% (or 1 pt) | Median abs diff | Worst |
|---|---|---|---|
| currentPrice | 99.7% | 3.05e-06 | ICICIPRULI (431 vs 462) |
| previousClose | 99.6% | 3.66e-06 | HEG (237.1 vs 264.5) |
| fiftyDayAverage | 54.6% | 3.08 | MRF (1.248e+05 vs 1.237e+05) |
| twoHundredDayAverage | 41.0% | 5.08 | ZFCVINDIA (8340 vs 2409) |
| fiftyTwoWeekHigh | 40.7% | 6.62 | ZFCVINDIA (1.666e+04 vs 3039) |
| comp_rs | 43.6% | 1.19 | CONCORDBIO (-63.87 vs -29.65) |
| trend_score | 74.3% | 1 | METROPOLIS (0 vs 98) |
| volatility | 91.6% | 0.1 | BUILDPRO (173.4 vs 54.2) |
| dist_52w | 44.7% | 1.2 | ANGELONE (-87.67 vs -8.14) |
| overall | 99.8% | 0 | PGIL (7.4 vs 8.8) |
| volume_signal_score | 98.6% | 0 | HEG (2 vs 8) |

| Signal | Same on | Different (NSE → Yahoo), examples |
|---|---|---|
| dna_signal | 97.2% | 33: 63MOONS HOLD→BUY, ANTHEM HOLD→BUY, ANUP HOLD→BUY, AWFIS HOLD→BUY, BANCOINDIA HOLD→BUY, BEARDSELL HOLD→WATCH, BFINVEST HOLD→BUY, BLUSPRING HOLD→BUY |
| trend_signal | 85.3% | 172: 21STCENMGM STRONG UPTREND→UPTREND, AAATECH STRONG DOWNTREND→DOWNTREND, AARTECH STRONG DOWNTREND→DOWNTREND, AARVI STRONG DOWNTREND→DOWNTREND, ABB UPTREND→STRONG UPTREND, ABBOTINDIA DOWNTREND→UPTREND, ABCOTS STRONG DOWNTREND→DOWNTREND, AEGISVOPAK UPTREND→STRONG UPTREND |
| fx_action | 98.8% | 14: ALLDIGI WATCH→WAIT, BELRISE WATCH→WAIT, BHARTIHEXA WATCH→BUY NOW, CANBK BUY ON TRIGGER→BUY NOW, CENTUM WATCH→BUY NOW, CUB WATCH→WAIT, CUPID WATCH→BUY NOW, DIXON BUY NOW→WATCH |
| fx_state | 98.5% | 18: ACC VETO→GO, ACCELYA VETO→GO, ALIVUS VETO→FADE, BAYERCROP MIXED anchor→BEAR anchor, BLSE WAIT (running)→MIXED anchor, CGPOWER FADE→WAIT (40-80%), CIEINDIA FADE→WAIT (40-80%), CONCORDBIO BEAR anchor→VETO |

Technicals BUY NOW: NSE 77 · Yahoo 85 · both 76 · only NSE ['DIXON'] · only Yahoo ['BHARTIHEXA', 'CANBK', 'CENTUM', 'CUPID', 'FILATEX', 'HDFCAMC', 'MRPL', 'ONESOURCE', 'PGIL']
