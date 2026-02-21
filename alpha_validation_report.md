# Alpha Validation Report

**Generated:** 2026-02-04 23:46
**Data Source:** yfinance (free tier)
**Data Period:** 2016-11-30 to 2025-09-10
**Tickers Analyzed:** 29
**Total Signals:** 11265

> [!WARNING]
> **Data Limitation:** yfinance free tier provided 3206 days of history.
> For full 2015-2019 out-of-sample testing, a premium data source would be required.

---

## Test 1: Market Regime Decomposition

*Does the "Holy Grail" (Trend 0-25 + Vol Jump +1 to +3) work in ALL regimes?*

| Regime | Signals | Avg Return (60d) | Win Rate | Avg Max DD | Verdict |
|:-------|:--------|:-----------------|:---------|:-----------|:--------|
| Bull | 139 | -1.7% | 47% | 20.5% | ❌ Fails |
| Bear | 92 | 14.8% | 70% | 16.2% | ✅ Works |
| Choppy | 129 | 5.0% | 55% | 19.1% | ✅ Works |

---

## Test 2: In-Sample vs Out-of-Sample

| Period | Date Range | Holy Grail Signals | Avg Return | Win Rate |
|:-------|:-----------|:-------------------|:-----------|:---------|
| In-Sample | 2016-11-30 - 2021-08-10 | 200 | 2.0% | 49% |
| Out-Sample | 2021-08-11 - 2025-09-10 | 160 | 8.6% | 64% |

---

## Test 3: Full Risk-Adjusted Matrix (60-Day Horizon)

| Trend | Volume | N | Avg Ret | Win Rate | Max DD | Sharpe | Calmar | Grade |
|:------|:-------|:--|:--------|:---------|:-------|:-------|:-------|:------|
| 0-25 | Drop | 444 | 4.5% | 58% | 18.7% | 0.30 | 0.24 | C |
| 0-25 | Flat | 1191 | 3.8% | 58% | 18.2% | 0.28 | 0.21 | C |
| 0-25 | Jump | 360 | 4.9% | 56% | 18.9% | 0.35 | 0.26 | C |
| 0-25 | Spike | 82 | 10.8% | 57% | 20.6% | 0.72 | 0.53 | B |
| 25-50 | Drop | 383 | 6.4% | 61% | 15.5% | 0.52 | 0.41 | B |
| 25-50 | Flat | 1009 | 5.8% | 62% | 15.0% | 0.53 | 0.39 | B |
| 25-50 | Jump | 353 | 6.1% | 62% | 15.4% | 0.52 | 0.39 | B |
| 25-50 | Spike | 63 | 7.2% | 60% | 17.6% | 0.61 | 0.41 | B |
| 50-75 | Drop | 410 | 6.4% | 65% | 14.2% | 0.73 | 0.45 | B |
| 50-75 | Flat | 1101 | 6.1% | 64% | 14.1% | 0.67 | 0.44 | B |
| 50-75 | Jump | 312 | 5.8% | 67% | 13.0% | 0.72 | 0.45 | B |
| 50-75 | Spike | 72 | 4.1% | 64% | 14.8% | 0.42 | 0.28 | C |
| 75-100 | Drop | 1142 | 6.8% | 61% | 13.7% | 0.65 | 0.49 | B |
| 75-100 | Flat | 3061 | 6.1% | 61% | 13.4% | 0.62 | 0.46 | B |
| 75-100 | Jump | 1025 | 6.4% | 63% | 13.4% | 0.69 | 0.48 | B |
| 75-100 | Spike | 257 | 5.3% | 58% | 15.3% | 0.50 | 0.35 | B |

---

## Key Findings

### 🏆 Best Risk-Adjusted Setup
- **50-75 Trend + Drop Volume**
- Sharpe: **0.73**
- Avg Return: 6.4%
- Win Rate: 65%

### Holy Grail Validation (Trend 0-25 + Vol Jump)
| Metric | Value | Benchmark | Verdict |
|:-------|:------|:----------|:--------|
| Avg Return | 4.9% | >4% | ✅ |
| Win Rate | 56% | >55% | ✅ |
| Sharpe | 0.35 | >0.5 | ❌ |
| Calmar | 0.26 | >0.3 | ❌ |

---

## Conclusion

**VERDICT: ⚠️ PARTIAL VALIDATION**

The strategy shows promise but requires additional data (longer history) or refinement to confirm robustness.
