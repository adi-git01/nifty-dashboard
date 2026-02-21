# Alpha Validation Report: Final Analysis

**Date:** February 4, 2026  
**Framework Version:** v1.0  
**Data Source:** yfinance (Free Tier)  
**Period Tested:** 2016-11-30 to 2025-09-10 (~9 Years)  
**Tickers Analyzed:** 29 (Mixed Large/Mid/Small Cap)  
**Total Signal Events:** 11,265  

---

## Executive Summary

We conducted rigorous validation of the previously identified "Smart Money Accumulation" strategy (Trend Score 0-25 + Volume Delta +1 to +3) to determine if the alpha is statistically robust or an artifact of data mining.

### 🎯 Key Verdict

| Dimension | Result | Status |
|:----------|:-------|:-------|
| Works in Bull Markets | -1.7% Return | ❌ FAILS |
| Works in Bear Markets | +14.8% Return | ✅ STRONG |
| Works in Choppy Markets | +5.0% Return | ✅ WORKS |
| Out-of-Sample Holds | +8.6% vs +2.0% | ✅ BETTER |
| Risk-Adjusted (Sharpe > 0.5) | 0.35 | ❌ FAILS |
| Win Rate > 55% | 56% | ✅ PASSES |

**OVERALL: ⚠️ PARTIAL VALIDATION**

The strategy is **NOT a universal contrarian edge**. It is specifically a **Bear Market Recovery Play**.

---

## Data Universe

### Stocks Analyzed
| Category | Tickers | Data Points |
|:---------|:--------|:------------|
| Large Caps | RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, HINDUNILVR, ITC, SBIN, BHARTIARTL, KOTAKBANK | 2,471 days each |
| Mid Caps | TATAPOWER, IRCTC, FEDERALBNK, INDHOTEL, TATACOMM, BALKRISIND, CUMMINSIND, PERSISTENT, COFORGE, MPHASIS | 1,565-2,471 days |
| Small/Volatile | PAYTM, NYKAA, DELHIVERY, POLICYBZR, IDEA, YESBANK, SUZLON, RPOWER, JSWENERGY | 917-2,471 days |

**Excluded:** ZOMATO.NS (delisted/no data)

---

## Test 1: Market Regime Decomposition

**Objective:** Does the "Holy Grail" setup work in ALL market conditions?

### Methodology
```
Bull Regime:  Nifty 50 MA(50) > MA(200) AND MA(50) is rising
Bear Regime:  Nifty 50 MA(50) < MA(200) AND MA(50) is falling
Choppy:       Everything else
```

### Results (Trend 0-25 + Vol Jump Only)

| Regime | Signal Count | Avg Return (60d) | Win Rate | Max Drawdown | Verdict |
|:-------|:-------------|:-----------------|:---------|:-------------|:--------|
| **Bull** | 139 | **-1.7%** | 47% | 20.5% | ❌ UNDERPERFORMS |
| **Bear** | 92 | **+14.8%** | 70% | 16.2% | ✅ STRONG ALPHA |
| **Choppy** | 129 | **+5.0%** | 55% | 19.1% | ✅ MODERATE ALPHA |

### Interpretation

> **Critical Insight:** The "contrarian" strategy of buying oversold stocks with rising volume is NOT contrarian at all—it **only works when the broader market is already weak**.
>
> In Bull markets, oversold stocks (Trend 0-25) are oversold for a reason—they're being left behind by the rally. Buying them leads to underperformance.
>
> In Bear markets, oversold stocks with rising volume are catching the bottom. This is the real edge.

---

## Test 2: In-Sample vs Out-of-Sample

**Objective:** Does the recent period (2021-2025) confirm the earlier period (2016-2021)?

### Data Split
| Period | Date Range | Signals | % of Total |
|:-------|:-----------|:--------|:-----------|
| In-Sample | 2016-11-30 to 2021-08-10 | 5,643 | 50% |
| Out-of-Sample | 2021-08-11 to 2025-09-10 | 5,622 | 50% |

### Holy Grail Performance (Trend 0-25 + Vol Jump)

| Period | N | Avg Return (60d) | Win Rate | Consistency |
|:-------|:--|:-----------------|:---------|:------------|
| In-Sample | 200 | **+2.0%** | 49% | ⚠️ Mediocre |
| Out-of-Sample | 160 | **+8.6%** | 64% | ✅ Strong |

### Interpretation

> **Positive Signal:** The strategy actually performed BETTER in the out-of-sample period. This is unusual—most strategies degrade out-of-sample. However, this may be explained by the 2022 bear market providing ideal conditions.
>
> **Caution:** The in-sample period (2016-2021) was predominantly bullish, which explains the weak +2.0% return.

---

## Test 3: Full Risk-Adjusted Matrix

**Objective:** Map all 16 Trend × Volume combinations with proper risk metrics.

### Definitions
- **Sharpe Ratio:** (Avg Return - Risk-Free Rate) / Std Dev, annualized
- **Calmar Ratio:** Avg Return / Max Drawdown (higher = better risk-adjusted)
- **Grade:** A (Sharpe > 1.0), B (0.5-1.0), C (0-0.5), F (< 0)

### Complete Matrix (60-Day Horizon)

| Trend Band | Volume State | N | Avg Ret | Win Rate | Max DD | Sharpe | Calmar | Grade |
|:-----------|:-------------|:--|:--------|:---------|:-------|:-------|:-------|:------|
| **0-25** | Drop | 444 | 4.5% | 58% | 18.7% | 0.30 | 0.24 | C |
| **0-25** | Flat | 1,191 | 3.8% | 58% | 18.2% | 0.28 | 0.21 | C |
| **0-25** | Jump | 360 | 4.9% | 56% | 18.9% | 0.35 | 0.26 | C |
| **0-25** | Spike | 82 | **10.8%** | 57% | 20.6% | **0.72** | 0.53 | B |
| 25-50 | Drop | 383 | 6.4% | 61% | 15.5% | 0.52 | 0.41 | B |
| 25-50 | Flat | 1,009 | 5.8% | 62% | 15.0% | 0.53 | 0.39 | B |
| 25-50 | Jump | 353 | 6.1% | 62% | 15.4% | 0.52 | 0.39 | B |
| 25-50 | Spike | 63 | 7.2% | 60% | 17.6% | 0.61 | 0.41 | B |
| **50-75** | Drop | 410 | 6.4% | 65% | 14.2% | **0.73** | 0.45 | B |
| 50-75 | Flat | 1,101 | 6.1% | 64% | 14.1% | 0.67 | 0.44 | B |
| 50-75 | Jump | 312 | 5.8% | **67%** | 13.0% | 0.72 | 0.45 | B |
| 50-75 | Spike | 72 | 4.1% | 64% | 14.8% | 0.42 | 0.28 | C |
| 75-100 | Drop | 1,142 | 6.8% | 61% | 13.7% | 0.65 | **0.49** | B |
| 75-100 | Flat | 3,061 | 6.1% | 61% | 13.4% | 0.62 | 0.46 | B |
| 75-100 | Jump | 1,025 | 6.4% | 63% | 13.4% | 0.69 | 0.48 | B |
| 75-100 | Spike | 257 | 5.3% | 58% | 15.3% | 0.50 | 0.35 | B |

### Key Observations

1. **Best Absolute Return:** 0-25 + Spike (+10.8%) — but only 82 signals (low confidence)
2. **Best Risk-Adjusted:** 50-75 + Drop (Sharpe 0.73) — stocks pulling back in an uptrend
3. **Highest Win Rate:** 50-75 + Jump (67%) — continuation breakouts
4. **Lowest Drawdown:** 50-75 + Jump (13.0%) — tight risk profile

---

## Test 4: Holy Grail Validation Scorecard

**Setup:** Trend 0-25 + Volume Jump (+1 to +3)

| Metric | Achieved | Benchmark | Pass/Fail |
|:-------|:---------|:----------|:----------|
| Avg Return (60d) | 4.9% | > 4% | ✅ PASS |
| Win Rate | 56% | > 55% | ✅ PASS |
| Sharpe Ratio | 0.35 | > 0.5 | ❌ FAIL |
| Calmar Ratio | 0.26 | > 0.3 | ❌ FAIL |
| Works in Bull | No | Yes | ❌ FAIL |
| Works in Bear | Yes | Yes | ✅ PASS |
| Out-of-Sample | +8.6% | Positive | ✅ PASS |

**Score: 4/7 Passed**

---

## Revised Strategy Recommendations

Based on the validation results, here are the adjusted trading rules:

### Strategy A: "Bear Market Recovery" (Original Holy Grail, Refined)
- **When to Use:** Only when Nifty 50 MA(50) < MA(200)
- **Entry:** Trend 0-25 + Vol Jump (+1 to +3)
- **Exit:** Day 60 or Trend > 40
- **Expected Return:** +14.8% (Bear), +5.0% (Choppy)
- **DO NOT USE in Bull Markets**

### Strategy B: "Trend Continuation" (Best Risk-Adjusted)
- **When to Use:** Any market regime
- **Entry:** Trend 50-75 + Volume Drop (pullback in uptrend)
- **Exit:** Day 60 or Trend < 40
- **Expected Return:** +6.4%
- **Sharpe:** 0.73

### Strategy C: "Momentum Breakout" (Highest Win Rate)
- **When to Use:** Any market regime
- **Entry:** Trend 50-75 + Volume Jump
- **Exit:** Day 60
- **Expected Return:** +5.8%
- **Win Rate:** 67%

---

## Limitations & Future Work

### Data Limitations
- yfinance free tier provided ~9 years of data (2016-2025)
- Ideal validation would use 20+ years including 2008 crisis
- Survivorship bias: Only current Nifty 500 constituents tested

### Untested Dimensions
- [ ] Walk-forward testing (rolling windows)
- [ ] Exit rule optimization (trailing stop, volume exit)
- [ ] Fundamental overlay (PE < median, ROE > 15%)
- [ ] Liquidity filter (Avg Volume > ₹10 Cr)
- [ ] Transaction cost simulation (0.5-1% slippage)

### Statistical Caveats
- 360 signals in the Holy Grail cell is adequate but not robust
- Multiple testing across 16 cells increases false positive risk
- Bear market regime had only 92 signals (limited sample)

---

## Conclusion

**The "Smart Money Accumulation" strategy is NOT a universal alpha source.**

It is a **regime-dependent strategy** that works exceptionally well in bear markets (+14.8%) but **underperforms in bull markets** (-1.7%).

### Actionable Takeaways

1. **Add a Regime Filter:** Only deploy the 0-25 + Vol Jump strategy when Nifty 50 trend is bearish.
2. **Consider Strategy B:** For all-weather trading, the 50-75 + Drop setup offers better risk-adjusted returns.
3. **Track Regime Daily:** Implement a Nifty 50 MA crossover indicator in your dashboard to toggle strategies.

---

*Report generated by Alpha Validation Framework v1.0*  
*Verified against 11,265 signal events*
