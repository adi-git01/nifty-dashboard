# 📘 The Indian Market Playing Manual

**A Data-Driven Playbook for Generating Alpha**

*Based on 22,304 signal events | 54 stocks | 11 sectors | 10 years (2017-2025)*

---

## Table of Contents

1. [Market Regimes](#1-market-regimes)
2. [Factor Performance Matrix](#2-factor-performance-matrix)
3. [Sector Playbooks](#3-sector-playbooks)
4. [Risk & Pain Metrics](#4-risk--pain-metrics)
5. [Execution Rules](#5-execution-rules)
6. [Quick Reference Card](#6-quick-reference-card)

---

## 1. Market Regimes

### Definition
| Regime | Condition | Signal Distribution |
|:-------|:----------|:--------------------|
| **Strong Bull** | Nifty > MA50 > MA200, MA50 Rising | 54.0% of signals |
| **Mild Bull** | Nifty > MA200, MA50 Flat | 21.6% of signals |
| **Strong Bear** | Nifty < MA50 < MA200, MA50 Falling | 11.2% of signals |
| **Mild Bear** | Nifty < MA200, MA50 Flat | 10.1% of signals |
| **Recovery** | MA50 crosses above MA200 | 3.1% of signals |

### Performance by Regime (60-Day Returns)
| Regime | Avg Return | Win Rate | Sharpe | Best Trade | Worst Trade |
|:-------|:-----------|:---------|:-------|:-----------|:------------|
| **Strong Bear** | **11.3%** | **77%** | **1.30** | +170% | -41% |
| Mild Bear | 5.9% | 67% | 0.93 | +67% | -43% |
| Recovery | 4.7% | 60% | 0.72 | +108% | -35% |
| Strong Bull | 4.5% | 60% | 0.55 | +147% | -57% |
| Mild Bull | 3.5% | 57% | 0.43 | +137% | -61% |

> **Key Insight:** Alpha concentration is in Bear markets. Bull market "alpha" is mostly beta.

---

## 2. Factor Performance Matrix

### Trend Score × Volume State (60-Day, Strong Bear Only)
| Trend | Volume | Return | Pain (MAE) | Count |
|:------|:-------|:-------|:-----------|:------|
| **0-20** | **Flat** | **15.75%** | **-9.04%** | 490 |
| 0-20 | Drop | 14.61% | -8.03% | 187 |
| 0-20 | Jump | 13.62% | -8.37% | 87 |
| 0-20 | Big Drop | 12.18% | -8.42% | 169 |
| 20-40 | Flat | 14.37% | -8.95% | 231 |
| 40-60 | Jump | 14.40% | -7.57% | 20 |
| 80-100 | Flat | 11.33% | -6.73% | 238 |

### Trend Score × Volume State (60-Day, Strong Bull Only)
| Trend | Volume | Return | Pain (MAE) | Count |
|:------|:-------|:-------|:-----------|:------|
| 80-100 | Drop | 5.66% | -8.43% | 1368 |
| 80-100 | Flat | 5.55% | -9.10% | 2669 |
| 60-80 | Jump | 6.46% | -8.81% | 264 |
| **0-20** | **Flat** | **-0.21%** | **-12.73%** | 511 |

> **Key Insight:** Buying oversold stocks in Bull markets is a losing strategy. Momentum (Trend 80-100) works better.

---

## 3. Sector Playbooks

### Best Regime for Each Sector
| Sector | Best Regime | Best Return | Win Rate | Setup |
|:-------|:------------|:------------|:---------|:------|
| **Defense** | Strong Bear | **24.3%** | 85% | Trend 0-20 |
| **Auto** | Strong Bear | 14.8% | 81% | Trend 0-20 |
| **Telecom** | Strong Bear | 14.5% | 79% | Trend 0-20 |
| **Realty** | Strong Bear | 12.4% | 78% | Trend 0-20 |
| **Metals** | Strong Bear | 11.3% | 74% | Trend 0-20 |
| **Banking** | Strong Bear | 10.8% | 91% | Trend 0-20 |
| **IT Services** | Mild Bear | 9.0% | 72% | Trend 80-100 |
| **Industrials** | Strong Bear | 9.1% | 76% | Trend 0-20 |
| **Consumer** | Strong Bear | 7.4% | 76% | Trend 0-20 |

### Sector Behavior in Bull Markets
| Sector | Strong Bull Return | Win Rate |
|:-------|:-------------------|:---------|
| **Metals** | 7.2% | 62% |
| **Defense** | 7.5% | 56% |
| **Realty** | 6.8% | 61% |
| **IT Services** | 5.1% | 66% |
| **Banking** | 3.2% | 62% |

> **Key Insight:** Defense and Auto are "high beta" sectors—they crash harder in bear markets but recover faster. IT is defensive.

---

## 4. Risk & Pain Metrics

### Average Pain (MAE) by Setup
| Setup | Avg Pain | Interpretation |
|:------|:---------|:---------------|
| Strong Bear + Trend 0-20 + Flat | -9.04% | Moderate stress, recoverable |
| Strong Bull + Trend 0-20 + Flat | -12.73% | High stress, often value trap |
| Strong Bear + Trend 80-100 + Flat | -6.73% | Low stress, but lower return |

### The "12% Rule"
Across all winning setups, the average Maximum Adverse Excursion (MAE) is **-8% to -10%**.
- **Implication:** If a position drops **>12%**, the probability of recovery drops significantly.
- **Rule:** Hard stop at **-12%**.

### Time Underwater
Winning trades in the best setups are typically underwater for **<35%** of the holding period.
- **Implication:** If a trade is red for >40 days (66% of 60), it is likely not recovering.
- **Rule:** Time stop at **60 days**.

---

## 5. Execution Rules

### Entry Checklist
1. ☐ Check Regime: Is Nifty above or below 200 DMA?
2. ☐ If Bear: Scan for Trend 0-20 + Flat/Drop Volume in Defense, Auto, Telecom, Realty.
3. ☐ If Bull: Scan for Trend 80-100 + Flat Volume in Metals, IT, Banking.
4. ☐ Confirm: No earnings event in next 10 days.
5. ☐ Size: Max 10% of portfolio per position. Max 10 positions.

### Exit Rules
| Condition | Action |
|:----------|:-------|
| Position drops **>12%** | Immediate Sell (Stop Loss) |
| Position held for **60 days** | Sell (Time Stop) |
| Position up **>15%** from entry | Trail stop 20% from peak |
| Regime changes from Bear to Bull | Review all Bear Market buys |

---

## 6. Quick Reference Card

### Bear Market Playbook (Nifty < 200 DMA)
| Step | Action |
|:-----|:-------|
| 1 | Identify stocks with Trend Score 0-20 |
| 2 | Filter for Flat or Drop Volume |
| 3 | Prioritize: Defense, Auto, Telecom, Realty |
| 4 | Enter with 10% size, Stop at -12% |
| 5 | Exit at 60 days or +20% trail stop |

### Bull Market Playbook (Nifty > 200 DMA)
| Step | Action |
|:-----|:-------|
| 1 | Identify stocks with Trend Score 80-100 (Breakouts) |
| 2 | Filter for Flat Volume (consolidation before move) |
| 3 | Prioritize: Metals, IT, Banking |
| 4 | Enter with 8% size, Stop at -10% |
| 5 | Exit at 60 days or +15% trail stop |

---

## Appendix: Data Sources

| File | Description |
|:-----|:------------|
| `analysis/MASTER_SIGNAL_DB.csv` | 22,304 individual signals with all metrics |
| `analysis/SECTOR_REGIME_SUMMARY.csv` | Sector × Regime aggregated stats |
| `analysis/FACTOR_PERFORMANCE.csv` | Trend × Volume × Regime aggregated stats |

---

*"The best alpha is the one you can actually hold."*

