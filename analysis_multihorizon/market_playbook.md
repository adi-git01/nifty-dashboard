# 📘 The Indian Market Playing Manual v4.0

**A Complete Playbook with Entry, Exit, Time Stops & Pain Management**

---

## 📊 Data Coverage

| Metric | Value |
|:-------|:------|
| **Stocks Analyzed** | 54 Large & Mid Caps |
| **Sectors** | 11 (Banking, IT, Pharma, Auto, Metals, Consumer, Industrials, Energy, Telecom, Realty, Defense) |
| **Time Period** | July 2008 to September 2022 (14 Years) |
| **Market Cycles Covered** | 2008 Crisis, 2011 Correction, 2015-16 Bear, 2018 NBFC Crisis, 2020 COVID Crash, 2022 Rate Hike |
| **Total Signals** | 8,768 multi-horizon events |

---

## ⏱️ TIME STOPS: When to Exit if Stock Won't Move

### The "Dead Money" Problem
Winners typically spend **32-42% of the holding period underwater**. If a stock is underwater for MORE than this, it's likely broken.

### Time Stop Rules by Horizon

| Holding Period | Max Days Underwater | Action |
|:---------------|:--------------------|:-------|
| **90 days** | 45 days (50%) | EXIT - Dead money |
| **180 days** | 100 days (55%) | EXIT - Thesis broken |
| **1 year** | 150 days (60%) | REVIEW - Likely value trap |
| **3 years** | 18 months | EXIT - Structural problem |

### Regime-Specific Time Stops

| Regime | 90d Time Stop | 6mo Time Stop | 1yr Time Stop |
|:-------|:--------------|:--------------|:--------------|
| **Strong Bull** | 40 days | 90 days | 140 days |
| **Mild Bull** | 42 days | 95 days | 145 days |
| **Recovery** | 45 days | 100 days | 135 days |
| **Strong Bear** | 50 days | 110 days | 120 days |

> **Rule:** If your stock hasn't gone green within the time stop period, **SELL**. Capital is precious.

---

## 🔪 FALLING KNIFE RULES: Max Pain Thresholds

### When Pain Becomes Terminal
Based on 14 years of data, here's when a trade is statistically "broken":

### Max Pain (MAE) by Regime

| Regime | 90d Max Pain | 6mo Max Pain | 1yr Max Pain |
|:-------|:-------------|:-------------|:-------------|
| **Strong Bull** | **-15%** | **-18%** | **-22%** |
| **Mild Bull** | **-16%** | **-20%** | **-24%** |
| **Recovery** | **-16%** | **-20%** | **-22%** |
| **Strong Bear** | **-18%** | **-22%** | **-25%** |

> **Rule:** If your stock drops MORE than the Max Pain threshold, **EXIT IMMEDIATELY**. It's a falling knife.

### The "Falling Knife" Detection System

```
IF current_loss > max_pain_threshold:
    → SELL IMMEDIATELY (Don't hope for recovery)
    → This is NOT a dip, it's structural damage

IF current_loss is WITHIN threshold:
    → HOLD (Normal volatility)
    → Check time stop instead
```

---

## 🎯 ENTRY RULES: Complete Checklist

### Before Every Trade

| Step | Check | Action if Failed |
|:-----|:------|:-----------------|
| 1 | Identify current regime | Don't trade if unsure |
| 2 | Check Trend Score | Must match strategy |
| 3 | Check Volume State | Big Drop or Flat preferred |
| 4 | Check sector alignment | Use sector rankings |
| 5 | Set stop loss BEFORE entry | No stop = No trade |
| 6 | Set time stop | Mark calendar |
| 7 | Size position correctly | Max 10-15% |

---

## 🚪 EXIT RULES: Complete Decision Tree

```
EVERY WEEK, CHECK:

1. Has stock hit PROFIT TARGET?
   → YES: Sell 50%, trail stop on rest
   → NO: Continue

2. Has stock hit STOP LOSS (Max Pain)?
   → YES: SELL IMMEDIATELY
   → NO: Continue

3. Has stock exceeded TIME STOP?
   → YES: SELL (Dead money)
   → NO: Continue

4. Has REGIME CHANGED?
   → Bull to Bear: Review all positions
   → Bear to Bull: Lock profits on Bear buys

5. All checks passed?
   → HOLD and check next week
```

---

## 📈 BULL MARKET STRATEGIES (68% of time)

### Strategy 1: Dip Hunter (Mild Bull)
| Parameter | Value |
|:----------|:------|
| **Entry** | Trend 0-20 + Big Drop Vol |
| **Stop Loss** | -16% |
| **Time Stop** | 45 days (90d), 95 days (6mo) |
| **Target** | +22% (90d), +133% (3yr) |
| **Best Sectors** | Consumer, Pharma, IT |

### Strategy 2: Momentum Rider (Strong Bull)
| Parameter | Value |
|:----------|:------|
| **Entry** | Trend 60-80 + Big Drop Vol |
| **Stop Loss** | -15% |
| **Time Stop** | 40 days (90d), 90 days (6mo) |
| **Target** | +14% (90d), +121% (3yr) |
| **Best Sectors** | Metals, Industrials, Auto |

### Strategy 3: Breakout Catcher (Strong Bull)
| Parameter | Value |
|:----------|:------|
| **Entry** | Trend 80-100 + Spike Vol |
| **Stop Loss** | -12% (tight) |
| **Time Stop** | 30 days (must work fast) |
| **Target** | +23% (6mo), +118% (3yr) |
| **Best Sectors** | Defense, Realty, Metals |

---

## 🐻 BEAR MARKET STRATEGY (15% of time but highest alpha)

### Fear Buyer Strategy
| Parameter | Value |
|:----------|:------|
| **Entry** | Trend 0-20 + Jump/Drop Vol |
| **Stop Loss** | -22% (wider due to volatility) |
| **Time Stop** | 60 days (90d), 120 days (6mo) |
| **Target** | +55% (6mo), +146% (3yr) |
| **Best Sectors** | Auto, Defense, Metals |

---

## 📋 QUICK REFERENCE CARD

### The 5 Rules
1. **Entry:** Match setup to regime
2. **Stop Loss:** Exit if loss exceeds Max Pain
3. **Time Stop:** Exit if underwater too long
4. **Target:** Take profits at target
5. **Regime Change:** Re-evaluate all positions

### The Numbers to Remember

| Holding Period | Max Pain | Max Time Underwater |
|:---------------|:---------|:--------------------|
| 90 days | -15% | 45 days |
| 6 months | -20% | 100 days |
| 1 year | -22% | 150 days |
| 3 years | -25% | 18 months |

---

## 📁 Data Files

| File | Location |
|:-----|:---------|
| MASTER_MULTIHORIZON_DB.csv | analysis_multihorizon/ |
| regime_by_horizon.csv | analysis_multihorizon/ |
| sector_by_horizon.csv | analysis_multihorizon/ |
| best_setups.csv | analysis_multihorizon/ |

---

*"Know when to hold. Know when to fold. The data tells you both."*

