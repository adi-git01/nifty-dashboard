# Backtest Conditions - Derived from Top 50 Winners Analysis

Based on the Strategy Learner v3 analysis of top 50 performers across 3mo, 6mo, and 12mo periods.

---

## STRATEGY 1: SHORT-TERM MOMENTUM (3-Month Hold)

**Objective**: Capture quick momentum bursts with tighter risk management.

### Entry Conditions

| Condition | Threshold | Rationale |
|-----------|-----------|-----------|
| **Momentum Score** | ≥ 8.0 | 3mo winners avg momentum = 8.1 |
| **Trend Score** | ≥ 60 | Most stay in trend (89.5% time) |
| **Volume Ratio** | ≥ 1.2x 20-day avg | Volume confirmation important |
| **Price vs MA50** | Above | 100% of winners crossed above |
| **Quality Score** | ≥ 5.0 | Avg = 6.4, avoid junk |
| **Sector Filter** | Financial, Banking, Metals, Auto | 60%+ of winners |

### Exit Conditions

| Condition | Action |
|-----------|--------|
| **Trailing Stop** | -15% from peak |
| **Time Stop** | 65 trading days (~3 months) |
| **Trend Score Exit** | Below 40 for 5 consecutive days |
| **Volume Collapse** | <0.5x avg for 10 days |

### Expected Performance (Based on Winners)
- Avg Return: **21.8%**
- Max Drawdown: **-16.3%**
- Win Rate: ~60% (volume breakout accuracy)

### Key Metrics to Monitor
```
Sideways < 25% of holding period = positive sign
Early phase return > 10% = strong start
Never hit -20% trailing = 50% of winners
```

---

## STRATEGY 2: MEDIUM-TERM TREND FOLLOWING (6-Month Hold)

**Objective**: Ride established trends with balanced risk/reward.

### Entry Conditions

| Condition | Threshold | Rationale |
|-----------|-----------|-----------|
| **Momentum Score** | ≥ 7.0 | 6mo winners avg = 7.2 (lower than 3mo) |
| **Trend Score** | ≥ 55 | Time in trend = 91.3% |
| **Quality Score** | ≥ 6.0 | Avg = 6.6, slightly higher bar |
| **Growth Score** | ≥ 6.0 | Avg = 6.8 among winners |
| **Volume Breakout** | ≥1.5x on entry day | Vol breakout accuracy = 58.9% |
| **Price Position** | >10% above 200DMA | Strong uptrend confirmation |
| **Super Uptrend Entry** | Score > 80 at least once in past 20 days | Winners average 2.2 super entries |

### Exit Conditions

| Condition | Action |
|-----------|--------|
| **Trailing Stop** | -20% from peak |
| **Time Stop** | 130 trading days (~6 months) |
| **Trend Score Exit** | Below 45 for 10 days |
| **Phase-Based Exit** | Exit if mid-phase return < 15% |

### Expected Performance
- Avg Return: **36.6%**
- Max Drawdown: **-19.1%**
- Recovery Days: **73 days avg**

### Advanced Filters
```python
# Winners profile for 6-month trades:
Entry:
  momentum >= 7.0
  trend_score >= 55
  quality >= 6.0
  growth >= 6.0
  volume_ratio >= 1.5
  dist_200dma > 10%
  
Exit:
  trailing_stop = -20%
  OR trend_score < 45 for 10 days
  OR holding_days >= 130
```

---

## STRATEGY 3: LONG-TERM MULTI-BAGGER (12-Month Hold)

**Objective**: Identify potential multi-baggers and hold through volatility.

### Entry Conditions

| Condition | Threshold | Rationale |
|-----------|-----------|-----------|
| **Trend Score** | ≥ 50 | Time in trend = 76% (lower due to volatility) |
| **Quality Score** | ≥ 5.5 | Avg = 6.1, quality matters |
| **Growth Score** | ≥ 6.5 | Avg = 6.8, growth is key |
| **Momentum Score** | ≥ 6.0 | Lower threshold (avg = 6.8) |
| **Value Score** | ≤ 6.0 | Winners are NOT cheap (avg = 4.6) |
| **Volume Increase** | ≥ 15% vs 3mo avg | Accumulation signal |
| **Sector** | Financial, Metals, Auto, Pharma | Top performing sectors |

### Entry Timing - Wait for Confirmation
```
# The "early phase pain" is real (only 4.3% return in early phase)
# Better entry: Wait for stock to prove itself

Optimal Entry Trigger:
  - Stock has been above MA50 for at least 20 days
  - Had at least 1 "super uptrend" (score >80) reading
  - Current drawdown from recent high < 15%
```

### Exit Conditions

| Condition | Action |
|-----------|--------|
| **Trailing Stop** | -25% from peak (wider!) |
| **Time Stop** | 250 trading days (~12 months) |
| **Fundamental Deterioration** | Quality OR Growth drops by 3+ points |
| **Trend Collapse** | Score < 35 for 15 days |

### Expected Performance
- Avg Return: **69.9%**
- Max Drawdown: **-29.7%**
- Recovery Days: **110 days avg**

### Super Filters for Multi-Baggers
```python
# Profile of 100%+ gainers (top performers):
# Force Motors: 211%, Hindustan Copper: 136%, MCX: 120%

Ultra_Winner_Profile:
  quality >= 8.0  OR  growth >= 9.0
  momentum >= 8.0
  sector in ['Financial Services', 'Metals & Mining', 'Auto']
  time_in_trend >= 80%
  sideways_pct < 10%  # Winners move FAST
  super_uptrend_entries >= 3
```

---

## COMPARATIVE BACKTEST MATRIX

### Run These Three Backtest Configurations:

| Parameter | 3-Month | 6-Month | 12-Month |
|-----------|---------|---------|----------|
| Universe | Nifty 500 | Nifty 500 | Nifty 500 |
| Min Momentum | 8.0 | 7.0 | 6.0 |
| Min Trend Score | 60 | 55 | 50 |
| Min Quality | 5.0 | 6.0 | 5.5 |
| Min Growth | 5.0 | 6.0 | 6.5 |
| Volume Filter | 1.2x | 1.5x | 1.15x |
| Trailing Stop | -15% | -20% | -25% |
| Max Hold Days | 65 | 130 | 250 |
| Position Size | 5% | 4% | 3% |
| Max Positions | 10 | 12 | 15 |
| Rebalance | Weekly | Bi-weekly | Monthly |

---

## TIMING INSIGHTS

### Best Entry Windows

Based on phase analysis:

**3-Month Strategy**:
- Entry can be anytime (returns evenly distributed)
- Early phase delivers +13.7% avg

**6-Month Strategy**:
- Wait for initial shakeout
- Best returns in mid phase (+32%)
- Early phase only +10.7%

**12-Month Strategy**:
- Be patient in first 2 months
- Early phase is nearly flat (+4.3%)
- Mid phase delivers +38.9%
- Consider adding after 1-month hold

### Sector Rotation Signals

Based on winners by sector:

| Period | Top Sectors | Weight |
|--------|-------------|--------|
| 3-Month | Metals, Banking, Financial | 50% allocation |
| 6-Month | Metals, Auto, Financial | 50% allocation |
| 12-Month | Financial, Metals, Auto | 55% allocation |

---

## RISK MANAGEMENT RULES

### Position Sizing by Timeframe

```python
# Conservative position sizing based on drawdown expectations

def position_size(strategy_type, account_value, risk_per_trade=0.02):
    """
    Risk 2% per trade, adjust size based on expected drawdown
    """
    drawdown_map = {
        '3mo': 0.163,  # 16.3% avg max DD
        '6mo': 0.191,  # 19.1% avg max DD
        '12mo': 0.297  # 29.7% avg max DD
    }
    
    max_loss = account_value * risk_per_trade
    position_value = max_loss / drawdown_map[strategy_type]
    
    return min(position_value, account_value * 0.10)  # Cap at 10%
```

### Correlation Management

From sector concentration analysis:
- **Financial/Banking dominates** all timeframes
- Don't have >40% in single sector
- Use 2-3 sector minimum diversification

---

## QUICK REFERENCE CARD

### 3-Month Trade Entry Checklist
- [ ] Momentum ≥ 8.0
- [ ] Trend Score ≥ 60
- [ ] Above MA50
- [ ] Volume > 1.2x avg
- [ ] Quality ≥ 5.0
- [ ] Set -15% trailing stop

### 6-Month Trade Entry Checklist  
- [ ] Momentum ≥ 7.0
- [ ] Trend Score ≥ 55
- [ ] Quality ≥ 6.0
- [ ] Growth ≥ 6.0
- [ ] >10% above 200DMA
- [ ] Set -20% trailing stop

### 12-Month Trade Entry Checklist
- [ ] Growth ≥ 6.5
- [ ] Quality ≥ 5.5
- [ ] Trend Score ≥ 50
- [ ] Had super uptrend reading
- [ ] Above MA50 for 20+ days
- [ ] Set -25% trailing stop

---

## GENERATED FROM ANALYSIS

Data Sources:
- `strategy_learner_v3_3mo_results.csv` - 50 stocks, 3-month winners
- `strategy_learner_v3_6mo_results.csv` - 50 stocks, 6-month winners  
- `strategy_learner_v3_12mo_results.csv` - 50 stocks, 12-month winners
- `strategy_learner_v3_comparative_summary.csv` - Cross-period summary
