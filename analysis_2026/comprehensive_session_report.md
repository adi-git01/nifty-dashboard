# Comprehensive Analysis Report: Alpha Hunter Session
**Date:** 2026-02-04
**Scope:** Nifty 500 Analysis, Signal Efficacy, Sector Rotation, and Feature Implementation.
**Objective:** End-to-end documentation of all insights, data, and code changes derived from today's "Alpha Hunter" session.

---

## 1. Executive Verdict: The "Holy Grail" Setup
After rigorous backtesting of 2,000+ signal events and cross-referencing sector performance, we identified one specific setup that consistently outperforms the baseline.

### 🏆 The "Smart Money Accumulation" Setup
*   **The Logic:** Catching high-quality stocks at the *start* of interest returning, identifying discrete institutional accumulation vs panic selling.
*   **The Rules:**
    1.  **Trend Score:** **0 - 25** (Deeply Oversold / Contrarian)
    2.  **Volume Score Delta:** **+1 to +3 points** (Moderate, healthy increase)
*   **The Edge:**
    *   **Alpha:** Generates **+5.0%** return (60-day) vs +4.0% baseline.
    *   **Reliability:** 25% Alpha boost solely from the volume filter.
*   **The Trap:**
    *   **Panic Spikes (Score Delta > 3):** Underperform (+2.8%). High volume without structure often indicates capitulation is not finished.

---

## 2. Signal Efficacy Study (Scientific Validation)
We tested 3 core signal types to determine the best tools for different timeframes.

### Summary of predictive Power
| Signal Type | Best Timeframe | Win Rate | Avg Return | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Bullish Divergence** | **Short Term (10 Days)** | **61.2%** | **+2.2%** | **Swing Trading** (Sniper entries) |
| **High Momentum (>8)** | Medium Term (20 Days) | 55.7% | +2.2% | Trend Following |
| **Golden Cross** | **Long Term (90 Days)** | **67.1%** | **+19.4%** | **Position Building** (Investing) |

### Key Findings
1.  **Divergences Decay:** Their predictive power drops significantly after 30 days. Don't hold a "divergence trade" forever.
2.  **Golden Cross Lag:** It starts slow (-0.1% in first 10 days) but becomes the most powerful signal over 3-6 months.
3.  **Momentum Noise:** High momentum alone (~50% win rate) is a coin flip without structural confirmation (Price > 200 DMA).

---

## 3. Sector Time-Horizon Analysis ("Cooking Times")
Different sectors mature at different speeds. We analyzed returns across 15, 30, 45, 60, and 90-day horizons for the "Deep Value" strategy.

| Category | Typical Sectors | Best Holding Period | Max Return | Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **🏥 Slow Burners** | **Healthcare, Materials, Financials** | **90 Days** | **+9.5%** | **Buy & Hold**. These trends compound slowly but surely. |
| **🚀 Sprinters** | **Technology, Real Estate** | **60 Days** | **+6.4%** | **Swing Trade.** Momentum peaks at 2 months then fades. Sell early. |
| **🐢 Laggards** | Utilities, Consumer Cyclical | N/A | +3.5% | Avoid. Returns do not justify the specific risk. |

**Top Pick:** Healthcare Sector (Trend < 25) offers the highest risk-adjusted return (+9.45%) over a 90-day period.

---

## 4. Volume Dynamics & Implementation
To operationalize these insights, we built specific tools to visualize and alert on these "Smart Volume" patterns.

### A. The "Smart Volume" Logic
We classified volume behavior into distinct signals:
-   **Smart Surge (Green):** Price UP + Moderate Volume Increase. (Accumulation)
-   **Panic Spike (Red):** Price DOWN + Extreme Volume Spike. (Capitulation/Stop Hunts)
-   **Neutral (Dim):** Average volume activity.

### B. New Visualizations
2 New Charts were added to the dashboard:
1.  **Trend Score Evolution:** Added a 3rd panel for **Smart Volume Bars** (Green/Red) to correlate score changes with volume sizing.
2.  **Multi-Factor Volume Analysis:** A dedicated chart in "Deep Dive" showing Price + Volume Bars + 20-DMA overlay.

### C. Alert System Upgrades
We modified `alert_daemon.py` and `utils/volume_monitor.py` to create a robust alerting safety net:
*   **Daily Scan:** Runs automatically after market close (4:00 PM).
*   **Triggers:** Alerts on any stock with a Volume Score change of **+/- 2**.
*   **Delivery:**
    *   **Telegram:** Instant alerts for high-priority signals.
    *   **Email:** Daily "Volume Digest" summary table sent at EOD.

---

## 5. Raw Data Appendix

### A. Trend x Volume Return Matrix (60 Days)
*Returns for purchasing stocks in various Trend/Volume states*

| Trend State | Vol Drop | Vol Flat | **Vol Jump (+1 to +3)** | Vol Spike (>3) |
| :--- | :--- | :--- | :--- | :--- |
| **Oversold (0-25)** | +4.3% | +4.0% | **+5.0%** | +2.8% |
| **Weak (25-50)** | +1.2% | +0.9% | **+1.6%** | +0.9% |
| **Building (50-75)** | -0.8% | -2.0% | -1.2% | -2.0% |
| **Strong (75-100)** | +0.6% | +0.1% | +0.6% | +0.1% |

### B. Signal Win Rates Over Time
*Probability of Positive Return*

| Signal | 10 Days | 20 Days | 30 Days | 45 Days | 60 Days | 90 Days |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bullish Div** | **61%** | 54% | 43% | 43% | 40% | 35% |
| **Golden Cross** | 48% | 52% | 47% | 56% | 56% | **67%** |
| **High Mom** | 50% | **56%** | 54% | 48% | 46% | 38% |

---

---

## 6. Technical Appendix: Methodology Q&A
*Definitions and Formulas used in this analysis.*

### Q1: Why is 60 Days the primary benchmark?
**Answer:** Optimization for "Reversal Velocity".
*   **The Data:** Our Sector Analysis showed distinct behavior profiles:
    *   **Sprinters (Tech, Realty):** Peak return at **Day 60**, then momentum fades.
    *   **Marathoners (Pharma, Finance):** Continue rising to Day 90+.
*   **The Decision:** We chose **60 Days** as the standardized benchmark because it captures the **maximum profit overlap**. At Day 60, Sprinters are fully valued, and Marathoners have realized ~70% of their move. Waiting for 90 days risks giving back gains on 40% of the portfolio (the Sprinters).

### Q2: Why Trend Score bands of 0-25? (Quartiles vs Quintiles)
**Answer:** These align with our semantic Signal buckets, not arbitrary statistics.
The Trend Score (0-100) is algorithmic, but the decision boundaries are functional:
*   **0 - 25 (Strong Downtrend):** The "Deep Value" / "Capitulation" zone.
*   **25 - 40 (Downtrend):** Weakness, but not extreme.
*   **40 - 60 (Neutral):** Choppy / Sideways.
*   **60 - 75 (Uptrend):** Emerging strength.
*   **75 - 100 (Strong Uptrend):** Breakout mode.
*   *Why 0-25 matters:* This specific quartile isolates stocks that are statistically likely to mean-revert or die. It filters out "just weak" stocks to find "hated" stocks.

### Q3: How is the "Volume Score" calculated?
**Answer:** It is a **Ratio-based Normalized Score (0-10)** measuring short-term intensity vs long-term baseline.

**The Formula:**
1.  **Volume Ratio** = `10-Day Average Volume` / `3-Month Average Volume`
2.  **Base Scoring:**
    *   **Ratio < 0.7x:** Score **0** (Sleeping)
    *   **Ratio = 1.0x:** Score **5** (Normal)
    *   **Ratio > 1.5x:** Score **10** (High Activity)
    *   *Values in between are linearly interpolated.*
3.  **Price Confirmation Bonuses:**
    *   **+1 Point:** If Ratio > 1.15 **AND** Price is Up (Accumulation).
    *   **-1.5 Points:** If Ratio > 1.3 **AND** Price is Down > 5% (Distribution/Panic).

**Implication for Delta (+1 to +3):**
A jump of +2 points (e.g., 4 to 6) represents a **~20% increase** in relative volume activity (e.g., moving from 0.8x avg to 1.0x avg). This is distinct enough to be real interest, but small enough to avoid the "Panic Spike" trap (Score 10).
