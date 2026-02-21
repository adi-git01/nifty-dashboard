# Behavioral Alpha Analysis Plan (The "Pain" Audit)

**Objective:** Quantify the *psychological difficulty* of holding various alpha strategies. Returns are not enough; we must measure the "Suffer Score" to understand execution probability.

---

## 🏗️ 1. Metric Definitions

### A. The "Path" Metrics
1.  **MAE (Maximum Adverse Excursion):** The deepest point below entry price during the trade.
    *   *Interpretation:* How much heat do you take?
    *   *Ratio:* **MFE/MAE** (Reward-to-Risk during the trade).
2.  **MFE (Maximum Favorable Excursion):** The highest point reached.
    *   *Interpretation:* Did we leave money on the table?
3.  **Drawdown Duration:**
    *   *Metric:* Max days spent in negative territory before recovering.
    *   *Threshold:* % of trades that *never* recover entry price within 60 days.

### B. The "Stress" Metrics
4.  **Ulcer Index:**
    *   Start with equity curve of the trade.
    *   Sum of squared percentage drawdowns.
    *   *Insight:* Penalty for *deep* and *long* drawdowns.
5.  **Volatility of Equity:** Standard deviation of daily changes *within* the trade.

### C. Breakout Quality
6.  **Retest Depth:** For Trend > 80 breakouts, how deep is the retest?
    *   *Low Stress:* Shallow retest (< 3%).
    *   *High Stress:* Deep retest (> 8%).

---

## 🛠️ 2. Execution Engine (`alpha_psychology.py`)

Will extend `alpha_discovery_v2.py` logic but focus on the *daily path* of each signal.

### Core Loop
For each signal event (22,000+):
1.  Extract Day 0 to Day 60 Price Series.
2.  Compute:
    - Min Price (MAE)
    - Max Price (MFE)
    - Days < Entry Price
    - Final Return
3.  Classify "Pain Types":
    - **Smooth Sail:** Never < -2%
    - **Rollercoaster:** < -10% then > +10%
    - **Slow Bleed:** Drift down, never recover
    - **Sudden Death:** Gap down > 10%

---

## 📊 3. Analysis Dimensions

### "Pain vs Gain" Matrix
We will map every Strategy Cell (e.g., Bear + Trend 0-20) to:
- **X-Axis:** Average Return
- **Y-Axis:** Ulcer Index (Pain)
- **Desired Quadrant:** High Return, Low Pain (The "Sleep Well" Zone)

### The "Human Fail Point"
Identify strategies where the MAE typically exceeds -15% before recovering.
*   *Hypothesis:* Humans will sell at the bottom of these, even if the algorithm wins.
*   *Action:* Flag these as "Algo-Only" (not for discretionary checks).

---

## 📝 4. Deliverables

1.  **`alpha_psychology.py`**: The analysis script.
2.  **`pain_report.csv`**: Raw metrics for every strategy cell.
3.  **"The Pain Report"**: A markdown addendum to the Tenets, detailing:
    - The "Easiest" Alpha to trade.
    - The "Hardest" Alpha to trade.
    - MAE Stop-loss rules (e.g., "If down 8%, it's likely dead").
