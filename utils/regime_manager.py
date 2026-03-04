"""
OptComp V2.2 — Regime Manager
==============================
Production module implementing the 4-tier Market Regime classifier,
regime-adjusted thresholds, circuit breaker, and volume quality gate.

Spec reference:
  CRISIS   if nifty_dd <= -30%
  BEAR     if nifty_dd <= -20%  AND  nifty < nifty_200ma * 0.95
  CAUTION  if nifty_dd <= -10%  OR   |nifty/nifty_200ma - 1| <= 5%
  BULL     otherwise
"""

import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════
# 1. REGIME CLASSIFIER (Step 1 of V2.2)
# ═══════════════════════════════════════════════════════════════
def classify_regime(nifty_close: float, nifty_ma200: float,
                    nifty_52w_high: float) -> str:
    """
    Classifies the current market regime using Nifty's relationship
    to its 200DMA and 52-week high.

    Returns: 'BULL', 'CAUTION', 'BEAR', or 'CRISIS'
    """
    if pd.isna(nifty_close) or pd.isna(nifty_ma200) or pd.isna(nifty_52w_high):
        return "CAUTION"  # safe fallback

    nifty_dd = (nifty_close / nifty_52w_high) - 1  # e.g. -0.25 = 25% drawdown

    # CRISIS: catastrophic drawdown
    if nifty_dd <= -0.30:
        return "CRISIS"

    # BEAR: significant drawdown + well below MA200
    if nifty_dd <= -0.20 and nifty_close < nifty_ma200 * 0.95:
        return "BEAR"

    # CAUTION: moderate drawdown OR hovering near MA200
    ma200_deviation = abs(nifty_close / nifty_ma200 - 1)
    if nifty_dd <= -0.10 or ma200_deviation <= 0.05:
        return "CAUTION"

    # BULL: everything else (above MA200, close to highs)
    return "BULL"


# ═══════════════════════════════════════════════════════════════
# 2. REGIME-ADJUSTED THRESHOLDS (Step 2 of V2.2)
# ═══════════════════════════════════════════════════════════════
REGIME_PARAMS = {
    "BULL": {
        "min_comp_rs":     0.17,
        "min_liquidity":   1.0,     # ₹1 Cr
        "trail_stop":      0.15,    # 15%
        "rebalance_freq":  13,      # days
        "new_entries":     True,
    },
    "CAUTION": {
        "min_comp_rs":     0.17,
        "min_liquidity":   1.2,     # ₹1.2 Cr
        "trail_stop":      0.12,    # 12%
        "rebalance_freq":  13,
        "new_entries":     True,
    },
    "BEAR": {
        "min_comp_rs":     0.22,
        "min_liquidity":   1.5,     # ₹1.5 Cr
        "trail_stop":      0.08,    # 8%
        "rebalance_freq":  21,
        "new_entries":     True,
    },
    "CRISIS": {
        "min_comp_rs":     0.22,
        "min_liquidity":   2.0,     # ₹2 Cr
        "trail_stop":      0.06,    # 6%
        "rebalance_freq":  999,     # Hold — no rebalance
        "new_entries":     False,   # ❌ Suspended
    },
}


def get_regime_params(regime: str) -> dict:
    """Returns the threshold dictionary for a given regime."""
    return REGIME_PARAMS.get(regime, REGIME_PARAMS["CAUTION"])


# ═══════════════════════════════════════════════════════════════
# 3. CIRCUIT BREAKER (Step 3 of V2.2)
# ═══════════════════════════════════════════════════════════════
def is_circuit_breaker_active(nifty_close: float,
                               nifty_52w_high: float) -> bool:
    """
    Hard gate: fires when Nifty drawdown from 52W high exceeds 25%.
    When active:
      - No new entries this rebalance cycle
      - Existing positions: tighter trail stop applies (from regime)
      - Turnaround module: fully suspended
    """
    if pd.isna(nifty_close) or pd.isna(nifty_52w_high):
        return False
    nifty_dd = (nifty_close / nifty_52w_high) - 1
    return nifty_dd <= -0.25


# ═══════════════════════════════════════════════════════════════
# 4. VOLUME QUALITY GATE (Step 4 of V2.2)
# ═══════════════════════════════════════════════════════════════
def calculate_volume_quality(close: pd.Series, open_px: pd.Series,
                              volume: pd.Series, window: int = 5) -> float:
    """
    Computes the ratio of volume on green days to total volume
    over the last `window` days.

    Applied only in CAUTION/BEAR regime:
      Candidate must have vol_quality >= 0.55
    """
    if len(close) < window:
        return 0.0

    tail = slice(-window, None)
    is_green = close.iloc[tail].values > open_px.iloc[tail].values
    vol_vals = volume.iloc[tail].values

    total = vol_vals.sum()
    if total == 0:
        return 0.0

    up_vol = vol_vals[is_green].sum()
    return float(up_vol / total)
