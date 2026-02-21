"""
Strategy Definitions for Multi-Strategy Backtester
===================================================
5 proposals to backtest and compare:
1. Baseline (pure momentum)
2. Optimized Entry/Exit
3. Multi-Sleeve Strategic (40/40/20)
4. Active Allocation (regime-based)
5. S/R Enhanced Momentum
"""

from typing import Dict, Any

# Rebalancing frequency options (in days)
REBALANCE_FREQUENCIES = {
    "bi-weekly": 14,
    "monthly": 30,
}

# ============================================
# PROPOSAL 1: BASELINE (Control Group)
# ============================================
PROPOSAL_1_BASELINE = {
    "name": "Baseline Momentum",
    "description": "Current pure momentum strategy - control group",
    "strategies": {
        "Momentum": {
            "allocation": 1.0,
            "entry": {
                "trend_score_min": 70,
                "dist_200dma_min": 0,  # Price above 200DMA
            },
            "exit": {
                "trend_score_max": 40,
                "stop_loss_pct": -15.0,
                "trailing_stop_pct": -8.0,
            },
            "portfolio_size": 20,
        }
    }
}

# ============================================
# PROPOSAL 2: OPTIMIZED ENTRY/EXIT (Enhanced)
# ============================================
PROPOSAL_2_OPTIMIZED = {
    "name": "Optimized Entry/Exit v2",
    "description": "Lower hot zone (85), partial profit at +20%, time-based exit",
    "strategies": {
        "MomentumOptimized": {
            "allocation": 1.0,
            "entry": {
                "trend_score_min": 70,
                "trend_score_max": 85,      # Lowered from 90 to avoid "too hot"
                "dist_52w_min": -15.0,      # Not too far from high
                "dist_52w_max": -2.0,       # ATH buffer - not within 2%
                "dist_200dma_min": 0,
                "quality_min": 5.0,         # Filter junk
                "volume_combined_min": 5,   # VPT+AD combined signal >= 5 (neutral+)
            },
            "exit": {
                "trend_score_max": 45,      # Exit on trend breakdown
                "stop_loss_pct": -12.0,     # Tighter stop
                "trailing_stop_pct": -6.0,  # Tighter trailing
                # New exit rules
                "partial_profit_pct": 20.0, # Take partial at +20%
                "partial_sell_pct": 0.33,   # Sell 33% at partial target
                "time_exit_days": 20,       # Exit if stagnant after 20 days
                "time_exit_min_return": 2.0,  # Only time-exit if return < 2%
                "time_exit_max_loss": -5.0,   # Don't time-exit if already losing > 5%
            },
            "portfolio_size": 20,
        }
    }
}

# ============================================
# PROPOSAL 3: MULTI-SLEEVE STRATEGIC (40/40/20)
# ============================================
PROPOSAL_3_MULTISLEEVE = {
    "name": "Multi-Sleeve Strategic",
    "description": "40% GARP, 40% Momentum Breakout, 20% True Value",
    "strategies": {
        "GARP": {
            "allocation": 0.40,
            "entry": {
                "growth_min": 7.0,
                "value_min": 5.0,
                "overall_min": 6.5,
            },
            "exit": {
                "overall_max": 5.0,
                "stop_loss_pct": -12.0,
            },
            "portfolio_size": 8,  # 40% of 20
        },
        "MomentumBreakout": {
            "allocation": 0.40,
            "entry": {
                "dist_52w_min": -15.0,
                "dist_52w_max": -2.0,       # ATH buffer
                "dist_200dma_min": 0,
                "trend_score_min": 60,
                "trend_score_max": 88,      # Not too hot
            },
            "exit": {
                "trend_score_max": 40,
                "stop_loss_pct": -10.0,
                "trailing_stop_pct": -7.0,
            },
            "portfolio_size": 8,
        },
        "TrueValue": {
            "allocation": 0.20,
            "entry": {
                "quality_min": 7.0,
                "value_min": 8.0,
                "trend_score_min": 35,      # Not in freefall
                "trend_score_max": 55,      # Bottoming zone
                "trend_improving": True,    # Turning up
            },
            "exit": {
                "value_max": 5.0,           # Exit when no longer cheap
                "stop_loss_pct": -15.0,
            },
            "portfolio_size": 4,
        }
    }
}

# ============================================
# PROPOSAL 4: ACTIVE ALLOCATION (Regime-Based)
# ============================================
PROPOSAL_4_ACTIVE = {
    "name": "Active Allocation",
    "description": "Shift weights based on market regime (Nifty 50 vs 200DMA)",
    "regime_based": True,
    "regimes": {
        "bull": {   # Nifty 50 > 200DMA
            "GARP": 0.25,
            "MomentumBreakout": 0.55,
            "TrueValue": 0.20,
        },
        "bear": {   # Nifty 50 < 200DMA
            "GARP": 0.50,
            "MomentumBreakout": 0.15,
            "TrueValue": 0.35,
        },
        "neutral": {  # Nifty 50 within ±5% of 200DMA
            "GARP": 0.40,
            "MomentumBreakout": 0.35,
            "TrueValue": 0.25,
        }
    },
    "strategies": {
        "GARP": {
            "entry": {
                "growth_min": 7.0,
                "value_min": 5.0,
                "overall_min": 6.5,
            },
            "exit": {
                "overall_max": 5.0,
                "stop_loss_pct": -12.0,
            },
            "portfolio_size": 8,  # ~40% of 20
        },
        "MomentumBreakout": {
            "entry": {
                "dist_52w_min": -15.0,
                "dist_52w_max": -2.0,
                "dist_200dma_min": 0,
                "trend_score_min": 60,
                "trend_score_max": 88,
            },
            "exit": {
                "trend_score_max": 40,
                "stop_loss_pct": -10.0,
                "trailing_stop_pct": -7.0,
            },
            "portfolio_size": 7,  # ~35% of 20
        },
        "TrueValue": {
            "entry": {
                "quality_min": 7.0,
                "value_min": 8.0,
                "trend_score_min": 35,
                "trend_score_max": 55,
                "trend_improving": True,
            },
            "exit": {
                "value_max": 5.0,
                "stop_loss_pct": -15.0,
            },
            "portfolio_size": 5,  # ~25% of 20
        }
    },
    "portfolio_size": 20,
}

# ============================================
# PROPOSAL 5: MOMENTUM + VOLUME CONFIRMATION
# ============================================
# NOTE: Simplified from S/R to use standard criteria that backtest supports
PROPOSAL_5_SR_ENHANCED = {
    "name": "Momentum + Volume",
    "description": "Momentum with volume confirmation - requires above-average volume and accumulation",
    "strategies": {
        "SRMomentum": {
            "allocation": 1.0,
            "entry": {
                # Standard momentum criteria
                "trend_score_min": 60,
                "trend_score_max": 85,    # Avoid overextended
                "dist_52w_min": -20.0,    # Not too far from high
                "dist_52w_max": -3.0,     # Pullback buffer (not at ATH)
                "dist_200dma_min": 0,     # Above 200DMA
                # Volume requirements (relaxed from 1.3)
                "volume_ratio_min": 1.1,  # At least 10% above average (was 1.3)
                # Removed obv_trend_required - too strict
            },
            "exit": {
                "trend_score_max": 40,    # Exit on trend breakdown
                "stop_loss_pct": -10.0,
                "trailing_stop_pct": -6.0,
            },
            "portfolio_size": 20,
        }
    }
}

# ============================================
# ALL PROPOSALS
# ============================================
ALL_PROPOSALS = {
    "P1_Baseline": PROPOSAL_1_BASELINE,
    "P2_Optimized": PROPOSAL_2_OPTIMIZED,
    "P3_MultiSleeve": PROPOSAL_3_MULTISLEEVE,
    "P4_Active": PROPOSAL_4_ACTIVE,
    "P5_SR_Enhanced": PROPOSAL_5_SR_ENHANCED,
}


def get_proposal(proposal_key: str) -> Dict[str, Any]:
    """Get a proposal by key."""
    return ALL_PROPOSALS.get(proposal_key, PROPOSAL_1_BASELINE)


def get_all_proposal_keys() -> list:
    """Get list of all proposal keys."""
    return list(ALL_PROPOSALS.keys())


def get_proposal_description(proposal_key: str) -> str:
    """Get human-readable description of a proposal."""
    proposal = ALL_PROPOSALS.get(proposal_key, {})
    return f"{proposal.get('name', 'Unknown')}: {proposal.get('description', '')}"
