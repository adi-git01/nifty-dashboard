"""
OptComp V2.2 — Regime Manager
==============================
Production module implementing the 4-tier Market Regime classifier,
regime-adjusted thresholds, circuit breaker, and volume quality gate.

Spec reference:
  CRISIS   if nifty_dd <= -30%
  BEAR     if nifty_dd <= -20%  AND  nifty < nifty_200ma * 0.95
  CAUTION  if nifty_dd <= -10%  OR   |nifty/nifty_200ma - 1| <= 5%
  RECOVERY early reversal override from BEAR/CAUTION on Zweig breadth thrust,
           52W new-lows peak rollover, and FII/DII institutional flow reversal
  BULL     otherwise
"""
from __future__ import annotations

import glob
import os

import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════
# 1. REGIME CLASSIFIER (Step 1 of V2.2 + Early RECOVERY Override)
# ═══════════════════════════════════════════════════════════════
def classify_regime(nifty_close: float, nifty_ma200: float,
                    nifty_52w_high: float, recovery: dict | bool | None = None) -> str:
    """
    Classifies the current market regime using Nifty's relationship
    to its 200DMA and 52-week high, with optional early RECOVERY override
    when capitulation breadth-thrust / new-lows rollover triggers.

    Returns: 'BULL', 'RECOVERY', 'CAUTION', 'BEAR', or 'CRISIS'
    """
    if pd.isna(nifty_close) or pd.isna(nifty_ma200) or pd.isna(nifty_52w_high):
        return "CAUTION"  # safe fallback

    nifty_dd = (nifty_close / nifty_52w_high) - 1  # e.g. -0.25 = 25% drawdown

    # CRISIS: catastrophic drawdown (never overridden)
    if nifty_dd <= -0.30:
        return "CRISIS"

    # BEAR: deep drawdown or >5% below 200DMA
    if (nifty_dd <= -0.20 and nifty_close < nifty_ma200 * 0.95) or nifty_close < nifty_ma200 * 0.95:
        base = "BEAR"
    else:
        # CAUTION: moderate drawdown OR hovering near MA200
        ma200_deviation = abs(nifty_close / nifty_ma200 - 1)
        base = "CAUTION" if (nifty_dd <= -0.10 or ma200_deviation <= 0.05) else "BULL"

    rec_on = bool(recovery.get("active")) if isinstance(recovery, dict) else bool(recovery)
    if rec_on and base in ("BEAR", "CAUTION"):
        return "RECOVERY"
    return base


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
    "RECOVERY": {
        "min_comp_rs":     0.15,    # early thrust leaders rotate before 6m RS catches up
        "min_liquidity":   1.0,     # ₹1 Cr
        "trail_stop":      0.12,    # 12% trail (wider than BEAR 8% to hold reversal leaders)
        "rebalance_freq":  10,      # faster rotation during capitulation reversals
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


# ═══════════════════════════════════════════════════════════════
# 5. BREADTH-THRUST & CAPITULATION REVERSAL DETECTOR (RECOVERY)
# ═══════════════════════════════════════════════════════════════
def detect_capitulation_recovery(
    pct_above_50dma: pd.Series | None,
    pct_52w_lows: pd.Series | None = None,
    fii_net: pd.Series | None = None,
    dii_net: pd.Series | None = None,
    fii_long_pct: pd.Series | None = None,
) -> dict:
    """
    Detects early capitulation reversals before classic 200DMA regimes turn BULL:
      1. 10-session Zweig breadth thrust: % > 50DMA rises >= +25pp from a < 20% oversold base
      2. 52W New Lows peak rollover: 10-session peak >= 10% of universe rolling down >= 40%
      3. Institutional flow / positioning turn: positive FII+DII cash flow (or FII buying)
         or FII index futures long% surging >= +4pp off < 28% extreme short positioning.

    `active` is True when the Zweig breadth thrust fires, OR when 52W New Lows roll over
    alongside institutional flow reversal or >= +10pp 50DMA breadth lift.
    """
    out = {
        "active": False,
        "breadth_thrust": False,
        "lows_rollover": False,
        "inst_flow_reversal": False,
        "abv50": np.nan,
        "min10_abv50": np.nan,
        "d10_abv50": np.nan,
        "lo52w": np.nan,
        "pk10_lo52w": np.nan,
        "rollover_pct": np.nan,
        "fii_net": np.nan,
        "dii_net": np.nan,
        "fii_long_pct": np.nan,
        "reasons": [],
    }

    # 1. 10d Zweig breadth thrust (% > 50DMA)
    if pct_above_50dma is not None:
        s50 = pd.to_numeric(pct_above_50dma, errors="coerce").dropna()
        if len(s50) >= 2:
            win50 = s50.tail(11)
            cur50 = float(win50.iloc[-1])
            min10 = float(win50.iloc[:-1].min())
            d10 = cur50 - min10
            out["abv50"] = round(cur50, 1)
            out["min10_abv50"] = round(min10, 1)
            out["d10_abv50"] = round(d10, 1)
            if min10 < 20.0 and d10 >= 25.0:
                out["breadth_thrust"] = True
                out["reasons"].append(f"Breadth thrust +{d10:.0f}pp ({min10:.0f}%→{cur50:.0f}% >50DMA)")

    # 2. 52W New Lows peak rollover
    if pct_52w_lows is not None:
        slo = pd.to_numeric(pct_52w_lows, errors="coerce").dropna()
        if len(slo) >= 2:
            winlo = slo.tail(11)
            cur_lo = float(winlo.iloc[-1])
            pk10 = float(winlo.max())
            roll = (1.0 - cur_lo / pk10) * 100.0 if pk10 > 0 else 0.0
            out["lo52w"] = round(cur_lo, 1)
            out["pk10_lo52w"] = round(pk10, 1)
            out["rollover_pct"] = round(roll, 1)
            if pk10 >= 10.0 and cur_lo <= 0.60 * pk10:
                out["lows_rollover"] = True
                out["reasons"].append(f"52W lows rollover {pk10:.0f}%→{cur_lo:.1f}% (−{roll:.0f}%)")

    # 3. Daily FII/DII cash flows & FII positioning turn
    cash_turn = False
    f_val = d_val = np.nan
    if fii_net is not None:
        sf = pd.to_numeric(fii_net, errors="coerce").dropna()
        if len(sf):
            f_val = float(sf.iloc[-1])
            out["fii_net"] = round(f_val, 1)
    if dii_net is not None:
        sd = pd.to_numeric(dii_net, errors="coerce").dropna()
        if len(sd):
            d_val = float(sd.iloc[-1])
            out["dii_net"] = round(d_val, 1)
    if np.isfinite(f_val) and np.isfinite(d_val):
        net_inst = f_val + d_val
        if net_inst > 0 and (f_val > 0 or d_val >= 1500.0):
            cash_turn = True
            out["reasons"].append(f"FII+DII cash +₹{net_inst:,.0f}cr")
    elif np.isfinite(f_val) and f_val > 0:
        cash_turn = True
        out["reasons"].append(f"FII cash +₹{f_val:,.0f}cr")

    pos_turn = False
    if fii_long_pct is not None:
        sp = pd.to_numeric(fii_long_pct, errors="coerce").dropna()
        if len(sp) >= 2:
            winp = sp.tail(11)
            cur_p = float(winp.iloc[-1])
            min_p = float(winp.iloc[:-1].min())
            out["fii_long_pct"] = round(cur_p, 1)
            if min_p < 28.0 and (cur_p - min_p) >= 4.0:
                pos_turn = True
                out["reasons"].append(f"FII long% {min_p:.0f}%→{cur_p:.0f}%")

    out["inst_flow_reversal"] = bool(cash_turn or pos_turn)
    d10_val = out["d10_abv50"]
    out["active"] = bool(
        out["breadth_thrust"]
        or (out["lows_rollover"] and (out["inst_flow_reversal"] or (np.isfinite(d10_val) and d10_val >= 10.0)))
    )
    return out


def load_recovery_signals(root: str = ".", as_of=None) -> dict:
    """Compute point-in-time breadth-thrust, 52W-low rollover, and FII/DII flow signals from local NSE data."""
    try:
        p_files = sorted(glob.glob(os.path.join(root, "data", "nse", "prices", "*.parquet")))
        d_files = sorted(glob.glob(os.path.join(root, "data", "nse", "prices", "d", "*.parquet")))
        cutoff = pd.Timestamp(as_of) if as_of is not None else None
        if cutoff is not None:
            # Keep only monthly files up to cutoff month (plus ~16 prior months for 252d rolling lows)
            ym_max = cutoff.strftime("%Y_%m")
            ym_min = (cutoff - pd.DateOffset(months=16)).strftime("%Y_%m")
            p_files = [p for p in p_files if ym_min <= os.path.basename(p)[:7] <= ym_max]
            d_files = [p for p in d_files if os.path.basename(p)[:10] <= cutoff.strftime("%Y-%m-%d")]
        else:
            p_files = p_files[-16:]
            d_files = d_files[-30:]

        pct_abv50 = pct_52w_lo = None
        all_p = p_files + d_files
        if all_p:
            parts = []
            for p in all_p:
                try:
                    parts.append(pd.read_parquet(p, columns=["date", "symbol", "close", "low"]))
                except Exception:
                    continue
            if parts:
                px = pd.concat(parts, ignore_index=True)
                px["date"] = pd.to_datetime(px["date"], errors="coerce")
                px = px.dropna(subset=["date", "symbol", "close"])
                if cutoff is not None:
                    px = px[px["date"] <= cutoff]
                if not px.empty:
                    C = px.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
                    L = px.pivot_table(index="date", columns="symbol", values="low", aggfunc="last").sort_index()
                    keep = C.notna().sum() >= 180
                    if keep.sum() >= 50:
                        C, L = C.loc[:, keep], L.loc[:, keep]
                        ma50 = C.rolling(50, min_periods=40).mean()
                        lo252 = L.rolling(252, min_periods=180).min()
                        n50 = ma50.notna().sum(axis=1).replace(0, np.nan)
                        nlo = lo252.notna().sum(axis=1).replace(0, np.nan)
                        pct_abv50 = (C > ma50).sum(axis=1) / n50 * 100.0
                        pct_52w_lo = (L <= lo252 * 1.01).sum(axis=1) / nlo * 100.0

        fii_s = dii_s = pos_s = None
        mc_path = os.path.join(root, "data", "nse", "market_context.csv")
        if os.path.exists(mc_path):
            mc = pd.read_csv(mc_path)
            if "date" in mc.columns:
                mc["date"] = pd.to_datetime(mc["date"], errors="coerce")
                mc = mc.dropna(subset=["date"]).sort_values("date")
                if cutoff is not None:
                    mc = mc[mc["date"] <= cutoff]
                if not mc.empty:
                    mc_tail = mc.tail(15)
                    # Only treat cash flows as current if printed within the last 3 sessions
                    if "fii_net" in mc_tail.columns and mc_tail["fii_net"].tail(3).notna().any():
                        fii_s = mc_tail["fii_net"].tail(3)
                    if "dii_net" in mc_tail.columns and mc_tail["dii_net"].tail(3).notna().any():
                        dii_s = mc_tail["dii_net"].tail(3)
                    if "fii_idx_long_pct" in mc_tail.columns:
                        pos_s = mc_tail["fii_idx_long_pct"]

        return detect_capitulation_recovery(pct_abv50, pct_52w_lo, fii_s, dii_s, pos_s)
    except Exception:
        return {"active": False, "reasons": []}
