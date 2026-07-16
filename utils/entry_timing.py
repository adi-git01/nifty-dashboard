"""
Entry Timing & Position Sizing helpers.

Answers the "this leader has already run up — is it still a good entry, and
how much should I buy?" problem for momentum picks. Everything is derived
from fields already in the daily cache (price, fiftyDayAverage, the RS
components, volatility), so it runs live in the UI with no extra fetch.

Two ideas:

1. Entry freshness — how *stretched* a name is vs its 50-day MA, combined
   with whether its relative strength is still *accelerating* or *fading*.
   Calibrated on the live universe: uptrends sit a median +6% above MA50,
   90th percentile +18.7%, so >18% = genuinely extended (top decile). E.g.
   AEGISLOG at +41%/decelerating flags Extended (poor entry) while GARFIBRES
   at +12%/accelerating flags Actionable — matching how a discretionary
   momentum trader would rank them.

2. Stop-distance position sizing — the further a name is above its MA50 (or
   the more volatile it is), the wider its logical stop, so the smaller the
   position for a fixed rupee risk. This lets you own an extended leader
   without over-committing to a chase, and naturally up-sizes the fresher,
   near-support entries. Capped at equal-weight so no single name dominates.
"""
import numpy as np
import pandas as pd

DEFAULT_MAX_POSITIONS = 15

# Freshness / label thresholds (percent above the 50-day MA), calibrated to
# the live universe distribution.
_PULLBACK_MAX = 6.0     # <= this above MA50 (incl. below) = near-support entry
_EXTENDED_MIN = 18.0    # > this above MA50 = stretched / chase risk
_FADING_ACCEL = -6.0    # RS decelerating harder than this = momentum rolling over


def _clip(s, lo, hi):
    return np.minimum(np.maximum(s, lo), hi)


def _col(df: pd.DataFrame, *names) -> pd.Series:
    """Return the first present column (numeric-coerced), else an all-NaN
    series aligned to df.index. Guards against missing legacy-cache columns."""
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors='coerce')
    return pd.Series(np.nan, index=df.index, dtype='float64')


def add_entry_freshness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds: dist_ma50, rs_accel, freshness (0-100), entry_label.
    Non-destructive (returns a copy). Safe on missing/NaN inputs.
    """
    out = df.copy()
    if out.empty:
        for c in ('dist_ma50', 'rs_accel', 'freshness'):
            out[c] = pd.Series(dtype='float64')
        out['entry_label'] = pd.Series(dtype='object')
        return out

    price = _col(out, 'price', 'currentPrice')
    ma50 = _col(out, 'fiftyDayAverage')
    with np.errstate(divide='ignore', invalid='ignore'):
        dist_ma50 = (price - ma50) / ma50 * 100.0
    dist_ma50 = dist_ma50.where(ma50 > 0)
    out['dist_ma50'] = dist_ma50.round(1)

    rs_1w = _col(out, 'rs_1w')
    rs_1m = _col(out, 'rs_1m')
    # Recent weekly RS vs the average weekly RS over the last month.
    # Positive => strength is speeding up; negative => fading despite a big
    # trailing 3M number (the classic "late" tell).
    rs_accel = rs_1w - rs_1m / 4.0
    out['rs_accel'] = rs_accel.round(2)

    trend = _col(out, 'trend_score')
    # An entry only makes sense if the name is actually in an uptrend.
    uptrend = (trend >= 55) | (dist_ma50 > 0)

    # Freshness score (higher = fresher / better risk-adjusted entry).
    ext_score = _clip(100 - dist_ma50.clip(lower=0) * 3.0, 0, 100)   # extension penalty
    accel_score = _clip(50 + rs_accel.fillna(0) * 4.0, 0, 100)        # 50 = neutral
    fresh = (0.65 * ext_score + 0.35 * accel_score)
    fresh = fresh.where(uptrend, 0)
    # NaN (e.g. missing MA50) -> 0 so the Fresh progress column never renders
    # a broken bar and these rows sort to the bottom.
    out['freshness'] = fresh.fillna(0).round(0)

    # Ordinal label (precedence matters).
    def _label(row):
        d = row['dist_ma50']
        a = row['rs_accel']
        up = (row.get('trend_score', 0) or 0) >= 55 or (pd.notna(d) and d > 0)
        if pd.isna(d):
            return "—"
        if not up:
            return "⚪ Weak"
        if pd.notna(a) and a < _FADING_ACCEL:
            return "🔴 Late/Fading"
        if d > _EXTENDED_MIN:
            return "🟡 Extended"
        if d <= _PULLBACK_MAX:
            return "🔵 Pullback Buy"
        return "🟢 Actionable"

    out['entry_label'] = out.apply(_label, axis=1)
    return out


def add_position_sizing(df: pd.DataFrame, capital: float, risk_pct: float,
                        max_positions: int = DEFAULT_MAX_POSITIONS) -> pd.DataFrame:
    """
    Adds: stop_pct, suggested_value, suggested_shares.
    Requires dist_ma50 (call add_entry_freshness first). Sizes each name so a
    stop-out costs ~`risk_pct`% of `capital`, with a wider stop -> smaller
    position, capped at equal-weight (capital / max_positions).
    """
    out = df.copy()
    if 'dist_ma50' not in out.columns:
        out = add_entry_freshness(out)

    if out.empty:
        for c in ('stop_pct', 'suggested_value', 'suggested_shares'):
            out[c] = pd.Series(dtype='float64')
        return out

    price = _col(out, 'price', 'currentPrice')
    vol_annual = _col(out, 'volatility').fillna(40.0)
    dist_ma50 = pd.to_numeric(out['dist_ma50'], errors='coerce')

    daily_vol = vol_annual / np.sqrt(252.0)
    atr_stop = 2.5 * daily_vol                              # volatility-based swing stop %
    # BUGFIX: fillna(0) BEFORE the maxima — a NaN dist_ma50 (missing MA50)
    # otherwise propagates through np.maximum (which is NOT nan-aware) and
    # nulls stop_pct entirely, silently sizing a valid stock to 0 shares.
    struct_stop = dist_ma50.clip(lower=0).fillna(0.0) + 4.0  # to just below MA50 + buffer
    atr_stop = atr_stop.fillna(0.0)
    stop_pct = _clip(np.maximum(np.maximum(atr_stop, struct_stop), 6.0), 6.0, 30.0)
    out['stop_pct'] = stop_pct.round(1)

    risk_rupees = float(capital) * float(risk_pct) / 100.0
    eq_weight_cap = float(capital) / max(int(max_positions), 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        raw_value = risk_rupees / (stop_pct / 100.0)
    value = np.minimum(raw_value, eq_weight_cap)
    out['suggested_value'] = value.round(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        shares = np.floor(value / price)
    out['suggested_shares'] = shares.where(price > 0).fillna(0).astype('int64', errors='ignore')
    return out
