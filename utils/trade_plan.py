"""
One trade plan per stock: entry, stop and targets from Technicals, value zones from the valuation.

Technicals (utils/forensics.py) supplies the levels a trade is managed on:
  entry      today's close, or a close above the overhang trigger when supply sits just above
  stop       max(anchor low, MA50 - 0.5 ATR); wider than 10% -> half size; at or above price -> no trade
  target 1   entry + 2 x risk          target 2   the next supply shelf (or 2 x trigger - stop)
  exit rule  two closes below MA50 (held: also 15% off the peak close since entry)
The valuation (utils/valuation.py) says where those levels sit against fair value:
  P10 deep value · P25 accumulate below · P50 fair value · P90 trim above
A plan is flagged when the two disagree, e.g. target 1 above the trim level, or entry above it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _fin(x) -> bool:
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def build(P: Dict[str, Any], val: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Levels (high to low), a one-line summary and flags. `val` is a value_stock() result or None."""
    r = P["r"]
    close, stop = float(r["close"]), float(r["stop"])
    trigger = P.get("trigger")
    entry = float(trigger) if trigger else close
    valid_stop = stop < entry
    risk = entry - stop if valid_stop else np.nan
    t1 = entry + 2 * risk if valid_stop else np.nan
    t2 = float(r["t2"]) if _fin(r.get("t2")) and float(r["t2"]) > entry else np.nan
    pct = lambda x: (x / close - 1) * 100
    rows: List[Dict[str, Any]] = []
    add = lambda lvl, name, src, note="": rows.append(
        dict(level=name, price=float(lvl), vs_now=pct(float(lvl)), source=src, note=note)) if _fin(lvl) else None

    if valid_stop:
        add(t1, "🎯 Target 1", "Technicals", "entry + 2× risk — book part, move the stop to entry")
        add(t2, "🎯 Target 2", "Technicals", f"next supply shelf — reward/risk {(t2 - entry) / risk:.1f}" if _fin(t2) else "")
    if trigger:
        add(trigger, "▶️ Entry (close above)", "Technicals", "supply overhang just above — wait for the close through it")
    add(close, "● Price now", "", "entry at the close" if not trigger else "")
    add(stop, "🛑 Stop-loss", "Technicals",
        (f"risk {(entry - stop) / entry * 100:.1f}% from entry" + (" — wider than 10%: half size" if r.get("half_size") else ""))
        if valid_stop else "price is already below the stop — no trade")
    if _fin(r.get("ma50")):
        add(r["ma50"], "↘ Exit line (MA50)", "Technicals", "two closes below = exit")
    if P.get("held") and _fin(P.get("trail")):
        add(P["trail"], "↘ Trail (15% off peak)", "Technicals", "held position: exit on a close below")

    flags, zone = [], ""
    if val and val.get("status") == "OK":
        d = val["dist"]
        add(d["P90"], "💰 Trim above (P90)", "Valuation", "expensive against nine in ten fair-value outcomes")
        add(d["P50"], "💰 Fair value (P50)", "Valuation", "middle of the fair-value range")
        add(d["P25"], "💰 Accumulate below (P25)", "Valuation", "")
        add(d["P10"], "💰 Deep value (P10)", "Valuation", "")
        b = val.get("band")
        zone = (f"valuation {val['band']['bucket_lo']}" + (f"–{b['bucket_hi']}" if b["bucket_lo"] != b["bucket_hi"] else "")
                + f" (P{b['p_lo']:.0f}–{b['p_hi']:.0f})") if b else f"valuation P{d['cmp_pct']:.0f}"
        if entry > d["P90"]:
            flags.append("Entry is above the valuation's trim level (P90): a momentum trade, not a value one — "
                         "keep the stop tight and do not add.")
        elif _fin(t1) and t1 > d["P90"]:
            flags.append("Target 1 is above the valuation's trim level (P90): take the P90 level as the first place to book.")
        if valid_stop and stop > d["P50"]:
            flags.append("Even the stop is above fair value (P50): a stop-out would not be a bargain to re-buy.")
        if b and b["firmness"] == "WIDE":
            flags.append("The valuation range is wide — lean on the technical stop and targets, not the fair-value levels.")
    elif val is not None:
        flags.append(f"No valuation for this stock ({str(val.get('reason') or val.get('status', '')).strip()}): "
                     "plan uses technical levels only.")
    if not valid_stop:
        flags.insert(0, "No valid stop: price is at or below the stop level. Not a buy until it reclaims it.")
    elif r.get("half_size"):
        flags.append("Stop is wider than 10%: position is half size.")

    rows.sort(key=lambda x: -x["price"])
    bits = []
    if valid_stop:
        bits.append((f"Buy on a close above ₹{entry:,.2f}" if trigger else f"Entry ₹{entry:,.2f}"))
        bits.append(f"Stop ₹{stop:,.2f} (−{(entry - stop) / entry * 100:.1f}%)")
        bits.append(f"Target 1 ₹{t1:,.2f} (+{(t1 / entry - 1) * 100:.1f}%)")
        if _fin(t2):
            bits.append(f"Target 2 ₹{t2:,.2f} (+{(t2 / entry - 1) * 100:.1f}%)")
        if P.get("shares"):
            bits.append(f"{P['shares']:,} shares")
    if zone:
        bits.append(zone)
    return dict(levels=pd.DataFrame(rows), summary=" · ".join(bits), flags=flags, valid=bool(valid_stop),
                action=P.get("action"), entry=entry, stop=stop, t1=t1, t2=t2)
