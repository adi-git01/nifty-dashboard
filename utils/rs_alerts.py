"""
Composite-RS alerts — crossing semantics.
=========================================

Fires when a stock's Composite RS (or one of its RS5/RS21/RS63 components)
*crosses* a threshold, upwards or downwards.

Why crossing and not level
--------------------------
The legacy metric path in utils/alerts.py tests a level ("comp_rs > 40") and
re-appends to the triggered list on every single check. Set that on a stock
already sitting at RS 45 and it fires forever, every refresh — noise, not a
signal. Here each alert remembers the value it last saw, and only fires on the
transition across the threshold:

    up   : last <  threshold <= now
    down : last >  threshold >= now

A brand-new alert is *seeded* with the current reading and does not fire on
that first observation, so creating "above 40" on a stock already at 45 does
not produce an instant fake trigger — it arms and waits for a real crossing.
The UI surfaces that state at creation time so it is never a surprise.

Because `last_value` advances on every check, a fired alert cannot re-fire
until the metric genuinely crosses back and over again. That is the whole
de-duplication mechanism; no cooldown timer is needed.

Bad-bar guard
-------------
A single corrupt price bar has previously produced physically impossible RS
readings (ARIHANT showed +759%). Firing an alert off one of those would be
worse than showing it in a table, so readings outside a sane band are treated
as missing: the alert neither fires nor advances `last_value`, and waits for a
clean bar.
"""
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.atomic_io import atomic_json_dump

# Tracked in git, unlike alerts.json. The EOD workflow evaluates these alerts
# headlessly and pushes to Telegram, so both the definitions and the last-seen
# values have to travel with the repo — that is what makes an alert fire when
# the dashboard is closed. It also means CI and a local dashboard can each
# advance the state; whichever observes a crossing first reports it, and the
# other goes quiet on the next pull because last_value has already moved.
RS_ALERTS_FILE = "data/rs_alerts.json"

# Alerts created before the move lived at the repo root. Read them once so an
# upgrade does not silently drop someone's alerts.
LEGACY_RS_ALERTS_FILE = "rs_alerts.json"

# Metric key -> (label, sane absolute ceiling). The ceilings mirror the
# display-time guard in main.py: a real 3-month relative strength tops out
# near +150pp, so anything past these is a corrupt bar, not a moonshot.
RS_METRICS: Dict[str, tuple] = {
    "comp_rs": ("CompRS (vs Nifty)", 200.0),
    "rs_1w":   ("RS5 — 1 week", 300.0),
    "rs_1m":   ("RS21 — 1 month", 300.0),
    "rs_3m":   ("RS63 — 3 months", 300.0),
}

DIRECTIONS = {
    "above": "Crosses ABOVE",
    "below": "Crosses BELOW",
    "either": "Crosses EITHER way",
}


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------
def _read(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        import json
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        # A corrupt/half-written file must not take the dashboard down.
        return []


def load_rs_alerts() -> List[Dict[str, Any]]:
    alerts = _read(RS_ALERTS_FILE)
    if alerts:
        return alerts
    return _read(LEGACY_RS_ALERTS_FILE)


def save_rs_alerts(alerts: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(RS_ALERTS_FILE) or ".", exist_ok=True)
    atomic_json_dump(alerts, RS_ALERTS_FILE, indent=2)


def normalize_ticker(ticker: str) -> str:
    t = (ticker or "").strip().upper()
    if t and not t.endswith((".NS", ".BO")):
        t += ".NS"
    return t


# --------------------------------------------------------------------------
# reading a metric safely
# --------------------------------------------------------------------------
def read_metric(df, ticker: str, metric: str) -> Optional[float]:
    """
    Current value of `metric` for `ticker`, or None when it is missing, NaN,
    or outside the sane band for that metric (suspected corrupt bar).
    """
    if df is None or metric not in getattr(df, "columns", []):
        return None
    row = df[df["ticker"] == ticker]
    if row.empty:
        return None
    val = pd.to_numeric(pd.Series([row.iloc[0].get(metric)]), errors="coerce").iloc[0]
    if pd.isna(val):
        return None
    ceiling = RS_METRICS.get(metric, (None, 200.0))[1]
    if abs(float(val)) > ceiling:
        return None
    return float(val)


def describe_arm_state(current: Optional[float], threshold: float, direction: str) -> str:
    """Human-readable state used at creation time and in the alert list."""
    if current is None:
        return "no reading yet — will arm on the next clean bar"
    if direction == "above":
        return ("already above — fires only if it drops back under and re-crosses"
                if current >= threshold else "armed — fires when it crosses up")
    if direction == "below":
        return ("already below — fires only if it climbs back over and re-crosses"
                if current <= threshold else "armed — fires when it crosses down")
    return "armed — fires on the next crossing either way"


# --------------------------------------------------------------------------
# create / delete
# --------------------------------------------------------------------------
def add_rs_alert(ticker: str, threshold: float, direction: str = "above",
                 metric: str = "comp_rs", notes: str = "", repeat: bool = True,
                 df=None) -> Dict[str, Any]:
    """
    Create a crossing alert. When `df` is supplied the alert is seeded with the
    current reading so it cannot fire spuriously on the first check.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {list(DIRECTIONS)}")
    if metric not in RS_METRICS:
        raise ValueError(f"metric must be one of {list(RS_METRICS)}")

    ticker = normalize_ticker(ticker)
    seed = read_metric(df, ticker, metric) if df is not None else None

    alerts = load_rs_alerts()
    alert = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "ticker": ticker,
        "metric": metric,
        "direction": direction,
        "threshold": float(threshold),
        "repeat": bool(repeat),
        "notes": notes,
        "created": datetime.now().isoformat(),
        "last_value": seed,
        "trigger_count": 0,
        "last_triggered": None,
        "last_trigger_direction": None,
    }
    alerts.append(alert)
    save_rs_alerts(alerts)
    return alert


def remove_rs_alert(alert_id: str) -> None:
    save_rs_alerts([a for a in load_rs_alerts() if a.get("id") != alert_id])


def rearm_rs_alert(alert_id: str) -> None:
    """Clear the fired state of a one-shot alert so it can trigger again."""
    alerts = load_rs_alerts()
    for a in alerts:
        if a.get("id") == alert_id:
            a["trigger_count"] = 0
            a["last_triggered"] = None
            a["last_trigger_direction"] = None
    save_rs_alerts(alerts)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def check_rs_alerts(df, persist: bool = True) -> List[Dict[str, Any]]:
    """
    Evaluate every RS alert against `df`. Returns the ones that crossed on this
    check, each with `current_value`, `previous_value`, `crossed` and a
    ready-to-render `alert_message`.
    """
    alerts = load_rs_alerts()
    if not alerts:
        return []

    triggered: List[Dict[str, Any]] = []
    dirty = False

    for alert in alerts:
        metric = alert.get("metric", "comp_rs")
        current = read_metric(df, alert.get("ticker", ""), metric)
        if current is None:
            continue  # missing or bad bar: hold state, wait for a clean read

        previous = alert.get("last_value")
        threshold = float(alert.get("threshold", 0.0))

        # Seed only — never fire on the first observation.
        if previous is None:
            alert["last_value"] = current
            dirty = True
            continue

        previous = float(previous)
        crossed_up = previous < threshold <= current
        crossed_down = previous > threshold >= current
        direction = alert.get("direction", "above")

        fires = ((direction == "above" and crossed_up)
                 or (direction == "below" and crossed_down)
                 or (direction == "either" and (crossed_up or crossed_down)))

        # One-shot alerts stay silent once they have fired, until re-armed.
        if fires and not alert.get("repeat", True) and alert.get("trigger_count", 0) > 0:
            fires = False

        if fires:
            way = "up" if crossed_up else "down"
            arrow = "🔼" if crossed_up else "🔽"
            label = RS_METRICS.get(metric, (metric, None))[0]
            name = alert["ticker"].replace(".NS", "").replace(".BO", "")
            alert["trigger_count"] = int(alert.get("trigger_count", 0)) + 1
            alert["last_triggered"] = datetime.now().isoformat()
            alert["last_trigger_direction"] = way
            triggered.append({
                **alert,
                "current_value": current,
                "previous_value": previous,
                "crossed": way,
                "alert_message": (f"{arrow} {name} {label} crossed {way} through "
                                  f"{threshold:+.1f} — now {current:+.1f} "
                                  f"(was {previous:+.1f})"),
            })

        alert["last_value"] = current
        dirty = True

    if dirty and persist:
        save_rs_alerts(alerts)
    return triggered


def get_rs_alerts_with_status(df) -> List[Dict[str, Any]]:
    """Alerts decorated with the current reading and distance to threshold."""
    out = []
    for alert in load_rs_alerts():
        a = dict(alert)
        current = read_metric(df, a.get("ticker", ""), a.get("metric", "comp_rs"))
        a["current_value"] = current
        a["distance"] = None if current is None else current - float(a.get("threshold", 0.0))
        a["state"] = describe_arm_state(current, float(a.get("threshold", 0.0)),
                                        a.get("direction", "above"))
        out.append(a)
    return out
