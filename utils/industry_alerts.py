"""
Industry / basket status alerts.
================================

Watch a sub-industry, or a custom basket such as "Hotels", and get told when it
changes status or when its trend score crosses a level you choose.

Reading (from the daily master snapshot, members with valid data only):
  above_50   % of members closing above their 50-day average
  above_200  % above their 200-day average
  med_rs     median Composite RS vs Nifty (pp)

Status ladder (industry_turn_study.py tested it on 2021-26 monthly data):
  🟢 UPTREND     >= 50% above 200d and >= 50% above 50d
  🟣 WEAKENING   >= 50% above 200d, < 50% above 50d
  🟡 TURNING UP  < 50% above 200d, >= 60% above 50d and median RS > 0
  🟠 BASING      < 50% above 200d, 40-60% above 50d (or RS not yet positive)
  🔴 DOWNTREND   < 50% above 200d, < 40% above 50d
  History: an early TURNING UP was not a good buy on its own (industries
  under-performed over the next 6 months); TURNING UP followed by UPTREND was
  the better-evidenced step. The alerts are a watch tool, not a buy signal.

Trend score (0-100) = 0.4 x above_50 + 0.4 x above_200 + 0.2 x RS score,
  RS score = 50 + 5 x median RS, clipped to 0-100 (+10 pp vs Nifty = 100).
  Roughly: < 30 downtrend, 30-50 basing, 50-70 improving, > 70 strong.

Firing (daily breadth is noisy: unsmoothed, 6 alerts produced 161 messages in
8 months of replayed snapshots)
  readings are averaged over the last 5 snapshot dates before status/score
  status : leaving the current status needs its line crossed by 5 points (RS 1 pp)
  status : a new status must hold on 3 consecutive snapshot dates
  score  : crossing semantics -- seeded on creation, fires when the score moves
           from 2+ points below the level to 2+ points above (or the reverse)

State lives in data/industry_alerts.json (tracked; the EOD workflow evaluates
and commits it, repo_store syncs it from the dashboard).
"""
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils import repo_store
from utils.atomic_io import atomic_json_dump

INDUSTRY_ALERTS_FILE = "data/industry_alerts.json"
MIN_VALID = 3
RS_CEILING = 200.0          # same corrupt-bar guard as the RS alerts
SMOOTH_DAYS = 5             # status and score use the average of the last 5 daily readings
CONFIRM_DAYS = 3            # a new status must hold on 3 consecutive snapshot dates
STATUS_BAND = 5.0           # leaving a status needs the line crossed by 5 points (RS: 1 pp)
SCORE_BAND = 4.0            # score must clear the level by 2 points either side

STATUSES = ["DOWNTREND", "BASING", "TURNING UP", "UPTREND", "WEAKENING"]
ICONS = {"DOWNTREND": "🔴", "BASING": "🟠", "TURNING UP": "🟡", "UPTREND": "🟢", "WEAKENING": "🟣"}
NOTES = {
    "TURNING UP": "early stage: historically not a buy on its own; UPTREND is the confirmation",
    "UPTREND": "confirmed: most members back above their 200-day average",
    "WEAKENING": "short-term breadth lost while the long trend still holds",
}
RANK = {"DOWNTREND": 0, "BASING": 1, "TURNING UP": 2, "WEAKENING": 2.5, "UPTREND": 3}
NOTIFY = {"all": "Every status change", "up": "Upgrades only (e.g. DOWNTREND → BASING → TURNING UP → UPTREND)"}
DIRECTIONS = {"above": "crosses ABOVE", "below": "crosses BELOW", "either": "crosses EITHER way"}

# Custom baskets offered in the dashboard next to the sub-industries.
PRESET_BASKETS = {
    "Hotels": ["INDHOTEL", "EIHOTEL", "EIHAHOTELS", "CHALET", "LEMONTREE", "ITCHOTELS", "VENTIVE",
               "SAMHI", "THELEELA", "PARKHOTELS", "ADVANIHOTR", "ASIANHOTNR", "AHLEAST",
               "BRIGHOTEL", "ADVENTHTL"],
}


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def classify(above_50: float, above_200: float, med_rs: float,
             prev: Optional[str] = None, band: float = 0.0) -> Optional[str]:
    """Status ladder. With `prev`, each line is moved `band` points against leaving
    the previous status (1/5 of that for RS), so a reading hovering on a line keeps it."""
    if not (np.isfinite(above_50) and np.isfinite(above_200)):
        return None

    def ge(x, t, held, b=band):
        if prev is None:
            return x >= t
        return x >= (t - b if held else t + b)

    up200 = ge(above_200, 50, prev in ("UPTREND", "WEAKENING"))
    if up200:
        return "UPTREND" if ge(above_50, 50, prev == "UPTREND") else "WEAKENING"
    rs_pos = np.isfinite(med_rs) and ge(med_rs, 0.0001, prev == "TURNING UP", band / 5)
    if ge(above_50, 60, prev == "TURNING UP") and rs_pos:
        return "TURNING UP"
    return "BASING" if ge(above_50, 40, prev != "DOWNTREND") else "DOWNTREND"


def trend_score(above_50: float, above_200: float, med_rs: float) -> float:
    rs_score = float(np.clip(50 + 5 * (med_rs if np.isfinite(med_rs) else 0.0), 0, 100))
    return round(0.4 * above_50 + 0.4 * above_200 + 0.2 * rs_score, 1)


def basket_members(alert: Dict[str, Any]) -> List[str]:
    if alert.get("tickers"):
        return [normalize_ticker(t) for t in alert["tickers"]]
    from utils.nifty1000_list import SUB_INDUSTRY_MAP
    return [t for t, g in SUB_INDUSTRY_MAP.items() if g == alert.get("sub_industry")]


def basket_reading(df, members: List[str]) -> Optional[Dict[str, Any]]:
    """Status, score and breadth for `members` in snapshot `df`, or None if too few valid."""
    if df is None or df.empty:
        return None
    x = df[df["ticker"].isin(members)]
    px = pd.to_numeric(x.get("currentPrice", x.get("price")), errors="coerce")
    m50 = pd.to_numeric(x.get("fiftyDayAverage"), errors="coerce")
    m200 = pd.to_numeric(x.get("twoHundredDayAverage"), errors="coerce")
    rs = pd.to_numeric(x.get("comp_rs"), errors="coerce")
    rs = rs.where(rs.abs() <= RS_CEILING)
    ok = px.notna() & (px > 0) & m50.notna() & m200.notna()
    if ok.sum() < MIN_VALID:
        return None
    a50 = float((px[ok] > m50[ok]).mean() * 100)
    a200 = float((px[ok] > m200[ok]).mean() * 100)
    mrs = float(rs[ok].median()) if rs[ok].notna().any() else float("nan")
    return {"status": classify(a50, a200, mrs), "score": trend_score(a50, a200, mrs),
            "above_50": round(a50, 1), "above_200": round(a200, 1),
            "med_rs": round(mrs, 2) if np.isfinite(mrs) else None,
            "n_valid": int(ok.sum()), "n_members": len(members)}


def all_sub_industry_readings(df) -> pd.DataFrame:
    """Every sub-industry's current status and score, for the overview table."""
    from utils.nifty1000_list import SUB_INDUSTRY_MAP
    groups: Dict[str, List[str]] = {}
    for t, g in SUB_INDUSTRY_MAP.items():
        if g and g != "Unknown":
            groups.setdefault(g, []).append(t)
    rows = []
    for g, mem in sorted(groups.items()):
        r = basket_reading(df, mem)
        if r:
            rows.append({"industry": g, **r})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# store (same pattern as utils/rs_alerts.py)
# --------------------------------------------------------------------------
def normalize_ticker(ticker: str) -> str:
    t = (ticker or "").strip().upper()
    if t and not t.endswith((".NS", ".BO")):
        t += ".NS"
    return t


def _read_local() -> List[Dict[str, Any]]:
    if not os.path.exists(INDUSTRY_ALERTS_FILE):
        return []
    try:
        import json
        with open(INDUSTRY_ALERTS_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _write_local(alerts: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(INDUSTRY_ALERTS_FILE), exist_ok=True)
    atomic_json_dump(alerts, INDUSTRY_ALERTS_FILE, indent=2)


def load_industry_alerts() -> List[Dict[str, Any]]:
    if repo_store.is_configured():
        data, _ = repo_store.get_json(INDUSTRY_ALERTS_FILE)
        if isinstance(data, list):
            try:
                _write_local(data)
            except Exception:
                pass
            return data
    return _read_local()


def save_industry_alerts(alerts: List[Dict[str, Any]], reason: str = "update") -> bool:
    _write_local(alerts)
    if not repo_store.is_configured():
        return False
    return repo_store.put_json(INDUSTRY_ALERTS_FILE, alerts, f"🔔 Industry alerts: {reason}")


def add_industry_alert(name: str, sub_industry: Optional[str] = None,
                       tickers: Optional[List[str]] = None, on_status: bool = True,
                       score_threshold: Optional[float] = None, direction: str = "above",
                       notify: str = "all", notes: str = "", df=None) -> Dict[str, Any]:
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {list(DIRECTIONS)}")
    if not sub_industry and not tickers:
        raise ValueError("give a sub_industry or a list of tickers")
    alert = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "name": name,
        "sub_industry": sub_industry,
        "tickers": [normalize_ticker(t) for t in tickers] if tickers else None,
        "on_status": bool(on_status),
        "notify": notify if notify in NOTIFY else "all",
        "score_threshold": None if score_threshold is None else float(score_threshold),
        "direction": direction,
        "notes": notes,
        "created": datetime.now().isoformat(),
        "last_status": None, "pending_status": None, "pending_days": 0,
        "last_score": None, "last_date": None, "score_side": None, "recent": [],
        "history": [],
    }
    r = basket_reading(df, basket_members(alert)) if df is not None else None
    if r:                                   # seed: never fire on the first observation
        thr = alert["score_threshold"]
        alert.update(last_status=r["status"], last_score=r["score"],
                     score_side=None if thr is None else ("above" if r["score"] >= thr else "below"))
    alerts = load_industry_alerts()
    alerts.append(alert)
    save_industry_alerts(alerts, f"add {name}")
    return alert


def remove_industry_alert(alert_id: str) -> None:
    alerts = load_industry_alerts()
    gone = next((a for a in alerts if a.get("id") == alert_id), {})
    save_industry_alerts([a for a in alerts if a.get("id") != alert_id],
                         f"remove {gone.get('name', alert_id)}")


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def _fmt(r: Dict[str, Any]) -> str:
    rs = "n/a" if r.get("med_rs") is None else f"{r['med_rs']:+.1f}"
    return (f"score {r['score']:.0f} · {r['above_50']:.0f}% above 50d · "
            f"{r['above_200']:.0f}% above 200d · median RS {rs} ({r['n_valid']} stocks)")


def _smoothed(alert: Dict[str, Any], r: Dict[str, Any], asof: str) -> Dict[str, Any]:
    """Average of the last SMOOTH_DAYS readings (one per snapshot date), status and score from it."""
    rec = [x for x in (alert.get("recent") or []) if x.get("date") != asof]
    rec = (rec + [{"date": asof, "a50": r["above_50"], "a200": r["above_200"], "rs": r["med_rs"]}])[-SMOOTH_DAYS:]
    alert["recent"] = rec
    a50 = float(np.mean([x["a50"] for x in rec]))
    a200 = float(np.mean([x["a200"] for x in rec]))
    rsv = [x["rs"] for x in rec if x.get("rs") is not None]
    mrs = float(np.mean(rsv)) if rsv else float("nan")
    return {**r, "status": classify(a50, a200, mrs, alert.get("last_status"), STATUS_BAND), "score": trend_score(a50, a200, mrs),
            "above_50": round(a50, 1), "above_200": round(a200, 1),
            "med_rs": round(mrs, 2) if np.isfinite(mrs) else None}


def evaluate(alert: Dict[str, Any], df, asof: str) -> List[str]:
    """Advance one alert's state with snapshot `df` dated `asof` (YYYY-MM-DD); return messages."""
    raw = basket_reading(df, basket_members(alert))
    if not raw or raw["status"] is None:
        return []
    new_day = alert.get("last_date") != asof
    r = _smoothed(alert, raw, asof)
    out: List[str] = []
    name = alert.get("name") or alert.get("sub_industry")
    st, last = r["status"], alert.get("last_status")

    # status change: the new status must hold for CONFIRM_DAYS consecutive snapshot dates
    if last is None:
        alert["last_status"] = st
    elif st == last:
        alert["pending_status"], alert["pending_days"] = None, 0
    elif new_day:
        if alert.get("pending_status") == st:
            alert["pending_days"] = int(alert.get("pending_days") or 0) + 1
        else:
            alert["pending_status"], alert["pending_days"] = st, 1
        if alert["pending_days"] >= CONFIRM_DAYS:
            upgrade = RANK.get(st, 0) > RANK.get(last, 0)
            if alert.get("on_status", True) and (alert.get("notify", "all") == "all" or upgrade):
                note = NOTES.get(st)
                out.append(f"{ICONS[st]} {name}: {ICONS.get(last, '')} {last} → {st}\n   {_fmt(r)}"
                           + (f"\n   {note}" if note else ""))
            alert["history"] = (alert.get("history") or [])[-29:] + [
                {"date": asof, "from": last, "to": st, "score": r["score"]}]
            alert["last_status"] = st
            alert["pending_status"], alert["pending_days"] = None, 0

    # trend score crossing, with a +/- SCORE_BAND/2 buffer around the level
    thr = alert.get("score_threshold")
    if thr is not None:
        now, side = r["score"], alert.get("score_side")
        new_side = ("above" if now >= thr + SCORE_BAND / 2 else
                    "below" if now <= thr - SCORE_BAND / 2 else side)
        if side is None:
            alert["score_side"] = new_side or ("above" if now >= thr else "below")
        elif new_side != side and new_day:
            up = new_side == "above"
            d = alert.get("direction", "above")
            if d == "either" or (d == "above") == up:
                out.append(f"{'🔼' if up else '🔽'} {name}: trend score crossed {'up' if up else 'down'} "
                           f"through {thr:.0f} — now {now:.0f}, {ICONS[st]} {st}\n   {_fmt(r)}")
            alert["score_side"] = new_side
    alert["last_score"] = r["score"]
    alert["last_date"] = asof
    alert["last_reading"] = r
    return out


def check_industry_alerts(df, asof: str, persist: bool = True) -> List[str]:
    alerts = load_industry_alerts()
    if not alerts:
        return []
    msgs: List[str] = []
    for a in alerts:
        msgs += evaluate(a, df, asof)
    if persist:
        save_industry_alerts(alerts, f"{len(msgs)} event(s)" if msgs else "state")
    return msgs
