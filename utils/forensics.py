"""
Structural forensics (chart-prompt v6) -- the logic behind the Technicals tab.

Shared by the backtest (forensics_gate_study.py) and the live side (engine
snapshot + single-stock report), so what the tab shows is exactly what was
tested. Evidence: analysis/FORENSICS_VERDICT.md (2016-2026, 134k Gate-1
stock-weeks). Median 3m / 6m vs the typical stock, pp:

  Gate 1 pass            +1.44 / +2.45   (the core)
  v6 GO                  +1.73 / +3.43   (v6 anchor beat v5 at 6m, both eras)
  GO + ABSORPTION        +3.24 / +5.40   (best cell, 12/12 yrs; strongest 2021-26)
  SHAKEOUT               +2.14 / +4.78   (above Gate 1 in both eras)
  NEAR OVERHANG          1m -0.20, 3m +0.81 -> wait for the clean-air trigger
  DISTRIBUTION           +1.34 / +3.01   (NOT a warning)
  2R profit target       halves per-trade return -> reference level only

Gate 1  close > MA50, > MA200, within 15% of the 252-day high
Gate 2  anchor = highest-volume day of the last 60 sessions with volume >= 3x
        its prior 50-session median and turnover >= Rs 5 cr; CLV direction;
        VETO / GO / WAIT / FADE; pullback signature; shakeout probe; gap
Gate 3  volume-by-price shelves (3% bins, last 252 sessions); trapped if not
        closed at/above for >= 21 sessions; NEAR OVERHANG within 1.5 ATR
Gate 4  stop = max(anchor low, MA50 - 0.5 ATR); > 10% -> half size
Gate 5  RSI14, ADX14 +DI/-DI, Bollinger(20,2) width percentile (context)
"""
from __future__ import annotations

from datetime import datetime, time as dtime, timedelta, timezone

import numpy as np
import pandas as pd

_IST = timezone(timedelta(hours=5, minutes=30))
LOOK = 60
UP_STATES = ("GO", "WAIT (running)", "WAIT (40-80%)", "FADE")
SHELF_WIN = 252
BIN = np.log(1.03)
IST_CLOSE = dtime(15, 30)
SUPPLY_LABEL = {"CLEAN AIR": "CLEAN AIR", "INSIDE shelf": "ABSORPTION ZONE", "NEAR OVERHANG": "ACTIVE OVERHANG",
                "FAR OVERHANG": "OVERHANG (far)", "NOT COMPUTED": "NOT COMPUTED"}


# ----------------------------------------------------------------------------
# indicators (work on a Series or a dates x tickers DataFrame)
# ----------------------------------------------------------------------------
def wilder(x, n):
    return x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def indicators(close, high, low):
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()]).groupby(level=0).max()
    tr = tr.reindex(close.index)
    atr = wilder(tr, 14)
    atr_nz = atr.replace(0, np.nan)
    d = close.diff()
    ag = wilder(d.clip(lower=0), 14)
    al = wilder((-d).clip(lower=0), 14)
    rsi = 100 - 100 / (1 + ag / al.replace(0, np.nan))
    rsi = rsi.where(al > 0, np.where(ag > 0, 100.0, np.where(ag.notna() & al.notna(), 50.0, np.nan)))
    up, dn = high.diff(), -low.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    pdi = 100 * wilder(pdm, 14) / atr_nz
    ndi = 100 * wilder(ndm, 14) / atr_nz
    di_sum = (pdi + ndi).replace(0, np.nan)
    adx = wilder(100 * (pdi - ndi).abs() / di_sum, 14)
    m20, s20 = close.rolling(20).mean(), close.rolling(20).std()
    bbw = 4 * s20 / m20.replace(0, np.nan)
    bbp = bbw.rolling(SHELF_WIN, min_periods=120).rank(pct=True) * 100
    return atr, rsi, adx, pdi, ndi, bbp


# ----------------------------------------------------------------------------
# Gate 2
# ----------------------------------------------------------------------------
MIN_TURNOVER = {"INR": 5e7, "USD": 6e5}      # Rs 5 cr; about the same in dollars for a US stock (the rule was tested on NSE only)


def qualifying(c, v, vmed, min_turnover: float = 5e7):
    return np.isfinite(vmed) & (vmed > 0) & (v >= 3 * vmed) & (c * v >= min_turnover) & np.isfinite(c) & (c > 0)


def _clv_arr(c: np.ndarray, h: np.ndarray, l: np.ndarray) -> np.ndarray:
    pc = np.r_[np.nan, c[:-1]]
    locked = np.where(np.isnan(pc), 0.5, np.where(c >= pc, 1.0, 0.0))
    return np.where(h > l, (c - l) / np.where(h > l, h - l, 1.0), locked)


def pick_anchors(qual, v, i, look: int = LOOK):
    """(v6 anchor, v5 anchor) indices for day i, or (None, None)."""
    lo = max(2, i - look + 1)
    q = qual[lo:i + 1]
    if not q.any():
        return None, None
    a6 = lo + int(np.argmax(np.where(q, v[lo:i + 1], -1.0)))
    a5 = lo + int(np.where(q)[0][-1])
    return a6, a5


def pick_clean(qual, v, clv, i):
    """v6-clean candidate: highest-volume qualifying day of the last LOOK sessions that
    CLOSED STRONG (CLV >= 0.5). Tested against v6 in forensics_gate_study.py section 8;
    not used by the live tab unless that test passes."""
    lo = max(2, i - LOOK + 1)
    q = qual[lo:i + 1] & (clv[lo:i + 1] >= 0.5)
    if not q.any():
        return None
    return lo + int(np.argmax(np.where(q, v[lo:i + 1], -1.0)))


def grade(c, h, l, v, vmed, ma50, atr, clv, a, i, window=None):
    """
    Gate 2 for one anchor day `a`, seen from day i. window=None measures
    everything since the anchor (v6); window=15 only the first 15 sessions
    after it (v5's 'retention n/15'). Returns (fields, stop).
    """
    f = {}
    ca, la = c[a], l[a]
    dirn = 1 if clv[a] >= 0.5 else (-1 if clv[a] <= 0.3 else 0)
    dd = i - a
    f["days"] = dd
    f["gap"] = bool(la > h[a - 1])
    e = i if window is None else min(i, a + window)
    post = slice(a + 1, e + 1)
    pc = c[a - 1] if not np.isnan(c[a - 1]) else c[a - 2]
    cp = c[post]
    ok = ~np.isnan(cp)
    veto = False
    if dirn == 1:
        veto = bool((c[a + 1:i + 1][~np.isnan(c[a + 1:i + 1])] < la).any())   # invalidation: any close since
        ret = float((cp[ok] >= pc).mean()) if ok.any() else np.nan
        f["retention"] = ret
        if veto:
            f["state"] = "VETO"
        elif dd < 5 or np.isnan(ret):
            f["state"] = "WAIT (running)"
        elif ret >= 0.8:
            f["state"] = "GO"
        elif ret >= 0.4:
            f["state"] = "WAIT (40-80%)"
        else:
            f["state"] = "FADE"
        ma_stop = (ma50[i] - 0.5 * atr[i]) if (np.isfinite(ma50[i]) and np.isfinite(atr[i])) else np.nan
        cands = [x for x in (la, ma_stop) if np.isfinite(x)]
        stop = float(max(cands)) if cands else np.nan
    else:
        f["state"] = "BEAR anchor" if dirn == -1 else "MIXED anchor"
        stop = float(ma50[i] - 0.5 * atr[i]) if (np.isfinite(ma50[i]) and np.isfinite(atr[i])) else np.nan
    if e - a >= 3:
        f["cum_clv"] = float(np.nanmean(clv[post]))
        k = np.arange(a + 1, e + 1)
        down = k[(c[k] < c[k - 1])]
        if len(down) >= 2:
            f["dn_vs_anchor"] = float(np.nanmean(v[down]) / v[a])
            f["dn_vs_med"] = float(np.nanmean(v[down] / vmed[down]))
            dist = bool(((v[down] >= 1.5 * vmed[down]) & (c[down] < ca)).any())
            f["signature"] = ("DISTRIBUTION" if dist else
                              "ABSORPTION" if (f["dn_vs_med"] <= 0.8 and not veto) else "NEUTRAL")
            f["literal_absorb"] = bool(0.15 <= f["dn_vs_anchor"] <= 0.40 and not veto)
        lowv = v[k] < 0.8 * vmed[k]
        strong = clv[k] >= 0.6
        probe = np.zeros(len(k), dtype=bool)
        for S in (ma50[k], np.full(len(k), la)):
            probe |= (l[k] < S) & (c[k] > S)
        hit = probe & lowv & strong
        f["shakeout"] = bool(hit.any())
        if hit.any():
            f["shakeout_i"] = int(k[hit][-1])
    return f, stop


# ----------------------------------------------------------------------------
# Gate 3
# ----------------------------------------------------------------------------
def supply_at(c, turn, atr_i, i):
    """
    Supply status for day i from the last SHELF_WIN sessions (day i excluded).
    Returns {} if too little history or zero turnover/ATR, else supply, and for the nearest trapped
    shelf above: shelf_low, shelf_top (top of its contiguous shelf run -- the
    clean-air trigger), shelf_age (sessions since last close at/above it),
    shelf_share (% of window turnover in that run).
    """
    if not (np.isfinite(c[i]) and c[i] > 0 and np.isfinite(atr_i) and atr_i > 0):
        return {}
    w0 = max(0, i - SHELF_WIN)
    cw, tw = c[w0:i], turn[w0:i]
    okw = np.isfinite(cw) & (cw > 0) & np.isfinite(tw) & (tw >= 0)
    if okw.sum() < 120:
        return {}
    cw, tw = cw[okw], tw[okw]
    if float(tw.sum()) <= 0:
        return {}
    idx = np.arange(w0, i)[okw]
    b = np.floor(np.log(cw) / BIN).astype(int)
    b0 = b.min()
    tb = np.bincount(b - b0, weights=tw)
    occ = tb > 0
    if not occ.any():
        return {}
    shelf = np.where(occ & (tb >= 2 * tb[occ].mean()))[0] + b0
    sset = set(shelf.tolist())
    cur = int(np.floor(np.log(c[i]) / BIN))
    out = {"supply": "INSIDE shelf" if cur in sset else "CLEAN AIR"}
    trapped = []
    for sb in shelf:
        L = np.exp(sb * BIN)
        if L <= c[i]:
            continue
        vis = idx[cw >= L]
        if len(vis) and i - vis[-1] >= 21:
            trapped.append((L, sb, i - vis[-1]))
    if trapped:
        runs, seen = [], set()
        for L_, sb_, age_ in sorted(trapped):          # contiguous runs of shelf bins, lowest first
            if sb_ in seen:
                continue
            top = sb_
            while top + 1 in sset:
                top += 1
                seen.add(top)
            runs.append(dict(low=L_, top=np.exp((top + 1) * BIN), age=int(age_),
                             share=float(tb[sb_ - b0:top - b0 + 1].sum() / tb.sum() * 100)))
        L = runs[0]["low"]
        out.update(shelf_low=L, shelf_top=runs[0]["top"], shelf_age=runs[0]["age"],
                   shelf_share=runs[0]["share"], shelves_above=runs)
        if L <= c[i] + 1.5 * atr_i:
            out["supply"] = "NEAR OVERHANG"
        elif out["supply"] == "CLEAN AIR":
            out["supply"] = "FAR OVERHANG"
    return out


# ----------------------------------------------------------------------------
# live side
# ----------------------------------------------------------------------------
def market_open_now() -> bool:
    now = datetime.now(_IST)
    return now.weekday() < 5 and dtime(9, 15) <= now.time() < IST_CLOSE


def drop_provisional(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Intraday rule: today's candle is provisional while the exchange is open (NSE, or NYSE for a US stock)."""
    if len(df) and df.attrs.get("currency") == "USD":
        try:
            from zoneinfo import ZoneInfo
            ny = datetime.now(ZoneInfo("America/New_York"))
            if ny.weekday() < 5 and dtime(9, 30) <= ny.time() < dtime(16, 0) and pd.Timestamp(df.index[-1]).date() == ny.date():
                out = df.iloc[:-1].copy()
                out.attrs.update(df.attrs)
                return out, True
        except Exception:
            pass
        return df, False
    if len(df) and market_open_now():
        today = datetime.now(_IST).date()
        if pd.Timestamp(df.index[-1]).date() == today:
            out = df.iloc[:-1].copy()
            out.attrs.update(df.attrs)
            return out, True
    return df, False


def action_for(gate1: bool, state: str, supply: str) -> str:
    if not gate1:
        return "WATCH"
    if state == "GO":
        return "BUY ON TRIGGER" if supply == "NEAR OVERHANG" else "BUY NOW"
    if state.startswith("WAIT"):
        return "WAIT"
    return "WATCH"


def analyse(df: pd.DataFrame) -> dict:
    """
    Full v6 read of one stock's daily bars (needs High, Low, Close, Volume;
    >= 200 bars for MA200). Returns a flat dict; empty if too little data.
    """
    if df is None or len(df) < 60 or not {"High", "Low", "Close", "Volume"} <= set(df.columns):
        return {}
    d = df[["High", "Low", "Close", "Volume"]].astype(float).dropna(subset=["Close"])
    d = d[d["Close"] > 0]
    if len(d) < 60:
        return {}
    c, h, l, v = (d[k].values for k in ("Close", "High", "Low", "Volume"))
    i = len(c) - 1
    ma50s = d.Close.rolling(50).mean()
    ma50 = ma50s.values
    ma200 = d.Close.rolling(200).mean().values
    atr, rsi, adx, pdi, ndi, bbp = (x.values for x in indicators(d.Close, d.High, d.Low))
    vmed = d.Volume.rolling(50, min_periods=30).median().shift(1).values
    clv = _clv_arr(c, h, l)
    hi = np.nanmax(c[-252:])
    r = dict(close=c[i], date=d.index[i], ma50=ma50[i], ma200=ma200[i], atr=atr[i], hi252=hi,
             dist_ma50=(c[i] / ma50[i] - 1) * 100, dist_hi=(c[i] / hi - 1) * 100,
             rsi=rsi[i], adx=adx[i], pdi=pdi[i], ndi=ndi[i], bbp=bbp[i], sessions=len(c))
    r["gate1"] = bool(c[i] > ma50[i] and not np.isnan(ma200[i]) and c[i] > ma200[i] and c[i] >= 0.85 * hi)
    qual = qualifying(c, v, vmed, MIN_TURNOVER.get(df.attrs.get("currency", "INR"), 5e7))
    a6, _ = pick_anchors(qual, v, i)
    ma_stop_i = float(ma50[i] - 0.5 * atr[i]) if (np.isfinite(ma50[i]) and np.isfinite(atr[i])) else np.nan
    stop = ma_stop_i
    if a6 is None:
        r["state"] = "NO ANCHOR"
    else:
        g, stop = grade(c, h, l, v, vmed, ma50, atr, clv, a6, i)
        r.update(g)
        r.update(anchor_date=d.index[a6], anchor_volx=v[a6] / vmed[a6], anchor_clv=clv[a6],
                 anchor_low=l[a6], anchor_close=c[a6], anchor_turnover_cr=c[a6] * v[a6] / 1e7)
        if "shakeout_i" in r:
            k = r.pop("shakeout_i")
            _s1 = ma50[k] if np.isfinite(ma50[k]) else l[a6]
            _s2 = l[a6] if np.isfinite(l[a6]) else _s1
            _lo_ref = min(_s1, _s2) if np.isfinite(_s1) and np.isfinite(_s2) else np.nan
            _hi_ref = max(_s1, _s2) if np.isfinite(_s1) and np.isfinite(_s2) else np.nan
            r.update(shakeout_date=d.index[k], shakeout_volx=v[k] / vmed[k], shakeout_clv=clv[k],
                     shakeout_level=_lo_ref if (np.isfinite(_lo_ref) and l[k] < _lo_ref) else _hi_ref)
    if r.get("state") not in UP_STATES:          # signature & shakeout were only tested on +1 anchors
        r.pop("signature", None)
        if "shakeout" in r or a6 is not None:
            r["shakeout"] = False
        for _sk in ("shakeout_date", "shakeout_volx", "shakeout_clv", "shakeout_level"):
            r.pop(_sk, None)
    if r["state"] == "VETO":                       # the anchor low is broken: fall back to the MA50 stop
        stop = ma_stop_i
    r.update(supply_at(c, c * v, atr[i], i))
    r.setdefault("supply", "NOT COMPUTED")
    r["stop"] = stop
    # where the trade starts: under a near overhang the entry is the close through the shelf top (BUY ON TRIGGER), so
    # risk, half size and the share count are measured from there -- from the close they came out up to ~6x too big
    top = r.get("shelf_top")
    entry = float(top) if r.get("supply") == "NEAR OVERHANG" and top is not None and np.isfinite(top) and top > c[i] else c[i]
    r["entry"] = entry
    valid_stop = bool(np.isfinite(stop) and 0 < stop < entry)
    r["stop_pct"] = (entry - stop) / entry * 100 if valid_stop else np.nan
    r["half_size"] = bool(r["stop_pct"] > 10) if np.isfinite(r["stop_pct"]) else False
    risk = entry - stop if valid_stop else np.nan
    r["t1"] = entry + 2 * risk if np.isfinite(risk) and risk > 0 else np.nan
    # T2: next resistance beyond where the trade starts. With a near overhang the
    # entry is the clean-air trigger (shelf top), so T2 is the next shelf above it
    # or, if none, a measured move of one trigger-to-stop height; otherwise the
    # nearest trapped shelf above the close.
    runs = r.get("shelves_above", [])
    if r["supply"] == "NEAR OVERHANG" and runs:
        trig = runs[0]["top"]
        r["t2"] = runs[1]["low"] if len(runs) > 1 else (2 * trig - stop if np.isfinite(stop) and 0 < stop < trig else np.nan)
        r["t2_kind"] = "next shelf" if len(runs) > 1 else "measured move"
    else:
        r["t2"] = runs[0]["low"] if runs else np.nan
        r["t2_kind"] = "nearest shelf"
    r["action"] = action_for(r["gate1"], r["state"], r["supply"])
    # exit-rule context (prompt: 2 closes < MA50 or close < 0.85 x running peak)
    r["closes_below_ma50"] = int((c[-2:] < ma50[-2:]).sum())
    return r


FIELDS = ["gate1", "dist_hi", "state", "anchor_date", "anchor_volx", "days", "retention", "signature", "shakeout",
          "gap", "supply", "shelf_low", "shelf_top", "stop", "stop_pct", "half_size", "t1", "t2", "entry",
          "adx", "pdi", "ndi", "bbp", "rsi", "action"]


def snapshot(df: pd.DataFrame) -> dict:
    """fx_* fields for the engine cache (one row per stock)."""
    df, _ = drop_provisional(df)
    r = analyse(df)
    if r.get("state") not in UP_STATES:          # signature / shakeout were only tested on +1 anchors
        r.pop("signature", None)
        r["shakeout"] = False if r else np.nan
    out = {}
    for k in FIELDS:
        x = r.get(k, np.nan)
        if isinstance(x, pd.Timestamp):
            x = x.strftime("%Y-%m-%d")
        elif isinstance(x, (np.floating, float)) and np.isfinite(x):
            x = round(float(x), 2)
        elif isinstance(x, (np.floating, float)):
            x = np.nan
        elif isinstance(x, np.bool_):
            x = bool(x)
        out[f"fx_{k}"] = x
    return out


# ----------------------------------------------------------------------------
# single-stock report in the prompt's v6 layout
# ----------------------------------------------------------------------------
def _p(x, nd=2):
    return "NOT COMPUTED" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{nd}f}"


def _d(x):
    return pd.Timestamp(x).strftime("%d-%b-%Y") if x is not None and not pd.isna(x) else "NOT COMPUTED"


def events(df: pd.DataFrame, look: int = LOOK) -> pd.DataFrame:
    """
    Every qualifying volume day of the last `look` sessions, each graded from
    today as if it were the anchor (v6 rules). The one with the highest volume
    is the anchor the tab uses.
    """
    d = df[["High", "Low", "Close", "Volume"]].astype(float).dropna(subset=["Close"])
    d = d[d["Close"] > 0]
    if len(d) < 3:
        return pd.DataFrame()
    c, h, l, v = (d[k].values for k in ("Close", "High", "Low", "Volume"))
    i = len(c) - 1
    ma50 = d.Close.rolling(50).mean().values
    atr = indicators(d.Close, d.High, d.Low)[0].values
    vmed = d.Volume.rolling(50, min_periods=30).median().shift(1).values
    clv = _clv_arr(c, h, l)
    qual = qualifying(c, v, vmed, MIN_TURNOVER.get(df.attrs.get("currency", "INR"), 5e7))
    lo = max(2, i - look + 1)
    idx = [k for k in range(lo, i + 1) if qual[k]]
    a6 = pick_anchors(qual, v, i, look=look)[0]
    rows = []
    for k in idx:
        g, _ = grade(c, h, l, v, vmed, ma50, atr, clv, k, i)
        dirn = "+1 bull" if clv[k] >= 0.5 else ("-1 bear" if clv[k] <= 0.3 else "0 mixed")
        rows.append(dict(date=d.index[k], vol_ratio=v[k] / vmed[k], clv=clv[k], direction=dirn,
                         close=c[k], low=l[k], high=h[k], turnover_cr=c[k] * v[k] / 1e7,
                         days=i - k, retention=g.get("retention", np.nan), state=g["state"],
                         anchor=(k == a6)))
    return pd.DataFrame(rows)


def plan(df: pd.DataFrame, held_price: float | None = None, held_date=None,
         risk_rupees: float = 10000.0) -> dict:
    """
    Everything the report and the chart show, computed once: the v6 read (r),
    the closed bars used, the action, triggers, size and exit-rule state.
    """
    df, provisional = drop_provisional(df)
    if df is None or len(df) < 60:
        return {}
    r = analyse(df)
    if not r:
        return {}
    c = pd.to_numeric(df["Close"], errors="coerce").dropna()
    c = c[c > 0]
    if getattr(c.index, "tz", None) is not None:
        c.index = c.index.tz_localize(None)
    close = r["close"]
    held = held_price is not None
    state, supply, stop = r["state"], r["supply"], r["stop"]
    entry = r.get("entry", close)
    # No active setup if price is already at/below the stop or stop <= 0
    stop_valid = bool(np.isfinite(stop) and 0 < stop < min(close, entry))
    risk = entry - stop if stop_valid else np.nan
    shares = int(risk_rupees / risk) if np.isfinite(risk) and risk > 0 else 0
    if r["half_size"]:
        shares //= 2
    since = pd.Timestamp(held_date).tz_localize(None) if held and held_date is not None else c.index[-1]
    c_since = c[c.index >= since] if held else c
    peak = (float(max(c_since.max(), held_price)) if len(c_since) else float(max(close, held_price))) if held else close
    fired = None
    if held and len(c_since):
        m50 = c.rolling(50).mean()
        cc, mm = c_since, m50.reindex(c_since.index)
        below = (cc < mm)
        hit = (below & below.shift(1, fill_value=False)) | (cc < 0.85 * cc.cummax())
        if hit.any():
            fired = hit[hit].index[0]
    action = ("EXIT" if fired is not None else "HOLD") if held else r["action"]

    trig_a = trig_b = None
    if state.startswith("WAIT") and "anchor_low" in r:
        trig_a = (f"stay above anchor low {_p(r['anchor_low'])} and reach >= 80% retention by day 5 "
                  f"(now day {r.get('days')})")
    if state == "GO":
        if close < r["ma50"]:
            trig_a = (f"reclaim MA50: daily close > {_p(r['ma50'])}, while holding anchor low "
                      f"{_p(r.get('anchor_low'))}")
        else:
            trig_a = (f"pullback toward MA50 {_p(r['ma50'])} on < 0.8x median volume, holding anchor low "
                      f"{_p(r.get('anchor_low'))}")
    if "shelf_top" in r:
        trig_b = f"daily close > {_p(r['shelf_top'])} (top of the shelf starting {_p(r['shelf_low'])}) into clean air"
    trigger = r.get("shelf_top") if supply == "NEAR OVERHANG" else None
    # R:R from here uses risk from today's close to stop (not trigger - stop), and is undefined when close <= stop.
    risk_now = close - stop if (np.isfinite(stop) and 0 < stop < close) else np.nan
    rr_now = ((r["t2"] - close) / risk_now
              if np.isfinite(risk_now) and risk_now > 0 and np.isfinite(r["t2"]) and r["t2"] > close
              else np.nan)
    rr_trig = ((r["t2"] - trigger) / (trigger - stop)
               if trigger and np.isfinite(stop) and stop > 0 and not np.isnan(r["t2"]) and trigger > stop and r["t2"] > trigger
               else np.nan)
    return dict(r=r, df=df, provisional=provisional, held=held, held_price=held_price, since=since,
                currency=df.attrs.get("currency", "INR"),
                peak=peak, trail=0.85 * peak, fired=fired, action=action, shares=shares, risk=risk,
                risk_rupees=risk_rupees, trig_a=trig_a, trig_b=trig_b, trigger=trigger,
                rr_now=rr_now, rr_trig=rr_trig)


def levels(P: dict) -> list[dict]:
    """Key price levels for the chart and the ladder: name, price, kind, note."""
    r = P["r"]
    out = [dict(name="Close", price=r["close"], kind="close", note=_d(r["date"]))]
    add = lambda n, p, k, note="": out.append(dict(name=n, price=float(p), kind=k, note=note)) \
        if p is not None and not pd.isna(p) else None
    if 0 < r["stop"] < r["close"]:
        add("Stop", r["stop"], "stop", f"{r['stop_pct']:.1f}% risk" + (" · half size" if r["half_size"] else ""))
    add("MA50 (exit line)", r["ma50"], "ma", "2 closes below = exit (prompt)")
    add("MA200", r["ma200"], "ma")
    if "anchor_low" in r and r["state"] in UP_STATES:
        add("Anchor low (VETO line)", r["anchor_low"], "veto", _d(r["anchor_date"]))
    for k, run in enumerate(r.get("shelves_above", [])[:3]):
        add(f"Shelf {k + 1} top" + (" = clean-air trigger" if k == 0 and P["trigger"] else ""), run["top"],
            "trigger" if k == 0 and P["trigger"] else "shelf", f"{run['share']:.0f}% of turnover, {run['age']}d trapped")
        add(f"Shelf {k + 1} low", run["low"], "shelf")
    add("T1 = 2R (reference)", r["t1"], "target", "a 2R exit halved returns")
    add(f"T2 ({r.get('t2_kind', '')})", r["t2"], "target")
    add("52-week high close", r["hi252"], "high")
    add("Trail 0.85 x peak", P["trail"], "trail", f"peak {_p(P['peak'])}")
    # merge duplicates (e.g. T2 == shelf 2 low)
    seen, uniq = set(), []
    for x in sorted(out, key=lambda x: -x["price"]):
        key = (round(x["price"], 2), x["kind"])
        if key not in seen:
            seen.add(key)
            uniq.append(x)
    return uniq


def report(df: pd.DataFrame, ticker: str, held_price: float | None = None, held_date=None,
           risk_rupees: float = 10000.0, source: str | None = None, P: dict | None = None) -> str:
    """Plain-text v6 report. held_price None = CANDIDATE mode."""
    P = P or plan(df, held_price, held_date, risk_rupees)
    if not P:
        return f"{ticker}: fewer than 60 closed sessions -- aborted (prompt rule)."
    src_txt = source or df.attrs.get("source") or "NSE / Yahoo daily"
    r, close = P["r"], P["r"]["close"]
    us = P.get("currency") == "USD"
    cur_txt = "$" if us else "Rs "
    min_tv = "$0.6M" if us else "Rs 5 cr"
    state, supply, stop, sp = r["state"], r["supply"], r["stop"], r["stop_pct"]
    held, peak, trail, fired, since = P["held"], P["peak"], P["trail"], P["fired"], P["since"]
    lag = (pd.Timestamp.now().normalize() - pd.Timestamp(r["date"]).normalize()).days
    L = []
    L.append(f"{ticker} | close {cur_txt}{close:,.2f} on {_d(r['date'])} | {r['sessions']} sessions | source {src_txt} | lag {lag}d"
             + (" | today's candle PROVISIONAL, excluded" if P["provisional"] else ""))
    L.append(f"MODE: {'HELD @ ' + cur_txt + _p(P['held_price']) + ' since ' + _d(P['since']) if held else 'CANDIDATE'}")
    L.append(f"\nACTION: {P['action']}")
    L.append(f"Trigger A: {P['trig_a'] or 'NOT COMPUTED'}")
    L.append(f"Trigger B: {P['trig_b'] or 'NOT COMPUTED'}")
    if np.isfinite(stop) and 0 < stop < close:
        L.append(f"Stop: {cur_txt}{_p(stop)} ({'max of anchor low, MA50 - 0.5 ATR' if state in UP_STATES else 'MA50 - 0.5 ATR'})"
                 f" = {_p(sp, 1)}%   Size: {P['shares']} shares for {cur_txt}{P['risk_rupees']:,.0f} risk"
                 + (" (HALVED: stop > 10%)" if r["half_size"] else ""))
    else:
        L.append(f"Stop: NOT COMPUTED -- price is below the stop level {cur_txt}{_p(stop)} (no long setup)   Size: 0")
    L.append(f"Targets: T1 {cur_txt}{_p(r['t1'])} (2R, reference only -- a 2R exit halved per-trade returns in the backtest) | "
             f"T2 {cur_txt}{_p(r['t2'])} ({r.get('t2_kind', '')})")
    L.append(f"R:R: from current {_p(P['rr_now'], 1)} (to T2) | from trigger {_p(P['rr_trig'], 1)}")
    L.append(f"Review on: {'day 5 of the anchor' if r.get('days', 99) < 5 else '+10 sessions'}")

    L.append("\nGATES")
    L.append(f"1 Trend    {'PASS' if r['gate1'] else 'FAIL'}  close {cur_txt}{close:,.2f} vs MA50 {cur_txt}{_p(r['ma50'])} "
             f"({r['dist_ma50']:+.1f}%) / MA200 {cur_txt}{_p(r['ma200'])} / 52w-high {cur_txt}{_p(r['hi252'])} ({r['dist_hi']:+.1f}%)   [DATA]")
    if "anchor_date" in r:
        ret = r.get("retention")
        ret_s = "NOT COMPUTED" if ret is None or np.isnan(ret) else f"{round(ret * r['days'])}/{r['days']}"
        tv_val = r["anchor_turnover_cr"] * 10 if us else r["anchor_turnover_cr"]
        tv_s = f"${tv_val:.1f}M" if us else f"Rs {tv_val:.1f} cr"
        L.append(f"2 Event    {state}  {_d(r['anchor_date'])} | vol_ratio {r['anchor_volx']:.1f}x | clv {r['anchor_clv']:.2f} | "
                 f"turnover {tv_s} | retention "
                 f"{ret_s} | "
                 f"cause NOT COMPUTED   [DATA]")
    else:
        L.append(f"2 Event    NEUTRAL  no day in the last 60 sessions with >= 3x median volume and >= {min_tv}   [DATA]")
    if "shelf_low" in r:
        L.append(f"3 Supply   {'OVERHANG' if 'OVERHANG' in supply else 'CLEAR'}  nearest trapped shelf "
                 f"{_p(r['shelf_low'])}-{_p(r['shelf_top'])}, {(r['shelf_low'] - close) / r['atr']:.1f} ATR above   [DATA]")
    else:
        L.append(f"3 Supply   CLEAR  {SUPPLY_LABEL[supply]}: no trapped shelf above in the last 252 sessions   [DATA]")
    L.append("4 Levels   stop / targets / R:R as above")
    squeeze = ("n/a" if not np.isfinite(r["bbp"]) else
               "squeeze" if r["bbp"] < 20 else ("wide" if r["bbp"] > 80 else "normal"))
    L.append(f"5 Lens     RSI {_p(r['rsi'], 0)} | ADX {_p(r['adx'], 0)} (+DI {_p(r['pdi'], 0)} / -DI {_p(r['ndi'], 0)}) | "
             f"Bollinger width {_p(r['bbp'], 0)} pct ({squeeze}). Context only; did not move the action.   [DATA]")

    L.append("\nFORENSICS & MARKET MICROSTRUCTURE")
    if "shelf_low" in r:
        L.append(f"Overhead Supply: {SUPPLY_LABEL[supply]}, shelf {_p(r['shelf_low'])}-{_p(r['shelf_top'])}, "
                 f"{r['shelf_share']:.0f}% of the year's turnover, untouched for {r['shelf_age']} sessions")
    else:
        L.append(f"Overhead Supply: {SUPPLY_LABEL[supply]}")
    if "cum_clv" in r:
        L.append(f"CLV Diagnostic (Since Anchor): mean {r['cum_clv']:.2f} -- no predictive value in the backtest; shown for context")
    else:
        L.append("CLV Diagnostic (Since Anchor): NOT COMPUTED (< 3 sessions since anchor)")
    if "dn_vs_med" in r:
        L.append(f"Pullback Signature: down days {r['dn_vs_med']:.2f}x median volume, {r['dn_vs_anchor'] * 100:.0f}% of anchor volume")
    else:
        L.append("Pullback Signature: NOT COMPUTED (< 2 down days since anchor)")
    if r.get("shakeout"):
        L.append(f"Microstructure Probe: {_d(r['shakeout_date'])} | level {_p(r['shakeout_level'])} | "
                 f"vol {r['shakeout_volx']:.2f}x | CLV {r['shakeout_clv']:.2f}")
    else:
        L.append("Microstructure Probe: none since the anchor")
    stance = {"ABSORPTION": "ABSORPTION (digestion)", "DISTRIBUTION": "DISTRIBUTION (liquidation) -- NOTE: did not "
              "underperform in the backtest", "NEUTRAL": "NEUTRAL"}.get(r.get("signature"), "NEUTRAL")
    if state not in UP_STATES and r.get("signature"):
        stance += " -- anchor is not a +1 event; the signature was only backtested on +1 anchors"
    L.append(f"Institutional Stance: {stance}   [INFER]")

    L.append(f"\nEXIT RULE ({'HELD' if held else 'if bought today'})")
    L.append(f"MA50 {_p(r['ma50'])} -- distance {r['dist_ma50']:+.1f}% -- closes below: {r['closes_below_ma50']}/2")
    L.append(f"Peak {_p(peak)} (running since {_d(since)}) x 0.85 = {_p(trail)} -- distance {(close / trail - 1) * 100:+.1f}%")
    L.append(f"Status: {'FIRED on ' + _d(fired) if fired is not None else 'NOT FIRED'}")
    L.append("(The live engine exits on ONE close < MA50 or a 12/15% trail; the 2-close rule looked better per trade "
             "but worse at portfolio level -- FORENSICS_VERDICT.md.)")

    L.append("\nWHAT FLIPS THIS")
    bull = (f"daily close > {cur_txt}{_p(r['shelf_top'])} (clean-air trigger)" if "shelf_top" in r and "OVERHANG" in supply
            else f"new volume anchor (>= 3x median, >= {min_tv}, CLV >= 0.5) above {cur_txt}{_p(r['hi252'])}")
    bear = (f"close < anchor low {_p(r['anchor_low'])} (VETO)" if "anchor_low" in r and state in UP_STATES
            else f"2 closes < MA50 {_p(r['ma50'])}")
    L.append(f"Bullish flip: {bull}")
    L.append(f"Bearish flip: {bear}")

    L.append("\nUNKNOWNS / CAVEATS")
    L.append("- Anchor cause: NOT COMPUTED (no filing / news feed); delivery % NOT COMPUTED.")
    L.append("- Supply shelves use the last 252 sessions only; older supply is invisible.")
    L.append("- Gap = anchor-day low above the prior high (open prices not used, as in the backtest).")
    L.append("- Edges are 2016-26 medians vs the typical stock (FORENSICS_VERDICT.md), not a forecast for this stock.")
    return "\n".join(L)
