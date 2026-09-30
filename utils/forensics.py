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

from datetime import datetime, time as dtime

import numpy as np
import pandas as pd

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
    d = close.diff()
    rsi = 100 - 100 / (1 + wilder(d.clip(lower=0), 14) / wilder((-d).clip(lower=0), 14))
    up, dn = high.diff(), -low.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    pdi = 100 * wilder(pdm, 14) / atr
    ndi = 100 * wilder(ndm, 14) / atr
    adx = wilder(100 * (pdi - ndi).abs() / (pdi + ndi), 14)
    m20, s20 = close.rolling(20).mean(), close.rolling(20).std()
    bbw = 4 * s20 / m20
    bbp = bbw.rolling(SHELF_WIN, min_periods=120).rank(pct=True) * 100
    return atr, rsi, adx, pdi, ndi, bbp


# ----------------------------------------------------------------------------
# Gate 2
# ----------------------------------------------------------------------------
def qualifying(c, v, vmed):
    return (v >= 3 * vmed) & (c * v >= 5e7) & ~np.isnan(c)


def pick_anchors(qual, v, i):
    """(v6 anchor, v5 anchor) indices for day i, or (None, None)."""
    lo = max(2, i - LOOK + 1)
    q = qual[lo:i + 1]
    if not q.any():
        return None, None
    a6 = lo + int(np.argmax(np.where(q, v[lo:i + 1], -1.0)))
    a5 = lo + int(np.where(q)[0][-1])
    return a6, a5


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
        stop = max(la, ma50[i] - 0.5 * atr[i])
    else:
        f["state"] = "BEAR anchor" if dirn == -1 else "MIXED anchor"
        stop = ma50[i] - 0.5 * atr[i]
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
    Returns {} if too little history, else supply, and for the nearest trapped
    shelf above: shelf_low, shelf_top (top of its contiguous shelf run -- the
    clean-air trigger), shelf_age (sessions since last close at/above it),
    shelf_share (% of window turnover in that run).
    """
    w0 = max(0, i - SHELF_WIN)
    cw, tw = c[w0:i], turn[w0:i]
    okw = ~np.isnan(cw) & ~np.isnan(tw)
    if okw.sum() < 120:
        return {}
    cw, tw = cw[okw], tw[okw]
    idx = np.arange(w0, i)[okw]
    b = np.floor(np.log(cw) / BIN).astype(int)
    b0 = b.min()
    tb = np.bincount(b - b0, weights=tw)
    occ = tb > 0
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
        L, sb, age = min(trapped)
        top = sb
        while top + 1 in sset:
            top += 1
        run = tb[sb - b0:top - b0 + 1].sum()
        out.update(shelf_low=L, shelf_top=np.exp((top + 1) * BIN), shelf_age=int(age),
                   shelf_share=float(run / tb.sum() * 100))
        if L <= c[i] + 1.5 * atr_i:
            out["supply"] = "NEAR OVERHANG"
        elif out["supply"] == "CLEAN AIR":
            out["supply"] = "FAR OVERHANG"
    return out


# ----------------------------------------------------------------------------
# live side
# ----------------------------------------------------------------------------
def market_open_now() -> bool:
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
    except Exception:
        return False
    return now.weekday() < 5 and dtime(9, 15) <= now.time() < IST_CLOSE


def drop_provisional(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Intraday rule: today's candle is provisional while NSE is open."""
    if len(df) and market_open_now():
        try:
            from zoneinfo import ZoneInfo
            today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
            if pd.Timestamp(df.index[-1]).date() == today:
                return df.iloc[:-1], True
        except Exception:
            pass
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
    c, h, l, v = (d[k].values for k in ("Close", "High", "Low", "Volume"))
    i = len(c) - 1
    ma50s = d.Close.rolling(50).mean()
    ma50 = ma50s.values
    ma200 = d.Close.rolling(200).mean().values
    atr, rsi, adx, pdi, ndi, bbp = (x.values for x in indicators(d.Close, d.High, d.Low))
    vmed = d.Volume.rolling(50, min_periods=30).median().shift(1).values
    clv = np.where(h > l, (c - l) / np.where(h > l, h - l, 1), np.nan)
    hi = np.nanmax(c[-252:])
    r = dict(close=c[i], date=d.index[i], ma50=ma50[i], ma200=ma200[i], atr=atr[i], hi252=hi,
             dist_ma50=(c[i] / ma50[i] - 1) * 100, dist_hi=(c[i] / hi - 1) * 100,
             rsi=rsi[i], adx=adx[i], pdi=pdi[i], ndi=ndi[i], bbp=bbp[i], sessions=len(c))
    r["gate1"] = bool(c[i] > ma50[i] and not np.isnan(ma200[i]) and c[i] > ma200[i] and c[i] >= 0.85 * hi)
    qual = qualifying(c, v, vmed)
    a6, _ = pick_anchors(qual, v, i)
    stop = ma50[i] - 0.5 * atr[i]
    if a6 is None:
        r["state"] = "NO ANCHOR"
    else:
        g, stop = grade(c, h, l, v, vmed, ma50, atr, clv, a6, i)
        r.update(g)
        r.update(anchor_date=d.index[a6], anchor_volx=v[a6] / vmed[a6], anchor_clv=clv[a6],
                 anchor_low=l[a6], anchor_close=c[a6], anchor_turnover_cr=c[a6] * v[a6] / 1e7)
        if "shakeout_i" in r:
            k = r.pop("shakeout_i")
            r.update(shakeout_date=d.index[k], shakeout_volx=v[k] / vmed[k], shakeout_clv=clv[k],
                     shakeout_level=min(ma50[k], l[a6]) if l[k] < min(ma50[k], l[a6]) else max(ma50[k], l[a6]))
    if r["state"] == "VETO":                       # the anchor low is broken: fall back to the MA50 stop
        stop = ma50[i] - 0.5 * atr[i]
    r.update(supply_at(c, c * v, atr[i], i))
    r.setdefault("supply", "NOT COMPUTED")
    r["stop"] = stop
    r["stop_pct"] = (c[i] - stop) / c[i] * 100 if stop < c[i] else np.nan
    r["half_size"] = bool(r["stop_pct"] > 10) if not np.isnan(r["stop_pct"]) else False
    risk = c[i] - stop
    r["t1"] = c[i] + 2 * risk if risk > 0 else np.nan
    r["t2"] = r.get("shelf_low", np.nan)
    r["action"] = action_for(r["gate1"], r["state"], r["supply"])
    # exit-rule context (prompt: 2 closes < MA50 or close < 0.85 x running peak)
    r["closes_below_ma50"] = int((c[-2:] < ma50[-2:]).sum())
    return r


FIELDS = ["gate1", "state", "anchor_date", "anchor_volx", "days", "retention", "signature", "shakeout",
          "gap", "supply", "shelf_low", "shelf_top", "stop", "stop_pct", "half_size", "t1", "t2",
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
        elif isinstance(x, (np.floating, float)) and not np.isnan(x):
            x = round(float(x), 2)
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


def report(df: pd.DataFrame, ticker: str, held_price: float | None = None, held_date=None,
           risk_rupees: float = 10000.0, source: str = "Yahoo Finance daily") -> str:
    """Plain-text v6 report. held_price None = CANDIDATE mode."""
    df, provisional = drop_provisional(df)
    if df is None or len(df) < 60:
        return f"{ticker}: fewer than 60 closed sessions -- aborted (prompt rule)."
    r = analyse(df)
    c = df["Close"].astype(float)
    close = r["close"]
    lag = (pd.Timestamp.now().normalize() - pd.Timestamp(r["date"]).normalize()).days
    held = held_price is not None
    L = []
    L.append(f"{ticker} | close {close:,.2f} on {_d(r['date'])} | {r['sessions']} sessions | source {source} | lag {lag}d"
             + (" | today's candle PROVISIONAL, excluded" if provisional else ""))
    L.append(f"MODE: {'HELD @ ' + _p(held_price) + ' since ' + _d(held_date) if held else 'CANDIDATE'}")

    state, supply = r["state"], r["supply"]
    stop, sp = r["stop"], r["stop_pct"]
    risk = close - stop if stop < close else np.nan
    shares = int(risk_rupees / risk) if risk and risk > 0 else 0
    if r["half_size"]:
        shares //= 2
    # exit rule (prompt default): 2 closes < MA50, or close < 0.85 x running peak
    since = pd.Timestamp(held_date) if held and held_date is not None else c.index[-1]
    peak = float(c[c.index >= since].max()) if held else close
    trail = 0.85 * peak
    fired = None
    if held:
        m50 = c.rolling(50).mean()
        cc, mm = c[c.index >= since], m50[c.index >= since]
        below = (cc < mm)
        two = below & below.shift(1, fill_value=False)
        pk = cc.cummax()
        hit = two | (cc < 0.85 * pk)
        if hit.any():
            fired = hit[hit].index[0]

    if held:
        action = "EXIT" if fired is not None else "HOLD"
    else:
        action = r["action"]
    L.append(f"\nACTION: {action}")
    trig_a = trig_b = "NOT COMPUTED"
    if state.startswith("WAIT") and "anchor_low" in r:
        trig_a = f"stay above anchor low {_p(r['anchor_low'])} and reach >= 80% retention by day 5 (now day {r.get('days')})"
    if state == "GO":
        trig_a = (f"pullback toward MA50 {_p(r['ma50'])} on < 0.8x median volume, holding anchor low "
                  f"{_p(r.get('anchor_low'))}")
    if "shelf_top" in r:
        trig_b = f"daily close > {_p(r['shelf_top'])} (top of the shelf starting {_p(r['shelf_low'])}) into clean air"
    L.append(f"Trigger A: {trig_a}")
    L.append(f"Trigger B: {trig_b}")
    if stop < close:
        L.append(f"Stop: {_p(stop)} ({'MA50 - 0.5 ATR' if state == 'VETO' or 'anchor_low' not in r else 'max of anchor low, MA50 - 0.5 ATR'})"
                 f" = {_p(sp, 1)}%   Size: {shares} shares for Rs {risk_rupees:,.0f} risk"
                 + (" (HALVED: stop > 10%)" if r["half_size"] else ""))
    else:
        L.append(f"Stop: NOT COMPUTED -- price is below the stop level {_p(stop)} (no long setup)   Size: 0")
    L.append(f"Targets: T1 {_p(r['t1'])} (2R, reference only -- a 2R exit halved per-trade returns in the backtest) | "
             f"T2 {_p(r['t2'])} (nearest trapped shelf)")
    rr_now = (r["t1"] - close) / risk if risk and risk > 0 else np.nan
    L.append(f"R:R: from current {_p(rr_now, 1)} | from trigger "
             + (_p((r['t2'] - r['shelf_top']) / (r['shelf_top'] - stop), 1) if 'shelf_top' in r and not np.isnan(r['t2'])
                and r['shelf_top'] > stop and r['t2'] > r['shelf_top'] else "NOT COMPUTED"))
    rev = "day 5 of the anchor" if r.get("days", 99) < 5 else "+10 sessions"
    L.append(f"Review on: {rev}")

    L.append("\nGATES")
    L.append(f"1 Trend    {'PASS' if r['gate1'] else 'FAIL'}  close {close:,.2f} vs MA50 {_p(r['ma50'])} "
             f"({r['dist_ma50']:+.1f}%) / MA200 {_p(r['ma200'])} / 52w-high {_p(r['hi252'])} ({r['dist_hi']:+.1f}%)   [DATA]")
    if "anchor_date" in r:
        ret = r.get("retention")
        ret_s = "NOT COMPUTED" if ret is None or np.isnan(ret) else f"{round(ret * r['days'])}/{r['days']}"
        L.append(f"2 Event    {state}  {_d(r['anchor_date'])} | vol_ratio {r['anchor_volx']:.1f}x | clv {r['anchor_clv']:.2f} | "
                 f"turnover Rs {r['anchor_turnover_cr']:.1f} cr | retention "
                 f"{ret_s} | "
                 f"cause NOT COMPUTED   [DATA]")
    else:
        L.append("2 Event    NEUTRAL  no day in the last 60 sessions with >= 3x median volume and >= Rs 5 cr   [DATA]")
    if "shelf_low" in r:
        L.append(f"3 Supply   {'OVERHANG' if 'OVERHANG' in supply else 'CLEAR'}  nearest trapped shelf "
                 f"{_p(r['shelf_low'])}-{_p(r['shelf_top'])}, {(r['shelf_low'] - close) / r['atr']:.1f} ATR above   [DATA]")
    else:
        L.append(f"3 Supply   CLEAR  {SUPPLY_LABEL[supply]}: no trapped shelf above in the last 252 sessions   [DATA]")
    L.append("4 Levels   stop / targets / R:R as above")
    squeeze = "squeeze" if r["bbp"] < 20 else ("wide" if r["bbp"] > 80 else "normal")
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
    L.append(f"Institutional Stance: {stance}   [INFER]")

    L.append(f"\nEXIT RULE ({'HELD' if held else 'if bought today'})")
    L.append(f"MA50 {_p(r['ma50'])} -- distance {r['dist_ma50']:+.1f}% -- closes below: {r['closes_below_ma50']}/2")
    L.append(f"Peak {_p(peak)} (running since {_d(since)}) x 0.85 = {_p(trail)} -- distance {(close / trail - 1) * 100:+.1f}%")
    L.append(f"Status: {'FIRED on ' + _d(fired) if fired is not None else 'NOT FIRED'}")
    L.append("(The live engine exits on ONE close < MA50 or a 12/15% trail; the 2-close rule looked better per trade "
             "but worse at portfolio level -- FORENSICS_VERDICT.md.)")

    L.append("\nWHAT FLIPS THIS")
    bull = (f"daily close > {_p(r['shelf_top'])} (clean-air trigger)" if "shelf_top" in r and "OVERHANG" in supply
            else f"new volume anchor (>= 3x median, >= Rs 5 cr, CLV >= 0.5) above {_p(r['hi252'])}")
    bear = (f"close < anchor low {_p(r['anchor_low'])} (VETO)" if "anchor_low" in r and state not in ("VETO",)
            else f"2 closes < MA50 {_p(r['ma50'])}")
    L.append(f"Bullish flip: {bull}")
    L.append(f"Bearish flip: {bear}")

    L.append("\nUNKNOWNS / CAVEATS")
    L.append("- Anchor cause: NOT COMPUTED (no filing / news feed); delivery % NOT COMPUTED.")
    L.append("- Supply shelves use the last 252 sessions only; older supply is invisible.")
    L.append("- Gap = anchor-day low above the prior high (open prices not used, as in the backtest).")
    L.append("- Edges are 2016-26 medians vs the typical stock (FORENSICS_VERDICT.md), not a forecast for this stock.")
    return "\n".join(L)
