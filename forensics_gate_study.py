"""
Do the "Structural Forensics" chart-prompt gates (v6) add anything?
===================================================================

The prompt turns a chart into BUY NOW / BUY ON TRIGGER / WATCH through five
gates. Each gate is rebuilt here from daily bars for every stock, sampled
weekly, 2016-2026, point-in-time top-1000 universe, and judged by what came
next: 2 weeks to 6 months, vs the typical (median) universe stock over the
same window, year by year and in two eras.

Gate 1 Trend      close > MA50 and > MA200 and within 15% of the 252-day high.
                  Everything below is tested INSIDE Gate 1, so each row is
                  compared with "Gate 1 pass, any" -- the question is whether
                  the later gates add to the trend filter, not whether trend
                  beats the market (breakout_sector_study.py covers that).
Gate 2 Event      anchor = the highest-volume day in the last 60 sessions with
                  volume >= 3x its prior 50-session median and turnover
                  >= Rs 5 cr. Direction by CLV (+1 >= 0.50, -1 <= 0.30, else
                  mixed). For a +1 anchor:
                    VETO  a later close below the anchor-day low
                    GO    day >= 5 and >= 80% of closes since the anchor held
                          at or above the close before it (retention)
                    WAIT  day < 5, or retention 40-80%
                    FADE  day >= 5 and retention < 40%
                  Since the anchor: mean CLV (>= 0.5 = closing strong);
                  pullback signature -- DISTRIBUTION if a down day ran >= 1.5x
                  median volume while closing below the anchor close,
                  ABSORPTION if down days averaged <= 0.8x median volume (dry
                  up) with no veto; the prompt's literal "down-day volume 15-40%
                  of the anchor day" is reported separately; shakeout probe --
                  a day that broke the MA50 or the anchor low intraday, closed
                  back above it with CLV >= 0.6 on < 0.8x median volume;
                  displacement -- the anchor day's low above the prior high
                  (a gap that held; open prices are not downloaded).
Gate 3 Supply     volume-by-price over the last 252 sessions in 3% price bins;
                  a shelf is a bin with >= 2x the average bin's turnover, and
                  it is TRAPPED if the stock has not closed at or above it for
                  >= 21 sessions. NEAR OVERHANG = a trapped shelf starting
                  within 1.5 ATR above the close; INSIDE = the close sits in a
                  shelf (absorption zone); FAR = trapped shelf further up;
                  CLEAN AIR = no trapped shelf above.
Gate 4 Levels     stop = max(anchor low, MA50 - 0.5 ATR14); stop % of entry.
Gate 5 Lens       RSI14, ADX14 with +DI/-DI, Bollinger(20,2) width percentile
                  vs the stock's own last 252 sessions.

The prompt's decision:  BUY NOW = Gate 1 + GO + no near overhang;
BUY ON TRIGGER = Gate 1 + GO + near overhang.

v5 vs v6 (section 7), same rows: v6 takes the HIGHEST-volume qualifying day
of the last 60 sessions and measures retention / signature / CLV over every
session since it, and adds the shakeout probe; v5 takes the MOST RECENT
qualifying day and measures its first 15 sessions ("retention n/15"). Both
veto on any close below the anchor low. Gates 1, 3, 4 are identical.

Exits (paired, same entries): the prompt's rule (2 closes < MA50 or close
< 0.85 x running peak), + its initial stop, + a 2R target, vs the live
engine's rule (1 close < MA50 or 15% trail). Max hold 126 sessions.

Every cell is one more test; ~40 are run. A gate counts only if it beats
"Gate 1 pass, any" in BOTH eras and in most years -- not on one median.

Run: python forensics_gate_study.py            (Actions: ranking_rotation_backtest.yml, script=forensics)
     python forensics_gate_study.py --synthetic (random-walk data, offline: every edge should be ~0)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

OUT = "analysis"
ERA_SPLIT = "2021-01-01"
LOOK = 60           # anchor search window
SHELF_WIN = 252
BIN = np.log(1.03)
MAX_HOLD = 126


# ----------------------------------------------------------------------------
# indicators (panels: dates x tickers)
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
# per-stock gate features at the sampled rows
# ----------------------------------------------------------------------------
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
        f["shakeout"] = bool((probe & lowv & strong).any())
    return f, stop


def stock_features(c, h, l, v, vmed, ma50, atr, rows):
    """
    c..atr: 1-d arrays for one stock; rows: sample indices (Gate 1 passes).
    v6 fields unprefixed (anchor = highest-volume qualifying day in 60
    sessions, measured since); v5 fields prefixed v5_ (anchor = the most
    recent qualifying day, retention over its first 15 sessions).
    """
    clv = np.where(h > l, (c - l) / np.where(h > l, h - l, 1), np.nan)
    turn = c * v
    qual = (v >= 3 * vmed) & (turn >= 5e7) & ~np.isnan(c)
    out = []
    for i in rows:
        f = dict(i=i)
        lo = max(2, i - LOOK + 1)
        q = qual[lo:i + 1]
        if not q.any():
            f["state"] = f["v5_state"] = "NO ANCHOR"
            stop = ma50[i] - 0.5 * atr[i]
        else:
            a6 = lo + int(np.argmax(np.where(q, v[lo:i + 1], -1.0)))
            a5 = lo + int(np.where(q)[0][-1])
            g6, stop = grade(c, h, l, v, vmed, ma50, atr, clv, a6, i)
            g5, stop5 = grade(c, h, l, v, vmed, ma50, atr, clv, a5, i, window=15)
            f.update(g6)
            f.update({f"v5_{k}": x for k, x in g5.items()})
            f["same_anchor"] = a5 == a6
            f["v5_stop_pct"] = (c[i] - stop5) / c[i] * 100 if stop5 < c[i] else np.nan
        f["stop_pct"] = (c[i] - stop) / c[i] * 100 if stop < c[i] else np.nan
        f.setdefault("v5_stop_pct", f["stop_pct"])
        # Gate 3: supply shelves from volume-by-price
        w0 = max(0, i - SHELF_WIN)
        cw, tw = c[w0:i], turn[w0:i]
        okw = ~np.isnan(cw) & ~np.isnan(tw)
        if okw.sum() >= 120:
            cw, tw = cw[okw], tw[okw]
            idx = np.arange(w0, i)[okw]
            b = np.floor(np.log(cw) / BIN).astype(int)
            b0 = b.min()
            tb = np.bincount(b - b0, weights=tw)
            occ = tb > 0
            shelf = np.where(occ & (tb >= 2 * tb[occ].mean()))[0] + b0
            cur = int(np.floor(np.log(c[i]) / BIN))
            status, near_top = "CLEAN AIR", np.nan
            if cur in set(shelf):
                status = "INSIDE shelf"
            trapped_above = []
            for sb in shelf:
                L = np.exp(sb * BIN)
                if L <= c[i]:
                    continue
                vis = idx[cw >= L]
                if len(vis) and i - vis[-1] >= 21:
                    trapped_above.append(L)
            if trapped_above:
                L = min(trapped_above)
                if L <= c[i] + 1.5 * atr[i]:
                    status = "NEAR OVERHANG"
                elif status == "CLEAN AIR":
                    status = "FAR OVERHANG"
            f["supply"] = status
        out.append(f)
    return out


# ----------------------------------------------------------------------------
# paired exit simulation
# ----------------------------------------------------------------------------
def simulate_exits(ent, close, ma50, nifty, stop_pct):
    """ent: DataFrame with i, j. Returns one row per entry per exit rule."""
    C, M, N = close.values, ma50.values, nifty.values
    T = len(C)
    res = []
    for (i, j, sp) in zip(ent.i.values, ent.j.values, stop_pct):
        e = min(i + MAX_HOLD, T - 1)
        if e - i < 21:
            continue
        p = C[i + 1:e + 1, j].copy()
        m = M[i + 1:e + 1, j]
        valid = ~np.isnan(p)
        p = pd.Series(p).ffill().values
        if np.isnan(p).all():
            continue
        entry = C[i, j]
        peak = np.maximum.accumulate(np.r_[entry, p])[1:]
        below = (p < m) & valid
        two = below & np.r_[False, below[:-1]]
        trail15 = p < 0.85 * peak
        stop = entry * (1 - sp / 100) if not np.isnan(sp) else -np.inf
        hit_stop = p < stop
        t1 = p >= entry * (1 + 2 * sp / 100) if not np.isnan(sp) else np.zeros(len(p), bool)

        def first(mask):
            k = np.argmax(mask) if mask.any() else len(p) - 1
            return k

        rules = {
            "engine: 1 close<MA50 or 15% trail": below | trail15,
            "prompt: 2 closes<MA50 or 15% trail": two | trail15,
            "prompt + initial stop": two | trail15 | hit_stop,
            "prompt + stop + 2R target": two | trail15 | hit_stop | t1,
            "hold 126 sessions": np.zeros(len(p), bool),
        }
        for name, mk in rules.items():
            k = first(mk)
            r = (p[k] / entry - 1) * 100
            n = (N[i + 1 + k] / N[i] - 1) * 100
            res.append(dict(rule=name, ret=r, vs_nifty=r - n, days=k + 1, i=i))
    return pd.DataFrame(res)


def exit_table(R, dates, label):
    rows = []
    for name, g in R.groupby("rule", sort=False):
        w = g.ret > 0
        yr = pd.Series(dates[g.i.values].year, index=g.index)
        by = g.groupby(yr).vs_nifty.mean()
        rows.append(dict(entries=label, rule=name, trades=len(g), win_pct=round(w.mean() * 100),
                         avg_win=round(g.ret[w].mean(), 1), avg_loss=round(g.ret[~w].mean(), 1),
                         mean_ret=round(g.ret.mean(), 2), median_ret=round(g.ret.median(), 2),
                         mean_vs_nifty=round(g.vs_nifty.mean(), 2), avg_days=round(g.days.mean()),
                         yrs_vs_nifty_pos=f"{(by > 0).sum()}/{len(by)}"))
    return rows


# ----------------------------------------------------------------------------
def synthetic(n_stocks=200, n_days=2600, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    cols = [f"S{k:03d}.NS" for k in range(n_stocks)]
    r = rng.normal(0.0004, 0.022, (n_days, n_stocks)) + rng.normal(0, 0.009, (n_days, 1))
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), idx, cols)
    rngd = np.abs(rng.normal(0, 0.015, (n_days, n_stocks)))
    u = rng.random((n_days, n_stocks))
    high = close * (1 + rngd * (1 - u) + 0.002)
    low = close * (1 - rngd * u - 0.002)
    vol = pd.DataFrame(np.exp(rng.normal(13, 0.6, (n_days, n_stocks))), idx, cols)
    vol *= np.where(rng.random((n_days, n_stocks)) < 0.01, rng.uniform(3, 8, (n_days, n_stocks)), 1)
    nifty = close.mean(axis=1)
    pit = close.notna()
    grp = rng.random((n_days // 21 + 1, n_stocks)) * 100
    gscore = pd.DataFrame(np.repeat(grp, 21, axis=0)[:n_days], idx, cols)
    return close, high, low, vol, nifty, pit, gscore


def real_data(max_tickers):
    from breakout_sector_study import fetch_ohlcv
    from momentum_factor_backtest import build_pit_universe, load_candidates
    from transfer_backtest import group_panels
    from utils.nifty1000_list import SUB_INDUSTRY_MAP

    close, high, low, vol, nifty = fetch_ohlcv(load_candidates("all", max_tickers), "2014-06-01")
    pit = build_pit_universe(close, vol, 1000)
    _, keys, _ = group_panels(close, pit, SUB_INDUSTRY_MAP, nifty, [(5, .10), (21, .50), (63, .40)], india_live=True)
    score = keys["live"].rank(axis=1, pct=True) * 100
    score = score.where(score.notna().sum(axis=1) >= 20)
    gscore = pd.DataFrame({t: score[g] if (g := SUB_INDUSTRY_MAP.get(t)) in score.columns else np.nan
                           for t in close.columns}, index=close.index)
    return close, high, low, vol, nifty, pit & gscore.notna(), gscore


def main():
    from breakout_sector_study import build, stats

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    tag = "_synthetic" if args.synthetic else ""

    close, high, low, vol, nifty, pit, gscore = synthetic() if args.synthetic else real_data(args.max_tickers)
    _, _, fwd, below50, retention, gscore, ma50, ma200 = build(close, high, low, vol, nifty, pit, gscore)
    atr, rsi, adx, pdi, ndi, bbp = indicators(close, high, low)
    vmed = vol.rolling(50, min_periods=30).median().shift(1)
    hi252 = close.rolling(252, min_periods=200).max()
    gate1 = (close > ma50) & (close > ma200) & (close >= 0.85 * hi252)

    sample = pd.DataFrame(False, index=close.index, columns=close.columns)
    sample.iloc[260::5] = True                      # weekly, after a year of history
    sample &= pit
    g1 = sample & gate1
    print(f"[rows] stock-weeks {int(sample.values.sum())}, Gate 1 pass {int(g1.values.sum())} "
          f"({g1.values.sum() / max(sample.values.sum(), 1):.0%})", flush=True)

    feats = []
    C, Hh, Ll, V, VM, M, A = (x.values for x in (close, high, low, vol, vmed, ma50, atr))
    for j, t in enumerate(close.columns):
        rows = np.where(g1.values[:, j])[0]
        if not len(rows):
            continue
        for f in stock_features(C[:, j], Hh[:, j], Ll[:, j], V[:, j], VM[:, j], M[:, j], A[:, j], rows):
            f["j"] = j
            feats.append(f)
        if j % 200 == 0:
            print(f"   features {j}/{close.shape[1]}", flush=True)
    F = pd.DataFrame(feats)
    ii, jj = F.i.values, F.j.values
    F["date"] = close.index[ii]
    F["ticker"] = close.columns[jj]
    gs = gscore.values[ii, jj]
    F["industry"] = np.select([gs >= 70, gs >= 40, gs >= 0], ["leader", "mid", "laggard"], "")
    for h in (10, 21, 42, 63, 126):
        r, vn, vu, vm = fwd[h]
        F[f"ret{h}"], F[f"vn{h}"] = r.values[ii, jj], vn.values[ii, jj]
        F[f"vu{h}"], F[f"vm{h}"] = vu.values[ii, jj], vm.values[ii, jj]
    F["below50_2wk"] = below50.values[ii, jj]
    F["retention2wk"] = retention.values[ii, jj]
    for k, p in dict(rsi=rsi, adx=adx, pdi=pdi, ndi=ndi, bbp=bbp).items():
        F[k] = p.values[ii, jj]
    F["era"] = np.where(F.date < ERA_SPLIT, "2016-20", "2021-26")

    # baseline of every sampled stock, for scale
    bi, bj = np.where(sample.values)
    B = pd.DataFrame({"date": close.index[bi]})
    for h in (10, 21, 42, 63, 126):
        r, vn, vu, vm = fwd[h]
        B[f"ret{h}"], B[f"vn{h}"], B[f"vu{h}"], B[f"vm{h}"] = (x.values[bi, bj] for x in (r, vn, vu, vm))
    B["below50_2wk"], B["retention"] = below50.values[bi, bj], retention.values[bi, bj]

    def st(d, label, group):
        d = d.rename(columns={"retention2wk": "retention"}) if "retention2wk" in d else d
        if "retention" not in d:
            d = d.assign(retention=np.nan)
        return dict(group=group, **stats(d, label))

    F2 = F.drop(columns=["retention"], errors="ignore")
    buy_now = (F.state == "GO") & (F.supply != "NEAR OVERHANG")
    trig = (F.state == "GO") & (F.supply == "NEAR OVERHANG")
    rows = [st(B, "every stock (weekly)", "0 scale"), st(F2, "Gate 1 pass, any", "0 scale")]
    for s in ["NO ANCHOR", "WAIT (running)", "GO", "WAIT (40-80%)", "FADE", "VETO", "MIXED anchor", "BEAR anchor"]:
        rows.append(st(F2[F.state == s], s, "2 event state"))
    up = F.state.isin(["GO", "WAIT (40-80%)", "FADE", "WAIT (running)"])
    for s in ["ABSORPTION", "NEUTRAL", "DISTRIBUTION"]:
        rows.append(st(F2[up & (F.signature == s)], f"+1 anchor, {s}", "2 pullback"))
    rows.append(st(F2[up & F.literal_absorb.eq(True)], "+1 anchor, literal 15-40% of anchor vol", "2 pullback"))
    rows.append(st(F2[up & F.shakeout.eq(True)], "+1 anchor, shakeout probe", "2 pullback"))
    rows.append(st(F2[up & F.shakeout.eq(False)], "+1 anchor, no shakeout", "2 pullback"))
    rows.append(st(F2[up & (F.cum_clv >= 0.5)], "+1 anchor, CLV since anchor >= 0.5", "2 pullback"))
    rows.append(st(F2[up & (F.cum_clv < 0.5)], "+1 anchor, CLV since anchor < 0.5", "2 pullback"))
    rows.append(st(F2[up & F.gap.eq(True)], "+1 anchor, gap (displacement)", "2 pullback"))
    rows.append(st(F2[up & F.gap.eq(False)], "+1 anchor, no gap", "2 pullback"))
    for s in ["CLEAN AIR", "INSIDE shelf", "FAR OVERHANG", "NEAR OVERHANG"]:
        rows.append(st(F2[F.supply == s], s, "3 supply"))
    rows.append(st(F2[F.stop_pct <= 10], "stop <= 10%", "4 stop"))
    rows.append(st(F2[F.stop_pct > 10], "stop > 10%", "4 stop"))
    rows += [st(F2[F.rsi > 70], "RSI > 70", "5 lens"), st(F2[F.rsi.between(50, 70)], "RSI 50-70", "5 lens"),
             st(F2[(F.adx > 25) & (F.pdi > F.ndi)], "ADX > 25 and +DI > -DI", "5 lens"),
             st(F2[F.adx < 20], "ADX < 20", "5 lens"),
             st(F2[F.bbp < 20], "Bollinger width < 20th pct (squeeze)", "5 lens"),
             st(F2[F.bbp > 80], "Bollinger width > 80th pct", "5 lens")]
    rows += [st(F2[buy_now], "BUY NOW (GO, no near overhang)", "6 decision"),
             st(F2[trig], "BUY ON TRIGGER (GO + near overhang)", "6 decision"),
             st(F2[buy_now & (F.industry == "leader")], "BUY NOW, leader industry", "6 decision"),
             st(F2[(F.industry == "leader")], "Gate 1 pass, leader industry", "6 decision"),
             st(F2[buy_now & (F.signature == "ABSORPTION")], "BUY NOW + absorption", "6 decision")]
    # ---- v5 vs v6: same rows, only the anchor choice and measuring window differ
    buy5 = (F.v5_state == "GO") & (F.supply != "NEAR OVERHANG")
    up5 = F.v5_state.isin(["GO", "WAIT (40-80%)", "FADE", "WAIT (running)"])
    for s in ["GO", "WAIT (40-80%)", "FADE", "VETO", "WAIT (running)"]:
        rows.append(st(F2[F.state == s], f"v6 {s}", "7 v5 vs v6"))
        rows.append(st(F2[F.v5_state == s], f"v5 {s}", "7 v5 vs v6"))
    for s in ["ABSORPTION", "DISTRIBUTION"]:
        rows.append(st(F2[up & (F.signature == s)], f"v6 {s}", "7 v5 vs v6"))
        rows.append(st(F2[up5 & (F.v5_signature == s)], f"v5 {s}", "7 v5 vs v6"))
    rows += [st(F2[buy_now], "v6 BUY NOW", "7 v5 vs v6"), st(F2[buy5], "v5 BUY NOW", "7 v5 vs v6"),
             st(F2[buy_now & (F.industry == "leader")], "v6 BUY NOW, leader", "7 v5 vs v6"),
             st(F2[buy5 & (F.industry == "leader")], "v5 BUY NOW, leader", "7 v5 vs v6"),
             st(F2[buy_now & buy5], "GO in both", "7 v5 vs v6"),
             st(F2[buy_now & ~buy5], "GO in v6 only", "7 v5 vs v6"),
             st(F2[buy5 & ~buy_now], "GO in v5 only", "7 v5 vs v6"),
             st(F2[up & F.shakeout.eq(True) & (F.state == "GO")], "v6 GO + shakeout (v6-only probe)", "7 v5 vs v6")]
    has = F.same_anchor.notna()
    print(f"[v5 vs v6] rows with an anchor {int(has.sum())}; same anchor day in {F.same_anchor[has].astype(bool).mean():.0%}; "
          f"BUY NOW v6 {int(buy_now.sum())}, v5 {int(buy5.sum())}, both {int((buy_now & buy5).sum())}")
    T = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    T.to_csv(f"{OUT}/forensics_gate_study{tag}.csv", index=False)

    print(f"\n{'=' * 150}\nGATES INSIDE GATE 1 (pp). vsTypical = median row minus the median universe stock over the "
          f"same window. Compare each row with 'Gate 1 pass, any'.\n{'=' * 150}")
    for g, x in T.groupby("group", sort=False):
        print(f"\n--- {g}")
        print(x.drop(columns="group").to_string(index=False))

    # by era: the rows that matter, vs Gate 1 pass
    er = []
    for era in ("2016-20", "2021-26"):
        e = F.era == era
        for lab, m in [("Gate 1 pass, any", e), ("v6 GO", e & (F.state == "GO")), ("v6 VETO", e & (F.state == "VETO")),
                       ("NO ANCHOR", e & (F.state == "NO ANCHOR")),
                       ("ABSORPTION", e & up & (F.signature == "ABSORPTION")),
                       ("DISTRIBUTION", e & up & (F.signature == "DISTRIBUTION")),
                       ("shakeout", e & up & F.shakeout.eq(True)),
                       ("CLEAN AIR", e & (F.supply == "CLEAN AIR")), ("NEAR OVERHANG", e & (F.supply == "NEAR OVERHANG")),
                       ("v6 BUY NOW", e & buy_now), ("v5 GO", e & (F.v5_state == "GO")),
                       ("v5 VETO", e & (F.v5_state == "VETO")), ("v5 BUY NOW", e & buy5),
                       ("v5 ABSORPTION", e & up5 & (F.v5_signature == "ABSORPTION")),
                       ("v5 DISTRIBUTION", e & up5 & (F.v5_signature == "DISTRIBUTION"))]:
            er.append(dict(era=era, **st(F2[m], lab, "era")))
    E = pd.DataFrame(er).drop(columns="group")
    E.to_csv(f"{OUT}/forensics_gate_era{tag}.csv", index=False)
    print(f"\n{'=' * 150}\nBY ERA\n{'=' * 150}\n{E.to_string(index=False)}")

    # paired exits
    g1s = F.sample(min(len(F), 25000), random_state=1)
    ex = []
    for lab, ent, sp in (("Gate 1 pass (sample)", g1s, "stop_pct"), ("v6 BUY NOW", F[buy_now], "stop_pct"),
                         ("v5 BUY NOW", F[buy5], "v5_stop_pct")):
        R = simulate_exits(ent, close, ma50, nifty, ent[sp].values)
        if len(R):
            ex += exit_table(R, close.index, lab)
    X = pd.DataFrame(ex)
    X.to_csv(f"{OUT}/forensics_exit_study{tag}.csv", index=False)
    print(f"\n{'=' * 150}\nEXITS on the same entries (return %, per trade; max hold {MAX_HOLD} sessions)\n{'=' * 150}")
    print(X.to_string(index=False))
    print(f"\nsaved -> {OUT}/forensics_gate_study{tag}.csv, forensics_gate_era{tag}.csv, forensics_exit_study{tag}.csv")


if __name__ == "__main__":
    main()
