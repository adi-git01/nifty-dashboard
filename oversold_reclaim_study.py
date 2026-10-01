"""
Oversold bounce, then a close back above the 50-day average: is there alpha?
============================================================================

Setup: a stock gets oversold, bounces, and closes back above its MA50. The
event day is that first close above the MA50 (bought at that close).

Oversold, three definitions (a setup that works on only one is suspect):
  RSI30    RSI(14) < 30
  RSI25    RSI(14) < 25 (deeper)
  STRETCH  close >= 15% below its MA50

  OS->CROSS  the first close above the MA50 within W sessions of the last
             oversold day, with no MA50 cross in between (W = 20 / 40 / 60)

Comparisons, so the oversold part is isolated:
  every stock, weekly            the yardstick (0 by construction)
  any MA50 reclaim               the same cross with no oversold condition
                                 (breakout_sector_study: no edge, worse with volume)
  buy the oversold day itself    first RSI < 30 after >= 20 sessions without one --
                                 is waiting for the cross better than buying the dip?

Splits at the cross (RSI30 within 40 sessions): above / below MA200; MA50
rising / falling (10 sessions); cross-day volume >= 1.5x median; speed (days
from the oversold day); depth of the fall (low of the last 40 sessions vs the
high of the 120 before); bounce off the low; sub-industry band; Nifty above /
below its own 200dma.

Outcome: 2 weeks - 6 months vs the typical stock (median), the equal-weight
basket vs the average stock (mean), share that fell back under the MA50
within 2 weeks, years positive, two eras. Exits on the same entries: the
engine's rule (1 close < MA50 or 15% trail), a stop at the oversold low,
and fixed holds.

Run: python oversold_reclaim_study.py              (Actions: ranking_rotation_backtest.yml, script=oversold)
     python oversold_reclaim_study.py --synthetic  (random walk: every edge should be ~0)
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
H = (10, 21, 42, 63, 126)
MAX_HOLD = 126


def first_cross_after(os_flag, cross, window):
    """First MA50 cross within `window` sessions of the last oversold day, no cross in between."""
    from breakout_sector_study import since_last
    s_os = since_last(os_flag)                 # sessions since the last oversold day (as of yesterday)
    s_cx = since_last(cross)
    return cross & (s_os <= window) & ((s_cx > s_os) | s_cx.isna()), s_os


def exits(ent, close, ma50, nifty, low40):
    """Paired exits per entry: engine rule, stop at the oversold low, fixed holds."""
    C, M, N, LO = close.values, ma50.values, nifty.values, low40.values
    T = len(C)
    res = []
    for i, j in zip(ent.i.values, ent.j.values):
        e = min(i + MAX_HOLD, T - 1)
        if e - i < 21:
            continue
        p = pd.Series(C[i + 1:e + 1, j]).ffill().values
        m = M[i + 1:e + 1, j]
        if np.isnan(p).all():
            continue
        entry = C[i, j]
        peak = np.maximum.accumulate(np.r_[entry, p])[1:]
        below = p < m
        trail = p < 0.85 * peak
        stop = p < LO[i, j]
        first = lambda mk: int(np.argmax(mk)) if mk.any() else len(p) - 1
        rules = {"engine: 1 close<MA50 or 15% trail": below | trail,
                 "stop at the oversold low, else 15% trail": stop | trail,
                 "hold 63 sessions": np.arange(len(p)) >= min(62, len(p) - 1),
                 "hold 126 sessions": np.zeros(len(p), bool)}
        for name, mk in rules.items():
            k = first(mk)
            r = (p[k] / entry - 1) * 100
            res.append(dict(rule=name, ret=r, vs_nifty=r - (N[i + 1 + k] / N[i] - 1) * 100, days=k + 1, i=i))
    return pd.DataFrame(res)


def main():
    from breakout_sector_study import build, since_last, stats
    from forensics_gate_study import exit_table, real_data, synthetic
    from utils.forensics import indicators

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    tag = "_synthetic" if args.synthetic else ""

    close, high, low, vol, nifty, pit, gscore = synthetic() if args.synthetic else real_data(args.max_tickers)
    _, _, fwd, below50, retention, gscore, ma50, ma200 = build(close, high, low, vol, nifty, pit, gscore)
    rsi = indicators(close, high, low)[1]
    vmed = vol.rolling(50, min_periods=30).median().shift(1)
    cf = close.ffill(limit=5)
    cross = (close > ma50) & (cf.shift(1) <= ma50.shift(1))
    low40 = close.rolling(40, min_periods=20).min()
    hi120 = close.shift(40).rolling(120, min_periods=60).max()
    nma = nifty.rolling(200).mean()
    live = pit & (close.notna().cumsum() >= 260)

    os_defs = {"RSI30": rsi < 30, "RSI25": rsi < 25, "STRETCH": close <= 0.85 * ma50}
    ev, s_os = {}, {}
    for name, f in os_defs.items():
        for w in (20, 40, 60):
            ev[f"{name} -> cross <= {w}d"], s = first_cross_after(f, cross, w)
            if w == 40:
                s_os[name] = s
    rsi_new = (rsi < 30) & (since_last(rsi < 30) >= 20)

    def rows_of(mask):
        m = (mask & live).fillna(False).values
        ii, jj = np.where(m)
        d = pd.DataFrame({"i": ii, "j": jj, "date": close.index[ii]})
        gs = gscore.values[ii, jj]
        d["industry"] = np.select([gs >= 70, gs >= 40, gs >= 0], ["leader", "mid", "laggard"], "")
        for h in H:
            r, vn, vu, vm = fwd[h]
            d[f"ret{h}"], d[f"vn{h}"], d[f"vu{h}"], d[f"vm{h}"] = (x.values[ii, jj] for x in (r, vn, vu, vm))
        d["below50_2wk"], d["retention"] = below50.values[ii, jj], retention.values[ii, jj]
        return d[d.industry != ""]

    wk = pd.DataFrame(False, index=close.index, columns=close.columns)
    wk.iloc[260::5] = True
    T = [dict(group="0 yardsticks", **stats(rows_of(wk), "every stock, weekly")),
         dict(group="0 yardsticks", **stats(rows_of(cross), "any MA50 reclaim (no oversold)")),
         dict(group="0 yardsticks", **stats(rows_of(rsi_new), "buy the oversold day (RSI < 30)"))]
    for name, m in ev.items():
        T.append(dict(group="1 oversold -> MA50 cross", **stats(rows_of(m), name)))

    # splits on the main definition
    main_ev = ev["RSI30 -> cross <= 40d"]
    D = rows_of(main_ev)
    ii, jj = D.i.values, D.j.values
    D["above200"] = (close.values > ma200.values)[ii, jj]
    D["ma50_up"] = (ma50.values > ma50.shift(10).values)[ii, jj]
    D["volx"] = (vol.values / vmed.values)[ii, jj]
    D["speed"] = s_os["RSI30"].values[ii, jj]
    D["fall"] = (low40.values / hi120.values - 1)[ii, jj] * 100
    D["bounce"] = (close.values / low40.values - 1)[ii, jj] * 100
    D["rsi"] = rsi.values[ii, jj]
    D["nifty_up"] = (nifty.values > nma.values)[ii]
    D["era"] = np.where(D.date < ERA_SPLIT, "2016-20", "2021-26")
    S = lambda m, lab: dict(group="2 splits (RSI30 -> cross <= 40d)", **stats(D[m], lab))
    T += [S(D.above200, "above MA200 at the cross"), S(~D.above200, "below MA200 at the cross"),
          S(D.ma50_up, "MA50 rising"), S(~D.ma50_up, "MA50 falling"),
          S(D.above200 & D.ma50_up, "above MA200 and MA50 rising"),
          S(D.volx >= 1.5, "cross-day volume >= 1.5x median"), S(D.volx < 1.0, "cross-day volume < 1x median"),
          S(D.speed <= 15, "fast: cross <= 15 sessions after oversold"), S(D.speed > 15, "slow: 16-40 sessions"),
          S(D.fall <= -30, "deep fall (>= 30% off the 120d high)"), S(D.fall > -15, "shallow fall (< 15%)"),
          S(D.bounce >= 15, "big bounce (>= 15% off the low)"), S(D.bounce < 8, "small bounce (< 8%)"),
          S(D.industry == "leader", "leader industry"), S(D.industry == "mid", "mid industry"),
          S(D.industry == "laggard", "laggard industry"),
          S(D.nifty_up, "Nifty above its 200dma"), S(~D.nifty_up, "Nifty below its 200dma")]
    R = pd.DataFrame(T)
    os.makedirs(OUT, exist_ok=True)
    R.to_csv(f"{OUT}/oversold_reclaim_study{tag}.csv", index=False)
    print(f"\n{'=' * 150}\nOVERSOLD BOUNCE -> MA50 CROSS (pp). vsTypical = median event minus the median universe stock "
          f"over the same window; portfolio vsAvg = mean vs mean.\n{'=' * 150}")
    for g, x in R.groupby("group", sort=False):
        print(f"\n--- {g}")
        print(x.drop(columns="group").to_string(index=False))

    # eras for the rows that matter
    E = []
    for era in ("2016-20", "2021-26"):
        def em(mask):
            d = rows_of(mask)
            return d[(d.date < ERA_SPLIT)] if era == "2016-20" else d[d.date >= ERA_SPLIT]
        E += [dict(era=era, **stats(em(cross), "any MA50 reclaim")),
              dict(era=era, **stats(em(rsi_new), "buy the oversold day")),
              dict(era=era, **stats(em(main_ev), "RSI30 -> cross <= 40d")),
              dict(era=era, **stats(em(ev["STRETCH -> cross <= 40d"]), "STRETCH -> cross <= 40d")),
              dict(era=era, **stats(D[(D.era == era) & D.above200], "RSI30 cross, above MA200")),
              dict(era=era, **stats(D[(D.era == era) & ~D.above200], "RSI30 cross, below MA200")),
              dict(era=era, **stats(D[(D.era == era) & (D.industry == "leader")], "RSI30 cross, leader industry"))]
    E = pd.DataFrame(E)
    E.to_csv(f"{OUT}/oversold_reclaim_era{tag}.csv", index=False)
    print(f"\n{'=' * 150}\nBY ERA\n{'=' * 150}\n{E.to_string(index=False)}")

    X = []
    anyx = rows_of(cross)
    for lab, d in (("RSI30 -> cross <= 40d", D), ("  ...above MA200 at the cross", D[D.above200]),
                   ("any MA50 reclaim (sample)", anyx.sample(min(20000, len(anyx)), random_state=1))):
        Rx = exits(d, close, ma50, nifty, low40)
        if len(Rx):
            X += exit_table(Rx, close.index, lab)
    X = pd.DataFrame(X)
    X.to_csv(f"{OUT}/oversold_reclaim_exits{tag}.csv", index=False)
    print(f"\n{'=' * 150}\nEXITS on the same entries (return %, per trade; max hold {MAX_HOLD} sessions)\n{'=' * 150}")
    print(X.to_string(index=False))
    print(f"\nsaved -> {OUT}/oversold_reclaim_study{tag}.csv, oversold_reclaim_era{tag}.csv, oversold_reclaim_exits{tag}.csv")


if __name__ == "__main__":
    main()
