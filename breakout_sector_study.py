"""
Stock breakouts inside leading sub-industries: is there alpha?
===============================================================

Event study. A stock breaks out on day t (bought at that close):

  reclaim50   close crosses back above its 50-day average
  golden      50-day average crosses above the 200-day average
  high52      new 52-week closing high after >= 20 sessions without one
  ath         new closing high since 2014 (or listing), >= 500 sessions of
              history, after >= 20 sessions without one -- "all-time high"
              as far as the data goes

Confirmations (the chart checks):
  vol   volume >= 1.5x its prior 50-day average (the "strong volume" cut in
        utils/scoring.py)
  clv   close location value (C - L) / (H - L) >= 0.8: closed in the top fifth
        of the day's range

Context: the stock's sub-industry score_0_100 that day (the rotation heatmap's
score, rebuilt with the live formula): leader >= 70, mid 40-69, laggard < 40.

Outcome over 2 weeks, 1, 2, 3 and 6 months (10/21/42/63/126 sessions):
  vs Nifty, and vs the average universe stock over the same window (removes
  the market's move -- the alpha); share that beat the average stock; and
  whether it fizzled: below the entry price at the horizon, or back under its
  50-day average within 2 weeks.

Baseline: every universe stock, sampled weekly, split by the same sub-industry
band -- a breakout has to beat simply owning a stock in that kind of industry.

Events cluster in time (breakouts come in waves), so each result is also
checked year by year: the share of calendar years in which it beat the
average stock. Point-in-time top-1000 universe; delisted names are missing,
which flatters absolute returns for every row alike.

Run: python breakout_sector_study.py
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

H = [10, 21, 42, 63, 126]
HLAB = {10: "2wk", 21: "1mo", 42: "2mo", 63: "3mo", 126: "6mo"}
ERA_SPLIT = "2021-01-01"
OUT = "analysis"


def fetch_ohlcv(tickers, start):
    from exit_rule_backtest import align_to_sessions
    from transfer_backtest import cut_partial_last_day
    from utils.yf_safe import safe_download, safe_history

    end = datetime.now().strftime("%Y-%m-%d")
    nifty = safe_history("^NSEI", start=start, end=end)
    if nifty.empty:
        raise SystemExit("Could not fetch Nifty")
    if nifty.index.tz is not None:
        nifty.index = nifty.index.tz_localize(None)
    bulk = safe_download(tickers, start=start, end=end, group_by="ticker", threads=False,
                         auto_adjust=True, min_coverage=0.5)
    fr = {k: {} for k in ("Close", "High", "Low", "Volume")}
    for t in tickers:
        try:
            sub = bulk[t]
            c = sub["Close"].dropna()
            if len(c) > 250:
                for k in fr:
                    fr[k][t] = sub[k].reindex(c.index)
        except Exception:
            continue
    P = {k: pd.DataFrame(v) for k, v in fr.items()}
    for k in P:
        if P[k].index.tz is not None:
            P[k].index = P[k].index.tz_localize(None)
    close, vol = align_to_sessions(P["Close"], P["Volume"], nifty)
    close, vol = cut_partial_last_day(close, vol)
    high, low = P["High"].reindex(close.index), P["Low"].reindex(close.index)
    print(f"[data] {close.shape[1]} tickers, {len(close)} sessions {close.index[0].date()} -> {close.index[-1].date()}")
    return close, high, low, vol, nifty["Close"].reindex(close.index).ffill()


def since_last(flag):
    """Sessions since the previous True, as of the day before (NaN if none)."""
    pos = np.arange(len(flag), dtype=float)[:, None]
    last = pd.DataFrame(np.where(flag.values, pos, np.nan), index=flag.index, columns=flag.columns).ffill()
    return pd.DataFrame(pos - last.values, index=flag.index, columns=flag.columns).shift(1)


def build(close, high, low, vol, nifty, pit, gscore):
    from exit_rule_backtest import ma50_per_stock
    ma50 = ma50_per_stock(close)
    ma200 = close.apply(lambda s: s.dropna().rolling(200).mean()).reindex(close.index)
    cf = close.ffill(limit=5)
    hi252 = close.rolling(252, min_periods=200).max()
    new52 = close.ge(hi252) & hi252.notna()
    prior_max = close.ffill().cummax().shift(1)      # a missing day must not blank the running max
    nobs = close.notna().cumsum()
    ath = (close > prior_max) & (nobs >= 500)
    ev = {
        "reclaim50": (close > ma50) & (cf.shift(1) <= ma50.shift(1)),
        "golden": (ma50 > ma200) & (ma50.shift(1) <= ma200.shift(1)),
        "high52": new52 & (since_last(new52) >= 20),
        "ath": ath & (since_last(ath) >= 20),
    }
    rng = (high - low).where(lambda x: x > 0)
    clv = (close - low) / rng
    vavg = vol.rolling(50, min_periods=30).mean().shift(1)
    conf = {"vol": vol >= 1.5 * vavg, "clv": clv >= 0.8}

    fwd = {}
    for h in H:
        r = (cf.shift(-h) / close - 1) * 100
        n = (nifty.shift(-h) / nifty - 1) * 100
        # Stock returns are right-skewed, so the MEDIAN stock trails the MEAN stock
        # (on random data by 1-2 pp over 3-6 months). Typical-vs-typical uses the
        # cross-sectional median; the portfolio view uses means on both sides.
        med = r.where(pit).median(axis=1)
        avg = r.where(pit).mean(axis=1)
        fwd[h] = (r, r.sub(n, axis=0), r.sub(med, axis=0), r.sub(avg, axis=0))
    # fell back under the 50-day average at any close in the next 10 sessions
    under = (close < ma50).astype(float).where(close.notna())
    below50 = under[::-1].rolling(10, min_periods=1).max()[::-1].shift(-1)
    return ev, conf, fwd, below50, gscore


def band(s):
    return np.select([s >= 70, s >= 40, s >= 0], ["leader", "mid", "laggard"], "")


def collect(mask, conf, fwd, below50, gscore, pit):
    m = mask & pit
    ii, jj = np.where(m.fillna(False).values)
    d = pd.DataFrame({"date": m.index[ii], "ticker": m.columns[jj]})
    d["industry"] = band(gscore.values[ii, jj])
    for k, v in conf.items():
        d[k] = v.values[ii, jj].astype(bool)
    for h in H:
        r, vn, vu, vm = fwd[h]
        d[f"ret{h}"], d[f"vn{h}"] = r.values[ii, jj], vn.values[ii, jj]
        d[f"vu{h}"], d[f"vm{h}"] = vu.values[ii, jj], vm.values[ii, jj]
    d["below50_2wk"] = below50.values[ii, jj]
    return d[d.industry != ""]


def stats(d, label):
    r = dict(case=label, n=len(d))
    for h in H:
        x = d[f"vu{h}"].dropna()
        if len(x) < 30:
            continue
        r[f"{HLAB[h]} vsTypical"] = round(x.median(), 2)
    for h in (21, 63, 126):
        x = d[f"vu{h}"].dropna()
        if len(x) >= 30:
            r[f"{HLAB[h]} beat%"] = round((x > 0).mean() * 100)
            r[f"{HLAB[h]} vsNifty"] = round(d[f"vn{h}"].dropna().median(), 2)
    for h in (63, 126):
        x = d[f"vm{h}"].dropna()
        if len(x) >= 30:
            r[f"{HLAB[h]} portfolio vsAvg"] = round(x.mean(), 2)
    for h in (10, 63):
        x = d[f"ret{h}"].dropna()
        if len(x) >= 30:
            r[f"below entry {HLAB[h]}%"] = round((x < 0).mean() * 100)
    if d.below50_2wk.notna().sum() >= 30:
        r["under 50dma in 2wk%"] = round(d.below50_2wk.mean() * 100)
    y = d.dropna(subset=["vu63"]).groupby(d.date.dt.year).vu63.median()
    y = y[d.dropna(subset=["vu63"]).groupby(d.date.dt.year).size() >= 10]
    if len(y):
        r["yrs 3mo>0"] = f"{(y > 0).sum()}/{len(y)}"
    return r


def main():
    from momentum_factor_backtest import build_pit_universe, load_candidates
    from transfer_backtest import group_panels
    from utils.nifty1000_list import SUB_INDUSTRY_MAP

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)

    close, high, low, vol, nifty = fetch_ohlcv(load_candidates("all", args.max_tickers), "2014-06-01")
    pit = build_pit_universe(close, vol, 1000)
    _, keys, n = group_panels(close, pit, SUB_INDUSTRY_MAP, nifty, [(5, .10), (21, .50), (63, .40)],
                              india_live=True)
    score = keys["live"].rank(axis=1, pct=True) * 100
    score = score.where(score.notna().sum(axis=1) >= 20)
    # each stock gets its sub-industry's score that day
    gmap = {t: SUB_INDUSTRY_MAP.get(t) for t in close.columns}
    gscore = pd.DataFrame({t: score[g] if g in score.columns else np.nan for t, g in gmap.items()},
                          index=close.index)
    pit = pit & gscore.notna()
    ev, conf, fwd, below50, gscore = build(close, high, low, vol, nifty, pit, gscore)

    # baseline: every universe stock, weekly
    wk = pd.DataFrame(False, index=close.index, columns=close.columns)
    wk.iloc[::5] = True
    base = collect(wk, conf, fwd, below50, gscore, pit)

    rows = []
    for b in ("leader", "mid", "laggard"):
        rows.append(dict(breakout="BASELINE any stock, weekly", **stats(base[base.industry == b], f"{b} industry")))
    allev = []
    for name, m in ev.items():
        d = collect(m, conf, fwd, below50, gscore, pit)
        d["breakout"] = name
        allev.append(d)
        for b in ("leader", "mid", "laggard"):
            x = d[d.industry == b]
            rows.append(dict(breakout=name, **stats(x, f"{b} industry")))
            rows.append(dict(breakout=name, **stats(x[x.vol], f"{b} + volume")))
            rows.append(dict(breakout=name, **stats(x[x.vol & x.clv], f"{b} + volume + CLV")))
    T = pd.DataFrame(rows)
    T.to_csv(f"{OUT}/breakout_sector_study.csv", index=False)

    print(f"\n{'=' * 150}\nBREAKOUTS BY SUB-INDUSTRY STRENGTH (pp).  vsTypical = median breakout minus the median universe "
          f"stock over the same window.\nportfolio vsAvg = mean breakout minus mean stock (what an equal-weight basket of these "
          f"would add).  beat% = share beating the median stock.\n'yrs 3mo>0' = calendar years in which the 3-month "
          f"vsTypical was positive.  Compare every row with the BASELINE row for the same industry band.\n{'=' * 150}")
    for name in ["BASELINE any stock, weekly"] + list(ev):
        print(f"\n--- {name}")
        print(T[T.breakout == name].drop(columns="breakout").to_string(index=False))

    E = pd.concat(allev)
    E["era"] = np.where(E.date < ERA_SPLIT, "2016-20", "2021-26")
    print(f"\n{'=' * 150}\nBY ERA -- leader industry + volume + CLV vs the same breakout in laggard industries\n{'=' * 150}")
    er = []
    for name in ev:
        for era in ("2016-20", "2021-26"):
            x = E[(E.breakout == name) & (E.era == era)]
            er.append(dict(breakout=name, era=era, **stats(x[(x.industry == "leader") & x.vol & x.clv], "leader+vol+CLV")))
            er.append(dict(breakout=name, era=era, **stats(x[x.industry == "laggard"], "laggard, any")))
    print(pd.DataFrame(er).to_string(index=False))

    # recent qualifying events, for a look at what it picks now
    last = E[(E.date >= close.index[-30]) & (E.industry == "leader") & E.vol & E.clv]
    last = last[["date", "ticker", "breakout"]].sort_values("date", ascending=False)
    last["ticker"] = last.ticker.str.replace(".NS", "", regex=False)
    last["sub_industry"] = last.ticker.add(".NS").map(SUB_INDUSTRY_MAP)
    last.to_csv(f"{OUT}/breakout_sector_recent.csv", index=False)
    print(f"\nLAST 30 SESSIONS -- breakouts in leader industries with volume + CLV: {len(last)}")
    print(last.head(40).to_string(index=False))
    print(f"\nsaved -> {OUT}/breakout_sector_study.csv, {OUT}/breakout_sector_recent.csv")


if __name__ == "__main__":
    main()
