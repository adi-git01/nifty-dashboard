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

  anchor_up   a bullish volume anchor on its own (no breakout needed)

Confirmations -- the user's chart-tool definitions:
  anchor  VolRatio = volume / median(volume, prior 50 sessions) >= 3.0 AND
          close x volume >= Rs 5 crore, with bullish direction CLV >= 0.50
  clv80   close location value (C - L) / (H - L) >= 0.8 (closed in the top
          fifth of the day's range)

Event retention (2 weeks): share of the next 10 sessions whose close held at
or above the close before the event day -- the day's gain not given back.

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
    vmed = vol.rolling(50, min_periods=30).median().shift(1)
    anchor = (vol >= 3.0 * vmed) & (close * vol >= 5e7) & (clv >= 0.5)
    ev["anchor_up"] = anchor & (since_last(anchor) >= 5)
    conf = {"anchor": anchor, "clv80": clv >= 0.8}
    prev = cf.shift(1)
    held = sum((cf.shift(-k) >= prev).astype(float) for k in range(1, 11))
    retention = (held / 10).where(cf.shift(-10).notna())

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
    return ev, conf, fwd, below50, retention, gscore, ma50, ma200


def scanner_tags(close, nifty, ma50, ma200):
    """
    The Trend Scanner's entry_label (utils/entry_timing.add_entry_freshness),
    rebuilt for every stock and day. Same thresholds and precedence:
      Weak         not in an uptrend (trend_score < 55 and price <= MA50)
      Late/Fading  rs_accel = rs_1w - rs_1m/4 < -6  (RS decelerating)
      Extended     > 18% above MA50
      Pullback Buy <= 6% above MA50 (incl. below)
      Actionable   6-18% above MA50
    trend_score is utils/scoring.calculate_trend_metrics with per-stock MAs;
    rs_1w / rs_1m are 5 / 21-session returns minus Nifty's, in points, as in
    utils/fast_data_engine.
    """
    dist = (close / ma50 - 1) * 100
    hi = close.rolling(252, min_periods=50).max()
    lo = close.rolling(252, min_periods=50).min()
    ts = pd.DataFrame(50.0, index=close.index, columns=close.columns)
    ts += np.where(ma50.notna(), np.where(close > ma50, 15, -10), 0)
    ts += np.where(ma200.notna(), np.where(close > ma200, 15, -15), 0)
    ts += np.where(ma50.notna() & ma200.notna(), np.where(ma50 > ma200, 10, -5), 0)
    rng = hi - lo
    ts += np.trunc((((close - lo) / rng.where(rng > 0)) - 0.5).fillna(0) * 30)
    d52 = (close - hi) / hi * 100
    ts += np.where(rng > 0, np.where(d52 > -5, 10, np.where(d52 < -30, -10, 0)), 0)
    ts = ts.clip(0, 100)
    cf = close.ffill(limit=5)
    ret = lambda k: (close / cf.shift(k) - 1) * 100
    nret = lambda k: (nifty / nifty.shift(k) - 1) * 100
    accel = ret(5).sub(nret(5), axis=0) - ret(21).sub(nret(21), axis=0) / 4
    up = (ts >= 55) | (dist > 0)
    tag = np.select([dist.isna(), ~up, accel < -6, dist > 18, dist <= 6],
                    ["", "Weak", "Late/Fading", "Extended", "Pullback Buy"], "Actionable")
    return pd.DataFrame(tag, index=close.index, columns=close.columns)


def band(s):
    return np.select([s >= 70, s >= 40, s >= 0], ["leader", "mid", "laggard"], "")


def collect(mask, conf, fwd, below50, retention, gscore, pit):
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
    d["retention"] = retention.values[ii, jj]
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
    if d.retention.notna().sum() >= 30:
        r["retention 2wk%"] = round(d.retention.mean() * 100)
        r["held all 2wk%"] = round((d.retention == 1).mean() * 100)
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
    ev, conf, fwd, below50, retention, gscore, ma50, ma200 = build(close, high, low, vol, nifty, pit, gscore)

    # baseline: every universe stock, weekly
    wk = pd.DataFrame(False, index=close.index, columns=close.columns)
    wk.iloc[::5] = True
    base = collect(wk, conf, fwd, below50, retention, gscore, pit)

    rows = []
    for b in ("leader", "mid", "laggard"):
        rows.append(dict(breakout="BASELINE any stock, weekly", **stats(base[base.industry == b], f"{b} industry")))
    allev = []
    for name, m in ev.items():
        d = collect(m, conf, fwd, below50, retention, gscore, pit)
        d["breakout"] = name
        allev.append(d)
        for b in ("leader", "mid", "laggard"):
            x = d[d.industry == b]
            rows.append(dict(breakout=name, **stats(x, f"{b} industry")))
            if name != "anchor_up":
                rows.append(dict(breakout=name, **stats(x[x.anchor], f"{b} + volume anchor")))
            rows.append(dict(breakout=name, **stats(x[x.anchor & x.clv80], f"{b} + anchor + CLV>=0.8")))
    T = pd.DataFrame(rows)
    T.to_csv(f"{OUT}/breakout_sector_study.csv", index=False)

    print(f"\n{'=' * 150}\nBREAKOUTS BY SUB-INDUSTRY STRENGTH (pp).  vsTypical = median breakout minus the median universe "
          f"stock over the same window.\nportfolio vsAvg = mean breakout minus mean stock (what an equal-weight basket of these "
          f"would add).  beat% = share beating the median stock.\n'yrs 3mo>0' = calendar years in which the 3-month "
          f"vsTypical was positive.  Compare every row with the BASELINE row for the same industry band.\n"
          f"Retention is partly mechanical: an event day closed UP, so even random prices stay above the pre-event\n"
          f"close for a while (on random data a 50dma reclaim 'retains' ~73% vs ~53% for any day). Judge it against\n"
          f"other rows of the same event type, not in absolute terms.\n{'=' * 150}")
    for name in ["BASELINE any stock, weekly"] + list(ev):
        print(f"\n--- {name}")
        print(T[T.breakout == name].drop(columns="breakout").to_string(index=False))

    E = pd.concat(allev)
    E["era"] = np.where(E.date < ERA_SPLIT, "2016-20", "2021-26")
    print(f"\n{'=' * 150}\nBY ERA -- leader industry + volume anchor vs the same breakout in laggard industries\n{'=' * 150}")
    er = []
    for name in ev:
        for era in ("2016-20", "2021-26"):
            x = E[(E.breakout == name) & (E.era == era)]
            er.append(dict(breakout=name, era=era, **stats(x[(x.industry == "leader") & x.anchor], "leader+anchor")))
            er.append(dict(breakout=name, era=era, **stats(x[x.industry == "laggard"], "laggard, any")))
    print(pd.DataFrame(er).to_string(index=False))

    # ---- Trend Scanner entry tags --------------------------------------------
    tags = scanner_tags(close, nifty, ma50, ma200)
    base["tag"] = tags.values[close.index.get_indexer(base.date), close.columns.get_indexer(base.ticker)]
    # a 52-week or all-time-high breakout in the last 10 sessions (incl. today)
    hb = (ev["high52"] | ev["ath"]).astype(float).rolling(10, min_periods=1).max().astype(bool)
    base["fresh_high"] = hb.values[close.index.get_indexer(base.date), close.columns.get_indexer(base.ticker)]
    base["era"] = np.where(base.date < ERA_SPLIT, "2016-20", "2021-26")
    order = ["Actionable", "Pullback Buy", "Extended", "Late/Fading", "Weak"]
    tr = [dict(view="all stocks", **stats(base, "ALL (baseline)"))]
    for tg in order:
        tr.append(dict(view="all stocks", **stats(base[base.tag == tg], tg)))
    for b in ("leader", "mid", "laggard"):
        for tg in order:
            x = base[(base.tag == tg) & (base.industry == b)]
            tr.append(dict(view=f"{b} industry", **stats(x, tg)))
    for tg in order:
        x = base[(base.tag == tg) & base.fresh_high]
        tr.append(dict(view="52w/ATH breakout in last 2 weeks", **stats(x, tg)))
        x = base[(base.tag == tg) & base.fresh_high & (base.industry == "leader")]
        tr.append(dict(view="52w/ATH breakout, leader industry", **stats(x, tg)))
    for era in ("2016-20", "2021-26"):
        for tg in order:
            tr.append(dict(view=f"era {era}", **stats(base[(base.tag == tg) & (base.era == era)], tg)))
    TG = pd.DataFrame(tr)
    TG.to_csv(f"{OUT}/scanner_tags_study.csv", index=False)
    share = base.tag.value_counts(normalize=True).mul(100).round(1)
    print(f"\n{'=' * 150}\nTREND SCANNER ENTRY TAGS -- every universe stock, weekly. Same columns as above; "
          f"compare each tag with ALL (baseline).\nShare of stock-weeks: "
          + ", ".join(f"{k} {v}%" for k, v in share.items() if k) + f"\n{'=' * 150}")
    for view in TG.view.unique():
        print(f"\n--- {view}")
        print(TG[TG.view == view].drop(columns="view").to_string(index=False))

    # recent qualifying events, for a look at what it picks now
    last = E[(E.date >= close.index[-30]) & (E.industry == "leader") & E.anchor]
    last = last[["date", "ticker", "breakout"]].sort_values("date", ascending=False)
    last["ticker"] = last.ticker.str.replace(".NS", "", regex=False)
    last["sub_industry"] = last.ticker.add(".NS").map(SUB_INDUSTRY_MAP)
    last.to_csv(f"{OUT}/breakout_sector_recent.csv", index=False)
    print(f"\nLAST 30 SESSIONS -- breakouts / anchors in leader industries with a volume anchor: {len(last)}")
    print(last.head(40).to_string(index=False))
    print(f"\nsaved -> {OUT}/breakout_sector_study.csv, {OUT}/breakout_sector_recent.csv, {OUT}/scanner_tags_study.csv")


if __name__ == "__main__":
    main()
