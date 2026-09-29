"""
Transfer test: does what we learned on the stock book hold for groups,
the AI theme and the US?
======================================================================

On the Indian stock book, ranking by 6-month relative strength beat the
1-month-heavy CompRS in 8/8 staggered starts, 3/3 separate periods and every
random-ranking seed (RANKING_BREADTH_VERDICT.md). The other rotation tools in
this repo still rank by short blends:

  india_subind  58 NSE sub-industries  live: 0.7 x mean member CompRS percentile
                                             + 0.3 x % members with CompRS > 0
                                             (CompRS = 10% 1w / 50% 1m / 40% 3m)
  ai            AI capex theme (US)    live: 30% 1w / 50% 1m / 20% 3m vs SMH
  us_sectors    11 SPDR sector ETFs    live: 30/50/20 vs SPY (US engine default)
  us_subind     S&P 500 sub-industries live: mean member 30/50/20 CompRS

Each mode ranks its assets on every rebalance, holds the top N equal weight
until the next one, and compares ranking keys: the live score, 3-, 6- and
12-month RS, and random order. The same robustness tests as the stock book:

  - last 1 / 3 / 5 / 10 years (nested -- not independent)
  - 8 ten-year runs with start dates 16 sessions apart (start-date sensitivity)
  - 3 separate periods, fresh book each (the regime test)
  - 8 random-order seeds (a key must beat them, or its edge is noise)

Data hygiene carried over from the stock-book bugs: rows off the benchmark's
calendar are dropped, a partial final day is cut, and a missing price never
reaches the book (a held asset with no return that day earns 0).

Survivorship: the AI list and the S&P 500 list are today's names, chosen with
hindsight, so absolute returns are flattered. The comparison between keys is
fair -- every key, and the random control, draws from the same list.

Run: python transfer_backtest.py --mode india_subind|ai|us_sectors|us_subind|all
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

COST = 0.002                 # per side, on turnover
N_RANDOM, N_STAGGER, STEP = 8, 8, 16
PERIOD_EDGES = ["2019-12-31", "2022-12-31"]
OUT = "analysis"

AI_TICKERS = [
    'NVDA', 'AMD', 'AVGO', 'MRVL', 'TSM', 'ASML', 'AMAT', 'LRCX', 'KLAC', 'MU',
    'GEV', 'ETN', 'VRT', 'PWR', 'MTZ', 'CEG', 'CCJ', 'BE', 'NVT', 'EQIX', 'DLR',
    'ANET', 'COHR', 'LITE', 'MTSI', 'FN', 'APH', 'SMTC',
    'SNPS', 'CDNS', 'ARM', 'INTC', 'GFS', 'ON', 'MPWR',
    'ECL', 'XYL', 'DOV', 'FIX', 'EME', 'OKLO', 'SMR',
    'MSFT', 'GOOGL', 'AMZN', 'META', 'ORCL',
    'AXTI', 'PLAB', 'ACLS', 'ONTO', 'WOLF', 'FSLR',
]
SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]

MODES = {
    #               bench     N  rebalance  live weights (1w, 1m, 3m)
    "india_subind": ("^NSEI", 5, 13, [(5, .10), (21, .50), (63, .40)]),
    "ai":           ("SMH",  10, 10, [(5, .30), (21, .50), (63, .20)]),
    "us_sectors":   ("SPY",   3, 21, [(5, .30), (21, .50), (63, .20)]),
    "us_subind":    ("SPY",   5, 13, [(5, .30), (21, .50), (63, .20)]),
}


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def cut_partial_last_day(close, vol=None):
    """Drop trailing sessions where far fewer assets than usual have a close."""
    cnt = close.notna().sum(axis=1)
    full = cnt >= 0.9 * cnt.rolling(20, min_periods=5).median().shift(1)
    full.iloc[:5] = True
    last_ok = full[full].index[-1]
    if last_ok != close.index[-1]:
        print(f"[data] {close.index[-1].date()} has closes for {cnt.iloc[-1]} of ~{int(cnt.iloc[-21:-1].median())} "
              f"-- using {last_ok.date()} as the last day")
        close = close.loc[:last_ok]
        vol = vol.loc[:last_ok] if vol is not None else None
    return close, vol


def fetch_prices(tickers, bench, start):
    from exit_rule_backtest import align_to_sessions
    from utils.yf_safe import safe_download, safe_history

    end = datetime.now().strftime("%Y-%m-%d")
    b = safe_history(bench, start=start, end=end)
    if b.empty:
        raise SystemExit(f"Could not fetch {bench}")
    if b.index.tz is not None:
        b.index = b.index.tz_localize(None)
    bulk = safe_download(tickers, start=start, end=end, group_by="ticker", threads=False,
                         auto_adjust=True, min_coverage=0.5)
    closes, vols = {}, {}
    multi = isinstance(bulk.columns, pd.MultiIndex)
    for t in tickers:
        try:
            sub = bulk[t] if multi else bulk
            c = sub["Close"].dropna()
            if len(c) > 60:
                closes[t], vols[t] = c, sub["Volume"].reindex(c.index)
        except Exception:
            continue
    close, vol = pd.DataFrame(closes), pd.DataFrame(vols)
    if close.index.tz is not None:
        close.index, vol.index = close.index.tz_localize(None), vol.index.tz_localize(None)
    close, vol = align_to_sessions(close, vol, b)
    close, vol = cut_partial_last_day(close, vol)
    bc = b["Close"].reindex(close.index).ffill()
    print(f"[data] {close.shape[1]}/{len(tickers)} assets, {len(close)} sessions "
          f"{close.index[0].date()} -> {close.index[-1].date()}, benchmark {bench}")
    return close, vol, bc


# ----------------------------------------------------------------------------
# scores
# ----------------------------------------------------------------------------
def rs(close, bc, k):
    """Excess return over the benchmark across k sessions, in points."""
    cf = close.ffill(limit=5)
    return ((close / cf.shift(k) - 1).sub(bc / bc.shift(k) - 1, axis=0) * 100).where(close.notna())


def blend(close, bc, weights):
    return sum(w * rs(close, bc, k) for k, w in weights)


def asset_keys(close, bc, live_w):
    return {"live": blend(close, bc, live_w),
            "optcomp_10_50_40": blend(close, bc, [(5, .10), (21, .50), (63, .40)]),
            "rs63": rs(close, bc, 63), "rs126": rs(close, bc, 126), "rs252": rs(close, bc, 252)}


def group_panels(close, member_mask, groups, bc, live_w, india_live):
    """
    Groups as assets. Daily group return = mean of member returns (members in
    the universe that day with a valid return), so a gap in one stock never
    blanks the group. Scores:
      live      -- what the live tool computes (India: 0.7 pctile + 0.3 breadth
                   on member CompRS; US: mean member CompRS)
      live_rs126 -- the same aggregation on member 6-month RS
      rs63/126/252 -- RS of the group's own index vs the benchmark
    """
    ret = close.pct_change(fill_method=None).where(member_mask & member_mask.shift(1, fill_value=False))
    cols = [c for c in close.columns if c in groups]
    g = pd.Series({c: groups[c] for c in cols})
    agg = lambda panel, how="mean": getattr(panel[cols].T.groupby(g), how)().T
    gret = agg(ret)
    n = agg(member_mask.astype(float), "sum")
    gret = gret.where(n >= 3)
    gidx = (1 + gret.fillna(0)).cumprod().where(n >= 3)

    def member_score(stock_rs):
        srs = stock_rs.where(member_mask)
        if india_live:
            pct = srs.rank(axis=1, pct=True) * 100
            return 0.7 * agg(pct) + 0.3 * agg((srs > 0).astype(float).where(srs.notna())) * 100
        return agg(srs)

    keys = {"live": member_score(blend(close, bc, live_w)),
            "live_rs126": member_score(rs(close, bc, 126))}
    for k in (63, 126, 252):
        keys[f"rs{k}"] = rs(gidx, bc, k)
    keys = {k: v.where(n >= 3) for k, v in keys.items()}
    return gret, keys, n


# ----------------------------------------------------------------------------
# simulator: top-N equal weight, hold to next rebalance
# ----------------------------------------------------------------------------
def simulate(ret, score, dates, top_n, reb, cost=None):
    """Decide at the close of a rebalance day, earn from the next session on.
    Returns (equity curve, one-way turnover per year)."""
    cost = COST if cost is None else cost
    R = ret.reindex(dates).fillna(0.0).values          # a held asset with no price that day earns 0
    S = score.reindex(dates).values
    w = np.zeros(R.shape[1])
    eq, curve, traded = 1.0, [1.0], 0.0
    for i in range(len(dates) - 1):
        if i % reb == 0:
            s = S[i]
            ok = np.where(np.isfinite(s))[0]
            if len(ok):
                pick = ok[np.argsort(-s[ok])[:top_n]]
                new = np.zeros_like(w)
                new[pick] = 1.0 / len(pick)
                traded += np.abs(new - w).sum() / 2
                eq *= 1 - cost * np.abs(new - w).sum()
                w = new
        r = R[i + 1]
        day = float((w * r).sum())
        eq *= 1 + day
        if w.sum() > 0:
            w = w * (1 + r)
            w = w / w.sum()
        curve.append(eq)
    yrs = max((dates[-1] - dates[0]).days / 365.25, 1e-9)
    return pd.Series(curve, index=dates), traded / yrs


def stats(curve):
    yrs = max((curve.index[-1] - curve.index[0]).days / 365.25, 1e-9)
    dr = curve.pct_change().dropna()
    return dict(cagr=round(((curve.iloc[-1] / curve.iloc[0]) ** (1 / yrs) - 1) * 100, 2),
                sharpe=round(dr.mean() / dr.std() * np.sqrt(252), 2) if dr.std() else 0.0,
                maxdd=round(((curve / curve.cummax()) - 1).min() * 100, 1))


# ----------------------------------------------------------------------------
def run_mode(mode, args):
    bench, top_n, reb, live_w = MODES[mode]
    start = "2014-06-01"
    if mode == "india_subind":
        from momentum_factor_backtest import build_pit_universe, load_candidates
        from utils.nifty1000_list import SUB_INDUSTRY_MAP
        tickers = load_candidates("all", args.max_tickers)
        close, vol, bc = fetch_prices(tickers, bench, start)
        mask = build_pit_universe(close, vol, 1000)
        ret, keys, n = group_panels(close, mask, SUB_INDUSTRY_MAP, bc, live_w, india_live=True)
        label = "sub-industries"
    elif mode == "us_subind":
        u = pd.read_csv("data/sp500_tickers.csv")
        u["yf"] = u.ticker.str.replace(".", "-", regex=False)
        close, vol, bc = fetch_prices(u.yf.tolist()[: args.max_tickers or None], bench, start)
        mask = close.notna()
        ret, keys, n = group_panels(close, mask, dict(zip(u.yf, u.sub_industry)), bc, live_w, india_live=False)
        label = "sub-industries"
    else:
        tick = AI_TICKERS if mode == "ai" else SECTOR_ETFS
        close, vol, bc = fetch_prices(tick, bench, start)
        ret = close.pct_change(fill_method=None)
        keys = asset_keys(close, bc, live_w)
        label = "assets"
    # One eligibility rule for every key: 12-month RS must exist, so the keys
    # rank the same set and differ only in order.
    elig = keys["rs252"].notna() if "rs252" in keys else None
    keys = {k: v.where(elig) for k, v in keys.items()}
    rng = np.random.default_rng(20260929)
    for s in range(N_RANDOM):
        keys[f"rand{s}"] = pd.DataFrame(rng.random(elig.shape), index=elig.index,
                                        columns=elig.columns).where(elig)
    dates = close.index[elig.sum(axis=1) >= top_n + 2]
    print(f"[{mode}] {elig.shape[1]} {label}, top {top_n}, rebalance every {reb} sessions, "
          f"tradable {dates[0].date()} -> {dates[-1].date()} ({len(dates)} sessions)")

    runs = []
    def add(test, period, sub):
        for k, sc in keys.items():
            net, turn = simulate(ret, sc, sub, top_n, reb)
            gross, _ = simulate(ret, sc, sub, top_n, reb, cost=0.0)
            runs.append(dict(mode=mode, test=test, period=period, key=k, **stats(net),
                             gross_sharpe=stats(gross)["sharpe"], turnover_pa=round(turn * 100)))
        ew = (ret.reindex(sub).where(elig.reindex(sub)).mean(axis=1).fillna(0) + 1).cumprod()
        runs.append(dict(mode=mode, test=test, period=period, key="equal_weight_all", **stats(ew)))
        runs.append(dict(mode=mode, test=test, period=period, key=f"benchmark {bench}",
                         **stats(bc.reindex(sub) / bc.reindex(sub).iloc[0])))

    end = dates[-1]
    for y in (1, 3, 5, 10):
        sub = dates[dates >= end - pd.Timedelta(days=365 * y)]
        if len(sub) > 200:
            add("window", f"{y}y", sub)
    L = int(np.searchsorted(dates, end - pd.Timedelta(days=3650)))
    for s in range(N_STAGGER):
        i0 = L - s * STEP
        if i0 < 0:
            break
        sub = dates[i0: i0 + len(dates) - L]
        add("stagger", f"start {sub[0].date()}", sub)
    edges = [dates[L]] + [pd.Timestamp(e) for e in PERIOD_EDGES] + [end]
    for a, b in zip(edges, edges[1:]):
        sub = dates[(dates >= a) & (dates <= b)] if a == edges[0] else dates[(dates > a) & (dates <= b)]
        if len(sub) > 200:
            add("period", f"{sub[0].date()} -> {sub[-1].date()}", sub)
    return pd.DataFrame(runs)


def summarise(T):
    for mode, t in T.groupby("mode", sort=False):
        print(f"\n{'=' * 100}\n{mode.upper()}  (Sharpe; d = key minus LIVE on the same start/period)\n{'=' * 100}")
        real = [k for k in t.key.unique() if not k.startswith("rand")]
        w = t[t.test == "window"].pivot_table(index="key", columns="period", values="sharpe")
        w = w.reindex(real)[[c for c in ("1y", "3y", "5y", "10y") if c in w]]
        print("Last 1/3/5/10 years:")
        print(w.to_string())
        rows = []
        for test in ("stagger", "period"):
            x = t[t.test == test]
            live = x[x.key == "live"].set_index("period").sharpe
            rnd = x[x.key.str.startswith("rand")].groupby("period").sharpe
            best_rand, med_rand = rnd.max(), rnd.median()
            live_g = x[x.key == "live"].set_index("period").gross_sharpe
            for k in real:
                xs = x[x.key == k].set_index("period")
                s, d = xs.sharpe, (xs.sharpe - live).dropna()
                dg = (xs.gross_sharpe - live_g).dropna()
                rows.append(dict(test=test, key=k,
                                 beats_live="-" if k == "live" else f"{(d > 0).sum()}/{len(d)}",
                                 median_d_sharpe=round(d.median(), 2),
                                 gross_beats_live="-" if k == "live" else f"{(dg > 0).sum()}/{len(dg)}",
                                 gross_d_sharpe=round(dg.median(), 2) if len(dg) else "",
                                 beats_best_random=f"{(s > best_rand.reindex(s.index)).sum()}/{len(s)}",
                                 median_sharpe=round(s.median(), 2),
                                 turnover_pa=f"{xs.turnover_pa.median():.0f}%" if xs.turnover_pa.notna().any() else ""))
            rows.append(dict(test=test, key="random (median seed)", median_sharpe=round(med_rand.median(), 2)))
        print("\n8 staggered 10-year starts / 3 separate periods. gross_ = before costs: a key that wins")
        print("net but not gross wins by trading less, not by ranking better.")
        print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=list(MODES) + ["all"])
    ap.add_argument("--max-tickers", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    modes = list(MODES) if args.mode == "all" else [args.mode]
    T = []
    for m in modes:
        try:
            T.append(run_mode(m, args))
        except SystemExit as e:
            print(f"[{m}] skipped: {e}")
    if not T:
        raise SystemExit("no mode produced results")
    T = pd.concat(T)
    T.to_csv(f"{OUT}/transfer_backtest.csv", index=False)
    summarise(T)
    print(f"\nsaved -> {OUT}/transfer_backtest.csv")


if __name__ == "__main__":
    main()
