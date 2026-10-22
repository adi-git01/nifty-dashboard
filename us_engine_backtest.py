"""
Does the India momentum engine work on US stocks?
=================================================

The live engine's exact rules (ranking_rotation_backtest.simulate: 6-month RS ranking, CompRS entry floor,
price > MA50, liquidity floor, 30% breadth gate, regime trail stop and rebalance, MA50 exit, 15 equal-weight
slots) on a point-in-time US universe, with SPY in Nifty's place (relative strength, regime, benchmark).

Universe: every US common stock listed on NASDAQ / NYSE / NYSE American today (nasdaqtrader symbol
directories; ETFs, test issues, warrants, units, rights and preferreds removed), then each day the top 1000
by 60-day median dollar volume -- the same point-in-time rule as India. Delisted companies are missing
(the same one-directional bias as India; it flatters every variant equally and the benchmarks less).
The liquidity floor reads the engine's crore threshold as $10-20M a day (px x volume / 1e7), which the
top-1000 clears anyway.

Variants
  US BASE         live rules, US costs (0.05% each way)
  US BASE, India costs   0.2% each way (the India harness) -- sensitivity
  US no RS floor  min CompRS 0 (US large caps disperse less than Indian stocks; the 17-22 pp floor may
                  starve the book)
  US 12m RS       rank by 12-month RS (the most reliable US ordering in transfer_backtest.py)
  US top 500      the same rules on the point-in-time top 500
Breadth gate: share of priced names above their MA50 among stocks ever in the point-in-time top 1000
(India's harness uses its ~2,000 downloaded candidates).
Benchmarks: SPY buy-and-hold, equal-weight point-in-time top 1000 (daily), 8 random-order seeds on US BASE's
eligible set.

Pre-registered: a variant is worth building only if its Sharpe beats SPY at >= 6 of 8 staggered 10-year
starts AND in >= 2 of 3 separate periods AND beats the best random seed at >= 6 of 8 starts.

Run: python us_engine_backtest.py           (Actions: ranking_rotation_backtest.yml, script=us_engine)
     python us_engine_backtest.py --synthetic
"""
from __future__ import annotations

import argparse
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

OUT_CSV = "analysis/us_engine_backtest.csv"
START = "2014-06-01"
N_RANDOM, N_STAGGER, STEP = 8, 8, 16
US_COST = 0.0005
SYMDIR = ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
          "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt")
BAD_NAME = r"(?i)\b(?:warrant|warrants|unit|units|right|rights|preferred|pfd|depositary share|notes?|debentures?|trust preferred|%)\b"


def us_symbols(max_tickers=0):
    import requests
    out = []
    try:
        texts = [requests.get(url, timeout=60).text for url in SYMDIR]
    except Exception as e:
        print(f"[universe] !! symbol directory unavailable ({e}); FALLING BACK to today's S&P 500 list -- "
              f"a survivor-only universe that flatters every momentum result. Treat this run as indicative only.")
        t = pd.read_csv("data/sp500_tickers.csv")
        col = "Ticker" if "Ticker" in t else t.columns[0]
        out = sorted(set(t[col].astype(str).str.replace(".", "-", regex=False)))
        return out[:max_tickers] if max_tickers else out
    for txt in texts:
        d = pd.read_csv(io.StringIO(txt), sep="|")
        d = d[~d.iloc[:, 0].astype(str).str.startswith("File Creation")]
        sym = "Symbol" if "Symbol" in d else "ACT Symbol"
        d = d[(d.get("ETF", "N") == "N") & (d.get("Test Issue", "N") == "N")]
        d = d[~d["Security Name"].astype(str).str.contains(BAD_NAME, regex=True)]
        s = d[sym].astype(str)
        s = s[~s.str.contains(r"[\$\^~ ]") & (s.str.len() <= 6)]
        out += list(s.str.replace(".", "-", regex=False))
    out = sorted(set(out))
    print(f"[universe] {len(out)} US common-stock symbols from the NASDAQ / NYSE directories")
    return out[:max_tickers] if max_tickers else out


def fetch_us(tickers, chunk=400):
    """Adjusted close and volume (float32), aligned to SPY's sessions."""
    from utils.yf_safe import safe_download, safe_history
    spy = safe_history("SPY", start=START)
    if spy.index.tz is not None:
        spy.index = spy.index.tz_localize(None)
    sessions = spy.index
    C, V = {}, {}
    for k in range(0, len(tickers), chunk):
        part = tickers[k:k + chunk]
        bulk = safe_download(part, start=START, group_by="ticker", threads=True, auto_adjust=True, min_coverage=0.3)
        for t in part:
            try:
                sub = bulk[t] if len(part) > 1 else bulk
                c = sub["Close"].dropna()
            except Exception:
                continue
            if len(c) < 250:
                continue
            if c.index.tz is not None:
                c.index = c.index.tz_localize(None)
            C[t] = c.reindex(sessions).astype("float32")
            v = sub["Volume"]
            if v.index.tz is not None:
                v.index = v.index.tz_localize(None)
            V[t] = v.reindex(sessions).astype("float32")
        print(f"[data] {min(k + chunk, len(tickers))}/{len(tickers)} requested, {len(C)} usable", flush=True)
    close, vol = pd.DataFrame(C), pd.DataFrame(V)
    # drop a trailing partial session (few names priced)
    cnt = close.notna().sum(axis=1)
    keep = cnt >= 0.8 * cnt.rolling(20, min_periods=5).median().shift(1).fillna(cnt)
    close, vol, spy = close[keep], vol[keep], spy.reindex(close.index[keep])
    print(f"[data] {close.shape[1]} tickers, {len(close)} sessions {close.index[0].date()} -> {close.index[-1].date()}")
    return close, vol, spy


def synthetic(n=300, days=3000, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-06-02", periods=days)
    r = rng.normal(0.0004, 0.02, (days, n)) + rng.normal(0.0003, 0.01, (days, 1))
    close = pd.DataFrame(50 * np.exp(np.cumsum(r, 0)), idx, [f"U{k:03d}" for k in range(n)]).astype("float32")
    vol = pd.DataFrame(np.exp(rng.normal(14, 1, (days, n))), idx, close.columns).astype("float32")
    s = close.mean(axis=1)
    spy = pd.DataFrame({"Close": s, "High": s * 1.005, "Low": s * 0.995})
    return close, vol, spy


def curve_stats(eq):
    eq = eq.dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    r = eq.pct_change().dropna()
    return dict(cagr_pct=round(((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100, 2),
                sharpe=round(r.mean() / r.std() * np.sqrt(252), 2) if r.std() > 0 else np.nan,
                max_drawdown_pct=round((eq / eq.cummax() - 1).min() * 100, 1))


def main():
    import exit_rule_backtest as erb
    import ranking_rotation_backtest as rrb
    from momentum_factor_backtest import build_pit_universe
    from utils import regime_manager as rm

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    pd.set_option("display.width", 240)

    if args.synthetic:
        close, vol, spy = synthetic()
    else:
        close, vol, spy = fetch_us(us_symbols(args.max_tickers))
    close, vol = close.astype("float64"), vol.astype("float64")
    dates = close.index
    pit1000 = build_pit_universe(close, vol, 1000)
    # keep only names that were ever in the point-in-time top 1000: eligibility on any day is unchanged,
    # and the simulator (one pass per day over every column) runs ~2x faster on ~5,500 US listings
    ever = pit1000.any()
    close, vol, pit1000 = close.loc[:, ever], vol.loc[:, ever], pit1000.loc[:, ever]
    print(f"[universe] {int(ever.sum())} names were ever in the point-in-time top 1000", flush=True)
    pit500 = build_pit_universe(close, vol, 500)
    base = erb.build_rs_panel(close, spy, dates).where(lambda x: x.abs() <= rrb.RS_CAP)
    keys = rrb.build_keys(close, spy, dates, base.where(pit1000))
    hist = close.notna().cumsum()
    rng = np.random.default_rng(20261020)
    for s in range(N_RANDOM):
        keys[f"rand{s}"] = pd.DataFrame(rng.random(base.shape), index=dates, columns=base.columns).where(pit1000 & base.notna())
    print(f"[universe] point-in-time top 1000 by dollar volume; median names with a CompRS "
          f"{int(base.where(pit1000).notna().sum(axis=1).median())}", flush=True)

    no_floor = {k: {**v, "min_comp_rs": 0.0} for k, v in rm.REGIME_PARAMS.items()}
    V = {
        "US BASE (6m RS)":          dict(cfg={"key": "rs126"}, elig=base.where(pit1000), cost=US_COST, params=None),
        "US BASE, India costs":     dict(cfg={"key": "rs126"}, elig=base.where(pit1000), cost=0.002, params=None),
        "US no RS floor":           dict(cfg={"key": "rs126"}, elig=base.where(pit1000), cost=US_COST, params=no_floor),
        "US 12m RS":                dict(cfg={"key": "rs252"}, elig=base.where(pit1000), cost=US_COST, params=None),
        "US top 500":               dict(cfg={"key": "rs126"}, elig=base.where(pit500), cost=US_COST, params=None),
    }
    RND = {f"RANDOM {s}": dict(cfg={"key": f"rand{s}"}, elig=base.where(pit1000), cost=US_COST, params=None)
           for s in range(N_RANDOM)}

    spy_eq = spy["Close"].reindex(dates).ffill()
    ew_ret = close.pct_change(fill_method=None).where(pit1000.shift(1, fill_value=False)).mean(axis=1).fillna(0).clip(-0.2, 0.2)
    ew_eq = (1 + ew_ret).cumprod()

    rows, trades_10y = [], []
    orig_params = dict(rm.REGIME_PARAMS)

    def run(name, spec, sub, test, period):
        erb_buy, erb_sell = rrb.BUY_COST, rrb.SELL_COST
        rrb.BUY_COST, rrb.SELL_COST = 1 + spec["cost"], 1 - spec["cost"]
        if spec["params"] is not None:
            rm.REGIME_PARAMS.clear()
            rm.REGIME_PARAMS.update(spec["params"])
        try:
            cfg = {**rrb.DEFAULTS, **spec["cfg"]}
            curve, trades = rrb.simulate(name, cfg, close, vol, spy, sub, spec["elig"], keys[cfg["key"]], hist)
        finally:
            rrb.BUY_COST, rrb.SELL_COST = erb_buy, erb_sell
            rm.REGIME_PARAMS.clear()
            rm.REGIME_PARAMS.update(orig_params)
        r = curve_stats(curve.set_index("date")["equity"])
        r.update(variant=name, test=test, period=period, trades=len(trades),
                 avg_holdings=round(curve.holdings.mean(), 1))
        rows.append(r)
        if test == "window" and period == "10y" and len(trades):
            trades_10y.append(trades.assign(variant=name))

    def bench(sub, test, period):
        for name, eq in (("SPY buy & hold", spy_eq), ("equal-weight top 1000", ew_eq)):
            r = curve_stats(eq.reindex(sub))
            r.update(variant=name, test=test, period=period)
            rows.append(r)

    end = dates[-1]
    for y in (1, 3, 5, 10):
        sub = dates[dates >= end - pd.Timedelta(days=365 * y)]
        for n, sp in V.items():
            run(n, sp, sub, "window", f"{y}y")
        bench(sub, "window", f"{y}y")
        print(f"   window {y}y done", flush=True)
    L = int(np.searchsorted(dates, end - pd.Timedelta(days=3650)))
    for k in range(N_STAGGER):
        i0 = max(L - k * STEP, 260)
        sub = dates[i0: i0 + len(dates) - L]
        for n, sp in {**V, **RND}.items():
            run(n, sp, sub, "stagger", f"start {sub[0].date()}")
        bench(sub, "stagger", f"start {sub[0].date()}")
        print(f"   stagger {sub[0].date()} done", flush=True)
    edges = [dates[L], pd.Timestamp("2019-12-31"), pd.Timestamp("2022-12-31"), end]
    for a, b in zip(edges, edges[1:]):
        sub = dates[(dates >= a) & (dates <= b)] if a == edges[0] else dates[(dates > a) & (dates <= b)]
        for n, sp in {**V, **RND}.items():
            run(n, sp, sub, "period", f"{sub[0].date()} -> {sub[-1].date()}")
        bench(sub, "period", f"{sub[0].date()} -> {sub[-1].date()}")
        print(f"   period {sub[0].date()} -> {sub[-1].date()} done", flush=True)

    T = pd.DataFrame(rows)
    os.makedirs("analysis", exist_ok=True)
    T.to_csv(OUT_CSV, index=False)
    order = list(V) + ["SPY buy & hold", "equal-weight top 1000"]

    print(f"\n{'=' * 120}\nLAST 1/3/5/10 YEARS (nested -- not independent)\n{'=' * 120}")
    w = T[T.test == "window"]
    for m in ("cagr_pct", "sharpe", "max_drawdown_pct"):
        print(f"\n{m}:\n" + w.pivot_table(index="variant", columns="period", values=m).reindex(order)[["1y", "3y", "5y", "10y"]].to_string())
    if trades_10y:
        tr = pd.concat(trades_10y)
        print("\n10-year trades:\n" + tr.groupby("variant").agg(trades=("ticker", "size"), win_pct=("pnl_pct", lambda x: round(100 * (x > 0).mean())),
                                                             avg_pnl_pct=("pnl_pct", "mean"),
                                                             top_exit=("reason", lambda x: x.value_counts().index[0])).round(1).to_string())

    verdict = {}
    for test, title in (("stagger", f"{N_STAGGER} STAGGERED 10-YEAR STARTS"), ("period", "3 SEPARATE PERIODS")):
        t = T[T.test == test]
        spy_s = t[t.variant == "SPY buy & hold"].set_index("period").sharpe
        best_r = t[t.variant.str.startswith("RANDOM")].groupby("period").sharpe.max()
        out = []
        for v in V:
            x = t[t.variant == v].set_index("period")
            beat_spy = int((x.sharpe > spy_s.reindex(x.index)).sum())
            beat_rnd = int((x.sharpe > best_r.reindex(x.index)).sum())
            out.append(dict(variant=v, beats_SPY=f"{beat_spy}/{len(x)}", beats_best_random=f"{beat_rnd}/{len(x)}",
                            median_sharpe=x.sharpe.median(), median_cagr=x.cagr_pct.median(),
                            median_maxdd=x.max_drawdown_pct.median()))
            verdict.setdefault(v, {})[test] = (beat_spy, beat_rnd, len(x))
        for bn in ("SPY buy & hold", "equal-weight top 1000"):
            x = t[t.variant == bn]
            out.append(dict(variant=bn, beats_SPY="-", beats_best_random="-", median_sharpe=x.sharpe.median(),
                            median_cagr=x.cagr_pct.median(), median_maxdd=x.max_drawdown_pct.median()))
        print(f"\n{'=' * 120}\n{title}\n{'=' * 120}\n" + pd.DataFrame(out).round(2).to_string(index=False))
        print("\nCAGR % by start/period:\n" + t[~t.variant.str.startswith("RANDOM")].pivot_table(
            index="period", columns="variant", values="cagr_pct").round(1)[order].to_string())

    print(f"\n{'=' * 120}\nPRE-REGISTERED: Sharpe beats SPY at >= 6/8 starts AND >= 2/3 periods AND beats the best random seed "
          f"at >= 6/8 starts\n{'=' * 120}")
    for v, d in verdict.items():
        s, p = d.get("stagger", (0, 0, 0)), d.get("period", (0, 0, 0))
        ok = s[0] >= 6 and p[0] >= 2 and s[1] >= 6
        print(f"  {'PASS' if ok else 'FAIL'}  {v}: beats SPY {s[0]}/{s[2]} starts, {p[0]}/{p[2]} periods; "
              f"beats best random {s[1]}/{s[2]} starts")
    print(f"\nsaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
