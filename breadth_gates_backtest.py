"""
Market-breadth gates and stock-level MA50 rules — 1/3/5/10y + robustness
=========================================================================

Question: does gating NEW entries on market breadth help the OptComp book, and
is "% of stocks with trend score >= 60" (the dashboard's Market Breadth chart)
a better gate than "% of stocks above their MA50" (the live engine's rule)?

Each gate REPLACES the live 30%-above-MA50 rule and only blocks new buys on
rebalance days; exits are untouched. Breadth is measured over the same
point-in-time top-1000 universe the strategy trades.

Trend score is rebuilt from closes with the live formula (utils/scoring.py
calculate_trend_metrics): +15/-10 price vs MA50, +15/-15 price vs MA200,
+10/-5 MA50 vs MA200, (position in 52w range - 0.5) x 30, +10 within 5% of the
52w high / -10 more than 30% below it, clamped 0-100. The one approximation:
the 52-week high/low use closes, not intraday prices. The rebuilt series is
checked against data/market_breadth_history.csv (pct_uptrends = % TS >= 60)
over the dates they share.

Note on the dashboard chart: its red "% TS<30" line is drawn as
100 - pct_uptrends, i.e. % TS < 60, so the two lines always sum to 100 and
their "crossover" is simply % TS >= 60 crossing 50%. The genuine strong-vs-weak
spread (% TS >= 60 vs % TS < 30) is tested separately here.

Stock-level MA50 rules are tested too: no MA50 exit, no MA50 entry filter.

Robustness is built in (see ranking_robustness_backtest.py for why): results
over 1/3/5/10y windows, over 8 staggered 10-year starts, and over three
non-overlapping periods. Baseline = live rules with the 30% MA50 gate.

Run: python breadth_gates_backtest.py --max-tickers 0
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from exit_rule_backtest import build_rs_panel, fetch, ma50_per_stock, stats
from ranking_rotation_backtest import DEFAULTS, RS_CAP, simulate

OUT_CSV = "analysis/breadth_gates_backtest.csv"
N_STAGGER, STAGGER_STEP = 8, 16
BASE = "BASE MA50>=30 (live rule)"


def trend_score(c):
    """Vectorised utils/scoring.calculate_trend_metrics, from closes."""
    ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
    hi, lo = c.rolling(252, min_periods=50).max(), c.rolling(252, min_periods=50).min()
    s = pd.DataFrame(50.0, index=c.index, columns=c.columns)
    s += np.where(ma50.notna(), np.where(c > ma50, 15, -10), 0)
    s += np.where(ma200.notna(), np.where(c > ma200, 15, -15), 0)
    s += np.where(ma50.notna() & ma200.notna(), np.where(ma50 > ma200, 10, -5), 0)
    rng = hi - lo
    pos = ((c - lo) / rng.where(rng > 0) - 0.5) * 30
    s += np.trunc(pos.fillna(0))                            # int() truncates toward zero
    dist = (c - hi) / hi * 100
    s += np.where(rng > 0, np.where(dist > -5, 10, np.where(dist < -30, -10, 0)), 0)
    return s.clip(0, 100).where(c.notna() & hi.notna())


def hysteresis(b, enter, exit_):
    """Risk-on once breadth >= enter; risk-off once it falls below exit_; else hold state."""
    out, on = [], bool(b.iloc[0] >= enter) if len(b) else False
    for v in b.values:
        if np.isnan(v):
            pass
        elif on and v < exit_:
            on = False
        elif not on and v >= enter:
            on = True
        out.append(on)
    return pd.Series(out, index=b.index)


def nifty_switch(nifty, sub, sw, cost=0.002):
    """Nifty buy-and-hold vs the same switch applied to Nifty (cash when off)."""
    n = nifty["Close"].reindex(sub).ffill()
    r = n.pct_change().fillna(0)
    on = sw.reindex(sub).fillna(False).astype(bool)
    pos = on.shift(1, fill_value=bool(on.iloc[0]))
    flips = on.astype(int).diff().abs().fillna(0)
    rs = r * pos - flips * cost
    def st(x):
        eq = (1 + x).cumprod()
        sh = x.mean() / x.std() * np.sqrt(252) if x.std() else 0.0
        return round((eq.iloc[-1] - 1) * 100, 1), round(((eq / eq.cummax()) - 1).min() * 100, 1), round(sh, 2)
    return st(r), st(rs), int(flips.sum())


def pct(mask, universe):
    return 100.0 * (mask & universe).sum(axis=1) / universe.sum(axis=1).replace(0, np.nan)


def main():
    from momentum_factor_backtest import build_pit_universe, load_candidates

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int, default=0)
    ap.add_argument("--top-n", type=int, default=1000)
    args = ap.parse_args()

    end = datetime.now()
    start = end - timedelta(days=365 * 10 + 450 + 200)
    tickers = load_candidates("all", args.max_tickers)
    close_df, vol_df, nifty = fetch(tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    dates = close_df.index[close_df.index.isin(nifty.index)]
    c = close_df.reindex(dates)
    pit = build_pit_universe(close_df, vol_df, args.top_n).reindex(dates).fillna(False)
    base = build_rs_panel(close_df, nifty, dates).where(lambda x: x.abs() <= RS_CAP).where(pit)
    hist = c.notna().cumsum()

    ma50 = ma50_per_stock(c)
    ts = trend_score(c)
    uni_ma = pit & ma50.notna()
    uni_ts = pit & ts.notna()
    b_ma50 = pct(c > ma50, uni_ma)
    b_ts60 = pct(ts >= 60, uni_ts)
    b_ts30 = pct(ts < 30, uni_ts)
    b_ts50 = pct(ts >= 50, uni_ts)
    print(f"[data] {len(dates)} days {dates[0].date()} -> {dates[-1].date()}, {c.shape[1]} tickers")

    # validation against the dashboard's stored breadth history
    try:
        h = pd.read_csv("data/market_breadth_history.csv", parse_dates=["date"]).set_index("date")
        j = pd.DataFrame({"rebuilt_ts60": b_ts60, "rebuilt_ts50": b_ts50}).join(h, how="inner")
        print(f"[validate] {len(j)} shared dates. rebuilt %TS>=60 vs stored pct_uptrends: "
              f"corr {j.rebuilt_ts60.corr(j.pct_uptrends):.2f}, mean gap "
              f"{(j.rebuilt_ts60 - j.pct_uptrends).mean():+.1f} pts | rebuilt %TS>=50 vs stored "
              f"pct_above_50dma (itself a TS>=50 proxy): corr {j.rebuilt_ts50.corr(j.pct_above_50dma):.2f}")
    except Exception as e:
        print(f"[validate] skipped: {e}")

    print(f"[breadth] median over 10y: %>MA50 {b_ma50.median():.0f}, %TS>=60 {b_ts60.median():.0f}, "
          f"%TS<30 {b_ts30.median():.0f}; corr(%>MA50, %TS>=60) {b_ma50.corr(b_ts60):.2f}")

    G = {
        "NO GATE (how live ran)": pd.Series(True, index=dates),
        BASE:                     b_ma50 >= 30,
        "MA50>=40":               b_ma50 >= 40,
        "MA50>=50":               b_ma50 >= 50,
        "TS60>=30":               b_ts60 >= 30,
        "TS60>=40":               b_ts60 >= 40,
        "TS60>=50 (chart crossover)": b_ts60 >= 50,
        "TS60 > TS30 (strong beats weak)": b_ts60 > b_ts30,
        "TS60 above its 20d avg (improving)": b_ts60 >= b_ts60.rolling(20).mean(),
    }
    VARIANTS = {k: dict(gate=v) for k, v in G.items()}
    VARIANTS["MA50>=30 + no MA50 exit"] = dict(gate=G[BASE], ma50_exit=False)
    VARIANTS["MA50>=30 + no MA50 entry filter"] = dict(gate=G[BASE], ma50_entry=False)
    # Regime switch: hold only while breadth is healthy; SELL EVERYTHING when it
    # drops below the lower level, stay in cash until it recovers above the upper.
    SW = {f"SWITCH TS60 on>={a} / sell-all<{b}": hysteresis(b_ts60, a, b)
          for a, b in [(40, 30), (45, 35), (50, 30), (50, 40)]}
    SW.update({f"SWITCH MA50 on>={a} / sell-all<{b}": hysteresis(b_ma50, a, b)
               for a, b in [(40, 30), (50, 30)]})
    for k, sw in SW.items():
        VARIANTS[k] = dict(gate=G[BASE], switch=sw)
        flips = int(sw.astype(int).diff().abs().sum())
        print(f"[switch] {k}: risk-on {sw.mean()*100:.0f}% of days, {flips} switches in {len(sw)} days")

    def run(name, over, sub):
        cfg = {**DEFAULTS, **over}
        curve, trades = simulate(name, cfg, close_df, vol_df, nifty, sub, base, base, hist)
        s = stats(curve, trades, name)
        g = (cfg["switch"] if cfg.get("switch") is not None else cfg["gate"]).reindex(sub).fillna(False)
        s["pct_days_open"] = round(float(g.mean()) * 100)
        return s

    rows = []
    for y in (1, 3, 5, 10):
        sub = dates[dates >= dates[-1] - pd.Timedelta(days=365 * y)]
        for name, over in VARIANTS.items():
            r = run(name, over, sub); r.update(test="A_window", period=f"last {y}y"); rows.append(r)
        print(f"   window {y}y done", flush=True)
    i0 = int(np.searchsorted(dates, dates[-1] - pd.Timedelta(days=3650)))
    for k in range(N_STAGGER):
        si = i0 - k * STAGGER_STEP
        sub = dates[si: si + len(dates) - i0]
        for name, over in VARIANTS.items():
            r = run(name, over, sub); r.update(test="B_stagger", period=f"start {sub[0].date()}"); rows.append(r)
        print(f"   stagger {sub[0].date()} done", flush=True)
    edges = [dates[i0], pd.Timestamp("2019-12-31"), pd.Timestamp("2022-12-31"), dates[-1]]
    periods = []
    for a, b in zip(edges, edges[1:]):
        sub = dates[(dates >= a) & (dates <= b)] if a == edges[0] else dates[(dates > a) & (dates <= b)]
        periods.append((f"{sub[0].date()} -> {sub[-1].date()}", sub))
        for name, over in VARIANTS.items():
            r = run(name, over, sub); r.update(test="C_period", period=periods[-1][0])
            rows.append(r)

    T = pd.DataFrame(rows)
    os.makedirs("analysis", exist_ok=True)
    T.to_csv(OUT_CSV, index=False)

    cols = ["variant", "total_return_pct", "cagr_pct", "max_drawdown_pct", "sharpe", "trades", "pct_days_open"]
    for p in [f"last {y}y" for y in (1, 3, 5, 10)]:
        print(f"\n{'=' * 100}\n{p}\n{'=' * 100}")
        print(T[(T.test == "A_window") & (T.period == p)][cols].to_string(index=False))

    print(f"\n{'=' * 100}\nSCORECARD vs {BASE} — Sharpe wins, median differences "
          f"(d_maxdd > 0 = shallower drawdown)\n{'=' * 100}")
    out = []
    for v in VARIANTS:
        row = {"variant": v}
        for test, lab in [("A_window", "windows"), ("B_stagger", "8 starts"), ("C_period", "3 periods")]:
            t = T[T.test == test]
            b = t[t.variant == BASE].set_index("period")
            x = t[t.variant == v].set_index("period")
            d_sh = x.sharpe - b.sharpe.reindex(x.index)
            d_dd = x.max_drawdown_pct - b.max_drawdown_pct.reindex(x.index)
            row[f"{lab}: wins"] = f"{(d_sh > 0).sum()}/{len(d_sh)}"
            row[f"{lab}: med dSharpe"] = round(d_sh.median(), 2)
            row[f"{lab}: med dMaxDD"] = round(d_dd.median(), 1)
        out.append(row)
    print(pd.DataFrame(out).to_string(index=False))

    print(f"\n{'=' * 100}\nCONTROL — the same switches applied to plain Nifty (in Nifty when on, cash when off)."
          f"\nIf Nifty improves as much as the book, the switch is generic market timing.\n{'=' * 100}")
    ctl = []
    for label, sub in ([(f"last {y}y", dates[dates >= dates[-1] - pd.Timedelta(days=365 * y)]) for y in (3, 5, 10)]
                       + periods):
        for k, sw in SW.items():
            (r0, d0, s0), (r1, d1, s1), fl = nifty_switch(nifty, sub, sw)
            book = T[(T.variant == k) & (T.period == label)]
            bb = T[(T.variant == BASE) & (T.period == label)]
            ctl.append(dict(period=label, switch=k.replace("SWITCH ", ""), nifty_ret=r0, nifty_sw_ret=r1,
                            nifty_dd=d0, nifty_sw_dd=d1, nifty_dSharpe=round(s1 - s0, 2),
                            book_dSharpe=round(book.sharpe.iloc[0] - bb.sharpe.iloc[0], 2) if len(book) and len(bb) else np.nan,
                            switches=fl))
    print(pd.DataFrame(ctl).to_string(index=False))
    print(f"\nsaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
