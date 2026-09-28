"""
Ranking weights, rotation and IPO eligibility — 1/3/5/10-year backtest
=======================================================================

Tests three changes suggested by the winners post-mortem
(analysis/WINNERS_POSTMORTEM.md), each of which looked promising on seven
months of snapshots but was within that window's noise:

  K  RANKING KEY. The 1-month leg carries 50% of CompRS but was the weakest
     single measure in the snapshot window. Variants rank the SAME eligible set
     by keys with less or no 1-month weight.
  R  ROTATION. On rebalance days, sell holdings whose ranking key has fallen
     below rank N in the universe, freeing slots for current leaders.
  I  RECENT IPOs. The live engine skips any stock with <= 200 bars of history.

Design choices that keep the comparison clean:
  - Entry GATES are identical in every variant: standard CompRS >= regime floor,
    price > MA50, liquidity, breadth >= 30%. Only the ORDER in which eligible
    names fill free slots changes (K), or when a holding is released (R), or
    which names are eligible by age (I). Reweighting CompRS would otherwise
    change the scale of the 17-point floor and confound the test.
  - The baseline is the live engine as specified: CompRS 10/50/40, 200-bar
    history rule, no rotation, working breadth gate.
  - Point-in-time universe (top N by trailing turnover, per date) and the RS
    corrupt-bar cap, as in the H2 and momentum-factor backtests.
  - Each window is a fresh book started at the window's first date.

PRE-REGISTERED DECISION RULE (fixed before any result was seen):
  A variant is worth adopting only if it beats the baseline's Sharpe in at
  least 3 of the 4 windows, one of which must be the 10-year window, and does
  not worsen max drawdown by more than 3 points in any window. Everything
  else is reported but treated as not established.

Residual bias: delisted names are absent from every source here, which
flatters all variants roughly equally.

Run: python ranking_rotation_backtest.py --years 10 --max-tickers 0
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from exit_rule_backtest import (BREADTH_NARROW_THRESHOLD, BUY_COST, INITIAL_CAPITAL,
                                MAX_POSITIONS, MIN_INVEST, SELL_COST, build_rs_panel,
                                fetch, ma50_per_stock, stats)
from utils.regime_manager import classify_regime, get_regime_params

RS_CAP = 200.0
OUT_CSV = "analysis/ranking_rotation_backtest.csv"
WINDOWS = (1, 3, 5, 10)

VARIANTS = {
    "BASE live (cRS 10/50/40, hist>200)": dict(key="base"),
    "K1 cRS 10/25/65":                    dict(key="rw_10_25_65"),
    "K2 RS 3m":                           dict(key="rs63"),
    "K3 RS 6m":                           dict(key="rs126"),
    "K4 RS 1y":                           dict(key="rs252"),
    "K5 12-1 momentum":                   dict(key="mom_12_1"),
    "K6 blend 20/40/40 (1m/3m/1y)":       dict(key="blend_long"),
    "K7 dist above 200dma":               dict(key="dist200"),
    "R1 rotate if rank > 100":            dict(rotate=100),
    "R2 rotate if rank > 50":             dict(rotate=50),
    "I1 IPOs: hist > 64":                 dict(min_hist=64),
    "I2 IPOs: hist > 126":                dict(min_hist=126),
}
# gate      : None = breadth >= 30% over all priced names (the harness rule);
#             a boolean Series by date = allow new entries only where True
# ma50_exit : sell a holding that closes below its MA50 (live rule)
# ma50_entry: require price > MA50 to enter (live rule)
# switch    : None, or a boolean Series by date (True = risk-on). On a risk-off
#             day every holding is sold and nothing is bought; the day it
#             turns back on forces a rebalance instead of waiting for the
#             13-session timer.
DEFAULTS = dict(key="base", rotate=None, min_hist=200, gate=None, ma50_exit=True, ma50_entry=True,
                switch=None)


def rel(c, n, p):
    """Excess return over Nifty across p bars, in points."""
    return (c / c.shift(p) - 1) * 100 - ((n / n.shift(p) - 1) * 100).values[:, None]


def build_keys(close_df, nifty, dates, base):
    c = close_df.reindex(dates)
    n = nifty["Close"].reindex(dates).ffill()
    r5, r21, r63 = rel(c, n, 5), rel(c, n, 21), rel(c, n, 63)
    r126, r252 = rel(c, n, 126), rel(c, n, 252)
    m12_1 = (c.shift(21) / c.shift(252) - 1) * 100 \
        - ((n.shift(21) / n.shift(252) - 1) * 100).values[:, None]
    keys = {
        "base": base,
        "rw_10_25_65": 0.10 * r5 + 0.25 * r21 + 0.65 * r63,
        "rs63": r63, "rs126": r126, "rs252": r252, "mom_12_1": m12_1,
        "blend_long": 0.20 * r21 + 0.40 * r63 + 0.40 * r252,
        "dist200": (c / c.rolling(200).mean() - 1) * 100,
    }
    # Same universe mask and corrupt-bar guard as the base panel: a key is
    # only defined where the standard CompRS is.
    return {k: v.where(base.notna()) for k, v in keys.items()}


def simulate(name, cfg, close_df, vol_df, nifty, dates, elig_rs, key, hist):
    """
    The exit_rule_backtest.simulate() baseline (regime trail, MA50 exit, cooldown
    after trailing stops, breadth gate, equal weight), with three switches:
    a separate ranking key, rotation on rebalance days, and a history rule.
    Decisions at date d use only data up to d.
    """
    cash = float(INITIAL_CAPITAL)
    holdings, cooldown, trades, curve = {}, {}, [], []
    last_rebal_idx = None
    was_on = True
    ma50_all = ma50_per_stock(close_df)
    ma200_n = nifty["Close"].rolling(200).mean()
    hi52_n = nifty["High"].rolling(252).max()
    hist_ok = hist > cfg["min_hist"]

    def sell(t, d, p, reason, regime):
        nonlocal cash
        h = holdings.pop(t)
        cash += h["shares"] * p * SELL_COST
        trades.append(dict(variant=name, ticker=t, entry_date=h["entry_date"], exit_date=d,
                           entry_price=h["entry_price"], exit_price=p,
                           pnl=h["shares"] * (p * SELL_COST - h["entry_price"]),
                           pnl_pct=(p / h["entry_price"] - 1) * 100, reason=reason, regime=regime))

    for i, d in enumerate(dates):
        px, ma50 = close_df.loc[d], ma50_all.loc[d]
        try:
            regime = classify_regime(float(nifty["Close"].asof(d)), float(ma200_n.asof(d)),
                                     float(hi52_n.asof(d)))
        except Exception:
            regime = "CAUTION"
        params = get_regime_params(regime)
        keep = 1.0 - params["trail_stop"]

        on = True if cfg["switch"] is None else bool(cfg["switch"].get(d, False))
        if not on:
            for t in list(holdings):
                p = px.get(t)
                if p is not None and not np.isnan(p):
                    sell(t, d, p, "Breadth risk-off", regime)
        turned_on = on and not was_on
        was_on = on

        for t in list(holdings):
            p, m = px.get(t), ma50.get(t)
            if p is None or np.isnan(p):
                continue
            h = holdings[t]
            h["peak"] = max(h["peak"], p)
            if cfg["ma50_exit"] and m is not None and not np.isnan(m) and p < m:
                sell(t, d, p, "Trend Break", regime)
            elif p < h["peak"] * keep:
                sell(t, d, p, "Trailing Stop", regime)
                cooldown[t] = d

        rebal_days = params["rebalance_freq"]
        due = last_rebal_idx is None or turned_on or \
            (rebal_days < 999 and (i - last_rebal_idx) >= rebal_days)
        if due and on and params.get("new_entries", True):
            last_rebal_idx = i
            cooldown = {t: dt for t, dt in cooldown.items()
                        if len(dates[(dates > dt) & (dates <= d)]) < rebal_days}
            k_today = key.loc[d]

            if cfg["rotate"]:
                rank = k_today.rank(ascending=False)
                for t in list(holdings):
                    r, p = rank.get(t), px.get(t)
                    if r is not None and not np.isnan(r) and r > cfg["rotate"] \
                            and p is not None and not np.isnan(p):
                        sell(t, d, p, "Rotate", regime)

            valid = px.notna() & ma50.notna()
            breadth = 100.0 * ((px > ma50) & valid).sum() / max(valid.sum(), 1)
            free = MAX_POSITIONS - len(holdings)
            allowed = (breadth >= BREADTH_NARROW_THRESHOLD) if cfg["gate"] is None \
                else bool(cfg["gate"].get(d, False))
            if free > 0 and allowed:
                rs_total = elig_rs.loc[d]
                liq_cr = (px * vol_df.loc[d]) / 1e7
                above = (px > ma50) if cfg["ma50_entry"] else pd.Series(True, index=px.index)
                elig = (rs_total.notna() & px.notna() & ma50.notna() & above
                        & (rs_total >= params["min_comp_rs"] * 100)
                        & (liq_cr >= params["min_liquidity"]) & hist_ok.loc[d])
                for t in list(holdings) + list(cooldown):
                    if t in elig.index:
                        elig[t] = False
                # Rank the eligible set by this variant's key; names the key
                # cannot score yet (e.g. <1y of history for the 1y key) go last.
                order = k_today[elig].fillna(-np.inf).sort_values(ascending=False).index
                for t in order[:free]:
                    equity = cash + sum(h["shares"] * px.get(k, h["entry_price"])
                                        for k, h in holdings.items())
                    invest = min(equity / MAX_POSITIONS, cash / max(free, 1))
                    p = float(px[t])
                    if invest < MIN_INVEST or p <= 0:
                        continue
                    sh = int(invest / p)
                    if sh > 0 and cash >= sh * p * BUY_COST:
                        cash -= sh * p * BUY_COST
                        holdings[t] = dict(entry_price=p, shares=sh, peak=p, entry_date=d)

        eq = cash + sum(h["shares"] * (px.get(t) if px.get(t) and not np.isnan(px.get(t))
                                       else h["entry_price"]) for t, h in holdings.items())
        curve.append(dict(date=d, equity=eq, regime=regime, holdings=len(holdings)))
    return pd.DataFrame(curve), pd.DataFrame(trades)


def verdict(tab):
    """Apply the pre-registered rule to the per-window table."""
    base_name = next(iter(VARIANTS))
    rows = []
    for v in VARIANTS:
        if v == base_name:
            continue
        wins, dd_ok, have10, d10 = 0, True, False, np.nan
        for w in tab.window.unique():
            b = tab[(tab.window == w) & (tab.variant == base_name)].iloc[0]
            x = tab[(tab.window == w) & (tab.variant == v)]
            if x.empty:
                continue
            x = x.iloc[0]
            if x.sharpe > b.sharpe:
                wins += 1
                have10 = have10 or w == "last 10y"
            if x.max_drawdown_pct < b.max_drawdown_pct - 3:
                dd_ok = False
            if w == "last 10y":
                d10 = round(x.sharpe - b.sharpe, 2)
        ok = wins >= 3 and have10 and dd_ok
        rows.append(dict(variant=v, sharpe_wins=f"{wins}/{tab.window.nunique()}",
                         wins_10y=have10, dd_within_3pts=dd_ok, sharpe_diff_10y=d10,
                         verdict="ADOPT-WORTHY" if ok else "not established"))
    return pd.DataFrame(rows)


def main():
    from momentum_factor_backtest import build_pit_universe, load_candidates

    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--max-tickers", type=int, default=0)
    ap.add_argument("--top-n", type=int, default=1000)
    args = ap.parse_args()

    end = datetime.now()
    start = end - timedelta(days=365 * args.years + 450)   # 1y key + MA200 warm-up
    tickers = load_candidates("all", args.max_tickers)
    print(f"[universe] {len(tickers)} candidates; point-in-time top {args.top_n} by turnover")
    close_df, vol_df, nifty = fetch(tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    dates = close_df.index[close_df.index.isin(nifty.index)]

    base = build_rs_panel(close_df, nifty, dates).where(lambda x: x.abs() <= RS_CAP)
    base = base.where(build_pit_universe(close_df, vol_df, args.top_n).reindex(dates))
    keys = build_keys(close_df, nifty, dates, base)
    hist = close_df.reindex(dates).notna().cumsum()
    print(f"[data] {len(dates)} days {dates[0].date()} -> {dates[-1].date()}, "
          f"{close_df.shape[1]} tickers, {len(VARIANTS)} variants x {len(WINDOWS)} windows")

    rows = []
    for y in WINDOWS:
        cut = dates[-1] - pd.Timedelta(days=365 * y)
        sub = dates[dates >= cut]
        if len(sub) < 60:
            continue
        n0, n1 = nifty["Close"].asof(sub[0]), nifty["Close"].asof(sub[-1])
        label = f"last {y}y"
        for name, over in VARIANTS.items():
            cfg = {**DEFAULTS, **over}
            print(f"  {label:9s} {name} ...", flush=True)
            # full price history in, window dates to iterate: MA50/MA200 are
            # then already warm on the window's first day
            curve, trades = simulate(name, cfg, close_df, vol_df, nifty, sub,
                                     base, keys[cfg["key"]], hist)
            s = stats(curve, trades, name)
            s.update(window=label, nifty_pct=round((n1 / n0 - 1) * 100, 1),
                     turnover_trades_per_yr=round(len(trades) / max(y, 1), 1))
            if len(trades):
                s["rotations"] = int((trades.reason == "Rotate").sum())
            rows.append(s)

    tab = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    tab.to_csv(OUT_CSV, index=False)
    cols = ["variant", "total_return_pct", "cagr_pct", "max_drawdown_pct", "sharpe",
            "trades", "win_rate_pct"]
    for w in tab.window.unique():
        t = tab[tab.window == w]
        print(f"\n{'=' * 100}\n{w}   (Nifty {t.nifty_pct.iloc[0]:+.1f}%)\n{'=' * 100}")
        print(t[cols].to_string(index=False))
    print(f"\n{'=' * 100}\nPRE-REGISTERED VERDICT: beat baseline Sharpe in >=3 of 4 windows incl. "
          f"10y, max DD no worse than baseline by >3 pts in any window\n{'=' * 100}")
    print(verdict(tab).to_string(index=False))
    print(f"\nsaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
