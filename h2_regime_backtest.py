"""
H2 — regime-conditional exposure. Does gating NEW ENTRIES on a
"is the factor paying right now?" signal beat always being invested?

Why this and not another entry signal
-------------------------------------
The 10-year run settled two things. The ranking earns its keep across the
decade (real beats 100% of 20 random-permutation seeds, above p95), but it has
added nothing for three years (mid-band, 65th percentile), and the strategy
trailed Nifty by 4.3pp with a -29.6% drawdown over that stretch. Four separate
entry-refinement tests came back null. So the open question is not "which names"
but "when to be in at all".

How the gates are applied
-------------------------
simulate() treats a NaN in rs_panel as ineligible, so blanking the panel on
"off" dates blocks NEW entries while existing holdings continue to be managed
by the normal exit rules. That is deliberately the conservative form of the
hypothesis — stop adding risk, do not force liquidation — and it needs no
change to the simulator.

LOOK-AHEAD IS THE WHOLE DIFFICULTY
----------------------------------
A trailing-IC gate is only honest if, at date d, it uses forward returns that
have ALREADY COMPLETED. A 21-day forward return measured from signal date s is
not knowable until s+21. So the IC available at d is computed on signal dates
up to d-21 only, and gate_ic() enforces that lag explicitly. Getting this wrong
would manufacture a spectacular and entirely fake result.

    python h2_regime_backtest.py --years 10
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from exit_rule_backtest import (build_rs_panel, fetch, simulate, stats,
                                VARIANTS, BREADTH_NARROW_THRESHOLD)
from momentum_factor_backtest import (build_pit_universe, load_candidates,
                                      shuffle_panel, RS_CAP)

OUT_DIR = "analysis"
H = 21


# --------------------------------------------------------------------------
# gates: each returns a boolean Series over dates — True = allow new entries
# --------------------------------------------------------------------------
def gate_always(dates, **kw):
    return pd.Series(True, index=dates)


def trailing_ic(dates, rs_panel, close_df, window=126):
    """
    Rolling realised IC of CompRS, lagged so it uses only completed returns.

    At date d we may look at signal dates s <= d-H, because the H-day forward
    return from s finished at s+H <= d. Anything more recent is unknowable, so
    the raw IC series is shifted by H before the rolling mean. Exposed
    separately from the gate so the no-look-ahead property is testable on a
    continuous series rather than on a boolean that may simply not flip.
    """
    fwd = close_df.shift(-H) / close_df - 1
    ics = pd.Series(index=dates, dtype=float)
    for d in dates:
        a, b = rs_panel.loc[d], fwd.loc[d]
        m = a.notna() & b.notna()
        ics[d] = spearmanr(a[m], b[m])[0] if m.sum() > 50 else np.nan
    return ics.shift(H).rolling(window, min_periods=window // 2).mean()


def gate_ic(dates, rs_panel, close_df, window=126, **kw):
    return (trailing_ic(dates, rs_panel, close_df, window) > 0).fillna(True)


def gate_equity_trend(dates, ma=50, **kw):
    """Placeholder — resolved inside run() because it needs the equity curve."""
    return pd.Series(True, index=dates)


def gate_nifty_ma200(dates, nifty, **kw):
    n = nifty["Close"].reindex(dates).ffill()
    return (n > n.rolling(200, min_periods=100).mean()).fillna(True)


def gate_breadth(dates, close_df, thresh=45, **kw):
    ma50 = close_df.rolling(50).mean()
    b = ((close_df > ma50).sum(axis=1) / close_df.notna().sum(axis=1) * 100).reindex(dates)
    return (b >= thresh).fillna(True)


def gate_breadth55(dates, close_df, **kw):
    return gate_breadth(dates, close_df, thresh=55)


def gate_combined(dates, **kw):
    """Both the index trend and the factor's own recent payoff must agree."""
    return gate_nifty_ma200(dates, **kw) & gate_ic(dates, **kw)


GATES = {
    "A_baseline (always in)":     gate_always,
    "B_nifty > MA200":            gate_nifty_ma200,
    "C_breadth >= 45%":           gate_breadth,
    "C2_breadth >= 55%":          gate_breadth55,
    "D_trailing IC > 0 (lagged)": gate_ic,
    "E_MA200 AND IC > 0":         gate_combined,
}


def run_window(label, dates_sub, rs_panel, close_df, vol_df, nifty, gates_ctx):
    rows = []
    n0 = nifty["Close"].reindex(dates_sub).ffill()
    nifty_ret = float(n0.iloc[-1] / n0.iloc[0] - 1) * 100
    for name, fn in GATES.items():
        allow = fn(dates_sub, **gates_ctx)
        masked = rs_panel.loc[dates_sub].where(
            pd.DataFrame(np.tile(allow.values[:, None], (1, rs_panel.shape[1])),
                         index=dates_sub, columns=rs_panel.columns))
        curve, trades = simulate(name, VARIANTS["baseline"], close_df, vol_df,
                                 nifty, dates_sub, rs_panel=masked)
        s = stats(curve, trades, name)
        s["window"] = label
        s["pct_days_invested_allowed"] = round(float(allow.mean()) * 100, 0)
        s["excess_vs_nifty_pp"] = round(s["total_return_pct"] - nifty_ret, 1)
        rows.append(s)
    return rows, nifty_ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--max-tickers", type=int, default=0)
    ap.add_argument("--top-n", type=int, default=1000)
    args = ap.parse_args()

    end = datetime.now()
    start = end - timedelta(days=365 * args.years + 400)
    tickers = load_candidates("all", args.max_tickers)
    print(f"[universe] {len(tickers)} candidates; point-in-time top {args.top_n} by turnover")
    close_df, vol_df, nifty = fetch(tickers, start.strftime("%Y-%m-%d"),
                                    end.strftime("%Y-%m-%d"))
    dates = close_df.index
    rs_panel = build_rs_panel(close_df, nifty, dates).where(lambda x: x.abs() <= RS_CAP)
    rs_panel = rs_panel.where(build_pit_universe(close_df, vol_df, args.top_n))
    print(f"[data] {len(dates)} days {dates[0].date()} -> {dates[-1].date()}")

    ctx = dict(rs_panel=rs_panel, close_df=close_df, nifty=nifty)
    all_rows = []
    for y in (1, 3, 5, 10):
        cut = dates[-1] - pd.Timedelta(days=365 * y)
        sub = dates[dates >= cut]
        if len(sub) < 60:
            continue
        rows, nret = run_window(f"last {y}y", sub, rs_panel, close_df, vol_df, nifty, ctx)
        print(f"\n{'=' * 96}\nlast {y}y   (Nifty {nret:+.1f}%)\n{'=' * 96}")
        df = pd.DataFrame(rows)[["variant", "total_return_pct", "cagr_pct",
                                 "max_drawdown_pct", "sharpe", "trades",
                                 "win_rate_pct", "pct_days_invested_allowed",
                                 "excess_vs_nifty_pp"]]
        print(df.to_string(index=False))
        all_rows += rows

    out = pd.DataFrame(all_rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(f"{OUT_DIR}/h2_regime_gates.csv", index=False)
    print(f"\nsaved -> {OUT_DIR}/h2_regime_gates.csv")
    print("\nGates block NEW ENTRIES only; open positions still exit on the normal "
          "rules.\nThe trailing-IC gate is lagged by the forward-return horizon, so it "
          "uses only\nreturns that had completed by the decision date.")


if __name__ == "__main__":
    main()
