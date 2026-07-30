"""
Local (offline) runner for the exit-rule variants.
==================================================

exit_rule_backtest.py is the real deal: it downloads 2019->today from
yfinance and spans the COVID crash and the 2022 bear. It needs network and
is meant to run in CI.

This runner answers a narrower question with no network at all, by reusing
the daily parquet snapshots the engine already writes:

    data/cache/market_master_YYYY_MM_DD.parquet
        price, comp_rs, fiftyDayAverage, averageVolume  (per ticker per day)
    data/dna3_equity_curve.csv
        Regime per day

Because comp_rs and the 50-day MA are already stored, no Nifty series is
required. The variant definitions and the trailing-stop resolution are
imported from exit_rule_backtest so both runners test *identical* rules.

IMPORTANT LIMITATION: the parquet history is only ~4.5 months (Feb-Jul 2026)
and sits almost entirely in ONE regime (CAUTION). It therefore CANNOT answer
the question the wider stops actually hinge on — how they behave in a
sustained bear. Treat this as a real-data sanity run, not the verdict.

    python exit_rule_backtest_local.py
"""
import glob
import os
import re

import numpy as np
import pandas as pd

from exit_rule_backtest import (VARIANTS, MAX_POSITIONS, BUY_COST, SELL_COST,
                                MIN_INVEST, BREADTH_NARROW_THRESHOLD,
                                INITIAL_CAPITAL, _trail_keep, stats)
from utils.regime_manager import get_regime_params

OUT_DIR = "analysis"


def load_panels():
    files = {}
    for f in glob.glob("data/cache/market_master_*.parquet"):
        m = re.search(r"(\d{4}_\d{2}_\d{2})", f)
        if m:
            files[pd.Timestamp(m.group(1).replace("_", "-"))] = f
    cols = ["price", "comp_rs", "fiftyDayAverage", "averageVolume"]
    acc = {c: {} for c in cols}
    for d in sorted(files):
        s = pd.read_parquet(files[d]).set_index("ticker")
        for c in cols:
            if c in s.columns:
                acc[c][d] = pd.to_numeric(s[c], errors="coerce")
    P = {c: pd.DataFrame(v).T.sort_index() for c, v in acc.items()}

    reg = pd.read_csv("data/dna3_equity_curve.csv")
    reg["Date"] = pd.to_datetime(reg["Date"])
    regime = reg.set_index("Date")["Regime"].replace("", np.nan).ffill().fillna("CAUTION")
    return P["price"], P["comp_rs"], P["fiftyDayAverage"], P["averageVolume"], regime



def _px_or(px, t, fallback):
    """
    Series.get(t, default) returns the STORED value when the key exists — so a
    NaN price comes back as NaN, not the default, and poisons equity/sizing
    (observed: invest became NaN -> int(NaN) ValueError). Same failure class as
    the held-price bug in the live engine: never let a missing quote value a
    position at NaN/zero.
    """
    v = px.get(t)
    return float(v) if v is not None and pd.notna(v) else float(fallback)


def simulate_local(name, cfg, PX, RS, MA, VOL, regime, dates):
    cash = float(INITIAL_CAPITAL)
    holdings, cooldown, trades, curve = {}, {}, [], []
    last_rebal = None
    rs_peak30 = RS.rolling(30, min_periods=5).max()

    for i, d in enumerate(dates):
        px, ma50 = PX.loc[d], MA.loc[d]
        rg = regime.asof(d) if len(regime) else "CAUTION"
        if not isinstance(rg, str):
            rg = "CAUTION"
        params = get_regime_params(rg)
        keep = _trail_keep(rg, params, cfg)

        # ---- exits (every bar) ----
        for t in list(holdings):
            p, m = px.get(t), ma50.get(t)
            if p is None or pd.isna(p):
                continue
            h = holdings[t]
            h["peak"] = max(h["peak"], p)
            h["below"] = h["below"] + 1 if (pd.notna(m) and p < m) else 0

            reason = None
            if h["below"] >= cfg["ma50_confirm"]:
                reason = f"Trend Break (MA50 x{cfg['ma50_confirm']})"
            elif p < h["peak"] * keep:
                reason = f"Trailing Stop ({(1-keep)*100:.0f}% [{rg}])"
            elif cfg.get("rs_decel"):
                rn, rp = RS.at[d, t] if t in RS.columns else np.nan, \
                         rs_peak30.at[d, t] if t in rs_peak30.columns else np.nan
                if pd.notna(rn) and pd.notna(rp) and (rp - rn) >= cfg["rs_decel"]:
                    reason = f"RS Decel (-{cfg['rs_decel']:.0f}pt)"

            if reason:
                proceeds = h["shares"] * p * SELL_COST
                cash += proceeds
                trades.append(dict(variant=name, ticker=t, entry_date=h["entry_date"],
                                   exit_date=d, entry_price=h["entry"], exit_price=p,
                                   pnl=proceeds - h["shares"] * h["entry"],
                                   pnl_pct=(p / h["entry"] - 1) * 100,
                                   reason=reason, regime=rg,
                                   held=(d - h["entry_date"]).days))
                if "Trailing" in reason or "RS Decel" in reason:
                    cooldown[t] = d
                del holdings[t]

        # ---- rebalance / entries ----
        freq = params["rebalance_freq"]
        due = last_rebal is None or (freq < 999 and (i - last_rebal) >= freq)
        if due and params.get("new_entries", True):
            last_rebal = i
            cooldown = {t: dt for t, dt in cooldown.items()
                        if len(dates[(dates > dt) & (dates <= d)]) < freq}
            valid = px.notna() & ma50.notna()
            breadth = 100.0 * ((px > ma50) & valid).sum() / max(valid.sum(), 1)
            free = MAX_POSITIONS - len(holdings)
            if free > 0 and breadth >= BREADTH_NARROW_THRESHOLD:
                rs = RS.loc[d]
                liq = (px * VOL.loc[d]) / 1e7
                elig = (rs.notna() & px.notna() & ma50.notna() & (px > ma50)
                        & (rs >= params["min_comp_rs"] * 100)
                        & (liq >= params["min_liquidity"]))
                for t in list(holdings) + list(cooldown):
                    if t in elig.index:
                        elig[t] = False
                for t in rs[elig].sort_values(ascending=False).index[:free]:
                    equity = cash + sum(h["shares"] * _px_or(px, k, h["entry"])
                                        for k, h in holdings.items())
                    invest = min(equity / MAX_POSITIONS, cash / max(free, 1))
                    p = float(px[t])
                    if invest < MIN_INVEST or p <= 0:
                        continue
                    sh = int(invest / p)
                    if sh > 0 and cash >= sh * p * BUY_COST:
                        cash -= sh * p * BUY_COST
                        holdings[t] = dict(entry=p, shares=sh, peak=p,
                                           entry_date=d, below=0)

        eq = cash + sum(h["shares"] * _px_or(px, t, h["entry"])
                        for t, h in holdings.items())
        curve.append(dict(date=d, equity=eq, regime=rg, holdings=len(holdings)))

    return pd.DataFrame(curve), pd.DataFrame(trades)


def main():
    PX, RS, MA, VOL, regime = load_panels()
    dates = PX.index
    print(f"[local] {len(dates)} trading days {dates.min().date()} -> {dates.max().date()}, "
          f"{PX.shape[1]} tickers")
    print(f"[local] regimes present: {regime.reindex(dates).ffill().value_counts().to_dict()}")
    print("[local] NOTE: single-regime window — NOT the multi-regime verdict.\n")

    rows, alltr = [], []
    for name, cfg in VARIANTS.items():
        curve, tr = simulate_local(name, cfg, PX, RS, MA, VOL, regime, dates)
        s = stats(curve, tr, name)
        s["med_hold_days"] = round(tr["held"].median(), 0) if len(tr) else 0
        s["open_at_end"] = curve["holdings"].iloc[-1]
        rows.append(s)
        if len(tr):
            alltr.append(tr)
    summary = pd.DataFrame(rows)
    base = summary.loc[summary.variant == "baseline", "final_equity"].iloc[0]
    summary["vs_baseline"] = (summary["final_equity"] - base).round()
    pd.set_option("display.width", 220)
    print("=" * 100)
    print("RESULTS (real prices, Feb-Jul 2026, CAUTION regime)")
    print("=" * 100)
    print(summary.to_string(index=False))

    if alltr:
        td = pd.concat(alltr, ignore_index=True)
        print("\nexit-reason mix by variant:")
        td["why"] = td["reason"].str.split(" (", regex=False).str[0]
        print(pd.crosstab(td["variant"], td["why"]).to_string())
        os.makedirs(OUT_DIR, exist_ok=True)
        td.to_csv(f"{OUT_DIR}/exit_rule_local_trades.csv", index=False)
        summary.to_csv(f"{OUT_DIR}/exit_rule_local_summary.csv", index=False)
        print(f"\nsaved -> {OUT_DIR}/exit_rule_local_*.csv")


if __name__ == "__main__":
    main()
