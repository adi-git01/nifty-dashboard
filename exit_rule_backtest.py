"""
Multi-Regime Exit-Rule Backtest for OptComp-V22
===============================================

Answers one question properly: do the proposed exit-rule changes
(-15% trailing stop, 2-day MA50 confirmation) actually beat the current
rules, or did they only look good on one V-shaped recovery?

WHY THIS EXISTS
---------------
An external review claimed a 15x improvement from those changes. That
estimate had three defects this script is built to avoid:

  1. RE-ENTRY DOUBLE-COUNT. In the live log STLTECH was two trades: exit
     17-Mar (+1.5%), re-entry 16-Apr, exit 11-Jun (+121.7%). Under a wider
     stop the first trade never exits, so the re-entry cannot happen — yet
     the review counted BOTH (+205% and +114%) over the same overlapping
     window, roughly a third of its headline gain. Here each variant runs
     its OWN full portfolio simulation, so a name that is still held simply
     cannot be bought again, and capital is genuinely consumed.

  2. MARK-TO-TODAY. "Saved" trades were valued at the final price rather
     than re-simulated. Here a saved trade keeps living under its own
     variant's rules and exits at a real later trigger.

  3. ONE REGIME. The sample was a single 6-week correction that V-recovered.
     Widening stops is optimal for a V and dangerous in a sustained bear, so
     results here are broken out per calendar year AND per market regime
     (the COVID crash and the 2022 bear are the acid tests).

A further subtlety the review missed: the trailing stop is already
regime-adaptive (BULL 15% / CAUTION 12% / BEAR 8% / CRISIS 6%). "-12% ->
-15%" is really "make CAUTION behave like BULL", so both that narrow change
and a naive flat-15% (which discards the adaptivity) are tested separately.

All variants share identical entry logic, so any difference is attributable
to exits alone.

USAGE
-----
    python exit_rule_backtest.py                      # 2019-01-01 -> today
    python exit_rule_backtest.py --start 2019-01-01 --max-tickers 250
    python exit_rule_backtest.py --smoke               # tiny run, checks wiring

Outputs analysis/exit_rule_backtest_summary.csv + _trades.csv.
Network access is required (yfinance); it works in CI where the daily
engine already runs.
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.regime_manager import classify_regime, get_regime_params  # noqa: E402

# ── Live-engine constants (mirrored from dna3_current_portfolio.py) ──────────
INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS = 15
RS_WEIGHTS = [(5, 0.10), (21, 0.50), (63, 0.40)]
BREADTH_NARROW_THRESHOLD = 30      # skip new buys when breadth is this weak
BUY_COST, SELL_COST = 1.002, 0.998  # impact/transaction cost
MIN_INVEST = 5000
OUT_DIR = "analysis"


# ── Exit-rule variants ───────────────────────────────────────────────────────
# trail_delta : percentage POINTS added to the regime's trailing stop
#               (applied only to regimes listed in trail_regimes, None = all)
# trail_flat  : if set, overrides the regime trail entirely (kills adaptivity)
# ma50_confirm: consecutive closes below MA50 required to exit (1 = current)
#
# rs_decel : exit when composite RS falls more than this many POINTS below its
#            own trailing 30-day peak (None = rule disabled). Motivated by the
#            live trade log: every exit occurred long AFTER RS had rolled over
#            (median RS drop from 30d peak at exit = -27.9pts; winners -37.8),
#            i.e. the price stop is a lagging confirmation of a deterioration
#            RS flagged much earlier.
VARIANTS = {
    "baseline":            dict(trail_delta=0.0,  trail_regimes=None,        trail_flat=None, ma50_confirm=1, rs_decel=None),
    "B1_caution_trail15":  dict(trail_delta=0.03, trail_regimes=["CAUTION"], trail_flat=None, ma50_confirm=1, rs_decel=None),
    "B1b_trail_flat15":    dict(trail_delta=0.0,  trail_regimes=None,        trail_flat=0.15, ma50_confirm=1, rs_decel=None),
    "B2_ma50_confirm2":    dict(trail_delta=0.0,  trail_regimes=None,        trail_flat=None, ma50_confirm=2, rs_decel=None),
    "B3_combined":         dict(trail_delta=0.03, trail_regimes=["CAUTION"], trail_flat=None, ma50_confirm=2, rs_decel=None),
    "B3b_flat15_confirm2": dict(trail_delta=0.0,  trail_regimes=None,        trail_flat=0.15, ma50_confirm=2, rs_decel=None),
    # --- new: wider stops, sized from the measured drawdown distribution ---
    # Winners that ran >40% to peak gave back a median -15.6% (p25 -17.9%)
    # from their running peak DURING the ascent, so a -12% trail ejects ~73%
    # of them before they top out. -18% is the level that survives ~90%.
    "B4_caution_trail18":  dict(trail_delta=0.06, trail_regimes=["CAUTION"], trail_flat=None, ma50_confirm=1, rs_decel=None),
    "B4b_trail_flat18":    dict(trail_delta=0.0,  trail_regimes=None,        trail_flat=0.18, ma50_confirm=1, rs_decel=None),
    # --- new: RS-deceleration exit ---
    "B5_rs_decel30":       dict(trail_delta=0.0,  trail_regimes=None,        trail_flat=None, ma50_confirm=1, rs_decel=30.0),
    "B6_trail18_decel30":  dict(trail_delta=0.06, trail_regimes=["CAUTION"], trail_flat=None, ma50_confirm=1, rs_decel=30.0),
}


def _trail_keep(regime: str, params: dict, cfg: dict) -> float:
    """Fraction of peak price to hold above, per variant."""
    if cfg["trail_flat"] is not None:
        stop = cfg["trail_flat"]
    else:
        stop = params["trail_stop"]
        if cfg["trail_delta"] and (cfg["trail_regimes"] is None or regime in cfg["trail_regimes"]):
            stop += cfg["trail_delta"]
    return 1.0 - stop


# ── Data ─────────────────────────────────────────────────────────────────────
def fetch(tickers, start, end):
    """Bulk-download closes; returns (close_df, nifty_df). Uses yf_safe retries."""
    from utils.yf_safe import safe_download, safe_history

    print(f"[data] Nifty {start} -> {end}")
    nifty = safe_history("^NSEI", start=start, end=end)
    if nifty.empty:
        raise SystemExit("Could not fetch Nifty — aborting.")
    if nifty.index.tz is not None:
        nifty.index = nifty.index.tz_localize(None)

    print(f"[data] {len(tickers)} tickers (this is the slow part)...")
    bulk = safe_download(tickers, start=start, end=end, group_by="ticker",
                         threads=False, auto_adjust=True, min_coverage=0.5)
    if bulk is None or bulk.empty:
        raise SystemExit("Bulk download failed — aborting.")

    closes, vols = {}, {}
    multi = isinstance(bulk.columns, pd.MultiIndex)
    for t in tickers:
        try:
            sub = bulk[t] if multi else bulk
            c = sub["Close"].dropna()
            if len(c) > 250:                      # need history for MA200/RS63
                closes[t] = c
                vols[t] = sub["Volume"].reindex(c.index)
        except Exception:
            continue
    close_df = pd.DataFrame(closes)
    vol_df = pd.DataFrame(vols)
    if close_df.index.tz is not None:
        close_df.index = close_df.index.tz_localize(None)
        vol_df.index = vol_df.index.tz_localize(None)
    print(f"[data] usable tickers: {close_df.shape[1]}")
    return close_df, vol_df, nifty


# ── Simulation ───────────────────────────────────────────────────────────────
def build_rs_panel(close_df, nifty, dates):
    """
    Daily composite RS vs Nifty for every ticker (10% RS5 + 50% RS21 + 40% RS63),
    vectorised. Computed once and shared by all variants: entries need it for
    ranking/thresholds, and the RS-deceleration exit needs it per held name on
    every bar. Row i uses only prices up to row i, so it carries no look-ahead.
    """
    c = close_df.reindex(dates)
    n = nifty["Close"].reindex(dates).ffill()
    rs = pd.DataFrame(0.0, index=dates, columns=c.columns)
    valid = pd.DataFrame(True, index=dates, columns=c.columns)
    for period, w in RS_WEIGHTS:
        stock_r = (c / c.shift(period) - 1) * 100
        nifty_r = (n / n.shift(period) - 1) * 100
        rs = rs.add((stock_r).sub(nifty_r, axis=0) * w, fill_value=0.0)
        valid &= c.shift(period).notna()
    return rs.where(valid)


def simulate(name, cfg, close_df, vol_df, nifty, dates, rs_panel=None):
    """
    One independent portfolio simulation. Decisions at date d use only data
    up to and including d (no look-ahead).
    """
    if rs_panel is None:
        rs_panel = build_rs_panel(close_df, nifty, dates)
    # trailing 30-bar peak of RS, for the deceleration exit
    rs_peak30 = rs_panel.rolling(30, min_periods=5).max()
    cash = float(INITIAL_CAPITAL)
    holdings = {}          # ticker -> dict(entry_price, shares, peak, entry_date, below_ma50)
    cooldown = {}          # ticker -> exit date (post trailing-stop)
    trades, curve = [], []
    last_rebal_idx = None

    ma50_all = close_df.rolling(50).mean()
    ma200_n = nifty["Close"].rolling(200).mean()
    hi52_n = nifty["High"].rolling(252).max()

    for i, d in enumerate(dates):
        px = close_df.loc[d]
        ma50 = ma50_all.loc[d]

        # --- regime (from Nifty, as-of d) ---
        try:
            regime = classify_regime(float(nifty["Close"].asof(d)),
                                     float(ma200_n.asof(d)),
                                     float(hi52_n.asof(d)))
        except Exception:
            regime = "CAUTION"
        params = get_regime_params(regime)
        keep = _trail_keep(regime, params, cfg)

        # --- 1. EXITS (checked every day, as in the live engine) ---
        for t in list(holdings.keys()):
            p = px.get(t)
            m = ma50.get(t)
            if p is None or np.isnan(p):
                continue
            h = holdings[t]
            if p > h["peak"]:
                h["peak"] = p

            # consecutive-closes-below-MA50 counter
            if m is not None and not np.isnan(m) and p < m:
                h["below_ma50"] += 1
            else:
                h["below_ma50"] = 0

            reason = None
            if h["below_ma50"] >= cfg["ma50_confirm"]:
                reason = f"Trend Break (MA50 x{cfg['ma50_confirm']})"
            elif p < h["peak"] * keep:
                reason = f"Trailing Stop ({(1-keep)*100:.0f}% [{regime}])"
            elif cfg.get("rs_decel"):
                # RS rolled over well before price did — exit on the leading signal
                rs_now = rs_panel.at[d, t] if t in rs_panel.columns else np.nan
                rs_pk = rs_peak30.at[d, t] if t in rs_peak30.columns else np.nan
                if pd.notna(rs_now) and pd.notna(rs_pk) and (rs_pk - rs_now) >= cfg["rs_decel"]:
                    reason = f"RS Decel (-{cfg['rs_decel']:.0f}pt from 30d peak)"

            if reason:
                proceeds = h["shares"] * p * SELL_COST
                cost_basis = h["shares"] * h["entry_price"]
                cash += proceeds
                trades.append(dict(variant=name, ticker=t, entry_date=h["entry_date"],
                                   exit_date=d, entry_price=h["entry_price"], exit_price=p,
                                   pnl=proceeds - cost_basis,
                                   pnl_pct=(p / h["entry_price"] - 1) * 100,
                                   reason=reason, regime=regime))
                # Cooldown after a *momentum-failure* exit (trailing stop or RS
                # deceleration). Without it an RS-decel exit is re-bought at the
                # very next rebalance while RS is still depressed, producing
                # exit/re-entry churn — observed in testing as a same-week
                # round trip. A trend-break exit keeps the live engine's
                # behaviour (no cooldown).
                if "Trailing" in reason or "RS Decel" in reason:
                    cooldown[t] = d
                del holdings[t]

        # --- 2. REBALANCE / ENTRIES ---
        rebal_days = params["rebalance_freq"]
        due = last_rebal_idx is None or (rebal_days < 999 and (i - last_rebal_idx) >= rebal_days)
        if due and params.get("new_entries", True):
            last_rebal_idx = i
            cooldown = {t: dt for t, dt in cooldown.items()
                        if len(dates[(dates > dt) & (dates <= d)]) < rebal_days}

            # market breadth — skip buys in a narrow tape (live engine rule)
            valid = px.notna() & ma50.notna()
            breadth = 100.0 * ((px > ma50) & valid).sum() / max(valid.sum(), 1)

            free = MAX_POSITIONS - len(holdings)
            if free > 0 and breadth >= BREADTH_NARROW_THRESHOLD:
                # composite RS vs Nifty as-of d, from the shared panel
                rs_total = rs_panel.loc[d]
                ok = rs_total.notna()

                liq_cr = (px * vol_df.loc[d]) / 1e7          # ₹ Cr turnover
                elig = (ok & px.notna() & ma50.notna() & (px > ma50)
                        & (rs_total >= params["min_comp_rs"] * 100)
                        & (liq_cr >= params["min_liquidity"]))
                for t in list(holdings) + list(cooldown):
                    if t in elig.index:
                        elig[t] = False

                for t in rs_total[elig].sort_values(ascending=False).index[:free]:
                    equity = cash + sum(h["shares"] * px.get(k, h["entry_price"])
                                        for k, h in holdings.items())
                    invest = min(equity / MAX_POSITIONS, cash / max(free, 1))
                    p = float(px[t])
                    if invest < MIN_INVEST or p <= 0:
                        continue
                    sh = int(invest / p)
                    spend = sh * p * BUY_COST
                    if sh > 0 and cash >= spend:
                        cash -= spend
                        holdings[t] = dict(entry_price=p, shares=sh, peak=p,
                                           entry_date=d, below_ma50=0)

        # --- 3. mark equity ---
        eq = cash + sum(h["shares"] * (px.get(t) if px.get(t) and not np.isnan(px.get(t))
                                       else h["entry_price"]) for t, h in holdings.items())
        curve.append(dict(date=d, equity=eq, regime=regime, holdings=len(holdings)))

    return pd.DataFrame(curve), pd.DataFrame(trades)


# ── Reporting ────────────────────────────────────────────────────────────────
def stats(curve, trades, name):
    eq = curve["equity"]
    ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    yrs = max((curve["date"].iloc[-1] - curve["date"].iloc[0]).days / 365.25, 1e-9)
    cagr = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100
    dd = ((eq - eq.cummax()) / eq.cummax() * 100).min()
    dr = eq.pct_change().dropna()
    sharpe = (dr.mean() / dr.std() * np.sqrt(252)) if dr.std() else 0.0
    w = trades["pnl_pct"] > 0 if len(trades) else pd.Series(dtype=bool)
    return dict(variant=name, final_equity=round(eq.iloc[-1]), total_return_pct=round(ret, 1),
                cagr_pct=round(cagr, 2), max_drawdown_pct=round(dd, 1), sharpe=round(sharpe, 2),
                trades=len(trades), win_rate_pct=round(100 * w.mean(), 1) if len(trades) else 0.0,
                avg_win_pct=round(trades.loc[w, "pnl_pct"].mean(), 2) if w.any() else 0.0,
                avg_loss_pct=round(trades.loc[~w, "pnl_pct"].mean(), 2) if (~w).any() else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--max-tickers", type=int, default=400,
                    help="cap universe for speed; 0 = all")
    ap.add_argument("--smoke", action="store_true", help="tiny fast run to check wiring")
    a = ap.parse_args()
    if a.smoke:
        a.start, a.max_tickers = "2023-01-01", 60

    from utils.nifty1000_list import TICKERS_1000
    tickers = [t for t in TICKERS_1000 if t != "DUMMYALCAR.NS"]
    if a.max_tickers:
        tickers = tickers[: a.max_tickers]

    close_df, vol_df, nifty = fetch(tickers, a.start, a.end)
    dates = close_df.index[close_df.index >= pd.Timestamp(a.start)]
    dates = dates[dates.isin(nifty.index)]
    print(f"[sim] {len(dates)} trading days, {close_df.shape[1]} tickers, "
          f"{len(VARIANTS)} variants\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    print("[sim] building shared RS panel ...", flush=True)
    rs_panel = build_rs_panel(close_df, nifty, dates)

    rows, all_trades, curves = [], [], {}
    for name, cfg in VARIANTS.items():
        print(f"  running {name} ...", flush=True)
        curve, trades = simulate(name, cfg, close_df, vol_df, nifty, dates, rs_panel)
        curves[name] = curve
        rows.append(stats(curve, trades, name))
        if len(trades):
            all_trades.append(trades)

    summary = pd.DataFrame(rows)
    base = summary.loc[summary["variant"] == "baseline", "final_equity"].iloc[0]
    summary["vs_baseline_rs"] = (summary["final_equity"] - base).round()
    print("\n" + "=" * 78)
    print("OVERALL")
    print("=" * 78)
    print(summary.to_string(index=False))

    # Per-year and per-regime: where does each rule actually help or hurt?
    print("\n" + "=" * 78)
    print("RETURN % BY CALENDAR YEAR  (the regime acid test)")
    print("=" * 78)
    yr = {}
    for name, c in curves.items():
        c = c.set_index("date")
        yr[name] = c["equity"].resample("YE").last().pct_change().mul(100).round(1)
        first_y = c["equity"].resample("YE").first().iloc[0]
        last_y = c["equity"].resample("YE").last().iloc[0]
        yr[name].iloc[0] = round((last_y / first_y - 1) * 100, 1)
    print(pd.DataFrame(yr).to_string())

    print("\n" + "=" * 78)
    print("REALISED TRADE P&L % BY REGIME AT EXIT")
    print("=" * 78)
    if all_trades:
        td = pd.concat(all_trades, ignore_index=True)
        print(td.pivot_table(index="regime", columns="variant", values="pnl_pct",
                             aggfunc="mean").round(2).to_string())
        td.to_csv(f"{OUT_DIR}/exit_rule_backtest_trades.csv", index=False)
    summary.to_csv(f"{OUT_DIR}/exit_rule_backtest_summary.csv", index=False)
    print(f"\nSaved -> {OUT_DIR}/exit_rule_backtest_summary.csv and _trades.csv")


if __name__ == "__main__":
    main()
