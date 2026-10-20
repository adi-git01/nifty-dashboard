"""
What works in which market regime?
==================================

Every signal studied so far -- momentum, chart triggers, v6 states, valuation zones, size, industry --
re-scored by the market regime on the day it fired, plus the live engine's own return in each regime.
Same point-in-time data and fixes as entry_exit_study.py (weekly stock-dates 2016-2026, full-EQ top-1000
universe, past members' fundamentals, bonus/split repair, results lagged 92 days).

Regimes (defined before the run; computed daily, a row takes its own day's regime)
  NIFTY (primary, the engine's view; dd = fall from the 252-day closing high, slope = 20-day change in MA200)
    BEAR       dd <= -15% and close < MA200
    MILD BEAR  close < MA200 (not BEAR)
    CHOPPY     close >= MA200 but within 3% of it, or dd <= -8%
    BULL       close > 1.03 x MA200, dd > -4%, MA200 rising
    MILD BULL  everything else above MA200
  BREADTH (secondary, the broad market's view: % of the universe above its own 200-day average)
    BEAR < 30   MILD BEAR 30-40   CHOPPY 40-50   MILD BULL 50-60   BULL >= 60

A signal "works" in a regime when, pre-registered: median 3-month return vs the typical stock >= +1 pp AND
median 6-month >= +2 pp AND positive (6m) in the majority of that regime's separate episodes (>= 3
episodes). Cells need >= 200 rows from >= 30 stocks. Returns are vs the typical stock, so "works in a bear
market" means beating other stocks, not making money.

Run: python regime_study.py            (Actions: ranking_rotation_backtest.yml, script=regime)
     python regime_study.py --synthetic
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

OUT = "analysis"
REGIMES = ["BEAR", "MILD BEAR", "CHOPPY", "MILD BULL", "BULL"]
MIN_N, MIN_STOCKS, MIN_EPISODES = 200, 30, 3


def nifty_regime(nc: pd.Series) -> pd.Series:
    ma200 = nc.rolling(200, min_periods=150).mean()
    dd = nc / nc.rolling(252, min_periods=150).max() - 1
    slope = ma200 / ma200.shift(20) - 1
    r = np.select([(dd <= -0.15) & (nc < ma200), nc < ma200,
                   (nc <= 1.03 * ma200) | (dd <= -0.08),
                   (nc > 1.03 * ma200) & (dd > -0.04) & (slope > 0)],
                  ["BEAR", "MILD BEAR", "CHOPPY", "BULL"], "MILD BULL")
    return pd.Series(np.where(ma200.notna(), r, ""), index=nc.index)


def breadth_regime(close: pd.Series, pit: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    ma200 = close.apply(lambda s: s.dropna().rolling(200).mean()).reindex(close.index)
    above = (close > ma200).where(pit & ma200.notna())
    pct = above.mean(axis=1) * 100
    r = np.select([pct < 30, pct < 40, pct < 50, pct < 60], ["BEAR", "MILD BEAR", "CHOPPY", "MILD BULL"], "BULL")
    return pd.Series(np.where(pct.notna(), r, ""), index=close.index), pct


def placebo_thresholds(R, col, reps=200, seed=7):
    """
    95th percentile of the 6m median of RANDOM baskets drawn within each regime, by basket size. Baskets
    are built whole-stock at a time (all of a stock's rows in that regime), because a stock's overlapping
    weekly rows move together -- sampling single rows would understate how far a median can wander.
    """
    rng = np.random.default_rng(seed)
    sizes = np.array([200, 500, 1000, 2000, 5000, 10000, 20000, 50000])
    out = {}
    for rg in REGIMES:
        x = R[(R[col] == rg) & R.vu126.notna()].sort_values("ticker")
        if len(x) < 200:
            continue
        v = x.vu126.values
        codes, starts, counts = np.unique(x.ticker.values, return_index=True, return_counts=True)
        th = {}
        for n in sizes[sizes <= len(x)]:
            meds = []
            for _ in range(reps):
                order = rng.permutation(len(codes))
                k = int(np.searchsorted(np.cumsum(counts[order]), n)) + 1
                pick = order[:k]
                meds.append(np.median(np.concatenate([v[starts[p]:starts[p] + counts[p]] for p in pick])))
            th[int(n)] = float(np.percentile(meds, 95))
        out[rg] = th
    return out


def threshold_for(th, rg, n):
    """Conservative: the threshold of the largest tabulated basket size not above n."""
    t = th.get(rg, {})
    ks = [k for k in t if k <= n]
    return t[max(ks)] if ks else np.nan


def by_regime(R, cells, col, title):
    th = placebo_thresholds(R, col)
    rows = []
    for name, m in cells.items():
        x0 = R[m.reindex(R.index, fill_value=False)]
        for rg in REGIMES:
            x = x0[x0[col] == rg]
            r = dict(table=title, signal=name, regime=rg, n=len(x), stocks=x.ticker.nunique())
            if len(x) >= 30:
                for h, lab in ((21, "1m"), (63, "3m"), (126, "6m")):
                    v = x[f"vu{h}"].dropna()
                    r[f"{lab} median"] = round(v.median(), 2) if len(v) >= 30 else np.nan
                r["6m beat%"] = round(100 * (x.vu126.dropna() > 0).mean())
                # an episode = a stretch of this regime; gaps under 30 days (day-to-day flicker) do not split it
                xe = x.dropna(subset=["vu126"])
                d = pd.Series(sorted(xe.date.unique()))
                ep_id = dict(zip(d, (d.diff().dt.days > 30).cumsum()))
                g = xe.groupby(xe.date.map(ep_id)).vu126
                ep = g.median()[g.size() >= 10]
                r["episodes +/total"] = f"{(ep > 0).sum()}/{len(ep)}"
                r["placebo 95%"] = round(threshold_for(th, rg, len(x)), 2)
                r["works"] = bool(len(x) >= MIN_N and r["stocks"] >= MIN_STOCKS and len(ep) >= MIN_EPISODES
                                  and r.get("3m median", np.nan) >= 1 and r.get("6m median", np.nan) >= 2
                                  and (ep > 0).sum() > len(ep) / 2
                                  and r.get("6m median", np.nan) > r["placebo 95%"])
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 500)
    tag = "_synthetic" if args.synthetic else ""
    import entry_exit_study as es
    from valuation_study import engine_curve

    if args.synthetic:
        close, high, low, vol, nifty, pit, gscore, A, groups = es.synthetic_inputs()
        raw = close.copy()
        nifty_df = pd.DataFrame({"Close": nifty, "High": nifty * 1.01, "Low": nifty * 0.99})
    else:
        from utils.nifty1000_list import SUB_INDUSTRY_MAP
        from utils.yf_safe import safe_history
        close, high, low, vol, raw, nifty, pit = es.load_prices(args.max_tickers)
        ever = pit.iloc[260::5]
        ever = list(ever.columns[ever.loc[ever.index >= "2015-06-01"].any()])
        A = es.load_annual(ever, 0, 1.0, 0)
        groups = {t: g for t, g in SUB_INDUSTRY_MAP.items() if g and g != "Unknown"}
        if os.path.exists(es.EXTRA_IND):
            extra = pd.read_csv(es.EXTRA_IND).dropna()
            groups.update({t: g for t, g in zip(extra.ticker, extra.industry) if t not in groups and g})
        close, high, low, vol, raw = es.fix_corporate_actions(close, high, low, vol, raw, A)
        gscore = es.industry_scores(close, pit, groups, nifty)
        nifty_df = safe_history("^NSEI", start="2014-06-01")
        if nifty_df.index.tz is not None:
            nifty_df.index = nifty_df.index.tz_localize(None)
        nifty_df = nifty_df.reindex(close.index).ffill()
    close, high, low, vol, raw, _ = es.clean_spikes(close, high, low, vol, raw)
    fy = es.fy_table(A)
    R, P = es.build_rows(close, high, low, vol, raw, nifty, pit, gscore, fy, groups)
    R = es.add_forensics(R, P)

    # regimes
    nreg = nifty_regime(nifty_df["Close"].reindex(close.index).ffill())
    breg, bpct = breadth_regime(close, pit)
    for name, reg in (("nifty", nreg), ("breadth", breg)):
        R[name] = reg.values[R.i.values]
    last = close.index[-1]
    print(f"[regime] TODAY ({last:%d %b %Y}): Nifty {nreg.iloc[-1] or 'n/a'}; breadth {breg.iloc[-1]} "
          f"({bpct.iloc[-1]:.0f}% of the universe above its 200-day average)")
    span = close.index >= pd.Timestamp("2016-01-01")
    for name, reg in (("Nifty", nreg), ("breadth", breg)):
        share = reg[span].value_counts(normalize=True).mul(100).round(0).reindex(REGIMES).to_dict()
        n_ep = {}
        for rg in REGIMES:
            d = pd.Series(reg[span].index[reg[span] == rg])
            n_ep[rg] = int((d.diff().dt.days > 30).sum() + (len(d) > 0))
        print(f"[regime] {name}: % of days {share}; separate episodes {n_ep}")
    print(f"[regime] Nifty vs breadth agreement: {100 * (nreg[span] == breg[span]).mean():.0f}% of days")

    # signals (squeeze release from the S5 trigger panels, fresh within 10 sessions)
    trig = es.trigger_panels(P)
    sq = trig["squeeze release (BB width < 20th pct, close > upper band)"].astype(float).rolling(10, min_periods=1).max() > 0
    R["sq_fresh"] = sq.values[R.i.values, R.j.values]
    R["rsi14"] = R["rsi"]
    mom = (R.rk126 >= 90) & R.gate1
    m = lambda s: s.fillna(False) if hasattr(s, "fillna") else s
    peer_q = R.rel.where(R.pe.notna()).groupby(R.date).rank(pct=True)
    cells = {
        # momentum
        "momentum: 6m rank >= 90 + Gate 1": mom,
        "momentum + 1m ACCEL": mom & (R.velocity == "ACCEL"),
        "momentum + 1m FADING": mom & (R.velocity == "FADING"),
        "6m rank <= 10 (losers)": m(R.rk126 <= 10),
        # charts
        "momentum + fresh breakout": mom & R.fresh_bo,
        "momentum + squeeze release": mom & R.sq_fresh,
        "breakout + volume anchor + leader": R.fresh_bo_anchor & R.leader,
        "v6 BUY NOW": R.buy_now,
        "v6 ABSORPTION": R.signature == "ABSORPTION",
        "v6 DISTRIBUTION": R.signature == "DISTRIBUTION",
        "Gate 1 trend (any)": R.gate1,
        "within 5% of 52w high": m(R.dist_hi >= -5),
        "> 40% below 52w high": m(R.dist_hi < -40),
        "RSI(14) < 30 (oversold)": m(R.rsi14 < 30),
        "oversold above the 200d (RSI < 40, > MA200)": m(R.rsi14 < 40) & m(R.ext200 > 0),
        # valuation
        "valuation CHEAP": R.zone == "CHEAP", "valuation FAIR": R.zone == "FAIR",
        "valuation RICH": R.zone == "RICH", "loss-maker / P/E >= 200": R.zone == "LOSS",
        "P/E vs peers, cheapest 20%": m(peer_q <= 0.2), "P/E >= 60": m(R.pe >= 60),
        "momentum + CHEAP/FAIR": mom & R.zone.isin(["CHEAP", "FAIR"]), "momentum + RICH": mom & (R.zone == "RICH"),
        # size / industry
        "Large cap": R.tier == "Large", "Mid cap": R.tier == "Mid", "Small cap": R.tier == "Small",
        "Micro cap": R.tier == "Micro", "leader industry": R.leader, "laggard industry": m(R.gscore < 40),
    }
    tables = []
    for col, title in (("nifty", "Nifty regime"), ("breadth", "breadth regime")):
        T = by_regime(R, cells, col, title)
        tables.append(T)
        print(f"\n{'=' * 150}\nBY {title.upper()}: median 3m / 6m return vs the typical stock (pp), n, episodes positive at 6m\n{'=' * 150}")
        piv = T.pivot_table(index="signal", columns="regime", values="6m median", aggfunc="first").reindex(
            index=list(cells), columns=REGIMES)
        print("6-month median vs typical:\n" + piv.round(1).to_string())
        piv3 = T.pivot_table(index="signal", columns="regime", values="3m median", aggfunc="first").reindex(
            index=list(cells), columns=REGIMES)
        print("\n3-month median vs typical:\n" + piv3.round(1).to_string())
        print("\nWORKS (pre-registered rule):")
        for rg in REGIMES:
            w = T[(T.regime == rg) & T.works.fillna(False).astype(bool)]
            print(f"  {rg:9s}: " + (", ".join(f"{s} ({a:+.1f} / {b:+.1f}, ep {e})" for s, a, b, e in
                                               zip(w.signal, w["3m median"], w["6m median"], w["episodes +/total"])) or "nothing"))
        print("\nDetail:\n" + T.drop(columns="table").to_string(index=False))

    # the live engine by regime
    eng = engine_curve(close, vol, nifty_df, pit)
    er = eng.pct_change().dropna()
    nr = nifty_df["Close"].reindex(eng.index).pct_change().dropna()
    rows = []
    for col, reg in (("Nifty", nreg), ("breadth", breg)):
        rg_ = reg.reindex(er.index)
        for r_ in REGIMES:
            mk = rg_ == r_
            if mk.sum() < 20:
                continue
            e_, n_ = er[mk], nr.reindex(er.index)[mk]
            rows.append(dict(regime_by=col, regime=r_, days=int(mk.sum()),
                             engine_ann_ret=round(((1 + e_).prod() ** (252 / len(e_)) - 1) * 100, 1),
                             nifty_ann_ret=round(((1 + n_).prod() ** (252 / len(n_)) - 1) * 100, 1),
                             engine_daily_vol_ann=round(e_.std() * np.sqrt(252) * 100, 1),
                             engine_hit_days=round(100 * (e_ > 0).mean())))
    TE = pd.DataFrame(rows)
    tables.append(TE.assign(table="live engine by regime"))
    print(f"\n{'=' * 150}\nLIVE ENGINE BY REGIME (annualised return while in that regime)\n{'=' * 150}\n{TE.to_string(index=False)}")

    os.makedirs(OUT, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(f"{OUT}/regime_study{tag}.csv", index=False)
    print(f"\nsaved -> {OUT}/regime_study{tag}.csv")


if __name__ == "__main__":
    main()
