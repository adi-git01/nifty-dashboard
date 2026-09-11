"""
Backtest H1 / H3 / H4 from the momentum-article discussion.

H1  Does fundamental confirmation improve momentum entries?
H3  Do the IAS components and tiers predict anything?
H4  Does extension above MA50 at entry predict forward return?

Statistical note that governs every number here
-----------------------------------------------
Daily observations of a 21-day forward return overlap by 20 of 21 days, so
naive t-stats across ~80 daily cross-sections are inflated by roughly sqrt(21).
Every test below therefore reports BOTH the overlapping series (for direction,
using all data) and a non-overlapping series sampled every H days (for
significance, the honest sample). Where the two disagree, the non-overlapping
one wins.

The panel covers ~6 months in a single CAUTION regime. Nothing here can
establish that a rule works across regimes; it can only establish whether a
rule worked in the window we have.

    python hypothesis_backtest.py
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

OUT_DIR = "analysis"
CACHE = "data/cache/market_master_*.parquet"
IAS_LOG = "data/ias_signal_log.csv"


# ---------------------------------------------------------------------------
def load_panel(cols):
    files = {}
    for f in glob.glob(CACHE):
        m = re.search(r"(\d{4}_\d{2}_\d{2})", f)
        if m:
            files[pd.Timestamp(m.group(1).replace("_", "-"))] = f
    acc = {c: {} for c in cols}
    for d in sorted(files):
        s = pd.read_parquet(files[d]).set_index("ticker")
        for c in cols:
            if c in s.columns:
                acc[c][d] = pd.to_numeric(s[c], errors="coerce")
    P = {c: pd.DataFrame(v).T.sort_index() for c, v in acc.items()}
    common = sorted(set.intersection(*[set(v.index) for v in P.values()]))
    return {c: v.reindex(common) for c, v in P.items()}, common


def summarise(series, label, H):
    """Mean of a per-date statistic, with an honest non-overlapping variant."""
    s = pd.Series(series).dropna()
    if s.empty:
        return None
    nonover = s.iloc[::H]
    se = nonover.std(ddof=1) / np.sqrt(len(nonover)) if len(nonover) > 1 else np.nan
    return {
        "label": label,
        "mean_all": round(float(s.mean()), 2),
        "n_all": int(len(s)),
        "mean_nonoverlap": round(float(nonover.mean()), 2),
        "n_indep": int(len(nonover)),
        "t_indep": round(float(nonover.mean() / se), 2) if se and se == se else np.nan,
    }


def show(rows, title):
    print(f"\n{title}")
    df = pd.DataFrame([r for r in rows if r])
    if df.empty:
        print("  (no data)")
        return df
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
def h1_fundamental_confirmation(H=21):
    print("=" * 78)
    print("H1 — DOES FUNDAMENTAL CONFIRMATION IMPROVE MOMENTUM ENTRIES?")
    print("=" * 78)
    P, dates = load_panel(["price", "comp_rs", "fiftyDayAverage",
                           "earningsQuarterlyGrowth", "revenueGrowth",
                           "earningsTrend", "profitMargins"])
    PX, RS, MA = P["price"], P["comp_rs"].where(P["comp_rs"].abs() <= 200), P["fiftyDayAverage"]
    fwd = PX.shift(-H) / PX - 1
    idx = RS.index[:-H]
    print(f"panel: {len(dates)} dates {dates[0].date()} -> {dates[-1].date()}, "
          f"{PX.shape[1]} tickers | horizon {H}d")

    gates = {
        "baseline (CompRS>=17, >MA50)": lambda d: pd.Series(True, index=PX.columns),
        "+ earnings growth > 0":        lambda d: P["earningsQuarterlyGrowth"].loc[d] > 0,
        "+ revenue growth > 0":         lambda d: P["revenueGrowth"].loc[d] > 0,
        "+ earnings TREND > 0":         lambda d: P["earningsTrend"].loc[d] > 0,
        "+ eps AND rev growth > 0":     lambda d: (P["earningsQuarterlyGrowth"].loc[d] > 0)
                                                 & (P["revenueGrowth"].loc[d] > 0),
        "+ earnings growth <= 0 (inv)": lambda d: P["earningsQuarterlyGrowth"].loc[d] <= 0,
    }
    series = {k: [] for k in gates}
    counts = {k: [] for k in gates}
    paired = []          # per-date (confirmed - unconfirmed), the cleanest read
    for d in idx:
        base = (RS.loc[d] >= 17) & (PX.loc[d] > MA.loc[d]) & RS.loc[d].notna() & fwd.loc[d].notna()
        if base.sum() < 20:
            continue
        for k, g in gates.items():
            m = base & g(d).reindex(PX.columns).fillna(False)
            if m.sum() >= 5:
                series[k].append(fwd.loc[d][m].median() * 100)
                counts[k].append(int(m.sum()))
        c = base & (P["earningsQuarterlyGrowth"].loc[d] > 0).reindex(PX.columns).fillna(False)
        u = base & (P["earningsQuarterlyGrowth"].loc[d] <= 0).reindex(PX.columns).fillna(False)
        if c.sum() >= 5 and u.sum() >= 5:
            paired.append(fwd.loc[d][c].median() * 100 - fwd.loc[d][u].median() * 100)

    rows = [summarise(series[k], k, H) for k in gates]
    for r, k in zip(rows, gates):
        if r:
            r["avg_names"] = int(np.mean(counts[k])) if counts[k] else 0
    df = show(rows, f"median {H}d forward return by entry gate (%)")

    pr = summarise(paired, "confirmed MINUS unconfirmed (paired)", H)
    print(f"\npaired daily difference — the cleanest read:")
    print(f"  all dates      : {pr['mean_all']:+.2f}pp over {pr['n_all']} overlapping dates")
    print(f"  non-overlapping: {pr['mean_nonoverlap']:+.2f}pp over {pr['n_indep']} independent "
          f"windows, t={pr['t_indep']}")
    if pr["n_indep"] < 8:
        print("  NOTE: too few independent windows to call this significant either way.")
    return df


# ---------------------------------------------------------------------------
def h3_ias_components():
    print("\n" + "=" * 78)
    print("H3 — DO THE IAS COMPONENTS AND TIERS PREDICT ANYTHING?")
    print("=" * 78)
    if not os.path.exists(IAS_LOG):
        print("  no IAS log")
        return None
    d = pd.read_csv(IAS_LOG)
    comps = ["signal_ias", "signal_ias_vel", "signal_ias_lfl", "signal_ias_price",
             "signal_ias_rs63", "signal_comp_rs", "signal_off_ma50",
             "signal_liq_from_low", "signal_vol_quality", "signal_rs21_delta5",
             "signal_off_52w_low", "signal_shock_ratio"]
    hz = ["return_5d", "return_21d", "return_63d"]
    for c in comps + hz:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    rows = []
    for c in [x for x in comps if x in d.columns]:
        r = {"component": c.replace("signal_", "")}
        for h in hz:
            s = d.dropna(subset=[c, h])
            if len(s) > 30:
                ic, p = spearmanr(s[c], s[h])
                r[h.replace("return_", "IC_")] = round(ic, 3)
                r[h.replace("return_", "p_")] = round(p, 3)
                r[h.replace("return_", "n_")] = len(s)
        rows.append(r)
    comp_df = show(rows, "Spearman IC of each IAS input vs fixed-horizon return\n"
                         "(the log stores true fixed horizons — no holding-time confound)")

    print("\ntier separation (median % return):")
    for h in hz:
        if h in d.columns:
            s = d.dropna(subset=[h])
            g = s.groupby("signal_tier", observed=True)[h].agg(["median", "mean", "count"])
            g = g.reindex([t for t in ["WATCH", "READY", "ALERT"] if t in g.index])
            print(f"\n  {h}:")
            print(g.round(2).to_string())
            if {"ALERT", "WATCH"} <= set(s["signal_tier"]):
                a = s[s.signal_tier == "ALERT"][h]
                w = s[s.signal_tier == "WATCH"][h]
                if len(a) > 5 and len(w) > 5:
                    u, p = mannwhitneyu(a, w, alternative="two-sided")
                    print(f"    ALERT vs WATCH: Mann-Whitney p={p:.3f} "
                          f"({'ALERT worse' if a.median() < w.median() else 'ALERT better'})")
    return comp_df


# ---------------------------------------------------------------------------
def h4_extension(H=21):
    print("\n" + "=" * 78)
    print("H4 — DOES EXTENSION ABOVE MA50 AT ENTRY PREDICT FORWARD RETURN?")
    print("=" * 78)
    P, dates = load_panel(["price", "comp_rs", "fiftyDayAverage"])
    PX, RS, MA = P["price"], P["comp_rs"].where(P["comp_rs"].abs() <= 200), P["fiftyDayAverage"]
    fwd = PX.shift(-H) / PX - 1
    idx = RS.index[:-H]

    buckets = {"0-5%": (0, 5), "5-10%": (5, 10), "10-20%": (10, 20),
               "20-35%": (20, 35), ">35%": (35, 1e9)}

    for gate_name, rs_floor in [("entry gate CompRS>=17", 17.0), ("whole universe", None)]:
        series, counts, rsmean = {k: [] for k in buckets}, {k: [] for k in buckets}, {k: [] for k in buckets}
        for d in idx:
            p, ma, rs, f = PX.loc[d], MA.loc[d], RS.loc[d], fwd.loc[d]
            ext = (p / ma - 1) * 100
            base = p.notna() & ma.notna() & f.notna() & ext.notna() & (p > ma)
            if rs_floor is not None:
                base &= (rs >= rs_floor)
            for k, (lo, hi) in buckets.items():
                m = base & (ext >= lo) & (ext < hi)
                if m.sum() >= 5:
                    series[k].append(f[m].median() * 100)
                    counts[k].append(int(m.sum()))
                    rsmean[k].append(float(rs[m].median()))
        rows = []
        for k in buckets:
            r = summarise(series[k], k, H)
            if r:
                r["avg_names"] = int(np.mean(counts[k]))
                r["median_CompRS"] = round(float(np.mean(rsmean[k])), 1)
            rows.append(r)
        show(rows, f"median {H}d forward return by MA50 extension — {gate_name}")
        print("  (median_CompRS exposes the confound: if it rises with extension, the "
              "bucket comparison is\n   partly a CompRS comparison, not an extension one.)")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    h1 = h1_fundamental_confirmation()
    h3 = h3_ias_components()
    h4_extension()
    if h1 is not None and not h1.empty:
        h1.to_csv(f"{OUT_DIR}/h1_fundamental_confirmation.csv", index=False)
    if h3 is not None and not h3.empty:
        h3.to_csv(f"{OUT_DIR}/h3_ias_components.csv", index=False)
    print(f"\nsaved -> {OUT_DIR}/h1_*.csv, {OUT_DIR}/h3_*.csv")
