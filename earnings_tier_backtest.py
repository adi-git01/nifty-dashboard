"""
Can an earnings overlay turn CompRS's information coefficient positive?

The 10-year run showed CompRS ranking forward returns NEGATIVELY (IC -0.028,
117 independent windows). Binary "earnings growth > 0" nudged cohort returns
but was never tested as an IC conditioner. This asks the sharper question:

    is CompRS's IC positive INSIDE a high-earnings-growth cohort?

Four overlays, all evaluated the same way:
  TIER       growth above 0 / 20 / 40 / 60 / 80 / 100%
  SUSTAINED  above the threshold on every observation across a lookback
  TURNAROUND growth crossed from <= 0 to > 0 within the lookback
  ACCEL      growth today materially above growth a lookback ago

DATA LIMIT — the binding constraint here
----------------------------------------
earningsQuarterlyGrowth exists on 107 daily snapshots spanning ~6.5 months.
At a 21-day horizon that is ~5 independent windows. "Sustained over 2-3
quarters" is therefore NOT directly testable: 6.5 months is two quarters of
calendar, and the field is a trailing YoY figure that only steps when results
print. What SUSTAINED measures here is persistence across a 40-trading-day
lookback, which is a proxy, not the real thing. Read tier ordering and sign,
not significance.

    python earnings_tier_backtest.py
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT_DIR = "analysis"
H = 21
LOOKBACK = 40
RS_CAP = 200.0


def load(cols):
    files = {}
    for f in glob.glob("data/cache/market_master_*.parquet"):
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


def ic_of(rs, fwd, mask, min_n=40):
    """Spearman IC of CompRS vs forward return, restricted to `mask`."""
    ics, fwds, ns = [], [], []
    for d in rs.index:
        a, b, m = rs.loc[d], fwd.loc[d], mask.loc[d]
        sel = a.notna() & b.notna() & m.fillna(False)
        if sel.sum() < min_n:
            continue
        ics.append(spearmanr(a[sel], b[sel])[0])
        fwds.append(b[sel].median() * 100)
        ns.append(int(sel.sum()))
    if not ics:
        return None
    s, f = pd.Series(ics), pd.Series(fwds)
    no = s.iloc[::H]
    se = no.std(ddof=1) / np.sqrt(len(no)) if len(no) > 1 else np.nan
    return {"mean_IC": round(float(s.mean()), 4),
            "IC_nonoverlap": round(float(no.mean()), 4),
            "n_indep": int(len(no)),
            "t_indep": round(float(no.mean() / se), 2) if se and se == se else np.nan,
            "pct_IC_pos": round(float((s > 0).mean()) * 100, 0),
            "fwd_ret_pct": round(float(f.mean()), 2),
            "avg_names": int(np.mean(ns)), "n_days": int(len(s))}


def main():
    P, dates = load(["price", "comp_rs", "fiftyDayAverage", "earningsQuarterlyGrowth"])
    PX = P["price"]
    RS = P["comp_rs"].where(P["comp_rs"].abs() <= RS_CAP)
    MA, G = P["fiftyDayAverage"], P["earningsQuarterlyGrowth"]
    fwd = PX.shift(-H) / PX - 1
    idx = RS.index[:-H]
    RS, fwd, G, PX_, MA_ = RS.loc[idx], fwd.loc[idx], G.loc[idx], PX.loc[idx], MA.loc[idx]
    print(f"panel: {len(dates)} snapshots {dates[0].date()} -> {dates[-1].date()}, "
          f"{PX.shape[1]} tickers | horizon {H}d | usable dates {len(idx)}")
    print(f"earnings growth coverage: {G.notna().mean().mean()*100:.0f}% of cells\n")

    trend = PX_ > MA_                      # the live gate's trend leg
    Gs = G.shift(LOOKBACK)                 # growth as of LOOKBACK days earlier

    rows = []
    r = ic_of(RS, fwd, trend)
    if r:
        rows.append({"overlay": "none (CompRS + >MA50)", **r})

    for thr in (0, 20, 40, 60, 80, 100):
        t = thr / 100.0
        r = ic_of(RS, fwd, trend & (G > t))
        if r:
            rows.append({"overlay": f"TIER growth > {thr}%", **r})

    for thr in (0, 20, 40):
        t = thr / 100.0
        sus = trend & (G > t) & (Gs > t)
        r = ic_of(RS, fwd, sus)
        if r:
            rows.append({"overlay": f"SUSTAINED > {thr}% ({LOOKBACK}d)", **r})

    r = ic_of(RS, fwd, trend & (Gs <= 0) & (G > 0))
    if r:
        rows.append({"overlay": "TURNAROUND (neg -> pos)", **r})
    for lift in (20, 50):
        r = ic_of(RS, fwd, trend & (G > Gs + lift / 100.0) & (G > 0))
        if r:
            rows.append({"overlay": f"ACCEL (+{lift}pp vs {LOOKBACK}d ago)", **r})

    r = ic_of(RS, fwd, trend & (G <= 0))
    if r:
        rows.append({"overlay": "CONTROL growth <= 0", **r})

    df = pd.DataFrame(rows)
    print("=" * 104)
    print("DOES AN EARNINGS OVERLAY FLIP CompRS's IC POSITIVE?")
    print("=" * 104)
    print(df.to_string(index=False))

    base = df[df.overlay.str.startswith("none")]
    if len(base):
        b = base.iloc[0]["mean_IC"]
        df["IC_vs_baseline"] = (df["mean_IC"] - b).round(4)
        print(f"\nbaseline IC = {b:+.4f}; positive IC_vs_baseline means the overlay helps ranking")
        print(df[["overlay", "mean_IC", "IC_vs_baseline", "fwd_ret_pct",
                  "avg_names", "t_indep"]].to_string(index=False))

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(f"{OUT_DIR}/earnings_tier_ic.csv", index=False)
    print(f"\nsaved -> {OUT_DIR}/earnings_tier_ic.csv")
    print("\nCAVEAT: ~5 independent 21d windows and one CAUTION regime. Tier ORDERING "
          "and SIGN are\nthe readable part; no t-stat here can establish significance, "
          "and 'sustained over\n2-3 quarters' is approximated by a 40-day persistence "
          "check, not real quarterly data.")


if __name__ == "__main__":
    main()
