"""
Industry turn study -- does an industry-level "turning up" status predict that
industry beating the market? Run before building the industry alerts.

Monthly approximation (data/monthly_prices.csv.gz, current universe, 2020-26):
  f50  = % of members whose close is above their 3-month average  (~50-day)
  f200 = % of members above their 10-month average                 (~200-day)
  rs3  = median member 3-month return minus the universe median
Status ladder:
  TURNING UP  f200 < 50, f50 >= 60, rs3 > 0
  UPTREND     f200 >= 50, f50 >= 50
  WEAKENING   f200 >= 50, f50 < 50
  DOWNTREND   f200 < 50, f50 < 40
  BASING      everything else below the 200-day line
Outcome: equal-weight industry basket return minus the equal-weight universe,
over the next 3 and 6 months.

Pre-registered pass criteria for "TURNING UP" entries (from DOWNTREND/BASING):
  1. mean and median 3m and 6m excess > 0
  2. 6m hit rate (excess > 0) >= 55%
  3. better than buying while still in DOWNTREND (bottom-fishing)

Run: python industry_turn_study.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nifty1000_list import SUB_INDUSTRY_MAP  # noqa: E402

MIN_MEMBERS = 5


def classify(f50, f200, rs):
    if not (np.isfinite(f50) and np.isfinite(f200)):
        return None
    if f200 >= 50:
        return "UPTREND" if f50 >= 50 else "WEAKENING"
    if f50 >= 60 and np.isfinite(rs) and rs > 0:
        return "TURNING UP"
    return "DOWNTREND" if f50 < 40 else "BASING"


def main():
    m = pd.read_csv("data/monthly_prices.csv.gz", parse_dates=["date"])
    px = m.pivot_table(index="date", columns="ticker", values="close").sort_index()
    px = px[px.notna().mean(axis=1) > 0.5]             # stray mid-month rows (one ticker) break the windows
    px = px[px.index < px.index.max()]                 # drop the partial current month
    a50, a200 = px.rolling(3).mean(), px.rolling(10).mean()
    r3 = px.pct_change(3, fill_method=None)
    ret = px.pct_change(fill_method=None).shift(-1)                    # next month's return
    uni = ret.mean(axis=1)
    groups = pd.Series(SUB_INDUSTRY_MAP).reindex(px.columns)

    rows = []
    for g, mem in groups.groupby(groups):
        mem = list(mem.index)
        if len(mem) < MIN_MEMBERS:
            continue
        for d in px.index[10:]:
            ok = px.loc[d, mem].notna() & a200.loc[d, mem].notna()
            live = [t for t in mem if ok[t]]
            if len(live) < MIN_MEMBERS:
                continue
            f50 = 100 * (px.loc[d, live] > a50.loc[d, live]).mean()
            f200 = 100 * (px.loc[d, live] > a200.loc[d, live]).mean()
            rs3 = (r3.loc[d, live].median() - r3.loc[d].median()) * 100
            rows.append((g, d, f50, f200, rs3, classify(f50, f200, rs3)))
    S = pd.DataFrame(rows, columns=["group", "date", "f50", "f200", "rs3", "status"])
    # equal-weight basket index per group, and whether it sits above its 10-month average
    bidx = {}
    for g, mem in groups.groupby(groups):
        b = (1 + px[list(mem.index)].pct_change(fill_method=None).mean(axis=1).fillna(0)).cumprod()
        bidx[g] = b > b.rolling(10).mean()
    S["above10"] = [bool(bidx[g].get(d, False)) for g, d in zip(S.group, S.date)]

    # forward excess: compound basket and universe over h months
    def fwd(g, d, h):
        mem = [t for t in groups.index[groups == g]]
        i = px.index.get_loc(d)
        if i + h >= len(px.index):
            return np.nan
        idx = px.index[i:i + h]
        b = (1 + ret.loc[idx, mem].mean(axis=1)).prod() - 1
        u = (1 + uni.loc[idx]).prod() - 1
        return 100 * (b - u)

    S = S.sort_values(["group", "date"])
    S["prev"] = S.groupby("group").status.shift()
    S["prev_rs3"] = S.groupby("group").rs3.shift()
    S["prev_f200"] = S.groupby("group").f200.shift()
    S["prev_above10"] = S.groupby("group").above10.shift()
    for h in (3, 6):
        S[f"x{h}"] = [fwd(g, d, h) for g, d in zip(S.group, S.date)]

    def summ(mask, label):
        x = S[mask]
        out = {"n": len(x)}
        for h in (3, 6):
            v = x[f"x{h}"].dropna()
            out.update({f"mean{h}": v.mean(), f"med{h}": v.median(), f"hit{h}": 100 * (v > 0).mean()})
        print(f"{label:46s} " + "  ".join(f"{k} {v:6.1f}" for k, v in out.items()))
        return out

    print(f"{S.group.nunique()} industries, {S.date.min():%b %Y} - {S.date.max():%b %Y}; excess vs equal-weight universe, pp\n")
    entry = (S.status == "TURNING UP") & S.prev.isin(["DOWNTREND", "BASING"])
    t = summ(entry, "Enters TURNING UP (from down/basing)")
    summ(S.status == "TURNING UP", "Any month in TURNING UP")
    d = summ(S.status == "DOWNTREND", "Any month in DOWNTREND (bottom-fishing)")
    summ((S.status == "DOWNTREND") & (S.prev == "DOWNTREND"), "Still DOWNTREND (2nd month+)")
    summ(S.status == "BASING", "Any month in BASING")
    summ((S.status == "UPTREND") & (S.prev == "TURNING UP"), "TURNING UP -> UPTREND")
    summ(S.status == "UPTREND", "Any month in UPTREND")
    summ(S.status == "WEAKENING", "Any month in WEAKENING")
    summ((S.status == "WEAKENING") & (S.prev == "UPTREND"), "Enters WEAKENING (from uptrend)")
    summ(S.status.notna(), "All industry-months")
    print("\nPre-defined alternatives (confirmed turns):")
    summ((S.prev_f200 < 50) & (S.f200 >= 50) & (S.prev_f200 < 30).groupby(S.group).transform(lambda x: x.rolling(6, min_periods=1).max()).astype(bool),
         "A. >=50% back above 200d after a <30% low (6m)")
    summ((S.prev_rs3 <= 0) & (S.rs3 > 0) & (S.f200 < 50), "B. RS turns positive while below 200d")
    summ((S.prev_above10 == False) & S.above10, "C. basket index crosses its 10m average")
    summ((S.prev_above10 == False) & S.above10 & (S.rs3 > 0), "C+RS. index crosses 10m avg and RS > 0")

    c1 = all(t[k] > 0 for k in ("mean3", "med3", "mean6", "med6"))
    c2 = t["hit6"] >= 55
    c3 = t["mean6"] > d["mean6"] and t["med6"] > d["med6"]
    print(f"\ncriterion 1 (excess > 0): {c1}   2 (6m hit >= 55%): {c2}   3 (beats bottom-fishing): {c3}")

    # what does the history say for hotels-style groups right now
    print("\nLatest status by industry:")
    last = S[S.date == S.date.max()].sort_values("f200")
    print(last[["group", "f50", "f200", "rs3", "status"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
