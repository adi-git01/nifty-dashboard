"""
Does the business score (utils/fund_trends.read) say anything about future returns?

    python business_score_backtest.py

Each 1 July from 2021, score every stock using only fiscal years ended by the previous 31 March
(annual data only), then measure the next 12 months' return against the median stock.
Data: data/fundamentals_annual.csv.gz, data/monthly_prices.csv.gz (from Dec 2020).
Limits: today's ~1,170 stocks only (survivors); Screener figures are as restated today.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from utils import fund_trends as ft

M = pd.read_csv("data/monthly_prices.csv.gz", parse_dates=["date"]).pivot_table(index="date", columns="ticker", values="close").sort_index()
rows = []
for y in range(2021, 2027):
    start = pd.Timestamp(f"{y}-07-01")
    end = min(pd.Timestamp(f"{y + 1}-07-01"), M.index[-1])
    if start not in M.index or end not in M.index or end <= start:
        continue
    ret = (M.loc[end] / M.loc[start] - 1) * 100
    ret = ret.clip(-90, 400)
    ex = ret - ret.median()
    for t in M.columns:
        if pd.isna(ex.get(t)):
            continue
        r = ft.read(t, as_of=f"{y}-03-31")
        if r["score"] is None:
            continue
        rows.append(dict(year=y, months=round((end - start).days / 30.4), ticker=t, score=r["score"], verdict=r["verdict"],
                         direction=r["direction"], investing=bool(r["investing"]), hard=bool(r["hard_flags"]), fin=r["financial"],
                         ex=ex[t], **{f"p_{k.split()[0].lower()}": v["score"] for k, v in r["pillars"].items()},
                         **{f"d_{k.split()[0].lower()}": v["dir"] for k, v in r["pillars"].items()}))
D = pd.DataFrame(rows)
D.to_csv("analysis/business_score_backtest.csv", index=False)
pd.set_option("display.width", 200)
print(D.groupby("year").agg(n=("ex", "size"), months=("months", "first")).T.to_string())
full = D[D.months >= 12]
print("\nIC by year (Spearman, score vs next-12m excess):")
ic = lambda g, c: g[c].rank().corr(g["ex"].rank())
cols = ["score", "p_growth", "p_margins", "p_cash", "p_returns"]
print(pd.DataFrame({c: D.groupby("year").apply(lambda g: ic(g, c)) for c in cols}).round(3).to_string())
print("\nBy verdict (median excess pp / hit % / n), full-year cohorts:")
for k in ("verdict", "direction", "investing", "hard"):
    g = full.groupby(["year", k]).ex.median().unstack()
    print(k); print(g.round(1).to_string()); print("  avg:", g.mean().round(1).to_dict(), "| beat-median share:", full.groupby(k).apply(lambda x: round((x.ex > 0).mean() * 100)).to_dict())
full = full.assign(q=full.groupby("year").score.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])))
g = full.groupby(["year", "q"]).ex.median().unstack()
print("\nScore quintile (1 low .. 5 high) median excess:"); print(g.round(1).to_string()); print("  avg:", g.mean().round(1).to_dict(), " mean-based:", full.groupby(["year", "q"]).ex.mean().unstack().mean().round(1).to_dict())
for c in cols[1:]:
    q = full.dropna(subset=[c]).copy()
    q["q"] = q.groupby("year")[c].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]))
    print(c, q.groupby(["year", "q"]).ex.median().unstack().mean().round(1).to_dict())
