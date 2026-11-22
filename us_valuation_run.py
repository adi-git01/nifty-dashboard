"""
US valuation for every S&P 500 stock -> data/us_valuation_snapshot.csv (status, rating, P10/P50/P90, CMP percentile).

Price: the latest month-end close in data/us_monthly_prices.csv (the report re-values live at today's
price; this file is a coverage and sanity check after each SEC refresh). Run after us_fundamentals.py.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from utils.valuation import load_context, summary_row, value_stock

OUT = "data/us_valuation_snapshot.csv"


def main():
    ctx = load_context(market="US")
    if ctx.annual.empty or ctx.monthly is None or len(ctx.monthly) == 0:
        sys.exit("[us valuation] inputs missing -- run us_fundamentals.py first")
    last = ctx.monthly.ffill().iloc[-1]
    rows = []
    for t in ctx.funda.index:
        p = last.get(t)
        try:
            res = value_stock(t, ctx, price=float(p) if pd.notna(p) else None, n=4000)
            rows.append(summary_row(res))
        except Exception as e:
            rows.append(dict(ticker=t, status="ERROR", reason=f"{type(e).__name__}: {e}"[:200]))
    S = pd.DataFrame(rows)
    S["as_of"] = pd.Timestamp.today().strftime("%Y-%m-%d")
    S.to_csv(OUT, index=False)
    print("[us valuation] status:", S["status"].value_counts().to_dict())
    if "rating" in S:
        print("[us valuation] ratings:", S["rating"].astype(str).str.split().str[0].value_counts().to_dict())
    if "reason" in S:
        print("[us valuation] top reasons:", S["reason"].dropna().str[:60].value_counts().head(8).to_dict())
    if (S["status"] == "ERROR").mean() > 0.05:
        sys.exit("[us valuation] more than 5% of stocks errored -- check the log")


if __name__ == "__main__":
    main()
