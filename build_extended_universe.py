"""
Build the extended list: NSE stocks outside the ranked universe with a market cap of Rs 1,000 cr+.

    python build_extended_universe.py [--minutes 70] [--delay 1.0] [--seed]

Weekly (fundamentals_weekly.yml), after the core scrape:
  1. candidates = every NSE equity-series symbol not in the core list (fresh EQUITY_L.csv from
     the exchange, or the copy in the repo), tracked extras and the current extended list first
  2. scrape Screener for each (fundamentals_annual.scrape): fundamentals, market cap, industry
  3. keep those at or above MIN_MCAP_CR, plus anything you hold or watch
  4. write data/extended_universe.csv (Ticker, Company_Name, Sub_Industry, mcap_cr, price, as_of)
     and data/fundamentals_ext_annual.csv.gz / fundamentals_ext_quarterly.csv.gz

A stock not reached this run (time budget, Screener hiccup) keeps last week's rows.
--seed builds the list without any network from data/fundamentals_annual_extra.csv.gz and
data/industry_extra.csv (how the first list was made).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

from utils import extended
from utils.nifty1000_list import TICKERS_1000

OUT = "data/extended_universe.csv"
A_OUT, Q_OUT = "data/fundamentals_ext_annual.csv.gz", "data/fundamentals_ext_quarterly.csv.gz"
EQ_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
COLS = ["Ticker", "Company_Name", "Sub_Industry", "mcap_cr", "price", "as_of"]


def refresh_eq_list() -> None:
    try:
        import requests
        r = requests.get(EQ_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and "SYMBOL" in r.text[:200]:
            with open(extended.EQ_LIST, "w") as fh:
                fh.write(r.text)
            print(f"[extended] refreshed {extended.EQ_LIST}")
    except Exception as e:
        print(f"[extended] using the EQUITY_L.csv in the repo ({e})")


def _latest(A: pd.DataFrame, field: str) -> pd.Series:
    x = A[A.field == field].sort_values("period")
    return x.groupby("ticker").value.last()


def assemble(A: pd.DataFrame, industries: dict, names: dict, keep_always: set, prev: pd.DataFrame) -> pd.DataFrame:
    """The list rows from annual rows (need mcap_cr), an industry map and names. Pure."""
    mcap, price = _latest(A, "mcap_cr"), _latest(A, "price_scr")
    asof = A[A.field == "mcap_cr"].groupby("ticker").period.max().astype(str)
    old = prev.set_index("Ticker") if len(prev) else pd.DataFrame()
    rows = []
    for t in sorted(set(mcap.index) | (keep_always & set(A.ticker))):
        m = float(mcap.get(t, float("nan")))
        if not (m >= extended.MIN_MCAP_CR or t in keep_always):
            continue
        ind = industries.get(t) or (old.at[t, "Sub_Industry"] if len(old) and t in old.index else "") or "Unknown"
        rows.append(dict(Ticker=t, Company_Name=names.get(t) or (old.at[t, "Company_Name"] if len(old) and t in old.index else t[:-3]),
                         Sub_Industry=" ".join(str(ind).split()), mcap_cr=round(m, 0) if m == m else None,
                         price=price.get(t), as_of=asof.get(t, "")))
    return pd.DataFrame(rows, columns=COLS).sort_values("mcap_cr", ascending=False, na_position="last").reset_index(drop=True)


def _merge(new: pd.DataFrame, path: str, keep: set) -> pd.DataFrame:
    """New rows for scraped tickers, last run's rows for the rest; only `keep` tickers are stored."""
    if os.path.exists(path):
        try:
            old = pd.read_csv(path)
            new = pd.concat([new, old[~old.ticker.isin(set(new.ticker))].reindex(columns=new.columns)], ignore_index=True)
        except Exception:
            pass
    return new[new.ticker.isin(keep)]


def seed() -> pd.DataFrame:
    A = pd.read_csv("data/fundamentals_annual_extra.csv.gz")
    ind = dict(pd.read_csv("data/industry_extra.csv").values)
    names = extended.eq_symbols()
    core = set(TICKERS_1000)
    A = A[~A.ticker.isin(core) & A.ticker.isin(set(names))]          # still listed in the equity series
    return assemble(A, ind, names, set(), pd.DataFrame())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=70)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--seed", action="store_true")
    args = ap.parse_args()
    if args.seed:
        U = seed()
        U.to_csv(OUT, index=False)
        print(f"[extended] seeded {len(U)} stocks at or above Rs {extended.MIN_MCAP_CR:,} cr -> {OUT}")
        return 0
    import fundamentals_annual as fa
    refresh_eq_list()
    names, core = extended.eq_symbols(), set(TICKERS_1000)
    prev = extended.universe()
    track = set(extended.tracked(core))
    first = [t for t in list(track) + prev.Ticker.tolist() if t not in core]
    rest = sorted(set(names) - core - set(first))
    if rest:                                      # start somewhere new each week: the time budget must not starve the same tail
        k = (pd.Timestamp.today().isocalendar()[1] * 211) % len(rest)
        rest = rest[k:] + rest[:k]
    order = list(dict.fromkeys(first + rest))
    print(f"[extended] {len(order)} candidates ({len(track)} tracked, {len(prev)} already listed); budget {args.minutes:.0f} min")
    industries: dict = {}
    A, Q = fa.scrape(order, args.delay, industries, deadline=time.time() + args.minutes * 60)
    if A is None or len(A) == 0:
        print("[extended] nothing scraped -- keeping the previous list")
        return 0
    A = pd.DataFrame(A, columns=["ticker", "field", "period", "value", "basis"]) if not isinstance(A, pd.DataFrame) else A
    Q = pd.DataFrame(Q, columns=["ticker", "field", "period", "value"]) if not isinstance(Q, pd.DataFrame) else Q
    if os.path.exists(A_OUT):                                         # stocks not reached this run keep last week's figures
        oldA = pd.read_csv(A_OUT)
        A_all = pd.concat([A, oldA[~oldA.ticker.isin(set(A.ticker))].reindex(columns=A.columns)], ignore_index=True)
    else:
        A_all = A
    U = assemble(A_all.assign(period=A_all.period.astype(str)), {**dict(zip(prev.Ticker, prev.Sub_Industry)), **industries},
                 names, track, prev)
    if len(U) < 0.5 * max(len(prev), 1):
        print(f"[extended] only {len(U)} stocks (was {len(prev)}) -- keeping the previous list")
        return 0
    keep = set(U.Ticker)
    _merge(A, A_OUT, keep).to_csv(A_OUT, index=False)
    _merge(Q, Q_OUT, keep).to_csv(Q_OUT, index=False)
    U.to_csv(OUT, index=False)
    print(f"[extended] {len(U)} stocks -> {OUT}; industries: {U.Sub_Industry.nunique()}; unknown industry: {int((U.Sub_Industry == 'Unknown').sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
