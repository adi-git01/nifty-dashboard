"""
US fundamentals for the valuation engine, from SEC EDGAR (official 10-K / 10-Q filings, free).

Writes the same long format the Indian files use (ticker, field, period, value), so utils/valuation.py
runs unchanged on US stocks:

    data/us_fundamentals_annual.csv.gz      10-K years: sales, op, interest, dep, pbt, np, reserves (equity),
                                            eq_cap (0), borrowings, fixed_assets, total_liab, deposits,
                                            cfo, cfi, cff, opm, roce, payout, mcap_cr, price_scr
    data/us_fundamentals_quarterly.csv.gz   quarters: sales, op, interest, np, opm (Q4 = year - Q1..Q3)
    data/us_monthly_prices.csv.gz           6 years of month-end closes (Yahoo)
    data/us_fundamentals_cache.csv          ticker, name, sector, sector_granular, pe, pb, marketCap

Money is stored in units of USD 10 million (a "crore of dollars") so the engine's crore arithmetic
holds; per-share figures and prices are plain dollars.

SEC requires a User-Agent naming the requester with a contact address. Set the repository secret
SEC_USER_AGENT to "Your Name you@example.com"; without it this script stops.

    python us_fundamentals.py [--limit N] [--tickers AAPL,MSFT]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

UNIT = 1e7                       # USD per stored unit
UNIVERSE = "data/sp500_tickers.csv"
OUT_A, OUT_Q = "data/us_fundamentals_annual.csv.gz", "data/us_fundamentals_quarterly.csv.gz"
OUT_M, OUT_F = "data/us_monthly_prices.csv.gz", "data/us_fundamentals_cache.csv"

# first tag with a value for a period wins (companies switch tags over the years)
FLOW = {
    "sales": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
              "RevenueFromContractWithCustomerIncludingAssessedTax", "RevenuesNetOfInterestExpense",
              "SalesRevenueGoodsNet", "InterestAndDividendIncomeOperating"],
    "np": ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss"],
    "op": ["OperatingIncomeLoss"],
    "pbt": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic"],
    "interest": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt", "InterestAndDebtExpense"],
    "dep": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "DepreciationAmortizationAndAccretionNet",
            "Depreciation"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "cfi": ["NetCashProvidedByUsedInInvestingActivities", "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations"],
    "cff": ["NetCashProvidedByUsedInFinancingActivities", "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations"],
    "div": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends", "DividendsCommonStockCash"],
}
STOCK = {
    "reserves": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "ltd": ["LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
    "ltd_nc": ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligationsNoncurrent"],
    "ltd_c": ["LongTermDebtCurrent", "DebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"],
    "stb": ["ShortTermBorrowings", "CommercialPaper"],
    "fixed_assets": ["PropertyPlantAndEquipmentNet",
                     "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization"],
    "total_liab": ["Liabilities"],
    "deposits": ["Deposits"],
}


def _rows(facts: dict, tags: list, unit: str = "USD") -> pd.DataFrame:
    """All filed values for a list of tags, the first tag winning per (start, end)."""
    out = []
    for rank, tag in enumerate(tags):
        for v in (facts.get(tag) or {}).get("units", {}).get(unit, []):
            out.append(dict(rank=rank, start=v.get("start"), end=v.get("end"), val=v.get("val"),
                            form=str(v.get("form", "")), filed=v.get("filed")))
    d = pd.DataFrame(out)
    if d.empty:
        return d
    d["end"] = pd.to_datetime(d["end"], errors="coerce")
    d["start"] = pd.to_datetime(d["start"], errors="coerce") if "start" in d else pd.NaT
    d["filed"] = pd.to_datetime(d["filed"], errors="coerce")
    d = d.dropna(subset=["end", "val"])
    d = d[d.form.str.match(r"^(10-K|10-Q|20-F|40-F|10-KT)")]
    # latest filing of the best-ranked tag (restatements replace earlier figures)
    d = d.sort_values(["rank", "filed"], ascending=[True, False])
    return d


def _flows(facts, tags):
    """-> (annual Series by FY end, quarterly Series by quarter end) for a duration item."""
    d = _rows(facts, tags)
    if d.empty or "start" not in d:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    d = d.dropna(subset=["start"])
    days = (d.end - d.start).dt.days
    a = d[days.between(350, 380)].drop_duplicates("end").set_index("end").val.sort_index()
    q = d[days.between(80, 100)].drop_duplicates("end").set_index("end").val.sort_index()
    a = a[~a.index.duplicated()].astype(float)
    q = q[~q.index.duplicated()].astype(float)
    # Q4 is not filed on its own: the year less the three quarters inside it
    for end, v in a.items():
        inside = q[(q.index > end - pd.Timedelta(days=300)) & (q.index < end - pd.Timedelta(days=20))]
        if len(inside) == 3 and not any(abs((x - end).days) < 20 for x in q.index):
            q.loc[end] = v - inside.sum()
    return a, q.sort_index()


def _stock(facts, tags):
    d = _rows(facts, tags)
    if d.empty:
        return pd.Series(dtype=float)
    s = d.drop_duplicates("end").set_index("end").val.astype(float).sort_index()
    return s[~s.index.duplicated()]


def _shares(facts_all: dict) -> float:
    """Latest shares outstanding (sum of classes on the cover page), else latest diluted weighted average."""
    dei = (facts_all.get("dei") or {}).get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares", [])
    d = pd.DataFrame(dei)
    if len(d):
        d["end"] = pd.to_datetime(d["end"], errors="coerce")
        last = d[d.end == d.end.max()]
        last = last.drop_duplicates("val") if "frame" not in last else last.drop_duplicates(["val", "frame"])
        if last.val.sum() > 0:
            return float(last.val.sum())
    w = _rows(facts_all.get("us-gaap", {}), ["WeightedAverageNumberOfDilutedSharesOutstanding",
                                              "WeightedAverageNumberOfSharesOutstandingBasic"], unit="shares")
    return float(w.sort_values("end").val.iloc[-1]) if len(w) else np.nan


def _qend(ts: pd.Timestamp) -> pd.Timestamp:
    """Snap a fiscal quarter end to the nearest calendar quarter end (NVIDIA's late-January quarter -> 31 Dec)."""
    c = [pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0) for y in (ts.year - 1, ts.year, ts.year + 1) for m in (3, 6, 9, 12)]
    return min(c, key=lambda x: abs((x - ts).days))


def parse(t: str, facts_all: dict, price: float | None = None) -> tuple[list, list]:
    """One company's companyfacts JSON -> (annual rows, quarterly rows) in the engine's long format."""
    full = facts_all.get("facts", facts_all)
    g = full.get("us-gaap", {})
    A, Q = {}, {}
    for f, tags in FLOW.items():
        A[f], Q[f] = _flows(g, tags)
    S = {f: _stock(g, tags) for f, tags in STOCK.items()}
    fy = A["np"].index
    if len(fy) == 0:
        return [], []
    fy = fy[fy >= fy.max() - pd.DateOffset(years=11)]
    at = lambda s, e: float(s[(s.index > e - pd.Timedelta(days=10)) & (s.index < e + pd.Timedelta(days=10))].iloc[-1]) \
        if len(s) and ((s.index > e - pd.Timedelta(days=10)) & (s.index < e + pd.Timedelta(days=10))).any() else np.nan
    arows = []
    for e in fy:
        v = {f: at(A[f], e) for f in FLOW}
        b = {f: at(S[f], e) for f in STOCK}
        if not np.isfinite(v["op"]) and np.isfinite(v["pbt"]):
            v["op"] = v["pbt"] + (v["interest"] if np.isfinite(v["interest"]) else 0)
        debt = b["ltd"] if np.isfinite(b["ltd"]) else np.nansum([b["ltd_nc"], b["ltd_c"]]) if (
            np.isfinite(b["ltd_nc"]) or np.isfinite(b["ltd_c"])) else np.nan
        if np.isfinite(b["stb"]):
            debt = (debt if np.isfinite(debt) else 0) + b["stb"]
        ebit = v["pbt"] + (v["interest"] if np.isfinite(v["interest"]) else 0) if np.isfinite(v["pbt"]) else v["op"]
        cap = b["reserves"] + (debt if np.isfinite(debt) else 0)
        rec = dict(sales=v["sales"], op=v["op"], interest=v["interest"], dep=v["dep"], pbt=v["pbt"], np=v["np"],
                   cfo=v["cfo"], cfi=v["cfi"], cff=v["cff"], reserves=b["reserves"], borrowings=debt,
                   fixed_assets=b["fixed_assets"], total_liab=b["total_liab"], deposits=b["deposits"])
        rec = {k: x / UNIT for k, x in rec.items() if np.isfinite(x)}
        rec["eq_cap"] = 0.0
        if np.isfinite(v["sales"]) and v["sales"] > 0 and np.isfinite(v["op"]):
            rec["opm"] = 100 * v["op"] / v["sales"]
        if np.isfinite(ebit) and np.isfinite(cap) and cap > 0:
            rec["roce"] = 100 * ebit / cap
        if np.isfinite(v["div"]) and np.isfinite(v["np"]) and v["np"] > 0:
            rec["payout"] = 100 * abs(v["div"]) / v["np"]
        arows += [dict(ticker=t, field=k, period=e.date(), value=x) for k, x in rec.items()]
    sh = _shares(full)
    if price and np.isfinite(sh) and sh > 0:
        today = pd.Timestamp.today().normalize().date()
        arows += [dict(ticker=t, field="mcap_cr", period=today, value=sh * price / UNIT),
                  dict(ticker=t, field="price_scr", period=today, value=float(price))]
    qrows = []
    qe = Q["np"].index
    qe = qe[qe >= qe.max() - pd.DateOffset(years=6)] if len(qe) else qe
    for e in qe:
        v = {f: at(Q[f], e) for f in ("sales", "np", "op", "interest", "pbt")}
        if not np.isfinite(v["op"]) and np.isfinite(v["pbt"]):
            v["op"] = v["pbt"] + (v["interest"] if np.isfinite(v["interest"]) else 0)
        rec = {k: v[k] / UNIT for k in ("sales", "np", "op", "interest") if np.isfinite(v[k])}
        if np.isfinite(v["sales"]) and v["sales"] > 0 and np.isfinite(v["op"]):
            rec["opm"] = 100 * v["op"] / v["sales"]
        qrows += [dict(ticker=t, field=k, period=_qend(e).date(), value=x) for k, x in rec.items()]
    return arows, qrows


def _prices(tickers):
    """Six years of month-end closes and the latest close, from Yahoo."""
    import yfinance as yf
    frames = []
    for i in range(0, len(tickers), 100):
        part = tickers[i:i + 100]
        h = yf.download(part, period="6y", interval="1mo", auto_adjust=False, progress=False, group_by="column",
                        threads=True)
        c = h["Close"] if isinstance(h.columns, pd.MultiIndex) else h[["Close"]].rename(columns={"Close": part[0]})
        frames.append(c)
    M = pd.concat(frames, axis=1)
    M = M.loc[:, ~M.columns.duplicated()]
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tickers", default="")
    a = ap.parse_args()
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in ua:
        sys.exit("SEC_USER_AGENT is not set (or has no contact email). SEC refuses anonymous requests: add the "
                 "repository secret SEC_USER_AGENT = 'Your Name you@example.com'.")
    import requests
    h = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
    U = pd.read_csv(UNIVERSE)
    U["ticker"] = U["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    tickers = [x.strip().upper() for x in a.tickers.split(",") if x.strip()] or list(U.ticker)
    if a.limit:
        tickers = tickers[:a.limit]
    m = requests.get("https://www.sec.gov/files/company_tickers.json", headers=h, timeout=60)
    m.raise_for_status()
    cik = {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in m.json().values()}

    print(f"[prices] {len(tickers)} tickers", flush=True)
    M = _prices(tickers)
    last = M.ffill().iloc[-1]
    ml = M.stack().rename("close").reset_index()
    ml.columns = ["date", "ticker", "close"]
    ml["date"] = pd.to_datetime(ml["date"]).dt.to_period("M").dt.to_timestamp("M").dt.date

    AR, QR, miss = [], [], []
    for i, t in enumerate(tickers):
        c = cik.get(t)
        if c is None:
            miss.append(t)
            continue
        for attempt in range(3):
            try:
                r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{c:010d}.json", headers=h, timeout=60)
                if r.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                    continue
                if r.status_code != 200:
                    miss.append(t)
                    break
                ar, qr = parse(t, r.json(), float(last.get(t)) if pd.notna(last.get(t)) else None)
                AR += ar
                QR += qr
                if not ar:
                    miss.append(t)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"[sec] {t}: {type(e).__name__}", flush=True)
                    miss.append(t)
                time.sleep(2)
        time.sleep(0.12)                     # SEC allows 10 requests a second
        if (i + 1) % 50 == 0:
            print(f"[sec] {i + 1}/{len(tickers)}", flush=True)

    A, Q = pd.DataFrame(AR), pd.DataFrame(QR)
    A["basis"] = "us"
    # cache: name, sector, sub-industry and peer multiples (market cap / TTM profit, market cap / book)
    F = U.set_index("ticker").reindex(tickers)[["name", "sector", "sub_industry"]].rename(columns={"sub_industry": "sector_granular"})
    lastv = A.sort_values("period").groupby(["ticker", "field"]).value.last().unstack()
    qnp = Q[Q.field == "np"].sort_values("period").groupby("ticker").value.apply(lambda s: s.iloc[-4:].sum() if len(s) >= 4 else np.nan)
    mc = lastv.get("mcap_cr")
    F["marketCap"] = (mc * UNIT).reindex(F.index) if mc is not None else np.nan
    F["pe"] = (mc / qnp.reindex(lastv.index)).where(lambda x: x > 0).reindex(F.index) if mc is not None else np.nan
    F["pb"] = (mc / lastv.get("reserves")).where(lambda x: x > 0).reindex(F.index) if mc is not None else np.nan
    F.index.name = "ticker"
    print(f"[done] annual rows {len(A):,} for {A.ticker.nunique()} companies; quarterly {len(Q):,}; "
          f"missing {len(miss)}: {', '.join(miss[:30])}{' …' if len(miss) > 30 else ''}")
    if A.ticker.nunique() < 0.8 * len(tickers):
        sys.exit("Fewer than 80% of companies parsed -- not trusting this run; files left as they were.")
    A.to_csv(OUT_A, index=False)
    Q.to_csv(OUT_Q, index=False)
    ml.to_csv(OUT_M, index=False)
    F.reset_index().to_csv(OUT_F, index=False)


if __name__ == "__main__":
    main()
