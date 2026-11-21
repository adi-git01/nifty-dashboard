"""US valuation from SEC EDGAR companyfacts (synthetic filings; no network)."""
import os

import numpy as np
import pandas as pd

import us_fundamentals as uf
from utils.valuation import load_context, value_stock


def _facts(rev0, margin, growth, shares, fy_month=12, seed=0):
    """companyfacts-shaped JSON: 8 fiscal years of 10-K figures plus Q1-Q3 10-Q figures each year."""
    rng = np.random.default_rng(seed)
    g = {k: {"units": {"USD": []}} for k in ("Revenues", "NetIncomeLoss", "OperatingIncomeLoss", "InterestExpense",
                                             "StockholdersEquity", "LongTermDebt", "PropertyPlantAndEquipmentNet",
                                             "NetCashProvidedByUsedInOperatingActivities",
                                             "NetCashProvidedByUsedInInvestingActivities", "PaymentsOfDividendsCommonStock",
                                             "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest")}
    add = lambda tag, start, end, val, form: g[tag]["units"]["USD"].append(
        dict(start=str(start.date()) if start is not None else None, end=str(end.date()), val=float(val), form=form,
             filed=str((end + pd.Timedelta(days=40)).date()), fy=end.year, fp="FY" if form == "10-K" else "Q"))
    today = pd.Timestamp.today()
    last_fy = pd.Timestamp(today.year, fy_month, 1) + pd.offsets.MonthEnd(0)
    if last_fy > today:
        last_fy = pd.Timestamp(today.year - 1, fy_month, 1) + pd.offsets.MonthEnd(0)
    for k in range(9):                                   # 8 filed years + the current, unfinished one
        end = last_fy - pd.DateOffset(years=7 - k) + pd.offsets.MonthEnd(0)
        start = end - pd.DateOffset(years=1) + pd.Timedelta(days=1)
        rev = rev0 * (1 + growth) ** k * (1 + rng.normal(0, 0.02))
        ni = rev * margin
        if k == 8:
            for qn in range(3):
                qs = start + pd.DateOffset(months=3 * qn)
                qe = qs + pd.DateOffset(months=3) - pd.Timedelta(days=1)
                if qe < today - pd.Timedelta(days=30):
                    add("NetIncomeLoss", qs, qe, ni / 4, "10-Q")
                    add("Revenues", qs, qe, rev / 4, "10-Q")
            break
        for tag, v in (("Revenues", rev), ("NetIncomeLoss", ni), ("OperatingIncomeLoss", ni * 1.35),
                       ("InterestExpense", rev * 0.01), ("NetCashProvidedByUsedInOperatingActivities", ni * 1.2),
                       ("NetCashProvidedByUsedInInvestingActivities", -ni * 0.4), ("PaymentsOfDividendsCommonStock", ni * 0.3),
                       ("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest", ni * 1.25)):
            add(tag, start, end, v, "10-K")
        for tag, v in (("StockholdersEquity", rev * 0.8), ("LongTermDebt", rev * 0.2), ("PropertyPlantAndEquipmentNet", rev * 0.3)):
            add(tag, None, end, v, "10-K")
        for qn in range(3):                              # Q1-Q3; Q4 must be derived
            qs = start + pd.DateOffset(months=3 * qn)
            qe = qs + pd.DateOffset(months=3) - pd.Timedelta(days=1)
            for tag, v in (("Revenues", rev / 4), ("NetIncomeLoss", ni / 4), ("OperatingIncomeLoss", ni * 1.35 / 4)):
                add(tag, qs, qe, v, "10-Q")
    dei = {"EntityCommonStockSharesOutstanding": {"units": {"shares": [dict(end=str(last_fy.date()), val=shares)]}}}
    return {"cik": 1, "entityName": "X", "facts": {"us-gaap": g, "dei": dei}}


def test_parse_q4_and_units():
    f = _facts(1e10, 0.1, 0.08, 1e9, fy_month=9)
    A, Q = uf.parse("TEST", f, price=50.0)
    A, Q = pd.DataFrame(A), pd.DataFrame(Q)
    s = A[A.field == "sales"].sort_values("period")
    assert len(s) == 8 and s.value.iloc[0] > 900          # USD 10 million units: ~$10 bn -> ~1,000
    q = Q[Q.field == "np"].sort_values("period")
    yr = q[pd.to_datetime(q.period).dt.year == 2023]
    assert len(yr) == 4                                    # 3 filed + 1 derived quarter a year
    assert len(q) >= 22
    assert set(pd.to_datetime(q.period).dt.month) <= {3, 6, 9, 12}
    fy = A[A.field == "np"].sort_values("period")
    end = pd.Timestamp(fy.period.iloc[-1])
    inq = q[(pd.to_datetime(q.period) > end - pd.Timedelta(days=360)) & (pd.to_datetime(q.period) <= end)]
    assert len(inq) == 4 and abs(inq.value.sum() - fy.value.iloc[-1]) < 1e-6
    mc = A[A.field == "mcap_cr"].value.iloc[0]
    assert abs(mc - 1e9 * 50 / 1e7) < 1e-6
    assert {"opm", "roce", "payout", "reserves", "borrowings", "eq_cap"} <= set(A.field)


def test_value_us_stock(tmp_path):
    os.makedirs(tmp_path / "data")
    AR, QR, M, F = [], [], [], []
    idx = pd.date_range(end=pd.Timestamp.today(), periods=72, freq="ME")
    for i in range(12):
        t = f"T{i:02d}"
        eps = 1e10 * (1 + 0.06 * i / 12) * 0.12 / 1e9
        px = 22 * eps * (1 + 0.03 * np.sin(np.arange(72) / 5 + i))
        a, q = uf.parse(t, _facts(1e10 * (1 + 0.06 * i / 12), 0.12, 0.07 + 0.005 * i, 1e9, seed=i), price=float(px[-1]))
        AR += a; QR += q
        M += [dict(date=d.date(), ticker=t, close=c) for d, c in zip(idx, px)]
        F.append(dict(ticker=t, name=f"Test {i}", sector="Industrials", sector_granular="Industrial Machinery"))
    A = pd.DataFrame(AR); A["basis"] = "us"
    A.to_csv(tmp_path / "data/us_fundamentals_annual.csv.gz", index=False)
    pd.DataFrame(QR).to_csv(tmp_path / "data/us_fundamentals_quarterly.csv.gz", index=False)
    pd.DataFrame(M).to_csv(tmp_path / "data/us_monthly_prices.csv.gz", index=False)
    F = pd.DataFrame(F)
    F["marketCap"], F["pe"], F["pb"] = 1e11, 22.0, 3.0
    F.to_csv(tmp_path / "data/us_fundamentals_cache.csv", index=False)
    c = load_context(str(tmp_path), market="US")
    assert c.market == "US" and len(c.features) == 12
    px = float(c.monthly["T05"].iloc[-1])
    res = value_stock("T05", c, price=px)
    assert res["status"] == "OK", res
    assert res["ticker"] == "T05" and res["identity"] == "INDUSTRIAL"
    assert 0.07 <= res["hurdle"] <= 0.11                  # US hurdle, not India's 12%
    assert 0.3 * px < res["dist"]["P50"] < 3 * px
    assert res["peers"] and all(p.startswith("T") for p in res["peers"])
    assert value_stock("T05", c, price=px * 0.5)["dist"]["cmp_pct"] < res["dist"]["cmp_pct"]


def test_us_out_of_scope_identities(tmp_path):
    from utils import valuation as v
    assert "Regional Banks" in v.BFSI_GRAN and "Property & Casualty Insurance" in v.OUT_GRAN
    assert "Oil & Gas Exploration & Production" in v.COMMODITY_GRAN


def test_main_offline(tmp_path, monkeypatch):
    """The whole fetch with SEC and Yahoo mocked: files written, units right, refuses without a contact."""
    import sys
    import requests
    os.makedirs(tmp_path / "data")
    pd.DataFrame(dict(ticker=["AAA", "BRK.B"], name=["A Co", "Berkshire"], sector=["Industrials", "Financials"],
                      sub_industry=["Industrial Machinery", "Multi-Sector Holdings"])).to_csv(tmp_path / "data/sp500_tickers.csv", index=False)
    monkeypatch.chdir(tmp_path)

    class R:
        def __init__(self, js, code=200):
            self._js, self.status_code = js, code
        def json(self):
            return self._js
        def raise_for_status(self):
            pass

    seen = []

    def get(url, headers=None, timeout=None):
        seen.append(headers["User-Agent"])
        if url.endswith("company_tickers.json"):
            return R({"0": {"cik_str": 1, "ticker": "AAA", "title": "A"}, "1": {"cik_str": 2, "ticker": "BRK-B", "title": "B"}})
        return R(_facts(1e10, 0.1, 0.05, 1e9, seed=int(url[-6])))

    monkeypatch.setattr(requests, "get", get)
    idx = pd.date_range(end=pd.Timestamp.today(), periods=70, freq="ME")
    monkeypatch.setattr(uf, "_prices", lambda t: pd.DataFrame({x: np.linspace(20, 30, 70) for x in t}, index=idx))
    monkeypatch.setattr(uf.time, "sleep", lambda s: None)
    monkeypatch.setenv("SEC_USER_AGENT", "")
    monkeypatch.setattr(sys, "argv", ["us_fundamentals.py"])
    try:
        uf.main()
        assert False, "must refuse without a contact"
    except SystemExit as e:
        assert "SEC_USER_AGENT" in str(e)
    monkeypatch.setenv("SEC_USER_AGENT", "Tester t@example.com")
    uf.main()
    assert seen and all(u == "Tester t@example.com" for u in seen)
    A = pd.read_csv("data/us_fundamentals_annual.csv.gz")
    F = pd.read_csv("data/us_fundamentals_cache.csv")
    assert set(A.ticker) == {"AAA", "BRK-B"} and set(F.ticker) == {"AAA", "BRK-B"}
    assert abs(A[(A.ticker == "AAA") & (A.field == "mcap_cr")].value.iloc[0] - 1e9 * 30 / 1e7) < 1e-6
    assert F.set_index("ticker").at["AAA", "pe"] > 0
    c = load_context(".", market="US")
    assert value_stock("BRK.B", c, price=30.0)["status"] == "OUT OF SCOPE"     # holding company


def test_audit_regressions(tmp_path):
    """PR 126-137 audit: hyphenated symbols in Dhan ids; a stock with no sub-industry must not crash valuation."""
    from utils.dhan_orders import correlation_id
    assert correlation_id("2026-11-13-BAJAJ-AUTO-BUY") == "20261113-BAJAJAUTO-B"
    assert correlation_id("2026-11-13-BAJAJ-BUY") != correlation_id("2026-11-13-BAJAJ-AUTO-BUY")
    assert len(correlation_id("2026-11-13-VERYLONGSYMBOLNAME-SELL")) <= 25
    test_value_us_stock(tmp_path)                                  # builds the synthetic US files
    F = pd.read_csv(tmp_path / "data/us_fundamentals_cache.csv")
    F.loc[F.ticker == "T05", "sector_granular"] = np.nan
    F.to_csv(tmp_path / "data/us_fundamentals_cache.csv", index=False)
    c = load_context(str(tmp_path), market="US")
    assert value_stock("T05", c, price=float(c.monthly["T05"].iloc[-1]), n=500)["status"] in ("OK", "NO DATA", "OUT OF SCOPE")
