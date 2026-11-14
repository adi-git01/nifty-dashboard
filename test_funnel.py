"""
Tests for the industry leadership board, business trends, the funnel and the stock report.

    python -m pytest -q test_funnel.py
"""
import numpy as np
import pandas as pd
import pytest

from utils import fund_trends, funnel, industry_board, stock_report


def _snap():
    from utils.nifty1000_list import SUB_INDUSTRY_MAP
    by = {}
    for t, g in SUB_INDUSTRY_MAP.items():
        by.setdefault(g, []).append(t)
    groups = [g for g, m in by.items() if len(m) >= 6 and g != "Unknown"][:4]
    rows = []
    for k, g in enumerate(groups):
        for j, t in enumerate(by[g][:6]):
            strong = k == 0
            rows.append(dict(ticker=t, name=t, currentPrice=100.0, fiftyDayAverage=90.0 if strong else 110.0,
                             twoHundredDayAverage=80.0 if strong else 120.0, dist_52w=-3.0 if strong else -40.0,
                             comp_rs=10.0 - j if strong else -10.0 - k, rs_3m=8.0 if strong else -8.0 - k,
                             dna_signal="BUY" if strong else "HOLD", fx_action="BUY NOW" if (strong and j == 0) else "WATCH",
                             fx_stop=92.0 if j == 0 else 70.0, fx_t1=116.0, fx_t2=np.nan, simba_stage=2.0 if strong else 4.0,
                             simba_m_since=np.nan, simba_si_since=np.nan))
    return pd.DataFrame(rows), groups


def test_board_ranks_broad_strength_first_and_groups():
    df, groups = _snap()
    B = industry_board.board(df)
    assert B.iloc[0].industry == groups[0] and B.iloc[0].score == 100 and B.iloc[0].status == "UPTREND"
    assert B.iloc[0].group == "LEADING" and B.iloc[0].above_50 == 100 and B.iloc[-1].above_200 == 0
    assert set(B.group) <= set(industry_board.GROUP_ICON) and B.change.isna().all()      # no month-ago snapshot given
    prev = df.assign(fiftyDayAverage=110.0, twoHundredDayAverage=120.0, dist_52w=-40.0, comp_rs=-5.0, rs_3m=-5.0)
    B2 = industry_board.board(df, prev)
    assert B2.change.notna().all()
    v = industry_board.stock_industry_view(B, df.ticker.iloc[0])
    assert v["rank"] == 1 and v["of"] == len(B)


def _files(tmp_path, cfo=(80, 90, 100, 110, 120, 130), opm=(10, 10, 10, 14, 15, 16), deposits=False):
    d = tmp_path / "data"
    d.mkdir()
    a = []
    for i, y in enumerate(range(2021, 2027)):
        p = f"{y}-03-31"
        sales = 1000 * 1.15 ** i
        a += [("X.NS", "sales", p, sales), ("X.NS", "op", p, sales * opm[i] / 100), ("X.NS", "opm", p, opm[i]),
              ("X.NS", "np", p, sales * 0.08), ("X.NS", "roce", p, 15 + 2 * i), ("X.NS", "cfo", p, sales * opm[i] / 100 * cfo[i] / 100),
              ("X.NS", "borrowings", p, 100), ("X.NS", "eq_cap", p, 50), ("X.NS", "reserves", p, 450 + 50 * i),
              ("X.NS", "interest", p, 10), ("X.NS", "fixed_assets", p, 400 + 20 * i), ("X.NS", "dep", p, 30),
              ("X.NS", "inventory_days", p, 90 - 5 * i)]
        if deposits:
            a.append(("X.NS", "deposits", p, 5000))
    pd.DataFrame(a, columns=["ticker", "field", "period", "value"]).to_csv(d / "fundamentals_annual.csv.gz", index=False)
    q = []
    for i, p in enumerate(pd.date_range("2024-06-30", periods=10, freq="QE")):
        q += [("X.NS", "sales", p.strftime("%Y-%m-%d"), 250 * 1.04 ** i), ("X.NS", "np", p.strftime("%Y-%m-%d"), 20 * 1.06 ** i),
              ("X.NS", "promoters", p.strftime("%Y-%m-%d"), 60 + 0.2 * i)]
    pd.DataFrame(q, columns=["ticker", "field", "period", "value"]).to_csv(d / "fundamentals_quarterly.csv.gz", index=False)
    return str(tmp_path)


def test_business_trends_direction_and_tables(tmp_path):
    root = _files(tmp_path)
    a = fund_trends.annual("X.NS", root)
    assert len(a) == 6 and round(a.sales_g.iloc[-1], 0) == 15 and round(a.cash_conv.iloc[-1], 0) == 130
    assert round(a.debt_eq.iloc[-1], 2) == round(100 / 750, 2) and a.inventory_days.iloc[-1] == 65
    r = fund_trends.read("X.NS", root)
    way = {f["metric"]: f["way"] for f in r["findings"]}
    assert way["Operating margin"] == "IMPROVING" and way["ROCE"] == "IMPROVING" and way["Sales growth"] == "STABLE"
    assert way["Inventory days"] == "IMPROVING" and way["Promoter holding"] == "IMPROVING"      # fewer days = better
    assert r["verdict"] == "IMPROVING" and r["latest"]["quarter"] == "Sep 2026" and r["has_wc_days"] and not r["has_q_opm"]
    assert round(r["latest"]["sales_yoy"], 0) == round((1.04 ** 4 - 1) * 100, 0)
    assert "OPM %" in stock_report.annual_view(a).index and "Sales YoY %" in stock_report.quarterly_view(fund_trends.quarterly("X.NS", root)).index


def test_poor_cash_conversion_is_flagged_and_fails_business(tmp_path):
    root = _files(tmp_path, cfo=(90, 80, 60, 40, 30, 20))
    r = fund_trends.read("X.NS", root)
    cc = next(f for f in r["findings"] if f["metric"] == "Cash conversion")
    assert cc["way"] == "WORSENING" and "not turning into cash" in cc["text"]
    b = fund_trends.brief("X.NS", root)
    assert not b["ok"] and any(f.startswith("cash conversion") for f in b["flags"])


def test_lenders_skip_cash_and_debt_tests(tmp_path):
    root = _files(tmp_path, cfo=(5, 5, 5, 5, 5, 5), deposits=True)
    r = fund_trends.read("X.NS", root)
    names = {f["metric"] for f in r["findings"]}
    assert r["financial"] and "Cash conversion" not in names and "Debt / equity" not in names and "ROE" in names
    assert fund_trends.read("NOPE.NS", root)["verdict"] == "NO DATA"


def test_funnel_stages_counts_near_misses_and_wide_stop(monkeypatch):
    df, groups = _snap()
    monkeypatch.setattr(fund_trends, "brief", lambda t, root=".": dict(ok=True, verdict="IMPROVING", flags=[], roce=20.0,
                                                                         quarter="Sep 2026", sales_yoy=12.0, np_yoy=15.0))
    B = industry_board.board(df)
    val = pd.DataFrame({"status": ["OK"], "cmp_pct": [95.0], "p50": [80.0], "rating": ["REDUCE"]}, index=[df.ticker.iloc[0]])
    T = funnel.table(df, B, val)
    lead = T[T.industry == groups[0]]
    assert lead.s_industry.all() and lead.s_trend.all() and not T[T.industry != groups[0]].s_trend.any()
    assert lead.s_setup.sum() == 1 and round(lead.risk_pct.iloc[0], 1) == 8.0          # only the BUY NOW stock, stop 8% away
    assert T.s_business.isna().sum() == len(T) - len(lead)                             # business only evaluated after TREND
    R = funnel.run(T)
    assert R["counts"].left.tolist() == [len(T), 6, 6, 6, 1] and R["picks"].ticker.tolist() == [df.ticker.iloc[0]]
    assert len(R["near"]) == 5 and set(R["near"].missed) == {"🎯 Setup"}
    assert len(funnel.run(T, funnel.DEFAULT_ON + ("value",))["picks"]) == 0            # 95th percentile fails VALUE
    assert T[T.ticker != df.ticker.iloc[0]].s_value.all()                              # unvalued stocks pass as n/a
    wide = df.assign(fx_stop=70.0)                                                     # 30% stop: signal but not tradable
    Tw = funnel.table(wide, B, None)
    assert Tw.signal.sum() == 1 and Tw.s_setup.sum() == 0


def test_stock_report_verdicts_and_markdown(tmp_path, monkeypatch):
    df, groups = _snap()
    root = _files(tmp_path)
    for fn in ("read", "annual", "quarterly"):
        monkeypatch.setattr(fund_trends, fn, (lambda f: lambda t, r=".": f("X.NS", root))(getattr(fund_trends, fn)))
    monkeypatch.setattr(fund_trends, "brief", lambda t, r=".": dict(ok=True, verdict="IMPROVING", flags=[], roce=20.0,
                                                                      quarter="Sep 2026", sales_yoy=12.0, np_yoy=15.0))
    B = industry_board.board(df)
    R = stock_report.gather(df.ticker.iloc[0], df, B, None, {"regime": "NORMAL BULL", "risk_scale": 6})
    assert R["verdict"]["tag"] == "ACTIONABLE" and round(R["plan"]["risk_pct"], 1) == 8.0 and R["plan"]["t1"] == 116.0
    assert round(R["plan"]["size_pct"], 1) == 12.5                                     # 1% risk / 8% stop
    md = stock_report.markdown(R)
    for part in ("## Verdict: ACTIONABLE", "## Funnel checklist", "## Industry", "## Entry plan", "## Business trends: IMPROVING",
                 "### Quarterly", "## Valuation"):
        assert part in md
    assert stock_report.gather(df.ticker.iloc[1], df, B)["verdict"]["tag"] == "WAIT FOR A TIGHTER ENTRY" or \
        stock_report.gather(df.ticker.iloc[1], df, B)["verdict"]["tag"] == "WATCHLIST"
    assert stock_report.gather(df.ticker.iloc[-1], df, B)["verdict"]["tag"] == "NOT NOW"
    assert not stock_report.gather("NOPE.NS", df)["ok"]


def test_scraper_collects_working_capital_days_and_quarterly_margin():
    import glob
    import fundamentals_annual as fa
    files = sorted(glob.glob("analysis/screener_samples/*.html"))
    if not files:
        pytest.skip("no saved Screener sample")
    html = open(files[0]).read()
    a, q = {k[0] for k in fa.parse_annual(html)}, {k[0] for k in fa.parse_quarterly(html)}
    assert {"debtor_days", "inventory_days", "ccc_days", "wc_days", "roce", "sales"} <= a
    assert {"opm", "op", "sales", "np", "promoters"} <= q


def test_signal_scorecard_measures_excess_and_verdicts(tmp_path):
    from utils import signal_scorecard as sc
    (tmp_path / "data" / "cache").mkdir(parents=True)
    days = pd.bdate_range("2026-01-05", periods=120)
    tick = [f"S{i}.NS" for i in range(40)]
    for k, d in enumerate(days):                       # S0-S9 rise 0.5% a day, the rest are flat
        px = [100 * 1.005 ** k if i < 10 else 100 + 0.01 * ((k + i) % 3) for i in range(40)]
        pd.DataFrame({"ticker": tick, "currentPrice": px, "dna_signal": ["BUY" if i < 20 else "HOLD" for i in range(40)],
                      "trend_score": 50, "volume_signal_score": 5}).to_parquet(
            tmp_path / "data" / "cache" / f"market_master_{d:%Y_%m_%d}.parquet")
    sig = [(d.strftime("%Y-%m-%d"), t) for d in days[:60:1] for t in tick[:4]]           # winners
    pd.DataFrame(sig, columns=["signal_date", "ticker"]).to_csv(tmp_path / "data" / "tc_log.csv", index=False)
    los = [(d.strftime("%Y-%m-%d"), t) for d in days[:60] for t in tick[30:34]]
    pd.DataFrame(los, columns=["signal_date", "ticker"]).to_csv(tmp_path / "data" / "rs_divergence_log.csv", index=False)
    S = sc.build(str(tmp_path), save=True).set_index("signal")
    tc = S.loc["Turnaround Catalyst"]
    assert tc.signals == 16 and round(tc.avg_21d, 1) == round((1.005 ** 21 - 1) * 100, 1)   # a re-signal within 21 days does not count again
    assert S.loc[sc.YARDSTICK].verdict == "YARDSTICK" and S.loc[sc.YARDSTICK].avg_21d > 0
    assert tc.verdict == "TOO NEW" and abs(S.loc["RS Divergence"].avg_21d) < 0.1
    assert (tmp_path / "data" / "signal_scorecard.csv").exists()
    r = pd.Series(dict(n_21d=500, n_63d=300, avg_21d=-1.0, avg_63d=-2.0, hit_21d=40))
    base = pd.Series(dict(avg_21d=1.0, avg_63d=4.0))
    assert sc._verdict(r, base) == "LAGS MARKET"
    assert sc._verdict(r.replace({-1.0: 2.5, -2.0: 6.0, 40: 55}), base) == "BEATS MOMENTUM"
    assert sc._verdict(r.replace({-1.0: 0.9, -2.0: 3.9}), base) == "LIKE MOMENTUM"
    assert sc._verdict(r.replace({-1.0: 0.2, -2.0: 3.0}), base) == "NO EDGE"


def test_book_diff_status_and_funnel_label():
    from utils import book_view as bv
    known = {"CUMMINSIND.NS", "FEDERALBNK.NS", "SBIN.NS"}
    assert bv.norm("cumminsind", known) == "CUMMINSIND.NS" and bv.norm("FEDERALBANK", known) == "FEDERALBNK.NS"
    assert bv.norm("", known) is None and bv.norm(None, known) is None
    before = pd.DataFrame({"Stock": ["CUMMINSIND", "FEDERALBANK"], "Entry": [4000.0, 270.0], "Stop": [3850.0, 230.0], "Notes": ["", ""]})
    after = pd.DataFrame({"Stock": ["CUMMINSIND", "SBIN", "NOSUCH", None], "Entry": [4000.0, 1000.0, 5.0, None],
                          "Stop": [3900.0, 950.0, None, None], "Notes": ["", "new", "", ""]})
    d = bv.diff(before, after, known, ["Stock", "Entry", "Stop", "Notes"])
    assert [t for t, _ in d["add"]] == ["SBIN.NS"] and [t for t, _ in d["update"]] == ["CUMMINSIND.NS"]
    assert d["remove"] == ["FEDERALBNK.NS"] and [t for t, _ in d["unknown"]] == ["NOSUCH.NS"]
    same = bv.diff(before, before.copy(), known, ["Stock", "Entry", "Stop", "Notes"])
    assert not same["add"] and not same["update"] and not same["remove"]
    assert bv.status(100, 110, None, "BUY", False, 0) == "🔴 stop hit" and bv.status(120, 100, 115, "BUY", False, 0) == "🎯 target hit"
    assert bv.status(102, 100, 150, "BUY", False, 0).startswith("⚠️") and bv.status(120, 100, 150, "HOLD", True, 0) == "🟣 trend weakening"
    assert bv.status(120, 100, 150, "BUY", False, 0) == "✅ on track" and bv.status(None, 100, 150, "BUY", False, 0) == "❔ no price"
    r = pd.Series(dict(s_industry=True, s_trend=True, s_business=True, s_setup=False))
    assert bv.funnel_label(r) == "Missing: entry" and bv.funnel_label(r.replace({False: True})) == "✅ Buyable now"
    assert bv.funnel_label(None) == "—"


def test_strong_names_group_and_label(tmp_path, monkeypatch):
    from utils import strong_names as sn
    rs = pd.DataFrame({"signal_date": ["2026-11-02", "2026-11-05", "2026-11-05", "2026-09-01"], "ticker": ["A.NS", "A.NS", "B.NS", "C.NS"],
                       "name": ["A", "A", "B", "C"], "signal_price": [100.0, 104.0, 50.0, 10.0], "delta_rs": [1.0, 3.0, 2.0, 5.0]})
    es = pd.DataFrame({"signal_date": ["2026-11-03", "2026-11-06", "2026-11-04"], "ticker": ["A.NS", "A.NS", "B.NS"], "name": ["A", "A", "B"],
                       "signal_price": [100.0, 108.0, 50.0], "jump_pct": [6.0, 5.0, 9.0], "vol_mult": [3.0, 2.6, 4.0]})
    rs.to_csv(tmp_path / "rs.csv", index=False), es.to_csv(tmp_path / "es.csv", index=False)
    monkeypatch.setattr(sn, "RS_LOG", str(tmp_path / "rs.csv")), monkeypatch.setattr(sn, "ES_LOG", str(tmp_path / "es.csv"))
    T = pd.DataFrame({"ticker": ["A.NS", "B.NS"], "industry": ["X", "Y"], "ind_group": ["LEADING", "LAGGING"], "scanner": ["BUY", "HOLD"],
                      "technicals": ["BUY NOW", "WATCH"], "comp_rs": [5.0, -2.0], "dist_52w": [-2.0, -30.0], "price": [110.0, 45.0],
                      "s_industry": [True, False], "s_trend": [True, False], "s_business": [True, np.nan], "s_setup": [True, False],
                      "risk_pct": [8.0, np.nan]})
    R = sn.rs_divergence(T, 30).set_index("ticker")
    assert list(R.index) == ["A.NS", "B.NS"] and R.at["A.NS", "red_days"] == 2 and round(R.at["A.NS", "since_first"], 0) == 10
    assert R.at["A.NS", "funnel"] == "✅ Buyable" and R.at["B.NS", "funnel"] == "— Not in uptrend" and R.today.all()
    E = sn.earnings_drift(T, 45).set_index("ticker")
    assert E.at["A.NS", "jump_pct"] == 6.0 and E.at["A.NS", "holds"] and not E.at["B.NS", "holds"]      # first jump in the window
