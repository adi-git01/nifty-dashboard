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
    way = {f["metric"]: f["way"] for f in r["findings"] + r["ownership"]}
    assert way["Operating margin"] == "IMPROVING" and way["ROCE"] == "IMPROVING" and way["Sales growth"] == "STABLE"
    assert way["Inventory days"] == "IMPROVING" and way["Promoter holding"] == "IMPROVING"      # fewer days = better
    assert set(r["pillars"]) == set(fund_trends.WEIGHTS) and r["verdict"] in ("STRONG", "HEALTHY") and r["score"] >= 60
    assert r["direction"] == "IMPROVING" and r["investing"] is None and not r["hard_flags"]
    assert all(not f["scored"] for f in r["ownership"]) and fund_trends.headline(r).startswith(r["verdict"].title())
    assert r["latest"]["quarter"] == "Sep 2026" and r["has_wc_days"] and not r["has_q_opm"]
    assert round(r["latest"]["sales_yoy"], 0) == round((1.04 ** 4 - 1) * 100, 0)
    assert "OPM %" in stock_report.annual_view(a).index and "Sales YoY %" in stock_report.quarterly_view(fund_trends.quarterly("X.NS", root)).index


def test_poor_cash_conversion_is_flagged_and_fails_business(tmp_path):
    root = _files(tmp_path, cfo=(90, 80, 60, 40, 30, 20))
    r = fund_trends.read("X.NS", root)
    cc = next(f for f in r["findings"] if f["metric"] == "Cash conversion")
    assert cc["way"] == "WORSENING" and "not turning into cash" in cc["text"]
    b = fund_trends.brief("X.NS", root)
    assert not b["ok"] and any(f.startswith("cash conversion") for f in b["flags"]) and b["score"] is not None


def test_investment_phase_does_not_count_roce_dilution_against_the_business(tmp_path):
    root = _files(tmp_path)
    f = tmp_path / "data" / "fundamentals_annual.csv.gz"
    A = pd.read_csv(f)
    late = A.period >= "2025-03-31"
    A.loc[late & (A.field == "borrowings"), "value"] = 550.0            # an acquisition: capital employed up about 70%
    A.loc[late & (A.field == "fixed_assets"), "value"] = 1400.0
    A.loc[A.field == "roce", "value"] = A.loc[A.field == "roce", "period"].map(
        dict(zip(sorted(A.period.unique()), [30, 30, 30, 28, 16, 14]))).values
    A.to_csv(f, index=False)
    fund_trends._load.cache_clear()
    r = fund_trends.read("X.NS", root)
    way = {x["metric"]: x["way"] for x in r["findings"]}
    assert r["investing"] and "FY25" in r["investing"]["text"] and way["ROCE"] == "INVESTING" and way["Investment phase"] == "INVESTING"
    assert way["Fixed-asset turnover"] == "INVESTING" and r["down"] == 0 and r["verdict"] in ("STRONG", "HEALTHY")
    assert r["pillars"]["Returns on capital"]["dir"] == 0 and "investment phase" in fund_trends.headline(r)
    assert fund_trends.brief("X.NS", root)["ok"]


def test_lenders_skip_cash_and_debt_tests(tmp_path):
    root = _files(tmp_path, cfo=(5, 5, 5, 5, 5, 5), deposits=True)
    r = fund_trends.read("X.NS", root)
    names = {f["metric"] for f in r["findings"]}
    assert r["financial"] and r["lender_kind"] == "BANK" and "Cash conversion" not in names and "Debt / equity" not in names
    assert "Cash & balance sheet" not in r["pillars"] and not r["hard_flags"]
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
        monkeypatch.setattr(fund_trends, fn, (lambda f: lambda t, r=".", as_of=None: f("X.NS", root))(getattr(fund_trends, fn)))
    monkeypatch.setattr(fund_trends, "brief", lambda t, r=".": dict(ok=True, verdict="IMPROVING", flags=[], roce=20.0,
                                                                      quarter="Sep 2026", sales_yoy=12.0, np_yoy=15.0))
    B = industry_board.board(df)
    R = stock_report.gather(df.ticker.iloc[0], df, B, None, {"regime": "NORMAL BULL", "risk_scale": 6})
    assert R["verdict"]["tag"] == "ACTIONABLE" and round(R["plan"]["risk_pct"], 1) == 8.0 and R["plan"]["t1"] == 116.0
    assert round(R["plan"]["size_pct"], 1) == 12.5                                     # 1% risk / 8% stop
    md = stock_report.markdown(R)
    for part in ("## Verdict: ACTIONABLE", "## Funnel checklist", "## Industry", "## Entry plan", "## Business: ", "Pillars: Growth",
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


def test_signals_for_collects_every_log(tmp_path, monkeypatch):
    from utils import strong_names as sn
    pd.DataFrame({"signal_date": ["2026-11-05", "2026-10-01"], "ticker": ["A.NS", "B.NS"], "signal_price": [100.0, 50.0],
                  "return_since_signal": [np.nan, 4.0], "stock_ret_on_day": [1.2, 0.5], "nifty_ret_on_day": [-0.8, -0.6]}).to_csv(tmp_path / "rs.csv", index=False)
    pd.DataFrame({"breakout_date": ["2026-11-10"], "ticker": ["A.NS"], "kind": ["52W breakout"], "vol_x": [3.2], "signal_price": [105.0],
                  "return_pct": [2.0]}).to_csv(tmp_path / "bo.csv", index=False)
    src = [(l, {"💪 RS Divergence": str(tmp_path / "rs.csv"), "🚀 Breakout": str(tmp_path / "bo.csv")}.get(l, str(tmp_path / "none.csv")), *rest)
           for l, _, *rest in sn.SOURCES]
    monkeypatch.setattr(sn, "SOURCES", src)
    S = sn.signals_for("A.NS", {"A.NS": 110.0})
    assert S.signal.tolist() == ["🚀 Breakout", "💪 RS Divergence"] and S.return_since.round(1).tolist() == [4.8, 10.0]      # to the latest close
    assert "52W breakout" in S.detail.iloc[0] and "v Nifty -0.8%" in S.detail.iloc[1]
    assert len(sn.signals_for(["A.NS", "B.NS"])) == 3 and sn.signals_for("Z.NS").empty


def test_sector_history_has_a_benchmark_the_chart_accepts():
    from utils import trend_engine as te
    from utils.visuals import chart_sector_rotation
    idx = pd.bdate_range("2026-01-05", periods=40)
    P = pd.DataFrame({f"A{i}.NS": 100 * 1.01 ** np.arange(40) for i in range(3)} | {f"B{i}.NS": 100.0 for i in range(3)} | {"C0.NS": 50.0},
                     index=idx)
    smap = {**{f"A{i}.NS": "Up" for i in range(3)}, **{f"B{i}.NS": "Flat" for i in range(3)}, "C0.NS": "Tiny"}
    H = te.sector_indices(P, smap)
    assert set(H.columns) == {"Up", "Flat", te.BENCH} and H["Up"].iloc[-1] > H[te.BENCH].iloc[-1] > H["Flat"].iloc[-1] == 100
    assert len(chart_sector_rotation(H).data) == 2                                        # benchmark is the baseline, not a line
    assert len(chart_sector_rotation(H.rename(columns={te.BENCH: "Nifty 500 (Eq Wt)"})).data) == 2   # the old name still works
    assert chart_sector_rotation(H.drop(columns=[te.BENCH])) is None                      # no benchmark: no crash


def test_funnel_log_adds_once_and_reenters(tmp_path, monkeypatch):
    from utils import funnel_log
    df, groups = _snap()
    monkeypatch.setattr(fund_trends, "brief", lambda t, root=".": dict(ok=True, verdict="STRONG", score=75, headline="Strong 75/100",
                                                                         flags=[], roce=20.0, quarter="Sep 2026", sales_yoy=12.0, np_yoy=15.0))
    monkeypatch.setattr(funnel_log, "LOG_FILE", str(tmp_path / "fl.csv"))
    B = industry_board.board(df)
    d1, d2, d3 = pd.Timestamp("2026-11-09"), pd.Timestamp("2026-11-10"), pd.Timestamp("2026-12-10")
    log, new = funnel_log.update(df, d1, B, pd.DataFrame())
    assert new.kind.value_counts().to_dict() == {"NEAR": 5, "PASS": 1} and set(new.missed) == {"", "🎯 Setup"}
    assert new[new.kind == "PASS"].ticker.tolist() == [df.ticker.iloc[0]] and log.active.all()
    log, new = funnel_log.update(df, d2, B, pd.DataFrame())
    assert len(new) == 0 and len(log) == 6 and (pd.to_datetime(log.last_seen) == d2).all()
    log, new = funnel_log.update(df, d3, B, pd.DataFrame())                 # a month later: fresh rows
    assert len(new) == 6 and len(log) == 12 and int(log.active.sum()) == 6


def test_exit_guide_and_portfolio_risk():
    from utils import book_view as bv
    g = bv.exit_guide(100, 90, 120, 0.12, 80)                               # 12% below a 120 peak = 105.6 > 50-day 90
    assert g["hit"] and round(g["level"], 1) == 105.6 and "peak" in g["text"]
    g = bv.exit_guide(100, 95, 105, 0.12, 80)
    assert not g["hit"] and g["level"] == 95 and g["text"].startswith("Raise stop to ₹95.0")
    assert bv.exit_guide(100, 90, 100, 0.12, 93)["text"].startswith("Stop is at or above") and bv.exit_guide(None, 90, 100, 0.12, 80)["level"] is None
    H = pd.DataFrame({"Stock": ["A", "B", "C"], "Price": [100.0, 200.0, 50.0], "Qty": [10, 5, 20], "Stop": [90.0, None, 45.0],
                      "Exit below": [92.0, 180.0, 46.0], "Sector": ["Banks", "Banks", "IT"], "Daily traded": [1e7, 1e4, 1e7]})
    K = bv.risk_summary(H, capital=10000)
    assert K["invested"] == 3000 and K["risk"] == 100 + 100 + 100 and K["risk_pct"] == 3.0     # B uses its exit level
    assert K["no_stop"] == ["B"] and K["thin"] == ["B"] and K["largest"][0] in ("A", "B", "C")
    assert round(K["by_industry"]["Banks"], 1) == 66.7 and bv.risk_summary(H.assign(Price=np.nan)) == {}


def test_daily_message_leads_with_the_book(monkeypatch):
    from utils import book_view as bv, daily_message
    df, groups = _snap()
    monkeypatch.setattr(fund_trends, "brief", lambda t, root=".": dict(ok=True, verdict="STRONG", score=75, headline="Strong 75/100",
                                                                         flags=[], roce=20.0, quarter="Sep 2026", sales_yoy=12.0, np_yoy=15.0))
    monkeypatch.setattr(bv, "peak_since", lambda t, d, px: px)
    a, b, c = df.ticker.iloc[0], df.ticker.iloc[1], df.ticker.iloc[-1]
    pos = [dict(ticker=a, status="active", entry_price=80.0, stop_loss=95.0, target=150.0),       # on track, stop above exit level
           dict(ticker=c, status="active", entry_price=120.0, stop_loss=90.0, target=None),       # below its 50-day: exit rule
           dict(ticker=a, status="watching"), dict(ticker=b, status="watching")]
    new = pd.DataFrame({"kind": ["PASS", "NEAR"], "ticker": [a, b], "industry": ["X", "X"], "signal_price": [100.0, 100.0], "risk_pct": [8.0, 30.0]})
    msg = daily_message.build(df, pd.Timestamp("2026-11-13"), None, new, [{"Ticker": a, "Jump_Pct": 5.0}, {"Ticker": c, "Jump_Pct": 6.0}],
                              [{"Ticker": c}], 3, pos)
    t = lambda x: x.replace(".NS", "")
    assert "Holdings (2)</b>: 1 need a look" in msg and f"🔴 {t(c)}" in msg and "exit rule hit" in msg
    assert f"{t(a)} now buyable" in msg and "New funnel passes (1)" in msg
    assert f"Earnings jumps in an uptrend (1)</b>: {t(a)}" in msg and "Up on a red day" not in msg     # c is not in an uptrend
    assert "2 earnings jumps, 1 red-day gainers, 3 turnaround catalysts" in msg
    assert msg.index("Holdings") < msg.index("funnel passes") < msg.index("Other scanners")


def test_lender_ceiling_is_not_below_the_banks_own_trough():
    import os
    from utils import valuation as V
    if not os.path.exists("data/fundamentals_annual.csv.gz"):
        pytest.skip("no fundamentals on disk")
    ctx = V.load_context()
    o = V.value_stock("ICICIBANK", ctx, price=1500.0)
    if o.get("status", "OK") != "OK" or o["multiple"]["kind"] != "P/B":
        pytest.skip("ICICIBANK not valued on P/B in this data")
    m = o["multiple"]
    assert m["ceiling"] >= m["own_5y_p10"] - 1e-9 or m["ceiling"] >= 1.6 * m["sector_median"] - 1e-9


def _lender_files(tmp_path, deposits=True, roa=(1.0, 1.1, 1.2, 1.3, 1.4, 1.5), lev=10.0):
    d = tmp_path / "data"
    d.mkdir()
    rows = []
    for i, y in enumerate(range(2021, 2027)):
        p, assets = f"{y}-03-31", 100000.0
        eq = assets / lev
        sales, interest = assets * 0.08, assets * 0.045            # NII 3.5% of assets
        np_ = assets * roa[i] / 100
        pbt = np_ / 0.75
        op = pbt - assets * 0.01                                   # other income 1% of assets, no depreciation
        f = dict(sales=sales, interest=interest, op=op, pbt=pbt, np=np_, total_liab=assets, eq_cap=eq * 0.1, reserves=eq * 0.9,
                 borrowings=assets * 0.2, opm=10.0)
        if deposits:
            f["deposits"] = assets * 0.7
        rows += [("X.NS", k, p, v) for k, v in f.items()]
    pd.DataFrame(rows, columns=["ticker", "field", "period", "value"]).to_csv(d / "fundamentals_annual.csv.gz", index=False)
    q = [("X.NS", f, p.strftime("%Y-%m-%d"), v) for i, p in enumerate(pd.date_range("2024-06-30", periods=10, freq="QE"))
         for f, v in (("sales", 2000 * 1.03 ** i), ("np", 300 * 1.04 ** i), ("nnpa", 1.5 - 0.08 * i))]
    pd.DataFrame(q, columns=["ticker", "field", "period", "value"]).to_csv(d / "fundamentals_quarterly.csv.gz", index=False)
    fund_trends._load.cache_clear()
    return str(tmp_path)


def test_roa_tree_adds_up_and_scores_a_bank(tmp_path):
    root = _lender_files(tmp_path)
    T = fund_trends.roa_tree("X.NS", root)
    L = T.iloc[-1]
    assert T.attrs["kind"] == "BANK" and round(L.nii, 2) == 3.5 and round(L.other_income, 2) == 1.0 and round(L.roa, 2) == 1.5
    assert abs(L.nii + L.other_income - L.costs - L.pretax_roa) < 1e-9 and abs(L.pretax_roa - L.tax - L.roa) < 1e-9     # the tree adds up
    assert round(L.leverage, 1) == 10.0 and round(L.roe, 1) == 15.0 and abs(L.roa * L.leverage - L.roe) < 1e-6
    r = fund_trends.read("X.NS", root)
    assert r["lender_kind"] == "BANK" and set(r["pillars"]) == set(fund_trends.LENDER_WEIGHTS) and r["financial"]
    way = {f["metric"]: f["way"] for f in r["findings"]}
    assert way["ROA"] == "IMPROVING" and way["Net NPA"] == "IMPROVING" and way["Costs / assets"] == "IMPROVING"
    assert "costs" in next(f["text"] for f in r["findings"] if f["metric"] == "What moved ROA in 3 years")
    assert r["verdict"] in ("STRONG", "HEALTHY") and not r["hard_flags"] and fund_trends.brief("X.NS", root)["ok"]
    assert "= ROA %" in stock_report.roa_tree_view(T).index and abs(sum(v["weight"] for v in r["pillars"].values()) - 1) < 1e-9


def test_lender_is_judged_against_its_own_type_and_flags(tmp_path, monkeypatch):
    root = _lender_files(tmp_path, roa=(1.0,) * 6)
    bank = fund_trends.read("X.NS", root)["pillars"]["ROA & ROE"]["score"]
    monkeypatch.setattr(fund_trends, "GOLD_LOAN", {"X.NS"})
    root2 = _lender_files(tmp_path / "g", deposits=False, roa=(1.0,) * 6) if (tmp_path / "g").mkdir() is None else None
    r = fund_trends.read("X.NS", root2)
    assert r["lender_kind"] == "GOLD LOAN" and r["pillars"]["ROA & ROE"]["score"] < bank          # 1% ROA: fine for a bank, poor for gold loans
    assert any(f.startswith("leverage") for f in r["hard_flags"])                                # 10x is too much for a gold-loan company
    monkeypatch.setattr(fund_trends, "GOLD_LOAN", set())
    (tmp_path / "l").mkdir()
    root3 = _lender_files(tmp_path / "l", roa=(1.0, 1.0, 1.0, 0.5, -0.2, -0.8))
    r = fund_trends.read("X.NS", root3)
    assert "loss-making lender" in r["hard_flags"] and not fund_trends.brief("X.NS", root3)["ok"]
    assert fund_trends.roa_tree("NOPE.NS", root3).empty


def test_broker_file_parsing_zerodha_dhan_and_excel():
    import io
    from utils import portfolio_import as pi
    z = b"Instrument,Qty.,Avg. cost,LTP,Cur. val,P&L\nCUMMINSIND,8,4198.5,4369,34952,1364\nFEDERALBNK,100,270.00,360,36000,9000\nGOLDBEES,50,60,70,3500,500\n"
    Z = pi.parse(z, "holdings.csv")
    assert Z.symbol.tolist() == ["CUMMINSIND", "FEDERALBNK", "GOLDBEES"] and Z.qty.tolist() == [8, 100, 50] and Z.avg.iloc[0] == 4198.5
    d = b'Name,Symbol,Quantity,Avg Price,LTP\nState Bank of India,NSE:SBIN-EQ,"1,000","812.40",1080\nCummins India Limited,CUMMINSIND,2,4000,4369\nSold Out Ltd,SOLD,0,10,11\n'
    Dn = pi.parse(d, "dhan.csv")
    assert Dn.symbol.tolist() == ["SBIN", "CUMMINSIND"] and Dn.qty.iloc[0] == 1000 and Dn.avg.iloc[0] == 812.4       # zero-quantity row dropped
    x = pd.DataFrame([["Holdings Statement", None, None, None, None], ["Client ID", "AB1234", None, None, None], [None] * 5,
                      ["Symbol", "ISIN", "Quantity Available", "Quantity Pledged (Margin)", "Average Price"],
                      ["TCS", "INE467B01029", "10", "5", "3500.5"], ["Total", None, "15", None, None]])
    b = io.BytesIO()
    x.to_excel(b, header=False, index=False)
    X = pi.parse(b.getvalue(), "console.xlsx")
    assert X.symbol.tolist() == ["TCS"] and X.qty.iloc[0] == 15                                    # title rows skipped, pledged added, Total dropped
    C = pi.combine([Z, Dn]).set_index("symbol")
    assert C.at["CUMMINSIND", "qty"] == 10 and round(C.at["CUMMINSIND", "avg"], 1) == 4158.8       # two brokers, weighted average
    with pytest.raises(ValueError):
        pi.parse(b"a,b\n1,2\n", "junk.csv")
    assert pi.clean_symbol("NSE:TCS-EQ") == "TCS" and pi.clean_symbol("m&m.ns") == "M&M"


def test_broker_import_reconciles_without_deleting(monkeypatch):
    from utils import portfolio_import as pi, positions as pos
    known = {"CUMMINSIND.NS", "FEDERALBNK.NS", "SBIN.NS", "TCS.NS", "HDFCBANK.NS"}
    names = {"SBIN.NS": "State Bank of India", "TCS.NS": "Tata Consultancy Services Limited"}
    book = [dict(ticker="CUMMINSIND.NS", status="active", quantity=8, entry_price=4198.5, stop_loss=3850),      # unchanged
            dict(ticker="FEDERALBANK.NS", status="active", quantity=50, entry_price=250.0),                   # changed (old symbol spelling)
            dict(ticker="HDFCBANK.NS", status="active", quantity=10, entry_price=800.0),                      # not in the file
            dict(ticker="SBIN.NS", status="watching")]                                                        # watched, now bought
    rows = pd.DataFrame({"symbol": ["CUMMINSIND", "FEDERALBNK", "SBIN", "", "GOLDBEES"],
                         "name": ["", "", "", "Tata Consultancy Services Ltd", ""], "qty": [8, 100, 20, 5, 50],
                         "avg": [4198.5, 270.0, 812.4, 3500.0, 60.0]})
    plan = pi.reconcile(rows, book, known, names, whole=False)
    assert [u["ticker"] for u in plan["unchanged"]] == ["CUMMINSIND.NS"] and plan["close"] == []
    assert [(u["orig"], u["qty"], u["old_qty"]) for u in plan["update"]] == [("FEDERALBANK.NS", 100, 50)]
    assert {a["ticker"]: a["was"] for a in plan["add"]} == {"SBIN.NS": "on the watchlist", "TCS.NS": "not in the book"}   # TCS matched by name
    assert [u["symbol"] for u in plan["unknown"]] == ["GOLDBEES"]
    assert [c["ticker"] for c in pi.reconcile(rows, book, known, names, whole=True)["close"]] == ["HDFCBANK.NS"]
    calls = []
    monkeypatch.setattr(pos, "add_position", lambda t, **k: calls.append(("add", t, k)))
    monkeypatch.setattr(pos, "update_position", lambda t, **k: calls.append(("upd", t, k)))
    monkeypatch.setattr(pos, "close_position", lambda t, px: calls.append(("close", t, px)))
    monkeypatch.setattr(pos, "remove_position", lambda t: calls.append(("REMOVE", t)))
    n = pi.apply(pi.reconcile(rows, book, known, names, whole=True), {"HDFCBANK.NS": 804.0}, names, close=True)
    assert {k: n[k] for k in ("added", "updated", "closed")} == dict(added=2, updated=1, closed=1) and not any(c[0] == "REMOVE" for c in calls)
    upd = {c[1]: c[2] for c in calls if c[0] == "upd"}
    assert upd["FEDERALBANK.NS"]["quantity"] == 100 and "stop_loss" not in upd["FEDERALBANK.NS"]            # stops are never touched
    assert upd["SBIN.NS"]["status"] == "active" and ("close", "HDFCBANK.NS", 804.0) in calls
    assert pi.apply(pi.reconcile(rows, book, known, names, whole=True), {}, names, close=False)["closed"] == 0
    acts = {(e[0], e[1]): e for e in n["events"]}
    assert acts[("BUY", "TCS.NS")][2:] == (5, 3500.0, False) and acts[("SELL", "HDFCBANK.NS")][2:] == (10, 804.0, True)
    add = acts[("ADD", "FEDERALBNK.NS")]
    assert add[2] == 50 and round(add[3], 2) == 290.0 and add[4] is False              # price implied by the change in average cost


def test_sync_detects_sales_per_broker():
    from utils import portfolio_import as pi
    known = {"SBIN.NS", "TCS.NS", "INFY.NS", "ITC.NS"}
    row = lambda sym, q, a: pd.DataFrame({"symbol": sym, "name": [""] * len(sym), "qty": q, "avg": a})
    book = [dict(ticker="SBIN.NS", status="active", quantity=30, entry_price=800.0,                 # held at both brokers
                 lots={"Zerodha": {"qty": 20, "avg": 790.0}, "Dhan": {"qty": 10, "avg": 820.0}}),
            dict(ticker="TCS.NS", status="active", quantity=5, entry_price=3500.0, lots={"Zerodha": {"qty": 5, "avg": 3500.0}}),
            dict(ticker="INFY.NS", status="active", quantity=7, entry_price=1500.0, lots={"Dhan": {"qty": 7, "avg": 1500.0}}),
            dict(ticker="ITC.NS", status="active", quantity=100, entry_price=400.0)]                      # typed in by hand
    # Zerodha now shows 12 SBIN (8 sold) and no TCS (all sold). Dhan was not synced.
    plan = pi.reconcile_sources({"Zerodha": row(["SBIN"], [12], [790.0])}, book, known)
    u = plan["update"][0]
    assert (u["ticker"], u["qty"], u["old_qty"]) == ("SBIN.NS", 22, 30) and u["lots"]["Dhan"]["qty"] == 10   # Dhan's shares untouched
    assert [(c["ticker"], c["auto"]) for c in plan["close"]] == [("TCS.NS", True)]                        # sold: no tick-box needed
    assert not any(x["ticker"] in ("INFY.NS", "ITC.NS") for k in ("update", "close") for x in plan[k])  # other broker / typed in: left alone
    calls = []
    from utils import positions as pos
    import pytest as _pt
    mp = _pt.MonkeyPatch()
    mp.setattr(pos, "update_position", lambda t, **k: calls.append(("upd", t, k)))
    mp.setattr(pos, "close_position", lambda t, px: calls.append(("close", t, px)))
    try:
        n = pi.apply(plan, {"SBIN.NS": 850.0, "TCS.NS": 3900.0}, close=False)
    finally:
        mp.undo()
    assert sorted((e[0], e[1], e[2], e[4]) for e in n["events"]) == [("REDUCE", "SBIN.NS", 8, True), ("SELL", "TCS.NS", 5, True)]
    assert ("close", "TCS.NS", 3900.0) in calls and n["closed"] == 1
    # a broker figure replaces a typed-in one for the same stock rather than adding to it
    p2 = pi.reconcile_sources({"Dhan": row(["ITC"], [100], [400.0])}, book, known)
    assert "ITC.NS" not in [u["ticker"] for u in p2["update"]] and p2["lots_only"][0]["lots"] == {"Dhan": {"qty": 100, "avg": 400.0}}


def test_mcp_insight_tools_return_json_with_dates():
    import json
    from mcp_server import data_access as D
    if not D.snapshot_dates():
        pytest.skip("no snapshots on disk")
    from mcp_server import insight as I
    for fn in (lambda: I.get_funnel(3), lambda: I.get_industry_board("LEADING", 3), lambda: I.get_scorecard(),
               lambda: I.get_business("HDFCBANK"), lambda: I.get_stock_report("NOSUCHSTOCK")):
        out = fn()
        json.dumps(out)
        assert isinstance(out, dict)
    f = I.get_funnel(3)
    assert f["as_of"] == D.snapshot_dates()[-1] and f["counts"][0]["stage"] == "Universe" and "evidence" in f
    assert I.get_business("HDFCBANK")["lender_kind"] == "BANK" and "error" in I.get_stock_report("NOSUCHSTOCK")


def test_remote_mcp_needs_a_long_token_and_hides_behind_it(monkeypatch):
    pytest.importorskip("mcp")
    from starlette.testclient import TestClient
    from mcp_server import remote
    monkeypatch.delenv("MCP_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        remote.token()
    monkeypatch.setenv("MCP_TOKEN", "short")
    with pytest.raises(SystemExit):
        remote.token()
    tok = "test-token-0123456789abcdefgh"
    monkeypatch.setenv("MCP_TOKEN", tok)
    assert remote.token() == tok
    with TestClient(remote.build_app(tok)) as c:
        assert c.get("/").text.startswith("ok")
        h = {"content-type": "application/json", "accept": "application/json, text/event-stream"}
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        assert c.post("/mcp", json=body, headers=h).status_code == 404                      # no token: nothing there
        assert c.post("/wrong-token-0123456789abcdef/mcp", json=body, headers=h).status_code == 404
        r = c.post(f"/{tok}/mcp", json=body, headers=h)
        assert r.status_code == 200 and len(r.json()["result"]["tools"]) == 17


def test_hot_reload_picks_up_a_changed_module(tmp_path, monkeypatch):
    import importlib, os, sys, time
    from utils import hot_reload
    pkg = tmp_path / "hrpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "base.py").write_text("def f():\n    return 1\n")
    (pkg / "user_view.py").write_text("from hrpkg.base import f\n\ndef g():\n    return f() * 10\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    import hrpkg.base, hrpkg.user_view                                         # noqa: E401
    assert hot_reload.refresh(("hrpkg.",)) == [] and hrpkg.user_view.g() == 10   # first sight: nothing stale
    (pkg / "base.py").write_text("def f():\n    return 2\n\ndef new():\n    return 'added'\n")
    future = time.time() + 5
    os.utime(pkg / "base.py", (future, future))
    assert hot_reload.refresh(("hrpkg.",)) == ["hrpkg.base"]
    assert sys.modules["hrpkg.base"].new() == "added" and sys.modules["hrpkg.user_view"].g() == 20   # the dependant sees the new code
    hot_reload._SEEN["hrpkg.base"] = future + 1
    assert hot_reload.refresh(("hrpkg.",)) == []
    for n in ("hrpkg.base", "hrpkg.user_view", "hrpkg"):
        sys.modules.pop(n, None)
        hot_reload._SEEN.pop(n, None)


class _Resp:
    def __init__(self, code, body):
        self.status_code, self._b = code, body

    def json(self):
        if isinstance(self._b, Exception):
            raise self._b
        return self._b


def test_broker_sync_kite_and_dhan_read_holdings_only(monkeypatch):
    import hashlib
    from utils import broker_sync as bs
    calls = []

    class H:
        @staticmethod
        def post(url, **k):
            calls.append(("POST", url, k))
            if url.endswith("/session/token"):
                ok = k["data"]["checksum"] == hashlib.sha256(b"KEYreqSECRET").hexdigest()
                return _Resp(200, {"status": "success", "data": {"access_token": "ATOK"}}) if ok else _Resp(403, {"message": "Invalid `checksum`"})
            if url.endswith("/app/generate-consent"):
                return _Resp(200, {"consentAppId": "CID-1"})
            if url.endswith("/app/consumeApp-consent"):
                return _Resp(200, {"accessToken": "DTOK"})
            return _Resp(404, {})

        @staticmethod
        def get(url, **k):
            calls.append(("GET", url, k))
            if url.endswith("/portfolio/holdings"):
                if k["headers"]["Authorization"] != "token KEY:ATOK":
                    return _Resp(403, {"status": "error", "message": "Incorrect `api_key` or `access_token`."})
                return _Resp(200, {"status": "success", "data": [
                    {"tradingsymbol": "CUMMINSIND", "exchange": "NSE", "quantity": 6, "t1_quantity": 2, "collateral_quantity": 0, "average_price": 4198.5},
                    {"tradingsymbol": "SBIN-BE", "exchange": "BSE", "quantity": 0, "t1_quantity": 0, "collateral_quantity": 10, "average_price": 800},
                    {"tradingsymbol": "NIFTY26DECFUT", "exchange": "NFO", "quantity": 50, "average_price": 1},
                    {"tradingsymbol": "SOLD", "exchange": "NSE", "quantity": 0, "average_price": 5}]})
            if url.endswith("/v2/holdings"):
                if k["headers"]["access-token"] == "EMPTY":
                    return _Resp(500, {"errorCode": "DH-1111", "errorMessage": "No holdings available"})
                if k["headers"]["access-token"] != "DTOK":
                    return _Resp(401, {"errorCode": "DH-901", "errorMessage": "Client ID or user generated access token is invalid or expired."})
                return _Resp(200, [{"exchange": "ALL", "tradingSymbol": "HDFCBANK", "totalQty": 1000, "avgCostPrice": 780.5},
                                   {"exchange": "NSE_EQ", "tradingSymbol": "TCS", "totalQty": 0, "avgCostPrice": 3000}])
            return _Resp(404, {})

    monkeypatch.setattr(bs, "_http", lambda: H)
    assert bs.kite_login_url("KEY") == "https://kite.zerodha.com/connect/login?v=3&api_key=KEY"
    assert bs.kite_session("KEY", "SECRET", "req") == "ATOK"
    with pytest.raises(bs.BrokerError, match="checksum"):
        bs.kite_session("KEY", "WRONG", "req")
    Z = bs.kite_holdings("KEY", "ATOK")
    assert Z.symbol.tolist() == ["CUMMINSIND", "SBIN"] and Z.qty.tolist() == [8, 10] and Z.avg.iloc[0] == 4198.5   # T1 and pledged counted, F&O and zero rows dropped
    with pytest.raises(bs.BrokerError, match="expired"):
        bs.kite_holdings("KEY", "STALE")
    assert bs.dhan_login_url(bs.dhan_consent("1000", "APP", "SEC")).endswith("consentApp-login?consentAppId=CID-1")
    assert bs.dhan_session("tid", "APP", "SEC") == "DTOK"
    Dn = bs.dhan_holdings("DTOK")
    assert Dn.symbol.tolist() == ["HDFCBANK"] and Dn.qty.iloc[0] == 1000 and Dn.avg.iloc[0] == 780.5
    assert bs.dhan_holdings("EMPTY").empty
    with pytest.raises(bs.BrokerError, match="24 hours"):
        bs.dhan_holdings("BAD")
    urls = {u for _, u, _ in calls}
    assert all("order" not in u and "/positions" not in u and "/funds" not in u and "/margins" not in u for u in urls)   # holdings and login only
    assert {u for m, u, _ in calls if m == "GET"} == {"https://api.kite.trade/portfolio/holdings", "https://api.dhan.co/v2/holdings"}
    from utils import portfolio_import as pi
    C = pi.combine([Z, Dn]).set_index("symbol")
    assert set(C.index) == {"CUMMINSIND", "SBIN", "HDFCBANK"}                                # both brokers feed one portfolio
    assert bs.config({"kite": {"api_key": "a", "api_secret": "b"}, "dhan": {"client_id": "1"}}) == {"kite": {"api_key": "a", "api_secret": "b"}}
    assert bs.config({}) == {}
    want = {"dhan": {"client_id": "1000", "app_id": "KEY", "app_secret": "SEC"}}
    assert bs.config({"dhan": {"client_id": 1000, "api_key": " KEY ", "api_secret": "SEC"}}) == want      # Dhan's own names
    assert bs.config({"dhan": {"client_id": "1000", "app_id": "KEY", "app_secret": "SEC"}}) == want


def _fake_bars(n=320, drift=0.0015, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-11-13", periods=n)
    c = pd.Series(100 * np.cumprod(1 + drift + rng.normal(0, 0.01, n)), idx)
    return pd.DataFrame({"Open": c * 0.995, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 2e6}, index=idx)


def test_extended_rows_use_the_nightly_engine_without_saving(tmp_path, monkeypatch):
    import glob
    from utils import extended, fast_data_engine as fde
    stock = _fake_bars()
    multi = pd.concat({"NEWCO.NS": stock}, axis=1)                         # yfinance's [Ticker, Field] shape for a list input
    monkeypatch.setattr(fde, "safe_download", lambda tickers, **k: multi)
    monkeypatch.setattr(fde, "safe_history", lambda sym, **k: _fake_bars(drift=0.0003, seed=2))
    before = sorted(glob.glob("data/cache/market_master_*.parquet"))
    R = extended.rows_for(["newco"])
    assert sorted(glob.glob("data/cache/market_master_*.parquet")) == before          # the day's snapshot is never touched
    assert R.ticker.tolist() == ["NEWCO.NS"] and bool(R.extended.iloc[0])
    r = R.iloc[0]
    assert abs(r.currentPrice - stock.Close.iloc[-1]) < 1e-6 and r.fiftyDayAverage > 0 and r.twoHundredDayAverage > 0
    for c in ("comp_rs", "rs_3m", "dist_52w", "dna_signal", "trend_score", "return_1m"):
        assert c in R.columns and pd.notna(r[c]), c
    base = pd.DataFrame({"ticker": ["A.NS"], "currentPrice": [10.0], "bar_date": ["2026-11-13"]})
    W = extended.with_rows(base, ["NEWCO", "A.NS"])
    assert W.ticker.tolist() == ["A.NS", "NEWCO.NS"] and W.bar_date.tolist() == ["2026-11-13"] * 2
    monkeypatch.setattr(fde, "safe_download", lambda tickers, **k: pd.DataFrame())
    assert extended.rows_for(["NOPE"]).empty and len(extended.with_rows(base, ["NOPE"])) == 1


def test_tracked_extras_and_extended_list():
    from utils import extended
    import build_extended_universe as bx
    core = {"SBIN.NS", "TCS.NS"}
    book = [dict(ticker="SBIN.NS", status="active"), dict(ticker="NEWCO", status="active"), dict(ticker="OLDCO.NS", status="closed"),
            dict(ticker="WATCHME.NS", status="watching"), dict(ticker="bad symbol!", status="active"), dict(ticker="NEWCO.NS", status="watching")]
    import utils.extended as _e
    real = _e.eq_symbols
    _e.eq_symbols = lambda: {"NEWCO.NS": "New Co", "WATCHME.NS": "Watch Me", "SBIN.NS": "SBI", "TCS.NS": "TCS"}
    try:
        assert extended.tracked(core, book) == ["NEWCO.NS", "WATCHME.NS"]              # core, closed, invalid and duplicates dropped
        # a misspelt universe stock is not an extra, and a symbol that is not an NSE equity is never downloaded
        assert extended.tracked({"FEDERALBNK.NS"}, [dict(ticker="FEDERALBANK.NS", status="active"),
                                                    dict(ticker="GOLDBEES.NS", status="active")]) == []
    finally:
        _e.eq_symbols = real
    f = extended.add_tracked(pd.DataFrame({"ticker": ["SBIN.NS", "TCS.NS"], "name": ["S", "T"]}), ["NEWCO.NS", "SBIN.NS"])
    assert f.ticker.tolist() == ["SBIN.NS", "TCS.NS", "NEWCO.NS"] and f.sector_granular.iloc[-1] in ("Unknown", f.sector_granular.iloc[-1])
    A = pd.DataFrame([("BIG.NS", "mcap_cr", "2026-11-10", 5000.0), ("BIG.NS", "price_scr", "2026-11-10", 250.0),
                      ("SMALL.NS", "mcap_cr", "2026-11-10", 400.0), ("HELD.NS", "mcap_cr", "2026-11-10", 300.0),
                      ("BIG.NS", "sales", "2026-03-31", 900.0)], columns=["ticker", "field", "period", "value"])
    U = bx.assemble(A, {"BIG.NS": "Industrial  Products"}, {"BIG.NS": "Big Co Ltd"}, {"HELD.NS"}, pd.DataFrame())
    assert U.Ticker.tolist() == ["BIG.NS", "HELD.NS"]                                    # Rs 1,000 cr and above, plus what you hold
    assert U.iloc[0].to_dict() | {} == dict(Ticker="BIG.NS", Company_Name="Big Co Ltd", Sub_Industry="Industrial Products",
                                              mcap_cr=5000.0, price=250.0, as_of="2026-11-10")
    assert U.Sub_Industry.iloc[1] == "Unknown" and extended.norm(" tatacap ") == "TATACAP.NS"
    from utils.nifty1000_list import EXTENDED, SUB_INDUSTRY_MAP, TICKERS_1000
    assert not (set(EXTENDED) & set(TICKERS_1000)) and all(t in SUB_INDUSTRY_MAP for t in list(EXTENDED)[:20])


def test_trade_journal_folds_events_into_round_trips():
    from utils import trade_journal as tj
    ev = lambda a, t, q, p, d, **k: {**tj.event(a, t, q, p, date=d, **k), "ts": d + "T10:00:00"}
    evs = [ev("BUY", "AAA.NS", 10, 100.0, "2026-01-05", notes="breakout", metrics={"stages_passed": 4, "technicals": "BUY NOW", "comp_rs": 12.0}),
           ev("ADD", "AAA.NS", 10, 120.0, "2026-01-20"),
           ev("REDUCE", "AAA.NS", 5, 130.0, "2026-02-01", estimated=True),
           ev("SELL", "AAA.NS", 0, 143.0, "2026-03-06", estimated=True, metrics={"status": "🔴 stop hit", "exit_rule_hit": True}),
           ev("BUY", "BBB.NS", 4, 50.0, "2026-02-10", metrics={"stages_passed": 2, "technicals": "WAIT", "comp_rs": -3.0}),
           ev("SELL", "BBB.NS", 4, 45.0, "2026-02-20"),
           ev("BUY", "AAA.NS", 3, 150.0, "2026-04-01"),                                       # bought again: a second trade
           ev("BUY", "CCC.NS", 1, 10.0, "2026-04-02"), ev("VOID", "CCC.NS", 0, None, "2026-04-03"),   # deleted by mistake: no trade
           ev("SELL", "ZZZ.NS", 5, 9.0, "2026-04-04")]                                        # a sale with no buy on record is ignored
    T = tj.trades(evs, {"AAA.NS": 165.0})
    assert list(T.status) == ["closed", "closed", "open"] and "CCC.NS" not in set(T.ticker) and "ZZZ.NS" not in set(T.ticker)
    a = T[(T.ticker == "AAA.NS") & (T.status == "closed")].iloc[0]
    assert a.entry_price == 110.0 and round(a.exit_price, 2) == 139.75 and round(a.return_pct, 2) == 27.05     # 5 @130 + 15 @143
    assert (a.days, a.adds, a.partial_exits, a.entry_note) == (60, 1, 1, "breakout") and a.exit_estimated
    assert a.entry_stages_passed == 4 and a.exit_status == "🔴 stop hit"                   # entry and exit readings side by side
    o = T[T.status == "open"].iloc[0]
    assert o.open_qty == 3 and round(o.return_pct, 1) == 10.0
    S = tj.stats(T)
    assert (S["closed"], S["open"], S["win_rate"]) == (2, 1, 50.0) and round(S["avg_loss"], 1) == -10.0
    P = tj.patterns(T, "Technicals read at entry").set_index("group")
    assert P.loc["BUY NOW", "trades"] == 1 and P.loc["WAIT", "win_rate"] == 0
    assert set(tj.patterns(T, "RS vs Nifty at entry").group) == {"5 to 15", "below 0"}
    assert tj.patterns(T, "Your tag").group.tolist() == ["not recorded"]


def test_trade_journal_catches_up_with_the_book():
    from utils import trade_journal as tj
    book = [dict(ticker="FEDERALBANK.NS", status="active", quantity=50, entry_price=250.0, entry_date="2025-12-01", notes="old"),
            dict(ticker="TCS.NS", status="closed", quantity=5, entry_price=3500.0, exit_price=3900.0, exit_date="2026-05-02"),
            dict(ticker="SBIN.NS", status="watching")]
    res = lambda t: {"FEDERALBANK.NS": "FEDERALBNK.NS"}.get(t, t)
    first = tj.catch_up(book, [], resolve=res)                       # a holding older than the journal gets its opening buy
    assert [(e["action"], e["ticker"], e["qty"], e["price"], e["date"]) for e in first] == [("BUY", "FEDERALBNK.NS", 50.0, 250.0, "2025-12-01")]
    assert first[0]["source"] == "before the journal" and tj.catch_up(book, first, resolve=res) == []      # and only once
    evs = first + [tj.event("BUY", "TCS.NS", 5, 3500.0, date="2026-03-01"), tj.event("BUY", "ITC.NS", 9, 400.0, date="2026-03-02")]
    more = {e["ticker"]: e for e in tj.catch_up(book, evs, resolve=res)}
    assert (more["TCS.NS"]["action"], more["TCS.NS"]["price"], more["TCS.NS"]["date"]) == ("SELL", 3900.0, "2026-05-02")   # closed on a classic tab
    assert more["ITC.NS"]["action"] == "VOID"                                             # deleted from the book
    T = tj.trades(evs + list(more.values()))
    assert T[T.ticker == "TCS.NS"].iloc[0].status == "closed" and "ITC.NS" not in set(T.ticker)


def test_trade_journal_captures_the_days_readings():
    from utils import trade_journal as tj
    df, _ = _snap()
    t = df.ticker.iloc[0]
    M = tj.capture([t, "NOTINIT.NS"], df, regime=dict(regime="BULL", breadth_regime="NEUTRAL", breadth=55.5))
    assert M["NOTINIT.NS"] == dict(regime_nifty="BULL", regime_breadth="NEUTRAL", breadth_pct=55.5)   # unknown stock: market regime only
    m = M[t]
    assert m["regime_nifty"] == "BULL" and 0 <= m["stages_passed"] <= 4 and "technicals" in m and "industry" in m
    import json
    json.dumps(M)                                                                          # must be storable as it is
