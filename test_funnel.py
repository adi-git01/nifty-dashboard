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
