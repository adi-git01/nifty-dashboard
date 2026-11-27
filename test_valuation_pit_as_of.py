"""Regression tests ensuring utils/valuation.py value_stock(..., as_of=...) enforces strict PIT causality."""
import copy
import os
import numpy as np
import pandas as pd

from utils import valuation as V


def _synthetic_ctx() -> V.Context:
    t = "TESTCO.NS"
    peer = "PEERCO.NS"
    # Annual FY2020..FY2026 + headline mcap_cr / price_scr
    ann_rows = []
    for yr in range(2020, 2027):
        dt = pd.Timestamp(f"{yr}-03-31")
        mult = 1.0 + 0.12 * (yr - 2020)
        ann_rows.extend([
            (t, "sales", dt, 1000.0 * mult),
            (t, "np", dt, 120.0 * mult),
            (t, "cfo", dt, 130.0 * mult),
            (t, "cfi", dt, -40.0 * mult),
            (t, "roce", dt, 22.0),
            (t, "eq_cap", dt, 100.0),
            (t, "reserves", dt, 400.0 * mult),
            (t, "borrowings", dt, 50.0),
            (t, "eps", dt, 12.0 * mult),
            (t, "payout", dt, 20.0),
            (t, "fixed_assets", dt, 300.0),
            (t, "opm", dt, 18.0),
        ])
    # Headline Screener snapshot stamped today (shares = 10 Cr)
    ann_rows.extend([
        (t, "mcap_cr", pd.Timestamp("2026-11-26"), 3000.0),
        (t, "price_scr", pd.Timestamp("2026-11-26"), 300.0),
    ])
    A = pd.DataFrame(ann_rows, columns=["ticker", "field", "period", "value"])

    # Quarterly 2023-09-30 .. 2026-09-30 (13 quarters)
    q_dates = pd.date_range("2023-09-30", "2026-09-30", freq="QE")
    q_rows = []
    for i, qd in enumerate(q_dates):
        q_rows.extend([
            (t, "sales", qd, 350.0 + 10.0 * i),
            (t, "np", qd, 40.0 + 2.0 * i),
            (t, "promoters", qd, 55.0 + 0.1 * i),
            (t, "fii", qd, 15.0 + 0.2 * i),
            (t, "dii", qd, 10.0 + 0.1 * i),
        ])
    Q = pd.DataFrame(q_rows, columns=["ticker", "field", "period", "value"])

    m_dates = pd.date_range("2021-01-31", "2026-11-30", freq="ME")
    M = pd.DataFrame({t: np.linspace(150.0, 320.0, len(m_dates))}, index=m_dates)

    # Peer rows so peer_med is clean & deterministic
    f_rows = [
        dict(ticker=t, name="Test Co", sector="Industrials", sector_granular="Industrial Products", pe=22.0, pb=4.0, marketCap=3000e7),
    ]
    for k in range(1, 7):
        f_rows.append(dict(ticker=f"PEER{k}.NS", name=f"Peer {k}", sector="Industrials",
                           sector_granular="Industrial Products", pe=20.0 + k, pb=3.5, marketCap=4000e7))
    F = pd.DataFrame(f_rows).set_index("ticker")

    ctx = V.Context(A, Q, M, F, {})
    ctx.peer_cagr3 = {}
    ctx.peer_mult = pd.DataFrame()
    ctx.features = pd.DataFrame()
    return ctx


def test_as_of_blocks_future_quarterly_annual_monthly_and_shareholding_leakage():
    as_of = "2025-07-01"
    ctx_clean = _synthetic_ctx()
    res_clean = V.value_stock("TESTCO", ctx_clean, price=240.0, n=500, knobs=False, as_of=as_of)
    assert res_clean["status"] == "OK"

    # Mutate ALL future records after as_of to extreme 10x values (must have ZERO effect!)
    ctx_poison = copy.deepcopy(ctx_clean)
    # Future quarterly P&L (Q2 2025 ended 2025-06-30 -> filed +45d = 2025-08-14 > 2025-07-01!)
    q_mask_pl = ctx_poison.quarterly["field"].isin(["np", "sales"]) & (ctx_poison.quarterly["period"] >= "2025-06-30")
    ctx_poison.quarterly.loc[q_mask_pl, "value"] *= 10.0
    # Future shareholding (2025-06-30 + 21d = 2025-07-21 > 2025-07-01!)
    q_mask_sh = ctx_poison.quarterly["field"].isin(["promoters", "fii", "dii"]) & (ctx_poison.quarterly["period"] >= "2025-06-30")
    ctx_poison.quarterly.loc[q_mask_sh, "value"] = 95.0
    # Future annual FY2026 (2026-03-31 + 60d > 2025-07-01)
    a_mask = (~ctx_poison.annual["field"].isin(["mcap_cr", "price_scr"])) & (ctx_poison.annual["period"] >= "2026-03-31")
    ctx_poison.annual.loc[a_mask, "value"] *= 10.0
    # Future monthly prices (> 2025-07-01)
    ctx_poison.monthly.loc[ctx_poison.monthly.index > pd.Timestamp(as_of), "TESTCO.NS"] = 9999.0

    res_poison = V.value_stock("TESTCO", ctx_poison, price=240.0, n=500, knobs=False, as_of=as_of)
    assert res_poison["status"] == "OK"
    assert res_poison["snapshot"]["ttm_np_cr"] == res_clean["snapshot"]["ttm_np_cr"]
    assert res_poison["snapshot"]["ttm_sales_cr"] == res_clean["snapshot"]["ttm_sales_cr"]
    assert res_poison["snapshot"]["hi52"] == res_clean["snapshot"]["hi52"]
    assert res_poison["snapshot"]["lo52"] == res_clean["snapshot"]["lo52"]
    assert res_poison["snapshot"]["years"] == res_clean["snapshot"]["years"]
    assert res_poison["score_total"] == res_clean["score_total"]
    assert res_poison["blends"] == res_clean["blends"]


def test_as_of_q4_march_uses_60d_lag():
    ctx = _synthetic_ctx()
    # On 2025-05-20 (only 50 days after 2025-03-31), Q4 FY25 (2025-03-31) must NOT yet be visible!
    r_before = V.value_stock("TESTCO", ctx, price=240.0, n=500, knobs=False, as_of="2025-05-20")
    # On 2025-06-05 (66 days after 2025-03-31), Q4 FY25 IS visible!
    r_after = V.value_stock("TESTCO", ctx, price=240.0, n=500, knobs=False, as_of="2025-06-05")
    assert r_before["status"] == "OK" and r_after["status"] == "OK"
    assert r_after["snapshot"]["quarters"] == r_before["snapshot"]["quarters"] + 1
    assert r_after["snapshot"]["years"] == r_before["snapshot"]["years"] + 1


def test_live_as_of_none_uses_all_latest_data():
    ctx = _synthetic_ctx()
    res = V.value_stock("TESTCO", ctx, price=300.0, n=500, knobs=False, as_of=None)
    assert res["status"] == "OK"
    assert res["snapshot"]["quarters"] == 13
    assert res["snapshot"]["years"] == 7
