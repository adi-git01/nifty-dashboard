"""
Tests for the Nov 2026 audit fixes: valuation percentile band, health check, stale-snapshot guard.

    python -m pytest -q test_audit_fixes.py
"""
import glob
import os

import numpy as np
import pandas as pd
import pytest

from utils import valuation as V


@pytest.fixture(scope="module")
def results():
    ctx = V.load_context()
    snap = pd.read_parquet(sorted(glob.glob("data/cache/market_master_*.parquet"))[-1]).set_index("ticker")
    out = []
    for t in list(snap.index)[::40]:
        r = V.value_stock(t, ctx, price=float(snap.at[t, "currentPrice"]))
        if r.get("status") == "OK":
            out.append(r)
    assert len(out) >= 10
    return out


def test_band_contains_the_point_and_is_ordered(results):
    for r in results:
        b, p = r["band"], r["dist"]["cmp_pct"]
        assert b["p_lo"] <= p <= b["p_hi"] and 0 <= b["p_lo"] and b["p_hi"] <= 100
        assert b["k_lo"] <= 1.0 <= b["k_hi"]
        assert b["firmness"] in ("FIRM", "SOFT", "WIDE") and abs(b["width"] - (b["p_hi"] - b["p_lo"])) < 1e-9
        assert set(r["sensitivity"]) <= set(b["runs"])          # the old knobs are part of the band


def test_band_from_grid_tracks_the_exact_band(results):
    err = []
    for r in results:
        b = r["band"]
        lo, p, hi = V.band_from_grid(r["grid"], r["snapshot"]["price"], b["k_lo"], b["k_hi"])
        assert lo <= p <= hi
        err += [abs(lo - b["p_lo"]), abs(hi - b["p_hi"])]
    assert np.mean(err) < 2.0                                   # stored-grid shortcut vs full re-simulation


def test_band_from_grid_moves_the_right_way():
    g = np.linspace(50, 150, 99)                                # fair value P1..P99
    lo, p, hi = V.band_from_grid(g, 100, k_lo=0.8, k_hi=1.25)
    assert lo < p < hi                                          # higher fair value -> cheaper -> lower percentile
    assert V.band_from_grid(g, 100) == (p, p, p)


def test_snapshot_row_carries_the_band(results):
    row = V.summary_row(results[0])
    assert {"band_k_lo", "band_k_hi", "band_lo_why", "band_hi_why", "grid"} <= set(row)


def test_health_check_fails_without_todays_snapshot_and_passes_with_it(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("hc", os.path.join(os.path.dirname(__file__), "scripts", "health_check.py"))
    hc = importlib.util.module_from_spec(spec); spec.loader.exec_module(hc)
    src = sorted(glob.glob("data/cache/market_master_*.parquet"))[-1]
    df = pd.read_parquet(src)
    monkeypatch.chdir(tmp_path)
    os.makedirs("data/cache")
    df.to_parquet("data/cache/market_master_2026_11_06.parquet")
    fails, _ = hc.check("2026_11_09")
    assert fails and "no snapshot for today" in fails[0]
    df.to_parquet("data/cache/market_master_2026_11_09.parquet")
    fails, warns = hc.check("2026_11_09")
    assert not fails and any("equal the previous snapshot" in w for w in warns)
    df.head(100).to_parquet("data/cache/market_master_2026_11_10.parquet")
    pd.DataFrame({"t": range(1000)}).to_csv("data/nifty1000_list.csv", index=False)
    assert any("under 80%" in f for f in hc.check("2026_11_10")[0])


def test_screener_peer_multiples_replace_yahoo():
    ctx = V.load_context()
    pm = ctx.peer_mult
    assert pm["pe"].notna().sum() > 800 and pm["pb"].notna().sum() > 900
    idx = pm.index[:50]
    got = V._peer_col(ctx, idx, "pe")
    has = pm["pe"].reindex(idx).notna()
    assert (got[has] == pm["pe"].reindex(idx)[has]).all()        # Screener wins where it exists


def test_trade_plan_orders_levels_and_flags_conflicts():
    from test_simba import _bars
    from utils import trade_plan
    from utils.forensics import plan
    h, _ = _bars(breakout=True, n=500, seed=0)
    P = plan(h, None, None, 1.0)
    px = float(P["r"]["close"])
    val = {"status": "OK", "dist": {"P10": px * 0.4, "P25": px * 0.5, "P50": px * 0.6, "P90": px * 0.8, "cmp_pct": 97.0},
           "band": {"p_lo": 90.0, "p_hi": 99.0, "bucket_lo": "SELL", "bucket_hi": "SELL", "firmness": "FIRM"}}
    tp = trade_plan.build(P, val)
    assert tp["levels"].price.is_monotonic_decreasing
    assert tp["valid"] and tp["stop"] < tp["entry"] < tp["t1"]
    assert any("above the valuation's trim level" in f for f in tp["flags"])
    assert any("stop is above fair value" in f for f in tp["flags"])
    assert "Stop ₹" in tp["summary"] and "Target 1 ₹" in tp["summary"]
    none = trade_plan.build(P, {"status": "OUT OF SCOPE", "reason": "loss-making"})
    assert any("No valuation" in f for f in none["flags"]) and "Valuation" not in set(none["levels"].source)
