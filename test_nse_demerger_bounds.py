"""Focused regression tests for demerger boundary handling & pre-Nov-2015 CNX index aliases in utils/nse_prices.py."""
import numpy as np
import pandas as pd

from utils import nse_prices as npx
from utils.nse_prices import _factors_split_aware


def _mk_df(ex_date: str, c0: float, o1: float, c1: float) -> pd.DataFrame:
    t0 = pd.Timestamp(ex_date) - pd.offsets.BDay(1)
    t1 = pd.Timestamp(ex_date)
    t2 = pd.Timestamp(ex_date) + pd.offsets.BDay(1)
    return pd.DataFrame({
        "date": [t0, t1, t2],
        "open": [c0, o1, c1],
        "close": [c0, c1, c1],
        "prev_close": [c0, c0, c1],
    })


def test_kesoram_large_carveout_below_005():
    ex = "2025-03-10"
    g = _mk_df(ex, c0=204.72, o1=10.23, c1=9.71)
    fp, fs = _factors_split_aware(g, [(pd.Timestamp(ex), np.nan, "demerger", None)])
    expected = 10.23 / 204.72
    assert abs(fp[1] - expected) < 1e-6
    assert fs[1] == 1.0


def test_hindunilvr_small_carveout_with_verified_factor():
    ex = "2025-12-05"
    g = _mk_df(ex, c0=2462.20, o1=2422.00, c1=2338.60)
    fp, fs = _factors_split_aware(g, [(pd.Timestamp(ex), 0.980911, "demerger", None)])
    assert abs(fp[1] - 0.980911) < 1e-6
    assert fs[1] == 1.0


def test_genuspower_unverified_small_gap_unadjusted():
    ex = "2026-02-06"
    g = _mk_df(ex, c0=253.85, o1=249.90, c1=262.35)
    fp, fs = _factors_split_aware(g, [(pd.Timestamp(ex), np.nan, "demerger", None)])
    assert fp[1] == 1.0
    assert fs[1] == 1.0


def test_standard_demerger_prefers_open_gap_over_coa_factor():
    ex = "2024-12-06"
    c0, o1 = 1485.05, 650.05
    g = _mk_df(ex, c0=c0, o1=o1, c1=682.55)
    fp, fs = _factors_split_aware(g, [(pd.Timestamp(ex), 0.893505, "demerger", None)])
    assert abs(fp[1] - (o1 / c0)) < 1e-6
    assert fs[1] == 1.0


def test_cnx_nifty_pre_2015_alias_stitches_with_nifty_50(monkeypatch):
    ix_df = pd.DataFrame({
        "date": [pd.Timestamp("2015-11-06"), pd.Timestamp("2015-11-09")],
        "index": ["CNX Nifty", "Nifty 50"],
        "open": [7950.0, 7900.0],
        "high": [7980.0, 7940.0],
        "low": [7920.0, 7880.0],
        "close": [7954.3, 7915.2],
        "volume": [1000.0, 1200.0],
    })
    monkeypatch.setattr(npx, "raw", lambda kind: ix_df if kind == "indices" else pd.DataFrame())
    out = npx.history(["^NSEI"])
    assert "^NSEI" in out
    assert len(out["^NSEI"]) == 2
    assert list(out["^NSEI"].index) == [pd.Timestamp("2015-11-06"), pd.Timestamp("2015-11-09")]
