"""
Tests for utils/simba.py (Ghoomega / Machayega / Bachayega port of GMB_Suite_Reconstruction.pine).

    python -m pytest -q test_simba.py
"""
import numpy as np
import pandas as pd
import pytest

from utils import simba


def _bars(breakout=True, n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-11-06", periods=n)          # ends on a Friday
    base = 100 + rng.normal(0, 1.2, n - 100).cumsum() * 0.15   # Stage 1: flat, noisy
    turn = np.linspace(base[-1], base[-1] * 1.18, 84) + rng.normal(0, 0.6, 84)
    tight = np.full(15, turn[-1]) + rng.normal(0, 0.5, 15)
    last = tight[-1] * (1.07 if breakout else 0.995)
    c = pd.Series(np.r_[base, turn, tight, [last]], idx)
    h, l, o = c * 1.01, c * 0.99, c.shift(1).fillna(c)
    v = pd.Series(1e6, idx)
    v.iloc[-26:-1] = 6e5
    if breakout:
        h.iloc[-1], l.iloc[-1], o.iloc[-1] = last * 1.002, tight[-1] * 1.004, tight[-1] * 1.005
        v.iloc[-1] = 3.5e6
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}), \
        pd.Series(np.linspace(100, 104, n), idx)


def test_stage2_breakout_fires_ghoomega_machayega_and_si():
    df, bench = _bars()
    s = simba.chart_signals(df, bench)
    assert s["simba_g_bits"] == "1111111" and s["simba_g"]
    assert s["simba_m_bits"] == "11111" and s["simba_m"] and s["simba_m_since"] == 0
    assert s["simba_si_since"] == 0 and s["simba_mtf"] == 3
    assert s["simba_stage"] == 2 and s["simba_good_wc"] is True     # Friday close above the pivot
    assert s["simba_vol_x_m"] >= 3 and s["simba_dryup"]


def test_no_breakout_no_machayega():
    df, bench = _bars(breakout=False)
    s = simba.chart_signals(df, bench)
    assert not s["simba_m"] and np.isnan(s["simba_m_since"]) and np.isnan(s["simba_si_since"])


def test_falling_stock_is_stage4_with_breach_caution():
    idx = pd.bdate_range(end="2026-11-06", periods=300)
    c = pd.Series(np.linspace(200, 100, 300), idx)
    df = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1e6})
    s = simba.chart_signals(df, pd.Series(100.0, idx))
    assert s["simba_stage"] == 4 and s["simba_caution"] == 3 and not s["simba_g"]


def test_short_history_returns_only_bar_count():
    df, bench = _bars()
    assert simba.chart_signals(df.tail(100), bench) == {"simba_bars": 100}


def test_weekly_rsi_uses_last_completed_week():
    idx = pd.bdate_range("2026-01-01", periods=200)
    c = pd.Series(np.linspace(100, 200, 200), idx)
    w = simba._htf_rsi(c, "W-FRI")
    # every bar of the same week sees the same (previous-week) value
    wk = idx.to_period("W-FRI")
    assert w.groupby(wk).nunique().max() == 1


def test_fundamentals_yoy_matches_by_date_and_ignores_shareholding_rows(tmp_path):
    (tmp_path / "data").mkdir()
    q = [("X.NS", "sales", "2025-06-30", 100), ("X.NS", "np", "2025-06-30", 10),
         ("X.NS", "sales", "2025-09-30", 100), ("X.NS", "np", "2025-09-30", 10),
         ("X.NS", "sales", "2025-12-31", 110), ("X.NS", "np", "2025-12-31", 12),
         ("X.NS", "sales", "2026-03-31", 115), ("X.NS", "np", "2026-03-31", 14),
         ("X.NS", "sales", "2026-06-30", 130), ("X.NS", "np", "2026-06-30", 20),   # +30% sales, +100% profit
         ("X.NS", "promoters", "2026-09-30", 65)]                                  # a quarter ahead
    pd.DataFrame(q, columns=["ticker", "field", "period", "value"]).to_csv(
        tmp_path / "data" / "fundamentals_quarterly.csv.gz", index=False)
    F = simba.fundamentals(root=str(tmp_path), stale_days=10_000).set_index("ticker")
    r = F.loc["X.NS"]
    assert r["quarter"] == "Jun 2026" and r["sales_g"] == 30.0 and r["np_g"] == 100.0
    assert r["er_oplev"] and r["er"] and r["promoters"] == 65


def test_regime_classification():
    idx = pd.bdate_range(end="2026-11-06", periods=400)
    up = pd.DataFrame({"Close": np.linspace(20000, 26000, 400)}, index=idx)
    assert simba.market_regime(up, 60)["regime"] == "NORMAL BULL"
    down = pd.DataFrame({"Close": np.linspace(26000, 20000, 400)}, index=idx)
    assert simba.market_regime(down, 30)["regime"] == "BEAR CONTRACTION"
    assert simba.market_regime(down, 15)["regime"] == "WASHOUT"
    r = simba.market_regime(up, 60)
    assert 1 <= r["risk_scale"] <= 10 and r["holdings"][0] <= r["holdings"][1]


def test_score_table_grades_and_gates():
    df, bench = _bars()
    s = simba.chart_signals(df, bench)
    snap = pd.DataFrame([{**s, "ticker": "X.NS", "name": "X", "currentPrice": float(df.Close.iloc[-1]),
                          "averageVolume": 1e6, "return_6m": 30.0, "pe": 20, "sector": "A"},
                         {**s, "ticker": "Y.NS", "name": "Y", "currentPrice": float(df.Close.iloc[-1]),
                          "averageVolume": 10, "return_6m": 5.0, "pe": 20, "sector": "A"}])
    fund = pd.DataFrame([{"ticker": "X.NS", "er": True, "er_oplev": True, "roce": 25, "opm_cv": 0.1,
                          "promoters": 70, "de": 0.2, "cfo_ebitda": 0.9}])
    S = simba.score_table(snap, {"regime": "NORMAL BULL"}, fund, {"X.NS": "LEADING"}).set_index("ticker")
    # X: top-down 20 + Ghoomega 14+3(RS) + Machayega 12+3+2+2 + quality 12+4+4+4 + float 4 + MTF 5 = 89
    assert S.at["X.NS", "score"] == 89 and S.at["X.NS", "grade"] == "A+" and S.at["X.NS", "gates"] == ""
    assert "liquidity" in S.at["Y.NS", "gates"] and S.at["Y.NS", "grade"].endswith("(gated)")
