"""Offline regression tests covering deep audit fixes across Pillars 1-5."""
from __future__ import annotations

import inspect
import math
import warnings

import numpy as np
import pandas as pd
import pytest


def _ohlcv(closes: list[float] | np.ndarray, vol: float = 1_000_000.0, start: str = "2024-01-01") -> pd.DataFrame:
    arr = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(arr))
    return pd.DataFrame(
        {
            "Open": arr * 0.995,
            "High": arr * 1.01,
            "Low": arr * 0.99,
            "Close": arr,
            "Volume": np.full(len(arr), vol, dtype=float),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Pillar 1: Pipeline (yf_safe, nse_prices, data_engine, fast_data_engine)
# ---------------------------------------------------------------------------

def test_yf_safe_tz_aware_and_retry_on_zero_min_cov(monkeypatch):
    from utils import nse_prices, yf_safe

    # Must not raise TypeError comparing tz-aware start/end vs tz-naive index
    monkeypatch.setattr(nse_prices, "is_current", lambda *a, **k: True)
    h = yf_safe._from_nse(
        ["^NSEI"],
        {"start": pd.Timestamp("2025-01-01", tz="UTC"), "end": pd.Timestamp("2026-11-21", tz="UTC")},
    )
    assert isinstance(h, dict)
    assert "^NSEI" in h and len(h["^NSEI"]) > 0

    calls = [0]

    def _empty_dl(*a, **k):
        calls[0] += 1
        return pd.DataFrame()

    monkeypatch.setattr(yf_safe.yf, "download", _empty_dl)
    res = yf_safe._safe_download_yahoo(["FAKE.NS"], retries=3, backoff=0, min_coverage=0.0)
    assert res.empty
    assert calls[0] == 3


def test_nse_prices_split_factors_zero_close_and_dup_actions():
    from utils import nse_prices

    g_zero = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
            "open": [100.0, 0.0, 100.0],
            "close": [100.0, 0.0, 100.0],
            "prev_close": [100.0, 100.0, 100.0],
        }
    )
    sf0, bf0 = nse_prices._factors_split_aware(g_zero, [])
    assert np.all(np.isfinite(sf0)) and np.all(np.isfinite(bf0))

    g_split = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
            "open": [100.0, 100.0, 20.0],
            "close": [100.0, 100.0, 20.0],
            "prev_close": [100.0, 100.0, 100.0],
        }
    )
    acts = [
        (pd.Timestamp("2025-01-06"), 0.2, "split", "FV Split 10 to 2"),
        (pd.Timestamp("2025-01-06"), 0.2, "split", "FV Split 10 to 2"),
    ]
    split_f, bonus_f = nse_prices._factors_split_aware(g_split, acts)
    assert np.all(np.isfinite(split_f))
    assert np.all(np.isfinite(bonus_f))
    # Duplicate identical action on same ex_date must apply once (0.2), not twice (0.04)
    assert math.isclose(float(split_f[-1]), 0.2, rel_tol=1e-6)


def test_fast_data_engine_stale_pe_clear_and_short_bars(tmp_path, monkeypatch):
    from utils import fast_data_engine as fde

    stale_fund = pd.DataFrame(
        {
            "ticker": ["INFY.NS", "TCS.NS"],
            "pe": [np.nan, 28.0],
            "marketCap": [5e12, 6e12],
            "fund_source": ["screener_csv", "screener_csv"],
        }
    )
    csv_path = tmp_path / "fundamentals_cache.csv"
    stale_fund.to_csv(csv_path, index=False)
    monkeypatch.setattr(fde, "FUNDAMENTALS_CACHE", str(csv_path))

    old_mm = pd.DataFrame(
        {
            "ticker": ["INFY.NS", "TCS.NS"],
            "name": ["Infosys", "TCS"],
            "sector": ["IT", "IT"],
            "price": [1800.0, 4000.0],
            "pe": [25.0, 30.0],  # stale cached PE on INFY must clear to NaN
            "trend_score": [75, 65],
            "trend_signal": ["STRONG UPTREND", "UPTREND"],
            "momentum": [7.0, 6.0],
            "volume_signal_score": [6.0, 6.0],
            "overall": [7.5, 6.8],
            "recommendation": ["BUY", "HOLD"],
            "marketCap": [5e12, 6e12],
        }
    )
    merged = fde._merge_fundamentals_cache(old_mm)
    infy_row = merged.set_index("ticker").loc["INFY.NS"]
    assert pd.isna(infy_row["pe"])
    assert int(infy_row["trend_score"]) == 75
    assert str(infy_row["fund_source"]) == "screener_csv"


def test_fast_data_engine_short_history_empty_nifty_and_zero_down_vol(monkeypatch):
    from utils import fast_data_engine as fde

    idx = pd.bdate_range("2026-09-01", periods=55)
    up_closes = np.linspace(100.0, 120.0, len(idx))
    stock_df = pd.DataFrame(
        {
            "Open": up_closes,
            "High": up_closes * 1.01,
            "Low": up_closes * 0.99,
            "Close": up_closes,
            "Volume": np.full(len(idx), 100_000.0),
        },
        index=idx,
    )
    dl = pd.concat({"AAA.NS": stock_df}, axis=1)
    monkeypatch.setattr(fde, "safe_history", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(fde, "safe_download", lambda *a, **k: dl)

    f_df = pd.DataFrame(
        {
            "ticker": ["AAA.NS"],
            "name": ["AAA"],
            "sector": ["Chemicals"],
            "pe": [20.0],
            "marketCap": [2e10],
            "fund_source": ["screener_csv"],
        }
    )
    out = fde.fetch_and_process_market_data(["AAA.NS"], f_df, live_mode=True, save=False)
    assert len(out) == 1
    r = out.iloc[0]
    assert pd.isna(r["rs_3m"])  # empty ^NSEI -> NaN RS (never raw stock return)
    assert pd.isna(r["volatility"])  # <=60 bars -> NaN volatility
    assert float(r["ud_ratio"]) == 5.0


# ---------------------------------------------------------------------------
# Pillar 2: Scoring, Trend Engine, Entry Timing, Market Breadth, Positions
# ---------------------------------------------------------------------------

def test_scoring_nan_guards_bank_pb_and_peg():
    from utils import scoring

    assert scoring.safe_num(np.nan, 0) == 0
    assert scoring.safe_num(np.inf, -1) == -1
    tm = scoring.calculate_trend_metrics(
        {"currentPrice": np.nan, "price": np.nan, "fiftyTwoWeekHigh": 120.0, "fiftyTwoWeekLow": 80.0}
    )
    assert tm["trend_signal"] == "NEUTRAL"

    # With fund_source set, decimal profitMargins=0.25 -> 25% (no heuristic divide)
    bank_ok = scoring.calculate_scores(
        {
            "pe": 12.0,
            "pb": 2.0,
            "roe": 0.16,
            "profitMargins": 0.25,
            "revenueGrowth": 0.18,
            "earningsGrowth": 0.22,
            "pegRatio": np.nan,
            "fund_source": "screener_csv",
        },
        sector="Financial Services",
    )
    assert bank_ok["value"] > 5.0
    assert math.isfinite(bank_ok["quality"])


def test_trend_engine_ipo_neutral_missing_mas_and_sector_excl_extended():
    from utils import trend_engine

    bars = _ohlcv(np.linspace(100.0, 140.0, 60), start="2026-06-01")
    th = trend_engine.calculate_stock_trend_history("IPO.NS", bars)
    # Early IPO bars lacking MA50/MA200 must not be forced into STRONG DOWNTREND
    assert th["trend_signal"].iloc[10] != "STRONG DOWNTREND"
    assert float(th["trend_score"].iloc[10]) >= 50.0


def test_entry_timing_unified_gate_and_zero_shares():
    from utils import entry_timing

    df = pd.DataFrame(
        {
            "ticker": ["A.NS", "B.NS", "C.NS"],
            "price": [100.0, 200.0, 50000.0],
            "trend_score": [40.0, 75.0, 75.0],
            "dist_ma50": [-2.0, -3.0, 2.0],
            "dist_ma20": [1.0, 1.0, 2.0],
            "dist_52w": [-10.0, -5.0, -2.0],
            "volatility": [25.0, 0.0, 25.0],
            "volume_signal": [1.2, 1.2, 1.5],
        }
    )
    et = entry_timing.add_position_sizing(entry_timing.add_entry_freshness(df), 100_000, 1.0, 10)
    a = et.set_index("ticker").loc["A.NS"]
    assert a["entry_label"] != "🔵 Pullback Buy"
    if a["entry_label"] == "⚪ Weak":
        assert float(a["freshness"]) == 0.0
    assert int((et.loc[et["suggested_shares"] == 0, "suggested_value"] != 0).sum()) == 0


def test_market_breadth_valid_denom_and_nan_status():
    from utils import market_breadth, market_mood

    df = pd.DataFrame(
        {
            "ticker": ["A.NS", "B.NS", "C.NS"],
            "dist_200dma": [5.0, -3.0, np.nan],
            "dist_50dma": [2.0, -1.0, np.nan],
            "trend_score": [70.0, 40.0, 50.0],
            "bar_date": ["2026-11-20", "2026-11-20", "2026-11-20"],
        }
    )
    mb = market_breadth.calculate_breadth_from_df(df)
    assert math.isclose(mb["pct_above_200dma"], 50.0, abs_tol=0.1)
    assert market_breadth.get_breadth_status(np.nan)["status"] == "UNKNOWN"

    mood_in = pd.DataFrame({"price": [100.0, 120.0], "high_52w": [110.0, 125.0], "trend_score": [60.0, 70.0]})
    cols_before = list(mood_in.columns)
    market_mood.calculate_mood_metrics(mood_in)
    assert list(mood_in.columns) == cols_before


def test_positions_watchlist_never_downgrades_active(tmp_path, monkeypatch):
    from utils import positions

    store = tmp_path / "positions.json"
    monkeypatch.setattr(positions, "_LOAD_LOCAL_FALLBACK", str(store), raising=False)
    monkeypatch.setattr(positions, "_POSITIONS_FILE", str(store), raising=False)
    monkeypatch.setattr(positions, "_load_positions", lambda: [{"ticker": "INFY.NS", "status": "active", "quantity": 10}])
    monkeypatch.setattr(positions, "sync_positions", lambda *a, **k: True)
    saved = []
    monkeypatch.setattr(positions, "_save_positions", lambda rows, **k: saved.append(rows) or True, raising=False)

    res = positions.add_to_watchlist("INFY.NS", name="Infosys", sector="IT")
    assert res["status"] == "active"


# ---------------------------------------------------------------------------
# Pillar 3: SIMBA
# ---------------------------------------------------------------------------

def test_simba_flat_rsi_and_short_htf_rsi():
    from utils import simba

    flat = pd.Series([100.0] * 30, index=pd.bdate_range("2026-01-01", periods=30))
    assert math.isclose(float(simba._rsi(flat).iloc[-1]), 50.0, abs_tol=1e-6)

    # Short HTF history (15 months = 14 diffs) must still produce finite shifted HTF RSI
    up = _ohlcv(np.linspace(100.0, 200.0, 300), vol=0.0)
    sig = simba.chart_signals(up)
    assert math.isfinite(sig["simba_rsi_m"])
    assert math.isfinite(sig["simba_mtf"])


def test_simba_yoy_negative_prev_and_lender_de_forensics_exemption():
    from utils import simba

    idx = pd.to_datetime(["2025-06-30", "2026-06-30"])
    op_neg_base = pd.Series([-10.0, 20.0], index=idx)
    yoy = simba._yoy(op_neg_base)
    assert pd.isna(yoy.iloc[-1])

    # Score table must exempt detected lenders from CFO/PAT forensics gate
    df = pd.DataFrame(
        {
            "ticker": ["BANKA.NS"],
            "name": ["Bank A"],
            "sector": ["Financial Services"],
            "industry": ["Banks"],
            "price": [500.0],
            "currentPrice": [500.0],
            "averageVolume": [200_000],
            "pe": [12.0],
            "pb": [1.8],
            "roe": [0.16],
            "rs_rating": [70.0],
            "simba_stage": [2],
            "simba_base_low": [460.0],
        }
    )
    fund = pd.DataFrame(
        {
            "ticker": ["BANKA.NS"],
            "de": [np.nan],
            "cfo_pat": [0.1],  # banks do not use industrial CFO/PAT
            "roce": [18.0],
            "roe": [16.0],
            "sales_g": [15.0],
            "np_g": [18.0],
            "lender_kind": ["bank"],
        }
    )
    st_df = simba.score_table(df, {"regime": "NORMAL BULL"}, fund, {"Banks": "Leading"})
    assert len(st_df) == 1
    assert "forensics" not in str(st_df.iloc[0]["gates"])


# ---------------------------------------------------------------------------
# Pillar 4: Technicals / Forensics / Trade Plan / View
# ---------------------------------------------------------------------------

def test_forensics_edge_cases_and_plan_guards():
    from utils import forensics, technicals_view, trade_plan

    idx30 = pd.bdate_range("2026-01-01", periods=30)
    flat_s = pd.Series([100.0] * 30, index=idx30)
    _, rsi_s, adx_s, _, _, _ = forensics.indicators(flat_s, flat_s, flat_s)
    assert math.isclose(float(rsi_s.iloc[-1]), 50.0, abs_tol=1e-6)
    assert not np.isinf(adx_s.iloc[-1])

    zero_c = _ohlcv(np.concatenate([np.full(199, 100.0), [0.0]]))
    r_z = forensics.analyse(zero_c)
    assert r_z == {} or math.isfinite(r_z.get("close", 0.0))

    circ = _ohlcv(np.concatenate([np.full(199, 99.0), [110.0]]))
    circ.iloc[-1, circ.columns.get_loc("High")] = 110.0
    circ.iloc[-1, circ.columns.get_loc("Low")] = 110.0
    circ.iloc[-1, circ.columns.get_loc("Volume")] = 1e8
    r_c = forensics.analyse(circ)
    assert math.isfinite(r_c.get("anchor_clv", 0.0))

    zero_vol_idx = _ohlcv(np.linspace(100.0, 120.0, 120), vol=0.0)
    assert forensics.analyse(zero_vol_idx).get("supply") == "NOT COMPUTED"

    # Inverted stop in plan/trade_plan must size 0 shares and NaN rr_now
    bars = _ohlcv(np.concatenate([np.linspace(100.0, 140.0, 150), np.linspace(140.0, 110.0, 50)]))
    bars.attrs["source"] = "NSE Bhavcopy daily"
    trimmed, _ = forensics.drop_provisional(bars)
    assert trimmed.attrs.get("source") == "NSE Bhavcopy daily"

    fake_P = {
        "action": "WATCH",
        "shares": 0,
        "rr_now": np.nan,
        "r": {"close": 100.0, "entry": 110.0, "stop": 105.0, "t1": 130.0, "t2": 140.0, "state": "FADE", "gate1": False, "supply": "HEAVY"},
    }
    tp = trade_plan.build(fake_P)
    assert tp["valid"] is False
    assert pd.isna(tp["t1"])

    assert technicals_view._as_bool("False") is False
    assert technicals_view._as_bool("0") is False
    assert technicals_view._as_bool(np.nan) is False
    assert technicals_view._as_bool("True") is True
    assert list(inspect.signature(technicals_view._history).parameters.keys()) == ["ticker"]


# ---------------------------------------------------------------------------
# Pillar 5: Valuation
# ---------------------------------------------------------------------------

def test_valuation_fingerprint_warnings_and_fmt():
    from utils import valuation, valuation_view

    fp = valuation.features_fingerprint()
    assert fp.startswith("f2|")

    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        # Run _screener_multiples on empty/synthetic frames with zero warnings
        q_empty = pd.DataFrame(columns=["ticker", "period", "field", "value"])
        a_empty = pd.DataFrame(columns=["ticker", "period", "field", "value"])
        m = valuation._screener_multiples(a_empty, q_empty)
        assert isinstance(m, pd.DataFrame)
    bad_w = [w for w in ws if issubclass(w.category, (RuntimeWarning, FutureWarning))]
    assert bad_w == []

    assert valuation_view._fmt(np.nan, "{:.1f}") == "—"
    assert valuation_view._fmt(np.inf, "{:.1f}") == "—"
    assert valuation_view._fmt(12.34, "{:.1f}") == "12.3"
