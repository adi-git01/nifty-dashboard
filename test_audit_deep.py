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


def test_technicals_view_distance_columns():
    from utils import technicals_view

    sample = pd.DataFrame(
        {
            "price": [100.0, 100.0, 100.0],
            "fx_stop": [90.0, 105.0, 0.0],
            "fiftyDayAverage": [95.0, 102.0, 0.0],
            "dist_52w": [-4.0, -12.0, np.nan],
            "fx_dist_hi": [-3.0, -10.0, -6.5],
        }
    )
    res = technicals_view._add_distance_cols(sample)
    assert math.isclose(float(res.loc[0, "dist_stop_pct"]), 10.0, abs_tol=1e-6)
    assert math.isclose(float(res.loc[0, "dist_ma50_pct"]), (100.0 / 95.0 - 1.0) * 100.0, rel_tol=1e-6)
    assert math.isclose(float(res.loc[0, "dist_52w_pct"]), -4.0, abs_tol=1e-6)

    assert math.isclose(float(res.loc[1, "dist_stop_pct"]), -5.0, abs_tol=1e-6)
    assert pd.isna(res.loc[2, "dist_stop_pct"])
    assert pd.isna(res.loc[2, "dist_ma50_pct"])
    assert math.isclose(float(res.loc[2, "dist_52w_pct"]), -6.5, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Recent Commits Audit Fixes (PR #157–#160) & Unified Funnel / Stock Compare
# ---------------------------------------------------------------------------

def test_recent_commits_bug_fixes(tmp_path, monkeypatch):
    from utils import funnel, funnel_log, industry_board_view, regime_badge, snapshots, stock_report

    # 1. snapshots.is_new_session: empty today -> False; mixed currentPrice vs price -> no KeyError
    prev_df = pd.DataFrame({"ticker": ["A.NS", "B.NS"], "price": [100.0, 200.0]})
    assert snapshots.is_new_session(pd.DataFrame(), prev_df) is False
    today_same = pd.DataFrame({"ticker": ["A.NS", "B.NS"], "currentPrice": [100.0, 200.0]})
    assert snapshots.is_new_session(today_same, prev_df) is False
    today_diff = pd.DataFrame({"ticker": ["A.NS", "B.NS"], "currentPrice": [105.0, 210.0]})
    assert snapshots.is_new_session(today_diff, prev_df) is True

    # 2. industry_board_view.get_board on empty DataFrame with bar_date column -> no IndexError
    empty_df = pd.DataFrame(columns=["ticker", "bar_date", "comp_rs", "industry"])
    b_empty = industry_board_view.get_board(empty_df)
    assert isinstance(b_empty, pd.DataFrame)

    # 3. stock_report.gather: inverted stop (px < stop < entry) -> risk_pct & size_pct are NaN; duplicate val index ok
    inv_df = pd.DataFrame(
        {
            "ticker": ["INV.NS"],
            "name": ["Inverted Stop Co"],
            "industry": ["Auto Components"],
            "sector": ["Consumer Cyclical"],
            "price": [170.0],
            "currentPrice": [170.0],
            "fx_stop": [175.0],
            "fx_entry": [190.0],
            "fx_action": ["WAIT"],
            "half_size": [np.nan],
        }
    )
    dup_val = pd.DataFrame(
        {"status": ["OK", "OK"], "cmp_pct": [30.0, 35.0], "p50": [200.0, 210.0], "rating": ["BUY", "ACCUMULATE"]},
        index=["INV.NS", "INV.NS"],
    )
    rep = stock_report.gather("INV.NS", inv_df, board=pd.DataFrame(), val=dup_val)
    assert pd.isna(rep["plan"]["risk_pct"])
    assert pd.isna(rep["plan"]["size_pct"])

    # 4. funnel.checklist: NaN in string fields must not raise AttributeError
    nan_row = pd.Series(
        {
            "s_industry": False,
            "industry": np.nan,
            "ind_group": np.nan,
            "ind_status": np.nan,
            "ind_score": np.nan,
            "s_trend": False,
            "scanner": np.nan,
            "stage": np.nan,
            "comp_rs": np.nan,
            "dist_52w": np.nan,
            "s_business": np.nan,
            "biz_verdict": np.nan,
            "biz_headline": np.nan,
            "biz_flags": np.nan,
            "quarter": np.nan,
            "sales_yoy": np.nan,
            "np_yoy": np.nan,
            "s_setup": False,
            "technicals": np.nan,
            "simba_days": np.nan,
            "risk_pct": np.nan,
            "signal": False,
            "val_pct": np.nan,
            "s_value": True,
        }
    )
    chk = funnel.checklist(nan_row)
    assert len(chk) == 5

    # 5. funnel_log.update: tz-aware snap & out-of-order earlier backfill date must not be swallowed
    flog = tmp_path / "funnel_log.csv"
    monkeypatch.setattr(funnel_log, "LOG_FILE", str(flog))
    df_in = pd.DataFrame(
        {
            "ticker": ["MINDACORP.NS"],
            "name": ["Minda Corp"],
            "industry": ["Auto Components"],
            "price": [750.0],
            "currentPrice": [750.0],
            "dna_signal": ["BUY"],
            "simba_stage": [2],
            "comp_rs": [2.6],
            "fx_action": ["BUY NOW"],
            "fx_entry": [750.0],
            "fx_stop": [660.0],
        }
    )
    board_in = pd.DataFrame({"industry": ["Auto Components"], "group": ["LEADING"], "status": ["UPTREND"], "score": [87.0]})
    funnel_log.update(df_in, "2026-11-20T15:30:00+05:30", board=board_in, val=pd.DataFrame())
    df_log, _ = funnel_log.update(df_in, "2026-09-01", board=board_in, val=pd.DataFrame())
    assert len(df_log) == 2

    # 6. regime_badge.render: partial engine dict {"mode": "BULL"} must not raise KeyError
    rendered = []

    class _DummySt:
        def markdown(self, text, **kwargs):
            rendered.append(text)

        def columns(self, n):
            return [self] * int(n)

        def caption(self, text, **kwargs):
            rendered.append(text)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    partial_info = {
        "nifty": {"regime": "BULL", "price": 25000.0, "ma50": 24500.0, "ma200": 23500.0, "dist_200": 6.4, "dist_52w": -2.0},
        "breadth": {"regime": "MILD BULL", "pct": 62.0, "total": 500},
        "engine": {"mode": "BULL"},
    }
    regime_badge.render(partial_info, _DummySt())
    assert len(rendered) >= 1 and "BULL" in rendered[0]


def test_unified_funnel_and_stock_compare():
    from utils import fund_trends, funnel, funnel_view, stock_compare

    opm_map = fund_trends.q_opm_map()
    assert isinstance(opm_map, dict)
    if "MINDACORP.NS" in opm_map:
        opm_val, opm_chg = opm_map["MINDACORP.NS"]
        assert math.isfinite(opm_val)

    synth_df = pd.DataFrame(
        {
            "ticker": ["MINDACORP.NS", "ASKAUTOLTD.NS"],
            "name": ["Minda Corporation", "ASK Automotive"],
            "industry": ["Auto Components", "Auto Components"],
            "price": [749.3, 630.85],
            "currentPrice": [749.3, 630.85],
            "dna_signal": ["BUY", "HOLD"],
            "simba_stage": [2, 2],
            "simba_mtf": [3, 1],
            "comp_rs": [2.6, -3.7],
            "rs_1m": [10.0, 2.2],
            "rs_3m": [-5.8, -9.6],
            "dist_52w": [-2.6, -8.3],
            "fiftyDayAverage": [672.5, 622.3],
            "twoHundredDayAverage": [607.2, 566.5],
            "pe": [31.5, 37.6],
            "fx_action": ["BUY NOW", "BUY ON TRIGGER"],
            "fx_state": ["GO", "SETUP"],
            "fx_supply": ["CLEAN AIR", "NEAR OVERHANG"],
            "fx_signature": ["ABSORPTION", "NEUTRAL"],
            "fx_shakeout": [False, False],
            "fx_adx": [38.7, 19.8],
            "fx_rsi": [66.0, 52.0],
            "fx_entry": [749.3, 687.1],
            "fx_stop": [658.9, 594.9],
            "fx_half_size": [True, True],
        }
    )
    board = pd.DataFrame(
        {
            "industry": ["Auto Components"],
            "group": ["LEADING"],
            "status": ["UPTREND"],
            "score": [87.0],
        }
    )
    val = pd.DataFrame(
        {
            "status": ["OK", "OK"],
            "cmp_pct": [36.0, 23.0],
            "p50": [810.0, 790.0],
            "rating": ["UNRATED (gate A)", "ACCUMULATE"],
        },
        index=["MINDACORP.NS", "ASKAUTOLTD.NS"],
    )

    T = funnel.table(synth_df, board=board, val=val)
    for col in ("dist_stop_pct", "dist_ma50", "trigger_gap_pct", "fx_supply", "fx_adx", "simba_mtf", "pe", "opm", "opm_chg", "liq_cr", "prom_pct", "prom_q"):
        assert col in T.columns

    assert len(funnel_view._ind_labels(T)) == 2
    assert "⭐ ABSORPTION" in funnel_view._supply_labels(T)[0]
    assert "ADX 39" in funnel_view._trend_labels(T)[0]

    cmp_res = stock_compare.build_comparison(["MINDACORP", "ASKAUTOLTD.NS"], synth_df, board=board, val=val)
    mat = cmp_res["matrix"]
    assert list(mat.columns) == ["Metric", "MINDACORP", "ASKAUTOLTD", "Advantage"]
    assert len(mat) == 12
    adv_by_metric = dict(zip(mat["Metric"], mat["Advantage"]))
    assert "Institutions & Deals" in adv_by_metric
    assert "Liquidity, Delivery & Flags" in adv_by_metric
    assert adv_by_metric["Funnel Stages"] == "MINDACORP"
    assert adv_by_metric["Scanner Momentum"] == "MINDACORP"
    assert adv_by_metric["Technicals v6 Action"] == "MINDACORP"
    assert adv_by_metric["Entry & Trigger Gap"] == "MINDACORP"
    assert adv_by_metric["Trend Strength"] == "MINDACORP"
    assert adv_by_metric["Trailing P/E"] == "MINDACORP"
    assert adv_by_metric["Business & Valuation v4.1"] == "ASKAUTOLTD"
    assert cmp_res["swing_winner"]["bare"] == "MINDACORP"
    assert cmp_res["garp_winner"]["bare"] == "ASKAUTOLTD"


def test_compare_scoring_follows_the_evidence():
    """PR 161 audit: DISTRIBUTION is not penalised, real supply labels are scored, NaN stages are not passes,
    and the swing score ignores valuation."""
    import numpy as np
    import pandas as pd
    from utils import stock_compare as sc
    base = dict(technicals="BUY NOW", fx_state="GO", fx_supply="CLEAN AIR", fx_signature="NEUTRAL", fx_shakeout=False)
    _, neutral = sc._fmt_technicals(pd.Series(base))
    _, dist = sc._fmt_technicals(pd.Series({**base, "fx_signature": "DISTRIBUTION"}))
    _, absn = sc._fmt_technicals(pd.Series({**base, "fx_signature": "ABSORPTION"}))
    assert dist == neutral and absn > neutral
    pts = {lab: sc._fmt_technicals(pd.Series({**base, "fx_supply": lab}))[1]
           for lab in ("CLEAN AIR", "FAR OVERHANG", "INSIDE shelf", "NEAR OVERHANG")}
    assert pts["CLEAN AIR"] > pts["FAR OVERHANG"] > pts["INSIDE shelf"] > pts["NEAR OVERHANG"]
    assert sc._ok(np.nan) is False and sc._ok(True) is True and sc._ok(False) is False
    row = dict(s_industry=True, s_trend=True, s_business=True, s_setup=True, comp_rs=5.0)
    assert sc._swing_stages(pd.Series({**row, "s_value": True})) == sc._swing_stages(pd.Series({**row, "s_value": False}))
    cell, _ = sc._fmt_funnel_stages(pd.Series({**row, "s_business": np.nan, "s_value": True}))
    assert cell.startswith("4 / 5")


def test_opm_hidden_for_lenders_and_bounded(tmp_path, monkeypatch):
    import pandas as pd
    from utils import fund_trends
    q = pd.DataFrame({"ticker": ["AAA.NS"] * 4 + ["BBB.NS"] * 2,
                      "field": ["op", "sales", "op", "sales", "op", "sales"],
                      "period": ["2025-09-30", "2025-09-30", "2026-09-30", "2026-09-30", "2026-09-30", "2026-09-30"],
                      "value": [10, 100, 15, 100, -500, 1]})
    q["period"] = pd.to_datetime(q["period"])
    monkeypatch.setattr(fund_trends, "_data", lambda root=".": ({}, {t: g for t, g in q.groupby("ticker")}))
    m = fund_trends._q_opm_map("x", 1.0)
    assert m["AAA.NS"] == (15.0, 5.0) and "BBB.NS" not in m          # -50,000% margin dropped
    from utils import funnel
    assert {"Banks", "Finance"} <= funnel.LENDER_INDUSTRIES


def test_context_history_and_positioning_backfill(tmp_path, monkeypatch):
    """Daily Brief history: P/E from the index store joined with flows; the back-fill fills blanks, never overwrites."""
    import pandas as pd
    from utils import nse_data as nd, nse_prices as npx
    ctx = tmp_path / "market_context.csv"
    pd.DataFrame({"date": ["2026-11-20"], "fii_net": [10308.4], "dii_net": [-4859.8], "fii_idx_long_pct": [34.1]}).to_csv(ctx, index=False)
    monkeypatch.setitem(nd.F, "ctx", str(ctx))
    monkeypatch.setattr(nd, "DIR", str(tmp_path))
    idx = pd.DataFrame({"date": pd.to_datetime(["2026-11-18", "2026-11-19", "2026-11-20"] * 3),
                        "index": ["Nifty 50"] * 3 + ["Nifty Midcap 150"] * 3 + ["Nifty Smallcap 250"] * 3,
                        "close": [25000.0, 25500, 25800] * 3, "pe": [20.5, 20.8, 21.0, 27.8, 28.0, 28.2, 31.9, 32.2, 32.1],
                        "pb": 3.1, "dy": 1.3, "open": 1.0, "high": 1.0, "low": 1.0, "volume": 0.0})
    monkeypatch.setattr(npx, "raw", lambda kind="stocks": idx if kind == "indices" else pd.DataFrame())
    H = nd.context_history()
    assert list(H.index.strftime("%Y-%m-%d")) == ["2026-11-18", "2026-11-19", "2026-11-20"]
    assert H.loc["2026-11-20", "midcap_pe"] == 28.2 and H.loc["2026-11-20", "fii_net"] == 10308.4
    assert pd.isna(H.loc["2026-11-18", "fii_net"])

    raw = "Participant wise OI\nClient Type,Future Index Long,Future Index Short\nFII,60000,140000\nDII,1,1\n"

    class FakeS:
        def get(self, url, tries=2, timeout=20):
            return type("R", (), {"content": raw.encode()})()
    monkeypatch.setattr(nd, "Session", FakeS)
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("bf", os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "nse_context_backfill.py"))
    bf = importlib.util.module_from_spec(spec); spec.loader.exec_module(bf)
    monkeypatch.setattr(bf.time, "sleep", lambda s: None)
    monkeypatch.setattr(pd.Timestamp, "today", classmethod(lambda cls: pd.Timestamp("2026-11-21")))
    assert bf.main(["--days", "30"]) == 0
    out = pd.read_csv(ctx).set_index("date")
    assert out.loc["2026-11-20", "fii_idx_long_pct"] == 34.1                    # existing value kept
    assert round(out.loc["2026-11-18", "fii_idx_long_pct"], 1) == 30.0          # back-filled
    assert out.loc["2026-11-20", "fii_net"] == 10308.4


# ---- institutional holding change, bulk/block deal buckets, 52-week window ----------------------------------------
def test_inst_label_bounds():
    from utils.fund_trends import inst_label
    assert inst_label(1.0, 0.0) == "ACCUMULATING"
    assert inst_label(0.2, 2.0) == "ACCUMULATING"
    assert inst_label(-0.5, -2.0) == "DISTRIBUTING"
    assert inst_label(-0.4, -3.0) == "NEUTRAL"          # both legs required for distributing
    assert inst_label(float("nan"), 3.0) == ""          # no current change -> unlabeled
    assert inst_label(0.1, float("nan")) == "NEUTRAL"


def test_inst_map_matches_calendar_quarters(monkeypatch):
    from utils import fund_trends as ft
    rows = []
    for p, fii, dii in (("2025-09-30", 10, 10), ("2026-03-31", 11, 10), ("2026-06-30", 11, 11), ("2026-09-30", 12, 12)):
        for f, v in (("fii", fii), ("dii", dii), ("promoters", 50)):
            rows.append(dict(ticker="AAA", period=pd.Timestamp(p), field=f, value=v))
    # BBB skips a quarter: no QoQ match within 45 days, so no quarterly change
    for p, v in (("2026-03-31", 5), ("2026-09-30", 9)):
        for f in ("fii", "dii"):
            rows.append(dict(ticker="BBB.NS", period=pd.Timestamp(p), field=f, value=v))
    Q = {"q": pd.DataFrame(rows)}
    monkeypatch.setattr(ft, "_data", lambda root: ({}, Q))
    ft._inst_map.cache_clear()
    m = ft._inst_map("x", 1.0)
    a = m["AAA.NS"]
    assert a["d_inst_q"] == pytest.approx(2.0) and a["d_inst_y"] == pytest.approx(4.0)
    assert a["label"] == "ACCUMULATING"
    b = m["BBB.NS"]
    assert pd.isna(b["d_inst_q"]) and b["label"] == ""
    ft._inst_map.cache_clear()


def test_deal_kind_buckets():
    from utils.nse_data import deal_kind
    assert deal_kind("HDFC LIFE INSURANCE COMPANY LIMITED") == "Insurance"
    assert deal_kind("NIPPON LIFE INDIA TRUSTEE LTD-A/C NIPPON INDIA SMALL CAP FUND") == "MF"
    assert deal_kind("SBI MUTUAL FUND") == "MF"
    assert deal_kind("MORGAN STANLEY ASIA (SINGAPORE) PTE.") == "FII"
    assert deal_kind("QE SECURITIES LLP") == "Other"
    assert deal_kind(None) == "Other"


def test_deal_tally_nets_by_bucket_and_window():
    from utils.nse_data import deal_tally
    d = pd.DataFrame(dict(
        date=pd.to_datetime(["2026-11-20", "2026-11-20", "2026-11-19", "2026-07-01"]),
        symbol=["XYZ", "XYZ", "XYZ", "XYZ"],
        client=["SBI MUTUAL FUND", "GOLDMAN SACHS (SINGAPORE) PTE", "QE SECURITIES LLP", "SBI MUTUAL FUND"],
        side=["BUY", "SELL", "BUY", "BUY"], qty=[1_000_000, 500_000, 100_000, 9_000_000], price=[100.0, 100.0, 100.0, 100.0],
        kind=["bulk"] * 4))
    t = deal_tally(90, d)["XYZ.NS"]
    assert t["MF"] == pytest.approx(10.0) and t["FII"] == pytest.approx(-5.0) and t["Other"] == pytest.approx(1.0)
    assert t["inst"] == pytest.approx(5.0) and t["net"] == pytest.approx(6.0) and t["n"] == 3   # July print is outside 90 days
    assert deal_tally(90, d.iloc[0:0]) == {}


def test_delivery_old_names_single_concat(monkeypatch):
    from utils import nse_data as nd
    monkeypatch.setattr(nd, "_rename_aliases", lambda: {"OLD": "NEW", "GONE": "MISSING"})
    piv = pd.DataFrame({"NEW.NS": [50.0, 60.0]}, index=pd.to_datetime(["2026-11-19", "2026-11-20"]))
    out = nd._with_old_names(piv)
    assert list(out.columns) == ["NEW.NS", "OLD.NS"] and out["OLD.NS"].tolist() == [50.0, 60.0]


def test_52w_high_uses_365_calendar_days():
    # ~248 NSE sessions a year: 252 rows reach back past one year and can pick up a stale high
    b = pd.bdate_range(end="2026-11-20", periods=300)
    idx = b[[i % 20 != 7 for i in range(len(b))]]         # drop ~13 "holidays" a year -> ~248 sessions
    df = pd.DataFrame({"High": 100.0, "Low": 90.0, "Close": 95.0}, index=idx)
    df.loc[idx[-251], "High"] = 200.0                     # inside the last 252 rows, older than 365 days
    assert idx[-251] <= idx[-1] - pd.Timedelta(days=365)
    yr = df[df.index > df.index[-1] - pd.Timedelta(days=365)]
    assert yr["High"].max() == 100.0 and df.iloc[-252:]["High"].max() == 200.0


def test_pr162_to_165_audit_fixes(tmp_path, monkeypatch):
    from utils import daily_brief_view as dbv, fund_trends as ft, nse_data as nd, regime_badge as rb, stock_compare as sc

    # 1. inst_label must NOT label a stock as ACCUMULATING if institutions sold >= 0.5 pp this quarter
    assert ft.inst_label(-2.0, 4.0) == "NEUTRAL"
    assert ft.inst_label(0.2, 2.0) == "ACCUMULATING"
    assert ft.inst_label(-0.5, -2.0) == "DISTRIBUTING"

    # 2. _inst_map keeps single-side FII/DII filings, merges bare + .NS rows, and returns promoter deltas
    rows = [
        dict(ticker="ONLYFII", period=pd.Timestamp("2026-06-30"), field="fii", value=36.0),
        dict(ticker="ONLYFII.NS", period=pd.Timestamp("2026-06-30"), field="promoters", value=52.0),
        dict(ticker="ONLYFII.NS", period=pd.Timestamp("2026-09-30"), field="fii", value=36.84),
        dict(ticker="ONLYFII", period=pd.Timestamp("2026-09-30"), field="promoters", value=53.5),
    ]
    monkeypatch.setattr(ft, "_data", lambda root=".": ({}, {"q": pd.DataFrame(rows)}))
    ft._inst_map.cache_clear()
    im = ft._inst_map("x", 1.0)
    assert "ONLYFII.NS" in im
    assert im["ONLYFII.NS"]["fii"] == pytest.approx(36.84)
    assert pd.isna(im["ONLYFII.NS"]["dii"])
    assert im["ONLYFII.NS"]["d_inst_q"] == pytest.approx(0.84)
    assert im["ONLYFII.NS"]["d_prom_q"] == pytest.approx(1.5)
    ft._inst_map.cache_clear()

    # 3. deals_for & deal_tally: normalize symbols and resolve renames bidirectionally
    deals_csv = tmp_path / "deals.csv"
    pd.DataFrame(
        {
            "date": ["2026-11-20", "2026-11-20"],
            "symbol": ["oldsym", "newsym"],
            "client": ["SBI MUTUAL FUND", "GOLDMAN SACHS (SINGAPORE) PTE"],
            "side": ["BUY", "BUY"],
            "qty": [1_000_000, 500_000],
            "price": [100.0, 100.0],
            "kind": ["bulk", "block"],
        }
    ).to_csv(deals_csv, index=False)
    monkeypatch.setitem(nd.F, "deals", str(deals_csv))
    monkeypatch.setattr(nd, "_rename_aliases", lambda: {"OLDSYM": "NEWSYM"})
    tally = nd.deal_tally(90)
    assert tally["NEWSYM.NS"]["MF"] == pytest.approx(10.0)
    assert tally["NEWSYM.NS"]["FII"] == pytest.approx(5.0)
    assert tally["NEWSYM.NS"]["inst"] == pytest.approx(15.0)
    assert tally["OLDSYM.NS"]["inst"] == pytest.approx(15.0)
    assert len(nd.deals_for(["NEWSYM.NS"], days=90)) == 2
    assert len(nd.deals_for(["OLDSYM.NS"], days=90)) == 2

    # 4. regime_badge.nifty_regime uses 365 calendar days, ignoring stale highs > 365 days ago inside 252 rows
    b = pd.bdate_range(end="2026-11-20", periods=300)
    idx = b[[i % 20 != 7 for i in range(len(b))]]  # ~248 sessions/yr
    nc = pd.Series(100.0, index=idx)
    nc.iloc[-251] = 130.0  # > 365 days ago, inside last 252 rows
    rg = rb.nifty_regime(nc)
    assert rg["dd"] == pytest.approx(0.0)
    assert rb.engine_mode(pd.DataFrame())["mode"] == "CAUTION"

    # 5. stock_compare: WAIT scores above WATCH; new institutional & execution rows render cleanly
    _, wait_pts = sc._fmt_technicals(pd.Series(dict(technicals="WAIT", fx_state="WAIT", fx_supply="CLEAN AIR")))
    _, watch_pts = sc._fmt_technicals(pd.Series(dict(technicals="WATCH", fx_state="VETO", fx_supply="CLEAN AIR")))
    assert wait_pts > watch_pts
    inst_txt, inst_sc = sc._fmt_institutions(pd.Series(dict(inst_label="ACCUMULATING", inst_q=1.5, inst_y=3.0, fii_pct=12.0, dii_pct=8.0, deal_inst_cr=5.2)))
    assert "ACCUMULATING" in inst_txt and "Deals +5.2 cr" in inst_txt and inst_sc > 20.0
    exec_txt, exec_sc = sc._fmt_exec_risk(pd.Series(dict(liq_cr=25.0, deliv_pct=60.0, deliv_avg20=40.0, nse_flags="⚠️ ASM")))
    assert "Deliv 60% (1.5×)" in exec_txt and "⚠️ ASM" in exec_txt and exec_sc < 10.0

    # 6. daily_brief_view helpers: multi-year axis labels & last valid coordinate
    single_yr = pd.DatetimeIndex(["2026-01-05", "2026-11-20"])
    multi_yr = pd.DatetimeIndex(["2025-12-30", "2026-01-05"])
    assert dbv._flow_x_labels(single_yr) == ["05 Jan", "20 Nov"]
    assert dbv._flow_x_labels(multi_yr) == ["30 Dec 25", "05 Jan 26"]
    assert dbv._last_valid_xy(["05 Jan", "06 Jan", "07 Jan"], pd.Series([10.0, 20.0, np.nan])) == ("06 Jan", 20.0)
    assert dbv._last_valid_xy(["05 Jan"], pd.Series([np.nan])) is None
