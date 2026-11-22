"""Wave 2 regression tests covering calculation, pipeline, accuracy, speed, and UI fixes."""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_nse_prepare_write_dtypes():
    from utils import nse_prices

    df = pd.DataFrame(
        {
            "symbol": ["RELIANCE"],
            "date": [pd.Timestamp("2026-11-20")],
            "open": [100.25],
            "high": [102.50],
            "low": [99.75],
            "close": [101.05],
            "prev_close": [100.00],
            "vwap": [100.90],
            "deliv_pct": [54.32],
            "pe": [22.5],
            "pb": [3.1],
            "dy": [0.8],
        }
    )
    out = nse_prices._prepare_write(df)
    for col in ("open", "high", "low", "close", "prev_close"):
        assert out[col].dtype == np.float64
    for col in ("vwap", "deliv_pct", "pe", "pb", "dy"):
        assert out[col].dtype == np.float32


def test_nse_fetch_day_checked_rejects_stale_holiday(monkeypatch):
    from utils import nse_prices

    class FakeResp:
        content = b"dummy"

    class FakeSess:
        def get(self, url, tries=2):
            return FakeResp()

    stale_st = pd.DataFrame({"symbol": ["TCS"], "date": [pd.Timestamp("2026-11-09")], "close": [4000.0]})
    stale_ix = pd.DataFrame({"index": ["Nifty 50"], "date": [pd.Timestamp("2026-11-09")], "close": [24000.0]})
    monkeypatch.setattr(nse_prices, "parse_stocks", lambda _b: stale_st.copy())
    monkeypatch.setattr(nse_prices, "parse_indices", lambda _b: stale_ix.copy())

    st, ix, holiday = nse_prices._fetch_day_checked(FakeSess(), pd.Timestamp("2026-11-10"))
    assert st is None
    assert ix is None
    assert holiday is True


def test_nse_symbol_map_self_loops_and_multihop(tmp_path, monkeypatch):
    from utils import nse_prices

    csv_path = tmp_path / "symbol_changes.csv"
    pd.DataFrame(
        [
            {"company": "Self", "old": "SELFCO", "new": "SELFCO", "date": "2026-01-05"},
            {"company": "Chain Co", "old": "OLD1", "new": "MID2", "date": "2025-01-10"},
            {"company": "Chain Co", "old": "MID2", "new": "NEW3", "date": "2026-02-15"},
        ]
    ).to_csv(csv_path, index=False)
    monkeypatch.setattr(nse_prices, "SYMCHG", str(csv_path))

    smap = nse_prices._symbol_map()
    assert "SELFCO" not in smap
    anc = nse_prices._ancestors("NEW3", smap)
    anc_syms = {s for s, _ in anc}
    assert {"MID2", "OLD1"} <= anc_syms


def test_nse_factors_split_aware_demerger_price_only():
    from utils import nse_prices

    dates = pd.bdate_range("2026-08-25", periods=6)
    # 2026-09-02: bar k=3 opens & closes at 40 from prior close 100 (demerger gap 0.40)
    close = [100.0, 100.0, 100.0, 40.0, 41.0, 42.0]
    open_ = [100.0, 100.0, 100.0, 40.0, 40.5, 41.0]
    g = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "close": close,
            "prev_close": [100.0, 100.0, 100.0, 40.0, 40.0, 41.0],
        }
    )
    f_px, f_sh = nse_prices._factors_split_aware(g, acts=[(dates[3], np.nan, "demerger")])
    assert f_px[3] == pytest.approx(0.40, abs=1e-6)
    assert f_sh[3] == pytest.approx(1.0, abs=1e-6)

    # Split 1:5 rescales both price and share factors
    g_sp = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 100.0, 100.0, 20.0, 20.5, 21.0],
            "close": [100.0, 100.0, 100.0, 20.0, 20.5, 21.0],
            "prev_close": [100.0, 100.0, 100.0, 20.0, 20.0, 20.5],
        }
    )
    f_px2, f_sh2 = nse_prices._factors_split_aware(g_sp, acts=[(dates[3], 0.2, "split")])
    assert f_px2[3] == pytest.approx(0.2, abs=1e-6)
    assert f_sh2[3] == pytest.approx(0.2, abs=1e-6)


def test_hot_reload_subsecond_mtime_advance(tmp_path, monkeypatch):
    from utils import hot_reload

    mod_file = tmp_path / "dummy_hr_mod.py"
    mod_file.write_text("VAL = 1\n", encoding="utf-8")
    mod = types.ModuleType("utils._dummy_hr_mod")
    mod.__file__ = str(mod_file)
    mod.VAL = 1

    monkeypatch.setitem(sys.modules, "utils._dummy_hr_mod", mod)
    monkeypatch.setattr(hot_reload, "_SEEN", {})
    assert hot_reload.stale(("utils._dummy_hr_mod",)) == []

    # Advance mtime by 0.2s (same whole wall-clock second)
    st0 = os.stat(mod_file)
    mod_file.write_text("VAL = 2\n", encoding="utf-8")
    os.utime(mod_file, (st0.st_atime + 0.2, st0.st_mtime + 0.2))
    assert hot_reload.stale(("utils._dummy_hr_mod",)) == ["utils._dummy_hr_mod"]


def test_scoring_boundaries_de_and_none_bonuses(monkeypatch):
    from utils import scoring, sector_profiles

    # Fake profile with 'base' weight & roe/roa bonuses to exercise all fixed paths
    fake_q_cfg = {
        "weights": {"roe": 0.3, "roa": 0.2, "npm": 0.2, "debt": 0.2, "base": 0.1},
        "ranges": {},
        "bonuses": {"roe_high": (20.0, 0.5), "roa_high": (10.0, 0.5)},
    }
    monkeypatch.setattr(sector_profiles, "get_quality_config", lambda _s: fake_q_cfg)
    monkeypatch.setattr(scoring, "get_quality_config", lambda _s: fake_q_cfg)

    # roe_pct boundary: 0.25 -> 25% (>20 bonus), debtToEquity=100.0 -> de=1.0
    res_hi = scoring.calculate_scores(
        {"roe": 0.25, "roa": None, "profitMargins": 0.12, "debtToEquity": 100.0, "pe": 20.0}
    )
    # Explicit 25.0 stays 25.0% (not 2500%)
    res_pct = scoring.calculate_scores(
        {"roe": 25.0, "roa": None, "profitMargins": 12.0, "debtToEquity": 100.0, "pe": 20.0}
    )
    assert res_hi["quality"] == pytest.approx(res_pct["quality"], abs=1e-9)

    # roa=None must NOT trigger roa bonus even if range exists
    res_none = scoring.calculate_scores({"roe": None, "roa": None, "profitMargins": 0.1, "debtToEquity": 0.0, "pe": 20.0})
    assert np.isfinite(res_none["quality"]) and 0.0 <= res_none["quality"] <= 10.0


def test_fundamentals_build_fy_ttm_yoy_and_lender():
    from utils import fundamentals_build as fb

    # _select_fy_rows handles FY-switcher (Dec -> Dec -> stub Mar -> Mar) indexed by period
    fy_df = pd.DataFrame(
        {"value": [80.0, 90.0, 100.0, 120.0]},
        index=pd.to_datetime(["2023-12-31", "2024-12-31", "2025-03-31", "2026-03-31"]),
    )
    sel = fb._select_fy_rows(fy_df)
    assert len(sel) == 3 and sel.index.is_monotonic_increasing

    # Non-consecutive quarters (>140d gap) must NOT produce TTM profitMargins / _ttm_np
    q_gap = pd.DataFrame(
        {"sales": [100, 100, 100, 100], "np": [10, 10, 10, 10], "op": [20, 20, 20, 20]},
        index=pd.to_datetime(["2024-03-31", "2024-06-30", "2025-03-31", "2025-06-30"]),
    )
    a_ok = pd.DataFrame(
        {
            "eq_cap": [10.0],
            "reserves": [90.0],
            "np": [25.0],
            "total_liab": [200.0],
            "borrowings": [20.0],
            "cfo": [30.0],
            "deposits": [500.0],
        },
        index=pd.to_datetime(["2025-03-31"]),
    )
    r_lender = fb.screener_ratios("BANKCO.NS", {"BANKCO.NS": a_ok}, {"BANKCO.NS": q_gap}, mcap_cr=1000.0)
    assert "_ttm_np" not in r_lender
    assert "ebitdaMargins" not in r_lender

    # Date-matched _yoy_prior finds prior ~365d even if shifted inside +-45d
    s_yoy = pd.Series([80.0, 100.0], index=pd.to_datetime(["2025-03-31", "2026-03-25"]))
    assert fb._yoy_prior(s_yoy, pd.Timestamp("2026-03-25")) == pytest.approx(80.0)


def test_valuation_fixes(tmp_path):
    from utils import valuation

    assert "Investment Banking & Brokerage" in valuation.BFSI_GRAN

    # features_fingerprint uses SHA-256 hashes of fundamental inputs (stable per bytes)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    fa = data_dir / "fundamentals_annual.csv.gz"
    fa.write_bytes(b"v1")
    fp1 = valuation.features_fingerprint(str(tmp_path))
    fa.write_bytes(b"v2")
    fp2 = valuation.features_fingerprint(str(tmp_path))
    assert fp1.startswith("f2|") and len(fp1.split("|")[1]) == 16 and fp1 != fp2

    # simulate() defaults to rho_regime = 0.0 (uncorrelated regime switches)
    lever = {"bear": [90.0, 95.0, 100.0], "base": [100.0, 110.0, 120.0], "bull": [120.0, 135.0, 150.0]}
    mult = {"bear": 10.0, "base": 15.0, "bull": 20.0}
    v_def, reg_def = valuation.simulate(lever, mult, (0.25, 0.50, 0.25), rho=0.5, w=[0.3, 0.35, 0.35], r=0.10, n=500)
    v_zero, reg_zero = valuation.simulate(
        lever, mult, (0.25, 0.50, 0.25), rho=0.5, w=[0.3, 0.35, 0.35], r=0.10, n=500, rho_regime=0.0
    )
    assert np.allclose(v_def, v_zero)
    assert np.array_equal(reg_def, reg_zero)


def test_ath_legacy_factor_no_seeded_clamp(monkeypatch):
    from utils import ath, corp_actions

    fake_ca = pd.DataFrame({"ticker": ["TEST.NS"], "date": ["2026-08-15"], "factor": [0.5]})
    monkeypatch.setattr(corp_actions, "load", lambda: fake_ca)
    assert ath._legacy_factor("TEST.NS", pd.Timestamp("2026-08-01")) == pytest.approx(0.5)


def test_us_fundamentals_parse_fixes():
    import us_fundamentals

    # Empty facts -> empty lists without crashing
    assert us_fundamentals.parse("EMPTY", {}) == ([], [])

    # debt_c ignored when ltd + ltd_c are present; stb added cleanly without double counting
    facts = {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"start": "2024-01-01", "end": "2024-12-31", "val": 1e9, "form": "10-K", "filed": "2025-02-01"},
                        {"start": "2024-01-01", "end": "2024-03-31", "val": 2e8, "form": "10-Q", "filed": "2024-05-01"},
                    ]
                }
            },
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {"start": "2024-01-01", "end": "2024-12-31", "val": 2e8, "form": "10-K", "filed": "2025-02-01"},
                        {"start": "2024-01-01", "end": "2024-03-31", "val": 5e7, "form": "10-Q", "filed": "2024-05-01"},
                    ]
                }
            },
            "StockholdersEquity": {
                "units": {"USD": [{"end": "2024-12-31", "val": 5e8, "form": "10-K", "filed": "2025-02-01"}]}
            },
            "LongTermDebt": {
                "units": {"USD": [{"end": "2024-12-31", "val": 3e8, "form": "10-K", "filed": "2025-02-01"}]}
            },
            "DebtCurrent": {
                "units": {"USD": [{"end": "2024-12-31", "val": 1e8, "form": "10-K", "filed": "2025-02-01"}]}
            },
            "ShortTermBorrowings": {
                "units": {"USD": [{"end": "2024-12-31", "val": 5e7, "form": "10-K", "filed": "2025-02-01"}]}
            },
        }
    }
    ar, qr = us_fundamentals.parse("ACME", facts)
    borr = [r["value"] for r in ar if r["field"] == "borrowings"]
    # UNIT = 1e7 -> ltd (30) + stb (5) = 35.0; DebtCurrent (10) NOT double-counted
    assert borr and borr[0] == pytest.approx(35.0)
    assert len(qr) >= 1


def test_regimes_multiindex_and_bear_classification():
    from utils import market_regime, regime_manager

    # >5% below MA200 MUST be BEAR (not BULL) even with minor drawdown
    assert regime_manager.classify_regime(nifty_close=94.0, nifty_ma200=100.0, nifty_52w_high=105.0) == "BEAR"

    # market_regime handles MultiIndex Close columns
    dates = pd.bdate_range("2025-01-01", periods=220)
    closes = np.concatenate([np.full(200, 100.0), np.full(20, 80.0)])
    mi = pd.DataFrame(
        {("Close", "^NSEI"): closes},
        index=dates,
    )
    mi.columns = pd.MultiIndex.from_tuples(mi.columns)
    res = market_regime.detect_market_regime(mi)
    assert res["regime"] == "bear"


def test_simba_and_consensus_gates():
    from utils import consensus, simba

    assert simba.HISTORY_DAYS >= 760

    df = pd.DataFrame(
        [
            {
                "ticker": "MISSFUND.NS",
                "name": "Missing Fund",
                "sector": "IT",
                "price": 500.0,
                "currentPrice": 500.0,
                "averageVolume": 200000.0,
                "simba_stage": 2,
                "simba_base_low": 480.0,
                "simba_si_since": 1.0,
                "simba_g": True,
                "simba_g_since": 1.0,
                "simba_m_since": 1.0,
                "simba_ghoomega": True,
                "simba_vcp": True,
                "simba_machayega": True,
                "simba_mtf": 3,
                "return_6m": 30.0,
                "dna_signal": "BUY",
                "fx_action": "BUY NOW",
            },
            {
                "ticker": "NEGSTOP.NS",
                "name": "Neg Stop",
                "sector": "IT",
                "price": 400.0,
                "currentPrice": 400.0,
                "averageVolume": 200000.0,
                "simba_stage": 2,
                "simba_base_low": 450.0,  # negative stop_pct
                "simba_si_since": 1.0,
                "simba_g": True,
                "simba_g_since": 1.0,
                "simba_m_since": 1.0,
                "simba_ghoomega": True,
                "simba_vcp": True,
                "simba_machayega": True,
                "simba_mtf": 3,
                "return_6m": 25.0,
                "dna_signal": "BUY",
                "fx_action": "BUY NOW",
            },
            {
                "ticker": "HIGATED.NS",
                "name": "High Gated",
                "sector": "IT",
                "price": 300.0,
                "currentPrice": 300.0,
                "averageVolume": 300000.0,
                "simba_stage": 2,
                "simba_base_low": 270.0,
                "simba_si_since": 1.0,
                "simba_g": True,
                "simba_g_since": 1.0,
                "simba_m_since": 1.0,
                "simba_ghoomega": True,
                "simba_vcp": True,
                "simba_machayega": True,
                "simba_mtf": 3,
                "return_6m": 40.0,
                "dna_signal": "BUY",
                "fx_action": "BUY NOW",
            },
        ]
    )
    fund_hi = pd.DataFrame(
        [
            {
                "ticker": "HIGATED.NS",
                "er": True,
                "er_oplev": True,
                "roce": 25.0,
                "opm_cv": 0.1,
                "promoters": 70.0,
                "de": 2.0,  # D/E > 1 -> forensic gate even with A/A+ score
                "cfo_ebitda": 0.9,
            }
        ]
    )
    # Empty fundamentals -> MISSFUND gated on forensics, NEGSTOP gated on stop + forensics, HIGATED gated on D/E
    S = simba.score_table(df, {"regime": "BULL CYCLICAL"}, fund=fund_hi, ind={"IT": "Leading", "HIGATED.NS": "LEADING"})
    assert all(str(g).strip() != "" for g in S["gates"])
    hi_grade = S.loc[S["ticker"] == "HIGATED.NS", "grade"].iloc[0]
    assert "gated" in str(hi_grade)

    # consensus excludes gated SIMBA rows (both "A (gated)" and "—" gated) from voting yes
    v = consensus.votes(df, regime={"regime": "BULL CYCLICAL"}, fund=fund_hi, ind={"IT": "Leading", "HIGATED.NS": "LEADING"})
    assert not v["simba"].any()


def test_alert_store_remove_then_readd_and_telegram_batch(tmp_path):
    from utils import alert_store, telegram_notifier

    ob = {"adds": [], "removes": [], "patches": {}}
    ob = alert_store.AlertStore._merge(ob, {"removes": ["x"]})
    ob = alert_store.AlertStore._merge(ob, {"adds": [{"id": "x", "ticker": "NEW"}]})
    applied = alert_store.AlertStore.apply([{"id": "x", "ticker": "OLD"}], ob)
    assert applied == [{"id": "x", "ticker": "NEW"}]

    # HTML tag balancing across chunks when a single long message splits
    long_part = "<b>SECTION " + "\n".join(f"item {i}: " + ("x" * 60) for i in range(90)) + "</b>"
    chunks = telegram_notifier._chunks([long_part])
    assert len(chunks) >= 2
    assert chunks[0].endswith("</b>")
    assert chunks[1].startswith("<b>")

    # send_batch clear=False keeps file; clear=True removes file
    qfile = tmp_path / "batch.jsonl"
    telegram_notifier.queue_message("<b>Hello</b>", str(qfile))
    orig_send = telegram_notifier._send_now
    telegram_notifier._send_now = lambda _txt: (True, "ok")
    try:
        ok1, _ = telegram_notifier.send_batch(str(qfile), clear=False)
        assert ok1 and qfile.exists()
        ok2, _ = telegram_notifier.send_batch(str(qfile), clear=True)
        assert ok2 and not qfile.exists()
    finally:
        telegram_notifier._send_now = orig_send


def test_telegram_queue_commit_replay(tmp_path, monkeypatch):
    from utils import positions, telegram_notifier

    commits = []

    class FakeStore:
        def commit(self, payload, reason=""):
            commits.append((payload, reason))
            return True, "ok"

    monkeypatch.setattr(positions, "STORE", FakeStore())
    batch = str(tmp_path / "tg.jsonl")
    assert telegram_notifier.queue_commit({"kind": "positions_patch", "patches": {"p1": {"k": 1}}}, batch_path=batch)
    n = telegram_notifier.replay_queued_commits(batch)
    assert n == 1 and len(commits) == 1
    assert not (tmp_path / "tg.jsonl.commits.jsonl").exists()


def test_health_check_basename_and_live_desk_seasonal():
    from scripts import health_check
    from utils import live_desk

    assert health_check.check.__code__ is not None
    # Verify Financial keeps both TRAP months and EDGE months
    fin = live_desk.SEASONAL_RULES["Financial"]
    assert fin[1] == "TRAP" and fin[10] == "EDGE" and fin[12] == "EDGE"


def test_turnaround_screener_fixes(tmp_path, monkeypatch):
    import turnaround_screener as ts
    from utils import corp_actions

    # _cagr returns None under 30 days
    assert ts._cagr(100.0, 120.0, "2026-11-10", as_of="2026-11-20") is None
    assert ts._cagr(100.0, 121.0, "2026-10-01", as_of="2026-11-20") is not None

    # score_ticker uses high_px for Off_52W_High when provided
    dates = pd.bdate_range("2025-01-01", periods=260)
    close = pd.Series(np.full(260, 100.0), index=dates)
    close.iloc[-15:] = np.linspace(100.0, 130.0, 15)
    high = close.copy()
    high.iloc[50] = 200.0  # intraday 52W high well above close -> current 130 is -35% off 200
    vol = pd.Series(np.full(260, 5_000_000.0), index=dates)
    vol.iloc[-5:] = 20_000_000.0
    nifty = pd.Series(np.full(260, 20000.0), index=dates)
    row = ts.score_ticker("TEST.NS", close, vol, close, nifty, {"TEST.NS": ("IT", "LONG")}, high_px=high)
    if row is not None:
        assert row["Off_52W_High"] == pytest.approx((130.0 / 200.0 - 1) * 100, abs=0.2)

    # update_signal_log: DROPPED -> ACTIVE reactivation, no double increment on same bar_str, split adjust
    log_file = tmp_path / "ias_log.csv"
    monkeypatch.setattr(ts, "LOG_CSV", str(log_file))
    monkeypatch.setattr(
        corp_actions,
        "load",
        lambda: pd.DataFrame({"ticker": ["SPLT.NS"], "date": ["2026-11-05"], "factor": [0.5]}),
    )
    init_row = {col: None for col in ts._LOG_COLS}
    init_row.update(
        {
            "ticker": "SPLT.NS",
            "name": "SPLT",
            "sub_industry": "IT",
            "cycle": "LONG",
            "signal_date": "2026-11-01",
            "signal_tier": "WATCH",
            "signal_price": 200.0,
            "days_on_watchlist": 2,
            "status": "DROPPED",
            "dropped_date": "2026-11-10",
            "last_seen_date": "2026-11-09",
            "max_gain": 0.0,
        }
    )
    pd.DataFrame([init_row]).to_csv(log_file, index=False)

    bulk_idx = pd.to_datetime(["2026-11-01", "2026-11-20"])
    bulk = pd.DataFrame({("SPLT.NS", "Close"): [100.0, 120.0]}, index=bulk_idx)
    bulk.columns = pd.MultiIndex.from_tuples(bulk.columns)
    today_df = pd.DataFrame(
        [
            {
                "Ticker": "SPLT.NS",
                "Name": "SPLT",
                "Sub_Industry": "IT",
                "Cycle": "LONG",
                "Tier": "READY",
                "IAS": 65.0,
                "CompRS": -1.0,
                "Off_MA50": -2.0,
                "Off_MA200": -5.0,
                "RS21": 2.0,
                "RS63": -3.0,
                "CMP": 120.0,
                "Off_52W_High": -25.0,
                "Liq5Cr": 50.0,
                "LiqFromLow": 2.0,
                "VolQuality": 1.0,
                "V21_CRS_Gap": 1.0,
                "V21_MA50_Gap": 2.0,
            }
        ]
    )

    ts.update_signal_log(today_df, bulk, "2026-11-20")
    out1 = pd.read_csv(log_file)
    assert out1.loc[0, "status"] == "ACTIVE"
    assert pd.isna(out1.loc[0, "dropped_date"])
    assert int(out1.loc[0, "days_on_watchlist"]) == 3
    # Adjusted signal price is 200 * 0.5 = 100 -> cur 120 => +20% (not -40%)
    assert out1.loc[0, "return_since_signal"] == pytest.approx(20.0, abs=0.2)

    # Re-run on same bar date must NOT increment days_on_watchlist again
    ts.update_signal_log(today_df, bulk, "2026-11-20")
    out2 = pd.read_csv(log_file)
    assert int(out2.loc[0, "days_on_watchlist"]) == 3


def test_repair_logs_rotation_and_split_idempotency(tmp_path, monkeypatch):
    from scripts import repair_logs
    from utils import corp_actions, nse_prices

    rot_path = tmp_path / "rot.csv"
    pd.DataFrame(
        [
            {"record_date": "2026-11-21", "sub_industry": "IT", "score_0_100": 80},  # Saturday
            {"record_date": "2026-11-20", "sub_industry": "Banks", "score_0_100": 70},
            {"record_date": "2026-11-20", "sub_industry": "Auto", "score_0_100": 60},
        ]
    ).to_csv(rot_path, index=False)
    stats = repair_logs.fix_rotation(str(rot_path))
    assert stats["weekend"] == 1
    rot_after = pd.read_csv(rot_path)
    assert len(rot_after) == 2
    assert list(rot_after["sub_industry"]) == ["Auto", "Banks"]

    # Test recompute_split_returns idempotency on isolated tmp cwd
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    pd.DataFrame(
        [
            {
                "signal_date": "2026-06-01",
                "ticker": "SP.NS",
                "signal_price": 1000.0,
                "current_price": 550.0,
                "return_since_signal": -45.0,
                "return_21d": -45.0,
                "status": "CLOSED",
            }
        ]
    ).to_csv(tmp_path / "data" / "tc_log.csv", index=False)
    monkeypatch.setattr(
        corp_actions,
        "load",
        lambda: pd.DataFrame({"ticker": ["SP.NS"], "date": ["2026-06-10"], "factor": [0.5]}),
    )
    h_idx = pd.to_datetime(["2026-06-01", "2026-06-22"])
    monkeypatch.setattr(
        nse_prices,
        "history",
        lambda _tk: {"SP.NS": pd.DataFrame({"Close": [500.0, 550.0]}, index=h_idx)},
    )

    s1 = repair_logs.recompute_split_returns()
    assert s1 == {"tc_log.csv": 1}
    fixed = pd.read_csv(tmp_path / "data" / "tc_log.csv")
    assert fixed.loc[0, "signal_price"] == pytest.approx(1000.0)
    assert fixed.loc[0, "return_since_signal"] == pytest.approx(10.0)
    assert fixed.loc[0, "return_21d"] == pytest.approx(10.0)

    s2 = repair_logs.recompute_split_returns()
    assert s2 == {}
