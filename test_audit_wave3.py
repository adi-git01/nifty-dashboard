"""Wave 3 regression tests: auth gate, secret hygiene, proxy TLS, MCP safety, broker sync/orders, edge bugs, US/BFSI valuation, slim data, EOD consolidation, and views modularization."""
from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
import glob
import importlib
import importlib.util
import json
import os
import py_compile
import types

import numpy as np
import pandas as pd
import pytest


IST = timezone(timedelta(hours=5, minutes=30))


# ---------------------------------------------------------------------------
# Phase A1 — Auth gate
# ---------------------------------------------------------------------------
def test_auth_gate_password_and_cloud(monkeypatch):
    from utils import auth_gate as ag

    assert ag.verify_password("secret-123", "secret-123") is True
    assert ag.verify_password("wrong", "secret-123") is False
    assert ag.verify_password("", "") is False

    fake_st = types.SimpleNamespace(secrets={})
    monkeypatch.setattr(ag, "st", fake_st)
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    assert ag.configured_password() == ""

    monkeypatch.setenv("APP_PASSWORD", "env-pw")
    assert ag.configured_password() == "env-pw"

    fake_st.secrets = {"auth": {"password": "cloud-secret-pw"}}
    assert ag.configured_password() == "cloud-secret-pw"

    monkeypatch.setenv("STREAMLIT_SHARING_MODE", "1")
    assert ag.is_cloud() is True


# ---------------------------------------------------------------------------
# Phase A2 — Telegram notifier secret isolation & redaction
# ---------------------------------------------------------------------------
def test_telegram_notifier_secret_isolation_and_redaction(tmp_path, monkeypatch):
    import requests
    from utils import telegram_notifier as tn

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"auto_alert_interval": 15}), encoding="utf-8")
    monkeypatch.setattr(tn, "CONFIG_FILE", str(cfg_path))

    class FakeSecrets(dict):
        def __contains__(self, k):
            return dict.__contains__(self, k)

    fake_st = types.SimpleNamespace(
        secrets=FakeSecrets({
            "telegram": {"bot_token": "999999:AAFakeSecretToken", "chat_id": "12345", "enabled": True},
            "email": {"gmail_address": "user@example.com", "app_password": "app-pw-value"},
        })
    )
    import sys
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)

    loaded = tn._load_config()
    assert loaded["telegram_bot_token"] == "999999:AAFakeSecretToken"
    assert loaded["gmail_address"] == "user@example.com"
    assert loaded["gmail_app_password"] == "app-pw-value"

    tn._save_config({"auto_alert_enabled": True, "auto_alert_interval": 45})
    on_disk = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert "999999:AAFakeSecretToken" not in json.dumps(on_disk)
    assert "app-pw-value" not in json.dumps(on_disk)
    assert on_disk["auto_alert_interval"] == 45

    redacted = tn._redact_secrets(
        "HTTPSConnectionPool host='api.telegram.org' path='/bot999999:AAFakeSecretToken/sendMessage' pw=app-pw-value",
        "999999:AAFakeSecretToken",
        "app-pw-value",
    )
    assert "AAFakeSecretToken" not in redacted
    assert "app-pw-value" not in redacted
    assert "bot***" in redacted

    def boom(*a, **kw):
        raise requests.exceptions.ConnectionError("Failed at https://api.telegram.org/bot999999:AAFakeSecretToken/sendMessage")

    monkeypatch.setattr(requests, "post", boom)
    ok, err = tn._send_now("hello")
    assert ok is False
    assert "AAFakeSecretToken" not in err


# ---------------------------------------------------------------------------
# Phase A3 — Order proxy TLS builder
# ---------------------------------------------------------------------------
def test_order_proxy_build_ssl_context(tmp_path):
    spec = importlib.util.spec_from_file_location("order_proxy", "deploy/order-proxy/proxy.py")
    proxy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proxy)

    assert proxy.build_ssl_context("", "") is None
    with pytest.raises(ValueError):
        proxy.build_ssl_context(str(tmp_path / "cert.pem"), "")
    with pytest.raises(FileNotFoundError):
        proxy.build_ssl_context(str(tmp_path / "missing.crt"), str(tmp_path / "missing.key"))


# ---------------------------------------------------------------------------
# Phase A4 — MCP data access traversal guard & modern valuation
# ---------------------------------------------------------------------------
def test_mcp_data_access_safe_ticker_and_valuation():
    from mcp_server import data_access as da

    assert da._safe_ticker_filename("RELIANCE.NS") == "RELIANCE"
    assert da._safe_ticker_filename("m&m.ns") == "M&M"
    for bad in ("", "../../positions", "../config", "a/b", "a\\b", ".hidden", "foo\x00bar"):
        with pytest.raises(ValueError):
            da._safe_ticker_filename(bad)

    with pytest.raises(ValueError):
        da.get_valuation("../../positions")

    if os.path.exists("data/valuation_snapshot.csv"):
        snap = pd.read_csv("data/valuation_snapshot.csv", nrows=5)
        if not snap.empty:
            t = str(snap["ticker"].iloc[0])
            res = da.get_valuation(t)
            assert "snapshot_summary" in res or "valuation" in res


# ---------------------------------------------------------------------------
# Phase A5 — Model book synchronous SENDING lock
# ---------------------------------------------------------------------------
def test_model_book_sending_lock_is_synchronous_and_guards_failure(monkeypatch):
    from utils import model_book as mb, repo_store

    calls = []

    class FakeStore:
        last_error = "503 GitHub down"

        def load(self):
            return []

        def commit(self, payload, msg, background=True):
            calls.append((payload, msg, background))
            return False

    monkeypatch.setattr(mb, "LOG", FakeStore())
    monkeypatch.setattr(repo_store, "is_configured", lambda: True)

    order = {"id": "2026-11-20-TCS-BUY", "ticker": "TCS.NS", "side": "BUY", "qty": 1, "limit": 3500.0}
    with pytest.raises(RuntimeError, match="Could not lock order"):
        mb.log_order(order, "SENDING")
    assert len(calls) == 1 and calls[0][2] is False

    # FILLED can stay asynchronous and non-fatal
    monkeypatch.setattr(mb, "LOG", type("OkStore", (FakeStore,), {"commit": lambda self, p, m, background=True: calls.append((p, m, background)) or True})())
    mb.log_order(order, "FILLED")
    assert calls[-1][2] is True


# ---------------------------------------------------------------------------
# Phase A6 — Dhan orders holidays, pre-open window, and RequestException
# ---------------------------------------------------------------------------
def test_dhan_orders_holiday_preopen_and_exceptions(tmp_path, monkeypatch):
    import requests
    from utils import dhan_orders as do

    hol_file = tmp_path / "holidays.csv"
    hol_file.write_text("date,occasion\n2026-10-02,Gandhi Jayanti\n", encoding="utf-8")
    monkeypatch.setattr(do, "HOLIDAYS_FILE", str(hol_file))

    holiday_dt = datetime(2026, 10, 2, 11, 0, tzinfo=IST)
    preopen_dt = datetime(2026, 10, 5, 9, 5, tzinfo=IST)   # Monday 09:05 IST
    open_dt = datetime(2026, 10, 5, 10, 30, tzinfo=IST)     # Monday 10:30 IST

    assert do.is_nse_holiday(holiday_dt) is True
    assert do.market_open(holiday_dt) is False
    assert do.is_pre_open(holiday_dt) is False
    assert do.is_pre_open(preopen_dt) is True
    assert do.is_nse_holiday(open_dt) is False
    assert do.market_open(open_dt) is True

    cfg = {"enabled": True, "client_id": "1001", "proxy": ""}
    order = {"id": "2026-10-05-TCS-BUY", "ticker": "TCS.NS", "side": "BUY", "qty": 1, "limit": 3500.0}
    with pytest.raises(do.BrokerError, match="pre-open"):
        do.place("tok", order, cfg=cfg, ids={"TCS": "11536"}, now=preopen_dt)

    class DeadHttp:
        def get(self, *a, **kw):
            raise requests.exceptions.Timeout("read timed out")

        def delete(self, *a, **kw):
            raise requests.exceptions.ConnectionError("connection reset")

    monkeypatch.setattr(do, "_http", lambda: DeadHttp())
    with pytest.raises(do.BrokerError, match="status"):
        do.status("tok", "ORD123", cfg=cfg)
    with pytest.raises(do.BrokerError, match="cancel"):
        do.cancel("tok", "ORD123", cfg=cfg)


# ---------------------------------------------------------------------------
# Phase A7 — SEBI 80% peak-margin & SELL-before-BUY sequencing
# ---------------------------------------------------------------------------
def test_model_orders_peak_margin_and_sell_first(monkeypatch):
    from utils import dhan_orders as do, model_book as mb, model_orders_view as mov

    pm = mov.sebi_peak_margin_check(cash=10_000, sell_val=50_000, buy_val=60_000)
    # Usable = 10k + 0.8*50k = 50k < 60k -> shortfall 10k
    assert pm["has_both"] is True
    assert pm["usable_sell"] == pytest.approx(40_000)
    assert pm["available_for_buy"] == pytest.approx(50_000)
    assert pm["shortfall"] == pytest.approx(10_000)

    sent_seq = []
    monkeypatch.setattr(mov, "st", types.SimpleNamespace(
        session_state={"dhan_token": "tok"},
        error=lambda *a, **k: None,
        info=lambda *a, **k: None,
    ))
    monkeypatch.setattr(do, "check", lambda *a, **k: None)
    monkeypatch.setattr(do, "security_ids", lambda: {"AAA": "1", "BBB": "2"})
    monkeypatch.setattr(do, "place", lambda tok, o, *a: sent_seq.append((o["side"], o["ticker"])) or {"order_id": f"OID_{o['side']}", "status": "PENDING"})
    monkeypatch.setattr(mb, "log_order", lambda *a, **k: None)

    orders = [
        {"id": "2026-11-20-AAA-BUY", "ticker": "AAA.NS", "side": "BUY", "qty": 1, "limit": 100.0},
        {"id": "2026-11-20-BBB-SELL", "ticker": "BBB.NS", "side": "SELL", "qty": 1, "limit": 100.0},
    ]
    n = mov._send(orders, [{"ticker": "BBB.NS", "qty": 5}], [], {"slot": 20_000}, {"capital": 100_000})
    assert n == 2
    assert sent_seq == [("SELL", "BBB.NS"), ("BUY", "AAA.NS")]


# ---------------------------------------------------------------------------
# Phase A8 — Dhan sync symbol normalization & t1Qty handling
# ---------------------------------------------------------------------------
def test_broker_sync_dhan_merge_clean_symbol_and_t1qty():
    from utils.broker_sync import dhan_merge

    holdings = [{
        "tradingSymbol": "NSE:TCS-EQ",
        "totalQty": 10,
        "availableQty": 4,
        "t1Qty": 6,
        "collateralQty": 0,
        "avgCostPrice": 3500.0,
    }]
    day = [{
        "tradingSymbol": "TCS",
        "dayBuyQty": 2,
        "dayBuyValue": 7200.0,
        "daySellQty": 0,
    }]
    rows = dhan_merge(holdings, day)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "TCS"
    # avail = 4 + 6 = 10 == total, so same-day sale adjustment doesn't falsely strip T1 holdings
    assert rows[0]["qty"] == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# Phase A9 & B1 — No double split-adjustment after sync & resolve_ticker .NS fix
# ---------------------------------------------------------------------------
def test_positions_no_double_split_after_sync(tmp_path, monkeypatch):
    from utils import corp_actions, portfolio_import as pi, positions

    store_data = []

    class MemStore:
        def load(self):
            return [dict(r) for r in store_data]

        def commit(self, payload, reason="", background=True):
            for row in payload.get("adds", []):
                store_data.append(dict(row))
            for pid, patch in payload.get("patches", {}).items():
                for r in store_data:
                    if r.get("id") == pid or r.get("ticker") == pid:
                        r.update(patch)
            return True

    monkeypatch.setattr(positions, "STORE", MemStore())
    monkeypatch.setattr(corp_actions, "load", lambda: pd.DataFrame([
        {"ticker": "SPLITCO.NS", "date": "2026-06-01", "factor": 0.2},   # pre-sync 1:5 split
        {"ticker": "SPLITCO.NS", "date": "2026-12-15", "factor": 0.5},   # post-sync 1:2 split
    ]))

    # Existing old lot bought on 2026-01-10 updated via broker sync on today's post-split basis
    plan = {
        "add": [{"ticker": "SPLITCO.NS", "qty": 50, "avg": 200.0, "lots": {"Dhan": {"qty": 50, "avg": 200.0}}}],
        "update": [],
        "lots_only": [],
        "close": [],
    }
    res = pi.apply(plan, prices={"SPLITCO.NS": 100.0}, source="Dhan sync")
    assert res["added"] == 1
    # Simulate an older original entry_date (e.g. merged history) while keeping price_basis_date at sync date
    store_data[0]["entry_date"] = "2026-01-10"
    store_data[0]["price_basis_date"] = "2026-10-01"

    mkt = pd.DataFrame([{"ticker": "SPLITCO.NS", "currentPrice": 110.0, "trend_signal": "BUY", "trend_score": 80}])
    out = positions.get_positions_with_pnl(mkt)
    # Only the post-2026-10-01 split (0.5x) applies -> 200 * 0.5 = 100 (NOT 0.2 * 0.5 = 0.1 -> 20)
    assert out[0]["entry_price_adj"] == pytest.approx(100.0)
    assert out[0]["adj_factor"] == pytest.approx(0.5)


def test_positions_resolve_ticker_exact_ns_guard(monkeypatch):
    from utils import extended, positions

    monkeypatch.setattr(extended, "eq_symbols", lambda: {
        "AVANCE.NS": "Avance Technologies Ltd",
        "AVANTIFEED.NS": "Avanti Feeds Ltd",
    })
    known = {"AVANTI.NS", "TCS.NS"}
    assert positions.resolve_ticker("AVANCE.NS", known) is None
    assert positions.resolve_ticker("FEDERALBANK.NS", known | {"FEDERALBNK.NS"}) == "FEDERALBNK.NS"
    assert positions.resolve_ticker("TCS.NS", known) == "TCS.NS"


# ---------------------------------------------------------------------------
# Phase B2–B6 — Edge bug fixes
# ---------------------------------------------------------------------------
def test_build_nifty1000_enforce_gates_respects_min_target():
    import build_nifty1000_ci as bnc

    top = pd.DataFrame({
        "ticker": [f"{m}.NS" for m in bnc.MEGA_CAPS] + [f"STK{i}.NS" for i in range(1000 - len(bnc.MEGA_CAPS))],
        "market_cap": np.linspace(1e12, 1e10, 1000),
    })
    prev = pd.DataFrame({"Ticker": top["ticker"].tolist() + [f"OLD{i}.NS" for i in range(170)]})
    # Previously len(previous)==1170 gave floor 1111 > 1000 and aborted; now expected_base = min(1170, 1000) = 1000
    bnc.enforce_gates(top, resolved_n=2200, eq_n=2300, previous=prev, target=1000)


def test_extended_eq_symbols_includes_be_and_bz(tmp_path, monkeypatch):
    from utils import extended

    eq = tmp_path / "EQUITY_L.csv"
    eq.write_text(
        "SYMBOL,NAME OF COMPANY,SERIES\n"
        "RELIANCE,Reliance Industries,EQ\n"
        "BECO,BE Stock Ltd,BE\n"
        "BZCO,BZ Stock Ltd,BZ\n"
        "OTHER,Debt Security,N1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(extended, "EQ_LIST", str(eq))
    extended._eq_cached.cache_clear()
    syms = extended.eq_symbols()
    assert {"RELIANCE.NS", "BECO.NS", "BZCO.NS"} <= set(syms)
    assert "OTHER.NS" not in syms


def test_stock_picker_invalid_bse_preserves_nse_pick(monkeypatch):
    import sys
    from utils import stock_picker as sp

    warnings = []
    fake_st = types.SimpleNamespace(
        selectbox=lambda *a, **kw: "RELIANCE.NS",
        text_input=lambda *a, **kw: "INVALID_TEXT",
        warning=lambda msg: warnings.append(msg),
    )
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.setattr(sp, "names", lambda df=None: {"RELIANCE.NS": "Reliance Industries"})
    picked = sp.pick("Stock", "k_pick")
    assert picked == "RELIANCE.NS"
    assert warnings and "6 digits" in warnings[0]


def test_entry_timing_deeply_negative_dist_ma50_not_pullback():
    from utils.entry_timing import add_entry_freshness

    df = pd.DataFrame([
        {"ticker": "FALLING.NS", "price": 75.0, "fiftyDayAverage": 100.0, "rs_1w": 4.0, "rs_1m": 8.0, "trend_score": 70},
        {"ticker": "PULL.NS", "price": 102.0, "fiftyDayAverage": 100.0, "rs_1w": 4.0, "rs_1m": 8.0, "trend_score": 70},
    ])
    out = add_entry_freshness(df)
    assert out.loc[0, "entry_label"] == "⚪ Weak"
    assert out.loc[1, "entry_label"] == "🔵 Pullback Buy"


def test_market_explorer_comprs_weights():
    with open("utils/market_explorer.py", encoding="utf-8") as f:
        src = f.read()
    assert "rs5 * 0.10 + rs21 * 0.50 + rs63 * 0.40" in src
    assert "0.30" not in src.split("6-Month CompRS History")[1]


# ---------------------------------------------------------------------------
# Phase B8 & C1 — US IFRS/EBITDA/shares + BFSI NPA/CET1 ingestion & scoring
# ---------------------------------------------------------------------------
def test_us_fundamentals_ifrs_ebitda_shares():
    import us_fundamentals as uf

    facts = {
        "facts": {
            "ifrs-full": {
                "Revenue": {"units": {"USD": [
                    {"start": f"{y}-01-01", "end": f"{y}-12-31", "val": 100e9 + y * 1e9, "form": "20-F", "filed": f"{y+1}-03-01"}
                    for y in range(2021, 2026)
                ]}},
                "ProfitLossFromOperatingActivities": {"units": {"USD": [
                    {"start": f"{y}-01-01", "end": f"{y}-12-31", "val": 20e9, "form": "20-F", "filed": f"{y+1}-03-01"}
                    for y in range(2021, 2026)
                ]}},
                "DepreciationAndAmortisationExpense": {"units": {"USD": [
                    {"start": f"{y}-01-01", "end": f"{y}-12-31", "val": 5e9, "form": "20-F", "filed": f"{y+1}-03-01"}
                    for y in range(2021, 2026)
                ]}},
                "ProfitLoss": {"units": {"USD": [
                    {"start": f"{y}-01-01", "end": f"{y}-12-31", "val": 15e9, "form": "20-F", "filed": f"{y+1}-03-01"}
                    for y in range(2021, 2026)
                ]}},
                "WeightedAverageShares": {"units": {"shares": [
                    {"start": f"{y}-01-01", "end": f"{y}-12-31", "val": 2e9, "form": "20-F", "filed": f"{y+1}-03-01"}
                    for y in range(2021, 2026)
                ]}},
            }
        }
    }
    arows, _ = uf.parse("SAP", facts, price=150.0)
    adf = pd.DataFrame(arows)
    # EBITDA = (20e9 + 5e9) / 1e7 = 2500.0 Cr-equivalent units
    op_vals = adf[adf["field"] == "op"]["value"]
    assert not op_vals.empty and op_vals.iloc[-1] == pytest.approx(2500.0)
    sh_vals = adf[adf["field"] == "shares_cr"]["value"]
    assert not sh_vals.empty and sh_vals.iloc[-1] == pytest.approx(200.0)


def test_nse_data_fetch_bank_npa_and_delivery_panel():
    from utils import nse_data

    class MockSession:
        def json(self, path, **kw):
            return {
                "bankNonBnking": "B",
                "resCmpData": [
                    {"re_to_dt": "30-Sep-2026", "re_per_grs_npa": "2.10", "re_grs_npa_per": "0.52", "re_cet_1_ret": "14.20"}
                ],
            }

    rows = nse_data.fetch_bank_npa(MockSession(), "SBIN")
    by_fld = {r["field"]: r["value"] for r in rows}
    assert by_fld == pytest.approx({"gnpa": 2.10, "nnpa": 0.52, "cet1": 14.20})
    assert all(r["ticker"] == "SBIN.NS" for r in rows)

    if os.path.exists(nse_data.DELIV_20D):
        dp = nse_data.delivery_panel(20)
        assert not dp.empty and len(dp) <= 20


# ---------------------------------------------------------------------------
# Phase C2, C4, C5, C6 — Slim NSE prices, scorecard incremental cache, EOD & views
# ---------------------------------------------------------------------------
def test_nse_prices_parse_stocks_drops_bloat_columns():
    from utils import nse_prices

    csv_bytes = (
        b"SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, CLOSE_PRICE, TTL_TRD_QNTY, DELIV_PER, NO_OF_TRADES, AVG_PRICE, DELIV_QTY\n"
        b"TCS, EQ, 20-Nov-2026, 3900.0, 3910.0, 3950.0, 3890.0, 3940.0, 1200000, 58.4, 90000, 3925.0, 700800\n"
    )
    df = nse_prices.parse_stocks(csv_bytes)
    assert len(df) == 1
    assert not any(c in df.columns for c in ("vwap", "trades", "deliv_qty"))
    assert {"date", "symbol", "series", "close", "deliv_pct"} <= set(df.columns)


def test_signal_scorecard_preserves_derived_history_when_snapshots_pruned(tmp_path):
    from utils import book_logic, signal_scorecard

    cache = tmp_path / "data" / "cache"
    cache.mkdir(parents=True)
    # Build two minimal snapshots where filename date differs from bar_date
    for fname, bdate, px in (
        ("market_master_2026_11_18.parquet", "2026-11-17", 100.0),
        ("market_master_2026_11_19.parquet", "2026-11-18", 105.0),
    ):
        pd.DataFrame({
            "ticker": ["AAA.NS"],
            "currentPrice": [px],
            "bar_date": [bdate],
        }).to_parquet(cache / fname, index=False)

    closes = book_logic._closes(sorted(glob.glob(str(cache / "market_master_*.parquet"))))
    assert pd.Timestamp("2026-11-17") in closes
    assert pd.Timestamp("2026-11-18") in closes
    assert pd.Timestamp("2026-11-19") not in closes

    # Check incremental preservation invariant: old_der rows for dates earlier than on-disk snapshots survive build()
    if os.path.exists(signal_scorecard.DERIVED_HIST) and os.path.exists("data/signal_scorecard.csv"):
        dh = pd.read_parquet(signal_scorecard.DERIVED_HIST)
        sc = pd.read_csv("data/signal_scorecard.csv")
        assert int(sc["sessions"].iloc[0]) >= 170
        assert len(sc) >= 17
        assert pd.to_datetime(dh["date"]).min() < pd.Timestamp("2026-04-01")


def test_eod_pipeline_steps_and_prune_guard(tmp_path):
    from scripts import eod_pipeline as ep

    assert "turnaround_screener" not in ep.STEPS
    assert "rs_alert_daemon" in ep.STEPS
    assert "prune_snapshots" in ep.STEPS

    cache = tmp_path / "data" / "cache"
    cache.mkdir(parents=True)
    for d in range(1, 10):
        pd.DataFrame({"ticker": ["A.NS"], "currentPrice": [10.0]}).to_parquet(
            cache / f"market_master_2026_11_{d:02d}.parquet", index=False
        )
    # Must refuse to prune when close_panel or scorecard_derived_history is missing
    assert ep.prune_old_snapshots(keep=3, root=str(tmp_path)) == 0
    assert len(list(cache.glob("market_master_*.parquet"))) == 9

    # Create both artifacts non-empty -> prunes cleanly to keep=3
    pd.DataFrame({"A.NS": [10.0]}, index=pd.to_datetime(["2026-11-01"])).to_parquet(cache / "close_panel.parquet")
    pd.DataFrame({"date": ["2026-11-01"], "signal": ["X"], "ticker": ["A.NS"]}).to_parquet(cache / "scorecard_derived_history.parquet")
    assert ep.prune_old_snapshots(keep=3, root=str(tmp_path)) == 6
    kept = sorted(p.name for p in cache.glob("market_master_*.parquet"))
    assert kept == [
        "market_master_2026_11_07.parquet",
        "market_master_2026_11_08.parquet",
        "market_master_2026_11_09.parquet",
    ]


def test_views_modularization_and_orphan_cleanup():
    files = ["main.py"] + sorted(glob.glob("views/*.py"))
    assert len(files) == 10
    for f in files:
        py_compile.compile(f, doraise=True)
        with open(f, encoding="utf-8") as fh:
            ast.parse(fh.read(), filename=f)

    for orphan in (
        "utils/dna_metrics.py",
        "utils/excel_export.py",
        "utils/historical_tracker.py",
        "utils/mc_valuation.py",
        "utils/support_resistance.py",
    ):
        assert not os.path.exists(orphan)


def test_fundamentals_build_skips_quote_probe_on_github_actions(monkeypatch):
    from utils import fundamentals_build as fb
    from utils.nifty1000_list import TICKERS_1000

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("FORCE_NSE_QUOTES", raising=False)
    calls = {"n": 0}
    monkeypatch.setattr(fb, "nse_quote", lambda *a, **kw: calls.__setitem__("n", calls["n"] + 1) or {})
    logs = []
    out = fb.build(list(TICKERS_1000)[:5], quotes=True, log=logs.append, sleep=0)
    assert calls["n"] == 0
    assert len(out) == 5
    assert any("skipping NSE quote-equity probe on GitHub Actions" in m for m in logs)


def test_fundamentals_build_handles_none_values(tmp_path, monkeypatch):
    from utils import fund_paths, fundamentals_build as fb, nse_prices as npx

    assert np.isnan(fb._div(None, 5))
    assert np.isnan(fb._div(5, None))
    assert np.isnan(fb._div(5, 0))

    # Missing quarterly rows -> no earningsGrowth / _ttm_np, plus None values in nse_quote
    d = tmp_path / "data"
    d.mkdir()
    pd.DataFrame([
        {"ticker": "NONECO.NS", "field": "sales", "period": "2026-03-31", "value": 100.0},
        {"ticker": "NONECO.NS", "field": "dep", "period": "2026-03-31", "value": 10.0},
        {"ticker": "NONECO.NS", "field": "mcap_cr", "period": "2026-03-31", "value": 500.0},
        {"ticker": "NONECO.NS", "field": "price_scr", "period": "2026-03-31", "value": 50.0},
    ]).to_csv(d / "fundamentals_annual.csv", index=False)
    pd.DataFrame(columns=["ticker", "field", "period", "value"]).to_csv(d / "fundamentals_quarterly.csv", index=False)

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setattr(fb, "ANNUAL", str(d / "fundamentals_annual.csv"))
    monkeypatch.setattr(fb, "QUARTERLY", str(d / "fundamentals_quarterly.csv"))
    monkeypatch.setattr(fund_paths, "resolve", lambda p, root=".": p)
    monkeypatch.setattr(npx, "history", lambda *a, **kw: {})
    monkeypatch.setattr("utils.extended.eq_symbols", lambda: {"NONECO.NS": "None Co"})
    import utils.nse_data as nd
    monkeypatch.setattr(nd, "Session", lambda: types.SimpleNamespace(warm_quote=lambda *a: None))
    monkeypatch.setattr(fb, "nse_quote", lambda S, sym: {"pe": None, "shares": None})

    out = fb.build(["NONECO.NS"], quotes=True, log=lambda *a: None, sleep=0)
    assert len(out) == 1
    assert pd.isna(out.iloc[0]["pegRatio"])
