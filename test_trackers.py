"""
Tests for the position / watchlist / price-alert trackers and the log fixes of Nov 2026.

    python -m pytest -q test_trackers.py
"""
import importlib

import numpy as np
import pandas as pd
import pytest

from test_alert_store import gh  # noqa: F401  (fake GitHub fixture)
from utils import repo_store


def _pos():
    from utils import positions
    return importlib.reload(positions)


def _market(**px):
    return pd.DataFrame({"ticker": list(px), "price": list(px.values()), "currentPrice": list(px.values()),
                         "trend_signal": "UPTREND", "trend_score": 60})


def test_positions_are_saved_to_github_one_change_at_a_time(gh, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "1")          # save synchronously in the test
    gh.seed("positions.json", [])
    P = _pos()
    P.add_position("AAA.NS", entry_price=100, stop_loss=90, target=130)
    P.add_position("BBB.NS", status="watching")
    assert [p["ticker"] for p in gh.data("positions.json")] == ["AAA.NS", "BBB.NS"]
    assert gh.put_branches == {"alert-state"}            # never the deploy branch
    P.update_position("AAA.NS", stop_loss=95)
    saved = {p["ticker"]: p for p in gh.data("positions.json")}
    assert saved["AAA.NS"]["stop_loss"] == 95 and saved["BBB.NS"]["status"] == "watching"
    P.remove_position("AAA.NS")
    assert [p["ticker"] for p in gh.data("positions.json")] == ["BBB.NS"]


def test_position_added_elsewhere_is_not_overwritten(gh, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "1")
    gh.seed("positions.json", [])
    P = _pos()
    P.add_position("AAA.NS", entry_price=100)
    repo_store._cache["alert-state:positions.json"] = (repo_store.time.time(), [], "stale")   # stale device
    P.add_position("BBB.NS", entry_price=50)
    assert {p["ticker"] for p in gh.data("positions.json")} == {"AAA.NS", "BBB.NS"}


def test_misspelt_symbol_is_resolved_and_priced():
    P = _pos()
    known = {"FEDERALBNK.NS", "HDFCBANK.NS", "TATAMOTORS.NS"}
    assert P.resolve_ticker("FEDERALBANK.NS", known) == "FEDERALBNK.NS"
    assert P.resolve_ticker("HDFCBANK.NS", known) == "HDFCBANK.NS"
    assert P.resolve_ticker("ZZZUNKNOWN.NS", known) is None


def test_pnl_uses_resolved_symbol_and_split_factor(gh, monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_ACTIONS", "1")
    gh.seed("positions.json", [
        {"id": "1", "ticker": "FEDERALBANK.NS", "status": "active", "entry_price": 270.0, "entry_date": "2026-01-22",
         "stop_loss": 230.0, "target": 400.0, "quantity": 1, "alerts_enabled": True},
        {"id": "2", "ticker": "SPLIT.NS", "status": "active", "entry_price": 1000.0, "entry_date": "2026-01-22",
         "stop_loss": 900.0, "target": 1300.0, "quantity": 10, "alerts_enabled": True}])
    pd.DataFrame([{"ticker": "SPLIT.NS", "date": "2026-06-01", "factor": 0.5, "prev_snapshot": "2026-05-29"}]).to_csv(
        tmp_path / "data" / "corporate_actions.csv", index=False)
    P = _pos()
    out = {p["ticker"]: p for p in P.get_positions_with_pnl(_market(**{"FEDERALBNK.NS": 300.0, "SPLIT.NS": 600.0}))}
    f = out["FEDERALBANK.NS"]
    assert f["has_data"] and f["resolved_ticker"] == "FEDERALBNK.NS" and round(f["pnl_pct"], 1) == 11.1
    s = out["SPLIT.NS"]                                  # 1:1 bonus: 1000 -> 500 basis, price 600 = +20%
    assert round(s["pnl_pct"], 1) == 20.0 and s["stop_loss_adj"] == 450.0 and not s["sl_triggered"]
    assert round(s["pnl_abs"]) == 2000                   # 20 shares x Rs 100


def test_price_alerts_saved_and_removed_individually(gh, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "1")
    from utils import alerts
    A = importlib.reload(alerts)
    A.STORE.commit({"adds": [{"id": "a", "ticker": "AAA.NS", "stop_loss": 90, "target": 120}]}, "add AAA")
    A.STORE.commit({"adds": [{"id": "b", "ticker": "BBB.NS", "stop_loss": 40, "target": 70}]}, "add BBB")
    assert [x["ticker"] for x in gh.data("data/price_alerts.json")] == ["AAA.NS", "BBB.NS"]
    n = gh.puts
    hits = A.check_alerts(_market(**{"AAA.NS": 85.0, "BBB.NS": 50.0}))
    assert [h["trigger_type"] for h in hits] == ["STOP_LOSS"] and gh.puts == n      # checking writes nothing
    A.remove_alert("a")
    assert [x["ticker"] for x in gh.data("data/price_alerts.json")] == ["BBB.NS"]


# ---------------------------------------------------------------- corporate actions and logs
def test_corporate_action_factor_from_consecutive_snapshots():
    from utils import corp_actions as ca
    prev = pd.DataFrame({"ticker": ["A.NS", "B.NS", "C.NS", "D.NS"], "currentPrice": [1000.0, 200.0, 50.0, 100.0]})
    cur = pd.DataFrame({"ticker": ["A.NS", "B.NS", "C.NS", "D.NS"], "previousClose": [500.0, 201.0, 60.0, 75.0],
                        "currentPrice": [510.0, 205.0, 61.0, 81.0]})
    f = ca.factors(prev, cur)
    assert f.to_dict() == {"A.NS": 0.5, "D.NS": 0.75}       # C's +20% is a missed bar, not an action
    assert ca.factors(prev, cur, consecutive=False).to_dict() == {"A.NS": 0.5}
    T = pd.DataFrame([{"ticker": "A.NS", "date": "2026-06-10", "factor": 0.5}, {"ticker": "A.NS", "date": "2026-09-01", "factor": 0.2}])
    assert ca.cumulative_factor(T, "A.NS", "2026-06-01") == pytest.approx(0.1)
    assert ca.cumulative_factor(T, "A.NS", "2026-06-10") == pytest.approx(0.2)      # same-day signal is already on the new basis
    assert ca.cumulative_factor(T, "A.NS", "2026-06-01", "2026-07-01") == pytest.approx(0.5)
    assert ca.cumulative_factor(T, "Z.NS", "2026-01-01") == 1.0


def test_signal_log_returns_are_split_adjusted_and_21d_filled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    pd.DataFrame([{"ticker": "A.NS", "date": "2026-10-08", "factor": 0.2, "prev_snapshot": "2026-10-07"}]).to_csv(
        "data/corporate_actions.csv", index=False)
    today = pd.Timestamp.now().normalize()
    rows = [dict(signal_date="2026-10-01", ticker="A.NS", signal_price=1000.0, current_price=215.0,
                 return_since_signal=-78.5, return_5d=np.nan, return_21d=np.nan, days_held=40, status="CLOSED"),
            dict(signal_date=(today - pd.Timedelta(days=8)).strftime("%Y-%m-%d"), ticker="B.NS", signal_price=100.0,
                 current_price=100.0, return_since_signal=0.0, return_5d=np.nan, return_21d=np.nan, days_held=8, status="ACTIVE")]
    pd.DataFrame(rows).to_csv("log.csv", index=False)
    from utils import advanced_scanners as S
    out = S.refresh_signal_log_prices("log.csv", list(rows[0]), {"A.NS": 300.0, "B.NS": 110.0}).set_index("ticker")
    assert out.at["A.NS", "return_since_signal"] == 7.5      # 215 / (1000 x 0.2) - 1, not -78.5
    assert out.at["A.NS", "current_price"] == 215.0          # closed rows are not re-priced
    assert out.at["A.NS", "return_21d"] == 7.5
    assert out.at["B.NS", "return_since_signal"] == 10.0 and out.at["B.NS", "return_5d"] == 10.0


def test_snapshot_session_helpers():
    from utils import snapshots as sn
    a = pd.DataFrame({"ticker": ["A", "B"], "currentPrice": [1.0, 2.0], "bar_date": "2026-11-09"})
    b = pd.DataFrame({"ticker": ["A", "B"], "currentPrice": [1.1, 2.1], "bar_date": "2026-11-09"})
    c = pd.DataFrame({"ticker": ["A", "B"], "currentPrice": [1.1, 2.1], "bar_date": "2026-11-11"})
    assert sn.bar_date(a) == pd.Timestamp("2026-11-09")
    assert not sn.is_new_session(b, a) and sn.is_new_session(c, a)      # holiday run vs real session
    assert sn.bar_date(pd.DataFrame({"ticker": ["A"]}), "data/cache/market_master_2026_11_06.parquet") == pd.Timestamp("2026-11-06")
