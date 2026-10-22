"""
Regression tests for the alert store (utils/alert_store.py) against a fake GitHub
Contents API. Each test is one way alerts used to get lost.

    python -m pytest -q test_alert_store.py
"""
import base64
import json

import pandas as pd
import pytest

from utils import alert_store, repo_store


class FakeGitHub:
    """Contents API with real sha semantics; `can_push` False = read-only token."""

    def __init__(self, can_push=True):
        self.files, self.n, self.can_push, self.puts = {}, 0, can_push, 0
        self.before_put = None          # hook to inject a concurrent write

    def _sha(self):
        self.n += 1
        return f"sha{self.n}"

    def seed(self, path, obj):
        self.files[path] = (json.dumps(obj), self._sha())

    def data(self, path):
        return json.loads(self.files[path][0]) if path in self.files else None

    def get(self, url, headers=None, params=None, timeout=None):
        if url.endswith("/repos/" + repo_store.get_repo()):
            return _Resp(200, {"permissions": {"push": self.can_push}})
        path = url.split("/contents/")[1]
        if path not in self.files:
            return _Resp(404, {})
        raw, sha = self.files[path]
        return _Resp(200, {"content": base64.b64encode(raw.encode()).decode(), "sha": sha})

    def put(self, url, headers=None, json=None, timeout=None):
        path = url.split("/contents/")[1]
        if not self.can_push:
            return _Resp(403, {}, "Resource not accessible by personal access token")
        if self.before_put:
            hook, self.before_put = self.before_put, None
            hook()
        cur = self.files.get(path)
        if cur and json.get("sha") != cur[1]:
            return _Resp(409, {}, "sha does not match")
        self.puts += 1
        raw = base64.b64decode(json["content"]).decode()
        self.files[path] = (raw, self._sha())
        return _Resp(200, {"content": {"sha": self.files[path][1]}})


class _Resp:
    def __init__(self, code, body, text=""):
        self.status_code, self._b, self.text = code, body, text

    def json(self):
        return self._b

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


@pytest.fixture
def gh(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "test")
    monkeypatch.delenv("REPO_STORE_DISABLED", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    fake = FakeGitHub()
    monkeypatch.setattr(repo_store.requests, "get", fake.get)
    monkeypatch.setattr(repo_store.requests, "put", fake.put)
    monkeypatch.setattr(repo_store.time, "sleep", lambda s: None)
    repo_store._cache.clear(); repo_store._last.clear(); repo_store._perm.clear()
    (tmp_path / "data").mkdir()
    fake.seed("data/rs_alerts.json", [])
    return fake


def _df(**vals):
    return pd.DataFrame({"ticker": list(vals), "comp_rs": list(vals.values())})


def _rs():
    import importlib
    from utils import rs_alerts
    return importlib.reload(rs_alerts)


def test_alerts_for_different_stocks_accumulate(gh):
    rs = _rs()
    rs.add_rs_alert("AAA", 20, df=_df(**{"AAA.NS": 10}))
    rs.add_rs_alert("BBB", 20, df=_df(**{"BBB.NS": 10}))
    assert [a["ticker"] for a in gh.data("data/rs_alerts.json")] == ["AAA.NS", "BBB.NS"]


def test_read_only_token_keeps_alerts_and_says_so(gh):
    gh.can_push = False                       # the original bug: reads work, writes 403
    rs = _rs()
    a = rs.add_rs_alert("AAA", 20, df=_df(**{"AAA.NS": 10}))
    rs.add_rs_alert("BBB", 20, df=_df(**{"BBB.NS": 10}))
    repo_store._cache.clear()                 # next refresh re-reads GitHub ([])
    assert [x["ticker"] for x in rs.load_rs_alerts()] == ["AAA.NS", "BBB.NS"]
    assert a["synced"] is False
    stt = rs.storage_status()
    assert stt["can_write"] is False and stt["pending"] == 2 and "403" in stt["error"]
    gh.can_push = True                        # token fixed: the outbox drains on load
    repo_store._perm.clear()
    rs.load_rs_alerts()
    assert len(gh.data("data/rs_alerts.json")) == 2 and rs.storage_status()["pending"] == 0


def test_two_devices_do_not_overwrite_each_other(gh, tmp_path, monkeypatch):
    rs = _rs()
    rs.add_rs_alert("AAA", 20)
    # Device 2 holds a stale cached copy ([]) and adds BBB.
    repo_store._cache["data/rs_alerts.json"] = (repo_store.time.time(), [], "stale")
    rs.add_rs_alert("BBB", 20)
    assert {a["ticker"] for a in gh.data("data/rs_alerts.json")} == {"AAA.NS", "BBB.NS"}


def test_conflict_retry_reapplies_instead_of_resending_stale_list(gh):
    rs = _rs()
    rs.add_rs_alert("AAA", 20)
    gh.before_put = lambda: gh.seed("data/rs_alerts.json",
                                    gh.data("data/rs_alerts.json") + [{"id": "x", "ticker": "ZZZ.NS"}])
    rs.add_rs_alert("BBB", 20)
    assert {a["ticker"] for a in gh.data("data/rs_alerts.json")} == {"AAA.NS", "ZZZ.NS", "BBB.NS"}


def test_evaluation_keeps_alert_added_mid_run(gh):
    rs = _rs()
    rs.add_rs_alert("AAA", 20, df=_df(**{"AAA.NS": 10}))
    # Another device adds BBB while the EOD daemon is evaluating.
    gh.before_put = lambda: gh.seed("data/rs_alerts.json", gh.data("data/rs_alerts.json") + [
        {"id": "b", "ticker": "BBB.NS", "metric": "comp_rs", "direction": "above", "threshold": 5,
         "last_value": 1}])
    fired = rs.check_rs_alerts(_df(**{"AAA.NS": 25, "BBB.NS": 8}))
    saved = {a["ticker"]: a for a in gh.data("data/rs_alerts.json")}
    assert set(saved) == {"AAA.NS", "BBB.NS"}
    assert saved["AAA.NS"]["last_value"] == 25 and saved["AAA.NS"]["trigger_count"] == 1
    assert {f["ticker"] for f in fired} == {"AAA.NS", "BBB.NS"}   # re-evaluated on fresh state
    # The same crossing never fires twice.
    assert rs.check_rs_alerts(_df(**{"AAA.NS": 25, "BBB.NS": 8})) == []


def test_dashboard_preview_writes_nothing(gh):
    rs = _rs()
    rs.add_rs_alert("AAA", 20, df=_df(**{"AAA.NS": 10}))
    n = gh.puts
    assert len(rs.check_rs_alerts(_df(**{"AAA.NS": 25}), persist=False)) == 1
    assert gh.puts == n and gh.data("data/rs_alerts.json")[0]["last_value"] == 10


def test_no_op_state_check_makes_no_commit(gh):
    rs = _rs()
    rs.add_rs_alert("AAA", 20, df=_df(**{"AAA.NS": 10}))
    n = gh.puts
    rs.check_rs_alerts(_df(**{"AAA.NS": 10}))
    assert gh.puts == n


def test_remove_touches_only_that_alert_and_beats_stale_patch(gh):
    rs = _rs()
    a = rs.add_rs_alert("AAA", 20)
    rs.add_rs_alert("BBB", 20)
    rs.remove_rs_alert(a["id"])
    assert [x["ticker"] for x in gh.data("data/rs_alerts.json")] == ["BBB.NS"]
    assert alert_store.AlertStore.apply([{"id": "1"}], {"removes": ["1"], "patches": {"1": {"x": 1}}}) == []


def test_duplicate_alert_is_not_added(gh):
    rs = _rs()
    rs.add_rs_alert("AAA", 20)
    assert rs.add_rs_alert("AAA", 20)["duplicate"] is True
    assert len(gh.data("data/rs_alerts.json")) == 1


def test_local_only_survives_tracked_file_reset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "rs_alerts.json").write_text("[]")
    rs = _rs()
    rs.add_rs_alert("AAA", 20)
    rs.add_rs_alert("BBB", 20)
    (tmp_path / "data" / "rs_alerts.json").write_text("[]")   # git pull / checkout
    assert [a["ticker"] for a in rs.load_rs_alerts()] == ["AAA.NS", "BBB.NS"]


def test_industry_alerts_use_the_same_store(gh):
    import importlib
    from utils import industry_alerts as ia
    ia = importlib.reload(ia)
    gh.seed("data/industry_alerts.json", [])
    ia.STORE.commit({"adds": [{"id": "h", "name": "Hotels", "tickers": ["A.NS"]}]}, "add Hotels")
    ia.STORE.commit({"adds": [{"id": "c", "name": "Cement", "tickers": ["B.NS"]}]}, "add Cement")
    assert [a["name"] for a in gh.data("data/industry_alerts.json")] == ["Hotels", "Cement"]
    ia.remove_industry_alert("h", "Hotels")
    assert [a["name"] for a in gh.data("data/industry_alerts.json")] == ["Cement"]
