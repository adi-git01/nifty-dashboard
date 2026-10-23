"""
Durable alert store shared by RS alerts and industry alerts.
============================================================

What went wrong before (Oct 2026 audit)
---------------------------------------
Every change rewrote the whole alert list from whatever copy the caller had:

1. Write fails, read works. A public repo can be read by a token that cannot
   write to it. Creating alert A wrote it locally, the push failed silently,
   the next refresh read the repo ([]) and overwrote the local copy. Creating
   B then saved [B]. Each new stock "reset" the previous one, and the UI still
   said "Synced". The repo has never received a dashboard-created alert.
2. A conflict retry re-sent the stale list. repo_store.put_json re-read only
   the sha after a 409, then overwrote the newer file with the old payload.
3. The EOD workflow committed its checkout copy with `-X theirs`, which
   replaced any alert added while the run was in progress.
4. A failed GitHub read fell back to the local copy. On Streamlit Cloud that
   copy is the git checkout, and the next save pushed it over the real list.

How it works now
----------------
* State lives on its own branch (default `alert-state`, created from main on
  first use). Saving an alert used to be a commit on main, the branch Streamlit
  Cloud deploys from; each one updated the running app under the user and left
  the page greyed until a reload. The copies under data/ on main are no longer
  read when a token is configured.
* The file (e.g. data/rs_alerts.json on the alert branch) is changed only through
  repo_store.update_json(): read fresh, apply the *operation*, write against
  that sha, and on a conflict re-read and re-apply. Two writers can no longer
  erase each other's alerts.
* Changes are operations, not lists: add(alert), remove(id), patch(id, fields).
  A patch for an alert deleted elsewhere is dropped, so nothing comes back
  from the dead. A remove beats a later stale patch.
* The dashboard queues changes and returns at once; a background timer
  publishes everything queued within a couple of seconds as one commit, so
  entering several alerts never waits on GitHub.
* Until GitHub confirms an operation it waits in a local outbox
  (data/.alert_cache/<name>.pending.json, gitignored). The outbox is applied on
  top of every read, so a pending alert stays visible and is retried on each
  load. A failed push is shown, not hidden.
* Evaluation (advancing last_value, firing) runs inside the update function,
  against the freshest state, so the dashboard and the EOD daemon cannot both
  fire the same crossing.
* Without a token everything still works on one device: the tracked file is
  the base and the outbox holds this device's alerts. Because the outbox is
  untracked, a git pull or checkout cannot wipe it.
"""
from __future__ import annotations

import copy
import json
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils import repo_store
from utils.atomic_io import atomic_json_dump

CACHE_DIR = os.path.join("data", ".alert_cache")
_EMPTY = {"adds": [], "removes": [], "patches": {}, "reasons": []}
FLUSH_DELAY = 2.0      # seconds: entries made in quick succession go out as one commit


def _read_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default        # a half-written file must not take the dashboard down


class AlertStore:
    def __init__(self, path: str, label: str, legacy: Optional[str] = None):
        self.path = path                  # tracked in git, written only via the API
        self.label = label
        self.legacy = legacy
        base = os.path.splitext(os.path.basename(path))[0]
        self.cache = os.path.join(CACHE_DIR, base + ".json")
        self.outbox_path = os.path.join(CACHE_DIR, base + ".pending.json")
        self._lock = threading.RLock()    # Streamlit sessions are threads in one process
        self.last_source = None           # "repo" | "cache" | "file"
        self.last_error: Optional[str] = None
        self._timer: Optional[threading.Timer] = None
        self._branch: Optional[str] = None
        self.flushing = False

    @property
    def branch(self) -> str:
        if self._branch is None:
            self._branch = repo_store.get_alerts_branch()
        return self._branch

    def _ready(self) -> bool:
        """Configured, and the alert branch exists (created from the deploy branch on first use)."""
        return repo_store.is_configured() and repo_store.ensure_branch(self.branch)

    # ------------------------------------------------------------------ local files
    def _outbox(self) -> Dict[str, Any]:
        ob = _read_json(self.outbox_path, None)
        if not isinstance(ob, dict):
            return copy.deepcopy(_EMPTY)
        return {"adds": list(ob.get("adds") or []), "removes": list(ob.get("removes") or []),
                "patches": dict(ob.get("patches") or {}), "reasons": list(ob.get("reasons") or [])}

    def _save_outbox(self, ob: Dict[str, Any]) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        if not (ob["adds"] or ob["removes"] or ob["patches"]):
            try:
                os.remove(self.outbox_path)
            except FileNotFoundError:
                pass
            return
        atomic_json_dump(ob, self.outbox_path, indent=2)

    def pending(self) -> int:
        ob = self._outbox()
        return len(ob["adds"]) + len(ob["removes"]) + len(ob["patches"])

    # ------------------------------------------------------------------ operations
    @staticmethod
    def apply(alerts: Any, ob: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply outbox operations to a list of alerts. Idempotent: safe to re-apply."""
        alerts = alerts if isinstance(alerts, list) else []
        removed = set(ob.get("removes") or [])
        out = [dict(a) for a in alerts if a.get("id") not in removed]
        ids = {a.get("id") for a in out}
        for a in ob.get("adds") or []:
            if a.get("id") not in ids and a.get("id") not in removed:
                out.append(dict(a))
                ids.add(a.get("id"))
        patches = ob.get("patches") or {}
        for a in out:
            if a.get("id") in patches:
                a.update(patches[a["id"]])
        return out

    @staticmethod
    def _merge(ob: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
        ob = copy.deepcopy(ob)
        for rid in delta.get("removes") or []:
            if rid not in ob["removes"]:
                ob["removes"].append(rid)
            ob["patches"].pop(rid, None)
        ob["adds"] = [a for a in ob["adds"] if a.get("id") not in set(ob["removes"])]
        seen = {a.get("id") for a in ob["adds"]}
        ob["adds"] += [a for a in delta.get("adds") or [] if a.get("id") not in seen]
        for rid, f in (delta.get("patches") or {}).items():
            if rid not in ob["removes"]:
                ob["patches"].setdefault(rid, {}).update(f)
        ob["reasons"] = (ob.get("reasons") or []) + list(delta.get("reasons") or [])
        return ob

    @staticmethod
    def _subtract(ob: Dict[str, Any], sent: Dict[str, Any]) -> Dict[str, Any]:
        """What is left in the outbox after `sent` landed; anything queued meanwhile stays."""
        ob = copy.deepcopy(ob)
        sent_adds = {a.get("id") for a in sent["adds"]}
        ob["adds"] = [a for a in ob["adds"] if a.get("id") not in sent_adds]
        ob["removes"] = [r for r in ob["removes"] if r not in sent["removes"]]
        for rid, f in sent["patches"].items():
            left = {k: v for k, v in ob["patches"].get(rid, {}).items() if f.get(k) != v}
            if left:
                ob["patches"][rid] = left
            else:
                ob["patches"].pop(rid, None)
        ob["reasons"] = ob["reasons"][len(sent.get("reasons") or []):]
        return ob

    # ------------------------------------------------------------------ reading
    def _base(self) -> Tuple[List[Dict[str, Any]], str]:
        if self._ready():
            data, _, ok = repo_store.read_json(self.path, branch=self.branch)
            if ok:
                data = data if isinstance(data, list) else []
                try:
                    os.makedirs(CACHE_DIR, exist_ok=True)
                    atomic_json_dump(data, self.cache, indent=2)
                except Exception:
                    pass
                return data, "repo"
            cached = _read_json(self.cache, None)
            if isinstance(cached, list):
                return cached, "cache"
        data = _read_json(self.path, None)
        if not isinstance(data, list) or (not data and self.legacy):
            legacy = _read_json(self.legacy, None) if self.legacy else None
            data = legacy if isinstance(legacy, list) and legacy else (data if isinstance(data, list) else [])
        return data, "file"

    def load(self) -> List[Dict[str, Any]]:
        """Current alerts: last known repo state plus anything still waiting to sync."""
        with self._lock:
            base, src = self._base()
            self.last_source = src
            if src == "repo" and self.pending():
                self.flush_soon()          # never block a page render on GitHub
            return self.apply(base, self._outbox())

    # ------------------------------------------------------------------ writing
    def commit(self, delta: Dict[str, Any], reason: str, background: bool = False) -> bool:
        """
        Queue an operation. background=True returns at once (the dashboard): a timer
        publishes everything queued within FLUSH_DELAY as one commit. Otherwise publish
        now. True only when GitHub already has it.
        """
        with self._lock:
            self._save_outbox(self._merge(self._outbox(), {**delta, "reasons": [reason]}))
        if background:
            self.flush_soon()
            return False
        return self.flush()

    def flush_soon(self, delay: float = FLUSH_DELAY) -> None:
        """Debounced background publish; repeated calls within `delay` share one commit."""
        if not repo_store.is_configured():
            return
        repo_store.get_token(); repo_store.get_repo(); repo_store.get_branch()   # resolve secrets on this thread
        with self._lock:
            if self._timer is not None and self._timer.is_alive():
                return
            self._timer = threading.Timer(delay, self._flush_quiet)
            self._timer.daemon = True
            self._timer.start()

    def _flush_quiet(self):
        try:
            self.flush()
        except Exception as e:                       # a background thread must never raise
            self.last_error = str(e)
        if self.pending() and not self.last_error:
            self.flush_soon()                        # more was queued while we were sending

    def flush(self, reason: Optional[str] = None) -> bool:
        """Publish the outbox. The lock is held only for local file work, not the network."""
        with self._lock:
            ob = self._outbox()
            if not (ob["adds"] or ob["removes"] or ob["patches"]):
                return True
            if not self._ready():
                self.last_error = None if not repo_store.is_configured() else \
                    f"could not open branch {self.branch} -- check the token's Contents: Read and write"
                return False
            self.flushing = True
        try:
            why = ob["reasons"] or [reason or "sync"]
            msg = f"🔔 {self.label}: " + ("; ".join(why) if len(why) <= 3 else f"{len(why)} changes")
            ok, new, err = repo_store.update_json(self.path, lambda cur: self.apply(cur, ob), msg, branch=self.branch)
        finally:
            self.flushing = False
        with self._lock:
            self.last_error = err
            if ok:
                self._save_outbox(self._subtract(self._outbox(), ob))
                try:
                    atomic_json_dump(new, self.cache, indent=2)
                except Exception:
                    pass
            return ok

    def mutate(self, fn: Callable[[List[Dict[str, Any]]], Any], reason: Callable[[Any], str]) -> Tuple[Any, bool]:
        """
        Run fn(alerts) against the freshest state and save what it changed in place.
        fn may run more than once (conflict retry), so it must start fresh each call
        and return its result. Returns (result of the last call, synced).
        """
        with self._lock:
            box: Dict[str, Any] = {}

            def step(cur):
                alerts = self.apply(cur, self._outbox())
                box["res"] = fn(alerts)
                return alerts

            if self._ready():
                sent = self._outbox()
                ok, new, err = repo_store.update_json(
                    self.path, step, lambda _new: f"🔔 {self.label}: " + reason(box.get("res")), branch=self.branch)
                self.last_error = err
                if ok:
                    self._save_outbox(self._subtract(self._outbox(), sent))
                    try:
                        atomic_json_dump(new, self.cache, indent=2)
                    except Exception:
                        pass
                    return box.get("res"), True
            # Offline / local-only: run on the local view, keep the changes as patches.
            alerts = self.load()
            before = {a.get("id"): copy.deepcopy(a) for a in alerts}
            res = fn(alerts)
            patches = {}
            for a in alerts:
                old = before.get(a.get("id"), {})
                diff = {k: v for k, v in a.items() if old.get(k) != v}
                if diff:
                    patches[a["id"]] = diff
            if patches:
                self._save_outbox(self._merge(self._outbox(), {"patches": patches}))
            return res, False

    # ------------------------------------------------------------------ UI
    def status(self) -> Dict[str, Any]:
        """Plain facts for the UI: where state lives and whether the last write landed."""
        ls = repo_store.last_status(self.path)
        return {"configured": repo_store.is_configured(), "where": repo_store.status(self.branch),
                "branch": self.branch,
                "can_write": repo_store.can_write() if repo_store.is_configured() else False,
                "pending": self.pending(), "source": self.last_source, "flushing": self.flushing,
                "ok": None if ls is None else ls["ok"],
                "error": (ls or {}).get("error") or self.last_error,
                "when": None if ls is None else time.strftime("%H:%M", time.localtime(ls["when"]))}

