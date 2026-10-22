"""
Repo-backed JSON store.
=======================

This project uses the git repo as its database: positions.json, the parquet
cache and every log file live there, which is why they survive a device change
— any clone gets them.

Anything the dashboard writes only to the local filesystem does NOT survive.
On Streamlit Cloud the container is ephemeral, and a second machine is just a
different clone. That is why RS alerts vanished: the app wrote them locally and
nothing carried them back to the repo.

This module closes that gap by reading and writing a JSON file directly in the
GitHub repo via the Contents API, so state written on one device is visible on
every other one and to the CI alert daemon.

Concurrency
-----------
Use update_json() for anything more than one writer touches. It re-reads the
file, applies a *function* to the fresh copy and writes with that blob's sha;
on a stale-sha rejection it re-reads and re-applies the function. put_json()
writes a precomputed payload, so on a conflict it can only overwrite what
landed in between -- that is how RS alerts used to replace each other.

Read vs write failures
----------------------
A public repo can be read by a token that cannot write to it (read-only PAT,
or a fine-grained PAT whose repository list does not include this repo). So a
successful read proves nothing about writes. Every write records its outcome
in last_status(path), and the UI shows that instead of assuming success.

Configuration (all optional — absent means local-only mode)
-----------------------------------------------------------
    st.secrets["github"]["token"]   or env GITHUB_TOKEN
    st.secrets["github"]["repo"]    or env GITHUB_REPO    (default below)
    st.secrets["github"]["branch"]  or env GITHUB_BRANCH  (default "main")

The token needs `contents: write` on this repo only — a fine-grained PAT is
the right choice. Without one the app keeps working exactly as before, storing
state on the local device, and the UI says so plainly rather than pretending
the data is synced.
"""
import base64
import copy
import json
import os
import time
from typing import Any, Callable, Optional, Tuple

import requests

DEFAULT_REPO = "adi-git01/nifty-dashboard"
API = "https://api.github.com"
TIMEOUT = 15

# The dashboard re-runs its whole script on every interaction; without a cache
# that would mean a GitHub round-trip per widget click.
_CACHE_TTL = 30.0
_cache: dict = {}


def _cfg(key: str, env: str, default: Optional[str] = None) -> Optional[str]:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "github" in st.secrets:
            val = st.secrets["github"].get(key)
            if val:
                return str(val)
    except Exception:
        pass
    return os.environ.get(env) or default


def get_token() -> Optional[str]:
    return _cfg("token", "GITHUB_TOKEN")


def get_repo() -> str:
    return _cfg("repo", "GITHUB_REPO", DEFAULT_REPO)


def get_branch() -> str:
    return _cfg("branch", "GITHUB_BRANCH", "main")


def is_disabled() -> bool:
    """
    Hard off-switch. The EOD workflow sets this so the alert daemon writes only
    the checked-out file and lets git-auto-commit publish it — if the daemon
    also pushed through the API, the same run would write the file twice by two
    different routes and race with its own commit.
    """
    return os.environ.get("REPO_STORE_DISABLED", "").strip() not in ("", "0", "false", "False")


def is_configured() -> bool:
    return bool(get_token()) and not is_disabled()


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {get_token()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def status() -> str:
    """One-line description of where state is being stored, for the UI."""
    if not is_configured():
        return "local-only (this device)"
    return f"synced to {get_repo()}@{get_branch()}"


# ---------------------------------------------------------------------------
_last: dict = {}


def last_status(path: str) -> Optional[dict]:
    """Outcome of the most recent write to `path` from this process: {ok, when, error}."""
    return _last.get(path)


def _explain(code: int, text: str) -> str:
    if code == 401:
        return "401: token invalid or expired -- create a new one"
    if code in (403, 404):
        return (f"{code}: token cannot write to {get_repo()} -- give it Contents: Read and write, "
                f"and include this repo in its repository access")
    if code == 422:
        return f"422: rejected by GitHub ({text[:120]})"
    return f"{code}: {text[:160]}"


def read_json(path: str, use_cache: bool = True) -> Tuple[Optional[Any], Optional[str], bool]:
    """(data, sha, ok). ok is False when the read itself failed; a missing file is ok=True, data None."""
    if not is_configured():
        return None, None, False
    if use_cache:
        hit = _cache.get(path)
        if hit and (time.time() - hit[0]) < _CACHE_TTL:
            return hit[1], hit[2], True
    try:
        r = requests.get(f"{API}/repos/{get_repo()}/contents/{path}",
                         headers=_headers(), params={"ref": get_branch()}, timeout=TIMEOUT)
        if r.status_code == 404:
            _cache[path] = (time.time(), None, None)
            return None, None, True
        r.raise_for_status()
        body = r.json()
        raw = base64.b64decode(body.get("content", "")).decode("utf-8")
        data = json.loads(raw) if raw.strip() else None
        _cache[path] = (time.time(), data, body.get("sha"))
        return data, body.get("sha"), True
    except Exception as e:
        print(f"[repo_store] read {path} failed: {e}")
        return None, None, False


def update_json(path: str, fn: Callable[[Any], Any], message: Any,
                attempts: int = 4) -> Tuple[bool, Any, Optional[str]]:
    """
    Atomic read-modify-write: new = fn(fresh_copy), written against fresh_copy's sha.
    On a concurrent write GitHub rejects the sha; we re-read and call fn again, so
    nothing written in between is lost. fn must be safe to call more than once.
    `message` may be a callable of the new value. Returns (ok, new_value, error).
    """
    if not is_configured():
        return False, None, "not configured"
    err = None
    for i in range(attempts):
        cur, sha, ok = read_json(path, use_cache=False)
        if not ok:
            err = "could not read the current file from GitHub"
            break
        new = fn(copy.deepcopy(cur))
        if new == cur and sha:
            _last[path] = {"ok": True, "when": time.time(), "error": None}
            return True, new, None          # nothing changed: no empty commit
        body = {"message": message(new) if callable(message) else message, "branch": get_branch(),
                "content": base64.b64encode(json.dumps(new, indent=2, default=str).encode()).decode("ascii")}
        if sha:
            body["sha"] = sha
        try:
            r = requests.put(f"{API}/repos/{get_repo()}/contents/{path}",
                             headers=_headers(), json=body, timeout=TIMEOUT)
        except Exception as e:
            err = f"network error: {e}"
            break
        if r.status_code in (200, 201):
            _cache[path] = (time.time(), new, r.json().get("content", {}).get("sha"))
            _last[path] = {"ok": True, "when": time.time(), "error": None}
            return True, new, None
        # 409 = stale sha; 422 = sha missing/mismatched after a concurrent create
        if r.status_code in (409, 422) and i < attempts - 1:
            _cache.pop(path, None)
            time.sleep(0.5 * (i + 1))
            continue
        err = _explain(r.status_code, r.text)
        break
    print(f"[repo_store] write {path} failed: {err}")
    _last[path] = {"ok": False, "when": time.time(), "error": err}
    return False, None, err


_perm: dict = {}


def can_write() -> Optional[bool]:
    """
    Whether the token may push to the repo, from GitHub's own answer (GET /repos
    returns permissions.push for the caller). None when unknown. Cached 10 min.
    """
    if not is_configured():
        return False
    hit = _perm.get("push")
    if hit and time.time() - hit[0] < 600:
        return hit[1]
    try:
        r = requests.get(f"{API}/repos/{get_repo()}", headers=_headers(), timeout=TIMEOUT)
        if r.status_code == 401:
            val = False
        elif r.status_code != 200:
            return None
        else:
            # Fine-grained PATs report push from their Contents permission.
            val = bool((r.json().get("permissions") or {}).get("push"))
    except Exception:
        return None
    _perm["push"] = (time.time(), val)
    return val


def get_json(path: str, use_cache: bool = True) -> Tuple[Optional[Any], Optional[str]]:
    """Return (parsed_json, sha). (None, None) when unconfigured or absent."""
    if not is_configured():
        return None, None

    if use_cache:
        hit = _cache.get(path)
        if hit and (time.time() - hit[0]) < _CACHE_TTL:
            return hit[1], hit[2]

    try:
        r = requests.get(
            f"{API}/repos/{get_repo()}/contents/{path}",
            headers=_headers(), params={"ref": get_branch()}, timeout=TIMEOUT,
        )
        if r.status_code == 404:
            _cache[path] = (time.time(), None, None)
            return None, None
        r.raise_for_status()
        body = r.json()
        raw = base64.b64decode(body.get("content", "")).decode("utf-8")
        data = json.loads(raw) if raw.strip() else None
        sha = body.get("sha")
        _cache[path] = (time.time(), data, sha)
        return data, sha
    except Exception as e:
        print(f"[repo_store] read {path} failed: {e}")
        return None, None


def put_json(path: str, obj: Any, message: str) -> bool:
    """
    Overwrite `path` with a precomputed `obj`. Only for single-writer files: on a
    conflict it overwrites what landed in between. Shared state -> update_json().
    """
    if not is_configured():
        return False

    payload = json.dumps(obj, indent=2, default=str)
    for attempt in (1, 2):
        _, sha = get_json(path, use_cache=False)
        body = {
            "message": message,
            "content": base64.b64encode(payload.encode("utf-8")).decode("ascii"),
            "branch": get_branch(),
        }
        if sha:
            body["sha"] = sha
        try:
            r = requests.put(f"{API}/repos/{get_repo()}/contents/{path}",
                             headers=_headers(), json=body, timeout=TIMEOUT)
            if r.status_code in (200, 201):
                _cache[path] = (time.time(), obj, r.json().get("content", {}).get("sha"))
                return True
            if r.status_code == 409 and attempt == 1:
                _cache.pop(path, None)
                continue  # someone else wrote first — re-read and retry
            print(f"[repo_store] write {path} failed: {r.status_code} {r.text[:200]}")
            return False
        except Exception as e:
            print(f"[repo_store] write {path} error: {e}")
            return False
    return False


def invalidate(path: str) -> None:
    _cache.pop(path, None)
