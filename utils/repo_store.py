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
Every write is a read-modify-write against the blob's current sha. GitHub
rejects a stale sha with 409, and we retry once against the fresh copy rather
than clobbering whatever landed in between (the EOD daemon advancing alert
state, or the same user on another device).

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
import json
import os
import time
from typing import Any, Optional, Tuple

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
    Read-modify-write `obj` to `path`. Retries once on a 409 sha conflict so a
    concurrent write (CI daemon, or this user on another device) is not lost.
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
