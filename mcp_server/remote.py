"""
Alpha Trend MCP server over HTTPS, for use from a phone or any remote MCP client.

    MCP_TOKEN=<long random string> python -m mcp_server.remote

The same 17 read-only tools as the local server (mcp_server/server.py), served at

    https://<host>/<MCP_TOKEN>/mcp

The token is part of the address, so the address itself is the password: anyone who has it can
read the dashboard's data and your book. Keep it out of screenshots and chats; to revoke it,
change MCP_TOKEN and restart. The server refuses to start without a token of 24+ characters.
"/" answers "ok" with the data date and nothing else.

Data: the server reads the repo checkout it runs from and refreshes it with `git pull` every
REFRESH_MINUTES (default 30), so it follows the nightly end-of-day commit. Set GITHUB_TOKEN and
GITHUB_REPO to read the live book (holdings and watchlist); a read-only token is enough.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

try:
    from . import data_access as D
    from .server import mcp
except ImportError:  # pragma: no cover - direct-script execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mcp_server import data_access as D
    from mcp_server.server import mcp

MIN_TOKEN = 24


def token() -> str:
    t = os.environ.get("MCP_TOKEN", "").strip()
    if len(t) < MIN_TOKEN or not t.replace("-", "").replace("_", "").isalnum():
        raise SystemExit(f"MCP_TOKEN must be set to {MIN_TOKEN}+ letters, digits, '-' or '_' (it is the password in the URL).")
    return t


def refresh() -> bool:
    """Fast-forward the checkout to the latest commit and drop cached snapshots. True if anything changed."""
    try:
        before = subprocess.run(["git", "rev-parse", "HEAD"], cwd=D.ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
        subprocess.run(["git", "pull", "--ff-only", "--quiet"], cwd=D.ROOT, capture_output=True, text=True, timeout=180)
        after = subprocess.run(["git", "rev-parse", "HEAD"], cwd=D.ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        print(f"[refresh] failed: {e}", file=sys.stderr)
        return False
    if before and after and before != after:
        try:
            from . import insight
            insight._ctx.cache_clear()
        except Exception:
            pass
        print(f"[refresh] {before[:7]} -> {after[:7]}", file=sys.stderr)
        return True
    return False


def _refresher(minutes: float) -> None:
    while True:
        time.sleep(max(60.0, minutes * 60))
        refresh()


def build_app(tok: str):
    """The Starlette app: MCP at /<token>/mcp, a bare health answer at /."""
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.responses import PlainTextResponse

    # behind a host's proxy the Host header is the public name, so the localhost-only check is off;
    # access is controlled by the secret path instead
    app = mcp.streamable_http_app(streamable_http_path=f"/{tok}/mcp", stateless_http=True, json_response=True, host="0.0.0.0",
                                  transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))

    async def health(_request):
        try:
            asof = D.snapshot_dates()[-1]
        except Exception:
            asof = "no data"
        return PlainTextResponse(f"ok — data as of {asof}")

    app.add_route("/", health, methods=["GET"])
    return app


def main() -> None:
    import uvicorn
    tok = token()
    refresh()
    threading.Thread(target=_refresher, args=(float(os.environ.get("REFRESH_MINUTES", "30")),), daemon=True).start()
    port = int(os.environ.get("PORT", "7860"))
    print(f"[remote] serving on :{port} at /<MCP_TOKEN>/mcp", file=sys.stderr)
    uvicorn.run(build_app(tok), host="0.0.0.0", port=port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
