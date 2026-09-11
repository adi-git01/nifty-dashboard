# Alpha Trend MCP server

Read-only access to the dashboard's data so an MCP client can query the
pipeline directly instead of being handed pasted CSVs.

Everything is served from the committed data files — the repo is this system's
database, so a checkout is a complete point-in-time dataset with no server to
host and no credential to manage.

## Tools

| tool | what it returns |
|---|---|
| `list_datasets` | every dataset, row counts, date ranges — call first |
| `list_snapshot_dates` | dates with a daily market snapshot |
| `get_stock` | all metrics for one ticker (price, CompRS + components, PE/ROE, 52w) |
| `screen` | filter the universe on CompRS, sector, MA50 position/extension, liquidity |
| `get_portfolio` | OptComp-V22 holdings, equity curve, return/drawdown, recent trades |
| `get_signals` | `ias`, `turnaround`, `rs_divergence`, `earnings_shock`, `tc` |
| `get_rotation` | sub-industry rotation matrix |
| `get_valuation` | Monte Carlo card, or the Phase-0 dossier + research brief if none exists |

No tool writes. A client cannot alter portfolio state, alerts or logs here.

## Install

```bash
pip install -r mcp_server/requirements.txt
python -m mcp_server.server --selftest    # exercises every tool, no protocol
```

## Client config

Use the **absolute path to `server.py`**. It works from any working directory,
so no client needs a `cwd` setting:

**Claude Desktop** — `claude_desktop_config.json`
(macOS: `~/Library/Application Support/Claude/`, Windows: `%APPDATA%\Claude\`):

```json
{
  "mcpServers": {
    "alpha-trend": {
      "command": "python",
      "args": ["C:/path/to/nifty-dashboard/mcp_server/server.py"]
    }
  }
}
```

**Claude Code**

```bash
claude mcp add alpha-trend -- python /absolute/path/to/nifty-dashboard/mcp_server/server.py
```

**OpenCode** — same command and args under its MCP section.

On Windows use `python`; on macOS/Linux use `python3` if `python` is not on PATH.

## Notes

- Every response carries an `as_of` date. Snapshots are end-of-day, not live
  quotes — say so when quoting a figure.
- `screen` drops rows with `|CompRS| > 200`, the same corrupt-bar guard the
  dashboard applies at display time.
- CompRS is `0.10*RS5 + 0.50*RS21 + 0.40*RS63` vs Nifty, in percentage points.
  The strategy's entry floor is ~17.
