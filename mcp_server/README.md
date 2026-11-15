# Alpha Trend MCP server

Read-only access to the dashboard's data and conclusions for any MCP client (OpenCode, Claude
Desktop / Code, Cursor). The model picks a tool; every number comes from the dashboard's own
files, with the data date attached. Nothing is written and nothing is fetched from the internet.

## Tools

Start with the decision-level ones:

| Tool | Answers |
|---|---|
| `get_daily_brief` | What matters today: breadth, leading industries, funnel passes, your holdings needing attention |
| `get_funnel` | Which stocks pass Industry → Trend → Business → Setup, with stop and target; near misses |
| `get_stock_report` | Should I buy X: verdict, checklist, entry plan, business score, full report as markdown |
| `get_business` | Fundamentals: score by pillar, findings, yearly and quarterly tables, ROA tree for lenders |
| `get_industry_board` | Which industries lead, improve, weaken or lag |
| `get_my_book` | Your holdings (status, exit level), portfolio risk, watchlist readiness |
| `get_signals_for` | Every scanner signal a stock has triggered, with the return since |
| `get_strong_names` | RS-divergence and earnings-drift idea lists with funnel context |
| `get_scorecard` | How each scanner has done against the market |

Raw-data tools: `list_datasets`, `list_snapshot_dates`, `get_stock`, `screen`, `get_portfolio`
(the paper model portfolio), `get_signals`, `get_rotation`, `get_valuation`.

## Run

    pip install -r mcp_server/requirements.txt
    python -m mcp_server.server --selftest     # every tool, no protocol
    python -m mcp_server.server                # stdio, for an MCP client

Keep the repo up to date (`git pull`): the tools read the committed daily snapshot.

## Connect a client

OpenCode (`opencode.json`), with any model you have a key for, e.g. DeepSeek:

    {
      "mcp": {
        "alpha-trend": {
          "type": "local",
          "command": ["python", "/abs/path/to/nifty-dashboard/mcp_server/server.py"],
          "environment": {"GITHUB_TOKEN": "<token>", "GITHUB_REPO": "owner/nifty-dashboard"}
        }
      }
    }

Claude Desktop (`claude_desktop_config.json`):

    {
      "mcpServers": {
        "alpha-trend": {
          "command": "python",
          "args": ["/abs/path/to/nifty-dashboard/mcp_server/server.py"],
          "env": {"GITHUB_TOKEN": "<token>", "GITHUB_REPO": "owner/nifty-dashboard"}
        }
      }
    }

Check the exact config keys against your client's own documentation; they change between versions.

## Your book

`get_my_book` and `get_daily_brief` read your holdings and watchlist. With `GITHUB_TOKEN` and
`GITHUB_REPO` set (the same values as the dashboard's Streamlit secrets) they read the live book
from the `alert-state` branch; without them they fall back to the local `positions.json`, which
may be stale. The result's `source` field says which was used. A token with read access is enough.

## What the model is told

Each result carries an `evidence` note where the dashboard has measured how well something works
(for example: the business score did not predict returns in the 2021-25 test), so the model can
report a signal without overselling it. Snapshot figures are end-of-day, never live quotes.
