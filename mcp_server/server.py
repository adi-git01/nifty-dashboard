"""
Alpha Trend MCP server — read-only access to the dashboard's data.

Lets an MCP client (Claude Desktop, Claude Code, OpenCode) query the pipeline
directly instead of being handed pasted CSVs: screen the universe, pull one
stock's metrics, read the portfolio and signal logs, and fetch a Monte Carlo
valuation card or the Phase-0 dossier behind it.

Read-only by construction — no tool writes, so a client cannot alter portfolio
state, alerts or logs through this surface.

    python -m mcp_server.server            # stdio (normal MCP use)
    python -m mcp_server.server --selftest # exercise every tool, no protocol
"""
from __future__ import annotations

import sys
from typing import Optional

from mcp.server.mcpserver import MCPServer

from . import data_access as D

mcp = MCPServer(
    name="alpha-trend",
    instructions=(
        "Read-only access to the Alpha Trend Indian equity dashboard. Data comes "
        "from committed daily snapshots, so every answer carries an as_of date — "
        "quote it, and never present a snapshot figure as a live quote. CompRS is "
        "composite relative strength vs Nifty in percentage points "
        "(0.10*RS5 + 0.50*RS21 + 0.40*RS63); the strategy's entry floor is ~17. "
        "Start with list_datasets to see what exists and how fresh it is."
    ),
)


@mcp.tool(description="List every available dataset with row counts, date ranges "
                      "and freshness. Call this first to see what exists.")
def list_datasets() -> dict:
    return D.list_datasets()


@mcp.tool(description="Dates for which a daily market snapshot exists (YYYY-MM-DD).")
def list_snapshot_dates() -> dict:
    d = D.snapshot_dates()
    return {"count": len(d), "earliest": d[0] if d else None,
            "latest": d[-1] if d else None, "dates": d}


@mcp.tool(description="All stored metrics for one ticker (price, CompRS and its "
                      "RS5/RS21/RS63 components, trend score, PE/ROE/margins, "
                      "52-week range, signals). NSE symbol with or without .NS.")
def get_stock(ticker: str, date: Optional[str] = None) -> dict:
    return D.get_stock(ticker, date)


@mcp.tool(description="Screen the universe. Filter on CompRS, sector, sub-industry, "
                      "position vs the 50-day MA, extension above it, and liquidity "
                      "in Rs crore/day. Corrupt-bar rows (|CompRS| > 200) are excluded.")
def screen(min_comp_rs: Optional[float] = None, max_comp_rs: Optional[float] = None,
           sector: Optional[str] = None, sub_industry: Optional[str] = None,
           above_ma50: Optional[bool] = None, max_ma50_extension: Optional[float] = None,
           min_liquidity_cr: Optional[float] = None, limit: int = 50,
           sort_by: str = "comp_rs", date: Optional[str] = None) -> dict:
    return D.screen(min_comp_rs, max_comp_rs, sector, sub_industry, above_ma50,
                    max_ma50_extension, min_liquidity_cr, limit, sort_by, date)


@mcp.tool(description="OptComp-V22 model portfolio: current holdings, equity-curve "
                      "tail, return and max drawdown, and recent realised trades "
                      "with their exit reasons.")
def get_portfolio() -> dict:
    return D.get_portfolio()


@mcp.tool(description="Signal log entries. kind is one of: ias (institutional "
                      "accumulation), turnaround, rs_divergence, earnings_shock, tc. "
                      "Newest first; optional ISO `since` date filter.")
def get_signals(kind: str = "ias", limit: int = 50,
                since: Optional[str] = None) -> dict:
    return D.get_signals(kind, limit, since)


@mcp.tool(description="Sub-industry rotation matrix for a date (default newest).")
def get_rotation(limit: int = 60, date: Optional[str] = None) -> dict:
    return D.get_rotation(limit, date)


@mcp.tool(description="Monte Carlo intrinsic valuation card for a ticker. If no card "
                      "has been generated, returns the Phase-0 dossier instead: scope "
                      "decision, valuation identity, peer multiples, and the research "
                      "brief naming the fields still needed.")
def get_valuation(ticker: str) -> dict:
    return D.get_valuation(ticker)


def _selftest() -> int:
    import json
    checks = [
        ("list_datasets",      lambda: D.list_datasets()),
        ("list_snapshot_dates", lambda: D.snapshot_dates()),
        ("get_stock",          lambda: D.get_stock("RELIANCE")),
        ("screen",             lambda: D.screen(min_comp_rs=17, limit=5)),
        ("get_portfolio",      lambda: D.get_portfolio()),
        ("get_signals",        lambda: D.get_signals("ias", limit=3)),
        ("get_rotation",       lambda: D.get_rotation(limit=3)),
        ("get_valuation",      lambda: D.get_valuation("CUB")),
    ]
    bad = 0
    for name, fn in checks:
        try:
            json.dumps(fn())
            print(f"  ok    {name}")
        except Exception as e:
            bad += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    tools = [n for n, f in globals().items()
             if callable(f) and not n.startswith("_") and n in
             {"list_datasets", "list_snapshot_dates", "get_stock", "screen",
              "get_portfolio", "get_signals", "get_rotation", "get_valuation"}]
    print(f"\n{len(tools)} tools registered, {bad} failure(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    mcp.run(transport="stdio")
