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

# Work both as a package (`python -m mcp_server.server`) and as a plain script
# (`python /abs/path/to/mcp_server/server.py`). Many MCP clients offer no way to
# set a working directory, and a config that only runs from the repo root is a
# config that silently fails on someone else's machine.
try:
    from . import data_access as D
    from . import insight as I
except ImportError:  # pragma: no cover - direct-script execution
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mcp_server import data_access as D
    from mcp_server import insight as I

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


# ---- decision-level tools (mcp_server/insight.py): what the dashboard concludes ----------------
@mcp.tool(description="START HERE for 'what should I look at today'. Market breadth regime, leading / improving / "
                      "weakening industries, stocks passing the investment funnel, and the user's own holdings that "
                      "need attention plus watchlist names that became buyable.")
def get_daily_brief(limit: int = 10) -> dict:
    return I.get_daily_brief(limit)


@mcp.tool(description="The investment funnel: Industry -> Trend -> Business -> Setup (-> Value if include_value_stage). "
                      "Returns stage counts, stocks passing every stage with stop and target, and near misses that fail "
                      "exactly one stage.")
def get_funnel(limit: int = 25, include_value_stage: bool = False) -> dict:
    return I.get_funnel(limit, include_value_stage)


@mcp.tool(description="One-shot report on a stock: verdict (ACTIONABLE / WATCHLIST / NOT NOW ...), funnel checklist with "
                      "reasons, entry plan (stop, targets, position size), business score, and the full report as markdown. "
                      "Use this for 'should I buy / what about X'.")
def get_stock_report(ticker: str) -> dict:
    return I.get_stock_report(ticker)


@mcp.tool(description="Business fundamentals for a stock: 0-100 score by pillar, every finding with its direction, yearly "
                      "and quarterly tables, and for banks, NBFCs and gold-loan companies the ROA tree (NII, costs, ROA, "
                      "leverage, ROE as % of assets).")
def get_business(ticker: str, years: int = 6) -> dict:
    return I.get_business(ticker, years)


@mcp.tool(description="Industry leadership board: score 0-100, one-month change, breadth, trend status and strongest "
                      "stocks per sub-industry. group filters to LEADING, IMPROVING, WEAKENING or LAGGING.")
def get_industry_board(group: Optional[str] = None, limit: int = 60) -> dict:
    return I.get_industry_board(group, limit)


@mcp.tool(description="The user's own book: each holding's P&L, status (on track, near stop, stop hit, trend weakening), "
                      "exit level and guide; portfolio risk (money at risk if every stop is hit, concentration, missing "
                      "stops); and the watchlist with which funnel stage each stock still lacks. capital is optional total "
                      "capital in rupees.")
def get_my_book(capital: Optional[float] = None) -> dict:
    return I.get_my_book(capital)


@mcp.tool(description="Every scanner signal one stock has triggered (RS divergence, earnings shock, breakout, funnel pass, "
                      "turnaround ...), newest first, with the price then and the return since.")
def get_signals_for(ticker: str) -> dict:
    return I.get_signals_for(ticker)


@mcp.tool(description="Idea lists with funnel context. kind='rs_divergence': stocks that rose on days the Nifty fell. "
                      "kind='earnings_drift': stocks that jumped 4%+ on heavy volume. uptrend_only keeps Scanner BUY stocks.")
def get_strong_names(kind: str = "rs_divergence", days: int = 30, uptrend_only: bool = True, limit: int = 40) -> dict:
    return I.get_strong_names(kind, days, uptrend_only, limit)


@mcp.tool(description="Scorecard: how each scanner's signals have performed against the market, with a verdict per "
                      "scanner. Check this before trusting any signal list.")
def get_scorecard() -> dict:
    return I.get_scorecard()


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
        ("get_daily_brief",    lambda: I.get_daily_brief(5)),
        ("get_funnel",         lambda: I.get_funnel(5)),
        ("get_stock_report",   lambda: I.get_stock_report("SAREGAMA")),
        ("get_business",       lambda: I.get_business("MUTHOOTFIN")),
        ("get_industry_board", lambda: I.get_industry_board("LEADING", 5)),
        ("get_my_book",        lambda: I.get_my_book()),
        ("get_signals_for",    lambda: I.get_signals_for("SAREGAMA")),
        ("get_strong_names",   lambda: I.get_strong_names("earnings_drift", 45, True, 5)),
        ("get_scorecard",      lambda: I.get_scorecard()),
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
              "get_portfolio", "get_signals", "get_rotation", "get_valuation", "get_daily_brief", "get_funnel",
              "get_stock_report", "get_business", "get_industry_board", "get_my_book", "get_signals_for",
              "get_strong_names", "get_scorecard"}]
    print(f"\n{len(tools)} tools registered, {bad} failure(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    mcp.run(transport="stdio")
