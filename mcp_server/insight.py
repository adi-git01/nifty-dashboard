"""
Decision-level tools for the MCP server: what the dashboard concludes, not just its raw tables.

Every function returns plain JSON with an `as_of` date and, where the dashboard has measured how
well something works, an `evidence` note. Nothing here writes, and nothing is fetched from the
network: the daily snapshot, the weekly Screener files and the logs in data/ are the only inputs.

Your own book (holdings and watchlist) is read through utils/positions.py: from the GitHub store
when GITHUB_TOKEN and GITHUB_REPO are set in the server's environment, otherwise from the local
positions.json, which may be stale. The result says which.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd

from . import data_access as D

os.chdir(D.ROOT)                       # the dashboard's modules use paths relative to the repo root

EVIDENCE = {
    "funnel": "The funnel as a whole is not backtested; its passes are logged daily and scored on the scorecard. "
              "Stage 1+2 (Scanner BUY in a leading industry) beat plain Scanner BUY over Feb-Nov 2026.",
    "business": "The business score describes quality; in a 2021-25 test it did not predict next-year returns. "
                "Stocks in an 'investment phase' lagged the median by about 5 pp over the following year.",
    "industry": "Top-fifth industries beat the average stock by about 0.6 pp over the next month on Feb-Oct 2026 data: small, one regime.",
    "exit": "The exit level (50-day average or the regime's trailing stop from the peak) is the rule the exit studies kept.",
    "valuation": "Model output with wide ranges; in the entry/exit study expensive momentum beat cheap, so it is context, not a gate.",
}


def _f(v, nd=1):
    try:
        return None if v is None or pd.isna(v) else round(float(v), nd)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=2)
def _ctx(date: str) -> Dict[str, Any]:
    from utils import funnel, industry_board
    df = D.load_snapshot(date)
    board = industry_board.board(df, industry_board.month_ago_snapshot())
    val = funnel.load_valuation()
    return dict(df=df, board=board, val=val, table=funnel.table(df, board, val))


def _now() -> Dict[str, Any]:
    c = _ctx(D.snapshot_dates()[-1])
    return {**c, "as_of": D.snapshot_dates()[-1]}


def _breadth(df: pd.DataFrame) -> Dict[str, Any]:
    from utils import regime_badge
    b = regime_badge.breadth_regime(df)
    return dict(breadth_regime=b.get("regime"), pct_above_200dma=_f(b.get("pct")),
                note="Breadth regime only: the Nifty regime needs index history the server does not fetch.")


def _pick_rows(P: pd.DataFrame, limit: int) -> List[Dict]:
    cols = ["ticker", "name", "industry", "ind_group", "price", "comp_rs", "technicals", "stop", "risk_pct", "t1", "biz_headline",
            "biz_flags", "quarter", "np_yoy", "val_pct", "val_upside"] + (["missed"] if "missed" in P.columns else [])
    return D._records(P[[c for c in cols if c in P.columns]].round(2), limit)


def get_funnel(limit: int = 25, include_value_stage: bool = False) -> Dict:
    """Stage counts, the stocks passing every stage, and near misses (fail exactly one)."""
    from utils import funnel
    c = _now()
    on = funnel.DEFAULT_ON + (("value",) if include_value_stage else ())
    R = funnel.run(c["table"], on)
    return D._clean(dict(as_of=c["as_of"], stages={k: funnel.WHAT[k] for k in on},
                         counts=R["counts"].to_dict("records"), passing=_pick_rows(R["picks"], limit),
                         near_misses=_pick_rows(R["near"], limit), market=_breadth(c["df"]), evidence=EVIDENCE["funnel"]))


def get_industry_board(group: Optional[str] = None, limit: int = 60) -> Dict:
    """Industry leadership board; group is LEADING, IMPROVING, WEAKENING or LAGGING."""
    c = _now()
    B = c["board"]
    if group:
        B = B[B.group == group.upper()]
    cols = ["industry", "group", "score", "change", "status", "rs_3m", "above_50", "above_200", "near_high", "buy_pct", "members", "leaders"]
    return D._clean(dict(as_of=c["as_of"], rows=D._records(B[cols], limit), groups=c["board"].group.value_counts().to_dict(),
                         evidence=EVIDENCE["industry"]))


def get_stock_report(ticker: str) -> Dict:
    """The one-shot report for a stock: verdict, funnel checklist, industry, trend, entry plan, business, valuation."""
    from utils import stock_report
    c = _now()
    t = D.normalize_ticker(ticker)
    R = stock_report.gather(t, c["df"], c["board"], c["val"], {})
    if not R.get("ok"):
        return dict(as_of=c["as_of"], ticker=t, error=R.get("error", "not found"))
    tr = R["trends"]
    return D._clean(dict(
        as_of=c["as_of"], ticker=t, name=R["name"], price=_f(R["price"], 2), industry=R["industry"],
        verdict=R["verdict"], checklist=[dict(stage=x["stage"], passed=x["ok"], why=x["why"]) for x in R["checks"]],
        entry_plan={k: (_f(v, 2) if isinstance(v, (int, float)) else v) for k, v in R["plan"].items()},
        business=dict(verdict=tr["verdict"], score=tr["score"], direction=tr["direction"], lender_kind=tr.get("lender_kind") or None,
                      investment_phase=(tr["investing"] or {}).get("text"), pillars=tr["pillars"], hard_flags=tr["hard_flags"]),
        report_markdown=stock_report.markdown(R),
        evidence=dict(funnel=EVIDENCE["funnel"], business=EVIDENCE["business"], valuation=EVIDENCE["valuation"])))


def get_business(ticker: str, years: int = 6) -> Dict:
    """Business score with every finding, yearly and quarterly tables, and the ROA tree for a lender."""
    from utils import fund_trends, stock_report
    t = D.normalize_ticker(ticker)
    r = fund_trends.read(t)
    tab = lambda v: {} if v is None or v.empty else D._clean(v.to_dict())
    return D._clean(dict(
        ticker=t, headline=fund_trends.headline(r), score=r["score"], verdict=r["verdict"], direction=r["direction"],
        lender_kind=r.get("lender_kind") or None, pillars=r["pillars"], hard_flags=r["hard_flags"],
        investment_phase=(r["investing"] or {}).get("text"), latest_quarter=r["latest"],
        findings=[dict(pillar=f["pillar"], metric=f["metric"], direction=f["way"], text=f["text"], scored=f["scored"])
                  for f in r["findings"] + r["ownership"]],
        yearly=tab(stock_report.annual_view(fund_trends.annual(t), years)),
        quarterly=tab(stock_report.quarterly_view(fund_trends.quarterly(t))),
        roa_tree=tab(stock_report.roa_tree_view(fund_trends.roa_tree(t), years)),
        evidence=EVIDENCE["business"]))


def get_signals_for(ticker: str) -> Dict:
    """Every scanner signal a stock has triggered, newest first, with the return since."""
    from utils import strong_names
    c = _now()
    t = D.normalize_ticker(ticker)
    px = pd.to_numeric(c["df"].drop_duplicates("ticker").set_index("ticker")["currentPrice"], errors="coerce").to_dict()
    S = strong_names.signals_for(t, px)
    S = S.assign(date=S.date.dt.strftime("%Y-%m-%d")) if len(S) else S
    return D._clean(dict(as_of=c["as_of"], ticker=t, count=len(S), by_signal=S.signal.value_counts().to_dict() if len(S) else {},
                         signals=D._records(S.drop(columns="ticker").round(2)) if len(S) else []))


def get_strong_names(kind: str = "rs_divergence", days: int = 30, uptrend_only: bool = True, limit: int = 40) -> Dict:
    """kind: rs_divergence (rose on days the Nifty fell) or earnings_drift (4%+ jump on 2.5x volume)."""
    from utils import strong_names
    c = _now()
    X = strong_names.rs_divergence(c["table"], days) if kind == "rs_divergence" else strong_names.earnings_drift(c["table"], days)
    if len(X) and uptrend_only:
        X = X[X.s_trend.eq(True)]
    for col in ("first", "latest", "jump_date"):
        if col in X:
            X[col] = pd.to_datetime(X[col]).dt.strftime("%Y-%m-%d")
    drop = [x for x in ("s_industry", "s_trend", "s_business", "s_setup", "first_price", "signal_price") if x in X]
    return D._clean(dict(as_of=c["as_of"], kind=kind, days=days, count=len(X), rows=D._records(X.drop(columns=drop).round(2), limit),
                         evidence="Earnings jumps drifted on only when the stock was already a Scanner BUY; RS divergence is too new to judge."))


def get_scorecard() -> Dict:
    """How each scanner's signals have done against the market (rebuilt nightly)."""
    p = D._p("data", "signal_scorecard.csv")
    if not os.path.exists(p):
        return dict(error="scorecard not built yet")
    S = pd.read_csv(p)
    cols = ["signal", "verdict", "signals", "avg_5d", "avg_21d", "hit_21d", "n_21d", "avg_63d", "hit_63d", "n_63d", "where", "claim"]
    return D._clean(dict(as_of=str(S.as_of.iloc[0]), sessions=int(S.sessions.iloc[0]), rows=D._records(S[[c for c in cols if c in S]]),
                         how_to_read="Average return of signalled stocks minus the median stock, by sessions after the signal. "
                                     "The yardstick is every Scanner BUY stock, every day. One regime, under a year of data."))


def get_my_book(capital: Optional[float] = None) -> Dict:
    """Your holdings (status, exit level, P&L), portfolio risk, and the watchlist with what each stock still lacks."""
    from utils import book_logic as bl, funnel, positions as pos
    c = _now()
    df, known = c["df"], set(c["df"]["ticker"])
    try:
        allp = pos.get_all_positions()
        src = pos.storage_status()
        source = src.get("where") if src.get("configured") else "local positions.json (may be stale; set GITHUB_TOKEN and GITHUB_REPO to read the live book)"
    except Exception as e:
        return dict(as_of=c["as_of"], error=f"could not read positions: {e}")
    hold = [p for p in allp if p.get("status") == "active"]
    watch = [p for p in allp if p.get("status") == "watching"]
    res = {p["ticker"]: (pos.resolve_ticker(p["ticker"], known) or p["ticker"]) for p in hold + watch}
    tk = [t for t in res.values() if t in known]
    T = funnel.table(df[df.ticker.isin(tk)], c["board"], c["val"], business_for=tk).set_index("ticker") if tk else pd.DataFrame()
    by = df.drop_duplicates("ticker").set_index("ticker")
    rows, H = [], []
    for p in hold:
        t = res[p["ticker"]]
        r = by.loc[t] if t in by.index else None
        px = _f(r.get("currentPrice"), 2) if r is not None else None
        e, sl, tg = bl._num(p.get("entry_price")), bl._num(p.get("stop_loss")), bl._num(p.get("target"))
        ma50 = pd.to_numeric(r.get("fiftyDayAverage"), errors="coerce") if r is not None else float("nan")
        g = bl.exit_guide(px, float(ma50) if pd.notna(ma50) else None, bl.peak_since(t, p.get("entry_date"), px), 0.12, sl)
        st_ = bl.status(px, sl, tg, str(r.get("dna_signal")) if r is not None else "", bool(px and pd.notna(ma50) and px < ma50),
                        float(pd.to_numeric(r.get("simba_caution"), errors="coerce") or 0) if r is not None and pd.notna(r.get("simba_caution")) else 0)
        ind = T.at[t, "industry"] if len(T) and t in T.index else None
        rows.append(dict(ticker=t, qty=p.get("quantity"), entry=e, price=px, pnl_pct=_f((px / e - 1) * 100) if px and e else None,
                         stop=sl, target=tg, status=st_, exit_below=_f(g["level"], 2), exit_guide=g["text"], industry=ind,
                         industry_group=T.at[t, "ind_group"] if len(T) and t in T.index else None, entry_date=p.get("entry_date"),
                         notes=p.get("notes") or None))
        H.append({"Stock": t.replace(".NS", ""), "Price": px, "Qty": p.get("quantity") or 1, "Stop": sl, "Exit below": g["level"],
                  "Sector": ind or "Unclassified",
                  "Daily traded": float(pd.to_numeric(r.get("averageVolume"), errors="coerce") * px) if r is not None and px else None})
    K = bl.risk_summary(pd.DataFrame(H), capital) if H else {}
    risk = dict(invested=_f(K["invested"], 0), at_risk_if_all_stops_hit=_f(K["risk"], 0), at_risk_pct_of_capital=_f(K["risk_pct"]),
                largest_position=dict(stock=K["largest"][0], pct_of_capital=_f(K["largest"][1])),
                by_industry_pct={k: _f(v) for k, v in K["by_industry"].items()}, no_stop=K["no_stop"],
                hard_to_exit=K["thin"]) if K else {}
    wl = [dict(ticker=res[p["ticker"]], funnel=bl.funnel_label(T.loc[res[p["ticker"]]] if len(T) and res[p["ticker"]] in T.index else None),
               technicals=T.at[res[p["ticker"]], "technicals"] if len(T) and res[p["ticker"]] in T.index else None,
               price=_f(by.at[res[p["ticker"]], "currentPrice"], 2) if res[p["ticker"]] in by.index else None,
               notes=p.get("notes") or None) for p in watch]
    return D._clean(dict(as_of=c["as_of"], source=source, holdings=rows, risk=risk, watchlist=wl,
                         needs_attention=[x["ticker"] for x in rows if not x["status"].startswith("✅") or x["exit_guide"].startswith("🔴")],
                         evidence=EVIDENCE["exit"],
                         note="Exit levels use a 12% trail (CAUTION regime default); the dashboard uses the live regime's figure."))


def get_daily_brief(limit: int = 10) -> Dict:
    """One call for 'what matters today': market breadth, leading industries, funnel passes, your book."""
    from utils import funnel
    c = _now()
    B = c["board"]
    R = funnel.run(c["table"])
    grp = lambda g: [dict(industry=r.industry, score=_f(r.score, 0), change=_f(r.change, 0)) for r in B[(B.group == g) & ~B.small].head(6).itertuples()]
    book = get_my_book()
    return D._clean(dict(as_of=c["as_of"], market=_breadth(c["df"]),
                         industries=dict(leading=grp("LEADING"), improving=grp("IMPROVING"), weakening=grp("WEAKENING")),
                         funnel=dict(counts=R["counts"].to_dict("records"), passing=_pick_rows(R["picks"], limit)),
                         book=dict(source=book.get("source"), holdings=len(book.get("holdings", [])),
                                   needs_attention=[h for h in book.get("holdings", []) if h["ticker"] in book.get("needs_attention", [])],
                                   watchlist_buyable=[w["ticker"] for w in book.get("watchlist", []) if str(w["funnel"]).startswith("✅")],
                                   risk=book.get("risk")),
                         evidence=EVIDENCE["funnel"]))
