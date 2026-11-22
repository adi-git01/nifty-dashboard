"""
Funnel log: a daily record of what the Investment Funnel passed, so it can be judged later.

data/funnel_log.csv gets one row the first day a stock
  PASS   passes Industry, Trend, Business and Setup (the default stages), or
  NEAR   fails exactly one of them (the missed stage is recorded)
with the price that day. A stock that leaves and returns more than REENTRY_DAYS later starts a
new row; a NEAR stock that later passes gets its own PASS row. Written by run_daily_signals.py
(EOD). The Signal Lab scorecard measures both kinds against the market, so the funnel earns or
loses its place on evidence. Nothing here is back-filled: the business stage needs fundamentals
as they stood on the day.
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from utils import funnel, industry_board

LOG_FILE = "data/funnel_log.csv"
COLS = ["first_seen", "ticker", "name", "kind", "missed", "industry", "ind_group", "signal_price", "biz_score", "technicals",
        "risk_pct", "last_seen", "active"]
REENTRY_DAYS = 15


def load() -> pd.DataFrame:
    if os.path.exists(LOG_FILE):
        try:
            return pd.read_csv(LOG_FILE, parse_dates=["first_seen", "last_seen"]).reindex(columns=COLS)
        except Exception:
            pass
    return pd.DataFrame(columns=COLS)


def update(df: pd.DataFrame, snap: pd.Timestamp, board: Optional[pd.DataFrame] = None, val: Optional[pd.DataFrame] = None,
           save: bool = True):
    """Add today's new passes and near misses, refresh last_seen / active. Returns (log, new rows)."""
    snap = pd.Timestamp(snap).normalize()
    if board is None:
        board = industry_board.board(df, industry_board.month_ago_snapshot(current_df=df))
    R = funnel.run(funnel.table(df, board, funnel.load_valuation() if val is None else val))
    today = pd.concat([R["picks"].assign(kind="PASS", missed=""), R["near"].assign(kind="NEAR")], ignore_index=True)
    log = load()
    new = []
    for r in today.itertuples():
        prior = log[(log.ticker == r.ticker) & (log.kind == r.kind)]
        last_dt = pd.to_datetime(prior.last_seen, errors="coerce") if len(prior) else pd.Series(dtype="datetime64[ns]")
        if last_dt.notna().any():
            j = last_dt.idxmax()
            last_seen = pd.Timestamp(last_dt.loc[j])
            if (snap - last_seen).days <= REENTRY_DAYS:
                if snap >= last_seen:
                    log.loc[j, "last_seen"] = snap
                    if r.kind == "NEAR":
                        log.loc[j, "missed"] = r.missed
                continue
        new.append(dict(first_seen=snap, ticker=r.ticker, name=r.name, kind=r.kind, missed=r.missed, industry=r.industry,
                        ind_group=r.ind_group, signal_price=float(r.price), biz_score=r.biz_score, technicals=r.technicals,
                        risk_pct=r.risk_pct, last_seen=snap, active=True))
    new = pd.DataFrame(new, columns=COLS)
    if len(new):
        log = new if log.empty else pd.concat([log, new], ignore_index=True)
    if len(log):
        latest_seen = max(snap, pd.to_datetime(log.last_seen, errors="coerce").max())
        log["active"] = pd.to_datetime(log.last_seen, errors="coerce") >= latest_seen
        log = log.sort_values(["first_seen", "kind", "ticker"], ascending=[False, False, True]).reset_index(drop=True)
    if save:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        log.to_csv(LOG_FILE, index=False)
    return log, new
