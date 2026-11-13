"""
Agreement tracker: stocks that SIMBA, the Trend Scanner and Technicals all like.
=================================================================================

Three independent "yes" votes per stock, from the daily master snapshot:
  SIMBA      grade A or A+ with no hard gate hit, or an SI Opportunity in the last 5 sessions
             (utils/simba.py)
  Scanner    the engine's own entry test: dna_signal == BUY (RS vs Nifty > 0, above the 50-day
             average, liquid)
  Technicals fx_action == BUY NOW (Gate 1 + GO state + no overhang; utils/forensics.py)

The log (data/consensus_log.csv) gets one row the first day a stock has 2 or 3 votes, with
the price that day, and is refreshed by every EOD run: current price, return since, best
vote count reached, and the last day it still had 2+ votes. A stock that drops out and
comes back more than REENTRY_DAYS later starts a new row. Written by run_daily_signals.py.

This is a record, not a tested rule: nothing here has been backtested, and the SIMBA vote
only exists from the first engine run that computed simba_* fields.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

LOG_FILE = "data/consensus_log.csv"
COLS = ["first_seen", "ticker", "name", "votes", "max_votes", "simba", "scanner", "technicals", "simba_score",
        "simba_grade", "signal_price", "nifty_signal", "current_price", "return_pct", "nifty_return_pct",
        "excess_pct", "last_agree", "active", "sessions"]
REENTRY_DAYS = 15          # calendar days out of agreement before a comeback is a new row
MIN_VOTES = 2


def votes(df: pd.DataFrame, regime: Optional[Dict[str, Any]] = None, fund: Optional[pd.DataFrame] = None,
          ind: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """One row per stock with the three votes and their count."""
    from utils import simba
    x = df.copy()
    x["scanner"] = x.get("dna_signal", pd.Series("", index=x.index)).astype(str).eq("BUY")
    x["technicals"] = x.get("fx_action", pd.Series("", index=x.index)).astype(str).eq("BUY NOW")
    x["simba"], x["simba_score"], x["simba_grade"] = False, np.nan, ""
    if "simba_stage" in x.columns and x["simba_stage"].notna().any():
        if fund is None:
            fund = simba.fundamentals()
        if ind is None:
            try:
                ind = simba.industry_stages(df)
            except Exception:
                ind = {}
        S = simba.score_table(df, regime or {"regime": "UNKNOWN"}, fund, ind).set_index("ticker")
        si = pd.to_numeric(S["simba_si_since"], errors="coerce") <= 5
        ok = S["grade"].isin(["A+", "A"]) | si
        x["simba"] = x["ticker"].map(ok.to_dict()).eq(True)
        x["simba_score"] = x["ticker"].map(S["score"])
        x["simba_grade"] = x["ticker"].map(S["grade"]).fillna("")
    x["votes"] = x[["simba", "scanner", "technicals"]].sum(axis=1).astype(int)
    return x


def load_log() -> pd.DataFrame:
    if os.path.exists(LOG_FILE):
        try:
            return pd.read_csv(LOG_FILE, parse_dates=["first_seen", "last_agree"]).reindex(columns=COLS)
        except Exception:
            pass
    return pd.DataFrame(columns=COLS)


def update_log(df: pd.DataFrame, snap: pd.Timestamp, nifty_close: Optional[float] = None,
               regime: Optional[Dict[str, Any]] = None, save: bool = True):
    """Add today's new agreements, refresh every logged row. Returns (log, new rows)."""
    log = load_log()
    v = votes(df, regime)
    px = pd.to_numeric(v.get("currentPrice", v.get("price")), errors="coerce")
    v = v.assign(_px=px).set_index("ticker")
    agree = v[(v["votes"] >= MIN_VOTES) & v["_px"].notna()]
    new = []
    for t, r in agree.iterrows():
        prior = log[log.ticker == t]
        if len(prior) and (snap - pd.to_datetime(prior.last_agree).max()).days <= REENTRY_DAYS:
            j = prior.index[-1]
            log.loc[j, ["votes", "simba", "scanner", "technicals"]] = [int(r.votes), bool(r.simba), bool(r.scanner),
                                                                       bool(r.technicals)]
            log.loc[j, "max_votes"] = max(int(log.loc[j, "max_votes"]), int(r.votes))
            log.loc[j, "last_agree"] = snap
            continue
        new.append(dict(first_seen=snap, ticker=t, name=r.get("name", t), votes=int(r.votes), max_votes=int(r.votes),
                        simba=bool(r.simba), scanner=bool(r.scanner), technicals=bool(r.technicals),
                        simba_score=r.simba_score, simba_grade=r.simba_grade, signal_price=float(r._px),
                        nifty_signal=nifty_close, current_price=float(r._px), return_pct=0.0, nifty_return_pct=0.0,
                        excess_pct=0.0, last_agree=snap, active=True, sessions=0))
    new = pd.DataFrame(new, columns=COLS)
    if len(new):
        log = new if log.empty else pd.concat([log, new], ignore_index=True)
    if len(log):
        cur = log.ticker.map(v["_px"])
        ok = cur.notna()
        log.loc[ok, "current_price"] = cur[ok]
        try:                               # splits / bonuses since the stock was added
            from utils import corp_actions
            fac = corp_actions.factor_map(corp_actions.load(), log.ticker, pd.to_datetime(log.first_seen))
            fac.index = log.index
        except Exception:
            fac = 1.0
        log["return_pct"] = (pd.to_numeric(log.current_price) / (pd.to_numeric(log.signal_price) * fac) - 1) * 100
        ns = pd.to_numeric(log.nifty_signal, errors="coerce")
        if nifty_close:
            log["nifty_return_pct"] = (nifty_close / ns - 1) * 100
            log["excess_pct"] = log.return_pct - log.nifty_return_pct
        log["active"] = pd.to_datetime(log.last_agree) >= snap
        still = log.ticker.map(v["votes"]).fillna(0).astype(int)
        log.loc[~log.active, "votes"] = still[~log.active]
        log["sessions"] = [len(pd.bdate_range(d, snap)) - 1 for d in pd.to_datetime(log.first_seen)]
        log = log.sort_values(["first_seen", "max_votes", "ticker"], ascending=[False, False, True])
    if save:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        log.to_csv(LOG_FILE, index=False)
    return log, new


def summary(log: pd.DataFrame) -> pd.DataFrame:
    """Average return, hit rate and count by the best vote count a row reached."""
    if log.empty:
        return pd.DataFrame()
    g = log.assign(r=pd.to_numeric(log.return_pct, errors="coerce"),
                   e=pd.to_numeric(log.excess_pct, errors="coerce")).groupby("max_votes")
    return pd.DataFrame({"stocks": g.size(), "avg_return": g.r.mean().round(1), "median_return": g.r.median().round(1),
                         "winners_pct": (g.r.apply(lambda s: (s > 0).mean()) * 100).round(0),
                         "avg_vs_nifty": g.e.mean().round(1)}).reset_index()
