"""
Extended coverage: any NSE stock outside the ranked universe, analysed on demand.

Two tiers
  core      the ~1,170 stocks the nightly engine scores and ranks (data/nifty1000_list.csv).
            The funnel, industry board, scanners and scorecard are built on these and do not change.
  extended  every other NSE stock with a market cap of Rs 1,000 cr or more
            (data/extended_universe.csv, rebuilt weekly with fundamentals by
            build_extended_universe.py), plus any NSE stock you type in.

An extended stock gets the same snapshot row as a core stock (price, returns, RS, moving
averages, SIMBA and Technicals fields) computed on request from 14 months of daily bars by the
SAME engine function the nightly run uses, so the Stock Report, Technicals, business score and
valuation all work on it. It is not ranked against the universe and does not appear in the
funnel list, industry board or scanners.

Tracked extras: an extended stock you hold or watch is added to the nightly run (tracked()), so
from the next end-of-day run it is in the daily snapshot like any other stock: My Book prices
it, exit levels and the Telegram brief cover it. Until then with_rows() fills it in on demand.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Optional

import pandas as pd

MIN_MCAP_CR = 1000
MAX_TRACKED = 150
EQ_LIST = "EQUITY_L.csv"
_SYM = re.compile(r"^[A-Z0-9&\-]{1,20}\.NS$")
_BSE_SYM = re.compile(r"^(\d{6}|[A-Z0-9&\-]{1,20})\.BO$")      # BSE-only companies, priced from Yahoo


def norm(t: str) -> str:
    t = str(t or "").strip().upper()
    return t if t.endswith((".NS", ".BO")) else t + ".NS" if t else t


def universe() -> pd.DataFrame:
    """The extended list: Ticker, Company_Name, Sub_Industry, mcap_cr (may be empty before the first weekly build)."""
    p = "data/extended_universe.csv"
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Company_Name", "Sub_Industry", "mcap_cr", "price", "as_of"])


def eq_symbols() -> Dict[str, str]:
    """Every NSE equity-series symbol -> company name, from the exchange's list (EQUITY_L.csv). Cached by file time."""
    try:
        return dict(_eq_cached(os.path.getmtime(EQ_LIST)))
    except Exception:
        return {}


@lru_cache(maxsize=2)
def _eq_cached(mtime: float) -> tuple:
    try:
        e = pd.read_csv(EQ_LIST)
        e.columns = [c.strip() for c in e.columns]
        e = e[e["SERIES"].astype(str).str.strip() == "EQ"]
        return tuple((f"{s.strip()}.NS", n.strip()) for s, n in zip(e["SYMBOL"].astype(str), e["NAME OF COMPANY"].astype(str)))
    except Exception:
        return ()


def known_outside_core(core: Iterable[str]) -> Dict[str, str]:
    """Tickers the dashboard can analyse beyond the core: the extended list and the NSE equity list. -> name"""
    core = set(core)
    out = {t: n for t, n in eq_symbols().items() if t not in core}
    u = universe()
    out.update({t: n for t, n in zip(u.get("Ticker", []), u.get("Company_Name", [])) if t not in core})
    return out


def tracked(core: Optional[Iterable[str]] = None, positions: Optional[List[dict]] = None) -> List[str]:
    """NSE tickers you hold or watch that are outside the core universe (capped, validated)."""
    if core is None:
        from utils.nifty1000_list import TICKERS_1000
        core = TICKERS_1000
    core = set(core)
    if positions is None:
        try:
            from utils import positions as pos
            positions = pos.get_all_positions()
        except Exception:
            positions = []
        try:                                      # price alerts too: an alert on a stock outside the universe never had a price
            from utils.alerts import load_alerts
            positions = list(positions) + [dict(ticker=a.get("ticker"), status="watching") for a in load_alerts()]
        except Exception:
            pass
    # Only real NSE equity symbols outside the universe count. A misspelt holding (FEDERALBANK for
    # FEDERALBNK) resolves to a universe stock and is not an extra; an unknown symbol is skipped
    # rather than downloaded -- trying one stalled every cold start by ~25 s of retries.
    try:
        from utils.positions import resolve_ticker
    except Exception:
        resolve_ticker = lambda t, known: t if t in known else None
    valid = set(eq_symbols()) | set(universe().get("Ticker", []))
    out = []
    for p in positions:
        if p.get("status") not in ("active", "watching"):
            continue
        t = norm(p.get("ticker"))
        if not t or t in core or t in out:
            continue
        if _BSE_SYM.match(t):                     # a BSE code the picker accepted: no NSE list to check it against
            out.append(t)
            continue
        if not _SYM.match(t):
            continue
        if resolve_ticker(t, core):
            continue
        if valid and t not in valid:
            continue
        out.append(t)
    return out[:MAX_TRACKED]


def _fund_frame(tickers: List[str]) -> pd.DataFrame:
    from utils.nifty1000_list import EXTENDED, SUB_INDUSTRY_MAP
    from utils.sector_mapping import consolidate_sector
    names = eq_symbols()
    f = pd.DataFrame({"ticker": tickers})
    f["name"] = [(EXTENDED.get(t) or {}).get("name") or names.get(t) or t.replace(".NS", "") for t in tickers]
    f["sector_granular"] = f["ticker"].map(SUB_INDUSTRY_MAP).fillna("Unknown")
    f["sector"] = f["sector_granular"].map(consolidate_sector)
    f["marketCap"] = [((EXTENDED.get(t) or {}).get("mcap_cr") or float("nan")) * 1e7 for t in tickers]
    return f


def add_tracked(fundamentals: pd.DataFrame, extras: Optional[List[str]] = None) -> pd.DataFrame:
    """Nightly engine: append rows for tracked extras so they are downloaded and scored with the universe."""
    try:
        have = set(fundamentals["ticker"])
        extras = [t for t in (tracked(have) if extras is None else extras) if t not in have]
        if not extras:
            return fundamentals
        print(f"[EXTENDED] adding {len(extras)} tracked stock(s) outside the universe: {' '.join(t[:-3] for t in extras[:20])}", flush=True)
        return pd.concat([fundamentals, _fund_frame(extras)], ignore_index=True)
    except Exception as e:
        print(f"[EXTENDED] tracked extras not added: {e}", flush=True)
        return fundamentals


def rows_for(tickers: Iterable[str]) -> pd.DataFrame:
    """Snapshot rows for stocks outside the snapshot, computed now from daily bars (one Yahoo download)."""
    tickers = [norm(t) for t in tickers if t]
    if not tickers:
        return pd.DataFrame()
    from utils.fast_data_engine import fetch_and_process_market_data
    out = fetch_and_process_market_data(tickers, _fund_frame(tickers), live_mode=True, save=False)
    if out is None or out.empty or "currentPrice" not in out.columns:
        return pd.DataFrame()
    out = out[out["ticker"].isin(tickers) & out["currentPrice"].notna()].copy()
    out["extended"] = True
    return out


def with_rows(df: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    """`df` plus on-demand rows for any of `tickers` it does not already contain."""
    have = set(df["ticker"])
    need = [t for t in dict.fromkeys(norm(x) for x in tickers) if t and t not in have]
    if not need:
        return df
    extra = rows_for(need)
    if extra.empty:
        return df
    for c in ("bar_date", "nifty_close", "nifty_prev_close"):          # the on-demand rows are for the same session
        if c in df.columns and c not in extra.columns:
            extra[c] = df[c].iloc[0]
    return pd.concat([df, extra.reindex(columns=list(dict.fromkeys(list(df.columns) + ["extended"])))], ignore_index=True)
