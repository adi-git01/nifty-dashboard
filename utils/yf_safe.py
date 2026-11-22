"""
Resilient wrappers around yfinance calls.

Yahoo Finance intermittently rate-limits, 401s, or times out — especially
under GitHub Actions' shared IP ranges. Every pipeline script previously made
these calls with zero retries, so a single transient blip could: abort an
entire day's engine run (turnaround_screener.py, dna3_current_portfolio.py
used to just return/abort), silently fall back to a default regime
(trading_engine.py), or worst of all, return a partially-empty dataset that
still gets written to disk and corrupts everything downstream — this is the
root cause behind the Jul-8 breadth/mood zero-row incident and the -27%
phantom equity loss earlier: a degraded fetch wasn't treated as a failure at
all, just quietly propagated.

This module centralizes retry-with-backoff (and, for bulk downloads, a
coverage check) so every call site gets the same resilience instead of each
reinventing — or skipping — it.
"""
import os
import time
import pandas as pd
import yfinance as yf

DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 8  # seconds; multiplied by attempt number for a simple backoff

# ---------------------------------------------------------------------------------------------------
# NSE first. Daily bars for NSE stocks (.NS) and the Nifty 50 / Nifty 500 indices come from NSE's own
# files (utils/nse_prices.py) when the local store holds the latest session; Yahoo is asked only for
# what NSE cannot serve: intraday, live (source="yahoo"), US / other tickers, or a store that is behind.
# PRICE_SOURCE=yahoo switches NSE off everywhere (the one-line rollback).
# ---------------------------------------------------------------------------------------------------
_SOURCE_LOG = {"nse": 0, "yahoo": 0}


def _nse_on(source) -> bool:
    if (source or "").lower() == "yahoo" or os.environ.get("PRICE_SOURCE", "nse").strip().lower() == "yahoo":
        return False
    try:
        from utils import nse_prices
        return nse_prices.is_current()
    except Exception:
        return False


def _window(kwargs):
    """(start, end) for a daily request, or None when NSE cannot serve its shape (intraday, odd period)."""
    iv = str(kwargs.get("interval", "1d")).lower()
    if iv not in ("1d", "1day"):
        return None
    start, end = kwargs.get("start"), kwargs.get("end")
    per = kwargs.get("period")
    if per and not start:
        per = str(per).lower()
        today = pd.Timestamp.today().normalize()
        if per == "max":
            start = ("max", None)
        elif per == "ytd":
            start = pd.Timestamp(today.year, 1, 1)
        elif per == "mtd":
            start = pd.Timestamp(today.year, today.month, 1)
        elif per.endswith("d"):
            start = ("sessions", int(per[:-1]))
        elif per.endswith("mo"):
            start = today - pd.DateOffset(months=int(per[:-2]))
        elif per.endswith("y"):
            start = today - pd.DateOffset(years=int(per[:-1]))
        else:
            return None
    return start, end


def _from_nse(tickers, kwargs, source=None):
    """{ticker: frame} NSE can serve for this request ({} when off or not applicable)."""
    served = [t for t in tickers if str(t).endswith(".NS") or str(t).startswith("^")]
    if not served or not _nse_on(source):
        return {}
    w = _window(kwargs)
    if w is None:
        return {}
    start, end = w
    if isinstance(start, tuple) and start[0] == "max":
        return {}
    try:
        from utils import nse_prices
        sess = start[1] if isinstance(start, tuple) else None
        if start is not None and not sess:
            first_dt = nse_prices.first_date("stocks")
            if first_dt is not None and pd.Timestamp(start) < first_dt - pd.Timedelta(days=45):
                return {}
        out = nse_prices.history(served, start=None if sess or start is None else start, end=end)
    except Exception as e:
        print(f"[yf_safe] NSE store unavailable ({type(e).__name__}: {e}); using Yahoo", flush=True)
        return {}
    keep_actions = bool(kwargs.get("actions", False))
    res = {}
    for t, d in out.items():
        if sess:
            d = d.tail(sess)
        cols = ["Open", "High", "Low", "Close", "Volume"]
        if kwargs.get("auto_adjust", True) is False:
            d = d.assign(**{"Adj Close": d["Close"]})
            cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        if keep_actions:
            cols += ["Dividends", "Stock Splits"]
        res[t] = d[cols]
    return res


def source_counts() -> dict:
    """How many tickers each source served in this process (printed by the engine)."""
    return dict(_SOURCE_LOG)


def safe_history(ticker: str, retries=DEFAULT_RETRIES, backoff=DEFAULT_BACKOFF, source=None, **kwargs) -> pd.DataFrame:
    """
    Daily bars for one ticker: NSE's store first (see above), else yf.Ticker(ticker).history(**kwargs)
    with retry-with-backoff. Never raises — returns an empty DataFrame if every attempt fails, so
    callers can keep their existing `if df.empty:` handling unchanged.
    """
    kw = {"auto_adjust": True, "actions": True, **kwargs}           # Ticker.history's own defaults
    got = _from_nse([ticker], kw, source)
    if ticker in got and len(got[ticker]):
        _SOURCE_LOG["nse"] += 1
        return got[ticker]
    _SOURCE_LOG["yahoo"] += 1
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.Ticker(ticker).history(**kwargs)
            if df is not None and not df.empty:
                return df
            last_err = "empty result"
        except Exception as e:
            last_err = e
        if attempt < retries:
            wait = backoff * attempt
            print(f"[yf_safe] {ticker}.history() attempt {attempt}/{retries} failed ({last_err}); retrying in {wait}s...", flush=True)
            time.sleep(wait)
    print(f"[yf_safe] {ticker}.history() failed after {retries} attempts: {last_err}", flush=True)
    return pd.DataFrame()


def safe_download(tickers, retries=DEFAULT_RETRIES, backoff=DEFAULT_BACKOFF, min_coverage=0.5, **kwargs):
    """
    yf.download(tickers, **kwargs) with retry-with-backoff AND a coverage
    check: if fewer than `min_coverage` fraction of tickers came back with
    any usable data, the attempt is treated as failed and retried rather
    than silently accepted. Returns the best-covered attempt's result
    (never raises) so callers keep their existing empty/partial-data
    handling — but callers should still check the returned coverage-worthy
    data size before trusting it fully after all retries are exhausted.
    """
    source = kwargs.pop("source", None)
    single = isinstance(tickers, str)
    tickers_list = list(tickers) if not single else [tickers]
    got = _from_nse(tickers_list, {"auto_adjust": True, **kwargs}, source)
    if got:
        _SOURCE_LOG["nse"] += len(got)
        rest = [t for t in tickers_list if t not in got]
        nse_frame = _shape(got, tickers_list if not rest else list(got), kwargs.get("group_by", "column"), single and not rest)
        if not rest:
            print(f"[yf_safe] {len(got)}/{len(tickers_list)} tickers from NSE's files (no Yahoo call)", flush=True)
            return nse_frame
        print(f"[yf_safe] {len(got)} tickers from NSE; {len(rest)} not in NSE's files go to Yahoo: "
              f"{' '.join(map(str, rest[:15]))}{' …' if len(rest) > 15 else ''}", flush=True)
        _SOURCE_LOG["yahoo"] += len(rest)
        ya = _safe_download_yahoo(rest, retries, backoff, min_coverage=0.0, **kwargs)
        return _merge(nse_frame, ya, kwargs.get("group_by", "column"), rest)
    _SOURCE_LOG["yahoo"] += len(tickers_list)
    return _safe_download_yahoo(tickers if single else tickers_list, retries, backoff, min_coverage, **kwargs)


def _shape(got: dict, order, group_by: str, flat: bool) -> pd.DataFrame:
    """yf.download's layout: flat for one ticker, else (ticker, field) with group_by='ticker', (field, ticker) otherwise."""
    if flat:
        return next(iter(got.values()))
    order = [t for t in order if t in got]
    df = pd.concat({t: got[t] for t in order}, axis=1)
    if group_by != "ticker":
        df = df.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0, sort_remaining=False)
    return df.sort_index()


def _merge(nse_frame: pd.DataFrame, ya, group_by: str, rest) -> pd.DataFrame:
    if ya is None or getattr(ya, "empty", True):
        return nse_frame
    if not isinstance(ya.columns, pd.MultiIndex):                    # Yahoo returned one ticker flat
        ya = pd.concat({rest[0]: ya}, axis=1)
        if group_by != "ticker":
            ya = ya.swaplevel(0, 1, axis=1)
    if getattr(ya.index, "tz", None) is not None:
        ya = ya.copy()
        ya.index = ya.index.tz_localize(None)
    return pd.concat([nse_frame, ya], axis=1).sort_index()


def _safe_download_yahoo(tickers, retries=DEFAULT_RETRIES, backoff=DEFAULT_BACKOFF, min_coverage=0.5, **kwargs):
    tickers_list = list(tickers) if not isinstance(tickers, str) else [tickers]
    total = max(len(tickers_list), 1)
    best_result, best_coverage = None, -1.0

    for attempt in range(1, retries + 1):
        try:
            data = yf.download(tickers_list, progress=False, **kwargs)
        except Exception as e:
            print(f"[yf_safe] bulk download attempt {attempt}/{retries} raised ({e})", flush=True)
            data = None

        coverage = 0.0
        if data is not None and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                lvl0 = set(data.columns.get_level_values(0))
                lvl1 = set(data.columns.get_level_values(1))
                if any(t in lvl0 for t in tickers_list):
                    have = sum(1 for t in tickers_list if t in lvl0 and not data[t].dropna(how='all').empty)
                elif any(t in lvl1 for t in tickers_list):
                    have = sum(1 for t in tickers_list if t in lvl1 and not data.xs(t, level=1, axis=1).dropna(how='all').empty)
                else:
                    have = 0
            else:
                have = 1 if not data.dropna(how='all').empty else 0
            coverage = have / total

        print(f"[yf_safe] bulk download attempt {attempt}/{retries}: "
              f"{coverage*100:.0f}% coverage ({int(round(coverage*total))}/{total} tickers)", flush=True)

        if coverage > best_coverage:
            best_result, best_coverage = data, coverage

        if coverage >= min_coverage:
            return data

        if attempt < retries:
            wait = backoff * attempt
            print(f"[yf_safe] coverage below {min_coverage*100:.0f}% threshold; retrying in {wait}s...", flush=True)
            time.sleep(wait)

    print(f"[yf_safe] bulk download gave up after {retries} attempts; "
          f"best coverage {best_coverage*100:.0f}% — caller must validate before use.", flush=True)
    return best_result
