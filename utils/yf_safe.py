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
import time
import pandas as pd
import yfinance as yf

DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 8  # seconds; multiplied by attempt number for a simple backoff


def safe_history(ticker: str, retries=DEFAULT_RETRIES, backoff=DEFAULT_BACKOFF, **kwargs) -> pd.DataFrame:
    """
    yf.Ticker(ticker).history(**kwargs) with retry-with-backoff.
    Never raises — returns an empty DataFrame if every attempt fails, so
    callers can keep their existing `if df.empty:` handling unchanged.
    """
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
                present = set(data.columns.get_level_values(0))
                have = sum(1 for t in tickers_list if t in present and not data[t].dropna(how='all').empty)
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
