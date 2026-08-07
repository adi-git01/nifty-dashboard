"""
Bi-Weekly Fundamentals Fetcher
==============================
Fetches PE, ROE, margins, growth rates and earnings quality for all Nifty 1000
stocks via yfinance .info + financials calls. Saves to data/fundamentals_cache.csv
which is committed to the repo and merged into the daily parquet on every
Streamlit load — eliminating the 75-second per-restart fetch cost.

Run schedule: 1st and 15th of each month via GitHub Actions.
Can also be triggered manually: python fetch_fundamentals.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

# ── Sector mapping (same as data_engine.py, without Streamlit dependency) ──
try:
    from utils.nifty1000_list import TICKERS_1000, SUB_INDUSTRY_MAP as NIFTY1000_SECTOR_MAP
except ImportError:
    TICKERS_1000 = []
    NIFTY1000_SECTOR_MAP = {}
try:
    from utils.nifty500_list import SECTOR_MAP as NIFTY500_SECTOR_MAP
except ImportError:
    NIFTY500_SECTOR_MAP = {}
try:
    from utils.sector_mapping import consolidate_sector
except ImportError:
    consolidate_sector = lambda x: x
from utils.atomic_io import atomic_to_csv

OUTPUT_PATH = "data/fundamentals_cache.csv"

# Refuse to publish a cache that covers less of the universe than this.
MIN_COVERAGE = 0.60


def _fmt(value, label, pct=False):
    """Progress-line formatter that cannot raise. Logging must never be able
    to kill the fetch — that is the failure this whole module just suffered."""
    v = _num(value)
    if v is None:
        return f"{label}=N/A"
    return f"{label}={v * 100:.1f}%" if pct else f"{label}={v:.1f}"


def _num(value):
    """
    Coerce a yfinance field to a real float, or None.

    Yahoo returns the *strings* 'Infinity' / '-Infinity' for undefined ratios
    (typically PE on a zero-EPS company). Those are truthy, so `if pe:` passes
    and every downstream arithmetic or format op then explodes on a str. That
    is exactly what broke this script: a progress-line f-string, `f"PE=
    {data['pe']:.1f}"`, raised ValueError at ticker 287 and killed an 8-minute
    run — discarding 286 good fetches and never writing the file. Three
    consecutive scheduled refreshes died this way.

    Same sentinel class already handled in data_engine.py; centralise it here
    so no field can carry a string into the cache.
    """
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if s.lower() in ("infinity", "-infinity", "inf", "-inf", "nan", "none", ""):
            return None
        try:
            value = float(s)
        except ValueError:
            return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def fetch_one(ticker: str) -> dict | None:
    """
    Fetches fundamental data for a single ticker.
    Mirrors data_engine.get_stock_info() without the @st.cache_data decorator.
    """
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        # ── Sector (nifty1000 sub-industry takes priority) ───────────────────
        granular = (NIFTY1000_SECTOR_MAP.get(ticker)
                    or NIFTY500_SECTOR_MAP.get(ticker)
                    or info.get("sector", "Unknown")
                    or "Unknown")
        broad    = consolidate_sector(granular)

        # ── PE ──────────────────────────────────────────────────────────────
        # Every value below goes through _num() before it is used in arithmetic
        # or stored — Yahoo's 'Infinity' string sentinel must never get past here.
        pe = _num(info.get("trailingPE")) or _num(info.get("pe")) or _num(info.get("priceToEarnings"))
        if pe is None:
            price = _num(info.get("currentPrice")) or _num(info.get("previousClose")) or 0
            eps   = _num(info.get("trailingEps")) or _num(info.get("forwardEps"))
            if price and eps and eps != 0:
                pe = price / eps

        forward_pe = _num(info.get("forwardPE"))
        if forward_pe is None:
            price   = _num(info.get("currentPrice")) or _num(info.get("previousClose")) or 0
            fwd_eps = _num(info.get("forwardEps"))
            if price and fwd_eps and fwd_eps != 0:
                forward_pe = price / fwd_eps

        # ── PEG ─────────────────────────────────────────────────────────────
        peg = _num(info.get("pegRatio")) or _num(info.get("trailingPegRatio"))
        if peg is None and pe:
            g = _num(info.get("earningsGrowth")) or _num(info.get("revenueGrowth"))
            if g and g != 0:
                peg = pe / (g * 100)

        # ── ROE / ROA ────────────────────────────────────────────────────────
        roe = _num(info.get("returnOnEquity")) or _num(info.get("roe"))
        roa = _num(info.get("returnOnAssets")) or _num(info.get("roa"))

        if roe is None:
            try:
                bs  = stock.balance_sheet
                inc = stock.financials
                if not bs.empty and not inc.empty:
                    eq = next((bs.loc[k].iloc[0] for k in ['Stockholders Equity'] if k in bs.index), None)
                    ni = next((inc.loc[k].iloc[0] for k in ['Net Income'] if k in inc.index), None)
                    if eq and eq != 0 and ni is not None:
                        roe = ni / eq
            except Exception:
                pass

        if roa is None:
            try:
                bs  = stock.balance_sheet
                inc = stock.financials
                if not bs.empty and not inc.empty:
                    ta = next((bs.loc[k].iloc[0] for k in ['Total Assets'] if k in bs.index), None)
                    ni = next((inc.loc[k].iloc[0] for k in ['Net Income'] if k in inc.index), None)
                    if ta and ta != 0 and ni is not None:
                        roa = ni / ta
            except Exception:
                pass

        # ── Earnings Quality (OCF / Net Income) ─────────────────────────────
        earnings_quality = None
        ocf = info.get("operatingCashflow")
        ni  = info.get("netIncomeToCommon")
        if ocf and ni and ni != 0:
            earnings_quality = ocf / ni
        else:
            try:
                cf  = stock.cashflow
                inc = stock.financials
                if not cf.empty and not inc.empty:
                    ocf_val = next((cf.loc[k].iloc[0] for k in ['Operating Cash Flow'] if k in cf.index), None)
                    ni_val  = next((inc.loc[k].iloc[0] for k in ['Net Income'] if k in inc.index), None)
                    if ocf_val and ni_val and ni_val != 0:
                        earnings_quality = ocf_val / ni_val
            except Exception:
                pass

        # ── Earnings Trend (YoY Net Income growth) ──────────────────────────
        earnings_trend = None
        try:
            inc = stock.financials
            if not inc.empty and 'Net Income' in inc.index:
                ni_series = inc.loc['Net Income'].dropna()
                if len(ni_series) >= 2:
                    curr = ni_series.iloc[0]
                    prev = ni_series.iloc[1]
                    if prev and prev != 0:
                        earnings_trend = (curr - prev) / abs(prev)
        except Exception:
            pass

        # ── Profit Margins (fallback) ────────────────────────────────────────
        profit_margins = info.get("profitMargins")
        if profit_margins is None:
            try:
                inc = stock.financials
                if not inc.empty:
                    ni_key  = next((k for k in ['Net Income', 'Net Income Common Stockholders'] if k in inc.index), None)
                    rev_key = next((k for k in ['Total Revenue', 'Operating Revenue'] if k in inc.index), None)
                    if ni_key and rev_key:
                        ni_v  = inc.loc[ni_key].iloc[0]
                        rev_v = inc.loc[rev_key].iloc[0]
                        if rev_v and rev_v != 0:
                            profit_margins = ni_v / rev_v
            except Exception:
                pass

        return {
            "ticker":                    ticker,
            "name":                      info.get("longName", ticker),
            "sector":                    broad,
            "sector_granular":           granular,
            "pe":                        _num(pe),
            "forwardPE":                 _num(forward_pe),
            "pegRatio":                  _num(peg),
            "pb":                        _num(info.get("priceToBook")),
            "roe":                       _num(roe),
            "roa":                       _num(roa),
            "profitMargins":             _num(profit_margins),
            "grossMargins":              _num(info.get("grossMargins")),
            "operatingMargins":          _num(info.get("operatingMargins")),
            "ebitdaMargins":             _num(info.get("ebitdaMargins")),
            "revenueGrowth":             _num(info.get("revenueGrowth")),
            "earningsGrowth":            _num(info.get("earningsGrowth")),
            "earningsQuarterlyGrowth":   _num(info.get("earningsQuarterlyGrowth")),
            "debtToEquity":              _num(info.get("debtToEquity")),
            "marketCap":                 _num(info.get("marketCap")),
            "beta":                      _num(info.get("beta")),
            "earningsQuality":           _num(earnings_quality),
            "earningsTrend":             _num(earnings_trend),
            "fund_last_updated":         datetime.now().strftime("%Y-%m-%d"),
        }

    except Exception as e:
        print(f"  FAILED {ticker}: {e}")
        return None


def main():
    os.makedirs("data", exist_ok=True)
    tickers = TICKERS_1000
    total   = len(tickers)
    results = []

    print(f"[FUND] Fetching fundamentals for {total} stocks (sequential, 0.6s delay)...")
    print(f"[FUND] Estimated time: ~{total * 0.6 / 60:.0f} minutes")

    failed = []
    for i, ticker in enumerate(tickers, 1):
        # Nothing inside this loop may abort the run. An 8-minute sequential
        # fetch is far too expensive to throw away over one malformed field.
        try:
            data = fetch_one(ticker)
        except Exception as e:
            data = None
            print(f"  FAILED {ticker}: {e}")

        if data:
            results.append(data)
            print(f"  [{i:4d}/{total}] {ticker:<20} "
                  f"{_fmt(data.get('pe'), 'PE')}  {_fmt(data.get('roe'), 'ROE', pct=True)}")
        else:
            failed.append(ticker)
            print(f"  [{i:4d}/{total}] {ticker:<20} SKIPPED")

        time.sleep(0.6)  # conservative: avoids Yahoo Finance rate-limit on shared CI IPs

    if not results:
        print("[FUND] ERROR: No data fetched. Aborting — previous cache left intact.")
        raise SystemExit(1)

    df = pd.DataFrame(results)

    # Top-up, never shrink. A ticker that failed this run keeps its previous
    # (stale but real) row rather than losing its fundamentals entirely — the
    # scanner degrades to old data instead of to nothing.
    carried = 0
    if os.path.exists(OUTPUT_PATH):
        try:
            old = pd.read_csv(OUTPUT_PATH)
            keep = old[~old["ticker"].isin(df["ticker"])]
            if len(keep):
                df = pd.concat([df, keep], ignore_index=True)
                carried = len(keep)
        except Exception as e:
            print(f"[FUND] Could not merge previous cache ({e}) — writing fresh rows only")

    # Defence in depth: coerce every numeric column at the boundary, so nothing
    # non-numeric can reach the cache regardless of which path produced it —
    # including rows carried over from an older, pre-fix cache file.
    _text = {"ticker", "name", "sector", "sector_granular", "fund_last_updated"}
    for col in df.columns:
        if col not in _text:
            df[col] = (pd.to_numeric(df[col], errors="coerce")
                         .replace([np.inf, -np.inf], np.nan))

    coverage = len(results) / max(total, 1)
    print(f"\n[FUND] Fetched {len(results)}/{total} ({coverage:.1%}); "
          f"{len(failed)} failed, {carried} row(s) carried over from the previous cache")
    if coverage < MIN_COVERAGE:
        print(f"[FUND] ERROR: coverage {coverage:.1%} below the {MIN_COVERAGE:.0%} floor — "
              f"refusing to publish. Previous cache left intact.")
        raise SystemExit(1)

    atomic_to_csv(df, OUTPUT_PATH, index=False)
    print(f"[FUND] Saved {len(df)} rows → {OUTPUT_PATH}")
    print(f"[FUND] PE coverage:  {df['pe'].notna().sum()}/{len(df)}")
    print(f"[FUND] ROE coverage: {df['roe'].notna().sum()}/{len(df)}")
    print(f"[FUND] NPM coverage: {df['profitMargins'].notna().sum()}/{len(df)}")
    if failed:
        print(f"[FUND] Failed tickers ({len(failed)}): {failed[:25]}"
              + (" ..." if len(failed) > 25 else ""))


if __name__ == "__main__":
    main()
