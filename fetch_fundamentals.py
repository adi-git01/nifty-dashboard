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
    from utils.nifty1000_list import TICKERS_1000
    from utils.nifty500_list import SECTOR_MAP
    from utils.sector_mapping import consolidate_sector
except ImportError:
    SECTOR_MAP = {}
    consolidate_sector = lambda x: x
    from utils.nifty1000_list import TICKERS_1000

OUTPUT_PATH = "data/fundamentals_cache.csv"


def fetch_one(ticker: str) -> dict | None:
    """
    Fetches fundamental data for a single ticker.
    Mirrors data_engine.get_stock_info() without the @st.cache_data decorator.
    """
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        # ── Sector ──────────────────────────────────────────────────────────
        granular = SECTOR_MAP.get(ticker, info.get("sector", "Unknown")) or "Unknown"
        broad    = consolidate_sector(granular)

        # ── PE ──────────────────────────────────────────────────────────────
        pe = info.get("trailingPE") or info.get("pe") or info.get("priceToEarnings")
        if pe is None:
            price = info.get("currentPrice") or info.get("previousClose") or 0
            eps   = info.get("trailingEps") or info.get("forwardEps")
            if price and eps and eps != 0:
                pe = price / eps

        forward_pe = info.get("forwardPE")
        if forward_pe is None:
            price   = info.get("currentPrice") or info.get("previousClose") or 0
            fwd_eps = info.get("forwardEps")
            if price and fwd_eps and fwd_eps != 0:
                forward_pe = price / fwd_eps

        # ── PEG ─────────────────────────────────────────────────────────────
        peg = info.get("pegRatio") or info.get("trailingPegRatio")
        if peg is None and pe:
            g = info.get("earningsGrowth") or info.get("revenueGrowth")
            if g and g != 0:
                peg = pe / (g * 100)

        # ── ROE / ROA ────────────────────────────────────────────────────────
        roe = info.get("returnOnEquity") or info.get("roe")
        roa = info.get("returnOnAssets") or info.get("roa")

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
            "pe":                        pe,
            "forwardPE":                 forward_pe,
            "pegRatio":                  peg,
            "pb":                        info.get("priceToBook"),
            "roe":                       roe,
            "roa":                       roa,
            "profitMargins":             profit_margins,
            "grossMargins":              info.get("grossMargins"),
            "operatingMargins":          info.get("operatingMargins"),
            "ebitdaMargins":             info.get("ebitdaMargins"),
            "revenueGrowth":             info.get("revenueGrowth"),
            "earningsGrowth":            info.get("earningsGrowth"),
            "earningsQuarterlyGrowth":   info.get("earningsQuarterlyGrowth"),
            "debtToEquity":              info.get("debtToEquity"),
            "marketCap":                 info.get("marketCap"),
            "beta":                      info.get("beta"),
            "earningsQuality":           earnings_quality,
            "earningsTrend":             earnings_trend,
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

    for i, ticker in enumerate(tickers, 1):
        data = fetch_one(ticker)
        if data:
            results.append(data)
            pe_str = f"PE={data['pe']:.1f}" if data['pe'] else "PE=N/A"
            roe_str = f"ROE={data['roe']*100:.1f}%" if data['roe'] else "ROE=N/A"
            print(f"  [{i:4d}/{total}] {ticker:<20} {pe_str}  {roe_str}")
        else:
            print(f"  [{i:4d}/{total}] {ticker:<20} SKIPPED")

        time.sleep(0.6)  # conservative: avoids Yahoo Finance rate-limit on shared CI IPs

    if not results:
        print("[FUND] ERROR: No data fetched. Aborting.")
        return

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[FUND] Saved {len(df)}/{total} stocks → {OUTPUT_PATH}")
    print(f"[FUND] Columns: {list(df.columns)}")
    print(f"[FUND] PE coverage:  {df['pe'].notna().sum()}/{len(df)}")
    print(f"[FUND] ROE coverage: {df['roe'].notna().sum()}/{len(df)}")
    print(f"[FUND] NPM coverage: {df['profitMargins'].notna().sum()}/{len(df)}")


if __name__ == "__main__":
    main()
