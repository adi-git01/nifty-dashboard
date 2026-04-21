"""
Nifty 1000 Universe Builder (CI-optimised)
==========================================
Ranks all NSE EQ-series stocks by market cap (single .info call per stock)
and writes the top 1000 with sub-industry classifications to
data/nifty1000_list.csv.

Designed for quarterly GitHub Actions runs (~15 min, 0.3s delay per stock).
Falls back to the repo's existing EQUITY_L.csv if the NSE download is blocked.

Run: python build_nifty1000_ci.py
"""

import pandas as pd
import yfinance as yf
import urllib.request
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

EQUITY_L_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
EQUITY_L_PATH = "EQUITY_L.csv"
OUTPUT_PATH   = "data/nifty1000_list.csv"
TARGET        = 1000
BATCH_WORKERS = 20   # concurrent yfinance calls
DELAY         = 0.3  # seconds between batches


def load_equity_list() -> pd.DataFrame:
    try:
        req = urllib.request.Request(EQUITY_L_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            df = pd.read_csv(resp)
        print(f"[BUILD] Downloaded fresh EQUITY_L.csv ({len(df)} rows)")
        df.to_csv(EQUITY_L_PATH, index=False)
    except Exception as e:
        print(f"[BUILD] EQUITY_L download failed ({e}), using local copy")
        df = pd.read_csv(EQUITY_L_PATH)

    df.columns = [c.strip() for c in df.columns]
    df["SYMBOL"] = df["SYMBOL"].str.strip()
    df = df[df["SERIES"] == "EQ"].copy()
    print(f"[BUILD] {len(df)} EQ-series stocks in universe")
    return df


def fetch_market_cap(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        market_cap = info.get("marketCap") or 0
        short_name = info.get("shortName") or info.get("longName") or ticker
        industry   = info.get("industry")  or info.get("sector") or "Unknown"
        return {"ticker": ticker, "market_cap": market_cap, "name": short_name, "industry": industry}
    except Exception:
        return {"ticker": ticker, "market_cap": 0, "name": ticker, "industry": "Unknown"}


def main():
    os.makedirs("data", exist_ok=True)

    eq_df    = load_equity_list()
    tickers  = [sym + ".NS" for sym in eq_df["SYMBOL"].tolist()]
    total    = len(tickers)
    print(f"[BUILD] Fetching market cap for {total} tickers ({BATCH_WORKERS} workers)...")

    results  = []
    done     = 0
    batch_sz = 100

    for i in range(0, total, batch_sz):
        batch = tickers[i : i + batch_sz]
        with ThreadPoolExecutor(max_workers=BATCH_WORKERS) as ex:
            futures = {ex.submit(fetch_market_cap, t): t for t in batch}
            for fut in as_completed(futures):
                results.append(fut.result())
                done += 1
        if done % 200 == 0:
            print(f"  [{done}/{total}]  results so far with market cap > 0: "
                  f"{sum(1 for r in results if r['market_cap'] > 0)}")
        time.sleep(DELAY)

    df_all = pd.DataFrame(results)
    df_all = df_all[df_all["market_cap"] > 0].sort_values("market_cap", ascending=False)
    top    = df_all.head(TARGET).copy()

    print(f"\n[BUILD] Top {len(top)} stocks by market cap selected")
    threshold = top["market_cap"].iloc[-1] / 1e7
    print(f"[BUILD] Market cap threshold (rank {TARGET}): ₹{threshold:.0f} Cr")

    out = pd.DataFrame({
        "Ticker":       top["ticker"],
        "Company_Name": top["name"],
        "Sub_Industry": top["industry"],
    })
    out = out.sort_values("Ticker").reset_index(drop=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"\n[BUILD] Saved {len(out)} stocks → {OUTPUT_PATH}")
    print(f"[BUILD] Unique sub-industries: {out['Sub_Industry'].nunique()}")
    unknown = (out["Sub_Industry"] == "Unknown").sum()
    if unknown:
        print(f"[BUILD] Warning: {unknown} stocks with Unknown sub-industry")


if __name__ == "__main__":
    main()
