"""
Nifty 1000 Universe Builder (CI-optimised)
==========================================
Ranks all NSE EQ-series stocks by market cap and writes the top 1000 with
sub-industry classifications to data/nifty1000_list.csv.

Why this was rewritten
----------------------
The previous version had two silent failure modes, and both fired in the
quarterly run af1c7b0:

1. Every per-ticker exception was swallowed and turned into `market_cap = 0`,
   which then got filtered out. Yahoo rate-limits `.info` hard at 20 concurrent
   workers, so a large slice of the universe simply vanished — the run wrote
   777 names instead of 1000 and dropped RELIANCE, TCS, INFY, LT, SBIN and ITC,
   six of India's ten largest companies. Nothing failed loudly; the degraded
   list was force-pushed over the good one and every scanner downstream has
   been running on a universe with the biggest index weights missing.

2. It wrote yfinance's raw `industry` string into Sub_Industry. That column is
   the Playbook-58 taxonomy ("Chemicals & Petrochemicals", "IT - Software"),
   hand-curated across several commits. Overwriting it with Yahoo's vocabulary
   ("Specialty Chemicals", "Banks - Regional") silently destroys the taxonomy
   the sub-industry rotation matrix is built on.

The fixes: retry failed tickers instead of dropping them, refuse to write a
list that fails a coverage / mega-cap / shrink check, and preserve curated
Sub_Industry values — only classifying genuinely new tickers.

An aborted run leaves the previous good list untouched. That is the point: a
stale universe is recoverable, a silently-truncated one is not.

Run: python build_nifty1000_ci.py [--dry-run]
"""

import argparse
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf

from build_58_sub_industry_map import YFINANCE_TO_PLAYBOOK, PLAYBOOK_58
from utils.atomic_io import atomic_to_csv

EQUITY_L_URL  = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
EQUITY_L_PATH = "EQUITY_L.csv"
OUTPUT_PATH   = "data/nifty1000_list.csv"
TARGET        = 1000

BATCH_WORKERS = 8      # was 20 — the aggressive setting is what got rate-limited
DELAY         = 0.5
RETRY_PASSES  = 3      # failed tickers get re-queued, not dropped
RETRY_BACKOFF = 20     # seconds before each retry pass

# --- Safety gates -----------------------------------------------------------
MIN_COVERAGE  = 0.85   # fraction of EQ tickers that must resolve
MIN_RETENTION = 0.95   # new list may not be smaller than this * previous list

# Unambiguously top-30 NSE names. If any of these is missing from the final
# ranking, the market-cap fetch was degraded — no correct top-1000 excludes
# them. This is the canary that would have caught af1c7b0.
MEGA_CAPS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL",
    "SBIN", "ITC", "LT", "HINDUNILVR", "BAJFINANCE", "KOTAKBANK",
    "AXISBANK", "MARUTI", "SUNPHARMA",
]


class BuildAborted(RuntimeError):
    """Raised when the result is not trustworthy enough to overwrite the list."""


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def load_equity_list() -> pd.DataFrame:
    try:
        req = urllib.request.Request(EQUITY_L_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            df = pd.read_csv(resp)
        print(f"[BUILD] Downloaded fresh EQUITY_L.csv ({len(df)} rows)")
        atomic_to_csv(df, EQUITY_L_PATH, index=False)
    except Exception as e:
        print(f"[BUILD] EQUITY_L download failed ({e}), using local copy")
        df = pd.read_csv(EQUITY_L_PATH)

    df.columns = [c.strip() for c in df.columns]
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
    df["SERIES"] = df["SERIES"].astype(str).str.strip()
    df = df[df["SERIES"] == "EQ"].copy()
    print(f"[BUILD] {len(df)} EQ-series stocks in universe")
    return df


def load_existing() -> pd.DataFrame:
    if os.path.exists(OUTPUT_PATH):
        try:
            return pd.read_csv(OUTPUT_PATH)
        except Exception:
            pass
    return pd.DataFrame(columns=["Ticker", "Company_Name", "Sub_Industry"])


# ---------------------------------------------------------------------------
# market cap
# ---------------------------------------------------------------------------
def fetch_market_cap(ticker: str) -> dict:
    """
    Returns {'ticker','market_cap','name','industry','ok'}.

    `ok` distinguishes "Yahoo answered and this company is genuinely tiny/absent"
    from "the call failed". The old version collapsed both into market_cap=0,
    which is precisely how live mega caps got filtered out of the universe.
    """
    t = yf.Ticker(ticker)

    # fast_info is a much lighter endpoint than .info and survives rate limiting
    # far better; .info is only consulted when it comes back empty.
    try:
        fi = t.fast_info
        cap = fi.get("market_cap") if hasattr(fi, "get") else getattr(fi, "market_cap", None)
        if cap:
            return {"ticker": ticker, "market_cap": float(cap),
                    "name": None, "industry": None, "ok": True}
    except Exception:
        pass

    try:
        info = t.info or {}
    except Exception:
        return {"ticker": ticker, "market_cap": 0.0, "name": None,
                "industry": None, "ok": False}

    if not info:
        return {"ticker": ticker, "market_cap": 0.0, "name": None,
                "industry": None, "ok": False}

    return {
        "ticker": ticker,
        "market_cap": float(info.get("marketCap") or 0),
        "name": info.get("shortName") or info.get("longName"),
        "industry": info.get("industry") or info.get("sector"),
        "ok": True,
    }


def fetch_all(tickers) -> pd.DataFrame:
    """Fetch every ticker, re-queueing failures across several passes."""
    resolved, pending = {}, list(tickers)

    for attempt in range(1, RETRY_PASSES + 1):
        if not pending:
            break
        if attempt > 1:
            wait = RETRY_BACKOFF * (attempt - 1)
            print(f"[BUILD] Pass {attempt}: retrying {len(pending)} unresolved "
                  f"ticker(s) after {wait}s backoff...")
            time.sleep(wait)

        failed, done = [], 0
        for i in range(0, len(pending), 100):
            batch = pending[i:i + 100]
            with ThreadPoolExecutor(max_workers=BATCH_WORKERS) as ex:
                futures = {ex.submit(fetch_market_cap, t): t for t in batch}
                for fut in as_completed(futures):
                    r = fut.result()
                    (resolved.setdefault(r["ticker"], r) if r["ok"]
                     else failed.append(r["ticker"]))
                    done += 1
            time.sleep(DELAY)

        print(f"[BUILD] Pass {attempt}: {len(resolved)} resolved, {len(failed)} failed")
        pending = failed

    if pending:
        print(f"[BUILD] {len(pending)} ticker(s) never resolved, e.g. {pending[:10]}")

    return pd.DataFrame(list(resolved.values())), len(pending)


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def resolve_sub_industry(ticker: str, yf_industry, curated: dict) -> str:
    """
    Curated value wins. Only genuinely new tickers get classified from Yahoo's
    vocabulary, and only through the Playbook-58 mapping — never Yahoo's raw
    string, which does not belong to this taxonomy.
    """
    existing = curated.get(ticker)
    if existing and str(existing).strip() and str(existing) != "nan":
        return existing
    if yf_industry:
        mapped = YFINANCE_TO_PLAYBOOK.get(str(yf_industry).strip())
        if mapped in PLAYBOOK_58:
            return mapped
    return "Diversified"


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def enforce_gates(top: pd.DataFrame, resolved_n: int, eq_n: int,
                  previous: pd.DataFrame) -> None:
    coverage = resolved_n / max(eq_n, 1)
    print(f"[GATE] Coverage: {resolved_n}/{eq_n} = {coverage:.1%} "
          f"(floor {MIN_COVERAGE:.0%})")
    if coverage < MIN_COVERAGE:
        raise BuildAborted(
            f"only {coverage:.1%} of the EQ universe resolved — refusing to "
            f"overwrite the universe with a partial fetch")

    symbols = {t.replace(".NS", "") for t in top["ticker"]}
    missing = [m for m in MEGA_CAPS if m not in symbols]
    print(f"[GATE] Mega-cap sentinels present: {len(MEGA_CAPS) - len(missing)}/{len(MEGA_CAPS)}")
    if missing:
        raise BuildAborted(
            f"mega caps missing from the top {TARGET}: {missing} — no correct "
            f"ranking excludes these, so the fetch was degraded")

    if len(previous):
        floor = int(len(previous) * MIN_RETENTION)
        print(f"[GATE] Size: {len(top)} vs previous {len(previous)} (floor {floor})")
        if len(top) < floor:
            raise BuildAborted(
                f"new list has {len(top)} names vs {len(previous)} previously — "
                f"a shrink this large means a failed fetch, not an index change")

        dropped = ({t.replace('.NS', '') for t in previous["Ticker"]} - symbols)
        mega_dropped = [d for d in dropped if d in MEGA_CAPS]
        if mega_dropped:
            raise BuildAborted(f"would drop mega caps still listed: {mega_dropped}")
        if dropped:
            print(f"[GATE] {len(dropped)} name(s) drop out of the top {TARGET} "
                  f"(normal churn), e.g. {sorted(dropped)[:8]}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="run every check but do not write the file")
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)

    eq_df    = load_equity_list()
    previous = load_existing()
    curated  = dict(zip(previous.get("Ticker", []), previous.get("Sub_Industry", [])))
    prev_names = dict(zip(previous.get("Ticker", []), previous.get("Company_Name", [])))

    tickers = [s + ".NS" for s in eq_df["SYMBOL"].tolist()]
    print(f"[BUILD] Fetching market cap for {len(tickers)} tickers "
          f"({BATCH_WORKERS} workers, up to {RETRY_PASSES} passes)...")

    df_all, unresolved = fetch_all(tickers)
    if df_all.empty:
        raise BuildAborted("no market-cap data at all — nothing to rank")

    ranked = df_all[df_all["market_cap"] > 0].sort_values("market_cap", ascending=False)
    top = ranked.head(TARGET).copy()

    print(f"\n[BUILD] {len(ranked)} stocks with a market cap; top {len(top)} selected")
    if len(top):
        print(f"[BUILD] Market cap threshold (rank {len(top)}): "
              f"₹{top['market_cap'].iloc[-1] / 1e7:,.0f} Cr")

    enforce_gates(top, len(df_all), len(eq_df), previous)

    out = pd.DataFrame({
        "Ticker": top["ticker"].values,
        "Company_Name": [
            (n if n else prev_names.get(t, t.replace(".NS", "")))
            for t, n in zip(top["ticker"], top["name"])
        ],
        "Sub_Industry": [
            resolve_sub_industry(t, ind, curated)
            for t, ind in zip(top["ticker"], top["industry"])
        ],
    }).sort_values("Ticker").reset_index(drop=True)

    new_names = [t for t in out["Ticker"] if t not in curated]
    print(f"[BUILD] {len(new_names)} new ticker(s) entering the universe")
    unclassified = out[(out["Sub_Industry"] == "Diversified")
                       & (out["Ticker"].isin(new_names))]["Ticker"].tolist()
    if unclassified:
        print(f"[BUILD] {len(unclassified)} new ticker(s) fell back to "
              f"'Diversified' and want curating: {unclassified[:15]}")

    if args.dry_run:
        print(f"\n[BUILD] --dry-run: would write {len(out)} stocks → {OUTPUT_PATH}")
        return

    atomic_to_csv(out, OUTPUT_PATH, index=False)
    print(f"\n[BUILD] Saved {len(out)} stocks → {OUTPUT_PATH}")
    print(f"[BUILD] Unique sub-industries: {out['Sub_Industry'].nunique()}")
    if unresolved:
        print(f"[BUILD] Note: {unresolved} ticker(s) never resolved but coverage "
              f"stayed above the floor")


if __name__ == "__main__":
    try:
        main()
    except BuildAborted as e:
        print(f"\n[ABORT] {e}")
        print("[ABORT] Existing data/nifty1000_list.csv left untouched.")
        raise SystemExit(1)
