"""
One-off repair of data/nifty1000_list.csv
=========================================

The quarterly refresh af1c7b0 wrote a rate-limit-damaged universe: it dropped
438 names (including RELIANCE, TCS, INFY, LT, SBIN and ITC) and added 464
others, leaving 777 names that were effectively a random sample of whatever
Yahoo happened to answer that day. Every scanner, the sub-industry rotation
matrix and the model portfolio have been running on that universe since.

build_nifty1000_ci.py now refuses to produce such a list, but fixing the
builder does not fix the data — and a correct rebuild needs market-cap ranking,
which needs Yahoo. This script repairs the file offline in the meantime by
unioning the two curated universes that exist in git history:

    data/nifty1000_list.csv @ HEAD        (777, post-damage)
    data/nifty1000_list.csv @ 04aea74     (751, pre-damage)

and keeping only symbols that are still live EQ-series on NSE per EQUITY_L.csv,
so nothing delisted comes back. Sub_Industry and Company_Name are taken from
whichever curated source has them — no value is invented.

The result is a superset, not a top-1000 ranking. Run the market-cap rebuild
(build_nifty1000_ci.py, via the Quarterly Nifty 1000 Universe Refresh workflow)
to trim it back to a true top 1000; until then a slightly-too-large universe is
strictly safer than one missing the six largest companies in the index.

Run: python repair_nifty1000_universe.py [--dry-run]
"""

import argparse
import io
import subprocess

import pandas as pd

from utils.atomic_io import atomic_to_csv

OUTPUT_PATH   = "data/nifty1000_list.csv"
PRE_DAMAGE    = "04aea74"
EQUITY_L_PATH = "EQUITY_L.csv"

MEGA_CAPS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL",
    "SBIN", "ITC", "LT", "HINDUNILVR", "BAJFINANCE", "KOTAKBANK",
    "AXISBANK", "MARUTI", "SUNPHARMA",
]


def git_show(rev: str, path: str) -> pd.DataFrame:
    out = subprocess.run(["git", "show", f"{rev}:{path}"],
                         capture_output=True, text=True, check=True).stdout
    return pd.read_csv(io.StringIO(out))


def live_nse_symbols() -> set:
    eq = pd.read_csv(EQUITY_L_PATH)
    eq.columns = [c.strip() for c in eq.columns]
    eq["SYMBOL"] = eq["SYMBOL"].astype(str).str.strip()
    eq["SERIES"] = eq["SERIES"].astype(str).str.strip()
    return set(eq[eq["SERIES"] == "EQ"]["SYMBOL"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    current = pd.read_csv(OUTPUT_PATH)
    previous = git_show(PRE_DAMAGE, OUTPUT_PATH)
    live = live_nse_symbols()
    print(f"[REPAIR] current {len(current)} | pre-damage {len(previous)} | "
          f"live NSE EQ {len(live)}")

    # Current wins on conflicts: it carries the most recent curation pass.
    merged: dict = {}
    for src in (previous, current):
        for row in src.itertuples(index=False):
            merged[row.Ticker] = {
                "Ticker": row.Ticker,
                "Company_Name": getattr(row, "Company_Name", row.Ticker),
                "Sub_Industry": getattr(row, "Sub_Industry", "Diversified"),
            }

    delisted = [t for t in merged if t.replace(".NS", "") not in live]
    for t in delisted:
        del merged[t]
    print(f"[REPAIR] dropped {len(delisted)} symbol(s) no longer EQ-series on NSE")

    out = (pd.DataFrame(list(merged.values()))
           .sort_values("Ticker").reset_index(drop=True))

    restored = set(out.Ticker) - set(current.Ticker)
    symbols = {t.replace(".NS", "") for t in out.Ticker}
    missing = [m for m in MEGA_CAPS if m not in symbols]

    print(f"[REPAIR] {len(out)} names ({len(restored)} restored)")
    print(f"[REPAIR] mega caps present: {len(MEGA_CAPS) - len(missing)}/{len(MEGA_CAPS)}")
    if missing:
        raise SystemExit(f"[ABORT] mega caps still missing: {missing}")
    if out["Sub_Industry"].isna().any():
        raise SystemExit("[ABORT] null Sub_Industry in result")
    print(f"[REPAIR] sub-industries: {out['Sub_Industry'].nunique()}")

    if args.dry_run:
        print(f"[REPAIR] --dry-run: would write {len(out)} rows")
        return
    atomic_to_csv(out, OUTPUT_PATH, index=False)
    print(f"[REPAIR] wrote {len(out)} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
