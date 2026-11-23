#!/usr/bin/env python3
"""
Back-fill FII index-futures positioning (long % of long + short) into data/nse/market_context.csv from NSE's daily
participant-wise open-interest files, one per session in the NSE index store. Existing rows keep their values;
only missing positioning is filled. FII / DII cash has no NSE archive, so it builds up from the nightly run.

    python scripts/nse_context_backfill.py [--days 520]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from utils import nse_data as nd  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=520)
    a = ap.parse_args(argv)
    from utils import nse_prices as npx
    X = npx.raw("indices")
    sessions = sorted(pd.to_datetime(X.loc[X["index"].astype(str).str.lower() == "nifty 50", "date"]).unique())
    sessions = [pd.Timestamp(d) for d in sessions if pd.Timestamp(d) >= pd.Timestamp.today().normalize() - pd.Timedelta(days=a.days)]
    old = nd.market_context()
    have = set()
    if len(old) and "fii_idx_long_pct" in old:
        have = set(pd.to_datetime(old.loc[old["fii_idx_long_pct"].notna(), "date"]).dt.normalize())
    todo = [d for d in sessions if d not in have]
    print(f"[ctx] {len(sessions)} sessions, {len(todo)} missing FII positioning", flush=True)
    S, rows, fails = nd.Session(), [], 0
    for i, d in enumerate(todo):
        try:
            raw = S.get(f"{nd.ARCH}/content/nsccl/fao_participant_oi_{d:%d%m%Y}.csv", tries=2, timeout=20).content.decode("utf-8", "ignore")
            r = nd.parse_participant_oi(raw)
            if r:
                rows.append(dict(date=d.strftime("%Y-%m-%d"), **r))
        except Exception as e:
            fails += 1
            if fails <= 3:
                print(f"[ctx] {d:%d %b %Y}: {e}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"[ctx] {i + 1}/{len(todo)} · {len(rows)} filled", flush=True)
        time.sleep(0.3)
    if not rows:
        print("[ctx] nothing new")
        return 0
    new = pd.DataFrame(rows)
    out = new if old.empty else old.copy()
    if len(old):
        out["date"] = out["date"].astype(str)
        out = out.set_index("date")
        n = new.set_index("date")
        for c in n.columns:
            if c not in out.columns:
                out[c] = pd.NA
        add = n.loc[~n.index.isin(out.index)]
        for c in n.columns:                       # fill blanks on existing dates, never overwrite
            m = out.index.isin(n.index)
            out.loc[m, c] = out.loc[m, c].where(out.loc[m, c].notna(), n[c].reindex(out.index[m]))
        out = pd.concat([out, add]).reset_index().rename(columns={"index": "date"})
    out = out.sort_values("date")
    os.makedirs(nd.DIR, exist_ok=True)
    out.to_csv(nd.F["ctx"], index=False)
    print(f"[ctx] wrote {len(out)} rows ({len(rows)} positioning values added, {fails} sessions failed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
