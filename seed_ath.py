"""
One-off: seed data/ath_levels.csv with every scanner stock's all-time high.

Downloads full daily history (auto-adjusted, the same basis as the scanner's
prices) in batches and stores each ticker's highest close up to 32 sessions
ago plus the current all-time high (utils/ath.py explains the layout). After
this the daily engine keeps the file current from its one-year window.

Run (GitHub Actions: "Seed All-Time Highs"): python seed_ath.py
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from utils.ath import ATH_FILE, load_ath, save_ath, seed_row
from utils.nifty1000_list import TICKERS_1000
from utils.yf_safe import safe_download

BATCH = 50


def main():
    tickers = sorted(set(TICKERS_1000))
    rows, missing = {}, []
    for i in range(0, len(tickers), BATCH):
        chunk = tickers[i:i + BATCH]
        bulk = safe_download(chunk, period="max", interval="1d", group_by="ticker", threads=False,
                             auto_adjust=True, min_coverage=0.5)
        for t in chunk:
            try:
                c = (bulk[t] if len(chunk) > 1 else bulk)["Close"].dropna()
                if c.index.tz is not None:
                    c.index = c.index.tz_localize(None)
                r = seed_row(t, c)
                if r:
                    rows[t] = r
                else:
                    missing.append(t)
            except Exception:
                missing.append(t)
        print(f"[seed] {min(i + BATCH, len(tickers))}/{len(tickers)}  seeded {len(rows)}", flush=True)
        time.sleep(1)
    if len(rows) < 0.8 * len(tickers):
        sys.exit(f"[seed] only {len(rows)}/{len(tickers)} seeded -- not writing {ATH_FILE}")
    old = load_ath()
    kept = {t: r for t, r in old.items() if t not in rows}          # keep earlier rows for misses
    save_ath({**kept, **rows})
    t = pd.DataFrame(rows.values())
    yrs = (t.ath_date.max() - t.history_start).dt.days / 365.25
    print(f"[seed] wrote {len(rows) + len(kept)} rows -> {ATH_FILE}; {len(missing)} without data: "
          f"{' '.join(m.replace('.NS', '') for m in missing[:30])}")
    print(f"[seed] history: median {yrs.median():.1f} years, {int((t.sessions >= 500).sum())} with >= 2 years "
          f"(ATH breakouts are only flagged for these)")
    near = t.assign(gap=(t.ath_close / t.base_max - 1) * 100)
    print(f"[seed] {int((near.gap > 0).sum())} made a new all-time high in the last 32 sessions")


if __name__ == "__main__":
    main()
