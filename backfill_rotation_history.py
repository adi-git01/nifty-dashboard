"""
backfill_rotation_history.py
============================
Backfills data/sub_industry_rotation.csv with the last 12 months of
sub-industry RS scores, using the SAME sector mapping as the live engine.

Sector source (priority):
  1. Latest Parquet cache (data/cache/market_master_*.parquet) - ticker->sector
  2. Fallback: nifty1000_list.csv SUB_INDUSTRY_MAP

This ensures historical heatmap columns exactly match what the live engine produces.

Run once:
    python backfill_rotation_history.py

Re-running is safe - already-filled months are skipped.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys
import glob
from dateutil.relativedelta import relativedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nifty1000_list import TICKERS_1000, SUB_INDUSTRY_MAP

DATA_DIR  = "data"
CACHE_DIR = "data/cache"
ROTATION_CSV = os.path.join(DATA_DIR, "sub_industry_rotation.csv")
os.makedirs(DATA_DIR, exist_ok=True)

RS_WEIGHTS      = [(5, 0.10), (21, 0.50), (63, 0.40)]
LOOKBACK_MONTHS = 12


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_sector_map() -> dict:
    """
    Load ticker -> sub-industry mapping.
    PRIMARY: nifty1000_list.csv Sub_Industry column (updated with 58 playbook names)
    SECONDARY: Parquet sector column for any tickers missing from CSV
    """
    # Primary: CSV (now has 58 playbook sub-industry names)
    csv_path = os.path.join(DATA_DIR, "nifty1000_list.csv")
    mapping = {}
    if os.path.exists(csv_path):
        csv_df = pd.read_csv(csv_path)
        for _, row in csv_df.iterrows():
            t = row.get('Ticker', '')
            s = str(row.get('Sub_Industry', '')).strip()
            if t and s and s not in ('', 'Unknown', 'nan'):
                mapping[t] = s
        print(f"  [OK] Loaded {len(mapping)} ticker->sub-industry from nifty1000_list.csv")
        print(f"  Unique sub-industries: {pd.Series(mapping.values()).nunique()} "
              f"({pd.Series(mapping.values()).value_counts().head(5).to_dict()}...)")
    else:
        print("  [WARN] nifty1000_list.csv not found")

    # Secondary: Parquet sector for any tickers NOT in CSV
    parquet_files = sorted(glob.glob(os.path.join(CACHE_DIR, "market_master_*.parquet")))
    if parquet_files:
        try:
            pq = pd.read_parquet(parquet_files[-1], columns=['ticker', 'sector'])
            for _, row in pq.iterrows():
                t = row['ticker']
                s = str(row.get('sector', '')).strip()
                if t not in mapping and s and s not in ('', 'Unknown', 'Other', 'nan'):
                    mapping[t] = s
        except Exception as e:
            print(f"  [WARN] Parquet read error: {e}")

    print(f"  Total: {len(mapping)} tickers mapped to {pd.Series(mapping.values()).nunique()} sub-industries")
    return mapping



def get_month_end_dates(n_months: int) -> list:
    today  = datetime.now().replace(day=1)
    dates  = []
    for i in range(n_months, 0, -1):
        m_start = today - relativedelta(months=i)
        m_end   = m_start + relativedelta(months=1) - timedelta(days=1)
        if m_end > datetime.now():
            m_end = datetime.now() - timedelta(days=1)
        dates.append(m_end.strftime("%Y-%m-%d"))
    return dates


def composite_rs(stock_close: pd.Series, nifty_close: pd.Series) -> float | None:
    """Composite RS at the last available date: 10%*1W + 50%*1M + 40%*3M vs Nifty."""
    n = len(stock_close)
    if n < 64:
        return None
    price    = float(stock_close.iloc[-1])
    n_price  = float(nifty_close.iloc[-1])
    rs_total = 0.0
    for period, weight in RS_WEIGHTS:
        if n <= period or len(nifty_close) <= period:
            return None
        s_past = float(stock_close.iloc[-period - 1])
        n_past = float(nifty_close.iloc[-period - 1])
        if s_past == 0 or n_past == 0:
            return None
        rs_total += ((price / s_past - 1) - (n_price / n_past - 1)) * 100 * weight
    return round(rs_total, 4)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # 1. Load existing CSV
    if os.path.exists(ROTATION_CSV):
        existing_df     = pd.read_csv(ROTATION_CSV)
        existing_months = set(
            pd.to_datetime(existing_df['record_date']).dt.to_period('M').astype(str)
        ) if 'record_date' in existing_df.columns else set()
        print(f"  Existing months : {sorted(existing_months)}")
    else:
        existing_df     = pd.DataFrame()
        existing_months = set()

    # 2. Determine months to fill
    all_dates    = get_month_end_dates(LOOKBACK_MONTHS)
    dates_to_fill = [d for d in all_dates
                     if pd.Period(d, freq='M').strftime('%Y-%m') not in existing_months]

    if not dates_to_fill:
        print("[OK] All 12 months already in CSV. Nothing to do.")
        return

    print(f"\n{'='*70}")
    print(f"  BACKFILL: Sub-Industry Rotation (last 12 months)")
    print(f"{'='*70}")
    print(f"  Months to fill : {dates_to_fill}")

    # 3. Load sector map (same as live engine)
    print("\n  Loading sector mapping...")
    sector_map = load_sector_map()
    tickers    = [t for t in TICKERS_1000 if t in sector_map]
    print(f"  Tickers with sector labels: {len(tickers)}")

    # 4. Download price history
    fetch_start = (datetime.now() - relativedelta(months=LOOKBACK_MONTHS + 3)).strftime("%Y-%m-%d")
    fetch_end   = datetime.now().strftime("%Y-%m-%d")

    print(f"\n  Fetching Nifty {fetch_start} -> {fetch_end}...")
    nifty_raw = yf.Ticker("^NSEI").history(start=fetch_start, end=fetch_end)
    if nifty_raw.empty:
        print("  ERROR: Could not fetch Nifty. Aborting.")
        return
    if nifty_raw.index.tz is not None:
        nifty_raw.index = nifty_raw.index.tz_localize(None)
    nifty_close = nifty_raw['Close']

    print(f"  Fetching {len(tickers)} tickers (bulk download, ~2-5 min)...")
    try:
        bulk = yf.download(
            tickers, start=fetch_start, end=fetch_end,
            group_by='ticker', threads=True, progress=True, auto_adjust=True
        )
    except Exception as e:
        print(f"  ERROR in bulk download: {e}")
        return

    print("  Parsing bulk data...")
    stock_closes = {}
    for t in tickers:
        try:
            if t in bulk.columns.get_level_values(0):
                s = bulk[t]['Close'].dropna()
                if s.index.tz is not None:
                    s.index = s.index.tz_localize(None)
                if len(s) > 70:
                    stock_closes[t] = s
        except Exception:
            pass
    print(f"  Usable tickers: {len(stock_closes)}")

    # 5. Compute RS per month-end
    new_rows = []

    for date_str in dates_to_fill:
        ref_date = pd.Timestamp(date_str)
        print(f"\n  Processing {date_str}...")

        valid_nifty = nifty_close[nifty_close.index <= ref_date]
        if valid_nifty.empty:
            print(f"    [SKIP] No Nifty data before {date_str}")
            continue
        actual_ref = valid_nifty.index[-1]

        stock_rs = {}
        for t, s in stock_closes.items():
            slice_s = s[s.index <= actual_ref]
            if len(slice_s) < 65:
                continue
            # Align Nifty to stock's trading dates
            nifty_aligned = nifty_close.reindex(s.index, method='ffill')
            slice_n = nifty_aligned[nifty_aligned.index <= actual_ref].dropna()
            if len(slice_n) < 65:
                continue
            rs = composite_rs(slice_s.iloc[-65:], slice_n.iloc[-65:])
            if rs is not None:
                stock_rs[t] = rs

        if not stock_rs:
            print(f"    [SKIP] No RS scores for {date_str}")
            continue

        # Group by sector (same logic as live engine)
        groups: dict = {}
        for t, rs in stock_rs.items():
            sec = sector_map.get(t)
            if not sec or str(sec) in ('', 'Unknown', 'Other', 'nan'):
                continue
            groups.setdefault(sec, []).append((t, rs))

        rows = []
        for sec, items in groups.items():
            avg_rs   = float(np.mean([x[1] for x in items]))
            top3     = sorted(items, key=lambda x: -x[1])[:3]
            top3_str = ", ".join(t.replace('.NS', '') for t, _ in top3)
            rows.append({
                'record_date':    date_str,
                'sub_industry':   sec,
                'rs_momentum':    round(avg_rs, 4),
                'top_components': top3_str,
            })

        rot_df = pd.DataFrame(rows)
        if not rot_df.empty:
            rs_min = rot_df['rs_momentum'].min()
            rs_max = rot_df['rs_momentum'].max()
            if rs_max > rs_min:
                rot_df['score_0_100'] = (
                    (rot_df['rs_momentum'] - rs_min) / (rs_max - rs_min) * 100
                ).round(0).astype(int)
            else:
                rot_df['score_0_100'] = 50

            new_rows.append(rot_df)
            print(f"    [OK] {len(rot_df)} sectors | RS {rs_min:.1f} -> {rs_max:.1f}")

    # 6. Merge and save
    if new_rows:
        new_df   = pd.concat(new_rows, ignore_index=True)
        combined = pd.concat([existing_df, new_df], ignore_index=True) \
                    if not existing_df.empty else new_df
        combined = (
            combined
            .sort_values('record_date')
            .drop_duplicates(subset=['record_date', 'sub_industry'], keep='last')
            .reset_index(drop=True)
        )
        combined.to_csv(ROTATION_CSV, index=False)
        print(f"\n{'='*70}")
        print(f"  [DONE] BACKFILL COMPLETE")
        print(f"  Rows saved  : {len(combined)}")
        print(f"  Unique dates: {combined['record_date'].nunique()}")
        print(f"  Unique sectors: {combined['sub_industry'].nunique()}")
        print(f"  CSV: {ROTATION_CSV}")
        print(f"{'='*70}")
        print(f"\n  >> Restart your Streamlit dashboard to see the 12-month heatmap.")
    else:
        print("\n  No new data generated.")


if __name__ == "__main__":
    main()
