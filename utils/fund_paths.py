"""
Canonical paths and sorted CSV I/O helpers for Indian & US fundamentals and monthly prices.

Storing uncompressed, deterministically sorted `.csv` files instead of binary `.csv.gz` blobs
makes git pack/delta compression far more effective across weekly refreshes while keeping pandas
reads faster on startup and preserving `.csv.gz` fallback compatibility in tests/fixtures.
"""
from __future__ import annotations

import os
from typing import Sequence

import pandas as pd

ANNUAL = "data/fundamentals_annual.csv"
QUARTERLY = "data/fundamentals_quarterly.csv"
MONTHLY = "data/monthly_prices.csv"
EXT_ANNUAL = "data/fundamentals_ext_annual.csv"
EXT_QUARTERLY = "data/fundamentals_ext_quarterly.csv"
EXTRA_ANNUAL = "data/fundamentals_annual_extra.csv"

US_ANNUAL = "data/us_fundamentals_annual.csv"
US_QUARTERLY = "data/us_fundamentals_quarterly.csv"
US_MONTHLY = "data/us_monthly_prices.csv"


def resolve(path: str, root: str = ".") -> str:
    """
    Resolve a fundamentals/monthly file path checking uncompressed `.csv` first, falling back to
    `.csv.gz` if only the gzipped copy exists (e.g. legacy test fixtures).
    """
    base = path[:-3] if path.endswith(".csv.gz") else path
    csv_rel = base
    gz_rel = f"{base}.gz" if base.endswith(".csv") else path
    p_csv = csv_rel if root in ("", ".") else os.path.join(root, csv_rel)
    p_gz = gz_rel if root in ("", ".") else os.path.join(root, gz_rel)
    if os.path.exists(p_csv):
        return p_csv
    if os.path.exists(p_gz):
        return p_gz
    return p_csv


def write_sorted_csv(df: pd.DataFrame, path: str, sort_cols: Sequence[str]) -> str:
    """Sort deterministically by `sort_cols` (present columns only) and write UTF-8 CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    by = [c for c in sort_cols if c in df.columns]
    out = df.sort_values(by, kind="mergesort") if by else df
    out.to_csv(path, index=False)
    return path
