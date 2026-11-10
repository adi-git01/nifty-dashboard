"""A search box for any stock table: type part of a company name or ticker."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def match(df: pd.DataFrame, query: str, cols=("name", "ticker")) -> pd.DataFrame:
    """Rows where any of `cols` contains any comma-separated term (case-insensitive)."""
    terms = [t.strip().lower() for t in (query or "").replace("\n", ",").split(",") if t.strip()]
    if not terms or df is None or df.empty:
        return df
    hay = pd.Series("", index=df.index)
    for c in cols:
        if c in df.columns:
            hay = hay + " " + df[c].astype(str).str.lower()
    keep = pd.Series(False, index=df.index)
    for t in terms:
        keep |= hay.str.contains(t, regex=False)
    return df[keep]


def search_box(df: pd.DataFrame, key: str, cols=("name", "ticker"), label: str = "🔎 Search name or ticker",
               container=None) -> pd.DataFrame:
    """Render the box and return `df` filtered by it. Several names: separate with commas."""
    q = (container or st).text_input(label, key=key, placeholder="e.g. cummins, HINDCOPPER, tata",
                                     help="Part of a company name or ticker. Separate several with commas.")
    out = match(df, q, cols)
    if q and out is not None:
        (container or st).caption(f"{len(out)} match(es) for “{q}”")
    return out
