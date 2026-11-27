"""
Active NSE Stock Universe with Sub-Industry Classifications.
Combines the core Nifty 1000 list (`data/nifty1000_list.csv`) with the dynamically synced
guarded liquid >= Rs 2 Cr/day equity expansion (`data/liquid_universe.csv`).
"""
from __future__ import annotations

import os
import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_core_nifty1000_tickers() -> list[str]:
    """Core ranked Nifty 1000 tickers from data/nifty1000_list.csv only."""
    csv_path = os.path.join(_DATA_DIR, "nifty1000_list.csv")
    try:
        df = pd.read_csv(csv_path)
        return [str(t).strip() for t in df["Ticker"].dropna().tolist() if str(t).strip()]
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []


def load_liquid_expansion_tickers() -> list[str]:
    """Guarded >= Rs 2 Cr/day liquid NSE equities outside the core Nifty 1000 list."""
    csv_path = os.path.join(_DATA_DIR, "liquid_universe.csv")
    if not os.path.exists(csv_path):
        return []
    try:
        df = pd.read_csv(csv_path)
        return [str(t).strip() for t in df["Ticker"].dropna().tolist() if str(t).strip()]
    except Exception:
        return []


def load_nifty1000_tickers() -> list[str]:
    """Active ranked universe: core Nifty 1000 union guarded dynamic >= Rs 2 Cr/day liquid equities."""
    core = load_core_nifty1000_tickers()
    extra = load_liquid_expansion_tickers()
    if not extra:
        return core
    return list(dict.fromkeys(core + extra))


def get_sub_industry_mapping() -> dict[str, str]:
    """Sub-industry mapping across core Nifty 1000 + dynamic liquid expansion (core wins on ties)."""
    out: dict[str, str] = {}
    liq_path = os.path.join(_DATA_DIR, "liquid_universe.csv")
    if os.path.exists(liq_path):
        try:
            ldf = pd.read_csv(liq_path)
            for t, ind in zip(ldf.get("Ticker", []), ldf.get("Sub_Industry", [])):
                ts, inds = str(t or "").strip(), str(ind or "").strip()
                if ts and inds and inds not in ("Unknown", "nan", "None"):
                    out[ts] = inds
        except Exception:
            pass
    csv_path = os.path.join(_DATA_DIR, "nifty1000_list.csv")
    try:
        df = pd.read_csv(csv_path)
        for t, ind in zip(df["Ticker"], df["Sub_Industry"]):
            ts = str(t or "").strip()
            if ts:
                out[ts] = ind
    except Exception:
        pass
    return out


CORE_TICKERS_1000 = load_core_nifty1000_tickers()
LIQUID_TICKERS = load_liquid_expansion_tickers()
TICKERS_1000 = load_nifty1000_tickers()
SUB_INDUSTRY_MAP = get_sub_industry_mapping()


def load_extended() -> dict:
    """
    Stocks outside the active ranked universe that the dashboard can analyse on demand (market cap of
    Rs 1,000 cr and above, plus anything you hold or watch). data/extended_universe.csv is rebuilt
    weekly by build_extended_universe.py. Returns {ticker: {name, industry, mcap_cr}}.
    """
    csv_path = os.path.join(_DATA_DIR, "extended_universe.csv")
    active = set(TICKERS_1000)
    try:
        df = pd.read_csv(csv_path)
        return {
            r.Ticker: dict(name=r.Company_Name, industry=r.Sub_Industry, mcap_cr=getattr(r, "mcap_cr", None))
            for r in df.itertuples()
            if r.Ticker not in active and r.Ticker not in SUB_INDUSTRY_MAP
        }
    except Exception:
        return {}


EXTENDED = load_extended()
# Extended names get an industry everywhere (reports, funnel checklist, My Book); TICKERS_1000 --
# the ranked universe the nightly engine scores -- excludes them.
for _t, _v in EXTENDED.items():
    if isinstance(_v.get("industry"), str) and _v["industry"] and _v["industry"] != "Unknown":
        SUB_INDUSTRY_MAP[_t] = _v["industry"]


def reload() -> list[str]:
    """Reload module-level universe tables in-place (used after sync_liquid_universe in EOD pipeline)."""
    global CORE_TICKERS_1000, LIQUID_TICKERS, TICKERS_1000, SUB_INDUSTRY_MAP, EXTENDED
    CORE_TICKERS_1000 = load_core_nifty1000_tickers()
    LIQUID_TICKERS = load_liquid_expansion_tickers()
    TICKERS_1000 = load_nifty1000_tickers()
    SUB_INDUSTRY_MAP = get_sub_industry_mapping()
    EXTENDED = load_extended()
    for _t, _v in EXTENDED.items():
        if isinstance(_v.get("industry"), str) and _v["industry"] and _v["industry"] != "Unknown":
            SUB_INDUSTRY_MAP[_t] = _v["industry"]
    return TICKERS_1000
