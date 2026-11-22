"""
Pick a stock by company name or symbol, everywhere the Portfolio Manager takes a ticker.

Typed tickers went wrong silently: "INVPRVECQ" for INVPRECQ, "MAGNA ELECTRO" (a BSE-only company) or a
symbol outside the ranked universe. Such an entry never got a price, so its stop and target were never
checked and nothing said so. The picker lists every NSE equity (EQUITY_L.csv) plus the ranked universe,
searchable by name ("magna", "synergy green") or symbol, and takes a BSE code for BSE-only companies.

    pick(label, key, df)          -> "SGIL.NS" / "517449.BO" / None
    name_filter(label, key)       -> the text typed in a list's filter box
    matches(query, ticker, name)  -> True when the query is in the symbol or the company name
    suggest(ticker, df)           -> (symbol, name) of the closest real stock for a symbol with no data, or None
"""
from __future__ import annotations

import difflib
import re
from typing import Dict, Optional, Tuple

import pandas as pd

_BSE = re.compile(r"^\d{6}$")


def names(df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    """Ticker -> company name: every NSE equity, the extended list, and the loaded snapshot (BSE rows included)."""
    out: Dict[str, str] = {}
    try:
        from utils.extended import eq_symbols, universe
        out.update(eq_symbols())
        u = universe()
        out.update({t: n for t, n in zip(u.get("Ticker", []), u.get("Company_Name", [])) if isinstance(t, str)})
    except Exception:
        pass
    if df is not None and len(df) and "ticker" in df:
        nm = df["name"] if "name" in df else df["ticker"]
        out.update({t: (str(n) if isinstance(n, str) and n.strip() else out.get(t, t)) for t, n in zip(df["ticker"], nm)})
    return out


def label(t: str, nm: Dict[str, str]) -> str:
    sym = t.replace(".NS", "").replace(".BO", " (BSE)")
    n = nm.get(t, "")
    return f"{sym} — {n}" if n and n != t else sym


def pick(text: str, key: str, df: Optional[pd.DataFrame] = None, default: Optional[str] = None,
         universe_only: bool = False, help: Optional[str] = None) -> Optional[str]:
    """A searchable stock box (type part of the name or symbol). universe_only: just the loaded snapshot."""
    import streamlit as st
    nm = names(df)
    if universe_only and df is not None:
        nm = {t: nm.get(t, t) for t in df["ticker"]}
    opts = sorted(nm, key=lambda t: t.replace(".NS", ""))
    d = default if default in nm else (f"{default}.NS" if default and f"{default}.NS" in nm else None)
    t = st.selectbox(text, opts, index=opts.index(d) if d else None, key=key, placeholder="Type a company name or symbol",
                     format_func=lambda x: label(x, nm), help=help or "Search by company name (e.g. 'synergy green') or NSE symbol")
    if not universe_only:
        code = st.text_input("…or a BSE-only company: its 6-digit BSE code", key=f"{key}_bse", placeholder="e.g. 500325",
                             help="For companies not listed on NSE. Find the code on bseindia.com. Prices come from Yahoo (CODE.BO).")
        code = (code or "").strip().upper().replace(".BO", "")
        if code:
            return f"{code}.BO" if _BSE.match(code) else None
    return t


def name_filter(text: str, key: str) -> str:
    import streamlit as st
    return st.text_input(text, key=key, placeholder="Filter by company name or symbol").strip().lower()


def matches(q: str, ticker: str, name: str = "") -> bool:
    if not q:
        return True
    return q in str(ticker).lower().replace(".ns", "") or q in str(name or "").lower()


def suggest(ticker: str, df: Optional[pd.DataFrame] = None) -> Optional[Tuple[str, str]]:
    """The real stock a mistyped symbol or a company name most likely meant (INVPRVECQ -> INVPRECQ)."""
    nm = names(df)
    base = str(ticker).upper().replace(".NS", "").replace(".BO", "")
    try:
        from utils import nse_prices as npx
        chg = npx._symbol_map()
        fwd = {}
        for new_s, olds in chg.items():
            for old_s, dt in olds:
                if old_s not in fwd or dt > fwd[old_s][1]:
                    fwd[old_s] = new_s
        if base in fwd:
            nt = f"{fwd[base]}.NS"
            if nt in nm:
                return nt, nm[nt]
    except Exception:
        pass
    syms = {t.replace(".NS", ""): t for t in nm if t.endswith(".NS")}
    m = difflib.get_close_matches(base, list(syms), n=1, cutoff=0.8)
    if m:
        return syms[m[0]], nm[syms[m[0]]]
    # a company name typed as the ticker ("MAGNA ELECTRO"): match it against company names
    words = re.sub(r"[^A-Z0-9 ]", " ", base).split()
    if words:
        hits = [t for t, n in nm.items() if t.endswith(".NS") and all(w in str(n).upper() for w in words)]
        if len(hits) == 1:
            return hits[0], nm[hits[0]]
    return None
