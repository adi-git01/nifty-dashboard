"""
My Book's rules, free of any UI: holding status, the exit level, portfolio-level risk and the
funnel label. Used by the My Book tab (utils/book_view.py), the daily Telegram brief
(utils/daily_message.py) and the MCP server (mcp_server/insight.py).
"""
from __future__ import annotations

import glob
from functools import lru_cache
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

NEAR_STOP = 3.0


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return f if np.isfinite(f) and f > 0 else None
    except (TypeError, ValueError):
        return None


def status(px, stop, target, scan, below50, caution) -> str:
    if not px:
        return "❔ no price"
    if stop and px <= stop:
        return "🔴 stop hit"
    if target and px >= target:
        return "🎯 target hit"
    if stop and (px / stop - 1) * 100 < NEAR_STOP:
        return f"⚠️ {(px / stop - 1) * 100:.1f}% above stop"
    if caution >= 2 or (below50 and scan != "BUY"):
        return "🟣 trend weakening"
    return "✅ on track"


PANEL_FILE = "data/cache/close_panel.parquet"


def _snap_date(f: str) -> pd.Timestamp:
    return pd.Timestamp(f[-18:-8].replace("_", "-"))


def _closes(files) -> Dict[pd.Timestamp, pd.Series]:
    P = {}
    for f in files:
        try:
            d = pd.read_parquet(f, columns=["ticker", "currentPrice"]).drop_duplicates("ticker")
        except Exception:
            continue
        P[_snap_date(f)] = pd.to_numeric(d.set_index("ticker")["currentPrice"], errors="coerce")
    return P


def build_close_panel(path: str = PANEL_FILE) -> int:
    """
    Every snapshot's closing prices in one small file, written by the nightly run. Reading ~185
    snapshots took about 5 s the first time My Book opened after each restart; one file takes a
    fraction of that. Prices are stored as recorded; splits are applied when it is read.
    """
    files = sorted(glob.glob("data/cache/market_master_*.parquet"))
    P = pd.DataFrame(_closes(files)).T.sort_index()
    P.index.name = "date"
    P.astype("float32").to_parquet(path)
    return len(P)


@lru_cache(maxsize=2)
def _panel(n_files: int) -> pd.DataFrame:
    """
    Closing prices by session and ticker, split-adjusted: the nightly panel file plus any snapshot
    newer than it (so it is never stale), or every snapshot when the file is missing.
    """
    files = sorted(glob.glob("data/cache/market_master_*.parquet"))
    base = None
    try:
        base = pd.read_parquet(PANEL_FILE).astype(float)
        base.index = pd.to_datetime(base.index)
    except Exception:
        base = None
    if base is not None and len(base):
        new = _closes([f for f in files if _snap_date(f) > base.index.max()])
        P = pd.concat([base, pd.DataFrame(new).T]) if new else base
    else:
        P = pd.DataFrame(_closes(files)).T
    P = P[~P.index.duplicated(keep="last")].sort_index()
    try:
        from utils import corp_actions
        for _, r in corp_actions.load().iterrows():
            if r["ticker"] in P.columns and float(r["factor"]) > 0:
                P.loc[P.index < pd.Timestamp(r["date"]), r["ticker"]] /= float(r["factor"])
    except Exception:
        pass
    return P


def peak_since(ticker: str, entry_date, price: Optional[float]) -> Optional[float]:
    """Highest close since the entry date from the stored snapshots (split-adjusted), at least today's price."""
    try:
        P = _panel(len(glob.glob("data/cache/market_master_*.parquet")))
        s = P[ticker].dropna() if ticker in P.columns else pd.Series(dtype=float)
        if entry_date:
            s = s[s.index >= pd.Timestamp(entry_date)]
        hi = float(s.max()) if len(s) else None
    except Exception:
        hi = None
    vals = [v for v in (hi, price) if v]
    return max(vals) if vals else None


def exit_guide(price, ma50, peak, trail: float, stop) -> Dict[str, Any]:
    """
    The live engine's exit, the one rule the exit studies kept (analysis/ENTRY_EXIT_VERDICT.md,
    exit_rule_multiregime_result.md): sell on a close below the 50-day average, or `trail` below
    the highest close since entry, whichever level is higher.
    """
    lv = [x for x in (ma50, peak * (1 - trail) if peak else None) if x and np.isfinite(x)]
    if not lv or not price:
        return dict(level=None, text="—", hit=False)
    level = max(lv)
    which = "50-day average" if ma50 and level == ma50 else f"{trail * 100:.0f}% below the peak ₹{peak:,.0f}"
    if price < level:
        return dict(level=level, text=f"🔴 Exit rule hit: closed below ₹{level:,.1f} ({which})", hit=True)
    if not stop or stop < level * 0.995:
        return dict(level=level, text=f"Raise stop to ₹{level:,.1f} ({which})", hit=False)
    return dict(level=level, text=f"Stop is at or above the exit level ₹{level:,.1f}", hit=False)


def risk_summary(H: pd.DataFrame, capital: Optional[float] = None) -> Dict[str, Any]:
    """Portfolio-level numbers from holdings rows (Price, Qty, Stop, Exit below, Industry name, Daily traded)."""
    h = H[H["Price"].notna()].copy()
    if h.empty:
        return {}
    h["value"] = h["Price"] * h["Qty"].fillna(1)
    floor = h["Stop"].where(h["Stop"].notna() & (h["Stop"] < h["Price"]), h["Exit below"])
    h["risk"] = ((h["Price"] - floor) * h["Qty"].fillna(1)).clip(lower=0)
    invested = float(h.value.sum())
    cap = capital if capital and capital > 0 else invested
    ind = (h.groupby("Sector")["value"].sum() / invested * 100).sort_values(ascending=False)
    thin = h[h["Daily traded"].notna() & (h.value > 0.05 * h["Daily traded"])]
    return dict(invested=invested, capital=cap, risk=float(h.risk.sum()), risk_pct=float(h.risk.sum() / cap * 100) if cap else np.nan,
                no_stop=h.loc[h["Stop"].isna(), "Stock"].tolist(), largest=(h.loc[h.value.idxmax(), "Stock"], float(h.value.max() / cap * 100)),
                by_industry=ind, thin=thin["Stock"].tolist(), n=len(h),
                unprotected=h.loc[floor.isna(), "Stock"].tolist())


def funnel_label(r: Optional[pd.Series]) -> str:
    if r is None:
        return "—"
    miss = [lab for k, lab in (("s_industry", "industry"), ("s_trend", "trend"), ("s_business", "business"), ("s_setup", "entry"))
            if not (pd.notna(r.get(k)) and bool(r.get(k))) and not (k == "s_business" and pd.isna(r.get(k)))]
    return "✅ Buyable now" if not miss else "Missing: " + ", ".join(miss)
