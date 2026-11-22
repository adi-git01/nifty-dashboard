"""
Read-only accessors over the dashboard's committed data files.

Kept separate from the MCP wiring so the query logic is testable without a
protocol round-trip, and so a REST layer could reuse it later.

Everything here reads the repo working tree. That is deliberate: the repo is
this system's database — the EOD workflow commits the master snapshots, signal
logs, portfolio state and universe, so a checkout is a complete, point-in-time
dataset with no server to run.
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


# --------------------------------------------------------------------------
def _clean(obj: Any) -> Any:
    """JSON-safe: NaN/NaT/numpy scalars become None or plain Python."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if obj is None:
        return None
    if isinstance(obj, float) and obj != obj:
        return None
    if hasattr(obj, "item"):
        try:
            v = obj.item()
            return None if isinstance(v, float) and v != v else v
        except Exception:
            return str(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    return obj


def _parse_dates(series: pd.Series) -> pd.Series:
    """
    Parse a date column that may mix formats.

    sub_industry_rotation.csv carries two: the monthly backfill wrote
    "2025-03-31 00:00:00" while daily appends write "2026-02-24". A plain
    to_datetime(errors="coerce") infers one format from the leading rows and
    silently NaTs the other 7203 of 7755 — which made "newest" resolve to
    January. main.py already uses format="mixed" for this file; match it.
    """
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except Exception:
        return pd.to_datetime(series.astype(str).str.slice(0, 10), errors="coerce")


def _records(df: pd.DataFrame, limit: Optional[int] = None) -> List[Dict]:
    if limit:
        df = df.head(limit)
    return [_clean(r) for r in df.to_dict(orient="records")]


# --------------------------------------------------------------------------
def snapshot_dates() -> List[str]:
    out = []
    for f in glob.glob(_p("data", "cache", "market_master_*.parquet")):
        m = re.search(r"(\d{4}_\d{2}_\d{2})", f)
        if m:
            out.append(m.group(1).replace("_", "-"))
    return sorted(out)


def load_snapshot(date: Optional[str] = None) -> pd.DataFrame:
    """Master market snapshot for `date` (YYYY-MM-DD), or the newest."""
    dates = snapshot_dates()
    if not dates:
        raise FileNotFoundError("no market snapshots in data/cache/")
    target = date or dates[-1]
    if target not in dates:
        raise ValueError(f"no snapshot for {target}; newest is {dates[-1]}")
    return pd.read_parquet(_p("data", "cache",
                              f"market_master_{target.replace('-', '_')}.parquet"))


def normalize_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    return t if (not t or t.endswith((".NS", ".BO")) or "." in t) else t + ".NS"


# --------------------------------------------------------------------------
SIGNAL_LOGS = {
    "ias":            "data/ias_signal_log.csv",
    "turnaround":     "data/turnaround_watchlist.csv",
    "rs_divergence":  "data/rs_divergence_log.csv",
    "earnings_shock": "data/earnings_shock_log.csv",
    "tc":             "data/tc_log.csv",
}

DATASETS = {
    "market_snapshot":       ("parquet, one row per ticker per day", "data/cache/market_master_*.parquet"),
    "universe":              ("Nifty 1000 universe + sub-industry", "data/nifty1000_list.csv"),
    "fundamentals":          ("PE, ROE, margins, growth per ticker", "data/fundamentals_cache.csv"),
    "portfolio_snapshot":    ("OptComp-V22 holdings", "data/dna3_portfolio_snapshot.json"),
    "equity_curve":          ("daily equity, cash, holdings, regime", "data/dna3_equity_curve.csv"),
    "trade_log":             ("realised trades with exit reasons", "data/dna3_trade_log.csv"),
    "sub_industry_rotation": ("sub-industry rotation matrix history", "data/sub_industry_rotation.csv"),
    "market_mood":           ("mood/breadth history", "data/market_mood_history.csv"),
    "market_breadth":        ("% above 50/200 DMA history", "data/market_breadth_history.csv"),
    **{f"signals:{k}": ("signal log", v) for k, v in SIGNAL_LOGS.items()},
}


def list_datasets() -> Dict:
    out = {}
    for name, (desc, path) in DATASETS.items():
        if "*" in path:
            dates = snapshot_dates()
            out[name] = {"description": desc, "path": path,
                         "count": len(dates),
                         "range": [dates[0], dates[-1]] if dates else None}
        else:
            full = _p(path)
            exists = os.path.exists(full)
            rows = None
            if exists and path.endswith(".csv"):
                try:
                    rows = int(sum(1 for _ in open(full)) - 1)
                except Exception:
                    rows = None
            out[name] = {"description": desc, "path": path,
                         "available": exists, "rows": rows}
    return out


# --------------------------------------------------------------------------
def get_stock(ticker: str, date: Optional[str] = None) -> Dict:
    t = normalize_ticker(ticker)
    snap = load_snapshot(date)
    row = snap[snap["ticker"] == t]
    if row.empty:
        near = [x for x in snap["ticker"].tolist()
                if x.split(".")[0].startswith(t.split(".")[0][:4])][:8]
        return {"ticker": t, "found": False,
                "hint": "not in this snapshot", "similar": near}
    out = {"ticker": t, "found": True,
           "as_of": date or snapshot_dates()[-1],
           "data": _clean(row.iloc[0].to_dict())}
    uni = pd.read_csv(_p("data", "nifty1000_list.csv"))
    u = uni[uni["Ticker"] == t]
    if not u.empty:
        out["sub_industry"] = str(u.iloc[0]["Sub_Industry"])
    return out


def screen(min_comp_rs: Optional[float] = None, max_comp_rs: Optional[float] = None,
           sector: Optional[str] = None, sub_industry: Optional[str] = None,
           above_ma50: Optional[bool] = None, max_ma50_extension: Optional[float] = None,
           min_liquidity_cr: Optional[float] = None, limit: int = 50,
           sort_by: str = "comp_rs", date: Optional[str] = None) -> Dict:
    df = load_snapshot(date).copy()
    for c in ("comp_rs", "price", "fiftyDayAverage", "averageVolume", "trend_score"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Same display-time guard the dashboard uses: |comp_rs| > 200 is a corrupt
    # bar artefact, never a real relative strength.
    if "comp_rs" in df.columns:
        df = df[df["comp_rs"].abs() <= 200]

    if min_comp_rs is not None:
        df = df[df["comp_rs"] >= min_comp_rs]
    if max_comp_rs is not None:
        df = df[df["comp_rs"] <= max_comp_rs]
    if sector:
        df = df[df["sector"].astype(str).str.contains(sector, case=False, na=False)]
    if sub_industry:
        uni = pd.read_csv(_p("data", "nifty1000_list.csv"))
        keep = set(uni[uni["Sub_Industry"].astype(str)
                       .str.contains(sub_industry, case=False, na=False)]["Ticker"])
        df = df[df["ticker"].isin(keep)]
    if above_ma50 is not None and "fiftyDayAverage" in df.columns:
        cond = df["price"] > df["fiftyDayAverage"]
        df = df[cond if above_ma50 else ~cond]
    if max_ma50_extension is not None and "fiftyDayAverage" in df.columns:
        ext = (df["price"] / df["fiftyDayAverage"] - 1) * 100
        df = df[ext <= max_ma50_extension]
    if min_liquidity_cr is not None and "averageVolume" in df.columns:
        liq = (df["price"] * df["averageVolume"]) / 1e7
        df = df[liq >= min_liquidity_cr]

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    cols = [c for c in ["ticker", "name", "sector", "price", "comp_rs", "rs_1w",
                        "rs_1m", "rs_3m", "trend_score", "fiftyDayAverage",
                        "dist_52w", "trend_signal", "dna_signal"] if c in df.columns]
    return {"as_of": date or snapshot_dates()[-1],
            "matched": int(len(df)), "returned": int(min(limit, len(df))),
            "results": _records(df[cols], limit)}


def get_portfolio() -> Dict:
    out: Dict[str, Any] = {}
    snap = _p("data", "dna3_portfolio_snapshot.json")
    if os.path.exists(snap):
        try:
            out["snapshot"] = _clean(json.load(open(snap)))
        except Exception as e:
            out["snapshot_error"] = str(e)
    eq = _p("data", "dna3_equity_curve.csv")
    if os.path.exists(eq):
        d = pd.read_csv(eq)
        out["equity_curve_tail"] = _records(d.tail(30))
        if len(d) > 1 and "Equity" in d.columns:
            e = pd.to_numeric(d["Equity"], errors="coerce").dropna()
            if len(e) > 1:
                out["summary"] = {
                    "start": float(e.iloc[0]), "latest": float(e.iloc[-1]),
                    "total_return_pct": round((e.iloc[-1] / e.iloc[0] - 1) * 100, 2),
                    "max_drawdown_pct": round(((e / e.cummax()) - 1).min() * 100, 2),
                    "days": int(len(e)),
                }
    tl = _p("data", "dna3_trade_log.csv")
    if os.path.exists(tl):
        out["recent_trades"] = _records(pd.read_csv(tl).tail(20))
    return out


def get_signals(kind: str = "ias", limit: int = 50,
                since: Optional[str] = None) -> Dict:
    if kind not in SIGNAL_LOGS:
        return {"error": f"unknown kind '{kind}'",
                "available": sorted(SIGNAL_LOGS)}
    path = _p(SIGNAL_LOGS[kind])
    if not os.path.exists(path):
        return {"kind": kind, "available": False, "rows": 0, "signals": []}
    d = pd.read_csv(path)
    datecol = next((c for c in d.columns if "date" in c.lower()), None)
    if since and datecol:
        d = d[_parse_dates(d[datecol]) >= pd.Timestamp(since)]
    if datecol:
        d = d.sort_values(datecol, ascending=False)
    return {"kind": kind, "date_column": datecol, "rows": int(len(d)),
            "signals": _records(d, limit)}


def get_rotation(limit: int = 60, date: Optional[str] = None) -> Dict:
    path = _p("data", "sub_industry_rotation.csv")
    if not os.path.exists(path):
        return {"available": False}
    d = pd.read_csv(path)
    datecol = next((c for c in d.columns if "date" in c.lower()), None)
    if datecol:
        d[datecol] = _parse_dates(d[datecol])
        target = pd.Timestamp(date) if date else d[datecol].max()
        d = d[d[datecol] == target]
        return {"as_of": str(target.date()), "rows": int(len(d)),
                "rotation": _records(d, limit)}
    return {"rows": int(len(d)), "rotation": _records(d, limit)}


def _safe_ticker_filename(ticker: str) -> str:
    """Validate and return a path-safe ticker stem (blocks '/', '\\', '..', leading dots)."""
    raw = str(ticker or "").strip()
    if not raw or "/" in raw or "\\" in raw or ".." in raw or "\x00" in raw:
        raise ValueError(f"Invalid ticker: {ticker!r}")
    stem = raw.upper()
    for sfx in (".NS", ".BO"):
        if stem.endswith(sfx):
            stem = stem[:-3]
            break
    if not stem or stem.startswith(".") or not re.fullmatch(r"[A-Za-z0-9._&-]+", stem):
        raise ValueError(f"Invalid ticker: {ticker!r}")
    return stem


def get_valuation(ticker: str) -> Dict:
    """
    Intrinsic valuation (utils/valuation.py v4.1) for an Indian (.NS) or US stock.

    Reads the precomputed valuation snapshot row when available, runs live
    valuation estimation (`value_stock`) when fundamental inputs are present,
    and falls back safely with an explicit error if the ticker is invalid or
    missing inputs.
    """
    safe_stem = _safe_ticker_filename(ticker)
    t = normalize_ticker(ticker)
    out: Dict[str, Any] = {"ticker": t, "has_card": False}

    # Path-contained optional legacy MC card attachment (never allows traversal)
    mc_dir = os.path.realpath(_p("data", "mc_valuations"))
    card_path = os.path.realpath(os.path.join(mc_dir, f"{safe_stem}.json"))
    if not card_path.startswith(mc_dir + os.sep):
        raise ValueError(f"Invalid ticker path: {ticker!r}")
    if os.path.exists(card_path):
        try:
            with open(card_path, encoding="utf-8") as fh:
                out["legacy_card"] = _clean(json.load(fh))
        except Exception:
            pass

    # Look up latest valuation snapshot (IN or US)
    is_us = False
    for market_flag, snap_file, key in (
        (False, _p("data", "valuation_snapshot.csv"), safe_stem),
        (True, _p("data", "us_valuation_snapshot.csv"), str(ticker or "").strip().upper()),
    ):
        if os.path.exists(snap_file):
            try:
                vs = pd.read_csv(snap_file)
                hit = vs[vs["ticker"].astype(str).str.upper().str.replace(".NS", "", regex=False) == key.replace(".NS", "")]
                if not hit.empty:
                    out["snapshot_summary"] = _clean(hit.iloc[0].to_dict())
                    is_us = market_flag
                    break
            except Exception:
                pass

    try:
        import sys
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        from utils import valuation as val_mod

        market = "US" if is_us else "IN"
        ctx = val_mod.load_context(root=ROOT, market=market)
        sym = str(ticker or "").strip().upper() if market == "US" else t
        if market == "IN" and sym not in set(ctx.annual["ticker"]) if not ctx.annual.empty else False:
            us_ctx = val_mod.load_context(root=ROOT, market="US")
            us_sym = safe_stem
            if not us_ctx.annual.empty and us_sym in set(us_ctx.annual["ticker"]):
                ctx, sym, market = us_ctx, us_sym, "US"

        px = None
        if market == "IN":
            try:
                snap_df = load_snapshot()
                row = snap_df[snap_df["ticker"] == sym]
                if not row.empty:
                    px = float(pd.to_numeric(row.iloc[0].get("currentPrice", row.iloc[0].get("price")), errors="coerce"))
            except Exception:
                px = None
        if (px is None or px != px) and ctx.monthly is not None and sym in getattr(ctx.monthly, "columns", []):
            s_px = ctx.monthly[sym].dropna()
            if len(s_px):
                px = float(s_px.iloc[-1])

        res = val_mod.value_stock(sym, ctx, price=px)
        out["market"] = market
        out["has_card"] = res.get("status") == "OK"
        out["status"] = res.get("status")
        out["identity"] = res.get("identity")
        out["rating"] = val_mod.verdict(res) if res.get("status") == "OK" else res.get("reason")
        out["valuation"] = _clean({
            k: res.get(k) for k in ("snapshot", "dist", "cagr", "multiple", "scorecard", "confidence", "hurdle", "notes", "reason")
            if k in res
        })
        return out
    except Exception as e:
        out["error"] = f"valuation failed: {e}"
        return out
