"""
Business trends: how a company's sales, margins, returns, cash conversion, working capital, debt
and ownership are changing over years and quarters.

Reads the weekly Screener files (data/fundamentals_annual.csv.gz, _annual_extra, _quarterly).
Nothing is fetched here.

  annual(ticker)     one row per fiscal year: sales, growth, OPM, net margin, ROCE, cash
                     conversion (CFO / operating profit, and the 3-year cumulative figure, which
                     smooths working-capital swings), debt / equity, interest cover, fixed-asset
                     turnover, capex, and -- once the weekly refresh has collected them -- debtor,
                     inventory, payable, cash-conversion-cycle and working-capital days
  quarterly(ticker)  one row per quarter: sales, YoY, net profit, YoY, net margin, OPM (once
                     collected), promoter / FII / DII holding and shareholder count
  read(ticker)       plain-language findings, each IMPROVING / STABLE / WORSENING, and a verdict

A trend is the average of the last 3 years against the 3 before (2 v 2 with only 4-5 years).
The thresholds are judgement, not fitted: they describe direction, they do not predict returns.
Banks and lenders: cash conversion, debt and working-capital days are skipped (deposits and
borrowings are their raw material) and ROE replaces operating margin and ROCE.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

FILES = ("data/fundamentals_annual.csv.gz", "data/fundamentals_annual_extra.csv.gz")
QFILE = "data/fundamentals_quarterly.csv.gz"
UP, FLAT, DOWN = "IMPROVING", "STABLE", "WORSENING"
ICON = {UP: "🟢", FLAT: "⚪", DOWN: "🔴"}


def _mtime(paths) -> float:
    return max([os.path.getmtime(p) for p in paths if os.path.exists(p)] or [0.0])


@lru_cache(maxsize=2)
def _load(root: str, stamp: float):
    p = lambda f: os.path.join(root, f)
    A = [pd.read_csv(p(f)) for f in FILES if os.path.exists(p(f))]
    A = pd.concat(A, ignore_index=True).drop_duplicates(["ticker", "field", "period"]) if A else pd.DataFrame(
        columns=["ticker", "field", "period", "value"])
    Q = pd.read_csv(p(QFILE)) if os.path.exists(p(QFILE)) else pd.DataFrame(columns=["ticker", "field", "period", "value"])
    for d in (A, Q):
        d["period"] = pd.to_datetime(d["period"], errors="coerce")
    return {t: g for t, g in A.groupby("ticker")}, {t: g for t, g in Q.groupby("ticker")}


def _data(root: str = "."):
    return _load(root, _mtime([os.path.join(root, f) for f in FILES + (QFILE,)]))


def _key(ticker: str, d: dict) -> Optional[str]:
    for k in (ticker, ticker.replace(".NS", ""), ticker + ".NS"):
        if k in d:
            return k
    return None


def _div(a, b):
    return (a / b).where(b > 0)


def annual(ticker: str, root: str = ".") -> pd.DataFrame:
    A, _ = _data(root)
    k = _key(ticker, A)
    if k is None:
        return pd.DataFrame()
    w = A[k].pivot_table(index="period", columns="field", values="value", aggfunc="last").sort_index()
    if "sales" not in w:
        return pd.DataFrame()
    fy = w[w["sales"].notna()].copy()          # fiscal-year rows; half-year balance sheets and the price row drop out
    if fy.empty:
        return fy
    g = lambda c: fy[c] if c in fy else pd.Series(np.nan, index=fy.index)
    out = pd.DataFrame(index=fy.index)
    out["sales"] = g("sales")
    out["sales_g"] = (g("sales").pct_change() * 100).where(g("sales").shift(1) > 0)
    out["opm"] = g("opm")
    out["np"] = g("np")
    out["npm"] = _div(g("np"), g("sales")) * 100
    out["roce"] = g("roce")
    out["roe"] = _div(g("np"), (g("eq_cap") + g("reserves")).rolling(2, min_periods=1).mean()) * 100
    fin = bool(g("deposits").notna().any())
    out["financial"] = fin
    if not fin:
        out["cash_conv"] = _div(g("cfo"), g("op")) * 100
        out["cash_conv_3y"] = _div(g("cfo").rolling(3).sum(), g("op").rolling(3).sum()) * 100
        out["debt_eq"] = _div(g("borrowings"), g("eq_cap") + g("reserves"))
        out["int_cover"] = _div(g("op"), g("interest"))
        gross = g("fixed_assets") + g("cwip").fillna(0)
        out["fa_turn"] = _div(g("sales"), gross.rolling(2, min_periods=1).mean())
        out["capex"] = gross.diff() + g("dep")
        for c in ("debtor_days", "inventory_days", "payable_days", "ccc_days", "wc_days"):
            out[c] = g(c)
    out["payout"] = g("payout")
    return out


def quarterly(ticker: str, root: str = ".") -> pd.DataFrame:
    _, Q = _data(root)
    k = _key(ticker, Q)
    if k is None:
        return pd.DataFrame()
    w = Q[k].pivot_table(index="period", columns="field", values="value", aggfunc="last").sort_index()
    w = w[w.index.month.isin([3, 6, 9, 12])]
    g = lambda c: w[c] if c in w else pd.Series(np.nan, index=w.index)
    yoy = lambda s: ((s / s.shift(4) - 1) * 100).where(s.shift(4) > 0)
    out = pd.DataFrame(index=w.index)
    out["sales"], out["sales_yoy"] = g("sales"), yoy(g("sales"))
    out["op"], out["opm"] = g("op"), g("opm")
    out["np"], out["np_yoy"] = g("np"), yoy(g("np"))
    out["npm"] = _div(g("np"), g("sales")) * 100
    for c in ("promoters", "fii", "dii", "public", "holders"):
        out[c] = g(c)
    return out


def _trend(s: pd.Series, tol: float, higher_better: bool = True, rel: bool = False) -> Optional[Dict[str, Any]]:
    """Recent average vs the average before it. tol in the series' own units (or % if rel)."""
    s = s.dropna()
    if len(s) < 4:
        return None
    n = 3 if len(s) >= 6 else 2
    now, was = float(s.iloc[-n:].mean()), float(s.iloc[-2 * n:-n].mean())
    d = (now / was - 1) * 100 if rel and was else now - was
    if rel and not was:
        return None
    way = FLAT if abs(d) < tol else (UP if (d > 0) == higher_better else DOWN)
    return dict(now=now, was=was, delta=d, way=way, n=n, last=float(s.iloc[-1]))


def read(ticker: str, root: str = ".") -> Dict[str, Any]:
    """Findings (list of dicts: metric, way, text), verdict, counts and the latest quarter."""
    a, q = annual(ticker, root), quarterly(ticker, root)
    F: List[Dict[str, Any]] = []
    add = lambda m, way, text: F.append(dict(metric=m, way=way, text=text))
    if not a.empty:
        fin = bool(a["financial"].iloc[-1])
        yrs = f"last {{n}} yrs vs the {{n}} before"
        s = a["sales_g"].dropna()
        if len(s) >= 4:
            n = 3 if len(s) >= 6 else 2
            now, was = s.iloc[-n:].mean(), s.iloc[-2 * n:-n].mean()
            way = FLAT if abs(now - was) < 4 else UP if now > was else DOWN
            add("Sales growth", way, f"{now:.0f}% a year over the last {n} yrs vs {was:.0f}% before")
        for col, lab, tol, unit in (("opm", "Operating margin", 1.5, "%"), ("roce", "ROCE", 2.0, "%"),
                                    ("roe", "ROE", 2.0, "%"), ("npm", "Net margin", 1.0, "%")):
            if (fin and col in ("opm", "roce")) or (not fin and col == "roe"):
                continue       # Screener's financing margin excludes other income and is often negative
            t = _trend(a[col], tol) if col in a else None
            if t:
                add(lab, t["way"], f"{t['now']:.0f}{unit} ({yrs.format(n=t['n'])}: {t['was']:.0f}{unit}); latest {t['last']:.0f}{unit}")
        if not fin:
            cc = a["cash_conv_3y"].dropna() if "cash_conv_3y" in a else pd.Series(dtype=float)
            if len(cc):
                now = float(cc.iloc[-1])
                was = float(cc.iloc[-4]) if len(cc) >= 4 else np.nan
                way = DOWN if now < 50 or (np.isfinite(was) and now < was - 20) else UP if (
                    np.isfinite(was) and now > was + 20 and now >= 60) else FLAT
                add("Cash conversion", way, f"{now:.0f}% of operating profit became operating cash over 3 yrs"
                    + (f" ({was:.0f}% three years ago)" if np.isfinite(was) else "")
                    + (" — under half, profits are not turning into cash" if now < 50 else ""))
            t = _trend(a["debt_eq"], 0.15, higher_better=False)
            if t and (t["now"] > 0.1 or t["was"] > 0.1):
                add("Debt / equity", t["way"], f"{t['last']:.2f} now; {t['now']:.2f} avg vs {t['was']:.2f} before")
            for col, lab in (("ccc_days", "Cash conversion cycle"), ("inventory_days", "Inventory days"),
                             ("debtor_days", "Debtor days")):
                t = _trend(a[col], 10, higher_better=False) if col in a else None
                if t:
                    add(lab, t["way"], f"{t['last']:.0f} days now; {t['now']:.0f} avg vs {t['was']:.0f} before")
            t = _trend(a["fa_turn"], 12, rel=True)
            if t:
                add("Fixed-asset turnover", t["way"], f"{t['last']:.1f}x sales per rupee of fixed assets; avg {t['now']:.1f}x vs {t['was']:.1f}x")
    latest: Dict[str, Any] = {}
    if not q.empty:
        r = q.dropna(subset=["sales", "np"])
        if len(r):
            x = r.iloc[-1]
            latest = dict(quarter=r.index[-1].strftime("%b %Y"), sales=x.sales, sales_yoy=x.sales_yoy, np=x.np,
                          np_yoy=x.np_yoy, npm=x.npm, opm=x.opm)
            if len(r) >= 8:
                g4, p4 = r.sales_yoy.iloc[-2:].mean(), r.sales_yoy.iloc[-6:-2].mean()
                if np.isfinite(g4) and np.isfinite(p4):
                    way = FLAT if abs(g4 - p4) < 5 else UP if g4 > p4 else DOWN
                    add("Quarterly sales momentum", way, f"last 2 quarters {g4:+.0f}% YoY vs {p4:+.0f}% in the 4 before")
                m = r["opm"] if r["opm"].notna().sum() >= 8 else r["npm"]
                lab = "Quarterly operating margin" if m is r["opm"] else "Quarterly net margin"
                m4, mp = m.iloc[-4:].mean(), m.iloc[-8:-4].mean()
                if np.isfinite(m4) and np.isfinite(mp):
                    way = FLAT if abs(m4 - mp) < 1 else UP if m4 > mp else DOWN
                    add(lab, way, f"{m4:.1f}% over the last 4 quarters vs {mp:.1f}% the 4 before; latest {m.iloc[-1]:.1f}%")
        for col, lab, tol in (("promoters", "Promoter holding", 0.5), ("fii", "FII holding", 0.75), ("dii", "DII holding", 0.75)):
            s = q[col].dropna()
            if len(s) >= 5:
                d = float(s.iloc[-1] - s.iloc[-5])
                add(lab, FLAT if abs(d) < tol else UP if d > 0 else DOWN, f"{s.iloc[-1]:.1f}% ({d:+.1f} pp over a year)")
    up, dn = sum(f["way"] == UP for f in F), sum(f["way"] == DOWN for f in F)
    verdict = "NO DATA" if not F else "IMPROVING" if up >= dn + 2 else "WORSENING" if dn >= up + 2 else "MIXED" if (
        up and dn) else "STABLE"
    return dict(findings=F, verdict=verdict, up=up, down=dn, latest=latest, years=len(a), quarters=len(q),
                financial=bool(a["financial"].iloc[-1]) if not a.empty else False,
                has_wc_days=bool(not a.empty and any(c in a and a[c].notna().any() for c in ("ccc_days", "inventory_days", "debtor_days"))),
                has_q_opm=bool(not q.empty and q["opm"].notna().any()))


@lru_cache(maxsize=4096)
def _brief(ticker: str, root: str, stamp: float) -> Dict[str, Any]:
    r = read(ticker, root)
    a = annual(ticker, root)
    flags: List[str] = []
    if not a.empty and not r["financial"]:
        cc = a["cash_conv_3y"].dropna()
        de = a["debt_eq"].dropna()
        if len(cc) and cc.iloc[-1] < 50:
            flags.append(f"cash conversion {cc.iloc[-1]:.0f}%")
        if len(de) and de.iloc[-1] > 1.5:
            flags.append(f"debt/equity {de.iloc[-1]:.1f}")
    lq = r["latest"]
    if lq and pd.notna(lq.get("np_yoy")) and lq["np_yoy"] < -25:
        flags.append(f"{lq['quarter']} profit {lq['np_yoy']:+.0f}% YoY")
    roce = a["roce"].dropna().iloc[-1] if not a.empty and a["roce"].notna().any() else np.nan
    return dict(verdict=r["verdict"], up=r["up"], down=r["down"], flags=flags, roce=roce,
                quarter=lq.get("quarter") if lq else None, sales_yoy=lq.get("sales_yoy") if lq else np.nan,
                np_yoy=lq.get("np_yoy") if lq else np.nan,
                ok=r["verdict"] not in ("WORSENING", "NO DATA") and not any(f.startswith(("cash", "debt")) for f in flags))


def brief(ticker: str, root: str = ".") -> Dict[str, Any]:
    """Verdict, red flags, latest quarter and a pass / fail for the funnel's business stage. Cached."""
    return _brief(ticker, root, _mtime([os.path.join(root, f) for f in FILES + (QFILE,)]))
