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
UP, FLAT, DOWN, BUILD = "IMPROVING", "STABLE", "WORSENING", "INVESTING"
ICON = {UP: "🟢", FLAT: "⚪", DOWN: "🔴", BUILD: "🏗️"}
WEIGHTS = {"Growth": 0.30, "Margins": 0.25, "Cash & balance sheet": 0.25, "Returns on capital": 0.20}
LABELS = ((70, "STRONG"), (55, "HEALTHY"), (40, "MIXED"), (0, "WEAK"))
PASS_SCORE = 45            # the funnel's business stage passes at or above this, with no hard flag
INVEST_JUMP = 50           # capital employed up this % in one of the last 3 years = investment phase
INVEST_CWIP = 20           # or work-in-progress at least this % of capital employed


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
        ce = g("eq_cap") + g("reserves") + g("borrowings").fillna(0)
        out["cap_employed"] = ce
        out["ce_growth"] = (ce.pct_change() * 100).where(ce.shift(1) > 0)
        out["cwip_share"] = _div(g("cwip").fillna(0), ce) * 100
        # ROCE on capital that is already working: operating profit after depreciation over average
        # capital employed less work-in-progress (assets not yet earning)
        out["roce_core"] = _div(g("op") - g("dep").fillna(0), (ce - g("cwip").fillna(0)).rolling(2, min_periods=1).mean()) * 100
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


def _lin(x: float, pts) -> float:
    """Piecewise-linear map of x through (x, score) points, clipped at the ends."""
    if x is None or not np.isfinite(x):
        return np.nan
    xs, ys = zip(*pts)
    return float(np.interp(x, xs, ys))


def _mean(vals, wts=None) -> float:
    v = np.array(vals, dtype=float)
    w = np.ones(len(v)) if wts is None else np.array(wts, dtype=float)
    ok = np.isfinite(v)
    return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan


def investment_phase(a: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    A large recent addition to capital (capex or an acquisition) that has not had time to earn.
    True when capital employed rose INVEST_JUMP% or more in any of the last 3 years, or
    work-in-progress is INVEST_CWIP% or more of capital employed.
    """
    if a.empty or "ce_growth" not in a or a["financial"].iloc[-1]:
        return None
    g = a["ce_growth"].dropna().tail(3)
    cw = a["cwip_share"].dropna()
    big = g[g >= INVEST_JUMP]
    if len(big):
        y = big.index[-1]
        prev = a["cap_employed"].shift(1).get(y)
        return dict(year=y.strftime("FY%y") if y.month == 3 else y.strftime("%b %y"), jump=float(big.iloc[-1]),
                    text=f"capital employed rose {big.iloc[-1]:.0f}% in {y.strftime('FY%y')} "
                         f"(₹{prev:,.0f} cr to ₹{a['cap_employed'].get(y):,.0f} cr)")
    if len(cw) and cw.iloc[-1] >= INVEST_CWIP:
        return dict(year="now", jump=np.nan, text=f"work-in-progress is {cw.iloc[-1]:.0f}% of capital employed")
    return None


def read(ticker: str, root: str = ".") -> Dict[str, Any]:
    """
    Score the business 0-100 on four pillars and describe what is changing.

      Growth (30%)                3-year sales growth, latest two quarters' sales YoY, last-4-quarter profit YoY
      Margins (25%)               operating margin: last 3 years v the 3 before, and last 4 quarters v the 4 before
      Cash & balance sheet (25%)  3-year cash conversion, debt / equity, interest cover   (skipped for lenders)
      Returns on capital (20%)    ROCE level (ROE for lenders) and its change

    Each pillar is LEVEL plus DIRECTION, so a falling but still-good number is not scored like a bad
    one. One cause counts once: net margin, fixed-asset turnover and ownership are shown but not scored
    (they move with margins, capital and investor mix). During an investment phase (investment_phase)
    the fall in ROCE that a larger capital base causes is not penalised; ROCE on capital already at
    work is shown instead, and the items affected are marked INVESTING.

    verdict   STRONG >= 70, HEALTHY >= 55, MIXED >= 40, WEAK below
    direction IMPROVING / STEADY / SLIPPING from the scored pillars' directions
    The weights and break-points are judgement, not fitted to returns.
    """
    a, q = annual(ticker, root), quarterly(ticker, root)
    F: List[Dict[str, Any]] = []
    add = lambda pillar, m, way, text, scored=True: F.append(dict(pillar=pillar, metric=m, way=way, text=text, scored=scored))
    P: Dict[str, Dict[str, Any]] = {}
    latest: Dict[str, Any] = {}
    fin = bool(a["financial"].iloc[-1]) if not a.empty else False
    inv = investment_phase(a)
    r = q.dropna(subset=["sales", "np"]) if not q.empty else q

    # ---------------------------------------------------------------- growth
    lv, dr = [], []
    if not a.empty:
        s = a["sales_g"].dropna()
        if len(s) >= 4:
            n = 3 if len(s) >= 6 else 2
            now, was = float(s.iloc[-n:].mean()), float(s.iloc[-2 * n:-n].mean())
            way = FLAT if abs(now - was) < 4 else UP if now > was else DOWN
            add("Growth", "Sales growth", way, f"{now:.0f}% a year over the last {n} yrs vs {was:.0f}% before")
            lv.append(_lin(now, [(-5, 0), (0, 20), (10, 55), (20, 85), (30, 100)]))
            dr.append({UP: 1, FLAT: 0, DOWN: -1}[way])
    if len(r):
        x = r.iloc[-1]
        latest = dict(quarter=r.index[-1].strftime("%b %Y"), sales=x.sales, sales_yoy=x.sales_yoy, np=x.np, np_yoy=x.np_yoy,
                      npm=x.npm, opm=x.opm)
    if len(r) >= 8:
        g2, p4 = float(r.sales_yoy.iloc[-2:].mean()), float(r.sales_yoy.iloc[-6:-2].mean())
        if np.isfinite(g2) and np.isfinite(p4):
            way = FLAT if abs(g2 - p4) < 5 else UP if g2 > p4 else DOWN
            add("Growth", "Quarterly sales", way, f"last 2 quarters {g2:+.0f}% YoY vs {p4:+.0f}% in the 4 before")
            lv.append(_lin(g2, [(-10, 0), (0, 25), (10, 55), (20, 85), (30, 100)]))
            dr.append({UP: 1, FLAT: 0, DOWN: -1}[way])
        t4, t8 = float(r.np.iloc[-4:].sum()), float(r.np.iloc[-8:-4].sum())
        if t8 > 0:
            pg = (t4 / t8 - 1) * 100
            add("Growth", "Profit, last 4 quarters", UP if pg > 10 else DOWN if pg < -5 else FLAT,
                f"₹{t4:,.0f} cr, {pg:+.0f}% on the 4 quarters before")
            lv.append(_lin(pg, [(-25, 0), (0, 35), (15, 65), (30, 90), (50, 100)]))
    if lv:
        P["Growth"] = dict(score=_mean(lv), dir=_mean(dr) if dr else 0.0)

    # ---------------------------------------------------------------- margins
    lv, dr = [], []
    if not a.empty and not fin:
        t = _trend(a["opm"], 1.5)
        if t:
            add("Margins", "Operating margin", t["way"], f"{t['now']:.0f}% (last {t['n']} yrs vs the {t['n']} before: {t['was']:.0f}%); latest {t['last']:.0f}%")
            lv.append(_lin(t["delta"], [(-8, 0), (-3, 35), (0, 62), (3, 85), (6, 100)]))
            dr.append({UP: 1, FLAT: 0, DOWN: -1}[t["way"]])
    if len(r) >= 8:
        opm_ok = r["opm"].notna().sum() >= 8
        m = r["opm"] if opm_ok else r["npm"]
        m4, mp = float(m.iloc[-4:].mean()), float(m.iloc[-8:-4].mean())
        if np.isfinite(m4) and np.isfinite(mp):
            way = FLAT if abs(m4 - mp) < 1 else UP if m4 > mp else DOWN
            add("Margins", "Quarterly operating margin" if opm_ok else "Quarterly net margin", way,
                f"{m4:.1f}% over the last 4 quarters vs {mp:.1f}% the 4 before; latest {m.iloc[-1]:.1f}%")
            lv.append(_lin(m4 - mp, [(-5, 0), (-2, 35), (0, 62), (2, 85), (4, 100)]))
            dr.append({UP: 1, FLAT: 0, DOWN: -1}[way])
    if not a.empty:
        t = _trend(a["npm"], 1.0)
        if t:
            dil = inv is not None and t["way"] == DOWN
            add("Margins", "Net margin", BUILD if dil else t["way"],
                f"{t['now']:.0f}% (last {t['n']} yrs vs the {t['n']} before: {t['was']:.0f}%); latest {t['last']:.0f}%"
                + (" — interest and depreciation on the new capital" if dil else ""), scored=False)
    if lv:
        P["Margins"] = dict(score=_mean(lv), dir=_mean(dr))

    # ---------------------------------------------------------------- cash and balance sheet
    hard: List[str] = []
    if not a.empty and not fin:
        lv, dr = [], []
        cc = a["cash_conv_3y"].dropna()
        if len(cc):
            now = float(cc.iloc[-1])
            was = float(cc.iloc[-4]) if len(cc) >= 4 else np.nan
            way = DOWN if now < 50 or (np.isfinite(was) and now < was - 20) else UP if (
                np.isfinite(was) and now > was + 15 and now >= 60) else FLAT
            add("Cash & balance sheet", "Cash conversion", way, f"{now:.0f}% of operating profit became operating cash over 3 yrs"
                + (f" ({was:.0f}% three years ago)" if np.isfinite(was) else "")
                + (" — under half, profits are not turning into cash" if now < 50 else ""))
            lv += [_lin(now, [(20, 0), (50, 35), (70, 62), (90, 88), (110, 100)])] * 2          # double weight
            dr.append({UP: 1, FLAT: 0, DOWN: -1}[way])
            if now < 50:
                hard.append(f"cash conversion {now:.0f}%")
        de = a["debt_eq"].dropna()
        if len(de):
            d = float(de.iloc[-1])
            t = _trend(a["debt_eq"], 0.15, higher_better=False)
            way = t["way"] if t and (t["now"] > 0.1 or t["was"] > 0.1) else FLAT
            build = inv is not None and way == DOWN and d <= 1.0
            add("Cash & balance sheet", "Debt / equity", BUILD if build else way,
                f"{d:.2f} now" + (f"; {t['now']:.2f} avg vs {t['was']:.2f} before" if t else "")
                + (" — debt taken on for the investment, still moderate" if build else ""))
            lv.append(_lin(d, [(0, 100), (0.3, 88), (0.6, 70), (1.0, 48), (1.5, 25), (2.5, 0)]))
            if not build:
                dr.append({UP: 1, FLAT: 0, DOWN: -1}[way])
            if d > 1.5:
                hard.append(f"debt/equity {d:.1f}")
        ic = a["int_cover"].dropna()
        if len(ic) and len(de) and de.iloc[-1] > 0.1:
            c = float(ic.iloc[-1])
            add("Cash & balance sheet", "Interest cover", DOWN if c < 3 else FLAT, f"operating profit is {c:.1f}× interest")
            lv.append(_lin(c, [(1, 0), (2, 20), (4, 55), (8, 85), (15, 100)]))
        for col, lab in (("ccc_days", "Cash conversion cycle"), ("inventory_days", "Inventory days"), ("debtor_days", "Debtor days")):
            t = _trend(a[col], 10, higher_better=False) if col in a else None
            if t:
                add("Cash & balance sheet", lab, t["way"], f"{t['last']:.0f} days now; {t['now']:.0f} avg vs {t['was']:.0f} before")
                if col == "ccc_days":
                    dr.append({UP: 1, FLAT: 0, DOWN: -1}[t["way"]])
        if lv:
            P["Cash & balance sheet"] = dict(score=_mean(lv), dir=_mean(dr) if dr else 0.0)

    # ---------------------------------------------------------------- returns on capital
    if not a.empty:
        col = "roe" if fin else "roce"
        t = _trend(a[col], 2.0)
        if t:
            level = _lin(t["last"], [(4, 0), (8, 25), (15, 62), (20, 82), (28, 100)] if fin else [(5, 0), (10, 30), (15, 55), (22, 80), (32, 100)])
            adj = float(np.clip(t["delta"] * 2, -20, 15))
            build = inv is not None and t["way"] == DOWN
            core = a["roce_core"].dropna() if "roce_core" in a else pd.Series(dtype=float)
            txt = f"{t['now']:.0f}% (last {t['n']} yrs vs the {t['n']} before: {t['was']:.0f}%); latest {t['last']:.0f}%"
            if build:
                adj = max(adj, -5.0)
                txt += f" — diluted: {inv['text']}"
                if len(core):
                    level = max(level, _lin(float(core.iloc[-1]), [(5, 0), (10, 30), (15, 55), (22, 80), (32, 100)]))
            add("Returns on capital", "ROE" if fin else "ROCE", BUILD if build else t["way"], txt)
            P["Returns on capital"] = dict(score=float(np.clip(level + adj, 0, 100)), dir=0.0 if build else {UP: 1, FLAT: 0, DOWN: -1}[t["way"]])
        if not fin:
            if inv is not None:
                core = a["roce_core"].dropna()
                add("Returns on capital", "Investment phase", BUILD, inv["text"][0].upper() + inv["text"][1:]
                    + (f". ROCE on capital already at work: {core.iloc[-1]:.0f}%" if len(core) else "")
                    + ". Watch for returns to recover as the new assets start earning", scored=False)
            t = _trend(a["fa_turn"], 12, rel=True)
            if t:
                build = inv is not None and t["way"] == DOWN
                add("Returns on capital", "Fixed-asset turnover", BUILD if build else t["way"],
                    f"{t['last']:.1f}x sales per rupee of fixed assets; avg {t['now']:.1f}x vs {t['was']:.1f}x", scored=False)

    # ---------------------------------------------------------------- ownership (shown, not scored)
    own: List[Dict[str, Any]] = []
    if not q.empty:
        for col, lab, tol in (("promoters", "Promoter holding", 0.5), ("fii", "FII holding", 0.75), ("dii", "DII holding", 0.75)):
            s = q[col].dropna()
            if len(s) >= 5:
                d = float(s.iloc[-1] - s.iloc[-5])
                own.append(dict(pillar="Ownership", metric=lab, way=FLAT if abs(d) < tol else UP if d > 0 else DOWN,
                                text=f"{s.iloc[-1]:.1f}% ({d:+.1f} pp over a year)", scored=False))

    w = {k: WEIGHTS[k] for k in P}
    score = float(sum(P[k]["score"] * w[k] for k in P) / sum(w.values())) if P else np.nan
    drift = float(sum(P[k]["dir"] * w[k] for k in P) / sum(w.values())) if P else 0.0
    verdict = "NO DATA" if not P else next(lab for cut, lab in LABELS if round(score) >= cut)
    if not any(f["way"] == BUILD and f["metric"] != "Investment phase" for f in F):      # nothing was discounted: not worth a tag
        inv = None
        F = [f for f in F if f["metric"] != "Investment phase"]
    direction = "" if not P else "IMPROVING" if drift > 0.25 else "SLIPPING" if drift < -0.25 else "STEADY"
    sc = [f for f in F if f["scored"]]
    return dict(findings=F, ownership=own, pillars={k: dict(score=round(v["score"]), dir=round(v["dir"], 2)) for k, v in P.items()},
                score=None if not P else round(score), verdict=verdict, direction=direction, investing=inv, hard_flags=hard,
                up=sum(f["way"] == UP for f in sc), down=sum(f["way"] == DOWN for f in sc), latest=latest, years=len(a),
                quarters=len(q), financial=fin,
                has_wc_days=bool(not a.empty and any(c in a and a[c].notna().any() for c in ("ccc_days", "inventory_days", "debtor_days"))),
                has_q_opm=bool(not q.empty and q["opm"].notna().any()))


def headline(r: Dict[str, Any]) -> str:
    """'Healthy 62/100 · steady · investment phase'"""
    if r["verdict"] == "NO DATA":
        return "No data"
    return (f"{r['verdict'].title()} {r['score']}/100 · {r['direction'].lower()}" + (" · investment phase" if r["investing"] else ""))


@lru_cache(maxsize=4096)
def _brief(ticker: str, root: str, stamp: float) -> Dict[str, Any]:
    r = read(ticker, root)
    a = annual(ticker, root)
    flags = list(r["hard_flags"])
    lq = r["latest"]
    if lq and pd.notna(lq.get("np_yoy")) and lq["np_yoy"] < -25:
        flags.append(f"{lq['quarter']} profit {lq['np_yoy']:+.0f}% YoY")
    roce = a["roce"].dropna().iloc[-1] if not a.empty and a["roce"].notna().any() else np.nan
    return dict(verdict=r["verdict"], score=r["score"], direction=r["direction"], investing=bool(r["investing"]),
                headline=headline(r), up=r["up"], down=r["down"], flags=flags, roce=roce,
                quarter=lq.get("quarter") if lq else None, sales_yoy=lq.get("sales_yoy") if lq else np.nan,
                np_yoy=lq.get("np_yoy") if lq else np.nan,
                ok=r["score"] is not None and r["score"] >= PASS_SCORE and not r["hard_flags"])


def brief(ticker: str, root: str = ".") -> Dict[str, Any]:
    """Score, verdict, red flags, latest quarter and a pass / fail for the funnel's business stage. Cached."""
    return _brief(ticker, root, _mtime([os.path.join(root, f) for f in FILES + (QFILE,)]))
