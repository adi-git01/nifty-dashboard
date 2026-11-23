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

# core, then the extended list's weekly files (build_extended_universe.py), then the older one-off extra file;
# the first file that has a (ticker, field, period) wins
FILES = ("data/fundamentals_annual.csv", "data/fundamentals_ext_annual.csv", "data/fundamentals_annual_extra.csv")
QFILE = "data/fundamentals_quarterly.csv"
QFILES = (QFILE, "data/fundamentals_ext_quarterly.csv")
UP, FLAT, DOWN, BUILD = "IMPROVING", "STABLE", "WORSENING", "INVESTING"
ICON = {UP: "🟢", FLAT: "⚪", DOWN: "🔴", BUILD: "🏗️"}
WEIGHTS = {"Growth": 0.30, "Margins": 0.25, "Cash & balance sheet": 0.25, "Returns on capital": 0.20}
LABELS = ((70, "STRONG"), (55, "HEALTHY"), (40, "MIXED"), (0, "WEAK"))
SCORE_VERSION = "s4"       # bump when the scoring changes (invalidates data/business_scores.csv)
STALE_DAYS = 640           # ~21 months after the fiscal year end: a year's results should have come in by then
MIN_PILLARS = 3            # of 4
PASS_SCORE = 45            # the funnel's business stage passes at or above this, with no hard flag
INVEST_JUMP = 50           # capital employed up this % in one of the last 3 years = investment phase
INVEST_CWIP = 20           # or work-in-progress at least this % of capital employed


def _mtime(paths) -> float:
    return max([os.path.getmtime(p) for p in paths if os.path.exists(p)] or [0.0])


def _resolved_paths(root: str) -> tuple[list[str], list[str]]:
    from utils import fund_paths
    return ([fund_paths.resolve(f, root=root) for f in FILES],
            [fund_paths.resolve(f, root=root) for f in QFILES])


@lru_cache(maxsize=2)
def _load(root: str, stamp: float):
    a_files, q_files = _resolved_paths(root)
    A = [pd.read_csv(f) for f in a_files if os.path.exists(f)]
    A = pd.concat(A, ignore_index=True).drop_duplicates(["ticker", "field", "period"]) if A else pd.DataFrame(
        columns=["ticker", "field", "period", "value"])
    Q = [pd.read_csv(f) for f in q_files if os.path.exists(f)]
    Q = pd.concat(Q, ignore_index=True).drop_duplicates(["ticker", "field", "period"]) if Q else pd.DataFrame(
        columns=["ticker", "field", "period", "value"])
    for d in (A, Q):
        d["period"] = pd.to_datetime(d["period"], errors="coerce")
    return {t: g for t, g in A.groupby("ticker")}, {t: g for t, g in Q.groupby("ticker")}


def _data(root: str = "."):
    a_files, q_files = _resolved_paths(root)
    return _load(root, _mtime(a_files + q_files))


def _key(ticker: str, d: dict) -> Optional[str]:
    for k in (ticker, ticker.replace(".NS", ""), ticker + ".NS"):
        if k in d:
            return k
    return None


def _div(a, b):
    return (a / b).where(b > 0)


GOLD_LOAN = {"MUTHOOTFIN.NS", "MANAPPURAM.NS"}
LENDER_WEIGHTS = {"Growth": 0.25, "Spread & costs": 0.25, "ROA & ROE": 0.30, "Leverage & asset quality": 0.20}
# score break-points by lender type: (value, score) pairs
ROA_PTS = {"BANK": [(0, 0), (0.5, 30), (1.0, 60), (1.5, 85), (2.0, 100)],
           "NBFC": [(0, 0), (1.0, 30), (2.0, 60), (3.0, 85), (4.5, 100)],
           "GOLD LOAN": [(0, 0), (2.0, 30), (3.5, 60), (5.0, 85), (7.0, 100)]}
NII_PTS = {"BANK": [(1.5, 10), (2.5, 50), (3.5, 85), (4.5, 100)],
           "NBFC": [(2, 10), (4, 50), (6, 80), (9, 100)],
           "GOLD LOAN": [(5, 20), (8, 60), (11, 90), (14, 100)]}
LEV_PTS = {"BANK": [(6, 100), (10, 75), (14, 45), (18, 15), (22, 0)],
           "NBFC": [(3, 100), (5, 75), (7, 45), (10, 10), (13, 0)],
           "GOLD LOAN": [(2, 100), (4, 70), (6, 35), (8, 0)]}
LEV_MAX = {"BANK": 18, "NBFC": 10, "GOLD LOAN": 7}


def lender_kind(ticker: str, w: pd.DataFrame) -> Optional[str]:
    """
    BANK (takes deposits), GOLD LOAN, NBFC, or None for a non-lender. An NBFC is a company in the
    Finance sub-industry whose interest cost is 15%+ of revenue with assets at least 1.5x equity,
    which keeps out holding companies, brokers and asset managers.
    """
    if "deposits" in w and w["deposits"].notna().any():
        return "BANK"
    t = ticker if ticker.endswith(".NS") else ticker + ".NS"
    if t in GOLD_LOAN:
        return "GOLD LOAN"
    try:
        from utils.nifty1000_list import SUB_INDUSTRY_MAP
        ind = SUB_INDUSTRY_MAP.get(t)
    except Exception:
        ind = None
    if ind != "Finance" or "sales" not in w:
        return None
    x = w[w["sales"].notna()]
    if x.empty:
        return None
    x = x.iloc[-1]
    eq = x.get("eq_cap", np.nan) + x.get("reserves", np.nan)
    ok = (x["sales"] > 0 and pd.notna(x.get("interest")) and x["interest"] / x["sales"] >= 0.15
          and eq > 0 and pd.notna(x.get("total_liab")) and x["total_liab"] / eq >= 1.5)
    return "NBFC" if ok else None


def roa_tree(ticker: str, root: str = ".", as_of=None) -> pd.DataFrame:
    """
    The ROA tree for a lender, one row per fiscal year, every line as % of average total assets:

        interest income - interest cost = NII
        NII + other income - costs (operating expenses, provisions, depreciation) = pre-tax ROA
        pre-tax ROA - tax = ROA;   ROA x leverage (average assets / average equity) = ROE

    Screener reports operating expenses and provisions as one "Expenses" line, so they are one
    line here too. Empty for a non-lender.
    """
    A, _ = _data(root)
    k = _key(ticker, A)
    if k is None:
        return pd.DataFrame()
    w = A[k].pivot_table(index="period", columns="field", values="value", aggfunc="last").sort_index()
    if as_of is not None:
        w = w[w.index <= pd.Timestamp(as_of)]
    kind = lender_kind(k, w)
    need = {"sales", "interest", "op", "pbt", "np", "total_liab", "eq_cap", "reserves"}
    if kind is None or not need <= set(w.columns):
        return pd.DataFrame()
    y = w[w["sales"].notna()].copy()
    if len(y) < 2:
        return pd.DataFrame()
    dep = y["dep"].fillna(0) if "dep" in y else 0.0
    avg_a = y["total_liab"].rolling(2, min_periods=1).mean()
    eq = y["eq_cap"] + y["reserves"]
    avg_e = eq.rolling(2, min_periods=1).mean()
    pc = lambda v: _div(v, avg_a) * 100
    T = pd.DataFrame(index=y.index)
    T["int_income"] = pc(y["sales"])
    T["int_cost"] = pc(y["interest"])
    T["nii"] = T["int_income"] - T["int_cost"]
    # Screener shows lenders in one of two layouts and the scraper files both profit lines as "op":
    #   bank layout:     Financing profit = revenue - interest - expenses;  PBT = op + other income - dep
    #   standard layout: Operating profit = revenue - expenses (before interest);  PBT = op + other income - interest - dep
    # Reading a standard-layout NBFC with the bank formula made other income about -4% of assets and costs 4 pp too
    # low (BAJAJFINSV). Each year takes the layout that leaves the smaller other income, which is a small line for lenders.
    oi_bank = y["pbt"] - y["op"] + dep
    oi_std = oi_bank + y["interest"]
    std = oi_std.abs() < oi_bank.abs()
    T["other_income"] = pc(oi_bank.where(~std, oi_std))
    T["costs"] = pc((y["sales"] - y["interest"] - y["op"] + dep).where(~std, y["sales"] - y["op"] + dep))
    T.attrs["layout"] = "standard" if bool(std.iloc[-1]) else "bank"
    T["pretax_roa"] = pc(y["pbt"])
    T["tax"] = pc(y["pbt"] - y["np"])
    T["roa"] = pc(y["np"])
    T["leverage"] = _div(avg_a, avg_e)
    T["roe"] = _div(y["np"], avg_e) * 100
    T["cost_income"] = _div(T["costs"], T["nii"] + T["other_income"]) * 100
    T["assets"] = y["total_liab"]
    T["asset_g"] = (y["total_liab"].pct_change(fill_method=None) * 100).where(y["total_liab"].shift(1) > 0)
    T.attrs["kind"] = kind
    return T


def annual(ticker: str, root: str = ".", as_of=None) -> pd.DataFrame:
    """as_of: use only fiscal years ending on or before this date (for point-in-time tests).
    The result is cached and shared: treat it as read-only."""
    a_files, q_files = _resolved_paths(root)
    return _annual(ticker, root, None if as_of is None else str(as_of), _mtime(a_files + q_files))


@lru_cache(maxsize=8192)
def _annual(ticker: str, root: str, as_of, stamp: float) -> pd.DataFrame:
    A, _ = _data(root)
    k = _key(ticker, A)
    if k is None:
        return pd.DataFrame()
    w = A[k].pivot_table(index="period", columns="field", values="value", aggfunc="last").sort_index()
    if as_of is not None:
        w = w[w.index <= pd.Timestamp(as_of)]
    if "sales" not in w:
        return pd.DataFrame()
    fy = w[w["sales"].notna()].copy()          # fiscal-year rows; half-year balance sheets and the price row drop out
    if fy.empty:
        return fy
    g = lambda c: fy[c] if c in fy else pd.Series(np.nan, index=fy.index)
    out = pd.DataFrame(index=fy.index)
    out["sales"] = g("sales")
    out["sales_g"] = (g("sales").pct_change(fill_method=None) * 100).where(g("sales").shift(1) > 0)
    out["op"] = g("op")
    out["opm"] = g("opm")
    out["np"] = g("np")
    out["npm"] = _div(g("np"), g("sales")) * 100
    out["roce"] = g("roce")
    out["roe"] = _div(g("np"), (g("eq_cap") + g("reserves")).rolling(2, min_periods=1).mean()) * 100
    kind = lender_kind(k, w)
    fin = kind is not None
    out["financial"] = fin
    out["lender_kind"] = kind or ""
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
        out["ce_growth"] = (ce.pct_change(fill_method=None) * 100).where(ce.shift(1) > 0)
        out["cwip_share"] = _div(g("cwip").fillna(0), ce) * 100
        out["gross_block"] = gross
        # ROCE on capital that is already working: operating profit after depreciation over average
        # capital employed less work-in-progress (assets not yet earning)
        out["roce_core"] = _div(g("op") - g("dep").fillna(0), (ce - g("cwip").fillna(0)).rolling(2, min_periods=1).mean()) * 100
    out["payout"] = g("payout")
    return out


def quarterly(ticker: str, root: str = ".", as_of=None) -> pd.DataFrame:
    """Cached and shared: treat the result as read-only."""
    a_files, q_files = _resolved_paths(root)
    return _quarterly(ticker, root, None if as_of is None else str(as_of), _mtime(a_files + q_files))


@lru_cache(maxsize=8192)
def _quarterly(ticker: str, root: str, as_of, stamp: float) -> pd.DataFrame:
    _, Q = _data(root)
    k = _key(ticker, Q)
    if k is None or as_of is not None:        # quarterly history is too short for point-in-time tests
        return pd.DataFrame()
    w = Q[k].pivot_table(index="period", columns="field", values="value", aggfunc="last").sort_index()
    w = w[w.index.month.isin([3, 6, 9, 12])]
    g = lambda c: w[c] if c in w else pd.Series(np.nan, index=w.index)

    def _yoy(s: pd.Series) -> pd.Series:
        out_s = pd.Series(np.nan, index=s.index, dtype=float)
        idx = pd.DatetimeIndex(s.index)
        for pos, dt in enumerate(idx):
            cur = s.iloc[pos]
            if pos == 0 or pd.isna(cur):
                continue
            target = dt - pd.DateOffset(years=1)
            diffs = np.abs((idx[:pos] - target).days)
            j = int(np.argmin(diffs))
            if diffs[j] <= 45:
                prev = s.iloc[j]
                if pd.notna(prev) and prev > 0:
                    out_s.iloc[pos] = (cur / prev - 1.0) * 100.0
        return out_s

    out = pd.DataFrame(index=w.index)
    out["sales"], out["sales_yoy"] = g("sales"), _yoy(g("sales"))
    out["op"], out["opm"] = g("op"), g("opm")
    out["np"], out["np_yoy"] = g("np"), _yoy(g("np"))
    out["npm"] = _div(g("np"), g("sales")) * 100
    for c in ("promoters", "fii", "dii", "public", "holders", "gnpa", "nnpa"):
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
    # Capital employed also jumps on an IPO or QIP parked in cash, or on mark-to-market gains; that is not an
    # investment phase, and loss-making cash burners were being excused (AURUM, MOBIKWIK). It counts only when the
    # business makes an operating profit and fixed assets plus work-in-progress took at least 40% of the increase.
    op_s = a["op"].dropna() if "op" in a else pd.Series(dtype=float)
    opm_s = a["opm"].dropna() if "opm" in a else pd.Series(dtype=float)
    if len(op_s):
        if float(op_s.iloc[-1]) <= 0:
            return None
    elif len(opm_s) and float(opm_s.iloc[-1]) <= 0:
        return None
    g = a["ce_growth"].dropna().tail(3)
    cw = a["cwip_share"].dropna()
    big = g[g >= INVEST_JUMP]
    if len(big) and "gross_block" in a:
        d_ce, d_fa = a["cap_employed"].diff(), a["gross_block"].diff()
        big = big[[bool(d_ce.get(y) and np.isfinite(d_fa.get(y, np.nan)) and d_fa.get(y) >= 0.4 * d_ce.get(y)) for y in big.index]]
    if len(big):
        y = big.index[-1]
        prev = a["cap_employed"].shift(1).get(y)
        return dict(year=y.strftime("FY%y") if y.month == 3 else y.strftime("%b %y"), jump=float(big.iloc[-1]),
                    text=f"capital employed rose {big.iloc[-1]:.0f}% in {y.strftime('FY%y')} "
                         f"(₹{prev:,.0f} cr to ₹{a['cap_employed'].get(y):,.0f} cr)")
    if len(cw) and cw.iloc[-1] >= INVEST_CWIP:
        return dict(year="now", jump=np.nan, text=f"work-in-progress is {cw.iloc[-1]:.0f}% of capital employed")
    return None


def _lender_pillars(T: pd.DataFrame, q: pd.DataFrame, P: dict, add, hard: list) -> None:
    """Score a lender from its ROA tree: spread and costs, ROA and ROE, leverage and asset quality."""
    if T is None or T.empty:
        return
    kind = T.attrs.get("kind", "NBFC")
    sgn = {UP: 1, FLAT: 0, DOWN: -1}
    L = T.iloc[-1]
    # ---- spread and costs
    lv, dr = [], []
    t = _trend(T["nii"], 0.25)
    add("Spread & costs", "Net interest income / assets", t["way"] if t else FLAT,
        f"{L.nii:.2f}% (interest earned {L.int_income:.2f}% less interest paid {L.int_cost:.2f}%)"
        + (f"; {t['now']:.2f}% avg vs {t['was']:.2f}% before" if t else ""))
    lv.append(_lin(L.nii, NII_PTS[kind]))
    if t:
        dr.append(sgn[t["way"]])
    t = _trend(T["costs"], 0.25, higher_better=False)
    add("Spread & costs", "Costs / assets", t["way"] if t else FLAT,
        f"{L.costs:.2f}% (operating expenses and provisions together)" + (f"; {t['now']:.2f}% avg vs {t['was']:.2f}% before" if t else ""))
    if t:
        dr.append(sgn[t["way"]])
    if np.isfinite(L.cost_income):
        add("Spread & costs", "Costs / income", DOWN if L.cost_income > 75 else FLAT,
            f"costs take {L.cost_income:.0f}% of net interest plus other income")
        lv.append(_lin(L.cost_income, [(35, 100), (50, 80), (65, 55), (80, 25), (95, 0)]))
    add("Spread & costs", "Other income / assets", FLAT, f"{L.other_income:.2f}% (fees, treasury, recoveries)", scored=False)
    P["Spread & costs"] = dict(score=_mean(lv), dir=_mean(dr) if dr else 0.0)
    # ---- ROA and ROE
    t = _trend(T["roa"], 0.15)
    roa_s = _lin(L.roa, ROA_PTS[kind]) + (float(np.clip(t["delta"] * 20, -15, 15)) if t else 0.0)
    add("ROA & ROE", "ROA", t["way"] if t else FLAT, f"{L.roa:.2f}%" + (f"; {t['now']:.2f}% avg vs {t['was']:.2f}% before" if t else "")
        + f" (a good {kind.lower()} earns about {ROA_PTS[kind][2][0]:.1f}% or more)")
    t2 = _trend(T["roe"], 2.0)
    add("ROA & ROE", "ROE", t2["way"] if t2 else FLAT, f"{L.roe:.1f}% = ROA {L.roa:.2f}% × leverage {L.leverage:.1f}x"
        + (f"; {t2['now']:.0f}% avg vs {t2['was']:.0f}% before" if t2 else ""))
    roe_s = _lin(L.roe, [(4, 0), (8, 25), (12, 50), (16, 75), (22, 100)])
    P["ROA & ROE"] = dict(score=float(np.clip(_mean([roa_s, roe_s], [0.6, 0.4]), 0, 100)),
                          dir=_mean([sgn[t["way"]] if t else 0, sgn[t2["way"]] if t2 else 0]))
    if len(T) >= 4:                                     # what moved ROA over three years
        o = T.iloc[-4]
        d = {"net interest": L.nii - o.nii, "other income": L.other_income - o.other_income, "costs": -(L.costs - o.costs),
             "tax": -(L.tax - o.tax)}
        add("ROA & ROE", "What moved ROA in 3 years", UP if L.roa - o.roa > 0.15 else DOWN if L.roa - o.roa < -0.15 else FLAT,
            f"{L.roa - o.roa:+.2f} pp: " + ", ".join(f"{k} {v:+.2f}" for k, v in sorted(d.items(), key=lambda kv: -abs(kv[1]))), scored=False)
    if L.roa < 0:
        hard.append("loss-making lender")
    # ---- leverage and asset quality
    lv, dr = [_lin(L.leverage, LEV_PTS[kind])], []
    t = _trend(T["leverage"], 0.5, higher_better=False)
    add("Leverage & asset quality", "Leverage", t["way"] if t else FLAT,
        f"assets are {L.leverage:.1f}x equity" + (f"; {t['now']:.1f}x avg vs {t['was']:.1f}x before" if t else ""))
    if t:
        dr.append(sgn[t["way"]])
    if L.leverage > LEV_MAX[kind]:
        hard.append(f"leverage {L.leverage:.0f}x")
    if np.isfinite(L.asset_g):
        add("Leverage & asset quality", "Balance-sheet growth", FLAT, f"assets {L.asset_g:+.0f}% in the latest year", scored=False)
    for col, lab, pts in (("nnpa", "Net NPA", [(0.3, 100), (1, 75), (2, 45), (4, 10), (6, 0)]),
                          ("gnpa", "Gross NPA", [(1, 100), (2.5, 75), (4, 50), (7, 15), (10, 0)])):
        s = q[col].dropna() if not q.empty and col in q else pd.Series(dtype=float)
        if len(s):
            d = float(s.iloc[-1] - s.iloc[-5]) if len(s) >= 5 else 0.0
            way = FLAT if abs(d) < 0.2 else UP if d < 0 else DOWN
            add("Leverage & asset quality", lab, way, f"{s.iloc[-1]:.2f}%" + (f" ({d:+.2f} pp over a year)" if len(s) >= 5 else ""))
            lv.append(_lin(float(s.iloc[-1]), pts))
            dr.append(sgn[way])
    P["Leverage & asset quality"] = dict(score=_mean(lv), dir=_mean(dr) if dr else 0.0)


def read(ticker: str, root: str = ".", as_of=None) -> Dict[str, Any]:
    """
    Score the business 0-100 on four pillars and describe what is changing.

      Growth (30%)                3-year sales growth, latest two quarters' sales YoY, last-4-quarter profit YoY
      Margins (25%)               operating margin: last 3 years v the 3 before, and last 4 quarters v the 4 before
      Cash & balance sheet (25%)  3-year cash conversion, debt / equity, interest cover   (skipped for lenders)
      Returns on capital (20%)    ROCE level and its change

    Banks, NBFCs and gold-loan companies are scored on their ROA tree instead (roa_tree, LENDER_WEIGHTS):
    growth 25%, spread and costs 25%, ROA and ROE 30%, leverage and asset quality 20%, with ROA, spread
    and leverage judged against break-points for that type of lender.

    Each pillar is LEVEL plus DIRECTION, so a falling but still-good number is not scored like a bad
    one. One cause counts once: net margin, fixed-asset turnover and ownership are shown but not scored
    (they move with margins, capital and investor mix). During an investment phase (investment_phase)
    the fall in ROCE that a larger capital base causes is not penalised; ROCE on capital already at
    work is shown instead, and the items affected are marked INVESTING.

    verdict   STRONG >= 70, HEALTHY >= 55, MIXED >= 40, WEAK below
    direction IMPROVING / STEADY / SLIPPING from the scored pillars' directions
    The weights and break-points are judgement, not fitted to returns.
    """
    a, q = annual(ticker, root, as_of), quarterly(ticker, root, as_of)
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
    nf = not fin                       # lenders are scored on the ROA tree instead (see _lender_pillars)
    lv, dr = [], []
    if not a.empty and not fin:
        t = _trend(a["opm"], 1.5)
        if t:
            add("Margins", "Operating margin", t["way"], f"{t['now']:.0f}% (last {t['n']} yrs vs the {t['n']} before: {t['was']:.0f}%); latest {t['last']:.0f}%")
            lv.append(_lin(t["delta"], [(-8, 0), (-3, 35), (0, 62), (3, 85), (6, 100)]))
            dr.append({UP: 1, FLAT: 0, DOWN: -1}[t["way"]])
    if len(r) >= 8 and nf:
        opm_ok = r["opm"].notna().sum() >= 8
        m = r["opm"] if opm_ok else r["npm"]
        # the margin of the 4 quarters together (profit / sales, sales-weighted), not the average of 4 percentages:
        # CDSL Dec-25..Sep-26 is 642 / 1,254 = 51.2%, the plain average said 50.2%
        num = r["op"] if opm_ok and "op" in r else r["np"] if "np" in r else None
        if num is not None and "sales" in r and num.iloc[-8:].notna().all() and (r["sales"].iloc[-8:] > 0).all():
            m4 = float(num.iloc[-4:].sum() / r["sales"].iloc[-4:].sum() * 100)
            mp = float(num.iloc[-8:-4].sum() / r["sales"].iloc[-8:-4].sum() * 100)
        else:
            m4, mp = float(m.iloc[-4:].mean()), float(m.iloc[-8:-4].mean())
        if np.isfinite(m4) and np.isfinite(mp):
            way = FLAT if abs(m4 - mp) < 1 else UP if m4 > mp else DOWN
            add("Margins", "Quarterly operating margin" if opm_ok else "Quarterly net margin", way,
                f"{m4:.1f}% over the last 4 quarters vs {mp:.1f}% the 4 before; latest {m.iloc[-1]:.1f}%")
            lv.append(_lin(m4 - mp, [(-5, 0), (-2, 35), (0, 62), (2, 85), (4, 100)]))
            dr.append({UP: 1, FLAT: 0, DOWN: -1}[way])
    if not a.empty and nf:
        t = _trend(a["npm"], 1.0)
        if t:
            dil = inv is not None and t["way"] == DOWN
            add("Margins", "Net margin", BUILD if dil else t["way"],
                f"{t['now']:.0f}% (last {t['n']} yrs vs the {t['n']} before: {t['was']:.0f}%); latest {t['last']:.0f}%"
                + (" — interest and depreciation on the new capital" if dil else ""), scored=False)
    if lv:
        # LEVEL plus direction, like every other pillar: scored on change alone, a loss-maker whose losses narrowed
        # came out STRONG on margins (URBANCO -16% margin -> 82, SCILAL -179% -> 71)
        lvl = None
        if not a.empty and a["opm"].notna().sum() >= 1:
            lvl = float(a["opm"].dropna().iloc[-3:].mean())
        elif len(r) >= 4:
            lvl = float((r["opm"] if r["opm"].notna().sum() >= 4 else r["npm"]).iloc[-4:].mean())
        if lvl is not None and np.isfinite(lvl):
            lvl_pts = _lin(lvl, [(-10, 0), (0, 10), (5, 35), (12, 60), (20, 85), (30, 100)])
            score = 0.5 * lvl_pts + 0.5 * _mean(lv)
            if lvl < 0:
                score = min(score, 25.0)                  # losing money at the operating line: never a good margin pillar
            P["Margins"] = dict(score=score, dir=_mean(dr))
        else:
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
    if fin:
        _lender_pillars(roa_tree(ticker, root, as_of), q, P, add, hard)
    if not a.empty and nf:
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

    # Too old or too thin to score: Screener's consolidated page for some companies stops years ago (PFIZER's ends in
    # 2008 and scored STRONG 100), and a score built on one or two pillars said more than the data could.
    ref = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today()
    if not a.empty and (ref - a.index.max()).days > STALE_DAYS:
        add("Data", "Annual figures", DOWN, f"latest annual figures are for {a.index.max():%b %Y}: too old to score", scored=False)
        P = {}
    elif 0 < len(P) < MIN_PILLARS:
        add("Data", "Coverage", FLAT, f"only {len(P)} of 4 pillars could be measured: not enough to score", scored=False)
        P = {}
    w = {k: (LENDER_WEIGHTS if fin else WEIGHTS)[k] for k in P}
    score = float(sum(P[k]["score"] * w[k] for k in P) / sum(w.values())) if P else np.nan
    drift = float(sum(P[k]["dir"] * w[k] for k in P) / sum(w.values())) if P else 0.0
    verdict = "NO DATA" if not P else next(lab for cut, lab in LABELS if round(score) >= cut)
    if not any(f["way"] == BUILD and f["metric"] != "Investment phase" for f in F):      # nothing was discounted: not worth a tag
        inv = None
        F = [f for f in F if f["metric"] != "Investment phase"]
    direction = "" if not P else "IMPROVING" if drift > 0.25 else "SLIPPING" if drift < -0.25 else "STEADY"
    sc = [f for f in F if f["scored"]]
    return dict(findings=F, ownership=own,
                pillars={k: dict(score=round(v["score"]), dir=round(v["dir"], 2), weight=w[k]) for k, v in P.items()},
                lender_kind=(a["lender_kind"].iloc[-1] if fin and not a.empty else ""),
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


BRIEF_CACHE = "data/business_scores.csv"
_BRIEF_COLS = ["ticker", "verdict", "score", "direction", "investing", "headline", "up", "down", "flags", "roce", "quarter",
               "sales_yoy", "np_yoy", "ok", "fingerprint"]


@lru_cache(maxsize=32)
def _canonical_size(path: str, mtime: float, raw_size: int) -> int:
    """Canonical byte size normalized to LF line endings for CSVs (identical on Windows and Linux)."""
    if path.endswith(".gz"):
        return raw_size
    crlf = 0
    prev_cr = False
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            if prev_cr and chunk[:1] == b"\n":
                crlf += 1
            crlf += chunk.count(b"\r\n")
            prev_cr = chunk.endswith(b"\r")
    return raw_size - crlf


def fingerprint(root: str = ".") -> str:
    """Identifies the fundamentals files by canonical LF size, which is the same across OS checkouts."""
    # SCORE_VERSION: a change to the scoring must not be served from scores the old code saved
    a_files, q_files = _resolved_paths(root)
    parts = []
    for p in a_files + q_files:
        if os.path.exists(p):
            parts.append(str(_canonical_size(p, os.path.getmtime(p), os.path.getsize(p))))
        else:
            parts.append("-")
    return SCORE_VERSION + "|" + "|".join(parts)


@lru_cache(maxsize=4)
def _precomputed(root: str, mtime: float, fp: str) -> Dict[str, Dict[str, Any]]:
    try:
        d = pd.read_csv(os.path.join(root, BRIEF_CACHE))
        d = d[d.fingerprint == fp]
    except Exception:
        return {}
    out = {}
    for r in d.to_dict("records"):
        r.pop("fingerprint", None)
        r["flags"] = [x for x in str(r["flags"]).split("; ") if x and x != "nan"]
        r["score"] = None if pd.isna(r["score"]) else int(r["score"])
        r["quarter"] = None if pd.isna(r["quarter"]) else r["quarter"]
        r["direction"] = "" if pd.isna(r["direction"]) else r["direction"]
        r["investing"], r["ok"] = bool(r["investing"]), bool(r["ok"])
        out[r["ticker"]] = r
    return out


def build_brief_cache(tickers=None, root: str = ".") -> int:
    """Score every stock once and save it (EOD job), so a cold dashboard start does not recompute ~170 scores."""
    if tickers is None:
        from utils.nifty1000_list import EXTENDED, TICKERS_1000
        tickers = list(TICKERS_1000) + list(EXTENDED)
    a_files, q_files = _resolved_paths(root)
    stamp, fp, rows = _mtime(a_files + q_files), fingerprint(root), []
    for t in tickers:
        try:
            b = _brief(t, root, stamp)
        except Exception:
            continue
        rows.append({**b, "ticker": t, "flags": "; ".join(b["flags"]), "fingerprint": fp})
    pd.DataFrame(rows, columns=_BRIEF_COLS).to_csv(os.path.join(root, BRIEF_CACHE), index=False)
    return len(rows)


def brief(ticker: str, root: str = ".") -> Dict[str, Any]:
    """Score, verdict, red flags, latest quarter and a pass / fail for the funnel's business stage.
    Read from the nightly precomputed file when it matches the fundamentals on disk; computed otherwise."""
    p = os.path.join(root, BRIEF_CACHE)
    if os.path.exists(p):
        hit = _precomputed(root, os.path.getmtime(p), fingerprint(root)).get(ticker)
        if hit is not None:
            return dict(hit)
    a_files, q_files = _resolved_paths(root)
    return _brief(ticker, root, _mtime(a_files + q_files))


@lru_cache(maxsize=4)
def _q_opm_map(root: str, stamp: float) -> Dict[str, tuple[float, float]]:
    """Fast map of ticker -> (latest_opm_pct, yoy_opm_delta_pp) across all quarterly filings."""
    _, Q = _data(root)
    if not Q:
        return {}
    L = pd.concat(Q.values(), ignore_index=True)
    L = L[L["field"].isin(["op", "sales", "opm"]) & L["period"].dt.month.isin([3, 6, 9, 12])]
    if L.empty:
        return {}
    W = L.pivot_table(index=["ticker", "period"], columns="field", values="value", aggfunc="last")
    W = W.apply(pd.to_numeric, errors="coerce").sort_index()
    opm = pd.Series(np.nan, index=W.index)
    if {"op", "sales"} <= set(W.columns):
        opm = W["op"] / W["sales"].where(W["sales"] > 0) * 100.0
    if "opm" in W.columns:                      # older layout: the margin itself
        opm = opm.fillna(W["opm"])
    opm = opm[opm.abs() <= 100].dropna()       # beyond ±100% the "margin" is a near-zero-sales artefact
    out: Dict[str, tuple[float, float]] = {}
    for tk, g in opm.groupby(level=0):
        g = g.droplevel(0)
        opm_now, dt_now = float(g.iloc[-1]), g.index[-1]
        opm_chg = np.nan
        if len(g) >= 2:                          # the same quarter a year earlier, within 45 days
            prev_idx = g.index[:-1]
            diffs = np.abs((prev_idx - (dt_now - pd.DateOffset(years=1))).days)
            j = int(np.argmin(diffs))
            if diffs[j] <= 45:
                opm_chg = opm_now - float(g.iloc[j])
        full_tk = tk if str(tk).endswith(".NS") else f"{tk}.NS"
        out[full_tk] = (round(opm_now, 1), round(opm_chg, 1) if np.isfinite(opm_chg) else np.nan)
    return out


def q_opm_map(root: str = ".") -> Dict[str, tuple[float, float]]:
    """Cached map of `.NS` ticker -> `(latest_opm_pct, yoy_opm_delta_pp)`."""
    _, q_files = _resolved_paths(root)
    return _q_opm_map(root, _mtime(q_files))


# ---------------------------------------------------------------- institutional holding change (FII + DII)
INST_ACC_Y, INST_ACC_Q, INST_DIS_Y, INST_DIS_Q = 2.0, 1.0, -2.0, -0.5


def inst_label(d_q: float, d_y: float) -> str:
    """ACCUMULATING / DISTRIBUTING / NEUTRAL from the FII + DII change in percentage points.
    Backtest Nov 2023 - Nov 2026 (12 filing quarters, ~10k stock-quarters, entry a month after the filing deadline):
    ACCUMULATING beat the median stock over 6 months in 9 of 9 quarters (median +2.9 pp, 56% hit); DISTRIBUTING was
    mixed (median ~0), so it is context, not a warning. Shown, never a funnel gate."""
    if pd.notna(d_q) and (d_q >= INST_ACC_Q or (pd.notna(d_y) and d_y >= INST_ACC_Y)):
        return "ACCUMULATING"
    if pd.notna(d_y) and d_y <= INST_DIS_Y and pd.notna(d_q) and d_q <= INST_DIS_Q:
        return "DISTRIBUTING"
    return "NEUTRAL" if pd.notna(d_q) else ""


@lru_cache(maxsize=4)
def _inst_map(root: str, stamp: float) -> Dict[str, Dict[str, Any]]:
    _, Q = _data(root)
    if not Q:
        return {}
    L = pd.concat(Q.values(), ignore_index=True)
    L = L[L["field"].isin(["fii", "dii", "promoters"]) & L["period"].dt.month.isin([3, 6, 9, 12])]
    if L.empty:
        return {}
    W = L.pivot_table(index=["ticker", "period"], columns="field", values="value", aggfunc="last")
    W = W.apply(pd.to_numeric, errors="coerce").reset_index()
    if "fii" not in W or "dii" not in W:
        return {}
    if "promoters" not in W:
        W["promoters"] = np.nan
    W = W.dropna(subset=["fii", "dii"]).sort_values(["period", "ticker"])
    W["inst"] = W["fii"] + W["dii"]
    last = W.sort_values("period").groupby("ticker").tail(1)
    cols = ["ticker", "period", "fii", "dii", "inst"]
    out_df = last.copy()
    for tag, months in (("q", 3), ("y", 12)):           # the filing N months earlier, matched within 45 days
        probe = last[["ticker", "period"]].assign(target=last["period"] - pd.DateOffset(months=months)).sort_values("target")
        prev = W[cols].rename(columns={"period": "p_prev"}).sort_values("p_prev")
        m = pd.merge_asof(probe, prev, left_on="target", right_on="p_prev", by="ticker", direction="nearest",
                          tolerance=pd.Timedelta(days=45))
        m = m[m["p_prev"] < m["period"]].set_index("ticker")
        for c in ("fii", "dii", "inst"):
            out_df[f"d_{c}_{tag}"] = out_df["ticker"].map(out_df.set_index("ticker")[c] - m[c])
    out: Dict[str, Dict[str, Any]] = {}
    for r in out_df.itertuples(index=False):
        tk = str(r.ticker)
        full_tk = tk if tk.endswith(".NS") else f"{tk}.NS"
        f = lambda v: float(v) if pd.notna(v) else np.nan
        out[full_tk] = dict(period=pd.Timestamp(r.period).strftime("%b %Y"), fii=f(r.fii), dii=f(r.dii), promoters=f(r.promoters),
                            d_fii_q=f(r.d_fii_q), d_dii_q=f(r.d_dii_q), d_inst_q=f(r.d_inst_q),
                            d_fii_y=f(r.d_fii_y), d_dii_y=f(r.d_dii_y), d_inst_y=f(r.d_inst_y),
                            label=inst_label(f(r.d_inst_q), f(r.d_inst_y)))
    return out


def inst_map(root: str = ".") -> Dict[str, Dict[str, Any]]:
    """Cached `.NS` ticker -> latest FII / DII % and their quarter-on-quarter and year-on-year change (pp)."""
    _, q_files = _resolved_paths(root)
    return _inst_map(root, _mtime(q_files))
