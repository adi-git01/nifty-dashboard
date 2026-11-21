"""
The bi-weekly fundamentals cache (data/fundamentals_cache.csv) without Yahoo.

Same columns and units as the Yahoo version (fetch_fundamentals.py), built from:
    NSE quote API (/api/quote-equity)     P/E (NSE's symbol P/E), shares issued -> market cap, company name
    Screener files we scrape weekly        P/B, ROE, ROA, margins, growth, debt/equity, earnings quality / trend
    NSE price store (utils/nse_prices)     latest close (market cap, P/E fallback), beta vs the Nifty 50 (1 year daily)
Not available from either (no cost-of-goods line, no analyst estimates): grossMargins, forwardPE. The last
Yahoo values are carried forward so the scores do not jump; they go stale until a source is found.

Units match Yahoo's: ratios as fractions (ROE 0.15), debtToEquity in percent (45.0), marketCap in rupees.
"""
from __future__ import annotations

import re
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ANNUAL, QUARTERLY = "data/fundamentals_annual.csv.gz", "data/fundamentals_quarterly.csv.gz"
CARRY = ["grossMargins", "forwardPE"]                 # no NSE / broker source: Yahoo fills them (yahoo_fill)
# Yahoo .info field for each cache column, used only where NSE and Screener left a gap
YAHOO_FIELDS = {"grossMargins": "grossMargins", "forwardPE": "forwardPE", "pe": "trailingPE", "pb": "priceToBook",
                "roe": "returnOnEquity", "roa": "returnOnAssets", "profitMargins": "profitMargins",
                "operatingMargins": "operatingMargins", "ebitdaMargins": "ebitdaMargins", "revenueGrowth": "revenueGrowth",
                "earningsGrowth": "earningsGrowth", "earningsQuarterlyGrowth": "earningsQuarterlyGrowth",
                "debtToEquity": "debtToEquity", "marketCap": "marketCap", "beta": "beta"}


def yahoo_fill(df: pd.DataFrame, fields=None, sleep: float = 0.5, log=print, max_calls: int = 2000,
               budget_s: float = 1500) -> pd.DataFrame:
    """
    The last resort: Yahoo's .info, only for cells NSE and Screener could not fill (always grossMargins and forwardPE,
    for which no free NSE / broker source exists). A stock whose gaps Yahoo also cannot fill keeps NaN; the caller
    then carries the previous cache's value. Marks fund_source with '+yahoo' where Yahoo supplied something.
    """
    fields = fields or list(YAHOO_FIELDS)
    try:
        import yfinance as yf
    except Exception:
        return df
    out = df.copy()
    need = out[out[fields].isna().any(axis=1)]
    # stocks missing a core field first; then the ones missing only grossMargins / forwardPE, starting at a different
    # point each run, so a budget cut-off does not leave the same tail of the list stale every time
    core = [c for c in fields if c not in CARRY]
    gap = need[core].isna().any(axis=1) if core else pd.Series(False, index=need.index)
    rest = need[~gap]
    if len(rest):
        k = pd.Timestamp.today().dayofyear * 97 % len(rest)
        rest = pd.concat([rest.iloc[k:], rest.iloc[:k]])
    need = pd.concat([need[gap], rest])
    calls = filled = 0
    t0 = time.time()
    for i, r in need.iterrows():
        if calls >= max_calls or time.time() - t0 > budget_s:     # the rest carry the previous cache's values
            log(f"[fund] Yahoo fallback stopped after {calls} calls / {time.time() - t0:.0f}s")
            break
        try:
            info = yf.Ticker(r["ticker"]).info or {}
        except Exception:
            info = {}
        calls += 1
        got = False
        for c in fields:
            if pd.isna(out.at[i, c]):
                v = pd.to_numeric(info.get(YAHOO_FIELDS[c]), errors="coerce")
                if v == v and v not in (float("inf"), float("-inf")):
                    out.at[i, c] = float(v)
                    got = True
        if got:
            filled += 1
            out.at[i, "fund_source"] = f"{out.at[i, 'fund_source'] or 'screener'}+yahoo"
        if sleep:
            time.sleep(sleep)
        if calls % 100 == 0:
            log(f"[fund] Yahoo fallback: {calls}/{len(need)} asked, {filled} filled")
    log(f"[fund] Yahoo fallback: {calls} stocks asked for {', '.join(fields[:4])}…; {filled} got at least one value")
    return out
COLS = ["ticker", "name", "sector", "sector_granular", "pe", "forwardPE", "pegRatio", "pb", "roe", "roa", "profitMargins",
        "grossMargins", "operatingMargins", "ebitdaMargins", "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth",
        "debtToEquity", "marketCap", "beta", "earningsQuality", "earningsTrend", "fund_last_updated", "fund_source"]


def _pivot(path: str, quarterly: bool) -> Dict[str, pd.DataFrame]:
    d = pd.read_csv(path, parse_dates=["period"])
    d = d[~d.field.isin(["mcap_cr", "price_scr"])] if not quarterly else d
    if quarterly:
        d = d[d.period.dt.month.isin([3, 6, 9, 12])]
    out = {}
    for t, g in d.groupby("ticker"):
        p = g.pivot_table(index="period", columns="field", values="value", aggfunc="last").sort_index()
        if not quarterly and len(p):
            p = p[p.index.month == p.index.month.value_counts().idxmax()]      # fiscal year ends only
        out[t] = p
    return out


def _last(s: Optional[pd.Series], k: int = -1):
    if s is None:
        return np.nan
    s = s.dropna()
    return float(s.iloc[k]) if len(s) >= abs(k) else np.nan


def _div(a, b):
    return a / b if a == a and b == b and b not in (0, 0.0) else np.nan


def screener_ratios(t: str, A: Dict[str, pd.DataFrame], Q: Dict[str, pd.DataFrame], mcap_cr: float, lender_hint: bool = False) -> dict:
    a, q = A.get(t), Q.get(t)
    r: dict = {}
    if q is not None and len(q) >= 4:
        sales, np_ = q.get("sales"), q.get("np")
        ttm_s = float(sales.iloc[-4:].sum()) if sales is not None and sales.iloc[-4:].notna().all() else np.nan
        ttm_n = float(np_.iloc[-4:].sum()) if np_ is not None and np_.iloc[-4:].notna().all() else np.nan
        ttm_op = float(q["op"].iloc[-4:].sum()) if "op" in q and q["op"].iloc[-4:].notna().all() else np.nan
        r["profitMargins"] = _div(ttm_n, ttm_s)
        r["ebitdaMargins"] = _div(ttm_op, ttm_s)
        r["_ttm_np"] = ttm_n
        if len(q) >= 5:                                   # latest quarter vs the same quarter a year earlier (Yahoo's definition)
            r["revenueGrowth"] = _div(_last(sales) - _last(sales, -5), abs(_last(sales, -5)))
            g = _div(_last(np_) - _last(np_, -5), abs(_last(np_, -5)))
            r["earningsGrowth"] = r["earningsQuarterlyGrowth"] = g
    if a is not None and len(a):
        book = (a.get("eq_cap", pd.Series(0, index=a.index)).fillna(0) + a.get("reserves", pd.Series(np.nan, index=a.index)))
        npa = a.get("np")
        b1, b0 = _last(book), _last(book, -2)
        avg_book = np.nanmean([b1, b0]) if b0 == b0 else b1
        r["roe"] = _div(_last(npa), avg_book)
        r["roa"] = _div(_last(npa), _last(a.get("total_liab")))           # Screener's "total liabilities" = the balance sheet total
        r["debtToEquity"] = _div(_last(a.get("borrowings")), b1) * 100 if b1 == b1 and b1 > 0 else np.nan
        r["pb"] = _div(mcap_cr, b1) if b1 == b1 and b1 > 0 else np.nan
        r["earningsQuality"] = _div(_last(a.get("cfo")), _last(npa))
        r["earningsTrend"] = _div(_last(npa) - _last(npa, -2), abs(_last(npa, -2)))
        dep_ratio = _div(_last(a.get("dep")), _last(a.get("sales")))
        lender = "deposits" in a and a["deposits"].notna().any() or lender_hint
        if lender:
            # banks / NBFCs: Screener's "operating profit" is financing profit after interest; the pre-tax margin is the comparable figure
            r["operatingMargins"] = _div(_last(a.get("pbt")), _last(a.get("sales")))
        elif r.get("ebitdaMargins") == r.get("ebitdaMargins") and dep_ratio == dep_ratio:
            r["operatingMargins"] = r["ebitdaMargins"] - dep_ratio          # EBIT margin (Yahoo's operatingMargins)
    return r


def nse_quote(S, symbol: str) -> dict:
    j = S.json(f"/api/quote-equity?symbol={symbol.replace('&', '%26')}", tries=2, timeout=12)
    md, si, inf = j.get("metadata") or {}, j.get("securityInfo") or {}, j.get("info") or {}
    ind = j.get("industryInfo") or {}
    pe = pd.to_numeric(md.get("pdSymbolPe"), errors="coerce")
    return dict(name=inf.get("companyName"), pe=float(pe) if pe == pe else np.nan,
                shares=pd.to_numeric(si.get("issuedSize"), errors="coerce"),
                nse_sector=ind.get("sector"), nse_industry=ind.get("basicIndustry") or ind.get("industry"),
                sector_pe=pd.to_numeric(md.get("pdSectorPe"), errors="coerce"))


def beta_1y(px: pd.Series, bench: pd.Series) -> float:
    j = pd.concat([px.pct_change(), bench.pct_change()], axis=1).dropna().tail(250)
    if len(j) < 120 or j.iloc[:, 1].var() == 0:
        return np.nan
    return float(j.cov().iloc[0, 1] / j.iloc[:, 1].var())


def build(tickers: List[str], quotes: bool = True, log=print, sleep: float = 0.3, quote_budget_s: float = 1800) -> pd.DataFrame:
    from utils import nse_prices as npx
    try:
        from utils.nifty1000_list import SUB_INDUSTRY_MAP
    except Exception:
        SUB_INDUSTRY_MAP = {}
    try:
        from utils.sector_mapping import consolidate_sector
    except Exception:
        consolidate_sector = lambda x: x
    from utils.extended import eq_symbols
    names = eq_symbols()
    A, Q = _pivot(ANNUAL, False), _pivot(QUARTERLY, True)
    sc = pd.read_csv(ANNUAL)
    sc = sc[sc.field.isin(["mcap_cr", "price_scr"])].pivot_table(index="ticker", columns="field", values="value", aggfunc="last")
    H = npx.history(list(tickers) + ["^NSEI"], start=pd.Timestamp.today() - pd.Timedelta(days=420))
    bench = H.get("^NSEI", pd.DataFrame()).get("Close", pd.Series(dtype=float))
    S = None
    if quotes:
        try:
            from utils.nse_data import Session
            S = Session()
            S.warm_quote()
            nse_quote(S, "RELIANCE")                  # one test call: a refused endpoint costs one call, not fifteen
        except Exception as e:
            log(f"[fund] NSE quote unavailable ({e}); P/E and shares from Screener")
            S = None
    rows, nq, fails, t0 = [], 0, 0, time.time()
    for i, t in enumerate(tickers):
        # NSE stopped answering (15 failures in a row) or is too slow overall (a flaky NSE that fails every other call
        # never trips the first test and would run the job into its 90-minute limit, losing everything): Screener's
        # P/E and share count for the rest
        if S is not None and (fails >= 15 or time.time() - t0 > quote_budget_s):
            log(f"[fund] NSE quote stopped at {i}/{len(tickers)} ({fails} failures in a row, {time.time() - t0:.0f}s); "
                "Screener for the rest")
            S = None
        if i and i % 100 == 0:
            log(f"[fund] {i}/{len(tickers)} · NSE quotes {nq} · {time.time() - t0:.0f}s")
        close = float(H[t]["Close"].iloc[-1]) if t in H and len(H[t]) else (
            float(sc.at[t, "price_scr"]) if t in sc.index and sc.at[t, "price_scr"] == sc.at[t, "price_scr"] else np.nan)
        q = {}
        if S is not None:
            try:
                q = nse_quote(S, t[:-3])
                nq, fails = nq + 1, 0
            except Exception as e:
                q, fails = {}, fails + 1
                if fails in (1, 15):                  # say why, once at the start and once at the cut-off
                    log(f"[fund] NSE quote {t[:-3]} failed: {e}")
            if sleep:
                time.sleep(sleep)
        shares = q.get("shares")
        if not (shares and shares == shares) and t in sc.index and sc.at[t, "price_scr"] > 0:
            shares = sc.at[t, "mcap_cr"] / sc.at[t, "price_scr"] * 1e7                  # Screener's share count
        mcap = shares * close if shares and close == close else np.nan
        granular = SUB_INDUSTRY_MAP.get(t) or q.get("nse_industry") or "Unknown"
        r = screener_ratios(t, A, Q, mcap / 1e7 if mcap == mcap else np.nan,
                            lender_hint=bool(re.search(r"bank|financ|nbfc|lending|housing fin", str(granular), re.I)))
        pe = q.get("pe", np.nan)
        if not (pe == pe and pe != 0):
            pe = _div(mcap / 1e7, r.get("_ttm_np", np.nan)) if mcap == mcap else np.nan
        g = r.get("earningsGrowth")
        rows.append(dict(ticker=t, name=q.get("name") or names.get(t) or t[:-3], sector=consolidate_sector(granular),
                         sector_granular=granular, pe=pe, pegRatio=_div(pe, g * 100) if g == g and g and g > 0 and pe == pe else np.nan,
                         marketCap=mcap, beta=beta_1y(H[t]["Close"], bench) if t in H and len(bench) else np.nan,
                         fund_source="nse+screener" if q else "screener",
                         **{k: v for k, v in r.items() if not k.startswith("_")}))
        if (i + 1) % 100 == 0:
            log(f"[fund] {i + 1}/{len(tickers)} built ({nq} NSE quotes)")
    out = pd.DataFrame(rows)
    out["fund_last_updated"] = pd.Timestamp.today().strftime("%Y-%m-%d")
    return out.reindex(columns=COLS)


def compare(new: pd.DataFrame, old: pd.DataFrame) -> List[str]:
    """Markdown lines: how the Yahoo-free values line up with Yahoo's, column by column."""
    J = new.set_index("ticker").join(old.set_index("ticker"), lsuffix="_n", rsuffix="_y", how="inner")
    L = [f"Compared on {len(J):,} stocks.", "", "| Field | Both have | Median NSE/Screener ÷ Yahoo | Within 25% | Rank correlation |",
         "|---|---|---|---|---|"]
    for c in COLS:
        if c in ("ticker", "name", "sector", "sector_granular", "fund_last_updated", "fund_source") or f"{c}_n" not in J or f"{c}_y" not in J:
            continue
        x, y = pd.to_numeric(J[f"{c}_n"], errors="coerce"), pd.to_numeric(J[f"{c}_y"], errors="coerce")
        ok = x.notna() & y.notna() & (y != 0)
        if ok.sum() < 20:
            L.append(f"| {c} | {int(ok.sum())} | | | |")
            continue
        ratio = (x[ok] / y[ok])
        L.append(f"| {c} | {int(ok.sum()):,} | {ratio.median():.2f} | {((ratio - 1).abs() <= 0.25).mean():.0%} | "
                 f"{x[ok].rank().corr(y[ok].rank()):.2f} |")
    return L
