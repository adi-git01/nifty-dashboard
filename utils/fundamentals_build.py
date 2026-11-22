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

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

ANNUAL, QUARTERLY = "data/fundamentals_annual.csv", "data/fundamentals_quarterly.csv"
CARRY = ["grossMargins", "forwardPE"]                 # no NSE / broker source: Yahoo fills them (yahoo_fill)
# Yahoo .info field for each cache column, used only where NSE and Screener left a gap
YAHOO_FIELDS = {"grossMargins": "grossMargins", "forwardPE": "forwardPE", "pe": "trailingPE", "pb": "priceToBook",
                "roe": "returnOnEquity", "roa": "returnOnAssets", "profitMargins": "profitMargins",
                "operatingMargins": "operatingMargins", "ebitdaMargins": "ebitdaMargins", "revenueGrowth": "revenueGrowth",
                "earningsGrowth": "earningsGrowth", "earningsQuarterlyGrowth": "earningsQuarterlyGrowth",
                "debtToEquity": "debtToEquity", "marketCap": "marketCap", "beta": "beta"}


def yahoo_fill(df: pd.DataFrame, fields=None, sleep: float = 0.5, log=print, max_calls: int = 2000,
               budget_s: float = 1500, workers: int = 1) -> pd.DataFrame:
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
    skip_map: Dict[str, Set[str]] = out.attrs.get("skip_yahoo") or {}

    def _needs_mask(cols: List[str]) -> pd.Series:
        m = pd.Series(False, index=out.index)
        for c in cols:
            if c in out.columns:
                bad = out[c].isna()
                if skip_map:
                    bad = bad & ~out["ticker"].map(lambda tk, col=c: col in (skip_map.get(tk) or ()))
                m = m | bad
        return m

    need = out[_needs_mask(fields)]
    if need.empty:
        return out
    # stocks missing a core field first; then the ones missing only grossMargins / forwardPE, starting at a different
    # point each run, so a budget cut-off does not leave the same tail of the list stale every time
    core = [c for c in fields if c not in CARRY]
    gap = _needs_mask(core).loc[need.index] if core else pd.Series(False, index=need.index)
    rest = need[~gap]
    if len(rest):
        k = pd.Timestamp.today().dayofyear * 97 % len(rest)
        rest = pd.concat([rest.iloc[k:], rest.iloc[:k]])
    need = pd.concat([need[gap], rest])
    calls = filled = 0
    t0 = time.time()

    def _fetch_one(tk: str) -> dict:
        try:
            return yf.Ticker(tk).info or {}
        except Exception:
            return {}

    def _apply_info(i, tk: str, info: dict) -> bool:
        got = False
        sk = skip_map.get(tk) or ()
        for c in fields:
            if pd.isna(out.at[i, c]) and c not in sk:
                v = pd.to_numeric(info.get(YAHOO_FIELDS[c]), errors="coerce")
                if v == v and v not in (float("inf"), float("-inf")):
                    out.at[i, c] = float(v)
                    got = True
        if got:
            out.at[i, "fund_source"] = f"{out.at[i, 'fund_source'] or 'screener'}+yahoo"
        return got

    if workers and workers > 1:
        rows_list = [(i, str(r["ticker"])) for i, r in need.iterrows()][:max_calls]
        batch_sz = max(workers * 4, 16)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start_idx in range(0, len(rows_list), batch_sz):
                if calls >= max_calls or time.time() - t0 > budget_s:
                    log(f"[fund] Yahoo fallback stopped after {calls} calls / {time.time() - t0:.0f}s")
                    break
                chunk = rows_list[start_idx:start_idx + batch_sz]
                futs = {pool.submit(_fetch_one, tk): (i, tk) for i, tk in chunk}
                for fut in as_completed(futs):
                    i, tk = futs[fut]
                    info = fut.result()
                    calls += 1
                    if _apply_info(i, tk, info):
                        filled += 1
                if sleep:
                    time.sleep(sleep)
                if calls % 100 < len(chunk):
                    log(f"[fund] Yahoo fallback: {calls}/{len(need)} asked, {filled} filled")
    else:
        for i, r in need.iterrows():
            if calls >= max_calls or time.time() - t0 > budget_s:     # the rest carry the previous cache's values
                log(f"[fund] Yahoo fallback stopped after {calls} calls / {time.time() - t0:.0f}s")
                break
            tk = str(r["ticker"])
            info = _fetch_one(tk)
            calls += 1
            if _apply_info(i, tk, info):
                filled += 1
            if sleep:
                time.sleep(sleep)
            if calls % 100 == 0:
                log(f"[fund] Yahoo fallback: {calls}/{len(need)} asked, {filled} filled")
    log(f"[fund] Yahoo fallback: {calls} stocks asked for {', '.join(fields[:4])}…; {filled} got at least one value")
    out.attrs["skip_yahoo"] = skip_map
    return out
COLS = ["ticker", "name", "sector", "sector_granular", "pe", "forwardPE", "pegRatio", "pb", "roe", "roa", "profitMargins",
        "grossMargins", "operatingMargins", "ebitdaMargins", "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth",
        "debtToEquity", "marketCap", "beta", "earningsQuality", "earningsTrend", "fund_last_updated", "fund_source"]


def _select_fy_rows(p: pd.DataFrame) -> pd.DataFrame:
    """Keep one fiscal-year row per period while preserving pre-switch history for FY-end switchers."""
    if len(p) <= 1:
        return p
    p = p.sort_index()
    kept_pos: List[int] = []
    last_kept: Optional[pd.Timestamp] = None
    for pos in range(len(p) - 1, -1, -1):
        dt = pd.Timestamp(p.index[pos])
        if last_kept is None or (last_kept - dt).days >= 220:
            kept_pos.append(pos)
            last_kept = dt
    return p.iloc[sorted(kept_pos)]


def _pivot(path: str, quarterly: bool) -> Dict[str, pd.DataFrame]:
    d = pd.read_csv(path, parse_dates=["period"])
    if d.empty:
        return {}
    d["period"] = pd.to_datetime(d["period"], errors="coerce")
    d = d.dropna(subset=["period"])
    d = d[~d.field.isin(["mcap_cr", "price_scr"])] if not quarterly else d
    if quarterly:
        d = d[d.period.dt.month.isin([3, 6, 9, 12])]
    out = {}
    for t, g in d.groupby("ticker"):
        p = g.pivot_table(index="period", columns="field", values="value", aggfunc="last").sort_index()
        if not quarterly and len(p):
            p = _select_fy_rows(p)                                         # keep FY-end switchers' history
        out[t] = p
    return out


def _last(s: Optional[pd.Series], k: int = -1):
    if s is None:
        return np.nan
    s = s.dropna()
    return float(s.iloc[k]) if len(s) >= abs(k) else np.nan


def _yoy_prior(s: Optional[pd.Series], ref_dt: pd.Timestamp) -> float:
    """Value from ~1 year before ref_dt (+-45d), preventing skipped quarters from pairing wrong years."""
    if s is None or len(s) == 0 or pd.isna(ref_dt):
        return np.nan
    s = s.dropna()
    if s.empty:
        return np.nan
    target = pd.Timestamp(ref_dt) - pd.DateOffset(years=1)
    diffs = np.abs((pd.DatetimeIndex(s.index) - target).days)
    best = int(np.argmin(diffs))
    return float(s.iloc[best]) if diffs[best] <= 45 else np.nan


def _finite(v) -> bool:
    if v is None:
        return False
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def _div(a, b):
    return float(a) / float(b) if _finite(a) and _finite(b) and float(b) != 0.0 else np.nan


def screener_ratios(t: str, A: Dict[str, pd.DataFrame], Q: Dict[str, pd.DataFrame], mcap_cr: float, lender_hint: bool = False) -> dict:
    a, q = A.get(t), Q.get(t)
    r: dict = {}
    skip: Set[str] = set()
    lender = bool((a is not None and len(a) and "deposits" in a and a["deposits"].notna().any()) or lender_hint)
    if lender:
        skip.add("ebitdaMargins")
    if q is not None and len(q):
        q = q.sort_index()
        q = q[~q.index.duplicated(keep="last")]
        sales, np_ = q.get("sales"), q.get("np")
        if len(q) >= 4:
            q4_idx = pd.DatetimeIndex(q.index[-4:])
            gaps_ok = (
                (q4_idx[-1] - q4_idx[0]).days <= 395
                and all((q4_idx[i] - q4_idx[i - 1]).days <= 140 for i in range(1, 4))
            )
            if gaps_ok:
                ttm_s = float(sales.iloc[-4:].sum()) if sales is not None and sales.iloc[-4:].notna().all() else np.nan
                ttm_n = float(np_.iloc[-4:].sum()) if np_ is not None and np_.iloc[-4:].notna().all() else np.nan
                ttm_op = float(q["op"].iloc[-4:].sum()) if "op" in q and q["op"].iloc[-4:].notna().all() else np.nan
                r["profitMargins"] = _div(ttm_n, ttm_s)
                if not lender:
                    r["ebitdaMargins"] = _div(ttm_op, ttm_s)
                r["_ttm_np"] = ttm_n
        if sales is not None and len(sales.dropna()):
            s_c = sales.dropna()
            prev_s = _yoy_prior(s_c, pd.Timestamp(s_c.index[-1]))
            r["revenueGrowth"] = _div(float(s_c.iloc[-1]) - prev_s, abs(prev_s))
        if np_ is not None and len(np_.dropna()):
            n_c = np_.dropna()
            prev_n = _yoy_prior(n_c, pd.Timestamp(n_c.index[-1]))
            g = _div(float(n_c.iloc[-1]) - prev_n, abs(prev_n))
            r["earningsGrowth"] = r["earningsQuarterlyGrowth"] = g
    if a is not None and len(a):
        a = a.sort_index()
        book = (a.get("eq_cap", pd.Series(0, index=a.index)).fillna(0) + a.get("reserves", pd.Series(np.nan, index=a.index)))
        npa = a.get("np")
        npa_last = _last(npa)
        b1, b0 = _last(book), _last(book, -2)
        avg_book = np.nanmean([b1, b0]) if _finite(b0) else b1
        if _finite(avg_book) and float(avg_book) > 0:
            r["roe"] = _div(npa_last, avg_book)
        else:
            r["roe"] = np.nan
            skip.add("roe")
        tl_last = _last(a.get("total_liab"))
        r["roa"] = _div(npa_last, tl_last) if _finite(tl_last) and float(tl_last) > 0 else np.nan
        if _finite(b1) and float(b1) > 0:
            r["debtToEquity"] = _div(_last(a.get("borrowings")), b1) * 100
            r["pb"] = _div(mcap_cr, b1)
        else:
            r["debtToEquity"] = np.nan
            r["pb"] = np.nan
            skip.update({"pb", "debtToEquity"})
        if _finite(npa_last) and float(npa_last) > 0:
            r["earningsQuality"] = _div(_last(a.get("cfo")), npa_last)
        else:
            r["earningsQuality"] = np.nan
            skip.add("earningsQuality")
        r["earningsTrend"] = _div(npa_last - _last(npa, -2), abs(_last(npa, -2)))
        dep_ratio = _div(_last(a.get("dep")), _last(a.get("sales")))
        eb = r.get("ebitdaMargins")
        if lender:
            # banks / NBFCs: Screener's "operating profit" is financing profit after interest; the pre-tax margin is the comparable figure
            r["operatingMargins"] = _div(_last(a.get("pbt")), _last(a.get("sales")))
        elif _finite(eb) and _finite(dep_ratio):
            r["operatingMargins"] = float(eb) - float(dep_ratio)          # EBIT margin (Yahoo's operatingMargins)
    if skip:
        r["_skip_yahoo"] = skip
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
    from utils import fund_paths
    ann_path = fund_paths.resolve(ANNUAL)
    qtr_path = fund_paths.resolve(QUARTERLY)
    A, Q = _pivot(ann_path, False), _pivot(qtr_path, True)
    sc = pd.read_csv(ann_path)
    sc = sc[sc.field.isin(["mcap_cr", "price_scr"])].pivot_table(index="ticker", columns="field", values="value", aggfunc="last")
    # Resolve renamed symbols so legacy Screener rows follow symbol_changes.csv (e.g. HEG -> HEGAM).
    try:
        chg = npx._symbol_map()
        fwd = {}
        for new_s, olds in chg.items():
            for old_s, dt in olds:
                if old_s not in fwd or dt > fwd[old_s][1]:
                    fwd[old_s] = (new_s, dt)

        def _alias_keys(t_ns: str) -> List[str]:
            base = t_ns[:-3]
            cands = []
            cur, seen = base, {base}
            while cur in fwd and fwd[cur][0] not in seen:
                cur = fwd[cur][0]
                seen.add(cur)
                cands.append(f"{cur}.NS")
            for anc, _ in npx._ancestors(base, chg):
                if f"{anc}.NS" not in cands:
                    cands.append(f"{anc}.NS")
            return cands
    except Exception:
        _alias_keys = lambda t_ns: []

    def _pick_key(mapping, t_ns: str) -> str:
        if t_ns in mapping:
            return t_ns
        for alt in _alias_keys(t_ns):
            if alt in mapping:
                return alt
        return t_ns

    H = npx.history(list(tickers) + ["^NSEI"], start=pd.Timestamp.today() - pd.Timedelta(days=420))
    bench = H.get("^NSEI", pd.DataFrame()).get("Close", pd.Series(dtype=float))
    S = None
    skip_probe = (os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get("FORCE_NSE_QUOTES", "").strip() != "1")
    if quotes and skip_probe:
        log("[fund] skipping NSE quote-equity probe on GitHub Actions (HTTP 403); using Screener for P/E and shares")
    elif quotes:
        try:
            from utils.nse_data import Session
            S = Session()
            S.warm_quote()
            nse_quote(S, "RELIANCE")                  # one test call: a refused endpoint costs one call, not fifteen
        except Exception as e:
            log(f"[fund] NSE quote unavailable ({e}); P/E and shares from Screener")
            S = None
    rows, nq, fails, t0 = [], 0, 0, time.time()
    skip_map: Dict[str, Set[str]] = {}
    sc_index = set(sc.index)
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
        close = float(H[t]["Close"].iloc[-1]) if t in H and len(H[t]) else np.nan
        t_sc = _pick_key(sc_index, t)
        if not _finite(close) and t_sc in sc.index and _finite(sc.at[t_sc, "price_scr"]):
            close = float(sc.at[t_sc, "price_scr"])
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
        if (not _finite(shares) or float(shares) <= 0) and t_sc in sc.index and _finite(sc.at[t_sc, "price_scr"]) and float(sc.at[t_sc, "price_scr"]) > 0 and _finite(sc.at[t_sc, "mcap_cr"]):
            shares = float(sc.at[t_sc, "mcap_cr"]) / float(sc.at[t_sc, "price_scr"]) * 1e7                  # Screener's share count
        mcap = float(shares) * float(close) if _finite(shares) and float(shares) > 0 and _finite(close) else np.nan
        granular = SUB_INDUSTRY_MAP.get(t) or q.get("nse_industry") or "Unknown"
        t_aq = _pick_key(A, t) if t not in A and t not in Q else t
        if t_aq == t and t not in A and t not in Q:
            t_aq = _pick_key(Q, t)
        r = screener_ratios(t_aq, A, Q, float(mcap) / 1e7 if _finite(mcap) else np.nan,
                            lender_hint=bool(re.search(r"bank|financ|nbfc|lending|housing fin", str(granular), re.I)))
        sk = set(r.pop("_skip_yahoo", ()) or ())
        ttm_np = r.get("_ttm_np", np.nan)
        pe = q.get("pe", np.nan)
        if not (_finite(pe) and float(pe) > 0):
            if _finite(ttm_np) and float(ttm_np) > 0 and _finite(mcap):
                pe = _div(float(mcap) / 1e7, ttm_np)
            else:
                pe = np.nan
                if _finite(ttm_np) and float(ttm_np) <= 0:
                    sk.update({"pe", "pegRatio"})
        else:
            pe = float(pe)
        g = r.get("earningsGrowth")
        if _finite(pe) and float(pe) > 0 and _finite(g) and float(g) > 0:
            peg = _div(pe, float(g) * 100)
        else:
            peg = np.nan
            if (_finite(g) and float(g) <= 0) or (_finite(ttm_np) and float(ttm_np) <= 0):
                sk.add("pegRatio")
        if sk:
            skip_map[t] = sk
        rows.append(dict(ticker=t, name=q.get("name") or names.get(t) or t[:-3], sector=consolidate_sector(granular),
                         sector_granular=granular, pe=pe, pegRatio=peg,
                         marketCap=mcap, beta=beta_1y(H[t]["Close"], bench) if t in H and len(bench) else np.nan,
                         fund_source="nse+screener" if q else "screener",
                         **{k: v for k, v in r.items() if not k.startswith("_")}))
        if (i + 1) % 100 == 0:
            log(f"[fund] {i + 1}/{len(tickers)} built ({nq} NSE quotes)")
    out = pd.DataFrame(rows)
    out["fund_last_updated"] = pd.Timestamp.today().strftime("%Y-%m-%d")
    out = out.reindex(columns=COLS)
    out.attrs["skip_yahoo"] = skip_map
    return out


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
