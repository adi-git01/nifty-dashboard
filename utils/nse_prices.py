"""
Daily prices from NSE's own files: the store, its upkeep, and a Yahoo-shaped reader.

Store (committed, NSE data is for personal use):
    data/nse/prices/YYYY_MM.parquet        closed months: every EQ/BE/BZ/SME stock, one row per stock per session
    data/nse/prices/d/YYYYMMDD.parquet     sessions of the current month (folded into the month file when it closes)
    data/nse/indices/...                   the same layout for NSE indices (Nifty 50, Nifty 500, sectors) with P/E, P/B
Rows hold the prices as traded (unadjusted) plus NSE's PREV_CLOSE, delivery and trades.

Adjustment: on an ex-date NSE sets PREV_CLOSE to the previous close on the new basis (a 1:5 split's 1000
becomes 200). So factor = PREV_CLOSE(t) / CLOSE(t-1) is exactly the exchange's own adjustment for splits,
bonuses, rights and demergers, with no list to keep up to date; earlier prices are multiplied by it and
volumes divided. Ordinary dividends do not move PREV_CLOSE, so -- unlike Yahoo's auto_adjust -- they are
not taken out of the history (the series matches Yahoo's split-adjusted "Close").

Symbol changes (NSE's symbolchange.csv) are applied so a renamed stock keeps its history.

Reader: history(tickers, start, end) -> {ticker: DataFrame[Open, High, Low, Close, Volume, Dividends, Stock Splits]}
with .NS tickers and the Yahoo index symbols ^NSEI (Nifty 50) and ^CRSLDX (Nifty 500). utils/yf_safe.py
calls it first and goes to Yahoo only for what it cannot serve.
"""
from __future__ import annotations

import glob
import io
import os
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

IST = timezone(timedelta(hours=5, minutes=30))
ROOT = "data/nse"
PX, IX = os.path.join(ROOT, "prices"), os.path.join(ROOT, "indices")
SYMCHG = os.path.join(ROOT, "symbol_changes.csv")
SERIES = ["EQ", "BE", "BZ", "SM", "ST"]               # preferred order when one symbol trades in two on a day
INDEX_SYMBOLS = {"^NSEI": "Nifty 50", "^CRSLDX": "Nifty 500", "^NSEBANK": "Nifty Bank", "^CNXIT": "Nifty IT",
                 "^NSEMDCP50": "Nifty Midcap 50", "^CNXSC": "Nifty Smallcap 100"}
ADJ_TOL = 0.004                                       # PREV_CLOSE vs close: rounding is 0.05 at most; actions move it far more
PUBLISH = (18, 30)                                    # NSE's day files are normally out by ~18:00-19:00 IST


# ================================================================== the store
def _parts(base: str) -> List[str]:
    return sorted(glob.glob(os.path.join(base, "*.parquet"))) + sorted(glob.glob(os.path.join(base, "d", "*.parquet")))


def _stamp(base: str) -> str:
    """Changes whenever a file is added or rewritten (cache key)."""
    return "|".join(f"{os.path.basename(f)}:{os.path.getsize(f)}" for f in _parts(base))


@lru_cache(maxsize=4)
def _load(base: str, stamp: str) -> pd.DataFrame:
    fs = _parts(base)
    if not fs:
        return pd.DataFrame()
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return _tidy(d)


def _tidy(d: pd.DataFrame) -> pd.DataFrame:
    d["date"] = pd.to_datetime(d["date"])
    key = "symbol" if "symbol" in d else "index"
    d = d.drop_duplicates([key, "date"], keep="last").sort_values([key, "date"]).reset_index(drop=True)
    for c in ("vwap", "deliv_pct", "trades", "pe", "pb", "dy"):    # secondary columns only: prices stay float64 (exact paise)
        if c in d and d[c].dtype.kind in "fi":
            d[c] = d[c].astype("float32")
    for c in (key, "series"):
        if c in d:
            d[c] = d[c].astype("category")
    return d


def _rows_for(symbols: set) -> pd.DataFrame:
    """Only these symbols' rows, read with a parquet filter (the dashboard's one-stock reports)."""
    fs = _parts(PX)
    if not fs:
        return pd.DataFrame()
    parts = [pd.read_parquet(f, filters=[("symbol", "in", sorted(symbols))]) for f in fs]
    parts = [p for p in parts if len(p)]
    return _tidy(pd.concat(parts, ignore_index=True)) if parts else pd.DataFrame()


def raw(kind: str = "stocks") -> pd.DataFrame:
    base = PX if kind == "stocks" else IX
    return _load(base, _stamp(base))


def last_date(kind: str = "stocks") -> Optional[pd.Timestamp]:
    """The newest session in the store (from file names; no reading)."""
    base = PX if kind == "stocks" else IX
    days = [os.path.basename(f)[:8] for f in glob.glob(os.path.join(base, "d", "*.parquet"))]
    if days:
        return pd.Timestamp(max(days))
    months = sorted(glob.glob(os.path.join(base, "*.parquet")))
    if not months:
        return None
    return pd.to_datetime(pd.read_parquet(months[-1], columns=["date"])["date"]).max()


def write_day(kind: str, day: pd.Timestamp, rows: pd.DataFrame) -> str:
    base = PX if kind == "stocks" else IX
    os.makedirs(os.path.join(base, "d"), exist_ok=True)
    p = os.path.join(base, "d", f"{day:%Y%m%d}.parquet")
    rows.to_parquet(p, index=False)
    return p


def compact(kind: str = "stocks", today: Optional[pd.Timestamp] = None) -> int:
    """Fold daily files of finished months into one file per month (keeps git growth to ~70 KB a day)."""
    base = PX if kind == "stocks" else IX
    today = pd.Timestamp(today or datetime.now(IST).date())
    groups: Dict[str, List[str]] = {}
    for f in glob.glob(os.path.join(base, "d", "*.parquet")):
        ym = os.path.basename(f)[:6]
        if ym < today.strftime("%Y%m"):
            groups.setdefault(ym, []).append(f)
    for ym, fs in groups.items():
        mp = os.path.join(base, f"{ym[:4]}_{ym[4:]}.parquet")
        parts = ([pd.read_parquet(mp)] if os.path.exists(mp) else []) + [pd.read_parquet(f) for f in sorted(fs)]
        d = pd.concat(parts, ignore_index=True)
        key = "symbol" if "symbol" in d else "index"
        d.drop_duplicates([key, "date"], keep="last").sort_values(["date", key]).to_parquet(mp, index=False)
        for f in fs:
            os.remove(f)
    return len(groups)


# ================================================================== fetching
def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().str.replace(",", "").replace({"-": None, "": None}), errors="coerce")


def parse_stocks(content: bytes) -> pd.DataFrame:
    d = pd.read_csv(io.BytesIO(content))
    d.columns = [c.strip() for c in d.columns]
    d["SERIES"] = d["SERIES"].astype(str).str.strip()
    d = d[d["SERIES"].isin(SERIES)].copy()
    out = pd.DataFrame({
        "date": pd.to_datetime(d["DATE1"].astype(str).str.strip(), format="%d-%b-%Y", errors="coerce"),
        "symbol": d["SYMBOL"].astype(str).str.strip(), "series": d["SERIES"],
        "open": _num(d["OPEN_PRICE"]), "high": _num(d["HIGH_PRICE"]), "low": _num(d["LOW_PRICE"]),
        "close": _num(d["CLOSE_PRICE"]), "prev_close": _num(d["PREV_CLOSE"]), "vwap": _num(d.get("AVG_PRICE", pd.Series(dtype=str))),
        "volume": _num(d["TTL_TRD_QNTY"]), "trades": _num(d.get("NO_OF_TRADES", pd.Series(dtype=str))),
        "deliv_qty": _num(d.get("DELIV_QTY", pd.Series(dtype=str))), "deliv_pct": _num(d.get("DELIV_PER", pd.Series(dtype=str))),
    })
    out["series_rank"] = out["series"].map({s: i for i, s in enumerate(SERIES)})
    out = out.dropna(subset=["date", "close"]).sort_values("series_rank").drop_duplicates(["symbol", "date"])
    return out.drop(columns="series_rank").reset_index(drop=True)


def parse_indices(content: bytes) -> pd.DataFrame:
    d = pd.read_csv(io.BytesIO(content))
    d.columns = [c.strip() for c in d.columns]
    return pd.DataFrame({
        "date": pd.to_datetime(d["Index Date"].astype(str).str.strip(), format="%d-%m-%Y", errors="coerce"),
        "index": d["Index Name"].astype(str).str.strip(),
        "open": _num(d["Open Index Value"]), "high": _num(d["High Index Value"]), "low": _num(d["Low Index Value"]),
        "close": _num(d["Closing Index Value"]), "volume": _num(d.get("Volume", pd.Series(dtype=str))),
        "pe": _num(d.get("P/E", pd.Series(dtype=str))), "pb": _num(d.get("P/B", pd.Series(dtype=str))),
        "dy": _num(d.get("Div Yield", pd.Series(dtype=str))),
    }).dropna(subset=["date", "close"])


def fetch_day(S, day: pd.Timestamp, tries: int = 2) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """(stocks, indices) for one session, None for a file NSE does not have (holiday / not out yet)."""
    from utils.nse_data import ARCH
    st = ix = None
    try:
        st = parse_stocks(S.get(f"{ARCH}/products/content/sec_bhavdata_full_{day:%d%m%Y}.csv", tries=tries).content)
    except Exception:
        st = None
    if st is None and day.weekday() >= 5:
        return None, None                                # no weekend session: skip the index file too
    try:
        ix = parse_indices(S.get(f"{ARCH}/content/indices/ind_close_all_{day:%d%m%Y}.csv", tries=tries).content)
    except Exception:
        ix = None
    return st, ix


def fetch_symbol_changes(S) -> pd.DataFrame:
    from utils.nse_data import ARCH
    r = S.get(f"{ARCH}/content/equities/symbolchange.csv", tries=2)
    d = pd.read_csv(io.BytesIO(r.content), header=None, encoding_errors="ignore")
    # columns: company name, old symbol, new symbol, date (some copies carry a header row)
    d = d.iloc[:, :4]
    d.columns = ["company", "old", "new", "date"]
    d = d[~d["old"].astype(str).str.upper().str.contains("SYMBOL")]
    d["date"] = pd.to_datetime(d["date"].astype(str).str.strip(), errors="coerce", dayfirst=True)
    d["old"], d["new"] = d["old"].astype(str).str.strip(), d["new"].astype(str).str.strip()
    d = d.dropna(subset=["date"])
    os.makedirs(ROOT, exist_ok=True)
    d.to_csv(SYMCHG, index=False, date_format="%Y-%m-%d")
    return d


def update(S=None, upto: Optional[pd.Timestamp] = None, start: Optional[pd.Timestamp] = None, sleep: float = 0.25,
           log=print, finish: bool = True) -> Dict[str, int]:
    """Fetch every weekday after the store's last session (or from `start`) up to `upto`. Idempotent."""
    if S is None:
        from utils.nse_data import Session
        S = Session()
    now = datetime.now(IST)
    upto = pd.Timestamp(upto or now.date())
    last = last_date("stocks")
    day = pd.Timestamp(start) if start is not None else (last + pd.Timedelta(days=1) if last is not None else upto - pd.Timedelta(days=30))
    got = dict(sessions=0, holidays=0, index_days=0)
    closed = []
    while day <= upto:
        weekend = day.weekday() >= 5                     # NSE trades some weekends: budget day, Muhurat
        st, ix = fetch_day(S, day, tries=1 if weekend else 2)
        if st is not None and len(st):
            write_day("stocks", day, st)
            got["sessions"] += 1
        elif not weekend and (day < pd.Timestamp(now.date()) or (now.hour, now.minute) >= (22, 0)):
            got["holidays"] += 1                         # a past weekday with no file: NSE was closed
            closed.append(day)
        if ix is not None and len(ix):
            write_day("indices", day, ix)
            got["index_days"] += 1
        if sleep and (st is not None or not weekend):
            time.sleep(sleep)
        if not weekend and (got["sessions"] + got["holidays"]) % 50 == 0:
            log(f"[nse-prices] {day:%d %b %Y}: {got}")
        day += pd.Timedelta(days=1)
    if closed:
        note_holidays(closed)
    if finish:
        try:
            fetch_symbol_changes(S)
        except Exception as e:
            log(f"[nse-prices] symbol changes: {e}")
        compact("stocks")
        compact("indices")
    _load.cache_clear()
    return got


# ================================================================== reading (Yahoo-shaped)
def _symbol_map() -> Dict[str, List[Tuple[str, pd.Timestamp]]]:
    """new symbol -> [(old symbol, change date)] to stitch a renamed stock's history together."""
    try:
        d = pd.read_csv(SYMCHG, parse_dates=["date"])
    except Exception:
        return {}
    out: Dict[str, List[Tuple[str, pd.Timestamp]]] = {}
    for r in d.itertuples():
        out.setdefault(r.new, []).append((r.old, r.date))
    return out


JUMP = np.log(1.5)                                    # an overnight move this big is outside every price band: an action
COMMON = np.array([1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10, 2 / 5, 2 / 3, 3 / 5, 1 / 6, 1 / 8, 1 / 20, 2.0, 5.0, 10.0])


def _actions() -> Dict[str, list]:
    """symbol -> [(ex_date, factor | nan, kind)] from NSE's corporate-action records (data/nse/corp_actions.csv)."""
    try:
        from utils.nse_data import F
        ca = pd.read_csv(F["ca"], parse_dates=["ex_date"])
    except Exception:
        return {}
    ca = ca[ca["kind"].isin(["split", "bonus", "consolidation", "demerger", "rights"])]
    out: Dict[str, list] = {}
    for r in ca.itertuples():
        out.setdefault(str(r.symbol), []).append((r.ex_date, float(r.factor) if r.factor == r.factor else np.nan, r.kind))
    return out


def factors(g: pd.DataFrame, acts: Optional[list] = None) -> np.ndarray:
    """
    Price factor on each session (1 = nothing happened). NSE's PREV_CLOSE is not adjusted on every ex-date (CUPID's
    1:5 split on 9 Mar 2026 kept 402.20), so, in order:
      1. NSE's announced split / bonus / consolidation ratio, on the first session on or after its ex-date, when the
         price moved toward it;
      2. a demerger or rights issue: the session's own price ratio (what the stock lost to the spun-off part);
      3. any other overnight move beyond 50% / -33%: PREV_CLOSE's ratio if NSE adjusted it, else the nearest common
         split / bonus ratio of the opening gap if within 4%, else left alone (a real crash).
    """
    c = g["close"].values.astype(float)
    o = g["open"].values.astype(float) if "open" in g else np.full(len(g), np.nan)
    pc_nse = g["prev_close"].values.astype(float)
    d = pd.DatetimeIndex(g["date"].values)
    f = np.ones(len(g))
    q = np.r_[np.nan, c[1:] / c[:-1]]
    used = set()
    qo_all = np.r_[np.nan, o[1:] / c[:-1]]
    ratio: Dict[int, float] = {}                     # a split and a bonus on the same day multiply (DELPHIFX 0.2 x 1/3)
    for ex, fac, kind in acts or []:
        k = int(np.searchsorted(d.values, np.datetime64(ex)))
        if k <= 0 or k >= len(g) or (d[k] - ex).days > 6 or not np.isfinite(q[k]):
            continue
        if kind in ("split", "bonus", "consolidation") and fac == fac and fac > 0:
            ratio[k] = ratio.get(k, 1.0) * fac
        elif kind in ("demerger", "rights") and abs(np.log(q[k])) > np.log(1.08) and k not in ratio:
            f[k], _ = q[k], used.add(k)
    for k, fac in ratio.items():
        # the close can run far on an ex-date (GOLDIAM +17% after its 1:3 bonus): the opening gap is checked as well
        near = min(abs(np.log(x / fac)) - abs(np.log(x)) for x in (q[k], qo_all[k]) if np.isfinite(x) and x > 0)
        if near < 0:
            f[k], _ = fac, used.add(k)
    for k in np.flatnonzero(np.abs(np.log(np.where(np.isfinite(q), q, 1.0))) > JUMP):
        if k in used:
            continue
        r = pc_nse[k] / c[k - 1] if c[k - 1] > 0 else np.nan
        if np.isfinite(r) and abs(r - 1) > ADJ_TOL and abs(np.log(q[k] / r)) < 0.25:
            f[k] = r
            continue
        # the opening gap carries the action without the day's own move (CUPID 9 Mar 2026: open 82.0 / 402.2 = 0.204)
        qo = o[k] / c[k - 1] if np.isfinite(o[k]) and o[k] > 0 and c[k - 1] > 0 else q[k]
        best = COMMON[np.argmin(np.abs(np.log(qo / COMMON)))]
        if abs(np.log(qo / best)) < np.log(1.04):        # HEG's demerger gap (0.357) is not a 1:3 split
            f[k] = best
    return f


def adjust(g: pd.DataFrame, acts: Optional[list] = None) -> pd.DataFrame:
    """One symbol's raw rows (sorted by date) -> adjusted OHLCV with Stock Splits on each ex-date."""
    g = g.sort_values("date")
    f = pd.Series(factors(g, acts), index=g.index)
    # every price before an ex-date is multiplied by that day's factor (and all later ones)
    cum = f[::-1].cumprod()[::-1].shift(-1).fillna(1.0).values
    out = pd.DataFrame(index=pd.DatetimeIndex(g["date"].values, name="Date"))
    for a, b in (("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close")):
        out[b] = g[a].values * cum
    out["Volume"] = (g["volume"].values / cum).round()
    out["Dividends"] = 0.0
    out["Stock Splits"] = np.where(f.values != 1.0, 1.0 / f.values, 0.0)       # Yahoo's convention: 5.0 for a 1:5 split
    out["Delivery %"] = g["deliv_pct"].values if "deliv_pct" in g else np.nan
    return out


def history(tickers: Iterable[str], start=None, end=None) -> Dict[str, pd.DataFrame]:
    """Adjusted daily bars from the store for .NS tickers and the mapped index symbols. Missing -> absent."""
    want = list(dict.fromkeys(tickers))
    out: Dict[str, pd.DataFrame] = {}
    st = [t for t in want if str(t).endswith(".NS")]
    ix = [t for t in want if t in INDEX_SYMBOLS]
    s0 = pd.Timestamp(start) if start is not None else None
    e0 = pd.Timestamp(end) if end is not None else None
    cut = lambda d: d[d.index >= s0] if s0 is not None else d
    if st:
        chg = _symbol_map()
        # an old symbol asked for (ICICIPRULI after it became ICICILIFE) is served from its newest name, stitched
        fwd = {o: (n, w) for n, olds in chg.items() for o, w in olds}

        def newest(x: str) -> str:
            seen = set()
            while x in fwd and x not in seen:
                seen.add(x)
                x = fwd[x][0]
            return x
        cur = {t: newest(t[:-3]) for t in st}
        syms = set(cur.values()) | {t[:-3] for t in st} | {o for v in cur.values() for o, _ in chg.get(v, [])}
        R = raw("stocks") if len(syms) > 40 else _rows_for(syms)
        if len(R):
            R = R[R["symbol"].isin(syms)]
            R["symbol"] = R["symbol"].astype(str)
            acts = _actions()
            by = {k: g for k, g in R.groupby("symbol", sort=False)}
            for t in st:
                sym = cur[t]
                own = by.get(t[:-3])
                # the old name still trades after it was given up (DTIL, renamed DPTL in 2010, is another company now)
                if sym != t[:-3] and own is not None and own["date"].max() > fwd[t[:-3]][1] + pd.Timedelta(days=7):
                    sym = t[:-3]
                parts = [by[sym]] if sym in by else []
                for old, when in chg.get(sym, []):
                    if old in by:
                        parts.append(by[old][by[old]["date"] < when])
                if not parts:
                    continue
                g = pd.concat(parts).drop_duplicates("date", keep="first")
                a = adjust(g, acts.get(sym, []) + [x for o, _ in chg.get(sym, []) for x in acts.get(o, [])])
                if e0 is not None:
                    a = a[a.index < e0]
                a = cut(a)
                if len(a):
                    out[t] = a
    if ix:
        X = raw("indices")
        if len(X):
            for t in ix:
                g = X[X["index"].str.lower() == INDEX_SYMBOLS[t].lower()].sort_values("date")
                if g.empty:
                    continue
                a = pd.DataFrame({"Open": g.open.values, "High": g.high.values, "Low": g.low.values, "Close": g.close.values,
                                  "Volume": g.volume.fillna(0).values, "Dividends": 0.0, "Stock Splits": 0.0},
                                 index=pd.DatetimeIndex(g["date"].values, name="Date"))
                if e0 is not None:
                    a = a[a.index < e0]
                a = cut(a)
                if len(a):
                    out[t] = a
    return out


def expected_session(now: Optional[datetime] = None) -> pd.Timestamp:
    """The newest session whose NSE file should be out by now (weekdays; holidays are found out by fetching)."""
    now = now or datetime.now(IST)
    d = pd.Timestamp(now.date())
    if now.weekday() >= 5 or (now.hour, now.minute) < PUBLISH:
        d -= pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def is_current(now: Optional[datetime] = None) -> bool:
    """True when the store holds the latest session there should be (a holiday is fine: checked by update())."""
    last = last_date("stocks")
    if last is None:
        return False
    exp = expected_session(now)
    if last >= exp:
        return True
    # the gap is only weekdays NSE was closed on: update() tried them and found no file
    try:
        tried = pd.read_csv(os.path.join(ROOT, "holidays.csv"), parse_dates=["date"])["date"]
        gap = [d for d in pd.bdate_range(last + pd.Timedelta(days=1), exp)]
        return all(d in set(tried) for d in gap)
    except Exception:
        return False


def note_holidays(days: Iterable[pd.Timestamp]) -> None:
    p = os.path.join(ROOT, "holidays.csv")
    old = pd.read_csv(p, parse_dates=["date"]) if os.path.exists(p) else pd.DataFrame(columns=["date"])
    new = pd.concat([old, pd.DataFrame({"date": pd.to_datetime(list(days))})]).drop_duplicates()
    os.makedirs(ROOT, exist_ok=True)
    new.sort_values("date").to_csv(p, index=False, date_format="%Y-%m-%d")
