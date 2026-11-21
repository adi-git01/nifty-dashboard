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


def fetch_day(S, day: pd.Timestamp) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """(stocks, indices) for one session, None for a file NSE does not have (holiday / not out yet)."""
    from utils.nse_data import ARCH
    st = ix = None
    try:
        st = parse_stocks(S.get(f"{ARCH}/products/content/sec_bhavdata_full_{day:%d%m%Y}.csv", tries=2).content)
    except Exception:
        st = None
    try:
        ix = parse_indices(S.get(f"{ARCH}/content/indices/ind_close_all_{day:%d%m%Y}.csv", tries=2).content)
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
        if day.weekday() < 5:
            st, ix = fetch_day(S, day)
            if st is not None and len(st):
                write_day("stocks", day, st)
                got["sessions"] += 1
            elif day < pd.Timestamp(now.date()) or (now.hour, now.minute) >= (22, 0):
                got["holidays"] += 1                     # a past weekday with no file: NSE was closed
                closed.append(day)
            if ix is not None and len(ix):
                write_day("indices", day, ix)
                got["index_days"] += 1
            if sleep:
                time.sleep(sleep)
            if (got["sessions"] + got["holidays"]) % 50 == 0:
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


def adjust(g: pd.DataFrame) -> pd.DataFrame:
    """One symbol's raw rows (sorted by date) -> adjusted OHLCV with Stock Splits on each ex-date."""
    g = g.sort_values("date")
    prev_close = g["close"].shift(1)
    f = (g["prev_close"] / prev_close).where(prev_close > 0)
    f = f.where((f - 1).abs() > ADJ_TOL, 1.0).fillna(1.0)
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
        syms = {t[:-3] for t in st} | {o for t in st for o, _ in chg.get(t[:-3], [])}
        R = raw("stocks") if len(syms) > 40 else _rows_for(syms)
        if len(R):
            R = R[R["symbol"].isin(syms)]
            R["symbol"] = R["symbol"].astype(str)
            by = {k: g for k, g in R.groupby("symbol", sort=False)}
            for t in st:
                sym = t[:-3]
                parts = [by[sym]] if sym in by else []
                for old, when in chg.get(sym, []):
                    if old in by:
                        parts.append(by[old][by[old]["date"] < when])
                if not parts:
                    continue
                g = pd.concat(parts).drop_duplicates("date", keep="first")
                a = adjust(g)
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
