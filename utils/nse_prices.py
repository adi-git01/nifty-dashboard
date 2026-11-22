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

import re
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
    drop_cols = [c for c in ("vwap", "trades", "deliv_qty") if c in d.columns]
    if drop_cols:
        d = d.drop(columns=drop_cols)
    key = "symbol" if "symbol" in d else "index"
    d = d.drop_duplicates([key, "date"], keep="last").sort_values([key, "date"]).reset_index(drop=True)
    for c in ("deliv_pct", "pe", "pb", "dy"):    # secondary columns only: prices stay float64 (exact paise)
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


def _prepare_write(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unused columns (vwap/trades/deliv_qty) and downcast secondary columns to float32 while keeping OHLC float64."""
    out = df.drop(columns=[c for c in ("vwap", "trades", "deliv_qty") if c in df.columns]).copy()
    for c in ("deliv_pct", "pe", "pb", "dy"):
        if c in out.columns and out[c].dtype.kind in "fi":
            out[c] = out[c].astype("float32")
    return out


def first_date(kind: str = "stocks") -> Optional[pd.Timestamp]:
    """The earliest session in the store (from parquet date column or file names)."""
    base = PX if kind == "stocks" else IX
    months = sorted(glob.glob(os.path.join(base, "*.parquet")))
    if months:
        try:
            dt = pd.to_datetime(pd.read_parquet(months[0], columns=["date"])["date"], errors="coerce").min()
            if pd.notna(dt):
                return dt
        except Exception:
            pass
    days = sorted(glob.glob(os.path.join(base, "d", "*.parquet")))
    if days:
        return pd.Timestamp(os.path.basename(days[0])[:8])
    return None


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
    _prepare_write(rows).to_parquet(p, index=False, compression="zstd")
    return p


def compact(kind: str = "stocks", today: Optional[pd.Timestamp] = None) -> int:
    """Fold daily files of finished months into one file per month (keeps git growth small)."""
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
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d[d["date"].dt.strftime("%Y%m") == ym]
        d = _prepare_write(d.drop_duplicates([key, "date"], keep="last").sort_values(["date", key]))
        d.to_parquet(mp, index=False, compression="zstd")
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
        "close": _num(d["CLOSE_PRICE"]), "prev_close": _num(d["PREV_CLOSE"]),
        "volume": _num(d["TTL_TRD_QNTY"]), "deliv_pct": _num(d.get("DELIV_PER", pd.Series(dtype=str))),
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


def _fetch_day_checked(S, day: pd.Timestamp, tries: int = 2) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], bool]:
    """(stocks, indices, holiday_confirmed). Rejects stale prior-session copies NSE serves on Sundays/holidays."""
    from utils.nse_data import ARCH
    st = ix = None
    holiday_confirmed = False
    target = day.normalize()
    try:
        raw_st = parse_stocks(S.get(f"{ARCH}/products/content/sec_bhavdata_full_{day:%d%m%Y}.csv", tries=tries).content)
        exact = raw_st[raw_st["date"].dt.normalize() == target]
        if len(exact):
            st = exact.reset_index(drop=True)
        elif len(raw_st):
            # NSE serves HTTP 200 containing the previous trading day's file on Sundays and holidays.
            holiday_confirmed = True
    except Exception as e:
        if "HTTP 404" in str(e):
            holiday_confirmed = True
        st = None
    if st is None and day.weekday() >= 5:
        return None, None, holiday_confirmed             # no weekend session: skip the index file too
    try:
        raw_ix = parse_indices(S.get(f"{ARCH}/content/indices/ind_close_all_{day:%d%m%Y}.csv", tries=tries).content)
        exact_ix = raw_ix[raw_ix["date"].dt.normalize() == target]
        ix = exact_ix.reset_index(drop=True) if len(exact_ix) else None
    except Exception as e:
        if "HTTP 404" in str(e) and st is None:
            holiday_confirmed = True
        ix = None
    if st is not None and ix is None:
        # Index file unavailable for an active stock session; do not treat as holiday.
        holiday_confirmed = False
    elif st is None and ix is not None:
        # Index file exists for today, so NSE was open even if stock bhavcopy failed transiently.
        holiday_confirmed = False
    return st, ix, holiday_confirmed


def fetch_day(S, day: pd.Timestamp, tries: int = 2) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """(stocks, indices) for one session, None for a file NSE does not have (holiday / not out yet)."""
    st, ix, _ = _fetch_day_checked(S, day, tries=tries)
    return st, ix


def fetch_symbol_changes(S) -> pd.DataFrame:
    from utils.nse_data import ARCH
    r = S.get(f"{ARCH}/content/equities/symbolchange.csv", tries=2)
    d = pd.read_csv(io.BytesIO(r.content), header=None, encoding_errors="ignore")
    # columns: company name, old symbol, new symbol, date (some copies carry a header row)
    d = d.iloc[:, :4]
    d.columns = ["company", "old", "new", "date"]
    d = d[~d["old"].astype(str).str.upper().str.contains("SYMBOL")]
    d["date"] = pd.to_datetime(d["date"].astype(str).str.strip(), format="mixed", dayfirst=True, errors="coerce")
    d["old"], d["new"] = d["old"].astype(str).str.strip(), d["new"].astype(str).str.strip()
    d = d.dropna(subset=["date"])
    d = d[(d["old"] != "") & (d["new"] != "") & (d["old"] != d["new"])]
    os.makedirs(ROOT, exist_ok=True)
    d.to_csv(SYMCHG, index=False, date_format="%Y-%m-%d")
    return d


def update(S=None, upto: Optional[pd.Timestamp] = None, start: Optional[pd.Timestamp] = None, sleep: float = 0.25,
           log=print, finish: bool = True) -> Dict[str, int]:
    """Fetch every session after the store's last session (or from `start`) up to `upto`. Idempotent."""
    if S is None:
        from utils.nse_data import Session
        S = Session()
    now = datetime.now(IST)
    upto = pd.Timestamp(upto or now.date())
    last_st = last_date("stocks")
    last_ix = last_date("indices")
    day = pd.Timestamp(start) if start is not None else (last_st + pd.Timedelta(days=1) if last_st is not None else upto - pd.Timedelta(days=30))
    got = dict(sessions=0, holidays=0, index_days=0)
    if start is None and last_ix is not None and last_st is not None and last_ix < last_st:
        from utils.nse_data import ARCH
        ix_day = last_ix + pd.Timedelta(days=1)
        while ix_day <= last_st:
            st_p = os.path.join(PX, "d", f"{ix_day:%Y%m%d}.parquet")
            if ix_day.weekday() < 5 or os.path.exists(st_p):
                try:
                    raw_ix = parse_indices(S.get(f"{ARCH}/content/indices/ind_close_all_{ix_day:%d%m%Y}.csv", tries=2).content)
                    exact_ix = raw_ix[raw_ix["date"].dt.normalize() == ix_day.normalize()]
                    if len(exact_ix):
                        write_day("indices", ix_day, exact_ix.reset_index(drop=True))
                        got["index_days"] += 1
                except Exception:
                    pass
            ix_day += pd.Timedelta(days=1)
    closed, opened = [], []
    while day <= upto:
        weekend = day.weekday() >= 5                     # NSE trades some weekends: budget day, Muhurat
        st, ix, holiday_confirmed = _fetch_day_checked(S, day, tries=1 if weekend else 2)
        if st is not None and len(st):
            write_day("stocks", day, st)
            got["sessions"] += 1
            opened.append(day)
        elif not weekend and holiday_confirmed and (day < pd.Timestamp(now.date()) or (now.hour, now.minute) >= (20, 0)):
            got["holidays"] += 1                         # confirmed holiday / stale-date file: NSE was closed
            closed.append(day)
        if ix is not None and len(ix):
            write_day("indices", day, ix)
            got["index_days"] += 1
        if sleep and (st is not None or not weekend):
            time.sleep(sleep)
        if not weekend and (got["sessions"] + got["holidays"]) % 50 == 0:
            log(f"[nse-prices] {day:%d %b %Y}: {got}")
        day += pd.Timedelta(days=1)
    if opened:
        unnote_holidays(opened)
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
        old_s, new_s = str(r.old).strip(), str(r.new).strip()
        if not old_s or not new_s or old_s == new_s or pd.isna(r.date):
            continue
        out.setdefault(new_s, []).append((old_s, r.date))
    return out


def _ancestors(sym: str, chg: Dict[str, List[Tuple[str, pd.Timestamp]]]) -> List[Tuple[str, pd.Timestamp]]:
    """All ancestor (old_symbol, cutoff_date) pairs across multi-hop rename chains, cycle-safe."""
    out: List[Tuple[str, pd.Timestamp]] = []
    seen = {sym}
    stack: List[Tuple[str, Optional[pd.Timestamp]]] = [(sym, None)]
    while stack:
        cur_sym, bound = stack.pop()
        for old_sym, when in chg.get(cur_sym, []):
            if not old_sym or old_sym in seen or pd.isna(when):
                continue
            seen.add(old_sym)
            cutoff = when if bound is None else min(when, bound)
            out.append((old_sym, cutoff))
            stack.append((old_sym, cutoff))
    return out


JUMP = np.log(1.5)                                    # an overnight move this big is outside every price band: an action
COMMON = np.array([1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10, 2 / 5, 2 / 3, 3 / 5, 1 / 6, 1 / 8, 1 / 20, 2.0, 5.0, 10.0])


def _actions() -> Dict[str, list]:
    """symbol -> [(ex_date, factor | nan, kind, info)] from NSE's corporate-action records (data/nse/corp_actions.csv)."""
    try:
        from utils.nse_data import F
        ca = pd.read_csv(F["ca"], parse_dates=["ex_date"])
    except Exception:
        return {}
    ca = ca[ca["kind"].isin(["split", "bonus", "consolidation", "demerger", "rights"])]
    fv = _face_values()
    splits_by_sym: Dict[str, List[Tuple[pd.Timestamp, float]]] = {}
    for r in ca[ca["kind"].isin(["split", "consolidation"])].itertuples():
        fac = float(r.factor) if r.factor == r.factor else np.nan
        if np.isfinite(fac) and fac > 0:
            splits_by_sym.setdefault(str(r.symbol), []).append((r.ex_date, fac))
    out: Dict[str, list] = {}
    for r in ca.itertuples():
        fac = float(r.factor) if r.factor == r.factor else np.nan
        info = None
        if r.kind == "rights":
            # "Rights 3:2 @ Premium Re. 0.63/-": 3 new for every 2 held at face value + premium
            subj = str(r.subject)
            if re.search(r"\b(?:ccps|warrant|ncd|debenture|ocps|preference|ncrps)\b", subj, re.I):
                continue
            m = re.search(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*@?\s*(?:(Premium|Prm)\s*)?(?:Rs\.?|Re\.?)?\s*(\d+(?:\.\d+)?)",
                          subj, re.I)
            has_prm = bool(re.search(r"\b(?:Premium|Prm)\b", subj, re.I))
            if m:
                new_n, held_n, val = float(m.group(1)), float(m.group(2)), float(m.group(4))
                f0 = pd.to_numeric(getattr(r, "face_val", np.nan), errors="coerce")
                if not (np.isfinite(f0) and f0 > 0):
                    f0 = fv.get(str(r.symbol), np.nan)
                    if f0 and np.isfinite(f0):
                        # EQUITY_L.csv holds post-split face value; undo splits strictly after the rights ex-date.
                        later = 1.0
                        for sd, sf in splits_by_sym.get(str(r.symbol), []):
                            if sd > r.ex_date and sf > 0:
                                later *= sf
                        if later > 0:
                            f0 = float(f0) / later
                if has_prm and f0 and np.isfinite(f0) and f0 > 0:
                    info = (new_n, held_n, float(f0) + val)
                elif not has_prm and val > 0:
                    info = (new_n, held_n, val)
        out.setdefault(str(r.symbol), []).append((r.ex_date, fac, r.kind, info))
    return out


@lru_cache(maxsize=1)
def _face_values() -> Dict[str, float]:
    """symbol -> face value from NSE's equity list (EQUITY_L.csv), for the issue price of a rights issue."""
    try:
        e = pd.read_csv("EQUITY_L.csv")
        e.columns = [c.strip().upper() for c in e.columns]
        return dict(zip(e["SYMBOL"].astype(str), pd.to_numeric(e["FACE VALUE"], errors="coerce")))
    except Exception:
        return {}


def _factors_split_aware(g: pd.DataFrame, acts: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    (price_factors, share_factors) on each session (1 = nothing happened).
    Splits, bonuses, and consolidations rescale both prices and share count/volume.
    Rights issues and demergers rescale historical prices only (do not create Stock Splits or rescale share volume).
    """
    c = g["close"].values.astype(float)
    o = g["open"].values.astype(float) if "open" in g else np.full(len(g), np.nan)
    pc_nse = g["prev_close"].values.astype(float)
    d = pd.DatetimeIndex(g["date"].values)
    f_px = np.ones(len(g))
    f_sh = np.ones(len(g))
    valid_c = (c[:-1] > 0) & (c[1:] > 0)
    q = np.r_[np.nan, np.where(valid_c, c[1:] / np.where(c[:-1] > 0, c[:-1], np.nan), np.nan)]
    used = set()
    valid_o = (c[:-1] > 0) & (o[1:] > 0)
    qo_all = np.r_[np.nan, np.where(valid_o, o[1:] / np.where(c[:-1] > 0, c[:-1], np.nan), np.nan)]
    ratio: Dict[int, float] = {}                     # a split and a bonus on the same day multiply (DELPHIFX 0.2 x 1/3)
    px_only: Dict[int, float] = {}                   # rights / demergers (price-only scaling)
    seen_acts = set()
    for act in acts or []:
        ex, fac, kind = act[:3]
        info = act[3] if len(act) > 3 else None
        dedup_key = (pd.Timestamp(ex).normalize(), str(kind), round(float(fac), 6) if fac == fac else None, info)
        if dedup_key in seen_acts:
            continue
        seen_acts.add(dedup_key)
        k = int(np.searchsorted(d.values, np.datetime64(ex)))
        if k <= 0 or k >= len(g) or (d[k] - ex).days > 6 or not np.isfinite(q[k]):
            continue
        if kind in ("split", "bonus", "consolidation") and fac == fac and fac > 0:
            ratio[k] = ratio.get(k, 1.0) * fac
        elif kind == "rights" and info:
            # theoretical ex-rights price: (held x last close + new x issue price) / (held + new), as Zerodha does
            new, held, price = info
            terp = (held * c[k - 1] + new * price) / (held + new)
            if 0 < price < c[k - 1] and terp / c[k - 1] < 0.995:
                px_only[k] = px_only.get(k, 1.0) * (terp / c[k - 1])
                used.add(k)
        elif kind == "demerger":
            # a demerger's value goes at the special pre-open session: accept downward opening carve-outs only (< 0.96)
            gap = qo_all[k] if np.isfinite(qo_all[k]) and qo_all[k] > 0 else q[k]
            if np.isfinite(gap) and 0.05 < gap < 0.96:
                px_only[k] = px_only.get(k, 1.0) * gap
                used.add(k)
        elif kind == "rights" and np.isfinite(q[k]) and abs(np.log(q[k])) > np.log(1.08):
            gap = qo_all[k] if np.isfinite(qo_all[k]) and qo_all[k] > 0 else q[k]
            if np.isfinite(gap) and 0.1 < gap < 0.96:
                px_only[k] = px_only.get(k, 1.0) * gap
                used.add(k)
    for k, fac in ratio.items():
        # the close can run far on an ex-date (GOLDIAM +17% after its 1:3 bonus): the opening gap is checked as well
        cands = [x for x in (q[k], qo_all[k]) if np.isfinite(x) and x > 0]
        if not cands:
            continue
        near = min(abs(np.log(x / fac)) - abs(np.log(x)) for x in cands)
        small_bonus_ok = (
            0.80 <= fac < 1.0
            and (
                min(cands) < 0.99
                or (np.isfinite(pc_nse[k]) and c[k - 1] > 0 and abs(pc_nse[k] / c[k - 1] - fac) < 0.05)
            )
        )
        if near < 0 or small_bonus_ok:
            f_px[k] *= fac
            f_sh[k] *= fac
            used.add(k)
    for k, pfac in px_only.items():
        f_px[k] *= pfac
    for k in np.flatnonzero(np.abs(np.log(np.where(np.isfinite(q) & (q > 0), q, 1.0))) > JUMP):
        if k in used or (d[k] - d[k - 1]).days > 10:
            continue
        r = pc_nse[k] / c[k - 1] if c[k - 1] > 0 else np.nan
        if np.isfinite(r) and abs(r - 1) > ADJ_TOL and abs(np.log(q[k] / r)) < 0.25:
            f_px[k] *= r
            f_sh[k] *= r
            continue
        # the opening gap carries the action without the day's own move (CUPID 9 Mar 2026: open 82.0 / 402.2 = 0.204)
        qo = o[k] / c[k - 1] if np.isfinite(o[k]) and o[k] > 0 and c[k - 1] > 0 else q[k]
        best = COMMON[np.argmin(np.abs(np.log(qo / COMMON)))]
        tight = abs(np.log(qo / best)) < (np.log(1.04) if abs(np.log(qo)) > np.log(1.8) or best not in (2 / 3, 3 / 5) else np.log(1.015))
        if tight:        # real -34%..-42% crashes and HEG's demerger gap (0.357) are not snapped to splits
            f_px[k] *= best
            f_sh[k] *= best
    return f_px, f_sh


def factors(g: pd.DataFrame, acts: Optional[list] = None) -> np.ndarray:
    """
    Price factor on each session (1 = nothing happened). NSE's PREV_CLOSE is not adjusted on every ex-date (CUPID's
    1:5 split on 9 Mar 2026 kept 402.20).
    """
    return _factors_split_aware(g, acts)[0]


def adjust(g: pd.DataFrame, acts: Optional[list] = None) -> pd.DataFrame:
    """One symbol's raw rows (sorted by date) -> adjusted OHLCV with Stock Splits on split/bonus ex-dates."""
    g = g.sort_values("date")
    fp, fs = _factors_split_aware(g, acts)
    f_px = pd.Series(fp, index=g.index)
    f_sh = pd.Series(fs, index=g.index)
    # every price before an ex-date is multiplied by that day's price factor (and all later ones)
    cum_px = f_px[::-1].cumprod()[::-1].shift(-1).fillna(1.0).values
    cum_sh = f_sh[::-1].cumprod()[::-1].shift(-1).fillna(1.0).values
    out = pd.DataFrame(index=pd.DatetimeIndex(g["date"].values, name="Date"))
    for a, b in (("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close")):
        out[b] = g[a].values * cum_px
    out["Volume"] = (g["volume"].values / cum_sh).round()
    out["Dividends"] = 0.0
    out["Stock Splits"] = np.where(f_sh.values != 1.0, 1.0 / f_sh.values, 0.0)       # Yahoo's convention: 5.0 for a 1:5 split
    out["Delivery %"] = g["deliv_pct"].values if "deliv_pct" in g else np.nan
    return out


def history(tickers: Iterable[str], start=None, end=None) -> Dict[str, pd.DataFrame]:
    """Adjusted daily bars from the store for .NS tickers and the mapped index symbols. Missing -> absent."""
    want = list(dict.fromkeys(tickers))
    out: Dict[str, pd.DataFrame] = {}
    st = [t for t in want if str(t).endswith(".NS")]
    ix = [t for t in want if t in INDEX_SYMBOLS]
    s0 = pd.Timestamp(start) if start is not None else None
    if s0 is not None and getattr(s0, "tzinfo", None) is not None:
        s0 = s0.tz_localize(None)
    e0 = pd.Timestamp(end) if end is not None else None
    if e0 is not None and getattr(e0, "tzinfo", None) is not None:
        e0 = e0.tz_localize(None)
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
        anc_map = {v: _ancestors(v, chg) for v in set(cur.values())}
        syms = set(cur.values()) | {t[:-3] for t in st} | {o for ancs in anc_map.values() for o, _ in ancs}
        R = raw("stocks") if len(syms) > 40 else _rows_for(syms)
        if len(R):
            R = R[R["symbol"].isin(syms)].copy()
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
                sym_acts = list(acts.get(sym, []))
                for old, when in anc_map.get(sym, _ancestors(sym, chg)):
                    if old in by:
                        if by[old]["date"].max() > when + pd.Timedelta(days=7):
                            continue                       # old symbol was reused by another company: do not stitch
                        parts.append(by[old][by[old]["date"] < when])
                        sym_acts.extend(x for x in acts.get(old, []) if x[0] < when)
                if not parts:
                    continue
                g = pd.concat(parts).drop_duplicates("date", keep="first")
                a = adjust(g, sym_acts)
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
    """True when both stocks and indices stores hold the latest session there should be (holidays checked)."""
    last_st = last_date("stocks")
    if last_st is None:
        return False
    exp = expected_session(now)
    tried = None
    def _ok(last: Optional[pd.Timestamp]) -> bool:
        nonlocal tried
        if last is None:
            return False
        if last >= exp:
            return True
        try:
            if tried is None:
                hol_df = pd.read_csv(os.path.join(ROOT, "holidays.csv"), parse_dates=["date"])
                tried = set(pd.to_datetime(hol_df["date"], errors="coerce").dropna().dt.normalize())
            gap = [d.normalize() for d in pd.bdate_range(last + pd.Timedelta(days=1), exp)]
            return all(d in tried for d in gap)
        except Exception:
            return False

    if not _ok(last_st):
        return False
    has_ix = bool(glob.glob(os.path.join(IX, "*.parquet")) or glob.glob(os.path.join(IX, "d", "*.parquet")))
    if has_ix and not _ok(last_date("indices")):
        return False
    return True


def note_holidays(days: Iterable[pd.Timestamp]) -> None:
    p = os.path.join(ROOT, "holidays.csv")
    old = pd.read_csv(p, parse_dates=["date"]) if os.path.exists(p) else pd.DataFrame(columns=["date"])
    new = pd.concat([old, pd.DataFrame({"date": pd.to_datetime(list(days)).normalize()})]).drop_duplicates()
    os.makedirs(ROOT, exist_ok=True)
    new.sort_values("date").to_csv(p, index=False, date_format="%Y-%m-%d")


def unnote_holidays(days: Iterable[pd.Timestamp]) -> None:
    p = os.path.join(ROOT, "holidays.csv")
    if not os.path.exists(p):
        return
    drop = {pd.Timestamp(d).normalize() for d in days}
    old = pd.read_csv(p, parse_dates=["date"])
    if old.empty:
        return
    keep = old[~pd.to_datetime(old["date"], errors="coerce").dt.normalize().isin(drop)]
    if len(keep) < len(old):
        keep.sort_values("date").to_csv(p, index=False, date_format="%Y-%m-%d")
