"""
NSE's free daily data, fetched by the evening run (scripts/nse_daily.py) and read by the dashboard.

    data/nse/eod_YYYYMMDD.csv.gz   official EOD per stock incl. DELIVERY % (one file per session, ~400 kept)
    data/nse/corp_actions.csv      splits / bonuses / dividends with ex-dates (accumulated); splits and
                                   bonuses carry the price factor used by utils/corp_actions.py
    data/nse/surveillance.csv      ASM (long / short term) and GSM lists, today
    data/nse/events.csv            board meetings and results dates ahead
    data/nse/deals.csv             bulk and block deals (last 120 days)
    data/nse/market_context.csv    one row per session: Nifty P/E, P/B, dividend yield, FII / DII cash net,
                                   FII index-futures long / short

Every fetch is independent and failure-tolerant: one source down never stops the others. Probe results
(which sources answer a GitHub runner): analysis/NSE_DATA_PROBE.md. NSE data is for personal use.
"""
from __future__ import annotations

import glob
import io
import os
import re
import time
import zipfile
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd

IST = timezone(timedelta(hours=5, minutes=30))
ARCH = "https://nsearchives.nseindia.com"
WWW = "https://www.nseindia.com"
DIR = "data/nse"
DELIV_20D = os.path.join(DIR, "delivery_20d.parquet")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
HDR = {"User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9", "Referer": WWW + "/"}
F = {k: os.path.join(DIR, v) for k, v in dict(ca="corp_actions.csv", surv="surveillance.csv", events="events.csv",
                                               deals="deals.csv", ctx="market_context.csv").items()}
KEEP_EOD = 400
KEEP_DEAL_DAYS = 120


# ================================================================== fetching (evening run)
class Session:
    def __init__(self):
        import requests
        self.s = requests.Session()
        try:
            self.s.get(WWW, headers=HDR, timeout=20)          # cookies for the JSON endpoints
        except Exception:
            pass

    def get(self, url: str, tries: int = 3, timeout: float = 30):
        last = None
        for k in range(tries):
            try:
                r = self.s.get(url, headers=HDR, timeout=timeout)
                if r.status_code == 200 and r.content:
                    return r
                last = f"HTTP {r.status_code}"
                if r.status_code in (401, 403) and url.startswith(WWW):
                    self.s.get(WWW, headers=HDR, timeout=min(timeout, 20))     # cookie expired: refresh once
            except Exception as e:
                last = type(e).__name__
            if k + 1 < tries:
                time.sleep(2 * (k + 1))
        raise RuntimeError(f"{url.split('?')[0].rsplit('/', 1)[-1]}: {last}")

    def warm_quote(self, symbol: str = "RELIANCE") -> None:
        """quote-equity sits behind stricter bot protection than the other JSON endpoints (HTTP 403 from a bare session):
        opening the stock's quote page first collects the cookies it asks for."""
        try:
            self.s.get(f"{WWW}/get-quotes/equity?symbol={symbol}", headers=HDR, timeout=20)
        except Exception:
            pass

    def json(self, path: str, tries: int = 3, timeout: float = 30):
        return self.get(WWW + path, tries=tries, timeout=timeout).json()


def _d(day: pd.Timestamp) -> str:
    return day.strftime("%d%m%Y")


def fetch_delivery(S: Session, day: pd.Timestamp) -> pd.DataFrame:
    """The day's security-wise file: OHLC, volume, trades and delivery for every stock (EQ / BE / BZ / SME)."""
    r = S.get(f"{ARCH}/products/content/sec_bhavdata_full_{_d(day)}.csv")
    d = pd.read_csv(io.BytesIO(r.content))
    d.columns = [c.strip() for c in d.columns]
    d = d[d["SERIES"].astype(str).str.strip().isin(["EQ", "BE", "BZ", "SM", "ST"])].copy()
    keep = {"SYMBOL": "symbol", "SERIES": "series", "PREV_CLOSE": "prev_close", "OPEN_PRICE": "open", "HIGH_PRICE": "high",
            "LOW_PRICE": "low", "CLOSE_PRICE": "close", "AVG_PRICE": "vwap", "TTL_TRD_QNTY": "volume",
            "TURNOVER_LACS": "turnover_lacs", "NO_OF_TRADES": "trades", "DELIV_QTY": "deliv_qty", "DELIV_PER": "deliv_pct"}
    e = d[[c for c in keep if c in d.columns]].rename(columns=keep)
    for c in e.columns:
        e[c] = e[c].astype(str).str.strip()
        if c not in ("symbol", "series"):
            e[c] = pd.to_numeric(e[c].replace({"-": None}), errors="coerce")
    file_day = pd.to_datetime(d["DATE1"].astype(str).str.strip().iloc[0], errors="coerce") if "DATE1" in d and len(d) else day
    e.insert(0, "date", (file_day if pd.notna(file_day) else day).date())
    return e


def latest_session(S: Session, start: pd.Timestamp) -> Optional[pd.Timestamp]:
    for k in range(8):
        day = start - pd.Timedelta(days=k)
        if day.weekday() >= 5:
            continue
        try:
            S.get(f"{ARCH}/products/content/sec_bhavdata_full_{_d(day)}.csv", tries=1)
            return day
        except Exception:
            continue
    return None


def save_delivery(e: pd.DataFrame) -> str:
    os.makedirs(DIR, exist_ok=True)
    day = pd.Timestamp(e["date"].iloc[0])
    p = os.path.join(DIR, f"eod_{day:%Y%m%d}.csv.gz")
    e.to_csv(p, index=False)
    for old in sorted(glob.glob(os.path.join(DIR, "eod_*.csv.gz")))[:-KEEP_EOD]:
        os.remove(old)
    return p


_NUM = r"(\d+(?:\.\d+)?)"


def action_factor(subject: str) -> Tuple[str, float]:
    """(kind, price factor) from NSE's purpose text. factor = new/old price basis: 0.2 for a Rs10->Rs2 split,
    0.5 for a 1:1 bonus, 10 for a Rs1->Rs10 consolidation; NaN for dividends and everything else."""
    s = str(subject or "")
    low = s.lower()
    m = re.search(r"from\s*(?:rs\.?|re\.?|₹)?\s*" + _NUM + r".*?to\s*(?:rs\.?|re\.?|₹)?\s*" + _NUM, low)
    if ("split" in low or "sub-division" in low or "subdivision" in low) and m:
        a, b = float(m.group(1)), float(m.group(2))
        return "split", b / a if a > 0 and b > 0 else np.nan
    if "consolidat" in low and m:
        a, b = float(m.group(1)), float(m.group(2))
        return "consolidation", b / a if a > 0 and b > 0 else np.nan
    m = re.search(r"bonus\s*" + _NUM + r"\s*:\s*" + _NUM, low)
    if m:
        a, b = float(m.group(1)), float(m.group(2))           # a new shares for every b held
        return "bonus", b / (a + b) if a > 0 and b > 0 else np.nan
    if "dividend" in low:
        return "dividend", np.nan
    if "right" in low:
        return "rights", np.nan
    if "demerger" in low:
        return "demerger", np.nan                             # the price drops by the spun-off part: no simple factor
    return "other", np.nan


def fetch_corp_actions(S: Session, today: pd.Timestamp, start: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """NSE's corporate actions, 45 days back to 90 ahead; with `start`, everything from then in 90-day windows."""
    lo = pd.Timestamp(start) if start is not None else today - pd.Timedelta(days=45)
    hi = today + pd.Timedelta(days=90)
    got = []
    while lo < hi:
        to_ = min(lo + pd.Timedelta(days=90), hi)
        try:
            j = S.json(f"/api/corporates-corporateActions?index=equities&from_date={lo:%d-%m-%Y}&to_date={to_:%d-%m-%Y}")
            got += j if isinstance(j, list) else []
        except Exception as e:
            print(f"[nse] corporate actions {lo:%d %b %Y}: {e}")
        lo = to_ + pd.Timedelta(days=1)
        if start is not None:
            time.sleep(0.5)
    rows = []
    skip_series = {"GS", "GB", "TB", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9"}
    for x in got:
        ser = str(x.get("series") or "").strip().upper()
        if ser in skip_series:
            continue
        kind, fac = action_factor(x.get("subject"))
        fv = pd.to_numeric(x.get("faceVal"), errors="coerce")
        rows.append(dict(symbol=str(x.get("symbol", "")).strip(), ex_date=pd.to_datetime(x.get("exDate"), errors="coerce"),
                         record_date=pd.to_datetime(x.get("recDate"), errors="coerce"), subject=str(x.get("subject", "")).strip(),
                         kind=kind, factor=fac, face_val=float(fv) if pd.notna(fv) and fv > 0 else np.nan))
    cols = ["symbol", "ex_date", "record_date", "subject", "kind", "factor", "face_val"]
    new = pd.DataFrame(rows, columns=cols)
    old = _read(F["ca"], ["ex_date", "record_date"])
    out = pd.concat([old, new], ignore_index=True).dropna(subset=["ex_date"]).reindex(columns=cols)
    out = out.drop_duplicates(["symbol", "ex_date", "subject"], keep="last").sort_values(
        ["ex_date", "symbol", "kind", "subject"], kind="mergesort"
    )
    os.makedirs(DIR, exist_ok=True)
    out.to_csv(F["ca"], index=False, date_format="%Y-%m-%d")
    return new


def fetch_surveillance(S: Session) -> pd.DataFrame:
    rows = []
    asm = S.json("/api/reportASM")
    for term, lab in (("longterm", "ASM long-term"), ("shortterm", "ASM short-term")):
        v = asm.get(term) if isinstance(asm, dict) else None
        for x in (v.get("data") or []) if isinstance(v, dict) else (v or []) if isinstance(v, list) else []:
            rows.append(dict(symbol=str(x.get("symbol", "")).strip(), list=lab, stage=str(x.get("asmSurvIndicator") or x.get("stage") or "")))
    gsm_ok = False
    try:
        gsm = S.json("/api/reportGSM")
        if isinstance(gsm, list):
            gsm_ok = True
            for x in gsm:
                rows.append(dict(symbol=str(x.get("symbol", "")).strip(), list="GSM", stage=str(x.get("gsmStage") or "")))
    except Exception as e:
        print(f"[nse] GSM: {e}")
    out = pd.DataFrame(rows, columns=["symbol", "list", "stage"]).drop_duplicates()
    if not gsm_ok:
        old = _read(F["surv"])
        if len(old) and "list" in old:
            old_gsm = old[old["list"] == "GSM"][["symbol", "list", "stage"]]
            out = pd.concat([out, old_gsm], ignore_index=True).drop_duplicates()
    out["as_of"] = datetime.now(IST).strftime("%Y-%m-%d")
    out = out.sort_values(["symbol", "list", "stage"], kind="mergesort")
    os.makedirs(DIR, exist_ok=True)
    out.to_csv(F["surv"], index=False)
    return out


def fetch_events(S: Session, today: pd.Timestamp) -> pd.DataFrame:
    frm, to = today.strftime("%d-%m-%Y"), (today + pd.Timedelta(days=45)).strftime("%d-%m-%Y")
    rows = []
    any_ok = False
    # Event-calendar carries clean "Financial Results" labels; read it before generic board-meetings.
    for path, sym, date, purpose in ((f"/api/event-calendar?index=equities&from_date={frm}&to_date={to}", "symbol", "date", "purpose"),
                                     (f"/api/corporate-board-meetings?index=equities&from_date={frm}&to_date={to}", "bm_symbol", "bm_date", "bm_purpose")):
        try:
            j = S.json(path)
            if isinstance(j, list):
                any_ok = True
            for x in j if isinstance(j, list) else []:
                p_, desc = str(x.get(purpose) or "").strip(), str(x.get("bm_desc") or "").strip()
                # "Board Meeting Intimation" says nothing; the description says what for ("...to consider financial results")
                if re.search(r"result", desc, re.I) and not re.search(r"result", p_, re.I):
                    p_ = "Financial Results"
                rows.append(dict(symbol=str(x.get(sym, "")).strip(), date=pd.to_datetime(x.get(date), errors="coerce", dayfirst=True),
                                 purpose=p_ or desc[:60]))
        except Exception as e:
            print(f"[nse] events {path.split('?')[0]}: {e}")
    if not any_ok and os.path.exists(F["events"]):
        return _read(F["events"], ["date"])
    out = pd.DataFrame(rows, columns=["symbol", "date", "purpose"]).dropna(subset=["date"])
    out = out[out.date >= today - pd.Timedelta(days=1)].drop_duplicates(["symbol", "date"]).sort_values(
        ["date", "symbol", "purpose"], kind="mergesort"
    )
    os.makedirs(DIR, exist_ok=True)
    out.to_csv(F["events"], index=False, date_format="%Y-%m-%d")
    return out


def fetch_deals(S: Session) -> pd.DataFrame:
    rows = []
    for kind, url in (("bulk", f"{ARCH}/content/equities/bulk.csv"), ("block", f"{ARCH}/content/equities/block.csv")):
        try:
            d = pd.read_csv(io.BytesIO(S.get(url).content))
            d.columns = [c.strip() for c in d.columns]
            px = next((c for c in d.columns if c.lower().startswith("trade price")), None)
            for x in d.to_dict("records"):
                rows.append(dict(date=pd.to_datetime(str(x.get("Date", "")).strip(), errors="coerce"), symbol=str(x.get("Symbol", "")).strip(),
                                 client=str(x.get("Client Name", "")).strip(), side=str(x.get("Buy/Sell", "")).strip().upper(),
                                 qty=pd.to_numeric(str(x.get("Quantity Traded", "")).replace(",", ""), errors="coerce"),
                                 price=pd.to_numeric(str(x.get(px, "")).replace(",", ""), errors="coerce") if px else np.nan, kind=kind))
        except Exception as e:
            print(f"[nse] {kind} deals: {e}")
    new = pd.DataFrame(rows, columns=["date", "symbol", "client", "side", "qty", "price", "kind"]).dropna(subset=["date"])
    old = _read(F["deals"], ["date"])
    out = pd.concat([old, new], ignore_index=True).drop_duplicates(["date", "symbol", "client", "side", "qty", "kind"])
    if len(out):
        out = out[out.date >= out.date.max() - pd.Timedelta(days=KEEP_DEAL_DAYS)].sort_values(
            ["date", "symbol", "kind", "side", "client", "qty"], kind="mergesort"
        )
    os.makedirs(DIR, exist_ok=True)
    out.to_csv(F["deals"], index=False, date_format="%Y-%m-%d")
    return new


def fetch_market_context(S: Session, day: pd.Timestamp) -> Dict:
    row = dict(date=day.strftime("%Y-%m-%d"))
    try:
        d = pd.read_csv(io.BytesIO(S.get(f"{ARCH}/content/indices/ind_close_all_{_d(day)}.csv").content))
        d.columns = [c.strip() for c in d.columns]
        n = d[d["Index Name"].astype(str).str.strip().str.lower() == "nifty 50"]
        if len(n):
            g = lambda c: pd.to_numeric(str(n.iloc[0].get(c, "")).strip(), errors="coerce")
            row.update(nifty_close=g("Closing Index Value"), nifty_pe=g("P/E"), nifty_pb=g("P/B"), nifty_dy=g("Div Yield"))
        m = d[d["Index Name"].astype(str).str.strip().str.lower() == "nifty midcap 150"]
        if len(m):
            row["midcap_pe"] = pd.to_numeric(str(m.iloc[0].get("P/E", "")).strip(), errors="coerce")
        s = d[d["Index Name"].astype(str).str.strip().str.lower() == "nifty smallcap 250"]
        if len(s):
            row["smallcap_pe"] = pd.to_numeric(str(s.iloc[0].get("P/E", "")).strip(), errors="coerce")
    except Exception as e:
        print(f"[nse] index closes: {e}")
    try:
        for x in S.json("/api/fiidiiTradeReact"):
            cat = str(x.get("category", "")).upper()
            key = "fii_net" if "FII" in cat or "FPI" in cat else "dii_net" if "DII" in cat else None
            if key:
                d_str = str(x.get("date", "")).strip()
                fii_dt = pd.to_datetime(d_str, errors="coerce")
                if pd.notna(fii_dt) and fii_dt.date() != day.date():
                    continue
                row[key] = pd.to_numeric(str(x.get("netValue", "")).replace(",", ""), errors="coerce")
                row[key.replace("net", "date")] = d_str
    except Exception as e:
        print(f"[nse] FII/DII: {e}")
    try:
        raw = S.get(f"{ARCH}/content/nsccl/fao_participant_oi_{_d(day)}.csv").content.decode("utf-8", "ignore")
        row.update(parse_participant_oi(raw))
    except Exception as e:
        print(f"[nse] participant OI: {e}")
    old = _read(F["ctx"])
    out = pd.concat([old[old.get("date", pd.Series(dtype=str)).astype(str) != row["date"]] if len(old) else old,
                     pd.DataFrame([row])], ignore_index=True).sort_values("date")
    os.makedirs(DIR, exist_ok=True)
    out.to_csv(F["ctx"], index=False)
    return row


def fetch_bank_npa(S: Session, symbol: str) -> List[Dict[str, Any]]:
    """
    Fetch quarterly bank asset-quality metrics (GNPA %, NNPA %, CET1/CRAR %) from NSE
    `/api/results-comparision?symbol=<SYM>` (verified reachable from GitHub Actions).
    Returns long-format rows: {ticker, field, period, value}.
    """
    sym = str(symbol or "").replace(".NS", "").strip().upper()
    if not sym:
        return []
    j = S.json(f"/api/results-comparision?symbol={quote(sym, safe='')}", tries=2, timeout=15)
    if not isinstance(j, dict):
        return []
    recs = j.get("resCmpData") or []
    is_bank = j.get("bankNonBnking") == "B"
    rows: List[Dict[str, Any]] = []
    field_map = (("re_per_grs_npa", "gnpa"), ("re_grs_npa_per", "nnpa"), ("re_cet_1_ret", "cet1"))
    for r in recs:
        if not isinstance(r, dict):
            continue
        has_npa = any(r.get(k) not in (None, "", "-") for k, _ in field_map[:2])
        if not is_bank and not has_npa:
            continue
        dt_raw = r.get("re_to_dt") or r.get("re_create_dt")
        dt = pd.to_datetime(str(dt_raw or "").strip(), format="%d-%b-%Y", errors="coerce")
        if pd.isna(dt):
            dt = pd.to_datetime(str(dt_raw or "").strip(), errors="coerce")
        if pd.isna(dt):
            continue
        period = (dt + pd.offsets.MonthEnd(0)).date()
        for src_k, fld in field_map:
            val = pd.to_numeric(str(r.get(src_k, "")).replace(",", "").strip(), errors="coerce")
            if pd.notna(val) and float(val) >= 0:
                rows.append({"ticker": f"{sym}.NS", "field": fld, "period": period, "value": float(val)})
    return rows


# ================================================================== reading (dashboard, engine)
@lru_cache(maxsize=16)
def _read_cached(path: str, mtime: float, dates: Tuple[str, ...]) -> pd.DataFrame:
    d = pd.read_csv(path)
    for c in dates:
        if c in d:
            d[c] = pd.to_datetime(d[c], errors="coerce")
    return d


def _read(path: str, dates: Optional[List[str]] = None) -> pd.DataFrame:
    try:
        return _read_cached(path, os.path.getmtime(path), tuple(dates or ())).copy()
    except Exception:
        return pd.DataFrame()


def _rename_aliases() -> Dict[str, str]:
    """old_symbol -> newest_symbol from data/nse/symbol_changes.csv (walking multi-hop chains)."""
    sc = _read(os.path.join(DIR, "symbol_changes.csv"))
    if sc.empty or "old" not in sc or "new" not in sc:
        return {}
    fwd = {str(r.old).strip(): str(r.new).strip() for r in sc.itertuples()
           if str(r.old).strip() and str(r.new).strip() and str(r.old).strip() != str(r.new).strip()}
    out: Dict[str, str] = {}
    for old_s in fwd:
        cur, seen = old_s, {old_s}
        while cur in fwd and fwd[cur] not in seen:
            cur = fwd[cur]
            seen.add(cur)
        if cur != old_s:
            out[old_s] = cur
    return out


def _delivery_pivot_from_parquets(n: int = 20) -> pd.DataFrame:
    from utils import nse_prices
    px_dir = nse_prices.PX
    fs = sorted(glob.glob(os.path.join(px_dir, "*.parquet")))[-2:] + sorted(glob.glob(os.path.join(px_dir, "d", "*.parquet")))
    if not fs:
        return pd.DataFrame()
    parts = []
    for f in fs:
        try:
            parts.append(pd.read_parquet(f, columns=["date", "symbol", "series", "deliv_pct"]))
        except Exception:
            continue
    if not parts:
        return pd.DataFrame()
    R = pd.concat(parts, ignore_index=True)
    R["date"] = pd.to_datetime(R["date"], errors="coerce")
    days = sorted(R["date"].dropna().unique())[-n:]
    d = R[R["date"].isin(days) & R["series"].astype(str).isin(["EQ", "BE", "BZ"])].copy()
    if d.empty:
        return pd.DataFrame()
    d["ticker"] = d["symbol"].astype(str) + ".NS"
    return d.pivot_table(index="date", columns="ticker", values="deliv_pct", aggfunc="last").sort_index()


def build_delivery_20d(n: int = 20, path: str = DELIV_20D) -> pd.DataFrame:
    """Persist the last n sessions delivery % pivot as one small zstd parquet for instant UI/engine reads."""
    piv = _delivery_pivot_from_parquets(n)
    if len(piv):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        piv.astype("float32").to_parquet(path, compression="zstd")
    return piv


def delivery_panel(n: int = 20) -> pd.DataFrame:
    """Last n sessions of delivery %: index date, columns SYMBOL.NS. Prefers prebuilt delivery_20d.parquet."""
    if n <= 20 and os.path.exists(DELIV_20D):
        try:
            piv = pd.read_parquet(DELIV_20D)
            piv.index = pd.to_datetime(piv.index, errors="coerce")
            piv = piv.sort_index()
            from utils import nse_prices
            last_px = nse_prices.last_date("stocks")
            if len(piv) and (last_px is None or piv.index.max().normalize() >= last_px.normalize()):
                piv = piv.tail(n).copy()
                for old_s, new_s in _rename_aliases().items():
                    ot, nt = f"{old_s}.NS", f"{new_s}.NS"
                    if ot not in piv.columns and nt in piv.columns:
                        piv[ot] = piv[nt]
                return piv
        except Exception:
            pass
    try:
        piv = _delivery_pivot_from_parquets(n)
        if len(piv):
            for old_s, new_s in _rename_aliases().items():
                ot, nt = f"{old_s}.NS", f"{new_s}.NS"
                if ot not in piv.columns and nt in piv.columns:
                    piv[ot] = piv[nt]
            return piv
    except Exception:
        pass
    files = sorted(glob.glob(os.path.join(DIR, "eod_*.csv.gz")))[-n:]
    parts = []
    for f in files:
        try:
            e = pd.read_csv(f, usecols=["date", "symbol", "series", "deliv_pct"])
            parts.append(e[e.series.isin(["EQ", "BE", "BZ"])])
        except Exception:
            continue
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts)
    d["ticker"] = d.symbol.astype(str) + ".NS"
    return d.pivot_table(index="date", columns="ticker", values="deliv_pct", aggfunc="last").sort_index()


def surveillance_map() -> Dict[str, str]:
    s = _read(F["surv"])
    out: Dict[str, str] = {}
    for x in s.to_dict("records") if len(s) else []:
        lab = x["list"] + (f" {x['stage']}" if str(x.get("stage") or "").strip() not in ("", "nan") else "")
        t = f"{x['symbol']}.NS"
        out[t] = out[t] + " · " + lab if t in out else lab
    for old_s, new_s in _rename_aliases().items():
        ot, nt = f"{old_s}.NS", f"{new_s}.NS"
        if ot not in out and nt in out:
            out[ot] = out[nt]
    return out


def upcoming_events(days: int = 30, today: Optional[pd.Timestamp] = None) -> Dict[str, str]:
    """ticker -> the next event within `days`: results / board meeting, or a split / bonus / dividend / rights / demerger ex-date."""
    today = pd.Timestamp(today or datetime.now(IST).date())
    out: Dict[str, Tuple[pd.Timestamp, int, str]] = {}

    def put(t, d, txt, prio=0):
        if pd.notna(d) and today <= d <= today + pd.Timedelta(days=days):
            cur = out.get(t)
            if cur is None or d < cur[0] or (d == cur[0] and prio > cur[1]):
                out[t] = (d, prio, txt)
    ev = _read(F["events"], ["date"])
    for x in ev.to_dict("records") if len(ev) else []:
        p = str(x.get("purpose") or "")
        is_res = bool(re.search(r"result", p, re.I))
        txt = "results" if is_res else ("board meeting" if re.search(r"board meeting", p, re.I) or not p else p[:28])
        put(f"{x['symbol']}.NS", x["date"], txt, prio=2 if is_res else 0 if txt == "board meeting" else 1)
    ca = _read(F["ca"], ["ex_date"])
    for x in ca.to_dict("records") if len(ca) else []:
        if x.get("kind") in ("split", "bonus", "consolidation", "dividend", "rights", "demerger"):
            put(f"{x['symbol']}.NS", x["ex_date"], f"ex-{x['kind']}", prio=3)
    res = {t: f"{txt} {d:%d %b}" + ("" if d > today else " (today)") for t, (d, _, txt) in out.items()}
    for old_s, new_s in _rename_aliases().items():
        ot, nt = f"{old_s}.NS", f"{new_s}.NS"
        if ot not in res and nt in res:
            res[ot] = res[nt]
    return res


def official_factors() -> pd.DataFrame:
    """Splits / bonuses / consolidations NSE announced: ticker, date (ex-date), factor."""
    ca = _read(F["ca"], ["ex_date"])
    if ca.empty or "factor" not in ca:
        return pd.DataFrame(columns=["ticker", "date", "factor"])
    c = ca[ca.kind.isin(["split", "bonus", "consolidation"]) & pd.to_numeric(ca.factor, errors="coerce").gt(0)]
    return pd.DataFrame(dict(ticker=c.symbol.astype(str) + ".NS", date=c.ex_date, factor=pd.to_numeric(c.factor)))


def deals_for(tickers=None, days: int = 5) -> pd.DataFrame:
    d = _read(F["deals"], ["date"])
    if d.empty:
        return d
    d = d[d.date >= d.date.max() - pd.Timedelta(days=days)]
    d["ticker"] = d.symbol.astype(str) + ".NS"
    if tickers is not None:
        want = set(tickers)
        aliases = _rename_aliases()
        for t in list(want):
            base = str(t).replace(".NS", "")
            if base in aliases:
                want.add(f"{aliases[base]}.NS")
        d = d[d.ticker.isin(want)]
    d["value_cr"] = d.qty * d.price / 1e7
    return d.sort_values(["date", "value_cr"], ascending=[False, False])


def market_context() -> pd.DataFrame:
    return _read(F["ctx"])


CTX_INDICES = {"nifty 50": "nifty", "nifty midcap 150": "midcap", "nifty smallcap 250": "smallcap"}


def context_history() -> pd.DataFrame:
    """
    Daily valuation and flows for the Daily Brief charts: Nifty / Midcap 150 / Smallcap 250 close and P/E
    (+ Nifty P/B and yield) from the NSE index store (2 years), joined with FII / DII cash and FII index-futures
    positioning from market_context.csv (cash from 20 Nov 2026; positioning back-filled from NSE's participant files).
    """
    out = pd.DataFrame()
    try:
        from utils import nse_prices as npx
        X = npx.raw("indices")
        if len(X):
            X = X[X["index"].astype(str).str.lower().isin(CTX_INDICES)].copy()
            X["k"] = X["index"].astype(str).str.lower().map(CTX_INDICES)
            parts = []
            for col in ("close", "pe", "pb", "dy"):
                w = X.pivot_table(index="date", columns="k", values=col, aggfunc="last")
                w.columns = [f"{c}_{col}" for c in w.columns]
                parts.append(w)
            out = pd.concat(parts, axis=1)
            out = out[[c for c in out.columns if not (c.endswith("_pb") or c.endswith("_dy")) or c.startswith("nifty")]]
    except Exception:
        out = pd.DataFrame()
    C = market_context()
    if len(C):
        C = C.copy()
        C["date"] = pd.to_datetime(C["date"], errors="coerce")
        C = C.dropna(subset=["date"]).drop_duplicates("date", keep="last").set_index("date")
        flows = C[[c for c in ("fii_net", "dii_net", "fii_idx_long", "fii_idx_short", "fii_idx_long_pct") if c in C.columns]]
        out = flows if out.empty else out.join(flows, how="outer")
        for k in ("nifty_pe", "nifty_pb", "nifty_dy", "midcap_pe", "smallcap_pe"):   # fill a day the store lacks
            if k in C.columns:
                out[k] = out[k].combine_first(C[k]) if k in out.columns else C[k]
    if len(out):
        out.index = pd.DatetimeIndex(out.index, name="date")
        out = out.sort_index()
    return out


def parse_participant_oi(raw: str) -> Dict[str, float]:
    """FII long / short index-futures contracts from NSE's fao_participant_oi file."""
    d = pd.read_csv(io.StringIO(raw), skiprows=1)
    d.columns = [c.strip() for c in d.columns]
    fii = d[d.iloc[:, 0].astype(str).str.strip().str.upper() == "FII"]
    if fii.empty:
        return {}
    lo = pd.to_numeric(fii.iloc[0].get("Future Index Long"), errors="coerce")
    sh = pd.to_numeric(fii.iloc[0].get("Future Index Short"), errors="coerce")
    return dict(fii_idx_long=lo, fii_idx_short=sh, fii_idx_long_pct=100 * lo / (lo + sh) if lo + sh else np.nan)


def attach(df: pd.DataFrame) -> pd.DataFrame:
    """Adds deliv_pct (today), deliv_avg20, nse_flags (surveillance + next event) to a snapshot frame. Never raises."""
    try:
        out = df.copy()
        P = delivery_panel(20)
        if len(P):
            out["deliv_pct"] = out["ticker"].map(P.iloc[-1])
            out["deliv_avg20"] = out["ticker"].map(P.mean())
            out["deliv_date"] = str(P.index[-1])[:10]
        sv, ev = surveillance_map(), upcoming_events(30)
        out["surveillance"] = out["ticker"].map(sv).fillna("")
        out["next_event"] = out["ticker"].map(ev).fillna("")
        out["nse_flags"] = [" · ".join(x for x in (f"⚠️ {s}" if s else "", f"📅 {e}" if e else "") if x)
                            for s, e in zip(out["surveillance"], out["next_event"])]
        return out
    except Exception as e:
        print(f"[nse] attach skipped: {e}", flush=True)
        return df
