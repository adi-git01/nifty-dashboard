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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

IST = timezone(timedelta(hours=5, minutes=30))
ARCH = "https://nsearchives.nseindia.com"
WWW = "https://www.nseindia.com"
DIR = "data/nse"
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
    for x in got:
        kind, fac = action_factor(x.get("subject"))
        rows.append(dict(symbol=str(x.get("symbol", "")).strip(), ex_date=pd.to_datetime(x.get("exDate"), errors="coerce"),
                         record_date=pd.to_datetime(x.get("recDate"), errors="coerce"), subject=str(x.get("subject", "")).strip(),
                         kind=kind, factor=fac))
    new = pd.DataFrame(rows, columns=["symbol", "ex_date", "record_date", "subject", "kind", "factor"])
    old = _read(F["ca"], ["ex_date", "record_date"])
    out = pd.concat([old, new], ignore_index=True).dropna(subset=["ex_date"])
    out = out.drop_duplicates(["symbol", "ex_date", "subject"], keep="last").sort_values("ex_date")
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
    try:
        gsm = S.json("/api/reportGSM")
        for x in gsm if isinstance(gsm, list) else []:
            rows.append(dict(symbol=str(x.get("symbol", "")).strip(), list="GSM", stage=str(x.get("gsmStage") or "")))
    except Exception as e:
        print(f"[nse] GSM: {e}")
    out = pd.DataFrame(rows, columns=["symbol", "list", "stage"]).drop_duplicates()
    out["as_of"] = datetime.now(IST).strftime("%Y-%m-%d")
    os.makedirs(DIR, exist_ok=True)
    out.to_csv(F["surv"], index=False)
    return out


def fetch_events(S: Session, today: pd.Timestamp) -> pd.DataFrame:
    frm, to = today.strftime("%d-%m-%Y"), (today + pd.Timedelta(days=45)).strftime("%d-%m-%Y")
    rows = []
    for path, sym, date, purpose in ((f"/api/corporate-board-meetings?index=equities&from_date={frm}&to_date={to}", "bm_symbol", "bm_date", "bm_purpose"),
                                     (f"/api/event-calendar?index=equities&from_date={frm}&to_date={to}", "symbol", "date", "purpose")):
        try:
            j = S.json(path)
            for x in j if isinstance(j, list) else []:
                p_, desc = str(x.get(purpose) or "").strip(), str(x.get("bm_desc") or "").strip()
                # "Board Meeting Intimation" says nothing; the description says what for ("...to consider financial results")
                if re.search(r"result", desc, re.I) and not re.search(r"result", p_, re.I):
                    p_ = "Financial Results"
                rows.append(dict(symbol=str(x.get(sym, "")).strip(), date=pd.to_datetime(x.get(date), errors="coerce", dayfirst=True),
                                 purpose=p_ or desc[:60]))
        except Exception as e:
            print(f"[nse] events {path.split('?')[0]}: {e}")
    out = pd.DataFrame(rows, columns=["symbol", "date", "purpose"]).dropna(subset=["date"])
    out = out[out.date >= today - pd.Timedelta(days=1)].drop_duplicates(["symbol", "date"]).sort_values("date")
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
        out = out[out.date >= out.date.max() - pd.Timedelta(days=KEEP_DEAL_DAYS)].sort_values("date")
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
                row[key] = pd.to_numeric(str(x.get("netValue", "")).replace(",", ""), errors="coerce")
                row[key.replace("net", "date")] = str(x.get("date", ""))
    except Exception as e:
        print(f"[nse] FII/DII: {e}")
    try:
        raw = S.get(f"{ARCH}/content/nsccl/fao_participant_oi_{_d(day)}.csv").content.decode("utf-8", "ignore")
        d = pd.read_csv(io.StringIO(raw), skiprows=1)
        d.columns = [c.strip() for c in d.columns]
        fii = d[d.iloc[:, 0].astype(str).str.strip().str.upper() == "FII"]
        if len(fii):
            lo = pd.to_numeric(fii.iloc[0].get("Future Index Long"), errors="coerce")
            sh = pd.to_numeric(fii.iloc[0].get("Future Index Short"), errors="coerce")
            row.update(fii_idx_long=lo, fii_idx_short=sh, fii_idx_long_pct=100 * lo / (lo + sh) if lo + sh else np.nan)
    except Exception as e:
        print(f"[nse] participant OI: {e}")
    old = _read(F["ctx"])
    out = pd.concat([old[old.get("date", pd.Series(dtype=str)).astype(str) != row["date"]] if len(old) else old,
                     pd.DataFrame([row])], ignore_index=True).sort_values("date")
    os.makedirs(DIR, exist_ok=True)
    out.to_csv(F["ctx"], index=False)
    return row


# ================================================================== reading (dashboard, engine)
def _read(path: str, dates: Optional[List[str]] = None) -> pd.DataFrame:
    try:
        d = pd.read_csv(path)
        for c in dates or []:
            if c in d:
                d[c] = pd.to_datetime(d[c], errors="coerce")
        return d
    except Exception:
        return pd.DataFrame()


def delivery_panel(n: int = 20) -> pd.DataFrame:
    """Last n sessions of delivery %: index date, columns SYMBOL.NS. From the price store (utils/nse_prices.py)."""
    try:
        from utils import nse_prices
        R = nse_prices.raw("stocks")
        if len(R):
            days = sorted(R["date"].unique())[-n:]
            d = R[R["date"].isin(days) & R["series"].isin(["EQ", "BE", "BZ"])]
            return d.assign(ticker=d["symbol"] + ".NS").pivot_table(index="date", columns="ticker", values="deliv_pct",
                                                                    aggfunc="last").sort_index()
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
    return out


def upcoming_events(days: int = 30, today: Optional[pd.Timestamp] = None) -> Dict[str, str]:
    """ticker -> the next event within `days`: results / board meeting, or a split / bonus / dividend ex-date."""
    today = pd.Timestamp(today or datetime.now(IST).date())
    out: Dict[str, Tuple[pd.Timestamp, str]] = {}

    def put(t, d, txt):
        if pd.notna(d) and today <= d <= today + pd.Timedelta(days=days) and (t not in out or d < out[t][0]):
            out[t] = (d, txt)
    ev = _read(F["events"], ["date"])
    for x in ev.to_dict("records") if len(ev) else []:
        p = str(x.get("purpose") or "")
        put(f"{x['symbol']}.NS", x["date"], "results" if re.search(r"result", p, re.I) else
            "board meeting" if re.search(r"board meeting", p, re.I) or not p else p[:28])
    ca = _read(F["ca"], ["ex_date"])
    for x in ca.to_dict("records") if len(ca) else []:
        if x.get("kind") in ("split", "bonus", "consolidation", "dividend", "demerger"):
            put(f"{x['symbol']}.NS", x["ex_date"], f"ex-{x['kind']}")
    return {t: f"{txt} {d:%d %b}" + ("" if d > today else " (today)") for t, (d, txt) in out.items()}


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
        d = d[d.ticker.isin(set(tickers))]
    d["value_cr"] = d.qty * d.price / 1e7
    return d.sort_values(["date", "value_cr"], ascending=[False, False])


def market_context() -> pd.DataFrame:
    return _read(F["ctx"])


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
