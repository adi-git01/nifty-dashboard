"""
What NSE publishes for free, and whether a GitHub Actions runner can reach it.

For the latest trading day it tries NSE's daily archive files (no login) and the JSON endpoints behind
nseindia.com (a session cookie from the home page, no account), and writes:

    analysis/NSE_DATA_PROBE.md          per source: reachable?, rows, columns, a sample, what it is good for
    data/nse/eod_YYYYMMDD.csv.gz        the day's official EOD for every EQ stock: open/high/low/close/prev close,
                                        volume, turnover, trades, delivery qty and delivery %
    + a comparison of NSE's closes with the dashboard's Yahoo snapshot for the same session

    python nse_probe.py [--date YYYY-MM-DD]

Nothing here needs a broker account. NSE data is for personal use (no redistribution).
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import time
import zipfile
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

IST = timezone(timedelta(hours=5, minutes=30))
ARCH = "https://nsearchives.nseindia.com"
WWW = "https://www.nseindia.com"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
H = {"User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9", "Referer": WWW + "/"}
OUT_MD = "analysis/NSE_DATA_PROBE.md"

# (name, url template, kind, what it is for). {d}=DDMMYYYY {ymd}=YYYYMMDD {mon}=JAN {y}=YYYY
ARCHIVE = [
    ("CM bhavcopy (UDiFF)", ARCH + "/content/cm/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip", "zip",
     "official OHLC, volume, turnover, trades for every security"),
    ("Security-wise delivery (sec_bhavdata_full)", ARCH + "/products/content/sec_bhavdata_full_{d}.csv", "csv",
     "OHLC + DELIVERY quantity and %: real buying vs intraday churn"),
    ("MTO delivery file", ARCH + "/archives/equities/mto/MTO_{d}.DAT", "dat", "delivery position (older format)"),
    ("52-week high/low", ARCH + "/content/CM_52_wk_High_low_{d}.csv", "csv", "official 52W high/low per stock"),
    ("Index closes (all indices)", ARCH + "/content/indices/ind_close_all_{d}.csv", "csv",
     "close, P/E, P/B, dividend yield of every NSE index (sector indices too)"),
    ("Bulk deals (today)", ARCH + "/content/equities/bulk.csv", "csv", "large disclosed deals: who bought/sold"),
    ("Block deals (today)", ARCH + "/content/equities/block.csv", "csv", "block-window deals"),
    ("Short selling", ARCH + "/archives/equities/shortSelling/shortselling_{d}.csv", "csv", "shares sold short per stock"),
    ("Price band changes", ARCH + "/content/equities/eq_band_changes_{d}.csv", "csv", "circuit band changes"),
    ("Daily volatility (CMVOLT)", ARCH + "/archives/nsccl/volt/CMVOLT_{d}.CSV", "csv", "daily and annualised volatility"),
    ("F&O ban list", ARCH + "/content/fo/fo_secban_{d}.csv", "csv", "stocks in the F&O ban period"),
    ("F&O participant OI", ARCH + "/content/nsccl/fao_participant_oi_{d}.csv", "csv",
     "FII / DII / Pro / Client positions in index & stock futures/options"),
    ("F&O bhavcopy (UDiFF)", ARCH + "/content/fo/BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.zip", "zip",
     "futures/options OHLC and open interest"),
]
API = [
    ("FII / DII cash activity", "/api/fiidiiTradeReact", "net buying by FIIs and DIIs in the cash market"),
    ("Corporate actions (upcoming)", "/api/corporates-corporateActions?index=equities", "splits, bonuses, dividends with ex-dates"),
    ("Board meetings / results calendar", "/api/corporate-board-meetings?index=equities", "results dates ahead"),
    ("Event calendar", "/api/event-calendar", "results, AGMs, dividends"),
    ("Insider trading (PIT)", "/api/corporates-pit?index=equities", "promoter / insider buys and sells"),
    ("Pledged shares", "/api/corporate-pledgedata?index=equities", "promoter pledge levels"),
    ("ASM list (surveillance)", "/api/reportASM", "stocks under additional surveillance"),
    ("GSM list (surveillance)", "/api/reportGSM", "graded surveillance"),
    ("Announcements", "/api/corporate-announcements?index=equities", "filings: orders, results, management changes"),
    ("Market status", "/api/marketStatus", "is the market open, last session"),
    ("Quote: RELIANCE", "/api/quote-equity?symbol=RELIANCE", "P/E, sector P/E, shares issued, industry classification"),
    ("Results comparison: RELIANCE", "/api/results-comparision?symbol=RELIANCE", "last quarters' P&L lines: revenue, costs, profit, EPS"),
    ("Financial results list", "/api/corporates-financial-results?index=equities&period=Quarterly", "latest quarterly results filed"),
    ("Shareholding: RELIANCE", "/api/corporate-share-holdings-master?index=equities&symbol=RELIANCE", "promoter / public holding by quarter"),
    ("Nifty 50 constituents live", "/api/equity-stockIndices?index=NIFTY%2050", "live prices of an index's stocks"),
]


def fmt(day: pd.Timestamp) -> dict:
    return dict(d=day.strftime("%d%m%Y"), ymd=day.strftime("%Y%m%d"), mon=day.strftime("%b").upper(), y=day.strftime("%Y"))


def get(s: requests.Session, url: str, tries: int = 2):
    last = None
    for k in range(tries):
        try:
            r = s.get(url, headers=H, timeout=25)
            if r.status_code == 200 and r.content:
                return r, None
            last = f"HTTP {r.status_code}"
        except Exception as e:
            last = type(e).__name__
        time.sleep(1.5 * (k + 1))
    return None, last


def to_frame(r, kind: str) -> pd.DataFrame:
    raw = r.content
    if kind == "zip":
        z = zipfile.ZipFile(io.BytesIO(raw))
        raw = z.read(z.namelist()[0])
    if kind == "dat":
        return pd.read_csv(io.BytesIO(raw), skiprows=3, header=None)
    return pd.read_csv(io.BytesIO(raw))


def latest_day(s: requests.Session, start: pd.Timestamp):
    """The most recent day (from `start` back) with a delivery file: a trading day whose files are out."""
    for k in range(8):
        day = start - pd.Timedelta(days=k)
        if day.weekday() >= 5:
            continue
        r, _ = get(s, ARCHIVE[1][1].format(**fmt(day)), tries=1)
        if r is not None:
            return day
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="")
    a = ap.parse_args()
    s = requests.Session()
    start = pd.Timestamp(a.date) if a.date else pd.Timestamp(datetime.now(IST).date())
    try:
        s.get(WWW, headers=H, timeout=20)           # sets the cookies the JSON endpoints want
    except Exception as e:
        print(f"[nse] home page: {type(e).__name__}")
    day = latest_day(s, start) or start
    f = fmt(day)
    print(f"[nse] session {day:%a %d %b %Y}", flush=True)

    L = [f"# What NSE gives for free -- probe of {day:%d %b %Y} from GitHub Actions", "",
         f"Run at {datetime.now(IST):%d %b %Y %H:%M} IST. ✅ = reachable from a GitHub runner; ❌ = refused or missing.", "",
         "## Daily archive files (no login)", "", "| Source | Status | Rows | Columns | Use |", "|---|---|---|---|---|"]
    frames, samples = {}, []
    for name, tpl, kind, use in ARCHIVE:
        url = tpl.format(**f)
        r, err = get(s, url)
        if r is None:
            L.append(f"| {name} | ❌ {err} | | | {use} |")
            continue
        try:
            d = to_frame(r, kind)
            d.columns = [str(c).strip() for c in d.columns]
            frames[name] = d
            cols = ", ".join(map(str, d.columns[:14])) + (" …" if len(d.columns) > 14 else "")
            L.append(f"| {name} | ✅ | {len(d):,} | {cols} | {use} |")
            samples.append((name, d.head(3)))
        except Exception as e:
            L.append(f"| {name} | ⚠️ got {len(r.content):,} bytes, unreadable ({type(e).__name__}) | | | {use} |")
        time.sleep(0.6)

    L += ["", "## JSON endpoints behind nseindia.com (session cookie, no account)", "",
          "| Source | Status | Records | Fields | Use |", "|---|---|---|---|---|"]
    for name, path, use in API:
        r, err = get(s, WWW + path)
        if r is None:
            L.append(f"| {name} | ❌ {err} | | | {use} |")
            continue
        try:
            j = r.json()
            rec = j if isinstance(j, list) else next((v for v in j.values() if isinstance(v, list)), [j]) if isinstance(j, dict) else []
            keys = ", ".join(list(rec[0].keys())[:12]) if rec and isinstance(rec[0], dict) else ""
            L.append(f"| {name} | ✅ | {len(rec):,} | {keys} | {use} |")
            if rec and isinstance(rec[0], dict):
                samples.append((name, pd.DataFrame(rec[:3])))
        except Exception as e:
            L.append(f"| {name} | ⚠️ not JSON ({type(e).__name__}; {len(r.content):,} bytes) | | | {use} |")
        time.sleep(0.8)

    # the day's EOD file: delivery file (has OHLC + delivery) for EQ-series stocks
    dv = frames.get("Security-wise delivery (sec_bhavdata_full)")
    if dv is not None:
        dv = dv.rename(columns=lambda c: c.strip())
        dv = dv[dv["SERIES"].astype(str).str.strip().isin(["EQ", "BE", "BZ", "SM", "ST"])].copy()
        keep = {"SYMBOL": "symbol", "SERIES": "series", "PREV_CLOSE": "prev_close", "OPEN_PRICE": "open", "HIGH_PRICE": "high",
                "LOW_PRICE": "low", "CLOSE_PRICE": "close", "LAST_PRICE": "last", "AVG_PRICE": "vwap", "TTL_TRD_QNTY": "volume",
                "TURNOVER_LACS": "turnover_lacs", "NO_OF_TRADES": "trades", "DELIV_QTY": "deliv_qty", "DELIV_PER": "deliv_pct"}
        eod = dv[[c for c in keep if c in dv.columns]].rename(columns=keep)
        for c in eod.columns:
            if c not in ("symbol", "series"):
                eod[c] = pd.to_numeric(eod[c].astype(str).str.strip().replace({"-": None}), errors="coerce")
        eod["symbol"] = eod["symbol"].astype(str).str.strip()
        eod["series"] = eod["series"].astype(str).str.strip()
        eod.insert(0, "date", day.date())
        os.makedirs("data/nse", exist_ok=True)
        path = f"data/nse/eod_{f['ymd']}.csv.gz"
        eod.to_csv(path, index=False)
        L += ["", f"## Saved: `{path}` -- {len(eod):,} stocks (EQ/BE/BZ/SME series)", "",
              f"Delivery %: median {eod.deliv_pct.median():.0f}% · stocks with ≥ 60% delivery: {(eod.deliv_pct >= 60).sum():,}"]
        # NSE's official close vs the dashboard's Yahoo snapshot for the same session
        snap = sorted(glob.glob(f"data/cache/market_master_{day:%Y_%m_%d}.parquet"))
        if snap:
            m = pd.read_parquet(snap[-1])
            px = "price" if "price" in m else "currentPrice"
            m["symbol"] = m["ticker"].astype(str).str.replace(".NS", "", regex=False)
            j = m[["symbol", px]].merge(eod[eod.series == "EQ"][["symbol", "close"]], on="symbol", how="left")
            j["diff_pct"] = (j[px] / j["close"] - 1) * 100
            got = j.dropna(subset=["close"])
            off = got[got.diff_pct.abs() > 0.5].sort_values("diff_pct", key=abs, ascending=False)
            L += ["", f"## Yahoo snapshot vs NSE official close ({day:%d %b})", "",
                  f"- {len(got):,} of {len(m):,} snapshot stocks matched; median gap {got.diff_pct.abs().median():.3f}%",
                  f"- within 0.1%: {(got.diff_pct.abs() <= 0.1).mean():.0%} · off by more than 0.5%: {len(off)} "
                  f"· not in NSE's EQ list: {j.close.isna().sum()}"]
            if len(off):
                L += ["", "| Stock | Yahoo | NSE close | Gap |", "|---|---|---|---|"]
                L += [f"| {r.symbol} | {r[px]:,.2f} | {r.close:,.2f} | {r.diff_pct:+.2f}% |" for r in off.head(15).itertuples()]
        else:
            L += ["", f"(no dashboard snapshot for {day:%d %b} in this checkout to compare with)"]

    L += ["", "## Samples", ""]
    for name, d in samples:
        L += [f"**{name}**", "", "```", d.to_string(max_cols=14, max_colwidth=24)[:1800], "```", ""]
    os.makedirs("analysis", exist_ok=True)
    open(OUT_MD, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
