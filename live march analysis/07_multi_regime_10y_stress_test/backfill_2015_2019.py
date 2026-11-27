"""Backfill 2015-04-01 .. 2019-09-30 survivorship-free NSE daily OHLCV + Delivery % + Indices + Corp Actions.

Seeds an isolated 11.5-year store at scratch/multi_regime_10y/nse_11y/data/nse/ by copying
repo_nov21's 2019-10 .. 2026-11 store (Kite-verified demerger bounds) and prepending 2015-04 .. 2019-09
from NSE official historical archives (cm_bhav + MTO delivery + ind_close_all + corporate actions).
"""
from __future__ import annotations

import glob
import io
import json
import os
import shutil
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

REPO_N = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\repo_nov21"
OUT_ROOT = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\multi_regime_10y\nse_11y"
SRC_NSE = os.path.join(REPO_N, "data", "nse")
DST_NSE = os.path.join(OUT_ROOT, "data", "nse")

sys.path.insert(0, REPO_N)
from utils import nse_data as nd  # noqa: E402
from utils import nse_prices as npx  # noqa: E402

MONTH_ABBR = ["", "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
SPECIAL_WEEKEND = {"2015-11-11", "2016-10-30", "2017-10-19", "2018-11-07"}

_tls = threading.local()


def _sess() -> requests.Session:
    s = getattr(_tls, "sess", None)
    if s is None:
        s = requests.Session()
        s.headers.update(nd.HDR)
        ad = HTTPAdapter(pool_connections=16, pool_maxsize=16)
        s.mount("https://", ad)
        s.mount("http://", ad)
        _tls.sess = s
    return s


def _get(url: str, tries: int = 3, timeout: float = 15.0) -> Tuple[int, bytes]:
    s = _sess()
    last = 0
    for k in range(tries):
        try:
            r = s.get(url, timeout=timeout)
            last = r.status_code
            if r.status_code == 200 and r.content:
                return 200, r.content
            if r.status_code == 404:
                return 404, b""
        except Exception:
            last = -1
        if k + 1 < tries:
            time.sleep(0.4 * (2 ** k))
    return last, b""


def seed_store() -> None:
    os.makedirs(DST_NSE, exist_ok=True)
    for sub in ("prices", "indices"):
        src_sub = os.path.join(SRC_NSE, sub)
        dst_sub = os.path.join(DST_NSE, sub)
        if os.path.exists(src_sub) and not os.path.exists(dst_sub):
            shutil.copytree(src_sub, dst_sub)
    for fn in ("corp_actions.csv", "market_context.csv", "symbol_changes.csv", "holidays.csv"):
        sp = os.path.join(SRC_NSE, fn)
        dp = os.path.join(DST_NSE, fn)
        if os.path.exists(sp) and not os.path.exists(dp):
            shutil.copy2(sp, dp)
    eq_src = os.path.join(REPO_N, "EQUITY_L.csv")
    eq_dst = os.path.join(OUT_ROOT, "EQUITY_L.csv")
    if os.path.exists(eq_src) and not os.path.exists(eq_dst):
        shutil.copy2(eq_src, eq_dst)


def parse_mto(raw_bytes: bytes) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for line in raw_bytes.decode("latin1", errors="ignore").splitlines():
        if not line.startswith("20,"):
            continue
        parts = line.split(",")
        if len(parts) < 7:
            continue
        sym = parts[2].strip()
        ser = parts[3].strip()
        try:
            dp = float(parts[6].strip())
            out[(sym, ser)] = dp
        except Exception:
            pass
    return out


def parse_cm_bhav(raw_zip: bytes, day: pd.Timestamp, mto_map: Dict[Tuple[str, str], float]) -> Optional[pd.DataFrame]:
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    df.columns = [c.strip().upper() for c in df.columns]
    if "SYMBOL" not in df.columns or "CLOSE" not in df.columns:
        return None
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
    df["SERIES"] = df["SERIES"].astype(str).str.strip() if "SERIES" in df.columns else "EQ"
    # Keep equity-like series matching repo schema (EQ, BE, BZ, SM, ST, BL, IL)
    keep = df["SERIES"].isin({"EQ", "BE", "BZ", "SM", "ST", "E1"})
    df = df[keep].copy()
    if df.empty:
        return None
    out = pd.DataFrame({
        "date": pd.Timestamp(day.normalize()),
        "symbol": df["SYMBOL"].values,
        "series": df["SERIES"].values,
        "open": pd.to_numeric(df["OPEN"], errors="coerce").astype("float64").values,
        "high": pd.to_numeric(df["HIGH"], errors="coerce").astype("float64").values,
        "low": pd.to_numeric(df["LOW"], errors="coerce").astype("float64").values,
        "close": pd.to_numeric(df["CLOSE"], errors="coerce").astype("float64").values,
        "prev_close": pd.to_numeric(df["PREVCLOSE"], errors="coerce").astype("float64").values,
        "volume": pd.to_numeric(df["TOTTRDQTY"], errors="coerce").fillna(0).astype("int64").values,
    })
    dp_vals = [mto_map.get((s, r), np.nan) for s, r in zip(out["symbol"], out["series"])]
    out["deliv_pct"] = pd.Series(dp_vals, dtype="float32").values
    out = out.dropna(subset=["close"]).drop_duplicates(["date", "symbol", "series"], keep="last")
    return out


def parse_ind_legacy(raw_bytes: bytes, day: pd.Timestamp) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception:
        return None
    df.columns = [c.strip() for c in df.columns]
    if "Index Name" not in df.columns or "Closing Index Value" not in df.columns:
        return None
    num = lambda c: pd.to_numeric(df[c].replace("-", np.nan), errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)
    out = pd.DataFrame({
        "date": pd.Timestamp(day.normalize()),
        "index": df["Index Name"].astype(str).str.strip().values,
        "open": num("Open Index Value").astype("float64").values,
        "high": num("High Index Value").astype("float64").values,
        "low": num("Low Index Value").astype("float64").values,
        "close": num("Closing Index Value").astype("float64").values,
        "volume": num("Volume").astype("float64").values,
        "pe": num("P/E").astype("float32").values,
        "pb": num("P/B").astype("float32").values,
        "dy": num("Div Yield").astype("float32").values,
    })
    out = out.dropna(subset=["close"]).drop_duplicates(["date", "index"], keep="last")
    return out


def fetch_one_day(day: pd.Timestamp) -> Dict:
    y = day.year
    mon = MONTH_ABBR[day.month]
    dd = day.strftime("%d")
    ddmmyyyy = day.strftime("%d%m%Y")
    cm_url = f"{nd.ARCH}/content/historical/EQUITIES/{y}/{mon}/cm{dd}{mon}{y}bhav.csv.zip"
    code, cm_bytes = _get(cm_url, tries=3, timeout=15.0)
    if code == 404:
        return dict(day=day, status="holiday_404")
    if code != 200:
        return dict(day=day, status=f"retry_cm_{code}")

    mto_url = f"{nd.ARCH}/archives/equities/mto/MTO_{ddmmyyyy}.DAT"
    mc, mto_bytes = _get(mto_url, tries=2, timeout=12.0)
    mto_map = parse_mto(mto_bytes) if mc == 200 else {}

    try:
        st_df = parse_cm_bhav(cm_bytes, day, mto_map)
    except Exception as e:
        return dict(day=day, status=f"retry_parse_{type(e).__name__}")
    if st_df is None or st_df.empty:
        return dict(day=day, status="holiday_empty")

    ind_url = f"{nd.ARCH}/content/indices/ind_close_all_{ddmmyyyy}.csv"
    ic, ind_bytes = _get(ind_url, tries=2, timeout=12.0)
    ix_df = parse_ind_legacy(ind_bytes, day) if ic == 200 else None
    return dict(day=day, status="ok", st=st_df, ix=ix_df)


def write_month(kind: str, ym: str, frames: List[pd.DataFrame]) -> None:
    sub = "prices" if kind == "stocks" else "indices"
    key = ["date", "symbol", "series"] if kind == "stocks" else ["date", "index"]
    out_p = os.path.join(DST_NSE, sub, f"{ym[:4]}_{ym[4:]}.parquet")
    df = pd.concat(frames, ignore_index=True)
    if os.path.exists(out_p):
        old = pd.read_parquet(out_p)
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(key, keep="last").sort_values(key).reset_index(drop=True)
    df.to_parquet(out_p, index=False)


def main() -> int:
    t0 = time.time()
    print("[backfill] seeding 2019-10..2026-11 store into nse_11y...", flush=True)
    seed_store()
    os.chdir(OUT_ROOT)

    # Determine which months in 2015-04 .. 2019-09 are missing
    all_days = list(pd.bdate_range("2015-04-01", "2019-09-30"))
    for sp in SPECIAL_WEEKEND:
        ts = pd.Timestamp(sp)
        if ts not in all_days:
            all_days.append(ts)
    all_days = sorted(all_days)

    by_ym: Dict[str, List[pd.Timestamp]] = {}
    for d in all_days:
        ym = d.strftime("%Y%m")
        if os.path.exists(os.path.join(DST_NSE, "prices", f"{ym[:4]}_{ym[4:]}.parquet")):
            continue
        by_ym.setdefault(ym, []).append(d)

    print(f"[backfill] fetching {len(by_ym)} missing months (2015-04 .. 2019-09)...", flush=True)
    tot_ok = tot_hol = 0
    retry_days: List[pd.Timestamp] = []

    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, (ym, days_m) in enumerate(sorted(by_ym.items()), 1):
            futs = {ex.submit(fetch_one_day, d): d for d in days_m}
            st_f, ix_f = [], []
            m_ok = m_hol = m_ret = 0
            for fut in as_completed(futs):
                res = fut.result()
                stt = res["status"]
                if stt == "ok":
                    m_ok += 1
                    st_f.append(res["st"])
                    if res["ix"] is not None:
                        ix_f.append(res["ix"])
                elif stt.startswith("holiday"):
                    m_hol += 1
                else:
                    m_ret += 1
                    retry_days.append(res["day"])
            if st_f:
                write_month("stocks", ym, st_f)
            if ix_f:
                write_month("indices", ym, ix_f)
            tot_ok += m_ok
            tot_hol += m_hol
            if i % 6 == 0 or i == len(by_ym) or m_ret > 0:
                print(f"[backfill] {i:2d}/{len(by_ym)} {ym}: ok={m_ok:2d} hol={m_hol:2d} ret={m_ret} | cum_ok={tot_ok:4d} [{time.time()-t0:.1f}s]", flush=True)

    still_failed = []
    if retry_days:
        print(f"[backfill] retrying {len(retry_days)} days...", flush=True)
        rst, rix = {}, {}
        for d in sorted(retry_days):
            time.sleep(0.25)
            res = fetch_one_day(d)
            if res["status"] == "ok":
                ym = d.strftime("%Y%m")
                rst.setdefault(ym, []).append(res["st"])
                if res["ix"] is not None:
                    rix.setdefault(ym, []).append(res["ix"])
                tot_ok += 1
            elif not res["status"].startswith("holiday"):
                still_failed.append(f"{d:%Y-%m-%d}:{res['status']}")
        for ym, fs in rst.items():
            write_month("stocks", ym, fs)
        for ym, fs in rix.items():
            write_month("indices", ym, fs)

    # Fetch 2015-04-01 .. 2019-10-15 corporate actions directly into DST_NSE/corp_actions.csv
    print("[backfill] fetching 2015-04..2019-10 NSE corporate actions...", flush=True)
    ca_before = len(pd.read_csv(os.path.join(DST_NSE, "corp_actions.csv")))
    try:
        S_ca = nd.Session()
        nd.fetch_corp_actions(S_ca, pd.Timestamp("2019-10-15"), start=pd.Timestamp("2015-04-01"))
        ca_df = pd.read_csv(os.path.join(DST_NSE, "corp_actions.csv"))
        print(f"[backfill] corp_actions: {ca_before} -> {len(ca_df)} (min={ca_df['ex_date'].min()} max={ca_df['ex_date'].max()})", flush=True)
    except Exception as e:
        print(f"[backfill] corp_actions error: {e}", flush=True)

    # Spot-check adjusted prices on RELIANCE (2017-09-07 1:1 bonus) & TCS (2018-05-31 1:1 bonus)
    npx._load.cache_clear()
    h = npx.history(["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"])
    for tk, chk_dt in [("RELIANCE.NS", "2017-09-07"), ("TCS.NS", "2018-05-31"), ("HDFCBANK.NS", "2019-09-19")]:
        df = h.get(tk)
        if df is not None and not df.empty:
            sub = df.loc[pd.Timestamp(chk_dt) - pd.Timedelta(days=5):pd.Timestamp(chk_dt) + pd.Timedelta(days=5)]
            print(f"[check] {tk} ({df.index.min().date()}..{df.index.max().date()}, n={len(df)}) around {chk_dt}:")
            for dt_i, r in sub.iterrows():
                print(f"    {dt_i.date()}: Open={r['Open']:.2f} Close={r['Close']:.2f}")

    summary = dict(
        elapsed_sec=round(time.time() - t0, 1),
        added_sessions=tot_ok,
        holidays_skipped=tot_hol,
        still_failed=still_failed,
    )
    with open(os.path.join(OUT_ROOT, "backfill_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[backfill] DONE in {time.time()-t0:.1f}s | added={tot_ok} | failed={len(still_failed)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
