"""
Annual + quarterly fundamentals and monthly prices for the valuation engine.
===========================================================================

Scrapes each universe stock's Screener.in page (consolidated first, then
standalone) for:
  headline Market Cap and Current Price (same moment -> exact share count)
  annual   profit & loss (sales, operating profit, OPM, interest, depreciation,
           PBT, net profit, EPS, dividend payout), balance sheet (equity
           capital, reserves, borrowings, deposits, fixed assets, CWIP,
           investments, total liabilities), cash flow (operating / investing /
           financing), ratios (ROCE, ROE)
  quarterly sales and net profit, and the shareholding pattern (promoters,
           FII, DII, public) -- the same parser as winners_setup_scan.py
and downloads ~6 years of month-end prices (Yahoo, adjusted) so the engine
can rebuild each stock's own P/E and P/B history.

Outputs (long format: ticker, field, period, value):
  data/fundamentals_annual.csv.gz
  data/fundamentals_quarterly.csv.gz
  data/monthly_prices.csv.gz            (date, ticker, close)
  analysis/screener_samples/*.html      two raw pages, so the parser can be
                                        fixed offline if Screener's layout moves

Run (GitHub Actions: Ranking / Rotation workflow, script=fundamentals):
  python fundamentals_annual.py [--max-tickers N] [--delay 1.0] [--skip-prices]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from urllib.parse import quote

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from winners_setup_scan import QROWS, SROWS, _records, _table  # noqa: E402

ANNUAL = "data/fundamentals_annual.csv.gz"
QUARTERLY = "data/fundamentals_quarterly.csv.gz"
MONTHLY = "data/monthly_prices.csv.gz"
SAMPLES = "analysis/screener_samples"

PL = {"sales": "sales", "revenue": "sales", "operating profit": "op", "financing profit": "op",
      "opm %": "opm", "financing margin %": "opm", "interest": "interest", "depreciation": "dep",
      "profit before tax": "pbt", "net profit": "np", "eps in rs": "eps", "dividend payout %": "payout"}
BS = {"equity capital": "eq_cap", "reserves": "reserves", "borrowings": "borrowings", "deposits": "deposits",
      "total liabilities": "total_liab", "fixed assets": "fixed_assets", "cwip": "cwip",
      "investments": "investments"}
CF = {"cash from operating activity": "cfo", "cash from investing activity": "cfi",
      "cash from financing activity": "cff"}
RT = {"roce %": "roce", "roe %": "roe"}


def _section(html: str, sid: str) -> str:
    m = re.search(rf'<section[^>]*id="{sid}".*?</section>', html, re.S)
    return m.group(0) if m else ""


def top_ratios(html: str) -> dict:
    """Screener's headline Market Cap (Rs cr) and Current Price, same moment -> share count."""
    out = {}
    for lab, fld in (("Market Cap", "mcap_cr"), ("Current Price", "price_scr")):
        m = re.search(rf'class="name">\s*{lab}\s*</span>.*?class="number">([\d,\.]+)<', html, re.S)
        if m:
            try:
                out[fld] = float(m.group(1).replace(",", ""))
            except ValueError:
                pass
    return out


def parse_annual(html: str) -> dict:
    out = {}
    for sid, rows in (("profit-loss", PL), ("balance-sheet", BS), ("cash-flow", CF), ("ratios", RT)):
        out.update(_records(_table(_section(html, sid)), rows))
    today = pd.Timestamp.today().normalize()
    out.update({(f, today): v for f, v in top_ratios(html).items()})
    return out


def parse_quarterly(html: str) -> dict:
    q = _records(_table(_section(html, "quarters")), QROWS)
    i = html.find('id="quarterly-shp"')
    frag = html[i:] if i >= 0 else _section(html, "shareholding")
    q.update(_records(_table(frag), SROWS))
    return q


def scrape(tickers, delay):
    import requests
    sess = requests.Session()
    sess.headers["User-Agent"] = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
    os.makedirs(SAMPLES, exist_ok=True)
    A, Q, fails, codes = [], [], [], {}

    def get(url):
        for k in range(4):
            try:
                r = sess.get(url, timeout=30)
            except Exception:
                time.sleep(5 * (k + 1))
                continue
            codes[r.status_code] = codes.get(r.status_code, 0) + 1
            if r.status_code == 429:
                time.sleep(30 * (k + 1))
                continue
            return r
        return None

    for n, t in enumerate(tickers):
        sym = quote(t.replace(".NS", ""), safe="")
        best_a, best_q, basis = {}, {}, ""
        for path, b in ((f"/company/{sym}/consolidated/", "consolidated"), (f"/company/{sym}/", "standalone")):
            r = get("https://www.screener.in" + path)
            time.sleep(delay)
            if r is None or r.status_code != 200:
                continue
            if n < 2:
                with open(f"{SAMPLES}/sample_{n}_{b}.html", "w") as fh:
                    fh.write(r.text)
            a, q = parse_annual(r.text), parse_quarterly(r.text)
            years = sum(1 for (f, _) in a if f == "np")
            if years >= 3 or not best_a:
                best_a, best_q, basis = a, q, b
            if years >= 3:
                break
        if not best_a:
            fails.append(t)
        A += [(t, f, p.date(), v, basis) for (f, p), v in best_a.items()]
        Q += [(t, f, p.date(), v) for (f, p), v in best_q.items()]
        if n == 19 and len(fails) == 20:
            sys.exit(f"[screener] first 20 requests all failed (HTTP {codes}) -- not writing anything")
        if (n + 1) % 100 == 0:
            print(f"[screener] {n + 1}/{len(tickers)}  failed {len(fails)}  HTTP {codes}", flush=True)
    print(f"[screener] done: {len(tickers) - len(fails)} parsed, {len(fails)} failed; HTTP {codes}")
    if fails:
        print("   failed: " + " ".join(f[:-3] for f in fails[:40]) + (" ..." if len(fails) > 40 else ""))
    return (pd.DataFrame(A, columns=["ticker", "field", "period", "value", "basis"]),
            pd.DataFrame(Q, columns=["ticker", "field", "period", "value"]))


def monthly_prices(tickers, batch=60):
    from utils.yf_safe import safe_download
    rows = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        bulk = safe_download(chunk, period="6y", interval="1mo", group_by="ticker", threads=False,
                             auto_adjust=True, min_coverage=0.5)
        for t in chunk:
            try:
                c = (bulk[t] if len(chunk) > 1 else bulk)["Close"].dropna()
            except Exception:
                continue
            if c.index.tz is not None:
                c.index = c.index.tz_localize(None)
            rows += [(d.date(), t, round(float(v), 4)) for d, v in c.items()]
        print(f"[prices] {min(i + batch, len(tickers))}/{len(tickers)}", flush=True)
        time.sleep(1)
    return pd.DataFrame(rows, columns=["date", "ticker", "close"])


def main():
    from utils.nifty1000_list import TICKERS_1000

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int, default=0)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--skip-prices", action="store_true")
    args = ap.parse_args()
    tickers = sorted(set(TICKERS_1000))
    if args.max_tickers:
        tickers = tickers[:args.max_tickers]

    A, Q = scrape(tickers, args.delay)
    na, nq = A.ticker.nunique(), Q.ticker.nunique()
    if na < 0.6 * len(tickers):
        sys.exit(f"[screener] only {na}/{len(tickers)} stocks parsed -- keeping the previous files")
    A.to_csv(ANNUAL, index=False)
    Q.to_csv(QUARTERLY, index=False)
    yrs = A[A.field == "np"].groupby("ticker").size()
    print(f"[annual] {na} stocks; years of net profit: median {yrs.median():.0f}, "
          f">= 5 years for {(yrs >= 5).sum()}; fields: {A.field.value_counts().to_dict()}")
    print(f"[quarterly] {nq} stocks; fields: {Q.field.value_counts().to_dict()}")
    print(f"[basis] {A.drop_duplicates('ticker').basis.value_counts().to_dict()}")

    if not args.skip_prices:
        M = monthly_prices(tickers)
        if M.ticker.nunique() >= 0.6 * len(tickers):
            M.to_csv(MONTHLY, index=False)
            print(f"[prices] {M.ticker.nunique()} stocks, {M.date.nunique()} months -> {MONTHLY}")
        else:
            print(f"[prices] only {M.ticker.nunique()} stocks -- keeping the previous file")


if __name__ == "__main__":
    main()
