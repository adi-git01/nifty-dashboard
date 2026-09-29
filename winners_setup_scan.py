"""
2025-26 doublers: what set them up, does that setup work on ALL stocks, and
who looks like that now. Plus: where FII + DII ownership has been rising.
===========================================================================

Question 1 (setups). Stocks in the point-in-time top 1000 on the first session
of 2025 that at least doubled by 17 Sep 2026 (the 1manfund list is flagged).
For each, the LAUNCH is the first breakout of 2025: a new 52-week closing high
after >= BASE_MIN sessions without one. It is the first such event, never the
best, so no hindsight picks the date.

  A1  Winners' features at launch vs every other breakout in the same window.
  A2  THE TEST. Features where winners stand out become a score. The score is
      then applied to every breakout from 2016 to 2024 -- a different period --
      and we check whether higher scores doubled more often. A winners-only
      profile always looks convincing; only this step says whether it is an
      edge or a description of what went up.
  A3  Stocks breaking out now, or coiled just under a 52-week high, ranked by
      the score.

Question 2 (ownership) and fundamentals come from Screener.in pages (13
quarters of results, 12 of shareholding):

  B1  Winners' last reported quarter and ownership trend before launch,
      compared with the whole universe on the same date.
  B2  Base rate: at four decision dates (Nov 2024 - Aug 2025) does growth or
      rising FII + DII ownership predict the next 12 months? One regime, four
      dates -- descriptive, not proof.
  B3  Names where FII + DII rose in each of the last two quarters.

Point in time: results count from quarter end + 60 days, shareholding from
quarter end + 21 days (the filing deadlines).

Run: python winners_setup_scan.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime
from html.parser import HTMLParser
from urllib.parse import quote

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

WIN_START, WIN_END = "2025-01-01", "2026-09-17"
POOL_END = "2026-06-30"          # breakouts after this have too little runway to compare
BASE_MIN = 40                    # sessions without a new 52w high before a breakout
HORIZON = 420                    # ~20 months: the same span as the 2025-26 window
SCREENER_RAW = "analysis/screener_raw.csv.gz"   # ~250k rows; gzip keeps the repo small
OUT = "analysis"

TWEET = ["STLTECH", "GUJGASLTD", "MTARTECH", "BLISSGVS", "SALSTEEL", "LAURUSLABS",
         "WELCORP", "WHEELS", "SANSERA", "SYRMA", "GABRIEL", "RBLBANK", "INDOBORAX",
         "NAVINFLUOR", "BAJAJCON", "ITDCEM", "AVALON", "CHENNPETRO", "LTF", "ABCAPITAL",
         "HAPPYFORGE", "GVT&D", "UNIPARTS", "KIRLOSENG", "ASTRAMICRO", "CRAFTSMAN",
         "POWERINDIA", "REDINGTON", "CENTUM", "RELINFRA"]
TWEET = [t + ".NS" for t in TWEET]

NUMERIC = ["base_len", "depth", "rs63", "rs126", "prior1y", "dist200", "slope200",
           "contraction", "vol_surge", "turn_rank"]
BOOLS = ["high3y"]
MARKET = ["nifty_up", "nifty_dd"]         # market-level: reported, never scored
LABEL = {
    "base_len": "sessions since previous 52w high", "depth": "deepest fall in prior year (%)",
    "rs63": "3m RS vs Nifty (pp)", "rs126": "6m RS vs Nifty (pp)",
    "prior1y": "1y return before (%)", "dist200": "% above 200dma",
    "slope200": "200dma 20-day slope (%)", "contraction": "volatility 20d / 120d",
    "vol_surge": "volume 10d / 120d", "turn_rank": "turnover rank (1 = most liquid)",
    "high3y": "new 3-year high", "nifty_up": "Nifty above its 200dma",
    "nifty_dd": "Nifty % below 52w high",
}


# ----------------------------------------------------------------------------
# A. price setups
# ----------------------------------------------------------------------------
def build_features(close, vol, nifty):
    dates = close.index
    nc = nifty["Close"].reindex(dates).ffill()
    hi252 = close.rolling(252, min_periods=200).max()
    new_high = close.ge(hi252) & hi252.notna()
    pos = np.arange(len(dates), dtype=float)[:, None]
    last = pd.DataFrame(np.where(new_high.values, pos, np.nan), index=dates,
                        columns=close.columns).ffill()
    since = pd.DataFrame(pos - last.values, index=dates, columns=close.columns)
    ret = close.pct_change(fill_method=None)
    cf = close.ffill(limit=5)       # for "N sessions ago": a 1-5 day gap uses the last close
    ma200 = close.rolling(200, min_periods=150).mean()
    n_ret = lambda k: nc / nc.shift(k) - 1
    turn = (close * vol).rolling(60, min_periods=30).median()
    hi3y = close.rolling(756, min_periods=500).max()
    F = {
        "base_len": since.shift(1),
        "depth": (close.rolling(252, min_periods=200).min().shift(1) / hi252.shift(1) - 1) * 100,
        "rs63": (close / cf.shift(63) - 1).sub(n_ret(63), axis=0) * 100,
        "rs126": (close / cf.shift(126) - 1).sub(n_ret(126), axis=0) * 100,
        "prior1y": (cf.shift(1) / cf.shift(253) - 1) * 100,
        "dist200": (close / ma200 - 1) * 100,
        "slope200": (ma200 / ma200.shift(20) - 1) * 100,
        "contraction": (ret.rolling(20, min_periods=15).std()
                        / ret.rolling(120, min_periods=90).std()).shift(1),
        "vol_surge": vol.rolling(10, min_periods=5).mean()
                     / vol.rolling(120, min_periods=60).mean().shift(10),
        "turn_rank": turn.rank(axis=1, ascending=False),
        # unknown (NaN), not False, when there are < 500 sessions of history
        "high3y": close.ge(hi3y).astype(float).where(hi3y.notna()),
    }
    mkt = pd.DataFrame({"nifty_up": (nc > nc.rolling(200).mean()).astype(float),
                        "nifty_dd": (nc / nc.rolling(252).max() - 1) * 100}, index=dates)
    fmax = close.iloc[::-1].rolling(HORIZON, min_periods=1).max().iloc[::-1].shift(-1)
    doubled = (fmax / close >= 2).astype(float)
    doubled.iloc[len(dates) - HORIZON:] = np.nan
    fwd = cf.shift(-252) / close - 1
    fwd_x = fwd.sub(nc.shift(-252) / nc - 1, axis=0) * 100
    return F, mkt, new_high, since, hi252, doubled, fwd * 100, fwd_x


def events_table(mask, F, mkt, doubled, fwd, fwd_x):
    ii, jj = np.where(mask.values)
    dates, cols = mask.index, mask.columns
    T = pd.DataFrame({"date": dates[ii], "ticker": cols[jj]})
    for k, v in F.items():
        T[k] = v.values[ii, jj]
    for k in MARKET:
        T[k] = mkt[k].values[ii]
    T["doubled"] = doubled.values[ii, jj]
    T["fwd12"] = fwd.values[ii, jj]
    T["fwd12_x"] = fwd_x.values[ii, jj]
    return T


def winner_list(close, pit):
    d = close.index
    t0 = d[d >= WIN_START][0]
    t1 = d[d <= WIN_END][-1]
    cf = close.ffill(limit=5)
    tot = (cf.loc[t1] / cf.loc[t0] - 1) * 100
    inuni = pit.loc[t0:].head(5).any()                  # in the universe in the first week
    W = tot[(tot >= 100) & inuni].sort_values(ascending=False)
    print(f"\n[winners] {len(W)} stocks in the top-1000 on {t0.date()} doubled by {t1.date()}")
    for t in TWEET:
        if t in W.index:
            continue
        why = ("no price data" if t not in close.columns else
               "not in top-1000 in the first week of 2025" if not inuni.get(t, False) else
               f"{tot.get(t, np.nan):+.0f}% on adjusted prices")
        print(f"   tweet name not counted: {t[:-3]:<12} {why}")
    return W, t0, t1, tot


def launches(W, E, close, new_high, t0, t1):
    rows = []
    for t in W.index:
        e = E[(E.ticker == t) & (E.date >= t0) & (E.date <= t1)]
        if len(e):
            d, kind = e.date.iloc[0], "base breakout"
        else:
            nh = new_high[t].loc[t0:t1]
            if not nh.any():
                continue
            d, kind = nh.idxmax(), "no base (already making highs)"
        c = close[t].ffill(limit=5)
        rows.append(dict(ticker=t, launch=d, kind=kind, tweet=t in TWEET,
                         total_pct=round(W[t], 1),
                         after_launch_pct=round((c.loc[t1] / c.loc[d] - 1) * 100, 1),
                         sessions_to_launch=int(((close.index >= t0) & (close.index < d)).sum())))
    return pd.DataFrame(rows)


def profile(L, E, t0):
    pool = E[(E.date >= t0) & (E.date <= POOL_END) & ~E.ticker.isin(L.ticker)]
    win = E.merge(L[["ticker", "launch"]], left_on=["ticker", "date"], right_on=["ticker", "launch"])
    rows, rule = [], []
    for f in NUMERIC + BOOLS + MARKET:
        wv, pv = win[f].dropna(), pool[f].dropna()
        if f in BOOLS + MARKET:
            agg = (lambda x: x.median()) if f == "nifty_dd" else (lambda x: x.mean() * 100)
            wr, pr = agg(wv), agg(pv)
            side = (">" if wr > pr else "<") if f in BOOLS and abs(wr - pr) >= 15 else ""
            if side:
                rule.append((f, side, 0.5))
            rows.append(dict(feature=LABEL[f] + ("" if f == "nifty_dd" else " (% of stocks)"),
                             winners=round(wr, 1), other_breakouts=round(pr, 1),
                             winners_percentile="", scored="yes" if side else ""))
            continue
        med = wv.median()
        pct = (pv < med).mean() * 100
        side = ">" if pct >= 65 else "<" if pct <= 35 else ""
        if side:
            rule.append((f, side, pv.median()))
        rows.append(dict(feature=LABEL[f], winners=round(med, 2), other_breakouts=round(pv.median(), 2),
                         winners_percentile=round(pct), scored=f"{side} {pv.median():.2f}" if side else ""))
    print(f"\nA1. WINNERS AT LAUNCH ({len(win)}) vs OTHER BREAKOUTS {t0.date()} -> {POOL_END} ({len(pool)})")
    print("    winners_percentile = where the winners' median sits among other breakouts (50 = typical)")
    print(pd.DataFrame(rows).to_string(index=False))
    return rule, win, pool


def score(df, rule):
    s = pd.Series(0, index=df.index)
    for f, side, thr in rule:
        v = df[f]
        s += (v > thr) if side == ">" else (v < thr)
    return s


def history_test(E, rule, t0):
    H = E[(E.date < t0) & E.doubled.notna()].copy()
    H["score"] = score(H, rule)
    base = H.doubled.mean() * 100
    g = H.groupby("score").agg(breakouts=("doubled", "size"), doubled_pct=("doubled", "mean"),
                               med_fwd12_excess=("fwd12_x", "median"),
                               mean_fwd12_excess=("fwd12_x", "mean"))
    g["doubled_pct"] = (g.doubled_pct * 100).round(1)
    g["lift"] = (g.doubled_pct / base).round(2)
    g = g.round(1)
    print(f"\nA2. OUT-OF-PERIOD TEST: every base breakout {H.date.min().date()} -> {H.date.max().date()} "
          f"({len(H)}), scored with the 2025 winners' rule")
    print("    rule (one point each): " + "; ".join(
        f"{LABEL[f]} {'yes' if s == '>' else 'no'}" if f in BOOLS else f"{LABEL[f]} {s} {t:.2f}"
        for f, s, t in rule))
    print(f"    base rate: {base:.1f}% of breakouts doubled within {HORIZON} sessions")
    print(g.to_string())
    # the same split by year halves, so one era cannot carry the result
    H["era"] = np.where(H.date < "2021-01-01", "2016-20", "2021-24")
    top = H.score >= max(1, len(rule) - 1)
    e = H.groupby(["era", top.rename("high_score")]).doubled.agg(["size", "mean"])
    e["mean"] = (e["mean"] * 100).round(1)
    print(f"\n    high score = {max(1, len(rule) - 1)}+ of {len(rule)} conditions, by era (mean = % doubled):")
    print(e.to_string())
    # single-feature quintiles, for the record
    q = []
    for f in NUMERIC:
        x = H[[f, "doubled", "fwd12_x"]].dropna()
        if len(x) < 500:
            continue
        x["q"] = pd.qcut(x[f].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        r = x.groupby("q", observed=True).doubled.mean() * 100
        q.append(dict(feature=LABEL[f], **{f"Q{k}": round(v, 1) for k, v in r.items()},
                      Q5_minus_Q1=round(r.iloc[-1] - r.iloc[0], 1)))
    print("\n    % doubled by quintile of each feature on its own (Q1 = lowest value):")
    print(pd.DataFrame(q).to_string(index=False))
    return g, H


def scan_now(F, mkt, close, hi252, since, pit, rule, W, subind):
    d = close.index[-1]
    recent = close.index[-30]
    now = pd.DataFrame({k: v.loc[d] for k, v in F.items()})
    now["base_len"] = since.loc[d]
    now["from_high_pct"] = (close.loc[d] / hi252.loc[d] - 1) * 100
    last_bo = {}
    for t in close.columns:
        s = since[t].loc[recent:]
        b = F["base_len"][t].loc[recent:]
        hit = (s == 0) & (b >= BASE_MIN)
        if hit.any():
            last_bo[t] = hit[hit].index[-1]
    now["breakout_date"] = pd.Series(last_bo)
    for t, bd in last_bo.items():             # a recent breakout is judged as it broke out
        for k, v in F.items():
            now.at[t, k] = v.at[bd, t]
    now = now[pit.loc[d].values & ~now.index.isin(W.index)]
    now["score"] = score(now, rule)
    now["state"] = np.where(now.breakout_date.notna(), "broke out (30 sessions)",
                            np.where((now.from_high_pct >= -5) & (now.base_len >= BASE_MIN),
                                     "coiled under 52w high", ""))
    now["sub_industry"] = now.index.map(subind)
    out = now[now.state != ""].sort_values(["score", "rs126"], ascending=False)
    return out, d


# ----------------------------------------------------------------------------
# B. Screener.in results + shareholding
# ----------------------------------------------------------------------------
class _Rows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def _table(fragment):
    m = re.search(r"<table.*?</table>", fragment or "", re.S)
    if not m:
        return []
    p = _Rows()
    p.feed(m.group(0))
    return p.rows


def _num(s):
    s = (s or "").replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _records(rows, names):
    """rows -> {(field, quarter_end): value} for the row labels in `names`."""
    if len(rows) < 2:
        return {}
    qs = []
    for h in rows[0][1:]:
        try:
            qs.append(pd.Timestamp(datetime.strptime(h.strip(), "%b %Y")) + pd.offsets.MonthEnd(0))
        except ValueError:
            qs.append(None)
    out = {}
    for r in rows[1:]:
        label = re.sub(r"[^a-z% ]", "", r[0].lower()).strip()
        field = names.get(label)
        if not field:
            continue
        for q, v in zip(qs, r[1:]):
            if q is not None and not np.isnan(_num(v)):
                out[(field, q)] = _num(v)
    return out


QROWS = {"sales": "sales", "revenue": "sales", "net profit": "np"}
SROWS = {"promoters": "promoters", "fiis": "fii", "diis": "dii", "public": "public",
         "no of shareholders": "holders"}


def parse_screener(html):
    sec = re.search(r'<section[^>]*id="quarters".*?</section>', html, re.S)
    q = _records(_table(sec.group(0) if sec else ""), QROWS)
    i = html.find('id="quarterly-shp"')
    if i < 0:
        s = re.search(r'<section[^>]*id="shareholding".*?</section>', html, re.S)
        frag = s.group(0) if s else ""
    else:
        frag = html[i:]
    return q, _records(_table(frag), SROWS)


def scrape(tickers, delay):
    import requests
    sess = requests.Session()
    sess.headers["User-Agent"] = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
    recs, fails, codes = [], [], {}
    os.makedirs(OUT, exist_ok=True)

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
        best_q, best_s = {}, {}
        for path in (f"/company/{sym}/consolidated/", f"/company/{sym}/"):
            r = get("https://www.screener.in" + path)
            time.sleep(delay)
            if r is None or r.status_code != 200:
                continue
            if n < 2 and not os.path.exists(f"{OUT}/screener_sample_{n}.html"):
                open(f"{OUT}/screener_sample_{n}.html", "w").write(r.text)
            q, s = parse_screener(r.text)
            best_s = best_s or s
            if sum(1 for (f, _) in q if f == "np") >= 5:
                best_q = q
                break
            best_q = best_q or q
        if not best_q and not best_s:
            fails.append(t)
        for (f, qe), v in {**best_q, **best_s}.items():
            recs.append((t, f, qe, v))
        if n == 19 and len(fails) == 20:
            print(f"[screener] first 20 requests all failed (HTTP {codes}); skipping part B")
            return None
        if (n + 1) % 100 == 0:
            print(f"[screener] {n + 1}/{len(tickers)}  failed {len(fails)}  HTTP {codes}", flush=True)
    print(f"[screener] done: {len(tickers) - len(fails)} parsed, {len(fails)} failed; HTTP {codes}")
    if fails:
        print("   failed: " + " ".join(f[:-3] for f in fails[:40]) + (" ..." if len(fails) > 40 else ""))
    df = pd.DataFrame(recs, columns=["ticker", "field", "quarter", "value"])
    df.to_csv(SCREENER_RAW, index=False)
    return df


def _qe(q, back):
    """Quarter end `back` quarters before quarter end q."""
    return q - pd.offsets.MonthEnd(3 * back)


def fund_features(P, when):
    """One row per ticker: what was public on `when` (results +60d, holdings +21d).
    P: {ticker: DataFrame indexed by quarter end, one column per field}.

    Every comparison is by DATE (same quarter a year earlier, the quarters 3
    and 6 months earlier), never by column position: a missing column or an
    off-cycle filing (Screener shows e.g. "Feb 2025" after a merger) would
    otherwise pair the wrong quarters -- the same class of error as the MA50
    bug.
    """
    out = {}
    pos = lambda a: a if a > 0 else np.nan
    for t, g in P.items():
        row = {}
        g = g[g.index.is_month_end & g.index.month.isin([3, 6, 9, 12])]
        if {"sales", "np"} <= set(g.columns):
            r = g.loc[g.index + pd.Timedelta(days=60) <= when, ["sales", "np"]].dropna()
            if len(r):
                q = r.index[-1]
                at = lambda k: r.loc[_qe(q, k)] if _qe(q, k) in r.index else None
                yago, prev, prev_y = at(4), at(1), at(5)
                if yago is not None:
                    row["sales_yoy"] = (r.at[q, "sales"] / pos(yago.sales) - 1) * 100
                    row["np_yoy"] = (r.at[q, "np"] / pos(yago.np) - 1) * 100
                    row["turnaround"] = float(yago.np <= 0 < r.at[q, "np"])
                    if prev is not None and prev_y is not None:
                        row["np_accel"] = row["np_yoy"] - (prev.np / pos(prev_y.np) - 1) * 100
                    row["results_q"] = q.date()
        if {"fii", "dii"} <= set(g.columns):
            sh = g.loc[g.index + pd.Timedelta(days=21) <= when].dropna(subset=["fii", "dii"])
            if len(sh):
                h = sh.index[-1]
                if _qe(h, 1) in sh.index and _qe(h, 2) in sh.index:
                    x0, x1, x2 = sh.loc[h], sh.loc[_qe(h, 1)], sh.loc[_qe(h, 2)]
                    i0, i1, i2 = x0.fii + x0.dii, x1.fii + x1.dii, x2.fii + x2.dii
                    row.update(inst_now=i0, inst_chg_2q=i0 - i2,
                               inst_rising_2q=float(i0 > i1 > i2),
                               fii_chg_2q=x0.fii - x2.fii, dii_chg_2q=x0.dii - x2.dii,
                               holding_q=h.date())
                    if "promoters" in sh:
                        row["promoter_chg_2q"] = x0.promoters - x2.promoters
                    if "holders" in sh and x2.holders > 0:
                        row["holders_chg_2q_pct"] = (x0.holders / x2.holders - 1) * 100
        if row:
            out[t] = row
    return pd.DataFrame.from_dict(out, orient="index")


# the first session on which each quarter's results are public (quarter end + 60d)
B2_DATES = ("2024-11-29", "2025-03-03", "2025-06-02", "2025-08-29")
FUND = ["sales_yoy", "np_yoy", "np_accel", "turnaround", "inst_chg_2q", "inst_rising_2q",
        "fii_chg_2q", "dii_chg_2q", "promoter_chg_2q", "holders_chg_2q_pct"]


def part_b(raw, L, close, nifty, pit, subind, setup_now):
    dates = close.index
    piv = raw.pivot_table(index=["ticker", "quarter"], columns="field", values="value")
    P = {t: g.droplevel(0).sort_index() for t, g in piv.groupby(level=0)}
    inuni = lambda d: close.columns[pit.loc[d].values]
    snap = lambda d: dates[dates >= pd.Timestamp(d)][0]

    # B1 winners at launch vs everyone on the same day
    per = []
    for d, grp in L.groupby("launch"):
        ff = fund_features(P, d)
        ff = ff[ff.index.isin(inuni(d)) | ff.index.isin(grp.ticker)]
        for t in grp.ticker:
            if t not in ff.index:
                continue
            r = {"ticker": t[:-3], "launch": d.date()}
            for f in FUND:
                if f in ff and not pd.isna(ff.at[t, f]):
                    r[f] = ff.at[t, f]
                    if ff[f].nunique() > 2:
                        r[f + "_pctile"] = (ff[f].dropna() < ff.at[t, f]).mean() * 100
            per.append(r)
    if not per:
        print("\nB1. no fundamentals for the winners (Screener parse failed?)")
        return
    Wf = pd.DataFrame(per)
    Wf.round(1).to_csv(f"{OUT}/winners_fundamentals_at_launch.csv", index=False)
    print(f"\nB1. WINNERS' LAST PUBLIC QUARTER BEFORE LAUNCH ({len(Wf)} with data)")
    print("    median value, and median percentile vs every universe stock on the same day (50 = typical)")
    s = []
    for f in FUND:
        if f not in Wf:
            continue
        v = Wf[f].dropna()
        pc = Wf.get(f + "_pctile", pd.Series(dtype=float)).dropna()
        s.append(dict(field=f, winners_with_data=len(v),
                      median=round(v.median(), 1) if f not in ("turnaround", "inst_rising_2q") else "",
                      share_true_pct=round(v.mean() * 100) if f in ("turnaround", "inst_rising_2q") else "",
                      median_percentile=round(pc.median()) if len(pc) else ""))
    print(pd.DataFrame(s).to_string(index=False))

    # B2 base rate at four decision dates
    nc = nifty["Close"].reindex(dates).ffill()
    B = []
    for d in B2_DATES:
        d = snap(d)
        e = dates[min(dates.get_loc(d) + 252, len(dates) - 1)]
        ff = fund_features(P, d)
        ff = ff[ff.index.isin(inuni(d))]
        fx = (close.loc[e] / close.loc[d] - 1 - (nc[e] / nc[d] - 1)) * 100
        ff["fwd12_x"] = fx.reindex(ff.index)
        ff["doubled"] = (close.loc[d:e].max() / close.loc[d] >= 2).reindex(ff.index).astype(float)
        ff["date"] = d
        B.append(ff)
    B = pd.concat(B).dropna(subset=["fwd12_x"])
    print(f"\nB2. DOES IT PREDICT? {len(B)} stock-dates (4 decision dates, Nov 2024 - Aug 2025)")
    print(f"    base: median 12m excess {B.fwd12_x.median():+.1f} pp, doubled within 12m {B.doubled.mean() * 100:.1f}%")
    t = []
    for f in ["sales_yoy", "np_yoy", "np_accel", "inst_chg_2q", "fii_chg_2q", "dii_chg_2q",
              "promoter_chg_2q", "holders_chg_2q_pct"]:
        x = B[[f, "fwd12_x", "doubled", "date"]].dropna()
        if len(x) < 300:
            continue
        x["q"] = x.groupby("date")[f].transform(
            lambda v: pd.qcut(v.rank(method="first"), 5, labels=False) + 1)
        g = x.groupby("q").agg(m=("fwd12_x", "median"), d=("doubled", "mean"))
        t.append(dict(field=f, n=len(x), **{f"Q{int(k)}_excess": round(v, 1) for k, v in g.m.items()},
                      Q1_doubled=round(g.d.iloc[0] * 100, 1), Q5_doubled=round(g.d.iloc[-1] * 100, 1)))
    for f in ["inst_rising_2q", "turnaround"]:
        x = B[[f, "fwd12_x", "doubled"]].dropna()
        g = x.groupby(f).agg(n=("doubled", "size"), m=("fwd12_x", "median"), d=("doubled", "mean"))
        t.append(dict(field=f"{f} (no / yes)", n=len(x),
                      Q1_excess=round(g.m.get(0.0, np.nan), 1), Q5_excess=round(g.m.get(1.0, np.nan), 1),
                      Q1_doubled=round(g.d.get(0.0, np.nan) * 100, 1),
                      Q5_doubled=round(g.d.get(1.0, np.nan) * 100, 1)))
    T = pd.DataFrame(t)
    T.to_csv(f"{OUT}/fundamentals_lift.csv", index=False)
    print("    median 12m excess vs Nifty (pp) by quintile within each date; Q5 = highest value")
    print(T.to_string(index=False))

    # B3 FII + DII rising
    d = dates[-1]
    now = fund_features(P, d + pd.Timedelta(days=60))       # latest filings, whatever their date
    latest = pd.Series(now.holding_q).mode()[0]
    now = now[(now.holding_q == latest) & now.index.isin(inuni(d))]
    rs126 = ((close.loc[d] / close.iloc[-127] - 1) - (nc.iloc[-1] / nc.iloc[-127] - 1)) * 100
    hi = close.iloc[-252:].max()
    now["rs126_now"] = rs126.reindex(now.index)
    now["from_52w_high_pct"] = ((close.loc[d] / hi - 1) * 100).reindex(now.index)
    now["sub_industry"] = now.index.map(subind)
    now["setup_score"] = setup_now.reindex(now.index)
    rising = now[now.inst_rising_2q == 1].sort_values("inst_chg_2q", ascending=False)
    cols = ["inst_now", "inst_chg_2q", "fii_chg_2q", "dii_chg_2q", "promoter_chg_2q",
            "holders_chg_2q_pct", "np_yoy", "sales_yoy", "rs126_now", "from_52w_high_pct",
            "setup_score", "sub_industry"]
    out = rising[[c for c in cols if c in rising]].round(1)
    out.index = out.index.str.replace(".NS", "", regex=False)
    out.to_csv(f"{OUT}/fii_dii_rising.csv")
    both = ((rising.fii_chg_2q > 0) & (rising.dii_chg_2q > 0)).sum()
    print(f"\nB3. FII + DII ROSE IN EACH OF THE LAST TWO QUARTERS (to {latest}): {len(rising)} of {len(now)} "
          f"universe stocks; both FII and DII up in {both}")
    print(out.head(40).to_string())
    print(f"    full list -> {OUT}/fii_dii_rising.csv")


# ----------------------------------------------------------------------------
def main():
    from exit_rule_backtest import fetch
    from momentum_factor_backtest import build_pit_universe, load_candidates

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int, default=0)
    ap.add_argument("--top-n", type=int, default=1000)
    ap.add_argument("--screener", choices=["scrape", "reuse", "skip"], default="scrape")
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    tickers = load_candidates("all", args.max_tickers)
    close, vol, nifty = fetch(tickers, "2015-06-01", datetime.now().strftime("%Y-%m-%d"))
    pit = build_pit_universe(close, vol, args.top_n)
    print(f"[data] {len(close)} sessions {close.index[0].date()} -> {close.index[-1].date()}, "
          f"{close.shape[1]} tickers")
    sl = pd.read_csv("data/nifty1000_list.csv")
    subind = dict(zip(sl.Ticker, sl.Sub_Industry))

    F, mkt, new_high, since, hi252, doubled, fwd, fwd_x = build_features(close, vol, nifty)
    E = events_table(new_high & (F["base_len"] >= BASE_MIN) & pit, F, mkt, doubled, fwd, fwd_x)
    print(f"[events] {len(E)} base breakouts (new 52w closing high after >= {BASE_MIN} sessions without one)")

    W, t0, t1, tot = winner_list(close, pit)
    L = launches(W, E, close, new_high, t0, t1)
    L["sub_industry"] = L.ticker.map(subind)
    print(f"\n    launch type: {L.kind.value_counts().to_dict()}")
    print(f"    median move {L.total_pct.median():.0f}%, of which after the launch breakout "
          f"{L.after_launch_pct.median():.0f}%; median {L.sessions_to_launch.median():.0f} sessions into 2025")
    print("    sub-industries with 3+ winners: " + ", ".join(
        f"{k} {v}" for k, v in L.sub_industry.value_counts().items() if v >= 3))
    Lp = L.assign(ticker=L.ticker.str[:-3], launch=L.launch.dt.date)
    Lp.to_csv(f"{OUT}/winners_2025_launches.csv", index=False)
    print(Lp.head(60).to_string(index=False))

    rule, win, pool = profile(L[L.kind == "base breakout"], E, t0)
    if not rule:
        print("\nNo feature separates the winners from other breakouts -> no score to test.")
        setup_now = pd.Series(dtype=float)
    else:
        g, H = history_test(E, rule, t0)
        g.to_csv(f"{OUT}/setup_score_history.csv")
        now, d = scan_now(F, mkt, close, hi252, since, pit, rule, W, subind)
        setup_now = now.score
        keep = ["state", "score", "breakout_date", "from_high_pct", "base_len", "rs126", "rs63",
                "depth", "contraction", "vol_surge", "sub_industry"]
        show = now[keep].copy()
        show["breakout_date"] = show.breakout_date.dt.date
        show.index = show.index.str.replace(".NS", "", regex=False)
        show.round(1).to_csv(f"{OUT}/setup_scan_now.csv")
        print(f"\nA3. SIMILAR SETUPS NOW ({d.date()}): {len(show)} names breaking out or coiled; "
              f"top by score (max {len(rule)})")
        for st in ("broke out (30 sessions)", "coiled under 52w high"):
            print(f"\n  {st}:")
            print(show[show.state == st].drop(columns="state").head(25).round(1).to_string())

    if args.screener == "skip":
        return
    # Everyone in the universe on any date part B looks at -- not just today's
    # names, or stocks that fell out since (mostly losers) would vanish from the
    # base-rate test and flatter it.
    look = [pd.Timestamp(d) for d in B2_DATES] + list(L.launch) + [close.index[-1]]
    look = [close.index[close.index >= d][0] for d in look]
    names = sorted(set().union(*[set(close.columns[pit.loc[d].values]) for d in look])
                   | set(W.index) | set(TWEET))
    if args.screener == "reuse" and os.path.exists(SCREENER_RAW):
        raw = pd.read_csv(SCREENER_RAW, parse_dates=["quarter"])
    else:
        print(f"\n[screener] scraping {len(names)} names at ~{args.delay}s each...", flush=True)
        raw = scrape(names, args.delay)
    if raw is None or raw.empty:
        return
    raw["quarter"] = pd.to_datetime(raw.quarter)
    part_b(raw, L, close, nifty, pit, subind, setup_now)


if __name__ == "__main__":
    main()
