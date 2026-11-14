"""
Signal scorecard: has each scanner's signal list beaten the market after the signal?

For every signal log, the forward return of each signalled stock over 5 / 21 / 63 sessions is
measured from the daily snapshots (split-adjusted) as EXCESS over the median stock on the same
dates, then averaged by signal date (so one busy day counts once). A stock that re-signals within
21 days counts once. Returns are clipped at -60% / +150%.

The yardstick is "Scanner BUY, any day": every stock on the Trend Scanner's BUY list, every day.
A scanner earns its place only if it beats that. Verdict per row:

  BEATS MOMENTUM   21- and 63-session excess both at least 1 pp above the yardstick, hit rate > 50%
  LIKE MOMENTUM    positive, but no better than the yardstick
  NO EDGE          around zero, or positive at one horizon only
  LAGS MARKET      negative at both 21 and 63 sessions
  TOO NEW          fewer than 100 signals, or fewer than 60 signal days, with a 21-session outcome

One regime and under a year of data: read it as "what has this done so far", not proof.
build() is run by the EOD job (run_daily_signals.py) and written to data/signal_scorecard.csv;
it also writes data/model_portfolio_benchmark.csv (model portfolio vs the market).
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

OUT = "data/signal_scorecard.csv"
BENCH_OUT = "data/model_portfolio_benchmark.csv"
HORIZONS = (5, 21, 63)
YARDSTICK = "Scanner BUY, any day (yardstick)"
MIN_N = 100
MIN_DAYS = 60              # distinct signal days with a 21-session outcome: about three months of signals
# name, file, date column, where it shows, what it claims
LOGS = [
    ("Turnaround Radar (IAS watchlist)", "data/ias_signal_log.csv", "signal_date", "Turnaround Radar", "accumulation in beaten-down stocks"),
    ("Turnaround Catalyst", "data/tc_log.csv", "signal_date", "Live Desk / Morning Pulse", "big up day or trend-score jump off the lows"),
    ("Earnings Shock", "data/earnings_shock_log.csv", "signal_date", "Live Desk", "4%+ jump on 2.5x volume"),
    ("RS Divergence", "data/rs_divergence_log.csv", "signal_date", "Live Desk", "up on a red market day, near highs"),
    ("Breakouts (52W / ATH / squeeze)", "data/breakout_log.csv", "breakout_date", "Trend Scanner", "new high or squeeze release"),
    ("V2.2 watchlist BUY alert", "data/watchlist_history.csv", "added_date", "Live Desk / Telegram", "trend score 70+ with accumulation"),
    ("Agreement (2+ of SIMBA, Scanner, Technicals)", "data/consensus_log.csv", "first_seen", "Trend Scanner → Agreement", "three tools agree"),
]


def price_panel(root: str = ".") -> tuple[pd.DataFrame, Dict[pd.Timestamp, pd.DataFrame]]:
    """Sessions x tickers close panel from the snapshots, holiday copies dropped, splits adjusted."""
    P, extra, prev = {}, {}, None
    for f in sorted(glob.glob(os.path.join(root, "data/cache/market_master_*.parquet"))):
        try:
            d = pd.read_parquet(f)
        except Exception:
            continue
        d = d.drop_duplicates("ticker").set_index("ticker")
        px = pd.to_numeric(d["currentPrice"], errors="coerce")
        dt = pd.Timestamp(os.path.basename(f)[14:24].replace("_", "-"))
        if "bar_date" in d.columns and d["bar_date"].notna().any():
            dt = pd.Timestamp(d["bar_date"].dropna().iloc[0]).normalize()
        if dt.weekday() >= 5 or dt in P:
            continue
        if prev is not None:
            j = px.index.intersection(prev.index)
            if len(j) > 100 and (px[j] == prev[j]).mean() > 0.9:
                continue
        P[dt], prev = px, px
        extra[dt] = d[[c for c in ("dna_signal", "trend_score", "volume_signal_score", "marketCap") if c in d.columns]]
    P = pd.DataFrame(P).T.sort_index()
    try:
        from utils import corp_actions
        for _, r in corp_actions.load().iterrows():
            if r["ticker"] in P.columns and float(r["factor"]) > 0:
                P.loc[P.index < pd.Timestamp(r["date"]), r["ticker"]] /= float(r["factor"])
    except Exception:
        pass
    return P, extra


def _excess(P: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    out = {}
    for h in HORIZONS:
        R = ((P.shift(-h) / P - 1) * 100).clip(-60, 150)
        out[h] = R.sub(R.median(axis=1), axis=0)
    return out


def _score(name: str, sig: pd.DataFrame, F: Dict[int, pd.DataFrame], idx: pd.DatetimeIndex, first_days: int = 21) -> dict:
    d = sig[["ticker", "date"]].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
    d = d.dropna()
    pos = idx.searchsorted(d["date"])
    d = d[pos < len(idx)]
    d["s"] = idx[pos[pos < len(idx)]]
    d = d.sort_values("s")
    if first_days:
        last, keep = {}, []
        for t, s in zip(d.ticker, d.s):
            ok = t not in last or (s - last[t]).days > first_days
            if ok:
                last[t] = s
            keep.append(ok)
        d = d[keep]
    row = dict(signal=name, signals=len(d), first=d.s.min().strftime("%Y-%m-%d") if len(d) else "")
    for h in HORIZONS:
        col = F[h].columns
        x = np.array([F[h].at[s, t] if t in col else np.nan for s, t in zip(d.s, d.ticker)], dtype=float)
        v = pd.DataFrame({"x": x, "s": d.s.values}).dropna()
        m = v.groupby("s").x.mean()
        row[f"n_{h}d"] = len(v)
        row[f"days_{h}d"] = int(v.s.nunique())
        row[f"avg_{h}d"] = round(float(m.mean()), 2) if len(m) else np.nan
        row[f"median_{h}d"] = round(float(v.x.median()), 2) if len(v) else np.nan
        row[f"hit_{h}d"] = round(float((v.x > 0).mean() * 100), 0) if len(v) else np.nan
    return row


def _verdict(r: pd.Series, base: Optional[pd.Series]) -> str:
    if r["n_21d"] < MIN_N or r.get("days_21d", MIN_DAYS) < MIN_DAYS:
        return "TOO NEW"
    a21, a63 = r["avg_21d"], r["avg_63d"]
    has63 = r["n_63d"] >= MIN_N and pd.notna(a63)
    if a21 < 0 and (not has63 or a63 < 0):
        return "LAGS MARKET"
    if base is not None and has63 and a21 >= base["avg_21d"] + 1 and a63 >= base["avg_63d"] + 1 and r["hit_21d"] > 50:
        return "BEATS MOMENTUM"
    if a21 > 0.5 and (not has63 or a63 > 0.5):
        return "LIKE MOMENTUM"
    return "NO EDGE"


def build(root: str = ".", save: bool = True) -> pd.DataFrame:
    P, extra = price_panel(root)
    if len(P) < 30:
        return pd.DataFrame()
    F, idx = _excess(P), P.index
    rows: List[dict] = []
    for name, path, dcol, where, claim in LOGS:
        p = os.path.join(root, path)
        if not os.path.exists(p):
            continue
        try:
            L = pd.read_csv(p)
            r = _score(name, L.rename(columns={dcol: "date"}), F, idx)
            r.update(where=where, claim=claim)
            rows.append(r)
            if name.startswith("Turnaround Radar") and "signal_tier" in L:
                for tier in ("ALERT", "READY"):
                    r = _score(f"   …{tier} tier only", L[L.signal_tier == tier].rename(columns={dcol: "date"}), F, idx)
                    r.update(where=where, claim="the radar's highest-conviction tiers")
                    rows.append(r)
        except Exception as e:
            print(f"[scorecard] {name}: {e}")
    # derived lists the Morning Pulse page shows, and the yardstick
    der = {YARDSTICK: [], "Morning Pulse: top-10 trend-score gainers": [], "Morning Pulse: top-10 trend-score losers": [],
           "Morning Pulse: accumulation leaders (top 10)": []}
    dates = [d for d in extra if d in idx]
    for i, dt in enumerate(dates):
        d = extra[dt]
        if "dna_signal" in d:
            der[YARDSTICK] += [(t, dt) for t in d.index[d.dna_signal.astype(str) == "BUY"]]
        if i and "trend_score" in d and "trend_score" in extra[dates[i - 1]]:
            ch = (pd.to_numeric(d.trend_score, errors="coerce")
                  - pd.to_numeric(extra[dates[i - 1]].trend_score, errors="coerce").reindex(d.index)).dropna()
            der["Morning Pulse: top-10 trend-score gainers"] += [(t, dt) for t in ch.nlargest(10).index]
            der["Morning Pulse: top-10 trend-score losers"] += [(t, dt) for t in ch.nsmallest(10).index]
        if "volume_signal_score" in d:
            v = pd.to_numeric(d.volume_signal_score, errors="coerce")
            der["Morning Pulse: accumulation leaders (top 10)"] += [(t, dt) for t in v.nlargest(10).index]
    claims = {YARDSTICK: "plain momentum: RS > 0, above the 50-day, liquid",
              "Morning Pulse: top-10 trend-score gainers": "labelled 'turnaround candidates'",
              "Morning Pulse: top-10 trend-score losers": "labelled 'caution / exit watch'",
              "Morning Pulse: accumulation leaders (top 10)": "highest volume-accumulation score"}
    for name, lst in der.items():
        if lst:
            r = _score(name, pd.DataFrame(lst, columns=["ticker", "date"]), F, idx, first_days=0)
            r.update(where="Trend Scanner" if name == YARDSTICK else "Morning Pulse", claim=claims[name])
            rows.append(r)
    S = pd.DataFrame(rows)
    if S.empty:
        return S
    base = S[S.signal == YARDSTICK].iloc[0] if (S.signal == YARDSTICK).any() else None
    S["verdict"] = [("YARDSTICK" if r.signal == YARDSTICK else _verdict(r, base)) for _, r in S.iterrows()]
    S["as_of"], S["sessions"] = idx[-1].strftime("%Y-%m-%d"), len(idx)
    if save:
        S.to_csv(os.path.join(root, OUT), index=False)
        try:
            B = portfolio_benchmark(P, extra, root)
            if not B.empty:
                B.to_csv(os.path.join(root, BENCH_OUT), index=False)
        except Exception as e:
            print(f"[scorecard] portfolio benchmark: {e}")
    return S


def portfolio_benchmark(P: pd.DataFrame, extra: dict, root: str = ".") -> pd.DataFrame:
    """Model portfolio equity vs the equal-weight universe and the 50 largest stocks (cap-weighted), base 100."""
    f = os.path.join(root, "data/dna3_equity_curve.csv")
    if not os.path.exists(f):
        return pd.DataFrame()
    e = pd.read_csv(f, parse_dates=["Date"]).drop_duplicates("Date").set_index("Date")
    r = P.pct_change().clip(-0.25, 0.25)
    out = pd.DataFrame({"equal_weight": (1 + r.mean(axis=1).fillna(0)).cumprod()})
    last = extra[max(extra)]
    if "marketCap" in last:
        m = pd.to_numeric(last["marketCap"], errors="coerce").dropna().nlargest(50)
        m = m[m.index.isin(r.columns)]
        out["top50"] = (1 + (r[m.index].fillna(0) * m / m.sum()).sum(axis=1)).cumprod()
    j = e.index.intersection(out.index)
    if len(j) < 10:
        return pd.DataFrame()
    out = out.loc[j] / out.loc[j[0]] * 100
    out.insert(0, "model", e.Equity[j] / e.Equity[j[0]] * 100)
    out.insert(1, "cash_pct", (e.Cash[j] / e.Equity[j] * 100).round(1))
    out.insert(2, "holdings", e.Holdings[j])
    return out.round(2).rename_axis("date").reset_index()


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    S = build()
    print(S[["signal", "signals", "avg_5d", "avg_21d", "hit_21d", "n_21d", "avg_63d", "hit_63d", "n_63d", "verdict"]].to_string(index=False))
