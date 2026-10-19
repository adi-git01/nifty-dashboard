"""
Where does valuation help, outside the momentum factors?
========================================================

The entry/exit study found valuation useless as a 1-6 month timing or selection tool inside the momentum
book. This tests the six places it plausibly does help, on the same point-in-time data and data fixes as
entry_exit_study.py (full-EQ point-in-time top-1000 universe, fundamentals for every past member, unadjusted
bonus/split repair, results lagged 92 days, P/E = split-adjusted price / that year's EPS).

  1 Long horizons   1 / 2 / 3-year return vs the typical stock by valuation zone, own-history and
                    peer-relative P/E quintile, and absolute P/E band (monthly starts).
  2 Downside        in every broad-market crash (equal-weight universe down >= 15% peak to trough): the
                    peak-to-trough return by valuation zone at the peak; and, for all starts, the worst
                    drawdown over the next year and the odds of a -50% fall within 2 years.
  3 Extremes        P/E >= 100 (profitable), P/E 60-100, loss-makers, own >= 95th pct and >= 2x peers.
  4 Value sleeve    30 cheapest profitable names (mean of own-history and peer-relative P/E percentile)
                    among the 500 most liquid, rebalanced every 21 sessions, 0.4% round-trip cost; alone,
                    and blended 70/30 and 50/50 with the live engine (ranking_rotation_backtest.simulate,
                    6-month RS, as in entry_filter_portfolio_backtest.py). Blends rebalance daily (approx).
  5 Cyclicals and   commodity producers: P/E on 5-year average EPS (normalised) vs trailing P/E;
    lenders         banks/NBFCs: P/B and P/B / ROE.
  6 Turnarounds     cheap (own %ile <= 30 or <= 0.9x peers) AND earnings recovering (EPS +25% after a
                    fall, or back to profit) vs cheap alone.

Halves: starts before / from 1 Jul 2020 (valuation needs 2 years of own P/E history, so starts begin mid-2017).

Pre-registered (each must hold in BOTH halves unless stated; a cell needs >= 200 rows from >= 30 stocks):
  V1  CHEAP beats RICH by >= 6 pp at 2 years (median vs typical; ~3 pp a year).
  V2  In crash windows RICH falls >= 3 pp more than CHEAP/FAIR (median), on average AND in most windows.
  V3  Profitable P/E >= 100 and loss-makers each trail the typical stock by >= 5 pp at 2 years.
  V4  A 70/30 or a 50/50 engine/value blend beats the engine alone on Sharpe AND max drawdown over the
      common window AND in >= 2 of 3 periods (each blend judged on its own).
  V5a Commodities: cheapest-minus-dearest quintile on NORMALISED P/E >= 3 pp at 1 year, and larger than
      the same spread on trailing P/E.
  V5b Lenders: cheapest-minus-dearest quintile on P/B / ROE >= 3 pp at 1 year.
  V6  Cheap + earnings recovery beats cheap alone by >= 3 pp at 1 year.

Known limits: delisted companies are missing, which flatters cheap stocks (the value traps that went bust
are not here) -- a cheap-side win is weaker evidence than a cheap-side loss. Annual results only (quarterly
data starts 2023), so earnings can be up to ~15 months stale. ~6 years of start dates for 2-3 year horizons.

Run: python valuation_study.py            (Actions: ranking_rotation_backtest.yml, script=valuation)
     python valuation_study.py --synthetic
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

OUT = "analysis"
HALF = pd.Timestamp("2020-07-01")
HZ = {252: "1y", 504: "2y", 756: "3y"}
MIN_N, MIN_STOCKS = 200, 30
COST = 0.004


# ----------------------------------------------------------------------------
# panels
# ----------------------------------------------------------------------------
def fy_extended(A, fy):
    """fy_table() keeps np/eps; add equity, reserves and ROE for the same (non-holdco) names."""
    F = A[A.field.isin(["np", "eps", "eq_cap", "reserves", "roe"])].pivot_table(
        index=["ticker", "period"], columns="field", values="value", aggfunc="last").reset_index()
    out = {}
    for t, g in F.groupby("ticker"):
        if t in fy:
            out[t] = (g.sort_values("period").set_index("period"), fy[t][1])
    return out


def extra_valuation(raw_w, fx, V):
    from entry_exit_study import step_panel
    dates, tickers = raw_w.index, raw_w.columns
    P = {k: step_panel(fx, k, dates, tickers) for k in ("eps", "np", "eq_cap", "reserves", "roe")}
    book_ps = ((P["eq_cap"] + P["reserves"]) * P["eps"] / P["np"]).where((P["np"] > 0) & (P["eps"] > 0))
    pb = (raw_w / book_ps).where(book_ps > 0)
    hist = [step_panel(fx, "eps", dates, tickers, k) for k in range(5)]
    stack = np.stack([h.values for h in hist])
    n_ok = np.sum(~np.isnan(stack), axis=0)
    eps5 = pd.DataFrame(np.where(n_ok >= 3, np.nanmean(stack, axis=0), np.nan), index=dates, columns=tickers)
    npe = (raw_w / eps5).where(eps5 > 0).clip(upper=500)
    e1, e2 = hist[1], hist[2]
    g_now = (P["eps"] / e1 - 1).where(e1 > 0)
    g_prev = (e1 / e2 - 1).where(e2 > 0)
    recovery = ((g_now >= 0.25) & (g_prev < 0)) | ((e1 <= 0) & (P["eps"] > 0))
    return dict(pb=pb, pb_roe=(pb / P["roe"]).where(P["roe"] > 0), npe=npe, recovery=recovery, roe=P["roe"],
                eps=P["eps"])


def build(close, raw, vol, pit, fy, fx, groups):
    from entry_exit_study import valuation, family_of
    cf = close.ffill(limit=5)
    sidx = np.arange(260, len(close), 5)                    # weekly, as the own-history window expects
    dates_w = close.index[sidx]
    pit_w = pit.iloc[sidx]
    V = valuation(raw.iloc[sidx], pit_w, fy, groups)
    X = extra_valuation(raw.iloc[sidx], fx, V)
    turn = (close * vol).rolling(60, min_periods=30).median()
    liq_rank = turn.where(pit).rank(axis=1, ascending=False)
    mrank = V["mcap"].where(pit_w).rank(axis=1, ascending=False)
    tier = pd.DataFrame(np.select([mrank <= 100, mrank <= 250, mrank <= 500, mrank.notna()],
                                  ["Large", "Mid", "Small", "Micro"], ""), index=dates_w, columns=close.columns)

    fwd = {}
    for h in HZ:
        r = (cf.shift(-h) / close - 1) * 100
        fwd[h] = (r, r.sub(r.where(pit).median(axis=1), axis=0))
    path_min = {h: (cf[::-1].rolling(h, min_periods=h - 10).min()[::-1].shift(-1) / close - 1).where(cf.shift(-h).notna()) * 100
                for h in (252, 504)}

    m = np.arange(0, len(sidx), 4)                          # monthly starts out of the weekly grid
    keep = pit_w.values[m]
    ii, jj = np.where(keep)
    di, wk = sidx[m][ii], m[ii]
    R = pd.DataFrame(dict(i=di, j=jj, date=close.index[di], ticker=close.columns[jj]))
    R["half"] = np.where(R.date < HALF, "2017-20", "2020-26")
    R["year"] = R.date.dt.year
    for k in ("pe", "own", "rel", "zone"):
        R[k] = V[k].values[wk, jj]
    for k in ("pb", "pb_roe", "npe", "recovery", "roe"):
        R[k] = X[k].values[wk, jj]
    R["tier"] = tier.values[wk, jj]
    R["liq_rank"] = liq_rank.values[di, jj]
    R["family"] = R.ticker.map(family_of(groups)).fillna("industrial")
    R["group"] = R.ticker.map(groups).fillna("")
    for h, lab in HZ.items():
        R[f"ret_{lab}"], R[f"vu_{lab}"] = fwd[h][0].values[di, jj], fwd[h][1].values[di, jj]
    R["dd_1y"] = path_min[252].values[di, jj]
    R["dd_2y"] = path_min[504].values[di, jj]
    R = R[(R.date >= pd.Timestamp("2017-07-01")) & (R.zone != "")]
    return R, dict(cf=cf, sidx=sidx, V=V, X=X, liq_rank=liq_rank, pit=pit, tier=tier)


# ----------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------
def table(R, cells, title, horizons=("1y", "2y", "3y")):
    rows = []
    for half in ("2017-20", "2020-26"):
        e = R[R.half == half]
        for name, m in cells.items():
            x = e[m.reindex(e.index, fill_value=False)]
            r = dict(table=title, half=half, cell=name, n=len(x), stocks=x.ticker.nunique())
            for h in horizons:
                v = x[f"vu_{h}"].dropna()
                if len(v) >= 30:
                    r[f"{h} median vs typ"] = round(v.median(), 1)
                    r[f"{h} mean vs typ"] = round(v.mean(), 1)
                    r[f"{h} beat%"] = round(100 * (v > 0).mean())
            if x.dd_1y.notna().sum() >= 30:
                r["worst DD 1y median"] = round(x.dd_1y.median(), 1)
                r["-50% within 2y %"] = round(100 * (x.dd_2y.dropna() <= -50).mean(), 1)
            rows.append(r)
    return pd.DataFrame(rows)


def quintile_cells(R, col, label, base_mask=None):
    base = pd.Series(True, index=R.index) if base_mask is None else base_mask
    q = R[col].where(base).groupby(R.date).rank(pct=True)
    return {f"{label} Q{k} ({'cheapest' if k == 1 else 'dearest' if k == 5 else ''})".replace(" ()", ""):
            base & q.gt((k - 1) / 5) & q.le(k / 5) for k in range(1, 6)}


def cell(T, title, name, half, col):
    x = T[(T.table == title) & (T.cell == name) & (T.half == half)]
    if not len(x) or col not in x or pd.isna(x[col].iloc[0]) or x.n.iloc[0] < MIN_N or x.stocks.iloc[0] < MIN_STOCKS:
        return np.nan
    return x[col].iloc[0]


def crash_windows(cf, pit, min_dd=15.0):
    """Peak -> trough windows of the equal-weight point-in-time universe with a fall >= min_dd %.
    A window runs from a new high to the next new high (or the end); its trough is the low in between."""
    ew = (1 + cf.pct_change(fill_method=None).where(pit).mean(axis=1).fillna(0)).cumprod()
    peaks = np.flatnonzero(ew.values >= ew.cummax().values)
    out = []
    for a, b in zip(peaks, list(peaks[1:]) + [len(ew)]):
        if b - a < 2:
            continue
        seg = ew.iloc[a:b]
        t = seg.idxmin()
        dd = (seg.min() / ew.iloc[a] - 1) * 100
        if dd <= -min_dd:
            out.append((ew.index[a], t, round(dd, 1)))
    return out


# ----------------------------------------------------------------------------
# value sleeve and blends
# ----------------------------------------------------------------------------
def value_sleeve(P, n=30, liquid=500, every=21, cheap=True):
    cf, V, sidx = P["cf"], P["V"], P["sidx"]
    rets = cf.pct_change(fill_method=None).fillna(0).clip(-0.5, 1.0)
    dates = cf.index
    wk_of = np.searchsorted(sidx, np.arange(len(dates)), side="right") - 1
    start = int(np.searchsorted(dates, pd.Timestamp("2017-07-01")))
    eq, w, curve, hold = 1.0, None, [], []
    for i in range(start, len(dates)):
        if w is not None:
            day = rets.iloc[i].reindex(w.index).fillna(0).values
            g = float(np.dot(w.values, day))
            eq *= 1 + g
            w = w * (1 + day) / (1 + g) if (1 + g) != 0 else w
        if (i - start) % every == 0 and wk_of[i] >= 0:
            k = wk_of[i]
            own, rel, pe = V["own"].iloc[k], V["rel"].iloc[k], V["pe"].iloc[k]
            ok = P["liq_rank"].iloc[i].le(liquid) & pe.gt(0) & pe.lt(200) & own.notna() & rel.notna()
            if ok.sum() >= n:
                score = (own[ok].rank(pct=True) + rel[ok].rank(pct=True)) / 2
                pick = (score.nsmallest(n) if cheap else score.nlargest(n)).index
                new = pd.Series(1.0 / n, index=pick)
                idx = new.index if w is None else new.index.union(w.index)
                old_w = pd.Series(0.0, index=idx) if w is None else w.reindex(idx, fill_value=0)
                turnover = (new.reindex(idx, fill_value=0) - old_w).abs().sum()
                eq *= 1 - COST / 2 * turnover
                w = new
                hold.append(list(pick))
        curve.append((dates[i], eq))
    return pd.Series(dict(curve)), hold


def perf(eq):
    eq = eq.dropna()
    if len(eq) < 60:
        return dict(cagr=np.nan, sharpe=np.nan, maxdd=np.nan)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    r = eq.pct_change().dropna()
    return dict(cagr=round(((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100, 1),
                sharpe=round(r.mean() / r.std() * np.sqrt(252), 2) if r.std() > 0 else np.nan,
                maxdd=round((eq / eq.cummax() - 1).min() * 100, 1))


def engine_curve(close, vol, nifty_df, pit):
    from exit_rule_backtest import build_rs_panel
    from ranking_rotation_backtest import DEFAULTS, RS_CAP, build_keys, simulate
    dates = close.index
    base = build_rs_panel(close, nifty_df, dates).where(lambda x: x.abs() <= RS_CAP).where(pit)
    keys = build_keys(close, nifty_df, dates, base)
    curve, _ = simulate("BASE", {**DEFAULTS, "key": "rs126"}, close, vol, nifty_df, dates, base, keys["rs126"],
                        close.notna().cumsum())
    return curve.set_index("date")["equity"]


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    tag = "_synthetic" if args.synthetic else ""
    import entry_exit_study as es

    if args.synthetic:
        close, high, low, vol, nifty, pit, gscore, A, groups = es.synthetic_inputs()
        raw = close.copy()
        nifty_df = pd.DataFrame({"Close": nifty, "High": nifty * 1.01, "Low": nifty * 0.99})
    else:
        from utils.nifty1000_list import SUB_INDUSTRY_MAP
        from utils.yf_safe import safe_history
        close, high, low, vol, raw, nifty, pit = es.load_prices(args.max_tickers)
        ever = pit.iloc[260::5]
        ever = list(ever.columns[ever.loc[ever.index >= "2015-06-01"].any()])
        A = es.load_annual(ever, 0, 1.0, 0)                 # scraped by entry_exit_study; no new scrape here
        groups = {t: g for t, g in SUB_INDUSTRY_MAP.items() if g and g != "Unknown"}
        if os.path.exists(es.EXTRA_IND):
            extra = pd.read_csv(es.EXTRA_IND).dropna()
            groups.update({t: g for t, g in zip(extra.ticker, extra.industry) if t not in groups and g})
        close, high, low, vol, raw = es.fix_corporate_actions(close, high, low, vol, raw, A)
        nifty_df = safe_history("^NSEI", start="2014-06-01")
        if nifty_df.index.tz is not None:
            nifty_df.index = nifty_df.index.tz_localize(None)
        nifty_df = nifty_df.reindex(close.index).ffill()
    close, high, low, vol, raw, _ = es.clean_spikes(close, high, low, vol, raw)
    fy = es.fy_table(A)
    fx = fy_extended(A, fy)
    R, P = build(close, raw, vol, pit, fy, fx, groups)
    print(f"[rows] {len(R)} monthly stock-starts with a valuation, {R.ticker.nunique()} stocks, "
          f"{R.date.min():%b %Y} - {R.date.max():%b %Y}", flush=True)
    tables, verdict = [], {}
    m = lambda s: s.fillna(False) if hasattr(s, "fillna") else s

    # ---- 1 long horizons -------------------------------------------------------
    c1 = {"all": pd.Series(True, index=R.index)}
    c1.update({f"zone {z}": R.zone == z for z in ("CHEAP", "FAIR", "RICH", "LOSS")})
    c1.update(quintile_cells(R, "own", "own-history P/E", R.pe.notna()))
    c1.update(quintile_cells(R, "rel", "P/E vs peers", R.pe.notna()))
    for lo, hi in ((0, 10), (10, 20), (20, 35), (35, 60), (60, 100), (100, 501)):
        c1[f"P/E {lo}-{hi if hi < 501 else '+'}"] = m(R.pe.ge(lo) & R.pe.lt(hi))
    T1 = table(R, c1, "1 long horizons")
    tables.append(T1)
    print(f"\n{'=' * 150}\n1  LONG HORIZONS: return vs the typical stock (pp, cumulative over the horizon), monthly starts\n{'=' * 150}")
    print(T1.drop(columns="table").to_string(index=False))
    for h in ("2017-20", "2020-26"):
        a, b = cell(T1, "1 long horizons", "zone CHEAP", h, "2y median vs typ"), cell(T1, "1 long horizons", "zone RICH", h, "2y median vs typ")
        verdict[f"V1 {h}: CHEAP {a} >= RICH {b} + 6 (2y)"] = None if np.isnan(a) or np.isnan(b) else a >= b + 6

    # ---- 2 downside ------------------------------------------------------------
    W = crash_windows(P["cf"], P["pit"])
    W = [w for w in W if w[0] >= pd.Timestamp("2017-07-01")]
    rows = []
    cf = P["cf"]
    for pk, tr, dd in W:
        x = R[R.date <= pk].sort_values("date").groupby("ticker").tail(1)
        x = x[x.date >= pk - pd.Timedelta(days=45)]
        x = x.assign(crash=(cf.loc[tr, x.ticker].values / cf.loc[pk, x.ticker].values - 1) * 100)
        i_tr = cf.index.get_loc(tr)
        rec_i = min(i_tr + 126, len(cf) - 1)
        x = x.assign(rebound=(cf.iloc[rec_i][x.ticker].values / cf.loc[tr, x.ticker].values - 1) * 100)
        for z in ("CHEAP", "FAIR", "RICH", "LOSS"):
            y = x[x.zone == z]
            rows.append(dict(peak=pk.date(), trough=tr.date(), universe_dd=dd, zone=z, n=len(y),
                             crash_median=round(y.crash.median(), 1) if len(y) else np.nan,
                             rebound_6m_median=round(y.rebound.median(), 1) if len(y) else np.nan))
    T2 = pd.DataFrame(rows)
    tables.append(T2.assign(table="2 crash windows"))
    print(f"\n{'=' * 150}\n2  CRASH WINDOWS (equal-weight universe down >= 15%): median peak-to-trough return by zone at the "
          f"peak, and the next 6 months from the trough\n{'=' * 150}")
    print(T2.to_string(index=False) if len(T2) else "no window")
    if len(T2):
        pv = T2.pivot_table(index=["peak", "trough"], columns="zone", values="crash_median")
        if {"RICH", "CHEAP", "FAIR"} <= set(pv.columns):
            gap = pv["RICH"] - pv[["CHEAP", "FAIR"]].mean(axis=1)
            print(f"   RICH minus CHEAP/FAIR per window: {gap.round(1).to_dict()}")
            verdict[f"V2: RICH falls >= 3 pp more -- mean gap {gap.mean():.1f}, windows {int((gap <= -3).sum())}/{len(gap)}"] = \
                bool(gap.mean() <= -3 and (gap <= -3).sum() > len(gap) / 2)
    TD = table(R, {f"zone {z}": R.zone == z for z in ("CHEAP", "FAIR", "RICH", "LOSS")} |
               {"all": pd.Series(True, index=R.index)}, "2 drawdown risk", horizons=("1y",))
    tables.append(TD)
    print(f"\n--- worst drawdown in the next year and odds of a -50% fall within 2 years, by zone\n"
          f"{TD.drop(columns='table').to_string(index=False)}")

    # ---- 3 extremes --------------------------------------------------------------
    c3 = {"all": pd.Series(True, index=R.index), "P/E >= 100 (profitable)": m(R.pe.ge(100) & R.pe.lt(200)),
          "P/E 60-100": m(R.pe.ge(60) & R.pe.lt(100)), "loss-makers / P/E >= 200": R.zone == "LOSS",
          "own >= 95th pct and >= 2x peers": m(R.own.ge(95) & R.rel.ge(2))}
    T3 = table(R, c3, "3 extremes")
    tables.append(T3)
    print(f"\n{'=' * 150}\n3  EXPENSIVE EXTREMES\n{'=' * 150}\n{T3.drop(columns='table').to_string(index=False)}")
    for h in ("2017-20", "2020-26"):
        for nm in ("P/E >= 100 (profitable)", "loss-makers / P/E >= 200"):
            v = cell(T3, "3 extremes", nm, h, "2y median vs typ")
            verdict[f"V3 {h}: {nm} {v} <= -5 (2y)"] = None if np.isnan(v) else v <= -5

    # ---- 4 value sleeve + blends ---------------------------------------------------
    print(f"\n{'=' * 150}\n4  VALUE SLEEVE (30 cheapest of the 500 most liquid, monthly) AND BLENDS WITH THE LIVE ENGINE\n{'=' * 150}", flush=True)
    vs, hold = value_sleeve(P)
    rich_s, _ = value_sleeve(P, cheap=False)
    eng = engine_curve(close, vol, nifty_df, pit).reindex(vs.index).ffill()
    ew_ret = cf.pct_change(fill_method=None).where(P["liq_rank"].le(500)).mean(axis=1).fillna(0)
    ew = (1 + ew_ret.reindex(vs.index).fillna(0)).cumprod()
    re_, rv = eng.pct_change().fillna(0), vs.pct_change().fillna(0)
    curves = {"live engine": eng, "value sleeve (30 cheapest)": vs, "expensive sleeve (30 dearest)": rich_s,
              "equal-weight top-500": ew,
              "blend 70/30 engine/value": (1 + 0.7 * re_ + 0.3 * rv).cumprod(),
              "blend 50/50 engine/value": (1 + 0.5 * re_ + 0.5 * rv).cumprod()}
    edges = [vs.index[0], pd.Timestamp("2019-12-31"), pd.Timestamp("2022-12-31"), vs.index[-1]]
    prow = []
    for name, c in curves.items():
        row = dict(curve=name, **{f"full {k}": v for k, v in perf(c).items()})
        for a, b in zip(edges, edges[1:]):
            p = perf(c[(c.index >= a) & (c.index <= b)])
            row.update({f"{a:%Y}-{b:%y} {k}": v for k, v in p.items()})
        prow.append(row)
    T4 = pd.DataFrame(prow)
    tables.append(T4.assign(table="4 value sleeve"))
    print(T4.to_string(index=False))
    mo = pd.DataFrame({"engine": eng, "value": vs}).resample("ME").last().pct_change().dropna()
    print(f"   monthly return correlation engine vs value sleeve: {mo.corr().iloc[0, 1]:.2f}; "
          f"average names held {np.mean([len(h) for h in hold]) if hold else 0:.0f}")
    base = T4.set_index("curve").loc["live engine"]
    for bl in ("blend 70/30 engine/value", "blend 50/50 engine/value"):
        b = T4.set_index("curve").loc[bl]
        per = [c for c in T4.columns if c.endswith(" sharpe") and not c.startswith("full")]
        wins = sum(bool(b[c] > base[c] and b[c.replace("sharpe", "maxdd")] > base[c.replace("sharpe", "maxdd")]) for c in per)
        verdict[f"V4 {bl}: Sharpe {b['full sharpe']} vs {base['full sharpe']}, max DD {b['full maxdd']} vs "
                f"{base['full maxdd']}, periods {wins}/{len(per)}"] = bool(
            b["full sharpe"] > base["full sharpe"] and b["full maxdd"] > base["full maxdd"] and wins >= 2)

    # ---- 5 cyclicals and lenders ------------------------------------------------------
    com = R.family == "commodity"
    lend = R.group.isin(["Banks", "Finance"])
    c5 = {}
    c5.update(quintile_cells(R, "npe", "commodity, normalised P/E", com & R.npe.notna() & R.pe.notna()))
    c5.update(quintile_cells(R, "pe", "commodity, trailing P/E", com & R.npe.notna() & R.pe.notna()))
    c5.update(quintile_cells(R, "pb", "lender, P/B", lend & R.pb_roe.notna()))
    c5.update(quintile_cells(R, "pb_roe", "lender, P/B / ROE", lend & R.pb_roe.notna()))
    T5 = table(R, c5, "5 cyclicals and lenders", horizons=("1y", "2y"))
    tables.append(T5)
    print(f"\n{'=' * 150}\n5  CYCLICALS (normalised vs trailing P/E) AND LENDERS (P/B, P/B / ROE)\n{'=' * 150}")
    print(T5.drop(columns="table").to_string(index=False))
    for h in ("2017-20", "2020-26"):
        sp = lambda lab: cell(T5, "5 cyclicals and lenders", f"{lab} Q1 (cheapest)", h, "1y median vs typ") - \
            cell(T5, "5 cyclicals and lenders", f"{lab} Q5 (dearest)", h, "1y median vs typ")
        n_, t_ = sp("commodity, normalised P/E"), sp("commodity, trailing P/E")
        verdict[f"V5a {h}: normalised spread {n_:.1f} >= 3 and > trailing {t_:.1f}"] = \
            None if np.isnan(n_) or np.isnan(t_) else (n_ >= 3 and n_ > t_)
        l_ = sp("lender, P/B / ROE")
        verdict[f"V5b {h}: lender P/B/ROE spread {l_:.1f} >= 3"] = None if np.isnan(l_) else l_ >= 3

    # ---- 6 turnarounds ----------------------------------------------------------------
    cheap = m(R.own.le(30) | R.rel.le(0.9)) & R.pe.notna()
    c6 = {"all": pd.Series(True, index=R.index), "cheap": cheap, "cheap + earnings recovery": cheap & m(R.recovery),
          "earnings recovery (any price)": m(R.recovery), "cheap, no recovery": cheap & ~m(R.recovery)}
    T6 = table(R, c6, "6 turnarounds", horizons=("1y", "2y"))
    tables.append(T6)
    print(f"\n{'=' * 150}\n6  TURNAROUNDS: cheap + earnings recovering\n{'=' * 150}\n{T6.drop(columns='table').to_string(index=False)}")
    for h in ("2017-20", "2020-26"):
        a, b = cell(T6, "6 turnarounds", "cheap + earnings recovery", h, "1y median vs typ"), cell(T6, "6 turnarounds", "cheap", h, "1y median vs typ")
        verdict[f"V6 {h}: cheap+recovery {a} >= cheap {b} + 3 (1y)"] = None if np.isnan(a) or np.isnan(b) else a >= b + 3

    # ---- verdicts ---------------------------------------------------------------------
    print(f"\n{'=' * 150}\nPRE-REGISTERED (both halves unless stated; TOO FEW = cell under {MIN_N} rows / {MIN_STOCKS} stocks)\n{'=' * 150}")
    for k, v in verdict.items():
        print(f"  {'TOO FEW' if v is None else 'PASS' if v else 'FAIL'}  {k}")
    for hyp in ("V1", "V2", "V3", "V4 blend 70/30", "V4 blend 50/50", "V5a", "V5b", "V6"):
        vs_ = [v for k, v in verdict.items() if k.startswith(hyp)]
        if vs_:
            print(f"  => {hyp}: {'TOO FEW DATA' if None in vs_ else 'SUPPORTED' if all(vs_) else 'NOT SUPPORTED'}")
    os.makedirs(OUT, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(f"{OUT}/valuation_study{tag}.csv", index=False)
    print(f"\nsaved -> {OUT}/valuation_study{tag}.csv")


if __name__ == "__main__":
    main()
