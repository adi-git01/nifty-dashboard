"""
What works in the US market? Engine permutations, shorter frames, and valuation
==============================================================================

us_engine_backtest.py found the India engine does not beat SPY. This searches more widely -- and because a
wide search always finds something that looked good in the past, it is built around a HOLDOUT:

  A  Portfolio grid (live simulator, ranking_rotation_backtest.simulate, 15 slots), chosen on 2015-07 ->
     2021-12 ONLY (in-sample), judged on 2022-01 -> today (out-of-sample, never seen by the search).
       ranking key   1-week reversal, 1m, 3m, 6m, 12m, 12-1 momentum, 6m momentum + earnings yield
       rebalance     every 5 or 21 sessions
       exits         live (MA50 + regime trail) / 25% trail only / none (rotate out when rank > 30)
       universe      point-in-time top 200 or top 1000 by dollar volume
       valuation     none / no loss-makers and P/E < 60
       market gate   live breadth gate / SPY above its 200-day average
       entry filter  live (CompRS floor + price > MA50) / none (rank only) -- so reversal and value keys
                     are not filtered out before they are ranked
     672 configs. Pre-registered: the 5 best by in-sample Sharpe go to out-of-sample, with the live rules as
     a reference. A config PASSES only if out-of-sample its Sharpe AND CAGR beat SPY, it beats SPY's Sharpe
     in both halves (2022-23, 2024-26), and its Sharpe beats the best of 8 random-order seeds on its rules.

  B  Signal study (no portfolio), weekly stock-dates, 1/3/6-month return vs the typical stock, eras 2015-20
     and 2021-26, by size (dollar-volume rank 1-100 / 101-500 / 501-1000): momentum by lookback (shorter
     frames), short-term reversal, near the 52-week high, fresh breakout, squeeze release, v6 BUY NOW and
     absorption, valuation zones, P/E bands, earnings yield, loss-makers, momentum x valuation.
     "Works" = in BOTH eras 3m >= +1 pp, 6m >= +2 pp, and above the 95th percentile of 200 random
     whole-stock baskets of the same size.

Valuation, point in time: annual diluted EPS from SEC EDGAR XBRL (company facts) AS FIRST FILED, usable from
its filing date; P/E = actual traded price (split-adjusted close x later split factors, from Yahoo's
split history) / that EPS. Peers = 2-digit SIC from EDGAR submissions. Zones as in India (own 5-year P/E
percentile and P/E vs peers): RICH >= 80th pct and >= 1.25x peers, CHEAP <= 30th pct and <= 0.9x peers,
LOSS EPS <= 0 or P/E >= 200. If EDGAR is unreachable the valuation parts report n/a.

Limits: delisted companies missing (flatters momentum and value alike); dollar volume stands in for size;
foreign filers (20-F) have no US-GAAP EPS and sit outside the valuation cells.

Run: python us_permutation_study.py          (Actions: ranking_rotation_backtest.yml, script=us_perm)
     python us_permutation_study.py --synthetic
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

OUT = "analysis"
START = "2014-06-01"
IS_START, IS_END = pd.Timestamp("2015-07-01"), pd.Timestamp("2021-12-31")
ERA = pd.Timestamp("2021-01-01")
US_COST = 0.0005
UA = os.environ.get("SEC_USER_AGENT", "nifty-dashboard research script (github.com/adi-git01/nifty-dashboard)")
SEC_CACHE = "data/us_sec_eps.csv.gz"
MIN_N, MIN_STOCKS = 200, 30


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def fetch_us_full(tickers, chunk=300):
    """Close (split-adjusted), Adj Close (total return), High, Low, Volume, split factors -- SPY sessions."""
    from utils.yf_safe import safe_download, safe_history
    spy = safe_history("SPY", start=START)
    if spy.index.tz is not None:
        spy.index = spy.index.tz_localize(None)
    sess = spy.index
    P = {k: {} for k in ("close", "adj", "high", "low", "vol", "split")}
    for k in range(0, len(tickers), chunk):
        part = tickers[k:k + chunk]
        bulk = safe_download(part, start=START, group_by="ticker", threads=True, auto_adjust=False, actions=True,
                             min_coverage=0.3)
        for t in part:
            try:
                sub = bulk[t] if len(part) > 1 else bulk
                c = sub["Close"].dropna()
                if len(c) < 250:
                    continue
                f = lambda col: (sub[col].tz_localize(None) if getattr(sub[col].index, "tz", None) else sub[col]).reindex(sess).astype("float32")
                P["close"][t], P["high"][t], P["low"][t], P["vol"][t] = f("Close"), f("High"), f("Low"), f("Volume")
                P["adj"][t] = f("Adj Close") if "Adj Close" in sub else P["close"][t]
                P["split"][t] = (f("Stock Splits") if "Stock Splits" in sub else pd.Series(0.0, index=sess)).fillna(0)
            except Exception:
                continue
        print(f"[data] {min(k + chunk, len(tickers))}/{len(tickers)}, usable {len(P['close'])}", flush=True)
    D = {k: pd.DataFrame(v) for k, v in P.items()}
    cnt = D["close"].notna().sum(axis=1)
    keep = cnt >= 0.8 * cnt.rolling(20, min_periods=5).median().shift(1).fillna(cnt)
    D = {k: v[keep] for k, v in D.items()}
    return D, spy.reindex(D["close"].index)


def raw_price(close, split):
    """Price as actually traded: split-adjusted close x the product of all LATER split ratios."""
    s = split.where(split > 0, 1.0).astype("float64")
    later = s[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    return close * later


def sec_eps(tickers):
    """Annual diluted EPS as first filed (value, period end, filing date) + 2-digit SIC, from EDGAR."""
    import requests
    if os.path.exists(SEC_CACHE):
        E = pd.read_csv(SEC_CACHE, parse_dates=["end", "filed"])
        have = set(E.ticker)
    else:
        E, have = pd.DataFrame(), set()
    todo = [t for t in tickers if t not in have]
    if not todo:
        return E
    h = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
    try:
        m = requests.get("https://www.sec.gov/files/company_tickers.json", headers=h, timeout=60).json()
    except Exception as e:
        print(f"[sec] EDGAR unreachable ({e}) -- valuation cells will be n/a")
        return E
    cik = {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in m.values()}
    rows, n_ok = [], 0
    for i, t in enumerate(todo):
        c = cik.get(t)
        if c is None:
            rows.append(dict(ticker=t, end=pd.NaT, filed=pd.NaT, eps=np.nan, sic=np.nan))
            continue
        try:
            f = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{c:010d}.json", headers=h, timeout=60)
            time.sleep(0.12)
            s = requests.get(f"https://data.sec.gov/submissions/CIK{c:010d}.json", headers=h, timeout=60)
            time.sleep(0.12)
            sic = pd.to_numeric(s.json().get("sic"), errors="coerce") if s.status_code == 200 else np.nan
            facts = f.json().get("facts", {}).get("us-gaap", {}) if f.status_code == 200 else {}
            unit = (facts.get("EarningsPerShareDiluted") or facts.get("EarningsPerShareBasic") or {}).get("units", {})
            vals = unit.get("USD/shares", [])
            d = pd.DataFrame(vals)
            if len(d) and {"start", "end", "filed", "form"} <= set(d.columns):
                d = d[d.form.astype(str).str.startswith("10-K")]
                d["start"], d["end"], d["filed"] = (pd.to_datetime(d[k], errors="coerce") for k in ("start", "end", "filed"))
                d = d[((d.end - d.start).dt.days.between(350, 380))]
                d = d.sort_values("filed").groupby("end").first().reset_index()   # AS FIRST FILED
                for _, r in d.iterrows():
                    rows.append(dict(ticker=t, end=r.end, filed=r.filed, eps=float(r.val), sic=sic))
                n_ok += 1
            else:
                rows.append(dict(ticker=t, end=pd.NaT, filed=pd.NaT, eps=np.nan, sic=sic))
        except Exception:
            continue
        if (i + 1) % 200 == 0:
            print(f"[sec] {i + 1}/{len(todo)} companies, {n_ok} with 10-K EPS", flush=True)
    E = pd.concat([E, pd.DataFrame(rows)], ignore_index=True)
    E.to_csv(SEC_CACHE, index=False)
    print(f"[sec] EPS history for {E.dropna(subset=['eps']).ticker.nunique()} of {len(tickers)} names")
    return E


def eps_panel(E, dates, tickers):
    """Latest annual EPS public by each date (filing date), 'stale' after 18 months; SIC per ticker."""
    cols = {}
    for t, g in E.dropna(subset=["eps", "filed"]).groupby("ticker"):
        if t not in tickers:
            continue
        s = g.sort_values("filed").drop_duplicates("filed", keep="last").set_index("filed").eps
        v = s.reindex(dates, method="ffill")
        age = pd.Series(s.index, index=s.index).reindex(dates, method="ffill")
        v[((pd.Series(dates, index=dates) - age).dt.days > 550).values] = np.nan
        cols[t] = v.values
    sic = E.dropna(subset=["sic"]).drop_duplicates("ticker").set_index("ticker").sic
    return pd.DataFrame(cols, index=dates).reindex(columns=tickers), (sic // 100).reindex(tickers)


def valuation_panels(raw_w, eps_w, sic2, pit_w):
    from entry_exit_study import own_percentile
    pe = (raw_w / eps_w).where(eps_w > 0).clip(upper=500)
    own = own_percentile(pe)
    rel = pd.DataFrame(np.nan, index=pe.index, columns=pe.columns)
    pe_u = pe.where(pit_w)
    for g, mem in sic2.dropna().groupby(sic2.dropna()):
        m = list(mem.index)
        med = pe_u[m].median(axis=1).where(pe_u[m].notna().sum(axis=1) >= 4)
        rel[m] = pe[m].div(med, axis=0)
    zone = pd.DataFrame(np.select([eps_w.le(0) | (pe >= 200), (own >= 80) & (rel >= 1.25), (own <= 30) & (rel <= 0.9),
                                   pe.notna() & own.notna() & rel.notna()], ["LOSS", "RICH", "CHEAP", "FAIR"], ""),
                        index=pe.index, columns=pe.columns)
    ey = (eps_w / raw_w).where(raw_w > 0)
    return dict(pe=pe, own=own, rel=rel, zone=zone, ey=ey)


def synthetic(n=250, days=3000, seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-06-02", periods=days)
    r = rng.normal(0.0004, 0.02, (days, n)) + rng.normal(0.0003, 0.01, (days, 1))
    c = pd.DataFrame(50 * np.exp(np.cumsum(r, 0)), idx, [f"U{k:03d}" for k in range(n)])
    rngd = np.abs(rng.normal(0, 0.012, (days, n)))
    D = dict(close=c, adj=c, high=c * (1 + rngd), low=c * (1 - rngd),
             vol=pd.DataFrame(np.exp(rng.normal(14, 1, (days, n))), idx, c.columns),
             split=pd.DataFrame(0.0, idx, c.columns))
    s = c.mean(axis=1)
    spy = pd.DataFrame({"Close": s, "High": s * 1.005, "Low": s * 0.995})
    rows = []
    for t in c.columns:
        e = rng.uniform(0.5, 5)
        for y in range(2010, 2026):
            e *= np.exp(rng.normal(0.08, 0.3))
            rows.append(dict(ticker=t, end=pd.Timestamp(f"{y}-12-31"), filed=pd.Timestamp(f"{y + 1}-02-20"),
                             eps=e if rng.random() > 0.08 else -abs(e), sic=float(rng.integers(10, 80) * 100)))
    return D, spy, pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# A: portfolio grid with holdout
# ----------------------------------------------------------------------------
def perf(eq):
    eq = eq.dropna()
    if len(eq) < 60:
        return dict(cagr=np.nan, sharpe=np.nan, maxdd=np.nan)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    r = eq.pct_change().dropna()
    return dict(cagr=round(((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100, 2),
                sharpe=round(r.mean() / r.std() * np.sqrt(252), 2) if r.std() > 0 else np.nan,
                maxdd=round((eq / eq.cummax() - 1).min() * 100, 1))


def portfolio_grid(close, vol, spy, pit200, pit1000, ok_val, ey, tables, limit=0):
    import exit_rule_backtest as erb
    import ranking_rotation_backtest as rrb
    from utils import regime_manager as rm

    dates = close.index
    base = erb.build_rs_panel(close, spy, dates).where(lambda x: x.abs() <= rrb.RS_CAP)
    k = rrb.build_keys(close, spy, dates, base)
    c = close
    n = spy["Close"].reindex(dates).ffill()
    r5 = (c / c.shift(5) - 1) * 100 - ((n / n.shift(5) - 1) * 100).values[:, None]
    ey_d = ey.reindex(dates, method="ffill")
    keys = {"1w reversal": -r5.where(base.notna()), "1m": rrb.rel(c, n, 21).where(base.notna()),
            "3m": k["rs63"], "6m": k["rs126"], "12m": k["rs252"], "12-1": k["mom_12_1"],
            "6m + earnings yield": (k["rs126"].rank(axis=1, pct=True) + ey_d.rank(axis=1, pct=True)).where(base.notna())}
    hist = close.notna().cumsum()
    spy_gate = (n > n.rolling(200).mean())
    orig = {kk: dict(v) for kk, v in rm.REGIME_PARAMS.items()}
    spy_eq = n

    def params_for(rebal, exit_, entry="live"):
        p = {kk: dict(v) for kk, v in orig.items()}
        for kk in p:
            if entry != "live":
                p[kk]["min_comp_rs"] = -1e9
            if p[kk]["rebalance_freq"] < 999:
                p[kk]["rebalance_freq"] = rebal
            if exit_ != "live":
                p[kk]["trail_stop"] = 0.25 if exit_ == "trail 25%" else 0.99
        return p

    def run(name, key, elig, rebal, exit_, gate, sub, entry="live", cost=US_COST):
        rrb.BUY_COST, rrb.SELL_COST = 1 + cost, 1 - cost
        rm.REGIME_PARAMS.clear()
        rm.REGIME_PARAMS.update(params_for(rebal, exit_, entry))
        try:
            cfg = {**rrb.DEFAULTS, "key": "_k", "ma50_exit": exit_ == "live", "ma50_entry": entry == "live",
                   "rotate": 30 if exit_ == "none (rotate)" else None,
                   "gate": spy_gate if gate == "SPY > 200d" else None}
            curve, trades = rrb.simulate(name, cfg, close, vol, spy, sub, elig, key, hist)
        finally:
            rm.REGIME_PARAMS.clear()
            rm.REGIME_PARAMS.update(orig)
        return curve.set_index("date")["equity"], len(trades)

    is_dates = dates[(dates >= IS_START) & (dates <= IS_END)]
    oos_dates = dates[dates > IS_END]
    dims = dict(key=list(keys), rebal=[5, 21], exit=["live", "trail 25%", "none (rotate)"],
                universe=["top 200", "top 1000"], valuation=["none", "no loss, P/E < 60"],
                gate=["breadth (live)", "SPY > 200d"],
                entry=["live (CompRS floor + > MA50)", "none (rank only)"])
    combos = list(itertools.product(*dims.values()))
    if limit:
        combos = combos[::max(1, len(combos) // limit)][:limit]
    print(f"\n[grid] {len(combos)} configs on the in-sample window {is_dates[0].date()} -> {is_dates[-1].date()}", flush=True)
    G = []
    for j, (kn, rb, ex, un, va, ga, en) in enumerate(combos):
        pit = pit200 if un == "top 200" else pit1000
        elig = base.where(pit & (ok_val if va != "none" else True))
        key = keys[kn].where(elig.notna())
        eq, nt = run("grid", key, elig, rb, ex, ga, is_dates, entry=en[:4])
        G.append(dict(key=kn, rebal=rb, exit=ex, universe=un, valuation=va, gate=ga, entry=en, trades=nt,
                      **{f"IS {m}": v for m, v in perf(eq).items()}))
        if (j + 1) % 24 == 0:
            print(f"   {j + 1}/{len(combos)}", flush=True)
    G = pd.DataFrame(G).sort_values("IS sharpe", ascending=False)
    live_cfg = dict(key="6m", rebal=13, exit="live", universe="top 1000", valuation="none", gate="breadth (live)",
                    entry="live (CompRS floor + > MA50)")
    spy_is = perf(spy_eq.reindex(is_dates))
    tables.append(G.assign(table="A grid in-sample"))
    print(f"\nIn-sample SPY: CAGR {spy_is['cagr']}%, Sharpe {spy_is['sharpe']}, max DD {spy_is['maxdd']}%")
    print("\nTop 25 configs by in-sample Sharpe:\n" + G.head(25).to_string(index=False))
    print(f"\nShare of the {len(G)} configs beating SPY's in-sample Sharpe: {100 * (G['IS sharpe'] > spy_is['sharpe']).mean():.0f}%")
    print("\nWhich levers matter (median in-sample Sharpe by setting):")
    for dname in dims:
        print(f"   {dname:9s}: " + "  ".join(f"{v}={s:.2f}" for v, s in G.groupby(dname)["IS sharpe"].median().items()))

    # out-of-sample: top 5 + live
    top = G.head(5).to_dict("records") + [dict(**live_cfg)]
    O = []
    halves = [(oos_dates[0], pd.Timestamp("2023-12-31")), (pd.Timestamp("2024-01-01"), oos_dates[-1])]
    spy_o = perf(spy_eq.reindex(oos_dates))
    spy_h = [perf(spy_eq.reindex(oos_dates[(oos_dates >= a) & (oos_dates <= b)])) for a, b in halves]
    for rank, cfgd in enumerate(top, 1):
        pit = pit200 if cfgd["universe"] == "top 200" else pit1000
        elig = base.where(pit & (ok_val if cfgd["valuation"] != "none" else True))
        key = keys[cfgd["key"]].where(elig.notna())
        eq, nt = run("oos", key, elig, cfgd["rebal"], cfgd["exit"], cfgd["gate"], oos_dates, entry=cfgd["entry"][:4])
        p = perf(eq)
        ph = [perf(eq.reindex(oos_dates[(oos_dates >= a) & (oos_dates <= b)])) for a, b in halves]
        rnd = []
        rng = np.random.default_rng(100 + rank)
        for s in range(8):
            rk = pd.DataFrame(rng.random(elig.shape), index=dates, columns=elig.columns).where(elig.notna())
            rnd.append(perf(run("rnd", rk, elig, cfgd["rebal"], cfgd["exit"], cfgd["gate"], oos_dates,
                                entry=cfgd["entry"][:4])[0])["sharpe"])
        ok = (p["sharpe"] > spy_o["sharpe"] and p["cagr"] > spy_o["cagr"]
              and all(a["sharpe"] > b["sharpe"] for a, b in zip(ph, spy_h)) and p["sharpe"] > max(rnd))
        O.append(dict(pick=("live rules" if rank == 6 else f"IS #{rank}"),
                      config=" | ".join(f"{cfgd[x]}" for x in ("key", "rebal", "exit", "universe", "valuation", "gate", "entry")),
                      OOS_cagr=p["cagr"], OOS_sharpe=p["sharpe"], OOS_maxdd=p["maxdd"],
                      sharpe_2022_23=ph[0]["sharpe"], sharpe_2024_26=ph[1]["sharpe"], best_random_sharpe=max(rnd),
                      trades=nt, PASS=bool(ok) if rank < 6 else "-"))
    O = pd.DataFrame(O)
    tables.append(O.assign(table="A out-of-sample"))
    print(f"\n{'=' * 140}\nOUT-OF-SAMPLE {oos_dates[0].date()} -> {oos_dates[-1].date()} (never seen by the search). SPY: CAGR "
          f"{spy_o['cagr']}%, Sharpe {spy_o['sharpe']}, max DD {spy_o['maxdd']}%; halves Sharpe {spy_h[0]['sharpe']} / "
          f"{spy_h[1]['sharpe']}\n{'=' * 140}\n" + O.to_string(index=False))


# ----------------------------------------------------------------------------
# B: signal study
# ----------------------------------------------------------------------------
def signal_study(D, spy, pit1000, V, sidx, tables):
    from breakout_sector_study import since_last
    from forensics_gate_study import stock_features
    from utils.forensics import indicators

    close, high, low, vol = D["adj"], D["high"], D["low"], D["vol"]
    dates = close.index
    cf = close.ffill(limit=5)
    rk = {}
    for lab, h in (("1w", 5), ("1m", 21), ("3m", 63), ("6m", 126), ("12m", 252)):
        rk[lab] = (cf / cf.shift(h)).where(pit1000).rank(axis=1, pct=True) * 100
    rk["12-1"] = (cf.shift(21) / cf.shift(252)).where(pit1000).rank(axis=1, pct=True) * 100
    ma50 = close.apply(lambda s: s.dropna().rolling(50).mean()).reindex(dates)
    ma200 = close.apply(lambda s: s.dropna().rolling(200).mean()).reindex(dates)
    hi252 = close.rolling(252, min_periods=200).max()
    gate1 = (close > ma50) & (close > ma200) & (close >= 0.85 * hi252)
    new52 = close.ge(hi252) & hi252.notna()
    bo = new52 & (since_last(new52) >= 20)
    vmed = vol.rolling(50, min_periods=30).median().shift(1)
    clv = (close - low) / (high - low).where(lambda x: x > 0)
    anchor = (vol >= 3 * vmed) & (close * vol >= 5e6) & (clv >= 0.5)
    fresh_bo = bo.astype(float).rolling(10, min_periods=1).max() > 0
    fresh_bo_anchor = (bo & anchor).astype(float).rolling(10, min_periods=1).max() > 0
    atr, rsi, _, _, _, bbp = indicators(close, high, low)
    m20, s20 = close.rolling(20, min_periods=15).mean(), close.rolling(20, min_periods=15).std()
    sq = ((bbp.rolling(5, min_periods=3).min().shift(1) < 20) & (close > m20 + 2 * s20)).astype(float).rolling(10, min_periods=1).max() > 0
    dvr = (close * vol).rolling(60, min_periods=30).median().where(pit1000).rank(axis=1, ascending=False)
    fwd = {}
    for h in (21, 63, 126):
        r = (cf.shift(-h) / close - 1) * 100
        fwd[h] = r.sub(r.where(pit1000).median(axis=1), axis=0)

    ii, jj = np.where(pit1000.values[sidx])
    di = sidx[ii]
    R = pd.DataFrame(dict(i=di, j=jj, date=dates[di], ticker=close.columns[jj]))
    R["era"] = np.where(R.date < ERA, "2015-20", "2021-26")
    for lab, P in rk.items():
        R[f"rk_{lab}"] = P.values[di, jj]
    for lab, P in dict(gate1=gate1, fresh_bo=fresh_bo, fresh_bo_anchor=fresh_bo_anchor, sq=sq).items():
        R[lab] = P.values[di, jj].astype(bool)
    R["dist_hi"] = (close.values[di, jj] / hi252.values[di, jj] - 1) * 100
    R["rsi"] = rsi.values[di, jj]
    R["dvr"] = dvr.values[di, jj]
    for h in (21, 63, 126):
        R[f"vu{h}"] = fwd[h].values[di, jj]
    if V is not None:
        for k_ in ("pe", "own", "rel", "zone", "ey"):
            R[k_] = V[k_].values[ii, jj]
    else:
        for k_ in ("pe", "own", "rel", "ey"):
            R[k_] = np.nan
        R["zone"] = ""
    R = R[R.date >= pd.Timestamp("2015-07-01")]
    # v6 forensics on Gate 1 rows
    C, Hh, L, Vv, VM, M, A = (x.values for x in (close, high, low, vol, vmed, ma50, atr))
    fx = []
    for j, g in R[R.gate1].groupby("j"):
        for f in stock_features(C[:, j], Hh[:, j], L[:, j], Vv[:, j], VM[:, j], M[:, j], A[:, j], g.i.values):
            fx.append(dict(i=f["i"], j=j, state=f.get("state"), signature=f.get("signature"), supply=f.get("supply")))
    R = R.merge(pd.DataFrame(fx, columns=["i", "j", "state", "signature", "supply"]), on=["i", "j"], how="left")
    R[["state", "signature", "supply"]] = R[["state", "signature", "supply"]].fillna("")
    R["buy_now"] = (R.state == "GO") & (R.supply != "NEAR OVERHANG")
    print(f"\n[signals] {len(R)} weekly stock-dates, {R.ticker.nunique()} stocks; valuation coverage "
          f"{100 * (R.zone != '').mean():.0f}%", flush=True)

    mom = (R.rk_6m >= 90) & R.gate1
    m = lambda s: s.fillna(False) if hasattr(s, "fillna") else s
    rel_q = R.rel.groupby(R.date).rank(pct=True)
    ey_q = R.ey.groupby(R.date).rank(pct=True)
    cells = {"all": pd.Series(True, index=R.index)}
    for lab in ("1w", "1m", "3m", "6m", "12m", "12-1"):
        cells[f"top 10% by {lab} return"] = m(R[f"rk_{lab}"] >= 90)
    cells.update({
        "bottom 10% by 1w (short-term reversal)": m(R["rk_1w"] <= 10),
        "bottom 10% by 1m": m(R["rk_1m"] <= 10),
        "momentum (6m top 10% + Gate 1)": mom,
        "within 5% of 52w high": m(R.dist_hi >= -5), "> 40% below 52w high": m(R.dist_hi < -40),
        "RSI < 30": m(R.rsi < 30),
        "momentum + fresh breakout": mom & R.fresh_bo, "momentum + squeeze release": mom & R.sq,
        "breakout + volume anchor": R.fresh_bo_anchor, "v6 BUY NOW": R.buy_now,
        "v6 ABSORPTION": R.signature == "ABSORPTION", "Gate 1 trend": R.gate1,
        "valuation CHEAP": R.zone == "CHEAP", "valuation FAIR": R.zone == "FAIR", "valuation RICH": R.zone == "RICH",
        "loss-maker / P/E >= 200": R.zone == "LOSS", "P/E < 15": m(R.pe < 15), "P/E 15-30": m(R.pe.between(15, 30)),
        "P/E 30-60": m(R.pe.between(30, 60)), "P/E >= 60": m(R.pe >= 60),
        "cheapest 20% vs SIC peers": m(rel_q <= 0.2), "dearest 20% vs SIC peers": m(rel_q >= 0.8),
        "earnings yield top 20%": m(ey_q >= 0.8),
        "momentum + CHEAP/FAIR": mom & R.zone.isin(["CHEAP", "FAIR"]), "momentum + RICH": mom & (R.zone == "RICH"),
        "momentum, profitable": mom & R.zone.isin(["CHEAP", "FAIR", "RICH"]), "momentum + loss-maker": mom & (R.zone == "LOSS"),
    })
    sizes = {"mega (1-100)": m(R.dvr <= 100), "large (101-500)": m(R.dvr.between(101, 500)), "mid (501-1000)": m(R.dvr > 500)}

    # placebo per era: 95th pct of 6m median of whole-stock random baskets, by size
    rng = np.random.default_rng(9)
    th = {}
    for era in ("2015-20", "2021-26"):
        x = R[(R.era == era) & R.vu126.notna()].sort_values("ticker")
        v = x.vu126.values
        _, st_, ct = np.unique(x.ticker.values, return_index=True, return_counts=True)
        for n in (200, 500, 1000, 2000, 5000, 10000, 20000, 50000):
            if n > len(x):
                continue
            meds = []
            for _ in range(200):
                o = rng.permutation(len(st_))
                kk = int(np.searchsorted(np.cumsum(ct[o]), n)) + 1
                meds.append(np.median(np.concatenate([v[st_[p]:st_[p] + ct[p]] for p in o[:kk]])))
            th[(era, n)] = float(np.percentile(meds, 95))

    def thr(era, n):
        ks = [kk for (e, kk) in th if e == era and kk <= n]
        return th[(era, max(ks))] if ks else np.nan

    rows = []
    for sname, smask in {"all sizes": pd.Series(True, index=R.index), **sizes}.items():
        for cname, cm in cells.items():
            mask = cm.reindex(R.index, fill_value=False) & smask
            r = dict(size=sname, signal=cname)
            ok_all = True
            for era in ("2015-20", "2021-26"):
                x = R[mask & (R.era == era)]
                r[f"{era} n"] = len(x)
                if len(x) >= 30:
                    for h, lab in ((21, "1m"), (63, "3m"), (126, "6m")):
                        r[f"{era} {lab}"] = round(x[f"vu{h}"].median(), 2)
                    t_ = thr(era, len(x))
                    ok = (len(x) >= MIN_N and x.ticker.nunique() >= MIN_STOCKS and r[f"{era} 3m"] >= 1
                          and r[f"{era} 6m"] >= 2 and r[f"{era} 6m"] > t_)
                else:
                    ok = False
                ok_all &= ok
            r["WORKS (both eras)"] = ok_all
            rows.append(r)
    S = pd.DataFrame(rows)
    tables.append(S.assign(table="B signals"))
    print(f"\n{'=' * 150}\nB  SIGNALS: median return vs the typical stock (pp) by era; WORKS = both eras 3m >= +1, 6m >= +2, "
          f"above the placebo 95th pct\n{'=' * 150}")
    for sname in S["size"].unique():
        print(f"\n--- {sname}\n" + S[S["size"] == sname].drop(columns="size").to_string(index=False))
    w = S[S["WORKS (both eras)"]]
    print("\nWORKS IN BOTH ERAS:\n" + ("\n".join(f"  {a}: {b}  (6m {c} / {d})" for a, b, c, d in
                                                 zip(w["size"], w.signal, w["2015-20 6m"], w["2021-26 6m"])) or "  nothing"))


# ----------------------------------------------------------------------------
def main():
    from momentum_factor_backtest import build_pit_universe
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--skip-grid", action="store_true")
    ap.add_argument("--grid-limit", type=int, default=0, help="test runs only: evaluate an even slice of the grid")
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 400)
    tag = "_synthetic" if args.synthetic else ""

    if args.synthetic:
        D, spy, E = synthetic()
    else:
        from us_engine_backtest import us_symbols
        D, spy = fetch_us_full(us_symbols(args.max_tickers))
    D = {k: v.astype("float64") for k, v in D.items()}
    close, vol = D["adj"], D["vol"]
    pit1000 = build_pit_universe(close, vol, 1000)
    ever = pit1000.any()
    D = {k: v.loc[:, ever] for k, v in D.items()}
    close, vol, pit1000 = D["adj"], D["vol"], pit1000.loc[:, ever]
    pit200 = build_pit_universe(close, vol, 200)
    print(f"[universe] {int(ever.sum())} names ever in the point-in-time top 1000", flush=True)
    if not args.synthetic:
        E = sec_eps(list(close.columns))

    # valuation (weekly, as in India)
    sidx = np.arange(260, len(close), 5)
    dates_w = close.index[sidx]
    V, ok_val, ey = None, pd.DataFrame(True, index=close.index, columns=close.columns), \
        pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    if E is not None and len(E) and E.eps.notna().any():
        raw = raw_price(D["close"], D["split"])
        eps_w, sic2 = eps_panel(E, dates_w, list(close.columns))
        V = valuation_panels(raw.iloc[sidx], eps_w, sic2, pit1000.iloc[sidx])
        okw = (V["pe"] > 0) & (V["pe"] < 60) & (V["zone"] != "LOSS")
        ok_val = okw.reindex(close.index, method="ffill").fillna(False).astype(bool)
        ey = V["ey"]
        cov = (V["zone"].where(pit1000.iloc[sidx]) != "").where(pit1000.iloc[sidx]).mean().mean()
        print(f"[valuation] zone available for {100 * cov:.0f}% of point-in-time stock-weeks", flush=True)
    tables = []
    if not args.skip_grid:
        portfolio_grid(close, vol, spy, pit200, pit1000, ok_val, ey, tables, args.grid_limit)
    signal_study(D, spy, pit1000, V, sidx, tables)
    os.makedirs(OUT, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(f"{OUT}/us_permutation_study{tag}.csv", index=False)
    print(f"\nsaved -> {OUT}/us_permutation_study{tag}.csv")


if __name__ == "__main__":
    main()
