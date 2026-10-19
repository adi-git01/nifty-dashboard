"""
Can charts and valuation improve momentum entries and exits?
=============================================================

Weekly stock-dates, 2016-2026, point-in-time top-1000 universe (daily bars from
the full NSE EQ list, so names that later fell out of the top 1000 are included).
Every number is judged against what was knowable THAT day.

Scenarios
  S1 Winners      stocks that went on to gain 80%+ within 6 months. What did
                  valuation, the chart and the core factors look like at the
                  start of the run, and was there a trigger? Answered as LIFT:
                  P(winner | feature) / P(winner), never "x% of winners had y"
                  (most stocks share most features, so that number misleads).
  S2 Momentum     stocks up 50%+ in 6 months and in the top 10% by 6-month
     fades        return. What happened next -- CONTINUED (+20% or more in the
                  next 6 months), SIDEWAYS (time correction, -15% to +20%),
                  PULLBACK (-25% to -15%), GAVE BACK (-25% or worse) -- and
                  were there signs? Plus an exit test: does adding a velocity
                  or valuation exit to the live exit (1 close < MA50 or 15%
                  trail) help, counting both what it saves on the fades and
                  what it costs on the continuers.
  S3 Velocity x   strong 6-month momentum (top 20%) split by 1-month velocity
     valuation    (ACCEL / STEADY / FADING) and valuation zone.
  S4 By size      the same entry and exit rules by point-in-time size tier
     and family   (Large 1-100, Mid 101-250, Small 251-500, Micro 501-1000) and
                  by business family (industrial / lenders / commodity).
  Cases           ~15 named examples per scenario with a dated timeline
                  (analysis/ENTRY_EXIT_CASES.md).

Valuation, point in time (annual results; quarterly only exist from 2023)
  P/E          = split-adjusted (NOT dividend-adjusted) close / EPS of the
                 latest financial year whose results were PUBLIC: period end +
                 92 days (SEBI's 60-day limit plus margin). Screener restates
                 EPS for splits and bonuses, and EPS is attributable and on the
                 shares of its own year, so mergers, buybacks, dilution and
                 minority interests need no share-count guess. Holding
                 companies (< 60% of profit attributable) excluded.
  size tier    = split-adjusted close x today's share count, ranked each week.
  P/E own %ile = this week's P/E vs the stock's own last 5 years of weekly
                 P/Es (>= 2 years needed). P/E vs peers = P/E / median P/E of
                 the same sub-industry that week (>= 4 peers).
  Zone         RICH  own %ile >= 80 and >= 1.25x peers
               CHEAP own %ile <= 30 and <= 0.9x peers
               LOSS  latest year's profit <= 0, or P/E >= 200 (earnings ~0:
                     M&M 2020 read RICH at a capped P/E of 500 off one bad
                     year -- run 2 change, made for that bug, not for results)
               FAIR otherwise.
  This is the engine's evidence (own history, peers, growth), not its
  Monte Carlo verdict: the verdict uses today's peer set and data snapshot and
  cannot be rebuilt honestly for 2017.

Guards against the bugs this repo has hit before
  * survivorship: PIT universe from the full EQ list; fundamentals scraped for
    every name that was ever in the PIT top 1000, not just today's list.
    Delisted companies are still missing (one-directional bias, stated).
  * look-ahead: results lagged 92 days; today's share count only through
    adjusted prices; forward windows must be complete (no partial horizons).
  * corrupt bars: one-day spikes that reverse (ARIHANT +759%) are blanked
    before anything is computed; blank closes are never counted as sessions.
  * tiny earnings: P/E capped at 500, losses are their own zone.
  * outcome selection: every winner/fade feature is compared with the same
    week's base rate; overlapping weekly rows are reported with the number of
    distinct stocks and years, not just rows.
  * typical stock: returns vs the cross-sectional MEDIAN universe stock.
  * multiple testing: the hypotheses below are fixed before the run; a result
    counts only if it holds in BOTH eras (2016-20, 2021-26).

Pre-registered hypotheses (printed PASS / FAIL at the end)
  H1 Cheap starts: winner lift of CHEAP >= 1.2x the lift of RICH (week-matched), both eras.
  H2 Chart trigger: P(winner | fresh 52w/ATH breakout + volume anchor + leader
     industry) >= 1.5x base, both eras.
  H3 Rich momentum: among S2 momentum names, RICH underperforms the rest by
     >= 2 pp at 6 months (vs typical), both eras.
  H4 Velocity x value: within strong momentum, FADING+RICH trails FADING+FAIR
     by >= 2 pp at 6 months, both eras.
  H5 Exits: live exit + a velocity exit, or live exit + a valuation exit, beats
     the live exit alone by >= 0.5 pp mean return vs Nifty per trade, both
     eras (each extra exit judged on its own, not the better of the two).
  A cell needs >= 200 rows from >= 30 distinct stocks or it reports TOO FEW --
  on random data, 20-stock cells "passed" by chance.

Run: python entry_exit_study.py                (Actions: ranking_rotation_backtest.yml, script=entry_exit)
     python entry_exit_study.py --synthetic    (random data + random fundamentals: every lift ~1, offline)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

OUT = "analysis"
ERA_SPLIT = pd.Timestamp("2021-01-01")
H = 126                         # main horizon: 6 months
LAG_DAYS = 92                   # results public ~3 months after the period end
WIN = 0.80                      # S1 winner: +80% within 6 months
EXTRA_FUNDA = "data/fundamentals_annual_extra.csv.gz"
TIERS = [(100, "Large"), (250, "Mid"), (500, "Small"), (1000, "Micro")]


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_prices(max_tickers):
    """
    real_data() of forensics_gate_study.py, but downloads unadjusted bars once and keeps two closes:
      close  -- dividend + split adjusted (total return; every return in this study)
      raw    -- split/bonus adjusted only, the basis Screener's restated EPS is on (every P/E)
    Using the total-return close for P/E would understate past P/Es by the dividends paid since
    and push dividend payers towards "RICH" against their own history.
    """
    from datetime import datetime
    from exit_rule_backtest import align_to_sessions
    from momentum_factor_backtest import build_pit_universe, load_candidates
    from transfer_backtest import cut_partial_last_day
    from utils.yf_safe import safe_download, safe_history

    tickers = load_candidates("all", max_tickers)
    start, end = "2014-06-01", datetime.now().strftime("%Y-%m-%d")
    nifty = safe_history("^NSEI", start=start, end=end)
    if nifty.empty:
        raise SystemExit("Could not fetch Nifty")
    if nifty.index.tz is not None:
        nifty.index = nifty.index.tz_localize(None)
    bulk = safe_download(tickers, start=start, end=end, group_by="ticker", threads=False,
                         auto_adjust=False, min_coverage=0.5)
    fr = {k: {} for k in ("Close", "High", "Low", "Volume", "Raw")}
    for t in tickers:
        try:
            sub = bulk[t]
            raw = sub["Close"]
            adj = sub["Adj Close"] if "Adj Close" in sub else raw
            c = adj.dropna()
            if len(c) <= 250:
                continue
            ratio = (adj / raw).reindex(c.index)
            fr["Close"][t], fr["Raw"][t] = c, raw.reindex(c.index)
            fr["High"][t], fr["Low"][t] = sub["High"].reindex(c.index) * ratio, sub["Low"].reindex(c.index) * ratio
            fr["Volume"][t] = sub["Volume"].reindex(c.index)
        except Exception:
            continue
    P = {k: pd.DataFrame(v) for k, v in fr.items()}
    for k in P:
        if P[k].index.tz is not None:
            P[k].index = P[k].index.tz_localize(None)
    close, vol = align_to_sessions(P["Close"], P["Volume"], nifty)
    close, vol = cut_partial_last_day(close, vol)
    high, low, raw = (P[k].reindex(index=close.index, columns=close.columns) for k in ("High", "Low", "Raw"))
    nifty = nifty["Close"].reindex(close.index).ffill()
    print(f"[data] {close.shape[1]} tickers, {len(close)} sessions {close.index[0].date()} -> {close.index[-1].date()}")
    # NOT "& has an industry": the industry map only covers today's list, so that filter (used by the
    # earlier studies) silently turned the point-in-time universe back into today's survivors
    pit = build_pit_universe(close, vol, 1000)
    return close, high, low, vol, raw, nifty, pit


def industry_scores(close, pit, groups, nifty):
    """The live sub-industry rotation score (0-100) for every stock with a known industry."""
    from transfer_backtest import group_panels
    _, keys, _ = group_panels(close, pit, groups, nifty, [(5, .10), (21, .50), (63, .40)], india_live=True)
    score = keys["live"].rank(axis=1, pct=True) * 100
    score = score.where(score.notna().sum(axis=1) >= 20)
    return pd.DataFrame({t: score[g] if (g := groups.get(t)) in score.columns else np.nan
                         for t in close.columns}, index=close.index)


BONUS_SPLIT = [1 / 2, 1 / 3, 2 / 3, 1 / 4, 3 / 4, 1 / 5, 1 / 6, 1 / 10, 2 / 5, 3 / 5, 4 / 7]


def fix_corporate_actions(close, high, low, vol, raw, A):
    """
    Yahoo sometimes misses an Indian bonus or split (SHARDAMOTR 1:1 bonus, Jan 2025: 1790 -> 861 in a
    day, recorded as a -52% crash). A one-day fall within 2% of a bonus/split ratio is treated as one
    when confirmed by EITHER equity capital rising by ~1/ratio across that year (a bonus) OR volume
    stepping up by >= 0.7/ratio for the next 20 sessions (a split). Earlier bars are rescaled.
    """
    eq = A[A.field == "eq_cap"].pivot_table(index="period", columns="ticker", values="value").sort_index()
    r1 = raw / raw.ffill().shift(1)
    r2 = raw.bfill().shift(-1) / raw
    rc = close / close.ffill().shift(1)
    cand = (r1 < 0.8) & (r2 > 0.8) & ((rc / r1 - 1).abs() < 0.02)   # missing in BOTH series, not just one
    vb = vol.rolling(20, min_periods=10).median()
    va = vol[::-1].rolling(20, min_periods=10).median()[::-1]
    fixed = []
    for dt, t in cand.stack()[lambda x: x].index:
        f_obs = r1.at[dt, t]
        f = min(BONUS_SPLIT, key=lambda x: abs(x / f_obs - 1))
        if abs(f / f_obs - 1) > 0.02:
            continue
        i = close.index.get_loc(dt)
        ok_bonus = False
        if t in eq.columns:
            e = eq[t].dropna()
            before, after = e[e.index < dt], e[e.index >= dt]
            if len(before) and len(after) and before.iloc[-1] > 0:
                ok_bonus = abs((after.iloc[0] / before.iloc[-1]) * f - 1) < 0.15
        vbefore, vafter = vb.iat[i - 1, close.columns.get_loc(t)] if i else np.nan, va.at[dt, t]
        ok_split = bool(vbefore > 0 and vafter / vbefore >= 0.7 / f)
        if not (ok_bonus or ok_split):
            continue
        before = close.index < dt
        for P in (close, high, low, raw):
            P.loc[before, t] *= f
        vol.loc[before, t] /= f
        fixed.append(f"{t[:-3]} {dt:%Y-%m-%d} x{f:.2f} ({'bonus' if ok_bonus else 'split'})")
    print(f"[clean] repaired {len(fixed)} unadjusted bonus/split day(s): " + ", ".join(fixed[:40])
          + (" ..." if len(fixed) > 40 else ""))
    return close, high, low, vol, raw


def clean_spikes(close, high, low, vol, raw):
    """Blank one-day spikes that fully reverse (bad prints), in every panel."""
    r1 = close / close.ffill().shift(1)
    r2 = close.bfill().shift(-1) / close
    spike = ((r1 > 1.5) & (r2 < 0.72)) | ((r1 < 0.6) & (r2 > 1.45))
    n = int(spike.values.sum())
    if n:
        top = spike.sum().sort_values(ascending=False)
        print(f"[clean] blanked {n} one-day reversing spikes; most in {', '.join(top.index[:8].str[:-3])}")
    # a one-day fall of 40%+ that does NOT reverse is either a real crash or a split/bonus Yahoo
    # failed to adjust (both returns and P/E would be wrong) -- listed for review, not guessed at
    drop = (r1 < 0.6) & ~spike & (r2 > 0.8)
    if drop.values.sum():
        d = drop.stack()
        d = d[d]
        print(f"[check] {len(d)} one-day falls of 40%+ that did not reverse (review: crash or unadjusted "
              f"split/bonus?): " + ", ".join(f"{t[:-3]} {dt:%Y-%m-%d}" for dt, t in d.index[:25]))
    return (close.mask(spike), high.mask(spike), low.mask(spike), vol.mask(spike), raw.mask(spike))


EXTRA_IND = "data/industry_extra.csv"


EXTRA_TRIED = "data/fundamentals_extra_tried.txt"


def load_annual(tickers_needed, scrape_missing, delay, minutes):
    """
    Fundamentals (+ Screener broad industry for names outside today's map) for every PIT name.
    The scrape is saved every 50 names and remembers which names it tried, so a job killed by its
    time limit keeps its progress and the next run resumes; it also stops after `minutes`.
    """
    import time
    A = pd.read_csv("data/fundamentals_annual.csv.gz", parse_dates=["period"])
    if os.path.exists(EXTRA_FUNDA):
        A = pd.concat([A, pd.read_csv(EXTRA_FUNDA, parse_dates=["period"])])
    tried = set(open(EXTRA_TRIED).read().split()) if os.path.exists(EXTRA_TRIED) else set()
    have = set(A.ticker)
    missing = [t for t in tickers_needed if t not in have and t not in tried]
    print(f"[funda] {len(have)} stocks on file; {len(missing)} PIT-universe names still to try "
          f"({len(tried)} tried before)")
    if missing and scrape_missing:
        from fundamentals_annual import scrape
        missing = missing[:scrape_missing]
        print(f"[funda] scraping up to {len(missing)} from Screener, budget {minutes} min "
              f"(fixes survivorship in the valuation data)", flush=True)
        ind = {}
        base = pd.read_csv(EXTRA_FUNDA, parse_dates=["period"]) if os.path.exists(EXTRA_FUNDA) else None
        base_ind = pd.read_csv(EXTRA_IND) if os.path.exists(EXTRA_IND) else pd.DataFrame(columns=["ticker", "industry"])

        def save(rows, _q, done):
            X = pd.DataFrame(rows, columns=["ticker", "field", "period", "value", "basis"])
            X["period"] = pd.to_datetime(X["period"])
            X = pd.concat([base, X]) if base is not None else X
            X.drop_duplicates(["ticker", "field", "period"], keep="last").to_csv(EXTRA_FUNDA, index=False)
            new = pd.DataFrame(list(ind.items()), columns=["ticker", "industry"])
            pd.concat([base_ind, new]).drop_duplicates("ticker", keep="last").to_csv(EXTRA_IND, index=False)
            with open(EXTRA_TRIED, "w") as fh:
                fh.write("\n".join(sorted(tried | set(done))))
        try:
            scrape(missing, delay, industries=ind, deadline=time.time() + 60 * minutes, checkpoint=save)
        except SystemExit as e:                     # Screener blocked: carry on, say so
            print(f"[funda] scrape aborted ({e}); valuation limited to names already on file")
        if os.path.exists(EXTRA_FUNDA):
            A = pd.concat([A, pd.read_csv(EXTRA_FUNDA, parse_dates=["period"])])
    return A.drop_duplicates(["ticker", "field", "period"], keep="last")


def fy_table(A):
    """
    Per ticker: annual EPS by period (Screener restates EPS for splits and bonuses -- checked on TCS 2018
    and Bajaj Finance 2016 -- and EPS is attributable and on the shares of its year, so P/E = split-adjusted
    price / EPS handles mergers, buybacks, dilution and minority interests without guessing share counts),
    plus today's share count for the size tier. Holding companies (< 60% of profit attributable) excluded.
    """
    top = A[A.field.isin(["mcap_cr", "price_scr"])].pivot_table(index="ticker", columns="field", values="value",
                                                                 aggfunc="last")
    shares = (top.get("mcap_cr") * 1e7 / top.get("price_scr")).where(top.get("price_scr") > 0).dropna()
    F = A[A.field.isin(["np", "eps"])].pivot_table(index=["ticker", "period"], columns="field", values="value",
                                                   aggfunc="last").reset_index()
    out, st = {}, dict(stocks=0, holdco=0, no_shares=0, bad=0)
    for t, g in F.groupby("ticker"):
        g = g.sort_values("period").set_index("period")
        if "eps" not in g or g.eps.dropna().empty:
            continue
        sh = shares.get(t, np.nan)
        if np.isfinite(sh) and "np" in g:
            r = (g.eps * sh / (g.np * 1e7)).where(g.np > 0).dropna()
            if len(r) and r.iloc[-1] < 0.6:
                st["holdco"] += 1
                continue
            if len(r) and r.iloc[-1] > 1.6:            # EPS and profit disagree: a data error, not a company
                st["bad"] += 1
                continue
        else:
            st["no_shares"] += 1
        st["stocks"] += 1
        out[t] = (g, sh)
    print(f"[funda] EPS history for {st['stocks']} stocks; holdcos excluded {st['holdco']}; EPS/profit mismatch "
          f"excluded {st['bad']}; no share count (no size tier) {st['no_shares']}")
    return out


def step_panel(fy, field, dates, tickers, years_back=0):
    """Value of `field` for the latest FY public by each date (or that FY minus `years_back`)."""
    cols = {}
    for t in tickers:
        if t not in fy:
            continue
        g, _ = fy[t]
        if field not in g:
            continue
        s = g[field]
        if years_back:
            by_year = {p.year: v for p, v in s.items()}
            s = pd.Series([by_year.get(p.year - years_back, np.nan) for p in s.index], index=s.index)
        s.index = s.index + pd.Timedelta(days=LAG_DAYS)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        age = pd.Series(s.index, index=s.index)
        v = s.reindex(dates, method="ffill")
        a = age.reindex(dates, method="ffill")
        stale = (pd.Series(dates, index=dates) - a).dt.days > 550   # no newer results for 18 months
        v[stale.values] = np.nan
        cols[t] = v.values
    return pd.DataFrame(cols, index=dates).reindex(columns=tickers)


def own_percentile(x, window=260, min_obs=104):
    """Percentile of each row's value within its own trailing window (per column)."""
    from numpy.lib.stride_tricks import sliding_window_view
    X = x.values
    out = np.full(X.shape, np.nan)
    pad = np.full((window - 1, X.shape[1]), np.nan)
    Xp = np.vstack([pad, X])
    for j in range(X.shape[1]):
        w = sliding_window_view(Xp[:, j], window)          # rows x window, last element = current
        cur = w[:, -1]
        valid = ~np.isnan(w)
        n = valid.sum(1)
        le = (np.where(valid, w, np.inf) <= cur[:, None]).sum(1)
        ok = (n >= min_obs) & ~np.isnan(cur)
        out[ok, j] = 100 * le[ok] / n[ok]
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def valuation(raw_w, pit_w, fy, groups):
    dates, tickers = raw_w.index, raw_w.columns
    sh = pd.Series({t: fy[t][1] for t in fy}).reindex(tickers)
    eps = step_panel(fy, "eps", dates, tickers)
    eps1 = step_panel(fy, "eps", dates, tickers, 1)
    eps3 = step_panel(fy, "eps", dates, tickers, 3)
    mcap = raw_w * sh
    pe = (raw_w / eps).where(eps > 0).clip(upper=500)
    npa = eps                                   # sign of earnings for the LOSS zone
    g1 = (eps / eps1 - 1).where(eps1 > 0)
    g3 = ((eps / eps3) ** (1 / 3) - 1).where((eps3 > 0) & (eps > 0))
    own = own_percentile(pe)
    rel = pd.DataFrame(np.nan, index=dates, columns=tickers)
    gp = pd.Series(groups).reindex(tickers)
    pe_u = pe.where(pit_w)
    for g, mem in gp.groupby(gp):
        m = list(mem.index)
        med = pe_u[m].median(axis=1).where(pe_u[m].notna().sum(axis=1) >= 4)
        rel[m] = pe[m].div(med, axis=0)
    zone = pd.DataFrame(np.select(
        [npa.le(0) | (pe >= 200), (own >= 80) & (rel >= 1.25), (own <= 30) & (rel <= 0.9),
         pe.notna() & own.notna() & rel.notna()],
        ["LOSS", "RICH", "CHEAP", "FAIR"], ""), index=dates, columns=tickers)
    return dict(pe=pe, own=own, rel=rel, g1=g1, g3=g3, zone=zone, mcap=mcap)


def family_of(groups):
    from utils.valuation import BFSI_GRAN, COMMODITY_GRAN
    lend = BFSI_GRAN | {"Capital Markets", "Insurance"}
    return {t: ("lender" if g in lend else "commodity" if g in COMMODITY_GRAN else "industrial")
            for t, g in groups.items()}


# ----------------------------------------------------------------------------
# features and labels
# ----------------------------------------------------------------------------
def build_rows(close, high, low, vol, raw, nifty, pit, gscore, fy, groups):
    from breakout_sector_study import build
    from utils.forensics import indicators

    ev, conf, fwd, below50, retention, gscore, ma50, ma200 = build(close, high, low, vol, nifty, pit, gscore)
    cf = close.ffill(limit=5)
    hi252 = close.rolling(252, min_periods=200).max()
    r126, r21 = cf / cf.shift(126) - 1, cf / cf.shift(21) - 1
    rk126 = r126.where(pit).rank(axis=1, pct=True) * 100
    rk21 = r21.where(pit).rank(axis=1, pct=True) * 100
    _, rsi, adx, _, _, _ = indicators(close, high, low)
    bo = ev["high52"] | ev["ath"]
    bo_anchor = bo & conf["anchor"]
    fresh_bo = bo.astype(float).rolling(10, min_periods=1).max() > 0
    fresh_bo_anchor = bo_anchor.astype(float).rolling(10, min_periods=1).max() > 0
    gate1 = (close > ma50) & (close > ma200) & (close >= 0.85 * hi252)

    # forward path: full 6-month window required
    full = cf.shift(-H).notna()
    fmax = (cf[::-1].rolling(H, min_periods=H - 10).max()[::-1].shift(-1) / close - 1).where(full)
    fmin = (cf[::-1].rolling(H, min_periods=H - 10).min()[::-1].shift(-1) / close - 1).where(full)

    sidx = np.arange(260, len(close), 5)
    dates_w = close.index[sidx]
    pit_w = pit.iloc[sidx]
    V = valuation(raw.iloc[sidx], pit_w, fy, groups)
    mrank = V["mcap"].where(pit_w).rank(axis=1, ascending=False)
    tier = pd.DataFrame(np.select([mrank <= n for n, _ in TIERS], [l for _, l in TIERS], ""),
                        index=dates_w, columns=close.columns)
    fam = family_of(groups)

    ii, jj = np.where(pit_w.values)
    di = sidx[ii]
    take = lambda P: P.values[di, jj]
    takew = lambda P: P.values[ii, jj]
    R = pd.DataFrame(dict(i=di, j=jj, date=close.index[di], ticker=close.columns[jj]))
    R["era"] = np.where(R.date < ERA_SPLIT, "2016-20", "2021-26")
    R["year"] = R.date.dt.year
    for k, P in dict(r126=r126, r21=r21, rk126=rk126, rk21=rk21, rsi=rsi, adx=adx, gscore=gscore,
                     fmax=fmax, fmin=fmin).items():
        R[k] = take(P)
    for k, P in dict(gate1=gate1, fresh_bo=fresh_bo, fresh_bo_anchor=fresh_bo_anchor).items():
        R[k] = take(P).astype(bool)
    c = take(close)
    R["dist_hi"] = (c / take(hi252) - 1) * 100
    R["ext50"] = (c / take(ma50) - 1) * 100
    R["ext200"] = (c / take(ma200) - 1) * 100
    for h in (21, 63, 126):
        r, vn, vu, _ = fwd[h]
        R[f"ret{h}"], R[f"vn{h}"], R[f"vu{h}"] = take(r), take(vn), take(vu)
    for k in ("pe", "own", "rel", "g1", "g3", "zone"):
        R[k] = takew(V[k])
    R["tier"] = takew(tier)
    R["family"] = R.ticker.map(fam).fillna("industrial")
    R["leader"] = R.gscore >= 70
    R["velocity"] = np.select([R.rk21 >= 80, R.rk21 >= 40, R.rk21.notna()], ["ACCEL", "STEADY", "FADING"], "")
    R["win80"] = np.where(R.fmax.notna(), R.fmax >= WIN, np.nan)
    R = R[R.date >= pd.Timestamp("2016-01-01")]
    # the zone needs >= 2 years of own P/E history: first possible ~mid-2017
    R.loc[R.date < pd.Timestamp("2017-07-01"), "zone"] = ""
    panels = dict(close=close, cf=cf, ma50=ma50, nifty=nifty, rk21=rk21, rk126=rk126, sidx=sidx, V=V,
                  high=high, low=low, vol=vol, gate1=gate1, fresh_bo_anchor=fresh_bo_anchor, ev=ev)
    return R, panels


def add_forensics(R, panels):
    """v6 state / signature / supply for Gate 1 rows (as in forensics_gate_study.py)."""
    from forensics_gate_study import stock_features
    from utils.forensics import indicators
    close, high, low, vol, ma50 = (panels[k] for k in ("close", "high", "low", "vol", "ma50"))
    atr = indicators(close, high, low)[0]
    vmed = vol.rolling(50, min_periods=30).median().shift(1)
    C, Hh, L, Vv, VM, M, A = (x.values for x in (close, high, low, vol, vmed, ma50, atr))
    g = R[R.gate1]
    rows = []
    for j, gg in g.groupby("j"):
        for f in stock_features(C[:, j], Hh[:, j], L[:, j], Vv[:, j], VM[:, j], M[:, j], A[:, j], gg.i.values):
            rows.append(dict(i=f["i"], j=j, state=f.get("state"), signature=f.get("signature"),
                             supply=f.get("supply")))
    if not rows:
        R["state"] = R["signature"] = R["supply"] = ""
        R["buy_now"] = False
        return R
    X = pd.DataFrame(rows)
    R = R.merge(X, on=["i", "j"], how="left")
    R[["state", "signature", "supply"]] = R[["state", "signature", "supply"]].fillna("")
    R["buy_now"] = (R.state == "GO") & (R.supply != "NEAR OVERHANG")
    return R


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def lift_table(R, label, cells, title):
    """
    P(label | cell) vs the SAME WEEKS' base rate (each row is compared with the rate among all rows
    of its own date, so a cell concentrated in good years cannot look good by timing alone).
    """
    rows = []
    d = R[R[label].notna()]
    d = d.assign(_base=d.groupby("date")[label].transform("mean"))
    for era in ("2016-20", "2021-26"):
        e = d[d.era == era]
        for name, m in cells.items():
            x = e[m.reindex(e.index, fill_value=False)]
            if len(x) < 30:
                rows.append(dict(table=title, era=era, cell=name, n=len(x)))
                continue
            p, base = x[label].mean(), x._base.mean()
            by = x.groupby("year")[[label, "_base"]].mean()
            rows.append(dict(table=title, era=era, cell=name, n=len(x), stocks=x.ticker.nunique(),
                             p_pct=round(100 * p, 2), base_pct=round(100 * base, 2),
                             lift=round(p / base, 2) if base > 0 else np.nan,
                             yrs_above=f"{(by[label] > by._base).sum()}/{len(by)}"))
    return pd.DataFrame(rows)


def ret_table(R, cells, title):
    """Median return vs the typical stock at 1/3/6 months, per era."""
    rows = []
    for era in ("2016-20", "2021-26"):
        e = R[R.era == era]
        for name, m in cells.items():
            x = e[m.reindex(e.index, fill_value=False)]
            r = dict(table=title, era=era, cell=name, n=len(x), stocks=x.ticker.nunique())
            if len(x) >= 30:
                for h, lab in ((21, "1mo"), (63, "3mo"), (126, "6mo")):
                    v = x[f"vu{h}"].dropna()
                    r[f"{lab} vsTypical"] = round(v.median(), 2) if len(v) >= 30 else np.nan
                r["6mo beat%"] = round(100 * (x.vu126.dropna() > 0).mean())
                y = x.dropna(subset=["vu126"]).groupby("year").vu126.median()
                r["yrs 6mo>0"] = f"{(y > 0).sum()}/{len(y)}"
            rows.append(r)
    return pd.DataFrame(rows)


def episodes(R, mask, gap_days):
    """First row of each run of `mask` rows per stock (a new run after `gap_days` without one)."""
    x = R[mask].sort_values(["ticker", "date"])
    new = x.ticker.ne(x.ticker.shift()) | (x.date - x.date.shift()).dt.days.gt(gap_days)
    x = x.assign(ep=new.cumsum())
    return x, x[new]


# ----------------------------------------------------------------------------
# exits
# ----------------------------------------------------------------------------
def simulate(ent, panels, rules_weekly):
    """
    Daily walk for up to H sessions after each entry (row i, j). Live exit = first close
    < MA50 or < 85% of the running peak. Weekly signals (dict name -> weekly bool panel)
    are checked on sample days only. Returns one row per entry per rule.
    """
    C, M, N = panels["cf"].values, panels["ma50"].values, panels["nifty"].values
    sidx = panels["sidx"]
    pos_w = np.full(len(C), -1)
    pos_w[sidx] = np.arange(len(sidx))
    W = {k: v.values for k, v in rules_weekly.items()}
    T = len(C)
    out = []
    for i, j in zip(ent.i.values, ent.j.values):
        e = min(i + H, T - 1)
        if e - i < H:
            continue
        p = C[i + 1:e + 1, j]
        if np.isnan(p).all():
            continue
        p = pd.Series(p).ffill().bfill().values
        entry = C[i, j]
        peak = np.maximum.accumulate(np.r_[entry, p])[1:]
        live = (p < M[i + 1:e + 1, j]) | (p < 0.85 * peak)
        wk = pos_w[i + 1:e + 1]
        sig = {}
        for k, P in W.items():
            s = np.zeros(len(p), bool)
            on = wk >= 0
            s[on] = np.nan_to_num(P[wk[on], j].astype(float)) > 0
            sig[k] = s
        rules = {"hold 6 months": np.zeros(len(p), bool), "live exit (MA50 / 15% trail)": live}
        for k, s in sig.items():
            rules[f"live + {k}"] = live | s
            rules[f"{k} only"] = s
        for name, mk in rules.items():
            k = int(np.argmax(mk)) if mk.any() else len(p) - 1
            r = (p[k] / entry - 1) * 100
            out.append(dict(i=i, j=j, rule=name, ret=r, vs_nifty=r - (N[i + 1 + k] / N[i] - 1) * 100, days=k + 1))
    return pd.DataFrame(out)


def exit_summary(S, ent, title):
    S = S.merge(ent[["i", "j", "era", "year", "outcome", "tier"]], on=["i", "j"], how="left")
    rows = []
    for (era, rule), g in S.groupby(["era", "rule"], sort=False):
        by = g.groupby("year").vs_nifty.mean()
        r = dict(table=title, era=era, rule=rule, trades=len(g), mean_ret=round(g.ret.mean(), 2),
                 median_ret=round(g.ret.median(), 2), mean_vs_nifty=round(g.vs_nifty.mean(), 2),
                 win_pct=round(100 * (g.ret > 0).mean()), avg_days=round(g.days.mean()),
                 yrs_vs_nifty_pos=f"{(by > 0).sum()}/{len(by)}")
        for oc in ("CONTINUED", "SIDEWAYS", "PULLBACK", "GAVE BACK"):
            x = g[g.outcome == oc]
            r[f"ret if {oc}"] = round(x.ret.mean(), 1) if len(x) >= 10 else np.nan
        rows.append(r)
    return pd.DataFrame(rows), S


# ----------------------------------------------------------------------------
# case studies
# ----------------------------------------------------------------------------
def fmt(x, f="{:.0f}", na="–"):
    return na if x is None or (isinstance(x, float) and np.isnan(x)) else f.format(x)


def pick_cases(E, k, score_col, seed=11):
    """Up to k episodes: the strongest per year first (spread across years), then by tier."""
    E = E.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
    picks = list(E.groupby("year").head(1).index)
    rest = E.drop(picks)
    for t in ("Large", "Mid", "Small", "Micro"):
        x = rest[rest.tier == t]
        if len(x):
            picks.append(x.index[0])
    picks = picks[:k]
    if len(picks) < k:
        picks += list(E.drop(picks).sample(min(k - len(picks), len(E) - len(picks)), random_state=seed).index)
    return E.loc[picks].sort_values("date")


def row_at(R, ticker, date):
    x = R[(R.ticker == ticker) & (R.date <= date)]
    return x.iloc[-1] if len(x) else None


def case_lines(c, R, panels, kind):
    """One case: what was visible at the start, the first trigger, the peak and the exit."""
    from utils.forensics import analyse
    C, close = panels["cf"], panels["close"]
    i, j, t = int(c.i), int(c.j), c.ticker
    s = C.iloc[i:i + H + 1, j].dropna()
    pk_date, pk = s.idxmax(), s.max()
    end = s.iloc[-1]
    df = pd.DataFrame({k: panels[k].iloc[:i + 1, j] for k in ("high", "low", "vol")}).rename(
        columns={"high": "High", "low": "Low", "vol": "Volume"})
    df["Close"] = close.iloc[:i + 1, j]
    fx = analyse(df.dropna(subset=["Close"])) or {}
    later = R[(R.ticker == t) & (R.date > c.date) & (R.date <= c.date + pd.Timedelta(days=183))]
    trig = later[later.fresh_bo_anchor | ((later.state == "GO") if "state" in later else False)]
    pkr = row_at(R, t, pk_date)
    lines = [f"**{t[:-3]}** · {c.tier or 'n/a'} · {c.get('family', '')} · start {c.date:%d %b %Y} at "
             f"{fmt(C.iloc[i, j], '{:.1f}')} → peak {pk_date:%d %b %Y} ({fmt((pk / C.iloc[i, j] - 1) * 100, '{:+.0f}')}%), "
             f"6 months later {fmt((end / C.iloc[i, j] - 1) * 100, '{:+.0f}')}%"]
    lines.append(f"- At start: 6m return {fmt(c.r126 * 100, '{:+.0f}')}% (rank {fmt(c.rk126)}), 1m rank {fmt(c.rk21)} "
                 f"({c.velocity}); Gate 1 {'yes' if c.gate1 else 'no'}; {fmt(c.dist_hi, '{:+.0f}')}% from 52w high, "
                 f"{fmt(c.ext200, '{:+.0f}')}% vs 200d; industry score {fmt(c.gscore)}; chart state "
                 f"{fx.get('state', '–')} / {fx.get('signature') or '–'} / {fx.get('supply', '–')}; RSI {fmt(c.rsi)}")
    lines.append(f"- Valuation at start: P/E {fmt(c.pe, '{:.1f}')}, own 5y %ile {fmt(c.own)}, vs peers "
                 f"{fmt(c.rel, '{:.2f}')}x, 3y profit CAGR {fmt(c.g3 * 100 if pd.notna(c.g3) else np.nan, '{:+.0f}')}% "
                 f"→ **{c.zone or 'n/a'}**")
    if len(trig):
        tr = trig.iloc[0]
        left = (pk / C.iloc[int(tr.i), j] - 1) * 100
        lines.append(f"- First chart trigger: {tr.date:%d %b %Y} ({'breakout + volume anchor' if tr.fresh_bo_anchor else 'v6 GO'}), "
                     f"{fmt(left, '{:+.0f}')}% still to the peak from there")
    else:
        lines.append("- First chart trigger: none (no fresh breakout + volume anchor or v6 GO within 6 months)")
    if pkr is not None:
        lines.append(f"- At the peak: P/E {fmt(pkr.pe, '{:.1f}')}, own %ile {fmt(pkr.own)}, vs peers "
                     f"{fmt(pkr.rel, '{:.2f}')}x → {pkr.zone or 'n/a'}; 1m rank {fmt(pkr.rk21)} ({pkr.velocity}); "
                     f"{fmt(pkr.ext200, '{:+.0f}')}% vs 200d; RSI {fmt(pkr.rsi)}")
    return lines


# ----------------------------------------------------------------------------
def synthetic_inputs():
    from forensics_gate_study import synthetic
    close, high, low, vol, nifty, pit, gscore = synthetic()
    rng = np.random.default_rng(3)
    rows = []
    for t in close.columns:
        sh, npv = 1e8, rng.uniform(50, 500)
        for y in range(2012, 2026):
            npv *= np.exp(rng.normal(0.08, 0.3))
            rows += [(t, "np", pd.Timestamp(f"{y}-03-31"), npv), (t, "eps", pd.Timestamp(f"{y}-03-31"), npv * 1e7 / sh),
                     (t, "eq_cap", pd.Timestamp(f"{y}-03-31"), 10.0), (t, "reserves", pd.Timestamp(f"{y}-03-31"), npv * 3)]
        rows += [(t, "mcap_cr", pd.Timestamp("2026-10-01"), float(close[t].iloc[-1]) * sh / 1e7),
                 (t, "price_scr", pd.Timestamp("2026-10-01"), float(close[t].iloc[-1]))]
    A = pd.DataFrame(rows, columns=["ticker", "field", "period", "value"])
    groups = {t: f"G{k % 12}" for k, t in enumerate(close.columns)}
    return close, high, low, vol, nifty, pit, gscore, A, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--scrape-missing", type=int, default=1500,
                    help="scrape fundamentals for up to N PIT-universe names missing them (0 = off)")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--scrape-minutes", type=float, default=90,
                    help="time budget for the Screener scrape; progress is saved, a rerun resumes")
    ap.add_argument("--cases", type=int, default=15)
    args = ap.parse_args()
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    tag = "_synthetic" if args.synthetic else ""

    if args.synthetic:
        close, high, low, vol, nifty, pit, gscore, A, groups = synthetic_inputs()
        raw = close.copy()
    else:
        from utils.nifty1000_list import SUB_INDUSTRY_MAP
        close, high, low, vol, raw, nifty, pit = load_prices(args.max_tickers)
        ever = pit.iloc[260::5]
        ever = list(ever.columns[ever.loc[ever.index >= "2015-06-01"].any()])
        A = load_annual(ever, args.scrape_missing, args.delay, args.scrape_minutes)
        groups = {t: g for t, g in SUB_INDUSTRY_MAP.items() if g and g != "Unknown"}
        if os.path.exists(EXTRA_IND):
            extra = pd.read_csv(EXTRA_IND).dropna()
            groups.update({t: g for t, g in zip(extra.ticker, extra.industry) if t not in groups and g})
        print(f"[universe] {len(ever)} names ever in the PIT top 1000 since mid-2015; industry known for "
              f"{sum(t in groups for t in ever)}; fundamentals for {A[A.ticker.isin(ever)].ticker.nunique()}")
        close, high, low, vol, raw = fix_corporate_actions(close, high, low, vol, raw, A)
        gscore = industry_scores(close, pit, groups, nifty)
    close, high, low, vol, raw = clean_spikes(close, high, low, vol, raw)
    fy = fy_table(A)
    R, P = build_rows(close, high, low, vol, raw, nifty, pit, gscore, fy, groups)
    print(f"[rows] {len(R)} PIT stock-weeks, {R.ticker.nunique()} stocks, {R.date.min():%b %Y} - {R.date.max():%b %Y}",
          flush=True)
    R = add_forensics(R, P)

    # ---- data checks (bug detectors) --------------------------------------
    cov = R.assign(has=R.zone != "").groupby("year").has.mean().mul(100).round(0)
    print(f"[check] valuation coverage of PIT rows by year (%): {cov.to_dict()}")
    w = R[R.win80.notna() & (R.date >= "2017-07-01")]
    print(f"[check] valuation coverage, 80% winners {100 * (w[w.win80 == 1].zone != '').mean():.0f}% vs "
          f"others {100 * (w[w.win80 == 0].zone != '').mean():.0f}%  (a big gap would bias S1)")
    print(f"[check] 80%-winner base rate by year (%): "
          f"{R[R.win80.notna()].groupby('year').win80.mean().mul(100).round(1).to_dict()}")
    if not args.synthetic:
        try:
            import glob
            snap = pd.read_parquet(sorted(glob.glob("data/cache/market_master_*.parquet"))[-1]).set_index("ticker")
            last = R[R.date == R.date.max()].set_index("ticker")
            both = pd.concat([last.pe, pd.to_numeric(snap.get("pe"), errors="coerce")], axis=1, keys=["ours", "snap"]).dropna()
            both = both[(both.ours > 0) & (both.snap > 0)]
            lr = np.log(both.ours / both.snap).abs()
            print(f"[check] latest P/E vs dashboard snapshot P/E: {len(both)} stocks, median |log ratio| "
                  f"{lr.median():.2f}, within 30%: {100 * (lr < np.log(1.3)).mean():.0f}%  (ours = last FY, theirs ~TTM)")
        except Exception as e:
            print(f"[check] P/E cross-check skipped: {e}")

    tables = []
    has_v = R.zone != ""

    # ---- S1 winners --------------------------------------------------------
    m = lambda cond: cond.fillna(False) if hasattr(cond, "fillna") else cond
    S1 = {
        "all rows": pd.Series(True, index=R.index),
        "valuation CHEAP": R.zone == "CHEAP", "valuation FAIR": R.zone == "FAIR",
        "valuation RICH": R.zone == "RICH", "valuation LOSS": R.zone == "LOSS",
        "own P/E %ile <= 30": m(R.own <= 30), "own P/E %ile >= 80": m(R.own >= 80),
        "<= 0.9x peers": m(R.rel <= 0.9), ">= 1.5x peers": m(R.rel >= 1.5),
        "3y profit CAGR >= 25%": m(R.g3 >= 0.25), "3y profit CAGR < 0 or n/a": has_v & ~m(R.g3 >= 0),
        "last FY profit +50%": m(R.g1 >= 0.5),
        "Gate 1 trend": R.gate1, "no Gate 1": ~R.gate1,
        "6m rank >= 90": m(R.rk126 >= 90), "6m rank <= 30": m(R.rk126 <= 30),
        "1m ACCEL": R.velocity == "ACCEL", "1m FADING": R.velocity == "FADING",
        "within 10% of 52w high": m(R.dist_hi >= -10), "> 40% below 52w high": m(R.dist_hi < -40),
        "fresh breakout": R.fresh_bo, "fresh breakout + volume anchor": R.fresh_bo_anchor,
        "breakout + anchor + leader industry": R.fresh_bo_anchor & R.leader,
        "leader industry": R.leader, "v6 BUY NOW": R.buy_now,
        "BUY NOW + CHEAP/FAIR": R.buy_now & R.zone.isin(["CHEAP", "FAIR"]), "BUY NOW + RICH": R.buy_now & (R.zone == "RICH"),
        "6m rank >= 90 + CHEAP/FAIR": m(R.rk126 >= 90) & R.zone.isin(["CHEAP", "FAIR"]),
        "6m rank >= 90 + RICH": m(R.rk126 >= 90) & (R.zone == "RICH"),
    }
    S1.update({f"tier {t}": R.tier == t for _, t in TIERS})
    L1 = lift_table(R, "win80", S1, "S1 P(+80% within 6m)")
    tables.append(L1)
    print(f"\n{'=' * 140}\nS1  WHO BECAME AN 80% WINNER? lift = P(winner | cell) / P(winner) in the same weeks. "
          f"yrs_above = years the cell beat its matched base rate.\n{'=' * 140}")
    print(L1.drop(columns="table").to_string(index=False))

    _, E1 = episodes(R, R.win80 == 1, 28)
    E1 = E1.assign(gain=E1.fmax)
    # how much of each winner's run was still ahead at its first chart trigger?
    C = P["cf"].values
    trig_rows = R[R.fresh_bo_anchor | R.buy_now][["ticker", "date", "i", "fresh_bo_anchor"]]
    by_t = {t: g for t, g in trig_rows.groupby("ticker")}
    tl = []
    for _, e in E1.iterrows():
        g = by_t.get(e.ticker)
        end = min(int(e.i) + H, len(C) - 1)
        peak = np.nanmax(C[int(e.i):end + 1, int(e.j)])
        x = g[(g.i >= e.i) & (g.i <= end)] if g is not None else None
        blind = {w: (np.nanmax(C[int(e.i) + 5 * w:end + 1, int(e.j)]) / C[int(e.i) + 5 * w, int(e.j)] - 1) * 100
                 for w in (4, 8, 13) if int(e.i) + 5 * w <= end and np.isfinite(C[int(e.i) + 5 * w, int(e.j)])}
        if x is None or not len(x):
            tl.append(dict(era=e.era, trigger=False, **{f"blind{w}": v for w, v in blind.items()}))
            continue
        k = int(x.i.iloc[0])
        # baseline: the same delay in a winner run that had NO trigger is measured below via 'blind'
        tl.append(dict(era=e.era, trigger=True, wait_weeks=(k - int(e.i)) / 5,
                       left=(np.nanmax(C[k:end + 1, int(e.j)]) / C[k, int(e.j)] - 1) * 100,
                       total=(peak / C[int(e.i), int(e.j)] - 1) * 100,
                       kind="breakout + anchor" if x.fresh_bo_anchor.iloc[0] else "v6 BUY NOW",
                       **{f"blind{w}": v for w, v in blind.items()}))
    TL = pd.DataFrame(tl)
    print(f"\n--- S1 first chart trigger inside each 80% winner's run ({len(TL)} runs)")
    for era, g in TL.groupby("era"):
        h = g[g.trigger]
        print(f"   {era}: trigger in {100 * g.trigger.mean():.0f}% of runs; median wait {h.wait_weeks.median():.0f} weeks; "
              f"median gain still ahead {h.left.median():.0f}% of a median {h.total.median():.0f}% run; "
              f">= 40% ahead in {100 * (h.left >= 40).mean():.0f}%; by kind {h.kind.value_counts().to_dict()}")
        print("      baseline, ALL runs, gain still ahead if bought blind 4 / 8 / 13 weeks in: "
              + " / ".join(f"{g[f'blind{w}'].median():.0f}%" for w in (4, 8, 13) if f"blind{w}" in g)
              + "  (a trigger is useful only if it beats the blind entry at a similar wait)")

    # ---- S2 momentum fades ---------------------------------------------------
    mom = m(R.r126 >= 0.5) & m(R.rk126 >= 90)
    R["outcome"] = np.select([R.ret126 >= 20, R.ret126 > -15, R.ret126 > -25, R.ret126.notna()],
                             ["CONTINUED", "SIDEWAYS", "PULLBACK", "GAVE BACK"], "")
    Rm = R[mom & (R.outcome != "")].copy()
    print(f"\n{'=' * 140}\nS2  MOMENTUM NAMES (6m +50%, top 10%): {len(Rm)} stock-weeks, {Rm.ticker.nunique()} stocks. "
          f"Outcome over the next 6 months:\n{(Rm.groupby('era').outcome.value_counts(normalize=True) * 100).round(1).unstack()}")
    for oc in ("GAVE BACK", "SIDEWAYS", "CONTINUED"):
        Rm[f"is_{oc}"] = (Rm.outcome == oc).astype(float)
    S2 = {
        "all momentum rows": pd.Series(True, index=Rm.index),
        "RICH": Rm.zone == "RICH", "CHEAP/FAIR": Rm.zone.isin(["CHEAP", "FAIR"]), "LOSS": Rm.zone == "LOSS",
        "own %ile >= 90": m(Rm.own >= 90), ">= 2x peers": m(Rm.rel >= 2),
        "3y profit CAGR >= 25%": m(Rm.g3 >= 0.25), "profit falling (last FY < 0%)": m(Rm.g1 < 0),
        "1m FADING": Rm.velocity == "FADING", "1m ACCEL": Rm.velocity == "ACCEL",
        "> 60% above 200d": m(Rm.ext200 > 60), "< 25% above 200d": m(Rm.ext200 < 25),
        "RSI > 75": m(Rm.rsi > 75), "no Gate 1 (already off the high)": ~Rm.gate1,
        "v6 DISTRIBUTION": Rm.signature == "DISTRIBUTION", "v6 ABSORPTION": Rm.signature == "ABSORPTION",
        "v6 VETO": Rm.state == "VETO", "near overhead supply": Rm.supply == "NEAR OVERHANG",
        "leader industry": Rm.leader, "laggard industry": m(Rm.gscore < 40),
        "FADING + RICH": (Rm.velocity == "FADING") & (Rm.zone == "RICH"),
        "FADING + DISTRIBUTION": (Rm.velocity == "FADING") & (Rm.signature == "DISTRIBUTION"),
    }
    for oc in ("GAVE BACK", "SIDEWAYS", "CONTINUED"):
        t_ = lift_table(Rm, f"is_{oc}", S2, f"S2 P({oc})")
        tables.append(t_)
        print(f"\n--- S2 lift for {oc}\n{t_.drop(columns='table').to_string(index=False)}")
    T2 = ret_table(Rm, S2, "S2 returns")
    tables.append(T2)
    print(f"\n--- S2 returns vs typical stock\n{T2.drop(columns='table').to_string(index=False)}")

    # exits on momentum episodes (first qualifying week per run)
    _, E2 = episodes(Rm, pd.Series(True, index=Rm.index), 56)
    sidx = P["sidx"]
    fade_w = (P["rk21"].iloc[sidx] <= 30) & (P["rk126"].iloc[sidx] >= 50)
    rich_w = (P["V"]["own"] >= 95) & (P["V"]["rel"] >= 1.5)
    S = simulate(E2, P, {"velocity exit (1m rank <= 30)": fade_w, "valuation exit (own >= 95, >= 1.5x peers)": rich_w})
    X2, S2x = exit_summary(S, E2, "S2 exits on momentum episodes")
    tables.append(X2)
    print(f"\n{'=' * 140}\nS2 EXITS: {len(E2)} momentum episodes (first week each run). Mean return per trade; "
          f"'ret if X' = mean return of that rule on episodes that turned out X. mean_vs_nifty = excess over Nifty "
          f"while held, i.e. proceeds parked in Nifty after an exit; it does NOT credit an exit with rotating into "
          f"the next momentum name, which is what the live engine does.\n{'=' * 140}")
    print(X2.drop(columns="table").to_string(index=False))

    # ---- S3 velocity x valuation ---------------------------------------------
    strong = m(R.rk126 >= 80)
    S3 = {}
    for v in ("ACCEL", "STEADY", "FADING"):
        for z in ("CHEAP", "FAIR", "RICH", "LOSS"):
            S3[f"{v} + {z}"] = strong & (R.velocity == v) & (R.zone == z)
        S3[f"{v} (any valuation)"] = strong & (R.velocity == v)
    S3["strong momentum, all"] = strong
    S3["all rows"] = pd.Series(True, index=R.index)
    T3 = ret_table(R, S3, "S3 velocity x valuation (6m rank >= 80)")
    tables.append(T3)
    print(f"\n{'=' * 140}\nS3  VELOCITY x VALUATION within strong momentum (6m rank >= 80)\n{'=' * 140}")
    print(T3.drop(columns="table").to_string(index=False))

    # ---- S4 size tiers and families -------------------------------------------
    T4s = []
    for col, vals in (("tier", [t for _, t in TIERS]), ("family", ["industrial", "lender", "commodity"])):
        for v in vals:
            g = R[col] == v
            cells = {f"{v}: all": g, f"{v}: momentum (6m rank >= 90 + Gate 1)": g & m(R.rk126 >= 90) & R.gate1,
                     f"{v}: v6 BUY NOW": g & R.buy_now, f"{v}: breakout + anchor + leader": g & R.fresh_bo_anchor & R.leader,
                     f"{v}: FADING momentum": g & m(R.rk126 >= 80) & (R.velocity == "FADING"),
                     f"{v}: momentum + RICH": g & m(R.rk126 >= 90) & (R.zone == "RICH"),
                     f"{v}: momentum + CHEAP/FAIR": g & m(R.rk126 >= 90) & R.zone.isin(["CHEAP", "FAIR"])}
            T4s.append(ret_table(R, cells, f"S4 by {col}"))
    T4 = pd.concat(T4s)
    tables.append(T4)
    print(f"\n{'=' * 140}\nS4  BY SIZE TIER (point-in-time market-cap rank) AND FAMILY\n{'=' * 140}")
    print(T4.drop(columns="table").to_string(index=False))
    X4 = S2x.groupby(["tier", "era", "rule"], sort=False).agg(
        trades=("ret", "size"), mean_ret=("ret", "mean"), mean_vs_nifty=("vs_nifty", "mean"),
        avg_days=("days", "mean")).round(2).reset_index()
    X4 = X4[X4.rule.isin(["hold 6 months", "live exit (MA50 / 15% trail)"]) | X4.rule.str.startswith("live +")]
    tables.append(X4.assign(table="S4 exits by tier"))
    print(f"\n--- S4 exits by tier (momentum episodes)\n{X4.to_string(index=False)}")

    # ---- hypotheses ----------------------------------------------------------
    MIN_N, MIN_STOCKS = 200, 30

    def cell(T, table, name, era, col):
        x = T[(T.table == table) & (T.cell == name) & (T.era == era)]
        if not len(x) or col not in x or pd.isna(x[col].iloc[0]):
            return np.nan
        if x.n.iloc[0] < MIN_N or x.get("stocks", pd.Series([0])).fillna(0).iloc[0] < MIN_STOCKS:
            return np.nan
        return x[col].iloc[0]

    def judge(vals, ok):
        return None if any(pd.isna(v) for v in vals) else bool(ok)

    t1, t3 = L1.table.iloc[0], T3.table.iloc[0]
    verdict = {}
    for era in ("2016-20", "2021-26"):
        ch, ri = cell(L1, t1, "valuation CHEAP", era, "lift"), cell(L1, t1, "valuation RICH", era, "lift")
        verdict[f"H1 {era}: winner lift CHEAP {ch} >= 1.2 x RICH {ri}"] = judge([ch, ri], ch >= 1.2 * ri)
        lf = cell(L1, t1, "breakout + anchor + leader industry", era, "lift")
        verdict[f"H2 {era}: breakout+anchor+leader lift {lf} >= 1.5"] = judge([lf], lf >= 1.5)
        a, b = cell(T2, "S2 returns", "RICH", era, "6mo vsTypical"), cell(T2, "S2 returns", "CHEAP/FAIR", era, "6mo vsTypical")
        verdict[f"H3 {era}: momentum RICH {a} <= CHEAP/FAIR {b} - 2"] = judge([a, b], a <= b - 2)
        fr = cell(T3, t3, "FADING + RICH", era, "6mo vsTypical")
        fc = cell(T3, t3, "FADING + FAIR", era, "6mo vsTypical")
        verdict[f"H4 {era}: FADING+RICH {fr} <= FADING+FAIR {fc} - 2"] = judge([fr, fc], fr <= fc - 2)
        live = X2[(X2.era == era) & (X2.rule == "live exit (MA50 / 15% trail)")].set_index("rule").mean_vs_nifty
        for rule in X2[X2.rule.str.startswith("live + ")].rule.unique():
            v = X2[(X2.era == era) & (X2.rule == rule)].mean_vs_nifty
            a_, b_ = (v.iloc[0] if len(v) else np.nan), (live.iloc[0] if len(live) else np.nan)
            verdict[f"H5 {era}: {rule} {a_} >= live {b_} + 0.5"] = judge([a_, b_], a_ >= b_ + 0.5)
    print(f"\n{'=' * 140}\nPRE-REGISTERED HYPOTHESES (pass only if both eras pass; cells need >= {MIN_N} rows "
          f"from >= {MIN_STOCKS} stocks, else TOO FEW)\n{'=' * 140}")
    for k, v in verdict.items():
        print(f"  {'TOO FEW' if v is None else 'PASS' if v else 'FAIL'}  {k}")
    for h in ("H1", "H2", "H3", "H4"):
        vs = [v for k, v in verdict.items() if k.startswith(h)]
        print(f"  => {h}: {'TOO FEW DATA' if None in vs else 'SUPPORTED' if all(vs) else 'NOT SUPPORTED'}")
    for rule in X2[X2.rule.str.startswith("live + ")].rule.unique():
        vs = [v for k, v in verdict.items() if k.startswith("H5") and rule in k]
        print(f"  => H5 [{rule}]: {'TOO FEW DATA' if None in vs else 'SUPPORTED' if all(vs) else 'NOT SUPPORTED'}")

    # ---- cases ---------------------------------------------------------------
    md = ["# Entry / exit case studies" + (" (SYNTHETIC DATA)" if args.synthetic else ""),
          "", "Generated by `entry_exit_study.py`. Each case shows only what was knowable on its dates; "
          "valuation uses the latest annual results public 92 days after the year end.", ""]
    E1v = E1[E1.zone != ""]
    md += [f"## S1 · 80%+ winners ({len(E1)} runs; cases = the biggest run each year, then one per size tier)", ""]
    for _, c in pick_cases(E1v, args.cases, "gain").iterrows():
        md += case_lines(c, R, P, "win") + [""]
    for oc, title in (("GAVE BACK", "momentum names that gave the gain back"),
                      ("SIDEWAYS", "momentum names that went into a time correction"),
                      ("CONTINUED", "momentum names that kept going (control group)")):
        x = E2[(E2.outcome == oc) & (E2.zone != "")].assign(sev=lambda d: -d.ret126 if oc != "CONTINUED" else d.ret126)
        if oc == "SIDEWAYS":
            x = x.assign(sev=-x.ret126.abs())
        md += [f"## S2 · {title} ({(E2.outcome == oc).sum()} episodes)", ""]
        for _, c in pick_cases(x, args.cases, "sev").iterrows():
            md += case_lines(c, R, P, oc) + [""]
            ex = S2x[(S2x.i == c.i) & (S2x.j == c.j)].set_index("rule").ret
            md += ["- Exits: " + " · ".join(f"{k} {v:+.0f}%" for k, v in ex.items()), ""]
    # S3 cases: a RANDOM sample (not the extremes) of fading momentum, half RICH, half CHEAP/FAIR
    fad = strong & (R.velocity == "FADING")
    _, E3 = episodes(R, fad & R.zone.isin(["RICH", "CHEAP", "FAIR"]) & R.ret126.notna(), 56)
    for zs, title in ((["RICH"], "momentum slowing while RICH"), (["CHEAP", "FAIR"], "momentum slowing while CHEAP/FAIR")):
        x = E3[E3.zone.isin(zs)]
        md += [f"## S3 · {title} ({len(x)} episodes; a random sample, so typical rather than extreme)", ""]
        for _, c in x.sample(min(len(x), (args.cases + 1) // 2), random_state=5).sort_values("date").iterrows():
            md += case_lines(c, R, P, "fade") + [f"- Next 6 months: {c.ret126:+.0f}% ({c.vu126:+.0f} pp vs the typical stock)", ""]
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/ENTRY_EXIT_CASES{tag}.md", "w") as fh:
        fh.write("\n".join(md))
    pd.concat(tables, ignore_index=True).to_csv(f"{OUT}/entry_exit_study{tag}.csv", index=False)
    print(f"\nsaved -> {OUT}/entry_exit_study{tag}.csv, {OUT}/ENTRY_EXIT_CASES{tag}.md")


if __name__ == "__main__":
    main()
