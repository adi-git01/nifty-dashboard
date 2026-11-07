"""
SIMBA -- Ghoomega / Machayega / Bachayega, as a scanner.
=========================================================

Implements the rule-based reconstruction in adi-git01/desktop-tutorial
(trading-system/SIMBA_Ghoomega_Machayega_System.md and GMB_Suite_Reconstruction.pine)
of Kumar Saurabh's (Scientific Investing) Techno-Funda process. The original SIMBA and
GMB indicators are closed-source: every threshold here is that document's reconstruction,
not Scientific Investing's own numbers.

Chart side (chart_signals) is a line-for-line port of the Pine indicator, computed in the
EOD engine from each stock's daily bars and stored in the master snapshot as simba_*:
  Ghoomega  G1 prior dormancy (150-DMA rose <= 5% over the 100 bars ending 20 bars ago)
            G2 150-DMA curving up   G3 close > 150-DMA   G4 50-DMA > 150-DMA and rising
            G5 higher low (40-bar low above the 140-bar low)
            G6 Mansfield RS (vs Nifty 500, 250 bars) > 0 and rising   G7 <= 25% above 150-DMA
  Machayega M1 close > prior 50-bar high   M2 prior 15-bar range <= 15%
            M3 volume >= 2x 50-day average   M4 range >= 1.2x ATR(14)   M5 CLV >= 0.7, up day
  SI Opportunity = Machayega within 30 bars of an active Ghoomega, MTF RSI aligned
            (daily > 55, last completed week > 55, last completed month > 50), not extended
  Bachayega caution (no position needed): L1 >= 4 distribution days in 25 or weekly RSI < 50;
            L2 close < falling 50-DMA, Mansfield < 0, 10-day ROC <= -12% or climactic gap-down;
            L3 stage breach (close < flat/falling 150-DMA)

Fundamental side (fundamentals) reads data/fundamentals_quarterly.csv.gz and
fundamentals_annual.csv.gz: Earnings Reversal (profit outran sales, turnaround,
accelerating profit, net-margin inflection), ROCE, margin stability, cash conversion,
leverage, capex strain, promoter holding (low float).

Not available in our data, scored as 0 and shown as n/a: insider cluster buying (6 pts),
delivery % (1 pt), volume-vs-price growth split (3 pts). So the best possible score is 90.

Our own evidence disagrees with parts of this (analysis/REGIME_VERDICT.md, entry/exit and
industry-turn studies): early turns lagged confirmed uptrends, and expensive momentum beat
"no valuation froth". The tab shows SIMBA as written; it is not the live engine.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Pine inputs (defaults of GMB_Suite_Reconstruction.pine)
STAGE_LEN, FAST_LEN, BASE_LEN, DORM_MAX = 150, 50, 100, 5.0
HL_LEN, MAX_EXT, RS_LEN = 40, 25.0, 250
PIVOT_LEN, TIGHT_LEN, MAX_RANGE, VOL_LEN, VOL_MULT, SPREAD_MULT, MIN_CLV = 50, 15, 15.0, 50, 2.0, 1.2, 0.7
CONF_BARS, RSI_LEN, RSI_D, RSI_W, RSI_M = 30, 14, 55.0, 55.0, 50.0
SL_INIT, DIST_LEN, DIST_MAX, ROC_LEN, ROC_MIN = 15.0, 25, 4, 10, -12.0

MIN_BARS = STAGE_LEN + 20          # stage MA and its 20-bar slope
FULL_BARS = STAGE_LEN + 20 + BASE_LEN   # G1 needs the stage MA 120 bars back
HISTORY_DAYS = 430                 # calendar days the engine downloads (~290 sessions)
BENCH = "^CRSLDX"                  # Nifty 500, the document's RS benchmark
LOOKBACK = 60                      # "bars since" fields are reported within this window


def _rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's moving average, as Pine's ta.rma / ta.atr / ta.rsi use."""
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = RSI_LEN) -> pd.Series:
    d = close.diff()
    up, dn = _rma(d.clip(lower=0), n), _rma(-d.clip(upper=0), n)
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0)


def _htf_rsi(close: pd.Series, rule: str) -> pd.Series:
    """RSI of the last COMPLETED week/month, aligned to daily bars (Pine: rsi[1], lookahead_on)."""
    per = close.index.to_period(rule)
    htf = close.groupby(per).last()
    prev = _rsi(htf).shift(1)
    return pd.Series(prev.reindex(per).values, index=close.index)


def _bars_since(flag: pd.Series) -> float:
    """Bars since `flag` was last true (0 = today), NaN if not within LOOKBACK."""
    f = flag.fillna(False).values[-LOOKBACK:]
    idx = np.flatnonzero(f)
    return float(len(f) - 1 - idx[-1]) if len(idx) else np.nan


def _streak(flag: pd.Series) -> int:
    f = flag.fillna(False).values
    n = 0
    for v in f[::-1]:
        if not v:
            break
        n += 1
    return n


def chart_signals(df: pd.DataFrame, bench: Optional[pd.Series] = None) -> Dict[str, Any]:
    """SIMBA chart fields for the last bar of one stock's daily OHLCV (auto-adjusted)."""
    out: Dict[str, Any] = {"simba_bars": int(len(df))}
    if len(df) < MIN_BARS:
        return out
    c, h, l = df["Close"].astype(float), df["High"].astype(float), df["Low"].astype(float)
    o = df["Open"].astype(float) if "Open" in df else c
    v = df["Volume"].astype(float).fillna(0) if "Volume" in df else pd.Series(0.0, index=c.index)

    maS, maF = c.rolling(STAGE_LEN).mean(), c.rolling(FAST_LEN).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = _rma(tr, 14)
    vol_avg = v.rolling(VOL_LEN).mean()

    if bench is not None and len(bench):
        b = bench.reindex(c.index).ffill()
        rp = c / b
        mrs = (rp / rp.rolling(RS_LEN).mean() - 1) * 100
    else:
        mrs = pd.Series(np.nan, index=c.index)

    rsi_d = _rsi(c)
    rsi_w = _htf_rsi(c, "W-FRI")
    rsi_m = _htf_rsi(c, "M")
    mtf = (rsi_d > RSI_D) & (rsi_w > RSI_W) & (rsi_m > RSI_M)
    extended = (rsi_d > 80) & (c > maF * 1.25)

    slope_now = (maS - maS.shift(20)) / maS.shift(20) * 100
    slope_prev = (maS.shift(20) - maS.shift(60)) / maS.shift(60) * 100
    flat = slope_now.abs() < 1.0
    stage = np.where(flat, np.where(slope_prev < 0, 1, 3), np.where(slope_now > 0, 2, 4))

    g = {
        1: (maS.shift(20) - maS.shift(20 + BASE_LEN)) / maS.shift(20 + BASE_LEN) * 100 <= DORM_MAX,
        2: (maS > maS.shift(10)) & (maS.shift(10) >= maS.shift(20)),
        3: c > maS,
        4: (maF > maS) & (maF > maF.shift(5)),
        5: l.rolling(HL_LEN).min() > l.rolling(BASE_LEN + HL_LEN).min(),
        6: (mrs > 0) & (mrs > mrs.shift(10)),
        7: c <= maS * (1 + MAX_EXT / 100),
    }
    ghoo = g[1] & g[2] & g[3] & g[4] & g[5] & g[6] & g[7]

    pivot = h.rolling(PIVOT_LEN).max().shift(1)
    t_hi, t_lo = h.rolling(TIGHT_LEN).max().shift(1), l.rolling(TIGHT_LEN).min().shift(1)
    base_range = (t_hi - t_lo) / t_lo * 100
    rng = h - l
    clv = ((c - l) / rng).where(rng > 0, 0.0)
    m = {
        1: c > pivot,
        2: base_range <= MAX_RANGE,
        3: v >= vol_avg.shift(1) * VOL_MULT,
        4: rng >= atr.shift(1) * SPREAD_MULT,
        5: (clv >= MIN_CLV) & (c > c.shift(1)),
    }
    mach = m[1] & m[2] & m[3] & m[4] & m[5] & (vol_avg.shift(1) > 0)
    dry_up = v.rolling(10).mean().shift(1) < vol_avg.shift(1) * 0.8
    vol_x = v / vol_avg.shift(1)
    down_vol = v.where(c < c.shift(1), 0.0)
    pocket = (c > c.shift(1)) & (v > down_vol.rolling(10).max().shift(1)) & (c > maF) & ~mach

    # bars since the last Ghoomega, per bar (Pine: ta.barssince), for the SI Opportunity window
    gi = pd.Series(np.where(ghoo.fillna(False), np.arange(len(c)), np.nan), index=c.index).ffill()
    since_g = np.arange(len(c)) - gi
    si = mach & (since_g <= CONF_BARS) & mtf & ~extended

    dist_day = (c < c.shift(1) * 0.998) & (v > v.shift(1)) & (v > vol_avg)
    dist_cnt = dist_day.astype(int).rolling(DIST_LEN).sum()
    roc = (c / c.shift(ROC_LEN) - 1) * 100
    gap_climax = (o < l.shift(1)) & (c < c.shift(1) * 0.95) & (v > vol_avg * 2)
    l1 = (dist_cnt >= DIST_MAX) | (rsi_w < 50)
    l2 = ((c < maF) & (maF < maF.shift(5))) | (mrs < 0) | (roc <= ROC_MIN) | gap_climax
    breach = (c < maS) & (maS <= maS.shift(10))
    atr_pct_avg = (atr / c * 100).rolling(50).mean()
    atr_pct = atr / c * 100
    vcp = atr_pct <= atr_pct.rolling(126, min_periods=100).min() * 1.0001

    # good weekly close: last COMPLETED week closed above the pivot of the latest Machayega,
    # in the upper half of its weekly range
    wk = c.index.to_period("W-FRI")
    wc, whi, wlo = c.groupby(wk).last(), h.groupby(wk).max(), l.groupby(wk).min()
    m_since = _bars_since(mach)
    m_pivot = float(pivot.iloc[-1 - int(m_since)]) if np.isfinite(m_since) else np.nan
    good_wc = np.nan                     # unknown until a week closes after the breakout
    if np.isfinite(m_pivot):
        complete = wc if c.index[-1].weekday() == 4 else wc.iloc[:-1]   # EOD Friday closes the week
        after = complete[complete.index >= wk[-1 - int(m_since)]]
        if len(after):
            k = after.index[0]           # the breakout's own week
            good_wc = bool(wc[k] > m_pivot and wc[k] >= (whi[k] + wlo[k]) / 2)

    last = lambda s: (None if pd.isna(s.iloc[-1]) else s.iloc[-1])
    fl = lambda s: (np.nan if pd.isna(s.iloc[-1]) else round(float(s.iloc[-1]), 2))
    bits = lambda d: "".join("1" if bool(d[k].iloc[-1]) else "0" for k in sorted(d))
    lvl = 3 if bool(breach.iloc[-1]) else 2 if bool(l2.iloc[-1]) else 1 if bool(l1.iloc[-1]) else 0
    ap = fl(atr_pct_avg)
    out.update({
        "simba_full": bool(len(df) >= FULL_BARS and pd.notna(mrs.iloc[-1])),
        "simba_stage": int(stage[-1]),
        "simba_g": bool(ghoo.iloc[-1]),
        "simba_g_bits": bits(g),
        "simba_g_since": _bars_since(ghoo),
        "simba_g_streak": _streak(ghoo),
        "simba_m": bool(mach.iloc[-1]),
        "simba_m_bits": bits(m),
        "simba_m_since": m_since,
        "simba_m_pivot": None if not np.isfinite(m_pivot) else round(m_pivot, 2),
        "simba_si_since": _bars_since(si),
        "simba_pp_since": _bars_since(pocket),
        "simba_good_wc": good_wc,
        "simba_mrs": fl(mrs),
        "simba_mrs_cross": bool(((mrs > 0) & (mrs.shift(1) <= 0)).iloc[-20:].any()),
        "simba_rsi_d": fl(rsi_d), "simba_rsi_w": fl(rsi_w), "simba_rsi_m": fl(rsi_m),
        "simba_mtf": int(rsi_d.iloc[-1] > RSI_D) + int(rsi_w.iloc[-1] > RSI_W) + int(rsi_m.iloc[-1] > RSI_M),
        "simba_ext": bool(extended.iloc[-1]),
        "simba_ext_ma150": round(float(c.iloc[-1] / maS.iloc[-1] - 1) * 100, 1),
        "simba_base_range": fl(base_range),
        "simba_base_low": fl(t_lo),
        "simba_vol_x": fl(vol_x),
        "simba_vol_x_m": (round(float(vol_x.iloc[-1 - int(m_since)]), 2) if np.isfinite(m_since) else np.nan),
        "simba_dryup": bool(dry_up.iloc[-1]),
        "simba_vcp": bool(vcp.iloc[-1]),
        "simba_caution": lvl,
        "simba_dist_days": int(dist_cnt.iloc[-1]) if pd.notna(dist_cnt.iloc[-1]) else 0,
        "simba_trail": (np.nan if not np.isfinite(ap) else 20 if ap < 3 else 25 if ap < 5 else 30),
    })
    return out


# ----------------------------------------------------------------------------
# fundamentals (repo files, no network)
# ----------------------------------------------------------------------------
def _yoy(s: pd.Series) -> pd.Series:
    """Year-on-year change vs the same quarter a year earlier, matched by date (gaps-safe)."""
    prev = pd.Series(s.reindex(s.index - pd.DateOffset(years=1)).values, index=s.index)
    return ((s - prev) / prev.abs() * 100).where(prev.abs() > 0)


def fundamentals(root: str = ".", stale_days: int = 270) -> pd.DataFrame:
    """One row per ticker: Earnings Reversal legs, quality and forensic checks."""
    p = lambda f: os.path.join(root, f)
    rows: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(p("data/fundamentals_quarterly.csv.gz")):
        Q = pd.read_csv(p("data/fundamentals_quarterly.csv.gz"))
        Q["period"] = pd.to_datetime(Q["period"])
        Q = Q[Q.period.dt.month.isin([3, 6, 9, 12])]
        W = Q.pivot_table(index=["ticker", "period"], columns="field", values="value", aggfunc="last")
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=stale_days)
        for t, g in W.groupby(level=0):
            g = g.droplevel(0).sort_index()
            r = rows.setdefault(t, {})
            if "promoters" in g and g["promoters"].notna().any():
                r["promoters"] = float(g["promoters"].dropna().iloc[-1])
            if not {"sales", "np"} <= set(g.columns):
                continue
            g = g.dropna(subset=["sales", "np"])     # shareholding rows can run a quarter ahead
            if len(g) < 5 or g.index[-1] < cutoff:
                continue
            s, n = g["sales"], g["np"]
            sg, ng = _yoy(s), _yoy(n)
            npm = (n / s * 100).where(s > 0)
            r.update(quarter=g.index[-1].strftime("%b %Y"),
                     sales_g=None if pd.isna(sg.iloc[-1]) else round(float(sg.iloc[-1]), 1),
                     np_g=None if pd.isna(ng.iloc[-1]) else round(float(ng.iloc[-1]), 1))
            last_np = n.iloc[-1]
            prev_np = n.get(g.index[-1] - pd.DateOffset(years=1), np.nan)
            op_lev = bool(pd.notna(sg.iloc[-1]) and pd.notna(ng.iloc[-1]) and last_np > 0
                          and sg.iloc[-1] >= 15 and ng.iloc[-1] >= 1.5 * sg.iloc[-1])
            turn = bool(pd.notna(prev_np) and prev_np <= 0 < last_np)
            accel = bool(len(ng.dropna()) >= 3 and last_np > 0 and ng.iloc[-1] > 0
                         and ng.iloc[-1] > ng.iloc[-2] > ng.iloc[-3])
            npm_prev = npm.get(g.index[-1] - pd.DateOffset(years=1), np.nan)
            margin = bool(pd.notna(npm_prev) and pd.notna(npm.iloc[-1])
                          and npm.iloc[-1] - npm_prev >= 2.0 and last_np > 0)
            r.update(er_oplev=op_lev, er_turn=turn, er_accel=accel, er_margin=margin,
                     er=op_lev or turn or accel or margin)
    if os.path.exists(p("data/fundamentals_annual.csv.gz")):
        A = pd.read_csv(p("data/fundamentals_annual.csv.gz"))
        A["period"] = pd.to_datetime(A["period"])
        W = A.pivot_table(index=["ticker", "period"], columns="field", values="value", aggfunc="last")
        for t, g in W.groupby(level=0):
            g = g.droplevel(0).sort_index()
            g = g[g.index.month == g.index.month.value_counts().idxmax()].tail(10)   # fiscal year ends
            r = rows.setdefault(t, {})
            col = lambda f: g[f].dropna() if f in g else pd.Series(dtype=float)
            roce, opm = col("roce"), col("opm")
            if len(roce):
                r["roce"] = round(float(roce.iloc[-1]), 1)
                r["roce3"] = round(float(roce.tail(3).mean()), 1)
                r["roce_rising"] = bool(len(roce) >= 3 and roce.iloc[-1] > roce.iloc[-3])
            if len(opm) >= 4 and opm.tail(5).mean() > 0:
                r["opm_cv"] = round(float(opm.tail(5).std() / opm.tail(5).mean()), 2)
            cfo, op = col("cfo").tail(5), col("op").tail(5)
            if len(cfo) >= 3 and op.sum() > 0:
                r["cfo_ebitda"] = round(float(cfo.sum() / op.sum()), 2)
            eq = col("eq_cap").add(col("reserves"), fill_value=0)
            bor = col("borrowings")
            if len(eq) and eq.iloc[-1] > 0:
                r["de"] = round(float((bor.iloc[-1] if len(bor) else 0) / eq.iloc[-1]), 2)
            cw, fa = col("cwip"), col("fixed_assets")
            if len(cw) and len(fa) and fa.iloc[-1] > 0:
                r["cwip_fa"] = round(float(cw.iloc[-1] / fa.iloc[-1]), 2)
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "ticker"
    return out.reset_index()


# ----------------------------------------------------------------------------
# top-down: regime, risk scale, industries
# ----------------------------------------------------------------------------
def market_regime(nifty: Optional[pd.DataFrame], breadth_pct: Optional[float]) -> Dict[str, Any]:
    """
    SIMBA regime (document Part 3-A) on Nifty month-end closes vs their 10-month SMA plus
    breadth (% of the universe above its 200-DMA). The document's main gauge is the
    Smallcap 250; Nifty 50 is what the dashboard has, so froth is judged on it.
    """
    out: Dict[str, Any] = {"regime": "UNKNOWN", "breadth": breadth_pct}
    if nifty is None or getattr(nifty, "empty", True) or breadth_pct is None:
        return out
    c = pd.to_numeric(nifty["Close"], errors="coerce").dropna()
    if getattr(c.index, "tz", None) is not None:
        c.index = c.index.tz_localize(None)
    mc = c.groupby(c.index.to_period("M")).last()
    if len(mc) < 11:
        return out
    sma = mc.rolling(10).mean()
    px, s_now, s_prev = float(c.iloc[-1]), float(sma.iloc[-1]), float(sma.iloc[-2])
    rising = s_now > s_prev
    above = px > s_now
    dist = (px / s_now - 1) * 100
    rsi_m = float(_rsi(mc).iloc[-1]) if len(mc) > 15 else np.nan
    ret12 = (px / float(mc.iloc[-13]) - 1) * 100 if len(mc) >= 13 else np.nan
    b = breadth_pct
    if b < 20 and (not np.isfinite(rsi_m) or rsi_m < 40):
        reg = "WASHOUT"
    elif not above and not rising and b < 35:
        reg = "BEAR CONTRACTION"
    elif not above:
        reg = "CORRECTION"
    elif rising and b >= 50:
        froth = sum([dist > 25, np.isfinite(ret12) and ret12 > 50, b > 80])
        reg = "FROTH" if froth >= 2 else "NORMAL BULL"
    else:
        reg = "CORRECTION"
    # Risk Scale: 2 of the document's 5 inputs exist here (breadth, distance above the 10M SMA)
    raw = (0 if b < 40 else 1 if b <= 75 else 2) + (0 if dist < 10 else 1 if dist <= 25 else 2)
    risk = int(round(1 + raw * 9 / 4))
    n_lo, n_hi = {1: (5, 8), 2: (5, 8), 3: (9, 12), 4: (9, 12), 5: (13, 16), 6: (13, 16),
                  7: (17, 20), 8: (17, 20), 9: (21, 25), 10: (21, 25)}[risk]
    out.update(regime=reg, nifty=px, sma10m=s_now, sma_rising=bool(rising), dist_sma=round(dist, 1),
               rsi_m=None if not np.isfinite(rsi_m) else round(rsi_m, 1),
               ret12=None if not np.isfinite(ret12) else round(ret12, 1),
               risk_scale=risk, holdings=(n_lo, n_hi))
    return out


REGIME_PTS = {"NORMAL BULL": 8, "WASHOUT": 6, "CORRECTION": 4, "FROTH": 3, "BEAR CONTRACTION": 0, "UNKNOWN": 0}
REGIME_ACTION = {
    "WASHOUT": "Build the Ghoomega watchlist aggressively -- earliest Stage-1 bases form here.",
    "BEAR CONTRACTION": "No new breakout buys. Only Earnings-Reversal leaders with RS >= 90, half size.",
    "CORRECTION": "Only A+ setups, starter positions only; keep >= 25% cash.",
    "NORMAL BULL": "Full system: all setups, full scaling.",
    "FROTH": "Stop concentrating: more stocks (toward 25), trails tightened to 20%, no story stocks.",
    "UNKNOWN": "Regime unavailable (needs Nifty history and breadth).",
}
INDUSTRY_MAP = {"UPTREND": "LEADING", "TURNING UP": "IMPROVING", "DOWNTREND": "STAGE 4"}
INDUSTRY_PTS = {"LEADING": 12, "IMPROVING": 10, "NEUTRAL": 4, "STAGE 4": 0}


def industry_stages(df: pd.DataFrame) -> Dict[str, str]:
    """ticker -> LEADING / IMPROVING / NEUTRAL / STAGE 4, from the industry-alert status ladder."""
    from utils.industry_alerts import all_sub_industry_readings
    from utils.nifty1000_list import SUB_INDUSTRY_MAP
    rd = all_sub_industry_readings(df)
    stat = dict(zip(rd["industry"], rd["status"])) if not rd.empty else {}
    return {t: INDUSTRY_MAP.get(stat.get(g), "NEUTRAL") for t, g in SUB_INDUSTRY_MAP.items()}


# ----------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------
def _b(x) -> bool:
    return bool(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else False


def _f(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    return x


def score_table(df: pd.DataFrame, regime: Dict[str, Any], fund: Optional[pd.DataFrame] = None,
                ind: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """The document's 100-point score, grade and hard gates, one row per stock with SIMBA fields."""
    if "simba_stage" not in df.columns:
        return pd.DataFrame()
    x = df[df["simba_stage"].notna()].copy()
    if fund is not None and not fund.empty:
        x = x.merge(fund, on="ticker", how="left")
    ind = ind or {}
    r6 = pd.to_numeric(x.get("return_6m"), errors="coerce")
    x["rs_rating"] = (r6.rank(pct=True) * 98 + 1).round(0)
    sec_pe = pd.to_numeric(x.get("pe"), errors="coerce").where(lambda s: s > 0)
    x["pe_vs_sector"] = sec_pe / sec_pe.groupby(x.get("sector", pd.Series("", index=x.index))).transform("median")
    reg = regime.get("regime", "UNKNOWN")
    rows = []
    for _, r in x.iterrows():
        g_recent = _b(r.get("simba_g")) or (_f(r.get("simba_g_since")) <= CONF_BARS)
        m_recent = _f(r.get("simba_m_since")) <= 10
        istage = ind.get(r["ticker"], "NEUTRAL")
        er = _b(r.get("er"))
        top = REGIME_PTS.get(reg, 0) + INDUSTRY_PTS[istage]
        gh = (14 if g_recent else 0) + (3 if _b(r.get("simba_vcp")) else 0) + \
             (3 if (_f(r.get("rs_rating")) >= 80 or (_b(r.get("simba_mrs_cross")))) else 0)
        ma = 0
        if m_recent:
            ma = 12 + (3 if _f(r.get("simba_vol_x_m")) >= 3 else 0) + (2 if _b(r.get("simba_dryup")) else 0) + \
                 (2 if r.get("simba_good_wc") is True else 0)
        q = 0
        if er:
            q += 12 if _b(r.get("er_oplev")) else 10
        q += 4 if _f(r.get("roce")) >= 20 else 0
        q += 4 if _f(r.get("opm_cv")) <= 0.25 else 0
        pe_rel = _f(r.get("pe_vs_sector"))
        q += 4 if (np.isfinite(pe_rel) and pe_rel <= 2.0) else 0
        q = min(q, 25)
        flow = 4 if _f(r.get("promoters")) >= 60 else 0
        mtf = {3: 5, 2: 3}.get(int(_f(r.get("simba_mtf")) if np.isfinite(_f(r.get("simba_mtf"))) else 0), 0)
        total = top + gh + ma + q + flow + mtf

        px = _f(r.get("currentPrice", r.get("price")))
        adv = _f(r.get("averageVolume")) * px
        base_low = _f(r.get("simba_base_low"))
        stop_pct = (px - base_low) / px * 100 if np.isfinite(base_low) and px > 0 else np.nan
        gates = []
        if not (adv >= 5e7):
            gates.append("liquidity < ₹5 cr/day")
        if px < 20:
            gates.append("price < ₹20")
        if _f(r.get("de")) > 1 or (_f(r.get("cfo_ebitda")) < 0.6):
            gates.append("forensics (D/E > 1 or CFO/EBITDA < 60%)")
        if reg == "BEAR CONTRACTION" and not (_f(r.get("rs_rating")) >= 90 and er):
            gates.append("bear contraction")
        if _b(r.get("simba_ext")) or _f(r.get("simba_ext_ma150")) > MAX_EXT:
            gates.append("extended")
        if not (stop_pct <= SL_INIT):
            gates.append("stop > 15% below")
        grade = "A+" if total >= 80 else "A" if total >= 65 else "B" if total >= 50 else "—"
        if gates and grade != "—":
            grade = f"{grade} (gated)"
        rows.append({"ticker": r["ticker"], "score": total, "grade": grade, "gates": "; ".join(gates),
                     "pts_topdown": top, "pts_ghoomega": gh, "pts_machayega": ma, "pts_quality": q,
                     "pts_float": flow, "pts_mtf": mtf, "industry_stage": istage, "stop_pct": stop_pct})
    s = pd.DataFrame(rows)
    return x.merge(s, on="ticker", how="left").sort_values("score", ascending=False)
