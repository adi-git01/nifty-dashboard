"""
Breakout / industry / edge columns for the Trend Scanner.

Evidence (2016-2026, point-in-time top 1000; analysis/BREAKOUT_SECTOR_VERDICT.md
and analysis/scanner_tags_study.csv):
  * a new 52-week or all-time closing high in a LEADER sub-industry (rotation
    score >= 70) beat the typical stock by ~+2 to +3 pp over 3 months and
    ~+3.5 to +4.5 pp over 6 months; a volume anchor added ~+1 pp;
  * the same breakout in a laggard industry was worth about a third of that;
  * the scanner's entry tags differ less than their names suggest: Actionable
    +0.6 pp (3m), Pullback +0.3, Extended +1.0 (weak first 2 weeks), Fading
    +0.2, Weak -0.9 -- and every tag does better in a leader industry.
  * Preferring breakouts inside the live portfolio engine did NOT beat its
    6-month RS ranking -- these columns are for discretionary screening.

Engine side  hi52_breakout(df)   fresh 52-week-high breakout from daily bars
             (ATH breakouts: utils/ath.py)
UI side      add_breakout_tags(df)  ind_score, bo_type, bo_days, bo_anchor (volume anchor
             AND held: no close since below the breakout day's low), bo_held,
             breakout_label (with the breakout day's volume), dist_ath,
             edge_3m / edge_6m / edge_years
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

FRESH_SESSIONS = 10          # "fresh" = broke out within the last 2 weeks
BASE_SESSIONS = 20           # sessions without a new high before a breakout
MIN_HISTORY = 200
LEADER, LAGGARD = 70, 40     # rotation heatmap bands (score_0_100)
TAG_STUDY = "analysis/scanner_tags_study.csv"

BREAKOUT_OPTIONS = ["ATH breakout", "52W breakout"]
BREAKOUT_HELP = ("Fresh = a new closing high in the last 2 weeks after >= 20 sessions without one. "
                 "ATH = all-time high (data/ath_levels.csv), 52W = 52-week high. In a leader industry these beat "
                 "the typical stock by ~+2 to +3 pp over 3 months and ~+4 pp over 6 months (2016-2026). "
                 "The label ends with the breakout day's volume vs its 50-day median; 🔊 = a volume anchor "
                 "(>= 3x, >= Rs 5 cr turnover, CLV >= 0.5), worth ~+1 pp more. A volume spike WITHOUT a breakout "
                 "has no edge of its own, so it is not flagged. Blank = no fresh breakout. "
                 "Nothing selected = no breakout filter.")


# ----------------------------------------------------------------------------
# engine side
# ----------------------------------------------------------------------------
def hi52_breakout(df: pd.DataFrame) -> dict:
    """
    Most recent 52-week closing-high breakout within FRESH_SESSIONS.
    Returns {'hi52_bo_days': sessions ago (0 = today) or NaN,
             'hi52_bo_anchor': True/False (NaN if no fresh breakout),
             'hi52_bo_volx': breakout-day volume / prior 50-session median,
             'hi52_bo_held': no close since below the breakout day's low}.
    """
    out = {"hi52_bo_days": np.nan, "hi52_bo_anchor": np.nan, "hi52_bo_volx": np.nan, "hi52_bo_held": np.nan}
    if df is None or "Close" not in df:
        return out
    df = clean_bars(df)
    if len(df) < MIN_HISTORY + 1:
        return out
    c = df["Close"].astype(float).values
    n = len(c)
    prior_max = pd.Series(c).shift(1).rolling(251, min_periods=MIN_HISTORY).max().values
    new_high = np.zeros(n, dtype=bool)
    ok = ~np.isnan(prior_max)
    new_high[ok] = c[ok] >= prior_max[ok]
    for i in range(n - 1, max(n - 1 - FRESH_SESSIONS, BASE_SESSIONS) - 1, -1):
        if new_high[i] and not new_high[i - BASE_SESSIONS:i].any() and ok[i - BASE_SESSIONS]:
            out["hi52_bo_days"] = n - 1 - i
            out["hi52_bo_anchor"], out["hi52_bo_volx"] = vol_stats(df, i)
            out["hi52_bo_held"] = held_since(df, i)
            return out
    return out


def squeeze_release(df: pd.DataFrame) -> dict:
    """
    Most recent Bollinger squeeze release within FRESH_SESSIONS: the 20-day band width sat below its own
    20th percentile (last 252 sessions) at some point in the prior 5 sessions, and the close broke above
    the upper band (20, 2). Same definition as entry_exit_study.py S5, where it passed in both eras.
    Returns {'sq_days': sessions since the first release day in that window, or NaN}.
    """
    out = {"sq_days": np.nan}
    if df is None or "Close" not in df:
        return out
    c = clean_bars(df)["Close"].astype(float)
    if len(c) < 150:
        return out
    m20, s20 = c.rolling(20, min_periods=15).mean(), c.rolling(20, min_periods=15).std()
    bbw = 4 * s20 / m20
    pct = bbw.rolling(252, min_periods=120).rank(pct=True) * 100
    rel = (pct.rolling(5, min_periods=3).min().shift(1) < 20) & (c > m20 + 2 * s20)
    hits = np.flatnonzero(rel.values[-(FRESH_SESSIONS + 1):])
    if len(hits):
        out["sq_days"] = int(FRESH_SESSIONS - hits[0]) if len(c) > FRESH_SESSIONS else np.nan   # first release day
    return out


def clean_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows without a close, so 'sessions ago' counts real sessions -- the same
    bars the Technicals tab (utils/forensics.py) reads."""
    return df[df["Close"].notna()] if "Close" in df else df


def held_since(df: pd.DataFrame, i: int) -> bool:
    """True when no close after bar i is below bar i's low (the Technicals tab's VETO line)."""
    if "Low" not in df.columns:
        return True
    low = float(df["Low"].iloc[i])
    after = df["Close"].astype(float).values[i + 1:]
    return bool(not (after[~np.isnan(after)] < low).any()) if np.isfinite(low) else True


def vol_stats(df: pd.DataFrame, i: int) -> tuple[bool, float]:
    """
    (volume anchor?, volume multiple) for bar i. Anchor: volume >= 3x the prior
    50-session median, turnover >= Rs 5 cr, CLV >= 0.5.
    """
    if not {"Volume", "High", "Low"} <= set(df.columns) or i < 30:
        return False, np.nan
    v = df["Volume"].astype(float).values
    med = np.nanmedian(v[max(0, i - 50):i])
    volx = v[i] / med if med > 0 else np.nan
    h, l, c = float(df["High"].iloc[i]), float(df["Low"].iloc[i]), float(df["Close"].iloc[i])
    clv = (c - l) / (h - l) if h > l else np.nan
    return bool(med > 0 and v[i] >= 3 * med and c * v[i] >= 5e7 and clv >= 0.5), round(float(volx), 1)


# ----------------------------------------------------------------------------
# UI side
# ----------------------------------------------------------------------------
def load_industry_scores(path: str = "data/sub_industry_rotation.csv") -> dict:
    """Latest score_0_100 per sub-industry from the rotation history."""
    try:
        r = pd.read_csv(path)
        r["record_date"] = pd.to_datetime(r["record_date"], format="mixed")
        last = r[r.record_date == r.record_date.max()]
        return dict(zip(last["sub_industry"], pd.to_numeric(last["score_0_100"], errors="coerce")))
    except Exception:
        return {}


def load_tag_edges(path: str = TAG_STUDY) -> dict:
    """{(view, tag): (3m pp, 6m pp, 'years')} from the scanner-tag backtest."""
    if not os.path.exists(path):
        return {}
    t = pd.read_csv(path)
    return {(r["view"], r["case"]): (r.get("3mo vsTypical"), r.get("6mo vsTypical"), r.get("yrs 3mo>0"))
            for r in t.to_dict("records")}


def _edge_view(band: str, fresh: bool) -> list[str]:
    """Most specific backtest cell first, broader fallbacks after."""
    views = []
    if fresh and band == "leader":
        views.append("52w/ATH breakout, leader industry")
    if fresh:
        views.append("52w/ATH breakout in last 2 weeks")
    if band:
        views.append(f"{band} industry")
    views.append("all stocks")
    return views


def add_breakout_tags(df: pd.DataFrame, scores: dict | None = None, sub_map: dict | None = None,
                      ath: dict | None = None, edges: dict | None = None) -> pd.DataFrame:
    """Adds the scanner's breakout / industry / edge columns. Safe on an older cache."""
    out = df.copy()
    if scores is None:
        scores = load_industry_scores()
    if sub_map is None:
        from utils.nifty1000_list import SUB_INDUSTRY_MAP as sub_map
    if ath is None:
        from utils.ath import load_ath
        ath = load_ath()
    if edges is None:
        edges = load_tag_edges()
    col = lambda c: out[c] if c in out else pd.Series(np.nan, index=out.index)

    tick = col("ticker")
    out["ind_score"] = pd.to_numeric(tick.map(sub_map).map(scores), errors="coerce")
    band = np.select([out.ind_score >= LEADER, out.ind_score < LAGGARD, out.ind_score.notna()],
                     ["leader", "laggard", "mid"], "")

    d52 = pd.to_numeric(col("hi52_bo_days"), errors="coerce")
    dath = pd.to_numeric(col("ath_bo_days"), errors="coerce")
    f52 = d52.notna() & (d52 <= FRESH_SESSIONS)
    fath = dath.notna() & (dath <= FRESH_SESSIONS)
    out["bo_type"] = np.select([fath, f52], ["ATH", "52W"], "")
    out["bo_days"] = np.where(fath, dath, np.where(f52, d52, np.nan))
    # held = no close since below the breakout day's low; unknown (older cache) counts as held
    held = np.where(fath, col("ath_bo_held"), np.where(f52, col("hi52_bo_held"), np.nan))
    out["bo_held"] = pd.Series(held, index=out.index).map(lambda x: not (isinstance(x, (bool, np.bool_)) and not x))
    out["bo_anchor"] = ((fath & col("ath_bo_anchor").eq(True)) | (~fath & f52 & col("hi52_bo_anchor").eq(True))) \
        & out["bo_held"]
    out["bo_volx"] = pd.to_numeric(np.where(fath, col("ath_bo_volx"), np.where(f52, col("hi52_bo_volx"), np.nan)),
                                   errors="coerce")
    out["breakout_label"] = [
        "" if not t else f"⚡ {t} · {int(d)}d ago" + (
            f" · 🔊 {x:.1f}× vol" if a else f" · {x:.1f}× vol" if pd.notna(x) else "") + (
            "" if hd else " · ❌ failed (closed below breakout-day low)")
        for t, d, a, x, hd in zip(out.bo_type, out.bo_days, out.bo_anchor, out.bo_volx, out.bo_held)]

    # % from all-time high: engine value when present, else from the stored table
    price = pd.to_numeric(col("price").fillna(col("currentPrice")), errors="coerce")
    stored = pd.to_numeric(tick.map({t: r.get("ath_close") for t, r in ath.items()}), errors="coerce")
    athv = np.fmax(stored, price)
    out["dist_ath"] = pd.to_numeric(col("dist_ath"), errors="coerce").fillna((price / athv - 1) * 100)

    # historical edge of this tag x industry band x breakout cell
    tags = col("entry_label").astype(str).str.replace(r"^\S+\s", "", regex=True)
    e3, e6, ey = [], [], []
    for tg, b, fr in zip(tags, band, (fath | f52)):
        hit = next((edges[(v, tg)] for v in _edge_view(b, fr)
                    if (v, tg) in edges and pd.notna(edges[(v, tg)][0])), (np.nan, np.nan, ""))
        e3.append(hit[0]); e6.append(hit[1]); ey.append(hit[2] if isinstance(hit[2], str) else "")
    out["edge_3m"], out["edge_6m"], out["edge_years"] = e3, e6, ey
    return out


# ----------------------------------------------------------------------------
# momentum entry / size / loss-maker / absorption (entry_exit_study.py, analysis/ENTRY_EXIT_VERDICT.md)
# ----------------------------------------------------------------------------
MOMO_HELP = ("Top-10% 6-month return + Gate 1 trend, AND a fresh (last 2 weeks) 52-week/all-time-high breakout "
             "that has held (🚀) or a Bollinger-squeeze release (🎯). 2016-26, every momentum name: bought on the "
             "breakout +13.4 / +12.6 pp over 6 months vs the typical stock (2016-20 / 2021-26), squeeze +15.4 / "
             "+12.4, vs +11.2 / +10.3 buying at once and +12.1 / +10.1 buying blind after the same wait. Fires for "
             "about half of momentum names. CLV, volume anchor alone, gaps, pocket pivots, NR7, pullbacks and "
             "RSI(2) dips added nothing. Discretionary only: as an ENGINE rule (buy only these) it cut the live book's "
             "10-year CAGR from 27.5% to 22.5% -- the 6-month ranking already carries this momentum.")
SIZE_HELP = ("Market-cap rank in the loaded universe: Large 1-100, Mid 101-250, Small 251-500, Micro 501+. "
             "6-month edge vs the typical stock, 2016-20 / 2021-26: momentum Large +7.4 / -4.0 (no edge since 2021), "
             "Mid +12.1 / +4.3, Small +7.4 / +2.6, Micro +2.5 / +4.2; breakout + volume anchor + leader industry "
             "works best in Mid (+6.2 both eras); v6 BUY NOW is the steadiest in Small/Micro (positive every year).")
FLAG_HELP = ("⚠ loss-maker: latest published annual EPS <= 0 -- momentum names that lose money did worse "
             "(-4.8 / -1.4 pp over 6 months) and gave their gains back more often -- but skipping them in the live engine "
             "cut 10-year CAGR by ~2 pp (the turnarounds pay for the rest under an MA50 exit), so it is a caution, "
             "not a filter. 🟢 absorption: pullbacks on "
             "drying volume after the volume anchor (Technicals tab) -- the best chart sign on momentum names "
             "(+8.4 / +6.0 pp). ⛏ cheap commodity: a low P/E during a commodity run is peak-cycle earnings, not value.")
TIERS = ((100, "Large"), (250, "Mid"), (500, "Small"))


def latest_eps(path: str = "data/fundamentals_annual.csv.gz", lag_days: int = 92) -> dict:
    """Latest annual EPS whose results were public (period end + 92 days), per ticker."""
    try:
        A = pd.read_csv(path, parse_dates=["period"])
        A = A[(A.field == "eps") & (A.period + pd.Timedelta(days=lag_days) <= pd.Timestamp.today())]
        return A.sort_values("period").groupby("ticker").value.last().to_dict()
    except Exception:
        return {}


def add_momentum_tags(df: pd.DataFrame, eps: dict | None = None) -> pd.DataFrame:
    """mom_rank, size_tier, momo_entry (🚀/🎯 on momentum names), loss_maker, absorption, watch_flags."""
    out = df.copy()
    col = lambda c: out[c] if c in out else pd.Series(np.nan, index=out.index)
    out["mom_rank"] = pd.to_numeric(col("return_6m"), errors="coerce").rank(pct=True) * 100
    price = pd.to_numeric(col("price").fillna(col("currentPrice")), errors="coerce")
    if "fx_gate1" in out:
        gate1 = out["fx_gate1"].fillna(False).astype(bool)
    else:
        gate1 = (price > pd.to_numeric(col("fiftyDayAverage"), errors="coerce")) & \
                (price > pd.to_numeric(col("twoHundredDayAverage"), errors="coerce")) & \
                (pd.to_numeric(col("dist_52w"), errors="coerce") >= -15)
    rank = pd.to_numeric(col("marketCap"), errors="coerce").rank(ascending=False)
    out["size_tier"] = np.select([rank <= n for n, _ in TIERS] + [rank.notna()], [l for _, l in TIERS] + ["Micro"], "")

    bo_type = col("bo_type").fillna("") if "bo_type" in out else pd.Series("", index=out.index)
    bo_ok = (bo_type != "") & (col("bo_held").map(lambda x: not (isinstance(x, (bool, np.bool_)) and not x)))
    sq = pd.to_numeric(col("sq_days"), errors="coerce")
    momo = (out["mom_rank"] >= 90) & gate1
    out["momo_entry"] = [
        "" if not m else (f"🚀 {t} breakout {int(d)}d" if b else (f"🎯 squeeze {int(s)}d" if pd.notna(s) else ""))
        for m, b, t, d, s in zip(momo, bo_ok, bo_type, col("bo_days"), sq)]

    if eps is None:
        eps = latest_eps()
    e = pd.to_numeric(col("ticker").map(eps), errors="coerce")
    out["loss_maker"] = e.le(0)
    out["absorption"] = col("fx_signature").eq("ABSORPTION")
    try:
        from utils.valuation import COMMODITY_GRAN
        from utils.nifty1000_list import SUB_INDUSTRY_MAP
        commodity = col("ticker").map(SUB_INDUSTRY_MAP).isin(COMMODITY_GRAN)
    except Exception:
        commodity = pd.Series(False, index=out.index)
    pe = pd.to_numeric(col("pe"), errors="coerce")
    cheap_comm = commodity & momo & (pe > 0) & (pe < 10)
    out["watch_flags"] = [" ".join(x for x in (
        "⚠ loss-maker" if lm else "", "🟢 absorption" if ab else "", "⛏ cheap commodity" if cc else "",
        "Large-cap momentum: no edge since 2021" if (t == "Large" and m) else "") if x)
        for lm, ab, cc, t, m in zip(out.loss_maker, out.absorption, cheap_comm, out.size_tier, momo)]
    return out
