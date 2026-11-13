"""
Industry leadership board: which sub-industries lead, which are improving, which are fading.

The stored rotation score (trading_engine.generate_sub_industry_rotation) ranks industries on
short-term composite RS only. It says who is ahead this month, not whether the industry is in an
uptrend, how broad the strength is, or whether it is gaining or losing ground. This board adds
those, from the daily snapshot alone:

  rs_3m       median 3-month return vs Nifty of the members (pp)
  rs_1m       median composite RS (1w/1m/3m blend) of the members (pp)
  above_50 / above_200   % of members above their 50- / 200-day average
  near_high   % of members within 10% of their 52-week high
  buy_pct     % of members with the engine's BUY signal
  status      the industry-alert ladder: DOWNTREND, BASING, TURNING UP, UPTREND, WEAKENING
  score       0-100 leadership score = percentile rank across industries of
              35% rs_3m + 25% above_200 + 20% above_50 + 20% near_high
  change      score now minus score ~21 sessions ago
  group       LEADING (score >= 60 with at least half the members above the 50-day, not falling),
              WEAKENING (the same but down 10+ in a month), IMPROVING (not a leader but up 10+),
              LAGGING (the rest)

Evidence (Feb-Oct 2026 snapshots, 147 start dates, 21-session forward median member return vs
all stocks): top-20% by 3-month RS +0.68 pp, by this blend +0.63, by the old score +0.53; the
bottom 80% are flat. That is one regime and seven independent months -- a guide to where
strength is, not a tested edge. The 2021-26 monthly study found UPTREND status the
better-evidenced state (analysis/INDUSTRY_TURN_VERDICT.md).
"""
from __future__ import annotations

import glob
from typing import Optional

import numpy as np
import pandas as pd

MIN_MEMBERS = 3
SMALL_GROUP = 6
MIN_BREADTH = 50           # % of members above the 50-day average needed to count as a leader
WEIGHTS = {"rs_3m": 0.35, "above_200": 0.25, "above_50": 0.20, "near_high": 0.20}
GROUP_ICON = {"LEADING": "🟢 Leading", "IMPROVING": "🟡 Improving", "WEAKENING": "🟣 Weakening", "LAGGING": "🔴 Lagging"}


def _metrics(df: pd.DataFrame) -> pd.DataFrame:
    from utils.industry_alerts import classify
    from utils.nifty1000_list import SUB_INDUSTRY_MAP
    d = df.assign(industry=df["ticker"].map(SUB_INDUSTRY_MAP))
    d = d[d["industry"].notna() & (d["industry"] != "Unknown")]
    px = pd.to_numeric(d.get("currentPrice", d.get("price")), errors="coerce")
    num = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d.columns else pd.Series(np.nan, index=d.index)
    d = d.assign(_a50=(px > num("fiftyDayAverage")).where(num("fiftyDayAverage").notna()),
                 _a200=(px > num("twoHundredDayAverage")).where(num("twoHundredDayAverage").notna()),
                 _hi=(num("dist_52w") > -10).where(num("dist_52w").notna()),
                 _buy=d.get("dna_signal", pd.Series("", index=d.index)).astype(str).eq("BUY"),
                 _rs=num("comp_rs").where(num("comp_rs").abs() <= 200), _rs3=num("rs_3m").where(num("rs_3m").abs() <= 300))
    rows = []
    for g, x in d.groupby("industry"):
        if len(x) < MIN_MEMBERS:
            continue
        a50, a200 = float(x._a50.mean() * 100), float(x._a200.mean() * 100)
        rs = float(x._rs.median())
        lead = x[x._a50 == True].sort_values("_rs", ascending=False).head(3)
        rows.append(dict(industry=g, members=len(x), rs_3m=round(float(x._rs3.median()), 1), rs_1m=round(rs, 1),
                         above_50=round(a50, 0), above_200=round(a200, 0), near_high=round(float(x._hi.mean() * 100), 0),
                         buy_pct=round(float(x._buy.mean() * 100), 0), status=classify(a50, a200, rs),
                         leaders=", ".join(t.replace(".NS", "") for t in lead.ticker)))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    blend = sum(w * out[c].rank(pct=True) for c, w in WEIGHTS.items())
    out["score"] = (blend.rank(pct=True) * 100).round(0)
    return out


def board(df: pd.DataFrame, prev: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Leadership board for snapshot `df`; `prev` (about a month earlier) adds change and group."""
    now = _metrics(df)
    if now.empty:
        return now
    now["change"] = np.nan
    if prev is not None and not prev.empty:
        p = _metrics(prev)
        if not p.empty:
            now["change"] = now["industry"].map(p.set_index("industry")["score"])
            now["change"] = now["score"] - now["change"]
            now["status_was"] = now["industry"].map(p.set_index("industry")["status"])
    ch = now["change"].fillna(0)
    # The score is relative, so in a weak market the least-bad industries rank high. A leader must
    # also be strong in absolute terms: at least half its members above their 50-day average.
    strong = (now.score >= 60) & (now.above_50 >= MIN_BREADTH)
    now["group"] = np.select([strong & (ch > -10), strong, ch >= 10], ["LEADING", "WEAKENING", "IMPROVING"], "LAGGING")
    now["small"] = now["members"] < SMALL_GROUP
    return now.sort_values(["score", "rs_3m"], ascending=False).reset_index(drop=True)


def month_ago_snapshot(sessions: int = 21) -> Optional[pd.DataFrame]:
    """The snapshot about `sessions` trading days before the newest one, or None."""
    files = sorted(glob.glob("data/cache/market_master_*.parquet"))
    if len(files) <= sessions:
        return None
    cols = ["ticker", "currentPrice", "fiftyDayAverage", "twoHundredDayAverage", "dist_52w", "dna_signal", "comp_rs", "rs_3m"]
    try:
        return pd.read_parquet(files[-1 - sessions], columns=cols)
    except Exception:
        try:
            return pd.read_parquet(files[-1 - sessions])
        except Exception:
            return None


def stock_industry_view(board_df: pd.DataFrame, ticker: str) -> Optional[dict]:
    """The board row for a stock's own industry (for the stock report and the funnel)."""
    from utils.nifty1000_list import SUB_INDUSTRY_MAP
    g = SUB_INDUSTRY_MAP.get(ticker)
    if not g or board_df is None or board_df.empty:
        return None
    r = board_df[board_df.industry == g]
    if r.empty:
        return None
    out = r.iloc[0].to_dict()
    out["rank"] = int(r.index[0]) + 1
    out["of"] = len(board_df)
    return out
