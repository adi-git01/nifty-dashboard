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
  group       LEADING (score >= 60 with >= 50% of members above the 50-day, not falling),
              WEAKENING (the same but down 10+ in a month), IMPROVING (not a leader, up 10+,
              and at least 40% above the 50-day), LAGGING (the rest)

Evidence (Feb-Oct 2026 snapshots, 147 start dates, 21-session forward median member return vs
all stocks): top-20% by 3-month RS +0.68 pp, by this blend +0.63, by the old score +0.53; the
bottom 80% are flat. That is one regime and seven independent months -- a guide to where
strength is, not a tested edge. The 2021-26 monthly study found UPTREND status the
better-evidenced state (analysis/INDUSTRY_TURN_VERDICT.md).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

MIN_MEMBERS = 3
SMALL_GROUP = 6
MIN_BREADTH = 50            # % of members above the 50-day average needed to count as a leader
MIN_IMPROVING_BREADTH = 40  # % of members above the 50-day average needed to count as improving
WEIGHTS = {"rs_3m": 0.35, "above_200": 0.25, "above_50": 0.20, "near_high": 0.20}
GROUP_ICON = {"LEADING": "🟢 Leading", "IMPROVING": "🟡 Improving", "WEAKENING": "🟣 Weakening", "LAGGING": "🔴 Lagging"}


def _metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "ticker" not in df.columns:
        return pd.DataFrame()
    from utils.industry_alerts import classify
    from utils.nifty1000_list import SUB_INDUSTRY_MAP
    d = df.drop_duplicates("ticker")
    if "extended" in d.columns:
        d = d[d["extended"].ne(True)]
    d = d.assign(industry=d["ticker"].map(SUB_INDUSTRY_MAP))
    d = d[d["industry"].notna() & (d["industry"] != "Unknown")]
    if d.empty:
        return pd.DataFrame()
    num = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d.columns else pd.Series(np.nan, index=d.index)
    px = num("currentPrice").fillna(num("price"))
    px_ok = px.notna() & (px > 0)
    m50 = num("fiftyDayAverage")
    m200 = num("twoHundredDayAverage")
    d52 = num("dist_52w")
    crs = num("comp_rs")
    rs3 = num("rs_3m")
    d = d.assign(
        _a50=(px > m50).where(px_ok & m50.notna() & (m50 > 0)),
        _a200=(px > m200).where(px_ok & m200.notna() & (m200 > 0)),
        _hi=(d52 > -10).where(px_ok & d52.notna()),
        _buy=d.get("dna_signal", pd.Series("", index=d.index)).astype(str).eq("BUY"),
        _rs=crs.where(crs.abs() <= 200),
        _rs3=rs3.where(rs3.abs() <= 300),
    )
    rows = []
    for g, x in d.groupby("industry"):
        if len(x) < MIN_MEMBERS or int(x._a50.notna().sum()) < MIN_MEMBERS:
            continue
        a50 = float(x._a50.mean() * 100)
        a200 = float(x._a200.mean() * 100) if x._a200.notna().any() else np.nan
        nh = float(x._hi.mean() * 100) if x._hi.notna().any() else np.nan
        rs = float(x._rs.median()) if x._rs.notna().any() else np.nan
        r3 = float(x._rs3.median()) if x._rs3.notna().any() else np.nan
        lead = x[x._a50.eq(True)].sort_values("_rs", ascending=False).head(3)
        rows.append(dict(
            industry=g, members=len(x),
            rs_3m=round(r3, 1) if np.isfinite(r3) else np.nan,
            rs_1m=round(rs, 1) if np.isfinite(rs) else np.nan,
            above_50=round(a50, 0),
            above_200=round(a200, 0) if np.isfinite(a200) else np.nan,
            near_high=round(nh, 0) if np.isfinite(nh) else np.nan,
            buy_pct=round(float(x._buy.mean() * 100), 0),
            status=classify(a50, a200, rs),
            leaders=", ".join(t.replace(".NS", "") for t in lead.ticker),
            _raw_a50=a50, _raw_a200=a200, _raw_hi=nh, _raw_rs3=r3,
        ))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    raw_map = {"rs_3m": "_raw_rs3", "above_200": "_raw_a200", "above_50": "_raw_a50", "near_high": "_raw_hi"}
    num_sum = pd.Series(0.0, index=out.index)
    w_sum = pd.Series(0.0, index=out.index)
    for c, w in WEIGHTS.items():
        r = out[raw_map[c]].rank(pct=True)
        ok = r.notna()
        num_sum = num_sum + r.fillna(0.0) * w
        w_sum = w_sum + ok.astype(float) * w
    blend = (num_sum / w_sum).where(w_sum > 0)
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
    # also be strong in absolute terms: at least half its members above their 50-day average; and
    # improving industries must meet the 40% basing breadth floor so downtrends aren't mislabeled.
    strong = (now.score >= 60) & (now._raw_a50 >= MIN_BREADTH)
    improving = (ch >= 10) & (now._raw_a50 >= MIN_IMPROVING_BREADTH)
    now["group"] = np.select([strong & (ch > -10), strong, improving], ["LEADING", "WEAKENING", "IMPROVING"], "LAGGING")
    now["small"] = now["members"] < SMALL_GROUP
    now = now.drop(columns=[c for c in now.columns if c.startswith("_raw_")])
    return now.sort_values(["score", "rs_3m"], ascending=False).reset_index(drop=True)


def month_ago_snapshot(sessions: int = 21, current_df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """The snapshot about `sessions` unique trading days before `current_df` (or the newest one), or None."""
    from utils import snapshots
    path = snapshots.snapshot_n_sessions_ago(sessions, current_df=current_df)
    if not path:
        return None
    want = ("ticker", "currentPrice", "price", "fiftyDayAverage", "twoHundredDayAverage",
            "dist_52w", "dna_signal", "comp_rs", "rs_3m", "extended", "bar_date")
    try:
        import pyarrow.parquet as pq
        names = set(pq.read_schema(path).names)
        cols = [c for c in want if c in names]
        return pd.read_parquet(path, columns=cols) if cols else pd.read_parquet(path)
    except Exception:
        try:
            return pd.read_parquet(path)
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
