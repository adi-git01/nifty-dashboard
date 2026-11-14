"""
Strong names: the RS Divergence and Earnings Drift lists, as idea lists with context.

rs_divergence(T, days)   stocks that rose on a day the Nifty fell, from data/rs_divergence_log.csv,
                         one row per stock: how many red days it has held up on in the window,
                         the latest one, return since the first, and where it stands in the funnel
earnings_drift(T, days)  stocks that jumped 4%+ on 2.5x volume (data/earnings_shock_log.csv), one
                         row per stock: the jump, return since, whether the price still holds the
                         jump day's close, and the funnel status

T is the funnel table (utils/funnel.table) for today's snapshot.

What the logs say so far (analysis/PAGE_AUDIT_2026_11.md): earnings jumps drifted on only when
the stock was already on the Trend Scanner BUY list (+5.6 pp over 63 sessions v +0.8 pp when it
was not); a bigger jump did not help. RS Divergence has clean data only since September 2026,
too short to judge. Both are ways to FIND strong names; the funnel columns say which are buyable.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

RS_LOG, ES_LOG = "data/rs_divergence_log.csv", "data/earnings_shock_log.csv"
CTX = ["industry", "ind_group", "scanner", "technicals", "comp_rs", "dist_52w", "price", "s_industry", "s_trend", "s_business", "s_setup", "risk_pct"]


def _ctx(x: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    t = T.drop_duplicates("ticker").set_index("ticker")
    for c in CTX:
        x[c] = x.ticker.map(t[c]) if c in t else np.nan
    ok = lambda c: x[c].eq(True)
    biz = ~x["s_business"].eq(False)
    x["funnel"] = np.select([ok("s_trend") & ok("s_setup") & ok("s_industry") & biz, ok("s_trend") & ok("s_setup") & biz,
                             ok("s_trend") & ~biz, ok("s_trend")],
                            ["✅ Buyable", "🟡 Buyable, weak industry", "⚠️ Uptrend, business flags", "👀 Uptrend, no entry"],
                            "— Not in uptrend")
    return x


def _recent(path: str, days: int) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    L = pd.read_csv(path)
    L["signal_date"] = pd.to_datetime(L["signal_date"], errors="coerce")
    L = L.dropna(subset=["signal_date"])
    return L[L.signal_date >= L.signal_date.max() - pd.Timedelta(days=days)]


def rs_divergence(T: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    L = _recent(RS_LOG, days)
    if L.empty:
        return L
    L = L.sort_values("signal_date")
    g = L.groupby("ticker")
    x = pd.DataFrame({"name": g.name.last(), "red_days": g.size(), "first": g.signal_date.first(), "latest": g.signal_date.last(),
                      "first_price": g.signal_price.first(), "best_gap": g.delta_rs.max(), "avg_gap": g.delta_rs.mean()}).reset_index()
    x = _ctx(x, T)
    x["since_first"] = (x.price / x.first_price - 1) * 100
    x["today"] = x.latest == L.signal_date.max()
    return x.sort_values(["red_days", "avg_gap"], ascending=False).reset_index(drop=True)


def earnings_drift(T: pd.DataFrame, days: int = 45) -> pd.DataFrame:
    L = _recent(ES_LOG, days)
    if L.empty:
        return L
    L = L.sort_values("signal_date").drop_duplicates("ticker", keep="first")      # the first jump in the window
    x = L[["ticker", "name", "signal_date", "signal_price", "jump_pct", "vol_mult"]].rename(columns={"signal_date": "jump_date"})
    x = _ctx(x.reset_index(drop=True), T)
    x["since"] = (x.price / x.signal_price - 1) * 100
    x["holds"] = x.price >= x.signal_price
    x["sessions"] = [len(pd.bdate_range(d, L.signal_date.max())) - 1 for d in x.jump_date]
    return x.sort_values(["s_trend", "since"], ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ every signal for one stock
# label, file, date column, price column, return column, detail builder
_f = lambda v, fmt: "" if pd.isna(v) else fmt.format(v)
SOURCES = [
    ("💪 RS Divergence", RS_LOG, "signal_date", "signal_price", "return_since_signal",
     lambda r: f"stock {_f(r.get('stock_ret_on_day'), '{:+.1f}%')} v Nifty {_f(r.get('nifty_ret_on_day'), '{:+.1f}%')}"),
    ("⚡ Earnings Shock", ES_LOG, "signal_date", "signal_price", "return_since_signal",
     lambda r: f"jump {_f(r.get('jump_pct'), '{:+.1f}%')} on {_f(r.get('vol_mult'), '{:.1f}×')} volume"),
    ("🔄 Turnaround Catalyst", "data/tc_log.csv", "signal_date", "signal_price", "return_since_signal",
     lambda r: f"{r.get('pattern', '')}, jump {_f(r.get('jump_pct'), '{:+.1f}%')}"),
    ("🎯 Turnaround Radar", "data/ias_signal_log.csv", "signal_date", "signal_price", "return_since_signal",
     lambda r: f"tier {r.get('signal_tier', '')}, peak {r.get('peak_tier', '')}, now {r.get('status', '')}"),
    ("🚀 Breakout", "data/breakout_log.csv", "breakout_date", "signal_price", "return_pct",
     lambda r: str(r.get('kind', '')) + (f", {r.get('vol_x'):.1f}× volume" if pd.notna(r.get('vol_x')) else "")),
    ("🤝 Agreement", "data/consensus_log.csv", "first_seen", "signal_price", "return_pct",
     lambda r: " + ".join(k for k in ("simba", "scanner", "technicals") if str(r.get(k)) == "True") or f"{r.get('votes', '')} votes"),
    ("📅 V2.2 BUY alert", "data/watchlist_history.csv", "added_date", None, None,
     lambda r: f"RS score {_f(r.get('rs_score'), '{:.1f}')}"),
]


def signals_for(tickers, price: dict | None = None) -> pd.DataFrame:
    """Every logged signal for the given tickers, newest first: date, signal, detail, price then, return since."""
    tickers = {tickers} if isinstance(tickers, str) else set(tickers)
    rows = []
    try:
        from utils import corp_actions
        ca = corp_actions.load()
    except Exception:
        ca = None
    for label, path, dcol, pcol, rcol, detail in SOURCES:
        if not os.path.exists(path):
            continue
        try:
            L = pd.read_csv(path)
        except Exception:
            continue
        for _, r in L[L["ticker"].isin(tickers)].iterrows():
            then = pd.to_numeric(r.get(pcol), errors="coerce") if pcol else np.nan
            ret = pd.to_numeric(r.get(rcol), errors="coerce") if rcol else np.nan
            now = (price or {}).get(r["ticker"])
            if pd.notna(then) and then > 0 and now:       # to today's close (a log freezes its own figure when a row closes)
                fac = 1.0
                if ca is not None:
                    try:
                        fac = corp_actions.cumulative_factor(ca, r["ticker"], str(r.get(dcol))[:10]) or 1.0
                    except Exception:
                        fac = 1.0
                ret = (now / (then * fac) - 1) * 100
            rows.append(dict(date=pd.to_datetime(r.get(dcol), errors="coerce"), ticker=r["ticker"], signal=label,
                             detail=detail(r), price_then=then, return_since=ret))
    out = pd.DataFrame(rows, columns=["date", "ticker", "signal", "detail", "price_then", "return_since"])
    return out.dropna(subset=["date"]).sort_values(["date", "signal"], ascending=[False, True]).reset_index(drop=True)
