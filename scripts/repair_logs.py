"""
Log hygiene: repair rows the pipeline wrote wrongly. Safe to run every day (each fix is a no-op
once applied); the EOD workflow runs it before the engine.

    python scripts/repair_logs.py [--dry-run]

1. Mis-dated sessions. A run on a market holiday whose previous run failed logs the last real
   session's signals under the holiday's date. Known case: Monday 9 Nov 2026 filed under Tuesday
   10 Nov (Diwali Balipratipada). Rows are re-dated in the earnings-shock, turnaround-catalyst,
   breakout and agreement logs.
2. Mood and breadth history: weekend rows (21 from Dec 2025 - Mar 2026, written when the
   dashboard itself saved snapshots), the holiday row, and any row that repeats the previous
   row's every metric.
3. Sub-industry rotation: one date format (older rows carry a time), the stale copy of Friday
   recorded for 9 Nov dropped, and the 10 Nov rows (Monday's data) re-dated.

Returns are NOT touched here: split / bonus adjustment happens where each log refreshes its
prices (utils/corp_actions.py).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# holiday date under which a session was filed -> the session it really was
MISDATED = {"2026-11-10": "2026-11-09"}
DRY = "--dry-run" in sys.argv


def _save(df: pd.DataFrame, path: str) -> None:
    if not DRY:
        from utils.atomic_io import atomic_to_csv
        atomic_to_csv(df, path, index=False)


def redate(path: str, cols: list, key: list, shift_cols: tuple = (), drop_stale: bool = False) -> int:
    """
    Move rows dated on a holiday to the real session; `shift_cols` move back one business day.
    drop_stale: rows already filed under the real session came from the failed run re-scanning
    the session before it (9 Nov's run re-logged Friday's shocks) -- remove them first.
    """
    if not os.path.exists(path):
        return 0
    d = pd.read_csv(path)
    n = 0
    for bad, good in MISDATED.items():
        lead = cols[0]
        hit = d[lead].astype(str).str[:10] == bad
        if not hit.any():
            continue
        if drop_stale:
            stale = d[lead].astype(str).str[:10] == good
            n += int(stale.sum())
            d = d[~stale].reset_index(drop=True)
            hit = d[lead].astype(str).str[:10] == bad
        n += int(hit.sum())
        for c in shift_cols:                       # e.g. a breakout date counted back from the run date
            if c in d.columns:
                x = pd.to_datetime(d.loc[hit, c], errors="coerce") - pd.offsets.BDay(1)
                d.loc[hit, c] = x.dt.strftime("%Y-%m-%d")
        for c in cols:
            if c in d.columns:
                d.loc[d[c].astype(str).str[:10] == bad, c] = good
    if n:
        d = d.drop_duplicates(key, keep="first")
        _save(d, path)
    return n


def clean_history(path: str) -> dict:
    """Drop weekend rows, mis-dated holiday rows, and rows repeating the previous row exactly."""
    if not os.path.exists(path):
        return {}
    d = pd.read_csv(path).sort_values("date").reset_index(drop=True)
    dt = pd.to_datetime(d["date"], errors="coerce")
    weekend = dt.dt.weekday >= 5
    holiday = d["date"].isin(MISDATED) & d["date"].map(MISDATED).isin(set(d["date"]))
    vals = d.drop(columns=["date"])
    keep0 = d[~(weekend | holiday)].reset_index(drop=True)
    v = keep0.drop(columns=["date"])
    repeat = (v == v.shift(1)).all(axis=1)
    out = keep0[~repeat]
    stats = dict(weekend=int(weekend.sum()), holiday=int(holiday.sum()), repeat=int(repeat.sum()))
    if sum(stats.values()):
        _save(out, path)
    return stats


def fix_rotation(path: str = "data/sub_industry_rotation.csv") -> dict:
    if not os.path.exists(path):
        return {}
    d = pd.read_csv(path)
    raw = d["record_date"].astype(str)
    parsed = pd.to_datetime(raw, format="mixed", errors="coerce")
    norm = parsed.dt.strftime("%Y-%m-%d")
    wk = parsed.dt.weekday >= 5
    stats = dict(reformatted=int((raw != norm).sum()), weekend=int(wk.sum()), stale_dropped=0, redated=0)
    d["record_date"] = norm
    if wk.any():
        d = d[~wk].reset_index(drop=True)
    for bad, good in MISDATED.items():
        if not (d.record_date == bad).any():
            continue
        # rows recorded for the real session were a copy of the session before it (failed fetch)
        prior = d[d.record_date < good].record_date.max()
        g, p = d[d.record_date == good], d[d.record_date == prior]
        cols = [c for c in d.columns if c not in ("record_date",)]
        if len(g) and len(p) and g[cols].reset_index(drop=True).equals(p[cols].reset_index(drop=True)):
            stats["stale_dropped"] += len(g)
            d = d[d.record_date != good]
        if not (d.record_date == good).any():
            stats["redated"] += int((d.record_date == bad).sum())
            d.loc[d.record_date == bad, "record_date"] = good
        else:
            d = d[d.record_date != bad]
    if sum(stats.values()):
        sort_cols = ["record_date", "sub_industry"] if "sub_industry" in d.columns else ["record_date"]
        d = d.sort_values(sort_cols).reset_index(drop=True)
        _save(d, path)
    return stats


def _val_diff(cur, target, tol: float = 1e-2) -> bool:
    c_num = pd.to_numeric(pd.Series([cur]), errors="coerce").iloc[0]
    t_num = pd.to_numeric(pd.Series([target]), errors="coerce").iloc[0]
    if pd.isna(c_num) and pd.isna(t_num):
        return False
    if pd.isna(c_num) != pd.isna(t_num):
        return True
    return abs(float(c_num) - float(t_num)) > tol


def recompute_split_returns() -> dict:
    """Recompute split/bonus-affected returns in signal logs from NSE adjusted history (idempotent)."""
    from utils import corp_actions, nse_prices
    ca = corp_actions.load()
    if ca is None or ca.empty:
        return {}
    stats = {}
    targets = [
        ("data/tc_log.csv", "closed21"),
        ("data/earnings_shock_log.csv", "closed21"),
        ("data/ias_signal_log.csv", "ias"),
    ]
    for path, mode in targets:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if df.empty or "ticker" not in df.columns or "signal_date" not in df.columns:
            continue
        facs = corp_actions.factor_map(ca, df["ticker"], df["signal_date"])
        idxs = df.index[facs != 1.0].tolist()
        if not idxs:
            continue
        hit_tickers = sorted({str(df.at[i, "ticker"]) for i in idxs})
        hmap = nse_prices.history(hit_tickers)
        changed = 0
        for i in idxs:
            t = str(df.at[i, "ticker"])
            sd = pd.Timestamp(str(df.at[i, "signal_date"])[:10])
            h = hmap.get(t)
            if h is None or h.empty or h.loc[:sd].empty:
                continue
            sig_raw = pd.to_numeric(df.at[i, "signal_price"], errors="coerce")
            if pd.isna(sig_raw) or float(sig_raw) <= 0:
                continue
            sig_adj = float(h.loc[:sd, "Close"].iloc[-1])
            if sig_adj <= 0:
                continue
            row_changed = False
            if mode == "closed21":
                is_closed = str(df.at[i, "status"]).upper() == "CLOSED"
                if is_closed:
                    close_end = sd + pd.Timedelta(days=22)
                    sub = h.loc[:close_end, "Close"]
                    if sub.empty:
                        continue
                    cur_adj = float(sub.iloc[-1])
                    fac_closed = corp_actions.cumulative_factor(ca, t, sd, close_end)
                    new_cur = round(float(sig_raw) * fac_closed * (cur_adj / sig_adj), 2)
                    new_ret = round((cur_adj / sig_adj - 1.0) * 100.0, 2)
                else:
                    cur_adj = float(h["Close"].iloc[-1])
                    new_cur = round(cur_adj, 2)
                    new_ret = round((cur_adj / sig_adj - 1.0) * 100.0, 2)
                if _val_diff(df.at[i, "current_price"], new_cur):
                    df.at[i, "current_price"] = new_cur
                    row_changed = True
                if _val_diff(df.at[i, "return_since_signal"], new_ret):
                    df.at[i, "return_since_signal"] = new_ret
                    row_changed = True
                if is_closed and "return_21d" in df.columns and _val_diff(df.at[i, "return_21d"], new_ret):
                    df.at[i, "return_21d"] = new_ret
                    row_changed = True
            else:
                cur_adj = float(h["Close"].iloc[-1])
                new_cur = round(cur_adj, 1)
                new_ret = round((cur_adj / sig_adj - 1.0) * 100.0, 1)
                after_eq = h.loc[h.index >= sd, "Close"]
                new_max = round((float(after_eq.max()) / sig_adj - 1.0) * 100.0, 1) if len(after_eq) else new_ret
                new_max_dt = after_eq.idxmax().strftime("%Y-%m-%d") if len(after_eq) else str(df.at[i, "max_gain_date"])[:10]
                if _val_diff(df.at[i, "current_price"], new_cur):
                    df.at[i, "current_price"] = new_cur
                    row_changed = True
                if _val_diff(df.at[i, "return_since_signal"], new_ret):
                    df.at[i, "return_since_signal"] = new_ret
                    row_changed = True
                if _val_diff(df.at[i, "max_gain"], new_max):
                    df.at[i, "max_gain"] = new_max
                    row_changed = True
                if str(df.at[i, "max_gain_date"])[:10] != new_max_dt:
                    df.at[i, "max_gain_date"] = new_max_dt
                    row_changed = True
                after_gt = h.loc[h.index > sd, "Close"]
                for n, col_p, col_r in ((5, "signal_price_at_5d", "return_5d"),
                                        (21, "signal_price_at_21d", "return_21d"),
                                        (63, "signal_price_at_63d", "return_63d")):
                    if col_p in df.columns and col_r in df.columns and len(after_gt) >= n:
                        px = float(after_gt.iloc[n - 1])
                        t_px = round(px, 1)
                        t_ret = round((px / sig_adj - 1.0) * 100.0, 1)
                        if _val_diff(df.at[i, col_p], t_px):
                            df.at[i, col_p] = t_px
                            row_changed = True
                        if _val_diff(df.at[i, col_r], t_ret):
                            df.at[i, col_r] = t_ret
                            row_changed = True
                days = (h.index[-1] - sd).days
                new_xirr = round(((cur_adj / sig_adj) ** (365.0 / days) - 1.0) * 100.0, 1) if days >= 30 and cur_adj > 0 else np.nan
                if "xirr" in df.columns and _val_diff(df.at[i, "xirr"], new_xirr):
                    df.at[i, "xirr"] = new_xirr
                    row_changed = True
            if row_changed:
                changed += 1
        if changed:
            _save(df, path)
            stats[os.path.basename(path)] = changed
    return stats


def main() -> int:
    print(f"[HYGIENE] {'dry run -- ' if DRY else ''}repairing logs")
    jobs = [
        ("data/earnings_shock_log.csv", ["signal_date"], ["ticker", "signal_date"], (), True),
        ("data/tc_log.csv", ["signal_date"], ["ticker", "signal_date"], (), True),
        ("data/breakout_log.csv", ["first_seen"], ["ticker", "kind", "breakout_date"], ("breakout_date",), False),
        ("data/consensus_log.csv", ["first_seen", "last_agree"], ["ticker", "first_seen"], (), False),
        ("data/ias_signal_log.csv", ["signal_date"], ["ticker", "signal_date"], (), False),
    ]
    for path, cols, key, shift, stale in jobs:
        try:
            n = redate(path, cols, key, shift, stale)
            print(f"   {os.path.basename(path)}: {n} row(s) re-dated or removed" if n else f"   {os.path.basename(path)}: ok")
        except Exception as e:
            print(f"   {os.path.basename(path)}: FAILED {e}")
    for path in ("data/market_breadth_history.csv", "data/market_mood_history.csv"):
        try:
            s = clean_history(path)
            print(f"   {os.path.basename(path)}: {s if sum(s.values()) else 'ok'}")
        except Exception as e:
            print(f"   {os.path.basename(path)}: FAILED {e}")
    try:
        s = fix_rotation()
        print(f"   sub_industry_rotation.csv: {s if sum(s.values()) else 'ok'}")
    except Exception as e:
        print(f"   sub_industry_rotation.csv: FAILED {e}")
    try:
        sp = recompute_split_returns()
        print(f"   split return repair: {sp if sp else 'ok'}")
    except Exception as e:
        print(f"   split return repair: FAILED {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
