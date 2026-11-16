"""
Trade journal: every buy and sell, with what the dashboard saw at that moment.

Events (one per change in a holding) are stored in the same GitHub-backed store as the book
(trade_journal.json on the alert-state branch):

    BUY     a new holding            ADD     more of an existing one
    REDUCE  part sold                SELL    all sold
    VOID    the holding was deleted from the book (a mistake, not a trade): the open trade is dropped
    SPLIT   a split or bonus changed the share count (qty holds the ratio); the cost is unchanged

Each event keeps the quantity, the price, where it came from (Zerodha sync, Dhan sync, a file,
or typed in), an `estimated` flag when the price is the day's close rather than the real fill
(brokers' holdings do not give sell prices), your note and tags, and a snapshot of metrics:
funnel stages passed, industry group and score, Scanner and Technicals reads, RS, distance from
the 52-week high, chart stage, business score, valuation percentile, stop distance and the
market regime. On a sale it also keeps the holding's status and whether the exit rule had fired.

trades() folds the events into round trips (first BUY to the SELL that takes the quantity to
zero), with the entry and exit snapshots side by side, so closed trades can be grouped by what
was true when you bought. patterns() does that grouping. Fewer than ~20 closed trades per group
is anecdote, not evidence.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from utils.alert_store import AlertStore

STORE = AlertStore("trade_journal.json", "Trade journal")
BUYS, SELLS = ("BUY", "ADD"), ("REDUCE", "SELL")
KEEP = ["industry", "ind_group", "ind_score", "s_industry", "s_trend", "s_business", "s_setup", "scanner", "technicals", "comp_rs",
        "rs_3m", "dist_52w", "stage", "biz_score", "biz_verdict", "val_pct", "risk_pct", "stop"]


def _clean(v):
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if v is None or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return round(float(v), 4)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def capture(tickers: Iterable[str], df: pd.DataFrame, board: Optional[pd.DataFrame] = None, val: Optional[pd.DataFrame] = None,
            regime: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """The metrics snapshot for each ticker, from today's data. Tickers not in `df` get only the regime."""
    from utils import funnel
    tickers = list(dict.fromkeys(tickers))
    reg = {k: (regime or {}).get(k) for k in ("regime", "breadth_regime", "breadth")}
    reg = {"regime_nifty": reg["regime"], "regime_breadth": reg["breadth_regime"], "breadth_pct": _clean(reg["breadth"])}
    out = {t: dict(reg) for t in tickers}
    sub = df[df["ticker"].isin(tickers)]
    if sub.empty:
        return out
    T = funnel.table(sub, board, val, business_for=tickers).set_index("ticker")
    x = sub.drop_duplicates("ticker").set_index("ticker")
    for t in T.index:
        r = T.loc[t]
        m = {k: _clean(r.get(k)) for k in KEEP if k in T.columns}
        m["stages_passed"] = int(sum(bool(m.get(k)) for k in ("s_industry", "s_trend", "s_business", "s_setup")))
        px = pd.to_numeric(x.at[t, "currentPrice"], errors="coerce")
        for lab, col in (("above_50dma", "fiftyDayAverage"), ("above_200dma", "twoHundredDayAverage")):
            ma = pd.to_numeric(x.at[t, col], errors="coerce") if col in x.columns else np.nan
            m[lab] = bool(px > ma) if pd.notna(px) and pd.notna(ma) else None
        m["trend_score"] = _clean(pd.to_numeric(x.at[t, "trend_score"], errors="coerce")) if "trend_score" in x.columns else None
        m["data_date"] = str(x.at[t, "bar_date"])[:10] if "bar_date" in x.columns and pd.notna(x.at[t, "bar_date"]) else None
        out[t] = {**m, **reg}
    return out


def event(action: str, ticker: str, qty: float, price: Optional[float], source: str = "manual", estimated: bool = False,
          notes: str = "", metrics: Optional[Dict[str, Any]] = None, date: Optional[str] = None, **extra) -> Dict[str, Any]:
    now = datetime.now()
    return {"id": str(uuid.uuid4()), "ts": now.isoformat(timespec="seconds"), "date": date or now.strftime("%Y-%m-%d"),
            "ticker": ticker, "action": action, "qty": float(qty or 0), "price": _clean(price), "source": source,
            "estimated": bool(estimated), "notes": notes or "", "tags": "", "metrics": metrics or {}, **extra}


def events() -> List[Dict[str, Any]]:
    return sorted(STORE.load(), key=lambda e: (str(e.get("date")), str(e.get("ts"))))


def record(evs: List[Dict[str, Any]], background: bool = True) -> int:
    if evs:
        STORE.commit({"adds": evs}, f"journal: {len(evs)} event(s)", background=background)
    return len(evs)


def patch(event_id: str, **fields) -> None:
    STORE.commit({"patches": {event_id: fields}}, "journal: edit", background=True)


def catch_up(positions: List[Dict[str, Any]], evs: Optional[List[Dict[str, Any]]] = None, resolve=None) -> List[Dict[str, Any]]:
    """
    Events that bring the journal in line with the book, for changes made where the journal was not
    listening (before it existed, or on the classic tabs):
      - a holding with no open trade        -> BUY  "before the journal" (no metrics: none were captured)
      - an open trade the book has closed   -> SELL at the book's exit price
      - an open trade no longer in the book -> VOID
    `resolve` maps a book ticker to the symbol the journal uses (fixes old spellings).
    """
    if not positions:                                        # an empty book is more likely a failed read than a cleared one
        return []
    evs = events() if evs is None else evs
    T = trades(evs)
    open_t = set(T.loc[T.status == "open", "ticker"]) if len(T) else set()
    rs = resolve or (lambda t: t)
    closed_n = T[T.status == "closed"].groupby("ticker").groups if len(T) else {}     # a re-bought stock gets a new id
    first = dict(zip(T.loc[T.status == "open", "ticker"], T.loc[T.status == "open", "trade_id"])) if len(T) else {}
    out, active, closed = [], set(), {}
    for p in positions:
        t = rs(p.get("ticker"))
        if p.get("status") == "active":
            active.add(t)
            if t not in open_t and p.get("entry_price"):
                # fixed ids: if the journal could not be read (GitHub down) and looked empty, the same catch-up
                # written twice is stored once, instead of doubling every holding
                out.append(event("BUY", t, p.get("quantity") or 1, p.get("entry_price"), source="before the journal",
                                 notes=p.get("notes") or "", date=str(p.get("entry_date") or "")[:10] or None,
                                 id=f"base:{t}:{str(p.get('entry_date') or '')[:10]}:{len(closed_n.get(t, []))}"))
                open_t.add(t)
        elif p.get("status") == "closed":
            closed[t] = p
    for t in sorted(open_t - active):
        p = closed.get(t)
        if p is not None and p.get("exit_price"):
            out.append(event("SELL", t, 0, p["exit_price"], source="book", date=str(p.get("exit_date") or "")[:10] or None,
                             id=f"close:{first.get(t, t)}"))
        else:
            out.append(event("VOID", t, 0, None, source="book", id=f"void:{first.get(t, t)}"))
    return out


def open_entry(ticker: str, evs: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """The open trade's row for `ticker` (None if there is none)."""
    T = trades(events() if evs is None else evs)
    r = T[(T.status == "open") & (T.ticker == ticker)] if len(T) else T
    return r.iloc[0].to_dict() if len(r) else None


def trades(evs: Optional[List[Dict[str, Any]]] = None, prices: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Round trips, one row each, with entry_* and exit_* metric columns. Open trades have status 'open'."""
    evs = events() if evs is None else sorted(evs, key=lambda e: (str(e.get("date")), str(e.get("ts"))))
    rows, cur = [], {}

    def close(t, c, status):
        sold_q = c["sold_qty"]
        avg_in = c["cost"] / c["bought"] if c["bought"] else np.nan
        avg_out = c["proceeds"] / sold_q if sold_q else np.nan
        px = (prices or {}).get(t)
        ret = (avg_out / avg_in - 1) * 100 if status == "closed" and avg_in else (px / avg_in - 1) * 100 if px and avg_in else np.nan
        end = pd.Timestamp(c["exit_date"]) if status == "closed" else pd.Timestamp.today().normalize()
        row = dict(trade_id=c["first"]["id"], ticker=t, status=status, entry_date=c["first"]["date"],
                   exit_date=c["exit_date"] if status == "closed" else None,
                   days=int(max((end - pd.Timestamp(c["first"]["date"])).days, 0)) if c["first"].get("date") else None,
                   qty=c["bought"], open_qty=c["qty"], entry_price=avg_in, exit_price=avg_out if sold_q else np.nan,
                   return_pct=ret, pnl=(avg_out - avg_in) * sold_q if sold_q and avg_in else ((px - avg_in) * c["qty"] if px and avg_in else np.nan),
                   adds=c["adds"], partial_exits=c["reduces"], source=c["first"].get("source", ""),
                   entry_estimated=bool(c["first"].get("estimated")), exit_estimated=bool(c["last_sell"].get("estimated")) if c["last_sell"] else False,
                   entry_note=c["first"].get("notes", ""), exit_note=(c["last_sell"] or {}).get("notes", ""),
                   tags=c["first"].get("tags", ""), entry_event=c["first"]["id"], exit_event=(c["last_sell"] or {}).get("id"))
        for k, v in (c["first"].get("metrics") or {}).items():
            row[f"entry_{k}"] = v
        for k, v in ((c["last_sell"] or {}).get("metrics") or {}).items():
            row[f"exit_{k}"] = v
        rows.append(row)

    for e in evs:
        t, a, q, p = e.get("ticker"), e.get("action"), float(e.get("qty") or 0), e.get("price")
        if a in BUYS:
            c = cur.get(t)
            if c is None:
                c = cur[t] = dict(first=e, qty=0.0, bought=0.0, cost=0.0, sold_qty=0.0, proceeds=0.0, adds=0, reduces=0, last_sell=None,
                                  exit_date=None)
            else:
                c["adds"] += 1
            c["qty"] += q
            c["bought"] += q
            c["cost"] += q * float(p or 0)
        elif a == "VOID":
            cur.pop(t, None)
        elif a == "SPLIT" and t in cur and q > 0:            # qty holds the ratio (2 = one share became two); cost is unchanged
            for k in ("qty", "bought", "sold_qty"):
                cur[t][k] *= q
        elif a in SELLS and t in cur:
            c = cur[t]
            q = min(q, c["qty"]) if a == "REDUCE" else c["qty"] if a == "SELL" else q
            c["qty"] -= q
            c["sold_qty"] += q
            c["proceeds"] += q * float(p or 0)
            c["last_sell"], c["exit_date"] = e, e.get("date")
            if a == "REDUCE":
                c["reduces"] += 1
            if c["qty"] <= 1e-9:
                close(t, c, "closed")
                del cur[t]
    for t, c in cur.items():
        close(t, c, "open")
    T = pd.DataFrame(rows)
    return T.sort_values(["status", "exit_date", "entry_date"], ascending=[True, False, False]).reset_index(drop=True) if len(T) else T


def stats(T: pd.DataFrame) -> Dict[str, Any]:
    c = T[T.status == "closed"] if len(T) else T
    if c is None or c.empty:
        return dict(closed=0, open=int((T.status == "open").sum()) if len(T) else 0)
    r = pd.to_numeric(c.return_pct, errors="coerce").dropna()
    w, l = r[r > 0], r[r <= 0]
    return dict(closed=len(c), open=int((T.status == "open").sum()), win_rate=float((r > 0).mean() * 100) if len(r) else np.nan,
                avg_win=float(w.mean()) if len(w) else np.nan, avg_loss=float(l.mean()) if len(l) else np.nan,
                expectancy=float(r.mean()) if len(r) else np.nan, median=float(r.median()) if len(r) else np.nan,
                avg_days=float(pd.to_numeric(c.days, errors="coerce").mean()), pnl=float(pd.to_numeric(c.pnl, errors="coerce").sum()),
                payoff=float(w.mean() / abs(l.mean())) if len(w) and len(l) and l.mean() else np.nan)


# what a closed trade can be grouped by: label -> (column, kind)
DIMENSIONS = {
    "Funnel stages passed at entry": ("entry_stages_passed", "cat"),
    "Industry group at entry": ("entry_ind_group", "cat"),
    "Technicals read at entry": ("entry_technicals", "cat"),
    "Scanner signal at entry": ("entry_scanner", "cat"),
    "Business verdict at entry": ("entry_biz_verdict", "cat"),
    "Chart stage at entry": ("entry_stage", "cat"),
    "Market regime at entry": ("entry_regime_nifty", "cat"),
    "RS vs Nifty at entry": ("entry_comp_rs", [-1e9, 0, 5, 15, 1e9], ["below 0", "0 to 5", "5 to 15", "above 15"]),
    "Distance from 52-week high at entry": ("entry_dist_52w", [-1e9, -25, -10, -3, 1e9], ["over 25% below", "10-25% below", "3-10% below", "within 3%"]),
    "Stop distance at entry": ("entry_risk_pct", [-1e9, 6, 10, 15, 1e9], ["under 6%", "6-10%", "10-15%", "over 15%"]),
    "Holding period": ("days", [-1, 7, 30, 90, 1e9], ["a week or less", "8-30 days", "31-90 days", "over 90 days"]),
    "Status when sold": ("exit_status", "cat"),
    "Exit rule had fired when sold": ("exit_exit_rule_hit", "cat"),
    "Industry (name)": ("entry_industry", "cat"),
    "Your tag": ("tags", "cat"),
}


def patterns(T: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Closed trades grouped by one dimension: count, win rate, average and median return, average days held."""
    c = T[T.status == "closed"].copy() if len(T) else pd.DataFrame()
    spec = DIMENSIONS[dimension]
    if c.empty or spec[0] not in c.columns:
        return pd.DataFrame()
    col = c[spec[0]]
    if spec[1] == "cat":
        g = col.map(lambda v: "not recorded" if v is None or (isinstance(v, float) and v != v) or v == "" else str(v).replace(".0", ""))
    else:
        g = pd.cut(pd.to_numeric(col, errors="coerce"), bins=spec[1], labels=spec[2]).astype(object).where(col.notna(), "not recorded")
    c["group"], c["r"] = g, pd.to_numeric(c.return_pct, errors="coerce")
    out = c.groupby("group", observed=True).agg(trades=("r", "size"), win_rate=("r", lambda s: (s > 0).mean() * 100),
                                                 avg_return=("r", "mean"), median_return=("r", "median"),
                                                 avg_days=("days", "mean"), pnl=("pnl", "sum")).reset_index()
    return out.sort_values("trades", ascending=False).reset_index(drop=True)
