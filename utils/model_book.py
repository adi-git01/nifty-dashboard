"""
Real-money mirror of the model portfolio (OptComp V2.2): what you hold of it, and the orders that
bring it in line with the model.

The engine keeps running on paper exactly as before (data/dna3_portfolio_snapshot.json). This module
compares the model's holdings with your own "model book" and writes an ORDER TICKET:

    SELL  everything you hold that the model no longer holds
    BUY   everything the model holds that you do not, one equal slot each
          (slot = your model equity / the model's maximum positions), whole shares only

Holdings you both have are left alone: the model does not resize a position between entry and exit,
so neither does the ticket. Orders are limit orders a little through the last close (1% by default)
so they fill at the open without paying any price.

State lives in the same GitHub-backed store as the book:
    model_book.json   one row per holding (ticker, qty, avg) plus a "_meta" row (capital, cash, paused)
    order_log.json    one row per order: planned -> sent -> filled / rejected / cancelled

Nothing here talks to a broker. Sending orders is utils/dhan_orders.py; this module only decides
what the orders are and records what happened to them.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.alert_store import AlertStore

BOOK = AlertStore("model_book.json", "Model book")
LOG = AlertStore("order_log.json", "Order log")
SNAP = "data/dna3_portfolio_snapshot.json"
TRADES = "data/dna3_trade_log.csv"
META = "_meta"
DEFAULTS = dict(capital=200000.0, cash=200000.0, paused=False, buffer=0.01, max_order=0.0)
STALE_DAYS = 5              # a ticket built on a snapshot older than this is not offered
OVER_SLOT = 1.25            # one share may cost up to this multiple of a slot; dearer names are skipped
DONE = ("FILLED", "REJECTED", "CANCELLED", "SKIPPED")


def tick(price: float) -> float:
    """Round to ₹0.05: valid for every NSE tick size."""
    return round(round(float(price) / 0.05) * 0.05, 2)


# ------------------------------------------------------------------ state
def load() -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """-> (meta, holdings)."""
    rows = BOOK.load()
    meta = {**DEFAULTS, **next((r for r in rows if r.get("id") == META), {})}
    return meta, [r for r in rows if r.get("id") != META and float(r.get("qty") or 0) > 0]


def save_meta(**fields) -> None:
    rows = BOOK.load()
    if any(r.get("id") == META for r in rows):
        BOOK.commit({"patches": {META: fields}}, "model book: settings", background=True)
    else:
        BOOK.commit({"adds": [{"id": META, **DEFAULTS, "started": datetime.now().strftime("%Y-%m-%d"), **fields}]},
                    "model book: start", background=True)


def snapshot(root: str = ".") -> Dict[str, Any]:
    try:
        return json.load(open(os.path.join(root, SNAP)))
    except Exception:
        return {}


def exit_reasons(root: str = ".") -> Dict[str, str]:
    """Why the model sold each stock, latest sale first (for the ticket's Reason column)."""
    try:
        t = pd.read_csv(os.path.join(root, TRADES))
        t = t[t.Action == "SELL"].drop_duplicates("Ticker", keep="last")
        return dict(zip(t.Ticker, t.Reason.astype(str)))
    except Exception:
        return {}


# ------------------------------------------------------------------ the ticket
def ticket(snap: Dict[str, Any], meta: Dict[str, Any], book: List[Dict[str, Any]], prices: Dict[str, float],
           reasons: Optional[Dict[str, str]] = None, today: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
    """
    Orders that bring `book` in line with the model. Returns
      orders   list of dicts (id, date, ticker, side, qty, limit, value, ref, reason, note, stop)
      skipped  buys that could not be sized, with the reason
      summary  equity, cash, slot, invested, cash_after
      problem  a sentence when no ticket should be offered (stale snapshot, paused, no model)
    """
    out: Dict[str, Any] = dict(orders=[], skipped=[], summary={}, problem=None)
    eng = snap.get("holdings") or {}
    info = {p.get("Ticker"): p for p in (snap.get("portfolio") or [])}
    date = str(snap.get("date") or "")[:10]
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    if not snap or not date:
        out["problem"] = "The model portfolio snapshot is missing."
        return out
    age = (today - pd.Timestamp(date)).days
    n = int((snap.get("config") or {}).get("max_positions") or 15)
    buf = float(meta.get("buffer") or 0.01)
    px = lambda t: prices.get(t) or (info.get(t) or {}).get("Price")
    held = {b["ticker"]: b for b in book}
    cash = float(meta.get("cash") or 0)
    invested = sum(float(b["qty"]) * float(px(b["ticker"]) or b.get("avg") or 0) for b in book)
    equity = cash + invested
    slot = equity / n if n else 0
    out["summary"] = dict(equity=equity, cash=cash, invested=invested, slot=slot, positions=len(book), model_positions=len(eng),
                          model_date=date, regime=snap.get("regime"), age_days=age)
    if meta.get("paused"):
        out["problem"] = "Paused: no orders are proposed until you switch the model book back on."
        return out
    if age > STALE_DAYS:
        out["problem"] = f"The model was last updated on {date} ({age} days ago). Orders are not proposed from a stale model."
        return out

    oid = lambda t, side: f"{date}-{t.replace('.NS', '')}-{side}"
    avail = cash
    for t, b in held.items():                                # exits first: they free the cash
        if t in eng:
            continue
        p = px(t)
        if not p:
            out["skipped"].append(dict(ticker=t, side="SELL", why="no price: sell it by hand"))
            continue
        lim = tick(float(p) * (1 - buf))
        q = int(b["qty"])
        out["orders"].append(dict(id=oid(t, "SELL"), date=date, ticker=t, side="SELL", qty=q, limit=lim, value=round(q * lim, 2),
                                  ref=float(p), reason=(reasons or {}).get(t) or "Model exited", note="", stop=None))
        avail += q * lim
    cap = float(meta.get("max_order") or 0) or slot * OVER_SLOT
    new = sorted((t for t in eng if t not in held), key=lambda t: -float((info.get(t) or {}).get("RS_Score") or 0))
    for t in new:
        p = px(t)
        if not p:
            out["skipped"].append(dict(ticker=t, side="BUY", why="no price"))
            continue
        lim = tick(float(p) * (1 + buf))
        q = int(math.floor(slot / lim))
        if q == 0 and lim <= cap:
            q = 1                                            # one share, a little over a slot
        if q == 0:
            out["skipped"].append(dict(ticker=t, side="BUY", why=f"one share (₹{lim:,.0f}) is more than {OVER_SLOT:.2f}x a slot (₹{slot:,.0f})"))
            continue
        q = min(q, int(math.floor(cap / lim)))
        if q <= 0:
            out["skipped"].append(dict(ticker=t, side="BUY", why=f"one share (₹{lim:,.0f}) is over the per-order cap (₹{cap:,.0f})"))
            continue
        note = ""
        if q * lim > avail:
            q = int(math.floor(avail / lim))
            note = "reduced: not enough cash"
        if q <= 0:
            out["skipped"].append(dict(ticker=t, side="BUY", why="no cash left"))
            continue
        if q * lim > cash:
            note = (note + "; " if note else "") + "needs the sale proceeds first"
        e = float((eng.get(t) or {}).get("entry_price") or 0)
        gap = (float(p) / e - 1) * 100 if e else 0
        if abs(gap) >= 5:
            note = (note + "; " if note else "") + f"{gap:+.0f}% from the model's entry"
        out["orders"].append(dict(id=oid(t, "BUY"), date=date, ticker=t, side="BUY", qty=q, limit=lim, value=round(q * lim, 2), ref=float(p),
                                  reason=f"Model holding (RS {float((info.get(t) or {}).get('RS_Score') or 0):.0f})", note=note,
                                  stop=(info.get(t) or {}).get("Trail_Stop")))
        avail -= q * lim
        cash -= min(cash, q * lim)
    out["summary"]["cash_after"] = avail
    return out


def pending(orders: List[Dict[str, Any]], log: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Ticket orders not yet dealt with (an order id is used once: it cannot be sent twice)."""
    seen = {r["id"]: r for r in (LOG.load() if log is None else log)}
    return [o for o in orders if seen.get(o["id"], {}).get("status") not in ("SENDING", "SENT", "FILLED", "SKIPPED")]   # refused / unfilled: offer again


# ------------------------------------------------------------------ recording
def log_order(order: Dict[str, Any], status: str, **fields) -> Dict[str, Any]:
    """Create or update the log row for an order."""
    row = {**order, "status": status, "ts": datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat(timespec="seconds"), **fields}
    if any(r.get("id") == order["id"] for r in LOG.load()):
        LOG.commit({"patches": {order["id"]: {k: v for k, v in row.items() if k != "id"}}}, f"order {order['id']}: {status}", background=True)
    else:
        LOG.commit({"adds": [row]}, f"order {order['id']}: {status}", background=True)
    return row


def apply_fill(order: Dict[str, Any], qty: float, price: float, meta: Optional[Dict[str, Any]] = None,
               book: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Record a fill in the model book: holdings and cash. Returns {'cash', 'qty', 'avg'} after the fill.
    (The personal book and the journal are updated by the caller: utils/model_orders_view.py.)
    """
    if meta is None or book is None:
        meta, book = load()
    t, qty, price = order["ticker"], float(qty), float(price)
    cur = next((b for b in book if b["ticker"] == t), None)
    cash = float(meta.get("cash") or 0)
    if order["side"] == "BUY":
        q0, a0 = (float(cur["qty"]), float(cur["avg"])) if cur else (0.0, 0.0)
        q1 = q0 + qty
        a1 = (q0 * a0 + qty * price) / q1 if q1 else 0.0
        cash -= qty * price
        row = dict(ticker=t, qty=q1, avg=round(a1, 2), model_entry=order.get("ref"))
        if cur:
            BOOK.commit({"patches": {t: row}}, f"model book: bought {t}", background=True)
        else:
            BOOK.commit({"adds": [{"id": t, **row, "entry_date": datetime.now().strftime("%Y-%m-%d")}]}, f"model book: bought {t}",
                        background=True)
    else:
        q0 = float(cur["qty"]) if cur else 0.0
        qty = min(qty, q0)
        q1, a1 = q0 - qty, float(cur["avg"]) if cur else 0.0
        cash += qty * price
        if cur:
            BOOK.commit({"removes": [t]} if q1 <= 0 else {"patches": {t: dict(qty=q1)}}, f"model book: sold {t}", background=True)
    save_meta(cash=round(cash, 2))
    return dict(cash=cash, qty=q1, avg=a1, filled=qty)


def telegram_line(prices: Dict[str, float], root: str = ".") -> str:
    """One paragraph for the daily Telegram brief when the ticket has orders ('' when there is nothing to do or no model book)."""
    meta, book = load()
    if not meta.get("started"):
        return ""
    T = ticket(snapshot(root), meta, book, prices, exit_reasons(root))
    orders = pending(T["orders"])
    if T["problem"] or not orders:
        return ""
    s = lambda side: ", ".join(f"{o['ticker'].replace('.NS', '')} {o['qty']}@{o['limit']:g}" for o in orders if o["side"] == side)
    L = [f"<b>🎫 Model orders ({len(orders)})</b>"]
    if s("SELL"):
        L.append("Sell: " + s("SELL"))
    if s("BUY"):
        L.append("Buy: " + s("BUY"))
    L.append("Confirm on Portfolio Manager → Return Tracker → Model Orders.")
    return "\n".join(L)
