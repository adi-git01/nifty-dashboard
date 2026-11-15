"""
Import holdings from a broker file (Zerodha, Dhan or any broker) into My Book.

  parse(data, filename)                    -> rows: symbol, qty, avg  (one per stock)
  combine([rows, ...])                     -> one table across several files (qty summed, average cost weighted)
  reconcile(rows, positions, known, ...)   -> a plan: add / update / close / unchanged / unknown
  apply(plan, ...)                         -> writes the plan through utils/positions.py

The file is recognised by its column names, not by broker: a symbol column (Instrument, Symbol,
Trading Symbol, Scrip ...), a quantity column (Qty., Quantity, Quantity Available ...) and an
average-cost column (Avg. cost, Average Price, Buy Avg ...). Title rows above the header (as in
Zerodha Console's Excel export) are skipped. With no symbol column, a company-name column is
matched against the universe.

reconcile never deletes. A holding missing from the file is only PROPOSED for closing, and only
when `whole` is true (the files are the entire portfolio). Stops, targets and notes are kept.
Brokers do not give the purchase date, so a new holding's entry date is the import date.
"""
from __future__ import annotations

import difflib
import io
import re
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

SYMBOL = ("instrument", "symbol", "tradingsymbol", "trading symbol", "scrip", "scrip name", "stock symbol", "nse symbol", "ticker")
NAME = ("name", "stock name", "company", "company name", "security", "security name", "stock")
QTY = ("qty", "quantity", "quantity available", "total qty", "total quantity", "holding qty", "holdings qty", "net qty",
       "available qty", "shares", "units")
QTY_EXTRA = ("quantity pledged margin", "quantity pledged loan", "pledged qty", "collateral qty")     # added to the main quantity
AVG = ("avg cost", "average price", "avg price", "average cost", "buy avg", "buy average", "avg buy price", "buy price",
       "average buy price", "avg trading price", "cost price", "avg")


def _norm(c: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(c).lower())).strip()


def _num(v: Any) -> float:
    try:
        return float(str(v).replace(",", "").replace("₹", "").strip())
    except (TypeError, ValueError):
        return np.nan


def clean_symbol(s: Any) -> str:
    """'NSE:TCS-EQ' -> 'TCS'."""
    t = str(s or "").strip().upper()
    t = t.split(":")[-1]
    t = re.sub(r"\.(NS|BO)$", "", t)
    t = re.sub(r"-(EQ|BE|BZ|BL|SM|ST|GS|N\d|E\d|T\d?|X\d?|Z)$", "", t)
    return t.replace(" ", "")


def _table(data: bytes, filename: str) -> pd.DataFrame:
    """The sheet as raw rows (no header assumed)."""
    if filename.lower().endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(io.BytesIO(data), header=None, dtype=str)
    text = data.decode("utf-8-sig", errors="replace")
    sep = "\t" if text.count("\t") > text.count(",") else ","
    rows = [r for r in pd.read_csv(io.StringIO(text), header=None, dtype=str, sep=sep, engine="python", on_bad_lines="skip",
                                   skip_blank_lines=True).values.tolist()]
    return pd.DataFrame(rows)


def parse(data: bytes, filename: str = "holdings.csv") -> pd.DataFrame:
    """Rows of symbol (cleaned), name, qty, avg from a broker holdings file. Raises ValueError with a plain reason."""
    raw = _table(data, filename)
    head = None
    for i in range(min(len(raw), 40)):                       # the header is the first row with a quantity and a symbol/name column
        cells = [_norm(c) for c in raw.iloc[i].tolist()]
        if any(c in QTY for c in cells) and any(c in SYMBOL + NAME for c in cells):
            head = i
            break
    if head is None:
        raise ValueError("Could not find a header row with a symbol (or name) column and a quantity column.")
    cols = [_norm(c) for c in raw.iloc[head].tolist()]
    body = raw.iloc[head + 1:].reset_index(drop=True)
    body.columns = cols
    pick = lambda names: next((c for c in names if c in cols), None)
    sym, nam, qty, avg = pick(SYMBOL), pick(NAME), pick(QTY), pick(AVG)
    out = pd.DataFrame({"symbol": body[sym].map(clean_symbol) if sym else "", "name": body[nam].astype(str).str.strip() if nam else "",
                        "qty": body[qty].map(_num), "avg": body[avg].map(_num) if avg else np.nan})
    for c in QTY_EXTRA:
        if c in cols:
            out["qty"] = out["qty"].fillna(0) + body[c].map(_num).fillna(0)
    out = out[(out.qty > 0) & ((out.symbol != "") | (out.name != ""))]
    out = out[~out.symbol.isin(["TOTAL", "NAN", "NONE"])]
    if out.empty:
        raise ValueError("The file has a header but no rows with a positive quantity.")
    return out.reset_index(drop=True)


def combine(tables: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Several brokers' files as one portfolio: quantities add, average cost is quantity-weighted."""
    t = pd.concat(list(tables), ignore_index=True)
    t["key"] = t.symbol.where(t.symbol != "", t.name.str.upper())
    t["cost"] = t.qty * t.avg
    g = t.groupby("key", sort=False)
    out = pd.DataFrame({"symbol": g.symbol.first(), "name": g.name.first(), "qty": g.qty.sum(),
                        "avg": g.apply(lambda x: x.cost.sum() / x.loc[x.avg.notna(), "qty"].sum() if x.avg.notna().any() else np.nan)})
    return out.reset_index(drop=True)


def match(rows: pd.DataFrame, known: set, names: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Add a `ticker` column: the universe ticker for each row, or None."""
    from utils.positions import resolve_ticker
    by_name = {re.sub(r"\b(LIMITED|LTD|THE)\b|[^A-Z0-9]", "", str(n).upper()): t for t, n in (names or {}).items()}
    out = []
    for r in rows.itertuples():
        t = resolve_ticker(r.symbol + ".NS", known) if r.symbol else None
        if t is None and r.name:
            k = re.sub(r"\b(LIMITED|LTD|THE)\b|[^A-Z0-9]", "", r.name.upper())
            t = by_name.get(k)
            if t is None and k:
                near = difflib.get_close_matches(k, list(by_name), n=1, cutoff=0.92)
                t = by_name[near[0]] if near else None
        out.append(t)
    return rows.assign(ticker=out)


def reconcile(rows: pd.DataFrame, positions: List[Dict[str, Any]], known: set, names: Optional[Dict[str, str]] = None,
              whole: bool = False) -> Dict[str, list]:
    """Compare broker rows with the book. Returns lists of dicts under add / update / close / unchanged / unknown."""
    from utils.positions import resolve_ticker
    m = match(rows, known, names)
    plan: Dict[str, list] = dict(add=[], update=[], close=[], unchanged=[], unknown=[])
    book = {}
    for p in positions:
        if p.get("status") in ("active", "watching"):
            book.setdefault(resolve_ticker(p["ticker"], known) or p["ticker"], p)
    seen = set()
    for r in m.itertuples():
        if not isinstance(r.ticker, str) or not r.ticker:          # pandas turns a missing match into NaN
            plan["unknown"].append(dict(symbol=r.symbol or r.name, qty=r.qty, avg=r.avg))
            continue
        seen.add(r.ticker)
        qty = int(round(r.qty))
        avg = None if pd.isna(r.avg) else round(float(r.avg), 2)
        p = book.get(r.ticker)
        if p is None:
            plan["add"].append(dict(ticker=r.ticker, qty=qty, avg=avg, was="not in the book"))
        elif p.get("status") == "watching":
            plan["add"].append(dict(ticker=r.ticker, qty=qty, avg=avg, was="on the watchlist", orig=p["ticker"]))
        else:
            oq, oa = int(p.get("quantity") or 0), p.get("entry_price")
            same = oq == qty and (avg is None or (oa and abs(float(oa) - avg) <= 0.005 * avg))
            (plan["unchanged"] if same else plan["update"]).append(
                dict(ticker=r.ticker, orig=p["ticker"], qty=qty, avg=avg, old_qty=oq, old_avg=oa))
    if whole:
        for t, p in book.items():
            if p.get("status") == "active" and t not in seen:
                plan["close"].append(dict(ticker=t, orig=p["ticker"], old_qty=int(p.get("quantity") or 0), old_avg=p.get("entry_price")))
    return plan


def apply(plan: Dict[str, list], prices: Dict[str, float], names: Optional[Dict[str, str]] = None, close: bool = True,
          source: str = "broker file") -> Dict[str, int]:
    """Write the plan to the book. Stops, targets and notes of existing holdings are not touched."""
    from utils import positions as pos
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    n = dict(added=0, updated=0, closed=0)
    for a in plan["add"]:
        entry = a["avg"] or prices.get(a["ticker"])
        if a.get("orig"):                                   # promote from the watchlist
            pos.update_position(a["orig"], status="active", entry_price=entry, quantity=a["qty"], entry_date=today, source=source,
                                synced_at=today)
        else:
            pos.add_position(a["ticker"], name=(names or {}).get(a["ticker"]), status="active", entry_price=entry, quantity=a["qty"])
            pos.update_position(a["ticker"], source=source, synced_at=today)
        n["added"] += 1
    for u in plan["update"]:
        f = dict(quantity=u["qty"], source=source, synced_at=today)
        if u["avg"]:
            f["entry_price"] = u["avg"]
        pos.update_position(u["orig"], **f)
        n["updated"] += 1
    if close:
        for c in plan["close"]:
            px = prices.get(c["ticker"])
            if px:
                pos.close_position(c["orig"], float(px))
                n["closed"] += 1
    return n
