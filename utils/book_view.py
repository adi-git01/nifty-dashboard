"""
📒 My Book: holdings and watchlist as two editable tables.

Edit stops, targets, quantity and notes in place, add rows at the bottom, delete rows, then press
Save once: every change goes to the GitHub-backed store in one go (utils/positions.py), instead
of one form and one reload per entry. Read-only columns show what matters today: price, P&L,
distance to the stop, a status flag, and where the stock stands in the funnel. The watchlist
says when a watched stock becomes buyable and lets you move it to holdings with a tick.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from utils import funnel, positions as pos

NEAR_STOP = 3.0
EDIT_H = ["Stock", "Entry", "Qty", "Stop", "Target", "Notes"]
EDIT_W = ["Stock", "Notes", "Bought"]


def norm(s: Any, known: set) -> Optional[str]:
    """User text -> a ticker in the snapshot (adds .NS, fixes near-misses like FEDERALBANK)."""
    t = str(s or "").strip().upper()
    if not t or t == "NAN":
        return None
    t = t if t.endswith((".NS", ".BO")) else t + ".NS"
    return pos.resolve_ticker(t, known) or t


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return f if np.isfinite(f) and f > 0 else None
    except (TypeError, ValueError):
        return None


def status(px, stop, target, scan, below50, caution) -> str:
    if not px:
        return "❔ no price"
    if stop and px <= stop:
        return "🔴 stop hit"
    if target and px >= target:
        return "🎯 target hit"
    if stop and (px / stop - 1) * 100 < NEAR_STOP:
        return f"⚠️ {(px / stop - 1) * 100:.1f}% above stop"
    if caution >= 2 or (below50 and scan != "BUY"):
        return "🟣 trend weakening"
    return "✅ on track"


def funnel_label(r: Optional[pd.Series]) -> str:
    if r is None:
        return "—"
    miss = [lab for k, lab in (("s_industry", "industry"), ("s_trend", "trend"), ("s_business", "business"), ("s_setup", "entry"))
            if not (pd.notna(r.get(k)) and bool(r.get(k))) and not (k == "s_business" and pd.isna(r.get(k)))]
    return "✅ Buyable now" if not miss else "Missing: " + ", ".join(miss)


def diff(before: pd.DataFrame, after: pd.DataFrame, known: set, cols: List[str]) -> Dict[str, list]:
    """What changed between the table as shown and as edited: adds, removes, updates (by ticker)."""
    b = {norm(r["Stock"], known): r for _, r in before.iterrows()}
    a: Dict[str, pd.Series] = {}
    for _, r in after.iterrows():
        t = norm(r.get("Stock"), known)
        if t:
            a[t] = r
    out = dict(add=[], remove=[t for t in b if t not in a], update=[], unknown=[])
    same = lambda x, y: (pd.isna(x) and pd.isna(y)) or x == y or (str(x or "") == str(y or ""))
    for t, r in a.items():
        if t not in b:
            (out["add"] if t in known else out["unknown"]).append((t, r))
        elif any(not same(r.get(c), b[t].get(c)) for c in cols if c != "Stock"):
            out["update"].append((t, r))
    return out


def _context(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    from utils import funnel_view as fv
    from utils.industry_board_view import get_board
    sub = df[df.ticker.isin(tickers)]
    if sub.empty:
        return pd.DataFrame()
    T = funnel.table(sub, get_board(df), fv.valuation(), business_for=tickers).set_index("ticker")
    x = sub.drop_duplicates("ticker").set_index("ticker")
    T["below50"] = pd.to_numeric(x.get("currentPrice"), errors="coerce") < pd.to_numeric(x.get("fiftyDayAverage"), errors="coerce")
    T["caution"] = pd.to_numeric(x["simba_caution"], errors="coerce").fillna(0) if "simba_caution" in x else 0
    return T


def render(df: pd.DataFrame):
    known = set(df["ticker"])
    allp = pos.get_all_positions()
    hold = [p for p in allp if p.get("status") == "active"]
    watch = [p for p in allp if p.get("status") == "watching"]
    res = {p["ticker"]: (pos.resolve_ticker(p["ticker"], known) or p["ticker"]) for p in hold + watch}
    C = _context(df, list(res.values()))
    row = lambda t: C.loc[res[t]] if len(C) and res[t] in C.index else None
    ver = st.session_state.setdefault("book_ver", 0)
    money = lambda t: st.column_config.NumberColumn(t, format="₹%.2f", min_value=0.0)

    # ------------------------------------------------------------------ holdings
    st.markdown("#### 📈 Holdings")
    H = []
    for p in hold:
        r = row(p["ticker"])
        px = float(r.price) if r is not None and pd.notna(r.price) else None
        e, sl, tg = _num(p.get("entry_price")), _num(p.get("stop_loss")), _num(p.get("target"))
        H.append({"Stock": p["ticker"].replace(".NS", ""), "Entry": e, "Qty": p.get("quantity") or 1, "Stop": sl, "Target": tg,
                  "Notes": p.get("notes") or "", "Price": px, "P&L %": (px / e - 1) * 100 if px and e else None,
                  "Above stop %": (px / sl - 1) * 100 if px and sl else None,
                  "Status": status(px, sl, tg, r.scanner if r is not None else "", bool(r.below50) if r is not None else False,
                                   float(r.caution) if r is not None else 0),
                  "Technicals": r.technicals if r is not None else "—",
                  "Industry": str(r.ind_group).title() if r is not None and r.ind_group else "—"})
    Hd = pd.DataFrame(H, columns=["Stock", "Entry", "Qty", "Stop", "Target", "Notes", "Price", "P&L %", "Above stop %", "Status",
                                  "Technicals", "Industry"])
    need = Hd[~Hd.Status.str.startswith("✅")] if len(Hd) else Hd
    if len(need):
        st.warning("Needs a look: " + ", ".join(f"{r.Stock} ({r.Status})" for r in need.itertuples()))
    He = st.data_editor(Hd, key=f"book_h_{ver}", num_rows="dynamic", hide_index=True, use_container_width=True,
                        disabled=[c for c in Hd.columns if c not in EDIT_H], column_config={
        "Stock": st.column_config.TextColumn("Stock", help="NSE symbol, e.g. CUMMINSIND", required=True),
        "Entry": money("Entry"), "Stop": money("Stop"), "Target": money("Target"),
        "Qty": st.column_config.NumberColumn("Qty", min_value=1, step=1, format="%d"),
        "Notes": st.column_config.TextColumn("Notes", width="medium"),
        "Price": st.column_config.NumberColumn(format="₹%.2f"), "P&L %": st.column_config.NumberColumn(format="%+.1f%%"),
        "Above stop %": st.column_config.NumberColumn(format="%.1f%%")})
    st.caption("Click a cell to edit · add a row at the bottom · select a row and press Delete to remove it. "
               "Nothing is saved until you press **Save changes**.")

    # ------------------------------------------------------------------ watchlist
    st.markdown("#### 👀 Watchlist")
    W = []
    for p in watch:
        r = row(p["ticker"])
        px = float(r.price) if r is not None and pd.notna(r.price) else None
        e = _num(p.get("entry_price"))
        W.append({"Stock": p["ticker"].replace(".NS", ""), "Notes": p.get("notes") or "", "Bought": False,
                  "Funnel": funnel_label(r), "Technicals": r.technicals if r is not None else "—", "Price": px,
                  "Stop if bought": float(r.stop) if r is not None and pd.notna(r.stop) and px and r.stop < px else None,
                  "Risk %": float(r.risk_pct) if r is not None and pd.notna(r.risk_pct) else None,
                  "Since added %": (px / e - 1) * 100 if px and e else None, "Added": str(p.get("entry_date") or "")[:10],
                  "Industry": str(r.ind_group).title() if r is not None and r.ind_group else "—"})
    Wd = pd.DataFrame(W, columns=["Stock", "Notes", "Bought", "Funnel", "Technicals", "Price", "Stop if bought", "Risk %",
                                  "Since added %", "Added", "Industry"])
    ready = Wd[Wd.Funnel.str.startswith("✅")] if len(Wd) else Wd
    if len(ready):
        st.success("Buyable now: " + ", ".join(ready.Stock) + " — passes industry, trend, business and entry.")
    We = st.data_editor(Wd, key=f"book_w_{ver}", num_rows="dynamic", hide_index=True, use_container_width=True,
                        disabled=[c for c in Wd.columns if c not in EDIT_W], column_config={
        "Stock": st.column_config.TextColumn("Stock", required=True),
        "Notes": st.column_config.TextColumn("Why I'm watching", width="medium"),
        "Bought": st.column_config.CheckboxColumn("Bought", help="Tick and save: moves it to Holdings at today's price with the suggested stop"),
        "Funnel": st.column_config.TextColumn("Funnel", help="Which funnel stage is still missing"),
        "Price": st.column_config.NumberColumn(format="₹%.2f"), "Stop if bought": st.column_config.NumberColumn(format="₹%.2f"),
        "Risk %": st.column_config.NumberColumn(format="%.1f%%"), "Since added %": st.column_config.NumberColumn(format="%+.1f%%")})
    names = dict(zip(df["ticker"], df.get("name", df["ticker"]).astype(str)))
    have = set(res.values())
    q1, q2 = st.columns([4, 1])
    quick = q1.multiselect("Quick add to watchlist", [t for t in df["ticker"] if t not in have], key=f"book_quick_{ver}",
                           format_func=lambda t: f"{t.replace('.NS', '')} — {names.get(t, '')}", placeholder="Type names or tickers…",
                           label_visibility="collapsed")

    # ------------------------------------------------------------------ save
    dh, dw = diff(Hd, He, known, EDIT_H), diff(Wd, We, known, ["Stock", "Notes"])
    bought = [norm(r["Stock"], known) for _, r in We.iterrows() if r.get("Bought") is True and norm(r["Stock"], known) in known]
    n = sum(len(d[k]) for d in (dh, dw) for k in ("add", "remove", "update")) + len(quick) + len(bought)
    bad = [t.replace(".NS", "") for d in (dh, dw) for t, _ in d["unknown"]]
    if bad:
        st.error("Not found in the universe, so not saved: " + ", ".join(bad))
    if q2.button(f"💾 Save changes ({n})" if n else "💾 Save changes", type="primary", disabled=not n, use_container_width=True,
                 key="book_save"):
        for t, r in dh["add"]:
            px = float(df.loc[df.ticker == t, "currentPrice"].iloc[0])
            pos.add_position(t, name=names.get(t), status="active", entry_price=_num(r.get("Entry")) or px,
                             stop_loss=_num(r.get("Stop")), target=_num(r.get("Target")), quantity=int(_num(r.get("Qty")) or 1),
                             notes=str(r.get("Notes") or ""))
        for t, r in dh["update"]:
            orig = next(p["ticker"] for p in hold if res[p["ticker"]] == t or p["ticker"] == t)
            pos.update_position(orig, entry_price=_num(r.get("Entry")), stop_loss=_num(r.get("Stop")), target=_num(r.get("Target")),
                                quantity=int(_num(r.get("Qty")) or 1), notes=str(r.get("Notes") or ""))
        for t, r in dw["add"]:
            px = float(df.loc[df.ticker == t, "currentPrice"].iloc[0])
            pos.add_position(t, name=names.get(t), status="watching", entry_price=px, notes=str(r.get("Notes") or ""))
        for t, r in dw["update"]:
            orig = next(p["ticker"] for p in watch if res[p["ticker"]] == t or p["ticker"] == t)
            pos.update_position(orig, notes=str(r.get("Notes") or ""))
        for t in quick:
            pos.add_position(t, name=names.get(t), status="watching", entry_price=float(df.loc[df.ticker == t, "currentPrice"].iloc[0]))
        for t in bought:
            orig = next((p["ticker"] for p in watch if res[p["ticker"]] == t or p["ticker"] == t), t)
            r = C.loc[t] if len(C) and t in C.index else None
            px = float(df.loc[df.ticker == t, "currentPrice"].iloc[0])
            fields = dict(status="active", entry_price=px, entry_date=pd.Timestamp.today().strftime("%Y-%m-%d"))
            if r is not None and pd.notna(r.stop) and r.stop < px:
                fields["stop_loss"] = float(r.stop)
            if r is not None and pd.notna(r.t1) and r.t1 > px:
                fields["target"] = float(r.t1)
            pos.update_position(orig, **fields) if pos.get_position(orig) else pos.add_position(t, status="active", entry_price=px)
        for group, d in ((hold, dh), (watch, dw)):
            for p in group:
                if (res[p["ticker"]] in d["remove"] or p["ticker"] in d["remove"]) and res[p["ticker"]] not in bought:
                    pos.remove_position(p["ticker"])
        st.session_state["book_ver"] = ver + 1
        st.toast(f"Saved {n} change(s)", icon="💾")
        st.rerun()
    st.caption("Watchlist stops and risk are the Technicals levels from the last close. Price, RS and industry alerts are on the "
               "**Alerts** tab.")
