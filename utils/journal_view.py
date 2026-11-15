"""
📓 Journal tab: every buy and sell with what the dashboard saw at the time, folded into round
trips, and grouped to show what has and has not worked for you (utils/trade_journal.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils import positions as pos, trade_journal as tj

MIN_GROUP = 20          # closed trades per group before a difference means anything
ENTRY_FACTS = [("stages_passed", "Funnel stages passed (of 4)"), ("ind_group", "Industry group"), ("industry", "Industry"),
               ("scanner", "Scanner"), ("technicals", "Technicals"), ("comp_rs", "RS vs Nifty"), ("rs_3m", "3-month RS"),
               ("dist_52w", "From 52-week high %"), ("stage", "Chart stage"), ("above_50dma", "Above 50-day average"),
               ("above_200dma", "Above 200-day average"), ("biz_score", "Business score"), ("biz_verdict", "Business verdict"),
               ("val_pct", "Valuation percentile"), ("risk_pct", "Stop distance %"), ("regime_nifty", "Nifty regime"),
               ("regime_breadth", "Breadth regime"), ("breadth_pct", "Breadth %"), ("status", "Holding status"),
               ("exit_rule_hit", "Exit rule had fired"), ("peak_gain_pct", "Best gain while held %"),
               ("from_peak_pct", "Given back from the peak %")]


def _prices(df: pd.DataFrame) -> dict:
    return pd.to_numeric(df.drop_duplicates("ticker").set_index("ticker")["currentPrice"], errors="coerce").dropna().to_dict()


def _catch_up(df: pd.DataFrame, evs: list) -> list:
    """Once per book version: add what the journal missed (older holdings, closes on the classic tabs)."""
    from utils import book_view as bv
    key = ("journal_caught", st.session_state.get("book_ver", 0), len(evs))
    if st.session_state.get("journal_caught") == key:
        return evs
    known = bv._known(df)[0]
    new = tj.catch_up(pos.get_all_positions(), evs, resolve=lambda t: pos.resolve_ticker(t, known) or t)
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    fresh = [e for e in new if e["action"] == "BUY" and e["date"] == today]       # bought today on a classic tab: readings still valid
    if fresh:
        try:
            from utils import funnel_view as fv
            from utils.industry_board_view import get_board
            M = tj.capture([e["ticker"] for e in fresh], df, get_board(df), fv.valuation(), fv.regime(df))
            for e in fresh:
                e["metrics"], e["source"] = M.get(e["ticker"]) or {}, "book"
        except Exception:
            pass
    if new:
        tj.record(new)
        evs = tj.events()
    st.session_state["journal_caught"] = ("journal_caught", st.session_state.get("book_ver", 0), len(evs))
    return evs


def _show(T: pd.DataFrame, cols: dict) -> pd.DataFrame:
    out = pd.DataFrame({lab: (T[c] if c in T.columns else None) for lab, c in cols.items()})
    out["Stock"] = out["Stock"].astype(str).str.replace(".NS", "", regex=False)
    return out


def _fmt(v):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        return f"{v:.1f}"
    return str(v)


def render(df: pd.DataFrame):
    evs = _catch_up(df, tj.events())
    if not evs:
        st.info("No trades yet. The journal fills itself: sync a broker on **📥 Import**, or add / tick **Sold** on **📒 My Book**. "
                "Each buy and sell is stored with the funnel, technical, business, valuation and market readings of that day.")
        return
    T = tj.trades(evs, _prices(df))
    C, O = T[T.status == "closed"].reset_index(drop=True), T[T.status == "open"].reset_index(drop=True)
    S = tj.stats(T)
    ver = st.session_state.setdefault("journal_ver", 0)

    # ------------------------------------------------------------------ score
    m = st.columns(6)
    m[0].metric("Closed trades", S["closed"], f"{S['open']} open", delta_color="off")
    if S["closed"]:
        m[1].metric("Win rate", f"{S['win_rate']:.0f}%")
        m[2].metric("Average win", f"{S['avg_win']:+.1f}%" if pd.notna(S["avg_win"]) else "—")
        m[3].metric("Average loss", f"{S['avg_loss']:+.1f}%" if pd.notna(S["avg_loss"]) else "—")
        m[4].metric("Per trade (expectancy)", f"{S['expectancy']:+.1f}%", f"median {S['median']:+.1f}%", delta_color="off")
        m[5].metric("Realised P&L", f"₹{S['pnl']:,.0f}", f"{S['avg_days']:.0f} days held on average", delta_color="off")
        est = int(C.exit_estimated.sum())
        if est:
            st.caption(f"≈ {est} of {len(C)} sale price(s) are the day's close, not your fill (a holdings sync does not give it). "
                       "Type the real price in **Sold at** below to make the returns exact.")

    # ------------------------------------------------------------------ patterns
    st.markdown("#### 🔎 What worked")
    if len(C) < 5:
        st.caption(f"Patterns appear once there are closed trades to compare ({len(C)} so far). Until then the journal is collecting: "
                   "every entry already stores the readings below.")
    else:
        dim = st.selectbox("Group closed trades by", list(tj.DIMENSIONS), key="journal_dim")
        P = tj.patterns(T, dim)
        if P.empty:
            st.caption("Not recorded for these trades.")
        else:
            st.dataframe(P.rename(columns={"group": dim, "trades": "Trades", "win_rate": "Win rate", "avg_return": "Avg return",
                                           "median_return": "Median return", "avg_days": "Avg days", "pnl": "P&L"}),
                         hide_index=True, use_container_width=True, column_config={
                "Win rate": st.column_config.NumberColumn(format="%.0f%%"), "Avg return": st.column_config.NumberColumn(format="%+.1f%%"),
                "Median return": st.column_config.NumberColumn(format="%+.1f%%"), "Avg days": st.column_config.NumberColumn(format="%.0f"),
                "P&L": st.column_config.NumberColumn(format="₹%.0f")})
            small = int((P.trades < MIN_GROUP).sum())
            if small:
                st.caption(f"⚠️ {small} of {len(P)} group(s) have fewer than {MIN_GROUP} trades. A gap between such groups is "
                           "usually luck — treat it as a question to watch, not a rule to trade.")
        st.caption("'not recorded' = bought before the journal existed, so that day's readings were never captured.")

    # ------------------------------------------------------------------ closed
    edits = []
    if len(C):
        st.markdown("#### ✅ Closed trades")
        Cd = _show(C, {"Stock": "ticker", "Bought": "entry_date", "Sold": "exit_date", "Days": "days", "Qty": "qty",
                       "Entry": "entry_price", "Sold at": "exit_price", "Return %": "return_pct", "P&L": "pnl",
                       "Stages at entry": "entry_stages_passed", "Technicals at entry": "entry_technicals",
                       "Industry at entry": "entry_ind_group", "Status when sold": "exit_status", "Why I bought": "entry_note",
                       "Why I sold / lesson": "exit_note", "Tag": "tags"})
        Cd.insert(7, "≈", np.where(C.exit_estimated, "≈", ""))
        ok = ["Sold at", "Why I bought", "Why I sold / lesson", "Tag"]
        Ce = st.data_editor(Cd, key=f"journal_c_{ver}", hide_index=True, use_container_width=True,
                            disabled=[c for c in Cd.columns if c not in ok], column_config={
            "Entry": st.column_config.NumberColumn(format="₹%.2f"), "Sold at": st.column_config.NumberColumn(format="₹%.2f", min_value=0.0),
            "≈": st.column_config.TextColumn("≈", help="Sale price is the day's close, not your fill", width="small"),
            "Return %": st.column_config.NumberColumn(format="%+.1f%%"), "P&L": st.column_config.NumberColumn(format="₹%.0f"),
            "Stages at entry": st.column_config.NumberColumn(format="%d / 4"),
            "Tag": st.column_config.TextColumn("Tag", help="Your own label, e.g. breakout, results, tip, averaging down")})
        for i in range(len(C)):
            a, b, r = Ce.iloc[i], Cd.iloc[i], C.iloc[i]
            if pd.notna(a["Sold at"]) and (pd.isna(b["Sold at"]) or abs(a["Sold at"] - b["Sold at"]) > 1e-6) and r.exit_event:
                if r.partial_exits:
                    st.warning(f"{b['Stock']} was sold in parts: correct each sale in **All entries** below.")
                else:
                    edits.append((r.exit_event, dict(price=float(a["Sold at"]), estimated=False)))
            if str(a["Why I sold / lesson"] or "") != str(b["Why I sold / lesson"] or "") and r.exit_event:
                edits.append((r.exit_event, dict(notes=str(a["Why I sold / lesson"] or ""))))
            f = {k: str(a[c] or "") for c, k in (("Why I bought", "notes"), ("Tag", "tags")) if str(a[c] or "") != str(b[c] or "")}
            if f:
                edits.append((r.entry_event, f))

    # ------------------------------------------------------------------ open
    if len(O):
        st.markdown("#### 📈 Open trades")
        Od = _show(O, {"Stock": "ticker", "Bought": "entry_date", "Days": "days", "Qty": "open_qty", "Entry": "entry_price",
                       "Now %": "return_pct", "Stages at entry": "entry_stages_passed", "Technicals at entry": "entry_technicals",
                       "Industry at entry": "entry_ind_group", "From": "source", "Why I bought": "entry_note", "Tag": "tags"})
        Oe = st.data_editor(Od, key=f"journal_o_{ver}", hide_index=True, use_container_width=True,
                            disabled=[c for c in Od.columns if c not in ("Why I bought", "Tag")], column_config={
            "Entry": st.column_config.NumberColumn(format="₹%.2f"), "Now %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Stages at entry": st.column_config.NumberColumn(format="%d / 4")})
        for i in range(len(O)):
            a, b = Oe.iloc[i], Od.iloc[i]
            f = {k: str(a[c] or "") for c, k in (("Why I bought", "notes"), ("Tag", "tags")) if str(a[c] or "") != str(b[c] or "")}
            if f:
                edits.append((O.iloc[i].entry_event, f))

    # ------------------------------------------------------------------ one trade, entry against exit
    with st.expander("🔬 One trade: the readings at entry and at exit"):
        lab = lambda i: (f"{T.ticker[i].replace('.NS', '')} · {T.entry_date[i]} → {T.exit_date[i] if pd.notna(T.exit_date[i]) else 'open'}"
                         + (f" · {T.return_pct[i]:+.1f}%" if pd.notna(T.return_pct[i]) else ""))
        i = st.selectbox("Trade", list(T.index), format_func=lab, key="journal_pick")
        r = T.loc[i]
        rows = [{"Reading": lab_, "At entry": _fmt(r.get(f"entry_{k}")), "At exit": _fmt(r.get(f"exit_{k}"))} for k, lab_ in ENTRY_FACTS
                if f"entry_{k}" in T.columns or f"exit_{k}" in T.columns]
        if rows and any(x["At entry"] != "—" or x["At exit"] != "—" for x in rows):
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.caption("No readings were captured for this trade (it pre-dates the journal).")

    # ------------------------------------------------------------------ raw entries
    with st.expander(f"🧾 All entries ({len(evs)}) — correct a price, quantity or date"):
        E = pd.DataFrame([{"Date": e.get("date"), "Stock": str(e.get("ticker", "")).replace(".NS", ""), "Action": e.get("action"),
                           "Qty": e.get("qty"), "Price": e.get("price"), "≈": "≈" if e.get("estimated") else "",
                           "From": e.get("source"), "Note": e.get("notes") or ""} for e in reversed(evs)])
        ids = [e["id"] for e in reversed(evs)]
        Ee = st.data_editor(E, key=f"journal_e_{ver}", hide_index=True, use_container_width=True,
                            disabled=["Stock", "Action", "≈", "From"],
                            column_config={"Price": st.column_config.NumberColumn(format="₹%.2f", min_value=0.0),
                                           "Qty": st.column_config.NumberColumn(format="%.0f", min_value=0.0),
                                           "Date": st.column_config.TextColumn("Date", help="YYYY-MM-DD")})
        for j, eid in enumerate(ids):
            a, b, f = Ee.iloc[j], E.iloc[j], {}
            if pd.notna(a["Price"]) and (pd.isna(b["Price"]) or abs(a["Price"] - b["Price"]) > 1e-6):
                f.update(price=float(a["Price"]), estimated=False)
            if pd.notna(a["Qty"]) and a["Qty"] != b["Qty"]:
                f["qty"] = float(a["Qty"])
            if str(a["Date"] or "") != str(b["Date"] or "") and pd.notna(pd.to_datetime(a["Date"], errors="coerce")):
                f["date"] = pd.to_datetime(a["Date"]).strftime("%Y-%m-%d")
            if str(a["Note"] or "") != str(b["Note"] or ""):
                f["notes"] = str(a["Note"] or "")
            if f:
                edits.append((eid, f))

    merged: dict = {}
    for eid, f in edits:
        merged.setdefault(eid, {}).update(f)
    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button(f"💾 Save journal edits ({len(merged)})" if merged else "💾 Save journal edits", type="primary", disabled=not merged,
                 key="journal_save", use_container_width=True):
        for eid, f in merged.items():
            tj.patch(eid, **f)
        st.session_state["journal_ver"] = ver + 1
        st.toast(f"Saved {len(merged)} edit(s)", icon="📓")
        st.rerun()
    c2.download_button("⬇️ Trades (CSV)", T.to_csv(index=False).encode(), "trade_journal.csv", "text/csv", key="journal_csv",
                       use_container_width=True, help="One row per trade with every entry_ and exit_ reading, for your own analysis")
    info = pos.storage_status() if hasattr(pos, "storage_status") else {}
    c3.caption("Stored next to your book" + (f" ({info.get('where')})" if info.get("where") else "")
               + ". Buys via sync carry your broker's average cost; sales via sync carry the day's close until you correct them.")
