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
from utils.book_logic import NEAR_STOP, _num, exit_guide, funnel_label, peak_since, risk_summary, status  # noqa: F401

EDIT_H = ["Stock", "Entry", "Qty", "Stop", "Target", "Notes"]
EDIT_W = ["Stock", "Notes", "Bought"]


def norm(s: Any, known: set) -> Optional[str]:
    """User text -> a ticker in the snapshot (adds .NS, fixes near-misses like FEDERALBANK)."""
    t = str(s or "").strip().upper()
    if not t or t == "NAN":
        return None
    t = t if t.endswith((".NS", ".BO")) else t + ".NS"
    return pos.resolve_ticker(t, known) or t


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
    T["ma50"] = pd.to_numeric(x.get("fiftyDayAverage"), errors="coerce")
    T["traded"] = pd.to_numeric(x.get("averageVolume"), errors="coerce") * pd.to_numeric(x.get("currentPrice"), errors="coerce")
    return T


IMPORT_TAB = "📥 Import"


def handle_broker_redirect() -> bool:
    """
    Called early in main.py. If this page load is a broker sending the user back after login
    (Zerodha: ?request_token=..., Dhan: ?tokenId=...), exchange the one-time code, fetch holdings,
    keep them for the Import tab's preview, and send the user there. Returns True if it handled one.
    """
    qp = st.query_params
    kind = "kite" if "request_token" in qp else "dhan" if "tokenId" in qp else None
    if kind is None:
        return False
    from utils import broker_sync as bs
    cfg = bs.config()
    code = qp.get("request_token") if kind == "kite" else qp.get("tokenId")
    label = "Zerodha" if kind == "kite" else "Dhan"
    try:
        if kind not in cfg:
            raise bs.BrokerError(f"{label} keys are not in the app's secrets.")
        if kind == "kite":
            if qp.get("status", "success") != "success":
                raise bs.BrokerError("Zerodha login was cancelled.")
            tok = bs.kite_session(cfg["kite"]["api_key"], cfg["kite"]["api_secret"], code)
            rows = bs.kite_holdings(cfg["kite"]["api_key"], tok)
        else:
            tok = bs.dhan_session(code, cfg["dhan"]["app_id"], cfg["dhan"]["app_secret"])
            rows = bs.dhan_holdings(tok)
        st.session_state[f"{kind}_token"] = tok                      # this browser session only
        st.session_state.setdefault("sync_rows", {})[label] = rows
        st.session_state["sync_note"] = f"{label}: {len(rows)} holding(s) fetched. Check the preview below and press Apply."
    except Exception as e:
        st.session_state["sync_error"] = str(e) if isinstance(e, bs.BrokerError) else f"{label} sync failed: {type(e).__name__}"
    for k in ("request_token", "tokenId", "status", "action", "type"):
        if k in qp:
            del qp[k]
    st.session_state["nav_page"] = "📊 Return Tracker"
    st.session_state["desk_tab"] = IMPORT_TAB
    return True


def _sync_panel():
    """Buttons to pull holdings from Zerodha and Dhan (utils/broker_sync.py)."""
    from utils import broker_sync as bs
    cfg = bs.config()
    st.markdown("#### 🔄 Sync from your broker")
    if st.session_state.get("sync_error"):
        st.error(st.session_state.pop("sync_error"))
    if st.session_state.get("sync_note"):
        st.success(st.session_state.pop("sync_note"))
    got = st.session_state.setdefault("sync_rows", {})

    def fetch(label, fn):
        try:
            got[label] = fn()
            st.session_state["sync_note"] = f"{label}: {len(got[label])} holding(s) fetched. Check the preview below and press Apply."
        except Exception as e:
            st.session_state["sync_error"] = str(e) if isinstance(e, bs.BrokerError) else f"{label} sync failed: {type(e).__name__}"
        st.rerun()

    z, d = st.columns(2)
    with z:
        st.markdown("**Zerodha**" + (f" — {len(got['Zerodha'])} holding(s) loaded" if "Zerodha" in got else ""))
        if "kite" not in cfg:
            st.caption("Not set up. Add `[kite]` `api_key` and `api_secret` to the app's secrets (see the setup notes below).")
        elif st.session_state.get("kite_token"):
            if st.button("🔄 Fetch Zerodha holdings", key="sync_kite_fetch", use_container_width=True):
                fetch("Zerodha", lambda: bs.kite_holdings(cfg["kite"]["api_key"], st.session_state["kite_token"]))
        else:
            st.link_button("🔐 Log in to Zerodha and sync", bs.kite_login_url(cfg["kite"]["api_key"]), use_container_width=True)
            st.caption("Opens Zerodha's own login page, then returns here with your holdings.")
    with d:
        st.markdown("**Dhan**" + (f" — {len(got['Dhan'])} holding(s) loaded" if "Dhan" in got else ""))
        if st.session_state.get("dhan_token"):
            if st.button("🔄 Fetch Dhan holdings", key="sync_dhan_fetch", use_container_width=True):
                fetch("Dhan", lambda: bs.dhan_holdings(st.session_state["dhan_token"]))
        else:
            if "dhan" in cfg:
                if st.button("🔐 Log in to Dhan and sync", key="sync_dhan_login", use_container_width=True):
                    try:
                        st.session_state["dhan_login_url"] = bs.dhan_login_url(
                            bs.dhan_consent(cfg["dhan"]["client_id"], cfg["dhan"]["app_id"], cfg["dhan"]["app_secret"]))
                    except Exception as e:
                        st.session_state["sync_error"] = str(e) if isinstance(e, bs.BrokerError) else f"Dhan login failed: {type(e).__name__}"
                    st.rerun()
                if st.session_state.get("dhan_login_url"):
                    st.link_button("➡️ Continue to Dhan's login page", st.session_state["dhan_login_url"], use_container_width=True)
            tok = st.text_input("…or paste a Dhan access token", type="password", key="sync_dhan_paste",
                                help="web.dhan.co → My Profile → Access DhanHQ APIs → generate an access token (valid 24 hours). "
                                     "Kept for this browser session only.")
            if tok and st.button("Use this token and fetch", key="sync_dhan_use", use_container_width=True):
                st.session_state["dhan_token"] = tok.strip()
                fetch("Dhan", lambda: bs.dhan_holdings(st.session_state["dhan_token"]))
    if got and st.button("Clear fetched holdings", key="sync_clear"):
        st.session_state["sync_rows"] = {}
        st.rerun()
    with st.expander("Setup notes (one time)"):
        st.markdown(
            "**Zerodha** — create an app at developers.kite.trade (the Personal plan covers holdings), set its **Redirect URL** to "
            "this dashboard's address, then add to the app's secrets:\n\n```toml\n[kite]\napi_key = \"…\"\napi_secret = \"…\"\n```\n\n"
            "**Dhan** — simplest is to paste a 24-hour access token above. For one-tap login instead, create an API key at "
            "web.dhan.co (My Profile → Access DhanHQ APIs) with this dashboard's address as the redirect URL, then add:\n\n"
            "```toml\n[dhan]\nclient_id = \"…\"\napp_id = \"…\"\napp_secret = \"…\"\n```\n\n"
            "Both brokers make you log in again each day; the sync only reads holdings and cannot place orders. Login tokens stay in "
            "this browser session and are never saved.")


def _plan_panel(df: pd.DataFrame, tables: list, sources: list, allp: list, known: set, ver: int):
    """Preview what an import or sync would change, and apply it once."""
    from utils import portfolio_import as pi
    whole = st.checkbox("This is my whole portfolio (propose closing holdings that are not in it)", key="book_whole")
    names = _known(df)[1]
    prices = pd.to_numeric(df.drop_duplicates("ticker").set_index("ticker")["currentPrice"], errors="coerce").dropna().to_dict()
    plan = pi.reconcile(pi.combine(tables), allp, known, names, whole)
    outside = [a["ticker"] for a in plan["add"] + plan["update"] if a["ticker"] not in prices]
    if outside:
        st.info("🔭 Outside the ranked universe, tracked from now on: " + ", ".join(x.replace(".NS", "") for x in outside)
                + ". They are priced on the next page load and join the nightly run.")
    m = st.columns(5)
    for col, (k, lab) in zip(m, (("add", "New"), ("update", "Changed"), ("close", "Not in it"), ("unchanged", "Unchanged"),
                                 ("unknown", "Not recognised"))):
        col.metric(lab, len(plan[k]))
    t = lambda x: str(x).replace(".NS", "")
    rows = ([{"Change": "➕ Add", "Stock": t(a["ticker"]), "Qty": a["qty"], "Avg cost": a["avg"], "Before": a["was"]} for a in plan["add"]]
            + [{"Change": "✏️ Update", "Stock": t(u["ticker"]), "Qty": u["qty"], "Avg cost": u["avg"],
                "Before": f"qty {u['old_qty']}, avg {u['old_avg']}"} for u in plan["update"]]
            + [{"Change": "🚪 Close (sold?)", "Stock": t(c["ticker"]), "Qty": 0, "Avg cost": None,
                "Before": f"qty {c['old_qty']}, avg {c['old_avg']}"} for c in plan["close"]])
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={"Avg cost": st.column_config.NumberColumn(format="₹%.2f")})
    else:
        st.success("Your book already matches.")
    if plan["unknown"]:
        st.warning("Not an NSE equity share the dashboard can track (ETFs, bonds, BSE-only or unlisted names): "
                   + ", ".join(f"{u['symbol']} ({u['qty']:.0f})" for u in plan["unknown"]))
    info = pos.storage_status() if hasattr(pos, "storage_status") else {}
    st.caption("New holdings get today's date as their entry date (brokers do not give the purchase date) and no stop: "
               "set one from the Exit guide column. Stops, targets and notes you already have are kept."
               + (f" Your book is saved to {info.get('where', 'GitHub')}"
                  " — if that repository or this dashboard is public, so are these quantities." if info else ""))
    close = st.checkbox(f"Also close the {len(plan['close'])} holding(s) not in it", value=False, key="book_close") if plan["close"] else False
    n = len(plan["add"]) + len(plan["update"]) + (len(plan["close"]) if close else 0)
    if st.button(f"✅ Apply ({n} change(s))", type="primary", disabled=not n, key="book_apply"):
        done = pi.apply(plan, prices, names, close=close, source=", ".join(sources)[:80])
        st.session_state["book_ver"] = ver + 1
        st.session_state["sync_rows"] = {}
        st.toast(f"Imported: {done['added']} added, {done['updated']} updated, {done['closed']} closed", icon="📥")
        st.rerun()


def _import_panel(df: pd.DataFrame, allp: list, known: set, ver: int):
    """Sync from a broker and / or upload broker files, preview the changes, apply once."""
    from utils import portfolio_import as pi
    _sync_panel()
    st.markdown("#### 📥 Or import a holdings file")
    st.caption("Download your holdings from the broker (Zerodha: Kite → Holdings → download, or Console → Portfolio → Holdings; "
               "Dhan: Portfolio → Holdings → download) and drop the file here. Files and synced brokers are combined into one portfolio.")
    files = st.file_uploader("Holdings files", type=["csv", "xlsx", "xls", "tsv", "txt"], accept_multiple_files=True,
                             key=f"book_up_{ver}", label_visibility="collapsed")
    tables, sources = [], []
    for label, rows in (st.session_state.get("sync_rows") or {}).items():
        if rows is not None and len(rows):
            tables.append(rows)
            sources.append(f"{label} sync")
    for f in files or []:
        try:
            t = pi.parse(f.getvalue(), f.name)
            tables.append(t)
            sources.append(f.name)
            st.caption(f"✅ {f.name}: {len(t)} holding(s) read")
        except Exception as e:
            st.error(f"{f.name}: {e}")
    if not tables:
        return
    st.markdown("#### 👀 Preview — " + ", ".join(sources))
    _plan_panel(df, tables, sources, allp, known, ver)


def _px(df: pd.DataFrame, t: str) -> Optional[float]:
    """Latest price of `t` in the snapshot, or None for a stock not priced yet (new extended name)."""
    x = pd.to_numeric(df.loc[df.ticker == t, "currentPrice"], errors="coerce").dropna()
    return float(x.iloc[0]) if len(x) else None


def _known(df: pd.DataFrame):
    """Tickers the book accepts: the snapshot plus any other NSE equity (tracked nightly once added). -> (set, names)"""
    names = dict(zip(df["ticker"], df.get("name", df["ticker"]).astype(str)))
    try:
        from utils import extended
        for t, n in extended.known_outside_core(set(df["ticker"])).items():
            names.setdefault(t, n)
    except Exception:
        pass
    return set(names), names


def render_import(df: pd.DataFrame):
    """The 📥 Import tab: upload broker files, preview, apply."""
    _import_panel(df, pos.get_all_positions(), _known(df)[0], st.session_state.setdefault("book_ver", 0))


def render(df: pd.DataFrame):
    known = _known(df)[0]
    allp = pos.get_all_positions()
    st.info("📥 To load your holdings from **Zerodha, Dhan or any broker file**, open the **📥 Import** tab above.")
    hold = [p for p in allp if p.get("status") == "active"]
    watch = [p for p in allp if p.get("status") == "watching"]
    res = {p["ticker"]: (pos.resolve_ticker(p["ticker"], known) or p["ticker"]) for p in hold + watch}
    C = _context(df, list(res.values()))
    row = lambda t: C.loc[res[t]] if len(C) and res[t] in C.index else None
    ver = st.session_state.setdefault("book_ver", 0)
    money = lambda t: st.column_config.NumberColumn(t, format="₹%.2f", min_value=0.0)

    # ------------------------------------------------------------------ holdings
    from utils import funnel_view as fv
    try:
        trail = float(fv.regime(df)["info"]["engine"].get("trail_stop") or 0.12)
    except Exception:
        trail = 0.12
    st.markdown("#### 📈 Holdings")
    H = []
    for p in hold:
        r = row(p["ticker"])
        px = float(r.price) if r is not None and pd.notna(r.price) else None
        e, sl, tg = _num(p.get("entry_price")), _num(p.get("stop_loss")), _num(p.get("target"))
        g = exit_guide(px, float(r.ma50) if r is not None and pd.notna(r.ma50) else None,
                       peak_since(res[p["ticker"]], p.get("entry_date"), px), trail, sl)
        H.append({"Stock": p["ticker"].replace(".NS", ""), "Entry": e, "Qty": p.get("quantity") or 1, "Stop": sl, "Target": tg,
                  "Notes": p.get("notes") or "", "Price": px, "P&L %": (px / e - 1) * 100 if px and e else None,
                  "Above stop %": (px / sl - 1) * 100 if px and sl else None, "Exit below": g["level"], "Exit guide": g["text"],
                  "Sector": (r.industry if r is not None and r.industry else "Unclassified"),
                  "Daily traded": float(r.traded) if r is not None and pd.notna(r.traded) else None,
                  "Status": status(px, sl, tg, r.scanner if r is not None else "", bool(r.below50) if r is not None else False,
                                   float(r.caution) if r is not None else 0),
                  "Technicals": r.technicals if r is not None else "—",
                  "Industry": str(r.ind_group).title() if r is not None and r.ind_group else "—"})
    Hall = pd.DataFrame(H, columns=["Stock", "Entry", "Qty", "Stop", "Target", "Notes", "Price", "P&L %", "Above stop %", "Status",
                                    "Exit below", "Exit guide", "Technicals", "Industry", "Sector", "Daily traded"])
    Hd = Hall.drop(columns=["Sector", "Daily traded"])
    need = Hd[~Hd.Status.str.startswith("✅") | Hd["Exit guide"].str.startswith("🔴")] if len(Hd) else Hd
    if len(need):
        st.warning("Needs a look: " + ", ".join(
            f"{r['Stock']} ({'exit rule hit' if str(r['Exit guide']).startswith('🔴') else r['Status']})" for _, r in need.iterrows()))
    if len(Hall):
        cap = st.session_state.get("book_capital") or None
        K = risk_summary(Hall, cap)
        if K:
            m = st.columns(5)
            m[0].metric("Invested", f"₹{K['invested']:,.0f}", f"{K['n']} holding(s)", delta_color="off")
            m[1].metric("At risk if every stop is hit", f"₹{K['risk']:,.0f}", f"{K['risk_pct']:.1f}% of capital", delta_color="off")
            m[2].metric("Largest position", K["largest"][0], f"{K['largest'][1]:.0f}% of capital", delta_color="off")
            m[3].metric("Top industry", str(K["by_industry"].index[0])[:22], f"{K['by_industry'].iloc[0]:.0f}% of invested", delta_color="off")
            m[4].number_input("Total capital ₹ (optional)", min_value=0.0, step=10000.0, key="book_capital",
                              help="Leave 0 to measure against the invested amount. Kept for this session only.")
            notes = []
            if K["risk_pct"] > 8:
                notes.append(f"open risk is {K['risk_pct']:.0f}% of capital — above 6–8% a bad week hurts")
            if K["largest"][1] > 25:
                notes.append(f"{K['largest'][0]} is {K['largest'][1]:.0f}% of capital")
            if K["by_industry"].iloc[0] > 40 and K["n"] >= 3:
                notes.append(f"{K['by_industry'].iloc[0]:.0f}% sits in {K['by_industry'].index[0]}")
            if K["no_stop"]:
                notes.append("no stop set for " + ", ".join(K["no_stop"]) + " (the exit level is used for the risk figure)")
            if K["thin"]:
                notes.append("position is more than 5% of a day's traded value in " + ", ".join(K["thin"]) + " — hard to exit quickly")
            if notes:
                st.caption("⚠️ " + "; ".join(notes) + ".")
    He = st.data_editor(Hd, key=f"book_h_{ver}", num_rows="dynamic", hide_index=True, use_container_width=True,
                        disabled=[c for c in Hd.columns if c not in EDIT_H], column_config={
        "Stock": st.column_config.TextColumn("Stock", help="NSE symbol, e.g. CUMMINSIND", required=True),
        "Entry": money("Entry"), "Stop": money("Stop"), "Target": money("Target"),
        "Qty": st.column_config.NumberColumn("Qty", min_value=1, step=1, format="%d"),
        "Notes": st.column_config.TextColumn("Notes", width="medium"),
        "Price": st.column_config.NumberColumn(format="₹%.2f"), "P&L %": st.column_config.NumberColumn(format="%+.1f%%"),
        "Above stop %": st.column_config.NumberColumn(format="%.1f%%"),
        "Exit below": st.column_config.NumberColumn(format="₹%.1f", help=f"Higher of the 50-day average and {trail * 100:.0f}% below the "
                                                    "highest close since entry — the live engine's exit, kept by the exit studies"),
        "Exit guide": st.column_config.TextColumn("Exit guide", width="medium")})
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
            px = _px(df, t)
            pos.add_position(t, name=names.get(t), status="active", entry_price=_num(r.get("Entry")) or px,
                             stop_loss=_num(r.get("Stop")), target=_num(r.get("Target")), quantity=int(_num(r.get("Qty")) or 1),
                             notes=str(r.get("Notes") or ""))
        for t, r in dh["update"]:
            orig = next(p["ticker"] for p in hold if res[p["ticker"]] == t or p["ticker"] == t)
            pos.update_position(orig, entry_price=_num(r.get("Entry")), stop_loss=_num(r.get("Stop")), target=_num(r.get("Target")),
                                quantity=int(_num(r.get("Qty")) or 1), notes=str(r.get("Notes") or ""))
        for t, r in dw["add"]:
            px = _px(df, t)
            pos.add_position(t, name=names.get(t), status="watching", entry_price=px, notes=str(r.get("Notes") or ""))
        for t, r in dw["update"]:
            orig = next(p["ticker"] for p in watch if res[p["ticker"]] == t or p["ticker"] == t)
            pos.update_position(orig, notes=str(r.get("Notes") or ""))
        for t in quick:
            pos.add_position(t, name=names.get(t), status="watching", entry_price=_px(df, t))
        for t in bought:
            orig = next((p["ticker"] for p in watch if res[p["ticker"]] == t or p["ticker"] == t), t)
            r = C.loc[t] if len(C) and t in C.index else None
            px = _px(df, t)
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
