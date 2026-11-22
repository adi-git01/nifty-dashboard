"""
🎫 Model Orders tab: the order ticket that keeps your real-money copy of the model portfolio in line
with the model, your confirmation, and what happened to each order (utils/model_book.py).

Two ways to confirm a ticket:
  - "I'll place them myself": the ticket is a checklist for the Dhan app; you then record the fills here.
  - "Send to Dhan": the dashboard places the ticked orders (utils/dhan_orders.py) after you type CONFIRM.
    Off until orders_enabled is set in the secrets, and Dhan must know the sending IP address.
Either way a fill updates the model book, your own book (so a later Dhan sync sees nothing new) and the
trade journal, tagged "model", with the day's readings and the gap to the model's price.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from utils import model_book as mb, positions as pos

TAG = "model"
OPEN_STATES = ("PENDING", "TRANSIT", "PART_TRADED")


def _prices(df: pd.DataFrame) -> Dict[str, float]:
    return pd.to_numeric(df.drop_duplicates("ticker").set_index("ticker")["currentPrice"], errors="coerce").dropna().to_dict()


def fill(df: pd.DataFrame, order: Dict[str, Any], qty: float, price: float, mode: str) -> None:
    """One fill: model book, personal book, journal, order log."""
    from utils import book_view as bv
    known = bv._known(df)[0]
    t, qty, price = order["ticker"], float(qty), float(price)
    before = pos.get_all_positions()
    p = next((x for x in before if x.get("status") == "active" and (pos.resolve_ticker(x["ticker"], known) or x["ticker"]) == t), None)
    res = mb.apply_fill(order, qty, price)
    qty = res["filled"] if order["side"] == "SELL" else qty
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    if order["side"] == "BUY":
        if p:
            q0, a0 = float(p.get("quantity") or 0), float(p.get("entry_price") or price)
            f = dict(quantity=int(q0 + qty), entry_price=round((q0 * a0 + qty * price) / (q0 + qty), 2))
            if p.get("lots"):
                lots = dict(p["lots"])
                d = lots.get("Dhan") or dict(qty=0, avg=price)
                dq = float(d["qty"]) + qty
                lots["Dhan"] = dict(qty=int(dq), avg=round((float(d["qty"]) * float(d.get("avg") or price) + qty * price) / dq, 2))
                f["lots"] = lots
            pos.update_position(p["ticker"], **f)
            act = "ADD"
        else:
            pos.add_position(t, status="active", entry_price=price, quantity=int(qty), stop_loss=order.get("stop") or None,
                             notes="OptComp model")
            pos.update_position(t, entry_date=today, exit_price=None, exit_date=None, source="Model orders",
                                lots={"Dhan": dict(qty=int(qty), avg=round(price, 2))})
            act = "BUY"
    else:
        act = "SELL"
        if p:
            left = float(p.get("quantity") or 0) - qty
            if left <= 0:
                pos.close_position(p["ticker"], price)
            else:
                f, act = dict(quantity=int(left)), "REDUCE"
                if (p.get("lots") or {}).get("Dhan"):
                    lots = dict(p["lots"])
                    lots["Dhan"] = {**lots["Dhan"], "qty": max(int(float(lots["Dhan"]["qty"]) - qty), 0)}
                    f["lots"] = {k: v for k, v in lots.items() if v["qty"] > 0}
                pos.update_position(p["ticker"], **f)
    ref = float(order.get("ref") or 0)
    slip = (price / ref - 1) * 100 * (1 if order["side"] == "BUY" else -1) if ref else None      # + = worse than the model's price
    bv.journal(df, [(act, t, qty, price, False)], f"Model · {mode}", before, known, notes={t: order.get("reason", "")}, tags=TAG,
               extra={t: dict(model_price=ref or None, slippage_pct=round(slip, 2) if slip is not None else None)})
    mb.log_order(order, "FILLED", fill_qty=qty, fill_price=price, slippage_pct=round(slip, 2) if slip is not None else None,
                 mode=order.get("mode") or ("dhan" if mode == "Dhan" else "manual"))   # keep the stored mode: the daily caps count "dhan"


def _setup(meta: Dict[str, Any]) -> bool:
    """First run: set the capital. Returns True once the model book exists."""
    if meta.get("started"):
        return True
    st.markdown("#### Start a real-money copy of the model portfolio")
    st.caption("Set aside an amount for it. The dashboard then proposes the orders that copy the model (OptComp V2.2) with that "
               "money: one equal slot per stock, whole shares, limit orders. Nothing is sent anywhere until you confirm a ticket.")
    cap = st.number_input("Capital for the model book (₹)", min_value=10000.0, value=200000.0, step=10000.0, key="mo_cap")
    if st.button("Start the model book", type="primary", key="mo_start"):
        mb.save_meta(capital=float(cap), cash=float(cap))
        st.rerun()
    return False


def _ticket_table(orders: List[Dict[str, Any]], ver: int) -> pd.DataFrame:
    T = pd.DataFrame([{"Place": True, "Side": o["side"], "Stock": o["ticker"].replace(".NS", ""), "Qty": o["qty"], "Limit": o["limit"],
                       "Value": o["value"], "Last close": o["ref"], "Model stop": o.get("stop"), "Why": o["reason"], "Note": o["note"]}
                      for o in orders])
    return st.data_editor(T, key=f"mo_ticket_{ver}", hide_index=True, use_container_width=True,
                          disabled=[c for c in T.columns if c not in ("Place", "Qty", "Limit")], column_config={
        "Place": st.column_config.CheckboxColumn("Place", help="Untick to leave an order out"),
        "Qty": st.column_config.NumberColumn(min_value=1, step=1, format="%d"),
        "Limit": st.column_config.NumberColumn(format="₹%.2f", min_value=0.05, help="Limit price: the last close plus (buy) or minus (sell) the buffer"),
        "Value": st.column_config.NumberColumn(format="₹%.0f"), "Last close": st.column_config.NumberColumn(format="₹%.2f"),
        "Model stop": st.column_config.NumberColumn(format="₹%.2f", help="The model's trailing stop today. The model sells on a close below "
                                                    "it or below the 50-day average; that sale arrives here as a SELL order")})


def _chosen(orders: List[Dict[str, Any]], E: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    for o, (_, r) in zip(orders, E.iterrows()):
        if r["Place"] is True and pd.notna(r["Qty"]) and pd.notna(r["Limit"]) and r["Qty"] > 0:
            q, lim = int(r["Qty"]), mb.tick(r["Limit"])
            out.append({**o, "qty": q, "limit": lim, "value": round(q * lim, 2)})
    return out


def _awaiting(df: pd.DataFrame, log: List[Dict[str, Any]], ver: int) -> None:
    """Orders placed and not yet settled: fetch fills from Dhan, or type them in."""
    from utils import dhan_orders as do
    from utils.broker_sync import BrokerError
    wait = [r for r in log if r.get("status") in ("SENT", "SENDING")]
    if not wait:
        return
    st.markdown("#### ⏳ Placed, waiting for fills")
    token = st.session_state.get("dhan_token")
    auto = [r for r in wait if r.get("mode") == "dhan" and r.get("dhan_order_id")]
    if auto:
        if st.button(f"🔄 Check {len(auto)} order(s) with Dhan", key="mo_check", disabled=not token,
                     help=None if token else "Sync Dhan on the Import tab first (the day's login)"):
            done = 0
            for r in auto:
                try:
                    s = do.status(token, r["dhan_order_id"])
                except BrokerError as e:
                    st.error(f"{r['ticker'].replace('.NS', '')}: {e}")
                    continue
                if s["status"] == "TRADED" or (s["status"] in ("CANCELLED", "EXPIRED") and s["filled"] > 0):
                    fill(df, r, s["filled"] or r["qty"], s["price"] or r["limit"], "Dhan")
                    done += 1
                elif s["status"] in ("REJECTED", "CANCELLED", "EXPIRED"):
                    mb.log_order(r, "REJECTED" if s["status"] == "REJECTED" else "CANCELLED", message=s["message"] or s["status"].lower())
                    done += 1
            st.session_state["mo_ver"] = ver + 1
            st.toast(f"{done} order(s) settled" if done else "Still open at Dhan", icon="🔄")
            st.rerun()
    W = pd.DataFrame([{"Side": r["side"], "Stock": r["ticker"].replace(".NS", ""), "Qty": r["qty"], "Limit": r["limit"],
                       "Outcome": "waiting", "Filled qty": float(r["qty"]), "Fill price": float(r["limit"]),
                       "Via": "Dhan" if r.get("mode") == "dhan" else "by hand", "Placed": str(r.get("ts", ""))[:16].replace("T", " ")}
                      for r in wait])
    E = st.data_editor(W, key=f"mo_wait_{ver}", hide_index=True, use_container_width=True,
                       disabled=[c for c in W.columns if c not in ("Outcome", "Filled qty", "Fill price")], column_config={
        "Outcome": st.column_config.SelectboxColumn("Outcome", options=["waiting", "filled", "not filled"], required=True),
        "Filled qty": st.column_config.NumberColumn(min_value=0.0, step=1.0, format="%.0f"),
        "Fill price": st.column_config.NumberColumn(format="₹%.2f", min_value=0.0, help="Your average traded price from the Dhan order book"),
        "Limit": st.column_config.NumberColumn(format="₹%.2f")})
    todo = [(r, e) for r, (_, e) in zip(wait, E.iterrows()) if e["Outcome"] != "waiting"]
    if st.button(f"💾 Record {len(todo)} outcome(s)" if todo else "💾 Record outcomes", type="primary", disabled=not todo, key="mo_record"):
        for r, e in todo:
            if e["Outcome"] == "filled" and e["Filled qty"] and e["Fill price"]:
                fill(df, r, float(e["Filled qty"]), float(e["Fill price"]), "Dhan" if r.get("mode") == "dhan" else "by hand")
            else:
                mb.log_order(r, "CANCELLED", message="not filled")
        st.session_state["mo_ver"] = ver + 1
        st.rerun()
    st.caption("An order that did not fill is offered again on the next ticket while the model still holds (or has sold) the stock.")


def sebi_peak_margin_check(cash: float, sell_val: float, buy_val: float) -> Dict[str, Any]:
    """
    SEBI upfront/peak-margin check for simultaneous CNC SELL + BUY execution.
    Only 80% of same-day CNC sell proceeds are available immediately before settlement.
    """
    cash = float(cash or 0)
    sell_val = float(sell_val or 0)
    buy_val = float(buy_val or 0)
    usable_sell = 0.80 * sell_val
    available_for_buy = cash + usable_sell
    shortfall = max(0.0, buy_val - available_for_buy)
    return dict(
        has_both=bool(sell_val > 0 and buy_val > 0),
        usable_sell=usable_sell,
        available_for_buy=available_for_buy,
        shortfall=shortfall,
    )


def _send(orders: List[Dict[str, Any]], book: List[Dict[str, Any]], log: List[Dict[str, Any]], S: Dict[str, Any], meta: Dict[str, Any]) -> int:
    """Send the chosen orders to Dhan (SELLs strictly before BUYs). Returns how many were accepted."""
    from utils import dhan_orders as do
    from utils.broker_sync import BrokerError
    cfg, token = do.settings(), st.session_state.get("dhan_token")
    held = {b["ticker"]: float(b["qty"]) for b in book}
    max_order = float(meta.get("max_order") or 0) or S["slot"] * mb.OVER_SLOT
    ok = 0
    try:
        ids = do.security_ids()
    except BrokerError as e:
        st.error(str(e))
        return 0
    ordered = sorted(orders, key=lambda o: 0 if o.get("side") == "SELL" else 1)
    for o in ordered:
        why = do.check(o, held, log, max_order, float(meta.get("capital") or S["equity"]))
        if why:
            st.warning(f"{o['side']} {o['ticker'].replace('.NS', '')} not sent: {why}.")
            continue
        log.append(mb.log_order(o, "SENDING", mode="dhan"))  # recorded before the call: a rerun cannot send it twice
        try:
            r = do.place(token, o, cfg, ids)
            log.append(mb.log_order(o, "SENT", mode="dhan", dhan_order_id=r["order_id"], broker_status=r["status"]))
            ok += 1
        except do.UncertainOrder as e:
            log.append(mb.log_order(o, "SENT", mode="dhan", dhan_order_id=None, message=str(e)))   # resolve by hand below
            st.error(f"{o['side']} {o['ticker'].replace('.NS', '')}: {e}")
            break                                            # stop the batch until you have looked
        except BrokerError as e:
            log.append(mb.log_order(o, "REJECTED", mode="dhan", message=str(e)[:200]))
            st.error(f"{o['side']} {o['ticker'].replace('.NS', '')}: {e}")
            if "registered IP" in str(e):
                break                                        # the same refusal would repeat for every order
        except Exception as e:
            log.append(mb.log_order(o, "SENT", mode="dhan", dhan_order_id=None, message=f"unexpected error: {type(e).__name__}"))
            st.error(f"{o['side']} {o['ticker'].replace('.NS', '')}: unexpected error ({type(e).__name__}). Check the Dhan "
                     "order book; the order is listed below as waiting until you record what happened.")
            break
    return ok


def render(df: pd.DataFrame):
    from utils import dhan_orders as do
    meta, book = mb.load()
    if not _setup(meta):
        return
    ver = st.session_state.setdefault("mo_ver", 0)
    snap, prices, log = mb.snapshot(), _prices(df), mb.LOG.load()
    T = mb.ticket(snap, meta, book, prices, mb.exit_reasons())
    S = T["summary"]
    if not S:
        st.warning(T["problem"])
        return
    m = st.columns(5)
    m[0].metric("Your model book", f"₹{S['equity']:,.0f}", f"{(S['equity'] / float(meta['capital']) - 1) * 100:+.1f}% on ₹{float(meta['capital']):,.0f}",
                delta_color="off")
    m[1].metric("Cash", f"₹{S['cash']:,.0f}", f"{S['cash'] / S['equity'] * 100:.0f}%" if S["equity"] else None, delta_color="off")
    m[2].metric("Holdings", f"{S['positions']} of {S['model_positions']}", "you / model", delta_color="off")
    m[3].metric("One slot", f"₹{S['slot']:,.0f}")
    m[4].metric("Model as of", S["model_date"], str(S.get("regime") or ""), delta_color="off")

    _awaiting(df, log, ver)

    # ------------------------------------------------------------------ the ticket
    st.markdown("#### 🎫 Order ticket")
    orders = mb.pending(T["orders"], log)
    if T["problem"]:
        st.warning(T["problem"])
    elif not orders:
        st.success("Nothing to do: your model book matches the model" + (" (orders already placed are above)." if any(
            r.get("status") == "SENT" for r in log) else "."))
    else:
        E = _ticket_table(orders, ver)
        chosen = _chosen(orders, E)
        try:                                                     # exchange surveillance on a buy (utils/nse_data.py)
            from utils.nse_data import surveillance_map, upcoming_events
            _sv, _ev = surveillance_map(), upcoming_events(5)
            _flag = [f"{o['ticker'].replace('.NS', '')} ({_sv[o['ticker']]})" for o in chosen if o["side"] == "BUY" and o["ticker"] in _sv]
            if _flag:
                st.warning("⚠️ Under exchange surveillance: " + ", ".join(_flag) + ". Margins are higher and the stock may trade in a "
                           "narrow band or trade-for-trade (no intraday netting). The model does not look at this: decide before sending.")
            _soon = [f"{o['ticker'].replace('.NS', '')} ({_ev[o['ticker']]})" for o in chosen if o["side"] == "BUY" and o["ticker"] in _ev]
            if _soon:
                st.caption("📅 Coming up for buys: " + ", ".join(_soon) + ".")
        except Exception:
            pass
        buy = sum(o["value"] for o in chosen if o["side"] == "BUY")
        sell = sum(o["value"] for o in chosen if o["side"] == "SELL")
        pm = sebi_peak_margin_check(S["cash"], sell, buy)
        st.caption(f"{len(chosen)} order(s): buy ₹{buy:,.0f}, sell ₹{sell:,.0f}; cash afterwards about ₹{S['cash'] + sell - buy:,.0f}. "
                   f"Limits are {float(meta.get('buffer') or 0.01) * 100:.1f}% through the last close ({S['model_date']}), so they fill at a "
                   "normal open and do not chase a gap. Delivery (CNC), valid for the day.")
        if S["cash"] + sell - buy < -1:
            st.error("The ticked buys cost more than the cash and sale proceeds. Untick or reduce some.")
        elif pm["has_both"]:
            if pm["shortfall"] > 1:
                st.warning(
                    f"⚠️ **SEBI 80% peak-margin rule:** Same-day CNC sell proceeds free only 80% upfront "
                    f"(usable now: ₹{pm['available_for_buy']:,.0f} vs buys ₹{buy:,.0f}, shortfall ₹{pm['shortfall']:,.0f}). "
                    "Send SELL orders first and wait for fills before sending the BUYs."
                )
            else:
                st.info("ℹ️ SELL orders are sequenced before BUY orders to respect SEBI 80% same-day peak-margin credit.")
        cfg = do.settings()
        live = cfg["enabled"] and bool(st.session_state.get("dhan_token"))
        if cfg.get("proxy", "").lower().startswith("http://"):
            st.warning("⚠️ `order_proxy` uses cleartext `http://`. Enable TLS on the proxy and use `https://` before live trading.")
        how = st.radio("How to place them", ["I'll place them myself in the Dhan app", "Send to Dhan from here"], key="mo_how",
                       horizontal=True, index=0)
        if how.startswith("I'll"):
            st.download_button("⬇️ Ticket (CSV)", pd.DataFrame(chosen)[["side", "ticker", "qty", "limit", "value"]].to_csv(index=False).encode()
                               if chosen else b"", "order_ticket.csv", "text/csv", key="mo_csv", disabled=not chosen)
            if st.button(f"✅ I have placed these {len(chosen)} order(s)", type="primary", disabled=not chosen, key="mo_manual"):
                for o in chosen:
                    mb.log_order(o, "SENT", mode="manual")
                st.session_state["mo_ver"] = ver + 1
                st.rerun()
        elif not live:
            st.info("Sending from here needs three things: **orders_enabled = true** under [dhan] in the app's secrets"
                    + (" ✅" if cfg["enabled"] else " ❌") + "; today's **Dhan login** (sync Dhan on the Import tab)"
                    + (" ✅" if st.session_state.get("dhan_token") else " ❌") + "; and the sending **IP address registered with Dhan** "
                    "(Streamlit Cloud has no fixed address, so this needs a fixed-IP exit in **order_proxy** or a host with a fixed address).")
        else:
            sells_only = [o for o in chosen if o["side"] == "SELL"]
            word = st.text_input(f"Type CONFIRM to send {len(chosen)} order(s) worth ₹{buy + sell:,.0f} to Dhan", key=f"mo_word_{ver}")
            c_send1, c_send2 = st.columns([2, 1])
            with c_send1:
                if st.button(f"🚀 Send {len(chosen)} order(s) to Dhan", type="primary", key="mo_send",
                             disabled=not chosen or word.strip().upper() != "CONFIRM" or S["cash"] + sell - buy < -1 or pm["shortfall"] > 1):
                    n = _send(chosen, book, log, S, meta)
                    if n:
                        st.session_state["mo_ver"] = ver + 1
                        st.toast(f"{n} order(s) accepted by Dhan", icon="🚀")
                        st.rerun()
            with c_send2:
                if pm["has_both"] and st.button(f"📤 Send {len(sells_only)} SELL(s) first", key="mo_send_sells",
                                               disabled=not sells_only or word.strip().upper() != "CONFIRM"):
                    n = _send(sells_only, book, log, S, meta)
                    if n:
                        st.session_state["mo_ver"] = ver + 1
                        st.toast(f"{n} SELL order(s) accepted by Dhan", icon="📤")
                        st.rerun()
    if T["skipped"]:
        st.caption("Left out: " + "; ".join(f"{x['side']} {x['ticker'].replace('.NS', '')} — {x['why']}" for x in T["skipped"]) + ".")

    # ------------------------------------------------------------------ holdings
    if book:
        st.markdown("#### 📦 Your model holdings")
        info = {p.get("Ticker"): p for p in (snap.get("portfolio") or [])}
        H = pd.DataFrame([{"Stock": b["ticker"].replace(".NS", ""), "Qty": float(b["qty"]), "Avg cost": float(b["avg"]),
                           "Price": prices.get(b["ticker"]), "P&L %": (prices[b["ticker"]] / float(b["avg"]) - 1) * 100
                           if prices.get(b["ticker"]) and float(b["avg"]) else None,
                           "Value": float(b["qty"]) * float(prices.get(b["ticker"]) or b["avg"]),
                           "Model stop": (info.get(b["ticker"]) or {}).get("Trail_Stop"),
                           "In the model": "yes" if b["ticker"] in (snap.get("holdings") or {}) else "no — sell on the ticket"} for b in book])
        st.dataframe(H, hide_index=True, use_container_width=True, column_config={
            "Avg cost": st.column_config.NumberColumn(format="₹%.2f"), "Price": st.column_config.NumberColumn(format="₹%.2f"),
            "P&L %": st.column_config.NumberColumn(format="%+.1f%%"), "Value": st.column_config.NumberColumn(format="₹%.0f"),
            "Model stop": st.column_config.NumberColumn(format="₹%.2f"), "Qty": st.column_config.NumberColumn(format="%.0f")})

    # ------------------------------------------------------------------ history and settings
    done = [r for r in log if r.get("status") in mb.DONE]
    if done:
        with st.expander(f"🧾 Order history ({len(done)})"):
            D = pd.DataFrame([{"Date": str(r.get("ts", ""))[:10], "Side": r["side"], "Stock": r["ticker"].replace(".NS", ""), "Qty": r.get("fill_qty") or r["qty"],
                               "Limit": r["limit"], "Fill": r.get("fill_price"), "Vs model %": r.get("slippage_pct"), "Status": r["status"],
                               "Via": {"dhan": "Dhan", "Dhan": "Dhan"}.get(r.get("mode"), "by hand"), "Message": r.get("message", "")}
                              for r in sorted(done, key=lambda r: str(r.get("ts")), reverse=True)])
            st.dataframe(D, hide_index=True, use_container_width=True, column_config={
                "Limit": st.column_config.NumberColumn(format="₹%.2f"), "Fill": st.column_config.NumberColumn(format="₹%.2f"),
                "Vs model %": st.column_config.NumberColumn(format="%+.2f%%", help="Your fill against the model's price. + means you paid more "
                                                            "(or received less) than the paper model assumes")})
            f = pd.to_numeric(D["Vs model %"], errors="coerce").dropna()
            if len(f):
                st.caption(f"Average gap to the model's price over {len(f)} fill(s): {f.mean():+.2f}%. This is the cost the paper "
                           "results do not include.")
    with st.expander("⚙️ Settings"):
        c = st.columns(3)
        paused = c[0].toggle("Pause (propose no orders)", value=bool(meta.get("paused")), key="mo_paused")
        buf = c[1].number_input("Limit buffer %", min_value=0.0, max_value=5.0, value=float(meta.get("buffer") or 0.01) * 100, step=0.25,
                                key="mo_buf", help="How far through the last close the limit is set")
        move = c[2].number_input("Add (+) or withdraw (−) cash ₹", value=0.0, step=10000.0, key=f"mo_move_{ver}")
        if st.button("Save settings", key="mo_set"):
            f = dict(paused=bool(paused), buffer=round(buf / 100, 4))
            if move:
                f.update(cash=round(float(meta["cash"]) + move, 2), capital=round(float(meta["capital"]) + move, 2))
            mb.save_meta(**f)
            st.session_state["mo_ver"] = ver + 1
            st.rerun()
        cfg = do.settings()
        st.markdown("**Sending orders to Dhan**")
        st.caption(("✅" if cfg["enabled"] else "❌") + " orders_enabled in secrets · " + ("✅" if cfg["proxy"] else "❌")
                   + " order_proxy (fixed-IP exit) in secrets · " + ("✅" if st.session_state.get("dhan_token") else "❌")
                   + " today's Dhan login. Setup: docs/model_orders_setup.md.")
        if st.button("🌐 Check the address Dhan will see", key="mo_ip"):
            from utils.broker_sync import BrokerError
            try:
                ip = do.sending_ip(cfg)
                st.success(f"Orders from here reach Dhan from **{ip}**. " + ("This is your fixed-IP exit: register it on web.dhan.co "
                           "(Profile → DhanHQ Trading APIs → Add IP) if you have not." if cfg["proxy"] else
                           "No order_proxy is set, so this is Streamlit's own address, which changes: Dhan will refuse orders from it."))
            except BrokerError as e:
                st.error(str(e))
        st.caption("The paper model is unchanged by anything here. Fills are also written to **My Book** and the **Journal** (tag "
                   "'model'), so a Dhan sync afterwards finds nothing new.")
