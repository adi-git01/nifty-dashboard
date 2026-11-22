"""
Send the model's order ticket to Dhan (DhanHQ v2 orders API). This is the ONLY module that can
place an order, and it is off until you switch it on:

    [dhan]
    client_id      = "..."        # already there for holdings sync
    orders_enabled = true         # the switch; without it every call here refuses
    order_proxy    = "http://user:pass@host:port"   # optional: a fixed-IP exit, see below

Why a fixed IP: SEBI's retail algo rules make brokers accept API orders only from an IP address
registered with them (Dhan: web.dhan.co -> Profile -> DhanHQ Trading APIs -> Add IP; it can be changed
once in 7 days). Streamlit Cloud has no fixed address, so orders sent from the hosted dashboard are
rejected with an "invalid IP" error unless they leave through `order_proxy`, or the dashboard runs
on a machine with a fixed address. Holdings sync is not affected.

Guard rails, checked here whatever the caller does:
    - delivery (CNC) limit orders on NSE only, valid for the day; no market orders, no intraday, no F&O
    - one order per order id (the id is model date + stock + side), so a double tap cannot double an order
    - a rupee cap per order and per day, and a cap on the number of orders in a day
    - a sale is never larger than what the model book holds
Outside market hours an order goes in as an after-market order for the next open.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from utils.broker_sync import DHAN_API, TIMEOUT, BrokerError, _http, _json

class UncertainOrder(BrokerError):
    """No clear answer from Dhan: the order may or may not exist. Never resend it blindly."""


SCRIP_MASTER = "https://images.dhan.co/api-data/api-scrip-master.csv"
MAX_ORDERS_DAY = 40
IST = timezone(timedelta(hours=5, minutes=30))


HOLIDAYS_FILE = "data/nse/holidays.csv"


def _load_holidays(path: str = HOLIDAYS_FILE, _cache: Dict[str, Any] = {}) -> set:
    try:
        mt = os.path.getmtime(path) if os.path.exists(path) else None
    except Exception:
        mt = None
    if _cache.get("path") == path and _cache.get("mtime") == mt:
        return _cache.get("dates", set())
    dates = set()
    if mt is not None:
        try:
            df = pd.read_csv(path)
            col = "date" if "date" in df.columns else df.columns[0]
            parsed = pd.to_datetime(df[col], errors="coerce").dropna()
            dates = {d.date() for d in parsed}
        except Exception:
            dates = set()
    _cache.clear()
    _cache.update(path=path, mtime=mt, dates=dates)
    return dates


def is_nse_holiday(now: Optional[datetime] = None) -> bool:
    """True when the IST calendar date is listed in data/nse/holidays.csv."""
    dt = (now or datetime.now(IST)).astimezone(IST).date()
    return dt in _load_holidays()


def is_pre_open(now: Optional[datetime] = None) -> bool:
    """True during NSE equity pre-open / buffer window (09:00–09:15 IST on trading days)."""
    dt = (now or datetime.now(IST)).astimezone(IST)
    if dt.weekday() >= 5 or is_nse_holiday(dt):
        return False
    return (9, 0) <= (dt.hour, dt.minute) < (9, 15)


def settings(secrets: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """{'enabled', 'client_id', 'proxy'} from Streamlit secrets."""
    if secrets is None:
        try:
            import streamlit as st
            secrets = dict(st.secrets["dhan"]) if "dhan" in st.secrets else {}
        except Exception:
            secrets = {}
    return dict(enabled=str(secrets.get("orders_enabled", "")).strip().lower() in ("true", "1", "yes"),
                client_id=str(secrets.get("client_id") or "").strip(), proxy=str(secrets.get("order_proxy") or "").strip())


def market_open(now: Optional[datetime] = None) -> bool:
    now = (now or datetime.now(IST)).astimezone(IST)
    if now.weekday() >= 5 or is_nse_holiday(now):
        return False
    return (9, 15) <= (now.hour, now.minute) < (15, 30)


def parse_master(df: pd.DataFrame) -> Dict[str, str]:
    """NSE cash-equity trading symbol -> Dhan security id, from the scrip master (column names matched loosely)."""
    col = lambda *keys: next((c for c in df.columns if all(k in c.upper() for k in keys)), None)
    exch, seg, sid, sym, ser = (col("EXCH", "ID") or col("EXCH"), col("SEGMENT"), col("SECURITY", "ID"),
                                col("TRADING", "SYMBOL"), col("SERIES"))
    if not (sid and sym):
        raise BrokerError("Dhan's instrument list has an unexpected layout; orders cannot be sent until this is updated.")
    d = df
    if exch:
        d = d[d[exch].astype(str).str.upper().str.startswith("NSE")]
    if seg:
        d = d[d[seg].astype(str).str.upper().isin(["E", "EQ", "NSE_EQ", "EQUITY"])]
    if ser:
        d = d[d[ser].astype(str).str.upper().isin(["EQ", "BE"])]
    return dict(zip(d[sym].astype(str).str.upper().str.strip(), d[sid].astype(str).str.strip()))


def security_ids() -> Dict[str, str]:
    """Downloaded once a day (the file is large)."""
    try:
        import streamlit as st

        @st.cache_data(ttl=24 * 3600, show_spinner="Loading Dhan's instrument list…", max_entries=1)
        def _load():
            return parse_master(pd.read_csv(SCRIP_MASTER, low_memory=False))
        return _load()
    except BrokerError:
        raise
    except Exception as e:
        raise BrokerError(f"Could not load Dhan's instrument list ({type(e).__name__}).")


def check(order: Dict[str, Any], held: Dict[str, float], log: List[Dict[str, Any]], max_order: float, max_day: float,
          today: Optional[str] = None) -> Optional[str]:
    """A sentence saying why this order must not be sent, or None when it may."""
    today = today or datetime.now(IST).strftime("%Y-%m-%d")
    q, lim = int(order.get("qty") or 0), float(order.get("limit") or 0)
    if order.get("side") not in ("BUY", "SELL") or q <= 0 or lim <= 0:
        return "not a valid limit order"
    prior = next((r for r in reversed(log) if r.get("id") == order["id"]), None)      # the latest state of this order
    if prior and prior.get("status") in ("SENDING", "SENT", "FILLED"):
        return "already sent"
    if q * lim > max_order * 1.0001:
        return f"₹{q * lim:,.0f} is over the per-order cap of ₹{max_order:,.0f}"
    sent = [r for r in log if str(r.get("ts", ""))[:10] == today and r.get("status") in ("SENDING", "SENT", "FILLED")
            and str(r.get("mode", "")).lower() == "dhan"]
    if len(sent) >= MAX_ORDERS_DAY:
        return f"{MAX_ORDERS_DAY} orders already sent today"
    bought = sum(float(r.get("qty") or 0) * float(r.get("limit") or 0) for r in sent if r.get("side") == "BUY")
    if order["side"] == "BUY" and bought + q * lim > max_day * 1.0001:
        return f"would take today's buying over the cap of ₹{max_day:,.0f}"
    if order["side"] == "SELL" and q > float(held.get(order["ticker"], 0)) + 1e-9:
        return "more shares than the model book holds"
    return None


def body(order: Dict[str, Any], client_id: str, security_id: str, amo: bool) -> Dict[str, Any]:
    return {"dhanClientId": client_id, "correlationId": correlation_id(order["id"]), "transactionType": order["side"],
            "exchangeSegment": "NSE_EQ", "productType": "CNC", "orderType": "LIMIT", "validity": "DAY",
            "securityId": str(security_id), "quantity": int(order["qty"]), "disclosedQuantity": 0,
            "price": float(order["limit"]), "triggerPrice": 0, "afterMarketOrder": bool(amo), "amoTime": "OPEN"}


def correlation_id(order_id: str) -> str:
    """Dhan allows 25 characters; the order id (2026-11-13-EIMCOELECO-SELL) can be longer or hold '&' (M&M)."""
    parts = str(order_id).split("-")
    date = "".join(parts[:3]) if len(parts) >= 5 else ""
    sym, side = ("".join(parts[3:-1]), parts[-1]) if len(parts) >= 5 else (str(order_id), "")   # BAJAJ-AUTO -> BAJAJAUTO
    sym = re.sub(r"[^A-Za-z0-9]", "", sym)[:13]
    return f"{date}-{sym}-{side[:1]}"[:25] if date else re.sub(r"[^A-Za-z0-9-]", "", str(order_id))[:25]


def _proxies(cfg: Dict[str, Any]) -> Optional[Dict[str, str]]:
    return {"http": cfg["proxy"], "https": cfg["proxy"]} if cfg.get("proxy") else None


def _error(d: Any, code: int) -> str:
    msg = ""
    if isinstance(d, dict):
        msg = str(d.get("errorMessage") or d.get("message") or d.get("omsErrorDescription") or d.get("errorCode") or "")
    if re.search(r"\bip\b|whitelist", msg, re.I):            # not "scrip"
        msg += (" — Dhan only accepts orders from a registered IP address. Register this server's address on web.dhan.co "
                "(Profile → DhanHQ Trading APIs → Add IP), or set [dhan] order_proxy to a fixed-IP exit.")
    return msg or f"HTTP {code}"


def sending_ip(cfg: Optional[Dict[str, Any]] = None) -> str:
    """The address Dhan will see for an order sent from here (through order_proxy when one is set). Sends nothing about you."""
    cfg = cfg or settings()
    try:
        r = _http().get("https://api.ipify.org", timeout=TIMEOUT, proxies=_proxies(cfg))
    except Exception as e:
        raise BrokerError("The fixed-IP exit did not answer: check that its server is running, its port is open and the "
                          f"order_proxy line is exact ({type(e).__name__})." if cfg.get("proxy") else
                          f"Could not find this server's address ({type(e).__name__}).")
    if r.status_code == 407:
        raise BrokerError("The fixed-IP exit refused the password in order_proxy.")
    if r.status_code != 200:
        raise BrokerError(f"The address check failed (HTTP {r.status_code}).")
    return r.text.strip()


def place(token: str, order: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None, ids: Optional[Dict[str, str]] = None,
          now: Optional[datetime] = None) -> Dict[str, Any]:
    """Send one order. Returns {'order_id', 'status'}; raises BrokerError with a message safe to show."""
    cfg = cfg or settings()
    if not cfg.get("enabled"):
        raise BrokerError("Order placement is switched off (set orders_enabled = true under [dhan] in the app's secrets).")
    if not cfg.get("client_id") or not token:
        raise BrokerError("Dhan client id or the day's login token is missing: sync Dhan on the Import tab first.")
    sym = str(order["ticker"]).replace(".NS", "").upper()
    sid = (ids if ids is not None else security_ids()).get(sym)
    if not sid:
        raise BrokerError(f"{sym} was not found in Dhan's instrument list.")
    if is_pre_open(now):
        raise BrokerError("NSE pre-open session (09:00–09:15 IST): AMO window is closed; retry after 09:15 IST.")
    try:
        r = _http().post(f"{DHAN_API}/orders", json=body(order, cfg["client_id"], sid, not market_open(now)), timeout=TIMEOUT,
                         headers={"access-token": token, "Content-Type": "application/json", "Accept": "application/json"},
                         proxies=_proxies(cfg))
    except Exception as e:                                   # timeout / dropped connection: Dhan may have taken it
        raise UncertainOrder(f"No reply from Dhan ({type(e).__name__}). The order may have been placed: check the Dhan "
                             "order book before doing anything else; it will not be resent from here.")
    try:
        d = _json(r)
    except BrokerError:
        if r.status_code in (200, 201, 202):
            raise UncertainOrder("Dhan answered but the reply was unreadable; check the Dhan order book.")
        raise
    if r.status_code not in (200, 201, 202) or not isinstance(d, dict) or not d.get("orderId"):
        raise BrokerError(f"Dhan refused the order: {_error(d, r.status_code)}")
    return dict(order_id=str(d["orderId"]), status=str(d.get("orderStatus") or "PENDING").upper())


def status(token: str, order_id: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """{'status', 'filled', 'price', 'message'} for one order (status: TRADED, PENDING, TRANSIT, PART_TRADED, REJECTED, CANCELLED, EXPIRED)."""
    cfg = cfg or settings()
    try:
        r = _http().get(f"{DHAN_API}/orders/{order_id}", headers={"access-token": token, "Accept": "application/json"}, timeout=TIMEOUT,
                        proxies=_proxies(cfg))
    except requests.RequestException as e:
        raise BrokerError(f"Could not check order status with Dhan ({type(e).__name__}).") from e
    d = _json(r)
    if isinstance(d, list):
        d = d[0] if d else {}
    if r.status_code != 200 or not isinstance(d, dict) or not d.get("orderStatus"):
        raise BrokerError(f"Dhan did not return the order: {_error(d, r.status_code)}")
    g = lambda *ks: next((d[k] for k in ks if d.get(k) not in (None, "")), 0)
    return dict(status=str(d["orderStatus"]).upper(), filled=float(g("filledQty", "filled_qty", "tradedQuantity") or 0),
                price=float(g("averageTradedPrice", "tradedPrice", "price") or 0), message=str(d.get("omsErrorDescription") or ""))


def cancel(token: str, order_id: str, cfg: Optional[Dict[str, Any]] = None) -> bool:
    cfg = cfg or settings()
    try:
        r = _http().delete(f"{DHAN_API}/orders/{order_id}", headers={"access-token": token, "Accept": "application/json"}, timeout=TIMEOUT,
                           proxies=_proxies(cfg))
    except requests.RequestException as e:
        raise BrokerError(f"Could not cancel order with Dhan ({type(e).__name__}).") from e
    return r.status_code in (200, 202)
