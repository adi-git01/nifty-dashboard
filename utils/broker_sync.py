"""
Read holdings straight from Zerodha (Kite Connect) and Dhan (DhanHQ v2). Read-only: the only
account endpoints called are the holdings endpoints and each broker's positions (today's trades). Nothing here can place or change an order
(orders, only when you switch them on, are in utils/dhan_orders.py).

Both brokers issue a login token that lasts about a day and needs you to log in on the broker's
own page, so a sync is one tap plus a login, not something the nightly job can do unattended.
Tokens are kept in the browser session only; they are never written to the repo or the store.

Zerodha (secrets: [kite] api_key, api_secret; redirect URL in the Kite developer console = this app's URL)
    kite_login_url(api_key)                     -> send the user here
    ... Zerodha redirects back with ?request_token=...&status=success
    kite_session(api_key, api_secret, request_token) -> access_token
    kite_holdings(api_key, access_token)        -> rows

Dhan, either
  (a) paste the 24-hour access token from web.dhan.co (My Profile -> Access DhanHQ APIs), or
  (b) secrets: [dhan] client_id, api_key, api_secret (redirect URL = this app's URL). api_key / api_secret are the
      "API key" and "API secret" Dhan shows when you create the key (its API calls them app_id / app_secret);
      the app NAME is only a label and is not used
    dhan_consent(client_id, app_id, app_secret) -> consentAppId;  dhan_login_url(consentAppId)
    ... Dhan redirects back with ?tokenId=...
    dhan_session(token_id, app_id, app_secret)  -> access_token
    dhan_holdings(access_token)                 -> rows

rows are the same shape utils/portfolio_import.py uses (symbol, name, qty, avg), so a sync goes
through the same combine -> reconcile -> preview -> apply path as a file upload.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import numpy as np
import pandas as pd

KITE_API = "https://api.kite.trade"
KITE_LOGIN = "https://kite.zerodha.com/connect/login"
DHAN_API = "https://api.dhan.co/v2"
DHAN_AUTH = "https://auth.dhan.co"
TIMEOUT = 20
EQUITY_EXCHANGES = {"NSE", "BSE", "ALL", "NSE_EQ", "BSE_EQ", ""}


class BrokerError(RuntimeError):
    """A broker call failed; the message is safe to show (no secrets)."""


def _http():
    import requests
    return requests


def _json(r) -> Any:
    try:
        return r.json()
    except Exception:
        raise BrokerError(f"The broker returned an unreadable reply (HTTP {r.status_code}).")


def _rows(items: List[Dict[str, Any]]) -> pd.DataFrame:
    from utils.portfolio_import import clean_symbol
    out = pd.DataFrame(items, columns=["symbol", "name", "qty", "avg", "buy_px", "sell_px"])
    if out.empty:
        return out
    out["symbol"] = out["symbol"].map(clean_symbol)
    out["qty"] = pd.to_numeric(out["qty"], errors="coerce")
    for c in ("avg", "buy_px", "sell_px"):
        out[c] = pd.to_numeric(out[c], errors="coerce").replace(0, np.nan)
    # a row with no shares left is kept when the broker says it was sold today: that is a sale, not a gap
    return out[(out.qty > 0) | ((out.qty == 0) & out.sell_px.notna())].reset_index(drop=True)


# ------------------------------------------------------------------ Zerodha
def kite_login_url(api_key: str) -> str:
    return f"{KITE_LOGIN}?v=3&api_key={quote(api_key)}"


def kite_checksum(api_key: str, request_token: str, api_secret: str) -> str:
    return hashlib.sha256(f"{api_key}{request_token}{api_secret}".encode()).hexdigest()


def kite_session(api_key: str, api_secret: str, request_token: str) -> str:
    """Exchange the one-time request_token from the redirect for the day's access token."""
    r = _http().post(f"{KITE_API}/session/token", headers={"X-Kite-Version": "3"}, timeout=TIMEOUT,
                     data={"api_key": api_key, "request_token": request_token,
                           "checksum": kite_checksum(api_key, request_token, api_secret)})
    d = _json(r)
    tok = (d.get("data") or {}).get("access_token") if isinstance(d, dict) else None
    if r.status_code != 200 or not tok:
        raise BrokerError(f"Zerodha login was not accepted: {(d or {}).get('message', f'HTTP {r.status_code}')}. "
                          "The login link is single-use and expires in a few minutes; try Sync again.")
    return tok


def kite_holdings(api_key: str, access_token: str) -> pd.DataFrame:
    r = _http().get(f"{KITE_API}/portfolio/holdings", timeout=TIMEOUT,
                    headers={"X-Kite-Version": "3", "Authorization": f"token {api_key}:{access_token}"})
    d = _json(r)
    if r.status_code != 200 or not isinstance(d, dict) or d.get("status") != "success":
        raise BrokerError(f"Zerodha did not return holdings: {(d or {}).get('message', f'HTTP {r.status_code}')}. "
                          "The day's login may have expired; press Sync again.")
    return _rows(kite_merge(d.get("data") or [], kite_day_positions(api_key, access_token)))


def kite_day_positions(api_key: str, access_token: str) -> List[Dict[str, Any]]:
    """Today's trades (read-only). Holdings alone miss today's buys and say nothing of the price a sale got."""
    try:
        r = _http().get(f"{KITE_API}/portfolio/positions", timeout=TIMEOUT,
                        headers={"X-Kite-Version": "3", "Authorization": f"token {api_key}:{access_token}"})
        d = _json(r)
        return list(((d or {}).get("data") or {}).get("day") or []) if r.status_code == 200 else []
    except Exception:
        return []                                            # holdings still sync without it


def kite_merge(holdings: List[Dict[str, Any]], day: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Holdings plus today's delivery trades, as rows of symbol / qty / avg / buy_px / sell_px.
    Zerodha's holdings already have today's sales taken out (quantity falls, used_quantity rises) but do
    not have today's purchases, which sit in positions until the next day. So:
        qty = settled + T1 + pledged + bought today - sold today out of today's own purchases
    A stock sold out completely today stays as a row with qty 0 and its sale price.
    """
    f = lambda h, k: float(h.get(k) or 0)
    out: Dict[str, Dict[str, Any]] = {}
    for h in holdings:
        if str(h.get("exchange", "")).upper() not in EQUITY_EXCHANGES:
            continue
        sym = str(h.get("tradingsymbol") or "")
        q = sum(f(h, k) for k in ("quantity", "t1_quantity", "collateral_quantity"))
        o = out.setdefault(sym, dict(symbol=sym, name="", qty=0.0, cost=0.0, used=0.0, buy_px=None, sell_px=None))
        o["qty"] += q
        o["cost"] += q * f(h, "average_price")
        o["used"] += f(h, "used_quantity")
    for p in day:
        if str(p.get("product", "")).upper() != "CNC" or str(p.get("exchange", "")).upper() not in EQUITY_EXCHANGES:
            continue
        sym = str(p.get("tradingsymbol") or "")
        b, sl = f(p, "day_buy_quantity"), f(p, "day_sell_quantity")
        if not (b or sl):
            continue
        o = out.setdefault(sym, dict(symbol=sym, name="", qty=0.0, cost=0.0, used=0.0, buy_px=None, sell_px=None))
        add = max(b - max(sl - o["used"], 0.0), 0.0)          # sales beyond what came out of holdings were of today's purchase
        o["qty"] += add
        o["cost"] += add * f(p, "day_buy_price")
        if b:
            o["buy_px"] = f(p, "day_buy_price")
        if sl:
            o["sell_px"] = f(p, "day_sell_price")
    return [dict(symbol=o["symbol"], name="", qty=o["qty"], avg=(o["cost"] / o["qty"]) if o["qty"] else None,
                 buy_px=o["buy_px"], sell_px=o["sell_px"]) for o in out.values()
            if o["qty"] > 0 or o["used"] > 0]                # nothing left and nothing sold out of holdings: bought and sold today


# ------------------------------------------------------------------ Dhan
def dhan_holdings(access_token: str) -> pd.DataFrame:
    r = _http().get(f"{DHAN_API}/holdings", headers={"access-token": access_token, "Accept": "application/json"}, timeout=TIMEOUT)
    d = _json(r)
    if isinstance(d, dict):                      # an error object, or "no holdings"
        msg = str(d.get("errorMessage") or d.get("message") or d.get("errorCode") or "")
        if "no holding" in msg.lower() or str(d.get("errorCode", "")).upper() == "DH-1111":
            return _rows([])
        raise BrokerError(f"Dhan did not return holdings: {msg or f'HTTP {r.status_code}'}. Access tokens last 24 hours; "
                          "generate a new one or press Sync again.")
    if r.status_code != 200 or not isinstance(d, list):
        raise BrokerError(f"Dhan did not return holdings (HTTP {r.status_code}).")
    return _rows(dhan_merge([h for h in d if str(h.get("exchange", "")).upper() in EQUITY_EXCHANGES],
                            dhan_day_positions(access_token)))


def dhan_day_positions(access_token: str) -> List[Dict[str, Any]]:
    """Today's delivery (CNC) trades, read-only. Empty on any failure: holdings still sync without them."""
    try:
        r = _http().get(f"{DHAN_API}/positions", headers={"access-token": access_token, "Accept": "application/json"},
                        timeout=TIMEOUT)
        d = _json(r)
        return [p for p in d if str(p.get("productType", "")).upper() == "CNC"] if r.status_code == 200 and isinstance(d, list) else []
    except Exception:
        return []


def dhan_merge(holdings: List[Dict[str, Any]], day: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Holdings plus today's delivery trades, as rows of symbol / qty / avg / buy_px / sell_px.
    Like Zerodha, a purchase made today is not in Dhan's holdings until the next day, so a model order bought and
    synced the same day read as "sold". A sale made today stays in the holding's total until settlement, while its
    available quantity drops; when that is so the sale is taken off here (otherwise the total is already net).
    """
    g = lambda h, *ks: next((h[k] for k in ks if h.get(k) not in (None, "")), None)
    f = lambda h, *ks: float(g(h, *ks) or 0)
    out: Dict[str, Dict[str, Any]] = {}
    for h in holdings:
        sym = str(g(h, "tradingSymbol", "tradingsymbol", "symbol") or "")
        o = out.setdefault(sym, dict(symbol=sym, total=0.0, avail=None, cost=0.0, buy_px=None, sell_px=None, b=0.0, s=0.0))
        q, a = f(h, "totalQty", "totalQuantity", "quantity"), f(h, "avgCostPrice", "avgPrice", "averagePrice")
        o["total"] += q
        o["cost"] += q * a
        if g(h, "availableQty") is not None:
            o["avail"] = (o["avail"] or 0.0) + f(h, "availableQty") + f(h, "collateralQty")
    for p in day:
        sym = str(g(p, "tradingSymbol", "tradingsymbol") or "")
        b, sl = f(p, "dayBuyQty"), f(p, "daySellQty")
        if not (b or sl):
            continue
        o = out.setdefault(sym, dict(symbol=sym, total=0.0, avail=None, cost=0.0, buy_px=None, sell_px=None, b=0.0, s=0.0))
        o["b"] += b
        o["s"] += sl
        if b:
            o["buy_px"] = f(p, "dayBuyValue") / b if f(p, "dayBuyValue") else f(p, "buyAvg")
        if sl:
            o["sell_px"] = f(p, "daySellValue") / sl if f(p, "daySellValue") else f(p, "sellAvg")
    rows = []
    for o in out.values():
        used = min(o["s"], o["total"])                              # sold out of holdings today
        pending = o["avail"] is not None and o["avail"] <= o["total"] - used + 0.5
        base = o["total"] - (used if pending else 0.0)
        add = max(o["b"] - max(o["s"] - o["total"], 0.0), 0.0)     # today's buys not sold again today
        qty = base + add
        avg = ((o["cost"] / o["total"] if o["total"] else 0.0) * base + (o["buy_px"] or 0.0) * add) / qty if qty else None
        if qty > 0 or used > 0:
            rows.append(dict(symbol=o["symbol"], name="", qty=qty, avg=avg, buy_px=o["buy_px"], sell_px=o["sell_px"]))
    return rows


def dhan_consent(client_id: str, app_id: str, app_secret: str) -> str:
    """Step 1 of the API-key login: returns a consentAppId for dhan_login_url."""
    r = _http().post(f"{DHAN_AUTH}/app/generate-consent", params={"client_id": client_id}, timeout=TIMEOUT,
                     headers={"app_id": app_id, "app_secret": app_secret})
    d = _json(r)
    cid = d.get("consentAppId") if isinstance(d, dict) else None
    if not cid:
        why = (d.get("errorMessage") or d.get("message") or d.get("error") or d.get("status")) if isinstance(d, dict) else None
        raise BrokerError(f"Dhan did not start a login session: {why or f'HTTP {r.status_code}'}. Check that api_key is the API KEY "
                          "(not the app name), api_secret is the API secret, client_id is your Dhan client ID, and the key is active.")
    return cid


def dhan_login_url(consent_app_id: str) -> str:
    return f"{DHAN_AUTH}/login/consentApp-login?consentAppId={quote(consent_app_id)}"


def dhan_session(token_id: str, app_id: str, app_secret: str) -> str:
    """Step 3: exchange the tokenId from the redirect for a 24-hour access token."""
    r = _http().post(f"{DHAN_AUTH}/app/consumeApp-consent", params={"tokenId": token_id}, timeout=TIMEOUT,
                     headers={"app_id": app_id, "app_secret": app_secret})
    d = _json(r)
    tok = d.get("accessToken") if isinstance(d, dict) else None
    if not tok:
        raise BrokerError(f"Dhan login was not accepted: {(d or {}).get('errorMessage') or (d or {}).get('message') or f'HTTP {r.status_code}'}.")
    return tok


# ------------------------------------------------------------------ config
def config(secrets: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, str]]:
    """What is configured, from Streamlit secrets: {'kite': {...}, 'dhan': {...}} with only complete sets."""
    if secrets is None:
        try:
            import streamlit as st
            secrets = {k: dict(st.secrets[k]) for k in ("kite", "dhan") if k in st.secrets}
        except Exception:
            secrets = {}
    out: Dict[str, Dict[str, str]] = {}
    k = secrets.get("kite") or {}
    if k.get("api_key") and k.get("api_secret"):
        out["kite"] = dict(api_key=str(k["api_key"]), api_secret=str(k["api_secret"]))
    d = secrets.get("dhan") or {}
    # Dhan calls them "API key" and "API secret" on its site and app_id / app_secret in its API: accept either name.
    # The app NAME you typed when creating the key is only a label and is not used.
    key = d.get("api_key") or d.get("app_id")
    sec = d.get("api_secret") or d.get("app_secret")
    if d.get("client_id") and key and sec:
        out["dhan"] = dict(client_id=str(d["client_id"]).strip(), app_id=str(key).strip(), app_secret=str(sec).strip())
    return out
