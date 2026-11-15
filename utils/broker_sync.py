"""
Read holdings straight from Zerodha (Kite Connect) and Dhan (DhanHQ v2). Read-only: the only
account endpoints called are the two holdings endpoints. Nothing here can place or change an order.

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
  (b) secrets: [dhan] client_id, app_id, app_secret (redirect URL = this app's URL)
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
    out = pd.DataFrame(items, columns=["symbol", "name", "qty", "avg"])
    if out.empty:
        return out
    out["symbol"] = out["symbol"].map(clean_symbol)
    out["qty"] = pd.to_numeric(out["qty"], errors="coerce")
    out["avg"] = pd.to_numeric(out["avg"], errors="coerce").replace(0, np.nan)
    return out[out.qty > 0].reset_index(drop=True)


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
    items = []
    for h in d.get("data") or []:
        if str(h.get("exchange", "")).upper() not in EQUITY_EXCHANGES:
            continue
        # settled + bought-but-unsettled (T1) + pledged as collateral
        qty = sum(float(h.get(k) or 0) for k in ("quantity", "t1_quantity", "collateral_quantity"))
        items.append(dict(symbol=h.get("tradingsymbol"), name="", qty=qty, avg=h.get("average_price")))
    return _rows(items)


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
    g = lambda h, *ks: next((h[k] for k in ks if h.get(k) not in (None, "")), None)
    items = [dict(symbol=g(h, "tradingSymbol", "tradingsymbol", "symbol"), name="",
                  qty=g(h, "totalQty", "totalQuantity", "quantity"), avg=g(h, "avgCostPrice", "avgPrice", "averagePrice"))
             for h in d if str(h.get("exchange", "")).upper() in EQUITY_EXCHANGES]
    return _rows(items)


def dhan_consent(client_id: str, app_id: str, app_secret: str) -> str:
    """Step 1 of the API-key login: returns a consentAppId for dhan_login_url."""
    r = _http().post(f"{DHAN_AUTH}/app/generate-consent", params={"client_id": client_id}, timeout=TIMEOUT,
                     headers={"app_id": app_id, "app_secret": app_secret})
    d = _json(r)
    cid = d.get("consentAppId") if isinstance(d, dict) else None
    if not cid:
        raise BrokerError(f"Dhan did not start a login session: {(d or {}).get('errorMessage') or (d or {}).get('message') or f'HTTP {r.status_code}'}.")
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
    if d.get("client_id") and d.get("app_id") and d.get("app_secret"):
        out["dhan"] = dict(client_id=str(d["client_id"]), app_id=str(d["app_id"]), app_secret=str(d["app_secret"]))
    return out
