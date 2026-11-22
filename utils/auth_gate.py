"""
Optional Streamlit Cloud authentication gate (password check via Streamlit secrets or env).

Behaviour:
  - Password configured (`[auth] password` in st.secrets or `APP_PASSWORD` env var):
      Requires login before rendering holdings, alerts, or notification triggers.
      Constant-time comparison (`hmac.compare_digest`) + sidebar lock button once signed in.
  - No password configured:
      Local runs stay unlocked cleanly; on Streamlit Cloud, displays a prominent warning
      banner until `[auth] password` is configured in Secrets.
"""
from __future__ import annotations

import hmac
import os

import streamlit as st


def is_cloud() -> bool:
    """Best-effort detection of hosted Streamlit Cloud runtime."""
    return bool(
        os.path.exists("/mount/src")
        or os.environ.get("STREAMLIT_SHARING_MODE")
        or os.environ.get("STREAMLIT_RUNTIME_ENV") == "cloud"
        or os.environ.get("HOSTNAME", "").startswith("streamlit")
    )


def configured_password() -> str:
    """Return configured dashboard password, or empty string if unset."""
    try:
        if hasattr(st, "secrets"):
            if "auth" in st.secrets:
                pw = str(st.secrets["auth"].get("password", "") or "").strip()
                if pw:
                    return pw
            pw_top = str(st.secrets.get("APP_PASSWORD", "") or "").strip()
            if pw_top:
                return pw_top
    except Exception:
        pass
    return str(os.environ.get("APP_PASSWORD", "") or "").strip()


def verify_password(entered: str, expected: str) -> bool:
    """Constant-time password check."""
    if not expected:
        return False
    return hmac.compare_digest(
        str(entered or "").encode("utf-8"),
        str(expected).encode("utf-8"),
    )


def require_auth() -> bool:
    """Gate app execution when a password is configured; warn on Cloud if unconfigured."""
    expected = configured_password()
    if not expected:
        if is_cloud():
            st.warning(
                "🔓 **Dashboard password is not configured.** Add `[auth] password = \"...\"` "
                "(or `APP_PASSWORD`) in Streamlit Cloud Secrets to lock holdings, alerts, and notifications."
            )
        return True

    if st.session_state.get("__app_authed") is True:
        if st.sidebar.button("🔒 Lock Dashboard", key="__auth_lock_btn"):
            st.session_state["__app_authed"] = False
            st.rerun()
        return True

    st.markdown("## 🔒 Alpha Trend — Sign In")
    st.caption("Dashboard access is password-protected (`[auth] password` / `APP_PASSWORD`).")
    with st.form("__auth_login_form", clear_on_submit=False):
        entered = st.text_input("Dashboard password", type="password", key="__auth_pw_input")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        if verify_password(entered, expected):
            st.session_state["__app_authed"] = True
            st.rerun()
            return True
        st.error("Incorrect password.")
    st.stop()
    return False
