import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import json
import time
from datetime import datetime, timedelta
import sys
import io

# Force UTF-8 encoding for standard output/error to prevent cp1252 crashes on Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Pick up code that Streamlit Cloud pulled into this running app (utils/hot_reload.py). Must run
# before the utils imports below so they bind to the reloaded modules.
try:
    from utils import hot_reload as _hot_reload
    if _hot_reload.refresh():
        st.cache_data.clear()
except Exception as _e:
    print(f"[hot_reload] skipped: {_e}", flush=True)

import yfinance as yf

from utils.ui_components import css_styles, page_header, sidebar_alert_panel
from utils.analytics_engine import calculate_cycle_position
from utils.market_explorer import render_market_explorer
from utils.positions import get_all_positions, remove_position, check_position_alerts
from utils.email_notifier import is_email_configured, send_trend_change_alert
from utils.telegram_notifier import send_telegram_message, is_telegram_configured
from utils.rs_alerts import check_rs_alerts
from utils import repo_store
from utils.fast_data_engine import load_base_fundamentals, fetch_and_process_market_data


def _rerun_fragment() -> None:
    """Rerun just the current fragment; a full rerun when the click came through a full run."""
    from streamlit.errors import StreamlitAPIException
    try:
        st.rerun(scope="fragment")
    except StreamlitAPIException:
        st.rerun()


def _alert_sync_banner(stt: dict, retry, key: str) -> None:
    """Say plainly where alerts live and whether the last write reached GitHub."""
    if not stt["configured"]:
        st.warning(
            "⚠️ **Alerts are stored on this device only.** They will not appear on another "
            "machine, the EOD workflow cannot fire them, and on Streamlit Cloud a restart loses "
            "them. To sync, add a GitHub token with Contents: Read and write on this repo as "
            "`GITHUB_TOKEN` (env) or `[github] token` in Streamlit secrets.", icon="⚠️")
        return
    if stt.get("can_write") is False:
        st.error("⚠️ **The GitHub token can read this repo but not write to it**, so new alerts "
                 "cannot be saved there (this is why alerts kept replacing each other). Give the "
                 f"token Contents: Read and write on {repo_store.get_repo()}, or a new token with it. "
                 f"Until then alerts stay on this device ({stt['pending']} waiting to sync).")
    elif stt["pending"] and stt["ok"] is not False:
        st.caption(f"☁️ {stt['pending']} change(s) syncing to GitHub in the background — "
                   "carry on, nothing to wait for.")
    elif stt["ok"] is False:
        st.error(f"⚠️ **{stt['pending']} change(s) not saved to GitHub yet** — kept on this "
                 f"device and retried on every load. Last error: {stt['error'] or 'unknown'}")
        if st.button("↻ Retry sync now", key=f"retry_sync_{key}"):
            st.toast("Synced" if retry() else "Still failing — see the error above", icon="☁️")
            st.rerun()
    elif stt["source"] == "cache":
        st.warning("☁️ GitHub could not be read just now — showing the last synced copy.")
    else:
        st.caption(f"☁️ Synced to **{repo_store.get_repo()}** (`{stt.get('branch', repo_store.get_branch())}` branch)"
                   + (f" · last write {stt['when']} ✓" if stt["ok"] else "")
                   + " — alerts follow you across devices and fire from the EOD workflow.")


def _us_quote_url(ticker: str) -> str:
    """
    Quote URL for a US ticker.

    Yahoo, not Google Finance: a Google quote URL only resolves with an
    exchange suffix (AAPL:NASDAQ), and the US universe carries no exchange
    column, so every bare /finance/quote/AAPL link was dead. Yahoo resolves
    from the ticker alone, and data/sp500_tickers.csv already stores symbols
    in Yahoo's convention (BRK-B, BF-B) because yfinance feeds this pipeline —
    Google would need BRK.B, so those two were doubly broken.
    """
    return f"https://finance.yahoo.com/quote/{str(ticker).strip()}"

_US_DISPLAY_RE = r"https://finance\.yahoo\.com/quote/(.+)$"

# Debug mode: set DASH_DEBUG=1 to show debug panel
DASH_DEBUG = os.environ.get('DASH_DEBUG', '0') == '1'

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Nifty 500 Research Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PREMIUM DASHBOARD CSS ---
st.markdown(css_styles(), unsafe_allow_html=True)

from utils.auth_gate import require_auth as _require_auth
_require_auth()

# --- SIDEBAR & NAV ---
st.sidebar.title("🚀 Alpha Trend")

if st.sidebar.button("🔄 Hard Reset Cache", type="primary"):
    if os.path.exists("nifty500_cache.csv"):
        os.remove("nifty500_cache.csv")
    if os.path.exists("market_data.parquet"):
        os.remove("market_data.parquet")
    # Clear ALL caches including @st.cache_data / @st.cache_resource / @lru_cache
    st.cache_data.clear()
    st.cache_resource.clear()
    for _mod_name, _fn_name in (
        ("utils.nse_prices", "_load"),
        ("utils.nse_data", "_read_cached"),
        ("utils.corp_actions", "_load_cached"),
        ("utils.breakout_tags", "_load_industry_scores_cached"),
        ("utils.breakout_tags", "_load_tag_edges_cached"),
        ("utils.breakout_tags", "_latest_eps_cached"),
    ):
        try:
            _mod = __import__(_mod_name, fromlist=[_fn_name])
            getattr(_mod, _fn_name).cache_clear()
        except Exception:
            pass
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

live_mode_toggle = st.sidebar.checkbox("🚀 Bypass Master Cache (Live Mode)", value=False, help="Forces a fresh pull of all 1000 stocks from Yahoo Finance instead of using the daily Parquet cache.")

st.sidebar.markdown("---")

# === SCREENING PRESETS ===
st.sidebar.markdown("### 🎯 Quick Presets")
preset = st.sidebar.selectbox(
    "Strategy Filter",
    [
        "All Stocks",
        "🚀 Strong Momentum (Top 20%)",
        "💎 Quality at Reasonable Price",
        "📈 Breakout Candidates",
        "🔥 Turnaround Plays",
        "🧬 DNA-3 V2 Picks"
    ]
)
st.session_state['preset'] = preset

# === WATCHLIST (Persistent via positions.json) ===
st.sidebar.markdown("---")
st.sidebar.markdown("### ⭐ Watchlist")
_all_sidebar_positions = get_all_positions()
watchlist_positions = [p for p in _all_sidebar_positions if p.get('status', 'active') == 'watching']
watchlist_count = len(watchlist_positions)
st.sidebar.write(f"Watching: {watchlist_count} stocks")

if watchlist_count > 0:
    with st.sidebar.expander("View Watchlist"):
        for _wl_i, wp in enumerate(watchlist_positions):
            ticker = wp.get('ticker', '')
            display_name = ticker.replace('.NS', '').replace('.BO', '')
            wl_c1, wl_c2 = st.columns([3, 1])
            wl_c1.write(f"📌 {display_name}")
            if wl_c2.button("❌", key=f"wl_rm_{wp.get('id', _wl_i)}_{ticker}"):
                remove_position(ticker)
                st.rerun()

# === POSITIONS & ALERTS (Unified) ===
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Positions")
active_sidebar = [p for p in _all_sidebar_positions if p.get('status', 'active') == 'active']
active_count = len(active_sidebar)

if active_count > 0:
    st.sidebar.caption(f"Active: {active_count} positions")
    for pos in active_sidebar[:5]:
        ticker = pos.get('ticker', '').replace('.NS', '').replace('.BO', '')
        sl = f"SL:{pos['stop_loss']:.0f}" if pos.get('stop_loss') else ""
        tgt = f"T:{pos['target']:.0f}" if pos.get('target') else ""
        st.sidebar.caption(f"• {ticker} {sl} {tgt}")
    if active_count > 5:
        st.sidebar.caption(f"  +{active_count - 5} more...")
else:
    st.sidebar.caption("No active positions")

# Link to full manager
if st.sidebar.button("📋 Open Position Manager", use_container_width=True, type="secondary"):
    st.session_state['nav_page'] = "📊 Return Tracker"
    st.rerun()

# Persistent Alert Panel in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔔 Alerts")

# ============================================================
# CENTRALIZED DATA LOADING + AUTO-REFRESH
# ============================================================
AUTO_REFRESH_MINUTES = 30  # Auto-refresh data after 30 minutes

# NOTE: the client-side auto-refresh heartbeat is defined AFTER the data
# load block below (see _autorefresh_heartbeat). It must run after
# data_loaded_at is set on a refresh, otherwise its inline execution during
# a full rerun would see perpetually-stale data and re-trigger st.rerun()
# forever (that infinite loop took the app down — connection timeout — when
# the heartbeat was placed here at the top and called st.rerun()
# unconditionally).

@st.cache_resource
def _market_store() -> dict:
    return {}


def _load_market_data(live: bool, key: str) -> pd.DataFrame:
    """
    Market data shared by every session. `key` is the snapshot file + mtime, or "live".
    A plain process-wide store, not st.cache_data: the loaders draw Streamlit elements
    (progress, errors), and a cached function replaying those in another session raises
    CacheReplayClosureError.
    """
    store = _market_store()
    hit = store.get((live, key))
    if hit and time.time() - hit[0] < AUTO_REFRESH_MINUTES * 60:
        return hit[1].copy()
    fundamentals = load_base_fundamentals(live_mode=live)
    out = fetch_and_process_market_data(fundamentals['ticker'].tolist(), fundamentals, live_mode=live,
                                        price_source="yahoo" if live else None)   # Live Mode: intraday prices are Yahoo's
    if out is not None and not out.empty:
        store.clear()                                   # keep one entry: frames are ~1,170 x 125
        store[(live, key)] = (time.time(), out)
        return out.copy()
    return out


@st.cache_data(ttl=AUTO_REFRESH_MINUTES * 60, show_spinner=False, max_entries=2)
def _load_nifty(start: str, live: bool = False) -> pd.DataFrame:
    """Nifty history, shared across sessions. Raises on an empty fetch so a failure is not cached."""
    from utils.yf_safe import safe_history
    d = safe_history("^NSEI", start=start, source="yahoo" if live else None)
    if d is None or d.empty:
        raise RuntimeError("empty Nifty fetch")
    return d


@st.cache_data(ttl=900, show_spinner=False, max_entries=8)
def _yf_download_cached(tickers, **kw) -> pd.DataFrame:
    """yf.download shared across reruns and sessions for 15 minutes (these calls used to run on
    every widget click). Raises on an empty result so a failed fetch is not cached."""
    d = yf.download(list(tickers) if isinstance(tickers, tuple) else tickers, **kw)
    if d is None or d.empty:
        raise RuntimeError("empty download")
    return d


@st.cache_data(ttl=900, show_spinner=False, max_entries=4)
def _nifty_recent(period: str = "100d") -> pd.DataFrame:
    d = yf.Ticker("^NSEI").history(period=period)
    if d is None or d.empty:
        raise RuntimeError("empty Nifty fetch")
    return d


@st.cache_data(ttl=12 * 3600, show_spinner=False, max_entries=600)
def _cycle_position_ok(ticker: str) -> dict:
    r = calculate_cycle_position(ticker)
    if "error" in r:
        raise ValueError(r.get("error", "no data"))
    return r


@st.cache_resource
def _cycle_err_store() -> dict:
    return {}        # survives reruns (a module-level dict is rebuilt on every script run)


def _cycle_position(ticker: str) -> dict:
    """
    Margin-cycle read for one stock, cached. Each call downloads a quarterly income statement
    from Yahoo; Sector Pulse made ~23 of them on every widget click. Results are kept 12 hours
    (quarterly data), failures 30 minutes so a stock with no statement is not retried per click.
    """
    _CYCLE_ERR = _cycle_err_store()
    hit = _CYCLE_ERR.get(ticker)
    if hit and time.time() - hit[0] < 1800:
        return hit[1]
    try:
        return _cycle_position_ok(ticker)
    except ValueError as e:
        r = {"phase": "UNKNOWN", "phase_num": 0, "error": str(e)}
        _CYCLE_ERR[ticker] = (time.time(), r)
        return r


# Check if data needs refresh (stale data)
_needs_refresh = 'market_data' not in st.session_state
# switching Live Mode on or off must reload, not wait for the 30-minute timer
if st.session_state.get('_loaded_live_mode') is not None and st.session_state['_loaded_live_mode'] != bool(live_mode_toggle):
    _needs_refresh = True
if not _needs_refresh and 'data_loaded_at' in st.session_state:
    _minutes_old = (datetime.now() - st.session_state['data_loaded_at']).total_seconds() / 60
    if _minutes_old > AUTO_REFRESH_MINUTES:
        _needs_refresh = True
        st.toast("🔄 Data is stale, refreshing...", icon="⏰")

if _needs_refresh:
    
    load_status = st.empty()
    progress_bar = st.progress(0, text="Initializing...")
    
    # Steps 1-2: market data. Shared across sessions and reruns (st.cache_data), so a second
    # browser tab or a reload within the refresh window costs nothing. Without Live Mode this
    # reads the newest EOD snapshot (well under a second); Live Mode downloads from Yahoo.
    progress_bar.progress(20, text=("Fetching live prices from Yahoo (a few minutes)..." if live_mode_toggle
                                    else "Loading the latest end-of-day snapshot..."))
    _snap_files = sorted(glob.glob("data/cache/market_master_*.parquet"))
    _snap_key = (f"{_snap_files[-1]}:{os.path.getmtime(_snap_files[-1])}" if _snap_files and not live_mode_toggle
                 else "live")
    df = _load_market_data(bool(live_mode_toggle), _snap_key)
    
    if df.empty:
        progress_bar.empty()
        st.error("⚠️ Data Fetch Failed! Check internet.")
        st.stop()
    
    # Step 3: Pre-fetch Nifty index data (shared across all pages)
    progress_bar.progress(80, text="Step 3/4: Fetching Nifty index data...")
    try:
        from utils.yf_safe import safe_history
        _nifty_start = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
        # Retry-with-backoff, like every other fetch in the pipeline. A bare
        # .history() here had no retry, and on failure the `if not empty` guard
        # below silently LEFT THE PREVIOUS SERIES IN session_state — so one
        # transient Yahoo blip froze the Nifty series indefinitely while stock
        # prices kept refreshing. That is not just a stale label: the RS
        # Divergence scan compares today's stock change against
        # nifty_df.Close[-1] vs [-2], so a stale index series makes it compare
        # today's stock move to a days-old Nifty move.
        try:
            _nifty_data = _load_nifty(_nifty_start, live=bool(live_mode_toggle))
        except Exception:
            _nifty_data = pd.DataFrame()
        if _nifty_data is not None and not _nifty_data.empty:
            st.session_state['nifty_data'] = _nifty_data
            st.session_state['nifty_fetched_at'] = datetime.now()
        else:
            # Fail loudly rather than inheriting a stale index series.
            st.session_state.pop('nifty_data', None)
            st.session_state['nifty_fetch_failed'] = True
        if not _nifty_data.empty:
            
            # --- V2.2 Regime Manager ---
            from utils.regime_manager import classify_regime
            _nd = _nifty_data.copy()
            if _nd.index.tz is not None:
                _nd.index = _nd.index.tz_localize(None)
            _ma200 = _nd['Close'].rolling(200).mean().iloc[-1]
            _h52w  = _nd['High'].rolling(252).max().iloc[-1]
            _close = _nd['Close'].iloc[-1]
            current_regime = classify_regime(float(_close), float(_ma200), float(_h52w))
            st.session_state['market_regime'] = current_regime        # engine mode (drives the live book)
    except Exception:
        pass  # Non-fatal — individual pages can fallback
    
    # Step 4: Finalize
    progress_bar.progress(95, text="Step 4/4: Calculating scores...")
    st.session_state['market_data'] = df
    st.session_state['data_loaded_at'] = datetime.now()
    st.session_state['_loaded_live_mode'] = bool(live_mode_toggle)
    # what the prices are: a live pull, or the EOD snapshot of a given session
    from utils import snapshots as _snaps
    _asof = _snaps.bar_date(df, _snap_files[-1] if _snap_files else None)
    st.session_state['data_asof'] = None if live_mode_toggle else _asof
    
    # Cache sector PE for scoring consistency
    # 'pe' can arrive object-dtype / carry junk (Yahoo 'Infinity' sentinels in
    # cached data) — coerce before median or pandas raises TypeError.
    if 'pe' in df.columns and 'sector' in df.columns:
         _base_pe = df[df['extended'].ne(True)] if 'extended' in df.columns else df
         _pe_num = pd.to_numeric(_base_pe['pe'], errors='coerce').replace([np.inf, -np.inf], np.nan)
         _pe_num = _pe_num.where((_pe_num > 0) & (_pe_num < 500))
         st.session_state['sector_pe_cache'] = _base_pe.assign(pe=_pe_num).groupby('sector')['pe'].median().to_dict()
    
    progress_bar.progress(100, text="✅ Ready!")
    progress_bar.empty()
    load_status.empty()


df = st.session_state['market_data']

# Stocks you hold or watch that are outside the snapshot (new extended names not yet picked up by
# the nightly run) are computed once per data day so My Book, the brief and alerts can price them.
@st.cache_resource(show_spinner=False)
def _tracked_store() -> dict:
    return {}                                   # (snapshot key, tickers) -> rows, shared by every session

def _tracked_rows(key: str, tickers: tuple) -> pd.DataFrame:
    """One download per snapshot for the whole app; a failed attempt is remembered too, so it is not retried on every load."""
    store = _tracked_store()
    if (key, tickers) not in store:
        from utils import extended as _ext
        with st.spinner("Pricing your stocks outside the universe…"):
            try:
                store[(key, tickers)] = _ext.rows_for(list(tickers))
            except Exception as _err:
                print(f"[extended] on-demand pricing failed: {_err}", flush=True)
                store[(key, tickers)] = pd.DataFrame()
        for k in [k for k in store if k[0] != key]:
            del store[k]
    return store[(key, tickers)]

try:
    from utils import extended as _ext_main
    try:                                   # price-alert stocks outside the universe get a price too
        from utils.alerts import load_alerts as _load_px_alerts
        _px_alert_rows = [dict(ticker=_a.get('ticker'), status='watching') for _a in _load_px_alerts()]
    except Exception:
        _px_alert_rows = []
    _need = tuple(t for t in _ext_main.tracked(set(df['ticker']), list(_all_sidebar_positions) + _px_alert_rows)
                  if t not in set(df['ticker']))
    if _need:
        _rows = _tracked_rows(f"{len(df)}|{df['bar_date'].iloc[0] if 'bar_date' in df.columns else ''}", _need)
        if len(_rows):
            df = pd.concat([df, _rows.reindex(columns=list(dict.fromkeys(list(df.columns) + ['extended'])))], ignore_index=True)
except Exception as _e:
    print(f"[extended] tracked rows skipped: {_e}", flush=True)

# NSE's daily data (scripts/nse_daily.py): delivery %, surveillance lists, results / ex-dates. Cached per file set.
@st.cache_data(ttl=3600, show_spinner=False)
def _nse_cols(key: str, tickers: tuple) -> pd.DataFrame:
    from utils import nse_data as _nd
    return _nd.attach(pd.DataFrame({"ticker": list(tickers)})).set_index("ticker")

try:
    import glob as _glob_nse
    _nse_files = sorted(_glob_nse.glob("data/nse/*.csv") + _glob_nse.glob("data/nse/prices/d/*.parquet"))[-12:]
    _nse_key = "|".join(f"{os.path.basename(_f)}:{os.path.getmtime(_f)}" for _f in _nse_files)
    if _nse_key:
        _nc = _nse_cols(_nse_key, tuple(df['ticker']))
        for _c in _nc.columns:
            df[_c] = df['ticker'].map(_nc[_c])
except Exception as _e:
    print(f"[nse] columns skipped: {_e}", flush=True)

# --- Auto-refresh heartbeat (placed here, AFTER the data-load block) ---
# A Streamlit script only reruns on user interaction, not on a timer, so a
# tab left open and just watched never refreshes — data could sit hours
# stale under a "Today" label. This fragment's run_every timer fires on its
# own every AUTO_REFRESH_MINUTES even with no interaction.
#
# Two things make it safe (an earlier version that just called st.rerun()
# unconditionally at the top of the script infinite-looped and took the app
# down):
#   1. It only escalates to a full rerun when data is ACTUALLY stale.
#   2. It lives after the data-load block, so on the resulting full rerun the
#      data reloads and data_loaded_at resets BEFORE this fragment's inline
#      execution checks staleness again — so it sees fresh data and stops,
#      instead of re-triggering.
# It also pops 'market_data' before rerunning so the top-of-script loader
# treats data as stale unconditionally (robust to any threshold edge case).
@st.fragment(run_every=f"{AUTO_REFRESH_MINUTES}m")
def _autorefresh_heartbeat():
    _loaded_at = st.session_state.get('data_loaded_at')
    if _loaded_at is None:
        return
    if (datetime.now() - _loaded_at).total_seconds() >= AUTO_REFRESH_MINUTES * 60:
        st.session_state.pop('market_data', None)
        st.rerun()
_autorefresh_heartbeat()

# Data freshness indicator
if 'data_loaded_at' in st.session_state:
    loaded_at = st.session_state['data_loaded_at']
    minutes_ago = int((datetime.now() - loaded_at).total_seconds() / 60)
    if minutes_ago < 1:
        freshness_text = "🟢 Just now"
    elif minutes_ago < 60:
        freshness_text = f"🟢 {minutes_ago} min ago"
    elif minutes_ago < 240:
        freshness_text = f"🟡 {minutes_ago // 60}h ago"
    else:
        freshness_text = f"🔴 {minutes_ago // 60}h ago (stale)"
    _asof = st.session_state.get('data_asof')
    if _asof is not None:
        _age = len(pd.bdate_range(_asof, pd.Timestamp.today().normalize())) - 1
        st.sidebar.caption(f"📊 {len(df)} stocks • 📅 End-of-day prices for **{_asof:%d %b}**"
                           + (f" ({_age} sessions old ⚠️)" if _age > 1 else "")
                           + "  \nTick **Live Mode** above for today's prices (takes a few minutes).")
    else:
        st.sidebar.caption(f"📊 {len(df)} stocks • Live prices, updated: {freshness_text}")
    
    # One regime read for the whole app (utils/regime_badge.py): Nifty 5-state market regime, breadth, and
    # the engine's own mode. Computed per data load; the pages render the same object.
    from utils import regime_badge as _rb
    st.session_state['regime_info'] = _rb.compute(st.session_state.get('nifty_data'), df)
    st.sidebar.markdown(_rb.sidebar_line(st.session_state['regime_info']))
else:
    st.sidebar.success(f"Loaded {len(df)} Tickers")

# Check positions for SL/Target hits and notify via Telegram + Email
triggered_positions = check_position_alerts(df)
if triggered_positions and 'pos_alerts_notified' not in st.session_state:
    alert_messages = []
    for pos in triggered_positions:
        ticker_display = pos.get('ticker', '').replace('.NS', '').replace('.BO', '')
        alert_type = pos.get('alert_type', 'ALERT')
        current_price = pos.get('current_price', 0)
        
        if alert_type == 'STOP_LOSS':
            alert_messages.append(f"🔴 {ticker_display} SL HIT ₹{current_price:.0f}")
        else:
            alert_messages.append(f"🎯 {ticker_display} TARGET HIT ₹{current_price:.0f}")
        
        # Send Telegram
        try:
            if is_telegram_configured():
                entry = pos.get('entry_price', 0)
                pnl = pos.get('pnl_pct', 0)
                msg = (f"{'SL' if alert_type == 'STOP_LOSS' else 'TARGET'} HIT: {ticker_display}\n"
                       f"Price: ₹{current_price:.2f} | Entry: ₹{entry:.2f} | P&L: {pnl:+.1f}%\n"
                       f"Action Required: Review position")
                send_telegram_message(msg)
        except:
            pass
        
        # Send Email  
        try:
            if is_email_configured():
                send_trend_change_alert([{
                    'ticker': pos.get('ticker'),
                    'entry_trend_signal': 'ACTIVE',
                    'current_signal': alert_type.replace('_', ' '),
                    'return_pct': pos.get('pnl_pct', 0),
                    'days_tracked': 0
                }])
        except:
            pass
    
    # Single batched toast instead of one per alert
    if alert_messages:
        st.toast(f"🚨 {len(alert_messages)} Alert(s): {' | '.join(alert_messages)}", icon="🔔")
    
    st.session_state['pos_alerts_notified'] = True

# === COMPOSITE-RS CROSSING ALERTS ===
# Read-only here. The EOD workflow (rs_alert_daemon.py) is the one evaluator
# that advances last_value and sends Telegram, inside an atomic update of
# data/rs_alerts.json. When the dashboard also wrote state, every page load
# rewrote the alert file and raced the daemon and other devices. What shows
# here is "crossed since the last EOD check" on the data loaded now.
if 'rs_alerts_fired' not in st.session_state:
    try:
        st.session_state['rs_alerts_fired'] = check_rs_alerts(df, persist=False)
    except Exception as _e:
        st.session_state['rs_alerts_fired'] = []
        if DASH_DEBUG:
            st.sidebar.caption(f"RS alert check failed: {_e}")

_rs_fired = st.session_state.get('rs_alerts_fired', [])
if _rs_fired and 'rs_alerts_notified' not in st.session_state:
    st.toast(f"🔔 {len(_rs_fired)} RS alert(s) since the last close: "
             + " | ".join(a['alert_message'] for a in _rs_fired[:3]), icon="📈")
    st.session_state['rs_alerts_notified'] = True

# Render persistent alert panel in sidebar (regardless of notification state)
if triggered_positions:
    st.sidebar.markdown(sidebar_alert_panel(triggered_positions), unsafe_allow_html=True)
elif not _rs_fired:
    st.sidebar.caption("✅ No alerts")

if _rs_fired:
    st.sidebar.markdown("#### 📈 RS Crossings")
    for _a in _rs_fired:
        st.sidebar.warning(_a['alert_message'])

# Debug: Show data verification (only if DASH_DEBUG=1)
if DASH_DEBUG:
    with st.sidebar.expander("🔍 Debug Data"):
        st.write(f"Columns: {len(df.columns)}")
        if 'trend_signal' in df.columns:
            st.write("Signals:", df['trend_signal'].value_counts().to_dict())
        else:
            st.error("trend_signal column MISSING!")
        if 'trend_score' in df.columns:
            st.write(f"Trend Score Range: {df['trend_score'].min()} - {df['trend_score'].max()}")
        else:
            st.error("trend_score column MISSING!")
        sample = df.iloc[0].dropna().to_dict()
        price_keys = [k for k in sample.keys() if any(x in k.lower() for x in ['price', 'average', 'high', 'low', '52', 'week'])]
        st.write(price_keys[:15])


# trend_engine imports moved to top

# (Auto-alert check is now handled by the unified position alert system above)

# === DNA3 MORNING BRIEF AUTO-TRIGGER (Once per day on dashboard load) ===
try:
    _json = json
    _config_path = "config.json"
    _config = {}
    if os.path.exists(_config_path):
        with open(_config_path, 'r') as _f:
            _config = _json.load(_f)
    
    dna3_alert_config = _config.get("dna3_morning_alert", {})
    dna3_alert_enabled = dna3_alert_config.get("enabled", False)
    last_sent_date = dna3_alert_config.get("last_sent_date", "")
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    if dna3_alert_enabled and last_sent_date != today_str and 'dna3_alert_sent' not in st.session_state:
        # Auto-send DNA3 morning brief (first load of the day)
        from dna3_morning_alert import send_morning_alert
        results = send_morning_alert()
        if results:
            # Record sent date
            _config["dna3_morning_alert"]["last_sent_date"] = today_str
            with open(_config_path, 'w') as _f:
                _json.dump(_config, _f, indent=2)
            
            channels = []
            if results.get('telegram'): channels.append("Telegram")
            if results.get('email'): channels.append("Email")
            if channels:
                st.toast(f"📤 DNA3 Morning Brief sent via {' + '.join(channels)}!", icon="🧬")
        st.session_state['dna3_alert_sent'] = True
except Exception as e:
    pass  # Silently fail

# ... imports ...

# --- NAVIGATION CONTROLLER ---
st.sidebar.markdown("---")

_NAV = {
    "🔍 Market Specs": ("page_market_specs", ["☀️ Daily Brief", "🧭 Investment Funnel", "🌊 Trend Scanner", "🔬 Technicals",
                                             "📊 Sector Pulse", "🧪 Signal Lab", "🔍 Market Explorer", "🖥️ AI Capex",
                                             "🇺🇸 US AI Play", "🇺🇸 US Scanner"]),
    "📋 Portfolio Manager": ("page_portfolio", ["📊 Return Tracker", "📝 Notes"]),
    "⚖️ Analysis Lab": ("page_analysis", ["📋 Stock Report", "⚖️ Compare Stocks", "📉 Deep Dive", "⏳ Time Trends"]),
    # Replaced by Daily Brief + Signal Lab (analysis/PAGE_AUDIT_2026_11.md); kept, unchanged, for reference.
    "🗄️ Archive": ("page_archive", ["☀️ Morning Pulse", "🚀 Live Trading Desk", "🎯 Turnaround Radar"]),
}

# Shortcut buttons ("Open Position Manager", Deep Dive "Report", ...) leave a
# nav_page request. It used to override `page` for ONE run only, so the next
# interaction -- saving an alert, clicking a tab -- silently fell back to the
# sidebar's page (usually Morning Pulse), which looked like the dashboard
# reloading. Turn the request into the real sidebar selection instead, before
# those widgets are drawn, so the page sticks.
# A broker sending the user back after login (Zerodha / Dhan holdings sync, utils/broker_sync.py):
# exchange the one-time code, fetch holdings and go to the Import tab.
try:
    if "request_token" in st.query_params or "tokenId" in st.query_params:
        from utils import book_view as _bv_cb
        _bv_cb.handle_broker_redirect()
except Exception as _e:
    st.session_state["sync_error"] = f"Broker sync failed: {type(_e).__name__}"

if 'nav_page' in st.session_state:
    _want = st.session_state.pop('nav_page')
    for _ws, (_key, _pages) in _NAV.items():
        if _want in _pages:
            st.session_state['active_workspace'] = _ws
            st.session_state[_key] = _want
            break

# Initialize workspace in session state if not present
if 'active_workspace' not in st.session_state:
    st.session_state['active_workspace'] = "🔍 Market Specs"

active_workspace = st.sidebar.selectbox("📂 Workspace", [
    "🔍 Market Specs", 
    "📋 Portfolio Manager", 
    "⚖️ Analysis Lab",
    "🗄️ Archive"
], key="active_workspace")

page = "🌊 Trend Scanner" # Default

if active_workspace in _NAV:
    _key, _pages = _NAV[active_workspace]
    page = st.sidebar.radio("View" if active_workspace == "🔍 Market Specs" else "Tools", _pages, key=_key)



if page == "⚖️ Compare Stocks":
    from views.deep_dive_and_misc import render_compare_stocks
    render_compare_stocks(df)

elif page == "🔍 Market Explorer":
    render_market_explorer()

elif page == "🚀 Live Trading Desk":
    from views.live_desk import render as _render_live_desk
    _render_live_desk(
        df,
        load_nifty=_load_nifty,
        us_quote_url=_us_quote_url,
        us_display_re=_US_DISPLAY_RE,
        auto_refresh_minutes=AUTO_REFRESH_MINUTES,
    )

elif page == "🌊 Trend Scanner":
    from views.trend_scanner import render as _render_trend_scanner
    _render_trend_scanner(df, yf_download_cached=_yf_download_cached, nifty_recent=_nifty_recent)

elif page == "🔬 Technicals":
    from utils.technicals_view import render as _render_technicals
    _render_technicals(df)

elif page == "📊 Return Tracker":
    from views.return_tracker import render as _render_return_tracker
    _render_return_tracker(
        df,
        alert_sync_banner=_alert_sync_banner,
        rerun_fragment=_rerun_fragment,
        rs_fired=_rs_fired,
    )

elif page == "☀️ Morning Pulse":
    from views.pulse_and_trends import render_morning_pulse
    render_morning_pulse(df)

elif page == "📊 Sector Pulse":
    from views.sector_pulse import render as _render_sector_pulse
    _render_sector_pulse(df, cycle_position=_cycle_position)

elif page == "⏳ Time Trends":
    from views.pulse_and_trends import render_time_trends
    render_time_trends(df)

elif page == "☀️ Daily Brief":
    st.markdown(page_header("☀️ Daily Brief", "Market, leading industries, today's funnel picks, your positions and the model portfolio — the one page to read each day"), unsafe_allow_html=True)
    from utils import daily_brief_view
    daily_brief_view.render(df)

elif page == "🧪 Signal Lab":
    st.markdown(page_header("🧪 Signal Lab", "Every scanner's signals in one place, with a scorecard of how each has done against the market"), unsafe_allow_html=True)
    from utils import signal_lab_view
    signal_lab_view.render(df)

elif page == "🧭 Investment Funnel":
    st.markdown(page_header("🧭 Investment Funnel", "Market → Industry → Trend → Business → Setup → Value → Plan: one path from the universe to a short list"), unsafe_allow_html=True)
    from utils import funnel_view
    funnel_view.render(df)

elif page == "📋 Stock Report":
    st.markdown(page_header("📋 Stock Report", "One-shot report: funnel checklist, industry, trend, entry plan, business trends and valuation"), unsafe_allow_html=True)
    from utils import stock_report_view
    stock_report_view.render(df)

elif page == "📉 Deep Dive":
    from views.deep_dive_and_misc import render_deep_dive
    render_deep_dive()

elif page == "📝 Notes":
    from views.deep_dive_and_misc import render_notes
    render_notes()

elif page == "\U0001f3af Turnaround Radar":
    from views.turnaround_radar import render as _render_turnaround_radar
    _render_turnaround_radar(df)

elif page == "🖥️ AI Capex":
    from views.deep_dive_and_misc import render_ai_capex
    render_ai_capex(nifty_recent=_nifty_recent)

elif page == "🇺🇸 US AI Play":
    from views.us_pages import render_us_ai_play
    render_us_ai_play()

elif page == "🇺🇸 US Scanner":
    from views.us_pages import render_us_scanner
    render_us_scanner(us_quote_url=_us_quote_url, us_display_re=_US_DISPLAY_RE)
