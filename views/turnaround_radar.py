"""Turnaround Radar (IAS) page view (extracted from main.py)."""
from __future__ import annotations

from datetime import datetime
import os

import pandas as pd
import streamlit as st
import yfinance as yf

from utils.ui_components import page_header


WATCHLIST_CSV = "data/turnaround_watchlist.csv"

@st.cache_data(ttl=3600)
def load_turnaround_watchlist():
    if not os.path.exists(WATCHLIST_CSV):
        return pd.DataFrame()
    return pd.read_csv(WATCHLIST_CSV)


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_turnaround_live_prices(tickers_t: tuple) -> dict:
    """Batch-download last-close for log tickers not already in market_data."""
    if not tickers_t:
        return {}
    try:
        raw = yf.download(
            list(tickers_t), period="5d",
            threads=False, progress=False, auto_adjust=True
        )
        if raw is None or raw.empty:
            return {}
        close = (raw["Close"] if isinstance(raw.columns, pd.MultiIndex)
                 else raw.rename(columns={raw.columns[0]: tickers_t[0]
                                          if len(tickers_t) == 1 else raw.columns[0]}))
        result = {}
        for t in tickers_t:
            try:
                s = close[t].dropna() if t in close.columns else pd.Series(dtype=float)
                if not s.empty:
                    result[t] = round(float(s.iloc[-1]), 2)
            except Exception:
                pass
        return result
    except Exception:
        return {}


def render(df: pd.DataFrame) -> None:
    import plotly.graph_objects as go

    st.markdown(page_header(
        "\U0001f3af Institutional Turnaround Radar",
        "Beaten-down Nifty 1000 stocks showing early institutional accumulation \u2014 before V21 qualification."
    ), unsafe_allow_html=True)

    wdf = load_turnaround_watchlist()

    # Merge sector from main market df (not in watchlist CSV)
    if not wdf.empty and "Sector" not in wdf.columns and "sector" in df.columns:
        _sec_map = df.set_index("ticker")["sector"].to_dict()
        wdf["Sector"] = wdf["Ticker"].map(_sec_map).fillna("Unknown")
    elif not wdf.empty and "Sector" not in wdf.columns:
        wdf["Sector"] = "Unknown"

    regime = st.session_state.get('market_regime', 'UNKNOWN')
    if regime in ['BEAR', 'CRISIS']:
        from utils import regime_badge as _rb
        st.error(f"🚨 **Model rules: {_rb.engine_status({'mode': regime})}** 🚨\n\nTurnaround Module Suspended. The system is structurally blocking knife-catches during severe market drawdowns to preserve capital.")
        st.stop()

    if wdf.empty:
        st.warning("Watchlist not generated yet. Run `python turnaround_screener.py` locally or wait for the next GitHub Actions daily run.")
        st.code("python turnaround_screener.py", language="bash")
        st.stop()

    alert_n = int((wdf["Tier"] == "ALERT").sum())
    ready_n = int((wdf["Tier"] == "READY").sum())
    watch_n = int((wdf["Tier"] == "WATCH").sum())

    # Colour language: green = ready to fire, gold = warming up, blue = early radar ping
    TIER_COLORS = {
        "ALERT": "#00C853",  # Emerald green  — imminent V21, ready to fire
        "READY": "#FFB300",  # Amber gold     — RS velocity confirmed, getting warm
        "WATCH": "#42A5F5",  # Sky blue       — early accumulation ping
    }

    # --- Tier cards ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("\U0001f7e2 ALERT", alert_n, help="IAS 80+: RS21+RS63 turning. Strong liq floor. Imminent V21 — ready to fire.")
    c2.metric("\U0001f7e1 READY", ready_n, help="IAS 60-79: Liq stable, RS velocity confirmed. Getting warm.")
    c3.metric("\U0001f535 WATCH", watch_n, help="IAS 35-59: Early institutional ping. Set MA50 alert.")
    c4.metric("Total Pool", len(wdf))

    st.info("**How to use:** These are BELOW MA50 \u2014 do NOT buy. Set a price alert at each stock's MA50 level. When crossed with CompRS > 0, re-check the V21 scanner for fast-track to portfolio.")

    st.markdown("---")

    # --- Filters ---
    fcols = st.columns([1, 1, 1, 1.5, 1.5])
    tier_filter    = fcols[0].multiselect("Tier",  ["ALERT","READY","WATCH"], default=["ALERT","READY","WATCH"])
    cycle_filter   = fcols[1].multiselect("Cycle", ["LONG","MID","SHORT"],    default=["LONG","MID","SHORT"])
    min_ias        = fcols[2].slider("Min IAS", 35, 90, 35)
    _all_sectors   = sorted(wdf["Sector"].dropna().unique().tolist()) if "Sector" in wdf.columns else []
    sector_filter  = fcols[3].multiselect("Sector", _all_sectors, default=_all_sectors, placeholder="All sectors")
    search         = fcols[4].text_input("Search ticker or sub-industry")

    # --- Range filters (numeric columns) ---
    _WL_RF_DEFS = [
        # (column,          label,              step)
        ("IAS",            "IAS Score",         1.0),
        ("CompRS",         "CompRS",            0.01),
        ("RS21",           "RS21 %",            1.0),
        ("RS63",           "RS63 %",            1.0),
        ("Off_MA50",       "Off MA50 %",        1.0),
        ("Off_52W_High",   "Off 52W High %",    1.0),
        ("V21_CRS_Gap",    "V21 CRS Gap",       0.01),
        ("V21_MA50_Gap",   "V21 MA50 Gap %",    1.0),
        ("Liq5Cr",         "Liq 5Cr ₹Cr",      5.0),
        ("LiqFromLow",     "Liq From Low ×",    0.5),
        ("VolQuality",     "Vol Quality",       0.05),
    ]

    with st.expander("🎚️ Range Filters", expanded=False):
        import math as _wl_math
        _wl_range_vals: dict = {}
        _wl_rf_avail = [(c, lbl, step) for c, lbl, step in _WL_RF_DEFS if c in wdf.columns]
        _WL_N_COLS = 4
        for _ri in range(0, len(_wl_rf_avail), _WL_N_COLS):
            _row = _wl_rf_avail[_ri: _ri + _WL_N_COLS]
            _rcols = st.columns(_WL_N_COLS)
            for _j, (col, lbl, step) in enumerate(_row):
                _s = pd.to_numeric(wdf[col], errors="coerce").dropna()
                if len(_s) < 1:
                    continue
                _lo = _wl_math.floor(float(_s.min()) / step) * step
                _hi = _wl_math.ceil( float(_s.max()) / step) * step
                if _hi <= _lo:
                    _hi = _lo + step
                with _rcols[_j]:
                    _sel = st.slider(lbl, min_value=_lo, max_value=_hi,
                                     value=(_lo, _hi), step=step, key=f"wl_rf_{col}")
                    _wl_range_vals[col] = _sel

    fdf = wdf[
        wdf["Tier"].isin(tier_filter) &
        wdf["Cycle"].isin(cycle_filter) &
        (wdf["IAS"] >= min_ias) &
        (wdf["Sector"].isin(sector_filter) if sector_filter else pd.Series(True, index=wdf.index))
    ].copy()

    # Apply numeric range filters
    for _col, (_lo_sel, _hi_sel) in _wl_range_vals.items():
        _num = pd.to_numeric(fdf[_col], errors="coerce")
        fdf = fdf[_num.between(_lo_sel, _hi_sel, inclusive="both") | _num.isna()]

    if search:
        fdf = fdf[
            fdf["Ticker"].str.contains(search.upper(), na=False) |
            fdf["Sub_Industry"].str.contains(search, case=False, na=False)
        ]

    # --- Lifecycle explainer ---
    with st.expander("ℹ️ How stocks enter, graduate & drop from the IAS watchlist"):
        st.markdown("""
    **Entry — IAS ≥ 35, price below MA50**

    A stock enters the watchlist when `IAS ≥ 35`. IAS (Institutional Accumulation Score) combines
    RS velocity, liquidity surge from the low, and volume quality — all must confirm simultaneous
    early buying *while the stock is still below MA50* (not yet a V21 breakout, just radar ping).

    | Tier | IAS range | Meaning |
    |---|---|---|
    | 🔵 WATCH | 35–59 | Early institutional ping — set a price alert at MA50 |
    | 🟡 READY | 60–79 | RS velocity confirmed, liquidity floor stable — getting warm |
    | 🟢 ALERT | 80+ | RS21 + RS63 both turning, strong liq floor — imminent V21, ready to fire |

    **Graduation → GRADUATED (locked permanently)**

    When **both** conditions are met on any screener run:
    1. `Off_MA50 ≥ 0` — price has crossed *above* the 50-day MA
    2. `CompRS > 0` — stock is outperforming the benchmark

    This is the V21 breakout trigger. The stock is logged as GRADUATED in the IAS Signal Log
    and promoted to the main portfolio scanner. Status can **never** be reverted — once a grad, always a grad.

    **V21 CRS Gap & V21 MA50 Gap** tell you exactly how far each stock is from graduating:
    - `V21 CRS Gap = 0` → CompRS is already positive (half-done)
    - `V21 MA50 Gap % = 0` → price is at MA50 (half-done)
    - Both zero → next run may graduate it

    **Drop → DROPPED**

    When the stock disappears from the daily scan (IAS falls below 35 — accumulation signal evaporated)
    **and** it hasn't already GRADUATED. Typically means the smart-money move reversed or was a false signal.
    DROPPED stocks are tracked in the Signal Log for post-mortem analysis.

    **Typical holding time:** 2–12 weeks from ALERT to graduation. WATCH stocks can take months.
        """)

    # --- Watchlist table ---
    st.markdown(f"### Watchlist ({len(fdf)} stocks)")

    # Build screener.in links using the same pattern as other tabs
    display_df = fdf.copy().sort_values("IAS", ascending=False).reset_index(drop=True)
    display_df["screener_link"] = "https://www.screener.in/company/" + display_df["Ticker"].str.replace(r"\.(NS|BO)$", "", regex=True) + "/"

    display_cols = ["screener_link","Sub_Industry","Cycle","CMP","Off_52W_High","RS21","RS63",
                    "CompRS","Liq5Cr","LiqFromLow","VolQuality","IAS","Tier","Off_MA50","V21_CRS_Gap","V21_MA50_Gap"]
    available = [c for c in display_cols if c in display_df.columns]

    st.dataframe(
        display_df[available],
        column_config={
            "screener_link": st.column_config.LinkColumn("Ticker", display_text=r"https://www\.screener\.in/company/(.*?)/", width=110),
            "Sub_Industry": st.column_config.TextColumn("Sub-Industry", width=160),
            "Cycle":        st.column_config.TextColumn("Cycle", width=55),
            "CMP":          st.column_config.NumberColumn("CMP", format="%.1f"),
            "Off_52W_High": st.column_config.NumberColumn("Off52W%", format="%.1f"),
            "Off_MA50":     st.column_config.NumberColumn("OffMA50%", format="%.1f"),
            "RS21":         st.column_config.NumberColumn("RS21", format="%.1f"),
            "RS63":         st.column_config.NumberColumn("RS63", format="%.1f"),
            "CompRS":       st.column_config.NumberColumn("CompRS", format="%.3f"),
            "Liq5Cr":       st.column_config.NumberColumn("Liq5Cr", format="%.0f"),
            "LiqFromLow":   st.column_config.NumberColumn("LiqFromLow", format="%.1fx"),
            "VolQuality":   st.column_config.NumberColumn("Vol Quality", format="%.2f", help=">0.55 = Valid Accumulation"),
            "IAS":          st.column_config.ProgressColumn("IAS", format="%.0f", max_value=100),
            "Tier":         st.column_config.TextColumn("Tier", width=60),
            "V21_CRS_Gap":  st.column_config.NumberColumn("V21 CRS Gap", format="%.2f"),
            "V21_MA50_Gap": st.column_config.NumberColumn("V21 MA50 Gap%", format="%.1f"),
        },
        use_container_width=True,
        hide_index=True,
        height=600,
    )

    # --- IAS distribution + Sub-industry breakdown ---
    st.markdown("---")
    st.markdown("### Signal Analytics")
    dcols = st.columns([2, 1])

    with dcols[0]:
        st.markdown("**IAS Score Distribution**")
        fig_hist = go.Figure()
        for tier, color in TIER_COLORS.items():
            sub = wdf[wdf["Tier"] == tier]["IAS"]
            if not sub.empty:
                fig_hist.add_trace(go.Histogram(
                    x=sub, name=tier, marker_color=color, opacity=0.8,
                    xbins=dict(size=5)
                ))
        fig_hist.update_layout(
            barmode="stack", xaxis_title="IAS Score", yaxis_title="Stocks",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"), legend=dict(orientation="h"),
            height=280, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_hist, use_container_width=True, key="ias_hist_radar")

    with dcols[1]:
        st.markdown("**Top Sub-Industries**")
        si_counts = wdf["Sub_Industry"].value_counts().head(12).reset_index()
        si_counts.columns = ["Sub-Industry", "Count"]
        st.dataframe(si_counts, hide_index=True, use_container_width=True, height=280)

    # --- V21 Pipeline ---
    st.markdown("---")
    st.markdown("### V21 Pipeline (ALERT + READY tier)")
    st.caption("% below MA50 each stock must close to trigger V21 entry. Smaller bar = closer to qualification.")

    pipe_df = wdf[wdf["Tier"].isin(["ALERT","READY"])].copy()
    if not pipe_df.empty and "V21_MA50_Gap" in pipe_df.columns:
        pipe_df = pipe_df.sort_values("V21_MA50_Gap").head(30)
        fig_pipe = go.Figure()
        fig_pipe.add_trace(go.Bar(
            x=pipe_df["Ticker"].str.replace(".NS","", regex=False),
            y=pipe_df["V21_MA50_Gap"],
            marker_color=[TIER_COLORS.get(t, "#ccc") for t in pipe_df["Tier"]],
            text=[f"{t} | IAS {v}" for t, v in zip(pipe_df["Tier"], pipe_df["IAS"])],
            textposition="outside",
        ))
        fig_pipe.update_layout(
            xaxis_title="Stock", yaxis_title="% Below MA50",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"), height=340,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_pipe, use_container_width=True, key="pipeline_bar_radar")
    else:
        st.info("No ALERT or READY stocks today.")

    if "Date" in wdf.columns and len(wdf):
        st.caption(f"Last run: {wdf['Date'].iloc[0]} | Run turnaround_screener.py or wait for GitHub Actions.")

    # ── IAS Signal Log ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📓 IAS Signal Log — Multibagger Tracker")

    LOG_CSV_PATH = "data/ias_signal_log.csv"
    if not os.path.exists(LOG_CSV_PATH):
        st.info("No signal log yet. The log is created automatically after the next screener run.")
    else:
        log_df = pd.read_csv(LOG_CSV_PATH)

        # ── Live price refresh ────────────────────────────────────────────────
        # The screener only runs daily via GitHub Actions; refresh current_price
        # here so the tracker always shows today's prices, not the stale CSV value.

        # Fast path: use prices already loaded in the Trend Scanner session state
        _md_sess = st.session_state.get('market_data', pd.DataFrame())
        _md_prices: dict = {}
        if (isinstance(_md_sess, pd.DataFrame) and not _md_sess.empty
                and 'ticker' in _md_sess.columns and 'price' in _md_sess.columns):
            _md_prices = (
                _md_sess.dropna(subset=['price'])
                        .set_index('ticker')['price']
                        .to_dict()
            )

        _log_tickers_all = log_df["ticker"].dropna().unique().tolist()
        _need_yf = tuple(t for t in _log_tickers_all if t not in _md_prices)

        _col_btn, _col_info = st.columns([1, 4])
        with _col_btn:
            if st.button("🔄 Refresh Prices", key="turnaround_refresh_prices"):
                st.cache_data.clear()
                st.rerun()
        with _col_info:
            _src_note = (
                f"Prices from session data ({len(_md_prices)} tickers) "
                + (f"+ live fetch ({len(_need_yf)} tickers)" if _need_yf else "")
            )
            st.caption(_src_note)

        with st.spinner("Fetching live prices for signal log…"):
            _yf_prices = _fetch_turnaround_live_prices(_need_yf)

        _all_live: dict = {**_yf_prices, **_md_prices}  # md_prices wins (fresher)

        # Vectorised update of current_price, return_since_signal, max_gain, xirr
        log_df["_live"] = log_df["ticker"].map(_all_live)
        _sig_p = pd.to_numeric(log_df["signal_price"], errors="coerce")
        _live_p = pd.to_numeric(log_df["_live"], errors="coerce")
        _has = _live_p.notna() & (_live_p > 0) & _sig_p.notna() & (_sig_p > 0)

        if _has.any():
            log_df.loc[_has, "current_price"] = _live_p[_has].round(1)
            _ret = ((_live_p[_has] / _sig_p[_has]) - 1) * 100
            log_df.loc[_has, "return_since_signal"] = _ret.round(1)
            _prev_max = pd.to_numeric(log_df.loc[_has, "max_gain"], errors="coerce").fillna(0)
            log_df.loc[_has, "max_gain"] = _ret.where(_ret > _prev_max, _prev_max).round(1)

            # XIRR (row-by-row: depends on days elapsed per signal_date)
            _today = datetime.now()
            for _idx in log_df.index[_has]:
                try:
                    _days = (_today - datetime.strptime(
                        str(log_df.at[_idx, "signal_date"])[:10], "%Y-%m-%d")).days
                    _sp = float(log_df.at[_idx, "signal_price"])
                    _cp = float(log_df.at[_idx, "_live"])
                    if _days >= 1 and _sp > 0 and _cp > 0:
                        log_df.at[_idx, "xirr"] = round(
                            ((_cp / _sp) ** (365.0 / _days) - 1) * 100, 1)
                except Exception:
                    pass

        log_df = log_df.drop(columns=["_live"])

        # ── Categorical filters + sort ────────────────────────────────────────
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            status_filter = st.multiselect(
                "Status", ["ACTIVE", "GRADUATED", "DROPPED"],
                default=["ACTIVE", "GRADUATED"],
                key="log_status_filter"
            )
        with col_f2:
            tier_filter = st.multiselect(
                "Peak Tier", ["ALERT", "READY", "WATCH"],
                default=["ALERT", "READY", "WATCH"],
                key="log_tier_filter"
            )
        with col_f3:
            sort_by = st.selectbox(
                "Sort by",
                ["xirr", "return_since_signal", "max_gain", "peak_ias", "signal_date"],
                index=0, key="log_sort_by"
            )

        # ── Range filters ─────────────────────────────────────────────────────
        _RF_DEFS = [
            # (column,                label,                  step)
            ("signal_ias",          "IAS (signal)",           1.0),
            ("signal_rs21",         "RS21 (signal)",          1.0),
            ("signal_rs63",         "RS63 (signal)",          1.0),
            ("signal_comp_rs",      "CompRS (signal)",        0.01),
            ("signal_off_52w_high", "Off 52W High %",         1.0),
            ("signal_off_ma50",     "Off MA50 %",             1.0),
            ("signal_off_ma200",    "Off MA200 %",            1.0),
            ("signal_liq5cr",       "Liq5Cr ₹Cr (signal)",   5.0),
            ("signal_liq_from_low", "Liq From Low (signal)",  0.5),
            ("signal_vol_quality",  "Vol Quality (signal)",   0.05),
            ("signal_shock_ratio",  "Shock Ratio (signal)",   0.05),
            ("peak_ias",            "Peak IAS",               1.0),
            ("days_on_watchlist",   "Days on Watchlist",      1.0),
            ("return_since_signal", "Return Since Signal %",  5.0),
            ("max_gain",            "Max Gain %",             5.0),
            ("xirr",                "XIRR / CAGR %",         5.0),
            ("return_5d",           "Return 5d %",            1.0),
            ("return_21d",          "Return 21d %",           1.0),
            ("return_63d",          "Return 63d %",           1.0),
        ]

        with st.expander("🎚️ Range Filters", expanded=False):
            import math as _math
            _range_vals: dict = {}
            rf_avail = [(c, lbl, step) for c, lbl, step in _RF_DEFS if c in log_df.columns]
            _N_RF_COLS = 4
            for _ri in range(0, len(rf_avail), _N_RF_COLS):
                _row = rf_avail[_ri: _ri + _N_RF_COLS]
                _rcols = st.columns(_N_RF_COLS)
                for _j, (col, lbl, step) in enumerate(_row):
                    _s = pd.to_numeric(log_df[col], errors="coerce").dropna()
                    if len(_s) < 1:
                        continue
                    _lo = _math.floor(float(_s.min()) / step) * step
                    _hi = _math.ceil( float(_s.max()) / step) * step
                    if _hi <= _lo:
                        _hi = _lo + step
                    with _rcols[_j]:
                        _sel = st.slider(
                            lbl, min_value=_lo, max_value=_hi,
                            value=(_lo, _hi), step=step, key=f"rf_{col}"
                        )
                        _range_vals[col] = _sel

        # ── Build combined filter mask ────────────────────────────────────────
        mask = pd.Series([True] * len(log_df))
        if status_filter:
            mask &= log_df["status"].isin(status_filter)
        if tier_filter:
            mask &= log_df["peak_tier"].isin(tier_filter)
        for _col, (_lo_sel, _hi_sel) in _range_vals.items():
            _num = pd.to_numeric(log_df[_col], errors="coerce")
            # rows with NaN are kept (they haven't been measured yet)
            mask &= (_num.between(_lo_sel, _hi_sel, inclusive="both") | _num.isna())

        filtered = log_df[mask].copy()

        if sort_by in filtered.columns:
            ascending = sort_by == "signal_date"
            filtered = filtered.sort_values(sort_by, ascending=ascending, na_position="last")

        # ── Summary KPIs ──────────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Logged", len(log_df))
        k2.metric("Active", int((log_df["status"] == "ACTIVE").sum()))
        k3.metric("Graduated", int((log_df["status"] == "GRADUATED").sum()))
        k4.metric("Dropped", int((log_df["status"] == "DROPPED").sum()))
        multibaggers = int((pd.to_numeric(log_df["return_since_signal"], errors="coerce") >= 100).sum())
        k5.metric("Multibaggers (2×+)", multibaggers)

        # ── Main table ────────────────────────────────────────────────────────
        DISPLAY_COLS = [
            "ticker", "name", "sub_industry", "cycle", "status",
            "signal_date", "signal_tier", "signal_price",
            "signal_ias", "signal_rs21", "signal_rs63", "signal_comp_rs",
            "signal_off_52w_high", "signal_off_ma50",
            "peak_tier", "peak_ias", "days_on_watchlist",
            "return_5d", "return_21d", "return_63d",
            "current_price", "return_since_signal", "max_gain", "xirr",
        ]
        show_cols = [c for c in DISPLAY_COLS if c in filtered.columns]
        st.dataframe(
            filtered[show_cols].reset_index(drop=True),
            use_container_width=True,
            height=420,
        )

        # ── Multibagger spotlight ──────────────────────────────────────────────
        mb_df = log_df[pd.to_numeric(log_df["return_since_signal"], errors="coerce") >= 100].copy()
        if not mb_df.empty:
            st.markdown("#### 🚀 Multibagger Board (2×+ returns)")
            mb_show = [c for c in ["ticker", "name", "signal_date", "signal_price",
                                    "current_price", "return_since_signal", "max_gain",
                                    "xirr", "status", "days_on_watchlist"] if c in mb_df.columns]
            st.dataframe(
                mb_df[mb_show].sort_values("return_since_signal", ascending=False).reset_index(drop=True),
                use_container_width=True,
            )

        # ── Forward return distribution ────────────────────────────────────────
        with st.expander("📊 Forward Return Distribution (5d / 21d / 63d)"):
            ret_cols = ["return_5d", "return_21d", "return_63d"]
            avail = [c for c in ret_cols if c in log_df.columns and log_df[c].notna().any()]
            if avail:
                import plotly.graph_objects as go
                fig_ret = go.Figure()
                colours = {"return_5d": "#4fc3f7", "return_21d": "#81c784", "return_63d": "#ffb74d"}
                labels  = {"return_5d": "5-day", "return_21d": "21-day", "return_63d": "63-day"}
                for col in avail:
                    vals = pd.to_numeric(log_df[col], errors="coerce").dropna()
                    fig_ret.add_trace(go.Histogram(
                        x=vals, name=labels[col],
                        marker_color=colours[col], opacity=0.7, nbinsx=20,
                    ))
                fig_ret.update_layout(
                    barmode="overlay", height=280,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e0e0e0"),
                    xaxis_title="Return (%)", yaxis_title="Count",
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                fig_ret.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.6)
                st.plotly_chart(fig_ret, use_container_width=True, key="log_fwd_ret_hist")

                # Summary table
                summary_rows = []
                for col in avail:
                    vals = pd.to_numeric(log_df[col], errors="coerce").dropna()
                    summary_rows.append({
                        "Period": labels[col], "N": len(vals),
                        "Avg %": round(vals.mean(), 1),
                        "Median %": round(vals.median(), 1),
                        "Win Rate %": round((vals > 0).mean() * 100, 1),
                        "Max %": round(vals.max(), 1),
                        "Min %": round(vals.min(), 1),
                    })
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            else:
                st.info("Not enough data yet — forward returns will populate once signals age past 5/21/63 trading days.")

        # ── Full detail expander ───────────────────────────────────────────────
        with st.expander("🔬 Full Signal Detail (all columns)"):
            st.dataframe(filtered.reset_index(drop=True), use_container_width=True, height=360)
