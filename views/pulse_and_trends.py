"""Morning Pulse and Time Trends page views (extracted from main.py)."""
from __future__ import annotations

import glob as _glob
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

from utils.data_engine import get_stock_history
from utils.nifty500_list import TICKERS
from utils.ui_components import page_header
from utils.visuals import chart_relative_performance


@st.cache_data(ttl=1800, show_spinner=False)
def _load_pulse_data():
    files = sorted(_glob.glob("data/cache/market_master_*.parquet"))
    if len(files) < 2:
        return None
    dfs, labels = [], []
    for f in files[-7:]:
        # A parquet can be unreadable if it was written mid-crash by the
        # engine, or caught mid-sync during a Streamlit Cloud redeploy.
        # One bad day's file shouldn't take down the whole Pulse page —
        # skip it and keep the rolling window from the rest.
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            print(f"[Morning Pulse] Skipping unreadable cache file {f}: {e}", flush=True)
            continue
        if "rs_1m" in df.columns and "return_1m" not in df.columns:
            df["return_1m"] = df["rs_1m"]
        if "rs_1w" in df.columns and "return_1w" not in df.columns:
            df["return_1w"] = df["rs_1w"]
        dfs.append(df)
        labels.append(f.split("market_master_")[1].replace(".parquet","").replace("_","-"))
    if len(dfs) < 2:
        return None
    return dfs, labels


def render_morning_pulse(df: pd.DataFrame) -> None:

    st.markdown(page_header("☀️ Morning Pulse", "Daily market intelligence from parquet cache — breadth, rotation, catalysts, accumulation"), unsafe_allow_html=True)

    from utils import regime_badge as _rb
    _rb.render(st.session_state.get('regime_info') or _rb.compute(st.session_state.get('nifty_data'), df), st)

    import plotly.graph_objects as _pgo

    _pulse = _load_pulse_data()
    if _pulse is None:
        st.warning("No parquet cache found. Run the trading engine first.")
    else:
        _dfs, _labels = _pulse
        _today  = _dfs[-1]
        _prev   = _dfs[-2]
        _d2     = _dfs[-3] if len(_dfs) >= 3 else _dfs[-2]
        _d5     = _dfs[-6] if len(_dfs) >= 6 else _dfs[0]
        _today_label = _labels[-1]

        # ── SECTION 1: MARKET BREADTH ─────────────────────────────────────────
        st.markdown("### 🌡️ Market Breadth")

        def _breadth(df):
            ts = df["trend_score"].dropna()
            rs = df["return_1m"].dropna() if "return_1m" in df.columns else pd.Series(dtype=float)
            above50 = (df.get("dist_200dma", pd.Series(dtype=float)).dropna() > -20).sum() / max(len(df),1) * 100
            return dict(
                avg_ts=round(ts.mean(),1), med_ts=round(ts.median(),1),
                pct60=round((ts>=60).sum()/max(len(ts),1)*100,1),
                pct40=round((ts>=40).sum()/max(len(ts),1)*100,1),
                pct30=round((ts<30).sum()/max(len(ts),1)*100,1),
                rs_p75=round(rs.quantile(0.75),1) if len(rs) else 0,
                rs_p50=round(rs.quantile(0.50),1) if len(rs) else 0,
                rs_p25=round(rs.quantile(0.25),1) if len(rs) else 0,
                n=len(ts),
            )

        _b0 = _breadth(_today)
        _b1 = _breadth(_prev)
        _b2 = _breadth(_d2)

        def _arrow(val, ref, good_up=True):
            delta = val - ref
            if abs(delta) < 0.3:
                return "→"
            if good_up:
                return "🟢 ↑" if delta > 0 else "🔴 ↓"
            return "🟢 ↓" if delta < 0 else "🔴 ↑"

        _bc1, _bc2, _bc3, _bc4 = st.columns(4)
        with _bc1:
            st.metric("Avg Trend Score", f"{_b0['avg_ts']}", f"{_b0['avg_ts']-_b2['avg_ts']:+.1f} vs D-2")
        with _bc2:
            st.metric("% TS ≥ 60 (Strong)", f"{_b0['pct60']}%", f"{_b0['pct60']-_b2['pct60']:+.1f}%")
        with _bc3:
            st.metric("% TS < 30 (Weak)", f"{_b0['pct30']}%", f"{_b0['pct30']-_b2['pct30']:+.1f}%", delta_color="inverse")
        with _bc4:
            st.metric("RS21 Median", f"{_b0['rs_p50']:+.1f}%", f"{_b0['rs_p50']-_b2['rs_p50']:+.1f}%")

        _mb1, _mb2 = st.columns(2)
        with _mb1:
            _breadth_rows = {
                "Avg Trend Score":   [_b2["avg_ts"],  _b1["avg_ts"],  _b0["avg_ts"]],
                "Median TS":         [_b2["med_ts"],  _b1["med_ts"],  _b0["med_ts"]],
                "% TS ≥ 60":         [_b2["pct60"],   _b1["pct60"],   _b0["pct60"]],
                "% TS ≥ 40":         [_b2["pct40"],   _b1["pct40"],   _b0["pct40"]],
                "% TS < 30 (weak)":  [_b2["pct30"],   _b1["pct30"],   _b0["pct30"]],
                "RS21 P75":          [_b2["rs_p75"],  _b1["rs_p75"],  _b0["rs_p75"]],
                "RS21 P50 (median)": [_b2["rs_p50"],  _b1["rs_p50"],  _b0["rs_p50"]],
                "RS21 P25":          [_b2["rs_p25"],  _b1["rs_p25"],  _b0["rs_p25"]],
            }
            _bdf = pd.DataFrame(_breadth_rows, index=[_labels[-3], _labels[-2], _labels[-1]]).T
            _bdf["Slope (D-2)"] = (_bdf[_labels[-1]] - _bdf[_labels[-3]]).round(1)
            st.caption("Breadth Table (D-2 → Today)")
            st.dataframe(_bdf, use_container_width=True)

        with _mb2:
            # Breadth history chart
            _mb_hist = pd.read_csv("data/market_breadth_history.csv") if os.path.exists("data/market_breadth_history.csv") else pd.DataFrame()
            if not _mb_hist.empty:
                _mb_hist["date"] = pd.to_datetime(_mb_hist["date"])
                _mb_hist = _mb_hist.sort_values("date").tail(30)
                _fig_mb = _pgo.Figure()
                _fig_mb.add_trace(_pgo.Scatter(x=_mb_hist["date"], y=_mb_hist["pct_uptrends"], name="% TS≥60", line=dict(color="#00F0FF")))
                _fig_mb.add_trace(_pgo.Scatter(x=_mb_hist["date"], y=100-_mb_hist["pct_uptrends"], name="% TS<30", line=dict(color="#FF4B4B", dash="dash")))
                _fig_mb.update_layout(height=220, margin=dict(l=0,r=0,t=20,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#333"))
                st.caption("30-day Breadth Trend")
                st.plotly_chart(_fig_mb, use_container_width=True)

        st.markdown("---")

        # ── SECTION 2: SECTOR ROTATION ────────────────────────────────────────
        st.markdown("### 🔄 Sector Rotation")

        _sec_t  = _today.groupby("sector")["trend_score"].mean().round(1)
        _sec_p  = _prev.groupby("sector")["trend_score"].mean().round(1)
        _sec_d2 = _d2.groupby("sector")["trend_score"].mean().round(1)
        _sec_d5 = _d5.groupby("sector")["trend_score"].mean().round(1)
        _sec_n  = _today.groupby("sector")["ticker"].count()
        _sec_acc = _today.groupby("sector").apply(
            lambda g: round((g.get("volume_signal_text", pd.Series(dtype=str)) == "ACCUMULATION").sum() / max(len(g),1) * 100, 0)
        ) if "volume_signal_text" in _today.columns else pd.Series(dtype=float)

        _sec_df = pd.DataFrame({
            "TS Today": _sec_t,
            "TS Prev":  _sec_p,
            "TS D-2":   _sec_d2,
            "1D Slope": (_sec_t - _sec_p).round(1),
            "5D Slope": (_sec_t - _sec_d5).round(1),
            "Stocks":   _sec_n,
        }).dropna(subset=["TS Today"]).sort_values("TS Today", ascending=False)

        if not _sec_acc.empty:
            _sec_df["Accum%"] = _sec_acc

        def _slope_arrow(v):
            if v >= 3: return f"🟢 +{v:.1f}"
            if v >= 1: return f"↑ +{v:.1f}"
            if v > -1: return f"→ {v:.1f}"
            if v > -3: return f"↓ {v:.1f}"
            return f"🔴 {v:.1f}"

        _sec_df["1D"] = _sec_df["1D Slope"].apply(_slope_arrow)
        _sec_df["5D"] = _sec_df["5D Slope"].apply(_slope_arrow)

        _src1, _src2 = st.columns([3, 2])
        with _src1:
            st.caption("Sector TS with momentum arrows (1D / 5D slope)")
            st.dataframe(
                _sec_df[["TS Today", "1D Slope", "5D Slope", "Stocks"] + (["Accum%"] if "Accum%" in _sec_df.columns else [])],
                use_container_width=True,
                column_config={
                    "TS Today":  st.column_config.NumberColumn("TS Today",  format="%.0f"),
                    "1D Slope":  st.column_config.NumberColumn("1D Slope",  format="%+.1f"),
                    "5D Slope":  st.column_config.NumberColumn("5D Slope",  format="%+.1f"),
                    "Stocks":    st.column_config.NumberColumn("N",         format="%.0f"),
                    "Accum%":    st.column_config.NumberColumn("Accum%",    format="%.0f%%"),
                },
                height=420,
            )
        with _src2:
            # Bar chart: 5D slope ranked
            _slope_sorted = _sec_df["5D Slope"].sort_values()
            _colors = ["#FF4B4B" if v < 0 else "#00F0FF" for v in _slope_sorted.values]
            _fig_sec = _pgo.Figure(_pgo.Bar(
                x=_slope_sorted.values, y=_slope_sorted.index,
                orientation="h", marker_color=_colors,
            ))
            _fig_sec.update_layout(height=420, margin=dict(l=0,r=0,t=20,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="#333", zeroline=True, zerolinecolor="#666"),
                yaxis=dict(showgrid=False))
            st.caption("5D Sector TS Slope")
            st.plotly_chart(_fig_sec, use_container_width=True)

        st.markdown("---")

        # ── SECTION 3: TODAY'S TS MOVERS ────────────────────────────────────
        st.markdown("### 📈 Today's TS Movers")

        _merged = _today.set_index("ticker")[["trend_score","change_p","volume_score","dist_52w","name","sector"]].join(
            _prev.set_index("ticker")[["trend_score"]].rename(columns={"trend_score":"ts_prev"}), how="inner"
        )
        _merged["ts_gain"] = (_merged["trend_score"] - _merged["ts_prev"]).round(0)
        _merged["vol_signal"] = _today.set_index("ticker").get("volume_signal_text", pd.Series(dtype=str))

        _mv1, _mv2 = st.columns(2)

        with _mv1:
            st.markdown("#### 🟢 Top TS Gainers — Turnaround Candidates")
            _gainers = (_merged[_merged["ts_gain"] > 0]
                        .sort_values("ts_gain", ascending=False).head(12))
            if not _gainers.empty:
                _g_display = _gainers.copy().reset_index()
                _g_display["screener_link"] = "https://www.screener.in/company/" + _g_display["ticker"].str.replace(".NS","",regex=False) + "/"
                _g_display["Pattern"] = _g_display.apply(
                    lambda r: "A+B" if (r["change_p"] >= 5 and r["volume_score"] >= 7 and r["ts_gain"] >= 20)
                    else ("A" if (r["change_p"] >= 5 and r["volume_score"] >= 7)
                    else ("B" if (r["ts_gain"] >= 20 and r["volume_score"] >= 6 and r["change_p"] >= 2.5)
                    else "")), axis=1
                )
                st.dataframe(
                    _g_display[["screener_link","name","sector","change_p","volume_score","dist_52w","ts_prev","trend_score","ts_gain","Pattern"]],
                    column_config={
                        "screener_link":  st.column_config.LinkColumn("Ticker", display_text=r"https://www\.screener\.in/company/(.*?)/"),
                        "name":           st.column_config.TextColumn("Name"),
                        "sector":         st.column_config.TextColumn("Sector"),
                        "change_p":       st.column_config.NumberColumn("Day %",     format="%+.1f%%"),
                        "volume_score":   st.column_config.NumberColumn("Vol",       format="%.0f /10"),
                        "dist_52w":       st.column_config.NumberColumn("Dist 52W",  format="%.0f%%"),
                        "ts_prev":        st.column_config.NumberColumn("TS Prev",   format="%.0f"),
                        "trend_score":    st.column_config.NumberColumn("TS Now",    format="%.0f"),
                        "ts_gain":        st.column_config.NumberColumn("TS Gain",   format="+%.0f"),
                        "Pattern":        st.column_config.TextColumn("Pattern"),
                    },
                    hide_index=True, use_container_width=True, height=380,
                )
            else:
                st.info("No TS gainers today.")

        with _mv2:
            st.markdown("#### 🔴 Top TS Losers — Caution / Exit Watch")
            _losers = (_merged[_merged["ts_gain"] < 0]
                       .sort_values("ts_gain").head(12))
            if not _losers.empty:
                _l_display = _losers.copy().reset_index()
                _l_display["screener_link"] = "https://www.screener.in/company/" + _l_display["ticker"].str.replace(".NS","",regex=False) + "/"
                st.dataframe(
                    _l_display[["screener_link","name","sector","change_p","volume_score","ts_prev","trend_score","ts_gain"]],
                    column_config={
                        "screener_link":  st.column_config.LinkColumn("Ticker", display_text=r"https://www\.screener\.in/company/(.*?)/"),
                        "name":           st.column_config.TextColumn("Name"),
                        "sector":         st.column_config.TextColumn("Sector"),
                        "change_p":       st.column_config.NumberColumn("Day %",   format="%+.1f%%"),
                        "volume_score":   st.column_config.NumberColumn("Vol",     format="%.0f /10"),
                        "ts_prev":        st.column_config.NumberColumn("TS Prev", format="%.0f"),
                        "trend_score":    st.column_config.NumberColumn("TS Now",  format="%.0f"),
                        "ts_gain":        st.column_config.NumberColumn("TS Gain", format="%.0f"),
                    },
                    hide_index=True, use_container_width=True, height=380,
                )
            else:
                st.info("No TS losers today.")

        st.markdown("---")

        # ── SECTION 4: INSTITUTIONAL ACCUMULATION ────────────────────────────
        st.markdown("### 🏦 Institutional Accumulation Leaders")
        st.caption("Vol score ≥ 8 + positive day change — institutional fingerprint")

        if "volume_signal_text" in _today.columns:
            _accum = _today[
                (_today["volume_score"] >= 8) &
                (_today["change_p"] > 0)
            ].copy()
            if "return_1m" in _accum.columns:
                pass
            _accum = _accum.sort_values("volume_score", ascending=False)

            _ac1, _ac2 = st.columns([3, 2])
            with _ac1:
                _accum_disp = _accum.head(15).copy()
                _accum_disp["screener_link"] = "https://www.screener.in/company/" + _accum_disp["ticker"].str.replace(".NS","",regex=False) + "/"
                _disp_cols = ["screener_link","name","sector","price","change_p","volume_score","dist_52w","trend_score"]
                _disp_cols += ["return_1m"] if "return_1m" in _accum_disp.columns else []
                st.dataframe(
                    _accum_disp[_disp_cols],
                    column_config={
                        "screener_link":  st.column_config.LinkColumn("Ticker", display_text=r"https://www\.screener\.in/company/(.*?)/"),
                        "name":           st.column_config.TextColumn("Name"),
                        "sector":         st.column_config.TextColumn("Sector"),
                        "price":          st.column_config.NumberColumn("Price",    format="₹%.1f"),
                        "change_p":       st.column_config.NumberColumn("Day %",    format="%+.1f%%"),
                        "volume_score":   st.column_config.NumberColumn("Vol Score",format="%.0f /10"),
                        "dist_52w":       st.column_config.NumberColumn("Dist 52W", format="%.0f%%"),
                        "trend_score":    st.column_config.NumberColumn("TS",       format="%.0f"),
                        "return_1m":      st.column_config.NumberColumn("RS21",     format="%+.1f%%"),
                    },
                    hide_index=True, use_container_width=True, height=400,
                )
            with _ac2:
                # Accumulation% by sector
                _sec_accum = _today.groupby("sector").apply(
                    lambda g: round((g["volume_signal_text"] == "ACCUMULATION").sum() / max(len(g),1) * 100, 0)
                ).sort_values(ascending=False)
                _sa_colors = ["#00F0FF" if v >= 60 else ("#FFD700" if v >= 40 else "#888") for v in _sec_accum.values]
                _fig_acc = _pgo.Figure(_pgo.Bar(
                    x=_sec_accum.values, y=_sec_accum.index,
                    orientation="h", marker_color=_sa_colors,
                    text=[f"{v:.0f}%" for v in _sec_accum.values],
                    textposition="outside",
                ))
                _fig_acc.update_layout(height=400, margin=dict(l=0,r=0,t=20,b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(range=[0,110], showgrid=False),
                    yaxis=dict(showgrid=False))
                st.caption("% ACCUMULATION signal by sector")
                st.plotly_chart(_fig_acc, use_container_width=True)
        else:
            st.info("Volume signal data not available in cache.")

        st.markdown("---")

        # ── SECTION 5: IAS ACTIVE SIGNALS ─────────────────────────────────────
        st.markdown("### 📋 Active Watchlist Signals (IAS)")

        try:
            _ias = pd.read_csv("data/ias_signal_log.csv")
            _active = _ias[_ias["status"] == "ACTIVE"].sort_values("signal_date", ascending=False)
            if not _active.empty:
                _is1, _is2, _is3 = st.columns(3)
                _alerts = _active[_active["signal_tier"] == "ALERT"]
                _readys = _active[_active["signal_tier"] == "READY"]
                _watchs = _active[_active["signal_tier"] == "WATCH"]
                for _col, _tier_df, _label, _color in [
                    (_is1, _alerts, "🚨 ALERT", "#FF4B4B"),
                    (_is2, _readys, "✅ READY", "#00C853"),
                    (_is3, _watchs, "👁️ WATCH", "#FFD700"),
                ]:
                    with _col:
                        st.markdown(f"**{_label} ({len(_tier_df)})**")
                        if not _tier_df.empty:
                            for _, _r in _tier_df.iterrows():
                                _ias_delta = _r.get("last_ias", _r["signal_ias"]) - _r["signal_ias"]
                                _rs_delta  = _r.get("last_rs21", _r["signal_rs21"]) - _r["signal_rs21"]
                                st.markdown(
                                    f"**{_r['ticker'].replace('.NS','')}** — "
                                    f"IAS {_r['signal_ias']:.0f}→{_r.get('last_ias',_r['signal_ias']):.0f} "
                                    f"({_ias_delta:+.0f})  "
                                    f"RS21 {_r['signal_rs21']:.1f}→{_r.get('last_rs21',_r['signal_rs21']):.1f}"
                                )
                        else:
                            st.caption("None")
            else:
                st.info("No active IAS signals.")
        except Exception:
            st.info("IAS signal log not available.")

        # ── SECTION 6: TC LOG RECENT ──────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔄 Recent Turnaround Catalyst Signals")
        try:
            _tc = pd.read_csv("data/tc_log.csv")
            if not _tc.empty:
                _tc["signal_date"] = pd.to_datetime(_tc["signal_date"])
                _tc7 = _tc[_tc["signal_date"] >= pd.Timestamp.now() - pd.Timedelta(days=7)].sort_values("signal_date", ascending=False)
                if not _tc7.empty:
                    _tc7_disp = _tc7.copy()
                    _tc7_disp["screener_link"] = "https://www.screener.in/company/" + _tc7_disp["ticker"].str.replace(".NS","",regex=False) + "/"
                    _tc7_disp["signal_date"] = _tc7_disp["signal_date"].dt.strftime("%Y-%m-%d")
                    st.dataframe(
                        _tc7_disp[["signal_date","screener_link","name","sector","pattern","jump_pct","vol_score","dist_52w","ts_pre","ts_now","ts_gain","rs21_vel","return_since_signal","status"]],
                        column_config={
                            "signal_date":         st.column_config.TextColumn("Date"),
                            "screener_link":       st.column_config.LinkColumn("Ticker", display_text=r"https://www\.screener\.in/company/(.*?)/"),
                            "name":                st.column_config.TextColumn("Name"),
                            "sector":              st.column_config.TextColumn("Sector"),
                            "pattern":             st.column_config.TextColumn("Pattern"),
                            "jump_pct":            st.column_config.NumberColumn("Day %",    format="%+.1f%%"),
                            "vol_score":           st.column_config.NumberColumn("Vol",      format="%.0f /10"),
                            "dist_52w":            st.column_config.NumberColumn("Dist 52W", format="%.0f%%"),
                            "ts_pre":              st.column_config.NumberColumn("TS Pre",   format="%.0f"),
                            "ts_now":              st.column_config.NumberColumn("TS Now",   format="%.0f"),
                            "ts_gain":             st.column_config.NumberColumn("TS Gain",  format="+%.0f"),
                            "rs21_vel":            st.column_config.NumberColumn("RS21 Vel", format="%+.1f"),
                            "return_since_signal": st.column_config.NumberColumn("Return",   format="%+.1f%%"),
                            "status":              st.column_config.TextColumn("Status"),
                        },
                        hide_index=True, use_container_width=True,
                    )
                else:
                    st.info("No TC signals in the last 7 days.")
            else:
                st.info("TC log is empty — will populate after next engine run.")
        except Exception:
            st.info("TC log not yet available.")


def render_time_trends(df: pd.DataFrame) -> None:
    from utils.trend_engine import calculate_sector_history, calculate_stock_trend_history
    from utils.visuals import chart_sector_rotation, chart_stock_cycle

    st.markdown(page_header("⏳ Time Travel Trends", "Visualize Market Cycles and Historical Momentum"), unsafe_allow_html=True)

    tab_sector, tab_stock = st.tabs(["🔄 Sector Cycles", "📈 Stock Trend History"])

    # 1. Sector Rotation Tab
    with tab_sector:
        st.markdown("### Sector Rotation: Relative Strength vs the Market")
        st.caption("Each sector's equal-weight index against the equal-weight market, built from the stored daily snapshots (since Feb 2026). Lines going UP are outperforming. Click a legend name to hide or isolate a sector.")
    
        with st.spinner("Calculating Sector Histories... (This may take a moment)"):
            sector_hist = calculate_sector_history(df)
        
        if not sector_hist.empty:
            fig_sector = chart_sector_rotation(sector_hist)
            if fig_sector:
                st.plotly_chart(fig_sector, use_container_width=True)
            else:
                st.warning("Insufficient data to chart sector rotation.")
        else:
            st.error("Failed to load historical sector data.")

        st.markdown("---")
        st.subheader("🔄 Sector Stock Leadership")
        st.caption("Compare performance of specific stocks within a sector to spot rotation.")
    
        # Sector Stock Comparison UI
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            # Get list of sectors
            all_sectors_tt = sorted(df['sector'].unique().tolist())
            compare_sector = st.selectbox("Select Sector", all_sectors_tt, key="comp_sector_sel")
        
        with sc2:
            # Get stocks in that sector
            sector_stocks = df[df['sector'] == compare_sector]['ticker'].tolist()
            # Default to top 5 by trend score if available, else first 5
            default_stocks = sector_stocks[:5]
            compare_tickers = st.multiselect("Select Stocks", sector_stocks, default=default_stocks, key="comp_ticker_sel")
        
        with sc3:
            compare_period = st.selectbox("Timeframe", ["1y", "6mo", "3mo", "1mo"], index=0, key="comp_period_sel")
    
        if compare_tickers:
            with st.spinner("Fetching stock histories..."):
                comp_data = {}
                for t in compare_tickers:
                    # Use cache if possible, or fetch
                    h = get_stock_history(t, period=compare_period)
                    if not h.empty:
                        comp_data[t] = h['Close']
            
                if comp_data:
                    st.plotly_chart(chart_relative_performance(comp_data), use_container_width=True)
                else:
                    st.warning("No historical data available for selected stocks.")

    # 2. Stock Cycle Tab
    with tab_stock:
        st.markdown("### Historical Momentum Analysis")
        st.caption("Reconstructs the 'Trend Score' for the past 2 years to visualize signals and cycles.")
    
        # === SECTOR-BASED STOCK SELECTION ===
        filter_col1, filter_col2 = st.columns([1, 2])
    
        with filter_col1:
            # Sector filter
            all_sectors = sorted(df['sector'].dropna().unique().tolist())
            sector_options = ["All Sectors"] + all_sectors
            selected_sector_for_trend = st.selectbox(
                "Filter by Sector", 
                sector_options, 
                key="trend_sector_filter"
            )
    
        with filter_col2:
            # Filter tickers based on sector
            if selected_sector_for_trend == "All Sectors":
                filtered_tickers = TICKERS
                filtered_df_for_trend = df
            else:
                filtered_df_for_trend = df[df['sector'] == selected_sector_for_trend]
                filtered_tickers = filtered_df_for_trend['ticker'].tolist()
        
            if not filtered_tickers:
                st.warning(f"No stocks found in {selected_sector_for_trend}")
                trend_ticker = None
            else:
                # Try to grab from session state if available
                quick_ticker = st.session_state.get('quick_dive_ticker')
                default_index = 0
                if quick_ticker and quick_ticker in filtered_tickers:
                    default_index = filtered_tickers.index(quick_ticker)
            
                # Create display options with name
                def format_ticker(ticker):
                    stock_row = df[df['ticker'] == ticker]
                    if not stock_row.empty:
                        name = stock_row['name'].values[0]
                        trend_score = stock_row['trend_score'].values[0]
                        return f"{ticker} - {name} (Trend: {trend_score:.0f})"
                    return ticker
            
                trend_ticker = st.selectbox(
                    "Select Stock for Cycle Analysis", 
                    filtered_tickers, 
                    index=default_index, 
                    format_func=format_ticker,
                    key="trend_ticker_select"
                )
    
        # Show sector stats if a specific sector is selected
        if selected_sector_for_trend != "All Sectors" and not filtered_df_for_trend.empty:
            st.markdown(f"#### 📊 {selected_sector_for_trend} - Quick Stats")
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            stat_col1.metric("Stocks in Sector", len(filtered_df_for_trend))
            stat_col2.metric("Avg Trend Score", f"{filtered_df_for_trend['trend_score'].mean():.0f}")
        
            # Top and bottom performers
            if len(filtered_df_for_trend) > 1:
                top_stock = filtered_df_for_trend.loc[filtered_df_for_trend['trend_score'].idxmax()]
                worst_stock = filtered_df_for_trend.loc[filtered_df_for_trend['trend_score'].idxmin()]
                stat_col3.metric("🏆 Strongest", top_stock['ticker'], f"{top_stock['trend_score']:.0f}")
                stat_col4.metric("📉 Weakest", worst_stock['ticker'], f"{worst_stock['trend_score']:.0f}")
    
        st.markdown("---")
        
        if trend_ticker:
            # View Mode Toggle
            view_mode = st.radio("Analysis Mode", ["Standard Trend Cycle", "Multi-Factor Analysis"], horizontal=True, label_visibility="collapsed")
        
            if view_mode == "Standard Trend Cycle":
                with st.spinner(f"Reconstructing Trend History for {trend_ticker}..."):
                    trend_df = calculate_stock_trend_history(trend_ticker)
            
                if not trend_df.empty:
                    # Stats
                    last_row = trend_df.iloc[-1]
                
                    # Show current status with color coding
                    signal = last_row['trend_signal']
                    score = last_row['trend_score']
                
                    if signal in ['STRONG UPTREND', 'UPTREND']:
                        signal_color = '#00C853'
                    elif signal == 'NEUTRAL':
                        signal_color = '#FFD600'
                    else:
                        signal_color = '#D50000'
                
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid {signal_color};">
                        <h4 style="margin: 0;">Current Status: <span style="color: {signal_color};">{signal}</span></h4>
                        <p style="margin: 5px 0 0 0; color: #888;">Trend Score: {score:.0f}/100</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                    fig_cycle = chart_stock_cycle(trend_df)
                    if fig_cycle:
                        st.plotly_chart(fig_cycle, use_container_width=True)
                else:
                    st.error(f"Could not load history for {trend_ticker}")
        
            else: # Multi-Factor Analysis
                from utils.score_history import calculate_historical_scores
                from utils.scoring import calculate_scores

            
                # Defined locally to avoid Streamlit caching issues with utils module
                def detect_divergences(df, price_col='Close', indicator_col='momentum_score_hist', window=5):
                    if df.empty or len(df) < window * 2:
                        return pd.DataFrame({'div_bull': pd.Series(dtype=float), 'div_bear': pd.Series(dtype=float)})

                    d = df.copy()
                    d['min_local'] = d[price_col].rolling(window=window, center=True).min()
                    d['max_local'] = d[price_col].rolling(window=window, center=True).max()
                
                    is_min = (d[price_col] == d['min_local'])
                    is_max = (d[price_col] == d['max_local'])
                
                    min_indices = d.index[is_min]
                    max_indices = d.index[is_max]
                
                    d['div_bull'] = np.nan
                    d['div_bear'] = np.nan
                
                    for i in range(1, len(min_indices)):
                        curr_idx = min_indices[i]
                        prev_idx = min_indices[i-1]
                    
                        p_curr = d.loc[curr_idx, price_col]
                        p_prev = d.loc[prev_idx, price_col]
                        i_curr = d.loc[curr_idx, indicator_col]
                        i_prev = d.loc[prev_idx, indicator_col]
                    
                        if p_curr < p_prev and i_curr > i_prev:
                             d.loc[curr_idx, 'div_bull'] = p_curr
                         
                    for i in range(1, len(max_indices)):
                        curr_idx = max_indices[i]
                        prev_idx = max_indices[i-1]
                    
                        p_curr = d.loc[curr_idx, price_col]
                        p_prev = d.loc[prev_idx, price_col]
                        i_curr = d.loc[curr_idx, indicator_col]
                        i_prev = d.loc[prev_idx, indicator_col]
                    
                        if p_curr > p_prev and i_curr < i_prev:
                            d.loc[curr_idx, 'div_bear'] = p_curr
                        
                    return d[['div_bull', 'div_bear']]

            
                # Fetch history for calculation
                hist_data = yf.Ticker(trend_ticker).history(period="2y")
            
                if not hist_data.empty:
                    with st.spinner("Calculating multi-factor score history..."):
                        # 1. Calculate Historical Technicals
                        full_hist = calculate_historical_scores(hist_data)
                    
                        # 2. Get Current Fundamental Scores (Context)
                        # Try to get from cache first
                        stock_data = df[df['ticker'] == trend_ticker]
                        if not stock_data.empty:
                             stock_data = stock_data.iloc[0].to_dict()
                             scores = calculate_scores(stock_data)
                        else:
                             # Fallback defaults
                             scores = {'quality': 5, 'value': 5, 'growth': 5, 'momentum': 5, 'volume_signal_score': 5}
                    
                        # --- ROW 1: COMPONENT BREAKDOWN (Current) ---
                        st.markdown("### 🧩 Current Score Breakdown")
                        c1, c2, c3, c4, c5 = st.columns(5)
                    
                        c1.metric("Quality", f"{scores['quality']}/10", 
                                 help="Profitability, ROE, Margins")
                        c2.metric("Value", f"{scores['value']}/10",
                                 help="PE, PB, PEG vs Sector")
                        c3.metric("Growth", f"{scores['growth']}/10",
                                 help="Rev & Earnings Growth")
                        c4.metric("Momentum", f"{scores['momentum']}/10",
                                 help="Price Strength (1W/1M/3M)")
                        c5.metric("Volume", f"{scores['volume_signal_score']}/10",
                                 help="Accumulation vs Distribution")
                    
                        st.markdown("---")
                    
                        # --- ROW 2: HISTORICAL TRENDS ---
                        st.markdown("### 📈 Price vs Volume & Momentum History")
                        st.caption("Compare Price action with Momentum and Volume health to spot divergences.")
                    
                        # Tech Chart: Momentum vs Volume vs Price
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                           vertical_spacing=0.05, row_heights=[0.7, 0.3])
                    
                        # Main: Price & Trend
                        # Price Line
                        fig.add_trace(go.Scatter(x=full_hist.index, y=full_hist['Close'], name="Price", 
                                                line=dict(color='#1F2937', width=1.5)), row=1, col=1)
                    
                        # Divergence Signals
                        div_data = detect_divergences(full_hist)
                        if not div_data['div_bull'].isnull().all():
                            fig.add_trace(go.Scatter(
                                x=div_data.index, y=div_data['div_bull'],
                                mode='markers', marker=dict(color='#00E676', size=10, symbol='circle'),
                                name="Bullish Div"
                            ), row=1, col=1)
                        
                        if not div_data['div_bear'].isnull().all():
                            fig.add_trace(go.Scatter(
                                x=div_data.index, y=div_data['div_bear'],
                                mode='markers', marker=dict(color='#FF1744', size=10, symbol='circle'),
                                name="Bearish Div"
                            ), row=1, col=1)

                        # --- EMA Crossovers ---
                        # Use MA columns from history (Pre-calculated in score_history.py)
                        if 'ma50' in full_hist.columns and 'ma200' in full_hist.columns:
                            ma50 = full_hist['ma50']
                            ma200 = full_hist['ma200']
                        
                            # Golden Cross: 50 cross > 200
                            golden_cross = (ma50 > ma200) & (ma50.shift(1) <= ma200.shift(1))
                            # Death Cross: 50 cross < 200 
                            death_cross = (ma50 < ma200) & (ma50.shift(1) >= ma200.shift(1))
                        
                            golden_pts = full_hist[golden_cross]
                            death_pts = full_hist[death_cross]
                        
                            if not golden_pts.empty:
                                fig.add_trace(go.Scatter(
                                    x=golden_pts.index, y=golden_pts['Close'],
                                    mode='markers', marker=dict(color='#FFD700', size=14, symbol='star'),
                                    name="Golden Cross (50>200)"
                                ), row=1, col=1)
                            
                            if not death_pts.empty:
                                fig.add_trace(go.Scatter(
                                    x=death_pts.index, y=death_pts['Close'],
                                    mode='markers', marker=dict(color='black', line=dict(color='white', width=1), size=12, symbol='x'),
                                    name="Death Cross (50<200)"
                                ), row=1, col=1)
                    
                        # Trend Score Overlay (Secondary Axis logic needed? Or just normalize/color background?)
                        # Let's use background color for Trend Score zones or just add it as a line on secondary axis?
                        # Using simple price for now to keep it clean, maybe color price line by trend?
                    
                        # Sub: Momentum & Volume Scores
                        fig.add_trace(go.Scatter(x=full_hist.index, y=full_hist['momentum_score_hist'], name="Momentum (0-10)",
                                                line=dict(color='#2962FF', width=2)), row=2, col=1)
                        fig.add_trace(go.Scatter(x=full_hist.index, y=full_hist['volume_score_hist'], name="Volume (0-10)",
                                                line=dict(color='#FFD600', width=2)), row=2, col=1)
                    
                        # Reference lines for subchart
                        fig.add_hline(y=5, line_dash="dot", line_color="gray", row=2, col=1)
                        fig.add_hline(y=8, line_dash="dot", line_color="green", row=2, col=1, opacity=0.3)
                        fig.add_hline(y=2, line_dash="dot", line_color="red", row=2, col=1, opacity=0.3)
                    
                        fig.update_layout(height=600, template="plotly_white", hovermode="x unified")
                        fig.update_yaxes(title_text="Price", row=1, col=1)
                        fig.update_yaxes(title_text="Score (0-10)", range=[0, 11], row=2, col=1)
                    
                        st.plotly_chart(fig, use_container_width=True)
                    
                        # --- NEW: SMART VOLUME ANALYSIS ---
                        st.markdown("### 📊 Smart Volume Analysis")
                        st.caption("Detailed volume flow analysis with accumulation/distribution signals.")
                    
                        from utils.visuals import chart_volume_analysis
                        vol_fig = chart_volume_analysis(full_hist)
                        if vol_fig:
                             st.plotly_chart(vol_fig, use_container_width=True)

                        # --- ROW 3: INSIGHTS ---
                        with st.expander("🔎 Analysis Guide", expanded=False):
                            st.markdown("""
                            **Divergence Signals:**
                            1. **Bullish Divergence**: Price makes new Low, but Momentum/Volume Score makes Higher Low.
                            2. **Bearish Divergence**: Price makes new High, but Momentum/Volume Score makes Lower High.
                            3. **Volume Climax**: Extremely high volume score (9-10) often marks a turning point or breakout.
                            """)
                else:
                    st.error(f"No history found for {trend_ticker}")
