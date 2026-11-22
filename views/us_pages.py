"""US AI Play and US Scanner page views (extracted from main.py)."""
from __future__ import annotations

from datetime import datetime
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.theme_engine import AI_CAPEX_THEME, ThemeEngine
from utils.ui_components import page_header
from utils.us_data_engine import fetch_us_market_data
from utils.us_rotation_tracker import (
    backfill_us_rotation_if_needed,
    build_rotation_pivot,
    load_us_rotation_history,
    render_us_rotation_heatmap,
    render_us_rotation_table,
    save_us_rotation_snapshot,
)

_AI_LAYER_FILE = "data/ai_layer_rotation.csv"


@st.cache_data(ttl=3600, show_spinner=False)
def _run_us_ai_scan(rs_weights_t: tuple, benchmark: str) -> pd.DataFrame:
    cfg = dict(AI_CAPEX_THEME)
    cfg['benchmark'] = benchmark
    cfg['rs_weights'] = list(rs_weights_t)
    engine = ThemeEngine(cfg)
    return engine.scan()


def _save_ai_layer_snapshot(agg_df, today_str: str):
    os.makedirs("data", exist_ok=True)
    rows = agg_df[["Layer", "CompRS", "RS_Vel", "Pct_Above_MA"]].copy()
    rows["date"] = today_str
    if os.path.exists(_AI_LAYER_FILE):
        existing = pd.read_csv(_AI_LAYER_FILE)
        existing = existing[existing["date"] != today_str]  # replace today
    else:
        existing = pd.DataFrame()
    combined = pd.concat([existing, rows], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=120)
    combined = combined[combined["date"] >= cutoff]
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    combined.to_csv(_AI_LAYER_FILE, index=False)
    return combined

def _load_ai_layer_history():
    if not os.path.exists(_AI_LAYER_FILE):
        return pd.DataFrame()
    df = pd.read_csv(_AI_LAYER_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=86400, show_spinner=False)
def _backfill_ai_layers(rs_weights_t, benchmark):
    cfg = dict(AI_CAPEX_THEME)
    cfg['benchmark'] = benchmark
    cfg['rs_weights'] = list(rs_weights_t)
    eng = ThemeEngine(cfg)
    return eng.backfill_layer_rotation(n_months=12)


@st.cache_data(ttl=3600, show_spinner=False)
def _run_us_scan(benchmark: str, w5, w21, w63, live: bool) -> pd.DataFrame:
    weights = [(5, w5), (21, w21), (63, w63)] if w5 is not None else [(252, 1.0)]
    return fetch_us_market_data(benchmark=benchmark, rs_weights=weights, live_mode=live)


def render_us_ai_play() -> None:
    st.markdown(page_header(
        "🇺🇸 US AI Supply Chain",
        "Full 12-layer datacenter supply chain | Rotation detection vs SMH/SPY | Config D weights"
    ), unsafe_allow_html=True)

    # ── Sidebar controls ──────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🇺🇸 US AI Play Settings**")

    us_benchmark = st.sidebar.selectbox(
        "Benchmark", ["SMH", "SPY", "QQQ"], index=0, key="us_benchmark",
        help="SMH = Semis ETF (sector-relative alpha). SPY = broad market."
    )
    # Ranking: 12-month RS by default. analysis/TRANSFER_VERDICT.md: it beat the 1w/1m/3m blend at 8 of 8 starts for the
    # AI theme (and the blend was below random on US sectors); the blend stays available to compare.
    us_ai_rank = st.sidebar.radio("Rank by", ["12-month RS (recommended)", "Short blend (1w / 1m / 3m)"], key="us_ai_rank",
                                  help="12-month RS ranked best in every US test; holding the whole list still beat rotating it")
    if us_ai_rank.startswith("Short"):
        st.sidebar.markdown("**RS Weights (Config D)**")
        us_w5  = st.sidebar.slider("RS5  (1-week)",  0.10, 0.50, 0.30, 0.05, key="us_w5")
        us_w21 = st.sidebar.slider("RS21 (1-month)", 0.20, 0.60, 0.50, 0.05, key="us_w21")
        raw_us_w63 = round(1.0 - us_w5 - us_w21, 2)
        us_w63 = max(0.05, raw_us_w63)
        us_total_w = us_w5 + us_w21 + us_w63
        us_w5_n, us_w21_n, us_w63_n = us_w5 / us_total_w, us_w21 / us_total_w, us_w63 / us_total_w
        st.sidebar.caption(f"RS63 auto-set to **{us_w63_n:.0%}** | RS5={us_w5_n:.0%} RS21={us_w21_n:.0%}")
        if us_w63_n >= us_w5_n:
            st.sidebar.warning("⚠️ RS63 ≥ RS5 — raise RS5 to front-load theme momentum")
    else:
        us_w5_n = us_w21_n = us_w63_n = None

    # ── Cached ThemeEngine scan ───────────────────────────────────────────────
    rs_weights_t = (((5, round(us_w5_n, 4)), (21, round(us_w21_n, 4)), (63, round(us_w63_n, 4)))
                    if us_w5_n is not None else ((252, 1.0),))
    _us_rank_txt = ("12-month RS" if us_w5_n is None else
                    f"RS5={us_w5_n:.0%} / RS21={us_w21_n:.0%} / RS63={us_w63_n:.0%}")

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh Data", key="us_ai_refresh"):
            st.cache_data.clear()
            st.rerun()

    with st.spinner(f"Loading AI supply chain data vs {us_benchmark}…"):
        scan_df = _run_us_ai_scan(rs_weights_t, us_benchmark)

    if scan_df.empty:
        st.error("⚠️ Data fetch failed — check your connection. Click **Refresh Data** to retry.")
        st.stop()

    # ── 4 Pulse Metrics ───────────────────────────────────────────────────────
    pct_up    = scan_df['Signal'].isin(['STRONG UP', 'UPTREND']).mean() * 100
    avg_rs    = scan_df['CompRS'].mean()
    vel_vals  = scan_df['RS_Vel'].dropna()
    avg_vel   = vel_vals.mean() if not vel_vals.empty else 0.0
    top_layer_full = scan_df.groupby('Layer')['CompRS'].mean().idxmax() if not scan_df.empty else "—"
    top_layer_label = top_layer_full.split(': ', 1)[-1] if ': ' in top_layer_full else top_layer_full

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Stocks in Uptrend",            f"{pct_up:.0f}%")
    mc2.metric(f"Avg Comp RS vs {us_benchmark}", f"{avg_rs:+.1f}%")
    mc3.metric("Avg RS Velocity (5D)",           f"{avg_vel:+.2f}%",
               delta="↑ Accelerating" if avg_vel > 0 else "↓ Decelerating")
    mc4.metric("Leading Layer",                  top_layer_label)

    st.markdown("---")

    # ── Layer Rotation Matrix ─────────────────────────────────────────────────
    st.markdown("### 🔄 Layer Rotation Matrix")
    st.caption(
        f"Avg per supply-chain layer vs **{us_benchmark}** | "
        f"{_us_rank_txt} | "
        "Sorted best→worst Composite RS"
    )

    layer_agg = (
        scan_df.groupby('Layer')
        .agg(
            CompRS=('CompRS', 'mean'),
            RS_Vel=('RS_Vel', 'mean'),
            RS5vs=('RS5vs', 'mean'),
            RS21vs=('RS21vs', 'mean'),
            RS63vs=('RS63vs', 'mean'),
            Count=('Ticker', 'count'),
            Pct_Above_MA=('Above_MA', lambda x: round(x.mean() * 100, 0)),
        )
        .round(2)
        .reset_index()
        .sort_values('CompRS', ascending=False)
    )

    # ── Persist daily layer snapshot ─────────────────────────────────────────
    _today_str = datetime.now().strftime("%Y-%m-%d")
    _save_ai_layer_snapshot(layer_agg, _today_str)
    _layer_hist = _load_ai_layer_history()

    # Backfill 12 months of history on first visit (< 2 distinct dates in file)
    if _layer_hist["date"].nunique() < 2 if not _layer_hist.empty else True:
        with st.spinner("Building 12-month layer history (first visit only)…"):
            _bf_df = _backfill_ai_layers(rs_weights_t, us_benchmark)

        if not _bf_df.empty:
            _existing_dates = set(
                _layer_hist["date"].dt.strftime("%Y-%m-%d").tolist()
                if not _layer_hist.empty else []
            )
            _bf_new = _bf_df[~_bf_df["date"].isin(_existing_dates)]
            if not _bf_new.empty:
                os.makedirs("data", exist_ok=True)
                if os.path.exists(_AI_LAYER_FILE):
                    _curr_csv = pd.read_csv(_AI_LAYER_FILE)
                    _combined_bf = pd.concat([_bf_new, _curr_csv], ignore_index=True)
                else:
                    _combined_bf = _bf_new
                _combined_bf.to_csv(_AI_LAYER_FILE, index=False)
                _layer_hist = _load_ai_layer_history()

    # Build pivot: Layer × date (sorted chronologically)
    if not _layer_hist.empty:
        _hist_pivot = _layer_hist.pivot_table(
            index="Layer", columns="date", values="CompRS", aggfunc="mean"
        )
        _hist_pivot = _hist_pivot.sort_index(axis=1)  # dates ascending
        _n_days = len(_hist_pivot.columns)
    else:
        _hist_pivot = pd.DataFrame()
        _n_days = 0

    # ── Tabs: Trend Table | Heatmap ──────────────────────────────────────────
    rot_tab1, rot_tab2 = st.tabs(["📈 Trend Table (day-by-day)", "🌡️ Current Heatmap"])

    with rot_tab1:
        if _n_days < 2:
            st.info("Trend data accumulates from the second visit onwards — come back tomorrow for sparklines.")
        else:
            _trend_rows = []
            for _, r in layer_agg.iterrows():
                lyr = r["Layer"]
                if lyr in _hist_pivot.index:
                    series = _hist_pivot.loc[lyr].dropna().tolist()
                else:
                    series = [r["CompRS"]]

                # 5D delta: today vs ~5 sessions ago
                _delta_5d = None
                if lyr in _hist_pivot.index:
                    _valid = _hist_pivot.loc[lyr].dropna()
                    if len(_valid) >= 6:
                        _delta_5d = round(float(_valid.iloc[-1]) - float(_valid.iloc[-6]), 2)
                    elif len(_valid) >= 2:
                        _delta_5d = round(float(_valid.iloc[-1]) - float(_valid.iloc[0]), 2)

                # Momentum badge
                if _delta_5d is not None:
                    if _delta_5d > 1:
                        badge = "🟢 Rising"
                    elif _delta_5d < -1:
                        badge = "🔴 Falling"
                    else:
                        badge = "🟡 Flat"
                else:
                    badge = "—"

                _trend_rows.append({
                    "Layer":         lyr,
                    "Comp RS":       r["CompRS"],
                    "5D Δ":          _delta_5d,
                    "Momentum":      badge,
                    "RS Vel":        r["RS_Vel"],
                    "% > MA":        r["Pct_Above_MA"],
                    "Stocks":        int(r["Count"]),
                    f"Trend ({_n_days}d)": series,
                })

            _trend_df = pd.DataFrame(_trend_rows)
            _trend_col_cfg = {
                "Layer":    st.column_config.TextColumn("Layer"),
                "Comp RS":  st.column_config.NumberColumn("Comp RS", format="%+.1f%%"),
                "5D Δ":     st.column_config.NumberColumn("5D Δ", format="%+.1f%%"),
                "Momentum": st.column_config.TextColumn("Momentum"),
                "RS Vel":   st.column_config.NumberColumn("RS Velocity", format="%+.2f%%"),
                "% > MA":   st.column_config.NumberColumn("% > MA", format="%.0f%%"),
                "Stocks":   st.column_config.NumberColumn("Stocks", format="%d"),
                f"Trend ({_n_days}d)": st.column_config.LineChartColumn(
                    f"Trend ({_n_days}d)", y_min=-30, y_max=30
                ),
            }
            st.dataframe(_trend_df, column_config=_trend_col_cfg,
                         hide_index=True, use_container_width=True)

            _csv_bytes = _trend_df.drop(columns=[f"Trend ({_n_days}d)"]).to_csv(index=False).encode()
            st.download_button("⬇️ Download Layer Rotation CSV", _csv_bytes,
                               "ai_layer_rotation.csv", "text/csv", key="dl_layer_rot")

    with rot_tab2:
        _metrics  = ['RS_Vel', 'RS5vs', 'RS21vs', 'RS63vs', 'CompRS']
        _m_labels = ['RS Velocity (5D)', f'RS5 vs {us_benchmark}', f'RS21 vs {us_benchmark}',
                     f'RS63 vs {us_benchmark}', 'Composite RS']
        _layers   = layer_agg['Layer'].tolist()
        _z        = layer_agg[_metrics].values.tolist()

        _text = [
            [f"{v:+.1f}" if (v is not None and not (isinstance(v, float) and np.isnan(v))) else "—"
             for v in row]
            for row in _z
        ]

        fig_heat = go.Figure(go.Heatmap(
            z=_z,
            x=_m_labels,
            y=_layers,
            colorscale='RdYlGn',
            zmid=0,
            text=_text,
            texttemplate="%{text}",
            textfont={"size": 11},
            showscale=True,
            colorbar=dict(title="% vs Bench", thickness=14),
        ))
        fig_heat.update_layout(
            height=max(380, len(_layers) * 44),
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(side='top'),
            yaxis=dict(autorange='reversed'),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # ── Capex Cycle Position ───────────────────────────────────────────────────
    st.markdown("### 📍 Capex Cycle Position")

    _early_layers = [
        'L1: InP Substrate', 'L1: SiC/GaN',
        'L0: EDA Software', 'L0: IP Cores',
        'L3: Equipment', 'L3: Photomasks', 'L3: Ion Implant', 'L3: Metrology',
    ]
    _mid_layers = [
        'L4: Foundry', 'L5: HBM Memory',
        'L6: Chip Design', 'L6: Custom ASIC', 'L6: Power Semi', 'L6: VRM',
    ]
    _late_layers = [
        'L7: Packaging',
        'L8: Networking', 'L8: Optics', 'L8: Connectors',
        'L9: Power Dist', 'L9A: Power Gen', 'L9A: Nuclear', 'L9A: Fuel Cell', 'L9A: Solar',
        'L10: Cooling', 'L10A: Water',
        'L11: Construction', 'L11: DC REIT',
        'L12: Hyperscaler',
    ]

    def _stage_rs(layer_list):
        sub = scan_df[scan_df['Layer'].isin(layer_list)]
        return sub['CompRS'].mean() if not sub.empty else 0.0

    rs_early = _stage_rs(_early_layers)
    rs_mid   = _stage_rs(_mid_layers)
    rs_late  = _stage_rs(_late_layers)

    _max_rs = max(rs_early, rs_mid, rs_late)
    if _max_rs == rs_early:
        _stage_label = "🌱 Early Capex Cycle"
        _stage_desc  = "Equipment & EDA tools lead — foundries ramping capacity. **Long L3: ASML, AMAT, LRCX, KLAC.**"
        _stage_color = "#2196F3"
    elif _max_rs == rs_mid:
        _stage_label = "⚡ Mid Capex Cycle"
        _stage_desc  = "Chip production peaks — memory & custom ASIC demand surges. **Long L5-L6: NVDA, MU, AVGO, MRVL.**"
        _stage_color = "#FF9800"
    else:
        _stage_label = "🏗️ Late Capex Cycle"
        _stage_desc  = "Infrastructure buildout dominant: power, cooling, DC construction. **Long L9-L11: ETN, VRT, PWR, EQIX.**"
        _stage_color = "#4CAF50"

    st.markdown(
        f"""<div style="background:{_stage_color}20; border-left:4px solid {_stage_color};
        padding:14px 18px; border-radius:6px; margin-bottom:14px;">
        <span style="font-size:1.1em; font-weight:600">{_stage_label}</span><br>
        <span style="color:#bbb; font-size:0.95em">{_stage_desc}</span><br><br>
        <span style="font-size:0.85em; color:#aaa">
        🌱 Early RS: <b style="color:#ddd">{rs_early:+.1f}%</b> &nbsp;|&nbsp;
        ⚡ Mid RS: <b style="color:#ddd">{rs_mid:+.1f}%</b> &nbsp;|&nbsp;
        🏗️ Late RS: <b style="color:#ddd">{rs_late:+.1f}%</b>
        </span></div>""",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Entry Signals (top 3 leading layers) ─────────────────────────────────
    st.markdown("### 🎯 Entry Signals — Leading Layer Picks")

    _top3_layers = layer_agg.head(3)['Layer'].tolist()
    entry_df = scan_df[scan_df['Layer'].isin(_top3_layers)].copy()
    entry_df = entry_df[entry_df['RS_Vel'].notna()].copy()

    def _grade(r):
        if r['CompRS'] > 5 and r['RS_Vel'] > 0 and r['Above_MA']:
            return "🔥 BUY"
        if r['CompRS'] > 0 and r['Above_MA']:
            return "👀 WATCH"
        return "⚠️ WEAK"

    entry_df['Grade'] = entry_df.apply(_grade, axis=1)
    entry_df = entry_df.sort_values(['Grade', 'CompRS'], ascending=[True, False])

    _entry_display = entry_df[
        ['Ticker', 'Layer', 'Grade', 'CompRS', 'RS_Vel', 'RS5vs', 'RS21vs', 'Price', 'Bottleneck']
    ].copy()
    _entry_display.columns = ['Ticker', 'Layer', 'Signal', 'Comp RS', 'RS Vel', 'RS5 vs', 'RS21 vs', 'Price $', 'Chokepoint']

    st.dataframe(
        _entry_display.reset_index(drop=True),
        column_config={
            'Comp RS':   st.column_config.NumberColumn("Comp RS",        format="%+.2f%%"),
            'RS Vel':    st.column_config.NumberColumn("RS Vel (5D)",    format="%+.2f%%"),
            'RS5 vs':    st.column_config.NumberColumn(f"RS5 vs {us_benchmark}", format="%+.1f%%"),
            'RS21 vs':   st.column_config.NumberColumn(f"RS21 vs {us_benchmark}", format="%+.1f%%"),
            'Price $':   st.column_config.NumberColumn("Price $",        format="$%.2f"),
            'Chokepoint': st.column_config.TextColumn("Chokepoint"),
        },
        use_container_width=True,
        hide_index=True,
        height=min(len(_entry_display) * 38 + 42, 520),
    )
    st.caption(f"Showing top 3 layers by Composite RS | Refresh hourly | Benchmark: {us_benchmark}")

    st.markdown("---")

    # ── Full Universe Screener (expander) ────────────────────────────────────
    with st.expander("📋 Full Universe Screener", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            _sig_opts   = sorted(scan_df['Signal'].unique().tolist())
            _sig_def    = [s for s in ['STRONG UP', 'UPTREND'] if s in _sig_opts]
            sig_filter  = st.multiselect("Signal", _sig_opts, default=_sig_def, key="us_sig_filter")
        with fc2:
            layer_opts   = sorted(scan_df['Layer'].unique().tolist())
            layer_filter = st.multiselect("Layer", layer_opts, default=[], key="us_layer_filter")
        with fc3:
            btn_filter   = st.selectbox("Chokepoint", ["All", "CRITICAL", "TIGHT"], key="us_btn_filter")

        _filtered = scan_df.copy()
        if sig_filter:
            _filtered = _filtered[_filtered['Signal'].isin(sig_filter)]
        if layer_filter:
            _filtered = _filtered[_filtered['Layer'].isin(layer_filter)]
        if btn_filter != "All":
            _filtered = _filtered[_filtered['Bottleneck'] == btn_filter]

        _disp_cols = ['Ticker', 'Layer', 'Bottleneck', 'Signal', 'Price',
                      'CompRS', 'RS_Vel', 'RS5vs', 'RS21vs', 'RS63vs', 'Dist_52W']
        _disp = _filtered[_disp_cols].copy()
        _disp.columns = ['Ticker', 'Layer', 'Chokepoint', 'Signal', 'Price $',
                         'Comp RS', 'RS Vel', 'RS5 vs', 'RS21 vs', 'RS63 vs', '52W Off%']

        st.dataframe(
            _disp.reset_index(drop=True),
            column_config={
                'Price $':  st.column_config.NumberColumn("Price $",       format="$%.2f"),
                'Comp RS':  st.column_config.NumberColumn("Comp RS",       format="%+.2f%%"),
                'RS Vel':   st.column_config.NumberColumn("RS Vel (5D)",   format="%+.2f%%"),
                'RS5 vs':   st.column_config.NumberColumn(f"RS5 vs {us_benchmark}",  format="%+.1f%%"),
                'RS21 vs':  st.column_config.NumberColumn(f"RS21 vs {us_benchmark}", format="%+.1f%%"),
                'RS63 vs':  st.column_config.NumberColumn(f"RS63 vs {us_benchmark}", format="%+.1f%%"),
                '52W Off%': st.column_config.NumberColumn("52W Off%",      format="%.1f%%"),
            },
            use_container_width=True,
            hide_index=True,
            height=min(len(_disp) * 38 + 42, 700),
        )
        st.caption(
            f"Benchmark: {us_benchmark} | Ranked by: {_us_rank_txt} | {len(_disp)} of {len(scan_df)} stocks shown"
        )

    # ── Supply Chain Layer Map ────────────────────────────────────────────────
    with st.expander("🗺️ Supply Chain Layer Map"):
        st.markdown("""
    | Layer | Category | Key Companies | Feeds Into |
    |---|---|---|---|
    | **L0** | EDA & IP Cores | SNPS, CDNS (duopoly), ARM | L4, L6 |
    | **L1-L2** | Raw materials & substrates | AXTI (InP), WOLF (SiC), ENTG | L3, L8 |
    | **L3** | Semiconductor equipment | ASML (EUV monopoly), AMAT, LRCX, KLAC, ACLS | L4, L5 |
    | **L4-L5** | Foundry & memory | TSM (INTC, GFS), MU (HBM) | L6, L7 |
    | **L6** | Chip design | NVDA, AMD, AVGO, MRVL, MPWR, ON | L7, L8 |
    | **L7** | Advanced packaging | AMKR, KLIC | L8, L12 |
    | **L8** | Networking & optics | ANET, COHR, LITE, MTSI, FN, APH | L12 |
    | **L9-L10** | Power & cooling | ETN, VRT, GEV, PWR, NVT, ECL | L11, L12 |
    | **L11** | DC construction & REITs | EQIX, DLR, EME, FIX, CEG, CCJ | L12 |
    | **L12** | Hyperscalers (demand signal) | MSFT, AMZN, GOOGL, META, ORCL | Revenue |

    **Rotation insight:** L3 Equipment leads at the start of each capex wave (new fab orders).  
    L6 Chips peak mid-cycle. L9-L11 Power/Infra run late as sites come online.  
    ASML→TSM is the deepest serial chokepoint — 2 CRITICAL monopolies in series.  
    Power (L9) is the #1 near-term rate limiter for AI scaling.
        """)

    with st.expander("ℹ️ RS Weight Tuning Guide"):
        st.markdown(f"""
    **Standard OptComp-V22:** RS5 = 10%, RS21 = 50%, RS63 = 40%

    **AI Supply Chain (current):** {_us_rank_txt}

    **Backtest (analysis/TRANSFER_VERDICT.md):** 12-month RS beat the 1w/1m/3m blend at 8 of 8 starts, before and after
    costs, and the blend scored below random on US sectors. Holding the whole list still beat rotating it: use the ranking as
    a watchlist. The reasoning below was the case for the blend, kept for reference.

    AI datacenter is a **fast-rotating, news-driven theme.** The 3-month (RS63) lookback often
    includes time *before* the theme ignited, diluting the signal. Elevating RS5 catches
    fresh rotations earlier; RS21 stays dominant as the medium-term confirmation window.

    **Config D (backtest-optimal):** RS5=30% / RS21=50% / RS63=20% — +70.3% CAGR, +25.5% alpha vs SMH, Sharpe 4.48

    **Rotation Matrix** shows which supply-chain layer cluster is currently "in play" so you
    can size into the right pocket before the rotation broadens.
        """)


def render_us_scanner(*, us_quote_url, us_display_re) -> None:

    st.markdown(page_header("🇺🇸 S&P 500 Momentum Scanner", "OptComp RS • Sector Heatmap • Sub-Industry Matrix | Powered by SPY Benchmark"), unsafe_allow_html=True)

    # --- Sidebar controls ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🇺🇸 US Scanner Settings**")
    us_scan_benchmark = st.sidebar.selectbox(
        "Benchmark",
        ["SPY", "QQQ", "IWM"],
        index=0,
        key="us_scan_bm",
        help="SPY = S&P 500 | QQQ = Nasdaq 100 | IWM = Russell 2000"
    )
    us_s_rank = st.sidebar.radio("Rank by", ["12-month RS (recommended)", "Short blend (1w / 1m / 3m)"], key="us_s_rank",
                                 help="analysis/TRANSFER_VERDICT.md: 12-month RS was the most reliable ordering in every US test; "
                                      "the short blend scored below random on US sectors")
    if us_s_rank.startswith("Short"):
        st.sidebar.markdown("**RS Weights (Config D)**")
        us_s_w5  = st.sidebar.slider("RS5  (1-week)",  0.10, 0.50, 0.30, 0.05, key="us_s_w5")
        us_s_w21 = st.sidebar.slider("RS21 (1-month)", 0.20, 0.60, 0.50, 0.05, key="us_s_w21")
        us_s_w63_n = round(1.0 - us_s_w5 - us_s_w21, 4)
        st.sidebar.caption(f"RS63 auto-set to **{us_s_w63_n:.0%}**")
        if us_s_w63_n < 0:
            st.sidebar.error("Weights exceed 100% — reduce RS5 or RS21")
            us_s_w63_n = 0.0
    else:
        us_s_w5 = us_s_w21 = us_s_w63_n = None

    us_live_mode = st.sidebar.checkbox("Live Mode (bypass cache)", value=False, key="us_scan_live")
    if st.sidebar.button("🗑️ Clear US Cache", key="us_clear_cache", help="Force fresh download (needed after ticker list changes)"):
        from utils.us_data_engine import clear_us_cache
        n = clear_us_cache()
        st.cache_data.clear()
        st.sidebar.success(f"Cleared {n} cache file(s). Reloading…")
        st.rerun()

    # --- Cached fetch wrapper ---
    with st.spinner("Downloading S&P 500 data…  (first load ~60 s, then cached)"):
        us_df = _run_us_scan(us_scan_benchmark, us_s_w5, us_s_w21, us_s_w63_n, us_live_mode)

    if us_df.empty:
        st.error("Failed to load US market data. Check internet connection or try Live Mode.")
        st.stop()

    # Ensure required columns exist
    for col in ["comp_rs", "volatility", "dna_signal", "sub_industry", "dist_200dma",
                "rs_5d", "rs_21d", "rs_63d"]:
        if col not in us_df.columns:
            us_df[col] = None

    # ── QUICK STATS ──────────────────────────────────────────────────────────
    qs1, qs2, qs3, qs4 = st.columns(4)
    _strong = len(us_df[us_df["trend_signal"] == "STRONG UPTREND"])
    _uptrend = len(us_df[us_df["trend_signal"].isin(["STRONG UPTREND", "UPTREND"])])
    _avg_ts = us_df["trend_score"].mean()
    _breakouts = len(us_df[us_df["dist_52w"] > -2.0])
    qs1.metric("🚀 Strong Momentum", f"{_strong}", help="Trend Score ≥ 75")
    qs2.metric("📈 Total Uptrends", f"{_uptrend}")
    qs3.metric("📊 Avg Trend Score", f"{_avg_ts:.0f}/100")
    qs4.metric("🔥 Breakout Alerts", f"{_breakouts}", help="Within 2% of 52W High")

    st.markdown("---")

    # ── SECTOR RS HEATMAP ────────────────────────────────────────────────────
    with st.expander("🗺️ **Sector RS Heatmap**", expanded=True):
        _sec_grp = (
            us_df.groupby("sector")[["rs_5d", "rs_21d", "rs_63d", "comp_rs", "trend_score"]]
            .mean()
            .round(2)
            .reset_index()
        )
        _sec_grp = _sec_grp.sort_values("comp_rs", ascending=False)

        if not _sec_grp.empty:
            _heat_z  = _sec_grp[["rs_5d", "rs_21d", "rs_63d", "comp_rs"]].values.tolist()
            _heat_y  = _sec_grp["sector"].tolist()
            _heat_x  = ["RS5 (1W)", "RS21 (1M)", "RS63 (3M)", "CompRS"]
            _heat_txt = [[f"{v:+.1f}" for v in row] for row in _heat_z]

            fig_heat = go.Figure(go.Heatmap(
                z=_heat_z,
                x=_heat_x,
                y=_heat_y,
                text=_heat_txt,
                texttemplate="%{text}",
                colorscale="RdYlGn",
                zmid=0,
                showscale=True,
                colorbar=dict(title="RS %", thickness=12),
            ))
            fig_heat.update_layout(
                template="plotly_white",
                height=420,
                margin=dict(l=200, r=20, t=30, b=30),
                xaxis=dict(side="top"),
            )
            st.plotly_chart(fig_heat, use_container_width=True, key="us_sector_heatmap")

            # Sector table
            st.dataframe(
                _sec_grp.rename(columns={
                    "sector": "Sector", "rs_5d": "RS5 (1W)", "rs_21d": "RS21 (1M)",
                    "rs_63d": "RS63 (3M)", "comp_rs": "CompRS", "trend_score": "Avg Trend"
                }),
                column_config={
                    "RS5 (1W)":  st.column_config.NumberColumn(format="%+.2f%%"),
                    "RS21 (1M)": st.column_config.NumberColumn(format="%+.2f%%"),
                    "RS63 (3M)": st.column_config.NumberColumn(format="%+.2f%%"),
                    "CompRS":    st.column_config.NumberColumn(format="%+.2f%%"),
                    "Avg Trend": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                },
                hide_index=True,
                use_container_width=True,
            )

    # ── TOP MOVERS TAPE ──────────────────────────────────────────────────────
    _top8 = us_df.nlargest(8, "trend_score")[["ticker", "trend_score", "currentPrice"]]
    _tape_html = " &nbsp;•&nbsp; ".join([
        f"<span style='color:#34C759;font-weight:600'>{r['ticker']}</span>"
        f" <span style='color:#888'>${r['currentPrice']:.1f}</span>"
        f" <span style='background:rgba(52,199,89,0.2);padding:2px 8px;border-radius:10px;color:#34C759'>{r['trend_score']}</span>"
        for _, r in _top8.iterrows()
    ])
    st.markdown(
        f"""<div style="background:rgba(255,255,255,0.03);padding:12px 20px;border-radius:8px;
                        overflow-x:auto;white-space:nowrap;border:1px solid rgba(255,255,255,0.1);">
            <span style='color:#FFD700;margin-right:10px'>🔥 TOP MOVERS:</span> {_tape_html}
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── BREAKOUT ALERTS ───────────────────────────────────────────────────────
    _bkouts = us_df[us_df["dist_52w"] > -2.0].copy()
    if not _bkouts.empty:
        with st.expander(f"🚨 **{len(_bkouts)} BREAKOUT ALERTS** (Within 2% of 52W High)", expanded=False):
            _bkouts_s = _bkouts.nlargest(20, "dist_52w").copy()
            _bkouts_s["yf_link"] = _bkouts_s.apply(lambda r: us_quote_url(r['ticker']), axis=1)
            st.dataframe(
                _bkouts_s[["yf_link", "name", "currentPrice", "dist_52w", "trend_score", "overall"]],
                column_config={
                    "yf_link":      st.column_config.LinkColumn("Ticker", display_text=us_display_re),
                    "name":         "Company",
                    "currentPrice": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "dist_52w":     st.column_config.NumberColumn("% from 52W High", format="%.1f%%"),
                    "trend_score":  st.column_config.ProgressColumn("Trend", min_value=0, max_value=100),
                    "overall":      st.column_config.NumberColumn("Score", format="%.1f"),
                },
                hide_index=True,
                height=300,
            )

    # ── FILTERS ───────────────────────────────────────────────────────────────
    with st.expander("⚡ Filter", expanded=True):
        ff1, ff2 = st.columns([1, 2])
        with ff1:
            us_search = st.text_input("🔍 Search", placeholder="AAPL, Apple…", key="us_search")
        with ff2:
            _us_sectors = sorted(us_df["sector"].fillna("Unknown").unique().tolist())
            us_sel_sector = st.multiselect("Sector (Empty = All)", _us_sectors, default=[], key="us_sectors")

        ff3, ff4 = st.columns(2)
        with ff3:
            us_min_score = st.slider("Min Trend Score", 0, 100, 0, key="us_min_score")
        with ff4:
            _sig_opts = ["STRONG UPTREND", "UPTREND", "NEUTRAL", "DOWNTREND", "STRONG DOWNTREND"]
            us_sig_filter = st.multiselect("Signal", _sig_opts, default=_sig_opts, key="us_sig_filter")

        st.markdown("---")
        st.markdown("**🔬 Fundamental Filters**")
        uf1, uf2, uf3, uf4 = st.columns(4)
        with uf1:
            us_min_quality = st.slider("Min Quality", 0, 10, 0, key="us_min_qual")
        with uf2:
            us_min_value = st.slider("Min Value", 0, 10, 0, key="us_min_val")
        with uf3:
            us_min_growth = st.slider("Min Growth", 0, 10, 0, key="us_min_growth")
        with uf4:
            us_min_vol = st.slider("Min Volume Score", 0, 10, 0, key="us_min_vol")

    # Apply filters
    us_fdf = us_df.copy()
    if us_search:
        _q = us_search.lower().strip()
        us_fdf = us_fdf[
            us_fdf["ticker"].fillna("").astype(str).str.lower().str.contains(_q, regex=False) |
            us_fdf["name"].fillna("").astype(str).str.lower().str.contains(_q, regex=False)
        ]
    if us_sel_sector:
        us_fdf = us_fdf[us_fdf["sector"].isin(us_sel_sector)]
    us_fdf = us_fdf[us_fdf["trend_score"] >= us_min_score]
    if us_sig_filter:
        us_fdf = us_fdf[us_fdf["trend_signal"].isin(us_sig_filter)]
    if us_min_quality > 0 and "quality" in us_fdf.columns:
        us_fdf = us_fdf[us_fdf["quality"] >= us_min_quality]
    if us_min_value > 0 and "value" in us_fdf.columns:
        us_fdf = us_fdf[us_fdf["value"] >= us_min_value]
    if us_min_growth > 0 and "growth" in us_fdf.columns:
        us_fdf = us_fdf[us_fdf["growth"] >= us_min_growth]
    if us_min_vol > 0 and "volume_signal_score" in us_fdf.columns:
        us_fdf = us_fdf[us_fdf["volume_signal_score"] >= us_min_vol]

    # Dynamic column filter
    with st.expander("🌪️ **Custom Column Filter**", expanded=False):
        dc1, dc2, dc3 = st.columns([2, 1, 2])
        with dc1:
            us_filter_col = st.selectbox(
                "Filter Column",
                ["RS Score (vs SPY)", "Volatility", "Trend Score", "Distance from 52W High", "Price"],
                key="us_dyn_col",
            )
        _us_col_map = {
            "RS Score (vs SPY)": "comp_rs",
            "Volatility": "volatility",
            "Trend Score": "trend_score",
            "Distance from 52W High": "dist_52w",
            "Price": "currentPrice",
        }
        _us_tcol = _us_col_map[us_filter_col]
        with dc3:
            try:
                _dmin = float(us_fdf[_us_tcol].min()) if _us_tcol in us_fdf and us_fdf[_us_tcol].notna().any() else 0.0
                _dmax = float(us_fdf[_us_tcol].max()) if _us_tcol in us_fdf and us_fdf[_us_tcol].notna().any() else 100.0
                if _dmin == _dmax:
                    _dmax += 1.0
            except Exception:
                _dmin, _dmax = 0.0, 100.0
            us_dyn_range = st.slider(f"{us_filter_col} Range", _dmin, _dmax, (_dmin, _dmax), key="us_dyn_range")
        if _us_tcol in us_fdf.columns and (us_dyn_range[0] > _dmin or us_dyn_range[1] < _dmax):
            us_fdf = us_fdf[
                ((us_fdf[_us_tcol] >= us_dyn_range[0]) & (us_fdf[_us_tcol] <= us_dyn_range[1])) |
                us_fdf[_us_tcol].isna()
            ]
            st.caption(f"Showing {len(us_fdf)} stocks with {us_filter_col} between {us_dyn_range[0]:.1f} and {us_dyn_range[1]:.1f}")

    # ── MAIN TABLE ────────────────────────────────────────────────────────────
    if us_fdf.empty:
        st.warning("No stocks match the current filters.")
    else:
        _us_hdr, _us_dl = st.columns([3, 1])
        with _us_hdr:
            st.subheader(f"Found {len(us_fdf)} US Momentum Stocks")

        _EMOJI = {
            "STRONG UPTREND": "🟢", "UPTREND": "🔵",
            "NEUTRAL": "🟡", "DOWNTREND": "🟠", "STRONG DOWNTREND": "🔴",
        }
        us_fdf = us_fdf.copy()
        us_fdf["signal_display"] = us_fdf["trend_signal"].map(
            lambda s: f"{_EMOJI.get(s, '')} {s}" if s else s
        )
        us_fdf["yf_link"] = us_fdf.apply(lambda r: us_quote_url(r['ticker']), axis=1)

        _us_disp = [
            "yf_link", "name", "sector", "sub_industry", "currentPrice",
            "signal_display", "trend_score", "comp_rs", "rs_5d", "rs_21d", "rs_63d",
            "volatility", "dna_signal", "dist_52w", "dist_200dma",
            "quality", "value", "growth", "momentum", "volume_signal_score",
        ]
        _us_disp = [c for c in _us_disp if c in us_fdf.columns]

        st.dataframe(
            us_fdf[_us_disp].sort_values("trend_score", ascending=False),
            column_config={
                "yf_link":           st.column_config.LinkColumn("Ticker", display_text=us_display_re),
                "name":              "Company",
                "sector":            "Sector",
                "sub_industry":      "Sub-Industry",
                "currentPrice":      st.column_config.NumberColumn("Price", format="$%.2f"),
                "signal_display":    st.column_config.TextColumn("Signal"),
                "trend_score":       st.column_config.ProgressColumn("Trend", format="%d", min_value=0, max_value=100),
                "comp_rs":           st.column_config.NumberColumn("CompRS", format="%+.1f%%", help="Composite RS vs benchmark"),
                "rs_5d":             st.column_config.NumberColumn("RS5 (1W)", format="%+.1f%%"),
                "rs_21d":            st.column_config.NumberColumn("RS21 (1M)", format="%+.1f%%"),
                "rs_63d":            st.column_config.NumberColumn("RS63 (3M)", format="%+.1f%%"),
                "volatility":        st.column_config.NumberColumn("Volatility", format="%.0f%%"),
                "dna_signal":        st.column_config.TextColumn("Signal"),
                "dist_52w":          st.column_config.NumberColumn("% from 52W High", format="%.1f%%"),
                "dist_200dma":       st.column_config.NumberColumn("% vs 200DMA", format="%.1f%%"),
                "quality":           st.column_config.ProgressColumn("Quality", min_value=0, max_value=10, format="%.1f"),
                "value":             st.column_config.ProgressColumn("Value",   min_value=0, max_value=10, format="%.1f"),
                "growth":            st.column_config.ProgressColumn("Growth",  min_value=0, max_value=10, format="%.1f"),
                "momentum":          st.column_config.ProgressColumn("Momentum",min_value=0, max_value=10, format="%.1f"),
                "volume_signal_score": st.column_config.ProgressColumn("Volume", min_value=0, max_value=10, format="%.1f"),
            },
            height=520,
            use_container_width=True,
            hide_index=True,
        )
        with _us_dl:
            st.markdown("<div style='padding-top:10px'></div>", unsafe_allow_html=True)
            _us_export_cols = ['ticker'] + [c for c in _us_disp if c not in ('yf_link', 'ticker')]
            _us_export = us_fdf[_us_export_cols].sort_values('trend_score', ascending=False).copy()
            st.download_button(
                label="⬇️ Download CSV",
                data=_us_export.to_csv(index=False),
                file_name=f"us_momentum_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="us_scanner_download",
            )

    # ── SUB-INDUSTRY ROTATION MATRIX (day-by-day accumulating) ─────────────
    st.markdown("---")
    st.markdown("### 🧬 Sub-Industry Rotation Matrix")
    st.caption(
        "Capital flow across S&P 500 sub-industries. "
        "Score 0–100 (percentile rank of CompRS within each day). "
        "🟢 Leaders → 🟡 Mid → 🔴 Laggards.  Accumulates daily each time you visit this page."
    )

    # Save today's snapshot (idempotent — safe to call on every page load)
    try:
        save_us_rotation_snapshot(us_df)
    except Exception as _e:
        st.caption(f"⚠️ Could not save rotation snapshot: {_e}")

    # One-time historical backfill: pre-populate 12 months of monthly RS
    # so sparklines are meaningful from day 1 (runs ~30s, skipped thereafter)
    _us_rot_hist_check = load_us_rotation_history(days=365)
    _needs_backfill = (
        _us_rot_hist_check.empty or
        _us_rot_hist_check["date"].dt.to_period("M").nunique() < 2
    )
    if _needs_backfill:
        with st.spinner("📅 Building 12-month rotation history (one-time, ~30 s)…"):
            try:
                _ran = backfill_us_rotation_if_needed(
                    benchmark=us_scan_benchmark,
                    rs_weights=[(5, us_s_w5), (21, us_s_w21), (63, us_s_w63_n)] if us_s_w5 is not None else [(252, 1.0)],
                )
                if _ran:
                    st.success("✅ Historical rotation data ready!")
            except Exception as _be:
                st.caption(f"⚠️ Backfill error: {_be}")

    _us_rot_hist = load_us_rotation_history(days=365)

    if _us_rot_hist.empty:
        st.info("📊 Rotation history will appear after the first page load with live data.")
    else:
        _spiv, _hpiv, _morder = build_rotation_pivot(_us_rot_hist)

        if _spiv.empty:
            st.info("Building rotation history… visit again tomorrow for trend data.")
        else:
            rot_tab1, rot_tab2 = st.tabs(["📋 Rotation Table", "🌡️ Heatmap"])
            with rot_tab1:
                render_us_rotation_table(_spiv, _hpiv, _morder)
            with rot_tab2:
                render_us_rotation_heatmap(_spiv, _hpiv)

    # ── TODAY'S SNAPSHOT: point-in-time RS cross-section heatmap ─────────
    st.markdown("---")
    with st.expander("📸 **Today's Sub-Industry RS Snapshot** (RS5 / RS21 / RS63 / CompRS)", expanded=False):
        if "sub_industry" in us_df.columns:
            _sub_grp = (
                us_df.groupby(["sector", "sub_industry"])[["rs_5d", "rs_21d", "rs_63d", "comp_rs", "trend_score"]]
                .mean()
                .round(2)
                .reset_index()
                .sort_values("comp_rs", ascending=False)
            )
            if not _sub_grp.empty:
                _sub_z   = _sub_grp[["rs_5d", "rs_21d", "rs_63d", "comp_rs"]].values.tolist()
                _sub_y   = [f"{r['sector']} — {r['sub_industry']}" for _, r in _sub_grp.iterrows()]
                _sub_x   = ["RS5 (1W)", "RS21 (1M)", "RS63 (3M)", "CompRS"]
                _sub_txt = [[f"{v:+.1f}" for v in row] for row in _sub_z]

                fig_sub = go.Figure(go.Heatmap(
                    z=_sub_z, x=_sub_x, y=_sub_y,
                    text=_sub_txt, texttemplate="%{text}",
                    colorscale="RdYlGn", zmid=0, showscale=True,
                    colorbar=dict(title="RS %", thickness=12),
                ))
                fig_sub.update_layout(
                    template="plotly_white",
                    height=max(500, len(_sub_y) * 22),
                    margin=dict(l=280, r=20, t=30, b=30),
                    xaxis=dict(side="top"),
                )
                st.plotly_chart(fig_sub, use_container_width=True, key="us_sub_industry_snapshot")

                st.dataframe(
                    _sub_grp.rename(columns={
                        "sector": "Sector", "sub_industry": "Sub-Industry",
                        "rs_5d": "RS5", "rs_21d": "RS21", "rs_63d": "RS63",
                        "comp_rs": "CompRS", "trend_score": "Avg Trend",
                    }),
                    column_config={
                        "RS5":       st.column_config.NumberColumn(format="%+.2f%%"),
                        "RS21":      st.column_config.NumberColumn(format="%+.2f%%"),
                        "RS63":      st.column_config.NumberColumn(format="%+.2f%%"),
                        "CompRS":    st.column_config.NumberColumn(format="%+.2f%%"),
                        "Avg Trend": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                    },
                    hide_index=True, use_container_width=True, height=400,
                )
        else:
            st.caption("Sub-industry data not available.")

    # ── VCP BACKTEST ─────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📊 **VCP Backtest — Last 6 Months (S&P 500)**", expanded=False):
        st.caption(
            "Walk-forward scan: weekly steps over the last 6 months. "
            "Applies all 5 VCP gates at each point using only data available on that day. "
            "Shows top 15 signals by 10D forward return + summary stats."
        )
        if st.button("▶️ Run VCP Backtest", key="run_vcp_backtest"):
            from utils.advanced_scanners import backtest_vcp_us
            _bt_tickers = us_df.nlargest(150, "trend_score")["ticker"].dropna().tolist()
            with st.spinner(f"Running walk-forward VCP backtest on {len(_bt_tickers)} S&P 500 stocks… (~30–60 s)"):
                _bt_top, _bt_summary = backtest_vcp_us(_bt_tickers, n_months=6, top_n=15)
            if _bt_summary:
                _bm1, _bm2, _bm3, _bm4, _bm5 = st.columns(5)
                _bm1.metric("Total Signals", f"{_bt_summary['total_signals']}")
                _bm2.metric("Hit Rate (10D)", f"{_bt_summary['hit_rate_pct']}%")
                _bm3.metric("Avg Return (10D)", f"{_bt_summary['avg_10d_ret']:+.2f}%")
                _bm4.metric("Best (10D)", f"{_bt_summary['best_10d_ret']:+.2f}%")
                _bm5.metric("Worst (10D)", f"{_bt_summary['worst_10d_ret']:+.2f}%")
            if not _bt_top.empty:
                st.markdown("#### Top 15 Signals by 10D Return")
                _bt_top["INDmoney"] = _bt_top["Ticker"].apply(us_quote_url)
                st.dataframe(
                    _bt_top[["INDmoney", "Signal Date", "Signal Price",
                              "Compression", "Vol Ratio %", "Dist 52W %",
                              "Return 5D %", "Return 10D %", "Return 21D %"]],
                    column_config={
                        "INDmoney":      st.column_config.LinkColumn("Ticker", display_text=us_display_re),
                        "Signal Price":  st.column_config.NumberColumn(format="$%.2f"),
                        "Compression":   st.column_config.NumberColumn("ATR% (10D)", format="%.2f%%"),
                        "Vol Ratio %":   st.column_config.NumberColumn("Vol %", format="%.1f%%"),
                        "Dist 52W %":    st.column_config.NumberColumn("Dist 52W", format="%.1f%%"),
                        "Return 5D %":   st.column_config.NumberColumn("5D Ret", format="%+.2f%%"),
                        "Return 10D %":  st.column_config.NumberColumn("10D Ret", format="%+.2f%%"),
                        "Return 21D %":  st.column_config.NumberColumn("21D Ret", format="%+.2f%%"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )
            elif _bt_summary:
                st.info("No VCP signals found in the last 6 months with the current criteria.")
            else:
                st.warning("Backtest returned no data — check internet connection or reduce ticker count.")
