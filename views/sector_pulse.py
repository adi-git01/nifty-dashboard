"""Sector Pulse page view (extracted from main.py)."""
from __future__ import annotations

from datetime import datetime
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.analytics_engine import get_monthly_alpha_calendar
from utils.ui_components import page_header


def render(df: pd.DataFrame, *, cycle_position) -> None:

    st.markdown(page_header("📊 Sector Pulse", "Deep dive into sector performance and stock rotation dynamics"), unsafe_allow_html=True)

    # === SECTOR TIMING SIGNALS ===
    st.markdown("### 🎯 Sector Rotation & Timing Signals")

    # Calculate current market mood score for signals
    try:
        from utils.trend_engine import load_mood_history
        mood_history = load_mood_history()
        current_score = mood_history['avg_trend_score'].iloc[-1] if not mood_history.empty else 50
    except Exception:
        current_score = 50
    
    signal_cols = st.columns([1, 1])

    with signal_cols[0]:
        # Sector Rotation Signal Panel - Based on Correlation Analysis
        st.markdown("**📊 Current Rotation Signal**")
    
        if current_score < 40:
            st.success("🔥 **STRONG BUY**: Auto, Energy, Midcap")
            st.info("✅ **BUY**: Realty, PSE, Infra")
            st.warning("⏸️ **HOLD**: IT (wait for high mood)")
            st.caption("Expected 90D: Auto +10%, Energy +4%, Midcap +4%")
        elif current_score > 65:
            st.success("✅ **BUY**: IT sector (+8.7% expected)")
            st.error("❌ **AVOID**: Realty (-5%), Energy (-3%), Midcap (-1%)")
            st.warning("💰 Book profits in cyclicals")
        else:
            # 40-65 range
            st.info("⏸️ **HOLD**: Current positions")
            if current_score < 50:
                st.caption("Approaching BUY zone - watch Auto, Energy")
            else:
                st.caption("Approaching CAUTION zone - watch IT")
        st.caption(f"Market Mood Score: {current_score:.0f}/100 | Optimal Horizon: 60-90 days")

    with signal_cols[1]:
        # Enhanced Sector Signal Table based on analysis
        st.markdown("**📈 Sector Correlation Matrix**")
    
        # Tiered sector data from analysis
        sector_data = {
            "Sector": ["Auto 🏆", "Energy 🏆", "Midcap", "Realty", "IT 🔄"],
            "Corr": ["-0.72", "-0.78", "-0.81", "-0.64", "+0.26"],
            "@ <40": ["+10%", "+4%", "+4%", "+4%", "+5%"],
            "@ >70": ["+3%", "-3%", "-1%", "-5%", "+9%"],
        }
        st.dataframe(
            pd.DataFrame(sector_data),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Sector": st.column_config.TextColumn("Sector"),
                "Corr": st.column_config.TextColumn("Signal"),
                "@ <40": st.column_config.TextColumn("Low Mood"),
                "@ >70": st.column_config.TextColumn("High Mood"),
            }
        )
    
    st.markdown("---")

    # === INDUSTRY LEADERSHIP BOARD (breadth + trend + change; utils/industry_board.py) ===
    try:
        from utils import industry_board_view
        industry_board_view.render(df)
    except Exception as _e:
        st.warning(f"Industry leadership board unavailable: {_e}")
    st.markdown("---")

    # === SUB-INDUSTRY ROTATION HEATMAP ===
    st.markdown("### 🧬 Sub-Industry Rotation Matrix — short-term RS history")
    st.caption("Daily history of the short-term score (1-week / 1-month / 3-month RS blend only). Use it to see how a ranking moved over the year; use the Leadership Board above to judge who leads today.")

    sub_ind_file = "data/sub_industry_rotation.csv"
    if os.path.exists(sub_ind_file):
        try:
            sub_df = pd.read_csv(sub_ind_file)
            if not sub_df.empty:
                # Ensure score_0_100 exists (backward compat with old CSV)
                if 'score_0_100' not in sub_df.columns:
                    for date_val in sub_df['record_date'].unique():
                        mask = sub_df['record_date'] == date_val
                        rs_vals = sub_df.loc[mask, 'rs_momentum']
                        rs_min, rs_max = rs_vals.min(), rs_vals.max()
                        if rs_max > rs_min:
                            sub_df.loc[mask, 'score_0_100'] = ((rs_vals - rs_min) / (rs_max - rs_min) * 100).round(0).astype(int)
                        else:
                            sub_df.loc[mask, 'score_0_100'] = 50
            
                # Create month label for grouping (e.g., "Mar-26")
                sub_df['record_date'] = pd.to_datetime(sub_df['record_date'], format='mixed')
                sub_df['month_label'] = sub_df['record_date'].dt.strftime('%b-%y')
            
                # For each month, take the latest snapshot (in case multiple dates per month)
                sub_df = sub_df.sort_values('record_date')
                sub_df['month_key'] = sub_df['record_date'].dt.to_period('M')
                latest_per_month = sub_df.groupby(['month_key', 'sub_industry']).last().reset_index()
            
                # Pivot: rows = sub-industries, columns = months (use Period month_key for correct chronological ordering)
                # Last 12 months
                unique_months = sorted(latest_per_month['month_key'].unique())[-12:]
                latest_per_month = latest_per_month[latest_per_month['month_key'].isin(unique_months)]

                # Filter to only sub-industries present in the MOST RECENT month.
                # This drops ghost rows from old naming conventions (e.g. "Metals & Mining"
                # was later renamed to "Non-Ferrous Metals"), which would otherwise show as
                # mostly-white rows with data only on the left side of the heatmap.
                most_recent_month = max(unique_months)
                current_industries = set(
                    latest_per_month[latest_per_month['month_key'] == most_recent_month]['sub_industry'].unique()
                )
                latest_per_month = latest_per_month[latest_per_month['sub_industry'].isin(current_industries)]

                # Map each month_key Period to its month_label string (e.g. 'Jan-26')
                period_to_label = (
                    latest_per_month[['month_key', 'month_label']]
                    .drop_duplicates()
                    .set_index('month_key')['month_label']
                    .to_dict()
                )

                # Pivot on Period (guarantees chronological column order after sort)
                score_pivot = latest_per_month.pivot_table(
                    index='sub_industry', columns='month_key', values='score_0_100', aggfunc='last'
                )
                hover_pivot = latest_per_month.pivot_table(
                    index='sub_industry', columns='month_key', values='top_components', aggfunc='last'
                )

                # Sort columns chronologically (Periods sort naturally)
                score_pivot = score_pivot.sort_index(axis=1)
                hover_pivot = hover_pivot.sort_index(axis=1)

                # Forward-fill gaps so months with partial EOD data (e.g. Jan-26)
                # show the last known score instead of blank white cells
                score_pivot = score_pivot.ffill(axis=1)
                hover_pivot = hover_pivot.ffill(axis=1)

                # Rename Period columns -> readable month labels
                score_pivot.columns = [period_to_label.get(p, str(p)) for p in score_pivot.columns]
                hover_pivot.columns = [period_to_label.get(p, str(p)) for p in hover_pivot.columns]

                # Readable ordered list of month labels
                month_order = list(score_pivot.columns)

                # Sort rows by latest month score (best at top)
                if month_order:
                    last_month_col = month_order[-1]
                    score_pivot = score_pivot.sort_values(by=last_month_col, ascending=False, na_position='last')
                    hover_pivot = hover_pivot.reindex(score_pivot.index)

                # ── Compute 7D delta from raw daily data (so table updates every day) ──
                # Use last 10 records per sub_industry; compare today vs ~7 trading days ago
                _recent = sub_df[sub_df['sub_industry'].isin(current_industries)].copy()
                _recent = _recent.sort_values('record_date')
                _score_7d_ago = {}
                _score_today  = {}
                for _si, _grp in _recent.groupby('sub_industry'):
                    _grp = _grp.sort_values('record_date')
                    if len(_grp) >= 1:
                        _score_today[_si]  = float(_grp['score_0_100'].iloc[-1])
                    if len(_grp) >= 7:
                        _score_7d_ago[_si] = float(_grp['score_0_100'].iloc[-7])
                    elif len(_grp) >= 2:
                        _score_7d_ago[_si] = float(_grp['score_0_100'].iloc[0])

                # ── Two-tab view: sparkline table (primary) + heatmap (fallback) ──
                rot_tab1, rot_tab2 = st.tabs(["📋 Rotation Table", "🌡️ Heatmap"])

                with rot_tab1:
                    # Build one row per sub-industry
                    table_rows = []
                    for industry in score_pivot.index:
                        row_scores = score_pivot.loc[industry]
                        trend_vals = [int(round(float(v))) for v in row_scores.values if pd.notna(v)]

                        current_score = _score_today.get(industry, float(row_scores.iloc[-1]) if pd.notna(row_scores.iloc[-1]) else 0.0)
                        prev_month    = float(row_scores.iloc[-2]) if len(row_scores) >= 2 and pd.notna(row_scores.iloc[-2]) else current_score
                        mom_change    = round(current_score - prev_month, 1)
                        week_score    = _score_7d_ago.get(industry, current_score)
                        wow_change    = round(current_score - week_score, 1)

                        leaders_raw = hover_pivot.loc[industry].iloc[-1]
                        leaders = leaders_raw if pd.notna(leaders_raw) else '—'

                        if current_score >= 70:
                            signal = '🟢 Leader'
                        elif current_score >= 40:
                            signal = '🟡 Mid'
                        else:
                            signal = '🔴 Laggard'

                        table_rows.append({
                            'Sub-Industry': industry,
                            'Signal':       signal,
                            'Score':        round(current_score),
                            '7D Δ':         wow_change,
                            'MoM Δ':        mom_change,
                            'Trend (12M)':  trend_vals,
                            'Leaders':      leaders,
                        })

                    rot_table_df = (pd.DataFrame(table_rows)
                                    .sort_values('Score', ascending=False)
                                    .reset_index(drop=True))

                    st.dataframe(
                        rot_table_df,
                        column_config={
                            'Sub-Industry': st.column_config.TextColumn("Sub-Industry", width="medium"),
                            'Signal':       st.column_config.TextColumn("Signal",       width="small"),
                            'Score':        st.column_config.ProgressColumn(
                                                "Score", min_value=0, max_value=100, format="%d"),
                            '7D Δ':         st.column_config.NumberColumn(
                                                "7D Δ", format="%+.0f",
                                                help="Score change over last 7 trading days — updates daily"),
                            'MoM Δ':        st.column_config.NumberColumn(
                                                "MoM Δ", format="%+.0f",
                                                help="Month-over-month percentile rank change"),
                            'Trend (12M)':  st.column_config.LineChartColumn(
                                                "12M Trend", y_min=0, y_max=100),
                            'Leaders':      st.column_config.TextColumn("Leaders", width="medium"),
                        },
                        hide_index=True,
                        use_container_width=True,
                        height=min(len(rot_table_df) * 38 + 42, 950),
                    )

                    # Quick summary below table
                    if table_rows:
                        top3_t = rot_table_df.head(3)
                        bot3_t = rot_table_df.tail(3)
                        sc1, sc2 = st.columns(2)
                        with sc1:
                            st.markdown("**🟢 Current Leaders**")
                            for _, r in top3_t.iterrows():
                                st.caption(f"**{r['Sub-Industry']}** — {r['Score']}/100 | {r['Leaders']}")
                        with sc2:
                            st.markdown("**🔴 Current Laggards**")
                            for _, r in bot3_t.iterrows():
                                st.caption(f"**{r['Sub-Industry']}** — {r['Score']}/100 | {r['Leaders']}")

                with rot_tab2:
                    # ── Original heatmap (fallback / detail view) ──
                    hover_text = []
                    for idx in score_pivot.index:
                        row_text = []
                        for col in score_pivot.columns:
                            score_val = score_pivot.loc[idx, col]
                            leaders_h = hover_pivot.loc[idx, col] if pd.notna(hover_pivot.loc[idx, col]) else "N/A"
                            if pd.notna(score_val):
                                row_text.append(f"<b>{idx}</b><br>Score: {score_val:.0f}/100<br>Month: {col}<br>Leaders: {leaders_h}")
                            else:
                                row_text.append(f"<b>{idx}</b><br>No data<br>Month: {col}")
                        hover_text.append(row_text)

                    fig = go.Figure(data=go.Heatmap(
                        z=score_pivot.values,
                        x=score_pivot.columns.tolist(),
                        y=[s.replace('.NS', '') for s in score_pivot.index.tolist()],
                        colorscale=[
                            [0.0,  '#d32f2f'],
                            [0.25, '#ff7043'],
                            [0.5,  '#fdd835'],
                            [0.75, '#66bb6a'],
                            [1.0,  '#1b5e20'],
                        ],
                        zmin=0, zmax=100,
                        text=hover_text,
                        hovertemplate='%{text}<extra></extra>',
                        colorbar=dict(
                            title="Score",
                            tickvals=[0, 25, 50, 75, 100],
                            ticktext=["0 \U0001f534", "25", "50 \U0001f7e1", "75", "100 \U0001f7e2"],
                        ),
                        xgap=2, ygap=1,
                    ))
                    fig.update_layout(
                        height=max(600, len(score_pivot) * 22),
                        xaxis=dict(title="Month", side="top", tickangle=0),
                        yaxis=dict(title="", autorange="reversed", tickfont=dict(size=11)),
                        margin=dict(l=200, r=40, t=60, b=20),
                        template='plotly_white',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sub-Industry data is empty. Run trading_engine.py first.")
        except Exception as e:
            st.error(f"Error loading Sub-Industry rotation: {e}")
    else:
        st.info("Sub-Industry rotation engine has not run yet. (Run trading_engine.py locally or via GitHub Action to generate backend matrix)")

    st.markdown("---")

    # === GRANULAR INDUSTRY SEASONALITY CALENDAR ===
    st.markdown("### 📅 Monthly Alpha Calendar (12-Month Lookahead)")
    st.caption("Which granular industries historically peak or bottom out in each calendar month? Price Lead/Lag delays are explicitly calculated (e.g. Lead: 3mo means buy 3 months *before* this peak).")

    alpha_cal_df = get_monthly_alpha_calendar()

    if not alpha_cal_df.empty:
        # Style the dataframe to highlight the current month row
        current_month_abbr = datetime.now().strftime("%b")
    
        def highlight_current_month(row):
            if row['📅 Month'] == current_month_abbr:
                return ['background-color: rgba(99, 91, 255, 0.15); font-weight: bold'] * len(row)
            return [''] * len(row)
        
        styled_cal = alpha_cal_df.style.apply(highlight_current_month, axis=1)
    
        st.dataframe(
            styled_cal,
            column_config={
                "📅 Month": st.column_config.TextColumn("Month", width="small"),
                "🟢 Historical Best (Accumulate)": st.column_config.TextColumn("🟢 Historical Best (Accumulate)", width="large"),
                "🔴 Historical Worst (Avoid/Lighten)": st.column_config.TextColumn("🔴 Historical Worst (Avoid/Lighten)", width="large"),
            },
            hide_index=True,
            use_container_width=True,
            height=450  # Enough height to show all 12 months without scrolling
        )
    else:
        st.info("Seasonality Engine offline. Missing granular_industry_analysis.csv")

    st.markdown("---")

    # === SECTOR-LEVEL CYCLE POSITION ===
    st.markdown("### 🔄 Sector Margin Cycle")
    st.caption("Which sectors are at which stage of their profitability cycle? Based on Operating Margin vs historical average.")

    # Sample 1 stock from each major cyclical sector
    cyclical_sectors = {
        "Iron & Steel": df[df['sector'].str.contains('Iron', case=False, na=False)]['ticker'].head(1).tolist(),
        "Cement": df[df['sector'].str.contains('Cement', case=False, na=False)]['ticker'].head(1).tolist(),
        "Power": df[df['sector'].str.contains('Power', case=False, na=False)]['ticker'].head(1).tolist(),
        "Oil & Gas": df[df['sector'].str.contains('Oil|Refin', case=False, na=False)]['ticker'].head(1).tolist(),
        "Auto": df[df['sector'].str.contains('Auto|Vehicle', case=False, na=False)]['ticker'].head(1).tolist(),
        "IT Services": df[df['sector'].str.contains('Software|IT', case=False, na=False)]['ticker'].head(1).tolist(),
    }

    sector_cycles = []
    with st.spinner("Analyzing sector margin cycles..."):
        for sector_name, tickers in cyclical_sectors.items():
            if tickers:
                cycle = cycle_position(tickers[0])
                if "error" not in cycle:
                    phase_emoji = {"EARLY_RECOVERY": "🟢", "MID_CYCLE": "🟡", "LATE_CYCLE": "🟠", "DOWNTURN": "🔴"}.get(cycle["phase"], "⚪")
                    sector_cycles.append({
                        "Sector": sector_name,
                        "Phase": f"{phase_emoji} {cycle['phase'].replace('_', ' ')}",
                        "OPM %": f"{cycle['current_margin']:.1f}%",
                        "vs Avg": f"{cycle['margin_vs_avg']:.2f}x",
                        "Trend": "↗" if cycle["trend"] == "rising" else "↘" if cycle["trend"] == "falling" else "→"
                    })

    if sector_cycles:
        sector_cycle_df = pd.DataFrame(sector_cycles)
    
        # Group by phase
        col_early, col_mid, col_late, col_down = st.columns(4)
        early = [s["Sector"] for s in sector_cycles if "RECOVERY" in s["Phase"]]
        mid = [s["Sector"] for s in sector_cycles if "MID" in s["Phase"]]
        late = [s["Sector"] for s in sector_cycles if "LATE" in s["Phase"]]
        down = [s["Sector"] for s in sector_cycles if "DOWNTURN" in s["Phase"]]
    
        with col_early:
            st.markdown("**🟢 Early Recovery**")
            st.caption("Buy opportunity")
            for s in early: st.write(f"• {s}")
            if not early: st.write("*None*")
    
        with col_mid:
            st.markdown("**🟡 Mid-Cycle**")
            st.caption("Hold/Ride")
            for s in mid: st.write(f"• {s}")
            if not mid: st.write("*None*")
    
        with col_late:
            st.markdown("**🟠 Late Cycle**")
            st.caption("Be cautious")
            for s in late: st.write(f"• {s}")
            if not late: st.write("*None*")
    
        with col_down:
            st.markdown("**🔴 Downturn**")
            st.caption("Avoid")
            for s in down: st.write(f"• {s}")
            if not down: st.write("*None*")
    
        with st.expander("📊 Detailed Sector Cycle Data"):
            st.dataframe(sector_cycle_df, hide_index=True, use_container_width=True)

    st.markdown("---")

    # === SECTOR DRILL-DOWN ===
    st.markdown("### 🔍 Sector Deep Dive")

    all_sectors = sorted(df['sector'].dropna().unique().tolist())
    selected_sector = st.selectbox("Select a Sector to Analyze", all_sectors, key="sector_pulse_select")

    # Filter stocks for selected sector
    sector_stocks = df[df['sector'] == selected_sector].copy()

    if sector_stocks.empty:
        st.warning(f"No stocks found in {selected_sector}")
    else:
        # Sector stats
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        s_col1.metric("📊 Stocks", len(sector_stocks))
        s_col2.metric("📈 Avg Score", f"{sector_stocks['overall'].mean():.1f}")
        s_col3.metric("🚀 Avg Trend", f"{sector_stocks['trend_score'].mean():.0f}")
        uptrend_pct = len(sector_stocks[sector_stocks['trend_signal'].isin(['STRONG UPTREND', 'UPTREND'])]) / len(sector_stocks) * 100
        s_col4.metric("💹 % Uptrending", f"{uptrend_pct:.0f}%")
    
        # Top performers in sector
        st.markdown(f"#### 🏆 Top Performers in {selected_sector}")
        top_in_sector = sector_stocks.nlargest(10, 'overall')[['ticker', 'name', 'price', 'overall', 'trend_score', 'trend_signal', 'return_1m', 'return_3m']].copy()
        top_in_sector['screener_link'] = "https://www.screener.in/company/" + top_in_sector['ticker'].str.replace('.NS', '', regex=False) + "/"
        st.dataframe(
            top_in_sector[['screener_link', 'name', 'price', 'overall', 'trend_score', 'trend_signal', 'return_1m', 'return_3m']],
            column_config={
                "screener_link": st.column_config.LinkColumn("Ticker", display_text=r"https://www\.screener\.in/company/(.*?)/"),
                "name": "Company",
                "price": st.column_config.NumberColumn("Price", format="₹%.2f"),
                "overall": st.column_config.ProgressColumn("Score", min_value=0, max_value=10, format="%.1f"),
                "trend_score": st.column_config.ProgressColumn("Trend", min_value=0, max_value=100),
                "trend_signal": "Signal",
                "return_1m": st.column_config.NumberColumn("1M Return", format="%.1f%%"),
                "return_3m": st.column_config.NumberColumn("3M Return", format="%.1f%%"),
            },
            hide_index=True
        )
    
        # === MARGIN CYCLE POSITION ===
        st.markdown("---")
        st.markdown("### 🔄 Margin Cycle Position")
        st.caption("Where are stocks in their profitability cycle? Based on Operating Margin vs historical average.")
    
        with st.expander("📖 Cycle Phases Explained", expanded=False):
            st.markdown("""
            | Phase | Meaning | Action |
            |-------|---------|--------|
            | 🟢 **EARLY_RECOVERY** | Margins below avg but improving | 💡 Potential BUY |
            | 🟡 **MID_CYCLE** | Margins at/above avg, stable | ✅ HOLD/RIDE |
            | 🟠 **LATE_CYCLE** | Margins above avg but falling | ⚠️ BE CAUTIOUS |
            | 🔴 **DOWNTURN** | Margins below avg, still falling | ❌ AVOID |
            """)
    
        # Calculate cycle for each stock (limited to first 15 to avoid API limits)
        cycle_data = []
        sample_tickers = sector_stocks.nlargest(15, 'overall')['ticker'].tolist()
    
        with st.spinner(f"Analyzing margin cycles for {len(sample_tickers)} stocks..."):
            for ticker in sample_tickers:
                cycle = cycle_position(ticker)
                if "error" not in cycle:
                    cycle_data.append({
                        "Ticker": ticker.replace(".NS", ""),
                        "Phase": cycle["phase"],
                        "OPM %": cycle["current_margin"],
                        "vs Avg": f"{cycle['margin_vs_avg']:.2f}x",
                        "Trend": "↗" if cycle["trend"] == "rising" else "↘" if cycle["trend"] == "falling" else "→",
                        "Cyclical?": "Yes" if cycle.get("is_cyclical", False) else "No"
                    })
    
        if cycle_data:
            cycle_df = pd.DataFrame(cycle_data)
        
            # Add phase emojis
            phase_emoji = {
                "EARLY_RECOVERY": "🟢",
                "MID_CYCLE": "🟡",
                "LATE_CYCLE": "🟠",
                "DOWNTURN": "🔴",
            }
            cycle_df["Phase"] = cycle_df["Phase"].apply(lambda x: f"{phase_emoji.get(x, '⚪')} {x}")
        
            # Show phase distribution
            phase_counts = cycle_df["Phase"].value_counts()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🟢 Recovery", len([p for p in cycle_df["Phase"] if "RECOVERY" in p]))
            c2.metric("🟡 Mid-Cycle", len([p for p in cycle_df["Phase"] if "MID_CYCLE" in p]))
            c3.metric("🟠 Late Cycle", len([p for p in cycle_df["Phase"] if "LATE" in p]))
            c4.metric("🔴 Downturn", len([p for p in cycle_df["Phase"] if "DOWNTURN" in p]))
        
            st.dataframe(cycle_df, hide_index=True, use_container_width=True)
        else:
            st.info("Cycle data not available for this sector (may lack operating income data)")
    
        st.markdown("---")
    
        # === STOCK ROTATION CHART ===
        st.markdown("### 🔄 Stock Rotation (Within Sector)")
        st.caption("X: 3-Month Return | Y: 1-Month Return | Color: Trend Signal | Size: Market Cap")
    

    
        # Ensure return columns exist, fill missing with 0
        if 'return_1m' not in sector_stocks.columns:
            sector_stocks['return_1m'] = 0
        if 'return_3m' not in sector_stocks.columns:
            sector_stocks['return_3m'] = 0
    
        # Fill NaN values with 0 for returns and min market cap for size
        sector_stocks['return_1m'] = sector_stocks['return_1m'].fillna(0)
        sector_stocks['return_3m'] = sector_stocks['return_3m'].fillna(0)
        sector_stocks['marketCap'] = sector_stocks['marketCap'].fillna(1000000000)  # Default 1000 Cr
    
        # Filter stocks with valid data
        chart_df = sector_stocks[
            (sector_stocks['marketCap'] > 0)
        ].copy()
    
        if len(chart_df) >= 3:
            # Map trend signals to colors
            color_map = {
                'STRONG UPTREND': '#00C853',  # Green
                'UPTREND': '#69F0AE',         # Light green
                'NEUTRAL': '#FFD600',         # Yellow
                'DOWNTREND': '#FF6D00',       # Orange
                'STRONG DOWNTREND': '#D50000' # Red
            }
        
            fig = px.scatter(
                chart_df,
                x='return_3m',
                y='return_1m',
                size='marketCap',
                color='trend_signal',
                hover_name='ticker',
                hover_data={'name': True, 'overall': ':.1f', 'return_1m': ':.1f', 'return_3m': ':.1f'},
                color_discrete_map=color_map,
                labels={
                    'return_3m': '3-Month Return (%)',
                    'return_1m': '1-Month Return (%)',
                    'trend_signal': 'Trend'
                },
            )
        
            # Add quadrant lines
            fig.add_hline(y=0, line_dash="solid", line_width=2, line_color="#E0E0E0", opacity=1.0)
            fig.add_vline(x=0, line_dash="solid", line_width=2, line_color="#E0E0E0", opacity=1.0)
        
            # Dynamically calculate label positions based on max axis bounds
            x_max = max(abs(chart_df['return_3m'].min()), abs(chart_df['return_3m'].max()), 10)
            y_max = max(abs(chart_df['return_1m'].min()), abs(chart_df['return_1m'].max()), 5)
        
            # Update axes to fit data tightly with padding
            fig.update_xaxes(range=[-x_max*1.15, x_max*1.15])
            fig.update_yaxes(range=[-y_max*1.15, y_max*1.15])
        
            # Add light background quadrant colors using dynamic extents
            fig.add_shape(type="rect", x0=0, y0=0, x1=x_max*2, y1=y_max*2, fillcolor="#00C853", opacity=0.03, line_width=0, layer="below")
            fig.add_shape(type="rect", x0=-x_max*2, y0=0, x1=0, y1=y_max*2, fillcolor="#2196F3", opacity=0.03, line_width=0, layer="below")
            fig.add_shape(type="rect", x0=0, y0=-y_max*2, x1=x_max*2, y1=0, fillcolor="#FF9800", opacity=0.03, line_width=0, layer="below")
            fig.add_shape(type="rect", x0=-x_max*2, y0=-y_max*2, x1=0, y1=0, fillcolor="#F44336", opacity=0.03, line_width=0, layer="below")
        
            fig.add_annotation(x=x_max*0.7, y=y_max*0.7, text="🚀 Winners", showarrow=False, font=dict(color="green", size=12))
            fig.add_annotation(x=-x_max*0.7, y=y_max*0.7, text="📈 Turnarounds", showarrow=False, font=dict(color="blue", size=12))
            fig.add_annotation(x=x_max*0.7, y=-y_max*0.7, text="⚠️ Fading", showarrow=False, font=dict(color="orange", size=12))
            fig.add_annotation(x=-x_max*0.7, y=-y_max*0.7, text="🔴 Laggards", showarrow=False, font=dict(color="red", size=12))
        
            fig.update_layout(
                height=450,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                template='plotly_white',
            )
        
            st.plotly_chart(fig, use_container_width=True, key="stock_rotation_chart")
        else:
            st.info(f"Not enough stocks with valid data in this sector for rotation chart. Found {len(chart_df)} stocks.")
