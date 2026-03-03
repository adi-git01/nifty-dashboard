"""append_turnaround_tab.py — run once to add the Turnaround Radar page to main.py"""
import os

PAGE_CODE = '''

# --- VIEW: TURNAROUND RADAR ---
elif page == "\\U0001f3af Turnaround Radar":
    import plotly.graph_objects as go

    st.markdown(page_header(
        "\\U0001f3af Institutional Turnaround Radar",
        "Beaten-down Nifty 1000 stocks showing early institutional accumulation \\u2014 before V21 qualification."
    ), unsafe_allow_html=True)

    WATCHLIST_CSV = "data/turnaround_watchlist.csv"

    @st.cache_data(ttl=3600)
    def load_turnaround_watchlist():
        if not os.path.exists(WATCHLIST_CSV):
            return pd.DataFrame()
        return pd.read_csv(WATCHLIST_CSV)

    wdf = load_turnaround_watchlist()

    if wdf.empty:
        st.warning("Watchlist not generated yet. Run `python turnaround_screener.py` locally or wait for the next GitHub Actions daily run.")
        st.code("python turnaround_screener.py", language="bash")
        st.stop()

    alert_n = int((wdf["Tier"] == "ALERT").sum())
    ready_n = int((wdf["Tier"] == "READY").sum())
    watch_n = int((wdf["Tier"] == "WATCH").sum())

    TIER_COLORS = {"ALERT": "#00C853", "READY": "#FFB300", "WATCH": "#42A5F5"}

    # --- Tier cards ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("\\U0001f7e2 ALERT", alert_n, help="IAS 80+: RS21+RS63 turning. Strong liq floor. Imminent V21 \\u2014 ready to fire.")
    c2.metric("\\U0001f7e1 READY", ready_n, help="IAS 60-79: Liq stable, RS velocity confirmed. Getting warm.")
    c3.metric("\\U0001f535 WATCH", watch_n, help="IAS 35-59: Early institutional ping. Set MA50 alert.")
    c4.metric("Total Pool", len(wdf))

    st.info("**How to use:** These are BELOW MA50 \\u2014 do NOT buy. Set a price alert at each stock\'s MA50 level. When crossed with CompRS > 0, re-check the V21 scanner for fast-track to portfolio.")

    st.markdown("---")

    # --- Filters ---
    fcols = st.columns([1, 1, 1, 2])
    tier_filter  = fcols[0].multiselect("Tier",  ["ALERT","READY","WATCH"], default=["ALERT","READY","WATCH"])
    cycle_filter = fcols[1].multiselect("Cycle", ["LONG","MID","SHORT"],    default=["LONG","MID","SHORT"])
    min_ias      = fcols[2].slider("Min IAS", 35, 90, 35)
    search       = fcols[3].text_input("Search ticker or sub-industry")

    fdf = wdf[
        wdf["Tier"].isin(tier_filter) &
        wdf["Cycle"].isin(cycle_filter) &
        (wdf["IAS"] >= min_ias)
    ].copy()
    if search:
        fdf = fdf[
            fdf["Ticker"].str.contains(search.upper(), na=False) |
            fdf["Sub_Industry"].str.contains(search, case=False, na=False)
        ]

    # --- Watchlist table ---
    st.markdown(f"### Watchlist ({len(fdf)} stocks)")
    # Build screener.in links (strip .NS / .BO suffix for screener URL)
    display_df = fdf.copy().sort_values("IAS", ascending=False).reset_index(drop=True)
    display_df["Screener"] = display_df["Ticker"].str.replace(r"\\.(NS|BO)$", "", regex=True).apply(
        lambda sym: f"https://www.screener.in/company/{sym}/"
    )

    display_cols = ["Ticker","Screener","Sub_Industry","Cycle","CMP","Off_52W_High","RS21","RS63",
                    "CompRS","Liq5Cr","LiqFromLow","IAS","Tier","Off_MA50","V21_CRS_Gap","V21_MA50_Gap"]
    available = [c for c in display_cols if c in display_df.columns]

    st.dataframe(
        display_df[available],
        column_config={
            "Ticker":       st.column_config.TextColumn("Ticker", width=90),
            "Screener":     st.column_config.LinkColumn("Screener", display_text="\\U0001f517 View", width=80),
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
            text=[f"IAS {v}" for v in pipe_df["IAS"]],
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
'''

main_path = "main.py"
with open(main_path, "r", encoding="utf-8") as f:
    content = f.read()

if '"\\U0001f3af Turnaround Radar"' in content or "Turnaround Radar" in content.split("# --- VIEW: TURNAROUND")[-1] if "# --- VIEW: TURNAROUND" in content else False:
    print("Turnaround page already appended.")
else:
    with open(main_path, "a", encoding="utf-8") as f:
        f.write(PAGE_CODE)
    print("Turnaround Radar page appended to main.py")
    # Verify
    with open(main_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"main.py now has {len(lines)} lines")
