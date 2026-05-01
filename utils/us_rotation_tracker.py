"""
US Sub-Industry Rotation Tracker
=================================
Mirrors the Nifty sub_industry_rotation.csv pattern for the S&P 500 universe.

Schema (data/us_sub_industry_rotation.csv):
  date, sector, sub_industry, comp_rs, rs_5d, rs_21d, rs_63d,
  avg_trend_score, score_0_100, top_tickers

• score_0_100: percentile rank of comp_rs within that day's snapshot (0–100)
• top_tickers: comma-separated top-3 tickers by trend_score in each sub-industry
• One row per (date × sub_industry)
• Appended daily; old rows kept for 365 days
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import yfinance as yf

ROTATION_FILE = "data/us_sub_industry_rotation.csv"
ROTATION_COLS = [
    "date", "sector", "sub_industry",
    "comp_rs", "rs_5d", "rs_21d", "rs_63d",
    "avg_trend_score", "score_0_100", "top_tickers",
]


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_us_rotation_snapshot(us_df: pd.DataFrame) -> None:
    """
    Compute sub-industry aggregates from today's scan results and append
    to the rotation history CSV.  Idempotent: replaces today's rows if
    the page is visited more than once in a day.
    """
    if us_df is None or us_df.empty:
        return
    if "sub_industry" not in us_df.columns:
        return

    today = datetime.now().strftime("%Y-%m-%d")
    os.makedirs("data", exist_ok=True)

    # --- Build today's rows ---
    rs_cols = {
        "comp_rs": "comp_rs", "rs_5d": "rs_5d",
        "rs_21d": "rs_21d", "rs_63d": "rs_63d",
        "trend_score": "avg_trend_score",
    }
    agg_map = {col: "mean" for col in rs_cols if col in us_df.columns}
    grp = (
        us_df.groupby(["sector", "sub_industry"])
        .agg(agg_map)
        .reset_index()
        .rename(columns={"trend_score": "avg_trend_score"})
    )

    # Top-3 tickers by trend_score per sub-industry
    def _top3(rows):
        return ", ".join(
            rows.nlargest(3, "trend_score")["ticker"].tolist()
            if "ticker" in rows.columns and "trend_score" in rows.columns
            else []
        )

    top_map = (
        us_df.groupby("sub_industry")
        .apply(_top3)
        .reset_index()
        .rename(columns={0: "top_tickers"})
    )
    grp = grp.merge(top_map, on="sub_industry", how="left")

    # Percentile rank score_0_100 (within today)
    if "comp_rs" in grp.columns:
        rs_vals = grp["comp_rs"].fillna(0)
        rs_min, rs_max = rs_vals.min(), rs_vals.max()
        if rs_max > rs_min:
            grp["score_0_100"] = ((rs_vals - rs_min) / (rs_max - rs_min) * 100).round(0).astype(int)
        else:
            grp["score_0_100"] = 50
    else:
        grp["score_0_100"] = 50

    grp["date"] = today

    # Round float columns
    for col in ["comp_rs", "rs_5d", "rs_21d", "rs_63d", "avg_trend_score"]:
        if col in grp.columns:
            grp[col] = grp[col].round(2)

    # Ensure column order
    for col in ROTATION_COLS:
        if col not in grp.columns:
            grp[col] = None
    today_df = grp[ROTATION_COLS]

    # --- Merge with history ---
    if os.path.exists(ROTATION_FILE):
        hist = pd.read_csv(ROTATION_FILE)
        hist = hist[hist["date"] != today]          # drop stale today rows
    else:
        hist = pd.DataFrame(columns=ROTATION_COLS)

    combined = pd.concat([hist, today_df], ignore_index=True)

    # Keep last 365 days
    combined["date"] = pd.to_datetime(combined["date"])
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=365)
    combined = combined[combined["date"] >= cutoff]
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")

    combined.to_csv(ROTATION_FILE, index=False)
    print(f"[US ROTATION] Saved {len(today_df)} sub-industry rows for {today}", flush=True)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_us_rotation_history(days: int = 365) -> pd.DataFrame:
    if not os.path.exists(ROTATION_FILE):
        return pd.DataFrame()
    df = pd.read_csv(ROTATION_FILE)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    if days:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        df = df[df["date"] >= cutoff]
    return df


# ---------------------------------------------------------------------------
# Historical backfill
# ---------------------------------------------------------------------------

def _period_rs(close: pd.Series, bench: pd.Series, period: int) -> float:
    """Exact N-interval RS (iloc[-(period+1)] base) clamped to available data."""
    n = min(len(close), len(bench))
    if n < period + 2:
        return 0.0
    s_base = float(close.iloc[-(period + 1)])
    b_base = float(bench.iloc[-(period + 1)])
    if s_base == 0 or b_base == 0:
        return 0.0
    s_ret = (float(close.iloc[-1]) / s_base - 1.0) * 100.0
    b_ret = (float(bench.iloc[-1]) / b_base - 1.0) * 100.0
    return s_ret - b_ret


def backfill_us_rotation_if_needed(
    benchmark: str = "SPY",
    rs_weights=None,  # list of (period, weight) tuples or None
) -> bool:
    """
    One-time historical backfill: downloads 400 days of prices and computes
    monthly sub-industry RS snapshots for the past 12 months.

    Skips if rotation CSV already has ≥ 2 distinct months of data.
    Returns True if backfill ran, False if skipped.
    """
    if rs_weights is None:
        from utils.us_data_engine import DEFAULT_RS_WEIGHTS
        rs_weights = DEFAULT_RS_WEIGHTS

    hist = load_us_rotation_history()
    if not hist.empty:
        n_months = hist["date"].dt.to_period("M").nunique()
        if n_months >= 2:
            return False   # Already has enough history

    # --- Load universe ---
    try:
        from utils.us_data_engine import load_sp500_universe
        universe = load_sp500_universe()
    except Exception as e:
        print(f"[US BACKFILL] Could not load universe: {e}", flush=True)
        return False

    tickers  = universe["ticker"].tolist()
    sub_map  = universe.set_index("ticker")["sub_industry"].to_dict()
    sec_map  = universe.set_index("ticker")["sector"].to_dict()

    all_tickers = [benchmark] + tickers
    print(f"[US BACKFILL] Downloading 400d history for {len(all_tickers)} tickers…", flush=True)

    try:
        raw = yf.download(
            all_tickers, period="400d", interval="1d",
            group_by="ticker", threads=False, progress=False, auto_adjust=True,
        )
    except Exception as e:
        print(f"[US BACKFILL] Download failed: {e}", flush=True)
        return False

    # Extract benchmark close
    try:
        bench_all = (
            raw[benchmark]["Close"].dropna()
            if isinstance(raw.columns, pd.MultiIndex)
            else raw["Close"].dropna()
        )
    except Exception as e:
        print(f"[US BACKFILL] Benchmark extract failed: {e}", flush=True)
        return False

    # --- Month-end dates (last 13 business month-ends, skip most-recent = today) ---
    month_ends = pd.date_range(
        end=pd.Timestamp.now().normalize() - pd.Timedelta(days=1),
        freq="BME", periods=13
    )

    snapshot_rows = []

    for me_date in month_ends:
        # Slice data up to this month-end
        bench_slice = bench_all.loc[:me_date].dropna()
        if len(bench_slice) < 65:
            continue

        # Compute RS for each ticker at this point in time
        ticker_rs: dict[str, dict] = {}
        for ticker in tickers:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if ticker not in raw.columns.get_level_values(0):
                        continue
                    close_all = raw[ticker]["Close"].dropna()
                else:
                    close_all = raw["Close"].dropna()

                close_slice = close_all.loc[:me_date].dropna()
                if len(close_slice) < 65:
                    continue

                rs5  = _period_rs(close_slice, bench_slice, 5)
                rs21 = _period_rs(close_slice, bench_slice, 21)
                rs63 = _period_rs(close_slice, bench_slice, 63)
                comp = sum(rs5 * w if p == 5 else rs21 * w if p == 21 else rs63 * w
                           for p, w in rs_weights)

                ticker_rs[ticker] = {
                    "rs_5d": rs5, "rs_21d": rs21, "rs_63d": rs63, "comp_rs": comp,
                    "sub_industry": sub_map.get(ticker, "Unknown"),
                    "sector": sec_map.get(ticker, "Unknown"),
                }
            except Exception:
                continue

        if not ticker_rs:
            continue

        rs_df = pd.DataFrame(ticker_rs).T.reset_index().rename(columns={"index": "ticker"})

        # Group by sub-industry
        grp = (
            rs_df.groupby(["sector", "sub_industry"])[["rs_5d", "rs_21d", "rs_63d", "comp_rs"]]
            .mean()
            .round(2)
            .reset_index()
        )

        # Top-3 tickers by comp_rs per sub-industry
        top_map = {}
        for sub in grp["sub_industry"]:
            sub_tickers = [t for t, v in ticker_rs.items() if v["sub_industry"] == sub]
            top3 = sorted(sub_tickers, key=lambda t: ticker_rs[t]["comp_rs"], reverse=True)[:3]
            top_map[sub] = ", ".join(top3)
        grp["top_tickers"] = grp["sub_industry"].map(top_map)

        # Percentile rank
        rs_vals = grp["comp_rs"].fillna(0)
        rs_min, rs_max = rs_vals.min(), rs_vals.max()
        grp["score_0_100"] = (
            ((rs_vals - rs_min) / (rs_max - rs_min) * 100).round(0).astype(int)
            if rs_max > rs_min else 50
        )
        grp["avg_trend_score"] = 50.0  # not computable from monthly slice
        grp["date"] = me_date.strftime("%Y-%m-%d")

        for col in ROTATION_COLS:
            if col not in grp.columns:
                grp[col] = None

        snapshot_rows.append(grp[ROTATION_COLS])
        print(f"[US BACKFILL] {me_date.strftime('%Y-%m')} — {len(grp)} sub-industries", flush=True)

    if not snapshot_rows:
        print("[US BACKFILL] No snapshots computed.", flush=True)
        return False

    backfill_df = pd.concat(snapshot_rows, ignore_index=True)

    # Merge with any existing history (today's snapshot already saved)
    if os.path.exists(ROTATION_FILE):
        existing = pd.read_csv(ROTATION_FILE)
        existing["date"] = pd.to_datetime(existing["date"])
        backfill_df["date"] = pd.to_datetime(backfill_df["date"])
        # Backfill only fills historical months; don't overwrite today
        combined = pd.concat([backfill_df, existing], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "sub_industry"], keep="last")
    else:
        combined = backfill_df

    combined["date"] = pd.to_datetime(combined["date"])
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=365)
    combined = combined[combined["date"] >= cutoff].sort_values("date")
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    combined.to_csv(ROTATION_FILE, index=False)

    print(f"[US BACKFILL] Done — {len(combined)} total rows in rotation CSV", flush=True)
    return True


# ---------------------------------------------------------------------------
# Build pivot + sparklines (same logic as Nifty rotation matrix)
# ---------------------------------------------------------------------------

def build_rotation_pivot(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Returns (score_pivot, hover_pivot, month_order).
    score_pivot: rows=sub_industry, cols=month labels (last 12 months)
    hover_pivot: same shape, values=top_tickers string
    month_order: list of month label strings in chronological order
    """
    if history.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    history = history.copy()
    history["month_label"] = history["date"].dt.strftime("%b-%y")
    history["month_key"]   = history["date"].dt.to_period("M")
    history = history.sort_values("date")

    # One row per (month × sub_industry): use latest snapshot of the month
    latest = history.groupby(["month_key", "sub_industry"]).last().reset_index()

    # Last 12 months
    unique_months = sorted(latest["month_key"].unique())[-12:]
    latest = latest[latest["month_key"].isin(unique_months)]

    # Drop sub-industries not present in the most-recent month (naming drift)
    most_recent = max(unique_months)
    current_subs = set(latest[latest["month_key"] == most_recent]["sub_industry"].unique())
    latest = latest[latest["sub_industry"].isin(current_subs)]

    period_to_label = (
        latest[["month_key", "month_label"]]
        .drop_duplicates()
        .set_index("month_key")["month_label"]
        .to_dict()
    )

    score_pivot = latest.pivot_table(
        index="sub_industry", columns="month_key", values="score_0_100", aggfunc="last"
    )
    hover_pivot = latest.pivot_table(
        index="sub_industry", columns="month_key", values="top_tickers", aggfunc="last"
    )

    # Chronological column order
    score_pivot = score_pivot.sort_index(axis=1)
    hover_pivot = hover_pivot.sort_index(axis=1)

    # Forward-fill gaps
    score_pivot = score_pivot.ffill(axis=1)
    hover_pivot = hover_pivot.ffill(axis=1)

    # Rename Period → readable label
    score_pivot.columns = [period_to_label.get(p, str(p)) for p in score_pivot.columns]
    hover_pivot.columns = [period_to_label.get(p, str(p)) for p in hover_pivot.columns]

    month_order = list(score_pivot.columns)

    # Sort rows by latest month (best at top)
    if month_order:
        score_pivot = score_pivot.sort_values(by=month_order[-1], ascending=False, na_position="last")
        hover_pivot = hover_pivot.reindex(score_pivot.index)

    return score_pivot, hover_pivot, month_order


# ---------------------------------------------------------------------------
# Render helpers (called from main.py inside the US Scanner page)
# ---------------------------------------------------------------------------

def render_us_rotation_table(score_pivot: pd.DataFrame, hover_pivot: pd.DataFrame,
                              month_order: list) -> None:
    """Renders the Rotation Table tab (sparkline + signal + MoM)."""
    import streamlit as st

    table_rows = []
    for industry in score_pivot.index:
        row_scores = score_pivot.loc[industry]
        trend_vals = [int(round(float(v))) for v in row_scores.values if pd.notna(v)]

        current_score = float(row_scores.iloc[-1]) if pd.notna(row_scores.iloc[-1]) else 0.0
        prev_score    = float(row_scores.iloc[-2]) if len(row_scores) >= 2 and pd.notna(row_scores.iloc[-2]) else current_score
        mom_change    = round(current_score - prev_score, 1)

        leaders_raw = hover_pivot.loc[industry].iloc[-1] if not hover_pivot.empty else ""
        leaders     = leaders_raw if pd.notna(leaders_raw) else "—"

        if current_score >= 70:
            signal = "🟢 Leader"
        elif current_score >= 40:
            signal = "🟡 Mid"
        else:
            signal = "🔴 Laggard"

        table_rows.append({
            "Sub-Industry": industry,
            "Signal":       signal,
            "Score":        round(current_score),
            "MoM Δ":        mom_change,
            "Trend (12M)":  trend_vals,
            "Top Tickers":  leaders,
        })

    rot_df = (
        pd.DataFrame(table_rows)
        .sort_values("Score", ascending=False)
        .reset_index(drop=True)
    )

    st.dataframe(
        rot_df,
        column_config={
            "Sub-Industry": st.column_config.TextColumn("Sub-Industry", width="medium"),
            "Signal":       st.column_config.TextColumn("Signal",       width="small"),
            "Score":        st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
            "MoM Δ":        st.column_config.NumberColumn("MoM Δ", format="%+.0f",
                                help="Month-over-month percentile rank change"),
            "Trend (12M)":  st.column_config.LineChartColumn("12M Trend", y_min=0, y_max=100),
            "Top Tickers":  st.column_config.TextColumn("Top Tickers", width="medium"),
        },
        hide_index=True,
        use_container_width=True,
        height=min(len(rot_df) * 38 + 42, 950),
    )

    # Quick summary
    if table_rows:
        top3 = rot_df.head(3)
        bot3 = rot_df.tail(3)
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**🟢 Current Leaders**")
            for _, r in top3.iterrows():
                st.caption(f"**{r['Sub-Industry']}** — {r['Score']}/100 | {r['Top Tickers']}")
        with sc2:
            st.markdown("**🔴 Current Laggards**")
            for _, r in bot3.iterrows():
                st.caption(f"**{r['Sub-Industry']}** — {r['Score']}/100 | {r['Top Tickers']}")


def render_us_rotation_heatmap(score_pivot: pd.DataFrame, hover_pivot: pd.DataFrame) -> None:
    """Renders the Heatmap tab."""
    import streamlit as st

    hover_text = []
    for idx in score_pivot.index:
        row_text = []
        for col in score_pivot.columns:
            score_val   = score_pivot.loc[idx, col]
            leaders_raw = hover_pivot.loc[idx, col] if not hover_pivot.empty else ""
            leaders_h   = leaders_raw if pd.notna(leaders_raw) else "N/A"
            if pd.notna(score_val):
                row_text.append(
                    f"<b>{idx}</b><br>Score: {score_val:.0f}/100<br>Month: {col}<br>Leaders: {leaders_h}"
                )
            else:
                row_text.append(f"<b>{idx}</b><br>No data<br>Month: {col}")
        hover_text.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=score_pivot.values,
        x=score_pivot.columns.tolist(),
        y=score_pivot.index.tolist(),
        colorscale=[
            [0.00, "#d32f2f"],
            [0.25, "#ff7043"],
            [0.50, "#fdd835"],
            [0.75, "#66bb6a"],
            [1.00, "#1b5e20"],
        ],
        zmin=0, zmax=100,
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        colorbar=dict(
            title="Score",
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0 🔴", "25", "50 🟡", "75", "100 🟢"],
        ),
        xgap=2, ygap=1,
    ))
    fig.update_layout(
        height=max(600, len(score_pivot) * 22),
        xaxis=dict(title="Month", side="top", tickangle=0),
        yaxis=dict(title="", autorange="reversed", tickfont=dict(size=11)),
        margin=dict(l=240, r=40, t=60, b=20),
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, key="us_rotation_heatmap_chart")
