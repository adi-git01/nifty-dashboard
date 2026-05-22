"""
Advanced Quantitative Scanners
================================
1. VCP  — Volatility Contraction Pattern
2. RS Divergence — Green in a sea of Red (with persistent log)
3. Zero-Lag Earnings Shocks — Day-0 gap on volume (with persistent log)

Signal logs:
  data/rs_divergence_log.csv   — appended each time RS divergence fires
  data/earnings_shock_log.csv  — appended each time an earnings shock fires
Both logs are refreshed with live prices on every page visit so returns
are always current (mirrors the IAS signal log pattern).
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

RS_LOG_FILE      = "data/rs_divergence_log.csv"
SHOCK_LOG_FILE   = "data/earnings_shock_log.csv"

RS_LOG_COLS = [
    "signal_date", "ticker", "name", "sector",
    "signal_price", "current_price",
    "stock_ret_on_day", "nifty_ret_on_day", "delta_rs", "dist_52w",
    "return_since_signal", "days_held", "status",
]
SHOCK_LOG_COLS = [
    "signal_date", "ticker", "name", "sector",
    "signal_price", "current_price",
    "jump_pct", "vol_mult", "pead_action",
    "return_since_signal", "return_5d", "return_21d", "days_held", "status",
]

US_RS_LOG_FILE    = "data/us_rs_divergence_log.csv"
US_SHOCK_LOG_FILE = "data/us_earnings_shock_log.csv"


# ---------------------------------------------------------------------------
# VCP
# ---------------------------------------------------------------------------

def calculate_vcp_compression(df, window=10):
    """10-day ATR % — lower = tighter."""
    if len(df) < window + 1:
        return float("inf")
    recent = df.iloc[-window:]
    atr_pct = ((recent["High"].values - recent["Low"].values) / recent["Low"].values).mean() * 100
    return atr_pct


def find_vcp_setups(market_df, full_history_dict):
    """
    Finds Volatility Contraction Patterns (VCP).

    Criteria (calibrated against Indian market typical volatility):
    1. Trend score ≥ 45  (slight uptrend; VCP forms during healthy consolidation)
    2. Price > MA50       (above structural support)
    3. 10D ATR % < 5.5%  (price tightening — was 3.5%, too strict)
    4. Volume < 75% of 60D avg (supply drying up — was 50%, too strict)
    5. Not already within 1% of 52W high (shouldn't be at breakout yet)
    """
    if market_df is None or market_df.empty:
        return []

    results = []
    for _, row in market_df.iterrows():
        ticker = row["ticker"]
        score  = row.get("trend_score", 0)
        price  = row.get("price", 0) or row.get("currentPrice", 0)

        if score < 45 or price < 10:
            continue
        if ticker not in full_history_dict:
            continue

        hist = full_history_dict[ticker]
        if len(hist) < 65:
            continue

        ma50 = hist["Close"].rolling(50).mean().iloc[-1]
        if pd.isna(ma50) or price < ma50 * 0.98:
            continue

        # Volume dry-up: today's vol vs 60D avg
        vol_60d = hist["Volume"].rolling(60).mean().iloc[-1]
        if not vol_60d or vol_60d == 0:
            continue
        vol_ratio = hist["Volume"].iloc[-1] / vol_60d
        if vol_ratio > 0.75:       # loosened from 0.50
            continue

        # Price compression
        compression = calculate_vcp_compression(hist, 10)
        if compression > 5.5:      # loosened from 3.5
            continue

        # Not already at breakout
        dist_hi = row.get("dist_52w", -100)
        if dist_hi is not None and dist_hi > -1.0:
            continue               # already breaking out, not a VCP setup

        # MA50 distance (useful context)
        ma50_dist = ((price - ma50) / ma50) * 100

        results.append({
            "Ticker":      ticker,
            "Name":        row.get("name", ticker),
            "Sector":      row.get("sector", "Unknown"),
            "Price":       round(price, 1),
            "Score":       score,
            "Compression": round(compression, 2),
            "Vol_Ratio":   round(vol_ratio * 100, 1),
            "MA50_Dist":   round(ma50_dist, 1),
            "Dist_52W":    round(dist_hi, 1) if dist_hi is not None else None,
        })

    return sorted(results, key=lambda x: x["Compression"])


# ---------------------------------------------------------------------------
# RS Divergence
# ---------------------------------------------------------------------------

def find_rs_divergence(market_df, nifty_df):
    """
    Green in a sea of Red.
    Fires when Nifty falls > 0.3% and individual stocks close > +0.3%
    (loosened from 0.5%/0.5% to capture more signals on mild red days).
    """
    if nifty_df is None or len(nifty_df) < 2 or market_df is None or market_df.empty:
        return []

    nifty_ret = (float(nifty_df["Close"].iloc[-1]) - float(nifty_df["Close"].iloc[-2])) \
                / float(nifty_df["Close"].iloc[-2]) * 100

    if nifty_ret > -0.3:           # loosened from -0.5
        return []

    results = []
    for _, row in market_df.iterrows():
        day_chg  = row.get("change_p", 0) or 0
        dist_hi  = row.get("dist_52w", -100) or -100
        if day_chg < 0.3 or dist_hi < -20:   # loosened from 0.5/−15
            continue
        delta_rs = day_chg - nifty_ret
        results.append({
            "Ticker":     row["ticker"],
            "Name":       row.get("name", row["ticker"]),
            "Sector":     row.get("sector", "Unknown"),
            "Price":      row.get("price", 0) or row.get("currentPrice", 0),
            "Stock_Ret":  round(day_chg, 2),
            "Nifty_Ret":  round(nifty_ret, 2),
            "Delta_RS":   round(delta_rs, 2),
            "Dist_52W":   round(dist_hi, 1),
            "Trend_Score": row.get("trend_score", 0),
        })

    return sorted(results, key=lambda x: x["Delta_RS"], reverse=True)


# ---------------------------------------------------------------------------
# Earnings Shocks
# ---------------------------------------------------------------------------

def find_live_earnings_shocks(market_df, full_history_dict):
    """
    Day-0 gap on volume.
    Criteria: price jump > 4% (was 5%) on volume > 2.5x avg (was 3x).
    Loosened slightly to catch more actionable shocks.
    """
    if market_df is None or market_df.empty:
        return []

    from utils.live_desk import get_pead_edge

    results = []
    for _, row in market_df.iterrows():
        ticker  = row["ticker"]
        price   = row.get("price", 0) or row.get("currentPrice", 0)
        day_chg = row.get("change_p", 0) or 0

        if day_chg < 4.0:          # loosened from 5.0
            continue
        if ticker not in full_history_dict:
            continue

        hist = full_history_dict[ticker]
        if len(hist) < 25:
            continue

        vol_today = hist["Volume"].iloc[-1]
        vol_20d   = hist["Volume"].rolling(20).mean().shift(1).iloc[-1]
        if not vol_20d or vol_20d == 0:
            continue

        vol_ratio = vol_today / vol_20d
        if vol_ratio < 2.5:        # loosened from 3.0
            continue

        sector  = row.get("sector", "Unknown")
        profile = get_pead_edge(sector)

        results.append({
            "Ticker":      ticker,
            "Name":        row.get("name", ticker),
            "Sector":      sector,
            "Price":       round(price, 1),
            "Jump_Pct":    round(day_chg, 2),
            "Vol_Mult":    round(vol_ratio, 1),
            "PEAD_Action": profile,
        })

    return sorted(results, key=lambda x: x["Jump_Pct"], reverse=True)


# ---------------------------------------------------------------------------
# Signal log helpers
# ---------------------------------------------------------------------------

def _append_to_log(log_file, new_rows_df, key_cols, cols):
    """
    Append new signal rows to a log CSV.
    Deduplicates on key_cols (e.g. signal_date + ticker) so re-runs are safe.
    """
    os.makedirs("data", exist_ok=True)
    if os.path.exists(log_file):
        existing = pd.read_csv(log_file)
    else:
        existing = pd.DataFrame(columns=cols)

    if new_rows_df.empty:
        return existing

    # Ensure all required columns
    for c in cols:
        if c not in new_rows_df.columns:
            new_rows_df[c] = None
    new_rows_df = new_rows_df[cols]

    # Drop duplicates that are already logged
    if not existing.empty and key_cols:
        existing_keys = set(zip(*[existing[k].astype(str) for k in key_cols]))
        mask = ~new_rows_df.apply(
            lambda r: tuple(str(r[k]) for k in key_cols) in existing_keys, axis=1
        )
        new_rows_df = new_rows_df[mask]

    combined = pd.concat([existing, new_rows_df], ignore_index=True)
    # Keep rolling 365 days
    if "signal_date" in combined.columns:
        combined["signal_date"] = pd.to_datetime(combined["signal_date"])
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=365)
        combined = combined[combined["signal_date"] >= cutoff]
        combined["signal_date"] = combined["signal_date"].dt.strftime("%Y-%m-%d")
    combined.to_csv(log_file, index=False)
    return combined


def save_rs_divergence_signals(rs_list):
    """Append today's RS divergence signals to the log."""
    if not rs_list:
        return
    today = datetime.now().strftime("%Y-%m-%d")
    rows = []
    for r in rs_list:
        rows.append({
            "signal_date":       today,
            "ticker":            r["Ticker"],
            "name":              r["Name"],
            "sector":            r["Sector"],
            "signal_price":      r["Price"],
            "current_price":     r["Price"],
            "stock_ret_on_day":  r["Stock_Ret"],
            "nifty_ret_on_day":  r["Nifty_Ret"],
            "delta_rs":          r["Delta_RS"],
            "dist_52w":          r["Dist_52W"],
            "return_since_signal": 0.0,
            "days_held":         0,
            "status":            "ACTIVE",
        })
    _append_to_log(RS_LOG_FILE, pd.DataFrame(rows), ["signal_date", "ticker"], RS_LOG_COLS)


def save_earnings_shock_signals(shock_list):
    """Append today's earnings shock signals to the log."""
    if not shock_list:
        return
    today = datetime.now().strftime("%Y-%m-%d")
    rows = []
    for r in shock_list:
        rows.append({
            "signal_date":         today,
            "ticker":              r["Ticker"],
            "name":                r["Name"],
            "sector":              r["Sector"],
            "signal_price":        r["Price"],
            "current_price":       r["Price"],
            "jump_pct":            r["Jump_Pct"],
            "vol_mult":            r["Vol_Mult"],
            "pead_action":         r["PEAD_Action"],
            "return_since_signal": 0.0,
            "return_5d":           None,
            "return_21d":          None,
            "days_held":           0,
            "status":              "ACTIVE",
        })
    _append_to_log(SHOCK_LOG_FILE, pd.DataFrame(rows), ["signal_date", "ticker"], SHOCK_LOG_COLS)


def refresh_signal_log_prices(log_file, cols, price_map):
    """
    Update current_price, return_since_signal, days_held, status in a log CSV.
    price_map: dict {ticker -> live_price}
    """
    if not os.path.exists(log_file):
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(log_file)
    if df.empty:
        return df

    df["signal_date"] = pd.to_datetime(df["signal_date"])
    today = pd.Timestamp.now().normalize()
    df["days_held"] = (today - df["signal_date"]).dt.days

    sig_price  = pd.to_numeric(df["signal_price"], errors="coerce")
    live_price = df["ticker"].map(price_map)
    live_price = pd.to_numeric(live_price, errors="coerce")

    has_live = live_price.notna() & (live_price > 0) & sig_price.notna() & (sig_price > 0)
    if has_live.any():
        df.loc[has_live, "current_price"] = live_price[has_live].round(2)
        ret = (live_price[has_live] / sig_price[has_live] - 1) * 100
        df.loc[has_live, "return_since_signal"] = ret.round(2)

    # Auto-close after 21 days
    df.loc[df["days_held"] > 21, "status"] = "CLOSED"

    df["signal_date"] = df["signal_date"].dt.strftime("%Y-%m-%d")
    df.to_csv(log_file, index=False)
    return df


def load_signal_log(log_file, cols):
    if not os.path.exists(log_file):
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(log_file)
    if "signal_date" in df.columns:
        df["signal_date"] = pd.to_datetime(df["signal_date"])
        df = df.sort_values("signal_date", ascending=False)
    return df


def save_us_rs_divergence_signals(rs_list):
    """Append today's US RS divergence signals (vs SPY) to the US log."""
    if not rs_list:
        return
    today = datetime.now().strftime("%Y-%m-%d")
    rows = []
    for r in rs_list:
        rows.append({
            "signal_date":       today,
            "ticker":            r["Ticker"],
            "name":              r["Name"],
            "sector":            r["Sector"],
            "signal_price":      r["Price"],
            "current_price":     r["Price"],
            "stock_ret_on_day":  r["Stock_Ret"],
            "nifty_ret_on_day":  r["Nifty_Ret"],
            "delta_rs":          r["Delta_RS"],
            "dist_52w":          r["Dist_52W"],
            "return_since_signal": 0.0,
            "days_held":         0,
            "status":            "ACTIVE",
        })
    _append_to_log(US_RS_LOG_FILE, pd.DataFrame(rows), ["signal_date", "ticker"], RS_LOG_COLS)


def save_us_earnings_shock_signals(shock_list):
    """Append today's US earnings shock signals to the US log."""
    if not shock_list:
        return
    today = datetime.now().strftime("%Y-%m-%d")
    rows = []
    for r in shock_list:
        rows.append({
            "signal_date":         today,
            "ticker":              r["Ticker"],
            "name":                r["Name"],
            "sector":              r["Sector"],
            "signal_price":        r["Price"],
            "current_price":       r["Price"],
            "jump_pct":            r["Jump_Pct"],
            "vol_mult":            r["Vol_Mult"],
            "pead_action":         r["PEAD_Action"],
            "return_since_signal": 0.0,
            "return_5d":           None,
            "return_21d":          None,
            "days_held":           0,
            "status":              "ACTIVE",
        })
    _append_to_log(US_SHOCK_LOG_FILE, pd.DataFrame(rows), ["signal_date", "ticker"], SHOCK_LOG_COLS)


# ---------------------------------------------------------------------------
# VCP Backtest (US stocks, walk-forward)
# ---------------------------------------------------------------------------

def backtest_vcp_us(tickers, n_months=6, top_n=15):
    """
    Walk-forward VCP backtest over the last n_months on a list of US tickers.

    Method:
    - Download ~(n_months + 3) months of daily OHLCV for each ticker.
    - Scan weekly scan points (every 5 trading days) over the backtest window.
    - At each scan point apply all 5 VCP gates using only data up to that date.
    - Record 5D, 10D, 21D forward returns from the signal date.
    - Return all signals + summary stats (hit rate @ 10D, avg/best/worst 10D).

    tickers : list[str]   — e.g. top 150 by trend score
    n_months: int         — lookback window in calendar months (default 6)
    top_n   : int         — rows returned in the "top signals" table
    """
    import yfinance as yf

    if not tickers:
        return pd.DataFrame(), {}

    # Download enough history: backtest window + 3 months of warm-up for indicators
    total_months = n_months + 3
    raw = yf.download(
        tickers,
        period=f"{total_months}mo",
        group_by="ticker",
        threads=False,
        progress=False,
        auto_adjust=True,
    )
    if raw.empty:
        return pd.DataFrame(), {}

    # Build per-ticker dict
    hist_dict = {}
    for t in tickers:
        try:
            if t in raw.columns.get_level_values(0):
                sub = raw[t].dropna(how="all")
                if len(sub) >= 65:
                    hist_dict[t] = sub
        except Exception:
            pass

    if not hist_dict:
        return pd.DataFrame(), {}

    # Backtest window: last n_months calendar days
    cutoff_start = datetime.now() - timedelta(days=n_months * 30)
    signals = []

    for ticker, full_hist in hist_dict.items():
        dates = full_hist.index

        # Weekly scan steps within the backtest window (every 5 trading days)
        scan_dates = [d for d in dates if d >= pd.Timestamp(cutoff_start)]
        scan_dates = scan_dates[::5]  # every 5 trading days ≈ weekly

        for scan_date in scan_dates:
            # Slice history up to scan_date (no look-ahead)
            hist = full_hist.loc[:scan_date]
            if len(hist) < 65:
                continue

            price = float(hist["Close"].iloc[-1])
            if price < 1:
                continue

            # Gate 1: price > MA50
            ma50 = hist["Close"].rolling(50).mean().iloc[-1]
            if pd.isna(ma50) or price < ma50 * 0.98:
                continue

            # Gate 2: volume dry-up < 75% of 60D avg
            vol_60d = hist["Volume"].rolling(60).mean().iloc[-1]
            if not vol_60d or vol_60d == 0:
                continue
            vol_ratio = hist["Volume"].iloc[-1] / vol_60d
            if vol_ratio > 0.75:
                continue

            # Gate 3: 10D ATR% < 5.5%
            recent10 = hist.iloc[-10:]
            if len(recent10) < 10:
                continue
            atr_pct = ((recent10["High"].values - recent10["Low"].values) / recent10["Low"].values).mean() * 100
            if atr_pct > 5.5:
                continue

            # Gate 4: not already within 1% of 52W high
            hi_52w = hist["Close"].rolling(252).max().iloc[-1]
            if pd.isna(hi_52w) or hi_52w <= 0:
                continue
            dist_hi = (price / hi_52w - 1) * 100
            if dist_hi > -1.0:
                continue

            # Gate 5: mild uptrend — price above MA50 already checked; also check MA50 > MA100
            if len(hist) >= 100:
                ma100 = hist["Close"].rolling(100).mean().iloc[-1]
                if not pd.isna(ma100) and ma50 < ma100 * 0.98:
                    continue

            # Forward returns (using actual future prices)
            future = full_hist.loc[scan_date:]
            ret_5d  = ret_10d = ret_21d = None
            if len(future) > 5:
                ret_5d  = (float(future["Close"].iloc[5])  / price - 1) * 100
            if len(future) > 10:
                ret_10d = (float(future["Close"].iloc[10]) / price - 1) * 100
            if len(future) > 21:
                ret_21d = (float(future["Close"].iloc[21]) / price - 1) * 100

            signals.append({
                "Signal Date":  scan_date.strftime("%Y-%m-%d"),
                "Ticker":       ticker,
                "Signal Price": round(price, 2),
                "Compression":  round(atr_pct, 2),
                "Vol Ratio %":  round(vol_ratio * 100, 1),
                "Dist 52W %":   round(dist_hi, 1),
                "Return 5D %":  round(ret_5d,  2) if ret_5d  is not None else None,
                "Return 10D %": round(ret_10d, 2) if ret_10d is not None else None,
                "Return 21D %": round(ret_21d, 2) if ret_21d is not None else None,
            })

    if not signals:
        return pd.DataFrame(), {}

    df_signals = pd.DataFrame(signals)

    # Summary stats on signals that have 10D forward data
    df_with_10d = df_signals.dropna(subset=["Return 10D %"])
    n_total   = len(df_with_10d)
    n_winners = (df_with_10d["Return 10D %"] > 0).sum()
    hit_rate  = (n_winners / n_total * 100) if n_total else 0
    avg_ret   = df_with_10d["Return 10D %"].mean() if n_total else 0
    best_ret  = df_with_10d["Return 10D %"].max()  if n_total else 0
    worst_ret = df_with_10d["Return 10D %"].min()  if n_total else 0

    summary = {
        "total_signals": n_total,
        "hit_rate_pct":  round(hit_rate, 1),
        "avg_10d_ret":   round(avg_ret, 2),
        "best_10d_ret":  round(best_ret, 2),
        "worst_10d_ret": round(worst_ret, 2),
    }

    # Top N by 10D return
    top_df = (
        df_signals
        .dropna(subset=["Return 10D %"])
        .sort_values("Return 10D %", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return top_df, summary
