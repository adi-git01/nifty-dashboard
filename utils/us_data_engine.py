"""
US Market Data Engine — S&P 500 scanner mirroring fast_data_engine.py
Benchmark: SPY (S&P 500 ETF)
RS weights: Config-D  RS5=30%  RS21=50%  RS63=20%
"""

import os
import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from utils.scoring import calculate_trend_metrics, calculate_scores
from utils.volume_analysis import get_combined_volume_signal

SP500_CSV         = "data/sp500_tickers.csv"
US_PARQUET_PREFIX = "data/cache/us_market_master"

DEFAULT_RS_WEIGHTS = [(5, 0.30), (21, 0.50), (63, 0.20)]


# ---------------------------------------------------------------------------
# Ticker universe helpers
# ---------------------------------------------------------------------------

def load_sp500_universe() -> pd.DataFrame:
    """Returns DataFrame[ticker, name, sector, sub_industry]."""
    if not os.path.exists(SP500_CSV):
        raise FileNotFoundError(f"{SP500_CSV} not found")
    df = pd.read_csv(SP500_CSV)
    required = {"ticker", "name", "sector", "sub_industry"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sp500_tickers.csv missing columns: {missing}")
    return df.drop_duplicates("ticker").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _us_parquet_path() -> str:
    os.makedirs("data/cache", exist_ok=True)
    today = datetime.now().strftime("%Y_%m_%d")
    return f"{US_PARQUET_PREFIX}_{today}.parquet"


def clear_us_cache() -> int:
    """Delete all us_market_master_*.parquet files. Returns count deleted."""
    import glob
    files = glob.glob(f"{US_PARQUET_PREFIX}_*.parquet")
    for f in files:
        try:
            os.remove(f)
        except Exception:
            pass
    return len(files)


def _load_cached(live_mode: bool):
    if live_mode:
        return None
    path = _us_parquet_path()
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        required_cols = {"ticker", "currentPrice", "trend_signal", "comp_rs"}
        if not required_cols.issubset(df.columns):
            return None
        # Invalidate if CSV has grown significantly since parquet was written
        try:
            universe = load_sp500_universe()
            csv_count = len(universe)
            parquet_count = len(df)
            if parquet_count < csv_count * 0.85:
                print(
                    f"[US ENGINE] Parquet has {parquet_count} rows but CSV has "
                    f"{csv_count} tickers — invalidating stale cache",
                    flush=True,
                )
                os.remove(path)
                return None
        except Exception:
            pass
        print(f"[US ENGINE] Loaded parquet cache: {path}", flush=True)
        return df
    except Exception as e:
        print(f"[US ENGINE] Parquet load failed: {e}", flush=True)
    return None


def _save_cache(df: pd.DataFrame) -> None:
    # Write to a temp file and rename (atomic on POSIX) so a mid-write crash
    # can't leave a truncated parquet at the real path for readers to trip over.
    path = _us_parquet_path()
    tmp_path = f"{path}.tmp{os.getpid()}"
    try:
        df.to_parquet(tmp_path)
        os.replace(tmp_path, path)
        print(f"[US ENGINE] Cache saved to {path}", flush=True)
    except Exception as e:
        print(f"[US ENGINE] Cache save failed: {e}", flush=True)
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# RS helpers
# ---------------------------------------------------------------------------

def _period_return(series: pd.Series, period: int) -> float:
    """Exact N-interval return: use iloc[-(period+1)] as base."""
    if len(series) < period + 2:
        return 0.0
    base = float(series.iloc[-(period + 1)])
    curr = float(series.iloc[-1])
    if base == 0:
        return 0.0
    return (curr / base - 1.0) * 100.0


def _comp_rs(stock_series: pd.Series, bench_series: pd.Series,
             weights):
    """
    Returns (comp_rs, rs5, rs21, rs63).
    weights: list of (period, weight) tuples.
    """
    rs_vals = {}
    for period, _ in weights:
        s_ret = _period_return(stock_series, period)
        b_ret = _period_return(bench_series, period)
        rs_vals[period] = s_ret - b_ret

    comp = sum(rs_vals[p] * w for p, w in weights)
    return (
        round(comp, 2),
        round(rs_vals.get(5, 0.0), 2),
        round(rs_vals.get(21, 0.0), 2),
        round(rs_vals.get(63, 0.0), 2),
    )


# ---------------------------------------------------------------------------
# Main fetch function
# ---------------------------------------------------------------------------

def fetch_us_market_data(
    benchmark: str = "SPY",
    rs_weights=None,  # list of (period, weight) tuples or None
    lookback_days: int = 400,
    live_mode: bool = False,
) -> pd.DataFrame:
    """
    Downloads price history for all S&P 500 tickers + benchmark, computes
    RS, trend metrics, scores and returns a merged DataFrame.

    Args:
        benchmark:    Ticker used as benchmark for RS (default "SPY").
        rs_weights:   List of (period, weight) e.g. [(5,.3),(21,.5),(63,.2)].
        lookback_days: Calendar days of history to request (default 400).
        live_mode:    If True, skip parquet cache.
    """
    if rs_weights is None:
        rs_weights = DEFAULT_RS_WEIGHTS

    cached = _load_cached(live_mode)
    if cached is not None:
        return cached

    universe = load_sp500_universe()
    tickers = universe["ticker"].tolist()
    fund_lookup = universe.set_index("ticker").to_dict("index")

    period_str = f"{lookback_days}d"
    all_tickers = [benchmark] + tickers

    print(f"[US ENGINE] Downloading {len(all_tickers)} tickers (benchmark={benchmark})…", flush=True)
    try:
        raw = yf.download(
            all_tickers,
            period=period_str,
            interval="1d",
            group_by="ticker",
            threads=False,
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        print(f"[US ENGINE] yfinance download failed: {e}", flush=True)
        return pd.DataFrame()

    print(f"[US ENGINE] Download complete. Shape={raw.shape}", flush=True)

    # --- Extract benchmark series ---
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            bench_close = raw[benchmark]["Close"].dropna()
        else:
            bench_close = raw["Close"].dropna()
    except Exception as e:
        print(f"[US ENGINE] Benchmark extraction failed: {e}", flush=True)
        return pd.DataFrame()

    # --- Process each ticker ---
    rows = []
    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(0):
                    continue
                df = raw[ticker].dropna(how="all")
            else:
                df = raw.dropna(how="all")

            close = df["Close"].dropna()
            if len(close) < 22:
                continue

            base = dict(fund_lookup.get(ticker, {}))
            base["ticker"] = ticker
            base.setdefault("name", ticker)
            base.setdefault("sector", "Unknown")
            base.setdefault("sub_industry", "Unknown")

            # --- Price metrics ---
            current_price = float(close.iloc[-1])
            prev_close = float(close.iloc[-2]) if len(close) > 1 else current_price
            base["currentPrice"] = current_price
            base["price"] = current_price
            base["previousClose"] = prev_close
            base["change_p"] = ((current_price - prev_close) / prev_close) * 100 if prev_close else 0.0

            # --- Multi-period returns ---
            for label, period in [("return_1w", 5), ("return_1m", 21), ("return_3m", 63),
                                   ("return_6m", 126), ("return_1y", 252)]:
                base[label] = _period_return(close, period)

            # --- Moving averages ---
            if len(close) >= 50:
                base["fiftyDayAverage"] = float(close.rolling(50).mean().iloc[-1])
            if len(close) >= 200:
                base["twoHundredDayAverage"] = float(close.rolling(200).mean().iloc[-1])

            # --- 52-week high/low ---
            window = close.iloc[-252:] if len(close) >= 252 else close
            high_52 = float(df["High"].iloc[-252:].max() if len(df) >= 252 else df["High"].max())
            low_52  = float(df["Low"].iloc[-252:].min()  if len(df) >= 252 else df["Low"].min())
            base["fiftyTwoWeekHigh"] = high_52
            base["fiftyTwoWeekLow"]  = low_52
            base["dist_52w"] = ((current_price - high_52) / high_52) * 100 if high_52 else 0.0

            # --- Volume ---
            if "Volume" in df.columns:
                vol_series = df["Volume"].dropna()
                if len(vol_series) >= 20:
                    base["averageVolume"] = int(vol_series.iloc[-50:].mean())
                    try:
                        vol_sig = get_combined_volume_signal(
                            df["High"], df["Low"], df["Close"], vol_series
                        )
                        base["volume_signal_score"] = vol_sig["combined_score"]
                        base["volume_signal_text"]  = vol_sig["combined_signal"]
                        base["volume_score"]         = vol_sig["combined_score"]
                    except Exception:
                        base["volume_signal_score"] = 5.0
                        base["volume_signal_text"]  = "N/A"
                        base["volume_score"]         = 5.0

                    if len(vol_series) >= 50:
                        price_chg = close.diff()
                        up_vol   = vol_series.where(price_chg > 0, 0).rolling(50).sum()
                        dn_vol   = vol_series.where(price_chg < 0, 0).rolling(50).sum()
                        ud = float(up_vol.iloc[-1]) / float(dn_vol.iloc[-1]) if float(dn_vol.iloc[-1]) > 0 else 1.0
                        base["ud_ratio"] = round(ud, 2)
                    else:
                        base["ud_ratio"] = 1.0

                    if base.get("ud_ratio", 1.0) > 1.2:
                        base["vol_badge"] = "🟢 ⬆️ U/D > 1.2"
                    elif base.get("ud_ratio", 1.0) < 0.8:
                        base["vol_badge"] = "🔴 ⬇️ U/D < 0.8"
                    else:
                        base["vol_badge"] = "🟡 ➖ Neutral"

            # --- Trend metrics ---
            trends = calculate_trend_metrics(base)
            base.update(trends)

            # --- Scores (4-pillar; PE/fundamentals will be missing → neutral 5) ---
            sector = base.get("sector", "Unknown")
            scores = calculate_scores(base, sector_pe_median=None, sector=sector)
            base.update(scores)

            # --- OptComp RS vs benchmark ---
            comp, rs5, rs21, rs63 = _comp_rs(close, bench_close, rs_weights)
            base["comp_rs"] = comp
            base["rs_5d"]   = rs5
            base["rs_21d"]  = rs21
            base["rs_63d"]  = rs63

            # --- Volatility (60-day annualized) ---
            if len(close) > 60:
                daily_rets = close.pct_change().dropna().iloc[-60:]
                base["volatility"] = round(float(daily_rets.std()) * np.sqrt(252) * 100, 1)
            else:
                base["volatility"] = 0.0

            # --- DNA signal (US-adapted: CompRS > 0, price > 50MA, liquidity > $10M/day) ---
            above_ma50 = current_price > base.get("fiftyDayAverage", 0)
            daily_turnover = base.get("averageVolume", 0) * current_price
            if comp > 0 and above_ma50 and daily_turnover > 10_000_000:
                base["dna_signal"] = "BUY"
            elif comp > 0 and above_ma50:
                base["dna_signal"] = "WATCH"
            else:
                base["dna_signal"] = "HOLD"

            rows.append(base)

        except Exception as e:
            print(f"[US ENGINE] Error processing {ticker}: {e}", flush=True)
            continue

    print(f"[US ENGINE] Processed {len(rows)}/{len(tickers)} tickers.", flush=True)
    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    _save_cache(result)
    return result
