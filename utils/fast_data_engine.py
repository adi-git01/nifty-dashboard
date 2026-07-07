import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
import streamlit as st
from datetime import datetime
from utils.volume_analysis import get_combined_volume_signal
from utils.scoring import calculate_trend_metrics, calculate_scores
from utils.data_engine import get_stock_info
import concurrent.futures

CACHE_FILE_CSV       = "data/nifty500_cache.csv"
FUNDAMENTALS_CACHE   = "data/fundamentals_cache.csv"   # written by fetch_fundamentals.py (bi-weekly)

# Fundamental columns that the bi-weekly cache owns.
# These are merged over the parquet which only has price-derived data.
_FUND_COLS = [
    'pe', 'forwardPE', 'pegRatio', 'pb',
    'roe', 'roa', 'profitMargins', 'grossMargins', 'operatingMargins', 'ebitdaMargins',
    'revenueGrowth', 'earningsGrowth', 'earningsQuarterlyGrowth',
    'debtToEquity', 'marketCap', 'beta',
    'earningsQuality', 'earningsTrend',
    'name', 'sector', 'sector_granular', 'fund_last_updated',
]

# Columns that legitimately hold text; everything else exchanged between the
# caches and yfinance is numeric and must come back out of _safe_update numeric.
_TEXT_COLS = {'ticker', 'name', 'sector', 'sector_granular', 'industry',
              'summary', 'fund_last_updated', 'trend_signal', 'dna_signal'}

def _safe_update(df: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame.update() in modern pandas raises TypeError instead of silently
    upcasting a column's dtype (e.g. a float64 NaN column receiving a string
    from yfinance). Cast the overlapping columns to object first so the
    in-place update never needs to upcast, then coerce the numeric columns
    back — leaving them as object would crash every downstream median()/
    nlargest()/comparison under pandas' strict object-dtype rules, and
    pd.to_numeric conveniently drops junk like Yahoo's literal 'Infinity'
    sentinel (parsed to inf, stripped to NaN here) in the same pass.
    """
    common = df.columns.intersection(other.columns)
    if len(common):
        df[common] = df[common].astype(object)
    df.update(other)
    for col in common:
        if col in _TEXT_COLS:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return df

def get_parquet_cache_path():
    today_str = datetime.now().strftime("%Y_%m_%d")
    os.makedirs("data/cache", exist_ok=True)
    return f"data/cache/market_master_{today_str}.parquet"

def _merge_fundamentals_cache(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges data/fundamentals_cache.csv (written bi-weekly) into df.
    The parquet only has price-derived columns; the CSV provides PE, ROE, margins, etc.
    After merging, fetch_missing_fundamentals() finds 'pe' already populated and
    skips the 75-second per-restart yf.Ticker().info fetch loop.
    """
    if not os.path.exists(FUNDAMENTALS_CACHE):
        return _overlay_sub_industry(df)
    try:
        fund = pd.read_csv(FUNDAMENTALS_CACHE)
        if 'ticker' not in fund.columns:
            return _overlay_sub_industry(df)

        # Only bring in columns this cache owns; don't clobber live price data
        cols_to_merge = [c for c in _FUND_COLS if c in fund.columns]
        fund = fund[['ticker'] + cols_to_merge].set_index('ticker')

        df = df.set_index('ticker')
        df = _safe_update(df, fund)          # fills NaN + updates stale values
        for col in fund.columns:             # add any columns missing from parquet
            if col not in df.columns:
                df[col] = fund[col]
        df = df.reset_index()

        age_days = ""
        if 'fund_last_updated' in fund.columns:
            latest = fund['fund_last_updated'].dropna().max()
            try:
                from datetime import date
                delta = (date.today() - pd.to_datetime(latest).date()).days
                age_days = f", {delta}d old"
            except Exception:
                pass
        print(f"[ENGINE] Merged bi-weekly fundamentals cache "
              f"({len(fund)} stocks{age_days})", flush=True)
    except Exception as e:
        print(f"[ENGINE] fundamentals_cache merge failed: {e}", flush=True)
    return _overlay_sub_industry(df)


def _overlay_sub_industry(df: pd.DataFrame) -> pd.DataFrame:
    """
    The in-repo Playbook-58 map covers the full universe; the fundamentals
    cache only covers tickers it was built from, which left hundreds of
    stocks with sector_granular = Unknown in the UI and rotation matrix.
    """
    try:
        from utils.nifty1000_list import SUB_INDUSTRY_MAP
        mapped = df['ticker'].map(SUB_INDUSTRY_MAP)
        if 'sector_granular' in df.columns:
            df['sector_granular'] = mapped.fillna(df['sector_granular'])
        else:
            df['sector_granular'] = mapped
    except Exception as e:
        print(f"[ENGINE] sector_granular overlay failed: {e}", flush=True)
    return df


def load_base_fundamentals(live_mode=False):
    """
    Loads base data. Priority: Parquet > CSV > ticker list.
    Always overlays the bi-weekly fundamentals_cache.csv so the parquet
    (which has no PE/ROE from CI live_mode runs) gets fundamental columns
    without a live yfinance fetch.
    """
    parquet_path = get_parquet_cache_path()

    # Try Parquet unless in Live Mode
    if not live_mode and os.path.exists(parquet_path):
        try:
            df = pd.read_parquet(parquet_path)
            if 'ticker' in df.columns:
                return _merge_fundamentals_cache(df)
            print("Parquet cache missing 'ticker' column. Ignoring.")
        except Exception as e:
            print(f"Error loading parquet: {e}")

    # Try CSV (Legacy)
    if os.path.exists(CACHE_FILE_CSV):
        try:
            df = pd.read_csv(CACHE_FILE_CSV)
            required = ['ticker', 'name', 'sector']
            if all(col in df.columns for col in required):
                return _merge_fundamentals_cache(df)
        except Exception as e:
            print(f"Error loading CSV: {e}")

    # Fallback: bare ticker list
    from utils.nifty1000_list import TICKERS_1000, SUB_INDUSTRY_MAP
    df = pd.DataFrame({'ticker': TICKERS_1000})
    df['sector'] = df['ticker'].map(SUB_INDUSTRY_MAP).fillna("Unknown")
    df['name'] = df['ticker']
    return _merge_fundamentals_cache(df)

def fetch_missing_fundamentals(df):
    """
    Scans for stocks with missing fundamentals (PE, ROE, etc.) and fetches them via threads.
    Ensures ACCURACY over speed.
    """
    if df.empty: return df
    
    # Initialize PE if missing
    if 'pe' not in df.columns: df['pe'] = np.nan
    
    # Identify missing PE or unknown sector
    # We want to be aggressive: fetch if PE is missing or 0.
    mask = (df['pe'].isna()) | (df['pe'] == 0) | (df.get('sector', pd.Series(['Unknown']*len(df))) == "Unknown")
    
    if 'ticker' in df.columns:
        missing = df[mask]['ticker'].unique().tolist()
    else:
        return df

    if not missing: return df
    
    # 🕵️ BOOBY TRAP: log how many stocks need fundamental fetch — hints at run duration
    print(f"[ENGINE] fetch_missing_fundamentals: {len(missing)} stocks missing PE/sector. Fetching via 3 workers...", flush=True)
    try:
        st.toast(f"Deep scanning {len(missing)} stocks... (Accuracy Mode)", icon="🕵️")
    except Exception:
        pass  # headless mode, ignore

    
    new_data = []
    # ✅ FIX: Reduced workers 20→3 to avoid Yahoo Finance rate limiting on GitHub Actions
    # Sequential-ish fetching with small delay prevents 401 Invalid Crumb cascade
    def _fetch_with_delay(ticker):
        time.sleep(0.3)  # Stagger requests to avoid rate limits
        return get_stock_info(ticker)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_fetch_with_delay, t): t for t in missing}
        for f in concurrent.futures.as_completed(futures):
            try:
                res = f.result()
                if res:
                    res['ticker'] = futures[f]
                    new_data.append(res)
            except: pass
                
    if new_data:
        update_df = pd.DataFrame(new_data)
        df = df.set_index('ticker')
        update_df = update_df.set_index('ticker')

        df = _safe_update(df, update_df)
        for col in update_df.columns:
            if col not in df.columns:
                df[col] = update_df[col]

        df = df.reset_index()

        # Recompute pillar scores for freshly-fetched stocks so stale cached
        # scores (computed with all-zero fundamentals by the CI live_mode run)
        # are replaced immediately rather than persisting until next full run.
        try:
            if 'pe' in df.columns:
                # errors='coerce' still parses 'Infinity'/'-Infinity' strings into
                # real inf floats (Python float() semantics) instead of NaN — strip those too.
                pe_numeric = pd.to_numeric(df['pe'], errors='coerce').replace([np.inf, -np.inf], np.nan)
                sector_pe = df.assign(pe=pe_numeric).groupby('sector')['pe'].median().to_dict()
            else:
                sector_pe = {}
            updated_tickers = set(update_df.index)
            df_idx = df.set_index('ticker')
            for t in updated_tickers:
                if t not in df_idx.index:
                    continue
                row = df_idx.loc[t].to_dict()
                row['ticker'] = t
                sector = row.get('sector', 'Unknown')
                scores = calculate_scores(row, sector_pe_median=sector_pe.get(sector, 20), sector=sector)
                for k, v in scores.items():
                    df_idx.loc[t, k] = v
            df = df_idx.reset_index()
        except Exception as e:
            print(f"[ENGINE] Score recomputation after fundamental fetch failed: {e}")

    return df

def fetch_and_process_market_data(tickers, fundamental_df, live_mode=False):
    """
    Vectorized fetch of price data for all tickers + merging with fundamentals.
    """
    # 0. Ensure Fundamentals (Accuracy Mode) - checking/fetching missing PE/Sector
    # ✅ OPTIMIZATION: In live_mode (CI/GitHub Actions), skip the per-ticker .info fetch.
    # PE ratio is NOT needed for: dna_signal, comp_rs, sub-industry rotation, or stop checks.
    # Fetching PE for 750 stocks via yf.Ticker().info takes 4+ minutes and is the #1 bottleneck.
    # PE is only needed in the Streamlit UI's fundamentals table — it reads from Parquet cache there.
    if not live_mode:
        fundamental_df = fetch_missing_fundamentals(fundamental_df)
    else:
        print("[ENGINE] live_mode=True — skipping fetch_missing_fundamentals (PE not needed for signals)", flush=True)
    
    # If not live mode, and fundamental_df already has computed metrics like 'currentPrice', we just return it!
    # Because loading from Parquet already restores full computed state.
    # CRITICAL FIX: Ensure 'trend_signal' and 'dna_signal' truly exist to prevent GUI KeyErrors.
    if not live_mode and not fundamental_df.empty:
        required_cols = ['currentPrice', 'trend_signal', 'dna_signal']
        if all(col in fundamental_df.columns for col in required_cols):
            return fundamental_df
    
    if not tickers:
        return fundamental_df
    if not tickers:
        return fundamental_df

    print(f"[FAST ENGINE] Fetching data for {len(tickers)} stocks...")
    
    # 1. Bulk Fetch History (1 Year)
    try:
        # Fetch Nifty Data for RS Calculation
        nifty = yf.Ticker("^NSEI").history(period="1y")
        nifty_3m_ret = 0
        if not nifty.empty:
            nifty.index = nifty.index.tz_localize(None)
            curr_nifty = nifty['Close'].iloc[-1]
            if len(nifty) > 63:
                old_nifty = nifty['Close'].iloc[-63]
                nifty_3m_ret = ((curr_nifty - old_nifty) / old_nifty) * 100

        # ✅ FIX: threads=False prevents 20-thread storm that triggers Yahoo 401s on CI
        # Batch download is still fast (single HTTP session, vectorized)
        print(f"[ENGINE] Starting yf.download for {len(tickers)} tickers...", flush=True)
        history_data = yf.download(
            tickers, 
            period="1y", 
            interval="1d", 
            group_by='ticker', 
            threads=False,      # <-- KEY FIX: avoid parallel request storm
            progress=False,
            auto_adjust=True
        )
        # 🕵️ BOOBY TRAP: log download result structure
        print(f"[ENGINE] yf.download complete. Shape={history_data.shape}, "
              f"MultiIndex={isinstance(history_data.columns, pd.MultiIndex)}, "
              f"Cols preview={list(history_data.columns[:4]) if not isinstance(history_data.columns, pd.MultiIndex) else list(history_data.columns.levels[0][:4])}",
              flush=True)
    except Exception as e:
        # ✅ Graceful degradation: log but don't crash — use existing cache/fundamentals
        print(f"[ENGINE] yfinance download failed ({e}), returning cached fundamentals")
        try:
            st.error(f"Failed to fetch market data: {e}")
        except Exception:
            pass
        return fundamental_df

    processed_rows = []
    
    # Pre-calculate sector medians for scoring
    # NB: 'pe' can carry non-numeric junk (e.g. Yahoo's literal 'Infinity'
    # sentinel for near-zero EPS); coerce so groupby().median() can't raise.
    sector_pe = {}
    if 'pe' in fundamental_df.columns and 'sector' in fundamental_df.columns:
        pe_numeric = pd.to_numeric(fundamental_df['pe'], errors='coerce').replace([np.inf, -np.inf], np.nan)
        sector_pe = fundamental_df.assign(pe=pe_numeric).groupby('sector')['pe'].median().to_dict()

    # 2. Process each ticker (CPU bound, but fast)
    total_tickers = len(tickers)
    
    # Create lookup for fundamentals
    fund_lookup = fundamental_df.set_index('ticker').to_dict('index') if not fundamental_df.empty else {}
    
    for ticker in tickers:
        try:
            # Extract Ticker Data
            # Note: yfinance multi-level column structure [Ticker, Feature] or just Feature if 1 ticker
            if total_tickers > 1:
                if ticker not in history_data.columns.levels[0]:
                    continue
                df = history_data[ticker].dropna()
            else:
                df = history_data.dropna()
                
            if df.empty or len(df) < 20:
                continue
                
            # Get existing fundamentals (Sector, PE, Name, etc.)
            base_data = fund_lookup.get(ticker, {})
            # Ensure ticker and name exist
            base_data['ticker'] = ticker
            if 'name' not in base_data:
                base_data['name'] = ticker
            if 'sector' not in base_data:
                base_data['sector'] = "Unknown"
                
            # --- CALCULATE LIVE METRICS ---
            
            # 1. Price metrics
            current_price = float(df['Close'].iloc[-1])
            prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else current_price
            change_p = ((current_price - prev_close) / prev_close) * 100
            
            base_data['currentPrice'] = current_price
            base_data['price'] = current_price # Alias
            base_data['previousClose'] = prev_close
            base_data['change_p'] = change_p
            
            # 1b. Multi-period Returns
            def calc_return(days):
                if len(df) >= days:
                    old_price = float(df['Close'].iloc[-days])
                    return ((current_price - old_price) / old_price) * 100
                return 0.0
            
            base_data['return_1w'] = calc_return(5)
            base_data['return_1m'] = calc_return(21)
            base_data['return_3m'] = calc_return(63)
            base_data['return_6m'] = calc_return(126)
            base_data['return_1y'] = calc_return(252) if len(df) >= 252 else calc_return(len(df) - 1)
            
            # 2. Moving Averages & Highs
            if len(df) >= 50:
                base_data['fiftyDayAverage'] = float(df['Close'].rolling(50).mean().iloc[-1])
            if len(df) >= 200:
                base_data['twoHundredDayAverage'] = float(df['Close'].rolling(200).mean().iloc[-1])
                
            high_52 = df['High'].iloc[-252:].max() if len(df) >= 252 else df['High'].max()
            low_52 = df['Low'].iloc[-252:].min() if len(df) >= 252 else df['Low'].min()
            
            base_data['fiftyTwoWeekHigh'] = float(high_52)
            base_data['fiftyTwoWeekLow'] = float(low_52)
            base_data['dist_52w'] = ((current_price - high_52) / high_52) * 100
            
            # 3. Volume Metrics (VPT + A/D)
            if 'Volume' in df.columns:
                vol_sig = get_combined_volume_signal(
                    df['High'], df['Low'], df['Close'], df['Volume']
                )
                base_data['volume_signal_score'] = vol_sig['combined_score']
                base_data['volume_signal_text'] = vol_sig['combined_signal']
                base_data['volume_score'] = vol_sig['combined_score'] # Alias for sorting
                
                # Avg Volume
                base_data['averageVolume'] = int(df['Volume'].iloc[-50:].mean())
                
                # Up/Down Volume Ratio (50 Days)
                if len(df) >= 50:
                    price_change = df['Close'].diff()
                    up_vol = df['Volume'].where(price_change > 0, 0).rolling(window=50).sum()
                    down_vol = df['Volume'].where(price_change < 0, 0).rolling(window=50).sum()
                    ud_ratio = up_vol / down_vol.replace(0, np.nan)
                    val = float(ud_ratio.iloc[-1])
                    base_data['ud_ratio'] = round(val, 2) if not np.isnan(val) else 1.0
                else:
                    base_data['ud_ratio'] = 1.0
                    
                # Volume Badge Signal for UI
                if base_data.get('ud_ratio', 1.0) > 1.2:
                    base_data['vol_badge'] = "🟢 ⬆️ U/D > 1.2"
                elif base_data.get('ud_ratio', 1.0) < 0.8:
                    base_data['vol_badge'] = "🔴 ⬇️ U/D < 0.8"
                else:
                    base_data['vol_badge'] = "🟡 ➖ Neutral"

            # 4. Trend Metrics (Rely on utils/scoring which handles the logic)
            trends = calculate_trend_metrics(base_data)
            base_data.update(trends)
            
            # 5. Full Scoring (4-Pillar)
            sector = base_data.get('sector', 'Unknown')
            median_pe = sector_pe.get(sector, 20)
            scores = calculate_scores(base_data, sector_pe_median=median_pe, sector=sector)
            base_data.update(scores)
            
            # 6. OptComp-V22 Composite RS 
            # (10% 1W, 50% 1M, 40% 3M vs Nifty)
            def _calc_rs(ret, bench_ret):
                return ret - bench_ret
                
            rs_1w = _calc_rs(base_data.get('return_1w', 0), 0) # Simplification: Assume Nifty 1W ret ~ 0 for fast calc, ideally pass nifty_1w_ret
            # We'll calculate true RS vs Nifty directly here
            idx = df.index.searchsorted(df.index[-1])
            nifty_idx = nifty.index.searchsorted(df.index[-1])
            
            if nifty_idx >= 63 and idx >= 63:
                n_ret_1w = (nifty['Close'].iloc[nifty_idx] - nifty['Close'].iloc[nifty_idx-5]) / nifty['Close'].iloc[nifty_idx-5] * 100
                n_ret_1m = (nifty['Close'].iloc[nifty_idx] - nifty['Close'].iloc[nifty_idx-21]) / nifty['Close'].iloc[nifty_idx-21] * 100
                n_ret_3m = (nifty['Close'].iloc[nifty_idx] - nifty['Close'].iloc[nifty_idx-63]) / nifty['Close'].iloc[nifty_idx-63] * 100
                
                t_ret_1w = (df['Close'].iloc[idx] - df['Close'].iloc[idx-5]) / df['Close'].iloc[idx-5] * 100
                t_ret_1m = (df['Close'].iloc[idx] - df['Close'].iloc[idx-21]) / df['Close'].iloc[idx-21] * 100
                t_ret_3m = (df['Close'].iloc[idx] - df['Close'].iloc[idx-63]) / df['Close'].iloc[idx-63] * 100
                
                rs_1w = t_ret_1w - n_ret_1w
                rs_1m = t_ret_1m - n_ret_1m
                rs_3m_val = t_ret_3m - n_ret_3m
                
                comp_rs = (rs_1w * 0.10) + (rs_1m * 0.50) + (rs_3m_val * 0.40)
            else:
                comp_rs = 0
                rs_1w = 0
                rs_1m = 0
                rs_3m_val = base_data.get('return_3m', 0)

            base_data['comp_rs'] = round(comp_rs, 2)
            base_data['rs_3m']   = round(rs_3m_val, 2)
            base_data['rs_1w']   = round(rs_1w, 2)
            base_data['rs_1m']   = round(rs_1m, 2)
            
            # Volatility (Annualized)
            if len(df) > 60:
                daily_rets = df['Close'].pct_change().dropna()[-60:]
                volatility = daily_rets.std() * np.sqrt(252) * 100
            else:
                volatility = 0
            base_data['volatility'] = round(volatility, 1)
            
            # DNA Signal (Now OptComp-V22 Rules)
            # Entry: Composite RS > 0, Price > 50MA, Vol > 1Cr
            above_ma50 = current_price > base_data.get('fiftyDayAverage', 0)
            vol_val = base_data.get('averageVolume', 0) * current_price
            
            if comp_rs > 0 and above_ma50 and vol_val > 10_000_000:
                base_data['dna_signal'] = 'BUY'
            elif comp_rs > 0 and above_ma50:
                base_data['dna_signal'] = 'WATCH'
            else:
                base_data['dna_signal'] = 'HOLD'
            
            # 7. Recommendation
            overall = scores.get('overall', 0)
            if overall >= 7.5:
                base_data['recommendation'] = "BUY"
            elif overall >= 5.0:
                base_data['recommendation'] = "HOLD"
            else:
                base_data['recommendation'] = "AVOID"

            processed_rows.append(base_data)

        except Exception as e:
            # print(f"Error processing {ticker}: {e}")
            continue
            
    # Convert to DataFrame
    final_df = pd.DataFrame(processed_rows)
    # 🕵️ BOOBY TRAP: log how many tickers made it through
    print(f"[ENGINE] Processing complete: {len(processed_rows)}/{total_tickers} tickers returned valid data.", flush=True)
    if not final_df.empty:
        print(f"[ENGINE] Output shape={final_df.shape}, key_cols={[c for c in ['ticker','currentPrice','dna_signal','comp_rs'] if c in final_df.columns]}", flush=True)
    else:
        print("[ENGINE] ⚠️  final_df is EMPTY — yfinance returned no usable data.", flush=True)
    
    # Save to Parquet Cache for next time. Write to a temp file and rename
    # (atomic on POSIX) so a mid-write crash (OOM, CI timeout, disk full)
    # can never leave a truncated file at the real path for readers to trip
    # over — os.replace() only lands the fully-written file, or nothing.
    try:
        parquet_path = get_parquet_cache_path()
        tmp_path = f"{parquet_path}.tmp{os.getpid()}"
        final_df.to_parquet(tmp_path)
        os.replace(tmp_path, parquet_path)
    except Exception as e:
        print(f"[ENGINE] Parquet cache write failed: {e}", flush=True)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        
    return final_df
