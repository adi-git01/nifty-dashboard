import yfinance as yf
import pandas as pd
import numpy as np
import os
import streamlit as st
from datetime import datetime
from utils.volume_analysis import get_combined_volume_signal
from utils.scoring import calculate_trend_metrics, calculate_scores
from utils.data_engine import get_stock_info
import concurrent.futures

CACHE_FILE_CSV = "nifty500_cache.csv"
CACHE_FILE_PARQUET = "market_data.parquet"

def load_base_fundamentals():
    """
    Loads base fundamental data (Sector, Name, PE, MarketCap) from cache.
    Prefers Parquet (Fast) > CSV (Slow) > Empty DataFrame.
    """
    # Try Parquet
    if os.path.exists(CACHE_FILE_PARQUET):
        try:
            df = pd.read_parquet(CACHE_FILE_PARQUET)
            if 'ticker' in df.columns:
                return df
            print("⚠️ Parquet cache missing 'ticker' column. Ignoring.")
        except Exception as e:
            print(f"Error loading parquet: {e}")
            
    # Try CSV (Legacy)
    if os.path.exists(CACHE_FILE_CSV):
        try:
            df = pd.read_csv(CACHE_FILE_CSV)
            # Ensure critical columns exist
            required = ['ticker', 'name', 'sector']
            if all(col in df.columns for col in required):
                return df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            
    # Fallback: Just Tickers
    # Fallback: Restore from internal list
    from utils.nifty500_list import TICKERS, SECTOR_MAP
    df = pd.DataFrame({'ticker': TICKERS})
    df['sector'] = df['ticker'].map(SECTOR_MAP).fillna("Unknown")
    df['name'] = df['ticker'] # Default name
    return df

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
    
    st.toast(f"Deep scanning {len(missing)} stocks... (Accuracy Mode)", icon="🕵️")
    
    new_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(get_stock_info, t): t for t in missing}
        for f in concurrent.futures.as_completed(futures):
            try:
                res = f.result()
                if res:
                    # Ensure ticker is present
                    res['ticker'] = futures[f]
                    new_data.append(res)
            except: pass
                
    if new_data:
        update_df = pd.DataFrame(new_data)
        # Update original DF
        # We set index to ticker to align updates
        df = df.set_index('ticker')
        update_df = update_df.set_index('ticker')
        
        # Only update columns that exist in update_df
        # This fills NaNs and updates stale values
        df.update(update_df)
        
        # Add new columns if they were missing in original
        for col in update_df.columns:
            if col not in df.columns:
                df[col] = update_df[col]
                
        df = df.reset_index()
            
    return df

def fetch_and_process_market_data(tickers, fundamental_df):
    """
    Vectorized fetch of price data for all tickers + merging with fundamentals.
    """
    # 0. Ensure Fundamentals (Accuracy Mode) - checking/fetching missing PE/Sector
    fundamental_df = fetch_missing_fundamentals(fundamental_df)
    
    if not tickers:
        return fundamental_df
    if not tickers:
        return fundamental_df

    print(f"⚡ [FAST ENGINE] Fetching data for {len(tickers)} stocks...")
    
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

        # Threaded download is usually handled by yfinance internally
        history_data = yf.download(
            tickers, 
            period="1y", 
            interval="1d", 
            group_by='ticker', 
            threads=True,
            progress=False,
            auto_adjust=True
        )
    except Exception as e:
        st.error(f"Failed to fetch market data: {e}")
        return fundamental_df

    processed_rows = []
    
    # Pre-calculate sector medians for scoring
    sector_pe = {}
    if 'pe' in fundamental_df.columns and 'sector' in fundamental_df.columns:
        sector_pe = fundamental_df.groupby('sector')['pe'].median().to_dict()

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

            # 4. Trend Metrics (Rely on utils/scoring which handles the logic)
            trends = calculate_trend_metrics(base_data)
            base_data.update(trends)
            
            # 5. Full Scoring (4-Pillar)
            sector = base_data.get('sector', 'Unknown')
            median_pe = sector_pe.get(sector, 20)
            scores = calculate_scores(base_data, sector_pe_median=median_pe, sector=sector)
            base_data.update(scores)
            
            # 6. DNA-3 METRICS (Pre-Calculated for Speed)
            # RS 3M vs Nifty
            rs_3m = base_data['return_3m'] - nifty_3m_ret
            base_data['rs_3m'] = round(rs_3m, 2)
            
            # Volatility (Annualized)
            if len(df) > 60:
                daily_rets = df['Close'].pct_change().dropna()[-60:]
                volatility = daily_rets.std() * np.sqrt(252) * 100
            else:
                volatility = 0
            base_data['volatility'] = round(volatility, 1)
            
            # DNA Signal
            above_ma50 = current_price > base_data.get('fiftyDayAverage', 0)
            if rs_3m >= 2.0 and volatility >= 30 and above_ma50:
                base_data['dna_signal'] = 'BUY'
            elif rs_3m >= 2.0 and above_ma50:
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
    
    # Save to Parquet Cache for next time
    try:
        final_df.to_parquet(CACHE_FILE_PARQUET)
    except Exception:
        pass
        
    return final_df
