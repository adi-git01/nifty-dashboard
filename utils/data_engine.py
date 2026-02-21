
import yfinance as yf
import streamlit as st
import pandas as pd

# Import sector mapping from nifty500 list
try:
    from utils.nifty500_list import SECTOR_MAP
    from utils.sector_mapping import consolidate_sector
except ImportError:
    SECTOR_MAP = {}
    consolidate_sector = lambda x: x

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_stock_info(ticker):
    """
    Fetches fundamental info for a single ticker.
    Returns a dictionary of relevant metrics or None if failed.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # PRIORITY: Use CSV sector mapping (more reliable than yfinance)
        # SECTOR_MAP has the granular sector from our curated CSV
        granular_sector = SECTOR_MAP.get(ticker, info.get("sector", "Unknown"))
        if not granular_sector or granular_sector == "Unknown":
            granular_sector = info.get("sector", "Unknown")
        
        # Consolidate to broader sector category
        broad_sector = consolidate_sector(granular_sector)
        
        # === ROBUST PE FETCHING ===
        # Try multiple fields for PE
        pe = info.get("trailingPE") or info.get("pe") or info.get("priceToEarnings")
        
        # If still None, calculate from Price / EPS
        if pe is None:
            price = info.get("currentPrice") or info.get("previousClose") or info.get("regularMarketPrice")
            eps = info.get("trailingEps") or info.get("forwardEps")
            if price and eps and eps != 0:
                pe = price / eps
        
        # Forward PE
        forward_pe = info.get("forwardPE")
        if forward_pe is None:
            price = info.get("currentPrice") or info.get("previousClose")
            fwd_eps = info.get("forwardEps")
            if price and fwd_eps and fwd_eps != 0:
                forward_pe = price / fwd_eps
        
        # === ROBUST PEG FETCHING ===
        peg = info.get("pegRatio") or info.get("trailingPegRatio")
        
        # If PEG missing, calculate from PE / Growth
        if peg is None and pe:
            growth_rate = info.get("earningsGrowth") or info.get("revenueGrowth")
            if growth_rate and growth_rate != 0:
                peg = pe / (growth_rate * 100)  # Growth is decimal, convert to %
        
        # === ROBUST ROE/ROA FETCHING ===
        roe = info.get("returnOnEquity") or info.get("roe")
        roa = info.get("returnOnAssets") or info.get("roa")
        
        # If ROE still None, try calculating from financials
        if roe is None:
            try:
                bs = stock.balance_sheet
                inc = stock.financials
                if not bs.empty and not inc.empty:
                    if 'Stockholders Equity' in bs.index and 'Net Income' in inc.index:
                        equity = bs.loc['Stockholders Equity'].iloc[0]
                        net_income = inc.loc['Net Income'].iloc[0]
                        if equity and equity != 0:
                            roe = net_income / equity
            except:
                pass
        
        # If ROA still None, try calculating from financials
        if roa is None:
            try:
                bs = stock.balance_sheet
                inc = stock.financials
                if not bs.empty and not inc.empty:
                    if 'Total Assets' in bs.index and 'Net Income' in inc.index:
                        assets = bs.loc['Total Assets'].iloc[0]
                        net_income = inc.loc['Net Income'].iloc[0]
                        if assets and assets != 0:
                            roa = net_income / assets
            except:
                pass
        
        # === EARNINGS QUALITY (Operating Cash Flow / Net Income) ===
        earnings_quality = None
        operating_cf = info.get("operatingCashflow")
        net_income_ttm = info.get("netIncomeToCommon")
        
        if operating_cf and net_income_ttm and net_income_ttm != 0:
            earnings_quality = operating_cf / net_income_ttm
        else:
            # Try from cash flow statement
            try:
                cf = stock.cashflow
                inc = stock.financials
                if not cf.empty and not inc.empty:
                    if 'Operating Cash Flow' in cf.index and 'Net Income' in inc.index:
                        ocf = cf.loc['Operating Cash Flow'].iloc[0]
                        ni = inc.loc['Net Income'].iloc[0]
                        if ni and ni != 0:
                            earnings_quality = ocf / ni
            except:
                pass
        
        # === EARNINGS TREND (YoY Net Income Growth from last 2 years) ===
        earnings_trend = None
        try:
            inc = stock.financials
            if not inc.empty and 'Net Income' in inc.index:
                ni_series = inc.loc['Net Income'].dropna()
                if len(ni_series) >= 2:
                    ni_current = ni_series.iloc[0]
                    ni_previous = ni_series.iloc[1]
                    if ni_previous and ni_previous != 0:
                        earnings_trend = (ni_current - ni_previous) / abs(ni_previous)
        except:
            pass
        
        # === ROBUST MARGINS FETCHING ===
        profit_margins = info.get("profitMargins")
        if profit_margins is None:
            try:
                inc = stock.financials
                if not inc.empty:
                    # Try to find Net Income and Total Revenue
                    ni_key = next((k for k in ['Net Income', 'Net Income Common Stockholders'] if k in inc.index), None)
                    rev_key = next((k for k in ['Total Revenue', 'Operating Revenue'] if k in inc.index), None)
                    
                    if ni_key and rev_key:
                        ni = inc.loc[ni_key].iloc[0]
                        rev = inc.loc[rev_key].iloc[0]
                        if rev and rev != 0:
                            profit_margins = ni / rev
            except:
                pass
        
        # Extract only what we need to minimize data transfer/storage
        data = {
            "ticker": ticker,
            "name": info.get("longName", ticker),
            "sector": broad_sector,
            "sector_granular": granular_sector,
            "industry": info.get("industry", "Unknown"),
            "price": info.get("currentPrice") or info.get("previousClose") or info.get("regularMarketPrice") or 0.0,
            "currentPrice": info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose") or 0.0,
            "marketCap": info.get("marketCap", 0),
            "pe": pe,  # Robust PE
            "forwardPE": forward_pe,  # Robust Forward PE
            "pegRatio": peg,  # Robust PEG
            "pb": info.get("priceToBook"),
            "roe": roe,
            "roa": roa,  # Use our enhanced lookup (renamed from roic)
            "profitMargins": profit_margins,
            "grossMargins": info.get("grossMargins"),
            "revenueGrowth": info.get("revenueGrowth"),
            "earningsGrowth": info.get("earningsGrowth"),
            "debtToEquity": info.get("debtToEquity"),
            "freeCashflow": info.get("freeCashflow"),
            "52WeekChange": info.get("52WeekChange", 0),
            "beta": info.get("beta", 1.0),
            "summary": info.get("longBusinessSummary", "No summary available."),
            # === TECHNICAL FIELDS FOR TREND SCORING ===
            "fiftyDayAverage": info.get("fiftyDayAverage", 0),
            "twoHundredDayAverage": info.get("twoHundredDayAverage", 0),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh", 0),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow", 0),
            # === VOLUME DATA FOR VOLUME SIGNAL SCORE ===
            "averageVolume": info.get("averageVolume", 0),
            "averageVolume10days": info.get("averageVolume10days", 0),
            "volume": info.get("volume", 0),
            "averageDailyVolume10Day": info.get("averageDailyVolume10Day", 0),
            # === GROWTH METRICS ===
            "earningsQuarterlyGrowth": info.get("earningsQuarterlyGrowth"),  # QoQ
            "operatingMargins": info.get("operatingMargins"),      # Operating margin
            "ebitdaMargins": info.get("ebitdaMargins"),           # EBITDA margin
            # === EARNINGS QUALITY & TREND ===
            "earningsQuality": earnings_quality,  # OCF/Net Income (>1 is good)
            "earningsTrend": earnings_trend,      # YoY earnings growth from financials
        }
        return data
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

@st.cache_data(ttl=3600)
def get_stock_history(ticker, period="1y"):
    """
    Fetches historical price data.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        print(f"Error fetching history for {ticker}: {e}")
        return pd.DataFrame()

import os
import time
import concurrent.futures

CACHE_FILE = "nifty500_cache.csv"
CACHE_EXPIRY = 43200  # 12 hours in seconds

def batch_fetch_tickers(ticker_list, max_workers=20):
    """
    Fetches data for a list of tickers. Uses local CSV cache to speed up Nifty 500 loading.
    """
    # 1. Try Load from Cache
    if os.path.exists(CACHE_FILE):
        mod_time = os.path.getmtime(CACHE_FILE)
        if (time.time() - mod_time) < CACHE_EXPIRY:
            st.toast("Loaded data from local cache (fast load)", icon="⚡")
            try:
                df = pd.read_csv(CACHE_FILE)
                # Filter to only requested tickers if needed, or just return all
                # Ideally we check if cache covers the list. For simplicity, if cache exists we assume it's good.
                return df
            except Exception as e:
                print(f"Cache read error: {e}")
    
    # 2. Fetch Live
    st.toast("Fetching live data for Nifty 500... this may take 1-2 minutes.", icon="⏳")
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(ticker_list)
    completed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(get_stock_info, t): t for t in ticker_list}
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            try:
                data = future.result()
                if data:
                    results.append(data)
            except Exception:
                pass # Fail silently for bulk
            
            completed += 1
            if completed % 5 == 0: # Update UI every 5
                progress_bar.progress(min(completed / total, 1.0))
                status_text.text(f"Fetched {completed}/{total} companies...")
            
    progress_bar.empty()
    status_text.empty()
    
    final_df = pd.DataFrame(results)
    
    # 3. Save to Cache
    if not final_df.empty:
        final_df.to_csv(CACHE_FILE, index=False)
        st.toast(f"Cached {len(final_df)} stocks to disk.", icon="💾")
        
    return final_df
