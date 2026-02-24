import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import sys
import os
import time

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nifty500_list import TICKERS

OUTPUT_DIR = "analysis_2026/volume"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. VOLUME INDICATOR CALCULATION FUNCTIONS
# -------------------------------------------------------------
def calc_up_down_ratio(df, window=50):
    """Up/Down Volume Ratio: Ratio of Volume on Up-days vs Down-days."""
    if len(df) < window: return np.nan
    df['Price_Change'] = df['Close'].diff()
    up_vol = df['Volume'].where(df['Price_Change'] > 0, 0).rolling(window=window).sum()
    down_vol = df['Volume'].where(df['Price_Change'] < 0, 0).rolling(window=window).sum()
    # Avoid div/0
    ud_ratio = up_vol / down_vol.replace(0, np.nan)
    return ud_ratio.iloc[-1]

def calc_obv_trend(df, window=20):
    """On-Balance Volume (OBV) Trend: Is OBV making new highs?"""
    if len(df) < window: return np.nan
    
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    obv_ma = obv.rolling(window).mean()
    # Return % distance of OBV from its 20-day MA
    dist = (obv.iloc[-1] - obv_ma.iloc[-1]) / abs(obv_ma.iloc[-1]) if obv_ma.iloc[-1] != 0 else 0
    return dist

def calc_cmf(df, window=20):
    """Chaikin Money Flow (CMF): Measures buying/selling pressure."""
    if len(df) < window: return np.nan
    # Money Flow Multiplier
    mf_mult = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mf_mult = mf_mult.fillna(0)  # Handle div/0 if High == Low
    
    # Money Flow Volume
    mf_vol = mf_mult * df['Volume']
    
    # 20-day CMF
    cmf = mf_vol.rolling(window).sum() / df['Volume'].rolling(window).sum()
    return cmf.iloc[-1]

def calc_vol_surge(df, window=50):
    """Volume Surge: Is current 5-day volume significantly higher than 50-day?"""
    if len(df) < window: return np.nan
    vol_5d = df['Volume'].rolling(5).mean().iloc[-1]
    vol_50d = df['Volume'].rolling(50).mean().iloc[-1]
    return vol_5d / vol_50d if vol_50d > 0 else 1.0

# -------------------------------------------------------------
# 2. DATA FETCHING
# -------------------------------------------------------------
def fetch_data(years=5):
    start = (datetime.now() - timedelta(days=365 * years + 100)).strftime('%Y-%m-%d')
    print("Fetching Nifty...")
    nifty = yf.Ticker('^NSEI').history(start=start)
    nifty.index = nifty.index.tz_localize(None)
    
    print(f"Bulk downloading {len(TICKERS[:300])} stocks for Matrix Test...")
    bulk = yf.download(TICKERS[:300], start=start, progress=False, threads=True)
    
    cache = {}
    for t in TICKERS[:300]:
        try:
            # Handle yfinance multi-index columns properly
            if isinstance(bulk.columns, pd.MultiIndex):
                # Format: [('Close', 'AAPL'), ('Volume', 'AAPL')]
                if t in bulk.columns.get_level_values(1):
                    df = bulk.xs(t, axis=1, level=1).dropna(how='all')
                    if len(df) > 200:
                        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                        cache[t] = df
            else:
                # Fallback
                if t in bulk.columns:
                    df = bulk[t].dropna()
                    if len(df) > 200:
                        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                        cache[t] = df
        except Exception as e:
            pass
    print(f"Loaded {len(cache)} valid stocks.")
    return nifty, cache

# -------------------------------------------------------------
# 3. BACKTEST ENGINE
# -------------------------------------------------------------
def backtest_volume_filters(nifty, cache):
    """Test standard RS Breakout entry with varying Volume Filters."""
    
    dates = nifty.index[nifty.index > (datetime.now() - timedelta(days=365 * 5))]
    
    # Defines the variants we are testing
    variants = {
        'Baseline (No Vol Filter)': lambda ind: True,
        'Up/Down Ratio > 1.2':      lambda ind: ind['ud_ratio'] > 1.2,
        'OBV Trend > 0':            lambda ind: ind['obv_trend'] > 0,
        'CMF > 0.1':                lambda ind: ind['cmf'] > 0.1,
        'Volume Surge > 1.5x':      lambda ind: ind['vol_surge'] > 1.5,
        'U/D>1.2 AND CMF>0.1':      lambda ind: ind['ud_ratio'] > 1.2 and ind['cmf'] > 0.1
    }
    
    results_trades = {v: [] for v in variants}
    results_equity = {v: [] for v in variants}
    
    for name in variants:
        print(f"Testing Variant: {name}")
        capital = 1_000_000
        positions = {}
        day_counter = 0
        
        for date in dates:
            # 1. Check Exits (15% Trailing or MA50 break)
            to_remove = []
            for t, pos in positions.items():
                if t in cache:
                    df = cache[t]
                    idx = df.index.searchsorted(date)
                    if idx > 0 and idx < len(df):
                        price = df['Close'].iloc[idx]
                        ma50 = df['Close'].iloc[:idx+1].rolling(50).mean().iloc[-1]
                        
                        if price > pos['peak']: pos['peak'] = price
                        
                        if price < pos['peak'] * 0.85 or price < ma50:
                            ret = (price - pos['entry']) / pos['entry']
                            capital += pos['shares'] * price
                            results_trades[name].append({
                                'Ticker': t, 'PnL%': ret * 100, 'Hold_Days': (date - pos['entry_date']).days
                            })
                            to_remove.append(t)
            for t in to_remove: del positions[t]
            
            # 2. Scan & Enter (Every 10 Days)
            if day_counter % 10 == 0 and len(positions) < 10:
                candidates = []
                for t, df in cache.items():
                    if t in positions: continue
                    idx = df.index.searchsorted(date)
                    if idx < 100 or idx >= len(df): continue
                    
                    window = df.iloc[:idx+1]
                    price = window['Close'].iloc[-1]
                    ma50 = window['Close'].rolling(50).mean().iloc[-1]
                    
                    if price < ma50: continue # Basic trend filter
                    
                    # Calculate RS
                    nifty_idx = nifty.index.searchsorted(date)
                    if nifty_idx < 63: continue
                    t_ret = (price - window['Close'].iloc[-63]) / window['Close'].iloc[-63]
                    n_ret = (nifty['Close'].iloc[nifty_idx] - nifty['Close'].iloc[nifty_idx-63]) / nifty['Close'].iloc[nifty_idx-63]
                    rs_score = (t_ret - n_ret) * 100
                    
                    if rs_score < 10: continue # Base RS filter
                    
                    # Compute Volume Indicators
                    ind = {
                        'ud_ratio': calc_up_down_ratio(window, 50),
                        'obv_trend': calc_obv_trend(window, 20),
                        'cmf': calc_cmf(window, 20),
                        'vol_surge': calc_vol_surge(window, 50),
                        'rs_score': rs_score,
                        'price': price
                    }
                    
                    if variants[name](ind):
                        candidates.append((t, ind))
                
                # Rank by RS and Buy
                candidates.sort(key=lambda x: -x[1]['rs_score'])
                free_slots = 10 - len(positions)
                for t, ind in candidates[:free_slots]:
                    size = capital / (free_slots + 1)
                    shares = int(size / ind['price'])
                    if shares > 0:
                        capital -= (shares * ind['price'])
                        positions[t] = {'entry': ind['price'], 'peak': ind['price'], 'shares': shares, 'entry_date': date}
            
            # Equity record
            eq = capital
            for t, pos in positions.items():
                df = cache[t]
                idx = df.index.searchsorted(date)
                if idx < len(df): eq += pos['shares'] * df['Close'].iloc[idx]
            results_equity[name].append(eq)
            
            day_counter += 1

    # -------------------------------------------------------------
    # 4. PRINT RESULTS
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print("VOLUME MATRIX BACKTEST RESULTS (5 YEARS)")
    print("="*80)
    print(f"{'Variant':<25s} | {'CAGR%':>8s} | {'Trades':>6s} | {'Win Rate':>8s} | {'Avg PnL':>8s}")
    print("-" * 65)
    
    metrics = []
    for name in variants:
        eq = results_equity[name]
        cagr = ((eq[-1] / eq[0]) ** (1/5) - 1) * 100 if len(eq) > 0 else 0
        
        trades = pd.DataFrame(results_trades[name])
        if not trades.empty:
            num_trades = len(trades)
            win_rate = (trades['PnL%'] > 0).mean() * 100
            avg_pnl = trades['PnL%'].mean()
        else:
            num_trades, win_rate, avg_pnl = 0, 0, 0
            
        print(f"{name:<25s} | {cagr:>7.1f}% | {num_trades:>6d} | {win_rate:>7.1f}% | {avg_pnl:>7.1f}%")

if __name__ == "__main__":
    nifty, cache = fetch_data()
    backtest_volume_filters(nifty, cache)
