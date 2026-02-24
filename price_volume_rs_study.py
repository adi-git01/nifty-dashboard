import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import sys
import os

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nifty500_list import TICKERS

print("Fetching historical data for Nifty 500 (Last 5 Years)...")
start = (datetime.now() - timedelta(days=365*5 + 100)).strftime('%Y-%m-%d')
nifty = yf.Ticker('^NSEI').history(start=start)
nifty.index = nifty.index.tz_localize(None)

# Download a large representative sample for speed vs statistical significance
test_tickers = TICKERS[:200]
bulk = yf.download(test_tickers, start=start, progress=False, threads=True)

results = []

def get_trend(series, window=20):
    """Categorize trend over a window: UP, DOWN, or FLAT"""
    if len(series) < window: return np.nan
    pct_change = (series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100
    if pct_change > 5: return "UP"
    elif pct_change < -5: return "DOWN"
    else: return "FLAT"

print("Analyzing Price-Volume-RS dynamics across 5 years of daily data...")

# Use a step size to avoid processing every single day (for speed), process end of weeks
dates_to_test = nifty.index[100::5]

for t in test_tickers:
    try:
        if isinstance(bulk.columns, pd.MultiIndex):
            if t in bulk.columns.get_level_values(1):
                df = bulk.xs(t, axis=1, level=1).dropna(how='all')
        else:
            if t in bulk.columns:
                df = bulk[t].dropna()
        
        if df is None or len(df) < 200: continue
        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
        
        # Calculate Rolling RS vs Nifty (20 day)
        df['Ret_20'] = df['Close'].pct_change(20)
        # Nifty Ret 20 aligned to df index
        nifty_aligned = nifty['Close'].reindex(df.index, method='ffill')
        df['Nifty_Ret_20'] = nifty_aligned.pct_change(20)
        df['RS_20'] = df['Ret_20'] - df['Nifty_Ret_20']
        
        for date in dates_to_test:
            idx = df.index.searchsorted(date)
            if idx < 60 or idx >= len(df) - 60: continue # Need history and future
            
            # 20-Day Lookback Window
            window = df.iloc[idx-20:idx+1]
            
            price_trend = get_trend(window['Close'])
            vol_trend = get_trend(window['Volume'].rolling(5).mean().dropna(), window=15) # Smooth volume
            
            # RS Trend (did RS improve or degrade over 20 days)
            rs_start = df['RS_20'].iloc[idx-20]
            rs_end = df['RS_20'].iloc[idx]
            if rs_end - rs_start > 0.05: rs_trend = "UP"
            elif rs_end - rs_start < -0.05: rs_trend = "DOWN"
            else: rs_trend = "FLAT"
            
            # Forward Returns (20d and 60d)
            fwd_20d = (df['Close'].iloc[idx+20] - df['Close'].iloc[idx]) / df['Close'].iloc[idx] * 100
            fwd_60d = (df['Close'].iloc[idx+60] - df['Close'].iloc[idx]) / df['Close'].iloc[idx] * 100
            
            state = f"P:{price_trend} | V:{vol_trend} | RS:{rs_trend}"
            
            results.append({
                'Ticker': t,
                'Date': date,
                'State': state,
                'Price': price_trend,
                'Volume': vol_trend,
                'RS': rs_trend,
                'Fwd_20d': fwd_20d,
                'Fwd_60d': fwd_60d
            })
            
    except Exception as e:
        continue

res_df = pd.DataFrame(results)

print("\n" + "="*80)
print("DEEP DIVE: PRICE, VOLUME & RS RELATIONSHIPS (20-DAY FORWARD RETURNS)")
print("="*80)

# 1. Price vs Volume Matrix
print("\n1. THE PRICE-VOLUME MATRIX (Does volume confirm price?)")
pv_summary = res_df.groupby(['Price', 'Volume']).agg(
    Samples=('Fwd_20d', 'count'),
    WinRate_20d=('Fwd_20d', lambda x: (x>0).mean()*100),
    AvgRet_20d=('Fwd_20d', 'mean')
).reset_index()

# Sort logically
pv_summary['sort_key'] = pv_summary['Price'].map({'UP': 1, 'FLAT': 2, 'DOWN': 3}) * 10 + pv_summary['Volume'].map({'UP': 1, 'FLAT': 2, 'DOWN': 3})
pv_summary = pv_summary.sort_values('sort_key').drop('sort_key', axis=1)
print(pv_summary.to_string(index=False, float_format="%.1f"))

# 2. The Flat Zone
print("\n2. THE FLAT ZONE (Price FLAT, Volume FLAT)")
flat_stats = res_df[(res_df['Price'] == 'FLAT') & (res_df['Volume'] == 'FLAT')]
print(f"Samples: {len(flat_stats)} | 20d WinRate: {(flat_stats['Fwd_20d']>0).mean()*100:.1f}% | 20d Avg Ret: {flat_stats['Fwd_20d'].mean():.1f}%")
print("Insight: Flat price with flat volume is a coin flip. It is a waiting period, not a predictive edge on its own.")

# 3. RS Divergences
print("\n3. RELATIVE STRENGTH DIVERGENCES")
rs_div = res_df.groupby(['Price', 'RS']).agg(
    Samples=('Fwd_20d', 'count'),
    WinRate_20d=('Fwd_20d', lambda x: (x>0).mean()*100),
    AvgRet_20d=('Fwd_20d', 'mean')
).reset_index()
print(rs_div[rs_div['Price'] != rs_div['RS']].to_string(index=False, float_format="%.1f"))
print("\nInsight: Notice 'Price DOWN, RS UP'. Prices are falling, but falling SLOWER than the market. This divergence often predicts bottoms.")

# 4. The Holy Grail vs The Exhaustion
print("\n4. CONFIRMATION VS EXHAUSTION (Price UP scenarios)")
up_up = res_df[(res_df['Price']=='UP') & (res_df['Volume']=='UP') & (res_df['RS'] == 'UP')]
up_down = res_df[(res_df['Price']=='UP') & (res_df['Volume']=='DOWN')]

print(f"Holy Grail (Price UP, Vol UP, RS UP) -> 20d Win: {(up_up['Fwd_20d']>0).mean()*100:.1f}% | Ret: {up_up['Fwd_20d'].mean():.1f}% | 60d Ret: {up_up['Fwd_60d'].mean():.1f}%")
print(f"Exhaustion (Price UP, Vol DOWN)      -> 20d Win: {(up_down['Fwd_20d']>0).mean()*100:.1f}% | Ret: {up_down['Fwd_20d'].mean():.1f}% | 60d Ret: {up_down['Fwd_60d'].mean():.1f}%")

