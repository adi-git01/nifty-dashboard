"""
INDUSTRY LEAD/LAG EVOLUTION & STRUCTURAL BREAKS
===============================================
Examines whether the "Price vs Earnings" lag dynamics hold true across various timeframes:
- Pre-COVID (2010 - 2019)
- Post-COVID (2020 - 2026)
- Last 5 Years
- Last 10 Years
- Last 15 Years (Baseline)

Also evaluates a simple industry momentum rotation strategy (buying top 3 industries based on 3M momentum) across these horizons (6mo, 1y, 5y, 10y, 15y) to test if this phenomenon is tradable.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os
import sys
import time
import json

warnings.filterwarnings('ignore')

OUTPUT_DIR = "analysis_2026/industry_lags"
SCREENER_FILE = r"c:\Users\adity\.gemini\antigravity\playground\perihelion-lunar\screener_sectors_constituents.csv"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nifty500_list import TICKERS

def fetch_data():
    start = "2007-01-01" # Fetch as much as possible for full 15Y+ padding
    nifty = yf.Ticker("^NSEI").history(start=start)
    nifty.index = nifty.index.tz_localize(None)
    
    with open("analysis_2026/industry_cycles/ticker_industry_map.json", 'r') as f:
        mapping = json.load(f)
        
    all_mapped = list(mapping.keys())
    
    bulk = yf.download(all_mapped, start=start, group_by='ticker', threads=True, progress=False, auto_adjust=True)
    
    dc_m = {}
    for t in all_mapped:
        try:
            if t in bulk.columns.get_level_values(0):
                df = bulk[t].dropna(how='all')
                if not df.empty and len(df) > 100:
                    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                    dc_m[t] = df['Close'].resample('ME').last()
        except: pass
        
    return nifty, dc_m, mapping

def cross_correlation(stock_series, earning_proxy, lag_range=6):
    corrs = {}
    for lag in range(-lag_range, lag_range + 1):
        stock_rets = stock_series.pct_change(3).dropna()
        earn = earning_proxy.dropna()
        common = stock_rets.index.intersection(earn.index)
        if len(common) < 12: continue
        corr = stock_rets.loc[common].corr(earn.loc[common].shift(lag))
        corrs[lag] = corr
    return corrs

def get_lag_for_period(series, start_date, end_date):
    mask = (series.index >= start_date) & (series.index <= end_date)
    sub = series.loc[mask]
    if len(sub) < 24: return None, None
    
    proxy_eps = sub.pct_change(12) * 100
    corrs = cross_correlation(sub, proxy_eps, lag_range=6)
    
    if corrs:
        best_lag = max(corrs, key=corrs.get)
        return best_lag, corrs[best_lag]
    return None, None

def sector_momentum_strategy(ind_indices, start_date, end_date):
    """Backtest buying the top 3 industries based on 3M momentum, rebalanced monthly."""
    df_all = pd.DataFrame(ind_indices)
    mask = (df_all.index >= start_date) & (df_all.index <= end_date)
    df = df_all.loc[mask].dropna(how='all', axis=1)
    
    if len(df) < 5: return 0.0, 0.0
    
    mom_3m = df.pct_change(3).shift(1) # Shift 1 so we trade at next open
    rets_1m = df.pct_change(1)
    
    port_rets = []
    
    for dt in df.index[4:]:
        mom = mom_3m.loc[dt].dropna()
        if len(mom) < 5: 
            port_rets.append(0)
            continue
            
        top_3 = mom.nlargest(3).index.tolist()
        r = rets_1m.loc[dt, top_3].mean()
        port_rets.append(r if pd.notna(r) else 0)
        
    if not port_rets: return 0.0, 0.0
    
    port_idx = (1 + pd.Series(port_rets)).cumprod()
    s, e = port_idx.iloc[0], port_idx.iloc[-1]
    total_ret = (e/s - 1) * 100
    years = len(port_rets) / 12
    cagr = ((e/s)**(1/years) - 1) * 100 if years > 0 else total_ret
    return cagr, total_ret

def run():
    print("=" * 100)
    print("EVOLUTION OF PRICE-EARNINGS LAGS & STRATEGY EVALUATION")
    print("=" * 100)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nifty, dc_m, mapping = fetch_data()
    
    industry_tickers = {}
    for t, ind in mapping.items():
        if ind not in industry_tickers: industry_tickers[ind] = []
        industry_tickers[ind].append(t)
        
    valid_inds = {ind: ts for ind, ts in industry_tickers.items() if len(ts) >= 3}
    
    # Build Indices
    ind_indices = {}
    for ind, ts in valid_inds.items():
        avail = [t for t in ts if t in dc_m]
        if len(avail) < 3: continue
        df_ind = pd.DataFrame({t: dc_m[t] for t in avail}).dropna(how='all')
        rets = df_ind.pct_change().replace([np.inf, -np.inf], np.nan)
        ew_ret = rets.mean(axis=1).fillna(0)
        ind_indices[ind] = (1 + ew_ret).cumprod() * 100

    # Define Time Epochs
    today = datetime.now()
    epochs = {
        'Pre-COVID (2010-2019)': (pd.to_datetime('2010-01-01'), pd.to_datetime('2019-12-31')),
        'Post-COVID (2020-2026)': (pd.to_datetime('2020-01-01'), today),
        'Last 15 Years': (today - timedelta(days=365*15), today),
        'Last 10 Years': (today - timedelta(days=365*10), today),
        'Last 5 Years': (today - timedelta(days=365*5), today),
    }
    
    # 1. Evaluate Structural Changes in Lead/Lag
    print("\n[PART 1: STRUCTURAL BREAKS IN LEAD/LAG DYNAMICS]")
    lag_shifts = []
    
    target_industries = ['IT - Software', 'Realty', 'Capital Markets', 'Banks', 'Consumer Durables', 'Textiles & Apparels', 'Pharmaceuticals & Biotechnolog']
    
    for ind in valid_inds.keys():
        if ind not in ind_indices: continue
        series = ind_indices[ind]
        
        pre_l, pre_c = get_lag_for_period(series, epochs['Pre-COVID (2010-2019)'][0], epochs['Pre-COVID (2010-2019)'][1])
        post_l, post_c = get_lag_for_period(series, epochs['Post-COVID (2020-2026)'][0], epochs['Post-COVID (2020-2026)'][1])
        
        if pre_l is not None and post_l is not None:
            pre_str = f"Leads by {abs(pre_l)}mo" if pre_l < 0 else ("Concurrent" if pre_l == 0 else f"Lags by {pre_l}mo")
            post_str = f"Leads by {abs(post_l)}mo" if post_l < 0 else ("Concurrent" if post_l == 0 else f"Lags by {post_l}mo")
            
            if ind in target_industries or pre_l != post_l:
                lag_shifts.append({
                    'Industry': ind, 'Pre-COVID_Lag': pre_l, 'Post-COVID_Lag': post_l, 
                    'Shift': f"{pre_str} -> {post_str}"
                })
                
    df_shifts = pd.DataFrame(lag_shifts).sort_values('Industry')
    print(df_shifts.to_string(index=False))
    
    # 2. Strategy Evaluation across Horizons
    print("\n[PART 2: GRANULAR SECTOR MOMENTUM ROTATION STRATEGY BACKTEST]")
    print("Strategy: Buy top 3 granular industries based on 3M trailing momentum.")
    
    horizons = {
        '15 Years': (today - timedelta(days=365*15), today),
        '10 Years': (today - timedelta(days=365*10), today),
        '5 Years': (today - timedelta(days=365*5), today),
        '1 Year': (today - timedelta(days=365*1), today),
        '6 Months': (today - timedelta(days=180), today),
    }
    
    nifty_m = nifty['Close'].resample('ME').last()
    
    print(f"\n{'Horizon':<15} | {'Strat CAGR':>12} | {'Nifty CAGR':>12} | {'Alpha':>8}")
    print("-" * 55)
    
    results = []
    for name, (start, end) in horizons.items():
        cagr, tot = sector_momentum_strategy(ind_indices, start, end)
        
        # Nifty Baseline
        n_sub = nifty_m.loc[(nifty_m.index >= start) & (nifty_m.index <= end)]
        if len(n_sub) > 1:
            n_s, n_e = n_sub.iloc[0], n_sub.iloc[-1]
            n_tot = (n_e/n_s - 1) * 100
            nyrs = (n_sub.index[-1] - n_sub.index[0]).days / 365.25
            n_cagr = ((n_e/n_s)**(1/nyrs) - 1) * 100 if nyrs > 0 else n_tot
        else:
            n_cagr = 0
            
        alpha = cagr - n_cagr if nyrs > 0.6 else tot - n_tot # Use absolutes for 6mo
        
        if name in ['1 Year', '6 Months']:
            print(f"{name:<15} | {tot:>11.1f}% | {n_tot:>11.1f}% | {alpha:>+7.1f}% (Total Ret)")
        else:
            print(f"{name:<15} | {cagr:>11.1f}% | {n_cagr:>11.1f}% | {alpha:>+7.1f}%")

        results.append({'Horizon': name, 'Strategy': cagr if name not in ['1 Year', '6 Months'] else tot, 'Nifty': n_cagr if name not in ['1 Year', '6 Months'] else n_tot, 'Alpha': alpha})
        
    df_horizons = pd.DataFrame(results)
    
    # Export to log text file to be converted to PDF later
    with open(f"{OUTPUT_DIR}/evolution_report.txt", "w") as f:
        f.write("STRUCTURAL BREAKS IN LEAD/LAG\n")
        f.write(df_shifts.to_string(index=False))
        f.write("\n\nSTRATEGY EVALUATION (SECTOR MOMENTUM)\n")
        f.write(df_horizons.to_string(index=False))

if __name__ == "__main__":
    run()
