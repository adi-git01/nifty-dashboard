"""
DEEP INDUSTRY CYCLE & LAG ANALYSIS (15Y)
========================================
1. Cycle Length & Amplitude: Average months from trough to peak to trough
2. Earnings vs Price Lag: Does price lead or lag earnings (margin proxy) and by how much?
3. Deep Seasonality: Win rates by month and best entry windows per granular industry
4. Regime Performance: Which granular industries win in Bear vs Bull markets
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
import re

warnings.filterwarnings('ignore')

OUTPUT_DIR = "analysis_2026/industry_cycles"
SCREENER_FILE = r"c:\Users\adity\.gemini\antigravity\playground\perihelion-lunar\screener_sectors_constituents.csv"
MIN_STOCKS = 3

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nifty500_list import TICKERS

def clean_name(name):
    name = str(name).lower()
    for w in [' ltd.', ' ltd', ' limited', ' corp.', ' corp', ' corporation', ' inc.', ' inc', ' co.', ' co']:
        name = name.replace(w, '')
    return re.sub(r'[^a-z0-9]', '', name)

def map_tickers(df_scr):
    print("  Mapping Nifty 500 tickers to granular industries...")
    
    # Simple mapping file cache
    map_file = f"{OUTPUT_DIR}/ticker_industry_map.json"
    if os.path.exists(map_file):
        with open(map_file, 'r') as f:
            return json.load(f)
    
    scr_names = {clean_name(row['Name']): row['Sector'] for _, row in df_scr.iterrows()}
    
    mapping = {}
    
    # We use yfinance to get the shortName for each ticker, then match
    # Since this takes time, we'll do it batched/fast
    print("  Fetching Nifty 500 info to map to Screener names...")
    
    # Fast approach: use TICKERS list, strip .NS, and do substring match
    # Example: 'RELIANCE.NS' -> 'reliance'
    unmapped = []
    
    for t in TICKERS:
        base = clean_name(t.replace('.NS', '').replace('.BO', ''))
        matched = False
        
        # Exact match attempt
        for scr_n, sec in scr_names.items():
            if base == scr_n or base in scr_n or scr_n in base:
                mapping[t] = sec
                matched = True
                break
                
        if not matched:
            unmapped.append(t)
            
    # Try harder for unmapped using yfinance shortName
    if unmapped:
        print(f"  Attempting yfinance fallback for {len(unmapped)} unmapped tickers...")
        try:
            bulk_info = yf.Tickers(" ".join(unmapped[:100])) # Only try first 100 to save time
            for t in unmapped[:100]:
                try:
                    name = clean_name(bulk_info.tickers[t].info.get('shortName', ''))
                    if name:
                        for scr_n, sec in scr_names.items():
                            if name in scr_n or scr_n in name:
                                mapping[t] = sec
                                break
                except: pass
        except: pass

    with open(map_file, 'w') as f:
        json.dump(mapping, f)
        
    print(f"  Successfully mapped {len(mapping)} / {len(TICKERS)} tickers to {len(set(mapping.values()))} industries.")
    return mapping


def get_market_regimes(nifty):
    """Categorize months into Bull/Bear/Sideways based on 6M Nifty return."""
    nifty_m = nifty['Close'].resample('ME').last()
    regimes = {}
    
    ret_6m = nifty_m.pct_change(6) * 100
    ret_1m = nifty_m.pct_change(1) * 100
    
    for date, ret in ret_6m.dropna().items():
        if ret > 10:
            regimes[date] = 'BULL'
        elif ret < -5:
            regimes[date] = 'BEAR'
        else:
            regimes[date] = 'SIDEWAYS'
            
    return pd.Series(regimes)

def analyze_cycles(industry_series):
    """
    Detects market cycles (peak to peak / trough to trough) for an industry.
    Returns avg cycle length in months and avg amplitude.
    """
    if len(industry_series) < 60:
        return 0, 0, 0
        
    # Smoothing out noise (3-month SMA)
    smooth = industry_series.rolling(3).mean().dropna()
    
    # Find local peaks and troughs
    peaks = []
    troughs = []
    
    window = 6 # Must be peak/trough for 6 months around it
    for i in range(window, len(smooth) - window):
        chunk = smooth.iloc[i-window:i+window+1]
        mid = smooth.iloc[i]
        
        if mid == chunk.max():
            peaks.append((smooth.index[i], mid))
        elif mid == chunk.min():
            troughs.append((smooth.index[i], mid))
            
    # Clean up (alternate peaks and troughs)
    if not peaks or not troughs:
        return 0, 0, 0
        
    # Calculate cycle lengths (trough to trough)
    cycle_lengths_mo = []
    amplitudes = [] # Peak return from trough
    
    for i in range(1, len(troughs)):
        t1_date, t1_val = troughs[i-1]
        t2_date, t2_val = troughs[i]
        
        # Find peak in between
        middle_peaks = [p for p in peaks if t1_date < p[0] < t2_date]
        if middle_peaks:
            p_val = max([p[1] for p in middle_peaks])
            amp = (p_val - t1_val) / t1_val * 100
            if amp > 15: # Significant cycle
                cycle_lengths_mo.append(round((t2_date - t1_date).days / 30.4, 1))
                amplitudes.append(amp)
                
    if not cycle_lengths_mo:
        return 0, 0, 0
        
    return np.mean(cycle_lengths_mo), np.median(cycle_lengths_mo), np.mean(amplitudes)


def cross_correlation(stock_series, earning_proxy, lag_range=4):
    """Calculate cross-correlation between stock returns and an earnings proxy (like YoY returns acting as earnings growth) at various lags to find who leads."""
    # We use 12M trailing return as a proxy for earnings growth momentum
    corrs = {}
    
    for lag in range(-lag_range, lag_range + 1):
        # negative lag = earnings proxy is shifted backwards = price is correlated with FUTURE earnings -> PRICE LEADS
        # positive lag = earnings proxy is shifted forwards = price is correlated with PAST earnings -> EARNINGS LEAD
        stock_rets = stock_series.pct_change(3).dropna()
        earn = earning_proxy.dropna()
        
        # Align index
        common = stock_rets.index.intersection(earn.index)
        if len(common) < 20: continue
            
        corr = stock_rets.loc[common].corr(earn.loc[common].shift(lag))
        corrs[lag] = corr
        
    return corrs

def run():
    print("=" * 120)
    print("DEEP INDUSTRY CYCLE & LAG ANALYSIS (15 YEAR)")
    print("=" * 120)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_scr = pd.read_csv(SCREENER_FILE)
    
    # 1. Map tickers
    mapping = map_tickers(df_scr)
    industry_tickers = {}
    for t, ind in mapping.items():
        if ind not in industry_tickers: industry_tickers[ind] = []
        industry_tickers[ind].append(t)
        
    # Filter to valid industries
    valid_industries = {ind: ts for ind, ts in industry_tickers.items() if len(ts) >= MIN_STOCKS}
    print(f"\nAnalyzing {len(valid_industries)} granular industries (>= {MIN_STOCKS} tickers each)")
    
    # 2. Fetch Data
    print("\n[1/3] Fetching 15Y Nifty...")
    start = (datetime.now() - timedelta(days=365*16)).strftime('%Y-%m-%d')
    nifty = yf.Ticker("^NSEI").history(start=start)
    nifty.index = nifty.index.tz_localize(None)
    nifty_m = nifty['Close'].resample('ME').last()
    regimes = get_market_regimes(nifty)
    
    all_mapped = [t for ts in valid_industries.values() for t in ts]
    print(f"[2/3] Downloading {len(all_mapped)} stocks for 15Y history...")
    t0 = time.time()
    try:
        bulk = yf.download(all_mapped, start=start, group_by='ticker', threads=True, progress=False, auto_adjust=True)
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return
    print(f"[3/3] Completed download in {time.time()-t0:.0f}s")
    
    # Build monthly price dictionary
    dc_m = {}
    for t in all_mapped:
        try:
            if t in bulk.columns.get_level_values(0):
                df = bulk[t].dropna(how='all')
                if not df.empty and len(df) > 500:
                    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                    dc_m[t] = df['Close'].resample('ME').last()
        except: pass
        
    print(f"Data available for {len(dc_m)} stocks.")
    
    # 3. Build Industry Equal-Weight Indices
    print("\nBuilding Industry Indices...")
    ind_indices = {}
    for ind, ts in valid_industries.items():
        avail = [t for t in ts if t in dc_m]
        if len(avail) < 3: continue
        
        # We need a unified dataframe to mean() across columns
        df_ind = pd.DataFrame({t: dc_m[t] for t in avail}).dropna(how='all')
        
        # Calculate daily returns, replace inf with nan
        rets = df_ind.pct_change().replace([np.inf, -np.inf], np.nan)
        
        # Equal weight average return
        ew_ret = rets.mean(axis=1).fillna(0)
        
        # Construct index starting at 100
        ind_idx = (1 + ew_ret).cumprod() * 100
        ind_indices[ind] = ind_idx
        
    print(f"Generated indices for {len(ind_indices)} industries.")

    # 4. Perform Analysis
    print("\n" + "=" * 120)
    print("ANALYSIS RESULTS: CYCLES, SEASONALITY & LAG")
    print("=" * 120)
    
    results = []
    
    for ind, ind_series in ind_indices.items():
        if len(ind_series) < 60: continue
            
        print(f"\n{'_'*100}")
        print(f"INDUSTRY: {ind.upper()}")
        print(f"{'_'*100}")
        
        # A) Cycle Length
        mean_cyc, med_cyc, mean_amp = analyze_cycles(ind_series)
        print(f"  ▶ CYCLICALITY: ", end="")
        if mean_cyc > 0:
            print(f"Avg Cycle Length: {mean_cyc:.1f} months. Avg Amplitude: +{mean_amp:.1f}%")
        else:
            print("No clear cyclical pattern detected (Linear/Structural).")
            
        # B) Price vs Earnings Momentum Lag
        # Proxy: We correlate 3M price change with 12M price change 
        # (12M price change is often highly correlated with trailing EPS growth)
        rets_3m = ind_series.pct_change(3) * 100
        rets_1m = ind_series.pct_change(1) * 100
        proxy_eps = ind_series.pct_change(12) * 100 # Proxy for earnings trend
        
        corrs = cross_correlation(ind_series, proxy_eps, lag_range=6)
        
        if corrs:
            best_lag = max(corrs, key=corrs.get)
            best_corr = corrs[best_lag]
            print(f"  ▶ PRICE VS EARNINGS (Proxy) LAG: ", end="")
            if best_corr < 0.3:
                print("Low correlation between price and earnings proxy.")
            elif best_lag < 0:
                print(f"Price LEADS earnings proxy by {abs(best_lag)} months (Corr: {best_corr:.2f})")
            elif best_lag > 0:
                print(f"Price LAGS earnings proxy by {best_lag} months (Corr: {best_corr:.2f})")
            else:
                print(f"Price moves CONCURRENTLY with earnings proxy (Corr: {best_corr:.2f})")
        else:
            best_lag = 0
            
        # C) Regime Performance (Alpha over Nifty)
        # Average monthly alpha in Bull vs Bear
        regime_alpha = {'BULL': [], 'BEAR': []}
        
        for date, regime in regimes.items():
            if date in rets_1m.index and date in nifty_m.index:
                ind_r = rets_1m.loc[date]
                nif_r = (nifty_m.loc[date] / nifty_m.shift(1).loc[date] - 1) * 100
                if pd.notna(ind_r) and pd.notna(nif_r):
                    if regime in ['BULL', 'BEAR']:
                        regime_alpha[regime].append(ind_r - nif_r)
                        
        avg_bull_alpha = np.mean(regime_alpha['BULL']) if regime_alpha['BULL'] else 0
        avg_bear_alpha = np.mean(regime_alpha['BEAR']) if regime_alpha['BEAR'] else 0
        
        print(f"  ▶ REGIME ALPHA: Bull Alpha: {avg_bull_alpha:>+5.1f}%/mo | Bear Alpha: {avg_bear_alpha:>+5.1f}%/mo")
        
        # D) Seasonality
        ind_df = pd.DataFrame({'Ret': rets_1m})
        ind_df['Month'] = ind_df.index.month
        
        monthly_win = {}
        monthly_avg = {}
        for m in range(1, 13):
            m_rets = ind_df[ind_df['Month'] == m]['Ret'].dropna()
            if len(m_rets) > 3:
                monthly_win[m] = (m_rets > 0).mean() * 100
                monthly_avg[m] = m_rets.mean()
            else:
                monthly_win[m] = 0
                monthly_avg[m] = 0
                
        best_month = max(monthly_avg, key=monthly_avg.get)
        worst_month = min(monthly_avg, key=monthly_avg.get)
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        print(f"  ▶ SEASONALITY: Best= {months[best_month-1]} ({monthly_avg[best_month]:+.1f}%, {monthly_win[best_month]:.0f}% win) | Worst= {months[worst_month-1]} ({monthly_avg[worst_month]:+.1f}%, {monthly_win[worst_month]:.0f}% win)")
        
        results.append({
            'Industry': ind,
            'Cycle_Length_Mo': round(mean_cyc, 1),
            'Cycle_Amplitude%': round(mean_amp, 1),
            'Price_Leads_Mo': abs(best_lag) if best_lag < 0 else (-best_lag if best_lag > 0 else 0),
            'Bull_Alpha%': round(avg_bull_alpha, 2),
            'Bear_Alpha%': round(avg_bear_alpha, 2),
            'Best_Month': months[best_month-1],
            'Worst_Month': months[worst_month-1]
        })
        
    df_results = pd.DataFrame(results)
    df_results.to_csv(f"{OUTPUT_DIR}/granular_industry_analysis.csv", index=False)
    
    print(f"\n{'-'*120}")
    print("TOP CYCLICAL INDUSTRIES (Fastest Cycle Length)")
    print(df_results[df_results['Cycle_Length_Mo'] > 0].sort_values('Cycle_Length_Mo').head(5)[['Industry', 'Cycle_Length_Mo', 'Cycle_Amplitude%']].to_string(index=False))

    print(f"\nTOP BEAR MARKET PROTECTORS (Best Bear Alpha)")
    print(df_results.sort_values('Bear_Alpha%', ascending=False).head(5)[['Industry', 'Bear_Alpha%', 'Bull_Alpha%']].to_string(index=False))
    
    print(f"\nFiles saved to {OUTPUT_DIR}/granular_industry_analysis.csv")

if __name__ == "__main__":
    run()

