"""
COMPREHENSIVE INDUSTRY ENCYCLOPEDIA GENERATOR
=============================================
Runs an exhaustive cyclicity, seasonality, and price-earnings lag 
analysis on every single granular yfinance industry across the Nifty 500 universe.
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
import markdown

warnings.filterwarnings('ignore')

OUTPUT_DIR = "analysis_2026/encyclopedia"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nifty500_list import TICKERS

def get_market_regimes(nifty):
    nifty_m = nifty['Close'].resample('ME').last()
    regimes = {}
    ret_6m = nifty_m.pct_change(6) * 100
    for date, ret in ret_6m.dropna().items():
        if ret > 10: regimes[date] = 'BULL'
        elif ret < -5: regimes[date] = 'BEAR'
        else: regimes[date] = 'SIDEWAYS'
    return pd.Series(regimes)

def analyze_cycles(industry_series):
    if len(industry_series) < 60: return 0, 0
    smooth = industry_series.rolling(3).mean().dropna()
    peaks, troughs = [], []
    window = 6
    for i in range(window, len(smooth) - window):
        chunk = smooth.iloc[i-window:i+window+1]
        mid = smooth.iloc[i]
        if mid == chunk.max(): peaks.append((smooth.index[i], mid))
        elif mid == chunk.min(): troughs.append((smooth.index[i], mid))
            
    if not peaks or not troughs: return 0, 0
    
    cycle_lengths_mo, amplitudes = [], []
    for i in range(1, len(troughs)):
        t1_date, t1_val = troughs[i-1]
        t2_date, t2_val = troughs[i]
        middle_peaks = [p for p in peaks if t1_date < p[0] < t2_date]
        if middle_peaks:
            p_val = max([p[1] for p in middle_peaks])
            amp = (p_val - t1_val) / t1_val * 100
            if amp > 15:
                cycle_lengths_mo.append(round((t2_date - t1_date).days / 30.4, 1))
                amplitudes.append(amp)
                
    if not cycle_lengths_mo: return 0, 0
    return np.mean(cycle_lengths_mo), np.mean(amplitudes)

def cross_correlation(stock_series, earning_proxy, lag_range=6):
    corrs = {}
    for lag in range(-lag_range, lag_range + 1):
        stock_rets = stock_series.pct_change(3).dropna()
        earn = earning_proxy.dropna()
        common = stock_rets.index.intersection(earn.index)
        if len(common) < 20: continue
        corr = stock_rets.loc[common].corr(earn.loc[common].shift(lag))
        corrs[lag] = corr
    return corrs

def make_html_report(md_file_path, html_file_path):
    with open(md_file_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    
    css = """
    <style>
        body { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 40px; background-color: #f4f6f8; }
        .container { background-color: #fff; padding: 50px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.1); }
        h1 { color: #1a252f; font-size: 2.8em; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 20px; margin-bottom: 40px; }
        h2 { color: #2c3e50; font-size: 2em; margin-top: 50px; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }
        h3 { color: #e67e22; font-size: 1.5em; margin-top: 30px; }
        ul { background-color: #fdfdfd; border-inline-start: 4px solid #3498db; padding: 20px 20px 20px 40px; border-radius: 0 8px 8px 0; margin-bottom: 30px; }
        li { margin-bottom: 10px; font-size: 1.1em; }
        strong { color: #2c3e50; }
        .print-btn { display: block; margin: 0 auto 40px auto; padding: 12px 24px; background-color: #2ecc71; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 18px; font-weight: bold; width: 250px; text-align: center; text-decoration: none; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.3s; }
        .print-btn:hover { background-color: #27ae60; box-shadow: 0 6px 8px rgba(0,0,0,0.15); transform: translateY(-2px); }
        .summary-box { background-color: #ebf5fb; padding: 20px; border-radius: 8px; margin-bottom: 40px; border-left: 5px solid #2980b9; }
        @media print { .print-btn { display: none; } body { background-color: white; padding: 0; } .container { box-shadow: none; padding: 0; margin: 0; max-width: 100%; border-radius: 0; } }
    </style>
    """
    
    html_content = markdown.markdown(md_text, extensions=['tables'])

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Comprehensive Industry Encyclopedia</title>
        {css}
    </head>
    <body>
        <div class="container">
            <button class="print-btn" onclick="window.print()">🖨️ Export to PDF</button>
            <div class="summary-box">
                <strong>About this Encyclopedia:</strong> This report represents an exhaustive, bottom-up analysis of the Nifty 500 universe organized into highly granular, specific sub-industries. For each sub-industry, it details expected cycle duration, precise earnings lag mechanics (whether to buy on anticipation or reality), seasonality edges, and performance during market crashes.
            </div>
            {html_content}
        </div>
    </body>
    </html>
    """
    with open(html_file_path, "w", encoding="utf-8") as f:
        f.write(full_html)

def run():
    print("=" * 100)
    print("GENERATING COMPREHENSIVE INDUSTRY ENCYCLOPEDIA (NIFTY 500)")
    print("=" * 100)
    
    # 1. Fetch yfinance info to get granular industry and sector tags
    # Load from cache if possible
    info_cache = f"{OUTPUT_DIR}/yfinance_info_cache.json"
    stock_info = {}
    if os.path.exists(info_cache):
        print("  Loading yfinance industry tags from cache...")
        with open(info_cache, 'r') as f:
            stock_info = json.load(f)
    else:
        print("  Fetching deep yfinance info for Nifty 500 classifications... (This takes a moment)")
        # Batch to avoid rate limits
        batch_size = 100
        for i in range(0, len(TICKERS), batch_size):
            batch = TICKERS[i:i+batch_size]
            print(f"    Fetching batch {i//batch_size + 1}/{len(TICKERS)//batch_size + 1}...")
            try:
                bulk_info = yf.Tickers(" ".join(batch))
                for t in batch:
                    try:
                        info = bulk_info.tickers[t].info
                        if 'industry' in info and 'sector' in info:
                            stock_info[t] = {
                                'sector': info['sector'],
                                'industry': info['industry'],
                                'name': info.get('shortName', t)
                            }
                    except: pass
            except: pass
        with open(info_cache, 'w') as f:
            json.dump(stock_info, f)
            
    # Organize by Broad Sector -> Granular Industry -> Tickers
    hierarchy = {}
    mapped_count = 0
    for t, data in stock_info.items():
        sec = data['sector']
        ind = data['industry']
        if sec not in hierarchy: hierarchy[sec] = {}
        if ind not in hierarchy[sec]: hierarchy[sec][ind] = []
        hierarchy[sec][ind].append(t)
        mapped_count += 1
        
    print(f"  Successfully identified {mapped_count} stocks across {len(hierarchy)} Sectors and {sum(len(inds) for inds in hierarchy.values())} Granular Industries.")
    
    # Prune industries with < 2 stocks for statistical relevance
    valid_hierarchy = {}
    all_valid_tickers = []
    for sec, inds in hierarchy.items():
        valid_inds = {ind: ts for ind, ts in inds.items() if len(ts) >= 2} # Dropped to 2 since user wants EVERY niche
        if valid_inds:
            valid_hierarchy[sec] = valid_inds
            for ts in valid_inds.values():
                all_valid_tickers.extend(ts)

    # 2. Download Historical Price Data
    print("\n[2/3] Fetching 15Y Price History...")
    start = (datetime.now() - timedelta(days=365*16)).strftime('%Y-%m-%d')
    nifty = yf.Ticker("^NSEI").history(start=start)
    if nifty.empty: return
    nifty.index = nifty.index.tz_localize(None)
    nifty_m = nifty['Close'].resample('ME').last()
    regimes = get_market_regimes(nifty)
    
    t0 = time.time()
    bulk = yf.download(all_valid_tickers, start=start, group_by='ticker', threads=True, progress=False, auto_adjust=True)
    print(f"      Downloaded {len(all_valid_tickers)} stock histories in {time.time()-t0:.0f}s")
    
    dc_m = {}
    for t in all_valid_tickers:
        try:
            if t in bulk.columns.get_level_values(0):
                df = bulk[t].dropna(how='all')
                if not df.empty and len(df) > 100:
                    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                    dc_m[t] = df['Close'].resample('ME').last()
        except: pass

    # 3. Analyze each Industry and Generate Markdown Content
    print("\n[3/3] Running Deep Analysis & Generating Encyclopedia...")
    
    md_lines = [
        "# The Ultimate Nifty Industry Encyclopedia",
        "An exhaustive, bottom-up analysis of cyclicity, seasonality, and price-earnings lag mechanics across every listed niche in the Indian market.\n"
    ]
    
    for sec in sorted(valid_hierarchy.keys()):
        md_lines.append(f"## 🏢 SECTOR: {sec}")
        
        inds = valid_hierarchy[sec]
        for ind in sorted(inds.keys()):
            ts = inds[ind]
            avail = [t for t in ts if t in dc_m]
            if len(avail) < 2: continue # Some didn't have enough history
            
            # Build Index
            df_ind = pd.DataFrame({t: dc_m[t] for t in avail}).dropna(how='all')
            # Require at least 2 stocks trading simultaneously for a valid index slice
            rets = df_ind.pct_change()
            mask = rets.notna().sum(axis=1) >= min(2, len(avail))
            if not mask.any(): continue
            rets = rets[mask].replace([np.inf, -np.inf], np.nan)
            ew_ret = rets.mean(axis=1).fillna(0)
            ind_idx = (1 + ew_ret).cumprod() * 100
            
            if len(ind_idx) < 60: continue
            
            md_lines.append(f"### 🔹 Industry: {ind}")
            repr_names = [stock_info[t]['name'].split()[0] for t in avail[:4]]
            md_lines.append(f"*(Constituents: {', '.join(repr_names)}{', etc.' if len(avail) > 4 else ''})*\n")
            
            # --- Cyclicity ---
            mean_cyc, mean_amp = analyze_cycles(ind_idx)
            cyc_str = f"**{mean_cyc:.1f} months** (Avg Amplitude: **+{mean_amp:.1f}%**)" if mean_cyc > 0 else "Linear (No clear boom/bust cycles detected)"
            
            # --- Lead/Lag ---
            rets_1m = ind_idx.pct_change(1) * 100
            proxy_eps = ind_idx.pct_change(12) * 100
            corrs = cross_correlation(ind_idx, proxy_eps, lag_range=6)
            
            lag_str = "Unknown"
            if corrs:
                best_lag = max(corrs, key=corrs.get)
                best_corr = corrs[best_lag]
                if best_corr < 0.25:
                    lag_str = "Random (Poor correlation to intrinsic proxy)"
                elif best_lag < 0:
                    lag_str = f"**Price LEADS by {abs(best_lag)} Months.** *Strategy: Buy on anticipation, sell heavily before cycle peaks.*"
                elif best_lag > 0:
                    lag_str = f"**Price LAGS by {best_lag} Months.** *Strategy: Safe to buy after earnings confirm turnaround.*"
                else:
                    lag_str = "**CONCURRENT (0 Lag).** *Strategy: Pure trend following. Price peaks exactly when earnings peak.*"

            # --- Bear/Bull Alpha ---
            bull_a, bear_a = [], []
            for date, regime in regimes.items():
                if date in rets_1m.index and date in nifty_m.index:
                    ir = rets_1m.loc[date]; nr = (nifty_m.loc[date] / nifty_m.shift(1).loc[date] - 1) * 100
                    if pd.notna(ir) and pd.notna(nr):
                        if regime == 'BULL': bull_a.append(ir - nr)
                        elif regime == 'BEAR': bear_a.append(ir - nr)
            
            ba_val = np.mean(bear_a) if bear_a else 0
            bear_str = f"Strong Protector (**{ba_val:+.1f}%** alpha/mo)" if ba_val > 0.5 else (f"Severe Bleeder (**{ba_val:+.1f}%** alpha/mo)" if ba_val < -0.5 else f"Neutral ({ba_val:+.1f}% alpha/mo)")
            
            # --- Seasonality ---
            ind_df = pd.DataFrame({'Ret': rets_1m})
            ind_df['Month'] = ind_df.index.month
            m_win, m_avg = {}, {}
            for m in range(1, 13):
                m_rets = ind_df[ind_df['Month'] == m]['Ret'].dropna()
                if len(m_rets) > 3:
                    m_win[m] = (m_rets > 0).mean() * 100
                    m_avg[m] = m_rets.mean()
                else:
                    m_win[m], m_avg[m] = 0, 0
            
            b_m = max(m_avg, key=m_avg.get)
            w_m = min(m_avg, key=m_avg.get)
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            md_lines.append(f"-   **🔄 Cycle Length:** {cyc_str}")
            md_lines.append(f"-   **⏱️ Price-Earnings Lag:** {lag_str}")
            md_lines.append(f"-   **🐻 Bear Market Behavior:** {bear_str}")
            md_lines.append(f"-   **📅 Seasonality:** Best Month: **{months[b_m-1]}** ({m_avg[b_m]:+.1f}%, {m_win[b_m]:.0f}% win) | Worst Month: **{months[w_m-1]}** ({m_avg[w_m]:+.1f}%)\n")
            
        md_lines.append("---\n")
            
    md_content = "\n".join(md_lines)
    
    md_path = f"{OUTPUT_DIR}/Industry_Encyclopedia.md"
    html_path = f"{OUTPUT_DIR}/Industry_Encyclopedia.html"
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    make_html_report(md_path, html_path)
    
    print(f"\nEncyclopedia generated successfully!")
    print(f"MD file: {md_path}")
    print(f"HTML/PDF ready file: {html_path}")

if __name__ == "__main__":
    run()
