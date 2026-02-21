"""
DEEP SECTOR SEASONALITY ANALYSIS (10Y & 15Y)
==============================================
For every sector:
 1. Monthly returns: avg, median, stdev, win%, best/worst year
 2. Quarterly patterns (Q1-Q4 returns)
 3. Best 2-month entry windows (best consecutive months)
 4. Year-over-year consistency per month
 5. Seasonal alpha vs Nifty
 6. "Strong months" scoring per sector
 7. Festival/Budget/Earnings season effects
 8. Sector ranking by seasonal alpha

Period: 10Y and 15Y side-by-side for robustness check
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os
import sys
import time
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nifty500_list import TICKERS, SECTOR_MAP

warnings.filterwarnings('ignore')

OUTPUT_DIR = "analysis_2026"
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
QUARTER_NAMES = ['Q1 (Jan-Mar)', 'Q2 (Apr-Jun)', 'Q3 (Jul-Sep)', 'Q4 (Oct-Dec)']


def fetch_data(years):
    start = (datetime.now() - timedelta(days=365*years + 500)).strftime('%Y-%m-%d')
    print(f"  Fetching Nifty ({years}Y)...")
    nifty = yf.Ticker("^NSEI").history(start=start)
    if nifty.empty: return None, {}
    nifty.index = nifty.index.tz_localize(None)
    
    print(f"  Bulk downloading {len(TICKERS[:500])} stocks...")
    t0 = time.time()
    try:
        bulk = yf.download(TICKERS[:500], start=start, group_by='ticker', threads=True, progress=True, auto_adjust=True)
    except Exception as e:
        print(f"  Failed: {e}")
        return nifty, {'NIFTY': nifty}
    
    dc = {'NIFTY': nifty}
    loaded = 0
    for t in TICKERS[:500]:
        try:
            if t in bulk.columns.get_level_values(0):
                df = bulk[t].dropna(how='all')
                if not df.empty and len(df) > 200:
                    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                    dc[t] = df
                    loaded += 1
        except: pass
    
    print(f"  Loaded {loaded} stocks in {time.time()-t0:.0f}s")
    return nifty, dc


def compute_sector_monthly(dc, nifty, sector_stocks, start_date):
    """Compute monthly sector returns."""
    dates = nifty.index[nifty.index >= start_date]
    
    # Find month-end dates
    month_ends = []
    prev_d = dates[0]
    for d in dates:
        if d.month != prev_d.month:
            month_ends.append(prev_d)
        prev_d = d
    month_ends.append(dates[-1])
    
    records = []
    for i in range(1, len(month_ends)):
        d0 = month_ends[i-1]
        d1 = month_ends[i]
        
        # Nifty return
        ni0 = nifty.index.searchsorted(d0)
        ni1 = nifty.index.searchsorted(d1)
        if ni0 >= len(nifty) or ni1 >= len(nifty): continue
        n_ret = (nifty.iloc[ni1]['Close'] / nifty.iloc[ni0]['Close'] - 1) * 100
        
        for sec, tickers in sector_stocks.items():
            rets = []
            for t in tickers:
                if t not in dc: continue
                df = dc[t]
                i0 = df.index.searchsorted(d0)
                i1 = df.index.searchsorted(d1)
                if i0 >= len(df) or i1 >= len(df) or i0 == i1: continue
                r = (df.iloc[i1]['Close'] / df.iloc[i0]['Close'] - 1) * 100
                if abs(r) < 100: rets.append(r)
            
            if len(rets) >= 3:
                records.append({
                    'Date': d1, 'Year': d1.year, 'Month': d1.month,
                    'Quarter': (d1.month - 1) // 3 + 1,
                    'Sector': sec,
                    'Ret%': round(np.mean(rets), 2),
                    'Nifty%': round(n_ret, 2),
                    'Alpha%': round(np.mean(rets) - n_ret, 2),
                    'Stocks': len(rets),
                })
    
    return pd.DataFrame(records)


def deep_seasonality(sm_df, label, top_n=20):
    """Run full depth seasonality analysis."""
    
    # Get top sectors by data coverage
    sec_counts = sm_df.groupby('Sector')['Date'].count()
    all_sectors = sec_counts[sec_counts >= 24].index.tolist()  # Need 2+ years
    
    # Sort by average alpha
    sec_alpha = sm_df.groupby('Sector')['Alpha%'].mean().sort_values(ascending=False)
    top_sectors = [s for s in sec_alpha.index if s in all_sectors][:top_n]
    
    print(f"\n{'=' * 120}")
    print(f"DEEP SEASONALITY ({label}) — {len(sm_df.groupby('Date'))} months, {len(all_sectors)} sectors")
    print(f"{'=' * 120}")
    
    all_seasonal = []
    sector_scores = []
    
    # =============================================
    # 1. DETAILED MONTHLY BREAKDOWN PER SECTOR
    # =============================================
    print(f"\n{'_' * 120}")
    print(f"1. MONTHLY RETURN DEEP-DIVE (Top {top_n} sectors by alpha)")
    print(f"{'_' * 120}")
    
    for sec in top_sectors:
        sec_data = sm_df[sm_df['Sector'] == sec]
        n_years = sec_data['Year'].nunique()
        
        print(f"\n  {'=' * 100}")
        print(f"  {sec} ({n_years} years of data)")
        print(f"  {'=' * 100}")
        print(f"  {'Month':<6} {'Avg':>7} {'Med':>7} {'StdD':>7} {'Win%':>6} {'Best':>7} {'Worst':>7} {'Alpha':>7} {'Consist':>8}")
        print(f"  {'-'*68}")
        
        month_scores = {}
        
        for m in range(1, 13):
            m_data = sec_data[sec_data['Month'] == m]['Ret%']
            m_alpha = sec_data[sec_data['Month'] == m]['Alpha%']
            
            if len(m_data) < 2: continue
            
            avg = m_data.mean()
            med = m_data.median()
            std = m_data.std()
            win = (m_data > 0).mean() * 100
            best = m_data.max()
            worst = m_data.min()
            alpha = m_alpha.mean()
            
            # Consistency: % of years return is within 1 stdev of mean
            consist = ((m_data >= avg - std) & (m_data <= avg + std)).mean() * 100
            
            # Score: combines magnitude, win rate, and consistency
            score = avg * (win / 100) * (consist / 100)
            month_scores[m] = {'avg': avg, 'win': win, 'alpha': alpha, 'score': score}
            
            # Flag
            flag = '***' if win >= 70 and avg > 3 else '**' if win >= 60 and avg > 2 else '*' if win >= 55 and avg > 0 else '' 
            flag_neg = '!!!' if win <= 30 else '!!' if win <= 40 else '!' if win <= 45 else ''
            marker = flag or flag_neg
            
            print(f"  {MONTH_NAMES[m-1]:<6} {avg:>+6.1f}% {med:>+6.1f}% {std:>6.1f}% {win:>5.0f}% {best:>+6.1f}% {worst:>+6.1f}% {alpha:>+6.1f}% {consist:>6.0f}% {marker}")
            
            all_seasonal.append({
                'Period': label, 'Sector': sec, 'Month': m,
                'Month_Name': MONTH_NAMES[m-1],
                'Avg%': round(avg, 2), 'Median%': round(med, 2),
                'StdDev%': round(std, 2), 'Win%': round(win, 0),
                'Best%': round(best, 2), 'Worst%': round(worst, 2),
                'Alpha%': round(alpha, 2), 'Consistency%': round(consist, 0),
                'Score': round(score, 3), 'N_Years': len(m_data),
            })
        
        # Best and worst months
        if month_scores:
            best_m = max(month_scores, key=lambda k: month_scores[k]['avg'])
            worst_m = min(month_scores, key=lambda k: month_scores[k]['avg'])
            best_win = max(month_scores, key=lambda k: month_scores[k]['win'])
            
            print(f"\n  Best Month:    {MONTH_NAMES[best_m-1]} (avg {month_scores[best_m]['avg']:+.1f}%, {month_scores[best_m]['win']:.0f}% win)")
            print(f"  Worst Month:   {MONTH_NAMES[worst_m-1]} (avg {month_scores[worst_m]['avg']:+.1f}%, {month_scores[worst_m]['win']:.0f}% win)")
            print(f"  Most Reliable: {MONTH_NAMES[best_win-1]} ({month_scores[best_win]['win']:.0f}% win rate)")
            
            # Strong months count (avg > 2% AND win > 55%)
            strong = sum(1 for m, s in month_scores.items() if s['avg'] > 2 and s['win'] > 55)
            weak = sum(1 for m, s in month_scores.items() if s['avg'] < 0 and s['win'] < 45)
            
            total_seasonal_alpha = sum(s['alpha'] for s in month_scores.values())
            sector_scores.append({
                'Sector': sec, 'Strong_Months': strong, 'Weak_Months': weak,
                'Total_Alpha': round(total_seasonal_alpha, 1),
                'Best_Month': MONTH_NAMES[best_m-1],
                'Worst_Month': MONTH_NAMES[worst_m-1],
                'Best_Avg': round(month_scores[best_m]['avg'], 1),
                'Worst_Avg': round(month_scores[worst_m]['avg'], 1),
            })
    
    # =============================================
    # 2. QUARTERLY PATTERNS
    # =============================================
    print(f"\n\n{'_' * 120}")
    print(f"2. QUARTERLY PERFORMANCE")
    print(f"{'_' * 120}")
    
    print(f"\n  {'Sector':<40} {'Q1(Jan-Mar)':>11} {'Q2(Apr-Jun)':>11} {'Q3(Jul-Sep)':>11} {'Q4(Oct-Dec)':>11} {'Best Q':>7}")
    print(f"  {'-'*95}")
    
    quarterly = []
    for sec in top_sectors:
        sd = sm_df[sm_df['Sector'] == sec]
        q_avgs = {}
        for q in range(1, 5):
            qd = sd[sd['Quarter'] == q]['Ret%']
            q_avgs[q] = qd.mean() if len(qd) > 0 else 0
        
        best_q = max(q_avgs, key=q_avgs.get)
        print(f"  {sec[:39]:<40} {q_avgs[1]:>+10.1f}% {q_avgs[2]:>+10.1f}% {q_avgs[3]:>+10.1f}% {q_avgs[4]:>+10.1f}% {'Q'+str(best_q):>6}")
        
        for q in range(1, 5):
            quarterly.append({
                'Period': label, 'Sector': sec, 'Quarter': f'Q{q}',
                'Avg_Ret%': round(q_avgs[q], 2),
            })
    
    # =============================================
    # 3. BEST 2-MONTH ENTRY WINDOWS
    # =============================================
    print(f"\n\n{'_' * 120}")
    print(f"3. BEST 2-MONTH ENTRY WINDOWS (Buy at start, hold 2 months)")
    print(f"{'_' * 120}")
    
    window_data = []
    print(f"\n  {'Sector':<35} {'Best Window':<12} {'2M Avg':>7} {'Win%':>6} | {'Worst Window':<12} {'2M Avg':>7}")
    print(f"  {'-'*90}")
    
    for sec in top_sectors:
        sd = sm_df[sm_df['Sector'] == sec]
        best_window = None
        best_ret = -999
        worst_window = None
        worst_ret = 999
        best_win = 0
        
        for m in range(1, 12):  # Jan-Nov (2 month window)
            m1 = sd[sd['Month'] == m]['Ret%']
            m2 = sd[sd['Month'] == m + 1]['Ret%']
            if len(m1) < 3 or len(m2) < 3: continue
            
            # Approximate 2-month return
            avg_2m = m1.mean() + m2.mean()
            win_2m = 0
            # Match by year for proper win rate
            for yr in sd['Year'].unique():
                r1 = sd[(sd['Month'] == m) & (sd['Year'] == yr)]['Ret%']
                r2 = sd[(sd['Month'] == m+1) & (sd['Year'] == yr)]['Ret%']
                if len(r1) > 0 and len(r2) > 0:
                    if r1.iloc[0] + r2.iloc[0] > 0:
                        win_2m += 1
            
            n_yrs = sd['Year'].nunique()
            win_pct = win_2m / n_yrs * 100 if n_yrs > 0 else 0
            
            if avg_2m > best_ret:
                best_ret = avg_2m
                best_window = f"{MONTH_NAMES[m-1]}-{MONTH_NAMES[m]}"
                best_win = win_pct
            if avg_2m < worst_ret:
                worst_ret = avg_2m
                worst_window = f"{MONTH_NAMES[m-1]}-{MONTH_NAMES[m]}"
            
            window_data.append({
                'Period': label, 'Sector': sec,
                'Window': f"{MONTH_NAMES[m-1]}-{MONTH_NAMES[m]}",
                'Avg_2M%': round(avg_2m, 2), 'Win%': round(win_pct, 0),
            })
        
        if best_window:
            print(f"  {sec[:34]:<35} {best_window:<12} {best_ret:>+6.1f}% {best_win:>5.0f}% | {worst_window:<12} {worst_ret:>+6.1f}%")
    
    # =============================================
    # 4. INDIA SEASONAL EVENTS ANALYSIS
    # =============================================
    print(f"\n\n{'_' * 120}")
    print(f"4. INDIA EVENT CALENDAR IMPACT")
    print(f"{'_' * 120}")
    
    # Budget: Feb, Earnings: Jan/Apr/Jul/Oct, Diwali: Oct-Nov, Election years
    events = {
        'Budget Month (Feb)':    {'months': [2]},
        'Q4 Earnings (Apr-May)': {'months': [4, 5]},
        'Monsoon (Jul-Aug)':     {'months': [7, 8]},
        'Diwali Rally (Oct-Nov)':{'months': [10, 11]},
        'Year-End (Dec)':        {'months': [12]},
        'Jan Effect':            {'months': [1]},
    }
    
    print(f"\n  {'Event':<25}", end='')
    for sec in top_sectors[:10]:
        print(f" {sec[:8]:>9}", end='')
    print()
    print(f"  {'-'*115}")
    
    for event, cfg in events.items():
        ms = cfg['months']
        print(f"  {event:<25}", end='')
        for sec in top_sectors[:10]:
            sd = sm_df[(sm_df['Sector'] == sec) & (sm_df['Month'].isin(ms))]
            avg = sd['Ret%'].mean() if len(sd) > 0 else 0
            print(f" {avg:>+8.1f}%", end='')
        print()
    
    # =============================================
    # 5. MONTH-BY-MONTH CONSISTENCY MATRIX
    # =============================================
    print(f"\n\n{'_' * 120}")
    print(f"5. WIN RATE MATRIX (% of years each month is positive)")
    print(f"{'_' * 120}")
    
    print(f"\n  {'Sector':<35}", end='')
    for m in MONTH_NAMES:
        print(f" {m:>5}", end='')
    print(f" {'Strong':>6}")
    print(f"  {'-'*105}")
    
    for sec in top_sectors:
        sd = sm_df[sm_df['Sector'] == sec]
        print(f"  {sec[:34]:<35}", end='')
        strong_count = 0
        for m in range(1, 13):
            md = sd[sd['Month'] == m]['Ret%']
            wr = (md > 0).mean() * 100 if len(md) > 0 else 0
            # Visual indicator
            if wr >= 70: marker = '*'
            elif wr <= 30: marker = '!'
            else: marker = ' '
            print(f" {wr:>4.0f}{marker}", end='')
            if wr >= 60: strong_count += 1
        print(f" {strong_count:>5}/12")
    
    # =============================================
    # 6. SECTOR SEASONAL RANKING
    # =============================================
    print(f"\n\n{'_' * 120}")
    print(f"6. SECTOR SEASONAL ALPHA RANKING")
    print(f"{'_' * 120}")
    
    if sector_scores:
        ss_df = pd.DataFrame(sector_scores).sort_values('Total_Alpha', ascending=False)
        print(f"\n  {'Rank':>4} {'Sector':<40} {'Alpha':>7} {'Strong':>7} {'Weak':>5} {'Best':>6} {'Worst':>6}")
        print(f"  {'-'*78}")
        for rank, (_, row) in enumerate(ss_df.iterrows(), 1):
            print(f"  {rank:>4} {row['Sector'][:39]:<40} {row['Total_Alpha']:>+6.1f}% {row['Strong_Months']:>6}/12 {row['Weak_Months']:>4}/12 {row['Best_Month']:>5}({row['Best_Avg']:+.0f}) {row['Worst_Month']:>5}({row['Worst_Avg']:+.0f})")
    
    # Save all data
    pd.DataFrame(all_seasonal).to_csv(f"{OUTPUT_DIR}/sector_seasonality_deep_{label.lower().replace(' ','_')}.csv", index=False)
    pd.DataFrame(quarterly).to_csv(f"{OUTPUT_DIR}/sector_quarterly_{label.lower().replace(' ','_')}.csv", index=False)
    pd.DataFrame(window_data).to_csv(f"{OUTPUT_DIR}/sector_entry_windows_{label.lower().replace(' ','_')}.csv", index=False)
    if sector_scores:
        pd.DataFrame(sector_scores).to_csv(f"{OUTPUT_DIR}/sector_seasonal_ranking_{label.lower().replace(' ','_')}.csv", index=False)
    
    return all_seasonal, sector_scores


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ============================
    # LOAD DATA ONCE (15Y covers 10Y too)
    # ============================
    print("=" * 120)
    print("LOADING 15Y DATA (will slice for both 10Y and 15Y)")
    print("=" * 120)
    
    nifty, dc = fetch_data(15)
    if nifty is None: return
    
    sector_stocks = defaultdict(list)
    for t in dc:
        if t == 'NIFTY': continue
        sec = SECTOR_MAP.get(t, 'Unknown')
        if sec != 'Unknown':
            sector_stocks[sec].append(t)
    
    now = datetime.now()
    
    # ============================
    # 10Y ANALYSIS
    # ============================
    print(f"\n\n{'#' * 120}")
    print(f"{'#' * 40}   10-YEAR SEASONALITY   {'#' * 40}")
    print(f"{'#' * 120}")
    
    start_10y = now - timedelta(days=int(365.25 * 10))
    sm_10y = compute_sector_monthly(dc, nifty, sector_stocks, start_10y)
    seas_10y, scores_10y = deep_seasonality(sm_10y, "10Y", top_n=20)
    
    # ============================
    # 15Y ANALYSIS
    # ============================
    print(f"\n\n{'#' * 120}")
    print(f"{'#' * 40}   15-YEAR SEASONALITY   {'#' * 40}")
    print(f"{'#' * 120}")
    
    start_15y = now - timedelta(days=int(365.25 * 15))
    sm_15y = compute_sector_monthly(dc, nifty, sector_stocks, start_15y)
    seas_15y, scores_15y = deep_seasonality(sm_15y, "15Y", top_n=20)
    
    # ============================
    # ROBUSTNESS CHECK: 10Y vs 15Y
    # ============================
    print(f"\n\n{'=' * 120}")
    print(f"ROBUSTNESS CHECK: Do seasonal patterns hold across 10Y and 15Y?")
    print(f"{'=' * 120}")
    
    if seas_10y and seas_15y:
        df_10 = pd.DataFrame(seas_10y)
        df_15 = pd.DataFrame(seas_15y)
        
        merged = df_10.merge(df_15, on=['Sector', 'Month'], suffixes=('_10Y', '_15Y'))
        
        if len(merged) > 0:
            # Correlation of monthly returns
            corr = merged['Avg%_10Y'].corr(merged['Avg%_15Y'])
            print(f"\n  Correlation of monthly sector returns (10Y vs 15Y): {corr:.3f}")
            
            # Win rate agreement
            merged['Both_Positive'] = (merged['Avg%_10Y'] > 0) & (merged['Avg%_15Y'] > 0)
            merged['Both_Negative'] = (merged['Avg%_10Y'] < 0) & (merged['Avg%_15Y'] < 0)
            agreement = (merged['Both_Positive'] | merged['Both_Negative']).mean() * 100
            print(f"  Direction agreement (both positive or both negative): {agreement:.0f}%")
            
            # Find robust seasonal edges (positive in BOTH periods and win% > 55 in both)
            robust = merged[(merged['Avg%_10Y'] > 2) & (merged['Avg%_15Y'] > 2) & 
                           (merged['Win%_10Y'] > 55) & (merged['Win%_15Y'] > 55)]
            
            print(f"\n  ROBUST SEASONAL EDGES (>+2% avg AND >55% win in BOTH 10Y and 15Y):")
            print(f"  {'Sector':<40} {'Month':>5} {'10Y Avg':>8} {'10Y Win':>8} {'15Y Avg':>8} {'15Y Win':>8}")
            print(f"  {'-'*80}")
            
            robust = robust.sort_values('Avg%_10Y', ascending=False)
            for _, row in robust.head(30).iterrows():
                mn = MONTH_NAMES[int(row['Month'])-1]
                print(f"  {row['Sector'][:39]:<40} {mn:>5} {row['Avg%_10Y']:>+7.1f}% {row['Win%_10Y']:>6.0f}% {row['Avg%_15Y']:>+7.1f}% {row['Win%_15Y']:>6.0f}%")
            
            robust.to_csv(f"{OUTPUT_DIR}/robust_seasonal_edges.csv", index=False)
            
            # AVOID list (negative in both)
            avoid = merged[(merged['Avg%_10Y'] < -1) & (merged['Avg%_15Y'] < -1) &
                          (merged['Win%_10Y'] < 45) & (merged['Win%_15Y'] < 45)]
            
            if len(avoid) > 0:
                print(f"\n  ROBUST SEASONAL TRAPS (negative in BOTH periods, <45% win):")
                print(f"  {'Sector':<40} {'Month':>5} {'10Y Avg':>8} {'15Y Avg':>8}")
                print(f"  {'-'*58}")
                avoid = avoid.sort_values('Avg%_10Y')
                for _, row in avoid.head(15).iterrows():
                    mn = MONTH_NAMES[int(row['Month'])-1]
                    print(f"  {row['Sector'][:39]:<40} {mn:>5} {row['Avg%_10Y']:>+7.1f}% {row['Avg%_15Y']:>+7.1f}%")
                
                avoid.to_csv(f"{OUTPUT_DIR}/robust_seasonal_traps.csv", index=False)
    
    print(f"\n  All files saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    print("=" * 120)
    print("DEEP SECTOR SEASONALITY ANALYSIS (10Y & 15Y)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 120)
    run()
