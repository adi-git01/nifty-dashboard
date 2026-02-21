"""
SECTOR INTELLIGENCE & BEATEN-DOWN REVERSAL ANALYSIS
=====================================================
Deep-dive into:
1. Sector performance by regime (which sectors win in bear/bull/sideways)
2. Sector seasonal patterns (monthly rotation map)
3. Sector rotation momentum (which sectors lead reversals)
4. Beaten-down "Fallen Angel" reversal analysis
5. Value reversal strategy backtest

Period: 5Y across Nifty 500
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
YEARS = 15


def detect_regime(nifty, date):
    idx = nifty.index.searchsorted(date)
    if idx < 200: return 'UNKNOWN'
    w = nifty.iloc[max(0, idx-252):idx+1]
    if len(w) < 63: return 'UNKNOWN'
    p = w['Close'].iloc[-1]
    ma50 = w['Close'].rolling(50).mean().iloc[-1]
    ma200 = w['Close'].rolling(200).mean().iloc[-1]
    ret = (p - w['Close'].iloc[-63]) / w['Close'].iloc[-63] * 100
    pk = w['Close'].cummax().iloc[-1]
    dd = (p - pk) / pk * 100
    if p > ma50 and ma50 > ma200 and ret > 5: return 'BULL'
    elif p > ma50 and ret > 0: return 'MILD_BULL'
    elif p < ma50 and (ret < -5 or dd < -10): return 'BEAR'
    else: return 'SIDEWAYS'


def fetch_data():
    start = (datetime.now() - timedelta(days=365*YEARS + 500)).strftime('%Y-%m-%d')
    print("[1/3] Fetching Nifty...")
    nifty = yf.Ticker("^NSEI").history(start=start)
    if nifty.empty: return None, {}
    nifty.index = nifty.index.tz_localize(None)
    
    print(f"[2/3] Bulk downloading {len(TICKERS[:500])} stocks...")
    t0 = time.time()
    try:
        bulk = yf.download(TICKERS[:500], start=start, group_by='ticker', threads=True, progress=True, auto_adjust=True)
    except Exception as e:
        print(f"Failed: {e}")
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
    
    print(f"[3/3] Loaded {loaded} stocks in {time.time()-t0:.0f}s")
    return nifty, dc


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nifty, dc = fetch_data()
    if nifty is None: return
    
    now = datetime.now()
    bt_start = now - timedelta(days=int(365.25 * YEARS))
    si = nifty.index.searchsorted(bt_start)
    dates = nifty.index[si:]
    
    # Group stocks by sector
    sector_stocks = defaultdict(list)
    for t in dc:
        if t == 'NIFTY': continue
        sec = SECTOR_MAP.get(t, 'Unknown')
        if sec != 'Unknown':
            sector_stocks[sec].append(t)
    
    # ============================================
    # PRECOMPUTE: Monthly sector returns + regimes
    # ============================================
    print("\nComputing monthly sector returns...")
    
    # Monthly dates (last trading day of each month)
    month_ends = []
    current_month = None
    for d in dates:
        m = d.to_period('M')
        if current_month is not None and m != current_month:
            month_ends.append(prev_d)
        current_month = m
        prev_d = d
    month_ends.append(dates[-1])
    
    # Compute sector returns for each month
    sector_monthly = []
    
    for i in range(1, len(month_ends)):
        d_start = month_ends[i-1]
        d_end = month_ends[i]
        regime = detect_regime(nifty, d_end)
        cal_month = d_end.month
        year = d_end.year
        
        # Nifty return
        ni_s = nifty.index.searchsorted(d_start)
        ni_e = nifty.index.searchsorted(d_end)
        if ni_s >= len(nifty) or ni_e >= len(nifty): continue
        nifty_ret = (nifty.iloc[ni_e]['Close'] / nifty.iloc[ni_s]['Close'] - 1) * 100
        
        for sec, tickers in sector_stocks.items():
            if len(tickers) < 3: continue  # Need min 3 stocks for meaningful sector avg
            
            rets = []
            for t in tickers:
                if t not in dc: continue
                df = dc[t]
                i_s = df.index.searchsorted(d_start)
                i_e = df.index.searchsorted(d_end)
                if i_s >= len(df) or i_e >= len(df) or i_s == i_e: continue
                r = (df.iloc[i_e]['Close'] / df.iloc[i_s]['Close'] - 1) * 100
                if abs(r) < 100:  # Filter crazy outliers
                    rets.append(r)
            
            if len(rets) >= 3:
                sector_monthly.append({
                    'Date': d_end,
                    'Year': year,
                    'Month': cal_month,
                    'Sector': sec,
                    'Avg_Return%': round(np.mean(rets), 2),
                    'Median_Return%': round(np.median(rets), 2),
                    'Best_Stock%': round(max(rets), 2),
                    'Worst_Stock%': round(min(rets), 2),
                    'Stocks': len(rets),
                    'Regime': regime,
                    'Nifty_Return%': round(nifty_ret, 2),
                    'Alpha%': round(np.mean(rets) - nifty_ret, 2),
                })
    
    sm_df = pd.DataFrame(sector_monthly)
    sm_df.to_csv(f"{OUTPUT_DIR}/sector_monthly_returns.csv", index=False)
    
    # ============================================
    # ANALYSIS 1: SECTOR × REGIME PERFORMANCE
    # ============================================
    print(f"\n{'=' * 110}")
    print("ANALYSIS 1: SECTOR PERFORMANCE BY MARKET REGIME")
    print(f"{'=' * 110}")
    
    regime_sector = []
    
    for regime in ['BULL', 'MILD_BULL', 'SIDEWAYS', 'BEAR']:
        subset = sm_df[sm_df['Regime'] == regime]
        if len(subset) == 0: continue
        
        sector_perf = subset.groupby('Sector').agg(
            Avg_Ret=('Avg_Return%', 'mean'),
            Median_Ret=('Median_Return%', 'mean'),
            Alpha=('Alpha%', 'mean'),
            Months=('Date', 'count'),
            Pct_Pos=('Avg_Return%', lambda x: (x > 0).mean() * 100),
        ).reset_index()
        
        # Filter sectors with enough data
        sector_perf = sector_perf[sector_perf['Months'] >= 3].sort_values('Avg_Ret', ascending=False)
        
        print(f"\n  {'_' * 100}")
        print(f"  REGIME: {regime} ({len(subset.groupby('Date'))} months)")
        print(f"  {'_' * 100}")
        print(f"  {'Rank':>4} {'Sector':<40} {'Avg Ret':>8} {'Alpha':>8} {'%Pos':>6}")
        print(f"  {'-'*68}")
        
        for rank, (_, row) in enumerate(sector_perf.head(10).iterrows(), 1):
            emoji = '+' if row['Alpha'] > 0 else '-'
            print(f"  {rank:>4} {row['Sector']:<40} {row['Avg_Ret']:>+7.1f}% {row['Alpha']:>+7.1f}% {row['Pct_Pos']:>5.0f}%")
            regime_sector.append({
                'Regime': regime, 'Rank': rank, 'Sector': row['Sector'],
                'Avg_Return%': round(row['Avg_Ret'], 2), 'Alpha%': round(row['Alpha'], 2),
                'Pct_Positive': round(row['Pct_Pos'], 0),
            })
        
        # Bottom 5 (worst)
        print(f"\n  Bottom 5 (Weakest):")
        for rank, (_, row) in enumerate(sector_perf.tail(5).iterrows(), 1):
            print(f"       {row['Sector']:<40} {row['Avg_Ret']:>+7.1f}% {row['Alpha']:>+7.1f}%")
    
    pd.DataFrame(regime_sector).to_csv(f"{OUTPUT_DIR}/sector_regime_rankings.csv", index=False)
    
    # ============================================
    # ANALYSIS 2: SECTOR SEASONALITY
    # ============================================
    print(f"\n\n{'=' * 110}")
    print("ANALYSIS 2: SECTOR SEASONALITY (Monthly Heatmap)")
    print(f"{'=' * 110}")
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Find top sectors by total alpha
    top_sectors = sm_df.groupby('Sector')['Alpha%'].mean().sort_values(ascending=False).head(15).index.tolist()
    
    seasonal_data = []
    
    print(f"\n  {'Sector':<35}", end='')
    for m in month_names:
        print(f" {m:>5}", end='')
    print(f" {'Best':>5} {'Worst':>5}")
    print(f"  {'-'*105}")
    
    for sec in top_sectors:
        sec_data = sm_df[sm_df['Sector'] == sec]
        row_str = f"  {sec[:34]:<35}"
        monthly_avgs = {}
        
        for m in range(1, 13):
            m_data = sec_data[sec_data['Month'] == m]['Avg_Return%']
            avg = m_data.mean() if len(m_data) > 0 else 0
            monthly_avgs[m] = avg
            
            # Color coding via symbols
            if avg > 5: symbol = '++'
            elif avg > 2: symbol = '+ '
            elif avg > 0: symbol = '. '
            elif avg > -2: symbol = '- '
            else: symbol = '--'
            
            row_str += f" {avg:>+4.0f}%"
            
            seasonal_data.append({
                'Sector': sec, 'Month': m, 'Month_Name': month_names[m-1],
                'Avg_Return%': round(avg, 2),
            })
        
        best_m = max(monthly_avgs, key=monthly_avgs.get)
        worst_m = min(monthly_avgs, key=monthly_avgs.get)
        row_str += f" {month_names[best_m-1]:>5} {month_names[worst_m-1]:>5}"
        print(row_str)
    
    pd.DataFrame(seasonal_data).to_csv(f"{OUTPUT_DIR}/sector_seasonality.csv", index=False)
    
    # ============================================
    # ANALYSIS 3: BEATEN-DOWN REVERSAL ("FALLEN ANGEL")
    # ============================================
    print(f"\n\n{'=' * 110}")
    print("ANALYSIS 3: BEATEN-DOWN REVERSAL ANALYSIS ('Fallen Angel' Pattern)")
    print(f"{'=' * 110}")
    print("\n  Looking for stocks that fell 30%+ from highs and then recovered 20%+ from lows...")
    
    fallen_angels = []
    
    for t in dc:
        if t == 'NIFTY': continue
        df = dc[t]
        if len(df) < 252: continue
        
        sec = SECTOR_MAP.get(t, 'Unknown')
        
        # Find all 30%+ drawdown events
        df['Peak'] = df['Close'].cummax()
        df['DD%'] = (df['Close'] - df['Peak']) / df['Peak'] * 100
        
        # Find drawdown troughs (where DD < -30%)
        in_drawdown = False
        dd_start = None
        dd_trough = None
        dd_trough_price = None
        dd_trough_dd = 0
        
        for idx_pos in range(252, len(df)):
            d = df.index[idx_pos]
            dd = df['DD%'].iloc[idx_pos]
            price = df['Close'].iloc[idx_pos]
            
            if d < dates[0]: continue  # Only within our backtest period
            
            if dd < -30 and not in_drawdown:
                in_drawdown = True
                dd_start = d
                dd_trough = d
                dd_trough_price = price
                dd_trough_dd = dd
            
            elif in_drawdown:
                if dd < dd_trough_dd:
                    dd_trough = d
                    dd_trough_price = price
                    dd_trough_dd = dd
                
                # Check for recovery (20%+ from trough)
                if dd_trough_price and dd_trough_price > 0:
                    recovery = (price - dd_trough_price) / dd_trough_price * 100
                    
                    if recovery > 20:
                        # Measure forward returns (30d, 60d, 90d from recovery point)
                        fwd_30 = None
                        fwd_60 = None
                        fwd_90 = None
                        
                        if idx_pos + 21 < len(df):
                            fwd_30 = (df['Close'].iloc[idx_pos + 21] / price - 1) * 100
                        if idx_pos + 42 < len(df):
                            fwd_60 = (df['Close'].iloc[idx_pos + 42] / price - 1) * 100
                        if idx_pos + 63 < len(df):
                            fwd_90 = (df['Close'].iloc[idx_pos + 63] / price - 1) * 100
                        
                        regime = detect_regime(nifty, d)
                        
                        # RSI at recovery point
                        window = df.iloc[max(0, idx_pos-50):idx_pos+1]
                        delta = window['Close'].diff()
                        gain = delta.where(delta > 0, 0).rolling(14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                        rsi = 100 - (100/(1+gain.iloc[-1]/loss.iloc[-1])) if loss.iloc[-1] != 0 else 50
                        
                        fallen_angels.append({
                            'Ticker': t,
                            'Sector': sec,
                            'DD_Start': dd_start.strftime('%Y-%m-%d'),
                            'Trough_Date': dd_trough.strftime('%Y-%m-%d'),
                            'Recovery_Date': d.strftime('%Y-%m-%d'),
                            'Max_DD%': round(dd_trough_dd, 1),
                            'Recovery%': round(recovery, 1),
                            'RSI_at_Recovery': round(rsi, 0),
                            'Fwd_30d%': round(fwd_30, 1) if fwd_30 is not None else None,
                            'Fwd_60d%': round(fwd_60, 1) if fwd_60 is not None else None,
                            'Fwd_90d%': round(fwd_90, 1) if fwd_90 is not None else None,
                            'Regime': regime,
                            'DD_to_Recovery_Days': (d - dd_trough).days,
                        })
                        
                        in_drawdown = False
                        dd_trough_dd = 0
            
            # Reset if back near highs
            if dd > -10:
                in_drawdown = False
                dd_trough_dd = 0
    
    fa_df = pd.DataFrame(fallen_angels)
    fa_df.to_csv(f"{OUTPUT_DIR}/fallen_angels.csv", index=False)
    
    print(f"\n  Found {len(fa_df)} Fallen Angel events across {fa_df['Ticker'].nunique()} stocks")
    
    if len(fa_df) > 0:
        # Overall stats
        has_fwd = fa_df.dropna(subset=['Fwd_30d%'])
        
        print(f"\n  FORWARD RETURNS AFTER RECOVERY:")
        print(f"  {'Horizon':<12} {'Median':>8} {'Mean':>8} {'%Pos':>6} {'Best':>8} {'Worst':>8}")
        print(f"  {'-'*55}")
        for col, label in [('Fwd_30d%', '30-Day'), ('Fwd_60d%', '60-Day'), ('Fwd_90d%', '90-Day')]:
            subset = fa_df[col].dropna()
            if len(subset) > 0:
                print(f"  {label:<12} {subset.median():>+7.1f}% {subset.mean():>+7.1f}% {(subset>0).mean()*100:>5.0f}% {subset.max():>+7.1f}% {subset.min():>+7.1f}%")
        
        # By regime
        print(f"\n  FALLEN ANGEL FORWARD 60D RETURNS BY REGIME:")
        for regime in ['BULL', 'MILD_BULL', 'SIDEWAYS', 'BEAR']:
            subset = fa_df[(fa_df['Regime'] == regime) & fa_df['Fwd_60d%'].notna()]['Fwd_60d%']
            if len(subset) >= 3:
                print(f"    {regime:<12}: median {subset.median():>+7.1f}%, mean {subset.mean():>+7.1f}%, {(subset>0).mean()*100:.0f}% positive ({len(subset)} events)")
        
        # By sector
        print(f"\n  TOP SECTORS FOR FALLEN ANGEL TRADES (by median 60d fwd return):")
        sec_fa = fa_df.dropna(subset=['Fwd_60d%']).groupby('Sector').agg(
            Events=('Ticker', 'count'),
            Med_Fwd60=('Fwd_60d%', 'median'),
            Mean_Fwd60=('Fwd_60d%', 'mean'),
            Pct_Pos=('Fwd_60d%', lambda x: (x > 0).mean() * 100),
        ).reset_index()
        sec_fa = sec_fa[sec_fa['Events'] >= 3].sort_values('Med_Fwd60', ascending=False)
        
        print(f"  {'Sector':<40} {'Events':>6} {'Med 60d':>8} {'Mean 60d':>9} {'%Pos':>6}")
        print(f"  {'-'*72}")
        for _, row in sec_fa.head(15).iterrows():
            print(f"  {row['Sector'][:39]:<40} {row['Events']:>5} {row['Med_Fwd60']:>+7.1f}% {row['Mean_Fwd60']:>+8.1f}% {row['Pct_Pos']:>5.0f}%")
        
        # Worst sectors for fallen angels (avoid!)
        print(f"\n  WORST SECTORS (Fallen Angels that stay fallen):")
        for _, row in sec_fa.tail(5).iterrows():
            if row['Med_Fwd60'] < 0:
                print(f"  {row['Sector'][:39]:<40} {row['Events']:>5} {row['Med_Fwd60']:>+7.1f}% {row['Pct_Pos']:>5.0f}%")
        
        # Drawdown depth analysis
        print(f"\n  DOES DEEPER DRAWDOWN = BIGGER RECOVERY?")
        for dd_bucket, lo, hi in [('30-40% DD', -40, -30), ('40-50% DD', -50, -40), ('50%+ DD', -100, -50)]:
            subset = fa_df[(fa_df['Max_DD%'] >= hi) & (fa_df['Max_DD%'] < lo) & fa_df['Fwd_60d%'].notna()]['Fwd_60d%']
            if len(subset) >= 3:
                print(f"    {dd_bucket:<12}: median {subset.median():>+7.1f}%, mean {subset.mean():>+7.1f}%, {(subset>0).mean()*100:.0f}% pos ({len(subset)} events)")
        
        # RSI at recovery point
        print(f"\n  RSI AT RECOVERY POINT vs FORWARD RETURN:")
        for rsi_lo, rsi_hi, label in [(30, 45, 'RSI 30-45'), (45, 55, 'RSI 45-55'), (55, 70, 'RSI 55-70'), (70, 100, 'RSI 70+')]:
            subset = fa_df[(fa_df['RSI_at_Recovery'] >= rsi_lo) & (fa_df['RSI_at_Recovery'] < rsi_hi) & fa_df['Fwd_60d%'].notna()]['Fwd_60d%']
            if len(subset) >= 3:
                print(f"    {label:<12}: median {subset.median():>+7.1f}%, {(subset>0).mean()*100:.0f}% pos ({len(subset)} events)")
    
    # ============================================
    # ANALYSIS 4: SECTOR ROTATION SIGNALS
    # ============================================
    print(f"\n\n{'=' * 110}")
    print("ANALYSIS 4: SECTOR MOMENTUM & MEAN REVERSION SIGNALS")
    print(f"{'=' * 110}")
    
    # For each month, rank sectors by previous-month return
    # Then check: do top-ranked sectors continue (momentum) or reverse (mean reversion)?
    
    sorted_months = sorted(sm_df['Date'].unique())
    
    momentum_hits = 0
    reversion_hits = 0
    total_tests = 0
    
    momentum_data = []
    
    for i in range(1, len(sorted_months)):
        prev_month = sorted_months[i-1]
        curr_month = sorted_months[i]
        
        prev_data = sm_df[sm_df['Date'] == prev_month][['Sector', 'Avg_Return%']].rename(columns={'Avg_Return%': 'Prev_Ret'})
        curr_data = sm_df[sm_df['Date'] == curr_month][['Sector', 'Avg_Return%']].rename(columns={'Avg_Return%': 'Curr_Ret'})
        
        merged = prev_data.merge(curr_data, on='Sector')
        if len(merged) < 5: continue
        
        # Top 5 from last month
        top5 = merged.nlargest(5, 'Prev_Ret')
        bot5 = merged.nsmallest(5, 'Prev_Ret')
        
        top5_cont = top5['Curr_Ret'].mean()
        bot5_cont = bot5['Curr_Ret'].mean()
        
        momentum_data.append({
            'Date': curr_month,
            'Top5_Prev_Avg': round(top5['Prev_Ret'].mean(), 2),
            'Top5_Curr_Avg': round(top5_cont, 2),
            'Bot5_Prev_Avg': round(bot5['Prev_Ret'].mean(), 2),
            'Bot5_Curr_Avg': round(bot5_cont, 2),
            'Momentum_Works': top5_cont > bot5_cont,
        })
        
        if top5_cont > bot5_cont:
            momentum_hits += 1
        else:
            reversion_hits += 1
        total_tests += 1
    
    mom_rate = momentum_hits / total_tests * 100 if total_tests > 0 else 0
    rev_rate = reversion_hits / total_tests * 100 if total_tests > 0 else 0
    
    print(f"\n  SECTOR MOMENTUM vs MEAN REVERSION TEST:")
    print(f"  'Does buying last month's top 5 sectors beat buying last month's bottom 5?'")
    print(f"\n    Months Tested  : {total_tests}")
    print(f"    Momentum Wins  : {momentum_hits} ({mom_rate:.0f}%)")
    print(f"    Reversion Wins : {reversion_hits} ({rev_rate:.0f}%)")
    print(f"    Verdict        : {'MOMENTUM' if mom_rate > 55 else 'MEAN REVERSION' if rev_rate > 55 else 'MIXED'}")
    
    mom_df = pd.DataFrame(momentum_data)
    if len(mom_df) > 0:
        top5_avg = mom_df['Top5_Curr_Avg'].mean()
        bot5_avg = mom_df['Bot5_Curr_Avg'].mean()
        print(f"\n    Avg next-month return of Top 5 sectors: {top5_avg:+.2f}%")
        print(f"    Avg next-month return of Bottom 5 sectors: {bot5_avg:+.2f}%")
        print(f"    Spread: {top5_avg - bot5_avg:+.2f}%/month = {(top5_avg - bot5_avg)*12:+.1f}%/year")
    
    # By regime
    print(f"\n  MOMENTUM vs REVERSION BY REGIME:")
    for regime in ['BULL', 'MILD_BULL', 'SIDEWAYS', 'BEAR']:
        regime_months = sm_df[sm_df['Regime'] == regime]['Date'].unique()
        regime_mom = mom_df[mom_df['Date'].isin(regime_months)]
        if len(regime_mom) >= 3:
            pct = regime_mom['Momentum_Works'].mean() * 100
            label = 'MOMENTUM' if pct > 55 else 'REVERSION' if pct < 45 else 'MIXED'
            print(f"    {regime:<12}: Momentum wins {pct:.0f}% of the time -> {label}")
    
    mom_df.to_csv(f"{OUTPUT_DIR}/sector_momentum_test.csv", index=False)
    
    # ============================================
    # ANALYSIS 5: CURRENT SNAPSHOT
    # ============================================
    print(f"\n\n{'=' * 110}")
    print("ANALYSIS 5: CURRENT SECTOR SNAPSHOT (Latest Month)")
    print(f"{'=' * 110}")
    
    latest_month = sm_df['Date'].max()
    latest = sm_df[sm_df['Date'] == latest_month].sort_values('Avg_Return%', ascending=False)
    regime_now = detect_regime(nifty, nifty.index[-1])
    
    print(f"\n  Current Regime: {regime_now}")
    print(f"  Month: {latest_month.strftime('%Y-%m')}")
    print(f"\n  {'Rank':>4} {'Sector':<40} {'Return':>8} {'Alpha':>8}")
    print(f"  {'-'*62}")
    
    for rank, (_, row) in enumerate(latest.iterrows(), 1):
        marker = '<-- LEADER' if rank <= 3 else ('<-- LAGGARD' if rank >= len(latest) - 2 else '')
        print(f"  {rank:>4} {row['Sector'][:39]:<40} {row['Avg_Return%']:>+7.1f}% {row['Alpha%']:>+7.1f}% {marker}")
    
    # ============================================
    # FINAL SUMMARY
    # ============================================
    print(f"\n\n{'=' * 110}")
    print("KEY INSIGHTS SUMMARY")
    print(f"{'=' * 110}")
    
    # Find best sectors for each regime
    print(f"\n  SECTOR PLAYBOOK BY REGIME:")
    for regime in ['BULL', 'MILD_BULL', 'SIDEWAYS', 'BEAR']:
        subset = sm_df[sm_df['Regime'] == regime]
        if len(subset) == 0: continue
        top = subset.groupby('Sector')['Alpha%'].mean().sort_values(ascending=False).head(3)
        sectors = ', '.join([f"{s[:25]}({v:+.1f}%)" for s, v in top.items()])
        print(f"    {regime:<12}: {sectors}")
    
    print(f"\n  Files saved:")
    for f in ['sector_monthly_returns.csv', 'sector_regime_rankings.csv', 
              'sector_seasonality.csv', 'fallen_angels.csv', 'sector_momentum_test.csv']:
        print(f"    {OUTPUT_DIR}/{f}")


if __name__ == "__main__":
    print("=" * 110)
    print("SECTOR INTELLIGENCE & BEATEN-DOWN REVERSAL ANALYSIS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 110)
    run()
