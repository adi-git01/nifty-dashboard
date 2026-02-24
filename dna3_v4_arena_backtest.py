"""
DNA3-V4 ENGINE vs DNA3-V3.1 ENGINE BACKTEST SUITE
=================================================
Tests whether injecting mathematical edge-cases (VCP Contraction & Day-0 Earnings Gaps)
into the core V3.1 Momentum algorithm creates a superior alpha profile across 15 years.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings
import os
warnings.filterwarnings('ignore')

from utils.nifty500_list import TICKERS

def calculate_v3_base_score(hist, p_idx):
    """Calcs the V3.1 Base Score components."""
    try:
        if p_idx < 65: return 0
        
        c = hist['Close'].iloc[p_idx]
        ma50 = hist['Close'].iloc[p_idx-50:p_idx].mean()
        ma200 = hist['Close'].iloc[p_idx-200:p_idx].mean()
        
        # 1. Structural Filter
        if c < ma50 or ma50 < ma200:
            return 0
            
        # 2. Distance to MA50 (Tension)
        dist_50 = (c - ma50) / ma50 * 100
        dist_score = max(0, 100 - (abs(dist_50 - 5) * 5)) 
        
        # 3. Micro Relative Strength (Quarterly)
        ret_63 = (c - hist['Close'].iloc[max(0, p_idx-63)]) / hist['Close'].iloc[max(0, p_idx-63)] * 100
        rs_score = max(0, min(100, ret_63 * 3))
        
        # Base V3.1 Formula
        base = (rs_score * 0.7) + (dist_score * 0.3)
        return min(100, max(0, base))
    except (IndexError, pd.errors.InvalidIndexError):
        return 0

def detect_vcp_compression(hist, p_idx):
    """Checks if the last 10 days are extremely tight (<3.5% ATR) on drastically contracting volume (<0.5x)."""
    if p_idx < 65: return False
    
    # 10 Day True Range
    recent = hist.iloc[p_idx-10:p_idx+1]
    atr_pct = ((recent['High'] - recent['Low']) / recent['Low']).mean() * 100
    if atr_pct > 3.5: return False
    
    # Volume Dry up
    vol_today = hist['Volume'].iloc[p_idx]
    vol_60d_avg = hist['Volume'].iloc[p_idx-60:p_idx].mean()
    if vol_60d_avg == 0: return False
    
    if (vol_today / vol_60d_avg) > 0.6: return False # Need severe volume dry up indicating supply absorption
    
    return True

def detect_earnings_shock(hist, p_idx):
    """Checks for Day-0 Gap: >5% jump on >300% volume."""
    if p_idx < 25: return False
    
    c = hist['Close'].iloc[p_idx]
    c_prev = hist['Close'].iloc[p_idx-1]
    
    ret_1d = (c - c_prev) / c_prev * 100
    if ret_1d < 5.0: return False
    
    vol_today = hist['Volume'].iloc[p_idx]
    vol_20d_avg = hist['Volume'].iloc[p_idx-20:p_idx].mean()
    if vol_20d_avg == 0: return False
    
    if (vol_today / vol_20d_avg) < 3.0: return False
    
    return True


def simulate_portfolio(tickers, start_date, end_date):
    """
    Runs a bi-weekly portfolio simulation testing pure V3.1 vs the new V4 engine.
    Capital: 10 Lakhs. Max positions: 10. Trailing Stop: 12%.
    """
    print(f"\n[{start_date} -> {end_date}] Initializing Universe Download...")
    sim_start = (datetime.strptime(start_date, '%Y-%m-%d') - relativedelta(days=250)).strftime('%Y-%m-%d')
    data = yf.download(tickers, start=sim_start, end=end_date, group_by='ticker', threads=True, progress=False)

    date_series = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Trackers for both algorithms
    v3_portfolio = {} # ticker -> {'entry': p, 'peak': p}
    v4_portfolio = {}
    
    v3_equity = 1000000
    v4_equity = 1000000
    
    v3_cash = v3_equity
    v4_cash = v4_equity
    
    v3_peak_equity = v3_equity
    v4_peak_equity = v4_equity
    
    v3_max_dd = 0
    v4_max_dd = 0
    
    v3_history = []
    v4_history = []

    days_since_rebalance = 0
    
    print(f"Simulating daily pricing for {len(date_series)} trading days...")

    for d in date_series:
        d_str = d.strftime('%Y-%m-%d')
        
        # 1. Update existing positions & check trailing stops (12%)
        # --- V3 Engine Maintenance ---
        for t in list(v3_portfolio.keys()):
            if t in data and d_str in data[t].index:
                p = data[t].loc[d_str]['Close']
                if pd.isna(p): continue
                # Update peak
                if p > v3_portfolio[t]['peak']: v3_portfolio[t]['peak'] = p
                
                # Check -12% TSL
                if p < v3_portfolio[t]['peak'] * 0.88:
                    shares = v3_portfolio[t]['shares']
                    v3_cash += p * shares
                    del v3_portfolio[t]
        
        # --- V4 Engine Maintenance ---
        for t in list(v4_portfolio.keys()):
            if t in data and d_str in data[t].index:
                p = data[t].loc[d_str]['Close']
                if pd.isna(p): continue
                # Update peak
                if p > v4_portfolio[t]['peak']: v4_portfolio[t]['peak'] = p
                
                # Check -12% TSL
                if p < v4_portfolio[t]['peak'] * 0.88:
                    shares = v4_portfolio[t]['shares']
                    v4_cash += p * shares
                    del v4_portfolio[t]

        # 2. Rebalance (Every 10 days)
        if days_since_rebalance >= 10:
            days_since_rebalance = 0
            
            v3_candidates = []
            v4_candidates = []
            
            for t in tickers:
                if t not in data: continue
                hist = data[t].dropna(how='all')
                if hist.empty: continue
                if d_str not in hist.index: continue
                
                p_idx = hist.index.get_loc(d_str)
                if p_idx < 65: continue
                
                p = hist.iloc[p_idx]['Close']
                if pd.isna(p) or p < 20: continue
                
                # Calc Base V3 Score
                base_score = calculate_v3_base_score(hist, p_idx)
                
                if base_score > 70:
                    v3_candidates.append({'ticker': t, 'score': base_score, 'price': p})
                
                # Calc V4 Score (Fusion)
                v4_score = base_score
                if base_score > 60: # Lower base requirement if it has a massive edge
                    if detect_earnings_shock(hist, p_idx):
                        v4_score = 100 # Override: Ignite ZeroLag
                    elif detect_vcp_compression(hist, p_idx):
                        v4_score = min(100, base_score * 1.25) # Multiplier: Coil
                        
                if v4_score > 70:
                    v4_candidates.append({'ticker': t, 'score': v4_score, 'price': p})
            
            # --- EXECUTE V3.1 PORTFOLIO ---
            v3_candidates = sorted(v3_candidates, key=lambda x: x['score'], reverse=True)
            required_v3 = 10 - len(v3_portfolio)
            if required_v3 > 0 and len(v3_candidates) > 0:
                buy_list = [c for c in v3_candidates if c['ticker'] not in v3_portfolio][:required_v3]
                for b in buy_list:
                    alloc = v3_cash / required_v3 if required_v3 > 0 else 0
                    if alloc > 1000:
                        shares = int(alloc / b['price'])
                        v3_cash -= (shares * b['price'])
                        v3_portfolio[b['ticker']] = {'shares': shares, 'peak': b['price']}
                        
            # --- EXECUTE V4 (FUSION) PORTFOLIO ---
            v4_candidates = sorted(v4_candidates, key=lambda x: x['score'], reverse=True)
            required_v4 = 10 - len(v4_portfolio)
            if required_v4 > 0 and len(v4_candidates) > 0:
                buy_list = [c for c in v4_candidates if c['ticker'] not in v4_portfolio][:required_v4]
                for b in buy_list:
                    alloc = v4_cash / required_v4 if required_v4 > 0 else 0
                    if alloc > 1000:
                        shares = int(alloc / b['price'])
                        v4_cash -= (shares * b['price'])
                        v4_portfolio[b['ticker']] = {'shares': shares, 'peak': b['price']}
                        
        days_since_rebalance += 1

        # Calculate daily equity
        v3_eq = v3_cash
        for t, pos in v3_portfolio.items():
            if t in data and d_str in data[t].index:
                v3_eq += data[t].loc[d_str]['Close'] * pos['shares']
                
        v4_eq = v4_cash
        for t, pos in v4_portfolio.items():
            if t in data and d_str in data[t].index:
                v4_eq += data[t].loc[d_str]['Close'] * pos['shares']
                
        # Drawdowns
        if v3_eq > v3_peak_equity: v3_peak_equity = v3_eq
        v3_dd = (v3_eq - v3_peak_equity) / v3_peak_equity * 100
        if v3_dd < v3_max_dd: v3_max_dd = v3_dd
        
        if v4_eq > v4_peak_equity: v4_peak_equity = v4_eq
        v4_dd = (v4_eq - v4_peak_equity) / v4_peak_equity * 100
        if v4_dd < v4_max_dd: v4_max_dd = v4_dd

        v3_history.append({'date': d, 'equity': v3_eq})
        v4_history.append({'date': d, 'equity': v4_eq})

    # Final Returns Calculations
    days_total = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
    years = max(1, days_total / 365.25)
    
    v3_final = v3_history[-1]['equity']
    v3_cagr = ((v3_final / 1000000) ** (1/years) - 1) * 100
    
    v4_final = v4_history[-1]['equity']
    v4_cagr = ((v4_final / 1000000) ** (1/years) - 1) * 100

    return {
        'horizon': f"{start_date} to {end_date}",
        'years': years,
        'v3_cagr': v3_cagr,
        'v3_dd': v3_max_dd,
        'v4_cagr': v4_cagr,
        'v4_dd': v4_max_dd,
        'v4_vs_v3_alpha': v4_cagr - v3_cagr
    }

def run_multi_horizon_arena():
    os.makedirs("analysis_2026/backtests", exist_ok=True)
    
    today = datetime.now()
    horizons = {
        "6mo (Current Regime)": (today - relativedelta(months=6)).strftime('%Y-%m-%d'),
        "1 Yr (Recent Bull)": (today - relativedelta(years=1)).strftime('%Y-%m-%d'),
        "3 Yr (Post-Covid)": (today - relativedelta(years=3)).strftime('%Y-%m-%d'),
        "5 Yr (Full Cycle)": (today - relativedelta(years=5)).strftime('%Y-%m-%d'),
        "10 Yr (Secular Bull)": (today - relativedelta(years=10)).strftime('%Y-%m-%d'),
        "15 Yr (Deep History)": (today - relativedelta(years=15)).strftime('%Y-%m-%d')
    }
    
    end_date = today.strftime('%Y-%m-%d')
    universe = TICKERS[:200] # Top 200 for speed of testing over 15 years
    
    results = []
    
    print("====================================================================")
    print("INITIATING MULTI-HORIZON COMPARATIVE BACKTEST: DNA3-V4 vs DNA3-V3.1")
    print("Testing if adding VCP Multipliers and Earnings Shock Overrides produces genuine Alpha.")
    print("====================================================================")
    
    for name, start_date in horizons.items():
        print(f"\n⚙️ Running {name}...")
        metrics = simulate_portfolio(universe, start_date, end_date)
        metrics['Label'] = name
        results.append(metrics)
        print(f"  > V3.1 CAGR: {metrics['v3_cagr']:.1f}% | Max DD: {metrics['v3_dd']:.1f}%")
        print(f"  > V4.0 CAGR: {metrics['v4_cagr']:.1f}% | Max DD: {metrics['v4_dd']:.1f}%")
        print(f"  > Alpha Generated by V4 Integration: {metrics['v4_vs_v3_alpha']:+.1f}%")
        
    df_res = pd.DataFrame(results)
    
    # Save results
    df_res[['Label', 'v3_cagr', 'v3_dd', 'v4_cagr', 'v4_dd', 'v4_vs_v3_alpha']].to_csv("analysis_2026/backtests/v4_vs_v3_comparison.csv", index=False)
    print("\nMulti-Horizon Arena Match Complete. Results saved to analysis_2026/backtests/v4_vs_v3_comparison.csv")
    
if __name__ == "__main__":
    run_multi_horizon_arena()
