"""
DNA3 HYBRID BACKTEST (Standard + Turnaround)
============================================
Tests a Hybrid Strategy to solve the "Cash Drag" problem of DNA3-Standard.

Strategies:
1. PURE STANDARD: Invest in Standard DNA3 signals. If < 3 signals, hold Cash.
2. HYBRID: Invest in Standard DNA3. If < 3 Standard signals, fill remaining slots with Turnaround DNA3 (<40 Trend).

Hypothesis: Does filling the "Bear Market Cash" with "Sniper Turnaround Plays" improve CAGR?
Timelines: 1Y, 3Y, 5Y, 10Y.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os

# CONFIG
# Top 70 Liquid Stocks for realistic simulation
STOCKS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", 
          "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "TATAMOTORS.NS", "LT.NS",
          "BAJFINANCE.NS", "MARUTI.NS", "AXISBANK.NS", "SUNPHARMA.NS", "TITAN.NS",
          "ULTRACEMCO.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "TATASTEEL.NS",
          "M&M.NS", "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS",
          "HINDALCO.NS", "BAJAJFINSV.NS", "TECHM.NS", "WIPRO.NS", "HCLTECH.NS",
          "DRREDDY.NS", "CIPLA.NS", "APOLLOHOSP.NS", "DIVISLAB.NS", "GRASIM.NS",
          "HEROMOTOCO.NS", "EICHERMOT.NS", "BRITANNIA.NS", "NESTLEIND.NS", "TATACONSUM.NS",
          "INDUSINDBK.NS", "KOTAKBANK.NS", "SBILIFE.NS", "HDFCLIFE.NS", "BPCL.NS",
          "ASIANPAINT.NS", "UPL.NS", "PGEL.NS", "SUZLON.NS", "ZOMATO.NS", "PAYTM.NS", 
          "POLICYBZR.NS", "DMART.NS", "HAL.NS", "BEL.NS", "VBL.NS", "TRENT.NS", 
          "JIOFIN.NS", "IRFC.NS", "PFC.NS", "RECLTD.NS", "IOC.NS", "VEDL.NS", "GAIL.NS",
          "DLF.NS", "SIEMENS.NS", "ABB.NS", "HAVELLS.NS", "SRF.NS", "PIDILITIND.NS",
          "GODREJCP.NS", "DABUR.NS", "MARICO.NS", "AMBUJACEM.NS", "ACC.NS", "SHREECEM.NS"]

PERIODS = {
    '1Y': ('2025-01-01', '2026-02-01'),
    '3Y': ('2023-01-01', '2026-02-01'),
    '5Y': ('2021-01-01', '2026-02-01'),
    '10Y': ('2016-01-01', '2026-02-01')
}

CAPITAL = 1000000
SLOTS = 10

def calculate_trend_score(price, ma50, ma200, high52, low52):
    if pd.isna(ma200) or pd.isna(high52): return 50
    score = 50
    if price > ma50: score += 15
    else: score -= 10
    if price > ma200: score += 15
    else: score -= 15
    if ma50 > ma200: score += 10
    else: score -= 5
    if (high52 - low52) > 0:
        pos = (price - low52)/(high52 - low52)
        score += int((pos - 0.5) * 30)
    if high52 > 0:
        dd = (price - high52)/high52 * 100
        if dd > -5: score += 10
        elif dd < -30: score -= 10
    return max(0, min(100, score))

def run_portfolio_sim(mode, start_date, end_date):
    """
    mode: 'PURE' (Standard Only) or 'HYBRID' (Standard + Turnaround Filler)
    """
    print(f"  > Simulating {mode} ({start_date} to {end_date})...")
    
    # FETCH DATA
    data_cache = {}
    nifty = yf.Ticker("^NSEI").history(start=start_date, end=end_date)
    if nifty.empty: return None
    nifty['Close'] = nifty['Close'].ffill()
    
    for s in STOCKS:
        try:
            df = yf.Ticker(s).history(start=start_date, end=end_date)
            if not df.empty:
                df['MA50'] = df['Close'].rolling(50).mean()
                df['MA200'] = df['Close'].rolling(200).mean()
                df['High52'] = df['Close'].rolling(252).max()
                df['Low52'] = df['Close'].rolling(252).min()
                data_cache[s] = df
        except: pass
        
    cash = CAPITAL
    holdings = {} # {Ticker: {Qty, EntryPrice, MaxPrice}}
    equity_curve = []
    trades = []
    
    dates = nifty.index
    
    for d in dates:
        # 1. MARK TO MARKET & CHECK EXITS
        current_eq = cash
        
        # EXITS
        to_sell = []
        for t in list(holdings.keys()):
            df = data_cache[t]
            if d not in df.index: 
                current_eq += holdings[t]['Qty'] * holdings[t]['EntryPrice'] # Approx
                continue
            
            row = df.loc[d]
            price = row['Close']
            
            # Update Max Price (Trailing Stop Logic)
            if price > holdings[t]['MaxPrice']: holdings[t]['MaxPrice'] = price
            
            # EXIT RULES (Standard DNA3)
            # 1. Close < MA50
            # 2. 15% Trail from Peak
            ma50 = row['MA50']
            peak = holdings[t]['MaxPrice']
            
            exit_signal = False
            exit_reason = ""
            
            if price < ma50: 
                exit_signal = True
                exit_reason = "Trend Break (< MA50)"
            elif price < peak * 0.85:
                exit_signal = True
                exit_reason = "Trailing Stop (-15%)"
                
            if exit_signal:
                proceeds = holdings[t]['Qty'] * price
                cash += proceeds
                to_sell.append(t)
                ret = (price - holdings[t]['EntryPrice']) / holdings[t]['EntryPrice']
                trades.append({'Ticker': t, 'Entry': holdings[t]['EntryDate'], 'Exit': d, 'Return': ret, 'Reason': exit_reason})
            else:
                current_eq += holdings[t]['Qty'] * price
                
        for t in to_sell: del holdings[t]
        
        equity_curve.append({'Date': d, 'Equity': current_eq})
        
        # 2. CHECK ENTRIES
        # Candidates
        std_candidates = []
        turn_candidates = []
        
        for t in STOCKS:
            if t in holdings or t not in data_cache: continue
            df = data_cache[t]
            if d not in df.index: continue
            row = df.loc[d]
            
            if pd.isna(row['MA50']) or row['Close'] <= row['MA50']: continue
            
            # RS Check
            idx_loc = df.index.get_loc(d)
            if idx_loc < 63: continue
            
            p_today = row['Close']
            p_63 = df['Close'].iloc[idx_loc-63]
            
            if d not in nifty.index: continue
            n_loc = nifty.index.get_loc(d)
            n_today = nifty['Close'].iloc[n_loc]
            n_63 = nifty['Close'].iloc[n_loc-63]
            
            rs_stock = (p_today - p_63)/p_63
            rs_nifty = (n_today - n_63)/n_63
            rs_score = (rs_stock - rs_nifty) * 100
            
            if rs_score <= 0: continue
            
            ts = calculate_trend_score(row['Close'], row['MA50'], row['MA200'], row['High52'], row['Low52'])
            
            # Classify
            # Standard DNA3 considers ALL valid signals regardless of Trend Score
            # But usually prefers stronger trends.
            # Here: Standard = Any Trend Score (Baseline behavior)
            # Turnaround = Trend Score < 40
            
            std_candidates.append({'Ticker': t, 'RS': rs_score, 'Trend': ts, 'Price': row['Close']})
            if ts < 40:
                turn_candidates.append({'Ticker': t, 'RS': rs_score, 'Trend': ts, 'Price': row['Close']})
                
        # Sort by RS
        std_candidates.sort(key=lambda x: -x['RS'])
        turn_candidates.sort(key=lambda x: -x['RS'])
        
        # 3. POSITION SIZING & BUFFER LOGIC
        # Logic: 
        # - Always prioritized filling from Standard Candidates.
        # - If MODE == HYBRID and (Holdings + Std_Candidates) < 3:
        #   - Fill remaining slots with Turnaround Candidates
        
        # How many slots free?
        free_slots = SLOTS - len(holdings)
        
        if free_slots > 0:
            target_pos_size = current_eq / SLOTS
            
            # Buy Standard First
            for cand in std_candidates[:free_slots]:
                if cash < target_pos_size: break
                
                # Buy
                qty = int(target_pos_size / cand['Price'])
                cost = qty * cand['Price']
                cash -= cost
                holdings[cand['Ticker']] = {
                    'Qty': qty, 'EntryPrice': cand['Price'], 'MaxPrice': cand['Price'], 'EntryDate': d, 'Type': 'Standard'
                }
                free_slots -= 1
                
            # Buy Turnaround (Hybrid Only)
            if mode == 'HYBRID' and free_slots > 0:
                # Only if Standard count was low? 
                # User said: "in bear and sideways instead of going cash, use Turnaround"
                # This implies: If we couldn't find enough Standard trades to fill the portfolio (or specific count), use Turnaround.
                # Let's say we want to be fully invested.
                # If Standard didn't fill the slots, use Turnaround.
                
                for cand in turn_candidates:
                    if free_slots == 0 or cash < target_pos_size: break
                    if cand['Ticker'] in holdings: continue # Don't double buy
                    
                    qty = int(target_pos_size / cand['Price'])
                    cost = qty * cand['Price']
                    cash -= cost
                    holdings[cand['Ticker']] = {
                        'Qty': qty, 'EntryPrice': cand['Price'], 'MaxPrice': cand['Price'], 'EntryDate': d, 'Type': 'Turnaround'
                    }
                    free_slots -= 1

    # METRICS
    final_eq = equity_curve[-1]['Equity']
    years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
    cagr = (final_eq / CAPITAL) ** (1/years) - 1
    
    # Win Rate
    if trades:
        wins = [t for t in trades if t['Return'] > 0]
        win_rate = len(wins)/len(trades) * 100
        avg_ret = pd.DataFrame(trades)['Return'].mean() * 100
    else:
        win_rate, avg_ret = 0, 0
        
    # Drawdown
    eq_df = pd.DataFrame(equity_curve)
    eq_df['Peak'] = eq_df['Equity'].cummax()
    eq_df['DD'] = (eq_df['Equity'] - eq_df['Peak']) / eq_df['Peak']
    max_dd = eq_df['DD'].min() * 100
    
    return {
        'Mode': mode,
        'CAGR': cagr * 100,
        'Final Equity': final_eq,
        'Max DD': max_dd,
        'Win Rate': win_rate,
        'Avg Return': avg_ret,
        'Trades': len(trades)
    }

def main():
    results = []
    
    for period, (start, end) in PERIODS.items():
        print(f"\n--- {period} ---")
        m_pure = run_portfolio_sim('PURE', start, end)
        m_hybrid = run_portfolio_sim('HYBRID', start, end)
        
        if m_pure and m_hybrid:
            m_pure['Period'] = period
            m_hybrid['Period'] = period
            results.append(m_pure)
            results.append(m_hybrid)
            
    df = pd.DataFrame(results)
    cols = ['Period', 'Mode', 'CAGR', 'Max DD', 'Win Rate', 'Avg Return', 'Trades']
    print("\n" + "="*80)
    print("HYBRID STRATEGY BACKTEST RESULTS")
    print("="*80)
    print(df[cols].to_string(index=False, float_format="%.1f"))

if __name__ == "__main__":
    main()
