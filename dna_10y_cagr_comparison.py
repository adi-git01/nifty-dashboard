"""
DNA 10-YEAR CAGR COMPARISON
===========================
Compares two strategies over the last 10 years (2015-2025):
1. DNA3-V2.1 (Pure Momentum): Always buys leaders if Price > MA50.
2. DNA5-HYBRID (Adaptive):
   - BULL (Nifty > MA50): DNA3-V2.1 (Momentum)
   - BEAR (Nifty < MA50): Volume Surprise (> 3x Avg Vol) + VCP (Defense)

Goal: Determine if the Hybrid approach improves long-term CAGR/Drawdown.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nifty500_list import TICKERS, SECTOR_MAP

warnings.filterwarnings('ignore')

INITIAL_CAPITAL = 1000000
MAX_POSITIONS = 10
SECTOR_CAP = 4

class TenYearComparison:
    def __init__(self, start_date='2015-01-01', end_date='2026-02-01'):
        self.start_date = start_date
        self.end_date = end_date
        self.data_cache = {}
        self.sector_map = SECTOR_MAP
        
    def fetch_data(self):
        print("[10Y] Fetching Master Data (2014-2026)...")
        # Need extra buffer for moving averages
        s_date = (datetime.strptime(self.start_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
        nifty = yf.Ticker("^NSEI").history(start=s_date)
        nifty.index = nifty.index.tz_localize(None)
        self.data_cache['NIFTY'] = nifty
        
        loaded = 0
        limit = 500 # Full Nifty 500
        for t in TICKERS[:limit]:
            try:
                df = yf.Ticker(t).history(start=s_date)
                if not df.empty and len(df) > 500:
                    df.index = df.index.tz_localize(None)
                    self.data_cache[t] = df
                    loaded += 1
            except: pass
        print(f"Loaded {loaded} stocks.")

    def run_backtest(self, strategy_name):
        print(f"\nRunning Backtest: {strategy_name}...")
        capital = INITIAL_CAPITAL
        positions = {}
        history = []
        trade_log = []
        
        nifty = self.data_cache['NIFTY']
        s_date_obj = datetime.strptime(self.start_date, '%Y-%m-%d')
        e_date_obj = datetime.strptime(self.end_date, '%Y-%m-%d')
        
        start_idx = nifty.index.searchsorted(s_date_obj)
        end_idx = nifty.index.searchsorted(e_date_obj)
        dates = nifty.index[start_idx:end_idx+1]
        
        # Simulating weekly rebalance or check
        check_dates = dates[::5] # Every 5th trading day (Weekly)
        
        for date in check_dates:
            # Detect Regime
            nifty_idx = nifty.index.searchsorted(date)
            if nifty_idx < 50: continue
            
            n_price = nifty.iloc[nifty_idx]['Close']
            n_ma50 = nifty.iloc[:nifty_idx+1]['Close'].rolling(50).mean().iloc[-1]
            regime = 'BULL' if n_price > n_ma50 else 'BEAR'
            
            # EXITS
            to_exit = []
            for t, pos in positions.items():
                if t not in self.data_cache: continue
                df = self.data_cache[t]
                idx = df.index.searchsorted(date)
                if idx >= len(df): continue
                price = df.iloc[idx]['Close']
                
                ret = (price - pos['entry']) / pos['entry']
                exit_signal = False
                
                # DNA3 Exit (Trailing Stop)
                if price > pos['peak']: pos['peak'] = price
                if price < pos['peak'] * 0.85: exit_signal = True # 15% Trailing Stop
                
                # Hybrid Bear Exit (Tighter)
                if strategy_name == 'DNA5-HYBRID' and regime == 'BEAR':
                     if ret < -0.07: exit_signal = True # 7% Hard Stop in Bear
                     if ret > 0.15 and price < pos['peak'] * 0.95: exit_signal = True # Take profit fast
                
                if exit_signal:
                    capital += pos['shares'] * price * 0.995 # Slippage
                    trade_log.append({'ticker': t, 'pnl': ret*100, 'date': date})
                    to_exit.append(t)
            
            for t in to_exit: del positions[t]
            
            # ENTRIES
            if len(positions) < MAX_POSITIONS:
                candidates = []
                for t in self.data_cache:
                    if t == 'NIFTY' or t in positions: continue
                    df = self.data_cache[t]
                    idx = df.index.searchsorted(date)
                    if idx < 200 or idx >= len(df): continue
                    
                    window = df.iloc[max(0, idx-252):idx+1]
                    price = window['Close'].iloc[-1]
                    ma50 = window['Close'].rolling(50).mean().iloc[-1]
                    ma200 = window['Close'].rolling(200).mean().iloc[-1]
                    
                    score = -999
                    
                    # === STRATEGY LOGIC ===
                    
                    # 1. DNA3 (PURE MOMENTUM)
                    if strategy_name == 'DNA3-V2.1':
                        if price > ma50: # Only Buy above MA50
                            # RS Score
                            rs = (price / window['Close'].iloc[-63]) / (n_price / nifty.iloc[nifty_idx-63]['Close'])
                            if rs > 1.0: score = rs * 100
                    
                    # 2. DNA5 (HYBRID)
                    elif strategy_name == 'DNA5-HYBRID':
                        if regime == 'BULL':
                            # Same as DNA3
                            if price > ma50:
                                rs = (price / window['Close'].iloc[-63]) / (n_price / nifty.iloc[nifty_idx-63]['Close'])
                                if rs > 1.0: score = rs * 100
                        else:
                            # BEAR MODE: VOLUME SURPRISE ONLY
                            curr_vol = window['Volume'].iloc[-1]
                            avg_vol = window['Volume'].rolling(20).mean().iloc[-1]
                            if curr_vol > avg_vol * 3.0 and price > window['Close'].rolling(20).mean().iloc[-1]:
                                score = 1000 + (curr_vol/avg_vol) # Super Priority
                    
                    if score > -999:
                        candidates.append({'ticker': t, 'score': score, 'price': price})
                
                candidates.sort(key=lambda x: -x['score'])
                
                for c in candidates[:MAX_POSITIONS - len(positions)]:
                    size = capital / (MAX_POSITIONS - len(positions) + 2)
                    shares = int(size / c['price'])
                    if shares > 0:
                        cost = shares * c['price'] * 1.005
                        if capital >= cost:
                            capital -= cost
                            positions[c['ticker']] = {
                                'entry': c['price'], 'peak': c['price'], 'shares': shares, 'entry_date': date
                            }
            
            # Record Value
            val = capital
            for t, pos in positions.items():
                df = self.data_cache[t]
                idx = df.index.searchsorted(date)
                if idx < len(df): val += pos['shares'] * df.iloc[idx]['Close']
            history.append({'date': date, 'value': val})
            
        return pd.DataFrame(history)

def calculate_metrics(df, name):
    if df.empty: return {}
    start_val = df['value'].iloc[0]
    end_val = df['value'].iloc[-1]
    
    # CAGR
    years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
    cagr = (end_val / start_val) ** (1/years) - 1
    
    # Max DD
    df['peak'] = df['value'].cummax()
    df['dd'] = (df['value'] - df['peak']) / df['peak']
    max_dd = df['dd'].min()
    
    # Calmar (CAGR / MaxDD)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    return {
        'Strategy': name,
        'CAGR': f"{cagr*100:.1f}%",
        'Total Return': f"{(end_val/start_val - 1)*100:.1f}%",
        'Max DD': f"{max_dd*100:.1f}%",
        'Calmar': f"{calmar:.2f}",
        'Final Value': f"{end_val:,.0f}"
    }

def main():
    sim = TenYearComparison()
    sim.fetch_data()
    
    h_dna3 = sim.run_backtest('DNA3-V2.1')
    h_dna5 = sim.run_backtest('DNA5-HYBRID')
    
    metrics = []
    metrics.append(calculate_metrics(h_dna3, 'DNA3-V2.1 (Mom Only)'))
    metrics.append(calculate_metrics(h_dna5, 'DNA5 (Hybrid Alpha)'))
    
    print("\n" + "="*80)
    print("10-YEAR STRATEGY COMPARISON (2015-2025)")
    print("="*80)
    res_df = pd.DataFrame(metrics)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()
