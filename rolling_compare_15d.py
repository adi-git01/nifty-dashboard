"""
Final Granular Rolling Backtest
===============================
Period: 5 Years (2021-2026)
Window: 180 Days (6 Months)
Step: 15 Days (High Frequency Check)

Strategies:
1. v1 (Aggressive Concentration): 
   - Max 10 Stocks
   - NO Profit Cap (Let winners run until Time Stop or Stop Loss)
   - Fixed Stop -15%

2. v2 (Balanced Diversification):
   - Max 20 Stocks
   - Trailing Stop (10% activation, 7% trail)
   - Sector Cap 25%

3. Nifty 50 Benchmark

Output:
- Consistency of Alpha (How often do we beat Nifty?)
- Average Outperformance
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# === CONFIG ===
WINDOW_DAYS = 180
STEP_DAYS = 15
UNIVERSE = {
    'Consumer': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'TITAN.NS', 'DABUR.NS', 'MARICO.NS', 'TRENT.NS'],
    'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'LUPIN.NS', 'AUROPHARMA.NS'],
    'IT_Services': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTIM.NS', 'COFORGE.NS'],
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'INDUSINDBK.NS'],
    'Metals': ['TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 'VEDL.NS', 'JINDALSTEL.NS'],
    'Auto': ['MARUTI.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS'],
    'Industrials': ['LT.NS', 'SIEMENS.NS', 'ABB.NS', 'HAVELLS.NS', 'CUMMINSIND.NS'],
    'Energy': ['RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'IOC.NS', 'NTPC.NS', 'POWERGRID.NS']
}

class GranularRollingBacktest:
    def __init__(self):
        self.data_cache = {}
        self.results = []
        
    def fetch_data(self):
        print("Fetching data...")
        start_date = (datetime.now() - timedelta(days=365*5 + 200)).strftime('%Y-%m-%d')
        # Nifty
        nifty = yf.Ticker("^NSEI").history(start=start_date)
        nifty.index = nifty.index.tz_localize(None)
        self.data_cache['NIFTY'] = nifty
        # Stocks
        for t in [x for s in UNIVERSE.values() for x in s]:
            try:
                df = yf.Ticker(t).history(start=start_date)
                df.index = df.index.tz_localize(None)
                self.data_cache[t] = df
            except: pass

    def get_trend(self, ticker, date):
        if ticker not in self.data_cache: return 50
        df = self.data_cache[ticker]
        idx = df.index.searchsorted(date)
        if idx < 200: return 50
        window = df.iloc[max(0, idx-252):idx+1]
        price = window['Close'].iloc[-1]
        ma50 = window['Close'].rolling(50).mean().iloc[-1]
        ma200 = window['Close'].rolling(200).mean().iloc[-1]
        score = 50
        if price>ma50: score+=15
        else: score-=10
        if price>ma200: score+=15
        else: score-=15
        if ma50>ma200: score+=10
        return score

    def simulate_strategy(self, start_date, end_date, strategy_type):
        # 1. Regime
        nifty = self.data_cache['NIFTY']
        idx = nifty.index.searchsorted(start_date)
        if idx < 200: return 0
        window = nifty.iloc[max(0, idx-200):idx+1]
        p = window['Close'].iloc[-1]
        m50 = window['Close'].rolling(50).mean().iloc[-1]
        m200 = window['Close'].rolling(200).mean().iloc[-1]
        if m50 > m200: regime = "Strong_Bull" if p > m50 else "Mild_Bull"
        else: regime = "Strong_Bear" if p < m50 else "Recovery"
        
        # 2. Pick Stocks
        candidates = []
        priority = {
            'Strong_Bull': ['Metals', 'Auto', 'Industrials'],
            'Mild_Bull': ['Consumer', 'Pharma'],
            'Recovery': ['Banking', 'IT_Services'],
            'Strong_Bear': ['Auto', 'Metals']
        }
        sectors = priority.get(regime, [])
        if not sectors: sectors = list(UNIVERSE.keys())
        
        for sector in sectors:
            if strategy_type == 'v2':
                if len([c for c in candidates if c['sector']==sector]) >= 5: continue
                
            for t in UNIVERSE.get(sector, []):
                trend = self.get_trend(t, start_date)
                
                valid = False
                if regime == 'Strong_Bull' and trend > 60: valid = True
                elif regime == 'Mild_Bull' and trend < 30: valid = True
                elif regime == 'Recovery' and trend > 20: valid = True
                elif regime == 'Strong_Bear' and trend < 20: valid = True
                
                if valid:
                    candidates.append({'ticker': t, 'trend': trend, 'sector': sector})
        
        candidates.sort(key=lambda x: -x['trend'] if regime == 'Strong_Bull' else x['trend'])
        max_stocks = 10 if strategy_type == 'v1' else 20
        portfolio = candidates[:max_stocks]
        
        if not portfolio: return 0.0
        
        # Simulate Returns
        returns = []
        for stock in portfolio:
            df = self.data_cache.get(stock['ticker'])
            if df is None: continue
            
            s_idx = df.index.searchsorted(start_date)
            e_idx = df.index.searchsorted(end_date)
            
            if s_idx >= len(df) or e_idx >= len(df): continue
            
            path = df['Close'].iloc[s_idx:e_idx+1].values
            if len(path) == 0: continue
            
            entry = path[0]
            exit_price = path[-1]
            
            stop_price = entry * 0.85 # Fixed stop v1
            
            peak = entry
            triggered_stop = False
            for price in path:
                if price > peak: peak = price
                
                # Exits
                if strategy_type == 'v2': # Trailing
                    if (price - entry)/entry > 0.10:
                        trail = peak * 0.93
                        if trail > stop_price: stop_price = trail
                # v1: No trailing, no target, just fixed stop
                
                if price < stop_price:
                    exit_price = stop_price
                    triggered_stop = True
                    break
                    
            ret = (exit_price - entry) / entry
            returns.append(ret)
            
        if not returns: return 0.0
        return sum(returns) / len(returns)

    def run(self):
        self.fetch_data()
        nifty = self.data_cache['NIFTY']
        start_idx = nifty.index.searchsorted(datetime.now() - timedelta(days=365*5))
        end_idx = len(nifty) - 130
        
        print(f"Simulating rolling windows (Step {STEP_DAYS} days)...")
        # Step by 10 trading days (~15 calendar days)
        for idx in range(start_idx, end_idx, 10):
            s_date = nifty.index[idx]
            e_date = nifty.index[idx+126]
            
            n_s = nifty['Close'].iloc[idx]
            n_e = nifty.iloc[min(idx+126, len(nifty)-1)]['Close']
            n_ret = (n_e - n_s)/n_s
            
            v1_ret = self.simulate_strategy(s_date, e_date, 'v1')
            v2_ret = self.simulate_strategy(s_date, e_date, 'v2')
            
            self.results.append({
                'start_date': s_date.strftime('%Y-%m-%d'),
                'v1_return': v1_ret,
                'v2_return': v2_ret,
                'nifty_return': n_ret
            })
            
        df = pd.DataFrame(self.results)
        df.to_csv('analysis_2026/rolling_compare_15d.csv', index=False)
        
        print("\n" + "="*60)
        print("GRANULAR ROLLING COMPARISON (15-Day Steps)")
        print("="*60)
        print(f"Total Periods: {len(df)}")
        print(f"Avg v1 (Uncapped): {df['v1_return'].mean()*100:.2f}%")
        print(f"Avg v2 (Enhanced): {df['v2_return'].mean()*100:.2f}%")
        print(f"Avg Nifty: {df['nifty_return'].mean()*100:.2f}%")
        print("-" * 30)
        print("Consistency (Beat Nifty %):")
        print(f"v1 Beats Nifty: {(df['v1_return'] > df['nifty_return']).mean()*100:.1f}%")
        print(f"v2 Beats Nifty: {(df['v2_return'] > df['nifty_return']).mean()*100:.1f}%")

if __name__ == "__main__":
    t = GranularRollingBacktest()
    t.run()
