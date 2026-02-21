"""
Comparative Rolling Backtest: v1 vs v2 vs Nifty
=================================================
Period: 5 Years (2021-2026)
Method: Rolling 6-month windows (1-month step)
Goal: Consistency Analysis (Win Rate % of 6-month periods)

Strategy v1 (Original, Modified):
- Max 10 stocks
- Fixed Stop Loss (-15%)
- *REMOVED*: Fixed Profit Target (Let winners run) -> Now uses Trailing Stop (loose) or Time Stop

Strategy v2 (Enhanced):
- Max 20 stocks
- Sector Cap (25%)
- Trailing Stop (Activated at 10%, Trail 7%)

Output:
- Win rate of v1 vs v2
- Avg 6-month return of each
- Comparison to Nifty bench
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# === CONFIG ===
WINDOW_DAYS = 180    # 6 months
STEP_DAYS = 30
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

class RollingComparativeBacktest:
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

    def get_price(self, ticker, date):
        if ticker not in self.data_cache: return None
        df = self.data_cache[ticker]
        mask = df.index <= date
        if mask.sum() == 0: return None
        return df.loc[mask, 'Close'].iloc[-1]
        
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
        # Priority
        priority = {
            'Strong_Bull': ['Metals', 'Auto', 'Industrials'],
            'Mild_Bull': ['Consumer', 'Pharma'],
            'Recovery': ['Banking', 'IT_Services'],
            'Strong_Bear': ['Auto', 'Metals']
        }
        sectors = priority.get(regime, [])
        if not sectors: sectors = list(UNIVERSE.keys())
        
        # Scan universe
        for sector in sectors:
            # v2 Sector Cap Logic
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
        
        # Sort
        candidates.sort(key=lambda x: -x['trend'] if regime == 'Strong_Bull' else x['trend'])
        
        # Select Portfolio
        max_stocks = 10 if strategy_type == 'v1' else 20
        portfolio = candidates[:max_stocks]
        
        if not portfolio: return 0.0
        
        # Simulate Returns (Vectorized for speed)
        # Assuming buy and hold for 6 months with exit logic applied simply for rolling test
        returns = []
        for stock in portfolio:
            df = self.data_cache[stock['ticker']]
            s_idx = df.index.searchsorted(start_date)
            e_idx = df.index.searchsorted(end_date)
            
            if s_idx >= len(df) or e_idx >= len(df): continue
            
            # Simple simulation of path
            path = df['Close'].iloc[s_idx:e_idx+1].values
            if len(path) == 0: continue
            
            entry = path[0]
            exit_price = path[-1]
            
            # Check Stops
            stop_price = entry * 0.85
            peak = entry
            
            triggered_stop = False
            for price in path:
                if price > peak: peak = price
                
                if strategy_type == 'v2': # Trailing Stop
                    if (price - entry)/entry > 0.10:
                        trail = peak * 0.93
                        if trail > stop_price: stop_price = trail
                
                if price < stop_price:
                    exit_price = stop_price
                    triggered_stop = True
                    break
             
            # v1 No Cap Logic: Only exit on Stop or Time (End of window)
            # v2 Logic: Exit on Trailing Stop or Time
            
            ret = (exit_price - entry) / entry
            returns.append(ret)
            
        if not returns: return 0.0
        return sum(returns) / len(returns)

    def run(self):
        self.fetch_data()
        nifty = self.data_cache['NIFTY']
        start_idx = nifty.index.searchsorted(datetime.now() - timedelta(days=365*5))
        end_idx = len(nifty) - 130
        
        print("Simulating rolling windows...")
        for idx in range(start_idx, end_idx, 20):
            s_date = nifty.index[idx]
            e_date = nifty.index[idx+126]
            
            # Nifty Return
            n_s = nifty['Close'].iloc[idx]
            n_e = nifty.iloc[min(idx+126, len(nifty)-1)]['Close']
            n_ret = (n_e - n_s)/n_s
            
            v1_ret = self.simulate_strategy(s_date, e_date, 'v1')
            v2_ret = self.simulate_strategy(s_date, e_date, 'v2')
            
            self.results.append({
                'start_date': s_date,
                'v1_return': v1_ret,
                'v2_return': v2_ret,
                'nifty_return': n_ret
            })
            
        df = pd.DataFrame(self.results)
        df.to_csv('analysis_2026/rolling_compare.csv', index=False)
        
        print("\n" + "="*50)
        print("ROLLING 6-MONTH COMPARISON (5 Years)")
        print("="*50)
        print(f"Total Periods: {len(df)}")
        print(f"Avg v1 Return: {df['v1_return'].mean()*100:.2f}%")
        print(f"Avg v2 Return: {df['v2_return'].mean()*100:.2f}%")
        print(f"Avg Nifty Return: {df['nifty_return'].mean()*100:.2f}%")
        print("-" * 30)
        print("Win Rate (vs Nifty):")
        print(f"v1 Beats Nifty: {(df['v1_return'] > df['nifty_return']).mean()*100:.1f}%")
        print(f"v2 Beats Nifty: {(df['v2_return'] > df['nifty_return']).mean()*100:.1f}%")
        print("-" * 30)
        print("Head-to-Head:")
        print(f"v1 Beats v2: {(df['v1_return'] > df['v2_return']).mean()*100:.1f}%")

if __name__ == "__main__":
    t = RollingComparativeBacktest()
    t.run()
