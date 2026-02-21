"""
Satellite Portfolio Backtest: The Alpha Booster
===============================================
Goal: Generate 5%+ Alpha over Nifty.
Method: 
- 80% Core (v2 Strategy): Safe, Matches Nifty
- 20% Satellite (Aggressive Strategy): Concentrated, High Beta

Satellite Rules:
- Top 3 Stocks only (based on highest Trend)
- No Sector Cap
- Trailing Stop: Wide (15%) to let winners run hard
- Regime: Only Active in 'Strong Bull' or 'Recovery'

Hypothesis: The 20% Satellite will provide the extra kick to beat Nifty.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# === CONFIG ===
CORE_ALLOCATION = 0.80
SATELLITE_ALLOCATION = 0.20
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

class HybridBacktest:
    def __init__(self):
        self.data_cache = {}
        self.history = []
        self.capital = 1000000
        self.core_val = self.capital * CORE_ALLOCATION
        self.sat_val = self.capital * SATELLITE_ALLOCATION
        
        # Positions
        self.core_pos = {}
        self.sat_pos = {}
        
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

    def run(self):
        self.fetch_data()
        nifty = self.data_cache['NIFTY']
        start_idx = nifty.index.searchsorted(datetime.now() - timedelta(days=365*5))
        dates = nifty.index[start_idx:]
        
        print("Running Hybrid Simulation...")
        
        for date in dates:
            # 1. Regime
            idx = nifty.index.searchsorted(date)
            window = nifty.iloc[max(0, idx-200):idx+1]
            p = window['Close'].iloc[-1]
            m50 = window['Close'].rolling(50).mean().iloc[-1]
            m200 = window['Close'].rolling(200).mean().iloc[-1]
            if m50 > m200: regime = "Strong_Bull" if p > m50 else "Mild_Bull"
            else: regime = "Strong_Bear" if p < m50 else "Recovery"
            
            # === CORE LOGIC (v2) ===
            # Simply update value based on v2 performance (proxy)
            # Implemented simplified v2 logic here for speed
            
            # Manage Core Exits
            to_exit_core = []
            for t, pos in self.core_pos.items():
                price = self.get_price(t, date)
                if not price: continue
                if price > pos['peak']: pos['peak'] = price
                if (price - pos['entry'])/pos['entry'] > 0.10: # Trail Activation
                    trail = pos['peak'] * 0.93
                    if trail > pos['stop']: pos['stop'] = trail
                
                if price < pos['stop']:
                    self.core_val += pos['shares'] * price * 0.995
                    to_exit_core.append(t)
            for t in to_exit_core: del self.core_pos[t]
            
            # Manage Core Entries (Max 20)
            if len(self.core_pos) < 20:
                # Scan for v2 candidates (Diversified)
                cands = []
                for sec, ticks in UNIVERSE.items():
                    # Sector Cap check
                    cnt = sum(1 for p in self.core_pos.values() if p['sector'] == sec)
                    if cnt >= 5: continue
                    
                    for t in ticks:
                        if t in self.core_pos: continue
                        trend = self.get_trend(t, date)
                        if (regime == 'Strong_Bull' and trend > 60) or \
                           (regime == 'Mild_Bull' and trend < 30) or \
                           (regime == 'Recovery' and trend > 20):
                               cands.append({'t': t, 's': sec, 'trend': trend})
                
                cands.sort(key=lambda x: -x['trend'])
                for c in cands[:20-len(self.core_pos)]:
                    price = self.get_price(c['t'], date)
                    if price:
                        size = self.core_val / (20 - len(self.core_pos) + 2)
                        shares = int(size / price)
                        if shares > 0:
                            cost = shares * price * 1.005
                            if self.core_val >= cost:
                                self.core_val -= cost
                                self.core_pos[c['t']] = {
                                    'entry': price, 'stop': price*0.85, 'peak': price, 
                                    'shares': shares, 'sector': c['s']
                                }

            # === SATELLITE LOGIC (Aggressive) ===
            # Only active in Strong Bull / Recovery
            if regime in ['Strong_Bull', 'Recovery']:
                # Manage Exits (Wide Trail)
                to_exit_sat = []
                for t, pos in self.sat_pos.items():
                    price = self.get_price(t, date)
                    if not price: continue
                    if price > pos['peak']: pos['peak'] = price
                    
                    trail = pos['peak'] * 0.85 # 15% very loose trail
                    if trail > pos['stop']: pos['stop'] = trail
                    
                    if price < pos['stop']:
                        self.sat_val += pos['shares'] * price * 0.995
                        to_exit_sat.append(t)
                for t in to_exit_sat: del self.sat_pos[t]
                
                # Manage Entries (Max 3 - High Conviction)
                if len(self.sat_pos) < 3:
                     cands = []
                     for sec, ticks in UNIVERSE.items():
                         for t in ticks:
                             if t in self.sat_pos: continue
                             trend = self.get_trend(t, date)
                             if trend > 70: # Super Momentum
                                 cands.append({'t': t, 'trend': trend})
                     cands.sort(key=lambda x: -x['trend'])
                     
                     for c in cands[:3-len(self.sat_pos)]:
                         price = self.get_price(c['t'], date)
                         if price:
                             size = self.sat_val / (3 - len(self.sat_pos) + 1)
                             shares = int(size / price)
                             if shares > 0:
                                 cost = shares * price * 1.005
                                 if self.sat_val >= cost:
                                     self.sat_val -= cost
                                     self.sat_pos[c['t']] = {
                                         'entry': price, 'stop': price*0.85, 'peak': price, 
                                         'shares': shares
                                     }
            else:
                # Close Satellite in Bear/Mild Bull
                for t, pos in self.sat_pos.items():
                    price = self.get_price(t, date)
                    if price: self.sat_val += pos['shares'] * price * 0.995
                self.sat_pos = {}

            # Calculate Total Value
            curr_core = self.core_val
            for t, p in self.core_pos.items():
                curr_core += p['shares'] * self.get_price(t, date)
                
            curr_sat = self.sat_val
            for t, p in self.sat_pos.items():
                curr_sat += p['shares'] * self.get_price(t, date)
                
            self.history.append({'date': date, 'core': curr_core, 'sat': curr_sat, 'total': curr_core + curr_sat})

        # Results
        df = pd.DataFrame(self.history)
        total_ret = (df.iloc[-1]['total'] - 1000000) / 10000
        
        # Nifty Return
        n_s = nifty.loc[df.iloc[0]['date']]['Close']
        n_e = nifty.loc[df.iloc[-1]['date']]['Close']
        n_ret = (n_e - n_s)/n_s * 100
        
        print("\n" + "="*50)
        print("HYBRID PORTFOLIO RESULTS (5 Years)")
        print("="*50)
        print(f"Total Return: {total_ret:.2f}%")
        print(f"Nifty Return: {n_ret:.2f}%")
        print(f"Alpha: {total_ret - n_ret:.2f}%")
        
        # Breakdown
        core_ret = (df.iloc[-1]['core'] - 800000)/8000
        sat_ret = (df.iloc[-1]['sat'] - 200000)/2000
        print(f"Core (80%): {core_ret:.2f}%")
        print(f"Satellite (20%): {sat_ret:.2f}%")

if __name__ == "__main__":
    h = HybridBacktest()
    h.run()
