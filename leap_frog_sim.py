"""
Leap & Frog Alpha Simulation (The Dr. Strange Protocol)
=======================================================
Goal: Find the timeline where we win BIG.
Method: Adaptive Regime Switching (The "Smart Switcher")

Strategy Logic:
1. **The LEAP (Bull Mode):**
   - Trigger: Nifty > 200 DMA + Trend Up
   - Portfolio: **V1 Uncapped** (Max 7 Stocks - Super Concentrated)
   - Sector: Aggressive (Metals, Auto, Realty, PSU)
   - Stop: Loose (-15%) to stay in the rocket

2. **The FROG (Bear/Defensive Mode):**
   - Trigger: Nifty < 200 DMA or Trend Down
   - Portfolio: **V2 Defensive** (Max 20 Stocks - Diversified)
   - Sector: Defensive Only (FMCG, Pharma, IT)
   - Stop: Tight Trailing (-5%) to hop away from danger

Simulating over 10 Years (2016-2026) to catch every cycle.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# === CONFIG ===
START_DATE = (datetime.now() - timedelta(days=365*10 + 200)).strftime('%Y-%m-%d')
UNIVERSE = {
    'Aggressive': ['TATASTEEL.NS', 'HINDALCO.NS', 'VEDL.NS', 'DLF.NS', 'MARUTI.NS', 'M&M.NS', 'TVSMOTOR.NS', 'SBIN.NS', 'HAL.NS', 'BEL.NS'],
    'Defensive': ['HINDUNILVR.NS', 'ITC.NS', 'DABUR.NS', 'SUNPHARMA.NS', 'CIPLA.NS', 'TCS.NS', 'INFY.NS', 'BRITANNIA.NS', 'NESTLEIND.NS']
}

class LeapFrogSim:
    def __init__(self):
        self.data_cache = {}
        self.capital = 1000000
        self.positions = {}
        self.history = []
        self.active_mode = "None"
        
    def fetch_data(self):
        print("Fetching 10 years of data...")
        # Nifty
        nifty = yf.Ticker("^NSEI").history(start=START_DATE)
        nifty.index = nifty.index.tz_localize(None)
        self.data_cache['NIFTY'] = nifty
        # Stocks
        all_t = UNIVERSE['Aggressive'] + UNIVERSE['Defensive']
        for t in all_t:
            try:
                df = yf.Ticker(t).history(start=START_DATE)
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
        if not pd.isna(ma50) and price>ma50: score+=15
        else: score-=10
        if not pd.isna(ma200) and price>ma200: score+=15
        else: score-=15
        if not pd.isna(ma50) and not pd.isna(ma200) and ma50>ma200: score+=10
        return score

    def get_market_regime(self, date):
        nifty = self.data_cache['NIFTY']
        idx = nifty.index.searchsorted(date)
        if idx < 200: return "Unknown"
        window = nifty.iloc[max(0, idx-200):idx+1]
        price = window['Close'].iloc[-1]
        ma50 = window['Close'].rolling(50).mean().iloc[-1]
        ma100 = window['Close'].rolling(100).mean().iloc[-1]
        
        if ma50 > ma100 and price > ma50: return "LEAP" # Strong Bull (Faster Signal)
        return "FROG" # Any other state (Weakness/Chop/Bear)

    def run(self):
        self.fetch_data()
        nifty = self.data_cache['NIFTY']
        start_idx = nifty.index.searchsorted(datetime.now() - timedelta(days=365*10))
        dates = nifty.index[start_idx:]
        
        print(f"Running Dr. Strange Simulation (10 Years)...")
        
        for date in dates:
            mode = self.get_market_regime(date)
            
            # MODE SWITCH LOGIC
            if mode != self.active_mode:
                # print(f"*** SWITCH: {self.active_mode} -> {mode} on {date.date()} ***")
                # When switching, we re-balance aggressively
                # Ideally we sell incompatible stocks, but for sim simplicity we'll let exits handle it
                self.active_mode = mode
                
            # === EXITS ===
            to_exit = []
            for t, pos in self.positions.items():
                price = self.get_price(t, date)
                if not price: continue
                
                # Update Peak
                if price > pos['peak']: pos['peak'] = price
                
                # Dynamic Stop based on Mode
                should_exit = False
                
                if mode == "LEAP": # Bull Mode - Wide Stops
                    if price < pos['stop']: should_exit = True
                    # No target, let run
                else: # Frog Mode - Tight Stops
                     # Tight Trailing
                     if (price - pos['entry'])/pos['entry'] > 0.05:
                         trail = pos['peak'] * 0.95 # 5% Trail
                         if trail > pos['stop']: pos['stop'] = trail
                     
                     if price < pos['stop']: should_exit = True
                
                # Force Rotation checking
                # If in Frog mode, dump Aggressive stocks
                if mode == "FROG" and t in UNIVERSE['Aggressive']: should_exit = True
                
                if should_exit:
                    val = pos['shares'] * price * 0.995
                    self.capital += val
                    to_exit.append(t)
            
            for t in to_exit: del self.positions[t]
            
            # === ENTRIES ===
            target_universe = UNIVERSE['Aggressive'] if mode == "LEAP" else UNIVERSE['Defensive']
            max_stocks = 7 if mode == "LEAP" else 15
            
            if len(self.positions) < max_stocks:
                cands = []
                for t in target_universe:
                    if t in self.positions: continue
                    trend = self.get_trend(t, date)
                    
                    if mode == "LEAP" and trend > 60: cands.append({'t': t, 'trend': trend})
                    elif mode == "FROG" and trend > 30: cands.append({'t': t, 'trend': trend})
                
                cands.sort(key=lambda x: -x['trend'])
                
                for c in cands[:max_stocks-len(self.positions)]:
                    price = self.get_price(c['t'], date)
                    if price:
                        size = self.capital / (max_stocks - len(self.positions) + 2)
                        shares = int(size / price)
                        if shares > 0:
                            cost = shares * price * 1.005
                            if self.capital >= cost:
                                self.capital -= cost
                                stop = price * 0.85 if mode == "LEAP" else price * 0.95
                                self.positions[c['t']] = {
                                    'entry': price, 'stop': stop, 'peak': price, 'shares': shares
                                }
                                
            # Log Value
            val = self.capital
            for t, pos in self.positions.items():
                p = self.get_price(t, date)
                if p: val += pos['shares'] * p
            
            self.history.append({'date': date, 'value': val, 'mode': mode})
            
        # Analysis
        df = pd.DataFrame(self.history)
        final_val = df.iloc[-1]['value']
        total_ret = (final_val - 1000000)/10000
        
        # Nifty Benchmark
        n_s = nifty.loc[df.iloc[0]['date']]['Close']
        n_e = nifty.loc[df.iloc[-1]['date']]['Close']
        n_ret = (n_e - n_s)/n_s * 100
        
        print("\n" + "="*50)
        print("LEAP & FROG RESULTS (10 Years)")
        print("="*50)
        print(f"Final Value: Rs.{final_val:,.0f}")
        print(f"Total Return: {total_ret:.2f}%")
        print(f"Nifty Return: {n_ret:.2f}%")
        print(f"Alpha: {total_ret - n_ret:.2f}%")
        print(f"CAGR: {((final_val/1000000)**(1/10) - 1)*100:.2f}%")
        
        df.to_csv('analysis_2026/leap_frog_sim.csv', index=False)

if __name__ == "__main__":
    sim = LeapFrogSim()
    sim.run()
