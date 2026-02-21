"""
DNA5 HYBRID STRATEGY: MOMENTUM + VCP
====================================
The Ultimate Adaptive Strategy.

1. REGIME DETECTION:
   - BULL: Nifty > MA50 AND (Price > MA200 OR 1m Return > 0%)
   - BEAR/SIDEWAYS: Everything else

2. BULL MODE -> DNA3-V2.1 (MOMENTUM)
   - Buy Strength: Price > MA50, RS > 2%, Vol > 30%
   - Exit: Trailing Stop 10%

3. BEAR/SIDEWAYS MODE -> VCP BREAKOUT (SNIPER)
   - Buy Contraction + Breakout:
     - Volatility Contraction (ATR/Price < 3.5%)
     - Price near 20-day High (> 98%)
     - Volume Breakout (> 1.5x Avg Vol)
   - Exit: Target +20%, Stop -8%
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

class DNA5Strategy:
    def __init__(self):
        self.data_cache = {}
        self.sector_map = SECTOR_MAP
        
    def fetch_data(self):
        print("[DNA5] Fetching data...")
        start_date = (datetime.now() - timedelta(days=500)).strftime('%Y-%m-%d')
        
        nifty = yf.Ticker("^NSEI").history(start=start_date)
        nifty.index = nifty.index.tz_localize(None)
        self.data_cache['NIFTY'] = nifty
        
        loaded = 0
        for t in TICKERS:
            try:
                df = yf.Ticker(t).history(start=start_date)
                if not df.empty and len(df) > 200:
                    df.index = df.index.tz_localize(None)
                    self.data_cache[t] = df
                    loaded += 1
            except: pass
        print(f"   Loaded {loaded} stocks")

    def detect_regime(self):
        nifty = self.data_cache['NIFTY']
        price = nifty['Close'].iloc[-1]
        ma50 = nifty['Close'].rolling(50).mean().iloc[-1]
        ma200 = nifty['Close'].rolling(200).mean().iloc[-1]
        ret_1m = (price - nifty['Close'].iloc[-21]) / nifty['Close'].iloc[-21]
        
        if price > ma50 and (price > ma200 or ret_1m > 0):
            return 'BULL'
        return 'BEAR_SIDEWAYS'

    def run_scan(self):
        self.fetch_data()
        regime = self.detect_regime()
        print(f"\n[MARKET REGIME: {regime}]")
        
        candidates = []
        
        for t, df in self.data_cache.items():
            if t == 'NIFTY': continue
            if len(df) < 252: continue
            
            price = df['Close'].iloc[-1]
            ma50 = df['Close'].rolling(50).mean().iloc[-1]
            ma200 = df['Close'].rolling(200).mean().iloc[-1]
            
            # === LOGIC BRANCH ===
            if regime == 'BULL':
                # DNA3-V2.1 MOMENTUM
                if price < ma50: continue
                
                # RS Check
                nifty = self.data_cache['NIFTY']
                ret_3m = (price - df['Close'].iloc[-63])/df['Close'].iloc[-63] * 100
                nifty_ret_3m = (nifty['Close'].iloc[-1] - nifty['Close'].iloc[-63])/nifty['Close'].iloc[-63] * 100
                rs = ret_3m - nifty_ret_3m
                
                if rs < 2.0: continue
                
                # Volatility Check
                rets = df['Close'].pct_change().tail(60)
                vol = rets.std() * np.sqrt(252) * 100
                if vol < 30: continue
                
                candidates.append({
                    'ticker': t,
                    'sector': self.sector_map.get(t, 'Unknown'),
                    'price': price,
                    'score': rs, # Rank by RS
                    'strategy': 'MOMENTUM',
                    'setup': f"RS: {rs:.1f}, Vol: {vol:.1f}"
                })
                
            else:
                # BEAR/SIDEWAYS MODE: 2 Strategies (VCP + VOLUME SURPRISE)
                
                # A. VOLUME SURPRISE (Alpha Injection)
                curr_vol = df['Volume'].iloc[-1]
                avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
                high_20 = df['Close'].rolling(20).max().iloc[-1]
                
                if curr_vol > avg_vol * 3.0 and price > df['Close'].rolling(20).mean().iloc[-1]:
                     candidates.append({
                        'ticker': t,
                        'sector': self.sector_map.get(t, 'Unknown'),
                        'price': price,
                        'score': 100, # Priority Score
                        'strategy': 'VOL_SURPRISE',
                        'setup': f"Vol: {curr_vol/avg_vol:.1f}x Avg, Price > MA20"
                    })
                     continue

                # B. VCP BREAKOUT (Sniper Defense)
                # 1. Contraction (Low ATR%)
                tr = df['High'] - df['Low']
                atr = tr.rolling(14).mean().iloc[-1]
                vol_contraction = (atr / price) * 100
                
                if vol_contraction > 3.5: continue # Too volatile
                
                # 2. Near 20-Day High (Breakout zone)
                if price < high_20 * 0.98: continue
                
                # 3. Volume Breakout (Moderate)
                if curr_vol < avg_vol * 1.5: continue
                
                # Rank by RS Strength
                nifty = self.data_cache['NIFTY']
                ret_3m = (price - df['Close'].iloc[-63])/df['Close'].iloc[-63] * 100
                nifty_ret_3m = (nifty['Close'].iloc[-1] - nifty['Close'].iloc[-63])/nifty['Close'].iloc[-63] * 100
                rs = ret_3m - nifty_ret_3m
                
                candidates.append({
                    'ticker': t,
                    'sector': self.sector_map.get(t, 'Unknown'),
                    'price': price,
                    'score': rs, 
                    'strategy': 'VCP_BREAKOUT',
                    'setup': f"VCP: {vol_contraction:.1f}%, VolSpike: {curr_vol/avg_vol:.1f}x"
                })
        
        # Sort and Display
        candidates.sort(key=lambda x: -x['score'])
            
        print(f"\n{'='*90}")
        print(f"DNA5 STRATEGY RESULTS ({regime} MODE)")
        print(f"{'='*90}")
        print(f"{'Ticker':<15} {'Sector':<25} {'Price':<10} {'Strategy':<15} {'Setup Info':<25}")
        print(f"{'-'*90}")
        
        selected_count = 0
        sector_counts = {}
        
        for c in candidates:
            sec = c['sector']
            if sector_counts.get(sec, 0) < SECTOR_CAP:
                print(f"{c['ticker']:<15} {c['sector']:<25} {c['price']:<10.2f} {c['strategy']:<15} {c['setup']:<25}")
                
                sector_counts[sec] = sector_counts.get(sec, 0) + 1
                selected_count += 1
                if selected_count >= MAX_POSITIONS: break
                
        if selected_count == 0:
            print("No candidates found matching criteria.")

if __name__ == "__main__":
    bot = DNA5Strategy()
    bot.run_scan()
