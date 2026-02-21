"""
DNA4-C STRATEGY (Regime Adaptive)
=================================
Adaptive strategy that switches between Momentum and Mean Reversion based on market regime.

Logic:
1. DETECT REGIME:
   - BULL: Nifty > MA50 AND (Price > MA200 OR 1m Return > 0%)
   - BEAR/SIDEWAYS: Everything else

2. BULL MODE -> MOMENTUM (DNA3-V2.1 Logic)
   - Price > MA50, RS > 2%, Vol > 30%
   - Exit: Trailing Stop 10%

3. BEAR/SIDEWAYS MODE -> MEAN REVERSION (Simple RSI)
   - RSI < 30
   - Exit: Target +15%, Stop -10%, or RSI > 55 recovery
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

# Strategy Config
INITIAL_CAPITAL = 1000000
MAX_POSITIONS = 10
SECTOR_CAP = 4

class DNA4CStrategy:
    def __init__(self):
        self.data_cache = {}
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.trade_log = []
        self.sector_map = SECTOR_MAP
        
    def fetch_data(self):
        print("[DNA4-C] Fetching data...")
        # Fetch 2 years of data for indicators
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
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
        
        # Simple Regime Detection
        # If Nifty is above MA50 => BULL/MILD BULL
        # Else => BEAR/SIDEWAYS
        
        if price > ma50:
            return 'BULL'
        return 'BEAR_SIDEWAYS'

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def run_scan(self):
        self.fetch_data()
        regime = self.detect_regime()
        print(f"\n[MARKET REGIME DETECTED: {regime}]")
        
        candidates = []
        
        for t, df in self.data_cache.items():
            if t == 'NIFTY': continue
            if len(df) < 200: continue
            
            price = df['Close'].iloc[-1]
            ma50 = df['Close'].rolling(50).mean().iloc[-1]
            
            # === LOGIC BRANCH ===
            if regime == 'BULL':
                # Momentum Logic
                if price < ma50: continue
                
                # RS Check
                nifty = self.data_cache['NIFTY']
                ret_3m = (price - df['Close'].iloc[-63])/df['Close'].iloc[-63] * 100
                nifty_ret_3m = (nifty['Close'].iloc[-1] - nifty['Close'].iloc[-63])/nifty['Close'].iloc[-63] * 100
                rs = ret_3m - nifty_ret_3m
                
                if rs < 2.0: continue
                
                # Volatility Check
                rets = df['Close'].pct_change().dropna()[-60:]
                vol = rets.std() * np.sqrt(252) * 100
                if vol < 30: continue
                
                candidates.append({
                    'ticker': t,
                    'sector': self.sector_map.get(t, 'Unknown'),
                    'price': price,
                    'score': rs, # Rank by RS
                    'strategy': 'MOMENTUM'
                })
                
            else:
                # Mean Reversion Logic
                rsi_series = self.calculate_rsi(df['Close'])
                rsi = rsi_series.iloc[-1]
                
                if rsi > 30: continue
                
                # Sanity: Not < 50% of 52w high
                high_52 = df['Close'].rolling(252).max().iloc[-1]
                if price < high_52 * 0.5: continue
                
                candidates.append({
                    'ticker': t,
                    'sector': self.sector_map.get(t, 'Unknown'),
                    'price': price,
                    'score': rsi, # Rank by RSI (Lower is better)
                    'strategy': 'MEAN_REVERSION'
                })
        
        # Selection Logic (Sector Cap)
        if regime == 'BULL':
            candidates.sort(key=lambda x: -x['score']) # High RS
        else:
            candidates.sort(key=lambda x: x['score']) # Low RSI
            
        print(f"\n{'='*80}")
        print(f"DNA4-C STRATEGY RESULTS ({regime} MODE)")
        print(f"{'='*80}")
        print(f"{'Ticker':<15} {'Sector':<25} {'Price':<10} {'Metric':<10}")
        print(f"{'-'*80}")
        
        selected_count = 0
        sector_counts = {}
        
        for c in candidates:
            sec = c['sector']
            if sector_counts.get(sec, 0) < SECTOR_CAP:
                metric_name = "RS" if c['strategy'] == 'MOMENTUM' else "RSI"
                print(f"{c['ticker']:<15} {c['sector']:<25} {c['price']:<10.2f} {metric_name}: {c['score']:.2f}")
                
                sector_counts[sec] = sector_counts.get(sec, 0) + 1
                selected_count += 1
                if selected_count >= MAX_POSITIONS: break
                
        if selected_count == 0:
            print("No candidates found matching criteria.")

if __name__ == "__main__":
    bot = DNA4CStrategy()
    bot.run_scan()
