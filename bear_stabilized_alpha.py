"""
DNA5 BEAR STRATEGY: STABILIZED ALPHA
====================================
Tests the "Winning DNA" profile found from 10Y Bear Market Analysis against 2025 Market.

Winning Profile (Bear Survivors):
- RSI: 40-55 (Stabilized)
- RS Score (3m): > 0 (Beating Nifty)
- Drawdown: -15% to -35% from 52w High
- Volatility: 30% to 45% (Active but not crazy)

Compare vs VCP Breakout (Previous Winner: -0.6%)
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

class StabilizedAlphaTest:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.data_cache = {}
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
        self.sector_map = SECTOR_MAP
        
    def fetch_data(self):
        s_date = (datetime.strptime(self.start_date, '%Y-%m-%d') - timedelta(days=500)).strftime('%Y-%m-%d')
        nifty = yf.Ticker("^NSEI").history(start=s_date)
        nifty.index = nifty.index.tz_localize(None)
        self.data_cache['NIFTY'] = nifty
        
        loaded = 0
        for t in TICKERS[:500]:
            try:
                df = yf.Ticker(t).history(start=s_date)
                if not df.empty and len(df) > 200:
                    df.index = df.index.tz_localize(None)
                    self.data_cache[t] = df
                    loaded += 1
            except: pass
        return loaded

    def get_price(self, ticker, date):
        if ticker not in self.data_cache: return None
        df = self.data_cache[ticker]
        mask = df.index <= date
        if mask.sum() == 0: return None
        return df.loc[mask, 'Close'].iloc[-1]

    def passes_filter(self, ticker, date):
        if ticker not in self.data_cache: return False, {}
        df = self.data_cache[ticker]
        nifty = self.data_cache['NIFTY']
        
        idx = df.index.searchsorted(date)
        if idx < 200: return False, {}
        
        window = df.iloc[max(0, idx-252):idx+1]
        nifty_idx = nifty.index.searchsorted(date)
        nifty_window = nifty.iloc[max(0, nifty_idx-252):nifty_idx+1]
        
        price = window['Close'].iloc[-1]
        
        # 1. RSI (40-55) - The Stabilized Zone
        delta = window['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        rsi = 100 - (100 / (1 + gain / loss)) if loss != 0 else 50
        
        if rsi < 40 or rsi > 55: return False, {}
        
        # 2. RS Score (> 0) - Beating Market
        price_63 = window['Close'].iloc[-63]
        n_price_63 = nifty_window['Close'].iloc[-63]
        rs_stock = (price - price_63) / price_63
        rs_nifty = (nifty_window['Close'].iloc[-1] - n_price_63) / n_price_63
        rs_score = (rs_stock - rs_nifty) * 100
        
        if rs_score < 0: return False, {}
        
        # 3. Drawdown (-15% to -35%) - Correction not Crash
        high_52 = window['Close'].max()
        dd = (price - high_52) / high_52 * 100
        
        if dd > -15 or dd < -35: return False, {}
        
        # 4. Volatility (30-45%) - Active Range
        rets = window['Close'].pct_change().dropna()[-60:]
        vol = rets.std() * np.sqrt(252) * 100
        
        if vol < 30 or vol > 45: return False, {}
        
        return True, {'rsi': rsi, 'rs_score': rs_score, 'vol': vol, 'dd': dd}

    def run(self):
        nifty = self.data_cache['NIFTY']
        s_date_obj = datetime.strptime(self.start_date, '%Y-%m-%d')
        e_date_obj = datetime.strptime(self.end_date, '%Y-%m-%d')
        
        start_idx = nifty.index.searchsorted(s_date_obj)
        end_idx = nifty.index.searchsorted(e_date_obj)
        dates = nifty.index[start_idx:end_idx+1]
        
        print(f"Running Stabilized Alpha Pattern from {self.start_date} to {self.end_date}...")
        
        for date in dates:
            # EXITS
            to_exit = []
            for t, pos in self.positions.items():
                price = self.get_price(t, date)
                if not price: continue
                
                ret = (price - pos['entry']) / pos['entry']
                
                # Exit Logic: Target +15% or Stop -8%
                # Use trailing stop if profitable
                exit_signal = False
                reason = ''
                
                if ret > 0.15: exit_signal = True; reason = 'Target'
                elif ret < -0.08: exit_signal = True; reason = 'Stop'
                
                if price > pos['peak']: pos['peak'] = price
                if price < pos['peak'] * 0.92: exit_signal = True; reason = 'Trail'
                
                if exit_signal:
                    self.capital += pos['shares'] * price * 0.995
                    pnl = ret * 100
                    self.trade_log.append({
                        'ticker': t,
                        'pnl': pnl,
                        'reason': reason,
                        'entry_date': pos['entry_date'],
                        'exit_date': date,
                    })
                    to_exit.append(t)
            
            for t in to_exit: del self.positions[t]
            
            # ENTRIES
            if len(self.positions) < MAX_POSITIONS:
                candidates = []
                for t in self.data_cache:
                    if t == 'NIFTY' or t in self.positions: continue
                    passes, info = self.passes_filter(t, date)
                    if passes: candidates.append({'ticker': t, **info})
                
                # Rank by RS Strength (strongest first)
                candidates.sort(key=lambda x: -x['rs_score'])
                
                selected = self.select_with_sector_cap(candidates)
                
                for c in selected[:MAX_POSITIONS - len(self.positions)]:
                    price = self.get_price(c['ticker'], date)
                    if price:
                        size = self.capital / (MAX_POSITIONS - len(self.positions) + 2)
                        shares = int(size / price)
                        if shares > 0:
                            cost = shares * price * 1.005
                            if self.capital >= cost:
                                self.capital -= cost
                                self.positions[c['ticker']] = {
                                    'entry': price,
                                    'peak': price,
                                    'shares': shares,
                                    'entry_date': date,
                                }
            
            # Portfolio Value
            val = self.capital
            for t, pos in self.positions.items():
                p = self.get_price(t, date)
                if p: val += pos['shares'] * p
            self.history.append({'date': date, 'value': val})
            
        return pd.DataFrame(self.trade_log), pd.DataFrame(self.history)
        
    def select_with_sector_cap(self, candidates):
        selected = []
        sector_count = {}
        for c in candidates:
            sec = self.sector_map.get(c['ticker'], 'Unknown')
            curr = sum(1 for t in self.positions if self.sector_map.get(t, 'Unknown') == sec)
            if sector_count.get(sec, 0) + curr < SECTOR_CAP:
                selected.append(c)
                sector_count[sec] = sector_count.get(sec, 0) + 1
                if len(selected) + len(self.positions) >= MAX_POSITIONS: break
        return selected

def analyze(tf, hf, name):
    if hf.empty: return {}
    start = hf['value'].iloc[0]
    end = hf['value'].iloc[-1]
    ret = (end - start)/start * 100
    
    hf['peak'] = hf['value'].cummax()
    hf['dd'] = (hf['value'] - hf['peak']) / hf['peak'] * 100
    max_dd = hf['dd'].min()
    
    win_rate = (tf['pnl'] > 0).mean() * 100 if not tf.empty else 0
    
    return {
        'Strategy': name,
        'Return': ret,
        'Max DD': max_dd,
        'Win Rate': win_rate,
        'Trades': len(tf)
    }

def main():
    print("Fetching Master Data...")
    loader = StabilizedAlphaTest('2025-01-01', '2026-02-01')
    loader.fetch_data()
    print(f"Loaded {len(loader.data_cache)} stocks.")
    
    # Run 2025 Test
    eng = StabilizedAlphaTest('2025-01-01', '2026-02-01')
    eng.data_cache = loader.data_cache
    tf, hf = eng.run()
    
    res = analyze(tf, hf, "STABILIZED_ALPHA_2025")
    
    print("\n" + "="*80)
    print("STABILIZED ALPHA STRATEGY RESULTS (2025)")
    print("="*80)
    print(f"Return:    {res['Return']:.2f}%")
    print(f"Max DD:    {res['Max DD']:.2f}%")
    print(f"Win Rate:  {res['Win Rate']:.1f}%")
    print(f"Trades:    {res['Trades']}")
    
    # Comparison
    print("\nComparison vs VCP (-0.6%) and Simple RSI (-1.4%) in 2025")
    
    tf.to_csv('analysis_2026/stabilized_alpha_trades.csv', index=False)

if __name__ == "__main__":
    main()
