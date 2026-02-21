"""
BEAR ALPHA HUNTER
=================
Explores 4 distinct strategies to find Alpha in Bear/Sideways markets.
Test Periods: 2022 Bear (Jan-Jun) and 2025 Chop (Jan-Feb).

Strategies:
1. RS_LEADER: Nifty < MA50 but Stock hitting 20-day Highs (Relative Strength Breakout).
2. FORTRESS: Low Beta (<0.7) + Low Volatility + Dividend Yield (Proxy via sector/history).
3. DIP_BUY_REV: Stock in Uptrend (MA50 > MA200) but dips RSI < 30 on daily.
4. VCP_BO: Volatility Contraction Pattern (Low ADX/Range) + Volume Breakout.
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

class BearAlphaHunter:
    def __init__(self, start_date, end_date, strategy='RS_LEADER'):
        self.start_date = start_date
        self.end_date = end_date
        self.strategy = strategy
        self.data_cache = {}
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.trade_log = []
        self.history = []
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

    def calculate_indicators(self, window, nifty_window):
        price = window['Close'].iloc[-1]
        
        # Moving Averages
        ma20 = window['Close'].rolling(20).mean().iloc[-1]
        ma50 = window['Close'].rolling(50).mean().iloc[-1]
        ma200 = window['Close'].rolling(200).mean().iloc[-1]
        
        # Highs/Lows
        high_20 = window['Close'].rolling(20).max().iloc[-1]
        high_52 = window['Close'].rolling(252).max().iloc[-1]
        
        # RSI
        delta = window['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1])) if loss.iloc[-1] != 0 else 50
        
        # Volatility & Beta
        rets_stock = window['Close'].pct_change().dropna()[-60:]
        rets_nifty = nifty_window['Close'].pct_change().dropna()[-60:]
        
        # Align lengths
        min_len = min(len(rets_stock), len(rets_nifty))
        if min_len > 30:
            cov = np.cov(rets_stock[-min_len:], rets_nifty[-min_len:])[0][1]
            var = np.var(rets_nifty[-min_len:])
            beta = cov / var if var != 0 else 1.0
        else:
            beta = 1.0
            
        volatility = rets_stock.std() * np.sqrt(252) * 100
        
        # Narrow Range / ADX proxy (ATR / Price)
        tr = window['High'] - window['Low']
        atr = tr.rolling(14).mean().iloc[-1]
        volatility_contraction = (atr / price) * 100 # Normalized ATR
        
        # RS Rating (vs Nifty)
        rs_stock = (price - window['Close'].iloc[-63]) / window['Close'].iloc[-63]
        rs_nifty = (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-63]) / nifty_window['Close'].iloc[-63]
        rs_score = (rs_stock - rs_nifty) * 100
        
        return {
            'price': price,
            'ma20': ma20,
            'ma50': ma50,
            'ma200': ma200,
            'high_20': high_20,
            'high_52': high_52,
            'rsi': rsi,
            'beta': beta,
            'volatility': volatility,
            'vol_contraction': volatility_contraction,
            'rs_score': rs_score,
            'volume': window['Volume'].iloc[-1],
            'avg_vol_20': window['Volume'].rolling(20).mean().iloc[-1]
        }

    def passes_filter(self, ticker, date):
        if ticker not in self.data_cache: return False, {}
        df = self.data_cache[ticker]
        nifty = self.data_cache['NIFTY']
        
        idx = df.index.searchsorted(date)
        if idx < 200: return False, {}
        
        window = df.iloc[max(0, idx-252):idx+1]
        nifty_idx = nifty.index.searchsorted(date)
        nifty_window = nifty.iloc[max(0, nifty_idx-252):nifty_idx+1]
        
        ind = self.calculate_indicators(window, nifty_window)
        
        # STRATEGY LOGIC
        
        if self.strategy == 'RS_LEADER':
            # 1. Broad Market is Weak (Optional, or just find leaders anytime)
            # 2. Stock is hitting 20-day Highs (Breaking out)
            if ind['price'] < ind['high_20'] * 0.98: return False, {} # Must be near high
            
            # 3. RS Score > 10 (Strong Outperformance)
            if ind['rs_score'] < 10: return False, {}
            
            # 4. Stock Trend is UP (Price > MA50)
            if ind['price'] < ind['ma50']: return False, {}
            
            return True, ind
            
        elif self.strategy == 'FORTRESS':
            # 1. Low Beta
            if ind['beta'] > 0.8: return False, {}
            
            # 2. Low Volatility
            if ind['volatility'] > 30: return False, {}
            
            # 3. Not crashing
            if ind['price'] < ind['ma200']: return False, {}
            
            return True, ind
            
        elif self.strategy == 'DIP_BUY_REV':
            # 1. Primary Trend UP (MA50 > MA200 and Price > MA200)
            if not (ind['ma50'] > ind['ma200'] and ind['price'] > ind['ma200']): return False, {}
            
            # 2. Strong RS (Leader pulling back)
            if ind['rs_score'] < 5: return False, {}
            
            # 3. Oversold on Daily
            if ind['rsi'] > 35: return False, {}
            
            return True, ind
            
        elif self.strategy == 'VCP_BO':
            # 1. Volatility Contraction (Low ATR%)
            if ind['vol_contraction'] > 3.0: return False, {} # Tight range
            
            # 2. Breaking Out on Volume
            if ind['price'] > ind['high_20'] * 0.99 and \
               ind['volume'] > ind['avg_vol_20'] * 1.5:
                return True, ind
                
            return False, {}

        elif self.strategy == 'ATH_BEAR':
            # 1. Hitting 52-Week High (Strength)
            if ind['price'] < ind['high_52'] * 0.98: return False, {}
            
            # 2. Strong RS (> 10)
            if ind['rs_score'] < 10: return False, {}
            
            return True, ind

        elif self.strategy == 'VOLUME_SURPRISE':
            # 1. Massive Volume (> 3x Avg)
            if ind['volume'] < ind['avg_vol_20'] * 3.0: return False, {}
            
            # 2. Price UP (> 2%)
            if ind['price'] < ind['ma20'] * 1.02: return False, {}
            
            # 3. Not Penny Stock
            if ind['price'] < 50: return False, {}
            
            return True, ind
            
        return False, {}

    def run(self):
        nifty = self.data_cache['NIFTY']
        s_date_obj = datetime.strptime(self.start_date, '%Y-%m-%d')
        e_date_obj = datetime.strptime(self.end_date, '%Y-%m-%d')
        
        start_idx = nifty.index.searchsorted(s_date_obj)
        end_idx = nifty.index.searchsorted(e_date_obj)
        dates = nifty.index[start_idx:end_idx+1]
        
        print(f"Running {self.strategy}...")
        
        for date in dates:
            # EXITS
            to_exit = []
            for t, pos in self.positions.items():
                price = self.get_price(t, date)
                if not price: continue
                
                ret = (price - pos['entry']) / pos['entry']
                
                # Dynamic Logic based on Strategy
                exit_signal = False
                reason = ''
                
                if self.strategy == 'RS_LEADER':
                    # Ride the trend, trailing stop
                    if price > pos['peak']: pos['peak'] = price
                    if price < pos['peak'] * 0.90: # 10% Trailing
                        exit_signal = True; reason = 'TrailingStop'
                        
                elif self.strategy == 'FORTRESS':
                    # Stop if crashes below MA200 or 15% stop
                    if ret < -0.15: exit_signal = True; reason = 'Stop'
                    # Or rebalance/rotate (simple here)
                    if price > pos['peak']: pos['peak'] = price
                    if price < pos['peak'] * 0.92: exit_signal = True; reason = 'Trail'

                elif self.strategy == 'DIP_BUY_REV':
                    # Quick profit or stop
                    if ret > 0.10: exit_signal = True; reason = 'Target'
                    if ret < -0.07: exit_signal = True; reason = 'Stop'
                    
                elif self.strategy == 'VCP_BO':
                    if ret > 0.20: exit_signal = True; reason = 'Target'
                    if ret < -0.08: exit_signal = True; reason = 'Stop' # Tighter stop
                
                if exit_signal:
                    self.capital += pos['shares'] * price * 0.995
                    self.trade_log.append({'ticker': t, 'pnl': ret*100, 'reason': reason})
                    to_exit.append(t)
            
            for t in to_exit: del self.positions[t]
            
            # ENTRIES
            if len(self.positions) < MAX_POSITIONS:
                candidates = []
                for t in self.data_cache:
                    if t == 'NIFTY' or t in self.positions: continue
                    passes, info = self.passes_filter(t, date)
                    if passes: candidates.append({'ticker': t, **info})
                
                # Rank
                if self.strategy == 'RS_LEADER' or self.strategy == 'DIP_BUY_REV':
                    candidates.sort(key=lambda x: -x['rs_score']) # Strongest First
                elif self.strategy == 'FORTRESS':
                    candidates.sort(key=lambda x: x['beta']) # Lowest Beta First
                elif self.strategy == 'VCP_BO':
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
            
            # Value Tracking
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
    strats = ['RS_LEADER', 'FORTRESS', 'DIP_BUY_REV', 'VCP_BO', 'ATH_BEAR', 'VOLUME_SURPRISE']
    periods = [('2022', '2022-01-01', '2022-06-30'), ('2025', '2025-01-01', '2026-02-01')]
    
    print("Fetching Master Data...")
    loader = BearAlphaHunter('2022-01-01', '2026-02-01')
    loader.fetch_data()
    print("Loaded Master Data.")
    
    results = []
    
    for p_name, start, end in periods:
        print(f"\n--- Period: {p_name} ({start} to {end}) ---")
        for s in strats:
            eng = BearAlphaHunter(start, end, s)
            eng.data_cache = loader.data_cache
            tf, hf = eng.run()
            res = analyze(tf, hf, f"{s}_{p_name}")
            results.append(res)
            print(f"  {s}: {res['Return']:.1f}%")
    
    print("\n" + "="*80)
    print("ALPHA HUNTER RESULTS")
    print("="*80)
    print(pd.DataFrame(results).to_string(index=False))
    
    pd.DataFrame(results).to_csv('analysis_2026/alpha_hunter_results.csv', index=False)

if __name__ == "__main__":
    main()
