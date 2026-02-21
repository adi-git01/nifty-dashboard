"""
DNA4-SMART REVERSION: BEAR/SIDEWAYS OPTIMIZATION
================================================
Compares two approaches for Sideways/Bear markets:
1. DNA4-C (Baseline): Simple RSI < 30
2. DNA4-Smart: Quality + Value + Timing

Smart Logic:
- Quality Proxy: Hist Volatility < 40%, Long-term Uptrend (Price > 200DMA 6m ago)
- Value Proxy: Drawdown -20% to -50% from 52w High, Price near Multi-Year Support
- Timing: Volume Spikes (10d Vol > 1.5x 50d Vol) OR Price Stabilization

Test Periods:
1. 2022 Bear Market (Jan 2022 - Jun 2022)
2. 2025 Choppy Market (Jan 2025 - Feb 2026)
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

class SmartReversionComparison:
    def __init__(self, start_date, end_date, strategy='SIMPLE'):
        """strategy: 'SIMPLE' (RSI<30) or 'SMART' (Quality+Value+Timing)"""
        self.start_date = start_date
        self.end_date = end_date
        self.strategy = strategy
        self.data_cache = {}
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
        self.sector_map = SECTOR_MAP
        
    def fetch_data(self):
        # Fetch data with buffer
        s_date = (datetime.strptime(self.start_date, '%Y-%m-%d') - timedelta(days=730)).strftime('%Y-%m-%d')
        
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

    def calculate_technical_metrics(self, window):
        price = window['Close'].iloc[-1]
        
        # RSI
        delta = window['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1])) if loss.iloc[-1] != 0 else 50
        
        # Volatility (60d)
        rets = window['Close'].pct_change().dropna()[-60:]
        vol = rets.std() * np.sqrt(252) * 100 if len(rets) > 10 else 0
        
        # Volume Spike
        vol_10 = window['Volume'].rolling(10).mean().iloc[-1]
        vol_50 = window['Volume'].rolling(50).mean().iloc[-1]
        vol_ratio = vol_10 / vol_50 if vol_50 > 0 else 1.0
        
        # Drawdown from 52w High
        high_52 = window['Close'].rolling(252).max().iloc[-1]
        drawdown = (price - high_52) / high_52 * 100
        
        # Long-term Trend (200 DMA status 6 months ago to check quality)
        # 6 months ~ 126 trading days
        if len(window) > 130:
            price_6m_ago = window['Close'].iloc[-126]
            ma200_6m_ago = window['Close'].rolling(200).mean().iloc[-126]
            was_uptrend = price_6m_ago > ma200_6m_ago
        else:
            was_uptrend = False
            
        # Price Stabilization (Low 5-day variance)
        price_var_5d = window['Close'].pct_change().tail(5).std() * 100
        
        return {
            'rsi': rsi,
            'volatility': vol,
            'vol_ratio': vol_ratio,
            'drawdown': drawdown,
            'was_uptrend': was_uptrend,
            'price_var_5d': price_var_5d,
            'high_52': high_52
        }

    def passes_filter(self, ticker, date):
        if ticker not in self.data_cache: return False, {}
        df = self.data_cache[ticker]
        
        idx = df.index.searchsorted(date)
        if idx < 252: return False, {}
        
        window = df.iloc[max(0, idx-300):idx+1] # Need more history for 200dma check
        metrics = self.calculate_technical_metrics(window)
        price = window['Close'].iloc[-1]
        
        if self.strategy == 'SIMPLE':
            # Simple RSI < 30
            if metrics['rsi'] > 30: return False, {}
            if price < metrics['high_52'] * 0.5: return False, {} # Basic sanity
            return True, {'entry_metrics': metrics}
            
        elif self.strategy == 'SMART':
            # 1. Quality Proxy (Relaxed)
            if metrics['volatility'] > 55: return False, {} # Avoid extreme junk
            
            # Remove strict uptrend check - in deep bears, everything looks like a downtrend
            
            # 2. Value Proxy (Deep Discount)
            # Allow slightly wider drawdown
            if metrics['drawdown'] > -15 or metrics['drawdown'] < -60: return False, {} 
            
            # 3. Timing / Oversold Condition
            # A: Deeply Oversold Quality (RSI < 25) -> BUY
            if metrics['rsi'] < 25:
                return True, {'entry_metrics': metrics}
                
            # B: Moderate Oversold (RSI < 35) WITH Volume/Stabilization
            if metrics['rsi'] < 35:
                volume_spike = metrics['vol_ratio'] > 1.2
                stabilized = metrics['price_var_5d'] < 2.0
                if volume_spike or stabilized:
                    return True, {'entry_metrics': metrics}
            
            return False, {}
            
        return False, {}

    def run(self):
        nifty = self.data_cache['NIFTY']
        s_date_obj = datetime.strptime(self.start_date, '%Y-%m-%d')
        e_date_obj = datetime.strptime(self.end_date, '%Y-%m-%d')
        
        start_idx = nifty.index.searchsorted(s_date_obj)
        end_idx = nifty.index.searchsorted(e_date_obj)
        dates = nifty.index[start_idx:end_idx+1]
        
        print(f"Running {self.strategy} from {self.start_date} to {self.end_date}...")
        
        for date in dates:
            # EXITS
            to_exit = []
            for t, pos in self.positions.items():
                price = self.get_price(t, date)
                if not price: continue
                
                ret = (price - pos['entry']) / pos['entry']
                
                # Exit Logic: Target +15% or Stop -10% or RSI recovery
                exit_signal = False
                reason = ''
                
                if ret > 0.15: exit_signal = True; reason = 'Target'
                elif ret < -0.10: exit_signal = True; reason = 'Stop'
                
                # Check Time-based exit for Reversion (don't hold dead money)
                days_held = (date - pd.Timestamp(pos['entry_date'])).days
                if days_held > 30 and ret < 0: exit_signal = True; reason = 'TimeStop'
                
                if exit_signal:
                    self.capital += pos['shares'] * price * 0.995
                    pnl = ret * 100
                    self.trade_log.append({
                        'ticker': t,
                        'pnl': pnl,
                        'reason': reason,
                        'days_held': days_held,
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
                
                # Rank
                if self.strategy == 'SIMPLE':
                    candidates.sort(key=lambda x: x['entry_metrics']['rsi']) # Lowest RSI
                else:
                    # Smart Rank: Combo of Drawdown (Value) and Vol Spike (Timing)
                    # Prefer higher volume spike
                    candidates.sort(key=lambda x: -x['entry_metrics']['vol_ratio'])
                
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

def analyze_period(tf, hf, strat_name):
    if hf.empty: return {}
    start_val = hf['value'].iloc[0]
    end_val = hf['value'].iloc[-1]
    ret = (end_val - start_val)/start_val * 100
    
    hf['peak'] = hf['value'].cummax()
    hf['dd'] = (hf['value'] - hf['peak']) / hf['peak'] * 100
    max_dd = hf['dd'].min()
    
    if not tf.empty:
        win_rate = (tf['pnl'] > 0).mean() * 100
        avg_win = tf[tf['pnl'] > 0]['pnl'].mean()
        avg_loss = tf[tf['pnl'] <= 0]['pnl'].mean()
        trades = len(tf)
    else:
        win_rate = avg_win = avg_loss = trades = 0
        
    return {
        'Strategy': strat_name,
        'Return': ret,
        'Max DD': max_dd,
        'Win Rate': win_rate,
        'Trades': trades,
        'Avg Win': avg_win,
        'Avg Loss': avg_loss
    }

def main():
    # Define Test Periods
    periods = [
        ('2022_BEAR', '2022-01-01', '2022-06-30'),
        ('2025_CHOP', '2025-01-01', '2026-02-01')
    ]
    
    results = []
    
    # Pre-fetch for speed
    print("Fetching Master Data...")
    loader = SmartReversionComparison('2022-01-01', '2026-02-01')
    loader.fetch_data()
    print(f"Loaded {len(loader.data_cache)} stocks (Master Cache)")
    
    for name, start, end in periods:
        print(f"\n--- TESTING PERIOD: {name} ({start} to {end}) ---")
        
        # Test SIMPLE
        eng_simple = SmartReversionComparison(start, end, 'SIMPLE')
        eng_simple.data_cache = loader.data_cache
        tf_s, hf_s = eng_simple.run()
        results.append(analyze_period(tf_s, hf_s, f"SIMPLE ({name})"))
        
        # Test SMART
        eng_smart = SmartReversionComparison(start, end, 'SMART')
        eng_smart.data_cache = loader.data_cache
        tf_m, hf_m = eng_smart.run()
        results.append(analyze_period(tf_m, hf_m, f"SMART ({name})"))
        
    print("\n" + "="*100)
    print("SMART REVERSION vs SIMPLE RSI: HEAD-TO-HEAD")
    print("="*100)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    df.to_csv('analysis_2026/smart_reversion_results.csv', index=False)
    print("\nSaved to analysis_2026/smart_reversion_results.csv")

if __name__ == "__main__":
    main()
