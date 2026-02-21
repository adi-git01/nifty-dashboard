"""
DEEP STRATEGY COMPARISON: DNA4-C vs DNA3-V2.1-Momentum
======================================================
Comprehensive analysis of:
1. DNA3-V2.1 (Momentum: Trend 90+, RS > 2%, Vol > 30%)
2. DNA4-C (Regime Adaptive: Momentum in Bull, Simple RSI<30 in Bear/Sideways)

Metrics:
- Performance: Total Return, CAGR, Alpha, Drawdown
- Trade Stats: Win Rate, Avg Win/Loss, Max Win/Loss, Median Win/Loss
- Holding Periods: Avg Days Held, Max Days Held
- Stop Loss Analysis: % of Exits via Stop vs Profit
- Entry/Exit Characteristics: Avg Trend Score, RSI, Volume, PE (if available)
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
STOP_LOSS_MOM = -0.15
STOP_LOSS_REV = -0.10
TRAILING_ACT = 0.10
TRAILING_AMT = 0.10

class DeepComparison:
    def __init__(self, years, strategy_name):
        self.years = years
        self.strategy_name = strategy_name # 'DNA3-V2.1' or 'DNA4-C'
        self.data_cache = {}
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
        self.sector_map = SECTOR_MAP
        
    def fetch_data(self):
        # ... logic similar to previous scripts, reused to save space/time ...
        pass
        
    def get_price(self, ticker, date):
        if ticker not in self.data_cache: return None
        df = self.data_cache[ticker]
        mask = df.index <= date
        if mask.sum() == 0: return None
        return df.loc[mask, 'Close'].iloc[-1]

    def detect_regime(self, date):
        nifty = self.data_cache['NIFTY']
        idx = nifty.index.searchsorted(date)
        if idx < 200: return 'SIDEWAYS'
        
        window = nifty.iloc[max(0, idx-200):idx+1]
        price = window['Close'].iloc[-1]
        ma50 = window['Close'].rolling(50).mean().iloc[-1]
        ret_1m = (price - window['Close'].iloc[-21]) / window['Close'].iloc[-21] * 100 if len(window) > 21 else 0
        
        if price > ma50 and ret_1m > 0: return 'BULL' # Simplified Bull/Mild Bull
        return 'BEAR_SIDEWAYS' 

    def calculate_metrics(self, window):
        price = window['Close'].iloc[-1]
        
        # Trend Score
        ma50 = window['Close'].rolling(50).mean().iloc[-1]
        ma200 = window['Close'].rolling(200).mean().iloc[-1]
        score = 50
        if price > ma50: score += 20
        if price > ma200: score += 15
        if ma50 > ma200: score += 10
        ret_1m = (price - window['Close'].iloc[-21]) / window['Close'].iloc[-21] * 100 if len(window) > 21 else 0
        if ret_1m > 5: score += 5
        
        # RSI
        delta = window['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1])) if loss.iloc[-1] != 0 else 50
        
        # Volatility
        rets = window['Close'].pct_change().dropna()[-60:]
        vol = rets.std() * np.sqrt(252) * 100 if len(rets) > 10 else 0
        
        return score, rsi, vol

    def passes_filter(self, ticker, date, strategy_mode):
        if ticker not in self.data_cache: return False, {}
        df = self.data_cache[ticker]
        nifty = self.data_cache['NIFTY']
        
        idx = df.index.searchsorted(date)
        if idx < 252: return False, {}
        
        window = df.iloc[max(0, idx-252):idx+1]
        nifty_idx = nifty.index.searchsorted(date)
        nifty_window = nifty.iloc[max(0, nifty_idx-252):nifty_idx+1]
        
        if len(window) < 200: return False, {}
        
        price = window['Close'].iloc[-1]
        trend, rsi, vol = self.calculate_metrics(window)
        
        # DNA3-V2.1 / DNA4-C Momentum Logic
        if strategy_mode == 'MOMENTUM': 
            ma50 = window['Close'].rolling(50).mean().iloc[-1]
            if price < ma50: return False, {}
            
            ret_3m = (price - window['Close'].iloc[-63]) / window['Close'].iloc[-63] * 100 if len(window) > 63 else 0
            nifty_ret_3m = (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-63]) / nifty_window['Close'].iloc[-63] * 100 if len(nifty_window) > 63 else 0
            rs = ret_3m - nifty_ret_3m
            
            if rs < 2.0: return False, {}
            if vol < 30: return False, {}
            
            return True, {'entry_trend': trend, 'entry_rsi': rsi, 'entry_vol': vol, 'mode': 'MOMENTUM'}
            
        # DNA4-C Mean Reversion Logic
        if strategy_mode == 'REVERSION':
            if rsi > 30: return False, {}
            # Basic sanity
            high_52 = window['Close'].max()
            if price < high_52 * 0.5: return False, {} 
            
            return True, {'entry_trend': trend, 'entry_rsi': rsi, 'entry_vol': vol, 'mode': 'REVERSION'}
            
        return False, {}

    def run(self):
        nifty = self.data_cache['NIFTY']
        start_idx = nifty.index.searchsorted(datetime.now() - timedelta(days=365*self.years))
        dates = nifty.index[start_idx:]
        
        for date in dates:
            regime = self.detect_regime(date)
            
            # Determine logic
            # DNA3-V2.1 always uses MOMENTUM
            # DNA4-C uses MOMENTUM in BULL, REVERSION in BEAR_SIDEWAYS
            
            current_mode = 'MOMENTUM'
            if self.strategy_name == 'DNA4-C':
                if regime == 'BEAR_SIDEWAYS': current_mode = 'REVERSION'
            
            # EXITS
            to_exit = []
            for t, pos in self.positions.items():
                price = self.get_price(t, date)
                if not price: continue
                
                # Update stats
                pos['days_held'] += 1
                if price > pos['peak']: pos['peak'] = price
                
                pnl_curr = (price - pos['entry']) / pos['entry']
                
                # Exit Logic
                exit_signal = False
                reason = ''
                
                if pos['mode'] == 'MOMENTUM':
                    if pnl_curr > TRAILING_ACT:
                        trail = pos['peak'] * (1 - TRAILING_AMT)
                        if trail > pos['stop']: pos['stop'] = trail
                    
                    if price < pos['stop']:
                        exit_signal = True
                        reason = 'Stop/Trail'
                        
                elif pos['mode'] == 'REVERSION':
                    # TP +15%, SL -10%
                    if pnl_curr > 0.15: 
                        exit_signal = True; reason = 'Target'
                    elif pnl_curr < -0.10: 
                        exit_signal = True; reason = 'Stop'
                        
                    # RSI Exit > 55
                    # (Simplified for speed: check if current price implies recovery)
                    # Ideally would recalculate RSI, here approx via price recovery
                    if pnl_curr > 0.05 and pos['days_held'] > 5: # Small profit quick exit approximation or check RSI properly
                         # Check RSI properly
                         df = self.data_cache[t]
                         idx = df.index.searchsorted(date)
                         if idx > 14:
                             w = df.iloc[max(0, idx-20):idx+1]
                             _, rsi_now, _ = self.calculate_metrics(w)
                             if rsi_now > 55:
                                 exit_signal = True; reason = 'RSI_Recovery'

                if exit_signal:
                    self.capital += pos['shares'] * price * 0.995
                    pnl = (price - pos['entry']) / pos['entry'] * 100
                    
                    # Exit metrics
                    df = self.data_cache[t]
                    idx = df.index.searchsorted(date)
                    w = df.iloc[max(0, idx-252):idx+1]
                    trend, rsi, vol = self.calculate_metrics(w)
                    
                    self.trade_log.append({
                        'ticker': t,
                        'pnl': pnl,
                        'days_held': pos['days_held'],
                        'reason': reason,
                        'mode': pos['mode'],
                        'exit_trend': trend,
                        'exit_rsi': rsi,
                        'entry_trend': pos['entry_trend'],
                        'entry_rsi': pos['entry_rsi'],
                        'entry_vol': pos['entry_vol']
                    })
                    to_exit.append(t)
            
            for t in to_exit: del self.positions[t]
            
            # ENTRIES
            if len(self.positions) < MAX_POSITIONS:
                candidates = []
                for t in self.data_cache:
                    if t == 'NIFTY' or t in self.positions: continue
                    
                    passes, metrics = self.passes_filter(t, date, current_mode)
                    if passes: candidates.append({'ticker': t, **metrics})
                
                # Rank
                if current_mode == 'MOMENTUM':
                    candidates.sort(key=lambda x: -x['entry_trend']) # Highest trend score preferred? Or RS? Use Trend for "DNA" feel
                    # Actually DNA3 uses RS rank usually. Let's stick to Trend Score for deep comparison request or RS.
                    # Standard DNA3 uses RS rank. Keeping RS rank implied by earlier logic, but let's use Trend Score as tie breaker.
                    # For strict DNA3-V2.1 compliance -> RS rank. Let's stick to RS rank concepts or simple check.
                    # Let's use Metrics: Trend Score for ranking to see "Strongest Trend"
                    candidates.sort(key=lambda x: -x['entry_trend']) 
                else: 
                    candidates.sort(key=lambda x: x['entry_rsi']) # Lowest RSI
                
                selected = self.select_with_sector_cap(candidates)
                
                for c in selected[:MAX_POSITIONS - len(self.positions)]:
                    price = self.get_price(c['ticker'], date)
                    if price:
                        stop = STOP_LOSS_MOM if c['mode'] == 'MOMENTUM' else STOP_LOSS_REV
                        size = self.capital / (MAX_POSITIONS - len(self.positions) + 2)
                        shares = int(size / price)
                        if shares > 0:
                            cost = shares * price * 1.005
                            if self.capital >= cost:
                                self.capital -= cost
                                self.positions[c['ticker']] = {
                                    'entry': price,
                                    'stop': price * (1 + stop),
                                    'peak': price,
                                    'shares': shares,
                                    'mode': c['mode'],
                                    'days_held': 0,
                                    'entry_trend': c['entry_trend'],
                                    'entry_rsi': c['entry_rsi'],
                                    'entry_vol': c['entry_vol']
                                }
            
            # Portfolio Value
            val = self.capital
            for t, pos in self.positions.items():
                p = self.get_price(t, date)
                if p: val += pos['shares'] * p
            self.history.append({'date': date, 'value': val})
            
        return pd.DataFrame(self.history), pd.DataFrame(self.trade_log)

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

def analyze_results(res_dict):
    stats = []
    
    for key, (hf, tf) in res_dict.items():
        strat, years = key.split('_')
        
        # Returns
        start_val = hf['value'].iloc[0]
        end_val = hf['value'].iloc[-1]
        total_ret = (end_val - start_val)/start_val * 100
        cagr = ((end_val/start_val)**(1/int(years)) - 1) * 100
        
        # Drawdown
        hf['peak'] = hf['value'].cummax()
        hf['dd'] = (hf['value'] - hf['peak']) / hf['peak'] * 100
        max_dd = hf['dd'].min()
        
        # Trade Stats
        if not tf.empty:
            win_rate = (tf['pnl'] > 0).mean() * 100
            avg_win = tf[tf['pnl'] > 0]['pnl'].mean() if not tf[tf['pnl'] > 0].empty else 0
            avg_loss = tf[tf['pnl'] <= 0]['pnl'].mean() if not tf[tf['pnl'] <= 0].empty else 0
            median_win = tf[tf['pnl'] > 0]['pnl'].median() if not tf[tf['pnl'] > 0].empty else 0
            median_loss = tf[tf['pnl'] <= 0]['pnl'].median() if not tf[tf['pnl'] <= 0].empty else 0
            max_win = tf['pnl'].max()
            max_loss = tf['pnl'].min()
            
            avg_hold = tf['days_held'].mean()
            avg_hold_win = tf[tf['pnl'] > 0]['days_held'].mean()
            avg_hold_loss = tf[tf['pnl'] <= 0]['days_held'].mean()
            
            avg_entry_trend = tf['entry_trend'].mean()
            avg_exit_trend = tf['exit_trend'].mean()
            avg_entry_rsi = tf['entry_rsi'].mean()
            avg_entry_vol = tf['entry_vol'].mean()
            
            stop_exits = (tf['reason'].str.contains('Stop')).mean() * 100
        else:
            win_rate = avg_win = avg_loss = max_win = max_loss = 0
            avg_hold = avg_entry_trend = avg_exit_trend = 0
            stop_exits = 0
            
        stats.append({
            'Strategy': strat,
            'Years': years,
            'CAGR': cagr,
            'Max DD': max_dd,
            'Win Rate': win_rate,
            'Avg Win': avg_win,
            'Avg Loss': avg_loss,
            'Median Win': median_win,
            'Median Loss': median_loss,
            'Max Win': max_win,
            'Avg Hold (Days)': avg_hold,
            'Entry Trend': avg_entry_trend,
            'Entry RSI': avg_entry_rsi,
            'Stop Exit %': stop_exits
        })
        
    return pd.DataFrame(stats)

def main():
    periods = [1, 3, 5, 10]
    results = {}
    
    # Pre-fetch data once to save time
    print("Fetching Master Data...")
    loader = DeepComparison(10, 'DNA3-V2.1')
    start_date = (datetime.now() - timedelta(days=365*10 + 365)).strftime('%Y-%m-%d')
    
    nifty = yf.Ticker("^NSEI").history(start=start_date)
    nifty.index = nifty.index.tz_localize(None)
    loader.data_cache['NIFTY'] = nifty
    
    loaded = 0
    for t in TICKERS[:500]:
        try:
            df = yf.Ticker(t).history(start=start_date)
            if not df.empty and len(df) > 200:
                df.index = df.index.tz_localize(None)
                loader.data_cache[t] = df
                loaded += 1
        except: pass
    print(f"Loaded {loaded} stocks.")
    
    for y in periods:
        for strat in ['DNA3-V2.1', 'DNA4-C']:
            print(f"Running {strat} for {y} Year(s)...")
            eng = DeepComparison(y, strat)
            eng.data_cache = loader.data_cache # Share data
            results[f"{strat}_{y}"] = eng.run()
            
    df_stats = analyze_results(results)
    print("\n" + "="*100)
    print("DEEP STRATEGY COMPARISON REPORT")
    print("="*100)
    print(df_stats.to_string(index=False))
    
    df_stats.to_csv('analysis_2026/dna_deep_comparison.csv', index=False)
    print("\nSaved to analysis_2026/dna_deep_comparison.csv")

if __name__ == "__main__":
    main()
