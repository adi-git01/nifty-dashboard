"""
DNA6: STEADY COMPOUNDERS (BEAR MARKET SPECIALIST)
=================================================
Tests the user's hypothesis:
- Bull Market: DNA3-V2.1 (Momentum)
- Bear Market: Target "Steady Winners" (30-100% gainers)

Steady Winner Profile (from Deep Dive):
- RS Score > 0 (Leading Market)
- Drawdown < 20% (Resilient)
- Correlation < 0.5 (Decoupled)
- Price > MA200 (Long Term Uptrend)

Comparison:
- DNA3 (Pure Momentum)
- DNA5 (VCP + Volume Surprise)
- DNA6 (Steady Compounder)
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

class DNA6Comparison:
    def __init__(self, start_date, end_date, strategy='DNA6'):
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

    def detect_regime(self, date):
        nifty = self.data_cache['NIFTY']
        idx = nifty.index.searchsorted(date)
        if idx < 200: return 'BULL' # Default
        
        price = nifty.iloc[idx]['Close']
        ma50 = nifty.iloc[:idx+1]['Close'].rolling(50).mean().iloc[-1]
        
        # Simple Regime: Nifty > MA50 = BULL, Else BEAR
        if price > ma50: return 'BULL'
        return 'BEAR'

    def passes_filter(self, ticker, date, regime):
        if ticker not in self.data_cache: return False, {}
        df = self.data_cache[ticker]
        nifty = self.data_cache['NIFTY']
        
        idx = df.index.searchsorted(date)
        if idx < 200: return False, {}
        
        window = df.iloc[max(0, idx-252):idx+1]
        nifty_idx = nifty.index.searchsorted(date)
        nifty_window = nifty.iloc[max(0, nifty_idx-252):nifty_idx+1]
        
        price = window['Close'].iloc[-1]
        ma50 = window['Close'].rolling(50).mean().iloc[-1]
        ma200 = window['Close'].rolling(200).mean().iloc[-1]
        
        # === STRATEGY LOGIC ===
        
        # 1. BULL MARKET: ALWAYS MOMENTUM (DNA3-V2.1)
        if regime == 'BULL':
            if price < ma50: return False, {}
            
            # RS > 0
            price_63 = window['Close'].iloc[-63]
            n_price_63 = nifty_window['Close'].iloc[-63]
            rs_stock = (price - price_63)/price_63
            rs_nifty = (nifty_window['Close'].iloc[-1] - n_price_63)/n_price_63
            rs_score = (rs_stock - rs_nifty) * 100
            
            if rs_score < 0: return False, {}
            return True, {'score': rs_score}

        # 2. BEAR MARKET: DIFFERENT APPROACHES
        else:
            # DNA3: NAIVE MOMENTUM (Keeps buying leaders even if everything sucks)
            if self.strategy == 'DNA3':
                if price < ma50: return False, {}
                rs_score = 0 # Calculate if needed
                return True, {'score': 1} # Just buy anything above MA50

            # DNA5: VCP + VOLUME SURPRISE
            elif self.strategy == 'DNA5':
                # VCP Logic
                tr = window['High'] - window['Low']
                atr = tr.rolling(14).mean().iloc[-1]
                vol_contraction = (atr / price) * 100
                
                high_20 = window['Close'].rolling(20).max().iloc[-1]
                curr_vol = window['Volume'].iloc[-1]
                avg_vol = window['Volume'].rolling(20).mean().iloc[-1]
                
                # VCP
                is_vcp = vol_contraction < 3.5 and price > high_20 * 0.98 and curr_vol > avg_vol * 1.5
                
                # Vol Surprise (Alpha Boost)
                is_vol_surprise = curr_vol > avg_vol * 3.0 and price > window['Close'].rolling(20).mean().iloc[-1]
                
                if is_vcp or is_vol_surprise:
                    return True, {'score': 100 if is_vol_surprise else 50}
                return False, {}

            # DNA6: STEADY COMPOUNDERS (The New Hypothesis)
            elif self.strategy == 'DNA6':
                # 1. Trend: Long Term UP (Price > MA200)
                if price < ma200: return False, {}
                
                # 2. Resilient: Drawdown < 20%
                high_52 = window['Close'].max()
                dd = (price - high_52)/high_52 * 100
                if dd < -20: return False, {}
                
                # 3. Leader: RS Score > 0
                price_63 = window['Close'].iloc[-63]
                n_price_63 = nifty_window['Close'].iloc[-63]
                rs_stock = (price - price_63)/price_63
                rs_nifty = (nifty_window['Close'].iloc[-1] - n_price_63)/n_price_63
                rs_score = (rs_stock - rs_nifty) * 100
                
                if rs_score < 0: return False, {}
                
                # 4. Decoupled: Correlation < 0.5 (Optional, but found in deep dive)
                rets_stock = window['Close'].pct_change().dropna()[-60:]
                rets_nifty = nifty_window['Close'].pct_change().dropna()[-60:]
                common = rets_stock.index.intersection(rets_nifty.index)
                if len(common) > 30:
                    corr = rets_stock.loc[common].corr(rets_nifty.loc[common])
                    if corr > 0.5: return False, {} # Skip highly correlated stocks
                
                return True, {'score': rs_score}

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
            regime = self.detect_regime(date)
            
            # EXITS
            to_exit = []
            for t, pos in self.positions.items():
                price = self.get_price(t, date)
                if not price: continue
                
                ret = (price - pos['entry']) / pos['entry']
                
                exit_signal = False
                reason = ''
                
                # Standard Exit: Target +20%, Stop -10% or Trailing
                if ret > 0.20: exit_signal = True; reason = 'Target'
                elif ret < -0.10: exit_signal = True; reason = 'Stop'
                
                if price > pos['peak']: pos['peak'] = price
                if price < pos['peak'] * 0.90: exit_signal = True; reason = 'Trail'
                
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
                    passes, info = self.passes_filter(t, date, regime)
                    if passes: candidates.append({'ticker': t, **info})
                
                candidates.sort(key=lambda x: -x['score'])
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

def analyze(hf, name):
    if hf.empty: return {}
    start = hf['value'].iloc[0]
    end = hf['value'].iloc[-1]
    ret = (end - start)/start * 100
    hf['peak'] = hf['value'].cummax()
    hf['dd'] = (hf['value'] - hf['peak']) / hf['peak'] * 100
    max_dd = hf['dd'].min()
    return {'Strategy': name, 'Return': ret, 'Max DD': max_dd}

def main():
    print("Fetching Master Data...")
    loader = DNA6Comparison('2022-01-01', '2026-02-01')
    loader.fetch_data()
    print("Loaded Master Data.")
    
    strats = ['DNA3', 'DNA5', 'DNA6']
    # Focus only on Bear/Choppy periods to see the difference
    periods = [('2022', '2022-01-01', '2022-06-30'), ('2025', '2025-01-01', '2026-02-01')]
    
    results = []
    
    for p_name, start, end in periods:
        print(f"\n--- Period: {p_name} ---")
        for s in strats:
            eng = DNA6Comparison(start, end, s)
            eng.data_cache = loader.data_cache
            _, hf = eng.run()
            res = analyze(hf, f"{s}_{p_name}")
            results.append(res)
            print(f"  {s}: {res['Return']:.1f}%")
            
    print("\n" + "="*60)
    print("DNA6 vs OTHERS (Bear Market Performance)")
    print("="*60)
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()
