"""
DNA-3 V2: REFINED ALPHA STRATEGY
=================================
Learnings from V1:
- RS > 3% is valid but maybe too strict (lowering to 2%)
- Volatility > 35% is valid
- Pullback filter was TOO RESTRICTIVE (missed breakouts)

V2 Changes:
1. RS > 2% (Lowered threshold)
2. Volatility > 30% (Slightly lowered)
3. REMOVED pullback filter (Allow breakouts)
4. Added: Must be in TOP 5 sectors (Industrials, Metals, IT, Auto, Realty)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

INITIAL_CAPITAL = 1000000
MAX_POSITIONS = 10
STOP_LOSS = -0.15
TRAILING_ACTIVATION = 0.10
TRAILING_AMOUNT = 0.10

# FOCUS ON TOP 5 SECTORS ONLY
UNIVERSE = {
    'Industrials': ['LT.NS', 'SIEMENS.NS', 'ABB.NS', 'HAVELLS.NS', 'CUMMINSIND.NS', 'POLYCAB.NS', 'BEL.NS', 'HAL.NS', 'THERMAX.NS'],
    'Metals': ['TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 'VEDL.NS', 'JINDALSTEL.NS', 'NMDC.NS', 'NATIONALUM.NS'],
    'IT_Services': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTIM.NS', 'COFORGE.NS', 'PERSISTENT.NS', 'LTTS.NS'],
    'Auto': ['MARUTI.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS', 'ASHOKLEY.NS', 'MOTHERSON.NS'],
    'Realty': ['DLF.NS', 'GODREJPROP.NS', 'OBEROIRLTY.NS', 'PRESTIGE.NS']
}

class DNA3V2Strategy:
    def __init__(self):
        self.data_cache = {}
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
        
    def fetch_data(self):
        print("[DNA-3 V2] Fetching data...")
        start_date = (datetime.now() - timedelta(days=365*5 + 300)).strftime('%Y-%m-%d')
        
        nifty = yf.Ticker("^NSEI").history(start=start_date)
        nifty.index = nifty.index.tz_localize(None)
        self.data_cache['NIFTY'] = nifty
        
        all_tickers = [t for sector in UNIVERSE.values() for t in sector]
        for t in all_tickers:
            try:
                df = yf.Ticker(t).history(start=start_date)
                if not df.empty:
                    df.index = df.index.tz_localize(None)
                    self.data_cache[t] = df
            except: pass
        print(f"   Loaded {len(self.data_cache)} symbols")

    def get_price(self, ticker, date):
        if ticker not in self.data_cache: return None
        df = self.data_cache[ticker]
        mask = df.index <= date
        if mask.sum() == 0: return None
        return df.loc[mask, 'Close'].iloc[-1]

    def passes_dna_filter_v2(self, ticker, date):
        """V2: Relaxed DNA Filters."""
        if ticker not in self.data_cache: return False, {}
        df = self.data_cache[ticker]
        nifty = self.data_cache['NIFTY']
        
        idx = df.index.searchsorted(date)
        if idx < 252: return False, {}
        
        window = df.iloc[max(0, idx-252):idx+1]
        nifty_idx = nifty.index.searchsorted(date)
        nifty_window = nifty.iloc[max(0, nifty_idx-252):nifty_idx+1]
        
        if len(window) < 100: return False, {}
        
        price = window['Close'].iloc[-1]
        
        # === V2 FILTER 1: RELATIVE STRENGTH (Lowered to 2%) ===
        ret_3m = (price - window['Close'].iloc[-63]) / window['Close'].iloc[-63] * 100 if len(window) > 63 else 0
        nifty_ret_3m = 0
        if len(nifty_window) > 63:
            nifty_ret_3m = (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-63]) / nifty_window['Close'].iloc[-63] * 100
        rs_3m = ret_3m - nifty_ret_3m
        
        if rs_3m < 2.0: return False, {}  # Lowered from 3% to 2%
        
        # === V2 FILTER 2: VOLATILITY (Lowered to 30%) ===
        stock_returns = window['Close'].pct_change().dropna()[-60:]
        volatility = stock_returns.std() * np.sqrt(252) * 100 if len(stock_returns) > 10 else 0
        
        if volatility < 30: return False, {}  # Lowered from 35%
        
        # === V2 FILTER 3: TREND CONFIRMATION ===
        ma50 = window['Close'].rolling(50).mean().iloc[-1] if len(window) > 50 else price
        ma200 = window['Close'].rolling(200).mean().iloc[-1] if len(window) > 200 else price
        
        if price < ma50: return False, {}  # Must be above 50 MA
        
        return True, {'rs_3m': rs_3m, 'volatility': volatility}

    def run(self):
        self.fetch_data()
        nifty = self.data_cache['NIFTY']
        start_idx = nifty.index.searchsorted(datetime.now() - timedelta(days=365*5))
        dates = nifty.index[start_idx:]
        
        print("[DNA-3 V2] Running 5-Year Backtest...")
        
        for date in dates:
            # === EXITS ===
            to_exit = []
            for t, pos in self.positions.items():
                price = self.get_price(t, date)
                if not price: continue
                
                if price > pos['peak']: pos['peak'] = price
                
                ret = (price - pos['entry']) / pos['entry']
                if ret > TRAILING_ACTIVATION:
                    trail = pos['peak'] * (1 - TRAILING_AMOUNT)
                    if trail > pos['stop']: pos['stop'] = trail
                
                if price < pos['stop']:
                    val = pos['shares'] * price * 0.995
                    self.capital += val
                    pnl = (price - pos['entry']) / pos['entry'] * 100
                    self.trade_log.append({'ticker': t, 'pnl': pnl, 'date': date})
                    to_exit.append(t)
            
            for t in to_exit: del self.positions[t]
            
            # === ENTRIES ===
            if len(self.positions) < MAX_POSITIONS:
                candidates = []
                for sector, tickers in UNIVERSE.items():
                    for ticker in tickers:
                        if ticker in self.positions: continue
                        
                        passes, metrics = self.passes_dna_filter_v2(ticker, date)
                        if passes:
                            candidates.append({'ticker': ticker, 'sector': sector, 'rs': metrics['rs_3m']})
                
                candidates.sort(key=lambda x: -x['rs'])
                
                for c in candidates[:MAX_POSITIONS - len(self.positions)]:
                    price = self.get_price(c['ticker'], date)
                    if price:
                        size = self.capital / (MAX_POSITIONS - len(self.positions) + 2)
                        shares = int(size / price)
                        if shares > 0:
                            cost = shares * price * 1.005
                            if self.capital >= cost:
                                self.capital -= cost
                                self.positions[c['ticker']] = {
                                    'entry': price, 'stop': price * (1 + STOP_LOSS),
                                    'peak': price, 'shares': shares
                                }
            
            val = self.capital
            for t, pos in self.positions.items():
                p = self.get_price(t, date)
                if p: val += pos['shares'] * p
            self.history.append({'date': date, 'value': val})
        
        df = pd.DataFrame(self.history)
        final_val = df.iloc[-1]['value']
        total_ret = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        
        n_s = nifty.loc[df.iloc[0]['date']]['Close']
        n_e = nifty.loc[df.iloc[-1]['date']]['Close']
        n_ret = (n_e - n_s) / n_s * 100
        
        print("\n" + "="*60)
        print("[DNA-3 V2] RESULTS (5 Years)")
        print("="*60)
        print(f"Total Return: {total_ret:.2f}%")
        print(f"Nifty Return: {n_ret:.2f}%")
        print(f"ALPHA: {total_ret - n_ret:.2f}%")
        print(f"CAGR: {((final_val/INITIAL_CAPITAL)**(1/5) - 1)*100:.2f}%")
        print(f"Nifty CAGR: {((1 + n_ret/100)**(1/5) - 1)*100:.2f}%")
        
        wins = [t for t in self.trade_log if t['pnl'] > 0]
        losses = [t for t in self.trade_log if t['pnl'] <= 0]
        print(f"\nTrades: {len(self.trade_log)} | Win Rate: {len(wins)/len(self.trade_log)*100:.1f}%" if self.trade_log else "")
        if wins: print(f"Avg Win: {np.mean([t['pnl'] for t in wins]):.1f}%")
        if losses: print(f"Avg Loss: {np.mean([t['pnl'] for t in losses]):.1f}%")

if __name__ == "__main__":
    strat = DNA3V2Strategy()
    strat.run()
