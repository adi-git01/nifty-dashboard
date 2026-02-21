"""
DNA-3 V2 + PYRAMIDING: THE AMPLIFIER
=====================================
DNA-3 V2 is the proven winner (569% return).
Now let's AMPLIFY it with pyramiding.

PYRAMID RULES:
1. Initial Entry: Standard DNA-3 V2 filters
2. ADD Position: When up 15%+, add 50% more
3. MAX 3 layers per stock
4. Trail entire position together

The rich get richer.
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
PYRAMID_TRIGGER = 0.15  # Add when up 15%
PYRAMID_SIZE = 0.50  # Add 50% of original position

UNIVERSE = {
    'Industrials': ['LT.NS', 'SIEMENS.NS', 'ABB.NS', 'HAVELLS.NS', 'CUMMINSIND.NS', 'POLYCAB.NS', 'BEL.NS', 'HAL.NS'],
    'Metals': ['TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 'VEDL.NS', 'JINDALSTEL.NS', 'NMDC.NS', 'NATIONALUM.NS'],
    'IT_Services': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTIM.NS', 'COFORGE.NS', 'PERSISTENT.NS'],
    'Auto': ['MARUTI.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS', 'ASHOKLEY.NS', 'MOTHERSON.NS'],
    'Realty': ['DLF.NS', 'GODREJPROP.NS', 'OBEROIRLTY.NS', 'PRESTIGE.NS'],
    'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'LUPIN.NS', 'AUROPHARMA.NS'],
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'BANKBARODA.NS', 'FEDERALBNK.NS'],
    'Energy': ['RELIANCE.NS', 'ONGC.NS', 'NTPC.NS', 'TATAPOWER.NS', 'POWERGRID.NS'],
    'Consumer': ['HINDUNILVR.NS', 'ITC.NS', 'TITAN.NS', 'TRENT.NS', 'DABUR.NS', 'BRITANNIA.NS']
}

class DNA3Pyramid:
    def __init__(self, years=10):
        self.years = years
        self.data_cache = {}
        self.capital = INITIAL_CAPITAL
        self.positions = {}  # {ticker: {entry, shares, peak, stop, pyramid_count, avg_price}}
        self.history = []
        self.trade_log = []
        
    def fetch_data(self):
        print(f"[DNA-3 PYRAMID] Fetching {self.years}+ years of data...")
        start_date = (datetime.now() - timedelta(days=365*self.years + 300)).strftime('%Y-%m-%d')
        
        nifty = yf.Ticker("^NSEI").history(start=start_date)
        nifty.index = nifty.index.tz_localize(None)
        self.data_cache['NIFTY'] = nifty
        
        all_tickers = [t for sector in UNIVERSE.values() for t in sector]
        loaded = 0
        for t in all_tickers:
            try:
                df = yf.Ticker(t).history(start=start_date)
                if not df.empty and len(df) > 500:
                    df.index = df.index.tz_localize(None)
                    self.data_cache[t] = df
                    loaded += 1
            except: pass
        print(f"   Loaded {loaded} symbols")

    def get_price(self, ticker, date):
        if ticker not in self.data_cache: return None
        df = self.data_cache[ticker]
        mask = df.index <= date
        if mask.sum() == 0: return None
        return df.loc[mask, 'Close'].iloc[-1]

    def passes_dna_filter(self, ticker, date):
        if ticker not in self.data_cache: return False, 0
        df = self.data_cache[ticker]
        nifty = self.data_cache['NIFTY']
        
        idx = df.index.searchsorted(date)
        if idx < 252: return False, 0
        
        window = df.iloc[max(0, idx-252):idx+1]
        nifty_idx = nifty.index.searchsorted(date)
        nifty_window = nifty.iloc[max(0, nifty_idx-252):nifty_idx+1]
        
        if len(window) < 100: return False, 0
        
        price = window['Close'].iloc[-1]
        
        # RS > 2%
        ret_3m = (price - window['Close'].iloc[-63]) / window['Close'].iloc[-63] * 100 if len(window) > 63 else 0
        nifty_ret_3m = 0
        if len(nifty_window) > 63:
            nifty_ret_3m = (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-63]) / nifty_window['Close'].iloc[-63] * 100
        rs_3m = ret_3m - nifty_ret_3m
        
        if rs_3m < 2.0: return False, 0
        
        # Volatility > 30%
        stock_returns = window['Close'].pct_change().dropna()[-60:]
        volatility = stock_returns.std() * np.sqrt(252) * 100 if len(stock_returns) > 10 else 0
        
        if volatility < 30: return False, 0
        
        # Price > MA50
        ma50 = window['Close'].rolling(50).mean().iloc[-1] if len(window) > 50 else price
        if price < ma50: return False, 0
        
        return True, rs_3m

    def run(self):
        self.fetch_data()
        nifty = self.data_cache['NIFTY']
        start_idx = nifty.index.searchsorted(datetime.now() - timedelta(days=365*self.years))
        dates = nifty.index[start_idx:]
        
        print(f"[DNA-3 PYRAMID] Running {self.years}-Year Backtest...")
        
        for date in dates:
            # === PYRAMIDING on existing positions ===
            for t, pos in list(self.positions.items()):
                price = self.get_price(t, date)
                if not price: continue
                
                # Update peak
                if price > pos['peak']: pos['peak'] = price
                
                # Check for pyramid opportunity
                gain = (price - pos['avg_price']) / pos['avg_price']
                if gain >= PYRAMID_TRIGGER and pos['pyramid_count'] < 3:
                    # Add 50% more shares
                    add_size = (pos['shares'] * pos['avg_price']) * PYRAMID_SIZE
                    add_shares = int(add_size / price)
                    if add_shares > 0:
                        cost = add_shares * price * 1.005
                        if self.capital >= cost:
                            self.capital -= cost
                            # Update position
                            total_shares = pos['shares'] + add_shares
                            total_cost = pos['shares']*pos['avg_price'] + add_shares*price
                            pos['avg_price'] = total_cost / total_shares
                            pos['shares'] = total_shares
                            pos['pyramid_count'] += 1
            
            # === EXITS ===
            to_exit = []
            for t, pos in self.positions.items():
                price = self.get_price(t, date)
                if not price: continue
                
                ret = (price - pos['avg_price']) / pos['avg_price']
                
                # Trailing Stop
                if ret > TRAILING_ACTIVATION:
                    trail = pos['peak'] * (1 - TRAILING_AMOUNT)
                    if trail > pos['stop']: pos['stop'] = trail
                
                if price < pos['stop']:
                    val = pos['shares'] * price * 0.995
                    self.capital += val
                    pnl = (price - pos['avg_price']) / pos['avg_price'] * 100
                    self.trade_log.append({'ticker': t, 'pnl': pnl, 'pyramids': pos['pyramid_count']})
                    to_exit.append(t)
            
            for t in to_exit: del self.positions[t]
            
            # === NEW ENTRIES ===
            if len(self.positions) < MAX_POSITIONS:
                candidates = []
                for sector, tickers in UNIVERSE.items():
                    for ticker in tickers:
                        if ticker in self.positions: continue
                        passes, rs = self.passes_dna_filter(ticker, date)
                        if passes:
                            candidates.append({'ticker': ticker, 'rs': rs})
                
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
                                    'entry': price, 'avg_price': price,
                                    'stop': price * (1 + STOP_LOSS),
                                    'peak': price, 'shares': shares,
                                    'pyramid_count': 0
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
        print(f"[DNA-3 PYRAMID] RESULTS ({self.years} Years)")
        print("="*60)
        print(f"Total Return: {total_ret:.2f}%")
        print(f"Nifty Return: {n_ret:.2f}%")
        print(f"ALPHA: {total_ret - n_ret:.2f}%")
        print(f"CAGR: {((final_val/INITIAL_CAPITAL)**(1/self.years) - 1)*100:.2f}%")
        print(f"Nifty CAGR: {((1 + n_ret/100)**(1/self.years) - 1)*100:.2f}%")
        
        wins = [t for t in self.trade_log if t['pnl'] > 0]
        losses = [t for t in self.trade_log if t['pnl'] <= 0]
        if self.trade_log:
            print(f"\nTrades: {len(self.trade_log)} | Win Rate: {len(wins)/len(self.trade_log)*100:.1f}%")
            if wins: print(f"Avg Win: {np.mean([t['pnl'] for t in wins]):.1f}%")
            if losses: print(f"Avg Loss: {np.mean([t['pnl'] for t in losses]):.1f}%")
            
            # Pyramid stats
            pyramided = [t for t in self.trade_log if t['pyramids'] > 0]
            print(f"Pyramided Trades: {len(pyramided)} ({len(pyramided)/len(self.trade_log)*100:.1f}%)")

if __name__ == "__main__":
    p = DNA3Pyramid(years=10)
    p.run()
