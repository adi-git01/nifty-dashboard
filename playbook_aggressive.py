"""
Aggressive Alpha Backtest Engine
================================
Goal: Target 20%+ return in 6 months
Strategy:
- High Concentration (Max 7 stocks)
- Heavy Sector Bets (Max 40% per sector)
- Momentum Focus (Prioritize Trend > 70)
- Pyramiding (Add to winners)
- Fast Cutting (Initial stop -8%)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# === CONFIG ===
START_DATE = "2025-08-01"
END_DATE = "2026-02-07"
INITIAL_CAPITAL = 1000000
MAX_POSITIONS = 7
POSITION_SIZE = 0.14  # ~14% per position
SLIPPAGE = 0.005
SECTOR_CAP = 0.45  # Aggressive sector bet

# Aggressive Stops
INITIAL_STOP = -0.08  # Tight initial stop
TRAILING_ACTIVATION = 0.05
TRAILING_AMOUNT = 0.10  # Loose trail to let winners run

# Stock Universe (Same as before)
UNIVERSE = {
    'Consumer': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'TITAN.NS', 'DABUR.NS', 'MARICO.NS', 'TRENT.NS', 'COLPAL.NS', 'GODREJCP.NS'],
    'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'LUPIN.NS', 'AUROPHARMA.NS', 'TORNTPHARM.NS', 'ALKEM.NS'],
    'IT_Services': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTIM.NS', 'COFORGE.NS', 'PERSISTENT.NS', 'LTTS.NS', 'MPHASIS.NS'],
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'INDUSINDBK.NS', 'FEDERALBNK.NS', 'BANKBARODA.NS', 'PNB.NS'],
    'Metals': ['TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 'VEDL.NS', 'JINDALSTEL.NS', 'NMDC.NS'],
    'Auto': ['MARUTI.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS', 'ASHOKLEY.NS', 'BALKRISIND.NS'],
    'Industrials': ['LT.NS', 'SIEMENS.NS', 'ABB.NS', 'HAVELLS.NS', 'CUMMINSIND.NS', 'POLYCAB.NS', 'CROMPTON.NS', 'VOLTAS.NS', 'BHARATFORG.NS'],
    'Energy': ['RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'IOC.NS', 'NTPC.NS', 'POWERGRID.NS', 'TATAPOWER.NS', 'GAIL.NS']
}

SECTOR_PRIORITY = {
    'Strong_Bull': ['Metals', 'Auto', 'Banking', 'Industrials', 'Energy'], # Added Banking/Energy
    'Mild_Bull': ['Consumer', 'Pharma', 'IT_Services', 'Banking'],
    'Recovery': ['IT_Services', 'Banking', 'Industrials'],
    'Strong_Bear': ['Auto', 'Metals', 'Defensive']
}

class AggressiveBacktest:
    def __init__(self):
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.trade_log = []
        self.portfolio_history = []
        self.data_cache = {}
        
    def fetch_data(self):
        print("Fetching data...")
        nifty = yf.Ticker("^NSEI").history(start="2025-01-01", end=END_DATE)
        nifty.index = nifty.index.tz_localize(None)
        self.data_cache['NIFTY'] = nifty
        
        all_tickers = [t for sector in UNIVERSE.values() for t in sector]
        for ticker in all_tickers:
            try:
                df = yf.Ticker(ticker).history(start="2025-01-01", end=END_DATE)
                if not df.empty:
                    df.index = df.index.tz_localize(None)
                    self.data_cache[ticker] = df
            except: pass
            
    def get_regime(self, date):
        nifty = self.data_cache['NIFTY']
        idx = nifty.index.searchsorted(date)
        if idx < 200: return "Unknown"
        window = nifty.iloc[max(0, idx-200):idx+1]
        price = window['Close'].iloc[-1]
        ma50 = window['Close'].rolling(50).mean().iloc[-1]
        ma200 = window['Close'].rolling(200).mean().iloc[-1]
        if ma50 > ma200: return "Strong_Bull" if price > ma50 else "Mild_Bull"
        else: return "Strong_Bear" if price < ma50 else "Recovery"

    def get_trend(self, ticker, date):
        if ticker not in self.data_cache: return 0
        df = self.data_cache[ticker]
        idx = df.index.searchsorted(date)
        if idx < 200: return 50
        window = df.iloc[max(0, idx-252):idx+1]
        price = window['Close'].iloc[-1]
        ma50 = window['Close'].rolling(50).mean().iloc[-1]
        ma200 = window['Close'].rolling(200).mean().iloc[-1]
        score = 50
        if price > ma50: score += 15
        else: score -= 10
        if price > ma200: score += 15
        else: score -= 15
        if ma50 > ma200: score += 10
        
        # Aggressive Momentum Boost
        high_20 = window['Close'].iloc[-20:].max()
        if price >= high_20: score += 10 # Breakout bonus
        
        return score

    def get_price(self, ticker, date):
        if ticker not in self.data_cache: return None
        df = self.data_cache[ticker]
        mask = df.index <= date
        if mask.sum() == 0: return None
        return df.loc[mask, 'Close'].iloc[-1]

    def find_candidates(self, date, regime):
        candidates = []
        priority = SECTOR_PRIORITY.get(regime, [])
        
        for sector in priority:
            # Check sector cap
            current_exposure = sum(1 for t, p in self.positions.items() if p['sector'] == sector)
            if current_exposure >= int(MAX_POSITIONS * SECTOR_CAP): continue
            
            for ticker in UNIVERSE.get(sector, []):
                if ticker in self.positions: continue
                
                trend = self.get_trend(ticker, date)
                price = self.get_price(ticker, date)
                
                # Aggressive Filtration
                if regime == 'Strong_Bull' and trend >= 70: # Only super momentum
                    candidates.append({'ticker': ticker, 'trend': trend, 'sector': sector, 'price': price})
                elif regime == 'Mild_Bull' and trend <= 20: # Deep value
                    candidates.append({'ticker': ticker, 'trend': trend, 'sector': sector, 'price': price})
                    
        candidates.sort(key=lambda x: -x['trend'] if regime == 'Strong_Bull' else x['trend'])
        return candidates[:MAX_POSITIONS - len(self.positions)]

    def run(self):
        self.fetch_data()
        dates = pd.date_range(START_DATE, END_DATE, freq='B')
        
        for date in dates:
            regime = self.get_regime(date)
            
            # Check Exits
            to_exit = []
            for ticker, pos in self.positions.items():
                price = self.get_price(ticker, date)
                if not price: continue
                
                # Update peak
                if price > pos['peak']: pos['peak'] = price
                
                # Activate Trail
                ret = (price - pos['entry']) / pos['entry']
                if ret > TRAILING_ACTIVATION:
                    new_stop = pos['peak'] * (1 - TRAILING_AMOUNT)
                    if new_stop > pos['stop']: pos['stop'] = new_stop
                
                if price < pos['stop']:
                    to_exit.append(ticker)
            
            for ticker in to_exit:
                price = self.get_price(ticker, date)
                pos = self.positions[ticker]
                proceeds = pos['shares'] * price * (1 - SLIPPAGE)
                self.capital += proceeds
                del self.positions[ticker]
            
            # Find Entries
            if len(self.positions) < MAX_POSITIONS:
                cands = self.find_candidates(date, regime)
                for c in cands:
                    if len(self.positions) >= MAX_POSITIONS: break
                    price = c['price']
                    shares = int((self.capital * POSITION_SIZE) / price)
                    if shares > 0:
                        cost = shares * price * (1+SLIPPAGE)
                        if self.capital >= cost:
                            self.capital -= cost
                            self.positions[c['ticker']] = {
                                'entry': price, 'shares': shares, 
                                'stop': price * (1 + INITIAL_STOP),
                                'peak': price, 'sector': c['sector']
                            }
            
            # Value
            val = self.capital
            for t, p in self.positions.items():
                price = self.get_price(t, date)
                if price: val += p['shares'] * price
            self.portfolio_history.append(val)
            
        print(f"Final Value: Rs.{val:,.0f}")
        ret = (val - INITIAL_CAPITAL)/INITIAL_CAPITAL * 100
        print(f"Return: {ret:.2f}%")

if __name__ == "__main__":
    eng = AggressiveBacktest()
    eng.run()
