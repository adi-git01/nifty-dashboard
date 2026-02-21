"""
SECTOR MOMENTUM ROTATION: THE SECTOR SURFER
=============================================
Forget stock picking. SURF THE SECTOR WAVE.

RULES:
1. Every month, calculate avg RS for each sector
2. Go ALL-IN on the TOP sector (5 stocks)
3. Hold for 1 month, then rotate if needed
4. Pure concentrated momentum

This is the ULTIMATE momentum bet.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

INITIAL_CAPITAL = 1000000

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

class SectorSurfer:
    def __init__(self, years=10):
        self.years = years
        self.data_cache = {}
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
        
    def fetch_data(self):
        print(f"[SECTOR SURFER] Fetching {self.years}+ years of data...")
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

    def get_sector_momentum(self, date):
        """Calculate avg RS for each sector."""
        nifty = self.data_cache['NIFTY']
        nifty_idx = nifty.index.searchsorted(date)
        if nifty_idx < 63: return None
        
        nifty_ret = (nifty['Close'].iloc[nifty_idx] - nifty['Close'].iloc[nifty_idx-63]) / nifty['Close'].iloc[nifty_idx-63] * 100
        
        sector_scores = {}
        for sector, tickers in UNIVERSE.items():
            rs_list = []
            for t in tickers:
                if t not in self.data_cache: continue
                df = self.data_cache[t]
                idx = df.index.searchsorted(date)
                if idx < 63: continue
                
                ret = (df['Close'].iloc[idx] - df['Close'].iloc[idx-63]) / df['Close'].iloc[idx-63] * 100
                rs = ret - nifty_ret
                rs_list.append({'ticker': t, 'rs': rs})
            
            if rs_list:
                sector_scores[sector] = {
                    'avg_rs': np.mean([x['rs'] for x in rs_list]),
                    'stocks': sorted(rs_list, key=lambda x: -x['rs'])[:5]  # Top 5
                }
        
        return sector_scores

    def run(self):
        self.fetch_data()
        nifty = self.data_cache['NIFTY']
        start_idx = nifty.index.searchsorted(datetime.now() - timedelta(days=365*self.years))
        dates = nifty.index[start_idx:]
        
        print(f"[SECTOR SURFER] Running {self.years}-Year Backtest...")
        
        last_rebalance = None
        current_sector = None
        
        for date in dates:
            # Monthly Rebalance (every ~22 trading days)
            if last_rebalance is None or (date - last_rebalance).days >= 30:
                # Sell all current positions
                for t, pos in self.positions.items():
                    price = self.get_price(t, date)
                    if price:
                        val = pos['shares'] * price * 0.995
                        self.capital += val
                        pnl = (price - pos['entry']) / pos['entry'] * 100
                        self.trade_log.append({'pnl': pnl})
                self.positions = {}
                
                # Find top sector and buy top 5 stocks
                sector_scores = self.get_sector_momentum(date)
                if sector_scores:
                    top_sector = max(sector_scores.items(), key=lambda x: x[1]['avg_rs'])
                    current_sector = top_sector[0]
                    top_stocks = top_sector[1]['stocks']
                    
                    for stock in top_stocks:
                        price = self.get_price(stock['ticker'], date)
                        if price:
                            size = self.capital / (len(top_stocks) + 1)
                            shares = int(size / price)
                            if shares > 0:
                                cost = shares * price * 1.005
                                if self.capital >= cost:
                                    self.capital -= cost
                                    self.positions[stock['ticker']] = {
                                        'entry': price, 'shares': shares
                                    }
                
                last_rebalance = date
            
            # Log portfolio value
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
        print(f"[SECTOR SURFER] RESULTS ({self.years} Years)")
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

if __name__ == "__main__":
    surfer = SectorSurfer(years=10)
    surfer.run()
