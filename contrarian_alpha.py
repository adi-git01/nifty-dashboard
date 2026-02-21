"""
THE CONTRARIAN ALPHA: BUY THE FEAR
===================================
Everyone buys strength. Rocky Bhai buys FEAR.

THE CONTRARIAN RULES:
1. Quality Stock (Large Cap, High Liquidity)
2. Crashed 25-40% from Peak (Fear kicks in)
3. BUT Still above MA200 (Long-term uptrend intact)
4. Volume Spike (Capitulation happening)

This is DEGEN. But it might work.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

INITIAL_CAPITAL = 1000000
MAX_POSITIONS = 8  # Fewer positions, higher conviction
STOP_LOSS = -0.20  # Wider stop for contrarian plays
TARGET_PROFIT = 0.30  # 30% Target

# ONLY QUALITY LARGE CAPS
UNIVERSE = {
    'Mega_Cap': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'KOTAKBANK.NS'],
    'Large_Cap_Quality': ['LT.NS', 'AXISBANK.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'BAJFINANCE.NS', 'WIPRO.NS', 'HCLTECH.NS', 'NTPC.NS', 'POWERGRID.NS'],
    'Blue_Chip_Industrial': ['SIEMENS.NS', 'ABB.NS', 'HAVELLS.NS', 'PIDILITIND.NS', 'ASIANPAINT.NS', 'DRREDDY.NS', 'CIPLA.NS', 'M&M.NS', 'TATASTEEL.NS', 'HINDALCO.NS']
}

class ContrarianAlpha:
    def __init__(self, years=10):
        self.years = years
        self.data_cache = {}
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
        
    def fetch_data(self):
        print(f"[CONTRARIAN] Fetching {self.years}+ years of data...")
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

    def passes_contrarian_filter(self, ticker, date):
        """Buy when quality stock crashes but long-term trend intact."""
        if ticker not in self.data_cache: return False, 0
        df = self.data_cache[ticker]
        
        idx = df.index.searchsorted(date)
        if idx < 252: return False, 0
        
        window = df.iloc[max(0, idx-252):idx+1]
        if len(window) < 200: return False, 0
        
        price = window['Close'].iloc[-1]
        peak = window['High'].max()
        ma200 = window['Close'].rolling(200).mean().iloc[-1]
        
        # Crash from peak
        crash = (price - peak) / peak * 100
        
        # Volume spike (capitulation)
        today_vol = window['Volume'].iloc[-1]
        avg_vol = window['Volume'].iloc[-20:-1].mean()
        vol_spike = today_vol / avg_vol
        
        # CONTRARIAN CONDITIONS:
        # 1. Crashed 25-40% from peak
        # 2. STILL above 200 MA (long-term intact)
        # 3. Volume spike > 2x (capitulation)
        
        if -40 < crash < -25 and price > ma200 and vol_spike > 2:
            score = abs(crash) + vol_spike * 10  # More crash + more vol = higher score
            return True, score
        
        return False, 0

    def run(self):
        self.fetch_data()
        nifty = self.data_cache['NIFTY']
        start_idx = nifty.index.searchsorted(datetime.now() - timedelta(days=365*self.years))
        dates = nifty.index[start_idx:]
        
        print(f"[CONTRARIAN] Running {self.years}-Year Backtest...")
        
        for date in dates:
            # EXITS
            to_exit = []
            for t, pos in self.positions.items():
                price = self.get_price(t, date)
                if not price: continue
                
                ret = (price - pos['entry']) / pos['entry']
                
                # Target or Stop
                if ret >= TARGET_PROFIT or ret <= STOP_LOSS:
                    val = pos['shares'] * price * 0.995
                    self.capital += val
                    self.trade_log.append({'ticker': t, 'pnl': ret * 100})
                    to_exit.append(t)
            
            for t in to_exit: del self.positions[t]
            
            # ENTRIES
            if len(self.positions) < MAX_POSITIONS:
                candidates = []
                for sector, tickers in UNIVERSE.items():
                    for ticker in tickers:
                        if ticker in self.positions: continue
                        passes, score = self.passes_contrarian_filter(ticker, date)
                        if passes:
                            candidates.append({'ticker': ticker, 'score': score})
                
                candidates.sort(key=lambda x: -x['score'])
                
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
                                    'entry': price, 'shares': shares
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
        print(f"[CONTRARIAN] RESULTS ({self.years} Years)")
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
    c = ContrarianAlpha(years=10)
    c.run()
