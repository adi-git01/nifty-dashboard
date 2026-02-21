"""
KGF ALPHA FUSION: MULTI-FACTOR POWERHOUSE
==========================================
Standalone strategies failed. Time to COMBINE.

THE FUSION FORMULA:
DNA-3 V2 (Proven Winner) +
52W Breakout (Momentum) +
Volume Confirmation (Smart Money)

RULES:
1. RS > 2% vs Nifty (DNA Factor)
2. Within 10% of 52W High (Breakout Factor)
3. Volume > 1.5x Average (Confirmation Factor)
4. Price > MA50 (Trend Filter)

When ALL 4 align = HIGH CONVICTION ENTRY
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

class AlphaFusion:
    def __init__(self, years=10):
        self.years = years
        self.data_cache = {}
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
        
    def fetch_data(self):
        print(f"[ALPHA FUSION] Fetching {self.years}+ years of data...")
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

    def passes_fusion_filter(self, ticker, date):
        """FUSION: All 4 factors must align."""
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
        
        # === FACTOR 1: RELATIVE STRENGTH (DNA) ===
        ret_3m = (price - window['Close'].iloc[-63]) / window['Close'].iloc[-63] * 100 if len(window) > 63 else 0
        nifty_ret_3m = 0
        if len(nifty_window) > 63:
            nifty_ret_3m = (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-63]) / nifty_window['Close'].iloc[-63] * 100
        rs_3m = ret_3m - nifty_ret_3m
        
        if rs_3m < 2.0: return False, 0
        
        # === FACTOR 2: 52W BREAKOUT ===
        high_52w = window['High'].max()
        pct_from_high = (price - high_52w) / high_52w * 100
        
        if pct_from_high < -10: return False, 0  # Must be within 10% of 52W high
        
        # === FACTOR 3: VOLUME CONFIRMATION ===
        today_vol = window['Volume'].iloc[-1]
        avg_vol = window['Volume'].iloc[-20:-1].mean()
        
        if today_vol < avg_vol * 1.5: return False, 0  # Volume 1.5x+ required
        
        # === FACTOR 4: TREND (Price > MA50) ===
        ma50 = window['Close'].rolling(50).mean().iloc[-1] if len(window) > 50 else price
        
        if price < ma50: return False, 0
        
        # FUSION SCORE (Higher RS = Higher score)
        fusion_score = rs_3m + (100 + pct_from_high) + (today_vol / avg_vol * 10)
        
        return True, fusion_score

    def run(self):
        self.fetch_data()
        nifty = self.data_cache['NIFTY']
        start_idx = nifty.index.searchsorted(datetime.now() - timedelta(days=365*self.years))
        dates = nifty.index[start_idx:]
        
        print(f"[ALPHA FUSION] Running {self.years}-Year Backtest...")
        
        for date in dates:
            # EXITS
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
                    self.trade_log.append({'ticker': t, 'pnl': pnl})
                    to_exit.append(t)
            
            for t in to_exit: del self.positions[t]
            
            # ENTRIES
            if len(self.positions) < MAX_POSITIONS:
                candidates = []
                for sector, tickers in UNIVERSE.items():
                    for ticker in tickers:
                        if ticker in self.positions: continue
                        passes, score = self.passes_fusion_filter(ticker, date)
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
        print(f"[ALPHA FUSION] RESULTS ({self.years} Years)")
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
    fusion = AlphaFusion(years=10)
    fusion.run()
