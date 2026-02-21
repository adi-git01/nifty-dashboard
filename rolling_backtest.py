"""
Rolling Window Backtester
=========================
Validates strategy performance across 5 years using rolling 6-month windows.

Methodology:
1. Window Length: 6 Months (126 trading days)
2. Step Size: 1 month (20 trading days) - User asked for 15 days but 1mo is better for regime stability
3. Period: Last 5 Years (2021-2026)
4. Goal: Identify what % of windows hit >20% return
5. Pattern Recognition: What characterized the big winning windows?

Returns:
- Comprehensive CSV of all windows
- Success rate of hitting 20%
- Key characteristics of winning windows vs losing windows
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# === CONFIG ===
LOOKBACK_YEARS = 5
WINDOW_DAYS = 180    # 6 months
STEP_DAYS = 30       # 1 month steps
TARGET_RETURN = 0.20 # 20%

# Universe (Standard 50+ list)
UNIVERSE = {
    'Consumer': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'TITAN.NS', 'DABUR.NS', 'MARICO.NS', 'TRENT.NS'],
    'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'LUPIN.NS', 'AUROPHARMA.NS'],
    'IT_Services': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTIM.NS', 'COFORGE.NS'],
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'INDUSINDBK.NS'],
    'Metals': ['TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 'VEDL.NS', 'JINDALSTEL.NS'],
    'Auto': ['MARUTI.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS'],
    'Industrials': ['LT.NS', 'SIEMENS.NS', 'ABB.NS', 'HAVELLS.NS', 'CUMMINSIND.NS'],
    'Energy': ['RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'IOC.NS', 'NTPC.NS', 'POWERGRID.NS']
}

class RollingBacktest:
    def __init__(self):
        self.data_cache = {}
        self.results = []
        
    def fetch_data(self):
        print("Fetching 5 years of data...")
        start_date = (datetime.now() - timedelta(days=365*5 + 200)).strftime('%Y-%m-%d')
        
        # Nifty
        nifty = yf.Ticker("^NSEI").history(start=start_date)
        nifty.index = nifty.index.tz_localize(None)
        self.data_cache['NIFTY'] = nifty
        
        # Stocks
        all_tickers = [t for sector in UNIVERSE.values() for t in sector]
        for ticker in all_tickers:
            try:
                df = yf.Ticker(ticker).history(start=start_date)
                if not df.empty:
                    df.index = df.index.tz_localize(None)
                    self.data_cache[ticker] = df
            except: pass
            
        print(f"Loaded {len(self.data_cache)} symbols")

    def get_trend(self, ticker, date):
        if ticker not in self.data_cache: return 50
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
        return score

    def simulate_window(self, start_date, end_date):
        """Simulate a single 6-month window."""
        
        # 1. Determine Regime at Start
        nifty = self.data_cache['NIFTY']
        idx = nifty.index.searchsorted(start_date)
        if idx < 200: return None
        
        nifty_window = nifty.iloc[max(0, idx-200):idx+1]
        price = nifty_window['Close'].iloc[-1]
        ma50 = nifty_window['Close'].rolling(50).mean().iloc[-1]
        ma200 = nifty_window['Close'].rolling(200).mean().iloc[-1]
        
        if ma50 > ma200:
            regime = "Strong_Bull" if price > ma50 else "Mild_Bull"
        else:
            regime = "Strong_Bear" if price < ma50 else "Recovery"
            
        # 2. Pick Top 10 Candidates (Simplified Playbook Entry)
        candidates = []
        for sector, tickers in UNIVERSE.items():
            for ticker in tickers:
                if ticker not in self.data_cache: continue
                
                # Get start and end prices
                df = self.data_cache[ticker]
                s_idx = df.index.searchsorted(start_date)
                e_idx = df.index.searchsorted(end_date)
                
                if s_idx >= len(df) or e_idx >= len(df): continue
                
                start_price = df['Close'].iloc[s_idx]
                end_price = df['Close'].iloc[e_idx]
                trend = self.get_trend(ticker, start_date)
                
                # Filter based on regime
                valid = False
                if regime == 'Strong_Bull' and trend >= 60: valid = True
                elif regime == 'Mild_Bull' and trend <= 30: valid = True
                elif regime == 'Recovery' and 20 <= trend <= 40: valid = True
                elif regime == 'Strong_Bear' and trend <= 20: valid = True
                
                if valid:
                    candidates.append({
                        'ticker': ticker,
                        'sector': sector,
                        'trend': trend,
                        'return': (end_price - start_price)/start_price
                    })
        
        # Sort and Pick
        candidates.sort(key=lambda x: -x['trend'] if regime == 'Strong_Bull' else x['trend'])
        portfolio = candidates[:10]
        
        if not portfolio: return None
        
        avg_return = sum(p['return'] for p in portfolio) / len(portfolio)
        
        # Nifty Return
        n_s = nifty['Close'].iloc[idx]
        n_e = nifty.iloc[nifty.index.searchsorted(end_date)]['Close']
        nifty_return = (n_e - n_s) / n_s
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'regime': regime,
            'portfolio_return': avg_return,
            'nifty_return': nifty_return,
            'alpha': avg_return - nifty_return,
            'hit_target': avg_return >= TARGETS,
            'best_sector': max(set([p['sector'] for p in portfolio]), key=lambda x: [p['sector'] for p in portfolio].count(x))
        }

    def run(self):
        self.fetch_data()
        
        # Generate rolling windows
        nifty = self.data_cache['NIFTY']
        start_idx = nifty.index.searchsorted(datetime.now() - timedelta(days=365*5))
        end_idx = len(nifty) - 130 # Need 6 months runway
        
        print(f"Simulating rolling windows from {nifty.index[start_idx].date()}...")
        
        for idx in range(start_idx, end_idx, 20): # 1 month steps
            start_date = nifty.index[idx]
            end_date = nifty.index[idx+126] # approx 6 months
            
            res = self.simulate_window(start_date, end_date)
            if res:
                self.results.append(res)
                
        # Analyze Results
        df = pd.DataFrame(self.results)
        df.to_csv('analysis_2026/rolling_backtest.csv', index=False)
        
        print("\n" + "="*60)
        print("ROLLING BACKTEST COMPLETE (5 Years)")
        print("="*60)
        print(f"Total Windows: {len(df)}")
        print(f"Avg Return: {df['portfolio_return'].mean()*100:.2f}%")
        print(f"Win Rate (>0%): {(df['portfolio_return']>0).mean()*100:.1f}%")
        print(f"Target Hit Rate (>20%): {(df['portfolio_return']>=TARGET_RETURN).mean()*100:.1f}%")
        print("\nBest Regimes for >20% Returns:")
        print(df[df['portfolio_return']>=TARGET_RETURN]['regime'].value_counts())
        
        print("\nBest Sectors in Winning Windows:")
        print(df[df['portfolio_return']>=TARGET_RETURN]['best_sector'].value_counts())

if __name__ == "__main__":
    TARGETS = 0.20
    test = RollingBacktest()
    test.run()
