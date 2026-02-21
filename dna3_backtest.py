"""
DNA-3 STRATEGY BACKTEST
========================
Testing the 3 Alpha Signals discovered from Winner DNA Analysis:
1. Relative Strength: 3-Month RS > Nifty + 3%
2. Volatility: Annualized Vol > 35%
3. Pullback in Uptrend: 10-25% below 52w High AND 50%+ above 52w Low

Period: 5 Years (2021-2026)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# === CONFIG ===
INITIAL_CAPITAL = 1000000
MAX_POSITIONS = 10
STOP_LOSS = -0.15
TRAILING_ACTIVATION = 0.10
TRAILING_AMOUNT = 0.10

UNIVERSE = {
    'Industrials': ['LT.NS', 'SIEMENS.NS', 'ABB.NS', 'HAVELLS.NS', 'CUMMINSIND.NS', 'POLYCAB.NS', 'BEL.NS', 'HAL.NS'],
    'Metals': ['TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 'VEDL.NS', 'JINDALSTEL.NS', 'NMDC.NS'],
    'IT_Services': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTIM.NS', 'COFORGE.NS', 'PERSISTENT.NS'],
    'Auto': ['MARUTI.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS', 'ASHOKLEY.NS'],
    'Consumer': ['HINDUNILVR.NS', 'ITC.NS', 'TITAN.NS', 'TRENT.NS'],
    'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS'],
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'BANKBARODA.NS'],
    'Energy': ['RELIANCE.NS', 'ONGC.NS', 'NTPC.NS', 'TATAPOWER.NS']
}

class DNA3Strategy:
    def __init__(self):
        self.data_cache = {}
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
        
    def fetch_data(self):
        print("[DNA-3] Fetching 5+ years of data...")
        start_date = (datetime.now() - timedelta(days=365*5 + 300)).strftime('%Y-%m-%d')
        
        # Nifty
        nifty = yf.Ticker("^NSEI").history(start=start_date)
        nifty.index = nifty.index.tz_localize(None)
        self.data_cache['NIFTY'] = nifty
        
        # Stocks
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

    def passes_dna_filter(self, ticker, date):
        """Check if stock passes all 3 DNA filters."""
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
        
        # === DNA FILTER 1: RELATIVE STRENGTH ===
        ret_3m = (price - window['Close'].iloc[-63]) / window['Close'].iloc[-63] * 100 if len(window) > 63 else 0
        nifty_ret_3m = 0
        if len(nifty_window) > 63:
            nifty_ret_3m = (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-63]) / nifty_window['Close'].iloc[-63] * 100
        rs_3m = ret_3m - nifty_ret_3m
        
        if rs_3m < 3.0: return False, {}  # MUST outperform Nifty by 3%+
        
        # === DNA FILTER 2: VOLATILITY ===
        stock_returns = window['Close'].pct_change().dropna()
        volatility = stock_returns.std() * np.sqrt(252) * 100
        
        if volatility < 35: return False, {}  # MUST have high volatility
        
        # === DNA FILTER 3: PULLBACK IN UPTREND ===
        high_52w = window['High'].max()
        low_52w = window['Low'].min()
        pct_from_high = (price - high_52w) / high_52w * 100
        pct_from_low = (price - low_52w) / low_52w * 100
        
        # Must be 10-30% below 52w high AND 40%+ above 52w low
        if pct_from_high > -10 or pct_from_high < -30: return False, {}
        if pct_from_low < 40: return False, {}
        
        return True, {
            'rs_3m': rs_3m,
            'volatility': volatility,
            'pct_from_high': pct_from_high,
            'pct_from_low': pct_from_low
        }

    def run(self):
        self.fetch_data()
        nifty = self.data_cache['NIFTY']
        start_idx = nifty.index.searchsorted(datetime.now() - timedelta(days=365*5))
        dates = nifty.index[start_idx:]
        
        print("[DNA-3] Running 5-Year Backtest...")
        
        for date in dates:
            # === EXITS ===
            to_exit = []
            for t, pos in self.positions.items():
                price = self.get_price(t, date)
                if not price: continue
                
                if price > pos['peak']: pos['peak'] = price
                
                # Trail
                ret = (price - pos['entry']) / pos['entry']
                if ret > TRAILING_ACTIVATION:
                    trail = pos['peak'] * (1 - TRAILING_AMOUNT)
                    if trail > pos['stop']: pos['stop'] = trail
                
                should_exit = False
                reason = ""
                if price < pos['stop']:
                    should_exit = True
                    reason = "Stop/Trail"
                
                if should_exit:
                    val = pos['shares'] * price * 0.995
                    self.capital += val
                    pnl = (price - pos['entry']) / pos['entry'] * 100
                    self.trade_log.append({
                        'ticker': t, 'entry': pos['entry'], 'exit': price, 'pnl': pnl,
                        'reason': reason, 'date': date
                    })
                    to_exit.append(t)
            
            for t in to_exit: del self.positions[t]
            
            # === ENTRIES (Apply DNA-3 Filters) ===
            if len(self.positions) < MAX_POSITIONS:
                candidates = []
                for sector, tickers in UNIVERSE.items():
                    for ticker in tickers:
                        if ticker in self.positions: continue
                        
                        passes, metrics = self.passes_dna_filter(ticker, date)
                        if passes:
                            candidates.append({
                                'ticker': ticker,
                                'sector': sector,
                                'rs': metrics['rs_3m'],
                                'vol': metrics['volatility']
                            })
                
                # Sort by Relative Strength (Highest first)
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
                                    'entry': price,
                                    'stop': price * (1 + STOP_LOSS),
                                    'peak': price,
                                    'shares': shares,
                                    'sector': c['sector']
                                }
            
            # Log Value
            val = self.capital
            for t, pos in self.positions.items():
                p = self.get_price(t, date)
                if p: val += pos['shares'] * p
            self.history.append({'date': date, 'value': val})
        
        # Final Report
        df = pd.DataFrame(self.history)
        final_val = df.iloc[-1]['value']
        total_ret = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        
        # Nifty Return
        n_s = nifty.loc[df.iloc[0]['date']]['Close']
        n_e = nifty.loc[df.iloc[-1]['date']]['Close']
        n_ret = (n_e - n_s) / n_s * 100
        
        print("\n" + "="*60)
        print("[DNA-3] STRATEGY RESULTS (5 Years)")
        print("="*60)
        print(f"Final Value: Rs.{final_val:,.0f}")
        print(f"Total Return: {total_ret:.2f}%")
        print(f"Nifty Return: {n_ret:.2f}%")
        print(f"ALPHA: {total_ret - n_ret:.2f}%")
        print(f"CAGR: {((final_val/INITIAL_CAPITAL)**(1/5) - 1)*100:.2f}%")
        print(f"Nifty CAGR: {((1 + n_ret/100)**(1/5) - 1)*100:.2f}%")
        
        # Trade Stats
        wins = [t for t in self.trade_log if t['pnl'] > 0]
        losses = [t for t in self.trade_log if t['pnl'] <= 0]
        print(f"\nTrades: {len(self.trade_log)} | Wins: {len(wins)} | Losses: {len(losses)}")
        if self.trade_log:
            print(f"Win Rate: {len(wins)/len(self.trade_log)*100:.1f}%")
            print(f"Avg Win: {np.mean([t['pnl'] for t in wins]):.1f}%" if wins else "N/A")
            print(f"Avg Loss: {np.mean([t['pnl'] for t in losses]):.1f}%" if losses else "N/A")
        
        df.to_csv('analysis_2026/dna3_backtest.csv', index=False)
        pd.DataFrame(self.trade_log).to_csv('analysis_2026/dna3_trades.csv', index=False)

if __name__ == "__main__":
    strat = DNA3Strategy()
    strat.run()
