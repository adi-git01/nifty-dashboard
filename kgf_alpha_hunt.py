"""
KGF ALPHA MINE: UNCONVENTIONAL FACTOR DISCOVERY
=================================================
Think like Rocky Bhai. The only way out is finding what NOBODY else sees.

RADICAL STRATEGIES TO TEST:

1. THE "SMART MONEY FOOTPRINT"
   - High Volume + Small Price Move = Accumulation
   - Someone is loading up quietly
   - Signal: Volume > 3x Average BUT Price Change < 2%

2. THE "PHOENIX RESURRECTION"
   - Stocks that crashed 40%+ from peak
   - BUT now showing Relative Strength > 0
   - The turnaround play

3. THE "52-WEEK HIGH BREAKOUT MACHINE"
   - Near 52W High (within 5%)
   - New highs = More new highs
   - Momentum begets momentum

4. THE "CORRELATION BREAKER"
   - Stocks whose correlation with Nifty DROPPED
   - Something stock-specific is happening
   - Independent movers = Alpha

5. THE "SQUEEZE PLAY"
   - Low volatility (< 20%) followed by volume spike
   - Compression before explosion
   - The calm before the storm

Let's test all 5 and find the gold.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# FULL UNIVERSE
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

class KGFAlphaHunter:
    def __init__(self):
        self.data_cache = {}
        self.results = {}
        
    def fetch_data(self):
        print("[KGF MINE] Fetching 10 years of data...")
        start_date = (datetime.now() - timedelta(days=365*10 + 300)).strftime('%Y-%m-%d')
        
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
        print(f"   Loaded {loaded} stocks")

    def get_price(self, ticker, date):
        if ticker not in self.data_cache: return None
        df = self.data_cache[ticker]
        mask = df.index <= date
        if mask.sum() == 0: return None
        return df.loc[mask, 'Close'].iloc[-1]

    # ============ STRATEGY 1: SMART MONEY FOOTPRINT ============
    def check_accumulation(self, ticker, date):
        """Volume > 3x Avg BUT Price Change < 2% = Silent Accumulation"""
        if ticker not in self.data_cache: return False
        df = self.data_cache[ticker]
        idx = df.index.searchsorted(date)
        if idx < 50: return False
        
        window = df.iloc[max(0, idx-50):idx+1]
        if len(window) < 20: return False
        
        today_vol = window['Volume'].iloc[-1]
        avg_vol = window['Volume'].iloc[:-1].mean()
        price_change = abs(window['Close'].iloc[-1] - window['Close'].iloc[-2]) / window['Close'].iloc[-2] * 100
        
        # Volume 3x+ but price barely moves = accumulation
        if today_vol > avg_vol * 3 and price_change < 2:
            return True
        return False

    # ============ STRATEGY 2: PHOENIX RESURRECTION ============
    def check_phoenix(self, ticker, date):
        """Crashed 40%+ from peak BUT now RS > 0"""
        if ticker not in self.data_cache: return False
        df = self.data_cache[ticker]
        nifty = self.data_cache['NIFTY']
        
        idx = df.index.searchsorted(date)
        if idx < 252: return False
        
        window = df.iloc[max(0, idx-252):idx+1]
        nifty_idx = nifty.index.searchsorted(date)
        nifty_window = nifty.iloc[max(0, nifty_idx-252):nifty_idx+1]
        
        price = window['Close'].iloc[-1]
        peak = window['High'].max()
        
        # Crashed 40%+ from peak
        crash = (price - peak) / peak * 100
        if crash > -40: return False  # Needs 40%+ crash
        
        # BUT showing positive RS now
        ret_3m = (price - window['Close'].iloc[-63]) / window['Close'].iloc[-63] * 100 if len(window) > 63 else 0
        nifty_ret_3m = 0
        if len(nifty_window) > 63:
            nifty_ret_3m = (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-63]) / nifty_window['Close'].iloc[-63] * 100
        rs = ret_3m - nifty_ret_3m
        
        if rs > 0: return True
        return False

    # ============ STRATEGY 3: 52W HIGH BREAKOUT ============
    def check_52w_breakout(self, ticker, date):
        """Within 5% of 52W High"""
        if ticker not in self.data_cache: return False
        df = self.data_cache[ticker]
        
        idx = df.index.searchsorted(date)
        if idx < 252: return False
        
        window = df.iloc[max(0, idx-252):idx+1]
        price = window['Close'].iloc[-1]
        high_52w = window['High'].max()
        
        pct_from_high = (price - high_52w) / high_52w * 100
        
        # Within 5% of 52W high
        if pct_from_high > -5: return True
        return False

    # ============ STRATEGY 4: CORRELATION BREAKER ============
    def check_correlation_break(self, ticker, date):
        """Correlation with Nifty dropped significantly"""
        if ticker not in self.data_cache: return False
        df = self.data_cache[ticker]
        nifty = self.data_cache['NIFTY']
        
        idx = df.index.searchsorted(date)
        n_idx = nifty.index.searchsorted(date)
        if idx < 126 or n_idx < 126: return False
        
        # Recent 30-day correlation
        stock_ret = df['Close'].iloc[idx-30:idx].pct_change().dropna()
        nifty_ret = nifty['Close'].iloc[n_idx-30:n_idx].pct_change().dropna()
        
        if len(stock_ret) < 20 or len(nifty_ret) < 20: return False
        
        # Align lengths
        min_len = min(len(stock_ret), len(nifty_ret))
        recent_corr = np.corrcoef(stock_ret.values[-min_len:], nifty_ret.values[-min_len:])[0,1]
        
        # Historical 90-day correlation
        stock_ret_old = df['Close'].iloc[idx-90:idx-30].pct_change().dropna()
        nifty_ret_old = nifty['Close'].iloc[n_idx-90:n_idx-30].pct_change().dropna()
        
        if len(stock_ret_old) < 30 or len(nifty_ret_old) < 30: return False
        min_len_old = min(len(stock_ret_old), len(nifty_ret_old))
        old_corr = np.corrcoef(stock_ret_old.values[-min_len_old:], nifty_ret_old.values[-min_len_old:])[0,1]
        
        # Correlation dropped significantly
        if old_corr - recent_corr > 0.3: return True
        return False

    # ============ STRATEGY 5: SQUEEZE PLAY ============
    def check_squeeze(self, ticker, date):
        """Low volatility followed by volume breakout"""
        if ticker not in self.data_cache: return False
        df = self.data_cache[ticker]
        
        idx = df.index.searchsorted(date)
        if idx < 60: return False
        
        window = df.iloc[max(0, idx-60):idx+1]
        
        # Recent 20-day volatility (low = < 25%)
        recent_vol = window['Close'].iloc[-20:].pct_change().std() * np.sqrt(252) * 100
        
        # Volume breakout
        today_vol = window['Volume'].iloc[-1]
        avg_vol = window['Volume'].iloc[-20:-1].mean()
        
        # Low volatility + Volume spike
        if recent_vol < 25 and today_vol > avg_vol * 2:
            return True
        return False

    def backtest_strategy(self, strategy_name, check_func):
        """Generic backtest for any strategy."""
        nifty = self.data_cache['NIFTY']
        start_idx = nifty.index.searchsorted(datetime.now() - timedelta(days=365*10))
        dates = nifty.index[start_idx:]
        
        capital = 1000000
        positions = {}
        history = []
        trades = []
        
        for date in dates:
            # EXITS (Simple: Hold 60 days or -15% stop)
            to_exit = []
            for t, pos in positions.items():
                price = self.get_price(t, date)
                if not price: continue
                
                days_held = (date - pos['entry_date']).days
                ret = (price - pos['entry']) / pos['entry']
                
                if ret < -0.15 or days_held > 60:
                    val = pos['shares'] * price * 0.995
                    capital += val
                    trades.append({'pnl': ret * 100})
                    to_exit.append(t)
            
            for t in to_exit: del positions[t]
            
            # ENTRIES
            if len(positions) < 10:
                for sector, tickers in UNIVERSE.items():
                    if len(positions) >= 10: break
                    for ticker in tickers:
                        if ticker in positions: continue
                        if check_func(ticker, date):
                            price = self.get_price(ticker, date)
                            if price:
                                size = capital / 12
                                shares = int(size / price)
                                if shares > 0:
                                    cost = shares * price * 1.005
                                    if capital >= cost:
                                        capital -= cost
                                        positions[ticker] = {
                                            'entry': price, 'shares': shares,
                                            'entry_date': date
                                        }
            
            val = capital
            for t, pos in positions.items():
                p = self.get_price(t, date)
                if p: val += pos['shares'] * p
            history.append({'date': date, 'value': val})
        
        if not history: return None
        
        df = pd.DataFrame(history)
        final_val = df.iloc[-1]['value']
        total_ret = (final_val - 1000000) / 10000
        
        n_s = nifty.loc[df.iloc[0]['date']]['Close']
        n_e = nifty.loc[df.iloc[-1]['date']]['Close']
        n_ret = (n_e - n_s) / n_s * 100
        
        wins = [t for t in trades if t['pnl'] > 0]
        
        return {
            'strategy': strategy_name,
            'total_return': total_ret,
            'nifty_return': n_ret,
            'alpha': total_ret - n_ret,
            'cagr': ((final_val/1000000)**(1/10) - 1)*100,
            'trades': len(trades),
            'win_rate': len(wins)/len(trades)*100 if trades else 0
        }

    def run_all(self):
        self.fetch_data()
        
        strategies = [
            ("SMART_MONEY_FOOTPRINT", self.check_accumulation),
            ("PHOENIX_RESURRECTION", self.check_phoenix),
            ("52W_HIGH_BREAKOUT", self.check_52w_breakout),
            ("CORRELATION_BREAKER", self.check_correlation_break),
            ("SQUEEZE_PLAY", self.check_squeeze)
        ]
        
        print("\n[KGF MINE] Testing 5 Radical Strategies...")
        print("="*70)
        
        results = []
        for name, func in strategies:
            print(f"Testing {name}...")
            res = self.backtest_strategy(name, func)
            if res:
                results.append(res)
                print(f"  -> Return: {res['total_return']:.1f}%, Alpha: {res['alpha']:.1f}%, Win Rate: {res['win_rate']:.1f}%")
        
        print("\n" + "="*70)
        print("[KGF MINE] STRATEGY LEADERBOARD")
        print("="*70)
        
        results.sort(key=lambda x: -x['alpha'])
        
        for i, r in enumerate(results, 1):
            status = "GOLD" if r['alpha'] > 50 else "SILVER" if r['alpha'] > 0 else "TRASH"
            print(f"#{i} {r['strategy']}: {r['total_return']:.1f}% Return, {r['alpha']:.1f}% Alpha, {r['win_rate']:.1f}% Win Rate [{status}]")
        
        # Save results
        pd.DataFrame(results).to_csv('analysis_2026/kgf_strategies.csv', index=False)
        
        return results

if __name__ == "__main__":
    hunter = KGFAlphaHunter()
    hunter.run_all()
