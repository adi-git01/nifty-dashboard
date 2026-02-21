"""
EARLY MOMENTUM: MULTI-PERIOD BACKTEST
======================================
Test over 1, 5, and 10 years to validate strategy consistency.
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
STOP_LOSS = -0.15
TRAILING_ACTIVATION = 0.10
TRAILING_AMOUNT = 0.10

class EarlyMomentumMultiPeriod:
    def __init__(self, years):
        self.years = years
        self.data_cache = {}
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
        self.sector_map = SECTOR_MAP
        
    def fetch_data(self):
        print(f"\n[EARLY MOMENTUM {self.years}Y] Fetching data...")
        start_date = (datetime.now() - timedelta(days=365*self.years + 365)).strftime('%Y-%m-%d')
        
        nifty = yf.Ticker("^NSEI").history(start=start_date)
        nifty.index = nifty.index.tz_localize(None)
        self.data_cache['NIFTY'] = nifty
        
        loaded = 0
        for t in TICKERS[:500]:
            try:
                df = yf.Ticker(t).history(start=start_date)
                if not df.empty and len(df) > 200:
                    df.index = df.index.tz_localize(None)
                    self.data_cache[t] = df
                    loaded += 1
            except:
                pass
        print(f"   Loaded {loaded} stocks")

    def get_price(self, ticker, date):
        if ticker not in self.data_cache: return None
        df = self.data_cache[ticker]
        mask = df.index <= date
        if mask.sum() == 0: return None
        return df.loc[mask, 'Close'].iloc[-1]

    def calculate_trend_score(self, window):
        if len(window) < 200: return 0
        price = window['Close'].iloc[-1]
        ma50 = window['Close'].rolling(50).mean().iloc[-1]
        ma200 = window['Close'].rolling(200).mean().iloc[-1]
        
        score = 50
        if price > ma50: score += 20
        if price > ma200: score += 15
        if ma50 > ma200: score += 10
        
        ret_1m = (price - window['Close'].iloc[-21]) / window['Close'].iloc[-21] * 100 if len(window) > 21 else 0
        if ret_1m > 5: score += 5
        
        return score

    def passes_filter(self, ticker, date):
        if ticker not in self.data_cache: return False, {}
        df = self.data_cache[ticker]
        nifty = self.data_cache['NIFTY']
        
        idx = df.index.searchsorted(date)
        if idx < 200: return False, {}
        
        window = df.iloc[max(0, idx-252):idx+1]
        nifty_idx = nifty.index.searchsorted(date)
        nifty_window = nifty.iloc[max(0, nifty_idx-252):nifty_idx+1]
        
        if len(window) < 200: return False, {}
        
        price = window['Close'].iloc[-1]
        
        # Trend Score: 65-85
        trend_score = self.calculate_trend_score(window)
        if trend_score < 65 or trend_score > 85: return False, {}
        
        # RS: 0-30%
        ret_3m = (price - window['Close'].iloc[-63]) / window['Close'].iloc[-63] * 100 if len(window) > 63 else 0
        nifty_ret_3m = 0
        if len(nifty_window) > 63:
            nifty_ret_3m = (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-63]) / nifty_window['Close'].iloc[-63] * 100
        rs = ret_3m - nifty_ret_3m
        
        if rs < 0 or rs > 30: return False, {}
        
        # Volatility: 25-50%
        stock_returns = window['Close'].pct_change().dropna()[-60:]
        volatility = stock_returns.std() * np.sqrt(252) * 100 if len(stock_returns) > 10 else 0
        if volatility < 25 or volatility > 50: return False, {}
        
        # Recent MA50 crossover check
        ma50_series = window['Close'].rolling(50).mean()
        ma50 = ma50_series.iloc[-1]
        above_ma50_pct = (price - ma50) / ma50 * 100
        
        recent_crosses = 0
        for i in range(-10, 0):
            if len(window) > abs(i) + 1:
                prev_price = window['Close'].iloc[i-1]
                curr_price = window['Close'].iloc[i]
                prev_ma50 = ma50_series.iloc[i-1] if not np.isnan(ma50_series.iloc[i-1]) else prev_price
                curr_ma50 = ma50_series.iloc[i] if not np.isnan(ma50_series.iloc[i]) else curr_price
                if prev_price < prev_ma50 and curr_price > curr_ma50:
                    recent_crosses += 1
        
        if recent_crosses == 0 and above_ma50_pct > 5: return False, {}
        
        return True, {'rs': rs, 'volatility': volatility, 'sector': self.sector_map.get(ticker, 'Unknown'), 'trend_score': trend_score}

    def select_with_sector_cap(self, candidates):
        candidates.sort(key=lambda x: -x['rs'])
        selected = []
        sector_count = {}
        
        for c in candidates:
            sec = c['sector']
            existing = sum(1 for t in self.positions if self.sector_map.get(t, 'Unknown') == sec)
            if sector_count.get(sec, 0) + existing < SECTOR_CAP:
                selected.append(c)
                sector_count[sec] = sector_count.get(sec, 0) + 1
                if len(selected) + len(self.positions) >= MAX_POSITIONS: break
        
        return selected

    def run(self):
        self.fetch_data()
        nifty = self.data_cache['NIFTY']
        start_idx = nifty.index.searchsorted(datetime.now() - timedelta(days=365*self.years))
        dates = nifty.index[start_idx:]
        
        print(f"[EARLY MOMENTUM {self.years}Y] Running backtest...")
        
        for date in dates:
            # Exits
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
                    self.capital += pos['shares'] * price * 0.995
                    self.trade_log.append({'ticker': t, 'pnl': (price - pos['entry']) / pos['entry'] * 100, 'entry_trend': pos.get('entry_trend', 0)})
                    to_exit.append(t)
            for t in to_exit: del self.positions[t]
            
            # Entries
            if len(self.positions) < MAX_POSITIONS:
                candidates = []
                for ticker in self.data_cache.keys():
                    if ticker == 'NIFTY' or ticker in self.positions: continue
                    passes, metrics = self.passes_filter(ticker, date)
                    if passes:
                        candidates.append({'ticker': ticker, 'sector': metrics['sector'], 'rs': metrics['rs'], 'trend_score': metrics['trend_score']})
                
                for c in self.select_with_sector_cap(candidates)[:MAX_POSITIONS - len(self.positions)]:
                    price = self.get_price(c['ticker'], date)
                    if price:
                        size = self.capital / (MAX_POSITIONS - len(self.positions) + 2)
                        shares = int(size / price)
                        if shares > 0:
                            cost = shares * price * 1.005
                            if self.capital >= cost:
                                self.capital -= cost
                                self.positions[c['ticker']] = {'entry': price, 'stop': price * (1 + STOP_LOSS), 'peak': price, 'shares': shares, 'entry_trend': c['trend_score']}
            
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
        
        wins = [t for t in self.trade_log if t['pnl'] > 0]
        losses = [t for t in self.trade_log if t['pnl'] <= 0]
        win_rate = len(wins)/len(self.trade_log)*100 if self.trade_log else 0
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
        avg_trend = np.mean([t.get('entry_trend', 0) for t in self.trade_log]) if self.trade_log else 0
        
        return {
            'years': self.years,
            'total_ret': total_ret,
            'nifty_ret': n_ret,
            'alpha': total_ret - n_ret,
            'cagr': ((final_val/INITIAL_CAPITAL)**(1/self.years) - 1)*100,
            'nifty_cagr': ((1 + n_ret/100)**(1/self.years) - 1)*100,
            'trades': len(self.trade_log),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trend': avg_trend,
            'final_val': final_val
        }

def main():
    periods = [1, 5, 10]
    results = []
    
    print("="*80)
    print("EARLY MOMENTUM STRATEGY: MULTI-PERIOD BACKTEST")
    print("="*80)
    print("Filters: Trend 65-85, RS 0-30%, Vol 25-50%, Recent MA50 Cross")
    
    for years in periods:
        bt = EarlyMomentumMultiPeriod(years)
        r = bt.run()
        results.append(r)
    
    print("\n" + "="*80)
    print("EARLY MOMENTUM: MULTI-PERIOD RESULTS COMPARISON")
    print("="*80)
    
    print(f"\n{'Period':<12} {'Return':<12} {'Nifty':<12} {'Alpha':<12} {'CAGR':<12} {'Trades':<10} {'Win%':<10} {'AvgWin':<10}")
    print("-"*80)
    for r in results:
        print(f"{r['years']} Year{'':<6} {r['total_ret']:>+.1f}%{'':<4} {r['nifty_ret']:>+.1f}%{'':<4} {r['alpha']:>+.1f}%{'':<4} {r['cagr']:>+.1f}%{'':<5} {r['trades']:<10} {r['win_rate']:.1f}%{'':<3} {r['avg_win']:>+.1f}%")
    
    print("\n" + "-"*80)
    print(f"\n{'PORTFOLIO VALUE':<20}")
    for r in results:
        print(f"  {r['years']}-Year: Rs.{INITIAL_CAPITAL:,.0f} → Rs.{r['final_val']:,.0f} (P&L: Rs.{r['final_val']-INITIAL_CAPITAL:+,.0f})")
    
    print(f"\n{'AVERAGE ENTRY TREND SCORE':<30}")
    for r in results:
        print(f"  {r['years']}-Year: {r['avg_trend']:.1f} (vs DNA3-V2.1 ~93)")
    
    # Save summary
    df = pd.DataFrame(results)
    df.to_csv('analysis_2026/early_momentum_multi_period.csv', index=False)
    print(f"\n[SAVED] analysis_2026/early_momentum_multi_period.csv")

if __name__ == "__main__":
    main()
