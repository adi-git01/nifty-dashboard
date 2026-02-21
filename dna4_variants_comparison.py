"""
DNA4 VARIANTS COMPARISON
========================
Tests all approaches to handle Sideways/Bear markets:

1. DNA4-A: Relaxed Mean Reversion (RSI < 40, drawdown -10% to -35%)
2. DNA4-B: Cash Mode (stay out in Sideways/Bear, only trade in Bull)
3. DNA4-C: Simple Mean Reversion (just RSI < 30, no other filters)
4. DNA4-D: Original (momentum only - baseline)
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
STOP_LOSS_MOMENTUM = -0.15
STOP_LOSS_REVERSION = -0.10
TRAILING_ACTIVATION = 0.10
TRAILING_AMOUNT = 0.10

class DNA4Variant:
    def __init__(self, years, variant='A'):
        """
        variant: 'A' = Relaxed Reversion, 'B' = Cash Mode, 'C' = Simple RSI, 'D' = Momentum Only
        """
        self.years = years
        self.variant = variant
        self.data_cache = {}
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
        self.sector_map = SECTOR_MAP
        
    def fetch_data(self):
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
        return loaded

    def get_price(self, ticker, date):
        if ticker not in self.data_cache: return None
        df = self.data_cache[ticker]
        mask = df.index <= date
        if mask.sum() == 0: return None
        return df.loc[mask, 'Close'].iloc[-1]

    def detect_regime(self, date):
        nifty = self.data_cache['NIFTY']
        idx = nifty.index.searchsorted(date)
        if idx < 200: return 'UNKNOWN'
        
        window = nifty.iloc[max(0, idx-200):idx+1]
        price = window['Close'].iloc[-1]
        ma50 = window['Close'].rolling(50).mean().iloc[-1]
        ma200 = window['Close'].rolling(200).mean().iloc[-1]
        ret_1m = (price - window['Close'].iloc[-21]) / window['Close'].iloc[-21] * 100 if len(window) > 21 else 0
        dist_ma50 = (price - ma50) / ma50 * 100
        
        if price > ma50 and ma50 > ma200 and ret_1m > 3:
            return 'BULL'
        elif price > ma50 and ret_1m > 0:
            return 'MILD_BULL'
        elif abs(dist_ma50) < 2 or (-3 < ret_1m <= 0):
            return 'SIDEWAYS'
        elif price < ma50 and ret_1m < -3:
            return 'BEAR'
        else:
            return 'SIDEWAYS'

    def calculate_rsi(self, window):
        delta = window['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        if loss.iloc[-1] == 0: return 50
        rs = gain.iloc[-1] / loss.iloc[-1]
        return 100 - (100 / (1 + rs))

    def passes_momentum_filter(self, ticker, date):
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
        ma50 = window['Close'].rolling(50).mean().iloc[-1]
        
        if price < ma50: return False, {}
        
        ret_3m = (price - window['Close'].iloc[-63]) / window['Close'].iloc[-63] * 100 if len(window) > 63 else 0
        nifty_ret_3m = 0
        if len(nifty_window) > 63:
            nifty_ret_3m = (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-63]) / nifty_window['Close'].iloc[-63] * 100
        rs = ret_3m - nifty_ret_3m
        if rs < 2.0: return False, {}
        
        stock_returns = window['Close'].pct_change().dropna()[-60:]
        volatility = stock_returns.std() * np.sqrt(252) * 100 if len(stock_returns) > 10 else 0
        if volatility < 30: return False, {}
        
        return True, {'rs': rs, 'volatility': volatility, 'sector': self.sector_map.get(ticker, 'Unknown'), 'strategy': 'MOMENTUM'}

    def passes_reversion_filter_relaxed(self, ticker, date):
        """Variant A: Relaxed Mean Reversion (RSI < 40, drawdown -10% to -35%)"""
        if ticker not in self.data_cache: return False, {}
        df = self.data_cache[ticker]
        
        idx = df.index.searchsorted(date)
        if idx < 200: return False, {}
        
        window = df.iloc[max(0, idx-252):idx+1]
        if len(window) < 100: return False, {}
        
        price = window['Close'].iloc[-1]
        rsi = self.calculate_rsi(window)
        
        # Relaxed: RSI < 40 (was 35)
        if rsi > 40: return False, {}
        
        # Drawdown: -10% to -35% (was -15% to -40%)
        peak_3m = window['Close'].rolling(63).max().iloc[-1]
        drawdown = (price - peak_3m) / peak_3m * 100
        
        if drawdown > -10 or drawdown < -35: return False, {}
        
        return True, {'rsi': rsi, 'drawdown': drawdown, 'sector': self.sector_map.get(ticker, 'Unknown'), 'strategy': 'REVERSION'}

    def passes_reversion_filter_simple(self, ticker, date):
        """Variant C: Simple RSI < 30 only"""
        if ticker not in self.data_cache: return False, {}
        df = self.data_cache[ticker]
        
        idx = df.index.searchsorted(date)
        if idx < 50: return False, {}
        
        window = df.iloc[max(0, idx-100):idx+1]
        if len(window) < 50: return False, {}
        
        rsi = self.calculate_rsi(window)
        if rsi > 30: return False, {}
        
        # Basic sanity: not a total collapse (still > 50% of 52-week high)
        price = window['Close'].iloc[-1]
        high_52w = window['Close'].max()
        if price < high_52w * 0.5: return False, {}  # Avoid falling knives
        
        return True, {'rsi': rsi, 'sector': self.sector_map.get(ticker, 'Unknown'), 'strategy': 'REVERSION'}

    def select_with_sector_cap(self, candidates):
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
        nifty = self.data_cache['NIFTY']
        start_idx = nifty.index.searchsorted(datetime.now() - timedelta(days=365*self.years))
        dates = nifty.index[start_idx:]
        
        mom_trades = 0
        rev_trades = 0
        cash_days = 0
        
        for date in dates:
            regime = self.detect_regime(date)
            use_momentum = regime in ['BULL', 'MILD_BULL']
            
            # Variant B: Cash Mode - don't trade in Sideways/Bear
            if self.variant == 'B' and not use_momentum:
                cash_days += 1
                # Still need to manage existing positions
                pass  # Skip entries but process exits
            
            # === EXITS ===
            to_exit = []
            for t, pos in self.positions.items():
                price = self.get_price(t, date)
                if not price: continue
                
                if price > pos['peak']: pos['peak'] = price
                ret = (price - pos['entry']) / pos['entry']
                
                if pos['strategy'] == 'MOMENTUM':
                    if ret > TRAILING_ACTIVATION:
                        trail = pos['peak'] * (1 - TRAILING_AMOUNT)
                        if trail > pos['stop']: pos['stop'] = trail
                    if price < pos['stop']:
                        to_exit.append(t)
                else:  # REVERSION
                    # Take profit at +15% or stop at -10%
                    if ret > 0.15 or ret < -0.10:
                        to_exit.append(t)
                    # Exit if RSI > 55 (recovered)
                    if t in self.data_cache:
                        df = self.data_cache[t]
                        idx = df.index.searchsorted(date)
                        if idx > 20:
                            window = df.iloc[max(0, idx-30):idx+1]
                            rsi = self.calculate_rsi(window)
                            if rsi > 55:
                                to_exit.append(t)
            
            for t in set(to_exit):
                if t in self.positions:
                    pos = self.positions[t]
                    price = self.get_price(t, date)
                    if price:
                        self.capital += pos['shares'] * price * 0.995
                        pnl = (price - pos['entry']) / pos['entry'] * 100
                        self.trade_log.append({'ticker': t, 'pnl': pnl, 'strategy': pos['strategy']})
                        if pos['strategy'] == 'MOMENTUM': mom_trades += 1
                        else: rev_trades += 1
                    del self.positions[t]
            
            # === ENTRIES ===
            # Variant B: Skip entries in non-bull regimes
            if self.variant == 'B' and not use_momentum:
                val = self.capital
                for t, pos in self.positions.items():
                    p = self.get_price(t, date)
                    if p: val += pos['shares'] * p
                self.history.append({'date': date, 'value': val})
                continue
            
            if len(self.positions) < MAX_POSITIONS:
                candidates = []
                
                for ticker in self.data_cache.keys():
                    if ticker == 'NIFTY' or ticker in self.positions: continue
                    
                    if use_momentum or self.variant == 'D':
                        passes, metrics = self.passes_momentum_filter(ticker, date)
                        if passes: candidates.append({'ticker': ticker, **metrics})
                    else:
                        # Use variant-specific reversion filter
                        if self.variant == 'A':
                            passes, metrics = self.passes_reversion_filter_relaxed(ticker, date)
                        elif self.variant == 'C':
                            passes, metrics = self.passes_reversion_filter_simple(ticker, date)
                        else:
                            passes = False
                        if passes: candidates.append({'ticker': ticker, **metrics})
                
                # Sort
                if use_momentum or self.variant == 'D':
                    candidates.sort(key=lambda x: -x.get('rs', 0))
                else:
                    candidates.sort(key=lambda x: x.get('rsi', 50))  # Lowest RSI first
                
                selected = self.select_with_sector_cap(candidates)
                
                for c in selected[:MAX_POSITIONS - len(self.positions)]:
                    price = self.get_price(c['ticker'], date)
                    if price:
                        stop = STOP_LOSS_MOMENTUM if c['strategy'] == 'MOMENTUM' else STOP_LOSS_REVERSION
                        size = self.capital / (MAX_POSITIONS - len(self.positions) + 2)
                        shares = int(size / price)
                        if shares > 0:
                            cost = shares * price * 1.005
                            if self.capital >= cost:
                                self.capital -= cost
                                self.positions[c['ticker']] = {
                                    'entry': price, 
                                    'stop': price * (1 + stop),
                                    'peak': price, 
                                    'shares': shares,
                                    'strategy': c['strategy']
                                }
            
            val = self.capital
            for t, pos in self.positions.items():
                p = self.get_price(t, date)
                if p: val += pos['shares'] * p
            self.history.append({'date': date, 'value': val})
        
        # Results
        df = pd.DataFrame(self.history)
        final_val = df.iloc[-1]['value']
        total_ret = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        
        n_s = nifty.loc[df.iloc[0]['date']]['Close']
        n_e = nifty.loc[df.iloc[-1]['date']]['Close']
        n_ret = (n_e - n_s) / n_s * 100
        
        wins = [t for t in self.trade_log if t['pnl'] > 0]
        win_rate = len(wins)/len(self.trade_log)*100 if self.trade_log else 0
        
        mom_log = [t for t in self.trade_log if t['strategy'] == 'MOMENTUM']
        rev_log = [t for t in self.trade_log if t['strategy'] == 'REVERSION']
        
        return {
            'variant': self.variant,
            'years': self.years,
            'total_ret': total_ret,
            'nifty_ret': n_ret,
            'alpha': total_ret - n_ret,
            'cagr': ((final_val/INITIAL_CAPITAL)**(1/self.years) - 1)*100,
            'win_rate': win_rate,
            'mom_trades': len(mom_log),
            'rev_trades': len(rev_log),
            'final_val': final_val,
            'cash_days': cash_days if self.variant == 'B' else 0
        }

def main():
    variants = ['A', 'B', 'C', 'D']
    variant_names = {
        'A': 'Relaxed Reversion',
        'B': 'Cash Mode',
        'C': 'Simple RSI<30',
        'D': 'Momentum Only'
    }
    
    periods = [1, 3, 5]  # Skip 10 for speed
    all_results = []
    
    print("="*90)
    print("DNA4 VARIANTS COMPARISON")
    print("="*90)
    print("\nVariant A: Relaxed Mean Reversion (RSI < 40, drawdown -10% to -35%)")
    print("Variant B: Cash Mode (stay out in Sideways/Bear)")
    print("Variant C: Simple Mean Reversion (just RSI < 30)")
    print("Variant D: Momentum Only (baseline)")
    
    # Fetch data once
    print("\n[FETCHING DATA...]")
    base = DNA4Variant(5, 'A')
    base.fetch_data()
    
    for years in periods:
        for var in variants:
            print(f"\n[DNA4-{var}] Running {years}-year backtest...")
            bt = DNA4Variant(years, var)
            bt.data_cache = base.data_cache  # Reuse cached data
            r = bt.run()
            all_results.append(r)
    
    # Summary
    print("\n" + "="*90)
    print("DNA4 VARIANTS: HEAD-TO-HEAD COMPARISON")
    print("="*90)
    
    for years in periods:
        print(f"\n{years}-YEAR RESULTS:")
        print(f"{'Variant':<25} {'Return':<12} {'Alpha':<12} {'CAGR':<10} {'Win%':<10} {'Mom':<8} {'Rev':<8}")
        print("-"*90)
        
        year_results = [r for r in all_results if r['years'] == years]
        year_results.sort(key=lambda x: -x['total_ret'])
        
        for r in year_results:
            name = f"DNA4-{r['variant']} ({variant_names[r['variant']][:15]})"
            print(f"{name:<25} {r['total_ret']:>+.1f}%{'':<4} {r['alpha']:>+.1f}%{'':<4} {r['cagr']:>+.1f}%{'':<3} {r['win_rate']:.1f}%{'':<3} {r['mom_trades']:<8} {r['rev_trades']:<8}")
    
    # Best variant analysis
    print("\n" + "="*90)
    print("BEST VARIANT BY PERIOD")
    print("="*90)
    
    for years in periods:
        year_results = [r for r in all_results if r['years'] == years]
        best = max(year_results, key=lambda x: x['alpha'])
        print(f"  {years}Y: DNA4-{best['variant']} ({variant_names[best['variant']]}) with +{best['alpha']:.1f}% alpha")
    
    # Key insights
    print("\n" + "="*90)
    print("KEY INSIGHTS")
    print("="*90)
    
    # Check if any variant beat baseline (D) in 1Y
    one_year = [r for r in all_results if r['years'] == 1]
    baseline_1y = next(r for r in one_year if r['variant'] == 'D')
    
    better_than_baseline = [r for r in one_year if r['alpha'] > baseline_1y['alpha'] and r['variant'] != 'D']
    
    if better_than_baseline:
        best_1y = max(better_than_baseline, key=lambda x: x['alpha'])
        print(f"\nIn the tough 1-Year period:")
        print(f"  Baseline (Momentum Only): {baseline_1y['alpha']:+.1f}% alpha")
        print(f"  Best Alternative: DNA4-{best_1y['variant']} with {best_1y['alpha']:+.1f}% alpha")
        print(f"  Improvement: {best_1y['alpha'] - baseline_1y['alpha']:+.1f}%")
    else:
        print(f"\nNo variant outperformed momentum-only in 1Y. All struggled equally.")
    
    # Save results
    pd.DataFrame(all_results).to_csv('analysis_2026/dna4_variants_comparison.csv', index=False)
    print(f"\n[SAVED] analysis_2026/dna4_variants_comparison.csv")

if __name__ == "__main__":
    main()
