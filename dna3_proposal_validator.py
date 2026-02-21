"""
DNA-3 V3 PROPOSAL VALIDATION
=============================
Testing key enhancement proposals from the contemplation:
1. RS >20% threshold (vs current >2%)
2. Stop -12% (vs current -15%)
3. Sector cap 40%
4. Full universe with enhanced filters

Compare all variants head-to-head to validate insights.
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
YEARS = 10

class DNA3ProposalValidator:
    def __init__(self):
        self.data_cache = {}
        self.sector_map = SECTOR_MAP
        
    def fetch_data(self):
        print("[PROPOSAL VALIDATOR] Fetching 10+ years of data...")
        start_date = "2015-01-01"
        
        nifty = yf.Ticker("^NSEI").history(start=start_date)
        nifty.index = nifty.index.tz_localize(None)
        self.data_cache['NIFTY'] = nifty
        
        loaded = 0
        for t in TICKERS[:500]:
            try:
                df = yf.Ticker(t).history(start=start_date)
                if not df.empty and len(df) > 500:
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

    def calculate_metrics(self, ticker, date):
        """Calculate RS, Volatility, MA50 status for a stock at a date."""
        if ticker not in self.data_cache: return None
        df = self.data_cache[ticker]
        nifty = self.data_cache['NIFTY']
        
        idx = df.index.searchsorted(date)
        if idx < 252: return None
        
        window = df.iloc[max(0, idx-252):idx+1]
        nifty_idx = nifty.index.searchsorted(date)
        nifty_window = nifty.iloc[max(0, nifty_idx-252):nifty_idx+1]
        
        if len(window) < 100 or len(nifty_window) < 100: return None
        
        price = window['Close'].iloc[-1]
        
        # RS 3-month
        ret_3m = (price - window['Close'].iloc[-63]) / window['Close'].iloc[-63] * 100 if len(window) > 63 else 0
        nifty_ret = (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-63]) / nifty_window['Close'].iloc[-63] * 100 if len(nifty_window) > 63 else 0
        rs = ret_3m - nifty_ret
        
        # Volatility
        daily_ret = window['Close'].pct_change().dropna()[-60:]
        volatility = daily_ret.std() * np.sqrt(252) * 100 if len(daily_ret) > 10 else 0
        
        # MA50
        ma50 = window['Close'].rolling(50).mean().iloc[-1] if len(window) > 50 else price
        above_ma50 = price > ma50
        
        sector = self.sector_map.get(ticker, 'Unknown')
        
        return {
            'price': price,
            'rs': rs,
            'volatility': volatility,
            'above_ma50': above_ma50,
            'sector': sector
        }

    def run_backtest(self, config):
        """Run a single backtest with given configuration."""
        rs_min = config.get('rs_min', 2.0)
        rs_max = config.get('rs_max', 999)
        vol_min = config.get('vol_min', 30)
        vol_max = config.get('vol_max', 100)
        stop_loss = config.get('stop_loss', -0.15)
        max_positions = config.get('max_positions', 10)
        sector_cap = config.get('sector_cap', 10)  # Max per sector
        trailing_activation = config.get('trailing_activation', 0.10)
        trailing_amount = config.get('trailing_amount', 0.10)
        
        nifty = self.data_cache['NIFTY']
        start_idx = nifty.index.searchsorted(datetime.now() - timedelta(days=365*YEARS))
        dates = nifty.index[start_idx:]
        
        capital = INITIAL_CAPITAL
        positions = {}
        trade_log = []
        
        for date in dates:
            # EXITS
            to_exit = []
            for t, pos in positions.items():
                price = self.get_price(t, date)
                if not price: continue
                
                if price > pos['peak']: pos['peak'] = price
                ret = (price - pos['entry']) / pos['entry']
                
                # Trailing stop
                if ret > trailing_activation:
                    trail = pos['peak'] * (1 - trailing_amount)
                    if trail > pos['stop']: pos['stop'] = trail
                
                if price < pos['stop']:
                    val = pos['shares'] * price * 0.995
                    capital += val
                    trade_log.append({
                        'ticker': t,
                        'pnl': ret * 100,
                        'days': (date - pos['entry_date']).days,
                        'sector': pos['sector']
                    })
                    to_exit.append(t)
            
            for t in to_exit: del positions[t]
            
            # ENTRIES
            if len(positions) < max_positions:
                candidates = []
                sector_count = {}
                for t in positions:
                    sec = positions[t]['sector']
                    sector_count[sec] = sector_count.get(sec, 0) + 1
                
                for ticker in self.data_cache.keys():
                    if ticker == 'NIFTY' or ticker in positions: continue
                    metrics = self.calculate_metrics(ticker, date)
                    if not metrics: continue
                    
                    # Apply filters
                    if metrics['rs'] < rs_min or metrics['rs'] > rs_max: continue
                    if metrics['volatility'] < vol_min or metrics['volatility'] > vol_max: continue
                    if not metrics['above_ma50']: continue
                    
                    # Check sector cap
                    sec = metrics['sector']
                    if sector_count.get(sec, 0) >= sector_cap: continue
                    
                    candidates.append({
                        'ticker': ticker,
                        'rs': metrics['rs'],
                        'vol': metrics['volatility'],
                        'sector': sec,
                        'price': metrics['price']
                    })
                
                candidates.sort(key=lambda x: -x['rs'])
                
                for c in candidates[:max_positions - len(positions)]:
                    price = c['price']
                    size = capital / (max_positions - len(positions) + 2)
                    shares = int(size / price)
                    if shares > 0:
                        cost = shares * price * 1.005
                        if capital >= cost:
                            capital -= cost
                            positions[c['ticker']] = {
                                'entry': price,
                                'stop': price * (1 + stop_loss),
                                'peak': price,
                                'shares': shares,
                                'entry_date': date,
                                'sector': c['sector']
                            }
                            sector_count[c['sector']] = sector_count.get(c['sector'], 0) + 1
        
        # Final value
        final_val = capital
        for t, pos in positions.items():
            p = self.get_price(t, dates[-1])
            if p: final_val += pos['shares'] * p
        
        # Nifty returns
        n_s = nifty.loc[dates[0]]['Close']
        n_e = nifty.loc[dates[-1]]['Close']
        nifty_ret = (n_e - n_s) / n_s * 100
        nifty_cagr = ((n_e / n_s) ** (1/YEARS) - 1) * 100
        
        total_ret = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        cagr = ((final_val / INITIAL_CAPITAL) ** (1/YEARS) - 1) * 100
        
        winners = len([t for t in trade_log if t['pnl'] > 0])
        win_rate = winners / len(trade_log) * 100 if trade_log else 0
        avg_win = np.mean([t['pnl'] for t in trade_log if t['pnl'] > 0]) if winners else 0
        avg_loss = np.mean([t['pnl'] for t in trade_log if t['pnl'] <= 0]) if len(trade_log) - winners > 0 else 0
        
        return {
            'total_return': total_ret,
            'cagr': cagr,
            'nifty_return': nifty_ret,
            'nifty_cagr': nifty_cagr,
            'alpha': total_ret - nifty_ret,
            'trades': len(trade_log),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }

    def run(self):
        self.fetch_data()
        
        print("\n" + "="*100)
        print("DNA-3 PROPOSAL VALIDATION: HEAD-TO-HEAD COMPARISON")
        print("="*100)
        
        # Define test configurations
        configs = {
            'BASELINE (DNA-3 V2)': {
                'rs_min': 2.0, 'rs_max': 999, 'vol_min': 30, 'vol_max': 100,
                'stop_loss': -0.15, 'max_positions': 10, 'sector_cap': 10
            },
            'PROPOSAL 1: RS >20%': {
                'rs_min': 20.0, 'rs_max': 999, 'vol_min': 30, 'vol_max': 100,
                'stop_loss': -0.15, 'max_positions': 10, 'sector_cap': 10
            },
            'PROPOSAL 2: RS >20% + Cap 100%': {
                'rs_min': 20.0, 'rs_max': 100.0, 'vol_min': 30, 'vol_max': 100,
                'stop_loss': -0.15, 'max_positions': 10, 'sector_cap': 10
            },
            'PROPOSAL 3: Stop -12%': {
                'rs_min': 2.0, 'rs_max': 999, 'vol_min': 30, 'vol_max': 100,
                'stop_loss': -0.12, 'max_positions': 10, 'sector_cap': 10
            },
            'PROPOSAL 4: Sector Cap 40%': {
                'rs_min': 2.0, 'rs_max': 999, 'vol_min': 30, 'vol_max': 100,
                'stop_loss': -0.15, 'max_positions': 10, 'sector_cap': 4
            },
            'PROPOSAL 5: Vol Cap 60%': {
                'rs_min': 2.0, 'rs_max': 999, 'vol_min': 30, 'vol_max': 60,
                'stop_loss': -0.15, 'max_positions': 10, 'sector_cap': 10
            },
            'COMBINED V3': {
                'rs_min': 20.0, 'rs_max': 100.0, 'vol_min': 30, 'vol_max': 60,
                'stop_loss': -0.12, 'max_positions': 10, 'sector_cap': 4
            }
        }
        
        results = []
        for name, config in configs.items():
            print(f"\nTesting: {name}...", end=' ')
            result = self.run_backtest(config)
            result['name'] = name
            results.append(result)
            print(f"CAGR: {result['cagr']:.1f}% | Win Rate: {result['win_rate']:.1f}% | Alpha: {result['alpha']:.1f}%")
        
        # Print comparison table
        print("\n" + "="*100)
        print("COMPARISON TABLE")
        print("="*100)
        print(f"\n{'Configuration':<35} {'CAGR':<10} {'Alpha':<12} {'Win Rate':<12} {'Trades':<10} {'Avg Win':<10} {'Avg Loss':<10}")
        print("-"*100)
        
        for r in results:
            print(f"{r['name']:<35} {r['cagr']:>+7.1f}%  {r['alpha']:>+9.1f}%  {r['win_rate']:>9.1f}%  {r['trades']:<10} {r['avg_win']:>+8.1f}%  {r['avg_loss']:>+8.1f}%")
        
        # Find best
        best = max(results, key=lambda x: x['cagr'])
        baseline = results[0]
        
        print("-"*100)
        print(f"\nBEST CONFIGURATION: {best['name']}")
        print(f"  CAGR: {best['cagr']:.1f}% (vs Baseline {baseline['cagr']:.1f}%)")
        print(f"  Improvement: {best['cagr'] - baseline['cagr']:+.1f}% CAGR")
        print(f"  Win Rate: {best['win_rate']:.1f}%")
        print(f"  Alpha vs Nifty: {best['alpha']:.1f}%")
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv('analysis_2026/dna3_proposal_validation.csv', index=False)
        print("\n[SAVED] analysis_2026/dna3_proposal_validation.csv")

if __name__ == "__main__":
    validator = DNA3ProposalValidator()
    validator.run()
