"""
DNA-3 V2 EXTENDED: 2-YEAR ROLLING PERIOD ANALYSIS
==================================================
Test DNA-3 V2 on full Nifty 500 universe across 5 x 2-year periods.
Analyze how market regime affects returns.

Periods:
- 2016-2018
- 2018-2020
- 2020-2022
- 2022-2024
- 2024-2026
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
MAX_POSITIONS = 20
STOP_LOSS = -0.15
TRAILING_ACTIVATION = 0.10
TRAILING_AMOUNT = 0.10

class DNA3PeriodAnalysis:
    def __init__(self):
        self.data_cache = {}
        self.sector_map = SECTOR_MAP
        self.period_results = []
        
    def fetch_data(self):
        print("[DNA-3 V2 PERIOD ANALYSIS] Fetching 10+ years of data...")
        start_date = "2015-01-01"  # Start from 2015 to have lookback for 2016
        
        nifty = yf.Ticker("^NSEI").history(start=start_date)
        nifty.index = nifty.index.tz_localize(None)
        self.data_cache['NIFTY'] = nifty
        
        loaded = 0
        chunk_size = 50
        all_tickers = TICKERS[:500]
        
        for i in range(0, len(all_tickers), chunk_size):
            chunk = all_tickers[i:i+chunk_size]
            print(f"   Loading batch {i//chunk_size + 1}/{len(all_tickers)//chunk_size + 1}...", end='\r')
            
            for t in chunk:
                try:
                    df = yf.Ticker(t).history(start=start_date)
                    if not df.empty and len(df) > 500:
                        df.index = df.index.tz_localize(None)
                        self.data_cache[t] = df
                        loaded += 1
                except:
                    pass
        
        print(f"\n   Loaded {loaded} stocks")

    def get_price(self, ticker, date):
        if ticker not in self.data_cache: return None
        df = self.data_cache[ticker]
        mask = df.index <= date
        if mask.sum() == 0: return None
        return df.loc[mask, 'Close'].iloc[-1]

    def passes_dna_filter(self, ticker, date):
        if ticker not in self.data_cache: return False, {}
        df = self.data_cache[ticker]
        nifty = self.data_cache['NIFTY']
        
        idx = df.index.searchsorted(date)
        if idx < 252: return False, {}
        
        window = df.iloc[max(0, idx-252):idx+1]
        nifty_idx = nifty.index.searchsorted(date)
        nifty_window = nifty.iloc[max(0, nifty_idx-252):nifty_idx+1]
        
        if len(window) < 100: return False, {}
        
        price = window['Close'].iloc[-1]
        
        # RS > 2%
        ret_3m = (price - window['Close'].iloc[-63]) / window['Close'].iloc[-63] * 100 if len(window) > 63 else 0
        nifty_ret = 0
        if len(nifty_window) > 63:
            nifty_ret = (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-63]) / nifty_window['Close'].iloc[-63] * 100
        rs = ret_3m - nifty_ret
        
        if rs < 2.0: return False, {}
        
        # Volatility > 30%
        daily_ret = window['Close'].pct_change().dropna()[-60:]
        volatility = daily_ret.std() * np.sqrt(252) * 100 if len(daily_ret) > 10 else 0
        if volatility < 30: return False, {}
        
        # Price > MA50
        ma50 = window['Close'].rolling(50).mean().iloc[-1] if len(window) > 50 else price
        if price < ma50: return False, {}
        
        return True, {'rs': rs, 'volatility': volatility}

    def run_period(self, start_year, end_year):
        """Run backtest for a specific 2-year period."""
        nifty = self.data_cache['NIFTY']
        
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        
        # Filter dates
        mask = (nifty.index >= start_date) & (nifty.index <= end_date)
        dates = nifty.index[mask]
        
        if len(dates) == 0:
            return None
        
        capital = INITIAL_CAPITAL
        positions = {}
        history = []
        trade_log = []
        
        for date in dates:
            # EXITS
            to_exit = []
            for t, pos in positions.items():
                price = self.get_price(t, date)
                if not price: continue
                
                if price > pos['peak']: pos['peak'] = price
                ret = (price - pos['entry']) / pos['entry']
                
                if ret > TRAILING_ACTIVATION:
                    trail = pos['peak'] * (1 - TRAILING_AMOUNT)
                    if trail > pos['stop']: pos['stop'] = trail
                
                if price < pos['stop']:
                    val = pos['shares'] * price * 0.995
                    capital += val
                    trade_log.append({'pnl': ret * 100})
                    to_exit.append(t)
            
            for t in to_exit: del positions[t]
            
            # ENTRIES
            if len(positions) < MAX_POSITIONS:
                candidates = []
                for ticker in self.data_cache.keys():
                    if ticker == 'NIFTY' or ticker in positions: continue
                    passes, metrics = self.passes_dna_filter(ticker, date)
                    if passes:
                        candidates.append({'ticker': ticker, 'rs': metrics['rs']})
                
                candidates.sort(key=lambda x: -x['rs'])
                
                for c in candidates[:MAX_POSITIONS - len(positions)]:
                    price = self.get_price(c['ticker'], date)
                    if price:
                        size = capital / (MAX_POSITIONS - len(positions) + 2)
                        shares = int(size / price)
                        if shares > 0:
                            cost = shares * price * 1.005
                            if capital >= cost:
                                capital -= cost
                                positions[c['ticker']] = {
                                    'entry': price,
                                    'stop': price * (1 + STOP_LOSS),
                                    'peak': price,
                                    'shares': shares
                                }
            
            val = capital
            for t, pos in positions.items():
                p = self.get_price(t, date)
                if p: val += pos['shares'] * p
            history.append({'date': date, 'value': val})
        
        if len(history) < 2:
            return None
        
        # Calculate returns
        final_val = history[-1]['value']
        total_ret = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        years = (dates[-1] - dates[0]).days / 365
        cagr = ((final_val / INITIAL_CAPITAL) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Nifty returns for period
        nifty_start = nifty.loc[dates[0]]['Close']
        nifty_end = nifty.loc[dates[-1]]['Close']
        nifty_ret = (nifty_end - nifty_start) / nifty_start * 100
        nifty_cagr = ((nifty_end / nifty_start) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Win rate
        wins = len([t for t in trade_log if t['pnl'] > 0])
        win_rate = wins / len(trade_log) * 100 if trade_log else 0
        
        return {
            'period': f"{start_year}-{end_year}",
            'dna_return': total_ret,
            'dna_cagr': cagr,
            'nifty_return': nifty_ret,
            'nifty_cagr': nifty_cagr,
            'alpha': total_ret - nifty_ret,
            'alpha_cagr': cagr - nifty_cagr,
            'trades': len(trade_log),
            'win_rate': win_rate
        }

    def run(self):
        self.fetch_data()
        
        print("\n[DNA-3 V2 PERIOD ANALYSIS] Running 2-Year Rolling Windows...")
        
        # 5 periods of 2 years each
        periods = [
            (2016, 2017),
            (2018, 2019),
            (2020, 2021),
            (2022, 2023),
            (2024, 2025)  # Partial year
        ]
        
        for start, end in periods:
            print(f"   Testing {start}-{end}...", end=' ')
            result = self.run_period(start, end)
            if result:
                self.period_results.append(result)
                print(f"DONE | DNA: {result['dna_cagr']:.1f}% CAGR | Nifty: {result['nifty_cagr']:.1f}% CAGR | Alpha: {result['alpha_cagr']:+.1f}%")
            else:
                print("SKIPPED (insufficient data)")
        
        self.generate_report()

    def generate_report(self):
        print("\n" + "="*100)
        print("DNA-3 V2 EXTENDED: PERIOD-BY-PERIOD ANALYSIS")
        print("="*100)
        
        print(f"\n{'Period':<12} {'DNA Return':<15} {'DNA CAGR':<12} {'Nifty Return':<15} {'Nifty CAGR':<12} {'Alpha':<12} {'Alpha CAGR':<12} {'Trades':<10} {'Win Rate':<10}")
        print("-"*100)
        
        for r in self.period_results:
            print(f"{r['period']:<12} {r['dna_return']:>+12.1f}%   {r['dna_cagr']:>+10.1f}%  {r['nifty_return']:>+12.1f}%   {r['nifty_cagr']:>+10.1f}%  {r['alpha']:>+10.1f}%  {r['alpha_cagr']:>+10.1f}%  {r['trades']:<10} {r['win_rate']:.1f}%")
        
        print("-"*100)
        
        # Averages
        avg_dna_cagr = np.mean([r['dna_cagr'] for r in self.period_results])
        avg_nifty_cagr = np.mean([r['nifty_cagr'] for r in self.period_results])
        avg_alpha = np.mean([r['alpha_cagr'] for r in self.period_results])
        avg_win_rate = np.mean([r['win_rate'] for r in self.period_results])
        
        print(f"{'AVERAGE':<12} {'':<15} {avg_dna_cagr:>+10.1f}%  {'':<15} {avg_nifty_cagr:>+10.1f}%  {'':<12} {avg_alpha:>+10.1f}%  {'':<10} {avg_win_rate:.1f}%")
        
        # Best and Worst periods
        best = max(self.period_results, key=lambda x: x['dna_cagr'])
        worst = min(self.period_results, key=lambda x: x['dna_cagr'])
        
        print(f"\nBEST PERIOD:  {best['period']} | DNA CAGR: {best['dna_cagr']:+.1f}% | Alpha: {best['alpha_cagr']:+.1f}%")
        print(f"WORST PERIOD: {worst['period']} | DNA CAGR: {worst['dna_cagr']:+.1f}% | Alpha: {worst['alpha_cagr']:+.1f}%")
        
        # Consistency
        positive_alpha = len([r for r in self.period_results if r['alpha_cagr'] > 0])
        print(f"\nPOSITIVE ALPHA PERIODS: {positive_alpha}/{len(self.period_results)} ({positive_alpha/len(self.period_results)*100:.0f}%)")
        
        # Market regime analysis
        print("\n" + "="*100)
        print("REGIME ANALYSIS")
        print("="*100)
        
        for r in self.period_results:
            if r['nifty_cagr'] > 15:
                regime = "STRONG BULL"
            elif r['nifty_cagr'] > 5:
                regime = "MILD BULL"
            elif r['nifty_cagr'] < -5:
                regime = "BEAR"
            else:
                regime = "SIDEWAYS"
            
            outperform = "OUTPERFORMED" if r['alpha_cagr'] > 0 else "UNDERPERFORMED"
            print(f"{r['period']}: {regime:<12} | Nifty {r['nifty_cagr']:+.1f}% | DNA-3 {r['dna_cagr']:+.1f}% | {outperform} by {abs(r['alpha_cagr']):.1f}%")
        
        # Save to CSV
        df = pd.DataFrame(self.period_results)
        df.to_csv('analysis_2026/dna3_period_analysis.csv', index=False)
        print("\n[SAVED] analysis_2026/dna3_period_analysis.csv")

if __name__ == "__main__":
    analyzer = DNA3PeriodAnalysis()
    analyzer.run()
