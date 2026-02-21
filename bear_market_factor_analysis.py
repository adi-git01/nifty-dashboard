"""
BEAR MARKET FACTOR ANALYSIS (10-YEAR DEEP DIVE)
===============================================
Identifies Bear/Sideways periods in the last 10 years and analyzes the characteristics of the Top 50 winners in each period.

Goals:
1. Identify Regimes: 2015-16, 2018, 2020 Crash, 2021-22 Correction, 2025 Chop.
2. Find Winners: Stocks with >0% return while Nifty fell.
3. Extract DNA: Avg Beta, RS, Volatility, RSI, ADX at the *start* of the period.
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

class FactorAnalysis:
    def __init__(self):
        self.data_cache = {}
        self.nifty = None
        self.regimes = []
        
    def fetch_data(self):
        print("Fetching 13 years of data...")
        start_date = "2013-01-01"
        self.nifty = yf.Ticker("^NSEI").history(start=start_date)
        self.nifty.index = self.nifty.index.tz_localize(None)
        
        loaded = 0
        for t in TICKERS[:500]:
            try:
                df = yf.Ticker(t).history(start=start_date)
                if not df.empty and len(df) > 500:
                    df.index = df.index.tz_localize(None)
                    self.data_cache[t] = df
                    loaded += 1
            except: pass
        print(f"Loaded {loaded} stocks.")

    def identify_regimes(self):
        """Identify Bear/Sideways periods where Nifty < MA200 or Drawdown > 10%"""
        df = self.nifty.copy()
        df['MA200'] = df['Close'].rolling(200).mean()
        df['Peak'] = df['Close'].cummax()
        df['Drawdown'] = (df['Close'] - df['Peak']) / df['Peak'] * 100
        
        # Simple Logic: If Drawdown < -8% OR Price < MA200 for > 20 days -> Potential Bear/Choppy
        is_stress = (df['Drawdown'] < -8) | (df['Close'] < df['MA200'])
        
        # Group into contiguous periods
        periods = []
        start = None
        
        for date, stress in is_stress.items():
            if stress and start is None:
                start = date
            elif not stress and start is not None:
                # End of period
                duration = (date - start).days
                if duration > 60: # meaningful correction (2 months+)
                    # Check depth
                    dd_max = df.loc[start:date, 'Drawdown'].min()
                    periods.append({
                        'start': start,
                        'end': date,
                        'duration': duration,
                        'max_dd': dd_max
                    })
                start = None
                
        # Filter overlapping or minor ones
        final_periods = []
        for p in periods:
            if p['max_dd'] < -10: # Significant drop
                final_periods.append(p)
                
        self.regimes = final_periods
        print(f"\nIdentified {len(final_periods)} Stress Periods:")
        for p in final_periods:
            print(f"  {p['start'].date()} to {p['end'].date()} ({p['duration']} days, DD: {p['max_dd']:.1f}%)")

    def analyze_winners(self):
        all_factors = []
        
        for regime in self.regimes:
            start = regime['start']
            end = regime['end']
            print(f"\nAnalyzing Period: {start.date()} to {end.date()}...")
            
            # Find Winners
            performers = []
            
            # Calculate Nifty Return
            nifty_start = self.nifty.loc[start]['Close']
            nifty_end = self.nifty.loc[end]['Close']
            nifty_ret = (nifty_end - nifty_start) / nifty_start * 100
            
            for t, df in self.data_cache.items():
                if start not in df.index or end not in df.index:
                    # Finds nearest dates
                    s_idx = df.index.searchsorted(start)
                    e_idx = df.index.searchsorted(end)
                    if s_idx >= len(df) or e_idx >= len(df): continue
                    s_date = df.index[s_idx]
                    e_date = df.index[e_idx]
                else:
                    s_date = start
                    e_date = end
                    
                price_start = df.loc[s_date]['Close']
                price_end = df.loc[e_date]['Close']
                ret = (price_end - price_start) / price_start * 100
                
                # We want stocks that POSITIVELY performed or significantly beat Nifty
                if ret > 5: # Absolute winner
                    # Calculate Pre-Period Factors (at Start Date)
                    window = df.loc[:s_date].tail(252)
                    if len(window) < 200: continue
                    
                    # 1. Volatility
                    rets = window['Close'].pct_change().tail(60)
                    vol = rets.std() * np.sqrt(252) * 100
                    
                    # 2. Beta
                    nifty_window = self.nifty.loc[:s_date].tail(60)
                    n_rets = nifty_window['Close'].pct_change()
                    # Align
                    common = rets.index.intersection(n_rets.index)
                    if len(common) > 30:
                        cov = np.cov(rets.loc[common], n_rets.loc[common])[0][1]
                        var = np.var(n_rets.loc[common])
                        beta = cov / var
                    else:
                        beta = 1.0
                        
                    # 3. RS (Relative Strength vs Nifty - 63d/1yr)
                    price_63 = window['Close'].iloc[-63] if len(window)>63 else window['Close'].iloc[0]
                    n_price_63 = self.nifty.loc[:s_date].iloc[-63]['Close']
                    rs_stock = (price_start - price_63)/price_63
                    rs_nifty = (nifty_start - n_price_63)/n_price_63
                    rs_score = (rs_stock - rs_nifty) * 100
                    
                    # 4. RSI (14)
                    delta = window['Close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
                    rsi = 100 - (100/(1+gain/loss)) if loss!=0 else 50
                    
                    # 5. Distance from 52w High (Drawdown)
                    high_52 = window['Close'].max()
                    dd_start = (price_start - high_52) / high_52 * 100
                    
                    performers.append({
                        'ticker': t,
                        'period': f"{start.year}",
                        'return': ret,
                        'beta': beta,
                        'volatility': vol,
                        'rs_score': rs_score,
                        'rsi': rsi,
                        'dd_start': dd_start
                    })
            
            # Analyze Top 50 winners
            performers.sort(key=lambda x: -x['return'])
            top_50 = performers[:50]
            
            if top_50:
                print(f"  Top Winner: {top_50[0]['ticker']} ({top_50[0]['return']:.1f}%)")
                
                avg_beta = np.mean([x['beta'] for x in top_50])
                avg_vol = np.mean([x['volatility'] for x in top_50])
                avg_rs = np.mean([x['rs_score'] for x in top_50])
                avg_rsi = np.mean([x['rsi'] for x in top_50])
                avg_dd = np.mean([x['dd_start'] for x in top_50])
                
                all_factors.append({
                    'period': f"{start.date()} ({int(nifty_ret)}%)",
                    'avg_beta': avg_beta,
                    'avg_vol': avg_vol,
                    'avg_rs': avg_rs,
                    'avg_rsi': avg_rsi,
                    'avg_dd_start': avg_dd,
                    'count': len(top_50)
                })
        
        return pd.DataFrame(all_factors)

def main():
    analyzer = FactorAnalysis()
    analyzer.fetch_data()
    analyzer.identify_regimes()
    df_factors = analyzer.analyze_winners()
    
    print("\n" + "="*100)
    print("WINNING FACTORS IN BEAR MARKETS (Last 10Y)")
    print("="*100)
    print(df_factors.to_string(index=False))
    
    df_factors.to_csv('analysis_2026/bear_factors_10y.csv', index=False)
    print("\nSaved to analysis_2026/bear_factors_10y.csv")

if __name__ == "__main__":
    main()
