"""
BEAR WINNERS DEEP DIVE: COMMON FACTORS
======================================
Identifies the Top 50 Winners in 2022 (Bear) and 2025 (Chop) and extracts their 
key statistical properties at the START of the run to find common "Archetypes".

Factors Analyzed:
1. Beta (vs Nifty)
2. Correlation (vs Nifty)
3. Volatility (Annualized)
4. RS Rating (vs Nifty)
5. ADX (Trend Strength)
6. Distance from 52-Week High
7. Volume Trend (Accumulation)
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

class BearWinnerAnalysis:
    def __init__(self):
        self.data_cache = {}
        self.nifty = None
        self.periods = [
            {'name': '2022_BEAR', 'start': '2022-01-01', 'end': '2022-06-30'},
            {'name': '2025_CHOP', 'start': '2025-01-01', 'end': '2026-02-01'}
        ]
        self.sector_map = SECTOR_MAP

    def fetch_data(self):
        print("Fetching Data...")
        # Get enough history for 2022 start
        start_date = "2021-01-01"
        self.nifty = yf.Ticker("^NSEI").history(start=start_date)
        self.nifty.index = self.nifty.index.tz_localize(None)
        
        loaded = 0
        for t in TICKERS[:500]:
            try:
                df = yf.Ticker(t).history(start=start_date)
                if not df.empty and len(df) > 200:
                    df.index = df.index.tz_localize(None)
                    self.data_cache[t] = df
                    loaded += 1
            except: pass
        print(f"Loaded {loaded} stocks.")

    def calculate_factors(self, ticker, date):
        df = self.data_cache[ticker]
        nifty = self.nifty
        
        idx = df.index.searchsorted(date)
        n_idx = nifty.index.searchsorted(date)
        
        if idx < 200: return None
        
        window = df.iloc[max(0, idx-252):idx+1]
        nifty_window = nifty.iloc[max(0, n_idx-252):n_idx+1]
        price = window['Close'].iloc[-1]
        
        # 1. Beta & Correlation (60d)
        rets_stock = window['Close'].pct_change().dropna()[-60:]
        rets_nifty = nifty_window['Close'].pct_change().dropna()[-60:]
        
        common = rets_stock.index.intersection(rets_nifty.index)
        if len(common) > 30:
            cov = np.cov(rets_stock.loc[common], rets_nifty.loc[common])[0][1]
            var = np.var(rets_nifty.loc[common])
            beta = cov / var if var != 0 else 1.0
            corr = rets_stock.loc[common].corr(rets_nifty.loc[common])
        else:
            beta = 1.0; corr = 1.0
            
        # 2. Volatility (Annualized)
        vol = rets_stock.std() * np.sqrt(252) * 100
        
        # 3. RS Rating (vs Nifty 63d)
        price_63 = window['Close'].iloc[-63]
        n_price_63 = nifty_window['Close'].iloc[-63]
        rs_stock = (price - price_63)/price_63
        rs_nifty = (nifty_window['Close'].iloc[-1] - n_price_63)/n_price_63
        rs_score = (rs_stock - rs_nifty) * 100
        
        # 4. ADX (14) - Trend Strength
        high = window['High']
        low = window['Low']
        close = window['Close']
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0 # should be negative diff
        minus_dm = abs(minus_dm)
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='outer').max(axis=1)
        atr = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = dx.rolling(14).mean().iloc[-1]
        
        # 5. Distance from High
        high_52 = window['Close'].max()
        dd = (price - high_52)/high_52 * 100
        
        # 6. Volume Trend
        curr_vol = window['Volume'].rolling(5).mean().iloc[-1]
        avg_vol = window['Volume'].rolling(50).mean().iloc[-1]
        vol_trend = curr_vol / avg_vol if avg_vol > 0 else 1.0
        
        # 7. Volume Consistency (CV)
        vol_cv = window['Volume'].rolling(20).std() / window['Volume'].rolling(20).mean()
        vol_cv_val = vol_cv.iloc[-1] if not np.isnan(vol_cv.iloc[-1]) else 0.5
        
        # 8. ATR % (Stability)
        atr_pct = (adx / 100) * 0.5 # Proxy since we didn't calculate ATR explicitly before, let's allow ADX proxy for now or re-calc
        # Re-calc ATR properly
        tr1 = pd.DataFrame(high - low)
        tr = tr1.rolling(14).mean().iloc[-1]
        atr_pct = (tr / price) * 100
        
        return {
            'beta': beta,
            'corr': corr,
            'volatility': vol,
            'rs_score': rs_score,
            'adx': adx,
            'dd': dd,
            'vol_trend': vol_trend,
            'vol_cv': vol_cv_val,
            'atr_pct': atr_pct,
            'price': price,
        }

    def run_analysis(self):
        all_winners = []
        
        for p in self.periods:
            print(f"\nAnalyzing {p['name']} ({p['start']} to {p['end']})...")
            start = p['start']
            end = p['end']
            
            for t, df in self.data_cache.items():
                if start not in df.index or end not in df.index:
                     # Approximate
                     s_idx = df.index.searchsorted(start)
                     e_idx = df.index.searchsorted(end)
                     if s_idx >= len(df) or e_idx >= len(df): continue
                     s_date = df.index[s_idx]
                     e_date = df.index[e_idx]
                else:
                    s_date = start; e_date = end
                    
                p_start = df.loc[s_date]['Close']
                p_end = df.loc[e_date]['Close']
                ret = (p_end - p_start)/p_start * 100
                
                # Winner Threshold: +30%
                if ret > 30:
                    factors = self.calculate_factors(t, s_date)
                    if factors:
                        all_winners.append({
                            'ticker': t,
                            'period': p['name'],
                            'return': ret,
                            'type': 'Explosive (>100%)' if ret > 100 else 'Steady (30-100%)',
                            'sector': self.sector_map.get(t, 'Unknown'),
                            **factors
                        })
            
        # Analysis by Segment
        df = pd.DataFrame(all_winners)
        if df.empty:
            print("No winners found > 30%.")
            return

        print("\n" + "="*100)
        print("BEAR MARKET WINNERS: SEGMENT ANALYSIS")
        print("="*100)
        
        stats = df.groupby('type').agg({
            'beta': 'mean',
            'corr': 'mean',
            'rs_score': 'mean',
            'adx': 'mean',
            'vol_trend': 'mean',
            'vol_cv': 'mean', # Volume Consistency (Lower is better?)
            'atr_pct': 'mean', # Price Stability (Lower is better?)
            'dd': 'mean',
            'return': ['count', 'mean']
        }).round(2)
        
        print(stats.to_string())
        
        print("\n--- TOP STEADY COMPOUNDERS (30-60% Return) ---")
        steady = df[(df['return'] > 30) & (df['return'] < 60)].sort_values('return', ascending=False).head(10)
        print(steady[['ticker', 'return', 'beta', 'corr', 'atr_pct', 'vol_cv']].to_string(index=False))

        df.to_csv('analysis_2026/bear_winners_deep_dive_v2.csv', index=False)
        print("\nSaved to analysis_2026/bear_winners_deep_dive_v2.csv")

if __name__ == "__main__":
    analyst = BearWinnerAnalysis()
    analyst.fetch_data()
    analyst.run_analysis()
