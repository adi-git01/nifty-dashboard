"""
DNA-3 V2.1 CURRENT PORTFOLIO
============================
DNA-3 V2 + Sector Cap 40% (max 4 stocks per sector)
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

MAX_POSITIONS = 10
SECTOR_CAP = 4  # Max 40% = 4 out of 10

class DNA3v21Portfolio:
    def __init__(self):
        self.data_cache = {}
        self.sector_map = SECTOR_MAP
        
    def fetch_data(self):
        print("[DNA-3 V2.1] Fetching current data...")
        start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
        
        nifty = yf.Ticker("^NSEI").history(start=start_date)
        nifty.index = nifty.index.tz_localize(None)
        self.data_cache['NIFTY'] = nifty
        
        loaded = 0
        for t in TICKERS[:500]:
            try:
                df = yf.Ticker(t).history(start=start_date)
                if not df.empty and len(df) > 100:
                    df.index = df.index.tz_localize(None)
                    self.data_cache[t] = df
                    loaded += 1
            except:
                pass
        print(f"   Loaded {loaded} stocks")

    def passes_dna_filter(self, ticker):
        if ticker not in self.data_cache: return False, {}
        df = self.data_cache[ticker]
        nifty = self.data_cache['NIFTY']
        
        if len(df) < 100: return False, {}
        
        price = df['Close'].iloc[-1]
        
        # RS > 2%
        ret_3m = (price - df['Close'].iloc[-63]) / df['Close'].iloc[-63] * 100 if len(df) > 63 else 0
        nifty_ret = (nifty['Close'].iloc[-1] - nifty['Close'].iloc[-63]) / nifty['Close'].iloc[-63] * 100 if len(nifty) > 63 else 0
        rs = ret_3m - nifty_ret
        
        if rs < 2.0: return False, {}
        
        # Volatility > 30%
        daily_ret = df['Close'].pct_change().dropna()[-60:]
        volatility = daily_ret.std() * np.sqrt(252) * 100 if len(daily_ret) > 10 else 0
        if volatility < 30: return False, {}
        
        # Price > MA50
        ma50 = df['Close'].rolling(50).mean().iloc[-1]
        if price < ma50: return False, {}
        
        sector = self.sector_map.get(ticker, 'Unknown')
        
        return True, {
            'price': price,
            'rs': rs,
            'volatility': volatility,
            'sector': sector
        }

    def run(self):
        self.fetch_data()
        
        print("\n[DNA-3 V2.1] Scanning with Sector Cap 40%...")
        
        candidates = []
        for ticker in self.data_cache.keys():
            if ticker == 'NIFTY': continue
            passes, metrics = self.passes_dna_filter(ticker)
            if passes:
                candidates.append({
                    'ticker': ticker.replace('.NS', ''),
                    'ticker_ns': ticker,
                    'sector': metrics['sector'],
                    'price': metrics['price'],
                    'rs': metrics['rs'],
                    'volatility': metrics['volatility']
                })
        
        # Sort by RS (highest first)
        candidates.sort(key=lambda x: -x['rs'])
        
        # Apply sector cap (max 4 per sector)
        portfolio = []
        sector_count = {}
        
        for c in candidates:
            sec = c['sector']
            if sector_count.get(sec, 0) < SECTOR_CAP:
                portfolio.append(c)
                sector_count[sec] = sector_count.get(sec, 0) + 1
                if len(portfolio) == MAX_POSITIONS:
                    break
        
        # Calculate what would have been without cap (for comparison)
        uncapped = candidates[:MAX_POSITIONS]
        
        print("\n" + "="*100)
        print("DNA-3 V2.1 CURRENT PORTFOLIO (with Sector Cap 40%)")
        print("="*100)
        print(f"\n{'#':<3} {'Ticker':<12} {'Sector':<30} {'Price':<12} {'RS':<10} {'Volatility':<12}")
        print("-"*100)
        
        for i, p in enumerate(portfolio, 1):
            print(f"{i:<3} {p['ticker']:<12} {p['sector'][:28]:<30} Rs.{p['price']:<10.2f} {p['rs']:>+7.1f}% {p['volatility']:>9.1f}%")
        
        print("-"*100)
        
        # Sector breakdown
        print("\nSECTOR ALLOCATION:")
        for sec, count in sorted(sector_count.items(), key=lambda x: -x[1]):
            pct = count / len(portfolio) * 100
            print(f"  {sec[:30]:<32} {count} stocks ({pct:.0f}%)")
        
        # Compare with uncapped
        print("\n" + "="*100)
        print("COMPARISON: V2.1 vs UNCAPPED V2")
        print("="*100)
        
        uncapped_sectors = {}
        for c in uncapped:
            sec = c['sector']
            uncapped_sectors[sec] = uncapped_sectors.get(sec, 0) + 1
        
        max_uncapped_concentration = max(uncapped_sectors.values()) / len(uncapped) * 100
        max_capped_concentration = max(sector_count.values()) / len(portfolio) * 100
        
        print(f"\nUncapped V2 max sector concentration: {max_uncapped_concentration:.0f}%")
        print(f"Capped V2.1 max sector concentration: {max_capped_concentration:.0f}%")
        
        # Show which stocks were replaced
        uncapped_tickers = set([c['ticker'] for c in uncapped])
        capped_tickers = set([p['ticker'] for p in portfolio])
        
        dropped = uncapped_tickers - capped_tickers
        added = capped_tickers - uncapped_tickers
        
        if dropped:
            print(f"\nDropped due to sector cap: {', '.join(dropped)}")
        if added:
            print(f"Added due to sector diversification: {', '.join(added)}")
        
        # Save portfolio
        df = pd.DataFrame(portfolio)
        df.to_csv('analysis_2026/dna3_v21_portfolio.csv', index=False)
        print("\n[SAVED] analysis_2026/dna3_v21_portfolio.csv")

if __name__ == "__main__":
    p = DNA3v21Portfolio()
    p.run()
