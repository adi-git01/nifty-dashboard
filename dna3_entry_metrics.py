"""
DNA-3 V2 FULL UNIVERSE: CURRENT PORTFOLIO WITH ENTRY METRICS
=============================================================
Shows all entry conditions: Trend Score, PE, RSI, Volume, Volatility
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

MAX_POSITIONS = 20

class DNA3EntryMetrics:
    def __init__(self):
        self.data_cache = {}
        self.info_cache = {}
        self.sector_map = SECTOR_MAP
        
    def fetch_data(self):
        print("[DNA-3 V2] Fetching data with fundamentals...")
        start_date = (datetime.now() - timedelta(days=365 + 100)).strftime('%Y-%m-%d')
        
        nifty = yf.Ticker("^NSEI").history(start=start_date)
        nifty.index = nifty.index.tz_localize(None)
        self.data_cache['NIFTY'] = nifty
        
        loaded = 0
        for t in TICKERS[:500]:
            try:
                ticker_obj = yf.Ticker(t)
                df = ticker_obj.history(start=start_date)
                if not df.empty and len(df) > 100:
                    df.index = df.index.tz_localize(None)
                    self.data_cache[t] = df
                    try:
                        info = ticker_obj.info
                        self.info_cache[t] = {
                            'pe': info.get('trailingPE', None),
                            'forwardPE': info.get('forwardPE', None),
                            'marketCap': info.get('marketCap', None)
                        }
                    except:
                        self.info_cache[t] = {'pe': None}
                    loaded += 1
            except:
                pass
        print(f"   Loaded {loaded} stocks with fundamentals")

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None

    def calculate_volume_ratio(self, volumes):
        """Calculate 20-day/50-day volume ratio."""
        if len(volumes) < 50: return 1.0
        vol_20 = volumes[-20:].mean()
        vol_50 = volumes[-50:].mean()
        return vol_20 / vol_50 if vol_50 > 0 else 1.0

    def get_entry_metrics(self, ticker, entry_idx, df):
        """Get all metrics at a specific entry point."""
        if entry_idx < 100: return None
        
        window = df.iloc[max(0, entry_idx-252):entry_idx+1]
        nifty = self.data_cache['NIFTY']
        nifty_idx = nifty.index.searchsorted(df.index[entry_idx])
        nifty_window = nifty.iloc[max(0, nifty_idx-252):nifty_idx+1]
        
        price = window['Close'].iloc[-1]
        
        # Trend Score
        ma50 = window['Close'].rolling(50).mean().iloc[-1] if len(window) > 50 else price
        ma200 = window['Close'].rolling(200).mean().iloc[-1] if len(window) > 200 else price
        trend_score = 50
        if price > ma50: trend_score += 20
        if price > ma200: trend_score += 15
        if ma50 > ma200: trend_score += 10
        
        # RS
        ret_3m = (price - window['Close'].iloc[-63]) / window['Close'].iloc[-63] * 100 if len(window) > 63 else 0
        nifty_ret = (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-63]) / nifty_window['Close'].iloc[-63] * 100 if len(nifty_window) > 63 else 0
        rs = ret_3m - nifty_ret
        
        # Volatility
        daily_ret = window['Close'].pct_change().dropna()[-60:]
        volatility = daily_ret.std() * np.sqrt(252) * 100 if len(daily_ret) > 10 else 0
        
        # RSI
        rsi = self.calculate_rsi(window['Close'])
        
        # Volume Ratio
        volume_ratio = self.calculate_volume_ratio(window['Volume'])
        
        # PE (from info - current, not historical)
        pe = self.info_cache.get(ticker, {}).get('pe', None)
        
        return {
            'trend_score': trend_score,
            'rs': rs,
            'volatility': volatility,
            'rsi': rsi,
            'volume_ratio': volume_ratio,
            'pe': pe
        }

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
        
        # Find entry point (first time it passed filter in last 6 months)
        entry_idx = None
        for i in range(min(126, len(df)-100), 0, -1):
            idx = len(df) - i
            entry_metrics = self.get_entry_metrics(ticker, idx, df)
            if entry_metrics and entry_metrics['rs'] >= 2.0 and entry_metrics['volatility'] >= 30:
                test_price = df['Close'].iloc[idx]
                test_ma50 = df['Close'].iloc[max(0,idx-50):idx].mean()
                if test_price > test_ma50:
                    entry_idx = idx
                    break
        
        if entry_idx is None:
            entry_idx = len(df) - 63
        
        entry_metrics = self.get_entry_metrics(ticker, entry_idx, df)
        
        return True, {
            'current_price': price,
            'entry_date': df.index[entry_idx],
            'entry_price': df['Close'].iloc[entry_idx],
            'entry_trend': entry_metrics['trend_score'],
            'entry_rs': entry_metrics['rs'],
            'entry_volatility': entry_metrics['volatility'],
            'entry_rsi': entry_metrics['rsi'],
            'entry_volume_ratio': entry_metrics['volume_ratio'],
            'entry_pe': entry_metrics['pe'],
            'current_rs': rs,
            'sector': self.sector_map.get(ticker, 'Unknown')
        }

    def run(self):
        self.fetch_data()
        
        print("\n[DNA-3 V2] Scanning for current positions with entry metrics...")
        
        candidates = []
        for ticker in self.data_cache.keys():
            if ticker == 'NIFTY': continue
            passes, metrics = self.passes_dna_filter(ticker)
            if passes:
                current_return = (metrics['current_price'] - metrics['entry_price']) / metrics['entry_price'] * 100
                days_held = (datetime.now() - metrics['entry_date']).days
                
                candidates.append({
                    'ticker': ticker.replace('.NS', ''),
                    'sector': metrics['sector'][:20] if metrics['sector'] else 'Unknown',
                    'entry_date': metrics['entry_date'].strftime('%Y-%m-%d'),
                    'entry_price': round(metrics['entry_price'], 2),
                    'current_price': round(metrics['current_price'], 2),
                    'return_pct': round(current_return, 1),
                    'days_held': days_held,
                    'entry_trend': int(metrics['entry_trend']),
                    'entry_pe': round(metrics['entry_pe'], 1) if metrics['entry_pe'] else None,
                    'entry_rsi': round(metrics['entry_rsi'], 1) if metrics['entry_rsi'] else None,
                    'entry_vol_ratio': round(metrics['entry_volume_ratio'], 2),
                    'entry_volatility': round(metrics['entry_volatility'], 1),
                    'entry_rs': round(metrics['entry_rs'], 1)
                })
        
        candidates.sort(key=lambda x: -x['entry_rs'])
        portfolio = candidates[:MAX_POSITIONS]
        
        print("\n" + "="*140)
        print("DNA-3 V2 CURRENT PORTFOLIO: ENTRY METRICS")
        print("="*140)
        print(f"\n{'#':<3} {'Ticker':<12} {'Entry Date':<12} {'Entry Rs':<10} {'Now Rs':<10} {'Ret%':<8} {'Days':<6} {'TrendS':<8} {'PE':<8} {'RSI':<8} {'VolRatio':<10} {'Vol%':<8} {'RS%':<8}")
        print("-"*140)
        
        for i, p in enumerate(portfolio, 1):
            pe_str = f"{p['entry_pe']:.1f}" if p['entry_pe'] else "N/A"
            rsi_str = f"{p['entry_rsi']:.1f}" if p['entry_rsi'] else "N/A"
            print(f"{i:<3} {p['ticker']:<12} {p['entry_date']:<12} {p['entry_price']:<10} {p['current_price']:<10} {p['return_pct']:>+6.1f}%  {p['days_held']:<6} {p['entry_trend']:<8} {pe_str:<8} {rsi_str:<8} {p['entry_vol_ratio']:<10} {p['entry_volatility']:<8} {p['entry_rs']:>+6.1f}%")
        
        print("-"*140)
        
        # Summary stats
        avg_return = np.mean([p['return_pct'] for p in portfolio])
        avg_trend = np.mean([p['entry_trend'] for p in portfolio])
        avg_pe = np.mean([p['entry_pe'] for p in portfolio if p['entry_pe']])
        avg_rsi = np.mean([p['entry_rsi'] for p in portfolio if p['entry_rsi']])
        avg_vol_ratio = np.mean([p['entry_vol_ratio'] for p in portfolio])
        avg_volatility = np.mean([p['entry_volatility'] for p in portfolio])
        avg_rs = np.mean([p['entry_rs'] for p in portfolio])
        
        print("\nENTRY METRICS SUMMARY:")
        print(f"  Avg Trend Score at Entry: {avg_trend:.1f}")
        print(f"  Avg PE at Entry: {avg_pe:.1f}" if avg_pe else "  Avg PE: N/A")
        print(f"  Avg RSI at Entry: {avg_rsi:.1f}" if avg_rsi else "  Avg RSI: N/A")
        print(f"  Avg Volume Ratio at Entry: {avg_vol_ratio:.2f}x")
        print(f"  Avg Volatility at Entry: {avg_volatility:.1f}%")
        print(f"  Avg RS at Entry: {avg_rs:+.1f}%")
        
        print(f"\nPORTFOLIO PERFORMANCE:")
        print(f"  Average Return: {avg_return:+.1f}%")
        winners = len([p for p in portfolio if p['return_pct'] > 0])
        print(f"  Winners: {winners} / {len(portfolio)} ({winners/len(portfolio)*100:.0f}%)")
        
        df = pd.DataFrame(portfolio)
        df.to_csv('analysis_2026/dna3_entry_metrics.csv', index=False)
        print("\n[SAVED] analysis_2026/dna3_entry_metrics.csv")

if __name__ == "__main__":
    d = DNA3EntryMetrics()
    d.run()
