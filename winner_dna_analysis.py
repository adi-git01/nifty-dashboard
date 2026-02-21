"""
🧬 WINNER DNA ANALYSIS
=======================
The All-Nighter Deep Dive

Goal: Reverse-engineer stocks that gave 40%+ returns.
Method: For every 6-month window in the last 10 years, find winners and analyze their characteristics AT THE START.

Analysis Dimensions:
1. Trend Score (MA50 vs MA200 position)
2. Volume Pattern (Was there accumulation?)
3. Market Regime (Bull/Bear/Recovery)
4. Valuation (PE relative to sector)
5. Momentum (Recent 1m, 3m returns)
6. Volatility (High beta or low beta?)
7. Sector (Which sectors produce winners?)
8. Price Position (Near 52w High? Low?)

Output: Statistical profile of a "Winner" vs "Loser"
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# === EXPANDED UNIVERSE (100+ Stocks) ===
UNIVERSE = {
    'Consumer': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'TITAN.NS', 'DABUR.NS', 'MARICO.NS', 'TRENT.NS', 'COLPAL.NS', 'GODREJCP.NS', 'PIDILITIND.NS', 'VBL.NS'],
    'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'LUPIN.NS', 'AUROPHARMA.NS', 'TORNTPHARM.NS', 'ALKEM.NS', 'BIOCON.NS'],
    'IT_Services': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTIM.NS', 'COFORGE.NS', 'PERSISTENT.NS', 'LTTS.NS', 'MPHASIS.NS'],
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'INDUSINDBK.NS', 'FEDERALBNK.NS', 'BANKBARODA.NS', 'PNB.NS', 'IDFCFIRSTB.NS'],
    'Metals': ['TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 'VEDL.NS', 'JINDALSTEL.NS', 'NMDC.NS', 'NATIONALUM.NS'],
    'Auto': ['MARUTI.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS', 'ASHOKLEY.NS', 'BALKRISIND.NS', 'MOTHERSON.NS', 'BHARATFORG.NS'],
    'Industrials': ['LT.NS', 'SIEMENS.NS', 'ABB.NS', 'HAVELLS.NS', 'CUMMINSIND.NS', 'POLYCAB.NS', 'CROMPTON.NS', 'VOLTAS.NS', 'THERMAX.NS', 'BEL.NS', 'HAL.NS'],
    'Energy': ['RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'IOC.NS', 'NTPC.NS', 'POWERGRID.NS', 'TATAPOWER.NS', 'GAIL.NS', 'ADANIGREEN.NS'],
    'Realty': ['DLF.NS', 'GODREJPROP.NS', 'OBEROIRLTY.NS', 'PRESTIGE.NS', 'BRIGADE.NS'],
    'Infra': ['ADANIPORTS.NS', 'GMRAIRPORT.NS', 'IRB.NS']
}

class WinnerDNAAnalyzer:
    def __init__(self):
        self.data_cache = {}
        self.winners = []
        self.losers = []
        
    def fetch_data(self):
        print("[DNA] Fetching 10 years of data for 100+ stocks...")
        start_date = (datetime.now() - timedelta(days=365*10 + 300)).strftime('%Y-%m-%d')
        
        # Nifty
        nifty = yf.Ticker("^NSEI").history(start=start_date)
        nifty.index = nifty.index.tz_localize(None)
        self.data_cache['NIFTY'] = nifty
        
        # All Stocks
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
        print(f"   Loaded {loaded} stocks + Nifty")

    def get_characteristics(self, ticker, date):
        """Extract all characteristics of a stock at a given date."""
        if ticker not in self.data_cache: return None
        df = self.data_cache[ticker]
        nifty = self.data_cache['NIFTY']
        
        idx = df.index.searchsorted(date)
        if idx < 252: return None  # Need 1 year of history
        
        window = df.iloc[max(0, idx-252):idx+1]
        nifty_window = nifty.iloc[max(0, nifty.index.searchsorted(date)-252):nifty.index.searchsorted(date)+1]
        
        if len(window) < 200: return None
        
        price = window['Close'].iloc[-1]
        
        # 1. TREND SCORE
        ma20 = window['Close'].rolling(20).mean().iloc[-1]
        ma50 = window['Close'].rolling(50).mean().iloc[-1]
        ma200 = window['Close'].rolling(200).mean().iloc[-1]
        
        trend_score = 50
        if price > ma20: trend_score += 10
        if price > ma50: trend_score += 15
        if price > ma200: trend_score += 15
        if ma50 > ma200: trend_score += 10
        
        # 2. MOMENTUM (Recent Returns)
        ret_1m = (price - window['Close'].iloc[-21]) / window['Close'].iloc[-21] * 100 if len(window) > 21 else 0
        ret_3m = (price - window['Close'].iloc[-63]) / window['Close'].iloc[-63] * 100 if len(window) > 63 else 0
        ret_6m = (price - window['Close'].iloc[-126]) / window['Close'].iloc[-126] * 100 if len(window) > 126 else 0
        
        # 3. VOLUME PATTERN
        avg_vol_20 = window['Volume'].iloc[-20:].mean()
        avg_vol_50 = window['Volume'].iloc[-50:].mean()
        vol_ratio = avg_vol_20 / avg_vol_50 if avg_vol_50 > 0 else 1
        
        # 4. VOLATILITY (Beta Proxy)
        stock_returns = window['Close'].pct_change().dropna()
        volatility = stock_returns.std() * np.sqrt(252) * 100  # Annualized
        
        # 5. PRICE POSITION (52-Week High/Low)
        high_52w = window['High'].max()
        low_52w = window['Low'].min()
        pct_from_high = (price - high_52w) / high_52w * 100
        pct_from_low = (price - low_52w) / low_52w * 100
        
        # 6. MARKET REGIME
        n_price = nifty_window['Close'].iloc[-1] if len(nifty_window) > 0 else 0
        n_ma50 = nifty_window['Close'].rolling(50).mean().iloc[-1] if len(nifty_window) > 50 else n_price
        n_ma200 = nifty_window['Close'].rolling(200).mean().iloc[-1] if len(nifty_window) > 200 else n_price
        
        if n_ma50 > n_ma200:
            regime = "Strong_Bull" if n_price > n_ma50 else "Mild_Bull"
        else:
            regime = "Strong_Bear" if n_price < n_ma50 else "Recovery"
        
        # 7. RELATIVE STRENGTH (vs Nifty)
        nifty_ret_3m = 0
        if len(nifty_window) > 63:
            nifty_ret_3m = (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-63]) / nifty_window['Close'].iloc[-63] * 100
        rs_3m = ret_3m - nifty_ret_3m
        
        return {
            'trend_score': trend_score,
            'ret_1m': ret_1m,
            'ret_3m': ret_3m,
            'ret_6m': ret_6m,
            'vol_ratio': vol_ratio,
            'volatility': volatility,
            'pct_from_high': pct_from_high,
            'pct_from_low': pct_from_low,
            'regime': regime,
            'rs_3m': rs_3m
        }

    def analyze_windows(self):
        """Analyze every 6-month window in the last 10 years."""
        print("[ANALYSIS] Analyzing 6-month windows...")
        
        nifty = self.data_cache['NIFTY']
        start_idx = 300  # Skip initial period for data availability
        end_idx = len(nifty) - 130
        
        all_tickers = [t for sector, tickers in UNIVERSE.items() for t in tickers]
        
        # Process every 2 months (step of ~40 trading days)
        for idx in range(start_idx, end_idx, 40):
            start_date = nifty.index[idx]
            end_date = nifty.index[min(idx + 126, len(nifty) - 1)]
            
            for sector, tickers in UNIVERSE.items():
                for ticker in tickers:
                    if ticker not in self.data_cache: continue
                    
                    df = self.data_cache[ticker]
                    s_idx = df.index.searchsorted(start_date)
                    e_idx = df.index.searchsorted(end_date)
                    
                    if s_idx >= len(df) or e_idx >= len(df): continue
                    if s_idx < 252: continue  # Need history
                    
                    start_price = df['Close'].iloc[s_idx]
                    end_price = df['Close'].iloc[e_idx]
                    
                    ret = (end_price - start_price) / start_price * 100
                    
                    # Get characteristics AT THE START
                    chars = self.get_characteristics(ticker, start_date)
                    if chars is None: continue
                    
                    chars['ticker'] = ticker.replace('.NS', '')
                    chars['sector'] = sector
                    chars['start_date'] = start_date
                    chars['forward_return'] = ret
                    
                    # Classify
                    if ret >= 40:
                        chars['outcome'] = 'MEGA_WINNER'
                        self.winners.append(chars)
                    elif ret >= 20:
                        chars['outcome'] = 'WINNER'
                        self.winners.append(chars)
                    elif ret <= -20:
                        chars['outcome'] = 'LOSER'
                        self.losers.append(chars)
                    else:
                        chars['outcome'] = 'AVERAGE'
                        # Don't store average to keep analysis focused
                        
        print(f"   Found {len(self.winners)} winners and {len(self.losers)} losers")

    def generate_report(self):
        """Generate the DNA Profile Report."""
        print("[REPORT] Generating DNA Report...")
        
        winners_df = pd.DataFrame(self.winners)
        losers_df = pd.DataFrame(self.losers)
        
        # Mega Winners (40%+)
        mega_winners = winners_df[winners_df['outcome'] == 'MEGA_WINNER']
        
        print("\n" + "="*70)
        print("[DNA] WINNER DNA PROFILE (10 Years)")
        print("="*70)
        
        print(f"\nTotal Mega Winners (40%+): {len(mega_winners)}")
        print(f"Total Winners (20%+): {len(winners_df)}")
        print(f"Total Losers (-20%): {len(losers_df)}")
        
        # 1. TREND SCORE ANALYSIS
        print("\n" + "-"*50)
        print("[1] TREND SCORE AT ENTRY")
        print("-"*50)
        print(f"Mega Winners Avg: {mega_winners['trend_score'].mean():.1f}")
        print(f"Winners Avg:      {winners_df['trend_score'].mean():.1f}")
        print(f"Losers Avg:       {losers_df['trend_score'].mean():.1f}")
        
        # Distribution
        print("\nMega Winner Trend Score Distribution:")
        print(mega_winners['trend_score'].value_counts(bins=[0,30,50,70,90,100]).sort_index())
        
        # 2. MOMENTUM ANALYSIS
        print("\n" + "-"*50)
        print("[2] MOMENTUM AT ENTRY")
        print("-"*50)
        print(f"3-Month Return Before Rally:")
        print(f"  Mega Winners: {mega_winners['ret_3m'].mean():.1f}%")
        print(f"  Winners:      {winners_df['ret_3m'].mean():.1f}%")
        print(f"  Losers:       {losers_df['ret_3m'].mean():.1f}%")
        
        # 3. VOLUME PATTERN
        print("\n" + "-"*50)
        print("[3] VOLUME RATIO (20d/50d) AT ENTRY")
        print("-"*50)
        print(f"Mega Winners: {mega_winners['vol_ratio'].mean():.2f}")
        print(f"Winners:      {winners_df['vol_ratio'].mean():.2f}")
        print(f"Losers:       {losers_df['vol_ratio'].mean():.2f}")
        
        # 4. VOLATILITY
        print("\n" + "-"*50)
        print("[4] VOLATILITY (Annualized) AT ENTRY")
        print("-"*50)
        print(f"Mega Winners: {mega_winners['volatility'].mean():.1f}%")
        print(f"Winners:      {winners_df['volatility'].mean():.1f}%")
        print(f"Losers:       {losers_df['volatility'].mean():.1f}%")
        
        # 5. PRICE POSITION
        print("\n" + "-"*50)
        print("[5] PRICE POSITION AT ENTRY")
        print("-"*50)
        print(f"% From 52-Week High:")
        print(f"  Mega Winners: {mega_winners['pct_from_high'].mean():.1f}%")
        print(f"  Winners:      {winners_df['pct_from_high'].mean():.1f}%")
        print(f"  Losers:       {losers_df['pct_from_high'].mean():.1f}%")
        print(f"% From 52-Week Low:")
        print(f"  Mega Winners: {mega_winners['pct_from_low'].mean():.1f}%")
        print(f"  Winners:      {winners_df['pct_from_low'].mean():.1f}%")
        print(f"  Losers:       {losers_df['pct_from_low'].mean():.1f}%")
        
        # 6. REGIME ANALYSIS
        print("\n" + "-"*50)
        print("[6] MARKET REGIME AT ENTRY")
        print("-"*50)
        print("Mega Winners by Regime:")
        print(mega_winners['regime'].value_counts())
        print("\nLosers by Regime:")
        print(losers_df['regime'].value_counts())
        
        # 7. SECTOR ANALYSIS
        print("\n" + "-"*50)
        print("[7] SECTOR PERFORMANCE")
        print("-"*50)
        print("Top Sectors for Mega Winners:")
        print(mega_winners['sector'].value_counts().head(5))
        
        # 8. RELATIVE STRENGTH
        print("\n" + "-"*50)
        print("[8] RELATIVE STRENGTH (vs Nifty) AT ENTRY")
        print("-"*50)
        print(f"Mega Winners 3m RS: {mega_winners['rs_3m'].mean():.1f}%")
        print(f"Winners 3m RS:      {winners_df['rs_3m'].mean():.1f}%")
        print(f"Losers 3m RS:       {losers_df['rs_3m'].mean():.1f}%")
        
        # Save to CSV
        winners_df.to_csv('analysis_2026/mega_winners_dna.csv', index=False)
        losers_df.to_csv('analysis_2026/losers_dna.csv', index=False)
        
        print("\n" + "="*70)
        print("[DONE] CSV files saved to analysis_2026/")
        print("="*70)
        
        return winners_df, losers_df

    def run(self):
        self.fetch_data()
        self.analyze_windows()
        return self.generate_report()

if __name__ == "__main__":
    analyzer = WinnerDNAAnalyzer()
    winners_df, losers_df = analyzer.run()
