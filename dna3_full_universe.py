"""
DNA-3 V2 EXTENDED: FULL NSE UNIVERSE BACKTEST
==============================================
Testing DNA-3 V2 on the FULL Nifty 500 universe (all 500 stocks)
vs the previous 68-stock subset.

This tests if broader selection = better alpha.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os
import sys

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nifty500_list import TICKERS, SECTOR_MAP

warnings.filterwarnings('ignore')

INITIAL_CAPITAL = 1000000
MAX_POSITIONS = 20
STOP_LOSS = -0.15
TRAILING_ACTIVATION = 0.10
TRAILING_AMOUNT = 0.10

class DNA3FullUniverseBacktest:
    def __init__(self, years=10):
        self.years = years
        self.data_cache = {}
        self.sector_map = SECTOR_MAP
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
        self.peak_value = INITIAL_CAPITAL
        self.max_drawdown = 0
        self.stocks_loaded = 0
        
    def fetch_data(self):
        print(f"[DNA-3 V2 FULL UNIVERSE] Fetching {self.years}+ years of data...")
        print(f"   Total Universe: {len(TICKERS)} stocks")
        start_date = (datetime.now() - timedelta(days=365*self.years + 300)).strftime('%Y-%m-%d')
        
        # Nifty
        nifty = yf.Ticker("^NSEI").history(start=start_date)
        nifty.index = nifty.index.tz_localize(None)
        self.data_cache['NIFTY'] = nifty
        
        # Batch download for speed (in chunks to avoid rate limits)
        chunk_size = 50
        all_tickers = TICKERS[:500]  # Use full 500
        
        for i in range(0, len(all_tickers), chunk_size):
            chunk = all_tickers[i:i+chunk_size]
            print(f"   Loading batch {i//chunk_size + 1}/{len(all_tickers)//chunk_size + 1}...", end='\r')
            
            for t in chunk:
                try:
                    df = yf.Ticker(t).history(start=start_date)
                    if not df.empty and len(df) > 500:
                        df.index = df.index.tz_localize(None)
                        self.data_cache[t] = df
                        self.stocks_loaded += 1
                except:
                    pass
        
        print(f"\n   Successfully loaded: {self.stocks_loaded} stocks")

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
        nifty_ret_3m = 0
        if len(nifty_window) > 63:
            nifty_ret_3m = (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-63]) / nifty_window['Close'].iloc[-63] * 100
        rs_3m = ret_3m - nifty_ret_3m
        
        if rs_3m < 2.0: return False, {}
        
        # Volatility > 30%
        stock_returns = window['Close'].pct_change().dropna()[-60:]
        volatility = stock_returns.std() * np.sqrt(252) * 100 if len(stock_returns) > 10 else 0
        
        if volatility < 30: return False, {}
        
        # Price > MA50
        ma50 = window['Close'].rolling(50).mean().iloc[-1] if len(window) > 50 else price
        if price < ma50: return False, {}
        
        # Trend Score
        ma200 = window['Close'].rolling(200).mean().iloc[-1] if len(window) > 200 else price
        trend = 50
        if price > ma50: trend += 20
        if price > ma200: trend += 15
        if ma50 > ma200: trend += 10
        
        return True, {'rs_3m': rs_3m, 'volatility': volatility, 'trend_score': trend}

    def run(self):
        self.fetch_data()
        nifty = self.data_cache['NIFTY']
        start_idx = nifty.index.searchsorted(datetime.now() - timedelta(days=365*self.years))
        dates = nifty.index[start_idx:]
        
        print(f"\n[DNA-3 V2 FULL UNIVERSE] Running {self.years}-Year Backtest...")
        print(f"   Universe: {self.stocks_loaded} stocks")
        print(f"   Max Positions: {MAX_POSITIONS}")
        
        for date in dates:
            # EXITS
            to_exit = []
            for t, pos in self.positions.items():
                price = self.get_price(t, date)
                if not price: continue
                
                if price > pos['peak']: pos['peak'] = price
                
                ret = (price - pos['entry']) / pos['entry']
                exit_reason = None
                
                # Trailing
                if ret > TRAILING_ACTIVATION:
                    trail = pos['peak'] * (1 - TRAILING_AMOUNT)
                    if trail > pos['stop']: pos['stop'] = trail
                
                if price < pos['stop']:
                    exit_reason = 'Stop' if ret < 0 else 'Trail'
                
                if exit_reason:
                    val = pos['shares'] * price * 0.995
                    self.capital += val
                    days_held = (date - pos['entry_date']).days
                    sector = self.sector_map.get(t, 'Unknown')
                    
                    self.trade_log.append({
                        'ticker': t.replace('.NS', ''),
                        'sector': sector,
                        'entry_date': pos['entry_date'],
                        'exit_date': date,
                        'pnl_pct': ret * 100,
                        'days_held': days_held,
                        'entry_rs': pos['entry_rs'],
                        'entry_volatility': pos['entry_volatility'],
                        'entry_trend': pos['entry_trend'],
                        'exit_reason': exit_reason
                    })
                    to_exit.append(t)
            
            for t in to_exit: del self.positions[t]
            
            # ENTRIES
            if len(self.positions) < MAX_POSITIONS:
                candidates = []
                for ticker in self.data_cache.keys():
                    if ticker == 'NIFTY' or ticker in self.positions: continue
                    passes, metrics = self.passes_dna_filter(ticker, date)
                    if passes:
                        candidates.append({
                            'ticker': ticker,
                            'rs': metrics['rs_3m'],
                            'volatility': metrics['volatility'],
                            'trend': metrics['trend_score']
                        })
                
                # Sort by RS (highest first)
                candidates.sort(key=lambda x: -x['rs'])
                
                for c in candidates[:MAX_POSITIONS - len(self.positions)]:
                    price = self.get_price(c['ticker'], date)
                    if price:
                        size = self.capital / (MAX_POSITIONS - len(self.positions) + 2)
                        shares = int(size / price)
                        if shares > 0:
                            cost = shares * price * 1.005
                            if self.capital >= cost:
                                self.capital -= cost
                                self.positions[c['ticker']] = {
                                    'entry': price,
                                    'stop': price * (1 + STOP_LOSS),
                                    'peak': price,
                                    'shares': shares,
                                    'entry_date': date,
                                    'entry_rs': c['rs'],
                                    'entry_volatility': c['volatility'],
                                    'entry_trend': c['trend']
                                }
            
            # Portfolio value and drawdown
            val = self.capital
            for t, pos in self.positions.items():
                p = self.get_price(t, date)
                if p: val += pos['shares'] * p
            
            if val > self.peak_value:
                self.peak_value = val
            current_dd = (val - self.peak_value) / self.peak_value * 100
            if current_dd < self.max_drawdown:
                self.max_drawdown = current_dd
            
            self.history.append({'date': date, 'value': val, 'drawdown': current_dd})
        
        self.generate_report()

    def generate_report(self):
        print("\n" + "="*80)
        print("DNA-3 V2 EXTENDED: FULL NSE UNIVERSE RESULTS")
        print("="*80)
        
        trades_df = pd.DataFrame(self.trade_log)
        history_df = pd.DataFrame(self.history)
        
        if trades_df.empty:
            print("No trades to analyze.")
            return trades_df
        
        # Separate winners and losers
        winners = trades_df[trades_df['pnl_pct'] > 0]
        losers = trades_df[trades_df['pnl_pct'] <= 0]
        
        # === PORTFOLIO METRICS ===
        final_val = history_df.iloc[-1]['value']
        total_ret = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        nifty = self.data_cache['NIFTY']
        n_s = nifty.loc[history_df.iloc[0]['date']]['Close']
        n_e = nifty.loc[history_df.iloc[-1]['date']]['Close']
        n_ret = (n_e - n_s) / n_s * 100
        
        print("\n" + "-"*60)
        print("[1] PORTFOLIO PERFORMANCE")
        print("-"*60)
        print(f"{'Universe Size':<30} {self.stocks_loaded} stocks")
        print(f"{'Total Return':<30} {total_ret:.2f}%")
        print(f"{'Nifty Return':<30} {n_ret:.2f}%")
        print(f"{'ALPHA':<30} {total_ret - n_ret:.2f}%")
        print(f"{'CAGR':<30} {((final_val/INITIAL_CAPITAL)**(1/self.years) - 1)*100:.2f}%")
        print(f"{'Max Drawdown':<30} {self.max_drawdown:.2f}%")
        print(f"{'Total Trades':<30} {len(trades_df)}")
        print(f"{'WIN RATE':<30} {len(winners)/len(trades_df)*100:.1f}%")
        
        # === RETURN STATISTICS ===
        print("\n" + "-"*60)
        print("[2] RETURN STATISTICS")
        print("-"*60)
        print(f"{'Metric':<25} {'All':<12} {'Winners':<12} {'Losers':<12}")
        print(f"{'Average Return':<25} {trades_df['pnl_pct'].mean():.1f}%{'':<5} {winners['pnl_pct'].mean():.1f}%{'':<5} {losers['pnl_pct'].mean():.1f}%")
        print(f"{'Median Return':<25} {trades_df['pnl_pct'].median():.1f}%{'':<5} {winners['pnl_pct'].median():.1f}%{'':<5} {losers['pnl_pct'].median():.1f}%")
        print(f"{'Std Dev':<25} {trades_df['pnl_pct'].std():.1f}%{'':<5} {winners['pnl_pct'].std():.1f}%{'':<5} {losers['pnl_pct'].std():.1f}%")
        print(f"{'Max Win':<25} {trades_df['pnl_pct'].max():.1f}%")
        print(f"{'Max Loss':<25} {trades_df['pnl_pct'].min():.1f}%")
        
        # === HOLDING PERIOD ===
        print("\n" + "-"*60)
        print("[3] HOLDING PERIOD")
        print("-"*60)
        print(f"{'Avg Days':<25} {trades_df['days_held'].mean():.0f}{'':<10} {winners['days_held'].mean():.0f}{'':<10} {losers['days_held'].mean():.0f}")
        print(f"{'Median Days':<25} {trades_df['days_held'].median():.0f}{'':<10} {winners['days_held'].median():.0f}{'':<10} {losers['days_held'].median():.0f}")
        
        # === ENTRY METRICS ===
        print("\n" + "-"*60)
        print("[4] ENTRY METRICS: WINNERS vs LOSERS")
        print("-"*60)
        print(f"{'Metric':<25} {'Winners':<12} {'Losers':<12} {'Edge':<10}")
        w_rs, l_rs = winners['entry_rs'].mean(), losers['entry_rs'].mean()
        w_vol, l_vol = winners['entry_volatility'].mean(), losers['entry_volatility'].mean()
        w_trend, l_trend = winners['entry_trend'].mean(), losers['entry_trend'].mean()
        print(f"{'RS vs Nifty':<25} {w_rs:.1f}%{'':<6} {l_rs:.1f}%{'':<6} {w_rs-l_rs:+.1f}%")
        print(f"{'Volatility':<25} {w_vol:.1f}%{'':<6} {l_vol:.1f}%{'':<6} {w_vol-l_vol:+.1f}%")
        print(f"{'Trend Score':<25} {w_trend:.1f}{'':<6} {l_trend:.1f}{'':<6} {w_trend-l_trend:+.1f}")
        
        # === SECTOR ANALYSIS ===
        print("\n" + "-"*60)
        print("[5] SECTOR PERFORMANCE (Top 10)")
        print("-"*60)
        sector_perf = trades_df.groupby('sector').agg({
            'pnl_pct': ['mean', 'count', lambda x: (x > 0).mean() * 100]
        }).round(2)
        sector_perf.columns = ['Avg PnL', 'Trades', 'Win Rate']
        sector_perf = sector_perf.sort_values('Avg PnL', ascending=False).head(10)
        print(sector_perf.to_string())
        
        # === EXIT ANALYSIS ===
        print("\n" + "-"*60)
        print("[6] EXIT ANALYSIS")
        print("-"*60)
        exit_perf = trades_df.groupby('exit_reason').agg({
            'pnl_pct': ['mean', 'count', 'std']
        }).round(2)
        exit_perf.columns = ['Avg PnL', 'Count', 'Std']
        print(exit_perf.to_string())
        
        # === COMPARISON TO PREVIOUS ===
        print("\n" + "="*80)
        print("[7] COMPARISON: FULL UNIVERSE vs PREVIOUS (68 stocks)")
        print("="*80)
        print(f"{'Metric':<30} {'Full Universe':<20} {'Previous (68)':<20}")
        print(f"{'Stocks in Universe':<30} {self.stocks_loaded:<20} {'68':<20}")
        print(f"{'Total Trades':<30} {len(trades_df):<20} {'572':<20}")
        print(f"{'Total Return':<30} {total_ret:.1f}%{'':<15} {'520.5%':<20}")
        print(f"{'CAGR':<30} {((final_val/INITIAL_CAPITAL)**(1/self.years) - 1)*100:.1f}%{'':<15} {'20.0%':<20}")
        print(f"{'Alpha':<30} {total_ret - n_ret:.1f}%{'':<15} {'252.5%':<20}")
        print(f"{'Win Rate':<30} {len(winners)/len(trades_df)*100:.1f}%{'':<15} {'65.7%':<20}")
        
        # Save trade log
        trades_df.to_csv('analysis_2026/dna3_full_universe_trades.csv', index=False)
        print("\n[SAVED] analysis_2026/dna3_full_universe_trades.csv")
        
        return trades_df

if __name__ == "__main__":
    bt = DNA3FullUniverseBacktest(years=10)
    bt.run()
