"""
DNA-3 V2.1: 3-YEAR BACKTEST
============================
DNA-3 V2 + Sector Cap 40% (max 4 stocks per sector in 10-stock portfolio)
Period: Last 3 years
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
SECTOR_CAP = 4  # Max 40% = 4 out of 10
STOP_LOSS = -0.15
TRAILING_ACTIVATION = 0.10
TRAILING_AMOUNT = 0.10
YEARS = 3

class DNA3V21Backtest:
    def __init__(self):
        self.data_cache = {}
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
        self.sector_map = SECTOR_MAP
        
    def fetch_data(self):
        print(f"[DNA-3 V2.1] Fetching data for {YEARS}-year backtest...")
        start_date = (datetime.now() - timedelta(days=365*YEARS + 365)).strftime('%Y-%m-%d')
        
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

    def passes_dna_filter(self, ticker, date):
        """DNA-3 V2 Entry Filters."""
        if ticker not in self.data_cache: return False, {}
        df = self.data_cache[ticker]
        nifty = self.data_cache['NIFTY']
        
        idx = df.index.searchsorted(date)
        if idx < 100: return False, {}
        
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
        rs = ret_3m - nifty_ret_3m
        
        if rs < 2.0: return False, {}
        
        # Volatility > 30%
        stock_returns = window['Close'].pct_change().dropna()[-60:]
        volatility = stock_returns.std() * np.sqrt(252) * 100 if len(stock_returns) > 10 else 0
        
        if volatility < 30: return False, {}
        
        # Price > MA50
        ma50 = window['Close'].rolling(50).mean().iloc[-1] if len(window) > 50 else price
        if price < ma50: return False, {}
        
        sector = self.sector_map.get(ticker, 'Unknown')
        
        return True, {'rs': rs, 'volatility': volatility, 'sector': sector}

    def select_with_sector_cap(self, candidates):
        """Apply sector cap (max 4 per sector) when selecting candidates."""
        candidates.sort(key=lambda x: -x['rs'])
        
        selected = []
        sector_count = {}
        
        for c in candidates:
            sec = c['sector']
            # Count existing positions in this sector
            existing_in_sector = sum(1 for t in self.positions if self.sector_map.get(t, 'Unknown') == sec)
            
            if sector_count.get(sec, 0) + existing_in_sector < SECTOR_CAP:
                selected.append(c)
                sector_count[sec] = sector_count.get(sec, 0) + 1
                if len(selected) + len(self.positions) >= MAX_POSITIONS:
                    break
        
        return selected

    def run(self):
        self.fetch_data()
        nifty = self.data_cache['NIFTY']
        start_idx = nifty.index.searchsorted(datetime.now() - timedelta(days=365*YEARS))
        dates = nifty.index[start_idx:]
        
        print(f"\n[DNA-3 V2.1] Running {YEARS}-Year Backtest with Sector Cap 40%...")
        
        for date in dates:
            # === EXITS ===
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
                    val = pos['shares'] * price * 0.995
                    self.capital += val
                    pnl = (price - pos['entry']) / pos['entry'] * 100
                    self.trade_log.append({
                        'ticker': t, 
                        'entry': pos['entry'],
                        'exit': price,
                        'pnl': pnl, 
                        'date': date,
                        'sector': self.sector_map.get(t, 'Unknown')
                    })
                    to_exit.append(t)
            
            for t in to_exit: del self.positions[t]
            
            # === ENTRIES ===
            if len(self.positions) < MAX_POSITIONS:
                candidates = []
                for ticker in self.data_cache.keys():
                    if ticker == 'NIFTY' or ticker in self.positions: continue
                    
                    passes, metrics = self.passes_dna_filter(ticker, date)
                    if passes:
                        candidates.append({
                            'ticker': ticker, 
                            'sector': metrics['sector'], 
                            'rs': metrics['rs'],
                            'volatility': metrics['volatility']
                        })
                
                # Apply sector cap
                selected = self.select_with_sector_cap(candidates)
                
                for c in selected[:MAX_POSITIONS - len(self.positions)]:
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
                                    'shares': shares
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
        
        print("\n" + "="*70)
        print(f"[DNA-3 V2.1] RESULTS ({YEARS} Years with Sector Cap 40%)")
        print("="*70)
        print(f"\n{'Metric':<25} {'DNA-3 V2.1':<20} {'Nifty 50':<20}")
        print("-"*70)
        print(f"{'Total Return':<25} {total_ret:>+.2f}%{'':<12} {n_ret:>+.2f}%")
        print(f"{'CAGR':<25} {((final_val/INITIAL_CAPITAL)**(1/YEARS) - 1)*100:>+.2f}%{'':<12} {((1 + n_ret/100)**(1/YEARS) - 1)*100:>+.2f}%")
        print(f"{'Alpha':<25} {total_ret - n_ret:>+.2f}%")
        
        print("-"*70)
        wins = [t for t in self.trade_log if t['pnl'] > 0]
        losses = [t for t in self.trade_log if t['pnl'] <= 0]
        
        print(f"\n{'TRADE STATISTICS'}")
        print(f"  Total Trades: {len(self.trade_log)}")
        if self.trade_log:
            print(f"  Win Rate: {len(wins)/len(self.trade_log)*100:.1f}%")
            if wins: print(f"  Avg Win: {np.mean([t['pnl'] for t in wins]):+.1f}%")
            if losses: print(f"  Avg Loss: {np.mean([t['pnl'] for t in losses]):+.1f}%")
            print(f"  Max Win: {max([t['pnl'] for t in self.trade_log]):+.1f}%")
            print(f"  Max Loss: {min([t['pnl'] for t in self.trade_log]):+.1f}%")
        
        # Final value
        print(f"\n{'PORTFOLIO VALUE'}")
        print(f"  Starting: Rs.{INITIAL_CAPITAL:,.0f}")
        print(f"  Final:    Rs.{final_val:,.0f}")
        print(f"  P&L:      Rs.{final_val - INITIAL_CAPITAL:+,.0f}")
        
        # Sector breakdown of trades
        print(f"\n{'SECTOR PERFORMANCE'}")
        sector_pnl = {}
        for t in self.trade_log:
            sec = t['sector']
            if sec not in sector_pnl:
                sector_pnl[sec] = []
            sector_pnl[sec].append(t['pnl'])
        
        for sec, pnls in sorted(sector_pnl.items(), key=lambda x: -np.mean(x[1]))[:10]:
            print(f"  {sec[:30]:<32} Trades: {len(pnls):<4} Avg PnL: {np.mean(pnls):+.1f}%")
        
        # Save results
        df.to_csv('analysis_2026/dna3_v21_backtest_3y.csv', index=False)
        trades_df = pd.DataFrame(self.trade_log)
        trades_df.to_csv('analysis_2026/dna3_v21_trades_3y.csv', index=False)
        print(f"\n[SAVED] analysis_2026/dna3_v21_backtest_3y.csv")
        print(f"[SAVED] analysis_2026/dna3_v21_trades_3y.csv")
        
        return total_ret, n_ret

if __name__ == "__main__":
    bt = DNA3V21Backtest()
    bt.run()
