"""
REGIME-ADAPTIVE STRATEGY: DNA4
==============================
Switches between momentum and mean reversion based on market regime.

BULL/MILD BULL --> DNA3-V2.1 (Momentum: Trend 90+, RS > 2%, Vol > 30%)
SIDEWAYS/BEAR --> Mean Reversion (Trend 0-20, Volume drop, oversold)

Regime Detection:
- BULL: Nifty > MA50 AND MA50 > MA200 AND 1-month return > 3%
- MILD BULL: Nifty > MA50 AND 1-month return > 0%
- SIDEWAYS: Nifty near MA50 (+/- 2%) OR 1-month return between -3% and 0%
- BEAR: Nifty < MA50 AND 1-month return < -3%
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
STOP_LOSS_REVERSION = -0.10  # Tighter stop for mean reversion
TRAILING_ACTIVATION = 0.10
TRAILING_AMOUNT = 0.10

class RegimeAdaptiveStrategy:
    def __init__(self, years):
        self.years = years
        self.data_cache = {}
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
        self.regime_log = []
        self.sector_map = SECTOR_MAP
        
    def fetch_data(self):
        print(f"[DNA4 ADAPTIVE] Fetching data for {self.years}-year backtest...")
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
        print(f"   Loaded {loaded} stocks")

    def get_price(self, ticker, date):
        if ticker not in self.data_cache: return None
        df = self.data_cache[ticker]
        mask = df.index <= date
        if mask.sum() == 0: return None
        return df.loc[mask, 'Close'].iloc[-1]

    def detect_regime(self, date):
        """Detect market regime based on Nifty technical indicators."""
        nifty = self.data_cache['NIFTY']
        idx = nifty.index.searchsorted(date)
        if idx < 200: return 'UNKNOWN'
        
        window = nifty.iloc[max(0, idx-200):idx+1]
        
        price = window['Close'].iloc[-1]
        ma50 = window['Close'].rolling(50).mean().iloc[-1]
        ma200 = window['Close'].rolling(200).mean().iloc[-1]
        
        # 1-month return
        ret_1m = (price - window['Close'].iloc[-21]) / window['Close'].iloc[-21] * 100 if len(window) > 21 else 0
        
        # Distance from MA50
        dist_ma50 = (price - ma50) / ma50 * 100
        
        # Regime classification
        if price > ma50 and ma50 > ma200 and ret_1m > 3:
            return 'BULL'
        elif price > ma50 and ret_1m > 0:
            return 'MILD_BULL'
        elif abs(dist_ma50) < 2 or (-3 < ret_1m <= 0):
            return 'SIDEWAYS'
        elif price < ma50 and ret_1m < -3:
            return 'BEAR'
        else:
            return 'SIDEWAYS'  # Default to sideways

    def calculate_trend_score(self, window):
        if len(window) < 200: return 0
        price = window['Close'].iloc[-1]
        ma50 = window['Close'].rolling(50).mean().iloc[-1]
        ma200 = window['Close'].rolling(200).mean().iloc[-1]
        
        score = 50
        if price > ma50: score += 20
        if price > ma200: score += 15
        if ma50 > ma200: score += 10
        
        ret_1m = (price - window['Close'].iloc[-21]) / window['Close'].iloc[-21] * 100 if len(window) > 21 else 0
        if ret_1m > 5: score += 5
        
        return score

    def passes_momentum_filter(self, ticker, date):
        """DNA3-V2.1 Momentum Filter for Bull regimes."""
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
        
        # Price > MA50
        if price < ma50: return False, {}
        
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
        
        return True, {'rs': rs, 'volatility': volatility, 'sector': self.sector_map.get(ticker, 'Unknown'), 'strategy': 'MOMENTUM'}

    def passes_mean_reversion_filter(self, ticker, date):
        """Mean Reversion Filter for Sideways/Bear regimes."""
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
        
        # Trend Score 0-30 (beaten down, not trending)
        trend_score = self.calculate_trend_score(window)
        if trend_score > 30: return False, {}  # Want beaten down stocks
        
        # RSI < 35 (oversold)
        delta = window['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs_indicator = gain / loss
        rsi = 100 - (100 / (1 + rs_indicator.iloc[-1])) if loss.iloc[-1] != 0 else 50
        
        if rsi > 35: return False, {}  # Must be oversold
        
        # Volume drop or flat (not panic selling)
        vol_20d = window['Volume'].rolling(20).mean().iloc[-1]
        vol_50d = window['Volume'].rolling(50).mean().iloc[-1]
        vol_ratio = vol_20d / vol_50d if vol_50d > 0 else 1
        
        if vol_ratio > 1.5: return False, {}  # Avoid panic selling (high volume)
        
        # Must have fallen significantly (at least -15% from recent high)
        peak_3m = window['Close'].rolling(63).max().iloc[-1]
        drawdown = (price - peak_3m) / peak_3m * 100
        
        if drawdown > -15: return False, {}  # Must have fallen
        
        # But not too much (avoid falling knives)
        if drawdown < -40: return False, {}
        
        # RS can be negative but not catastrophic
        ret_3m = (price - window['Close'].iloc[-63]) / window['Close'].iloc[-63] * 100 if len(window) > 63 else 0
        nifty_ret_3m = 0
        if len(nifty_window) > 63:
            nifty_ret_3m = (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-63]) / nifty_window['Close'].iloc[-63] * 100
        rs = ret_3m - nifty_ret_3m
        
        return True, {
            'rs': rs, 
            'rsi': rsi, 
            'drawdown': drawdown,
            'vol_ratio': vol_ratio,
            'sector': self.sector_map.get(ticker, 'Unknown'),
            'strategy': 'MEAN_REVERSION'
        }

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
        self.fetch_data()
        nifty = self.data_cache['NIFTY']
        start_idx = nifty.index.searchsorted(datetime.now() - timedelta(days=365*self.years))
        dates = nifty.index[start_idx:]
        
        print(f"[DNA4 ADAPTIVE] Running {self.years}-Year Backtest...")
        
        last_regime = None
        regime_counts = {'BULL': 0, 'MILD_BULL': 0, 'SIDEWAYS': 0, 'BEAR': 0}
        
        for date in dates:
            regime = self.detect_regime(date)
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            if regime != last_regime:
                self.regime_log.append({'date': date, 'regime': regime})
            last_regime = regime
            
            # Determine which strategy to use
            use_momentum = regime in ['BULL', 'MILD_BULL']
            stop_loss = STOP_LOSS_MOMENTUM if use_momentum else STOP_LOSS_REVERSION
            
            # === EXITS ===
            to_exit = []
            for t, pos in self.positions.items():
                price = self.get_price(t, date)
                if not price: continue
                
                if price > pos['peak']: pos['peak'] = price
                
                ret = (price - pos['entry']) / pos['entry']
                
                # Different exit logic for different strategies
                if pos['strategy'] == 'MOMENTUM':
                    if ret > TRAILING_ACTIVATION:
                        trail = pos['peak'] * (1 - TRAILING_AMOUNT)
                        if trail > pos['stop']: pos['stop'] = trail
                    if price < pos['stop']:
                        to_exit.append(t)
                else:  # MEAN_REVERSION
                    # Take profit at +15% or stop at -10%
                    if ret > 0.15 or ret < -0.10:
                        to_exit.append(t)
                    # Also exit if RSI > 60 (overbought now)
                    if t in self.data_cache:
                        df = self.data_cache[t]
                        idx = df.index.searchsorted(date)
                        if idx > 14:
                            window = df.iloc[max(0, idx-20):idx+1]
                            delta = window['Close'].diff()
                            gain = delta.where(delta > 0, 0).rolling(14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                            if loss.iloc[-1] != 0:
                                rsi = 100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1]))
                                if rsi > 60:
                                    to_exit.append(t)
            
            for t in set(to_exit):
                if t in self.positions:
                    pos = self.positions[t]
                    price = self.get_price(t, date)
                    if price:
                        self.capital += pos['shares'] * price * 0.995
                        pnl = (price - pos['entry']) / pos['entry'] * 100
                        self.trade_log.append({
                            'ticker': t, 
                            'pnl': pnl, 
                            'strategy': pos['strategy'],
                            'regime': regime,
                            'date': date
                        })
                    del self.positions[t]
            
            # === ENTRIES ===
            if len(self.positions) < MAX_POSITIONS:
                candidates = []
                
                for ticker in self.data_cache.keys():
                    if ticker == 'NIFTY' or ticker in self.positions: continue
                    
                    if use_momentum:
                        passes, metrics = self.passes_momentum_filter(ticker, date)
                    else:
                        passes, metrics = self.passes_mean_reversion_filter(ticker, date)
                    
                    if passes:
                        candidates.append({'ticker': ticker, **metrics})
                
                # Sort differently for each strategy
                if use_momentum:
                    candidates.sort(key=lambda x: -x['rs'])  # Highest RS first
                else:
                    candidates.sort(key=lambda x: x.get('drawdown', 0))  # Most beaten down first
                
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
                                    'stop': price * (1 + stop_loss),
                                    'peak': price, 
                                    'shares': shares,
                                    'strategy': c['strategy']
                                }
            
            val = self.capital
            for t, pos in self.positions.items():
                p = self.get_price(t, date)
                if p: val += pos['shares'] * p
            self.history.append({'date': date, 'value': val, 'regime': regime})
        
        # Results
        df = pd.DataFrame(self.history)
        final_val = df.iloc[-1]['value']
        total_ret = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        
        n_s = nifty.loc[df.iloc[0]['date']]['Close']
        n_e = nifty.loc[df.iloc[-1]['date']]['Close']
        n_ret = (n_e - n_s) / n_s * 100
        
        # Trade analysis by strategy
        mom_trades = [t for t in self.trade_log if t['strategy'] == 'MOMENTUM']
        rev_trades = [t for t in self.trade_log if t['strategy'] == 'MEAN_REVERSION']
        
        mom_wins = [t for t in mom_trades if t['pnl'] > 0]
        rev_wins = [t for t in rev_trades if t['pnl'] > 0]
        
        print("\n" + "="*80)
        print(f"DNA4 REGIME-ADAPTIVE STRATEGY: {self.years}-YEAR RESULTS")
        print("="*80)
        
        print(f"\n{'OVERALL PERFORMANCE'}")
        print("-"*80)
        print(f"  Total Return:    {total_ret:>+.2f}%")
        print(f"  Nifty Return:    {n_ret:>+.2f}%")
        print(f"  Alpha:           {total_ret - n_ret:>+.2f}%")
        print(f"  CAGR:            {((final_val/INITIAL_CAPITAL)**(1/self.years) - 1)*100:>+.2f}%")
        print(f"  Final Value:     Rs.{final_val:,.0f}")
        
        print(f"\n{'REGIME BREAKDOWN (Trading Days)'}")
        print("-"*80)
        total_days = sum(regime_counts.values())
        for regime, count in regime_counts.items():
            print(f"  {regime:<15}: {count:>4} days ({count/total_days*100:.1f}%)")
        
        print(f"\n{'STRATEGY PERFORMANCE'}")
        print("-"*80)
        print(f"  {'Strategy':<20} {'Trades':<10} {'Win%':<10} {'Avg PnL':<12}")
        print(f"  {'-'*50}")
        
        if mom_trades:
            mom_avg = np.mean([t['pnl'] for t in mom_trades])
            mom_wr = len(mom_wins)/len(mom_trades)*100
            print(f"  {'MOMENTUM':<20} {len(mom_trades):<10} {mom_wr:.1f}%{'':<4} {mom_avg:>+.1f}%")
        
        if rev_trades:
            rev_avg = np.mean([t['pnl'] for t in rev_trades])
            rev_wr = len(rev_wins)/len(rev_trades)*100
            print(f"  {'MEAN REVERSION':<20} {len(rev_trades):<10} {rev_wr:.1f}%{'':<4} {rev_avg:>+.1f}%")
        
        # Regime switches
        print(f"\n{'REGIME SWITCHES'}")
        print("-"*80)
        print(f"  Total switches: {len(self.regime_log)}")
        for log in self.regime_log[-10:]:  # Last 10 switches
            print(f"    {log['date'].strftime('%Y-%m-%d')}: {log['regime']}")
        
        return {
            'years': self.years,
            'total_ret': total_ret,
            'nifty_ret': n_ret,
            'alpha': total_ret - n_ret,
            'cagr': ((final_val/INITIAL_CAPITAL)**(1/self.years) - 1)*100,
            'momentum_trades': len(mom_trades),
            'reversion_trades': len(rev_trades),
            'final_val': final_val,
            'regime_counts': regime_counts
        }

def main():
    periods = [1, 3, 5, 10]
    all_results = []
    
    print("="*80)
    print("DNA4: REGIME-ADAPTIVE STRATEGY BACKTEST")
    print("="*80)
    print("\nStrategy Logic:")
    print("  BULL/MILD_BULL --> DNA3-V2.1 Momentum (Trend 90+, RS > 2%)")
    print("  SIDEWAYS/BEAR  --> Mean Reversion (Trend < 30, RSI < 35, Vol drop)")
    
    for years in periods:
        bt = RegimeAdaptiveStrategy(years)
        r = bt.run()
        all_results.append(r)
    
    # Summary comparison
    print("\n" + "="*80)
    print("DNA4 MULTI-PERIOD SUMMARY")
    print("="*80)
    
    print(f"\n{'Period':<10} {'Return':<12} {'Nifty':<12} {'Alpha':<12} {'CAGR':<12} {'Mom Trades':<12} {'Rev Trades':<12}")
    print("-"*80)
    
    for r in all_results:
        print(f"{r['years']}Y{'':<7} {r['total_ret']:>+.1f}%{'':<4} {r['nifty_ret']:>+.1f}%{'':<4} {r['alpha']:>+.1f}%{'':<4} {r['cagr']:>+.1f}%{'':<5} {r['momentum_trades']:<12} {r['reversion_trades']:<12}")
    
    # Save results
    pd.DataFrame(all_results).to_csv('analysis_2026/dna4_regime_adaptive.csv', index=False)
    print(f"\n[SAVED] analysis_2026/dna4_regime_adaptive.csv")

if __name__ == "__main__":
    main()
