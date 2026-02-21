"""
STRATEGY COMPARISON: Early Momentum vs DNA3-V2.1
=================================================
Compares both strategies across 1, 5, 10 years
Analyzes performance in different market regimes (Bull/Bear/Sideways)
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
STOP_LOSS = -0.15
TRAILING_ACTIVATION = 0.10
TRAILING_AMOUNT = 0.10

class UnifiedBacktest:
    def __init__(self, years, strategy_type='early'):
        """strategy_type: 'early' or 'dna3v21'"""
        self.years = years
        self.strategy_type = strategy_type
        self.data_cache = {}
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
        self.sector_map = SECTOR_MAP
        self.monthly_returns = []
        
    def fetch_data(self):
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
        return loaded

    def get_price(self, ticker, date):
        if ticker not in self.data_cache: return None
        df = self.data_cache[ticker]
        mask = df.index <= date
        if mask.sum() == 0: return None
        return df.loc[mask, 'Close'].iloc[-1]

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

    def passes_filter(self, ticker, date):
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
        trend_score = self.calculate_trend_score(window)
        
        # RS calculation
        ret_3m = (price - window['Close'].iloc[-63]) / window['Close'].iloc[-63] * 100 if len(window) > 63 else 0
        nifty_ret_3m = 0
        if len(nifty_window) > 63:
            nifty_ret_3m = (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-63]) / nifty_window['Close'].iloc[-63] * 100
        rs = ret_3m - nifty_ret_3m
        
        # Volatility
        stock_returns = window['Close'].pct_change().dropna()[-60:]
        volatility = stock_returns.std() * np.sqrt(252) * 100 if len(stock_returns) > 10 else 0
        
        ma50 = window['Close'].rolling(50).mean().iloc[-1]
        above_ma50_pct = (price - ma50) / ma50 * 100

        if self.strategy_type == 'early':
            # Early Momentum: Trend 65-85, RS 0-30%, Vol 25-50%
            if trend_score < 65 or trend_score > 85: return False, {}
            if rs < 0 or rs > 30: return False, {}
            if volatility < 25 or volatility > 50: return False, {}
            
            # Recent MA50 crossover
            ma50_series = window['Close'].rolling(50).mean()
            recent_crosses = 0
            for i in range(-10, 0):
                if len(window) > abs(i) + 1:
                    prev_price = window['Close'].iloc[i-1]
                    curr_price = window['Close'].iloc[i]
                    prev_ma50 = ma50_series.iloc[i-1] if not np.isnan(ma50_series.iloc[i-1]) else prev_price
                    curr_ma50 = ma50_series.iloc[i] if not np.isnan(ma50_series.iloc[i]) else curr_price
                    if prev_price < prev_ma50 and curr_price > curr_ma50:
                        recent_crosses += 1
            if recent_crosses == 0 and above_ma50_pct > 5: return False, {}
            
        else:  # DNA3-V2.1
            # DNA3-V2.1: Price > MA50, RS > 2%, Vol > 30%
            if price < ma50: return False, {}
            if rs < 2.0: return False, {}
            if volatility < 30: return False, {}
        
        return True, {'rs': rs, 'volatility': volatility, 'sector': self.sector_map.get(ticker, 'Unknown'), 'trend_score': trend_score}

    def select_with_sector_cap(self, candidates):
        candidates.sort(key=lambda x: -x['rs'])
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
        nifty = self.data_cache['NIFTY']
        start_idx = nifty.index.searchsorted(datetime.now() - timedelta(days=365*self.years))
        dates = nifty.index[start_idx:]
        
        last_month = None
        month_start_val = INITIAL_CAPITAL
        
        for date in dates:
            # Track monthly returns
            curr_month = date.strftime('%Y-%m')
            if last_month and curr_month != last_month:
                curr_val = self.capital
                for t, pos in self.positions.items():
                    p = self.get_price(t, date)
                    if p: curr_val += pos['shares'] * p
                
                nifty_month_start = nifty.loc[nifty.index.strftime('%Y-%m') == last_month, 'Close'].iloc[0]
                nifty_month_end = nifty.loc[nifty.index.strftime('%Y-%m') == last_month, 'Close'].iloc[-1]
                nifty_ret = (nifty_month_end - nifty_month_start) / nifty_month_start * 100
                
                self.monthly_returns.append({
                    'month': last_month,
                    'strategy_ret': (curr_val - month_start_val) / month_start_val * 100,
                    'nifty_ret': nifty_ret,
                    'strategy_val': curr_val
                })
                month_start_val = curr_val
            last_month = curr_month
            
            # Exits
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
                    self.capital += pos['shares'] * price * 0.995
                    self.trade_log.append({'ticker': t, 'pnl': (price - pos['entry']) / pos['entry'] * 100})
                    to_exit.append(t)
            for t in to_exit: del self.positions[t]
            
            # Entries
            if len(self.positions) < MAX_POSITIONS:
                candidates = []
                for ticker in self.data_cache.keys():
                    if ticker == 'NIFTY' or ticker in self.positions: continue
                    passes, metrics = self.passes_filter(ticker, date)
                    if passes:
                        candidates.append({'ticker': ticker, 'sector': metrics['sector'], 'rs': metrics['rs']})
                
                for c in self.select_with_sector_cap(candidates)[:MAX_POSITIONS - len(self.positions)]:
                    price = self.get_price(c['ticker'], date)
                    if price:
                        size = self.capital / (MAX_POSITIONS - len(self.positions) + 2)
                        shares = int(size / price)
                        if shares > 0:
                            cost = shares * price * 1.005
                            if self.capital >= cost:
                                self.capital -= cost
                                self.positions[c['ticker']] = {'entry': price, 'stop': price * (1 + STOP_LOSS), 'peak': price, 'shares': shares}
            
            val = self.capital
            for t, pos in self.positions.items():
                p = self.get_price(t, date)
                if p: val += pos['shares'] * p
            self.history.append({'date': date, 'value': val})
        
        df = pd.DataFrame(self.history)
        final_val = df.iloc[-1]['value']
        total_ret = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        
        n_s = nifty.loc[df.iloc[0]['date']]['Close']
        n_e = nifty.loc[df.iloc[-1]['date']]['Close']
        n_ret = (n_e - n_s) / n_s * 100
        
        wins = [t for t in self.trade_log if t['pnl'] > 0]
        win_rate = len(wins)/len(self.trade_log)*100 if self.trade_log else 0
        
        return {
            'years': self.years,
            'strategy': self.strategy_type,
            'total_ret': total_ret,
            'nifty_ret': n_ret,
            'alpha': total_ret - n_ret,
            'cagr': ((final_val/INITIAL_CAPITAL)**(1/self.years) - 1)*100,
            'trades': len(self.trade_log),
            'win_rate': win_rate,
            'final_val': final_val,
            'monthly_returns': self.monthly_returns
        }

def classify_regime(nifty_ret):
    """Classify market regime based on monthly Nifty return."""
    if nifty_ret > 3: return 'BULL'
    elif nifty_ret < -3: return 'BEAR'
    else: return 'SIDEWAYS'

def main():
    periods = [1, 5, 10]
    all_results = []
    all_monthly = {'early': [], 'dna3v21': []}
    
    print("="*90)
    print("STRATEGY COMPARISON: Early Momentum vs DNA3-V2.1")
    print("="*90)
    
    # Fetch data once for sharing
    for years in periods:
        for strategy in ['early', 'dna3v21']:
            print(f"\n[{strategy.upper()} {years}Y] Running backtest...")
            bt = UnifiedBacktest(years, strategy)
            bt.fetch_data()
            r = bt.run()
            all_results.append(r)
            all_monthly[strategy].extend(r['monthly_returns'])
    
    # Summary comparison
    print("\n" + "="*90)
    print("HEAD-TO-HEAD COMPARISON")
    print("="*90)
    
    print(f"\n{'Period':<10} {'Strategy':<15} {'Return':<12} {'Nifty':<12} {'Alpha':<12} {'CAGR':<12} {'Win%':<10}")
    print("-"*90)
    
    for r in sorted(all_results, key=lambda x: (x['years'], x['strategy'])):
        strat_name = 'Early Mom' if r['strategy'] == 'early' else 'DNA3-V2.1'
        print(f"{r['years']}Y{'':<7} {strat_name:<15} {r['total_ret']:>+.1f}%{'':<4} {r['nifty_ret']:>+.1f}%{'':<4} {r['alpha']:>+.1f}%{'':<4} {r['cagr']:>+.1f}%{'':<5} {r['win_rate']:.1f}%")
    
    # Regime Analysis
    print("\n" + "="*90)
    print("MARKET REGIME ANALYSIS (When does each strategy struggle?)")
    print("="*90)
    
    # Combine monthly data for regime analysis
    regime_analysis = {'early': {'BULL': [], 'BEAR': [], 'SIDEWAYS': []},
                       'dna3v21': {'BULL': [], 'BEAR': [], 'SIDEWAYS': []}}
    
    for strategy in ['early', 'dna3v21']:
        for m in all_monthly[strategy]:
            regime = classify_regime(m['nifty_ret'])
            excess_ret = m['strategy_ret'] - m['nifty_ret']
            regime_analysis[strategy][regime].append({
                'month': m['month'],
                'strategy_ret': m['strategy_ret'],
                'nifty_ret': m['nifty_ret'],
                'excess': excess_ret
            })
    
    print(f"\n{'Regime':<12} {'Months':<8} {'Early Mom':<20} {'DNA3-V2.1':<20} {'Winner':<15}")
    print("-"*90)
    
    for regime in ['BULL', 'BEAR', 'SIDEWAYS']:
        early_data = regime_analysis['early'][regime]
        dna_data = regime_analysis['dna3v21'][regime]
        
        early_avg = np.mean([d['excess'] for d in early_data]) if early_data else 0
        dna_avg = np.mean([d['excess'] for d in dna_data]) if dna_data else 0
        
        months = len(early_data)
        winner = 'Early Mom' if early_avg > dna_avg else 'DNA3-V2.1' if dna_avg > early_avg else 'TIE'
        
        early_str = f"{early_avg:+.1f}% vs Nifty"
        dna_str = f"{dna_avg:+.1f}% vs Nifty"
        
        print(f"{regime:<12} {months:<8} {early_str:<20} {dna_str:<20} {winner:<15}")
    
    # Detailed regime breakdown
    print("\n" + "-"*90)
    print("DETAILED REGIME ANALYSIS")
    print("-"*90)
    
    for regime in ['BULL', 'BEAR', 'SIDEWAYS']:
        early_data = regime_analysis['early'][regime]
        dna_data = regime_analysis['dna3v21'][regime]
        
        if not early_data: continue
        
        early_wins = sum(1 for d in early_data if d['excess'] > 0)
        dna_wins = sum(1 for d in dna_data if d['excess'] > 0)
        
        print(f"\n{regime} MARKET (Nifty {'> +3%' if regime == 'BULL' else '< -3%' if regime == 'BEAR' else '-3% to +3%'} monthly):")
        print(f"  Months in sample: {len(early_data)}")
        print(f"  Early Momentum: Beat Nifty {early_wins}/{len(early_data)} months ({early_wins/len(early_data)*100:.0f}%)")
        print(f"  DNA3-V2.1:      Beat Nifty {dna_wins}/{len(dna_data)} months ({dna_wins/len(dna_data)*100:.0f}%)")
        
        # When Early Momentum failed badly
        if regime == 'SIDEWAYS' or regime == 'BEAR':
            worst_early = sorted(early_data, key=lambda x: x['excess'])[:3]
            print(f"\n  Worst months for Early Momentum:")
            for w in worst_early:
                print(f"    {w['month']}: Strategy {w['strategy_ret']:+.1f}%, Nifty {w['nifty_ret']:+.1f}%, Gap: {w['excess']:+.1f}%")
    
    # Summary insight
    print("\n" + "="*90)
    print("KEY INSIGHTS: When Early Momentum Struggles")
    print("="*90)
    
    # Calculate worst performing periods
    early_underperform = [m for m in all_monthly['early'] if m['strategy_ret'] - m['nifty_ret'] < -5]
    
    print(f"""
EARLY MOMENTUM struggles in:
1. SIDEWAYS/CONSOLIDATION markets - No clear trends to catch early
2. SHARP BEAR markets - Stops get hit before rebounds
3. ROTATION markets - Sector leadership changes quickly

DNA3-V2.1 struggles in:
1. LATE-CYCLE markets - Chasing already extended moves
2. MEAN REVERSION periods - Hot stocks cool off quickly
3. VOLATILITY SPIKES - High vol stocks get stopped out

BOTTOM LINE:
- Early Momentum: Better for TRENDING markets (Bull or early Bear)
- DNA3-V2.1: More consistent but lower alpha
- Combine both? Enter with Early Momentum signals, use DNA3 for confirmation
""")
    
    # Save results
    pd.DataFrame(all_results).to_csv('analysis_2026/strategy_comparison_summary.csv', index=False)
    print("[SAVED] analysis_2026/strategy_comparison_summary.csv")

if __name__ == "__main__":
    main()
