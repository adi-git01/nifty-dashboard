"""
DNA3 Composite RS - Grid Search Optimization
===========================================
Tests permutations of Rebalance Frequency, Composite Weights, and G-Factor
to find the optimal configuration that maximizes Expectancy/Sharpe while
reducing whipsaws.

Usage:
  python dna3_composite_optimization.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nifty500_list import TICKERS, SECTOR_MAP

warnings.filterwarnings('ignore')

INITIAL_CAPITAL = 1000000
COST_BPS = 50
OUTPUT_DIR = "analysis_2026"

# ---------------------------------------------------------
# REGIME DETECTION & CONFIG
# ---------------------------------------------------------
def detect_regime(nifty_df, date):
    idx = nifty_df.index.searchsorted(date)
    if idx < 200: return 'UNKNOWN'
    w = nifty_df.iloc[max(0, idx-252):idx+1]
    if len(w) < 63: return 'UNKNOWN'
    
    price = w['Close'].iloc[-1]
    ma50 = w['Close'].rolling(50).mean().iloc[-1]
    ma200 = w['Close'].rolling(200).mean().iloc[-1]
    ret_3m = (price - w['Close'].iloc[-63]) / w['Close'].iloc[-63] * 100
    peak = w['Close'].cummax().iloc[-1]
    dd = (price - peak) / peak * 100
    
    if price > ma50 and ma50 > ma200 and ret_3m > 5: return 'BULL'
    elif price > ma50 and ret_3m > 0: return 'MILD_BULL'
    elif price < ma50 and (ret_3m < -5 or dd < -10): return 'BEAR'
    else: return 'SIDEWAYS'

REGIME_CONFIG = {
    'BULL':      {'max_pos': 12, 'cash_reserve': 0.05},
    'MILD_BULL': {'max_pos': 10, 'cash_reserve': 0.10},
    'SIDEWAYS':  {'max_pos':  8, 'cash_reserve': 0.20},
    'BEAR':      {'max_pos':  6, 'cash_reserve': 0.40},
    'UNKNOWN':   {'max_pos':  8, 'cash_reserve': 0.20},
}


# ---------------------------------------------------------
# INDICATORS (Modified for G-Factor and Weights)
# ---------------------------------------------------------
def calc_timeframe_rs(stock_window, nifty_window, days):
    if len(stock_window) <= days or len(nifty_window) <= days: return 0.0
    try:
        s_ret = (stock_window['Close'].iloc[-1] / stock_window['Close'].iloc[-days]) - 1
        n_ret = (nifty_window['Close'].iloc[-1] / nifty_window['Close'].iloc[-days]) - 1
        return (s_ret - n_ret) * 100
    except: return 0.0

def calc_g_factor(stock_window, nifty_window, lookback=20):
    """
    Green in Sea of Red (G-Factor):
    Sum of relative outperformance on days where Nifty fell > 1%.
    """
    if len(stock_window) < lookback + 1 or len(nifty_window) < lookback + 1:
        return 0.0
        
    s_rets = stock_window['Close'].pct_change().dropna()[-lookback:]
    n_rets = nifty_window['Close'].pct_change().dropna()[-lookback:]
    
    g_score = 0.0
    # Align indices (assuming they are aligned by the caller mostly)
    common_dates = n_rets.index.intersection(s_rets.index)
    
    for d in common_dates:
        n_ret = n_rets.loc[d]
        if n_ret < -0.01:  # Nifty down > 1%
            s_ret = s_rets.loc[d]
            excess = s_ret - n_ret
            if excess > 0:
                g_score += excess * 100  # Points for outperforming on blood days
                
    return g_score


def calc_indicators(stock_window, nifty_window, weights, g_mode):
    if len(stock_window) < 100 or len(nifty_window) < 64: return None
    
    p = stock_window['Close'].iloc[-1]
    ma50 = stock_window['Close'].rolling(50).mean().iloc[-1]
    
    # Weights: tuple (w_1w, w_1m, w_3m)
    rs_1w = calc_timeframe_rs(stock_window, nifty_window, 5)
    rs_1m = calc_timeframe_rs(stock_window, nifty_window, 21)
    rs_3m = calc_timeframe_rs(stock_window, nifty_window, 63)
    
    rs_score = (rs_1w * weights[0]) + (rs_1m * weights[1]) + (rs_3m * weights[2])
    
    g_score = calc_g_factor(stock_window, nifty_window)
    
    if g_mode == 'Boost':
        rs_score += g_score * 2.0  # Moderate boost
        
    volatility = stock_window['Close'].pct_change().dropna()[-60:].std() * np.sqrt(252) * 100
    liquidity = stock_window['Volume'].rolling(20).mean().iloc[-1] * p
    
    return {
        'price': p, 'ma50': ma50, 'rs_score': rs_score,
        'g_score': g_score, 'volatility': volatility, 'liquidity': liquidity
    }


# ---------------------------------------------------------
# ENGINE
# ---------------------------------------------------------
class OptEngine:
    def __init__(self, reb_freq, weights, g_mode):
        self.reb_freq = reb_freq
        self.weights = weights
        self.g_mode = g_mode
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
        
    def get_price(self, dc, t, date):
        df = dc.get(t)
        if df is None: return None
        idx = df.index.searchsorted(date)
        if idx == 0: return None
        return df['Close'].iloc[min(idx, len(df)-1)]

    def scan_and_buy(self, dc, nifty, date, regime):
        cfg = REGIME_CONFIG.get(regime, REGIME_CONFIG['UNKNOWN'])
        max_pos = cfg['max_pos']
        
        # Check Exits (V2.1 Logic: 15% hard, trail after 10%)
        to_exit = []
        for t, pos in self.positions.items():
            price = self.get_price(dc, t, date)
            if not price: continue
            if price > pos['peak']: pos['peak'] = price
            ret = (price - pos['entry']) / pos['entry']
            
            reason = None
            if ret > 0.10:
                trail = pos['peak'] * 0.90
                if trail > pos['stop']: pos['stop'] = trail
            
            if price < pos['stop']:
                reason = 'Stop/Trail'
                
            if reason:
                proceeds = pos['shares'] * price * (1 - COST_BPS/10000)
                self.capital += proceeds
                hd = (date - pos['entry_date']).days
                self.trade_log.append({'PnL%': ret*100, 'Hold_Days': hd})
                to_exit.append(t)
                
        for t in to_exit: del self.positions[t]
        
        if len(self.positions) >= max_pos: return
        
        # Scan
        nifty_idx = nifty.index.searchsorted(date)
        if nifty_idx < 100: return
        nw = nifty.iloc[max(0, nifty_idx-252):nifty_idx+1]
        
        cands = []
        for t, df in dc.items():
            if t == 'NIFTY' or t in self.positions: continue
            idx = df.index.searchsorted(date)
            if idx < 100: continue
            w = df.iloc[max(0, idx-252):idx+1]
            ind = calc_indicators(w, nw, self.weights, self.g_mode)
            
            if not ind: continue
            
            # Filters
            if ind['rs_score'] < 2.0: continue
            if ind['volatility'] < 30: continue
            if ind['price'] < ind['ma50']: continue
            if self.g_mode == 'Filter' and ind['g_score'] <= 0: continue
            
            cands.append({'ticker': t, 'ind': ind})
            
        cands.sort(key=lambda x: -x['ind']['rs_score'])
        
        # Buy
        avail = self.capital - (self.get_equity(dc, date) * cfg['cash_reserve'])
        free = max_pos - len(self.positions)
        
        for c in cands[:free]:
            p = c['ind']['price']
            size = avail / (free + 1)
            sh = int(size / p)
            cost = sh * p * (1 + COST_BPS/10000)
            if sh > 0 and avail >= cost:
                avail -= cost
                self.capital -= cost
                self.positions[c['ticker']] = {
                    'entry': p, 'peak': p, 'shares': sh,
                    'stop': p * 0.85, 'entry_date': date
                }

    def get_equity(self, dc, date):
        val = self.capital
        for t, pos in self.positions.items():
            p = self.get_price(dc, t, date)
            if p: val += pos['shares'] * p
        return val

    def run(self, dc, nifty, start, end):
        si = nifty.index.searchsorted(start)
        ei = nifty.index.searchsorted(end)
        dates = nifty.index[si:ei+1]
        day = 0
        for d in dates:
            regime = detect_regime(nifty, d)
            if day % self.reb_freq == 0:
                self.scan_and_buy(dc, nifty, d, regime)
            else:
                # Still check exits daily
                to_exit = []
                for t, pos in self.positions.items():
                    price = self.get_price(dc, t, d)
                    if price:
                        if price > pos['peak']: pos['peak'] = price
                        ret = (price - pos['entry']) / pos['entry']
                        if ret > 0.10:
                            trail = pos['peak'] * 0.90
                            if trail > pos['stop']: pos['stop'] = trail
                        if price < pos['stop']:
                            proceeds = pos['shares'] * price * (1 - COST_BPS/10000)
                            self.capital += proceeds
                            self.trade_log.append({
                                'PnL%': ret*100, 
                                'Hold_Days': (d - pos['entry_date']).days
                            })
                            to_exit.append(t)
                for t in to_exit: del self.positions[t]
                
            self.history.append({'date': d, 'equity': self.get_equity(dc, d)})
            day += 1

# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------
def fetch_data():
    start = (datetime.now() - timedelta(days=365*5 + 252)).strftime('%Y-%m-%d')
    nifty = yf.Ticker("^NSEI").history(start=start)
    nifty.index = nifty.index.tz_localize(None)
    
    print("Downloading 500 stocks...")
    bulk = yf.download(TICKERS[:500], start=start, group_by='ticker', threads=False, progress=True, auto_adjust=True)
    dc = {'NIFTY': nifty}
    for t in TICKERS[:500]:
        try:
            if t in bulk.columns.get_level_values(0):
                df = bulk[t].dropna(how='all')
                if len(df) > 200:
                    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                    dc[t] = df
        except: pass
    return nifty, dc

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def run_optimization():
    print("=" * 80)
    print("PHASE 2: GRID SEARCH OPTIMIZATION (5-Year Window)")
    print("=" * 80)
    
    nifty, dc = fetch_data()
    end_date = nifty.index[-1]
    start_date = nifty.index[nifty.index.searchsorted(end_date - timedelta(days=365*5))]
    
    PARAM_GRID = {
        'Rebalance': [5, 10, 15],
        'Weights': [
            ((0.3, 0.3, 0.4), "Base 30/30/40"),
            ((0.2, 0.2, 0.6), "Smooth 20/20/60"),
            ((0.1, 0.5, 0.4), "Mid 10/50/40")
        ],
        'GFactor': ['Off', 'Filter', 'Boost']
    }
    
    results = []
    total_runs = len(PARAM_GRID['Rebalance']) * len(PARAM_GRID['Weights']) * len(PARAM_GRID['GFactor'])
    counter = 1
    
    print("\nStarting Grid Search...")
    print(f"{'Run':<5} {'Reb':<4} {'Weights':<15} {'GFactor':<8} | {'CAGR%':>8} {'Sharpe':>7} {'WinRate':>8} {'Exp':>6} {'Whipsaw':>8}")
    print("-" * 88)
    
    for reb in PARAM_GRID['Rebalance']:
        for w_val, w_name in PARAM_GRID['Weights']:
            for gm in PARAM_GRID['GFactor']:
                eng = OptEngine(reb, w_val, gm)
                eng.run(dc, nifty, start_date, end_date)
                
                # Calculate metrics
                eq = pd.DataFrame(eng.history)
                cagr = ((eq['equity'].iloc[-1] / INITIAL_CAPITAL) ** (1/5) - 1) * 100
                
                eq['ret'] = eq['equity'].pct_change()
                mo = eq.groupby(eq['date'].dt.to_period('M'))['equity'].last().pct_change().dropna()
                sharpe = (mo.mean() / mo.std()) * np.sqrt(12) if len(mo)>2 and mo.std()>0 else 0
                
                sells = eng.trade_log
                wins = [t for t in sells if t['PnL%'] > 0]
                wr = len(wins) / len(sells) * 100 if sells else 0
                
                aw = np.mean([t['PnL%'] for t in wins]) if wins else 0
                al = np.mean([t['PnL%'] for t in sells if t['PnL%'] <= 0]) if len(sells)>len(wins) else 0
                exp = (wr/100)*aw - (1-wr/100)*abs(al) if sells else 0
                
                whip = sum(1 for t in sells if t['Hold_Days'] < 10) / len(sells) * 100 if sells else 0
                
                print(f"{counter:<5} {reb:<4} {w_name:<15} {gm:<8} | {cagr:>8.2f} {sharpe:>7.2f} {wr:>7.1f}% {exp:>5.1f}% {whip:>7.1f}%")
                
                results.append({
                    'Rebalance': reb,
                    'Weights': w_name,
                    'GFactor': gm,
                    'CAGR': cagr,
                    'Sharpe': sharpe,
                    'WinRate': wr,
                    'Expectancy': exp,
                    'WhipsawRate': whip
                })
                counter += 1
                
    df = pd.DataFrame(results)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = f"{OUTPUT_DIR}/composite_optimization_grid.csv"
    df.to_csv(out, index=False)
    
    print("\n" + "=" * 80)
    print("TOP 5 PERFORMERS BY EXPECTANCY (per trade edge)")
    print("=" * 80)
    top_exp = df.sort_values('Expectancy', ascending=False).head(5)
    for _, r in top_exp.iterrows():
        print(f"Reb: {r['Rebalance']:<2} | {r['Weights']:<15} | GFact: {r['GFactor']:<6} --> Exp: +{r['Expectancy']:.1f}% (CAGR: {r['CAGR']:.1f}%)")

    print("\n" + "=" * 80)
    print("TOP 5 PERFORMERS BY SHARPE RATIO (risk adjusted)")
    print("=" * 80)
    top_shr = df.sort_values('Sharpe', ascending=False).head(5)
    for _, r in top_shr.iterrows():
        print(f"Reb: {r['Rebalance']:<2} | {r['Weights']:<15} | GFact: {r['GFactor']:<6} --> Sharpe: {r['Sharpe']:.2f} (CAGR: {r['CAGR']:.1f}%)")

if __name__ == "__main__":
    run_optimization()
