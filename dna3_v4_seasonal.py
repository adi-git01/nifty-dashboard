"""
DNA3-V4: THE SEASONAL MOMENTUM ENGINE (BACKTEST)
================================================
Compares standard DNA3-V3.1 (Pure Momentum) against DNA3-V4 (Momentum + Seasonality Edge Overlay).

Logic for DNA3-V4:
1. TRAP AVOIDANCE (Veto): If a stock triggers a buy signal, but its sector is historically in a "Trap" month (e.g., Financials in Jan, Auto in Feb), the buy signal is VETOED.
2. EDGE EXPLOITATION (Weighting): If a stock triggers a buy signal, and its sector is in a "Golden" month (e.g., Industrials in May, Bank in Nov), we double the conviction/weight.

Baseline Strategy:
- Max 10 positions
- Trialing stop (12%)
- Hard stop (-20%)
- Bi-weekly rebalance
- Regime-adaptive sizing
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
OUTPUT_DIR = "analysis_2026/dna3_v4"
MAX_YEARS = 15 # Fetch 15 years of data
REBALANCE_FREQ = 15
HORIZONS = [0.5, 1, 3, 5, 10, 15]

# ----------------------------------------------------------------------
# THE SEASONAL OVERLAY DICTIONARY (Derived from 15Y Encyclopedia Edge Data)
# Format: { 'Sector_Keyword': { Month_Num: 'TRAP' | 'EDGE' } }
# ----------------------------------------------------------------------

SEASONAL_RULES = {
    # TRAPS (Months where the sector bleeds out - VETO ALL BUYS)
    'Financial': {1: 'TRAP', 2: 'TRAP', 5: 'TRAP'},          # Financials famously bleed in Jan/Feb/May
    'Industrial': {1: 'TRAP'},                               # Industrials drop in Jan
    'IT': {2: 'TRAP', 3: 'TRAP'},                            # Tech struggles Feb-Mar
    'Auto': {2: 'TRAP'},                                     # Auto struggles in Feb
    'Real Estate': {2: 'TRAP'},                              # Realty struggles in Feb
    
    # EDGES (Golden months - ACCELERATE BUYS)
    'Industrial': {3: 'EDGE', 4: 'EDGE', 5: 'EDGE'},         # Industrials massive run Mar-May
    'Auto': {4: 'EDGE', 5: 'EDGE'},                          # Autos massive run Apr-May
    'Financial': {10: 'EDGE', 11: 'EDGE', 12: 'EDGE'},       # Financials Q4 run (Oct-Dec)
    'IT': {6: 'EDGE', 7: 'EDGE'},                            # Tech summer run (Jun-Jul)
    'Consumer': {9: 'EDGE', 10: 'EDGE'},                     # Festive demand (Sep-Oct)
    'Healthcare': {7: 'EDGE', 8: 'EDGE'},                    # Defensive rotation (Jul-Aug)
}

# General Sector classification mapping helper (since SECTOR_MAP strings vary)
def classify_sector(sector_str):
    s = str(sector_str).upper()
    if any(x in s for x in ['BANK', 'FINANC', 'INSUR', 'BROKER']): return 'Financial'
    if any(x in s for x in ['INDUST', 'CAPITAL GOODS', 'ENGINEERING', 'METAL']): return 'Industrial'
    if any(x in s for x in ['IT ', 'SOFTWARE', 'TECH']): return 'IT'
    if any(x in s for x in ['AUTO']): return 'Auto'
    if any(x in s for x in ['CONSUMER', 'FMCG', 'RETAIL', 'FOOD']): return 'Consumer'
    if any(x in s for x in ['HEALTH', 'PHARMA', 'HOSPITAL']): return 'Healthcare'
    if any(x in s for x in ['REALTY', 'REAL ESTATE', 'BUILDING']): return 'Real Estate'
    return 'Other'

def get_seasonal_overlay(sector_str, month):
    cat = classify_sector(sector_str)
    if cat in SEASONAL_RULES and month in SEASONAL_RULES[cat]:
        return SEASONAL_RULES[cat][month]
    return 'NEUTRAL'

# Regime config
REGIME_CFG = {
    'BULL':      {'max_pos': 12, 'cash': 0.05},
    'MILD_BULL': {'max_pos': 10, 'cash': 0.10},
    'SIDEWAYS':  {'max_pos':  8, 'cash': 0.20},
    'BEAR':      {'max_pos':  6, 'cash': 0.40},
    'UNKNOWN':   {'max_pos':  8, 'cash': 0.20},
}

def detect_regime(nifty, date):
    idx = nifty.index.searchsorted(date)
    if idx < 200: return 'UNKNOWN'
    w = nifty.iloc[max(0, idx-252):idx+1]
    if len(w) < 63: return 'UNKNOWN'
    p = w['Close'].iloc[-1]
    ma50 = w['Close'].rolling(50).mean().iloc[-1]
    ma200 = w['Close'].rolling(200).mean().iloc[-1]
    ret = (p - w['Close'].iloc[-63]) / w['Close'].iloc[-63] * 100
    pk = w['Close'].cummax().iloc[-1]
    dd = (p - pk) / pk * 100
    if p > ma50 and ma50 > ma200 and ret > 5: return 'BULL'
    elif p > ma50 and ret > 0: return 'MILD_BULL'
    elif p < ma50 and (ret < -5 or dd < -10): return 'BEAR'
    else: return 'SIDEWAYS'

# ============================================================
# BASE V3.1 SCAN LOGIC (Using Relative Strength)
# ============================================================
def get_v31_candidates(dc, nifty_window, d):
    """Returns list of dicts with basic V3.1 RS criteria."""
    cands = []
    for t in dc:
        if t == 'NIFTY': continue
        idx = dc[t].index.searchsorted(d)
        if idx < 100: continue
        w = dc[t].iloc[max(0, idx-252):idx+1]
        if len(w) < 100: continue
        
        price = w['Close'].iloc[-1]
        ma50 = w['Close'].rolling(50).mean().iloc[-1]
        
        # RS Calculation vs Nifty
        if len(w)>63 and len(nifty_window)>63:
            s_ret = (price - w['Close'].iloc[-63])/w['Close'].iloc[-63]
            n_ret = (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-63])/nifty_window['Close'].iloc[-63]
            rs = (s_ret - n_ret) * 100
        else:
            rs = 0
            
        rets = w['Close'].pct_change().dropna()[-60:]
        vol = rets.std() * np.sqrt(252) * 100 if len(rets) > 10 else 0
        liq = w['Volume'].rolling(20).mean().iloc[-1] * price
        
        # Base Engine Entry logic
        if rs >= 2.0 and vol >= 30 and price > ma50 and liq >= 5_000_000:
            cands.append({'ticker': t, 'rs': rs, 'price': price, 'sector': SECTOR_MAP.get(t, 'Unk')})
            
    cands.sort(key=lambda x: -x['rs'])
    return cands

# ============================================================
# ENGINE CLASSES
# ============================================================
class TrendEngineBase:
    def __init__(self, use_seasonality=False):
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
        self.use_seasonality = use_seasonality
        
    def reset(self):
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
        
    def get_price(self, dc, t, d):
        if t not in dc: return None
        i = dc[t].index.searchsorted(d)
        return dc[t]['Close'].iloc[min(i, len(dc[t])-1)] if i > 0 else None
        
    def check_exits(self, dc, d):
        exits = []
        for t, pos in self.positions.items():
            p = self.get_price(dc, t, d)
            if not p: continue
            
            if p > pos['peak']: pos['peak'] = p
            
            ret = (p - pos['entry']) / pos['entry']
            
            # V3.1 Exits
            reason = None
            if p < pos['peak'] * 0.88: # 12% Trailing Stop
                reason = 'Trail -12%'
            elif ret < -0.20: # 20% Hard Stop
                reason = 'HardStop -20%'
                
            if reason:
                self.capital += pos['shares'] * p * (1 - COST_BPS/10000)
                self.trade_log.append({
                    'Ticker': t, 'PnL%': round(ret*100, 2), 
                    'Reason': reason, 'Hold': (d - pos['entry_date']).days,
                    'EntryDate': pos['entry_date'], 'ExitDate': d
                })
                exits.append(t)
                
        for t in exits: del self.positions[t]

    def scan(self, dc, nifty, d, regime):
        cfg = REGIME_CFG.get(regime, REGIME_CFG['UNKNOWN'])
        max_pos = cfg['max_pos']
        if len(self.positions) >= max_pos: return
        
        ni = nifty.index.searchsorted(d)
        nw = nifty.iloc[max(0, ni-252):ni+1] if ni >= 252 else None
        if nw is None: return
        
        cands = get_v31_candidates(dc, nw, d)
        month = d.month
        
        # APPLY SEASONALITY OVERLAY (DNA3-V4 Logic)
        final_cands = []
        for c in cands:
            if c['ticker'] in self.positions: continue
            
            if self.use_seasonality:
                overlay = get_seasonal_overlay(c['sector'], month)
                if overlay == 'TRAP':
                    continue # VETO
                elif overlay == 'EDGE':
                    c['rs'] += 50 # Artificially boost conviction weight to jump to top of list
                    final_cands.append(c)
                else:
                    final_cands.append(c)
            else:
                final_cands.append(c)
                
        final_cands.sort(key=lambda x: -x['rs'])
        
        sel, sc = [], {}
        for c in final_cands:
            sec = c['sector']
            # Max 3 per sector rule
            if sc.get(sec, 0) + sum(1 for t in self.positions if SECTOR_MAP.get(t, 'Unk') == sec) < 3:
                sel.append(c); sc[sec] = sc.get(sec, 0) + 1
            if len(sel) + len(self.positions) >= max_pos: break
            
        eq = self.get_equity(dc, d)
        avail = max(0, self.capital - eq * cfg['cash'])
        free = max_pos - len(self.positions)
        
        for c in sel[:free]:
            p, size = c['price'], avail / (free + 1)
            sh = int(size / p)
            cost = sh * p * (1 + COST_BPS/10000)
            if sh > 0 and avail >= cost and cost > 5000:
                avail -= cost; self.capital -= cost
                self.positions[c['ticker']] = {'entry': p, 'peak': p, 'shares': sh, 'entry_date': d}

    def get_equity(self, dc, d):
        return self.capital + sum(pos['shares'] * (self.get_price(dc, t, d) or 0) for t, pos in self.positions.items())

    def run(self, dc, nifty, start, end):
        self.reset()
        dates = nifty.index[nifty.index.searchsorted(start):nifty.index.searchsorted(end)+1]
        for day, d in enumerate(dates):
            regime = detect_regime(nifty, d)
            self.check_exits(dc, d)
            if day % REBALANCE_FREQ == 0: self.scan(dc, nifty, d, regime)
            self.history.append({'date': d, 'equity': self.get_equity(dc, d), 'regime': regime})
        return pd.DataFrame(self.history)

def calc_metrics(eq_df, years):
    if eq_df is None or len(eq_df) < 2: return None
    s, e = eq_df['equity'].iloc[0], eq_df['equity'].iloc[-1]
    total = (e/s - 1) * 100
    cagr = ((e/s)**(1/years) - 1) * 100 if years > 0 else total
    eq = eq_df.copy()
    eq['pk'] = eq['equity'].cummax()
    eq['dd'] = (eq['equity'] - eq['pk']) / eq['pk'] * 100
    max_dd = eq['dd'].min()
    eq['month'] = eq['date'].dt.to_period('M')
    mo = eq.groupby('month')['equity'].last()
    mr = mo.pct_change().dropna()
    sharpe = (mr.mean() / mr.std()) * np.sqrt(12) if len(mr) > 2 and mr.std() > 0 else 0
    sortino = (mr.mean() / mr[mr < 0].std()) * np.sqrt(12) if len(mr[mr < 0]) > 0 and mr[mr < 0].std() > 0 else 0
    return {'CAGR%': round(cagr, 2), 'Total%': round(total, 2), 'MaxDD%': round(max_dd, 2), 'Sharpe': round(sharpe, 2), 'Sortino': round(sortino, 2)}

def fetch_data():
    start = (datetime.now() - timedelta(days=365*MAX_YEARS + 500)).strftime('%Y-%m-%d')
    print("[1/3] Fetching Nifty...")
    nifty = yf.Ticker("^NSEI").history(start=start)
    if nifty.empty: return None, {}
    nifty.index = nifty.index.tz_localize(None)
    print(f"[2/3] Bulk downloading {len(TICKERS[:500])} stocks...")
    t0 = time.time()
    try:
        bulk = yf.download(TICKERS[:500], start=start, group_by='ticker', threads=True, progress=False, auto_adjust=True)
    except Exception as e:
        print(f"Failed: {e}")
        return nifty, {'NIFTY': nifty}
    dc = {'NIFTY': nifty}
    loaded = 0
    for t in TICKERS[:500]:
        try:
            if t in bulk.columns.get_level_values(0):
                df = bulk[t].dropna(how='all')
                if not df.empty and len(df) > 200:
                    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                    dc[t] = df
                    loaded += 1
        except: pass
    print(f"[3/3] Loaded {loaded} stocks in {time.time()-t0:.0f}s")
    return nifty, dc

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nifty, dc = fetch_data()
    if nifty is None: return
    
    print(f"\n{'=' * 100}")
    print(f"DNA3-V4 (Seasonal Momentum) vs V3.1 (Pure Momentum) (MULTI-HORIZON BACKTEST)")
    print(f"{'=' * 100}")
    
    summary_results = []
    
    for horizon in HORIZONS:
        start = datetime.now() - timedelta(days=int(365.25 * horizon))
        if start < nifty.index[0]:
            start = nifty.index[0]
            
        si = nifty.index.searchsorted(start)
        actual_start, actual_end = nifty.index[si], nifty.index[-1]
        actual_years = (actual_end - actual_start).days / 365.25
        
        if actual_years < 0.4: continue # Too short
        
        ns, ne = nifty.iloc[si]['Close'], nifty.iloc[-1]['Close']
        n_cagr = ((ne/ns)**(1/actual_years) - 1) * 100 if actual_years > 0 else 0
        
        strategies = {
            'V3.1-Pure_Momentum': TrendEngineBase(use_seasonality=False),
            'V4.0-Seasonal_Overlay': TrendEngineBase(use_seasonality=True)
        }
        
        horiz_label = f"{horizon}Y" if horizon >= 1 else "6Mo"
        print(f"\nRunning {horiz_label} Backtest (Actual Years: {actual_years:.2f})...")
        
        results = {}
        for name, engine in strategies.items():
            eq = engine.run(dc, nifty, actual_start, actual_end)
            m = calc_metrics(eq, actual_years)
            sells = engine.trade_log
            wins = [t for t in sells if t['PnL%'] > 0]
            losses = [t for t in sells if t['PnL%'] <= 0]
            wr = len(wins)/len(sells)*100 if sells else 0
            aw = np.mean([t['PnL%'] for t in wins]) if wins else 0
            al = np.mean([t['PnL%'] for t in losses]) if losses else 0
            exp = (wr/100)*aw - (1-wr/100)*abs(al) if sells else 0
            
            m.update({
                'Trades': len(sells), 'WinRate%': round(wr, 1),
                'AvgWin%': round(aw, 1), 'AvgLoss%': round(al, 1),
                'AvgHold': round(np.mean([t['Hold'] for t in sells]), 0) if sells else 0,
                'Expectancy%': round(exp, 2),
            })
            results[name] = m
            
        summary_results.append({
            'Horizon': horiz_label,
            'V4_CAGR': results['V4.0-Seasonal_Overlay'].get('CAGR%', 0),
            'V3_CAGR': results['V3.1-Pure_Momentum'].get('CAGR%', 0),
            'Nifty_CAGR': round(n_cagr, 2),
            'V4_WR': results['V4.0-Seasonal_Overlay'].get('WinRate%', 0),
            'V3_WR': results['V3.1-Pure_Momentum'].get('WinRate%', 0),
            'V4_Exp': results['V4.0-Seasonal_Overlay'].get('Expectancy%', 0),
            'V3_Exp': results['V3.1-Pure_Momentum'].get('Expectancy%', 0),
        })
        
    print(f"\n{'_' * 120}")
    print(f"  CONSOLIDATED HORIZON COMPARISON: V4 (Seasonal) vs V3.1 (Pure)")
    print(f"{'_' * 120}")
    print(f"  {'Horizon':<10} | {'CAGR (V4 vs V3 | Nifty)':<35} | {'Win Rate (V4 vs V3)':<25} | {'Expectancy (V4 vs V3)':<25}")
    print(f"  {'-'*110}")
    for res in summary_results:
        cagr_str = f"{res['V4_CAGR']:>5.1f}% vs {res['V3_CAGR']:>5.1f}% | {res['Nifty_CAGR']:.1f}%"
        wr_str = f"{res['V4_WR']:>5.1f}% vs {res['V3_WR']:>5.1f}%"
        exp_str = f"{res['V4_Exp']:>5.1f}% vs {res['V3_Exp']:>5.1f}%"
        print(f"  {res['Horizon']:<10} | {cagr_str:<35} | {wr_str:<25} | {exp_str:<25}")

if __name__ == "__main__":
    run()
