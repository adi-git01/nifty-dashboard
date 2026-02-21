"""
SUB-PORTFOLIO CYCLES: GENERATIONAL WEALTH vs FAST FLIPS
=======================================================
Tests whether applying completely different risk-management rules 
based on a sector's historical Cycle Length (from the Encyclopedia) 
beats treating all stocks uniformly with V3.1 logic.

Logic:
1. Generational Wealth (Long Cycles > 35 months) 
   - Industries: Auto Makers, Heavy Machinery, Metal Fab, Med Care, etc.
   - Rules: NO profit targets. Very wide Trailing Stop (-20%). Hard Stop (-20%).

2. Fast Flips (Short Cycles < 24 months)
   - Industries: Insurance, Capital Markets, Specialty Chemicals, Real Estate, etc.
   - Rules: Hard Target (+20%). Tight Trailing Stop (-8%). Hard Stop (-10%).

3. Baseline V3.1
   - Uniform Trailing Stop (-12%), Hard Stop (-20%) for everything.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os
import sys
import time
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nifty500_list import TICKERS, SECTOR_MAP

warnings.filterwarnings('ignore')

INITIAL_CAPITAL = 1000000
COST_BPS = 50
OUTPUT_DIR = "analysis_2026/cycle_portfolios"
YEARS = 7 # 7 Years backtest
REBALANCE_FREQ = 21 # Utilizing our new 21-day "sweet spot" optimal cadence

# CYCLE MAPPINGS (Extracted from 15Y Encyclopedia)
LONG_CYCLE_KWS = [
    'Auto Manufacturer', 'Farm & Heavy', 'Metal Fabrication', 'Medical Care', 
    'Conglomerates', 'Specialty Industrial Machinery', 'Agricultural Inputs', 
    'Regulated Electric', 'Software - Application', 'Aluminum', 'Thermal Coal'
]

SHORT_CYCLE_KWS = [
    'Insurance', 'Capital Market', 'Specialty Chemical', 'Tobacco', 
    'Independent Power Producer', 'Regulated Gas', 'Tools & Accessories', 
    'Real Estate', 'Lodging', 'Diagnostic', 'Broker'
]

def classify_cycle(industry_str):
    if not industry_str: return 'MID'
    s = str(industry_str).lower()
    for kw in LONG_CYCLE_KWS:
        if kw.lower() in s: return 'LONG'
    for kw in SHORT_CYCLE_KWS:
        if kw.lower() in s: return 'SHORT'
    return 'MID'

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
# BASE SCAN
# ============================================================
def get_candidates(dc, nifty_window, d, info_cache):
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
        else: rs = 0
            
        rets = w['Close'].pct_change().dropna()[-60:]
        vol = rets.std() * np.sqrt(252) * 100 if len(rets) > 10 else 0
        liq = w['Volume'].rolling(20).mean().iloc[-1] * price
        
        if rs >= 2.0 and vol >= 30 and price > ma50 and liq >= 5_000_000:
            ind = info_cache.get(t, {}).get('industry', '')
            cycle = classify_cycle(ind)
            cands.append({'ticker': t, 'rs': rs, 'price': price, 'sector': SECTOR_MAP.get(t, 'Unk'), 'cycle': cycle})
            
    cands.sort(key=lambda x: -x['rs'])
    return cands

# ============================================================
# ENGINE CLASSES
# ============================================================
class TrendEngineBase:
    def __init__(self, mode):
        self.mode = mode # 'BASELINE', 'GENERATIONAL', 'FAST_FLIP'
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
        
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
            
            reason = None
            if self.mode == 'BASELINE':
                if p < pos['peak'] * 0.88: reason = 'Trail -12%'
                elif ret < -0.20: reason = 'Hard Stop -20%'
                    
            elif self.mode == 'GENERATIONAL':
                if p < pos['peak'] * 0.80: reason = 'Wide Trail -20%'
                elif ret < -0.20: reason = 'Hard Stop -20%'
                    
            elif self.mode == 'FAST_FLIP':
                if ret >= 0.20: reason = 'Target +20%'
                elif p < pos['peak'] * 0.92: reason = 'Tight Trail -8%'
                elif ret < -0.10: reason = 'Hard Stop -10%'
                
            if reason:
                self.capital += pos['shares'] * p * (1 - COST_BPS/10000)
                self.trade_log.append({
                    'Ticker': t, 'PnL%': round(ret*100, 2), 
                    'Reason': reason, 'Hold': (d - pos['entry_date']).days,
                    'EntryDate': pos['entry_date'], 'ExitDate': d
                })
                exits.append(t)
                
        for t in exits: del self.positions[t]

    def scan(self, dc, nifty, d, regime, info_cache):
        cfg = REGIME_CFG.get(regime, REGIME_CFG['UNKNOWN'])
        max_pos = cfg['max_pos']
        if len(self.positions) >= max_pos: return
        
        ni = nifty.index.searchsorted(d)
        nw = nifty.iloc[max(0, ni-252):ni+1] if ni >= 252 else None
        if nw is None: return
        
        cands = get_candidates(dc, nw, d, info_cache)
        
        sel, sc = [], {}
        for c in cands:
            if c['ticker'] in self.positions: continue
            
            if self.mode == 'GENERATIONAL' and c['cycle'] != 'LONG': continue
            if self.mode == 'FAST_FLIP' and c['cycle'] != 'SHORT': continue
            
            sec = c['sector']
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

    def run(self, dc, nifty, start, end, info_cache):
        self.reset()
        dates = nifty.index[nifty.index.searchsorted(start):nifty.index.searchsorted(end)+1]
        for day, d in enumerate(dates):
            regime = detect_regime(nifty, d)
            self.check_exits(dc, d)
            if day % REBALANCE_FREQ == 0: self.scan(dc, nifty, d, regime, info_cache)
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
    return {'CAGR%': round(cagr, 2), 'Total%': round(total, 2), 'MaxDD%': round(max_dd, 2), 'Sharpe': round(sharpe, 2)}

def fetch_data():
    start = (datetime.now() - timedelta(days=365*YEARS + 500)).strftime('%Y-%m-%d')
    print("[1/4] Fetching Nifty...")
    nifty = yf.Ticker("^NSEI").history(start=start)
    if nifty.empty: return None, {}, {}
    nifty.index = nifty.index.tz_localize(None)
    
    print("[2/4] Loading yfinance industry cache...")
    info_cache = {}
    try:
        with open("analysis_2026/encyclopedia/yfinance_info_cache.json", "r") as f:
            info_cache = json.load(f)
    except:
        print("Warning: info cache not found. Using empty.")
        
    print(f"[3/4] Bulk downloading {len(TICKERS[:500])} stocks...")
    t0 = time.time()
    try:
        bulk = yf.download(TICKERS[:500], start=start, group_by='ticker', threads=True, progress=False, auto_adjust=True)
    except Exception as e:
        print(f"Failed: {e}")
        return nifty, {'NIFTY': nifty}, info_cache
        
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
    print(f"[4/4] Loaded {loaded} stocks in {time.time()-t0:.0f}s")
    return nifty, dc, info_cache

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nifty, dc, info_cache = fetch_data()
    if nifty is None: return
    
    start = datetime.now() - timedelta(days=int(365.25 * YEARS))
    si = nifty.index.searchsorted(start)
    actual_start, actual_end = nifty.index[si], nifty.index[-1]
    actual_years = (actual_end - actual_start).days / 365.25
    
    print(f"\n{'=' * 100}")
    print(f"SUB-PORTFOLIO CYCLES: GENERATIONAL WEALTH vs FAST FLIPS ({YEARS}Y BACKTEST)")
    print(f"{'=' * 100}")
    
    strategies = {
        'Baseline Uniform V3.1': TrendEngineBase(mode='BASELINE'),
        'Generational Wealth (Wide Trail)': TrendEngineBase(mode='GENERATIONAL'),
        'Fast Flips (Targets + Tight Trail)': TrendEngineBase(mode='FAST_FLIP')
    }
    
    results = {}
    
    for name, engine in strategies.items():
        print(f"  Running {name}...")
        eq = engine.run(dc, nifty, actual_start, actual_end, info_cache)
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
        
        # Save Trade logs to verify exit reasons
        df_trades = pd.DataFrame(sells)
        if not df_trades.empty:
            df_trades.to_csv(f"{OUTPUT_DIR}/{name.replace(' ', '_')}_trades.csv", index=False)
    
    print(f"\n{'_' * 115}")
    print(f"  CYCLE-BASED RISK MANAGEMENT COMPARISON")
    print(f"{'_' * 115}")
    print(f"  {'Strategy':<35} | {'CAGR%':<8} | {'MaxDD%':<8} | {'WinRate%':<10} | {'Expectancy':<12} | {'Trades':<8} | {'AvgHold'}")
    print(f"  {'-'*110}")
    for name, res in results.items():
        print(f"  {name:<35} | {res['CAGR%']:>7.1f}% | {res['MaxDD%']:>7.1f}% | {res['WinRate%']:>9.1f}% | {res['Expectancy%']:>11.1f}% | {res['Trades']:>8} | {res['AvgHold']:>6.0f}d")

    # Exit reason breakdown
    print(f"\n  EXIT REASON BREAKDOWN:")
    for name in results.keys():
        df = pd.read_csv(f"{OUTPUT_DIR}/{name.replace(' ', '_')}_trades.csv")
        reasons = df['Reason'].value_counts(normalize=True) * 100
        print(f"  {name}:")
        for r, pct in reasons.items():
            print(f"    - {r}: {pct:.1f}%")

if __name__ == "__main__":
    run()
