"""
REBALANCE FREQUENCY SENSITIVITY ANALYSIS
========================================
Tests the V3.1 Momentum Engine across various operational cadences
to find the optimal tradeoff between returns, drawdowns, and 
brokerage churn (trading costs).

Test frequencies: 5 Days, 10 Days, 15 Days, 21 Days (1 month), 30 Days
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
OUTPUT_DIR = "analysis_2026/rebalance_optimization"
YEARS = 5 # 5 Years to get enough trades for statistical significance
FREQUENCIES = [5, 10, 15, 21, 30]

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
# SCAN LOGIC
# ============================================================
def get_v31_candidates(dc, nifty_window, d):
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
    def __init__(self, rebalance_freq):
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
        self.rebalance_freq = rebalance_freq
        
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
            if p < pos['peak'] * 0.88: # 12% Trailing Stop (Tight Risk Management)
                reason = 'Trail -12%'
            elif ret < -0.20: # 20% Hard Stop
                reason = 'HardStop -20%'
                
            if reason:
                # Deduct exit cost
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
        
        sel, sc = [], {}
        for c in cands:
            if c['ticker'] in self.positions: continue
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
            cost = sh * p * (1 + COST_BPS/10000) # Deduct entry cost
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
            # EXITS ARE CHECKED EVERY SINGLE DAY (not bound to rebalance freq to limit risk!)
            self.check_exits(dc, d)
            # ONLY ENTIRES AND RE-SCANS HAPPEN ON THE REBALANCE FREQUENCY
            if day % self.rebalance_freq == 0: 
                self.scan(dc, nifty, d, regime)
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
    start = (datetime.now() - timedelta(days=365*YEARS + 500)).strftime('%Y-%m-%d')
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
    
    start = datetime.now() - timedelta(days=int(365.25 * YEARS))
    si = nifty.index.searchsorted(start)
    actual_start, actual_end = nifty.index[si], nifty.index[-1]
    actual_years = (actual_end - actual_start).days / 365.25
    ns, ne = nifty.iloc[si]['Close'], nifty.iloc[-1]['Close']
    n_cagr = ((ne/ns)**(1/actual_years) - 1) * 100
    
    print(f"\n{'=' * 100}")
    print(f"REBALANCE FREQUENCY OPITMIZATION (V3.1 MOMENTUM ENGINE) ({YEARS}Y BACKTEST)")
    print(f"{'=' * 100}")
    
    summary_results = []
    
    for freq in FREQUENCIES:
        horiz_label = f"{freq} Days"
        print(f"\nRunning {horiz_label} Scan Rebalance Frequency...")
        
        engine = TrendEngineBase(rebalance_freq=freq)
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
        
        summary_results.append({
            'Freq': horiz_label,
            'CAGR': m.get('CAGR%', 0),
            'MaxDD': m.get('MaxDD%', 0),
            'Sharpe': m.get('Sharpe', 0),
            'WR': m.get('WinRate%', 0),
            'Trades': m.get('Trades', 0),
            'Hold': m.get('AvgHold', 0)
        })
        
    print(f"\n{'_' * 110}")
    print(f"  OPERATIONAL SENSITIVITY TABLE: How Often Should You Scan/Rebalance?")
    print(f"  Note: Exits (stops) are still checked daily for safety in all runs.")
    print(f"{'_' * 110}")
    print(f"  {'Frequency':<15} | {'CAGR%':<10} | {'Max DD%':<10} | {'Sharpe':<10} | {'WinRate%':<10} | {'Tot Trades':<10} | {'AvgHold (Days)':<15}")
    print(f"  {'-'*100}")
    for res in summary_results:
        print(f"  {res['Freq']:<15} | {res['CAGR']:>9.1f}% | {res['MaxDD']:>9.1f}% | {res['Sharpe']:>10.2f} | {res['WR']:>9.1f}% | {res['Trades']:>10} | {res['Hold']:>15}")

if __name__ == "__main__":
    run()
