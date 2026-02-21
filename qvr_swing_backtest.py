"""
QVR-SWING STRATEGY BACKTEST
===================================
Concept:
- Entry: QVR Logic (Quality >= 6, Value >= 6, Trend 40-65)
  - Buy high-quality, attractively-valued companies recovering from a bottom.
- Exit: SWING / MEAN REVERSION Logic
  - Take profits quickly at +20% (Hard Target)
  - Cut losses at -10% (Hard Stop)
  - Time stop at 60 days if neither is hit
- Sizing: Regime-adaptive

Comparison vs standard DNA3-V3.1 over 5 years.
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
OUTPUT_DIR = "analysis_2026/qvr_swing"
YEARS = 5
REBALANCE_FREQ = 15

# Mean Reversion Swing Parameters
PROFIT_TARGET = 0.20
STOP_LOSS = -0.10
MAX_HOLD_DAYS = 60

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
# SCORE PROXIES
# ============================================================
def calc_trend_score(window):
    if len(window) < 252: return 50
    price = window['Close'].iloc[-1]
    ma50 = window['Close'].rolling(50).mean().iloc[-1]
    ma200 = window['Close'].rolling(200).mean().iloc[-1]
    high_52 = window['High'].rolling(252, min_periods=50).max().iloc[-1]
    low_52 = window['Low'].rolling(252, min_periods=50).min().iloc[-1]
    
    score = 50
    if ma50 > 0: score += 15 if price > ma50 else -10
    if ma200 > 0: score += 15 if price > ma200 else -15
    if ma50 > 0 and ma200 > 0: score += 10 if ma50 > ma200 else -5
    
    if high_52 > low_52:
        position = (price - low_52) / (high_52 - low_52)
        score += int((position - 0.5) * 30)
        dist = (price - high_52) / high_52 * 100
        if dist > -5: score += 10
        elif dist < -30: score -= 10
    return max(0, min(100, score))

def calc_quality_proxy(window, nifty_window):
    if len(window) < 252: return 5.0
    rets = window['Close'].pct_change().dropna()[-252:]
    vol = rets.std() * np.sqrt(252) * 100
    s_vol = max(0, min(10, 10 - (vol - 20) / 5))
    
    monthly = window['Close'].resample('ME').last().pct_change().dropna()[-12:]
    s_consist = (monthly > 0).mean() * 10 if len(monthly) > 3 else 5.0
    
    peak = window['Close'].cummax()
    dd = ((window['Close'] - peak) / peak).min() * 100
    s_dd = max(0, min(10, 10 + (dd + 15) / 3.5))
    
    above_200 = (window['Close'] > window['Close'].rolling(200).mean()).iloc[-252:].mean()
    s_above = above_200 * 10
    
    return round(max(0, min(10, s_vol*0.3 + s_consist*0.25 + s_dd*0.25 + s_above*0.2)), 1)

def calc_value_proxy(window, nifty_window):
    if len(window) < 252: return 5.0
    price = window['Close'].iloc[-1]
    high_52 = window['High'].rolling(252, min_periods=50).max().iloc[-1]
    ma200 = window['Close'].rolling(200).mean().iloc[-1]
    
    dist_high = (price - high_52) / high_52 * 100
    if dist_high > -5: s_dist = 1.0
    elif dist_high > -15: s_dist = 3.0 + (-dist_high - 5) / 10 * 4
    elif dist_high > -30: s_dist = 7.0 + (-dist_high - 15) / 15 * 2
    else: s_dist = 9.0
    
    dist_200 = (price - ma200) / ma200 * 100
    if dist_200 < -15: s_200 = 9.0
    elif dist_200 < -5: s_200 = 6.0 + (-dist_200 - 5) / 10 * 3
    elif dist_200 < 5: s_200 = 4.0 + (5 - dist_200) / 10 * 2
    else: s_200 = 2.0
    
    if len(nifty_window) > 126:
        rs = ((price - window['Close'].iloc[-126]) / window['Close'].iloc[-126] - 
              (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-126]) / nifty_window['Close'].iloc[-126]) * 100
        if rs < -20: s_rs = 9.0
        elif rs < -5: s_rs = 5.0 + (-rs - 5) / 15 * 4
        elif rs < 5: s_rs = 4.0
        else: s_rs = 2.0
    else: s_rs = 5.0
    
    if len(window) > 63:
        ret_3m = (price - window['Close'].iloc[-63]) / window['Close'].iloc[-63] * 100
        if ret_3m < -20: s_3m = 9.0
        elif ret_3m < -5: s_3m = 5.0 + (-ret_3m - 5) / 15 * 4
        elif ret_3m < 5: s_3m = 4.0
        else: s_3m = 2.0
    else: s_3m = 5.0
    
    return round(max(0, min(10, s_dist*0.35 + s_200*0.25 + s_rs*0.2 + s_3m*0.2)), 1)


# ============================================================
# QVR-SWING ENGINE
# ============================================================
class QVRSwingEngine:
    def __init__(self, target=PROFIT_TARGET, stop=STOP_LOSS, time_stop=MAX_HOLD_DAYS):
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
        
        self.target = target
        self.stop = stop
        self.time_stop = time_stop
    
    def reset(self): self.__init__(self.target, self.stop, self.time_stop)
    
    def get_price(self, dc, t, d):
        if t not in dc: return None
        i = dc[t].index.searchsorted(d)
        return dc[t]['Close'].iloc[min(i, len(dc[t])-1)] if i > 0 else None
    
    def check_exits(self, dc, d):
        exits = []
        for t, pos in self.positions.items():
            p = self.get_price(dc, t, d)
            if not p: continue
            
            ret = (p - pos['entry']) / pos['entry']
            days_held = (d - pos['entry_date']).days
            
            reason = None
            if ret >= self.target:
                reason = f'Target +{int(self.target*100)}%'
            elif ret <= self.stop:
                reason = f'StopLoss {int(self.stop*100)}%'
            elif days_held >= self.time_stop:
                reason = f'TimeStop {self.time_stop}d'
            
            if reason:
                self.capital += pos['shares'] * p * (1 - COST_BPS/10000)
                self.trade_log.append({
                    'Ticker': t, 'PnL%': round(ret*100, 2),
                    'Reason': reason, 'Hold': days_held
                })
                exits.append(t)
                
        for t in exits: del self.positions[t]
    
    def scan(self, dc, nifty, d, regime):
        cfg = REGIME_CFG.get(regime, REGIME_CFG['UNKNOWN'])
        max_pos = cfg['max_pos']
        if len(self.positions) >= max_pos: return
        
        ni = nifty.index.searchsorted(d)
        if ni < 252: return
        nw = nifty.iloc[max(0, ni-252):ni+1]
        
        cands = []
        for t in dc:
            if t == 'NIFTY' or t in self.positions: continue
            idx = dc[t].index.searchsorted(d)
            if idx < 252: continue
            w = dc[t].iloc[max(0, idx-252):idx+1]
            if len(w) < 200: continue
            
            trend = calc_trend_score(w)
            quality = calc_quality_proxy(w, nw)
            value = calc_value_proxy(w, nw)
            liq = w['Volume'].rolling(20).mean().iloc[-1] * w['Close'].iloc[-1]
            
            # FUSION ENTRY
            if quality >= 6.0 and value >= 6.0 and 40 <= trend <= 65 and liq >= 5_000_000:
                cands.append({
                    'ticker': t, 'score': quality + value, 'price': w['Close'].iloc[-1]
                })
        
        cands.sort(key=lambda x: -x['score'])
        
        sel, sc = [], {}
        for c in cands:
            sec = SECTOR_MAP.get(c['ticker'], 'Unk')
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


# ============================================================
# V3.1 ENGINE (For clean baseline)
# ============================================================
class V31Engine:
    def __init__(self):
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
    
    def reset(self): self.__init__()
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
            reason = 'Trail' if p < pos['peak'] * 0.88 else 'HardStop' if ret < -0.20 else None
            if reason:
                self.capital += pos['shares'] * p * (1 - COST_BPS/10000)
                self.trade_log.append({'Ticker': t, 'PnL%': round(ret*100, 2), 'Reason': reason, 'Hold': (d - pos['entry_date']).days})
                exits.append(t)
        for t in exits: del self.positions[t]
    
    def scan(self, dc, nifty, d, regime):
        cfg = REGIME_CFG.get(regime, REGIME_CFG['UNKNOWN'])
        max_pos = cfg['max_pos']
        if len(self.positions) >= max_pos: return
        
        ni = nifty.index.searchsorted(d)
        nw = nifty.iloc[max(0, ni-252):ni+1] if ni >= 252 else None
        if nw is None: return
        
        cands = []
        for t in dc:
            if t == 'NIFTY' or t in self.positions: continue
            idx = dc[t].index.searchsorted(d)
            if idx < 100: continue
            w = dc[t].iloc[max(0, idx-252):idx+1]
            if len(w) < 100: continue
            
            price, ma50 = w['Close'].iloc[-1], w['Close'].rolling(50).mean().iloc[-1]
            rs = ((price - w['Close'].iloc[-63])/w['Close'].iloc[-63] - 
                  (nw['Close'].iloc[-1] - nw['Close'].iloc[-63])/nw['Close'].iloc[-63]) * 100 if len(w)>63 else 0
            
            rets = w['Close'].pct_change().dropna()[-60:]
            vol = rets.std() * np.sqrt(252) * 100 if len(rets) > 10 else 0
            liq = w['Volume'].rolling(20).mean().iloc[-1] * price
            
            if rs >= 2.0 and vol >= 30 and price > ma50 and liq >= 5_000_000:
                cands.append({'ticker': t, 'rs': rs, 'price': price})
        
        cands.sort(key=lambda x: -x['rs'])
        sel, sc = [], {}
        for c in cands:
            sec = SECTOR_MAP.get(c['ticker'], 'Unk')
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
    print(f"QVR-SWING (Target +20%, Stop -10%, Time 60d) vs DNA3-V3.1 ({YEARS}Y BACKTEST)")
    print(f"{'=' * 100}")
    
    strategies = {
        'QVR-Swing': QVRSwingEngine(target=PROFIT_TARGET, stop=STOP_LOSS, time_stop=MAX_HOLD_DAYS), 
        'DNA3-V3.1': V31Engine()
    }
    results, trade_stats = {}, {}
    
    for name, engine in strategies.items():
        print(f"  Running {name}...")
        eq = engine.run(dc, nifty, actual_start, actual_end)
        m = calc_metrics(eq, actual_years)
        sells = engine.trade_log
        wins = [t for t in sells if t['PnL%'] > 0]
        losses = [t for t in sells if t['PnL%'] <= 0]
        wr = len(wins)/len(sells)*100 if sells else 0
        aw = np.mean([t['PnL%'] for t in wins]) if wins else 0
        al = np.mean([t['PnL%'] for t in losses]) if losses else 0
        exp = (wr/100)*aw - (1-wr/100)*abs(al) if sells else 0
        
        results[name] = m
        results[name].update({
            'Trades': len(sells), 'WinRate%': round(wr, 1),
            'AvgWin%': round(aw, 1), 'AvgLoss%': round(al, 1),
            'MaxWin%': round(max([t['PnL%'] for t in sells], default=0), 1),
            'AvgHold': round(np.mean([t['Hold'] for t in sells]), 0) if sells else 0,
            'Expectancy%': round(exp, 2),
        })
        trade_stats[name] = sells
    
    print(f"\n{'_' * 100}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"{'_' * 100}")
    
    print(f"\n  {'Metric':<18} {'QVR-Swing':>12} {'V3.1 (Mom)':>12} {'Nifty':>12}")
    print(f"  {'-'*58}")
    for k in ['CAGR%', 'MaxDD%', 'Sharpe', 'Trades', 'WinRate%', 'AvgWin%', 'AvgLoss%', 'MaxWin%', 'AvgHold', 'Expectancy%']:
        f = results['QVR-Swing'].get(k, '-')
        v = results['DNA3-V3.1'].get(k, '-')
        n = round(n_cagr, 2) if k == 'CAGR%' else '-'
        print(f"  {k:<18} {f:>12} {v:>12} {n:>12}")
    
    # QVR-Swing Exit Reasons Breakdown
    print(f"\n{'_' * 100}")
    print(f"  QVR-SWING EXIT ANALYSIS")
    print(f"{'_' * 100}")
    sells = trade_stats['QVR-Swing']
    if sells:
        reasons = {}
        for t in sells:
            base_r = t['Reason'].split()[0]
            reasons[base_r] = reasons.get(base_r, 0) + 1
            
        for r, c in reasons.items():
            print(f"  {r:<18} {c:>5} trades ({c/len(sells)*100:.1f}%)")

if __name__ == "__main__":
    run()
