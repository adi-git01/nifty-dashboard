"""
DNA3-V3.1 HYBRID BACKTEST & 4-WAY COMPARISON
=============================================
V3.1 = V2.1's entry rules + V3's exit/risk mechanics

DNA3-V2.1:  RS > 2%, Vol > 30%, Price > MA50 | -15% stop, 10% trailing | 10 pos, 4/sector
DNA3-V3:    RS > 10-20% (regime), Price > MA50, near high | 12% trailing | 6-12 pos, 3/sector
DNA3-V3.1:  RS > 2%, Vol > 30%, Price > MA50 | 12% trailing | 6-12 pos (regime), 3/sector

Comparison: 6mo, 1y, 3y, 5y, 10y + rolling returns + regime breakdown vs Nifty

Usage:
  python dna3_v31_hybrid_comparison.py
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
MAX_YEARS = 10
REBALANCE_FREQ = 10

HORIZONS = {'6mo': 0.5, '1y': 1, '3y': 3, '5y': 5, '10y': 10}


# ============================================================
# REGIME DETECTION
# ============================================================
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


# V3/V3.1 regime config
REGIME_CONFIG = {
    'BULL':      {'rs_min_v3': 10, 'max_pos': 12, 'cash_reserve': 0.05, 'need_near_high': False},
    'MILD_BULL': {'rs_min_v3': 15, 'max_pos': 10, 'cash_reserve': 0.10, 'need_near_high': False},
    'SIDEWAYS':  {'rs_min_v3': 15, 'max_pos':  8, 'cash_reserve': 0.20, 'need_near_high': True},
    'BEAR':      {'rs_min_v3': 20, 'max_pos':  6, 'cash_reserve': 0.40, 'need_near_high': True},
    'UNKNOWN':   {'rs_min_v3': 15, 'max_pos':  8, 'cash_reserve': 0.20, 'need_near_high': False},
}


# ============================================================
# INDICATOR CALC
# ============================================================
def calc_indicators(stock_window, nifty_window):
    if len(stock_window) < 100 or len(nifty_window) < 64:
        return None
    
    price = stock_window['Close'].iloc[-1]
    ma50 = stock_window['Close'].rolling(50).mean().iloc[-1]
    ma200_s = stock_window['Close'].rolling(200).mean()
    ma200 = ma200_s.iloc[-1] if len(stock_window) >= 200 and not pd.isna(ma200_s.iloc[-1]) else ma50
    high_20 = stock_window['Close'].rolling(20).max().iloc[-1]
    
    # RS 63d
    if len(stock_window) > 63 and len(nifty_window) > 63:
        rs_stock = (price - stock_window['Close'].iloc[-63]) / stock_window['Close'].iloc[-63]
        rs_nifty = (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-63]) / nifty_window['Close'].iloc[-63]
        rs_score = (rs_stock - rs_nifty) * 100
    else:
        rs_score = 0
    
    # Volatility 60d
    rets = stock_window['Close'].pct_change().dropna()[-60:]
    volatility = rets.std() * np.sqrt(252) * 100 if len(rets) > 10 else 0
    
    # Liquidity
    vol_20d = stock_window['Volume'].rolling(20).mean().iloc[-1]
    liquidity = vol_20d * price
    
    return {
        'price': price, 'ma50': ma50, 'ma200': ma200,
        'high_20': high_20, 'rs_score': rs_score,
        'volatility': volatility, 'liquidity': liquidity,
    }


# ============================================================
# STRATEGY ENGINE
# ============================================================
class Engine:
    """Unified engine for v21, v3, or v31."""
    
    def __init__(self, name, stype):
        self.name = name
        self.stype = stype  # 'v21', 'v3', 'v31'
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
    
    def reset(self):
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
    
    def get_price(self, dc, t, date):
        if t not in dc: return None
        df = dc[t]
        idx = df.index.searchsorted(date)
        if idx == 0: return None
        return df['Close'].iloc[min(idx, len(df)-1)]
    
    # --- ENTRY FILTERS ---
    def passes_filter(self, ind, regime):
        if self.stype == 'v21':
            # V2.1: RS > 2%, Vol > 30%, Price > MA50
            return ind['rs_score'] >= 2.0 and ind['volatility'] >= 30 and ind['price'] > ind['ma50']
        
        elif self.stype == 'v3':
            # V3: RS > regime-threshold, Price > MA50, near high (if regime requires)
            cfg = REGIME_CONFIG.get(regime, REGIME_CONFIG['UNKNOWN'])
            if ind['rs_score'] < cfg['rs_min_v3']: return False
            if ind['price'] < ind['ma50']: return False
            if ind['liquidity'] < 5_000_000: return False
            if cfg['need_near_high'] and ind['price'] < ind['high_20'] * 0.98: return False
            return True
        
        elif self.stype == 'v31':
            # V3.1 HYBRID: V2.1 entry rules + min liquidity
            if ind['rs_score'] < 2.0: return False
            if ind['volatility'] < 30: return False
            if ind['price'] < ind['ma50']: return False
            if ind['liquidity'] < 5_000_000: return False
            return True
    
    # --- EXIT LOGIC ---
    def check_exits(self, dc, date):
        to_exit = []
        for t, pos in self.positions.items():
            price = self.get_price(dc, t, date)
            if not price: continue
            if price > pos['peak']: pos['peak'] = price
            ret = (price - pos['entry']) / pos['entry']
            reason = None
            
            if self.stype == 'v21':
                # V2.1: -15% hard stop + trailing after +10%
                if ret > 0.10:
                    trail = pos['peak'] * 0.90
                    if trail > pos['stop']: pos['stop'] = trail
                if price < pos['stop']:
                    reason = 'Stop/Trail'
            
            elif self.stype in ('v3', 'v31'):
                # V3/V3.1: 12% trailing stop always active + -20% hard stop
                if price < pos['peak'] * 0.88:
                    reason = 'TrailingStop'
                if ret < -0.20:
                    reason = 'HardStop'
            
            if reason:
                proceeds = pos['shares'] * price * (1 - COST_BPS / 10000)
                self.capital += proceeds
                self.trade_log.append({
                    'Ticker': t, 'PnL%': round(ret * 100, 2),
                    'Reason': reason,
                    'Entry_Date': pos['entry_date'].strftime('%Y-%m-%d'),
                    'Exit_Date': date.strftime('%Y-%m-%d'),
                    'Hold_Days': (date - pos['entry_date']).days,
                })
                to_exit.append(t)
        
        for t in to_exit: del self.positions[t]
    
    # --- POSITION SIZING ---
    def get_max_pos(self, regime):
        if self.stype == 'v21': return 10
        return REGIME_CONFIG.get(regime, REGIME_CONFIG['UNKNOWN'])['max_pos']
    
    def get_cash_reserve(self, regime):
        if self.stype == 'v21': return 0.0
        return REGIME_CONFIG.get(regime, REGIME_CONFIG['UNKNOWN'])['cash_reserve']
    
    def get_sector_cap(self):
        return 4 if self.stype == 'v21' else 3
    
    # --- SCAN & BUY ---
    def scan_and_buy(self, dc, nifty, date, regime):
        max_pos = self.get_max_pos(regime)
        
        # Reduce positions if regime tightened
        if self.stype in ('v3', 'v31') and len(self.positions) > max_pos:
            pos_list = []
            for t, pos in self.positions.items():
                p = self.get_price(dc, t, date)
                if p:
                    ret = (p - pos['entry']) / pos['entry']
                    pos_list.append((t, ret, p, pos))
            pos_list.sort(key=lambda x: x[1])
            while len(self.positions) > max_pos and pos_list:
                t, ret, p, pos = pos_list.pop(0)
                if t in self.positions:
                    proceeds = pos['shares'] * p * (1 - COST_BPS / 10000)
                    self.capital += proceeds
                    self.trade_log.append({
                        'Ticker': t, 'PnL%': round(ret*100, 2),
                        'Reason': 'RegimeReduce',
                        'Entry_Date': pos['entry_date'].strftime('%Y-%m-%d'),
                        'Exit_Date': date.strftime('%Y-%m-%d'),
                        'Hold_Days': (date - pos['entry_date']).days,
                    })
                    del self.positions[t]
        
        if len(self.positions) >= max_pos: return
        
        nifty_idx = nifty.index.searchsorted(date)
        if nifty_idx < 252: return
        nw = nifty.iloc[max(0, nifty_idx-252):nifty_idx+1]
        
        candidates = []
        for t in dc:
            if t == 'NIFTY' or t in self.positions: continue
            df = dc[t]
            idx = df.index.searchsorted(date)
            if idx < 100: continue
            w = df.iloc[max(0, idx-252):idx+1]
            if len(w) < 100: continue
            ind = calc_indicators(w, nw)
            if ind and self.passes_filter(ind, regime):
                candidates.append({'ticker': t, 'ind': ind})
        
        candidates.sort(key=lambda x: -x['ind']['rs_score'])
        
        # Sector cap
        scap = self.get_sector_cap()
        selected = []
        sc = {}
        for c in candidates:
            sec = SECTOR_MAP.get(c['ticker'], 'Unknown')
            curr = sum(1 for t in self.positions if SECTOR_MAP.get(t, 'Unknown') == sec)
            if sc.get(sec, 0) + curr < scap:
                selected.append(c)
                sc[sec] = sc.get(sec, 0) + 1
            if len(selected) + len(self.positions) >= max_pos: break
        
        # Cash reserve enforcement
        total_eq = self.get_equity(dc, date)
        min_cash = total_eq * self.get_cash_reserve(regime)
        avail = max(0, self.capital - min_cash)
        
        free = max_pos - len(self.positions)
        for c in selected[:free]:
            price = c['ind']['price']
            size = avail / (free + 1)
            shares = int(size / price)
            if shares < 1: continue
            cost = shares * price * (1 + COST_BPS / 10000)
            if avail >= cost and cost > 5000:
                avail -= cost
                self.capital -= cost
                stop = price * 0.85 if self.stype == 'v21' else price * 0.80
                self.positions[c['ticker']] = {
                    'entry': price, 'peak': price, 'shares': shares,
                    'stop': stop, 'entry_date': date,
                }
    
    def get_equity(self, dc, date):
        val = self.capital
        for t, pos in self.positions.items():
            p = self.get_price(dc, t, date)
            if p: val += pos['shares'] * p
        return val
    
    def run(self, dc, nifty, start, end):
        self.reset()
        si = nifty.index.searchsorted(start)
        ei = nifty.index.searchsorted(end)
        dates = nifty.index[si:ei+1]
        if len(dates) < 10: return None
        
        day = 0
        for d in dates:
            regime = detect_regime(nifty, d)
            self.check_exits(dc, d)
            if day % REBALANCE_FREQ == 0:
                self.scan_and_buy(dc, nifty, d, regime)
            eq = self.get_equity(dc, d)
            self.history.append({'date': d, 'equity': eq, 'regime': regime})
            day += 1
        
        return pd.DataFrame(self.history)


# ============================================================
# ANALYTICS
# ============================================================
def calc_metrics(eq_df, years):
    if eq_df is None or len(eq_df) < 2: return None
    s, e = eq_df['equity'].iloc[0], eq_df['equity'].iloc[-1]
    total = (e / s - 1) * 100
    cagr = ((e / s) ** (1 / years) - 1) * 100 if years > 0 else total
    
    eq = eq_df.copy()
    eq['pk'] = eq['equity'].cummax()
    eq['dd'] = (eq['equity'] - eq['pk']) / eq['pk'] * 100
    max_dd = eq['dd'].min()
    
    eq['month'] = eq['date'].dt.to_period('M')
    mo = eq.groupby('month')['equity'].last()
    mr = mo.pct_change().dropna()
    sharpe = (mr.mean() / mr.std()) * np.sqrt(12) if len(mr) > 2 and mr.std() > 0 else 0
    neg = mr[mr < 0]
    sortino = (mr.mean() / neg.std()) * np.sqrt(12) if len(neg) > 0 and neg.std() > 0 else 0
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0
    
    return {
        'Total%': round(total, 2), 'CAGR%': round(cagr, 2),
        'MaxDD%': round(max_dd, 2), 'Sharpe': round(sharpe, 2),
        'Sortino': round(sortino, 2), 'Calmar': round(calmar, 2),
    }


def calc_rolling(eq_df, window=12):
    if eq_df is None or len(eq_df) < 22: return pd.DataFrame()
    eq = eq_df.copy()
    eq['month'] = eq['date'].dt.to_period('M')
    mo = eq.groupby('month').agg({'equity': 'last', 'date': 'last'}).reset_index()
    mo['ret'] = mo['equity'].pct_change(window) * 100
    return mo[['date', 'ret']].dropna()


def calc_regime_perf(eq_df):
    res = {}
    for r in ['BULL', 'MILD_BULL', 'SIDEWAYS', 'BEAR']:
        rows = eq_df[eq_df['regime'] == r]
        if len(rows) < 2:
            res[r] = {'days': 0, 'total': 0, 'ann': 0}
            continue
        dr = rows['equity'].pct_change().dropna()
        total = (np.prod(1 + dr) - 1) * 100
        days = len(rows)
        ann = ((1 + total/100) ** (252/days) - 1) * 100 if days > 20 else total
        res[r] = {'days': days, 'total': round(total, 2), 'ann': round(ann, 2)}
    return res


# ============================================================
# DATA LOADING
# ============================================================
def fetch_data():
    start = (datetime.now() - timedelta(days=365 * MAX_YEARS + 500)).strftime('%Y-%m-%d')
    
    print("[1/3] Fetching Nifty 50...")
    nifty = yf.Ticker("^NSEI").history(start=start)
    if nifty.empty:
        print("ERROR: No Nifty data!")
        return None, {}
    nifty.index = nifty.index.tz_localize(None)
    
    print(f"[2/3] Bulk downloading {len(TICKERS[:500])} stocks...")
    t0 = time.time()
    try:
        bulk = yf.download(TICKERS[:500], start=start, group_by='ticker', threads=True, progress=True, auto_adjust=True)
    except Exception as e:
        print(f"Bulk failed: {e}")
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
    
    print(f"[3/3] Loaded {loaded} stocks in {time.time()-t0:.0f}s. Nifty: {len(nifty)} days.")
    return nifty, dc


# ============================================================
# MAIN
# ============================================================
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nifty, dc = fetch_data()
    if nifty is None: return
    
    now = datetime.now()
    strategies = [
        ('DNA3-V2.1', 'v21'),
        ('DNA3-V3',   'v3'),
        ('DNA3-V3.1', 'v31'),
    ]
    
    # ================================
    # PART 1: MULTI-HORIZON CAGR
    # ================================
    print("\n" + "=" * 100)
    print("PART 1: MULTI-HORIZON COMPARISON (V2.1 vs V3 vs V3.1 vs Nifty)")
    print("=" * 100)
    
    all_summary = []
    longest_eq = {}
    longest_trades = {}
    longest_name = None
    
    for hname, years in HORIZONS.items():
        start_dt = now - timedelta(days=int(365.25 * years))
        end_dt = now - timedelta(days=1)
        
        si = nifty.index.searchsorted(start_dt)
        if si >= len(nifty) - 10:
            print(f"\n  [{hname}] Insufficient data, skipping.")
            continue
        
        actual_start = nifty.index[si]
        actual_end = nifty.index[-1]
        actual_years = (actual_end - actual_start).days / 365.25
        
        # Nifty benchmark
        ns = nifty.iloc[si]['Close']
        ne = nifty.iloc[-1]['Close']
        n_total = (ne / ns - 1) * 100
        n_cagr = ((ne / ns) ** (1 / actual_years) - 1) * 100 if actual_years > 0 else n_total
        
        nifty_eq = nifty.iloc[si:].copy()
        nifty_eq['equity'] = nifty_eq['Close'] / ns * INITIAL_CAPITAL
        nifty_eq['date'] = nifty_eq.index
        nifty_eq['regime'] = 'N/A'
        n_m = calc_metrics(nifty_eq[['date', 'equity', 'regime']], actual_years)
        
        row = {'Horizon': hname, 'Years': round(actual_years, 1),
               'Nifty_CAGR%': round(n_cagr, 2), 'Nifty_Total%': round(n_total, 2),
               'Nifty_MaxDD%': n_m['MaxDD%'] if n_m else 'N/A'}
        
        print(f"\n{'_' * 100}")
        print(f"  {hname.upper()} ({actual_start.date()} to {actual_end.date()})")
        print(f"{'_' * 100}")
        
        header = f"  {'Metric':<18}"
        for sn, _ in strategies:
            header += f" {sn:>12}"
        header += f" {'Nifty':>12}"
        print(header)
        print(f"  {'-'*74}")
        
        strat_metrics = {}
        
        for sname, stype in strategies:
            eng = Engine(sname, stype)
            eq = eng.run(dc, nifty, actual_start, actual_end)
            
            if eq is not None and len(eq) > 10:
                m = calc_metrics(eq, actual_years)
                sells = [t for t in eng.trade_log if 'PnL%' in t]
                wins = [t for t in sells if t['PnL%'] > 0]
                
                row[f'{sname}_CAGR%'] = m['CAGR%']
                row[f'{sname}_Total%'] = m['Total%']
                row[f'{sname}_MaxDD%'] = m['MaxDD%']
                row[f'{sname}_Sharpe'] = m['Sharpe']
                row[f'{sname}_Sortino'] = m['Sortino']
                row[f'{sname}_Calmar'] = m['Calmar']
                row[f'{sname}_Alpha%'] = round(m['CAGR%'] - n_cagr, 2)
                row[f'{sname}_Trades'] = len(sells)
                row[f'{sname}_WinRate%'] = round(len(wins)/len(sells)*100, 1) if sells else 0
                
                strat_metrics[sname] = m
                
                # Store longest
                if hname == max(HORIZONS.keys(), key=lambda k: HORIZONS[k]):
                    longest_eq[sname] = eq
                    longest_trades[sname] = eng.trade_log
                    longest_name = hname
            else:
                strat_metrics[sname] = None
        
        # Print compact table
        for metric in ['CAGR%', 'Total%', 'MaxDD%', 'Sharpe', 'Alpha%', 'Trades', 'WinRate%']:
            line = f"  {metric:<18}"
            for sn, _ in strategies:
                val = row.get(f'{sn}_{metric}', 'N/A')
                line += f" {val:>12}" if isinstance(val, (int, float)) else f" {'N/A':>12}"
            n_val = row.get(f'Nifty_{metric}', '-')
            line += f" {n_val:>12}" if isinstance(n_val, (int, float)) else f" {'-':>12}"
            print(line)
        
        all_summary.append(row)
    
    pd.DataFrame(all_summary).to_csv(f"{OUTPUT_DIR}/dna3_v31_summary.csv", index=False)
    
    # ================================
    # PART 2: ROLLING RETURNS
    # ================================
    print(f"\n\n{'=' * 100}")
    print("PART 2: ROLLING RETURNS (10Y Data)")
    print("=" * 100)
    
    # Nifty rolling
    si = nifty.index.searchsorted(now - timedelta(days=int(365.25 * MAX_YEARS)))
    nifty_eq = nifty.iloc[si:].copy()
    nifty_eq['equity'] = nifty_eq['Close'] / nifty_eq['Close'].iloc[0] * INITIAL_CAPITAL
    nifty_eq['date'] = nifty_eq.index
    nifty_eq['regime'] = 'N/A'
    
    for window, label in [(3, '3-Month'), (6, '6-Month'), (12, '12-Month')]:
        print(f"\n  Rolling {label} Returns:")
        print(f"  {'Strategy':<15} {'Median':>8} {'Mean':>8} {'Best':>8} {'Worst':>8} {'%Pos':>6} {'%>15%':>6}")
        print(f"  {'-'*60}")
        
        all_roll = []
        for sn in list(longest_eq.keys()) + ['Nifty']:
            if sn == 'Nifty':
                r = calc_rolling(nifty_eq[['date', 'equity', 'regime']], window)
            elif sn in longest_eq:
                r = calc_rolling(longest_eq[sn], window)
            else:
                continue
            
            if len(r) > 0:
                s = r['ret']
                above_15 = (s > 15).mean() * 100
                print(f"  {sn:<15} {s.median():>+7.1f}% {s.mean():>+7.1f}% {s.max():>+7.1f}% {s.min():>+7.1f}% {(s>0).mean()*100:>5.0f}% {above_15:>5.0f}%")
                r['Strategy'] = sn
                r['Window'] = label
                all_roll.append(r)
        
        if all_roll:
            pd.concat(all_roll).to_csv(f"{OUTPUT_DIR}/dna3_v31_rolling_{window}m.csv", index=False)
    
    # ================================
    # PART 3: REGIME PERFORMANCE
    # ================================
    print(f"\n\n{'=' * 100}")
    print("PART 3: PER-REGIME PERFORMANCE (10Y)")
    print("=" * 100)
    
    regime_rows = []
    for sn in longest_eq:
        rp = calc_regime_perf(longest_eq[sn])
        print(f"\n  {sn}:")
        print(f"  {'Regime':<12} {'Days':>6} {'Total':>10} {'Ann.':>10}")
        print(f"  {'-'*40}")
        for r in ['BULL', 'MILD_BULL', 'SIDEWAYS', 'BEAR']:
            d = rp[r]
            print(f"  {r:<12} {d['days']:>6} {d['total']:>+9.1f}% {d['ann']:>+9.1f}%")
            regime_rows.append({'Strategy': sn, 'Regime': r, 'Days': d['days'], 'Total%': d['total'], 'Ann%': d['ann']})
    
    pd.DataFrame(regime_rows).to_csv(f"{OUTPUT_DIR}/dna3_v31_regime.csv", index=False)
    
    # ================================
    # PART 4: TRADE ANALYSIS
    # ================================
    print(f"\n\n{'=' * 100}")
    print("PART 4: TRADE ANALYSIS (10Y)")
    print("=" * 100)
    
    trade_summary = []
    for sn in longest_trades:
        trades = longest_trades[sn]
        sells = [t for t in trades if 'PnL%' in t]
        wins = [t for t in sells if t['PnL%'] > 0]
        losses = [t for t in sells if t['PnL%'] <= 0]
        
        wr = len(wins)/len(sells)*100 if sells else 0
        aw = np.mean([t['PnL%'] for t in wins]) if wins else 0
        al = np.mean([t['PnL%'] for t in losses]) if losses else 0
        mw = max([t['PnL%'] for t in sells]) if sells else 0
        ml = min([t['PnL%'] for t in sells]) if sells else 0
        ah = np.mean([t['Hold_Days'] for t in sells]) if sells else 0
        exp = (wr/100)*aw - (1-wr/100)*abs(al) if sells else 0
        
        print(f"\n  {sn}:")
        print(f"    Trades     : {len(sells)}")
        print(f"    Win Rate   : {wr:.0f}%")
        print(f"    Avg Win    : +{aw:.1f}%  |  Avg Loss: {al:.1f}%")
        print(f"    Max Win    : +{mw:.1f}%  |  Max Loss: {ml:.1f}%")
        print(f"    Avg Hold   : {ah:.0f} days")
        print(f"    Expectancy : {exp:+.2f}% per trade")
        
        trade_summary.append({
            'Strategy': sn, 'Trades': len(sells), 'WinRate%': round(wr,1),
            'AvgWin%': round(aw,1), 'AvgLoss%': round(al,1),
            'MaxWin%': round(mw,1), 'MaxLoss%': round(ml,1),
            'AvgHold': round(ah), 'Expectancy%': round(exp,2),
        })
    
    pd.DataFrame(trade_summary).to_csv(f"{OUTPUT_DIR}/dna3_v31_trades_summary.csv", index=False)
    
    # ================================
    # PART 5: VERDICT
    # ================================
    print(f"\n\n{'=' * 100}")
    print("PART 5: VERDICT")
    print("=" * 100)
    
    # Count CAGR wins per horizon
    wins_count = {sn: 0 for sn, _ in strategies}
    for row in all_summary:
        best_cagr = -999
        best_name = ''
        for sn, _ in strategies:
            c = row.get(f'{sn}_CAGR%', -999)
            if isinstance(c, (int, float)) and c > best_cagr:
                best_cagr = c
                best_name = sn
        if best_name:
            wins_count[best_name] += 1
    
    print(f"\n  CAGR Wins Across {len(all_summary)} Horizons:")
    for sn, cnt in wins_count.items():
        bar = '*' * cnt
        print(f"    {sn:<12}: {cnt} {bar}")
    
    # Risk comparison (longest horizon)
    print(f"\n  Risk-Adjusted Comparison (10Y):")
    for metric in ['Sharpe', 'Sortino', 'Calmar']:
        vals = {}
        for sn, _ in strategies:
            v = all_summary[-1].get(f'{sn}_{metric}', 0)
            vals[sn] = v if isinstance(v, (int, float)) else 0
        winner = max(vals, key=vals.get)
        parts = " | ".join([f"{sn}={vals[sn]}" for sn in vals])
        print(f"    {metric:<10}: {parts}  --> {winner}")
    
    # Alpha comparison
    print(f"\n  Alpha vs Nifty (CAGR difference):")
    for row in all_summary:
        h = row['Horizon']
        parts = []
        for sn, _ in strategies:
            a = row.get(f'{sn}_Alpha%', 'N/A')
            parts.append(f"{sn}={a:+.1f}%" if isinstance(a, (int, float)) else f"{sn}=N/A")
        print(f"    {h:>4s}: {' | '.join(parts)}")
    
    print(f"\n  Current Regime: {detect_regime(nifty, nifty.index[-1])}")
    
    # V3.1 specific highlight
    v31_10y = all_summary[-1] if all_summary else {}
    v31_cagr = v31_10y.get('DNA3-V3.1_CAGR%', 'N/A')
    v31_dd = v31_10y.get('DNA3-V3.1_MaxDD%', 'N/A')
    v31_sharpe = v31_10y.get('DNA3-V3.1_Sharpe', 'N/A')
    
    print(f"\n  DNA3-V3.1 HYBRID PROFILE (10Y):")
    print(f"    CAGR:     {v31_cagr}%")
    print(f"    Max DD:   {v31_dd}%")
    print(f"    Sharpe:   {v31_sharpe}")
    
    print(f"\n  Files saved to {OUTPUT_DIR}/dna3_v31_*.csv")


if __name__ == "__main__":
    print("=" * 100)
    print("DNA3-V3.1 HYBRID BACKTEST & 4-WAY COMPARISON")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 100)
    run()
