"""
NARROW MARKET SHIELD STUDY (OptComp-V21)
========================================
Tests the effectiveness of Market Breadth filters on the Nifty 1000 Universe.
Breadth Indicator: % of stocks above 50-day moving average.
Narrow Market Threshold: < 30% participation.

Variants evaluated across 10 & 12 positions:
1. Baseline: OptComp-V21 (No Shield)
2. Shield (Block Entries): If Breadth < 30%, halt all new buying.
3. Shield (Force Cash): If Breadth < 30%, liquidate all positions & halt buying.

Horizons: 6mo, 1y, 3y, 5y, 10y, 15y
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings, os, sys, time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nifty1000_list import TICKERS_1000

warnings.filterwarnings('ignore')

INITIAL_CAPITAL = 1000000
COST_BPS = 50
OUTPUT_DIR = "analysis_2026"
MAX_YEARS = 15
HORIZONS = {'6mo': 0.5, '1y': 1, '3y': 3, '5y': 5, '10y': 10, '15y': 15}

# Create a master sector map from Nifty 1000
try:
    df_sectors = pd.read_csv('data/nifty1000_list.csv')
    SECTOR_MAP = {row['Ticker']: row.get('Macro_Sector', row.get('Sub_Industry', 'Unknown')) for _, row in df_sectors.iterrows()}
except:
    SECTOR_MAP = {}

def tf_rs(sw, nw, days):
    if len(sw) <= days or len(nw) <= days: return 0.0
    try:
        sr = sw['Close'].iloc[-1] / sw['Close'].iloc[-days] - 1
        nr = nw['Close'].iloc[-1] / nw['Close'].iloc[-days] - 1
        return (sr - nr) * 100
    except: return 0.0

def calc_ind(sw, nw):
    if len(sw) < 100 or len(nw) < 64: return None
    p = sw['Close'].iloc[-1]
    ma50 = sw['Close'].rolling(50).mean().iloc[-1]
    rs = tf_rs(sw, nw, 5)*0.1 + tf_rs(sw, nw, 21)*0.5 + tf_rs(sw, nw, 63)*0.4
    vol = sw['Close'].pct_change().dropna()[-60:]
    volatility = vol.std() * np.sqrt(252) * 100 if len(vol) > 10 else 0
    return {'price': p, 'ma50': ma50, 'rs': rs, 'vol': volatility}


class Engine:
    def __init__(self, name, max_pos, shield_mode='None'):
        self.name = name
        self.mx = max_pos
        self.scap = max(3, int(max_pos * 0.35)) 
        self.reb = 13
        self.shield_mode = shield_mode # 'None', 'Block', 'ForceCash'
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trades = []

    def reset(self):
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trades = []

    def gp(self, dc, t, d):
        df = dc.get(t)
        if df is None: return None
        i = df.index.searchsorted(d)
        return df['Close'].iloc[min(i, len(df)-1)] if i > 0 else None

    def equity(self, dc, d):
        v = self.capital
        for t, pos in self.positions.items():
            p = self.gp(dc, t, d)
            if p: v += pos['sh'] * p
        return v

    def check_exits(self, dc, d, breadth_score):
        out = []
        
        # Shield: Force Cash check
        force_liquidate = (self.shield_mode == 'ForceCash' and breadth_score < 30)
        
        for t, pos in self.positions.items():
            p = self.gp(dc, t, d)
            if not p: continue
            if p > pos['pk']: pos['pk'] = p
            ret = (p - pos['en']) / pos['en']
            reason = None

            if force_liquidate:
                reason = 'Shield_Crash'
            else:
                # OptComp-V21 Trailing Logic
                if ret > 0.10:
                    tr = pos['pk'] * 0.90 
                    if tr > pos['stop']: pos['stop'] = tr
                if p < pos['stop']: reason = 'StopLoss'

            if reason:
                proc = pos['sh'] * p * (1 - COST_BPS/10000)
                self.capital += proc
                self.trades.append({
                    'Ticker': t, 'PnL': ret*100,
                    'Days': (d - pos['ed']).days, 'Reason': reason,
                    'Entry': pos['ed'].strftime('%Y-%m-%d'),
                    'Exit': d.strftime('%Y-%m-%d'),
                })
                out.append(t)
        for t in out: del self.positions[t]

    def scan(self, dc, nifty, d, breadth_score):
        # Shield Check: Block Entries
        if self.shield_mode in ['Block', 'ForceCash'] and breadth_score < 30:
            return

        if len(self.positions) >= self.mx: return

        ni = nifty.index.searchsorted(d)
        if ni < 252: return
        nw = nifty.iloc[max(0, ni-252):ni+1]

        cands = []
        for t, df in dc.items():
            if t == 'NIFTY' or t in self.positions: continue
            i = df.index.searchsorted(d)
            if i < 100: continue
            w = df.iloc[max(0, i-252):i+1]
            
            ind = calc_ind(w, nw)
            if not ind: continue
            if ind['rs'] < 2.0 or ind['vol'] < 30 or ind['price'] < ind['ma50']: continue
            cands.append({'t': t, 'ind': ind})

        cands.sort(key=lambda x: -x['ind']['rs'])

        sel = []; sc = {}
        for c in cands:
            sec = SECTOR_MAP.get(c['t'], 'Unknown')
            curr = sum(1 for t in self.positions if SECTOR_MAP.get(t, 'Unknown') == sec)
            if sc.get(sec, 0) + curr < self.scap:
                sel.append(c); sc[sec] = sc.get(sec, 0) + 1
            if len(sel) + len(self.positions) >= self.mx: break

        eq = self.equity(dc, d)
        avail = max(0, self.capital)
        free = self.mx - len(self.positions)
        for c in sel[:free]:
            p = c['ind']['price']
            sz = avail / (free + 1)
            sh = int(sz / p)
            cost = sh * p * (1 + COST_BPS/10000)
            if sh > 0 and avail >= cost and cost > 5000:
                avail -= cost; self.capital -= cost
                stop = p * 0.85
                self.positions[c['t']] = {
                    'en': p, 'pk': p, 'sh': sh, 'stop': stop, 'ed': d
                }

    def run(self, dc, nifty, breadth_series, start, end):
        self.reset()
        si = nifty.index.searchsorted(start)
        ei = nifty.index.searchsorted(end)
        dates = nifty.index[si:ei+1]
        if len(dates) < 10: return None
        day = 0
        for d in dates:
            bi = breadth_series.index.searchsorted(d)
            breadth_score = breadth_series.iloc[min(bi, len(breadth_series)-1)] if bi > 0 else 50
            
            self.check_exits(dc, d, breadth_score)
            if day % self.reb == 0:
                self.scan(dc, nifty, d, breadth_score)
            self.history.append({'date': d, 'equity': self.equity(dc, d), 'breadth': breadth_score})
            day += 1
        return pd.DataFrame(self.history)

def pain_metrics(eq_df, trades, years):
    if eq_df is None or len(eq_df) < 2: return {}
    s, e = eq_df['equity'].iloc[0], eq_df['equity'].iloc[-1]
    cagr = ((e/s) ** (1/years) - 1) * 100 if years > 0 else 0

    eq = eq_df.copy()
    eq['pk'] = eq['equity'].cummax()
    eq['dd'] = (eq['equity'] - eq['pk']) / eq['pk'] * 100
    max_dd = eq['dd'].min()

    eq['month'] = eq['date'].dt.to_period('M')
    mr = eq.groupby('month')['equity'].last().pct_change().dropna()
    sharpe = (mr.mean() / mr.std()) * np.sqrt(12) if len(mr) > 2 and mr.std() > 0 else 0

    sells = [t for t in trades if 'PnL' in t]
    wins = [t for t in sells if t['PnL'] > 0]
    wr = len(wins)/len(sells)*100 if sells else 0
    avg_win = np.mean([t['PnL'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['PnL'] for t in sells if t['PnL'] <= 0]) if len(sells) > len(wins) else 0

    eq['underwater'] = eq['equity'] < eq['pk']
    uw_groups = (eq['underwater'] != eq['underwater'].shift()).cumsum()
    uw_periods = eq[eq['underwater']].groupby(uw_groups).size()
    dd_days = int(uw_periods.max()) if len(uw_periods) > 0 else 0

    return {
        'CAGR%': round(cagr, 2),
        'MaxDD%': round(max_dd, 1),
        'DD_Days': dd_days,
        'Sharpe': round(sharpe, 2),
        'AvgWin%': round(avg_win, 1),
        'AvgLoss%': round(avg_loss, 1),
        'Trades': len(sells),
        'WinRate%': round(wr, 1)
    }

def fetch_data():
    start = (datetime.now() - timedelta(days=365*MAX_YEARS + 500)).strftime('%Y-%m-%d')
    print("[1/3] Fetching Nifty 50 Benchmark...")
    nifty = yf.Ticker("^NSEI").history(start=start)
    nifty.index = nifty.index.tz_localize(None)

    print(f"[2/3] Downloading {len(TICKERS_1000)} stocks from Nifty 1000 (15Y)...")
    t0 = time.time()
    try:
        bulk = yf.download(TICKERS_1000, start=start, group_by='ticker', threads=True, progress=False, auto_adjust=True)
    except:
        bulk = yf.download(TICKERS_1000, start=start, group_by='ticker', threads=False, progress=True, auto_adjust=True)
    
    dc = {'NIFTY': nifty}
    all_closes = {}
    loaded = 0
    for t in TICKERS_1000:
        try:
            if t in bulk.columns.get_level_values(0):
                df = bulk[t].dropna(how='all')
                if len(df) > 200:
                    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                    dc[t] = df
                    all_closes[t] = df['Close']
                    loaded += 1
        except: pass

    print(f"Loaded {loaded} stocks in {time.time()-t0:.0f}s. Nifty: {len(nifty)} days.")
    
    print("[3/3] Precomputing Market Breadth (MA50%)...")
    close_df = pd.DataFrame(all_closes)
    ma50_df = close_df.rolling(50).mean()
    above_ma50 = (close_df > ma50_df).sum(axis=1)
    total_active = close_df.notna().sum(axis=1)
    breadth_series = (above_ma50 / total_active) * 100
    breadth_series = breadth_series.fillna(0)
    
    return nifty, dc, breadth_series

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nifty, dc, breadth_series = fetch_data()
    if nifty is None or nifty.empty: return

    now = datetime.now()
    
    VARIANTS = [
        ("Base_10P", 10, 'None'),
        ("Shield_Block_10P", 10, 'Block'),
        ("Shield_ForceCash_10P", 10, 'ForceCash'),
        ("Base_12P", 12, 'None'),
        ("Shield_Block_12P", 12, 'Block'),
        ("Shield_ForceCash_12P", 12, 'ForceCash'),
    ]

    all_summary = []

    print("\n" + "=" * 130)
    print("OPTCOMP-V21 NARROW MARKET SHIELD STUDY")
    print("=" * 130)

    for hname, years in list(HORIZONS.items())[::-1]: 
        s_dt = now - timedelta(days=int(365.25 * years))
        si = nifty.index.searchsorted(s_dt)
        if si >= len(nifty) - 10: continue

        actual_start = nifty.index[si]
        actual_end = nifty.index[-1]
        actual_years = (actual_end - actual_start).days / 365.25

        ns = nifty.iloc[si]['Close']; ne = nifty.iloc[-1]['Close']
        n_cagr = ((ne/ns)**(1/actual_years) - 1)*100

        print(f"\n{'_' * 130}")
        print(f"  {hname.upper()} ({actual_start.date()} -> {actual_end.date()}, {actual_years:.1f}y)")
        print(f"{'_' * 130}")
        
        h = f"  {'Metric':<14}"
        for vname, _, _ in VARIANTS: h += f" {vname:>20}"
        h += f" {'Nifty':>9}"
        print(h); print(f"  {'-' * 145}")

        strat_m = {}
        row = {'Horizon': hname, 'Years': round(actual_years, 1), 'Nifty_CAGR%': round(n_cagr, 2)}
        
        for vname, mx, smode in VARIANTS:
            eng = Engine(vname, mx, smode)
            eq = eng.run(dc, nifty, breadth_series, actual_start, actual_end)
            m = pain_metrics(eq, eng.trades, actual_years) if eq is not None else {}
            strat_m[vname] = m
            for k, v in m.items(): row[f'{vname}_{k}'] = v

        for metric in ['CAGR%', 'MaxDD%', 'DD_Days', 'AvgWin%', 'AvgLoss%', 'Trades', 'WinRate%']:
            line = f"  {metric:<14}"
            for vname, _, _ in VARIANTS:
                v = strat_m[vname].get(metric, '-')
                line += f" {v:>20}" if isinstance(v, (int, float)) else f" {'-':>20}"
            nv = round(n_cagr, 2) if metric == 'CAGR%' else '-'
            line += f" {nv:>9}"
            print(line)

        all_summary.append(row)
        
        cagrs = {v: strat_m[v].get('CAGR%', -999) for v, _, _ in VARIANTS}
        best = max(cagrs, key=cagrs.get)
        print(f"\n  >>> WINNER for {hname}: {best} ({cagrs[best]:.1f}% CAGR)")

    pd.DataFrame(all_summary).to_csv(f"{OUTPUT_DIR}/narrow_shield_study.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/narrow_shield_study.csv")
    print("=" * 130)

if __name__ == "__main__":
    run()
