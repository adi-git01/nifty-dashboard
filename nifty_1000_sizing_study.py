"""
MULTI-HORIZON PORTFOLIO SIZING STUDY
====================================
Evaluates the OptComp-V21 Strategy across two dimensions:
1. Universe: Nifty 500 vs Nifty 1000
2. Portfolio Size: 10, 12, 15 positions
Horizons: 6mo, 1y, 3y, 5y, 15y

Strategy: OptComp-V21 (13d rebalance, Composite RS 1W/1M/3M, 15% Trail Stop)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings, os, sys, time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nifty500_list import TICKERS as TICKERS_500
from utils.nifty500_list import SECTOR_MAP as SECTOR_MAP_500
from utils.nifty1000_list import TICKERS_1000
from utils.nifty1000_list import SUB_INDUSTRY_MAP

warnings.filterwarnings('ignore')

INITIAL_CAPITAL = 1000000
COST_BPS = 50
OUTPUT_DIR = "analysis_2026"
MAX_YEARS = 15
HORIZONS = {'6mo': 0.5, '1y': 1, '3y': 3, '5y': 5, '15y': 15}

# Create a master sector map from Nifty 1000
SECTOR_MAP = {}
for _, row in pd.read_csv('data/nifty1000_list.csv').iterrows():
    SECTOR_MAP[row['Ticker']] = row.get('Macro_Sector', row.get('Sub_Industry', 'Unknown'))


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
    
    # Composite RS: 10% 1W, 50% 1M, 40% 3M
    rs = tf_rs(sw, nw, 5)*0.1 + tf_rs(sw, nw, 21)*0.5 + tf_rs(sw, nw, 63)*0.4

    vol = sw['Close'].pct_change().dropna()[-60:]
    volatility = vol.std() * np.sqrt(252) * 100 if len(vol) > 10 else 0
    liq = sw['Volume'].rolling(20).mean().iloc[-1] * p

    return {'price': p, 'ma50': ma50, 'rs': rs, 'vol': volatility, 'liq': liq}


class Engine:
    def __init__(self, name, universe_list, max_pos):
        self.name = name
        self.universe_list = set(universe_list)
        self.mx = max_pos
        # Scap = max allowed per macro sector (approx 40% of portfolio)
        self.scap = max(2, int(max_pos * 0.40)) 
        self.reb = 13 # OptComp-V21 Rebalance
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

    def check_exits(self, dc, d):
        out = []
        for t, pos in self.positions.items():
            p = self.gp(dc, t, d)
            if not p: continue
            if p > pos['pk']: pos['pk'] = p
            ret = (p - pos['en']) / pos['en']
            reason = None

            # 15% Trailing Stop
            tr = pos['pk'] * 0.85 
            if tr > pos['stop']: pos['stop'] = tr
            
            # 15% Hard Stop from Entry is implied by the first trailing stop marker
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

    def scan(self, dc, nifty, d):
        if len(self.positions) >= self.mx: return

        ni = nifty.index.searchsorted(d)
        if ni < 252: return
        nw = nifty.iloc[max(0, ni-252):ni+1]

        cands = []
        for t, df in dc.items():
            if t == 'NIFTY' or t not in self.universe_list or t in self.positions: continue
            i = df.index.searchsorted(d)
            if i < 100: continue
            w = df.iloc[max(0, i-252):i+1]
            
            ind = calc_ind(w, nw)
            if not ind: continue
            if ind['rs'] < 2.0: continue
            if ind['vol'] < 30: continue
            if ind['price'] < ind['ma50']: continue
            
            cands.append({'t': t, 'ind': ind})

        cands.sort(key=lambda x: -x['ind']['rs'])

        # Sector cap
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

    def run(self, dc, nifty, start, end):
        self.reset()
        si = nifty.index.searchsorted(start)
        ei = nifty.index.searchsorted(end)
        dates = nifty.index[si:ei+1]
        if len(dates) < 10: return None
        day = 0
        for d in dates:
            self.check_exits(dc, d)
            if day % self.reb == 0:
                self.scan(dc, nifty, d)
            self.history.append({'date': d, 'equity': self.equity(dc, d)})
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
    whip = sum(1 for t in sells if t['Days'] < 10) / len(sells) * 100 if sells else 0

    return {
        'CAGR%': round(cagr, 2),
        'MaxDD%': round(max_dd, 1),
        'Sharpe': round(sharpe, 2),
        'Trades': len(sells),
        'WinRate%': round(wr, 1),
        'Whipsaw%': round(whip, 1)
    }

def fetch_data():
    start = (datetime.now() - timedelta(days=365*MAX_YEARS + 500)).strftime('%Y-%m-%d')
    print("[1/2] Fetching Nifty 50 Benchmark...")
    nifty = yf.Ticker("^NSEI").history(start=start)
    nifty.index = nifty.index.tz_localize(None)

    print(f"[2/2] Downloading {len(TICKERS_1000)} stocks from Nifty 1000 (15Y)...")
    t0 = time.time()
    bulk = yf.download(TICKERS_1000, start=start, group_by='ticker', threads=True, progress=True, auto_adjust=True)
    
    dc = {'NIFTY': nifty}
    loaded = 0
    for t in TICKERS_1000:
        try:
            if t in bulk.columns.get_level_values(0):
                df = bulk[t].dropna(how='all')
                if len(df) > 200:
                    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                    dc[t] = df; loaded += 1
        except: pass

    print(f"Loaded {loaded} stocks in {time.time()-t0:.0f}s. Nifty: {len(nifty)} days.")
    return nifty, dc

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nifty, dc = fetch_data()
    if nifty is None or nifty.empty: return

    now = datetime.now()
    
    VARIANTS = [
        ("N500_10Pos", TICKERS_500, 10),
        ("N500_12Pos", TICKERS_500, 12),
        ("N500_15Pos", TICKERS_500, 15),
        ("N1000_10Pos", TICKERS_1000, 10),
        ("N1000_12Pos", TICKERS_1000, 12),
        ("N1000_15Pos", TICKERS_1000, 15),
    ]

    all_summary = []

    print("\n" + "=" * 110)
    print("OPTCOMP-V21 SIZING & UNIVERSE COMPARISON")
    print("=" * 110)

    for hname, years in list(HORIZONS.items())[::-1]: # DO 15Y first
        s_dt = now - timedelta(days=int(365.25 * years))
        si = nifty.index.searchsorted(s_dt)
        if si >= len(nifty) - 10: continue

        actual_start = nifty.index[si]
        actual_end = nifty.index[-1]
        actual_years = (actual_end - actual_start).days / 365.25

        ns = nifty.iloc[si]['Close']; ne = nifty.iloc[-1]['Close']
        n_cagr = ((ne/ns)**(1/actual_years) - 1)*100

        print(f"\n{'_' * 110}")
        print(f"  {hname.upper()} ({actual_start.date()} -> {actual_end.date()}, {actual_years:.1f}y)")
        print(f"{'_' * 110}")
        
        h = f"  {'Metric':<12}"
        for vname, _, _ in VARIANTS: h += f" {vname:>12}"
        h += f" {'Nifty':>9}"
        print(h); print(f"  {'-' * 105}")

        strat_m = {}
        row = {'Horizon': hname, 'Years': round(actual_years, 1), 'Nifty_CAGR%': round(n_cagr, 2)}
        
        for vname, uni, mx in VARIANTS:
            eng = Engine(vname, uni, mx)
            eq = eng.run(dc, nifty, actual_start, actual_end)
            m = pain_metrics(eq, eng.trades, actual_years) if eq is not None else {}
            strat_m[vname] = m
            for k, v in m.items(): row[f'{vname}_{k}'] = v

        for metric in ['CAGR%', 'MaxDD%', 'Sharpe', 'Trades', 'WinRate%', 'Whipsaw%']:
            line = f"  {metric:<12}"
            for vname, _, _ in VARIANTS:
                v = strat_m[vname].get(metric, '-')
                line += f" {v:>12}" if isinstance(v, (int, float)) else f" {'-':>12}"
            nv = round(n_cagr, 2) if metric == 'CAGR%' else '-'
            line += f" {nv:>9}"
            print(line)

        all_summary.append(row)
        
        cagrs = {v: strat_m[v].get('CAGR%', -999) for v, _, _ in VARIANTS}
        best = max(cagrs, key=cagrs.get)
        print(f"\n  >>> WINNER for {hname}: {best} ({cagrs[best]:.1f}% CAGR)")

    pd.DataFrame(all_summary).to_csv(f"{OUTPUT_DIR}/sizing_universe_study.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/sizing_universe_study.csv")
    print("=" * 110)

if __name__ == "__main__":
    run()
