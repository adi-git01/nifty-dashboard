"""
DNA3 ULTIMATE 15-YEAR COMPARISON
=================================
5-Way Battle with Full Behavioral Pain Metrics:

  1. Optimized V3.1 (Composite 10/50/40, G-Factor Filter, 15d rebalance, regime sizing)
  2. Optimized V2.1 (Composite 10/50/40, G-Factor Filter, 15d rebalance, NO regime sizing)
  3. Legacy V3.1   (63d RS, 10d rebalance, regime sizing)
  4. Legacy V2.1   (63d RS, 10d rebalance, NO regime sizing)
  5. Nifty 50      (Buy & Hold)

Horizons: 6mo, 1y, 3y, 5y, 10y, 15y
Metrics:  CAGR, Sharpe, Sortino, MaxDD, DD Duration, Rolling Returns,
          Win/Loss, Expectancy, Whipsaws, Max Consecutive Losses,
          Longest Negative Period, Euphoric vs Pain periods

Usage:
  python dna3_ultimate_comparison.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict
import warnings, os, sys, time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nifty500_list import TICKERS, SECTOR_MAP

warnings.filterwarnings('ignore')

INITIAL_CAPITAL = 1000000
COST_BPS = 50
OUTPUT_DIR = "analysis_2026"
MAX_YEARS = 15
HORIZONS = {'6mo': 0.5, '1y': 1, '3y': 3, '5y': 5, '10y': 10, '15y': 15}

# ============================================================
# REGIME DETECTION
# ============================================================
def detect_regime(nifty_df, date):
    idx = nifty_df.index.searchsorted(date)
    if idx < 200: return 'UNKNOWN'
    w = nifty_df.iloc[max(0, idx-252):idx+1]
    if len(w) < 63: return 'UNKNOWN'
    p = w['Close'].iloc[-1]
    ma50 = w['Close'].rolling(50).mean().iloc[-1]
    ma200 = w['Close'].rolling(200).mean().iloc[-1]
    r3m = (p - w['Close'].iloc[-63]) / w['Close'].iloc[-63] * 100
    pk = w['Close'].cummax().iloc[-1]
    dd = (p - pk) / pk * 100
    if p > ma50 and ma50 > ma200 and r3m > 5: return 'BULL'
    elif p > ma50 and r3m > 0: return 'MILD_BULL'
    elif p < ma50 and (r3m < -5 or dd < -10): return 'BEAR'
    else: return 'SIDEWAYS'

REGIME_CFG = {
    'BULL':      {'max_pos': 12, 'cash': 0.05},
    'MILD_BULL': {'max_pos': 10, 'cash': 0.10},
    'SIDEWAYS':  {'max_pos':  8, 'cash': 0.20},
    'BEAR':      {'max_pos':  6, 'cash': 0.40},
    'UNKNOWN':   {'max_pos':  8, 'cash': 0.20},
}

# ============================================================
# MULTI-TIMEFRAME RS + G-FACTOR
# ============================================================
def tf_rs(sw, nw, days):
    if len(sw) <= days or len(nw) <= days: return 0.0
    try:
        sr = sw['Close'].iloc[-1] / sw['Close'].iloc[-days] - 1
        nr = nw['Close'].iloc[-1] / nw['Close'].iloc[-days] - 1
        return (sr - nr) * 100
    except: return 0.0

def g_factor(sw, nw, lookback=20):
    if len(sw) < lookback+1 or len(nw) < lookback+1: return 0.0
    sr = sw['Close'].pct_change().dropna()[-lookback:]
    nr = nw['Close'].pct_change().dropna()[-lookback:]
    g = 0.0
    for d in nr.index.intersection(sr.index):
        if nr.loc[d] < -0.01:
            ex = sr.loc[d] - nr.loc[d]
            if ex > 0: g += ex * 100
    return g

def calc_ind(sw, nw, cfg):
    if len(sw) < 100 or len(nw) < 64: return None
    p = sw['Close'].iloc[-1]
    ma50 = sw['Close'].rolling(50).mean().iloc[-1]

    if cfg['use_composite']:
        w = cfg['weights']
        rs = tf_rs(sw, nw, 5)*w[0] + tf_rs(sw, nw, 21)*w[1] + tf_rs(sw, nw, 63)*w[2]
    else:
        rs = tf_rs(sw, nw, 63)

    gs = g_factor(sw, nw) if cfg['g_mode'] != 'Off' else 0

    vol = sw['Close'].pct_change().dropna()[-60:]
    volatility = vol.std() * np.sqrt(252) * 100 if len(vol) > 10 else 0
    liq = sw['Volume'].rolling(20).mean().iloc[-1] * p

    return {'price': p, 'ma50': ma50, 'rs': rs, 'gs': gs,
            'vol': volatility, 'liq': liq}


# ============================================================
# STRATEGY CONFIG
# ============================================================
STRATEGIES = {
    'OptV3.1': {
        'use_composite': True, 'weights': (0.1, 0.5, 0.4),
        'g_mode': 'Filter', 'rebalance': 15, 'stype': 'v31',
    },
    'OptV2.1': {
        'use_composite': True, 'weights': (0.1, 0.5, 0.4),
        'g_mode': 'Filter', 'rebalance': 15, 'stype': 'v21',
    },
    'LegV3.1': {
        'use_composite': False, 'weights': (0, 0, 1),
        'g_mode': 'Off', 'rebalance': 10, 'stype': 'v31',
    },
    'LegV2.1': {
        'use_composite': False, 'weights': (0, 0, 1),
        'g_mode': 'Off', 'rebalance': 10, 'stype': 'v21',
    },
}


# ============================================================
# ENGINE
# ============================================================
class Engine:
    def __init__(self, name, cfg):
        self.name = name
        self.cfg = cfg
        self.stype = cfg['stype']
        self.reb = cfg['rebalance']
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

            if self.stype == 'v21':
                if ret > 0.10:
                    tr = pos['pk'] * 0.90
                    if tr > pos['stop']: pos['stop'] = tr
                if p < pos['stop']: reason = 'Stop'
            else:  # v31
                if p < pos['pk'] * 0.88: reason = 'Trail'
                if ret < -0.20: reason = 'Hard'

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

    def scan(self, dc, nifty, d, regime):
        if self.stype == 'v21':
            mx = 10; cash_r = 0.0; scap = 4
        else:
            rc = REGIME_CFG.get(regime, REGIME_CFG['UNKNOWN'])
            mx = rc['max_pos']; cash_r = rc['cash']; scap = 3

        # Regime reduction (V3.1 only)
        if self.stype == 'v31' and len(self.positions) > mx:
            pl = []
            for t, pos in self.positions.items():
                p = self.gp(dc, t, d)
                if p: pl.append((t, (p-pos['en'])/pos['en'], p, pos))
            pl.sort(key=lambda x: x[1])
            while len(self.positions) > mx and pl:
                t, ret, p, pos = pl.pop(0)
                if t in self.positions:
                    proc = pos['sh'] * p * (1 - COST_BPS/10000)
                    self.capital += proc
                    self.trades.append({
                        'Ticker': t, 'PnL': ret*100,
                        'Days': (d - pos['ed']).days, 'Reason': 'Regime',
                        'Entry': pos['ed'].strftime('%Y-%m-%d'),
                        'Exit': d.strftime('%Y-%m-%d'),
                    })
                    del self.positions[t]

        if len(self.positions) >= mx: return

        ni = nifty.index.searchsorted(d)
        if ni < 252: return
        nw = nifty.iloc[max(0, ni-252):ni+1]

        cands = []
        for t, df in dc.items():
            if t == 'NIFTY' or t in self.positions: continue
            i = df.index.searchsorted(d)
            if i < 100: continue
            w = df.iloc[max(0, i-252):i+1]
            ind = calc_ind(w, nw, self.cfg)
            if not ind: continue
            if ind['rs'] < 2.0: continue
            if ind['vol'] < 30: continue
            if ind['price'] < ind['ma50']: continue
            if self.stype == 'v31' and ind['liq'] < 5e6: continue
            if self.cfg['g_mode'] == 'Filter' and ind['gs'] <= 0: continue
            cands.append({'t': t, 'ind': ind})

        cands.sort(key=lambda x: -x['ind']['rs'])

        # Sector cap
        sel = []; sc = {}
        for c in cands:
            sec = SECTOR_MAP.get(c['t'], 'Unk')
            curr = sum(1 for t in self.positions if SECTOR_MAP.get(t, 'Unk') == sec)
            if sc.get(sec, 0) + curr < scap:
                sel.append(c); sc[sec] = sc.get(sec, 0) + 1
            if len(sel) + len(self.positions) >= mx: break

        eq = self.equity(dc, d)
        avail = max(0, self.capital - eq * cash_r)
        free = mx - len(self.positions)
        for c in sel[:free]:
            p = c['ind']['price']
            sz = avail / (free + 1)
            sh = int(sz / p)
            cost = sh * p * (1 + COST_BPS/10000)
            if sh > 0 and avail >= cost and cost > 5000:
                avail -= cost; self.capital -= cost
                stop = p * 0.85 if self.stype == 'v21' else p * 0.80
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
            regime = detect_regime(nifty, d)
            self.check_exits(dc, d)
            if day % self.reb == 0:
                self.scan(dc, nifty, d, regime)
            self.history.append({'date': d, 'equity': self.equity(dc, d), 'regime': regime})
            day += 1
        return pd.DataFrame(self.history)


# ============================================================
# BEHAVIORAL PAIN METRICS
# ============================================================
def pain_metrics(eq_df, trades, years):
    if eq_df is None or len(eq_df) < 2: return {}
    s, e = eq_df['equity'].iloc[0], eq_df['equity'].iloc[-1]
    total = (e/s - 1) * 100
    cagr = ((e/s) ** (1/years) - 1) * 100 if years > 0 else total

    # Drawdown
    eq = eq_df.copy()
    eq['pk'] = eq['equity'].cummax()
    eq['dd'] = (eq['equity'] - eq['pk']) / eq['pk'] * 100
    max_dd = eq['dd'].min()

    # Drawdown Duration (longest time below peak)
    eq['underwater'] = eq['equity'] < eq['pk']
    uw_groups = (eq['underwater'] != eq['underwater'].shift()).cumsum()
    uw_periods = eq[eq['underwater']].groupby(uw_groups).size()
    max_dd_duration = int(uw_periods.max()) if len(uw_periods) > 0 else 0

    # Monthly returns
    eq['month'] = eq['date'].dt.to_period('M')
    mo = eq.groupby('month')['equity'].last()
    mr = mo.pct_change().dropna()
    sharpe = (mr.mean() / mr.std()) * np.sqrt(12) if len(mr) > 2 and mr.std() > 0 else 0
    neg = mr[mr < 0]
    sortino = (mr.mean() / neg.std()) * np.sqrt(12) if len(neg) > 0 and neg.std() > 0 else 0

    # Longest Negative/Flat Period (rolling 12M return < 5%)
    eq['y'] = eq['date'].dt.year
    yr_eq = eq.groupby('y')['equity'].last()
    yr_ret = yr_eq.pct_change() * 100
    flat_streak = 0; max_flat = 0
    for r in yr_ret.dropna():
        if r < 5: flat_streak += 1
        else: flat_streak = 0
        max_flat = max(max_flat, flat_streak)

    # Best / Worst Calendar Year
    best_yr = yr_ret.max() if len(yr_ret.dropna()) > 0 else 0
    worst_yr = yr_ret.min() if len(yr_ret.dropna()) > 0 else 0

    # Trade Metrics
    sells = [t for t in trades if 'PnL' in t]
    wins = [t for t in sells if t['PnL'] > 0]
    losses = [t for t in sells if t['PnL'] <= 0]
    wr = len(wins)/len(sells)*100 if sells else 0
    aw = np.mean([t['PnL'] for t in wins]) if wins else 0
    al = np.mean([t['PnL'] for t in losses]) if losses else 0
    mw = max([t['PnL'] for t in sells]) if sells else 0
    ml = min([t['PnL'] for t in sells]) if sells else 0
    exp = (wr/100)*aw - (1-wr/100)*abs(al) if sells else 0
    whip = sum(1 for t in sells if t['Days'] < 10) / len(sells) * 100 if sells else 0
    avg_hold = np.mean([t['Days'] for t in sells]) if sells else 0

    # Max consecutive losers
    max_consec_loss = 0; curr = 0
    for t in sells:
        if t['PnL'] <= 0: curr += 1; max_consec_loss = max(max_consec_loss, curr)
        else: curr = 0

    # Euphoria: % of months with > 5% return
    euphoria = (mr > 0.05).mean() * 100 if len(mr) > 0 else 0

    return {
        'CAGR%': round(cagr, 2), 'Total%': round(total, 1),
        'MaxDD%': round(max_dd, 1), 'DD_Days': max_dd_duration,
        'Sharpe': round(sharpe, 2), 'Sortino': round(sortino, 2),
        'Trades': len(sells), 'WinRate%': round(wr, 1),
        'AvgWin%': round(aw, 1), 'AvgLoss%': round(al, 1),
        'MaxWin%': round(mw, 1), 'MaxLoss%': round(ml, 1),
        'Expect%': round(exp, 1), 'Whipsaw%': round(whip, 1),
        'AvgHold': round(avg_hold), 'MaxConsecLoss': max_consec_loss,
        'BestYear%': round(best_yr, 1) if not pd.isna(best_yr) else 0,
        'WorstYear%': round(worst_yr, 1) if not pd.isna(worst_yr) else 0,
        'FlatYears': max_flat,
        'Euphoria%': round(euphoria, 1),
    }


# ============================================================
# DATA
# ============================================================
def fetch_data():
    start = (datetime.now() - timedelta(days=365*MAX_YEARS + 500)).strftime('%Y-%m-%d')
    print("[1/3] Fetching Nifty 50...")
    nifty = yf.Ticker("^NSEI").history(start=start)
    if nifty.empty:
        print("ERROR: No Nifty data!"); return None, {}
    nifty.index = nifty.index.tz_localize(None)

    print(f"[2/3] Downloading {len(TICKERS[:500])} stocks (15Y)...")
    t0 = time.time()
    try:
        bulk = yf.download(TICKERS[:500], start=start, group_by='ticker',
                           threads=True, progress=True, auto_adjust=True)
    except Exception as e:
        print(f"Threaded download failed: {e}. Retrying sequential...")
        bulk = yf.download(TICKERS[:500], start=start, group_by='ticker',
                           threads=False, progress=True, auto_adjust=True)

    dc = {'NIFTY': nifty}
    loaded = 0
    for t in TICKERS[:500]:
        try:
            if t in bulk.columns.get_level_values(0):
                df = bulk[t].dropna(how='all')
                if len(df) > 200:
                    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                    dc[t] = df; loaded += 1
        except: pass

    print(f"[3/3] Loaded {loaded} stocks in {time.time()-t0:.0f}s. Nifty: {len(nifty)} days.")
    return nifty, dc


# ============================================================
# MAIN
# ============================================================
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nifty, dc = fetch_data()
    if nifty is None or nifty.empty: return

    now = datetime.now()
    strat_names = list(STRATEGIES.keys())

    all_summary = []
    longest_eq = {}
    longest_trades = {}

    print("\n" + "=" * 130)
    print("THE ULTIMATE DNA3 SHOWDOWN -- 15-YEAR COMPARISON")
    print("=" * 130)

    for hname, years in HORIZONS.items():
        s_dt = now - timedelta(days=int(365.25 * years))
        si = nifty.index.searchsorted(s_dt)
        if si >= len(nifty) - 10:
            print(f"\n  [{hname}] Not enough data, skipping."); continue

        actual_start = nifty.index[si]
        actual_end = nifty.index[-1]
        actual_years = (actual_end - actual_start).days / 365.25

        # Nifty benchmark
        ns = nifty.iloc[si]['Close']; ne = nifty.iloc[-1]['Close']
        n_cagr = ((ne/ns)**(1/actual_years) - 1)*100
        nifty_eq = nifty.iloc[si:].copy()
        nifty_eq['equity'] = nifty_eq['Close'] / ns * INITIAL_CAPITAL
        nifty_eq['date'] = nifty_eq.index; nifty_eq['regime'] = 'N/A'
        nifty_eq_df = nifty_eq[['date', 'equity', 'regime']]

        # Nifty DD
        neq = nifty_eq_df.copy()
        neq['pk'] = neq['equity'].cummax()
        neq['dd'] = (neq['equity'] - neq['pk']) / neq['pk'] * 100
        n_dd = neq['dd'].min()

        row = {'Horizon': hname, 'Years': round(actual_years, 1),
               'Nifty_CAGR%': round(n_cagr, 2), 'Nifty_MaxDD%': round(n_dd, 1)}

        print(f"\n{'_' * 130}")
        print(f"  {hname.upper()} ({actual_start.date()} -> {actual_end.date()}, {actual_years:.1f}y)")
        print(f"{'_' * 130}")

        # Header
        h = f"  {'Metric':<16}"
        for sn in strat_names: h += f" {sn:>12}"
        h += f" {'Nifty':>10}"
        print(h); print(f"  {'-' * 110}")

        strat_m = {}
        for sn in strat_names:
            eng = Engine(sn, STRATEGIES[sn])
            eq = eng.run(dc, nifty, actual_start, actual_end)
            m = pain_metrics(eq, eng.trades, actual_years) if eq is not None else {}
            strat_m[sn] = m

            for k, v in m.items(): row[f'{sn}_{k}'] = v

            if hname == max(HORIZONS.keys(), key=lambda k: HORIZONS[k]):
                longest_eq[sn] = eq
                longest_trades[sn] = eng.trades

        # Print key metrics
        for metric in ['CAGR%', 'MaxDD%', 'DD_Days', 'Sharpe', 'Sortino',
                        'Trades', 'WinRate%', 'Expect%', 'Whipsaw%',
                        'MaxWin%', 'MaxLoss%', 'MaxConsecLoss',
                        'AvgHold', 'BestYear%', 'WorstYear%', 'FlatYears', 'Euphoria%']:
            line = f"  {metric:<16}"
            for sn in strat_names:
                v = strat_m[sn].get(metric, '-')
                line += f" {v:>12}" if isinstance(v, (int, float)) else f" {'-':>12}"
            nv = row.get(f'Nifty_{metric}', '-')
            line += f" {nv:>10}" if isinstance(nv, (int, float)) else f" {'-':>10}"
            print(line)

        # Winner
        best_cagr = -999; best = ''
        for sn in strat_names:
            c = strat_m[sn].get('CAGR%', -999)
            if isinstance(c, (int, float)) and c > best_cagr: best_cagr = c; best = sn
        print(f"\n  >>> WINNER: {best} ({best_cagr:+.1f}% CAGR)")

        all_summary.append(row)

    pd.DataFrame(all_summary).to_csv(f"{OUTPUT_DIR}/ultimate_summary.csv", index=False)

    # ================================
    # CALENDAR YEAR RETURNS
    # ================================
    print(f"\n\n{'=' * 130}")
    print("CALENDAR YEAR RETURNS (Longest Horizon)")
    print("=" * 130)

    yr_data = []
    h = f"  {'Year':<6}"
    for sn in strat_names: h += f" {sn:>12}"
    h += f" {'Nifty':>10}"
    print(h); print(f"  {'-' * 75}")

    # Build yearly returns for each strategy
    yr_rets = {}
    for sn in longest_eq:
        eq = longest_eq[sn]
        if eq is None or len(eq) < 22: continue
        eq['y'] = eq['date'].dt.year
        yg = eq.groupby('y')['equity'].last()
        yr_rets[sn] = yg.pct_change() * 100

    # Nifty yearly
    si = nifty.index.searchsorted(now - timedelta(days=int(365.25 * MAX_YEARS)))
    ne = nifty.iloc[si:].copy()
    ne['y'] = ne.index.year
    ng = ne.groupby('y')['Close'].last()
    nr = ng.pct_change() * 100
    yr_rets['Nifty'] = nr

    all_years = sorted(set().union(*[r.index for r in yr_rets.values() if r is not None]))
    for y in all_years:
        line = f"  {y:<6}"
        yr_row = {'Year': y}
        for sn in strat_names:
            v = yr_rets.get(sn, pd.Series())
            val = v.get(y, None)
            if val is not None and not pd.isna(val):
                line += f" {val:>+11.1f}%"
                yr_row[sn] = round(val, 1)
            else:
                line += f" {'---':>12}"
        nv = yr_rets.get('Nifty', pd.Series()).get(y, None)
        if nv is not None and not pd.isna(nv):
            line += f" {nv:>+9.1f}%"
            yr_row['Nifty'] = round(nv, 1)
        else:
            line += f" {'---':>10}"
        print(line)
        yr_data.append(yr_row)

    pd.DataFrame(yr_data).to_csv(f"{OUTPUT_DIR}/ultimate_yearly.csv", index=False)

    # ================================
    # FINAL VERDICT
    # ================================
    print(f"\n\n{'=' * 130}")
    print("FINAL VERDICT")
    print("=" * 130)

    for row in all_summary:
        h = row['Horizon']
        cagrs = {}
        for sn in strat_names:
            c = row.get(f'{sn}_CAGR%', -999)
            if isinstance(c, (int, float)): cagrs[sn] = c
        best = max(cagrs, key=cagrs.get) if cagrs else 'N/A'
        nc = row.get('Nifty_CAGR%', 0)
        alpha = cagrs.get(best, 0) - nc
        line = f"  {h:<6}: Winner = {best:<12} ({cagrs.get(best, 0):>+6.1f}% CAGR, Alpha: {alpha:>+5.1f}%)"

        # Show all CAGRs
        parts = []
        for sn in strat_names:
            if sn in cagrs: parts.append(f"{sn}={cagrs[sn]:+.1f}%")
        line += f"  [{', '.join(parts)}]"
        print(line)

    # Pain Scorecard (longest horizon)
    if all_summary:
        lr = all_summary[-1]
        print(f"\n  PAIN SCORECARD ({lr['Horizon']} horizon):")
        for metric, label in [
            ('MaxDD%', 'Best MaxDD (least pain)'),
            ('DD_Days', 'Shortest DD Duration'),
            ('MaxConsecLoss', 'Fewest Consec Losses'),
            ('Expect%', 'Highest Expectancy'),
            ('Whipsaw%', 'Fewest Whipsaws'),
            ('FlatYears', 'Fewest Flat Years'),
        ]:
            vals = {}
            for sn in strat_names:
                v = lr.get(f'{sn}_{metric}')
                if isinstance(v, (int, float)): vals[sn] = v
            if vals:
                if metric in ['MaxDD%', 'DD_Days', 'MaxConsecLoss', 'Whipsaw%', 'FlatYears']:
                    winner = max(vals, key=vals.get)  # Less negative = better for DD
                    if metric == 'MaxDD%': winner = max(vals, key=vals.get)  # -28 > -34
                    else: winner = min(vals, key=vals.get)
                else:
                    winner = max(vals, key=vals.get)
                print(f"    {label:<30}: {winner} ({vals[winner]})")

    print(f"\n  Files saved to {OUTPUT_DIR}/ultimate_*.csv")
    print("=" * 130)


if __name__ == "__main__":
    print("=" * 130)
    print("DNA3 ULTIMATE 15-YEAR COMPARISON")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 130)
    run()
