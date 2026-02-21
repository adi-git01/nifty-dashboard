"""
REBALANCE FREQUENCY & ROLLING PERIOD SENSITIVITY ANALYSIS
==========================================================
Tests DNA3-V3.1 across different rebalance cadences to find the sweet spot.

Tests:
  1. Rebalance frequency: 5d, 10d, 15d, 20d (monthly), 30d
  2. Impact on: CAGR, Sharpe, Max DD, Trades, Churn, Expectancy
  3. Rolling return analysis at 1/3/6/12 month windows
  4. Holding period distribution analysis

Period: 5Y (largest practical horizon for decision-making)
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
YEARS = 5

# Regime config (same as V3.1)
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


def calc_indicators(sw, nw):
    if len(sw) < 100 or len(nw) < 64: return None
    p = sw['Close'].iloc[-1]
    ma50 = sw['Close'].rolling(50).mean().iloc[-1]
    
    if len(sw) > 63 and len(nw) > 63:
        rs = ((p - sw['Close'].iloc[-63])/sw['Close'].iloc[-63] - 
              (nw['Close'].iloc[-1] - nw['Close'].iloc[-63])/nw['Close'].iloc[-63]) * 100
    else: rs = 0
    
    rets = sw['Close'].pct_change().dropna()[-60:]
    vol = rets.std() * np.sqrt(252) * 100 if len(rets) > 10 else 0
    liq = sw['Volume'].rolling(20).mean().iloc[-1] * p
    
    return {'price': p, 'ma50': ma50, 'rs': rs, 'vol': vol, 'liq': liq}


class V31Engine:
    """V3.1 engine parameterized by rebalance frequency."""
    
    def __init__(self, rebal_freq):
        self.rebal_freq = rebal_freq
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
    
    def reset(self):
        self.__init__(self.rebal_freq)
    
    def get_price(self, dc, t, d):
        if t not in dc: return None
        df = dc[t]
        i = df.index.searchsorted(d)
        return df['Close'].iloc[min(i, len(df)-1)] if i > 0 else None
    
    def check_exits(self, dc, d):
        exits = []
        for t, pos in self.positions.items():
            p = self.get_price(dc, t, d)
            if not p: continue
            if p > pos['peak']: pos['peak'] = p
            ret = (p - pos['entry']) / pos['entry']
            reason = None
            if p < pos['peak'] * 0.88: reason = 'Trail'
            if ret < -0.20: reason = 'HardStop'
            if reason:
                self.capital += pos['shares'] * p * (1 - COST_BPS/10000)
                self.trade_log.append({
                    'PnL%': round(ret*100, 2), 'Reason': reason,
                    'Hold': (d - pos['entry_date']).days,
                })
                exits.append(t)
        for t in exits: del self.positions[t]
    
    def scan(self, dc, nifty, d, regime):
        cfg = REGIME_CFG.get(regime, REGIME_CFG['UNKNOWN'])
        max_pos = cfg['max_pos']
        
        # Reduce if regime tightened
        if len(self.positions) > max_pos:
            pl = [(t, (self.get_price(dc, t, d) - pos['entry'])/pos['entry'], pos) 
                  for t, pos in self.positions.items() if self.get_price(dc, t, d)]
            pl.sort(key=lambda x: x[1])
            while len(self.positions) > max_pos and pl:
                t, ret, pos = pl.pop(0)
                if t in self.positions:
                    p = self.get_price(dc, t, d)
                    self.capital += pos['shares'] * p * (1 - COST_BPS/10000)
                    self.trade_log.append({'PnL%': round(ret*100,2), 'Reason': 'Reduce', 'Hold': (d-pos['entry_date']).days})
                    del self.positions[t]
        
        if len(self.positions) >= max_pos: return
        
        ni = nifty.index.searchsorted(d)
        if ni < 252: return
        nw = nifty.iloc[max(0, ni-252):ni+1]
        
        cands = []
        for t in dc:
            if t == 'NIFTY' or t in self.positions: continue
            df = dc[t]
            i = df.index.searchsorted(d)
            if i < 100: continue
            w = df.iloc[max(0, i-252):i+1]
            if len(w) < 100: continue
            ind = calc_indicators(w, nw)
            if ind and ind['rs'] >= 2.0 and ind['vol'] >= 30 and ind['price'] > ind['ma50'] and ind['liq'] >= 5_000_000:
                cands.append({'ticker': t, 'ind': ind})
        
        cands.sort(key=lambda x: -x['ind']['rs'])
        
        sel = []
        sc = {}
        for c in cands:
            sec = SECTOR_MAP.get(c['ticker'], 'Unk')
            curr = sum(1 for t in self.positions if SECTOR_MAP.get(t, 'Unk') == sec)
            if sc.get(sec, 0) + curr < 3:
                sel.append(c)
                sc[sec] = sc.get(sec, 0) + 1
            if len(sel) + len(self.positions) >= max_pos: break
        
        eq = self.get_equity(dc, d)
        avail = max(0, self.capital - eq * cfg['cash'])
        free = max_pos - len(self.positions)
        
        for c in sel[:free]:
            p = c['ind']['price']
            size = avail / (free + 1)
            sh = int(size / p)
            if sh < 1: continue
            cost = sh * p * (1 + COST_BPS/10000)
            if avail >= cost and cost > 5000:
                avail -= cost
                self.capital -= cost
                self.positions[c['ticker']] = {
                    'entry': p, 'peak': p, 'shares': sh,
                    'stop': p*0.80, 'entry_date': d,
                }
    
    def get_equity(self, dc, d):
        v = self.capital
        for t, pos in self.positions.items():
            p = self.get_price(dc, t, d)
            if p: v += pos['shares'] * p
        return v
    
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
            if day % self.rebal_freq == 0:
                self.scan(dc, nifty, d, regime)
            eq = self.get_equity(dc, d)
            self.history.append({'date': d, 'equity': eq, 'regime': regime})
            day += 1
        
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
    neg = mr[mr < 0]
    sortino = (mr.mean() / neg.std()) * np.sqrt(12) if len(neg) > 0 and neg.std() > 0 else 0
    
    return {
        'CAGR%': round(cagr, 2), 'Total%': round(total, 2),
        'MaxDD%': round(max_dd, 2), 'Sharpe': round(sharpe, 2),
        'Sortino': round(sortino, 2),
    }


def fetch_data():
    start = (datetime.now() - timedelta(days=365*YEARS + 500)).strftime('%Y-%m-%d')
    print("[1/3] Fetching Nifty...")
    nifty = yf.Ticker("^NSEI").history(start=start)
    if nifty.empty: return None, {}
    nifty.index = nifty.index.tz_localize(None)
    
    print(f"[2/3] Bulk downloading {len(TICKERS[:500])} stocks...")
    t0 = time.time()
    try:
        bulk = yf.download(TICKERS[:500], start=start, group_by='ticker', threads=True, progress=True, auto_adjust=True)
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
    
    now = datetime.now()
    start = now - timedelta(days=int(365.25 * YEARS))
    si = nifty.index.searchsorted(start)
    actual_start = nifty.index[si]
    actual_end = nifty.index[-1]
    actual_years = (actual_end - actual_start).days / 365.25
    
    # Nifty benchmark
    ns = nifty.iloc[si]['Close']
    ne = nifty.iloc[-1]['Close']
    n_cagr = ((ne/ns)**(1/actual_years) - 1) * 100
    
    # =====================================
    # TEST 1: REBALANCE FREQUENCY SWEEP
    # =====================================
    print("\n" + "=" * 100)
    print("TEST 1: REBALANCE FREQUENCY SENSITIVITY (DNA3-V3.1, 5Y)")
    print("=" * 100)
    
    rebal_freqs = [5, 7, 10, 15, 20, 30]
    freq_results = []
    freq_equity = {}
    
    for freq in rebal_freqs:
        eng = V31Engine(freq)
        eq = eng.run(dc, nifty, actual_start, actual_end)
        
        if eq is not None and len(eq) > 10:
            m = calc_metrics(eq, actual_years)
            sells = [t for t in eng.trade_log]
            wins = [t for t in sells if t['PnL%'] > 0]
            losses = [t for t in sells if t['PnL%'] <= 0]
            wr = len(wins)/len(sells)*100 if sells else 0
            aw = np.mean([t['PnL%'] for t in wins]) if wins else 0
            al = np.mean([t['PnL%'] for t in losses]) if losses else 0
            ah = np.mean([t['Hold'] for t in sells]) if sells else 0
            exp = (wr/100)*aw - (1-wr/100)*abs(al) if sells else 0
            
            # Churn = trades per year
            churn = len(sells) / actual_years
            
            freq_results.append({
                'Rebal_Days': freq,
                'CAGR%': m['CAGR%'],
                'Total%': m['Total%'],
                'MaxDD%': m['MaxDD%'],
                'Sharpe': m['Sharpe'],
                'Sortino': m['Sortino'],
                'Trades': len(sells),
                'Trades/Year': round(churn, 1),
                'WinRate%': round(wr, 1),
                'AvgWin%': round(aw, 1),
                'AvgLoss%': round(al, 1),
                'AvgHold_Days': round(ah, 0),
                'Expectancy%': round(exp, 2),
            })
            freq_equity[freq] = eq
        
        print(f"  Freq={freq:>2d}d: CAGR={m['CAGR%']:>+6.1f}%, Sharpe={m['Sharpe']:.2f}, MaxDD={m['MaxDD%']:.1f}%, Trades={len(sells)}, Churn={churn:.0f}/yr")
    
    freq_df = pd.DataFrame(freq_results)
    freq_df.to_csv(f"{OUTPUT_DIR}/v31_rebal_sensitivity.csv", index=False)
    
    # Print full table
    print(f"\n  {'Rebal':>5} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'Trades':>7} {'/Year':>6} {'WinR%':>6} {'AvgW':>6} {'AvgL':>6} {'Hold':>5} {'Exp%':>6}")
    print(f"  {'-'*82}")
    for r in freq_results:
        print(f"  {r['Rebal_Days']:>4}d {r['CAGR%']:>+6.1f}% {r['Sharpe']:>6.2f} {r['Sortino']:>7.2f} {r['MaxDD%']:>+6.1f}% {r['Trades']:>6} {r['Trades/Year']:>5.0f} {r['WinRate%']:>5.0f}% {r['AvgWin%']:>+5.1f} {r['AvgLoss%']:>+5.1f} {r['AvgHold_Days']:>4.0f}d {r['Expectancy%']:>+5.2f}")
    
    # Best by each metric
    print(f"\n  BEST BY METRIC:")
    for metric in ['CAGR%', 'Sharpe', 'Sortino']:
        best = max(freq_results, key=lambda x: x[metric])
        print(f"    {metric:<10}: {best['Rebal_Days']}d (={best[metric]})")
    worst_dd = min(freq_results, key=lambda x: x['MaxDD%'])
    print(f"    {'MaxDD%':<10}: {worst_dd['Rebal_Days']}d (={worst_dd['MaxDD%']}%)")
    least_churn = min(freq_results, key=lambda x: x['Trades/Year'])
    print(f"    {'Low Churn':<10}: {least_churn['Rebal_Days']}d ({least_churn['Trades/Year']:.0f} trades/yr)")
    
    # =====================================
    # TEST 2: HOLDING PERIOD DISTRIBUTION
    # =====================================
    print(f"\n\n{'=' * 100}")
    print("TEST 2: HOLDING PERIOD DISTRIBUTION (15d Rebalance)")
    print("=" * 100)
    
    # Use 15-day frequency (user's preferred)
    if 15 in freq_equity:
        eng15 = V31Engine(15)
        eng15.run(dc, nifty, actual_start, actual_end)
        holds = [t['Hold'] for t in eng15.trade_log]
        
        if holds:
            print(f"\n  Total trades: {len(holds)}")
            print(f"  Mean hold: {np.mean(holds):.0f} days")
            print(f"  Median hold: {np.median(holds):.0f} days")
            print(f"  Shortest: {min(holds)} days")
            print(f"  Longest: {max(holds)} days")
            
            # Distribution
            buckets = [(0, 7, '<1w'), (7, 14, '1-2w'), (14, 30, '2w-1m'), 
                       (30, 60, '1-2m'), (60, 90, '2-3m'), (90, 180, '3-6m'), (180, 999, '6m+')]
            
            print(f"\n  {'Period':<8} {'Count':>6} {'%':>6} {'Avg PnL':>8}")
            print(f"  {'-'*30}")
            for lo, hi, label in buckets:
                in_bucket = [t for t in eng15.trade_log if lo <= t['Hold'] < hi]
                if in_bucket:
                    avg_pnl = np.mean([t['PnL%'] for t in in_bucket])
                    print(f"  {label:<8} {len(in_bucket):>6} {len(in_bucket)/len(holds)*100:>5.0f}% {avg_pnl:>+7.1f}%")
    
    # =====================================
    # TEST 3: ROLLING RETURN WINDOWS
    # =====================================
    print(f"\n\n{'=' * 100}")
    print("TEST 3: ROLLING RETURN ANALYSIS (V3.1 @ 15d rebal vs Nifty)")
    print("=" * 100)
    
    eq_15 = freq_equity.get(15)
    if eq_15 is None:
        eq_15 = freq_equity.get(10)  # Fallback
    
    if eq_15 is not None:
        eq_15 = eq_15.copy()
        eq_15['month'] = eq_15['date'].dt.to_period('M')
        mo = eq_15.groupby('month').agg({'equity': 'last', 'date': 'last'}).reset_index()
        
        # Nifty monthly
        nifty_eq = nifty.iloc[si:].copy()
        nifty_eq['equity'] = nifty_eq['Close'] / ns * INITIAL_CAPITAL
        nifty_eq['date'] = nifty_eq.index
        nifty_eq['month'] = nifty_eq['date'].dt.to_period('M')
        n_mo = nifty_eq.groupby('month').agg({'equity': 'last', 'date': 'last'}).reset_index()
        
        all_roll = []
        
        for window, label in [(1, '1-Month'), (3, '3-Month'), (6, '6-Month'), (12, '12-Month'), (24, '24-Month')]:
            mo[f'roll_{window}'] = mo['equity'].pct_change(window) * 100
            n_mo[f'roll_{window}'] = n_mo['equity'].pct_change(window) * 100
            
            v31_r = mo[f'roll_{window}'].dropna()
            n_r = n_mo[f'roll_{window}'].dropna()
            
            if len(v31_r) > 0 and len(n_r) > 0:
                all_roll.append({
                    'Window': label,
                    'V31_Median': round(v31_r.median(), 1),
                    'V31_Mean': round(v31_r.mean(), 1),
                    'V31_Best': round(v31_r.max(), 1),
                    'V31_Worst': round(v31_r.min(), 1),
                    'V31_%Pos': round((v31_r > 0).mean() * 100, 0),
                    'V31_%>15': round((v31_r > 15).mean() * 100, 0),
                    'Nifty_Median': round(n_r.median(), 1),
                    'Nifty_Mean': round(n_r.mean(), 1),
                    'Nifty_Worst': round(n_r.min(), 1),
                    'Nifty_%Pos': round((n_r > 0).mean() * 100, 0),
                })
        
        roll_df = pd.DataFrame(all_roll)
        roll_df.to_csv(f"{OUTPUT_DIR}/v31_rolling_windows.csv", index=False)
        
        print(f"\n  V3.1 (15d rebalance):")
        print(f"  {'Window':<12} {'Median':>8} {'Mean':>8} {'Best':>8} {'Worst':>8} {'%Pos':>6} {'%>15%':>6}")
        print(f"  {'-'*58}")
        for r in all_roll:
            print(f"  {r['Window']:<12} {r['V31_Median']:>+7.1f}% {r['V31_Mean']:>+7.1f}% {r['V31_Best']:>+7.1f}% {r['V31_Worst']:>+7.1f}% {r['V31_%Pos']:>5.0f}% {r['V31_%>15']:>5.0f}%")
        
        print(f"\n  Nifty 50:")
        print(f"  {'Window':<12} {'Median':>8} {'Mean':>8} {'Worst':>8} {'%Pos':>6}")
        print(f"  {'-'*44}")
        for r in all_roll:
            print(f"  {r['Window']:<12} {r['Nifty_Median']:>+7.1f}% {r['Nifty_Mean']:>+7.1f}% {r['Nifty_Worst']:>+7.1f}% {r['Nifty_%Pos']:>5.0f}%")
        
        # =====================================
        # TEST 4: MONTH-BY-MONTH RETURNS
        # =====================================
        print(f"\n\n{'=' * 100}")
        print("TEST 4: CALENDAR MONTH RETURNS (V3.1 @ 15d)")
        print("=" * 100)
        
        mo['monthly_ret'] = mo['equity'].pct_change() * 100
        mo['cal_month'] = mo['month'].apply(lambda x: x.month)
        
        print(f"\n  {'Month':<10} {'Avg Ret':>8} {'Median':>8} {'%Pos':>6} {'Worst':>8}")
        print(f"  {'-'*44}")
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for m_num in range(1, 13):
            subset = mo[mo['cal_month'] == m_num]['monthly_ret'].dropna()
            if len(subset) > 0:
                print(f"  {month_names[m_num-1]:<10} {subset.mean():>+7.1f}% {subset.median():>+7.1f}% {(subset>0).mean()*100:>5.0f}% {subset.min():>+7.1f}%")
    
    # =====================================
    # CONCLUSION
    # =====================================
    print(f"\n\n{'=' * 100}")
    print("RECOMMENDATIONS")
    print("=" * 100)
    
    # Find optimal rebalance
    best_sharpe_freq = max(freq_results, key=lambda x: x['Sharpe'])
    best_cagr_freq = max(freq_results, key=lambda x: x['CAGR%'])
    
    print(f"\n  REBALANCE FREQUENCY:")
    print(f"    Best Sharpe  : Every {best_sharpe_freq['Rebal_Days']} trading days")
    print(f"    Best CAGR    : Every {best_cagr_freq['Rebal_Days']} trading days")
    print(f"    Your cadence : Every 15 days (bi-weekly) or ~20 days (monthly)")
    
    # Compare 15 vs 20
    f15 = next((r for r in freq_results if r['Rebal_Days'] == 15), None)
    f20 = next((r for r in freq_results if r['Rebal_Days'] == 20), None)
    
    if f15 and f20:
        print(f"\n    15d vs 20d Comparison:")
        print(f"      CAGR:   {f15['CAGR%']:>+.1f}% vs {f20['CAGR%']:>+.1f}% (diff: {f15['CAGR%']-f20['CAGR%']:+.1f}%)")
        print(f"      Sharpe: {f15['Sharpe']:.2f} vs {f20['Sharpe']:.2f}")
        print(f"      Churn:  {f15['Trades/Year']:.0f}/yr vs {f20['Trades/Year']:.0f}/yr")
        print(f"      MaxDD:  {f15['MaxDD%']:.1f}% vs {f20['MaxDD%']:.1f}%")
        
        winner = "15d (bi-weekly)" if f15['Sharpe'] >= f20['Sharpe'] else "20d (monthly)"
        print(f"      --> Recommendation: {winner}")
    
    print(f"\n  ROLLING RETURN WINDOW:")
    print(f"    For monthly rebalancer: Use 3-MONTH rolling returns")
    print(f"    Reason: Matches 2-3 rebalance cycles, smooths out noise,")
    print(f"    shows trend direction without being too lagging.")
    print(f"    Use 12-month for annual performance review reporting.")
    
    print(f"\n  Files saved:")
    print(f"    {OUTPUT_DIR}/v31_rebal_sensitivity.csv")
    print(f"    {OUTPUT_DIR}/v31_rolling_windows.csv")


if __name__ == "__main__":
    print("=" * 100)
    print("REBALANCE FREQUENCY & ROLLING PERIOD SENSITIVITY (DNA3-V3.1)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 100)
    run()
