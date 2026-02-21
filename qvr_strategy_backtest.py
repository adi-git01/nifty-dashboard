"""
QUALITY VALUE RECOVERY (QVR) STRATEGY BACKTEST
================================================
User's Proposed Strategy:
  Quality  >= 6.0 (high-quality business)
  Value    >= 6.0 (attractively valued)
  Trend    40-65  (recovering, not yet in full uptrend)

Since fundamental data (ROE, PE etc) is not available historically, we replicate
each score using price-derived proxies that capture the same essence:

QUALITY PROXY (0-10):
  - Low volatility (stable business = quality)
  - Consistent positive returns over 12mo (earnings = returns over time)
  - Low max drawdown (quality companies don't crash as hard)
  - Above 200 DMA most of the time (institutional ownership)

VALUE PROXY (0-10):
  - Distance from 52-week high (beaten down = cheap)
  - Price well below 200 DMA (undervalued vs intrinsic trend)
  - Low relative strength vs Nifty (out of favor = value)

TREND SCORE (0-100) — Exact replica of dashboard's calculate_trend_metrics:
  - MA50/MA200 alignment
  - 52-week position
  - Near high/low bonus/penalty
  Filter: 40-65 (NEUTRAL = recovering from bottom, not yet uptrend)

Comparison vs DNA3-V3.1: 5Y backtest, same framework.
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
REBALANCE_FREQ = 15

# Regime config (shared with V3.1)
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
# SCORE CALCULATIONS (Matching dashboard logic)
# ============================================================
def calc_trend_score(window):
    """Exact replica of dashboard's calculate_trend_metrics."""
    if len(window) < 252:
        return 50
    
    price = window['Close'].iloc[-1]
    ma50 = window['Close'].rolling(50).mean().iloc[-1]
    ma200 = window['Close'].rolling(200).mean().iloc[-1]
    high_52 = window['High'].rolling(252, min_periods=50).max().iloc[-1]
    low_52 = window['Low'].rolling(252, min_periods=50).min().iloc[-1]
    
    score = 50  # Start neutral
    
    # MA Position (40 pts max)
    if ma50 > 0:
        score += 15 if price > ma50 else -10
    if ma200 > 0:
        score += 15 if price > ma200 else -15
    if ma50 > 0 and ma200 > 0:
        score += 10 if ma50 > ma200 else -5
    
    # 52-Week Position (30 pts max)
    if high_52 > low_52:
        range_52 = high_52 - low_52
        position = (price - low_52) / range_52
        score += int((position - 0.5) * 30)
        
        dist_from_high = (price - high_52) / high_52 * 100
        if dist_from_high > -5:
            score += 10
        elif dist_from_high < -30:
            score -= 10
    
    return max(0, min(100, score))


def calc_quality_proxy(window, nifty_window):
    """
    Quality proxy from price data:
    - Low volatility (stable business)
    - Consistent returns (proxy for earnings quality)
    - Low drawdown (institutional support)
    - Above 200 DMA consistency
    """
    if len(window) < 252:
        return 5.0
    
    # 1. Volatility (lower = higher quality) — 30% weight
    rets = window['Close'].pct_change().dropna()[-252:]
    vol = rets.std() * np.sqrt(252) * 100
    # Quality companies: vol < 25% = 10, vol > 60% = 0
    s_vol = max(0, min(10, 10 - (vol - 20) / 5))
    
    # 2. Return consistency (% of months positive) — 25% weight
    monthly = window['Close'].resample('ME').last().pct_change().dropna()[-12:]
    if len(monthly) > 3:
        pct_pos = (monthly > 0).mean()
        s_consist = pct_pos * 10
    else:
        s_consist = 5.0
    
    # 3. Max drawdown (lower = higher quality) — 25% weight
    peak = window['Close'].cummax()
    dd = ((window['Close'] - peak) / peak).min() * 100
    # Quality: dd > -15% = 10, dd < -50% = 0
    s_dd = max(0, min(10, 10 + (dd + 15) / 3.5))
    
    # 4. Days above 200 DMA (institutional support) — 20% weight
    ma200 = window['Close'].rolling(200).mean()
    above_200 = (window['Close'] > ma200).iloc[-252:].mean()
    s_above = above_200 * 10
    
    quality = s_vol * 0.30 + s_consist * 0.25 + s_dd * 0.25 + s_above * 0.20
    return round(max(0, min(10, quality)), 1)


def calc_value_proxy(window, nifty_window):
    """
    Value proxy from price data:
    - Distance from 52-week high (beaten down = cheap)
    - Below 200 DMA (undervalued vs intrinsic)
    - Low RS vs Nifty (out of favor = contrarian value)
    - Low recent momentum (nobody wants it = value)
    """
    if len(window) < 252:
        return 5.0
    
    price = window['Close'].iloc[-1]
    high_52 = window['High'].rolling(252, min_periods=50).max().iloc[-1]
    ma200 = window['Close'].rolling(200).mean().iloc[-1]
    
    # 1. Distance from 52W high (further = more value) — 35% weight
    dist_high = (price - high_52) / high_52 * 100  # Negative number
    # Value: -30% from high = 8, -50% = 10, -5% = 2, at high = 0
    if dist_high > -5:
        s_dist = 1.0  # Near highs = not cheap
    elif dist_high > -15:
        s_dist = 3.0 + (-dist_high - 5) / 10 * 4  # 3-7
    elif dist_high > -30:
        s_dist = 7.0 + (-dist_high - 15) / 15 * 2  # 7-9
    else:
        s_dist = 9.0  # Very beaten down
    
    # 2. Below 200 DMA distance (below = value) — 25% weight
    dist_200 = (price - ma200) / ma200 * 100
    if dist_200 < -15:
        s_200 = 9.0
    elif dist_200 < -5:
        s_200 = 6.0 + (-dist_200 - 5) / 10 * 3
    elif dist_200 < 5:
        s_200 = 4.0 + (5 - dist_200) / 10 * 2
    else:
        s_200 = 2.0  # Above 200 DMA = not cheap
    
    # 3. Low RS vs Nifty (out of favor) — 20% weight
    if len(nifty_window) > 126:
        rs_stock = (price - window['Close'].iloc[-126]) / window['Close'].iloc[-126]
        rs_nifty = (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-126]) / nifty_window['Close'].iloc[-126]
        rs = (rs_stock - rs_nifty) * 100
        # Low RS = value (underperformer = cheap)
        if rs < -20:
            s_rs = 9.0
        elif rs < -5:
            s_rs = 5.0 + (-rs - 5) / 15 * 4
        elif rs < 5:
            s_rs = 4.0
        else:
            s_rs = 2.0  # Outperformer = not cheap
    else:
        s_rs = 5.0
    
    # 4. Low 3-month return (nobody wants it) — 20% weight
    if len(window) > 63:
        ret_3m = (price - window['Close'].iloc[-63]) / window['Close'].iloc[-63] * 100
        if ret_3m < -20:
            s_3m = 9.0
        elif ret_3m < -5:
            s_3m = 5.0 + (-ret_3m - 5) / 15 * 4
        elif ret_3m < 5:
            s_3m = 4.0
        else:
            s_3m = 2.0
    else:
        s_3m = 5.0
    
    value = s_dist * 0.35 + s_200 * 0.25 + s_rs * 0.20 + s_3m * 0.20
    return round(max(0, min(10, value)), 1)


# ============================================================
# STRATEGY ENGINES
# ============================================================
class QVREngine:
    """Quality Value Recovery strategy."""
    
    def __init__(self):
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
    
    def reset(self):
        self.__init__()
    
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
            
            # 12% trailing stop (same as V3.1)
            if p < pos['peak'] * 0.88:
                reason = 'Trail'
            # Hard stop at -20%
            if ret < -0.20:
                reason = 'HardStop'
            # Target exit: if trend score goes above 75 (fully recovered, take profit)
            df = dc.get(t)
            if df is not None:
                idx = df.index.searchsorted(d)
                if idx > 252:
                    w = df.iloc[max(0, idx-252):idx+1]
                    ts = calc_trend_score(w)
                    if ts >= 75:
                        reason = 'FullRecovery'
            
            if reason:
                self.capital += pos['shares'] * p * (1 - COST_BPS/10000)
                self.trade_log.append({
                    'PnL%': round(ret*100, 2), 'Reason': reason,
                    'Hold': (d - pos['entry_date']).days, 'Ticker': t,
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
            df = dc[t]
            idx = df.index.searchsorted(d)
            if idx < 252: continue
            w = df.iloc[max(0, idx-252):idx+1]
            if len(w) < 200: continue
            
            # Calculate all three scores
            trend = calc_trend_score(w)
            quality = calc_quality_proxy(w, nw)
            value = calc_value_proxy(w, nw)
            
            # Liquidity check
            liq = w['Volume'].rolling(20).mean().iloc[-1] * w['Close'].iloc[-1]
            
            # QVR Filter: Quality >= 6, Value >= 6, Trend 40-65
            if quality >= 6.0 and value >= 6.0 and 40 <= trend <= 65 and liq >= 5_000_000:
                # Rank by quality + value composite (higher = better)
                composite = quality * 0.5 + value * 0.5
                cands.append({
                    'ticker': t, 'quality': quality, 'value': value,
                    'trend': trend, 'composite': composite,
                    'price': w['Close'].iloc[-1],
                })
        
        # Rank by composite score (best quality-value combo first)
        cands.sort(key=lambda x: -x['composite'])
        
        # Sector cap = 3
        sel = []
        sc = {}
        for c in cands:
            sec = SECTOR_MAP.get(c['ticker'], 'Unk')
            curr = sum(1 for t in self.positions if SECTOR_MAP.get(t, 'Unk') == sec)
            if sc.get(sec, 0) + curr < 3:
                sel.append(c)
                sc[sec] = sc.get(sec, 0) + 1
            if len(sel) + len(self.positions) >= max_pos: break
        
        # Buy with cash reserve
        eq = self.get_equity(dc, d)
        avail = max(0, self.capital - eq * cfg['cash'])
        free = max_pos - len(self.positions)
        
        for c in sel[:free]:
            p = c['price']
            size = avail / (free + 1)
            sh = int(size / p)
            if sh < 1: continue
            cost = sh * p * (1 + COST_BPS/10000)
            if avail >= cost and cost > 5000:
                avail -= cost
                self.capital -= cost
                self.positions[c['ticker']] = {
                    'entry': p, 'peak': p, 'shares': sh,
                    'entry_date': d,
                    'quality': c['quality'], 'value': c['value'], 'trend': c['trend'],
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
            if day % REBALANCE_FREQ == 0:
                self.scan(dc, nifty, d, regime)
            eq = self.get_equity(dc, d)
            self.history.append({'date': d, 'equity': eq, 'regime': regime})
            day += 1
        
        return pd.DataFrame(self.history)


class V31Engine:
    """DNA3-V3.1 for comparison."""
    
    def __init__(self):
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
    
    def reset(self):
        self.__init__()
    
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
                self.trade_log.append({'PnL%': round(ret*100, 2), 'Reason': reason,
                    'Hold': (d - pos['entry_date']).days, 'Ticker': t})
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
            df = dc[t]
            idx = df.index.searchsorted(d)
            if idx < 100: continue
            w = df.iloc[max(0, idx-252):idx+1]
            if len(w) < 100: continue
            
            price = w['Close'].iloc[-1]
            ma50 = w['Close'].rolling(50).mean().iloc[-1]
            
            if len(w) > 63 and len(nw) > 63:
                rs = ((price - w['Close'].iloc[-63])/w['Close'].iloc[-63] - 
                      (nw['Close'].iloc[-1] - nw['Close'].iloc[-63])/nw['Close'].iloc[-63]) * 100
            else: rs = 0
            
            rets = w['Close'].pct_change().dropna()[-60:]
            vol = rets.std() * np.sqrt(252) * 100 if len(rets) > 10 else 0
            liq = w['Volume'].rolling(20).mean().iloc[-1] * price
            
            if rs >= 2.0 and vol >= 30 and price > ma50 and liq >= 5_000_000:
                cands.append({'ticker': t, 'rs': rs, 'price': price})
        
        cands.sort(key=lambda x: -x['rs'])
        
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
            p = c['price']
            size = avail / (free + 1)
            sh = int(size / p)
            if sh < 1: continue
            cost = sh * p * (1 + COST_BPS/10000)
            if avail >= cost and cost > 5000:
                avail -= cost
                self.capital -= cost
                self.positions[c['ticker']] = {'entry': p, 'peak': p, 'shares': sh, 'entry_date': d}
    
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
            if day % REBALANCE_FREQ == 0:
                self.scan(dc, nifty, d, regime)
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
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0
    return {
        'CAGR%': round(cagr, 2), 'Total%': round(total, 2),
        'MaxDD%': round(max_dd, 2), 'Sharpe': round(sharpe, 2),
        'Sortino': round(sortino, 2), 'Calmar': round(calmar, 2),
    }


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


# ============================================================
# MAIN
# ============================================================
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
    n_total = (ne/ns - 1) * 100
    nifty_eq = nifty.iloc[si:].copy()
    nifty_eq['equity'] = nifty_eq['Close'] / ns * INITIAL_CAPITAL
    nifty_eq['date'] = nifty_eq.index
    nifty_eq['regime'] = 'N/A'
    n_m = calc_metrics(nifty_eq[['date', 'equity', 'regime']], actual_years)
    
    # ============================================
    # RUN BOTH STRATEGIES
    # ============================================
    print(f"\n{'=' * 100}")
    print(f"QVR STRATEGY vs DNA3-V3.1 vs NIFTY ({YEARS}Y BACKTEST)")
    print(f"Period: {actual_start.date()} to {actual_end.date()} ({actual_years:.1f} years)")
    print(f"{'=' * 100}")
    
    strategies = {
        'QVR': QVREngine(),
        'DNA3-V3.1': V31Engine(),
    }
    
    results = {}
    eq_curves = {}
    trade_stats = {}
    
    for name, engine in strategies.items():
        print(f"\n  Running {name}...")
        eq = engine.run(dc, nifty, actual_start, actual_end)
        
        if eq is not None and len(eq) > 10:
            m = calc_metrics(eq, actual_years)
            sells = engine.trade_log
            wins = [t for t in sells if t['PnL%'] > 0]
            losses = [t for t in sells if t['PnL%'] <= 0]
            
            results[name] = m
            eq_curves[name] = eq
            trade_stats[name] = sells
            
            wr = len(wins)/len(sells)*100 if sells else 0
            aw = np.mean([t['PnL%'] for t in wins]) if wins else 0
            al = np.mean([t['PnL%'] for t in losses]) if losses else 0
            exp = (wr/100)*aw - (1-wr/100)*abs(al) if sells else 0
            
            results[name].update({
                'Trades': len(sells), 'WinRate%': round(wr, 1),
                'AvgWin%': round(aw, 1), 'AvgLoss%': round(al, 1),
                'MaxWin%': round(max([t['PnL%'] for t in sells], default=0), 1),
                'MaxLoss%': round(min([t['PnL%'] for t in sells], default=0), 1),
                'AvgHold': round(np.mean([t['Hold'] for t in sells]), 0) if sells else 0,
                'Expectancy%': round(exp, 2),
                'Alpha%': round(m['CAGR%'] - n_cagr, 2),
            })
    
    # ============================================
    # HEAD TO HEAD
    # ============================================
    print(f"\n{'_' * 100}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"{'_' * 100}")
    
    print(f"\n  {'Metric':<18} {'QVR':>12} {'DNA3-V3.1':>12} {'Nifty':>12}")
    print(f"  {'-'*58}")
    
    for metric in ['CAGR%', 'Total%', 'MaxDD%', 'Sharpe', 'Sortino', 'Calmar', 'Alpha%',
                    'Trades', 'WinRate%', 'AvgWin%', 'AvgLoss%', 'MaxWin%', 'MaxLoss%', 
                    'AvgHold', 'Expectancy%']:
        qvr_val = results.get('QVR', {}).get(metric, 'N/A')
        v31_val = results.get('DNA3-V3.1', {}).get(metric, 'N/A')
        n_val = n_m.get(metric, '-')
        
        qvr_str = f"{qvr_val}" if isinstance(qvr_val, (int, float)) else qvr_val
        v31_str = f"{v31_val}" if isinstance(v31_val, (int, float)) else v31_val
        n_str = f"{n_val}" if isinstance(n_val, (int, float)) else n_val
        
        print(f"  {metric:<18} {qvr_str:>12} {v31_str:>12} {n_str:>12}")
    
    # ============================================
    # REGIME BREAKDOWN
    # ============================================
    print(f"\n{'_' * 100}")
    print(f"  PER-REGIME PERFORMANCE")
    print(f"{'_' * 100}")
    
    regime_rows = []
    for name in ['QVR', 'DNA3-V3.1']:
        if name in eq_curves:
            rp = calc_regime_perf(eq_curves[name])
            print(f"\n  {name}:")
            print(f"  {'Regime':<12} {'Days':>6} {'Total':>10} {'Ann.':>10}")
            print(f"  {'-'*40}")
            for r in ['BULL', 'MILD_BULL', 'SIDEWAYS', 'BEAR']:
                d = rp[r]
                print(f"  {r:<12} {d['days']:>6} {d['total']:>+9.1f}% {d['ann']:>+9.1f}%")
                regime_rows.append({'Strategy': name, 'Regime': r, 'Days': d['days'],
                                   'Total%': d['total'], 'Ann%': d['ann']})
    
    # ============================================
    # QVR TRADE DETAILS
    # ============================================
    if 'QVR' in trade_stats:
        print(f"\n{'_' * 100}")
        print(f"  QVR TRADE DETAILS")
        print(f"{'_' * 100}")
        
        trades = trade_stats['QVR']
        
        # Exit reason breakdown
        reasons = {}
        for t in trades:
            r = t['Reason']
            if r not in reasons: reasons[r] = []
            reasons[r].append(t['PnL%'])
        
        print(f"\n  Exit Reason Breakdown:")
        print(f"  {'Reason':<18} {'Count':>6} {'Avg PnL':>8} {'Win%':>6}")
        print(f"  {'-'*42}")
        for r, pnls in sorted(reasons.items()):
            avg = np.mean(pnls)
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            print(f"  {r:<18} {len(pnls):>6} {avg:>+7.1f}% {wr:>5.0f}%")
        
        # Holding period analysis
        holds = [t['Hold'] for t in trades]
        print(f"\n  Holding Period: mean {np.mean(holds):.0f}d, median {np.median(holds):.0f}d")
    
    # ============================================
    # ROLLING RETURNS
    # ============================================
    print(f"\n{'_' * 100}")
    print(f"  ROLLING 12-MONTH RETURNS")
    print(f"{'_' * 100}")
    
    print(f"\n  {'Strategy':<15} {'Median':>8} {'Mean':>8} {'Best':>8} {'Worst':>8} {'%Pos':>6}")
    print(f"  {'-'*52}")
    
    for name in ['QVR', 'DNA3-V3.1']:
        if name not in eq_curves: continue
        eq = eq_curves[name].copy()
        eq['month'] = eq['date'].dt.to_period('M')
        mo = eq.groupby('month')['equity'].last()
        r = mo.pct_change(12).dropna() * 100
        if len(r) > 0:
            print(f"  {name:<15} {r.median():>+7.1f}% {r.mean():>+7.1f}% {r.max():>+7.1f}% {r.min():>+7.1f}% {(r>0).mean()*100:>5.0f}%")
    
    # Nifty rolling
    n_eq = nifty_eq[['date', 'equity']].copy()
    n_eq['month'] = n_eq['date'].dt.to_period('M')
    n_mo = n_eq.groupby('month')['equity'].last()
    n_r = n_mo.pct_change(12).dropna() * 100
    if len(n_r) > 0:
        print(f"  {'Nifty':<15} {n_r.median():>+7.1f}% {n_r.mean():>+7.1f}% {n_r.max():>+7.1f}% {n_r.min():>+7.1f}% {(n_r>0).mean()*100:>5.0f}%")
    
    # ============================================
    # VERDICT
    # ============================================
    print(f"\n{'=' * 100}")
    print(f"  VERDICT")
    print(f"{'=' * 100}")
    
    qvr = results.get('QVR', {})
    v31 = results.get('DNA3-V3.1', {})
    
    winner_cagr = 'QVR' if qvr.get('CAGR%', 0) > v31.get('CAGR%', 0) else 'V3.1'
    winner_sharpe = 'QVR' if qvr.get('Sharpe', 0) > v31.get('Sharpe', 0) else 'V3.1'
    winner_dd = 'QVR' if abs(qvr.get('MaxDD%', -99)) < abs(v31.get('MaxDD%', -99)) else 'V3.1'
    winner_exp = 'QVR' if qvr.get('Expectancy%', 0) > v31.get('Expectancy%', 0) else 'V3.1'
    
    print(f"\n  CAGR:        {winner_cagr} wins")
    print(f"  Sharpe:      {winner_sharpe} wins")
    print(f"  Max DD:      {winner_dd} wins")
    print(f"  Expectancy:  {winner_exp} wins")
    print(f"\n  Regime: {detect_regime(nifty, nifty.index[-1])}")
    
    # Save results
    pd.DataFrame([
        {'Strategy': 'QVR', **qvr},
        {'Strategy': 'DNA3-V3.1', **v31},
        {'Strategy': 'Nifty', 'CAGR%': round(n_cagr, 2), 'Total%': round(n_total, 2), 'MaxDD%': n_m['MaxDD%']},
    ]).to_csv(f"{OUTPUT_DIR}/qvr_vs_v31_summary.csv", index=False)
    
    pd.DataFrame(regime_rows).to_csv(f"{OUTPUT_DIR}/qvr_vs_v31_regime.csv", index=False)
    
    if 'QVR' in trade_stats:
        pd.DataFrame(trade_stats['QVR']).to_csv(f"{OUTPUT_DIR}/qvr_trades.csv", index=False)
    
    print(f"\n  Files saved to {OUTPUT_DIR}/qvr_*.csv")


if __name__ == "__main__":
    print("=" * 100)
    print("QVR (Quality Value Recovery) STRATEGY BACKTEST")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 100)
    run()
