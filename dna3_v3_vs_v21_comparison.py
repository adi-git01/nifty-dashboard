"""
DNA3-V3 ADAPTIVE vs DNA3-V2.1 vs NIFTY — COMPREHENSIVE COMPARISON
===================================================================
Side-by-side comparison across multiple time horizons with:
- CAGR comparison (6mo, 1y, 3y, 5y, 10y)
- Monthly rolling returns
- Regime-specific performance
- Drawdown analysis
- Trade statistics
- Risk-adjusted metrics (Sharpe, Sortino, Calmar)

DNA3-V2.1 Rules:
  Entry: RS > 2%, Vol > 30%, Price > MA50
  Exit:  -15% hard stop, trailing 10% after 10% gain
  Max:   10 positions, 4 per sector

DNA3-V3 Adaptive Rules:
  Entry: RS > regime-threshold, Price > MA50, near 20d high
  Exit:  12% trailing stop
  Size:  6-12 positions (regime dependent)
  Cash:  5-40% reserve (regime dependent)

Usage:
  python dna3_v3_vs_v21_comparison.py
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

# ============================================================
# CONFIG
# ============================================================
INITIAL_CAPITAL = 1000000
COST_BPS = 50  # 0.5% each way
OUTPUT_DIR = "analysis_2026"

# Time horizons to test
HORIZONS = {
    '6mo':  0.5,
    '1y':   1,
    '3y':   3,
    '5y':   5,
    '10y': 10,
}

# We fetch max data once, then slice per-horizon
MAX_YEARS = 10


# ============================================================
# REGIME DETECTION
# ============================================================
def detect_regime(nifty_df, date):
    """4-regime classification from price only."""
    idx = nifty_df.index.searchsorted(date)
    if idx < 200: return 'UNKNOWN'
    
    window = nifty_df.iloc[max(0, idx - 252):idx + 1]
    if len(window) < 63: return 'UNKNOWN'
    
    price = window['Close'].iloc[-1]
    ma50 = window['Close'].rolling(50).mean().iloc[-1]
    ma200 = window['Close'].rolling(200).mean().iloc[-1]
    ret_3m = (price - window['Close'].iloc[-63]) / window['Close'].iloc[-63] * 100
    peak = window['Close'].cummax().iloc[-1]
    drawdown = (price - peak) / peak * 100
    
    if price > ma50 and ma50 > ma200 and ret_3m > 5:
        return 'BULL'
    elif price > ma50 and ret_3m > 0:
        return 'MILD_BULL'
    elif price < ma50 and (ret_3m < -5 or drawdown < -10):
        return 'BEAR'
    else:
        return 'SIDEWAYS'


# ============================================================
# DNA3-V3 ADAPTIVE CONFIG
# ============================================================
V3_REGIME_CONFIG = {
    'BULL':      {'rs_min': 10, 'max_pos': 12, 'cash_reserve': 0.05, 'need_near_high': False},
    'MILD_BULL': {'rs_min': 15, 'max_pos': 10, 'cash_reserve': 0.10, 'need_near_high': False},
    'SIDEWAYS':  {'rs_min': 15, 'max_pos':  8, 'cash_reserve': 0.20, 'need_near_high': True},
    'BEAR':      {'rs_min': 20, 'max_pos':  6, 'cash_reserve': 0.40, 'need_near_high': True},
    'UNKNOWN':   {'rs_min': 15, 'max_pos':  8, 'cash_reserve': 0.20, 'need_near_high': False},
}


# ============================================================
# UNIFIED STRATEGY ENGINE
# ============================================================
class StrategyEngine:
    """Runs either DNA3-V2.1 or DNA3-V3 Adaptive."""
    
    def __init__(self, name, strategy_type):
        self.name = name
        self.strategy_type = strategy_type  # 'v21' or 'v3'
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
        self.regime_history = []
    
    def reset(self):
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
        self.regime_history = []
    
    def get_price(self, data_cache, ticker, date):
        if ticker not in data_cache: return None
        df = data_cache[ticker]
        idx = df.index.searchsorted(date)
        if idx == 0: return None
        return df['Close'].iloc[min(idx, len(df)-1)]
    
    def calculate_indicators(self, window, nifty_window):
        """Calculate common indicators."""
        if len(window) < 100 or len(nifty_window) < 64:
            return None
        
        price = window['Close'].iloc[-1]
        ma50 = window['Close'].rolling(50).mean().iloc[-1]
        ma200_s = window['Close'].rolling(200).mean()
        ma200 = ma200_s.iloc[-1] if len(window) >= 200 and not pd.isna(ma200_s.iloc[-1]) else ma50
        high_20 = window['Close'].rolling(20).max().iloc[-1]
        
        # RS (63-day)
        if len(window) > 63 and len(nifty_window) > 63:
            rs_stock = (price - window['Close'].iloc[-63]) / window['Close'].iloc[-63]
            rs_nifty = (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-63]) / nifty_window['Close'].iloc[-63]
            rs_score = (rs_stock - rs_nifty) * 100
        else:
            rs_score = 0
        
        # Volatility (60d)
        rets = window['Close'].pct_change().dropna()[-60:]
        volatility = rets.std() * np.sqrt(252) * 100 if len(rets) > 10 else 0
        
        # Liquidity
        vol_20d = window['Volume'].rolling(20).mean().iloc[-1]
        liquidity = vol_20d * price
        
        return {
            'price': price,
            'ma50': ma50,
            'ma200': ma200,
            'high_20': high_20,
            'rs_score': rs_score,
            'volatility': volatility,
            'liquidity': liquidity,
        }
    
    def passes_v21_filter(self, ind):
        """DNA3-V2.1: RS > 2%, Vol > 30%, Price > MA50"""
        if ind['rs_score'] < 2.0: return False
        if ind['volatility'] < 30: return False
        if ind['price'] < ind['ma50']: return False
        return True
    
    def passes_v3_filter(self, ind, regime):
        """DNA3-V3 Adaptive: regime-dependent thresholds."""
        cfg = V3_REGIME_CONFIG.get(regime, V3_REGIME_CONFIG['UNKNOWN'])
        
        if ind['rs_score'] < cfg['rs_min']: return False
        if ind['price'] < ind['ma50']: return False
        if ind['liquidity'] < 5_000_000: return False
        if cfg['need_near_high'] and ind['price'] < ind['high_20'] * 0.98: return False
        return True
    
    def get_max_positions(self, regime):
        if self.strategy_type == 'v21':
            return 10
        else:
            return V3_REGIME_CONFIG.get(regime, V3_REGIME_CONFIG['UNKNOWN'])['max_pos']
    
    def get_cash_reserve(self, regime):
        if self.strategy_type == 'v21':
            return 0.0
        else:
            return V3_REGIME_CONFIG.get(regime, V3_REGIME_CONFIG['UNKNOWN'])['cash_reserve']
    
    def check_exits(self, data_cache, date):
        to_exit = []
        
        for t, pos in self.positions.items():
            price = self.get_price(data_cache, t, date)
            if not price: continue
            
            if price > pos['peak']:
                pos['peak'] = price
            
            ret = (price - pos['entry']) / pos['entry']
            exit_reason = None
            
            if self.strategy_type == 'v21':
                # V2.1: -15% hard stop + trailing after +10%
                if price < pos['stop']:
                    exit_reason = 'Stop/Trail'
                if ret > 0.10:
                    trail = pos['peak'] * 0.90
                    if trail > pos['stop']:
                        pos['stop'] = trail
                    if price < pos['stop']:
                        exit_reason = 'TrailingStop'
            else:
                # V3: 12% trailing stop always active
                trail_price = pos['peak'] * 0.88
                if price < trail_price:
                    exit_reason = 'TrailingStop'
                # Hard stop at -20%
                if ret < -0.20:
                    exit_reason = 'HardStop'
            
            if exit_reason:
                proceeds = pos['shares'] * price * (1 - COST_BPS / 10000)
                self.capital += proceeds
                pnl = ret * 100
                
                self.trade_log.append({
                    'Ticker': t,
                    'PnL%': round(pnl, 2),
                    'Reason': exit_reason,
                    'Entry_Date': pos['entry_date'].strftime('%Y-%m-%d'),
                    'Exit_Date': date.strftime('%Y-%m-%d'),
                    'Hold_Days': (date - pos['entry_date']).days,
                })
                to_exit.append(t)
        
        for t in to_exit:
            del self.positions[t]
    
    def scan_and_buy(self, data_cache, nifty_df, date, regime):
        max_pos = self.get_max_positions(regime)
        cash_reserve = self.get_cash_reserve(regime)
        
        # Reduce positions if regime tightened
        if self.strategy_type == 'v3' and len(self.positions) > max_pos:
            # Sell weakest RS positions to get under limit
            pos_list = []
            for t, pos in self.positions.items():
                price = self.get_price(data_cache, t, date)
                if price:
                    ret = (price - pos['entry']) / pos['entry']
                    pos_list.append((t, ret, price, pos))
            pos_list.sort(key=lambda x: x[1])  # Weakest first
            
            while len(self.positions) > max_pos and pos_list:
                t, ret, price, pos = pos_list.pop(0)
                if t in self.positions:
                    proceeds = pos['shares'] * price * (1 - COST_BPS / 10000)
                    self.capital += proceeds
                    self.trade_log.append({
                        'Ticker': t, 'PnL%': round(ret*100, 2),
                        'Reason': 'RegimeReduce',
                        'Entry_Date': pos['entry_date'].strftime('%Y-%m-%d'),
                        'Exit_Date': date.strftime('%Y-%m-%d'),
                        'Hold_Days': (date - pos['entry_date']).days,
                    })
                    del self.positions[t]
        
        if len(self.positions) >= max_pos:
            return
        
        nifty_idx = nifty_df.index.searchsorted(date)
        if nifty_idx < 252: return
        nifty_window = nifty_df.iloc[max(0, nifty_idx - 252):nifty_idx + 1]
        
        candidates = []
        for t in data_cache:
            if t == 'NIFTY' or t in self.positions: continue
            df = data_cache[t]
            idx = df.index.searchsorted(date)
            if idx < 100: continue
            
            window = df.iloc[max(0, idx - 252):idx + 1]
            if len(window) < 100: continue
            
            ind = self.calculate_indicators(window, nifty_window)
            if ind is None: continue
            
            passes = False
            if self.strategy_type == 'v21':
                passes = self.passes_v21_filter(ind)
            else:
                passes = self.passes_v3_filter(ind, regime)
            
            if passes:
                candidates.append({'ticker': t, 'ind': ind})
        
        # Rank by RS (strongest first)
        candidates.sort(key=lambda x: -x['ind']['rs_score'])
        
        # Sector cap
        SECTOR_CAP = 4 if self.strategy_type == 'v21' else 3
        selected = []
        sector_count = {}
        for c in candidates:
            sec = SECTOR_MAP.get(c['ticker'], 'Unknown')
            curr = sum(1 for t in self.positions if SECTOR_MAP.get(t, 'Unknown') == sec)
            if sector_count.get(sec, 0) + curr < SECTOR_CAP:
                selected.append(c)
                sector_count[sec] = sector_count.get(sec, 0) + 1
            if len(selected) + len(self.positions) >= max_pos:
                break
        
        # Enforce cash reserve
        total_equity = self.get_equity(data_cache, date)
        min_cash = total_equity * cash_reserve
        available_cash = max(0, self.capital - min_cash)
        
        free = max_pos - len(self.positions)
        for c in selected[:free]:
            price = c['ind']['price']
            size = available_cash / (free + 1)
            shares = int(size / price)
            if shares < 1: continue
            cost = shares * price * (1 + COST_BPS / 10000)
            if available_cash >= cost and cost > 5000:
                available_cash -= cost
                self.capital -= cost
                
                stop = price * 0.85 if self.strategy_type == 'v21' else price * 0.80
                self.positions[c['ticker']] = {
                    'entry': price,
                    'peak': price,
                    'shares': shares,
                    'stop': stop,
                    'entry_date': date,
                }
    
    def get_equity(self, data_cache, date):
        val = self.capital
        for t, pos in self.positions.items():
            p = self.get_price(data_cache, t, date)
            if p: val += pos['shares'] * p
        return val
    
    def run_backtest(self, data_cache, nifty_df, start_date, end_date):
        """Run backtest for a specific period."""
        self.reset()
        
        start_idx = nifty_df.index.searchsorted(start_date)
        end_idx = nifty_df.index.searchsorted(end_date)
        dates = nifty_df.index[start_idx:end_idx + 1]
        
        if len(dates) < 10:
            return None
        
        day_count = 0
        rebalance_freq = 10  # Every 10 trading days
        
        for date in dates:
            regime = detect_regime(nifty_df, date)
            self.regime_history.append({'date': date, 'regime': regime})
            
            # Always check exits
            self.check_exits(data_cache, date)
            
            # Scan every rebalance_freq days
            if day_count % rebalance_freq == 0:
                self.scan_and_buy(data_cache, nifty_df, date, regime)
            
            eq = self.get_equity(data_cache, date)
            self.history.append({'date': date, 'equity': eq, 'regime': regime})
            day_count += 1
        
        return pd.DataFrame(self.history)


# ============================================================
# ANALYTICS
# ============================================================
def calc_metrics(eq_df, n_years):
    """Calculate key metrics from equity curve."""
    if eq_df is None or len(eq_df) < 2:
        return None
    
    start_val = eq_df['equity'].iloc[0]
    end_val = eq_df['equity'].iloc[-1]
    total_ret = (end_val / start_val - 1) * 100
    
    if n_years > 0:
        cagr = ((end_val / start_val) ** (1 / n_years) - 1) * 100
    else:
        cagr = total_ret
    
    # Max Drawdown
    eq_df = eq_df.copy()
    eq_df['peak'] = eq_df['equity'].cummax()
    eq_df['dd'] = (eq_df['equity'] - eq_df['peak']) / eq_df['peak'] * 100
    max_dd = eq_df['dd'].min()
    
    # Monthly returns for Sharpe
    eq_df['month'] = eq_df['date'].dt.to_period('M')
    monthly = eq_df.groupby('month')['equity'].last()
    monthly_rets = monthly.pct_change().dropna()
    
    if len(monthly_rets) > 2:
        sharpe = (monthly_rets.mean() / monthly_rets.std()) * np.sqrt(12) if monthly_rets.std() > 0 else 0
        neg_rets = monthly_rets[monthly_rets < 0]
        sortino = (monthly_rets.mean() / neg_rets.std()) * np.sqrt(12) if len(neg_rets) > 0 and neg_rets.std() > 0 else 0
    else:
        sharpe = 0
        sortino = 0
    
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0
    
    return {
        'Total_Return%': round(total_ret, 2),
        'CAGR%': round(cagr, 2),
        'Max_DD%': round(max_dd, 2),
        'Sharpe': round(sharpe, 2),
        'Sortino': round(sortino, 2),
        'Calmar': round(calmar, 2),
    }


def calc_monthly_rolling(eq_df, window_months=12):
    """Calculate rolling N-month returns from equity curve."""
    if eq_df is None or len(eq_df) < 22:
        return pd.DataFrame()
    
    eq_df = eq_df.copy()
    eq_df['month'] = eq_df['date'].dt.to_period('M')
    monthly = eq_df.groupby('month').agg({'equity': 'last', 'date': 'last'}).reset_index()
    monthly['return'] = monthly['equity'].pct_change(window_months) * 100
    
    return monthly[['date', 'return']].dropna()


def calc_regime_performance(eq_df):
    """Calculate returns during each regime."""
    results = {}
    for regime in ['BULL', 'MILD_BULL', 'SIDEWAYS', 'BEAR']:
        rows = eq_df[eq_df['regime'] == regime]
        if len(rows) < 2:
            results[regime] = {'days': 0, 'total_ret': 0, 'ann_ret': 0}
            continue
        
        daily_rets = rows['equity'].pct_change().dropna()
        total = (np.prod(1 + daily_rets) - 1) * 100
        days = len(rows)
        ann = ((1 + total/100) ** (252 / days) - 1) * 100 if days > 20 else total
        
        results[regime] = {'days': days, 'total_ret': round(total, 2), 'ann_ret': round(ann, 2)}
    
    return results


# ============================================================
# DATA LOADING
# ============================================================
def fetch_all_data():
    """Fetch max data (10Y + buffer) once."""
    buffer_days = 500
    start = (datetime.now() - timedelta(days=365 * MAX_YEARS + buffer_days)).strftime('%Y-%m-%d')
    
    print(f"[1/3] Fetching Nifty 50 ({MAX_YEARS}Y+ history)...")
    nifty = yf.Ticker("^NSEI").history(start=start)
    if nifty.empty:
        print("ERROR: Cannot fetch Nifty data!")
        return None, {}
    nifty.index = nifty.index.tz_localize(None)
    
    print(f"[2/3] Bulk downloading {len(TICKERS[:500])} stocks...")
    t0 = time.time()
    try:
        bulk = yf.download(
            TICKERS[:500], start=start,
            group_by='ticker', threads=True, progress=True, auto_adjust=True,
        )
    except Exception as e:
        print(f"Bulk download failed: {e}")
        return nifty, {'NIFTY': nifty}
    
    data_cache = {'NIFTY': nifty}
    loaded = 0
    for t in TICKERS[:500]:
        try:
            if t in bulk.columns.get_level_values(0):
                df = bulk[t].dropna(how='all')
                if not df.empty and len(df) > 200:
                    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                    data_cache[t] = df
                    loaded += 1
        except:
            pass
    
    print(f"[3/3] Loaded {loaded} stocks in {time.time()-t0:.0f}s. Nifty: {len(nifty)} days.")
    return nifty, data_cache


# ============================================================
# MAIN COMPARISON
# ============================================================
def run_comparison():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    nifty, data_cache = fetch_all_data()
    if nifty is None:
        return
    
    now = datetime.now()
    
    # ========================================
    # PART 1: MULTI-HORIZON CAGR COMPARISON
    # ========================================
    print("\n" + "=" * 90)
    print("PART 1: DNA3-V3 ADAPTIVE vs DNA3-V2.1 vs NIFTY  ---- MULTI-HORIZON COMPARISON")
    print("=" * 90)
    
    summary_table = []
    all_equity_curves = {}
    all_trade_stats = {}
    
    for horizon_name, years in HORIZONS.items():
        start_date = now - timedelta(days=int(365.25 * years))
        end_date = now - timedelta(days=1)
        
        # Check if we have enough data
        start_idx = nifty.index.searchsorted(start_date)
        if start_idx >= len(nifty) - 10:
            print(f"\n  [{horizon_name}] Insufficient data, skipping.")
            continue
        
        actual_start = nifty.index[start_idx]
        actual_end = nifty.index[-1]
        actual_years = (actual_end - actual_start).days / 365.25
        
        print(f"\n{'_' * 90}")
        print(f"  HORIZON: {horizon_name.upper()} ({actual_start.date()} to {actual_end.date()})")
        print(f"{'_' * 90}")
        
        # Nifty benchmark
        n_start_val = nifty.iloc[start_idx]['Close']
        n_end_val = nifty.iloc[-1]['Close']
        n_total = (n_end_val / n_start_val - 1) * 100
        n_cagr = ((n_end_val / n_start_val) ** (1 / actual_years) - 1) * 100 if actual_years > 0 else n_total
        
        # Nifty equity curve for DD
        nifty_eq = nifty.iloc[start_idx:].copy()
        nifty_eq['equity'] = nifty_eq['Close'] / n_start_val * INITIAL_CAPITAL
        nifty_eq['date'] = nifty_eq.index
        nifty_eq['regime'] = 'N/A'
        n_metrics = calc_metrics(nifty_eq[['date', 'equity', 'regime']], actual_years)
        
        results_row = {
            'Horizon': horizon_name,
            'Period': f"{actual_start.date()} to {actual_end.date()}",
            'Years': round(actual_years, 1),
        }
        
        # Run both strategies
        for strat_name, strat_type in [('DNA3-V2.1', 'v21'), ('DNA3-V3', 'v3')]:
            engine = StrategyEngine(strat_name, strat_type)
            eq_df = engine.run_backtest(data_cache, nifty, actual_start, actual_end)
            
            if eq_df is not None and len(eq_df) > 10:
                metrics = calc_metrics(eq_df, actual_years)
                
                sells = [t for t in engine.trade_log if 'PnL%' in t]
                wins = [t for t in sells if t['PnL%'] > 0]
                
                results_row[f'{strat_name}_Return%'] = metrics['Total_Return%']
                results_row[f'{strat_name}_CAGR%'] = metrics['CAGR%']
                results_row[f'{strat_name}_MaxDD%'] = metrics['Max_DD%']
                results_row[f'{strat_name}_Sharpe'] = metrics['Sharpe']
                results_row[f'{strat_name}_Sortino'] = metrics['Sortino']
                results_row[f'{strat_name}_Calmar'] = metrics['Calmar']
                results_row[f'{strat_name}_Alpha%'] = round(metrics['CAGR%'] - n_cagr, 2)
                results_row[f'{strat_name}_Trades'] = len(sells)
                results_row[f'{strat_name}_WinRate%'] = round(len(wins)/len(sells)*100, 1) if sells else 0
                
                # Store equity curve for longest horizon
                all_equity_curves[f'{strat_name}_{horizon_name}'] = eq_df
                all_trade_stats[f'{strat_name}_{horizon_name}'] = engine.trade_log
            else:
                for suffix in ['Return%', 'CAGR%', 'MaxDD%', 'Sharpe', 'Sortino', 'Calmar', 'Alpha%', 'Trades', 'WinRate%']:
                    results_row[f'{strat_name}_{suffix}'] = 'N/A'
        
        results_row['Nifty_Return%'] = round(n_total, 2)
        results_row['Nifty_CAGR%'] = round(n_cagr, 2)
        results_row['Nifty_MaxDD%'] = n_metrics['Max_DD%'] if n_metrics else 'N/A'
        
        summary_table.append(results_row)
        
        # Print compact summary
        print(f"\n  {'Metric':<20} {'DNA3-V2.1':>12} {'DNA3-V3':>12} {'Nifty':>12}")
        print(f"  {'-'*58}")
        for metric in ['CAGR%', 'Return%', 'MaxDD%', 'Sharpe', 'Alpha%', 'Trades', 'WinRate%']:
            v21 = results_row.get(f'DNA3-V2.1_{metric}', 'N/A')
            v3 = results_row.get(f'DNA3-V3_{metric}', 'N/A')
            nifty_val = results_row.get(f'Nifty_{metric}', '-')
            
            v21_str = f"{v21}" if isinstance(v21, (int, float)) else v21
            v3_str = f"{v3}" if isinstance(v3, (int, float)) else v3
            n_str = f"{nifty_val}" if isinstance(nifty_val, (int, float)) else nifty_val
            
            print(f"  {metric:<20} {v21_str:>12} {v3_str:>12} {n_str:>12}")
    
    # Save summary
    summary_df = pd.DataFrame(summary_table)
    summary_df.to_csv(f"{OUTPUT_DIR}/dna3_v3_vs_v21_summary.csv", index=False)
    
    # ========================================
    # PART 2: MONTHLY ROLLING RETURNS (Longest horizon)
    # ========================================
    print(f"\n\n{'=' * 90}")
    print("PART 2: MONTHLY ROLLING RETURNS (12-Month Window)")
    print("=" * 90)
    
    longest = max(HORIZONS.items(), key=lambda x: x[1])
    longest_name = longest[0]
    
    rolling_data = []
    
    for strat_name in ['DNA3-V2.1', 'DNA3-V3']:
        key = f'{strat_name}_{longest_name}'
        if key in all_equity_curves:
            eq_df = all_equity_curves[key]
            rolling = calc_monthly_rolling(eq_df, window_months=12)
            rolling['Strategy'] = strat_name
            rolling_data.append(rolling)
    
    # Nifty rolling
    start_idx = nifty.index.searchsorted(now - timedelta(days=int(365.25 * longest[1])))
    nifty_eq = nifty.iloc[start_idx:].copy()
    nifty_eq['equity'] = nifty_eq['Close'] / nifty_eq['Close'].iloc[0] * INITIAL_CAPITAL
    nifty_eq['date'] = nifty_eq.index
    nifty_eq['regime'] = 'N/A'
    nifty_rolling = calc_monthly_rolling(nifty_eq[['date', 'equity', 'regime']], window_months=12)
    nifty_rolling['Strategy'] = 'Nifty'
    rolling_data.append(nifty_rolling)
    
    if rolling_data:
        all_rolling = pd.concat(rolling_data, ignore_index=True)
        all_rolling.to_csv(f"{OUTPUT_DIR}/dna3_v3_vs_v21_rolling.csv", index=False)
        
        # Summary stats
        print(f"\n  {'Strategy':<15} {'Median':>8} {'Mean':>8} {'Best':>8} {'Worst':>8} {'%Positive':>10}")
        print(f"  {'-'*60}")
        for strat in ['DNA3-V2.1', 'DNA3-V3', 'Nifty']:
            subset = all_rolling[all_rolling['Strategy'] == strat]['return']
            if len(subset) > 0:
                print(f"  {strat:<15} {subset.median():>+7.1f}% {subset.mean():>+7.1f}% {subset.max():>+7.1f}% {subset.min():>+7.1f}% {(subset>0).mean()*100:>9.0f}%")
    
    # Also 3-month and 6-month rolling
    for window_m, label in [(3, '3-Month'), (6, '6-Month')]:
        print(f"\n  Rolling {label} Returns:")
        print(f"  {'Strategy':<15} {'Median':>8} {'Mean':>8} {'Best':>8} {'Worst':>8} {'%Positive':>10}")
        print(f"  {'-'*60}")
        for strat_name in ['DNA3-V2.1', 'DNA3-V3']:
            key = f'{strat_name}_{longest_name}'
            if key in all_equity_curves:
                r = calc_monthly_rolling(all_equity_curves[key], window_months=window_m)
                if len(r) > 0:
                    s = r['return']
                    print(f"  {strat_name:<15} {s.median():>+7.1f}% {s.mean():>+7.1f}% {s.max():>+7.1f}% {s.min():>+7.1f}% {(s>0).mean()*100:>9.0f}%")
        
        nifty_r = calc_monthly_rolling(nifty_eq[['date', 'equity', 'regime']], window_months=window_m)
        if len(nifty_r) > 0:
            s = nifty_r['return']
            print(f"  {'Nifty':<15} {s.median():>+7.1f}% {s.mean():>+7.1f}% {s.max():>+7.1f}% {s.min():>+7.1f}% {(s>0).mean()*100:>9.0f}%")
    
    # ========================================
    # PART 3: PER-REGIME PERFORMANCE (Longest horizon)
    # ========================================
    print(f"\n\n{'=' * 90}")
    print("PART 3: PER-REGIME PERFORMANCE")
    print("=" * 90)
    
    regime_table = []
    
    for strat_name in ['DNA3-V2.1', 'DNA3-V3']:
        key = f'{strat_name}_{longest_name}'
        if key in all_equity_curves:
            eq_df = all_equity_curves[key]
            rp = calc_regime_performance(eq_df)
            
            print(f"\n  {strat_name}:")
            print(f"  {'Regime':<12} {'Days':>6} {'Total Ret':>10} {'Ann. Ret':>10}")
            print(f"  {'-'*42}")
            for regime in ['BULL', 'MILD_BULL', 'SIDEWAYS', 'BEAR']:
                d = rp[regime]
                emoji = '+' if d['total_ret'] > 0 else '-'
                print(f"  {regime:<12} {d['days']:>6} {d['total_ret']:>+9.1f}% {d['ann_ret']:>+9.1f}%")
                regime_table.append({
                    'Strategy': strat_name, 'Regime': regime,
                    'Days': d['days'], 'Total_Return%': d['total_ret'], 'Annualized%': d['ann_ret']
                })
    
    pd.DataFrame(regime_table).to_csv(f"{OUTPUT_DIR}/dna3_v3_vs_v21_regime.csv", index=False)
    
    # ========================================
    # PART 4: TRADE ANALYSIS (Longest horizon)
    # ========================================
    print(f"\n\n{'=' * 90}")
    print("PART 4: TRADE ANALYSIS")
    print("=" * 90)
    
    for strat_name in ['DNA3-V2.1', 'DNA3-V3']:
        key = f'{strat_name}_{longest_name}'
        if key in all_trade_stats:
            trades = all_trade_stats[key]
            sells = [t for t in trades if 'PnL%' in t]
            wins = [t for t in sells if t['PnL%'] > 0]
            losses = [t for t in sells if t['PnL%'] <= 0]
            
            print(f"\n  {strat_name}:")
            print(f"    Total Trades : {len(sells)}")
            print(f"    Win Rate     : {len(wins)/len(sells)*100:.0f}%" if sells else "    Win Rate: N/A")
            if wins:
                print(f"    Avg Win      : +{np.mean([t['PnL%'] for t in wins]):.1f}%")
                print(f"    Max Win      : +{max([t['PnL%'] for t in wins]):.1f}%")
            if losses:
                print(f"    Avg Loss     : {np.mean([t['PnL%'] for t in losses]):.1f}%")
                print(f"    Max Loss     : {min([t['PnL%'] for t in losses]):.1f}%")
            if sells:
                print(f"    Avg Hold     : {np.mean([t['Hold_Days'] for t in sells]):.0f} days")
                # Expectancy
                wr = len(wins)/len(sells)
                avg_w = np.mean([t['PnL%'] for t in wins]) if wins else 0
                avg_l = abs(np.mean([t['PnL%'] for t in losses])) if losses else 0
                expectancy = wr * avg_w - (1-wr) * avg_l
                print(f"    Expectancy   : {expectancy:+.2f}% per trade")
    
    # ========================================
    # PART 5: HEAD-TO-HEAD VERDICT
    # ========================================
    print(f"\n\n{'=' * 90}")
    print("PART 5: HEAD-TO-HEAD VERDICT")
    print("=" * 90)
    
    # Count wins across horizons
    v21_wins = 0
    v3_wins = 0
    
    for row in summary_table:
        v21_cagr = row.get('DNA3-V2.1_CAGR%', 0)
        v3_cagr = row.get('DNA3-V3_CAGR%', 0)
        if isinstance(v21_cagr, (int, float)) and isinstance(v3_cagr, (int, float)):
            if v21_cagr > v3_cagr:
                v21_wins += 1
            elif v3_cagr > v21_cagr:
                v3_wins += 1
    
    print(f"\n  CAGR Wins Across Horizons:")
    print(f"    DNA3-V2.1 : {v21_wins}")
    print(f"    DNA3-V3   : {v3_wins}")
    
    # Risk-adjusted comparison
    print(f"\n  Risk-Adjusted Comparison (Longest Horizon):")
    for metric_name in ['Sharpe', 'Sortino', 'Calmar']:
        v21 = summary_table[-1].get(f'DNA3-V2.1_{metric_name}', 'N/A')
        v3 = summary_table[-1].get(f'DNA3-V3_{metric_name}', 'N/A')
        winner = "V2.1" if isinstance(v21, (int,float)) and isinstance(v3, (int,float)) and v21 > v3 else "V3"
        print(f"    {metric_name:<10}: V2.1={v21}  V3={v3}  --> {winner}")
    
    # Final recommendation
    print(f"\n  Current Regime: {detect_regime(nifty, nifty.index[-1])}")
    
    print(f"\n  Files saved:")
    for f in ['dna3_v3_vs_v21_summary.csv', 'dna3_v3_vs_v21_rolling.csv', 'dna3_v3_vs_v21_regime.csv']:
        print(f"    {OUTPUT_DIR}/{f}")


# ============================================================
if __name__ == "__main__":
    print("=" * 90)
    print("DNA3-V3 ADAPTIVE vs DNA3-V2.1 vs NIFTY — COMPREHENSIVE COMPARISON")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 90)
    run_comparison()
