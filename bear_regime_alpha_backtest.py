"""
BEAR REGIME ALPHA BACKTEST
==========================
A regime-aware backtest that identifies bearish/choppy/sideways markets,
runs 3 bear-optimized strategies across Nifty 500, and measures which
approach delivers 15%+ CAGR during stress periods.

Regimes (auto-detected from price only):
  BULL       : Nifty > MA50, MA50 > MA200, 3M ret > 5%
  MILD_BULL  : Nifty > MA50, 3M ret 0-5%
  SIDEWAYS   : Nifty within ±3% of MA50 or 3M ret between -5% and 0%
  BEAR       : Nifty < MA50, 3M ret < -5% OR Drawdown > 10%

Strategies:
  RS_LEADER   : Relative strength leaders making highs while market falls
  FORTRESS    : Low beta, low vol defensive compounders
  QUALITY_DIP : Quality stocks with RSI pullbacks in long-term uptrends

Usage:
  python bear_regime_alpha_backtest.py
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
INITIAL_CAPITAL = 1000000  # 10 Lakhs
MAX_POSITIONS = 10
SECTOR_CAP = 3
REBALANCE_FREQ = 10  # Trading days between scans
COST_BPS = 50  # 0.5% each way (entry + exit)

BACKTEST_YEARS = 5
OUTPUT_DIR = "analysis_2026"
RESULTS_FILE = f"{OUTPUT_DIR}/bear_regime_backtest_results.csv"
TRADES_FILE = f"{OUTPUT_DIR}/bear_regime_trades.csv"
EQUITY_FILE = f"{OUTPUT_DIR}/bear_regime_equity.csv"


# ============================================================
# REGIME DETECTION
# ============================================================
def detect_regime(nifty_df, date):
    """Classify market regime using only price and moving averages."""
    idx = nifty_df.index.searchsorted(date)
    if idx < 200:
        return 'UNKNOWN'
    
    window = nifty_df.iloc[max(0, idx - 252):idx + 1]
    if len(window) < 63:
        return 'UNKNOWN'
    
    price = window['Close'].iloc[-1]
    ma50 = window['Close'].rolling(50).mean().iloc[-1]
    ma200 = window['Close'].rolling(200).mean().iloc[-1]
    
    # 3-month return
    ret_3m = (price - window['Close'].iloc[-63]) / window['Close'].iloc[-63] * 100
    
    # Distance from MA50
    dist_ma50 = (price - ma50) / ma50 * 100
    
    # Drawdown from rolling peak
    peak = window['Close'].cummax().iloc[-1]
    drawdown = (price - peak) / peak * 100
    
    # Classification
    if price > ma50 and ma50 > ma200 and ret_3m > 5:
        return 'BULL'
    elif price > ma50 and ret_3m > 0:
        return 'MILD_BULL'
    elif price < ma50 and (ret_3m < -5 or drawdown < -10):
        return 'BEAR'
    else:
        return 'SIDEWAYS'


# ============================================================
# INDICATOR CALCULATIONS
# ============================================================
def calculate_indicators(stock_window, nifty_window):
    """Calculate all technical indicators needed by the 3 strategies."""
    if len(stock_window) < 200 or len(nifty_window) < 64:
        return None
    
    price = stock_window['Close'].iloc[-1]
    
    # Moving Averages
    ma20 = stock_window['Close'].rolling(20).mean().iloc[-1]
    ma50 = stock_window['Close'].rolling(50).mean().iloc[-1]
    ma200_series = stock_window['Close'].rolling(200).mean()
    ma200 = ma200_series.iloc[-1] if not pd.isna(ma200_series.iloc[-1]) else ma50
    
    # Highs
    high_20 = stock_window['Close'].rolling(20).max().iloc[-1]
    
    # RSI (14-period)
    delta = stock_window['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1])) if loss.iloc[-1] != 0 else 50
    
    # Volatility (annualized, 60-day)
    rets = stock_window['Close'].pct_change().dropna()[-60:]
    volatility = rets.std() * np.sqrt(252) * 100
    
    # Beta vs Nifty (60-day)
    nifty_rets = nifty_window['Close'].pct_change().dropna()[-60:]
    common = rets.index.intersection(nifty_rets.index)
    if len(common) > 30:
        cov = np.cov(rets.loc[common], nifty_rets.loc[common])[0][1]
        var = np.var(nifty_rets.loc[common])
        beta = cov / var if var != 0 else 1.0
    else:
        beta = 1.0
    
    # Relative Strength (63-day vs Nifty)
    if len(stock_window) > 63 and len(nifty_window) > 63:
        rs_stock = (price - stock_window['Close'].iloc[-63]) / stock_window['Close'].iloc[-63]
        rs_nifty = (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-63]) / nifty_window['Close'].iloc[-63]
        rs_score = (rs_stock - rs_nifty) * 100
    else:
        rs_score = 0
    
    # Volume ratio (20d vs 50d)
    vol_20d = stock_window['Volume'].rolling(20).mean().iloc[-1]
    vol_50d = stock_window['Volume'].rolling(50).mean().iloc[-1]
    vol_ratio = vol_20d / vol_50d if vol_50d > 0 else 1.0
    
    # Liquidity (20d avg turnover)
    liquidity = vol_20d * price
    
    return {
        'price': price,
        'ma20': ma20,
        'ma50': ma50,
        'ma200': ma200,
        'high_20': high_20,
        'rsi': rsi,
        'beta': beta,
        'volatility': volatility,
        'rs_score': rs_score,
        'vol_ratio': vol_ratio,
        'liquidity': liquidity,
    }


# ============================================================
# STRATEGY FILTERS
# ============================================================
def passes_rs_leader(ind):
    """RS_LEADER: Stocks making highs while market struggles."""
    if ind['price'] < ind['ma50']:           return False  # Must be above MA50
    if ind['rs_score'] < 15:                 return False  # Must outperform Nifty by 15%+
    if ind['price'] < ind['high_20'] * 0.98: return False  # Near 20-day high
    if ind['liquidity'] < 5_000_000:         return False  # Min liquidity ₹50L/day
    return True


def passes_fortress(ind):
    """FORTRESS: Low beta, low vol defensive compounders."""
    if ind['beta'] > 0.7:                    return False  # Low beta
    if ind['volatility'] > 25:               return False  # Low vol
    if ind['price'] < ind['ma200']:          return False  # Must hold MA200
    if ind['liquidity'] < 5_000_000:         return False  # Min liquidity
    return True


def passes_quality_dip(ind):
    """QUALITY_DIP: Quality stocks with RSI pullback in uptrend."""
    if ind['price'] < ind['ma200']:          return False  # Long-term uptrend intact
    if ind['rsi'] > 35:                      return False  # Must be oversold
    if ind['rs_score'] < -15:                return False  # Not the absolute worst
    if ind['vol_ratio'] > 1.5:               return False  # Quiet volume (no panic)
    if ind['liquidity'] < 5_000_000:         return False  # Min liquidity
    return True


STRATEGY_CONFIG = {
    'RS_LEADER': {
        'filter': passes_rs_leader,
        'rank_key': lambda ind: -ind['rs_score'],  # Highest RS first
        'trailing_pct': 0.12,  # 12% trailing stop
        'target_pct': None,    # Ride the trend
        'hard_stop': -0.20,    # 20% max loss
    },
    'FORTRESS': {
        'filter': passes_fortress,
        'rank_key': lambda ind: ind['beta'],  # Lowest beta first
        'trailing_pct': 0.15,  # 15% trailing
        'target_pct': None,
        'hard_stop': -0.15,
    },
    'QUALITY_DIP': {
        'filter': passes_quality_dip,
        'rank_key': lambda ind: ind['rsi'],  # Most oversold first
        'trailing_pct': None,
        'target_pct': 0.10,    # 10% target exit
        'hard_stop': -0.08,    # 8% hard stop
    },
}


# ============================================================
# PORTFOLIO ENGINE (per strategy)
# ============================================================
class StrategyEngine:
    def __init__(self, name):
        self.name = name
        self.config = STRATEGY_CONFIG[name]
        self.capital = INITIAL_CAPITAL
        self.positions = {}   # {ticker: {entry, peak, shares, entry_date, sector}}
        self.trade_log = []
        self.equity_curve = []
    
    def check_exits(self, data_cache, date):
        """Check all positions for exit signals."""
        to_exit = []
        
        for t, pos in self.positions.items():
            if t not in data_cache:
                continue
            df = data_cache[t]
            idx = df.index.searchsorted(date)
            if idx == 0: continue
            price = df['Close'].iloc[min(idx, len(df)-1)]
            
            # Update peak
            if price > pos['peak']:
                pos['peak'] = price
            
            ret = (price - pos['entry']) / pos['entry']
            exit_reason = None
            
            # Hard stop
            if self.config['hard_stop'] and ret < self.config['hard_stop']:
                exit_reason = 'HardStop'
            
            # Target (QUALITY_DIP)
            if self.config['target_pct'] and ret > self.config['target_pct']:
                exit_reason = 'Target'
            
            # Trailing stop
            if self.config['trailing_pct']:
                trail_price = pos['peak'] * (1 - self.config['trailing_pct'])
                if price < trail_price:
                    exit_reason = 'TrailingStop'
            
            # Universal: MA50 break for RS_LEADER and FORTRESS
            if self.name in ('RS_LEADER', 'FORTRESS'):
                if len(df.iloc[:idx+1]) > 50:
                    ma50 = df['Close'].iloc[:idx+1].rolling(50).mean().iloc[-1]
                    if price < ma50:
                        exit_reason = 'MA50Break'
            
            if exit_reason:
                proceeds = pos['shares'] * price * (1 - COST_BPS / 10000)
                self.capital += proceeds
                pnl_pct = ret * 100
                
                self.trade_log.append({
                    'Ticker': t,
                    'Strategy': self.name,
                    'Action': 'SELL',
                    'Entry_Price': pos['entry'],
                    'Exit_Price': price,
                    'Shares': pos['shares'],
                    'PnL%': round(pnl_pct, 2),
                    'Reason': exit_reason,
                    'Entry_Date': pos['entry_date'].strftime('%Y-%m-%d'),
                    'Exit_Date': date.strftime('%Y-%m-%d'),
                    'Sector': pos['sector'],
                    'Holding_Days': (date - pos['entry_date']).days,
                })
                to_exit.append(t)
        
        for t in to_exit:
            del self.positions[t]
    
    def scan_and_buy(self, data_cache, nifty_df, date, regime):
        """Scan universe for new entries."""
        if len(self.positions) >= MAX_POSITIONS:
            return
        
        nifty_idx = nifty_df.index.searchsorted(date)
        if nifty_idx < 252:
            return
        nifty_window = nifty_df.iloc[max(0, nifty_idx - 252):nifty_idx + 1]
        
        candidates = []
        
        for t in data_cache:
            if t == 'NIFTY' or t in self.positions:
                continue
            
            df = data_cache[t]
            idx = df.index.searchsorted(date)
            if idx < 200:
                continue
            
            window = df.iloc[max(0, idx - 252):idx + 1]
            if len(window) < 200:
                continue
            
            ind = calculate_indicators(window, nifty_window)
            if ind is None:
                continue
            
            if self.config['filter'](ind):
                candidates.append({'ticker': t, 'ind': ind})
        
        # Rank
        candidates.sort(key=lambda x: self.config['rank_key'](x['ind']))
        
        # Sector cap filter
        selected = []
        sector_count = {}
        for c in candidates:
            sec = SECTOR_MAP.get(c['ticker'], 'Unknown')
            curr_in_pos = sum(1 for t in self.positions if self.positions[t]['sector'] == sec)
            if sector_count.get(sec, 0) + curr_in_pos < SECTOR_CAP:
                selected.append(c)
                sector_count[sec] = sector_count.get(sec, 0) + 1
            if len(selected) + len(self.positions) >= MAX_POSITIONS:
                break
        
        # Buy
        free_slots = MAX_POSITIONS - len(self.positions)
        for c in selected[:free_slots]:
            price = c['ind']['price']
            size = self.capital / (free_slots + 1)  # Conservative sizing
            shares = int(size / price)
            if shares < 1:
                continue
            cost = shares * price * (1 + COST_BPS / 10000)
            if self.capital >= cost and cost > 5000:
                self.capital -= cost
                self.positions[c['ticker']] = {
                    'entry': price,
                    'peak': price,
                    'shares': shares,
                    'entry_date': date,
                    'sector': SECTOR_MAP.get(c['ticker'], 'Unknown'),
                }
                self.trade_log.append({
                    'Ticker': c['ticker'],
                    'Strategy': self.name,
                    'Action': 'BUY',
                    'Entry_Price': price,
                    'Exit_Price': 0,
                    'Shares': shares,
                    'PnL%': 0,
                    'Reason': f'Entry ({regime})',
                    'Entry_Date': date.strftime('%Y-%m-%d'),
                    'Exit_Date': '',
                    'Sector': SECTOR_MAP.get(c['ticker'], 'Unknown'),
                    'Holding_Days': 0,
                })
    
    def get_equity(self, data_cache, date):
        """Calculate total portfolio value."""
        val = self.capital
        for t, pos in self.positions.items():
            if t in data_cache:
                df = data_cache[t]
                idx = df.index.searchsorted(date)
                if idx > 0:
                    p = df['Close'].iloc[min(idx, len(df)-1)]
                    val += pos['shares'] * p
        return val


# ============================================================
# MAIN BACKTEST
# ============================================================
def fetch_all_data(years):
    """Bulk download Nifty + 500 stocks."""
    buffer_days = 500  # Extra lookback for indicators
    start = (datetime.now() - timedelta(days=365 * years + buffer_days)).strftime('%Y-%m-%d')
    
    print(f"[1/3] Fetching Nifty 50 index...")
    nifty = yf.Ticker("^NSEI").history(start=start)
    if nifty.empty:
        print("ERROR: Could not fetch Nifty data!")
        return None, {}
    nifty.index = nifty.index.tz_localize(None)
    
    print(f"[2/3] Bulk downloading {len(TICKERS[:500])} stocks (this takes 2-4 minutes)...")
    t0 = time.time()
    try:
        bulk = yf.download(
            TICKERS[:500],
            start=start,
            group_by='ticker',
            threads=True,
            progress=True,
            auto_adjust=True,
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
    
    elapsed = time.time() - t0
    print(f"[3/3] Loaded {loaded} stocks in {elapsed:.0f}s")
    return nifty, data_cache


def run_backtest():
    """Main backtest loop."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    nifty, data_cache = fetch_all_data(BACKTEST_YEARS)
    if nifty is None:
        return
    
    # Determine backtest date range
    bt_start = datetime.now() - timedelta(days=365 * BACKTEST_YEARS)
    start_idx = nifty.index.searchsorted(bt_start)
    dates = nifty.index[start_idx:]
    
    print(f"\nBacktest: {dates[0].date()} to {dates[-1].date()} ({len(dates)} trading days)")
    
    # Initialize engines
    engines = {name: StrategyEngine(name) for name in STRATEGY_CONFIG}
    
    # Track regimes
    regime_log = []
    day_counter = 0
    
    for i, date in enumerate(dates):
        regime = detect_regime(nifty, date)
        regime_log.append({'date': date, 'regime': regime})
        
        for name, engine in engines.items():
            # Always check exits
            engine.check_exits(data_cache, date)
            
            # Scan for entries every REBALANCE_FREQ days
            if day_counter % REBALANCE_FREQ == 0:
                engine.scan_and_buy(data_cache, nifty, date, regime)
            
            # Record equity
            eq = engine.get_equity(data_cache, date)
            engine.equity_curve.append({
                'Date': date,
                'Strategy': name,
                'Equity': eq,
                'Regime': regime,
                'Holdings': len(engine.positions),
                'Cash': engine.capital,
            })
        
        day_counter += 1
        
        # Progress
        if (i + 1) % 250 == 0:
            print(f"  Processed {i+1}/{len(dates)} days... ({date.date()})")
    
    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("BEAR REGIME ALPHA BACKTEST — RESULTS")
    print("=" * 80)
    
    # Regime Summary
    regime_df = pd.DataFrame(regime_log)
    regime_counts = regime_df['regime'].value_counts()
    print("\nREGIME DISTRIBUTION:")
    for r in ['BULL', 'MILD_BULL', 'SIDEWAYS', 'BEAR']:
        cnt = regime_counts.get(r, 0)
        pct = cnt / len(regime_df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {r:12s}: {cnt:4d} days ({pct:5.1f}%) {bar}")
    
    # Nifty benchmark
    n_start = nifty.iloc[start_idx]['Close']
    n_end = nifty.iloc[-1]['Close']
    n_ret = (n_end / n_start - 1) * 100
    n_years = (dates[-1] - dates[0]).days / 365.25
    n_cagr = ((n_end / n_start) ** (1 / n_years) - 1) * 100
    
    print(f"\nNIFTY BENCHMARK:")
    print(f"  Total Return: {n_ret:+.1f}%  |  CAGR: {n_cagr:+.1f}%")
    
    # Per-Strategy Summary
    all_results = []
    all_trades = []
    all_equity = []
    
    for name, engine in engines.items():
        eq_df = pd.DataFrame(engine.equity_curve)
        all_equity.append(eq_df)
        
        start_eq = eq_df['Equity'].iloc[0]
        end_eq = eq_df['Equity'].iloc[-1]
        total_ret = (end_eq / start_eq - 1) * 100
        cagr = ((end_eq / start_eq) ** (1 / n_years) - 1) * 100
        
        # Max Drawdown
        eq_df['Peak'] = eq_df['Equity'].cummax()
        eq_df['DD'] = (eq_df['Equity'] - eq_df['Peak']) / eq_df['Peak'] * 100
        max_dd = eq_df['DD'].min()
        
        # Trade stats
        sells = [t for t in engine.trade_log if t['Action'] == 'SELL']
        wins = [t for t in sells if t['PnL%'] > 0]
        losses = [t for t in sells if t['PnL%'] <= 0]
        win_rate = len(wins) / len(sells) * 100 if sells else 0
        avg_win = np.mean([t['PnL%'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['PnL%'] for t in losses]) if losses else 0
        avg_hold = np.mean([t['Holding_Days'] for t in sells]) if sells else 0
        
        print(f"\n{'─' * 60}")
        print(f"{name}")
        print(f"{'─' * 60}")
        print(f"  Final Equity : ₹{end_eq:,.0f}")
        print(f"  Total Return : {total_ret:+.1f}%")
        print(f"  CAGR         : {cagr:+.1f}%")
        print(f"  Max Drawdown : {max_dd:.1f}%")
        print(f"  Alpha vs Nifty: {cagr - n_cagr:+.1f}% (CAGR)")
        print(f"  Trades       : {len(sells)}")
        print(f"  Win Rate     : {win_rate:.0f}%")
        print(f"  Avg Win      : {avg_win:+.1f}%  |  Avg Loss: {avg_loss:+.1f}%")
        print(f"  Avg Hold     : {avg_hold:.0f} days")
        
        # Per-Regime Performance
        print(f"  \n  Per-Regime Returns:")
        for regime in ['BULL', 'MILD_BULL', 'SIDEWAYS', 'BEAR']:
            regime_rows = eq_df[eq_df['Regime'] == regime]
            if len(regime_rows) < 2:
                print(f"    {regime:12s}: N/A")
                continue
            
            # Calculate regime-specific return: sum of daily returns during that regime
            regime_eq = regime_rows['Equity'].values
            daily_rets = np.diff(regime_eq) / regime_eq[:-1]
            
            # Compound these returns
            regime_total = (np.prod(1 + daily_rets) - 1) * 100
            regime_days = len(regime_rows)
            
            # Annualize: compound to yearly rate
            if regime_days > 20:
                regime_ann = ((1 + regime_total/100) ** (252 / regime_days) - 1) * 100
            else:
                regime_ann = regime_total  # Too few days to annualize
            
            emoji = "🟢" if regime_total > 0 else "🔴"
            print(f"    {regime:12s}: {regime_total:+6.1f}% total ({regime_days} days) → {regime_ann:+.1f}% ann. {emoji}")
        
        # Collect
        all_results.append({
            'Strategy': name,
            'Final_Equity': end_eq,
            'Total_Return%': round(total_ret, 2),
            'CAGR%': round(cagr, 2),
            'Max_Drawdown%': round(max_dd, 2),
            'Alpha_CAGR%': round(cagr - n_cagr, 2),
            'Trades': len(sells),
            'Win_Rate%': round(win_rate, 1),
            'Avg_Win%': round(avg_win, 1),
            'Avg_Loss%': round(avg_loss, 1),
            'Avg_Hold_Days': round(avg_hold, 0),
        })
        
        all_trades.extend(engine.trade_log)
    
    # --------------------------------------------------------
    # WINNER DNA ANALYSIS (Bear period winners)
    # --------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("🧬 WINNER DNA: Stocks That Thrived in BEAR Regimes")
    print(f"{'=' * 80}")
    
    # Find trades that were entered during BEAR regime and had positive PnL
    bear_winners = [t for t in all_trades 
                    if t['Action'] == 'SELL' 
                    and t['PnL%'] > 5 
                    and 'BEAR' in t.get('Reason', '')]
    
    # Also check by matching entry dates to regime
    for t in all_trades:
        if t['Action'] == 'SELL' and t['PnL%'] > 5:
            entry_date = pd.Timestamp(t['Entry_Date'])
            matching = regime_df[regime_df['date'] == entry_date]
            if not matching.empty and matching.iloc[0]['regime'] in ('BEAR', 'SIDEWAYS'):
                if t not in bear_winners:
                    bear_winners.append(t)
    
    if bear_winners:
        print(f"\n  Found {len(bear_winners)} winning trades during BEAR/SIDEWAYS:")
        # Sector breakdown
        sectors = {}
        for t in bear_winners:
            s = t.get('Sector', 'Unknown')
            if s not in sectors:
                sectors[s] = {'count': 0, 'total_pnl': 0}
            sectors[s]['count'] += 1
            sectors[s]['total_pnl'] += t['PnL%']
        
        print(f"\n  {'Sector':<35s} {'Wins':>5s} {'Avg PnL%':>8s}")
        print(f"  {'─'*50}")
        for s, v in sorted(sectors.items(), key=lambda x: -x[1]['count']):
            avg = v['total_pnl'] / v['count']
            print(f"  {s:<35s} {v['count']:>5d} {avg:>+7.1f}%")
        
        # Strategy breakdown
        strat_wins = {}
        for t in bear_winners:
            s = t['Strategy']
            if s not in strat_wins:
                strat_wins[s] = 0
            strat_wins[s] += 1
        
        print(f"\n  Strategy Breakdown:")
        for s, c in sorted(strat_wins.items(), key=lambda x: -x[1]):
            print(f"    {s}: {c} winning trades")
        
        # Top individual winners
        bear_winners.sort(key=lambda x: -x['PnL%'])
        print(f"\n  Top 10 Individual Winners:")
        print(f"  {'Ticker':<18s} {'Strategy':<15s} {'PnL%':>7s} {'Hold':>5s} {'Sector':<25s}")
        print(f"  {'─'*72}")
        for t in bear_winners[:10]:
            print(f"  {t['Ticker']:<18s} {t['Strategy']:<15s} {t['PnL%']:>+6.1f}% {t['Holding_Days']:>4d}d {t['Sector']:<25s}")
    else:
        print("  No trades with >5% PnL found in bear/sideways regimes.")
    
    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------
    pd.DataFrame(all_results).to_csv(RESULTS_FILE, index=False)
    pd.DataFrame(all_trades).to_csv(TRADES_FILE, index=False)
    
    # Combined equity curve
    combined_eq = pd.concat(all_equity)
    combined_eq.to_csv(EQUITY_FILE, index=False)
    
    # Regime log
    regime_df.to_csv(f"{OUTPUT_DIR}/bear_regime_log.csv", index=False)
    
    print(f"\nSaved:")
    print(f"   {RESULTS_FILE}")
    print(f"   {TRADES_FILE}")
    print(f"   {EQUITY_FILE}")
    print(f"   {OUTPUT_DIR}/bear_regime_log.csv")
    
    # --------------------------------------------------------
    # KEY INSIGHTS
    # --------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("KEY INSIGHTS")
    print(f"{'=' * 80}")
    
    best = max(all_results, key=lambda x: x['CAGR%'])
    worst = min(all_results, key=lambda x: x['CAGR%'])
    
    print(f"\n  🏆 Best Overall Strategy : {best['Strategy']} ({best['CAGR%']:+.1f}% CAGR, {best['Alpha_CAGR%']:+.1f}% alpha)")
    print(f"  📉 Worst Strategy       : {worst['Strategy']} ({worst['CAGR%']:+.1f}% CAGR)")
    
    bear_target = 15
    for r in all_results:
        emoji = "✅" if r['CAGR%'] >= bear_target else "❌"
        print(f"  {emoji} {r['Strategy']}: CAGR {r['CAGR%']:+.1f}% vs {bear_target}% target")
    
    print(f"\n  📌 Current Regime: {detect_regime(nifty, dates[-1])}")
    print(f"  📌 Recommendation: Use {best['Strategy']} in current conditions")
    
    return all_results, all_trades


# ============================================================
if __name__ == "__main__":
    print("=" * 80)
    print("BEAR REGIME ALPHA BACKTEST")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    run_backtest()
