"""
ATR EFFECT STUDY - OptComp-V21
===============================
Tests ATR-based trailing stops vs fixed-percentage stops.

Fixed parameters (from research):
  - Composite RS (10% 1W + 50% 1M + 40% 3M)
  - 13-day rebalance
  - 10 positions, equal weight
  - Price > MA50 entry, Price < MA50 exit (trend break always active)

Variants tested:
  1. Fixed-15% (baseline)      : Peak * 0.85
  2. ATR-2x                    : Peak - 2*ATR(14)
  3. ATR-3x                    : Peak - 3*ATR(14)
  4. ATR-4x                    : Peak - 4*ATR(14)
  5. ATR-2x + ATR Sizing       : ATR stop + vol-adjusted sizing
  6. ATR-3x + ATR Sizing       : ATR stop + vol-adjusted sizing
  7. No Trailing               : MA50 exit only
  8. Keltner Exit              : Price < MA20 - 2*ATR(14)

Usage:
  python dna3_atr_study.py
"""

from dna3_ultimate_comparison import fetch_data
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nifty500_list import TICKERS, SECTOR_MAP

OUTPUT_DIR = "analysis_2026"
INITIAL_CAPITAL = 1000000

# Composite RS weights
RS_WEIGHTS = [(5, 0.10), (21, 0.50), (63, 0.40)]
REBALANCE_DAYS = 13
MAX_POS = 10

HORIZONS = {
    '1Y': 1, '3Y': 3, '5Y': 5, '10Y': 10, '15Y': 15
}

# Variant definitions
VARIANTS = {
    'Fixed-15%':      {'stop_type': 'fixed',   'stop_pct': 0.85,  'sizing': 'equal'},
    'ATR-2x':         {'stop_type': 'atr',     'atr_mult': 2.0,   'sizing': 'equal'},
    'ATR-3x':         {'stop_type': 'atr',     'atr_mult': 3.0,   'sizing': 'equal'},
    'ATR-4x':         {'stop_type': 'atr',     'atr_mult': 4.0,   'sizing': 'equal'},
    'ATR2x+Size':     {'stop_type': 'atr',     'atr_mult': 2.0,   'sizing': 'atr'},
    'ATR3x+Size':     {'stop_type': 'atr',     'atr_mult': 3.0,   'sizing': 'atr'},
    'NoTrail':        {'stop_type': 'none',    'sizing': 'equal'},
    'Keltner':        {'stop_type': 'keltner', 'sizing': 'equal'},
}


# ============================================================
# ATR CALCULATION
# ============================================================
def calc_atr(df, idx, period=14):
    """Calculate ATR at a given index in the dataframe."""
    if idx < period + 1:
        return None
    
    window = df.iloc[max(0, idx - period - 1):idx + 1]
    if len(window) < period + 1:
        return None
    
    high = window['High'].values
    low = window['Low'].values
    close = window['Close'].values
    
    tr_values = []
    for i in range(1, len(window)):
        tr = max(
            high[i] - low[i],                    # Current H-L
            abs(high[i] - close[i - 1]),          # H - prev Close
            abs(low[i] - close[i - 1]),           # L - prev Close
        )
        tr_values.append(tr)
    
    if len(tr_values) < period:
        return None
    
    # Use simple average for ATR
    return np.mean(tr_values[-period:])


def calc_keltner_lower(df, idx, atr_mult=2.0, ma_period=20, atr_period=14):
    """Calculate lower Keltner Channel band: MA20 - 2*ATR(14)."""
    if idx < max(ma_period, atr_period + 1):
        return None
    
    ma20 = df['Close'].iloc[max(0, idx - ma_period + 1):idx + 1].mean()
    atr = calc_atr(df, idx, atr_period)
    if atr is None:
        return None
    
    return ma20 - atr_mult * atr


# ============================================================
# COMPOSITE RS
# ============================================================
def composite_rs(dc, t, d, nifty):
    """Composite RS at date d."""
    i = dc[t].index.searchsorted(d)
    ni = nifty.index.searchsorted(d)
    if i < 64 or ni < 64:
        return None
    if i >= len(dc[t]) or ni >= len(nifty):
        return None
    
    price = dc[t]['Close'].iloc[i]
    n_price = nifty['Close'].iloc[ni]
    
    total = 0.0
    for period, weight in RS_WEIGHTS:
        if i < period or ni < period:
            return None
        sp = dc[t]['Close'].iloc[i - period]
        np_ = nifty['Close'].iloc[ni - period]
        if sp == 0 or np_ == 0:
            return None
        rs = ((price / sp - 1) - (n_price / np_ - 1)) * 100
        total += rs * weight
    return total


# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest(dc, nifty, start, end, config):
    """Run single backtest with given ATR/stop configuration."""
    cash = float(INITIAL_CAPITAL)
    positions = {}  # {ticker: {shares, entry, peak, entry_date, atr_at_entry}}
    trades = []
    equity_curve = []
    last_reb = None
    
    dates = nifty.index[(nifty.index >= start) & (nifty.index <= end)]
    
    stop_type = config['stop_type']
    sizing_type = config.get('sizing', 'equal')
    atr_mult = config.get('atr_mult', 3.0)
    stop_pct = config.get('stop_pct', 0.85)
    
    for d in dates:
        # 1. CHECK EXITS (every day)
        to_sell = []
        for t, pos in positions.items():
            if t not in dc:
                continue
            ti = dc[t].index.searchsorted(d)
            if ti >= len(dc[t]):
                continue
            
            price = dc[t]['Close'].iloc[ti]
            ma50 = dc[t]['Close'].iloc[max(0, ti - 50):ti + 1].mean()
            
            # Update peak
            if price > pos['peak']:
                pos['peak'] = price
            
            exit_reason = None
            
            # MA50 exit (always active regardless of variant)
            if price < ma50:
                exit_reason = 'MA50'
            else:
                # Trailing stop logic depends on variant
                if stop_type == 'fixed':
                    if price < pos['peak'] * stop_pct:
                        exit_reason = f'TSL({(1 - stop_pct) * 100:.0f}%)'
                
                elif stop_type == 'atr':
                    atr = calc_atr(dc[t], ti)
                    if atr is not None:
                        atr_stop = pos['peak'] - atr_mult * atr
                        if price < atr_stop:
                            exit_reason = f'ATR({atr_mult:.0f}x)'
                
                elif stop_type == 'keltner':
                    kelt_lower = calc_keltner_lower(dc[t], ti)
                    if kelt_lower is not None and price < kelt_lower:
                        exit_reason = 'Keltner'
                
                # 'none' = no trailing stop, MA50 only
            
            if exit_reason:
                proceeds = pos['shares'] * price * 0.998  # Transaction cost
                pnl_pct = (price - pos['entry']) / pos['entry'] * 100
                cash += proceeds
                to_sell.append(t)
                trades.append({
                    'exit_reason': exit_reason,
                    'pnl_pct': pnl_pct,
                    'hold_days': (d - pos['entry_date']).days,
                })
        
        for t in to_sell:
            del positions[t]
        
        # 2. REBALANCE CHECK
        is_reb = False
        if last_reb is None:
            is_reb = True
        else:
            days_since = len(nifty.index[(nifty.index > last_reb) & (nifty.index <= d)])
            if days_since >= REBALANCE_DAYS:
                is_reb = True
        
        if is_reb:
            last_reb = d
            
            # Scan for candidates
            candidates = []
            for t in dc:
                if t == 'NIFTY' or t in positions:
                    continue
                ti = dc[t].index.searchsorted(d)
                if ti < 64 or ti >= len(dc[t]):
                    continue
                
                price = dc[t]['Close'].iloc[ti]
                ma50 = dc[t]['Close'].iloc[max(0, ti - 50):ti + 1].mean()
                vol = dc[t]['Volume'].iloc[max(0, ti - 20):ti + 1].mean() * price
                
                if price > ma50 and vol > 10_000_000:
                    rs = composite_rs(dc, t, d, nifty)
                    if rs and rs > 0:
                        atr = calc_atr(dc[t], ti)
                        candidates.append({
                            't': t, 'rs': rs, 'price': price,
                            'atr': atr if atr else price * 0.02,  # fallback
                        })
            
            candidates.sort(key=lambda x: -x['rs'])
            
            # Buy
            free = MAX_POS - len(positions)
            if free > 0 and candidates:
                eq = cash + sum(
                    pos['shares'] * dc[t]['Close'].iloc[min(dc[t].index.searchsorted(d), len(dc[t]) - 1)]
                    for t, pos in positions.items() if t in dc
                )
                
                for c in candidates[:free]:
                    if sizing_type == 'atr':
                        # ATR-based position sizing: risk per trade = equity / (2 * max_pos)
                        risk_per_trade = eq / (2 * MAX_POS)
                        atr_risk = atr_mult * c['atr']
                        if atr_risk > 0:
                            shares = int(risk_per_trade / atr_risk)
                        else:
                            shares = int((eq / MAX_POS) / c['price'])
                        # Cap at equal-weight equivalent
                        max_shares = int((eq / MAX_POS) / c['price'])
                        shares = min(shares, max_shares)
                    else:
                        # Equal-weight sizing
                        target = eq / MAX_POS
                        invest = min(target, cash / max(free, 1))
                        shares = int(invest / c['price'])
                    
                    if shares < 1:
                        continue
                    cost = shares * c['price'] * 1.002  # Impact cost
                    if cash >= cost and cost > 5000:
                        cash -= cost
                        positions[c['t']] = {
                            'shares': shares,
                            'entry': c['price'],
                            'peak': c['price'],
                            'entry_date': d,
                            'atr_at_entry': c['atr'],
                        }
                        free -= 1
        
        # 3. Equity
        eq = cash
        for t, pos in positions.items():
            if t in dc:
                ti = min(dc[t].index.searchsorted(d), len(dc[t]) - 1)
                eq += pos['shares'] * dc[t]['Close'].iloc[ti]
        equity_curve.append({'date': d, 'equity': eq})
    
    return equity_curve, trades


# ============================================================
# METRICS
# ============================================================
def calc_metrics(equity_curve, trades, years):
    """Calculate performance metrics."""
    if not equity_curve:
        return {}
    
    edf = pd.DataFrame(equity_curve)
    final = edf['equity'].iloc[-1]
    cagr = ((final / INITIAL_CAPITAL) ** (1 / max(years, 0.25)) - 1) * 100
    
    # Drawdown
    peak = edf['equity'].cummax()
    dd = (edf['equity'] - peak) / peak * 100
    max_dd = dd.min()
    
    # Sharpe (daily)
    rets = edf['equity'].pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    
    # Sortino
    neg = rets[rets < 0]
    sortino = (rets.mean() / neg.std() * np.sqrt(252)) if len(neg) > 0 and neg.std() > 0 else 0
    
    # Calmar
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0
    
    # Trade stats
    n_trades = len(trades)
    if n_trades > 0:
        wins = sum(1 for t in trades if t['pnl_pct'] > 0)
        win_rate = wins / n_trades * 100
        avg_pnl = np.mean([t['pnl_pct'] for t in trades])
        avg_hold = np.mean([t['hold_days'] for t in trades])
        
        # Exit breakdown
        ma50_trades = [t for t in trades if t['exit_reason'] == 'MA50']
        trail_trades = [t for t in trades if t['exit_reason'] != 'MA50']
        
        ma50_exits = len(ma50_trades)
        trail_exits = len(trail_trades)
        ma50_avg = np.mean([t['pnl_pct'] for t in ma50_trades]) if ma50_trades else 0
        trail_avg = np.mean([t['pnl_pct'] for t in trail_trades]) if trail_trades else 0
        
        # Avg win / avg loss
        win_trades = [t for t in trades if t['pnl_pct'] > 0]
        loss_trades = [t for t in trades if t['pnl_pct'] <= 0]
        avg_win = np.mean([t['pnl_pct'] for t in win_trades]) if win_trades else 0
        avg_loss = np.mean([t['pnl_pct'] for t in loss_trades]) if loss_trades else 0
    else:
        win_rate = avg_pnl = ma50_exits = trail_exits = 0
        ma50_avg = trail_avg = avg_hold = avg_win = avg_loss = 0
    
    return {
        'CAGR%': round(cagr, 1),
        'MaxDD%': round(max_dd, 1),
        'Sharpe': round(sharpe, 2),
        'Sortino': round(sortino, 2),
        'Calmar': round(calmar, 2),
        'Trades': n_trades,
        'WinRate%': round(win_rate, 1),
        'AvgPnL%': round(avg_pnl, 1),
        'AvgWin%': round(avg_win, 1),
        'AvgLoss%': round(avg_loss, 1),
        'AvgHold': round(avg_hold),
        'MA50_Exits': ma50_exits,
        'Trail_Exits': trail_exits,
        'MA50_AvgPnL%': round(ma50_avg, 1),
        'Trail_AvgPnL%': round(trail_avg, 1),
    }


# ============================================================
# MAIN
# ============================================================
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nifty, dc = fetch_data()
    if nifty is None:
        return

    print("\n" + "=" * 150)
    print("ATR EFFECT STUDY - OptComp-V21")
    print("=" * 150)
    print(f"  Testing: {', '.join(VARIANTS.keys())}")
    print(f"  Across:  {', '.join(HORIZONS.keys())}")

    all_results = []

    for hz_name, hz_years in HORIZONS.items():
        end = nifty.index[-1]
        start = end - pd.DateOffset(years=hz_years)
        valid = nifty.index[nifty.index >= start]
        if len(valid) < 20:
            print(f"\n  [{hz_name}] Insufficient data, skipping.")
            continue
        start = valid[0]
        actual_years = (end - start).days / 365.25

        print(f"\n{'_' * 150}")
        print(f"  {hz_name} ({start.strftime('%Y-%m-%d')} -> {end.strftime('%Y-%m-%d')}, {actual_years:.1f}y)")
        print(f"{'_' * 150}")
        
        # Nifty benchmark
        ns = nifty.loc[start:end, 'Close']
        n_total = (ns.iloc[-1] / ns.iloc[0] - 1) * 100
        n_cagr = ((ns.iloc[-1] / ns.iloc[0]) ** (1 / actual_years) - 1) * 100
        
        # Print header
        metrics_order = ['CAGR%', 'MaxDD%', 'Sharpe', 'Calmar', 'WinRate%', 'Trades', 'AvgHold',
                         'MA50_Exits', 'Trail_Exits', 'MA50_AvgPnL%', 'Trail_AvgPnL%']
        header = f"  {'Variant':<14}"
        for m in metrics_order:
            header += f" {m:>12}"
        print(header)
        print(f"  {'-' * 150}")

        for var_name, var_config in VARIANTS.items():
            eq, trades = run_backtest(dc, nifty, start, end, var_config)
            metrics = calc_metrics(eq, trades, actual_years)
            
            row = f"  {var_name:<14}"
            for m in metrics_order:
                v = metrics.get(m, 0)
                if isinstance(v, float):
                    row += f" {v:>12.1f}"
                else:
                    row += f" {v:>12}"
            print(row)
            
            all_results.append({
                'Horizon': hz_name,
                'Years': round(actual_years, 1),
                'Variant': var_name,
                'Nifty_CAGR%': round(n_cagr, 1),
                **metrics,
            })

    # ================================
    # SUMMARY TABLE
    # ================================
    rdf = pd.DataFrame(all_results)
    rdf.to_csv(f"{OUTPUT_DIR}/atr_study_results.csv", index=False)

    print(f"\n\n{'=' * 150}")
    print("SUMMARY: BEST VARIANT PER HORIZON")
    print(f"{'=' * 150}")
    
    print(f"\n  {'Horizon':<10} {'Best CAGR':<16} {'Best Sharpe':<16} {'Best MaxDD':<16} {'Best Calmar':<16}")
    print(f"  {'-' * 75}")
    
    for hz in HORIZONS:
        sub = rdf[rdf['Horizon'] == hz]
        if sub.empty:
            continue
        best_cagr = sub.loc[sub['CAGR%'].idxmax(), 'Variant']
        best_sharpe = sub.loc[sub['Sharpe'].idxmax(), 'Variant']
        best_dd = sub.loc[sub['MaxDD%'].idxmax(), 'Variant']  # Least negative
        best_calmar = sub.loc[sub['Calmar'].idxmax(), 'Variant']
        print(f"  {hz:<10} {best_cagr:<16} {best_sharpe:<16} {best_dd:<16} {best_calmar:<16}")

    # ================================
    # ATR vs FIXED COMPARISON (longest horizon)
    # ================================
    longest = list(HORIZONS.keys())[-1]
    sub = rdf[rdf['Horizon'] == longest]
    
    if not sub.empty:
        print(f"\n\n{'=' * 150}")
        print(f"DETAILED COMPARISON ({longest})")
        print(f"{'=' * 150}")
        
        baseline = sub[sub['Variant'] == 'Fixed-15%']
        if not baseline.empty:
            b_cagr = baseline.iloc[0]['CAGR%']
            b_sharpe = baseline.iloc[0]['Sharpe']
            b_dd = baseline.iloc[0]['MaxDD%']
            
            print(f"\n  {'Variant':<14} {'CAGR%':>8} {'dCAGR':>8} {'MaxDD%':>8} {'Sharpe':>8} {'dSharpe':>10} {'WinRate%':>10} {'AvgWin%':>9} {'AvgLoss%':>10} {'AvgHold':>8}")
            print(f"  {'-' * 105}")
            
            for _, row in sub.iterrows():
                delta_cagr = row['CAGR%'] - b_cagr
                delta_sharpe = row['Sharpe'] - b_sharpe
                marker = " *" if row['Variant'] == 'Fixed-15%' else ""
                print(f"  {row['Variant']:<14} {row['CAGR%']:>+7.1f}% {delta_cagr:>+7.1f}% {row['MaxDD%']:>+7.1f}% {row['Sharpe']:>8.2f} {delta_sharpe:>+9.2f} {row['WinRate%']:>9.1f}% {row['AvgWin%']:>+8.1f}% {row['AvgLoss%']:>+9.1f}% {row['AvgHold']:>7.0f}d{marker}")

        # Exit breakdown
        print(f"\n  EXIT RULE ANALYSIS ({longest}):")
        print(f"  {'Variant':<14} {'MA50_Exits':>12} {'Trail_Exits':>12} {'MA50_Avg%':>12} {'Trail_Avg%':>12} {'Verdict':<30}")
        print(f"  {'-' * 100}")
        
        for _, row in sub.iterrows():
            total = row['MA50_Exits'] + row['Trail_Exits']
            if total > 0:
                if row['Trail_Exits'] == 0:
                    verdict = "MA50 only (no trailing)"
                elif abs(row['MA50_AvgPnL%']) > abs(row['Trail_AvgPnL%']):
                    verdict = f"MA50 hurts more ({row['MA50_AvgPnL%']:+.1f}%)"
                else:
                    verdict = f"Trail hurts more ({row['Trail_AvgPnL%']:+.1f}%)"
            else:
                verdict = "No trades"
            
            print(f"  {row['Variant']:<14} {row['MA50_Exits']:>12} {row['Trail_Exits']:>12} {row['MA50_AvgPnL%']:>+11.1f}% {row['Trail_AvgPnL%']:>+11.1f}% {verdict:<30}")

    # ================================
    # VERDICT
    # ================================
    print(f"\n\n{'=' * 150}")
    print("VERDICT")
    print(f"{'=' * 150}")
    
    # Count how many horizons each variant wins on CAGR
    cagr_wins = {}
    sharpe_wins = {}
    for hz in HORIZONS:
        sub = rdf[rdf['Horizon'] == hz]
        if sub.empty:
            continue
        best_c = sub.loc[sub['CAGR%'].idxmax(), 'Variant']
        best_s = sub.loc[sub['Sharpe'].idxmax(), 'Variant']
        cagr_wins[best_c] = cagr_wins.get(best_c, 0) + 1
        sharpe_wins[best_s] = sharpe_wins.get(best_s, 0) + 1
    
    print(f"\n  CAGR wins across {len(HORIZONS)} horizons:")
    for v in VARIANTS:
        cnt = cagr_wins.get(v, 0)
        bar = '#' * cnt
        print(f"    {v:<14}: {cnt} {bar}")
    
    print(f"\n  Sharpe wins across {len(HORIZONS)} horizons:")
    for v in VARIANTS:
        cnt = sharpe_wins.get(v, 0)
        bar = '#' * cnt
        print(f"    {v:<14}: {cnt} {bar}")
    
    # Overall recommendation
    overall_cagr_winner = max(cagr_wins, key=cagr_wins.get) if cagr_wins else 'N/A'
    overall_sharpe_winner = max(sharpe_wins, key=sharpe_wins.get) if sharpe_wins else 'N/A'
    
    print(f"\n  OVERALL:")
    print(f"    Best for returns:     {overall_cagr_winner}")
    print(f"    Best risk-adjusted:   {overall_sharpe_winner}")
    
    print(f"\n  Results saved to {OUTPUT_DIR}/atr_study_results.csv")
    print(f"{'=' * 150}")


if __name__ == "__main__":
    print("=" * 150)
    print("ATR EFFECT STUDY - OptComp-V21")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 150)
    run()
