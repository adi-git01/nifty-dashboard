"""
TRAILING STOP SENSITIVITY STUDY — OptComp-V21
==============================================
Tests trailing stop levels: 8%, 10%, 12%, 15%, 20%, 25%
Also tests: flat stop vs cyclicity-adjusted stops

Fixed parameters (from research):
  - Composite RS (10% 1W + 50% 1M + 40% 3M)
  - 13-day rebalance
  - 10 positions, equal weight
  - Price > MA50 entry, Price < MA50 exit (trend break always active)

Plus: Cyclicity-Adjusted variant:
  - Long-cycle sectors (metals, auto, capgoods): -20%
  - Short-cycle sectors (textiles, consumer, NBFC): -8%
  - Mid-cycle: -12%
"""

from dna3_ultimate_comparison import fetch_data
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

OUTPUT_DIR = "analysis_2026"
INITIAL_CAPITAL = 1000000

# Composite RS weights
RS_WEIGHTS = [(5, 0.10), (21, 0.50), (63, 0.40)]
REBALANCE_DAYS = 13
MAX_POS = 10

# Cyclicity keywords (from live_desk.py)
LONG_CYCLE_KWS = ['auto', 'farm', 'heavy', 'metal', 'medical', 'machinery', 'agricultural', 
                  'electric', 'aluminum', 'coal', 'defense', 'aerospace', 'cement', 'power', 
                  'infrastructure', 'equipment', 'engineering', 'construction', 'minerals',
                  'copper', 'steel', 'iron', 'compressor', 'pump', 'diesel', 'engine',
                  'vehicle', 'tyre', 'rubber']
SHORT_CYCLE_KWS = ['insurance', 'capital market', 'chemical', 'tobacco', 'gas', 'tools', 
                   'real estate', 'realty', 'lodging', 'diagnostic', 'broker', 'fmcg', 
                   'consumer', 'retail', 'textile', 'apparel', 'food', 'beverage', 
                   'leisure', 'media', 'entertainment', 'finance', 'bank', 'nbfc']

HORIZONS = {
    '1Y': 1, '3Y': 3, '5Y': 5, '10Y': 10, '15Y': 15
}

# Stop variants to test
STOP_VARIANTS = {
    'TSL-8%':  {'type': 'flat', 'pct': 0.92},
    'TSL-10%': {'type': 'flat', 'pct': 0.90},
    'TSL-12%': {'type': 'flat', 'pct': 0.88},
    'TSL-15%': {'type': 'flat', 'pct': 0.85},
    'TSL-20%': {'type': 'flat', 'pct': 0.80},
    'TSL-25%': {'type': 'flat', 'pct': 0.75},
    'Cyclic':  {'type': 'cyclic'},  # Cyclicity-adjusted
}


def get_sector(dc, t):
    """Get sector string for a ticker from the SECTOR_MAP."""
    try:
        from utils.nifty500_list import SECTOR_MAP
        return SECTOR_MAP.get(t, 'Unknown')
    except:
        return 'Unknown'


def get_cyclicity_stop(sector_str):
    """Return trailing stop factor based on sector cyclicity."""
    s = str(sector_str).lower()
    for kw in LONG_CYCLE_KWS:
        if kw in s:
            return 0.80  # -20%
    for kw in SHORT_CYCLE_KWS:
        if kw in s:
            return 0.92  # -8%
    return 0.88  # -12% (mid)


def composite_rs(dc, t, d, nifty):
    """Composite RS at date d."""
    i = dc[t].index.searchsorted(d)
    ni = nifty.index.searchsorted(d)
    if i < 64 or ni < 64:
        return None
    
    price = dc[t]['Close'].iloc[i]
    n_price = nifty['Close'].iloc[ni]
    
    total = 0.0
    for period, weight in RS_WEIGHTS:
        if i < period or ni < period:
            return None
        sp = dc[t]['Close'].iloc[i - period]
        np_ = nifty['Close'].iloc[ni - period]
        rs = ((price / sp - 1) - (n_price / np_ - 1)) * 100
        total += rs * weight
    return total


def run_backtest(dc, nifty, start, end, stop_config):
    """Run single backtest with given stop configuration."""
    capital = INITIAL_CAPITAL
    cash = capital
    positions = {}  # {ticker: {shares, entry, peak, sector}}
    trades = []
    equity_curve = []
    last_reb = None
    
    dates = nifty.index[(nifty.index >= start) & (nifty.index <= end)]
    
    for d in dates:
        di = nifty.index.searchsorted(d)
        
        # 1. CHECK EXITS (every day)
        to_sell = []
        for t, pos in positions.items():
            if t not in dc:
                continue
            ti = dc[t].index.searchsorted(d)
            if ti >= len(dc[t]):
                continue
            
            price = dc[t]['Close'].iloc[ti]
            ma50 = dc[t]['Close'].iloc[max(0, ti-50):ti+1].mean()
            
            if price > pos['peak']:
                pos['peak'] = price
            
            # Determine stop level
            if stop_config['type'] == 'flat':
                stop_factor = stop_config['pct']
            else:  # cyclic
                stop_factor = get_cyclicity_stop(pos['sector'])
            
            exit_reason = None
            if price < ma50:
                exit_reason = 'MA50'
            elif price < pos['peak'] * stop_factor:
                exit_reason = f'TSL({(1-stop_factor)*100:.0f}%)'
            
            if exit_reason:
                proceeds = pos['shares'] * price * 0.998
                pnl_pct = (price - pos['entry']) / pos['entry'] * 100
                cash += proceeds
                to_sell.append(t)
                trades.append({
                    'exit_reason': exit_reason,
                    'pnl_pct': pnl_pct,
                    'hold_days': (d - pos['entry_date']).days if hasattr(pos.get('entry_date', d), 'days') else 0,
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
            
            # Scan
            candidates = []
            for t in dc:
                if t == 'NIFTY' or t in positions:
                    continue
                ti = dc[t].index.searchsorted(d)
                if ti < 64 or ti >= len(dc[t]):
                    continue
                
                price = dc[t]['Close'].iloc[ti]
                ma50 = dc[t]['Close'].iloc[max(0, ti-50):ti+1].mean()
                vol = dc[t]['Volume'].iloc[max(0, ti-20):ti+1].mean() * price
                
                if price > ma50 and vol > 10_000_000:
                    rs = composite_rs(dc, t, d, nifty)
                    if rs and rs > 0:
                        candidates.append({'t': t, 'rs': rs, 'price': price})
            
            candidates.sort(key=lambda x: -x['rs'])
            
            # Buy
            free = MAX_POS - len(positions)
            if free > 0 and candidates:
                eq = cash + sum(pos['shares'] * dc[t]['Close'].iloc[min(dc[t].index.searchsorted(d), len(dc[t])-1)] 
                               for t, pos in positions.items() if t in dc)
                
                for c in candidates[:free]:
                    target = eq / MAX_POS
                    invest = min(target, cash / max(free, 1))
                    if invest > 5000:
                        shares = int(invest / c['price'])
                        cost = shares * c['price'] * 1.002
                        if cash >= cost:
                            cash -= cost
                            positions[c['t']] = {
                                'shares': shares,
                                'entry': c['price'],
                                'peak': c['price'],
                                'sector': get_sector(dc, c['t']),
                                'entry_date': d,
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
    
    # Sharpe
    rets = edf['equity'].pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    
    # Trade stats
    n_trades = len(trades)
    if n_trades > 0:
        wins = sum(1 for t in trades if t['pnl_pct'] > 0)
        win_rate = wins / n_trades * 100
        avg_pnl = np.mean([t['pnl_pct'] for t in trades])
        
        # Exit breakdown
        ma50_exits = sum(1 for t in trades if t['exit_reason'] == 'MA50')
        tsl_exits = n_trades - ma50_exits
        
        # MA50 exit outcomes
        ma50_trades = [t for t in trades if t['exit_reason'] == 'MA50']
        tsl_trades = [t for t in trades if t['exit_reason'] != 'MA50']
        
        ma50_avg = np.mean([t['pnl_pct'] for t in ma50_trades]) if ma50_trades else 0
        tsl_avg = np.mean([t['pnl_pct'] for t in tsl_trades]) if tsl_trades else 0
        
        # Consecutive losses
        consec_loss = 0
        max_consec = 0
        for t in trades:
            if t['pnl_pct'] < 0:
                consec_loss += 1
                max_consec = max(max_consec, consec_loss)
            else:
                consec_loss = 0
    else:
        win_rate = avg_pnl = ma50_exits = tsl_exits = ma50_avg = tsl_avg = max_consec = 0
    
    return {
        'CAGR%': round(cagr, 1),
        'MaxDD%': round(max_dd, 1),
        'Sharpe': round(sharpe, 2),
        'Trades': n_trades,
        'WinRate%': round(win_rate, 1),
        'AvgPnL%': round(avg_pnl, 1),
        'MA50_Exits': ma50_exits,
        'TSL_Exits': tsl_exits,
        'MA50_AvgPnL%': round(ma50_avg, 1),
        'TSL_AvgPnL%': round(tsl_avg, 1),
        'MaxConsecLoss': max_consec,
    }


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nifty, dc = fetch_data()
    if nifty is None:
        return

    print("\n" + "=" * 140)
    print("TRAILING STOP SENSITIVITY STUDY - OptComp-V21 (13d Rebalance)")
    print("=" * 140)
    print(f"  Testing: {', '.join(STOP_VARIANTS.keys())}")
    print(f"  Across:  {', '.join(HORIZONS.keys())}")

    all_results = []

    for hz_name, hz_years in HORIZONS.items():
        end = nifty.index[-1]
        start = end - pd.DateOffset(years=hz_years)
        start = nifty.index[nifty.index >= start][0]
        actual_years = (end - start).days / 365.25

        print(f"\n{'_' * 140}")
        print(f"  {hz_name} ({start.strftime('%Y-%m-%d')} -> {end.strftime('%Y-%m-%d')}, {actual_years:.1f}y)")
        print(f"{'_' * 140}")
        
        header = f"  {'Variant':<12}"
        metrics_order = ['CAGR%', 'MaxDD%', 'Sharpe', 'WinRate%', 'Trades', 'MA50_Exits', 'TSL_Exits', 'MA50_AvgPnL%', 'TSL_AvgPnL%', 'MaxConsecLoss']
        for m in metrics_order:
            header += f" {m:>12}"
        print(header)
        print(f"  {'-' * 135}")

        for var_name, var_config in STOP_VARIANTS.items():
            eq, trades = run_backtest(dc, nifty, start, end, var_config)
            metrics = calc_metrics(eq, trades, actual_years)
            
            row = f"  {var_name:<12}"
            for m in metrics_order:
                v = metrics.get(m, 0)
                if isinstance(v, float):
                    row += f" {v:>12.1f}"
                else:
                    row += f" {v:>12}"
                
            print(row)
            
            all_results.append({
                'Horizon': hz_name,
                'Variant': var_name,
                **metrics,
            })

    # Summary: Best variant per horizon
    rdf = pd.DataFrame(all_results)
    rdf.to_csv(f"{OUTPUT_DIR}/trailing_stop_study.csv", index=False)

    print(f"\n\n{'=' * 140}")
    print("SUMMARY: OPTIMAL TRAILING STOP PER HORIZON")
    print(f"{'=' * 140}")
    
    print(f"\n  {'Horizon':<10} {'Best CAGR':<14} {'Best Sharpe':<14} {'Best MaxDD':<14} {'Best WinRate':<14}")
    print(f"  {'-' * 65}")
    
    for hz in HORIZONS:
        sub = rdf[rdf['Horizon'] == hz]
        best_cagr = sub.loc[sub['CAGR%'].idxmax(), 'Variant']
        best_sharpe = sub.loc[sub['Sharpe'].idxmax(), 'Variant']
        best_dd = sub.loc[sub['MaxDD%'].idxmax(), 'Variant']  # Least negative
        best_wr = sub.loc[sub['WinRate%'].idxmax(), 'Variant']
        print(f"  {hz:<10} {best_cagr:<14} {best_sharpe:<14} {best_dd:<14} {best_wr:<14}")

    # Exit analysis: across all horizons, which exit fires more?
    print(f"\n\n{'=' * 140}")
    print("EXIT RULE BREAKDOWN (15Y Period)")
    print(f"{'=' * 140}")
    
    sub_15y = rdf[rdf['Horizon'] == '15Y']
    print(f"\n  {'Variant':<12} {'MA50_Exits':>12} {'TSL_Exits':>12} {'MA50_Avg':>12} {'TSL_Avg':>12} {'Verdict':<30}")
    print(f"  {'-' * 95}")
    
    for _, row in sub_15y.iterrows():
        total = row['MA50_Exits'] + row['TSL_Exits']
        ma50_pct = row['MA50_Exits'] / total * 100 if total > 0 else 0
        
        # Which exit is worse (more negative avg PnL)?
        if row['MA50_AvgPnL%'] < row['TSL_AvgPnL%']:
            verdict = f"MA50 hurts more ({row['MA50_AvgPnL%']:+.1f}%)"
        else:
            verdict = f"TSL hurts more ({row['TSL_AvgPnL%']:+.1f}%)"
        
        print(f"  {row['Variant']:<12} {row['MA50_Exits']:>12} {row['TSL_Exits']:>12} {row['MA50_AvgPnL%']:>+11.1f}% {row['TSL_AvgPnL%']:>+11.1f}% {verdict:<30}")

    print(f"\n  Results saved to {OUTPUT_DIR}/trailing_stop_study.csv")
    print(f"{'=' * 140}")


if __name__ == "__main__":
    print("=" * 140)
    print("TRAILING STOP SENSITIVITY STUDY")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 140)
    run()
