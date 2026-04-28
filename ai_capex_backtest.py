"""
AI Capex Theme Backtest — RS Weight Comparison
================================================
Tests multiple RS weight configurations against the AI datacenter
supply chain universe to find optimal theme-rotation parameters.

Compares:
  A) Current Indian OptComp weights: (5:10%, 21:50%, 63:40%)
  B) Balanced tweak:                 (5:20%, 21:50%, 63:30%)
  C) Aggressive (recommended):      (5:25%, 21:50%, 63:25%)
  D) Ultra-fast:                     (5:30%, 21:50%, 63:20%)

Strategy: Buy top-N by CompRS each rebalance, hold until next rebalance.
Benchmark: SMH (VanEck Semiconductor ETF)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# CONFIG
# ============================================================
TICKERS = [
    'NVDA', 'AMD', 'AVGO', 'MRVL', 'TSM', 'ASML', 'AMAT', 'LRCX', 'KLAC', 'MU',
    'GEV', 'ETN', 'VRT', 'PWR', 'MTZ', 'CEG', 'CCJ', 'BE', 'NVT', 'EQIX', 'DLR',
    'ANET', 'COHR', 'LITE', 'MTSI', 'FN', 'APH', 'SMTC',
    'SNPS', 'CDNS', 'ARM', 'INTC', 'GFS', 'ON', 'MPWR',
    'ECL', 'XYL', 'DOV', 'FIX', 'EME',
    'MSFT', 'GOOGL', 'AMZN', 'META', 'ORCL',
    'AXTI', 'PLAB', 'ACLS', 'ONTO', 'FSLR',
]

BENCHMARK = 'SMH'
LOOKBACK_YEARS = 2
TOP_N = 10           # Hold top-N stocks
REBALANCE_DAYS = 10  # Trading days between rebalances
INITIAL_CAPITAL = 100000

WEIGHT_CONFIGS = {
    'A_Current_Indian':   [(5, 0.10), (21, 0.50), (63, 0.40)],
    'B_Balanced':         [(5, 0.20), (21, 0.50), (63, 0.30)],
    'C_Aggressive':       [(5, 0.25), (21, 0.50), (63, 0.25)],
    'D_Ultra_Fast':       [(5, 0.30), (21, 0.50), (63, 0.20)],
}

MA_FILTER = 30  # Price > MA30 for entry

# ============================================================
# DATA FETCH
# ============================================================
def fetch_all_data():
    """Download all tickers + benchmark."""
    start = (datetime.now() - timedelta(days=LOOKBACK_YEARS * 365 + 100)).strftime('%Y-%m-%d')
    all_tickers = TICKERS + [BENCHMARK]

    print(f"Downloading {len(all_tickers)} tickers from {start}...")
    raw = yf.download(all_tickers, start=start, group_by='ticker',
                      threads=True, progress=True, auto_adjust=True)

    data = {}
    for t in all_tickers:
        try:
            if t in raw.columns.get_level_values(0):
                df = raw[t].dropna(how='all')
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                if len(df) > 100:
                    data[t] = df
        except Exception:
            pass

    print(f"Loaded: {len(data)} / {len(all_tickers)}")
    return data


def compute_rs(data, ticker, bench_ticker, date_idx, rs_weights):
    """Compute composite RS for a ticker at a given date index."""
    if ticker not in data or bench_ticker not in data:
        return None

    df = data[ticker]
    bench = data[bench_ticker]

    # Align to date
    df_slice = df.loc[:date_idx]
    bench_slice = bench.loc[:date_idx]

    if len(df_slice) < 64 or len(bench_slice) < 64:
        return None

    price = float(df_slice['Close'].iloc[-1])
    bench_price = float(bench_slice['Close'].iloc[-1])

    rs_total = 0.0
    for period, weight in rs_weights:
        if len(df_slice) < period + 1 or len(bench_slice) < period + 1:
            return None
        s_ret = price / float(df_slice['Close'].iloc[-period]) - 1
        b_ret = bench_price / float(bench_slice['Close'].iloc[-period]) - 1
        rs_total += (s_ret - b_ret) * 100 * weight

    return rs_total


# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest(data, rs_weights, config_name):
    """Simple top-N rebalance backtest."""
    bench = data[BENCHMARK]
    dates = bench.index

    # Start after warmup
    start_idx = max(100, MA_FILTER + 10)
    rebalance_dates = list(range(start_idx, len(dates), REBALANCE_DAYS))

    portfolio = {}   # ticker -> shares
    cash = INITIAL_CAPITAL
    equity_curve = []
    trades = []

    for reb_i, idx in enumerate(rebalance_dates):
        date = dates[idx]

        # 1. Mark-to-market existing holdings
        total_equity = cash
        for t, shares in portfolio.items():
            if t in data:
                df = data[t]
                prices = df.loc[:date, 'Close']
                if not prices.empty:
                    total_equity += shares * float(prices.iloc[-1])

        # 2. Score all stocks
        scores = []
        for t in TICKERS:
            if t not in data:
                continue
            df = data[t]
            prices = df.loc[:date, 'Close']
            if len(prices) < MA_FILTER + 10:
                continue

            price = float(prices.iloc[-1])
            ma = float(prices.rolling(MA_FILTER).mean().iloc[-1])

            if price <= ma:
                continue  # MA filter

            rs = compute_rs(data, t, BENCHMARK, date, rs_weights)
            if rs is not None:
                scores.append((t, rs, price))

        scores.sort(key=lambda x: -x[1])
        top_picks = scores[:TOP_N]

        # 3. Sell everything, re-buy top-N (equal weight)
        # Liquidate
        for t, shares in portfolio.items():
            if t in data:
                df = data[t]
                prices = df.loc[:date, 'Close']
                if not prices.empty:
                    sell_price = float(prices.iloc[-1])
                    cash += shares * sell_price * 0.999  # Slippage

        portfolio = {}

        # Buy top-N
        if top_picks:
            per_stock = cash / len(top_picks)
            for t, rs, price in top_picks:
                shares = int(per_stock / price)
                if shares > 0:
                    cost = shares * price * 1.001  # Slippage
                    cash -= cost
                    portfolio[t] = shares
                    trades.append({
                        'Date': date, 'Ticker': t, 'RS': round(rs, 2),
                        'Price': round(price, 2), 'Action': 'BUY'
                    })

        # Record equity
        eq = cash
        for t, shares in portfolio.items():
            if t in data:
                df = data[t]
                prices = df.loc[:date, 'Close']
                if not prices.empty:
                    eq += shares * float(prices.iloc[-1])

        equity_curve.append({'Date': date, 'Equity': eq})

    # Final mark-to-market
    final_date = dates[-1]
    final_eq = cash
    for t, shares in portfolio.items():
        if t in data:
            df = data[t]
            prices = df.loc[:final_date, 'Close']
            if not prices.empty:
                final_eq += shares * float(prices.iloc[-1])
    equity_curve.append({'Date': final_date, 'Equity': final_eq})

    eq_df = pd.DataFrame(equity_curve)
    eq_df['Date'] = pd.to_datetime(eq_df['Date'])
    eq_df = eq_df.drop_duplicates('Date').sort_values('Date').reset_index(drop=True)

    # Metrics
    total_return = (final_eq / INITIAL_CAPITAL - 1) * 100
    days = (eq_df['Date'].iloc[-1] - eq_df['Date'].iloc[0]).days
    cagr = ((final_eq / INITIAL_CAPITAL) ** (365 / max(days, 1)) - 1) * 100

    # Max drawdown
    eq_df['Peak'] = eq_df['Equity'].cummax()
    eq_df['DD'] = (eq_df['Equity'] - eq_df['Peak']) / eq_df['Peak'] * 100
    max_dd = eq_df['DD'].min()

    # Sharpe (approximate)
    eq_df['Ret'] = eq_df['Equity'].pct_change()
    sharpe = eq_df['Ret'].mean() / eq_df['Ret'].std() * np.sqrt(252) if eq_df['Ret'].std() > 0 else 0

    # Benchmark comparison
    bench_start = float(bench.loc[:eq_df['Date'].iloc[0], 'Close'].iloc[-1])
    bench_end = float(bench.loc[:eq_df['Date'].iloc[-1], 'Close'].iloc[-1])
    bench_return = (bench_end / bench_start - 1) * 100
    bench_cagr = ((bench_end / bench_start) ** (365 / max(days, 1)) - 1) * 100

    alpha = cagr - bench_cagr

    return {
        'Config': config_name,
        'Weights': str([(p, w) for p, w in rs_weights]),
        'Total_Return': round(total_return, 2),
        'CAGR': round(cagr, 2),
        'Max_DD': round(max_dd, 2),
        'Sharpe': round(sharpe, 2),
        'Bench_Return': round(bench_return, 2),
        'Bench_CAGR': round(bench_cagr, 2),
        'Alpha_CAGR': round(alpha, 2),
        'Trades': len(trades),
        'Rebalances': len(rebalance_dates),
    }, eq_df, pd.DataFrame(trades)


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("AI DATACENTER CAPEX — RS WEIGHT BACKTEST")
    print(f"Universe: {len(TICKERS)} US AI supply chain stocks")
    print(f"Benchmark: {BENCHMARK} | Top-{TOP_N} | Rebal: {REBALANCE_DAYS}d")
    print(f"Period: {LOOKBACK_YEARS} years | MA Filter: MA{MA_FILTER}")
    print("=" * 70)

    data = fetch_all_data()
    if BENCHMARK not in data:
        print("ERROR: Could not fetch benchmark. Exiting.")
        sys.exit(1)

    all_results = []
    all_equity = {}

    for config_name, weights in WEIGHT_CONFIGS.items():
        print(f"\n{'='*50}")
        print(f"Testing: {config_name} — {weights}")
        print(f"{'='*50}")

        metrics, eq_df, trades_df = run_backtest(data, weights, config_name)
        all_results.append(metrics)
        all_equity[config_name] = eq_df

        print(f"  Total Return: {metrics['Total_Return']:+.1f}%")
        print(f"  CAGR:         {metrics['CAGR']:+.1f}%")
        print(f"  Max Drawdown: {metrics['Max_DD']:.1f}%")
        print(f"  Sharpe:       {metrics['Sharpe']:.2f}")
        print(f"  Bench CAGR:   {metrics['Bench_CAGR']:+.1f}%")
        print(f"  Alpha (CAGR): {metrics['Alpha_CAGR']:+.1f}%")

    # Summary table
    summary = pd.DataFrame(all_results)
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))

    # Save results
    summary.to_csv('ai_capex_backtest_results.csv', index=False)
    print(f"\nResults saved to ai_capex_backtest_results.csv")

    # Save equity curves
    for name, eq in all_equity.items():
        eq.to_csv(f'ai_capex_equity_{name}.csv', index=False)

    # Winner
    best = summary.loc[summary['Alpha_CAGR'].idxmax()]
    print(f"\n🏆 BEST CONFIG: {best['Config']}")
    print(f"   Alpha: {best['Alpha_CAGR']:+.1f}% CAGR over {BENCHMARK}")
    print(f"   Sharpe: {best['Sharpe']:.2f}")
    print(f"   Max DD: {best['Max_DD']:.1f}%")
