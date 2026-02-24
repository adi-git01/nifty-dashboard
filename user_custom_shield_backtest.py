import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import sys
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath('main.py')))
from utils.nifty500_list import TICKERS

OUTPUT_DIR = "analysis_2026/custom_shield"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HORIZONS = {'1y': 1, '3y': 3, '5y': 5, '10y': 10, '15y': 15}

def fetch_data(years=15):
    start = (datetime.now() - timedelta(days=365 * years + 100)).strftime('%Y-%m-%d')
    print("Fetching Nifty...")
    nifty = yf.Ticker('^NSEI').history(start=start)
    nifty.index = nifty.index.tz_localize(None)
    nifty['MA50'] = nifty['Close'].rolling(50).mean()
    nifty['Dist_MA50'] = (nifty['Close'] - nifty['MA50']) / nifty['MA50'] * 100
    
    print(f"Bulk downloading 300 stocks...")
    bulk = yf.download(TICKERS[:300], start=start, progress=False, threads=True)
    
    cache = {}
    for t in TICKERS[:300]:
        try:
            if isinstance(bulk.columns, pd.MultiIndex):
                if t in bulk.columns.get_level_values(1):
                    df = bulk.xs(t, axis=1, level=1).dropna(how='all')
                    if len(df) > 200:
                        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                        cache[t] = df
            else:
                if t in bulk.columns:
                    df = bulk[t].dropna()
                    if len(df) > 200:
                        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                        cache[t] = df
        except: pass
    print(f"Loaded {len(cache)} valid stocks.")
    return nifty, cache

def run_backtest(nifty, cache, start_date, apply_custom_filter=False):
    capital = 1_000_000
    positions = {}
    trades = []
    equity_curve = []
    
    dates = nifty.index[nifty.index >= start_date]
    day_counter = 0
    
    for date in dates:
        nifty_idx = nifty.index.searchsorted(date)
        current_nifty_dist = 0
        if nifty_idx < len(nifty):
            current_nifty_dist = nifty['Dist_MA50'].iloc[nifty_idx]
            if pd.isna(current_nifty_dist): current_nifty_dist = 0
            
        force_cash = False
        skip_buys = False
        
        if apply_custom_filter:
            # Custom Rule: Exit to cash if Dist_MA50 is between -4% and +2%
            if -4.0 <= current_nifty_dist <= 2.0:
                force_cash = True
                skip_buys = True
                
        # Exits
        to_remove = []
        for t, pos in positions.items():
            if t in cache:
                df = cache[t]
                idx = df.index.searchsorted(date)
                if idx > 0 and idx < len(df):
                    price = df['Close'].iloc[idx]
                    ma50 = df['Close'].iloc[:idx+1].rolling(50).mean().iloc[-1]
                    
                    if price > pos['peak']: pos['peak'] = price
                    
                    # OptComp-V21 trailing/MA50 stop OR Force Cash
                    if force_cash or price < pos['peak'] * 0.85 or price < ma50:
                        ret = (price - pos['entry']) / pos['entry'] * 100
                        capital += pos['shares'] * price
                        
                        reason = "Force Cash" if force_cash else "Trailing/MA50 Stop"
                        trades.append({'Ticker': t, 'PnL%': ret, 'Entry': pos['entry_date'], 'Exit': date, 'Reason': reason})
                        to_remove.append(t)
        for t in to_remove: del positions[t]
        
        # Entries
        if day_counter % 13 == 0 and len(positions) < 10 and not skip_buys:
            if nifty_idx >= 63:
                candidates = []
                for t, df in cache.items():
                    if t in positions: continue
                    idx = df.index.searchsorted(date)
                    if idx < 100 or idx >= len(df): continue
                    
                    window = df.iloc[:idx+1]
                    price = window['Close'].iloc[-1]
                    ma50 = window['Close'].rolling(50).mean().iloc[-1]
                    if price < ma50: continue
                    
                    val = window['Volume'].iloc[-5:].mean() * price
                    if val < 10_000_000: continue
                    
                    def get_rs(days):
                        t_ret = (price - window['Close'].iloc[-days]) / window['Close'].iloc[-days]
                        n_ret = (nifty['Close'].iloc[nifty_idx] - nifty['Close'].iloc[nifty_idx-days]) / nifty['Close'].iloc[nifty_idx-days]
                        return (t_ret - n_ret) * 100
                    
                    comp_rs = (get_rs(5) * 0.10) + (get_rs(21) * 0.50) + (get_rs(63) * 0.40)
                    if comp_rs > 0:
                        candidates.append((t, comp_rs, price))
                
                candidates.sort(key=lambda x: -x[1])
                free_slots = 10 - len(positions)
                for t, score, price in candidates[:free_slots]:
                    size = capital / (free_slots + 1)
                    shares = int(size / price)
                    if shares > 0:
                        capital -= (shares * price)
                        positions[t] = {'entry': price, 'peak': price, 'shares': shares, 'entry_date': date}
        
        # Track Equity
        eq = capital
        for t, pos in positions.items():
            df = cache[t]
            idx = df.index.searchsorted(date)
            if idx < len(df): eq += pos['shares'] * df['Close'].iloc[idx]
        equity_curve.append({'date': date, 'equity': eq, 'cash_reserve': capital / eq * 100})
        day_counter += 1
        
    return pd.DataFrame(trades), pd.DataFrame(equity_curve)

def run_suite():
    nifty, cache = fetch_data(15)
    
    variants = [
        ('Baseline (OptComp-V21)', False),
        ('Chop Zone Shield (-4% to +2%)', True)
    ]
    
    all_results = []
    # 10Y specific for visualizations
    eq_curves_10y = {}
    
    for label, years in HORIZONS.items():
        print(f"\n--- Testing Horizon: {label} ({years} Years) ---")
        start_date = nifty.index[-1] - timedelta(days=int(365.25 * years))
        if start_date < nifty.index[0]: start_date = nifty.index[0]
        
        for name, use_filter in variants:
            trades, eq_curve = run_backtest(nifty, cache, start_date, apply_custom_filter=use_filter)
            
            if eq_curve.empty: continue
            eq = eq_curve['equity'].values
            if len(eq) < 20: continue
            
            if label == '10y':
                eq_curves_10y[name] = eq_curve
            
            cagr = ((eq[-1] / eq[0]) ** (1/max(1, years)) - 1) * 100
            
            peak = pd.Series(eq).cummax()
            dd = (pd.Series(eq) - peak) / peak * 100
            max_dd = dd.min()
            
            win_rate = (trades['PnL%'] > 0).mean() * 100 if not trades.empty else 0
            
            # Sharpe
            eq_series = pd.Series(eq)
            daily_returns = eq_series.pct_change().dropna()
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
            
            # Trades metrics
            total_trades = len(trades)
            avg_win = trades[trades['PnL%'] > 0]['PnL%'].mean() if not trades.empty and len(trades[trades['PnL%'] > 0]) > 0 else 0
            avg_loss = trades[trades['PnL%'] <= 0]['PnL%'].mean() if not trades.empty and len(trades[trades['PnL%'] <= 0]) > 0 else 0
            
            # Cash metric
            avg_cash = eq_curve['cash_reserve'].mean()
            
            all_results.append({
                'Horizon': label,
                'Variant': name,
                'CAGR%': cagr,
                'MaxDD%': max_dd,
                'Sharpe': sharpe,
                'WinRate%': win_rate,
                'Trades': total_trades,
                'AvgWin%': avg_win,
                'AvgLoss%': avg_loss,
                'AvgCash%': avg_cash
            })

    res_df = pd.DataFrame(all_results)
    res_df.to_csv(f"{OUTPUT_DIR}/summary.csv", index=False)
    
    # Generate charts for 10Y using Plotly
    if len(eq_curves_10y) >= 2:
        colors = {'Baseline (OptComp-V21)': '#1f77b4', 'Chop Zone Shield (-4% to +2%)': '#ff7f0e'}
        
        # Equity Curve Chart
        fig_eq = go.Figure()
        for name, df in eq_curves_10y.items():
            fig_eq.add_trace(go.Scatter(x=df['date'], y=df['equity'], mode='lines', name=name, line=dict(color=colors.get(name))))
            
        start_10y = eq_curves_10y['Baseline (OptComp-V21)']['date'].iloc[0]
        nifty_10y = nifty[nifty.index >= start_10y].copy()
        nifty_10y['equity'] = nifty_10y['Close'] / nifty_10y['Close'].iloc[0] * 1_000_000
        fig_eq.add_trace(go.Scatter(x=nifty_10y.index, y=nifty_10y['equity'], mode='lines', name='Nifty 50', line=dict(color='black', dash='dash')))
        
        fig_eq.update_layout(
            title='10-Year Equity Curve: Baseline vs Chop Zone Shield',
            yaxis_type="log",
            yaxis_title='Equity (Log Scale)',
            template='plotly_white',
            hovermode='x unified'
        )
        fig_eq.write_html(f"{OUTPUT_DIR}/equity_curve_10y.html")
        
        # Drawdown Chart
        fig_dd = go.Figure()
        for name, df in eq_curves_10y.items():
            peak = df['equity'].cummax()
            dd = (df['equity'] - peak) / peak * 100
            fig_dd.add_trace(go.Scatter(x=df['date'], y=dd, fill='tozeroy', mode='lines', name=name, line=dict(color=colors.get(name))))
            
        nifty_peak = nifty_10y['equity'].cummax()
        nifty_dd = (nifty_10y['equity'] - nifty_peak) / nifty_peak * 100
        fig_dd.add_trace(go.Scatter(x=nifty_10y.index, y=nifty_dd, fill='tozeroy', mode='lines', name='Nifty 50 DD', line=dict(color='black', dash='dash')))
            
        fig_dd.update_layout(
            title='10-Year Drawdown Profile',
            yaxis_title='Drawdown %',
            yaxis=dict(range=[min(nifty_dd.min(), -30) * 1.1, 5]),
            template='plotly_white',
            hovermode='x unified'
        )
        fig_dd.write_html(f"{OUTPUT_DIR}/drawdown_10y.html")
        
    print("\n" + "="*80)
    print("BACKTEST COMPLETE. Files saved to analysis_2026/custom_shield")
    print("="*80)

if __name__ == '__main__':    
    run_suite()
