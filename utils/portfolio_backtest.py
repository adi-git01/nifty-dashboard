"""
Portfolio Backtesting Engine (Advanced)
=======================================
Backtests trend-score based portfolio selection to validate predictive power.
Now includes:
- Volume Analysis (VPT + A/D)
- Partial Profit Taking
- Time-based Exits
- Factor Attribution & Heatmaps

Strategy:
- Entry: Trend Score >= 70, Volume Signal Positive (Score >= 5)
- Exit: Trend Score < 40, Stop Loss -15%, Trailing Stop -10%
- Partial: Sell 33% at +20% Profit
- Time Exit: Exit if < 2% return after 20 days (stagnant)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import streamlit as st

# Import shared volume logic
try:
    from utils.volume_analysis import get_combined_volume_signal
except ImportError:
    # Fallback if volume_analysis not found (shouldn't happen)
    def get_combined_volume_signal(*args, **kwargs):
        return {"combined_score": 5, "combined_signal": "NEUTRAL"}

# Configuration - OPTIMIZED based on grid search analysis
BACKTEST_CONFIG = {
    "initial_capital": 200000,      # ₹2 Lakhs
    "portfolio_size": 20,           # Top 20 stocks
    "rebalance_freq_days": 14,      # Bi-weekly
    "lookback_months": 12,          # 12 month backtest (increased for significance)
    
    # Entry Rules - "Hot Zone" with Volume Confirmation
    "entry_trend_score": 35,        # Enter early momentum (35-60 window won)
    "entry_trend_score_max": 60,    # Exit hot zone before peak
    "entry_overall_score": 0,       # Disabled (didn't help in testing)
    "use_volume_filter": True,      # Use VPT+A/D signal
    "volume_combined_min": 6,       # Require volume confirmation (breakout signal)
    
    # Fundamental Filters - DISABLED (didn't improve returns)
    "quality_min": 0,               # Disabled
    "value_min": 0,                 # Disabled
    "growth_min": 0,                # Disabled
    
    # Exit Rules
    "exit_trend_score": 40,         # Exit if trend score drops below
    "stop_loss_pct": -15.0,         # Hard stop loss
    "trailing_stop_pct": -8.0,      # Looser trailing (was -10, now -8)
    
    # Partial Profit
    "partial_profit_pct": 20.0,     # Take profit triggers at +20%
    "partial_sell_pct": 0.33,       # Sell 1/3rd of position
    
    # Time Exit (Opportunity Cost) - Extended from 20 to 45 days
    "time_exit_days": 45,           # Check after 45 days (was 20)
    "time_exit_min_return": 2.0,    # If return is less than 2%
    "time_exit_max_loss": -5.0,     # And not hit stop loss yet
}


def get_backtest_dates(lookback_months: int = 6) -> Tuple[datetime, datetime]:
    """Get start and end dates for backtest period."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_months * 30)
    return start_date, end_date


@st.cache_data(ttl=3600*2, show_spinner=False)
def fetch_historical_prices(tickers: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for all tickers.
    Returns DataFrame with MultiIndex (Ticker, Attributes) or Panel-like structure.
    """
    try:
        # We need OHLCV for Volume/Trend analysis, not just Close
        data = yf.download(
            tickers, 
            start=start_date, 
            end=end_date, 
            interval="1d", 
            progress=False,
            auto_adjust=True,
            group_by='ticker'
        )
        return data
    except Exception as e:
        print(f"Error fetching historical prices: {e}")
        return pd.DataFrame()


def get_ticker_data(full_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Helper to extract single ticker OHLCV from bulk download."""
    try:
        if isinstance(full_data.columns, pd.MultiIndex):
            # Check if ticker is in level 0
            if ticker in full_data.columns.get_level_values(0):
                df = full_data[ticker].copy()
                return df
        # Fallback if flat structure (rare with yf group_by='ticker')
        return pd.DataFrame()
    except:
        return pd.DataFrame()


# ==========================================
# ANALYTICS HELPERS (Heatmaps, Buckets)
# ==========================================

def get_bucket(val, buckets=[(0,4,"0-4"), (5,7,"5-7"), (8,10,"8-10")]):
    if pd.isna(val): return "Unknown"
    for low, high, label in buckets:
        if low <= val <= high: return label
    return "Unknown"

def get_trend_bucket(score):
    if pd.isna(score): return "Unknown"
    if score < 60: return "<60"
    elif score < 70: return "60-70"
    elif score < 85: return "70-85"
    else: return "85+"

def analyze_factor_performance(trades: List[Dict]) -> Dict[str, pd.DataFrame]:
    if not trades: return {}
    # Filter for sell trades
    # Filter for sell and held trades (include open positions in analysis)
    relevant_trades = [t for t in trades if t['action'] in ['SELL', 'HELD'] and 'return_pct' in t]
    if not relevant_trades: return {}
    
    df = pd.DataFrame(relevant_trades)
    results = {}
    
    for factor in ['quality', 'value', 'growth']:
        if factor in df.columns:
            df[f'{factor}_bucket'] = df[factor].apply(lambda x: get_bucket(x) if pd.notna(x) else "Unknown")
            agg = df.groupby(f'{factor}_bucket').agg({
                'return_pct': ['mean', 'count', lambda x: (x>0).mean()*100]
            }).round(2)
            agg.columns = ['avg_return', 'count', 'win_rate']
            results[factor.capitalize()] = agg.to_dict(orient='index')
            
    # Trend buckets
    if 'entry_trend_score' in df.columns:
         df['trend_bucket'] = df['entry_trend_score'].apply(get_trend_bucket)
         agg = df.groupby('trend_bucket').agg({
                'return_pct': ['mean', 'count', lambda x: (x>0).mean()*100]
            }).round(2)
         agg.columns = ['avg_return', 'count', 'win_rate']
         results['Trend Score'] = agg.to_dict(orient='index')
         
    return results

def generate_heatmap_data(trades: List[Dict], factor: str="quality") -> pd.DataFrame:
    if not trades: return pd.DataFrame()
    if not trades: return pd.DataFrame()
    relevant_trades = [t for t in trades if t['action'] in ['SELL', 'HELD'] and 'return_pct' in t]
    if not relevant_trades: return pd.DataFrame()
    
    df = pd.DataFrame(relevant_trades)
    if factor not in df.columns: return pd.DataFrame()
    
    if 'entry_trend_score' in df.columns:
        df['trend_bucket'] = df['entry_trend_score'].apply(get_trend_bucket)
    else:
        return pd.DataFrame()
        
    df[f'{factor}_bucket'] = df[factor].apply(lambda x: get_bucket(x) if pd.notna(x) else "Unknown")
    
    return df.pivot_table(
        values='return_pct', index=f'{factor}_bucket', columns='trend_bucket', aggfunc='mean'
    ).round(2)

def analyze_exit_reasons(trades: List[Dict]) -> pd.DataFrame:
    if not trades: return pd.DataFrame()
    if not trades: return pd.DataFrame()
    relevant_trades = [t for t in trades if t['action'] in ['SELL', 'HELD']]
    if not relevant_trades: return pd.DataFrame()
    
    df = pd.DataFrame(relevant_trades)
    if 'reason' not in df.columns: return pd.DataFrame()
    
    agg = df.groupby('reason').agg({
        'return_pct': ['mean', 'count'],
        'days_held': 'mean'
    }).round(2)
    agg.columns = ['Avg Return %', 'Count', 'Avg Days']
    return agg


# ==========================================
# CORE BACKTEST LOGIC (Single Config)
# ==========================================

def calculate_metrics_at_date(ticker: str, ohlcv: pd.DataFrame, date: datetime) -> Dict:
    """Calculate trend score, volume signal, and other metrics at a specific date."""
    # Filter data up to date
    data = ohlcv[ohlcv.index <= date].copy()
    if len(data) < 200: return {}
    
    close = data['Close']
    try:
        current_price = float(close.iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
        
        # Trend Score Calculation
        trend_score = 50
        if current_price > ma50: trend_score += 15
        else: trend_score -= 10
        if current_price > ma200: trend_score += 15
        else: trend_score -= 15
        if ma50 > ma200: trend_score += 10
        else: trend_score -= 5
        
        # 52W High Relation
        if 'High' in data.columns and 'Low' in data.columns:
            high_52 = data['High'].iloc[-252:].max()
            low_52 = data['Low'].iloc[-252:].min()
            if high_52 > low_52:
                pos = (current_price - low_52) / (high_52 - low_52)
                trend_score += int(round((pos - 0.5) * 30))
        
        # 52W Dist Bonus
        # (Simplified, assume calculated inside trend score loop usually)
        
        trend_score = max(0, min(100, trend_score))
        
        # Volume Signal
        vol_signal = {"combined_score": 5, "combined_signal": "NEUTRAL"}
        if 'Volume' in data.columns:
            vol_signal = get_combined_volume_signal(
                data['High'] if 'High' in data.columns else close, 
                data['Low'] if 'Low' in data.columns else close, 
                data['Close'], 
                data['Volume']
            )
            
        return {
            "price": current_price,
            "trend_score": trend_score,
            "ma200": ma200,
            "volume_score": vol_signal.get('combined_score', 5),
            "volume_signal": vol_signal.get('combined_signal', "NEUTRAL")
        }
    except:
        return {}


def run_backtest(market_df: pd.DataFrame, config: Dict = BACKTEST_CONFIG, progress_callback=None) -> Dict:
    """Run the advanced portfolio backtest."""
    start_date, end_date = get_backtest_dates(config['lookback_months'])
    
    # Fetch Data
    tickers = market_df['ticker'].tolist()
    fetch_start = start_date - timedelta(days=400) # Buffer for MAs
    full_data = fetch_historical_prices(tickers, fetch_start, end_date)
    
    if full_data.empty: return {"error": "No data"}
    
    # Setup
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=f'{config["rebalance_freq_days"]}D')
    equity_curve = []
    trades = []
    holdings = {} # ticker -> {shares, entry_price, peak_price, entry_date, partial_taken, ...}
    cash = config['initial_capital']
    
    # Main Loop
    for i, rebal_date in enumerate(rebalance_dates):
        if progress_callback: progress_callback((i+1)/len(rebalance_dates))
        
        # 1. Update Portfolio Value & Check Exits
        current_holdings_val = 0
        to_sell = []
        partial_sells = []
        
        # Get prices for all holdings first
        holding_prices = {}
        for ticker in holdings:
            df = get_ticker_data(full_data, ticker)
            d_slice = df[df.index <= rebal_date]
            if not d_slice.empty:
                holding_prices[ticker] = float(d_slice['Close'].iloc[-1])
            else:
                holding_prices[ticker] = holdings[ticker]['entry_price'] # Fallback
        
        # Check Exits
        for ticker, holding in holdings.items():
            price = holding_prices[ticker]
            current_holdings_val += holding['shares'] * price
            
            # Track peak for trailing stop
            holding['peak_price'] = max(holding.get('peak_price', 0), price)
            
            # metrics needed for partial/time exit
            ret_pct = (price / holding['entry_price'] - 1) * 100
            dd_pct = (price / holding['peak_price'] - 1) * 100
            days_held = (rebal_date - holding['entry_date']).days
            
            # A. Check Partial Profit
            if not holding.get('partial_taken') and ret_pct >= config['partial_profit_pct']:
                shares_sell = int(holding['shares'] * config['partial_sell_pct'])
                if shares_sell > 0:
                    partial_sells.append({
                        'ticker': ticker, 'price': price, 'shares': shares_sell, 
                        'reason': 'PARTIAL_PROFIT', 'return_pct': ret_pct, 'days_held': days_held,
                        'entry_trend_score': holding.get('entry_trend_score', 0),
                        'quality': holding.get('quality', 5),
                        'value': holding.get('value', 5),
                        'growth': holding.get('growth', 5)
                    })
                    holding['shares'] -= shares_sell
                    holding['partial_taken'] = True
            
            # B. Check Full Exits
            exit_reason = None
            if ret_pct <= config['stop_loss_pct']:
                exit_reason = "STOP_LOSS"
            elif dd_pct <= config['trailing_stop_pct']:
                exit_reason = "TRAILING_STOP"
            elif days_held >= config['time_exit_days'] and ret_pct < config['time_exit_min_return'] and ret_pct > config['time_exit_max_loss']:
                exit_reason = "TIME_STAGNANT"
            else:
                # Re-check trend score
                df = get_ticker_data(full_data, ticker)
                metrics = calculate_metrics_at_date(ticker, df, rebal_date)
                if metrics and metrics['trend_score'] < config['exit_trend_score']:
                    exit_reason = "TREND_REVERSAL"
            
            if exit_reason:
                to_sell.append({
                    'ticker': ticker, 'price': price, 'shares': holding['shares'],
                    'reason': exit_reason, 'return_pct': ret_pct, 'days_held': days_held,
                    'entry_trend_score': holding.get('entry_trend_score', 0),
                    'quality': holding.get('quality', 5), 
                    'value': holding.get('value', 5), 
                    'growth': holding.get('growth', 5)
                })
        
        # Execute Exits
        for sale in partial_sells:
            cash += sale['shares'] * sale['price']
            trades.append({
                'action': 'SELL', 'partial': True, 'date': rebal_date, **sale
            })
            
        for sale in to_sell:
            del holdings[sale['ticker']]
            cash += sale['shares'] * sale['price']
            current_holdings_val -= sale['shares'] * sale['price'] # Adjust val
            trades.append({
                'action': 'SELL', 'partial': False, 'date': rebal_date, **sale
            })
            
        total_equity = cash + current_holdings_val
        equity_curve.append({'date': rebal_date, 'equity': total_equity})
        
        # 2. Select New Entries
        open_slots = config['portfolio_size'] - len(holdings)
        if open_slots > 0 and cash > 5000:
            candidates = []
            
            # Find eligible stocks
            for idx, row in market_df.iterrows():
                ticker = row['ticker']
                if ticker in holdings: continue
                
                df = get_ticker_data(full_data, ticker)
                metrics = calculate_metrics_at_date(ticker, df, rebal_date)
                
                if not metrics: continue
                
                # Check Entry Criteria
                if metrics['trend_score'] >= config['entry_trend_score'] and metrics['trend_score'] <= config['entry_trend_score_max']:
                    if config['use_volume_filter'] and metrics['volume_score'] < 5:
                        continue # Skip based on volume
                        
                    candidates.append({
                        'ticker': ticker, 'price': metrics['price'], 
                        'trend_score': metrics['trend_score'],
                        'quality': row.get('quality', 5),
                        'value': row.get('value', 5),
                        'growth': row.get('growth', 5)
                    })
            
            # Sort by Trend Score and Buy
            candidates.sort(key=lambda x: x['trend_score'], reverse=True)
            to_buy = candidates[:open_slots]
            
            # Allocate capital per stock (Dynamic Sizing)
            target_pos_size = total_equity / config['portfolio_size']
            
            for buy in to_buy:
                shares = int(min(cash, target_pos_size) / buy['price'])
                if shares > 0:
                    cost = shares * buy['price']
                    cash -= cost
                    holdings[buy['ticker']] = {
                        'shares': shares, 'entry_price': buy['price'], 'peak_price': buy['price'],
                        'entry_date': rebal_date, 'entry_trend_score': buy['trend_score'],
                        'quality': buy['quality'], 'value': buy['value'], 'growth': buy['growth']
                    }
                    trades.append({
                        'action': 'BUY', 'ticker': buy['ticker'], 'price': buy['price'], 
                        'shares': shares, 'date': rebal_date, 'entry_trend_score': buy['trend_score'],
                        'entry_price': buy['price'], # Explicitly add for tracking
                        'quality': buy['quality'], 'value': buy['value'], 'growth': buy['growth']
                    })
    
    # 3. Mark Open Positions to Market (MTM) for Reporting
    # This allows external tools to see unrealized P&L in the trade log
    end_prices = {}
    last_date = equity_curve[-1]['date'] if equity_curve else end_date
    
    # Get closing prices for all open positions
    for ticker in holdings:
        df = get_ticker_data(full_data, ticker)
        d_slice = df[df.index <= last_date]
        if not d_slice.empty:
            end_prices[ticker] = float(d_slice['Close'].iloc[-1])
        else:
            end_prices[ticker] = holdings[ticker]['entry_price']
            
    for ticker, holding in holdings.items():
        current_price = end_prices[ticker]
        ret_pct = (current_price / holding['entry_price'] - 1) * 100
        days_held = (last_date - holding['entry_date']).days
        
        trades.append({
            'action': 'HELD', # Special status for open positions
            'ticker': ticker,
            'price': current_price,
            'shares': holding['shares'],
            'date': last_date,
            'reason': 'OPEN_POSITION',
            'return_pct': ret_pct,
            'days_held': days_held,
            'position_value': holding['shares'] * current_price,
            'entry_price': holding['entry_price'],
            'entry_trend_score': holding.get('entry_trend_score', 0),
            'quality': holding.get('quality', 5),
            'value': holding.get('value', 5),
            'growth': holding.get('growth', 5)
        })

    # Analytics
    factor_perf = analyze_factor_performance(trades)
    heatmaps = {
        "Quality": generate_heatmap_data(trades, "quality"),
        "Value": generate_heatmap_data(trades, "value"),
        "Growth": generate_heatmap_data(trades, "growth"),
    }
    exit_analysis = analyze_exit_reasons(trades)
    
    # Calculate Metrics
    final_equity = equity_curve[-1]['equity'] if equity_curve else config['initial_capital']
    metrics = calculate_performance_metrics(equity_curve, config['initial_capital'], trades)
    
    return {
        "equity_curve": pd.DataFrame(equity_curve),
        "trades": trades,
        "final_equity": final_equity,
        "metrics": metrics,
        "analytics": {
            "factor_perf": factor_perf,
            "heatmaps": heatmaps,
            "exit_analysis": exit_analysis
        }
    }


def calculate_performance_metrics(equity_list: List[Dict], initial_capital: float, trades: List[Dict] = None) -> Dict:
    """Calculate Sharpe, Drawdown, Alpha, etc."""
    if not equity_list:
        return {}
        
    df = pd.DataFrame(equity_list)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Calculate returns
    df['returns'] = df['equity'].pct_change().fillna(0)
    
    total_return_pct = (df['equity'].iloc[-1] / initial_capital - 1) * 100
    
    # Max Drawdown
    cumulative_returns = (1 + df['returns']).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown_pct = drawdown.min() * 100
    
    # Sharpe Ratio (annualized, assuming risk-free = 6%)
    risk_free_daily = 0.06 / 252
    excess_returns = df['returns'] - risk_free_daily
    sharpe_ratio = 0
    if excess_returns.std() > 0:
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        
    # Win Rate (Trade-based)
    if trades:
        sell_trades = [t for t in trades if t.get('action') in ['SELL', 'HELD'] and 'return_pct' in t]
        if sell_trades:
            wins = sum(1 for t in sell_trades if t['return_pct'] > 0)
            win_rate_pct = (wins / len(sell_trades)) * 100
            total_trades = len(sell_trades)
        else:
            win_rate_pct = 0
            total_trades = 0
    else:
        win_rate_pct = (df['returns'] > 0).mean() * 100
        total_trades = 0
    
    # Simple Benchmark (Nifty 500 approx 12% annual)
    days = (df.index[-1] - df.index[0]).days
    benchmark_return_pct = 12 * (days / 365)
    alpha = total_return_pct - benchmark_return_pct
    
    return {
        "total_return_pct": round(total_return_pct, 2),
        "final_value": round(df['equity'].iloc[-1], 0),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "win_rate_pct": round(win_rate_pct, 0),
        "total_trades": total_trades,
        "benchmark_return_pct": round(benchmark_return_pct, 2),
        "alpha": round(alpha, 2)
    }


def get_current_portfolio_from_scores(df: pd.DataFrame, config: Dict = BACKTEST_CONFIG) -> pd.DataFrame:
    """
    Get the current recommended portfolio based on config criteria.
    Now supports filtering by max trend score and volume.
    """
    if df.empty: return pd.DataFrame()
    
    # Handle Column Aliases
    if 'ma200' not in df.columns and 'twoHundredDayAverage' in df.columns:
        df['ma200'] = df['twoHundredDayAverage']
    
    # Base Filters
    # Check if we have necessary columns
    req_cols = ['trend_score', 'price', 'ma200']
    if not all(col in df.columns for col in req_cols):
        return pd.DataFrame()

    candidates = df[
        (df['trend_score'] >= config['entry_trend_score']) &
        (df['price'] > df['ma200']) # Basic filter
    ].copy()
    
    # Apply Max Score Cap (Hot Zone)
    if 'entry_trend_score_max' in config:
        candidates = candidates[candidates['trend_score'] <= config['entry_trend_score_max']]
    
    # Apply Volume Filter
    if config.get('use_volume_filter', False) and 'volume_score' in candidates.columns:
         candidates = candidates[candidates['volume_score'] >= 5]
    
    if candidates.empty: return pd.DataFrame()
    
    # Sort and Pick
    candidates = candidates.sort_values('trend_score', ascending=False).head(config['portfolio_size'])
    
    # Calculate Allocation
    total_capital = config['initial_capital']
    per_stock = total_capital / config['portfolio_size']
    
    candidates['target_allocation'] = per_stock
    candidates['target_shares'] = (per_stock / candidates['price']).astype(int)
    
    return candidates
