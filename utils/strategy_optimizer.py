"""
Strategy Optimizer
==================
Tests multiple parameter combinations to find high-alpha strategies.
Includes anti-overfitting safeguards and composite scoring.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Callable
from itertools import product
import streamlit as st

from utils.portfolio_backtest import (
    fetch_historical_prices, get_ticker_data, calculate_metrics_at_date,
    calculate_performance_metrics, analyze_factor_performance
)


# ============================================
# TWO-STAGE OPTIMIZATION GRIDS (v5 - Optimized)
# ============================================

# Stage 1: Entry Optimization - Raw Grid Values
ENTRY_GRID = {
    "entry_trend_score": [35, 45, 55, 65],     # Added 65 for pure Momentum
    "entry_trend_score_max": [60, 75, 85, 90], # Added 90 for wide windows
    "quality_min": [0, 5, 6],                   # Moderate anchor
    "value_min": [0, 5, 7],                     # Added 7 for Deep Value
    "growth_min": [0, 5, 7],                    # Added 7 for GARP
}

# Stage 2: Exit Optimization (refine exit params for winning entry)
EXIT_GRID = {
    "time_exit_days": [20, 30, 45],
    "trailing_stop_pct": [-8, -10, -15],
    "stop_loss_pct": [-15, -20],
    "volume_combined_min": [5, 7],
}

# Default fixed params for Stage 1 (OPTIMIZED exit defaults)
STAGE1_FIXED_EXITS = {
    "time_exit_days": 45,           # Extended from 30 (let winners run)
    "trailing_stop_pct": -8,        # Looser from -10
    "stop_loss_pct": -15,
    "volume_combined_min": 6,       # Volume confirmation for entry
}

# Fixed parameters (not optimized in either stage)
FIXED_PARAMS = {
    "initial_capital": 200000,
    "portfolio_size": 20,
    "rebalance_freq_days": 14,
    "lookback_months": 12,
    "exit_trend_score": 40,
    "partial_profit_pct": 20.0,
    "partial_sell_pct": 0.33,
    "time_exit_min_return": 2.0,
    "time_exit_max_loss": -5.0,
    "use_volume_filter": True,
    "entry_overall_score": 0,
}

# Probability estimates for coverage calculation
FILTER_PROBABILITIES = {
    "quality_min": {0: 1.0, 5: 0.60, 6: 0.40, 7: 0.25},
    "value_min": {0: 1.0, 5: 0.55, 6: 0.35, 7: 0.20},
    "growth_min": {0: 1.0, 5: 0.55, 6: 0.35, 7: 0.22},
}

# Coverage floor (minimum % of universe to consider valid)
COVERAGE_FLOOR = 6.0

# Legacy default grid (for backward compatibility)
DEFAULT_GRID = ENTRY_GRID


def calculate_coverage(trend_min: int, trend_max: int, quality: int, value: int, growth: int) -> float:
    """
    Calculate estimated coverage % for a parameter combination.
    Returns: Percentage of stock universe expected to pass filters.
    """
    # Trend window probability (assuming uniform distribution 0-100)
    trend_window = trend_max - trend_min
    p_trend = trend_window / 75.0  # Normalize to typical trading range
    
    # Fundamental probability (multiply probabilities)
    p_quality = FILTER_PROBABILITIES["quality_min"].get(quality, 1.0)
    p_value = FILTER_PROBABILITIES["value_min"].get(value, 1.0)
    p_growth = FILTER_PROBABILITIES["growth_min"].get(growth, 1.0)
    
    p_fund = p_quality * p_value * p_growth
    
    # Total coverage
    coverage = p_fund * p_trend * 100
    return round(coverage, 2)


def classify_archetype(trend_min: int, quality: int, value: int, growth: int) -> str:
    """
    Classify a parameter combination into an archetype.
    Priority order: Deep Value > High Growth > GARP > Momentum > Balanced > Baseline
    """
    if value >= 7:
        return "Deep Value"
    if growth >= 7:
        return "High Growth (GARP)"
    if growth >= 5 and value >= 5:
        return "GARP"
    if trend_min >= 55:
        return "Momentum"
    if quality >= 5 and (value >= 5 or growth >= 5):
        return "Balanced"
    return "Baseline"


def generate_param_grid(grid: Dict = None, fixed_overrides: Dict = None) -> List[Dict]:
    """
    Generate valid parameter combinations with intelligent filtering.
    
    Includes:
    - Anti-Unicorn exclusion rules (prevent impossible combinations)
    - Coverage floor (discard configs < 6% universe coverage)
    - Archetype classification
    
    Args:
        grid: Parameter grid to use (ENTRY_GRID, EXIT_GRID, etc.)
        fixed_overrides: Additional fixed params to merge (for two-stage optimization)
    """
    if grid is None:
        grid = DEFAULT_GRID
    
    # Check if this is an entry optimization (has fundamental filters)
    is_entry_grid = "quality_min" in grid or "value_min" in grid
    
    # Generate all combinations
    keys = list(grid.keys())
    values = list(grid.values())
    combinations = list(product(*values))
    
    valid_configs = []
    archetype_counts = {}
    
    for combo in combinations:
        config = dict(zip(keys, combo))
        
        # === VALIDATION RULES ===
        
        # Rule 0: Trend min < Trend max
        trend_min = config.get("entry_trend_score", 0)
        trend_max = config.get("entry_trend_score_max", 100)
        if trend_min >= trend_max:
            continue
        
        # Only apply advanced rules for entry grid
        if is_entry_grid:
            quality = config.get("quality_min", 0)
            value = config.get("value_min", 0)
            growth = config.get("growth_min", 0)
            
            # Rule A: Deep Value Isolation
            # If Value >= 7 (cheap), cannot also demand high Growth or Quality
            if value >= 7:
                if growth >= 5 or quality >= 6:
                    continue
            
            # Rule B: High Growth Isolation  
            # If Growth >= 7 (fast growers), cannot demand they be cheap
            if growth >= 7:
                if value >= 7:
                    continue
            
            # Rule C: Coverage Floor
            # Discard combinations that cover < 6% of universe
            coverage = calculate_coverage(trend_min, trend_max, quality, value, growth)
            if coverage < COVERAGE_FLOOR:
                continue
            
            # Classify archetype
            archetype = classify_archetype(trend_min, quality, value, growth)
            config["archetype"] = archetype
            config["coverage_pct"] = coverage
            archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1
        
        # Add fixed params + overrides
        full_config = {**FIXED_PARAMS}
        if fixed_overrides:
            full_config.update(fixed_overrides)
        full_config.update(config)
        
        valid_configs.append(full_config)
    
    return valid_configs


# ============================================
# QUICK BACKTEST (Lightweight for Grid Search)
# ============================================

def run_quick_backtest(
    market_df: pd.DataFrame,
    config: Dict,
    historical_data: pd.DataFrame = None,
    start_date: datetime = None,
    end_date: datetime = None
) -> Dict:
    """
    Run a lightweight backtest without UI callbacks or detailed analytics.
    Returns only key metrics for comparison.
    """
    # Date range
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=config.get('lookback_months', 12) * 30)
    
    # Fetch data if not provided (for standalone use)
    if historical_data is None or historical_data.empty:
        tickers = market_df['ticker'].tolist()
        fetch_start = start_date - timedelta(days=400)
        historical_data = fetch_historical_prices(tickers, fetch_start, end_date)
    
    if historical_data.empty:
        return {"error": "No data"}
    
    # Setup
    rebalance_dates = pd.date_range(
        start=start_date, 
        end=end_date, 
        freq=f'{config["rebalance_freq_days"]}D'
    )
    
    equity_curve = []
    trades = []
    holdings = {}
    cash = config['initial_capital']
    
    # Main Loop (simplified)
    for rebal_date in rebalance_dates:
        # Update holdings value and check exits
        current_holdings_val = 0
        to_sell = []
        
        holding_prices = {}
        for ticker in holdings:
            df = get_ticker_data(historical_data, ticker)
            d_slice = df[df.index <= rebal_date]
            if not d_slice.empty:
                holding_prices[ticker] = float(d_slice['Close'].iloc[-1])
            else:
                holding_prices[ticker] = holdings[ticker]['entry_price']
        
        # Check exits
        for ticker, holding in holdings.items():
            price = holding_prices[ticker]
            current_holdings_val += holding['shares'] * price
            holding['peak_price'] = max(holding.get('peak_price', 0), price)
            
            ret_pct = (price / holding['entry_price'] - 1) * 100
            dd_pct = (price / holding['peak_price'] - 1) * 100
            days_held = (rebal_date - holding['entry_date']).days
            
            exit_reason = None
            if ret_pct <= config['stop_loss_pct']:
                exit_reason = "STOP_LOSS"
            elif dd_pct <= config['trailing_stop_pct']:
                exit_reason = "TRAILING_STOP"
            elif days_held >= config['time_exit_days']:
                if ret_pct < config['time_exit_min_return'] and ret_pct > config['time_exit_max_loss']:
                    exit_reason = "TIME_STAGNANT"
            else:
                df = get_ticker_data(historical_data, ticker)
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
        
        # Execute exits
        for sale in to_sell:
            del holdings[sale['ticker']]
            cash += sale['shares'] * sale['price']
            current_holdings_val -= sale['shares'] * sale['price']
            trades.append({'action': 'SELL', 'date': rebal_date, **sale})
        
        total_equity = cash + current_holdings_val
        equity_curve.append({'date': rebal_date, 'equity': total_equity})
        
        # Select new entries
        open_slots = config['portfolio_size'] - len(holdings)
        if open_slots > 0 and cash > 5000:
            candidates = []
            
            for idx, row in market_df.iterrows():
                ticker = row['ticker']
                if ticker in holdings:
                    continue
                
                df = get_ticker_data(historical_data, ticker)
                metrics = calculate_metrics_at_date(ticker, df, rebal_date)
                
                if not metrics:
                    continue
                
                # Check entry criteria
                trend = metrics['trend_score']
                quality = row.get('quality', 5)
                value = row.get('value', 5)
                
                if trend < config['entry_trend_score']:
                    continue
                if trend > config['entry_trend_score_max']:
                    continue
                if quality < config.get('quality_min', 0):
                    continue
                if value < config.get('value_min', 0):
                    continue
                
                # Growth filter
                growth = row.get('growth', 5)
                if growth < config.get('growth_min', 0):
                    continue
                
                # Volume filter
                if config.get('use_volume_filter', True) and metrics.get('volume_score', 5) < config.get('volume_combined_min', 5):
                    continue
                
                candidates.append({
                    'ticker': ticker, 'price': metrics['price'],
                    'trend_score': trend,
                    'quality': quality,
                    'value': value,
                    'growth': row.get('growth', 5)
                })
            
            # Sort by trend score descending
            candidates.sort(key=lambda x: x['trend_score'], reverse=True)
            to_buy = candidates[:open_slots]
            
            target_pos_size = total_equity / config['portfolio_size']
            
            for buy in to_buy:
                shares = int(min(cash, target_pos_size) / buy['price'])
                if shares > 0:
                    cost = shares * buy['price']
                    cash -= cost
                    holdings[buy['ticker']] = {
                        'shares': shares, 
                        'entry_price': buy['price'],
                        'peak_price': buy['price'],
                        'entry_date': rebal_date, 
                        'entry_trend_score': buy['trend_score'],
                        'quality': buy['quality'], 
                        'value': buy['value'], 
                        'growth': buy['growth']
                    }
                    trades.append({
                        'action': 'BUY', 'ticker': buy['ticker'],
                        'price': buy['price'], 'shares': shares, 
                        'date': rebal_date, 'entry_trend_score': buy['trend_score']
                    })
    
    # Mark open positions
    if equity_curve:
        last_date = equity_curve[-1]['date']
        for ticker, holding in holdings.items():
            df = get_ticker_data(historical_data, ticker)
            d_slice = df[df.index <= last_date]
            if not d_slice.empty:
                current_price = float(d_slice['Close'].iloc[-1])
                ret_pct = (current_price / holding['entry_price'] - 1) * 100
                days_held = (last_date - holding['entry_date']).days
                trades.append({
                    'action': 'HELD', 'ticker': ticker, 'price': current_price,
                    'shares': holding['shares'], 'date': last_date,
                    'reason': 'OPEN_POSITION', 'return_pct': ret_pct,
                    'days_held': days_held,
                    'entry_trend_score': holding.get('entry_trend_score', 0),
                    'quality': holding.get('quality', 5),
                    'value': holding.get('value', 5),
                    'growth': holding.get('growth', 5)
                })
    
    # Calculate metrics
    if not equity_curve:
        return {"error": "No equity data"}
    
    metrics = calculate_performance_metrics(equity_curve, config['initial_capital'], trades)
    
    return {
        "equity_curve": pd.DataFrame(equity_curve),
        "trades": trades,
        "metrics": metrics,
        "config": config
    }


# ============================================
# COMPOSITE SCORING
# ============================================

def calculate_composite_score(metrics: Dict, trades: List[Dict]) -> float:
    """
    Calculate composite score for ranking strategies.
    Higher = better.
    """
    if not metrics:
        return 0.0
    
    # Extract metrics
    alpha = metrics.get('alpha', 0)
    sharpe = metrics.get('sharpe_ratio', 0)
    win_rate = metrics.get('win_rate_pct', 0)
    max_dd = abs(metrics.get('max_drawdown_pct', 0))
    total_trades = metrics.get('total_trades', 0)
    
    # Normalize (scale to 0-1 range approximately)
    alpha_norm = min(1, max(-1, alpha / 20))  # -20% to +20% range
    sharpe_norm = min(1, max(-1, sharpe / 2))  # -2 to +2 range
    win_norm = win_rate / 100  # 0-100%
    dd_norm = 1 - min(1, max_dd / 25)  # 0-25% range
    
    # Composite (equal weights)
    score = (
        alpha_norm * 0.30 +
        sharpe_norm * 0.30 +
        win_norm * 0.20 +
        dd_norm * 0.20
    )
    
    # Penalties
    if total_trades < 20:
        score *= 0.7  # Low confidence penalty
    
    # Avg days held penalty (churn)
    if trades:
        sell_trades = [t for t in trades if t.get('action') in ['SELL', 'HELD']]
        if sell_trades:
            avg_days = np.mean([t.get('days_held', 28) for t in sell_trades])
            if avg_days < 10:
                score *= 0.8
    
    return round(score, 4)


# ============================================
# OPTIMIZATION RUNNER
# ============================================

def run_optimization(
    market_df: pd.DataFrame,
    grid: Dict = None,
    progress_callback: Callable = None,
    max_configs: int = None
) -> pd.DataFrame:
    """
    Run optimization across all parameter combinations.
    Returns DataFrame with results sorted by composite score.
    
    Args:
        max_configs: Optional limit on number of configs to test
    """
    configs = generate_param_grid(grid)
    
    # Apply limit if specified
    if max_configs and len(configs) > max_configs:
        configs = configs[:max_configs]
    
    total = len(configs)
    
    if total == 0:
        return pd.DataFrame()
    
    # Fetch historical data once (major optimization)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year
    fetch_start = start_date - timedelta(days=400)
    
    tickers = market_df['ticker'].tolist()
    
    with st.spinner(f"Fetching historical data for {len(tickers)} stocks..."):
        historical_data = fetch_historical_prices(tickers, fetch_start, end_date)
    
    if historical_data.empty:
        return pd.DataFrame()
    
    results = []
    
    for i, config in enumerate(configs):
        if progress_callback:
            progress_callback((i + 1) / total, f"Testing config {i+1}/{total}")
        
        try:
            result = run_quick_backtest(
                market_df, config, historical_data, start_date, end_date
            )
            
            if "error" in result:
                continue
            
            metrics = result.get('metrics', {})
            trades = result.get('trades', [])
            
            score = calculate_composite_score(metrics, trades)
            
            results.append({
                'trend_min': config['entry_trend_score'],
                'trend_max': config['entry_trend_score_max'],
                'quality_min': config.get('quality_min', 0),
                'value_min': config.get('value_min', 0),
                'time_exit': config['time_exit_days'],
                'trailing_stop': config['trailing_stop_pct'],
                'return_pct': metrics.get('total_return_pct', 0),
                'alpha': metrics.get('alpha', 0),
                'sharpe': metrics.get('sharpe_ratio', 0),
                'win_rate': metrics.get('win_rate_pct', 0),
                'max_dd': metrics.get('max_drawdown_pct', 0),
                'trades': metrics.get('total_trades', 0),
                'score': score,
                'config': config
            })
        except Exception as e:
            continue
    
    if not results:
        return pd.DataFrame()
    
    # Create DataFrame and sort
    df = pd.DataFrame(results)
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    return df


def get_top_strategies(results_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Get top N strategies from optimization results."""
    if results_df.empty:
        return pd.DataFrame()
    
    cols = ['rank', 'trend_min', 'trend_max', 'quality_min', 'value_min',
            'return_pct', 'alpha', 'sharpe', 'win_rate', 'max_dd', 'trades', 'score']
    
    # Only use columns that exist
    cols = [c for c in cols if c in results_df.columns]
    
    return results_df.head(n)[cols]


def generate_heatmap_data(results_df: pd.DataFrame) -> pd.DataFrame:
    """Generate pivot table for heatmap visualization."""
    if results_df.empty:
        return pd.DataFrame()
    
    # Create trend range label
    results_df['trend_range'] = results_df['trend_min'].astype(str) + '-' + results_df['trend_max'].astype(str)
    
    # Pivot: trend_range vs value_min, color by alpha
    pivot = results_df.pivot_table(
        values='alpha',
        index='value_min',
        columns='trend_range',
        aggfunc='mean'
    ).round(2)
    
    return pivot
