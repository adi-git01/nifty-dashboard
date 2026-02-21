"""
Multi-Strategy Backtester
=========================
Runs backtests on multiple strategy proposals for comparison.
Supports configurable rebalancing frequency (bi-weekly/monthly).
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable
import streamlit as st

from utils.strategy_definitions import (
    ALL_PROPOSALS, REBALANCE_FREQUENCIES, 
    get_proposal, get_all_proposal_keys
)
from utils.market_regime import fetch_nifty50_data, detect_regime_for_date, get_regime_allocations


# ============================================
# HISTORICAL DATA FETCHING
# ============================================

@st.cache_data(ttl=3600*2, show_spinner=False)
def fetch_historical_data(
    tickers: List[str], 
    start_date: datetime, 
    end_date: datetime
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for all tickers.
    """
    try:
        # Add extra history for 200DMA calculation
        fetch_start = start_date - timedelta(days=365)
        
        data = yf.download(
            tickers,
            start=fetch_start,
            end=end_date,
            interval="1d",
            progress=False,
            auto_adjust=True,
            group_by='ticker'
        )
        
        return data
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()


def get_ticker_ohlcv(full_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Extract OHLCV for a single ticker from multi-ticker download."""
    try:
        if isinstance(full_data.columns, pd.MultiIndex):
            # Multi-ticker download
            if ticker in full_data.columns.get_level_values(0):
                return full_data[ticker].copy()
        else:
            # Single ticker - columns are OHLCV directly
            return full_data.copy()
        return pd.DataFrame()
    except:
        return pd.DataFrame()


# ============================================
# HISTORICAL SCORE CALCULATION
# ============================================

def calculate_historical_metrics(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_df: pd.DataFrame,
    date: datetime
) -> Dict[str, float]:
    """
    Calculate all metrics for a ticker at a specific historical date.
    
    Returns dict with: trend_score, dist_52w, dist_200dma, quality, value, growth, overall
    """
    if ohlcv.empty:
        return {}
    
    # Get data up to this date
    data_to_date = ohlcv[ohlcv.index <= date].copy()
    
    if len(data_to_date) < 200:
        return {}
    
    close = data_to_date['Close']
    high = data_to_date['High']
    low = data_to_date['Low']
    
    try:
        current_price = float(close.iloc[-1])
    except:
        return {}
    
    # Check for NaN in current price
    if pd.isna(current_price) or current_price <= 0:
        return {}
    
    # Moving averages - extract as float
    try:
        ma50 = float(close.rolling(50).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
    except:
        return {}
    
    # Check for NaN in MAs
    if pd.isna(ma50) or pd.isna(ma200):
        return {}
    
    # 52-week high/low - extract as float
    try:
        high_52w = float(high.iloc[-252:].max()) if len(high) >= 252 else float(high.max())
        low_52w = float(low.iloc[-252:].min()) if len(low) >= 252 else float(low.min())
    except:
        return {}
    
    # Check for NaN in 52W values
    if pd.isna(high_52w) or pd.isna(low_52w):
        return {}
    
    # Distance calculations - ensure float
    try:
        dist_200dma = float((current_price - ma200) / ma200 * 100) if ma200 > 0 else 0.0
        dist_52w = float((current_price - high_52w) / high_52w * 100) if high_52w > 0 else 0.0
    except:
        dist_200dma = 0.0
        dist_52w = 0.0
    
    # Check for NaN in distances
    if pd.isna(dist_200dma):
        dist_200dma = 0.0
    if pd.isna(dist_52w):
        dist_52w = 0.0
    
    # Trend score calculation (simplified from scoring.py)
    try:
        trend_score = 50
        
        if current_price > ma50:
            trend_score += 15
        else:
            trend_score -= 10
        
        if current_price > ma200:
            trend_score += 15
        else:
            trend_score -= 15
        
        if ma50 > ma200:
            trend_score += 10
        else:
            trend_score -= 5
        
        # 52-week position
        if high_52w > low_52w:
            range_52 = float(high_52w - low_52w)
            if range_52 > 0:
                position = float((current_price - low_52w) / range_52)
                # Guard against NaN - must check before int conversion
                if not pd.isna(position) and position is not None:
                    adjustment = int(round((position - 0.5) * 30))
                    trend_score += adjustment
            
            if dist_52w > -5:
                trend_score += 10
            elif dist_52w < -30:
                trend_score -= 10
        
        trend_score = max(0, min(100, trend_score))
    except Exception as e:
        # If any calculation fails, return neutral trend score
        trend_score = 50
    
    # Get fundamental scores from market_df (current snapshot - approximation)
    stock_row = market_df[market_df['ticker'] == ticker]
    if not stock_row.empty:
        quality = stock_row.iloc[0].get('quality', 5)
        value = stock_row.iloc[0].get('value', 5)
        growth = stock_row.iloc[0].get('growth', 5)
        overall = stock_row.iloc[0].get('overall', 5)
    else:
        quality = value = growth = overall = 5
    
    # Check if trend is improving (for TrueValue strategy)
    if len(close) >= 14:
        trend_2w_ago = calculate_trend_score_simple(close.iloc[:-10], ma200)
        trend_improving = trend_score > trend_2w_ago
    else:
        trend_improving = False
    
    # Calculate volume metrics if volume data available
    volume_ratio = 1.0
    obv_trend = "NEUTRAL"
    volume_score = 5
    volume_combined_score = 5
    
    if 'Volume' in data_to_date.columns:
        volume = data_to_date['Volume']
        if len(volume) >= 20 and not volume.isna().all():
            try:
                # Use the enhanced combined signal
                from utils.volume_analysis import get_combined_volume_signal
                high = data_to_date['High'] if 'High' in data_to_date.columns else close
                low = data_to_date['Low'] if 'Low' in data_to_date.columns else close
                
                vol_data = get_combined_volume_signal(high, low, close, volume)
                
                volume_ratio = calculate_volume_ratio(volume)  # Helper needed or extract from vol_data?
                # Actually get_combined doesn't return ratio, let's call legacy too or just assume ratio from raw calc
                avg_vol = volume.iloc[-20:].mean()
                if avg_vol > 0:
                    volume_ratio = volume.iloc[-1] / avg_vol
                
                obv_trend = vol_data.get('vpt_trend', 'NEUTRAL') # Use VPT as proxy for OBV trend
                volume_combined_score = vol_data.get('combined_score', 5)
                
            except:
                pass
    
    return {
        "price": current_price,
        "trend_score": trend_score,
        "dist_52w": dist_52w,
        "dist_200dma": dist_200dma,
        "quality": quality,
        "value": value,
        "growth": growth,
        "overall": overall,
        "trend_improving": trend_improving,
        "ma200": ma200,
        "volume_ratio": volume_ratio,
        "obv_trend": obv_trend,
        "volume_score": volume_score,
        "volume_combined_score": volume_combined_score,
    }


def calculate_trend_score_simple(close: pd.Series, ma200_current: float) -> float:
    """Simplified trend score for historical comparison."""
    if len(close) < 50:
        return 50
    
    try:
        current = float(close.iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        
        if pd.isna(current) or pd.isna(ma50) or pd.isna(ma200_current):
            return 50
        
        score = 50
        if current > ma50:
            score += 15
        if current > ma200_current:
            score += 15
        
        return min(100, max(0, score))
    except:
        return 50


# ============================================
# STOCK SELECTION BY STRATEGY
# ============================================


def select_stocks_for_strategy(
    strategy_name: str,
    strategy_config: Dict,
    market_df: pd.DataFrame,
    historical_data: pd.DataFrame,
    date: datetime,
    current_holdings: set = None
) -> List[Dict]:
    """
    Select stocks meeting a strategy's entry criteria.
    
    Returns list of dicts with ticker, entry price, metrics.
    """
    entry_criteria = strategy_config.get("entry", {})
    portfolio_size = strategy_config.get("portfolio_size", 10)
    
    candidates = []
    
    for _, row in market_df.iterrows():
        ticker = row.get('ticker', '')
        
        # Get historical metrics at this date
        ohlcv = get_ticker_ohlcv(historical_data, ticker)
        metrics = calculate_historical_metrics(ticker, ohlcv, market_df, date)
        
        if not metrics:
            continue
        
        # Check entry criteria
        if not check_entry_criteria(metrics, entry_criteria, strategy_name):
            continue
        
        candidates.append({
            "ticker": ticker,
            "name": row.get('name', ticker),
            "sector": row.get('sector', 'Unknown'),
            "entry_price": metrics['price'],
            "trend_score": metrics['trend_score'],
            "entry_trend_score": metrics['trend_score'],
            "dist_52w": metrics['dist_52w'],
            "dist_200dma": metrics['dist_200dma'],
            "quality": metrics['quality'],
            "value": metrics['value'],
            "growth": metrics['growth'],
            "overall": metrics['overall'],
        })
    
    # Sort by appropriate metric based on strategy
    if "Momentum" in strategy_name:
        candidates.sort(key=lambda x: x['trend_score'], reverse=True)
    elif strategy_name == "GARP":
        candidates.sort(key=lambda x: x['overall'], reverse=True)
    elif strategy_name == "TrueValue":
        candidates.sort(key=lambda x: x['value'], reverse=True)
    else:
        candidates.sort(key=lambda x: x['overall'], reverse=True)
    
    # Exclude already held stocks if provided
    if current_holdings:
        candidates = [c for c in candidates if c['ticker'] not in current_holdings]
    
    return candidates[:portfolio_size]


def check_entry_criteria(metrics: Dict, entry: Dict, strategy_name: str) -> bool:
    """Check if stock meets entry criteria for a strategy."""
    
    # Trend score range
    if "trend_score_min" in entry:
        if metrics['trend_score'] < entry['trend_score_min']:
            return False
    if "trend_score_max" in entry:
        if metrics['trend_score'] > entry['trend_score_max']:
            return False
    
    # Distance to 52W high (ATH buffer)
    if "dist_52w_min" in entry:
        if metrics['dist_52w'] < entry['dist_52w_min']:
            return False
    if "dist_52w_max" in entry:
        if metrics['dist_52w'] > entry['dist_52w_max']:
            return False
    
    # Distance to 200DMA
    if "dist_200dma_min" in entry:
        if metrics['dist_200dma'] < entry['dist_200dma_min']:
            return False
    
    # Quality filter
    if "quality_min" in entry:
        if metrics['quality'] < entry['quality_min']:
            return False
    
    # Value filter
    if "value_min" in entry:
        if metrics['value'] < entry['value_min']:
            return False
    
    # Growth filter
    if "growth_min" in entry:
        if metrics['growth'] < entry['growth_min']:
            return False
    
    # Overall filter
    if "overall_min" in entry:
        if metrics['overall'] < entry['overall_min']:
            return False
    
    # Trend improving (for TrueValue)
    if entry.get("trend_improving") and not metrics.get('trend_improving', False):
        return False
    
    # Volume ratio filter (for P5)
    if "volume_ratio_min" in entry:
        if metrics.get('volume_ratio', 1.0) < entry['volume_ratio_min']:
            return False
            
    # Combined Volume Score filter (for P2/P5)
    if "volume_combined_min" in entry:
        if metrics.get('volume_combined_score', 5) < entry['volume_combined_min']:
            return False
    
    # OBV trend filter (reject if in distribution when accumulation required)
    if entry.get("obv_trend_required"):
        obv_trend = metrics.get('obv_trend', 'NEUTRAL')
        if obv_trend == "DISTRIBUTION":
            return False  # Don't enter during distribution
    
    return True


def check_exit_criteria(
    holding: Dict,
    current_price: float,
    metrics: Dict,
    exit_config: Dict,
    current_date: datetime = None
) -> Tuple[bool, str]:
    """
    Check if a holding should be exited.
    
    Returns (should_exit, exit_reason)
    """
    entry_price = holding['entry_price']
    peak_price = holding.get('peak_price', entry_price)
    
    # Calculate returns
    return_pct = ((current_price / entry_price) - 1) * 100
    drawdown_from_peak = ((current_price / peak_price) - 1) * 100
    
    # Stop loss
    stop_loss = exit_config.get("stop_loss_pct", -15)
    if return_pct <= stop_loss:
        return True, "STOP_LOSS"
    
    # Trailing stop
    trailing_stop = exit_config.get("trailing_stop_pct")
    if trailing_stop and drawdown_from_peak <= trailing_stop:
        return True, "TRAILING_STOP"
    
    # Trend score exit
    trend_exit = exit_config.get("trend_score_max")
    if trend_exit and metrics.get('trend_score', 50) <= trend_exit:
        return True, "TREND_REVERSAL"
    
    # Overall score exit
    overall_exit = exit_config.get("overall_max")
    if overall_exit and metrics.get('overall', 5) <= overall_exit:
        return True, "SCORE_DECLINE"
    
    # Value exit (for TrueValue - stock no longer cheap)
    value_exit = exit_config.get("value_max")
    if value_exit and metrics.get('value', 5) <= value_exit:
        return True, "VALUE_EXIT"
    
    # OBV Distribution Exit (for P5)
    if exit_config.get("obv_distribution_exit"):
        obv_trend = metrics.get('obv_trend', 'NEUTRAL')
        if obv_trend == "DISTRIBUTION":
            return True, "OBV_DISTRIBUTION"
    
    # Time-based exit (stagnant position)
    time_exit_days = exit_config.get("time_exit_days")
    if time_exit_days and current_date and 'entry_date' in holding:
        try:
            entry_date = holding['entry_date']
            if hasattr(entry_date, 'to_pydatetime'):
                entry_date = entry_date.to_pydatetime()
            if hasattr(current_date, 'to_pydatetime'):
                current_date = current_date.to_pydatetime()
            
            days_held = (current_date - entry_date).days
            min_return = exit_config.get("time_exit_min_return", 2.0)
            max_loss = exit_config.get("time_exit_max_loss", -5.0)
            
            # Exit if held too long with weak return, but not if already stop-loss territory
            if days_held >= time_exit_days and return_pct < min_return and return_pct > max_loss:
                return True, "TIME_STAGNANT"
        except:
            pass
    
    return False, None


def check_partial_profit(
    holding: Dict,
    current_price: float,
    exit_config: Dict
) -> Tuple[bool, float]:
    """
    Check if holding qualifies for partial profit taking.
    
    Returns (should_take_partial, sell_portion)
    """
    if holding.get('partial_taken', False):
        return False, 0
    
    partial_pct = exit_config.get("partial_profit_pct")
    if not partial_pct:
        return False, 0
    
    entry_price = holding['entry_price']
    return_pct = ((current_price / entry_price) - 1) * 100
    
    if return_pct >= partial_pct:
        sell_portion = exit_config.get("partial_sell_pct", 0.33)
        return True, sell_portion
    
    return False, 0


# ============================================
# MAIN BACKTEST ENGINE
# ============================================

def run_proposal_backtest(
    proposal_key: str,
    market_df: pd.DataFrame,
    rebalance_freq: str = "bi-weekly",
    lookback_months: int = 6,
    initial_capital: float = 200000,
    progress_callback: Callable = None
) -> Dict:
    """
    Run backtest for a single proposal.
    
    Args:
        proposal_key: Key from ALL_PROPOSALS
        market_df: Current market data
        rebalance_freq: "bi-weekly" or "monthly"
        lookback_months: Backtest period
        initial_capital: Starting capital (₹)
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dict with backtest results
    """
    proposal = get_proposal(proposal_key)
    rebalance_days = REBALANCE_FREQUENCIES.get(rebalance_freq, 14)
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_months * 30)
    
    # Get all tickers
    tickers = market_df['ticker'].tolist()
    
    # Fetch historical data
    historical_data = fetch_historical_data(tickers, start_date, end_date)
    
    if historical_data.empty:
        return {"error": "Failed to fetch historical data"}
    
    # Fetch Nifty 50 for regime detection (if needed)
    nifty_data = pd.DataFrame()
    if proposal.get("regime_based"):
        nifty_data = fetch_nifty50_data(days=400)
    
    # Generate rebalance dates
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=f'{rebalance_days}D')
    
    # Initialize tracking
    equity_curve = []
    trades = []
    portfolio_history = []
    holdings = {}  # strategy_name -> {ticker -> holding_dict}
    cash = initial_capital
    
    strategies = proposal.get("strategies", {})
    
    for i, rebal_date in enumerate(rebalance_dates):
        if progress_callback:
            progress_callback((i + 1) / len(rebalance_dates))
        
        # Get allocations (may vary by regime for P4)
        if proposal.get("regime_based"):
            regime = detect_regime_for_date(nifty_data, rebal_date)
            allocations = get_regime_allocations(regime, proposal)
        else:
            allocations = {name: cfg.get("allocation", 0.5) for name, cfg in strategies.items()}
        
        # Calculate Total Portfolio Value (Cash + Current Value of All Holdings)
        # This fixes the "no compounding" bug where allocation was based on fixed initial_capital
        current_holdings_value = 0
        
        # Pre-calculate prices for all holdings to get accurate equity
        # (We need this anyway for exits, so it's not wasted)
        holding_prices = {} # (strategy, ticker) -> price
        
        for s_name, s_holdings in holdings.items():
            for ticker in s_holdings:
                ohlcv = get_ticker_ohlcv(historical_data, ticker)
                # Quick fetch of price at this date
                price_data = ohlcv[ohlcv.index <= rebal_date]['Close']
                if not price_data.empty:
                    p = float(price_data.iloc[-1])
                    holding_prices[(s_name, ticker)] = p
                    current_holdings_value += s_holdings[ticker]['shares'] * p
        
        total_equity = cash + current_holdings_value
        
        # Process each strategy
        for strategy_name, strategy_config in strategies.items():
            if strategy_name not in holdings:
                holdings[strategy_name] = {}
            
            strategy_holdings = holdings[strategy_name]
            exit_config = strategy_config.get("exit", {})
            
            # --- CHECK EXITS ---
            to_sell = []
            partial_sells = []
            
            for ticker, holding in list(strategy_holdings.items()):
                # Use pre-calculated price or fetch if missing (rare)
                current_price = holding_prices.get((strategy_name, ticker))
                if not current_price:
                    ohlcv = get_ticker_ohlcv(historical_data, ticker)
                    metrics = calculate_historical_metrics(ticker, ohlcv, market_df, rebal_date)
                    if metrics:
                         current_price = metrics['price']
                    else:
                        continue
                
                # We still need full metrics for exit decisions
                ohlcv = get_ticker_ohlcv(historical_data, ticker)
                metrics = calculate_historical_metrics(ticker, ohlcv, market_df, rebal_date)
                
                if not metrics:
                    continue
                    
                # Ensure price consistency
                metrics['price'] = current_price # Should match, but safety first
                
                # Update peak price
                holding['peak_price'] = max(holding.get('peak_price', holding['entry_price']), current_price)
                
                # Check for partial profit first
                should_partial, sell_portion = check_partial_profit(holding, current_price, exit_config)
                if should_partial:
                    shares_to_sell = int(holding['shares'] * sell_portion)
                    if shares_to_sell > 0:
                        return_pct = ((current_price / holding['entry_price']) - 1) * 100
                        partial_sells.append({
                            "ticker": ticker,
                            "exit_price": current_price,
                            "exit_reason": "PARTIAL_PROFIT",
                            "return_pct": return_pct,
                            "shares": shares_to_sell
                        })
                        holding['shares'] -= shares_to_sell
                        holding['partial_taken'] = True
                
                # Check full exit criteria (pass current date for time-based exit)
                should_exit, reason = check_exit_criteria(holding, current_price, metrics, exit_config, rebal_date)
                
                if should_exit:
                    return_pct = ((current_price / holding['entry_price']) - 1) * 100
                    to_sell.append({
                        "ticker": ticker,
                        "exit_price": current_price,
                        "exit_reason": reason,
                        "return_pct": return_pct,
                        "shares": holding['shares'],
                        "entry_date": holding.get('entry_date'),
                        "quality": holding.get('quality', 5),
                        "value": holding.get('value', 5),
                        "growth": holding.get('growth', 5),
                        "trend_score": holding.get('entry_trend_score', 50),
                        "sector": holding.get('sector', 'Unknown')
                    })
            
            # Execute partial sells first
            for sell in partial_sells:
                ticker = sell['ticker']
                sale_value = sell['shares'] * sell['exit_price']
                cash += sale_value
                
                trades.append({
                    "strategy": strategy_name,
                    "ticker": ticker,
                    "action": "SELL",
                    "date": rebal_date,
                    "price": sell['exit_price'],
                    "shares": sell['shares'],
                    "value": sale_value,
                    "reason": sell['exit_reason'],
                    "return_pct": sell['return_pct'],
                    "partial": True
                })
            
            # Execute full sells
            for sell in to_sell:
                ticker = sell['ticker']
                holding = strategy_holdings.pop(ticker)
                sale_value = sell['shares'] * sell['exit_price']
                cash += sale_value
                
                trades.append({
                    "strategy": strategy_name,
                    "ticker": ticker,
                    "action": "SELL",
                    "date": rebal_date,
                    "price": sell['exit_price'],
                    "shares": sell['shares'],
                    "value": sale_value,
                    "reason": sell['exit_reason'],
                    "return_pct": sell['return_pct'],
                    "entry_date": sell.get('entry_date'),
                    "quality": sell.get('quality', 5),
                    "value_score": sell.get('value_score', 5), # Corrected key
                    "growth": sell.get('growth', 5),
                    "trend_score": sell.get('trend_score', 50),
                    "sector": sell.get('sector', 'Unknown'),
                    "partial": False
                })
            
            # --- SELECT NEW STOCKS ---
            allocation = allocations.get(strategy_name, 0.33)
            # FIX: Use Total Equity for compounding, not static initial capital
            target_capital = total_equity * allocation
            
            # Calculate current holdings value
            holdings_value = 0
            for ticker, holding in strategy_holdings.items():
                ohlcv = get_ticker_ohlcv(historical_data, ticker)
                metrics = calculate_historical_metrics(ticker, ohlcv, market_df, rebal_date)
                if metrics:
                    holdings_value += holding['shares'] * metrics['price']
            
            # Calculate how much to allocate to new picks
            available_for_new = min(cash, target_capital - holdings_value)
            
            if available_for_new > 1000:  # Minimum allocation
                current_tickers = set(strategy_holdings.keys())
                new_picks = select_stocks_for_strategy(
                    strategy_name, strategy_config, market_df,
                    historical_data, rebal_date, current_tickers
                )
                
                # Allocate evenly across new picks
                open_slots = strategy_config.get("portfolio_size", 10) - len(strategy_holdings)
                
                if new_picks and open_slots > 0:
                    per_stock = available_for_new / min(len(new_picks), open_slots)
                    
                    for pick in new_picks[:open_slots]:
                        ticker = pick['ticker']
                        price = pick['entry_price']
                        
                        # Guard against NaN or zero price
                        if pd.isna(price) or price <= 0:
                            continue
                        
                        shares = int(per_stock / price)
                        
                        if shares <= 0:
                            continue
                        
                        cost = shares * price
                        if cost > cash:
                            continue
                        
                        cash -= cost
                        
                        strategy_holdings[ticker] = {
                            "shares": shares,
                            "entry_price": price,
                            "entry_date": rebal_date,
                            "peak_price": price,
                            "entry_trend_score": pick['trend_score'],
                            "sector": pick['sector']
                        }
                        
                        trades.append({
                            "strategy": strategy_name,
                            "ticker": ticker,
                            "action": "BUY",
                            "date": rebal_date,
                            "price": price,
                            "shares": shares,
                            "value": cost,
                            "reason": "ENTRY_CRITERIA",
                            "trend_score": pick['trend_score']
                        })
            
        # Calculate total holdings value
        holdings_val = 0
        for strat_holdings in holdings.values():
            for ticker, holding in strat_holdings.items():
                ohlcv = get_ticker_ohlcv(historical_data, ticker)
                metrics = calculate_historical_metrics(ticker, ohlcv, market_df, rebal_date)
                if metrics:
                    holdings_val += holding['shares'] * metrics['price']

        equity_curve.append({
            "date": rebal_date,
            "portfolio_value": cash + holdings_val,
            "cash": cash,
            "num_holdings": sum(len(h) for h in holdings.values())
        })
        
        portfolio_history.append({
            "date": rebal_date,
            "holdings": {s: list(h.keys()) for s, h in holdings.items()},
            "portfolio_value": equity_curve[-1]['portfolio_value']
        })
    
    # Calculate metrics
    equity_df = pd.DataFrame(equity_curve)
    
    if equity_df.empty:
        return {"error": "No backtest data generated"}
    
    metrics = calculate_backtest_metrics(equity_df, trades, initial_capital, start_date, end_date)
    
    # Build detailed open positions with current prices
    open_positions = []
    for strategy_name, strat_holdings in holdings.items():
        for ticker, holding in strat_holdings.items():
            # Get current market price (end of backtest)
            ohlcv = get_ticker_ohlcv(historical_data, ticker)
            current_metrics = calculate_historical_metrics(ticker, ohlcv, market_df, end_date)
            
            if current_metrics:
                current_price = current_metrics['price']
                entry_price = holding['entry_price']
                shares = holding['shares']
                
                unrealized_pnl = (current_price - entry_price) * shares
                unrealized_pct = ((current_price / entry_price) - 1) * 100
                
                # Calculate days held
                entry_date = holding.get('entry_date')
                if entry_date:
                    try:
                        if hasattr(entry_date, 'to_pydatetime'):
                            entry_date = entry_date.to_pydatetime()
                        days_held = (end_date - entry_date).days
                    except:
                        days_held = 0
                else:
                    days_held = 0
                
                open_positions.append({
                    "ticker": ticker,
                    "strategy": strategy_name,
                    "shares": shares,
                    "entry_price": round(entry_price, 2),
                    "current_price": round(current_price, 2),
                    "entry_date": str(entry_date)[:10] if entry_date else "N/A",
                    "days_held": days_held,
                    "cost_basis": round(entry_price * shares, 2),
                    "market_value": round(current_price * shares, 2),
                    "unrealized_pnl": round(unrealized_pnl, 2),
                    "unrealized_pct": round(unrealized_pct, 2),
                    "entry_trend_score": holding.get('entry_trend_score', 0),
                    "sector": holding.get('sector', 'Unknown'),
                    "peak_price": round(holding.get('peak_price', entry_price), 2),
                    "drawdown_from_peak": round(((current_price / holding.get('peak_price', entry_price)) - 1) * 100, 2),
                    "partial_taken": holding.get('partial_taken', False)
                })
    
    # Sort by unrealized P&L
    open_positions.sort(key=lambda x: x['unrealized_pct'], reverse=True)
    
    # Calculate summary stats for open positions
    total_cost_basis = sum(p['cost_basis'] for p in open_positions)
    total_market_value = sum(p['market_value'] for p in open_positions)
    total_unrealized_pnl = sum(p['unrealized_pnl'] for p in open_positions)
    
    open_positions_summary = {
        "count": len(open_positions),
        "total_cost_basis": round(total_cost_basis, 2),
        "total_market_value": round(total_market_value, 2),
        "total_unrealized_pnl": round(total_unrealized_pnl, 2),
        "avg_unrealized_pct": round(total_unrealized_pnl / total_cost_basis * 100, 2) if total_cost_basis > 0 else 0,
        "winners": len([p for p in open_positions if p['unrealized_pct'] > 0]),
        "losers": len([p for p in open_positions if p['unrealized_pct'] < 0]),
        "avg_days_held": round(sum(p['days_held'] for p in open_positions) / len(open_positions), 1) if open_positions else 0
    }
    
    # Factor analytics from closed trades
    factor_performance = analyze_factor_performance(trades)
    heatmaps = generate_all_heatmaps(trades)
    exit_analysis = analyze_exit_reasons(trades)
    sector_analysis = analyze_sector_performance(trades)
    
    return {
        "proposal": proposal_key,
        "proposal_name": proposal.get("name", proposal_key),
        "rebalance_freq": rebalance_freq,
        "backtest_start": start_date.strftime("%Y-%m-%d"),
        "backtest_end": end_date.strftime("%Y-%m-%d"),
        "equity_curve": equity_df,
        "trades": trades,
        "metrics": metrics,
        "portfolio_history": portfolio_history,
        "final_holdings": holdings,
        # New detailed outputs
        "open_positions": open_positions,
        "open_positions_summary": open_positions_summary,
        "factor_performance": factor_performance,
        "heatmaps": heatmaps,
        "exit_analysis": exit_analysis,
        "sector_analysis": sector_analysis
    }


def calculate_backtest_metrics(
    equity_df: pd.DataFrame,
    trades: List[Dict],
    initial_capital: float,
    start_date: datetime,
    end_date: datetime
) -> Dict:
    """Calculate performance metrics from backtest results."""
    
    start_val = initial_capital
    end_val = equity_df['portfolio_value'].iloc[-1]
    
    total_return = ((end_val / start_val) - 1) * 100
    
    # Benchmark approximation (assume 3% for 6 months = 6% annual)
    months = (end_date - start_date).days / 30
    benchmark_return = 3 * (months / 6)  # Simple approximation
    
    alpha = total_return - benchmark_return
    
    # Daily returns for Sharpe
    daily_returns = equity_df['portfolio_value'].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 0 and daily_returns.std() > 0 else 0
    
    # Max drawdown
    rolling_max = equity_df['portfolio_value'].cummax()
    drawdown = (equity_df['portfolio_value'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    # Win rate
    sell_trades = [t for t in trades if t['action'] == 'SELL' and 'return_pct' in t]
    wins = len([t for t in sell_trades if t['return_pct'] > 0])
    win_rate = (wins / len(sell_trades) * 100) if sell_trades else 0
    
    # Average win/loss
    win_returns = [t['return_pct'] for t in sell_trades if t['return_pct'] > 0]
    loss_returns = [t['return_pct'] for t in sell_trades if t['return_pct'] < 0]
    avg_win = np.mean(win_returns) if win_returns else 0
    avg_loss = np.mean(loss_returns) if loss_returns else 0
    
    return {
        "total_return_pct": round(total_return, 2),
        "benchmark_return_pct": round(benchmark_return, 2),
        "alpha": round(alpha, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "win_rate_pct": round(win_rate, 1),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "total_trades": len(trades),
        "final_value": round(end_val, 2),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d")
    }


# ============================================
# COMPARISON RUNNER
# ============================================

def run_all_proposals_comparison(
    market_df: pd.DataFrame,
    progress_callback: Callable = None
) -> Dict[str, Dict]:
    """
    Run backtests for all proposals with both rebalancing frequencies.
    
    Returns:
        Dict with results for each proposal_freq combination
    """
    results = {}
    
    proposal_keys = get_all_proposal_keys()
    frequencies = list(REBALANCE_FREQUENCIES.keys())
    
    total_runs = len(proposal_keys) * len(frequencies)
    current_run = 0
    
    for proposal_key in proposal_keys:
        for freq in frequencies:
            if progress_callback:
                progress_callback(current_run / total_runs, f"Testing {proposal_key} ({freq})")
            
            result_key = f"{proposal_key}_{freq}"
            results[result_key] = run_proposal_backtest(
                proposal_key=proposal_key,
                market_df=market_df,
                rebalance_freq=freq
            )
            
            current_run += 1
    
    return results


def get_best_proposal(comparison_results: Dict) -> Tuple[str, Dict]:
    """
    Find the best performing proposal based on alpha and risk-adjusted returns.
    
    Returns:
        Tuple of (result_key, result_dict)
    """
    best_key = None
    best_score = -float('inf')
    
    for key, result in comparison_results.items():
        if 'error' in result:
            continue
        
        metrics = result.get('metrics', {})
        
        # Score: Alpha + (Sharpe * 2) - (Max Drawdown / 2)
        score = (
            metrics.get('alpha', 0) + 
            metrics.get('sharpe_ratio', 0) * 2 -
            abs(metrics.get('max_drawdown_pct', 0)) / 2
        )
        
        if score > best_score:
            best_score = score
            best_key = key
    
    return best_key, comparison_results.get(best_key, {})


# ============================================
# FACTOR ATTRIBUTION ANALYTICS
# ============================================

def get_score_bucket(score: float, buckets: List[Tuple[int, int, str]] = None) -> str:
    """Get bucket label for a score."""
    if buckets is None:
        buckets = [(0, 4, "0-4"), (5, 7, "5-7"), (8, 10, "8-10")]
    
    for low, high, label in buckets:
        if low <= score <= high:
            return label
    return "Unknown"


def get_trend_bucket(trend_score: float) -> str:
    """Get bucket label for trend score."""
    if trend_score < 60:
        return "<60"
    elif trend_score < 70:
        return "60-70"
    elif trend_score < 80:
        return "70-80"
    elif trend_score < 85:
        return "80-85"
    else:
        return "85+"


def analyze_factor_performance(trades: List[Dict]) -> Dict[str, pd.DataFrame]:
    """
    Analyze trade performance by factor score buckets.
    
    Returns:
        Dict with DataFrames for each factor (Quality, Value, Growth, Trend)
    """
    if not trades:
        return {}
    
    # Filter to sell trades only (they have return_pct)
    sell_trades = [t for t in trades if t.get('action') == 'SELL' and 'return_pct' in t]
    
    if not sell_trades:
        return {}
    
    df = pd.DataFrame(sell_trades)
    results = {}
    
    # Factor buckets
    factors = ['quality', 'value', 'growth']
    
    for factor in factors:
        if factor in df.columns:
            df[f'{factor}_bucket'] = df[factor].apply(
                lambda x: get_score_bucket(x) if pd.notna(x) else "Unknown"
            )
            
            agg = df.groupby(f'{factor}_bucket').agg({
                'return_pct': ['mean', 'count', lambda x: (x > 0).mean() * 100]
            }).round(2)
            agg.columns = ['Avg Return %', 'Trade Count', 'Win Rate %']
            results[factor.capitalize()] = agg
    
    # Trend score buckets
    if 'trend_score' in df.columns:
        df['trend_bucket'] = df['trend_score'].apply(
            lambda x: get_trend_bucket(x) if pd.notna(x) else "Unknown"
        )
        
        agg = df.groupby('trend_bucket').agg({
            'return_pct': ['mean', 'count', lambda x: (x > 0).mean() * 100]
        }).round(2)
        agg.columns = ['Avg Return %', 'Trade Count', 'Win Rate %']
        results['Trend Score'] = agg
    
    return results


def generate_heatmap_data(trades: List[Dict], factor: str = "quality") -> pd.DataFrame:
    """
    Generate Trend Score x Factor heatmap data.
    
    Args:
        trades: List of trade dicts
        factor: "quality", "value", or "growth"
    
    Returns:
        DataFrame with avg return for each Trend/Factor combination
    """
    if not trades:
        return pd.DataFrame()
    
    sell_trades = [t for t in trades if t.get('action') == 'SELL' and 'return_pct' in t]
    
    if not sell_trades:
        return pd.DataFrame()
    
    df = pd.DataFrame(sell_trades)
    
    if 'trend_score' not in df.columns or factor not in df.columns:
        return pd.DataFrame()
    
    df['trend_bucket'] = df['trend_score'].apply(get_trend_bucket)
    df[f'{factor}_bucket'] = df[factor].apply(lambda x: get_score_bucket(x) if pd.notna(x) else "Unknown")
    
    # Create pivot table
    pivot = df.pivot_table(
        values='return_pct',
        index=f'{factor}_bucket',
        columns='trend_bucket',
        aggfunc='mean'
    ).round(2)
    
    return pivot


def generate_all_heatmaps(trades: List[Dict]) -> Dict[str, pd.DataFrame]:
    """
    Generate all Trend vs Pillar heatmaps.
    
    Returns:
        Dict with:
        - "Trend vs Quality": DataFrame
        - "Trend vs Value": DataFrame
        - "Trend vs Growth": DataFrame
    """
    return {
        "Trend vs Quality": generate_heatmap_data(trades, "quality"),
        "Trend vs Value": generate_heatmap_data(trades, "value"),
        "Trend vs Growth": generate_heatmap_data(trades, "growth")
    }


def analyze_exit_reasons(trades: List[Dict]) -> pd.DataFrame:
    """
    Analyze performance by exit reason.
    
    Returns:
        DataFrame with stats per exit reason
    """
    if not trades:
        return pd.DataFrame()
    
    sell_trades = [t for t in trades if t.get('action') == 'SELL']
    
    if not sell_trades:
        return pd.DataFrame()
    
    df = pd.DataFrame(sell_trades)
    
    if 'reason' not in df.columns:
        return pd.DataFrame()
    
    # Calculate days held if we have entry info
    if 'entry_date' in df.columns and 'date' in df.columns:
        df['days_held'] = (pd.to_datetime(df['date']) - pd.to_datetime(df['entry_date'])).dt.days
    else:
        df['days_held'] = 0
    
    agg = df.groupby('reason').agg({
        'return_pct': ['mean', 'count', lambda x: (x > 0).mean() * 100],
        'days_held': 'mean'
    }).round(2)
    
    agg.columns = ['Avg Return %', 'Count', 'Win Rate %', 'Avg Days Held']
    
    return agg.sort_values('Count', ascending=False)


def analyze_sector_performance(trades: List[Dict]) -> pd.DataFrame:
    """
    Analyze performance by sector.
    
    Returns:
        DataFrame with stats per sector
    """
    if not trades:
        return pd.DataFrame()
    
    sell_trades = [t for t in trades if t.get('action') == 'SELL' and 'return_pct' in t]
    
    if not sell_trades:
        return pd.DataFrame()
    
    df = pd.DataFrame(sell_trades)
    
    if 'sector' not in df.columns:
        return pd.DataFrame()
    
    agg = df.groupby('sector').agg({
        'return_pct': ['mean', 'count', lambda x: (x > 0).mean() * 100]
    }).round(2)
    
    agg.columns = ['Avg Return %', 'Trade Count', 'Win Rate %']
    
    return agg.sort_values('Avg Return %', ascending=False)

