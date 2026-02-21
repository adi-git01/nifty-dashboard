"""
Custom Backtest Runner - Multi-Timeframe Strategies
===================================================
Run backtests based on conditions derived from Strategy Learner v3 analysis.

Usage:
    python custom_backtest.py --strategy 3mo
    python custom_backtest.py --strategy 6mo
    python custom_backtest.py --strategy 12mo
    python custom_backtest.py --all
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import sys
import io
import argparse

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from utils.nifty500_list import TICKERS
from utils.data_engine import batch_fetch_tickers
from utils.scoring import calculate_scores, calculate_trend_metrics
from utils.volume_analysis import get_combined_volume_signal

# ==========================================
# STRATEGY CONFIGURATIONS
# Based on Strategy Learner v3 Analysis
# ==========================================

STRATEGY_CONFIGS = {
    "3mo": {
        "name": "Short-Term Momentum",
        "hold_days": 65,
        "trailing_stop": -15.0,
        "entry": {
            "min_momentum": 8.0,
            "min_trend_score": 60,
            "min_quality": 5.0,
            "min_growth": 5.0,
            "min_volume_score": 7.0,  # Accumulation
            "above_ma50": True,
        },
        "exit": {
            "trend_score_exit": 40,
            "trend_exit_days": 5,
        },
        "position_size": 0.05,  # 5% per position
        "max_positions": 10,
        "rebalance_days": 5,  # Weekly
        "expected_return": 21.8,
        "expected_dd": -16.3,
    },
    "6mo": {
        "name": "Medium-Term Trend Following",
        "hold_days": 130,
        "trailing_stop": -20.0,
        "entry": {
            "min_momentum": 7.0,
            "min_trend_score": 55,
            "min_quality": 6.0,
            "min_growth": 6.0,
            "min_volume_score": 8.0,  # Strong Accumulation
            "above_ma50": True,
            "min_dist_200dma": 10.0,
        },
        "exit": {
            "trend_score_exit": 45,
            "trend_exit_days": 10,
        },
        "position_size": 0.04,  # 4% per position
        "max_positions": 12,
        "rebalance_days": 10,  # Bi-weekly
        "expected_return": 36.6,
        "expected_dd": -19.1,
    },
    "12mo": {
        "name": "Long-Term Multi-Bagger",
        "hold_days": 250,
        "trailing_stop": -25.0,
        "entry": {
            "min_momentum": 6.0,
            "min_trend_score": 50,
            "min_quality": 5.5,
            "min_growth": 6.5,
            "min_volume_score": 6.0,  # Moderate Accumulation
            "above_ma50": True,
            "min_days_above_ma50": 20,
        },
        "exit": {
            "trend_score_exit": 35,
            "trend_exit_days": 15,
        },
        "position_size": 0.03,  # 3% per position
        "max_positions": 15,
        "rebalance_days": 20,  # Monthly
        "expected_return": 69.9,
        "expected_dd": -29.7,
    },
}

# Priority sectors based on winner analysis
PRIORITY_SECTORS = [
    "Financial Services",
    "Banking", 
    "Metals & Mining",
    "Auto",
    "Pharma & Healthcare",
]


def calculate_trend_score(ohlcv: pd.DataFrame) -> Optional[int]:
    """Calculate trend score for current state."""
    if len(ohlcv) < 252:
        return None
    
    try:
        close = ohlcv['Close']
        current_price = float(close.iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
        
        high_52 = ohlcv['High'].iloc[-252:].max()
        low_52 = ohlcv['Low'].iloc[-252:].min()
        
        score = 50
        
        if current_price > ma50:
            score += 15
        else:
            score -= 10
            
        if current_price > ma200:
            score += 15
        else:
            score -= 15
            
        if ma50 > ma200:
            score += 10
        else:
            score -= 5
        
        if high_52 > low_52:
            pos = (current_price - low_52) / (high_52 - low_52)
            score += int((pos - 0.5) * 30)
            
            dist_52w = ((current_price - high_52) / high_52) * 100
            if dist_52w > -5:
                score += 10
            elif dist_52w < -30:
                score -= 10
        
        return max(0, min(100, int(score)))
    except:
        return None


def calculate_volume_ratio(volume: pd.Series, lookback: int = 20) -> float:
    """Calculate current volume vs average."""
    if len(volume) < lookback:
        return 1.0
    return volume.iloc[-1] / volume.iloc[-lookback:].mean()


def check_entry_conditions(
    ticker: str,
    ohlcv: pd.DataFrame,
    fundamentals: Dict,
    config: Dict
) -> Tuple[bool, Dict]:
    """Check if stock meets entry conditions for the strategy."""
    entry = config["entry"]
    reasons = []
    
    # Check fundamentals
    momentum = fundamentals.get("momentum", 0)
    quality = fundamentals.get("quality", 0)
    growth = fundamentals.get("growth", 0)
    
    if momentum < entry["min_momentum"]:
        reasons.append(f"momentum {momentum:.1f} < {entry['min_momentum']}")
    if quality < entry["min_quality"]:
        reasons.append(f"quality {quality:.1f} < {entry['min_quality']}")
    if growth < entry["min_growth"]:
        reasons.append(f"growth {growth:.1f} < {entry['min_growth']}")
    
    # Check trend score
    trend_score = calculate_trend_score(ohlcv)
    if trend_score is None:
        reasons.append("insufficient data for trend")
    elif trend_score < entry["min_trend_score"]:
        reasons.append(f"trend {trend_score} < {entry['min_trend_score']}")
    
    # Check MA50
    if entry.get("above_ma50", False) and len(ohlcv) >= 50:
        ma50 = ohlcv['Close'].rolling(50).mean().iloc[-1]
        current = ohlcv['Close'].iloc[-1]
        if current < ma50:
            reasons.append("below MA50")
    
    # Check volume score (VPT + A/D)
    if len(ohlcv) >= 20 and 'Volume' in ohlcv.columns:
        try:
            vol_sig = get_combined_volume_signal(
                ohlcv['High'], ohlcv['Low'], ohlcv['Close'], ohlcv['Volume']
            )
            vol_score = vol_sig['combined_score']
            if vol_score < entry.get("min_volume_score", 5.0):
                reasons.append(f"volume_score {vol_score} < {entry['min_volume_score']}")
        except:
             reasons.append("volume analysis failed")
    
    # Check 200DMA distance if required
    if "min_dist_200dma" in entry and len(ohlcv) >= 200:
        ma200 = ohlcv['Close'].rolling(200).mean().iloc[-1]
        current = ohlcv['Close'].iloc[-1]
        dist = ((current - ma200) / ma200) * 100
        if dist < entry["min_dist_200dma"]:
            reasons.append(f"dist_200dma {dist:.1f}% < {entry['min_dist_200dma']}%")
    
    passed = len(reasons) == 0
    
    return passed, {
        "ticker": ticker,
        "passed": passed,
        "trend_score": trend_score,
        "momentum": momentum,
        "quality": quality,
        "growth": growth,
        "sector": fundamentals.get("sector", "Unknown"),
        "reasons": reasons if not passed else [],
    }


def screen_candidates(strategy_key: str) -> pd.DataFrame:
    """Screen Nifty 500 for strategy candidates."""
    config = STRATEGY_CONFIGS[strategy_key]
    
    print(f"\n{'='*70}")
    print(f"SCREENING FOR: {config['name']} ({strategy_key})")
    print(f"{'='*70}")
    
    # Load current market data
    print("[DATA] Loading market fundamentals...")
    market_df = batch_fetch_tickers(TICKERS)
    
    from utils.historical_tracker import calculate_multi_period_returns
    scored_data = []
    for _, row in market_df.iterrows():
        data = row.to_dict()
        returns, _ = calculate_multi_period_returns(row['ticker'])
        data.update(returns)
        trends = calculate_trend_metrics(data)
        data.update(trends)
        scores = calculate_scores(data)
        data.update(scores)
        scored_data.append(data)
    market_df = pd.DataFrame(scored_data)
    
    # Get historical data for technical analysis
    print("[DATA] Fetching historical prices...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)  # ~1.5 years for 200DMA
    
    try:
        prices = yf.download(
            TICKERS,
            start=start_date,
            end=end_date,
            interval="1d",
            progress=True,
            auto_adjust=True,
            group_by='ticker'
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        return pd.DataFrame()
    
    # Screen each stock
    print(f"\n[SCREENING] Checking {len(TICKERS)} stocks...")
    candidates = []
    
    for ticker in TICKERS:
        try:
            # Get OHLCV
            if len(TICKERS) == 1:
                ohlcv = prices
            else:
                if ticker not in prices.columns.get_level_values(0):
                    continue
                ohlcv = prices[ticker].dropna()
            
            if len(ohlcv) < 200:
                continue
            
            # Get fundamentals
            row = market_df[market_df['ticker'] == ticker]
            if row.empty:
                continue
            fundamentals = row.iloc[0].to_dict()
            
            # Check conditions
            passed, result = check_entry_conditions(ticker, ohlcv, fundamentals, config)
            
            if passed:
                # Add current price for reference
                result["current_price"] = float(ohlcv['Close'].iloc[-1])
                result["ma50"] = float(ohlcv['Close'].rolling(50).mean().iloc[-1])
                result["ma200"] = float(ohlcv['Close'].rolling(200).mean().iloc[-1])
                
                # Volume metrics
                if 'Volume' in ohlcv.columns:
                    vol_sig = get_combined_volume_signal(
                        ohlcv['High'], ohlcv['Low'], ohlcv['Close'], ohlcv['Volume']
                    )
                    result["volume_score"] = vol_sig['combined_score']
                    result["volume_signal"] = vol_sig['combined_signal']
                    result["avg_volume"] = int(ohlcv['Volume'].iloc[-20:].mean())
                
                candidates.append(result)
        except Exception as e:
            continue
    
    # Create DataFrame and rank
    if not candidates:
        print("[RESULT] No candidates found matching criteria")
        return pd.DataFrame()
    
    df = pd.DataFrame(candidates)
    
    # Rank by momentum + trend score
    df["rank_score"] = df["momentum"] * 0.4 + df["trend_score"] * 0.3 + df["quality"] * 0.15 + df["growth"] * 0.15
    df = df.sort_values("rank_score", ascending=False)
    
    # Add sector priority ranking
    df["sector_priority"] = df["sector"].apply(
        lambda x: PRIORITY_SECTORS.index(x) if x in PRIORITY_SECTORS else 99
    )
    
    return df


def generate_trade_signals(strategy_key: str) -> None:
    """Generate trade signals for a strategy."""
    config = STRATEGY_CONFIGS[strategy_key]
    
    # Screen candidates
    candidates = screen_candidates(strategy_key)
    
    if candidates.empty:
        return
    
    # Display top candidates
    max_positions = config["max_positions"]
    top = candidates.head(max_positions * 2)  # Show 2x for selection
    
    print(f"\n{'='*70}")
    print(f"TOP CANDIDATES FOR {config['name'].upper()}")
    print(f"{'='*70}")
    print(f"Strategy: {strategy_key}")
    print(f"Hold Period: {config['hold_days']} days")
    print(f"Trailing Stop: {config['trailing_stop']}%")
    print(f"Position Size: {config['position_size']*100:.0f}%")
    print(f"Max Positions: {max_positions}")
    print()
    
    display_cols = ["ticker", "sector", "trend_score", "momentum", "quality", "growth", "current_price", "volume_score"]
    display_cols = [c for c in display_cols if c in top.columns]
    
    print(top[display_cols].to_string(index=False))
    
    # Save to CSV
    output_file = f"trade_signals_{strategy_key}_{datetime.now().strftime('%Y%m%d')}.csv"
    candidates.to_csv(output_file, index=False)
    print(f"\n[SAVED] {output_file}")
    
    # Position sizing calculation
    print(f"\n{'='*70}")
    print("POSITION SIZING (assuming ₹10,00,000 portfolio)")
    print(f"{'='*70}")
    
    portfolio_value = 1000000
    risk_per_trade = 0.02  # 2% risk per trade
    
    for i, row in top.head(max_positions).iterrows():
        stop_loss = config["trailing_stop"]
        position_value = (portfolio_value * risk_per_trade) / abs(stop_loss / 100)
        position_value = min(position_value, portfolio_value * 0.10)  # Cap at 10%
        shares = int(position_value / row["current_price"])
        
        print(f"{row['ticker']:15} | ₹{position_value:,.0f} | {shares} shares @ ₹{row['current_price']:.2f}")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("STRATEGY EXPECTATIONS")
    print(f"{'='*70}")
    print(f"Expected Return: {config['expected_return']:.1f}%")
    print(f"Expected Max DD: {config['expected_dd']:.1f}%")
    print(f"Recovery Time:   {config['hold_days'] // 3} days avg")
    
    # Entry timing advice
    print(f"\n{'='*70}")
    print("ENTRY TIMING ADVICE")
    print(f"{'='*70}")
    
    if strategy_key == "3mo":
        print("• Returns evenly distributed - can enter anytime")
        print("• Early phase delivers ~14% avg")
        print("• Tight trailing stop (-15%) is acceptable")
        
    elif strategy_key == "6mo":
        print("• Early phase only delivers ~11% avg")
        print("• Mid phase is best (+32%) - consider scaling in")
        print("• Wait for initial shakeout if possible")
        
    elif strategy_key == "12mo":
        print("• Early phase is nearly flat (+4%) - BE PATIENT")
        print("• Mid phase delivers bulk of gains (+39%)")
        print("• Consider adding to position after 1-2 months")
        print("• Use wider -25% stop to stay in winners")


def run_all_strategies():
    """Run screening for all strategies."""
    print("\n" + "="*70)
    print("MULTI-TIMEFRAME STRATEGY SCREENING")
    print("="*70)
    print("Based on Strategy Learner v3 Analysis of Top 50 Winners")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()
    
    for strategy_key in ["3mo", "6mo", "12mo"]:
        generate_trade_signals(strategy_key)
        print("\n" + "-"*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Custom Backtest Runner")
    parser.add_argument("--strategy", choices=["3mo", "6mo", "12mo", "all"], 
                       default="all", help="Strategy to run")
    args = parser.parse_args()
    
    if args.strategy == "all":
        run_all_strategies()
    else:
        generate_trade_signals(args.strategy)


if __name__ == "__main__":
    main()
