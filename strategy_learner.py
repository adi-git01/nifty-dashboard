"""
Strategy Learner Analysis
=========================
Analyzes historical winners to learn what makes stocks successful
and whether our scoring strategies would have caught them early.

Outputs: strategy_learner_results.csv

Run: python strategy_learner.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

# Import from existing modules
from utils.nifty500_list import TICKERS
from utils.data_engine import batch_fetch_tickers
from utils.scoring import calculate_scores, calculate_trend_metrics

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "top_n": 25,              # Top 25 performers to analyze
    "lookback_months": 12,    # 12-month returns
    "output_file": "strategy_learner_results.csv",
    "summary_file": "strategy_learner_summary.csv",
}

# Strategy definitions (matching main.py presets)
STRATEGIES = {
    "Strong Momentum (Top 20%)": {
        "filter": lambda row, threshold: row.get("trend_score", 0) >= threshold,
        "threshold_type": "percentile_80"
    },
    "Quality at Reasonable Price": {
        "filter": lambda row: row.get("overall", 0) >= 6 and row.get("value", 0) >= 6,
    },
    "Breakout Candidates": {
        "filter": lambda row: row.get("dist_52w", -100) > -5,
    },
    "Turnaround Plays": {
        "filter": lambda row: row.get("momentum", 0) >= 5 and row.get("overall", 0) < 5,
    }
}


# ==========================================
# CORE FUNCTIONS
# ==========================================

def get_top_performers(n: int = 25, months: int = 12) -> pd.DataFrame:
    """
    Identify top N performers by actual price returns over the lookback period.
    """
    print(f"[DATA] Fetching Nifty 500 data to identify top {n} performers over {months} months...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)
    
    # Fetch historical prices for all tickers
    print(f"   Downloading price data from {start_date.date()} to {end_date.date()}...")
    
    try:
        # Download all at once for efficiency
        prices = yf.download(TICKERS, start=start_date, end=end_date, 
                            interval="1d", progress=True, auto_adjust=True)['Close']
    except Exception as e:
        print(f"   Error downloading prices: {e}")
        return pd.DataFrame()
    
    # Calculate returns
    returns = []
    for ticker in TICKERS:
        try:
            if ticker in prices.columns:
                col = prices[ticker].dropna()
                if len(col) >= 50:  # Need enough data
                    start_price = col.iloc[0]
                    end_price = col.iloc[-1]
                    ret = ((end_price - start_price) / start_price) * 100
                    returns.append({
                        "ticker": ticker,
                        "start_price": start_price,
                        "end_price": end_price,
                        "return_pct": ret
                    })
        except:
            continue
    
    df = pd.DataFrame(returns)
    df = df.sort_values("return_pct", ascending=False).head(n)
    
    print(f"   [OK] Found {len(df)} top performers")
    print(f"   Top 5: {df['ticker'].head().tolist()}")
    
    return df


def identify_rally_start(ohlcv: pd.DataFrame) -> Optional[datetime]:
    """
    Find the date when the major uptrend began.
    Uses 50 DMA crossover as the primary signal.
    """
    if len(ohlcv) < 60:
        return None
    
    close = ohlcv['Close']
    ma50 = close.rolling(50).mean()
    
    # Find first date where price crosses above 50 DMA and stays above
    # Look for sustained crossover (at least 5 days above)
    above_ma = close > ma50
    
    # Find the earliest sustained crossover
    for i in range(50, len(above_ma) - 10):
        if not above_ma.iloc[i-1] and above_ma.iloc[i]:
            # Crossed above - check if it stays above for next 5 days
            if above_ma.iloc[i:i+5].all():
                return ohlcv.index[i]
    
    # Fallback: return date at 20% from start
    return ohlcv.index[len(ohlcv) // 5] if len(ohlcv) > 5 else ohlcv.index[0]


def calculate_trend_score_at_date(ohlcv: pd.DataFrame, target_date: datetime) -> Dict:
    """
    Calculate trend score at a specific historical date.
    Uses only price data available up to that date.
    """
    # Filter data up to target date
    data = ohlcv[ohlcv.index <= target_date].copy()
    
    if len(data) < 200:
        return {"trend_score": None, "dist_52w": None, "dist_200dma": None}
    
    try:
        close = data['Close']
        current_price = float(close.iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
        
        # Calculate 52W high/low from available data
        high_52 = data['High'].iloc[-252:].max() if 'High' in data.columns else close.iloc[-252:].max()
        low_52 = data['Low'].iloc[-252:].min() if 'Low' in data.columns else close.iloc[-252:].min()
        
        # Build score (simplified version of scoring.py logic)
        score = 50
        
        # MA Position
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
        
        # 52W Position
        if high_52 > low_52:
            pos = (current_price - low_52) / (high_52 - low_52)
            score += int((pos - 0.5) * 30)
            
            dist_52w = ((current_price - high_52) / high_52) * 100
            if dist_52w > -5:
                score += 10
            elif dist_52w < -30:
                score -= 10
        else:
            dist_52w = 0
        
        # Clamp
        score = max(0, min(100, score))
        
        # 200 DMA distance
        dist_200dma = ((current_price - ma200) / ma200) * 100 if ma200 > 0 else 0
        
        return {
            "trend_score": int(score),
            "dist_52w": round(dist_52w, 2),
            "dist_200dma": round(dist_200dma, 2),
            "price": round(current_price, 2)
        }
    except Exception as e:
        return {"trend_score": None, "dist_52w": None, "dist_200dma": None}


def get_score_trajectory(ticker: str, ohlcv: pd.DataFrame, rally_start: datetime) -> Dict:
    """
    Calculate trend scores at multiple points in the rally.
    """
    trajectory = {}
    
    # Define checkpoints
    checkpoints = {
        "rally_start": rally_start,
        "plus_1mo": rally_start + timedelta(days=30),
        "plus_3mo": rally_start + timedelta(days=90),
        "plus_6mo": rally_start + timedelta(days=180),
        "current": ohlcv.index[-1] if len(ohlcv) > 0 else datetime.now()
    }
    
    for label, date in checkpoints.items():
        if date <= ohlcv.index[-1]:
            scores = calculate_trend_score_at_date(ohlcv, date)
            trajectory[label] = {
                "date": date.strftime("%Y-%m-%d"),
                **scores
            }
        else:
            trajectory[label] = {"date": None, "trend_score": None}
    
    return trajectory


def calculate_score_velocity(trajectory: Dict) -> float:
    """
    Calculate the rate of score change per month.
    """
    scores = []
    dates = []
    
    for label in ["rally_start", "plus_1mo", "plus_3mo", "plus_6mo", "current"]:
        if label in trajectory and trajectory[label].get("trend_score") is not None:
            scores.append(trajectory[label]["trend_score"])
            dates.append(trajectory[label]["date"])
    
    if len(scores) < 2:
        return 0.0
    
    # Simple: (last - first) / months elapsed
    first_date = datetime.strptime(dates[0], "%Y-%m-%d")
    last_date = datetime.strptime(dates[-1], "%Y-%m-%d")
    months = max(1, (last_date - first_date).days / 30)
    
    velocity = (scores[-1] - scores[0]) / months
    return round(velocity, 2)


def check_strategy_hits(trajectory: Dict, current_fundamentals: Dict) -> Dict:
    """
    Check which strategies would have caught this stock and when.
    """
    hits = {}
    
    # For each checkpoint, simulate if strategies would have selected
    for strategy_name, strategy in STRATEGIES.items():
        hit_info = {"hit": False, "first_hit_checkpoint": None, "current_hit": False}
        
        for label in ["rally_start", "plus_1mo", "plus_3mo", "plus_6mo", "current"]:
            if label not in trajectory or trajectory[label].get("trend_score") is None:
                continue
            
            # Build a mock row for strategy evaluation
            mock_row = {
                "trend_score": trajectory[label]["trend_score"],
                "dist_52w": trajectory[label].get("dist_52w", -50),
                # Use current fundamentals (we can't reconstruct historical ones)
                "overall": current_fundamentals.get("overall", 5),
                "quality": current_fundamentals.get("quality", 5),
                "value": current_fundamentals.get("value", 5),
                "growth": current_fundamentals.get("growth", 5),
                "momentum": current_fundamentals.get("momentum", 5),
            }
            
            # Test strategy
            try:
                if strategy_name == "Strong Momentum (Top 20%)":
                    # Use 70 as threshold (approx 80th percentile)
                    if mock_row["trend_score"] >= 70:
                        if not hit_info["hit"]:
                            hit_info["hit"] = True
                            hit_info["first_hit_checkpoint"] = label
                elif strategy_name == "Quality at Reasonable Price":
                    if mock_row["overall"] >= 6 and mock_row["value"] >= 6:
                        if not hit_info["hit"]:
                            hit_info["hit"] = True
                            hit_info["first_hit_checkpoint"] = label
                elif strategy_name == "Breakout Candidates":
                    if mock_row["dist_52w"] > -5:
                        if not hit_info["hit"]:
                            hit_info["hit"] = True
                            hit_info["first_hit_checkpoint"] = label
                elif strategy_name == "Turnaround Plays":
                    if mock_row["momentum"] >= 5 and mock_row["overall"] < 5:
                        if not hit_info["hit"]:
                            hit_info["hit"] = True
                            hit_info["first_hit_checkpoint"] = label
                            
                # Track if it hits now
                if label == "current":
                    hit_info["current_hit"] = hit_info["hit"]
                    
            except Exception as e:
                continue
        
        hits[strategy_name] = hit_info
    
    return hits


def get_current_fundamentals(ticker: str, market_df: pd.DataFrame) -> Dict:
    """
    Get current fundamental scores for a stock.
    """
    if market_df.empty:
        return {}
    
    row = market_df[market_df['ticker'] == ticker]
    if row.empty:
        return {}
    
    row = row.iloc[0]
    return {
        "name": row.get("name", ticker),
        "sector": row.get("sector", "Unknown"),
        "overall": row.get("overall", 5),
        "quality": row.get("quality", 5),
        "value": row.get("value", 5),
        "growth": row.get("growth", 5),
        "momentum": row.get("momentum", 5),
        "current_trend_score": row.get("trend_score", 50),
        "current_price": row.get("price", 0),
    }


def run_analysis():
    """
    Main analysis function.
    """
    print("\n" + "="*60)
    print("STRATEGY LEARNER ANALYSIS")
    print("="*60)
    print(f"Analyzing top {CONFIG['top_n']} performers over {CONFIG['lookback_months']} months")
    print(f"Output: {CONFIG['output_file']}")
    print()
    
    # Step 1: Get top performers
    top_df = get_top_performers(n=CONFIG['top_n'], months=CONFIG['lookback_months'])
    if top_df.empty:
        print("[ERROR] Failed to get top performers")
        return
    
    # Step 2: Get current market data for fundamentals
    print("\n[DATA] Loading current market data for fundamentals...")
    market_df = batch_fetch_tickers(TICKERS)
    
    # Calculate scores for market data
    from utils.historical_tracker import calculate_multi_period_returns
    scored_data = []
    for _, row in market_df.iterrows():
        data = row.to_dict()
        returns = calculate_multi_period_returns(row['ticker'])
        data.update(returns)
        trends = calculate_trend_metrics(data)
        data.update(trends)
        scores = calculate_scores(data)
        data.update(scores)
        scored_data.append(data)
    market_df = pd.DataFrame(scored_data)
    
    # Step 3: Fetch detailed historical data for top performers
    print("\n[DATA] Fetching detailed historical data for winners...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=CONFIG['lookback_months'] * 30 + 200)  # Extra for MAs
    
    top_tickers = top_df['ticker'].tolist()
    
    try:
        historical = yf.download(
            top_tickers, 
            start=start_date, 
            end=end_date,
            interval="1d",
            progress=True,
            auto_adjust=True,
            group_by='ticker'
        )
    except Exception as e:
        print(f"[ERROR] Error fetching historical data: {e}")
        return
    
    # Step 4: Analyze each winner
    print("\n[ANALYSIS] Analyzing each winner...")
    results = []
    
    for idx, row in top_df.iterrows():
        ticker = row['ticker']
        print(f"   [{idx+1}/{len(top_df)}] Analyzing {ticker}...")
        
        try:
            # Get OHLCV for this ticker
            if len(top_tickers) == 1:
                ohlcv = historical
            else:
                ohlcv = historical[ticker] if ticker in historical.columns.get_level_values(0) else pd.DataFrame()
            
            if ohlcv.empty or len(ohlcv) < 200:
                print(f"      [WARN] Insufficient data for {ticker}")
                continue
            
            # Get fundamentals
            fundamentals = get_current_fundamentals(ticker, market_df)
            
            # Identify rally start
            rally_start = identify_rally_start(ohlcv)
            
            # Get score trajectory
            trajectory = get_score_trajectory(ticker, ohlcv, rally_start)
            
            # Calculate velocity
            velocity = calculate_score_velocity(trajectory)
            
            # Check strategy hits
            strategy_hits = check_strategy_hits(trajectory, fundamentals)
            
            # Build result row
            result = {
                "ticker": ticker,
                "name": fundamentals.get("name", ticker),
                "sector": fundamentals.get("sector", "Unknown"),
                "return_12mo": round(row['return_pct'], 1),
                "rally_start_date": rally_start.strftime("%Y-%m-%d") if rally_start else None,
                
                # Score trajectory
                "score_at_rally_start": trajectory.get("rally_start", {}).get("trend_score"),
                "score_at_1mo": trajectory.get("plus_1mo", {}).get("trend_score"),
                "score_at_3mo": trajectory.get("plus_3mo", {}).get("trend_score"),
                "score_at_6mo": trajectory.get("plus_6mo", {}).get("trend_score"),
                "score_current": trajectory.get("current", {}).get("trend_score"),
                
                # Score velocity
                "score_velocity": velocity,
                
                # Current fundamentals
                "quality": fundamentals.get("quality"),
                "value": fundamentals.get("value"),
                "growth": fundamentals.get("growth"),
                "momentum": fundamentals.get("momentum"),
                "overall": fundamentals.get("overall"),
                
                # Strategy hits (flatten)
                "strong_momentum_hit": strategy_hits.get("Strong Momentum (Top 20%)", {}).get("hit"),
                "strong_momentum_first": strategy_hits.get("Strong Momentum (Top 20%)", {}).get("first_hit_checkpoint"),
                "quality_price_hit": strategy_hits.get("Quality at Reasonable Price", {}).get("hit"),
                "quality_price_first": strategy_hits.get("Quality at Reasonable Price", {}).get("first_hit_checkpoint"),
                "breakout_hit": strategy_hits.get("Breakout Candidates", {}).get("hit"),
                "breakout_first": strategy_hits.get("Breakout Candidates", {}).get("first_hit_checkpoint"),
                "turnaround_hit": strategy_hits.get("Turnaround Plays", {}).get("hit"),
                "turnaround_first": strategy_hits.get("Turnaround Plays", {}).get("first_hit_checkpoint"),
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"      [ERROR] Error analyzing {ticker}: {e}")
            continue
    
    # Step 5: Create output DataFrames
    results_df = pd.DataFrame(results)
    
    # Step 6: Calculate summary statistics
    print("\n[SUMMARY] Calculating summary statistics...")
    
    summary = {
        "metric": [],
        "value": []
    }
    
    # Average score at rally start
    avg_start_score = results_df["score_at_rally_start"].mean()
    summary["metric"].append("Avg Score at Rally Start")
    summary["value"].append(round(avg_start_score, 1) if pd.notna(avg_start_score) else "N/A")
    
    # Average score velocity
    avg_velocity = results_df["score_velocity"].mean()
    summary["metric"].append("Avg Score Velocity (pts/month)")
    summary["value"].append(round(avg_velocity, 2) if pd.notna(avg_velocity) else "N/A")
    
    # Strategy hit rates
    for col, name in [
        ("strong_momentum_hit", "Strong Momentum Hit Rate"),
        ("quality_price_hit", "Quality at Price Hit Rate"),
        ("breakout_hit", "Breakout Hit Rate"),
        ("turnaround_hit", "Turnaround Hit Rate"),
    ]:
        hit_rate = results_df[col].sum() / len(results_df) * 100 if len(results_df) > 0 else 0
        summary["metric"].append(name)
        summary["value"].append(f"{hit_rate:.0f}%")
    
    # Sector distribution
    sector_counts = results_df["sector"].value_counts().head(5)
    summary["metric"].append("Top Sectors")
    summary["value"].append(", ".join([f"{s}: {c}" for s, c in sector_counts.items()]))
    
    summary_df = pd.DataFrame(summary)
    
    # Step 7: Save outputs
    results_df.to_csv(CONFIG["output_file"], index=False)
    summary_df.to_csv(CONFIG["summary_file"], index=False)
    
    print("\n" + "="*60)
    print("[OK] ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {CONFIG['output_file']}")
    print(f"Summary saved to: {CONFIG['summary_file']}")
    
    # Print key insights
    print("\nKEY INSIGHTS:")
    print(f"   * Average trend score at rally start: {avg_start_score:.0f}")
    print(f"   * Average score velocity: {avg_velocity:.1f} pts/month")
    print(f"   * Winners analyzed: {len(results_df)}")
    
    print("\nSTRATEGY HIT RATES:")
    for col, name in [
        ("strong_momentum_hit", "Strong Momentum"),
        ("quality_price_hit", "Quality at Price"),
        ("breakout_hit", "Breakout Candidates"),
        ("turnaround_hit", "Turnaround Plays"),
    ]:
        hit_rate = results_df[col].sum() / len(results_df) * 100 if len(results_df) > 0 else 0
        early_catches = results_df[results_df[col.replace("_hit", "_first")] == "rally_start"].shape[0]
        print(f"   * {name}: {hit_rate:.0f}% hit rate ({early_catches} caught at rally start)")
    
    print("\nTOP SECTORS AMONG WINNERS:")
    for sector, count in sector_counts.items():
        print(f"   * {sector}: {count} stocks ({count/len(results_df)*100:.0f}%)")
    
    print("\n")
    return results_df, summary_df


if __name__ == "__main__":
    run_analysis()
