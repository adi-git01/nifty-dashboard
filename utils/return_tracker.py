"""
Return Tracker Module
=====================
Tracks stocks picked from the Trend Scanner to measure real-world returns.
Persists data to tracker_data.json for continuity.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional

TRACKER_FILE = "tracker_data.json"

def _load_tracker_data() -> Dict:
    """Load tracker data from JSON file."""
    if os.path.exists(TRACKER_FILE):
        try:
            with open(TRACKER_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        "tracked_stocks": [],
        "archived_stocks": [],
        "trend_change_history": [],
        "last_notification_date": None
    }

def _save_tracker_data(data: Dict):
    """Save tracker data to JSON file."""
    with open(TRACKER_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def add_to_tracker(ticker: str, entry_price: float, entry_signal: str, 
                   entry_score: float, sector: str = None, name: str = None) -> bool:
    """
    Add a stock to the return tracker.
    
    Args:
        ticker: Stock ticker (e.g., "RELIANCE.NS")
        entry_price: Current price at time of adding
        entry_signal: Trend signal at entry (e.g., "STRONG UPTREND")
        entry_score: Trend score at entry (0-100)
        sector: Stock sector (optional)
        name: Company name (optional)
    
    Returns:
        True if added successfully, False if already tracked
    """
    data = _load_tracker_data()
    
    # Check if already tracked
    existing = [s for s in data["tracked_stocks"] if s["ticker"] == ticker]
    if existing:
        return False
    
    new_entry = {
        "ticker": ticker,
        "entry_price": entry_price,
        "entry_date": datetime.now().isoformat(),
        "entry_trend_signal": entry_signal,
        "entry_trend_score": entry_score,
        "sector": sector or "Unknown",
        "name": name or ticker.replace(".NS", "").replace(".BO", ""),
        "last_known_signal": entry_signal,
        "last_known_score": entry_score
    }
    
    data["tracked_stocks"].append(new_entry)
    _save_tracker_data(data)
    return True

def remove_from_tracker(ticker: str, exit_price: float = None, reason: str = "removed_manually") -> bool:
    """
    Remove a stock from tracking and archive it.
    
    Args:
        ticker: Stock ticker to remove
        exit_price: Current price at exit (optional, for return calculation)
        reason: Why removed (e.g., "removed_manually", "trend_reversed")
    
    Returns:
        True if removed, False if not found
    """
    data = _load_tracker_data()
    
    # Find the stock
    stock = None
    for s in data["tracked_stocks"]:
        if s["ticker"] == ticker:
            stock = s
            break
    
    if not stock:
        return False
    
    # Calculate return if exit price provided
    total_return_pct = None
    if exit_price and stock.get("entry_price"):
        total_return_pct = ((exit_price / stock["entry_price"]) - 1) * 100
    
    # Archive it
    archived = {
        **stock,
        "exit_date": datetime.now().isoformat(),
        "exit_price": exit_price,
        "total_return_pct": total_return_pct,
        "reason": reason
    }
    data["archived_stocks"].append(archived)
    
    # Remove from tracked
    data["tracked_stocks"] = [s for s in data["tracked_stocks"] if s["ticker"] != ticker]
    _save_tracker_data(data)
    return True

def get_all_tracked() -> List[Dict]:
    """Get all currently tracked stocks."""
    data = _load_tracker_data()
    return data.get("tracked_stocks", [])

def get_archived() -> List[Dict]:
    """Get all archived (previously tracked) stocks."""
    data = _load_tracker_data()
    return data.get("archived_stocks", [])

def calculate_current_returns(market_df) -> List[Dict]:
    """
    Calculate current returns for all tracked stocks.
    
    Args:
        market_df: DataFrame with current market data (ticker, price, trend_signal, trend_score)
    
    Returns:
        List of tracked stocks with current returns and trend info
    """
    tracked = get_all_tracked()
    result = []
    
    for stock in tracked:
        ticker = stock["ticker"]
        
        # Find current data
        current = market_df[market_df['ticker'] == ticker]
        
        if current.empty:
            # Stock not in current market data
            result.append({
                **stock,
                "current_price": None,
                "return_pct": None,
                "return_value": None,
                "current_signal": "N/A",
                "current_score": 0,
                "signal_changed": False,
                "days_tracked": _days_since(stock.get("entry_date"))
            })
            continue
        
        current_row = current.iloc[0]
        current_price = current_row.get('price', 0)
        current_signal = current_row.get('trend_signal', 'N/A')
        current_score = current_row.get('trend_score', 0)
        
        entry_price = stock.get("entry_price", current_price)
        return_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        return_value = current_price - entry_price
        
        # Check if signal changed from entry
        entry_signal = stock.get("entry_trend_signal", "")
        signal_changed = entry_signal != current_signal
        
        result.append({
            **stock,
            "current_price": current_price,
            "return_pct": return_pct,
            "return_value": return_value,
            "current_signal": current_signal,
            "current_score": current_score,
            "signal_changed": signal_changed,
            "days_tracked": _days_since(stock.get("entry_date"))
        })
    
    return result

def detect_trend_changes(market_df) -> List[Dict]:
    """
    Detect stocks whose trend signal has changed since entry.
    
    Args:
        market_df: Current market data DataFrame
    
    Returns:
        List of stocks with trend changes
    """
    returns = calculate_current_returns(market_df)
    changes = [r for r in returns if r.get("signal_changed", False)]
    return changes

def record_trend_change(ticker: str, old_signal: str, new_signal: str):
    """Record a trend change event for history."""
    data = _load_tracker_data()
    
    change_event = {
        "ticker": ticker,
        "old_signal": old_signal,
        "new_signal": new_signal,
        "detected_at": datetime.now().isoformat()
    }
    
    data["trend_change_history"].append(change_event)
    
    # Also update last known signal in tracked stocks
    for stock in data["tracked_stocks"]:
        if stock["ticker"] == ticker:
            stock["last_known_signal"] = new_signal
            break
    
    _save_tracker_data(data)

def get_trend_change_history() -> List[Dict]:
    """Get history of all trend changes."""
    data = _load_tracker_data()
    return data.get("trend_change_history", [])

def get_tracker_summary(market_df) -> Dict:
    """
    Get a summary of tracker performance.
    
    Returns:
        Dict with total_tracked, total_return_pct, best_performer, worst_performer, etc.
    """
    returns = calculate_current_returns(market_df)
    
    if not returns:
        return {
            "total_tracked": 0,
            "total_return_pct": 0,
            "total_return_value": 0,
            "best_performer": None,
            "worst_performer": None,
            "trend_changes": 0,
            "positive_count": 0,
            "negative_count": 0
        }
    
    valid_returns = [r for r in returns if r.get("return_pct") is not None]
    
    if not valid_returns:
        return {
            "total_tracked": len(returns),
            "total_return_pct": 0,
            "total_return_value": 0,
            "best_performer": None,
            "worst_performer": None,
            "trend_changes": 0,
            "positive_count": 0,
            "negative_count": 0
        }
    
    # Calculate weighted average return (simple average for now)
    avg_return = sum(r["return_pct"] for r in valid_returns) / len(valid_returns)
    total_value = sum(r.get("return_value", 0) or 0 for r in valid_returns)
    
    # Best and worst
    sorted_by_return = sorted(valid_returns, key=lambda x: x.get("return_pct", 0), reverse=True)
    best = sorted_by_return[0] if sorted_by_return else None
    worst = sorted_by_return[-1] if sorted_by_return else None
    
    # Counts
    positive = len([r for r in valid_returns if r.get("return_pct", 0) > 0])
    negative = len([r for r in valid_returns if r.get("return_pct", 0) < 0])
    changes = len([r for r in returns if r.get("signal_changed", False)])
    
    return {
        "total_tracked": len(returns),
        "total_return_pct": avg_return,
        "total_return_value": total_value,
        "best_performer": best,
        "worst_performer": worst,
        "trend_changes": changes,
        "positive_count": positive,
        "negative_count": negative
    }

def update_last_known_signals(market_df):
    """
    Update last known signals for all tracked stocks.
    Call this to sync after checking for changes.
    """
    data = _load_tracker_data()
    
    for stock in data["tracked_stocks"]:
        ticker = stock["ticker"]
        current = market_df[market_df['ticker'] == ticker]
        
        if not current.empty:
            stock["last_known_signal"] = current.iloc[0].get('trend_signal', stock.get("last_known_signal"))
            stock["last_known_score"] = current.iloc[0].get('trend_score', stock.get("last_known_score"))
    
    _save_tracker_data(data)

def export_weekly_summary(market_df) -> Dict:
    """
    Export data formatted for weekly email summary.
    
    Returns:
        Dict with all data needed for email template
    """
    summary = get_tracker_summary(market_df)
    returns = calculate_current_returns(market_df)
    changes = detect_trend_changes(market_df)
    
    # Sort returns for top/bottom performers
    sorted_returns = sorted(
        [r for r in returns if r.get("return_pct") is not None],
        key=lambda x: x.get("return_pct", 0),
        reverse=True
    )
    
    return {
        "summary": summary,
        "all_returns": sorted_returns,
        "top_performers": sorted_returns[:3] if len(sorted_returns) >= 3 else sorted_returns,
        "bottom_performers": sorted_returns[-3:][::-1] if len(sorted_returns) >= 3 else [],
        "trend_changes": changes,
        "generated_at": datetime.now().isoformat()
    }

def _days_since(date_str: str) -> int:
    """Calculate days since a given date string."""
    if not date_str:
        return 0
    try:
        entry_date = datetime.fromisoformat(date_str)
        return (datetime.now() - entry_date).days
    except:
        return 0

def clear_all_tracked():
    """Clear all tracked stocks (for testing/reset)."""
    data = _load_tracker_data()
    data["tracked_stocks"] = []
    _save_tracker_data(data)

def is_tracked(ticker: str) -> bool:
    """Check if a ticker is currently being tracked."""
    tracked = get_all_tracked()
    return any(s["ticker"] == ticker for s in tracked)
