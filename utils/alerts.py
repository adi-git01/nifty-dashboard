
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

ALERTS_FILE = "alerts.json"

def load_alerts() -> List[Dict[str, Any]]:
    """Load alerts from JSON file."""
    if os.path.exists(ALERTS_FILE):
        try:
            with open(ALERTS_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_alerts(alerts: List[Dict[str, Any]]) -> None:
    """Save alerts to JSON file."""
    with open(ALERTS_FILE, 'w') as f:
        json.dump(alerts, f, indent=2)

def add_alert(
    ticker: str, 
    metric: str, 
    condition: str, 
    threshold: float,
    stop_loss: Optional[float] = None,
    target: Optional[float] = None,
    notes: str = ""
) -> Dict[str, Any]:
    """
    Add a new alert rule.
    
    Args:
        ticker: Stock ticker (e.g. "HDFCBANK.NS")
        metric: One of "trend_score", "overall", "quality", "value", "growth", "momentum", "price"
        condition: One of ">", "<", ">=", "<="
        threshold: Numeric threshold value
        stop_loss: Optional stop loss price - triggers alert if price falls below
        target: Optional target price - triggers alert if price rises above
        notes: Fundamental notes/investment thesis
    """
    alerts = load_alerts()
    
    new_alert = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "ticker": ticker.upper(),
        "metric": metric,
        "condition": condition,
        "threshold": threshold,
        "stop_loss": stop_loss,
        "target": target,
        "notes": notes,
        "entry_price": None,  # Set when tracking starts
        "created": datetime.now().isoformat(),
        "triggered": False,
        "last_triggered": None,
        "trigger_type": None  # "METRIC", "STOP_LOSS", "TARGET"
    }
    
    alerts.append(new_alert)
    save_alerts(alerts)
    return new_alert


def add_price_alert(
    ticker: str,
    entry_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    target: Optional[float] = None,
    notes: str = ""
) -> Dict[str, Any]:
    """
    Add a simple price-based alert with stop loss and target.
    
    Args:
        ticker: Stock ticker
        entry_price: Price at entry (for P&L tracking)
        stop_loss: Stop loss price
        target: Target price
        notes: Fundamental thesis or notes
    """
    alerts = load_alerts()
    
    new_alert = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "ticker": ticker.upper(),
        "metric": "price",  # Special type for price alerts
        "condition": "range",  # Between SL and Target
        "threshold": entry_price,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "target": target,
        "notes": notes,
        "created": datetime.now().isoformat(),
        "triggered": False,
        "last_triggered": None,
        "trigger_type": None
    }
    
    alerts.append(new_alert)
    save_alerts(alerts)
    return new_alert


def update_alert_notes(alert_id: str, notes: str) -> bool:
    """Update notes for an existing alert."""
    alerts = load_alerts()
    for alert in alerts:
        if alert.get('id') == alert_id:
            alert['notes'] = notes
            save_alerts(alerts)
            return True
    return False


def remove_alert(alert_id: str) -> None:
    """Remove an alert by ID."""
    alerts = load_alerts()
    alerts = [a for a in alerts if a.get('id') != alert_id]
    save_alerts(alerts)


def check_alerts(market_data_df) -> List[Dict[str, Any]]:
    """
    Check all alerts against current market data.
    
    Checks:
    1. Metric thresholds (trend_score, quality, etc.)
    2. Stop loss triggers
    3. Target triggers
    
    Args:
        market_data_df: DataFrame with columns for ticker and metrics
        
    Returns:
        List of triggered alerts with current values
    """
    alerts = load_alerts()
    triggered = []
    
    for alert in alerts:
        ticker = alert['ticker']
        
        # Find the stock in data
        row = market_data_df[market_data_df['ticker'] == ticker]
        
        if row.empty:
            continue
        
        stock_data = row.iloc[0]
        current_price = stock_data.get('price')
        
        # === CHECK STOP LOSS ===
        stop_loss = alert.get('stop_loss')
        if stop_loss and current_price and current_price <= stop_loss:
            triggered.append({
                **alert,
                'current_value': current_price,
                'trigger_type': 'STOP_LOSS',
                'alert_message': f"🔴 STOP LOSS HIT! Price ₹{current_price:.2f} ≤ ₹{stop_loss:.2f}"
            })
            alert['triggered'] = True
            alert['last_triggered'] = datetime.now().isoformat()
            alert['trigger_type'] = 'STOP_LOSS'
            continue  # Stop loss takes priority
        
        # === CHECK TARGET ===
        target = alert.get('target')
        if target and current_price and current_price >= target:
            triggered.append({
                **alert,
                'current_value': current_price,
                'trigger_type': 'TARGET',
                'alert_message': f"🎯 TARGET HIT! Price ₹{current_price:.2f} ≥ ₹{target:.2f}"
            })
            alert['triggered'] = True
            alert['last_triggered'] = datetime.now().isoformat()
            alert['trigger_type'] = 'TARGET'
            continue  # Target achieved
        
        # === CHECK METRIC THRESHOLD ===
        metric = alert.get('metric')
        condition = alert.get('condition')
        threshold = alert.get('threshold')
        
        if metric == 'price' and condition == 'range':
            # Price range alert - already handled by SL/Target above
            continue
        
        current_value = stock_data.get(metric)
        
        if current_value is None:
            continue
        
        # Check condition
        is_triggered = False
        if condition == ">" and current_value > threshold:
            is_triggered = True
        elif condition == "<" and current_value < threshold:
            is_triggered = True
        elif condition == ">=" and current_value >= threshold:
            is_triggered = True
        elif condition == "<=" and current_value <= threshold:
            is_triggered = True
        
        if is_triggered:
            triggered.append({
                **alert,
                'current_value': current_value,
                'trigger_type': 'METRIC',
                'alert_message': f"📊 {metric.title()} is {current_value:.1f} ({condition} {threshold})"
            })
            
            # Update alert as triggered
            alert['triggered'] = True
            alert['last_triggered'] = datetime.now().isoformat()
            alert['trigger_type'] = 'METRIC'
    
    # Save updated alerts
    save_alerts(alerts)
    
    return triggered


def get_alert_display_text(alert: Dict[str, Any]) -> str:
    """Generate human-readable text for an alert."""
    ticker = alert['ticker'].replace('.NS', '').replace('.BO', '')
    
    parts = [ticker]
    
    # Metric condition
    metric = alert.get('metric', '')
    if metric and metric != 'price':
        condition = alert.get('condition', '')
        threshold = alert.get('threshold', '')
        parts.append(f"{metric.replace('_', ' ').title()} {condition} {threshold}")
    
    # Stop loss
    if alert.get('stop_loss'):
        parts.append(f"SL: ₹{alert['stop_loss']:.0f}")
    
    # Target
    if alert.get('target'):
        parts.append(f"T: ₹{alert['target']:.0f}")
    
    return " | ".join(parts)


def get_triggered_alert_text(alert: Dict[str, Any]) -> str:
    """Generate notification text for a triggered alert."""
    ticker = alert['ticker'].replace('.NS', '').replace('.BO', '')
    
    # Use custom message if available
    if 'alert_message' in alert:
        return f"🔔 {ticker}: {alert['alert_message']}"
    
    metric = alert['metric'].replace('_', ' ').title()
    current = alert.get('current_value', 'N/A')
    trigger_type = alert.get('trigger_type', 'METRIC')
    
    if isinstance(current, float):
        current = f"{current:.1f}"
    
    if trigger_type == 'STOP_LOSS':
        return f"🔴 {ticker}: STOP LOSS triggered at ₹{current}"
    elif trigger_type == 'TARGET':
        return f"🎯 {ticker}: TARGET reached at ₹{current}"
    else:
        condition = alert['condition']
        threshold = alert['threshold']
        return f"🔔 {ticker}: {metric} is {current} ({condition} {threshold})"


def get_alerts_with_pnl(market_data_df) -> List[Dict[str, Any]]:
    """Get all alerts with current P&L calculated."""
    alerts = load_alerts()
    
    for alert in alerts:
        ticker = alert['ticker']
        row = market_data_df[market_data_df['ticker'] == ticker]
        
        if not row.empty:
            current_price = row.iloc[0].get('price')
            entry_price = alert.get('entry_price')
            
            alert['current_price'] = current_price
            
            if entry_price and current_price:
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                alert['pnl_pct'] = pnl_pct
            else:
                alert['pnl_pct'] = None
                
            # Distance to stop loss
            stop_loss = alert.get('stop_loss')
            if stop_loss and current_price:
                alert['dist_to_sl'] = ((current_price - stop_loss) / current_price) * 100
            
            # Distance to target
            target = alert.get('target')
            if target and current_price:
                alert['dist_to_target'] = ((target - current_price) / current_price) * 100
    
    return alerts
