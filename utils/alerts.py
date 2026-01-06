
import json
import os
from datetime import datetime

ALERTS_FILE = "alerts.json"

def load_alerts():
    """Load alerts from JSON file."""
    if os.path.exists(ALERTS_FILE):
        try:
            with open(ALERTS_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_alerts(alerts):
    """Save alerts to JSON file."""
    with open(ALERTS_FILE, 'w') as f:
        json.dump(alerts, f, indent=2)

def add_alert(ticker, metric, condition, threshold):
    """
    Add a new alert rule.
    
    Args:
        ticker: Stock ticker (e.g. "HDFCBANK.NS")
        metric: One of "trend_score", "overall", "quality", "value", "growth", "momentum"
        condition: One of ">", "<", ">=", "<="
        threshold: Numeric threshold value
    """
    alerts = load_alerts()
    
    new_alert = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "ticker": ticker.upper(),
        "metric": metric,
        "condition": condition,
        "threshold": threshold,
        "created": datetime.now().isoformat(),
        "triggered": False,
        "last_triggered": None
    }
    
    alerts.append(new_alert)
    save_alerts(alerts)
    return new_alert

def remove_alert(alert_id):
    """Remove an alert by ID."""
    alerts = load_alerts()
    alerts = [a for a in alerts if a.get('id') != alert_id]
    save_alerts(alerts)

def check_alerts(market_data_df):
    """
    Check all alerts against current market data.
    
    Args:
        market_data_df: DataFrame with columns for ticker and metrics
        
    Returns:
        List of triggered alerts with current values
    """
    alerts = load_alerts()
    triggered = []
    
    for alert in alerts:
        ticker = alert['ticker']
        metric = alert['metric']
        condition = alert['condition']
        threshold = alert['threshold']
        
        # Find the stock in data
        row = market_data_df[market_data_df['ticker'] == ticker]
        
        if row.empty:
            continue
            
        current_value = row.iloc[0].get(metric)
        
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
                'current_value': current_value
            })
            
            # Update alert as triggered
            alert['triggered'] = True
            alert['last_triggered'] = datetime.now().isoformat()
    
    # Save updated alerts
    save_alerts(alerts)
    
    return triggered

def get_alert_display_text(alert):
    """Generate human-readable text for an alert."""
    ticker = alert['ticker'].replace('.NS', '').replace('.BO', '')
    metric = alert['metric'].replace('_', ' ').title()
    condition = alert['condition']
    threshold = alert['threshold']
    
    return f"{ticker}: {metric} {condition} {threshold}"

def get_triggered_alert_text(alert):
    """Generate notification text for a triggered alert."""
    ticker = alert['ticker'].replace('.NS', '').replace('.BO', '')
    metric = alert['metric'].replace('_', ' ').title()
    current = alert.get('current_value', 'N/A')
    condition = alert['condition']
    threshold = alert['threshold']
    
    if isinstance(current, float):
        current = f"{current:.1f}"
    
    return f"🔔 {ticker}: {metric} is {current} ({condition} {threshold})"
