"""
Telegram Notifier Module
=========================
Handles Telegram bot integration for sending alerts and notifications.

Setup Instructions:
1. Create a bot via @BotFather on Telegram
2. Get the bot token from BotFather
3. Start a chat with your bot and send /start
4. Get your chat_id from @userinfobot or the API
5. Configure in the dashboard settings

Auto-Trigger:
- Runs as a background check when dashboard loads
- Can be scheduled via cron/Task Scheduler
"""

import requests
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

CONFIG_FILE = "config.json"


def _load_config() -> Dict:
    """Load notification configuration."""
    # Try Streamlit secrets first (for cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'telegram' in st.secrets:
            return {
                'telegram_bot_token': st.secrets.telegram.get('bot_token', ''),
                'telegram_chat_id': st.secrets.telegram.get('chat_id', ''),
                'telegram_enabled': st.secrets.telegram.get('enabled', False),
                'auto_alert_enabled': st.secrets.get('auto_alert', {}).get('enabled', False),
                'gmail_address': st.secrets.get('gmail', {}).get('address', ''),
                'gmail_app_password': st.secrets.get('gmail', {}).get('app_password', ''),
            }
    except:
        pass
    
    # Check Environment Variables (Best for GitHub Actions / Docker)
    if os.environ.get('TELEGRAM_TOKEN') and os.environ.get('TELEGRAM_CHAT_ID'):
        return {
            'telegram_bot_token': os.environ.get('TELEGRAM_TOKEN'),
            'telegram_chat_id': os.environ.get('TELEGRAM_CHAT_ID'),
            'telegram_enabled': True
        }

    # Fall back to local config file
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    
    return {}


def _save_config(config: Dict) -> None:
    """Save configuration to file."""
    print(f"DEBUG: Saving config: {config}")
    try:
        existing = _load_config()
        print(f"DEBUG: Existing config before update: {existing}")
        existing.update(config)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(existing, f, indent=2)
        print("DEBUG: Config saved successfully to", os.path.abspath(CONFIG_FILE))
    except Exception as e:
        print(f"ERROR: Failed to save config: {e}")


# ==========================================
# TELEGRAM NOTIFIER
# ==========================================

def configure_telegram(bot_token: str, chat_id: str) -> bool:
    """
    Configure Telegram bot credentials.
    
    Args:
        bot_token: Bot token from @BotFather
        chat_id: Your chat ID (or group chat ID for negative numbers)
    
    Returns:
        True if configured
    """
    _save_config({
        'telegram_bot_token': bot_token,
        'telegram_chat_id': chat_id,
        'telegram_enabled': True
    })
    return True


def is_telegram_configured() -> bool:
    """Check if Telegram is properly configured."""
    config = _load_config()
    token = config.get('telegram_bot_token', '')
    chat_id = config.get('telegram_chat_id', '')
    enabled = config.get('telegram_enabled', False)
    return bool(token and chat_id and enabled and token != 'YOUR_BOT_TOKEN')


def disable_telegram() -> None:
    """Disable Telegram notifications."""
    _save_config({'telegram_enabled': False})


def send_telegram_message(message: str) -> Tuple[bool, str]:
    """
    Send a message via Telegram bot.
    
    Args:
        message: Text message to send (supports HTML formatting)
    
    Returns:
        (success, message)
    """
    config = _load_config()
    bot_token = config.get('telegram_bot_token', '')
    chat_id = config.get('telegram_chat_id', '')
    
    if not bot_token or not chat_id:
        return False, "Telegram not configured"
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        # Method to send with specific parse mode
        def _attempt_send(mode=None):
            payload = {
                "chat_id": chat_id,
                "text": message,
                "disable_web_page_preview": True
            }
            if mode:
                payload["parse_mode"] = mode
            return requests.post(url, json=payload, timeout=10)

        # Attempt 1: HTML
        response = _attempt_send("HTML")
        
        # Attempt 2: If Bad Request (likely formatting), try Plaintext
        if response.status_code == 400:
            # Try to log the error for debugging
            try:
                print(f"Telegram HTML failed: {response.text}") 
            except: 
                pass
            
            # Retry without parse_mode
            response = _attempt_send(None)
        
        if response.status_code == 200:
            return True, "Message sent successfully"
        else:
            error_data = response.json()
            return False, f"API Error: {error_data.get('description', 'Unknown error')}"
            
    except requests.exceptions.Timeout:
        return False, "Request timed out"
    except Exception as e:
        return False, f"Error: {str(e)}"


def test_telegram() -> Tuple[bool, str]:
    """Send a test message to verify Telegram configuration."""
    test_msg = "🔔 <b>Nifty Dashboard Test</b>\n\n"
    test_msg += f"✅ Telegram is configured correctly!\n"
    test_msg += f"📊 You will receive alerts here.\n"
    test_msg += f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    return send_telegram_message(test_msg)


# ==========================================
# ALERT NOTIFICATIONS
# ==========================================

def send_alert_notification(alert_type: str, ticker: str, message: str, 
                            use_telegram: bool = True, use_email: bool = True) -> Dict[str, bool]:
    """
    Send alert notification to configured channels.
    
    Args:
        alert_type: Type of alert (STOP_LOSS, TARGET, METRIC, TREND_CHANGE)
        ticker: Stock ticker
        message: Alert message
        use_telegram: Send via Telegram
        use_email: Send via Email
    
    Returns:
        Dict with channel: success status
    """
    results = {}
    
    # Format message for Telegram (HTML)
    emoji_map = {
        'STOP_LOSS': '🔴',
        'TARGET': '🎯',
        'ENTRY_ZONE': '🟢',
        'TREND_CHANGE': '⚠️',
        'VOLUME_SPIKE': '📊',
        'MOMENTUM_FADE': '⚡',
        'METRIC': '📈'
    }
    emoji = emoji_map.get(alert_type, '🔔')
    
    telegram_msg = f"{emoji} <b>{ticker}</b> - {alert_type.replace('_', ' ')}\n\n"
    telegram_msg += message
    telegram_msg += f"\n\n🕐 {datetime.now().strftime('%H:%M:%S')}"
    
    if use_telegram and is_telegram_configured():
        success, _ = send_telegram_message(telegram_msg)
        results['telegram'] = success
    
    if use_email:
        # Use existing email notifier
        try:
            from utils.email_notifier import _send_email, is_email_configured
            if is_email_configured():
                subject = f"{emoji} Alert: {ticker} - {alert_type}"
                success, _ = _send_email(subject, f"<pre>{message}</pre>")
                results['email'] = success
        except:
            pass
    
    return results


def send_triggered_alerts(triggered_alerts: List[Dict]) -> Dict[str, int]:
    """
    Send batch of triggered alerts.
    
    Args:
        triggered_alerts: List of triggered alert dicts from check_alerts()
    
    Returns:
        Dict with channel: count of successful sends
    """
    results = {'telegram': 0, 'email': 0}
    
    if not triggered_alerts:
        return results
    
    # Group alerts by type for summary
    for alert in triggered_alerts:
        ticker = alert.get('ticker', '').replace('.NS', '')
        alert_type = alert.get('trigger_type', 'METRIC')
        message = alert.get('alert_message', alert.get('message', ''))
        
        resp = send_alert_notification(alert_type, ticker, message)
        
        if resp.get('telegram'):
            results['telegram'] += 1
        if resp.get('email'):
            results['email'] += 1
    
    return results


# ==========================================
# AUTO-TRIGGER SYSTEM
# ==========================================

def configure_auto_alert(enabled: bool = True, check_interval_minutes: int = 30) -> None:
    """
    Configure auto-alert settings.
    
    Args:
        enabled: Whether auto-alerts are enabled
        check_interval_minutes: How often to check (for reference)
    """
    _save_config({
        'auto_alert_enabled': enabled,
        'auto_alert_interval': check_interval_minutes,
        'auto_alert_last_check': None
    })


def is_auto_alert_enabled() -> bool:
    """Check if auto-alerts are enabled."""
    config = _load_config()
    return config.get('auto_alert_enabled', False)


def should_run_auto_check() -> bool:
    """
    Determine if we should run an auto-check based on last check time.
    Prevents spamming checks on every page load.
    """
    config = _load_config()
    
    if not config.get('auto_alert_enabled', False):
        return False
    
    last_check = config.get('auto_alert_last_check')
    interval = config.get('auto_alert_interval', 30)  # Default 30 minutes
    
    if not last_check:
        return True
    
    try:
        last_dt = datetime.fromisoformat(last_check)
        minutes_since = (datetime.now() - last_dt).total_seconds() / 60
        return minutes_since >= interval
    except:
        return True


def record_auto_check() -> None:
    """Record that an auto-check was performed."""
    _save_config({
        'auto_alert_last_check': datetime.now().isoformat()
    })


def run_auto_alert_check(df, triggered_alerts: List[Dict]) -> Dict[str, int]:
    """
    Run automatic alert check and send notifications if any triggered.
    Called on dashboard load if auto-alerts enabled and interval passed.
    
    Args:
        df: Market data DataFrame
        triggered_alerts: List of triggered alerts from check_alerts()
    
    Returns:
        Count of alerts sent per channel
    """
    if not should_run_auto_check():
        return {'skipped': True}
    
    record_auto_check()
    
    if triggered_alerts:
        return send_triggered_alerts(triggered_alerts)
    
    return send_triggered_alerts(triggered_alerts)


# ==========================================
# MOOD-BASED ALERTS (Market Timing)
# ==========================================

def send_mood_alert(current_score: float, previous_score: float, 
                    send_tg: bool = True, send_email: bool = True) -> Dict[str, bool]:
    """
    Send mood-based market timing alert when score crosses key thresholds.
    
    Triggers:
    - Score drops below 40 (Bearish Entry Signal)
    - Score rises above 65 (Caution - High Mood)
    
    Args:
        current_score: Current avg_trend_score
        previous_score: Previous day's score
        send_tg: Send via Telegram
        send_email: Send via Email
    
    Returns:
        Dict with channel: success status
    """
    results = {'telegram': False, 'email': False}
    
    # Determine if threshold crossed
    alert_type = None
    if current_score < 40 and previous_score >= 40:
        alert_type = "BEARISH_ENTRY"
        emoji = "🔴"
        title = "BEARISH ENTRY SIGNAL"
        action = "Score dropped below 40 - Consider BUYING Midcap, Bank, Nifty 50"
        sectors_buy = "Midcap, Bank, Nifty 50"
        sectors_avoid = "IT (wait for high mood)"
    elif current_score > 65 and previous_score <= 65:
        alert_type = "CAUTION_HIGH"
        emoji = "🟡"
        title = "CAUTION - HIGH MOOD"
        action = "Score crossed above 65 - Consider IT sector, reduce Midcap exposure"
        sectors_buy = "IT"
        sectors_avoid = "Midcap, Small caps"
    elif current_score < 35 and previous_score >= 35:
        alert_type = "EXTREME_LOW"
        emoji = "🔴🔴"
        title = "EXTREME LOW - STRONG BUY SIGNAL"
        action = "Score dropped below 35 - Strong contrarian buy opportunity!"
        sectors_buy = "Midcap, Bank, Nifty 50 (aggressive)"
        sectors_avoid = "IT"
    else:
        return results  # No threshold crossed
    
    # Build message
    msg = f"{emoji} <b>{title}</b>\n\n"
    msg += f"📊 Current Score: <b>{current_score:.0f}</b> (was {previous_score:.0f})\n\n"
    msg += f"📈 {action}\n\n"
    msg += f"✅ <b>BUY:</b> {sectors_buy}\n"
    msg += f"❌ <b>AVOID:</b> {sectors_avoid}\n\n"
    msg += f"⏱️ Optimal Holding: 60-90 days\n"
    msg += f"📅 Signal Based on -0.81 correlation with Midcap returns\n\n"
    msg += f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    # Send via Telegram
    if send_tg and is_telegram_configured():
        success, _ = send_telegram_message(msg)
        results['telegram'] = success
    
    # Send via Email
    if send_email:
        try:
            from utils.email_notifier import _send_email, is_email_configured
            if is_email_configured():
                subject = f"{emoji} Market Timing Alert: {title}"
                email_body = msg.replace('<b>', '<strong>').replace('</b>', '</strong>')
                success, _ = _send_email(subject, f"<pre>{email_body}</pre>")
                results['email'] = success
        except:
            pass
    
    return results


def check_and_send_mood_alerts(df, mood_history) -> Dict:
    """
    Check current mood metrics and send alerts if thresholds crossed.
    Called from main.py on dashboard load.
    
    Args:
        df: Market data DataFrame
        mood_history: Mood history DataFrame
    
    Returns:
        Dict with alert status and channels notified
    """
    if mood_history.empty or len(mood_history) < 2:
        return {'sent': False, 'reason': 'Insufficient mood history'}
    
    current_score = mood_history['avg_trend_score'].iloc[-1]
    previous_score = mood_history['avg_trend_score'].iloc[-2]
    
    # Check if we should send (respect interval to avoid spam)
    if not should_run_auto_check():
        return {'sent': False, 'reason': 'Interval not passed'}
    
    results = send_mood_alert(current_score, previous_score)
    
    if results.get('telegram') or results.get('email'):
        record_auto_check()
        return {'sent': True, 'channels': results}
    
    return {'sent': False, 'reason': 'No threshold crossed'}


# ==========================================
# DAILY SUMMARY
# ==========================================

def send_daily_summary(summary_data: Dict) -> Tuple[bool, str]:
    """
    Send daily portfolio summary via Telegram.
    
    Args:
        summary_data: Dict from return_tracker.export_weekly_summary()
    
    Returns:
        (success, message)
    """
    if not is_telegram_configured():
        return False, "Telegram not configured"
    
    # Build summary message
    msg = "📊 <b>Daily Portfolio Summary</b>\n"
    msg += f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    
    total = summary_data.get('total_tracked', 0)
    avg_return = summary_data.get('avg_return_pct', 0)
    winners = summary_data.get('positive_count', 0)
    losers = summary_data.get('negative_count', 0)
    
    msg += f"📈 Tracked: {total} stocks\n"
    msg += f"💰 Avg Return: {avg_return:+.1f}%\n"
    msg += f"✅ Winners: {winners} | ❌ Losers: {losers}\n\n"
    
    # Top performers
    top = summary_data.get('top_performers', [])
    if top:
        msg += "🏆 <b>Top Performers:</b>\n"
        for stock in top[:3]:
            ticker = stock.get('ticker', '').replace('.NS', '')
            ret = stock.get('return_pct', 0)
            msg += f"  • {ticker}: {ret:+.1f}%\n"
    
    # Worst performers
    bottom = summary_data.get('bottom_performers', [])
    if bottom:
        msg += "\n📉 <b>Needs Attention:</b>\n"
        for stock in bottom[:3]:
            ticker = stock.get('ticker', '').replace('.NS', '')
            ret = stock.get('return_pct', 0)
            msg += f"  • {ticker}: {ret:+.1f}%\n"
    
    return send_telegram_message(msg)


if __name__ == "__main__":
    print("Testing Telegram Notifier...")
    
    # Test connection
    if is_telegram_configured():
        success, msg = test_telegram()
        print(f"Test result: {success} - {msg}")
    else:
        print("Telegram not configured. Use configure_telegram(token, chat_id)")
