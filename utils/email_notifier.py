"""
Email Notifier Module
=====================
Handles Gmail SMTP integration for sending weekly summaries and trend change alerts.
Uses Gmail App Passwords for authentication (2FA compatible).

Supports:
- Local: config.json file
- Streamlit Cloud: st.secrets
"""

import smtplib
import json
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional

CONFIG_FILE = "config.json"

def _load_config() -> Dict:
    """
    Load email configuration.
    Priority: 1) Environment vars (GitHub Actions), 2) Streamlit secrets (cloud), 3) config.json (local)
    """
    # Try Environment Variables First (for GitHub Actions)
    if os.environ.get("GMAIL_ADDRESS") and os.environ.get("GMAIL_APP_PASSWORD"):
        return {
            "email": {
                "gmail_address": os.environ.get("GMAIL_ADDRESS"),
                "app_password": os.environ.get("GMAIL_APP_PASSWORD"),
                "enabled": True
            }
        }
        
    # Try Streamlit secrets next (for cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'email' in st.secrets:
            return {
                "email": {
                    "gmail_address": st.secrets["email"].get("gmail_address", ""),
                    "app_password": st.secrets["email"].get("app_password", ""),
                    "enabled": True
                }
            }
    except:
        pass
    
    # Fall back to local config file
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        "email": {
            "gmail_address": "",
            "app_password": "",
            "enabled": False
        }
    }


def _save_config(config: Dict):
    """Save email configuration to config file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def configure_email(gmail_address: str, app_password: str) -> bool:
    """
    Configure Gmail credentials for sending emails.
    
    Args:
        gmail_address: Your Gmail address
        app_password: Gmail App Password (NOT regular password)
    
    Returns:
        True if configured successfully
    """
    config = _load_config()
    config["email"] = {
        "gmail_address": gmail_address,
        "app_password": app_password,
        "enabled": True
    }
    _save_config(config)
    return True

def is_email_configured() -> bool:
    """Check if email is properly configured."""
    config = _load_config()
    email_config = config.get("email", {})
    return bool(
        email_config.get("gmail_address") and 
        email_config.get("app_password") and
        email_config.get("enabled")
    )

def get_email_address() -> Optional[str]:
    """Get configured email address."""
    config = _load_config()
    return config.get("email", {}).get("gmail_address")

def disable_email():
    """Disable email notifications."""
    config = _load_config()
    if "email" in config:
        config["email"]["enabled"] = False
    _save_config(config)

def _send_email(subject: str, html_body: str) -> tuple[bool, str]:
    """
    Send an email via Gmail SMTP.
    
    Args:
        subject: Email subject line
        html_body: HTML content of the email
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    config = _load_config()
    email_config = config.get("email", {})
    
    gmail_address = email_config.get("gmail_address")
    app_password = email_config.get("app_password")
    
    if not gmail_address or not app_password:
        return False, "Email not configured. Please set up Gmail credentials."
    
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = f"Alpha Trend Dashboard <{gmail_address}>"
        msg['To'] = gmail_address
        
        # Attach HTML body
        html_part = MIMEText(html_body, 'html')
        msg.attach(html_part)
        
        # Connect to Gmail SMTP
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(gmail_address, app_password)
            server.send_message(msg)
        
        return True, "Email sent successfully!"
    
    except smtplib.SMTPAuthenticationError:
        return False, "Authentication failed. Check your Gmail App Password."
    except smtplib.SMTPException as e:
        return False, f"SMTP error: {str(e)}"
    except Exception as e:
        return False, f"Error sending email: {str(e)}"

def send_weekly_summary(summary_data: Dict) -> tuple[bool, str]:
    """
    Send weekly portfolio summary email.
    
    Args:
        summary_data: Dict from return_tracker.export_weekly_summary()
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    summary = summary_data.get("summary", {})
    all_returns = summary_data.get("all_returns", [])
    top_performers = summary_data.get("top_performers", [])
    bottom_performers = summary_data.get("bottom_performers", [])
    trend_changes = summary_data.get("trend_changes", [])
    
    # Format date
    today = datetime.now().strftime("%B %d, %Y")
    
    # Build HTML email
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
            .container {{ max-width: 600px; margin: 0 auto; background: #16213e; border-radius: 16px; padding: 30px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 12px; text-align: center; margin-bottom: 25px; }}
            .header h1 {{ margin: 0; color: white; font-size: 24px; }}
            .header p {{ margin: 10px 0 0 0; color: rgba(255,255,255,0.8); }}
            .stats {{ display: flex; justify-content: space-around; margin-bottom: 25px; }}
            .stat-box {{ text-align: center; background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; flex: 1; margin: 0 5px; }}
            .stat-value {{ font-size: 28px; font-weight: bold; color: #667eea; }}
            .stat-label {{ font-size: 12px; color: #888; margin-top: 5px; }}
            .section {{ margin-bottom: 25px; }}
            .section-title {{ font-size: 16px; font-weight: bold; color: #667eea; margin-bottom: 15px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; }}
            .stock-row {{ display: flex; justify-content: space-between; padding: 10px; background: rgba(255,255,255,0.03); margin-bottom: 8px; border-radius: 8px; }}
            .stock-name {{ font-weight: 500; }}
            .stock-return {{ font-weight: bold; }}
            .positive {{ color: #00C853; }}
            .negative {{ color: #FF5252; }}
            .neutral {{ color: #FFD600; }}
            .change-alert {{ background: rgba(255,152,0,0.1); border-left: 3px solid #FF9800; padding: 12px; margin-bottom: 10px; border-radius: 0 8px 8px 0; }}
            .footer {{ text-align: center; font-size: 12px; color: #666; margin-top: 30px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.1); }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Alpha Trend Weekly Report</h1>
                <p>{today}</p>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value">{summary.get('total_tracked', 0)}</div>
                    <div class="stat-label">TRACKED STOCKS</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value {'positive' if summary.get('total_return_pct', 0) >= 0 else 'negative'}">{summary.get('total_return_pct', 0):+.1f}%</div>
                    <div class="stat-label">AVG RETURN</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{summary.get('positive_count', 0)}/{summary.get('negative_count', 0)}</div>
                    <div class="stat-label">WIN/LOSS</div>
                </div>
            </div>
    """
    
    # Top Performers
    if top_performers:
        html += """
            <div class="section">
                <div class="section-title">🏆 TOP PERFORMERS</div>
        """
        for stock in top_performers:
            return_pct = stock.get('return_pct', 0)
            color_class = 'positive' if return_pct >= 0 else 'negative'
            html += f"""
                <div class="stock-row">
                    <span class="stock-name">{stock.get('ticker', '').replace('.NS', '')}</span>
                    <span class="stock-return {color_class}">{return_pct:+.1f}%</span>
                </div>
            """
        html += "</div>"
    
    # Bottom Performers
    if bottom_performers:
        html += """
            <div class="section">
                <div class="section-title">📉 UNDERPERFORMERS</div>
        """
        for stock in bottom_performers:
            return_pct = stock.get('return_pct', 0)
            color_class = 'positive' if return_pct >= 0 else 'negative'
            html += f"""
                <div class="stock-row">
                    <span class="stock-name">{stock.get('ticker', '').replace('.NS', '')}</span>
                    <span class="stock-return {color_class}">{return_pct:+.1f}%</span>
                </div>
            """
        html += "</div>"
    
    # Trend Changes
    if trend_changes:
        html += """
            <div class="section">
                <div class="section-title">⚠️ TREND CHANGES</div>
        """
        for change in trend_changes:
            entry_signal = change.get('entry_trend_signal', 'N/A')
            current_signal = change.get('current_signal', 'N/A')
            html += f"""
                <div class="change-alert">
                    <strong>{change.get('ticker', '').replace('.NS', '')}</strong>: 
                    {entry_signal} → {current_signal}
                </div>
            """
        html += "</div>"
    
    # All Holdings
    if all_returns:
        html += """
            <div class="section">
                <div class="section-title">📋 ALL HOLDINGS</div>
        """
        for stock in all_returns:
            return_pct = stock.get('return_pct', 0) or 0
            color_class = 'positive' if return_pct >= 0 else 'negative'
            days = stock.get('days_tracked', 0)
            html += f"""
                <div class="stock-row">
                    <span class="stock-name">{stock.get('ticker', '').replace('.NS', '')} <span style="color:#666">({days}d)</span></span>
                    <span class="stock-return {color_class}">{return_pct:+.1f}%</span>
                </div>
            """
        html += "</div>"
    
    html += """
            <div class="footer">
                Generated by Alpha Trend Dashboard<br>
                Track smarter, invest better.
            </div>
        </div>
    </body>
    </html>
    """
    
    subject = f"📊 Alpha Trend Weekly Report - {today}"
    return _send_email(subject, html)

def send_trend_change_alert(changes: List[Dict]) -> tuple[bool, str]:
    """
    Send immediate alert when tracked stocks change trend.
    
    Args:
        changes: List of stocks with trend changes
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    if not changes:
        return True, "No changes to report"
    
    today = datetime.now().strftime("%B %d, %Y %H:%M")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
            .container {{ max-width: 600px; margin: 0 auto; background: #16213e; border-radius: 16px; padding: 30px; }}
            .header {{ background: linear-gradient(135deg, #FF6B6B 0%, #FF9800 100%); padding: 25px; border-radius: 12px; text-align: center; margin-bottom: 25px; }}
            .header h1 {{ margin: 0; color: white; font-size: 22px; }}
            .header p {{ margin: 10px 0 0 0; color: rgba(255,255,255,0.8); }}
            .change-card {{ background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; margin-bottom: 15px; border-left: 4px solid #FF9800; }}
            .ticker {{ font-size: 20px; font-weight: bold; color: #fff; }}
            .change-detail {{ margin-top: 10px; }}
            .signal {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; }}
            .signal-up {{ background: rgba(0,200,83,0.2); color: #00C853; }}
            .signal-down {{ background: rgba(255,82,82,0.2); color: #FF5252; }}
            .signal-neutral {{ background: rgba(255,214,0,0.2); color: #FFD600; }}
            .arrow {{ margin: 0 10px; color: #888; }}
            .return-info {{ margin-top: 10px; font-size: 14px; color: #888; }}
            .footer {{ text-align: center; font-size: 12px; color: #666; margin-top: 30px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>⚠️ Trend Change Alert</h1>
                <p>{today}</p>
            </div>
    """
    
    for change in changes:
        ticker = change.get('ticker', '').replace('.NS', '')
        entry_signal = change.get('entry_trend_signal', 'N/A')
        current_signal = change.get('current_signal', 'N/A')
        return_pct = change.get('return_pct', 0) or 0
        days = change.get('days_tracked', 0)
        
        # Determine signal classes
        def get_signal_class(signal):
            if 'UPTREND' in signal.upper():
                return 'signal-up'
            elif 'DOWNTREND' in signal.upper():
                return 'signal-down'
            return 'signal-neutral'
        
        entry_class = get_signal_class(entry_signal)
        current_class = get_signal_class(current_signal)
        return_color = '#00C853' if return_pct >= 0 else '#FF5252'
        
        html += f"""
            <div class="change-card">
                <div class="ticker">{ticker}</div>
                <div class="change-detail">
                    <span class="signal {entry_class}">{entry_signal}</span>
                    <span class="arrow">→</span>
                    <span class="signal {current_class}">{current_signal}</span>
                </div>
                <div class="return-info">
                    Return: <span style="color:{return_color}">{return_pct:+.1f}%</span> 
                    &nbsp;|&nbsp; Tracked for {days} days
                </div>
            </div>
        """
    
    html += """
            <div class="footer">
                Alpha Trend Dashboard - Trend Change Alert<br>
                Consider reviewing your positions.
            </div>
        </div>
    </body>
    </html>
    """
    
    subject = f"⚠️ Trend Change Alert: {len(changes)} stock(s) - {datetime.now().strftime('%b %d')}"
    return _send_email(subject, html)

def send_system_alert(subject: str, message: str) -> tuple[bool, str]:
    """
    Send a generic text-based system alert (e.g., from the automated trading engine).
    """
    html_body = f"""
    <html>
        <body style="font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; padding: 20px;">
            <div style="background: #16213e; padding: 20px; border-radius: 10px;">
                <h2 style="color: #667eea; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px;">{subject}</h2>
                <pre style="font-family: Consolas, monospace; white-space: pre-wrap; font-size: 14px; color: #fff;">{message}</pre>
            </div>
        </body>
    </html>
    """
    return _send_email(subject, html_body)

def test_email_connection() -> tuple[bool, str]:
    """
    Test email configuration by sending a test email.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
            .container { max-width: 500px; margin: 0 auto; background: #16213e; border-radius: 16px; padding: 30px; text-align: center; }
            .icon { font-size: 48px; margin-bottom: 20px; }
            h1 { color: #667eea; margin-bottom: 15px; }
            p { color: #888; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="icon">✅</div>
            <h1>Email Configuration Successful!</h1>
            <p>Your Alpha Trend Dashboard is now set up to send email notifications.</p>
            <p style="margin-top: 20px; font-size: 12px;">
                You'll receive weekly portfolio summaries and trend change alerts.
            </p>
        </div>
    </body>
    </html>
    """
    
    subject = "✅ Alpha Trend Dashboard - Email Setup Successful"
    return _send_email(subject, html)
