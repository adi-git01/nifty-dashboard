"""
Alert Daemon
============
Standalone script to check stock positions and send Telegram alerts.
Designed to run via GitHub Actions or Cron.

Usage:
  python alert_daemon.py              # Normal run (volume scan only after 4 PM IST)
  python alert_daemon.py --force-volume  # Force volume scan (for GitHub Actions)
"""

import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta
import time

# IST = UTC + 5:30
IST = timezone(timedelta(hours=5, minutes=30))

def now_ist():
    """Current time in IST regardless of server timezone (critical for GitHub Actions on UTC)."""
    return datetime.now(tz=IST)

# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.positions import get_all_positions
from utils.telegram_notifier import send_telegram_message, is_telegram_configured

from utils.volume_monitor import scan_volume_changes
from utils.nifty500_list import TICKERS

# Track last volume scan date to avoid repetition
LAST_VOLUME_SCAN = None

def check_alerts():
    print(f"⏰ Starting Alert Check: {datetime.now()}")
    
    # --- 1. POSITION & PRICE ALERTS (Runs Every Time) ---
    # 1. Load active positions
    positions = get_all_positions(status='active')
    
    alerts_triggered = [] # Initialize for position alerts

    if not positions:
        print("ℹ️ No active positions to check.")
    else:
        tickers = [p['ticker'] for p in positions]
        print(f"📋 Checking {len(tickers)} positions: {tickers}")

        # 2. Fetch live prices
        current_prices = {}
        try:
            # Use period='1d' to get latest data
            df = yf.download(tickers, period='1d', progress=False)['Close']
            if df.empty:
                print("❌ No price data fetched for positions.")
            else:
                # Handle single ticker case (Series) vs multiple (DataFrame)
                if isinstance(df, pd.Series):
                    current_prices = {tickers[0]: df.iloc[-1]}
                else:
                    current_prices = df.iloc[-1].to_dict()
                
        except Exception as e:
            print(f"❌ Error fetching prices for positions: {e}")

        # 3. Check conditions
        if current_prices: # Only proceed if prices were fetched
            for pos in positions:
                ticker = pos['ticker']
                # Handle ticker formatting differences in yfinance result
                # YF might return "HDFCBANK.NS" or just "HDFCBANK"
                price = current_prices.get(ticker)
                
                if not price:
                    # Try removing suffix
                    simple_ticker = ticker.replace('.NS', '').replace('.BO', '')
                    price = current_prices.get(simple_ticker)
                    
                if not price:
                    print(f"Price not found for {ticker}")
                    continue
                    
                # SL Check
                sl = pos.get('stop_loss')
                if sl and price <= sl:
                    msg = f"🔴 STOP LOSS HIT: {ticker}\nPrice: ₹{price:.2f}\nSL: ₹{sl:.2f}"
                    alerts_triggered.append(msg)
                    
                # Target Check
                target = pos.get('target')
                if target and price >= target:
                    msg = f"🎯 TARGET HIT: {ticker}\nPrice: ₹{price:.2f}\nTarget: ₹{target:.2f}"
                    alerts_triggered.append(msg)

        # 4. Send Notifications for positions
        if alerts_triggered:
            print(f"Triggering {len(alerts_triggered)} position alerts...")
            
            if is_telegram_configured():
                full_msg = "🚨 **MARKET ALERT** 🚨\n\n" + "\n\n".join(alerts_triggered)
                success, err = send_telegram_message(full_msg)
                if success:
                    print("Telegram sent successfully for position alerts.")
                else:
                    print(f"❌ Telegram failed for position alerts: {err}")
            else:
                print("Telegram not configured. Skipping position alert send.")
        else:
            print("No position alerts triggered.")

    # --- 2. VOLUME ALERTS (Runs Once Daily after Market Close IST) ---
    global LAST_VOLUME_SCAN
    now = now_ist()  # ✅ Always use IST, not UTC server time
    market_close_hour_ist = 16  # 4 PM IST
    force_volume = '--force-volume' in sys.argv  # GitHub Actions can pass this flag

    # Run if it's past 4 PM IST (or forced) and we haven't run it today
    is_after_market = now.hour >= market_close_hour_ist
    is_new_day = (LAST_VOLUME_SCAN is None) or (LAST_VOLUME_SCAN.date() < now.date())

    if (is_after_market and is_new_day) or force_volume:
        print(f"📢 Starting Daily Volume Scan (Nifty 500)... [IST hour={now.hour}, force={force_volume}]")
        # Scan entire universe
        vol_alerts = scan_volume_changes(TICKERS)
        
        if vol_alerts:
            print(f"{len(vol_alerts)} Volume Alerts found!")

            # Load portfolio holdings to split alerts
            held_tickers = set()
            try:
                import json
                snap_file = os.path.join(os.path.dirname(__file__), "data", "dna3_portfolio_snapshot.json")
                if os.path.exists(snap_file):
                    with open(snap_file) as f:
                        snap = json.load(f)
                    held_tickers = set(snap.get('holdings', {}).keys())
            except:
                pass

            # Split into held vs universe
            held_alerts = [a for a in vol_alerts if a['ticker'] in held_tickers]
            universe_alerts = [a for a in vol_alerts if a['ticker'] not in held_tickers]

            messages = []
            if held_alerts:
                messages.append("🏠 **YOUR PORTFOLIO**")
                for a in held_alerts:
                    icon = "🟢" if a['change'] > 0 else "🔴"
                    msg = f"{icon} **{a['ticker'].replace('.NS','')}**: Vol Score {a['previous']:.1f} → {a['current']:.1f} ({a['signal']})"
                    messages.append(msg)
                messages.append("")

            messages.append("🔊 **UNIVERSE (Top 10)**")
            for a in universe_alerts[:10]:
                icon = "🟢" if a['change'] > 0 else "🔴"
                msg = f"{icon} **{a['ticker'].replace('.NS','')}**: Vol Score {a['previous']:.1f} → {a['current']:.1f} ({a['signal']})"
                messages.append(msg)
            
            if len(universe_alerts) > 10:
                messages.append(f"...and {len(universe_alerts)-10} more.")
            
            full_msg = "🔊 **SMART VOLUME ALERTS** 🔊\n\n" + "\n".join(messages)
            
            # Send via Telegram
            if is_telegram_configured():
                success, err = send_telegram_message(full_msg)
                if success:
                    print("Telegram sent successfully for volume alerts.")
                else:
                    print(f"❌ Telegram failed for volume alerts: {err}")
            
            # Send via Email (Daily Digest)
            from utils.email_notifier import is_email_configured, _send_email
            if is_email_configured():
                print("📧 Sending Volume Digest via Email...")
                
                # Create HTML Table for Email
                html_rows = ""
                for a in vol_alerts:
                     color = "#00C853" if a['change'] > 0 else "#FF5252"
                     bg_color = "rgba(0,200,83,0.1)" if a['change'] > 0 else "rgba(255,82,82,0.1)"
                     html_rows += f"""
                     <tr style="background-color: {bg_color};">
                        <td style="padding: 10px; border-bottom: 1px solid #444;"><strong>{a['ticker'].replace('.NS','')}</strong></td>
                        <td style="padding: 10px; border-bottom: 1px solid #444; color: {color};"><strong>{a['previous']:.1f} → {a['current']:.1f}</strong></td>
                         <td style="padding: 10px; border-bottom: 1px solid #444;">{a['signal']}</td>
                         <td style="padding: 10px; border-bottom: 1px solid #444;">₹{a['price']:.2f}</td>
                     </tr>
                     """
                
                today_str = datetime.now().strftime("%d %b %Y")
                email_html = f"""
                <html>
                <body style="font-family: Arial, sans-serif; background-color: #1a1a2e; color: #eee; padding: 20px;">
                    <div style="background-color: #16213e; padding: 20px; border-radius: 10px; max-width: 600px; margin: auto;">
                        <h2 style="color: #00F0FF; text-align: center;">🔊 Smart Volume Digest ({today_str})</h2>
                        <p style="text-align: center; color: #aaa;">The following stocks showed significant volume accumulation/distribution today.</p>
                        <table style="width: 100%; border-collapse: collapse; margin-top: 20px;">
                            <tr style="background-color: #0f3460; color: white;">
                                <th style="padding: 10px; text-align: left;">Ticker</th>
                                <th style="padding: 10px; text-align: left;">Score Change</th>
                                <th style="padding: 10px; text-align: left;">Signal</th>
                                <th style="padding: 10px; text-align: left;">Price</th>
                            </tr>
                            {html_rows}
                        </table>
                         <p style="text-align: center; margin-top: 30px; font-size: 12px; color: #666;">Alpha Hunter Elite System</p>
                    </div>
                </body>
                </html>
                """
                
                success, msg = _send_email(f"🔊 Volume Alert Digest - {today_str}", email_html)
                if success:
                    print("Email sent successfully.")
                else:
                    print(f"❌ Email failed: {msg}")
        else:
            print("No volume alerts triggered.")
            
        LAST_VOLUME_SCAN = now

    print("Check Complete.")

if __name__ == "__main__":
    check_alerts()
