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
import re
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
from utils.telegram_notifier import send_telegram_message, is_telegram_configured, queue_commit

from utils.volume_monitor import scan_volume_changes
from utils.nifty500_list import TICKERS
from utils.yf_safe import safe_download

# Track last volume scan date to avoid repetition
LAST_VOLUME_SCAN = None

def _clean_sym(ticker: str) -> str:
    return re.sub(r"\.(NS|BO)$", "", str(ticker or ""))


def check_alerts():
    print(f"⏰ Starting Alert Check: {now_ist():%Y-%m-%d %H:%M:%S} IST")
    
    # --- 1. POSITION & PRICE ALERTS (Runs Every Time) ---
    # Positions and price alerts come from the GitHub-backed store (alert-state branch), the same
    # copy the dashboard edits. Symbols are resolved against the universe first: FEDERALBANK.NS
    # (NSE symbol FEDERALBNK) used to get "Price not found" on every run.
    from utils.positions import STORE as POS_STORE, resolve_ticker
    from utils.alerts import STORE as ALERTS_STORE, load_alerts
    from utils.nifty1000_list import TICKERS_1000
    positions = get_all_positions(status='active')
    try:
        price_alerts = [a for a in load_alerts() if a.get('stop_loss') or a.get('target')]
    except Exception as e:
        print(f"⚠️ Price alerts unreadable: {e}")
        price_alerts = []

    alerts_triggered = [] # Initialize for position alerts
    today = now_ist().strftime('%Y-%m-%d')

    if not positions and not price_alerts:
        print("ℹ️ No active positions or price alerts to check.")
    else:
        sym = {}
        for item in positions + price_alerts:
            t = item['ticker']
            sym[t] = resolve_ticker(t, TICKERS_1000) or t
            if sym[t] != t:
                print(f"   {t} priced as {sym[t]}")
        tickers = sorted(set(sym.values()))
        print(f"📋 Checking {len(positions)} position(s) and {len(price_alerts)} price alert(s): {tickers}")

        # 2. Fetch live prices — retry-with-backoff instead of a single
        # unguarded attempt (a transient Yahoo blip here used to just
        # silently skip every stop-loss/target check for the whole run).
        current_prices = {}
        data = safe_download(tickers, period='1d', group_by='ticker', auto_adjust=True, threads=False, min_coverage=0.5)
        if data is None or data.empty:
            print("❌ No price data fetched for positions after retries.")
        elif isinstance(data.columns, pd.MultiIndex):
            lvl0 = data.columns.get_level_values(0)
            lvl1 = data.columns.get_level_values(1)
            for t in tickers:
                try:
                    if t in lvl0:
                        s = data[t]["Close"]
                    elif "Close" in lvl0 and t in lvl1:
                        s = data["Close"][t]
                    else:
                        continue
                    if isinstance(s, pd.DataFrame):
                        s = s.iloc[:, 0]
                    close = s.dropna()
                    if not close.empty:
                        current_prices[t] = float(close.iloc[-1])
                except Exception:
                    pass
        else:
            close = data['Close'].dropna() if 'Close' in data.columns else pd.Series(dtype=float)
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            if not close.empty and len(tickers) == 1:
                current_prices = {tickers[0]: float(close.iloc[-1])}

        # 3. Check conditions. An alert that already went out is not repeated for 5 days: the
        # daemon used to resend the same stop-loss message on every run while price stayed below.
        def recently_sent(item, kind):
            last = item.get('last_alert') or {}
            try:
                return last.get('type') == kind and (pd.Timestamp(today) - pd.Timestamp(last.get('date'))).days < 5
            except Exception:
                return False

        missing, patches, pa_patches = [], {}, {}
        try:
            from utils import corp_actions
            ca = corp_actions.load()
        except Exception:
            ca = None
        for kind_label, items in (("position", positions), ("price alert", price_alerts)):
            for pos in items:
                ticker = pos['ticker']
                price = current_prices.get(sym[ticker])
                if not price:
                    missing.append(ticker)
                    print(f"Price not found for {ticker}")
                    continue
                fac = 1.0
                if ca is not None and pos.get('entry_date'):
                    fac = corp_actions.cumulative_factor(ca, sym[ticker], pos['entry_date'])
                name = _clean_sym(ticker)
                sl = pos.get('stop_loss')
                target = pos.get('target')
                hit = None
                if sl and price <= sl * fac:
                    hit = ('STOP_LOSS', f"🔴 STOP LOSS HIT: {name} ({kind_label})\nPrice: ₹{price:.2f}\nSL: ₹{sl * fac:.2f}")
                elif target and price >= target * fac:
                    hit = ('TARGET', f"🎯 TARGET HIT: {name} ({kind_label})\nPrice: ₹{price:.2f}\nTarget: ₹{target * fac:.2f}")
                if hit and not recently_sent(pos, hit[0]):
                    alerts_triggered.append(hit[1])
                    if pos.get('id'):
                        patch = {'last_alert': {'type': hit[0], 'date': today, 'price': round(float(price), 2)}}
                        if kind_label == "position":
                            patches[pos['id']] = patch
                        else:
                            pa_patches[pos['id']] = patch
                elif hit:
                    print(f"   {name}: {hit[0]} still in force (already sent {pos['last_alert']['date']})")
        if missing:
            alerts_triggered.append("⚠️ NO PRICE for: " + ", ".join(_clean_sym(t) for t in missing)
                                    + "\nStop-loss and target could not be checked. Fix the symbol in the Position Manager.")

        # 4. Send Notifications for positions
        if alerts_triggered:
            print(f"Triggering {len(alerts_triggered)} position alerts...")
            
            if is_telegram_configured():
                full_msg = "🚨 <b>MARKET ALERT</b> 🚨\n\n" + "\n\n".join(alerts_triggered)
                success, err = send_telegram_message(full_msg)
                if success:
                    print("Telegram sent successfully for position alerts.")
                    queued = bool(os.environ.get("TELEGRAM_BATCH", "").strip())
                    if patches:
                        if not (queued and queue_commit({"kind": "positions_patch", "patches": patches, "reason": f"{len(patches)} alert(s) sent"})):
                            POS_STORE.commit({"patches": patches}, f"{len(patches)} alert(s) sent")
                    if pa_patches:
                        if not (queued and queue_commit({"kind": "price_alerts_patch", "patches": pa_patches, "reason": f"{len(pa_patches)} price alert(s) sent"})):
                            ALERTS_STORE.commit({"patches": pa_patches}, f"{len(pa_patches)} price alert(s) sent")
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
                messages.append("🏠 <b>YOUR PORTFOLIO</b>")
                for a in held_alerts:
                    icon = "🟢" if a['change'] > 0 else "🔴"
                    msg = f"{icon} <b>{_clean_sym(a['ticker'])}</b>: Vol Score {a['previous']:.1f} → {a['current']:.1f} ({a['signal']})"
                    messages.append(msg)
                messages.append("")

            messages.append("🔊 <b>UNIVERSE (Top 10)</b>")
            for a in universe_alerts[:10]:
                icon = "🟢" if a['change'] > 0 else "🔴"
                msg = f"{icon} <b>{_clean_sym(a['ticker'])}</b>: Vol Score {a['previous']:.1f} → {a['current']:.1f} ({a['signal']})"
                messages.append(msg)
            
            if len(universe_alerts) > 10:
                messages.append(f"...and {len(universe_alerts)-10} more.")
            
            full_msg = "🔊 <b>SMART VOLUME ALERTS</b> 🔊\n\n" + "\n".join(messages)
            
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
                        <td style="padding: 10px; border-bottom: 1px solid #444;"><strong>{_clean_sym(a['ticker'])}</strong></td>
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
