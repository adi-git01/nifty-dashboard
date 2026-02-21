"""
DNA3 ALERT DAEMON
=================
Runs the DNA3 Portfolio Scanner periodically (e.g. every 60 mins) 
and sends alerts if the portfolio changes.

- Checks file `data/dna3_portfolio_snapshot.json`
- Compares with Previous Snapshot
- Uses `utils/email_notifier` or `utils/telegram_notifier` to send alerts.
"""

import time
import json
import os
import sys
from datetime import datetime
import subprocess

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.telegram_notifier import send_telegram_message

SNAPSHOT_FILE = "data/dna3_portfolio_snapshot.json"
HISTORY_FILE = "data/dna3_history.json"

def load_portfolio():
    if not os.path.exists(SNAPSHOT_FILE): return set()
    try:
        with open(SNAPSHOT_FILE, 'r') as f:
            data = json.load(f)
            return {s['Ticker'] for s in data.get('portfolio', [])}
    except: return set()

def main():
    print("Starting DNA3 Alert Daemon...")
    
    last_portfolio = load_portfolio()
    
    while True:
        try:
            print(f"\n[{datetime.now()}] Running DNA3 Scan...")
            subprocess.run(["python", "dna3_current_portfolio.py"], check=True)
            
            current_portfolio = load_portfolio()
            
            new_buys = current_portfolio - last_portfolio
            new_sells = last_portfolio - current_portfolio
            
            if new_buys:
                msg = f"🚀 **DNA3 BUY ALERT**\n" + "\n".join([f"• {t}" for t in new_buys])
                print(msg)
                send_telegram_message(msg)
                
            if new_sells:
                msg = f"🔴 **DNA3 SELL ALERT**\n" + "\n".join([f"• {t}" for t in new_sells])
                print(msg)
                send_telegram_message(msg)
                
            last_portfolio = current_portfolio
            
            # Wait 1 hour
            time.sleep(3600)
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
