import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging

# ✅ Silence streamlit "missing ScriptRunContext" spam in headless CI mode
# These are harmless when running as a plain Python script but flood the logs
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime").setLevel(logging.ERROR)

from trading_db import TradingDatabase
from utils.fast_data_engine import load_base_fundamentals, fetch_and_process_market_data, get_parquet_cache_path
from utils.telegram_notifier import send_telegram_message
from utils.email_notifier import send_trend_change_alert

def _log(msg):
    """Timestamped print — makes CI log timelines readable."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def get_nifty1000_universe():
    fundamentals = load_base_fundamentals(live_mode=True)
    return fundamentals

def generate_daily_master_cache():
    """
    Downloads the entire market universe, calculates technical indicators, 
    and saves it to the daily Parquet cache block.
    """
    _log("START: generate_daily_master_cache")
    fundamentals = get_nifty1000_universe()
    tickers = fundamentals['ticker'].dropna().tolist()
    _log(f"Universe loaded: {len(tickers)} tickers, fundamentals shape={fundamentals.shape}")
    
    # This also auto-saves the Parquet inside the function
    df = fetch_and_process_market_data(tickers, fundamentals, live_mode=True)
    _log(f"END: generate_daily_master_cache — result shape={df.shape}, columns={list(df.columns[:8])}{'...' if len(df.columns)>8 else ''}")
    return df

def generate_sub_industry_rotation(df, db):
    """
    Groups the Nifty 1000 universe by 58 Sub-Industries from our Encyclopedia.
    Calculates average composite RS and volume signals to populate the Heatmap.
    Appends historical data (one snapshot per date) for monthly heatmap tracking.
    """
    _log("START: generate_sub_industry_rotation")
    if df.empty or 'sector' not in df.columns or 'comp_rs' not in df.columns:
        _log(f"SKIP sub_industry_rotation — df.empty={df.empty}, columns={list(df.columns[:5])}")
        return
        
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # Group by the Sub-Industry (which is in the `sector` column due to our mapping)
    groups = df.groupby('sector')
    rotation_rows = []
    
    for sector, group in groups:
        if sector == "Unknown": continue
        avg_rs = float(group['comp_rs'].mean())
        # Find top 3 stocks in this sub-industry by RS
        top_stocks = group.sort_values(by='comp_rs', ascending=False).head(3)['ticker'].tolist()
        top_comps_str = ", ".join(top_stocks)
        
        rotation_rows.append({
            'record_date': today_str,
            'sub_industry': sector,
            'rs_momentum': round(avg_rs, 2),
            'top_components': top_comps_str
        })
        
    rot_df = pd.DataFrame(rotation_rows)
    
    # Normalize rs_momentum to 0-100 percentile rank within this snapshot
    if not rot_df.empty:
        rs_min = rot_df['rs_momentum'].min()
        rs_max = rot_df['rs_momentum'].max()
        if rs_max > rs_min:
            rot_df['score_0_100'] = ((rot_df['rs_momentum'] - rs_min) / (rs_max - rs_min) * 100).round(0).astype(int)
        else:
            rot_df['score_0_100'] = 50
    
    # Delete existing rows for today to prevent duplicates, then append
    try:
        db.cursor.execute("DELETE FROM sub_industry_rotation WHERE record_date = ?", (today_str,))
        db.conn.commit()
    except Exception:
        pass
    
    rot_df.to_sql('sub_industry_rotation', db.conn, if_exists='append', index=False)
    print(f"Computed rotation for {len(rot_df)} Sub-Industries ({today_str}).")


def check_portfolio_stops(df, db):
    """
    Orchestrates the trailing stop loss logic on the current active portfolio.
    Issues telegram alerts if stops are hit.
    """
    print("Checking Portfolio Stops...")
    portfolio = pd.read_sql_query("SELECT * FROM portfolio", db.conn)
    
    if portfolio.empty:
        print("Portfolio is empty. No stops to check.")
        return
    
    # Guard: if yfinance data fetch failed, df may have no 'currentPrice' column
    if df.empty or 'currentPrice' not in df.columns:
        print("[STOP CHECK] Skipped: market data unavailable (no currentPrice column).")
        return

    df_lookup = df.set_index('ticker')
    sells = []
    alerts = []
    
    for _, pos in portfolio.iterrows():
        ticker = pos['ticker']
        if ticker in df_lookup.index:
            current_price = df_lookup.loc[ticker, 'currentPrice']
            ma50 = df_lookup.loc[ticker, 'fiftyDayAverage']
            
            # Simple 50-Day MA trailing stop check
            # Real exit logic should check if price < MA50
            if current_price < ma50:
                print(f"STOP LOSS HIT: {ticker} (Price: {current_price} < MA50: {ma50})")
                sells.append((ticker, current_price, "MA50 Break"))
                alerts.append(f"🚨 *STOP LOSS HIT* 🚨\\n*{ticker}*\\nPrice: ₹{current_price:,.2f}\\nReason: Dropped below 50-Day MA (₹{ma50:,.2f})")
                
            # Update peak price for trailing
            if current_price > pos.get('peak_price', pos['entry_price']):
                db.cursor.execute("UPDATE portfolio SET peak_price = ? WHERE ticker = ?", (current_price, ticker))

    # Execute Sells
    for ticker, exit_price, reason in sells:
        # Move from portfolio to ledger
        db.cursor.execute("SELECT * FROM portfolio WHERE ticker = ?", (ticker,))
        pos = db.cursor.fetchone()
        
        # Calculate PnL
        entry_price = pos[2]
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        
        db.cursor.execute("""
            INSERT INTO ledger (ticker, buy_date, sell_date, entry_price, exit_price, pnl_pct, reason, strategy_tag) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (ticker, pos[1], datetime.now().strftime("%Y-%m-%d"), entry_price, exit_price, pnl_pct, reason, pos[6]))
        
        db.cursor.execute("DELETE FROM portfolio WHERE ticker = ?", (ticker,))
        
    db.conn.commit()
    
    
    # Send all alerts in ONE Telegram message
    if alerts:
        send_telegram_message("\\n\\n".join(alerts))
        for alert in alerts:
            db.cursor.execute("INSERT INTO alerts_log (alert_date, ticker, alert_type, message) VALUES (?, ?, ?, ?)",
                              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "SYSTEM", "SELL", alert))
        db.conn.commit()
    
    # Send all alerts in ONE Email
    if sells:
        email_alerts = []
        for ticker, exit_price, reason in sells:
            db.cursor.execute("SELECT * FROM portfolio WHERE ticker = ?", (ticker,))
            pos = db.cursor.fetchone()
            if pos:
                entry_price = pos[2]
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                email_alerts.append({
                    'ticker': ticker,
                    'entry_trend_signal': 'ACTIVE',
                    'current_signal': f'🚨 STOP LOSS ({reason})',
                    'return_pct': pnl_pct,
                    'days_tracked': 0
                })
        if email_alerts:
            send_trend_change_alert(email_alerts)

def run_rebalance(df, db):
    """
    Orchestrates finding new buys based on the V3.1 Momentum conditions.
    """
    print("Running Portfolio Rebalance Scanners...")
    # Guard: if yfinance data fetch failed, df may have no 'dna_signal' column
    if df.empty or 'dna_signal' not in df.columns:
        print("[REBALANCE] Skipped: market data unavailable (no dna_signal column).")
        return
    # Find active buys
    buyers = df[df['dna_signal'] == 'BUY']
    if buyers.empty:
        print("No BUY signals generated today.")
        return
        
    top_buys = buyers.sort_values(by='comp_rs', ascending=False).head(5)
    alerts = []
    
    for _, row in top_buys.iterrows():
        ticker = row['ticker']
        price = row['currentPrice']
        rs = row['comp_rs']
        msg = f"🟢 *NEW BUY SIGNAL* 🟢\\n*{ticker}*\\nPrice: ₹{price:,.2f}\\nRS Score: {rs}"
        alerts.append(msg)
        
        # Add to watchlist natively
        db.cursor.execute("""
            INSERT OR REPLACE INTO watchlist (ticker, added_date, v3_score, rs_score, sector, status) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, (ticker, datetime.now().strftime("%Y-%m-%d"), row.get('overall', 0), rs, row.get('sector', ''), 'ACTIVE'))
        
    if alerts:
        send_telegram_message("\\n\\n".join(alerts))
        for alert in alerts:
            db.cursor.execute("INSERT INTO alerts_log (alert_date, ticker, alert_type, message) VALUES (?, ?, ?, ?)",
                              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "SYSTEM", "BUY_ALERT", alert))
        db.conn.commit()
        
    # Build batch email alert for buys
    if not top_buys.empty:
        email_alerts = []
        for _, row in top_buys.iterrows():
            email_alerts.append({
                'ticker': row['ticker'],
                'entry_trend_signal': 'CASH',
                'current_signal': '🟢 NEW BUY SIGNAL',
                'return_pct': 0,
                'days_tracked': 0
            })
        send_trend_change_alert(email_alerts)

def send_daily_heartbeat(df, mode):
    """
    Sends a positive confirmation that the engine ran successfully.
    Acts as a fail-safe check against silent GitHub Actions failures.
    """
    print("Sending Daily Heartbeat Confirmation...")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    universe_size = len(df) if not df.empty else 0
    buyers = df[df['dna_signal'] == 'BUY'] if not df.empty and 'dna_signal' in df.columns else pd.DataFrame()
    buy_count = len(buyers)
    
    msg = f"✅ <b>EOD Engine Run: SUCCESS</b>\n\n"
    msg += f"🕒 <b>Time:</b> {now_str} IST\n"
    msg += f"⚙️ <b>Mode:</b> {mode.upper()}\n"
    msg += f"📊 <b>Universe Scanned:</b> {universe_size} Nifty Stocks\n"
    msg += f"🟢 <b>New Buy Signals:</b> {buy_count}\n\n"
    msg += "<i>Systems operating normally.</i>"
    
    try:
        from utils.telegram_notifier import send_telegram_message
        send_telegram_message(msg)
    except: pass
    
    try:
        from utils.email_notifier import _send_email, is_email_configured
        if is_email_configured():
            _send_email(f"✅ EOD Engine Run: SUCCESS ({buy_count} Buys)", f"<pre>{msg.replace('<b>', '').replace('</b>', '').replace('<i>', '').replace('</i>', '')}</pre>")
    except: pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['daily_cache', 'stop_check', 'rebalance', 'full'], default='full')
    args = parser.parse_args()
    
    db = TradingDatabase()
    
    try:
        if args.mode in ['daily_cache', 'full']:
            df = generate_daily_master_cache()
            generate_sub_industry_rotation(df, db)
        else:
            # Just load existing parquet cache for fast ops if daily scan already ran
            parquet_path = get_parquet_cache_path()
            if os.path.exists(parquet_path):
                df = pd.read_parquet(parquet_path)
            else:
                print("[ENGINE] No parquet cache found. Creating empty DataFrame.")
                df = pd.DataFrame()

        if args.mode in ['stop_check', 'full']:
            check_portfolio_stops(df, db)
            
        if args.mode in ['rebalance', 'full']:
            run_rebalance(df, db)

    except Exception as e:
        print(f"[ENGINE] Critical error during execution: {e}")
        import traceback
        traceback.print_exc()
        df = pd.DataFrame()  # Ensure heartbeat still sends
        raise  # Re-raise so GitHub Actions marks job as failed (intentional if truly broken)
        
    db.close()
    
    # Send confirming heartbeat before shutting down
    send_daily_heartbeat(df, args.mode)
    
    print("Trading Engine Execution Completed.")
