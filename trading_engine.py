import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import os
from trading_db import TradingDatabase
from utils.fast_data_engine import load_base_fundamentals, fetch_and_process_market_data, get_parquet_cache_path
from utils.telegram_notifier import send_telegram_message

def get_nifty1000_universe():
    fundamentals = load_base_fundamentals(live_mode=True)
    return fundamentals

def generate_daily_master_cache():
    """
    Downloads the entire market universe, calculates technical indicators, 
    and saves it to the daily Parquet cache block.
    """
    print("Generating Daily Master Cache for Nifty 1000...")
    fundamentals = get_nifty1000_universe()
    tickers = fundamentals['ticker'].dropna().tolist()
    
    # This also auto-saves the Parquet inside the function
    df = fetch_and_process_market_data(tickers, fundamentals, live_mode=True)
    return df

def generate_sub_industry_rotation(df, db):
    """
    Groups the Nifty 1000 universe by 58 Sub-Industries from our Encyclopedia.
    Calculates average composite RS and volume signals to populate the Heatmap.
    """
    print("Calculating Sub-Industry Rotation...")
    if df.empty or 'sector' not in df.columns or 'comp_rs' not in df.columns:
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
    rot_df.to_sql('sub_industry_rotation', db.conn, if_exists='replace', index=False)
    print(f"Computed rotation for {len(rot_df)} Sub-Industries.")


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
    
    # Send all alerts in one Telegram message
    if alerts:
        send_telegram_message("\\n\\n".join(alerts))
        for alert in alerts:
            db.cursor.execute("INSERT INTO alerts_log (alert_date, ticker, alert_type, message) VALUES (?, ?, ?, ?)",
                              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "SYSTEM", "SELL", alert))
        db.conn.commit()

def run_rebalance(df, db):
    """
    Orchestrates finding new buys based on the V3.1 Momentum conditions.
    """
    print("Running Portfolio Rebalance Scanners...")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['daily_cache', 'stop_check', 'rebalance', 'full'], default='full')
    args = parser.parse_args()
    
    db = TradingDatabase()
    
    if args.mode in ['daily_cache', 'full']:
        df = generate_daily_master_cache()
        generate_sub_industry_rotation(df, db)
    else:
        # Just load existing parquet cache for fast ops if daily scan already ran
        df = pd.read_parquet(get_parquet_cache_path())

    if args.mode in ['stop_check', 'full']:
        check_portfolio_stops(df, db)
        
    if args.mode in ['rebalance', 'full']:
        run_rebalance(df, db)
        
    db.close()
    print("Trading Engine Execution Completed.")
