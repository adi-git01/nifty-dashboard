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
from utils.regime_manager import classify_regime, get_regime_params
from utils.market_mood import calculate_mood_metrics, save_mood_snapshot
from utils.market_breadth import calculate_breadth_from_df, save_breadth_snapshot

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

    # Save daily mood snapshot so the Market Mood History chart stays current
    try:
        mood_metrics = calculate_mood_metrics(df)
        if mood_metrics:
            save_mood_snapshot(mood_metrics)
            _log(f"Mood snapshot saved: avg_trend={mood_metrics.get('avg_trend_score')}, uptrends={mood_metrics.get('total_uptrends')}")
    except Exception as e:
        _log(f"WARN: mood snapshot failed ({e}) — skipping")

    # Save daily breadth snapshot so the Market Breadth chart stays current
    try:
        breadth_metrics = calculate_breadth_from_df(df)
        if breadth_metrics:
            save_breadth_snapshot(breadth_metrics)
            _log(f"Breadth snapshot saved: {breadth_metrics.get('pct_above_50dma')}% above 50DMA, {breadth_metrics.get('pct_above_200dma')}% above 200DMA")
    except Exception as e:
        _log(f"WARN: breadth snapshot failed ({e}) — skipping")

    return df

def generate_sub_industry_rotation(df, db):
    """
    Groups the Nifty 1000 universe by 58 Sub-Industries from our Encyclopedia.
    Calculates average composite RS and volume signals to populate the Heatmap.
    Appends historical data (one snapshot per date) for monthly heatmap tracking.
    """
    _log("START: generate_sub_industry_rotation")
    if df.empty or 'comp_rs' not in df.columns:
        _log(f"SKIP sub_industry_rotation — df.empty={df.empty}, columns={list(df.columns[:5])}")
        return

    # Use sector_granular (54 sub-industries) if available, else fall back to sector
    group_col = 'sector_granular' if 'sector_granular' in df.columns else 'sector'
    if group_col not in df.columns:
        _log("SKIP sub_industry_rotation — no sector column found")
        return

    today_str = datetime.now().strftime("%Y-%m-%d")

    groups = df.groupby(group_col)
    rotation_rows = []

    for sector, group in groups:
        if not sector or sector in ("Unknown", "nan"): continue
        avg_rs = float(group['comp_rs'].mean())
        # Find top 3 stocks in this sub-industry by RS
        top_stocks = group.sort_values(by='comp_rs', ascending=False).head(3)['ticker'].tolist()
        top_comps_str = ", ".join([t.replace(".NS", "") for t in top_stocks])
        
        rotation_rows.append({
            'record_date': today_str,
            'sub_industry': sector,
            'rs_momentum': round(avg_rs, 2),
            'top_components': top_comps_str
        })
        
    rot_df = pd.DataFrame(rotation_rows)
    
    # Rank-based percentile score (0-100). One outlier sub-industry can't compress
    # all other scores toward zero the way min-max normalization would.
    if not rot_df.empty:
        if len(rot_df) > 1:
            rot_df['score_0_100'] = (rot_df['rs_momentum'].rank(pct=True) * 100).round(0).astype(int)
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
    V2.2: Uses regime-adaptive trailing stop percentages.
    Issues telegram alerts if stops are hit.
    """
    print("Checking Portfolio Stops (V2.2 Regime-Adaptive)...")
    portfolio = pd.read_sql_query("SELECT * FROM portfolio", db.conn)
    
    if portfolio.empty:
        print("Portfolio is empty. No stops to check.")
        return
    
    if df.empty or 'currentPrice' not in df.columns:
        print("[STOP CHECK] Skipped: market data unavailable (no currentPrice column).")
        return

    # V2.2: Compute current regime for adaptive trailing stop
    import yfinance as yf
    try:
        _nifty = yf.Ticker("^NSEI").history(period="2y")
        _n_close  = float(_nifty['Close'].iloc[-1])
        _n_ma200  = float(_nifty['Close'].rolling(200).mean().iloc[-1])
        _n_52w_hi = float(_nifty['High'].rolling(252).max().iloc[-1])
        regime = classify_regime(_n_close, _n_ma200, _n_52w_hi)
    except Exception:
        regime = 'CAUTION'

    params = get_regime_params(regime)
    trail_pct = params['trail_stop']  # 0.15 / 0.12 / 0.08 / 0.06
    trail_keep = 1.0 - trail_pct
    print(f"  Regime: {regime} | Trail Stop: {trail_pct*100:.0f}%")

    df_lookup = df.set_index('ticker')
    sells = []
    alerts = []
    
    for _, pos in portfolio.iterrows():
        ticker = pos['ticker']
        if ticker in df_lookup.index:
            current_price = df_lookup.loc[ticker, 'currentPrice']
            ma50 = df_lookup.loc[ticker, 'fiftyDayAverage']
            peak = pos.get('peak_price', pos['entry_price'])
            
            exit_reason = None

            # EXIT: MA50 break
            if current_price < ma50:
                exit_reason = "MA50 Break"
            # EXIT: Regime-adaptive trailing stop
            elif peak and current_price < peak * trail_keep:
                exit_reason = f"Trail Stop (-{trail_pct*100:.0f}% [{regime}])"

            if exit_reason:
                print(f"STOP HIT: {ticker} (Price: {current_price:.2f} | Reason: {exit_reason})")
                sells.append((ticker, current_price, exit_reason))
                alerts.append(f"\xf0\x9f\x9a\xa8 *STOP LOSS HIT* \xf0\x9f\x9a\xa8\\n*{ticker}*\\nPrice: {current_price:,.2f}\\nReason: {exit_reason}")
                
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

        # Also write to centralized trade log CSV
        try:
            import csv
            log_path = "data/dna3_trade_log.csv"
            file_exists = os.path.exists(log_path)
            with open(log_path, 'a', newline='') as csvf:
                writer = csv.DictWriter(csvf, fieldnames=['Ticker','Action','Date','Price','PnL','PnL%','Reason'])
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    'Ticker': ticker, 'Action': 'SELL',
                    'Date': datetime.now().strftime("%Y-%m-%d"),
                    'Price': round(exit_price, 2),
                    'PnL': round((exit_price - entry_price) * 1, 2),
                    'PnL%': round(pnl_pct, 2),
                    'Reason': reason
                })
        except Exception:
            pass  # Don't let CSV write failure block the main flow
        
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
    Orchestrates finding new buys based on the V2.2 Momentum conditions.
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
        
    # Filter out stocks we already own or are already tracking
    db.cursor.execute("SELECT ticker FROM watchlist UNION SELECT ticker FROM portfolio")
    existing_tickers = {row[0] for row in db.cursor.fetchall()}
    buyers = buyers[~buyers['ticker'].isin(existing_tickers)]
    
    if buyers.empty:
        print("No NEW, unalerted BUY signals generated today.")
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
    
    # V2.2: Include regime state in heartbeat
    try:
        import yfinance as yf
        _nifty = yf.Ticker("^NSEI").history(period="2y")
        _regime = classify_regime(
            float(_nifty['Close'].iloc[-1]),
            float(_nifty['Close'].rolling(200).mean().iloc[-1]),
            float(_nifty['High'].rolling(252).max().iloc[-1])
        )
    except Exception:
        _regime = 'UNKNOWN'

    # V2.2: Load portfolio health for heartbeat
    portfolio_line = ""
    try:
        import json
        snap_file = os.path.join(os.path.dirname(__file__), "data", "dna3_portfolio_snapshot.json")
        if os.path.exists(snap_file):
            with open(snap_file) as f:
                snap = json.load(f)
            equity = snap.get('equity', 0)
            count = snap.get('count', 0)
            max_pos = snap.get('config', {}).get('max_positions', 15)
            initial = 1000000
            ret_pct = (equity - initial) / initial * 100

            # Rank all holdings by proximity to exit (Danger zone first, then closest distance)
            holdings = snap.get('portfolio', [])
            if holdings:
                portfolio_line += f"🏠 <b>Holdings:</b> {count}/{max_pos}\n"
                portfolio_line += f"💰 <b>Portfolio:</b> Rs {equity:,.0f} ({ret_pct:+.1f}%)\n"
                portfolio_line += f"\n<b>--- Distance to Exit ---</b>\n"
                
                def exit_dist(h):
                    cmp = h.get('Price', 1)
                    ts = h.get('Trail_Stop', -1)
                    ma = h.get('MA50', -1)
                    hard_floor = max(ts, ma) if ma > 0 and ts > 0 else (ts if ts > 0 else cmp*0.5)
                    # pct distance to floor
                    return (cmp / hard_floor - 1) * 100 if hard_floor > 0 else 999
                
                sorted_holds = sorted(holdings, key=exit_dist)
                for h in sorted_holds:
                    ticker = h['Ticker'].replace('.NS', '')
                    dist = exit_dist(h)
                    danger = " 🚨" if h.get('Danger') else (" ⚠️" if dist < 8.0 else "")
                    portfolio_line += f"  {ticker:<10} +{dist:.1f}%{danger}\n"
            else:
                portfolio_line += f"🏠 <b>Holdings:</b> {count}/{max_pos}\n"
                portfolio_line += f"💰 <b>Portfolio:</b> Rs {equity:,.0f} ({ret_pct:+.1f}%)\n"
    except:
        pass

    trail_pct = {'BULL': 15, 'CAUTION': 12, 'BEAR': 8, 'CRISIS': 6}.get(_regime, '?')

    msg = f"✅ <b>EOD Engine Run: SUCCESS</b>\n\n"
    msg += f"🕒 <b>Time:</b> {now_str} IST\n"
    msg += f"⚙️ <b>Mode:</b> {mode.upper()}\n"
    msg += f"📊 <b>Regime:</b> {_regime} | Trail: {trail_pct}%\n"
    msg += portfolio_line
    msg += f"🔍 <b>Universe Scanned:</b> {universe_size} stocks\n\n"
    msg += "<i>Systems operating normally.</i>"
    
    try:
        from utils.telegram_notifier import send_telegram_message
        send_telegram_message(msg)
    except: pass
    
    try:
        from utils.email_notifier import _send_email, is_email_configured
        if is_email_configured():
            _send_email(f"✅ EOD Engine Run: SUCCESS", f"<pre>{msg.replace('<b>', '').replace('</b>', '').replace('<i>', '').replace('</i>', '')}</pre>")
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
