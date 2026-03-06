"""
OptComp-V22 WEEKLY SUMMARY
===========================
Sends a weekly portfolio performance summary every Saturday morning.
Covers: portfolio vs Nifty, regime duration, breadth, trade stats.

Usage:
  python weekly_summary.py          # Run once (send summary now)

Schedule via GitHub Actions cron or Windows Task Scheduler for Saturday 9 AM IST.
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

SNAPSHOT_FILE = "data/dna3_portfolio_snapshot.json"
TRADE_LOG_FILE = "data/dna3_trade_log.csv"
EQUITY_CURVE_FILE = "data/dna3_equity_curve.csv"
INITIAL_CAPITAL = 1000000


def load_data():
    """Load all required data files."""
    data = {}

    # Portfolio snapshot
    if os.path.exists(SNAPSHOT_FILE):
        with open(SNAPSHOT_FILE) as f:
            data['snapshot'] = json.load(f)
    else:
        print("No portfolio snapshot found.")
        return None

    # Equity curve
    if os.path.exists(EQUITY_CURVE_FILE):
        data['equity'] = pd.read_csv(EQUITY_CURVE_FILE)
    else:
        data['equity'] = pd.DataFrame()

    # Trade log
    if os.path.exists(TRADE_LOG_FILE):
        data['trades'] = pd.read_csv(TRADE_LOG_FILE)
    else:
        data['trades'] = pd.DataFrame()

    return data


def build_telegram_summary(data):
    """Build Telegram weekly summary message."""
    snap = data['snapshot']
    equity_df = data['equity']
    trades_df = data['trades']

    equity = snap.get('equity', INITIAL_CAPITAL)
    regime = snap.get('regime', 'UNKNOWN')
    count = snap.get('count', 0)
    config = snap.get('config', {})
    trail_pct = config.get('trailing_stop', 0.15)
    max_pos = config.get('max_positions', 15)

    total_ret = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    start = datetime(2026, 2, 1)
    days = (datetime.now() - start).days
    cagr = ((equity / INITIAL_CAPITAL) ** (365.25 / max(days, 1)) - 1) * 100 if equity > 0 else 0

    # Nifty comparison (from equity curve if available)
    nifty_ret = "N/A"
    try:
        import yfinance as yf
        nifty = yf.Ticker("^NSEI").history(period="1mo")
        if len(nifty) >= 5:
            week_ago_price = nifty['Close'].iloc[-6] if len(nifty) >= 6 else nifty['Close'].iloc[0]
            curr_nifty = nifty['Close'].iloc[-1]
            nifty_week_ret = (curr_nifty - week_ago_price) / week_ago_price * 100
            nifty_ret = f"{nifty_week_ret:+.1f}%"

            # Nifty vs MA200
            nifty_2y = yf.Ticker("^NSEI").history(period="2y")
            if len(nifty_2y) >= 200:
                sma200 = nifty_2y['Close'].rolling(200).mean().iloc[-1]
                nifty_vs_sma = (curr_nifty - sma200) / sma200 * 100
                high_52w = nifty_2y['High'].rolling(252).max().iloc[-1]
                dd_52w = (curr_nifty - high_52w) / high_52w * 100
    except:
        nifty_vs_sma = None
        dd_52w = None
        curr_nifty = None

    # Weekly P&L from equity curve
    week_pnl = "N/A"
    if not equity_df.empty and len(equity_df) >= 2:
        last_week_eq = equity_df.iloc[-min(6, len(equity_df))]['Equity']
        week_pnl_val = (equity - last_week_eq) / last_week_eq * 100
        week_pnl = f"{week_pnl_val:+.1f}%"

    # Trade stats this week
    weekly_trades = pd.DataFrame()
    if not trades_df.empty:
        try:
            trades_df['Date'] = pd.to_datetime(trades_df['Date'])
            week_start = datetime.now() - timedelta(days=7)
            weekly_trades = trades_df[trades_df['Date'] >= week_start]
        except:
            pass

    buys = len(weekly_trades[weekly_trades['Action'] == 'BUY']) if not weekly_trades.empty else 0
    sells = len(weekly_trades[weekly_trades['Action'] == 'SELL']) if not weekly_trades.empty else 0
    avg_sell_pnl = weekly_trades[weekly_trades['Action'] == 'SELL']['PnL%'].mean() if sells > 0 else 0

    # Peak equity & drawdown
    peak_eq = equity_df['Equity'].max() if not equity_df.empty else equity
    dd_from_peak = (equity - peak_eq) / peak_eq * 100

    # Regime duration
    regime_days = "?"
    if not equity_df.empty and 'Regime' in equity_df.columns:
        try:
            latest_regime = equity_df.iloc[-1].get('Regime', regime)
            # Count consecutive days of same regime
            regime_streak = 0
            for i in range(len(equity_df) - 1, -1, -1):
                if equity_df.iloc[i].get('Regime', '') == latest_regime:
                    regime_streak += 1
                else:
                    break
            regime_days = str(regime_streak)
        except:
            pass

    regime_icon = {'BULL': '\U0001f7e2', 'CAUTION': '\U0001f7e1', 'BEAR': '\U0001f7e0', 'CRISIS': '\U0001f534'}.get(regime, '\u26aa')

    msg = f"<b>\U0001f4ca OptComp-V22 Weekly Summary</b>\n"
    msg += f"<i>Week ending {datetime.now().strftime('%d %b %Y')}</i>\n\n"

    msg += f"<b>--- Portfolio ---</b>\n"
    msg += f"  Value: Rs {equity:,.0f} ({total_ret:+.1f}%)\n"
    msg += f"  Week P&L: {week_pnl}\n"
    msg += f"  CAGR: {cagr:+.1f}%\n"
    msg += f"  Peak: Rs {peak_eq:,.0f} | DD: {dd_from_peak:.1f}%\n"
    msg += f"  Holdings: {count}/{max_pos}\n\n"

    msg += f"<b>--- Market ---</b>\n"
    msg += f"  {regime_icon} Regime: {regime} ({regime_days} days)\n"
    msg += f"  Trail Stop: {trail_pct*100:.0f}%\n"
    msg += f"  Nifty Week: {nifty_ret}\n"
    if nifty_vs_sma is not None:
        msg += f"  Nifty vs SMA200: {nifty_vs_sma:+.1f}%\n"
    if dd_52w is not None:
        msg += f"  Nifty DD from 52W: {dd_52w:.1f}%\n"
    msg += "\n"

    msg += f"<b>--- Trades This Week ---</b>\n"
    msg += f"  Buys: {buys} | Sells: {sells}\n"
    if sells > 0:
        msg += f"  Avg Exit P&L: {avg_sell_pnl:+.1f}%\n"

    if not weekly_trades.empty:
        for _, t in weekly_trades.iterrows():
            action = t.get('Action', '')
            ticker = str(t.get('Ticker', '')).replace('.NS', '')
            pnl = t.get('PnL%', 0)
            icon = '\U0001f7e2' if action == 'BUY' else ('\U0001f534' if pnl < 0 else '\U0001f7e2')
            msg += f"  {icon} {action} {ticker} ({pnl:+.1f}%)\n"

    msg += f"\n<i>Auto-generated Saturday Review | OptComp-V22</i>"
    return msg


def build_email_summary(data):
    """Build rich HTML email for weekly summary."""
    snap = data['snapshot']
    equity_df = data['equity']
    trades_df = data['trades']

    equity = snap.get('equity', INITIAL_CAPITAL)
    regime = snap.get('regime', 'UNKNOWN')
    count = snap.get('count', 0)
    config = snap.get('config', {})
    trail_pct = config.get('trailing_stop', 0.15)

    total_ret = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    start = datetime(2026, 2, 1)
    days = (datetime.now() - start).days
    cagr = ((equity / INITIAL_CAPITAL) ** (365.25 / max(days, 1)) - 1) * 100 if equity > 0 else 0

    peak_eq = equity_df['Equity'].max() if not equity_df.empty else equity
    dd_from_peak = (equity - peak_eq) / peak_eq * 100

    ret_color = '#00C853' if total_ret >= 0 else '#FF5252'
    regime_color = {'BULL': '#00C853', 'CAUTION': '#FFD600', 'BEAR': '#FF6D00', 'CRISIS': '#FF1744'}.get(regime, '#888')

    # Weekly trades
    trade_rows = ""
    if not trades_df.empty:
        try:
            trades_df['Date'] = pd.to_datetime(trades_df['Date'])
            week_start = datetime.now() - timedelta(days=7)
            weekly = trades_df[trades_df['Date'] >= week_start]
            for _, t in weekly.iterrows():
                action = t.get('Action', '')
                ticker = str(t.get('Ticker', '')).replace('.NS', '')
                pnl = t.get('PnL%', 0)
                pnl_color = '#00C853' if pnl >= 0 else '#FF5252'
                reason = t.get('Reason', '')
                trade_rows += f"""
                <tr>
                    <td><strong>{action}</strong></td>
                    <td>{ticker}</td>
                    <td style="color:{pnl_color}">{pnl:+.1f}%</td>
                    <td style="font-size:11px;color:#888">{reason}</td>
                </tr>"""
        except:
            pass

    # Holdings table
    holdings_rows = ""
    for h in snap.get('portfolio', [])[:15]:
        ticker = h.get('Ticker', '').replace('.NS', '')
        price = h.get('Price', 0)
        pnl = h.get('PnL%', 0)
        rs = h.get('RS_Score', 0)
        pnl_class = 'color:#00C853' if pnl >= 0 else 'color:#FF5252'
        holdings_rows += f"""
        <tr>
            <td><strong>{ticker}</strong></td>
            <td>Rs {price:,.0f}</td>
            <td style="{pnl_class}">{pnl:+.1f}%</td>
            <td>{rs:.1f}</td>
        </tr>"""

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f0f23; color: #eee; padding: 20px; }}
            .container {{ max-width: 650px; margin: 0 auto; background: #1a1a2e; border-radius: 16px; padding: 30px; }}
            .header {{ background: linear-gradient(135deg, #1a237e 0%, #4a148c 100%); padding: 25px; border-radius: 12px; text-align: center; margin-bottom: 25px; }}
            .header h1 {{ margin: 0; color: white; font-size: 22px; }}
            .header p {{ margin: 8px 0 0 0; color: rgba(255,255,255,0.7); font-size: 14px; }}
            .stats {{ display: flex; justify-content: space-around; margin-bottom: 25px; }}
            .stat-box {{ text-align: center; background: rgba(255,255,255,0.05); padding: 15px 10px; border-radius: 10px; flex: 1; margin: 0 4px; }}
            .stat-value {{ font-size: 20px; font-weight: bold; }}
            .stat-label {{ font-size: 10px; color: #888; margin-top: 5px; text-transform: uppercase; }}
            .regime-badge {{ display: inline-block; background: {regime_color}22; border: 1px solid {regime_color}; color: {regime_color}; padding: 4px 14px; border-radius: 20px; font-size: 13px; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th {{ text-align: left; padding: 8px; color: #888; font-size: 11px; border-bottom: 1px solid rgba(255,255,255,0.1); }}
            td {{ padding: 8px; font-size: 12px; border-bottom: 1px solid rgba(255,255,255,0.05); }}
            .section {{ margin-top: 25px; }}
            .section h3 {{ color: #667eea; margin: 0 0 10px 0; font-size: 15px; }}
            .footer {{ text-align: center; font-size: 11px; color: #555; margin-top: 25px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1); }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>\U0001f4ca OptComp-V22 Weekly Review</h1>
                <p>Week ending {datetime.now().strftime('%d %b %Y')}</p>
            </div>

            <div style="text-align:center; margin-bottom: 20px;">
                <span class="regime-badge">{regime} ({regime_days} days) | Trail: {trail_pct*100:.0f}%</span>
            </div>

            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value" style="color:{ret_color}">{total_ret:+.1f}%</div>
                    <div class="stat-label">Total Return</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" style="color:#667eea">Rs {equity:,.0f}</div>
                    <div class="stat-label">Portfolio</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{dd_from_peak:.1f}%</div>
                    <div class="stat-label">DD from Peak</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{count}</div>
                    <div class="stat-label">Holdings</div>
                </div>
            </div>
    """

    if trade_rows:
        html += f"""
            <div class="section">
                <h3>Trades This Week</h3>
                <table>
                    <tr><th>Action</th><th>Stock</th><th>P&L</th><th>Reason</th></tr>
                    {trade_rows}
                </table>
            </div>
        """

    html += f"""
            <div class="section">
                <h3>Current Holdings</h3>
                <table>
                    <tr><th>Stock</th><th>Price</th><th>P&L</th><th>RS</th></tr>
                    {holdings_rows}
                </table>
            </div>

            <div class="footer">
                OptComp-V22 | Alpha Trend Dashboard | CAGR: {cagr:+.1f}%
            </div>
        </div>
    </body>
    </html>
    """
    return html


def send_weekly_summary():
    """Main function: Load data and send via Telegram + Email."""
    data = load_data()
    if not data:
        print("No data available for weekly summary.")
        return False

    results = {}

    # 1. TELEGRAM
    try:
        from utils.telegram_notifier import send_telegram_message, is_telegram_configured
        if is_telegram_configured():
            msg = build_telegram_summary(data)
            success, resp = send_telegram_message(msg)
            results['telegram'] = success
            print(f"Telegram: {'Sent' if success else 'Failed'} - {resp}")
        else:
            print("Telegram: Not configured.")
            results['telegram'] = False
    except Exception as e:
        print(f"Telegram Error: {e}")
        results['telegram'] = False

    # 2. EMAIL
    try:
        from utils.email_notifier import _send_email, is_email_configured
        if is_email_configured():
            html = build_email_summary(data)
            subject = f"\U0001f4ca Weekly Review — {datetime.now().strftime('%d %b %Y')}"
            success, resp = _send_email(subject, html)
            results['email'] = success
            print(f"Email: {'Sent' if success else 'Failed'} - {resp}")
        else:
            print("Email: Not configured.")
            results['email'] = False
    except Exception as e:
        print(f"Email Error: {e}")
        results['email'] = False

    return results


if __name__ == "__main__":
    print(f"[{datetime.now()}] OptComp-V22 Weekly Summary")
    print("=" * 50)
    send_weekly_summary()
