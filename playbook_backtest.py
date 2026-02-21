"""
Playbook Backtest Engine
=========================
Backtests the Market Playbook strategy for the last 6 months.

Rules:
1. Weekly regime detection
2. Entry based on regime-specific setup (Trend + Volume)
3. Exit on: Stop Loss, Time Stop, or Target
4. Track all trades with metrics

Output: Trade log, portfolio performance, comparison to Nifty
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

# === CONFIG ===
START_DATE = "2025-08-01"
END_DATE = "2026-02-07"
INITIAL_CAPITAL = 1000000  # 10 Lakh
MAX_POSITIONS = 10
POSITION_SIZE = 0.10  # 10% per position
SLIPPAGE = 0.005  # 0.5% per trade

# Stop losses by regime
STOPS = {
    'Strong_Bear': -0.22,
    'Recovery': -0.10,
    'Mild_Bull': -0.16,
    'Strong_Bull': -0.15
}

TIME_STOP_DAYS = 45  # Exit if underwater for 45 days

# Target returns (for partial profit taking)
TARGETS = {
    'Strong_Bear': 0.30,
    'Recovery': 0.15,
    'Mild_Bull': 0.22,
    'Strong_Bull': 0.14
}

# Stock Universe
UNIVERSE = {
    'Consumer': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'TITAN.NS', 'DABUR.NS', 'MARICO.NS', 'TRENT.NS'],
    'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'LUPIN.NS', 'AUROPHARMA.NS', 'TORNTPHARM.NS'],
    'IT_Services': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTIM.NS', 'COFORGE.NS', 'PERSISTENT.NS', 'LTTS.NS'],
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'INDUSINDBK.NS', 'FEDERALBNK.NS'],
    'Metals': ['TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 'VEDL.NS', 'JINDALSTEL.NS'],
    'Auto': ['MARUTI.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS'],
    'Industrials': ['LT.NS', 'SIEMENS.NS', 'ABB.NS', 'HAVELLS.NS', 'CUMMINSIND.NS', 'POLYCAB.NS', 'CROMPTON.NS']
}

# Sector priority by regime
SECTOR_PRIORITY = {
    'Strong_Bear': ['Auto', 'Metals', 'Banking', 'Industrials'],
    'Recovery': ['IT_Services', 'Banking', 'Industrials'],
    'Mild_Bull': ['Consumer', 'Pharma', 'IT_Services'],
    'Strong_Bull': ['Metals', 'Industrials', 'Auto']
}

class BacktestEngine:
    def __init__(self):
        self.capital = INITIAL_CAPITAL
        self.positions = {}  # {ticker: {entry_price, entry_date, shares, stop, regime, trend, volume}}
        self.trade_log = []
        self.portfolio_history = []
        self.data_cache = {}
        
    def fetch_data(self):
        """Fetch all required data."""
        print("Fetching historical data...")
        
        # Nifty
        nifty = yf.Ticker("^NSEI").history(start="2025-01-01", end=END_DATE)
        nifty.index = nifty.index.tz_localize(None)
        self.data_cache['NIFTY'] = nifty
        
        # Stocks
        all_tickers = [t for sector in UNIVERSE.values() for t in sector]
        for ticker in all_tickers:
            try:
                df = yf.Ticker(ticker).history(start="2025-01-01", end=END_DATE)
                if not df.empty:
                    df.index = df.index.tz_localize(None)
                    self.data_cache[ticker] = df
            except:
                pass
        
        print(f"  Loaded {len(self.data_cache)-1} stocks + Nifty")
        
    def get_regime(self, date):
        """Determine market regime on a given date."""
        nifty = self.data_cache['NIFTY']
        idx = nifty.index.searchsorted(date)
        if idx < 200:
            return "Unknown"
        
        window = nifty.iloc[max(0, idx-200):idx+1]
        if len(window) < 50:
            return "Unknown"
            
        price = window['Close'].iloc[-1]
        ma50 = window['Close'].rolling(50).mean().iloc[-1]
        ma200 = window['Close'].rolling(200).mean().iloc[-1]
        
        if pd.isna(ma50) or pd.isna(ma200):
            return "Unknown"
        
        if ma50 > ma200:
            return "Strong_Bull" if price > ma50 else "Mild_Bull"
        else:
            return "Strong_Bear" if price < ma50 else "Recovery"
    
    def get_trend_score(self, ticker, date):
        """Calculate trend score for a stock on a date."""
        if ticker not in self.data_cache:
            return 50
        
        df = self.data_cache[ticker]
        idx = df.index.searchsorted(date)
        if idx < 200:
            return 50
        
        window = df.iloc[max(0, idx-252):idx+1]
        if len(window) < 50:
            return 50
            
        price = window['Close'].iloc[-1]
        ma50 = window['Close'].rolling(50).mean().iloc[-1]
        ma200 = window['Close'].rolling(200).mean().iloc[-1]
        high_52 = window['Close'].max()
        low_52 = window['Close'].min()
        
        score = 50
        if not pd.isna(ma50):
            if price > ma50: score += 15
            else: score -= 10
        if not pd.isna(ma200):
            if price > ma200: score += 15
            else: score -= 15
            if not pd.isna(ma50) and ma50 > ma200: score += 10
            else: score -= 5
        
        if high_52 - low_52 > 0:
            pos = (price - low_52) / (high_52 - low_52)
            score += int((pos - 0.5) * 30)
        
        return max(0, min(100, score))
    
    def get_volume_state(self, ticker, date):
        """Calculate volume state for a stock on a date."""
        if ticker not in self.data_cache:
            return "Flat"
        
        df = self.data_cache[ticker]
        idx = df.index.searchsorted(date)
        if idx < 60:
            return "Flat"
        
        window = df.iloc[max(0, idx-60):idx+1]
        vol_10 = window['Volume'].iloc[-10:].mean()
        vol_60 = window['Volume'].mean()
        
        if vol_60 == 0:
            return "Flat"
        
        ratio = vol_10 / vol_60
        if ratio < 0.5: return "Big_Drop"
        elif ratio < 0.7: return "Drop"
        elif ratio > 2.0: return "Spike"
        elif ratio > 1.5: return "Jump"
        else: return "Flat"
    
    def get_price(self, ticker, date):
        """Get closing price on a date."""
        if ticker not in self.data_cache:
            return None
        df = self.data_cache[ticker]
        idx = df.index.searchsorted(date)
        if idx >= len(df) or idx == 0:
            return None
        return df['Close'].iloc[idx-1] if df.index[idx] > date else df['Close'].iloc[idx]
    
    def find_candidates(self, date, regime):
        """Find entry candidates based on regime."""
        candidates = []
        priority_sectors = SECTOR_PRIORITY.get(regime, list(UNIVERSE.keys()))
        
        for sector in priority_sectors:
            for ticker in UNIVERSE.get(sector, []):
                if ticker in self.positions:
                    continue
                if ticker not in self.data_cache:
                    continue
                
                trend = self.get_trend_score(ticker, date)
                volume = self.get_volume_state(ticker, date)
                price = self.get_price(ticker, date)
                
                if price is None:
                    continue
                
                # Apply regime-specific entry rules
                valid = False
                if regime == 'Strong_Bear' and trend <= 20 and volume in ['Jump', 'Drop']:
                    valid = True
                elif regime == 'Recovery' and 20 <= trend <= 40 and volume in ['Flat', 'Drop']:
                    valid = True
                elif regime == 'Mild_Bull' and trend <= 20 and volume in ['Big_Drop', 'Drop', 'Flat']:
                    valid = True
                elif regime == 'Strong_Bull' and 60 <= trend <= 80 and volume in ['Big_Drop', 'Flat']:
                    valid = True
                
                if valid:
                    candidates.append({
                        'ticker': ticker,
                        'sector': sector,
                        'trend': trend,
                        'volume': volume,
                        'price': price
                    })
        
        # Sort by trend (lower is better for dip strategies)
        if regime in ['Strong_Bear', 'Mild_Bull', 'Recovery']:
            candidates.sort(key=lambda x: x['trend'])
        else:
            candidates.sort(key=lambda x: -x['trend'])
        
        return candidates[:MAX_POSITIONS - len(self.positions)]
    
    def enter_position(self, ticker, sector, price, date, regime, trend, volume):
        """Enter a new position."""
        position_value = self.capital * POSITION_SIZE
        shares = int(position_value / price)
        if shares <= 0:
            return
        
        cost = shares * price * (1 + SLIPPAGE)
        if cost > self.capital:
            return
        
        self.capital -= cost
        stop_price = price * (1 + STOPS[regime])
        
        self.positions[ticker] = {
            'entry_price': price,
            'entry_date': date,
            'shares': shares,
            'stop': stop_price,
            'regime': regime,
            'trend': trend,
            'volume': volume,
            'sector': sector,
            'days_underwater': 0,
            'mae': 0,
            'mfe': 0
        }
        
        self.trade_log.append({
            'date': date,
            'ticker': ticker.replace('.NS', ''),
            'action': 'BUY',
            'price': price,
            'shares': shares,
            'value': shares * price,
            'regime': regime,
            'trend': trend,
            'volume': volume,
            'reason': 'Entry Signal'
        })
    
    def check_exits(self, date):
        """Check all positions for exit conditions."""
        to_exit = []
        
        for ticker, pos in self.positions.items():
            price = self.get_price(ticker, date)
            if price is None:
                continue
            
            entry_price = pos['entry_price']
            current_return = (price - entry_price) / entry_price
            
            # Update MAE/MFE
            if current_return < pos['mae']:
                pos['mae'] = current_return
            if current_return > pos['mfe']:
                pos['mfe'] = current_return
            
            # Update underwater days
            if price < entry_price:
                pos['days_underwater'] += 1
            else:
                pos['days_underwater'] = 0
            
            # Check exit conditions
            reason = None
            if price <= pos['stop']:
                reason = 'Stop Loss'
            elif pos['days_underwater'] >= TIME_STOP_DAYS:
                reason = 'Time Stop'
            elif current_return >= TARGETS.get(pos['regime'], 0.15):
                reason = 'Target Hit'
            
            if reason:
                to_exit.append((ticker, price, reason))
        
        # Execute exits
        for ticker, price, reason in to_exit:
            self.exit_position(ticker, price, date, reason)
    
    def exit_position(self, ticker, price, date, reason):
        """Exit a position."""
        pos = self.positions[ticker]
        proceeds = pos['shares'] * price * (1 - SLIPPAGE)
        self.capital += proceeds
        
        pnl = proceeds - (pos['shares'] * pos['entry_price'])
        pnl_pct = pnl / (pos['shares'] * pos['entry_price']) * 100
        days_held = (date - pos['entry_date']).days
        
        self.trade_log.append({
            'date': date,
            'ticker': ticker.replace('.NS', ''),
            'action': 'SELL',
            'price': price,
            'shares': pos['shares'],
            'value': proceeds,
            'regime': pos['regime'],
            'trend': pos['trend'],
            'volume': pos['volume'],
            'reason': reason,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'days_held': days_held,
            'mae': pos['mae'] * 100,
            'mfe': pos['mfe'] * 100
        })
        
        del self.positions[ticker]
    
    def get_portfolio_value(self, date):
        """Calculate total portfolio value."""
        value = self.capital
        for ticker, pos in self.positions.items():
            price = self.get_price(ticker, date)
            if price:
                value += pos['shares'] * price
        return value
    
    def run(self):
        """Run the backtest."""
        self.fetch_data()
        
        print(f"\nRunning backtest: {START_DATE} to {END_DATE}")
        print(f"Initial Capital: Rs.{INITIAL_CAPITAL:,.0f}")
        print()
        
        # Generate weekly dates
        start = pd.Timestamp(START_DATE)
        end = pd.Timestamp(END_DATE)
        dates = pd.date_range(start, end, freq='W-FRI')
        
        for date in dates:
            # Get regime
            regime = self.get_regime(date)
            if regime == "Unknown":
                continue
            
            # Check exits first
            self.check_exits(date)
            
            # Find new entries
            candidates = self.find_candidates(date, regime)
            for c in candidates:
                self.enter_position(
                    c['ticker'], c['sector'], c['price'], date,
                    regime, c['trend'], c['volume']
                )
            
            # Record portfolio value
            portfolio_value = self.get_portfolio_value(date)
            nifty_price = self.get_price('NIFTY', date) or 0
            
            self.portfolio_history.append({
                'date': date,
                'value': portfolio_value,
                'positions': len(self.positions),
                'regime': regime,
                'nifty': nifty_price
            })
            
            print(f"{date.strftime('%Y-%m-%d')} | Regime: {regime:12} | Positions: {len(self.positions):2} | Portfolio: Rs.{portfolio_value:,.0f}")
        
        # Exit all remaining positions at end
        for ticker in list(self.positions.keys()):
            price = self.get_price(ticker, end)
            if price:
                self.exit_position(ticker, price, end, 'End of Backtest')
        
        self.generate_report()
    
    def generate_report(self):
        """Generate backtest report."""
        print("\n" + "="*60)
        print("BACKTEST COMPLETE")
        print("="*60)
        
        # Trade Log
        trades_df = pd.DataFrame(self.trade_log)
        trades_df.to_csv('analysis_2026/backtest_trades.csv', index=False)
        print(f"\nTrade log saved: analysis_2026/backtest_trades.csv")
        
        # Portfolio History
        hist_df = pd.DataFrame(self.portfolio_history)
        hist_df.to_csv('analysis_2026/backtest_portfolio.csv', index=False)
        
        # Calculate metrics
        sells = trades_df[trades_df['action'] == 'SELL']
        
        if len(sells) > 0:
            total_pnl = sells['pnl'].sum()
            win_rate = (sells['pnl'] > 0).mean() * 100
            avg_win = sells[sells['pnl'] > 0]['pnl_pct'].mean() if len(sells[sells['pnl'] > 0]) > 0 else 0
            avg_loss = sells[sells['pnl'] < 0]['pnl_pct'].mean() if len(sells[sells['pnl'] < 0]) > 0 else 0
            avg_mae = sells['mae'].mean()
            avg_mfe = sells['mfe'].mean()
            avg_days = sells['days_held'].mean()
        else:
            total_pnl = win_rate = avg_win = avg_loss = avg_mae = avg_mfe = avg_days = 0
        
        final_value = hist_df['value'].iloc[-1] if len(hist_df) > 0 else INITIAL_CAPITAL
        total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        max_drawdown = ((hist_df['value'].cummax() - hist_df['value']) / hist_df['value'].cummax()).max() * 100
        
        # Nifty return
        if 'NIFTY' in self.data_cache:
            nifty = self.data_cache['NIFTY']
            nifty_start = nifty[nifty.index >= pd.Timestamp(START_DATE)]['Close'].iloc[0]
            nifty_end = nifty['Close'].iloc[-1]
            nifty_return = (nifty_end - nifty_start) / nifty_start * 100
        else:
            nifty_return = 0
        
        print(f"\n{'='*60}")
        print("PERFORMANCE SUMMARY")
        print('='*60)
        print(f"Initial Capital:     Rs.{INITIAL_CAPITAL:,.0f}")
        print(f"Final Value:         Rs.{final_value:,.0f}")
        print(f"Total Return:        {total_return:+.2f}%")
        print(f"Nifty Return:        {nifty_return:+.2f}%")
        print(f"Alpha:               {total_return - nifty_return:+.2f}%")
        print(f"\nMax Drawdown:        {max_drawdown:.2f}%")
        print(f"\nTotal Trades:        {len(sells)}")
        print(f"Win Rate:            {win_rate:.1f}%")
        print(f"Avg Win:             {avg_win:+.1f}%")
        print(f"Avg Loss:            {avg_loss:.1f}%")
        print(f"Avg Days Held:       {avg_days:.0f}")
        print(f"\nAvg MAE (Pain):      {avg_mae:.1f}%")
        print(f"Avg MFE (Potential): {avg_mfe:.1f}%")
        print('='*60)

if __name__ == "__main__":
    engine = BacktestEngine()
    engine.run()
