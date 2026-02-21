"""
Enhanced Playbook Backtest Engine v2.0
======================================
Improvements:
- 15 stock minimum portfolio
- Daily regime check
- 25% sector cap  
- Trailing stops (not fixed targets)
- Fundamentals logging (PE, earnings)
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
MIN_POSITIONS = 15
MAX_POSITIONS = 20
POSITION_SIZE = 0.065  # ~6.5% per position (to fit 15 stocks)
SLIPPAGE = 0.005  # 0.5% per trade
SECTOR_CAP = 0.25  # Max 25% per sector

# Stop losses by regime
STOPS = {
    'Strong_Bear': -0.22,
    'Recovery': -0.10,
    'Mild_Bull': -0.16,
    'Strong_Bull': -0.15
}

# Trailing stop activation and trail amount
TRAIL_ACTIVATION = 0.10  # Activate trailing stop after 10% gain
TRAIL_AMOUNT = 0.07       # Trail by 7% from peak

TIME_STOP_DAYS = 45

# Stock Universe (expanded)
UNIVERSE = {
    'Consumer': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'TITAN.NS', 'DABUR.NS', 'MARICO.NS', 'TRENT.NS', 'COLPAL.NS', 'GODREJCP.NS'],
    'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'LUPIN.NS', 'AUROPHARMA.NS', 'TORNTPHARM.NS', 'ALKEM.NS'],
    'IT_Services': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTIM.NS', 'COFORGE.NS', 'PERSISTENT.NS', 'LTTS.NS', 'MPHASIS.NS'],
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'INDUSINDBK.NS', 'FEDERALBNK.NS', 'BANKBARODA.NS', 'PNB.NS'],
    'Metals': ['TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 'VEDL.NS', 'JINDALSTEL.NS', 'NMDC.NS'],
    'Auto': ['MARUTI.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS', 'ASHOKLEY.NS', 'BALKRISIND.NS'],
    'Industrials': ['LT.NS', 'SIEMENS.NS', 'ABB.NS', 'HAVELLS.NS', 'CUMMINSIND.NS', 'POLYCAB.NS', 'CROMPTON.NS', 'VOLTAS.NS', 'BHARATFORG.NS'],
    'Energy': ['RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'IOC.NS', 'NTPC.NS', 'POWERGRID.NS', 'TATAPOWER.NS', 'GAIL.NS']
}

SECTOR_PRIORITY = {
    'Strong_Bear': ['Auto', 'Metals', 'Banking', 'Industrials', 'Energy'],
    'Recovery': ['IT_Services', 'Banking', 'Industrials', 'Consumer'],
    'Mild_Bull': ['Consumer', 'Pharma', 'IT_Services', 'Banking'],
    'Strong_Bull': ['Metals', 'Industrials', 'Auto', 'Energy']
}

class EnhancedBacktestEngine:
    def __init__(self):
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.trade_log = []
        self.portfolio_history = []
        self.data_cache = {}
        self.fundamentals_cache = {}
        
    def fetch_data(self):
        """Fetch all required data including fundamentals."""
        print("Fetching historical data...")
        
        # Nifty
        nifty = yf.Ticker("^NSEI").history(start="2025-01-01", end=END_DATE)
        nifty.index = nifty.index.tz_localize(None)
        self.data_cache['NIFTY'] = nifty
        
        # Stocks
        all_tickers = [t for sector in UNIVERSE.values() for t in sector]
        for ticker in all_tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start="2025-01-01", end=END_DATE)
                if not df.empty:
                    df.index = df.index.tz_localize(None)
                    self.data_cache[ticker] = df
                    
                    # Get fundamentals
                    try:
                        info = stock.info
                        self.fundamentals_cache[ticker] = {
                            'pe': info.get('trailingPE', None),
                            'forward_pe': info.get('forwardPE', None),
                            'peg': info.get('pegRatio', None),
                            'earnings_growth': info.get('earningsGrowth', None),
                            'revenue_growth': info.get('revenueGrowth', None),
                            'market_cap': info.get('marketCap', None)
                        }
                    except:
                        self.fundamentals_cache[ticker] = {}
            except:
                pass
        
        print(f"  Loaded {len(self.data_cache)-1} stocks + Nifty")
        
    def get_regime(self, date):
        """Determine market regime on a given date."""
        nifty = self.data_cache['NIFTY']
        idx = nifty.index.searchsorted(date)
        if idx < 200 or idx >= len(nifty):
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
        """Calculate trend score."""
        if ticker not in self.data_cache:
            return 50
        
        df = self.data_cache[ticker]
        idx = df.index.searchsorted(date)
        if idx < 200 or idx >= len(df):
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
        """Calculate volume state."""
        if ticker not in self.data_cache:
            return "Flat"
        
        df = self.data_cache[ticker]
        idx = df.index.searchsorted(date)
        if idx < 60 or idx >= len(df):
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
        """Get closing price."""
        if ticker not in self.data_cache:
            return None
        df = self.data_cache[ticker]
        mask = df.index <= date
        if mask.sum() == 0:
            return None
        return df.loc[mask, 'Close'].iloc[-1]

    def get_sector_exposure(self, sector):
        """Calculate current exposure to a sector."""
        total_value = self.get_portfolio_value(pd.Timestamp.now())
        sector_value = 0
        for ticker, pos in self.positions.items():
            if pos['sector'] == sector:
                price = self.get_price(ticker, pd.Timestamp.now())
                if price:
                    sector_value += pos['shares'] * price
        return sector_value / total_value if total_value > 0 else 0
    
    def find_candidates(self, date, regime):
        """Find entry candidates with sector cap enforcement."""
        candidates = []
        priority_sectors = SECTOR_PRIORITY.get(regime, list(UNIVERSE.keys()))
        
        for sector in priority_sectors:
            # Check sector cap
            current_exposure = sum(1 for t, p in self.positions.items() if p['sector'] == sector)
            max_sector_positions = int(MAX_POSITIONS * SECTOR_CAP)
            if current_exposure >= max_sector_positions:
                continue
                
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
                
                # Entry rules by regime
                valid = False
                if regime == 'Strong_Bear' and trend <= 20 and volume in ['Jump', 'Drop']:
                    valid = True
                elif regime == 'Recovery' and 20 <= trend <= 40 and volume in ['Flat', 'Drop']:
                    valid = True
                elif regime == 'Mild_Bull' and trend <= 25 and volume in ['Big_Drop', 'Drop', 'Flat']:
                    valid = True
                elif regime == 'Strong_Bull' and 60 <= trend <= 85 and volume in ['Big_Drop', 'Flat']:
                    valid = True
                
                if valid:
                    fund = self.fundamentals_cache.get(ticker, {})
                    candidates.append({
                        'ticker': ticker,
                        'sector': sector,
                        'trend': trend,
                        'volume': volume,
                        'price': price,
                        'pe': fund.get('pe'),
                        'earnings_growth': fund.get('earnings_growth'),
                        'revenue_growth': fund.get('revenue_growth')
                    })
        
        # Sort by trend
        if regime in ['Strong_Bear', 'Mild_Bull', 'Recovery']:
            candidates.sort(key=lambda x: x['trend'])
        else:
            candidates.sort(key=lambda x: -x['trend'])
        
        return candidates[:MAX_POSITIONS - len(self.positions)]
    
    def enter_position(self, ticker, sector, price, date, regime, trend, volume, pe, earnings_growth, revenue_growth):
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
            'mfe': 0,
            'peak_price': price,
            'trailing_active': False
        }
        
        self.trade_log.append({
            'date': date,
            'ticker': ticker.replace('.NS', ''),
            'action': 'BUY',
            'price': round(price, 2),
            'shares': shares,
            'value': round(shares * price, 2),
            'regime': regime,
            'trend': trend,
            'volume': volume,
            'sector': sector,
            'pe': round(pe, 1) if pe else 'N/A',
            'earnings_growth': f"{earnings_growth*100:.1f}%" if earnings_growth else 'N/A',
            'revenue_growth': f"{revenue_growth*100:.1f}%" if revenue_growth else 'N/A',
            'reason': 'Entry Signal'
        })
    
    def check_exits(self, date):
        """Check all positions for exit conditions with trailing stops."""
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
            
            # Update peak price for trailing stop
            if price > pos['peak_price']:
                pos['peak_price'] = price
            
            # Activate trailing stop after 10% gain
            if current_return >= TRAIL_ACTIVATION and not pos['trailing_active']:
                pos['trailing_active'] = True
                pos['stop'] = pos['peak_price'] * (1 - TRAIL_AMOUNT)
            
            # Update trailing stop
            if pos['trailing_active']:
                new_trail = pos['peak_price'] * (1 - TRAIL_AMOUNT)
                if new_trail > pos['stop']:
                    pos['stop'] = new_trail
            
            # Check underwater days
            if price < entry_price:
                pos['days_underwater'] += 1
            else:
                pos['days_underwater'] = 0
            
            # Exit conditions
            reason = None
            if price <= pos['stop']:
                if pos['trailing_active']:
                    reason = 'Trailing Stop'
                else:
                    reason = 'Stop Loss'
            elif pos['days_underwater'] >= TIME_STOP_DAYS:
                reason = 'Time Stop'
            
            if reason:
                to_exit.append((ticker, price, reason))
        
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
        
        fund = self.fundamentals_cache.get(ticker, {})
        
        self.trade_log.append({
            'date': date,
            'ticker': ticker.replace('.NS', ''),
            'action': 'SELL',
            'price': round(price, 2),
            'shares': pos['shares'],
            'value': round(proceeds, 2),
            'regime': pos['regime'],
            'trend': pos['trend'],
            'volume': pos['volume'],
            'sector': pos['sector'],
            'pe': round(fund.get('pe'), 1) if fund.get('pe') else 'N/A',
            'reason': reason,
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 2),
            'days_held': days_held,
            'mae': round(pos['mae'] * 100, 2),
            'mfe': round(pos['mfe'] * 100, 2)
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
        """Run the backtest with daily checks."""
        self.fetch_data()
        
        print(f"\nRunning ENHANCED backtest: {START_DATE} to {END_DATE}")
        print(f"Initial Capital: Rs.{INITIAL_CAPITAL:,.0f}")
        print(f"Min Positions: {MIN_POSITIONS}, Max: {MAX_POSITIONS}")
        print(f"Sector Cap: {SECTOR_CAP*100:.0f}%")
        print(f"Trailing Stop: Activates at +{TRAIL_ACTIVATION*100:.0f}%, trails by {TRAIL_AMOUNT*100:.0f}%")
        print()
        
        # Generate DAILY dates
        start = pd.Timestamp(START_DATE)
        end = pd.Timestamp(END_DATE)
        all_dates = pd.date_range(start, end, freq='B')  # Business days
        
        last_regime = None
        
        for date in all_dates:
            regime = self.get_regime(date)
            if regime == "Unknown":
                continue
            
            # Log regime changes
            if regime != last_regime and last_regime is not None:
                print(f"{date.strftime('%Y-%m-%d')} | *** REGIME CHANGE: {last_regime} -> {regime} ***")
            last_regime = regime
            
            # Check exits first (daily)
            self.check_exits(date)
            
            # Find new entries if below minimum
            if len(self.positions) < MIN_POSITIONS:
                candidates = self.find_candidates(date, regime)
                for c in candidates:
                    if len(self.positions) >= MAX_POSITIONS:
                        break
                    self.enter_position(
                        c['ticker'], c['sector'], c['price'], date,
                        regime, c['trend'], c['volume'],
                        c['pe'], c['earnings_growth'], c['revenue_growth']
                    )
            
            # Record weekly snapshot
            if date.weekday() == 4:  # Friday
                portfolio_value = self.get_portfolio_value(date)
                sectors = {}
                for t, p in self.positions.items():
                    sectors[p['sector']] = sectors.get(p['sector'], 0) + 1
                
                self.portfolio_history.append({
                    'date': date,
                    'value': portfolio_value,
                    'positions': len(self.positions),
                    'regime': regime,
                    'sectors': str(sectors)
                })
                
                print(f"{date.strftime('%Y-%m-%d')} | {regime:12} | Pos: {len(self.positions):2} | Rs.{portfolio_value:,.0f} | Sectors: {sectors}")
        
        # Exit all at end
        for ticker in list(self.positions.keys()):
            price = self.get_price(ticker, end)
            if price:
                self.exit_position(ticker, price, end, 'End of Backtest')
        
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive backtest report."""
        print("\n" + "="*70)
        print("ENHANCED BACKTEST COMPLETE")
        print("="*70)
        
        # Save trade log
        trades_df = pd.DataFrame(self.trade_log)
        trades_df.to_csv('analysis_2026/backtest_v2_trades.csv', index=False)
        print(f"\nTrade log saved: analysis_2026/backtest_v2_trades.csv")
        
        # Portfolio history
        hist_df = pd.DataFrame(self.portfolio_history)
        hist_df.to_csv('analysis_2026/backtest_v2_portfolio.csv', index=False)
        
        # Metrics
        sells = trades_df[trades_df['action'] == 'SELL']
        
        if len(sells) > 0:
            winners = sells[sells['pnl'] > 0]
            losers = sells[sells['pnl'] < 0]
            
            total_pnl = sells['pnl'].sum()
            win_rate = len(winners) / len(sells) * 100
            avg_win = winners['pnl_pct'].mean() if len(winners) > 0 else 0
            avg_loss = losers['pnl_pct'].mean() if len(losers) > 0 else 0
            avg_mae = sells['mae'].mean()
            avg_mfe = sells['mfe'].mean()
            avg_days = sells['days_held'].mean()
            
            # Exit reasons
            stop_loss = len(sells[sells['reason'] == 'Stop Loss'])
            trail_stop = len(sells[sells['reason'] == 'Trailing Stop'])
            time_stop = len(sells[sells['reason'] == 'Time Stop'])
            end_exit = len(sells[sells['reason'] == 'End of Backtest'])
        else:
            total_pnl = win_rate = avg_win = avg_loss = avg_mae = avg_mfe = avg_days = 0
            stop_loss = trail_stop = time_stop = end_exit = 0
        
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
        
        print(f"\n{'='*70}")
        print("PERFORMANCE SUMMARY")
        print('='*70)
        print(f"Initial Capital:     Rs.{INITIAL_CAPITAL:,.0f}")
        print(f"Final Value:         Rs.{final_value:,.0f}")
        print(f"Total Return:        {total_return:+.2f}%")
        print(f"Nifty Return:        {nifty_return:+.2f}%")
        print(f"Alpha:               {total_return - nifty_return:+.2f}%")
        print(f"\nMax Drawdown:        {max_drawdown:.2f}%")
        print(f"\nTotal Trades:        {len(sells)}")
        print(f"Win Rate:            {win_rate:.1f}%")
        print(f"Winners:             {len(winners) if 'winners' in dir() else 0}")
        print(f"Losers:              {len(losers) if 'losers' in dir() else 0}")
        print(f"Avg Win:             {avg_win:+.1f}%")
        print(f"Avg Loss:            {avg_loss:.1f}%")
        print(f"Avg Days Held:       {avg_days:.0f}")
        print(f"\nExit Breakdown:")
        print(f"  Stop Loss:         {stop_loss}")
        print(f"  Trailing Stop:     {trail_stop}")
        print(f"  Time Stop:         {time_stop}")
        print(f"  End of Backtest:   {end_exit}")
        print(f"\nAvg MAE (Pain):      {avg_mae:.1f}%")
        print(f"Avg MFE (Potential): {avg_mfe:.1f}%")
        print('='*70)

if __name__ == "__main__":
    engine = EnhancedBacktestEngine()
    engine.run()
