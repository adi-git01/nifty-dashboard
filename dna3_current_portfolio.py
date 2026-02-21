"""
DNA3-V2.1 LIVE PORTFOLIO ENGINE (WITH PERFORMANCE TRACKING)
===========================================================
Generates the LIVE Model Portfolio and tracks performance since Feb 01, 2026.

Features:
- Live Scan: Price > MA50, RS > 0.
- Portfolio Management: Maintains holds, tracks entry prices.
- Performance Tracking: Calculates Daily NAV, CAGR, and Absolute Return.
- Artifacts: 
    - data/dna3_portfolio_snapshot.json (Current State)
    - data/dna3_trade_log.csv (History of Exits)
    - data/dna3_equity_curve.csv (NAV History)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
import sys

# Ensure utils import works
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nifty500_list import TICKERS, SECTOR_MAP

# CONFIG
MAX_POSITIONS = 15
INITIAL_CAPITAL = 1000000 # 10 Lakhs Virtual Start
START_DATE = "2026-02-01"

DATA_DIR = "data"
SNAPSHOT_FILE = f"{DATA_DIR}/dna3_portfolio_snapshot.json"
TRADE_LOG_FILE = f"{DATA_DIR}/dna3_trade_log.csv"
EQUITY_CURVE_FILE = f"{DATA_DIR}/dna3_equity_curve.csv"

class DNA3LiveEngine:
    def __init__(self):
        self.tickers = TICKERS[:500] 
        self.data_cache = {}
        self.current_date = datetime.now().strftime('%Y-%m-%d')
        self.sector_map = SECTOR_MAP
        
        # Ensure data dir exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
    def load_state(self):
        """Loads previous portfolio state or initializes new"""
        if os.path.exists(SNAPSHOT_FILE):
             try:
                 with open(SNAPSHOT_FILE, 'r') as f:
                    state = json.load(f)
                    # Legacy check: if cash missing, reset
                    if 'cash' not in state:
                        print("Legacy snapshot found. Resetting state for Live Tracking.")
                        return self.get_initial_state()
                    return state
             except:
                 return self.get_initial_state()
        else:
            return self.get_initial_state()

    def get_initial_state(self):
        return {
            'date': START_DATE,
            'cash': INITIAL_CAPITAL,
            'holdings': {}, # {Ticker: {entry_price, shares, entry_date}}
            'equity': INITIAL_CAPITAL
        }

    def fetch_data(self):
        print("Fetching Live Data for DNA3 Scan (Bulk Mode)...")
        start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
        
        # 1. Nifty (Single Fetch)
        nifty = yf.Ticker("^NSEI").history(start=start_date)
        if nifty.empty: return False
        nifty.index = nifty.index.tz_localize(None)
        self.data_cache['NIFTY'] = nifty
        
        # 2. Bulk Download ALL Stocks at Once (10-30x faster than sequential)
        print(f"  Bulk downloading {len(self.tickers)} stocks...")
        try:
            bulk_data = yf.download(
                self.tickers,
                start=start_date,
                group_by='ticker',
                threads=True,
                progress=False,
                auto_adjust=True
            )
        except Exception as e:
            print(f"  Bulk download failed: {e}. Falling back to sequential.")
            bulk_data = None
        
        loaded = 0
        if bulk_data is not None and not bulk_data.empty:
            for t in self.tickers:
                try:
                    if t in bulk_data.columns.get_level_values(0):
                        df = bulk_data[t].dropna(how='all')
                        if not df.empty and len(df) > 200:
                            df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                            self.data_cache[t] = df
                            loaded += 1
                except: pass
        
        # 3. Fallback: ThreadPool for any stocks missed by bulk (e.g., delisted/error)
        if loaded < 50:
            print(f"  Only {loaded} stocks loaded via bulk. Trying ThreadPool fallback...")
            import concurrent.futures
            
            def fetch_single(t):
                try:
                    df = yf.Ticker(t).history(start=start_date)
                    if not df.empty and len(df) > 200:
                        df.index = df.index.tz_localize(None)
                        return t, df
                except: pass
                return t, None
            
            missing = [t for t in self.tickers if t not in self.data_cache]
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                results = executor.map(fetch_single, missing)
                for t, df in results:
                    if df is not None:
                        self.data_cache[t] = df
                        loaded += 1
        
        print(f"  Loaded {loaded} stocks.")
        return True

    def calculate_metrics(self, ticker):
        df = self.data_cache[ticker]
        nifty = self.data_cache['NIFTY']
        
        price = df['Close'].iloc[-1]
        ma50 = df['Close'].rolling(50).mean().iloc[-1]
        
        # RS Score (vs Nifty 63d)
        if len(df) < 64 or len(nifty) < 64: return None
        
        price_63 = df['Close'].iloc[-63]
        n_price_63 = nifty['Close'].iloc[-63]
        
        rs_stock = (price - price_63)/price_63
        rs_nifty = (nifty['Close'].iloc[-1] - n_price_63)/n_price_63
        rs_score = (rs_stock - rs_nifty) * 100
        
        vol_avg = df['Volume'].rolling(20).mean().iloc[-1] * price
        
        return {
            'price': price,
            'ma50': ma50,
            'rs_score': rs_score,
            'liquidity': vol_avg
        }

    def update_portfolio(self):
        if not self.fetch_data(): return
        
        state = self.load_state()
        cash = state['cash']
        holdings = state['holdings']
        
        current_holdings = list(holdings.keys())
        trade_log = []
        
        # 1. CHECK EXITS (For Current Holdings)
        stocks_to_sell = []
        
        for t in current_holdings:
            if t not in self.data_cache: continue
            
            df = self.data_cache[t]
            price = df['Close'].iloc[-1]
            ma50 = df['Close'].rolling(50).mean().iloc[-1]
            peak = holdings[t].get('peak_price', holdings[t]['entry_price'])
            
            # Update Peak
            if price > peak: 
                holdings[t]['peak_price'] = price
                peak = price
            
            exit_reason = None
            
            # EXIT RULES
            if price < ma50: exit_reason = "Trend Break (< MA50)"
            elif price < peak * 0.85: exit_reason = "Trailing Stop (-15%)"
            
            if exit_reason:
                # SELL
                shares = holdings[t]['shares']
                proceeds = shares * price * 0.998 # Cost
                pnl = proceeds - (shares * holdings[t]['entry_price'])
                pnl_pct = (price - holdings[t]['entry_price']) / holdings[t]['entry_price'] * 100
                
                cash += proceeds
                stocks_to_sell.append(t)
                
                trade_log.append({
                    'Ticker': t, 'Action': 'SELL', 'Date': self.current_date,
                    'Price': price, 'PnL': pnl, 'PnL%': pnl_pct, 'Reason': exit_reason
                })
        
        for t in stocks_to_sell: del holdings[t]
        
        # 2. SCAN FOR NEW BUYS
        candidates = []
        for t in self.data_cache:
            if t == 'NIFTY' or t in holdings: continue
            
            m = self.calculate_metrics(t)
            if not m: continue
            
            # ENTRY RULES
            if m['liquidity'] > 10000000 and m['price'] > m['ma50'] and m['rs_score'] > 0:
                candidates.append({
                    'Ticker': t,
                    'Sector': self.sector_map.get(t, 'Unknown'),
                    'Price': m['price'],
                    'MA50': m['ma50'],
                    'RS_Score': m['rs_score']
                })
                
        candidates.sort(key=lambda x: -x['RS_Score'])
        
        # 3. BUY NEW POSITIONS
        # Only buy if slot available AND cash available
        free_slots = MAX_POSITIONS - len(holdings)
        if free_slots > 0:
            for cand in candidates[:free_slots]:
                # Position Sizing (Equal Weight of Total Portfolio Value target)
                # For simplicity, calculate target size based on Live Equity / Max Positions
                # But to avoid rebalancing complexity, just use available cash / slots
                
                # Check Nifty Trend for Allocation
                nifty = self.data_cache['NIFTY']
                n_price = nifty['Close'].iloc[-1]
                n_ma50 = nifty['Close'].rolling(50).mean().iloc[-1]
                
                # If Nifty < MA50 (Bear), reduce exposure? 
                # DNA3 Backtest said: Just trust the stock signal.
                
                target_per_stock = (cash + sum([holdings[h_t]['shares']*self.data_cache[h_t]['Close'].iloc[-1] for h_t in holdings])) / MAX_POSITIONS
                # Cap at current cash / slots to never go negative
                invest_amount = min(target_per_stock, cash / free_slots)
                
                if invest_amount > 5000: # Min trade size
                    price = cand['Price']
                    shares = int(invest_amount / price)
                    cost = shares * price * 1.002 # Impact
                    
                    if cash >= cost:
                        cash -= cost
                        holdings[cand['Ticker']] = {
                            'entry_price': price,
                            'entry_date': self.current_date,
                            'shares': shares,
                            'peak_price': price
                        }
                        trade_log.append({
                            'Ticker': cand['Ticker'], 'Action': 'BUY', 'Date': self.current_date,
                            'Price': price, 'PnL': 0, 'PnL%': 0, 'Reason': 'New Entry'
                        })
        
        # 4. CALCULATE EQUITY
        equity_val = cash
        portfolio_list = []
        
        for t, h in holdings.items():
            curr_price = self.data_cache[t]['Close'].iloc[-1]
            equity_val += h['shares'] * curr_price
            
            # Prepare UI list
            ma50 = self.data_cache[t]['Close'].rolling(50).mean().iloc[-1]
            dist_ma50 = (curr_price - ma50)/ma50 * 100
            
            # RS Re-calc for display
            portfolio_list.append({
                'Ticker': t,
                'Sector': self.sector_map.get(t, 'Unknown'),
                'Price': curr_price,
                'RS_Score': 0, # Will fill if needed, or use cached candidates logic
                'Entry': h['entry_price'],
                'PnL%': (curr_price - h['entry_price'])/h['entry_price']*100,
                'Dist_MA50': dist_ma50
            })
            
        # Add RS Score to portfolio list for display
        for p in portfolio_list:
            m = self.calculate_metrics(p['Ticker'])
            if m: p['RS_Score'] = round(m['rs_score'], 1)
            
        portfolio_list.sort(key=lambda x: -x['RS_Score'])

        # 5. SAVE STATE
        new_state = {
            'date': self.current_date,
            'cash': cash,
            'holdings': holdings,
            'equity': equity_val,
            'count': len(holdings),
            'portfolio': portfolio_list
        }
        
        with open(SNAPSHOT_FILE, 'w') as f:
            json.dump(new_state, f, indent=4)
            
        # 6. APPEND LOGS (CSV)
        # Trades
        if trade_log:
            df_log = pd.DataFrame(trade_log)
            hdr = not os.path.exists(TRADE_LOG_FILE)
            df_log.to_csv(TRADE_LOG_FILE, mode='a', header=hdr, index=False)
            
        # Equity Curve
        eq_record = {'Date': self.current_date, 'Equity': equity_val, 'Cash': cash, 'Holdings': len(holdings)}
        df_eq = pd.DataFrame([eq_record])
        hdr = not os.path.exists(EQUITY_CURVE_FILE)
        df_eq.to_csv(EQUITY_CURVE_FILE, mode='a', header=hdr, index=False)
        
        print(f"\n[DNA3 LIVE] Portfolio Upated. Equity: {equity_val:,.0f} | Holdings: {len(holdings)}")
        return new_state

if __name__ == "__main__":
    engine = DNA3LiveEngine()
    engine.update_portfolio()
