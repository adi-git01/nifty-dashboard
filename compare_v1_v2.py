"""
Comparative Backtest: Playbook v1 vs v2 (5 Years)
=================================================
Period: Feb 2021 - Feb 2026

Strategy v1 (Original):
- Max 10 stocks
- Sector Concentration Allowed
- Fixed Targets (+15-20%)
- Stop Loss (-15%)

Strategy v2 (Enhanced):
- Max 20 stocks (User requested)
- Sector Cap (25%)
- Trailing Stops (Trail 7% after 10% gain)
- Daily Regime Checks

Goal: Compare performance over 5 years.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# === DATA LOADER ===
class DataLoader:
    def __init__(self):
        self.data = {}
        
    def fetch_data(self, tickers):
        print("Fetching 5 years of data...")
        start_date = (datetime.now() - timedelta(days=365*5 + 200)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Nifty
        nifty = yf.Ticker("^NSEI").history(start=start_date, end=end_date)
        nifty.index = nifty.index.tz_localize(None)
        self.data['NIFTY'] = nifty
        
        for t in tickers:
            try:
                df = yf.Ticker(t).history(start=start_date, end=end_date)
                if not df.empty:
                    df.index = df.index.tz_localize(None)
                    self.data[t] = df
            except: pass
            
    def get_price(self, ticker, date):
        if ticker not in self.data: return None
        df = self.data[ticker]
        mask = df.index <= date
        if mask.sum() == 0: return None
        return df.loc[mask, 'Close'].iloc[-1]

    def get_regime(self, date):
        nifty = self.data['NIFTY']
        idx = nifty.index.searchsorted(date)
        if idx < 200: return "Unknown"
        window = nifty.iloc[max(0, idx-200):idx+1]
        price = window['Close'].iloc[-1]
        ma50 = window['Close'].rolling(50).mean().iloc[-1]
        ma200 = window['Close'].rolling(200).mean().iloc[-1]
        
        if ma50 > ma200: return "Strong_Bull" if price > ma50 else "Mild_Bull"
        else: return "Strong_Bear" if price < ma50 else "Recovery"

    def get_trend(self, ticker, date):
        if ticker not in self.data: return 50
        df = self.data[ticker]
        idx = df.index.searchsorted(date)
        if idx < 200: return 50
        window = df.iloc[max(0, idx-252):idx+1]
        price = window['Close'].iloc[-1]
        ma50 = window['Close'].rolling(50).mean().iloc[-1]
        ma200 = window['Close'].rolling(200).mean().iloc[-1]
        score = 50
        if price > ma50: score += 15
        else: score -= 10
        if price > ma200: score += 15
        else: score -= 15
        if ma50 > ma200: score += 10
        return score

# === STRATEGY CLASS ===
class Strategy:
    def __init__(self, name, loader, max_stocks, use_trailing, sector_cap):
        self.name = name
        self.loader = loader
        self.max_stocks = max_stocks
        self.use_trailing = use_trailing
        self.sector_cap = sector_cap
        
        self.capital = 1000000
        self.positions = {}
        self.history = []
        self.trade_log = []
        
    def run_day(self, date):
        regime = self.loader.get_regime(date)
        if regime == "Unknown": return
        
        # 1. Check Exits
        to_exit = []
        for t, pos in self.positions.items():
            price = self.loader.get_price(t, date)
            if not price: continue
            
            # Trailing Stop Logic (v2)
            if self.use_trailing:
                if price > pos['peak']: pos['peak'] = price
                gain = (price - pos['entry']) / pos['entry']
                if gain > 0.10: # Activation
                    trail = pos['peak'] * 0.93 # 7% trail
                    if trail > pos['stop']: pos['stop'] = trail
            
            # Exit Conditions
            reason = None
            gain = (price - pos['entry']) / pos['entry']
            
            if price < pos['stop']: reason = "Stop Loss"
            elif not self.use_trailing and gain > 0.15: reason = "Target Hit" # v1 Fixed Target
            elif (date - pos['date']).days > 45 and gain < 0: reason = "Time Stop"
            
            if reason: to_exit.append((t, price, reason))
            
        for t, price, reason in to_exit:
            pos = self.positions[t]
            val = pos['shares'] * price * 0.995 # Slippage
            self.capital += val
            self.trade_log.append({'date': date, 'ticker': t, 'action': 'SELL', 'pnl': (val - pos['cost'])/pos['cost'], 'reason': reason})
            del self.positions[t]
            
        # 2. Find Entries
        if len(self.positions) < self.max_stocks:
            # Simple candidate scan (simplified for speed)
            # In real system, this uses the full universe logic
            # Here we randomly pick valid candidates from universe to simulate selection
            # prioritization is handled by sector priority list in full version
            pass 
            # Note: For this comparison script to be accurate, we need the full scanning logic.
            # I will inject the abbreviated scanning logic below.
            
            candidates = self.scan(date, regime)
            for c in candidates:
                if len(self.positions) >= self.max_stocks: break
                
                # Sector Cap Check
                if self.sector_cap:
                    exposure = sum(1 for p in self.positions.values() if p['sector'] == c['sector'])
                    if exposure >= int(self.max_stocks * 0.25): continue
                
                price = c['price']
                size = self.capital / (self.max_stocks - len(self.positions) + 5) # Dynamic sizing
                size = min(size, self.capital)
                shares = int(size / price)
                
                if shares > 0:
                    cost = shares * price * 1.005
                    if self.capital >= cost:
                        self.capital -= cost
                        stop = price * 0.85 # -15% fixed stop initially
                        self.positions[c['ticker']] = {
                            'entry': price, 'shares': shares, 'cost': cost,
                            'stop': stop, 'peak': price, 'date': date, 'sector': c['sector']
                        }
        
        # 3. Log Value
        val = self.capital
        for t, pos in self.positions.items():
            p = self.loader.get_price(t, date)
            if p: val += pos['shares'] * p
        self.history.append({'date': date, 'value': val})

    def scan(self, date, regime):
        # Simplified scanning logic for the comparison
        cands = []
        # Priority map
        priority = {
            'Strong_Bull': ['Metals', 'Auto', 'Industrials'],
            'Mild_Bull': ['Consumer', 'Pharma'],
            'Recovery': ['Banking', 'IT_Services'],
            'Strong_Bear': ['Auto', 'Metals'] # Short/Bear strategies
        }
        target_sectors = priority.get(regime, [])
        
        for sector, tickers in UNIVERSE.items():
            if sector not in target_sectors: continue
            for t in tickers:
                if t in self.positions: continue
                if t not in self.loader.data: continue
                
                trend = self.loader.get_trend(t, date)
                price = self.loader.get_price(t, date)
                if not price: continue
                
                valid = False
                if regime == 'Strong_Bull' and trend > 60: valid = True
                elif regime == 'Mild_Bull' and trend < 30: valid = True
                elif regime == 'Recovery' and trend > 20: valid = True
                
                if valid:
                    cands.append({'ticker': t, 'sector': sector, 'price': price, 'trend': trend})
        
        cands.sort(key=lambda x: -x['trend']) # Momentum sort
        return cands

# Universe Definition
UNIVERSE = {
    'Consumer': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'TITAN.NS', 'DABUR.NS', 'MARICO.NS', 'TRENT.NS'],
    'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'LUPIN.NS', 'AUROPHARMA.NS'],
    'IT_Services': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTIM.NS', 'COFORGE.NS'],
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'INDUSINDBK.NS'],
    'Metals': ['TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 'VEDL.NS', 'JINDALSTEL.NS'],
    'Auto': ['MARUTI.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS'],
    'Industrials': ['LT.NS', 'SIEMENS.NS', 'ABB.NS', 'HAVELLS.NS', 'CUMMINSIND.NS'],
    'Energy': ['RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'IOC.NS', 'NTPC.NS', 'POWERGRID.NS']
}

# === MAIN RUNNER ===
loader = DataLoader()
tickers = [t for s in UNIVERSE.values() for t in s]
loader.fetch_data(tickers)

# Setup Strategies
v1 = Strategy("v1_Original", loader, max_stocks=10, use_trailing=False, sector_cap=False)
v2 = Strategy("v2_Enhanced", loader, max_stocks=20, use_trailing=True, sector_cap=True)

print("Running simulations (5 Years)...")
dates = pd.date_range(start=(datetime.now() - timedelta(days=365*5)), end=datetime.now(), freq='B')

for d in dates:
    v1.run_day(d)
    v2.run_day(d)
    
# Report
v1_ret = (v1.history[-1]['value'] - 1000000)/10000
v2_ret = (v2.history[-1]['value'] - 1000000)/10000

print("\n" + "="*50)
print(f"5-YEAR RESULTS (Feb 2021 - Feb 2026)")
print(f"v1 (Original, 10 stocks): {v1_ret:.2f}%")
print(f"v2 (Enhanced, 20 stocks): {v2_ret:.2f}%")
print("="*50)

# Metrics
def get_metrics(hist, trades):
    # CAGR
    years = 5
    final = hist[-1]['value']
    cagr = (final/1000000)**(1/years) - 1
    
    # Drawdown
    vals = pd.Series([h['value'] for h in hist])
    dd = (vals.cummax() - vals)/vals.cummax()
    max_dd = dd.max()
    
    return cagr*100, max_dd*100, len(trades)

c1, d1, t1 = get_metrics(v1.history, v1.trade_log)
c2, d2, t2 = get_metrics(v2.history, v2.trade_log)

print(f"v1: CAGR {c1:.1f}% | MaxDD {d1:.1f}% | Trades {t1}")
print(f"v2: CAGR {c2:.1f}% | MaxDD {d2:.1f}% | Trades {t2}")

df1 = pd.DataFrame(v1.history)
df2 = pd.DataFrame(v2.history)
df1.to_csv('analysis_2026/v1_history.csv', index=False)
df2.to_csv('analysis_2026/v2_history.csv', index=False)
