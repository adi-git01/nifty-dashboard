"""
OptComp-V21 LIVE PORTFOLIO ENGINE
=================================
Production portfolio engine implementing the optimized Composite RS strategy.

Strategy: OptComp-V21
  - RS Signal: Composite (10% 1W + 50% 1M + 40% 3M)
  - Rebalance: Every 13 trading days
  - G-Factor: Off
  - Regime Sizing: Off (full deployment)
  - Positions: 10 equal-weight
  - Breadth Gate: Skip new buys when market breadth < 30%

Entry Rules (only on rebalance days):
  1. Price > 50-day MA
  2. Composite RS > 0 (outperforming Nifty on blended timeframes)
  3. Daily traded value > 1 Crore
  4. Market breadth (% above 50DMA proxy) >= 30%
  5. Rank by composite RS, buy top candidates

Exit Rules (checked EVERY run):
  1. Price < 50-day MA -> SELL (Trend Break)
  2. Price < Peak * 0.85 -> SELL (15% Trailing Stop)

Artifacts:
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

# ============================================================
# STRATEGY CONFIGURATION — OptComp-V21 (Locked from Research)
# ============================================================
STRATEGY_NAME = "OptComp-V21"
MAX_POSITIONS = 10
INITIAL_CAPITAL = 1000000  # 10 Lakhs
START_DATE = "2026-02-23"  # Fresh start
REBALANCE_DAYS = 13        # Trading days between rebalances

# Composite RS Weights (10% 1W + 50% 1M + 40% 3M)
RS_WEIGHTS = [
    (5,   0.10),   # 1 Week
    (21,  0.50),   # 1 Month
    (63,  0.40),   # 3 Months (Quarter)
]

# Breadth Gate Threshold
BREADTH_NARROW_THRESHOLD = 30  # % of stocks above 50DMA — below this = skip buys

# Risk Management
TRAILING_STOP_PCT = 0.85     # Sell if price drops 15% from peak
MIN_LIQUIDITY = 10_000_000   # Rs 1 Crore min daily value

DATA_DIR = "data"
SNAPSHOT_FILE = f"{DATA_DIR}/dna3_portfolio_snapshot.json"
TRADE_LOG_FILE = f"{DATA_DIR}/dna3_trade_log.csv"
EQUITY_CURVE_FILE = f"{DATA_DIR}/dna3_equity_curve.csv"


class OptCompV21Engine:
    def __init__(self):
        self.tickers = TICKERS[:500]
        self.data_cache = {}
        self.current_date = datetime.now().strftime('%Y-%m-%d')
        self.sector_map = SECTOR_MAP
        os.makedirs(DATA_DIR, exist_ok=True)

    def load_state(self):
        """Load previous portfolio state or initialize new."""
        if os.path.exists(SNAPSHOT_FILE):
            try:
                with open(SNAPSHOT_FILE, 'r') as f:
                    state = json.load(f)
                    if 'strategy' in state and state['strategy'] == STRATEGY_NAME:
                        return state
                    else:
                        print(f"  Old strategy detected ({state.get('strategy', 'unknown')}). Resetting for {STRATEGY_NAME}.")
                        return self.get_initial_state()
            except:
                return self.get_initial_state()
        return self.get_initial_state()

    def get_initial_state(self):
        return {
            'strategy': STRATEGY_NAME,
            'date': START_DATE,
            'cash': INITIAL_CAPITAL,
            'holdings': {},
            'equity': INITIAL_CAPITAL,
            'last_rebalance_date': None,   # Force first rebalance
            'rebalance_count': 0,
        }

    def fetch_data(self):
        """Bulk-download Nifty + 500 stocks."""
        print(f"  [{STRATEGY_NAME}] Fetching market data...")
        start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')

        # 1. Nifty
        nifty = yf.Ticker("^NSEI").history(start=start_date)
        if nifty.empty:
            print("  ERROR: Could not fetch Nifty data.")
            return False
        nifty.index = nifty.index.tz_localize(None)
        self.data_cache['NIFTY'] = nifty

        # 2. Bulk download
        print(f"  Bulk downloading {len(self.tickers)} stocks...")
        try:
            bulk_data = yf.download(
                self.tickers, start=start_date,
                group_by='ticker', threads=True, progress=False, auto_adjust=True
            )
        except Exception as e:
            print(f"  Bulk download failed: {e}")
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
                except:
                    pass

        # 3. Fallback for missed stocks
        if loaded < 50:
            print(f"  Only {loaded} stocks via bulk. Trying ThreadPool fallback...")
            import concurrent.futures
            def fetch_single(t):
                try:
                    df = yf.Ticker(t).history(start=start_date)
                    if not df.empty and len(df) > 200:
                        df.index = df.index.tz_localize(None)
                        return t, df
                except:
                    pass
                return t, None

            missing = [t for t in self.tickers if t not in self.data_cache]
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                results = executor.map(fetch_single, missing)
                for t, df in results:
                    if df is not None:
                        self.data_cache[t] = df
                        loaded += 1

        print(f"  Loaded {loaded} stocks. Nifty: {len(nifty)} days.")
        return True

    def composite_rs(self, ticker):
        """
        Composite Relative Strength: 10% 1W + 50% 1M + 40% 3M
        Returns weighted RS score vs Nifty.
        """
        df = self.data_cache[ticker]
        nifty = self.data_cache['NIFTY']
        
        price = df['Close'].iloc[-1]
        nifty_price = nifty['Close'].iloc[-1]

        if len(df) < 64 or len(nifty) < 64:
            return None

        rs_total = 0.0
        for period, weight in RS_WEIGHTS:
            if len(df) < period + 1 or len(nifty) < period + 1:
                return None
            stock_past = df['Close'].iloc[-period]
            nifty_past = nifty['Close'].iloc[-period]
            
            rs_stock = (price / stock_past - 1)
            rs_nifty = (nifty_price / nifty_past - 1)
            rs_total += (rs_stock - rs_nifty) * 100 * weight

        return rs_total

    def calculate_metrics(self, ticker):
        """Calculate all entry metrics for a stock."""
        df = self.data_cache[ticker]
        
        price = df['Close'].iloc[-1]
        ma50 = df['Close'].rolling(50).mean().iloc[-1]
        vol_avg = df['Volume'].rolling(20).mean().iloc[-1] * price
        
        rs = self.composite_rs(ticker)
        if rs is None:
            return None

        return {
            'price': price,
            'ma50': ma50,
            'rs_score': rs,
            'liquidity': vol_avg,
        }

    def calculate_breadth(self):
        """
        Calculate market breadth: % of stocks above their 50-day MA.
        Uses actual MA50 from price data (not proxy).
        """
        above_50 = 0
        total = 0
        
        for t, df in self.data_cache.items():
            if t == 'NIFTY':
                continue
            if len(df) < 51:
                continue
            try:
                price = df['Close'].iloc[-1]
                ma50 = df['Close'].rolling(50).mean().iloc[-1]
                total += 1
                if price > ma50:
                    above_50 += 1
            except:
                pass
        
        if total == 0:
            return 50.0  # Default to neutral
        
        return round(above_50 / total * 100, 1)

    def is_rebalance_day(self, state):
        """Check if today is a rebalance day (every 13 trading days)."""
        last_reb = state.get('last_rebalance_date')
        
        if last_reb is None:
            return True  # First run, force rebalance

        # Count trading days since last rebalance using Nifty calendar
        nifty = self.data_cache['NIFTY']
        try:
            last_reb_dt = pd.Timestamp(last_reb)
            today_dt = nifty.index[-1]
            
            # Count trading days between last_reb and today
            trading_days = nifty.index[(nifty.index > last_reb_dt) & (nifty.index <= today_dt)]
            days_since = len(trading_days)
            
            return days_since >= REBALANCE_DAYS
        except:
            return True  # If error, force rebalance

    def update_portfolio(self):
        """Main portfolio update loop."""
        if not self.fetch_data():
            return

        state = self.load_state()
        cash = state['cash']
        holdings = state['holdings']
        trade_log = []
        
        today = self.data_cache['NIFTY'].index[-1].strftime('%Y-%m-%d')
        self.current_date = today

        print(f"\n  {'='*70}")
        print(f"  {STRATEGY_NAME} LIVE ENGINE")
        print(f"  Date: {today}")
        print(f"  {'='*70}")
        
        # ============================================================
        # 1. CHECK EXITS (ALWAYS — every run, not just rebalance)
        # ============================================================
        stocks_to_sell = []

        for t in list(holdings.keys()):
            if t not in self.data_cache:
                continue

            df = self.data_cache[t]
            price = df['Close'].iloc[-1]
            ma50 = df['Close'].rolling(50).mean().iloc[-1]
            peak = holdings[t].get('peak_price', holdings[t]['entry_price'])

            # Update peak
            if price > peak:
                holdings[t]['peak_price'] = price
                peak = price

            exit_reason = None

            # EXIT RULES
            if price < ma50:
                exit_reason = "Trend Break (< MA50)"
            elif price < peak * TRAILING_STOP_PCT:
                exit_reason = f"Trailing Stop (-{(1-TRAILING_STOP_PCT)*100:.0f}%)"

            if exit_reason:
                shares = holdings[t]['shares']
                proceeds = shares * price * 0.998  # Transaction cost
                pnl = proceeds - (shares * holdings[t]['entry_price'])
                pnl_pct = (price - holdings[t]['entry_price']) / holdings[t]['entry_price'] * 100

                cash += proceeds
                stocks_to_sell.append(t)

                trade_log.append({
                    'Ticker': t, 'Action': 'SELL', 'Date': today,
                    'Price': round(price, 2), 'PnL': round(pnl, 2),
                    'PnL%': round(pnl_pct, 2), 'Reason': exit_reason
                })
                print(f"    SELL {t}: {exit_reason} | P&L: {pnl_pct:+.1f}%")

        for t in stocks_to_sell:
            del holdings[t]

        # ============================================================
        # 2. REBALANCE CHECK — only scan for buys every 13 days
        # ============================================================
        is_reb_day = self.is_rebalance_day(state)

        if is_reb_day:
            print(f"\n  >> REBALANCE DAY (#{state.get('rebalance_count', 0) + 1})")
            
            # Calculate breadth gate
            breadth = self.calculate_breadth()
            print(f"     Market Breadth: {breadth:.0f}% above 50DMA", end="")

            if breadth < BREADTH_NARROW_THRESHOLD:
                print(f" -> NARROW MARKET (< {BREADTH_NARROW_THRESHOLD}%) -> SKIPPING NEW BUYS")
            else:
                print(f" -> Healthy (>= {BREADTH_NARROW_THRESHOLD}%) -> Scanning...")

                # ============================================================
                # 3. SCAN FOR NEW BUYS (only on rebalance day + healthy breadth)
                # ============================================================
                candidates = []
                for t in self.data_cache:
                    if t == 'NIFTY' or t in holdings:
                        continue

                    m = self.calculate_metrics(t)
                    if not m:
                        continue

                    # ENTRY RULES
                    if (m['liquidity'] > MIN_LIQUIDITY and 
                        m['price'] > m['ma50'] and 
                        m['rs_score'] > 0):
                        candidates.append({
                            'Ticker': t,
                            'Sector': self.sector_map.get(t, 'Unknown'),
                            'Price': m['price'],
                            'MA50': m['ma50'],
                            'RS_Score': m['rs_score'],
                        })

                candidates.sort(key=lambda x: -x['RS_Score'])

                # ============================================================
                # 4. BUY NEW POSITIONS (fill empty slots)
                # ============================================================
                free_slots = MAX_POSITIONS - len(holdings)
                if free_slots > 0 and candidates:
                    print(f"     {len(candidates)} candidates found, {free_slots} slots open")
                    
                    for cand in candidates[:free_slots]:
                        # Equal-weight sizing
                        total_equity = cash + sum([
                            holdings[h]['shares'] * self.data_cache[h]['Close'].iloc[-1]
                            for h in holdings if h in self.data_cache
                        ])
                        target_per_stock = total_equity / MAX_POSITIONS
                        invest_amount = min(target_per_stock, cash / max(free_slots, 1))

                        if invest_amount > 5000:
                            price = cand['Price']
                            shares = int(invest_amount / price)
                            cost = shares * price * 1.002  # Impact cost

                            if cash >= cost:
                                cash -= cost
                                holdings[cand['Ticker']] = {
                                    'entry_price': price,
                                    'entry_date': today,
                                    'shares': shares,
                                    'peak_price': price,
                                }
                                trade_log.append({
                                    'Ticker': cand['Ticker'], 'Action': 'BUY', 'Date': today,
                                    'Price': round(price, 2), 'PnL': 0, 'PnL%': 0,
                                    'Reason': f"Composite RS: {cand['RS_Score']:+.1f}"
                                })
                                free_slots -= 1
                                print(f"    BUY  {cand['Ticker']}: RS={cand['RS_Score']:+.1f} @ Rs.{price:.0f}")

            # Update rebalance tracking
            state['last_rebalance_date'] = today
            state['rebalance_count'] = state.get('rebalance_count', 0) + 1
        else:
            days_since = "?"
            try:
                nifty = self.data_cache['NIFTY']
                last_reb_dt = pd.Timestamp(state.get('last_rebalance_date'))
                trading_days = nifty.index[(nifty.index > last_reb_dt) & (nifty.index <= nifty.index[-1])]
                days_since = len(trading_days)
            except:
                pass
            print(f"\n  >> NOT REBALANCE DAY (day {days_since}/{REBALANCE_DAYS}). Exits only.")

        # ============================================================
        # 5. CALCULATE EQUITY & BUILD DISPLAY
        # ============================================================
        equity_val = cash
        portfolio_list = []

        for t, h in holdings.items():
            if t not in self.data_cache:
                continue
            curr_price = self.data_cache[t]['Close'].iloc[-1]
            equity_val += h['shares'] * curr_price

            ma50 = self.data_cache[t]['Close'].rolling(50).mean().iloc[-1]
            dist_ma50 = (curr_price - ma50) / ma50 * 100

            rs = self.composite_rs(t)

            portfolio_list.append({
                'Ticker': t,
                'Sector': self.sector_map.get(t, 'Unknown'),
                'Price': curr_price,
                'RS_Score': round(rs, 1) if rs else 0,
                'Entry': h['entry_price'],
                'PnL%': (curr_price - h['entry_price']) / h['entry_price'] * 100,
                'Dist_MA50': dist_ma50,
            })

        portfolio_list.sort(key=lambda x: -x['RS_Score'])

        # ============================================================
        # 6. SAVE STATE
        # ============================================================
        new_state = {
            'strategy': STRATEGY_NAME,
            'date': today,
            'cash': cash,
            'holdings': holdings,
            'equity': equity_val,
            'count': len(holdings),
            'portfolio': portfolio_list,
            'last_rebalance_date': state.get('last_rebalance_date'),
            'rebalance_count': state.get('rebalance_count', 0),
            'config': {
                'rs_weights': RS_WEIGHTS,
                'rebalance_days': REBALANCE_DAYS,
                'max_positions': MAX_POSITIONS,
                'breadth_threshold': BREADTH_NARROW_THRESHOLD,
                'trailing_stop': TRAILING_STOP_PCT,
            }
        }

        with open(SNAPSHOT_FILE, 'w') as f:
            json.dump(new_state, f, indent=4)

        # 7. APPEND LOGS
        if trade_log:
            df_log = pd.DataFrame(trade_log)
            hdr = not os.path.exists(TRADE_LOG_FILE)
            df_log.to_csv(TRADE_LOG_FILE, mode='a', header=hdr, index=False)

        eq_record = {
            'Date': today, 'Equity': round(equity_val, 2),
            'Cash': round(cash, 2), 'Holdings': len(holdings),
        }
        df_eq = pd.DataFrame([eq_record])
        hdr = not os.path.exists(EQUITY_CURVE_FILE)
        df_eq.to_csv(EQUITY_CURVE_FILE, mode='a', header=hdr, index=False)

        # Print summary
        ret_pct = (equity_val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        print(f"\n  {'='*70}")
        print(f"  PORTFOLIO SUMMARY")
        print(f"  {'='*70}")
        print(f"  Equity:    Rs.{equity_val:>12,.0f} ({ret_pct:+.1f}%)")
        print(f"  Cash:      Rs.{cash:>12,.0f}")
        print(f"  Holdings:  {len(holdings)}/{MAX_POSITIONS}")
        print(f"  Rebalance: #{state.get('rebalance_count', 0)} (every {REBALANCE_DAYS} days)")
        print(f"  {'='*70}")

        if portfolio_list:
            print(f"\n  {'Ticker':<16} {'Sector':<30} {'Price':>8} {'Entry':>8} {'P&L%':>8} {'RS':>8}")
            print(f"  {'-'*100}")
            for p in portfolio_list:
                print(f"  {p['Ticker']:<16} {p['Sector'][:28]:<30} {p['Price']:>8.0f} {p['Entry']:>8.0f} {p['PnL%']:>+7.1f}% {p['RS_Score']:>+7.1f}")

        return new_state


if __name__ == "__main__":
    print("=" * 70)
    print(f"{STRATEGY_NAME} Live Portfolio Engine")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    engine = OptCompV21Engine()
    engine.update_portfolio()
