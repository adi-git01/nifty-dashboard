"""
OptComp-V22 LIVE PORTFOLIO ENGINE (Regime-Protected)
=====================================================
Production portfolio engine implementing the Composite RS strategy
with the V2.2 Regime Shield.

Strategy: OptComp-V22
  - RS Signal: Composite (10% 1W + 50% 1M + 40% 3M)
  - Rebalance: 13d (BULL/CAUTION) / 21d (BEAR) / HOLD (CRISIS)
  - Regime: 4-tier (BULL/CAUTION/BEAR/CRISIS) from Nifty MA200 + drawdown
  - Positions: 15 equal-weight
  - Breadth Gate: Skip new buys when market breadth < 30%

Entry Rules (only on rebalance days):
  1. Price > 50-day MA
  2. CompRS >= regime threshold (17pp BULL/CAUTION, 22pp BEAR/CRISIS)
  3. Liquidity >= regime threshold (1Cr/1.2Cr/1.5Cr/2Cr)
  4. Market breadth >= 30%
  5. Circuit breaker NOT active (Nifty -25% from 52W high)
  6. Vol quality >= 0.55 (CAUTION/BEAR only)
  7. Rank by CompRS, buy top candidates

Exit Rules (checked EVERY run, regime-adaptive):
  1. Price < 50-day MA -> SELL (Trend Break)
  2. Price < Peak * (1 - trail_stop[regime]) -> SELL (Regime Trail Stop)
     BULL: 15% | CAUTION: 12% | BEAR: 8% | CRISIS: 6%

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
from utils.nifty1000_list import TICKERS_1000 as TICKERS, SUB_INDUSTRY_MAP as SECTOR_MAP
from utils.regime_manager import (classify_regime, get_regime_params,
                                   is_circuit_breaker_active,
                                   calculate_volume_quality)

# ============================================================
# ALERT HELPERS — send Telegram + Email for trade events
# ============================================================
def _send_trade_alerts(trade_events, regime=''):
    """Send batched Telegram + Email alerts for BUY/SELL events."""
    if not trade_events:
        return
    try:
        from utils.telegram_notifier import send_telegram_message, is_telegram_configured
        if is_telegram_configured():
            lines = [f"<b>OptComp-V22 Trade Alert</b> [{regime}]\n"]
            for ev in trade_events:
                action = ev['Action']
                ticker = ev['Ticker'].replace('.NS', '')
                price = ev['Price']
                if action == 'SELL':
                    pnl = ev.get('PnL%', 0)
                    reason = ev.get('Reason', '')
                    icon = '🟢' if pnl >= 0 else '🔴'
                    lines.append(f"{icon} SELL <b>{ticker}</b>  ₹{price:,.0f}  P&L: {pnl:+.1f}%\n    {reason}")
                else:
                    rs = ev.get('Reason', '')
                    lines.append(f"🟢 BUY <b>{ticker}</b>  ₹{price:,.0f}\n    {rs}")
            send_telegram_message("\n".join(lines))
    except Exception as e:
        print(f"  [ALERT] Telegram failed: {e}")

    try:
        from utils.email_notifier import send_trend_change_alert, is_email_configured
        if is_email_configured():
            email_changes = []
            for ev in trade_events:
                ticker = ev['Ticker']
                action = ev['Action']
                if action == 'SELL':
                    signal = f"🚨 EXIT ({ev.get('Reason', '')})" 
                else:
                    signal = '🟢 NEW BUY'
                email_changes.append({
                    'ticker': ticker,
                    'entry_trend_signal': 'ACTIVE' if action == 'SELL' else 'CASH',
                    'current_signal': signal,
                    'return_pct': ev.get('PnL%', 0),
                    'days_tracked': 0,
                })
            send_trend_change_alert(email_changes)
    except Exception as e:
        print(f"  [ALERT] Email failed: {e}")

# ============================================================
# STRATEGY CONFIGURATION — OptComp-V22 (Regime-Protected)
# ============================================================
STRATEGY_NAME = "OptComp-V22"
MAX_POSITIONS = 15
INITIAL_CAPITAL = 1000000  # 10 Lakhs
START_DATE = "2026-02-23"  # Fresh start
DEFAULT_REBALANCE_DAYS = 13  # Overridden by regime

# Composite RS Weights (10% 1W + 50% 1M + 40% 3M)
RS_WEIGHTS = [
    (5,   0.10),   # 1 Week
    (21,  0.50),   # 1 Month
    (63,  0.40),   # 3 Months (Quarter)
]

# Breadth Gate Threshold
BREADTH_NARROW_THRESHOLD = 30  # % of stocks above 50DMA — below this = skip buys

# V2.2 Regime Thresholds (CompRS is in *100 scale, so 0.17 -> 17.0)
# These are applied dynamically per regime via regime_manager.get_regime_params()
# Static fallbacks (only used if regime computation fails):
FALLBACK_TRAILING_STOP_PCT = 0.85     # 15% trail
FALLBACK_MIN_LIQUIDITY     = 10_000_000  # Rs 1 Cr
FALLBACK_MIN_COMP_RS       = 17.0     # 17pp CompRS

DATA_DIR = "data"
SNAPSHOT_FILE = f"{DATA_DIR}/dna3_portfolio_snapshot.json"
TRADE_LOG_FILE = f"{DATA_DIR}/dna3_trade_log.csv"
EQUITY_CURVE_FILE = f"{DATA_DIR}/dna3_equity_curve.csv"


class OptCompV21Engine:
    def __init__(self):
        self.tickers = TICKERS  # Full Nifty 1000 universe
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
                    saved_strategy = state.get('strategy', '')
                    # Backward-compatible: accept V21 snapshots and migrate
                    compatible = (saved_strategy == STRATEGY_NAME or
                                  saved_strategy in ['OptComp-V21', 'OptComp-V22'])
                    if compatible:
                        state['strategy'] = STRATEGY_NAME  # migrate forward
                        return state
                    else:
                        print(f"  Old strategy detected ({saved_strategy}). Resetting for {STRATEGY_NAME}.")
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
            'recently_exited': {},         # {ticker: exit_date} — stop-loss cooldown
            'last_dd_alert_date': None,    # Dedup drawdown Telegram alerts (1/day)
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
                group_by='ticker', threads=False, progress=False, auto_adjust=True
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
            # ✅ FIX: max_workers=3 (was 20) — 20 parallel yfinance calls = Yahoo 429 rate limit
            print(f"  [FALLBACK] Fetching {len(missing)} missing stocks with 3 workers...", flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
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
        """Legacy wrapper; uses default 13d frequency."""
        return self._is_rebalance_day_v22(state, DEFAULT_REBALANCE_DAYS)

    def _is_rebalance_day_v22(self, state, rebalance_freq):
        """Check if today is a rebalance day using regime-dynamic frequency."""
        last_reb = state.get('last_rebalance_date')
        
        if last_reb is None:
            return True  # First run, force rebalance

        # CRISIS = HOLD (freq=999), never rebalance
        if rebalance_freq >= 999:
            return False

        nifty = self.data_cache['NIFTY']
        try:
            last_reb_dt = pd.Timestamp(last_reb)
            today_dt = nifty.index[-1]
            
            trading_days = nifty.index[(nifty.index > last_reb_dt) & (nifty.index <= today_dt)]
            days_since = len(trading_days)
            
            return days_since >= rebalance_freq
        except:
            return True  # If error, force rebalance

    def update_portfolio(self):
        """Main portfolio update loop with V2.2 Regime Shield."""
        if not self.fetch_data():
            return

        state = self.load_state()
        cash = state['cash']
        holdings = state['holdings']
        recently_exited = state.get('recently_exited', {})  # stop-loss cooldown tracker
        trade_log = []
        
        today = self.data_cache['NIFTY'].index[-1].strftime('%Y-%m-%d')
        self.current_date = today

        # ============================================================
        # V2.2 REGIME CLASSIFICATION
        # ============================================================
        nifty_df = self.data_cache['NIFTY']
        try:
            n_close  = float(nifty_df['Close'].iloc[-1])
            n_ma200  = float(nifty_df['Close'].rolling(200).mean().iloc[-1])
            n_52w_hi = float(nifty_df['High'].rolling(252).max().iloc[-1])
            regime   = classify_regime(n_close, n_ma200, n_52w_hi)
        except Exception:
            regime = 'CAUTION'  # safe fallback

        params      = get_regime_params(regime)
        trail_pct   = params['trail_stop']      # 0.15 / 0.12 / 0.08 / 0.06
        trail_keep  = 1.0 - trail_pct            # 0.85 / 0.88 / 0.92 / 0.94
        min_liq_cr  = params['min_liquidity']    # 1 / 1.2 / 1.5 / 2 (in Cr)
        min_liq_raw = min_liq_cr * 1e7           # convert Cr -> raw Rs
        min_comp_rs = params['min_comp_rs'] * 100  # 0.17 -> 17.0 (match *100 scaling)
        rebal_days  = params['rebalance_freq']   # 13 / 13 / 21 / 999
        can_enter   = params['new_entries']       # True / True / True / False

        # Circuit Breaker
        try:
            cb_active = is_circuit_breaker_active(n_close, n_52w_hi)
        except Exception:
            cb_active = False

        self.regime = regime  # store for snapshot

        regime_icon = {'BULL': '[BULL]', 'CAUTION': '[CAUTION]',
                       'BEAR': '[BEAR]', 'CRISIS': '[CRISIS]'}.get(regime, '[?]')

        print(f"\n  {'='*70}")
        print(f"  {STRATEGY_NAME} LIVE ENGINE  {regime_icon}")
        print(f"  Date: {today}")
        print(f"  Regime: {regime} | Trail: {trail_pct*100:.0f}% | Liq: Rs{min_liq_cr:.1f}Cr | CompRS: {min_comp_rs:.0f}pp")
        if cb_active:
            print(f"  *** CIRCUIT BREAKER ACTIVE *** (Nifty >25% off 52W high)")
        print(f"  {'='*70}")

        # 🔔 REGIME CHANGE ALERT
        prev_regime = state.get('regime', regime)
        if prev_regime != regime:
            print(f"  ⚠️ REGIME SHIFT: {prev_regime} → {regime}")
            prev_params = get_regime_params(prev_regime)
            try:
                from utils.telegram_notifier import send_telegram_message, is_telegram_configured
                if is_telegram_configured():
                    msg = (f"⚠️ <b>REGIME SHIFT</b>: {prev_regime} → {regime}\n\n"
                           f"Trailing Stop: {prev_params['trail_stop']*100:.0f}% → {trail_pct*100:.0f}%\n"
                           f"Rebalance: {prev_params['rebalance_freq']}d → {rebal_days}d\n"
                           f"Min CompRS: {prev_params['min_comp_rs']*100:.0f}pp → {min_comp_rs:.0f}pp\n"
                           f"New Entries: {'✅' if can_enter else '❌ Suspended'}")
                    send_telegram_message(msg)
            except Exception as e:
                print(f"  [ALERT] Regime shift Telegram failed: {e}")
            try:
                from utils.email_notifier import send_system_alert, is_email_configured
                if is_email_configured():
                    send_system_alert(
                        f"⚠️ Regime Shift: {prev_regime} → {regime}",
                        f"Trail: {prev_params['trail_stop']*100:.0f}% → {trail_pct*100:.0f}% | "
                        f"Rebal: {prev_params['rebalance_freq']}d → {rebal_days}d | "
                        f"Entries: {'Open' if can_enter else 'SUSPENDED'}"
                    )
            except Exception as e:
                print(f"  [ALERT] Regime shift Email failed: {e}")

        # 🔔 CIRCUIT BREAKER CHANGE ALERT
        prev_cb = state.get('circuit_breaker', False)
        if cb_active != prev_cb:
            cb_msg = ("🚨 <b>CIRCUIT BREAKER ACTIVATED</b>\n\n"
                      f"Nifty drawdown exceeds -25% from 52W high.\n"
                      f"ALL new entries SUSPENDED until recovery.") if cb_active else (
                      "✅ <b>CIRCUIT BREAKER DEACTIVATED</b>\n\n"
                      f"Nifty recovered above -25% threshold.\n"
                      f"New entries RESUMED under {regime} regime rules.")
            print(f"  {'🚨' if cb_active else '✅'} Circuit Breaker {'ON' if cb_active else 'OFF'}")
            try:
                from utils.telegram_notifier import send_telegram_message, is_telegram_configured
                if is_telegram_configured():
                    send_telegram_message(cb_msg)
            except Exception as e:
                print(f"  [ALERT] Circuit breaker TG failed: {e}")
            try:
                from utils.email_notifier import send_system_alert, is_email_configured
                if is_email_configured():
                    send_system_alert(
                        f"🚨 Circuit Breaker {'ON' if cb_active else 'OFF'}",
                        cb_msg.replace('<b>', '').replace('</b>', '')
                    )
            except Exception as e:
                print(f"  [ALERT] Circuit breaker Email failed: {e}")

        # Drawdown alert is sent AFTER equity is recalculated (see section 5 below)
        
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

            # EXIT RULES (V2.2: regime-adaptive trailing stop)
            if price < ma50:
                exit_reason = "Trend Break (< MA50)"
            elif price < peak * trail_keep:
                exit_reason = f"Trailing Stop (-{trail_pct*100:.0f}% [{regime}])"

            if exit_reason:
                shares = holdings[t]['shares']
                proceeds = shares * price * 0.998  # Transaction cost
                pnl = proceeds - (shares * holdings[t]['entry_price'])
                pnl_pct = (price - holdings[t]['entry_price']) / holdings[t]['entry_price'] * 100

                cash += proceeds
                stocks_to_sell.append(t)

                # Track trailing-stop exits for re-entry cooldown (1 rebalance period)
                if 'Trailing Stop' in exit_reason:
                    recently_exited[t] = today

                trade_log.append({
                    'Ticker': t, 'Action': 'SELL', 'Date': today,
                    'Price': round(price, 2), 'PnL': round(pnl, 2),
                    'PnL%': round(pnl_pct, 2), 'Reason': exit_reason
                })
                print(f"    SELL {t}: {exit_reason} | P&L: {pnl_pct:+.1f}%")

        # 🔔 SEND EXIT ALERTS (Telegram + Email)
        exit_events = [e for e in trade_log if e['Action'] == 'SELL']
        _send_trade_alerts(exit_events, regime=regime)

        for t in stocks_to_sell:
            del holdings[t]

        # ============================================================
        # 2. REBALANCE CHECK (V2.2: regime-adaptive frequency)
        # ============================================================
        # Override rebalance frequency from regime
        is_reb_day = self._is_rebalance_day_v22(state, rebal_days)

        if is_reb_day:
            print(f"\n  >> REBALANCE DAY (#{state.get('rebalance_count', 0) + 1}) [{regime}: every {rebal_days}d]")

            # V2.2: Circuit breaker blocks ALL new entries
            if cb_active:
                print("     CIRCUIT BREAKER ACTIVE -> ALL NEW ENTRIES BLOCKED")
            elif not can_enter:
                print(f"     {regime} REGIME -> NEW ENTRIES SUSPENDED")
            else:
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
                # Expire old cooldown entries (older than 1 rebalance period)
                nifty_idx = self.data_cache['NIFTY'].index
                today_ts = pd.Timestamp(today)
                recently_exited = {
                    t: dt for t, dt in recently_exited.items()
                    if len(nifty_idx[(nifty_idx > pd.Timestamp(dt)) & (nifty_idx <= today_ts)]) < rebal_days
                }

                candidates = []
                for t in self.data_cache:
                    if t == 'NIFTY' or t in holdings:
                        continue

                    # Skip stocks in stop-loss cooldown (1 rebalance period after trailing stop)
                    if t in recently_exited:
                        continue

                    m = self.calculate_metrics(t)
                    if not m:
                        continue

                    # ENTRY RULES (V2.2: regime-adaptive thresholds)
                    if (m['liquidity'] > min_liq_raw and 
                        m['price'] > m['ma50'] and 
                        m['rs_score'] >= min_comp_rs):

                        # V2.2 Step 4: Vol Quality gate (CAUTION/BEAR only)
                        if regime in ['CAUTION', 'BEAR']:
                            df_t = self.data_cache[t]
                            if len(df_t) >= 5:
                                vq = calculate_volume_quality(
                                    df_t['Close'].iloc[-5:],
                                    df_t['Open'].iloc[-5:],
                                    df_t['Volume'].iloc[-5:]
                                )
                                if vq < 0.55:
                                    continue  # filtered: panic selling volume

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

                # 🔔 SEND BUY ALERTS (Telegram + Email)
                buy_events = [e for e in trade_log if e['Action'] == 'BUY']
                _send_trade_alerts(buy_events, regime=regime)

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
            print(f"\n  >> NOT REBALANCE DAY (day {days_since}/{rebal_days}). Exits only.")

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

            peak = h.get('peak_price', h['entry_price'])
            trail_stop_price = peak * trail_keep
            dist_trail = (curr_price - trail_stop_price) / trail_stop_price * 100

            # Danger flag: within 5% of either exit trigger
            danger = ''
            if dist_ma50 < 5:
                danger = f'⚠️ {dist_ma50:.1f}% to MA50'
            if dist_trail < 5:
                trail_warn = f'⚠️ {dist_trail:.1f}% to trail'
                danger = trail_warn if not danger else f'{danger} | {trail_warn}'

            rs = self.composite_rs(t)

            portfolio_list.append({
                'Ticker': t,
                'Sector': self.sector_map.get(t, 'Unknown'),
                'Price': curr_price,
                'RS_Score': round(rs, 1) if rs else 0,
                'Entry': h['entry_price'],
                'PnL%': (curr_price - h['entry_price']) / h['entry_price'] * 100,
                'Dist_MA50': dist_ma50,
                'MA50': round(float(ma50), 2),
                'Trail_Stop': round(float(trail_stop_price), 2),
                'Danger': danger,
            })

        portfolio_list.sort(key=lambda x: -x['RS_Score'])

        # 🔔 PORTFOLIO DRAWDOWN ALERT — checked here using CURRENT equity (not stale snapshot)
        # Deduped to once per calendar day so it doesn't spam Telegram every run.
        try:
            if os.path.exists(EQUITY_CURVE_FILE):
                eq_df = pd.read_csv(EQUITY_CURVE_FILE)
                if len(eq_df) > 1:
                    peak_equity = eq_df['Equity'].max()
                    dd_pct = (equity_val - peak_equity) / peak_equity * 100
                    last_dd_date = state.get('last_dd_alert_date', '')
                    if dd_pct <= -5 and last_dd_date != today:
                        dd_msg = (f"📉 <b>PORTFOLIO DRAWDOWN WARNING</b>\n\n"
                                  f"Portfolio: Rs {equity_val:,.0f}\n"
                                  f"Peak: Rs {peak_equity:,.0f}\n"
                                  f"Drawdown: {dd_pct:.1f}%\n"
                                  f"Regime: {regime}")
                        print(f"  📉 Portfolio Drawdown: {dd_pct:.1f}%")
                        state['last_dd_alert_date'] = today  # persist via new_state below
                        try:
                            from utils.telegram_notifier import send_telegram_message, is_telegram_configured
                            if is_telegram_configured():
                                send_telegram_message(dd_msg)
                        except Exception:
                            pass
        except Exception as e:
            print(f"  [ALERT] Drawdown check failed: {e}")

        # ============================================================
        # 6. SAVE STATE
        # ============================================================
        new_state = {
            'strategy': STRATEGY_NAME,
            'date': today,
            'cash': cash,
            'holdings': holdings,
            'equity': equity_val,
            'cash_pct': round(cash / equity_val * 100, 1) if equity_val > 0 else 0,
            'count': len(holdings),
            'portfolio': portfolio_list,
            'last_rebalance_date': state.get('last_rebalance_date'),
            'rebalance_count': state.get('rebalance_count', 0),
            'regime': regime,
            'circuit_breaker': cb_active,
            'recently_exited': recently_exited,
            'last_dd_alert_date': state.get('last_dd_alert_date', ''),
            'config': {
                'rs_weights': RS_WEIGHTS,
                'rebalance_days': rebal_days,
                'max_positions': MAX_POSITIONS,
                'breadth_threshold': BREADTH_NARROW_THRESHOLD,
                'trailing_stop': trail_pct,
                'min_comp_rs': min_comp_rs,
                'min_liquidity_cr': min_liq_cr,
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
            'Regime': regime,
        }
        # Dedup: only write one row per day
        if os.path.exists(EQUITY_CURVE_FILE):
            existing_eq = pd.read_csv(EQUITY_CURVE_FILE)
            existing_eq = existing_eq[existing_eq['Date'] != today]
            existing_eq = pd.concat([existing_eq, pd.DataFrame([eq_record])], ignore_index=True)
            existing_eq.to_csv(EQUITY_CURVE_FILE, index=False)
        else:
            pd.DataFrame([eq_record]).to_csv(EQUITY_CURVE_FILE, index=False)

        # Print summary
        ret_pct = (equity_val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        print(f"\n  {'='*70}")
        print(f"  PORTFOLIO SUMMARY")
        print(f"  {'='*70}")
        print(f"  Equity:    Rs.{equity_val:>12,.0f} ({ret_pct:+.1f}%)")
        print(f"  Cash:      Rs.{cash:>12,.0f}")
        print(f"  Holdings:  {len(holdings)}/{MAX_POSITIONS}")
        print(f"  Regime:    {regime} | Trail: {trail_pct*100:.0f}% | Rebal: {rebal_days}d")
        print(f"  Rebalance: #{state.get('rebalance_count', 0)}")
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
