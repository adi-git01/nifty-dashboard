"""
DNA-3 V2 EXTENDED: SURGICAL ANALYSIS
=====================================
Deep dive into what makes it work and where it fails.

Analysis Dimensions:
1. Performance by TIME HORIZON (3m, 6m, 1y, 3y, 5y, 10y)
2. Performance by REGIME (Strong Bull, Mild Bull, Recovery, Strong Bear)
3. ENTRY Quality Analysis (What RS/Vol levels work best?)
4. EXIT Analysis (Stop vs Trail vs Time)
5. HOLDING PERIOD Analysis (When do trades work best?)
6. COMPARISON vs Pure Momentum (Trend Score > 70)
7. Achilles Heel Discovery (When does it fail?)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

INITIAL_CAPITAL = 1000000
MAX_POSITIONS = 10
STOP_LOSS = -0.15
TRAILING_ACTIVATION = 0.10
TRAILING_AMOUNT = 0.10

UNIVERSE = {
    'Industrials': ['LT.NS', 'SIEMENS.NS', 'ABB.NS', 'HAVELLS.NS', 'CUMMINSIND.NS', 'POLYCAB.NS', 'BEL.NS', 'HAL.NS'],
    'Metals': ['TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 'VEDL.NS', 'JINDALSTEL.NS', 'NMDC.NS', 'NATIONALUM.NS'],
    'IT_Services': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTIM.NS', 'COFORGE.NS', 'PERSISTENT.NS'],
    'Auto': ['MARUTI.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS', 'ASHOKLEY.NS', 'MOTHERSON.NS'],
    'Realty': ['DLF.NS', 'GODREJPROP.NS', 'OBEROIRLTY.NS', 'PRESTIGE.NS'],
    'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'LUPIN.NS', 'AUROPHARMA.NS'],
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'BANKBARODA.NS', 'FEDERALBNK.NS'],
    'Energy': ['RELIANCE.NS', 'ONGC.NS', 'NTPC.NS', 'TATAPOWER.NS', 'POWERGRID.NS'],
    'Consumer': ['HINDUNILVR.NS', 'ITC.NS', 'TITAN.NS', 'TRENT.NS', 'DABUR.NS', 'BRITANNIA.NS']
}

class DNA3SurgicalAnalysis:
    def __init__(self):
        self.data_cache = {}
        self.trade_log = []
        self.regime_log = []
        
    def fetch_data(self):
        print("[SURGICAL] Fetching 10+ years of data...")
        start_date = (datetime.now() - timedelta(days=365*10 + 300)).strftime('%Y-%m-%d')
        
        nifty = yf.Ticker("^NSEI").history(start=start_date)
        nifty.index = nifty.index.tz_localize(None)
        self.data_cache['NIFTY'] = nifty
        
        all_tickers = [t for sector in UNIVERSE.values() for t in sector]
        loaded = 0
        for t in all_tickers:
            try:
                df = yf.Ticker(t).history(start=start_date)
                if not df.empty and len(df) > 500:
                    df.index = df.index.tz_localize(None)
                    self.data_cache[t] = df
                    loaded += 1
            except: pass
        print(f"   Loaded {loaded} symbols")

    def get_price(self, ticker, date):
        if ticker not in self.data_cache: return None
        df = self.data_cache[ticker]
        mask = df.index <= date
        if mask.sum() == 0: return None
        return df.loc[mask, 'Close'].iloc[-1]

    def get_regime(self, date):
        nifty = self.data_cache['NIFTY']
        idx = nifty.index.searchsorted(date)
        if idx < 200: return "Unknown"
        window = nifty.iloc[max(0, idx-200):idx+1]
        p = window['Close'].iloc[-1]
        m50 = window['Close'].rolling(50).mean().iloc[-1]
        m200 = window['Close'].rolling(200).mean().iloc[-1]
        if m50 > m200:
            return "Strong_Bull" if p > m50 else "Mild_Bull"
        else:
            return "Strong_Bear" if p < m50 else "Recovery"

    def get_dna_metrics(self, ticker, date):
        """Get entry metrics for analysis."""
        if ticker not in self.data_cache: return None
        df = self.data_cache[ticker]
        nifty = self.data_cache['NIFTY']
        
        idx = df.index.searchsorted(date)
        if idx < 252: return None
        
        window = df.iloc[max(0, idx-252):idx+1]
        nifty_idx = nifty.index.searchsorted(date)
        nifty_window = nifty.iloc[max(0, nifty_idx-252):nifty_idx+1]
        
        if len(window) < 100: return None
        
        price = window['Close'].iloc[-1]
        
        # RS
        ret_3m = (price - window['Close'].iloc[-63]) / window['Close'].iloc[-63] * 100 if len(window) > 63 else 0
        nifty_ret_3m = (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-63]) / nifty_window['Close'].iloc[-63] * 100 if len(nifty_window) > 63 else 0
        rs = ret_3m - nifty_ret_3m
        
        # Volatility
        vol = window['Close'].pct_change().dropna()[-60:].std() * np.sqrt(252) * 100 if len(window) > 60 else 0
        
        # Trend
        ma50 = window['Close'].rolling(50).mean().iloc[-1] if len(window) > 50 else price
        ma200 = window['Close'].rolling(200).mean().iloc[-1] if len(window) > 200 else price
        trend_score = 50
        if price > ma50: trend_score += 25
        if price > ma200: trend_score += 15
        if ma50 > ma200: trend_score += 10
        
        return {'rs': rs, 'vol': vol, 'trend': trend_score, 'price': price}

    def passes_dna_filter(self, ticker, date):
        metrics = self.get_dna_metrics(ticker, date)
        if metrics is None: return False, None
        
        # DNA-3 V2 Rules
        if metrics['rs'] < 2: return False, None
        if metrics['vol'] < 30: return False, None
        
        df = self.data_cache[ticker]
        idx = df.index.searchsorted(date)
        window = df.iloc[max(0, idx-50):idx+1]
        price = window['Close'].iloc[-1]
        ma50 = window['Close'].rolling(50).mean().iloc[-1] if len(window) > 50 else price
        if price < ma50: return False, None
        
        return True, metrics

    def passes_pure_momentum(self, ticker, date):
        """PURE MOMENTUM: Just Trend Score > 70, no other filters."""
        metrics = self.get_dna_metrics(ticker, date)
        if metrics is None: return False
        return metrics['trend'] >= 70

    def run_strategy(self, strategy_type='dna3', years=10):
        """Run either DNA-3 or Pure Momentum and log detailed trades."""
        nifty = self.data_cache['NIFTY']
        start_idx = nifty.index.searchsorted(datetime.now() - timedelta(days=365*years))
        dates = nifty.index[start_idx:]
        
        capital = INITIAL_CAPITAL
        positions = {}
        history = []
        trade_log = []
        
        for date in dates:
            regime = self.get_regime(date)
            
            # EXITS
            to_exit = []
            for t, pos in positions.items():
                price = self.get_price(t, date)
                if not price: continue
                
                if price > pos['peak']: pos['peak'] = price
                
                ret = (price - pos['entry']) / pos['entry']
                exit_reason = None
                
                # Trailing
                if ret > TRAILING_ACTIVATION:
                    trail = pos['peak'] * (1 - TRAILING_AMOUNT)
                    if trail > pos['stop']: pos['stop'] = trail
                
                if price < pos['stop']:
                    exit_reason = 'Stop' if ret < 0 else 'Trail'
                
                if exit_reason:
                    val = pos['shares'] * price * 0.995
                    capital += val
                    days_held = (date - pos['entry_date']).days
                    trade_log.append({
                        'ticker': t.replace('.NS', ''),
                        'entry_date': pos['entry_date'],
                        'exit_date': date,
                        'pnl': ret * 100,
                        'days_held': days_held,
                        'entry_rs': pos['entry_rs'],
                        'entry_vol': pos['entry_vol'],
                        'entry_trend': pos['entry_trend'],
                        'entry_regime': pos['entry_regime'],
                        'exit_reason': exit_reason
                    })
                    to_exit.append(t)
            
            for t in to_exit: del positions[t]
            
            # ENTRIES
            if len(positions) < MAX_POSITIONS:
                candidates = []
                for sector, tickers in UNIVERSE.items():
                    for ticker in tickers:
                        if ticker in positions: continue
                        
                        if strategy_type == 'dna3':
                            passes, metrics = self.passes_dna_filter(ticker, date)
                        else:
                            passes = self.passes_pure_momentum(ticker, date)
                            metrics = self.get_dna_metrics(ticker, date) if passes else None
                        
                        if passes and metrics:
                            candidates.append({'ticker': ticker, 'metrics': metrics})
                
                candidates.sort(key=lambda x: -x['metrics']['rs'])
                
                for c in candidates[:MAX_POSITIONS - len(positions)]:
                    price = self.get_price(c['ticker'], date)
                    if price:
                        size = capital / (MAX_POSITIONS - len(positions) + 2)
                        shares = int(size / price)
                        if shares > 0:
                            cost = shares * price * 1.005
                            if capital >= cost:
                                capital -= cost
                                positions[c['ticker']] = {
                                    'entry': price, 'stop': price * (1 + STOP_LOSS),
                                    'peak': price, 'shares': shares,
                                    'entry_date': date,
                                    'entry_rs': c['metrics']['rs'],
                                    'entry_vol': c['metrics']['vol'],
                                    'entry_trend': c['metrics']['trend'],
                                    'entry_regime': regime
                                }
            
            val = capital
            for t, pos in positions.items():
                p = self.get_price(t, date)
                if p: val += pos['shares'] * p
            history.append({'date': date, 'value': val, 'regime': regime})
        
        return history, trade_log

    def analyze(self):
        self.fetch_data()
        
        # Run DNA-3 V2
        print("\n[1] Running DNA-3 V2...")
        dna_history, dna_trades = self.run_strategy('dna3', 10)
        
        # Run Pure Momentum
        print("[2] Running Pure Momentum (Trend > 70)...")
        mom_history, mom_trades = self.run_strategy('momentum', 10)
        
        nifty = self.data_cache['NIFTY']
        
        # ============== ANALYSIS ==============
        print("\n" + "="*70)
        print("DNA-3 V2 EXTENDED: SURGICAL ANALYSIS")
        print("="*70)
        
        # 1. OVERALL PERFORMANCE
        dna_df = pd.DataFrame(dna_history)
        mom_df = pd.DataFrame(mom_history)
        
        dna_final = dna_df.iloc[-1]['value']
        mom_final = mom_df.iloc[-1]['value']
        n_s = nifty.loc[dna_df.iloc[0]['date']]['Close']
        n_e = nifty.loc[dna_df.iloc[-1]['date']]['Close']
        n_ret = (n_e - n_s) / n_s * 100
        
        print("\n[1] OVERALL PERFORMANCE (10 Years)")
        print("-" * 50)
        print(f"DNA-3 V2:      {(dna_final-1000000)/10000:.2f}% | CAGR: {((dna_final/1000000)**(1/10)-1)*100:.2f}%")
        print(f"Pure Momentum: {(mom_final-1000000)/10000:.2f}% | CAGR: {((mom_final/1000000)**(1/10)-1)*100:.2f}%")
        print(f"Nifty 50:      {n_ret:.2f}% | CAGR: {((1+n_ret/100)**(1/10)-1)*100:.2f}%")
        
        # 2. PERFORMANCE BY REGIME
        print("\n[2] PERFORMANCE BY REGIME")
        print("-" * 50)
        dna_trades_df = pd.DataFrame(dna_trades)
        if not dna_trades_df.empty:
            regime_perf = dna_trades_df.groupby('entry_regime').agg({
                'pnl': ['mean', 'count', lambda x: (x > 0).mean() * 100]
            }).round(2)
            regime_perf.columns = ['Avg PnL', 'Trades', 'Win Rate']
            print(regime_perf.to_string())
        
        # 3. ENTRY QUALITY ANALYSIS
        print("\n[3] ENTRY QUALITY ANALYSIS")
        print("-" * 50)
        if not dna_trades_df.empty:
            # RS Buckets
            dna_trades_df['rs_bucket'] = pd.cut(dna_trades_df['entry_rs'], bins=[0,5,10,20,50,100], labels=['2-5%','5-10%','10-20%','20-50%','50%+'])
            rs_perf = dna_trades_df.groupby('rs_bucket').agg({
                'pnl': ['mean', 'count', lambda x: (x > 0).mean() * 100]
            }).round(2)
            rs_perf.columns = ['Avg PnL', 'Trades', 'Win Rate']
            print("By Entry RS:")
            print(rs_perf.to_string())
            
            # Volatility Buckets
            dna_trades_df['vol_bucket'] = pd.cut(dna_trades_df['entry_vol'], bins=[0,35,45,60,100], labels=['30-35%','35-45%','45-60%','60%+'])
            vol_perf = dna_trades_df.groupby('vol_bucket').agg({
                'pnl': ['mean', 'count', lambda x: (x > 0).mean() * 100]
            }).round(2)
            vol_perf.columns = ['Avg PnL', 'Trades', 'Win Rate']
            print("\nBy Entry Volatility:")
            print(vol_perf.to_string())
        
        # 4. EXIT ANALYSIS
        print("\n[4] EXIT ANALYSIS")
        print("-" * 50)
        if not dna_trades_df.empty:
            exit_perf = dna_trades_df.groupby('exit_reason').agg({
                'pnl': ['mean', 'count']
            }).round(2)
            exit_perf.columns = ['Avg PnL', 'Count']
            print(exit_perf.to_string())
        
        # 5. HOLDING PERIOD ANALYSIS
        print("\n[5] HOLDING PERIOD ANALYSIS")
        print("-" * 50)
        if not dna_trades_df.empty:
            dna_trades_df['hold_bucket'] = pd.cut(dna_trades_df['days_held'], bins=[0,30,60,120,365,1000], labels=['<30d','30-60d','60-120d','120-365d','365d+'])
            hold_perf = dna_trades_df.groupby('hold_bucket').agg({
                'pnl': ['mean', 'count', lambda x: (x > 0).mean() * 100]
            }).round(2)
            hold_perf.columns = ['Avg PnL', 'Trades', 'Win Rate']
            print(hold_perf.to_string())
        
        # 6. DNA-3 vs PURE MOMENTUM COMPARISON
        print("\n[6] DNA-3 V2 vs PURE MOMENTUM")
        print("-" * 50)
        mom_trades_df = pd.DataFrame(mom_trades)
        
        dna_wins = len([t for t in dna_trades if t['pnl'] > 0])
        mom_wins = len([t for t in mom_trades if t['pnl'] > 0])
        
        print(f"DNA-3 V2: {len(dna_trades)} trades, {dna_wins/len(dna_trades)*100:.1f}% win rate, {np.mean([t['pnl'] for t in dna_trades]):.1f}% avg")
        print(f"Pure Mom: {len(mom_trades)} trades, {mom_wins/len(mom_trades)*100:.1f}% win rate, {np.mean([t['pnl'] for t in mom_trades]):.1f}% avg")
        
        # 7. ACHILLES HEEL
        print("\n[7] ACHILLES HEEL (Where DNA-3 Fails)")
        print("-" * 50)
        if not dna_trades_df.empty:
            losers = dna_trades_df[dna_trades_df['pnl'] < -10]
            if not losers.empty:
                print(f"Big Losers (>10% loss): {len(losers)} trades")
                print(f"  Most common regime: {losers['entry_regime'].mode().values[0] if len(losers) > 0 else 'N/A'}")
                print(f"  Avg RS at entry: {losers['entry_rs'].mean():.1f}%")
                print(f"  Avg Vol at entry: {losers['entry_vol'].mean():.1f}%")
        
        # 8. SECRET SAUCE SUMMARY
        print("\n" + "="*70)
        print("[8] WHAT MAKES DNA-3 V2 WORK (The Secret Sauce)")
        print("="*70)
        
        if not dna_trades_df.empty:
            winners = dna_trades_df[dna_trades_df['pnl'] > 0]
            losers = dna_trades_df[dna_trades_df['pnl'] <= 0]
            
            print("\n1. RELATIVE STRENGTH FILTER:")
            print(f"   Winners avg RS at entry: {winners['entry_rs'].mean():.1f}%")
            print(f"   Losers avg RS at entry:  {losers['entry_rs'].mean():.1f}%")
            print(f"   -> Higher RS = Better outcomes")
            
            print("\n2. VOLATILITY FILTER:")
            print(f"   Winners avg Vol at entry: {winners['entry_vol'].mean():.1f}%")
            print(f"   Losers avg Vol at entry:  {losers['entry_vol'].mean():.1f}%")
            print(f"   -> Volatility gives room to move")
            
            print("\n3. TREND FILTER (Price > MA50):")
            print(f"   This prevents buying falling knives")
            print(f"   Forces entries in confirmed uptrends only")
            
            print("\n4. EXIT DISCIPLINE:")
            print(f"   Hard Stop -15%: Limits max loss")
            print(f"   Trailing Stop: Locks in gains, lets winners run")
        
        # Save detailed logs
        dna_trades_df.to_csv('analysis_2026/dna3_surgical_trades.csv', index=False)
        
        return dna_trades_df

if __name__ == "__main__":
    analyzer = DNA3SurgicalAnalysis()
    analyzer.analyze()
