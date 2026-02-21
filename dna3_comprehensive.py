"""
DNA-3 V2 EXTENDED: COMPREHENSIVE ANALYSIS
==========================================
Detailed metrics capture for every trade including:
- Entry metrics: Trend Score, PE, Volume Ratio, RSI, Volatility
- Trade outcomes: PnL, Holding Period
- Portfolio metrics: Drawdown, Max Win, Median/Avg/SD Returns

Max Portfolio Size: 20 stocks
Period: 10 Years
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

INITIAL_CAPITAL = 1000000
MAX_POSITIONS = 20  # Increased to 20 as requested
STOP_LOSS = -0.15
TRAILING_ACTIVATION = 0.10
TRAILING_AMOUNT = 0.10

UNIVERSE = {
    'Industrials': ['LT.NS', 'SIEMENS.NS', 'ABB.NS', 'HAVELLS.NS', 'CUMMINSIND.NS', 'POLYCAB.NS', 'BEL.NS', 'HAL.NS', 'THERMAX.NS'],
    'Metals': ['TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 'VEDL.NS', 'JINDALSTEL.NS', 'NMDC.NS', 'NATIONALUM.NS'],
    'IT_Services': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTIM.NS', 'COFORGE.NS', 'PERSISTENT.NS', 'LTTS.NS'],
    'Auto': ['MARUTI.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS', 'ASHOKLEY.NS', 'MOTHERSON.NS'],
    'Realty': ['DLF.NS', 'GODREJPROP.NS', 'OBEROIRLTY.NS', 'PRESTIGE.NS', 'BRIGADE.NS'],
    'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'LUPIN.NS', 'AUROPHARMA.NS', 'TORNTPHARM.NS'],
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'BANKBARODA.NS', 'FEDERALBNK.NS', 'IDFCFIRSTB.NS'],
    'Energy': ['RELIANCE.NS', 'ONGC.NS', 'NTPC.NS', 'TATAPOWER.NS', 'POWERGRID.NS', 'ADANIGREEN.NS'],
    'Consumer': ['HINDUNILVR.NS', 'ITC.NS', 'TITAN.NS', 'TRENT.NS', 'DABUR.NS', 'BRITANNIA.NS', 'NESTLEIND.NS', 'MARICO.NS']
}

class DNA3ComprehensiveAnalysis:
    def __init__(self, years=10):
        self.years = years
        self.data_cache = {}
        self.info_cache = {}
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.history = []
        self.trade_log = []
        self.peak_value = INITIAL_CAPITAL
        self.max_drawdown = 0
        
    def fetch_data(self):
        print(f"[DNA-3 V2 ANALYSIS] Fetching {self.years}+ years of data...")
        start_date = (datetime.now() - timedelta(days=365*self.years + 300)).strftime('%Y-%m-%d')
        
        nifty = yf.Ticker("^NSEI").history(start=start_date)
        nifty.index = nifty.index.tz_localize(None)
        self.data_cache['NIFTY'] = nifty
        
        all_tickers = [t for sector in UNIVERSE.values() for t in sector]
        loaded = 0
        for t in all_tickers:
            try:
                ticker_obj = yf.Ticker(t)
                df = ticker_obj.history(start=start_date)
                if not df.empty and len(df) > 500:
                    df.index = df.index.tz_localize(None)
                    self.data_cache[t] = df
                    # Get PE ratio if available
                    try:
                        info = ticker_obj.info
                        self.info_cache[t] = {
                            'pe': info.get('trailingPE', None),
                            'marketCap': info.get('marketCap', None)
                        }
                    except:
                        self.info_cache[t] = {'pe': None, 'marketCap': None}
                    loaded += 1
            except: pass
        print(f"   Loaded {loaded} symbols")

    def get_price(self, ticker, date):
        if ticker not in self.data_cache: return None
        df = self.data_cache[ticker]
        mask = df.index <= date
        if mask.sum() == 0: return None
        return df.loc[mask, 'Close'].iloc[-1]

    def calculate_rsi(self, ticker, date, period=14):
        """Calculate RSI at a specific date."""
        if ticker not in self.data_cache: return None
        df = self.data_cache[ticker]
        idx = df.index.searchsorted(date)
        if idx < period + 10: return None
        
        window = df.iloc[max(0, idx-period*2):idx+1]['Close']
        delta = window.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None

    def calculate_trend_score(self, ticker, date):
        """Calculate trend score at entry."""
        if ticker not in self.data_cache: return None
        df = self.data_cache[ticker]
        idx = df.index.searchsorted(date)
        if idx < 252: return 50
        
        window = df.iloc[max(0, idx-252):idx+1]
        price = window['Close'].iloc[-1]
        
        # MAs
        ma50 = window['Close'].rolling(50).mean().iloc[-1] if len(window) > 50 else price
        ma200 = window['Close'].rolling(200).mean().iloc[-1] if len(window) > 200 else price
        
        score = 50
        if price > ma50: score += 20
        if price > ma200: score += 15
        if ma50 > ma200: score += 10
        
        # 52W position
        high_52w = window['High'].max()
        low_52w = window['Low'].min()
        range_52 = high_52w - low_52w
        if range_52 > 0:
            pos = (price - low_52w) / range_52
            score += (pos - 0.5) * 10
        
        return min(100, max(0, score))

    def calculate_volume_ratio(self, ticker, date):
        """Calculate 20d/50d volume ratio."""
        if ticker not in self.data_cache: return None
        df = self.data_cache[ticker]
        idx = df.index.searchsorted(date)
        if idx < 60: return 1.0
        
        window = df.iloc[max(0, idx-60):idx+1]
        vol_20 = window['Volume'].iloc[-20:].mean()
        vol_50 = window['Volume'].iloc[-50:].mean()
        
        return vol_20 / vol_50 if vol_50 > 0 else 1.0

    def get_entry_metrics(self, ticker, date):
        """Get comprehensive entry metrics."""
        return {
            'trend_score': self.calculate_trend_score(ticker, date),
            'rsi': self.calculate_rsi(ticker, date),
            'volume_ratio': self.calculate_volume_ratio(ticker, date),
            'pe': self.info_cache.get(ticker, {}).get('pe', None)
        }

    def passes_dna_filter(self, ticker, date):
        if ticker not in self.data_cache: return False, {}
        df = self.data_cache[ticker]
        nifty = self.data_cache['NIFTY']
        
        idx = df.index.searchsorted(date)
        if idx < 252: return False, {}
        
        window = df.iloc[max(0, idx-252):idx+1]
        nifty_idx = nifty.index.searchsorted(date)
        nifty_window = nifty.iloc[max(0, nifty_idx-252):nifty_idx+1]
        
        if len(window) < 100: return False, {}
        
        price = window['Close'].iloc[-1]
        
        # RS > 2%
        ret_3m = (price - window['Close'].iloc[-63]) / window['Close'].iloc[-63] * 100 if len(window) > 63 else 0
        nifty_ret_3m = 0
        if len(nifty_window) > 63:
            nifty_ret_3m = (nifty_window['Close'].iloc[-1] - nifty_window['Close'].iloc[-63]) / nifty_window['Close'].iloc[-63] * 100
        rs_3m = ret_3m - nifty_ret_3m
        
        if rs_3m < 2.0: return False, {}
        
        # Volatility > 30%
        stock_returns = window['Close'].pct_change().dropna()[-60:]
        volatility = stock_returns.std() * np.sqrt(252) * 100 if len(stock_returns) > 10 else 0
        
        if volatility < 30: return False, {}
        
        # Price > MA50
        ma50 = window['Close'].rolling(50).mean().iloc[-1] if len(window) > 50 else price
        if price < ma50: return False, {}
        
        return True, {'rs_3m': rs_3m, 'volatility': volatility, 'price': price}

    def run(self):
        self.fetch_data()
        nifty = self.data_cache['NIFTY']
        start_idx = nifty.index.searchsorted(datetime.now() - timedelta(days=365*self.years))
        dates = nifty.index[start_idx:]
        
        print(f"[DNA-3 V2 ANALYSIS] Running {self.years}-Year Comprehensive Backtest...")
        print(f"   Max Portfolio Size: {MAX_POSITIONS}")
        
        for date in dates:
            # EXITS
            to_exit = []
            for t, pos in self.positions.items():
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
                    self.capital += val
                    days_held = (date - pos['entry_date']).days
                    
                    # Calculate trade drawdown
                    trade_peak = pos['peak']
                    trade_dd = (price - trade_peak) / trade_peak * 100 if trade_peak > 0 else 0
                    
                    self.trade_log.append({
                        'ticker': t.replace('.NS', ''),
                        'sector': pos['sector'],
                        'entry_date': pos['entry_date'],
                        'exit_date': date,
                        'entry_price': pos['entry'],
                        'exit_price': price,
                        'pnl_pct': ret * 100,
                        'days_held': days_held,
                        'entry_rs': pos['entry_rs'],
                        'entry_volatility': pos['entry_volatility'],
                        'entry_trend_score': pos['entry_trend_score'],
                        'entry_rsi': pos['entry_rsi'],
                        'entry_volume_ratio': pos['entry_volume_ratio'],
                        'entry_pe': pos['entry_pe'],
                        'trade_max_drawdown': trade_dd,
                        'trade_peak_gain': (trade_peak - pos['entry']) / pos['entry'] * 100,
                        'exit_reason': exit_reason
                    })
                    to_exit.append(t)
            
            for t in to_exit: del self.positions[t]
            
            # ENTRIES
            if len(self.positions) < MAX_POSITIONS:
                candidates = []
                for sector, tickers in UNIVERSE.items():
                    for ticker in tickers:
                        if ticker in self.positions: continue
                        passes, metrics = self.passes_dna_filter(ticker, date)
                        if passes:
                            entry_metrics = self.get_entry_metrics(ticker, date)
                            candidates.append({
                                'ticker': ticker, 'sector': sector,
                                'rs': metrics['rs_3m'], 'volatility': metrics['volatility'],
                                **entry_metrics
                            })
                
                candidates.sort(key=lambda x: -x['rs'])
                
                for c in candidates[:MAX_POSITIONS - len(self.positions)]:
                    price = self.get_price(c['ticker'], date)
                    if price:
                        size = self.capital / (MAX_POSITIONS - len(self.positions) + 2)
                        shares = int(size / price)
                        if shares > 0:
                            cost = shares * price * 1.005
                            if self.capital >= cost:
                                self.capital -= cost
                                self.positions[c['ticker']] = {
                                    'entry': price,
                                    'stop': price * (1 + STOP_LOSS),
                                    'peak': price,
                                    'shares': shares,
                                    'entry_date': date,
                                    'sector': c['sector'],
                                    'entry_rs': c['rs'],
                                    'entry_volatility': c['volatility'],
                                    'entry_trend_score': c['trend_score'],
                                    'entry_rsi': c['rsi'],
                                    'entry_volume_ratio': c['volume_ratio'],
                                    'entry_pe': c['pe']
                                }
            
            # Portfolio value and drawdown
            val = self.capital
            for t, pos in self.positions.items():
                p = self.get_price(t, date)
                if p: val += pos['shares'] * p
            
            if val > self.peak_value:
                self.peak_value = val
            current_dd = (val - self.peak_value) / self.peak_value * 100
            if current_dd < self.max_drawdown:
                self.max_drawdown = current_dd
            
            self.history.append({'date': date, 'value': val, 'drawdown': current_dd})
        
        self.generate_report()

    def generate_report(self):
        print("\n" + "="*80)
        print("DNA-3 V2 EXTENDED: COMPREHENSIVE ANALYSIS REPORT")
        print("="*80)
        
        trades_df = pd.DataFrame(self.trade_log)
        history_df = pd.DataFrame(self.history)
        
        if trades_df.empty:
            print("No trades to analyze.")
            return
        
        # Separate winners and losers
        winners = trades_df[trades_df['pnl_pct'] > 0]
        losers = trades_df[trades_df['pnl_pct'] <= 0]
        
        # === PORTFOLIO METRICS ===
        final_val = history_df.iloc[-1]['value']
        total_ret = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        nifty = self.data_cache['NIFTY']
        n_s = nifty.loc[history_df.iloc[0]['date']]['Close']
        n_e = nifty.loc[history_df.iloc[-1]['date']]['Close']
        n_ret = (n_e - n_s) / n_s * 100
        
        print("\n" + "-"*60)
        print("[1] PORTFOLIO PERFORMANCE")
        print("-"*60)
        print(f"{'Metric':<30} {'Value':<20}")
        print(f"{'Total Return':<30} {total_ret:.2f}%")
        print(f"{'Nifty Return':<30} {n_ret:.2f}%")
        print(f"{'Alpha':<30} {total_ret - n_ret:.2f}%")
        print(f"{'CAGR':<30} {((final_val/INITIAL_CAPITAL)**(1/self.years) - 1)*100:.2f}%")
        print(f"{'Max Portfolio Drawdown':<30} {self.max_drawdown:.2f}%")
        print(f"{'Max Portfolio Size':<30} {MAX_POSITIONS}")
        print(f"{'Total Trades':<30} {len(trades_df)}")
        print(f"{'Win Rate':<30} {len(winners)/len(trades_df)*100:.1f}%")
        
        # === RETURN STATISTICS ===
        print("\n" + "-"*60)
        print("[2] RETURN STATISTICS")
        print("-"*60)
        print(f"{'Metric':<25} {'All Trades':<15} {'Winners':<15} {'Losers':<15}")
        print(f"{'Average Return':<25} {trades_df['pnl_pct'].mean():.2f}%{'':<7} {winners['pnl_pct'].mean():.2f}%{'':<7} {losers['pnl_pct'].mean():.2f}%")
        print(f"{'Median Return':<25} {trades_df['pnl_pct'].median():.2f}%{'':<7} {winners['pnl_pct'].median():.2f}%{'':<7} {losers['pnl_pct'].median():.2f}%")
        print(f"{'Std Dev':<25} {trades_df['pnl_pct'].std():.2f}%{'':<7} {winners['pnl_pct'].std():.2f}%{'':<7} {losers['pnl_pct'].std():.2f}%")
        print(f"{'Max Win':<25} {trades_df['pnl_pct'].max():.2f}%")
        print(f"{'Max Loss':<25} {trades_df['pnl_pct'].min():.2f}%")
        
        # === HOLDING PERIOD ===
        print("\n" + "-"*60)
        print("[3] HOLDING PERIOD ANALYSIS")
        print("-"*60)
        print(f"{'Metric':<25} {'All Trades':<15} {'Winners':<15} {'Losers':<15}")
        print(f"{'Avg Holding (Days)':<25} {trades_df['days_held'].mean():.0f}{'':<12} {winners['days_held'].mean():.0f}{'':<12} {losers['days_held'].mean():.0f}")
        print(f"{'Median Holding (Days)':<25} {trades_df['days_held'].median():.0f}{'':<12} {winners['days_held'].median():.0f}{'':<12} {losers['days_held'].median():.0f}")
        print(f"{'Min Holding (Days)':<25} {trades_df['days_held'].min():.0f}")
        print(f"{'Max Holding (Days)':<25} {trades_df['days_held'].max():.0f}")
        
        # === ENTRY METRICS COMPARISON ===
        print("\n" + "-"*60)
        print("[4] ENTRY METRICS: WINNERS vs LOSERS")
        print("-"*60)
        print(f"{'Entry Metric':<25} {'Winners Avg':<15} {'Losers Avg':<15} {'Edge':<10}")
        
        # RS
        w_rs = winners['entry_rs'].mean()
        l_rs = losers['entry_rs'].mean()
        print(f"{'RS vs Nifty':<25} {w_rs:.1f}%{'':<9} {l_rs:.1f}%{'':<9} {w_rs-l_rs:+.1f}%")
        
        # Volatility
        w_vol = winners['entry_volatility'].mean()
        l_vol = losers['entry_volatility'].mean()
        print(f"{'Volatility':<25} {w_vol:.1f}%{'':<9} {l_vol:.1f}%{'':<9} {w_vol-l_vol:+.1f}%")
        
        # Trend Score
        w_trend = winners['entry_trend_score'].dropna().mean()
        l_trend = losers['entry_trend_score'].dropna().mean()
        print(f"{'Trend Score':<25} {w_trend:.1f}{'':<10} {l_trend:.1f}{'':<10} {w_trend-l_trend:+.1f}")
        
        # RSI
        w_rsi = winners['entry_rsi'].dropna().mean()
        l_rsi = losers['entry_rsi'].dropna().mean()
        print(f"{'RSI':<25} {w_rsi:.1f}{'':<10} {l_rsi:.1f}{'':<10} {w_rsi-l_rsi:+.1f}")
        
        # Volume Ratio
        w_vol_ratio = winners['entry_volume_ratio'].dropna().mean()
        l_vol_ratio = losers['entry_volume_ratio'].dropna().mean()
        print(f"{'Volume Ratio (20/50)':<25} {w_vol_ratio:.2f}{'':<10} {l_vol_ratio:.2f}{'':<10} {w_vol_ratio-l_vol_ratio:+.2f}")
        
        # PE
        w_pe = winners['entry_pe'].dropna().mean()
        l_pe = losers['entry_pe'].dropna().mean()
        if not pd.isna(w_pe) and not pd.isna(l_pe):
            print(f"{'PE Ratio':<25} {w_pe:.1f}{'':<10} {l_pe:.1f}{'':<10} {l_pe-w_pe:+.1f}")
        
        # === TRADE DRAWDOWN ===
        print("\n" + "-"*60)
        print("[5] INTRA-TRADE METRICS")
        print("-"*60)
        print(f"{'Metric':<25} {'Winners':<15} {'Losers':<15}")
        print(f"{'Avg Peak Gain':<25} {winners['trade_peak_gain'].mean():.1f}%{'':<9} {losers['trade_peak_gain'].mean():.1f}%")
        print(f"{'Avg Trade Drawdown':<25} {winners['trade_max_drawdown'].mean():.1f}%{'':<9} {losers['trade_max_drawdown'].mean():.1f}%")
        
        # === SECTOR ANALYSIS ===
        print("\n" + "-"*60)
        print("[6] SECTOR PERFORMANCE")
        print("-"*60)
        sector_perf = trades_df.groupby('sector').agg({
            'pnl_pct': ['mean', 'count', lambda x: (x > 0).mean() * 100]
        }).round(2)
        sector_perf.columns = ['Avg PnL', 'Trades', 'Win Rate']
        sector_perf = sector_perf.sort_values('Avg PnL', ascending=False)
        print(sector_perf.to_string())
        
        # === EXIT ANALYSIS ===
        print("\n" + "-"*60)
        print("[7] EXIT REASON ANALYSIS")
        print("-"*60)
        exit_perf = trades_df.groupby('exit_reason').agg({
            'pnl_pct': ['mean', 'count', 'std']
        }).round(2)
        exit_perf.columns = ['Avg PnL', 'Count', 'Std Dev']
        print(exit_perf.to_string())
        
        # === KEY INSIGHTS ===
        print("\n" + "="*80)
        print("[8] KEY INSIGHTS")
        print("="*80)
        
        # Best entry profile
        print("\nWINNER PROFILE AT ENTRY:")
        print(f"  - RS vs Nifty: {w_rs:.1f}% (higher is better)")
        print(f"  - Volatility: {w_vol:.1f}%")
        print(f"  - Trend Score: {w_trend:.1f}")
        print(f"  - RSI: {w_rsi:.1f}")
        print(f"  - Volume Ratio: {w_vol_ratio:.2f}")
        print(f"  - Avg Holding: {winners['days_held'].mean():.0f} days")
        
        print("\nLOSER PROFILE AT ENTRY:")
        print(f"  - RS vs Nifty: {l_rs:.1f}%")
        print(f"  - Volatility: {l_vol:.1f}%")
        print(f"  - Trend Score: {l_trend:.1f}")
        print(f"  - RSI: {l_rsi:.1f}")
        print(f"  - Volume Ratio: {l_vol_ratio:.2f}")
        print(f"  - Avg Holding: {losers['days_held'].mean():.0f} days")
        
        # Edge summary
        print("\nEDGE SUMMARY (What differentiates winners):")
        if w_rs > l_rs:
            print(f"  * Higher RS at entry ({w_rs-l_rs:+.1f}%)")
        if w_trend > l_trend:
            print(f"  * Higher Trend Score at entry ({w_trend-l_trend:+.1f})")
        if w_rsi > l_rsi:
            print(f"  * Higher RSI at entry ({w_rsi-l_rsi:+.1f})")
        if winners['days_held'].mean() > losers['days_held'].mean():
            print(f"  * Longer holding period ({winners['days_held'].mean() - losers['days_held'].mean():.0f} more days)")
        
        # Save detailed trade log
        trades_df.to_csv('analysis_2026/dna3_comprehensive_trades.csv', index=False)
        print("\n[SAVED] Detailed trade log: analysis_2026/dna3_comprehensive_trades.csv")

if __name__ == "__main__":
    analyzer = DNA3ComprehensiveAnalysis(years=10)
    analyzer.run()
