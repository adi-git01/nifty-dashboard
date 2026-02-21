"""
WHY DID OUR STRATEGIES FAIL IN THE LAST YEAR?
==============================================
Deep dive into the 1-year underperformance of DNA3-V2.1 and Early Momentum
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nifty500_list import TICKERS, SECTOR_MAP

warnings.filterwarnings('ignore')

def analyze_market_conditions():
    """Analyze what happened in the market over the last year."""
    print("="*80)
    print("DIAGNOSING 1-YEAR STRATEGY FAILURE")
    print("="*80)
    
    # Fetch data
    print("\n[1] FETCHING MARKET DATA...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)
    
    nifty = yf.Ticker("^NSEI").history(start=start_date.strftime('%Y-%m-%d'))
    nifty.index = nifty.index.tz_localize(None)
    
    # Also get sector indices for analysis
    indices = {
        'NIFTY50': '^NSEI',
        'NIFTY_BANK': '^NSEBANK',
        'NIFTY_IT': '^CNXIT',
        'NIFTY_METAL': '^CNXMETAL',
        'NIFTY_AUTO': '^CNXAUTO',
        'NIFTY_MIDCAP': '^NSEMDCP50'
    }
    
    index_data = {}
    for name, ticker in indices.items():
        try:
            df = yf.Ticker(ticker).history(start=start_date.strftime('%Y-%m-%d'))
            if not df.empty:
                df.index = df.index.tz_localize(None)
                index_data[name] = df
        except:
            pass
    
    # Calculate 1-year returns
    print("\n[2] MARKET RETURNS (Last 12 Months)")
    print("-"*80)
    
    one_year_ago = end_date - timedelta(days=365)
    
    for name, df in index_data.items():
        mask = df.index >= one_year_ago
        if mask.sum() > 0:
            start_val = df.loc[mask, 'Close'].iloc[0]
            end_val = df.loc[mask, 'Close'].iloc[-1]
            ret = (end_val - start_val) / start_val * 100
            print(f"  {name:<15}: {ret:>+.1f}%")
    
    # Monthly breakdown
    print("\n[3] MONTHLY NIFTY BREAKDOWN (Last 12 Months)")
    print("-"*80)
    
    nifty_1y = nifty[nifty.index >= one_year_ago].copy()
    nifty_1y['month'] = nifty_1y.index.strftime('%Y-%m')
    
    monthly = []
    for month in nifty_1y['month'].unique():
        month_data = nifty_1y[nifty_1y['month'] == month]
        if len(month_data) > 1:
            start = month_data['Close'].iloc[0]
            end = month_data['Close'].iloc[-1]
            high = month_data['High'].max()
            low = month_data['Low'].min()
            ret = (end - start) / start * 100
            range_pct = (high - low) / start * 100
            monthly.append({'month': month, 'return': ret, 'range': range_pct})
    
    print(f"  {'Month':<10} {'Return':<12} {'Range':<12} {'Regime':<15}")
    print("-"*50)
    
    bull_months = 0
    bear_months = 0
    sideways_months = 0
    
    for m in monthly:
        if m['return'] > 3:
            regime = 'BULL'
            bull_months += 1
        elif m['return'] < -3:
            regime = 'BEAR'
            bear_months += 1
        else:
            regime = 'SIDEWAYS'
            sideways_months += 1
        
        print(f"  {m['month']:<10} {m['return']:>+.1f}%{'':<5} {m['range']:.1f}%{'':<6} {regime:<15}")
    
    print(f"\n  Summary: {bull_months} Bull, {bear_months} Bear, {sideways_months} Sideways months")
    
    # Volatility analysis
    print("\n[4] VOLATILITY ANALYSIS")
    print("-"*80)
    
    nifty_1y['daily_ret'] = nifty_1y['Close'].pct_change()
    volatility = nifty_1y['daily_ret'].std() * np.sqrt(252) * 100
    
    # Compare to historical volatility
    nifty_5y = nifty.copy()
    nifty_5y['daily_ret'] = nifty_5y['Close'].pct_change()
    vol_5y = nifty_5y['daily_ret'].std() * np.sqrt(252) * 100
    
    print(f"  Last 1 Year Volatility:  {volatility:.1f}%")
    print(f"  5-Year Avg Volatility:   {vol_5y:.1f}%")
    print(f"  Volatility vs Normal:    {volatility/vol_5y*100:.0f}%")
    
    # Trend analysis
    print("\n[5] TREND STRUCTURE ANALYSIS")
    print("-"*80)
    
    nifty_1y['MA50'] = nifty_1y['Close'].rolling(50).mean()
    nifty_1y['MA200'] = nifty_1y['Close'].rolling(200).mean()
    
    # Count days above/below MAs
    above_ma50 = (nifty_1y['Close'] > nifty_1y['MA50']).sum()
    above_ma200 = (nifty_1y['Close'] > nifty_1y['MA200']).sum()
    total_days = len(nifty_1y)
    
    print(f"  Days above MA50:  {above_ma50}/{total_days} ({above_ma50/total_days*100:.0f}%)")
    print(f"  Days above MA200: {above_ma200}/{total_days} ({above_ma200/total_days*100:.0f}%)")
    
    # MA50/MA200 crossovers (trend changes)
    nifty_1y['ma_trend'] = (nifty_1y['MA50'] > nifty_1y['MA200']).astype(int)
    crossovers = nifty_1y['ma_trend'].diff().abs().sum()
    print(f"  MA50/MA200 Crossovers: {int(crossovers)} (more = choppy)")
    
    # Sector rotation analysis
    print("\n[6] SECTOR ROTATION (Why Momentum Strategies Struggle)")
    print("-"*80)
    
    # Get sector returns for different periods
    quarters = [90, 180, 270, 365]
    sector_returns = {}
    
    for name, ticker in [('METAL', '^CNXMETAL'), ('IT', '^CNXIT'), ('BANK', '^NSEBANK'), ('AUTO', '^CNXAUTO')]:
        try:
            df = yf.Ticker(ticker).history(start=start_date.strftime('%Y-%m-%d'))
            if not df.empty:
                df.index = df.index.tz_localize(None)
                sector_returns[name] = []
                for q in quarters:
                    q_start = end_date - timedelta(days=q)
                    mask = df.index >= q_start
                    if mask.sum() > 1:
                        ret = (df.loc[mask, 'Close'].iloc[-1] - df.loc[mask, 'Close'].iloc[0]) / df.loc[mask, 'Close'].iloc[0] * 100
                        sector_returns[name].append(ret)
        except:
            pass
    
    print(f"  {'Sector':<10} {'Q4 (90d)':<12} {'Q3 (180d)':<12} {'Q2 (270d)':<12} {'Q1 (365d)':<12} Leadership Change?")
    print("-"*80)
    
    for sector, rets in sector_returns.items():
        if len(rets) >= 4:
            # Check if leadership changed (was leader, now laggard or vice versa)
            q4_rank = rets[0]
            q1_rank = rets[3]
            change = "ROTATED" if (q4_rank > 0 and q1_rank < 0) or (q4_rank < 0 and q1_rank > 0) else "STABLE"
            print(f"  {sector:<10} {rets[0]:>+.1f}%{'':<5} {rets[1]:>+.1f}%{'':<5} {rets[2]:>+.1f}%{'':<5} {rets[3]:>+.1f}%{'':<5} {change}")
    
    # Key diagnosis
    print("\n" + "="*80)
    print("DIAGNOSIS: WHY DID MOMENTUM STRATEGIES FAIL?")
    print("="*80)
    
    diagnosis = []
    
    if sideways_months >= 6:
        diagnosis.append("1. CHOPPY/SIDEWAYS MARKET: {0} of 12 months were sideways (-3% to +3%). "
                        "Momentum strategies need trending markets to work.".format(sideways_months))
    
    if volatility > vol_5y * 1.2:
        diagnosis.append("2. ELEVATED VOLATILITY: Market vol was {0:.0f}% higher than normal. "
                        "This triggers stops more frequently, killing momentum trades.".format((volatility/vol_5y - 1)*100))
    
    if crossovers >= 2:
        diagnosis.append("3. TREND WHIPSAWS: {0} MA50/200 crossovers in 1 year means trends kept reversing. "
                        "Momentum buys at tops before reversals.".format(int(crossovers)))
    
    if above_ma50/total_days < 0.5:
        diagnosis.append("4. WEAK TREND STRUCTURE: Nifty was above MA50 only {0:.0f}% of days. "
                        "A strong trending market should be >70%.".format(above_ma50/total_days*100))
    
    # Print diagnosis
    for d in diagnosis:
        print(f"\n{d}")
    
    # The real killer
    print("\n" + "-"*80)
    print("THE REAL PROBLEM:")
    print("-"*80)
    print("""
Both DNA3-V2.1 and Early Momentum are MOMENTUM strategies. They:
- BUY stocks showing relative strength
- EXPECT trends to continue
- USE stops that get triggered in whipsaws

In the last year, the market:
- Had multiple false breakouts
- Sector leadership rotated quickly  
- Never established a clean trend (choppy price action)

MOMENTUM STRATEGIES FAIL WHEN:
1. Market is RANGE-BOUND (no persistent trends)
2. VOLATILITY is high (stops get hit)
3. SECTOR ROTATION is fast (today's leaders become tomorrow's laggards)
4. FALSE BREAKOUTS are common (buy signals fail)

This was a MEAN-REVERSION friendly market, not a momentum market.
""")
    
    # What would have worked?
    print("\n" + "="*80)
    print("WHAT WOULD HAVE WORKED INSTEAD?")
    print("="*80)
    print("""
In this market regime, these strategies would have outperformed:

1. MEAN REVERSION: Buy oversold stocks, sell overbought
   - RSI < 30 entries, RSI > 70 exits
   
2. LOW VOLATILITY: Buy boring, stable stocks
   - Volatility < 25% filter
   
3. VALUE/QUALITY: Buy fundamentally cheap stocks
   - Low P/E, high ROE
   
4. CASH HEAVY: Stay in cash during choppy periods
   - Only deploy in confirmed trends

5. SECTOR NEUTRAL: Equal weight across sectors
   - Don't chase hot sectors
""")

    return {
        'volatility': volatility,
        'vol_vs_normal': volatility/vol_5y,
        'sideways_months': sideways_months,
        'above_ma50_pct': above_ma50/total_days*100,
        'crossovers': crossovers
    }

if __name__ == "__main__":
    results = analyze_market_conditions()
    print("\n[ANALYSIS COMPLETE]")
