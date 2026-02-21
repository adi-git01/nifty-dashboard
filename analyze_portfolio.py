"""
Portfolio Analysis Script
=========================
Parses user portfolio, applies Alpha Tenets, and generates actionable advice.

Alpha Tenets Rules:
1. Regime Check: Nifty MA50 vs MA200
2. Trend Check: 0-20 (Oversold), 20-40 (Weak), 40-60 (Neutral), 60-80 (Strong), 80-100 (Breakout)
3. Volume Check: Drop/Flat vs Spike
4. Sector Check: Cyclical vs Defensive

Verdict Logic:
- BEAR Market: Buy Trend 0-20 + Vol Drop. Sell Trend 80-100.
- BULL Market: Buy Trend 0-20 (Banking). Hold Trend 60-80. Trim Trend > 90.
- RECOVERY: Hold quality, trim weak.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import re
from datetime import datetime

# --- RAW PORTFOLIO STRING ---
RAW_PORTFOLIO = """
ALPL30IETF548120.6827.521,13,334.251,50,837.1237,502.87+33.09%+0.18%
AMBER EVENT86,015.506,594.3048,124.0053,640.40630.40+9.62%+3.03%
APOLLOHOSP17,000.007,144.507,000.007,144.50144.50+2.06%+0.31%
AXISBANK401,027.951,341.0041,118.0053,640.0012,522.00+30.45%+0.78%
COFORGE01,691.801,542.80N/AN/A0.00-8.81%-3.58%
CUMMINSIND84,175.584,339.2033,404.6534,713.601,308.95+3.92%-1.19%
DECNGOLDT1: 14 159126.89131.8021,952.1522,801.40849.25+3.87%-3.69%
FEDERALBNK109274.70286.4529,942.3031,223.051,280.75+4.28%-0.40%
HCLTECH01,644.801,593.60N/AN/A0.00-3.11%-1.02%
HINDALCO29713.14932.5020,681.1027,042.506,361.40+30.76%-0.32%
ICICIBANK101793.571,403.9080,150.201,41,793.9061,643.70+76.91%+0.53%
IIFL32454.33512.5014,538.6016,400.001,861.40+12.80%+1.49%
IRBINVIT-IV17055.2361.939,389.3810,528.101,138.72+12.13%+0.06%
KPRMILL EVENT29980.17952.1528,425.0027,612.35-812.65-2.86%-3.61%
LT93,642.404,061.5032,781.6036,553.503,771.90+11.51%-0.04%
LTFOODS410116.16401.2047,627.301,64,492.001,16,864.70+245.37%+1.19%
MAHABANK77264.6065.0049,871.2050,180.00308.80+0.62%+0.02%
MARINE93193.22187.6617,969.1717,452.38-516.79-2.88%-1.70%
MAXHEALTH EVENT101,009.501,030.1510,095.0010,301.50206.50+2.05%-1.02%
NORTHARC200252.50255.1550,500.0051,030.00530.00+1.05%-0.55%
NV20IETF677510.6715.0072,301.971,01,625.0029,323.03+40.56%+0.07%
SENORES56703.00825.7039,368.0046,239.206,871.20+17.45%+0.28%
SONACOMS0493.95509.35N/AN/A0.00+3.12%-2.93%
TEGA471,533.281,781.8072,064.3583,744.6011,680.25+16.21%-1.73%
UJJIVANSFB37666.3762.4324,955.1223,473.68-1,481.44-5.94%-1.03%
WELSPUNLIV233128.35139.9029,905.5532,596.702,691.15+9.00%-1.48%
"""

TICKER_MAP = {
    # Manual Override
    "ALPL30IETF": "ALPL30IETF.NS",
    "DECNGOLD": "GOLDBEES.NS",  # Verify if user meant simple Gold ETF
    "IRBINVIT-IV": "IRBINVIT.NS", 
    "NV20IETF": "NV20IETF.NS",
    "SENORES": "SENORES.NS", # Likely Senco Gold or similar, check manually or skip
}

def parse_portfolio(raw_text):
    """
    Robust parser for messy copy-pasted text.
    Handles lines like 'AMBER EVENT86,015...' by isolating the initial alpha string.
    """
    portfolio = []
    lines = raw_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Strategy: Tickers are typically uppercase letters at the start of the line.
        # Everything after the first digit is likely price/qty data.
        
        # 1. Extract leading letters/symbols (excluding digits)
        match = re.match(r'^([A-Z0-9\-&]+)', line)
        if match:
            raw_start = match.group(1)
            
            # 2. Aggressively strip trailing numbers if they look like part of the price string
            # Example: "AXISBANK401" -> "AXISBANK"
            # We assume valid tickers usually don't end in multiple digits unless it's an ETF like "NV20"
            
            # Split alpha vs numeric parts
            # e.g. "AXISBANK401" -> ["AXISBANK", "401"]
            parts = re.split(r'(\d+)', raw_start)
            ticker_candidate = parts[0]
            
            # Special handling for ETFs/stocks with legitimate numbers (e.g. 3MINDIA, A2ZINFRA)
            # But the user's list seems to simply mash price data.
            # "AXISBANK401" -> AXISBANK (Price 1027...)
            # "COFORGE01" -> COFORGE
            
            clean_ticker = ticker_candidate.replace("EVENT", "").strip()
            
            # Manual Mapping for edge cases
            if "NV20" in raw_start: clean_ticker = "NV20IETF"
            if "ALPL30" in raw_start: clean_ticker = "ALPL30IETF"
            if "IRBINVIT" in raw_start: clean_ticker = "IRBINVIT"
            if "DECNGOLD" in raw_start: clean_ticker = "GOLDBEES" # Override
            
            # 3. Construct Yahoo Symbol
            if clean_ticker in TICKER_MAP:
                symbol = TICKER_MAP[clean_ticker]
            else:
                symbol = f"{clean_ticker}.NS"
            
            # Extract P&L % (unchanged)
            pnl_match = re.findall(r'([-+]?\d+\.\d+)%', line)
            pnl_pct = float(pnl_match[-1]) if pnl_match else 0.0 # Take the LAST % which is usually total returns
            
            portfolio.append({
                'symbol': symbol,
                'raw_ticker': clean_ticker,
                'sub_name': clean_ticker,
                'pnl_pct': pnl_pct
            })
            
    return portfolio

def get_market_regime():
    """Fetch Nifty 50 and determine regime."""
    nifty = yf.Ticker("^NSEI").history(period="1y")
    if nifty.empty:
        return "Unknown", 0
    
    ma50 = nifty['Close'].rolling(50).mean().iloc[-1]
    ma200 = nifty['Close'].rolling(200).mean().iloc[-1]
    price = nifty['Close'].iloc[-1]
    
    if ma50 > ma200:
        if price > ma50:
            return "Strong Bull", price
        else:
            return "Mild Bull", price
    else:
        if price < ma50:
            return "Strong Bear", price
        else:
            return "Recovery", price

def analyze_stock(symbol, regime):
    """Calculate technicals and verdict."""
    try:
        data = yf.Ticker(symbol).history(period="1y")
        if data.empty:
            return {'symbol': symbol, 'error': 'No Data'}
        
        # Trend Score
        price = data['Close'].iloc[-1]
        ma50 = data['Close'].rolling(50).mean().iloc[-1]
        ma200 = data['Close'].rolling(200).mean().iloc[-1]
        high_52 = data['Close'].rolling(252).max().iloc[-1]
        low_52 = data['Close'].rolling(252).min().iloc[-1]
        
        score = 50
        if price > ma50: score += 15
        else: score -= 10
        if price > ma200: score += 15
        else: score -= 15
        if ma50 > ma200: score += 10
        else: score -= 5
        
        range_52 = high_52 - low_52
        if range_52 > 0:
            pos = (price - low_52) / range_52
            score += int((pos - 0.5) * 30)
        
        score = max(0, min(100, score))
        
        # Volume Delta
        vol_10 = data['Volume'].iloc[-10:].mean()
        vol_60 = data['Volume'].iloc[-60:].mean()
        vol_ratio = vol_10 / vol_60 if vol_60 > 0 else 1.0
        
        vol_state = "Flat"
        if vol_ratio > 1.5: vol_state = "Spike"
        elif vol_ratio < 0.7: vol_state = "Drop"
        
        # Verdict Logic
        verdict = "HOLD"
        reason = "Neutral"
        
        # BEAR REGIME LOGIC
        if "Bear" in regime:
            if score < 25 and vol_state in ["Drop", "Flat"]:
                verdict = "BUY Aggressive"
                reason = "Alpha Tenet: Bear Reversion"
            elif score > 80:
                verdict = "SELL/TRIM"
                reason = "Overextended in Bear Market"
            elif score < 40:
                verdict = "ACCUMULATE"
                reason = "Oversold"
        
        # BULL REGIME LOGIC
        elif "Bull" in regime:
            if score < 25:
                verdict = "BUY Dip"
                reason = "Dip in Bull Market"
            elif score > 90:
                verdict = "TRIM"
                reason = "Frothy (Trend > 90)"
            elif score > 60:
                verdict = "HOLD"
                reason = "Ride the Trend"
        
        return {
            'symbol': symbol,
            'price': price,
            'trend_score': int(score),
            'vol_state': vol_state,
            'verdict': verdict,
            'reason': reason
        }
        
    except Exception as e:
        return {'symbol': symbol, 'error': str(e)}

def main():
    print("Parsing portfolio...")
    portfolio = parse_portfolio(RAW_PORTFOLIO)
    print(f"Found {len(portfolio)} positions.")
    
    print("Checking Market Regime...")
    regime, nifty_price = get_market_regime()
    print(f"MARKET REGIME: {regime} (Nifty: {nifty_price:.0f})")
    print("-" * 60)
    
    results = []
    for item in portfolio:
        # Filter out ETFs/Gold for stock analysis if needed, but keeping for now
        if "ETF" in item['symbol'] or "GOLD" in item['symbol']:
            item.update({'trend_score': 'N/A', 'verdict': 'PASS (ETF)', 'reason': 'Passive'})
            results.append(item)
            continue
            
        analysis = analyze_stock(item['symbol'], regime)
        item.update(analysis)
        results.append(item)
        print(f"Analyzed {item['symbol']}...")
    
    # Sort by Verdict Priority
    priority_map = {"BUY Aggressive": 0, "BUY Dip": 1, "ACCUMULATE": 2, "SELL/TRIM": 3, "TRIM": 4, "HOLD": 5, "PASS (ETF)": 6}
    results.sort(key=lambda x: priority_map.get(x.get('verdict', 'HOLD'), 99))
    
    # Generate Report
    print("\n" + "="*80)
    print(f"PORTFOLIO HEALTH CHECK - {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Regime: {regime} | Strategy: {'Defensive/Reversion' if 'Bear' in regime else 'Trend Following'}")
    print("="*80)
    print(f"{'Ticker':<15} {'Trend':<6} {'Vol':<8} {'P&L%':<8} {'Verdict':<15} {'Reason'}")
    print("-" * 80)
    
    for r in results:
        if 'error' in r:
            print(f"{r['symbol']:<15} ERROR  ERROR    {r['pnl_pct']:<8.2f} MANUAL CHECK    {r['error']}")
            continue
            
        trend = str(r.get('trend_score', 'N/A'))
        vol = r.get('vol_state', '-')
        pnl = r['pnl_pct']
        verdict = r.get('verdict', 'UNKNOWN')
        reason = r.get('reason', '')
        
        print(f"{r['sub_name'] if 'sub_name' in r else r['raw_ticker']:<15} {trend:<6} {vol:<8} {pnl:<8.2f} {verdict:<15} {reason}")

if __name__ == "__main__":
    main()
