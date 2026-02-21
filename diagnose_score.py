
import yfinance as yf
import pandas as pd
# Removed pandas_ta import
from utils.scoring import calculate_trend_metrics
from utils.volume_analysis import calculate_volume_trend_score

def diagnose_stock(ticker):
    print(f"\nDiagnosing {ticker}...")
    
    # Fetch data
    df = yf.Ticker(ticker).history(period="2y") # Need > 200 days
    if df.empty:
        print("No data found.")
        return

    # Calculate indicators using pure Pandas
    # EMA 50
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    # EMA 200
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # Get latest row
    latest = df.iloc[-1]
    
    # Calculate Trend Score
    data_dict = {
        "currentPrice": latest['Close'],
        "price": latest['Close'],
        "fiftyDayAverage": latest['EMA_50'],
        "twoHundredDayAverage": latest['EMA_200'],
        "fiftyTwoWeekHigh": df['Close'].tail(252).max(),
        "fiftyTwoWeekLow": df['Close'].tail(252).min(),
    }
    
    trend_result = calculate_trend_metrics(data_dict)
    
    print("\nTREND SCORE CALCULATION:")
    print("-" * 30)
    print(f"Price: {data_dict['price']:.2f}")
    print(f"MA50: {data_dict['fiftyDayAverage']:.2f} ({'BELOW' if data_dict['price'] < data_dict['fiftyDayAverage'] else 'ABOVE'})")
    print(f"MA200: {data_dict['twoHundredDayAverage']:.2f} ({'BELOW' if data_dict['price'] < data_dict['twoHundredDayAverage'] else 'ABOVE'})")
    print(f"52W High: {data_dict['fiftyTwoWeekHigh']:.2f}")
    print(f"52W Low: {data_dict['fiftyTwoWeekLow']:.2f}")
    
    range_52 = data_dict['fiftyTwoWeekHigh'] - data_dict['fiftyTwoWeekLow']
    pos = (data_dict['price'] - data_dict['fiftyTwoWeekLow']) / range_52 if range_52 > 0 else 0
    print(f"Position in Range: {pos:.2f} (0=Low, 1=High)")
    
    print(f"Result: {trend_result}")
    
    # Calculate Volume Score
    vol_result = calculate_volume_trend_score(df['Close'], df['Volume'])
    
    print("\nVOLUME SCORE DIAGNOSIS:")
    print("-" * 30)
    print(f"Latest Date: {latest.name.date()}")
    print(f"Latest Volume: {df['Volume'].iloc[-1]:,}")
    print(f"20D Avg Volume: {df['Volume'].iloc[-20:].mean():,.0f}")
    print(f"Volume Ratio: {vol_result['volume_ratio']}x")
    print(f"OBV Trend: {vol_result['obv_trend']}")
    print(f"OBV Slope: {vol_result['obv_slope']}")
    print(f"Divergence: {vol_result['obv_divergence']} ({vol_result.get('divergence_type', 'NONE')})")
    print(f"CALCULATED SCORE: {vol_result['volume_score']}")
    
    print("\nWhy is it {vol_result['volume_score']}?")
    print("- Base Score: 5")
    if vol_result['volume_ratio'] > 1.5:
        print(f"- High Volume (>1.5x): +2")
    elif vol_result['volume_ratio'] > 1.2:
        print(f"- Elevated Volume (>1.2x): +1")
    elif vol_result['volume_ratio'] < 0.7:
        print(f"- Low Volume (<0.7x): -1")
        
    if vol_result['obv_trend'] == "ACCUMULATION":
        print("- OBV Accumulation: +2")
    elif vol_result['obv_trend'] == "DISTRIBUTION":
        print("- OBV Distribution: -2")
        
    if vol_result['obv_divergence']:
        dt = vol_result.get('divergence_type')
        if dt == "BEARISH": print("- Bearish Divergence: -2")
        if dt == "BULLISH": print("- Bullish Divergence: +1")

if __name__ == "__main__":
    diagnose_stock("BLUEDART.NS")
