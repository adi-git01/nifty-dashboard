import pandas as pd
import yfinance as yf
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import time
import os

print("Downloading total NSE listed universe...")
url = 'https://archives.nseindia.com/content/equities/EQUITY_L.csv'
try:
    urllib.request.urlretrieve(url, 'EQUITY_L.csv')
    df_nse = pd.read_csv('EQUITY_L.csv')
except Exception as e:
    print(f"Failed to download EQUITY_L: {e}")
    # Fallback to existing if available
    pass

df_nse = pd.read_csv('EQUITY_L.csv')
df_nse = df_nse[df_nse[' SERIES'] == 'EQ']
tickers = [t.strip() + ".NS" for t in df_nse['SYMBOL']]
print(f"Total NSE EQ tickers: {len(tickers)}")

print("Batch downloading 1 month price data to rank by liquidity...")
# Chunk downloading to avoid yF massive memory leaks
batch_size = 500
liquidity = []

for i in range(0, len(tickers), batch_size):
    batch = tickers[i:i+batch_size]
    try:
        data = yf.download(batch, period="1mo", progress=False, timeout=30)
        
        # Determine format (multi-index if len(batch) > 1)
        if hasattr(data.columns, 'levels'):
            for t in batch:
                if t in data['Close']:
                    close_px = data['Close'][t].mean()
                    vol = data['Volume'][t].mean()
                    if pd.notna(close_px) and pd.notna(vol):
                        liquidity.append({'Ticker': t, 'AvgValue': close_px * vol})
        else:
            # Single ticker case
            for t in batch:
                close_px = data['Close'].mean()
                vol = data['Volume'].mean()
                liquidity.append({'Ticker': t, 'AvgValue': close_px * vol})
                
    except Exception as e:
        print(f"Failed batch: {e}")
    time.sleep(2)

df_liq = pd.DataFrame(liquidity).sort_values(by='AvgValue', ascending=False)
top_1000 = df_liq.head(1000)['Ticker'].tolist()

print(f"Successfully ranked Top 1000 by liquidity. Threshold Value: {df_liq.iloc[999]['AvgValue'] / 1e7:.2f} Cr")

# Now fetch the Sub-Industry (yfinance industry tag) for the Top 1000
def get_industry(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            'Ticker': ticker,
            'Company_Name': info.get('shortName', ticker),
            'Sub_Industry': info.get('industry', 'Unknown'),
            'Macro_Sector': info.get('sector', 'Unknown'),
            'MarketCap': info.get('marketCap', 0)
        }
    except Exception:
        return {'Ticker': ticker, 'Company_Name': ticker, 'Sub_Industry': 'Unknown', 'Macro_Sector': 'Unknown', 'MarketCap': 0}

print("Fetching detailed industry classifications for Top 1000...")
results = []
with ThreadPoolExecutor(max_workers=30) as executor:
    for idx, res in enumerate(executor.map(get_industry, top_1000)):
        results.append(res)
        if (idx+1) % 100 == 0:
            print(f"  Processed {idx+1}/1000...")

# Build Final DataFrame
df_final = pd.DataFrame(results)
# Sort final by MarketCap to be safe
df_final = df_final.sort_values(by='MarketCap', ascending=False)

os.makedirs('data', exist_ok=True)
df_final.to_csv('data/nifty1000_list.csv', index=False)
print("Saved data/nifty1000_list.csv!")
print(f"Found {df_final['Sub_Industry'].nunique()} unique Sub-Industries.")

# Create the python access utility
util_code = '''
"""
Nifty 1000 Stock List with Sub-Industry Classifications
Loads from CSV file with proper NSE ticker format
"""
import pandas as pd
import os

def load_nifty1000_tickers():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "nifty1000_list.csv")
    try:
        df = pd.read_csv(csv_path)
        return df['Ticker'].tolist()
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []

def get_sub_industry_mapping():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "nifty1000_list.csv")
    try:
        df = pd.read_csv(csv_path)
        return dict(zip(df['Ticker'], df['Sub_Industry']))
    except Exception:
        return {}
        
TICKERS_1000 = load_nifty1000_tickers()
SUB_INDUSTRY_MAP = get_sub_industry_mapping()
'''

with open('utils/nifty1000_list.py', 'w') as f:
    f.write(util_code.strip())
print("Created utils/nifty1000_list.py access layer.")
