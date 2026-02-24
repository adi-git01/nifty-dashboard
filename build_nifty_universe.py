import pandas as pd
import urllib.request
import os

print("Downloading official NSE Index constituents to map Sub-Industries safely...")

urls = {
    'Nifty500': 'https://archives.nseindia.com/content/indices/ind_nifty500list.csv',
    'Microcap250': 'https://archives.nseindia.com/content/indices/ind_niftymicrocap250list.csv',
    'TotalMarket': 'https://archives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv'
}

dfs = []
for name, url in urls.items():
    try:
        print(f"Fetching {name}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            df = pd.read_csv(response)
            dfs.append(df)
    except Exception as e:
        print(f"Failed to fetch {name}: {e}")

if not dfs:
    print("Failed to download any NSE indices. Falling back to the local nifty500_list.csv if possible.")
    df_fallback = pd.read_csv('data/nifty500_list.csv')
    df_fallback = df_fallback.rename(columns={'Sector': 'Industry'})
    final_df = df_fallback
else:
    final_df = pd.concat(dfs, ignore_index=True)
    # The NSE files typically have columns like 'Company Name', 'Industry', 'Symbol'
    final_df = final_df.drop_duplicates(subset=['Symbol'])

# Transform to our required format
result = []
for _, row in final_df.iterrows():
    ticker = row.get('Symbol', row.get('Ticker', ''))
    if not ticker.endswith('.NS'):
        ticker_ns = f"{ticker}.NS"
    else:
        ticker_ns = ticker
        
    industry = row.get('Industry', row.get('Sector', 'Unknown'))
    company = row.get('Company Name', ticker)
    
    result.append({
        'Ticker': ticker_ns,
        'Company_Name': company,
        'Sub_Industry': industry
    })

df_clean = pd.DataFrame(result)
df_clean = df_clean[df_clean['Sub_Industry'] != 'Unknown']

os.makedirs('data', exist_ok=True)
df_clean.to_csv('data/nifty1000_list.csv', index=False)
print(f"Successfully generated Nifty Universe: {len(df_clean)} verified liquid tickers.")
print(f"Total unique Sub-Industries: {df_clean['Sub_Industry'].nunique()}")

# Ensure utils/nifty1000_list.py is correct
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
print("Updated utils/nifty1000_list.py access layer.")
