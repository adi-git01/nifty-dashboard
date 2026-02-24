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