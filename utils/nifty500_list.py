"""
Nifty 500 Stock List with Sector Classifications
Loads from CSV file with proper NSE ticker format
"""

import pandas as pd
import os

def load_nifty500_tickers():
    """
    Loads the complete Nifty 500 list from CSV.
    Returns list of tickers in NSE format (with .NS suffix)
    """
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "nifty500_list.csv")
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Using fallback list.")
        return get_fallback_tickers()
    
    try:
        df = pd.read_csv(csv_path)
        # Add .NS suffix for yfinance
        tickers = [f"{t}.NS" for t in df['Ticker'].tolist() if pd.notna(t) and t.strip()]
        return tickers
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return get_fallback_tickers()


def get_sector_mapping():
    """
    Returns a dictionary mapping ticker to sector.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "nifty500_list.csv")
    
    if not os.path.exists(csv_path):
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        return dict(zip(df['Ticker'].apply(lambda x: f"{x}.NS"), df['Sector']))
    except Exception:
        return {}


def get_fallback_tickers():
    """
    Fallback list of major tickers if CSV is unavailable.
    """
    return [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
        "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
        "BAJFINANCE.NS", "WIPRO.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "TECHM.NS"
    ]


# Load tickers on import
TICKERS = load_nifty500_tickers()
SECTOR_MAP = get_sector_mapping()
