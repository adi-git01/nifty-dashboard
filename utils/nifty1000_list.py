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


def load_extended():
    """
    Stocks outside the ranked universe that the dashboard can analyse on demand (market cap of
    Rs 1,000 cr and above, plus anything you hold or watch). data/extended_universe.csv is rebuilt
    weekly by build_extended_universe.py. Returns {ticker: {name, industry, mcap_cr}}.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "extended_universe.csv")
    try:
        df = pd.read_csv(csv_path)
        return {r.Ticker: dict(name=r.Company_Name, industry=r.Sub_Industry, mcap_cr=getattr(r, "mcap_cr", None))
                for r in df.itertuples() if r.Ticker not in SUB_INDUSTRY_MAP}
    except Exception:
        return {}


EXTENDED = load_extended()
# Extended names get an industry everywhere (reports, funnel checklist, My Book); TICKERS_1000 --
# the ranked universe the nightly engine scores -- is unchanged.
for _t, _v in EXTENDED.items():
    if isinstance(_v.get("industry"), str) and _v["industry"] and _v["industry"] != "Unknown":
        SUB_INDUSTRY_MAP[_t] = _v["industry"]
