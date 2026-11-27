import os, glob, sys, requests
import pandas as pd

REPO_N = r'C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\repo_nov21'
REPO_L = r'C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\repo_live'
sys.path.insert(0, REPO_N)
from utils import nse_prices, nse_data as nd, fund_trends as ft, valuation as V

h = nse_prices.history(['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS'])
for tk, df in h.items():
    print(f'repo_nov21 nse_prices[{tk}]: {df.index.min()} -> {df.index.max()} ({len(df)} rows)')

dp = nd.delivery_panel()
print(f'repo_nov21 delivery_panel: shape={dp.shape}, {dp.index.min()} -> {dp.index.max()}')

# Fundamentals date spans
ann_raw, qtr_raw = ft._data(REPO_L)
ann_dates = pd.concat([df['period'] for df in list(ann_raw.values())[:400]])
qtr_dates = pd.concat([df['period'] for df in list(qtr_raw.values())[:400]])
print(f'\nfundamentals_annual ({len(ann_raw)} tks): {ann_dates.min()} -> {ann_dates.max()}')
print('  Annual FY counts (Mar):', ann_dates[ann_dates.dt.month==3].dt.year.value_counts().sort_index().to_dict())
print(f'fundamentals_quarterly ({len(qtr_raw)} tks): {qtr_dates.min()} -> {qtr_dates.max()}')
print('  Quarterly period counts:', {str(k)[:10]: int(v) for k, v in qtr_dates.value_counts().sort_index().items()})

# Monthly prices
for fname in ['monthly_prices.csv.gz', 'monthly_prices.csv']:
    p = os.path.join(REPO_L, 'data', fname)
    if os.path.exists(p):
        mp = pd.read_csv(p, usecols=['date', 'ticker'])
        print(f'\n{fname}: {mp["date"].min()} -> {mp["date"].max()}, {mp["ticker"].nunique()} tickers, {len(mp)} rows')

# Check index files in repo_nov21
idx_pqs = sorted(glob.glob(os.path.join(REPO_N, 'data', 'nse', 'indices', '*.parquet')))
if idx_pqs:
    ix0 = pd.read_parquet(idx_pqs[0])
    print(f'\nFirst index parquet ({os.path.basename(idx_pqs[0])}): cols={list(ix0.columns)}, sample names={ix0["index"].unique()[:8].tolist() if "index" in ix0.columns else "no index col"}')

# Check if valuation.py has as_of parameter
import inspect
print('\nV.value_stock signature:', inspect.signature(V.value_stock))

# Probe NSE pre-2019 archives (2016-01-04 & 2018-01-02)
s = requests.Session()
s.headers.update(nd.HDR)
for label, url in [
    ('sec_bhav 2019-09-20', f'{nd.ARCH}/products/content/sec_bhavdata_full_20092019.csv'),
    ('sec_bhav 2018-01-02', f'{nd.ARCH}/products/content/sec_bhavdata_full_02012018.csv'),
    ('cm_bhav  2016-01-04', f'{nd.ARCH}/content/historical/EQUITIES/2016/JAN/cm04JAN2016bhav.csv.zip'),
    ('cm_bhav  2018-01-02', f'{nd.ARCH}/content/historical/EQUITIES/2018/JAN/cm02JAN2018bhav.csv.zip'),
    ('MTO      2016-01-04', f'{nd.ARCH}/archives/equities/mto/MTO_04012016.DAT'),
    ('MTO      2018-01-02', f'{nd.ARCH}/archives/equities/mto/MTO_02012018.DAT'),
    ('IND      2016-01-04', f'{nd.ARCH}/content/indices/ind_close_all_04012016.csv'),
    ('IND      2018-01-02', f'{nd.ARCH}/content/indices/ind_close_all_02012018.csv'),
]:
    try:
        r = s.get(url, timeout=8)
        print(f'  PROBE {label:22s}: HTTP {r.status_code} ({len(r.content)} bytes)')
    except Exception as e:
        print(f'  PROBE {label:22s}: ERR {e}')
