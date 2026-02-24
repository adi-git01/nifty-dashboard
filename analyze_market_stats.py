import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

print("Loading trade data...")
try:
    trades = pd.read_csv('analysis_2026/bear_regime_trades.csv')
    trades = trades[(trades['Strategy'] == 'RS_LEADER') & (trades['Action'] == 'SELL')].copy()
    trades['Entry_Date'] = pd.to_datetime(trades['Entry_Date'])
    trades['Exit_Date'] = pd.to_datetime(trades['Exit_Date'])
    
    eq = pd.read_csv('analysis_2026/bear_regime_equity.csv')
    eq = eq[eq['Strategy'] == 'RS_LEADER'].copy()
    eq['Date'] = pd.to_datetime(eq['Date'])
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

print("Fetching Nifty data for context...")
start_date = trades['Entry_Date'].min() - pd.Timedelta(days=200)
nifty = yf.Ticker('^NSEI').history(start=start_date.strftime('%Y-%m-%d'))
nifty.index = nifty.index.tz_localize(None)
nifty['MA50'] = nifty['Close'].rolling(50).mean()
nifty['Dist_MA50'] = (nifty['Close'] - nifty['MA50']) / nifty['MA50'] * 100

print("\n--- NIFTY 50-DAY MA PROXIMITY VS TRADE WIN RATES ---")
# Map Nifty's MA50 distance to each trade's entry date
trades['Nifty_Dist_MA50'] = trades['Entry_Date'].map(lambda d: nifty.loc[:d, 'Dist_MA50'].iloc[-1] if not nifty.loc[:d].empty else np.nan)

# Group by Nifty MA50 distance buckets
bins = [-100, -5, -2, 0, 2, 5, 100]
labels = ['Crash (< -5%)', 'Oversold (-5% to -2%)', 'Support (-2% to 0%)', 'Early Trend (0% to +2%)', 'Strong (+2% to +5%)', 'Extended (> +5%)']
trades['Nifty_MA50_Bucket'] = pd.cut(trades['Nifty_Dist_MA50'], bins=bins, labels=labels)

summary = trades.groupby('Nifty_MA50_Bucket').agg(
    Trades=('PnL%', 'count'),
    Win_Rate=('PnL%', lambda x: (x > 0).mean() * 100),
    Avg_PnL=('PnL%', 'mean'),
    Avg_Win=('PnL%', lambda x: x[x>0].mean()),
    Avg_Loss=('PnL%', lambda x: x[x<=0].mean())
).dropna()

print(summary.round(1).to_string())

print("\n--- DRAWDOWNS AND SPRINGBACKS (RECOVERIES) ---")
# Calculate rolling drawdowns on Equity
eq['Peak'] = eq['Equity'].cummax()
eq['Drawdown'] = (eq['Equity'] - eq['Peak']) / eq['Peak'] * 100

# Find significant drawdowns (>5%)
in_dd = False
dd_start = None
dd_trough = None
dd_max = 0
recoveries = []

for idx, row in eq.iterrows():
    if row['Drawdown'] < 0:
        if not in_dd:
            in_dd = True
            dd_start = row['Date']
            dd_max = row['Drawdown']
            dd_trough = row['Date']
        else:
            if row['Drawdown'] < dd_max:
                dd_max = row['Drawdown']
                dd_trough = row['Date']
    elif row['Drawdown'] == 0 and in_dd:
        # Recovered
        in_dd = False
        res = {
            'Start': dd_start.strftime('%Y-%m-%d'),
            'Trough': dd_trough.strftime('%Y-%m-%d'),
            'Recovered': row['Date'].strftime('%Y-%m-%d'),
            'Max_DD_Pct': round(dd_max, 1),
            'Days_To_Trough': (dd_trough - dd_start).days,
            'Days_To_Recover': (row['Date'] - dd_trough).days,
            'Total_DD_Days': (row['Date'] - dd_start).days
        }
        if dd_max <= -5.0: # Only care about >5% drops
            recoveries.append(res)

rec_df = pd.DataFrame(recoveries)
if not rec_df.empty:
    print(rec_df.to_string(index=False))
    print(f"\nAverage Springback Time (Trough to New High): {rec_df['Days_To_Recover'].mean():.0f} days")
    print(f"Average Total Drawdown Duration: {rec_df['Total_DD_Days'].mean():.0f} days")

print("\n--- REGIME VS PERFORMANCE ---")
# Merge regime onto trades
eq_regime = eq[['Date', 'Regime']].set_index('Date')
trades['Entry_Regime'] = trades['Entry_Date'].map(lambda d: eq_regime.loc[d, 'Regime'] if d in eq_regime.index else 'UNKNOWN')

reg_summary = trades.groupby('Entry_Regime').agg(
    Trades=('PnL%', 'count'),
    Win_Rate=('PnL%', lambda x: (x > 0).mean() * 100),
    Avg_PnL=('PnL%', 'mean'),
    Avg_Hold=('Holding_Days', 'mean')
).dropna()
print(reg_summary.round(1).to_string())
