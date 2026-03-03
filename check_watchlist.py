import pandas as pd
df = pd.read_csv("data/turnaround_watchlist.csv")
print("=== TURNAROUND WATCHLIST ===")
print(f"Stocks: {len(df)}")
print(f"Tiers: {df['Tier'].value_counts().to_dict()}")
print(f"Cycles: {df['Cycle'].value_counts().to_dict()}")
print()
print("=== ALERT TIER ===")
alert = df[df['Tier']=='ALERT'][['Ticker','Sub_Industry','CMP','Off_52W_High','RS21','RS63','Liq5Cr','LiqFromLow','IAS']]
print(alert.to_string(index=False))
print()
print("=== COLUMNS ===")
print(list(df.columns))
