import pandas as pd
try:
    df = pd.read_csv("nifty500_cache.csv", nrows=2)
    print("Columns:", df.columns.tolist())
except Exception as e:
    print("Error:", e)
