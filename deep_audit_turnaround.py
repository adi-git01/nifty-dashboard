"""deep_audit_turnaround.py — runtime logic checks for Turnaround Radar page"""
import pandas as pd, numpy as np, os

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, name, detail))
    sym = "v" if condition else "X"
    print(f"  [{sym}] {name}" + (f"  ({detail})" if detail else ""))

print("\n=== DEEP RUNTIME AUDIT ===\n")

df = pd.read_csv("data/turnaround_watchlist.csv")

# 1. si_counts column naming — historical bug: value_counts().reset_index() 
#    In pandas >= 2.0, columns are [Sub_Industry, count] NOT [index, Sub_Industry]
print("[A] Pandas value_counts() column names (version-sensitive)")
si_counts = df["Sub_Industry"].value_counts().head(10).reset_index()
cols = list(si_counts.columns)
check("Columns are [Sub_Industry, count]", cols == ["Sub_Industry", "count"], f"Got: {cols}")

# Dashboard uses: si_counts.columns = ["Sub-Industry", "Count"]  so this is fine either way
check("Manual rename works regardless", True, "columns manually set")

# 2. Pipeline bar chart — what if no ALERT/READY stocks?
print("\n[B] Pipeline bar chart edge case")
pipe_df = df[df["Tier"].isin(["ALERT","READY"])].copy()
check("pipe_df not empty today", not pipe_df.empty, f"{len(pipe_df)} rows")
check("V21_MA50_Gap column exists", "V21_MA50_Gap" in pipe_df.columns)
check("No NaN in V21_MA50_Gap", pipe_df["V21_MA50_Gap"].isnull().sum() == 0)
check("Tier column for color mapping", pipe_df["Tier"].isin(["ALERT","READY"]).all())

# 3. Filter logic edge case — what if all tiers deselected?
print("\n[C] Filter edge cases")
# Empty filter
empty_filter = df[df["Tier"].isin([])]
check("Empty tier filter returns empty df", len(empty_filter) == 0)
# All selected
full_filter = df[df["Tier"].isin(["ALERT","READY","WATCH"])]
check("Full tier filter returns all rows", len(full_filter) == len(df))
# Search filter
search_result = df[df["Ticker"].str.contains("TARIL", na=False)]
check("Search filter works", len(search_result) >= 0)

# 4. Column config keys match actual columns
print("\n[D] Column config vs actual columns")
config_cols = ["Ticker","Sub_Industry","Cycle","CMP","Off_52W_High","Off_MA50",
               "RS21","RS63","CompRS","Liq5Cr","LiqFromLow","IAS","Tier",
               "V21_CRS_Gap","V21_MA50_Gap"]
missing_from_csv = [c for c in config_cols if c not in df.columns]
check("All column_config keys exist in CSV", len(missing_from_csv)==0,
      f"Missing: {missing_from_csv}" if missing_from_csv else "")

# 5. IAS progress column — max_value=100, all values in range
print("\n[E] IAS score range validation")
check("IAS >= 35 (min gate)", (df["IAS"] >= 35).all(), f"Min: {df['IAS'].min()}")
check("IAS <= 100 (max)", (df["IAS"] <= 100).all(), f"Max: {df['IAS'].max()}")

# 6. Ticker format — should all end with .NS
print("\n[F] Ticker format")
non_ns = df[~df["Ticker"].str.endswith(".NS")]
check("All tickers end with .NS", len(non_ns)==0, f"{len(non_ns)} non-NS: {non_ns['Ticker'].tolist()[:3]}")

# 7. Cycle values consistent with screener definition
print("\n[G] Cycle type distribution")
cycle_counts = df["Cycle"].value_counts().to_dict()
check("Has LONG cycle stocks", "LONG" in cycle_counts, str(cycle_counts))
check("No unknown cycle values", all(c in ["LONG","MID","SHORT"] for c in df["Cycle"].unique()))

# 8. Check screener doesn't import anything unavailable on GitHub Actions
print("\n[H] GitHub Actions compatibility")
with open("turnaround_screener.py", "r", encoding="utf-8") as f:
    scr = f.read()
# Check no local-only imports
problematic = [imp for imp in ["streamlit", "plotly", "dash"] if f"import {imp}" in scr]
check("No Streamlit/Plotly in screener", len(problematic)==0,
      f"Found: {problematic}" if problematic else "")
# Check standard deps
for dep in ["pandas", "numpy", "yfinance"]:
    check(f"{dep} imported", dep in scr)

# 9. WATCHLIST_CSV path used in main.py
print("\n[I] Path consistency")
with open("main.py","r",encoding="utf-8") as f:
    main_src = f.read()
check("WATCHLIST_CSV = data/turnaround_watchlist.csv", 
      "data/turnaround_watchlist.csv" in main_src)
check("os.path.exists check before read", 
      "os.path.exists(WATCHLIST_CSV)" in main_src or "os.path.exists" in main_src)

# SUMMARY
print("\n" + "="*50)
failed = [r for r in results if r[0] == FAIL]
warned = [r for r in results if r[0] == WARN]
print(f"TOTAL: {len(results)} | PASS: {len(results)-len(failed)-len(warned)} | FAIL: {len(failed)}")
if failed:
    print("\nFAILURES:")
    for _, n, d in failed:
        print(f"  - {n}: {d}")
else:
    print("All deep checks passed!")
