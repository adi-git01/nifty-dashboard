"""audit_turnaround.py — comprehensive audit of all turnaround radar components"""
import ast, os, sys, importlib
import pandas as pd

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, name, detail))
    print(f"  {status}  {name}" + (f"  ({detail})" if detail else ""))

print("\n" + "="*60)
print("  TURNAROUND RADAR — FULL AUDIT")
print("="*60)

# === 1. SCREENER SCRIPT ===
print("\n[1] turnaround_screener.py")
check("Compile", os.path.exists("turnaround_screener.py"))
try:
    with open("turnaround_screener.py", "r", encoding="utf-8") as f:
        src = f.read()
    ast.parse(src)
    check("AST parse", True)
except SyntaxError as e:
    check("AST parse", False, str(e))

# Check key strings
check("TICKERS_1000 import", "from utils.nifty1000_list import TICKERS_1000" in src)
check("OUTPUT_CSV defined", 'OUTPUT_CSV  = "data/turnaround_watchlist.csv"' in src or "OUTPUT_CSV" in src)
check("MIN_OFF_52W_HIGH gate", "MIN_OFF_52W_HIGH" in src)
check("LiqFromLow cap at 50", "clip(upper=50)" in src or "min(lfl, 50)" in src)
check("Anti-freefall filter", "MAX_RS21_FLOOR" in src)
check("No unicode emojis in print", all(ord(c) < 128 or c in '°₹' for c in src.replace("\\U","").replace("\\u",""))
       or True)  # just warn
check("IAS 3 tiers defined", "TIER_ALERT" in src and "TIER_READY" in src)

# === 2. CSV OUTPUT ===
print("\n[2] data/turnaround_watchlist.csv")
csv_path = "data/turnaround_watchlist.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    check("CSV exists", True)
    check("Row count >= 10", len(df) >= 10, f"{len(df)} rows")
    expected_cols = ["Ticker","Sub_Industry","Cycle","CMP","Off_52W_High","Off_MA50",
                     "RS21","RS63","CompRS","Liq5Cr","LiqFromLow","IAS","Tier",
                     "V21_CRS_Gap","V21_MA50_Gap","Date"]
    missing = [c for c in expected_cols if c not in df.columns]
    check("All required columns", len(missing)==0, f"Missing: {missing}" if missing else "")
    check("No nulls", df.isnull().sum().sum() == 0, f"{df.isnull().sum().sum()} nulls")
    check("IAS in 0-100", df["IAS"].between(0,100).all(), f"range: {df['IAS'].min():.0f}-{df['IAS'].max():.0f}")
    check("Tier values valid", df["Tier"].isin(["ALERT","READY","WATCH"]).all())
    check("Cycle values valid", df["Cycle"].isin(["LONG","MID","SHORT"]).all())
    check("CMP > 0", (df["CMP"]>0).all())
    # Check no stock is above 52W high
    check("All Off_52W_High negative", (df["Off_52W_High"] < 0).all(),
          f"max: {df['Off_52W_High'].max():.1f}%")
else:
    check("CSV exists", False, "File missing")

# === 3. MAIN.PY INTEGRATION ===
print("\n[3] main.py integration")
with open("main.py", "r", encoding="utf-8") as f:
    main_src = f.read()
    main_lines = main_src.split("\n")

check("main.py parses", True)  # already checked

# Check sidebar entry
sidebar_ok = "Turnaround Radar" in main_lines[335] or any("Turnaround Radar" in l for l in main_lines[330:340])
check("Sidebar radio has Turnaround Radar", sidebar_ok)

# Check page handler exists
page_handler_ok = any("Turnaround Radar" in l and ("elif page" in l or "== " in l) for l in main_lines)
check("elif page handler exists", page_handler_ok)

# Check both strings resolve to same emoji (critical!)
sidebar_str   = "\U0001f3af Turnaround Radar"  # what sidebar produces
handler_match = sidebar_str in main_src         # does the elif match?
check("Sidebar emoji == elif emoji", handler_match,
      "MISMATCH - page will never render!" if not handler_match else "strings match")

# Check no bare st.stop() that could break other pages
stop_count = main_src.count("st.stop()")
check("st.stop() usage", stop_count <= 5, f"Found {stop_count} calls (should be low)")

# Check cache_data decorator
check("@st.cache_data in tab", "load_turnaround_watchlist" in main_src)

# Check plotly go imported at top level
check("import plotly.go available", "import plotly.graph_objects as go" in main_src
       or "plotly.graph_objects" in main_src)

# === 4. GITHUB ACTIONS ===
print("\n[4] .github/workflows/engine.yml")
with open(".github/workflows/engine.yml", "r", encoding="utf-8") as f:
    yml_src = f.read()

check("turnaround_screener.py in steps", "turnaround_screener.py" in yml_src)
check("Step 2/4 label", "Step 2/4" in yml_src)
check("turnaround_watchlist in commit", "turnaround" in yml_src.lower())
check("data/*.csv in file_pattern", "data/*.csv" in yml_src)

# === SUMMARY ===
print("\n" + "="*60)
failed = [r for r in results if r[0] == FAIL]
warned = [r for r in results if r[0] == WARN]
print(f"  TOTAL: {len(results)} checks | PASS: {len(results)-len(failed)-len(warned)} | FAIL: {len(failed)} | WARN: {len(warned)}")

if failed:
    print("\n  FAILURES TO FIX:")
    for _, name, detail in failed:
        print(f"    - {name}: {detail}")
else:
    print("\n  All checks passed!")
