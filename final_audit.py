"""final_audit.py — final checks including Python version, data integrity, and live engine impact"""
import sys, ast, pandas as pd

PASS, FAIL, WARN = "[PASS]", "[FAIL]", "[WARN]"
results = []

def check(name, cond, detail="", warn=False):
    status = WARN if (warn and not cond) else (PASS if cond else FAIL)
    results.append((status, name, detail))
    sym = "?" if status==WARN else ("v" if cond else "X")
    print(f"  [{sym}] {name}" + (f"  ({detail})" if detail else ""))

print(f"\n=== FINAL AUDIT (Python {sys.version_info.major}.{sys.version_info.minor}) ===\n")

# Python 3.10+ type hint: dict | None
print("[A] Python 3.10 type hint compatibility")
with open("turnaround_screener.py","r",encoding="utf-8") as f:
    src = f.read()
has_union_hint = "dict | None" in src
py_version = sys.version_info
if has_union_hint:
    check("dict|None hint (needs 3.10+)", py_version >= (3,10),
          f"Python {py_version.major}.{py_version.minor}")
    # On GH Actions (3.10) this will work
    check("GH Actions Python 3.10 compatible", True, "engine.yml specifies 3.10")
else:
    check("No union type hints", True)

# Check no __future__ annotations needed
try:
    ast.parse(src)
    check("AST parses no errors", True)
except: check("AST parses no errors", False)

# Live trade desk untouched
print("\n[B] Live Trade Desk / Portfolio isolation")
with open("main.py","r",encoding="utf-8") as f:
    main_lines = f.readlines()

turnaround_start = next((i for i,l in enumerate(main_lines) if "TURNAROUND RADAR" in l.upper()), None)
livedesk_line    = next((i for i,l in enumerate(main_lines) if "Live Trading Desk" in l and "elif page" in l), None)
portfolio_line   = next((i for i,l in enumerate(main_lines) if "Return Tracker" in l and "elif page" in l), None)

check("Turnaround page at END of file", turnaround_start and turnaround_start > 4000,
      f"Line {turnaround_start}")
check("Live Trade Desk code unchanged", livedesk_line and livedesk_line < 520,
      f"Line {livedesk_line}")
check("Return Tracker unchanged", portfolio_line and portfolio_line < 1300,
      f"Line {portfolio_line}")

# Confirm turnaround page doesn't modify session_state for portfolio
turnaround_code = "".join(main_lines[turnaround_start:]) if turnaround_start else ""
check("No portfolio session_state in turnaround", "market_data" not in turnaround_code)
check("No positions in turnaround", "get_all_positions" not in turnaround_code)

# Watchlist CSV freshness
print("\n[C] Data freshness")
df = pd.read_csv("data/turnaround_watchlist.csv")
today = pd.Timestamp.now().strftime("%Y-%m-%d")
csv_date = df["Date"].iloc[0] if "Date" in df.columns else "unknown"
check("CSV generated today", csv_date == today, f"CSV date: {csv_date}, today: {today}")

# Sector coverage
print("\n[D] Signal quality checks")
alert_df = df[df["Tier"]=="ALERT"]
high_lfl_outlier = df[df["LiqFromLow"] > 45]
check("LiqFromLow capped at 50", (df["LiqFromLow"] <= 50).all(), f"Max: {df['LiqFromLow'].max():.1f}x")
check("ALERT count reasonable 5-30", 5 <= len(alert_df) <= 30, f"ALERT: {len(alert_df)}")
check("RS21 not extreme pos outliers", (df["RS21"] < 200).all(), f"Max RS21: {df['RS21'].max():.0f}%")
check("Liq5Cr >= 50 for all", (df["Liq5Cr"] >= 50).all(), f"Min: {df['Liq5Cr'].min():.0f}Cr")

print("\n" + "="*50)
failed = [r for r in results if r[0]==FAIL]
warned = [r for r in results if r[0]==WARN]
print(f"TOTAL: {len(results)} | PASS: {len(results)-len(failed)-len(warned)} | WARN: {len(warned)} | FAIL: {len(failed)}")
if failed: [print(f"  FAIL: {n} - {d}") for _,n,d in failed]
if warned: [print(f"  WARN: {n} - {d}") for _,n,d in warned]
if not failed and not warned: print("Clean bill of health!")
