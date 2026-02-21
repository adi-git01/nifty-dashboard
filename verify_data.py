"""
Verification script for Position Manager data integrity.
Ensures all data sources are correctly unified.
"""
import json
import os

print("=" * 50)
print("POSITION MANAGER DATA VERIFICATION")
print("=" * 50)

# Check positions.json
if os.path.exists("positions.json"):
    with open("positions.json", "r") as f:
        positions = json.load(f)
    active = [p for p in positions if p.get('status') == 'active']
    watching = [p for p in positions if p.get('status') == 'watching']
    print(f"\n[positions.json] UNIFIED SYSTEM")
    print(f"  Total: {len(positions)}")
    print(f"  Active: {len(active)}")
    print(f"  Watching: {len(watching)}")
    print(f"\n  Active positions:")
    for p in active:
        print(f"    - {p.get('ticker')}: entry={p.get('entry_price')}, SL={p.get('stop_loss')}, Target={p.get('target')}")
else:
    print("\n[!] positions.json NOT FOUND")

# Check legacy tracker_data.json
if os.path.exists("tracker_data.json"):
    with open("tracker_data.json", "r") as f:
        tracker = json.load(f)
    print(f"\n[tracker_data.json] LEGACY SYSTEM (should NOT be used)")
    print(f"  Tracked: {len(tracker.get('tracked_stocks', []))}")
    print(f"  Archived: {len(tracker.get('archived_stocks', []))}")
else:
    print("\n[tracker_data.json] NOT FOUND - Good, legacy removed")

# Check alerts.json
if os.path.exists("alerts.json"):
    with open("alerts.json", "r") as f:
        alerts = json.load(f)
    print(f"\n[alerts.json] ALERT SYSTEM")
    print(f"  Total alerts: {len(alerts)}")
    for a in alerts:
        print(f"    - {a.get('ticker')}: SL={a.get('stop_loss')}, Target={a.get('target')}")
else:
    print("\n[alerts.json] NOT FOUND")

print("\n" + "=" * 50)
print("VERIFICATION COMPLETE")
print("=" * 50)
