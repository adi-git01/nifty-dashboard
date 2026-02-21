
from utils.volume_monitor import scan_volume_changes
from utils.nifty500_list import TICKERS

# Test on a small sample
sample = TICKERS[:5]
print(f"Testing on: {sample}")

alerts = scan_volume_changes(sample)
print(f"Alerts Found: {len(alerts)}")

if alerts:
    for a in alerts:
        print(a)
else:
    print("No alerts (changes < 2) in this sample. This is normal if volatility is low.")
