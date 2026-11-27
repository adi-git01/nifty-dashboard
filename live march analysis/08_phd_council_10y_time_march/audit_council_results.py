import json

with open(r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\phd_council_10y\phd_council_10y_results.json", "r", encoding="utf-8") as f:
    D = json.load(f)

print("=== 1. REGIME-BY-REGIME COMPARISON ===")
for rn, rv in D["regimes"].items():
    p, n, e = rv["PHD_COUNCIL"], rv["NIFTY_50"], rv["EW_TRADABLE"]
    print(f"  {rn:18s} | PhD: {p['ret']:+7.2f}% (CAGR {p['cagr']:+6.1f}%, DD {p['mdd']:6.2f}%, Sh {p['sharpe']:5.2f}) | N50: {n['ret']:+7.2f}% (DD {n['mdd']:6.2f}%) | EW: {e['ret']:+7.2f}% (DD {e['mdd']:6.2f}%)")

print("\n=== 2. EPOCH-BY-EPOCH TIMELINE (E01 -> E22) ===")
for ep in D["epoch_logs"]:
    print(f"  {ep['epoch']} {ep['date']} | {ep['macro']['mode']:8s} B50={ep['macro']['breadth_above_50dma_pct']:4.1f}% VIX={ep['macro']['vix']:4.1f} | Gross={ep['target_gross_pct']:3.0f}% {ep['gross_votes']} | B={len(ep['buys']):2d} X={len(ep['exits']):2d} S={len(ep['stops']):2d} Flips={ep['n_flips']:2d} | Ret={ep['ep_ret_pct']:+6.1f}% | NAV=Rs {ep['end_nav']:>10,.0f}")

print("\n=== 3. TOP 15 MULTIBAGGERS & EXACT TRADE LIFECYCLES ===")
trades_by_tk = {}
for t in D["trades"]:
    trades_by_tk.setdefault(t["tk"], []).append(t)

for tk, pnl in D["top15_winners"]:
    trs = trades_by_tk.get(tk, [])
    seq = " -> ".join(f"{t['side']}({t['epoch']} {t['date']} @{t['px']} {t['reason']} {t['pnl_pct']:+.0f}%)" for t in trs)
    print(f"  {tk:15s} | NetPnL: Rs {pnl:+9,d} | {seq}")

print("\n=== 4. WORST 10 LOSERS ===")
for tk, pnl in D["bot10_losers"]:
    trs = trades_by_tk.get(tk, [])
    seq = " -> ".join(f"{t['side']}({t['epoch']} {t['date']} @{t['px']} {t['reason']} {t['pnl_pct']:+.0f}%)" for t in trs)
    print(f"  {tk:15s} | NetPnL: Rs {pnl:+9,d} | {seq}")

print("\n=== 5. KEY CRISIS & INFLECTION EPOCH DEBATES (E05 2018-Bear, E09 COVID-Crash, E10 COVID-Recovery, E14 2022-Bear, E16 SuperBull, E19 Post-24) ===")
for ep_id in ["E05", "E09", "E10", "E14", "E16", "E19", "E20"]:
    ep = next(x for x in D["epoch_logs"] if x["epoch"] == ep_id)
    print(f"\n--- {ep['epoch']} ({ep['date']}, {ep['macro']['mode']}, Gross={ep['target_gross_pct']}%, EpRet={ep['ep_ret_pct']:+.1f}%) ---")
    for r2 in ep["r2"]:
        print(f"  {r2['expert']} (Gross={r2['gross']}%): {r2['summary']}")
        if r2["flips"]:
            print(f"    Flips: {r2['flips'][:3]}")

# Check if any notorious 2018-2020 blowups were ever bought
notorious = {"DHFL.NS", "YESBANK.NS", "JETAIRWAYS.NS", "VAKRANGEE.NS", "PCJEWELLER.NS", "HDIL.NS", "RELCAPITAL.NS", "RELINFRA.NS", "RCOM.NS", "IDEA.NS", "PAYTM.NS", "BYJUS.NS"}
hit_not = [t for t in D["trades"] if t["tk"] in notorious]
print(f"\n=== 6. NOTORIOUS BLOWUP AUDIT ===")
print(f"  Notorious blowups traded: {len(hit_not)} -> {hit_not}")
print(f"  Open holdings at end (2026-11-20): {D['open_holdings_final']}")
