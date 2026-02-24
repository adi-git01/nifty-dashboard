"""
DNA3 FINAL VALIDATION SUITE
============================
Three analyses in one script (shared data download):

  PART 1: 2025 Underperformance Diagnosis
    - Market breadth analysis (how narrow was the Nifty rally?)
    - What stocks did the strategies hold vs what was working?
    
  PART 2: Rebalance Cadence Fine-Tuning
    - Test 10, 12, 13, 14, 15, 20 day rebalance for RawComp-V21
    
  PART 3: Walk-Forward Validation
    - Train: 2011-2019 (in-sample)
    - Test: 2020-2026 (out-of-sample)
    - Compare to detect overfitting

Usage:
  python dna3_final_validation.py
"""

from dna3_ultimate_comparison import (
    Engine, pain_metrics, fetch_data, detect_regime, tf_rs, g_factor,
    calc_ind, INITIAL_CAPITAL, OUTPUT_DIR, REGIME_CFG
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ============================================================
# PART 1: 2025 DIAGNOSIS
# ============================================================
def diagnose_2025(nifty, dc):
    print("\n" + "=" * 130)
    print("PART 1: WHY DID EVERYTHING FAIL IN 2025 WHILE NIFTY WAS +10.5%?")
    print("=" * 130)

    # Define 2025 window
    y25_start = pd.Timestamp('2025-01-01')
    y25_end = nifty.index[-1]

    si = nifty.index.searchsorted(y25_start)
    if si >= len(nifty) - 5:
        print("  Not enough 2025 data."); return
    
    nifty_25 = nifty.iloc[si:]
    n_ret = (nifty_25['Close'].iloc[-1] / nifty_25['Close'].iloc[0] - 1) * 100

    # 1. Market Breadth: What % of stocks outperformed Nifty in 2025?
    print(f"\n  Nifty 2025 YTD Return: {n_ret:+.1f}%")
    print(f"  Period: {nifty_25.index[0].date()} -> {nifty_25.index[-1].date()}")

    stock_rets = {}
    winners = 0; losers = 0; outperformers = 0
    for t, df in dc.items():
        if t == 'NIFTY': continue
        si2 = df.index.searchsorted(y25_start)
        if si2 >= len(df) - 5: continue
        d25 = df.iloc[si2:]
        if len(d25) < 10: continue
        ret = (d25['Close'].iloc[-1] / d25['Close'].iloc[0] - 1) * 100
        stock_rets[t] = ret
        if ret > 0: winners += 1
        else: losers += 1
        if ret > n_ret: outperformers += 1

    total = winners + losers
    print(f"\n  MARKET BREADTH ANALYSIS:")
    print(f"    Stocks with +ve return in 2025 : {winners}/{total} ({winners/total*100:.0f}%)")
    print(f"    Stocks with -ve return in 2025 : {losers}/{total} ({losers/total*100:.0f}%)")
    print(f"    Stocks beating Nifty           : {outperformers}/{total} ({outperformers/total*100:.0f}%)")

    # 2. Top/Bottom performers
    sorted_rets = sorted(stock_rets.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  TOP 10 PERFORMERS IN 2025:")
    for t, r in sorted_rets[:10]:
        sec = __import__('utils.nifty500_list', fromlist=['SECTOR_MAP']).SECTOR_MAP.get(t, 'Unknown')
        print(f"    {t:<20} {r:>+8.1f}%  [{sec}]")

    print(f"\n  BOTTOM 10 PERFORMERS IN 2025:")
    for t, r in sorted_rets[-10:]:
        sec = __import__('utils.nifty500_list', fromlist=['SECTOR_MAP']).SECTOR_MAP.get(t, 'Unknown')
        print(f"    {t:<20} {r:>+8.1f}%  [{sec}]")

    # 3. Sector breakdown
    from utils.nifty500_list import SECTOR_MAP
    sector_rets = {}
    for t, r in stock_rets.items():
        sec = SECTOR_MAP.get(t, 'Unknown')
        if sec not in sector_rets: sector_rets[sec] = []
        sector_rets[sec].append(r)

    print(f"\n  SECTOR PERFORMANCE IN 2025 (avg return, sorted):")
    sec_avg = [(s, np.mean(rs), len(rs)) for s, rs in sector_rets.items() if len(rs) >= 3]
    sec_avg.sort(key=lambda x: x[1], reverse=True)
    for s, avg, n in sec_avg[:15]:
        pct_pos = sum(1 for r in sector_rets[s] if r > 0) / len(sector_rets[s]) * 100
        print(f"    {s:<40} {avg:>+7.1f}% (n={n:>3}, {pct_pos:.0f}% +ve)")

    print(f"\n  WORST SECTORS IN 2025:")
    for s, avg, n in sec_avg[-10:]:
        pct_pos = sum(1 for r in sector_rets[s] if r > 0) / len(sector_rets[s]) * 100
        print(f"    {s:<40} {avg:>+7.1f}% (n={n:>3}, {pct_pos:.0f}% +ve)")

    # 4. What RS characteristics were working in 2025?
    print(f"\n  DID HIGH RS STOCKS WORK IN 2025?")
    # Calculate RS as of Jan 1 2025 and see if high RS predicted 2025 return
    rs_vs_ret = []
    for t, df in dc.items():
        if t == 'NIFTY' or t not in stock_rets: continue
        i = df.index.searchsorted(y25_start)
        if i < 100: continue
        w = df.iloc[max(0, i-252):i+1]
        ni = nifty.index.searchsorted(y25_start)
        nw = nifty.iloc[max(0, ni-252):ni+1]
        rs = tf_rs(w, nw, 63)
        rs_vs_ret.append((t, rs, stock_rets[t]))

    rs_vs_ret.sort(key=lambda x: x[1], reverse=True)
    
    # Top RS quintile vs Bottom RS quintile
    q = len(rs_vs_ret) // 5
    top_q = rs_vs_ret[:q]
    bot_q = rs_vs_ret[-q:]
    mid_q = rs_vs_ret[2*q:3*q]

    print(f"    Top RS quintile (entered 2025 with highest RS):")
    print(f"      Avg 2025 Return: {np.mean([x[2] for x in top_q]):+.1f}%")
    print(f"      % Positive:      {sum(1 for x in top_q if x[2] > 0)/len(top_q)*100:.0f}%")
    
    print(f"    Middle RS quintile:")
    print(f"      Avg 2025 Return: {np.mean([x[2] for x in mid_q]):+.1f}%")
    
    print(f"    Bottom RS quintile (entered 2025 with lowest RS):")
    print(f"      Avg 2025 Return: {np.mean([x[2] for x in bot_q]):+.1f}%")
    print(f"      % Positive:      {sum(1 for x in bot_q if x[2] > 0)/len(bot_q)*100:.0f}%")

    is_reversal = np.mean([x[2] for x in bot_q]) > np.mean([x[2] for x in top_q])
    if is_reversal:
        print(f"\n    --> MOMENTUM REVERSAL: Low-RS stocks outperformed High-RS stocks in 2025!")
        print(f"        This is exactly why momentum strategies fail — 2025 was a mean-reversion market.")
    else:
        print(f"\n    --> Momentum still worked but the narrow breadth limited how many high-RS stocks were available.")


# ============================================================
# PART 2: REBALANCE CADENCE FINE-TUNING
# ============================================================
def tune_rebalance(nifty, dc):
    print("\n\n" + "=" * 130)
    print("PART 2: REBALANCE CADENCE FINE-TUNING (RawComp-V21)")
    print("=" * 130)

    now = datetime.now()
    cadences = [7, 10, 12, 13, 14, 15, 18, 20]

    cfg_base = {
        'use_composite': True, 'weights': (0.1, 0.5, 0.4),
        'g_mode': 'Off', 'stype': 'v21',
    }

    results = []
    horizons = {'5y': 5, '10y': 10, '15y': 15}

    for hname, years in horizons.items():
        s_dt = now - timedelta(days=int(365.25 * years))
        si = nifty.index.searchsorted(s_dt)
        if si >= len(nifty) - 10: continue
        actual_start = nifty.index[si]
        actual_end = nifty.index[-1]
        actual_years = (actual_end - actual_start).days / 365.25

        print(f"\n  {hname.upper()} ({actual_start.date()} -> {actual_end.date()}):")
        print(f"  {'Reb':<5} {'CAGR%':>8} {'Sharpe':>8} {'Expect':>8} {'WinRate':>8} {'Whipsaw':>8} {'DD%':>8} {'DDDays':>8}")
        print(f"  {'-' * 70}")

        for reb in cadences:
            cfg = {**cfg_base, 'rebalance': reb}
            eng = Engine(f'Reb{reb}', cfg)
            eq = eng.run(dc, nifty, actual_start, actual_end)
            m = pain_metrics(eq, eng.trades, actual_years) if eq is not None else {}

            cagr = m.get('CAGR%', 0)
            sharpe = m.get('Sharpe', 0)
            exp = m.get('Expect%', 0)
            wr = m.get('WinRate%', 0)
            whip = m.get('Whipsaw%', 0)
            dd = m.get('MaxDD%', 0)
            ddd = m.get('DD_Days', 0)

            print(f"  {reb:<5} {cagr:>8.2f} {sharpe:>8.2f} {exp:>7.1f}% {wr:>7.1f}% {whip:>7.1f}% {dd:>7.1f}% {ddd:>7}")

            results.append({
                'Horizon': hname, 'Rebalance': reb, 'CAGR': cagr,
                'Sharpe': sharpe, 'Expectancy': exp, 'WinRate': wr,
                'Whipsaw': whip, 'MaxDD': dd, 'DD_Days': ddd
            })

    df = pd.DataFrame(results)
    df.to_csv(f"{OUTPUT_DIR}/rebalance_tuning.csv", index=False)

    # Find optimal
    for hname in horizons:
        hdf = df[df['Horizon'] == hname]
        if hdf.empty: continue
        best_cagr = hdf.loc[hdf['CAGR'].idxmax()]
        best_sharpe = hdf.loc[hdf['Sharpe'].idxmax()]
        print(f"\n  {hname.upper()} OPTIMAL:")
        print(f"    Best CAGR:   Reb={int(best_cagr['Rebalance'])} ({best_cagr['CAGR']:.1f}%)")
        print(f"    Best Sharpe: Reb={int(best_sharpe['Rebalance'])} ({best_sharpe['Sharpe']:.2f})")


# ============================================================
# PART 3: WALK-FORWARD VALIDATION
# ============================================================
def walk_forward(nifty, dc):
    print("\n\n" + "=" * 130)
    print("PART 3: WALK-FORWARD VALIDATION (Is Composite RS Overfit?)")
    print("=" * 130)
    print("  Train: 2011-2019 (in-sample)  |  Test: 2020-2026 (out-of-sample)")

    train_start = pd.Timestamp('2011-06-01')
    train_end = pd.Timestamp('2019-12-31')
    test_start = pd.Timestamp('2020-01-02')
    test_end = nifty.index[-1]

    strategies = {
        'RawComp-V21': {
            'use_composite': True, 'weights': (0.1, 0.5, 0.4),
            'g_mode': 'Off', 'rebalance': 15, 'stype': 'v21',
        },
        'LegV2.1': {
            'use_composite': False, 'weights': (0, 0, 1),
            'g_mode': 'Off', 'rebalance': 10, 'stype': 'v21',
        },
    }

    for period_name, p_start, p_end in [
        ('TRAIN (2011-2019)', train_start, train_end),
        ('TEST (2020-2026)', test_start, test_end),
    ]:
        si = nifty.index.searchsorted(p_start)
        ei = nifty.index.searchsorted(p_end)
        if si >= ei - 10:
            print(f"\n  [{period_name}] Insufficient data."); continue
        
        a_start = nifty.index[si]
        a_end = nifty.index[min(ei, len(nifty)-1)]
        years = (a_end - a_start).days / 365.25

        # Nifty
        ns = nifty.iloc[si]['Close']
        ne = nifty.iloc[min(ei, len(nifty)-1)]['Close']
        n_cagr = ((ne/ns)**(1/years) - 1)*100

        print(f"\n  {period_name} ({a_start.date()} -> {a_end.date()}, {years:.1f}y)")
        print(f"  {'Strategy':<16} {'CAGR%':>8} {'Sharpe':>8} {'Expect':>8} {'WinRate':>8} {'MaxDD':>8} {'Whipsaw':>8}")
        print(f"  {'-' * 75}")

        for sn, cfg in strategies.items():
            eng = Engine(sn, cfg)
            eq = eng.run(dc, nifty, a_start, a_end)
            m = pain_metrics(eq, eng.trades, years) if eq is not None else {}
            print(f"  {sn:<16} {m.get('CAGR%', 0):>8.2f} {m.get('Sharpe', 0):>8.2f} {m.get('Expect%', 0):>7.1f}% {m.get('WinRate%', 0):>7.1f}% {m.get('MaxDD%', 0):>7.1f}% {m.get('Whipsaw%', 0):>7.1f}%")

        print(f"  {'Nifty':<16} {n_cagr:>8.2f}{'':>8}{'':>8}{'':>8}{'':>8}{'':>8}")

    # Consistency check
    print(f"\n  OVERFITTING CHECK:")
    print(f"    If RawComp-V21 beats LegV2.1 in BOTH train AND test periods,")
    print(f"    the composite RS signal is robust and NOT overfit to a specific window.")


# ============================================================
# MAIN
# ============================================================
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nifty, dc = fetch_data()
    if nifty is None or nifty.empty: return

    print("\n" + "=" * 130)
    print("DNA3 FINAL VALIDATION SUITE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 130)

    diagnose_2025(nifty, dc)
    tune_rebalance(nifty, dc)
    walk_forward(nifty, dc)

    print("\n\n" + "=" * 130)
    print("ALL ANALYSES COMPLETE")
    print(f"Files saved to {OUTPUT_DIR}/")
    print("=" * 130)


if __name__ == "__main__":
    run()
