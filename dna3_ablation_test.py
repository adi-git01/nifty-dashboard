"""
ABLATION TEST: What's Killing OptV3.1?
=======================================
Strips G-Factor and Regime Sizing one layer at a time:

  1. RawComp-V31  = Composite 10/50/40, 15d rebal, Regime sizing, NO G-Factor
  2. RawComp-V21  = Composite 10/50/40, 15d rebal, NO regime, NO G-Factor  
  3. OptV3.1      = Composite 10/50/40, 15d rebal, Regime sizing, WITH G-Factor (baseline)
  4. LegV2.1      = 63d RS, 10d rebal, no regime, no G-Factor (benchmark)
  5. Nifty

Usage:
  python dna3_ablation_test.py
"""

# Reuse the full engine from the ultimate comparison script
from dna3_ultimate_comparison import (
    Engine, pain_metrics, fetch_data, detect_regime,
    INITIAL_CAPITAL, OUTPUT_DIR, HORIZONS, MAX_YEARS
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

STRATEGIES = {
    'RawComp-V31': {
        'use_composite': True, 'weights': (0.1, 0.5, 0.4),
        'g_mode': 'Off', 'rebalance': 15, 'stype': 'v31',
    },
    'RawComp-V21': {
        'use_composite': True, 'weights': (0.1, 0.5, 0.4),
        'g_mode': 'Off', 'rebalance': 15, 'stype': 'v21',
    },
    'OptV3.1-GF': {
        'use_composite': True, 'weights': (0.1, 0.5, 0.4),
        'g_mode': 'Filter', 'rebalance': 15, 'stype': 'v31',
    },
    'LegV2.1': {
        'use_composite': False, 'weights': (0, 0, 1),
        'g_mode': 'Off', 'rebalance': 10, 'stype': 'v21',
    },
}

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nifty, dc = fetch_data()
    if nifty is None or nifty.empty: return

    now = datetime.now()
    snames = list(STRATEGIES.keys())
    all_summary = []
    longest_eq = {}

    print("\n" + "=" * 130)
    print("ABLATION TEST: STRIPPING G-FACTOR & REGIME SIZING FROM OptV3.1")
    print("=" * 130)
    print("  RawComp-V31 = Composite RS + Regime Sizing (NO G-Factor)")
    print("  RawComp-V21 = Composite RS only (NO G-Factor, NO Regime Sizing)")
    print("  OptV3.1-GF  = Composite RS + Regime Sizing + G-Factor (original)")
    print("  LegV2.1     = 63d RS benchmark")

    for hname, years in HORIZONS.items():
        s_dt = now - timedelta(days=int(365.25 * years))
        si = nifty.index.searchsorted(s_dt)
        if si >= len(nifty) - 10:
            print(f"\n  [{hname}] Insufficient data."); continue

        actual_start = nifty.index[si]
        actual_end = nifty.index[-1]
        actual_years = (actual_end - actual_start).days / 365.25

        # Nifty
        ns = nifty.iloc[si]['Close']; ne = nifty.iloc[-1]['Close']
        n_cagr = ((ne/ns)**(1/actual_years) - 1)*100

        row = {'Horizon': hname, 'Years': round(actual_years, 1),
               'Nifty_CAGR%': round(n_cagr, 2)}

        print(f"\n{'_' * 130}")
        print(f"  {hname.upper()} ({actual_start.date()} -> {actual_end.date()}, {actual_years:.1f}y)")
        print(f"{'_' * 130}")

        h = f"  {'Metric':<16}"
        for sn in snames: h += f" {sn:>14}"
        h += f" {'Nifty':>10}"
        print(h); print(f"  {'-' * 110}")

        strat_m = {}
        for sn in snames:
            eng = Engine(sn, STRATEGIES[sn])
            eq = eng.run(dc, nifty, actual_start, actual_end)
            m = pain_metrics(eq, eng.trades, actual_years) if eq is not None else {}
            strat_m[sn] = m
            for k, v in m.items(): row[f'{sn}_{k}'] = v

            if hname == max(HORIZONS.keys(), key=lambda k: HORIZONS[k]):
                longest_eq[sn] = eq

        for metric in ['CAGR%', 'MaxDD%', 'DD_Days', 'Sharpe', 'Sortino',
                        'Trades', 'WinRate%', 'Expect%', 'Whipsaw%',
                        'MaxWin%', 'MaxLoss%', 'MaxConsecLoss', 'AvgHold']:
            line = f"  {metric:<16}"
            for sn in snames:
                v = strat_m[sn].get(metric, '-')
                line += f" {v:>14}" if isinstance(v, (int, float)) else f" {'-':>14}"
            nv = row.get(f'Nifty_{metric}', '-')
            line += f" {nv:>10}" if isinstance(nv, (int, float)) else f" {'-':>10}"
            print(line)

        # Delta vs LegV2.1
        leg = strat_m.get('LegV2.1', {}).get('CAGR%', 0)
        for sn in snames:
            c = strat_m[sn].get('CAGR%', 0)
            if isinstance(c, (int, float)) and isinstance(leg, (int, float)):
                row[f'{sn}_vs_Leg'] = round(c - leg, 2)

        best_cagr = -999; best = ''
        for sn in snames:
            c = strat_m[sn].get('CAGR%', -999)
            if isinstance(c, (int, float)) and c > best_cagr: best_cagr = c; best = sn
        print(f"\n  >>> WINNER: {best} ({best_cagr:+.1f}% CAGR)")

        # Show delta
        print(f"  >>> vs LegV2.1: ", end="")
        for sn in snames:
            d = row.get(f'{sn}_vs_Leg', 0)
            if isinstance(d, (int, float)):
                print(f" {sn}={d:+.1f}%", end="")
        print()

        all_summary.append(row)

    pd.DataFrame(all_summary).to_csv(f"{OUTPUT_DIR}/ablation_summary.csv", index=False)

    # Calendar year returns for longest horizon
    if longest_eq:
        print(f"\n\n{'=' * 130}")
        print("CALENDAR YEAR RETURNS (Longest Horizon)")
        print("=" * 130)

        yr_rets = {}
        for sn in longest_eq:
            eq = longest_eq[sn]
            if eq is None or len(eq) < 22: continue
            eq['y'] = eq['date'].dt.year
            yg = eq.groupby('y')['equity'].last()
            yr_rets[sn] = yg.pct_change() * 100

        # Nifty
        si = nifty.index.searchsorted(now - timedelta(days=int(365.25 * MAX_YEARS)))
        ne = nifty.iloc[si:].copy(); ne['y'] = ne.index.year
        yr_rets['Nifty'] = ne.groupby('y')['Close'].last().pct_change() * 100

        h = f"  {'Year':<6}"
        for sn in snames: h += f" {sn:>14}"
        h += f" {'Nifty':>10}"
        print(h); print(f"  {'-' * 80}")

        all_years = sorted(set().union(*[r.index for r in yr_rets.values() if r is not None]))
        for y in all_years:
            line = f"  {y:<6}"
            for sn in snames:
                v = yr_rets.get(sn, pd.Series()).get(y, None)
                if v is not None and not pd.isna(v): line += f" {v:>+13.1f}%"
                else: line += f" {'---':>14}"
            nv = yr_rets.get('Nifty', pd.Series()).get(y, None)
            if nv is not None and not pd.isna(nv): line += f" {nv:>+9.1f}%"
            else: line += f" {'---':>10}"
            print(line)

    # Verdict
    print(f"\n\n{'=' * 130}")
    print("VERDICT: WHAT IS KILLING OptV3.1?")
    print("=" * 130)

    if len(all_summary) >= 5:
        lr = all_summary[-1]  # longest horizon
        h = lr['Horizon']
        raw31 = lr.get('RawComp-V31_CAGR%', 0)
        raw21 = lr.get('RawComp-V21_CAGR%', 0)
        opt31 = lr.get('OptV3.1-GF_CAGR%', 0)
        leg21 = lr.get('LegV2.1_CAGR%', 0)

        gf_cost = raw31 - opt31 if isinstance(raw31, (int, float)) and isinstance(opt31, (int, float)) else 0
        regime_cost = raw21 - raw31 if isinstance(raw21, (int, float)) and isinstance(raw31, (int, float)) else 0

        print(f"\n  At {h}:")
        print(f"    RawComp-V31 (no G-Factor)           : {raw31:>+.1f}% CAGR")
        print(f"    RawComp-V21 (no G-Factor, no regime) : {raw21:>+.1f}% CAGR")
        print(f"    OptV3.1-GF  (with G-Factor)          : {opt31:>+.1f}% CAGR")
        print(f"    LegV2.1     (63d RS benchmark)        : {leg21:>+.1f}% CAGR")
        print(f"\n    G-Factor Filter cost     : {gf_cost:+.1f}% CAGR (RawComp-V31 minus OptV3.1-GF)")
        print(f"    Regime Sizing cost       : {regime_cost:+.1f}% CAGR (RawComp-V21 minus RawComp-V31)")
        print(f"    Combined cost            : {gf_cost + regime_cost:+.1f}% CAGR")

        if abs(gf_cost) > abs(regime_cost):
            print(f"\n    --> G-FACTOR is the bigger drag ({gf_cost:+.1f}% vs {regime_cost:+.1f}%)")
        else:
            print(f"\n    --> REGIME SIZING is the bigger drag ({regime_cost:+.1f}% vs {gf_cost:+.1f}%)")

    print(f"\n  Files saved to {OUTPUT_DIR}/ablation_*.csv")
    print("=" * 130)


if __name__ == "__main__":
    print("=" * 130)
    print("DNA3 ABLATION TEST: What's Killing OptV3.1?")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 130)
    run()
