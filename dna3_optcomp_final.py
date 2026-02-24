"""
FINAL CONFIGURATION LOCKDOWN: OptComp-V21 (13d) vs LegV2.1
============================================================
The definitive head-to-head to confirm the production config.

  OptComp-V21: Composite RS (10% 1W, 50% 1M, 40% 3M), 13-day rebalance, no filters
  LegV2.1:     63-day RS, 10-day rebalance, no filters (the benchmark)
"""
from dna3_ultimate_comparison import (
    Engine, pain_metrics, fetch_data, INITIAL_CAPITAL, OUTPUT_DIR, HORIZONS
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

STRATEGIES = {
    'OptComp-V21': {
        'use_composite': True, 'weights': (0.1, 0.5, 0.4),
        'g_mode': 'Off', 'rebalance': 13, 'stype': 'v21',
    },
    'LegV2.1': {
        'use_composite': False, 'weights': (0, 0, 1),
        'g_mode': 'Off', 'rebalance': 10, 'stype': 'v21',
    },
}

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nifty, dc = fetch_data()
    if nifty is None: return

    now = datetime.now()
    snames = list(STRATEGIES.keys())

    print("\n" + "=" * 100)
    print("FINAL CONFIGURATION: OptComp-V21 (13d) vs LegV2.1")
    print("=" * 100)
    print("  OptComp-V21 = Composite RS (10/50/40), 13-day rebalance, no filters")
    print("  LegV2.1     = 63-day RS, 10-day rebalance (benchmark)")

    all_rows = []
    longest_eq = {}

    for hname, years in HORIZONS.items():
        s_dt = now - timedelta(days=int(365.25 * years))
        si = nifty.index.searchsorted(s_dt)
        if si >= len(nifty) - 10: continue

        actual_start = nifty.index[si]
        actual_end = nifty.index[-1]
        actual_years = (actual_end - actual_start).days / 365.25

        ns = nifty.iloc[si]['Close']; ne = nifty.iloc[-1]['Close']
        n_cagr = ((ne/ns)**(1/actual_years) - 1)*100

        print(f"\n{'_' * 100}")
        print(f"  {hname.upper()} ({actual_start.date()} -> {actual_end.date()}, {actual_years:.1f}y)")
        print(f"{'_' * 100}")
        print(f"  {'Metric':<22} {'OptComp-V21':>14} {'LegV2.1':>14} {'Delta':>10} {'Nifty':>10}")
        print(f"  {'-' * 75}")

        strat_m = {}
        for sn in snames:
            eng = Engine(sn, STRATEGIES[sn])
            eq = eng.run(dc, nifty, actual_start, actual_end)
            m = pain_metrics(eq, eng.trades, actual_years) if eq is not None else {}
            strat_m[sn] = m
            if hname == max(HORIZONS.keys(), key=lambda k: HORIZONS[k]):
                longest_eq[sn] = eq

        metrics = ['CAGR%', 'MaxDD%', 'DD_Days', 'Sharpe', 'Sortino',
                   'Trades', 'WinRate%', 'Expect%', 'Whipsaw%',
                   'MaxWin%', 'MaxLoss%', 'MaxConsecLoss', 'AvgHold',
                   'BestYear%', 'WorstYear%', 'FlatYears']

        for metric in metrics:
            o = strat_m['OptComp-V21'].get(metric, '-')
            l = strat_m['LegV2.1'].get(metric, '-')
            d = ''
            if isinstance(o, (int, float)) and isinstance(l, (int, float)):
                diff = o - l
                # For MaxDD, DD_Days, Whipsaw, MaxConsecLoss, MaxLoss - lower is better
                if metric in ['MaxDD%', 'DD_Days', 'Whipsaw%', 'MaxConsecLoss', 'MaxLoss%', 'FlatYears', 'WorstYear%']:
                    winner = '✓' if diff > 0 else '✗' if diff < 0 else ''
                else:
                    winner = '✓' if diff > 0 else '✗' if diff < 0 else ''
                d = f"{diff:+.1f}" if isinstance(diff, float) else f"{diff:+d}"
            print(f"  {metric:<22} {str(o):>14} {str(l):>14} {d:>10}")

        row = {'Horizon': hname}
        for sn in snames:
            for k, v in strat_m[sn].items(): row[f'{sn}_{k}'] = v
        row['Nifty_CAGR%'] = round(n_cagr, 2)
        all_rows.append(row)

        oc = strat_m['OptComp-V21'].get('CAGR%', 0)
        lc = strat_m['LegV2.1'].get('CAGR%', 0)
        winner = 'OptComp-V21' if oc > lc else 'LegV2.1'
        print(f"\n  >>> {hname} WINNER: {winner} ({max(oc, lc):+.1f}% CAGR, delta: {oc-lc:+.1f}%)")

    # Calendar year returns
    if longest_eq:
        print(f"\n\n{'=' * 100}")
        print("CALENDAR YEAR RETURNS")
        print(f"{'=' * 100}")
        yr_rets = {}
        for sn in longest_eq:
            eq = longest_eq[sn]
            if eq is None or len(eq) < 22: continue
            eq['y'] = eq['date'].dt.year
            yg = eq.groupby('y')['equity'].last()
            yr_rets[sn] = yg.pct_change() * 100

        si2 = nifty.index.searchsorted(now - timedelta(days=int(365.25 * 15)))
        ne2 = nifty.iloc[si2:].copy(); ne2['y'] = ne2.index.year
        yr_rets['Nifty'] = ne2.groupby('y')['Close'].last().pct_change() * 100

        print(f"  {'Year':<6} {'OptComp-V21':>14} {'LegV2.1':>14} {'Delta':>10} {'Winner':>10} {'Nifty':>10}")
        print(f"  {'-' * 70}")

        all_years = sorted(set().union(*[r.index for r in yr_rets.values()]))
        opt_wins = 0; leg_wins = 0
        for y in all_years:
            ov = yr_rets.get('OptComp-V21', pd.Series()).get(y, None)
            lv = yr_rets.get('LegV2.1', pd.Series()).get(y, None)
            nv = yr_rets.get('Nifty', pd.Series()).get(y, None)
            if ov is not None and not pd.isna(ov) and lv is not None and not pd.isna(lv):
                d = ov - lv
                w = 'Opt' if ov > lv else 'Leg'
                if ov > lv: opt_wins += 1
                else: leg_wins += 1
                nstr = f"{nv:>+9.1f}%" if nv is not None and not pd.isna(nv) else f"{'---':>10}"
                print(f"  {y:<6} {ov:>+13.1f}% {lv:>+13.1f}% {d:>+9.1f}% {w:>10} {nstr}")
            else:
                print(f"  {y:<6} {'---':>14} {'---':>14}")

        print(f"\n  OptComp-V21 wins {opt_wins}/{opt_wins+leg_wins} years ({opt_wins/(opt_wins+leg_wins)*100:.0f}%)")

    # Final Config
    print(f"\n\n{'=' * 100}")
    print("PRODUCTION CONFIGURATION LOCKED")
    print(f"{'=' * 100}")
    print(f"  Strategy Name : OptComp-V21")
    print(f"  RS Signal     : Composite (10% 1W, 50% 1M, 40% 3M)")
    print(f"  Rebalance     : 13 trading days")
    print(f"  G-Factor      : Off")
    print(f"  Regime Sizing : Off (full deployment)")
    print(f"  Position Size : Equal-weight, 10 positions")
    print(f"{'=' * 100}")

    pd.DataFrame(all_rows).to_csv(f"{OUTPUT_DIR}/optcomp_v21_final.csv", index=False)
    print(f"\n  Results saved to {OUTPUT_DIR}/optcomp_v21_final.csv")


if __name__ == "__main__":
    print("=" * 100)
    print("DNA3 FINAL CONFIG LOCKDOWN")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 100)
    run()
