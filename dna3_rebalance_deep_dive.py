"""
DETAILED REBALANCE PAIN METRICS + REGIME DETECTION ANALYSIS
=============================================================
PART A: Full pain metrics for every rebalance cadence (7-20 days)
PART B: Breadth indicator analysis for 2025 regime detection

Usage:
  python dna3_rebalance_deep_dive.py
"""
from dna3_ultimate_comparison import (
    Engine, pain_metrics, fetch_data, detect_regime, tf_rs,
    INITIAL_CAPITAL, OUTPUT_DIR
)
from utils.nifty500_list import TICKERS, SECTOR_MAP
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ============================================================
# PART A: REBALANCE CADENCE — FULL PAIN METRICS
# ============================================================
def rebalance_deep_dive(nifty, dc):
    print("\n" + "=" * 140)
    print("PART A: REBALANCE CADENCE — FULL BEHAVIORAL PAIN METRICS (15Y)")
    print("=" * 140)

    now = datetime.now()
    cadences = [7, 10, 12, 13, 14, 15, 18, 20]
    cfg_base = {
        'use_composite': True, 'weights': (0.1, 0.5, 0.4),
        'g_mode': 'Off', 'stype': 'v21',
    }

    # Run all cadences over 15Y
    s_dt = now - timedelta(days=int(365.25 * 15))
    si = nifty.index.searchsorted(s_dt)
    actual_start = nifty.index[si]
    actual_end = nifty.index[-1]
    years = (actual_end - actual_start).days / 365.25

    results = []
    for reb in cadences:
        cfg = {**cfg_base, 'rebalance': reb}
        eng = Engine(f'Reb{reb}', cfg)
        eq = eng.run(dc, nifty, actual_start, actual_end)
        m = pain_metrics(eq, eng.trades, years) if eq is not None else {}
        m['Rebalance'] = reb
        results.append(m)

    df = pd.DataFrame(results)
    df = df.set_index('Rebalance')

    # Print comprehensive table
    metrics_display = [
        ('CAGR%', 'CAGR (%)'),
        ('Sharpe', 'Sharpe'),
        ('Sortino', 'Sortino'),
        ('MaxDD%', 'Max Drawdown (%)'),
        ('DD_Days', 'Max DD Duration (days)'),
        ('Trades', 'Total Trades'),
        ('WinRate%', 'Win Rate (%)'),
        ('Expect%', 'Expectancy (%)'),
        ('Whipsaw%', 'Whipsaw Rate (%)'),
        ('MaxWin%', 'Best Trade (%)'),
        ('MaxLoss%', 'Worst Trade (%)'),
        ('MaxConsecLoss', 'Max Losing Streak'),
        ('AvgHold', 'Avg Hold (days)'),
        ('BestYear%', 'Best Year (%)'),
        ('WorstYear%', 'Worst Year (%)'),
        ('FlatYears', 'Flat/Negative Years'),
        ('Euphoria%', 'Euphoria Mo. (%)'),
    ]

    # Header
    h = f"  {'Metric':<28}"
    for reb in cadences: h += f" {reb:>6}d"
    print(h)
    print(f"  {'-' * (28 + 8 * len(cadences))}")

    for key, label in metrics_display:
        line = f"  {label:<28}"
        vals = []
        for reb in cadences:
            v = df.loc[reb].get(key, '-')
            vals.append(v)
            if isinstance(v, (int, float)):
                line += f" {v:>6}" if isinstance(v, int) else f" {v:>6.1f}"
            else:
                line += f" {'---':>6}"

        # Mark the best
        print(line)

    # Save
    df.to_csv(f"{OUTPUT_DIR}/rebalance_deep_dive.csv")

    # Stability score: rank each cadence across key metrics, sum ranks
    print(f"\n\n  STABILITY RANKING (lower = better across all metrics)")
    print(f"  {'Reb':<5} {'CAGR':>6} {'Sharpe':>6} {'MaxDD':>6} {'DDDays':>6} {'Expect':>6} {'Whip':>6} {'LStrk':>6} {'Total':>6}")
    print(f"  {'-' * 55}")

    rank_metrics = ['CAGR%', 'Sharpe', 'MaxDD%', 'DD_Days', 'Expect%', 'Whipsaw%', 'MaxConsecLoss']
    # For each metric, rank cadences (1=best)
    ranks = {}
    for reb in cadences: ranks[reb] = []
    for key in rank_metrics:
        vals = [(reb, df.loc[reb].get(key, 0)) for reb in cadences]
        # Higher is better for CAGR, Sharpe, Expect; lower is better for DD, DDDays, Whipsaw, LStrk
        if key in ['MaxDD%']:  # less negative = better
            vals.sort(key=lambda x: x[1], reverse=True)
        elif key in ['DD_Days', 'Whipsaw%', 'MaxConsecLoss']:
            vals.sort(key=lambda x: x[1])
        else:  # CAGR, Sharpe, Expect — higher is better
            vals.sort(key=lambda x: x[1], reverse=True)
        for rank, (reb, v) in enumerate(vals, 1):
            ranks[reb].append(rank)

    for reb in cadences:
        r = ranks[reb]
        total = sum(r)
        line = f"  {reb:<5}"
        for rv in r: line += f" {rv:>6}"
        line += f" {total:>6}"
        print(line)

    best_reb = min(ranks.keys(), key=lambda k: sum(ranks[k]))
    print(f"\n  >>> MOST STABLE CADENCE: {best_reb} days (total rank score: {sum(ranks[best_reb])})")


# ============================================================
# PART B: BREADTH INDICATOR ANALYSIS
# ============================================================
def breadth_analysis(nifty, dc):
    print("\n\n" + "=" * 140)
    print("PART B: MARKET BREADTH ANALYSIS — COULD IT HAVE PREDICTED 2025?")
    print("=" * 140)

    # Calculate breadth indicators over time
    # For each trading day, compute:
    # 1. % of stocks above their 50-day MA
    # 2. % of stocks above their 200-day MA
    # 3. % of stocks with positive 63d RS vs Nifty
    # 4. Advance/Decline ratio (% positive daily returns)

    # Focus on 2024-2025 transition
    analysis_start = pd.Timestamp('2024-01-01')
    analysis_end = nifty.index[-1]

    si = nifty.index.searchsorted(analysis_start)
    dates = nifty.index[si:]

    # Sample every 5th trading day to keep computation reasonable
    sample_dates = dates[::5]

    breadth_data = []
    print(f"\n  Computing breadth indicators for {len(sample_dates)} sample dates...")

    for d in sample_dates:
        above_50 = 0; above_200 = 0; pos_rs = 0; total = 0; pos_daily = 0

        ni = nifty.index.searchsorted(d)
        nw = nifty.iloc[max(0, ni-252):ni+1]

        for t, df in dc.items():
            if t == 'NIFTY': continue
            i = df.index.searchsorted(d)
            if i < 200: continue

            try:
                p = df['Close'].iloc[i]
                ma50 = df['Close'].iloc[max(0,i-50):i+1].mean()
                ma200 = df['Close'].iloc[max(0,i-200):i+1].mean()
                rs63 = tf_rs(df.iloc[max(0,i-100):i+1], nw, 63)
                daily_ret = (p / df['Close'].iloc[i-1] - 1) if i > 0 else 0

                total += 1
                if p > ma50: above_50 += 1
                if p > ma200: above_200 += 1
                if rs63 > 0: pos_rs += 1
                if daily_ret > 0: pos_daily += 1
            except: pass

        if total > 50:
            breadth_data.append({
                'Date': d,
                'PctAbove50': round(above_50/total*100, 1),
                'PctAbove200': round(above_200/total*100, 1),
                'PctPosRS': round(pos_rs/total*100, 1),
                'ADRatio': round(pos_daily/total*100, 1),
                'Nifty': nifty.iloc[ni]['Close'],
            })

    bdf = pd.DataFrame(breadth_data)
    bdf.to_csv(f"{OUTPUT_DIR}/breadth_indicators.csv", index=False)

    # Print key dates
    print(f"\n  BREADTH INDICATORS OVER TIME:")
    print(f"  {'Date':<12} {'%>50DMA':>8} {'%>200DMA':>9} {'%+veRS':>7} {'A/D%':>6} {'Nifty':>10}")
    print(f"  {'-' * 60}")

    # Show monthly snapshots
    bdf['Month'] = bdf['Date'].dt.to_period('M')
    monthly = bdf.groupby('Month').last().reset_index()
    for _, r in monthly.iterrows():
        nifty_val = r.get('Nifty', 0)
        print(f"  {str(r['Month']):<12} {r['PctAbove50']:>7.1f}% {r['PctAbove200']:>8.1f}% {r['PctPosRS']:>6.1f}% {r['ADRatio']:>5.1f}% {nifty_val:>10.0f}")

    # Detect divergence: Nifty making new highs while breadth is falling
    if len(monthly) > 6:
        print(f"\n  DIVERGENCE DETECTION:")
        # Compare Sept 2024 peak breadth vs Dec 2024
        peak_idx = monthly['PctAbove50'].idxmax()
        peak = monthly.iloc[peak_idx]
        last = monthly.iloc[-1]

        print(f"    Peak breadth (%>50DMA): {peak['PctAbove50']:.0f}% on {peak['Month']}")
        print(f"    Latest:                 {last['PctAbove50']:.0f}% on {last['Month']}")
        drop = peak['PctAbove50'] - last['PctAbove50']
        print(f"    Breadth collapse:       {drop:+.0f} percentage points")

        # Was Nifty higher or lower?
        nifty_chg = (last['Nifty'] - peak['Nifty']) / peak['Nifty'] * 100
        print(f"    Nifty change (peak -> now): {nifty_chg:+.1f}%")

        if drop > 20 and nifty_chg > -5:
            print(f"\n    --> BEARISH DIVERGENCE DETECTED: Breadth collapsed {drop:.0f}% while Nifty only fell {nifty_chg:.1f}%")
            print(f"        This is the 'narrow market' signal that could have warned about 2025.")
        elif drop > 20:
            print(f"\n    --> BREADTH COLLAPSE: Both breadth and Nifty fell, confirming broad weakness.")

    # When breadth < 30% above 50DMA, what happens to momentum strategies?
    print(f"\n  BREADTH AS MOMENTUM KILL SWITCH:")
    low_breadth = bdf[bdf['PctAbove50'] < 30]
    high_breadth = bdf[bdf['PctAbove50'] > 60]
    print(f"    Days with <30% above 50DMA: {len(low_breadth)} ({len(low_breadth)/len(bdf)*100:.0f}% of period)")
    print(f"    Days with >60% above 50DMA: {len(high_breadth)} ({len(high_breadth)/len(bdf)*100:.0f}% of period)")
    print(f"    Avg %+veRS when breadth <30%: {low_breadth['PctPosRS'].mean():.1f}%")
    print(f"    Avg %+veRS when breadth >60%: {high_breadth['PctPosRS'].mean():.1f}%")


# ============================================================
# MAIN
# ============================================================
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nifty, dc = fetch_data()
    if nifty is None or nifty.empty: return

    rebalance_deep_dive(nifty, dc)
    breadth_analysis(nifty, dc)

    print("\n\n" + "=" * 140)
    print("ALL ANALYSES COMPLETE")
    print(f"Files saved to {OUTPUT_DIR}/")
    print("=" * 140)


if __name__ == "__main__":
    print("=" * 140)
    print("DNA3 REBALANCE DEEP DIVE + BREADTH ANALYSIS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 140)
    run()
