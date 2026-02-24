"""
MARKET MOOD AS LEADING BEAR INDICATOR — 15-YEAR HISTORICAL TEST
================================================================
Reconstructs the 4 market_mood indicators across 15 years and tests
whether they predicted EVERY major bear/correction period.

Mood Indicators (from market_mood.py):
  1. % Strong Momentum   (trend_score >= 80)
  2. % Total Uptrends    (trend_signal contains 'UPTREND')
  3. Avg Trend Score     (mean of all trend scores)
  4. % Breakout Alerts   (within 5% of 52-week high)

Bear Periods to Test:
  - 2011 European Debt / India Slowdown
  - 2015-16 China Deval / Commodity Crash
  - 2018 IL&FS / NBFC Crisis
  - 2020 COVID Crash
  - 2022 Rate Hike Selloff
  - 2024-25 Narrow Market Crash

For each bear, we check: Did any mood indicator LEAD the crash?
"""
from dna3_ultimate_comparison import fetch_data
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

OUTPUT_DIR = "analysis_2026"

# Known bear/correction periods (approximate start dates and bottoms)
BEAR_PERIODS = [
    {'name': '2011 EU Debt Crisis', 'pre': '2011-07-01', 'bottom': '2011-12-20', 'recovery': '2012-03-01'},
    {'name': '2013 Taper Tantrum', 'pre': '2013-05-01', 'bottom': '2013-08-28', 'recovery': '2013-11-01'},
    {'name': '2015-16 China Deval', 'pre': '2015-12-01', 'bottom': '2016-02-29', 'recovery': '2016-06-01'},
    {'name': '2018 NBFC/IL&FS', 'pre': '2018-08-01', 'bottom': '2018-10-26', 'recovery': '2019-01-01'},
    {'name': '2020 COVID Crash', 'pre': '2020-01-15', 'bottom': '2020-03-23', 'recovery': '2020-06-01'},
    {'name': '2022 Rate Hike', 'pre': '2022-01-01', 'bottom': '2022-06-17', 'recovery': '2022-09-01'},
    {'name': '2024-25 Narrow Mkt', 'pre': '2024-09-01', 'bottom': '2025-03-04', 'recovery': '2025-06-01'},
]


def calculate_trend_score(close, high, low, ma50, ma200, high_52w, low_52w):
    """Vectorized trend score matching trend_engine.py logic."""
    score = 50.0

    # MA alignment
    score += np.where(close > ma50, 15, -10)
    score += np.where(close > ma200, 15, -15)
    score += np.where((ma50 > 0) & (ma200 > 0) & (ma50 > ma200), 10, -5)

    # 52W position
    range_52 = high_52w - low_52w
    range_52 = np.where(range_52 == 0, np.nan, range_52)
    pct_pos = (close - low_52w) / range_52
    pos_score = (pct_pos - 0.5) * 30
    pos_score = np.where(np.isnan(pos_score), 0, pos_score)
    score += pos_score

    # Near high bonus
    dist_from_high = (close - high_52w) / np.where(high_52w == 0, 1, high_52w)
    score += np.where(dist_from_high > -0.05, 10, np.where(dist_from_high < -0.30, -10, 0))

    return np.clip(score, 0, 100)


def compute_mood_for_date(dc, nifty, target_date):
    """Compute all 4 mood indicators for a specific date."""
    ni = nifty.index.searchsorted(target_date)
    if ni < 252 or ni >= len(nifty):
        return None

    total = 0
    strong_momentum = 0
    uptrends = 0
    breakouts = 0
    scores = []

    for t, df in dc.items():
        if t == 'NIFTY':
            continue
        i = df.index.searchsorted(target_date)
        if i < 252 or i >= len(df):
            continue

        try:
            close = float(df['Close'].iloc[i])
            ma50 = float(df['Close'].iloc[max(0, i-50):i+1].mean())
            ma200 = float(df['Close'].iloc[max(0, i-200):i+1].mean())
            high_52w = float(df['High'].iloc[max(0, i-252):i+1].max())
            low_52w = float(df['Low'].iloc[max(0, i-252):i+1].min())
            high_val = float(df['High'].iloc[i])

            ts = calculate_trend_score(
                np.array([close]), np.array([high_val]), np.array([close]),
                np.array([ma50]), np.array([ma200]),
                np.array([high_52w]), np.array([low_52w])
            )[0]

            total += 1
            scores.append(ts)
            if ts >= 80:
                strong_momentum += 1
            if ts >= 60:  # UPTREND or STRONG UPTREND
                uptrends += 1
            # Breakout: within 5% of 52W high
            if high_52w > 0 and (close - high_52w) / high_52w >= -0.05:
                breakouts += 1
        except:
            pass

    if total < 50:
        return None

    return {
        'date': target_date,
        'pct_strong_momentum': round(strong_momentum / total * 100, 1),
        'pct_uptrends': round(uptrends / total * 100, 1),
        'avg_trend_score': round(np.mean(scores), 1),
        'pct_breakouts': round(breakouts / total * 100, 1),
        'nifty': float(nifty.iloc[ni]['Close']),
        'total_stocks': total,
    }


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nifty, dc = fetch_data()
    if nifty is None:
        return

    print("\n" + "=" * 130)
    print("MARKET MOOD AS LEADING BEAR INDICATOR — 15-YEAR HISTORICAL TEST")
    print("=" * 130)
    print("  Mood indicators: % Strong Momentum, % Uptrends, Avg Trend Score, % Breakouts")
    print("  Testing across 7 bear/correction periods from 2011 to 2025")

    # ================================================================
    # 1. Compute mood indicators leading up to and during each bear
    # ================================================================
    print("\n" + "=" * 130)
    print("BEAR PERIOD ANALYSIS")
    print("=" * 130)

    indicator_signals = []

    for bear in BEAR_PERIODS:
        name = bear['name']
        pre = pd.Timestamp(bear['pre'])
        bottom = pd.Timestamp(bear['bottom'])

        # Sample 5 dates: 6mo before, 3mo before, at pre-crash, at midpoint, at bottom
        dates = [
            ('6mo Before', pre - timedelta(days=180)),
            ('3mo Before', pre - timedelta(days=90)),
            ('Pre-Crash', pre),
            ('Midpoint', pre + (bottom - pre) / 2),
            ('Bottom', bottom),
        ]

        print(f"\n  {'-' * 120}")
        print(f"  {name}")
        print(f"  {'-' * 120}")
        print(f"  {'Phase':<14} {'Date':<12} {'%Strong':>8} {'%Uptrend':>9} {'AvgScore':>9} {'%Brkout':>8} {'Nifty':>8}")
        print(f"  {'-' * 75}")

        pre_values = {}
        for label, d in dates:
            # Find nearest trading day
            idx = nifty.index.searchsorted(d)
            if idx >= len(nifty):
                idx = len(nifty) - 1
            actual_d = nifty.index[idx]

            m = compute_mood_for_date(dc, nifty, actual_d)
            if m is None:
                print(f"  {label:<14} {str(actual_d.date()):<12} {'---':>8} {'---':>9} {'---':>9} {'---':>8} {'---':>8}")
                continue

            print(f"  {label:<14} {str(actual_d.date()):<12} {m['pct_strong_momentum']:>7.1f}% {m['pct_uptrends']:>8.1f}% {m['avg_trend_score']:>8.1f} {m['pct_breakouts']:>7.1f}% {m['nifty']:>8.0f}")

            if label == '6mo Before':
                pre_values = m.copy()
            elif label == 'Pre-Crash' and pre_values:
                # Check for divergence/deterioration
                sm_drop = pre_values['pct_strong_momentum'] - m['pct_strong_momentum']
                up_drop = pre_values['pct_uptrends'] - m['pct_uptrends']
                sc_drop = pre_values['avg_trend_score'] - m['avg_trend_score']
                br_drop = pre_values['pct_breakouts'] - m['pct_breakouts']

                signals = []
                if sm_drop > 10: signals.append(f"%StrongMom dropped {sm_drop:.0f}pts")
                if up_drop > 10: signals.append(f"%Uptrends dropped {up_drop:.0f}pts")
                if sc_drop > 5: signals.append(f"AvgScore dropped {sc_drop:.0f}pts")
                if br_drop > 10: signals.append(f"%Breakouts dropped {br_drop:.0f}pts")

                if m['avg_trend_score'] < 45:
                    signals.append(f"AvgScore BELOW 45 ({m['avg_trend_score']:.0f})")
                if m['pct_uptrends'] < 40:
                    signals.append(f"%Uptrends BELOW 40% ({m['pct_uptrends']:.0f}%)")

                indicator_signals.append({
                    'bear': name,
                    'signals': signals,
                    'sm_at_pre': m['pct_strong_momentum'],
                    'up_at_pre': m['pct_uptrends'],
                    'sc_at_pre': m['avg_trend_score'],
                    'br_at_pre': m['pct_breakouts'],
                    'sm_drop': sm_drop,
                    'up_drop': up_drop,
                    'sc_drop': sc_drop,
                    'br_drop': br_drop,
                })

                if signals:
                    print(f"     WARNINGS: {' | '.join(signals)}")
                else:
                    print(f"     NO EARLY WARNING detected from mood indicators")

    # ================================================================
    # 2. Summary: Which indicators led which bears?
    # ================================================================
    print(f"\n\n{'=' * 130}")
    print("SCORECARD: DID MOOD INDICATORS PREDICT EACH BEAR?")
    print(f"{'=' * 130}")

    print(f"\n  {'Bear Period':<25} {'%Strong':>8} {'%Uptrend':>9} {'AvgScore':>9} {'%Brkout':>8} {'#Signals':>9}")
    print(f"  {'-' * 70}")

    total_caught = 0
    for s in indicator_signals:
        sm = 'Y' if s['sm_drop'] > 10 or s['sm_at_pre'] < 15 else 'N'
        up = 'Y' if s['up_drop'] > 10 or s['up_at_pre'] < 40 else 'N'
        sc = 'Y' if s['sc_drop'] > 5 or s['sc_at_pre'] < 45 else 'N'
        br = 'Y' if s['br_drop'] > 10 or s['br_at_pre'] < 15 else 'N'
        n = len(s['signals'])
        if n > 0: total_caught += 1
        print(f"  {s['bear']:<25} {sm:>8} {up:>9} {sc:>9} {br:>8} {n:>9}")

    print(f"\n  Bears caught: {total_caught}/{len(indicator_signals)}")

    # ================================================================
    # 3. Optimal Thresholds
    # ================================================================
    print(f"\n\n{'=' * 130}")
    print("OPTIMAL THRESHOLD ANALYSIS")
    print(f"{'=' * 130}")

    # Compute mood at monthly intervals over 15Y
    print(f"\n  Computing monthly mood snapshots over 15 years...")
    analysis_start = pd.Timestamp('2011-06-01')
    dates_monthly = pd.date_range(analysis_start, nifty.index[-1], freq='ME')

    monthly_mood = []
    for d in dates_monthly:
        idx = nifty.index.searchsorted(d)
        if idx >= len(nifty): idx = len(nifty) - 1
        actual_d = nifty.index[idx]
        m = compute_mood_for_date(dc, nifty, actual_d)
        if m:
            # Forward 3-month Nifty return
            fi = min(idx + 63, len(nifty) - 1)
            fwd_ret = (nifty.iloc[fi]['Close'] / nifty.iloc[idx]['Close'] - 1) * 100
            m['fwd_3m_nifty'] = round(fwd_ret, 1)
            monthly_mood.append(m)

    mdf = pd.DataFrame(monthly_mood)
    mdf.to_csv(f"{OUTPUT_DIR}/mood_historical_15y.csv", index=False)

    # Test thresholds
    print(f"\n  WHEN MOOD INDICATORS SIGNAL DANGER, WHAT HAPPENS NEXT?")
    print(f"  (Forward 3-month Nifty returns after indicator crosses threshold)\n")

    thresholds = [
        ('avg_trend_score', '<', 45, 'Avg Trend Score < 45'),
        ('avg_trend_score', '<', 40, 'Avg Trend Score < 40'),
        ('pct_uptrends', '<', 40, '% Uptrends < 40%'),
        ('pct_uptrends', '<', 30, '% Uptrends < 30%'),
        ('pct_strong_momentum', '<', 15, '% Strong Momentum < 15%'),
        ('pct_strong_momentum', '<', 10, '% Strong Momentum < 10%'),
        ('pct_breakouts', '<', 15, '% Breakouts < 15%'),
        ('pct_breakouts', '<', 10, '% Breakouts < 10%'),
    ]

    print(f"  {'Threshold':<30} {'Triggers':>8} {'Avg Fwd 3M':>12} {'Median':>8} {'%Negative':>10}")
    print(f"  {'-' * 72}")

    for col, op, val, label in thresholds:
        if op == '<':
            mask = mdf[col] < val
        else:
            mask = mdf[col] > val

        triggered = mdf[mask]
        n = len(triggered)
        if n == 0:
            print(f"  {label:<30} {0:>8}")
            continue

        avg_fwd = triggered['fwd_3m_nifty'].mean()
        med_fwd = triggered['fwd_3m_nifty'].median()
        pct_neg = (triggered['fwd_3m_nifty'] < 0).sum() / n * 100

        print(f"  {label:<30} {n:>8} {avg_fwd:>+11.1f}% {med_fwd:>+7.1f}% {pct_neg:>9.0f}%")

    # And when indicators are healthy
    print(f"\n  BASELINE (healthy market):")
    healthy = mdf[mdf['avg_trend_score'] >= 55]
    if len(healthy) > 0:
        print(f"  {'Avg Score >= 55':<30} {len(healthy):>8} {healthy['fwd_3m_nifty'].mean():>+11.1f}% {healthy['fwd_3m_nifty'].median():>+7.1f}% {(healthy['fwd_3m_nifty'] < 0).sum()/len(healthy)*100:>9.0f}%")

    print(f"\n  Files saved to {OUTPUT_DIR}/mood_historical_15y.csv")
    print(f"{'=' * 130}")


if __name__ == "__main__":
    print("=" * 130)
    print("MARKET MOOD LEADING INDICATOR TEST")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 130)
    run()
