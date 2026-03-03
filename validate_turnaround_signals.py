"""
validate_turnaround_signals.py
==============================
Post-mortem analysis: did the proposed turnaround signals fire EARLY on stocks
that subsequently went 2x+ in a short period?

Test cohort (Nifty 1000 stocks that doubled+ in 2024-2025):
  STLTECH.NS   - Sterlite Tech: Rs86 -> Rs163 (+90%, Jan-Feb 2026)
  TITAGARH.NS  - Titagarh Rail: Rs500 -> Rs1100 (+120%, Jun-Sep 2024) 
  GRSE.NS      - Garden Reach: Rs900 -> Rs2200 (+145%, Apr-Jul 2024)
  COCHINSHIP.NS- Cochin Shipyard: Rs1200 -> Rs2700 (+125%, Apr-Aug 2024)
  JSWENERGY.NS - JSW Energy: Rs370 -> Rs750 (+103%, Apr-Aug 2024)
  GPIL.NS      - Gallantt Ispat: Rs250 -> Rs750 (+200%, Jun-Aug 2024)

Output: For each stock, shows:
  - The date the proposed signal would have FIRST fired
  - Entry price at signal
  - Max gain available from signal date to peak
  - Comparison: signal date vs stock's actual doubling timeline
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ── Test cohort: stocks that went 2x+ ─────────────────────────────────────────
TEST_STOCKS = {
    # Original cohort
    "STLTECH.NS":    {"name": "Sterlite Tech",       "approx_low": 62,   "approx_peak": 165},
    "TITAGARH.NS":   {"name": "Titagarh Rail",       "approx_low": 500,  "approx_peak": 1150},
    "GRSE.NS":       {"name": "Garden Reach Ship",   "approx_low": 760,  "approx_peak": 2300},
    "COCHINSHIP.NS": {"name": "Cochin Shipyard",     "approx_low": 860,  "approx_peak": 2750},
    "JSWENERGY.NS":  {"name": "JSW Energy",          "approx_low": 330,  "approx_peak": 750},
    "GPIL.NS":       {"name": "Gallantt Ispat",      "approx_low": 230,  "approx_peak": 750},
    # Extended cohort — diverse sectors, all 2x+ within ~12 months
    "MAZDOCK.NS":    {"name": "Mazagon Dock",        "approx_low": 1500, "approx_peak": 5100},  # Defense, +240%
    "IRFC.NS":       {"name": "Indian Railway FC",   "approx_low": 95,   "approx_peak": 230},   # NBFC-Infra, +142%
    "BEL.NS":        {"name": "Bharat Electronics",  "approx_low": 140,  "approx_peak": 340},   # Defense, +143%
    "KAYNES.NS":     {"name": "Kaynes Technology",   "approx_low": 1800, "approx_peak": 5800},  # Electronics, +222%
    "APARINDS.NS":   {"name": "Apar Industries",     "approx_low": 4000, "approx_peak": 10800}, # Cables, +170%
    "PNBHOUSING.NS": {"name": "PNB Housing Finance", "approx_low": 500,  "approx_peak": 1250},  # HFC, +150%
}

# ── Signal thresholds (from proposed turnaround module) ───────────────────────
MIN_LIQ_CR       = 50.0    # Min ₹ Cr/day — basic liquidity gate
DEEP_DRAWDOWN    = 0.65    # ≤ 65% of 52W high = at least 35% off peak
IAS_PASS_SCORE   = 40      # Minimum IAS to appear in watchlist
IMMINENT_SCORE   = 70      # IMMINENT tier threshold (lowered from 80 for generalization)


def calc_rs(stock_close: pd.Series, bench_close: pd.Series, window: int) -> pd.Series:
    """RS vs benchmark over rolling window (%)"""
    stock_ret  = stock_close.pct_change(window) * 100
    bench_ret  = bench_close.pct_change(window) * 100
    return stock_ret - bench_ret


def calc_daily_liq(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Daily traded value in ₹ Cr"""
    return (close * volume) / 1e7   # 1 Cr = 1e7


def calc_ias_score(row: pd.Series) -> float:
    """
    Turnaround IAS Score (0-100).
    Generalized — doesn't overfit to STLTECH.

    Components:
      RS Velocity  (35): How fast is RS21 improving? (5-day delta of RS21)
      Liq-from-Low (30): liq5 / liq10_min — recovering from trough, not vs 63d avg
      Price Recovery (20): How far has stock bounced from 52W low?
      RS63 Recovery (15): Is 3-month RS still very negative? Penalise if yes.
    """
    # 1. RS Velocity (35 pts)
    rs_vel = row.get("rs21_delta5", 0)
    vel_score = min(35, max(0, rs_vel / 5.0 * 35))   # +5% RS improvement = full 35 pts

    # 2. Liq From Low (30 pts)
    lfl = row.get("liq_from_low", 1.0)
    lfl_score = min(30, max(0, (lfl - 1.0) * 15))   # 3x recovery from trough = full 30 pts

    # 3. Price Recovery from 52W Low (20 pts)
    pct_off_lo = row.get("pct_off_low", 0)
    price_score = min(20, pct_off_lo / 15.0 * 20)   # 15% above 52W low = full 20 pts

    # 4. RS63 Recovery Bonus (15 pts)
    # Still negative RS63 is normal for turnarounds — we score *direction*, not level
    rs63 = row.get("rs63", -999)
    if rs63 > 0:
        r63_score = 15                  # Fully positive = full bonus
    elif rs63 > -5:
        r63_score = 10                  # Nearly there
    elif rs63 > -15:
        r63_score = 5                    # Improving but still negative
    else:
        r63_score = 0                    # Very negative still

    return round(vel_score + lfl_score + price_score + r63_score, 1)


def analyze_stock(ticker: str, meta: dict, nifty: pd.DataFrame) -> dict:
    """Run signal post-mortem on a single stock."""
    print(f"\n  Analyzing {ticker} ({meta['name']})...")
    try:
        raw = yf.Ticker(ticker).history(start="2023-06-01", end="2026-03-01")
        if raw.empty or len(raw) < 200:
            print(f"    [SKIP] Insufficient data")
            return {}
        if raw.index.tz is not None:
            raw.index = raw.index.tz_localize(None)
    except Exception as e:
        print(f"    [ERR] {e}")
        return {}

    close  = raw["Close"]
    volume = raw["Volume"]
    nifty_close = nifty["Close"].reindex(close.index, method="ffill")

    # Rolling calculations
    rs21  = calc_rs(close, nifty_close, 21)
    rs63  = calc_rs(close, nifty_close, 63)
    comp_rs = (rs21 * 0.10) + (rs21 * 0.50) + (rs63 * 0.40)   # simplified composite
    comp_rs = (0.10 * calc_rs(close, nifty_close, 5)
             + 0.50 * rs21
             + 0.40 * rs63)

    liq_cr     = calc_daily_liq(close, volume)
    liq5       = liq_cr.rolling(5).mean()
    liq10_min  = liq_cr.rolling(10).min()
    liq_from_low = liq5 / liq10_min.clip(lower=0.01)

    high_252   = close.rolling(252).max()
    low_252    = close.rolling(252).min()
    rs21_delta = rs21.diff(5)   # 5-day change in RS21

    pct_off_high = (close / high_252 - 1) * 100
    pct_off_low  = (close / low_252 - 1)  * 100

    # Build daily DataFrame
    df = pd.DataFrame({
        "close":        close,
        "rs21":         rs21,
        "rs63":         rs63,
        "comp_rs":      comp_rs,
        "liq5":         liq5,
        "liq_from_low": liq_from_low,
        "pct_off_high": pct_off_high,
        "pct_off_low":  pct_off_low,
        "rs21_delta5":  rs21_delta,
    }).dropna()

    # Find actual peak and drawdown trough
    stock_low  = df["close"].min()
    stock_peak = df["close"].max()
    peak_date  = df["close"].idxmax()
    low_date   = df["close"].idxmin()

    # Compute IAS score for each day
    df["ias"] = df.apply(calc_ias_score, axis=1)

    # Filter to days where:
    # 1. Stock is in deep drawdown (≥ 35% off 52W high)
    # 2. Liquidity ≥ ₹50Cr/day
    # 3. IAS ≥ 40
    signals = df[
        (df["pct_off_high"] <= -35) &
        (df["liq5"] >= MIN_LIQ_CR) &
        (df["ias"] >= IAS_PASS_SCORE)
    ]

    imminent = df[
        (df["pct_off_high"] <= -35) &
        (df["liq5"] >= MIN_LIQ_CR) &
        (df["ias"] >= IMMINENT_SCORE)
    ]

    result = {
        "ticker":        ticker,
        "name":          meta["name"],
        "stock_low":     round(stock_low, 1),
        "stock_peak":    round(stock_peak, 1),
        "total_move":    round((stock_peak / stock_low - 1) * 100, 1),
        "low_date":      low_date.strftime("%Y-%m-%d"),
        "peak_date":     peak_date.strftime("%Y-%m-%d"),
    }

    if not signals.empty:
        first_sig     = signals.index[0]
        sig_price     = round(df.loc[first_sig, "close"], 1)
        gain_to_peak  = round((stock_peak / sig_price - 1) * 100, 1)
        days_before_peak = (peak_date - first_sig).days

        result.update({
            "first_signal_date":     first_sig.strftime("%Y-%m-%d"),
            "signal_price":          sig_price,
            "signal_ias":            round(signals.iloc[0]["ias"], 1),
            "gain_signal_to_peak":   gain_to_peak,
            "days_before_peak":      days_before_peak,
            "signal_rs21":           round(signals.iloc[0]["rs21"], 1),
            "signal_rs63":           round(signals.iloc[0]["rs63"], 1),
            "signal_liq5":           round(signals.iloc[0]["liq5"], 1),
            "signal_liq_from_low":   round(signals.iloc[0]["liq_from_low"], 1),
            "days_from_low_to_sig":  (first_sig - low_date).days,
        })
    else:
        result.update({
            "first_signal_date":   "NO_SIGNAL",
            "signal_price":        None,
            "gain_signal_to_peak": None,
        })

    if not imminent.empty:
        fi = imminent.index[0]
        result["imminent_date"]  = fi.strftime("%Y-%m-%d")
        result["imminent_price"] = round(df.loc[fi, "close"], 1)
        result["imminent_gain"]  = round((stock_peak / df.loc[fi, "close"] - 1) * 100, 1)
    else:
        result["imminent_date"]  = "NO_IMMINENT"
        result["imminent_price"] = None
        result["imminent_gain"]  = None

    return result


def main():
    print("=" * 70)
    print("  TURNAROUND SIGNAL POST-MORTEM: 2x+ Stocks Validation")
    print("=" * 70)

    # Load Nifty
    print("\n  Loading Nifty 50 benchmark...")
    nifty = yf.Ticker("^NSEI").history(start="2023-06-01", end="2026-03-01")
    if nifty.index.tz is not None:
        nifty.index = nifty.index.tz_localize(None)

    results = []
    for ticker, meta in TEST_STOCKS.items():
        r = analyze_stock(ticker, meta, nifty)
        if r:
            results.append(r)

    if not results:
        print("\n  No results generated.")
        return

    df = pd.DataFrame(results)

    print("\n\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    for _, row in df.iterrows():
        print(f"\n  {row['ticker']} — {row['name']}")
        print(f"    Range    : Rs{row['stock_low']} -> Rs{row['stock_peak']} (+{row['total_move']}%)")
        print(f"    Low date : {row['low_date']}   |   Peak: {row['peak_date']}")
        print(f"    --- TURNAROUND SIGNAL (IAS >= {IAS_PASS_SCORE}) ---")
        if row['first_signal_date'] != 'NO_SIGNAL':
            print(f"    First sig: {row['first_signal_date']} @ Rs{row['signal_price']} (IAS {row['signal_ias']})")
            print(f"    RS21={row['signal_rs21']}%  RS63={row['signal_rs63']}%  Liq={row['signal_liq5']}Cr  LiqFromLow={row['signal_liq_from_low']}x")
            print(f"    Days from low to signal : {row['days_from_low_to_sig']}")
            print(f"    Days before peak        : {row['days_before_peak']}")
            print(f"    Gain (signal -> peak)    : +{row['gain_signal_to_peak']}%")
        else:
            print(f"    NO SIGNAL fired! (check thresholds)")
        print(f"    --- IMMINENT TIER (IAS >= {IMMINENT_SCORE}) ---")
        if row['imminent_date'] != 'NO_IMMINENT':
            print(f"    Imminent : {row['imminent_date']} @ Rs{row['imminent_price']} (+{row['imminent_gain']}% to peak)")
        else:
            print(f"    No IMMINENT signal.")

    df.to_csv("analysis_2026/turnaround_postmortem.csv", index=False)
    print(f"\n\n  Results saved to analysis_2026/turnaround_postmortem.csv")
    print("  Use these findings to calibrate the IAS score weights before going live.")


if __name__ == "__main__":
    import os; os.makedirs("analysis_2026", exist_ok=True)
    main()
