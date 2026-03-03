"""Diagnose why BEL and PNBHOUSING missed the signal, and validate fixes."""
import pandas as pd
import numpy as np
import yfinance as yf

def calc_rs(sc, bc, w): return sc.pct_change(w)*100 - bc.pct_change(w)*100
def calc_liq(c, v): return (c * v) / 1e7

nifty = yf.Ticker("^NSEI").history(start="2023-06-01", end="2026-03-01")
if nifty.index.tz: nifty.index = nifty.index.tz_localize(None)

stocks = {
    "BEL.NS":        "Bharat Electronics   (missed — WHY?)",
    "PNBHOUSING.NS": "PNB Housing Finance  (missed — WHY?)",
    "MAZDOCK.NS":    "Mazagon Dock         (extreme LflLow check)",
}

for tk, label in stocks.items():
    raw = yf.Ticker(tk).history(start="2023-06-01", end="2026-03-01")
    if raw.empty: continue
    if raw.index.tz: raw.index = raw.index.tz_localize(None)
    c, v = raw["Close"], raw["Volume"]
    nb = nifty["Close"].reindex(c.index, method="ffill")

    liq_raw  = calc_liq(c, v)
    liq5     = liq_raw.rolling(5).mean()
    liq10min = liq_raw.rolling(10).min().clip(lower=0.01)
    lfl_raw  = liq5 / liq10min                   # uncapped
    lfl_cap  = lfl_raw.clip(upper=50)            # capped at 50x for scoring

    rs21     = calc_rs(c, nb, 21)
    rs63     = calc_rs(c, nb, 63)
    h252     = c.rolling(252).max()
    l252     = c.rolling(252).min()
    off_high = (c / h252 - 1) * 100
    off_low  = (c / l252 - 1) * 100
    rs21d5   = rs21.diff(5)

    df = pd.DataFrame({
        "close":    c,
        "rs21":     rs21,
        "rs63":     rs63,
        "liq5":     liq5,
        "lfl_raw":  lfl_raw,
        "lfl_cap":  lfl_cap,
        "off_high": off_high,
        "off_low":  off_low,
        "rs21d5":   rs21d5,
    }).dropna()

    print(f"\n{'='*60}")
    print(f"  {tk} — {label}")
    print(f"{'='*60}")
    print(f"  Price range : {df.close.min():.0f} -> {df.close.max():.0f}  "
          f"(+{(df.close.max()/df.close.min()-1)*100:.0f}%)")
    print(f"  off_high min: {df.off_high.min():.1f}%")
    print(f"  Days with off_high <= -35%: {(df.off_high<=-35).sum()}")
    print(f"  Days with off_high <= -20%: {(df.off_high<=-20).sum()}")
    print(f"  liq5 range  : {df.liq5.min():.1f} - {df.liq5.max():.1f} Cr")
    print(f"  lfl_raw max : {df.lfl_raw.max():.1f}x  |  lfl_cap max: {df.lfl_cap.max():.1f}x")

    # Test with 20% gate + 30Cr + lfl_cap
    for drawdown_gate, liq_gate in [(-35, 50), (-20, 30)]:
        mask = (df.off_high <= drawdown_gate) & (df.liq5 >= liq_gate)
        sub  = df[mask]
        if sub.empty:
            print(f"  [Gate {drawdown_gate}% / Liq{liq_gate}Cr]: 0 days pass — NO SIGNAL POSSIBLE")
            continue

        vel_s = sub.rs21d5.clip(lower=0).div(5).mul(35).clip(upper=35)
        lfl_s = (sub.lfl_cap - 1).mul(15).clip(0, 30)
        pr_s  = sub.off_low.div(15).mul(20).clip(0, 20)
        r63_s = sub.rs63.apply(
            lambda x: 15 if x > 0 else (10 if x > -5 else (5 if x > -15 else 0))
        )
        ias = (vel_s + lfl_s + pr_s + r63_s).round(1)

        max_ias = ias.max()
        passing = ias[ias >= 35]
        print(f"\n  [Gate {drawdown_gate}% / Liq{liq_gate}Cr]:")
        print(f"    Days passing gate   : {len(sub)}")
        print(f"    Max IAS achievable  : {max_ias:.1f}")
        if not passing.empty:
            first = passing.index[0]
            entry_price = df.loc[first, "close"]
            peak_price  = df.loc[first:, "close"].max()
            gain        = (peak_price / entry_price - 1) * 100
            print(f"    First IAS>=35 date  : {first.date()} @ Rs{entry_price:.0f}  (IAS {ias[first]:.1f})")
            print(f"    Gain to forward peak: +{gain:.1f}%")
        else:
            print(f"    Max IAS {max_ias:.1f} never reached 35 — MISSED")
