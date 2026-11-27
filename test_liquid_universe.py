"""Unit & integration tests for utils/liquid_universe.py guardrails, UI filters, and Nifty-500 regime anchor."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from utils import liquid_universe as lu
from utils import regime_badge as rb


def test_build_liquid_universe_enforces_all_guardrails():
    liq_df = pd.DataFrame(
        [
            # 1. Valid verified mid-cap >= 2Cr/d, MCap >= 500 Cr -> VERIFIED
            dict(symbol="GOODMID", days60=60, days20=20, med_turn_60d_cr=8.5, med_turn_20d_cr=6.2, last_close=420.0, series="EQ"),
            # 2. Already in core nifty1000 -> excluded from liquid expansion (no duplicates)
            dict(symbol="RELIANCE", days60=60, days20=20, med_turn_60d_cr=900.0, med_turn_20d_cr=850.0, last_close=1400.0, series="EQ"),
            # 3. ETF / not in EQUITY_L -> blocked
            dict(symbol="NIFTYBEES", days60=60, days20=20, med_turn_60d_cr=120.0, med_turn_20d_cr=110.0, last_close=280.0, series="EQ"),
            # 4. BE trade-to-trade series -> blocked
            dict(symbol="BESTOCK", days60=60, days20=20, med_turn_60d_cr=15.0, med_turn_20d_cr=12.0, last_close=250.0, series="BE"),
            # 5. Penny stock < Rs 20 -> blocked
            dict(symbol="PENNYCO", days60=60, days20=20, med_turn_60d_cr=4.0, med_turn_20d_cr=3.5, last_close=4.2, series="EQ"),
            # 6. Known microcap < Rs 500 Cr -> blocked
            dict(symbol="MICROCO", days60=60, days20=20, med_turn_60d_cr=6.0, med_turn_20d_cr=5.0, last_close=95.0, series="EQ"),
            # 7. Hard surveillance (GSM / ASM Stage 4) -> blocked
            dict(symbol="GSMTRAP", days60=60, days20=20, med_turn_60d_cr=25.0, med_turn_20d_cr=20.0, last_close=310.0, series="EQ"),
            # 8. Unscraped brand-new institutional IPO (>= 5Cr/d, >= 20 sessions) -> IPO_UNVERIFIED
            dict(symbol="BIGIPO", days60=22, days20=20, med_turn_60d_cr=45.0, med_turn_20d_cr=45.0, last_close=520.0, series="EQ"),
            # 9. Unscraped small unknown (3 Cr/d < 5 Cr/d unverified floor) -> blocked until Screener verifies MCap
            dict(symbol="SMALLUNK", days60=60, days20=20, med_turn_60d_cr=3.0, med_turn_20d_cr=2.8, last_close=110.0, series="EQ"),
            # 10. Previously admitted verified stock that dipped to 1.3 Cr/d -> RETAINED via hysteresis (>= 1.0 Cr)
            dict(symbol="HYSTKEEP", days60=60, days20=20, med_turn_60d_cr=1.3, med_turn_20d_cr=1.1, last_close=300.0, series="EQ"),
            # 11. Previously admitted verified stock that collapsed to 0.6 Cr/d -> dropped (< 1.0 Cr)
            dict(symbol="HYSTDROP", days60=60, days20=20, med_turn_60d_cr=0.6, med_turn_20d_cr=0.5, last_close=300.0, series="EQ"),
            # 12. Stale burst (60D >= 2.0 but 20D collapsed to 0.4) -> blocked
            dict(symbol="FADEDVOL", days60=60, days20=20, med_turn_60d_cr=2.5, med_turn_20d_cr=0.4, last_close=180.0, series="EQ"),
        ]
    ).set_index("symbol")

    eq_names = {
        s: f"{s} Ltd"
        for s in ["GOODMID", "RELIANCE", "BESTOCK", "PENNYCO", "MICROCO", "GSMTRAP",
                  "BIGIPO", "SMALLUNK", "HYSTKEEP", "HYSTDROP", "FADEDVOL"]
    }
    eq_series = {s: ("BE" if s == "BESTOCK" else "EQ") for s in eq_names}
    mcap_map = {
        "GOODMID.NS": 3200.0,
        "MICROCO.NS": 280.0,
        "GSMTRAP.NS": 1500.0,
        "HYSTKEEP.NS": 2100.0,
        "HYSTDROP.NS": 1900.0,
        "FADEDVOL.NS": 1200.0,
    }
    ind_map = {"GOODMID.NS": "IT - Software", "HYSTKEEP.NS": "Chemicals & Petrochemicals"}
    surv_map = {"GSMTRAP": "GSM Stage 3"}
    prev_df = pd.DataFrame([
        dict(Ticker="HYSTKEEP.NS", Company_Name="HYSTKEEP Ltd", Sub_Industry="Chemicals & Petrochemicals", added_date="2026-10-01"),
        dict(Ticker="HYSTDROP.NS", Company_Name="HYSTDROP Ltd", Sub_Industry="Finance", added_date="2026-10-01"),
    ])

    out = lu.build_liquid_universe(
        liq_df=liq_df,
        as_of="2026-11-26",
        core_tickers={"RELIANCE.NS"},
        eq_names=eq_names,
        eq_series=eq_series,
        mcap_map=mcap_map,
        ind_map=ind_map,
        ext_names={},
        surv_map=surv_map,
        prev_df=prev_df,
    )

    got = dict(zip(out["Ticker"], out["status"]))
    assert set(got.keys()) == {"GOODMID.NS", "BIGIPO.NS", "HYSTKEEP.NS"}
    assert got["GOODMID.NS"] == "VERIFIED"
    assert got["BIGIPO.NS"] == "IPO_UNVERIFIED"
    assert got["HYSTKEEP.NS"] == "RETAINED"
    # Preserves original added_date on retained stocks
    assert out.set_index("Ticker").loc["HYSTKEEP.NS", "added_date"] == "2026-10-01"
    assert out.set_index("Ticker").loc["GOODMID.NS", "Sub_Industry"] == "IT - Software"


def test_apply_screen_filter_tiers_and_noise_hygiene():
    df = pd.DataFrame([
        dict(ticker="MEGA.NS", currentPrice=1000.0, med_turn_60d_cr=150.0, nse_flags="", nse_series="EQ"),
        dict(ticker="CORE.NS", currentPrice=400.0, med_turn_60d_cr=8.0, nse_flags="", nse_series="EQ"),
        dict(ticker="TRAD.NS", currentPrice=250.0, med_turn_60d_cr=2.5, nse_flags="", nse_series="EQ"),
        dict(ticker="ILLIQ.NS", currentPrice=300.0, med_turn_60d_cr=0.8, nse_flags="", nse_series="EQ"),
        dict(ticker="PENNY.NS", currentPrice=9.5, med_turn_60d_cr=5.0, nse_flags="", nse_series="EQ"),
        dict(ticker="ASMTRAP.NS", currentPrice=320.0, med_turn_60d_cr=25.0, nse_flags="⚠️ LT-ASM Stage 1", nse_series="EQ"),
        dict(ticker="BETRAP.NS", currentPrice=220.0, med_turn_60d_cr=12.0, nse_flags="", nse_series="BE"),
    ])

    f_trad = lu.apply_screen_filter(df, tier=lu.TIER_TRADABLE, hide_noisy=True)
    assert set(f_trad["ticker"]) == {"MEGA.NS", "CORE.NS", "TRAD.NS"}

    f_core = lu.apply_screen_filter(df, tier=lu.TIER_CORE, hide_noisy=True)
    assert set(f_core["ticker"]) == {"MEGA.NS", "CORE.NS"}

    f_inst = lu.apply_screen_filter(df, tier=lu.TIER_INST, hide_noisy=True)
    assert set(f_inst["ticker"]) == {"MEGA.NS"}

    f_n500 = lu.apply_screen_filter(df, tier=lu.TIER_NIFTY500, hide_noisy=True, nifty500_set={"MEGA.NS", "ILLIQ.NS"})
    assert set(f_n500["ticker"]) == {"MEGA.NS", "ILLIQ.NS"}

    f_all_noisy = lu.apply_screen_filter(df, tier=lu.TIER_ALL, hide_noisy=False)
    assert len(f_all_noisy) == len(df)

    # Disabling hide_noisy reveals ASMTRAP and BETRAP on Tradable tier
    f_show_noisy = lu.apply_screen_filter(df, tier=lu.TIER_TRADABLE, hide_noisy=False)
    assert {"ASMTRAP.NS", "BETRAP.NS"} <= set(f_show_noisy["ticker"])


def test_regime_badge_anchors_breadth_to_nifty500_when_present():
    n500 = sorted(lu.nifty500_tickers())
    assert len(n500) >= 400
    # 250 Nifty-500 stocks: 60% above 200DMA -> BULL
    # 400 small-cap tail stocks: 0% above 200DMA (would drag unanchored breadth down to 23% BEAR!)
    rows = []
    for i, t in enumerate(n500[:250]):
        rows.append(dict(ticker=t, currentPrice=110.0 if i < 150 else 90.0, twoHundredDayAverage=100.0, extended=False))
    for i in range(400):
        rows.append(dict(ticker=f"TAIL{i}.NS", currentPrice=80.0, twoHundredDayAverage=100.0, extended=False))
    df = pd.DataFrame(rows)

    b = rb.breadth_regime(df)
    assert b["anchor"] == "Nifty 500"
    assert b["n"] == 250
    assert round(b["pct"], 1) == 60.0
    assert b["regime"] == "BULL"
    assert b["broad_n"] == 650
    assert round(b["broad_pct"], 1) == round(150 / 650 * 100, 1)


def test_effective_liquidity_prefers_median_over_spike_inflated_mean():
    df = pd.DataFrame([
        dict(ticker="SPIKE.NS", currentPrice=100.0, averageVolume=5_000_000, med_turn_60d_cr=0.9, med_turn_20d_cr=0.8),
        dict(ticker="LEGACY.NS", currentPrice=100.0, averageVolume=500_000),
    ])
    liq = lu.effective_liquidity_cr(df)
    # SPIKE uses 60D median (0.9 Cr), NOT mean (50 Cr)!
    assert liq.iloc[0] == 0.9
    # LEGACY falls back cleanly to averageVolume * currentPrice / 1e7 = 5.0 Cr
    assert liq.iloc[1] == 5.0
