"""Regression tests for P0/P1 fixes uncovered in the 3-agent blind time-march.

Covers:
  1. utils/forensics.py:
     - Bear/Mixed anchor hijacking (prefer CLV >= 0.5 bullish anchors over higher-volume MIXED/BEAR bars)
     - Large-cap 3.0x volume wall (20d median turnover >= Rs 150 cr qualifies on >= 1.8x with CLV >= 0.65 & up-close)
     - Escalating-volume breakout-cluster ignition preservation (do not reset 5-day WAIT lock on follow-through spikes)
     - Blow-off / active liquidation entry-bar guard (downgrade BUY NOW -> WAIT on weak red distribution bar)
  2. utils/valuation.py:
     - Deep-value provisional rating (BUY* <Bear) instead of opaque UNRATED on shape-only gate A/B failures
     - summary_row exports verdict_raw, bear_blend, base_blend; rating_from_grid preserves <Bear provisional rating
  3. utils/regime_badge.py:
     - Live India VIX level, 5d delta, 252d percentile in compute(), sidebar_line(), and render()
  4. utils/fund_trends.py:
     - Split point-in-time SEBI filing lags (21d Reg 31 for shareholding vs 45d Reg 33 for P&L)
  5. utils/funnel.py, utils/funnel_ui.py, utils/funnel_view.py:
     - Separate FII/DII deltas, absolute delivery value (abs_deliv_cr), and provisional BUY* chips
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int = 260, base_price: float = 1000.0, base_vol: float = 1_000_000.0) -> pd.DataFrame:
    idx = pd.bdate_range("2025-10-01", periods=n)
    c = np.linspace(base_price * 0.85, base_price, n)
    h = c * 1.01
    l = c * 0.99
    v = np.full(n, base_vol, dtype=float)
    return pd.DataFrame({"Open": c * 0.998, "High": h, "Low": l, "Close": c, "Volume": v}, index=idx)


def test_forensics_mixed_anchor_does_not_hijack_valid_bullish_breakout():
    from utils import forensics as fx

    df = _make_ohlcv(260, base_price=12000.0, base_vol=200_000.0)
    # 30 sessions ago: 5.5x MIXED rebalance bar (CLV ~ 0.30)
    k_mix = 230
    df.iloc[k_mix, df.columns.get_loc("High")] = 12300.0
    df.iloc[k_mix, df.columns.get_loc("Low")] = 11700.0
    df.iloc[k_mix, df.columns.get_loc("Close")] = 11880.0   # CLV = 0.30
    df.iloc[k_mix, df.columns.get_loc("Volume")] = 1_100_000.0  # 5.5x

    # 8 sessions ago: 3.3x clean bullish breakout (CLV = 0.85), stays above low
    k_bull = 251
    df.iloc[k_bull, df.columns.get_loc("Low")] = 11950.0
    df.iloc[k_bull, df.columns.get_loc("High")] = 12350.0
    df.iloc[k_bull, df.columns.get_loc("Close")] = 12290.0  # CLV = 0.85
    df.iloc[k_bull, df.columns.get_loc("Volume")] = 660_000.0  # 3.3x
    for j in range(k_bull + 1, 260):
        df.iloc[j, df.columns.get_loc("Close")] = 12320.0 + (j - k_bull) * 15.0
        df.iloc[j, df.columns.get_loc("High")] = df.iloc[j]["Close"] + 40.0
        df.iloc[j, df.columns.get_loc("Low")] = df.iloc[j]["Close"] - 40.0

    r = fx.analyse(df)
    assert r["anchor_date"] == df.index[k_bull]
    assert r["state"] == "GO"
    assert r["action"] == "BUY NOW"


def test_forensics_large_cap_liquidity_tiered_volume_threshold():
    from utils import forensics as fx

    # Median daily turnover = 12,000 * 250,000 = Rs 300 cr (>= Rs 150 cr large-cap tier)
    df = _make_ohlcv(260, base_price=12000.0, base_vol=250_000.0)
    k_lc = 252
    prev_c = df.iloc[k_lc - 1]["Close"]
    df.iloc[k_lc, df.columns.get_loc("Low")] = prev_c * 1.002
    df.iloc[k_lc, df.columns.get_loc("High")] = prev_c * 1.038
    df.iloc[k_lc, df.columns.get_loc("Close")] = prev_c * 1.035  # CLV ~ 0.92, +3.5%
    df.iloc[k_lc, df.columns.get_loc("Volume")] = 250_000.0 * 1.90  # 1.90x (< 3.0x, >= 1.8x)
    for j in range(k_lc + 1, 260):
        df.iloc[j, df.columns.get_loc("Close")] = prev_c * 1.04 + (j - k_lc) * 10.0
        df.iloc[j, df.columns.get_loc("High")] = df.iloc[j]["Close"] + 30.0
        df.iloc[j, df.columns.get_loc("Low")] = df.iloc[j]["Close"] - 30.0

    r = fx.analyse(df)
    assert r["anchor_date"] == df.index[k_lc]
    assert r["state"] == "GO"
    assert r["action"] == "BUY NOW"

    # Small/mid-cap (turnover Rs 20 cr < Rs 150 cr) must STILL require 3.0x
    df_sm = _make_ohlcv(260, base_price=200.0, base_vol=1_000_000.0)  # Rs 20 cr/day
    prev_s = df_sm.iloc[k_lc - 1]["Close"]
    df_sm.iloc[k_lc, df_sm.columns.get_loc("Low")] = prev_s * 1.002
    df_sm.iloc[k_lc, df_sm.columns.get_loc("High")] = prev_s * 1.038
    df_sm.iloc[k_lc, df_sm.columns.get_loc("Close")] = prev_s * 1.035
    df_sm.iloc[k_lc, df_sm.columns.get_loc("Volume")] = 1_000_000.0 * 1.90
    r_sm = fx.analyse(df_sm)
    assert r_sm["state"] == "NO ANCHOR"


def test_forensics_breakout_cluster_preserves_ignition_and_guards_blowoff():
    from utils import forensics as fx

    df = _make_ohlcv(260, base_price=500.0, base_vol=1_000_000.0)
    # Escalating breakout cluster: day 252 (4.6x, CLV 0.85) -> day 255 (12.0x, CLV 0.70)
    k0, k1 = 252, 255
    for k, mult, clv_hi, px in ((k0, 4.6, 535.0, 530.0), (k1, 12.0, 610.0, 595.0)):
        df.iloc[k, df.columns.get_loc("Low")] = px - 25.0
        df.iloc[k, df.columns.get_loc("High")] = clv_hi
        df.iloc[k, df.columns.get_loc("Close")] = px
        df.iloc[k, df.columns.get_loc("Volume")] = 1_000_000.0 * mult
    for j in range(k0 + 1, 260):
        if j == k1:
            continue
        px = 545.0 if j < k1 else 605.0
        df.iloc[j, df.columns.get_loc("Close")] = px
        df.iloc[j, df.columns.get_loc("High")] = px + 8.0
        df.iloc[j, df.columns.get_loc("Low")] = px - 8.0

    # At day 257 (5 sessions after k0, only 2 sessions after k1): ignition k0 is preserved -> GO!
    r_257 = fx.analyse(df.iloc[:258])
    assert r_257["anchor_date"] == df.index[k0]
    assert r_257["days"] == 5
    assert r_257["state"] == "GO"
    # k0 close=530, day 257 close=605, ATR~16 -> extended > 3.5 ATR -> blocked to WAIT!
    assert r_257["ext_atr"] > fx.EXT_ATR_MAX
    assert r_257["entry_weak"] is True
    assert r_257["action"] == "WAIT"

    # When price stays within <= 3.5 ATR of anchor close and closes strong (CLV >= 0.35), action is BUY NOW
    df_calm = df.iloc[:258].copy()
    for j in range(k1 + 1, 258):
        df_calm.iloc[j, df_calm.columns.get_loc("Close")] = 555.0
        df_calm.iloc[j, df_calm.columns.get_loc("High")] = 560.0
        df_calm.iloc[j, df_calm.columns.get_loc("Low")] = 548.0
    r_calm = fx.analyse(df_calm)
    assert r_calm["anchor_date"] == df.index[k0]
    assert r_calm["state"] == "GO"
    assert r_calm["ext_atr"] <= fx.EXT_ATR_MAX
    assert r_calm["action"] == "BUY NOW"

    # Now turn the final bar into a -4.8% CLV 0.10 distribution selloff bar -> must downgrade to WAIT
    df_bad = df.iloc[:259].copy()
    df_bad.iloc[-1, df_bad.columns.get_loc("High")] = 608.0
    df_bad.iloc[-1, df_bad.columns.get_loc("Low")] = 572.0
    df_bad.iloc[-1, df_bad.columns.get_loc("Close")] = 575.5  # -4.9%, CLV = 0.097
    df_bad.iloc[-1, df_bad.columns.get_loc("Volume")] = 2_400_000.0
    r_bad = fx.analyse(df_bad)
    assert r_bad["state"] == "GO"
    assert r_bad["entry_weak"] is True
    assert "CLV" in r_bad["entry_guard"]
    assert r_bad["action"] == "WAIT"


def test_valuation_provisional_deep_value_below_bear_blend():
    from utils import valuation as V

    # rating_from_grid should surface BUY* (<Bear · gate A, B) instead of opaque UNRATED when price <= bear_blend
    grid = np.linspace(550.0, 1500.0, 99)
    pct, rat = V.rating_from_grid(
        grid, 410.0, gates_ok=False, confidence="HIGH", identity="COMPOUNDER",
        label_if_unrated="UNRATED (gate A, B)", bear_blend=552.8,
    )
    assert pct < 5.0
    assert rat == "BUY* (<Bear · gate A, B)"

    # Hard-gate failures (C, 1F, E, F) must NEVER be promoted to BUY*
    _, rat_hard = V.rating_from_grid(
        grid, 410.0, gates_ok=False, confidence="HIGH", identity="COMPOUNDER",
        label_if_unrated="UNRATED (gate A, C)", bear_blend=552.8,
    )
    assert rat_hard == "UNRATED (gate A, C)"


def test_regime_badge_vix_compute_sidebar_and_render():
    from utils import regime_badge as rb

    idx = pd.bdate_range("2026-01-01", periods=220)
    nifty_df = pd.DataFrame({
        "Close": np.linspace(23000.0, 25800.0, 220),
        "High": np.linspace(23100.0, 25900.0, 220),
    }, index=idx)
    vix_vals = np.linspace(14.0, 17.5, 220)
    vix_vals[-6:] = [15.5, 16.0, 16.4, 17.0, 17.2, 17.7]
    vix_df = pd.DataFrame({"date": idx, "close": vix_vals})
    univ = pd.DataFrame({
        "ticker": [f"S{i}.NS" for i in range(60)],
        "currentPrice": [110.0 if i < 36 else 90.0 for i in range(60)],
        "twoHundredDayAverage": [100.0] * 60,
    })

    info = rb.compute(nifty_df, univ, vix_df=vix_df)
    assert info["vix"]["close"] == pytest.approx(17.7)
    assert info["vix"]["chg_5d"] == pytest.approx(2.2)
    line = rb.sidebar_line(info)
    assert "VIX 17.7 (+2.2)" in line
    assert "BULL" not in line.split("**Model rules:**", 1)[1]

    rendered = []

    class _FakeSt:
        def markdown(self, body, **kwargs):
            rendered.append(body)

    rb.render(info, _FakeSt(), compact=True)
    assert "India VIX: <b>17.7</b>" in rendered[0]


def test_fund_trends_split_sebi_pit_lags(monkeypatch):
    from utils import fund_trends as ft

    rows = [
        # Jun 2026 quarter (both SHP & P&L live well before Oct 30)
        dict(ticker="ULTRACEMCO", period=pd.Timestamp("2026-06-30"), field="sales", value=18000.0),
        dict(ticker="ULTRACEMCO", period=pd.Timestamp("2026-06-30"), field="fii", value=16.0),
        dict(ticker="ULTRACEMCO", period=pd.Timestamp("2026-06-30"), field="dii", value=19.7),
        # Sep 2026 quarter: SHP due Oct 21 (+21d), P&L due Nov 14 (+45d)
        dict(ticker="ULTRACEMCO", period=pd.Timestamp("2026-09-30"), field="sales", value=19500.0),
        dict(ticker="ULTRACEMCO", period=pd.Timestamp("2026-09-30"), field="fii", value=15.0),
        dict(ticker="ULTRACEMCO", period=pd.Timestamp("2026-09-30"), field="dii", value=20.9),
    ]
    Q = {"ULTRACEMCO": pd.DataFrame(rows)}
    monkeypatch.setattr(ft, "_data", lambda root=".": ({}, Q))
    ft._quarterly.cache_clear()
    ft._inst_map.cache_clear()

    # On 2026-10-30 (+30d after Sep 30): Sep SHP IS visible (30 >= 21), Sep P&L is NOT (30 < 45)!
    q_oct30 = ft.quarterly("ULTRACEMCO.NS", root="x", as_of="2026-10-30")
    assert pd.Timestamp("2026-09-30") in q_oct30.index
    assert q_oct30.loc[pd.Timestamp("2026-09-30"), "dii"] == pytest.approx(20.9)
    assert pd.isna(q_oct30.loc[pd.Timestamp("2026-09-30"), "sales"])
    assert q_oct30.loc[pd.Timestamp("2026-06-30"), "sales"] == pytest.approx(18000.0)

    # On 2026-10-15 (+15d after Sep 30): neither Sep SHP nor Sep P&L is visible
    im_oct15 = ft.inst_map(root="x", as_of="2026-10-15")
    assert im_oct15["ULTRACEMCO.NS"]["period"] == "Jun 2026"

    # On 2026-10-30 (+30d after Sep 30): Sep SHP IS live in inst_map with separate FII/DII deltas
    im_oct30 = ft.inst_map(root="x", as_of="2026-10-30")
    assert im_oct30["ULTRACEMCO.NS"]["period"] == "Sep 2026"
    assert im_oct30["ULTRACEMCO.NS"]["d_dii_q"] == pytest.approx(1.2)
    assert im_oct30["ULTRACEMCO.NS"]["d_fii_q"] == pytest.approx(-1.0)
    ft._quarterly.cache_clear()
    ft._inst_map.cache_clear()


def test_funnel_ui_and_view_surface_dii_sweep_abs_delivery_and_provisional_val():
    from utils import funnel_ui, funnel_view

    row = pd.Series(dict(
        ticker="ULTRACEMCO.NS",
        name="UltraTech Cement",
        industry="Cement",
        funnel_score=76.0,
        funnel_grade="A",
        score_note="Mom 18 · Set 20 · Str 10 · Qual 16 · Cat 0",
        price=12500.0,
        comp_rs=2.0,
        risk_pct=11.0,
        pe=35.0,
        biz_score=65.0,
        technicals="WAIT",
        fx_supply="CLEAN AIR",
        inst_label="NEUTRAL",
        inst_q=0.2,
        inst_y=0.5,
        dii_q=1.2,
        dii_y=3.4,
        fii_q=-1.0,
        fii_y=-2.9,
        deliv_pct=15.9,
        deliv_avg20=35.0,
        liq_cr=2033.0,
        abs_deliv_cr=323.2,
        val_rating="BUY* (<Bear · gate A, B)",
        val_pct=0.6,
        nse_flags="",
        biz_flags="",
    ))
    chips = funnel_ui.build_signal_chips(row)
    pro_txt = " | ".join(c["text"] for c in chips["pros"])
    assert "🏦 DII +1.2q/+3.4y" in pro_txt
    assert "📦 ₹323cr/d Deliv (16%)" in pro_txt
    assert "💰 Val BUY* <Bear (P1)" in pro_txt or "💰 Val BUY* <Bear (P0)" in pro_txt

    df = pd.DataFrame([row])
    assert "(DII +1.2 / FII -1.0)" in funnel_view._inst_labels(df)[0]
    assert "₹323cr" in funnel_view._deliv_labels(df)[0]


def test_live_ultracemco_and_kalyankjil_forensics_fixes():
    if not os.path.exists(os.path.join("data", "nse", "prices")):
        pytest.skip("NSE price cache not present")
    from utils import forensics as fx, nse_prices

    hist = nse_prices.history(["ULTRACEMCO.NS", "KALYANKJIL.NS"])
    ul = hist.get("ULTRACEMCO.NS")
    kj = hist.get("KALYANKJIL.NS")
    if ul is None or kj is None or len(ul) < 250 or len(kj) < 250:
        pytest.skip("Insufficient UltraTech / Kalyan history")

    # Blind Spot 1: UltraTech Jul 28 must NOT be hijacked by May 29 MIXED bar -> valid Jul anchor + GO/BUY NOW
    r_ul_jul = fx.analyse(ul.loc[: "2026-07-28"])
    assert r_ul_jul["state"] == "GO"
    assert r_ul_jul["action"] == "BUY NOW"
    assert pd.Timestamp(r_ul_jul["anchor_date"]) >= pd.Timestamp("2026-07-20")

    # Blind Spot 2: UltraTech Nov rally must NOT be stranded at NO ANCHOR -> Nov 5 (1.92x) anchors GO/BUY NOW
    r_ul_nov = fx.analyse(ul.loc[: "2026-11-12"])
    assert r_ul_nov["state"] == "GO"
    assert r_ul_nov["action"] == "BUY NOW"
    assert pd.Timestamp(r_ul_nov["anchor_date"]) == pd.Timestamp("2026-11-05")

    # Blind Spot 3A: Kalyan Jul escalating volume keeps Jul 8 ignition -> GO by Jul 15,
    # AND >3.5 ATR vertical extension (+6.96 ATR) blocks chasing BUY NOW -> WAIT
    r_kj_jul = fx.analyse(kj.loc[: "2026-07-15"])
    assert pd.Timestamp(r_kj_jul["anchor_date"]) == pd.Timestamp("2026-07-08")
    assert r_kj_jul["state"] == "GO"
    assert r_kj_jul["ext_atr"] > fx.EXT_ATR_MAX
    assert r_kj_jul["action"] == "WAIT"

    # Blind Spot 3A + 3B: Kalyan Nov keeps Nov 6 ignition, blocks Nov 13/16 (>3.5 ATR) AND Nov 18 (CLV 0.10)
    r_kj_nov16 = fx.analyse(kj.loc[: "2026-11-16"])
    assert pd.Timestamp(r_kj_nov16["anchor_date"]) == pd.Timestamp("2026-11-06")
    assert r_kj_nov16["ext_atr"] > fx.EXT_ATR_MAX
    assert r_kj_nov16["action"] == "WAIT"

    r_kj_nov18 = fx.analyse(kj.loc[: "2026-11-18"])
    assert pd.Timestamp(r_kj_nov18["anchor_date"]) == pd.Timestamp("2026-11-06")
    assert r_kj_nov18["entry_weak"] is True
    assert r_kj_nov18["action"] == "WAIT"


def test_regime_manager_breadth_thrust_and_recovery_mode():
    from utils import regime_badge as rb, regime_manager as rm

    # 1. Classic 10d Zweig breadth thrust (+33pp from 10% base) + 52W lows rollover triggers RECOVERY from BEAR
    b50 = pd.Series([18.0, 14.0, 10.3, 15.0, 17.0, 21.3, 25.4, 43.6])
    lo52 = pd.Series([20.0, 30.0, 44.3, 4.7, 5.8, 1.6, 1.0, 0.3])
    rec = rm.detect_capitulation_recovery(b50, lo52)
    assert rec["active"] is True
    assert rec["breadth_thrust"] is True
    assert rec["lows_rollover"] is True
    assert rm.classify_regime(23000.0, 24500.0, 26000.0) == "BEAR"
    assert rm.classify_regime(23000.0, 24500.0, 26000.0, recovery=rec) == "RECOVERY"
    # CRISIS must never be overridden by recovery
    assert rm.classify_regime(17000.0, 24500.0, 26000.0, recovery=rec) == "CRISIS"
    assert rb.engine_short({"mode": "RECOVERY"}) == "🚀 Early recovery"
    assert rb.engine_status({"mode": "RECOVERY"}) == "🚀 Early recovery (12% trail)"

    # 2. 52W lows rollover + FII/DII positive cash flow turn also triggers RECOVERY
    b50_mild = pd.Series([15.0, 12.0, 13.5, 16.5, 19.5])
    lo52_roll = pd.Series([14.6, 20.4, 17.1, 8.4, 4.9])
    fii_cash = pd.Series([-2000.0, -500.0, 1800.0])
    dii_cash = pd.Series([2500.0, 1200.0, 900.0])
    rec2 = rm.detect_capitulation_recovery(b50_mild, lo52_roll, fii_net=fii_cash, dii_net=dii_cash)
    assert rec2["lows_rollover"] is True
    assert rec2["inst_flow_reversal"] is True
    assert rec2["active"] is True

    # 3. Live 2026 NSE check: Mar 30 capitulation is NOT recovery yet; Apr 8 thrust IS RECOVERY
    if os.path.exists(os.path.join("data", "nse", "prices")):
        rec_mar30 = rm.load_recovery_signals(as_of="2026-03-30")
        rec_apr08 = rm.load_recovery_signals(as_of="2026-04-08")
        assert rec_mar30["active"] is False
        assert rec_apr08["active"] is True
        assert rec_apr08["breadth_thrust"] is True
        assert rec_apr08["lows_rollover"] is True
