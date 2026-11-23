"""
🔬 Technicals tab -- chart-prompt v6 structural forensics for every stock.

Reads the fx_* columns the daily engine writes (utils/forensics.snapshot) and
ranks them by the backtested edge of their cell (analysis/
forensics_gate_study.csv). A single-stock deep dive re-runs the same logic
on fresh 2-year bars and prints the v6 report.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import streamlit as st

STUDY = "analysis/forensics_gate_study.csv"
ACTIONS = ["BUY NOW", "BUY ON TRIGGER", "WAIT", "WATCH"]
SUPPLY_LABEL = {"CLEAN AIR": "🌤️ Clean air", "INSIDE shelf": "🧱 In a shelf", "NEAR OVERHANG": "⛔ Overhang near",
                "FAR OVERHANG": "☁️ Overhang far", "NOT COMPUTED": ""}
STATE_LABEL = {"GO": "🟢 GO", "WAIT (running)": "⏳ WAIT (<5d)", "WAIT (40-80%)": "⏳ WAIT (40-80%)",
               "FADE": "🟠 FADE", "VETO": "🔴 VETO", "NO ANCHOR": "⚪ No anchor",
               "MIXED anchor": "🟤 Mixed anchor", "BEAR anchor": "🟤 Bear anchor"}

SIGNAL_HELP = ("Technicals × valuation. Technical + = forensics BUY NOW (trend + GO volume anchor, no near overhang); "
               "technical − = below both the 50- and 200-day averages. BUY = technical + and valuation BUY/ACCUMULATE/"
               "HOLD. SELL = technical − and valuation HOLD/REDUCE/SELL, or valuation SELL with neutral technicals. "
               "A 🟠 weak valuation counts one notch less and a 🔴 unreliable one is ignored. Everything else HOLD: an expensive valuation downgrades a technical BUY to HOLD (momentum winners "
               "usually look expensive) and a cheap stock in a broken trend stays HOLD (no falling knives). Only the "
               "technical side is backtested; the valuation side is untested so far.")

HELP = """**How to read this (2016-26 backtest, median vs the typical stock, 3m / 6m):**
- **Gate 1** (above MA50 & MA200, within 15% of the 52w high) is the core: **+1.4 / +2.5 pp**.
- **🟢 GO** (biggest volume day of 60 sessions, held >= 80% of closes for 5+ days): +1.7 / +3.4.
- **⭐ Absorption** (down days on <= 0.8x median volume): GO + absorption **+3.2 / +5.4**, 12/12 years
  -- the best cell; strongest in 2021-26.
- **🪤 Shakeout** (broke MA50 / anchor low intraday, closed high on light volume): +2.1 / +4.8.
- **⛔ Overhang near** (trapped volume shelf within 1.5 ATR above): lags 1-3 months -> *BUY ON TRIGGER* =
  wait for a close above the shelf top.
- **Distribution** is shown but is **not** a warning (it did as well as Gate 1). RSI > 70 is not a negative.
- **T1 (2R)** is a reference level only: exiting at 2R halved per-trade returns. Stops > 10% -> half size.
- Event-study medians, not a portfolio: preferring breakouts did not beat the engine's 6-month RS ranking.
  Use this for your own picks and timing. Full write-up: analysis/FORENSICS_VERDICT.md."""


@st.cache_data(ttl=3600)
def _edges() -> dict:
    if not os.path.exists(STUDY):
        return {}
    t = pd.read_csv(STUDY)
    return {r["case"]: (r.get("3mo vsTypical"), r.get("6mo vsTypical"), r.get("yrs 3mo>0")) for r in t.to_dict("records")}


def _as_bool(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    return str(x).strip().lower() in ("true", "1", "yes")


def _cell(row) -> str:
    """Most specific backtested cell for this row."""
    if not _as_bool(row.get("fx_gate1")):
        return ""
    s, sup, sig = row.get("fx_state"), row.get("fx_supply"), row.get("fx_signature")
    if s == "GO":
        if sig == "ABSORPTION" and sup != "NEAR OVERHANG":
            return "BUY NOW + absorption"
        if _as_bool(row.get("fx_shakeout")):
            return "v6 GO + shakeout (v6-only probe)"
        return "BUY ON TRIGGER (GO + near overhang)" if sup == "NEAR OVERHANG" else "BUY NOW (GO, no near overhang)"
    return s if isinstance(s, str) else ""


def _history_mode(ticker: str, source: str | None = None) -> pd.DataFrame:
    from utils.yf_safe import safe_history
    from utils.fast_data_engine import _sanitize_price_bars
    h = safe_history(ticker, period="2y", source=source)
    if h is None or h.empty:
        return pd.DataFrame()
    h = h.copy()
    if getattr(h.index, "tz", None) is not None:
        h.index = h.index.tz_localize(None)
    h = _sanitize_price_bars(h)
    try:
        from utils import nse_prices
        is_nse = (source != "yahoo") and (str(ticker).endswith(".NS") or str(ticker).startswith("^")) and nse_prices.is_current()
        h.attrs["source"] = "NSE Bhavcopy daily" if is_nse else ("Yahoo Finance live" if source == "yahoo" else "Yahoo Finance daily")
    except Exception:
        pass
    return h


@st.cache_data(ttl=900, show_spinner=False)
def _history(ticker: str) -> pd.DataFrame:
    return _history_mode(ticker, None)


@st.cache_data(ttl=900, show_spinner=False)
def _history_live(ticker: str) -> pd.DataFrame:
    return _history_mode(ticker, "yahoo")


def render(df: pd.DataFrame):
    from utils.ui_components import page_header
    st.markdown(page_header("🔬 Technicals + Valuation", "Structural forensics (chart-prompt v6, backtested) followed "
                            "by a mechanical v4.1 Monte Carlo valuation -- no LLM calls"), unsafe_allow_html=True)
    with st.expander("ℹ️ What each flag is worth (backtest)", expanded=False):
        st.markdown(HELP)

    if "fx_state" not in df.columns:
        st.info("The loaded snapshot predates the Technicals fields -- they arrive with the next daily engine run. "
                "The single-stock deep dive below works now.")
    else:
        _table(df)
    st.divider()
    _deep_dive(df)
    from utils.valuation_view import render_reference
    render_reference(df)


def _add_distance_cols(v: pd.DataFrame) -> pd.DataFrame:
    """Compute price % distances to technical stop-loss, 50-DMA, and 52W high."""
    out = v.copy()
    px = pd.to_numeric(out["price"], errors="coerce") if "price" in out else pd.Series(np.nan, index=out.index, dtype="float64")

    stop = pd.to_numeric(out["fx_stop"], errors="coerce") if "fx_stop" in out else pd.Series(np.nan, index=out.index, dtype="float64")
    valid_stop = px.notna() & (px > 0) & stop.notna() & (stop > 0)
    out["dist_stop_pct"] = np.where(valid_stop, (px - stop) / px * 100.0, np.nan)

    ma50 = pd.to_numeric(out["fiftyDayAverage"], errors="coerce") if "fiftyDayAverage" in out else pd.Series(np.nan, index=out.index, dtype="float64")
    valid_ma50 = px.notna() & (px > 0) & ma50.notna() & (ma50 > 0)
    d_ma50 = pd.Series(np.where(valid_ma50, (px / ma50 - 1.0) * 100.0, np.nan), index=out.index, dtype="float64")
    for col in ("dist_ma50", "dist_50dma"):
        if col in out:
            d_ma50 = d_ma50.fillna(pd.to_numeric(out[col], errors="coerce"))
    out["dist_ma50_pct"] = d_ma50

    d_52w = pd.to_numeric(out["dist_52w"], errors="coerce") if "dist_52w" in out else pd.Series(np.nan, index=out.index, dtype="float64")
    for col in ("fiftyTwoWeekHigh", "high_52w"):
        if col in out:
            hi52 = pd.to_numeric(out[col], errors="coerce")
            valid_hi = px.notna() & (px > 0) & hi52.notna() & (hi52 > 0)
            d_52w = d_52w.fillna(pd.Series(np.where(valid_hi, (px / hi52 - 1.0) * 100.0, np.nan), index=out.index, dtype="float64"))
    if "fx_dist_hi" in out:
        d_52w = d_52w.fillna(pd.to_numeric(out["fx_dist_hi"], errors="coerce"))
    out["dist_52w_pct"] = d_52w
    return out


def _table(df: pd.DataFrame):
    from utils.breakout_tags import load_industry_scores
    from utils.nifty1000_list import SUB_INDUSTRY_MAP

    d = df[df["extended"] != True].copy() if "extended" in df.columns else df.copy()
    d["sub_industry"] = d["ticker"].map(SUB_INDUSTRY_MAP)
    d["ind_score"] = pd.to_numeric(d["sub_industry"].map(load_industry_scores()), errors="coerce")
    if "fx_dist_hi" not in d:
        d["fx_dist_hi"] = np.nan
    for c in ("fx_gate1", "fx_shakeout", "fx_half_size", "fx_gap"):
        d[c] = d[c].map(_as_bool) if c in d else False
    px_main = pd.to_numeric(d["price"], errors="coerce") if "price" in d else pd.Series(np.nan, index=d.index)
    px_cur = pd.to_numeric(d["currentPrice"], errors="coerce") if "currentPrice" in d else pd.Series(np.nan, index=d.index)
    d["price"] = px_main.fillna(px_cur)
    from utils.valuation_view import QUALITY_LEVELS, VAL_BUCKETS, snapshot_date, table_columns
    d = table_columns(d)

    g1 = d[d.fx_gate1]
    go = g1[g1.fx_state == "GO"]
    m = st.columns(5)
    m[0].metric("Gate 1 pass", len(g1), help="Above MA50 & MA200, within 15% of the 52w high")
    m[1].metric("🟢 BUY NOW", int((d.fx_action == "BUY NOW").sum()))
    m[2].metric("⛔ BUY ON TRIGGER", int((d.fx_action == "BUY ON TRIGGER").sum()))
    m[3].metric("⭐ GO + absorption", int((go.fx_signature == "ABSORPTION").sum()))
    m[4].metric("🪤 Shakeouts (Gate 1)", int(g1.fx_shakeout.sum()))

    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
    with f1:
        acts = st.multiselect("Action", ACTIONS, default=["BUY NOW", "BUY ON TRIGGER"], key="tx_actions")
    with f2:
        absorb = st.toggle("⭐ Absorption only", value=False, key="tx_absorb")
    with f3:
        shake = st.toggle("🪤 Shakeout only", value=False, key="tx_shake")
    with f4:
        lead = st.toggle("🟢 Leader industries", value=False, key="tx_lead",
                         help="Sub-industry rotation score >= 70. BUY NOW in a leader industry: +2.2 / +3.7 pp, 12/12 years")
    g1c, g2c, g3c = st.columns(3)
    with g1c:
        sigs = st.multiselect("🎯 Signal (technicals × valuation)", ["BUY", "HOLD", "SELL"], default=[], key="tx_sig",
                              help=SIGNAL_HELP)
    with g2c:
        vals = st.multiselect("💰 Valuation (v4.1 mechanical)", VAL_BUCKETS + ["NOT VALUED"], default=[],
                              key="tx_val", help="Where today's price sits in the stock's own fair-value distribution: "
                              "<P20 BUY · P20-40 ACCUMULATE · P40-70 HOLD · P70-88 REDUCE · >P88 SELL. The icon after "
                              "it is the quality grade; the reason and which way it bends the verdict are in Val note. "
                              "Nothing selected = no filter."
                              + (f" Distribution as of {snapshot_date()}." if snapshot_date() else
                                 " Not built yet -- run the workflow with script = fundamentals."))
    with g3c:
        quals = st.multiselect("🧭 Valuation quality", QUALITY_LEVELS, default=[], key="tx_vq",
                               help="✅ clean: all v4.1 gates pass · 🟡 caution: read the reason · 🟠 weak: counts one "
                                    "notch less in the signal · 🔴 unreliable: ignored in the signal")
    v = d
    if quals:
        v = v[v.val_quality.isin(quals)]
    if sigs:
        v = v[v.signal_raw.isin(sigs)]
    if vals:
        v = v[v.val_bucket.isin(vals)]
    from utils.ui_search import match
    q = st.text_input("🔎 Search name or ticker", key="tx_search", placeholder="e.g. cummins, HINDCOPPER",
                      help="Part of a company name or ticker; separate several with commas. A search shows "
                           "the stock whatever its action.")
    if q:
        v = match(v, q)
    if acts and not q:
        v = v[v.fx_action.isin(acts)]
    if absorb:
        v = v[v.fx_signature == "ABSORPTION"]
    if shake:
        v = v[v.fx_shakeout]
    if lead:
        v = v[v.ind_score >= 70]

    s1, s2, s3 = st.columns(3)
    with s1:
        cap = st.number_input("Capital (₹)", min_value=10000.0, value=1_000_000.0, step=50000.0, key="tx_cap")
    with s2:
        riskp = st.number_input("Risk / trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.25, key="tx_risk")
    with s3:
        maxpos = st.number_input("Max positions", min_value=1, max_value=50, value=15, step=1, key="tx_maxpos")

    v = v.copy()
    E = _edges()
    cells = [_cell(r) for r in v.to_dict("records")]
    v["edge_3m"] = [E.get(c, (np.nan,))[0] for c in cells]
    v["edge_6m"] = [E.get(c, (np.nan, np.nan))[1] for c in cells]
    v["edge_years"] = [E.get(c, (None, None, ""))[2] for c in cells]
    stop = pd.to_numeric(v.fx_stop, errors="coerce")
    entry = pd.to_numeric(v.fx_entry, errors="coerce").fillna(v.price) if "fx_entry" in v else v.price   # trigger price for BUY ON TRIGGER
    stop_valid = stop.notna() & (stop > 0) & v.price.notna() & (v.price > 0) & (stop < v.price) & (stop < entry)
    risk_ps = (entry - stop).where(stop_valid)
    risk_rs = cap * riskp / 100 * np.where(v.fx_half_size, 0.5, 1.0)
    slot = cap / int(maxpos)
    by_risk = np.floor(risk_rs / risk_ps)
    by_slot = np.floor(slot / entry)
    # no valid stop (<= 0 or price at/below it) -> no position
    valid = stop_valid & risk_ps.notna()
    v["shares"] = np.where(valid, np.fmin(by_risk, by_slot), 0)
    v["shares"] = pd.to_numeric(v["shares"], errors="coerce").fillna(0).astype(int)
    v["sized_by"] = np.where(~valid, "no valid stop",
                             np.where(by_risk <= by_slot, "stop risk" + np.where(v.fx_half_size, " (½)", ""), "slot cap"))
    v["buy_value"] = (v["shares"] * entry).round(0)
    v["fx_retention"] = pd.to_numeric(v.fx_retention, errors="coerce") * 100
    v["trigger"] = np.where(v.fx_supply == "NEAR OVERHANG", pd.to_numeric(v.fx_shelf_top, errors="coerce"), np.nan)
    v["state_l"] = v.fx_state.map(STATE_LABEL).fillna("")
    v["supply_l"] = v.fx_supply.map(SUPPLY_LABEL).fillna("")
    v["flags"] = (np.where(v.fx_signature == "ABSORPTION", "⭐ ", "") + np.where(v.fx_shakeout, "🪤 ", "")
                  + np.where(v.fx_gap, "↗️ gap ", "") + np.where(v.fx_signature == "DISTRIBUTION", "dist. ", ""))
    _bare_tk = v.ticker.astype(str).str.replace(".NS", "", regex=False).str.replace(".BO", "", regex=False)
    v["link"] = "https://www.screener.in/company/" + _bare_tk + "/"
    v["_bare_tk"] = _bare_tk
    v = _add_distance_cols(v)
    try:                                       # FII + DII holding change from the quarterly filings (context, not scored)
        from utils import fund_trends as _ft
        _im = _ft.inst_map()
        _icon = {"ACCUMULATING": "🟢", "DISTRIBUTING": "🔴", "NEUTRAL": "⚪"}
        v["inst_y"] = [(_im.get(t) or {}).get("d_inst_y", np.nan) for t in v.ticker]
        v["inst_l"] = [_icon.get((_im.get(t) or {}).get("label", ""), "") for t in v.ticker]
    except Exception:
        v["inst_y"], v["inst_l"] = np.nan, ""
    order = {a: k for k, a in enumerate(ACTIONS)}
    v = v.assign(_o=v.fx_action.map(order)).sort_values(["_o", "edge_6m", "ind_score"],
                                                        ascending=[True, False, False], na_position="last")
    cols = [
        "link", "name", "sub_industry", "ind_score", "price",
        "dist_stop_pct", "dist_ma50_pct", "dist_52w_pct",
        "signal", "fx_action", "inst_l", "inst_y",
        "val_rating", "val_band", "val_pct", "val_cagr_base", "val_note",
        "state_l", "flags", "fx_anchor_date", "fx_anchor_volx", "fx_days",
        "fx_retention", "supply_l", "trigger", "fx_stop", "fx_stop_pct",
        "shares", "buy_value", "sized_by", "fx_t1", "fx_t2",
        "edge_3m", "edge_6m", "edge_years", "fx_dist_hi", "fx_adx", "fx_bbp", "fx_rsi",
    ]
    st.caption(f"Showing **{len(v)}** stocks. Sorted by action, then the backtested 6-month edge of the row's cell.")
    st.dataframe(
        v[cols],
        column_config={
            "link": st.column_config.LinkColumn("Ticker", display_text=r"https://www\.screener\.in/company/(.*?)/"),
            "name": "Name", "sub_industry": "Sub-industry",
            "ind_score": st.column_config.ProgressColumn("Industry", min_value=0, max_value=100, format="%d"),
            "price": st.column_config.NumberColumn("Price", format="₹ %.2f"),
            "dist_stop_pct": st.column_config.NumberColumn("vs Stop", format="%+.1f%%",
                                                            help="% distance of current price above (+) or below (−) the technical stop-loss. Negative means price is already below the stop."),
            "dist_ma50_pct": st.column_config.NumberColumn("vs MA50", format="%+.1f%%",
                                                            help="% distance of current price above (+) or below (−) its 50-day moving average."),
            "dist_52w_pct": st.column_config.NumberColumn("vs 52W High", format="%+.1f%%",
                                                           help="% distance of current price from its 52-week high (0% = at the high)."),
            "fx_action": "Action", "state_l": "Event state",
            "inst_l": st.column_config.TextColumn("Inst.", help="🟢 FII + DII accumulating (≥ +1 pp QoQ or ≥ +2 pp YoY) · 🔴 distributing · "
                                                  "⚪ steady. Quarterly filings; context, not part of the action"),
            "inst_y": st.column_config.NumberColumn("FII+DII 1y", format="%+.1f pp", help="Change in FII + DII holding over a year"),
            "flags": st.column_config.TextColumn("Flags", help="⭐ absorption · 🪤 shakeout · ↗️ gap on the anchor day · "
                                                 "dist. = distribution (not a warning in the backtest)"),
            "fx_anchor_date": "Anchor", "fx_anchor_volx": st.column_config.NumberColumn("Vol ×", format="%.1f×"),
            "fx_days": st.column_config.NumberColumn("Days", format="%d"),
            "fx_retention": st.column_config.NumberColumn("Retention", format="%.0f%%",
                                                          help="Share of closes since the anchor at/above the close before it"),
            "supply_l": "Supply",
            "trigger": st.column_config.NumberColumn("Trigger (close >)", format="₹ %.2f",
                                                     help="Top of the overhead shelf -- buy on a close above it"),
            "fx_stop": st.column_config.NumberColumn("Stop", format="₹ %.2f"),
            "fx_stop_pct": st.column_config.NumberColumn("Stop %", format="%.1f%%"),
            "shares": st.column_config.NumberColumn("Shares", format="%d"),
            "buy_value": st.column_config.NumberColumn("Buy ₹", format="₹ %.0f"),
            "sized_by": st.column_config.TextColumn("Sized by", help="stop risk = a stop-out loses your risk per trade "
                                                    "(½ = halved, stop > 10%); slot cap = capital ÷ max positions"),
            "fx_t1": st.column_config.NumberColumn("T1 2R (ref)", format="₹ %.2f",
                                                   help="Reference only: exiting at 2R halved returns in the backtest"),
            "fx_t2": st.column_config.NumberColumn("T2 shelf", format="₹ %.2f"),
            "edge_3m": st.column_config.NumberColumn("Edge 3m", format="%+.1f pp"),
            "edge_6m": st.column_config.NumberColumn("Edge 6m", format="%+.1f pp",
                                                     help="Median 6-month return vs the typical stock for this cell, 2016-26"),
            "edge_years": "Years +",
            "fx_adx": st.column_config.NumberColumn("ADX", format="%.0f"),
            "fx_bbp": st.column_config.NumberColumn("BB width pct", format="%.0f",
                                                    help="Bollinger width percentile vs the stock's last year (< 20 = squeeze)"),
            "fx_rsi": st.column_config.NumberColumn("RSI", format="%.0f"),
            "fx_dist_hi": st.column_config.NumberColumn("From high", format="%+.1f%%",
                                                        help="Distance from the 252-day CLOSING high -- the number Gate 1 "
                                                             "tests (within 15%). The scanner's '% from 52W High' uses "
                                                             "intraday highs, so the two can differ by a point or two."),
            "signal": st.column_config.TextColumn("Signal", help=SIGNAL_HELP),
            "val_rating": st.column_config.TextColumn("Valuation", help="Mechanical v4.1 verdict at today's price "
                                                      "(BUY / ACCUMULATE / HOLD / REDUCE / SELL) + quality: ✅ clean · "
                                                      "🟡 caution · 🟠 weak · 🔴 unreliable"),
            "val_note": st.column_config.TextColumn("Val note", help="Main reason behind the quality grade and which "
                                                    "way it bends the verdict (may be too optimistic / too harsh), "
                                                    "or why the stock is not valued"),
            "val_band": st.column_config.TextColumn(
                "Val range", help="CMP percentile as a range across the assumptions the data cannot settle "
                                  "(hurdle, own-vs-peer weight, peer multiple ± its standard error, 3y/5y/TTM "
                                  "growth). A rating shown as BUY–HOLD means the range spans those ratings."),
            "val_pct": st.column_config.NumberColumn("Val pct", format="P%.0f",
                                                     help="Where today's price sits in the stock's own fair-value "
                                                          "distribution (P50 = fair, <P20 BUY zone, >P88 SELL zone)"),
            "val_cagr_base": st.column_config.NumberColumn("3y CAGR (base)", format="%+.0f%%",
                                                           help="Implied 3-year return at today's price if the base "
                                                                "scenario plays out, incl. dividend yield; compare with "
                                                                "the 12% hurdle"),
        },
        height=520, use_container_width=True, hide_index=True,
    )
    _dl_cols = ["ticker"] + [c for c in cols if c not in ("link", "ticker")]
    st.download_button("⬇️ Download CSV", v[_dl_cols].to_csv(index=False).encode(), "technicals.csv", "text/csv",
                       key="tx_dl")


def _norm_nse_ticker(raw: str) -> str:
    s = str(raw or "").strip().upper()
    if not s or s.endswith((".NS", ".BO")) or s.startswith("^"):
        return s
    return s + ".NS"


def _deep_dive(df: pd.DataFrame):

    st.markdown("#### 🧾 Single-stock report (v6 format)")
    mkt = st.radio("Market", ["🇮🇳 India (NSE)", "🇺🇸 US"], horizontal=True, key="tx_mkt", label_visibility="collapsed")
    if mkt.endswith("US"):
        return _deep_dive_us()
    st.caption("Fresh 2-year daily bars, same logic as the table. While NSE is open, today's candle is treated as "
               "provisional and left out.")
    core_df = df[df["extended"] != True] if (df is not None and "extended" in df.columns) else df
    tickers = sorted(core_df["ticker"].dropna().unique().tolist()) if core_df is not None and "ticker" in core_df else []
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        pick = st.selectbox("Stock", tickers, index=None, placeholder="Choose a stock", key="tx_pick")
        custom = st.text_input("…or any NSE symbol", value="", key="tx_custom", placeholder="e.g. ASIANHOTNR")
    # a stock you hold: fill buy price and date from My Book, once per stock picked (your edits then stand)
    t0 = _norm_nse_ticker(custom.strip().upper() or pick)
    mine = _holding(t0, set(tickers))
    for k, v in (("tx_held", False), ("tx_hp", 0.0), ("tx_hd", None)):
        st.session_state.setdefault(k, v)                   # defaults live here, so filling them in raises no widget warning
    if t0 and st.session_state.get("tx_filled_for") != t0:
        st.session_state["tx_filled_for"] = t0
        if mine:
            from utils import positions as _pos
            _fac, _ = _pos.split_factor_for_position(mine, t0)
            st.session_state["tx_held"] = True
            st.session_state["tx_hp"] = round(float(mine.get("entry_price") or 0.0) * _fac, 2)
            try:
                st.session_state["tx_hd"] = pd.Timestamp(mine["entry_date"]).date() if mine.get("entry_date") else None
            except Exception:
                st.session_state["tx_hd"] = None
        else:
            st.session_state["tx_held"], st.session_state["tx_hp"], st.session_state["tx_hd"] = False, 0.0, None
    with c2:
        held = st.toggle("I hold it", key="tx_held")
    with c3:
        hp = st.number_input("Bought @", min_value=0.0, step=1.0, key="tx_hp", disabled=not held)
    with c4:
        hd = st.date_input("Since", key="tx_hd", disabled=not held)
    risk = st.number_input("Risk per trade (₹)", min_value=500.0, value=10000.0, step=500.0, key="tx_riskrs")
    if mine:
        from utils import positions as _pos
        _fac, _basis = _pos.split_factor_for_position(mine, t0)
        _adj_ep = float(mine.get("entry_price") or 0) * _fac
        _sl_adj = float(mine["stop_loss"]) * _fac if mine.get("stop_loss") else 0.0
        _split_txt = f" (split-adjusted ×{_fac:.3g})" if _fac != 1.0 else ""
        st.caption(f"📒 In My Book: {mine.get('quantity') or '?'} shares @ ₹{_adj_ep:,.2f}{_split_txt} since "
                   f"{str(mine.get('entry_date') or '?')[:10]}" + (f", stop ₹{_sl_adj:,.2f}" if _sl_adj else
                   ", no stop set") + ". Filled in above.")
    t = _norm_nse_ticker(custom.strip().upper() or pick)
    if not t:
        return
    if held and (hp <= 0 or hd is None):
        st.warning("Held mode needs the buy price and date (the exit rule tracks the peak since then).")
        return
    live_src = "yahoo" if st.session_state.get("_loaded_live_mode") else None
    key = (t, hp if held else None, str(hd) if held else None, risk, live_src)
    if st.button(f"Run report for {t.replace('.NS', '').replace('.BO', '')}", key="tx_run", type="primary"):
        st.session_state["tx_last"] = key
    if st.session_state.get("tx_last") != key:
        return
    with st.spinner("Fetching 2 years of daily bars…"):
        h = _history_live(t) if live_src == "yahoo" else _history(t)
    if h is None or h.empty:
        st.error(f"No data for {t}.")
        return
    _is_ext = t.endswith(".NS") and t not in set(tickers)
    _visual_report(h, t.replace(".NS", ""), hp if held else None, hd if held else None, risk, extended=_is_ext)
    if held and mine:
        _update_holding(h, mine, hp, hd, risk)


@st.cache_data(ttl=3600, show_spinner=False)
def _sp500_names(mtime: float) -> dict:
    sp = pd.read_csv("data/sp500_tickers.csv")
    return dict(zip(sp.iloc[:, 0].astype(str), sp.get("name", sp.iloc[:, 0]).astype(str)))


def _deep_dive_us():
    """The same v6 technical read for a US stock: chart structure, volume anchor, supply, stop, size and exit rule."""
    st.caption("Fresh 2-year daily bars from Yahoo. While New York is open, today's candle is provisional and left out. "
               "⚠️ The v6 rules were built and backtested on NSE stocks only; on US stocks they are untested. The volume "
               "anchor needs about $0.6M traded that day (the ₹5 cr rule), so nearly every US day qualifies on value. "
               "Valuation below uses SEC filings (refreshed weekly) and the same v4.1 model with S&P 500 peers and a 9% "
               "hurdle; it too was calibrated on Indian stocks.")
    try:
        names = _sp500_names(os.path.getmtime("data/sp500_tickers.csv"))
    except Exception:
        names = {}
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        pick = st.selectbox("Stock", sorted(names), index=None, placeholder="Choose an S&P 500 stock", key="tx_us_pick",
                            format_func=lambda t: f"{t} — {names.get(t, '')}")
        custom = st.text_input("…or any US symbol", value="", key="tx_us_custom", placeholder="e.g. PLTR, ARM")
    with c2:
        held = st.toggle("I hold it", key="tx_us_held")
    with c3:
        hp = st.number_input("Bought @ ($)", min_value=0.0, step=0.5, key="tx_us_hp", disabled=not held)
    with c4:
        hd = st.date_input("Since", value=None, key="tx_us_hd", disabled=not held)
    risk = st.number_input("Risk per trade ($)", min_value=10.0, value=500.0, step=50.0, key="tx_us_risk")
    t = re.sub(r"\.([A-Z])$", r"-\1", (custom.strip().upper() or pick or "").replace(".NS", ""))   # Yahoo: BRK-B, not BRK.B
    if not t:
        return
    if held and (hp <= 0 or hd is None):
        st.warning("Held mode needs the buy price and date (the exit rule tracks the peak since then).")
        return
    key = ("US", t, hp if held else None, str(hd) if held else None, risk)
    if st.button(f"Run report for {t}", key="tx_us_run", type="primary"):
        st.session_state["tx_us_last"] = key
    if st.session_state.get("tx_us_last") != key:
        return
    with st.spinner("Fetching 2 years of daily bars…"):
        h = _history_live(t)
    if h is None or h.empty:
        st.error(f"No data for {t}.")
        return
    h = h.copy()
    h.attrs["currency"] = "USD"
    _visual_report(h, t, hp if held else None, hd if held else None, risk)


def _holding(t, known: set):
    """The active My Book position for ticker `t` (old spellings resolved), or None."""
    if not t:
        return None
    try:
        from utils import positions as pos
        for p in pos.get_all_positions():
            if p.get("status") == "active" and (p.get("ticker") == t or (pos.resolve_ticker(p.get("ticker", ""), known) or "") == t):
                return p
    except Exception:
        return None
    return None


def _update_holding(h: pd.DataFrame, mine: dict, hp: float, hd, risk: float):
    """Write what the report found back to My Book: the buy details as entered, and the exit level as the stop."""
    from utils import positions as pos
    from utils.forensics import plan
    P = plan(h, hp, hd, risk)
    if not P:
        return
    r = P["r"]
    close, lvl = float(r["close"]), max(float(r["ma50"]), float(P["trail"]))
    old_stop = float(mine.get("stop_loss") or 0)
    changes, notes = {}, []
    if abs(float(mine.get("entry_price") or 0) - hp) > 0.005 or str(mine.get("entry_date") or "")[:10] != str(hd)[:10]:
        changes.update(entry_price=round(hp, 2), entry_date=str(hd)[:10])
        notes.append(f"buy price ₹{hp:,.2f} and date {hd}")
    if lvl < close and abs(lvl - old_stop) > 0.005:
        notes.append(f"stop ₹{old_stop:,.2f} → ₹{lvl:,.2f} (higher of the 50-day average and 15% below the peak since you bought)"
                     if old_stop else f"stop ₹{lvl:,.2f} (higher of the 50-day average and 15% below the peak since you bought)")
    st.markdown("#### 📒 Update My Book")
    if P["action"] == "EXIT":
        st.error(f"The exit rule fired on {pd.Timestamp(P['fired']):%d %b %Y}. If you sell, tick **Sold** on My Book "
                 "(or sync your broker) so it reaches the Journal.")
    if lvl < close and old_stop > lvl:
        notes = [n for n in notes if not n.startswith("stop")]
        st.caption(f"Your stop ₹{old_stop:,.2f} is already above this report's exit level ₹{lvl:,.2f}; it is left as it is.")
    if not notes:
        st.caption("My Book already matches this report.")
        return
    c = st.columns([3, 1])
    c[0].caption("Will save: " + "; ".join(notes) + ". Quantity, target and notes are not touched.")
    if c[1].button("💾 Update holding", key="tx_update", type="primary", use_container_width=True):
        if any(n.startswith("stop") for n in notes):
            changes["stop_loss"] = round(lvl, 2)
        pos.update_position(mine["ticker"], **changes)
        st.session_state["book_ver"] = st.session_state.get("book_ver", 0) + 1
        st.toast(f"{str(mine['ticker']).replace('.NS', '')} updated in My Book", icon="📒")
        st.rerun()


def _trade_plan_card(P: dict, name: str, h: pd.DataFrame, us: bool = False, extended: bool = False):
    """Entry, stop, targets and value zones in one place, with the dates the data is from."""
    from utils import trade_plan
    from utils.valuation import _series, value_stock
    from utils.valuation_view import ctx as _val_ctx, ctx_us as _val_ctx_us
    r, val, c = P["r"], None, None
    try:
        c = _val_ctx_us() if us else _val_ctx(extended=extended)
        if not c.annual.empty:
            hh = h[h.index > h.index[-1] - pd.Timedelta(days=365)] if isinstance(h.index, pd.DatetimeIndex) else h.tail(252)
            val = value_stock(name, c, price=float(r["close"]), hi52=float(hh["High"].max()), lo52=float(hh["Low"].min()))
    except Exception:
        val = None
    tp = trade_plan.build(P, val)
    st.markdown("#### 🎯 Trade plan")
    (st.success if tp["valid"] and P["action"] == "BUY NOW" else st.info if tp["valid"] else st.error)(
        f"**{P['action']}** — {tp['summary']}" if tp["summary"] else f"**{P['action']}** — no tradable plan today.")
    for f in tp["flags"]:
        st.warning(f)
    st.dataframe(tp["levels"], hide_index=True, use_container_width=True,
                 column_config={"level": "Level", "price": st.column_config.NumberColumn("Price", format=("$" if us else "₹") + "%.2f"),
                                "vs_now": st.column_config.NumberColumn("vs now", format="%+.1f%%"),
                                "source": "From", "note": "What it means"})
    fresh = [f"prices to {pd.Timestamp(h.index[-1]):%d %b %Y}"]
    try:
        if us:
            q = _series(c.quarterly, name, "np") if c is not None else pd.Series(dtype=float)
            if len(q):
                age = (pd.Timestamp.today() - q.index.max()).days
                fresh.append(f"latest US quarterly result {q.index.max():%b %Y}" + (" ⚠️ over 6 months old" if age > 200 else ""))
            from utils import fund_paths
            us_qf = fund_paths.resolve("data/us_fundamentals_quarterly.csv")
            if os.path.exists(us_qf):
                fresh.append(f"SEC filings refreshed {pd.Timestamp(os.path.getmtime(us_qf), unit='s'):%d %b %Y}")
        else:
            t = name if name.endswith((".NS", ".BO")) else name + ".NS"
            q = _series(c.quarterly, t, "np")
            if len(q):
                age = (pd.Timestamp.today() - q.index.max()).days
                fresh.append(f"latest quarterly result {q.index.max():%b %Y}" + (" ⚠️ over 6 months old" if age > 200 else ""))
            from utils import fund_paths
            in_qf = fund_paths.resolve("data/fundamentals_quarterly.csv")
            if os.path.exists(in_qf):
                fresh.append(f"fundamentals file refreshed {pd.Timestamp(os.path.getmtime(in_qf), unit='s'):%d %b %Y}")
    except Exception:
        pass
    st.caption("Data used: " + " · ".join(fresh) + ". Targets and stops are computed levels, not advice.")
    return val


def _visual_report(h: pd.DataFrame, name: str, held_price, held_date, risk: float, extended: bool = False):
    from utils.forensics import SUPPLY_LABEL, _p, events, plan, report
    from utils.forensics_chart import levels_ladder, price_chart, retention_chart

    P = plan(h, held_price, held_date, risk)
    if not P:
        st.error(f"{name}: fewer than 60 closed sessions -- aborted.")
        return
    r = P["r"]
    dc = P["df"]
    us = P.get("currency") == "USD"
    cur = "$" if us else "₹"
    cfg = {"displayModeBar": False}
    if P["provisional"]:
        st.caption(f"⏱️ {'New York' if us else 'NSE'} is open: today's candle is provisional and left out.")

    # headline tiles -- the report's decision block at a glance
    k = st.columns(6)
    k[0].metric("Action", P["action"], help="Gate 1 + GO + no near overhang = BUY NOW; GO under an overhang = "
                "BUY ON TRIGGER; held mode applies the exit rule")
    k[1].metric("Event", r["state"], f"day {r['days']}" if "days" in r else None, delta_color="off")
    k[2].metric("Supply", SUPPLY_LABEL.get(r["supply"], r["supply"]))
    if np.isfinite(r["stop"]) and 0 < r["stop"] < r["close"]:
        k[3].metric("Stop", f"{cur}{r['stop']:,.2f}", f"-{r['stop_pct']:.1f}%" + (" · ½ size" if r["half_size"] else ""),
                    delta_color="inverse")
    else:
        k[3].metric("Stop", "—", "stop invalid (<= 0)" if not (np.isfinite(r["stop"]) and r["stop"] > 0) else "price below stop", delta_color="off")
    if P["trigger"]:
        k[4].metric("Trigger (close >)", f"{cur}{P['trigger']:,.2f}", f"{(P['trigger'] / r['close'] - 1) * 100:+.1f}%",
                    delta_color="off")
    else:
        k[4].metric("Trend (Gate 1)", "PASS" if r["gate1"] else "FAIL",
                    f"{r['dist_ma50']:+.1f}% vs MA50", delta_color="off")
    _entry_px = r.get("entry", r["close"])
    k[5].metric("Size", f"{P['shares']:,} sh", f"{cur}{P['shares'] * _entry_px:,.0f}", delta_color="off",
                help=f"Risk {cur}{risk:,.0f} ÷ ({'trigger' if P['trigger'] else 'close'} − stop), halved when the stop is wider than 10%")

    val = _trade_plan_card(P, name, dc, us=us, extended=extended)
    st.plotly_chart(price_chart(P, name), use_container_width=True, config=cfg)

    if P.get("currency", "INR") == "INR":                        # NSE delivery %, surveillance, next results (utils/nse_data.py)
        try:
            from utils import nse_data as _nd
            _t = name if name.endswith((".NS", ".BO")) else name + ".NS"
            _dp = _nd.delivery_panel(20)
            _bits = []
            if len(_dp) and _t in _dp.columns and _dp[_t].notna().any():
                _last_del = _dp[_t].dropna()
                _bits.append(f"Delivery {_last_del.iloc[-1]:.0f}% on {pd.Timestamp(_last_del.index[-1]):%d %b} (20-day avg {_last_del.mean():.0f}%)")
            _sv, _ev = _nd.surveillance_map().get(_t), _nd.upcoming_events(30).get(_t)
            if _sv:
                _bits.append(f"⚠️ {_sv}")
            if _ev:
                _bits.append(f"📅 {_ev}")
            if _bits:
                st.caption(" · ".join(_bits))
        except Exception:
            pass
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(levels_ladder(P), use_container_width=True, config=cfg)
    with c2:
        fr = retention_chart(P)
        if fr is not None:
            st.plotly_chart(fr, use_container_width=True, config=cfg)
        # the report's qualitative lines, as short cards
        sig = r.get("signature")
        notes = []
        if P["trig_a"]:
            notes.append(f"**Trigger A** · {P['trig_a']}")
        if P["trig_b"]:
            notes.append(f"**Trigger B** · {P['trig_b']}")
        if sig:
            notes.append(f"**Pullback** · {sig.lower()} -- down days {_p(r.get('dn_vs_med'), 2)}× median volume"
                         + (" ⭐ (best backtested cell with GO)" if sig == "ABSORPTION" else "")
                         + (" (not a warning in the backtest)" if sig == "DISTRIBUTION" else ""))
        if r.get("shakeout"):
            notes.append(f"**🪤 Shakeout** · {pd.Timestamp(r['shakeout_date']):%d %b}: broke {cur}{_p(r['shakeout_level'])}"
                         f" intraday, closed back above on {_p(r['shakeout_volx'], 2)}× volume")
        _sq = " (squeeze)" if np.isfinite(r.get("bbp", np.nan)) and r["bbp"] < 20 else ""
        notes.append(f"**Lens** · RSI {_p(r.get('rsi'), 0)} · ADX {_p(r.get('adx'), 0)} (+DI {_p(r.get('pdi'), 0)} / −DI {_p(r.get('ndi'), 0)}) · "
                     f"Bollinger width {_p(r.get('bbp'), 0)} pct{_sq}")
        ex = (f"**Exit rule** · MA50 {cur}{_p(r['ma50'])} ({r['dist_ma50']:+.1f}%), closes below {r['closes_below_ma50']}/2"
              + (f" · trail {cur}{P['trail']:,.2f} (0.85 × peak {cur}{P['peak']:,.2f})" if P["held"] else "")
              + (f" · **FIRED {pd.Timestamp(P['fired']):%d %b %Y}**" if P["fired"] is not None else " · not fired"))
        notes.append(ex)
        st.markdown("\n\n".join(f"- {n}" for n in notes))

    ev = events(dc)
    if len(ev):
        st.markdown(f"**Volume events, last 60 sessions** (≥ 3× median volume, ≥ {'$0.6M' if us else '₹5 cr'}) -- the "
                    "highest-volume one is the anchor")
        show = ev.sort_values("date", ascending=False).assign(
            date=lambda d_: pd.to_datetime(d_["date"]).dt.strftime("%d %b %Y"),
            anchor=np.where(ev.sort_values("date", ascending=False).anchor, "⚓", ""),
            turnover_cr=lambda d_: d_["turnover_cr"] * (10 if us else 1),
            retention=lambda d_: d_["retention"] * 100)
        st.dataframe(show[["anchor", "date", "vol_ratio", "clv", "direction", "turnover_cr", "close", "low", "days",
                           "retention", "state"]],
                     column_config={"anchor": "", "date": "Date",
                                    "vol_ratio": st.column_config.NumberColumn("Vol ×", format="%.1f×"),
                                    "clv": st.column_config.NumberColumn("CLV", format="%.2f"),
                                    "direction": "Dir",
                                    "turnover_cr": st.column_config.NumberColumn("Turnover", format="$%.0fM" if us else "₹%.0f cr"),
                                    "close": st.column_config.NumberColumn("Close", format=cur + "%.2f"),
                                    "low": st.column_config.NumberColumn("Low", format=cur + "%.2f"),
                                    "days": st.column_config.NumberColumn("Days ago", format="%d"),
                                    "retention": st.column_config.NumberColumn("Retention", format="%.0f%%"),
                                    "state": "Graded as anchor"},
                     hide_index=True, use_container_width=True)
    with st.expander("🧾 Full v6 report (text)"):
        st.code(report(dc, name, P=P), language=None)
    st.divider()
    from utils.valuation_view import render_stock
    hh = dc[dc.index > dc.index[-1] - pd.Timedelta(days=365)] if isinstance(dc.index, pd.DatetimeIndex) else dc.tail(252)
    render_stock(name, float(r["close"]), hi52=float(hh["High"].max()), lo52=float(hh["Low"].min()),
                 tech_buy=P["r"]["action"] == "BUY NOW" and not P["held"],
                 below_both=bool(r["close"] < r["ma50"] and r["close"] < r["ma200"]),
                 market="US" if us else "IN", precomputed=val, extended=extended)
