"""
🔬 Technicals tab -- chart-prompt v6 structural forensics for every stock.

Reads the fx_* columns the daily engine writes (utils/forensics.snapshot) and
ranks them by the backtested edge of their cell (analysis/
forensics_gate_study.csv). A single-stock deep dive re-runs the same logic
on fresh 2-year bars and prints the v6 report.
"""
from __future__ import annotations

import os

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


def _cell(row) -> str:
    """Most specific backtested cell for this row."""
    if row.get("fx_gate1") is not True:
        return ""
    s, sup, sig = row.get("fx_state"), row.get("fx_supply"), row.get("fx_signature")
    if s == "GO":
        if sig == "ABSORPTION" and sup != "NEAR OVERHANG":
            return "BUY NOW + absorption"
        if row.get("fx_shakeout") is True:
            return "v6 GO + shakeout (v6-only probe)"
        return "BUY ON TRIGGER (GO + near overhang)" if sup == "NEAR OVERHANG" else "BUY NOW (GO, no near overhang)"
    return s if isinstance(s, str) else ""


@st.cache_data(ttl=900, show_spinner=False)
def _history(ticker: str) -> pd.DataFrame:
    from utils.yf_safe import safe_history
    h = safe_history(ticker, period="2y")
    if h is not None and not h.empty and h.index.tz is not None:
        h.index = h.index.tz_localize(None)
    return h


def render(df: pd.DataFrame):
    from utils.ui_components import page_header
    st.markdown(page_header("🔬 Technicals", "Structural forensics: volume anchors, absorption, shakeouts, supply "
                            "shelves, stops -- the chart-prompt v6 gates, backtested"), unsafe_allow_html=True)
    with st.expander("ℹ️ What each flag is worth (backtest)", expanded=False):
        st.markdown(HELP)

    if "fx_state" not in df.columns:
        st.info("The loaded snapshot predates the Technicals fields -- they arrive with the next daily engine run. "
                "The single-stock deep dive below works now.")
    else:
        _table(df)
    st.divider()
    _deep_dive(df)


def _table(df: pd.DataFrame):
    from utils.breakout_tags import load_industry_scores
    from utils.nifty1000_list import SUB_INDUSTRY_MAP

    d = df.copy()
    d["sub_industry"] = d["ticker"].map(SUB_INDUSTRY_MAP)
    d["ind_score"] = pd.to_numeric(d["sub_industry"].map(load_industry_scores()), errors="coerce")
    for c in ("fx_gate1", "fx_shakeout", "fx_half_size", "fx_gap"):
        d[c] = d[c].map(lambda x: True if x is True or x == "True" or x == 1 else False) if c in d else False
    price = pd.to_numeric(d.get("price", d.get("currentPrice")), errors="coerce")
    d["price"] = price

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
    v = d
    if acts:
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
    risk_ps = (v.price - stop).where(lambda x: x > 0)
    risk_rs = cap * riskp / 100 * np.where(v.fx_half_size, 0.5, 1.0)
    slot = cap / int(maxpos)
    by_risk = np.floor(risk_rs / risk_ps)
    by_slot = np.floor(slot / v.price)
    v["shares"] = np.fmin(by_risk, by_slot).fillna(0).astype(int)
    v["sized_by"] = np.where(by_risk <= by_slot, "stop risk" + np.where(v.fx_half_size, " (½)", ""), "slot cap")
    v["buy_value"] = (v.shares * v.price).round(0)
    v["fx_retention"] = pd.to_numeric(v.fx_retention, errors="coerce") * 100
    v["trigger"] = np.where(v.fx_supply == "NEAR OVERHANG", pd.to_numeric(v.fx_shelf_top, errors="coerce"), np.nan)
    v["state_l"] = v.fx_state.map(STATE_LABEL).fillna("")
    v["supply_l"] = v.fx_supply.map(SUPPLY_LABEL).fillna("")
    v["flags"] = (np.where(v.fx_signature == "ABSORPTION", "⭐ ", "") + np.where(v.fx_shakeout, "🪤 ", "")
                  + np.where(v.fx_gap, "↗️ gap ", "") + np.where(v.fx_signature == "DISTRIBUTION", "dist. ", ""))
    v["link"] = "https://www.screener.in/company/" + v.ticker.str.replace(".NS", "", regex=False) + "/"
    order = {a: k for k, a in enumerate(ACTIONS)}
    v = v.assign(_o=v.fx_action.map(order)).sort_values(["_o", "edge_6m", "ind_score"],
                                                        ascending=[True, False, False], na_position="last")
    cols = ["link", "name", "sub_industry", "ind_score", "price", "fx_action", "state_l", "flags", "fx_anchor_date",
            "fx_anchor_volx", "fx_days", "fx_retention", "supply_l", "trigger", "fx_stop", "fx_stop_pct",
            "shares", "buy_value", "sized_by", "fx_t1", "fx_t2", "edge_3m", "edge_6m", "edge_years",
            "fx_adx", "fx_bbp", "fx_rsi"]
    st.caption(f"Showing **{len(v)}** stocks. Sorted by action, then the backtested 6-month edge of the row's cell.")
    st.dataframe(
        v[cols],
        column_config={
            "link": st.column_config.LinkColumn("Ticker", display_text=r"https://www\.screener\.in/company/(.*?)/"),
            "name": "Name", "sub_industry": "Sub-industry",
            "ind_score": st.column_config.ProgressColumn("Industry", min_value=0, max_value=100, format="%d"),
            "price": st.column_config.NumberColumn("Price", format="₹ %.2f"),
            "fx_action": "Action", "state_l": "Event state",
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
        },
        height=520, use_container_width=True, hide_index=True,
    )
    st.download_button("⬇️ Download CSV", v[cols].to_csv(index=False).encode(), "technicals.csv", "text/csv",
                       key="tx_dl")


def _deep_dive(df: pd.DataFrame):

    st.markdown("#### 🧾 Single-stock report (v6 format)")
    st.caption("Fresh 2-year daily bars, same logic as the table. While NSE is open, today's candle is treated as "
               "provisional and left out.")
    tickers = sorted(df["ticker"].dropna().unique().tolist()) if "ticker" in df else []
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        pick = st.selectbox("Stock", tickers, index=None, placeholder="Choose a stock", key="tx_pick")
        custom = st.text_input("…or any NSE symbol", value="", key="tx_custom", placeholder="e.g. ASIANHOTNR")
    with c2:
        held = st.toggle("I hold it", value=False, key="tx_held")
    with c3:
        hp = st.number_input("Bought @", min_value=0.0, value=0.0, step=1.0, key="tx_hp", disabled=not held)
    with c4:
        hd = st.date_input("Since", value=None, key="tx_hd", disabled=not held)
    risk = st.number_input("Risk per trade (₹)", min_value=500.0, value=10000.0, step=500.0, key="tx_riskrs")
    t = custom.strip().upper() or pick
    if not t:
        return
    if not t.endswith(".NS") and not t.startswith("^"):
        t += ".NS"
    if held and (hp <= 0 or hd is None):
        st.warning("Held mode needs the buy price and date (the exit rule tracks the peak since then).")
        return
    key = (t, hp if held else None, str(hd) if held else None, risk)
    if st.button(f"Run report for {t.replace('.NS', '')}", key="tx_run", type="primary"):
        st.session_state["tx_last"] = key
    if st.session_state.get("tx_last") != key:
        return
    with st.spinner("Fetching 2 years of daily bars…"):
        h = _history(t)
    if h is None or h.empty:
        st.error(f"No data for {t}.")
        return
    _visual_report(h, t.replace(".NS", ""), hp if held else None, hd if held else None, risk)


def _visual_report(h: pd.DataFrame, name: str, held_price, held_date, risk: float):
    from utils.forensics import SUPPLY_LABEL, events, plan, report
    from utils.forensics_chart import levels_ladder, price_chart, retention_chart

    P = plan(h, held_price, held_date, risk)
    if not P:
        st.error(f"{name}: fewer than 60 closed sessions -- aborted.")
        return
    r = P["r"]
    cfg = {"displayModeBar": False}
    if P["provisional"]:
        st.caption("⏱️ NSE is open: today's candle is provisional and left out.")

    # headline tiles -- the report's decision block at a glance
    k = st.columns(6)
    k[0].metric("Action", P["action"], help="Gate 1 + GO + no near overhang = BUY NOW; GO under an overhang = "
                "BUY ON TRIGGER; held mode applies the exit rule")
    k[1].metric("Event", r["state"], f"day {r['days']}" if "days" in r else None, delta_color="off")
    k[2].metric("Supply", SUPPLY_LABEL.get(r["supply"], r["supply"]))
    if r["stop"] < r["close"]:
        k[3].metric("Stop", f"₹{r['stop']:,.2f}", f"-{r['stop_pct']:.1f}%" + (" · ½ size" if r["half_size"] else ""),
                    delta_color="inverse")
    else:
        k[3].metric("Stop", "—", "price below stop", delta_color="off")
    if P["trigger"]:
        k[4].metric("Trigger (close >)", f"₹{P['trigger']:,.2f}", f"{(P['trigger'] / r['close'] - 1) * 100:+.1f}%",
                    delta_color="off")
    else:
        k[4].metric("Trend (Gate 1)", "PASS" if r["gate1"] else "FAIL",
                    f"{r['dist_ma50']:+.1f}% vs MA50", delta_color="off")
    k[5].metric("Size", f"{P['shares']:,} sh", f"₹{P['shares'] * r['close']:,.0f}", delta_color="off",
                help=f"Risk ₹{risk:,.0f} ÷ (close − stop), halved when the stop is wider than 10%")

    st.plotly_chart(price_chart(P, name), use_container_width=True, config=cfg)

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
            notes.append(f"**Pullback** · {sig.lower()} -- down days {r['dn_vs_med']:.2f}× median volume"
                         + (" ⭐ (best backtested cell with GO)" if sig == "ABSORPTION" else "")
                         + (" (not a warning in the backtest)" if sig == "DISTRIBUTION" else ""))
        if r.get("shakeout"):
            notes.append(f"**🪤 Shakeout** · {pd.Timestamp(r['shakeout_date']):%d %b}: broke ₹{r['shakeout_level']:,.2f} "
                         f"intraday, closed back above on {r['shakeout_volx']:.2f}× volume")
        notes.append(f"**Lens** · RSI {r['rsi']:.0f} · ADX {r['adx']:.0f} (+DI {r['pdi']:.0f} / −DI {r['ndi']:.0f}) · "
                     f"Bollinger width {r['bbp']:.0f} pct" + (" (squeeze)" if r["bbp"] < 20 else ""))
        ex = (f"**Exit rule** · MA50 ₹{r['ma50']:,.2f} ({r['dist_ma50']:+.1f}%), closes below {r['closes_below_ma50']}/2"
              + (f" · trail ₹{P['trail']:,.2f} (0.85 × peak ₹{P['peak']:,.2f})" if P["held"] else "")
              + (f" · **FIRED {pd.Timestamp(P['fired']):%d %b %Y}**" if P["fired"] is not None else " · not fired"))
        notes.append(ex)
        st.markdown("\n\n".join(f"- {n}" for n in notes))

    ev = events(P["df"])
    if len(ev):
        st.markdown("**Volume events, last 60 sessions** (≥ 3× median volume, ≥ ₹5 cr) -- the highest-volume one is "
                    "the anchor")
        show = ev.assign(date=ev.date.dt.strftime("%d %b %Y"), anchor=np.where(ev.anchor, "⚓", ""),
                         retention=ev.retention * 100).sort_values("date", ascending=False)
        st.dataframe(show[["anchor", "date", "vol_ratio", "clv", "direction", "turnover_cr", "close", "low", "days",
                           "retention", "state"]],
                     column_config={"anchor": "", "date": "Date",
                                    "vol_ratio": st.column_config.NumberColumn("Vol ×", format="%.1f×"),
                                    "clv": st.column_config.NumberColumn("CLV", format="%.2f"),
                                    "direction": "Dir",
                                    "turnover_cr": st.column_config.NumberColumn("Turnover", format="₹%.0f cr"),
                                    "close": st.column_config.NumberColumn("Close", format="₹%.2f"),
                                    "low": st.column_config.NumberColumn("Low", format="₹%.2f"),
                                    "days": st.column_config.NumberColumn("Days ago", format="%d"),
                                    "retention": st.column_config.NumberColumn("Retention", format="%.0f%%"),
                                    "state": "Graded as anchor"},
                     hide_index=True, use_container_width=True)
    with st.expander("🧾 Full v6 report (text)"):
        st.code(report(h, name, P=P), language=None)
