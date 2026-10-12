"""
Valuation panels for the Technicals tab (mechanical v4.1, utils/valuation.py).

  table_columns(df)   adds val_rating / val_pct / val_cagr_base from
                      data/valuation_snapshot.csv, re-percentiled at today's price
  render_stock(...)   the single-stock valuation section that follows the
                      technical report: verdict tiles, fair-value histogram with
                      CMP and the entry/exit ladder, scenario build, multiple
                      model, 1C scorecard, gates, regimes, sensitivity, notes
  render_reference()  your own v4.1 runs (data/valuation_v41_reference.csv)
                      side by side with the mechanical rating
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.valuation import KNOWN_LIMITS, REFERENCE, load_context, rating_from_grid, value_stock

SNAPSHOT = "data/valuation_snapshot.csv"
RATING_ORDER = ["STRONG BUY", "BUY", "ACCUMULATE", "HOLD", "REDUCE", "SELL/AVOID", "STRONG SELL/AVOID"]
RATING_ICON = {"STRONG BUY": "🟢🟢", "BUY": "🟢", "ACCUMULATE": "🟩", "HOLD": "⚪", "REDUCE": "🟧",
               "SELL/AVOID": "🔴", "STRONG SELL/AVOID": "🔴🔴"}


@st.cache_data(ttl=3600, show_spinner=False)
def _snapshot(mtime: float) -> pd.DataFrame:
    return pd.read_csv(SNAPSHOT)


@st.cache_resource(ttl=6 * 3600, show_spinner=False)
def _ctx(key: tuple):
    return load_context()


def ctx():
    files = ("data/fundamentals_annual.csv.gz", "data/fundamentals_quarterly.csv.gz",
             "data/monthly_prices.csv.gz", "data/valuation_overrides.csv")
    return _ctx(tuple(os.path.getmtime(f) if os.path.exists(f) else 0 for f in files))


def _base_label(r: str) -> str:
    r = str(r)
    for lab in sorted(RATING_ORDER, key=len, reverse=True):
        if r.startswith(lab):
            return lab
    return ""


def table_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds val_rating (icon + verdict at today's price), val_bucket, val_pct, val_cagr_base, val_fv_p50."""
    out = df.copy()
    for c in ("val_rating", "val_bucket"):
        out[c] = ""
    out["val_pct"] = out["val_cagr_base"] = out["val_fv_p50"] = np.nan
    if not os.path.exists(SNAPSHOT):
        return out
    S = _snapshot(os.path.getmtime(SNAPSHOT)).set_index("ticker")
    tick = out["ticker"].str.replace(".NS", "", regex=False)
    price = pd.to_numeric(out.get("price", out.get("currentPrice")), errors="coerce")
    rat, buck, pct, cagr, p50 = [], [], [], [], []
    for t, p in zip(tick, price):
        if t not in S.index:
            rat.append(""); buck.append(""); pct.append(np.nan); cagr.append(np.nan); p50.append(np.nan)
            continue
        r = S.loc[t]
        if r.status != "OK":
            short = "out of scope" if r.status == "OUT OF SCOPE" else str(r.status).lower()
            rat.append(f"— {short}"); buck.append("NOT VALUED"); pct.append(np.nan); cagr.append(np.nan); p50.append(np.nan)
            continue
        unrated = str(r.rating).startswith(("UNRATED", "INDETERMINATE"))
        q, v = rating_from_grid(json.loads(r.grid), p if np.isfinite(p) else r.price, bool(r.gates_ok),
                                r.confidence, r.identity)
        lab = _base_label(v)
        if unrated:
            why = "multiple indeterminate" if str(r.rating).startswith("INDETERMINATE") else \
                str(r.rating).replace("UNRATED (", "").rstrip(")")
            rat.append(f"({lab}) · {why}"); buck.append("UNRATED")
        else:
            rat.append(f"{RATING_ICON.get(lab, '')} {v}"); buck.append(lab)
        pct.append(q); cagr.append(r.cagr_base * 100 if pd.notna(r.cagr_base) else np.nan); p50.append(r.p50)
    out["val_rating"], out["val_bucket"], out["val_pct"] = rat, buck, pct
    out["val_cagr_base"], out["val_fv_p50"] = cagr, p50
    return out


def snapshot_date() -> str:
    if not os.path.exists(SNAPSHOT):
        return ""
    S = _snapshot(os.path.getmtime(SNAPSHOT))
    return str(S["as_of"].iloc[0]) if "as_of" in S and len(S) else ""


# ----------------------------------------------------------------------------
# single stock
# ----------------------------------------------------------------------------
def _hist(res: dict) -> go.Figure:
    d, lad, price = res["dist"], res["ladder"], res["snapshot"]["price"]
    g = np.asarray(res["grid"])
    # density from the 1..99 grid (enough for a picture; the engine keeps the full N trials)
    edges = np.linspace(d["P5"] * 0.9, d["P95"] * 1.1, 60)
    probs = np.diff(np.interp(edges, g, np.arange(1, 100), left=0, right=100))
    fig = go.Figure(go.Bar(x=(edges[1:] + edges[:-1]) / 2, y=probs, marker_color="#2a78d6", opacity=0.75,
                           hovertemplate="₹%{x:,.0f}: %{y:.1f}% of trials<extra></extra>", showlegend=False))
    zones = [(edges[0], lad["deep_add"], "rgba(27,175,122,0.16)", "deep add ≤P10"),
             (lad["deep_add"], lad["accumulate"], "rgba(27,175,122,0.08)", "accumulate"),
             (lad["fair_hi"], lad["trim"], "rgba(235,104,52,0.08)", "trim"),
             (lad["trim"], edges[-1], "rgba(227,73,72,0.14)", "exit >P90")]
    for x0, x1, col, lab in zones:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=col, line_width=0, annotation_text=lab,
                      annotation_position="top left", annotation_font=dict(size=10, color="#52514e"))
    for lab, x, dash in (("P10", d["P10"], "dot"), ("P50", d["P50"], "dash"), ("P90", d["P90"], "dot")):
        fig.add_vline(x=x, line=dict(color="#52514e", dash=dash, width=1),
                      annotation_text=f"{lab} ₹{x:,.0f}", annotation_position="bottom right", annotation_font=dict(size=10))
    fig.add_vline(x=price, line=dict(color="#0b0b0b", width=3),
                  annotation_text=f"<b>CMP ₹{price:,.0f} · P{d['cmp_pct']:.0f}</b>", annotation_position="top right")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), bargap=0.05,
                      title=dict(text="<b>Fair value per share</b> (present value, 3-year horizon) · "
                                      f"{d['n']:,} trials", x=0, font=dict(size=14)),
                      xaxis=dict(title="₹ per share", range=[edges[0], max(edges[-1], price * 1.05)]),
                      yaxis=dict(title="% of trials", gridcolor="rgba(128,128,128,0.15)"))
    return fig


def render_stock(ticker: str, price: float, hi52: float | None = None, lo52: float | None = None):
    st.markdown("#### 💰 Valuation — v4.1, mechanical")
    c = ctx()
    if c.annual.empty:
        st.info("Valuation inputs are not in the repo yet -- run the Ranking / Rotation workflow with "
                "script = fundamentals (about 40 minutes), then reload.")
        return
    res = value_stock(ticker, c, price=price, hi52=hi52, lo52=lo52)
    if res.get("status") != "OK":
        st.warning(f"**{res.get('status')}** — {res.get('reason', '')}. Being out of scope is a property of the "
                   "business, not a criticism of it.")
        return
    d, snap, m = res["dist"], res["snapshot"], res["multiple"]
    k = st.columns(6)
    lab = res["rating"]
    k[0].metric("Valuation", lab if len(lab) < 22 else lab[:21] + "…", res.get("provisional") or None, delta_color="off",
                help="v4.1 verdict ladder on the CMP percentile: <P20 BUY · P20-40 ACCUMULATE · P40-70 HOLD · "
                     "P70-88 REDUCE · >P88 SELL/AVOID. Prints only when Gates A-F pass; otherwise UNRATED with the "
                     "provisional verdict shown underneath.")
    k[1].metric("CMP percentile", f"P{d['cmp_pct']:.0f}", res["label"], delta_color="off")
    k[2].metric("Fair value P50", f"₹{d['P50']:,.0f}", f"P10 ₹{d['P10']:,.0f} · P90 ₹{d['P90']:,.0f}", delta_color="off")
    k[3].metric("3y CAGR at CMP (base)", f"{res['cagr']['base']:+.1%}",
                f"hurdle {res['hurdle']:.0%} · bear {res['cagr']['bear']:+.0%} · bull {res['cagr']['bull']:+.0%}",
                delta_color="off", help="(undiscounted Y3 value ÷ CMP)^(1/3) − 1 + dividend yield, per scenario")
    k[4].metric("Identity", res["identity"], f"confidence {res['confidence']}", delta_color="off")
    k[5].metric(f"Multiple ({m['kind']})", f"{m['spot']:.1f}× spot", f"bear {m['bear']:.1f} · base {m['base']:.1f} · "
                f"bull {m['bull']:.1f}", delta_color="off")
    if res["gates"]["D"]["triggered"]:
        st.warning(f"**This is a multiple-de-rating call.** {res['gates']['D']['detail']}")
    elif res["gates"]["D"]["near"]:
        st.caption(f"Gate D near-trigger: {res['gates']['D']['detail']}")
    if res.get("cycle", {}).get("statement"):
        st.info(f"**Cycle position (outside the simulation):** {res['cycle']['statement']}")
    st.plotly_chart(_hist(res), use_container_width=True, config={"displayModeBar": False})

    c1, c2 = st.columns(2)
    with c1:
        lv = res["lever"]
        kind = "BVPS" if res["identity"] == "BFSI" else "EPS"
        sc = pd.DataFrame({s.title(): [f"{x:,.1f}" for x in lv[s]] + [f"{m[s]:.1f}×", f"₹{res['blends'][s]:,.0f}",
                                                                   f"{res['cagr'][s]:+.1%}", f"{p:.0%}"]
                           for s, p in zip(("bear", "base", "bull"), res["probs"])},
                          index=[f"{kind} Y1", f"{kind} Y2", f"{kind} Y3", "Multiple", "Blend (PV)", "3y CAGR at CMP",
                                 "Probability"])
        st.markdown("**Scenario build**")
        st.dataframe(sc, use_container_width=True)
        si = res["scenario_inputs"]
        if "growth" in si:
            st.caption(f"Revenue growth bear/base/bull {si['growth']['bear']:+.0%} / {si['growth']['base']:+.0%} / "
                       f"{si['growth']['bull']:+.0%} a year; terminal net margin {si['terminal_margin']['bear']:.1%} / "
                       f"{si['terminal_margin']['base']:.1%} / {si['terminal_margin']['bull']:.1%} (TTM "
                       f"{si['margin_ttm']:.1%}, worst demonstrated {si['worst_margin']:.1%}). {res['path_note']}.")
        else:
            st.caption(f"ROE bear/base/bull {si['ROE']['bear']:.1%} / {si['ROE']['base']:.1%} / {si['ROE']['bull']:.1%}, "
                       f"payout {si['payout']:.0%}, BVPS today ₹{si['BVPS0']:,.1f}. {res['path_note']}.")
        mm = pd.DataFrame({"value": [m["own_5y_median"], m["own_5y_p10"], m["peer_median"], m["sector_median"],
                                     m["neutral"], m["modifiers_sum"] * 100, m["ceiling"], m["spot"]]},
                          index=[f"own 5Y median ({m['own_months']} months)", "own 5Y P10 (trough)",
                                 f"sub-industry peers median ({m['peers']})", "sector median", "NEUTRAL (50/50)",
                                 "modifiers (%)", "cap (1.3x sector, ROCE/growth ceiling)", "spot"])
        st.markdown("**Multiple model**")
        st.dataframe(mm.round(2), use_container_width=True)
        flags = []
        if m["capped"]:
            flags.append("BASE capped")
        if m["precedence"]:
            flags.append("BASE set by the precedence rule")
        if m["indeterminate"]:
            flags.append("STRUCTURALLY INDETERMINATE")
        flags.append(f"durable re-rating test {'PASS' if m['rerating_test'] else 'fail'}")
        st.caption(" · ".join(flags))
    with c2:
        st.markdown("**1C probability scorecard**")
        scd = pd.DataFrame([(k_, v["score"], v["evidence"]) for k_, v in res["scorecard"].items()],
                           columns=["row", "score", "evidence"])
        st.dataframe(scd, hide_index=True, use_container_width=True, height=280)
        st.caption(f"Total {res['score_total']:+.1f} → probabilities bear/base/bull "
                   f"{' / '.join(f'{p:.0%}' for p in res['probs'])}")
        st.markdown("**Gates**")
        gt = pd.DataFrame([(g, "✅" if v["ok"] else "❌", v["detail"]) for g, v in res["gates"].items()] +
                          [("1F", "✅" if res["sanity_1f"]["ok"] else "❌",
                            f"base Y1 ÷ TTM = {res['sanity_1f']['ratio']:.2f}" +
                            (" (1.5-2.5x: explain)" if res["sanity_1f"]["flag"] else ""))],
                          columns=["gate", "", "detail"])
        st.dataframe(gt, hide_index=True, use_container_width=True)
        rg = pd.DataFrame({s: [f"{v['share']:.0%}", f"₹{v['mean']:,.0f}"] for s, v in res["regimes"].items()},
                          index=["trial share", "conditional mean"])
        st.dataframe(rg, use_container_width=True)
        if res.get("sensitivity"):
            st.caption("P4 knob sensitivity (CMP percentile): " +
                       " · ".join(f"{k_} P{v:.0f}" for k_, v in res["sensitivity"].items()) +
                       (" — **KNOB-DEPENDENT**" if res.get("knob_dependent") else ""))
    lad = res["ladder"]
    st.caption(f"Entry/exit ladder (only meaningful when gates pass): deep add ≤ ₹{lad['deep_add']:,.0f} · accumulate "
               f"₹{lad['deep_add']:,.0f}-{lad['accumulate']:,.0f} · fair ₹{lad['fair_lo']:,.0f}-{lad['fair_hi']:,.0f} · "
               f"trim ₹{lad['fair_hi']:,.0f}-{lad['trim']:,.0f} · exit > ₹{lad['trim']:,.0f}. Pivot: {res['pivot']}.")
    if res["notes"]:
        st.markdown("\n".join(f"- {n}" for n in res["notes"]))
    ref = _reference()
    t = res["ticker"]
    if len(ref) and t in set(ref.ticker):
        r = ref[ref.ticker == t].sort_values("run_date").iloc[-1]
        st.info(f"**Your v4.1 run ({r.run_date}):** {r.verdict} at P{r.cmp_percentile:.0f}"
                + (f", fair value P50 ₹{r.p50:,.0f}" if pd.notna(r.get("p50")) else "")
                + f" — mechanical: {res['rating']} at P{d['cmp_pct']:.0f}"
                + (f", P50 ₹{d['P50']:,.0f}" if pd.notna(r.get("p50")) else ""))
    with st.expander("Known limits · engine"):
        st.markdown(KNOWN_LIMITS)
        st.caption(res["engine_line"] + f" · weights {res['weights']} · data: {snap['years']} annual years, "
                   f"{snap['quarters']} quarters, own multiple history {m['own_months']} months")


# ----------------------------------------------------------------------------
# your v4.1 runs vs the mechanical rating
# ----------------------------------------------------------------------------
def _reference() -> pd.DataFrame:
    if not os.path.exists(REFERENCE):
        return pd.DataFrame()
    R = pd.read_csv(REFERENCE)
    R["ticker"] = R["ticker"].astype(str).str.replace(".NS", "", regex=False).str.strip().str.upper()
    return R


def render_reference(df: pd.DataFrame):
    R = _reference()
    with st.expander(f"🧪 Mechanical vs your v4.1 runs ({len(R)} logged)", expanded=False):
        st.caption(f"Log each full v4.1 run you do (with an LLM or by hand) in `{REFERENCE}` with columns "
                   "`ticker, run_date, verdict, cmp_percentile, p10, p50, p90, cmp, notes`. This panel lines them up "
                   "with the mechanical rating at the same price, so you can see where the two agree and where the "
                   "judgment rows (event risk, drivers, governance, modifiers) moved the answer.")
        if R.empty or not os.path.exists(SNAPSHOT):
            st.info("No reference runs logged yet." if R.empty else "Mechanical snapshot not built yet.")
            return
        c = ctx()
        rows = []
        for r in R.to_dict("records"):
            res = value_stock(r["ticker"], c, price=float(r["cmp"]), knobs=False) if pd.notna(r.get("cmp")) else {}
            if res.get("status") != "OK":
                rows.append(dict(ticker=r["ticker"], run_date=r["run_date"], v41=r["verdict"],
                                 v41_pct=r.get("cmp_percentile"), mech=res.get("status", "n/a")))
                continue
            rows.append(dict(ticker=r["ticker"], run_date=r["run_date"], cmp=r["cmp"], v41=r["verdict"],
                             v41_pct=r.get("cmp_percentile"), mech=res["rating"], mech_prov=res.get("provisional", ""),
                             mech_pct=res["dist"]["cmp_pct"], v41_p50=r.get("p50"), mech_p50=res["dist"]["P50"],
                             same_bucket=_base_label(str(r["verdict"]).upper()) ==
                             _base_label(res["rating"] if not res["rating"].startswith(("UNRATED", "INDET")) else
                                         res.get("provisional", "").strip("(").split(")")[0])))
        T = pd.DataFrame(rows)
        if "mech_pct" in T:
            T["pct_gap"] = T.mech_pct - pd.to_numeric(T.v41_pct, errors="coerce")
            ok = T.dropna(subset=["pct_gap"])
            st.markdown(f"**Agreement:** same verdict bucket {ok.same_bucket.mean():.0%} of {len(ok)} runs · median "
                        f"percentile gap {ok.pct_gap.median():+.0f} · mean absolute gap {ok.pct_gap.abs().mean():.0f} points")
        st.dataframe(T, hide_index=True, use_container_width=True)
        st.caption("Note: the mechanical run uses the price you logged, but today's fundamentals files -- a run logged "
                   "before a results season is not compared like-for-like.")
