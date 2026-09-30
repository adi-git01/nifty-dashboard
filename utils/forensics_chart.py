"""
Plotly visuals for the Technicals single-stock report (utils/forensics.plan):

  price_chart       candles + MA20/50/100/200, trapped supply shelves, stop /
                    trigger / targets / VETO / 52w-high / trail lines, every
                    volume event of the last 60 sessions (anchor ringed),
                    shakeout, and a volume panel with the 50-session median
  levels_ladder     the key levels as a vertical ladder with % from the close
  retention_chart   each close since the anchor vs the close before it

Colours follow the reference palette (dataviz skill): identity hues for the
MAs, red reserved for the stop / VETO (always labelled), green for the
clean-air trigger.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.forensics import SUPPLY_LABEL, UP_STATES, events, levels

UP, DOWN = "#1baf7a", "#e34948"
MA_COL = {"MA20": "#eda100", "MA50": "#2a78d6", "MA100": "#4a3aa7", "MA200": "#52514e"}
LEVEL_STYLE = {           # kind: (colour, dash, width)
    "stop": ("#e34948", "solid", 2),
    "veto": ("#e34948", "dot", 1.5),
    "trigger": ("#008300", "dash", 2),
    "target": ("#8a8984", "dot", 1.5),
    "high": ("#2a78d6", "dashdot", 1.5),
    "trail": ("#eb6834", "dot", 1.5),
    "held": ("#0b0b0b", "solid", 1.5),
}
MUTED = "#8a8984"


def _fmt(x):
    return f"₹{x:,.2f}"


def price_chart(P: dict, ticker: str, months: int = 9) -> go.Figure:
    r, d = P["r"], P["df"].copy()
    d = d.astype({k: float for k in ("High", "Low", "Close", "Volume")})
    for n in (20, 50, 100, 200):
        d[f"MA{n}"] = d.Close.rolling(n).mean()
    d["vmed"] = d.Volume.rolling(50, min_periods=30).median().shift(1)
    start = d.index[-1] - pd.DateOffset(months=months)
    view = d[d.index >= start]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24], vertical_spacing=0.03)

    # price
    if "Open" in d and d["Open"].notna().any():
        fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close, name="Price",
                                     increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
                                     decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN),
                                     showlegend=False), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=d.index, y=d.Close, name="Close", line=dict(color="#0b0b0b", width=2)), row=1, col=1)
    for k, col in MA_COL.items():
        fig.add_trace(go.Scatter(x=d.index, y=d[k], name=k, mode="lines", hoverinfo="skip",
                                 line=dict(color=col, width=2 if k == "MA50" else 1.2,
                                           dash="dash" if k == "MA200" else "solid")), row=1, col=1)

    # supply shelves (trapped volume above the close)
    for k, run in enumerate(r.get("shelves_above", [])[:3]):
        near = k == 0 and r["supply"] == "NEAR OVERHANG"
        fig.add_hrect(y0=run["low"], y1=run["top"], row=1, col=1, line_width=0,
                      fillcolor="rgba(227,73,72,0.16)" if near else "rgba(138,137,132,0.14)",
                      annotation_text=f"Trapped shelf · {run['share']:.0f}% of 1y turnover · untouched {run['age']}d",
                      annotation_position="top left", annotation_font=dict(size=10, color="#52514e"),
                      annotation_bgcolor="rgba(255,255,255,0.8)")

    # since-anchor window
    if "anchor_date" in r:
        fig.add_vrect(x0=r["anchor_date"], x1=d.index[-1], row=1, col=1, line_width=0,
                      fillcolor="rgba(42,120,214,0.06)")

    # key horizontal levels, labelled at the right edge
    lv = {x["name"]: x for x in levels(P)}
    lines = []
    for name, x in lv.items():
        kind = x["kind"]
        if kind in ("close", "ma", "shelf"):
            continue
        shelf_edges = {round(run["low"], 2) for run in r.get("shelves_above", [])}
        if name.startswith("T2") and round(x["price"], 2) in shelf_edges:
            continue                                              # T2 is a shelf's bottom edge, already shaded
        if kind == "trail" and not P["held"]:
            continue
        lines.append((name, x["price"], kind))
    if P["held"]:
        lines.append((f"Bought @ {P['held_price']:,.2f}", P["held_price"], "held"))
    for name, y, kind in lines:
        col, dash, w = LEVEL_STYLE.get(kind, (MUTED, "dot", 1))
        fig.add_hline(y=y, row=1, col=1, line=dict(color=col, dash=dash, width=w),
                      annotation_text=f"{name.split(' (')[0].replace(' = clean-air trigger', ' → clean-air trigger')} {_fmt(y)}",
                      annotation_position="right", annotation_font=dict(size=10, color=col))

    # volume events of the last 60 sessions
    ev = events(d)
    if len(ev):
        sym = {"+1 bull": "triangle-up", "-1 bear": "triangle-down", "0 mixed": "circle"}
        colr = {"+1 bull": UP, "-1 bear": DOWN, "0 mixed": MUTED}
        for dirn, g in ev.groupby("direction"):
            y = g.high * 1.02
            fig.add_trace(go.Scatter(
                x=g.date, y=y, mode="markers", name=f"Volume event {dirn.split()[0]}",
                marker=dict(symbol=sym[dirn], size=np.clip(g.vol_ratio * 2.2 + 6, 9, 22), color=colr[dirn],
                            line=dict(color=np.where(g.anchor, "#0b0b0b", "#fcfcfb"), width=np.where(g.anchor, 2.5, 1.5))),
                customdata=np.c_[g.vol_ratio, g.clv, g.turnover_cr, g.retention * 100, g.state, g.days,
                                 np.where(g.anchor, "ANCHOR (highest volume, 60 sessions)", "")],
                hovertemplate="<b>%{x|%d %b %Y}</b> %{customdata[6]}<br>vol %{customdata[0]:.1f}× median · "
                              "CLV %{customdata[1]:.2f}<br>turnover ₹%{customdata[2]:,.0f} cr<br>"
                              "retention since %{customdata[3]:.0f}% · %{customdata[4]} · day %{customdata[5]}"
                              "<extra></extra>"), row=1, col=1)
        a = ev[ev.anchor]
        if len(a):
            a = a.iloc[0]
            fig.add_annotation(x=a.date, y=a.high * 1.02, row=1, col=1, text=f"<b>ANCHOR</b> {a.vol_ratio:.1f}×",
                               showarrow=True, arrowhead=0, ay=-34, font=dict(size=11), bgcolor="rgba(255,255,255,0.8)")
    if r.get("shakeout") and "shakeout_date" in r:
        sd = pd.Timestamp(r["shakeout_date"])
        fig.add_trace(go.Scatter(x=[sd], y=[float(d.loc[sd, "Low"]) * 0.985], mode="markers", name="Shakeout",
                                 marker=dict(symbol="star", size=16, color="#eda100", line=dict(color="#0b0b0b", width=1)),
                                 hovertemplate=f"<b>Shakeout</b> %{{x|%d %b %Y}}<br>broke {_fmt(r['shakeout_level'])} "
                                               f"intraday, closed back above<br>vol {r['shakeout_volx']:.2f}× · "
                                               f"CLV {r['shakeout_clv']:.2f}<extra></extra>"), row=1, col=1)

    # volume panel
    up = d.Close >= d.get("Open", d.Close.shift(1)).fillna(d.Close.shift(1))
    anchor_day = pd.Timestamp(r["anchor_date"]) if "anchor_date" in r else None
    vcol = np.where(up, "rgba(27,175,122,0.55)", "rgba(227,73,72,0.55)")
    if anchor_day is not None and anchor_day in d.index:
        vcol[d.index.get_loc(anchor_day)] = "#0b0b0b"
    fig.add_trace(go.Bar(x=d.index, y=d.Volume, name="Volume", marker=dict(color=vcol, line_width=0),
                         showlegend=False, hovertemplate="%{x|%d %b}: %{y:,.0f}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d.vmed, name="50-session median", mode="lines",
                             line=dict(color="#eb6834", width=1.5), hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=3 * d.vmed, name="3× median (event line)", mode="lines",
                             line=dict(color="#eb6834", width=1, dash="dot"), hoverinfo="skip"), row=2, col=1)

    # y range from what is visible, plus levels within reach
    lo, hi = float(view.Low.min()), float(view.High.max())
    near = [y for _, y, _ in lines if 0.7 * lo <= y <= 1.3 * hi] + \
           [x for run in r.get("shelves_above", [])[:2] for x in (run["low"], run["top"]) if x <= 1.3 * hi]
    lo, hi = min([lo] + near), max([hi] + near)
    pad = (hi - lo) * 0.06
    vmax = float(view.Volume.max())
    title = (f"<b>{ticker}</b> · {P['action']} · {r['state']} · {SUPPLY_LABEL.get(r['supply'], r['supply'])}"
             f"{' · Gate 1 PASS' if r['gate1'] else ' · Gate 1 FAIL'}")
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15)), height=680, hovermode="x unified",
        margin=dict(l=10, r=150, t=60, b=40), legend=dict(orientation="h", y=1.04, x=0, font=dict(size=11)),
        xaxis_rangeslider_visible=False, bargap=0.15,
    )
    fig.update_xaxes(range=[view.index[0], d.index[-1] + pd.Timedelta(days=3)],
                     rangebreaks=[dict(bounds=["sat", "mon"])], showgrid=False)
    fig.update_yaxes(range=[lo - pad, hi + pad], row=1, col=1, title_text="Price (₹)", gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(range=[0, vmax * 1.1], row=2, col=1, title_text="Volume", gridcolor="rgba(128,128,128,0.15)")
    return fig


def levels_ladder(P: dict) -> go.Figure:
    """Key levels, top to bottom, with distance from the close. Labels are
    nudged apart so close levels stay readable; a leader line joins each label
    to its true price."""
    r = P["r"]
    lv = [x for x in levels(P) if x["kind"] != "trail" or P["held"]]
    for t2 in [x for x in lv if x["name"].startswith("T2")]:
        twin = [x for x in lv if x is not t2 and x["kind"] == "shelf" and round(x["price"], 2) == round(t2["price"], 2)]
        if twin:
            twin[0]["name"] += " = T2"
            lv.remove(t2)
    close = r["close"]
    ys = [x["price"] for x in lv]
    lo, hi = min(ys), max(ys)
    gap = (hi - lo) * 0.045 or 1
    # nudge label positions (top-down) so they are >= gap apart
    order = sorted(range(len(lv)), key=lambda k: -lv[k]["price"])
    pos = {}
    last = None
    for k in order:
        y = lv[k]["price"]
        if last is not None and last - y < gap:
            y = last - gap
        pos[k] = last = y
    fig = go.Figure()
    for k, x in enumerate(lv):
        col, dash, w = LEVEL_STYLE.get(x["kind"], (MUTED, "dot", 1))
        if x["kind"] == "close":
            col, dash, w = "#0b0b0b", "solid", 3
        elif x["kind"] == "ma":
            col, dash, w = MA_COL["MA50"] if "MA50" in x["name"] else MA_COL["MA200"], "solid", 1.5
        elif x["kind"] == "shelf":
            col, dash, w = "#e34948" if r["supply"] == "NEAR OVERHANG" else MUTED, "solid", 1
        fig.add_trace(go.Scatter(x=[0, 0.55], y=[x["price"]] * 2, mode="lines", line=dict(color=col, dash=dash, width=w),
                                 hovertemplate=f"{x['name']}: {_fmt(x['price'])}<extra></extra>", showlegend=False))
        fig.add_trace(go.Scatter(x=[0.55, 0.62], y=[x["price"], pos[k]], mode="lines",
                                 line=dict(color=MUTED, width=0.8), hoverinfo="skip", showlegend=False))
        pct = (x["price"] / close - 1) * 100
        txt = (f"<b>{x['name']}</b>  {_fmt(x['price'])}" + ("" if x["kind"] == "close" else f"  ({pct:+.1f}%)")
               + (f"  · {x['note']}" if x.get("note") else ""))
        fig.add_annotation(x=0.63, y=pos[k], text=txt, showarrow=False, xanchor="left", font=dict(size=11))
    for run in r.get("shelves_above", [])[:3]:
        fig.add_hrect(y0=run["low"], y1=run["top"], x0=0, x1=0.55 / 1.6, line_width=0,
                      fillcolor="rgba(227,73,72,0.12)" if r["supply"] == "NEAR OVERHANG" else "rgba(138,137,132,0.12)")
    span = max(hi, max(pos.values())) - min(lo, min(pos.values()))
    fig.update_layout(height=max(360, 34 * len(lv) + 80), margin=dict(l=10, r=10, t=40, b=10),
                      title=dict(text="<b>Key levels</b> · distance from the close", x=0, font=dict(size=14)),
                      xaxis=dict(visible=False, range=[0, 1.6]),
                      yaxis=dict(title="Price (₹)", range=[min(lo, min(pos.values())) - span * 0.05,
                                                           max(hi, max(pos.values())) + span * 0.05],
                                 gridcolor="rgba(128,128,128,0.12)"))
    return fig


def retention_chart(P: dict) -> go.Figure | None:
    """Each close since the anchor vs the close before the anchor (the retention test)."""
    r, d = P["r"], P["df"]
    if "anchor_date" not in r:
        return None
    c = d["Close"].astype(float)
    a = c.index.get_loc(pd.Timestamp(r["anchor_date"]))
    base = c.iloc[a - 1]
    post = c.iloc[a + 1:]
    if not len(post):
        return None
    pct = (post / base - 1) * 100
    held = post >= base
    fig = go.Figure(go.Bar(x=np.arange(1, len(post) + 1), y=pct, marker=dict(color=np.where(held, UP, DOWN)),
                           customdata=np.c_[post.index.strftime("%d %b"), post.values],
                           hovertemplate="day %{x} · %{customdata[0]}<br>close ₹%{customdata[1]:,.2f}<br>"
                                         "%{y:+.1f}% vs pre-anchor close<extra></extra>", showlegend=False))
    fig.add_hline(y=0, line=dict(color="#0b0b0b", width=1),
                  annotation_text=f"pre-anchor close {_fmt(base)}", annotation_position="top left",
                  annotation_font=dict(size=10))
    if r["state"] in UP_STATES or r["state"] == "VETO":
        vy = (r["anchor_low"] / base - 1) * 100
        fig.add_hline(y=vy, line=dict(color="#e34948", dash="dot", width=1.5),
                      annotation_text=f"anchor low {_fmt(r['anchor_low'])} (VETO below)", annotation_position="bottom right",
                      annotation_font=dict(size=10, color="#e34948"))
    if len(post) >= 5:
        fig.add_vline(x=5.5, line=dict(color=MUTED, dash="dash", width=1),
                      annotation_text="day 5: GO needs ≥ 80%", annotation_font=dict(size=10, color=MUTED))
    n_held = int(held.sum())
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10), bargap=0.2,
                      title=dict(text=f"<b>Retention since anchor</b> · {n_held}/{len(post)} closes held "
                                      f"({n_held / len(post):.0%}) · {r['state']}", x=0, font=dict(size=14)),
                      xaxis=dict(title="Sessions since anchor", dtick=5 if len(post) > 20 else 1),
                      yaxis=dict(title="% vs pre-anchor close", gridcolor="rgba(128,128,128,0.15)", zeroline=False))
    return fig
