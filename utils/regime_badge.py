"""
One market-regime read for the whole dashboard (replaces the separate sidebar / Live Desk classifiers).

  Market regime  -- Nifty, 5 states, the exact definition tested in regime_study.py
                    (analysis/REGIME_VERDICT.md): BEAR / MILD BEAR / CHOPPY / MILD BULL / BULL
  Breadth        -- % of the loaded universe above its own 200-day average, same 5 bands. The study
                    measured it on a broader point-in-time top-1000; on 16 Oct 2026 that read 54% vs 42%
                    here, so this label can run a notch more bearish than the study's.
  Engine mode    -- utils.regime_manager.classify_regime (BULL / CAUTION / BEAR / CRISIS), which is what
                    actually sets the live engine's entry floor, trail stop and rebalance. Unchanged.

"What works" for each Nifty regime comes from the regime study (6-month median vs the typical stock,
pre-registered rule + placebo test); these are discretionary leans, not engine rules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIMES = ["BEAR", "MILD BEAR", "CHOPPY", "MILD BULL", "BULL"]
COLORS = {"BEAR": "#FF3B30", "MILD BEAR": "#FF9500", "CHOPPY": "#FFD60A", "MILD BULL": "#32D74B", "BULL": "#34C759"}
ICONS = {"BEAR": "🔴", "MILD BEAR": "🟠", "CHOPPY": "🟡", "MILD BULL": "🟢", "BULL": "🟢"}
ENGINE_ICONS = {"BULL": "🟢", "CAUTION": "🟡", "BEAR": "🟠", "CRISIS": "🔴"}

# regime_study.py, Nifty view: (favour, avoid, live engine annualised in this regime vs Nifty)
WHAT_WORKS = {
    "BEAR": (["6-month losers / beaten-down names (+9.6 pp)", "cheapest 20% vs peers (+6.2)", "micro caps (+2.8)"],
             ["momentum (-7.1), esp. expensive momentum (-18.8)", "P/E >= 60 (-11.6)", "near-high names (-5.0)"],
             "engine -7%/yr vs Nifty -32% (only ~3 episodes, mainly Mar 2020 -- thin evidence)"),
    "MILD BEAR": (["⭐ absorption pullbacks (+6.5 pp)", "momentum + Gate 1 (+2.6)", "near the 52w high (+2.8)",
                   "momentum even when expensive (+5.8)", "mid caps (+2.6)"],
                  ["chasing fresh breakouts (+1.4, fails)", "> 40% below the high (-3.9)", "cheap for its own sake (-0.6)",
                   "loss-makers (-1.2)", "micro caps (-2.0)"],
                  "engine -5%/yr vs Nifty -22% -- mostly preserves capital"),
    "CHOPPY": (["momentum + fresh breakout (+3.7 pp)", "v6 BUY NOW (+3.0)", "near the 52w high (+3.2)",
                "momentum even when expensive (+7.0)", "mid / large caps (+3.3 / +2.1)"],
               ["> 40% below the high (-4.7)", "RSI < 30 dips (-1.3)", "loss-makers (-2.3)", "micro caps (-2.2)"],
               "engine +19%/yr vs Nifty +6%"),
    "MILD BULL": (["v6 BUY NOW (+2.8 pp)", "near the 52w high (+2.7)", "Gate 1 trend (+2.3)", "mid caps (+2.1)"],
                  ["> 40% below the high (-5.1)", "loss-makers (-3.1)"],
                  "engine +26%/yr vs Nifty +3%"),
    "BULL": (["momentum + squeeze release (+7.5 pp)", "momentum + breakout (+6.0)", "momentum (+6.1)",
              "absorption (+5.4)", "breakout + anchor + leader (+4.0)", "leader industries (+2.2)"],
             ["6-month losers (-6.6)", "> 40% below the high (-6.9)", "RSI < 30 dips (-3.8)"],
             "engine +50%/yr vs Nifty +40%"),
}


def nifty_regime(nc: pd.Series) -> dict:
    """5-state Nifty regime (regime_study.py definition) on the last bar, with its inputs."""
    nc = pd.to_numeric(nc, errors="coerce").dropna()
    if len(nc) < 200:
        return {"regime": "UNKNOWN"}
    ma200 = nc.rolling(200, min_periods=150).mean()
    hi = nc.rolling(252, min_periods=150).max()
    c, m, h = float(nc.iloc[-1]), float(ma200.iloc[-1]), float(hi.iloc[-1])
    dd = c / h - 1
    slope = m / float(ma200.iloc[-21]) - 1 if len(ma200.dropna()) > 21 else np.nan
    if dd <= -0.15 and c < m:
        r = "BEAR"
    elif c < m:
        r = "MILD BEAR"
    elif c <= 1.03 * m or dd <= -0.08:
        r = "CHOPPY"
    elif c > 1.03 * m and dd > -0.04 and slope > 0:
        r = "BULL"
    else:
        r = "MILD BULL"
    return {"regime": r, "close": c, "vs_ma200": (c / m - 1) * 100, "dd": dd * 100,
            "ma200_rising": bool(slope > 0) if np.isfinite(slope) else None, "asof": nc.index[-1]}


def breadth_regime(df: pd.DataFrame) -> dict:
    px = pd.to_numeric(df.get("price", df.get("currentPrice")), errors="coerce")
    m2 = pd.to_numeric(df.get("twoHundredDayAverage"), errors="coerce")
    ok = px.notna() & m2.notna()
    if ok.sum() < 50:
        return {"regime": "UNKNOWN"}
    pct = float((px[ok] > m2[ok]).mean() * 100)
    r = "BEAR" if pct < 30 else "MILD BEAR" if pct < 40 else "CHOPPY" if pct < 50 else "MILD BULL" if pct < 60 else "BULL"
    return {"regime": r, "pct": pct, "n": int(ok.sum())}


def engine_mode(nifty_df: pd.DataFrame) -> dict:
    from utils.regime_manager import classify_regime, get_regime_params
    try:
        c = nifty_df["Close"]
        mode = classify_regime(float(c.iloc[-1]), float(c.rolling(200).mean().iloc[-1]),
                               float(nifty_df["High"].rolling(252).max().iloc[-1]))
    except Exception:
        mode = "CAUTION"
    return {"mode": mode, **get_regime_params(mode)}


def compute(nifty_df: pd.DataFrame | None, df: pd.DataFrame) -> dict:
    nd = None
    if nifty_df is not None and not getattr(nifty_df, "empty", True):
        nd = nifty_df.copy()
        if getattr(nd.index, "tz", None) is not None:
            nd.index = nd.index.tz_localize(None)
    return {"nifty": nifty_regime(nd["Close"]) if nd is not None else {"regime": "UNKNOWN"},
            "breadth": breadth_regime(df),
            "engine": engine_mode(nd) if nd is not None else {"mode": "UNKNOWN"}}


def sidebar_line(info: dict) -> str:
    n, b, e = info["nifty"], info["breadth"], info["engine"]
    s = f"**Market:** {ICONS.get(n['regime'], '⚪')} **{n['regime']}**"
    if b.get("regime", "UNKNOWN") != "UNKNOWN":
        s += f" · Breadth {b['pct']:.0f}% ({b['regime'].title()})"
    s += f"  \n**Engine mode:** {ENGINE_ICONS.get(e['mode'], '⚪')} {e['mode']}"
    return s


def render(info: dict, st, compact: bool = False) -> None:
    """Full regime card: both reads, engine mode, and what has worked in this Nifty regime."""
    n, b, e = info["nifty"], info["breadth"], info["engine"]
    rg = n.get("regime", "UNKNOWN")
    col = COLORS.get(rg, "#8E8E93")
    detail = ""
    if rg != "UNKNOWN":
        detail = (f"Nifty {n['vs_ma200']:+.1f}% vs its 200-day average, {n['dd']:.1f}% from its 52-week high, "
                  f"200-day average {'rising' if n.get('ma200_rising') else 'falling'}")
    br = (f"{b['pct']:.0f}% of {b['n']} stocks above their 200-day average → {ICONS.get(b['regime'], '')} {b['regime']}"
          if b.get("regime", "UNKNOWN") != "UNKNOWN" else "n/a")
    split = ""
    if rg != "UNKNOWN" and b.get("regime", "UNKNOWN") != "UNKNOWN" and b["regime"] != rg:
        split = (" — <b>the two reads disagree</b> (they agree on only ~half of days); "
                 + ("index weak, broad market firmer" if REGIMES.index(b["regime"]) > REGIMES.index(rg)
                    else "index firmer than the broad market"))
    eng = ""
    if e.get("mode", "UNKNOWN") != "UNKNOWN":
        eng = (f"{ENGINE_ICONS.get(e['mode'], '')} <b>{e['mode']}</b> — entry floor CompRS ≥ {e['min_comp_rs'] * 100:.0f}, "
               f"trail stop {e['trail_stop'] * 100:.0f}%, rebalance every {e['rebalance_freq']} sessions"
               + ("" if e.get("new_entries", True) else ", <b>new entries suspended</b>"))
    st.markdown(f"""
    <div style="background:{col}22;border-left:5px solid {col};padding:12px 16px;border-radius:6px;">
      <div style="font-size:1.15em;font-weight:700;color:{col};">{ICONS.get(rg, '⚪')} Market regime: {rg}</div>
      <div style="opacity:.85;margin-top:4px;">{detail}</div>
      <div style="margin-top:6px;">Breadth: {br}{split}</div>
      <div style="margin-top:6px;">Engine mode (drives the live book): {eng or 'n/a'}</div>
    </div>""", unsafe_allow_html=True)
    if compact or rg not in WHAT_WORKS:
        return
    fav, avoid, engine_line = WHAT_WORKS[rg]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**What has worked in this regime**  \n" + "  \n".join(f"✅ {x}" for x in fav))
    with c2:
        st.markdown("**What has not**  \n" + "  \n".join(f"⛔ {x}" for x in avoid))
    st.caption(f"Live engine in this regime: {engine_line}. 6-month median vs the typical stock, 2016-26, "
               "analysis/REGIME_VERDICT.md -- discretionary leans for your own picks, not engine rules.")
