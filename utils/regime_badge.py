"""
One market-regime read for the whole dashboard (replaces the separate sidebar / Live Desk classifiers).

  Market regime  -- Nifty, 5 states, the exact definition tested in regime_study.py
                    (analysis/REGIME_VERDICT.md): BEAR / MILD BEAR / CHOPPY / MILD BULL / BULL
  Breadth        -- % of the loaded universe above its own 200-day average, same 5 bands. The study
                    measured it on a broader point-in-time top-1000; on 16 Oct 2026 that read 54% vs 42%
                    here, so this label can run a notch more bearish than the study's.
  Model rules    -- action-based status for utils.regime_manager.classify_regime
                    (Full risk-on / Tightened / Defensive / Entries paused), which is what
                    actually sets the live engine's entry floor, trail stop and rebalance. Unchanged.

"What works" for each Nifty regime comes from the regime study (6-month median vs the typical stock,
pre-registered rule + placebo test); these are discretionary leans, not engine rules.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

REGIMES = ["BEAR", "MILD BEAR", "CHOPPY", "MILD BULL", "BULL"]
COLORS = {"BEAR": "#FF3B30", "MILD BEAR": "#FF9500", "CHOPPY": "#FFD60A", "MILD BULL": "#32D74B", "BULL": "#34C759"}
ICONS = {"BEAR": "🔴", "MILD BEAR": "🟠", "CHOPPY": "🟡", "MILD BULL": "🟢", "BULL": "🟢"}
ENGINE_ICONS = {"BULL": "🟢", "RECOVERY": "🚀", "CAUTION": "🟡", "BEAR": "🟠", "CRISIS": "🔴"}
ENGINE_ACTION = {"BULL": "Full risk-on", "RECOVERY": "Early recovery", "CAUTION": "Tightened", "BEAR": "Defensive", "CRISIS": "Entries paused"}

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
    if isinstance(nc.index, pd.DatetimeIndex):
        h = float(nc[nc.index > nc.index[-1] - pd.Timedelta(days=365)].max())
    else:
        h = float(nc.tail(252).max())
    c, m = float(nc.iloc[-1]), float(ma200.iloc[-1])
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
    if df is None or df.empty:
        return {"regime": "UNKNOWN"}
    d = df.drop_duplicates("ticker") if "ticker" in df.columns else df
    if "extended" in d.columns:
        d = d[d["extended"].ne(True)]
    px_cur = pd.to_numeric(d["currentPrice"], errors="coerce") if "currentPrice" in d.columns else pd.Series(np.nan, index=d.index)
    px_alt = pd.to_numeric(d["price"], errors="coerce") if "price" in d.columns else pd.Series(np.nan, index=d.index)
    px = px_cur.fillna(px_alt)
    m2 = pd.to_numeric(d["twoHundredDayAverage"], errors="coerce") if "twoHundredDayAverage" in d.columns else pd.Series(np.nan, index=d.index)
    ok = px.notna() & (px > 0) & m2.notna() & (m2 > 0)
    if ok.sum() < 50:
        return {"regime": "UNKNOWN"}

    broad_pct = float((px[ok] > m2[ok]).mean() * 100)
    broad_n = int(ok.sum())

    # Guardrail: anchor canonical regime breadth to Nifty 500 when present in a live snapshot,
    # so expanding the universe with small-caps/IPOs never shifts historical BULL/BEAR cutoffs.
    anchor_label = "Universe"
    pct, n_anchor = broad_pct, broad_n
    if "ticker" in d.columns:
        try:
            from utils.liquid_universe import nifty500_tickers
            n500 = nifty500_tickers()
            if n500:
                is_n500 = d["ticker"].astype(str).isin(n500) & ok
                if int(is_n500.sum()) >= 200:
                    pct = float((px[is_n500] > m2[is_n500]).mean() * 100)
                    n_anchor = int(is_n500.sum())
                    anchor_label = "Nifty 500"
        except Exception:
            pass

    r = "BEAR" if pct < 30 else "MILD BEAR" if pct < 40 else "CHOPPY" if pct < 50 else "MILD BULL" if pct < 60 else "BULL"
    out = {"regime": r, "pct": pct, "n": n_anchor, "anchor": anchor_label}
    if anchor_label == "Nifty 500" and broad_n > n_anchor:
        out["broad_pct"] = broad_pct
        out["broad_n"] = broad_n
    return out


def engine_mode(nifty_df: pd.DataFrame, recovery: dict | bool | None = None, root: str = ".") -> dict:
    from utils.regime_manager import classify_regime, get_regime_params, load_recovery_signals
    if nifty_df is None or getattr(nifty_df, "empty", True) or "Close" not in nifty_df.columns or "High" not in nifty_df.columns:
        return {"mode": "CAUTION", **get_regime_params("CAUTION")}
    try:
        c = nifty_df["Close"]
        c_last = float(c.iloc[-1])
        m200 = float(c.rolling(200).mean().iloc[-1])
        hi52 = float(nifty_df["High"][nifty_df.index > nifty_df.index[-1] - pd.Timedelta(days=365)].max()
                     if isinstance(nifty_df.index, pd.DatetimeIndex) else nifty_df["High"].tail(252).max())
        base_mode = classify_regime(c_last, m200, hi52)
        rec = recovery
        if rec is None and base_mode in ("BEAR", "CAUTION"):
            asof = nifty_df.index[-1] if isinstance(nifty_df.index, pd.DatetimeIndex) else None
            rec = load_recovery_signals(root=root, as_of=asof)
        mode = classify_regime(c_last, m200, hi52, recovery=rec)
    except Exception:
        mode, rec = "CAUTION", None
    out = {"mode": mode, **get_regime_params(mode)}
    if isinstance(rec, dict) and rec.get("active"):
        out["recovery"] = rec
    return out


def _vix_stats(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {}
    c = float(s.iloc[-1])
    chg_5d = float(c - float(s.iloc[-6])) if len(s) >= 6 else np.nan
    win = s.tail(252)
    pctile = float((win <= c).mean() * 100.0) if len(win) >= 20 else np.nan
    asof = s.index[-1] if isinstance(s.index, pd.DatetimeIndex) else None
    return {"close": round(c, 2), "chg_5d": round(chg_5d, 2) if np.isfinite(chg_5d) else np.nan,
            "pctile_252": round(pctile, 0) if np.isfinite(pctile) else np.nan, "asof": asof}


def load_vix(root: str = ".") -> dict:
    """Latest India VIX close, 5-session change, and 252-day percentile from NSE index parquets."""
    try:
        files = sorted(glob.glob(os.path.join(root, "data", "nse", "indices", "*.parquet")))
        if not files:
            return {}
        frames = []
        for p in files[-13:]:
            part = pd.read_parquet(p, columns=["date", "index", "close"])
            part = part[part["index"] == "India VIX"]
            if not part.empty:
                frames.append(part)
        if not frames:
            return {}
        v = pd.concat(frames, ignore_index=True)
        v["date"] = pd.to_datetime(v["date"], errors="coerce")
        v = v.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last").sort_values("date")
        if v.empty:
            return {}
        return _vix_stats(v.set_index("date")["close"])
    except Exception:
        return {}


def vix_read(vix_df=None, root: str = ".") -> dict:
    if vix_df is not None:
        if isinstance(vix_df, pd.Series):
            return _vix_stats(vix_df)
        if isinstance(vix_df, pd.DataFrame) and not vix_df.empty:
            col = "close" if "close" in vix_df.columns else ("Close" if "Close" in vix_df.columns else None)
            if col:
                s = vix_df.set_index(pd.to_datetime(vix_df["date"]))[col] if "date" in vix_df.columns else vix_df[col]
                return _vix_stats(s)
        return {}
    return load_vix(root)


def compute(nifty_df: pd.DataFrame | None, df: pd.DataFrame, vix_df=None, root: str = ".", recovery=None) -> dict:
    nd = None
    if nifty_df is not None and not getattr(nifty_df, "empty", True):
        nd = nifty_df.copy()
        if getattr(nd.index, "tz", None) is not None:
            nd.index = nd.index.tz_localize(None)
    return {"nifty": nifty_regime(nd["Close"]) if nd is not None else {"regime": "UNKNOWN"},
            "breadth": breadth_regime(df),
            "engine": engine_mode(nd, recovery=recovery, root=root) if nd is not None else {"mode": "UNKNOWN"},
            "vix": vix_read(vix_df, root=root)}


def engine_short(e: dict | None) -> str:
    """Action-based label for the engine's current posture (without repeating a second BULL/BEAR regime)."""
    e = e or {}
    m = str(e.get("mode") or "UNKNOWN")
    if m in ENGINE_ACTION:
        return f"{ENGINE_ICONS.get(m, '⚪')} {ENGINE_ACTION[m]}"
    if m == "UNKNOWN":
        return "⚪ —"
    return f"{ENGINE_ICONS.get(m, '⚪')} {m.title()}"


def engine_status(e: dict | None) -> str:
    """Action-based engine status with active trailing-stop percentage."""
    e = e or {}
    m = str(e.get("mode") or "UNKNOWN")
    if m in ENGINE_ACTION:
        trail = e.get("trail_stop")
        if trail is None:
            from utils.regime_manager import get_regime_params
            trail = get_regime_params(m).get("trail_stop")
        trail_str = f" ({float(trail) * 100:.0f}% trail)" if trail is not None else ""
        return f"{ENGINE_ICONS.get(m, '⚪')} {ENGINE_ACTION[m]}{trail_str}"
    if m == "UNKNOWN":
        return "⚪ —"
    return f"{ENGINE_ICONS.get(m, '⚪')} {m.title()}"


def _fmt_vix(v: dict | None) -> str:
    if not v:
        return ""
    c = v.get("close")
    if c is None or not np.isfinite(float(c)):
        return ""
    c = float(c)
    d5 = v.get("chg_5d")
    d_txt = f" ({float(d5):+.1f})" if d5 is not None and np.isfinite(float(d5)) else ""
    flag = " 🔥" if c >= 22.0 else ""
    return f"VIX {c:.1f}{d_txt}{flag}"


def sidebar_line(info: dict) -> str:
    n, b, e = info["nifty"], info["breadth"], info["engine"]
    s = f"**Market:** {ICONS.get(n['regime'], '⚪')} **{n['regime']}**"
    if b.get("regime", "UNKNOWN") != "UNKNOWN":
        s += f" · Breadth {b['pct']:.0f}% ({b['regime'].title()})"
    v_txt = _fmt_vix(info.get("vix"))
    if v_txt:
        s += f" · {v_txt}"
    s += f"  \n**Model rules:** {engine_status(e)}"
    return s


def render(info: dict, st, compact: bool = False) -> None:
    """Full regime card: both reads, model rules status, and what has worked in this Nifty regime."""
    n, b, e = info["nifty"], info["breadth"], info["engine"]
    rg = n.get("regime", "UNKNOWN")
    col = COLORS.get(rg, "#8E8E93")
    detail = ""
    if rg != "UNKNOWN" and n.get("vs_ma200") is not None and n.get("dd") is not None:
        detail = (f"Nifty {float(n['vs_ma200']):+.1f}% vs its 200-day average, {float(n['dd']):.1f}% from its 52-week high, "
                  f"200-day average {'rising' if n.get('ma200_rising') else 'falling'}")
    b_n = b.get("n", b.get("total", 0))
    anchor_lbl = f"{b['anchor']} ({b_n})" if b.get("anchor") == "Nifty 500" else f"{b_n} stocks"
    broad_suffix = (f" · Broad {float(b['broad_pct']):.0f}% ({int(b['broad_n'])})"
                    if b.get("broad_pct") is not None and b.get("broad_n") else "")
    br = (f"{float(b.get('pct', 0)):.0f}% of {anchor_lbl} above their 200-day average → {ICONS.get(b['regime'], '')} {b['regime']}{broad_suffix}"
          if b.get("regime", "UNKNOWN") != "UNKNOWN" and b.get("pct") is not None else "n/a")
    split = ""
    if rg != "UNKNOWN" and b.get("regime", "UNKNOWN") != "UNKNOWN" and b["regime"] != rg:
        split = (" — <b>the two reads disagree</b> (they agree on only ~half of days); "
                 + ("index weak, broad market firmer" if REGIMES.index(b["regime"]) > REGIMES.index(rg)
                    else "index firmer than the broad market"))
    eng = ""
    if e.get("mode", "UNKNOWN") != "UNKNOWN":
        m = str(e["mode"])
        from utils.regime_manager import get_regime_params
        p = {**get_regime_params(m), **{k: v for k, v in e.items() if v is not None}}
        act = ENGINE_ACTION.get(m, m.title())
        reb = ("rebalance paused"
               if (not p.get("new_entries", True) or int(p.get("rebalance_freq", 0) or 0) >= 999)
               else f"rebalance every {int(p.get('rebalance_freq', 15))} sessions")
        eng = (f"{ENGINE_ICONS.get(m, '⚪')} <b>{act}</b> — entry floor CompRS ≥ {float(p.get('min_comp_rs', 0)) * 100:.0f}, "
               f"trail stop {float(p.get('trail_stop', 0.15)) * 100:.0f}%, {reb}"
               + ("" if p.get("new_entries", True) else ", <b>new entries suspended</b>"))
        rec_meta = e.get("recovery")
        if m == "RECOVERY" and isinstance(rec_meta, dict) and rec_meta.get("reasons"):
            eng += f" · <i>{' · '.join(rec_meta['reasons'][:2])}</i>"
    vix = info.get("vix") or {}
    vix_row = ""
    if vix.get("close") is not None and np.isfinite(float(vix["close"])):
        vc = float(vix["close"])
        d5 = vix.get("chg_5d")
        p252 = vix.get("pctile_252")
        bits = [f"5d {float(d5):+.1f}" if d5 is not None and np.isfinite(float(d5)) else "",
                f"P{float(p252):.0f} (1Y)" if p252 is not None and np.isfinite(float(p252)) else ""]
        meta = " · ".join(x for x in bits if x)
        vix_row = f'<div style="margin-top:6px;">India VIX: <b>{vc:.1f}</b>' + (f" ({meta})" if meta else "") + (" — elevated stress" if vc >= 22.0 else "") + "</div>"
    st.markdown(f"""
    <div style="background:{col}22;border-left:5px solid {col};padding:12px 16px;border-radius:6px;">
      <div style="font-size:1.15em;font-weight:700;color:{col};">{ICONS.get(rg, '⚪')} Market regime: {rg}</div>
      <div style="opacity:.85;margin-top:4px;">{detail}</div>
      <div style="margin-top:6px;">Breadth: {br}{split}</div>
      {vix_row}
      <div style="margin-top:6px;">Model rules (drives the live book): {eng or 'n/a'}</div>
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
