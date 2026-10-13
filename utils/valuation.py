"""
Monte Carlo intrinsic valuation -- master prompt v4.1, run MECHANICALLY.
=======================================================================

Every formula of v4.1 that is arithmetic runs here exactly as written
(Phase 2D engine, gates A-F, score->probability map, anchors, precedence
rule, rho / hurdle policy, verdict ladder). The parts v4.1 leaves to
analyst judgment are replaced by explicit, history-based defaults and
labelled MECHANICAL:

  scenario bands   from the company's own record (growth: recent CAGRs
                   +/- the spread of its annual growth; net margin: worst /
                   median / best of its history and TTM)
  1C scorecard     only the numeric rows (promoter, FII/DII, execution as
                   YoY profit growth, FCF quality, balance sheet); guidance,
                   CEO/CFO, governance, pledge are listed as NOT SCORED (0)
  modifiers        only promoter trend, FII/DII and earnings predictability
  peers            same sub-industry (a proxy for "same business model")
  event risk,      data/valuation_overrides.csv (ticker, key, value, note)
  policy cliff,    -- with any override present the label becomes
  SOTP, drivers    MECHANICAL + OVERRIDES

Override keys (data/valuation_overrides.csv: ticker, key, value, note), all optional:
  identity INDUSTRIAL|BFSI|COMMODITY|OUT   g_bear/g_base/g_bull (revenue growth, 0.12 = 12%)
  m_bear/m_base/m_bull (terminal net margin)   roe_bear/roe_base/roe_bull (lenders)
  mult_bear/mult_base/mult_bull   modifier_pct (eight-factor sum, %)   rerating_pass 1|0
  p_bear_pp / p_bull_pp (event adjustment, capped +/-20; must come with a changed bear path or
  Gate F1 fails)   event_note   weights FRONT|BALANCED|BACK   rho   hurdle   confidence LOW|MEDIUM|HIGH

Inputs (fundamentals_annual.py): data/fundamentals_annual.csv.gz,
data/fundamentals_quarterly.csv.gz (falls back to analysis/screener_raw.csv.gz),
data/monthly_prices.csv.gz, data/fundamentals_cache.csv (market cap, sector),
plus the live price. Screener figures are Rs crore.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def ndtr(x):
    """Standard normal CDF (Abramowitz-Stegun 7.1.26 erf, |error| < 1.5e-7) -- avoids a SciPy dependency."""
    x = np.asarray(x, float)
    z = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-z * z)
    return 0.5 * (1.0 + np.sign(x) * y)

ENGINE = "v4.1 banded-mixture MC, regime-persistent (mechanical)"
N_DEFAULT = 10_000
SEED = 41
H = np.array([1, 2, 3])
WEIGHTS = {"FRONT": (0.5, 0.3, 0.2), "BALANCED": (0.34, 0.33, 0.33), "BACK": (0.2, 0.3, 0.5)}
PROB_MAP = [(4, (15, 55, 30)), (2, (20, 50, 30)), (0, (25, 50, 25)), (-2, (30, 50, 20)), (-99, (40, 45, 15))]
LADDER = [(20, "BUY"), (40, "ACCUMULATE"), (70, "HOLD"), (88, "REDUCE"), (101, "SELL/AVOID")]
BFSI_GRAN = {"Banks", "Finance"}
COMMODITY_GRAN = {"Ferrous Metals", "Non - Ferrous Metals", "Diversified Metals", "Minerals & Mining",
                  "Consumable Fuels", "Oil", "Metals & Minerals Trading", "Petroleum Products"}
OUT_GRAN = {"Insurance": "insurance", "Realty": "NAV-driven real estate"}
MIXED_GRAN = {"Diversified"}
KNOWN_LIMITS = ("Scenario inputs are judgments; the engine only guarantees the distribution faithfully represents "
                "them. Outcomes can land outside P5-P95. Multiple-dependent verdicts are timing-uncertain even when "
                "direction is right. The commodity/deep-cyclical track runs on an interim normalized-earnings "
                "approach after its original inverse-pairing mechanism failed live validation; its verdicts print "
                "INDICATIVE ONLY until the rebuilt track is validated. This is a probabilistic estimate for a "
                "decision framework, not investment advice.")
OVERRIDES = "data/valuation_overrides.csv"
REFERENCE = "data/valuation_v41_reference.csv"


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
@dataclass
class Context:
    annual: pd.DataFrame            # ticker, field, period, value
    quarterly: pd.DataFrame         # ticker, field, period, value
    monthly: pd.DataFrame           # date x ticker closes
    funda: pd.DataFrame             # fundamentals_cache, indexed by ticker
    overrides: dict = field(default_factory=dict)
    peer_cagr3: dict = field(default_factory=dict)


def _read(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def load_context(root: str = ".") -> Context:
    p = lambda f: os.path.join(root, f)
    A = _read(p("data/fundamentals_annual.csv.gz"))
    Q = _read(p("data/fundamentals_quarterly.csv.gz"))
    if Q.empty:
        Q = _read(p("analysis/screener_raw.csv.gz")).rename(columns={"quarter": "period"})
    for df in (A, Q):
        if not df.empty:
            df["period"] = pd.to_datetime(df["period"])
    M = _read(p("data/monthly_prices.csv.gz"))
    if not M.empty:
        M["date"] = pd.to_datetime(M["date"])
        M = M.pivot_table(index="date", columns="ticker", values="close").sort_index()
    F = _read(p("data/fundamentals_cache.csv"))
    F = F.set_index("ticker") if not F.empty else F
    ov = {}
    O = _read(p(OVERRIDES))
    for r in O.to_dict("records"):
        ov.setdefault(str(r["ticker"]).strip(), {})[str(r["key"]).strip()] = r["value"]
    ctx = Context(A, Q, M, F, ov)
    ctx.peer_cagr3 = _all_cagr3(A)
    return ctx


_GROUPS: dict = {}


def _series(df, ticker, fld):
    """One (ticker, field) series, from a per-frame index built once (the frames are large)."""
    key = id(df)
    if key not in _GROUPS:
        _GROUPS.clear() if len(_GROUPS) > 8 else None
        _GROUPS[key] = {k: g.set_index("period")["value"].sort_index()
                        for k, g in df.groupby(["ticker", "field"])} if not df.empty else {}
    s = _GROUPS[key].get((ticker, fld), pd.Series(dtype=float, index=pd.DatetimeIndex([])))
    return s[~s.index.duplicated(keep="last")]


def _fy(s):
    """Annual series restricted to fiscal year ends (month-end columns that repeat yearly)."""
    if s.empty:
        return s
    m = s.index.month.value_counts().idxmax()
    return s[s.index.month == m]


def _all_cagr3(A):
    out = {}
    if A.empty:
        return out
    S = A[A.field == "sales"]
    for t, g in S.groupby("ticker"):
        s = _fy(g.set_index("period")["value"].sort_index())
        if len(s) >= 4 and s.iloc[-4] > 0 and s.iloc[-1] > 0:
            out[t] = (s.iloc[-1] / s.iloc[-4]) ** (1 / 3) - 1
    return out


# ----------------------------------------------------------------------------
# triangular bands and the Phase 2D engine
# ----------------------------------------------------------------------------
def tri_inv(u, a, c, b):
    """Inverse CDF of Tri(a, c, b), vectorized over u (a <= c <= b)."""
    a, c, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(c, float), np.asarray(b, float))
    w = np.maximum(b - a, 1e-12)
    fc = (c - a) / w
    lo = a + np.sqrt(np.clip(u * w * (c - a), 0, None))
    hi = b - np.sqrt(np.clip((1 - u) * w * (b - c), 0, None))
    return np.where(u < fc, lo, hi)


def _pad(a, c, b):
    """Minimum band width 2% of the mode (zero-width bands crash the inverse)."""
    need = 0.02 * abs(c)
    if b - a < need:
        extra = (need - (b - a)) / 2
        a, b = a - extra, b + extra
    return a, c, b


def bands(bear, base, bull, floor=None):
    """(a, c, b) per regime for one lever value set. BEAR lower edge floored at 0.25*bear, never < 0."""
    lo = bear - 0.5 * (base - bear)
    lo = max(lo, 0.25 * bear, 0.0) if bear > 0 else max(lo, 0.0)
    if floor is not None:
        lo = max(lo, min(floor, bear))
    return [_pad(lo, bear, base), _pad(bear, base, bull), _pad(base, bull, bull + 0.5 * (bull - base))]


def tri_mean(t):
    return (t[0] + t[1] + t[2]) / 3


def simulate(lever, mult, probs, rho, w, r, n=N_DEFAULT, seed=SEED, mult_floor=None, link=True, rho_regime=None):
    """
    lever: {"bear": [y1,y2,y3], "base": [...], "bull": [...]}; mult: {"bear": m, ...}
    probs: (p_bear, p_base, p_bull) summing to 1. Returns V and per-trial regime.
    link=False draws the multiple independently of the lever (Gate C implementation check).
    """
    rng = np.random.default_rng(seed)
    rr = rho if rho_regime is None else rho_regime
    ze = rng.standard_normal(n)
    zr = rr * ze + np.sqrt(1 - rr ** 2) * rng.standard_normal(n)
    e2 = rng.standard_normal(n)
    zp = (rho * ze + np.sqrt(1 - rho ** 2) * e2) if link else e2
    ur, ue, up = ndtr(zr), ndtr(ze), ndtr(zp)
    reg = np.where(ur < probs[0], 0, np.where(ur < probs[0] + probs[1], 1, 2))
    mb = bands(mult["bear"], mult["base"], mult["bull"], floor=mult_floor)
    m = np.empty(n)
    for k in range(3):
        sel = reg == k
        m[sel] = tri_inv(up[sel], *mb[k])
    V = np.zeros(n)
    for j, h in enumerate(H):
        lb = bands(lever["bear"][j], lever["base"][j], lever["bull"][j])
        lev = np.empty(n)
        for k in range(3):
            sel = reg == k
            lev[sel] = tri_inv(ue[sel], *lb[k])
        V += w[j] * lev * m / (1 + r) ** h
    return V, reg


def blends(lever, mult, w, r):
    """Deterministic scenario blends at the modes."""
    return {s: float(sum(w[j] * lever[s][j] * mult[s] / (1 + r) ** h for j, h in enumerate(H)))
            for s in ("bear", "base", "bull")}


def analytic_mean(lever, mult, probs, w, r, mult_floor=None):
    mb = bands(mult["bear"], mult["base"], mult["bull"], floor=mult_floor)
    tot = 0.0
    for k, s in enumerate(("bear", "base", "bull")):
        em = tri_mean(mb[k])
        for j, h in enumerate(H):
            lb = bands(lever["bear"][j], lever["base"][j], lever["bull"][j])
            tot += probs[k] * w[j] * tri_mean(lb[k]) * em / (1 + r) ** h
    return tot


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _cagr(s, years):
    s = s.dropna()
    if len(s) <= years or s.iloc[-1 - years] <= 0 or s.iloc[-1] <= 0:
        return np.nan
    return (s.iloc[-1] / s.iloc[-1 - years]) ** (1 / years) - 1


def _verdict(pct, conf, gates_ok, severe_ok=True):
    if pct is None or not np.isfinite(pct):
        return "NOT VALUED"
    v = LADDER[-1][1]
    for cut, lab in LADDER:
        if pct < cut:
            v = lab
            break
    if gates_ok and severe_ok and conf in ("MEDIUM", "HIGH"):
        if pct < 5:
            v = "STRONG BUY"
        elif pct > 95:
            v = "STRONG SELL/AVOID"
    return v


def _order(lever: dict) -> dict:
    """bear <= 0.98 base <= bull / 1.02 at every horizon, bear >= 5% of base (bands must never invert)."""
    for j in range(3):
        b = lever["base"][j]
        lever["bear"][j] = float(min(max(lever["bear"][j], 0.05 * b), 0.98 * b))
        lever["bull"][j] = float(max(lever["bull"][j], 1.02 * b))
    return lever


def _ov(ov, key, default=np.nan, cast=float):
    if key not in ov:
        return default
    try:
        return cast(ov[key])
    except (TypeError, ValueError):
        return default


# ----------------------------------------------------------------------------
# the run
# ----------------------------------------------------------------------------
def value_stock(ticker: str, ctx: Context, price: float | None = None, n: int = N_DEFAULT,
                knobs: bool = True, hi52: float | None = None, lo52: float | None = None) -> dict:
    """Full mechanical v4.1 run for one stock. Returns a dict (see keys at the end)."""
    t = ticker if ticker.endswith(".NS") else ticker + ".NS"
    ov = ctx.overrides.get(t.replace(".NS", ""), ctx.overrides.get(t, {}))
    out = {"ticker": t.replace(".NS", ""), "engine": ENGINE, "notes": [], "label": "MECHANICAL"}
    note = out["notes"].append
    if ov:
        out["label"] = "MECHANICAL + OVERRIDES"

    f = ctx.funda.loc[t] if t in ctx.funda.index else None
    gran = (f["sector_granular"] if f is not None else "") or ""
    name = (f["name"] if f is not None else t) or t
    out.update(name=name, sector=f["sector"] if f is not None else "", sub_industry=gran)
    if price is None or not np.isfinite(price):
        return {**out, "status": "NO PRICE"}
    mcap = float(f["marketCap"]) if f is not None and pd.notna(f.get("marketCap")) else np.nan

    # ---- scope -----------------------------------------------------------
    ident = str(ov.get("identity", "")).upper() or None
    if ident == "OUT" or gran in OUT_GRAN or any(k in name.upper() for k in (" REIT", "INVIT", " TRUST")):
        why = OUT_GRAN.get(gran, "REIT/InvIT/trust" if gran not in OUT_GRAN else gran)
        return {**out, "status": "OUT OF SCOPE", "reason": f"{why} -- outside the framework's identities"}
    q_np, q_sales = _series(ctx.quarterly, t, "np"), _series(ctx.quarterly, t, "sales")
    q_np = q_np[q_np.index.month.isin([3, 6, 9, 12])]
    q_sales = q_sales[q_sales.index.month.isin([3, 6, 9, 12])]
    if len(q_np) < 4:
        return {**out, "status": "OUT OF SCOPE", "reason": "fewer than 4 reported quarters"}
    ttm_np = float(q_np.iloc[-4:].sum())
    ttm_sales = float(q_sales.iloc[-4:].sum()) if len(q_sales) >= 4 else np.nan
    if ttm_np <= 0:
        return {**out, "status": "OUT OF SCOPE", "reason": "TTM net profit is not positive -- EPS x PE undefined"}
    sc_mcap, sc_px = _series(ctx.annual, t, "mcap_cr"), _series(ctx.annual, t, "price_scr")
    if len(sc_mcap) and len(sc_px) and sc_px.iloc[-1] > 0:
        shares = float(sc_mcap.iloc[-1] / sc_px.iloc[-1])        # crore shares, Screener, same moment
        mcap = shares * price * 1e7
    elif not np.isfinite(mcap) or mcap <= 0:
        return {**out, "status": "NO DATA", "reason": "market cap missing"}
    else:
        shares = mcap / 1e7 / price                              # fallback: cache market cap / live price
        note("share count from the cached market cap and today's price (Screener headline missing) -- "
             "off by the price move since the cache date")
    ttm_eps = ttm_np / shares

    A = lambda fld: _fy(_series(ctx.annual, t, fld))
    a_sales, a_np = A("sales"), A("np")
    a_book = (A("eq_cap") + A("reserves")).dropna()
    a_cfo, a_cfi, a_roce, a_pay = A("cfo"), A("cfi"), A("roce"), A("payout")
    a_borr = A("borrowings")
    years = int(a_np.dropna().shape[0])
    if ident is None:
        ident = ("BFSI" if gran in BFSI_GRAN else "COMMODITY" if gran in COMMODITY_GRAN else "INDUSTRIAL")
    if gran in MIXED_GRAN and "identity" not in ov:
        note("Diversified group: may need a mini-SOTP (>10% PAT from a different multiple class) -- "
             "set it via overrides; run as INDUSTRIAL")
    out["identity"] = ident

    # ---- Phase 0 snapshot ------------------------------------------------------
    if hi52 is None or lo52 is None or not (np.isfinite(hi52) and np.isfinite(lo52)):
        mp = ctx.monthly[t].dropna() if t in getattr(ctx.monthly, "columns", []) else pd.Series(dtype=float)
        mp = mp[mp.index >= mp.index.max() - pd.DateOffset(months=12)] if len(mp) else mp
        hi52 = float(max(mp.max(), price)) if len(mp) else np.nan
        lo52 = float(min(mp.min(), price)) if len(mp) else np.nan
    hl_mid = (hi52 - lo52) / ((hi52 + lo52) / 2) if np.isfinite(hi52) and np.isfinite(lo52) else np.nan
    book = float(a_book.iloc[-1]) if len(a_book) else np.nan
    bvps = book / shares if np.isfinite(book) else (price / float(f["pb"]) if f is not None and f.get("pb") else np.nan)
    margins = (a_np / a_sales).replace([np.inf, -np.inf], np.nan).dropna()
    roe_hist = (a_np / ((a_book + a_book.shift(1)) / 2)).dropna()
    roce_now = float(a_roce.dropna().iloc[-1]) / 100 if len(a_roce.dropna()) else np.nan
    cfo_pat = float(a_cfo.iloc[-3:].sum() / a_np.iloc[-3:].sum()) if len(a_cfo) >= 3 and a_np.iloc[-3:].sum() > 0 else np.nan
    fcf_pat = (float((a_cfo + a_cfi).dropna().iloc[-3:].sum() / a_np.iloc[-3:].sum())
               if len((a_cfo + a_cfi).dropna()) >= 3 and a_np.iloc[-3:].sum() > 0 else np.nan)
    de = float(a_borr.iloc[-1] / a_book.iloc[-1]) if len(a_borr) and len(a_book) and a_book.iloc[-1] > 0 else np.nan
    spot_pe = price / ttm_eps
    spot_pb = price / bvps if np.isfinite(bvps) and bvps > 0 else np.nan
    out["snapshot"] = dict(price=price, shares_cr=shares, mcap_cr=mcap / 1e7, ttm_eps=ttm_eps, ttm_np_cr=ttm_np,
                           ttm_sales_cr=ttm_sales, bvps=bvps, spot_pe=spot_pe, spot_pb=spot_pb, roce=roce_now,
                           cfo_pat_3y=cfo_pat, fcf_pat_3y=fcf_pat, de=de, hi52=hi52, lo52=lo52, hl_mid=hl_mid,
                           years=years, quarters=int(len(q_np)),
                           sales_cagr3=_cagr(a_sales, 3), sales_cagr5=_cagr(a_sales, 5),
                           np_cagr3=_cagr(a_np, 3), margin_ttm=ttm_np / ttm_sales if ttm_sales else np.nan)

    # ---- own multiple history (positive-earnings months) --------------------------
    own = pd.Series(dtype=float)
    if t in getattr(ctx.monthly, "columns", []) and years:
        px = ctx.monthly[t].dropna()
        px = px[px.index >= px.index.max() - pd.DateOffset(years=5)]
        if ident == "BFSI":
            src = a_book
        elif ident == "COMMODITY" and len(margins) >= 3:
            src = a_sales * float(margins.median())        # normalized (mid-cycle) earnings, not spot
        else:
            src = a_np
        avail = pd.Series(src.values, index=src.index + pd.Timedelta(days=60)).sort_index()
        lever_m = avail.reindex(px.index, method="ffill") / shares
        own = (px / lever_m)[lever_m > 0].dropna()
    # ---- peers (same sub-industry) -----------------------------------------------
    col = "pb" if ident == "BFSI" else "pe"
    pool = ctx.funda[(ctx.funda.sector_granular == gran) & (ctx.funda.index != t)] if gran else ctx.funda.iloc[:0]
    peer_vals = pd.to_numeric(pool[col], errors="coerce")
    peer_vals = peer_vals[(peer_vals > 0) & (peer_vals < (60 if col == "pb" else 400))]
    sec_vals = pd.to_numeric(ctx.funda[ctx.funda.sector == out["sector"]][col], errors="coerce")
    sec_vals = sec_vals[(sec_vals > 0) & (sec_vals < (60 if col == "pb" else 400))]
    peer_med = float(peer_vals.median()) if len(peer_vals) >= 3 else (float(sec_vals.median()) if len(sec_vals) >= 3 else np.nan)
    sector_med = float(sec_vals.median()) if len(sec_vals) >= 3 else peer_med
    if len(peer_vals) < 3:
        note(f"fewer than 3 sub-industry peers -- peer anchor uses the {out['sector']} sector median")
    own_med = float(own.median()) if len(own) >= 12 else np.nan
    own_p10 = float(own.quantile(0.10)) if len(own) >= 12 else np.nan
    own_min = float(own.min()) if len(own) >= 12 else np.nan

    # ---- confidence ----------------------------------------------------------------
    conf = "MEDIUM" if (years >= 5 and len(own) >= 36 and len(peer_vals) >= 5 and len(q_np) >= 8) else "LOW"
    conf = str(ov.get("confidence", conf)).upper()
    widen = 1.25 if conf == "LOW" else 1.10 if conf == "MEDIUM" else 1.0
    out["confidence"] = conf

    # ---- 1C probability scorecard (numeric rows only) -------------------------------
    sc = {}
    pro = _series(ctx.quarterly, t, "promoters")
    if len(pro) >= 5:
        d = pro.iloc[-1] - pro.iloc[-5]
        sc["Promoter trend (4Q)"] = (1 if d >= 1 else -1 if d <= -2 else 0, f"{d:+.1f} pp")
    inst = _series(ctx.quarterly, t, "fii").add(_series(ctx.quarterly, t, "dii"), fill_value=0)
    if len(inst) >= 5:
        d = inst.iloc[-1] - inst.iloc[-5]
        sc["FII+DII trend (4Q)"] = (1 if d >= 2 else -1 if d <= -2 else 0, f"{d:+.1f} pp")
    if len(q_np) >= 8:
        yoy = (q_np.iloc[-4:].values - q_np.iloc[-8:-4].values) / np.abs(q_np.iloc[-8:-4].values)
        hits = int((yoy > 0).sum())
        s = 1 if hits >= 3 else -1 if hits <= 1 else 0
        if (yoy < -0.5).any():
            s = min(s, 0)
        sc["Quarterly execution"] = (s, f"{hits}/4 quarters with YoY profit growth (basis: same quarter last "
                                        f"year, no estimates available){'; catastrophic miss caps at 0' if (yoy < -0.5).any() else ''}")
    if np.isfinite(fcf_pat):
        s = 1 if fcf_pat >= 0.5 else (0 if (fcf_pat < 0 and np.isfinite(cfo_pat) and cfo_pat >= 0.8) else -1 if fcf_pat < 0 else 0)
        sc["FCF quality (3y)"] = (s, f"FCF/PAT {fcf_pat:.2f}, CFO/PAT {cfo_pat:.2f}"
                                    + (" -- negative FCF with strong CFO read as growth capex (0, not -1)" if s == 0 and fcf_pat < 0 else ""))
    if ident != "BFSI" and np.isfinite(de):
        sc["Balance sheet D/E"] = (1 if de <= 0.5 else -1 if de > 1.5 else 0, f"D/E {de:.2f}")
    for row in ("Guidance record", "CEO/MD change", "CFO change", "Governance (ledger)", "Pledge") + \
               (("Asset quality", "Capital adequacy", "Funding & liquidity") if ident == "BFSI" else ()):
        sc[row] = (0, "NOT SCORED (no numeric feed) -- 0")
    total = sum(v[0] for v in sc.values())
    total_r = int(np.trunc(total))                    # rounds toward zero
    p = next(pr for cut, pr in PROB_MAP if total_r >= cut)
    pb, pbase, pbu = [x / 100 for x in p]
    adj_b, adj_u = _ov(ov, "p_bear_pp", 0.0), _ov(ov, "p_bull_pp", 0.0)
    adj_b, adj_u = float(np.clip(adj_b, -20, 20)) / 100, float(np.clip(adj_u, -20, 20)) / 100
    if adj_b or adj_u:
        pb, pbu = max(pb + adj_b, 0.02), max(pbu + adj_u, 0.02)
        pbase = max(1 - pb - pbu, 0.05)
        s_ = pb + pbase + pbu
        pb, pbase, pbu = pb / s_, pbase / s_, pbu / s_
        note(f"event adjustment p_bear {adj_b * 100:+.0f}pp / p_bull {adj_u * 100:+.0f}pp ({ov.get('event_note', 'override')})")
    probs = (pb, pbase, pbu)
    out["scorecard"] = {k: {"score": v[0], "evidence": v[1]} for k, v in sc.items()}
    out["score_total"], out["probs"] = total, probs

    # ---- Phase 1 scenarios -------------------------------------------------------------
    g_hist = a_sales.pct_change().dropna().iloc[-5:]
    ttm_yoy = (float(q_sales.iloc[-4:].sum() / q_sales.iloc[-8:-4].sum() - 1)
               if len(q_sales) >= 8 and q_sales.iloc[-8:-4].sum() > 0 else np.nan)
    parts = [(0.4, _cagr(a_sales, 3)), (0.3, _cagr(a_sales, 5)), (0.3, ttm_yoy)]
    parts = [(wt, v) for wt, v in parts if np.isfinite(v)]
    base_g = sum(wt * v for wt, v in parts) / sum(wt for wt, _ in parts) if parts else 0.08
    base_g = float(np.clip(base_g, -0.05, 0.35))
    # robust spread of annual growth (IQR/1.35, not the std a single boom/bust year inflates), 4-15pp
    sd_g = float(np.clip((g_hist.quantile(0.75) - g_hist.quantile(0.25)) / 1.35, 0.04, 0.15)) if len(g_hist) >= 3 else 0.08
    # scenarios are 3-YEAR growth paths: the spread of a 3-year average is the annual spread / sqrt(3)
    sd_g = sd_g / np.sqrt(3)
    g = {"bear": float(np.clip(base_g - sd_g, -0.15, base_g - 0.02)), "base": base_g,
         "bull": float(np.clip(base_g + sd_g, base_g + 0.02, 0.50))}
    for s in g:
        g[s] = _ov(ov, f"g_{s}", g[s])
    q_marg = (q_np.rolling(4).sum() / q_sales.rolling(4).sum()).dropna() if len(q_sales) >= 4 else pd.Series(dtype=float)
    m_ttm = ttm_np / ttm_sales if ttm_sales and ttm_sales > 0 else np.nan
    hist_m = margins.iloc[-6:]
    # F2: the worst demonstrated run-rate of the LAST stress -- last 5 fiscal years and the quarterly
    # rolling-4Q record -- not the worst year in a decade-plus of history
    all_m = pd.concat([hist_m, q_marg])
    worst = float(all_m.min()) if len(all_m) else np.nan
    lever, path_note = {}, ""
    if ident == "BFSI":
        roe_ttm = ttm_np / book if np.isfinite(book) and book > 0 else np.nan
        rh = roe_hist.iloc[-5:]
        base_roe = float(np.nanmedian(list(rh) + [roe_ttm])) if len(rh) else roe_ttm
        bear_roe = float(min(rh.min() if len(rh) else roe_ttm, roe_ttm) * 0.85)
        bull_roe = float(max(rh.max() if len(rh) else roe_ttm, roe_ttm))
        bear_roe = max(bear_roe, 0.02)
        roe = {"bear": _ov(ov, "roe_bear", bear_roe), "base": _ov(ov, "roe_base", base_roe),
               "bull": _ov(ov, "roe_bull", bull_roe)}
        pay = float(a_pay.dropna().iloc[-5:].median()) / 100 if len(a_pay.dropna()) else 0.15
        for s in roe:
            b_, path = bvps, []
            for _h in H:
                b_ = b_ * (1 + roe[s] * (1 - pay))
                path.append(b_)
            lever[s] = path
        out["scenario_inputs"] = {"ROE": roe, "payout": pay, "BVPS0": bvps}
        path_note = "BVPS_h = BVPS_{h-1} x (1 + ROE x (1 - payout)); no dilution modelled"
        if len(rh) and rh.min() < 0.5 * np.median(rh):
            note("stress year in the ROE record -- base uses the median (one stress year does not set the base)")
    else:
        if ident == "COMMODITY":
            mid = float(margins.median()) if len(margins) >= 3 else m_ttm
            m_term = {"bear": worst if np.isfinite(worst) and worst > 0 else
                      (float(margins.quantile(0.25)) if len(margins) >= 4 else 0.7 * mid), "base": mid,
                      "bull": float(margins.quantile(0.75)) if len(margins) >= 4 else 1.3 * mid}
            g = {k: float(np.clip(v, -0.10, 0.10)) for k, v in g.items()}
            out["cycle"] = dict(mid_margin=mid, ttm_margin=m_ttm,
                                mid_eps=ttm_sales * mid / shares if ttm_sales else np.nan)
        else:
            base_m = float(hist_m.median()) if len(hist_m) else m_ttm
            if len(q_marg) >= 4 and np.isfinite(m_ttm) and m_ttm > base_m and (q_marg.iloc[-4:] > base_m).all():
                base_m = (base_m + m_ttm) / 2
                note("structural-improvement override: 4 straight quarters of TTM margin above the 5Y median -- "
                     "base terminal margin lifted halfway to TTM")
            m_term = {"bear": worst, "base": base_m,
                      "bull": float(max(hist_m.max() if len(hist_m) else m_ttm, m_ttm))}
        if not np.isfinite(m_term["bear"]) or m_term["bear"] <= 0:
            m_term["bear"] = max(0.25 * m_term["base"], float(all_m[all_m > 0].min()) if (all_m > 0).any() else 0.25 * m_term["base"])
            note("losses in the record: bear margin floored positive (EPS x PE is meaningless at negative earnings) "
                 "-- Gate F2 deviation, defended")
            out["f2_deviation"] = True
        for s in m_term:
            m_term[s] = _ov(ov, f"m_{s}", m_term[s])
        m_term["bear"] = min(m_term["bear"], m_term["base"] * 0.98)
        m_term["bull"] = max(m_term["bull"], m_term["base"] * 1.02)
        for s in ("bear", "base", "bull"):
            path = []
            for h in H:
                sales_h = ttm_sales * (1 + g[s]) ** h
                mh = m_term[s] if ident == "COMMODITY" else m_ttm + (m_term[s] - m_ttm) * h / 3
                path.append(sales_h * mh / shares)
            lever[s] = path
        out["scenario_inputs"] = {"growth": g, "terminal_margin": m_term, "margin_ttm": m_ttm, "worst_margin": worst}
        path_note = ("EPS_h = TTM sales x (1+g)^h x margin_h / shares; margin moves linearly from TTM to the "
                     "terminal by Y3" if ident != "COMMODITY" else
                     "COMMODITY: EPS at mid-cycle (through-cycle) margins from Y1; scenarios are structural, not cycle timing")
    # widen spreads for confidence (pre-sim, Gate E rule)
    if widen != 1.0:
        for j in range(3):
            b = lever["base"][j]
            lever["bear"][j] = b - widen * (b - lever["bear"][j])
            lever["bull"][j] = b + widen * (lever["bull"][j] - b)
            lever["bear"][j] = max(lever["bear"][j], 0.05 * b)
    for s in lever:
        if lever[s][0] > lever[s][-1] * 1.0001 or (lever[s][1] < min(lever[s][0], lever[s][2]) - 1e-9):
            note(f"{s} path is non-monotonic ({', '.join(f'{x:.1f}' for x in lever[s])}) -- the Y3 figure is not a run-rate")
            break
    if not all(np.isfinite(lever["base"])):
        return {**out, "status": "NO DATA", "reason": "base earnings path not computable (missing sales or margins)"}
    if not (all(np.isfinite(lever["bear"])) and all(np.isfinite(lever["bull"]))):
        lever["bear"] = [0.7 * b for b in lever["base"]]
        lever["bull"] = [1.3 * b for b in lever["base"]]
        note("bear/bull margins missing in the source data -- scenarios set at +/-30% of base (DATA caution)")
        out["data_fallback"] = True
    lever = _order(lever)
    out["lever"] = {s: [float(x) for x in v] for s, v in lever.items()}
    out["path_note"] = path_note
    # 1F sanity
    lev0 = ttm_eps if ident != "BFSI" else bvps
    ratio = lever["base"][0] / lev0 if lev0 else np.nan
    out["sanity_1f"] = dict(ratio=ratio, ok=bool(ratio < 2.5) if np.isfinite(ratio) else False,
                            flag=bool(1.5 <= ratio < 2.5) if np.isfinite(ratio) else False)

    # ---- Phase 2C multiple model ---------------------------------------------------------
    anchors = [(v, w_) for v, w_ in ((own_med, 0.5), (peer_med, 0.5)) if np.isfinite(v)]
    if not anchors:
        return {**out, "status": "NO DATA", "reason": "no own multiple history and no peers"}
    neutral = sum(v * w_ for v, w_ in anchors) / sum(w_ for _, w_ in anchors)
    if len(anchors) == 1:
        note("only one multiple anchor available (" + ("own 5Y" if np.isfinite(own_med) else "peers") + ")")
    diverge = (abs(own_med - peer_med) / min(own_med, peer_med)
               if np.isfinite(own_med) and np.isfinite(peer_med) else 0.0)
    mods = {}
    if "Promoter trend (4Q)" in sc:
        mods["promoter trend"] = 0.03 * sc["Promoter trend (4Q)"][0]
    if "FII+DII trend (4Q)" in sc:
        mods["FII/DII"] = 0.03 * sc["FII+DII trend (4Q)"][0]
    if len(a_np) >= 5:
        ups = int((a_np.diff().dropna().iloc[-4:] > 0).sum())
        mods["earnings predictability"] = 0.03 if ups == 4 else -0.03 if ups <= 1 else 0.0
    for k in ("management", "governance", "news flow", "analyst revisions", "policy/macro"):
        mods[k] = 0.0
    mods_sum = sum(mods.values()) + _ov(ov, "modifier_pct", 0.0) / 100
    base_m_ = neutral * (1 + mods_sum)
    ceiling = 1.3 * sector_med if np.isfinite(sector_med) else np.inf
    if ident != "BFSI" and np.isfinite(roce_now) and roce_now >= 0.25 and g["base"] >= 0.15:
        ceiling = max(ceiling, 1.6 * sector_med)
    capped = base_m_ > ceiling
    base_m_ = min(base_m_, ceiling)
    bear_m = own_p10 if np.isfinite(own_p10) else 0.7 * base_m_
    if ident == "BFSI" and np.isfinite(own_min):
        bear_m = float(np.clip(bear_m, 0.9 * own_min, 1.2 * own_min))
    bear_anchor = bear_m                                 # trough anchor before any spacing adjustment
    bull_m = 1.15 * base_m_
    spot_mult = spot_pb if ident == "BFSI" else spot_pe
    peer_g = ctx.peer_cagr3.get(t)
    peers_g = [ctx.peer_cagr3[x] for x in pool.index if x in ctx.peer_cagr3]
    share_gain = (peer_g is not None and len(peers_g) >= 3 and peer_g > float(np.median(peers_g)))
    rerate = bool(share_gain and np.isfinite(roce_now) and roce_now >= 0.20 and np.isfinite(cfo_pat) and cfo_pat >= 0.8)
    rerate = bool(_ov(ov, "rerating_pass", float(rerate)))
    precedence, indeterminate = False, False
    # Spot ceiling (durable re-rating branch). Mechanical reading: it binds only when the market already
    # pays at least the anchor-based BASE -- then a bull above today's multiple claims a RE-RATING and needs
    # the evidence test. Below BASE, a bull above spot is reversion toward the stock's own / peer anchor,
    # not a re-rating, so the ceiling does not apply (a judgment run decides this case by case).
    out["ceiling_applies"] = bool(np.isfinite(spot_mult) and spot_mult >= base_m_)
    if not rerate and out["ceiling_applies"] and bull_m >= spot_mult:
        bull_m = 0.98 * spot_mult                       # spot ceiling wins
        if 1.15 * base_m_ > bull_m:
            base_m_ = bull_m / 1.15                     # re-examine BASE
            precedence = True
            # BASE cannot go below the trough anchor without contradicting it: both constraints
            # cannot hold -> structurally indeterminate (a finding, not a failure)
            trough = own_min if np.isfinite(own_min) else 0.7 * peer_med
            if np.isfinite(trough) and base_m_ < trough:
                indeterminate = True
                note(f"multiple STRUCTURALLY INDETERMINATE: the spot ceiling needs BASE {base_m_:.1f}x, below the "
                     f"lowest multiple in the stock's own 5Y record ({trough:.1f}x) -- both constraints are printed")
    bear_m = min(bear_anchor, 0.9 * base_m_)
    mult = {"bear": _ov(ov, "mult_bear", bear_m), "base": _ov(ov, "mult_base", base_m_),
            "bull": _ov(ov, "mult_bull", bull_m)}
    # Book floor for a bear branch that reaches (near-)losses: EPS x PE says ~0 there, which is
    # meaningless for a going concern; v4.1 floors that branch with book logic -> bear value >= 0.8 x BVPS
    if ident != "BFSI" and np.isfinite(bvps) and bvps > 0:
        floor_eps = 0.8 * bvps / mult["bear"]
        if any(x < floor_eps for x in lever["bear"]):
            # the floor lifts the bear branch only -- never to or above base (a book floor above the base case
            # means EPS x PE undervalues the book for this company; that is disclosed, not forced into base)
            lever["bear"] = [max(x, min(floor_eps, 0.95 * b_)) for x, b_ in zip(lever["bear"], lever["base"])]
            lever = _order(lever)
            out["lever"] = {s_: [float(x) for x in v_] for s_, v_ in lever.items()}
            note(f"bear branch floored at 0.8x book (BVPS {bvps:.1f}): its earnings path reaches near-losses, "
                 "where EPS x PE is meaningless")
    mult_floor = max(0.8, 0.9 * own_min) if ident == "BFSI" and np.isfinite(own_min) else None
    out["multiple"] = dict(kind="P/B" if ident == "BFSI" else "PE", own_5y_median=own_med, own_5y_p10=own_p10,
                           own_5y_min=own_min, own_months=int(len(own)), peer_median=peer_med,
                           peers=int(len(peer_vals)), sector_median=sector_med, neutral=neutral,
                           modifiers=mods, modifiers_sum=mods_sum, ceiling=ceiling, capped=capped,
                           spot=spot_mult, rerating_test=rerate, share_gain=share_gain,
                           precedence=precedence, indeterminate=indeterminate, anchor_divergence=diverge, **mult)
    if precedence:
        note("BASE multiple set by the precedence rule (spot ceiling wins over the 1.15x bull floor), "
             "not by the modifier table")

    # ---- rho, hurdle, weights ---------------------------------------------------------------
    if ident in ("BFSI", "COMMODITY"):
        rho = 0.25
    elif np.isfinite(fcf_pat) and fcf_pat >= 0.5 and np.isfinite(roce_now) and roce_now >= 0.20:
        rho = 0.50
    else:
        rho = 0.35
    rho = _ov(ov, "rho", rho)
    large = mcap / 1e7 >= 100_000
    small = mcap / 1e7 < 5_000
    predictable = len(a_np) >= 5 and int((a_np.diff().dropna().iloc[-4:] > 0).sum()) >= 3
    if ident == "BFSI":
        r = 0.13 + (0.01 if small else 0.0)
    else:
        r = 0.12 - (0.01 if (large and np.isfinite(de) and de < 0.3 and predictable) else 0.0) + (0.01 if small else 0.0)
    r = _ov(ov, "hurdle", r)
    wkey = str(ov.get("weights", "")).upper()
    if wkey not in WEIGHTS:
        wkey = "BACK" if (ident == "COMMODITY" and np.isfinite(m_ttm) and m_ttm > out.get("cycle", {}).get("mid_margin", np.inf)) else "BALANCED"
    w = WEIGHTS[wkey]
    out.update(rho=rho, hurdle=r, weights=wkey)

    # ---- Gate E as a FLOOR (mechanical runs) -----------------------------------------------------
    # v4.1 Gate E is "dispersion & floors": the fair-value range may not be narrower than 0.8x the
    # stock's own 52-week price range. A judgment run would revisit its bands; the mechanical run
    # widens bear/bull spreads around the unchanged base (levers and multiples alike) up to the floor,
    # at most 2x, and says so. Base case, probabilities and the CMP are untouched.
    out["e_widened"] = 1.0
    if np.isfinite(hl_mid) and hl_mid > 0:
        V_, _ = simulate(lever, mult, probs, rho, w, r, n=4000, mult_floor=mult_floor)
        q10, q50, q90 = np.percentile(V_, [10, 50, 90])
        disp0 = (q90 - q10) / q50 if q50 > 0 else np.nan
        if np.isfinite(disp0) and 0 < disp0 < 0.8 * hl_mid:
            k = float(min(2.0, 0.8 * hl_mid / disp0 * 1.05))
            for j in range(3):
                b = lever["base"][j]
                lever["bear"][j] = max(b - k * (b - lever["bear"][j]), 0.05 * b)
                lever["bull"][j] = b + k * (lever["bull"][j] - b)
            mb_ = mult["base"]
            mult = {"bear": max(mb_ - k * (mb_ - mult["bear"]), 0.3 * mb_), "base": mb_,
                    "bull": mb_ + k * (mult["bull"] - mb_)}
            lever = _order(lever)
            out["lever"] = {s_: [float(x) for x in v_] for s_, v_ in lever.items()}
            out["multiple"].update(mult)
            out["e_widened"] = k
            note(f"fair-value range widened x{k:.2f} around the unchanged base case to the price-volatility floor "
                 f"(Gate E: range {disp0:.2f} vs 0.8 x 52-week range {0.8 * hl_mid:.2f})")

    # ---- Phase 2D simulation --------------------------------------------------------------------
    V, reg = simulate(lever, mult, probs, rho, w, r, n=n, mult_floor=mult_floor)
    pct = float((V < price).mean() * 100)
    qs = np.percentile(V, [5, 10, 20, 25, 40, 50, 70, 75, 88, 90, 95])
    Q = dict(zip(["P5", "P10", "P20", "P25", "P40", "P50", "P70", "P75", "P88", "P90", "P95"], map(float, qs)))
    bl = blends(lever, mult, w, r)
    am = analytic_mean(lever, mult, probs, w, r, mult_floor)
    out["dist"] = dict(Q, mean=float(V.mean()), sd=float(V.std()), cmp_pct=pct, n=n, seed=SEED)
    out["grid"] = [float(x) for x in np.percentile(V, np.arange(1, 100))]
    out["blends"] = bl
    out["regimes"] = {s: dict(share=float((reg == k).mean()), mean=float(V[reg == k].mean()) if (reg == k).any() else np.nan)
                      for k, s in enumerate(("bear", "base", "bull"))}
    y3 = {s: lever[s][2] * mult[s] for s in lever}
    dy = (float(a_pay.dropna().iloc[-3:].median()) / 100 * ttm_eps / price) if len(a_pay.dropna()) else 0.0
    out["cagr"] = {s: float((y3[s] / price) ** (1 / 3) - 1 + dy) if y3[s] > 0 else np.nan for s in y3}
    out["div_yield"] = dy

    # ---- gates ---------------------------------------------------------------------------------
    gates = {}
    disc = (bl["bull"] - bl["base"]) / bl["base"] if bl["base"] else np.nan
    a_ok = Q["P90"] >= 0.90 * bl["bull"] and Q["P10"] <= 1.10 * bl["bear"]
    gates["A"] = dict(ok=bool(a_ok), informative=bool(disc >= 0.15),
                      detail=f"P90 {Q['P90']:.0f} vs 0.9x bull blend {0.9 * bl['bull']:.0f}; P10 {Q['P10']:.0f} vs "
                             f"1.1x bear blend {1.1 * bl['bear']:.0f}; bull-base spacing {disc:.0%}"
                             + ("" if disc >= 0.15 else " -- NON-INFORMATIVE (<15%), rely on Gate F"))
    mass = float((V < bl["base"]).mean())
    target = pb + pbase / 2
    gates["B"] = dict(ok=bool(abs(mass - target) <= 0.10), detail=f"P(V<base blend) {mass:.0%} vs {target:.0%} +/-10pp")
    drift = am / (sum(probs[k] * bl[s] for k, s in enumerate(("bear", "base", "bull")))) - 1
    # Implementation check: the same engine with the lever-multiple link switched off (rho = 0 for the
    # multiple draw) must reproduce the closed-form mixture mean within 1%. The rho covariance is the
    # model working as designed (P5) -- disclosed, not failed.
    V0, _ = simulate(lever, mult, probs, 0.0, w, r, n=n, mult_floor=mult_floor, link=False, rho_regime=0.0)
    impl = V0.mean() / am - 1
    cov = V.mean() / V0.mean() - 1
    se = float(V0.std() / np.sqrt(n) / am) if am else 0.0      # Monte Carlo standard error of the mean
    gates["C"] = dict(ok=bool(abs(impl) <= max(0.01, 3 * se)), detail=f"engine mean (independent multiple draw) {V0.mean():.0f} vs "
                      f"analytic mixture mean {am:.0f} ({impl:+.2%}, tolerance max(1%, 3 x MC s.e. {se:.2%})); rho-covariance residual {cov:+.2%} "
                      f"(disclosed); scenario-asymmetry drift {drift:+.1%} (information, not a fail)")
    trig = np.isfinite(spot_mult) and spot_mult > mult["bull"]
    near = np.isfinite(spot_mult) and not trig and spot_mult >= 0.9 * mult["bull"]
    cf = sum(w[j] * lever["base"][j] * spot_mult / (1 + r) ** h for j, h in enumerate(H)) if np.isfinite(spot_mult) else np.nan
    gates["D"] = dict(ok=True, triggered=bool(trig), near=bool(near), counterfactual=float(cf),
                      detail=(f"spot {spot_mult:.1f}x vs bull {mult['bull']:.1f}x" +
                              (f" -- MULTIPLE-DE-RATING CALL; base path at today's multiple = {cf:.0f}"
                               + (" (above CMP)" if cf > price else "") if trig else
                               f" -- near trigger; counterfactual {cf:.0f}" if near else "")))
    disp = (Q["P90"] - Q["P10"]) / Q["P50"] if Q["P50"] > 0 else np.nan
    e_ok = (not np.isfinite(hl_mid)) or disp >= 0.8 * hl_mid
    if conf == "LOW" and (pct > 95 or pct < 5):
        e_ok = False
    gates["E"] = dict(ok=bool(e_ok), detail=f"(P90-P10)/P50 {disp:.2f} vs 0.8 x 52w (H-L)/mid {0.8 * hl_mid:.2f}"
                      + ("; extreme percentile with LOW confidence" if conf == "LOW" and (pct > 95 or pct < 5) else ""))
    f1_ok = not (("p_bear_pp" in ov) and not any(k in ov for k in ("g_bear", "m_bear", "roe_bear", "mult_bear")))
    if ident == "BFSI":
        f2_ok = (not np.isfinite(own_min)) or mult["bear"] <= 1.2 * own_min + 1e-9
        f2d = f"bear P/B {mult['bear']:.2f} vs 1.2x own 5Y trough {1.2 * own_min:.2f}" if np.isfinite(own_min) else "no P/B history"
    else:
        bear_term = out["scenario_inputs"]["terminal_margin"]["bear"]
        if np.isfinite(worst) and worst <= 0 and not out.get("f2_deviation"):
            out["f2_deviation"] = True
            note("worst demonstrated run-rate is a LOSS: the bear path stops at a small positive margin because "
                 "EPS x PE is undefined below zero -- Gate F2 deviation, defended")
        f2_ok = (not np.isfinite(worst)) or bear_term <= max(worst, 0) * 1.10 + 1e-9 or out.get("f2_deviation", False)
        f2d = (f"bear terminal margin {bear_term:.1%} vs worst demonstrated run-rate {worst:.1%}"
               + (" (losses: floored, defended)" if out.get("f2_deviation") else ""))
    gates["F"] = dict(ok=bool(f1_ok and f2_ok), detail=("F1 " + ("ok: bear path is the pivot case" if f1_ok else
                      "FAIL: event probability shifted without changing the bear earnings path") + "; F2 " + f2d))
    out["gates"] = gates
    gates_ok = all(gates[k]["ok"] for k in "ABCEF") and out["sanity_1f"]["ok"]
    out["gates_ok"] = gates_ok

    # ---- verdict -----------------------------------------------------------------------------
    v = _verdict(pct, conf, gates_ok)
    if indeterminate:
        out["provisional"] = f"({v}) multiple indeterminate"
        v = "INDETERMINATE (multiple)"
    if not gates_ok:
        failed = [k for k in "ABCEF" if not gates[k]["ok"]] + ([] if out["sanity_1f"]["ok"] else ["1F"])
        out["rating"] = f"UNRATED (gate {', '.join(failed)})"
        out["provisional"] = f"({v}) gate {', '.join(failed)}"
    else:
        out["rating"] = v + (" · INDICATIVE" if ident == "COMMODITY" else "")
    out["verdict_raw"] = v
    out["quality"] = quality_profile(out, gates, Q, bl, mass, target, indeterminate, conf, ident)
    out["ladder"] = dict(deep_add=Q["P10"], accumulate=Q["P25"], fair_lo=Q["P25"], fair_hi=Q["P75"], trim=Q["P90"])
    pivot = "multiple" if ident != "BFSI" and np.std([mult[s] for s in mult]) / mult["base"] > np.std(
        [lever[s][2] for s in lever]) / lever["base"][2] else ("ROE / book path" if ident == "BFSI" else "margin path")
    out["pivot"] = pivot
    if ident == "COMMODITY":
        c_ = out["cycle"]
        out["cycle"]["statement"] = (f"TTM EPS {ttm_eps:.1f} is {ttm_eps / c_['mid_eps'] - 1:+.0%} vs the mid-cycle "
                                     f"{c_['mid_eps']:.1f}; trailing {spot_pe:.1f}x is {price / c_['mid_eps']:.1f}x on "
                                     f"mid-cycle earnings") if c_.get("mid_eps") else ""

    # ---- P4 knob sensitivity -------------------------------------------------------------------------
    if knobs:
        sens = {}
        def pct_with(lever_=lever, mult_=mult, rho_=rho, r_=r):
            V2, _ = simulate(lever_, mult_, probs, rho_, w, r_, n=4000, mult_floor=mult_floor)
            return float((V2 < price).mean() * 100)
        for lab, rr in (("hurdle -1pp", r - 0.01), ("hurdle +1pp", r + 0.01)):
            sens[lab] = pct_with(r_=rr)
        for lab, rh_ in (("rho 0.25", 0.25), ("rho 0.50", 0.50)):
            if abs(rh_ - rho) > 1e-9:
                sens[lab] = pct_with(rho_=rh_)
        if diverge > 0.25 and np.isfinite(own_med) and np.isfinite(peer_med):
            for lab, wo in (("anchor 65/35 own/peer", 0.65), ("anchor 35/65 own/peer", 0.35)):
                k = (wo * own_med + (1 - wo) * peer_med) / neutral
                sens[lab] = pct_with(mult_={s: mult[s] * k for s in mult})
        cuts = [c for c, _ in LADDER[:-1]]
        band = lambda x: sum(x >= c for c in cuts)
        out["sensitivity"] = sens
        out["knob_dependent"] = any(band(x) != band(pct) for x in sens.values())
        if out["knob_dependent"]:
            note("verdict is KNOB-DEPENDENT: a plausible knob setting moves CMP across a verdict boundary (P4)")
    out["status"] = "OK"
    out["engine_line"] = (f"{ENGINE}, rho={rho:.2f}, hurdle={r:.0%}, N={n}, seed={SEED}"
                          + ("; track=normalized-earnings (INDICATIVE)" if ident == "COMMODITY" else ""))
    out["known_limits"] = KNOWN_LIMITS
    return out


QUALITY = {0: ("✅", "Clean"), 1: ("🟡", "Caution"), 2: ("🟠", "Weak"), 3: ("🔴", "Unreliable")}


def quality_profile(out, gates, Q, bl, mass, target, indeterminate, conf, ident) -> dict:
    """
    Translate gate results into plain-language reasons, each with a severity and the DIRECTION it bends
    the verdict. Severity: 1 caution (read the reason), 2 weak (valuation counts one notch less in the
    combined signal), 3 unreliable (valuation ignored in the combined signal).
    """
    iss = []
    add = lambda sev, code, title, plain, lean="": iss.append(dict(sev=sev, code=code, title=title, plain=plain, lean=lean))
    if not gates["C"]["ok"]:
        add(3, "C", "Engine check failed", "The simulation did not reproduce its own closed-form mean -- treat the "
            "numbers as broken for this stock.")
    if not out["sanity_1f"]["ok"]:
        add(3, "1F", "Earnings jump looks like a data issue", "Year-1 base EPS is 2.5x or more of trailing EPS -- "
            "usually a one-off, a restatement or a units problem in the source data.", "optimistic")
    a_low = Q["P10"] > 1.10 * bl["bear"]
    a_high = Q["P90"] < 0.90 * bl["bull"]
    if a_low:
        add(2, "A", "Downside under-represented", "The low end of the fair-value range sits above the bear scenario, "
            "so the bear case barely shows up -- 'cheap' may be flattered.", "optimistic")
    if a_high:
        add(2, "A", "Upside under-represented", "The top of the range sits below the bull scenario -- 'expensive' "
            "may be overstated.", "pessimistic")
    if not gates["B"]["ok"]:
        if mass < target:
            add(1, "B", "Upside case much bigger than downside", "The bull scenario sits far further above base than "
                "the bear sits below, so part of the value comes from the bull case.", "optimistic")
        else:
            add(1, "B", "Downside case dominates", "The bear scenario sits far further below base than the bull sits "
                "above, so the verdict leans on the bear case.", "pessimistic")
    if not gates["E"]["ok"]:
        add(2, "E", "Range narrower than the stock's own price swings", "Even after widening to the floor, the "
            "fair-value range is narrower than the stock moves in a year -- percentiles near the ends overstate "
            "how cheap or expensive it is.")
    if out.get("e_widened", 1.0) > 1.0:
        add(0, "E+", f"Range widened x{out['e_widened']:.2f} to the price-volatility floor",
            "Mechanical floor rule; base case and probabilities unchanged.")
    if indeterminate:
        add(2, "MULT", "Price above anything its history supports", "Today's multiple is above any level in the "
            "stock's own 5-year record and the durable re-rating test failed (share gains + ROCE >= 20% + cash "
            "conversion). If the re-rating is real, this verdict is too harsh.", "pessimistic")
    elif gates["D"]["triggered"] and out["multiple"].get("precedence"):
        add(0, "D", "Bull case capped just below today's multiple", "The stock trades at or above its fair multiple "
            "and the durable re-rating test failed, so the model does not pay more than today's multiple -- value "
            "has to come from earnings growth.")
    elif gates["D"]["triggered"]:
        add(1, "D", "De-rating call", "Today's multiple is above even the bull case: the verdict needs the multiple "
            "to fall. If the market keeps paying it, the verdict is too harsh.", "pessimistic")
    if out.get("knob_dependent"):
        add(1, "P4", "Close to a verdict boundary", "A reasonable change in one assumption (hurdle, correlation or "
            "anchor weight) moves it into the next verdict.")
    if ident == "COMMODITY":
        add(1, "CYC", "Commodity: indicative only", "Valued on mid-cycle (normalized) earnings; v4.1 keeps this "
            "track indicative until it is validated.")
    if out.get("f2_deviation"):
        add(1, "F2", "Recent losses: bear case floored", "The worst recent run-rate was a loss; the bear branch stops "
            "at a small profit / book value because EPS x PE is meaningless below zero.", "optimistic")
    m_ = out["multiple"]
    if np.isfinite(m_.get("spot", np.nan)) and m_["spot"] < 0.8 * m_["base"]:
        add(1, "REV", "Cheap mainly on multiple recovery", f"Today's {m_['kind']} of {m_['spot']:.1f}x is well below "
            f"the {m_['base']:.1f}x fair multiple (own 5-year / peer anchor). The verdict assumes it recovers -- if the "
            "5-year norm came from a re-rated era that is over, it won't.", "optimistic")
    if out.get("data_fallback"):
        add(2, "DATA", "Scenario data missing", "Bear/bull margins were not in the source data; scenarios fell back "
            "to +/-30% of base.")
    if conf == "LOW":
        add(1, "DATA", "Limited history", "Fewer than 5 annual years, 3 years of multiple history, 5 peers or "
            "8 quarters -- bands are wider and less certain.")
    sev = max([i["sev"] for i in iss], default=0)
    leans = [i["lean"] for i in iss if i["sev"] >= 1 and i["lean"]]
    opt, pes = leans.count("optimistic"), leans.count("pessimistic")
    lean = ("may be too optimistic" if opt > pes else "may be too harsh" if pes > opt else
            "mixed" if opt else "")
    main = sorted([i for i in iss if i["sev"] >= 1], key=lambda i: -i["sev"])
    icon, grade = QUALITY[sev]
    return dict(sev=sev, icon=icon, grade=grade, lean=lean, issues=iss,
                note=(main[0]["title"] + (f" (+{len(main) - 1} more)" if len(main) > 1 else "")) if main else "")


def signal_bucket(vb: str, sev: int) -> str:
    """Valuation bucket as used by the combined signal: weak -> one notch toward HOLD, unreliable -> ignored."""
    if sev >= 3 or vb not in VAL_BUCKETS:
        return "NOT VALUED"
    if sev == 2:
        return {"BUY": "ACCUMULATE", "ACCUMULATE": "HOLD", "HOLD": "HOLD", "REDUCE": "HOLD", "SELL": "REDUCE"}[vb]
    return vb


def summary_row(res: dict) -> dict:
    """Flat row for the snapshot table."""
    row = dict(ticker=res.get("ticker"), status=res.get("status"), reason=res.get("reason", ""),
               identity=res.get("identity", ""), label=res.get("label", ""))
    if res.get("status") != "OK":
        return row
    d, m = res["dist"], res["multiple"]
    row.update(rating=res["rating"], provisional=res.get("provisional", ""), cmp_pct=round(d["cmp_pct"], 1), price=res["snapshot"]["price"],
               p10=d["P10"], p50=d["P50"], p90=d["P90"], confidence=res["confidence"],
               gates_ok=res["gates_ok"], derating=res["gates"]["D"]["triggered"],
               knob_dependent=res.get("knob_dependent", False), cagr_bear=res["cagr"]["bear"],
               cagr_base=res["cagr"]["base"], cagr_bull=res["cagr"]["bull"], hurdle=res["hurdle"],
               spot_mult=m["spot"], base_mult=m["base"], mult_kind=m["kind"],
               q_sev=res["quality"]["sev"], q_icon=res["quality"]["icon"], q_grade=res["quality"]["grade"],
               q_note=res["quality"]["note"], q_lean=res["quality"]["lean"],
               q_issues=json.dumps([[i["sev"], i["title"], i["lean"]] for i in res["quality"]["issues"]]),
               e_widened=round(res.get("e_widened", 1.0), 2),
               grid=json.dumps([round(x, 2) for x in res["grid"]]))
    return row


def rating_from_grid(grid, price, gates_ok, confidence, identity, label_if_unrated=None):
    """Re-percentile a stored distribution at today's price (the table uses this)."""
    g = np.asarray(grid, float)
    pct = float(np.interp(price, g, np.arange(1, 100), left=0.5, right=99.5))
    if label_if_unrated:
        return pct, label_if_unrated
    v = _verdict(pct, confidence, gates_ok)
    return pct, v + (" · INDICATIVE" if identity == "COMMODITY" else "")


# ----------------------------------------------------------------------------
# five-way valuation verdict and the combined technical x valuation signal
# ----------------------------------------------------------------------------
VAL_BUCKETS = ["BUY", "ACCUMULATE", "HOLD", "REDUCE", "SELL"]


def val_bucket(pct: float) -> str:
    """v4.1 ladder on the CMP percentile, always one of five (gate status is reported separately)."""
    if not np.isfinite(pct):
        return ""
    return "BUY" if pct < 20 else "ACCUMULATE" if pct < 40 else "HOLD" if pct < 70 else "REDUCE" if pct < 88 else "SELL"


def combined_signal(tech_buy: bool, below_both: bool, vb: str) -> tuple[str, str]:
    """
    BUY / HOLD / SELL from technicals x valuation.
      technical +  : forensics action BUY NOW (Gate 1 trend + GO volume anchor, no near overhang)
      technical -  : close below both the 50- and 200-day averages (backtested negative)
      cheap = BUY/ACCUMULATE, fair = HOLD, expensive = REDUCE/SELL
    The technical side carries the 10-year evidence, so an expensive valuation can downgrade a
    technical BUY to HOLD but never flip it to SELL, and a cheap stock in a broken trend is HOLD,
    not BUY (no falling knives). Without a valuation the signal is technical-only.
    """
    tech = "+" if tech_buy else "-" if below_both else "0"
    if vb not in VAL_BUCKETS:
        return {"+": "BUY", "-": "SELL", "0": "HOLD"}[tech], "technicals only (no valuation)"
    cheap, fair = vb in ("BUY", "ACCUMULATE"), vb == "HOLD"
    if tech == "+":
        sig = "BUY" if (cheap or fair) else "HOLD"
    elif tech == "-":
        sig = "HOLD" if cheap else "SELL"
    else:
        sig = "SELL" if vb == "SELL" else "HOLD"
    why = {"+": "technical BUY NOW", "0": "technical neutral", "-": "below 50 & 200 DMA"}[tech]
    return sig, f"{why} x valuation {vb}"

