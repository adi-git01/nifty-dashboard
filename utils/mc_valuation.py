"""
Monte Carlo intrinsic valuation — engine + hard gates (Master Prompt v4.1 §2D, §3).
==================================================================================

Why this is Python and not part of the LLM's job
------------------------------------------------
The judgment phases (−1, 0-IDENTITY, 1A–1E, 2C) are reasoning and belong to the
model. Everything here is arithmetic, and arithmetic that *checks* the reasoning:
Gate C compares the simulation mean to a closed-form mixture mean, Gate B checks
probability mass, Gate A checks tail reach. A gate the author evaluates about
their own run is a promise; a gate an independent implementation evaluates is a
test. (The first live run audited here reported Gate C as PASS at a 1.1%
deviation against a stated 1% threshold — precisely the error class this closes.)

The CMP-independence that makes quarterly caching valid
-------------------------------------------------------
Read §2D closely: the simulated value distribution is built from levers,
multiples, probabilities, rho, weights and the hurdle. CMP appears nowhere in
it. Market price enters only *after* the distribution exists — to locate a
percentile, to test Gate D, and to compute the implied CAGR.

So the expensive half is a per-quarter artifact. Simulate once, store the CDF,
and a price move becomes an interpolation rather than a re-run. `ValuationCard`
is that artifact; `evaluate_at_price()` is the cheap daily half.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

ENGINE_VERSION = "4.1"
REGIMES = ("bear", "base", "bull")

# Verdict ladder (v4.1 VERDICT & OUTPUT).
LADDER = [
    (10.0, "DEEP ADD"),
    (25.0, "ACCUMULATE"),
    (75.0, "FAIR VALUE"),
    (90.0, "TRIM"),
    (100.1, "EXIT / AVOID"),
]


def verdict_for(percentile: float) -> str:
    for edge, label in LADDER:
        if percentile < edge:
            return label
    return LADDER[-1][1]


# ---------------------------------------------------------------------------
# triangular bands
# ---------------------------------------------------------------------------
def _band(regime: str, bear: float, base: float, bull: float,
          floor: Optional[float] = None) -> tuple:
    """(a, mode, c) per §2D, with the minimum-width pad and optional hard floor."""
    if regime == "bear":
        a, m, c = bear - 0.5 * (base - bear), bear, base
        a = max(a, 0.25 * bear, 0.0)
    elif regime == "base":
        a, m, c = bear, base, bull
    else:
        a, m, c = base, bull, bull + 0.5 * (bull - base)

    if floor is not None:
        a = max(a, floor)
        m = max(m, floor)
        c = max(c, m)

    # zero-width bands crash the inverse CDF; pad symmetrically to 2% of mode
    span = c - a
    need = 0.02 * abs(m)
    if span < need:
        pad = (need - span) / 2.0
        a, c = a - pad, c + pad
        if floor is not None:
            a = max(a, floor)
    return a, m, c


def _tri_mean(band: Sequence[float]) -> float:
    a, m, c = band
    return (a + m + c) / 3.0


def _tri_ppf(u: np.ndarray, band: Sequence[float]) -> np.ndarray:
    """Inverse CDF of Triangular(a, mode, c)."""
    a, m, c = band
    if c <= a:
        return np.full_like(u, m, dtype=float)
    fm = (m - a) / (c - a)
    out = np.empty_like(u, dtype=float)
    lo = u < fm
    out[lo] = a + np.sqrt(np.clip(u[lo] * (c - a) * (m - a), 0, None))
    hi = ~lo
    out[hi] = c - np.sqrt(np.clip((1 - u[hi]) * (c - a) * (c - m), 0, None))
    return out


# ---------------------------------------------------------------------------
@dataclass
class ValuationCard:
    """
    Per-quarter artifact. Everything here is CMP-independent; price is applied
    later by evaluate_at_price(). Serialise this to JSON and a daily refresh
    costs one interpolation instead of one LLM call plus 10,000 trials.
    """
    ticker: str
    identity: str                      # INDUSTRIAL | BFSI | MIXED | COMMODITY
    as_of: str
    lever_name: str                    # "BVPS" or "EPS"
    levers: Dict[str, List[float]]     # regime -> [Y1, Y2, Y3]
    multiples: Dict[str, float]        # regime -> exit multiple
    probs: Dict[str, float]
    rho: float
    hurdle: float
    weights: List[float]
    lever_floor: Optional[float] = None
    mult_floor: Optional[float] = None      # BFSI book floor on the bear band's reach
    indicative_only: bool = False           # commodity track
    seed: int = 404
    n_trials: int = 10_000
    regime_corr: bool = False               # see note in simulate()

    # filled by simulate()
    percentiles: Dict[str, float] = field(default_factory=dict)
    cdf_x: List[float] = field(default_factory=list)
    cdf_p: List[float] = field(default_factory=list)
    mean: float = 0.0
    sd: float = 0.0
    blends: Dict[str, float] = field(default_factory=dict)
    analytic_mean: float = 0.0
    regime_share: Dict[str, float] = field(default_factory=dict)
    regime_mean: Dict[str, float] = field(default_factory=dict)
    y3_undiscounted: Dict[str, float] = field(default_factory=dict)

    def lever_bands(self, regime: str) -> List[tuple]:
        return [_band(regime, self.levers["bear"][h], self.levers["base"][h],
                      self.levers["bull"][h], floor=self.lever_floor)
                for h in range(len(self.weights))]

    def mult_band(self, regime: str) -> tuple:
        return _band(regime, self.multiples["bear"], self.multiples["base"],
                     self.multiples["bull"], floor=self.mult_floor)


def deterministic_blend(card: ValuationCard, regime: str) -> float:
    """Σ w_h · lever_h · mult / (1+r)^h at the modes — the 'blend' Gate A uses."""
    m = card.multiples[regime]
    return sum(w * card.levers[regime][h] * m / (1 + card.hurdle) ** (h + 1)
               for h, w in enumerate(card.weights))


def analytic_mixture_mean(card: ValuationCard) -> float:
    """Σ_s p_s Σ_h w_h · E[band_s,h] · E[mult_s] / (1+r)^h  (Gate C reference)."""
    total = 0.0
    for s in REGIMES:
        em = _tri_mean(card.mult_band(s))
        bands = card.lever_bands(s)
        inner = sum(w * _tri_mean(bands[h]) * em / (1 + card.hurdle) ** (h + 1)
                    for h, w in enumerate(card.weights))
        total += card.probs[s] * inner
    return total


def simulate(card: ValuationCard) -> ValuationCard:
    """
    §2D engine. One regime draw per trial (persistent), comonotonic lever path
    within a trial, multiple correlated to the lever draw at rho.

    Spec note: §2D reads "z_r, z_e ~ correlated N(0,1) with corr rho; z_p =
    rho·z_e + sqrt(1−rho²)·ε". Taken literally that correlates the regime draw
    with the earnings draw *as well as* the multiple draw, which would double
    count the regime effect — good earnings draws would preferentially land in
    bull regimes on top of the bull bands already being higher. The default here
    treats z_r as independent and applies rho only between lever and multiple,
    which is what the explicit z_p formula describes. `regime_corr=True`
    reproduces the literal reading for comparison.
    """
    rng = np.random.default_rng(card.seed)
    n = card.n_trials
    H = len(card.weights)

    z_e = rng.standard_normal(n)
    z_r = (card.rho * z_e + math.sqrt(1 - card.rho ** 2) * rng.standard_normal(n)
           if card.regime_corr else rng.standard_normal(n))
    z_p = card.rho * z_e + math.sqrt(1 - card.rho ** 2) * rng.standard_normal(n)

    u_r = _norm_cdf(z_r)
    u_e = _norm_cdf(z_e)
    u_p = _norm_cdf(z_p)

    edges = np.array([card.probs["bear"],
                      card.probs["bear"] + card.probs["base"]])
    reg = np.digitize(u_r, edges)          # 0 bear, 1 base, 2 bull

    V = np.zeros(n)
    disc = np.array([(1 + card.hurdle) ** (h + 1) for h in range(H)])
    w = np.asarray(card.weights, dtype=float)

    for idx, s in enumerate(REGIMES):
        mask = reg == idx
        if not mask.any():
            continue
        mult = _tri_ppf(u_p[mask], card.mult_band(s))
        bands = card.lever_bands(s)
        acc = np.zeros(mask.sum())
        for h in range(H):
            lev = _tri_ppf(u_e[mask], bands[h])
            acc += w[h] * lev * mult / disc[h]
        V[mask] = acc

    card.percentiles = {f"P{p}": float(np.percentile(V, p))
                        for p in (5, 10, 25, 50, 75, 90, 95)}
    grid = np.linspace(0.1, 99.9, 999)
    card.cdf_x = [float(x) for x in np.percentile(V, grid)]
    card.cdf_p = [float(p) for p in grid]
    card.mean, card.sd = float(V.mean()), float(V.std(ddof=1))
    card.blends = {s: deterministic_blend(card, s) for s in REGIMES}
    card.analytic_mean = analytic_mixture_mean(card)
    card.regime_share = {s: float((reg == i).mean()) for i, s in enumerate(REGIMES)}
    card.regime_mean = {s: (float(V[reg == i].mean()) if (reg == i).any() else 0.0)
                        for i, s in enumerate(REGIMES)}
    card.y3_undiscounted = {s: card.levers[s][-1] * card.multiples[s] for s in REGIMES}
    return card


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# the cheap daily half
# ---------------------------------------------------------------------------
def percentile_at(card: ValuationCard, cmp_price: float) -> float:
    """Where CMP sits in the stored distribution. Interpolation, not simulation."""
    x, p = card.cdf_x, card.cdf_p
    if cmp_price <= x[0]:
        return 0.0
    if cmp_price >= x[-1]:
        return 100.0
    return float(np.interp(cmp_price, x, p))


def implied_cagr(card: ValuationCard, cmp_price: float,
                 yield_pct: float = 0.0) -> Dict[str, float]:
    """(undiscounted Y3 value ÷ CMP)^(1/3) − 1, plus yield — per §2D."""
    out = {}
    for s in REGIMES:
        out[s] = (card.y3_undiscounted[s] / cmp_price) ** (1 / 3) - 1 + yield_pct / 100.0
    return out


def evaluate_at_price(card: ValuationCard, cmp_price: float,
                      book_or_eps: Optional[float] = None,
                      wk52_high: Optional[float] = None,
                      wk52_low: Optional[float] = None,
                      yield_pct: float = 0.0) -> Dict:
    """
    Everything that legitimately moves with price, and nothing that doesn't.
    Cheap enough to run for the whole universe on every dashboard refresh.
    """
    pct = percentile_at(card, cmp_price)
    res = {
        "ticker": card.ticker,
        "cmp": cmp_price,
        "percentile": round(pct, 1),
        "verdict": verdict_for(pct),
        "indicative_only": card.indicative_only,
        "as_of": card.as_of,
        "implied_cagr": implied_cagr(card, cmp_price, yield_pct),
        "ladder": {
            "deep_add_below": card.percentiles["P10"],
            "accumulate_to": card.percentiles["P25"],
            "fair_to": card.percentiles["P75"],
            "trim_to": card.percentiles["P90"],
        },
    }
    # Gate D re-tests on every price move: a rally can push CMP÷book (or ÷EPS)
    # through the bull multiple mid-quarter and turn the call into a
    # multiple-de-rating one without any fundamental changing.
    if book_or_eps:
        spot_mult = cmp_price / book_or_eps
        bull = card.multiples["bull"]
        res["gate_d"] = {
            "spot_multiple": round(spot_mult, 3),
            "bull_multiple": bull,
            "triggered": spot_mult > bull,
            "near_trigger": bull * 0.90 <= spot_mult <= bull,
            "counterfactual_base_at_spot": deterministic_blend(card, "base")
            / card.multiples["base"] * spot_mult,
        }
    # Gate E's floor moves too: a new 52-week extreme rewidens the requirement.
    if wk52_high and wk52_low:
        mid = (wk52_high + wk52_low) / 2.0
        req = 0.8 * (wk52_high - wk52_low) / mid
        got = (card.percentiles["P90"] - card.percentiles["P10"]) / card.percentiles["P50"]
        res["gate_e"] = {"dispersion": round(got, 3),
                         "required": round(req, 3), "pass": got >= req}
    return res


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def run_gates(card: ValuationCard, cmp_price: float,
              book_or_eps: Optional[float] = None,
              wk52_high: Optional[float] = None,
              wk52_low: Optional[float] = None) -> Dict[str, Dict]:
    g: Dict[str, Dict] = {}
    P = card.percentiles

    spread = (card.blends["bull"] - card.blends["base"]) / card.blends["base"]
    g["A"] = {
        "p90_ok": P["P90"] >= 0.90 * card.blends["bull"],
        "p10_ok": P["P10"] <= 1.10 * card.blends["bear"],
        "bull_base_spread": round(spread * 100, 1),
        "discriminating": spread >= 0.15,
    }
    g["A"]["pass"] = g["A"]["p90_ok"] and g["A"]["p10_ok"]

    below = float(np.interp(card.blends["base"], card.cdf_x, card.cdf_p))
    target = 100 * (card.probs["bear"] + card.probs["base"] / 2)
    g["B"] = {"mass_below_base": round(below, 1), "target": round(target, 1),
              "shift_pp": round(below - target, 1),
              "pass": abs(below - target) <= 10.0}

    dev = (card.mean - card.analytic_mean) / card.analytic_mean
    g["C"] = {"sim_mean": round(card.mean, 2),
              "analytic_mean": round(card.analytic_mean, 2),
              "deviation_pct": round(dev * 100, 2),
              "pass": abs(dev) <= 0.01,
              "rho_residual_allowance": abs(dev) <= 0.005}
    mode_blend = sum(card.probs[s] * card.blends[s] for s in REGIMES)
    g["C"]["asymmetry_drift_pct"] = round((card.analytic_mean / mode_blend - 1) * 100, 2)

    if book_or_eps:
        spot = cmp_price / book_or_eps
        g["D"] = {"spot_multiple": round(spot, 3), "bull_multiple": card.multiples["bull"],
                  "triggered": spot > card.multiples["bull"],
                  "near_trigger": card.multiples["bull"] * 0.9 <= spot <= card.multiples["bull"]}
        g["D"]["pass"] = True  # disclosure gate, never a hard stop

    if wk52_high and wk52_low:
        mid = (wk52_high + wk52_low) / 2.0
        req = 0.8 * (wk52_high - wk52_low) / mid
        got = (P["P90"] - P["P10"]) / P["P50"]
        g["E"] = {"dispersion": round(got, 3), "required": round(req, 3),
                  "pass": got >= req}
    return g
