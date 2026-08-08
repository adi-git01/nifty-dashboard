"""
Phase-0 auto-pack for the Monte Carlo valuation framework (v4.1).
=================================================================

Two jobs, both cheap and both offline:

1. **Scope gate.** SCOPE and PHASE 0-IDENTITY are mechanically decidable from
   the cache — positive TTM EPS, sector exclusions, lender vs operating company.
   Deciding this in Python costs nothing and stops an out-of-scope name from
   ever consuming an LLM call. (Measured on the live universe: 1020 of 1174 are
   in scope; 107 fail on non-positive TTM EPS, 38 are NAV-driven, 12 insurance.)

2. **Fill what we have, and name what we don't.** The pack carries every Phase-0
   field the cache can supply *with its provenance*, and — the part that matters
   — an explicit `gaps` list of the fields it cannot supply, each tagged with the
   phase that needs it and why. That list is the research brief: it is what a
   web-capable model is asked to fetch, rather than letting it free-range over
   the whole company.

P6 says anything unverifiable is flagged, not assumed. Until now the framework
stated that but had nowhere to put the flag; `provenance` and `gaps` are that
slot. Confidence is capped mechanically by how many PHASE 1C rows are unscorable
from available data, so a thin dossier cannot silently produce a confident verdict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# --- identity routing -------------------------------------------------------
BFSI_SUB_INDUSTRIES = {"Banks", "Finance", "Capital Markets",
                       "Financial Technology (Fintech)"}
# Out of scope per SCOPE: NAV-driven and insurance.
NAV_DRIVEN = {"Realty"}
INSURANCE = {"Insurance"}
# >50% of EBITDA from a spot commodity — taxonomy is a proxy, the analyst confirms.
COMMODITY_SUB_INDUSTRIES = {
    "Ferrous Metals", "Non - Ferrous Metals", "Diversified Metals",
    "Minerals & Mining", "Oil", "Petroleum Products", "Consumable Fuels",
    "Paper, Forest & Jute Products",
}

# --- what the framework needs that yfinance does not carry ------------------
# (field, phase that needs it, why it matters, where to look)
RESEARCH_FIELDS = [
    ("pe_pb_5y_history", "2C",
     "half the anchor: own 5Y median multiple from positive-earnings periods",
     "Screener.in / Trendlyne multi-year ratio table"),
    ("promoter_holding_pledge", "-1 / 1C",
     "promoter-trend row and the chronic-pledge Tier 1 ledger test",
     "BSE/NSE shareholding pattern, latest quarter"),
    ("fii_dii_4q_trend", "0 / 1C",
     "the FII/DII row; mixed reads score 0 per P2",
     "quarterly shareholding pattern, last 4 quarters"),
    ("fcf_cfo_pat", "0 / 1C",
     "FCF-quality row; negative FCF from dated growth capex scores 0, not -1",
     "annual report cash-flow statement"),
    ("roce", "0 / 2C",
     "Phase-0 snapshot and the ROCE-justified multiple ceiling",
     "Screener.in ratios"),
    ("quarterly_pat_sequence", "1C",
     "the literal met/beat-3-of-4 execution test",
     "last 4-8 quarterly results"),
    ("order_book_visibility", "0 / 1A",
     "visibility metric and the revenue driver bands",
     "management commentary / investor presentation"),
    ("guidance_record", "1C",
     "guidance-record row: stated targets vs delivered",
     "earnings calls, last 4 quarters"),
    ("event_risks", "-1 / 1C",
     "named discrete risks with base rates drive the +/-20pp probability adjustment "
     "AND must appear in the bear earnings path (Gate F1)",
     "regulatory filings, exchange disclosures, news"),
    ("cycle_position", "0 / 1B",
     "TROUGH/EARLY-UP/MID/LATE/DOWN with evidence; sets bull shape for cyclicals",
     "spreads vs 10Y, utilisation, inventory cover"),
]


@dataclass
class ValuationPack:
    ticker: str
    name: str = ""
    in_scope: bool = True
    out_of_scope_reason: str = ""
    identity: str = ""
    identity_trigger: str = ""
    indicative_only: bool = False
    snapshot: Dict = field(default_factory=dict)
    provenance: Dict[str, str] = field(default_factory=dict)
    peers: Dict = field(default_factory=dict)
    gaps: List[Dict] = field(default_factory=list)
    confidence_cap: str = "HIGH"
    notes: List[str] = field(default_factory=list)

    def research_brief(self) -> str:
        """The prompt fragment handed to a web-capable model — only the gaps."""
        if not self.gaps:
            return ""
        lines = [f"Fetch the following for {self.name or self.ticker} "
                 f"({self.ticker}). Cite a source and an as-of date for each. "
                 f"If a figure cannot be verified, say so — do not estimate.\n"]
        for g in self.gaps:
            lines.append(f"- **{g['field']}** (needed by Phase {g['phase']}): "
                         f"{g['why']}. Look in: {g['source_hint']}.")
        return "\n".join(lines)


def _num(v):
    try:
        f = float(v)
        return None if (f != f or f in (float("inf"), float("-inf"))) else f
    except (TypeError, ValueError):
        return None


def choose_identity(sub_industry: str, pb: Optional[float]) -> tuple:
    si = (sub_industry or "").strip()
    if si in BFSI_SUB_INDUSTRIES:
        return "BFSI", f"'{si}' is a lender — earnings are a spread on a levered book"
    if si in COMMODITY_SUB_INDUSTRIES:
        return "COMMODITY", (f"'{si}' is a price-taker on a spot commodity — "
                             f"routes to the interim normalized-earnings track")
    return "INDUSTRIAL", f"'{si}' is an operating company — EPS x PE default"


def build_pack(ticker: str, market_df: pd.DataFrame,
               universe_df: Optional[pd.DataFrame] = None) -> ValuationPack:
    """
    market_df: the daily master snapshot (price, pe, pb, roe, 52w H/L, ...).
    universe_df: nifty1000_list.csv, for the Sub_Industry peer set.
    """
    row = market_df[market_df["ticker"] == ticker]
    if row.empty:
        return ValuationPack(ticker=ticker, in_scope=False,
                             out_of_scope_reason="not in the loaded universe")
    r = row.iloc[0]

    sub = ""
    if universe_df is not None:
        u = universe_df[universe_df["Ticker"] == ticker]
        if not u.empty:
            sub = str(u.iloc[0].get("Sub_Industry", ""))
    if not sub:
        sub = str(r.get("sector_granular") or r.get("sector") or "")

    pack = ValuationPack(ticker=ticker, name=str(r.get("name") or ticker))

    # ---- SCOPE ----
    pe, pb = _num(r.get("pe")), _num(r.get("pb"))
    price = _num(r.get("price"))
    if sub in NAV_DRIVEN:
        pack.in_scope = False
        pack.out_of_scope_reason = (f"'{sub}' is NAV-driven — this is a NAV problem, "
                                    f"not a multiple problem")
    elif sub in INSURANCE:
        pack.in_scope = False
        pack.out_of_scope_reason = "insurance is explicitly out of scope"
    elif pe is None or pe <= 0:
        pack.in_scope = False
        pack.out_of_scope_reason = ("non-positive or absent TTM EPS — EPS x PE is "
                                    "undefined; equity may be an option on survival")
    if not pack.in_scope:
        return pack

    # ---- IDENTITY ----
    pack.identity, pack.identity_trigger = choose_identity(sub, pb)
    pack.indicative_only = pack.identity == "COMMODITY"
    if pack.indicative_only:
        pack.notes.append("COMMODITY track is interim (normalized-earnings); "
                          "verdict must print INDICATIVE ONLY")

    # ---- PHASE 0 SNAPSHOT (what the cache genuinely has) ----
    eps = price / pe if (price and pe) else None
    hi, lo = _num(r.get("fiftyTwoWeekHigh")), _num(r.get("fiftyTwoWeekLow"))
    snap = {
        "cmp": price,
        "ttm_eps_derived": round(eps, 2) if eps else None,
        "pe": pe, "pb": pb,
        "bvps_derived": round(price / pb, 2) if (price and pb) else None,
        "roe": _num(r.get("roe")), "roa": _num(r.get("roa")),
        "profit_margin": _num(r.get("profitMargins")),
        "gross_margin": _num(r.get("grossMargins")),
        "operating_margin": _num(r.get("operatingMargins")),
        "debt_to_equity": _num(r.get("debtToEquity")),
        "revenue_growth": _num(r.get("revenueGrowth")),
        "earnings_growth": _num(r.get("earningsGrowth")),
        "earnings_growth_qoq": _num(r.get("earningsQuarterlyGrowth")),
        "market_cap": _num(r.get("marketCap")),
        "beta": _num(r.get("beta")),
        "wk52_high": hi, "wk52_low": lo,
        "sub_industry": sub,
        "fund_as_of": str(r.get("fund_last_updated") or ""),
    }
    if hi and lo and (hi + lo):
        snap["wk52_range_over_mid"] = round((hi - lo) / ((hi + lo) / 2), 4)
    pack.snapshot = snap
    for k, v in snap.items():
        pack.provenance[k] = ("derived from cache" if k.endswith("_derived")
                              else "yfinance cache")

    # ---- PEER SET (same business model — the 58-bucket taxonomy) ----
    if universe_df is not None and sub:
        members = universe_df[universe_df["Sub_Industry"] == sub]["Ticker"].tolist()
        peer = market_df[market_df["ticker"].isin(members)
                         & (market_df["ticker"] != ticker)].copy()
        for c in ("pe", "pb", "roe"):
            peer[c] = pd.to_numeric(peer.get(c), errors="coerce")
        peer = peer[peer["pe"] > 0] if pack.identity != "BFSI" else peer[peer["pb"] > 0]
        pack.peers = {
            "sub_industry": sub,
            "n_members": int(len(peer)),
            "median_pe": round(float(peer["pe"].median()), 2) if len(peer) else None,
            "median_pb": round(float(peer["pb"].median()), 2) if len(peer) else None,
            "median_roe": round(float(peer["roe"].median()), 4) if len(peer) else None,
            "names": peer.nlargest(min(8, len(peer)), "marketCap")["ticker"].tolist()
            if "marketCap" in peer.columns and len(peer) else peer["ticker"].tolist()[:8],
        }
        if len(peer) < 3:
            pack.notes.append(f"peer set has only {len(peer)} member(s) — the peer "
                              f"anchor is weak; lean on own history and say so")

    # ---- GAPS: the research brief for the fetch step ----
    for fname, phase, why, hint in RESEARCH_FIELDS:
        pack.gaps.append({"field": fname, "phase": phase, "why": why,
                          "source_hint": hint, "status": "MISSING"})

    # ---- CONFIDENCE CAP (mechanical, not a judgment) ----
    # PHASE 1C has ~11 standard rows; five of them depend on fields the cache
    # cannot supply. A dossier that never fills them must not print HIGH.
    unscorable = sum(1 for g in pack.gaps
                     if g["field"] in {"promoter_holding_pledge", "fii_dii_4q_trend",
                                       "fcf_cfo_pat", "quarterly_pat_sequence",
                                       "guidance_record"})
    pack.confidence_cap = "LOW" if unscorable >= 4 else "MEDIUM" if unscorable >= 2 else "HIGH"
    pack.notes.append(f"{unscorable} of 5 research-dependent 1C rows unfilled "
                      f"-> confidence capped at {pack.confidence_cap} until researched")
    return pack


def scope_report(market_df: pd.DataFrame, universe_df: pd.DataFrame) -> Dict:
    """Universe-wide scope counts — cheap, and it sizes the LLM bill."""
    m = universe_df.merge(market_df, left_on="Ticker", right_on="ticker", how="left")
    pe = pd.to_numeric(m.get("pe"), errors="coerce")
    nav = m["Sub_Industry"].isin(NAV_DRIVEN)
    ins = m["Sub_Industry"].isin(INSURANCE)
    bad = pe.isna() | (pe <= 0)
    ok = ~(nav | ins | bad)
    return {
        "universe": int(len(m)),
        "out_non_positive_eps": int(bad.sum()),
        "out_nav_driven": int(nav.sum()),
        "out_insurance": int(ins.sum()),
        "in_scope": int(ok.sum()),
        "bfsi": int((ok & m["Sub_Industry"].isin(BFSI_SUB_INDUSTRIES)).sum()),
        "commodity": int((ok & m["Sub_Industry"].isin(COMMODITY_SUB_INDUSTRIES)).sum()),
    }
