"""
Guarded Dynamic Liquid Universe (>= Rs 2 Cr/day NSE Equities) & Non-Noisy UI Screen Filters.
=============================================================================================

Expands the dashboard's active universe beyond the static quarterly Nifty 1000 list to capture
every genuinely tradable NSE equity (emerging mid/small-caps, recent institutional IPOs,
demergers, and symbol renames) WITHOUT polluting scanners with ETFs, penny stocks, operator
one-day spikes, trade-to-trade (BE) names, or GSM/ASM surveillance traps.

Guardrails enforced on dynamic additions (`data/liquid_universe.csv`):
  1. Official NSE equities only (`EQUITY_L.csv`, series == "EQ") — excludes ETFs, REITs, InvITs, BE/BZ.
  2. 60-session MEDIAN daily turnover (not mean) >= Rs 2.0 Cr/day AND 20-session median >= Rs 1.5 Cr/day,
     with hysteresis (retains admitted names until 60D median drops below Rs 1.0 Cr/day).
  3. Market Cap >= Rs 500 Cr whenever MCap is known from Screener / extended_universe / fundamentals cache.
     Brand-new IPOs / listings not yet scraped by Screener must clear >= Rs 5.0 Cr/day (60D median) and
     >= Rs 2.5 Cr/day (20D median) until Screener verifies MCap >= Rs 500 Cr.
  4. Share price floor >= Rs 20 to exclude penny/operator churn stocks.
  5. Excludes hard NSE surveillance (GSM and LT-ASM Stage 3/4) from new liquid admissions.

UI Screen Filter (`apply_screen_filter`):
  Separates full universe ingestion (used for positions, alerts, Stock Report, Expert Council, and
  industry board) from scanner/funnel discovery so users can switch Liquidity Tier
  (Tradable >= 2Cr | Core >= 5Cr | Institutional >= 20Cr | Nifty 500 | All Loaded) and hide
  ASM/GSM & BE names cleanly with zero recompute overhead.

Run standalone: python -m utils.liquid_universe [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from utils.atomic_io import atomic_to_csv

LIQUID_CSV = "data/liquid_universe.csv"
CORE_CSV = "data/nifty1000_list.csv"
NIFTY500_CSV = "data/nifty500_list.csv"
EQ_LIST_CSV = "EQUITY_L.csv"
SURV_CSV = "data/nse/surveillance.csv"

# --- Ingestion guardrails ---------------------------------------------------
ENTER_MED_60D_CR = 2.0
ENTER_MED_20D_CR = 1.5
KEEP_MED_60D_CR = 1.0          # hysteresis exit floor once admitted
UNVERIFIED_MED_60D_CR = 5.0    # stricter floor for brand-new unscraped listings (IPOs)
UNVERIFIED_MED_20D_CR = 2.5
MIN_MCAP_CR = 500.0
MIN_PRICE_RS = 20.0
MIN_SESSIONS_60D = 20
MIN_IPO_SESSIONS = 20           # aligned with fast_data_engine's 20-bar minimum so every admitted stock renders cleanly
MIN_ACTIVE_STORE_EQUITIES = 500  # safety gate: refuse overwrite if NSE store is degraded

COLS = [
    "Ticker", "Company_Name", "Sub_Industry",
    "med_turn_60d_cr", "med_turn_20d_cr", "mcap_cr", "last_close",
    "series", "surveillance", "status", "added_date", "as_of",
]

# --- UI Liquidity Tier options ----------------------------------------------
TIER_TRADABLE = "💧 Tradable (≥ ₹2 Cr)"
TIER_CORE = "🏦 Core (≥ ₹5 Cr)"
TIER_INST = "🏛️ Institutional (≥ ₹20 Cr)"
TIER_NIFTY500 = "📊 Nifty 500"
TIER_ALL = "🌐 All Loaded"

LIQUIDITY_TIERS: Tuple[str, ...] = (
    TIER_TRADABLE,
    TIER_CORE,
    TIER_INST,
    TIER_NIFTY500,
    TIER_ALL,
)

TIER_MIN_CR: Dict[str, float] = {
    TIER_TRADABLE: 2.0,
    TIER_CORE: 5.0,
    TIER_INST: 20.0,
}

_HARD_SURV_RE = re.compile(r"\b(GSM|STAGE\s*[34]|IV|III)\b", re.I)


def _norm_ticker(t: str) -> str:
    s = str(t or "").strip().upper()
    if not s:
        return ""
    return s if s.endswith((".NS", ".BO")) else f"{s}.NS"


def _load_equity_list(root: str = ".") -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return ({SYMBOL: company_name}, {SYMBOL: series}) from NSE EQUITY_L.csv."""
    path = os.path.join(root, EQ_LIST_CSV)
    if not os.path.exists(path):
        return {}, {}
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().upper() for c in df.columns]
        syms = df["SYMBOL"].astype(str).str.strip()
        names = df["NAME OF COMPANY"].astype(str).str.strip() if "NAME OF COMPANY" in df.columns else syms
        series = df["SERIES"].astype(str).str.strip().str.upper() if "SERIES" in df.columns else pd.Series("EQ", index=df.index)
        return dict(zip(syms, names)), dict(zip(syms, series))
    except Exception:
        return {}, {}


def _load_surveillance(root: str = ".") -> Dict[str, str]:
    """Return {SYMBOL: stage/flag string} from data/nse/surveillance.csv."""
    path = os.path.join(root, SURV_CSV)
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
        if df.empty or "symbol" not in df.columns:
            return {}
        out: Dict[str, List[str]] = {}
        for r in df.itertuples():
            sym = str(getattr(r, "symbol", "") or "").strip().upper()
            if not sym:
                continue
            parts = []
            for col in ("list", "stage", "kind", "flag"):
                v = getattr(r, col, None)
                if v is not None and pd.notna(v) and str(v).strip():
                    parts.append(str(v).strip())
            label = " ".join(parts) if parts else "SURVEILLANCE"
            out.setdefault(sym, []).append(label)
        return {k: "; ".join(dict.fromkeys(v)) for k, v in out.items()}
    except Exception:
        return {}


def _is_hard_surveillance(flag: str) -> bool:
    if not flag:
        return False
    return bool(_HARD_SURV_RE.search(str(flag)))


def _load_known_mcap_and_industry(root: str = ".") -> Tuple[Dict[str, float], Dict[str, str], Dict[str, str]]:
    """
    Gather known Market Cap (Rs Cr), curated Sub_Industry, and Company_Name from local
    Screener / extended / cache files (no network calls).
    Returns ({ticker_ns: mcap_cr}, {ticker_ns: sub_industry}, {ticker_ns: company_name}).
    """
    from utils import fund_paths

    mcap: Dict[str, float] = {}
    ind: Dict[str, str] = {}
    names: Dict[str, str] = {}

    # 1. Screener annual files (lower priority -> higher priority)
    for rel in (fund_paths.EXTRA_ANNUAL, fund_paths.EXT_ANNUAL, fund_paths.ANNUAL):
        p = fund_paths.resolve(rel, root=root)
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p, usecols=["ticker", "field", "period", "value"])
            sub = df[df["field"] == "mcap_cr"].sort_values("period").groupby("ticker")["value"].last()
            for t, v in sub.items():
                fv = pd.to_numeric(v, errors="coerce")
                if pd.notna(fv) and float(fv) > 0:
                    mcap[_norm_ticker(t)] = float(fv)
        except Exception:
            continue

    # 2. fundamentals_cache.csv (Yahoo/NSE quote marketCap in Rs -> Rs Cr)
    fc_path = os.path.join(root, "data/fundamentals_cache.csv")
    if os.path.exists(fc_path):
        try:
            fc = pd.read_csv(fc_path)
            for r in fc.itertuples():
                t = _norm_ticker(getattr(r, "ticker", ""))
                if not t:
                    continue
                mc = pd.to_numeric(getattr(r, "marketCap", np.nan), errors="coerce")
                if pd.notna(mc) and float(mc) > 0 and t not in mcap:
                    mcap[t] = float(mc) / 1e7
                sg = str(getattr(r, "sector_granular", "") or "").strip()
                if sg and sg not in ("Unknown", "nan", "None"):
                    ind.setdefault(t, sg)
                nm = str(getattr(r, "name", "") or "").strip()
                if nm and nm not in ("nan", "None"):
                    names.setdefault(t, nm)
        except Exception:
            pass

    # 3. industry_extra.csv
    ie_path = os.path.join(root, "data/industry_extra.csv")
    if os.path.exists(ie_path):
        try:
            ie = pd.read_csv(ie_path)
            if len(ie.columns) >= 2:
                for t_raw, sec_raw in ie.iloc[:, :2].itertuples(index=False):
                    t = _norm_ticker(t_raw)
                    s = str(sec_raw or "").strip()
                    if t and s and s not in ("Unknown", "nan", "None"):
                        ind[t] = " ".join(s.split())
        except Exception:
            pass

    # 4. extended_universe.csv (highest priority for non-core names — Screener scraped weekly)
    ext_path = os.path.join(root, "data/extended_universe.csv")
    if os.path.exists(ext_path):
        try:
            ext = pd.read_csv(ext_path)
            for r in ext.itertuples():
                t = _norm_ticker(getattr(r, "Ticker", ""))
                if not t:
                    continue
                mc = pd.to_numeric(getattr(r, "mcap_cr", np.nan), errors="coerce")
                if pd.notna(mc) and float(mc) > 0:
                    mcap[t] = float(mc)
                si = str(getattr(r, "Sub_Industry", "") or "").strip()
                if si and si not in ("Unknown", "nan", "None"):
                    ind[t] = " ".join(si.split())
                nm = str(getattr(r, "Company_Name", "") or "").strip()
                if nm and nm not in ("nan", "None"):
                    names[t] = nm
        except Exception:
            pass

    return mcap, ind, names


def compute_nse_liquidity_table(eq_df: Optional[pd.DataFrame] = None, root: str = ".") -> Tuple[pd.DataFrame, str]:
    """
    Compute 60-session and 20-session median daily turnover (Rs Cr) per NSE symbol from the local
    NSE price store (`utils/nse_prices.raw('stocks')`). Returns (table indexed by symbol, as_of str).
    """
    if eq_df is None:
        from utils import nse_prices as npx
        old_cwd = os.getcwd()
        try:
            if root not in ("", "."):
                os.chdir(root)
            eq_df = npx.raw("stocks")
        finally:
            if root not in ("", "."):
                os.chdir(old_cwd)

    if eq_df is None or eq_df.empty:
        return pd.DataFrame(), ""

    df = eq_df.copy()
    if "series" in df.columns:
        df = df[df["series"].isin(["EQ", "BE", "BZ"])]
    if df.empty:
        return pd.DataFrame(), ""

    dates = sorted(pd.to_datetime(df["date"]).dropna().unique())
    if not dates:
        return pd.DataFrame(), ""
    as_of = pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")
    last_60 = set(dates[-60:])
    last_20 = set(dates[-20:])

    dcol = pd.to_datetime(df["date"])
    r60 = df[dcol.isin(last_60)].copy()
    r60["date"] = pd.to_datetime(r60["date"])
    r60 = r60.sort_values("date")
    r60["close"] = pd.to_numeric(r60["close"], errors="coerce")
    r60["volume"] = pd.to_numeric(r60["volume"], errors="coerce")
    r60 = r60[(r60["close"] > 0) & (r60["volume"] >= 0)]
    r60["turn_cr"] = r60["close"] * r60["volume"] / 1e7

    g60 = r60.groupby("symbol", observed=True).agg(
        days60=("date", "nunique"),
        med_turn_60d_cr=("turn_cr", "median"),
        last_close=("close", "last"),
        series=("series", "last"),
    )
    r20 = r60[r60["date"].isin(last_20)]
    g20 = r20.groupby("symbol", observed=True).agg(
        days20=("date", "nunique"),
        med_turn_20d_cr=("turn_cr", "median"),
    )
    out = g60.join(g20, how="left").fillna({"days20": 0, "med_turn_20d_cr": 0.0})
    out.index = out.index.astype(str).str.strip().str.upper()
    return out, as_of


def build_liquid_universe(
    liq_df: pd.DataFrame,
    as_of: str,
    core_tickers: Set[str],
    eq_names: Dict[str, str],
    eq_series: Dict[str, str],
    mcap_map: Dict[str, float],
    ind_map: Dict[str, str],
    ext_names: Dict[str, str],
    surv_map: Dict[str, str],
    prev_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Pure, deterministic builder applying all 5 guardrails + hysteresis to produce the
    dynamic liquid expansion table (non-core equities only).
    """
    if liq_df is None or liq_df.empty:
        return pd.DataFrame(columns=COLS)

    prev_map: Dict[str, dict] = {}
    if prev_df is not None and not prev_df.empty and "Ticker" in prev_df.columns:
        for r in prev_df.to_dict("records"):
            prev_map[_norm_ticker(r["Ticker"])] = r

    rows = []
    for sym, r in liq_df.iterrows():
        t = _norm_ticker(sym)
        if not t or t in core_tickers:
            continue
        # Guardrail 1: Official NSE EQUITY_L equity only (drops ETFs, REITs, InvITs, debt)
        if eq_names and sym not in eq_names:
            continue

        ser = str(r.get("series") or eq_series.get(sym, "")).strip().upper()
        if ser != "EQ":
            continue

        last_px = float(r.get("last_close") or 0.0)
        if not np.isfinite(last_px) or last_px < MIN_PRICE_RS:
            continue

        surv_flag = surv_map.get(sym, "")
        if _is_hard_surveillance(surv_flag):
            continue

        days60 = int(r.get("days60") or 0)
        med60 = float(r.get("med_turn_60d_cr") or 0.0)
        med20 = float(r.get("med_turn_20d_cr") or 0.0)
        mc = mcap_map.get(t, np.nan)
        mc_known = bool(pd.notna(mc) and np.isfinite(mc) and float(mc) > 0)

        # Guardrail 3: MCap >= Rs 500 Cr whenever known
        if mc_known and float(mc) < MIN_MCAP_CR:
            continue

        was_in = t in prev_map
        status = ""
        if mc_known:
            # Verified >= 500 Cr equity: enter at 60D >= 2.0 & 20D >= 1.5; retain down to 60D >= 1.0
            if days60 >= MIN_SESSIONS_60D and med60 >= ENTER_MED_60D_CR and med20 >= ENTER_MED_20D_CR:
                status = "VERIFIED"
            elif was_in and days60 >= MIN_IPO_SESSIONS and med60 >= KEEP_MED_60D_CR and med20 >= 0.75:
                status = "RETAINED"
            else:
                continue
        else:
            # Unscraped new listing / IPO: stricter liquidity bar until Screener confirms MCap >= 500 Cr
            if days60 >= MIN_IPO_SESSIONS and med60 >= UNVERIFIED_MED_60D_CR and med20 >= UNVERIFIED_MED_20D_CR:
                status = "IPO_UNVERIFIED"
            elif was_in and days60 >= MIN_IPO_SESSIONS and med60 >= ENTER_MED_60D_CR and med20 >= ENTER_MED_20D_CR:
                status = "RETAINED_UNVERIFIED"
            else:
                continue

        prev_row = prev_map.get(t, {})
        sub_ind = (
            ind_map.get(t)
            or (str(prev_row.get("Sub_Industry") or "").strip() if prev_row else "")
            or "Unknown"
        )
        comp_name = (
            ext_names.get(t)
            or eq_names.get(sym)
            or (str(prev_row.get("Company_Name") or "").strip() if prev_row else "")
            or sym
        )
        added_date = str(prev_row.get("added_date") or "").strip() or as_of

        rows.append(dict(
            Ticker=t,
            Company_Name=comp_name,
            Sub_Industry=" ".join(str(sub_ind).split()),
            med_turn_60d_cr=round(med60, 2),
            med_turn_20d_cr=round(med20, 2),
            mcap_cr=round(float(mc), 0) if mc_known else np.nan,
            last_close=round(last_px, 2),
            series=ser,
            surveillance=surv_flag,
            status=status,
            added_date=added_date,
            as_of=as_of,
        ))

    if not rows:
        return pd.DataFrame(columns=COLS)
    out = pd.DataFrame(rows, columns=COLS)
    return out.sort_values(["med_turn_60d_cr", "Ticker"], ascending=[False, True]).reset_index(drop=True)


def sync_liquid_universe(root: str = ".", write: bool = True, eq_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Refresh data/liquid_universe.csv from local NSE bhavcopy parquets + EQUITY_L.csv + local
    Screener metadata. Pure local I/O (< 2 seconds, zero network dependency).
    Refuses to overwrite if the NSE store is missing or degraded.
    """
    core_path = os.path.join(root, CORE_CSV)
    core_tickers: Set[str] = set()
    if os.path.exists(core_path):
        try:
            cdf = pd.read_csv(core_path)
            core_tickers = {_norm_ticker(t) for t in cdf["Ticker"].dropna()}
        except Exception:
            pass

    eq_names, eq_series = _load_equity_list(root)
    liq_df, as_of = compute_nse_liquidity_table(eq_df=eq_df, root=root)
    if liq_df.empty or (eq_df is None and len(liq_df) < MIN_ACTIVE_STORE_EQUITIES):
        print(f"[LIQUID] NSE store has only {len(liq_df)} active symbols (< {MIN_ACTIVE_STORE_EQUITIES}) — keeping existing {LIQUID_CSV}", flush=True)
        return load_liquid_df(root=root)

    mcap_map, ind_map, ext_names = _load_known_mcap_and_industry(root)
    surv_map = _load_surveillance(root)
    prev_df = load_liquid_df(root=root)

    new_df = build_liquid_universe(
        liq_df=liq_df,
        as_of=as_of,
        core_tickers=core_tickers,
        eq_names=eq_names,
        eq_series=eq_series,
        mcap_map=mcap_map,
        ind_map=ind_map,
        ext_names=ext_names,
        surv_map=surv_map,
        prev_df=prev_df,
    )

    if write and len(new_df) > 0:
        out_path = os.path.join(root, LIQUID_CSV)
        atomic_to_csv(new_df, out_path, index=False)
        n_ver = int(new_df["status"].isin(["VERIFIED", "RETAINED"]).sum())
        n_ipo = int(new_df["status"].isin(["IPO_UNVERIFIED", "RETAINED_UNVERIFIED"]).sum())
        print(f"[LIQUID] Synced {len(new_df)} liquid >=2Cr equities ({n_ver} MCap-verified >=500Cr, {n_ipo} IPO/unverified >=5Cr) as of {as_of} -> {LIQUID_CSV}", flush=True)
    return new_df


def load_liquid_df(root: str = ".") -> pd.DataFrame:
    path = os.path.join(root, LIQUID_CSV)
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLS)
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=COLS)


@lru_cache(maxsize=4)
def _nifty500_set_cached(path: str, mtime: float) -> frozenset:
    try:
        d = pd.read_csv(path)
        return frozenset(_norm_ticker(t) for t in d["Ticker"].dropna() if str(t).strip())
    except Exception:
        return frozenset()


def nifty500_tickers(root: str = ".") -> Set[str]:
    """Canonical Nifty 500 ticker set (used for Regime Badge breadth anchor & Nifty 500 UI tier)."""
    path = os.path.join(root, NIFTY500_CSV)
    if not os.path.exists(path):
        return set()
    return set(_nifty500_set_cached(path, os.path.getmtime(path)))


def effective_liquidity_cr(df: pd.DataFrame) -> pd.Series:
    """
    Best available median daily turnover (Rs Cr/day) per row in `df`:
    prefers 60D median (`med_turn_60d_cr`) -> 20D median (`med_turn_20d_cr`) ->
    precomputed `liq_cr` -> `averageVolume * currentPrice / 1e7`.
    """
    idx = df.index
    out = pd.Series(np.nan, index=idx, dtype=float)
    for col in ("med_turn_60d_cr", "med_turn_20d_cr", "liq_cr"):
        if col in df.columns:
            out = out.fillna(pd.to_numeric(df[col], errors="coerce"))
    if out.isna().any() and "averageVolume" in df.columns:
        px = pd.to_numeric(df["currentPrice"], errors="coerce") if "currentPrice" in df.columns else (
            pd.to_numeric(df["price"], errors="coerce") if "price" in df.columns else pd.Series(np.nan, index=idx)
        )
        av = pd.to_numeric(df["averageVolume"], errors="coerce")
        est = (av * px / 1e7).where((av > 0) & (px > 0))
        out = out.fillna(est)
    return out


def is_noisy_mask(df: pd.DataFrame) -> pd.Series:
    """True for trade-to-trade (BE/BZ) or NSE surveillance (⚠️ / ASM / GSM) rows."""
    idx = df.index
    noisy = pd.Series(False, index=idx, dtype=bool)
    if "nse_flags" in df.columns:
        nf = df["nse_flags"].fillna("").astype(str)
        noisy = noisy | nf.str.contains("⚠️|ASM|GSM", case=False, regex=True)
    if "surveillance" in df.columns:
        sv = df["surveillance"].fillna("").astype(str).str.strip()
        noisy = noisy | sv.ne("")
    if "nse_series" in df.columns:
        sr = df["nse_series"].fillna("").astype(str).str.strip().str.upper()
        noisy = noisy | sr.isin(["BE", "BZ", "ST", "SM"])
    elif "series" in df.columns:
        sr = df["series"].fillna("").astype(str).str.strip().str.upper()
        noisy = noisy | sr.isin(["BE", "BZ", "ST", "SM"])
    return noisy


def apply_screen_filter(
    df: pd.DataFrame,
    tier: str = TIER_TRADABLE,
    hide_noisy: bool = True,
    nifty500_set: Optional[Set[str]] = None,
    root: str = ".",
) -> pd.DataFrame:
    """
    Apply the user's selected Liquidity Tier & Surveillance/BE hygiene filter to a market snapshot
    or funnel table. Pure, non-mutating, safe on partial test DataFrames.
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    out = df.copy()
    mask = pd.Series(True, index=out.index, dtype=bool)

    if tier == TIER_NIFTY500:
        n500 = nifty500_set if nifty500_set is not None else nifty500_tickers(root=root)
        if n500 and "ticker" in out.columns:
            mask &= out["ticker"].astype(str).map(_norm_ticker).isin(n500)
    elif tier in TIER_MIN_CR:
        min_cr = TIER_MIN_CR[tier]
        liq = effective_liquidity_cr(out)
        # Only drop rows when liquidity is computable and strictly below threshold
        mask &= liq.isna() | (liq >= min_cr)
        # Also keep sub-Rs20 penny stocks out of tradable/core/inst discovery tiers
        px_col = "currentPrice" if "currentPrice" in out.columns else ("price" if "price" in out.columns else None)
        if px_col is not None:
            px = pd.to_numeric(out[px_col], errors="coerce")
            mask &= px.isna() | (px >= MIN_PRICE_RS)

    if hide_noisy:
        mask &= ~is_noisy_mask(out)

    return out[mask].reset_index(drop=True)


def filter_summary(df_full: pd.DataFrame, df_screen: pd.DataFrame, tier: str, hide_noisy: bool) -> str:
    """Compact 1-line sidebar caption explaining active screen scope (zero clutter)."""
    if df_full is None or df_full.empty:
        return ""
    base = df_full[df_full["extended"].ne(True)] if "extended" in df_full.columns else df_full
    scr = df_screen[df_screen["extended"].ne(True)] if (df_screen is not None and "extended" in df_screen.columns) else df_screen
    n_tot = len(base)
    n_scr = len(scr) if scr is not None else 0
    hidden = max(n_tot - n_scr, 0)
    if hidden == 0:
        return f"Screening all **{n_scr:,}** stocks"
    hyg = " + ASM/BE" if hide_noisy else ""
    short_tier = tier.split(" ", 1)[-1] if " " in tier else tier
    return f"Screening **{n_scr:,}** / {n_tot:,} ({short_tier}{hyg} · hid {hidden:,})"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Sync guarded dynamic >= Rs 2 Cr/day NSE liquid equity universe.")
    ap.add_argument("--dry-run", action="store_true", help="Compute and print stats without writing CSV")
    args = ap.parse_args(argv)
    df = sync_liquid_universe(root=".", write=not args.dry_run)
    print(f"Total liquid additions outside core nifty1000_list.csv: {len(df)}")
    if not df.empty:
        print("Status breakdown:", df["status"].value_counts().to_dict())
        print("Top 15 by 60D median turnover:")
        print(df.head(15)[["Ticker", "med_turn_60d_cr", "med_turn_20d_cr", "mcap_cr", "Sub_Industry", "status"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
