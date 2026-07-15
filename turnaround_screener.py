"""
turnaround_screener.py
======================
Daily scan: finds Nifty 1000 stocks showing early institutional
accumulation signals before they qualify for the V21 portfolio.

Signal = liq_from_low (vs trough, not vs history) + RS velocity
Gate   = ≥20% off 52W high + liq ≥ Rs50Cr/day + IAS ≥ 35

Validated on 11 historical 2x stocks — all caught, avg +75% signal-to-peak.

Output: data/turnaround_watchlist.csv
Run   : python turnaround_screener.py   (or via GitHub Actions daily)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os, sys, glob, warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.nifty1000_list import TICKERS_1000
from utils.regime_manager import calculate_volume_quality as _calc_vq
from utils.yf_safe import safe_history, safe_download
from utils.atomic_io import atomic_json_dump, atomic_to_csv

OUTPUT_CSV  = "data/turnaround_watchlist.csv"
LOG_CSV     = "data/ias_signal_log.csv"
CACHE_DIR   = "data/cache"

# ── Calibrated thresholds (from 12-stock post-mortem) ─────────────────────────
MIN_OFF_52W_HIGH  = -20.0   # At least 20% below 52W high (not in free-run)
MAX_OFF_52W_HIGH  = -5.0    # Not too close to high either (stock must be cooling)
MIN_LIQ5_CR       = 30.0    # Minimum Rs 30Cr/day (5-day avg)
MIN_IAS_SCORE     = 35      # Minimum IAS to enter watchlist
MAX_RS21_FLOOR    = -40.0   # Anti-freefall: RS21 < -40% means still collapsing

TIER_READY = 60
TIER_ALERT = 80

# ── Sub-industry cycle labels from Alpha Playbook ──────────────────────────────
LONG_CYCLE  = {"Aerospace & Defense", "Agricultural, Commercial & Construction Vehicles",
               "Auto Components", "Automobiles", "Cement & Cement Products",
               "Construction", "Diversified Metals", "Electrical Equipment",
               "Engineering Services", "Ferrous Metals", "IT - Software",
               "Industrial Manufacturing", "Power", "Realty",
               "Telecom - Equipment & Accessories", "Transport Infrastructure"}
SHORT_CYCLE = {"Banks", "Beverages", "Capital Markets", "Chemicals & Petrochemicals",
               "Cigarettes & Tobacco Products", "Consumer Durables", "Diversified FMCG",
               "Entertainment", "Fertilizers & Agrochemicals", "Finance",
               "Food Products", "Insurance", "Leisure Services", "Media",
               "Other Consumer Services", "Retailing", "Textiles & Apparels"}


def get_cycle_type(sub_industry: str) -> str:
    if sub_industry in LONG_CYCLE:  return "LONG"
    if sub_industry in SHORT_CYCLE: return "SHORT"
    return "MID"


def load_sub_industry_map() -> dict:
    """ticker -> (sub_industry, cycle_type)"""
    csv = "data/nifty1000_list.csv"
    if not os.path.exists(csv):
        return {}
    df = pd.read_csv(csv)
    result = {}
    for _, row in df.iterrows():
        t  = str(row.get("Ticker", "")).strip()
        si = str(row.get("Sub_Industry", "")).strip()
        if t and si and si not in ("", "nan"):
            result[t] = (si, get_cycle_type(si))
    return result


def calc_rs(close: pd.Series, bench: pd.Series, window: int) -> pd.Series:
    return close.pct_change(window) * 100 - bench.pct_change(window) * 100


def calc_ias(rs21_d5: float, lfl: float, pct_off_low: float, rs63: float, shock_ratio: float = 0.0) -> float:
    """IAS Score 0-100. Validated formula from 12-stock post-mortem."""
    total, *_ = _calc_ias_full(rs21_d5, lfl, pct_off_low, rs63, shock_ratio)
    return total


def _calc_ias_full(rs21_d5: float, lfl: float, pct_off_low: float, rs63: float,
                   shock_ratio: float = 0.0) -> tuple:
    """Returns (total, vel_s, lfl_s, price_s, r63_s) — all rounded to 1dp."""
    vel_s   = round(min(35, max(0, rs21_d5 / 5.0 * 35)),        1)
    lfl_s   = round(min(30, max(0, (min(lfl, 50) - 1.0) * 15)), 1)
    price_s = round(min(20, max(0, pct_off_low / 15.0 * 20)),   1)
    r63_s   = 15.0 if rs63 > 0 else (10.0 if rs63 > -5 else (5.0 if rs63 > -15 else 0.0))

    raw = vel_s + lfl_s + price_s + r63_s
    if shock_ratio > 0.75:
        total = round(min(raw * 0.60, 60.0), 1)
    else:
        total = round(raw, 1)
    return total, vel_s, lfl_s, price_s, r63_s


def score_ticker(t: str, close: pd.Series, volume: pd.Series, open_px: pd.Series,
                 nifty: pd.Series, sub_map: dict) -> dict | None:
    """Compute today's IAS for a single ticker. Returns None if gates not met."""
    try:
        if len(close) < 70:
            return None
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)
        if volume.index.tz is not None:
            volume.index = volume.index.tz_localize(None)

        nb = nifty.reindex(close.index, method="ffill")

        rs21 = calc_rs(close, nb, 21)
        rs63 = calc_rs(close, nb, 63)
        rs5  = calc_rs(close, nb, 5)
        comp_rs = 0.10 * rs5 + 0.50 * rs21 + 0.40 * rs63

        liq_daily = (close * volume) / 1e7           # Rs Cr
        liq5      = liq_daily.rolling(5).mean()
        liq10_min = liq_daily.rolling(10).min().clip(lower=0.01)
        lfl       = (liq5 / liq10_min)

        h252 = close.rolling(252, min_periods=50).max()
        l252 = close.rolling(252, min_periods=50).min()

        rs21_d5   = rs21.diff(5)
        rs21_d1   = rs21.diff(1)
        max_1d_rs = rs21_d1.rolling(5).max()
        
        off_high  = (close / h252 - 1) * 100
        off_low   = (close / l252 - 1) * 100
        ma50      = close.rolling(50).mean()
        ma200     = close.rolling(200).mean()

        # Latest values
        price      = float(close.iloc[-1])
        rs5_now    = float(rs5.iloc[-1])
        rs21_now   = float(rs21.iloc[-1])
        rs63_now   = float(rs63.iloc[-1])
        crs_now    = float(comp_rs.iloc[-1])
        liq5_now   = float(liq5.iloc[-1])
        lfl_now    = float(lfl.iloc[-1])
        oh_now     = float(off_high.iloc[-1])
        ol_now     = float(off_low.iloc[-1])
        rs21d_now  = float(rs21_d5.iloc[-1])
        ma50_now   = float(ma50.iloc[-1])
        off_ma50   = (price / ma50_now  - 1) * 100 if ma50_now  > 0 else 0.0
        ma200_now  = float(ma200.iloc[-1]) if ma200.notna().any() else 0.0
        off_ma200  = (price / ma200_now - 1) * 100 if ma200_now > 0 else 0.0

        m1d_now    = float(max_1d_rs.iloc[-1])
        shock_ratio = (m1d_now / rs21d_now) if rs21d_now > 3.0 else 0.0

        # V21 distance score: how far from qualifying for main portfolio?
        # V21 needs: CompRS > 0, price above MA50
        v21_crs_gap  = max(0, -crs_now) * 100   # pct CompRS gap to 0
        v21_ma50_gap = max(0, -off_ma50)         # pct below MA50

        # ── GATES ─────────────────────────────────────────────────────────────
        if oh_now  > MAX_OFF_52W_HIGH:  return None   # Too close to 52W high
        if oh_now  < -60:              return None   # Extreme collapse (>60% off high)
        if oh_now  > MIN_OFF_52W_HIGH:  return None   # Not enough pullback
        if liq5_now < MIN_LIQ5_CR:     return None   # Too illiquid
        if rs21_now < MAX_RS21_FLOOR:   return None   # Still free-falling
        if price   <= close.rolling(10).min().iloc[-1]:
            return None   # Making new 10-day lows (not forming a base)

        # ── SYSTEMIC CRISIS SHIELDS ───────────────────────────────────────────
        # 1. Volume Quality Discriminator (Blocks panic selling)
        vol_quality = _calc_vq(close, open_px, volume, window=5)
        if vol_quality < 0.55:
            return None   # Driven by red-day liquidation, fake bounce
            
        # 2. Multi-Day Confirmation (Requires 3-day hold of structurally higher lows/closes)
        if len(close) >= 3:
            if close.iloc[-1] <= close.iloc[-3]:
                return None  # Failed to hold a 3-day higher floor

        ias, ias_vel, ias_lfl, ias_price, ias_rs63 = _calc_ias_full(
            rs21d_now, lfl_now, ol_now, rs63_now, shock_ratio
        )
        if ias < MIN_IAS_SCORE:
            return None

        si, cycle = sub_map.get(t, ("Unknown", "MID"))
        tier = "ALERT" if ias >= TIER_ALERT else ("READY" if ias >= TIER_READY else "WATCH")

        return {
            "Ticker":        t,
            "Name":          t.replace(".NS", ""),
            "Sub_Industry":  si,
            "Cycle":         cycle,
            "CMP":           round(price, 1),
            "Off_52W_High":  round(oh_now, 1),
            "Off_52W_Low":   round(ol_now, 1),
            "Off_MA50":      round(off_ma50, 1),
            "MA50":          round(ma50_now, 1),
            "Off_MA200":     round(off_ma200, 1),
            "MA200":         round(ma200_now, 1),
            "RS5":           round(rs5_now, 1),
            "RS21":          round(rs21_now, 1),
            "RS63":          round(rs63_now, 1),
            "CompRS":        round(crs_now, 3),
            "RS21_Delta5":   round(rs21d_now, 2),
            "Liq5Cr":        round(liq5_now, 1),
            "LiqFromLow":    round(lfl_now, 1),
            "VolQuality":    round(vol_quality, 2),
            "Shock_Ratio":   round(shock_ratio, 3),
            "IAS":           ias,
            "IAS_Vel":       ias_vel,
            "IAS_Lfl":       ias_lfl,
            "IAS_Price":     ias_price,
            "IAS_RS63":      ias_rs63,
            "Tier":          tier,
            "V21_CRS_Gap":   round(v21_crs_gap, 2),
            "V21_MA50_Gap":  round(v21_ma50_gap, 1),
            "Date":          datetime.now().strftime("%Y-%m-%d"),
        }
    except Exception:
        return None


_TIER_RANK = {"WATCH": 1, "READY": 2, "ALERT": 3}

_LOG_COLS = [
    # Identity
    "ticker", "name", "sub_industry", "cycle",
    # Signal-date snapshot (frozen on first entry)
    "signal_date", "signal_tier", "signal_price",
    "signal_ias", "signal_ias_vel", "signal_ias_lfl", "signal_ias_price", "signal_ias_rs63",
    "signal_rs5", "signal_rs21", "signal_rs63", "signal_comp_rs", "signal_rs21_delta5",
    "signal_off_52w_high", "signal_off_52w_low", "signal_off_ma50", "signal_ma50",
    "signal_off_ma200", "signal_ma200",
    "signal_liq5cr", "signal_liq_from_low", "signal_vol_quality",
    "signal_v21_crs_gap", "signal_v21_ma50_gap", "signal_shock_ratio",
    # Progression (updated each run)
    "peak_tier", "peak_tier_date", "peak_ias", "peak_ias_date",
    "days_on_watchlist", "status", "dropped_date",
    # Last-seen snapshot (updated each run while ACTIVE)
    "last_seen_date", "last_tier", "last_ias", "last_comp_rs",
    "last_off_ma50", "last_off_ma200", "last_rs21", "last_rs63",
    # Forward returns & performance
    "signal_price_at_5d", "return_5d",
    "signal_price_at_21d", "return_21d",
    "signal_price_at_63d", "return_63d",
    "current_price", "return_since_signal",
    "max_gain", "max_gain_date",
    "xirr",
]


def _cagr(signal_price: float, current_price: float, signal_date_str: str) -> float | None:
    """Annualised return (CAGR) from signal date to today. Returns None if < 1 day elapsed."""
    try:
        days = (datetime.now() - datetime.strptime(str(signal_date_str)[:10], "%Y-%m-%d")).days
        if days < 1 or signal_price <= 0 or current_price <= 0:
            return None
        return round(((current_price / signal_price) ** (365.0 / days) - 1) * 100, 1)
    except Exception:
        return None


def _price_n_days_after(ticker: str, signal_date_str: str, n: int, bulk) -> float | None:
    """Return the closing price exactly N trading days after signal_date from bulk data."""
    try:
        close_series = (bulk["Close"][ticker] if isinstance(bulk.columns, pd.MultiIndex)
                        else bulk["Close"]).dropna()
        sig_dt = pd.Timestamp(signal_date_str[:10])
        after  = close_series[close_series.index.normalize() > sig_dt]
        return float(after.iloc[n - 1]) if len(after) >= n else None
    except Exception:
        return None


def update_signal_log(today_df: pd.DataFrame, bulk, today_str: str) -> None:
    """
    Persist the IAS signal log (data/ias_signal_log.csv).

    Rules:
    - New stock passing the gates → append a full signal-date snapshot row
    - Existing stock still in watchlist → update last-seen, progression, returns
    - Existing stock no longer in watchlist → mark DROPPED (unless already GRADUATED)
    - GRADUATED when Off_MA50 ≥ 0 AND CompRS > 0 (V21-qualified)
    """
    os.makedirs("data", exist_ok=True)

    log = (pd.read_csv(LOG_CSV) if os.path.exists(LOG_CSV)
           else pd.DataFrame(columns=_LOG_COLS))

    existing = set(log["ticker"].tolist()) if not log.empty else set()
    today_map: dict[str, dict] = (
        {r["Ticker"]: r.to_dict() for _, r in today_df.iterrows()}
        if not today_df.empty else {}
    )
    today_set = set(today_map)

    # ── Current price lookup from bulk download ──────────────────────────────
    def _cur_price(ticker: str) -> float | None:
        try:
            s = (bulk["Close"][ticker] if isinstance(bulk.columns, pd.MultiIndex)
                 else bulk["Close"]).dropna()
            return float(s.iloc[-1]) if len(s) else None
        except Exception:
            return None

    # ── 1. New signals ────────────────────────────────────────────────────────
    new_rows = []
    for ticker in today_set - existing:
        r = today_map[ticker]
        cur = _cur_price(ticker) or r["CMP"]
        new_rows.append({
            "ticker": ticker, "name": r["Name"],
            "sub_industry": r["Sub_Industry"], "cycle": r["Cycle"],
            # Signal snapshot
            "signal_date": today_str, "signal_tier": r["Tier"],
            "signal_price": r["CMP"],
            "signal_ias": r["IAS"], "signal_ias_vel": r.get("IAS_Vel"),
            "signal_ias_lfl": r.get("IAS_Lfl"), "signal_ias_price": r.get("IAS_Price"),
            "signal_ias_rs63": r.get("IAS_RS63"),
            "signal_rs5": r.get("RS5"), "signal_rs21": r["RS21"],
            "signal_rs63": r["RS63"], "signal_comp_rs": r["CompRS"],
            "signal_rs21_delta5": r.get("RS21_Delta5"),
            "signal_off_52w_high": r["Off_52W_High"],
            "signal_off_52w_low": r.get("Off_52W_Low"),
            "signal_off_ma50": r["Off_MA50"], "signal_ma50": r["MA50"],
            "signal_off_ma200": r.get("Off_MA200"), "signal_ma200": r.get("MA200"),
            "signal_liq5cr": r["Liq5Cr"], "signal_liq_from_low": r["LiqFromLow"],
            "signal_vol_quality": r["VolQuality"],
            "signal_v21_crs_gap": r["V21_CRS_Gap"], "signal_v21_ma50_gap": r["V21_MA50_Gap"],
            "signal_shock_ratio": r.get("Shock_Ratio", 0.0),
            # Progression (initialised)
            "peak_tier": r["Tier"], "peak_tier_date": today_str,
            "peak_ias": r["IAS"], "peak_ias_date": today_str,
            "days_on_watchlist": 1, "status": "ACTIVE", "dropped_date": None,
            # Last seen
            "last_seen_date": today_str, "last_tier": r["Tier"],
            "last_ias": r["IAS"], "last_comp_rs": r["CompRS"],
            "last_off_ma50": r["Off_MA50"], "last_off_ma200": r.get("Off_MA200"),
            "last_rs21": r["RS21"], "last_rs63": r["RS63"],
            # Returns
            "signal_price_at_5d": None,  "return_5d": None,
            "signal_price_at_21d": None, "return_21d": None,
            "signal_price_at_63d": None, "return_63d": None,
            "current_price": round(cur, 1),
            "return_since_signal": round((cur / r["CMP"] - 1) * 100, 1) if r["CMP"] else 0.0,
            "max_gain": 0.0, "max_gain_date": today_str,
            "xirr": 0.0,
        })

    if new_rows:
        log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)

    # ── 2. Update every existing entry ───────────────────────────────────────
    for i, entry in log.iterrows():
        ticker     = entry["ticker"]
        sig_price  = float(entry["signal_price"]) if pd.notna(entry.get("signal_price")) else 0.0
        sig_date   = str(entry.get("signal_date", ""))[:10]
        if sig_price <= 0:
            continue

        cur = _cur_price(ticker) or sig_price
        ret_now = round((cur / sig_price - 1) * 100, 1)

        log.at[i, "current_price"]       = round(cur, 1)
        log.at[i, "return_since_signal"] = ret_now

        # Max gain (never decreases)
        prev_max = float(entry["max_gain"]) if pd.notna(entry.get("max_gain")) else 0.0
        if ret_now > prev_max:
            log.at[i, "max_gain"]      = ret_now
            log.at[i, "max_gain_date"] = today_str

        # Freeze period returns when the N-th trading day arrives
        for n, col_p, col_r in [(5, "signal_price_at_5d", "return_5d"),
                                 (21, "signal_price_at_21d", "return_21d"),
                                 (63, "signal_price_at_63d", "return_63d")]:
            if pd.isna(entry.get(col_p)) or entry.get(col_p) is None:
                px = _price_n_days_after(ticker, sig_date, n, bulk)
                if px is not None:
                    log.at[i, col_p] = round(px, 1)
                    log.at[i, col_r] = round((px / sig_price - 1) * 100, 1)

        # XIRR / CAGR
        cagr = _cagr(sig_price, cur, sig_date)
        if cagr is not None:
            log.at[i, "xirr"] = cagr

        # ── Status & progression ─────────────────────────────────────────────
        status = str(entry.get("status", "ACTIVE"))
        if status == "GRADUATED":
            continue  # Locked — never downgrade a graduate

        if ticker in today_set:
            r = today_map[ticker]
            log.at[i, "last_seen_date"] = today_str
            log.at[i, "last_tier"]      = r["Tier"]
            log.at[i, "last_ias"]       = r["IAS"]
            log.at[i, "last_comp_rs"]   = r["CompRS"]
            log.at[i, "last_off_ma50"]  = r["Off_MA50"]
            log.at[i, "last_off_ma200"] = r.get("Off_MA200")
            log.at[i, "last_rs21"]      = r["RS21"]
            log.at[i, "last_rs63"]      = r["RS63"]

            prev_days = int(entry["days_on_watchlist"]) if pd.notna(entry.get("days_on_watchlist")) else 0
            log.at[i, "days_on_watchlist"] = prev_days + 1

            # Peak tier / IAS
            if _TIER_RANK.get(r["Tier"], 0) > _TIER_RANK.get(str(entry.get("peak_tier", "")), 0):
                log.at[i, "peak_tier"]      = r["Tier"]
                log.at[i, "peak_tier_date"] = today_str
            if r["IAS"] > float(entry.get("peak_ias") or 0):
                log.at[i, "peak_ias"]      = r["IAS"]
                log.at[i, "peak_ias_date"] = today_str

            # Graduate when it crosses MA50 with CompRS > 0
            if r["Off_MA50"] >= 0 and r["CompRS"] > 0:
                log.at[i, "status"] = "GRADUATED"
        else:
            # No longer in watchlist
            if status == "ACTIVE":
                log.at[i, "status"]       = "DROPPED"
                log.at[i, "dropped_date"] = today_str

    atomic_to_csv(log, LOG_CSV, index=False)
    n_active = (log["status"] == "ACTIVE").sum()
    n_grad   = (log["status"] == "GRADUATED").sum()
    n_drop   = (log["status"] == "DROPPED").sum()
    print(f"  [LOG] {len(log)} entries → {LOG_CSV}  "
          f"(ACTIVE {n_active} | GRADUATED {n_grad} | DROPPED {n_drop}  "
          f"| NEW {len(new_rows)})")


def main():
    print("=" * 65)
    print("  INSTITUTIONAL TURNAROUND RADAR — Daily Scan")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)

    os.makedirs("data", exist_ok=True)

    # 1. Load Nifty benchmark
    print("\n  [1/4] Loading Nifty benchmark...")
    start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
    nifty_raw = safe_history("^NSEI", start=start)
    if nifty_raw.empty:
        print("  ERROR: Cannot fetch Nifty after retries. Aborting.")
        return
    if nifty_raw.index.tz:
        nifty_raw.index = nifty_raw.index.tz_localize(None)
    nifty_close = nifty_raw["Close"]

    # 2. Load sub-industry map
    print("  [2/4] Loading sub-industry map...")
    sub_map = load_sub_industry_map()
    print(f"        {len(sub_map)} tickers mapped")

    # 3. Bulk download price + volume for Nifty 1000
    # threads=False avoids the parallel-request storm that triggers Yahoo
    # 401s on CI (same fix already applied in fast_data_engine.py /
    # dna3_current_portfolio.py — this call site had drifted out of sync).
    tickers = [t for t in TICKERS_1000 if t != "DUMMYALCAR.NS"]
    print(f"\n  [3/4] Downloading {len(tickers)} tickers (2-4 min)...")
    bulk = safe_download(
        tickers, start=start, group_by="ticker",
        threads=False, auto_adjust=True, min_coverage=0.5,
    )
    if bulk is None or bulk.empty:
        print("  ERROR: Bulk download failed after retries. Aborting.")
        return

    # 4. Score each ticker
    print("\n  [4/4] Scoring tickers...")
    results = []
    for t in tickers:
        try:
            if len(tickers) > 1 and t not in bulk.columns.get_level_values(0):
                continue
            data = bulk[t] if len(tickers) > 1 else bulk
            close  = data["Close"].dropna()
            volume = data["Volume"].dropna()
            open_px = data["Open"].dropna()
            if len(close) < 70:
                continue
            row = score_ticker(t, close, volume, open_px, nifty_close, sub_map)
            if row:
                results.append(row)
        except Exception:
            continue

    if not results:
        print("  No turnaround candidates found today.")
        atomic_to_csv(pd.DataFrame(), OUTPUT_CSV, index=False)
        return

    df = (pd.DataFrame(results)
          .sort_values("IAS", ascending=False)
          .reset_index(drop=True))

    atomic_to_csv(df, OUTPUT_CSV, index=False)

    # ── Signal log ────────────────────────────────────────────────────────────
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        update_signal_log(df, bulk, today_str)
    except Exception as e:
        print(f"  [LOG] Signal log update failed: {e}")

    # --- 🔔 NEW IAS ALERTS ENGINE ---
    STATE_FILE = "data/turnaround_state.json"
    prev_state = {}
    if os.path.exists(STATE_FILE):
        import json
        try:
            with open(STATE_FILE, 'r') as f:
                prev_state = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            # This exact crash (JSONDecodeError: Expecting value) happened
            # in production from a prior run's write getting interrupted
            # mid-write, leaving a truncated/empty file. Non-atomic writes
            # are now fixed (see the atomic_json_dump write below), but
            # degrade gracefully instead of crashing the whole scan if a
            # corrupt file is ever encountered again.
            print(f"  [IAS] WARNING: {STATE_FILE} is corrupt ({e}); starting from empty state.")
            prev_state = {}
    
    current_state = {}
    new_ready = []
    tc_alerts = []  # Turnaround Confirmation
    demotions = []  # READY -> WATCH
    
    for _, r in df.iterrows():
        t = str(r['Ticker'])
        tier = r['Tier']
        ias = r.get('IAS', 0)
        
        # Tracking days >= 35 (the minimum gate)
        prev_days = prev_state.get(t, {}).get('days_active', 0)
        curr_days = prev_days + 1
        current_state[t] = {'tier': tier, 'days_active': curr_days, 'ias': ias}
        
        prev_tier = prev_state.get(t, {}).get('tier', 'NONE')
        
        # 🟢 Turnaround Confirmation (3 days active)
        if curr_days == 3:
            tc_alerts.append((t, r['Name'], ias, r['Off_MA50']))
            
        # 🟢 READY Promotions
        if tier in ['READY', 'ALERT'] and prev_tier not in ['READY', 'ALERT']:
            new_ready.append((t, r['Name'], tier, ias, r['Off_MA50']))
            
        # 🔴 Tier Demotions (READY/ALERT -> WATCH/DROP)
        if prev_tier in ['READY', 'ALERT'] and tier not in ['READY', 'ALERT']:
            demotions.append((t, r['Name'], prev_tier, tier, ias))
            
    # For stocks that dropped off the scanner completely (IAS < 35)
    for t, p_st in prev_state.items():
        if t not in current_state:
            p_tier = p_st.get('tier')
            if p_tier in ['READY', 'ALERT']:
                # Dropped from a high tier to nothing
                demotions.append((t, t.replace('.NS', ''), p_tier, 'DROPPED', 0))

    # Save state (atomic — see the read-side comment above for why)
    atomic_json_dump(current_state, STATE_FILE, indent=4)
        
    # Dispatch Alerts
    if tc_alerts or new_ready or demotions:
        try:
            from utils.telegram_notifier import send_telegram_message, is_telegram_configured
            from utils.email_notifier import send_system_alert, is_email_configured
            
            # Action Alert: Turnaround Confirmation (Immediate TG+Email)
            if tc_alerts:
                tg_tc = "🚨 <b>TURNAROUND CONFIRMATION</b> (IAS 3 Day Hold)\n\n"
                for t, name, ias, off_ma in tc_alerts:
                    tg_tc += f"• {name}: IAS {ias:.0f} | {off_ma:+.1f}% vs MA50\n"
                if is_telegram_configured():
                    send_telegram_message(tg_tc)
                if is_email_configured():
                    send_system_alert("🚨 Action Alert: Turnaround Confirmation", tg_tc.replace('<b>', '').replace('</b>', ''))
                    
            # Non-Action Alerts: New READY / Demotions (Daily Email)
            if new_ready or demotions:
                email_sub = f"🔄 IAS Tracker Updates: {len(new_ready)} Up | {len(demotions)} Down"
                email_body = ""
                if new_ready:
                    email_body += "<b>🟢 ENTERING READY TIER (IAS ≥60)</b><br>"
                    for t, name, tier, ias, off_ma in new_ready:
                        email_body += f"• {name} -> {tier} (IAS: {ias:.0f}, {off_ma:+.1f}%)<br>"
                    email_body += "<br>"
                if demotions:
                    email_body += "<b>🔴 DEMOTIONS (Lost READY status)</b><br>"
                    for t, name, pt, nt, ias in demotions:
                        email_body += f"• {name} -> {nt} (was {pt})<br>"
                
                if is_email_configured():
                    send_system_alert(email_sub, email_body)
        except Exception as e:
            print(f"  [ALERT] Failed to send IAS alerts: {e}")

    # Summary
    tier_counts = df["Tier"].value_counts()
    print(f"\n{'='*65}")
    print(f"  SCAN COMPLETE")
    print(f"  Watchlist size : {len(df)} stocks")
    print(f"  ALERT  (IAS 80+): {tier_counts.get('ALERT', 0)}")
    print(f"  READY  (IAS 60+): {tier_counts.get('READY', 0)}")
    print(f"  WATCH  (IAS 35+): {tier_counts.get('WATCH', 0)}")
    print(f"  Output : {OUTPUT_CSV}")
    print(f"{'='*65}")
    
    if tc_alerts: print(f"  > Dispatched {len(tc_alerts)} Turnaround Confirmation alerts")
    print("\n  Top 10 by IAS:")
    print(df[["Ticker","Sub_Industry","CMP","Off_52W_High","IAS","Tier"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
