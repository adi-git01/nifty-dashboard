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

OUTPUT_CSV  = "data/turnaround_watchlist.csv"
CACHE_DIR   = "data/cache"

# ── Calibrated thresholds (from 12-stock post-mortem) ─────────────────────────
MIN_OFF_52W_HIGH  = -20.0   # At least 20% below 52W high (not in free-run)
MAX_OFF_52W_HIGH  = -5.0    # Not too close to high either (stock must be cooling)
MIN_LIQ5_CR       = 50.0    # Minimum Rs 50Cr/day (5-day avg)
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


def calc_ias(rs21_d5: float, lfl: float, pct_off_low: float, rs63: float) -> float:
    """IAS Score 0-100. Validated formula from 12-stock post-mortem."""
    vel_s   = min(35, max(0, rs21_d5 / 5.0 * 35))          # RS velocity (35 pts)
    lfl_s   = min(30, max(0, (min(lfl, 50) - 1.0) * 15))   # Liq from low, capped 50x (30 pts)
    price_s = min(20, max(0, pct_off_low / 15.0 * 20))      # Bounce from 52W low (20 pts)
    r63_s   = 15 if rs63 > 0 else (10 if rs63 > -5 else (5 if rs63 > -15 else 0))
    return round(vel_s + lfl_s + price_s + r63_s, 1)


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
        off_high  = (close / h252 - 1) * 100
        off_low   = (close / l252 - 1) * 100
        ma50      = close.rolling(50).mean()

        # Latest values
        price      = float(close.iloc[-1])
        rs21_now   = float(rs21.iloc[-1])
        rs63_now   = float(rs63.iloc[-1])
        crs_now    = float(comp_rs.iloc[-1])
        liq5_now   = float(liq5.iloc[-1])
        lfl_now    = float(lfl.iloc[-1])
        oh_now     = float(off_high.iloc[-1])
        ol_now     = float(off_low.iloc[-1])
        rs21d_now  = float(rs21_d5.iloc[-1])
        ma50_now   = float(ma50.iloc[-1])
        off_ma50   = (price / ma50_now - 1) * 100 if ma50_now > 0 else 0

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

        ias = calc_ias(rs21d_now, lfl_now, ol_now, rs63_now)
        if ias < MIN_IAS_SCORE:
            return None

        si, cycle = sub_map.get(t, ("Unknown", "MID"))
        tier = "ALERT" if ias >= TIER_ALERT else ("READY" if ias >= TIER_READY else "WATCH")

        return {
            "Ticker":       t,
            "Name":         t.replace(".NS", ""),
            "Sub_Industry": si,
            "Cycle":        cycle,
            "CMP":          round(price, 1),
            "Off_52W_High": round(oh_now, 1),
            "Off_MA50":     round(off_ma50, 1),
            "MA50":         round(ma50_now, 1),
            "RS21":         round(rs21_now, 1),
            "RS63":         round(rs63_now, 1),
            "CompRS":       round(crs_now, 3),
            "Liq5Cr":       round(liq5_now, 1),
            "LiqFromLow":   round(min(lfl_now, 50), 1),
            "VolQuality":   round(vol_quality, 2),
            "IAS":          ias,
            "Tier":         tier,
            "V21_CRS_Gap":  round(v21_crs_gap, 2),
            "V21_MA50_Gap": round(v21_ma50_gap, 1),
            "Date":         datetime.now().strftime("%Y-%m-%d"),
        }
    except Exception:
        return None


def main():
    print("=" * 65)
    print("  INSTITUTIONAL TURNAROUND RADAR — Daily Scan")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)

    os.makedirs("data", exist_ok=True)

    # 1. Load Nifty benchmark
    print("\n  [1/4] Loading Nifty benchmark...")
    start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
    nifty_raw = yf.Ticker("^NSEI").history(start=start)
    if nifty_raw.empty:
        print("  ERROR: Cannot fetch Nifty. Aborting.")
        return
    if nifty_raw.index.tz:
        nifty_raw.index = nifty_raw.index.tz_localize(None)
    nifty_close = nifty_raw["Close"]

    # 2. Load sub-industry map
    print("  [2/4] Loading sub-industry map...")
    sub_map = load_sub_industry_map()
    print(f"        {len(sub_map)} tickers mapped")

    # 3. Bulk download price + volume for Nifty 1000
    tickers = [t for t in TICKERS_1000 if t != "DUMMYALCAR.NS"]
    print(f"\n  [3/4] Downloading {len(tickers)} tickers (2-4 min)...")
    try:
        bulk = yf.download(
            tickers, start=start, group_by="ticker",
            threads=True, progress=True, auto_adjust=True
        )
    except Exception as e:
        print(f"  ERROR: {e}")
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
        pd.DataFrame().to_csv(OUTPUT_CSV, index=False)
        return

    df = (pd.DataFrame(results)
          .sort_values("IAS", ascending=False)
          .reset_index(drop=True))

    df.to_csv(OUTPUT_CSV, index=False)
    
    # --- 🔔 NEW IAS ALERTS ENGINE ---
    STATE_FILE = "data/turnaround_state.json"
    prev_state = {}
    if os.path.exists(STATE_FILE):
        import json
        with open(STATE_FILE, 'r') as f:
            prev_state = json.load(f)
    
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

    # Save state
    with open(STATE_FILE, 'w') as f:
        json.dump(current_state, f, indent=4)
        
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
