"""
Turnaround Catalyst Backtest Engine
Patterns: A (BigDay), B (TSJump), A+B (both)
Entry: T+1 close  |  Exit: fixed hold OR stop-loss
"""

import glob
import json
import os
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────

CACHE_DIR   = "data/cache"
RESULTS_DIR = "data"
RESULTS_FILE = os.path.join(RESULTS_DIR, "tc_backtest_results.json")
TRADES_FILE  = os.path.join(RESULTS_DIR, "tc_backtest_trades.csv")

# Pattern thresholds (must match production scanner)
PATTERN_A = dict(min_chg=5.0, min_vol=7, max_dist=-20, max_ts_pre=80)
PATTERN_B = dict(min_ts_gain=20, min_vol=6, max_dist=-20, max_ts_pre=55, min_chg=2.5)

# Hold periods to test (trading days)
HOLD_PERIODS = [3, 5, 10, 20]

# Stop-loss levels to test (negative %)
STOP_LEVELS = [-5.0, -7.0, -10.0]

# Risk-free rate per trading day (approx 7% annual / 252)
RF_DAILY = 0.07 / 252


# ── Data Loader ─────────────────────────────────────────────────────────────

def load_all_parquets() -> tuple[list[pd.DataFrame], list[str]]:
    files = sorted(glob.glob(f"{CACHE_DIR}/market_master_*.parquet"))
    dfs, labels = [], []
    for f in files:
        df = pd.read_parquet(f)
        if "rs_1m" in df.columns and "return_1m" not in df.columns:
            df["return_1m"] = df["rs_1m"]
        df["_file"] = f
        dfs.append(df)
        labels.append(
            f.split("market_master_")[1].replace(".parquet", "").replace("_", "-")
        )
    return dfs, labels


# ── Signal Detection ─────────────────────────────────────────────────────────

def detect_signals(today: pd.DataFrame, prev: pd.DataFrame, date: str, day_idx: int) -> list[dict]:
    today = today.set_index("ticker")
    prev  = prev.set_index("ticker")
    common = today.index.intersection(prev.index)

    ts_gain = (today.loc[common, "trend_score"] - prev.loc[common, "trend_score"])

    signals = []
    for ticker in common:
        r   = today.loc[ticker]
        chg  = float(r.get("change_p", 0) or 0)
        vol  = float(r.get("volume_score", 0) or 0)
        dist = float(r.get("dist_52w", -50) if pd.notna(r.get("dist_52w")) else -50)
        ts_n = float(r.get("trend_score", 50) or 50)
        ts_p = float(prev.loc[ticker, "trend_score"]) if ticker in prev.index else ts_n
        tsg  = float(ts_gain.get(ticker, 0) or 0)
        price = float(r.get("price", r.get("currentPrice", np.nan)))

        pat_a = (chg >= PATTERN_A["min_chg"]
                 and vol >= PATTERN_A["min_vol"]
                 and dist < PATTERN_A["max_dist"]
                 and ts_p < PATTERN_A["max_ts_pre"])

        pat_b = (tsg >= PATTERN_B["min_ts_gain"]
                 and vol >= PATTERN_B["min_vol"]
                 and dist < PATTERN_B["max_dist"]
                 and ts_p < PATTERN_B["max_ts_pre"]
                 and chg >= PATTERN_B["min_chg"])

        if not (pat_a or pat_b):
            continue

        pattern = "A+B" if (pat_a and pat_b) else ("A" if pat_a else "B")

        signals.append(dict(
            signal_date   = date,
            day_idx       = day_idx,
            ticker        = ticker,
            name          = str(r.get("name", "")),
            sector        = str(r.get("sector", "")),
            sub_industry  = str(r.get("sector_granular", "")),
            pattern       = pattern,
            change_p      = round(chg, 2),
            vol_score     = round(vol, 1),
            ts_pre        = round(ts_p, 1),
            ts_now        = round(ts_n, 1),
            ts_gain       = round(tsg, 1),
            dist_52w      = round(dist, 1),
            rs21          = round(float(r.get("return_1m", 0) or 0), 2),
            signal_price  = round(price, 2),
        ))

    return signals


# ── Forward Returns ──────────────────────────────────────────────────────────

def compute_forward_returns(
    signals: list[dict],
    dfs: list[pd.DataFrame],
    labels: list[str],
) -> pd.DataFrame:
    """For every signal, look up price at T+1 (entry) and T+1+hold (exit)."""
    df_sig = pd.DataFrame(signals)
    n = len(dfs)

    for hold in HOLD_PERIODS:
        df_sig[f"ret_{hold}d"]      = np.nan
        df_sig[f"entry_{hold}d"]    = np.nan
        df_sig[f"exit_{hold}d"]     = np.nan

    for stop in STOP_LEVELS:
        col = f"ret_sl{abs(int(stop))}pct"
        df_sig[col] = np.nan

    for idx, row in df_sig.iterrows():
        di = int(row["day_idx"])
        tk = row["ticker"]

        # Entry: T+1 close
        entry_idx = di + 1
        if entry_idx >= n:
            continue
        entry_df = dfs[entry_idx].set_index("ticker")
        if tk not in entry_df.index:
            continue
        ep = float(entry_df.loc[tk, "price"] if "price" in entry_df.columns
                   else entry_df.loc[tk].get("currentPrice", np.nan))
        if pd.isna(ep) or ep == 0:
            continue

        # Fixed hold exits
        for hold in HOLD_PERIODS:
            exit_idx = di + 1 + hold
            if exit_idx >= n:
                continue
            exit_df = dfs[exit_idx].set_index("ticker")
            if tk not in exit_df.index:
                continue
            xp = float(exit_df.loc[tk, "price"] if "price" in exit_df.columns
                       else exit_df.loc[tk].get("currentPrice", np.nan))
            if pd.isna(xp) or xp == 0:
                continue
            ret = (xp / ep - 1) * 100
            df_sig.at[idx, f"ret_{hold}d"]   = round(ret, 3)
            df_sig.at[idx, f"entry_{hold}d"] = round(ep, 2)
            df_sig.at[idx, f"exit_{hold}d"]  = round(xp, 2)

        # Stop-loss exits (check each day, exit on first breach)
        for stop in STOP_LEVELS:
            col = f"ret_sl{abs(int(stop))}pct"
            max_hold = max(HOLD_PERIODS)
            ret_with_sl = np.nan
            for d in range(1, max_hold + 1):
                check_idx = di + 1 + d
                if check_idx >= n:
                    break
                check_df = dfs[check_idx].set_index("ticker")
                if tk not in check_df.index:
                    break
                cp = float(check_df.loc[tk, "price"] if "price" in check_df.columns
                           else check_df.loc[tk].get("currentPrice", np.nan))
                if pd.isna(cp) or cp == 0:
                    break
                cur_ret = (cp / ep - 1) * 100
                if cur_ret <= stop:
                    ret_with_sl = round(cur_ret, 3)
                    break
                if d == max_hold:
                    ret_with_sl = round(cur_ret, 3)
            df_sig.at[idx, col] = ret_with_sl

    return df_sig


# ── Statistics ───────────────────────────────────────────────────────────────

def stats(returns: pd.Series, label: str = "") -> dict:
    r = returns.dropna()
    if len(r) < 3:
        return {"label": label, "n": len(r), "note": "insufficient data"}
    wins = (r > 0).sum()
    losses = (r <= 0).sum()
    avg_win  = r[r > 0].mean() if wins > 0 else 0
    avg_loss = r[r <= 0].mean() if losses > 0 else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
    # Sharpe (annualised assuming 252 trading days)
    daily_r = r / 100
    sharpe = (daily_r.mean() - RF_DAILY) / daily_r.std() * np.sqrt(252) if daily_r.std() > 0 else np.nan
    # Max drawdown on cumulative equity curve (equal-weight)
    eq = (1 + daily_r).cumprod()
    peak = eq.cummax()
    dd = ((eq - peak) / peak * 100)
    max_dd = dd.min()

    return dict(
        label       = label,
        n           = len(r),
        win_rate    = round(wins / len(r) * 100, 1),
        median      = round(r.median(), 2),
        mean        = round(r.mean(), 2),
        std         = round(r.std(), 2),
        p25         = round(r.quantile(0.25), 2),
        p75         = round(r.quantile(0.75), 2),
        avg_win     = round(avg_win, 2),
        avg_loss    = round(avg_loss, 2),
        reward_risk = round(rr, 2) if not np.isnan(rr) else None,
        sharpe_ann  = round(sharpe, 2) if not np.isnan(sharpe) else None,
        max_dd_pct  = round(max_dd, 2),
        best        = round(r.max(), 2),
        worst       = round(r.min(), 2),
        total_ret   = round(r.sum(), 2),
    )


def build_summary(trades: pd.DataFrame, baseline_rets: dict) -> dict:
    summary = {"patterns": {}, "sectors": {}, "sub_industries": {}, "stop_loss": {}, "baseline": {}}

    # Baseline
    for hold in HOLD_PERIODS:
        b = baseline_rets.get(hold, [])
        if b:
            bs = pd.Series(b)
            summary["baseline"][f"{hold}d"] = dict(
                median=round(bs.median(), 2),
                mean=round(bs.mean(), 2),
                win_rate=round((bs > 0).mean() * 100, 1),
                n=len(bs),
            )

    # By pattern × hold period
    for pat in ["A", "B", "A+B", "ALL"]:
        sub = trades if pat == "ALL" else trades[trades["pattern"] == pat]
        summary["patterns"][pat] = {}
        for hold in HOLD_PERIODS:
            col = f"ret_{hold}d"
            s = stats(sub[col], f"{pat} {hold}d")
            # Add edge vs baseline
            bmed = summary["baseline"].get(f"{hold}d", {}).get("median", 0)
            s["edge_vs_baseline"] = round(s.get("median", 0) - bmed, 2)
            summary["patterns"][pat][f"{hold}d"] = s

    # Stop-loss analysis
    for stop in STOP_LEVELS:
        col = f"ret_sl{abs(int(stop))}pct"
        if col not in trades.columns:
            continue
        summary["stop_loss"][f"sl_{abs(int(stop))}pct"] = {}
        for pat in ["A", "B", "A+B"]:
            sub = trades[trades["pattern"] == pat]
            s = stats(sub[col], f"{pat} SL{stop}%")
            summary["stop_loss"][f"sl_{abs(int(stop))}pct"][pat] = s

    # Best sectors (Pattern A+B at 10d, min N=5)
    for hold in [5, 10]:
        col = f"ret_{hold}d"
        top_secs = []
        for sec, grp in trades[trades["pattern"].isin(["A", "A+B"])].groupby("sector"):
            r = grp[col].dropna()
            if len(r) < 4:
                continue
            top_secs.append(dict(
                sector=sec, n=len(r),
                win_rate=round((r > 0).mean() * 100, 1),
                median=round(r.median(), 2),
                mean=round(r.mean(), 2),
            ))
        summary["sectors"][f"{hold}d"] = sorted(top_secs, key=lambda x: -x["median"])

    # Best sub-industries (Pattern A at 10d, min N=4)
    for hold in [5, 10]:
        col = f"ret_{hold}d"
        top_si = []
        for si, grp in trades[trades["pattern"].isin(["A", "A+B"])].groupby("sub_industry"):
            r = grp[col].dropna()
            if len(r) < 4:
                continue
            top_si.append(dict(
                sub_industry=si, n=len(r),
                win_rate=round((r > 0).mean() * 100, 1),
                median=round(r.median(), 2),
                mean=round(r.mean(), 2),
            ))
        summary["sub_industries"][f"{hold}d"] = sorted(top_si, key=lambda x: -x["median"])

    return summary


# ── Baseline ─────────────────────────────────────────────────────────────────

def compute_baseline(dfs: list[pd.DataFrame], max_idx: int) -> dict:
    baseline = {h: [] for h in HOLD_PERIODS}
    for i in range(1, min(max_idx + 1, len(dfs) - max(HOLD_PERIODS) - 1)):
        today_df = dfs[i].set_index("ticker")
        if "price" not in today_df.columns:
            continue
        for hold in HOLD_PERIODS:
            fwd_idx = i + 1 + hold
            if fwd_idx >= len(dfs):
                continue
            fwd_df = dfs[fwd_idx].set_index("ticker")
            if "price" not in fwd_df.columns:
                continue
            common = today_df.index.intersection(fwd_df.index)
            ep = today_df.loc[common, "price"]
            xp = fwd_df.loc[common, "price"]
            rets = ((xp / ep) - 1) * 100
            baseline[hold].extend(rets.dropna().tolist())
    return baseline


# ── Main ─────────────────────────────────────────────────────────────────────

def run_backtest():
    print("Loading parquet cache...")
    dfs, labels = load_all_parquets()
    n = len(dfs)
    print(f"  {n} files: {labels[0]} → {labels[-1]}")

    max_signal_idx = n - max(HOLD_PERIODS) - 2  # leave room for T+1 entry + hold

    # Detect all signals
    print("Detecting signals...")
    all_signals = []
    for i in range(1, max_signal_idx + 1):
        sigs = detect_signals(dfs[i], dfs[i - 1], labels[i], i)
        all_signals.extend(sigs)

    print(f"  Total signals: {len(all_signals)}")
    pat_counts = pd.Series([s["pattern"] for s in all_signals]).value_counts()
    for pat, cnt in pat_counts.items():
        print(f"    Pattern {pat}: {cnt}")

    # Compute forward returns
    print("Computing forward returns (T+1 entry)...")
    trades = compute_forward_returns(all_signals, dfs, labels)

    # Baseline
    print("Computing baseline returns...")
    baseline_rets = compute_baseline(dfs, max_signal_idx)

    # Build summary
    print("Building summary statistics...")
    summary = build_summary(trades, baseline_rets)
    summary["meta"] = dict(
        date_range   = f"{labels[0]} to {labels[-1]}",
        total_files  = n,
        total_signals = len(all_signals),
        pattern_counts = pat_counts.to_dict(),
        hold_periods   = HOLD_PERIODS,
        stop_levels    = STOP_LEVELS,
        pattern_rules  = dict(
            A   = PATTERN_A,
            B   = PATTERN_B,
        ),
    )

    # Save
    trades.to_csv(TRADES_FILE, index=False)
    with open(RESULTS_FILE, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved trades → {TRADES_FILE}")
    print(f"Saved summary → {RESULTS_FILE}")

    # ── Print report ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("BACKTEST REPORT  —  Turnaround Catalyst Patterns A / B / A+B")
    print(f"Period: {labels[0]} to {labels[-1]}  |  Universe: {dfs[-1]['ticker'].nunique()} stocks")
    print("=" * 72)

    print("\n── BASELINE (all stocks, equal weight) ──")
    print(f"  {'Hold':<8} {'Median%':>9} {'Mean%':>9} {'Win%':>8} {'N':>7}")
    for hold in HOLD_PERIODS:
        b = summary["baseline"].get(f"{hold}d", {})
        print(f"  {hold}d{'':<5} {b.get('median',0):>8.2f}% {b.get('mean',0):>8.2f}% {b.get('win_rate',0):>7.1f}% {b.get('n',0):>7,}")

    for pat in ["A", "B", "A+B", "ALL"]:
        n_sig = pat_counts.get(pat, len(all_signals)) if pat != "ALL" else len(all_signals)
        print(f"\n── PATTERN {pat}  (N={n_sig} signals) ──")
        print(f"  {'Hold':<8} {'Median%':>9} {'Mean%':>9} {'Win%':>8} {'P25%':>8} {'P75%':>8} {'AvgWin':>8} {'AvgLoss':>9} {'R:R':>6} {'Edge':>7} {'Sharpe':>7} {'MaxDD%':>8}")
        for hold in HOLD_PERIODS:
            s = summary["patterns"][pat].get(f"{hold}d", {})
            if "note" in s:
                print(f"  {hold}d{'':<5} (insufficient data)")
                continue
            rr  = f"{s['reward_risk']:.2f}" if s.get("reward_risk") else "  —"
            sh  = f"{s['sharpe_ann']:.2f}"  if s.get("sharpe_ann")  else "  —"
            edge = s.get("edge_vs_baseline", 0)
            esign = "+" if edge >= 0 else ""
            print(
                f"  {hold}d{'':<5}"
                f" {s.get('median',0):>8.2f}%"
                f" {s.get('mean',0):>8.2f}%"
                f" {s.get('win_rate',0):>7.1f}%"
                f" {s.get('p25',0):>7.2f}%"
                f" {s.get('p75',0):>7.2f}%"
                f" {s.get('avg_win',0):>7.2f}%"
                f" {s.get('avg_loss',0):>8.2f}%"
                f" {rr:>6}"
                f" {esign}{edge:>5.2f}%"
                f" {sh:>7}"
                f" {s.get('max_dd_pct',0):>7.2f}%"
            )

    # Stop-loss summary
    print("\n── STOP-LOSS ANALYSIS (A+B, 20d max hold) ──")
    print(f"  {'SL Level':<12} {'Median%':>9} {'Win%':>8} {'MaxDD%':>8} {'Sharpe':>8} {'N':>6}")
    for stop in STOP_LEVELS:
        key = f"sl_{abs(int(stop))}pct"
        s = summary["stop_loss"].get(key, {}).get("A+B", {})
        if not s or "note" in s:
            continue
        sh = f"{s['sharpe_ann']:.2f}" if s.get("sharpe_ann") else "  —"
        print(f"  SL {stop}%{'':<6} {s.get('median',0):>8.2f}% {s.get('win_rate',0):>7.1f}% {s.get('max_dd_pct',0):>7.2f}% {sh:>8} {s.get('n',0):>6}")

    # Top sectors
    print("\n── TOP SECTORS for Pattern A (5d returns, min N=4) ──")
    print(f"  {'Sector':<32} {'N':>4} {'Win%':>7} {'Median5d%':>11} {'Median10d%':>12}")
    secs5  = {s["sector"]: s for s in summary["sectors"].get("5d", [])}
    secs10 = {s["sector"]: s for s in summary["sectors"].get("10d", [])}
    for sec, s5 in sorted(secs5.items(), key=lambda x: -x[1]["median"])[:12]:
        s10 = secs10.get(sec, {})
        m10 = f"{s10.get('median',0):+.2f}%" if s10 else "  —"
        print(f"  {sec[:32]:<32} {s5['n']:>4} {s5['win_rate']:>6.1f}% {s5['median']:>+10.2f}% {m10:>12}")

    # Best sub-industries
    print("\n── TOP SUB-INDUSTRIES for Pattern A (10d, min N=4) ──")
    si10 = summary["sub_industries"].get("10d", [])
    print(f"  {'Sub-Industry':<38} {'N':>4} {'Win%':>7} {'Median10d%':>12}")
    for s in si10[:12]:
        print(f"  {s['sub_industry'][:38]:<38} {s['n']:>4} {s['win_rate']:>6.1f}% {s['median']:>+11.2f}%")

    print("\nDone.")
    return trades, summary


if __name__ == "__main__":
    run_backtest()
