#!/usr/bin/env python3
"""
Single-process EOD Pipeline orchestrator for .github/workflows/engine.yml.

Replaces 11 separate Python invocations (avoiding repeated pandas/numpy/parquet cold starts),
isolates failures per stage so later stages still run, retires nightly IAS (`turnaround_screener`)
and Turnaround Catalyst scanning while KEEPING RS Divergence active, and prunes old snapshots
only once `close_panel.parquet` and `scorecard_derived_history.parquet` are present.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import glob
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

STEPS: Tuple[str, ...] = (
    "repair_logs",
    "nse_daily",
    "trading_engine",
    "dna3_portfolio",
    "heartbeat",
    "dna3_morning_alert",
    "alert_daemon",
    "run_daily_signals",
    "valuation_run",
    "rs_alert_daemon",
    "industry_alert_daemon",
    "prune_snapshots",
)


@contextmanager
def _argv(argv: List[str]):
    """Temporarily replace sys.argv so queued Telegram briefs and CLI parsers see expected step names."""
    old = list(sys.argv)
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = old


def prune_old_snapshots(keep: int = 30, root: str = ".") -> int:
    """
    Prune old data/cache/market_master_*.parquet snapshots beyond the newest `keep` sessions.
    Refuses to delete anything unless both close_panel.parquet and scorecard_derived_history.parquet
    exist and are non-empty, guaranteeing historical close & scorecard fidelity is preserved.
    """
    keep = max(int(keep), 1)
    panel = os.path.join(root, "data/cache/close_panel.parquet")
    derived = os.path.join(root, "data/cache/scorecard_derived_history.parquet")
    if not (os.path.exists(panel) and os.path.getsize(panel) > 0 and os.path.exists(derived) and os.path.getsize(derived) > 0):
        print("[EOD] Skipping snapshot prune: close_panel.parquet or scorecard_derived_history.parquet missing/empty")
        return 0
    files = sorted(glob.glob(os.path.join(root, "data/cache/market_master_*.parquet")))
    if len(files) <= keep:
        return 0
    victim_list = files[:-keep]
    removed = 0
    for f in victim_list:
        try:
            os.remove(f)
            removed += 1
        except OSError as e:
            print(f"[EOD] Could not remove {f}: {e}")
    print(f"[EOD] Pruned {removed} older snapshot(s); kept newest {keep}")
    return removed


def _run_step(name: str, keep_snapshots: int = 30, retry_sleep: int = 180) -> bool:
    if name == "repair_logs":
        with _argv(["scripts/repair_logs.py"]):
            from scripts import repair_logs
            return repair_logs.main() == 0

    if name == "nse_daily":
        with _argv(["scripts/nse_daily.py"]):
            from scripts import nse_daily
            return nse_daily.main() == 0

    if name == "trading_engine":
        import trading_engine
        for attempt in (1, 2):
            print(f"[EOD] Trading Engine attempt {attempt}/2...", flush=True)
            with _argv(["trading_engine.py", "--mode", "full", "--no-heartbeat"]):
                rc = trading_engine.main(["--mode", "full", "--no-heartbeat"])
            if rc == 0:
                return True
            if attempt == 1 and retry_sleep > 0:
                print(f"[EOD] Waiting {retry_sleep}s before Trading Engine retry...", flush=True)
                time.sleep(retry_sleep)
        return False

    if name == "dna3_portfolio":
        with _argv(["dna3_current_portfolio.py"]):
            from dna3_current_portfolio import OptCompV21Engine
            OptCompV21Engine().update_portfolio()
        return True

    if name == "heartbeat":
        with _argv(["trading_engine.py", "--heartbeat-only"]):
            import trading_engine
            return trading_engine.main(["--heartbeat-only"]) == 0

    if name == "dna3_morning_alert":
        with _argv(["dna3_morning_alert.py"]):
            import dna3_morning_alert
            dna3_morning_alert.send_morning_alert()
        return True

    if name == "alert_daemon":
        with _argv(["alert_daemon.py", "--force-volume"]):
            import alert_daemon
            alert_daemon.check_alerts()
        return True

    if name == "run_daily_signals":
        with _argv(["run_daily_signals.py"]):
            import run_daily_signals
            run_daily_signals.main([])
        return True

    if name == "valuation_run":
        with _argv(["valuation_run.py"]):
            import valuation_run
            valuation_run.main()
        return True

    if name == "rs_alert_daemon":
        with _argv(["rs_alert_daemon.py"]):
            import rs_alert_daemon
            return rs_alert_daemon.main() == 0

    if name == "industry_alert_daemon":
        with _argv(["industry_alert_daemon.py"]):
            import industry_alert_daemon
            return industry_alert_daemon.main() == 0

    if name == "prune_snapshots":
        prune_old_snapshots(keep=keep_snapshots)
        return True

    raise ValueError(f"Unknown EOD pipeline step: {name}")


def run_pipeline(steps: Optional[Iterable[str]] = None, skip: Optional[Iterable[str]] = None,
                 keep_snapshots: int = 30, retry_sleep: int = 180) -> Dict[str, str]:
    skip_set = set(skip or ())
    chosen = [s for s in (steps or STEPS) if s not in skip_set]
    results: Dict[str, str] = {}
    t_total = time.time()
    for name in chosen:
        t0 = time.time()
        print(f"\n=== [EOD] Step: {name} ===", flush=True)
        try:
            ok = _run_step(name, keep_snapshots=keep_snapshots, retry_sleep=retry_sleep)
            status = "OK" if ok else "WARN"
        except SystemExit as se:
            status = "OK" if se.code in (0, None) else f"WARN(exit={se.code})"
        except Exception as e:
            import traceback
            traceback.print_exc()
            status = f"FAIL({type(e).__name__})"
        elapsed = time.time() - t0
        results[name] = status
        print(f"=== [EOD] {name}: {status} ({elapsed:.1f}s) ===", flush=True)
    print(f"\n[EOD] Pipeline finished in {time.time() - t_total:.1f}s: {results}", flush=True)
    return results


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run Alpha Trend EOD pipeline in one Python process.")
    ap.add_argument("--steps", default="", help=f"Comma-separated subset of: {','.join(STEPS)}")
    ap.add_argument("--skip", default="", help="Comma-separated steps to skip")
    ap.add_argument("--keep-snapshots", type=int, default=30, help="Snapshots to retain on disk after scorecard build")
    ap.add_argument("--retry-sleep", type=int, default=180, help="Seconds between Trading Engine retries")
    args = ap.parse_args(argv)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()] if args.steps else None
    skip = [s.strip() for s in args.skip.split(",") if s.strip()] if args.skip else None
    run_pipeline(steps=steps, skip=skip, keep_snapshots=args.keep_snapshots, retry_sleep=args.retry_sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
