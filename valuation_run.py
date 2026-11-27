"""
Mechanical v4.1 valuation for every in-scope stock -> data/valuation_snapshot.csv
==================================================================================

One row per stock: rating, CMP percentile, P10/P50/P90, implied 3-year CAGR
per scenario, gate status, and the full 1-99 percentile grid of the fair-value
distribution. The dashboard re-percentiles that grid at today's price, so the
table stays current between runs; the single-stock view re-runs the engine
live (utils/valuation.value_stock).

Prices: the latest daily cache (data/cache/market_master_*.parquet), with its
52-week high/low for Gate E.

Run: python valuation_run.py   (also runs after fundamentals_annual.py in the
     Ranking / Rotation workflow, script=fundamentals)
"""
from __future__ import annotations

import glob
import sys
import time

import pandas as pd

from utils.valuation import load_context, summary_row, value_stock

OUT = "data/valuation_snapshot.csv"


def main():
    ctx = load_context()
    if ctx.annual.empty:
        sys.exit("[valuation] data/fundamentals_annual.csv missing -- run fundamentals_annual.py first")
    try:                                     # so the dashboard loads them instead of rebuilding on each restart
        from utils.valuation import save_features, FEATURES_FILE
        save_features(ctx)
        print(f"[valuation] peer features -> {FEATURES_FILE}")
    except Exception as e:
        print(f"[valuation] could not save peer features: {e}")
    files = sorted(glob.glob("data/cache/market_master_*.parquet"))
    snap = pd.read_parquet(files[-1]) if files else pd.DataFrame()
    if not snap.empty and "extended" in snap.columns:
        snap = snap[snap["extended"] != True]
    px = snap.drop_duplicates(subset=["ticker"], keep="last").set_index("ticker") if not snap.empty else pd.DataFrame()
    core_annual_syms = set(ctx.annual.ticker)
    tickers = sorted(core_annual_syms & set(px.index)) if not px.empty else sorted(core_annual_syms)
    # Also value newly promoted liquid >=2Cr equities whose Screener fundamentals live in
    # fundamentals_ext_annual / fundamentals_annual_extra WITHOUT modifying core peer sets.
    extra_tickers: list[str] = []
    liq_last_px: dict[str, float] = {}
    ctx_ext = None
    try:
        from utils import liquid_universe as _lu
        ldf = _lu.load_liquid_df()
        if not ldf.empty and "Ticker" in ldf.columns and "last_close" in ldf.columns:
            liq_last_px = {
                str(r.Ticker).strip(): float(r.last_close)
                for r in ldf.itertuples()
                if pd.notna(r.last_close) and float(r.last_close) > 0
            }
        liq_want = set(liq_last_px) | (set(ldf["Ticker"].dropna().astype(str)) if not ldf.empty and "Ticker" in ldf.columns else set())
        cand_extra = (liq_want | (set(px.index) if not px.empty else set())) - core_annual_syms
        if cand_extra:
            ctx_ext = load_context(extended=True)
            ext_have = set(ctx_ext.annual.ticker) if not ctx_ext.annual.empty else set()
            extra_tickers = sorted(cand_extra & ext_have)
    except Exception as e:
        print(f"[valuation] extended liquid context skipped: {e}")

    def _run_batch(batch_tickers, batch_ctx):
        batch_rows = []
        for k, t in enumerate(batch_tickers):
            price = None
            if t in px.index:
                for _pcol in ("price", "currentPrice"):
                    if _pcol in px.columns and pd.notna(px.at[t, _pcol]):
                        price = float(px.at[t, _pcol])
                        break
            if price is None and t in liq_last_px:
                price = liq_last_px[t]
            hi = float(px.at[t, "fiftyTwoWeekHigh"]) if t in px.index and "fiftyTwoWeekHigh" in px and pd.notna(px.at[t, "fiftyTwoWeekHigh"]) else None
            lo = float(px.at[t, "fiftyTwoWeekLow"]) if t in px.index and "fiftyTwoWeekLow" in px and pd.notna(px.at[t, "fiftyTwoWeekLow"]) else None
            try:
                res = value_stock(t, batch_ctx, price=price, knobs=True, hi52=hi, lo52=lo)
            except Exception as e:                       # one bad stock must not stop the run
                res = {"ticker": t.replace(".NS", ""), "status": "ERROR", "reason": str(e)[:120]}
            batch_rows.append(summary_row(res))
            if (k + 1) % 200 == 0:
                print(f"[valuation] {k + 1}/{len(batch_tickers)} ({time.time() - t0:.0f}s)", flush=True)
        return batch_rows

    t0 = time.time()
    rows = _run_batch(tickers, ctx)
    if extra_tickers and ctx_ext is not None:
        print(f"[valuation] valuing {len(extra_tickers)} liquid expansion stocks via extended Screener context...", flush=True)
        rows.extend(_run_batch(extra_tickers, ctx_ext))
    T = pd.DataFrame(rows)
    try:
        from utils.snapshots import bar_date as _snap_bar_date
        _bd = _snap_bar_date(snap, files[-1] if files else None)
        T["as_of"] = _bd.strftime("%Y-%m-%d") if _bd is not None else ""
    except Exception:
        T["as_of"] = files[-1].split("master_")[1][:10].replace("_", "-") if files else ""
    T.to_csv(OUT, index=False)
    ok = T[T.status == "OK"]
    print(f"[valuation] {len(T)} stocks in {time.time() - t0:.0f}s -> {OUT}")
    print(f"   status: {T.status.value_counts().to_dict()}")
    if len(ok):
        rat = ok.rating.str.replace(r" \(gate.*", " (gate fail)", regex=True)
        print(f"   ratings: {rat.value_counts().to_dict()}")
        print(f"   identity: {ok.identity.value_counts().to_dict()}; median CMP percentile {ok.cmp_pct.median():.0f}")
    if (T.status == "ERROR").any():
        print("   errors:", T[T.status == "ERROR"][["ticker", "reason"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
