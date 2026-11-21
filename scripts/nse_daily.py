#!/usr/bin/env python3
"""
Evening run (.github/workflows/engine.yml): NSE's free daily data (utils/nse_data.py) and a short Telegram
section for the brief: market valuation and FII/DII flows, plus bulk/block deals, surveillance and results dates
for the stocks you hold or watch. Every source is independent; this script never fails the run.

    python scripts/nse_daily.py [--date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from utils import nse_data as nd  # noqa: E402


def _my_tickers() -> set:
    out = set()
    try:
        from utils.positions import get_all_positions
        out |= {p.get("ticker") for p in get_all_positions() if p.get("status") in ("active", "watching")}
    except Exception:
        pass
    try:
        from utils.alerts import load_alerts
        out |= {a.get("ticker") for a in load_alerts()}
    except Exception:
        pass
    return {t for t in out if t}


def brief(ctx: dict, mine: set, prev_ctx: dict) -> str:
    L = []
    if ctx.get("nifty_pe") == ctx.get("nifty_pe"):            # not NaN
        L.append(f"Nifty P/E {ctx['nifty_pe']:.1f} · P/B {ctx.get('nifty_pb', float('nan')):.2f} · yield {ctx.get('nifty_dy', float('nan')):.2f}%")
    flows = [f"{k.split('_')[0].upper()} {'+' if v >= 0 else ''}{v:,.0f} cr" for k, v in ctx.items()
             if k in ("fii_net", "dii_net") and v == v]
    if flows:
        L.append("Cash: " + " · ".join(flows))
    if ctx.get("fii_idx_long_pct") == ctx.get("fii_idx_long_pct") and "fii_idx_long_pct" in ctx:
        ch = ""
        if prev_ctx.get("fii_idx_long_pct") == prev_ctx.get("fii_idx_long_pct") and "fii_idx_long_pct" in prev_ctx:
            ch = f" ({ctx['fii_idx_long_pct'] - prev_ctx['fii_idx_long_pct']:+.1f} pp)"
        L.append(f"FII index futures {ctx['fii_idx_long_pct']:.0f}% long{ch}")
    d = nd.deals_for(mine, days=1)
    if len(d):
        L.append("Deals in your stocks: " + "; ".join(
            f"{r.symbol} {r.side.lower()} {r.qty / 1e5:.1f}L @ {r.price:,.0f} ({r.client[:22]})" for r in d.head(6).itertuples()))
    sv = nd.surveillance_map()
    hit = sorted(t.replace(".NS", "") + f" ({sv[t]})" for t in mine if t in sv)
    if hit:
        L.append("Under surveillance: " + ", ".join(hit))
    ev = nd.upcoming_events(7)
    soon = sorted(f"{t.replace('.NS', '')} {ev[t]}" for t in mine if t in ev)
    if soon:
        L.append("This week: " + ", ".join(soon))
    return ("<b>📊 NSE</b>\n" + "\n".join(L)) if L else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="")
    a = ap.parse_args()
    S = nd.Session()
    start = pd.Timestamp(a.date) if a.date else pd.Timestamp(datetime.now(nd.IST).date())
    ok = {}
    from utils import nse_prices as npx
    try:                                       # the daily price store every later step reads (utils/yf_safe.py)
        got = npx.update(S, upto=start)
        day = npx.last_date("stocks")
        print(f"[nse] prices: {got} · store holds sessions to {day:%d %b %Y} · current: {npx.is_current()}")
        ok["prices"] = True
    except Exception as x:
        print(f"[nse] prices: {x}")
        day = None
    if day is None:
        day = nd.latest_session(S, start)
    today = start
    steps = [("corporate actions", lambda: nd.fetch_corp_actions(S, today)), ("surveillance", lambda: nd.fetch_surveillance(S)),
             ("events", lambda: nd.fetch_events(S, today)), ("deals", lambda: nd.fetch_deals(S))]
    for name, fn in steps:
        try:
            r = fn()
            print(f"[nse] {name}: {len(r):,} rows")
            ok[name] = True
        except Exception as x:
            print(f"[nse] {name}: {x}")
    ctx, prev = {}, {}
    if day is not None:
        try:
            old = nd.market_context()
            prev = old[old.date.astype(str) < day.strftime("%Y-%m-%d")].iloc[-1].to_dict() if len(old) else {}
        except Exception:
            prev = {}
        try:
            ctx = nd.fetch_market_context(S, day)
            print(f"[nse] market context: {ctx}")
            ok["context"] = True
        except Exception as x:
            print(f"[nse] market context: {x}")
    text = brief(ctx, _my_tickers(), prev)
    if text:
        try:
            from utils.telegram_notifier import is_telegram_configured, send_telegram_message
            if is_telegram_configured():
                print(f"[nse] telegram: {send_telegram_message(text)[1]}")
        except Exception as x:
            print(f"[nse] telegram: {x}")
        print(text)
    print(f"[nse] done: {', '.join(ok) or 'nothing fetched'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
