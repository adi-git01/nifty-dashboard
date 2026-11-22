"""
The one daily Telegram message: what to act on, in the order you would act.

  1 Market        Nifty and breadth regime, risk guidance
  2 Holdings      only those needing attention: stop hit, target hit, near stop, exit rule hit,
                  or a stop that should be raised to the exit level
  3 Watchlist     watched stocks that pass the funnel today
  4 Funnel        stocks that pass all four stages for the first time today
  5 Strong names  today's earnings jumps and red-day gainers that are also in an uptrend
  6 one line of counts for the scanners that have shown no edge

build() returns HTML text and needs no network. Positions come from the GitHub store
(utils/positions.py), so the EOD step needs GITHUB_TOKEN in its environment.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

LADDER = ["BEAR", "MILD BEAR", "CHOPPY", "MILD BULL", "BULL"]
GUIDE = {"BEAR": "no new positions", "MILD BEAR": "half size, best setups only", "CHOPPY": "normal size, be selective",
         "MILD BULL": "full size", "BULL": "full size"}


def _t(x: str) -> str:
    return str(x).replace(".NS", "")


def build(df: pd.DataFrame, date: pd.Timestamp, nifty: Optional[pd.DataFrame] = None, funnel_new: Optional[pd.DataFrame] = None,
          shocks: Optional[List[Dict[str, Any]]] = None, rs: Optional[List[Dict[str, Any]]] = None, tc_count: int = 0,
          positions: Optional[List[Dict[str, Any]]] = None) -> str:
    from utils import book_view as bv, funnel, industry_board, regime_badge as rb
    L = [f"<b>📋 Daily Brief — {pd.Timestamp(date):%d %b %Y}</b>"]
    known = set(df["ticker"])
    by = df.drop_duplicates("ticker").set_index("ticker")

    # 1 market
    trail = 0.12
    try:
        info = rb.compute(nifty, df)
        n, b = info["nifty"].get("regime", "UNKNOWN"), info["breadth"].get("regime", "UNKNOWN")
        ok = [x for x in (n, b) if x in LADDER]
        weak = min(ok, key=LADDER.index) if ok else None
        trail = float(info["engine"].get("trail_stop") or trail)
        L.append(f"Market: Nifty <b>{n}</b>" + (f", breadth {info['breadth']['pct']:.0f}% ({b})" if b != "UNKNOWN" else "")
                 + (f" → {GUIDE[weak]}" if weak else ""))
    except Exception as e:
        L.append(f"Market: unavailable ({e})")

    # 2 + 3 your book
    if positions is None:
        try:
            from utils import positions as pos
            positions = pos.get_all_positions()
        except Exception:
            positions = []
    hold = [p for p in positions if p.get("status") == "active"]
    watch = [p for p in positions if p.get("status") == "watching"]
    res = {p["ticker"]: (bv.pos.resolve_ticker(p["ticker"], known) or p["ticker"]) for p in hold + watch}
    T = pd.DataFrame()
    board = None
    try:
        board = industry_board.board(df, industry_board.month_ago_snapshot(current_df=df))
        tk = [t for t in res.values() if t in known]
        if tk:
            T = funnel.table(df[df.ticker.isin(tk)], board, None, business_for=tk).set_index("ticker")
    except Exception:
        pass
    attn = []
    for p in hold:
        t = res[p["ticker"]]
        if t not in by.index:
            attn.append(f"  {_t(p['ticker'])} — no price found")
            continue
        r = by.loc[t]
        px = float(pd.to_numeric(r.get("currentPrice"), errors="coerce"))
        sl, tg = bv._num(p.get("stop_loss")), bv._num(p.get("target"))
        ma50 = pd.to_numeric(r.get("fiftyDayAverage"), errors="coerce")
        st_ = bv.status(px, sl, tg, str(r.get("dna_signal")), bool(pd.notna(ma50) and px < ma50),
                        float(pd.to_numeric(r.get("simba_caution"), errors="coerce") or 0) if pd.notna(r.get("simba_caution")) else 0)
        g = bv.exit_guide(px, float(ma50) if pd.notna(ma50) else None, bv.peak_since(t, p.get("entry_date"), px), trail, sl)
        e = bv._num(p.get("entry_price"))
        pnl = f" ({(px / e - 1) * 100:+.0f}%)" if e else ""
        if g["hit"]:
            attn.append(f"  🔴 {_t(p['ticker'])}{pnl} — exit rule hit: below ₹{g['level']:,.0f}")
        elif not st_.startswith("✅"):
            attn.append(f"  {st_.split(' ')[0]} {_t(p['ticker'])}{pnl} — {' '.join(st_.split(' ')[1:])}")
        elif g["text"].startswith("Raise"):
            attn.append(f"  ⬆️ {_t(p['ticker'])}{pnl} — {g['text'][0].lower() + g['text'][1:]}")
    L.append("")
    if hold:
        L.append(f"<b>📈 Holdings ({len(hold)})</b>: " + ("all on track" if not attn else f"{len(attn)} need a look"))
        L += attn
    else:
        L.append("📈 No holdings on file.")
    ready = [res[p["ticker"]] for p in watch if len(T) and res[p["ticker"]] in T.index
             and bv.funnel_label(T.loc[res[p["ticker"]]]).startswith("✅")]
    if watch:
        L.append(f"<b>👀 Watchlist ({len(watch)})</b>: " + (", ".join(_t(t) for t in ready) + " now buyable" if ready else "none buyable today"))

    # 4 funnel
    L.append("")
    new = funnel_new[funnel_new.kind == "PASS"] if funnel_new is not None and len(funnel_new) else pd.DataFrame()
    if len(new):
        L.append(f"<b>🧭 New funnel passes ({len(new)})</b>")
        for r in new.head(8).itertuples():
            L.append(f"  {_t(r.ticker)} — {r.industry}, ₹{r.signal_price:,.0f}" + (f", stop {r.risk_pct:.0f}% away" if pd.notna(r.risk_pct) else ""))
        if len(new) > 8:
            L.append(f"  …+{len(new) - 8} more")
    else:
        L.append("🧭 No new funnel passes today.")

    # 5 strong names in an uptrend
    buy = set(by.index[by.get("dna_signal", pd.Series(dtype=str)).astype(str) == "BUY"]) if "dna_signal" in by else set()
    es = [r for r in (shocks or []) if r.get("Ticker") in buy]
    if es:
        L.append(f"<b>⚡ Earnings jumps in an uptrend ({len(es)})</b>: "
                 + ", ".join(f"{_t(r['Ticker'])} +{r.get('Jump_Pct')}%" for r in es[:6]) + (" …" if len(es) > 6 else ""))
    rd = [r for r in (rs or []) if r.get("Ticker") in buy]
    if rd:
        L.append(f"<b>💪 Up on a red day, in an uptrend ({len(rd)})</b>: " + ", ".join(_t(r["Ticker"]) for r in rd[:8]) + (" …" if len(rd) > 8 else ""))

    # 6 the rest, as counts
    L.append("")
    other_parts = [f"{len(shocks or [])} earnings jumps", f"{len(rs or [])} red-day gainers"]
    if tc_count:
        other_parts.append(f"{tc_count} turnaround catalysts")
    L.append("Other scanners: " + ", ".join(other_parts) + " (full lists and their track record in Signal Lab).")
    return "\n".join(L)
