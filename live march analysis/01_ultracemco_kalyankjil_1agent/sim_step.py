"""Deterministic daily order/stop/trigger execution simulator for the blind ULTRACEMCO & KALYANKJIL time-march."""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from typing import Any, Dict, List

import numpy as np
import pandas as pd

OUT_DIR = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\time_march"
SEALED_PATH = os.path.join(OUT_DIR, "sealed_daily.pkl")
STATE_PATH = os.path.join(OUT_DIR, "book_state.json")
COST_SIDE = 0.0020  # 20 bps per side = 40 bps RT
INIT_CAP = 1_000_000.0
TARGETS = ["ULTRACEMCO.NS", "KALYANKJIL.NS"]


def _load_sealed() -> dict:
    with open(SEALED_PATH, "rb") as f:
        return pickle.load(f)


def _save_state(st: dict) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)


def _load_state() -> dict:
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def init_book(start_date: str) -> dict:
    st = dict(
        start_date=start_date,
        as_of=start_date,
        initial_capital=INIT_CAP,
        cash=INIT_CAP,
        positions={},
        pending_triggers={},
        trades=[],
        equity_curve=[{"date": start_date, "nav": INIT_CAP}],
        gate_history=[],
    )
    _save_state(st)
    return st


def _nav_at(st: dict, bar_map: Dict[str, dict], date: str) -> float:
    eq = float(st["cash"])
    for tk, pos in st["positions"].items():
        px = float(bar_map[tk][date]["close"])
        eq += pos["shares"] * px
    return eq


def step_book(from_date: str, to_date: str, orders: Dict[str, dict], notes: dict | None = None) -> dict:
    """
    Execute Trader's decisions placed at `from_date` EOD strictly across daily bars (from_date, to_date].
    - BUY_NOW / ADD / TRIM / EXIT fill at Open of the first trading day > from_date.
    - PLACE_TRIGGER waits for Close >= trigger_px, then fills at Next Open.
    - Hard stops (`stop_px` on Low <= stop_px, filled at min(Open, stop_px)), optional MA50 close exit
      (next open after Close < MA50), and trailing stop % from peak close are evaluated every day.
    """
    sealed = _load_sealed()
    st = _load_state()
    traces = sealed["daily_traces"]
    bar_map = {tk: {r["date"]: r for r in traces[tk]} for tk in TARGETS}
    dates = [r["date"] for r in traces[TARGETS[0]] if from_date < r["date"] <= to_date]

    # Queue immediate EOD orders for next-day open + update triggers/stops
    next_open_actions: List[dict] = []
    st["pending_triggers"] = {}
    for tk, ord_spec in orders.items():
        act = str(ord_spec.get("action", "HOLD")).upper()
        if act in ("BUY_NOW", "ADD", "TRIM", "EXIT"):
            next_open_actions.append({"ticker": tk, **ord_spec, "action": act})
        elif act == "PLACE_TRIGGER":
            st["pending_triggers"][tk] = {
                "trigger_px": float(ord_spec["trigger_px"]),
                "target_pct": float(ord_spec.get("target_pct", 25.0)),
                "stop_px": ord_spec.get("stop_px"),
                "trail_pct": ord_spec.get("trail_pct"),
                "use_ma50_exit": bool(ord_spec.get("use_ma50_exit", False)),
                "armed_on": from_date,
                "triggered_next_open": False,
            }
        # Update risk parameters on existing positions if provided (e.g., raising stop on HOLD)
        if tk in st["positions"] and act in ("HOLD", "ADD", "TRIM"):
            if ord_spec.get("stop_px") is not None:
                st["positions"][tk]["stop_px"] = float(ord_spec["stop_px"])
            if ord_spec.get("trail_pct") is not None:
                st["positions"][tk]["trail_pct"] = float(ord_spec["trail_pct"])
            if "use_ma50_exit" in ord_spec:
                st["positions"][tk]["use_ma50_exit"] = bool(ord_spec["use_ma50_exit"])

    window_events: List[dict] = []
    pending_ma50_or_trail_exit: Dict[str, str] = {}

    for day_i, dt in enumerate(dates):
        # 1. Execute queued next-open orders (initial gate orders + triggered orders + MA50/trail exits)
        if day_i == 0 and next_open_actions:
            # Exits/trims first to free cash
            for o in sorted(next_open_actions, key=lambda x: 0 if x["action"] in ("EXIT", "TRIM") else 1):
                tk, act = o["ticker"], o["action"]
                px_open = float(bar_map[tk][dt]["open"])
                cur_eq = st["cash"] + sum(p["shares"] * float(bar_map[t_h][dt]["open"]) for t_h, p in st["positions"].items())
                if act == "EXIT" and tk in st["positions"]:
                    pos = st["positions"].pop(tk)
                    gross = pos["shares"] * px_open
                    net = gross * (1.0 - COST_SIDE)
                    st["cash"] += net
                    pnl_pct = (px_open / pos["avg_px"] - 1.0) * 100.0
                    ev = dict(date=dt, ticker=tk, side="SELL", reason="GATE_EXIT", px=round(px_open, 2),
                              shares=pos["shares"], pnl_pct=round(pnl_pct, 2), pnl_rs=round(net - pos["cost_basis"], 0))
                    st["trades"].append(ev); window_events.append(ev)
                elif act == "TRIM" and tk in st["positions"]:
                    pos = st["positions"][tk]
                    tgt_val = cur_eq * (float(o.get("target_pct", 0.0)) / 100.0)
                    cur_val = pos["shares"] * px_open
                    sell_val = max(0.0, cur_val - tgt_val)
                    sell_sh = min(pos["shares"], int(sell_val / px_open))
                    if sell_sh > 0:
                        gross = sell_sh * px_open
                        net = gross * (1.0 - COST_SIDE)
                        st["cash"] += net
                        basis_sold = pos["cost_basis"] * (sell_sh / pos["shares"])
                        pos["shares"] -= sell_sh
                        pos["cost_basis"] -= basis_sold
                        pnl_pct = (px_open / pos["avg_px"] - 1.0) * 100.0
                        ev = dict(date=dt, ticker=tk, side="SELL", reason="GATE_TRIM", px=round(px_open, 2),
                                  shares=sell_sh, pnl_pct=round(pnl_pct, 2), pnl_rs=round(net - basis_sold, 0))
                        st["trades"].append(ev); window_events.append(ev)
                        if pos["shares"] == 0:
                            del st["positions"][tk]
                elif act in ("BUY_NOW", "ADD"):
                    tgt_val = cur_eq * (float(o.get("target_pct", 25.0)) / 100.0)
                    cur_val = st["positions"][tk]["shares"] * px_open if tk in st["positions"] else 0.0
                    buy_val = min(max(0.0, tgt_val - cur_val), st["cash"] / (1.0 + COST_SIDE))
                    sh = int(buy_val / px_open)
                    if sh > 0:
                        gross = sh * px_open
                        cost = gross * (1.0 + COST_SIDE)
                        st["cash"] -= cost
                        if tk in st["positions"]:
                            p = st["positions"][tk]
                            p["shares"] += sh
                            p["cost_basis"] += cost
                            p["avg_px"] = round(p["cost_basis"] / p["shares"], 2)
                            p["peak_close"] = max(p["peak_close"], px_open)
                            if o.get("stop_px") is not None:
                                p["stop_px"] = float(o["stop_px"])
                            if o.get("trail_pct") is not None:
                                p["trail_pct"] = float(o["trail_pct"])
                            if "use_ma50_exit" in o:
                                p["use_ma50_exit"] = bool(o["use_ma50_exit"])
                        else:
                            st["positions"][tk] = dict(
                                entry_date=dt, shares=sh, avg_px=round(cost / sh, 2), cost_basis=round(cost, 2),
                                peak_close=px_open,
                                stop_px=float(o["stop_px"]) if o.get("stop_px") is not None else None,
                                trail_pct=float(o["trail_pct"]) if o.get("trail_pct") is not None else None,
                                use_ma50_exit=bool(o.get("use_ma50_exit", False)),
                            )
                        ev = dict(date=dt, ticker=tk, side="BUY", reason=act, px=round(px_open, 2),
                                  shares=sh, value=round(gross, 0))
                        st["trades"].append(ev); window_events.append(ev)

        # Queued close-confirmed exits (MA50 / trail) fill at today's open
        for tk, rsn in list(pending_ma50_or_trail_exit.items()):
            if tk in st["positions"]:
                pos = st["positions"].pop(tk)
                px_open = float(bar_map[tk][dt]["open"])
                gross = pos["shares"] * px_open
                net = gross * (1.0 - COST_SIDE)
                st["cash"] += net
                pnl_pct = (px_open / pos["avg_px"] - 1.0) * 100.0
                ev = dict(date=dt, ticker=tk, side="SELL", reason=rsn, px=round(px_open, 2),
                          shares=pos["shares"], pnl_pct=round(pnl_pct, 2), pnl_rs=round(net - pos["cost_basis"], 0))
                st["trades"].append(ev); window_events.append(ev)
        pending_ma50_or_trail_exit.clear()

        # Triggered entries (close >= trigger_px yesterday) fill at today's open
        for tk, trig in list(st["pending_triggers"].items()):
            if trig.get("triggered_next_open") and tk not in st["positions"]:
                px_open = float(bar_map[tk][dt]["open"])
                cur_eq = st["cash"] + sum(p["shares"] * float(bar_map[t_h][dt]["open"]) for t_h, p in st["positions"].items())
                tgt_val = cur_eq * (trig["target_pct"] / 100.0)
                buy_val = min(tgt_val, st["cash"] / (1.0 + COST_SIDE))
                sh = int(buy_val / px_open)
                if sh > 0:
                    gross = sh * px_open
                    cost = gross * (1.0 + COST_SIDE)
                    st["cash"] -= cost
                    st["positions"][tk] = dict(
                        entry_date=dt, shares=sh, avg_px=round(cost / sh, 2), cost_basis=round(cost, 2),
                        peak_close=px_open,
                        stop_px=float(trig["stop_px"]) if trig.get("stop_px") is not None else None,
                        trail_pct=float(trig["trail_pct"]) if trig.get("trail_pct") is not None else None,
                        use_ma50_exit=bool(trig.get("use_ma50_exit", False)),
                    )
                    ev = dict(date=dt, ticker=tk, side="BUY", reason=f"TRIGGER@{trig['trigger_px']}", px=round(px_open, 2),
                              shares=sh, value=round(gross, 0))
                    st["trades"].append(ev); window_events.append(ev)
                del st["pending_triggers"][tk]

        # 2. Intraday hard stop check + EOD MA50/trail checks & trigger arming
        for tk in list(st["positions"].keys()):
            pos = st["positions"][tk]
            bar = bar_map[tk][dt]
            if pos.get("stop_px") is not None and float(bar["low"]) <= float(pos["stop_px"]):
                fill_px = min(float(bar["open"]), float(pos["stop_px"]))
                gross = pos["shares"] * fill_px
                net = gross * (1.0 - COST_SIDE)
                st["cash"] += net
                pnl_pct = (fill_px / pos["avg_px"] - 1.0) * 100.0
                ev = dict(date=dt, ticker=tk, side="SELL", reason=f"HARD_STOP@{pos['stop_px']}", px=round(fill_px, 2),
                          shares=pos["shares"], pnl_pct=round(pnl_pct, 2), pnl_rs=round(net - pos["cost_basis"], 0))
                st["trades"].append(ev); window_events.append(ev)
                del st["positions"][tk]
                continue
            cl = float(bar["close"])
            if cl > pos["peak_close"]:
                pos["peak_close"] = cl
            if pos.get("use_ma50_exit") and bar.get("ma50") is not None and cl < float(bar["ma50"]):
                pending_ma50_or_trail_exit[tk] = f"MA50_BREAK({cl:.1f}<{bar['ma50']:.1f})"
            elif pos.get("trail_pct") is not None and cl < pos["peak_close"] * (1.0 - float(pos["trail_pct"])):
                pending_ma50_or_trail_exit[tk] = f"TRAIL_{int(pos['trail_pct']*100)}%_HIT"

        for tk, trig in st["pending_triggers"].items():
            if tk not in st["positions"] and not trig.get("triggered_next_open"):
                if float(bar_map[tk][dt]["close"]) >= float(trig["trigger_px"]):
                    trig["triggered_next_open"] = True

        st["equity_curve"].append({"date": dt, "nav": round(_nav_at(st, bar_map, dt), 2)})

    st["as_of"] = to_date
    st["gate_history"].append(dict(from_date=from_date, to_date=to_date, orders=orders, events=window_events, notes=notes or {}))
    _save_state(st)

    # Return summary strictly as of to_date
    pos_mtm = {}
    for tk, pos in st["positions"].items():
        cl = float(bar_map[tk][to_date]["close"])
        val = pos["shares"] * cl
        pos_mtm[tk] = dict(
            shares=pos["shares"], avg_px=pos["avg_px"], close=cl, value=round(val, 0),
            unrealized_pct=round((cl / pos["avg_px"] - 1.0) * 100.0, 2),
            stop_px=pos.get("stop_px"), trail_pct=pos.get("trail_pct"), use_ma50_exit=pos.get("use_ma50_exit"),
            peak_close=round(pos["peak_close"], 2),
        )
    nav_now = _nav_at(st, bar_map, to_date)
    return dict(
        as_of=to_date,
        nav=round(nav_now, 2),
        cum_ret_pct=round((nav_now / INIT_CAP - 1.0) * 100.0, 2),
        cash=round(st["cash"], 2),
        positions=pos_mtm,
        window_events=window_events,
    )


def _nav_metrics(nav_series: pd.Series) -> dict:
    r = nav_series.pct_change().dropna()
    tot = (float(nav_series.iloc[-1]) / float(nav_series.iloc[0]) - 1.0) * 100.0
    mdd = float((nav_series / nav_series.cummax() - 1.0).min() * 100.0)
    sd = float(r.std(ddof=1)) if len(r) > 2 else 0.0
    sh = float((r.mean() / sd) * math.sqrt(252.0)) if sd > 1e-12 else 0.0
    return dict(total_ret_pct=round(tot, 2), max_dd_pct=round(mdd, 2), sharpe=round(sh, 3))


def _mechanical_dashboard_rule(traces_tk: List[dict], start_date: str) -> dict:
    """Pure mechanical rule: BUY next open on Gate 1 + (BUY NOW or ABSORPTION); exit next open on MA50 break or 15% trail."""
    rows = [r for r in traces_tk if r["date"] >= start_date]
    cash = 1_000_000.0
    shares = 0
    peak = 0.0
    pend_buy, pend_sell = False, False
    navs = []
    n_trades = 0
    for r in rows:
        op, cl = float(r["open"]), float(r["close"])
        if pend_sell and shares > 0:
            cash += shares * op * (1.0 - COST_SIDE)
            shares = 0; peak = 0.0; pend_sell = False; n_trades += 1
        if pend_buy and shares == 0:
            alloc = cash / (1.0 + COST_SIDE)
            shares = int(alloc / op)
            if shares > 0:
                cash -= shares * op * (1.0 + COST_SIDE)
                peak = cl; n_trades += 1
            pend_buy = False
        if shares > 0:
            peak = max(peak, cl)
            if (r["ma50"] is not None and cl < float(r["ma50"])) or cl < peak * 0.85:
                pend_sell = True
        elif r["gate1"] and (r["fx_action"] == "BUY NOW" or r["fx_signature"] == "ABSORPTION"):
            pend_buy = True
        navs.append(cash + shares * cl)
    s = pd.Series(navs, index=[r["date"] for r in rows])
    return {**_nav_metrics(s), "trades": n_trades}


def finalize_book() -> dict:
    sealed = _load_sealed()
    st = _load_state()
    traces = sealed["daily_traces"]
    start_dt = st["start_date"]
    end_dt = st["as_of"]

    ec = pd.DataFrame(st["equity_curve"]).drop_duplicates("date", keep="last").set_index("date")["nav"]
    trader_m = _nav_metrics(ec)

    # Benchmarks over exact same [start_dt, end_dt]
    bms = {}
    bh_series = {}
    for tk in TARGETS:
        sub = [r for r in traces[tk] if start_dt <= r["date"] <= end_dt]
        s = pd.Series([r["close"] for r in sub], index=[r["date"] for r in sub])
        bh_series[tk] = s / s.iloc[0]
        bms[f"BH_{tk[:-3]}"] = {
            **_nav_metrics(s),
            "start_close": sub[0]["close"],
            "end_close": sub[-1]["close"],
            "min_close": min(r["close"] for r in sub),
            "max_close": max(r["close"] for r in sub),
        }
        bms[f"MECH_G1_BUYNOW_{tk[:-3]}"] = _mechanical_dashboard_rule(sub, start_dt)

    bh50 = 0.5 * bh_series[TARGETS[0]] + 0.5 * bh_series[TARGETS[1]]
    bms["BH_50_50"] = _nav_metrics(bh50)

    n_map = sealed["nifty"]
    n_s = pd.Series({d: n_map[d] for d in sorted(n_map) if start_dt <= d <= end_dt})
    bms["NIFTY_50"] = {**_nav_metrics(n_s), "start_close": float(n_s.iloc[0]), "end_close": float(n_s.iloc[-1])}

    # Per-stock attribution (realized + unrealized MTM)
    bar_end = {tk: {r["date"]: r for r in traces[tk]}[end_dt]["close"] for tk in TARGETS}
    per_stock = {}
    for tk in TARGETS:
        realized_rs = sum(t.get("pnl_rs", 0.0) for t in st["trades"] if t["ticker"] == tk and t["side"] == "SELL")
        unreal_rs = 0.0
        if tk in st["positions"]:
            p = st["positions"][tk]
            unreal_rs = p["shares"] * bar_end[tk] - p["cost_basis"]
        per_stock[tk] = dict(
            realized_rs=round(realized_rs, 0),
            unrealized_rs=round(unreal_rs, 0),
            total_pnl_rs=round(realized_rs + unreal_rs, 0),
            n_buys=sum(1 for t in st["trades"] if t["ticker"] == tk and t["side"] == "BUY"),
            n_sells=sum(1 for t in st["trades"] if t["ticker"] == tk and t["side"] == "SELL"),
        )

    out = dict(
        window=[start_dt, end_dt],
        trader_summary={**trader_m, "final_nav": round(float(ec.iloc[-1]), 2), "total_pnl_rs": round(float(ec.iloc[-1]) - INIT_CAP, 0), "total_trades": len(st["trades"])},
        per_stock=per_stock,
        benchmarks=bms,
        trades=st["trades"],
        open_positions_at_end=st["positions"],
    )
    with open(os.path.join(OUT_DIR, "final_attribution.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_init = sub.add_parser("init")
    p_init.add_argument("--start", required=True)
    p_step = sub.add_parser("step")
    p_step.add_argument("--from", dest="from_date", required=True)
    p_step.add_argument("--to", dest="to_date", required=True)
    p_step.add_argument("--orders-file", required=True)
    sub.add_parser("finalize")
    args = ap.parse_args()

    if args.cmd == "init":
        init_book(args.start)
        print(json.dumps({"status": "initialized", "start": args.start, "cash": INIT_CAP}))
    elif args.cmd == "step":
        with open(args.orders_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        res = step_book(args.from_date, args.to_date, payload["orders"], payload.get("notes"))
        print(json.dumps(res, indent=2))
    elif args.cmd == "finalize":
        res = finalize_book()
        print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
