"""Multi-Agent 3-Book Deterministic Daily Execution Simulator for Parallel Blind Event-March."""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from typing import Dict, List

import pandas as pd

OUT_DIR = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\time_march_3agents"
SEALED_PATH = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\time_march\sealed_daily.pkl"
STATE_PATH = os.path.join(OUT_DIR, "books_state.json")
COST_SIDE = 0.0020  # 20 bps/side = 40 bps RT
INIT_CAP = 1_000_000.0
TARGETS = ["ULTRACEMCO.NS", "KALYANKJIL.NS"]
AGENTS = ["mom_trader", "garp_investor", "hybrid_pm"]


def _load_sealed() -> dict:
    with open(SEALED_PATH, "rb") as f:
        return pickle.load(f)


def _load_state() -> dict:
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(st: dict) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)


def _blank_book(start_date: str) -> dict:
    return dict(
        cash=INIT_CAP,
        positions={},
        pending_triggers={},
        trades=[],
        equity_curve=[{"date": start_date, "nav": INIT_CAP}],
        event_history=[],
    )


def init_books(start_date: str) -> dict:
    st = dict(
        start_date=start_date,
        as_of=start_date,
        initial_capital=INIT_CAP,
        books={ag: _blank_book(start_date) for ag in AGENTS},
    )
    _save_state(st)
    return st


def _nav_at(book: dict, bar_map: Dict[str, dict], date: str) -> float:
    eq = float(book["cash"])
    for tk, pos in book["positions"].items():
        eq += pos["shares"] * float(bar_map[tk][date]["close"])
    return eq


def _norm_trail(v) -> float | None:
    if v is None:
        return None
    fv = float(v)
    return fv / 100.0 if fv > 1.0 else fv


def _step_one_book(
    book: dict,
    bar_map: Dict[str, dict],
    dates: List[str],
    from_date: str,
    to_date: str,
    orders: Dict[str, dict],
    notes: dict | None = None,
) -> dict:
    next_open_actions: List[dict] = []
    book["pending_triggers"] = {}
    for tk, ord_spec in orders.items():
        act = str(ord_spec.get("action", "HOLD")).upper()
        tr_pct = _norm_trail(ord_spec.get("trail_pct"))
        spec = {**ord_spec, "action": act, "trail_pct": tr_pct}
        if act in ("BUY_NOW", "ADD", "TRIM", "EXIT"):
            next_open_actions.append({"ticker": tk, **spec})
        elif act == "PLACE_TRIGGER" and ord_spec.get("trigger_px") is not None:
            book["pending_triggers"][tk] = {
                "trigger_px": float(ord_spec["trigger_px"]),
                "target_pct": float(ord_spec.get("target_pct", 20.0)),
                "stop_px": float(ord_spec["stop_px"]) if ord_spec.get("stop_px") is not None else None,
                "trail_pct": tr_pct,
                "use_ma50_exit": bool(ord_spec.get("use_ma50_exit", False)),
                "max_gap_pct": float(ord_spec.get("max_gap_pct", 99.0)) if ord_spec.get("max_gap_pct") is not None else 99.0,
                "armed_on": from_date,
                "triggered_next_open": False,
            }
        if tk in book["positions"] and act in ("HOLD", "ADD", "TRIM"):
            if "stop_px" in ord_spec:
                book["positions"][tk]["stop_px"] = float(ord_spec["stop_px"]) if ord_spec["stop_px"] is not None else None
            if "trail_pct" in ord_spec:
                book["positions"][tk]["trail_pct"] = tr_pct
            if "use_ma50_exit" in ord_spec:
                book["positions"][tk]["use_ma50_exit"] = bool(ord_spec["use_ma50_exit"])

    window_events: List[dict] = []
    pending_close_exits: Dict[str, str] = {}

    for day_i, dt in enumerate(dates):
        # 1. Immediate next-open actions on first bar after event
        if day_i == 0 and next_open_actions:
            for o in sorted(next_open_actions, key=lambda x: 0 if x["action"] in ("EXIT", "TRIM") else 1):
                tk, act = o["ticker"], o["action"]
                px_open = float(bar_map[tk][dt]["open"])
                cur_eq = book["cash"] + sum(p["shares"] * float(bar_map[t_h][dt]["open"]) for t_h, p in book["positions"].items())
                if act == "EXIT" and tk in book["positions"]:
                    pos = book["positions"].pop(tk)
                    net = pos["shares"] * px_open * (1.0 - COST_SIDE)
                    book["cash"] += net
                    pnl_pct = (px_open / pos["avg_px"] - 1.0) * 100.0
                    ev = dict(date=dt, ticker=tk, side="SELL", reason="EVENT_EXIT", px=round(px_open, 2),
                              shares=pos["shares"], pnl_pct=round(pnl_pct, 2), pnl_rs=round(net - pos["cost_basis"], 0))
                    book["trades"].append(ev); window_events.append(ev)
                elif act == "TRIM" and tk in book["positions"]:
                    pos = book["positions"][tk]
                    tgt_val = cur_eq * (float(o.get("target_pct", 0.0)) / 100.0)
                    sell_val = max(0.0, pos["shares"] * px_open - tgt_val)
                    sell_sh = min(pos["shares"], int(sell_val / px_open))
                    if sell_sh > 0:
                        net = sell_sh * px_open * (1.0 - COST_SIDE)
                        book["cash"] += net
                        basis_sold = pos["cost_basis"] * (sell_sh / pos["shares"])
                        pos["shares"] -= sell_sh
                        pos["cost_basis"] -= basis_sold
                        pnl_pct = (px_open / pos["avg_px"] - 1.0) * 100.0
                        ev = dict(date=dt, ticker=tk, side="SELL", reason="EVENT_TRIM", px=round(px_open, 2),
                                  shares=sell_sh, pnl_pct=round(pnl_pct, 2), pnl_rs=round(net - basis_sold, 0))
                        book["trades"].append(ev); window_events.append(ev)
                        if pos["shares"] == 0:
                            del book["positions"][tk]
                elif act in ("BUY_NOW", "ADD"):
                    tgt_pct = min(50.0, max(0.0, float(o.get("target_pct", 20.0))))
                    tgt_val = cur_eq * (tgt_pct / 100.0)
                    cur_val = book["positions"][tk]["shares"] * px_open if tk in book["positions"] else 0.0
                    buy_val = min(max(0.0, tgt_val - cur_val), book["cash"] / (1.0 + COST_SIDE))
                    sh = int(buy_val / px_open)
                    if sh > 0:
                        gross = sh * px_open
                        cost = gross * (1.0 + COST_SIDE)
                        book["cash"] -= cost
                        if tk in book["positions"]:
                            p = book["positions"][tk]
                            p["shares"] += sh
                            p["cost_basis"] = round(p["cost_basis"] + cost, 2)
                            p["avg_px"] = round(p["cost_basis"] / p["shares"], 2)
                            p["peak_close"] = max(p["peak_close"], px_open)
                            if "stop_px" in o:
                                p["stop_px"] = float(o["stop_px"]) if o["stop_px"] is not None else None
                            if "trail_pct" in o:
                                p["trail_pct"] = o["trail_pct"]
                            if "use_ma50_exit" in o:
                                p["use_ma50_exit"] = bool(o["use_ma50_exit"])
                        else:
                            book["positions"][tk] = dict(
                                entry_date=dt, shares=sh, avg_px=round(cost / sh, 2), cost_basis=round(cost, 2),
                                peak_close=px_open,
                                stop_px=float(o["stop_px"]) if o.get("stop_px") is not None else None,
                                trail_pct=o.get("trail_pct"),
                                use_ma50_exit=bool(o.get("use_ma50_exit", False)),
                            )
                        ev = dict(date=dt, ticker=tk, side="BUY", reason=act, px=round(px_open, 2),
                                  shares=sh, value=round(gross, 0))
                        book["trades"].append(ev); window_events.append(ev)

        # 2. Queued close-confirmed exits (MA50 / trail)
        for tk, rsn in list(pending_close_exits.items()):
            if tk in book["positions"]:
                pos = book["positions"].pop(tk)
                px_open = float(bar_map[tk][dt]["open"])
                net = pos["shares"] * px_open * (1.0 - COST_SIDE)
                book["cash"] += net
                pnl_pct = (px_open / pos["avg_px"] - 1.0) * 100.0
                ev = dict(date=dt, ticker=tk, side="SELL", reason=rsn, px=round(px_open, 2),
                          shares=pos["shares"], pnl_pct=round(pnl_pct, 2), pnl_rs=round(net - pos["cost_basis"], 0))
                book["trades"].append(ev); window_events.append(ev)
        pending_close_exits.clear()

        # 3. Triggered entries (Close >= trigger_px yesterday -> fill today open, respecting optional max_gap_pct)
        for tk, trig in list(book["pending_triggers"].items()):
            if trig.get("triggered_next_open") and tk not in book["positions"]:
                px_open = float(bar_map[tk][dt]["open"])
                gap_pct = (px_open / trig["trigger_px"] - 1.0) * 100.0
                if gap_pct <= trig["max_gap_pct"]:
                    cur_eq = book["cash"] + sum(p["shares"] * float(bar_map[t_h][dt]["open"]) for t_h, p in book["positions"].items())
                    tgt_val = cur_eq * (min(50.0, trig["target_pct"]) / 100.0)
                    buy_val = min(tgt_val, book["cash"] / (1.0 + COST_SIDE))
                    sh = int(buy_val / px_open)
                    if sh > 0:
                        gross = sh * px_open
                        cost = gross * (1.0 + COST_SIDE)
                        book["cash"] -= cost
                        book["positions"][tk] = dict(
                            entry_date=dt, shares=sh, avg_px=round(cost / sh, 2), cost_basis=round(cost, 2),
                            peak_close=px_open, stop_px=trig.get("stop_px"), trail_pct=trig.get("trail_pct"),
                            use_ma50_exit=bool(trig.get("use_ma50_exit", False)),
                        )
                        ev = dict(date=dt, ticker=tk, side="BUY", reason=f"TRIGGER@{trig['trigger_px']}", px=round(px_open, 2),
                                  shares=sh, value=round(gross, 0))
                        book["trades"].append(ev); window_events.append(ev)
                else:
                    window_events.append(dict(date=dt, ticker=tk, side="SKIP", reason=f"GAP_GUARD({gap_pct:.1f}%>{trig['max_gap_pct']}%)", px=round(px_open, 2)))
                del book["pending_triggers"][tk]

        # 4. Intraday hard stops + EOD MA50/trail + trigger arming
        for tk in list(book["positions"].keys()):
            pos = book["positions"][tk]
            bar = bar_map[tk][dt]
            if pos.get("stop_px") is not None and float(bar["low"]) <= float(pos["stop_px"]):
                fill_px = min(float(bar["open"]), float(pos["stop_px"]))
                net = pos["shares"] * fill_px * (1.0 - COST_SIDE)
                book["cash"] += net
                pnl_pct = (fill_px / pos["avg_px"] - 1.0) * 100.0
                ev = dict(date=dt, ticker=tk, side="SELL", reason=f"HARD_STOP@{pos['stop_px']}", px=round(fill_px, 2),
                          shares=pos["shares"], pnl_pct=round(pnl_pct, 2), pnl_rs=round(net - pos["cost_basis"], 0))
                book["trades"].append(ev); window_events.append(ev)
                del book["positions"][tk]
                continue
            cl = float(bar["close"])
            if cl > pos["peak_close"]:
                pos["peak_close"] = cl
            if pos.get("use_ma50_exit") and bar.get("ma50") is not None and cl < float(bar["ma50"]):
                pending_close_exits[tk] = f"MA50_BREAK({cl:.1f}<{bar['ma50']:.1f})"
            elif pos.get("trail_pct") is not None and cl < pos["peak_close"] * (1.0 - float(pos["trail_pct"])):
                pending_close_exits[tk] = f"TRAIL_{int(round(pos['trail_pct']*100))}%_HIT"

        for tk, trig in book["pending_triggers"].items():
            if tk not in book["positions"] and not trig.get("triggered_next_open"):
                if float(bar_map[tk][dt]["close"]) >= float(trig["trigger_px"]):
                    trig["triggered_next_open"] = True

        book["equity_curve"].append({"date": dt, "nav": round(_nav_at(book, bar_map, dt), 2)})

    book["event_history"].append(dict(from_date=from_date, to_date=to_date, orders=orders, events=window_events, notes=notes or {}))

    pos_mtm = {}
    nav_now = _nav_at(book, bar_map, to_date)
    for tk, pos in book["positions"].items():
        cl = float(bar_map[tk][to_date]["close"])
        val = pos["shares"] * cl
        pos_mtm[tk] = dict(
            shares=pos["shares"], avg_px=pos["avg_px"], close=cl, value=round(val, 0),
            weight_pct=round(val / nav_now * 100.0, 1),
            unrealized_pct=round((cl / pos["avg_px"] - 1.0) * 100.0, 2),
            stop_px=pos.get("stop_px"), trail_pct=pos.get("trail_pct"), use_ma50_exit=pos.get("use_ma50_exit"),
        )
    return dict(
        as_of=to_date, nav=round(nav_now, 2), cum_ret_pct=round((nav_now / INIT_CAP - 1.0) * 100.0, 2),
        cash=round(book["cash"], 2), cash_pct=round(book["cash"] / nav_now * 100.0, 1),
        positions=pos_mtm, window_events=window_events,
    )


def step_all(from_date: str, to_date: str, orders_by_agent: Dict[str, dict]) -> dict:
    sealed = _load_sealed()
    st = _load_state()
    bar_map = {tk: {r["date"]: r for r in sealed["daily_traces"][tk]} for tk in TARGETS}
    dates = [r["date"] for r in sealed["daily_traces"][TARGETS[0]] if from_date < r["date"] <= to_date]
    out = {}
    for ag in AGENTS:
        spec = orders_by_agent.get(ag, {})
        out[ag] = _step_one_book(st["books"][ag], bar_map, dates, from_date, to_date, spec.get("orders", {}), spec.get("notes", {}))
    st["as_of"] = to_date
    _save_state(st)
    return out


def _nav_metrics(nav_s: pd.Series) -> dict:
    r = nav_s.pct_change().dropna()
    tot = (float(nav_s.iloc[-1]) / float(nav_s.iloc[0]) - 1.0) * 100.0
    mdd = float((nav_s / nav_s.cummax() - 1.0).min() * 100.0)
    sd = float(r.std(ddof=1)) if len(r) > 2 else 0.0
    sh = float((r.mean() / sd) * math.sqrt(252.0)) if sd > 1e-12 else 0.0
    return dict(total_ret_pct=round(tot, 2), max_dd_pct=round(mdd, 2), sharpe=round(sh, 3))


def finalize_all() -> dict:
    sealed = _load_sealed()
    st = _load_state()
    start_dt, end_dt = st["start_date"], st["as_of"]
    bar_end = {tk: {r["date"]: r for r in sealed["daily_traces"][tk]}[end_dt]["close"] for tk in TARGETS}

    agent_res = {}
    for ag in AGENTS:
        b = st["books"][ag]
        ec = pd.DataFrame(b["equity_curve"]).drop_duplicates("date", keep="last").set_index("date")["nav"]
        m = _nav_metrics(ec)
        sells = [t for t in b["trades"] if t["side"] == "SELL"]
        wins = [t for t in sells if t.get("pnl_rs", 0) > 0]
        per_stock = {}
        for tk in TARGETS:
            real_rs = sum(t.get("pnl_rs", 0.0) for t in b["trades"] if t["ticker"] == tk and t["side"] == "SELL")
            unr_rs = 0.0
            if tk in b["positions"]:
                p = b["positions"][tk]
                unr_rs = p["shares"] * bar_end[tk] - p["cost_basis"]
            per_stock[tk] = dict(
                realized_rs=round(real_rs, 0), unrealized_rs=round(unr_rs, 0),
                total_pnl_rs=round(real_rs + unr_rs, 0),
                n_buys=sum(1 for t in b["trades"] if t["ticker"] == tk and t["side"] == "BUY"),
                n_sells=sum(1 for t in b["trades"] if t["ticker"] == tk and t["side"] == "SELL"),
            )
        final_nav = float(ec.iloc[-1])
        agent_res[ag] = dict(
            **m, final_nav=round(final_nav, 2), total_pnl_rs=round(final_nav - INIT_CAP, 0),
            total_fills=len(b["trades"]), closed_exits=len(sells), win_exits=len(wins),
            open_positions=b["positions"], per_stock=per_stock, trades=b["trades"],
        )

    # Benchmarks
    bms = {}
    bh_s = {}
    for tk in TARGETS:
        sub = [r for r in sealed["daily_traces"][tk] if start_dt <= r["date"] <= end_dt]
        s = pd.Series([r["close"] for r in sub], index=[r["date"] for r in sub])
        bh_s[tk] = s / s.iloc[0]
        bms[f"BH_{tk[:-3]}"] = {**_nav_metrics(s), "start_close": sub[0]["close"], "end_close": sub[-1]["close"]}
    bms["BH_50_50"] = _nav_metrics(0.5 * bh_s[TARGETS[0]] + 0.5 * bh_s[TARGETS[1]])
    n_map = sealed["nifty"]
    n_s = pd.Series({d: n_map[d] for d in sorted(n_map) if start_dt <= d <= end_dt})
    bms["NIFTY_50"] = {**_nav_metrics(n_s), "start_close": float(n_s.iloc[0]), "end_close": float(n_s.iloc[-1])}

    rep = dict(window=[start_dt, end_dt], agents=agent_res, benchmarks=bms)
    with open(os.path.join(OUT_DIR, "final_3agent_attribution.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_i = sub.add_parser("init"); p_i.add_argument("--start", required=True)
    p_s = sub.add_parser("step"); p_s.add_argument("--from", dest="f", required=True); p_s.add_argument("--to", dest="t", required=True); p_s.add_argument("--orders-file", required=True)
    sub.add_parser("finalize")
    args = ap.parse_args()
    if args.cmd == "init":
        init_books(args.start)
        print(json.dumps({"status": "initialized_3_books", "start": args.start, "agents": AGENTS}))
    elif args.cmd == "step":
        payload = json.load(open(args.orders_file, encoding="utf-8"))
        print(json.dumps(step_all(args.f, args.t, payload), indent=2))
    elif args.cmd == "finalize":
        print(json.dumps(finalize_all(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
