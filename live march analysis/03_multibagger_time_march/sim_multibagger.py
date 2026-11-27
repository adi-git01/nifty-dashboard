"""Deterministic Daily Execution Simulator for Multibagger Time-Marches (Type 1 Monthly & Type 2 3-Agent).

Adds the two mandatory post-run validation metrics:
  1. Cash Drag % — mean & median daily uninvested cash % across all 181 sessions AND broken down by engine regime
     (BEAR, RECOVERY, CAUTION, BULL).
  2. Max Adverse Excursion (MAE) per tranche — worst intraday low & worst close drawdown from every BUY/ADD tranche,
     tagged with point-in-time `cmp_lt_bear` and `fcf_pat >= 0.8` + counterfactual `-25%` circuit-breaker audit!
"""
from __future__ import annotations

import json
import math
import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd

OUT_DIR = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\multibagger_march"
SEALED_PATH = os.path.join(OUT_DIR, "sealed_multibaggers.pkl")
COST_SIDE = 0.0020  # 20 bps/side
INIT_CAP = 1_000_000.0
MAX_POS_PCT = 20.0  # 20% single-stock cap across an 11-stock multibagger basket


def load_sealed() -> dict:
    with open(SEALED_PATH, "rb") as f:
        return pickle.load(f)


def _norm_trail(v) -> float | None:
    if v is None:
        return None
    fv = float(v)
    return fv / 100.0 if fv > 1.0 else fv


def blank_book(start_date: str) -> dict:
    return dict(
        cash=INIT_CAP,
        positions={},
        pending_triggers={},
        trades=[],
        tranches=[],      # per-tranche MAE & fundamental tag audit
        equity_curve=[{"date": start_date, "nav": INIT_CAP, "cash": INIT_CAP, "cash_pct": 100.0}],
        event_history=[],
    )


def _nav_at(book: dict, bar_map: Dict[str, dict], date: str) -> float:
    eq = float(book["cash"])
    for tk, pos in book["positions"].items():
        eq += pos["shares"] * float(bar_map[tk][date]["close"])
    return eq


def step_book(
    book: dict,
    sealed: dict,
    from_date: str,
    to_date: str,
    orders: Dict[str, dict],
    notes: dict | None = None,
) -> dict:
    targets = sealed["targets"]
    bar_map = {tk: {r["date"]: r for r in sealed["daily_traces"][tk]} for tk in targets}
    dates = [d for d in sealed["dates"] if from_date < d <= to_date]
    pkt_from = sealed["packets"].get(from_date, {}).get("stocks", {})

    next_open_actions: List[dict] = []
    book["pending_triggers"] = {}
    for tk, ord_spec in orders.items():
        if tk not in bar_map:
            continue
        act = str(ord_spec.get("action", "HOLD")).upper()
        tr_pct = _norm_trail(ord_spec.get("trail_pct"))
        spec = {**ord_spec, "action": act, "trail_pct": tr_pct}
        if act in ("BUY_NOW", "ADD", "TRIM", "EXIT"):
            next_open_actions.append({"ticker": tk, **spec})
        elif act == "PLACE_TRIGGER" and ord_spec.get("trigger_px") is not None:
            book["pending_triggers"][tk] = {
                "trigger_px": float(ord_spec["trigger_px"]),
                "target_pct": min(MAX_POS_PCT, float(ord_spec.get("target_pct", 12.0))),
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

    def _record_tranche(tk: str, dt: str, entry_px: float, shares: int, reason: str):
        sp = pkt_from.get(tk, {})
        bear_b = sp.get("bear_blend")
        fcf_p = sp.get("fcf_pat")
        cfo_p = sp.get("cfo_pat")
        cmp_lt_bear = bool(bear_b is not None and entry_px <= float(bear_b))
        book["tranches"].append({
            "id": len(book["tranches"]) + 1,
            "ticker": tk,
            "entry_date": dt,
            "entry_px": round(entry_px, 2),
            "shares": shares,
            "reason": reason,
            "bear_blend": bear_b,
            "cmp_lt_bear": cmp_lt_bear,
            "fcf_pat": fcf_p,
            "cfo_pat": cfo_p,
            "fcf_ge_08": bool(fcf_p is not None and float(fcf_p) >= 0.8),
            "min_low": round(entry_px, 2),
            "min_close": round(entry_px, 2),
            "max_high": round(entry_px, 2),
            "mae_low_pct": 0.0,
            "mae_close_pct": 0.0,
            "mfe_high_pct": 0.0,
            "hit_minus25": False,
            "hit_minus25_date": None,
            "closed_date": None,
            "exit_px": None,
        })

    def _close_tranches(tk: str, dt: str, exit_px: float):
        for tr in book["tranches"]:
            if tr["ticker"] == tk and tr["closed_date"] is None:
                tr["closed_date"] = dt
                tr["exit_px"] = round(exit_px, 2)

    for day_i, dt in enumerate(dates):
        # 1. Immediate next-open actions
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
                    _close_tranches(tk, dt, px_open)
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
                            _close_tranches(tk, dt, px_open)
                            del book["positions"][tk]
                elif act in ("BUY_NOW", "ADD"):
                    tgt_pct = min(MAX_POS_PCT, max(0.0, float(o.get("target_pct", 12.0))))
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
                        _record_tranche(tk, dt, px_open, sh, act)
                        ev = dict(date=dt, ticker=tk, side="BUY", reason=act, px=round(px_open, 2),
                                  shares=sh, value=round(gross, 0))
                        book["trades"].append(ev); window_events.append(ev)

        # 2. Queued close-confirmed exits
        for tk, rsn in list(pending_close_exits.items()):
            if tk in book["positions"]:
                pos = book["positions"].pop(tk)
                px_open = float(bar_map[tk][dt]["open"])
                net = pos["shares"] * px_open * (1.0 - COST_SIDE)
                book["cash"] += net
                pnl_pct = (px_open / pos["avg_px"] - 1.0) * 100.0
                _close_tranches(tk, dt, px_open)
                ev = dict(date=dt, ticker=tk, side="SELL", reason=rsn, px=round(px_open, 2),
                          shares=pos["shares"], pnl_pct=round(pnl_pct, 2), pnl_rs=round(net - pos["cost_basis"], 0))
                book["trades"].append(ev); window_events.append(ev)
        pending_close_exits.clear()

        # 3. Triggered entries
        for tk, trig in list(book["pending_triggers"].items()):
            if trig.get("triggered_next_open") and tk not in book["positions"]:
                px_open = float(bar_map[tk][dt]["open"])
                gap_pct = (px_open / trig["trigger_px"] - 1.0) * 100.0
                if gap_pct <= trig["max_gap_pct"]:
                    cur_eq = book["cash"] + sum(p["shares"] * float(bar_map[t_h][dt]["open"]) for t_h, p in book["positions"].items())
                    tgt_val = cur_eq * (min(MAX_POS_PCT, trig["target_pct"]) / 100.0)
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
                        _record_tranche(tk, dt, px_open, sh, f"TRIGGER@{trig['trigger_px']}")
                        ev = dict(date=dt, ticker=tk, side="BUY", reason=f"TRIGGER@{trig['trigger_px']}", px=round(px_open, 2),
                                  shares=sh, value=round(gross, 0))
                        book["trades"].append(ev); window_events.append(ev)
                else:
                    window_events.append(dict(date=dt, ticker=tk, side="SKIP", reason=f"GAP_GUARD({gap_pct:.1f}%>{trig['max_gap_pct']}%)", px=round(px_open, 2)))
                del book["pending_triggers"][tk]

        # 4. Update active tranche MAE/MFE + Intraday hard stops + EOD MA50/trail + trigger arming
        for tr in book["tranches"]:
            if tr["closed_date"] is None and tr["ticker"] in book["positions"]:
                bar = bar_map[tr["ticker"]][dt]
                lo, hi, cl = float(bar["low"]), float(bar["high"]), float(bar["close"])
                if lo < tr["min_low"]:
                    tr["min_low"] = round(lo, 2)
                    tr["mae_low_pct"] = round((lo / tr["entry_px"] - 1.0) * 100.0, 2)
                if cl < tr["min_close"]:
                    tr["min_close"] = round(cl, 2)
                    tr["mae_close_pct"] = round((cl / tr["entry_px"] - 1.0) * 100.0, 2)
                if hi > tr["max_high"]:
                    tr["max_high"] = round(hi, 2)
                    tr["mfe_high_pct"] = round((hi / tr["entry_px"] - 1.0) * 100.0, 2)
                if not tr["hit_minus25"] and tr["mae_low_pct"] <= -25.0:
                    tr["hit_minus25"] = True
                    tr["hit_minus25_date"] = dt

        for tk in list(book["positions"].keys()):
            pos = book["positions"][tk]
            bar = bar_map[tk][dt]
            if pos.get("stop_px") is not None and float(bar["low"]) <= float(pos["stop_px"]):
                fill_px = min(float(bar["open"]), float(pos["stop_px"]))
                net = pos["shares"] * fill_px * (1.0 - COST_SIDE)
                book["cash"] += net
                pnl_pct = (fill_px / pos["avg_px"] - 1.0) * 100.0
                _close_tranches(tk, dt, fill_px)
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

        nav_dt = _nav_at(book, bar_map, dt)
        book["equity_curve"].append({
            "date": dt,
            "nav": round(nav_dt, 2),
            "cash": round(book["cash"], 2),
            "cash_pct": round(book["cash"] / nav_dt * 100.0, 2) if nav_dt > 0 else 100.0,
        })

    book["event_history"].append(dict(from_date=from_date, to_date=to_date, orders=orders, events=window_events, notes=notes or {}))
    nav_now = _nav_at(book, bar_map, to_date)
    pos_mtm = {}
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


def summarize_book(book: dict, sealed: dict) -> dict:
    targets = sealed["targets"]
    end_dt = sealed["dates"][-1]
    bar_end = {tk: sealed["daily_traces"][tk][-1]["close"] for tk in targets}
    regimes = sealed["daily_regimes"]

    ec = pd.DataFrame(book["equity_curve"]).drop_duplicates("date", keep="last").set_index("date")
    nav = ec["nav"]
    r = nav.pct_change().dropna()
    tot = (float(nav.iloc[-1]) / float(nav.iloc[0]) - 1.0) * 100.0
    mdd = float((nav / nav.cummax() - 1.0).min() * 100.0)
    sd = float(r.std(ddof=1)) if len(r) > 2 else 0.0
    sh = float((r.mean() / sd) * math.sqrt(252.0)) if sd > 1e-12 else 0.0

    # 1. Cash Drag % (overall + by engine_mode regime)
    ec["engine_mode"] = [regimes.get(d, {}).get("engine_mode", "UNKNOWN") for d in ec.index]
    cash_drag_mean = round(float(ec["cash_pct"].mean()), 1)
    cash_drag_median = round(float(ec["cash_pct"].median()), 1)
    cash_by_regime = {
        m: round(float(g["cash_pct"].mean()), 1)
        for m, g in ec.groupby("engine_mode")
    }

    # 2. Tranche MAE summary (overall + split by cmp_lt_bear & fcf_ge_08)
    tranches = book["tranches"]
    for tr in tranches:
        if tr["closed_date"] is None:
            tr["final_ret_pct"] = round((bar_end[tr["ticker"]] / tr["entry_px"] - 1.0) * 100.0, 2)
        else:
            tr["final_ret_pct"] = round((tr["exit_px"] / tr["entry_px"] - 1.0) * 100.0, 2)
    def _mae_grp(sub):
        if not sub:
            return {"n": 0}
        return {
            "n": len(sub),
            "avg_mae_low_pct": round(float(np.mean([x["mae_low_pct"] for x in sub])), 2),
            "worst_mae_low_pct": round(float(np.min([x["mae_low_pct"] for x in sub])), 2),
            "avg_mae_close_pct": round(float(np.mean([x["mae_close_pct"] for x in sub])), 2),
            "hit_minus25_count": sum(1 for x in sub if x["hit_minus25"]),
            "avg_final_ret_pct": round(float(np.mean([x["final_ret_pct"] for x in sub])), 2),
        }
    mae_summary = {
        "all": _mae_grp(tranches),
        "cmp_lt_bear_and_fcf_ge_08": _mae_grp([t for t in tranches if t["cmp_lt_bear"] and t["fcf_ge_08"]]),
        "cmp_lt_bear_only": _mae_grp([t for t in tranches if t["cmp_lt_bear"]]),
        "other": _mae_grp([t for t in tranches if not t["cmp_lt_bear"]]),
    }

    sells = [t for t in book["trades"] if t["side"] == "SELL"]
    wins = [t for t in sells if t.get("pnl_rs", 0) > 0]
    per_stock = {}
    for tk in targets:
        real_rs = sum(t.get("pnl_rs", 0.0) for t in book["trades"] if t["ticker"] == tk and t["side"] == "SELL")
        unr_rs = (book["positions"][tk]["shares"] * bar_end[tk] - book["positions"][tk]["cost_basis"]) if tk in book["positions"] else 0.0
        per_stock[tk] = {
            "realized_rs": round(real_rs, 0),
            "unrealized_rs": round(unr_rs, 0),
            "total_pnl_rs": round(real_rs + unr_rs, 0),
            "n_buys": sum(1 for t in book["trades"] if t["ticker"] == tk and t["side"] == "BUY"),
            "n_sells": sum(1 for t in book["trades"] if t["ticker"] == tk and t["side"] == "SELL"),
        }

    return {
        "final_nav": round(float(nav.iloc[-1]), 2),
        "total_ret_pct": round(tot, 2),
        "max_dd_pct": round(mdd, 2),
        "sharpe": round(sh, 3),
        "cash_drag_mean_pct": cash_drag_mean,
        "cash_drag_median_pct": cash_drag_median,
        "cash_drag_by_regime": cash_by_regime,
        "mae_summary": mae_summary,
        "total_fills": len(book["trades"]),
        "closed_exits": len(sells),
        "win_exits": len(wins),
        "open_count": len(book["positions"]),
        "per_stock": per_stock,
        "tranches": tranches,
    }
