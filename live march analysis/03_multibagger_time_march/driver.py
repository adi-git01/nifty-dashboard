"""CLI driver for running Type 1 (Monthly-Gate) & Type 2 (Live 3-Agent) Multibagger Marches."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

DIR = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\multibagger_march"
sys.path.insert(0, DIR)
import sim_multibagger as sim

STATE_PATH = os.path.join(DIR, "books_state.json")
AGENTS = ["monthly_quant", "mom_trader", "garp_investor", "hybrid_pm"]


def fmt_packet(sealed: dict, dt_str: str) -> str:
    pkt = sealed["packets"][dt_str]
    m = pkt["market"]
    rec_s = f" | RECOVERY_ACTIVE: {'; '.join(m['recovery_reasons'])}" if m.get("recovery_active") else ""
    hdr = (
        f"AS_OF: {pkt['as_of']} | Nifty={m['nifty_close']} (PE {m['nifty_pe']}) | "
        f"Engine={m['engine_mode']} (base={m['base_mode']}){rec_s}\n"
        f"Breadth: >50DMA={m['breadth_50']}% | >200DMA={m['breadth_200']}% | 52W_Lows={m['lows_52w']}% | "
        f"VIX={m['vix']} | FII_Long%={m['fii_long_pct']}%"
    )
    rows = []
    for tk, d in pkt["stocks"].items():
        rows.append({
            "Ticker": tk.replace(".NS", ""),
            "Px": d["close"],
            "Stg": d["stage"],
            "RS": d["comp_rs"],
            "1M": d["rs_1m"],
            "G1": int(d["gate1"]),
            "State": (d["fx_state"] or "")[:10],
            "Act": d["fx_action"],
            "Guard": (d["fx_guard"] or "")[:18],
            "ExtATR": d["fx_ext_atr"],
            "Stop": d["fx_stop"],
            "Stp%": d["fx_stop_pct"],
            "BBP": d["fx_bbp"],
            "LiqCr": d["liq_cr"],
            "Dlv%": d["deliv_pct"],
            "AbsDlv": d["abs_deliv_cr"],
            "Q_NP%": d["q_np_yoy"],
            "Q_Sl%": d["q_sales_yoy"],
            "DIIq": d["d_dii_q"],
            "FIIq": d["d_fii_q"],
            "Val": (d["val_rating"] or "")[:18],
            "Bear": d["bear_blend"],
            "FCF/P": d["fcf_pat"],
            "CFO/P": d["cfo_pat"],
            "ROCE": d["roce"],
        })
    return hdr + "\n" + pd.DataFrame(rows).to_string(index=False)


def cmd_init():
    sealed = sim.load_sealed()
    st = {
        "start_date": sealed["dates"][0],
        "books": {ag: sim.blank_book(sealed["dates"][0]) for ag in AGENTS},
    }
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)
    print("Initialized 4 books at", sealed["dates"][0])
    print("\n" + fmt_packet(sealed, "2026-02-27"))


def cmd_step(orders_file: str):
    sealed = sim.load_sealed()
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        st = json.load(f)
    with open(orders_file, "r", encoding="utf-8") as f:
        spec = json.load(f)

    res = {}
    for ag, cfg in spec.items():
        r = sim.step_book(st["books"][ag], sealed, cfg["from_date"], cfg["to_date"], cfg.get("orders", {}), cfg.get("notes", {}))
        res[ag] = r
        print(f"=== {ag} ({cfg['from_date']} -> {cfg['to_date']}) === NAV=Rs {r['nav']:,.0f} ({r['cum_ret_pct']:+.2f}%) | Cash={r['cash_pct']:.1f}% | Pos={len(r['positions'])} | Events={len(r['window_events'])}")
        for ev in r["window_events"]:
            print("   ", ev)
        for tk, p in r["positions"].items():
            print(f"    HOLD {tk.replace('.NS',''):11s}: {p['shares']} sh @ {p['avg_px']} -> {p['close']} ({p['unrealized_pct']:+.1f}%, wt={p['weight_pct']}%, stop={p['stop_px']}, trail={p['trail_pct']})")

    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)

    # Print next packet(s) requested
    next_dts = sorted(set(cfg["to_date"] for cfg in spec.values() if cfg["to_date"] in sealed["packets"]))
    for dt in next_dts:
        print(f"\n--- NEXT PACKET: {dt} ---")
        print(fmt_packet(sealed, dt))


def cmd_finalize():
    sealed = sim.load_sealed()
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        st = json.load(f)
    out = {"agents": {}, "benchmarks": {}}
    for ag in AGENTS:
        out["agents"][ag] = sim.summarize_book(st["books"][ag], sealed)

    # Equal-weight 11-stock B&H benchmark & Nifty 50 benchmark
    dates = sealed["dates"]
    targets = sealed["targets"]
    ew_nav = pd.Series(0.0, index=dates)
    for tk in targets:
        s = pd.Series({r["date"]: r["close"] for r in sealed["daily_traces"][tk]}).reindex(dates)
        ew_nav += (sim.INIT_CAP / len(targets)) * (s / s.iloc[0])
    r = ew_nav.pct_change().dropna()
    ew_tot = (float(ew_nav.iloc[-1]) / float(ew_nav.iloc[0]) - 1.0) * 100.0
    ew_mdd = float((ew_nav / ew_nav.cummax() - 1.0).min() * 100.0)
    ew_sh = float((r.mean() / r.std(ddof=1)) * np.sqrt(252.0))
    out["benchmarks"]["BH_EW_11"] = {"final_nav": round(float(ew_nav.iloc[-1]), 2), "total_ret_pct": round(ew_tot, 2), "max_dd_pct": round(ew_mdd, 2), "sharpe": round(ew_sh, 3)}

    nf = pd.Series({d: sealed["daily_regimes"][d]["nifty_close"] for d in dates})
    nr = nf.pct_change().dropna()
    out["benchmarks"]["NIFTY_50"] = {
        "total_ret_pct": round((float(nf.iloc[-1]) / float(nf.iloc[0]) - 1.0) * 100.0, 2),
        "max_dd_pct": round(float((nf / nf.cummax() - 1.0).min() * 100.0), 2),
        "sharpe": round(float((nr.mean() / nr.std(ddof=1)) * np.sqrt(252.0)), 3),
    }

    with open(os.path.join(DIR, "final_multibagger_attribution.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== FINAL SCORECARD & VALIDATION METRICS ===")
    for ag, res in out["agents"].items():
        mae = res["mae_summary"]
        print(f"  {ag:15s}: NAV=Rs {res['final_nav']:10,.0f} ({res['total_ret_pct']:+6.2f}%) | MaxDD={res['max_dd_pct']:+6.2f}% | Sharpe={res['sharpe']:.3f} | "
              f"CashDrag={res['cash_drag_mean_pct']:.1f}% (by_reg={res['cash_drag_by_regime']}) | Fills={res['total_fills']} (W={res['win_exits']}/{res['closed_exits']}, Open={res['open_count']})")
        print(f"     MAE all={mae['all']} | <Bear&FCF>=0.8={mae['cmp_lt_bear_and_fcf_ge_08']} | <Bear={mae['cmp_lt_bear_only']} | other={mae['other']}")
    print("  BH_EW_11       :", out["benchmarks"]["BH_EW_11"])
    print("  NIFTY_50       :", out["benchmarks"]["NIFTY_50"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "step", "finalize"])
    ap.add_argument("--orders", default="")
    args = ap.parse_args()
    if args.cmd == "init":
        cmd_init()
    elif args.cmd == "step":
        cmd_step(args.orders)
    elif args.cmd == "finalize":
        cmd_finalize()
