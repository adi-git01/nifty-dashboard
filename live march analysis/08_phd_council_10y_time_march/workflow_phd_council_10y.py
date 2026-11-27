"""10.5-Year Strictly Causal, Fully Blinded 3-PhD Expert Council Time-March Workflow (2016-06 -> 2026-11).

Pure Python stdlib (zero numpy/pandas dependency inside workflow runtime!).
Orchestrates 22 semi-annual/regime-aligned epochs across 10.5 years of survivorship-free NSE data
(PIT MCap >= 500 Cr & 20D Median Liquidity >= 2 Cr/d, Max 15 Stocks).
At EVERY epoch E01..E22:
  - Round 1 (parallel): 3 PhD Specialists independently audit holdings & top multi-style candidates.
  - Round 2 (parallel): All 3 PhDs cross-examine each other's Round-1 theses & cast binding votes.
  - Portfolio Desk: Applies 2-of-3 majority on all Entries & Exits (max 15 stocks), executes at
    next-day T+1 Open (0.25% one-way friction), and marches daily OHLCV with PhD-voted stops to E(k+1).
ZERO ticker names and ZERO calendar dates are ever exposed to the 3 PhDs!
"""
import datetime
import json
import math
import os
import pickle
import statistics

OUT_DIR = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\phd_council_10y"
PANEL_PKL = os.path.join(OUT_DIR, "blind_10y_stdlib.pkl")
RESULTS_JSON = os.path.join(OUT_DIR, "phd_council_10y_results.json")
CHECKPOINT_JSON = os.path.join(OUT_DIR, "phd_council_10y_checkpoint.json")

INIT_CAP = 1_000_000.0
COST_SIDE = 0.0025
MAX_POSITIONS = 15


def _clip(v, lo, hi):
    return max(lo, min(hi, float(v)))


def _isfinite(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def _fmt(v, suf=""):
    if not _isfinite(v):
        return "-"
    return f"{v}{suf}"


R1_SCHEMA = {
    "type": "object",
    "properties": {
        "gross_exposure_pct": {"type": "number", "description": "Recommended portfolio gross equity exposure 35-100"},
        "regime_view": {"type": "string", "description": "1-2 sentence diagnosis of macro/breadth regime"},
        "exits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["code", "reason"],
            },
        },
        "holds": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "trail_pct": {"type": "number", "description": "Trailing stop % from peak (16-28)"},
                    "note": {"type": "string"},
                },
                "required": ["code", "trail_pct", "note"],
            },
        },
        "buys": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "weight_pct": {"type": "number", "description": "Target position weight 5-10"},
                    "stop_pct": {"type": "number", "description": "Initial hard stop % (12-22)"},
                    "trail_pct": {"type": "number", "description": "Trailing stop % from peak (18-28)"},
                    "thesis": {"type": "string"},
                },
                "required": ["code", "weight_pct", "stop_pct", "trail_pct", "thesis"],
            },
        },
        "top_risk": {"type": "string"},
    },
    "required": ["gross_exposure_pct", "regime_view", "exits", "holds", "buys", "top_risk"],
}

R2_SCHEMA = {
    "type": "object",
    "properties": {
        "final_gross_pct": {"type": "number", "description": "Final binding gross equity exposure 35-100"},
        "final_exits": {"type": "array", "items": {"type": "string"}, "description": "Held STK codes to SELL"},
        "final_holds": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "trail_pct": {"type": "number"},
                },
                "required": ["code", "trail_pct"],
            },
        },
        "final_buys": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "weight_pct": {"type": "number"},
                    "stop_pct": {"type": "number"},
                    "trail_pct": {"type": "number"},
                },
                "required": ["code", "weight_pct", "stop_pct", "trail_pct"],
            },
        },
        "flips_from_r1": {"type": "array", "items": {"type": "string"}, "description": "Changes made after hearing peers"},
        "rebuttal_summary": {"type": "string", "description": "2-3 sentence cross-examination synthesis"},
    },
    "required": ["final_gross_pct", "final_exits", "final_holds", "final_buys", "flips_from_r1", "rebuttal_summary"],
}

EXPERTS = [
    (
        "PhD1_Momentum",
        "Dr. Vikram Rao — PhD Financial Microstructure & Momentum",
        "You are Dr. Vikram Rao, PhD in Financial Microstructure & Momentum Investing. "
        "Your mandate is to MAXIMIZE 10-year compounding CAGR in a concentrated max-15-stock portfolio. "
        "You champion Stage-2 leadership (Close > 50DMA & 200DMA), Dual-ATR BUY NOW breakouts, high CompRS, "
        "top Industry Strength (irank), institutional delivery conviction (dcr, drat >= 1.15), and RIDING BIG WINNERS "
        "with generous trailing stops (20-26%) rather than selling multibaggers prematurely. "
        "You ruthlessly advocate exiting Stage-4 breakdowns and stagnant laggards blocking slots for faster leaders."
    ),
    (
        "PhD2_Compounder",
        "Dr. Meera Subramanian — PhD Corporate Finance & Compounding Forensics",
        "You are Dr. Meera Subramanian, PhD in Corporate Finance & Earnings Forensics. "
        "Your mandate is to MAXIMIZE 10-year compounding CAGR in a concentrated max-15-stock portfolio. "
        "You champion high-ROCE (>=18%) secular compounders, accelerating 3Y Sales & PAT CAGR, expanding OPM (opm_d > 0), "
        "clean cash conversion (cfo_pat >= 0.70, or strong ROE/PAT growth for BFSI), reasonable valuation (PE or CMP < Bear), "
        "and strict balance-sheet discipline (low D/E). You ruthlessly veto value traps with negative profit growth, weak CFO, "
        "or deteriorating margins, while defending high-ROCE compounders through routine pullbacks."
    ),
    (
        "PhD3_MacroPM",
        "Dr. Arvind Kulkarni — PhD Macro-Regime & Portfolio Risk PM",
        "You are Dr. Arvind Kulkarni, PhD in Macro-Finance & Portfolio Construction (Risk PM). "
        "Your mandate is to MAXIMIZE 10-year compounding CAGR in a concentrated max-15-stock portfolio. "
        "You optimize capital deployment across regimes: stay 90-100% invested in BULL/RECOVERY (cash drag kills 10Y CAGR!), "
        "tighten liquidity quality (>=15-20 Cr/d) and trim gross exposure to 45-65% in BEAR stress (VIX > 25 or breadth < 35%), "
        "and pivot aggressively back into quality leaders after deep breadth washouts (pct_down_35pct_from_hi > 45%). "
        "Enforce sector diversification (max 3 stocks per sub-industry) and balance Momentum aggression with Fundamental durability."
    ),
]


def build_candidate_slate(stocks: dict, held_codes: set, macro: dict) -> list:
    pool = [s for c, s in stocks.items() if c not in held_codes]
    selected = []
    seen = set()

    def _add(items, bucket, limit):
        cnt = 0
        for s in items:
            if s["code"] in seen:
                continue
            seen.add(s["code"])
            item = dict(s)
            item["bucket"] = bucket
            selected.append(item)
            cnt += 1
            if cnt >= limit:
                break

    b_mom = [
        s for s in pool
        if s["buy_now"] and (s["fy_np_yoy"] is None or s["fy_np_yoy"] > 0)
        and (s["liq_cr"] >= (15.0 if macro["mode"] in ("BEAR", "CAUTION") else 3.0))
    ]
    b_mom.sort(key=lambda s: 0.40 * s["irank"] + 0.35 * min(100.0, math.log1p(max(0.0, s["dcr"])) * 22.0) + 0.25 * s["crs"], reverse=True)
    _add(b_mom, "MOM_LEAD", 9)

    b_comp = [
        s for s in pool
        if s["stage"] == 2
        and (s["fy_np_yoy"] is not None and s["fy_np_yoy"] >= 15.0)
        and ((s["is_bfsi"]) or ((s["roce"] is not None and s["roce"] >= 18.0) and (s["cfo_pat"] is None or s["cfo_pat"] >= 0.65)))
        and (s["cagr3_s"] is None or s["cagr3_s"] >= 10.0)
    ]
    b_comp.sort(key=lambda s: 0.35 * min(s["fy_np_yoy"], 100.0) + 0.30 * (s["roce"] or 20.0) + 0.35 * s["crs"], reverse=True)
    _add(b_comp, "COMPOUNDER", 8)

    b_garp = [
        s for s in pool
        if s["cmp_lt_bear"] and s["stage"] in (1, 2)
        and (s["fy_np_yoy"] is not None and s["fy_np_yoy"] > 0.0)
        and (s["is_bfsi"] or (s["cfo_pat"] is not None and s["cfo_pat"] >= 0.70))
    ]
    b_garp.sort(key=lambda s: 0.40 * s["crs"] + 0.30 * s["irank"] + 0.30 * min(s["fy_np_yoy"], 80.0), reverse=True)
    _add(b_garp, "GARP_S2", 7)

    b_pead = [
        s for s in pool
        if s["stage"] == 2 and s["crs"] >= 65.0 and s["drat"] >= 1.12
        and (s["fy_np_yoy"] is not None and s["fy_np_yoy"] >= 25.0)
        and (s["fy_sales_yoy"] is not None and s["fy_sales_yoy"] >= 15.0)
    ]
    b_pead.sort(key=lambda s: 0.45 * min(s["fy_np_yoy"], 120.0) + 0.30 * s["drat"] * 25.0 + 0.25 * s["irank"], reverse=True)
    _add(b_pead, "FLOW_ACCEL", 6)

    if len(selected) < 22 or macro["breadth_above_50dma_pct"] < 35.0:
        b_crash = [
            s for s in pool
            if (s["fy_np_yoy"] is not None and s["fy_np_yoy"] >= 10.0)
            and (s["is_bfsi"] or ((s["roce"] is not None and s["roce"] >= 18.0) and (s["cfo_pat"] is not None and s["cfo_pat"] >= 0.75)))
            and (s["de"] is None or s["de"] <= 0.6 or s["is_bfsi"])
            and s["liq_cr"] >= 10.0
        ]
        b_crash.sort(key=lambda s: 0.50 * (s["roce"] or 20.0) + 0.30 * min(s["fy_np_yoy"], 80.0) + 0.20 * s["crs"], reverse=True)
        _add(b_crash, "CRASH_QUAL", 8)

    return selected


def format_dossier(macro: dict, holdings_rows: list, cands: list, nav: float, cash_pct: float, prev_memory: str) -> str:
    lines = [
        f"=== BLINDED EPOCH {macro['epoch']} | PORTFOLIO NAV: Rs {nav:,.0f} (Init Rs 10,00,000) | CASH: {cash_pct:.1f}% | HOLDINGS: {len(holdings_rows)}/{MAX_POSITIONS} ===",
        f"MACRO REGIME: Mode={macro['mode']} | VIX={macro['vix']} | Breadth >50DMA={macro['breadth_above_50dma_pct']}% | >200DMA={macro['breadth_above_200dma_pct']}% | Near52WHi={macro['pct_near_52w_hi']}% | Down>35%={macro['pct_down_35pct_from_hi']}%",
        f"INDEX TELEMETRY: vs50DMA={macro['idx_vs_50dma_pct']:+.1f}% | vs200DMA={macro['idx_vs_200dma_pct']:+.1f}% | DDfromHi={macro['idx_dd_from_52w_hi_pct']:+.1f}% | 3M={macro['idx_3m_ret_pct']:+.1f}% | 6M={macro['idx_6m_ret_pct']:+.1f}% | 12M={macro['idx_12m_ret_pct']:+.1f}% | Mid6M={macro['midcap_6m_ret_pct']:+.1f}% | Sml6M={macro['smallcap_6m_ret_pct']:+.1f}%",
    ]
    if prev_memory:
        lines.append(f"RECENT PORTFOLIO OUTCOMES (LAST EPOCH): {prev_memory}")

    lines.append("\n--- CURRENT HOLDINGS (Audit EVERY held stock: HOLD or EXIT) ---")
    if not holdings_rows:
        lines.append("(None — starting fresh at 100% cash. Pick up to 15 high-conviction positions.)")
    else:
        lines.append("Code | Ind | Wt% | PnL% | Peak% | Days | Stg | RS | IndR | LiqCr | DelivR | ROCE | CFO/P | NP_YoY | Q_NP | PE | <Bear")
        for h in holdings_rows:
            lines.append(
                f"{h['code']} | {h['ind'][:16]} | {h['wt']:.1f} | {h['pnl']:+.1f} | {h['peak_pnl']:+.1f} | {h['days']} | "
                f"S{h['stage']} | {_fmt(h['crs'])} | {_fmt(h['irank'])} | {_fmt(h['liq_cr'])} | {_fmt(h['drat'])} | "
                f"{_fmt(h['roce'])} | {_fmt(h['cfo_pat'])} | {_fmt(h['fy_np_yoy'])} | {_fmt(h['q_np_yoy'])} | {_fmt(h['pe'])} | {'Y' if h['cmp_lt_bear'] else 'N'}"
            )

    lines.append(f"\n--- CURATED CANDIDATES ({len(cands)} stocks from {macro['active_universe_count']} active >=500Cr & >=2Cr/d universe) ---")
    lines.append("Code | Bucket | Ind | MCapCr | LiqCr | Stg | BN | RS | IndR | 3M/6M/12M | D_Cr | DRat | ATR% | ROCE | OPM(d) | CFO/P | D/E | 3YS/3YP | NP_YoY | Q_NP | PE | BearD%")
    for s in cands:
        m_str = f"{_fmt(s['r3m'])}/{_fmt(s['r6m'])}/{_fmt(s['r12m'])}"
        opm_str = f"{_fmt(s['opm'])}({_fmt(s['opm_d'])})"
        c3_str = f"{_fmt(s['cagr3_s'])}/{_fmt(s['cagr3_p'])}"
        lines.append(
            f"{s['code']} | {s['bucket']} | {s['ind'][:15]} | {_fmt(s['mcap_cr'])} | {s['liq_cr']} | S{s['stage']} | {'Y' if s['buy_now'] else 'N'} | "
            f"{s['crs']} | {s['irank']} | {m_str} | {s['dcr']} | {s['drat']} | {s['atrp']} | {_fmt(s['roce'])} | {opm_str} | "
            f"{_fmt(s['cfo_pat'])} | {_fmt(s['de'])} | {c3_str} | {_fmt(s['fy_np_yoy'])} | {_fmt(s['q_np_yoy'])} | {_fmt(s['pe'])} | {_fmt(s['bear_disc'])}"
        )
    return "\n".join(lines)


def perf_metrics_lists(dates_s: list, vals: list) -> dict:
    if len(vals) < 5:
        return dict(ret=0.0, cagr=0.0, mdd=0.0, sharpe=0.0, calmar=0.0)
    tot = (vals[-1] / vals[0] - 1.0) * 100.0
    d0 = datetime.date.fromisoformat(dates_s[0])
    d1 = datetime.date.fromisoformat(dates_s[-1])
    yrs = max(0.1, (d1 - d0).days / 365.25)
    cagr = ((vals[-1] / vals[0]) ** (1.0 / yrs) - 1.0) * 100.0
    pk = vals[0]
    mdd = 0.0
    rets = []
    for i, v in enumerate(vals):
        if v > pk:
            pk = v
        dd = (v / pk - 1.0) * 100.0
        if dd < mdd:
            mdd = dd
        if i > 0 and vals[i - 1] > 0:
            rets.append(v / vals[i - 1] - 1.0)
    if len(rets) > 2:
        mu = statistics.mean(rets)
        sd = statistics.stdev(rets)
        sh = (mu / sd) * math.sqrt(252.0) if sd > 1e-12 else 0.0
    else:
        sh = 0.0
    cal = cagr / abs(mdd) if abs(mdd) > 0.1 else 0.0
    return dict(ret=round(tot, 2), cagr=round(cagr, 2), mdd=round(mdd, 2), sharpe=round(sh, 3), calmar=round(cal, 2))


# ==================== MAIN WORKFLOW EXECUTION ====================
phase("Load Blinded 10.5Y Panel")
with open(PANEL_PKL, "rb") as f:
    P = pickle.load(f)

sim_dates = P["sim_dates"]
gate_dates = P["gate_dates"]
gate_snaps = P["gate_snaps"]
code_to_tk = P["code_to_tk"]
prices = P["prices"]  # tk -> (O_list, L_list, C_list)
n50_all = P["n50"]
ew_all = P["ew"]

end_di = len(sim_dates) - 1
log(f"Loaded pure-stdlib blinded panel: {len(gate_dates)} epochs ({gate_dates[0]} -> {sim_dates[-1]}), {len(prices)} active tickers.")

cash = INIT_CAP
positions = {}  # code -> dict(tk, shares, cost_basis, avg_px, peak, last_px, entry_di, entry_epoch, stop_pct, trail_pct, ind, miss)
nav_hist = {}
cash_hist = {}
trades = []
epoch_logs = []
prev_memory = ""

for gi in range(1, len(gate_dates) + 1):
    gs = gate_snaps[gi]
    g_di = gs["di"]
    next_di = gate_snaps[gi + 1]["di"] if gi < len(gate_dates) else end_di
    macro = gs["macro"]
    stocks = gs["stocks"]
    ep_lbl = macro["epoch"]

    phase(f"{ep_lbl} ({macro['mode']}, B50={macro['breadth_above_50dma_pct']}%)")

    eq_now = cash
    holdings_rows = []
    for code, p in positions.items():
        c_px = prices[p["tk"]][2][g_di]
        if not _isfinite(c_px) or c_px <= 0:
            c_px = p["last_px"]
        p["last_px"] = c_px
        mv = p["shares"] * c_px
        eq_now += mv
        s = stocks.get(code, {})
        holdings_rows.append(dict(
            code=code, ind=s.get("ind", p["ind"]), wt=0.0, mv=mv,
            pnl=(c_px / p["avg_px"] - 1.0) * 100.0,
            peak_pnl=(p["peak"] / p["avg_px"] - 1.0) * 100.0,
            days=g_di - p["entry_di"],
            stage=s.get("stage", 0), crs=s.get("crs"), irank=s.get("irank"),
            liq_cr=s.get("liq_cr"), drat=s.get("drat"), roce=s.get("roce"),
            cfo_pat=s.get("cfo_pat"), fy_np_yoy=s.get("fy_np_yoy"),
            q_np_yoy=s.get("q_np_yoy"), pe=s.get("pe"), cmp_lt_bear=s.get("cmp_lt_bear", False),
        ))
    for h in holdings_rows:
        h["wt"] = (h["mv"] / eq_now * 100.0) if eq_now > 0 else 0.0
    holdings_rows.sort(key=lambda x: x["pnl"], reverse=True)
    cash_pct = (cash / eq_now * 100.0) if eq_now > 0 else 100.0

    cands = build_candidate_slate(stocks, set(positions.keys()), macro)
    valid_cand_codes = {s["code"] for s in cands}
    held_codes = set(positions.keys())
    dossier = format_dossier(macro, holdings_rows, cands, eq_now, cash_pct, prev_memory)

    # ---- ROUND 1: Independent PhD Analysis (parallel) ----
    r1_common = (
        f"{dossier}\n\n"
        f"RULES FOR ROUND 1:\n"
        f"- Max {MAX_POSITIONS} total holdings. Goal: MAXIMIZE 10-year compounding CAGR while surviving bear crashes.\n"
        f"- Audit EVERY currently held stock (put each in either `exits` or `holds`).\n"
        f"- Propose ONLY from the curated candidate codes above for `buys` (typically 4-12 picks depending on free slots & regime).\n"
        f"- CRITICAL: All data is in this prompt. Do NOT call any file/search tools. Immediately analyze and call `finish` with valid JSON."
    )

    r1_results = await parallel(
        EXPERTS,
        lambda ex: agent(
            f"{ex[2]}\n\n{r1_common}",
            schema=R1_SCHEMA,
            role=f"{ep_lbl} R1 {ex[0]}",
            type_name="research",
            model="flash",
        ),
    )

    r1_briefs = []
    for (eid, ename, _), r1 in zip(EXPERTS, r1_results):
        ex_list = [f"{x['code']}({x['reason'][:40]})" for x in r1.get("exits", [])]
        hd_list = [f"{x['code']}(tr={x['trail_pct']}%)" for x in r1.get("holds", [])]
        by_list = [f"{x['code']}({x['weight_pct']}%,st={x['stop_pct']}%,tr={x['trail_pct']}%: {x['thesis'][:45]})" for x in r1.get("buys", [])]
        r1_briefs.append(
            f"[{ename}] GrossCap={r1['gross_exposure_pct']}% | View: {r1['regime_view']}\n"
            f"  EXITS: {', '.join(ex_list) if ex_list else 'None'}\n"
            f"  HOLDS: {', '.join(hd_list) if hd_list else 'None'}\n"
            f"  BUYS : {', '.join(by_list) if by_list else 'None'}\n"
            f"  RISK : {r1['top_risk']}"
        )
    r1_text = "\n\n".join(r1_briefs)

    # ---- ROUND 2: Adversarial Cross-Examination & Final Binding Vote (parallel) ----
    r2_common = (
        f"EPOCH {ep_lbl} PORTFOLIO & REGIME SNAPSHOT:\n{dossier}\n\n"
        f"=== ROUND 1 POSITIONS FROM ALL 3 PhD EXPERTS ===\n{r1_text}\n\n"
        f"RULES FOR ROUND 2 (ADVERSARIAL CROSS-EXAMINATION & BINDING VOTE):\n"
        f"- Challenge your peers: push back on weak fundamentals, Stage-3/4 value traps, premature exits of big winners, or excessive cash drag in bull markets.\n"
        f"- Update your own position if a peer exposed a blind spot (list changes in `flips_from_r1`).\n"
        f"- Cast your FINAL binding vote (`final_gross_pct`, `final_exits`, `final_holds`, `final_buys`).\n"
        f"- Binding rule: Exits and New Buys execute on >=2-of-3 PhD majority vote (max {MAX_POSITIONS} holdings).\n"
        f"- CRITICAL: Do NOT call any file/search tools. Immediately call `finish` with valid JSON."
    )

    r2_results = await parallel(
        EXPERTS,
        lambda ex: agent(
            f"{ex[2]}\n\n{r2_common}",
            schema=R2_SCHEMA,
            role=f"{ep_lbl} R2 {ex[0]}",
            type_name="research",
            model="flash",
        ),
    )

    # ---- DETERMINISTIC PORTFOLIO DESK RESOLUTION ----
    gross_votes = [_clip(r2.get("final_gross_pct", 80.0), 30.0, 100.0) for r2 in r2_results]
    target_gross = statistics.median(gross_votes) / 100.0

    exit_votes = {}
    for r2 in r2_results:
        for c in set(r2.get("final_exits", [])):
            if c in held_codes:
                exit_votes[c] = exit_votes.get(c, 0) + 1
    exec_exits = [c for c, cnt in exit_votes.items() if cnt >= 2]

    hold_trails = {}
    for r2 in r2_results:
        for h in r2.get("final_holds", []):
            c = h.get("code")
            if c in held_codes and c not in exec_exits:
                hold_trails.setdefault(c, []).append(_clip(h.get("trail_pct", 22.0), 14.0, 30.0))
    for c, trs in hold_trails.items():
        if c in positions and trs:
            positions[c]["trail_pct"] = statistics.mean(trs)

    buy_votes = {}
    for r2 in r2_results:
        seen_e = set()
        for b in r2.get("final_buys", []):
            c = b.get("code")
            if c in valid_cand_codes and c not in held_codes and c not in seen_e:
                seen_e.add(c)
                buy_votes.setdefault(c, []).append(dict(
                    w=_clip(b.get("weight_pct", 7.0), 4.0, 10.0),
                    st=_clip(b.get("stop_pct", 16.0), 10.0, 25.0),
                    tr=_clip(b.get("trail_pct", 22.0), 15.0, 30.0),
                ))

    ranked_buys = sorted(
        [(c, v) for c, v in buy_votes.items() if len(v) >= 2],
        key=lambda kv: (len(kv[1]), statistics.mean([x["w"] for x in kv[1]])),
        reverse=True,
    )

    pending_sells = set(exec_exits)
    pending_buys = []
    ind_cnt = {}
    for c, p in positions.items():
        if c not in pending_sells:
            ind_cnt[p["ind"]] = ind_cnt.get(p["ind"], 0) + 1

    slots_avail = max(0, MAX_POSITIONS - (len(positions) - len(pending_sells)))
    for c, vlist in ranked_buys:
        if slots_avail <= 0:
            break
        s_info = stocks[c]
        if ind_cnt.get(s_info["ind"], 0) >= 3:
            continue
        pending_buys.append(dict(
            code=c,
            wt=statistics.mean([x["w"] for x in vlist]),
            stop_pct=statistics.mean([x["st"] for x in vlist]),
            trail_pct=statistics.mean([x["tr"] for x in vlist]),
            votes=len(vlist),
            ind=s_info["ind"],
        ))
        ind_cnt[s_info["ind"]] = ind_cnt.get(s_info["ind"], 0) + 1
        slots_avail -= 1

    # ---- DAILY EXECUTION SIMULATOR FROM g_di -> next_di ----
    ep_stopped = []
    ep_bought = []
    ep_sold = []
    pending_daily_sells = set()

    for di in range(g_di, next_di + 1):
        if di == g_di + 1:
            for c in sorted(pending_sells):
                if c in positions:
                    p = positions.pop(c)
                    px = prices[p["tk"]][0][di]
                    if not _isfinite(px) or px <= 0:
                        px = p["last_px"]
                    net = p["shares"] * px * (1.0 - COST_SIDE)
                    cash += net
                    pnl_p = (px / p["avg_px"] - 1.0) * 100.0
                    trades.append(dict(epoch=ep_lbl, date=sim_dates[di], code=c, tk=p["tk"], side="SELL", reason="COUNCIL_EXIT", px=round(px, 2), pnl_pct=round(pnl_p, 1), pnl_rs=round(net - p["cost_basis"], 0)))
                    ep_sold.append(f"{c}({pnl_p:+.0f}%)")
            pending_sells.clear()

            cur_mv = sum(p["shares"] * (prices[p["tk"]][0][di] if _isfinite(prices[p["tk"]][0][di]) and prices[p["tk"]][0][di] > 0 else p["last_px"]) for p in positions.values())
            cur_eq = cash + cur_mv
            if cur_eq > 0 and (cur_mv / cur_eq) > target_gross + 0.12:
                scale = (cur_eq * target_gross) / cur_mv
                for c, p in list(positions.items()):
                    px = prices[p["tk"]][0][di]
                    if not _isfinite(px) or px <= 0:
                        px = p["last_px"]
                    keep_sh = max(1, int(p["shares"] * scale))
                    sell_sh = p["shares"] - keep_sh
                    if sell_sh > 0:
                        net = sell_sh * px * (1.0 - COST_SIDE)
                        cb_sold = p["cost_basis"] * (sell_sh / p["shares"])
                        cash += net
                        p["shares"] = keep_sh
                        p["cost_basis"] -= cb_sold
                        trades.append(dict(epoch=ep_lbl, date=sim_dates[di], code=c, tk=p["tk"], side="TRIM", reason="REGIME_TRIM", px=round(px, 2), pnl_pct=round((px / p["avg_px"] - 1.0) * 100.0, 1), pnl_rs=round(net - cb_sold, 0)))

            for b in pending_buys:
                c = b["code"]
                if c in positions or len(positions) >= MAX_POSITIONS:
                    continue
                tk = code_to_tk[c]
                px = prices[tk][0][di]
                if not _isfinite(px) or px <= 0:
                    continue
                cur_eq = cash + sum(p["shares"] * (prices[p["tk"]][0][di] if _isfinite(prices[p["tk"]][0][di]) and prices[p["tk"]][0][di] > 0 else p["last_px"]) for p in positions.values())
                room = max(0.0, cur_eq * target_gross - (cur_eq - cash))
                bv = min(cur_eq * (b["wt"] / 100.0), room, cash / (1.0 + COST_SIDE))
                sh = int(bv / px)
                if sh > 0:
                    cost = sh * px * (1.0 + COST_SIDE)
                    cash -= cost
                    positions[c] = dict(
                        tk=tk, shares=sh, cost_basis=cost, avg_px=cost / sh,
                        peak=px, last_px=px, entry_di=di, entry_epoch=ep_lbl,
                        stop_pct=b["stop_pct"], trail_pct=b["trail_pct"], ind=b["ind"], miss=0,
                    )
                    trades.append(dict(epoch=ep_lbl, date=sim_dates[di], code=c, tk=tk, side="BUY", reason=f"COUNCIL_{b['votes']}V", px=round(px, 2), pnl_pct=0.0, pnl_rs=0.0))
                    ep_bought.append(f"{c}({b['votes']}V)")
            pending_buys.clear()

        if pending_daily_sells:
            for c in sorted(pending_daily_sells):
                if c in positions:
                    p = positions.pop(c)
                    px = prices[p["tk"]][0][di]
                    if not _isfinite(px) or px <= 0:
                        px = p["last_px"]
                    net = p["shares"] * px * (1.0 - COST_SIDE)
                    cash += net
                    pnl_p = (px / p["avg_px"] - 1.0) * 100.0
                    trades.append(dict(epoch=ep_lbl, date=sim_dates[di], code=c, tk=p["tk"], side="SELL", reason="STOP_TRAIL", px=round(px, 2), pnl_pct=round(pnl_p, 1), pnl_rs=round(net - p["cost_basis"], 0)))
                    ep_stopped.append(f"{c}({pnl_p:+.0f}%)")
            pending_daily_sells.clear()

        eq_d = cash
        for c in list(positions.keys()):
            p = positions[c]
            c_px = prices[p["tk"]][2][di]
            l_px = prices[p["tk"]][1][di]
            if not _isfinite(c_px) or c_px <= 0:
                p["miss"] += 1
                if p["miss"] >= 15:
                    net = p["shares"] * p["last_px"] * 0.85
                    cash += net
                    trades.append(dict(epoch=ep_lbl, date=sim_dates[di], code=c, tk=p["tk"], side="SELL", reason="HALT_DELIST", px=round(p["last_px"] * 0.85, 2), pnl_pct=-15.0, pnl_rs=round(net - p["cost_basis"], 0)))
                    positions.pop(c)
                    continue
                c_px = p["last_px"]
                l_px = c_px
            else:
                p["last_px"] = c_px
                p["miss"] = 0

            eq_d += p["shares"] * c_px
            if c_px > p["peak"]:
                p["peak"] = c_px

            if di > g_di and di < next_di:
                if (l_px / p["avg_px"] - 1.0) * 100.0 <= -p["stop_pct"] or (c_px / p["peak"] - 1.0) * 100.0 <= -p["trail_pct"]:
                    pending_daily_sells.add(c)

        nav_hist[di] = eq_d
        cash_hist[di] = (cash / eq_d * 100.0) if eq_d > 0 else 100.0

    ep_end_nav = nav_hist[next_di]
    ep_ret = (ep_end_nav / eq_now - 1.0) * 100.0
    n_flips = sum(len(r2.get("flips_from_r1", [])) for r2 in r2_results)
    prev_memory = (
        f"Last epoch {ep_lbl} ({macro['mode']}, GrossTarget={target_gross*100:.0f}%) returned {ep_ret:+.1f}%. "
        f"Bought: {', '.join(ep_bought[:6]) or 'None'}. Council Exits: {', '.join(ep_sold[:6]) or 'None'}. "
        f"Stopped Out Intra-Epoch: {', '.join(ep_stopped[:6]) or 'None'}."
    )
    log(
        f"[{ep_lbl} {gs['date']} | {macro['mode']:8s}] Gross={target_gross*100:3.0f}% | "
        f"Buys={len(ep_bought):2d} Exits={len(ep_sold):2d} Stops={len(ep_stopped):2d} Flips={n_flips:2d} | "
        f"Pos={len(positions):2d} | EpRet={ep_ret:+6.1f}% | NAV=Rs {ep_end_nav:,.0f}"
    )

    epoch_logs.append(dict(
        epoch=ep_lbl, date=gs["date"], macro=macro,
        gross_votes=gross_votes, target_gross_pct=round(target_gross * 100.0, 1),
        buys=ep_bought, exits=ep_sold, stops=ep_stopped,
        n_flips=n_flips, start_nav=round(eq_now, 0), end_nav=round(ep_end_nav, 0), ep_ret_pct=round(ep_ret, 2),
        r1=[dict(expert=EXPERTS[i][0], gross=r1_results[i]["gross_exposure_pct"], view=r1_results[i]["regime_view"], risk=r1_results[i]["top_risk"]) for i in range(3)],
        r2=[dict(expert=EXPERTS[i][0], gross=r2_results[i]["final_gross_pct"], flips=r2_results[i]["flips_from_r1"], summary=r2_results[i]["rebuttal_summary"]) for i in range(3)],
    ))

    with open(CHECKPOINT_JSON, "w", encoding="utf-8") as cf:
        json.dump(dict(completed_epochs=gi, last_nav=round(ep_end_nav, 0), epoch_logs=epoch_logs, trades=trades), cf, indent=2)

# ==================== FINAL DE-ANONYMIZED AUDIT ====================
phase("De-Anonymize & Compute 10.5Y Analytics")
di_sorted = sorted(nav_hist.keys())
c_dates = [sim_dates[i] for i in di_sorted]
c_vals = [nav_hist[i] for i in di_sorted]
cash_vals = [cash_hist[i] for i in di_sorted]
n50_vals = [n50_all[i] / n50_all[di_sorted[0]] * INIT_CAP for i in di_sorted]
ew_vals = [ew_all[i] / ew_all[di_sorted[0]] * INIT_CAP for i in di_sorted]

pnl_by_tk = {}
for tr in trades:
    if tr["side"] in ("SELL", "TRIM"):
        pnl_by_tk[tr["tk"]] = pnl_by_tk.get(tr["tk"], 0.0) + float(tr["pnl_rs"])
open_holdings_final = []
for c, p in positions.items():
    mv = p["shares"] * p["last_px"]
    u_pnl = mv - p["cost_basis"]
    pnl_by_tk[p["tk"]] = pnl_by_tk.get(p["tk"], 0.0) + u_pnl
    open_holdings_final.append(dict(code=c, tk=p["tk"], ind=p["ind"], entry_epoch=p["entry_epoch"], pnl_pct=round((p["last_px"] / p["avg_px"] - 1.0) * 100.0, 1), pnl_rs=round(u_pnl, 0)))

REGIMES = [
    ("R1_16_17_Bull",      "2016-06-01", "2018-01-31"),
    ("R2_18_19_Bear",      "2018-02-01", "2020-01-31"),
    ("R3_20_COVID",        "2020-02-01", "2020-09-30"),
    ("R4_20_21_PostCovid", "2020-10-01", "2021-12-31"),
    ("R5_22_23_Chop",      "2022-01-01", "2023-03-31"),
    ("R6_23_24_SuperBull", "2023-04-01", "2024-09-30"),
    ("R7_24_26_Post24",    "2024-10-01", "2026-11-20"),
]

reg_table = {}
for rn, rs, re in REGIMES:
    idxs = [k for k, d in enumerate(c_dates) if rs <= d <= re]
    sd = [c_dates[k] for k in idxs]
    reg_table[rn] = dict(
        PHD_COUNCIL=perf_metrics_lists(sd, [c_vals[k] for k in idxs]),
        NIFTY_50=perf_metrics_lists(sd, [n50_vals[k] for k in idxs]),
        EW_TRADABLE=perf_metrics_lists(sd, [ew_vals[k] for k in idxs]),
    )

full_council = perf_metrics_lists(c_dates, c_vals)
full_n50 = perf_metrics_lists(c_dates, n50_vals)
full_ew = perf_metrics_lists(c_dates, ew_vals)
sells_all = [t for t in trades if t["side"] == "SELL"]
win_rate = round(sum(1 for t in sells_all if t["pnl_pct"] > 0) / len(sells_all) * 100.0, 1) if sells_all else 0.0
top15 = [(k, round(v)) for k, v in sorted(pnl_by_tk.items(), key=lambda x: x[1], reverse=True)[:15]]
bot10 = [(k, round(v)) for k, v in sorted(pnl_by_tk.items(), key=lambda x: x[1])[:10]]

final_out = dict(
    span=f"{c_dates[0]} -> {c_dates[-1]}",
    full_10y=dict(PHD_COUNCIL=full_council, NIFTY_50=full_n50, EW_TRADABLE=full_ew),
    stats=dict(
        final_nav=round(c_vals[-1], 0),
        mean_cash_pct=round(statistics.mean(cash_vals), 1),
        total_buys=sum(1 for t in trades if t["side"] == "BUY"),
        total_sells=len(sells_all),
        closed_win_rate_pct=win_rate,
        total_r2_flips=sum(e["n_flips"] for e in epoch_logs),
    ),
    regimes=reg_table,
    top15_winners=top15,
    bot10_losers=bot10,
    open_holdings_final=sorted(open_holdings_final, key=lambda x: x["pnl_rs"], reverse=True),
    epoch_logs=epoch_logs,
    trades=trades,
)

with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(final_out, f, indent=2)

log("=" * 95)
log(f"FINAL 10.5Y BLINDED 3-PhD COUNCIL RESULT ({final_out['span']}):")
log(f"  3-PhD COUNCIL : NAV Rs {final_out['stats']['final_nav']:>10,.0f} | Ret {full_council['ret']:+7.1f}% | CAGR {full_council['cagr']:+6.2f}% | MaxDD {full_council['mdd']:6.2f}% | Sharpe {full_council['sharpe']:5.3f} | Calmar {full_council['calmar']:4.2f}")
log(f"  NIFTY_50      : Ret {full_n50['ret']:+7.1f}% | CAGR {full_n50['cagr']:+6.2f}% | MaxDD {full_n50['mdd']:6.2f}% | Sharpe {full_n50['sharpe']:5.3f} | Calmar {full_n50['calmar']:4.2f}")
log(f"  EW_TRADABLE   : Ret {full_ew['ret']:+7.1f}% | CAGR {full_ew['cagr']:+6.2f}% | MaxDD {full_ew['mdd']:6.2f}% | Sharpe {full_ew['sharpe']:5.3f} | Calmar {full_ew['calmar']:4.2f}")
log(f"  Cash Mean: {final_out['stats']['mean_cash_pct']}% | Buys: {final_out['stats']['total_buys']} | Win%: {win_rate}% | R2 Flips: {final_out['stats']['total_r2_flips']}")
log(f"  TOP 10 MULTIBAGGERS CAUGHT: {top15[:10]}")
log(f"  WORST 5 LOSERS: {bot10[:5]}")
log(f"Saved full de-anonymized results to {RESULTS_JSON}")
