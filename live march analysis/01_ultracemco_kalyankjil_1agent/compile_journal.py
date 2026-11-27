"""Compile full march_journal.json + daily signal milestones + strategy counterfactuals for the 3 critics."""
from __future__ import annotations

import json
import os
import pickle

D = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\time_march"


def main() -> int:
    mf = json.load(open(os.path.join(D, "manifest.json"), encoding="utf-8"))
    fa = json.load(open(os.path.join(D, "final_attribution.json"), encoding="utf-8"))
    st = json.load(open(os.path.join(D, "book_state.json"), encoding="utf-8"))
    sd = pickle.load(open(os.path.join(D, "sealed_daily.pkl"), "rb"))

    gates = []
    for g in mf["gates"]:
        pkt = json.load(open(g["path"], encoding="utf-8"))
        op = os.path.join(D, f"orders_gate_{g['gate_num']:02d}.json")
        ord_ = json.load(open(op, encoding="utf-8")) if os.path.exists(op) else {}
        sc = {}
        for tk, s in pkt["stocks"].items():
            sc[tk] = dict(
                price=s["price"], ma50=s["ma50"], ma200=s["ma200"],
                vs_50dma=s["vs_50dma_pct"], vs_200dma=s["vs_200dma_pct"], vs_52wh=s["vs_52w_high_pct"],
                comp_rs=s["rs"]["comp_rs"], rs1m=s["rs"]["rs1m"], rs3m=s["rs"]["rs3m"],
                dna=s["dna_signal"], gate1=s["gate1"],
                ind_rank=s["industry"]["rank"], ind_grp=s["industry"]["group"], ind_st=s["industry"]["status"],
                fx_act=s["forensics"]["action"], fx_st=s["forensics"]["state"], fx_sup=s["forensics"]["supply"],
                fx_sig=s["forensics"]["signature"], fx_stop=s["forensics"]["stop"], fx_stop_pct=s["forensics"]["stop_pct"],
                half_size=s["forensics"]["half_size"],
                simba_stage=s["simba"]["stage"], simba_mtf=s["simba"]["mtf"],
                caution=s["simba"]["caution"], dist_days=s["simba"]["dist_days"],
                deliv_pct=s["microstructure"]["deliv_pct"], deliv_ratio=s["microstructure"]["deliv_ratio"],
                vol20=s["microstructure"]["vol20_ann_pct"], invvol=s["microstructure"]["invvol20_w_mult"],
                liq_cr=s["microstructure"]["liq_cr_day"],
                biz_score=s["business_pit"]["score"], qtr=s["business_pit"]["quarterly"]["quarter"],
                sales_yoy=s["business_pit"]["quarterly"]["sales_yoy"], opm=s["business_pit"]["quarterly"]["opm"],
                opm_chg=s["business_pit"]["quarterly"]["opm_chg"], np_yoy=s["business_pit"]["quarterly"]["np_yoy"],
                inst_q=s["business_pit"]["quarterly"]["inst_q"], inst_lbl=s["business_pit"]["quarterly"]["inst_label"],
                val_rating=s["valuation_pit"]["rating"], val_pct=s["valuation_pit"]["cmp_pct"],
                val_p50=s["valuation_pit"]["p50"], pe=s["valuation_pit"]["pe"],
                funnel_score=s["funnel_v3"]["score"], funnel_grade=s["funnel_v3"]["grade"],
                funnel_4=s["funnel_v3"]["passes_default_4"],
                pros=s["funnel_v3"]["pros"], cons=s["funnel_v3"]["cons"],
            )
        gates.append(dict(gate_num=g["gate_num"], date=g["date"], market=pkt["market"], stocks=sc, decision=ord_))

    milestones = {}
    for tk, rows in sd["daily_traces"].items():
        lo = min(rows, key=lambda x: x["close"])
        hi = max(rows, key=lambda x: x["close"])
        buys = [r for r in rows if r["fx_action"] == "BUY NOW"]
        trigs = [r for r in rows if r["fx_action"] == "BUY ON TRIGGER"]
        abs_ = [r for r in rows if r["fx_signature"] == "ABSORPTION"]
        shk = [r for r in rows if r["fx_shakeout"]]
        g1_days = [r for r in rows if r["gate1"]]
        g1_buys = [r for r in rows if r["gate1"] and (r["fx_action"] == "BUY NOW" or r["fx_signature"] == "ABSORPTION")]
        # Also detect July explosion path for KALYANKJIL and Nov rally for both
        milestones[tk] = dict(
            n_days=len(rows),
            start_close=rows[0]["close"],
            end_close=rows[-1]["close"],
            min_close=dict(date=lo["date"], close=lo["close"]),
            max_close=dict(date=hi["date"], close=hi["close"]),
            gate1_days=len(g1_days),
            buy_now_days=len(buys),
            buy_now_sample=[dict(date=r["date"], close=r["close"], stop=r["fx_stop"], g1=r["gate1"]) for r in buys[:8]]
                           + ([dict(date=r["date"], close=r["close"], stop=r["fx_stop"], g1=r["gate1"]) for r in buys[-5:]] if len(buys) > 8 else []),
            buy_on_trig_days=len(trigs),
            buy_on_trig_sample=[dict(date=r["date"], close=r["close"], entry=r["fx_entry"], stop=r["fx_stop"], g1=r["gate1"]) for r in trigs[:10]],
            absorption_dates=[dict(date=r["date"], close=r["close"], act=r["fx_action"], g1=r["gate1"]) for r in abs_],
            shakeout_dates=[dict(date=r["date"], close=r["close"], act=r["fx_action"], g1=r["gate1"]) for r in shk],
            g1_buynow_or_abs_count=len(g1_buys),
            g1_buynow_or_abs_first5=[dict(date=r["date"], close=r["close"], act=r["fx_action"], sig=r["fx_signature"]) for r in g1_buys[:5]],
        )

    journal = dict(
        gates=gates,
        gate_history=st["gate_history"],
        final_attribution=fa,
        daily_milestones=milestones,
    )
    out = os.path.join(D, "march_journal.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(journal, f, indent=2)
    print(json.dumps(milestones, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
