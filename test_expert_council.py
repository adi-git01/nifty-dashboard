"""Unit & integration tests for utils/expert_council.py and expert_council_view.py."""
from __future__ import annotations

import copy
import os
import py_compile

from utils import expert_council as ec

REPO = os.path.dirname(os.path.abspath(__file__))


def test_resolve_ticker_variants():
    t1, n1 = ec.resolve_ticker("Manorama Industries", repo=REPO)
    assert t1 == "MANORAMA.NS"
    assert "Manorama" in n1

    t2, _ = ec.resolve_ticker("MANORAMA", repo=REPO)
    assert t2 == "MANORAMA.NS"

    t3, _ = ec.resolve_ticker("ULTRACEMCO.NS", repo=REPO)
    assert t3 == "ULTRACEMCO.NS"


def test_manorama_live_dossier_and_round2_vote_flip():
    dossier = ec.build_dossier("Manorama Industries", repo=REPO)
    assert dossier["ticker"] == "MANORAMA.NS"
    assert dossier["market_regime"]["nifty_regime"]["regime"] != "UNKNOWN"
    assert dossier["forensics_v6"]["post_shp_bear_turn_cr"] > 100.0
    assert dossier["forensics_v6"]["post_shp_bear_turn_mult"] >= 5.0
    assert len(dossier["fundamentals_and_flows"]["annual_cash_history"]) >= 4
    fy_last = dossier["fundamentals_and_flows"]["annual_cash_history"][-1]
    assert fy_last["cfo_pat"] is not None and fy_last["cfo_pat"] >= 0.80

    res = ec.run_quant_council(dossier)
    assert res["round1_vote"].startswith("2–1")
    assert res["round2_final_vote"].startswith("2–1 WAIT FOR TRIGGER")
    assert res["vote_flipped"] is True
    assert res["chair_verdict"] == "WAIT FOR TRIGGER (WATCHLIST)"

    # GARP halves starter from 2.0% -> 1.0%; Hybrid flips 2.5% -> 0.0%; Mom stays 0.0%
    assert res["round1"]["garp_investor"]["target_weight_now_pct"] == 2.0
    assert res["round2"]["garp_investor"]["final_weight_now_pct"] == 1.0
    assert res["round1"]["hybrid_pm"]["target_weight_now_pct"] == 2.5
    assert res["round2"]["hybrid_pm"]["final_weight_now_pct"] == 0.0
    assert res["round2"]["mom_trader"]["final_weight_now_pct"] == 0.0

    # Non-empty Bear & Bull Contra Statements, Playbooks & Markdown export
    assert len(res["bear_contra_statement"]) > 60
    assert len(res["bull_contra_statement"]) > 60
    assert len(res["playbooks"]) == 3
    assert len(res["diligence_questions"]) >= 2

    md = ec.to_markdown(res)
    assert "Manorama Industries" in md
    assert "Explicit Contra Statements" in md
    assert "FLIP!" in md


def test_ultracemco_and_kalyankjil_smoke():
    for sym in ("ULTRACEMCO", "KALYANKJIL"):
        d = ec.build_dossier(sym, repo=REPO)
        res = ec.run_quant_council(d)
        assert res["chair_verdict"] in {
            "BUY NOW",
            "STAGED BUY / ACCUMULATE",
            "WAIT FOR TRIGGER (WATCHLIST)",
            "AVOID / SELL",
        }
        assert len(res["playbooks"]) == 3
        assert ec.to_markdown(res).startswith("# 🏛️ Expert Council Verdict:")


def test_edge_case_missing_valuation_and_negative_cfo():
    d = ec.build_dossier("MANORAMA", repo=REPO)

    # Edge 1: Missing Valuation v4.1 coverage
    d_noval = copy.deepcopy(d)
    d_noval["valuation_v41"] = {"status": "UNAVAILABLE", "reason": "No valuation coverage"}
    r_noval = ec.run_quant_council(d_noval)
    assert r_noval["round2"]["garp_investor"]["final_verdict"] == "WAIT FOR TRIGGER (WATCHLIST)"
    assert r_noval["round2"]["garp_investor"]["final_weight_now_pct"] == 0.0

    # Edge 2: Negative latest-FY CFO/PAT (HFCL cash-flow exit gate)
    d_negcfo = copy.deepcopy(d)
    d_negcfo["fundamentals_and_flows"]["annual_cash_history"][-1]["cfo_pat"] = -0.45
    d_negcfo["fundamentals_and_flows"]["annual_cash_history"][-1]["cfo_cr"] = -80.0
    r_negcfo = ec.run_quant_council(d_negcfo)
    assert r_negcfo["round2"]["garp_investor"]["final_verdict"] == "AVOID / SELL"
    assert r_negcfo["round2"]["garp_investor"]["final_weight_now_pct"] == 0.0


def test_edge_case_clean_stage2_buy_now():
    d = ec.build_dossier("MANORAMA", repo=REPO)
    d2 = copy.deepcopy(d)
    d2["price_and_trend"]["stage"] = 2
    d2["price_and_trend"]["comp_rs"] = 14.5
    d2["forensics_v6"]["gate1"] = True
    d2["forensics_v6"]["state"] = "GO"
    d2["forensics_v6"]["action"] = "BUY NOW"
    d2["forensics_v6"]["post_shp_bear_bars"] = []
    d2["forensics_v6"]["post_shp_bear_turn_cr"] = 0.0
    d2["forensics_v6"]["post_shp_bear_turn_mult"] = 0.0
    d2["valuation_v41"]["cmp_vs_bear_pct"] = -4.0

    r = ec.run_quant_council(d2)
    assert r["chair_verdict"] == "BUY NOW"
    assert r["round2_final_vote"] == "3–0 BUY NOW"
    assert r["vote_flipped"] is False


def test_ui_modules_compile_cleanly():
    for rel in ("main.py", "utils/expert_council.py", "utils/expert_council_view.py", "utils/stock_report_view.py"):
        py_compile.compile(os.path.join(REPO, rel), doraise=True)


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(tests)}/{len(tests)} EXPERT COUNCIL TESTS PASSED")
