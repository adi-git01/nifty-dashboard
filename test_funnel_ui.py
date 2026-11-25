"""Tests for Funnel Signal Strip & By Sector UI view models, HTML renderers, and Streamlit integration."""
from __future__ import annotations

import os
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest

from utils import funnel, funnel_ui, funnel_view


def _base_row(**overrides) -> pd.Series:
    data = dict(
        ticker="TEST.NS",
        name="Test Co",
        industry="Healthcare Services",
        ind_group="LEADING",
        ind_score=78.0,
        funnel_score=80.0,
        funnel_grade="A",
        score_note="Mom 18 · Set 20 · Str 12 · Qual 16 · Cat 14",
        price=500.0,
        comp_rs=9.5,
        risk_pct=7.5,
        pe=21.0,
        biz_score=75.0,
        technicals="BUY NOW",
        fx_supply="CLEAN AIR",
        fx_signature="NEUTRAL",
        fx_shakeout=False,
        fx_bbp=45.0,
        fx_gap=False,
        hi52_bo_days=np.nan,
        ath_bo_days=np.nan,
        inst_label="NEUTRAL",
        inst_q=0.0,
        inst_y=0.0,
        deal_inst_cr=np.nan,
        deliv_pct=50.0,
        deliv_avg20=45.0,
        np_yoy=25.0,
        sales_yoy=12.0,
        opm_chg=0.5,
        val_rating="UNRATED (gate A)",
        val_pct=np.nan,
        val_loss=False,
        biz_flags="",
        nse_flags="",
        dist_ma50=8.0,
        dist_200dma=15.0,
        trigger_gap_pct=0.0,
        liq_cr=25.0,
    )
    data.update(overrides)
    return pd.Series(data)


def test_parse_pillar_scores_handles_standard_negative_and_missing():
    assert funnel_ui.parse_pillar_scores("Mom 17 · Set 18 · Str 10 · Qual 17 · Cat 21") == {
        "Mom": 17, "Set": 18, "Str": 10, "Qual": 17, "Cat": 21,
    }
    assert funnel_ui.parse_pillar_scores("Mom 20 · Set 20 · Str 12 · Qual -2 · Cat 0") == {
        "Mom": 20, "Set": 20, "Str": 12, "Qual": -2, "Cat": 0,
    }
    empty = {"Mom": 0, "Set": 0, "Str": 0, "Qual": 0, "Cat": 0}
    assert funnel_ui.parse_pillar_scores("") == empty
    assert funnel_ui.parse_pillar_scores(None) == empty
    assert funnel_ui.parse_pillar_scores(np.nan) == empty


def test_build_signal_chips_catalysts_penalties_cautions_and_zero_distribution_penalty():
    # Pure positive catalyst stock; DISTRIBUTION and DISTRIBUTING must NEVER add caution chips
    c_row = _base_row(
        fx_signature="ABSORPTION",
        fx_shakeout=True,
        inst_label="ACCUMULATING",
        inst_q=1.8,
        inst_y=3.2,
        deal_inst_cr=14.5,
        deliv_pct=70.0,
        deliv_avg20=50.0,
        np_yoy=52.0,
        val_rating="BUY",
        val_pct=18.0,
    )
    res = funnel_ui.build_signal_chips(c_row)
    pro_txt = " ".join(c["text"] for c in res["pros"])
    assert "⭐ Absorption (+13)" in pro_txt
    assert "🪤 Shakeout (+9)" in pro_txt
    assert "Inst +1.8q/+3.2y" in pro_txt
    assert "Inst Deals +₹14.5cr" in pro_txt
    assert "Deliv 70% (1.4×)" in pro_txt
    assert len(res["pros"]) == 5
    assert res["has_catalyst"] is True
    assert res["has_caution"] is False

    # DISTRIBUTION & DISTRIBUTING carry zero penalty per v3 scorecard
    dist_row = _base_row(fx_signature="DISTRIBUTION", inst_label="DISTRIBUTING", inst_y=-4.0, score_note="Mom 18 · Set 20 · Str 12 · Qual 16 · Cat 0")
    dist_res = funnel_ui.build_signal_chips(dist_row)
    assert dist_res["cons"] == []
    assert dist_res["has_caution"] is False
    assert dist_res["has_catalyst"] is False

    # Hard penalties & soft cautions
    bad_row = _base_row(
        nse_flags="⚠️ ASM Stage I",
        pe=-25.0,
        risk_pct=1.4,
        fx_supply="NEAR OVERHANG",
        val_rating="STRONG SELL/AVOID",
        val_pct=98.0,
        biz_flags="Weak cash conversion",
        dist_ma50=23.0,
        dist_200dma=65.0,
        Missed="🎯 Setup",
        **{"Gap to pass": "Stop 16.0%"},
    )
    bad_res = funnel_ui.build_signal_chips(bad_row)
    assert bad_res["has_caution"] is True
    assert bad_res["cons"][0]["text"].startswith("🎯 Missed 🎯 Setup")
    assert any(c["kind"] == "bad" and "ASM Stage I" in c["text"] for c in bad_res["cons"])

    # Soft caution + BUY ON TRIGGER informational tag
    trig_row = _base_row(technicals="BUY ON TRIGGER", trigger_gap_pct=2.4, fx_supply="INSIDE SHELF", risk_pct=12.2)
    trig_res = funnel_ui.build_signal_chips(trig_row)
    assert trig_res["has_caution"] is True
    assert any("Inside Shelf" in c["text"] for c in trig_res["cons"])
    assert any("Half-size 12.2% stop" in c["text"] for c in trig_res["cons"])
    assert any(c["kind"] == "info" and "Trigger +2.4%" in c["text"] for c in trig_res["cons"])

    # BUY ON TRIGGER alone has zero caution (trigger is informational)
    trig_clean = funnel_ui.build_signal_chips(_base_row(technicals="BUY ON TRIGGER", trigger_gap_pct=1.9, fx_supply="CLEAN AIR", risk_pct=7.5))
    assert trig_clean["has_caution"] is False
    assert trig_clean["cons"] == [{"text": "⏳ Trigger +1.9%", "kind": "info"}]


def test_filter_annotated_and_group_by_sector():
    df = pd.DataFrame([
        _base_row(ticker="MED.NS", industry="Healthcare Services", funnel_score=82.5, funnel_grade="A", fx_signature="ABSORPTION", score_note="Mom 17 · Set 18 · Str 10 · Qual 17 · Cat 21"),
        _base_row(ticker="ICICI.NS", industry="Banks", funnel_score=70.3, funnel_grade="B", score_note="Mom 20 · Set 20 · Str 12 · Qual 17 · Cat 0"),
        _base_row(ticker="PNB.NS", industry="Banks", funnel_score=69.5, funnel_grade="B", risk_pct=14.9, dist_ma50=26.4, score_note="Mom 26 · Set 17 · Str 12 · Qual 14 · Cat 0"),
        _base_row(ticker="UNKNOWN.NS", industry=np.nan, funnel_score=52.0, funnel_grade="C", technicals="WATCH", score_note="Mom 15 · Set 6 · Str 10 · Qual -2 · Cat 0"),
    ])
    items = funnel_ui.annotate_rows(df)
    all_items, counts = funnel_ui.filter_annotated(items, "ALL")
    assert counts == {"ALL": 4, "A": 1, "CAT": 1, "CLEAN": 2, "WARN": 2}
    assert len(all_items) == 4
    assert [x["bare"] for x in funnel_ui.filter_annotated(items, "A")[0]] == ["MED"]
    assert [x["bare"] for x in funnel_ui.filter_annotated(items, "CAT")[0]] == ["MED"]
    assert [x["bare"] for x in funnel_ui.filter_annotated(items, "CLEAN")[0]] == ["MED", "ICICI"]
    assert [x["bare"] for x in funnel_ui.filter_annotated(items, "WARN")[0]] == ["PNB", "UNKNOWN"]

    groups = funnel_ui.group_by_sector(items)
    assert [g["sector"] for g in groups] == ["Healthcare Services", "Banks", "Unclassified"]
    banks = groups[1]
    assert banks["count"] == 2 and banks["best"]["bare"] == "ICICI" and banks["top_score"] == 70.3


def test_html_builders_escape_xss_and_handle_negative_qual():
    xss_row = _base_row(
        ticker="XSS.NS",
        name='<script>alert("x")</script>',
        industry="Pharma & <Bio>",
        funnel_score=49.0,
        funnel_grade="C",
        score_note="Mom 18 · Set 18 · Str 10 · Qual -2 · Cat 0",
        biz_flags="<img src=x onerror=alert(1)>",
        technicals="WATCH",
    )
    items = funnel_ui.annotate_rows(pd.DataFrame([xss_row]))
    groups = funnel_ui.group_by_sector(items)

    strip_html = funnel_ui.signal_strip_html(items)
    card_html = funnel_ui.sector_card_html(groups[0], selected=True)
    faceoff_html = funnel_ui.sector_faceoff_html(groups[0], max_peers=3)

    for frag in (strip_html, card_html, faceoff_html):
        assert frag
        assert "<script>" not in frag
        assert "<img src=x" not in frag
    assert "&lt;script&gt;" in strip_html
    assert "Qual" in faceoff_html and "-2/18" in faceoff_html


class _FakeCol:
    def __init__(self, parent):
        self._p = parent

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __getattr__(self, name):
        return getattr(self._p, name)


class _FakeSt:
    """Minimal Streamlit stub to smoke-test funnel_view._show across all 4 modes."""

    def __init__(self, view_mode: str, strip_filter: str = "ALL", inst_only: bool = False, click_label: str | None = None):
        import streamlit as _real_st

        self.view_mode = view_mode
        self.strip_filter = strip_filter
        self.inst_only = inst_only
        self.click_label = click_label
        self.column_config = _real_st.column_config
        self.session_state = {}
        self.markdowns = []
        self.dataframes = []
        self.captions = []
        self.reruns = 0

    def columns(self, spec, **kwargs):
        n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [_FakeCol(self) for _ in range(n)]

    def text_input(self, *args, **kwargs):
        return ""

    def toggle(self, *args, **kwargs):
        return self.inst_only

    def segmented_control(self, *args, **kwargs):
        return self.view_mode

    def pills(self, *args, **kwargs):
        return self.strip_filter

    def radio(self, *args, **kwargs):
        return self.view_mode

    def markdown(self, body, **kwargs):
        self.markdowns.append(str(body))

    def caption(self, body, **kwargs):
        self.captions.append(str(body))

    def dataframe(self, df, **kwargs):
        self.dataframes.append(df)

    def button(self, label, *args, **kwargs):
        return bool(self.click_label and str(label).startswith(self.click_label))

    def rerun(self):
        self.reruns += 1

    def container(self, *args, **kwargs):
        return nullcontext()


def test_funnel_ui_live_snapshot_and_show_all_modes(monkeypatch):
    snap_path = os.path.join("data", "cache", "market_master_2026_11_20.parquet")
    if not os.path.exists(snap_path):
        pytest.skip("live snapshot parquet not present")

    from utils import industry_board as ib

    df = pd.read_parquet(snap_path)
    B = ib.board(df)
    V = funnel.load_valuation("data/valuation_snapshot.csv")
    T = funnel.table(df, B, V)
    R = funnel.run(T)
    picks = R["picks"]
    near = R["near"].rename(columns={"missed": "Missed", "miss_gap": "Gap to pass"})
    assert len(picks) > 0 and len(near) > 0

    for mode in ("🏷️ Signal Strip", "🏭 By Sector", "⚡ Quick Table", "🔬 Full Table"):
        for frame_key, frame in (("t_picks", picks), ("t_near", near)):
            fake = _FakeSt(view_mode=mode)
            monkeypatch.setattr(funnel_view, "st", fake)
            monkeypatch.setattr("utils.ui_search.st", fake)
            funnel_view._show(frame, frame_key, extra=("Missed", "Gap to pass") if frame_key == "t_near" else ())
            if mode in ("🏷️ Signal Strip", "🏭 By Sector"):
                assert len(fake.markdowns) >= 2 and len(fake.dataframes) == 0
                joined = "\n".join(fake.markdowns)
                assert "\n    <" not in joined  # no 4-space indented HTML blocks
            else:
                assert len(fake.dataframes) == 1 and len(fake.dataframes[0]) == len(frame)

    # Verify Signal Strip filter pill and By Sector compare navigation actions
    fake_cat = _FakeSt(view_mode="🏷️ Signal Strip", strip_filter="CAT")
    monkeypatch.setattr(funnel_view, "st", fake_cat)
    monkeypatch.setattr("utils.ui_search.st", fake_cat)
    funnel_view._show(picks, "t_cat")
    assert "fn-strip-row" in fake_cat.markdowns[-1]

    fake_cmp = _FakeSt(view_mode="🏭 By Sector", click_label="⚖️ Open Full Compare")
    monkeypatch.setattr(funnel_view, "st", fake_cmp)
    monkeypatch.setattr("utils.ui_search.st", fake_cmp)
    funnel_view._show(picks, "t_sec_cmp")
    assert fake_cmp.reruns == 1
    assert fake_cmp.session_state["nav_page"] == "⚖️ Compare Stocks"
    assert len(fake_cmp.session_state["compare_tickers"]) >= 2
