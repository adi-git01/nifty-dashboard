"""🏛️ Expert Council Streamlit view — 3-Member 2-Round Adversarial Investment Committee."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import expert_council as ec
from utils import funnel_view as fv


_MEMBER_META = (
    ("mom_trader", "⚡ Momentum & Microstructure", "Stage-2 / Forensics v6 / Post-SHP Supply"),
    ("garp_investor", "💎 GARP & Valuation-Ladder", "CFO/PAT ≥ 0.80 / Bear Blend / Counterfactual"),
    ("hybrid_pm", "🌐 Macro-Regime Hybrid PM", "VIX / Breadth / FII-DII / 2×ATR Sniper"),
)


def _tone(verdict: str) -> str:
    v = (verdict or "").upper()
    if "BUY NOW" in v:
        return "success"
    if "STAGED BUY" in v or "ACCUMULATE" in v:
        return "info"
    if "AVOID" in v or "SELL" in v:
        return "error"
    return "warning"


@st.cache_data(ttl=1800, show_spinner="Building grounded telemetry dossier…")
def _cached_dossier(snap_key: str, ticker: str, _df: pd.DataFrame, _nifty: pd.DataFrame | None) -> dict:
    return ec.build_dossier(ticker, df=_df, nifty_df=_nifty, repo=".")


def render(df: pd.DataFrame) -> None:
    from utils import extended
    from utils.nifty1000_list import EXTENDED

    core = df["ticker"].dropna().tolist()
    have = set(core)
    ext = [t for t in EXTENDED if t not in have]
    tickers = core + ext
    names = dict(zip(df["ticker"], df.get("name", df["ticker"]).astype(str)))
    names.update({t: f"{EXTENDED[t]['name']}  · extended" for t in ext})

    explicit = st.session_state.pop("council_ticker", None)
    if explicit in tickers:
        st.session_state["ec_pick"] = explicit
    elif "ec_seeded" not in st.session_state:
        want = st.session_state.get("sr_pick") or st.session_state.get("quick_dive_ticker")
        if want in tickers:
            st.session_state["ec_pick"] = want
    st.session_state["ec_seeded"] = True

    c1, c2, c3 = st.columns([2.6, 1.3, 1.5])
    t = c1.selectbox(
        "Stock",
        tickers,
        key="ec_pick",
        format_func=lambda x: f"{x.replace('.NS', '')} — {names.get(x, '')}",
        help="Pick any ranked or extended NSE stock to convene the 3-member Expert Council.",
    )
    other = c2.text_input(
        "…or any NSE symbol / name",
        key="ec_any",
        placeholder="e.g. MANORAMA",
        help="Type any NSE symbol or company fragment and press Enter.",
    )
    if other and other.strip():
        try:
            t, _ = ec.resolve_ticker(other.strip(), df=df, repo=".")
        except Exception:
            t = extended.norm(other.strip())

    mode_opt = c3.radio(
        "Council Engine",
        ["⚡ Quant Council (Instant)", "🧠 Live LLM Debate"],
        horizontal=False,
        key="ec_mode",
        help="Quant Council runs the empirical 2-round adversarial rules instantly with zero API key. Live LLM Debate runs 6 parallel calls (Round 1 + Round 2) via OpenCode, DeepSeek, Gemini, or any OpenAI-compatible endpoint.",
    )

    llm_provider = None
    llm_api_key = None
    llm_model = None
    llm_base_url = None
    if mode_opt.startswith("🧠"):
        auto_prov, _ = ec.llm_backend_status()
        prov_keys = ["opencode_go", "opencode", "deepseek", "gemini", "custom"]
        default_idx = prov_keys.index(auto_prov) if auto_prov in prov_keys else 0
        if "ec_llm_prov" not in st.session_state or st.session_state["ec_llm_prov"] not in prov_keys:
            st.session_state["ec_llm_prov"] = prov_keys[default_idx]

        p_sel = st.session_state["ec_llm_prov"]
        active_code, status_txt = ec.llm_backend_status(provider=p_sel)
        with st.expander(f"LLM Backend · {status_txt}", expanded=(active_code == "none")):
            lc1, lc2, lc3 = st.columns([1.4, 1.5, 1.7])
            llm_provider = lc1.selectbox(
                "Provider",
                prov_keys,
                key="ec_llm_prov",
                format_func=lambda k: ec.LLM_PROVIDERS[k]["label"],
            )
            pspec = ec.LLM_PROVIDERS[llm_provider]
            live_cache_k = f"ec_live_models_{llm_provider}"
            merged_models = list(dict.fromkeys(list(pspec["models"]) + list(st.session_state.get(live_cache_k) or [])))
            model_opts = merged_models + ["Custom model…"]
            picked_m = lc2.selectbox("Model", model_opts, key=f"ec_model_{llm_provider}")
            if picked_m == "Custom model…":
                llm_model = lc2.text_input("Custom model ID", value=pspec["default_model"], key=f"ec_custom_m_{llm_provider}").strip()
            else:
                llm_model = picked_m

            _has_saved_key = ec.llm_backend_status(provider=llm_provider)[0] != "none"
            llm_api_key = lc3.text_input(
                f"{pspec['label']} API Key ({'✅ loaded from secrets' if _has_saved_key else 'optional if in secrets/env'})",
                type="password",
                key=f"ec_key_{llm_provider}",
                placeholder=("Loaded from Streamlit secrets — leave blank" if _has_saved_key else f"Paste key or add {pspec['env_key']} in Secrets"),
                help="Stored only in this browser session. Leave blank when already configured in Streamlit secrets or environment variables.",
            ).strip() or None

            if llm_provider in ("opencode_go", "opencode", "deepseek", "custom"):
                u1, u2 = st.columns([3.2, 1.4])
                llm_base_url = u1.text_input(
                    "Base URL (OpenAI-compatible /v1)",
                    value=ec.resolve_llm_config(provider=llm_provider).get("base_url") or pspec["base_url"],
                    key=f"ec_url_{llm_provider}",
                    help="OpenCode Go: https://opencode.ai/zen/go/v1 · OpenCode Zen: https://opencode.ai/zen/v1 · DeepSeek: https://api.deepseek.com/v1 · Local: http://localhost:4096/v1",
                ).strip() or None
                u2.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if u2.button("🔄 Fetch Live Models", key=f"ec_fetch_m_{llm_provider}", use_container_width=True):
                    try:
                        live_ids = ec.list_provider_models(llm_provider, api_key=llm_api_key, base_url=llm_base_url)
                        if live_ids:
                            st.session_state[live_cache_k] = live_ids
                            st.toast(f"Loaded {len(live_ids)} live models from {pspec['label']}", icon="✅")
                            st.rerun()
                        else:
                            st.warning("Endpoint returned an empty model list.")
                    except Exception as exc:
                        st.warning(f"Could not fetch live models: {exc}")

    # Build strictly grounded dossier
    try:
        snap_key = fv.snap_key(df)
        dossier = _cached_dossier(snap_key, t, df, st.session_state.get("nifty_data"))
    except Exception as exc:
        st.error(f"Could not assemble telemetry dossier for `{t}`: {exc}")
        return

    # Run Council (Quant instant or Live LLM on click / cached in session state)
    if mode_opt.startswith("🧠"):
        cache_key = f"ec_llm_res_{dossier['ticker']}_{dossier['as_of']}_{llm_provider}_{llm_model}"
        run_col, _ = st.columns([1.6, 2.8])
        if run_col.button(
            f"🏛️ Run Live 2-Round Debate ({ec.LLM_PROVIDERS[llm_provider]['label']} · {llm_model})",
            type="primary",
            use_container_width=True,
            key="ec_run_llm",
        ):
            with st.spinner(f"Convening 3 Council Members via {ec.LLM_PROVIDERS[llm_provider]['label']} ({llm_model})…"):
                try:
                    st.session_state[cache_key] = ec.run_llm_council(
                        dossier,
                        provider=llm_provider,
                        api_key=llm_api_key,
                        model=llm_model,
                        base_url=llm_base_url,
                    )
                except Exception as exc:
                    st.error(f"Live LLM debate failed ({exc}). Falling back to Quant Council below.")
                    st.session_state.pop(cache_key, None)
        res = st.session_state.get(cache_key)
        if res is None:
            st.info("Click **🏛️ Run Live 2-Round Debate** above to run live LLM agents — showing the instant **Quant Council** baseline below in the meantime.")
            res = ec.run_quant_council(dossier)
    else:
        res = ec.run_quant_council(dossier)

    pt = dossier["price_and_trend"]
    fx = dossier["forensics_v6"]
    mr = dossier["market_regime"]
    val = dossier["valuation_v41"]

    # Header KPI strip
    h = st.columns([2.4, 1, 1, 1, 1.3])
    h[0].markdown(f"### {res['name']}\n`{res['ticker']}` · {dossier['industry_and_funnel'].get('industry') or '—'} · As of **{res['as_of']}**")
    h[1].metric("CMP", f"₹{pt['close']:,.1f}", f"{pt.get('dist_ma50_pct'):+.1f}% vs MA50" if pt.get("dist_ma50_pct") is not None else None)
    h[2].metric("Forensics", str(fx.get("state") or "—"), f"Stage {pt.get('stage')} · {fx.get('action')}", delta_color="off")
    h[3].metric("Valuation", str(val.get("rating") or val.get("status") or "—").split(" (")[0],
                f"P{val['cmp_pct']:.0f} · Bear ₹{val['bear_blend']:,.0f}" if val.get("status") == "OK" and val.get("bear_blend") else None, delta_color="off")
    vix = mr.get("vix") or {}
    h[4].metric("Regime", str(mr.get("engine_mode") or "—"),
                f"VIX {vix.get('close')} (P{vix.get('pctile_252'):.0f})" if vix.get("close") and vix.get("pctile_252") is not None else None, delta_color="off")

    # Chair Verdict Banner + Vote Shift
    flip_badge = f" · 🔀 **VOTE FLIP!** (`{res['round1_vote']}` → `{res['round2_final_vote']}`)" if res["vote_flipped"] else f" · Vote held: `{res['round2_final_vote']}`"
    getattr(st, _tone(res["chair_verdict"]))(
        f"**🏛️ CHAIR VERDICT: {res['chair_verdict']}**{flip_badge}\n\n{res['chair_headline']}"
    )

    # Action bar (Download Markdown + Jump to Stock Report)
    b1, b2, _ = st.columns([1.4, 1.2, 2.4])
    b1.download_button(
        "⬇️ Download Council Brief (Markdown)",
        ec.to_markdown(res),
        file_name=f"{res['ticker'].replace('.NS', '')}_expert_council.md",
        mime="text/markdown",
        key="ec_dl_md",
        use_container_width=True,
    )
    if b2.button("📋 Open Stock Report", key="ec_to_sr", use_container_width=True):
        st.session_state["report_ticker"] = dossier["ticker"]
        st.session_state["nav_page"] = "📋 Stock Report"
        st.rerun()

    # 3-Member Council Scorecard
    st.markdown("#### 1 · Three-Expert Council Scorecard (Round 1 → Round 2)")
    cols = st.columns(3)
    for col, (m_key, title, subtitle) in zip(cols, _MEMBER_META):
        o1, o2 = res["round1"][m_key], res["round2"][m_key]
        v1, v2 = o1.get("verdict", "—"), o2.get("final_verdict", o1.get("verdict", "—"))
        changed = v1 != v2
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(subtitle)
            st.markdown(f"**Final:** `{v2}`" + (f" *(flipped from `{v1}`)*" if changed else ""))
            m_a, m_b = st.columns(2)
            m_a.metric("Conviction", f"{o2.get('final_conviction_0_10', '—')}/10",
                       f"R1: {o1.get('conviction_0_10', '—')}", delta_color="off")
            m_b.metric("Weight Now", f"{o2.get('final_weight_now_pct', 0)}%",
                       f"Max {o2.get('final_max_weight_pct', 0)}%", delta_color="off")
            st.caption(f"**Adaptation:** {o2.get('revised_plan_delta', '')}")

    # Explicit Contra Statements (Bear vs Bull)
    st.markdown("#### 2 · Explicit Contra Statements (What Breaks Both Sides)")
    cc1, cc2 = st.columns(2)
    cc1.error(f"**🔴 Bear / Thesis-Killer Contra**\n\n{res['bear_contra_statement']}")
    cc2.success(f"**🟢 Bull / Opportunity-Cost Contra**\n\n{res['bull_contra_statement']}")

    # Split Actionable Playbooks
    st.markdown("#### 3 · Split Actionable Playbooks by Mandate")
    pb_df = pd.DataFrame(res["playbooks"]).rename(columns={
        "mandate": "Mandate",
        "action_today": "Action Today",
        "triggers": "Triggers / Ladder",
        "stop_rule": "Stop & Invalidation",
        "payoff_anchor": "Payoff Anchor",
    })
    st.dataframe(pb_df, hide_index=True, use_container_width=True)

    # Deep-Dive Tabs
    t_debate, t_telem, t_dil = st.tabs([
        "💬 Round 1 → Round 2 Debate Transcript",
        "📊 Telemetry, Cash-Flow & Anchor Audit",
        "🔎 External Diligence Checklist",
    ])

    with t_debate:
        for m_key, title, _ in _MEMBER_META:
            o1, o2 = res["round1"][m_key], res["round2"][m_key]
            with st.expander(f"{title} — Round 1 (`{o1.get('verdict')}`) → Round 2 (`{o2.get('final_verdict')}`)", expanded=False):
                st.markdown(f"**Round 1 Action:** {o1.get('action_now', '')}")
                st.markdown(f"**Entry / Trigger Plan:** {o1.get('entry_or_trigger_plan', '')}")
                st.markdown(f"**Stop / Invalidation:** {o1.get('stop_and_invalidation', '')}")
                if o1.get("key_evidence"):
                    st.markdown("**Key Evidence:**\n" + "\n".join(f"- {x}" for x in o1["key_evidence"]))
                if o1.get("key_red_flags"):
                    st.markdown("**Key Red Flags:**\n" + "\n".join(f"- {x}" for x in o1["key_red_flags"]))
                st.markdown("---")
                st.markdown("**Round 2 Peer Rebuttals:**")
                for k in ("rebuttal_to_mom", "rebuttal_to_garp", "rebuttal_to_hybrid", "rebuttal_to_peer_1", "rebuttal_to_peer_2"):
                    if o2.get(k):
                        st.markdown(f"- *{k.replace('_', ' ').title()}:* {o2[k]}")
                st.markdown(f"**Sharpest Contra Statement:** {o2.get('strongest_contra_statement', '')}")
                st.markdown(f"**What Flips My Call:** {o2.get('what_flips_my_call', '')}")

    with t_telem:
        ff = dossier["fundamentals_and_flows"]
        a_hist = ff.get("annual_cash_history") or []
        q_hist = ff.get("quarterly_history") or []
        ancs = fx.get("recent_anchors_60d") or []
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("**5-Year Annual Cash-Flow & ROCE Audit (`CFO/PAT >= 0.80` Primary Gate)**")
            if a_hist:
                st.dataframe(pd.DataFrame(a_hist), hide_index=True, use_container_width=True)
            else:
                st.caption("No multi-year annual cash-flow table on file.")
        with c_right:
            st.markdown("**6-Quarter P&L & Institutional / Promoter SHP**")
            if q_hist:
                st.dataframe(pd.DataFrame(q_hist), hide_index=True, use_container_width=True)
            else:
                st.caption("No quarterly history on file.")
        st.markdown("**Recent 60-Session Forensics Volume Anchors (Post-SHP Supply Check)**")
        if ancs:
            st.dataframe(pd.DataFrame(ancs), hide_index=True, use_container_width=True)
        else:
            st.caption("No qualifying 3×+ volume anchor bars in the last 60 sessions.")
        if fx.get("post_shp_bear_turn_cr", 0) > 0:
            st.warning(
                f"⚠️ **Post-SHP Liquidation Detected:** ₹{fx['post_shp_bear_turn_cr']:.1f} Cr "
                f"({fx['post_shp_bear_turn_mult']:.1f}× median daily turnover) of heavy CLV < 0.35 selling printed after the latest SHP period."
            )

    with t_dil:
        st.markdown("**Critical Verification Checks Outside Dashboard Telemetry (Before Committing Full Size):**")
        for idx, q in enumerate(res["diligence_questions"], 1):
            st.markdown(f"{idx}. {q}")
        st.caption(f"Engine mode: `{res['mode']}` · All figures grounded in stored EOD snapshot, Screener fundamentals, NSE delivery/index data, and Valuation v4.1. Not investment advice.")
