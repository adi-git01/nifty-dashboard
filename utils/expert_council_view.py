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

    # Run Council (Quant instant or Live Multi-Round LLM on click / cached in session state)
    if mode_opt.startswith("🧠"):
        cache_key = f"ec_llm_res_{dossier['ticker']}_{dossier['as_of']}_{llm_provider}_{llm_model}"
        r_col1, r_col2, _ = st.columns([0.9, 1.9, 1.6])
        n_init_rounds = int(r_col1.selectbox(
            "Debate Rounds",
            [1, 2, 3, 4, 5],
            index=1,  # default = 2 rounds (Round 1 stances + Round 2 cross-examination)
            key="ec_init_rounds",
            help="Default is 2 rounds (Round 1 independent stances + Round 2 peer cross-examination). You can also add extra follow-up rounds afterwards.",
        ))
        r_col2.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if r_col2.button(
            f"🏛️ Run Live {n_init_rounds}-Round Debate ({ec.LLM_PROVIDERS[llm_provider]['label']} · {llm_model})",
            type="primary",
            use_container_width=True,
            key="ec_run_llm",
        ):
            with st.status(
                f"Convening 3 Council Members for {n_init_rounds} round(s) via {ec.LLM_PROVIDERS[llm_provider]['label']} ({llm_model})…",
                expanded=True,
            ) as _st_box:
                def _on_rc(r_i: int, r_tot: int, rec: dict):
                    ms = rec.get("member_secs") or {}
                    per_m = " · ".join(f"{k.split('_')[0]} {v}s" for k, v in ms.items())
                    _st_box.write(f"✅ **Round {r_i}/{r_tot}** complete in **{rec.get('elapsed_sec', '?')}s** (`{rec['tally']}`)" + (f" — {per_m}" if per_m else ""))
                try:
                    _out = ec.run_llm_council(
                        dossier,
                        rounds=n_init_rounds,
                        provider=llm_provider,
                        api_key=llm_api_key,
                        model=llm_model,
                        base_url=llm_base_url,
                        on_round_complete=_on_rc,
                    )
                    st.session_state[cache_key] = _out
                    _st_box.update(
                        label=f"✅ {n_init_rounds}-Round Live Debate finished in {_out.get('total_elapsed_sec', '?')}s ({llm_model})",
                        state="complete",
                        expanded=False,
                    )
                except Exception as exc:
                    _st_box.update(label=f"❌ Live LLM debate failed: {exc}", state="error", expanded=True)
                    st.error(f"Live LLM debate failed ({exc}). Falling back to Quant Council below.")
                    st.session_state.pop(cache_key, None)
        res = st.session_state.get(cache_key)
        if res is None:
            st.info(f"Click **🏛️ Run Live {n_init_rounds}-Round Debate** above to run live LLM agents — showing the instant **Quant Council** baseline below in the meantime.")
            res = ec.run_quant_council(dossier)
        else:
            # Allow user to request additional follow-up rounds (Round N+1) with optional stress-test question
            cur_n = int(res.get("num_rounds") or len(res.get("rounds") or []) or 2)
            with st.container(border=True):
                ac1, ac2 = st.columns([3.0, 1.4])
                extra_focus = ac1.text_input(
                    f"Add Round {cur_n + 1} · Optional challenge / scenario for the Council",
                    key=f"ec_extra_q_{cache_key}_{cur_n}",
                    placeholder="e.g. Stress-test OPM dropping 300 bps or Nifty falling 10% — do any members flip?",
                )
                ac2.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if ac2.button(
                    f"➕ Run Round {cur_n + 1}",
                    key=f"ec_add_round_{cache_key}_{cur_n}",
                    use_container_width=True,
                ):
                    with st.spinner(f"Running Round {cur_n + 1} cross-examination via {ec.LLM_PROVIDERS[llm_provider]['label']} ({llm_model})…"):
                        try:
                            st.session_state[cache_key] = ec.run_additional_round(
                                res,
                                user_prompt=(extra_focus or None),
                                provider=llm_provider,
                                api_key=llm_api_key,
                                model=llm_model,
                                base_url=llm_base_url,
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Round {cur_n + 1} failed: {exc}")
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

    # Chair Verdict Banner + Multi-Round Vote Trajectory
    num_rounds = int(res.get("num_rounds") or len(res.get("rounds") or []) or 2)
    traj_txt = res.get("vote_trajectory") or f"R1: {res['round1_vote']} → R{num_rounds}: {res['round2_final_vote']}"
    flip_badge = f" · 🔀 **VOTE SHIFT!** (`{traj_txt}`)" if res["vote_flipped"] else f" · Vote held ({num_rounds} rounds): `{traj_txt}`"
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
    st.markdown(f"#### 1 · Three-Expert Council Scorecard (Round 1 → Round {num_rounds})")
    cols = st.columns(3)
    for col, (m_key, title, subtitle) in zip(cols, _MEMBER_META):
        o1, o_last = res["round1"][m_key], res["round2"][m_key]
        mv1 = o1.get("_norm") or ec._member_view(o1, is_r1=True)
        mv_last = o_last.get("_norm") or ec._member_view(o_last, is_r1=(num_rounds == 1))
        v1, v2 = mv1["verdict"], mv_last["verdict"]
        changed = v1 != v2
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(subtitle)
            st.markdown(f"**Final (R{num_rounds}):** `{v2}`" + (f" *(flipped from R1 `{v1}`)*" if changed else ""))
            m_a, m_b = st.columns(2)
            m_a.metric("Conviction", f"{mv_last['conviction']}/10",
                       f"R1: {mv1['conviction']}/10", delta_color="off")
            m_b.metric("Weight Now", f"{mv_last['weight_now']:.1f}%",
                       f"R1: {mv1['weight_now']:.1f}% · Max {mv_last['max_weight']:.1f}%", delta_color="off")
            st.caption(f"**Latest Adaptation:** {mv_last['delta']}")

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

    # Deep-Dive Tabs (with Round-by-Round Evolution Tracker!)
    t_evo, t_debate, t_telem, t_dil = st.tabs([
        f"📈 Round-by-Round Evolution ({num_rounds} Rounds)",
        f"💬 Full Debate Transcript (R1 → R{num_rounds})",
        "📊 Telemetry, Cash-Flow & Anchor Audit",
        "🔎 External Diligence Checklist",
    ])

    with t_evo:
        rounds_list = res.get("rounds") or []
        if rounds_list:
            st.markdown("**Chair Vote Progression:** " + " &nbsp;→&nbsp; ".join(
                f"`R{r['round']}: {r['tally']}`" + (f" *(focus: {r['user_prompt']})*" if r.get("user_prompt") else "")
                for r in rounds_list
            ))
        evo_rows = res.get("evolution_rows") or ec.build_evolution_rows(res)
        if evo_rows:
            evo_df = pd.DataFrame(evo_rows)[
                ["round_label", "member", "verdict", "change", "conviction", "weight_now_pct", "max_weight_pct", "summary"]
            ].rename(columns={
                "round_label": "Round",
                "member": "Council Member",
                "verdict": "Verdict",
                "change": "Shift vs Prior Round",
                "conviction": "Conviction /10",
                "weight_now_pct": "Weight Now %",
                "max_weight_pct": "Max Weight %",
                "summary": "Key Reasoning / Adaptation",
            })
            st.dataframe(evo_df, hide_index=True, use_container_width=True)

            # Compact trajectory chart for Conviction & Weight Now across rounds
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Conviction (/10) by Round", "Target Weight Now (%) by Round"))
                edf = pd.DataFrame(evo_rows)
                colors = {"Momentum Trader": "#3B82F6", "GARP Investor": "#10B981", "Macro-Hybrid PM": "#F59E0B"}
                for m_lbl in ("Momentum Trader", "GARP Investor", "Macro-Hybrid PM"):
                    sub = edf[edf["member"] == m_lbl]
                    if sub.empty:
                        continue
                    c_hex = colors[m_lbl]
                    fig.add_trace(go.Scatter(x=sub["round_label"], y=sub["conviction"], name=m_lbl, mode="lines+markers", line=dict(color=c_hex, width=2.5), legendgroup=m_lbl), row=1, col=1)
                    fig.add_trace(go.Scatter(x=sub["round_label"], y=sub["weight_now_pct"], name=m_lbl, mode="lines+markers", line=dict(color=c_hex, width=2.5, dash="dot"), legendgroup=m_lbl, showlegend=False), row=1, col=2)
                fig.update_yaxes(range=[0, 10.5], row=1, col=1)
                fig.update_layout(height=270, margin=dict(l=10, r=10, t=35, b=10), legend=dict(orientation="h", y=-0.22),
                                  template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            except Exception:
                pass

    with t_debate:
        rounds_list = res.get("rounds") or []
        for rec in rounds_list:
            r_num = rec["round"]
            hdr = f"**{rec['label']}** — Tally: `{rec['tally']}`" + (f" · *User challenge:* _{rec['user_prompt']}_" if rec.get("user_prompt") else "")
            st.markdown(hdr)
            for m_key, title, _ in _MEMBER_META:
                om = rec["members"][m_key]
                mv = om.get("_norm") or ec._member_view(om, is_r1=(r_num == 1))
                with st.expander(f"R{r_num} · {title} — `{mv['verdict']}` ({mv['conviction']}/10, {mv['weight_now']:.1f}% now)", expanded=False):
                    if r_num == 1:
                        st.markdown(f"**Action Now:** {om.get('action_now', '')}")
                        st.markdown(f"**Entry / Trigger Plan:** {om.get('entry_or_trigger_plan', '')}")
                        st.markdown(f"**Stop / Invalidation:** {om.get('stop_and_invalidation', '')}")
                        if om.get("key_evidence"):
                            st.markdown("**Key Evidence:**\n" + "\n".join(f"- {x}" for x in om["key_evidence"]))
                        if om.get("key_red_flags"):
                            st.markdown("**Key Red Flags:**\n" + "\n".join(f"- {x}" for x in om["key_red_flags"]))
                        if om.get("preemptive_challenge_to_peers"):
                            st.markdown(f"**Challenge to Peers:** {om['preemptive_challenge_to_peers']}")
                    else:
                        st.markdown(f"**Adaptation / Delta:** {om.get('revised_plan_delta', '')}")
                        for k in ("rebuttal_to_mom", "rebuttal_to_garp", "rebuttal_to_hybrid", "rebuttal_to_peer_1", "rebuttal_to_peer_2"):
                            if om.get(k):
                                st.markdown(f"- *{k.replace('_', ' ').title()}:* {om[k]}")
                        if om.get("strongest_contra_statement"):
                            st.markdown(f"**Sharpest Contra Statement:** {om['strongest_contra_statement']}")
                        if om.get("what_flips_my_call"):
                            st.markdown(f"**What Flips My Call:** {om['what_flips_my_call']}")

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
