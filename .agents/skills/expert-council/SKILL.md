---
name: expert-council
description: >-
  Convenes a 3-member adversarial Expert Council (Momentum & Microstructure Trader,
  GARP & Valuation-Ladder Investor, and Macro-Regime Hybrid PM) to analyze any NSE stock
  using live Nifty Dashboard telemetry, execute a 2-round cross-examination debate with
  explicit Bear/Bull contra statements, and synthesize split actionable playbooks.
  Use whenever the user asks to "run expert council on <stock>", "convene council on <stock>",
  "debate <stock>", or wants a multi-expert bull/bear/contra verdict grounded in dashboard data.
---

# Expert Council Workflow

Run a strictly grounded, 2-round adversarial 3-expert investment committee on any NSE company name or ticker.

## Step 1 — Build the Grounded Telemetry Dossier

Run the bundled CLI script [scripts/build_dossier.py](./scripts/build_dossier.py) against the latest Nifty Dashboard repository (containing `utils/expert_council.py` and `data/cache/market_master_*.parquet`):

```powershell
python "<skill_dir>\scripts\build_dossier.py" "<Company Name or Ticker>" --repo "<nifty_dashboard_repo>" --out "<output_dir>\dossier.json"
```

Verify exit code `0` and confirm `dossier.json` contains:
- `market_regime` (Nifty regime, breadth, engine mode, live India VIX + 252d percentile, `RECOVERY` signals, FII/DII cash & FII Long%)
- `price_and_trend` (CMP, Stage 1–4, MA20/50/200, RS 1M/3M/6M, CompRS, 20d median turnover)
- `forensics_v6` (Gate-1, State, Action, ATR%, `atr_gap_guard_pct`, 60d volume anchors, **`post_shp_bear_bars`**)
- `fundamentals_and_flows` (Biz Score, **5-year `annual_cash_history` with `CFO/PAT` & `FCF/PAT`**, 6-quarter P&L & split Promoter/FII/DII SHP, delivery%)
- `valuation_v41` (`rating`, `verdict_raw`, `bear_blend`, `p10`, `p50`, `counterfactual_px`, `quality_grade`/`quality_note`)

> **CRITICAL CHECK BEFORE ROUND 1:** Inspect `annual_cash_history` vs `biz_flags` (`cash conversion`) and `post_shp_bear_bars` vs `institutional_shp` so multi-year cash inflections and post-SHP institutional liquidation are never missed!

## Step 2 — Round 1: Parallel Independent Stances (3 Subagents)

Read [references/council_playbooks.md](./references/council_playbooks.md) for the exact empirical playbooks and JSON schemas. Launch **3 parallel `research` subagents** (`Model="inherit"`, instructing them NOT to call external tools and to stay 100% grounded in the inline `dossier.json`):
1. **`mom_trader`** — Momentum & Microstructure Trader
2. **`garp_investor`** — GARP & Valuation-Ladder Investor
3. **`hybrid_pm`** — Macro-Regime & Hybrid Contrarian PM

Wait for all 3 Round 1 JSON stances to return (do not poll).

## Step 3 — Round 2: Live Cross-Examination & Contra Statements

Send all 3 peers' Round 1 verdicts, sizing, triggers, stops, and challenges back to the **same 3 persistent subagents in parallel** via `send_message`. Force every member to confront the sharpest tensions:
- Post-SHP heavy bear liquidation vs lagging quarterly SHP accumulation
- Valuation Quality `Caution` (multiple-recovery `P50` vs zero-rerating `Counterfactual` R:R)
- Any stop-buffer collisions (`< 1.5× ATR` between an add bid and a hard stop)
- Single-year vs multi-year `CFO/PAT >= 0.80` cash conversion
- Rotating `CAUTION`-regime cash into `Gate-1=True` Stage-2 leaders vs pilot sizing

Each member returns a revised Round 2 JSON with peer rebuttals, their **sharpest Contra Statement**, final verdict/conviction/weight, and **What Flips My Call**.

## Step 4 — Synthesize Chair Verdict & Publish Report

Publish a structured Markdown report artifact containing:
1. **Executive Council Verdict & Round 1 → Round 2 Vote Shift** (explicitly highlight any organic vote flips!)
2. **Grounded Telemetry Snapshot** (`as_of` date, Regime, Forensics, 5Y Cash-Flow Audit, Valuation v4.1)
3. **Round 1 → Round 2 Debate Turning Points**
4. **Explicit Contra Statements** (both **Bear / Thesis-Killer Contra** AND **Bull / Opportunity-Cost Contra**)
5. **Split Actionable Playbooks** (Swing/Momentum, Contrarian Sniper, Core GARP) with exact triggers, ATR gap guard, stops, and invalidation rules
6. **External Diligence Questions** (specific unknowns outside dashboard telemetry to verify before full sizing)

## Optional Fast Deterministic Mode

For a zero-latency baseline or quick batch screening across multiple tickers without spawning subagents, pass `--quant`:

```powershell
python "<skill_dir>\scripts\build_dossier.py" "<Company Name or Ticker>" --repo "<nifty_dashboard_repo>" --quant --md-out "<output_dir>\council_brief.md"
```
