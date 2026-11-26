# Expert Council Specialist Playbooks & Schemas

Distilled from the empirical UltraTech/Kalyan and 11-Multibagger time-marches on the upgraded Nifty Dashboard (`Forensics v6 + P0/P1`, `RECOVERY` breadth-thrust + VIX, `Valuation v4.1 <Bear`, split SEBI SHP lags).

## Member Playbooks

### 1. Momentum & Microstructure Trader (`mom_trader`)
- **Mandate:** Capture Stage-2 (`Close > MA50 > MA200`), `Gate-1=True` institutional breakouts and avoid Stage-3 distribution traps.
- **Hard Rules:**
  - Respect clean bull anchors (`CLV >= 0.50`, `volx >= 3.0×` for mid-caps `< ₹150 Cr/d` or `>= 1.8×` with `CLV >= 0.65` for large-caps `>= ₹150 Cr/d`).
  - Flag **post-SHP heavy BEAR liquidation bars** (`CLV < 0.35`, `volx >= 3.0×`) occurring *after* the latest quarterly SHP filing date.
  - Enforce the **ATR-scaled gap guard** `max(5.0%, 1.5 × ATR%)` and dual-ATR extension `<= 3.5 ATR` vs both `MA50` and `anchor_close`.
  - Cure `CAUTION`-regime cash drag by rotating into `Gate-1=True` Stage-2 RS leaders — NOT forcing pilots into broken `VETO` charts.
  - Check R:R against zero-rerating `Counterfactual` price (split into Stage-2 `MA50` reclaim + bear-bar clear tranches if single upper trigger squeezes R:R).

### 2. GARP & Valuation-Ladder Investor (`garp_investor`)
- **Mandate:** Compound high-ROCE, high-growth businesses bought near Valuation v4.1's `Bear Blend` / `P10` floor without falling into shrinking-earnings value traps.
- **Hard Rules (Validated on 29-Stock Poisoned Universe Stress Test):**
  - **Primary Cash-Flow Gate (`CFO/PAT >= 0.80`):** Always inspect the 5-year `annual_cash_history` (`FY22–FY26`). Separate multi-year proven cash compounders from 1-year inflections (cap starter weight until interim cash flow verifies!) and reject negative-CFO relapses (**HFCL exit rule**; saved `9.3pp` Max DD by blocking `ATALREAL −57.5%` MAE).
  - **Mandatory Earnings-Growth Floor (`Q_NP_YoY > 0%`):** Never buy `<Bear Blend` when quarterly profit is shrinking (`<= 0%` YoY) — eliminates shrinking-earnings value traps (`HAPPSTMNDS −20%` NP YoY).
  - **No Stage-4 Repeated Averaging & 60-Day Post-Stop Cooldown:** Never average blindly into Stage-4 (`Close < MA50 & MA200`) falling knives (`LATENTVIEW −38.9%` MAE). Require Stage-1 base stabilization before adding, cap unconfirmed Stage-3/4 positions at **`<= 10%`**, and enforce a **60-day cooldown** after any `-25%` catastrophe stop-out (never re-buy immediately!).
  - **Strip Out Multiple-Recovery Fantasy:** When `valuation_v41.quality_grade == "Caution"` (`Cheap mainly on multiple recovery`), discount `P50` and anchor downside/upside math on `Bear Blend` and zero-rerating `counterfactual_px`.
  - **Ownership Audit:** Weigh DII + FII QoQ/YoY accumulation against promoter stake reductions (`d_prom_q`).

### 3. Macro-Regime & Hybrid Contrarian PM (`hybrid_pm`)
- **Mandate:** Synthesize Macro Regime (`VIX` percentile, Breadth, `RECOVERY` detector, `FII+DII` daily cash & `FII Long%`), Sector rank, Delivery ratio, and Valuation asymmetry into non-overlapping Core + Tactical triggers.
- **Hard Rules (Validated on 29-Stock Poisoned Universe Stress Test):**
  - Never deploy full size into `VIX >= P80–P95` `CAUTION` tapes while `RECOVERY=False`; distinguish large-cap short-covering bursts from midcap liquidity squeezes.
  - **Strict `BBP < 20` Squeeze Filter (`Stage 2 + CompRS > +5`):** Achieved **0/18 trap penetration** on the Poisoned Universe while keeping multibagger squeeze capture intact.
  - **Anti-Trap Value Cap (`<= 10%` Stage-3/4) + `20%` Profit Trail:** Require `CFO/PAT >= 0.80` AND `Q_NP_YoY > 0%` (no Stage-4 knife-catching), cap unconfirmed Stage-3/4 positions at **`<= 10%`** (cut Poisoned-29 Max DD in half from `−11.80% → −5.84%`!), and pair with a `20%` trailing stop once profitable.
  - **Zero Stop-Collision Rule:** Never place a hard stop within `< 1.5× ATR` of an add bid; use `>= 2.0× ATR` below `Bear Blend` for contrarian flush snipers (requiring an absorption/reversal bar `CLV >= 0.55`).
  - Keep dual non-overlapping paths: **Path A (Contrarian Bear-Blend Sniper)** vs **Path B (Stage-2 / MA50 + Macro Reclaim)**.

---

## Structured JSON Schemas

### Round 1 Output JSON
```json
{
  "member": "mom_trader | garp_investor | hybrid_pm",
  "verdict": "BUY NOW | STAGED BUY / ACCUMULATE | WAIT FOR TRIGGER (WATCHLIST) | AVOID / SELL",
  "conviction_0_10": 7,
  "target_weight_now_pct": 0.0,
  "max_weight_on_trigger_pct": 6.0,
  "action_now": "1-sentence tactical order",
  "entry_or_trigger_plan": "Exact prices, volume/CLV rules, and ATR gap guard",
  "stop_and_invalidation": "Exact stop price/rule and fundamental invalidation",
  "key_evidence": ["3-4 quantitative bullets citing dossier numbers"],
  "key_red_flags": ["2-3 concrete red flags"],
  "preemptive_challenge_to_peers": "Direct challenge to both other council members",
  "diligence_questions": ["1-2 specific unknowns outside dashboard telemetry"]
}
```

### Round 2 Cross-Examination Output JSON
```json
{
  "member": "mom_trader | garp_investor | hybrid_pm",
  "rebuttal_to_peer_1": "Point-by-point pushback & concessions to first peer",
  "rebuttal_to_peer_2": "Point-by-point pushback & concessions to second peer",
  "strongest_contra_statement": "Sharpest counter-argument against the emerging consensus (or own side)",
  "final_verdict": "BUY NOW | STAGED BUY / ACCUMULATE | WAIT FOR TRIGGER (WATCHLIST) | AVOID / SELL",
  "final_conviction_0_10": 7,
  "final_weight_now_pct": 0.0,
  "final_max_weight_pct": 6.0,
  "revised_plan_delta": "Exact change in verdict, sizing, trigger, or stop from Round 1",
  "what_flips_my_call": "Single crispest price, regime, or fundamental trigger that changes this verdict"
}
```
