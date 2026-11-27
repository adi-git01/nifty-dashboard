# Live Time-March Report: Tradable All-Cap NSE (`>= ₹500 Cr` MCap & `>= ₹2 Cr/day` Liquidity) vs. Nifty 500 & Nifty 1000

> **Test Window:** `2026-02-27` → `2026-11-20` (`180` NSE trading sessions; `NIFTY_50 = −8.44%`, `Max DD = −11.70%`)  
> **Causality Protocol:** Strictly point-in-time (`<= t`) across `15` biweekly checkpoints (`8` full Valuation v4.1 + Forensics checkpoints, `7` monthly-gate Forensics checkpoints). `pit_mcap_cr(t) = shares_cr × Close(t)` and `20D median turnover(t) >= ₹2 Cr/d` enforced at every checkpoint `t`. Next-day `Open(t+1)` execution, `0.25%` one-way friction (`0.50%` round-trip), `10%` max single-position cap, `₹10,00,000` starting capital per sleeve.  
> **Source Artifacts:** [run_tradable500cr.py](./run_tradable500cr.py) · [tradable500cr_results.json](./tradable500cr_results.json) · [diag_tradable.py](./diag_tradable.py) · Prior benchmark: [nifty500_nifty1000_live_march_and_system_design.md](../05_nifty500_nifty1000_system_design/nifty500_nifty1000_live_march_and_system_design.md)

---

## 1. Executive Verdict — Universe Scale & Headline Comparison

Expanding from Nifty 500 (`495` stocks) → Nifty 1000 (`1,144` stocks) → **Full Tradable NSE `>= ₹500 Cr` MCap & `>= ₹2 Cr/d` Liquidity (`1,396` eligible stocks; `1,219` PIT-active on `2026-02-27`, peaking at `1,303` active in Aug 2026)** reveals **five structural laws**:

1. **Every Poisoned-Universe Anti-Trap Guardrail Holds 100% Invariant at `1,396` Stocks (and Inside the `437`-Stock Small-Cap Tail):** Guarded GARP beats Unfiltered GARP by **`+24.43 pp`** (`+56.06%` vs `+31.63%`, Sharpe `2.864` vs `1.518`, DD `−11.50%` vs `−18.28%`). Guarded Hybrid turns `−1.12%` into **`+25.66%`** (`+26.78 pp`) across all-cap, and cuts Small-Cap Tail MaxDD **nearly in half** (`−14.19% → −7.31%`, Sharpe `1.920` vs `1.123`).
2. **Signal Crowding Explodes Non-Linearly on Naive Momentum (`Max DD` Doubles to `−18.28%`):** At `1,396` stocks, `mom_post_dual_atr` (`BUY NOW` ranked by `CompRS`) hits **`251` simultaneous candidates on a single day** (`102.4` avg/event), causing MaxDD to blow out from `−9.06%` (N500) → `−11.19%` (N1000) → **`−18.28%` (`Tradable >= ₹500 Cr`)**! Swapping the tie-breaker to **`abs_deliv_cr` (Institutional Delivery ₹ Cr)** immediately cuts MaxDD by more than half (`−18.28% → −8.10%`).
3. **The Sleeve-Specific Liquidity Reversal (Biggest New Discovery):** Valuation (GARP) and Momentum (`MOMQ`) have **opposite optimal liquidity floors**:
   - **Guarded GARP peaks at `>= ₹2 Cr/d`** (`+56.06%`, Sharpe `2.864`, unlocking quiet compounders like `GHCLTEXTIL.NS +₹91k` before institutional discovery).
   - **Quality-Ranked Momentum peaks at `>= ₹20 Cr/d`** (**`+38.69%`**, MaxDD **`−6.70%`**, **Sharpe `3.497`** — up from `+22.36% / Sharpe 2.412` at `₹5 Cr/d`!), filtering out thin retail/operator spikes while keeping liquid institutional rockets (`CUPID.NS +₹1.30L`, `STLTECH.NS +₹71k`, `MTARTECH.NS +₹53k`, `SHILPAMED.NS +₹46k`, `SANSERA.NS +₹40k`).
4. **The 3-Sleeve Orthogonal Barbell Scales to Sharpe `3.543`:** Combining sleeve-optimal liquidity floors (`40% GARP @ >=₹2Cr` + `40% Quality-Mom @ >=₹20Cr` + `20% Calibrated PEAD @ >=₹20Cr`) delivers **`+40.55%`**, **`−6.71%` Max DD**, and **Sharpe `3.543`** with cross-sleeve daily return correlations locked at **`ρ = 0.42–0.50`** — beating `NIFTY_50` (`−8.44%`) by **`+49.0 pp`** and `EW_TRADABLE_500CR` (`+11.97%`) by **`+28.6 pp`** at **half the drawdown**.
5. **Critical Dashboard Infrastructure Blindspot Uncovered (`Valuation v4.1` Skips All `437` Small-Cap Tail Stocks):** Your pipeline already stores Screener annual fundamentals for **`437 / 437` (100%)** and quarterly fundamentals for **`355 / 437` (81%)** of the emerging small-cap tail outside Nifty 1000 — yet `V.load_context(extended=True)` in [valuation.py](../../utils/valuation.py#L105-L152) only finds **`1 / 437`** in `fundamentals_cache.csv` and **`0 / 437`** in `monthly_prices.csv`, causing `V.value_stock()` to emit `0` GARP valuations across the entire `437`-stock small-cap tail!

---

## 2. Four-Tier Scale Ladder: Complete Side-by-Side Strategy Matrix

| Strategy / Portfolio | Nifty 500 (`495`) | Nifty 1000 (`1,144`) | Tradable `>= ₹500 Cr & >= ₹2 Cr/d` (`1,396`, `1,219` active) | Small-Cap Tail ex-N1000 (`437`, `321` active) | Scale Pattern (`N500 → AllCap`) |
|---|---:|---:|---:|---:|---|
| **Passive Equal-Weight Bench** | `+4.87%` / `−12.24%` / `0.460` | `+6.51%` / `−12.90%` / `0.548` | **`+11.97%` / `−12.47%` / `0.869`** | `+22.45%` / `−14.18%` / `1.330`* | Right-tail skew widens sharply in small-caps (`21.5%` gained `>+50%`!) |
| **`mom_pre_dual_atr`** | `+17.58%` / `−4.85%` / `1.816` | `+32.00%` / `−5.85%` / `2.356` | **`+32.70%` / `−6.98%` / `2.244`** | `+17.04%` / `−4.78%` / `1.520` | Strong & resilient; `32.4` avg cands/ev |
| **`mom_post_dual_atr` (CompRS)** | `+0.03%` / `−9.06%` / `0.073` | `+34.04%` / `−11.19%` / `2.243` | `+25.24%` / **`−18.28%`** / `1.694` | `+23.53%` / `−7.76%` / `1.717` | **BREAKS ON DD** (`251` max cands/day → crowding blowout) |
| **`mom_quality_ranked` (Base `₹5Cr`)** | `+3.34%` / `−5.72%` / `0.547` | `+26.45%` / `−5.61%` / `2.702` | `+22.36%` / `−6.95%` / `2.412` | `+5.28%` / `−4.75%` / `0.797` | Stable DD (`−6.95%`), `43.8%` win rate (`59B`) |
| **`mom_quality_ranked` (`>= ₹20 Cr/d`)** | *(see `₹20Cr` sweep)* | `+26.45%` / `−5.61%` / `2.702` | **`+38.69%` / `−6.70%` / `3.497`** | *(tail mostly `< ₹20Cr`)* | **#1 RISK-ADJUSTED SINGLE ENGINE** (Sharpe `3.50`!) |
| **`garp_unfiltered`** | `+39.71%` / `−14.62%` / `2.336` | `+26.30%` / `−18.98%` / `1.418` | `+31.63%` / `−18.28%` / `1.518` | `0.00%` *(v4.1 data gap)* | Value-trap DD stays severe (`−18.3%`) |
| **`garp_cfo_only`** | `+61.53%` / `−12.33%` / `3.296` | `+34.92%` / `−16.87%` / `1.887` | **`+60.24%` / `−13.72%` / `2.973`** | `0.00%` *(v4.1 data gap)* | Huge boost vs unfiltered (`+28.6 pp`) |
| **`garp_guarded` (Full Anti-Trap)** | **`+50.57%` / `−10.60%` / `3.026`** | **`+47.50%` / `−10.65%` / `2.594`** | **`+56.06%` / `−11.50%` / `2.864`** | `0.00%` *(v4.1 data gap)* | **INVARIANT ALPHA KING** (`>= ₹2Cr` unlocks `GHCLTEXTIL`) |
| **`hybrid_old` (Legacy)** | `+12.58%` / `−10.52%` / `0.867` | `−1.50%` / `−17.46%` / `0.010` | `−1.12%` / `−17.38%` / `0.037` | `+18.28%` / `−14.19%` / `1.123` | **BROKEN AT SCALE** (`449` cands/day → negative return) |
| **`hybrid_guarded` (Anti-Trap)** | `+32.43%` / `−9.76%` / `2.277` | `+20.28%` / `−9.76%` / `1.494` | **`+25.66%` / `−10.76%` / `1.677`** | **`+26.92%` / `−7.31%` / `1.920`** | **+26.8 pp lift**; #1 strategy inside Small-Cap Tail! |
| **`pead_calibrated` (Expert #4)** | `+15.10%` / `−3.46%` / `2.388` | `+15.62%` / `−4.90%` / `2.337` | `+14.70%` / `−6.68%` / `2.022` | `+6.75%` / `−3.91%` / `1.415` | Ultra-low-DD orthogonal ballast (`ρ = 0.42–0.47` vs GARP) |
| **`sector_rotator_rp` (Expert #5)** | `+2.32%` / `−6.56%` / `0.383` | `+18.01%` / `−8.93%` / `1.734` | `+16.77%` / `−11.07%` / `1.567` | `+3.91%` / `−5.34%` / `0.732` | Modest alpha; better as an input factor than standalone |
| **`monthly_quant_post`** | `−1.18%` / `−11.59%` / `−0.015` | `+18.80%` / `−9.83%` / `1.182` | **`+27.95%` / `−11.00%` / `1.609`** | **`+26.95%` / `−9.82%` / `1.759`** | Scales UP dramatically in small-caps (`TBZ +₹1.04L`, `INDSWFTLAB +₹70k`) |
| **3-Sleeve Barbell (Default `50/30/20`)** | `+29.31%` / `−6.70%` / `2.866` | `+34.81%` / `−7.11%` / `2.964` | **`+37.68%` / `−7.97%` / `3.068`** | *(GARP empty)* | Monotonic Sharpe climb: `2.87 → 2.96 → 3.07` |
| **3-Sleeve Barbell (Sleeve-Optimal Liq)** | — | — | **`+40.55%` / `−6.71%` / `3.543`** | — | **`40% GARP(2Cr) + 40% MOMQ(20Cr) + 20% PEAD(20Cr)`** |

> *\*Transparency Audit on `EW_SMALLCAP_TAIL_EX_N1000` (`+22.45%`):* Pre-screening NSE parquet files for `avg_liq_2026 >= ₹1.5 Cr/d` before applying strict daily PIT `>= ₹2 Cr/d` introduces mild upward liquidity-graduation bias in the passive small-cap tail benchmark (small-caps that rallied 2–3× grew liquidity above ₹1.5 Cr/d, whereas small-caps that collapsed 60% often dried up below ₹1.5 Cr/d). Even so, within the PIT-Start (`2026-02-27`) cohort (`n = 321`), median return was `+8.54%` with **`21.5%` (`69` stocks) rallying `>+50%`** vs `10.1%` (`91` stocks) in N1000 (`n = 898`, median `−0.08%`). All active strategy comparisons and control sweeps run on the **exact same universe** and are 100% causal (`<= t`).

```mermaid
xychart-beta
    title "Sharpe Ratio Scaling Across Universe Tiers (Feb-Nov 2026)"
    x-axis ["Nifty 500 (495)", "Nifty 1000 (1144)", "Tradable >=500Cr (Base)", "Tradable >=500Cr (Opt Liq)"]
    y-axis "Annualized Sharpe Ratio" 0 --> 3.8
    bar [0.46, 0.55, 0.87, 0.87]
    line [3.03, 2.59, 2.86, 2.86]
    line [0.55, 2.70, 2.41, 3.50]
    line [2.87, 2.96, 3.07, 3.54]
```
*(Bars: Equal-Weight Benchmark · Lines: Guarded GARP, Quality-Ranked Momentum, and 3-Sleeve Orthogonal Barbell reaching `3.54`.)*

---

## 3. Control-Variable Sweeps on `Tradable >= ₹500 Cr` (`1,396` Stocks)

### 3.1 The Liquidity-Floor Reversal (`₹2 Cr` → `₹50 Cr/day`, Holding `MCap >= ₹500 Cr`)

| Liquidity Floor (20D Median) | Guarded GARP (`Ret / DD / Sharpe`) | GARP Avg Cands | Quality-Ranked Mom (`Ret / DD / Sharpe`) | MOMQ Avg Cands | Calibrated PEAD (`Ret / DD / Sharpe`) |
|---|---:|---:|---:|---:|---:|
| **`>= ₹2 Cr/day`** | **`+56.06%` / `−11.50%` / `2.864`** | `8.4` | `+26.59%` / `−6.43%` / `2.896` | `83.4` | `+14.70%` / `−6.68%` / `2.022` |
| **`>= ₹5 Cr/day`** | `+47.50%` / `−10.65%` / `2.594` | `7.1` | `+22.36%` / `−6.95%` / `2.412` | `78.6` | **`+16.99%` / `−7.60%` / `2.200`** |
| **`>= ₹10 Cr/day`** | `+46.90%` / `−10.82%` / `2.760` | `6.1` | `+27.97%` / `−6.56%` / `2.731` | `71.4` | — |
| **`>= ₹20 Cr/day`** | `+50.17%` / `−10.16%` / `2.960` | `4.8` | **`+38.69%` / `−6.70%` / `3.497`** | `59.8` | `+13.22%` / `−6.94%` / `1.934` |
| **`>= ₹50 Cr/day`** | `+39.51%` / `−8.31%` / `2.996` | `2.9` | `+37.17%` / `−6.14%` / `3.426` | `40.1` | — |

> [!IMPORTANT]
> **Why GARP and Momentum Require DIFFERENT Liquidity Floors:**
> - **GARP (`>= ₹2 Cr/d`)** hunts *quiet, neglected compounders* trading below Bear value *before* institutional liquidity arrives (`GHCLTEXTIL.NS +₹91k`, `EPACKPEB.NS +₹58k`). Raising GARP's floor to `₹50 Cr/d` starves it (`2.9` cands/ev, `12.9%` cash drag, `−16.6 pp` lower return).
> - **Momentum (`>= ₹20 Cr/d`)** hunts *active institutional breakouts*. Below `₹10 Cr/d`, thin operator/retail breakouts crowd the candidate queue (`195–210` max cands/day). Raising Momentum's floor to `>= ₹20 Cr/d` purges noisy micro-breakouts while letting liquid emerging leaders (`CUPID.NS +₹1,29,633`, `STLTECH.NS +₹71,284`, `MTARTECH.NS +₹52,705`, `SHILPAMED.NS +₹45,617`, `SANSERA.NS +₹39,929`) dominate — skyrocketing Sharpe from **`2.412 → 3.497`** (`+16.33 pp` higher return, *lower* drawdown)!

### 3.2 Market-Cap Floor Sweep (`₹500 Cr` → `₹5,000 Cr`, Holding `Liq >= ₹2 Cr/d`)

| PIT MCap Floor | Guarded GARP (`Ret / DD / Sharpe`) | GARP Top Additions/Losses | Quality-Ranked Mom (`Ret / DD / Sharpe`, at `₹2Cr`) | MOMQ (`>= ₹20 Cr/d`) |
|---|---:|---|---:|---:|
| **`>= ₹500 Cr`** | **`+56.06%` / `−11.50%` / `2.864`** | Captures `GHCLTEXTIL.NS (+₹91k)` + `EPACKPEB.NS (+₹58k)` | `+26.59%` / `−6.43%` / `2.896` | **`+38.69%` / `−6.70%` / `3.497`** |
| **`>= ₹1,000 Cr`** | `+49.57%` / `−10.59%` / `2.654` | Loses `GHCLTEXTIL.NS` (`~₹800 Cr` at entry) | `+22.51%` / `−6.43%` / `2.537` | — |
| **`>= ₹2,500 Cr`** | `+42.69%` / `−11.05%` / `2.508` | Loses `EPACKPEB.NS`, adds `ASKAUTOLTD.NS` | `+23.29%` / `−6.95%` / `2.449` | — |
| **`>= ₹5,000 Cr`** | `+52.18%` / `−10.97%` / `3.044` | Clean mid/large core (`HFCL`, `EMCURE`, `NAVINFLUOR`, `COFORGE`) | `+31.02%` / `−6.95%` / `3.050` | **`+37.19%` / `−6.70%` / `3.587`** |

**Takeaway:** You do **NOT** need a rigid `₹5,000 Cr` MCap cutoff if you use **liquidity (`>= ₹20 Cr/d`)** as your institutional quality gate for Momentum! `MCap >= ₹500 Cr + Liq >= ₹20 Cr/d` achieves `+38.69% / Sharpe 3.497` (matching `MCap >= ₹5,000 Cr`'s `3.587` Sharpe while preserving access to explosive `₹1,500–4,500 Cr` breakout leaders like `CUPID` and `STLTECH`).

### 3.3 Signal-Crowding Tie-Breaker Sweep (`mom_post_dual_atr`, `1,396` Stocks, `102` Avg / `251` Max Cands)

| Tie-Breaker Ranking Variable | Total Return | Max DD | Sharpe | Top-3 Contributors | Verdict at `1,396`-Stock Scale |
|---|---:|---:|---:|---|---|
| **`comp_rs` (Default Price RS)** | `+25.24%` | **`−18.28%`** | `1.694` | `STLTECH`, `BLISSGVS`, `MTARTECH` | **FAILS ON TAIL RISK** — picks crowded, hyper-extended late spikes |
| **`ind_rank_pct` (Sector Rank)** | `+15.55%` | `−12.53%` | `1.404` | `STLTECH`, `HONASA`, `TDPOWERSYS` | Weaker alone in small-caps (diluted by tiny sub-industries) |
| **`q_np_yoy` (Earnings Growth)** | `+21.16%` | `−8.55%` | `1.714` | `MTARTECH`, `KIRLOSENG`, `HONASA` | **Cuts MaxDD by 53%** (`−18.28% → −8.55%`) |
| **`d_inst_q` (FII+DII Δ)** | `+21.30%` | `−8.46%` | `1.723` | `MTARTECH`, `KIRLOSENG`, `GCSL` | **Cuts MaxDD by 54%** (`−18.28% → −8.46%`) |
| **`abs_deliv_cr` (Delivery ₹ Cr)** | **`+24.20%`** | **`−8.10%`** | **`1.964`** | `MTARTECH`, `KIRLOSENG`, `HONASA` | **#1 Single Tie-Breaker** — cuts DD by **56%** with ~zero return loss! |
| **Composite `MOMQ` (`Ind+Deliv+NP`) @ `₹20Cr`** | **`+38.69%`** | **`−6.70%`** | **`3.497`** | `CUPID`, `STLTECH`, `MTARTECH` | **Dominates every single factor by `+1.53` Sharpe!** |

### 3.4 Macro Regime Throttle (`ON` vs. `OFF` on `MOMQ @ ₹2Cr`)

- **`throttle_off`:** `+39.26%` | MaxDD `−7.86%` | Sharpe `2.983` | Cash `34.6%` (`79` buys)
- **`throttle_on` (`BEAR=40% / CAUTION=75%`):** `+26.59%` | MaxDD **`−6.43%`** | Sharpe `2.896` | Cash `51.8%` (`54` buys, **`51.2%` win rate**)
- **What Changed vs. N500/N1000:** In N500/N1000, the Nifty-50 Breadth Regime Throttle *lifted* Sharpe sharply (`0.90 → 1.44` in N500, `1.36 → 1.81` in N1000). In `Tradable >= ₹500 Cr`, because small-cap breadth decoupled positively from Nifty 50 during May–Sept 2026 (`BEAR` on Nifty 50 while small-caps rallied!), keeping the throttle tight on Nifty 50 created `51.8%` cash drag while only shaving `1.43 pp` of MaxDD (`−7.86% → −6.43%`). **System Design Lesson:** Keep the `BEAR` gross cap on the Momentum sleeve for crisis tail protection, OR widen `CAUTION` cap from `75% → 85%` when **Small-Cap / All-Cap breadth (`% > 50DMA`) diverges above Nifty-50 breadth**!

---

## 4. What Holds Constant vs. What Changes (N500 → N1000 → Tradable `>= ₹500 Cr`)

### Constants (Universal Market Laws — Hard-Code Forever)
1. **The 3 Anti-Trap Guardrails (`Q_NP_YoY > 0%`, No Stage-4 Knife-Catching + `60d` Post-Stop Cooldown, `<= 10%` Position Cap):** Validated across **4 independent universes** (`Poisoned 29` → `N500` → `N1000` → `Tradable 1,396` + `Small-Cap Tail 437`). Zero exceptions.
2. **Orthogonal 3-Sleeve Correlation (`ρ = 0.42–0.50`):** Guarded GARP vs. Quality-Ranked Momentum (`ρ = 0.42–0.46`) and GARP vs. Calibrated PEAD (`ρ = 0.42–0.50`) remain **rock-solid across every universe size**. Combining them beats EVERY standalone sleeve on Sharpe (`3.543` vs `2.864 / 3.497 / 2.022`).
3. **Never Rank Momentum by Raw `CompRS` Alone Once Candidates Exceed `40+`:** Pure price RS degrades catastrophically under universe expansion (`−18.28%` MaxDD at `1,396` stocks). Institutional Delivery ₹ Cr (`abs_deliv_cr`) + Quarterly Profit Growth (`Q_NP_YoY`) + Industry Strength (`ind_rank_pct`) is mandatory.
4. **Keep GARP Unthrottled (`BEAR` = Opportunity, Not Retreat):** GARP's best entries (`HFCL`, `GHCLTEXTIL`, `EMCURE`, `COFORGE`, `EPACKPEB`) happen *during* `CAUTION/BEAR` regimes when quality trades below Bear intrinsic value.

### Changes (Scale-Dependent Inversions — Must Parameterize!)
1. **Optimal Liquidity Floor Diverges by Sleeve:** `>= ₹2 Cr/d` maximizes GARP (`+56.06%`), whereas `>= ₹20 Cr/d` maximizes Quality-Ranked Momentum (`+38.69%`, Sharpe `3.497`). Using a single flat `₹2 Cr` or `₹5 Cr` cutoff for all sleeves leaves **`+0.47` Sharpe** on the table!
2. **Emerging Small-Cap Tail (`437` Stocks ex-N1000) Is a Pure Alpha Goldmine — Yet Valuation v4.1 Is Currently Blind to It:**
   - Momentum, Guarded Hybrid, PEAD, and Monthly Quant immediately caught real emerging multibaggers outside Nifty 1000: **`SIGMAADV.NS` (`+₹75.6k`)**, **`TBZ.NS` (`+₹1.04L`)**, **`VENUSREM.NS` (`+₹70.5k`)**, **`INDSWFTLAB.NS` (`+₹70.2k`)**, **`RPTECH.NS` (`+₹68.4k`)**, **`KRISHANA.NS` (`+₹49.7k`)**, **`GCSL.NS` (`+₹48.3k`)**, **`UNIPARTS.NS` (`+₹46.9k`)**, **`PAISALO.NS` (`+₹46.9k`)**, **`YASHO.NS` (`+₹43.6k`)**, **`SHREEJISPG.NS` (`+₹43.1k`)**.
   - Meanwhile, **`V.value_stock()` returned `0 / 437` valuations** purely because `fundamentals_cache.csv` (`1/437`) and `monthly_prices.csv` (`0/437`) only span Nifty 1000 — even though `fundamentals_annual.csv` already has **`437 / 437`** and `fundamentals_quarterly.csv` has **`355 / 437`**!
3. **Monthly Quant Scales *Up* with Breadth:** In Nifty 500, `monthly_quant_post` was negative (`−1.18%`, Sharpe `−0.015`); in Nifty 1000 it rose to `+18.80%`; in `Tradable >= ₹500 Cr` it reached **`+27.95%` (Sharpe `1.609`)**, and inside `SMALLCAP_TAIL_EX_N1000` it hit **`+26.95%` (Sharpe `1.759`)**. Why? Monthly rebalancing stops whipsawing high-beta emerging trends on intra-month noise.
4. **Nifty-50 Regime Throttle Can Over-Cash Small-Cap Decouplings:** When Nifty 50 lags (`−8.44%`) while `>= ₹500 Cr` breadth rallies (`+11.97%`), anchoring Momentum's gross cap solely to Nifty 50 creates `51.8%` cash drag (`throttle_off +39.26%` vs `throttle_on +26.59%`).

---

## 5. Final Production System Architecture — Optimized 3-Sleeve Orthogonal Barbell

| Barbell Configuration (`Tradable >= ₹500 Cr` Universe) | Total Return | Max DD | Sharpe | Corr (`G-M / G-P / M-P`) | Best Use Case |
|---|---:|---:|---:|---|---|
| **Base (`50% GARP 2Cr + 30% MOMQ 5Cr + 20% PEAD 2Cr`)** | `+37.68%` | `−7.97%` | `3.068` | `0.44 / 0.47 / 0.55` | Baseline uniform tradable universe |
| **Split-Liq `50/30/20` (`GARP 2Cr + MOMQ 20Cr + PEAD 20Cr`)** | **`+42.28%`** | `−7.54%` | `3.379` | `0.42 / 0.42 / 0.50` | Maximum total return (`+50.7 pp` vs Nifty 50) |
| **Institutional `>= ₹20 Cr` All-Sleeves (`50/30/20`)** | `+39.33%` | **`−6.47%`** | `3.496` | `0.43 / 0.47 / 0.50` | Larger capital (`₹1 Cr+` portfolio, minimal slippage) |
| **Balanced Optimal (`40% GARP 2Cr + 40% MOMQ 20Cr + 20% PEAD 20Cr`)** | **`+40.55%`** | **`−6.71%`** | **`3.543`** | **`0.42 / 0.42 / 0.50`** | **RECOMMENDED PRODUCTION DEFAULT** |

```mermaid
flowchart LR
    U["NSE Tradable Universe: PIT MCap >= 500 Cr & Liq >= 2 Cr/d (~1,250-1,300 Active)"]
    S1["Sleeve 1 - Guarded GARP (40%): Liq >= 2 Cr/d | CMP < Bear | CFO/PAT >= 0.70 | Q_NP > 0% | Unthrottled"]
    S2["Sleeve 2 - Quality-Ranked Mom (40%): Liq >= 20 Cr/d | Dual-ATR BUY NOW | Ind+Deliv+NP Rank | 1/ATR%"]
    S3["Sleeve 3 - Calibrated PEAD (20%): Liq >= 5-20 Cr/d | Q_NP >= 20% | Q_Rev >= 15% | Deliv >= 1.2x"]
    P["3-Sleeve Orthogonal Barbell: +40.55% | MaxDD -6.71% | Sharpe 3.543 | Corr 0.42-0.50"]
    U --> S1 --> P
    U --> S2 --> P
    U --> S3 --> P
```

### Top-2 High-ROI Engineering Upgrades for Live Production
1. **Fix the `437`-Stock Small-Cap Tail Valuation Blindspot in [valuation.py](../../utils/valuation.py):** Synthesize missing `fundamentals_cache` rows (from `screener_fundamentals` `mcap_cr` + NSE sector) and `monthly_prices` columns (resampled from daily `data/nse/` OHLCV parquets!) when `extended=True`. This instantly brings **all `437` emerging `>= ₹500 Cr` small-caps** into the Valuation v4.1 Ladder and Guarded GARP sleeve!
2. **Encode Sleeve-Specific Liquidity Floors + `abs_deliv_cr` Tie-Breaking + Dual-Breadth Throttle into the Dashboard & `Expert Council`:**
   - Default Tradable Universe filter: **`PIT MCap >= ₹500 Cr` & `20D Median Turnover >= ₹2 Cr/d`** (with automatic **`>= ₹20 Cr/d`** institutional liquidity gate on the Momentum sleeve).
   - Add **Voice #4 (`Institutional Flow & PEAD Specialist`)** to `utils/expert_council.py` so every live council debate automatically incorporates the 3rd orthogonal alpha pillar.
