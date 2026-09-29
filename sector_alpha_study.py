"""
Is there alpha in the sub-industry ranking -- and how long does a lead last?
============================================================================

The rotation heatmap ranks 58 NSE sub-industries daily by score_0_100
(trading_engine.generate_sub_industry_rotation: percentile across groups of
0.7 x mean member CompRS percentile + 0.3 x % members with CompRS > 0). The
stored history starts Mar 2025, so this rebuilds the same score daily for ~10
years (point-in-time top-1000 members, groups of >= 3), checks it against the
stored scores, and asks:

  A  LEVEL   Does a high score today mean a higher return over the next week,
             month, 3 and 6 months? (vs Nifty, and vs the average sub-industry
             on the same day -- the second removes the market's move.)
  B  DECAY   How long does the lead keep paying? The rank correlation between
             today's score and the 1-month return starting 0, 1, 3, 6, 9, 12
             months later. Where it turns negative, leaders start to lag.
  C  SPELLS  How long do leadership (score >= 70) and laggard (< 40) spells
             run, and does a FRESH leader beat a long-standing one? Does a
             long-time laggard rebound (turnaround)?
  D  GROUPS  Does any sub-industry have its own character -- leads that keep
             going, or laggards that reliably turn around? With 58 groups,
             some will look special by chance, so each group is measured in
             2016-20 and 2021-26 separately. If the ranking of groups by the
             statistic does not repeat across eras, group-specific patterns
             are noise.
  E  TRANSITIONS  What follows a colour change (the heatmap's red / yellow /
             green bands) vs staying in the same band.
  F  NOW     Each sub-industry's score, how long it has led or lagged, and its
             measured character.

Checkpoints are weekly (every 5 sessions). Forward windows overlap, so event
counts overstate the evidence; the era split and the per-date averaging in B
are the guards.

Run: python sector_alpha_study.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

HORIZONS = [5, 21, 63, 126]
ERA_SPLIT = "2021-01-01"
OUT = "analysis"


def colour(v):
    return np.select([v >= 70, v >= 40, v >= 0], ["green", "yellow", "red"], "")


def run_length(flag):
    """Consecutive True count per column, 0 where False."""
    f = flag.astype(int)
    grp = (f == 0).cumsum()
    return f.groupby(grp).cumsum() if isinstance(f, pd.Series) else f.apply(
        lambda s: s.groupby((s == 0).cumsum()).cumsum())


def spells(flag):
    """Lengths of completed True runs, all columns pooled (in checkpoints)."""
    out = []
    for c in flag.columns:
        s = flag[c].dropna().astype(int).values
        n = 0
        for v in s:
            if v:
                n += 1
            elif n:
                out.append(n)
                n = 0
    return pd.Series(out, dtype=float)


def summary(df, by, cols):
    g = df.groupby(by, observed=True)
    t = g.size().rename("n").to_frame()
    for c in cols:
        t[c] = g[c].median().round(2)
    for c in [c for c in cols if c.startswith("vsgrp")]:
        t[c.replace("vsgrp", "beat%")] = g[c].apply(lambda x: round((x > 0).mean() * 100)).astype(int)
    return t


def main():
    from momentum_factor_backtest import build_pit_universe, load_candidates
    from transfer_backtest import fetch_prices, group_panels
    from utils.nifty1000_list import SUB_INDUSTRY_MAP

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250)

    close, vol, bench = fetch_prices(load_candidates("all", args.max_tickers), "^NSEI", "2014-06-01")
    mask = build_pit_universe(close, vol, 1000)
    gret, keys, n = group_panels(close, mask, SUB_INDUSTRY_MAP, bench,
                                 [(5, .10), (21, .50), (63, .40)], india_live=True)
    score = keys["live"].rank(axis=1, pct=True) * 100        # score_0_100 as the live engine ranks it
    score = score[score.notna().sum(axis=1) >= 20]
    gidx = (1 + gret.fillna(0)).cumprod().where(n >= 3).reindex(score.index)
    bench = bench.reindex(score.index)
    print(f"[score] {score.shape[1]} sub-industries, {len(score)} sessions "
          f"{score.index[0].date()} -> {score.index[-1].date()}")

    try:
        live = pd.read_csv("data/sub_industry_rotation.csv")
        live["record_date"] = pd.to_datetime(live.record_date, format="mixed").dt.normalize()
        st = score.stack()
        st.index.names = ["record_date", "sub_industry"]
        m = live.merge(st.rename("rebuilt").reset_index(), on=["record_date", "sub_industry"])
        print(f"[validate] {len(m)} stored group-days since {m.record_date.min().date()}: rank corr "
              f"{m[['score_0_100', 'rebuilt']].corr(method='spearman').iloc[0, 1]:.2f}, same colour "
              f"{(colour(m.score_0_100) == colour(m.rebuilt)).mean() * 100:.0f}%")
    except Exception as e:
        print(f"[validate] skipped: {e}")

    # ---- long table at weekly checkpoints -----------------------------------
    cp = score.index[::5]
    S = score.loc[cp]
    green, red = (S >= 70).where(S.notna()), (S < 40).where(S.notna())
    gage, rage = run_length(green.fillna(False)), run_length(red.fillna(False))
    cols = {}
    for h in HORIZONS:
        g = (gidx.shift(-h) / gidx - 1) * 100
        nf = (bench.shift(-h) / bench - 1) * 100
        cols[f"vsnifty{h}"] = g.sub(nf, axis=0).loc[cp]
        cols[f"vsgrp{h}"] = g.sub(g.mean(axis=1), axis=0).loc[cp]
    L = pd.DataFrame({"score": S.stack(), "green_age": gage.stack(), "red_age": rage.stack(),
                      "band": pd.DataFrame(colour(S), index=S.index, columns=S.columns).stack(),
                      "prev_band": pd.DataFrame(colour(S.shift(1)), index=S.index, columns=S.columns).stack(),
                      **{k: v.stack() for k, v in cols.items()}})
    L.index.names = ["date", "group"]
    L = L.reset_index().dropna(subset=["score"])
    L["era"] = np.where(L.date < ERA_SPLIT, "2016-20", "2021-26")
    L["quintile"] = pd.cut(L.score, [0, 20, 40, 60, 80, 100.01],
                           labels=["0-20", "20-40", "40-60", "60-80", "80-100"], include_lowest=True)
    vcols = [f"vsnifty{h}" for h in HORIZONS] + [f"vsgrp{h}" for h in HORIZONS]

    # ---- A: level ------------------------------------------------------------
    print(f"\n{'=' * 120}\nA. DOES A HIGH SCORE PREDICT RETURNS?  median forward excess (pp) by score band, weekly checkpoints"
          f"\n   vsnifty = vs Nifty; vsgrp = vs the average sub-industry that day; beat% = share beating the average group\n{'=' * 120}")
    A = summary(L, "quintile", vcols)
    print(A.to_string())
    print("\n   by era (vs average group):")
    print(summary(L, ["era", "quintile"], [f"vsgrp{h}" for h in (21, 63, 126)]).to_string())

    # ---- B: decay ------------------------------------------------------------
    print(f"\n{'=' * 120}\nB. HOW LONG DOES THE LEAD PAY?  rank corr of today's score with the 1-month return "
          f"starting k sessions later\n{'=' * 120}")
    rows = []
    for k in (0, 21, 63, 126, 189, 252):
        f = (gidx.shift(-(k + 21)) / gidx.shift(-k) - 1).loc[cp]
        ic = S.rank(axis=1).corrwith(f.rank(axis=1), axis=1).dropna()
        era = ic.index < ERA_SPLIT
        rows.append(dict(start_after=f"{k} sessions (~{k // 21} mo)", mean_IC=round(ic.mean(), 3),
                         pct_weeks_positive=round((ic > 0).mean() * 100),
                         IC_2016_20=round(ic[era].mean(), 3), IC_2021_26=round(ic[~era].mean(), 3),
                         weeks=len(ic)))
    print(pd.DataFrame(rows).to_string(index=False))
    print("   IC 0.05 is a usable ranking signal for groups; sign flip = leaders start lagging.")

    # ---- C: spells -----------------------------------------------------------
    print(f"\n{'=' * 120}\nC. SPELLS  (weeks)\n{'=' * 120}")
    for name, flag in (("leader (>=70)", green), ("laggard (<40)", red)):
        s = spells(flag.fillna(False))
        print(f"   {name}: {len(s)} spells, median {s.median():.0f} wk, 75th pct {s.quantile(.75):.0f} wk, "
              f"90th pct {s.quantile(.9):.0f} wk, longest {s.max():.0f} wk")
    ab = [0, 1, 4, 12, 26, 1000]
    lab = ["1 wk", "2-4 wk", "5-12 wk", "13-26 wk", "26+ wk"]
    Lg = L[L.green_age > 0].assign(age=lambda d: pd.cut(d.green_age, ab, labels=lab))
    Lr = L[L.red_age > 0].assign(age=lambda d: pd.cut(d.red_age, ab, labels=lab))
    print("\n   LEADERS by how long they have led -- fresh vs mature:")
    print(summary(Lg, "age", [f"vsgrp{h}" for h in HORIZONS] + ["vsnifty63"]).to_string())
    print("\n   LAGGARDS by how long they have lagged -- do long-time laggards turn around?:")
    print(summary(Lr, "age", [f"vsgrp{h}" for h in HORIZONS] + ["vsnifty63"]).to_string())

    # ---- D: group character ---------------------------------------------------
    print(f"\n{'=' * 120}\nD. SUB-INDUSTRY CHARACTER  (3-month forward return vs average group, median pp)\n{'=' * 120}")
    def char(d):
        gr, rd = d.loc[d.band == "green", "vsgrp63"], d.loc[d.band == "red", "vsgrp63"]
        return pd.Series(dict(when_leading=gr.median(), when_lagging=rd.median(),
                              persistence=gr.median() - rd.median(), n_lead=len(gr), n_lag=len(rd),
                              pct_time_leading=(d.band == "green").mean() * 100))
    C = L.groupby(["group", "era"]).apply(char).unstack("era")
    full = L.groupby("group").apply(char)
    stable = {}
    for stat in ("persistence", "when_leading", "when_lagging", "pct_time_leading"):
        a, b = C[(stat, "2016-20")], C[(stat, "2021-26")]
        ok = a.notna() & b.notna()
        stable[stat] = round(a[ok].corr(b[ok], method="spearman"), 2)
    print("   Does a group's character repeat? rank corr of each statistic, 2016-20 vs 2021-26 "
          "(~0 = group-specific patterns are noise):")
    print("   " + ", ".join(f"{k} {v:+.2f}" for k, v in stable.items()))
    print("   (pct_time_leading repeats even on random data: small, volatile groups sit at the extremes more")
    print("    often. It describes volatility, not an edge -- judge character by the return columns.)")
    G = pd.DataFrame({
        "lead_persist_16_20": C[("persistence", "2016-20")], "lead_persist_21_26": C[("persistence", "2021-26")],
        "turnaround_16_20": C[("when_lagging", "2016-20")], "turnaround_21_26": C[("when_lagging", "2021-26")],
        "pct_time_leading": full.pct_time_leading,
    }).round(1)
    G["persistent_both_eras"] = (G.lead_persist_16_20 > 3) & (G.lead_persist_21_26 > 3)
    G["turnaround_both_eras"] = (G.turnaround_16_20 > 2) & (G.turnaround_21_26 > 2)
    G.to_csv(f"{OUT}/sector_group_character.csv")
    print(f"\n   Leads persist (> +3 pp spread) in BOTH eras: {int(G.persistent_both_eras.sum())} of {len(G)}")
    print(G[G.persistent_both_eras].sort_values("lead_persist_21_26", ascending=False).to_string())
    print(f"\n   Laggards rebound (> +2 pp vs avg group) in BOTH eras: {int(G.turnaround_both_eras.sum())} of {len(G)}")
    print(G[G.turnaround_both_eras].sort_values("turnaround_21_26", ascending=False).to_string())
    # what chance alone gives: shuffle the era-2 column across groups
    rng = np.random.default_rng(7)
    a, b = G.lead_persist_16_20.values, G.lead_persist_21_26.values
    ok = ~np.isnan(a) & ~np.isnan(b)
    sims = [((a[ok] > 3) & (rng.permutation(b[ok]) > 3)).sum() for _ in range(2000)]
    a2, b2 = G.turnaround_16_20.values, G.turnaround_21_26.values
    ok2 = ~np.isnan(a2) & ~np.isnan(b2)
    sims2 = [((a2[ok2] > 2) & (rng.permutation(b2[ok2]) > 2)).sum() for _ in range(2000)]
    print(f"\n   Chance baseline (era labels shuffled across groups): persistent in both eras "
          f"{np.mean(sims):.1f} groups on average, turnaround in both {np.mean(sims2):.1f}. "
          f"Counts near these are luck.")

    # ---- E: transitions -------------------------------------------------------
    print(f"\n{'=' * 120}\nE. COLOUR CHANGES (week on week) vs staying in the band -- median pp vs average group\n{'=' * 120}")
    T = L[(L.prev_band != "") & (L.band != "")].assign(change=lambda d: d.prev_band + " -> " + d.band)
    order = ["red -> red", "red -> yellow", "red -> green", "yellow -> red", "yellow -> yellow",
             "yellow -> green", "green -> red", "green -> yellow", "green -> green"]
    Te = summary(T, "change", [f"vsgrp{h}" for h in HORIZONS])
    print(Te.reindex([o for o in order if o in Te.index]).to_string())

    # ---- E2: buy on the day it turns green -----------------------------------
    print(f"\n{'=' * 120}\nE2. BOUGHT AT THE CLOSE ON THE DAY A SUB-INDUSTRY FIRST TURNS GREEN (score crosses 70; daily)"
          f"\n    return over the next 1 / 2 / 4 weeks, pp. mean shows the average buyer; median the typical case.\n{'=' * 120}")
    prev = score.shift(1)
    entry = (score >= 70) & (prev < 70)
    frm = pd.DataFrame(colour(prev), index=score.index, columns=score.columns)
    rows = []
    for h in (5, 10, 21):
        g = (gidx.shift(-h) / gidx - 1) * 100
        nf = (bench.shift(-h) / bench - 1) * 100
        vn, vg = g.sub(nf, axis=0), g.sub(g.mean(axis=1), axis=0)
        e = pd.DataFrame({"from": frm[entry].stack(), "vn": vn[entry].stack(), "vg": vg[entry].stack(),
                          "abs": g[entry].stack()}).dropna()
        e["era"] = np.where(e.index.get_level_values(0) < ERA_SPLIT, "2016-20", "2021-26")
        for key, sub in [("all entries", e)] + [(f"from {b}", e[e["from"] == b]) for b in ("yellow", "red")] \
                        + [(f"all, {er}", e[e.era == er]) for er in ("2016-20", "2021-26")]:
            if len(sub) < 20:
                continue
            rows.append(dict(horizon=f"{h // 5} wk", entry=key, n=len(sub),
                             abs_median=round(sub["abs"].median(), 2), vs_nifty_median=round(sub.vn.median(), 2),
                             vs_nifty_mean=round(sub.vn.mean(), 2), beat_nifty_pct=round((sub.vn > 0).mean() * 100),
                             vs_grp_median=round(sub.vg.median(), 2), vs_grp_mean=round(sub.vg.mean(), 2),
                             beat_grp_pct=round((sub.vg > 0).mean() * 100)))
    # baseline: any sub-industry on any day, same horizons
    for h in (5, 10, 21):
        g = (gidx.shift(-h) / gidx - 1) * 100
        nf = (bench.shift(-h) / bench - 1) * 100
        vn, vg = g.sub(nf, axis=0).stack().dropna(), g.sub(g.mean(axis=1), axis=0).stack().dropna()
        rows.append(dict(horizon=f"{h // 5} wk", entry="baseline: any group, any day", n=len(vn),
                         abs_median=round(g.stack().median(), 2), vs_nifty_median=round(vn.median(), 2),
                         vs_nifty_mean=round(vn.mean(), 2), beat_nifty_pct=round((vn > 0).mean() * 100),
                         vs_grp_median=round(vg.median(), 2), vs_grp_mean=round(vg.mean(), 2),
                         beat_grp_pct=round((vg > 0).mean() * 100)))
    E2 = pd.DataFrame(rows).sort_values(["horizon", "entry"], kind="stable")
    E2.to_csv(f"{OUT}/sector_green_entries.csv", index=False)
    print(E2.to_string(index=False))

    # ---- F: now ---------------------------------------------------------------
    d = cp[-1]
    now = pd.DataFrame({"score": S.loc[d].round(0), "band": colour(S.loc[d]),
                        "weeks_leading": gage.loc[d], "weeks_lagging": rage.loc[d],
                        "score_4wk_ago": S.shift(4).loc[d].round(0), "score_13wk_ago": S.shift(13).loc[d].round(0)})
    now = now.join(G[["lead_persist_16_20", "lead_persist_21_26", "turnaround_16_20", "turnaround_21_26",
                      "persistent_both_eras", "turnaround_both_eras"]]).dropna(subset=["score"])
    now = now.sort_values("score", ascending=False)
    now.to_csv(f"{OUT}/sector_now.csv")
    print(f"\n{'=' * 120}\nF. SUB-INDUSTRIES NOW (as of {d.date()})\n{'=' * 120}")
    print(now.to_string())
    print(f"\nsaved -> {OUT}/sector_group_character.csv, {OUT}/sector_now.csv")


if __name__ == "__main__":
    main()
