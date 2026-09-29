"""
Sub-industry colour changes: do they predict the next days, weeks, months?
==========================================================================

The rotation heatmap colours each of the 58 sub-industries by score_0_100 (its
percentile rank across groups; trading_engine.generate_sub_industry_rotation):
green >= 70 (Leader), yellow 40-69 (Mid), red < 40 (Laggard). The stored history
starts in Mar 2025 -- one year, too short to judge -- so this rebuilds the same
score daily for ~10 years with the same formula (0.7 x mean member CompRS
percentile + 0.3 x % of members with CompRS > 0, point-in-time top-1000
members, groups of >= 3) and checks it against the stored scores.

A colour change is read two ways:
  weekly   the band now vs 5 sessions ago (what the 7D column shows)
  monthly  the band at month end vs the previous month end (the heatmap)

For every change (red->yellow, yellow->green, green->yellow, ...) and every
"stayed" case, the group's forward return is measured over 5 / 10 / 21 / 63 /
126 sessions against:
  - Nifty (what the question asks), and
  - the average sub-industry on the same day (removes the market's move, so it
    isolates whether picking THIS group helped).

Forward windows of the same group overlap, so the event count overstates the
evidence; results are also split by era (2016-20 vs 2021-26) so one period
cannot carry them.

Run: python sector_transition_study.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

HORIZONS = [5, 10, 21, 63, 126]
BANDS = [(70, "green"), (40, "yellow"), (-1, "red")]
OUT = "analysis"


def band(score):
    b = pd.DataFrame(np.select([score >= 70, score >= 40, score >= 0], ["green", "yellow", "red"], ""),
                     index=score.index, columns=score.columns)
    return b.where(score.notna(), "")


def forward(gidx, bench, h):
    """Group return over the next h sessions vs Nifty and vs the average group, in pp."""
    g = (gidx.shift(-h) / gidx - 1) * 100
    n = (bench.shift(-h) / bench - 1) * 100
    return g.sub(n, axis=0), g.sub(g.mean(axis=1), axis=0)


def events(bands, step):
    """(date, group, from, to) at every `step`-session checkpoint."""
    prev = bands.shift(step)
    return prev, bands


def table(ev):
    rows = []
    groups = [(("all", "all"), ev)] + list(ev.groupby(["from", "to"]))
    for (a, b), g in groups:
        r = dict(change="all group-dates" if a == "all" else f"{a} -> {b}", events=len(g))
        for h in HORIZONS:
            x, c = g[f"x{h}"].dropna(), g[f"c{h}"].dropna()
            if len(x) < 20:
                continue
            r[f"{h}d vs Nifty"] = round(x.median(), 2)
            r[f"{h}d vs avg grp"] = round(c.median(), 2)
            r[f"{h}d beat grp %"] = round((c > 0).mean() * 100)
        rows.append(r)
    order = ["all group-dates", "red -> red", "red -> yellow", "red -> green", "yellow -> red", "yellow -> yellow",
             "yellow -> green", "green -> red", "green -> yellow", "green -> green"]
    t = pd.DataFrame(rows).set_index("change")
    return t.reindex([o for o in order if o in t.index])


def main():
    from momentum_factor_backtest import build_pit_universe, load_candidates
    from transfer_backtest import fetch_prices, group_panels
    from utils.nifty1000_list import SUB_INDUSTRY_MAP

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    close, vol, bench = fetch_prices(load_candidates("all", args.max_tickers), "^NSEI", "2014-06-01")
    mask = build_pit_universe(close, vol, 1000)
    gret, keys, n = group_panels(close, mask, SUB_INDUSTRY_MAP, bench,
                                 [(5, .10), (21, .50), (63, .40)], india_live=True)
    strength = keys["live"]
    score = strength.rank(axis=1, pct=True) * 100           # score_0_100, as the live engine ranks it
    score = score[score.notna().sum(axis=1) >= 20]
    gidx = (1 + gret.fillna(0)).cumprod().where(n >= 3).reindex(score.index)
    bench = bench.reindex(score.index)
    bands = band(score)
    print(f"[score] {score.shape[1]} sub-industries, {len(score)} sessions "
          f"{score.index[0].date()} -> {score.index[-1].date()}")

    # --- validate against the stored live history ---------------------------
    col = lambda v: np.select([v >= 70, v >= 40], ["green", "yellow"], "red")
    try:
        live = pd.read_csv("data/sub_industry_rotation.csv")
        live["record_date"] = pd.to_datetime(live.record_date, format="mixed").dt.normalize()
        st = score.stack()
        st.index.names = ["record_date", "sub_industry"]
        m = live.merge(st.rename("rebuilt").reset_index(), on=["record_date", "sub_industry"])
        if len(m):
            print(f"[validate] {len(m)} stored group-days since {m.record_date.min().date()}: "
                  f"rank corr {m[['score_0_100', 'rebuilt']].corr(method='spearman').iloc[0, 1]:.2f}, "
                  f"same colour {(col(m.score_0_100) == col(m.rebuilt)).mean() * 100:.0f}%")
        else:
            print("[validate] no overlapping dates with the stored history")
    except Exception as e:
        print(f"[validate] skipped: {e}")

    fx = {h: forward(gidx, bench, h) for h in HORIZONS}
    allrows = []
    for name, step, idx in (("weekly", 5, score.index[::5]),
                            ("monthly", None, score.groupby(score.index.to_period("M")).tail(1).index)):
        prev = bands.shift(5) if step else bands.reindex(idx).shift(1)
        cur = bands.reindex(idx)
        prev = prev.reindex(idx)
        ev = pd.DataFrame({"from": prev.stack(), "to": cur.stack()}).reset_index()
        ev.columns = ["date", "group", "from", "to"]
        ev = ev[(ev["from"] != "") & (ev["to"] != "")]
        for h in HORIZONS:
            x, c = fx[h]
            ev[f"x{h}"] = x.stack().reindex(pd.MultiIndex.from_frame(ev[["date", "group"]])).values
            ev[f"c{h}"] = c.stack().reindex(pd.MultiIndex.from_frame(ev[["date", "group"]])).values
        ev["cadence"] = name
        allrows.append(ev)

        print(f"\n{'=' * 110}\n{name.upper()} COLOUR CHANGES -> forward return (median, pp). "
              f"'vs avg grp' removes the market's move.\n{'=' * 110}")
        t = table(ev)
        cols = [c for h in (5, 21, 63, 126) for c in (f"{h}d vs Nifty", f"{h}d vs avg grp", f"{h}d beat grp %")
                if c in t]
        print(t[["events"] + cols].to_string())
        # by era, the two upgrades and the two downgrades
        ev["era"] = np.where(ev.date < "2021-01-01", "2016-20", "2021-26")
        e = (ev[ev["from"] != ev["to"]].assign(change=lambda d: d["from"] + " -> " + d["to"])
             .groupby(["change", "era"]).agg(n=("c21", "size"), vs_grp_21d=("c21", "median"),
                                             vs_grp_63d=("c63", "median"), vs_nifty_63d=("x63", "median"))
             .round(2))
        print("\n  by era (21d / 63d, median pp):")
        print(e.to_string())

    E = pd.concat(allrows)
    E.to_csv(f"{OUT}/sector_transitions.csv", index=False)

    # --- current state ------------------------------------------------------
    d = score.index[-1]
    cur = pd.DataFrame({"score": score.loc[d].round(0), "band": bands.loc[d],
                        "band_1w_ago": bands.iloc[-6], "band_1m_ago": bands.iloc[-22],
                        "score_1w_ago": score.iloc[-6].round(0), "score_1m_ago": score.iloc[-22].round(0)})
    cur = cur[cur.band != ""].sort_values("score", ascending=False)
    ch = cur[(cur.band != cur.band_1w_ago) | (cur.band != cur.band_1m_ago)]
    print(f"\nCHANGED COLOUR IN THE LAST WEEK OR MONTH (as of {d.date()}):")
    print(ch.to_string())
    cur.to_csv(f"{OUT}/sector_bands_now.csv")
    print(f"\nsaved -> {OUT}/sector_transitions.csv, {OUT}/sector_bands_now.csv")


if __name__ == "__main__":
    main()
