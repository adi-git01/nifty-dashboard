"""
Replay of the Nifty 500 from a start date (default 1 Jul 2026) to today: what each way of acting on
the dashboard would have returned, per stock, using only what was known on each day.

    hold        buy at the start date's close, hold to today
    sl_hold     Signal Lab: buy at the close of the first scanner signal on/after the start (Turnaround
                Catalyst, Breakouts, RS Divergence, Earnings Shock, Turnaround Radar, V2.2 watchlist
                BUY), hold to today (Signal Lab gives no exits)
    sl_exit     the same entries with the technical exit rule; re-enter on the next signal
    tech        Technicals: buy on the single-stock report's BUY NOW (v6 forensics on the bars up to
                that day), exit on 2 closes < MA50 or close < 0.85 x peak since entry; re-enter on the
                next BUY NOW
    techval     Technicals + valuation: buy only on a combined BUY (BUY NOW and valuation not
                expensive), same exits plus a combined SELL. Valuation uses only results published by
                that day (annual +60 days, quarterly +45, March quarter +60) and monthly prices up to
                that day; peer data is today's (not point-in-time).

Prices are Yahoo daily closes (split-adjusted); trades fill at the signal day's close, no costs.
Money not invested earns nothing. Output: analysis/signal_replay_stocks.csv, signal_replay_trades.csv,
and a summary on stdout (and analysis/SIGNAL_REPLAY.md).

    python signal_replay.py [--start 2026-07-01] [--limit N]
"""
from __future__ import annotations

import argparse
import copy
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from utils.forensics import analyse
from utils.valuation import combined_signal, load_context, signal_bucket, val_bucket, value_stock

OUT_S, OUT_T, OUT_MD = "analysis/signal_replay_stocks.csv", "analysis/signal_replay_trades.csv", "analysis/SIGNAL_REPLAY.md"
LOGS = [("tc_log.csv", "signal_date", "Turnaround Catalyst"), ("breakout_log.csv", "breakout_date", "Breakout"),
        ("rs_divergence_log.csv", "signal_date", "RS Divergence"), ("earnings_shock_log.csv", "signal_date", "Earnings Shock"),
        ("ias_signal_log.csv", "signal_date", "Turnaround Radar"), ("watchlist_history.csv", "added_date", "V2.2 watchlist BUY")]


def signals(start):
    L = []
    for f, dc, src in LOGS:
        p = os.path.join("data", f)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        tc = "ticker" if "ticker" in d else "Ticker"
        L.append(pd.DataFrame(dict(stock=d[tc].astype(str).str.replace(".NS", "", regex=False),
                                   date=pd.to_datetime(d[dc].astype(str).str[:10], errors="coerce"), src=src)))
    D = pd.concat(L).dropna(subset=["date"])
    return D[D.date >= start].sort_values("date")


def bars(tickers, start):
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 50):
        part = [t + ".NS" for t in tickers[i:i + 50]]
        for attempt in range(3):
            try:
                h = yf.download(part, start=start - pd.DateOffset(years=2, months=1), auto_adjust=False, progress=False,
                                group_by="ticker", threads=True)
                break
            except Exception as e:
                print(f"[yahoo] retry {attempt + 1}: {e}", flush=True)
                time.sleep(10)
        else:
            continue
        for t in part:
            try:
                d = h[t][["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
            except Exception:
                continue
            if len(d) > 260:
                d.index = pd.to_datetime(d.index).tz_localize(None)
                out[t[:-3]] = d
        print(f"[yahoo] {min(i + 50, len(tickers))}/{len(tickers)}: {len(out)} with data", flush=True)
    return out


def exit_hit(c, ma50, since, i):
    return (c[i] < ma50[i] and c[i - 1] < ma50[i - 1]) or c[i] < 0.85 * c[since:i + 1].max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-07-01")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    START = pd.Timestamp(a.start)
    U = pd.read_csv("data/nifty500_list.csv").Ticker.astype(str).str.replace(".NS", "", regex=False).tolist()
    if a.limit:
        U = U[:a.limit]
    H = bars(U, START)
    import yfinance as yf
    nf = yf.download("^NSEI", start=START - pd.Timedelta(days=7), auto_adjust=False, progress=False)["Close"].squeeze().dropna()
    nifty = float(nf.iloc[-1] / nf[nf.index >= START].iloc[0] - 1) * 100
    D = signals(START)
    base = load_context()
    A_all, Q_all = base.annual, base.quarterly

    rows, trades = [], []
    t0 = time.time()
    for k, n in enumerate(sorted(H)):
        h = H[n]
        c = h.Close.values.astype(float)
        ma50 = h.Close.rolling(50).mean().values
        ma200 = h.Close.rolling(200).mean().values
        days = [i for i, d in enumerate(h.index) if d >= START]
        if not days:
            continue
        t = n + ".NS"
        A0, Q0 = A_all[A_all.ticker == t], Q_all[Q_all.ticker == t]
        qlag = pd.to_timedelta(np.where(Q0.period.dt.month == 3, 60, 45), unit="D")

        def val(D_, i):
            ctx = copy.copy(base)
            ctx.annual = A0[A0.field.isin(["mcap_cr", "price_scr"]) | (A0.period + pd.Timedelta(days=60) <= D_)]
            ctx.quarterly = Q0[Q0.period + qlag <= D_]
            ctx.monthly = base.monthly[base.monthly.index <= D_]
            hh = h.iloc[max(0, i - 251):i + 1]
            try:
                v = value_stock(n, ctx, price=float(c[i]), n=1500, hi52=float(hh.High.max()), lo52=float(hh.Low.min()))
            except Exception:
                return "", 3, "ERROR"
            if v.get("status") != "OK":
                return "", 3, v.get("status")
            return val_bucket(v["dist"]["cmp_pct"]), v["quality"]["sev"], "OK"

        pos = {"T": None, "TV": None}
        eq = {"T": 1.0, "TV": 1.0}
        inv = {"T": 0, "TV": 0}
        nbuy, ncomb, vstat = 0, 0, {}
        for i in days:
            r = analyse(h.iloc[:i + 1])
            tech_buy = r.get("action") == "BUY NOW"
            nbuy += tech_buy
            need_val = tech_buy or pos["TV"] is not None
            comb = "HOLD"
            if need_val:
                vb, sev, st_ = val(h.index[i], i)
                vstat[st_] = vstat.get(st_, 0) + 1
                comb, _ = combined_signal(tech_buy, bool(c[i] < ma50[i] and c[i] < ma200[i]), signal_bucket(vb, sev))
                ncomb += comb == "BUY"
            for m in ("T", "TV"):
                p = pos[m]
                if p is not None:
                    inv[m] += 1
                if p is None:
                    if tech_buy if m == "T" else comb == "BUY":
                        pos[m] = i
                elif exit_hit(c, ma50, p, i) or (m == "TV" and comb == "SELL"):
                    eq[m] *= c[i] / c[p]
                    trades.append(dict(stock=n, mode=m, buy=h.index[p].date(), buy_px=c[p], sell=h.index[i].date(),
                                       sell_px=c[i], ret=(c[i] / c[p] - 1) * 100, days=i - p))
                    pos[m] = None
        for m in ("T", "TV"):
            p = pos[m]
            if p is not None:
                eq[m] *= c[-1] / c[p]
                trades.append(dict(stock=n, mode=m, buy=h.index[p].date(), buy_px=c[p], sell="open", sell_px=c[-1],
                                   ret=(c[-1] / c[p] - 1) * 100, days=len(c) - 1 - p))
        rec = dict(stock=n, hold=(c[-1] / c[days[0]] - 1) * 100, buy_now_days=nbuy, comb_buy_days=ncomb,
                   tech=(eq["T"] - 1) * 100, techval=(eq["TV"] - 1) * 100, tech_inv_days=inv["T"], techval_inv_days=inv["TV"],
                   days=len(days), val_status=";".join(f"{k_}:{v_}" for k_, v_ in vstat.items()))
        g = D[D.stock == n]
        if len(g):
            sig = sorted({min(h.index.searchsorted(d), len(c) - 1) for d in g.date if d <= h.index[-1]})
            if sig:
                i0 = sig[0]
                rec.update(sl_date=h.index[i0].date(), sl_src=g.iloc[0].src, n_signals=len(g), sl_hold=(c[-1] / c[i0] - 1) * 100)
                e, p = 1.0, i0
                for j in range(i0 + 1, len(c)):
                    if p is None:
                        if j in sig:
                            p = j
                    elif exit_hit(c, ma50, p, j):
                        e *= c[j] / c[p]
                        p = None
                if p is not None:
                    e *= c[-1] / c[p]
                rec["sl_exit"] = (e - 1) * 100
        rows.append(rec)
        if (k + 1) % 25 == 0:
            print(f"[replay] {k + 1}/{len(H)} stocks, {time.time() - t0:.0f}s", flush=True)

    S, T = pd.DataFrame(rows), pd.DataFrame(trades)
    os.makedirs("analysis", exist_ok=True)
    S.round(2).to_csv(OUT_S, index=False)
    T.round(2).to_csv(OUT_T, index=False)
    summarise(S, T, nifty, START)


def summarise(S, T, nifty, START):
    cols = {"hold": "Hold every stock", "sl_hold": "Signal Lab: first signal, held", "sl_exit": "Signal Lab entries + technical exit",
            "tech": "Technicals only", "techval": "Technicals + valuation"}
    L = [f"# Signal replay: Nifty 500 from {START:%d %b %Y} to {pd.Timestamp.today():%d %b %Y}", "",
         f"{len(S)} stocks with Yahoo data. Nifty 50 over the same days: {nifty:+.1f}%.", "",
         "| Approach | Stocks acted on | Avg return per stock (all stocks, 0 if never bought) | Avg return on stocks acted on | Median (acted on) | Share of acted-on stocks up | Days invested (share of stock-days) |",
         "|---|---|---|---|---|---|---|"]
    for k, lab in cols.items():
        if k not in S:
            continue
        if k == "hold":
            acted = S[k].dropna()
        elif k.startswith("sl"):
            acted = S[k].dropna()
        else:
            acted = S.loc[S[f"{k}_inv_days"] > 0, k]
        allv = S[k].fillna(0)
        inv = (f"{S[f'{k}_inv_days'].sum() / S['days'].sum():.0%}" if f"{k}_inv_days" in S else "")
        L.append(f"| {lab} | {len(acted)} | {allv.mean():+.1f}% | {acted.mean():+.1f}% | {acted.median():+.1f}% | "
                 f"{(acted > 0).mean():.0%} | {inv} |")
    if len(T):
        L += ["", "**Trades** (technical modes)", "", "| Mode | Trades | Avg | Median | Win rate | Avg days |", "|---|---|---|---|---|---|"]
        for m, lab in (("T", "Technicals only"), ("TV", "Technicals + valuation")):
            t = T[T["mode"] == m]
            if len(t):
                L.append(f"| {lab} | {len(t)} | {t.ret.mean():+.1f}% | {t.ret.median():+.1f}% | {(t.ret > 0).mean():.0%} | {t.days.mean():.0f} |")
    # what the valuation veto did: technical trades it blocked vs the ones it allowed
    if len(T):
        tv = T[T["mode"] == "TV"]
        tt = T[T["mode"] == "T"].copy()
        end = lambda x: pd.Timestamp.today() if x == "open" else pd.Timestamp(x)
        # allowed = valuation let a position open at some point during this technical trade
        tt["allowed"] = [bool(((tv.stock == s) & (pd.to_datetime(tv.buy) >= pd.Timestamp(b)) &
                               (pd.to_datetime(tv.buy) <= end(e))).any()) for s, b, e in zip(tt.stock, tt.buy, tt.sell)]
        L += ["", "**Technical BUY NOW trades, split by whether valuation let a position open during the trade**", "",
              "| | Trades | Avg | Median | Win rate |", "|---|---|---|---|---|"]
        for lab, t in (("Allowed by valuation", tt[tt.allowed]), ("Blocked by valuation", tt[~tt.allowed])):
            if len(t):
                L.append(f"| {lab} | {len(t)} | {t.ret.mean():+.1f}% | {t.ret.median():+.1f}% | {(t.ret > 0).mean():.0%} |")
    # big movers: who caught them
    top, bot = S.nlargest(25, "hold"), S.nsmallest(25, "hold")
    L += ["", "**Top 25 and bottom 25 stocks by holding return: average captured by each approach**", "",
          "| Group | Hold | Signal Lab held | Technicals | Tech + valuation |", "|---|---|---|---|---|"]
    for lab, g in (("Top 25", top), ("Bottom 25", bot)):
        L.append(f"| {lab} | {g.hold.mean():+.1f}% | {g.sl_hold.fillna(0).mean():+.1f}% | {g.tech.mean():+.1f}% | {g.techval.mean():+.1f}% |")
    L += ["", "Fills at the signal day's close, no costs, equal weight per stock, uninvested money earns nothing. "
          "Valuation peers are today's. Signal Lab has no exits of its own."]
    txt = "\n".join(L)
    open(OUT_MD, "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
