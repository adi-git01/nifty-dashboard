"""
One-shot stock report: everything the dashboard knows about one stock, in funnel order.

  gather(ticker, df, board, val, regime, simba_row)  -> dict (pure data; no Streamlit, no network)
  markdown(R)                                        -> the report as text (download / paste)
  render(R)                                          -> the Streamlit page

Order: verdict -> funnel checklist -> market and industry -> trend and chart -> entry plan ->
business trends (years, quarters, ownership) -> valuation. All from the daily snapshot, the
weekly Screener files and the valuation snapshot; nothing is fetched, so it opens instantly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils import fund_trends, funnel, industry_board

RISK_PCT = 1.0          # % of capital risked per trade in the sizing example


def _f(v, fmt="{:.1f}", na="—"):
    try:
        return na if v is None or pd.isna(v) else fmt.format(float(v))
    except (TypeError, ValueError):
        return na


def _verdict(c: Dict[str, Optional[bool]], caution: float, wide: bool = False) -> Dict[str, str]:
    if caution and caution >= 3:
        return dict(tag="EXIT / AVOID", tone="error", why="SIMBA caution level 3: the uptrend structure is broken.")
    if not c["trend"]:
        return dict(tag="NOT NOW", tone="error", why="No confirmed uptrend. The funnel only buys strength; revisit when the Trend stage passes.")
    if c["business"] is False:
        return dict(tag="CAUTION", tone="warning", why="The chart is strong but the business stage fails — see the flags under Business trends before acting.")
    if c["setup"] and c["industry"]:
        return dict(tag="ACTIONABLE", tone="success", why="Passes Industry, Trend, Business and Setup. Use the entry plan below and size by the stop.")
    if c["setup"]:
        return dict(tag="ACTIONABLE — WEAK INDUSTRY", tone="info", why="Stock passes on its own but its industry is not leading: a lone strong stock, size smaller.")
    if wide:
        return dict(tag="WAIT FOR A TIGHTER ENTRY", tone="warning", why="There is a buy signal but the stop is more than 15% away: the stock is extended. Wait for a pullback or a new base.")
    return dict(tag="WATCHLIST", tone="info", why="Uptrend and business are in place but there is no entry today. Set an alert and wait for a Technicals BUY or a breakout.")


def gather(ticker: str, df: pd.DataFrame, board: Optional[pd.DataFrame] = None, val: Optional[pd.DataFrame] = None,
           regime: Optional[Dict[str, Any]] = None, simba_row: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    hit = df[df["ticker"] == ticker]
    if hit.empty:
        return dict(ok=False, ticker=ticker, error="Not in the loaded snapshot.")
    r = hit.iloc[0]
    T = funnel.table(hit, board, val, business_for=[ticker]).iloc[0]
    checks = funnel.checklist(T)
    c = dict(zip(["industry", "trend", "business", "setup", "value"], [x["ok"] for x in checks]))
    n = lambda k: pd.to_numeric(r.get(k), errors="coerce")
    px = float(T.price)
    stop, t1, t2 = n("fx_stop"), n("fx_t1"), n("fx_t2")
    risk = (px - stop) / px * 100 if pd.notna(stop) and stop < px else np.nan
    plan = dict(action=str(r.get("fx_action") or "—"), state=str(r.get("fx_state") or "—"), entry=px, stop=stop, risk_pct=risk,
                t1=t1 if pd.notna(t1) and t1 > px else (px + 2 * (px - stop) if pd.notna(risk) else np.nan),
                t2=t2 if pd.notna(t2) and t2 > px else np.nan, half=bool(r.get("fx_half_size")) if pd.notna(r.get("fx_half_size")) else False,
                supply=str(r.get("fx_supply") or "—"), trail=n("simba_trail"),
                size_pct=min(RISK_PCT / risk * 100, 25.0) if pd.notna(risk) and risk > 0 else np.nan)
    vrow = None
    if val is not None and not val.empty and ticker in val.index:
        v = val.loc[ticker]
        vrow = {k: v.get(k) for k in ("status", "rating", "confidence", "cmp_pct", "p10", "p50", "p90", "band_k_lo", "band_k_hi",
                                      "q_grade", "q_note", "cagr_base", "mult_kind", "spot_mult", "base_mult", "reason", "as_of")}
    tech = dict(ret={k: n(f"return_{k}") for k in ("1w", "1m", "3m", "6m", "1y")}, comp_rs=n("comp_rs"), rs_3m=n("rs_3m"),
                dist_52w=n("dist_52w"), ma50=n("fiftyDayAverage"), ma200=n("twoHundredDayAverage"), stage=n("simba_stage"),
                g_bits=str(r.get("simba_g_bits") or ""), m_bits=str(r.get("simba_m_bits") or ""), mtf=n("simba_mtf"),
                caution=n("simba_caution"), scanner=str(r.get("dna_signal") or "—"), trend_score=n("trend_score"),
                rsi=n("fx_rsi"), adx=n("fx_adx"), vol_x=n("simba_vol_x"), bar_date=r.get("bar_date"))
    return dict(ok=True, ticker=ticker, name=str(r.get("name") or ticker), price=px, sector=str(r.get("sector") or ""),
                industry=T.industry, mcap=n("marketCap"), pe=n("pe"), checks=checks, passes=c, verdict=_verdict(c, tech["caution"], bool(T.signal) and pd.notna(T.risk_pct) and T.risk_pct > funnel.MAX_RISK_PCT),
                regime=regime or {}, ind=industry_board.stock_industry_view(board, ticker) if board is not None else None,
                tech=tech, plan=plan, simba=simba_row, val=vrow, trends=fund_trends.read(ticker),
                annual=fund_trends.annual(ticker), quarterly=fund_trends.quarterly(ticker))


# ------------------------------------------------------------------------------------------ tables
def annual_view(a: pd.DataFrame, years: int = 7) -> pd.DataFrame:
    if a.empty:
        return a
    cols = [("sales", "Sales ₹cr"), ("sales_g", "Sales growth %"), ("opm", "OPM %"), ("npm", "Net margin %"), ("roce", "ROCE %"),
            ("roe", "ROE %"), ("cash_conv", "CFO / op. profit %"), ("cash_conv_3y", "…3-yr cumulative %"), ("debt_eq", "Debt / equity"),
            ("int_cover", "Interest cover x"), ("fa_turn", "Fixed-asset turnover x"), ("capex", "Capex ₹cr"),
            ("debtor_days", "Debtor days"), ("inventory_days", "Inventory days"), ("payable_days", "Payable days"),
            ("ccc_days", "Cash conversion cycle (days)"), ("wc_days", "Working-capital days"), ("payout", "Dividend payout %")]
    t = a.tail(years)
    out = pd.DataFrame({lab: t[c] for c, lab in cols if c in t and t[c].notna().any()})
    out.index = [d.strftime("FY%y") if d.month == 3 else d.strftime("%b %y") for d in t.index]
    return out.T.round(1)


def quarterly_view(q: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    if q.empty:
        return q
    cols = [("sales", "Sales ₹cr"), ("sales_yoy", "Sales YoY %"), ("opm", "OPM %"), ("np", "Net profit ₹cr"), ("np_yoy", "Profit YoY %"),
            ("npm", "Net margin %"), ("promoters", "Promoters %"), ("fii", "FII %"), ("dii", "DII %"), ("public", "Public %"),
            ("holders", "Shareholders")]
    t = q.tail(n)
    out = pd.DataFrame({lab: t[c] for c, lab in cols if c in t and t[c].notna().any()})
    out.index = [d.strftime("%b %y") for d in t.index]
    return out.T.round(1)


def _md_table(t: pd.DataFrame) -> str:
    if t is None or t.empty:
        return "_no data_\n"
    head = "| | " + " | ".join(map(str, t.columns)) + " |\n|---|" + "---:|" * len(t.columns) + "\n"
    return head + "".join(f"| {i} | " + " | ".join(_f(v, "{:,.1f}") for v in row) + " |\n" for i, row in zip(t.index, t.values))


def markdown(R: Dict[str, Any]) -> str:
    if not R.get("ok"):
        return f"# {R['ticker']}\n\n{R.get('error', '')}\n"
    t, p, v, ind = R["tech"], R["plan"], R["val"], R["ind"]
    icon = {True: "✅", False: "❌", None: "➖"}
    L: List[str] = [f"# {R['name']} ({R['ticker'].replace('.NS', '')}) — ₹{R['price']:,.2f}",
                    f"_{R['industry'] or R['sector']} · prices as of {str(t['bar_date'])[:10] if t['bar_date'] is not None else 'latest snapshot'}_",
                    "", f"## Verdict: {R['verdict']['tag']}", R["verdict"]["why"], "", "## Funnel checklist"]
    L += [f"- {icon[c['ok']]} **{c['stage']}** — {c['why']}" for c in R["checks"]]
    if R["regime"].get("regime"):
        g = R["regime"]
        L += ["", "## Market", f"Regime **{g['regime']}**" + (f", risk scale {g['risk_scale']}/10" if g.get("risk_scale") else "") + "."]
    if ind:
        L += ["", "## Industry", f"**{ind['industry']}** ranks {ind['rank']} of {ind['of']} (score {_f(ind['score'], '{:.0f}')}, "
              f"{_f(ind['change'], '{:+.0f}')} in a month): {ind['group'].title()}, {ind['status'].title()}. "
              f"{_f(ind['above_50'], '{:.0f}')}% of its {ind['members']} stocks are above the 50-day, {_f(ind['above_200'], '{:.0f}')}% above the 200-day; "
              f"median 3-month return vs Nifty {_f(ind['rs_3m'], '{:+.1f}')} pp. Strongest: {ind['leaders'] or '—'}."]
    L += ["", "## Trend and chart",
          "Returns: " + ", ".join(f"{k} {_f(x, '{:+.1f}')}%" for k, x in t["ret"].items()) + ".",
          f"RS vs Nifty {_f(t['comp_rs'], '{:+.1f}')} (3-month {_f(t['rs_3m'], '{:+.1f}')}); {_f(t['dist_52w'], '{:.0f}')}% from the 52-week high; "
          f"50-day ₹{_f(t['ma50'], '{:,.0f}')}, 200-day ₹{_f(t['ma200'], '{:,.0f}')}; chart stage {_f(t['stage'], '{:.0f}')}; "
          f"Scanner {t['scanner']}; Technicals {p['action']} ({p['state']})."]
    if R["simba"]:
        s = R["simba"]
        L.append(f"SIMBA score {_f(s.get('score'), '{:.0f}')} / 100, grade {s.get('grade', '—')}" + (f"; gates: {s['gates']}" if s.get("gates") else "") + ".")
    L += ["", "## Entry plan"]
    if pd.notna(p["risk_pct"]):
        L += [f"- Entry near ₹{p['entry']:,.2f}; stop ₹{p['stop']:,.2f} ({p['risk_pct']:.1f}% risk)",
              f"- Target 1 ₹{_f(p['t1'], '{:,.2f}')}; Target 2 ₹{_f(p['t2'], '{:,.2f}')}",
              f"- Size: risking {RISK_PCT:.0f}% of capital means about {_f(p['size_pct'], '{:.0f}')}% of capital in this stock" + (" (half size flagged)" if p["half"] else "")]
    else:
        L.append("- No valid stop below the price in today's data: no plan.")
    tr = R["trends"]
    L += ["", f"## Business trends: {tr['verdict']} ({tr['up']} improving, {tr['down']} worsening)"]
    L += [f"- {fund_trends.ICON[f['way']]} **{f['metric']}** — {f['text']}" for f in tr["findings"]] or ["- no fundamentals on file"]
    L += ["", "### Yearly", _md_table(annual_view(R["annual"])), "### Quarterly", _md_table(quarterly_view(R["quarterly"]))]
    L += ["## Valuation"]
    if v and v.get("status") == "OK":
        L.append(f"Value range ₹{_f(v['p10'], '{:,.0f}')} – ₹{_f(v['p50'], '{:,.0f}')} – ₹{_f(v['p90'], '{:,.0f}')} (low / mid / high); the price sits at the "
                 f"{_f(v['cmp_pct'], '{:.0f}')}th percentile. Rating: {v.get('rating')}; confidence {v.get('confidence')}. {v.get('q_note') or ''}")
    else:
        L.append(f"Not valued by the model ({(v or {}).get('status', 'no row')}).")
    L += ["", "_Generated from the dashboard's stored data. The funnel is a checklist, not a backtested system; not investment advice._"]
    return "\n".join(L)
