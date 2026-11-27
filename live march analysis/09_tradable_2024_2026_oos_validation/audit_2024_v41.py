import os, sys
import pandas as pd, numpy as np

REPO_L = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\repo_live"
os.chdir(REPO_L)
sys.path.insert(0, REPO_L)
from utils import valuation as V, fund_trends as ft

vctx = V.load_context(extended=True)
ann_raw, qtr_raw = ft._data(REPO_L)
sample_tks = sorted(set(vctx.funda.index) & set(vctx.annual["ticker"]))[:100]

print(f"Testing {len(sample_tks)} sample stocks across 2024-2025 as_of dates:")
for dt in ["2024-01-31", "2024-05-31", "2024-08-30", "2024-09-30", "2024-11-15", "2025-01-31"]:
    ts = pd.Timestamp(dt)
    # Check V.value_stock with strict PIT filtering on quarterly/annual/monthly
    q_sub = vctx.quarterly[vctx.quarterly["period"] <= ts - pd.Timedelta(days=45)]
    a_sub = vctx.annual[vctx.annual["period"] <= ts - pd.Timedelta(days=60)]
    m_sub = vctx.monthly.loc[:ts]
    sub_ctx = V.Context(a_sub, q_sub, m_sub, vctx.funda, vctx.overrides, vctx.peer_cagr3, vctx.features)
    sub_ctx.peer_mult = vctx.peer_mult

    v_ok = q_yoy_ok = 0
    for tk in sample_tks:
        px = float(m_sub[tk].dropna().iloc[-1]) if tk in m_sub.columns and not m_sub[tk].dropna().empty else 100.0
        res = V.value_stock(tk, sub_ctx, price=px, n=2000, as_of=ts)
        if res.get("status") == "OK":
            v_ok += 1
        kq = ft._key(tk, qtr_raw)
        if kq is not None:
            wq = qtr_raw[kq]
            wq = wq[wq["period"] <= ts - pd.Timedelta(days=45)]
            np_s = wq[wq["field"] == "np"].set_index("period")["value"].sort_index()
            if len(np_s) >= 5:
                q_yoy_ok += 1
    print(f"  {dt}: V.value_stock OK = {v_ok:3d}/100 | Q_NP_YoY (>=5 qtrs) = {q_yoy_ok:3d}/100")
