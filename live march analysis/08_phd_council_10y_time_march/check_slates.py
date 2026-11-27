import pickle, sys
sys.path.insert(0, r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\phd_council_10y")
with open(r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\phd_council_10y\blind_10y_council_panel.pkl", "rb") as f:
    P = pickle.load(f)
for gi in range(1, 23):
    gs = P["gate_snaps"][gi]
    stocks, macro = gs["stocks"], gs["macro"]
    pool = list(stocks.values())
    m_cnt = sum(1 for s in pool if s["buy_now"] and (s["fy_np_yoy"] is None or s["fy_np_yoy"]>0) and s["liq_cr"]>=(15 if macro["mode"] in ("BEAR","CAUTION") else 3))
    c_cnt = sum(1 for s in pool if s["stage"]==2 and s["fy_np_yoy"] is not None and s["fy_np_yoy"]>=15 and (s["is_bfsi"] or (s["roce"] is not None and s["roce"]>=18 and (s["cfo_pat"] is None or s["cfo_pat"]>=0.65))))
    g_cnt = sum(1 for s in pool if s["cmp_lt_bear"] and s["stage"] in (1,2) and s["fy_np_yoy"] is not None and s["fy_np_yoy"]>0 and (s["is_bfsi"] or (s["cfo_pat"] is not None and s["cfo_pat"]>=0.70)))
    cr_cnt = sum(1 for s in pool if s["fy_np_yoy"] is not None and s["fy_np_yoy"]>=10 and (s["is_bfsi"] or (s["roce"] is not None and s["roce"]>=18 and s["cfo_pat"] is not None and s["cfo_pat"]>=0.75)) and s["liq_cr"]>=10)
    print(f"E{gi:02d} {gs['date'].date()} ({macro['mode']:8s}): MOM={m_cnt:3d} COMP={c_cnt:3d} GARP={g_cnt:2d} CRASH_Q={cr_cnt:3d}")
