import pickle, sys, math
import pandas as pd
import numpy as np

sys.path.insert(0, r'C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\repo_live')
sys.path.insert(0, r'C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\tradable500cr_march')
import run_tradable500cr as R
from utils import valuation as V

with open(r'C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\tradable500cr_march\panel_tradable500cr.pkl', 'rb') as f:
    P = pickle.load(f)

all_tks = P['tradable_all']
tail = set(P['tail_ex_n1000'])
n1000 = set(all_tks) - tail
ev_set = set(P['event_dates'])
print(f'Total universe: {len(all_tks)} | In N1000: {len(n1000)} | Tail ex-N1000: {len(tail)}')
for tk in ['GHCLTEXTIL.NS', 'CUPID.NS', 'SIGMAADV.NS', 'BLISSGVS.NS', 'VENUSREM.NS', 'TBZ.NS', 'KRISHANA.NS', 'SHREEJISPG.NS', 'GCSL.NS', 'INDSWFTLAB.NS', 'RPTECH.NS']:
    print(f'  {tk:15s}: {"TAIL" if tk in tail else ("N1000" if tk in n1000 else "OTHER")}')

# 1. Diagnose why Tail had 0 GARP candidates
s0 = P['ckpts']['2026-02-27']
n1k_bear = sum(1 for tk in n1000 if tk in s0 and s0[tk].get('bear_val') is not None)
tail_bear = sum(1 for tk in tail if tk in s0 and s0[tk].get('bear_val') is not None)
print(f'\n2026-02-27 bear_val non-null -> N1000: {n1k_bear}/{len(n1000)} | Tail: {tail_bear}/{len(tail)}')

vctx = V.load_context(root=r'C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\repo_live', extended=True)
a_tks = set(vctx.annual['ticker'].astype(str))
q_tks = set(vctx.quarterly['ticker'].astype(str))
f_tks = set(vctx.funda.index.astype(str))
m_tks = set(vctx.monthly.columns.astype(str))
print(f'vctx totals -> annual:{len(a_tks)} quarterly:{len(q_tks)} funda:{len(f_tks)} monthly:{len(m_tks)}')
print(f'  tail overlap -> annual:{len(a_tks & tail)} quarterly:{len(q_tks & tail)} funda:{len(f_tks & tail)} monthly:{len(m_tks & tail)}')
print(f'  n1k  overlap -> annual:{len(a_tks & n1000)} quarterly:{len(q_tks & n1000)} funda:{len(f_tks & n1000)} monthly:{len(m_tks & n1000)}')

# Check what V.value_stock returns on 3 tail stocks that ARE in vctx.annual
sample_tail = sorted(a_tks & tail)[:5]
print('Sample tail in vctx.annual:', sample_tail)
for tk in sample_tail:
    c0 = float(P['C'][0, P['ticker_to_col'][tk]])
    try:
        res = V.value_stock(tk, vctx, price=c0)
        print(f'  {tk}: verdict={res.get("verdict")} bear={res.get("bear_val")} err={res.get("error")} msg={str(res)[:120]}')
    except Exception as e:
        print(f'  {tk}: EXC={e}')

# 2. Check PIT-start active counts & return distribution (survivorship/graduation audit)
print('\n--- ACTIVE COUNTS (>=500Cr & >=2Cr/d) ACROSS TIME ---')
for dt in ['2026-02-27', '2026-05-18', '2026-08-31', '2026-11-06']:
    sn = P['ckpts'][dt]
    a_all = [tk for tk, v in sn.items() if v['pit_mcap_cr'] >= 500.0 and v['liq_cr'] >= 2.0]
    a_n1k = [tk for tk in a_all if tk in n1000]
    a_tl = [tk for tk in a_all if tk in tail]
    print(f'  {dt}: Active ALL={len(a_all)} | N1000={len(a_n1k)} | Tail={len(a_tl)}')

t2c = P['ticker_to_col']
C = P['C']
def dist_stats(tks, label):
    rets = []
    for tk in tks:
        c = C[:, t2c[tk]]
        if np.isfinite(c[0]) and c[0] > 0 and np.isfinite(c[-1]):
            rets.append((c[-1]/c[0] - 1)*100)
    arr = np.array(rets)
    print(f'  {label:28s} (n={len(arr):4d}): mean={arr.mean():+6.2f}% | med={np.median(arr):+6.2f}% | >+50%={(arr>50).sum():3d} ({(arr>50).mean()*100:4.1f}%) | <-25%={(arr<-25).sum():3d} ({(arr<-25).mean()*100:4.1f}%)')

print('\n--- PIT-START (>=500Cr & >=2Cr on 2026-02-27) RETURN DISTRIBUTION ---')
elig_n1k_0 = [tk for tk in n1000 if tk in s0 and s0[tk]['pit_mcap_cr'] >= 500 and s0[tk]['liq_cr'] >= 2.0]
elig_tl_0  = [tk for tk in tail  if tk in s0 and s0[tk]['pit_mcap_cr'] >= 500 and s0[tk]['liq_cr'] >= 2.0]
dist_stats(elig_n1k_0 + elig_tl_0, 'ALL PIT-Start')
dist_stats(elig_n1k_0, 'N1000 PIT-Start')
dist_stats(elig_tl_0,  'SMALLCAP_TAIL PIT-Start')

# 3. Optimized 3-Sleeve Barbells + PEAD liquidity sweep + MOMQ(20Cr) attribution
r_garp2  = R.run_sim(P, all_tks, 'garp_guarded', decision_dates=ev_set, mcap_floor_cr=500.0, liq_floor_cr=2.0, max_pos_pct=10.0, cooldown_days=60)
r_garp20 = R.run_sim(P, all_tks, 'garp_guarded', decision_dates=ev_set, mcap_floor_cr=500.0, liq_floor_cr=20.0, max_pos_pct=10.0, cooldown_days=60)
r_mom2   = R.run_sim(P, all_tks, 'mom_quality_ranked', decision_dates=ev_set, mcap_floor_cr=500.0, liq_floor_cr=2.0, max_pos_pct=10.0, use_inv_vol=True, max_per_industry=2, use_regime_throttle=True, cooldown_days=30)
r_mom5   = R.run_sim(P, all_tks, 'mom_quality_ranked', decision_dates=ev_set, mcap_floor_cr=500.0, liq_floor_cr=5.0, max_pos_pct=10.0, use_inv_vol=True, max_per_industry=2, use_regime_throttle=True, cooldown_days=30)
r_mom20  = R.run_sim(P, all_tks, 'mom_quality_ranked', decision_dates=ev_set, mcap_floor_cr=500.0, liq_floor_cr=20.0, max_pos_pct=10.0, use_inv_vol=True, max_per_industry=2, use_regime_throttle=True, cooldown_days=30)
r_mom20_5k = R.run_sim(P, all_tks, 'mom_quality_ranked', decision_dates=ev_set, mcap_floor_cr=5000.0, liq_floor_cr=20.0, max_pos_pct=10.0, use_inv_vol=True, max_per_industry=2, use_regime_throttle=True, cooldown_days=30)
r_pead2  = R.run_sim(P, all_tks, 'pead_calibrated', decision_dates=ev_set, mcap_floor_cr=500.0, liq_floor_cr=2.0, max_pos_pct=10.0, use_inv_vol=True, max_per_industry=2, use_regime_throttle=True, cooldown_days=30)
r_pead5  = R.run_sim(P, all_tks, 'pead_calibrated', decision_dates=ev_set, mcap_floor_cr=500.0, liq_floor_cr=5.0, max_pos_pct=10.0, use_inv_vol=True, max_per_industry=2, use_regime_throttle=True, cooldown_days=30)
r_pead20 = R.run_sim(P, all_tks, 'pead_calibrated', decision_dates=ev_set, mcap_floor_cr=500.0, liq_floor_cr=20.0, max_pos_pct=10.0, use_inv_vol=True, max_per_industry=2, use_regime_throttle=True, cooldown_days=30)

print('\n--- PEAD LIQUIDITY SWEEP ---')
for lbl, r in [('PEAD >=2Cr', r_pead2), ('PEAD >=5Cr', r_pead5), ('PEAD >=20Cr', r_pead20)]:
    print(f"  {lbl:12s} | Ret: {r['total_ret_pct']:+6.2f}% | DD: {r['max_dd_pct']:6.2f}% | Sharpe: {r['sharpe']:5.3f} | Cash: {r['cash_drag_mean']:4.1f}% | Buys: {r['buys']} | Top3: {r['top5'][:3]}")

def bb(g, m, p, label, w=(0.50, 0.30, 0.20)):
    comb = w[0]*g['curve'] + w[1]*m['curve'] + w[2]*p['curve']
    c_ret = (float(comb.iloc[-1])/1e6 - 1)*100
    c_dd = float((comb/comb.cummax() - 1).min()*100)
    dr = comb.pct_change().dropna()
    c_sh = float((dr.mean()/dr.std(ddof=1))*math.sqrt(252)) if dr.std(ddof=1)>0 else 0
    corr = pd.DataFrame({'G': g['curve'].pct_change(), 'M': m['curve'].pct_change(), 'P': p['curve'].pct_change()}).dropna().corr().round(2)
    gm, gp, mp = corr.loc['G', 'M'], corr.loc['G', 'P'], corr.loc['M', 'P']
    print(f'  {label:48s} | Ret: {c_ret:+6.2f}% | DD: {c_dd:6.2f}% | Sharpe: {c_sh:5.3f} | GM={gm:.2f} GP={gp:.2f} MP={mp:.2f}')

print('\n--- OPTIMIZED 3-SLEEVE BARBELLS (TRADABLE >=500Cr) ---')
bb(r_garp2,  r_mom5,  r_pead2,  'Default: 50% GARP(2Cr)+30% MOMQ(5Cr)+20% PEAD(2Cr)')
bb(r_garp2,  r_mom2,  r_pead2,  'All-2Cr: 50% GARP(2Cr)+30% MOMQ(2Cr)+20% PEAD(2Cr)')
bb(r_garp2,  r_mom20, r_pead2,  'Split  : 50% GARP(2Cr)+30% MOMQ(20Cr)+20% PEAD(2Cr)')
bb(r_garp2,  r_mom20, r_pead20, 'OptLiq : 50% GARP(2Cr)+30% MOMQ(20Cr)+20% PEAD(20Cr)')
bb(r_garp20, r_mom20, r_pead20, 'Inst20 : 50% GARP(20Cr)+30% MOMQ(20Cr)+20% PEAD(20Cr)')
bb(r_garp2,  r_mom20_5k, r_pead20, 'MidMom : 50% GARP(2Cr)+30% MOMQ(5k/20)+20% PEAD(20Cr)')
bb(r_garp2,  r_mom20, r_pead20, 'Eq40/40/20: GARP(2Cr)+MOMQ(20Cr)+PEAD(20Cr)', w=(0.40, 0.40, 0.20))

print('\n--- MOMQ(5000Cr / 20Cr) ---')
print(f"  MOMQ(5000Cr/20Cr): Ret {r_mom20_5k['total_ret_pct']:+6.2f}% | DD {r_mom20_5k['max_dd_pct']:6.2f}% | Sharpe {r_mom20_5k['sharpe']:5.3f} | Top5: {r_mom20_5k['top5']}")
