"""Convert blind_10y_council_panel.pkl into 100% pure Python stdlib pickle (zero numpy/pandas)."""
import os, pickle, time
import numpy as np

OUT_DIR = r"C:\Users\adity\.gemini\antigravity\brain\40c6abb7-9cd2-4dc5-b585-90209394eacd\scratch\phd_council_10y"
t0 = time.time()
with open(os.path.join(OUT_DIR, "blind_10y_council_panel.pkl"), "rb") as f:
    P = pickle.load(f)

# Convert arrays to compact per-ticker float lists (or list-of-lists with None for NaN? Keep NaN as float('nan') — native Python!)
def mat_to_cols(arr):
    # Store column-major (per ticker) list of Python floats — only for tickers that actually appear in gate_snaps!
    return arr

# Collect all tickers ever active across the 22 gates so the stdlib pickle stays lean & lightning-fast
active_codes = set()
for gi, gs in P["gate_snaps"].items():
    active_codes.update(gs["stocks"].keys())
active_tks = sorted({P["code_to_tk"][c] for c in active_codes})
t2c = P["t2c"]
O, L, C = P["O"], P["L"], P["C"]

# Pure Python dicts: prices[tk] = (o_list, l_list, c_list)
prices = {}
for tk in active_tks:
    col = t2c[tk]
    prices[tk] = (
        [float(x) for x in O[:, col].tolist()],
        [float(x) for x in L[:, col].tolist()],
        [float(x) for x in C[:, col].tolist()],
    )

sim_dates = [str(d.date()) for d in P["sim_dates"]]
gate_dates = [str(d.date()) for d in P["gate_dates"]]
gate_snaps = {}
for gi, gs in P["gate_snaps"].items():
    # Sanitize every value in macro and stocks to pure Python types
    mac = {k: (float(v) if isinstance(v, (np.floating,)) else int(v) if isinstance(v, (np.integer,)) else v) for k, v in gs["macro"].items()}
    stks = {}
    for code, s in gs["stocks"].items():
        stks[code] = {k: (bool(v) if isinstance(v, (np.bool_,)) else float(v) if isinstance(v, (np.floating,)) else int(v) if isinstance(v, (np.integer,)) else v) for k, v in s.items()}
    gate_snaps[int(gi)] = dict(gate_idx=int(gi), date=str(gs["date"].date()), di=int(gs["di"]), macro=mac, stocks=stks)

n50_list = [float(x) for x in P["n50"].reindex(P["sim_dates"]).ffill().tolist()]
ew_list = [float(x) for x in P["ew"].reindex(P["sim_dates"]).ffill().tolist()]

out = dict(
    sim_dates=sim_dates,
    gate_dates=gate_dates,
    gate_snaps=gate_snaps,
    code_to_tk=dict(P["code_to_tk"]),
    prices=prices,
    n50=n50_list,
    ew=ew_list,
)
out_p = os.path.join(OUT_DIR, "blind_10y_stdlib.pkl")
with open(out_p, "wb") as f:
    pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Exported pure-stdlib panel: {len(active_tks)} tickers x {len(sim_dates)} days -> {os.path.getsize(out_p)/1e6:.1f} MB in {time.time()-t0:.1f}s")
