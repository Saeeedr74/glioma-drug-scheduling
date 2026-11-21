"""E5.3: Table 12 with two sub-columns per controller (days with dW/dt > 1e-10, first such day).
Sources: results/tables/T_app_lyapunov_groups.csv (stored table; columns n_days_wdot_pos, first_day_wdot_pos,
m, beta2, condition_m_lt_beta2) and, as a cross-check, the regenerated results/figure_data/lyapunov_wdot_summary.csv
(protocol E1_replay) and the day-by-day series results/figure_data/lyapunov_W_timeseries.csv (column Wdot).
Run from the root of the repository:  python tables/build_lyapunov_groups_v2.py"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-10
st = pd.read_csv(os.path.join(ROOT, "results", "tables", "T_app_lyapunov_groups.csv"))
sm = pd.read_csv(os.path.join(ROOT, "results", "figure_data", "lyapunov_wdot_summary.csv"))
sm = sm[sm.protocol == "E1_replay"]
ts = pd.read_csv(os.path.join(ROOT, "results", "figure_data", "lyapunov_W_timeseries.csv"))
ts = ts[ts.protocol == "E1_replay"]

GROUPS = ["nominal", "G1", "G2", "G3", "G4", "G5"]
CTL = ["PID", "SMC", "mu"]
# 1) stored table vs regenerated summary vs recount from the time series
mism = []
for c in CTL:
    for g in GROUPS:
        a = st[(st.controller == c) & (st.group == g)].iloc[0]
        b = sm[(sm.controller == c) & (sm.group == g)].iloc[0]
        t = ts[(ts.controller == c) & (ts.group == g)].sort_values("time_day")
        bad = np.flatnonzero(t.Wdot.to_numpy() > TOL)
        n_ts, f_ts = int(bad.size), (float(bad[0]) if bad.size else np.nan)
        for what, v1, v2 in (("n", a.n_days_wdot_pos, b.n_days_wdot_pos), ("n_vs_timeseries", a.n_days_wdot_pos, n_ts),
                             ("first", a.first_day_wdot_pos, b.first_day_wdot_pos), ("first_vs_timeseries", a.first_day_wdot_pos, f_ts)):
            if not (v1 == v2 or (pd.isna(v1) and pd.isna(v2))):
                mism.append((c, g, what, v1, v2))
        for k in ("m", "beta2", "condition_m_lt_beta2"):
            if a[k] != b[k]:
                mism.append((c, g, k, a[k], b[k]))
print("mismatches stored table vs regenerated summary / time series:", mism)

# 2) rebuilt table
rows = []
for g in GROUPS:
    r = dict(group=g)
    a0 = st[(st.controller == "PID") & (st.group == g)].iloc[0]
    r.update(m=a0.m, beta2=a0.beta2, m_lt_beta2=bool(a0.condition_m_lt_beta2))
    for c in CTL:
        a = st[(st.controller == c) & (st.group == g)].iloc[0]
        assert a.m == a0.m and a.beta2 == a0.beta2
        r[f"{c}_days_wdot_gt_1e-10"] = int(a.n_days_wdot_pos)
        r[f"{c}_first_day"] = (int(a.first_day_wdot_pos) if pd.notna(a.first_day_wdot_pos) else "")
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "tables", "T_lyapunov_groups_v2.csv"), index=False, float_format="%.17g")
print(df.to_string())

LABEL = {"nominal": "Nominal",
         "G1": r"G1: $\Omega_1 \downarrow,\ \Omega_2 \uparrow$",
         "G2": r"G2: $\beta_1 \uparrow,\ \beta_2 \downarrow$",
         "G3": r"G3: $p_1 \uparrow,\ p_2 \downarrow$",
         "G4": r"G4: $a_1 \uparrow,\ a_2 \uparrow$",
         "G5": r"G5: $\zeta \uparrow$"}
L = [r"\begin{tabular}{lcrrrrrr}", r"\toprule",
     r" & & \multicolumn{2}{c}{PID} & \multicolumn{2}{c}{SMC} & \multicolumn{2}{c}{$\mu$-synthesis} \\",
     r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
     r"Plant & $m<\beta_2$ & days & first day & days & first day & days & first day \\", r"\midrule"]
for _, r in df.iterrows():
    cells = [LABEL[r.group], "yes" if r.m_lt_beta2 else "no"]
    for c in CTL:
        cells += [str(r[f"{c}_days_wdot_gt_1e-10"]), (str(r[f"{c}_first_day"]) if r[f"{c}_first_day"] != "" else "--")]
    L.append(" & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "results", "tables", "T_lyapunov_groups_v2.tex"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

# 3) nominal: max dW/dt per controller (all 2001 samples), tolerance
for c in CTL:
    t = ts[(ts.controller == c) & (ts.group == "nominal")]
    s = sm[(sm.controller == c) & (sm.group == "nominal")].iloc[0]
    print(f"nominal {c}: samples {len(t)}, max Wdot (time series) {t.Wdot.max():.6e} on day {int(t.time_day.iloc[int(np.argmax(t.Wdot.to_numpy()))])}, "
          f"summary max_wdot {s.max_wdot:.6e}, days > {TOL:g}: {int((t.Wdot > TOL).sum())}, days >= 0: {int((t.Wdot >= 0).sum())}")
