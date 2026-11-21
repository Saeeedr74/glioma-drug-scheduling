"""Rows of the table of metrics (three controllers and the maximum-dose baseline) and of the replay-test table.
Sources: data/metrics_all.csv (controllers), results/tables/baseline_metrics_v2.json (baseline with the stop rule
'first day with x2 < 1e-4'), results/tables/T6_uncertainty.csv. Bold: best of the three controllers, plus values
within 2 days (day metrics) or 1% (other metrics) of the best.

Run from the repository root:  python tables/make_table_controller_summary.py
Output: LaTeX table rows in results/tables/paper/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
CW = os.path.join(ROOT, "")
T = os.path.join(ROOT, "results", "tables", "")
OUT = os.path.join(ROOT, "results", "tables", "paper", "")
os.makedirs(OUT, exist_ok=True)
import json

import numpy as np
import pandas as pd

from fmt import num, sci, m
M = pd.read_csv(CW + "data/metrics_all.csv")
B = json.load(open(CW + "results/tables/baseline_metrics_v2.json"))
def row(set_, ctrl, group="nominal"):
    r = M[(M["set"] == set_) & (M.controller == ctrl) & (M.group == group)]
    assert len(r) == 1; return r.iloc[0]
C = {c: row("nominal_discrete", c) for c in ("PID", "SMC", "mu")}
S = {c: row("E1_summary", c, "mean_G1_G5") for c in ("PID", "SMC", "mu")}
order = ("PID", "SMC", "mu")
def bold_set(vals, better, kind):
    """vals: dict ctrl->float. better: 'low'/'high'. kind: 'day' (within 2 days) or 'rel' (within 1%)."""
    best = min(vals.values()) if better == "low" else max(vals.values())
    out = set()
    for c, v in vals.items():
        if kind == "day":
            ok = abs(v - best) <= 2
        else:
            ok = abs(v - best) <= 0.01 * abs(best)
        if ok: out.add(c)
    return out
def cells(key, src, integer=False, better=None, kind="rel", fmt=None):
    vals = {c: float(src[c][key]) for c in order}
    b = bold_set(vals, better, kind) if better else set()
    out = []
    for c in order:
        t = (fmt or (lambda v: num(v, integer=integer)))(vals[c])
        out.append(f"$\\mathbf{{{t}}}$" if c in b else f"${t}$")
    return out, vals, b
L = []; log = []
def add(label, base, key, src=C, **kw):
    cs, vals, b = cells(key, src, **kw)
    L.append(f"{label} & {base} & " + " & ".join(cs) + r" \\")
    log.append((label, base, {c: repr(vals[c]) for c in order}, sorted(b)))
dash = "--"
L.append(r"\multicolumn{5}{l}{\textit{Tumor suppression}} \\")
add(r"First day with $x_2<10^{-4}$ (day)", f"${int(B['t_x2_first'])}$", "t_x2_first", integer=True, better="low", kind="day")
add(r"$x_2(T_{\mathrm{ref}})$", "$0$", "x2_T", better="low")
L.append(r"\addlinespace"); L.append(r"\multicolumn{5}{l}{\textit{Healthy cells}} \\")
add(r"Glial recovery to 99\% of capacity (day)", f"${int(B['t_x1_perm'])}$", "t_x1_perm", integer=True, better="low", kind="day")
add(r"$x_3(T_{\mathrm{ref}})$", m(num(B["x3_T"])), "x3_T", better="high")
L.append(r"\addlinespace"); L.append(r"\multicolumn{5}{l}{\textit{Drug dose}} \\")
add(r"Cumulative dose ($\mathrm{mg\,m^{-2}}$)", f"${int(B['cum_dose'])}$", "cum_dose", integer=True, better="low")
add(r"Mean daily dose ($\mathrm{mg\,m^{-2}\,day^{-1}}$)", m(num(B["mean_dose"])), "mean_dose", better="low")
add(r"Number of high-dose phases", dash, "n_hi", integer=True)
add(r"Mean high-dose phase (days)", f"${int(B['hi_mean'])}$", "hi_mean")
add(r"Longest high-dose phase (days)", f"${int(B['hi_max'])}$", "hi_max", integer=True)
add(r"High-dose phases longer than 5 days", dash, "n_hi_gt5", integer=True)
add(r"Number of maintenance phases between high-dose phases", dash, "n_lo_between", integer=True)
add(r"Mean maintenance phase (days)", dash, "lo_between_mean", better="high")
add(r"Last high-dose day", f"${int(B['last_hi_day'])}$", "last_hi_day", integer=True)
L.append(r"\addlinespace"); L.append(r"\multicolumn{5}{l}{\textit{Tracking}} \\")
add(r"Tracking error (RMS), relative to the initial tumor level", dash, "rmse_over_x2pre", better="low")
L.append(r"\addlinespace"); L.append(r"\multicolumn{5}{l}{\textit{Sensitivity to parameter changes, replay test}} \\")
add(r"Sensitivity factor (mean of the 5 groups)", dash, "sensitivity_factor", src=S, better="low")
add(r"Largest $x_2(T_{\mathrm{ref}})$ over the 5 groups", dash, "x2_T_worst", src=S, better="low")
open(OUT + "controller_summary_rows.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L))
print()
for l in log: print(l)

# ---- Table 8 (replay test)
T6 = pd.read_csv(CW + "results/tables/T6_uncertainty.csv")
def t6(c, g):
    r = T6[(T6.controller == c) & (T6.group == g)]; assert len(r) == 1; return r.iloc[0]
names = {"G1": r"G1: $\Omega_1 \downarrow,\ \Omega_2 \uparrow$ & $-20\%$, $+20\%$",
         "G2": r"G2: $\beta_1 \uparrow,\ \beta_2 \downarrow$ & $+50\%$, $-50\%$",
         "G3": r"G3: $p_1 \uparrow,\ p_2 \downarrow$ & $+15\%$, $-15\%$",
         "G4": r"G4: $a_1 \uparrow,\ a_2 \uparrow$ & $+20\%$, $+20\%$",
         "G5": r"G5: $\zeta \uparrow$ & $+20\%$"}
R = [r"Nominal & -- & " + " & ".join(m(num(t6(c, "nominal").x2_T)) for c in order) + r" \\"]
for g in ("G1", "G2", "G3", "G4", "G5"):
    R.append(names[g] + " & " + " & ".join(f"{m(num(t6(c, g).x2_T))} ({num(t6(c, g).ratio_to_nominal)})" for c in order) + r" \\")
R.append(r"\midrule")
R.append(r"Sensitivity factor (mean ratio) & -- & " + " & ".join(m(num(S[c].sensitivity_factor)) for c in order) + r" \\")
R.append(r"Mean $x_2(T_{\mathrm{ref}})$ of the five groups & -- & " + " & ".join(m(num(S[c].x2_T_mean_G1_G5)) for c in order) + r" \\")
open(OUT + "replay_test_rows.tex", "w").write("\n".join(R) + "\n")
print("\n".join(R))
# consistency: T6 vs metrics_all
for c in order:
    for g in ("G1","G2","G3","G4","G5"):
        assert abs(t6(c,g).x2_T - row("E1", c, g).x2_T) == 0
    assert abs(np.mean([t6(c,g).ratio_to_nominal for g in ("G1","G2","G3","G4","G5")]) - S[c].sensitivity_factor) < 1e-9
print("T6 == metrics_all E1 rows; SF = mean of ratios: OK")
