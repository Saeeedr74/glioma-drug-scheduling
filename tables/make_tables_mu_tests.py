"""Rows of the two tables of the appendix 'Supplementary results': mu-synthesis in closed loop on the perturbed plants
and the random-plant test. Sources: results/tables/T_app_mu_E2.csv, T_app_mu_monte_carlo.csv (MATLAB exports).

Run from the repository root:  python tables/make_tables_mu_tests.py
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
import pandas as pd

from fmt import num

e = pd.read_csv(T + "T_app_mu_E2.csv")
names = {"nominal": "Nominal", "G1": "G1", "G2": "G2", "G3": "G3", "G4": "G4", "G5": "G5"}
L = []
for _, r in e.iterrows():
    L.append(f"{names[r.group]} & ${num(r.x2_T_E1_replay)}$ & ${num(r.x2_T_E2_closed_loop)}$ & {int(r.t_x2_first)} & {int(r.cum_dose)} & "
             f"{int(r.n_hi)} & {int(r.hi_max)} & {int(r.n_hi_gt5)} & {int(r.n_redesigns)}" + r" \\")
open(OUT + "mu_closed_loop_rows.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L))
mc = pd.read_csv(T + "T_app_mu_monte_carlo.csv")
lab = {"replay, N=300": "Schedule replay, 300 plants", "replay, first 20": "Schedule replay, first 20 plants",
       "closed loop, first 20": "Controller in the loop, first 20 plants"}
L = []
for _, r in mc.iterrows():
    hi = "--" if pd.isna(r.maxHi_mean) else f"${num(r.maxHi_mean)}$ / {int(r.maxHi_worst)}"
    n5 = "--" if pd.isna(r.n_runs_with_phase_gt5) else str(int(r.n_runs_with_phase_gt5))
    L.append(f"{lab[r.case]} & ${num(r.x2T_median)}$ & ${num(r.x2T_worst)}$ & {int(r.reach)}/{int(r.N)} & "
             f"${num(r.reach_ci95_lo)}$ to ${num(r.reach_ci95_hi)}$ & {hi} & {n5}" + r" \\")
open(OUT + "mu_random_plants_rows.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L))
