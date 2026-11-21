"""Rows of the three observer tables (scenario M1 on the nominal plant; scenarios M2 and M3; perturbed plants).
Sources: results/tables/T_observer_M1_v2_cells.csv, T_observer_noise_v2_cells.csv, T_observer_groups_v2.csv, which
hold the full-precision values of observer/results/metrics_all.csv and summary_by_scenario.csv with the source
row of every cell. Pairs are written '-20% / +20%' (initial estimate of x1 below / above the true value).

Run from the repository root:  python tables/make_tables_observer.py
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

from fmt import num, m
def f(v, integer=False):
    return num(v, integer=integer)
# ---------------- Table 9
c = pd.read_csv(T + "T_observer_M1_v2_cells.csv")
print(c.column.unique(), c.row.unique(), c.position.unique())
def cell9(row, col, integer=False):
    a = c[(c.row == row) & (c.column == col)]
    lo = float(a[a.position == "-20%"].value_full.iloc[0]); hi = float(a[a.position == "+20%"].value_full.iloc[0])
    if integer: return f"{f(lo, True)} / {f(hi, True)}"
    return f"${f(lo)}$ / ${f(hi)}$"
cols9 = list(c.column.unique())
L = []
for sched, lab in (("mu-synthesis", r"$\mu$-synthesis"), ("PID", "PID"), ("SMC", "SMC")):
    for i, (obs, olab) in enumerate((("EKF", "EKF"), ("Luenberger", "Luenberger"))):
        row = f"{sched} / {obs}"
        assert row in set(c.row), row
        cells = [cell9(row, cols9[0], True)] + [cell9(row, k) for k in cols9[1:]]
        L.append((lab if i == 0 else "") + f" & {olab} & " + " & ".join(cells) + r" \\")
    L.append(r"\addlinespace")
L = L[:-1]
open(OUT + "observer_M1_rows.tex", "w").write("\n".join(L) + "\n"); print("\n".join(L)); print(cols9)
# ---------------- Table 10
c = pd.read_csv(T + "T_observer_noise_v2_cells.csv")
print(c.column.unique(), c.row.unique(), c.position.unique())
def v10(row, col, pos):
    a = c[(c.row == row) & (c.column == col) & (c.position == pos)]
    assert len(a) == 1, (row, col, pos, len(a)); return a.iloc[0]
L = []
for meas in c.row.str.split(" / ").str[0].unique():
    for i, obs in enumerate(("EKF", "Luenberger")):
        row = f"{meas} / {obs}"
        conv = v10(row, "Converged runs", "k of 20").displayed
        md = float(v10(row, "Convergence day", "median").value_full)
        md_s = f(md, True) if abs(md - round(md)) < 1e-9 else f(md)
        lg = v10(row, "Convergence day", "largest")
        lg_s = "n.c." if (str(lg.displayed).strip().startswith("n.c") or pd.isna(lg.value_full)) else f(float(lg.value_full), True)
        cells = [conv, f"{md_s} / {lg_s}"]
        for col in [k for k in c.column.unique() if k not in ("Converged runs", "Convergence day")]:
            a = float(v10(row, col, "median").value_full); b = float(v10(row, col, "largest").value_full)
            cells.append(f"${f(a)}$ / ${f(b)}$")
        L.append((meas if i == 0 else "") + f" & {obs} & " + " & ".join(cells) + r" \\")
    L.append(r"\addlinespace")
L = L[:-1]
open(OUT + "observer_noise_rows.tex", "w").write("\n".join(L) + "\n"); print("\n".join(L)); print(list(c.column.unique()))
# ---------------- Table 11 (mu schedule, pairs -20% / +20%)
g = pd.read_csv(T + "T_observer_groups_v2.csv")
print(g.observer.unique(), g.schedule.unique(), g.plant.unique(), g.status.unique())
names = {"nominal": "Nominal", "G1": "G1", "G2": "G2", "G3": "G3", "G4": "G4", "G5": "G5"}
L = []
for pl in ("nominal", "G1", "G2", "G3", "G4", "G5"):
    for i, (ob, olab) in enumerate((("ekf", "EKF"), ("lsd", "Luenberger"))):
        r = {}
        for ie in ("-20%", "+20%"):
            a = g[(g.plant == pl) & (g.observer == ob) & (g.schedule == "mu") & (g.initial_error == ie)]
            assert len(a) == 1; r[ie] = a.iloc[0]
        def pair(key, fmt=f):
            out = []
            for ie in ("-20%", "+20%"):
                x = r[ie]
                out.append("div." if x.status != "ok" else f"${fmt(float(x[key]))}$")
            return " / ".join(out)
        div = " / ".join(("no" if r[ie].status == "ok" else f"day {int(r[ie].diverged_on_day)}") for ie in ("-20%", "+20%"))
        L.append((names[pl] if i == 0 else "") + f" & {olab} & {pair('e1_2000')} & {pair('x1hat_error_2000_pct')} & {pair('e2_2000')} & {div}" + r" \\")
    L.append(r"\addlinespace")
L = L[:-1]
open(OUT + "observer_groups_rows.tex", "w").write("\n".join(L) + "\n"); print("\n".join(L))
# counts of diverged runs over all schedules
d = g[(g.plant != "nominal")]
for ob in ("ekf", "lsd"):
    dd = d[d.observer == ob]
    print(ob, "diverged", int((dd.status != "ok").sum()), "of", len(dd), dd[dd.status != "ok"].groupby("plant").size().to_dict())
# ranges of signed error and e1 over all schedules for the text
for ob in ("ekf", "lsd"):
    for pl in ("G1","G2","G3","G4","G5"):
        dd = d[(d.observer == ob) & (d.plant == pl) & (d.status == "ok")]
        print(ob, pl, "e1 range", dd.e1_2000.min(), dd.e1_2000.max(), "signed% range", dd.x1hat_error_2000_pct.min(), dd.x1hat_error_2000_pct.max(), "e2 max", dd.e2_2000.max(), "x1hat range", dd.x1hat_2000.min(), dd.x1hat_2000.max())
