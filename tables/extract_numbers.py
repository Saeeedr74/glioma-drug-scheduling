"""numbers the new text needs -> results/tables/extracted_numbers.csv.
Run from the root of the repository:  python tables/extract_numbers.py"""
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
from glioma_paper.model import NOMINAL, rhs  # noqa: E402

FD = "results/figure_data/"
ND = FD + "nominal_discrete.csv"
BL = FD + "baseline_max_dose.csv"
rows = []


def add(q, who, val, unit, f, col, rd, how):
    rows.append(dict(quantity=q, controller_or_group=who, value=val, unit=unit, source_file=f,
                     source_column=col, source_row_or_day=rd, how_computed=how))


def first_below(x, th):
    i = np.flatnonzero(x < th)
    return float(i[0]) if i.size else math.nan


def runs_of(mask):
    m = np.concatenate(([0], mask.astype(int), [0]))
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1     # start day, end day (inclusive)


def stats3(v):
    v = np.asarray(v, dtype=float)
    return (float(v.min()), float(np.median(v)), float(v.max())) if v.size else (math.nan,) * 3


nd = pd.read_csv(os.path.join(ROOT, ND))
bl = pd.read_csv(os.path.join(ROOT, BL))
ZETA = NOMINAL["zeta"]
DIM = "normalized"

# ---------------- states of the three controllers and of the baseline ----------------
sets = [(c, nd[nd.controller == c].sort_values("time_day"), ND) for c in ("PID", "SMC", "mu")] + \
       [("max-dose baseline", bl.sort_values("time_day"), BL)]
for who, t, f in sets:
    assert len(t) == 2001 and (t.time_day.to_numpy() == np.arange(2001)).all()
    X = t[["x1", "x2", "x3", "x4"]].to_numpy()
    u = t.u.to_numpy()
    x1, x2, x3, x4 = X.T
    rd = f"controller/run={who}"
    # ---- x1
    k = int(np.argmin(x1))
    add("x1 minimum", who, float(x1[k]), DIM, f, "x1", f"{rd}, time_day={k}", "min over days 0..2000 (first index of the minimum)")
    add("x1 minimum, day", who, float(k), "day", f, "x1", f"{rd}, time_day={k}", "argmin of x1 over days 0..2000")
    d = np.diff(x1)
    kd = int(np.flatnonzero(d > 0)[0])
    add("x1: last day before the first rise (day-to-day difference)", who, float(kd), "day", f, "x1", f"{rd}, time_day={kd}->{kd + 1}",
        "smallest k with x1[k+1]-x1[k] > 0 (x1 is higher on day k+1 than on day k); x1 falls on every earlier day")
    add("x1: first day with a rise over the previous day", who, float(kd + 1), "day", f, "x1", f"{rd}, time_day={kd + 1}",
        "smallest day k+1 with x1[k+1] > x1[k] (= previous row + 1)")
    dx1 = np.array([rhs(tuple(x), 0.0, NOMINAL)[0] for x in X])      # dx1/dt depends on x1,x2,x4 only (u irrelevant)
    kr = int(np.flatnonzero(dx1[1:] > 0)[0]) + 1
    add("dx1/dt on day 0", who, float(dx1[0]), "1/day", f, "x1,x2,x4", f"{rd}, time_day=0", "model right-hand side dx1/dt (model/glioma_paper/model.py rhs) evaluated at the stored state of the day; negative = decline")
    add("dx1/dt: first day positive after day 0 (model right-hand side)", who, float(kr), "day", f, "x1,x2,x4", f"{rd}, time_day={kr}",
        "smallest day >= 1 with dx1/dt > 0, dx1/dt = Om1*x1*(1-x1) - b1*x1*x2 - p1*x1*x4/(a1+x1) at the stored state (nominal parameters)")
    add("days with dx1/dt < 0 after that first positive day", who, float((dx1[kr:] < 0).sum()), "days", f, "x1,x2,x4", f"{rd}, days {kr}..2000",
        "count of days >= first positive day with dx1/dt < 0 (x1 falls again after later dose pulses if > 0)")
    add("x1 at day 2000", who, float(x1[-1]), DIM, f, "x1", f"{rd}, time_day=2000", "value on day 2000")
    # ---- x3
    k3 = int(np.argmin(x3))
    add("x3 at day 0", who, float(x3[0]), DIM, f, "x3", f"{rd}, time_day=0", "value on day 0")
    add("x3 minimum", who, float(x3[k3]), DIM, f, "x3", f"{rd}, time_day={k3}", "min over days 0..2000 (first index)")
    add("x3 minimum, day", who, float(k3), "day", f, "x3", f"{rd}, time_day={k3}", "argmin of x3 over days 0..2000")
    add("x3 at day 2000", who, float(x3[-1]), DIM, f, "x3", f"{rd}, time_day=2000", "value on day 2000")
    add("x3 on the day of the x1 minimum", who, float(x3[k]), DIM, f, "x3", f"{rd}, time_day={k}", f"x3[{k}], where {k} = argmin of x1")
    # ---- x2
    add("x2 at day 0", who, float(x2[0]), DIM, f, "x2", f"{rd}, time_day=0", "value on day 0")
    for th, nm in ((1e-1, "1e-1"), (1e-2, "1e-2"), (1e-3, "1e-3"), (1e-4, "1e-4")):
        v = first_below(x2, th)
        add(f"x2: first day below {nm}", who, v, "day", f, "x2", f"{rd}, time_day={'' if math.isnan(v) else int(v)}",
            f"smallest day with x2 < {nm} (NaN: never)")
    add("x2 at day 2000", who, float(x2[-1]), DIM, f, "x2", f"{rd}, time_day=2000", "value on day 2000")
    if who == "max-dose baseline":
        add("baseline: last day with u = 200", who, float(np.flatnonzero(u == 200.0)[-1]), "day", f, "u", rd, "largest day index with u == 200 (u = 0 afterwards, u NaN on day 2000)")
        continue
    # ---- x4
    ka = int(np.argmax(x4))
    add("x4 highest peak", who, float(x4[ka]), "mg m^-2", f, "x4", f"{rd}, time_day={ka}", "max over days 0..2000")
    add("x4 highest peak, day", who, float(ka), "day", f, "x4", f"{rd}, time_day={ka}", "argmax of x4 over days 0..2000")
    add("x4 at day 2000", who, float(x4[-1]), "mg m^-2", f, "x4", f"{rd}, time_day=2000", "value on day 2000")
    ud = u[:2000]
    hs, he = runs_of(ud == 200.0)
    ls, le = runs_of(ud == 50.0)
    assert (ud == 200.0).sum() + (ud == 50.0).sum() == 2000
    # peak of a high-dose phase: max of x4 over days s..e+1 (e+1 = state after the last high dose);
    # valley of a maintenance phase: min of x4 over days s..e+1.  Last maintenance phase ends at the horizon (open-ended) -> excluded from valleys.
    peaks = np.array([x4[s:e + 2].max() for s, e in zip(hs, he)])
    peak_start = hs
    closed = le < 1999
    valleys = np.array([x4[s:e + 2].min() for s, e in zip(ls[closed], le[closed])])
    valley_start = ls[closed]
    add("number of high-dose phases (u=200)", who, float(hs.size), "phases", f, "u", rd, "runs of u == 200 on days 0..1999")
    add("number of maintenance phases (u=50)", who, float(ls.size), "phases", f, "u", rd, "runs of u == 50 on days 0..1999 (includes the open-ended last phase and, for mu, the phase starting on day 0)")
    add("number of maintenance phases ended by a high-dose phase", who, float(closed.sum()), "phases", f, "u", rd, "maintenance runs whose last day is < 1999; the last open-ended phase is excluded from the valley statistics")
    for sel_name, pm, vm in (("all phases", np.ones(peaks.size, bool), np.ones(valleys.size, bool)),
                              ("phases starting on day >= 50", peak_start >= 50, valley_start >= 50)):
        for nm, vals in (("peak", peaks[pm]), ("valley", valleys[vm])):
            for stat, v in zip(("min", "median", "max"), stats3(vals)):
                add(f"x4 per-phase {nm} ({sel_name}): {stat}", who, v, "mg m^-2", f, "x4", rd,
                    ("per high-dose phase (maximal run of u=200): max of x4 over its dose days s..e and day e+1" if nm == "peak" else
                     "per maintenance phase (maximal run of u=50) that is followed by a high-dose phase: min of x4 over its dose days s..e and day e+1")
                    + f"; selection: {sel_name} (by first dose day of the phase); {stat} over {int(pm.sum() if nm == 'peak' else vm.sum())} phases")
    # peak/valley of the last complete high-dose phase and the first/last maintenance valleys for context
    add("x4 last high-dose phase: peak", who, float(peaks[-1]), "mg m^-2", f, "x4", f"{rd}, last u=200 run days {hs[-1]}..{he[-1]}", "max of x4 over days s..e+1 of the last u=200 run")
for lvl in (50.0, 200.0):
    add(f"x4 steady level under u={lvl:g} forever", "all", lvl / ZETA, "mg m^-2", "model/glioma_paper/model.py", "NOMINAL['zeta']=0.2", "-", "u/zeta (dx4/dt = u - zeta*x4 = 0)")

# ---------------- uncertainty (T6) ----------------
t6 = pd.read_csv(os.path.join(ROOT, "results/tables/T6_uncertainty.csv"))
ma = pd.read_csv(os.path.join(ROOT, "data/metrics_all.csv"))
e1 = ma[ma.set == "E1"]
e1g = pd.read_csv(os.path.join(ROOT, FD + "e1_uncertainty.csv.gz"))
T6F = "results/tables/T6_uncertainty.csv"
start = float(nd[(nd.controller == "PID") & (nd.time_day == 0)].x2.iloc[0])
for c in ("SMC", "mu"):
    assert float(nd[(nd.controller == c) & (nd.time_day == 0)].x2.iloc[0]) == start
add("starting x2 (day 0)", "all", start, DIM, ND, "x2", "time_day=0 (identical for PID, SMC, mu)", "value on day 0 (state after 600 days without treatment)")
cells = t6[t6.group.isin(["nominal", "G1", "G2", "G3", "G4", "G5"])]
xcheck = []
xcheck_ts = []
for _, r in cells.iterrows():
    c, g = r.controller, r.group
    xe = float(e1[(e1.controller == c) & (e1.group == g)].x2_T.iloc[0]) if g != "nominal" else float(ma[(ma.set == "nominal_discrete") & (ma.controller == c)].x2_T.iloc[0])
    xcheck.append(abs(xe - r.x2_T))
    add("final x2 (day 2000)", f"{c}, {g}", float(r.x2_T), DIM, T6F, "x2_T", f"controller={c}, group={g}",
        "stored value; equals data/metrics_all.csv (E1 / nominal_discrete) x2_T")
    add("reduction factor start/final x2", f"{c}, {g}", start / float(r.x2_T), "-", T6F, "x2_T", f"controller={c}, group={g}",
        "x2(day 0) / x2(day 2000), x2(day 0) from nominal_discrete.csv")
    add("percent reduction of x2 from the starting level", f"{c}, {g}", 100*(1 - float(r.x2_T)/start), "%", T6F, "x2_T", f"controller={c}, group={g}",
        "100*(1 - x2(2000)/x2(0))")
g15 = cells[cells.group.isin(["G1", "G2", "G3", "G4", "G5"])]
w = g15.loc[g15.x2_T.idxmax()]
add("worst case: highest final x2 over G1..G5 and PID, SMC, mu", f"{w.controller}, {w.group}", float(w.x2_T), DIM, T6F, "x2_T", f"controller={w.controller}, group={w.group}", "max of x2_T over the 15 group x controller cells (nominal row excluded)")
add("worst case: reduction factor start/final", f"{w.controller}, {w.group}", start/float(w.x2_T), "-", T6F, "x2_T", f"controller={w.controller}, group={w.group}", "x2(0)/x2(2000)")
add("worst case: percent reduction from the starting level", f"{w.controller}, {w.group}", 100*(1 - float(w.x2_T)/start), "%", T6F, "x2_T", f"controller={w.controller}, group={w.group}", "100*(1 - x2(2000)/x2(0))")
all_nom = cells
wn = all_nom.loc[all_nom.x2_T.idxmax()]
add("check: highest final x2 including the nominal rows", f"{wn.controller}, {wn.group}", float(wn.x2_T), DIM, T6F, "x2_T", f"controller={wn.controller}, group={wn.group}", "max over all 18 cells (equals the 15-cell maximum when group is not nominal)")
for c in ("PID", "SMC", "mu"):
    s = g15[g15.controller == c]
    w_c = s.loc[s.x2_T.idxmax()]
    add("worst group per controller: final x2", f"{c}, {w_c.group}", float(w_c.x2_T), DIM, T6F, "x2_T", f"controller={c}, group={w_c.group}", "max over G1..G5")
    add("worst group per controller: percent reduction", f"{c}, {w_c.group}", 100*(1 - float(w_c.x2_T)/start), "%", T6F, "x2_T", f"controller={c}, group={w_c.group}", "100*(1 - x2(2000)/x2(0))")
    n_end = int((s.x2_T < 1e-4).sum())
    n_first = int(s.t_x2_first.notna().sum())
    add("number of groups G1..G5 with final x2 < 1e-4", c, float(n_end), "groups of 5", T6F, "x2_T", f"controller={c}, groups={','.join(s[s.x2_T < 1e-4].group)}", "count of G1..G5 with x2_T < 1e-4")
    add("number of groups G1..G5 that reach x2 < 1e-4 on some day", c, float(n_first), "groups of 5", T6F, "t_x2_first", f"controller={c}, groups={','.join(s[s.t_x2_first.notna()].group)}", "count of G1..G5 with a non-empty t_x2_first")
    # cross-check with the day-2000 value in the E1 time series
    for g in ("G1", "G2", "G3", "G4", "G5"):
        v = float(e1g[(e1g.controller == c) & (e1g.group == g) & (e1g.time_day == 2000)].x2.iloc[0])
        xcheck_ts.append(abs(v - float(s[s.group == g].x2_T.iloc[0])))
print("max |T6 x2_T - metrics_all x2_T| over the 18 cells:", max(xcheck))
print("max |T6 x2_T - e1_uncertainty.csv.gz x2(day 2000)| over 15 cells:", max(xcheck_ts))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results/tables/extracted_numbers.csv"), index=False, float_format="%.17g")
print(len(df), "rows written")
