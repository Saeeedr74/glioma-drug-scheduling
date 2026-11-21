"""small comparison table (discrete SMC, 5/28 at 200, 5/28 at 400) with model/glioma_paper/metrics.py.
Reads the regenerated results/runs/smc_simulation.csv (simulation_timeseries section), cross-checks against
data/metrics_all.csv. Run from the root of the repository:  python tables/build_protocol_table.py"""
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
from glioma_paper.metrics import metrics  # noqa: E402

runs = pd.read_csv(os.path.join(ROOT, "results", "runs", "smc_simulation.csv"))
ts = runs[runs.section == "simulation_timeseries"]
ref_m = pd.read_csv(os.path.join(ROOT, "data", "metrics_all.csv"))

CASES = [("SMC (discrete, nominal)", "smc_discrete", "nominal_discrete", "SMC"),
         ("5/28, 200 mg m^-2", "standard_5_28_dose200", "standard_protocol", "standard_5_28_dose200"),
         ("5/28, 400 mg m^-2", "standard_5_28_dose400", "standard_protocol", "standard_5_28_dose400")]
rows, checks = [], []
for label, run, mset, mctl in CASES:
    w = ts[ts.run == run].pivot(index="row", columns="name", values="value").sort_index()
    X = w[["x1", "x2", "x3", "x4"]].to_numpy()
    u = w["u"].to_numpy()[:2000]                      # doses on days 0..1999 (day-2000 entry is NaN)
    m = metrics(X, u, discrete=(run == "smc_discrete"))
    rows.append(dict(row=label, total_dose_mg_m2=m["cum_dose"], mean_daily_dose_mg_m2=m["mean_dose"],
                     x1_T=m["x1_T"], x2_T=m["x2_T"], x3_T=m["x3_T"],
                     first_day_x2_below_1e4=("not reached" if math.isnan(m["t_x2_first"]) else int(m["t_x2_first"])),
                     min_x3=m["x3_min"], x2_day0=float(X[0, 1])))
    s = ref_m[(ref_m.set == mset) & (ref_m.controller == mctl)].iloc[0]
    for k_new, k_ref in (("cum_dose", "cum_dose"), ("mean_dose", "mean_dose"), ("x1_T", "x1_T"), ("x2_T", "x2_T"),
                         ("x3_T", "x3_T"), ("t_x2_first", "t_x2_first"), ("x3_min", "x3_min")):
        a, b = m[k_new], float(s[k_ref])
        both_nan = math.isnan(a) and math.isnan(b)
        checks.append((label, k_new, a, b, 0.0 if both_nan else abs(a - b),
                       0.0 if both_nan else abs(a - b)/max(abs(b), 1e-300)))
df = pd.DataFrame(rows)
out_csv = os.path.join(ROOT, "results", "tables", "T_protocol_comparison.csv")
df.drop(columns=["x2_day0"]).to_csv(out_csv, index=False, float_format="%.17g")
chk = pd.DataFrame(checks, columns=["row", "metric", "new", "stored_metrics_all", "abs_diff", "rel_diff"])
chk.to_csv(os.path.join(ROOT, "tables", "protocol_table_crosscheck.csv"), index=False, float_format="%.17g")
print(chk.to_string())
print("max abs diff", chk.abs_diff.max(), "max rel diff", chk.rel_diff.max())
print("x2(0) =", df.x2_day0.tolist())


def num(x, integer=False):
    if integer:
        return str(int(round(x)))
    if abs(x) >= 0.01:
        return f"{x:.2f}"
    mant, ex = f"{x:.2e}".split("e")
    return rf"${mant}\times10^{{{int(ex)}}}$"


lines = [r"\begin{tabular}{lrrccccc}", r"\toprule",
         r"Run & Total dose & Mean dose & $x_1(2000)$ & $x_2(2000)$ & $x_3(2000)$ & First day $x_2<10^{-4}$ & $\min x_3$ \\",
         r" & (mg\,m$^{-2}$) & (mg\,m$^{-2}$\,day$^{-1}$) & & & & & \\", r"\midrule"]
names = {"SMC (discrete, nominal)": "SMC", "5/28, 200 mg m^-2": r"5/28 at 200", "5/28, 400 mg m^-2": r"5/28 at 400"}
for _, r in df.iterrows():
    fd = r["first_day_x2_below_1e4"]
    lines.append(" & ".join([names[r["row"]], num(r.total_dose_mg_m2, True), num(r.mean_daily_dose_mg_m2, True),
                             num(r.x1_T), num(r.x2_T), num(r.x3_T),
                             fd if isinstance(fd, str) else str(fd), num(r.min_x3)]) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
with open(os.path.join(ROOT, "results", "tables", "T_protocol_comparison.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
