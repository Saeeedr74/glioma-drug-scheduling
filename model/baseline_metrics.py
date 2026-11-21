"""Metrics of the maximum-dose baseline (stop rule: first day with x2 < 1e-4), computed with the metric
function of the paper (model/glioma_paper/metrics.py).

Run after model/baseline_max_dose.py, from the repository root:  python model/baseline_metrics.py
Input : results/runs/baseline_max_dose.csv
Output: results/figure_data/baseline_max_dose.csv, data/baseline_max_dose.csv (time series of the run),
        results/tables/baseline_metrics_v2.json (metrics)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from glioma_paper.metrics import metrics  # noqa: E402

runs = pd.read_csv(os.path.join(ROOT, "results", "runs", "baseline_max_dose.csv"))
ts = (runs[runs.section == "baseline_timeseries"]
      .pivot(index="row", columns="name", values="value").sort_index())
new = pd.DataFrame({"run": "max_dose", "time_day": ts.t_day.astype(int).values, "u": ts.u.values,
                    "x1": ts.x1.values, "x2": ts.x2.values, "x3": ts.x3.values, "x4": ts.x4.values})
for rel in (("results", "figure_data", "baseline_max_dose.csv"), ("data", "baseline_max_dose.csv")):
    new.to_csv(os.path.join(ROOT, *rel), index=False, float_format="%.17g")

X = new[["x1", "x2", "x3", "x4"]].values
u = np.nan_to_num(new.u.values[:2000])          # doses on days 0..1999
m = metrics(X, u, discrete=True)
x1 = X[:, 0]
m.update({"switch_off_day": int(np.flatnonzero(u == 0)[0]), "x1_min_day": int(x1.argmin()),
          "x3_at_stop": float(X[int(np.flatnonzero(u == 0)[0]), 2]),
          "x1_at_stop": float(X[int(np.flatnonzero(u == 0)[0]), 0]),
          "x4_at_stop": float(X[int(np.flatnonzero(u == 0)[0]), 3]),
          "first_day_x1_ge_099": float(np.flatnonzero(x1 >= 0.99)[0])})
with open(os.path.join(ROOT, "results", "tables", "baseline_metrics_v2.json"), "w") as f:
    json.dump(m, f, indent=1)
for k, v in m.items():
    print(f"{k:22s} {v}")
