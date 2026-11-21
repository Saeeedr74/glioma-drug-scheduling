# MATLAB part (mu-synthesis)

Folder: `controllers/mu_synthesis/`.

## Version and toolboxes
Run and checked with MATLAB R2023a Update 4 (9.14.0.2306882) and
- Robust Control Toolbox 6.11.3 (`musyn`, `robstab`, uncertain systems),
- Control System Toolbox 10.13 (required by the Robust Control Toolbox),
- Statistics and Machine Learning Toolbox 12.5 (`bayesopt` in the weight search).

`mu_export_for_python.m` and `mu_export_extras.m` need base MATLAB only.

## Files
| file | role |
|---|---|
| `mu_optimize_final.m` | weight search (5 random points + 25 Bayesian-optimization evaluations, seed 7), final design D1, verification, continuous run, redesign-rule comparison, uncertainty runs E1 (replay), E2 (closed loop), Monte Carlo E3a/E3b, figures |
| `mu_opt_results.mat` | all results of the final run (read by the two export scripts) |
| `mu_opt_cache.mat`, `mu_opt_rules_cache.mat`, `mu_opt_search_history.mat` | caches of the full simulations and of the search history (each below 3 MB, included) |
| `mu_export_for_python.m` | writes `mu_py_timeseries.csv`, `mu_py_metrics.csv`, `mu_py_mc.csv`, `mu_py_design.csv` |
| `mu_export_extras.m` | writes the additional CSVs used for the appendix (states of the continuous run, redesign days and mu per design, redesign-rule runs, search table, closed-loop Monte Carlo trajectories, scalars) |
| `u_schedule_mu.csv`, `mu_opt_u_D1.csv` | discrete dose schedule of design D1 (days 0..2000) |
| `mu_opt_u_D1_continuous.csv` | continuous dose of design D1 (time, u, x2, reference) |
| `mu_opt_u_miu_column.csv` | dose column of the discrete run |

`mu_py_timeseries.csv` columns: controller, mode (nominal / E1_replay / E2_closed_loop), scenario, group_id,
time_day (0 = start of treatment = simulation day 600), x1..x4, u, x2_reference. `mu_py_metrics.csv`:
t_x2_first/perm = first / permanent day with x2 < 1e-4 (NaN = never); t_x1_first/perm the same for x1 >= 0.99.
Use E1 for the three-controller comparison.

## How to run
In MATLAB, with this folder as the current folder:

    mu_optimize_final        % resumable; uses the caches
    mu_export_for_python     % 125 s measured with matlab -batch, MATLAB start-up included
    mu_export_extras         % reads the .mat only; not timed

With the caches present, `mu_optimize_final` reuses the stored search history and simulations. Without
them it repeats the full search; the authors' final run took 309.5 min (`elapsed_min` stored in
`mu_opt_results.mat`). The settings structs (`cfg`, `S`, `VERSION`) key the caches: changing any setting
starts a new search. mu_export_for_python was re-run from this folder and its four CSV files are
byte-identical to the shipped ones. The console of a run is written to `mu_opt_console.txt` (not tracked).
