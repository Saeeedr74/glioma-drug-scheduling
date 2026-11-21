# Robust drug scheduling for glioma chemotherapy based on a multivariable dynamic model

Code and data for the article by **Saeed Rahimi** and **Hamed Moradi** (corresponding author),
Sharif University of Technology, Tehran, Iran. Submitted to *Computer Methods and Programs in Biomedicine*.

## Summary
The normalized four-state glioma model of Iarosz et al. (glial cells x1, glioma cells x2, neurons x3,
drug concentration x4) is simulated with RK4 and a step of 1 day. After 600 days without treatment, three
feedback controllers schedule the daily dose u in {50, 200} mg m^-2 day^-1 (maintenance phase and high-dose
phase) so that x2 follows a smooth reference over 2000 days: a gain-scheduled PID, a sliding-mode
controller (SMC) and a mu-synthesis controller redesigned along the trajectory. The repository also holds
the tuning of the PID and SMC parameters (differential evolution), the uncertainty study (five parameter
groups G1..G5), the state observers (extended Kalman filter and an adaptive Luenberger observer), the
numerical checks of the Lyapunov analysis (nominal model) and the script that draws every figure.

## Requirements
- Python 3.13 with the pinned packages in `requirements.txt` (`pip install -r requirements.txt`).
- MATLAB R2023a with the Robust Control, Control System and Statistics and Machine Learning toolboxes,
  for the mu-synthesis part only (see `MATLAB.md`).

## Layout
| folder | content |
|---|---|
| `model/` | pre-conditioning, maximum-dose baseline, `glioma_paper/` (model, replay of a dose schedule, the metric function used for every table) |
| `controllers/pid/`, `controllers/smc/` | nominal discrete PID; discrete SMC, continuous SMC and the standard 5/28 protocol |
| `controllers/mu_synthesis/` | MATLAB design and weight search, results file, caches, CSV exports (`MATLAB.md`) |
| `tuning/` | differential-evolution tuning of PID and SMC |
| `uncertainty/` | protocol E1: nominal PID and SMC schedules replayed on G1..G5 (mu: `mu_py_timeseries.csv`, mode E1_replay) |
| `observer/` | observer study (EKF, Luenberger) and the script that summarizes its results |
| `lyapunov/` | symbolic and grid check of the Lyapunov certificate, the same sign check in exact rational arithmetic, dW/dt along the simulated runs, tracking-error bounds |
| `figures/` | `make_figures.py` (pre-conditioning, PID gains, sliding band, schedules, states, drug concentration, appendix figures) and `make_figures_v2.py` (baseline, 5/28 comparison, replay test, observers, Lyapunov function; written to `figures/main/*_v2.pdf`) |
| `tables/` | scripts that write the rows of the tables of the paper to `results/tables/paper/` (two-decimal formatting in `tables/fmt.py`) |
| `results/` | `tables/` (CSV and LaTeX of every table), `figure_data/` (the data behind every figure) |

## How to run (from the repository root)
Measured on a Windows 11 laptop (Python 3.13.0) in one verification run; times are wall-clock.

| step | command | time |
|---|---|---|
| pre-conditioning | `python model/simulation_preconditioning.py` | 3 s |
| maximum-dose baseline | `python model/baseline_max_dose.py` | 4 s |
| metrics of the baseline | `python model/baseline_metrics.py` | 1 s |
| PID | `python controllers/pid/pid_simulation.py` | 5 s |
| SMC, continuous SMC, 5/28 protocol | `python controllers/smc/smc_simulation.py` | 9 s |
| E1 for PID (after the PID run) | `python uncertainty/pid_uncertainty.py` | 4 s |
| E1 for SMC (after the SMC run) | `python uncertainty/smc_uncertainty.py` | 4 s |
| observers (after the PID and SMC runs) | `python observer/observer.py` | 282 s (11 worker processes) |
| observer summary | `python observer/make_facts_block.py` | 3 s |
| Lyapunov certificate | `python lyapunov/verify_proof_symbolic_and_grid.py` | 4 s |
| sign condition of the certificate in exact arithmetic | `python lyapunov/verify_lemma_exact.py` | 1 s |
| dW/dt along the runs | `python lyapunov/verify_proof_plots.py` | 7 s |
| tracking-error bounds | `python lyapunov/error_dynamics.py` | 2 s |
| all figures | `python figures/make_figures.py` | 17 s |
| baseline, 5/28, replay, observer and Lyapunov figures | `python figures/make_figures_v2.py` | 30 s |
| tables of the paper | `python tables/build_protocol_table.py`, `python tables/build_lyapunov_groups_v2.py`, `python tables/make_table_controller_summary.py`, `python tables/make_tables_observer.py`, `python tables/make_tables_mu_tests.py` | a few seconds |
| tuning (not needed to reproduce the results) | `python tuning/tune_pid_smc.py` | 161 min (authors' run) |
| mu-synthesis (MATLAB) | see `MATLAB.md` | 309.5 min for the full search (authors' run); CSV export 125 s |

The scripts write their CSV files to `results/runs/` and their own diagnostic plots to
`results/diagnostic_plots/`; the figures of the paper come only from `figures/make_figures.py`, which reads
`results/figure_data/`. Every Python script was run from this layout and its output compared with the
original output: all numbers agree to 1e-10 (relative), most files byte for byte. The tuning search
itself was not repeated; its objective was re-evaluated at the hand design and at the selected parameters
and agrees with the stored history.

Conventions: day 0 is the start of treatment (simulation day 600). Threshold for x2: 1e-4. Cumulative dose
= sum of the daily doses on days 0..1999. The maximum-dose baseline gives u = 200 until the first day with
x2 < 1e-4 (day 236); from that day on x2 := 0 and u = 0. The continuous SMC, which is not used in the paper,
stops at x2 <= 1e-5 (a stopping criterion of the code, then x2 := 0 and u = 0).

## Figures and tables
| item | script | data |
|---|---|---|
| model schematic | TikZ in the manuscript | - |
| `fig_preconditioning` | figures/make_figures.py | preconditioning.csv (model/simulation_preconditioning.py) |
| `fig_baseline_v2` | figures/make_figures_v2.py | baseline_max_dose.csv (model/baseline_max_dose.py, model/baseline_metrics.py) |
| `fig_schedules`, `fig_states`, `fig_x4` | figures/make_figures.py | nominal_discrete.csv (PID, SMC, mu runs) |
| `fig_pid_gains` | figures/make_figures.py | pid_gain_schedule.csv (controllers/pid) |
| `fig_smc_band` | figures/make_figures.py | smc_sliding_band.csv (controllers/smc) |
| `fig_continuous` | figures/make_figures.py | nominal_continuous.csv (controllers/smc, mu_synthesis) |
| `fig_uncertainty_v2` | figures/make_figures_v2.py | e1_uncertainty.csv (uncertainty/, mu E1) |
| `fig_protocol_comparison_v2` (one panel, used in the paper), `fig_protocol_comparison_4panel_v2` | figures/make_figures_v2.py | standard_protocol.csv, nominal_discrete.csv (controllers/smc) |
| `fig_observer_M1_stacked_v2`, `fig_observer_noise20_stacked_v2`, `fig_observer_mismatch_stacked_v2` | figures/make_figures_v2.py | observer/results/ (observer/observer.py); the noise figure recomputes the 20 noise sequences with the functions of observer.py |
| `fig_lyapunov_W_dW_v2`, `fig_lyapunov_G5_SMC_v2` | figures/make_figures_v2.py | lyapunov_W_timeseries.csv (lyapunov/verify_proof_plots.py) |
| `fig_app_tuning` | figures/make_figures.py | tuning_convergence.csv (tuning/) |
| `fig_app_mu_search`, `fig_app_mu_E2_mc`, `fig_app_mu_rules` | figures/make_figures.py | mu_*.csv (controllers/mu_synthesis) |
| `fig_app_lyapunov` | figures/make_figures.py | lyapunov_W_timeseries.csv (lyapunov/verify_proof_plots.py) |
| main tables (`T_main_discrete`, `T_main_continuous`, `T6_uncertainty`) | metric function in model/glioma_paper/metrics.py | results/tables/ |
| tables as printed in the paper (metrics of the controllers, replay test, observers, mu-synthesis tests, 5/28 comparison, Lyapunov groups) | tables/*.py | results/tables/paper/, `T_protocol_comparison`, `T_lyapunov_groups_v2` |
| `T_observer`, `T_app_observer_*` | observer/observer.py | results/tables/ |
| `T_design_pid`, `T_design_smc`, `T_app_tuning_runs` | controllers/pid, controllers/smc, tuning/ | results/tables/ |
| `T_design_mu`, `T_app_mu_*` | controllers/mu_synthesis | results/tables/ |
| `T_lyapunov`, `T_app_lyapunov_groups` | lyapunov/ | results/tables/ |

`results/figure_data/` was assembled from the outputs of the scripts above (long format, one file per
figure family); the larger files are stored gzipped and are read as they are.

## Notes on the stored results
- Baseline: `model/baseline_max_dose.py` stops the drug on the first day with x2 < 1e-4, the suppression
  threshold of the paper. The run stops on day 236, with a cumulative dose of 47200 mg m^-2, x3(T) = 0.476 and
  glial recovery on day 664 (`results/tables/baseline_metrics_v2.json`, written by `model/baseline_metrics.py`).
- The baseline entries of `data/metrics_all.csv`, `results/tables/T_main_discrete.*`,
  `results/tables/T_app_baseline_protocols.*` and Section 3 of `FACTS_FINAL.md` belong to a run with the stop
  criterion x2 <= 1e-5 (day 299, 59800 mg m^-2). The paper uses `baseline_metrics_v2.json` and
  `results/tables/paper/controller_summary_rows.tex` for the baseline.
- `figures/make_figures_v2.py` uses the style settings of `make_figures.py`. Its observer figures read
  `observer/results/`, and its noise figure recomputes the 20 noise sequences with the functions of
  `observer/observer.py`.
- A new run of `controllers/smc/smc_simulation.py` and of the three scripts in `lyapunov/` reproduces the stored
  outputs to 1e-10 (relative).

## License and citation
MIT License (`LICENSE`). Please cite the article (`CITATION.cff`).
