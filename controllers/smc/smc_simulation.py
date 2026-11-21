"""smc_simulation.py

Discrete SMC relay (final v2 parameters), continuous SMC and the standard 5/28 protocol (doses 200 and 400).

Run:  python controllers/smc/smc_simulation.py

Part of the code for: S. Rahimi, H. Moradi, "Robust drug scheduling for glioma chemotherapy based on a
multivariable dynamic model" (Sharif University of Technology). MIT License, see LICENSE.
"""
import os                  # needed by the parameter block (all imports follow below)
import numpy as np

# =============================================================================
# 0) PARAMETERS
# =============================================================================
# --- Normalized model, Table 2 ------------------------------------------------
OMEGA_1 = 0.0068    # glial proliferation rate [1/day]
OMEGA_2 = 0.012     # glioma proliferation rate [1/day]
BETA_1 = 1.8e-2     # competition, glioma -> glial [1/day]
BETA_2 = 1.8e-3     # competition, glial -> glioma [1/day]
P_1 = 4.7e-8        # max kill rate, glia    [m^2/(mg day)]
P_2 = 4.7e-5        # max kill rate, glioma  [m^2/(mg day)]
P_3 = 4.7e-8        # max kill rate, neurons [m^2/(mg day)]

# Half-saturation constants a_1 = a_2 = a_3 = 1. They are kept as the integer 1
# (not 1.0) on purpose: the symbolic derivation of f(x), g(x) below then builds
# exactly the same expression tree as the original notebook did.
A_1 = 1
A_2 = 1
A_3 = 1
ALPHA = 2.0         # neurodegeneration scaling (Table 2 gives the range 0-10)
ZETA = 0.2          # drug clearance rate [1/day]

# --- Simulation ---------------------------------------------------------------
DT = 1.0            # RK4 step AND controller update period [day]
T_START = 600.0     # pre-conditioning length, u = 0 [day]
T_REF = 2000.0      # therapy horizon [day]
X0 = np.array([1.0, 1e-3, 1.0, 0.0])   # [x1, x2, x3, x4] at the start of pre-conditioning

# --- Dose levels (shared by PID, SMC and the baseline) -----------------------
U_MIN = 50.0        # maintenance-phase dose [mg m^-2 day^-1]
U_MAX = 200.0       # high-dose-phase dose   [mg m^-2 day^-1]

# --- Thresholds ---------------------------------------------------------------
X2_THRESHOLD = 1e-4     # tumor suppression threshold reported in the paper
X1_RECOVERY = 0.99      # glial recovery threshold reported in the paper
X2_STOP = 1e-5          # stop criterion of the CONTINUOUS SMC only (v2, 2026-11-14: no longer used by the
                        # discrete relay). Former use: SMC loops, and the switch-off
                        # point of the maximum-dose baseline). NOT a reported threshold.

# --- SMC, Table 4 -------------------------------------------------------------
SMC_A_REF = 7.5         # reference shaping a; unified with PID to 7.5 (retune 2026-11-14; was 8.5)
SMC_B_REF = 0.0         # reference shaping b
LAMBDA_P = 1.8813613583755686      # coefficient of e (lambda_0); tuned v2 by General/tune_pid_smc (DE seed 7, a = 7.5, 2026-11-14); hand design 1.5
LAMBDA_D = 9.619198702305367       # coefficient of de/dt (lambda_1); tuned v2 by General/tune_pid_smc (DE seed 7, a = 7.5, 2026-11-14); hand design 2.5
LAMBDA_I = 2.9470610386036095e-05  # coefficient of int(e); tuned v2 by General/tune_pid_smc (DE seed 7, a = 7.5, 2026-11-14); hand design 1.0e-04
GAMMA_1 = 1.0           # tanh smoothing on |e|     (continuous SMC; not yet in Table 4 - A-5)
GAMMA_2 = 1.0           # tanh smoothing on |de/dt| (continuous SMC; not yet in Table 4 - A-5)
DELTA_INIT = 0.013664689577712264  # initial hysteresis half-width; tuned v2 by General/tune_pid_smc (DE seed 7, a = 7.5, 2026-11-14); hand design 0.01
DELTA_FINAL = 2.676680232434659e-07 # final hysteresis half-width; tuned v2 by General/tune_pid_smc (DE seed 7, a = 7.5, 2026-11-14); hand design 1e-9
A_DELTA = 7.824136642893967        # hysteresis sharpness a_delta; tuned v2 by General/tune_pid_smc (DE seed 7, a = 7.5, 2026-11-14); hand design 7.5
B_DELTA = 0.10364584462506671      # hysteresis offset b_delta; tuned v2 by General/tune_pid_smc (DE seed 7, a = 7.5, 2026-11-14); hand design 0.0

# --- Standard 5/28 protocol (comparison figure) -------------------------------
STD_CYCLE_DAYS = 28
STD_ON_DAYS = 5
STD_DOSES = (200.0, 400.0)  # 200 = clinical maximum; 400 = dose used by Iarosz et al.

# --- Output -------------------------------------------------------------------
SHOW_FIGURES = False            # True -> also display each figure (plt.show); False -> headless, files only
SAVE_DPI = 600
# Paths are relative to the repository root (REPO). Outputs: one csv, results/runs/smc_simulation.csv, and the
# diagnostic plots of this script in results/diagnostic_plots (file names in FIGURE_FILES).
ROOT = "."                             # base for the printed paths
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # repository root
RESULTS_DIR = os.path.join(REPO, "results", "runs")
PLOT_DIR = os.path.join(REPO, "results", "diagnostic_plots")
RESULTS_FILE = os.path.join(RESULTS_DIR, "smc_simulation.csv")
FIGURE_FILES = {
    "smc_simulation_sliding_band": "SMC_Fig7_sliding_band",
    "smc_simulation_u": "SMC_Fig8_u",
    "smc_simulation_x4": "SMC_Fig8_x4",
    "smc_simulation_drug_schedule": "SMC_Fig8_drug_schedule",
    "smc_simulation_x1x3": "SMC_Fig9_x1x3",
    "smc_simulation_x2": "SMC_Fig10_x2",
    "smc_simulation_comparison_standard": "SMC_Fig17_comparison_5_28",
}
SECTION_PREFIX = "simulation_"   # this script writes (and replaces) only these sections
SECTION_ORDER = ("simulation_", "observer_", "uncertainty_")

# ---- the tuned / design parameters of this notebook (identical in the simulation,
#      observer and uncertainty notebooks of the same controller) --------------------
for _k in ('SMC_A_REF', 'SMC_B_REF', 'LAMBDA_P', 'LAMBDA_D', 'LAMBDA_I', 'GAMMA_1', 'GAMMA_2', 'DELTA_INIT', 'DELTA_FINAL', 'A_DELTA', 'B_DELTA'):
    if _k in globals():
        print(f"{_k:<12} = {globals()[_k]!r}")

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from matplotlib.ticker import MaxNLocator, ScalarFormatter

# =============================================================================
# 1) MODEL
# =============================================================================
def rhs_x1(x1, x2, x4):
    return OMEGA_1*x1*(1-x1) - BETA_1*x1*x2 - P_1*x1*x4/(A_1+x1)


def rhs_x2(x1, x2, x4):
    return OMEGA_2*x2*(1-x2) - BETA_2*x1*x2 - P_2*x2*x4/(A_2+x2)


def rhs_x4(x4, u):
    return u - ZETA*x4


def plant_nominal(x, u):
    """Normalized Iarosz et al. model, H(0) = 1/2."""
    x1, x2, x3, x4 = x
    dx1 = rhs_x1(x1, x2, x4)
    dx2 = rhs_x2(x1, x2, x4)
    heav = np.heaviside(-dx1, 0.5)
    dx3 = ALPHA*heav*dx1*x3 - P_3*x3*x4/(A_3+x3)
    dx4 = rhs_x4(x4, u)
    return np.array([dx1, dx2, dx3, dx4])


def rk4_step(f, x, u, dt):
    """One classical RK4 step with the input held constant over the step."""
    k1 = f(x, u)
    k2 = f(x + (dt/2)*k1, u)
    k3 = f(x + (dt/2)*k2, u)
    k4 = f(x + dt*k3, u)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# =============================================================================
# 2) S-CURVES - reference trajectory and hysteresis band
# =============================================================================
def s_curve(tau, a, b):
    """S(tau) = 0.5 * (1 - tanh(a * (tau - b))), tau = t / T_ref."""
    return 0.5 * (1 - np.tanh(a * (tau - b)))


def tumor_reference(x2_0, times, a, b):
    """r(t) = r(0) * (S(t) - S(tau=1)) / (S(0) - S(tau=1)),  r(0) = x2(T_start)."""
    S = s_curve(times/T_REF, a, b)
    S0 = s_curve(0.0, a, b)
    S1 = s_curve(1.0, a, b)
    return x2_0 * ((S - S1) / (S0 - S1)), S0, S1

# =============================================================================
# 3) SYMBOLIC f(x), g(x) for the SMC:  d^2 x2/dt^2 = f(x) + g(x) u
# =============================================================================
def derive_smc_affine_terms():
    t = sp.symbols('t')
    x1_s, x2_s, x3_s, x4_s = sp.symbols('x1 x2 x3 x4', cls=sp.Function)
    u_s = sp.symbols('u')
    X1, X2, X3, X4 = x1_s(t), x2_s(t), x3_s(t), x4_s(t)

    DX1 = rhs_x1(X1, X2, X4)
    DX2 = rhs_x2(X1, X2, X4)
    DX4 = rhs_x4(X4, u_s)
    # x3 does not enter dx2/dt, so its derivative never appears in d^2x2/dt^2.
    DDX2 = sp.diff(DX2, t).subs({
        sp.diff(X1, t): DX1,
        sp.diff(X2, t): DX2,
        sp.diff(X4, t): DX4,
    })
    f_sym = sp.simplify(DDX2.subs(u_s, 0))
    g_sym = sp.diff(DDX2, u_s)
    f_func = sp.lambdify((X1, X2, X3, X4), f_sym, 'numpy')
    g_func = sp.lambdify((X1, X2, X3, X4), g_sym, 'numpy')
    return f_func, g_func

# =============================================================================
# 4) SIMULATIONS
# =============================================================================
def simulate_preconditioning():
    """u = 0 on [0, T_start]. Returns times and the full state history."""
    n = int(T_START / DT)
    X = np.zeros((n + 1, 4))
    x = X0.copy()
    X[0] = x
    for k in range(1, n + 1):
        x = rk4_step(plant_nominal, x, 0.0, DT)
        x[1] = max(x[1], 0.0)
        X[k] = x
    return np.arange(n + 1) * DT, X


def simulate_smc_discrete(x_init, ref_arr, drdt_arr, ddrdt_arr, delta_arr, f_func, g_func):
    """Discrete SMC relay with a shrinking hysteresis band.

    U_list[0] is a placeholder equal to U_MAX at t = 0 (it is also the initial
    u(t^-) used in the sliding variable); U_list[k], k >= 1, is the dose applied
    on [k-1, k). S_list[k] and D_list[k] are evaluated at the state of day k.
    """
    x = x_init.copy()
    t = 0.0
    X_list = [x.copy()]
    U_list = [U_MAX]
    T_list = [0.0]
    int_e = 0.0
    u_prev = U_MAX
    idx_max = len(ref_arr) - 1
    S_list, D_list = [], []
    delta_prev = None

    # v2 (2026-11-14): no end-of-treatment stop rule - the relay stays active, with its normal
    # dose rule, up to T_ref (was: while (x[1] > X2_STOP) and (t < T_REF)).
    while t < T_REF:
        idx = min(int(np.floor(t/DT)), idx_max)
        r = ref_arr[idx]
        drdt = drdt_arr[idx]
        ddrdt = ddrdt_arr[idx]
        dx_nom = plant_nominal(x, 0.0)    # dx2/dt does not depend on u
        e = x[1] - r
        de = dx_nom[1] - drdt
        int_e += e * DT

        gval = g_func(*x)
        s = (f_func(*x) + gval*u_prev - ddrdt) + LAMBDA_D*de + LAMBDA_P*e + LAMBDA_I*int_e

        delta_k = delta_arr[idx]
        if delta_prev is not None:
            delta_k = min(delta_prev, delta_k)
        delta_prev = delta_k

        S_list.append(s)
        D_list.append(delta_k)

        if gval >= 0.0:
            if s > delta_k:
                u = U_MIN
            elif s < -delta_k:
                u = U_MAX
            else:
                u = u_prev
        else:
            if s > delta_k:
                u = U_MAX
            elif s < -delta_k:
                u = U_MIN
            else:
                u = u_prev

        U_list.append(u)
        u_prev = u
        x = rk4_step(plant_nominal, x, u, DT)
        x[1] = max(x[1], 0.0)
        t += DT
        X_list.append(x.copy())
        T_list.append(t)

    # v2 (2026-11-14): the former tail "if x2 fell below X2_STOP: tumor set to zero, no drug,
    # until T_ref" was removed together with the stop rule.

    return (np.vstack(X_list), np.array(U_list), np.array(T_list),
            np.array(S_list), np.array(D_list))


def simulate_smc_continuous(x_init, ref_arr, drdt_arr, ddrdt_arr, f_func, g_func):
    """Continuous SMC (equivalent control with tanh-weighted gains), u in [0, U_MAX].

    Uc[k] is the dose applied on [k, k+1).
    """
    x = x_init.copy()
    t = 0.0
    Xc = [x.copy()]
    Uc = []
    Tc = [0.0]
    int_e = 0.0
    idx_max = len(ref_arr) - 1
    while (x[1] > X2_STOP) and (t < T_REF):
        idx = min(int(np.floor(t/DT)), idx_max)
        r = ref_arr[idx]
        drdt = drdt_arr[idx]
        ddrdt = ddrdt_arr[idx]
        dx_nom = plant_nominal(x, 0.0)
        e = x[1] - r
        de = dx_nom[1] - drdt
        int_e += e * DT

        w_e = np.tanh(GAMMA_1 * abs(e))
        w_de = np.tanh(GAMMA_2 * abs(de))
        gval = g_func(*x)
        u_eq = (ddrdt - f_func(*x) - LAMBDA_D*w_de*de - LAMBDA_P*w_e*e - LAMBDA_I*int_e) / gval
        u = np.clip(u_eq, 0.0, U_MAX)
        Uc.append(u)
        x = rk4_step(plant_nominal, x, u, DT)
        x[1] = max(x[1], 0.0)
        t += DT
        Xc.append(x.copy())
        Tc.append(t)

    while t < T_REF:
        Uc.append(0.0)
        x = rk4_step(plant_nominal, np.array([x[0], 0.0, x[2], x[3]]), 0.0, DT)
        x[1] = 0.0
        t += DT
        Xc.append(x.copy())
        Tc.append(t)

    return np.vstack(Xc), np.array(Uc), np.array(Tc)


def simulate_standard_protocol(x_init, dose):
    """Open-loop 5/28 cycle: `dose` for 5 days, then 23 days with u = 0."""
    x = x_init.copy()
    t = 0.0
    X_list = [x.copy()]
    U_list = []
    T_list = [0.0]
    while t < T_REF:
        day_in_cycle = int(t) % STD_CYCLE_DAYS
        u = dose if day_in_cycle < STD_ON_DAYS else 0.0
        x = rk4_step(plant_nominal, x, u, DT)
        x[1] = max(x[1], 0.0)
        U_list.append(u)
        t += DT
        X_list.append(x.copy())
        T_list.append(t)
    return np.vstack(X_list), np.array(U_list), np.array(T_list)

# =============================================================================
# 5) METRICS AND SCHEDULE STATISTICS
# =============================================================================
def first_index(cond):
    idx = np.where(cond)[0]
    return int(idx[0]) if idx.size else None


def permanent_index(cond):
    """First index from which `cond` holds until the end of the array."""
    bad = np.where(~cond)[0]
    if bad.size == 0:
        return 0
    i = int(bad[-1]) + 1
    return i if i < len(cond) else None


def _fmt_day(T, i):
    return "not reached" if i is None else f"{T[i]:.1f}"


def line(label, value):
    print(f"  {label:<46}= {value}")


def print_metrics(label, T, X, convention):
    """Print the paper's metrics with both crossing definitions.

    convention: which definition the current manuscript value was taken from
    ('first' = first sample satisfying the condition,
     'permanent' = first sample from which the condition holds to the end).
    """
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    c2 = x2 <= X2_THRESHOLD
    c1 = x1 >= X1_RECOVERY
    i2000 = int(np.argmin(np.abs(T - T_REF)))
    print(f"[{label}]")
    line("time to x2 <= 1e-4, first crossing [d]", _fmt_day(T, first_index(c2)))
    line("time to x2 <= 1e-4, permanent [d]", _fmt_day(T, permanent_index(c2)))
    line("glial recovery x1 >= 0.99, first [d]", _fmt_day(T, first_index(c1)))
    line("glial recovery x1 >= 0.99, permanent [d]", _fmt_day(T, permanent_index(c1)))
    line(f"x3(T = {T[i2000]:.0f})", f"{x3[i2000]:.6f}")
    line(f"x2(T = {T[i2000]:.0f})", f"{x2[i2000]:.6e}")
    line("manuscript convention for the two times", convention)
    print()


def build_segments(T, U):
    """Constant-dose segments (start, end, dose) from a dose series."""
    segs, cur_u, start = [], U[0], T[0]
    for i in range(1, len(U)):
        if U[i] != cur_u:
            segs.append((start, T[i], float(cur_u)))
            cur_u, start = U[i], T[i]
    segs.append((start, T[-1], float(cur_u)))
    return segs


def schedule_rows(T_ctrl, U, X):
    """One row per segment: start, end, duration, dose, x2 at the two ends."""
    rows = []
    for s, e, u in build_segments(T_ctrl, U):
        s_int = int(np.floor(s))
        e_int = int(np.floor(e))
        dur = max(1, e_int - s_int)
        idx_start = int(np.argmin(np.abs(T_ctrl - s)))
        idx_end = int(np.argmin(np.abs(T_ctrl - e)))
        rows.append({
            "start": s_int, "end": e_int, "duration": dur, "dose": u,
            "x2_start": X[idx_start, 1],
            "x2_end": X[min(idx_end, len(X) - 1), 1],
        })
    return rows


def phase_durations(rows, tol=1e-9):
    """High-dose and maintenance durations in chronological order.

    The last segment is dropped (it is cut off by the end of the horizon) and
    zero-dose segments are skipped.
    """
    rows_sorted = sorted(rows, key=lambda r: r["start"])[:-1]
    high_x, high_y, maint_x, maint_y = [], [], [], []
    pulse_index = 0
    for r in rows_sorted:
        if r["dose"] <= tol:
            continue
        if abs(r["dose"] - U_MAX) <= tol:
            high_x.append(pulse_index)
            high_y.append(r["duration"])
            pulse_index += 1
        elif abs(r["dose"] - U_MIN) <= tol:
            maint_x.append(pulse_index)
            maint_y.append(r["duration"])
            pulse_index += 1
    return high_x, high_y, maint_x, maint_y


def print_phase_stats(label, rows):
    _, high, _, maint = phase_durations(rows)
    print(f"[{label}] phase statistics (last, horizon-truncated segment excluded)")
    for name, d in (("high-dose phases (200)  ", high), ("maintenance phases (50) ", maint)):
        if d:
            print(f"  {name}: count = {len(d)}, mean = {np.mean(d):.1f} d, "
                  f"min = {int(np.min(d))} d, max = {int(np.max(d))} d")
        else:
            print(f"  {name}: none")
    if high:
        line("longest continuous high-dose interval", f"{int(np.max(high))} d")
    print()


def print_schedule_table(title, rows):
    print(title)
    print(f"  {'start':>6} {'end':>6} {'dur':>5} {'dose':>7}   x2 from -> to")
    for r in rows:
        print(f"  {r['start']:6d} {r['end']:6d} {r['duration']:5d} {r['dose']:7.2f}   "
              f"{r['x2_start']:.2e} -> {r['x2_end']:.2e}")
    print()

# =============================================================================
# 6) RESULTS FILE (tidy CSV, read-modify-write)
# =============================================================================
CSV_COLUMNS = ["section", "run", "row", "name", "value", "text"]


def long_rows(section, run, columns):
    """Tidy (long) form of one table: one CSV row per cell.

    `columns` maps a column name to an equal-length sequence (a scalar is a one-row
    table). Numbers go to `value`, strings and booleans to `text`.
    """
    rows = []
    for name, seq in columns.items():
        if isinstance(seq, str) or np.isscalar(seq):
            seq = [seq]
        for i, v in enumerate(seq):
            if isinstance(v, (str, bool, np.bool_)):
                rows.append((section, run, i, name, np.nan, str(v)))
            elif v is None:
                rows.append((section, run, i, name, np.nan, ""))
            else:
                rows.append((section, run, i, name, float(v), ""))
    return rows


def read_results():
    """The whole results file (an empty table if it does not exist yet)."""
    if not os.path.exists(RESULTS_FILE):
        return pd.DataFrame(columns=CSV_COLUMNS)
    df = pd.read_csv(RESULTS_FILE,
                     dtype={"section": str, "run": str, "name": str, "text": str})
    for c in ("section", "run", "name", "text"):
        df[c] = df[c].fillna("")
    return df


def write_sections(rows):
    """Read-modify-write: replace this script's sections, keep every other section."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    old = read_results()
    keep = old[~old["section"].str.startswith(SECTION_PREFIX)]
    new = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df = pd.concat([keep, new], ignore_index=True) if len(keep) else new
    rank = [next((i for i, p in enumerate(SECTION_ORDER) if s.startswith(p)), len(SECTION_ORDER))
            for s in df["section"]]
    df = df.iloc[np.argsort(rank, kind="stable")]
    df.to_csv(RESULTS_FILE, index=False)
    print(f"  wrote {os.path.relpath(RESULTS_FILE, ROOT)}  "
          f"({len(new)} rows in the {SECTION_PREFIX}* sections, {len(df)} rows in total)")


def read_table(section, run):
    """One table back from the results file: column name -> numpy array (by row)."""
    df = read_results()
    sub = df[(df["section"] == section) & (df["run"] == run)]
    if sub.empty:
        raise FileNotFoundError(f"{section} / {run} not found in {RESULTS_FILE} - "
                                "run the simulation script first.")
    print(f"  read results/{os.path.basename(RESULTS_FILE)}: {section} / {run}")
    return {name: g.sort_values("row")["value"].to_numpy(float)
            for name, g in sub.groupby("name", sort=False)}


def day_or_nan(T, i):
    return np.nan if i is None else T[i]


def pad_nan(a, n):
    a = np.asarray(a, dtype=float)
    return np.concatenate([a, np.full(n - len(a), np.nan)])


def metric_values(T, X):
    """The numbers print_metrics() prints, as a dict for the results file
    (days; NaN = not reached)."""
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    c2 = x2 <= X2_THRESHOLD
    c1 = x1 >= X1_RECOVERY
    i2000 = int(np.argmin(np.abs(T - T_REF)))
    return {"t_x2_1e4_first_day": day_or_nan(T, first_index(c2)),
            "t_x2_1e4_perm_day": day_or_nan(T, permanent_index(c2)),
            "t_x1_099_first_day": day_or_nan(T, first_index(c1)),
            "t_x1_099_perm_day": day_or_nan(T, permanent_index(c1)),
            "T_day": T[i2000], "x3_T": x3[i2000], "x2_T": x2[i2000]}


def phase_stat_values(rows):
    """The numbers print_phase_stats() prints, as a dict for the results file."""
    _, high, _, maint = phase_durations(rows)
    out = {}
    for key, d in (("high_dose", high), ("maintenance", maint)):
        out[f"{key}_count"] = len(d)
        out[f"{key}_mean_d"] = np.mean(d) if d else np.nan
        out[f"{key}_min_d"] = int(np.min(d)) if d else np.nan
        out[f"{key}_max_d"] = int(np.max(d)) if d else np.nan
    return out


def full_schedule_table(U_ctrl, T_state, X_state, total_days=2000):
    """[Day, Dose, x2_end_of_day] - the former <X>-FullSchedule.csv, same arithmetic.

    Dose on row d is U_ctrl[d-1]; x2_end_of_day on row d is x2(t = d).
    """
    U_ctrl = np.asarray(U_ctrl, dtype=float)
    N = int(total_days)
    daily_dose = U_ctrl[:N]
    days = np.arange(1, N + 1, dtype=float)
    tq = np.clip(days, T_state[0], T_state[-1])
    x2_daily = np.maximum(np.interp(tq, T_state, X_state[:, 1]), 0.0)
    return {"Day": days, "Dose": daily_dose, "x2_end_of_day": x2_daily}

# =============================================================================
# 7) FIGURE STYLE
# =============================================================================
RC_PRECONDITIONING = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
}


RC_RESULTS = {**RC_PRECONDITIONING,
              "figure.dpi": 300,
              "savefig.dpi": 600,
              "font.size": 14,
              "axes.titlesize": 14,
              "axes.linewidth": 1.0,
              "legend.frameon": False}


RC_SCHEDULE = {**RC_RESULTS,
               "figure.figsize": (9, 4.2),
               "font.size": 11,
               "axes.labelsize": 11,
               "axes.titlesize": 11,
               "legend.fontsize": 10}

FIGSIZE = (9, 4.2)


def save(fig, name):
    """Save ../Figures/<paper file name>.png (600 dpi), PNG only. FIGURE_FILES maps each
    figure of this notebook to its paper file name; any other figure is closed unsaved."""
    if name not in FIGURE_FILES:
        plt.close(fig)
        return
    fig.savefig(os.path.join(PLOT_DIR, f"{FIGURE_FILES[name]}.png"), dpi=SAVE_DPI,
                bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def finish(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="best")
    ax.margins(x=0)
    ax.figure.tight_layout()


def beautify_axes(ax, xlabel=None, ylabel=None, title=None):
    """Axis cosmetics used by the protocol-comparison figure."""
    if title:
        ax.set_title(title, pad=6, loc='center')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))


UNIT_U = r"Infusion rate [mg m$^{-2}$ day$^{-1}$]"


CELLS = "Normalized cell population density"


def plot_schedule(rows, title, name):
    high_x, high_y, maint_x, maint_y = phase_durations(rows)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if high_x:
        ax.plot(high_x, high_y, '-o', color="tab:blue", lw=1.6, ms=4,
                label='High-dose phase (200)')
    if maint_x:
        ax.plot(maint_x, maint_y, '-o', color="tab:orange", lw=1.6, ms=4,
                label='Maintenance phase (50)')
    ax.set_xlabel('Pulse index')
    ax.set_ylabel('Duration [days]')
    ax.set_title(title, pad=6)
    ax.legend(loc="best")
    ax.margins(x=0)
    fig.tight_layout()
    save(fig, name)

# =============================================================================
# 8) MAIN
# =============================================================================
# Diagnostic plots of this script (FIGURE_FILES).
PLOTS = tuple(FIGURE_FILES)


def main():
    if not SHOW_FIGURES:
        plt.switch_backend("Agg")
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- x_start: nominal pre-conditioned state, computed silently ----------
    # (the pre-conditioning run itself, its prints, CSV and figure are in
    #  General/simulation_preconditioning)
    _, X_pre = simulate_preconditioning()
    x_burn = X_pre[-1].copy()

    # ---- Reference, its derivatives, hysteresis band ------------------------
    times_ref = np.arange(int(T_REF / DT) + 1) * DT
    x2_ref_smc, S0_smc, S1_smc = tumor_reference(x_burn[1], times_ref, SMC_A_REF, SMC_B_REF)
    z_smc = times_ref / T_REF - SMC_B_REF
    sech2_smc = 1 / (np.cosh(SMC_A_REF * z_smc) ** 2)
    dSdx = -0.5 * SMC_A_REF * sech2_smc
    drdt_arr = x2_ref_smc[0] * (dSdx / (S0_smc - S1_smc)) * (1 / T_REF)
    d2Sdx2 = (SMC_A_REF * SMC_A_REF) * sech2_smc * np.tanh(SMC_A_REF * z_smc)
    ddrdt_arr = x2_ref_smc[0] * (d2Sdx2 / (S0_smc - S1_smc)) * (1 / (T_REF ** 2))

    Sd0 = s_curve(0.0, A_DELTA, B_DELTA)
    Sd1 = s_curve(1.0, A_DELTA, B_DELTA)
    Sd = s_curve(times_ref / T_REF, A_DELTA, B_DELTA)
    delta_arr = DELTA_FINAL + ((Sd - Sd1) / (Sd0 - Sd1)) * (DELTA_INIT - DELTA_FINAL)
    delta_arr = np.minimum.accumulate(delta_arr)

    # ---- Simulations --------------------------------------------------------
    f_func, g_func = derive_smc_affine_terms()
    X_smc, U_smc, T_smc, S_smc, D_smc = simulate_smc_discrete(
        x_burn.copy(), x2_ref_smc, drdt_arr, ddrdt_arr, delta_arr, f_func, g_func)
    X_cont, U_cont, T_cont = simulate_smc_continuous(
        x_burn.copy(), x2_ref_smc, drdt_arr, ddrdt_arr, f_func, g_func)
    std_runs = {d: simulate_standard_protocol(x_burn.copy(), d) for d in STD_DOSES}
    smc_rows = schedule_rows(T_smc, U_smc, X_smc)
    chk_rows = schedule_rows(T_smc[1:], U_smc[1:], X_smc[1:])

    # ---- Console ------------------------------------------------------------
    print("=" * 72)
    print("smc_simulation - console output for FACTS.md")
    print("=" * 72)
    print_metrics("SMC (discrete)", T_smc, X_smc, "first")
    print_metrics("SMC (continuous)", T_cont, X_cont, "first")
    for d, (Xs, _, Ts) in std_runs.items():
        j = int(np.argmin(np.abs(Ts - T_REF)))
        print(f"[Standard 5/28, dose {d:.0f}]  x2(T = {Ts[j]:.0f}) = {Xs[j, 1]:.6e}")
    print()

    print_phase_stats("SMC (discrete)", smc_rows)
    # Check line: U_smc[0] is the t = 0 placeholder, so the first SMC segment
    # above is one sample longer than the dose actually applied.
    print("[SMC (discrete)] same statistics with the t = 0 placeholder excluded (check)")
    _, hi_chk, _, ma_chk = phase_durations(chk_rows)
    print(f"  high-dose: mean = {np.mean(hi_chk):.1f} d, min = {int(np.min(hi_chk))} d, "
          f"max = {int(np.max(hi_chk))} d | maintenance: mean = {np.mean(ma_chk):.1f} d, "
          f"min = {int(np.min(ma_chk))} d, max = {int(np.max(ma_chk))} d")
    print()

    print_schedule_table("[SMC] drug administration schedule (discrete)", smc_rows)

    # ---- Results file -------------------------------------------------------
    print("Writing csv ...")
    rows = long_rows("simulation_metrics", "smc_discrete", metric_values(T_smc, X_smc))
    rows += long_rows("simulation_metrics", "smc_continuous", metric_values(T_cont, X_cont))
    for d, (Xs, _, Ts) in std_runs.items():
        j = int(np.argmin(np.abs(Ts - T_REF)))
        rows += long_rows("simulation_metrics", f"standard_5_28_dose{d:.0f}",
                          {"T_day": Ts[j], "x2_T": Xs[j, 1]})
    rows += long_rows("simulation_phase_stats", "smc_discrete", phase_stat_values(smc_rows))
    rows += long_rows("simulation_phase_stats", "smc_discrete_placeholder_excluded",
                      phase_stat_values(chk_rows))
    rows += long_rows("simulation_drug_schedule", "smc_discrete",
                      {k: [r[k] for r in smc_rows] for k in smc_rows[0]})
    # In the series below, u is the dose applied on [t, t + 1 day); empty on the last row.
    n = len(T_smc)
    rows += long_rows("simulation_timeseries", "smc_discrete", {
        "t_day": T_smc, "r": pad_nan(x2_ref_smc, n), "s": pad_nan(S_smc, n),
        "delta": pad_nan(D_smc, n), "u": pad_nan(U_smc[1:], n),
        "x1": X_smc[:, 0], "x2": X_smc[:, 1], "x3": X_smc[:, 2], "x4": X_smc[:, 3]})
    n = len(T_cont)
    rows += long_rows("simulation_timeseries", "smc_continuous", {
        "t_day": T_cont, "r": pad_nan(x2_ref_smc, n), "u": pad_nan(U_cont, n),
        "x1": X_cont[:, 0], "x2": X_cont[:, 1], "x3": X_cont[:, 2], "x4": X_cont[:, 3]})
    for d, (Xs, Us, Ts) in std_runs.items():
        n = len(Ts)
        rows += long_rows("simulation_timeseries", f"standard_5_28_dose{d:.0f}", {
            "t_day": Ts, "u": pad_nan(Us, n), "x1": Xs[:, 0], "x2": Xs[:, 1],
            "x3": Xs[:, 2], "x4": Xs[:, 3]})
    # Legacy daily layout (former SMC_daily.csv): u on row d includes the t = 0 placeholder.
    N = min(len(T_smc), len(U_smc), X_smc.shape[0])
    rows += long_rows("simulation_daily", "smc_discrete", {
        "day": np.arange(1, N + 1), "u": U_smc[:N], "x4": X_smc[:N, 3],
        "x1": X_smc[:N, 0], "x2": X_smc[:N, 1], "x3": X_smc[:N, 2]})
    rows += long_rows("simulation_full_schedule", "smc_discrete",
                      full_schedule_table(U_smc, T_smc, X_smc))
    write_sections(rows)
    print()

    # ---- Figures ------------------------------------------------------------
    print("Writing plots ...")
    with plt.rc_context(RC_RESULTS):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(T_smc, X_smc[:, 0], label="Glial cells ($x_1$) – discrete")
        ax.plot(T_cont, X_cont[:, 0], label="Glial cells ($x_1$) – continuous")
        ax.plot(T_smc, X_smc[:, 2], label="Neurons ($x_3$) – discrete")
        ax.plot(T_cont, X_cont[:, 2], label="Neurons ($x_3$) – continuous")
        finish(ax, "Glial and neuronal dynamics under sliding-mode control",
               "Time [days]", CELLS)
        save(fig, "smc_simulation_x1x3")

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(T_smc, X_smc[:, 1], label="Glioma cells ($x_2$) – discrete")
        ax.plot(T_cont, X_cont[:, 1], label="Glioma cells ($x_2$) – continuous")
        ax.plot(times_ref, x2_ref_smc, linestyle='--', color='black', linewidth=2,
                label="Tumor reference")
        finish(ax, "Tumor trajectory tracking under sliding-mode control",
               "Time [days]", "Normalized tumor population")
        save(fig, "smc_simulation_x2")

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(T_smc, X_smc[:, 3], label="Drug concentration ($x_4$)\n– discrete")
        ax.plot(T_cont, X_cont[:, 3], label="Drug concentration ($x_4$)\n– continuous")
        finish(ax, "Tissue-level chemotherapy concentration under SMC",
               "Time [days]", "Drug concentration")
        save(fig, "smc_simulation_x4")

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.step(T_smc, U_smc, where='post', label="Infusion rate – discrete")
        ax.plot(T_cont[:-1], U_cont, label="Infusion rate – continuous")
        finish(ax, "Chemotherapy infusion profile under sliding-mode control",
               "Time [days]", UNIT_U)
        save(fig, "smc_simulation_u")

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(T_smc[1:], S_smc, label="Sliding variable $S(t)$")
        ax.plot(T_smc[1:], D_smc, linestyle="--", label=r"Upper bound $+\delta(t)$")
        ax.plot(T_smc[1:], -D_smc, linestyle="--", label=r"Lower bound $-\delta(t)$")
        finish(ax, "Sliding surface and adaptive hysteresis band", "Time [days]",
               r"$S(t)$ and $\delta(t)$")
        save(fig, "smc_simulation_sliding_band")

    with plt.rc_context(RC_SCHEDULE):
        plot_schedule(smc_rows, '[SMC] Drug pulse durations (Discrete SMC)',
                      "smc_simulation_drug_schedule")

        X200, _, T200 = std_runs[200.0]
        X400, _, T400 = std_runs[400.0]
        lw = 1.2
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(T_smc, X_smc[:, 1], '-', label="Proposed SMC (Max Dose: 200)",
                color='tab:blue', lw=lw)
        ax.plot(T200, X200[:, 1], '-', label="Standard Protocol 5/28 (Dose: 200)",
                color='tab:red', lw=lw)
        ax.plot(T400, X400[:, 1], '-', label="Base Paper Protocol 5/28 (Dose: 400)",
                color='tab:orange', lw=lw)
        ax.plot(times_ref, x2_ref_smc, '--', color='k', lw=1.0, label="Reference Target")
        beautify_axes(ax, "Time [days]", "Normalized tumor population",
                      "Comparison: Proposed SMC vs Standard Protocols")
        ax.set_xlim(left=0, right=T_REF)
        ax.legend(loc='center right', frameon=False)
        fig.tight_layout()
        save(fig, "smc_simulation_comparison_standard")

    for name in PLOTS:
        print(f"  wrote {os.path.join(PLOT_DIR, FIGURE_FILES[name] + '.png')}")

if __name__ == "__main__":
    main()
