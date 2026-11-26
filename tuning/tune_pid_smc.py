"""tune_pid_smc.py

Differential-evolution tuning of the PID and SMC parameters (seeds 7, 11, 23). About 160 min on one core
(measured by the authors' run: 161 min). The published history is tuning_history.csv (114 445 evaluations).

Run:  python tuning/tune_pid_smc.py

Part of the code for: S. Rahimi, H. Moradi, "Robust drug scheduling for glioma chemotherapy based on a
multivariable dynamic model and state estimation via an extended Kalman filter" (Sharif University of Technology). MIT License, see LICENSE.
"""
import contextlib
import inspect
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sympy as sp
from scipy.optimize import differential_evolution

# =============================================================================
# SEARCH SETTINGS
# =============================================================================
A_REF_UNIFIED = 7.5         # tumor reference shaping a for PID and SMC (b = 0)
SEEDS = (7, 11, 23)         # three independent runs per controller
DE_STRATEGY = "best1bin"
DE_POLISH = False
DE_WORKERS = 1
DE_UPDATING = "immediate"
# popsize, mutation, recombination, tol, atol and init stay at the scipy defaults
# (read from scipy's signature and printed at run time).
BUDGET_MIN_PER_CONTROLLER = 60.0   # at most about 60 min for the three runs of ONE controller
# maxiter is FIXED (None would choose it from the timing below). The values come from the
# timed evaluation (budget / 3 runs / (t_eval * population) - 1) on this machine at normal
# speed: PID t_eval = 0.0533 s -> 249; SMC t_eval = 0.0766 s -> 148 (2026-11-14). They are
# fixed because the machine's speed varied by up to 4x during the first v2 attempt, which
# would otherwise change the budget.
MAXITER_PID = 249
MAXITER_SMC = 148
N_TIMING_EVALS = 3          # evaluations of the current parameters timed (printed for reference)
# Safety stop only, on the PROCESS CPU time of a run (time.process_time, which does not count
# sleep or other processes): a run is stopped after SAFETY_CAP_FACTOR x its budget share. It is
# set high so that it triggers only on a genuine hang; a stopped run is flagged "cap = yes".
SAFETY_CAP_FACTOR = 6.0

# --- Objective ----------------------------------------------------------------
W_PROT = 2.0                # weight of Jprot
W_SW = 2.0                  # weight of Jsw
TRACK_FLOOR = 1e-4          # floor of x2 and r in the log tracking error
PROT_FREE_DAYS = 4          # high-dose phase length without penalty [d]
PROT_CYCLE_DAYS = 28        # Jprot is divided by (T / 28); N_std = T / 28
J_FAIL = 1e4                # J of a non-finite, aborted or infeasible candidate

# --- Search variables: (name, lower, upper, scale) - fixed before the search ---
# PID: m_p, m_i, m_d in [0.1x, 10x] of the hand design (10, 1.8, 3.3); one common Ygain;
# gain-schedule sharpness; gain-schedule offset. The offset b is the centre of the tanh
# transition in z = t / T_ref (gain_profile); b in [0, 1] keeps that centre inside the
# horizon, and for a sharpness in [2, 15] the normalizer h(0) - h(1) of gain_profile stays
# >= 0.5*tanh(2) = 0.48, so the profile never degenerates (checked and printed at run time).
PID_SPACE = (("PID_M_P", 1.0, 100.0, "log"),
             ("PID_M_I", 0.18, 18.0, "log"),
             ("PID_M_D", 0.33, 33.0, "log"),
             ("PID_YGAIN", 2e5, 2e7, "log"),
             ("PID_A_GAIN", 2.0, 15.0, "log"),
             ("PID_B_GAIN", 0.0, 1.0, "lin"))
# SMC: same offset rule for the hysteresis band (s_curve in smc_reference_arrays).
SMC_SPACE = (("LAMBDA_P", 0.1, 10.0, "log"),        # lambda_0, coefficient of e
             ("LAMBDA_D", 0.1, 10.0, "log"),        # lambda_1, coefficient of de/dt
             ("LAMBDA_I", 1e-6, 1e-2, "log"),       # coefficient of int(e)
             ("DELTA_INIT", 1e-3, 1e-1, "log"),
             ("DELTA_FINAL", 1e-9, 1e-2, "log"),    # constraint: DELTA_FINAL <= DELTA_INIT
             ("A_DELTA", 2.0, 15.0, "log"),         # hysteresis sharpness
             ("B_DELTA", 0.0, 1.0, "lin"))          # hysteresis offset

# --- Reporting ----------------------------------------------------------------
BOUND_FLAG_FRAC = 0.02      # flag a parameter within 2 % of a bound (log scale for log variables)
AGREE_FRAC = 0.05           # runs "agree" if no parameter differs by >= 5 % of its range
PROGRESS_EVERY = 10         # progress line every N generations

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

# --- PID, Table 3: CURRENT values = hand design (the starting point of the search) ---
PID_A_REF = A_REF_UNIFIED   # reference shaping a (7.5, unified)
PID_B_REF = 0.0         # reference shaping b
PID_XGAIN = 1e9         # common scale of the initial gains (fixed, not tuned)
PID_M_P = 10.0          # KP_INIT = PID_M_P * PID_XGAIN
PID_M_I = 1.8           # KI_INIT = PID_M_I * PID_XGAIN
PID_M_D = 3.3           # KD_INIT = PID_M_D * PID_XGAIN
PID_YGAIN = 2e6         # K*_FINAL = K*_INIT / PID_YGAIN
PID_A_GAIN = 7          # gain-schedule sharpness a_p = a_i = a_d
PID_B_GAIN = 0.00       # gain-schedule offset b_p = b_i = b_d

# --- SMC, Table 4: CURRENT values = hand design (the starting point of the search) ---
SMC_A_REF = A_REF_UNIFIED   # reference shaping a (7.5, unified)
SMC_B_REF = 0.0         # reference shaping b
LAMBDA_P = 1.5          # coefficient of e      (lambda_0)
LAMBDA_D = 2.5          # coefficient of de/dt  (lambda_1)
LAMBDA_I = 1.0e-04      # coefficient of int(e)
GAMMA_1 = 1.0           # tanh smoothing on |e|     (continuous SMC only; not used here)
GAMMA_2 = 1.0           # tanh smoothing on |de/dt| (continuous SMC only; not used here)
DELTA_INIT = 0.01       # initial hysteresis half-width
DELTA_FINAL = 1e-9      # final hysteresis half-width
A_DELTA = 7.5           # hysteresis sharpness a_delta
B_DELTA = 0.0           # hysteresis offset b_delta

# --- Output -------------------------------------------------------------------
SHOW_FIGURES = False            # True -> also display each figure (plt.show); False -> headless, files only
SAVE_DPI = 600
# This file sits in tuning/ of the repository. Outputs: one csv with the full evaluation
# history, results/runs/tune_pid_smc.csv, and one PNG per controller.
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                      # running as a notebook: cwd is General/
    _HERE = os.getcwd()
ROOT = os.path.dirname(_HERE)          # the repository root
OUT_DIR = os.path.join(ROOT, "results", "runs")
RESULTS_FILE = os.path.join(OUT_DIR, "tune_pid_smc.csv")
PLOTS = ("tune_pid_convergence", "tune_smc_convergence")

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
# 2) S-CURVES
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


def gain_profile(start, end, times, T, sharpness, offset):
    """PID gain schedule from `start` to `end` over [0, T] (monotone)."""
    z = times / T
    h = 0.5 * (1 - np.tanh(sharpness * (z - offset)))    # decreasing in z
    denom = (h[0] - h[-1])
    if abs(denom) < 1e-12:
        r = z
    else:
        r = (h[0] - h) / (denom + 1e-12)                  # 0 -> 1
    prof = start + r * (end - start)
    if end < start:
        prof = np.minimum.accumulate(prof)
    elif end > start:
        prof = np.maximum.accumulate(prof)
    return prof


def smc_reference_arrays(x2_0, times_ref):
    x2_ref, S0, S1 = tumor_reference(x2_0, times_ref, SMC_A_REF, SMC_B_REF)
    z = times_ref / T_REF - SMC_B_REF
    sech2 = 1 / (np.cosh(SMC_A_REF * z) ** 2)
    dSdx = -0.5 * SMC_A_REF * sech2
    drdt_arr = x2_ref[0] * (dSdx / (S0 - S1)) * (1 / T_REF)
    d2Sdx2 = (SMC_A_REF * SMC_A_REF) * sech2 * np.tanh(SMC_A_REF * z)
    ddrdt_arr = x2_ref[0] * (d2Sdx2 / (S0 - S1)) * (1 / (T_REF ** 2))

    Sd0 = s_curve(0.0, A_DELTA, B_DELTA)
    Sd1 = s_curve(1.0, A_DELTA, B_DELTA)
    Sd = s_curve(times_ref / T_REF, A_DELTA, B_DELTA)
    delta_arr = DELTA_FINAL + ((Sd - Sd1) / (Sd0 - Sd1)) * (DELTA_INIT - DELTA_FINAL)
    delta_arr = np.minimum.accumulate(delta_arr)
    return x2_ref, drdt_arr, ddrdt_arr, delta_arr

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


def simulate_pid(x0, x2_ref, Kp_prof, Ki_prof, Kd_prof):
    """Gain-scheduled PID; the dose is the nearest of {U_MIN, U_MAX}.

    The loop runs once per reference sample (len(x2_ref) steps). The first dose
    is forced to U_MAX. The PID acts on the true state x2 (no observer here).
    """
    x = x0.copy()
    integ = 0.0
    prev_err = 0.0
    X_hist = [x.copy()]
    U_hist = []
    for k, r in enumerate(x2_ref):
        err = x[1] - r
        integ += err * DT
        deriv = (err - prev_err) / DT
        prev_err = err
        raw_u = Kp_prof[k]*err + Ki_prof[k]*integ + Kd_prof[k]*deriv
        u = U_MIN if abs(raw_u - U_MIN) <= abs(raw_u - U_MAX) else U_MAX
        if k == 0:
            u = U_MAX
        U_hist.append(u)
        x = rk4_step(plant_nominal, x, u, DT)
        x[1] = max(x[1], 0.0)
        X_hist.append(x.copy())
    return np.array(X_hist), np.array(U_hist)


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

# =============================================================================
# 5) ONE DESIGN RUN
# =============================================================================
SPACES = {"PID": PID_SPACE, "SMC": SMC_SPACE}
N_DAYS = int(T_REF / DT)              # 2000 daily doses
TIMES = np.arange(N_DAYS + 1) * DT    # states at t = 0 .. 2000
N_STD = T_REF / PROT_CYCLE_DAYS       # number of high-dose phases of the 5/28 protocol
X_BURN = None                         # nominal pre-conditioned state (set in main)
F_FUNC = G_FUNC = None                # SMC affine terms (set in main)


@contextlib.contextmanager
def design_parameters(values):
    """Temporarily set this file's design constants (module globals) to `values`, so
    that the copied simulators, which read them as globals, run the candidate. The
    constants are restored on exit, also after an error."""
    g = globals()
    saved = {name: g[name] for name in values}
    try:
        g.update(values)
        yield
    finally:
        g.update(saved)


def run_pid(values):
    """pid_simulation's main() for one parameter set (gains built as in pid_simulation v2:
    K*_INIT = m * PID_XGAIN, K*_FINAL = K*_INIT / PID_YGAIN).
    Returns t (0..2000), X (states at t), U (the 2000 daily doses), r (reference at t)."""
    with design_parameters(values):
        kp0, ki0, kd0 = PID_M_P * PID_XGAIN, PID_M_I * PID_XGAIN, PID_M_D * PID_XGAIN
        x2_ref_pid, _, _ = tumor_reference(X_BURN[1], TIMES, PID_A_REF, PID_B_REF)
        Kp_prof = gain_profile(kp0, kp0 / PID_YGAIN, TIMES, T_REF, PID_A_GAIN, PID_B_GAIN)
        Ki_prof = gain_profile(ki0, ki0 / PID_YGAIN, TIMES, T_REF, PID_A_GAIN, PID_B_GAIN)
        Kd_prof = gain_profile(kd0, kd0 / PID_YGAIN, TIMES, T_REF, PID_A_GAIN, PID_B_GAIN)
        X_pid, U_pid = simulate_pid(X_BURN, x2_ref_pid, Kp_prof, Ki_prof, Kd_prof)
    # X_pid has one row per reference sample plus one (t = 0..2001); U_pid[k] is the
    # dose on [k, k+1).
    return TIMES, X_pid[:N_DAYS + 1], U_pid[:N_DAYS], x2_ref_pid


def run_smc(values):
    """smc_simulation's discrete-SMC run for one parameter set (same outputs as run_pid)."""
    with design_parameters(values):
        x2_ref_smc, drdt_arr, ddrdt_arr, delta_arr = smc_reference_arrays(X_BURN[1], TIMES)
        X_smc, U_smc, T_smc, _, _ = simulate_smc_discrete(
            X_BURN.copy(), x2_ref_smc, drdt_arr, ddrdt_arr, delta_arr, F_FUNC, G_FUNC)
    # U_smc[0] is the t = 0 placeholder; U_smc[k], k >= 1, is the dose on [k-1, k).
    return T_smc[:N_DAYS + 1], X_smc[:N_DAYS + 1], U_smc[1:N_DAYS + 1], x2_ref_smc


def infeasible(ctrl, v):
    """Constraint check (no simulation for an infeasible candidate)."""
    if ctrl == "SMC":
        if not (v["LAMBDA_D"] * v["LAMBDA_P"] > v["LAMBDA_I"]):
            return "infeasible (Hurwitz)"
        if not (v["DELTA_FINAL"] <= v["DELTA_INIT"]):
            return "infeasible (delta_final > delta_init)"
    return None

# =============================================================================
# 6) OBJECTIVE AND METRICS
# =============================================================================
def run_starts_lengths(mask):
    """Start indices and lengths of the runs of consecutive True values (last run included)."""
    m = np.concatenate(([0], np.asarray(mask, dtype=int), [0]))
    d = np.diff(m)
    st = np.where(d == 1)[0]
    return st, np.where(d == -1)[0] - st


def run_lengths(mask):
    return run_starts_lengths(mask)[1]


def objective_terms(t, X, U, r):
    """Jtrack, Jprot, Jsw and N_high of one run (definitions in the header)."""
    x2 = X[:, 1]
    el = np.log10(np.maximum(x2, TRACK_FLOOR)) - np.log10(np.maximum(r, TRACK_FLOOR))
    jtrack = np.trapezoid(el**2, t) / T_REF
    high = run_lengths(np.abs(U - U_MAX) <= 1e-9) * DT
    jprot = np.sum(np.maximum(0.0, high - PROT_FREE_DAYS)**2) / (T_REF / PROT_CYCLE_DAYS)
    jsw = max(0.0, high.size - N_STD) / N_STD
    return float(jtrack), float(jprot), float(jsw), int(high.size)


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


def day_or_nan(T, i):
    return np.nan if i is None else T[i]


METRIC_LABELS = (
    ("t_x2_first", "first day x2 < 1e-4 [d]"),
    ("regrowth", "days after it with x2 > 1e-4 [%]"),
    ("hi_after", "high-dose phases after that day"),
    ("t_x2_perm", "permanent day x2 <= 1e-4 [d]"),
    ("t_x1_first", "glial recovery x1 >= 0.99, first [d]"),
    ("t_x1_perm", "glial recovery x1 >= 0.99, permanent [d]"),
    ("x1_T", "x1(2000)"),
    ("x3_T", "x3(2000)"),
    ("x2_T", "x2(2000)"),
    ("hi_count", "N_high (high-dose phases)"),
    ("hi_mean", "high-dose phases: mean [d]"),
    ("hi_max", "high-dose phases: max [d]"),
    ("hi_1day", "high-dose phases of 1 day"),
    ("hi_gt5", "high-dose phases longer than 5 d"),
    ("lo_count", "maintenance phases: count"),
    ("lo_mean", "maintenance phases: mean [d]"),
    ("cum_dose", "cumulative dose [mg m^-2]"),
)


def schedule_metrics(t, X, U):
    """Metrics of one run. Phases are the runs of the 2000 daily doses (u = 200: high-dose,
    u = 50: maintenance), the last, horizon-truncated run included. 'After that day' =
    after the first day with x2 < 1e-4. Cumulative dose = sum of the daily doses x DT."""
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    st, high = run_starts_lengths(np.abs(U - U_MAX) <= 1e-9)
    high = high * DT
    low = run_lengths(np.abs(U - U_MIN) <= 1e-9) * DT
    i2 = first_index(x2 < X2_THRESHOLD)
    after = x2[i2 + 1:] if i2 is not None else np.array([])
    return {"t_x2_first": day_or_nan(t, i2),
            "regrowth": 100.0 * np.mean(after > X2_THRESHOLD) if after.size else np.nan,
            "hi_after": int(np.sum(st > i2)) if i2 is not None else np.nan,
            "t_x2_perm": day_or_nan(t, permanent_index(x2 <= X2_THRESHOLD)),
            "t_x1_first": day_or_nan(t, first_index(x1 >= X1_RECOVERY)),
            "t_x1_perm": day_or_nan(t, permanent_index(x1 >= X1_RECOVERY)),
            "x1_T": x1[-1], "x3_T": x3[-1], "x2_T": x2[-1],
            "hi_count": high.size, "hi_mean": high.mean() if high.size else np.nan,
            "hi_max": high.max() if high.size else np.nan,
            "hi_1day": int(np.sum(high == 1)), "hi_gt5": int(np.sum(high > 5)),
            "lo_count": low.size, "lo_mean": low.mean() if low.size else np.nan,
            "cum_dose": float(np.sum(U) * DT)}

# =============================================================================
# 7) EVALUATOR
# =============================================================================
ALL_NAMES = tuple(n for n, *_ in PID_SPACE) + tuple(n for n, *_ in SMC_SPACE)


def to_values(space, z):
    return {n: float(10.0 ** v) if sc == "log" else float(v)
            for (n, _, _, sc), v in zip(space, z)}


def to_z(space, values):
    return np.array([np.log10(values[n]) if sc == "log" else values[n]
                     for n, _, _, sc in space])


def z_bounds(space):
    return [(np.log10(lo), np.log10(hi)) if sc == "log" else (lo, hi)
            for _, lo, hi, sc in space]


def position(space_row, v):
    """Position of v in its search range, 0 = lower bound, 1 = upper (log for log vars)."""
    _, lo, hi, sc = space_row
    if sc == "log":
        return (np.log10(v) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    return (v - lo) / (hi - lo)


class Evaluator:
    """The DE objective for one controller; every call is appended to `history`."""

    def __init__(self, ctrl, history):
        self.ctrl = ctrl
        self.space = SPACES[ctrl]
        self.run = run_pid if ctrl == "PID" else run_smc
        self.history = history
        self.jtrack_ref = None
        self.role, self.seed, self.gen, self.n = "", "", 0, 0
        self.t0 = time.perf_counter()

    def evaluate(self, values):
        """(J, Jtrack, Jprot, Jsw, N_high, status) of one parameter set."""
        jtrack = jprot = jsw = np.nan
        n_high = -1
        status = infeasible(self.ctrl, values)
        if status is None:
            try:
                with np.errstate(all="ignore"):
                    t, X, U, r = self.run(values)
                    if (X.shape[0] != N_DAYS + 1 or U.shape[0] != N_DAYS
                            or not np.all(np.isfinite(X)) or not np.all(np.isfinite(U))):
                        status = "nonfinite"
                    else:
                        jtrack, jprot, jsw, n_high = objective_terms(t, X, U, r)
                        status = "ok"
            except (FloatingPointError, OverflowError, ZeroDivisionError, ValueError) as exc:
                status = f"aborted ({type(exc).__name__})"
        J = J_FAIL
        if status == "ok":
            J = jtrack / self.jtrack_ref + W_PROT * jprot + W_SW * jsw
            if not np.isfinite(J):
                status, J = "nonfinite", J_FAIL
        return float(J), jtrack, jprot, jsw, n_high, status

    def record(self, values, res):
        J, jtrack, jprot, jsw, n_high, status = res
        self.n += 1
        row = {"controller": self.ctrl, "role": self.role, "seed": self.seed,
               "eval": self.n, "generation": self.gen, "status": status, "J": J,
               "log10_1pJ": np.log10(1.0 + J), "Jtrack": jtrack,
               "Jtrack_over_ref": jtrack / self.jtrack_ref, "Jprot": jprot, "Jsw": jsw,
               "N_high": n_high}
        row.update({k: values.get(k, np.nan) for k in ALL_NAMES})
        row["elapsed_s"] = time.perf_counter() - self.t0
        self.history.append(row)

    def __call__(self, z):
        values = to_values(self.space, z)
        res = self.evaluate(values)
        self.record(values, res)
        return np.log10(1.0 + res[0])

# =============================================================================
# 8) TUNING OF ONE CONTROLLER
# =============================================================================
DE_DEFAULTS = {k: v.default for k, v in inspect.signature(differential_evolution).parameters.items()
               if k in ("popsize", "mutation", "recombination", "tol", "atol", "init")}


def offset_check(space):
    """Smallest normalizer h(0) - h(1) of the tanh schedule over the sharpness / offset box."""
    rows = {n: (lo, hi) for n, lo, hi, _ in space}
    sk, bk = ("PID_A_GAIN", "PID_B_GAIN") if "PID_A_GAIN" in rows else ("A_DELTA", "B_DELTA")
    s_grid = np.linspace(*rows[sk], 27)
    b_grid = np.linspace(*rows[bk], 101)
    S, B = np.meshgrid(s_grid, b_grid)
    den = 0.5 * (np.tanh(S * (1 - B)) + np.tanh(S * B))
    return sk, bk, float(den.min())


def tune(ctrl, history):
    space = SPACES[ctrl]
    names = [n for n, *_ in space]
    current = {k: float(globals()[k]) for k in names}
    bounds = z_bounds(space)
    x0 = to_z(space, current)
    ev = Evaluator(ctrl, history)

    # ---- Jtrack_ref: the current parameters at a = 7.5 (Stage A) -------------
    ev.jtrack_ref = 1.0
    res0 = ev.evaluate(current)
    if res0[-1] != "ok":
        raise RuntimeError(f"{ctrl}: the current parameters do not simulate ({res0[-1]})")
    ev.jtrack_ref = res0[1]
    res0 = ev.evaluate(current)
    ev.role, ev.seed = "current", "current"
    ev.record(current, res0)
    J_cur = res0[0]
    print(f"[{ctrl}] Jtrack_ref (current parameters, a = {A_REF_UNIFIED}, no stop rule) = "
          f"{ev.jtrack_ref:.10g};  J_current = {J_cur:.10g}")
    sk, bk, den = offset_check(space)
    print(f"[{ctrl}] offset bound {bk} in [0, 1]: centre of the tanh transition inside the "
          f"horizon; min of h(0) - h(1) over {sk} x {bk} = {den:.4f} (> 0: no degenerate profile)")

    # ---- Timing of one evaluation -> maxiter -----------------------------------
    ev.role, ev.seed = "timing", "timing"
    t_each = []
    for _ in range(N_TIMING_EVALS):
        t1 = time.perf_counter()
        ev(x0)
        t_each.append(time.perf_counter() - t1)
    t_eval = float(np.mean(t_each))
    n_pop = max(5, DE_DEFAULTS["popsize"] * len(names))
    budget_s = BUDGET_MIN_PER_CONTROLLER * 60.0
    maxiter_set = MAXITER_PID if ctrl == "PID" else MAXITER_SMC
    maxiter_auto = max(1, int(budget_s / len(SEEDS) / (t_eval * n_pop)) - 1)
    maxiter = maxiter_auto if maxiter_set is None else int(maxiter_set)
    cap_s = SAFETY_CAP_FACTOR * budget_s / len(SEEDS)
    print(f"[{ctrl}] one evaluation = {t_eval:.4f} s (mean of {N_TIMING_EVALS}); population "
          f"= {n_pop}; maxiter = {maxiter} "
          f"({'chosen from the timing' if maxiter_set is None else 'set in SEARCH SETTINGS'})")

    # ---- Three independent DE runs ---------------------------------------------
    runs = []
    for seed in SEEDS:
        ev.role, ev.seed, ev.gen, ev.n = "search", seed, 0, 0
        t_run = time.perf_counter()
        c_run = time.process_time()
        state = {"cap": False}

        def make_callback(seed, t_run, c_run, state):
            # scipy passes an OptimizeResult only to a callback whose single parameter is
            # named `intermediate_result`, hence this factory.
            def callback(intermediate_result):
                ev.gen += 1
                el = time.perf_counter() - t_run
                cpu = time.process_time() - c_run
                if ev.gen % PROGRESS_EVERY == 0:
                    print(f"  [{ctrl} seed {seed:>2}] gen {ev.gen:4d}  nfev {ev.n:6d}  "
                          f"best J {10.0 ** intermediate_result.fun - 1.0:.6g}  {el:7.1f} s wall"
                          f"  {cpu:7.1f} s cpu", flush=True)
                if cpu > cap_s:
                    state["cap"] = True
                    return True
                return False
            return callback

        res = differential_evolution(
            ev, bounds, strategy=DE_STRATEGY, maxiter=maxiter, rng=seed,
            callback=make_callback(seed, t_run, c_run, state), polish=DE_POLISH,
            updating=DE_UPDATING, workers=DE_WORKERS, x0=x0)
        wall = time.perf_counter() - t_run
        best = to_values(space, res.x)
        terms = ev.evaluate(best)          # J terms of this run's winner (not recorded twice)
        runs.append({"seed": seed, "J": 10.0 ** res.fun - 1.0, "values": best,
                     "terms": terms, "nfev": res.nfev, "nit": res.nit, "wall_s": wall,
                     "message": res.message, "cap": state["cap"]})
        print(f"  [{ctrl} seed {seed:>2}] done: J = {runs[-1]['J']:.10g}, nfev = {res.nfev}, "
              f"nit = {res.nit}, {wall:.1f} s, {res.message}", flush=True)

    # ---- Winner (best run; current kept if no run beats it) -------------------
    i_best = int(np.argmin([r["J"] for r in runs]))
    beats = runs[i_best]["J"] < J_cur
    winner = runs[i_best]["values"] if beats else dict(current)
    ev.role, ev.seed = "winner_check", runs[i_best]["seed"] if beats else "current"
    res_w = ev.evaluate(winner)
    ev.record(winner, res_w)

    t, X, U, _ = ev.run(current)
    m_cur = schedule_metrics(t, X, U)
    t, X, U, _ = ev.run(winner)
    m_win = schedule_metrics(t, X, U)

    summary = {"ctrl": ctrl, "space": space, "names": names, "current": current,
               "res_cur": res0, "J_cur": J_cur, "jtrack_ref": ev.jtrack_ref, "runs": runs,
               "i_best": i_best, "beats": beats, "winner": winner, "res_win": res_w,
               "m_cur": m_cur, "m_win": m_win, "t_eval": t_eval, "n_pop": n_pop,
               "maxiter": maxiter, "maxiter_auto": maxiter_set is None, "cap_s": cap_s,
               "offset_check": (sk, bk, den)}
    print_block(summary)
    return summary

# =============================================================================
# 9) CONSOLE BLOCK
# =============================================================================
def fmt_day(v):
    return "not reached" if not np.isfinite(v) else f"{v:.0f}"


def fmt_metric(key, v):
    if key.startswith("t_"):
        return fmt_day(v)
    if key in ("x1_T", "x3_T"):
        return f"{v:.6f}"
    if key == "x2_T":
        return f"{v:.6e}"
    if key in ("regrowth",):
        return "-" if not np.isfinite(v) else f"{v:.1f}"
    if key.endswith("_mean"):
        return "-" if not np.isfinite(v) else f"{v:.2f}"
    if key == "cum_dose":
        return f"{v:.0f}"
    return "-" if not np.isfinite(v) else f"{v:.0f}"


def print_block(s):
    ctrl, space, names, runs = s["ctrl"], s["space"], s["names"], s["runs"]
    W = 100
    print()
    print("=" * W)
    print(f"{ctrl} - differential-evolution tuning v2, tumor reference a = {A_REF_UNIFIED}, b = 0, "
          f"nominal plant, no stop rule")
    print("=" * W)
    print("objective   : el = log10(max(x2,1e-4)) - log10(max(r,1e-4)), t = 0..2000 (states)")
    print("              Jtrack = trapz(t, el^2)/2000")
    print("              Jprot  = sum over high-dose phases (runs of u = 200) of max(0, L-4)^2 / (2000/28)")
    print(f"              Jsw    = max(0, N_high - N_std)/N_std,  N_std = 2000/28 = {N_STD:.4f}")
    print(f"              J = Jtrack/Jtrack_ref + {W_PROT:g}*Jprot + {W_SW:g}*Jsw,  Jtrack_ref = "
          f"{s['jtrack_ref']:.10g} (current parameters, Stage A)")
    print(f"              failed / non-finite / infeasible: J = {J_FAIL:g}; "
          "minimized: log10(1 + J)")
    print(f"optimizer   : scipy {scipy.__version__} differential_evolution (Storn & Price 1997), "
          f"strategy = {DE_STRATEGY}")
    print(f"              popsize = {DE_DEFAULTS['popsize']} (x {len(names)} = {s['n_pop']} "
          f"members), mutation = {DE_DEFAULTS['mutation']}, recombination = "
          f"{DE_DEFAULTS['recombination']}, tol = {DE_DEFAULTS['tol']}, atol = "
          f"{DE_DEFAULTS['atol']}, init = {DE_DEFAULTS['init']}")
    print(f"              polish = {DE_POLISH}, workers = {DE_WORKERS}, updating = "
          f"{DE_UPDATING}, x0 = current parameters")
    print(f"seeds       : {', '.join(str(r['seed']) for r in runs)}")
    print(f"maxiter     : {s['maxiter']}  ({'from timing: ' if s['maxiter_auto'] else 'fixed; timing: '}"
          f"{s['t_eval']:.4f} s per evaluation, budget {BUDGET_MIN_PER_CONTROLLER:g} min for "
          f"{len(runs)} runs); safety cap {s['cap_s']:.0f} s process CPU time per run")
    print(f"evaluations : {sum(r['nfev'] for r in runs)} in the searches "
          f"({' + '.join(str(r['nfev']) for r in runs)})")
    print(f"runtime     : {sum(r['wall_s'] for r in runs) / 60:.1f} min "
          f"({' + '.join(f'{r['wall_s'] / 60:.1f}' for r in runs)})")
    if ctrl == "PID":
        print(f"structure   : K*_INIT = m * Xgain (Xgain = {PID_XGAIN:g} fixed), K*_FINAL = "
              "K*_INIT / Ygain; sharpness and offset common to P, I, D")
        print("constraint  : none")
    else:
        print("structure   : gamma_1 = gamma_2 = 1 fixed (continuous SMC only)")
        print("constraint  : lambda_1 * lambda_0 > lambda_I (LAMBDA_D * LAMBDA_P > LAMBDA_I, "
              "Hurwitz s^3 + l1 s^2 + l0 s + lI); DELTA_FINAL <= DELTA_INIT")
    sk, bk, den = s["offset_check"]
    print(f"offset bound: {bk} in [0, 1] keeps the transition centre inside the horizon; "
          f"min h(0)-h(1) = {den:.4f}")
    print()
    print(f"  {'bounds':<12} {'lower':>13} {'upper':>13} {'scale':>6} {'current':>13}")
    for n, lo, hi, sc in space:
        print(f"  {n:<12} {lo:13.6g} {hi:13.6g} {sc:>6} {s['current'][n]:13.6g}")
    print()
    print(f"  {'run':<9} {'J':>14} {'Jtrack/ref':>12} {'Jprot':>10} {'Jsw':>10} {'N_high':>7} "
          f"{'nfev':>7} {'nit':>5} {'min':>6} {'cap':>4}  stop")
    J, jt, jp, jsw, nh, _ = s["res_cur"]
    print(f"  {'current':<9} {J:14.8g} {jt / s['jtrack_ref']:12.6g} {jp:10.6g} {jsw:10.6g} "
          f"{nh:7d} {'-':>7} {'-':>5} {'-':>6} {'-':>4}")
    for r in runs:
        J, jt, jp, jsw, nh, _ = r["terms"]
        print(f"  {'seed ' + str(r['seed']):<9} {J:14.8g} {jt / s['jtrack_ref']:12.6g} "
              f"{jp:10.6g} {jsw:10.6g} {nh:7d} {r['nfev']:7d} {r['nit']:5d} "
              f"{r['wall_s'] / 60:6.1f} {'yes' if r['cap'] else 'no':>4}  {r['message']}")
    Js = np.array([r["J"] for r in runs])
    print(f"  spread of the runs' best J: min {Js.min():.8g}, max {Js.max():.8g}, "
          f"(max-min)/min = {(Js.max() - Js.min()) / Js.min():.3%}")
    print()
    print(f"  {'parameter':<12} " + " ".join(f"{'seed ' + str(r['seed']):>14}" for r in runs)
          + f" {'range %':>8}")
    agree = True
    for row in space:
        n = row[0]
        pos = [position(row, r["values"][n]) for r in runs]
        rng_pct = (max(pos) - min(pos)) * 100
        agree &= rng_pct < AGREE_FRAC * 100
        print(f"  {n:<12} " + " ".join(f"{r['values'][n]:14.6g}" for r in runs)
              + f" {rng_pct:8.1f}")
    print(f"  convergence: the three runs {'AGREE' if agree else 'DO NOT AGREE'} (criterion: "
          f"every parameter within {AGREE_FRAC:.0%} of its search range across runs)")
    print()
    if s["beats"]:
        print(f"winner      : seed {runs[s['i_best']]['seed']} (J {s['res_win'][0]:.10g} < "
              f"current {s['J_cur']:.10g})")
    else:
        print("winner      : NO run beats the current parameters -> current parameters kept")
    print("winner parameters (full precision) and position in the search range "
          "(0 = lower bound, log for log variables):")
    flags = []
    for row in space:
        n = row[0]
        p = position(row, s["winner"][n])
        near = p < BOUND_FLAG_FRAC or p > 1 - BOUND_FLAG_FRAC
        if near:
            flags.append(n)
        print(f"  {n:<12} = {s['winner'][n]!r:<24} position {p:6.3f}"
              f"{'   <-- within 2% of a bound' if near else ''}")
    print(f"  parameters within {BOUND_FLAG_FRAC:.0%} of a bound: "
          f"{', '.join(flags) if flags else 'none'}")
    print()
    print(f"  {'metric':<42} {'current':>14} {'winner':>14}")
    for key, label in METRIC_LABELS:
        print(f"  {label:<42} {fmt_metric(key, s['m_cur'][key]):>14} "
              f"{fmt_metric(key, s['m_win'][key]):>14}")
    print("  (phases: runs of the 2000 daily doses, the last, horizon-truncated run included)")
    print()
    print("ready-to-paste parameter block:")
    for n in names:
        print(f"{n} = {s['winner'][n]!r}")
    if ctrl == "PID":
        w = s["winner"]
        k0 = {k: w[f"PID_M_{k[1]}"] * PID_XGAIN for k in ("KP", "KI", "KD")}
        print("# derived: " + ", ".join(f"{k}_INIT = {v:.10g}, {k}_FINAL = {v / w['PID_YGAIN']:.10g}"
                                        for k, v in k0.items()))
    print("=" * W)
    print(flush=True)

# =============================================================================
# 10) FIGURE
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
FIGSIZE = (9, 4.2)


def plot_convergence(s, hist):
    ctrl = s["ctrl"]
    with plt.rc_context({**RC_PRECONDITIONING, "lines.linewidth": 1.6}):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for r in s["runs"]:
            h = hist[(hist["controller"] == ctrl) & (hist["role"] == "search")
                     & (hist["seed"] == r["seed"])]
            ax.plot(h["eval"].to_numpy(), np.minimum.accumulate(h["J"].to_numpy()),
                    label=f"seed {r['seed']} (best J = {r['J']:.4g})")
        ax.axhline(s["J_cur"], color="black", linestyle="--", linewidth=1.0,
                   label=f"current parameters (J = {s['J_cur']:.4g})")
        ax.set_yscale("log")
        ax.set_xlabel("Evaluation")
        ax.set_ylabel("Best objective $J$ so far")
        ax.set_title(f"{ctrl}: differential-evolution convergence ($a$ = {A_REF_UNIFIED})")
        ax.legend(frameon=False, loc="best")
        ax.margins(x=0)
        fig.tight_layout()
        name = f"tune_{ctrl.lower()}_convergence"
        fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=SAVE_DPI, bbox_inches="tight")
        if SHOW_FIGURES:
            plt.show()
        plt.close(fig)
    print(f"  wrote {os.path.relpath(os.path.join(OUT_DIR, name + '.png'), ROOT)}")

# =============================================================================
# 11) MAIN
# =============================================================================
def main():
    global X_BURN, F_FUNC, G_FUNC
    if not SHOW_FIGURES:
        plt.switch_backend("Agg")
    os.makedirs(OUT_DIR, exist_ok=True)
    t_all = time.perf_counter()

    _, X_pre = simulate_preconditioning()
    X_BURN = X_pre[-1].copy()
    F_FUNC, G_FUNC = derive_smc_affine_terms()
    print("=" * 100)
    print("tune_pid_smc (v2) - console output")
    print("=" * 100)
    print(f"numpy {np.__version__}, scipy {scipy.__version__}, sympy {sp.__version__}; "
          f"x(T_start) = {np.array2string(X_BURN, precision=6)}")
    print()

    history = []
    summaries = [tune(ctrl, history) for ctrl in ("PID", "SMC")]

    hist = pd.DataFrame(history)
    hist["seed"] = hist["seed"].astype(str)
    hist.to_csv(RESULTS_FILE, index=False)
    print(f"  wrote {os.path.relpath(RESULTS_FILE, ROOT)}  ({len(hist)} evaluations)")
    for s in summaries:
        for r in s["runs"]:
            r["seed"] = str(r["seed"])
        plot_convergence(s, hist)
    print(f"total runtime {(time.perf_counter() - t_all) / 60:.1f} min")

if __name__ == "__main__":
    main()
