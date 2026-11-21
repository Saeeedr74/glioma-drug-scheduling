"""pid_simulation.py

Gain-scheduled PID with discrete doses {50, 200} on the nominal model (final v2 parameters).

Run:  python controllers/pid/pid_simulation.py

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

# --- PID, Table 3 -------------------------------------------------------------
PID_A_REF = 7.5         # reference shaping a; unified with SMC to 7.5 (retune 2026-11-14; was 8)
PID_B_REF = 0.0         # reference shaping b
# Gain structure of the hand design: initial gain = multiplier * PID_XGAIN,
# final gain = initial gain / PID_YGAIN (one common ratio for P, I and D).
PID_XGAIN = 1e9         # common scale of the initial gains (fixed, not tuned)
PID_M_P = 7.507644379102649        # Kp multiplier (KP_INIT = PID_M_P * PID_XGAIN); tuned v2 by General/tune_pid_smc (DE seed 7, a = 7.5, 2026-11-14); hand design 10.0
PID_M_I = 0.537876054596461        # Ki multiplier (KI_INIT = PID_M_I * PID_XGAIN); tuned v2 by General/tune_pid_smc (DE seed 7, a = 7.5, 2026-11-14); hand design 1.8
PID_M_D = 0.7406974830018409       # Kd multiplier (KD_INIT = PID_M_D * PID_XGAIN); tuned v2 by General/tune_pid_smc (DE seed 7, a = 7.5, 2026-11-14); hand design 3.3
PID_YGAIN = 676413.2897637554      # initial / final gain ratio; tuned v2 by General/tune_pid_smc (DE seed 7, a = 7.5, 2026-11-14); hand design 2e6
KP_INIT = PID_M_P * PID_XGAIN
KI_INIT = PID_M_I * PID_XGAIN
KD_INIT = PID_M_D * PID_XGAIN
KP_FINAL = KP_INIT / PID_YGAIN
KI_FINAL = KI_INIT / PID_YGAIN
KD_FINAL = KD_INIT / PID_YGAIN
PID_A_GAIN = 14.904652692478784    # gain-schedule sharpness; tuned v2 by General/tune_pid_smc (DE seed 7, a = 7.5, 2026-11-14); hand design 7
PID_B_GAIN = 0.37599424644399926   # gain-schedule offset; tuned v2 by General/tune_pid_smc (DE seed 7, a = 7.5, 2026-11-14); hand design 0.00

# --- Output -------------------------------------------------------------------
SHOW_FIGURES = False            # True -> also display each figure (plt.show); False -> headless, files only
SAVE_DPI = 600
# Paths are relative to the repository root (REPO). Outputs: one csv, results/runs/pid_simulation.csv, and the
# diagnostic plots of this script in results/diagnostic_plots (file names in FIGURE_FILES).
ROOT = "."                             # base for the printed paths
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # repository root
RESULTS_DIR = os.path.join(REPO, "results", "runs")
PLOT_DIR = os.path.join(REPO, "results", "diagnostic_plots")
RESULTS_FILE = os.path.join(RESULTS_DIR, "pid_simulation.csv")
FIGURE_FILES = {
    "pid_simulation_u": "PID_Fig4_u",
    "pid_simulation_x4": "PID_Fig4_x4",
    "pid_simulation_drug_schedule": "PID_Fig4_drug_schedule",
    "pid_simulation_x1x2x3": "PID_Fig5_states_x1x2x3",
    "pid_simulation_gains": "PID_Fig6_gains",
}
SECTION_PREFIX = "simulation_"   # this script writes (and replaces) only these sections
SECTION_ORDER = ("simulation_", "observer_", "uncertainty_")

# ---- the tuned / design parameters of this notebook (identical in the simulation,
#      observer and uncertainty notebooks of the same controller) --------------------
for _k in ('PID_A_REF', 'PID_B_REF', 'PID_XGAIN', 'PID_M_P', 'PID_M_I', 'PID_M_D', 'PID_YGAIN', 'KP_INIT', 'KI_INIT', 'KD_INIT', 'KP_FINAL', 'KI_FINAL', 'KD_FINAL', 'PID_A_GAIN', 'PID_B_GAIN'):
    if _k in globals():
        print(f"{_k:<12} = {globals()[_k]!r}")

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
# 2) S-CURVES - reference trajectory and PID gain schedule
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

# =============================================================================
# 3) SIMULATIONS
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

# =============================================================================
# 4) METRICS AND SCHEDULE STATISTICS
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
# 5) RESULTS FILE (tidy CSV, read-modify-write)
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
# 6) FIGURE STYLE
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
# 7) MAIN
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

    # ---- Reference and gain profiles ----------------------------------------
    times_ref = np.arange(int(T_REF / DT) + 1) * DT
    x2_ref_pid, _, _ = tumor_reference(x_burn[1], times_ref, PID_A_REF, PID_B_REF)
    Kp_prof = gain_profile(KP_INIT, KP_FINAL, times_ref, T_REF, PID_A_GAIN, PID_B_GAIN)
    Ki_prof = gain_profile(KI_INIT, KI_FINAL, times_ref, T_REF, PID_A_GAIN, PID_B_GAIN)
    Kd_prof = gain_profile(KD_INIT, KD_FINAL, times_ref, T_REF, PID_A_GAIN, PID_B_GAIN)

    # ---- Simulation ---------------------------------------------------------
    X_pid, U_pid = simulate_pid(x_burn, x2_ref_pid, Kp_prof, Ki_prof, Kd_prof)
    T_pid_states = np.arange(X_pid.shape[0]) * DT
    T_pid_control = np.arange(1, len(U_pid) + 1) * DT
    pid_rows = schedule_rows(T_pid_control, U_pid, X_pid)

    # ---- Console ------------------------------------------------------------
    print("=" * 72)
    print("pid_simulation - console output for FACTS.md")
    print("=" * 72)
    print_metrics("PID (discrete)", T_pid_states, X_pid, "permanent")
    print_phase_stats("PID", pid_rows)
    print_schedule_table("[PID] drug administration schedule", pid_rows)

    # ---- Results file -------------------------------------------------------
    print("Writing csv ...")
    rows = long_rows("simulation_metrics", "pid_discrete", metric_values(T_pid_states, X_pid))
    rows += long_rows("simulation_phase_stats", "pid_discrete", phase_stat_values(pid_rows))
    rows += long_rows("simulation_drug_schedule", "pid_discrete",
                      {k: [r[k] for r in pid_rows] for k in pid_rows[0]})
    # u on row t is the dose applied on [t, t + 1 day); empty on the last row.
    n = len(T_pid_states)
    rows += long_rows("simulation_timeseries", "pid_discrete", {
        "t_day": T_pid_states, "r": pad_nan(x2_ref_pid, n), "Kp": pad_nan(Kp_prof, n),
        "Ki": pad_nan(Ki_prof, n), "Kd": pad_nan(Kd_prof, n), "u": pad_nan(U_pid, n),
        "x1": X_pid[:, 0], "x2": X_pid[:, 1], "x3": X_pid[:, 2], "x4": X_pid[:, 3]})
    rows += long_rows("simulation_full_schedule", "pid_discrete",
                      full_schedule_table(U_pid, T_pid_states, X_pid))
    write_sections(rows)
    print()

    # ---- Figures ------------------------------------------------------------
    print("Writing plots ...")
    with plt.rc_context(RC_RESULTS):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(T_pid_states, X_pid[:, 0], label="Glial cells ($x_1$)")
        ax.plot(T_pid_states, X_pid[:, 1], label="Glioma cells ($x_2$)")
        ax.plot(T_pid_states, X_pid[:, 2], label="Neurons ($x_3$)")
        ax.plot(times_ref, x2_ref_pid, linestyle='--', color='black', linewidth=1.8,
                label="Tumor reference")
        finish(ax, "Closed-loop cellular dynamics under PID control", "Time [days]", CELLS)
        save(fig, "pid_simulation_x1x2x3")

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(T_pid_states, X_pid[:, 3], label="Drug concentration ($x_4$)")
        finish(ax, "Tissue-level chemotherapy concentration under PID control",
               "Time [days]", "Drug concentration")
        save(fig, "pid_simulation_x4")

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(T_pid_control, U_pid, label="Infusion rate $u(t)$")
        ax.axhline(U_MIN, linestyle='--', linewidth=1.2, alpha=0.6)
        ax.axhline(U_MAX, linestyle='--', linewidth=1.2, alpha=0.6)
        finish(ax, "Chemotherapy infusion profile under PID control", "Time [days]", UNIT_U)
        save(fig, "pid_simulation_u")

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(times_ref, Kp_prof, label="Proportional gain $K_p$")
        ax.plot(times_ref, Ki_prof, label="Integral gain $K_i$")
        ax.plot(times_ref, Kd_prof, label="Derivative gain $K_d$")
        finish(ax, "Time-varying PID gains (S-curve scheduling)", "Time [days]",
               "Effective gains")
        save(fig, "pid_simulation_gains")

    with plt.rc_context(RC_SCHEDULE):
        plot_schedule(pid_rows, '[PID] Drug pulse durations', "pid_simulation_drug_schedule")

    for name in PLOTS:
        print(f"  wrote {os.path.join(PLOT_DIR, FIGURE_FILES[name] + '.png')}")

if __name__ == "__main__":
    main()
