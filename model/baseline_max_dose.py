"""baseline_max_dose.py

Maximum-dose baseline: u = 200 until the first day with x2 < 1e-4 (the suppression threshold of the
paper); the tumor is then taken as eliminated (x2 := 0) and u = 0 until the end of the horizon.

Run:  python model/baseline_max_dose.py

Part of the code for: S. Rahimi, H. Moradi, "Robust drug scheduling for glioma chemotherapy based on a
multivariable dynamic model and state estimation via an extended Kalman filter" (Sharif University of Technology). MIT License, see LICENSE.
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
U_MAX = 200.0       # high-dose-phase dose   [mg m^-2 day^-1]

# --- Thresholds ---------------------------------------------------------------
X2_THRESHOLD = 1e-4     # tumor suppression threshold reported in the paper
X1_RECOVERY = 0.99      # glial recovery threshold reported in the paper
X2_STOP = X2_THRESHOLD  # switch-off point of the maximum-dose baseline: the first day with
                        # x2 < X2_STOP (the suppression threshold itself).

# --- Baseline figures ---------------------------------------------------------
BASELINE_PLOT_WINDOW = 700.0    # the baseline figures show days 0..700

# --- Output -------------------------------------------------------------------
SHOW_FIGURES = False            # True -> also display each figure (plt.show); False -> headless, files only
SAVE_DPI = 600
# Paths are relative to the repository root (REPO). Outputs: one csv, results/runs/baseline_max_dose.csv, and the
# diagnostic plots of this script in results/diagnostic_plots (file names in FIGURE_FILES).
ROOT = "."                             # base for the printed paths
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # repository root
RESULTS_DIR = os.path.join(REPO, "results", "runs")
PLOT_DIR = os.path.join(REPO, "results", "diagnostic_plots")
RESULTS_FILE = os.path.join(RESULTS_DIR, "baseline_max_dose.csv")
FIGURE_FILES = {
    "general_baseline_x1x2x3": "General_Fig3_baseline_x1x2x3",
    "general_baseline_x4": "General_Fig3_baseline_x4",
    "general_baseline_u": "General_Fig3_baseline_u",
}
SECTION_PREFIX = "baseline_"   # this script writes (and replaces) only these sections
SECTION_ORDER = ("preconditioning_", "baseline_")

# ---- the tuned / design parameters of this notebook (identical in the simulation,
#      observer and uncertainty notebooks of the same controller) --------------------
for _k in ('T_START', 'T_REF', 'DT', 'U_MAX'):
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
# 2) SIMULATIONS
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


def simulate_max_dose(x_init):
    """Baseline: u = U_MAX until the first day with x2 < X2_STOP, then tumor set to zero and u = 0.

    U_list[k] is the dose applied on [k, k+1).
    """
    x = x_init.copy()
    t = 0.0
    X_list = [x.copy()]
    U_list = []
    T_list = [0.0]
    full_on = True
    while t < T_REF:
        if full_on:
            u = U_MAX
            x = rk4_step(plant_nominal, x, u, DT)
            x[1] = max(x[1], 0.0)
            if x[1] < X2_STOP:
                full_on = False
                x[1] = 0.0
        else:
            u = 0.0
            x = rk4_step(plant_nominal, np.array([x[0], 0.0, x[2], x[3]]), u, DT)
            x[1] = 0.0
        U_list.append(u)
        t += DT
        X_list.append(x.copy())
        T_list.append(t)
    return np.vstack(X_list), np.array(U_list), np.array(T_list)

# =============================================================================
# 3) METRICS
# =============================================================================
def first_index(cond):
    idx = np.where(cond)[0]
    return int(idx[0]) if idx.size else None


def _fmt_day(T, i):
    return "not reached" if i is None else f"{T[i]:.1f}"


def line(label, value):
    print(f"  {label:<46}= {value}")

# =============================================================================
# 4) RESULTS FILE (tidy CSV, read-modify-write)
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

# =============================================================================
# 5) FIGURE STYLE
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

# =============================================================================
# 6) MAIN
# =============================================================================
def main():
    if not SHOW_FIGURES:
        plt.switch_backend("Agg")
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- x_start: nominal pre-conditioned state, computed silently ----------
    # (the pre-conditioning run itself is General/simulation_preconditioning)
    _, X_pre = simulate_preconditioning()
    x_burn = X_pre[-1].copy()

    X_fd, U_fd, T_fd = simulate_max_dose(x_burn.copy())

    # ---- Console ------------------------------------------------------------
    print("=" * 72)
    print("simulation_baseline - console output for FACTS.md")
    print("=" * 72)
    # Baseline. The manuscript's baseline metrics were computed on the 0-700 d
    # plotting window; the x3 it reports as x3(T = 2000) is the value at day 700.
    mask = T_fd <= BASELINE_PLOT_WINDOW
    T_w, X_w = T_fd[mask], X_fd[mask]
    off = first_index(U_fd == 0.0)
    print("[Continuous maximum dose]")
    print(f"  u = 200 applied on days [0, {off})  (switch-off on the first day with x2 < {X2_STOP:g})")
    line("time to x2 <= 1e-4, first crossing [d]",
         _fmt_day(T_w, first_index(X_w[:, 1] <= X2_THRESHOLD)))
    line("glial recovery x1 >= 0.99, first [d]",
         _fmt_day(T_w, first_index(X_w[:, 0] >= X1_RECOVERY)))
    line(f"x3 at day {T_w[-1]:.0f} (the paper's 'x3(T)')", f"{X_w[-1, 2]:.6f}")
    i2000 = int(np.argmin(np.abs(T_fd - T_REF)))
    line(f"x3(T = {T_fd[i2000]:.0f}), full horizon", f"{X_fd[i2000, 2]:.6f}")
    line("manuscript convention for the two times", "first")
    print()

    # ---- Results file -------------------------------------------------------
    print("Writing csv ...")
    rows = long_rows("baseline_metrics", "max_dose", {
        "switch_off_day": off,
        "t_x2_1e4_first_day": day_or_nan(T_w, first_index(X_w[:, 1] <= X2_THRESHOLD)),
        "t_x1_099_first_day": day_or_nan(T_w, first_index(X_w[:, 0] >= X1_RECOVERY)),
        "window_end_day": T_w[-1], "x3_window_end": X_w[-1, 2],
        "T_day": T_fd[i2000], "x3_T": X_fd[i2000, 2]})
    # u on row t is the dose applied on [t, t + 1 day); empty on the last row.
    n = len(T_fd)
    rows += long_rows("baseline_timeseries", "max_dose", {
        "t_day": T_fd, "u": pad_nan(U_fd, n), "x1": X_fd[:, 0], "x2": X_fd[:, 1],
        "x3": X_fd[:, 2], "x4": X_fd[:, 3]})
    write_sections(rows)
    print()

    # ---- Figures ------------------------------------------------------------
    print("Writing plots ...")
    with plt.rc_context(RC_RESULTS):
        # Baseline, days 0..700
        mask_u = T_fd[:-1] <= BASELINE_PLOT_WINDOW
        T_u_w, U_w = T_fd[:-1][mask_u], U_fd[mask_u]
        title = "Baseline: Continuous Maximum-Dose Chemotherapy"

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(T_w, X_w[:, 0], lw=1.8, label="Glial cells ($x_1$)")
        ax.plot(T_w, X_w[:, 1], lw=1.8, label="Glioma cells ($x_2$)")
        ax.plot(T_w, X_w[:, 2], lw=1.8, label="Neurons ($x_3$)")
        finish(ax, title, "Time [days]", CELLS)
        save(fig, "general_baseline_x1x2x3")

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(T_w, X_w[:, 3], lw=1.8, label="Drug concentration ($x_4$)")
        finish(ax, title, "Time [days]", "Drug concentration")
        save(fig, "general_baseline_x4")

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(T_u_w, U_w, lw=1.8, label="Infusion rate $u(t)$")
        finish(ax, title, "Time [days]", UNIT_U)
        save(fig, "general_baseline_u")
    for name in FIGURE_FILES:
        print(f"  wrote {os.path.join(PLOT_DIR, FIGURE_FILES[name] + '.png')}")

if __name__ == "__main__":
    main()
