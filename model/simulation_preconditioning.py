"""simulation_preconditioning.py

Pre-conditioning: 600 days without treatment from y0 = [1, 1e-3, 1, 0] (state at treatment start).

Run:  python model/simulation_preconditioning.py

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

# --- Output -------------------------------------------------------------------
SHOW_FIGURES = False            # True -> also display each figure (plt.show); False -> headless, files only
SAVE_DPI = 600
# Paths are relative to the repository root (REPO). Outputs: one csv, results/runs/simulation_preconditioning.csv, and the
# diagnostic plots of this script in results/diagnostic_plots (file names in FIGURE_FILES).
ROOT = "."                             # base for the printed paths
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # repository root
RESULTS_DIR = os.path.join(REPO, "results", "runs")
PLOT_DIR = os.path.join(REPO, "results", "diagnostic_plots")
RESULTS_FILE = os.path.join(RESULTS_DIR, "simulation_preconditioning.csv")
FIGURE_FILES = {
    "general_preconditioning_states": "General_Fig2_preconditioning_states",
}
SECTION_PREFIX = "preconditioning_"   # this script writes (and replaces) only these sections
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
# 2) SIMULATION
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

# =============================================================================
# 3) RESULTS FILE (tidy CSV, read-modify-write)
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

# =============================================================================
# 4) FIGURE STYLE
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


CELLS = "Normalized cell population density"

# =============================================================================
# 5) MAIN
# =============================================================================
def main():
    if not SHOW_FIGURES:
        plt.switch_backend("Agg")
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    T_pre, X_pre = simulate_preconditioning()
    x_burn = X_pre[-1].copy()

    # ---- Console ------------------------------------------------------------
    print("=" * 72)
    print("simulation_preconditioning - console output for FACTS.md")
    print("=" * 72)
    print(f"[Pre-conditioning, t = {T_START:.0f} d]")
    for i in range(4):
        print(f"  x{i+1}(T_start) = {x_burn[i]:.6f}")
    print()

    # ---- Results file -------------------------------------------------------
    print("Writing csv ...")
    rows = long_rows("preconditioning_state_T_start", "nominal",
                     {f"x{i+1}": x_burn[i] for i in range(4)})
    rows += long_rows("preconditioning_timeseries", "nominal", {
        "t_day": T_pre, "x1": X_pre[:, 0], "x2": X_pre[:, 1],
        "x3": X_pre[:, 2], "x4": X_pre[:, 3]})
    write_sections(rows)
    print()

    # ---- Figures ------------------------------------------------------------
    print("Writing plots ...")
    with plt.rc_context(RC_PRECONDITIONING):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(T_pre, X_pre[:, 0], label=r"Glial cells ($x_1$)")
        ax.plot(T_pre, X_pre[:, 1], label=r"Glioma cells ($x_2$)")
        ax.plot(T_pre, X_pre[:, 2], label=r"Neurons ($x_3$)")
        finish(ax, "Pre-conditioning dynamics prior to feedback activation",
               "Time [days]", CELLS)
        save(fig, "general_preconditioning_states")
    print(f"  wrote {os.path.join(PLOT_DIR, FIGURE_FILES['general_preconditioning_states'] + '.png')}")

if __name__ == "__main__":
    main()
