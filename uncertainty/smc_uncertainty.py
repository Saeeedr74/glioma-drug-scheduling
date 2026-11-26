"""smc_uncertainty.py

Protocol E1: the nominal SMC schedule replayed on the five parameter groups G1..G5. Needs results/runs/smc_simulation.csv.

Run:  python uncertainty/smc_uncertainty.py

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

# --- Thresholds ---------------------------------------------------------------
X2_THRESHOLD = 1e-4     # tumor suppression threshold reported in the paper
X1_RECOVERY = 0.99      # glial recovery threshold reported in the paper

# --- SMC, Table 4 -------------------------------------------------------------
SMC_A_REF = 7.5         # reference shaping a; unified with PID to 7.5 (retune 2026-11-14; was 8.5)
SMC_B_REF = 0.0         # reference shaping b

# --- Uncertainty groups -------------------------------------------------------
OMEGA_DEV = 0.20    # Omega_1 down, Omega_2 up
BETA_DEV = 0.50     # beta_1 up, beta_2 down
P_DEV = 0.15        # p_1 up, p_2 down
A_DEV = 0.20        # a_1 up, a_2 up
ZETA_DEV = 0.20     # zeta up

# (key, Table 6 row label, Table 6 deviation, {GLIOMA_2 constant: multiplier})
GROUPS = [
    ("nominal", "Nominal baseline", "---", {}),
    ("Omega", r"$\Omega_1 \downarrow,\ \Omega_2 \uparrow$", r"$\pm 20\%$",
     {"OMEGA_1": 1 - OMEGA_DEV, "OMEGA_2": 1 + OMEGA_DEV}),
    ("beta", r"$\beta_1 \uparrow,\ \beta_2 \downarrow$", r"$\pm 50\%$",
     {"BETA_1": 1 + BETA_DEV, "BETA_2": 1 - BETA_DEV}),
    ("p", r"$p_1 \uparrow,\ p_2 \downarrow$", r"$\pm 15\%$",
     {"P_1": 1 + P_DEV, "P_2": 1 - P_DEV}),
    ("a", r"$a_1 \uparrow,\ a_2 \uparrow$", r"$\pm 20\%$",
     {"A_1": 1 + A_DEV, "A_2": 1 + A_DEV}),
    ("zeta", r"$\zeta \uparrow$", r"$+20\%$", {"ZETA": 1 + ZETA_DEV}),
]
CONTROLLER = "SMC"
FIGURE_GROUP = "Omega"          # group drawn against nominal in the figures

N_DAYS = int(T_REF / DT)    # 2000 daily doses -> states at t = 0 .. 2000
FIGSIZE = (9, 4.2)

# --- Output -------------------------------------------------------------------
SHOW_FIGURES = False            # True -> also display each figure (plt.show); False -> headless, files only
SAVE_DPI = 600
# Paths are relative to the repository root (REPO). Outputs: one csv, results/runs/smc_uncertainty.csv, and the
# diagnostic plots of this script in results/diagnostic_plots (file names in FIGURE_FILES).
ROOT = "."                             # base for the printed paths
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # repository root
RESULTS_DIR = os.path.join(REPO, "results", "runs")
PLOT_DIR = os.path.join(REPO, "results", "diagnostic_plots")
RESULTS_FILE = os.path.join(RESULTS_DIR, "smc_uncertainty.csv")
SCHEDULE_FILE = os.path.join(RESULTS_DIR, "smc_simulation.csv")   # read only: nominal schedule of smc_simulation
FIGURE_FILES = {
    "smc_uncertainty_x2_trajectories": "SMC_Fig16_uncertainty_x2",
}
SECTION_PREFIX = "uncertainty_"   # this script writes (and replaces) only these sections
SECTION_ORDER = ("simulation_", "observer_", "uncertainty_")

# ---- the tuned / design parameters of this notebook (identical in the simulation,
#      observer and uncertainty notebooks of the same controller) --------------------
for _k in ('SMC_A_REF', 'SMC_B_REF', 'LAMBDA_P', 'LAMBDA_D', 'LAMBDA_I', 'GAMMA_1', 'GAMMA_2', 'DELTA_INIT', 'DELTA_FINAL', 'A_DELTA', 'B_DELTA'):
    if _k in globals():
        print(f"{_k:<12} = {globals()[_k]!r}")

import contextlib
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
# 2) S-CURVES AND PRE-CONDITIONING
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
# 3) PERTURBED PLANT AND OPEN-LOOP REPLAY
# =============================================================================
@contextlib.contextmanager
def perturbed_plant(changes):
    """Temporarily multiply this file's model constants (module globals), so that
    plant_nominal - and everything built on it - is the perturbed plant. The
    constants are restored on exit, also after an error."""
    g = globals()
    saved = {name: g[name] for name in changes}
    try:
        for name, factor in changes.items():
            g[name] = saved[name] * factor
        yield
    finally:
        for name, value in saved.items():
            g[name] = value


def simulate_schedule(x_init, u_sched):
    """Open-loop replay: u_sched[k] is applied on [k, k+1). Same RK4 step and the
    same x2 >= 0 clip as the closed-loop simulators in GLIOMA_2.py."""
    x = x_init.copy()
    X = [x.copy()]
    for u in u_sched:
        x = rk4_step(plant_nominal, x, u, DT)
        x[1] = max(x[1], 0.0)
        X.append(x.copy())
    return np.arange(len(X)) * DT, np.vstack(X)

# =============================================================================
# 4) INPUT SCHEDULE (from the results file)
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


def read_results(path=None):
    """A whole results file (default RESULTS_FILE; an empty table if it does not exist yet)."""
    path = RESULTS_FILE if path is None else path
    if not os.path.exists(path):
        return pd.DataFrame(columns=CSV_COLUMNS)
    df = pd.read_csv(path,
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
    """One table back from SCHEDULE_FILE (smc_simulation.csv): column name -> numpy array (by row)."""
    df = read_results(SCHEDULE_FILE)
    sub = df[(df["section"] == section) & (df["run"] == run)]
    if sub.empty:
        raise FileNotFoundError(f"{section} / {run} not found in {SCHEDULE_FILE} - "
                                "run the simulation script first.")
    print(f"  read {os.path.relpath(SCHEDULE_FILE, ROOT)}: {section} / {run}")
    return {name: g.sort_values("row")["value"].to_numpy(float)
            for name, g in sub.groupby("name", sort=False)}


def day_or_nan(T, i):
    return np.nan if i is None else T[i]


def load_schedule():
    """The SMC schedule (2000 daily doses), the one-day-late SMC schedule of the
    daily schedule table (check only) and the nominal closed-loop x2 of
    smc_simulation's run (self-check of the replay), read from the simulation_*
    sections of the results file."""
    ts = read_table("simulation_timeseries", "smc_discrete")
    full = read_table("simulation_full_schedule", "smc_discrete")
    return ts["u"][:N_DAYS], full["Dose"][:N_DAYS], ts["x2"]

# =============================================================================
# 5) METRICS
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


def line(label, value):
    print(f"  {label:<46}= {value}")


def metrics(T, X):
    c2, c1 = X[:, 1] <= X2_THRESHOLD, X[:, 0] >= X1_RECOVERY

    def day(i):
        return np.nan if i is None else T[i]

    return {"x2_T": X[-1, 1], "x3_T": X[-1, 2],
            "t_x2_1e4_first": day(first_index(c2)),
            "t_x2_1e4_perm": day(permanent_index(c2)),
            "t_x1_099_first": day(first_index(c1)),
            "t_x1_099_perm": day(permanent_index(c1))}


def add_ratios(rows):
    """x2_uncertain / x2_nominal within each (method, controller)."""
    for r in rows:
        nom = next(q for q in rows if q["method"] == r["method"]
                   and q["controller"] == r["controller"] and q["group"] == "nominal")
        r["sensitivity"] = r["x2_T"] / nom["x2_T"]


def mean_sensitivity(rows, method, ctrl):
    return np.mean([r["sensitivity"] for r in rows if r["method"] == method
                    and r["controller"] == ctrl and r["group"] != "nominal"])


def sci_tex(v):
    m, e = f"{v:.2e}".split("e")
    return f"${m}{{\\times}}10^{{{int(e)}}}$"


def _d(v):
    return "not reached" if np.isnan(v) else f"{v:.0f}"


def print_controller_block(rows, ctrl):
    print(f"[{ctrl}, open-loop replay, nominal pre-conditioning, perturbed dosing]")
    print(f"  {'group':>8} {'x2(T_start)':>11} {'x2(T=2000)':>11} {'x2/x2_nom':>9} "
          f"{'t1e-4 1st':>11} {'t1e-4 prm':>11} {'rec 1st':>11} {'rec prm':>11} {'x3(T)':>8}")
    for r in rows:
        if r["method"] == "main" and r["controller"] == ctrl:
            print(f"  {r['group']:>8} {r['x2_Tstart']:11.6f} {r['x2_T']:11.3e} "
                  f"{r['sensitivity']:9.1f} {_d(r['t_x2_1e4_first']):>11} "
                  f"{_d(r['t_x2_1e4_perm']):>11} {_d(r['t_x1_099_first']):>11} "
                  f"{_d(r['t_x1_099_perm']):>11} {r['x3_T']:8.4f}")
    line("mean sensitivity factor (5 groups)", f"{mean_sensitivity(rows, 'main', ctrl):.1f}")
    print()


def print_check_block(rows, method, title):
    print(title)
    for ctrl in sorted({r["controller"] for r in rows if r["method"] == method}):
        vals = "  ".join(f"{r['group']}={r['x2_T']:.3e}" for r in rows
                         if r["method"] == method and r["controller"] == ctrl)
        print(f"  {ctrl}: {vals}")
        line(f"{ctrl} mean sensitivity factor", f"{mean_sensitivity(rows, method, ctrl):.1f}")
    print()


def print_latex(rows):
    get = {r["group"]: r["x2_T"] for r in rows if r["method"] == "main"}
    print(f"LaTeX (Table 6, {CONTROLLER} column):")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(rf"Uncertainty group & Deviation & {CONTROLLER} \\")
    print(r"\midrule")
    for key, label, dev, _ in GROUPS:
        print(f"{label} & {dev} & {sci_tex(get[key])} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print()

# =============================================================================
# 6) FIGURES
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
RC_UNCERTAINTY = {**RC_PRECONDITIONING, "lines.linewidth": 1.8}


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


def plot_uncertainty(ctrl, T, x2_nom, x2_unc, times_ref, x2_ref):
    with plt.rc_context(RC_UNCERTAINTY):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(T, x2_nom, label=f"{ctrl} – nominal")
        ax.plot(T, x2_unc, linestyle="--",
                label=rf"{ctrl} – uncertain ($\Omega_1\downarrow,\Omega_2\uparrow$ "
                      rf"{int(round(OMEGA_DEV*100))}%)")
        ax.plot(times_ref, x2_ref, linestyle=":", color="black", linewidth=1.2,
                label="Reference")
        ax.set_xlabel("Time [days]")
        ax.set_ylabel(r"Tumor population $x_2(t)$")
        ax.legend(frameon=False, loc="best")
        ax.margins(x=0)
        fig.tight_layout()
        save(fig, f"{ctrl.lower()}_uncertainty_x2_trajectories")

# =============================================================================
# 7) MAIN
# =============================================================================
def main():
    if not SHOW_FIGURES:
        plt.switch_backend("Agg")
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("Reading the nominal schedule ...")
    u_sched, u_smc_late, x2_closed = load_schedule()
    print()

    # Pre-conditioning runs ONCE, on the NOMINAL plant (silent; the run itself is
    # General/simulation_preconditioning). Every group - nominal and the five
    # uncertainty groups - starts its dosing phase from this same state at T_start.
    _, X_pre_nom = simulate_preconditioning()
    x_start_nom = X_pre_nom[-1].copy()
    times_ref = np.arange(N_DAYS + 1) * DT
    x2_ref = tumor_reference(x_start_nom[1], times_ref, SMC_A_REF, SMC_B_REF)[0]

    nominal_constants = {name: globals()[name] for _, _, _, ch in GROUPS for name in ch}
    x_start = x_start_nom
    rows, traj = [], {"t_day": times_ref}
    ctrl = CONTROLLER
    for key, _, _, changes in GROUPS:
        runs = [("main", ctrl, x_start, u_sched)]
        # Check: nominal pre-conditioning (the old notebook's starting state).
        runs += [("check_nominal_precond", ctrl, x_start_nom, u_sched)]
        runs.append(("check_nominal_precond_smc_one_day_late", "SMC", x_start_nom,
                     u_smc_late))
        # Only the dosing phase (open-loop replay of the schedule) is perturbed.
        with perturbed_plant(changes):
            for method, c, x0, u in runs:
                T, X = simulate_schedule(x0, u)
                rows.append({"method": method, "controller": c, "group": key,
                             "x1_Tstart": x0[0], "x2_Tstart": x0[1], "x3_Tstart": x0[2],
                             **metrics(T, X)})
                if method == "main":
                    traj[f"{c}_{key}_x2"] = X[:, 1]
    assert all(globals()[n] == v for n, v in nominal_constants.items())   # restored
    add_ratios(rows)

    # ---- Console ------------------------------------------------------------
    print("=" * 72)
    print("smc_uncertainty - console output for FACTS.md")
    print("=" * 72)
    print("[Self-check: nominal replay vs smc_simulation closed-loop x2]")
    n = N_DAYS + 1
    line(f"{ctrl}: max |x2_replay - x2_closed_loop|",
         f"{np.max(np.abs(traj[f'{ctrl}_nominal_x2'] - x2_closed[:n])):.3e}")
    print()
    print("[Pre-conditioned state at T_start, per group (nominal pre-conditioning)]")
    for r in rows:
        if r["method"] == "main":
            print(f"  {r['group']:>8}: x1 = {r['x1_Tstart']:.6f}, x2 = {r['x2_Tstart']:.6f}, "
                  f"x3 = {r['x3_Tstart']:.6f}")
    print()
    print_controller_block(rows, ctrl)
    print_check_block(rows, "check_nominal_precond",
                      "[Check: nominal pre-conditioning (start state of the old notebook)]")
    print_check_block(rows, "check_nominal_precond_smc_one_day_late",
                      "[Check: nominal pre-conditioning, SMC schedule one day late "
                      "(simulation_full_schedule)]")
    print_latex(rows)

    # ---- Results file -------------------------------------------------------
    print("Writing csv ...")
    cols = ["method", "controller", "group", "x1_Tstart", "x2_Tstart", "x3_Tstart", "x2_T",
            "sensitivity", "t_x2_1e4_first", "t_x2_1e4_perm", "t_x1_099_first",
            "t_x1_099_perm", "x3_T"]
    out = long_rows("uncertainty_table6", "smc", {c: [r[c] for r in rows] for c in cols})
    methods = sorted({r["method"] for r in rows})
    out += long_rows("uncertainty_mean_sensitivity", "smc",
                     {m: mean_sensitivity(rows, m, ctrl) for m in methods})
    out += long_rows("uncertainty_x2_trajectories", "smc", traj)
    write_sections(out)
    print()

    # ---- Figures ------------------------------------------------------------
    print("Writing plots ...")
    plot_uncertainty(ctrl, times_ref, traj[f"{ctrl}_nominal_x2"],
                     traj[f"{ctrl}_{FIGURE_GROUP}_x2"], times_ref, x2_ref)
    print(f"  wrote {os.path.join(PLOT_DIR, FIGURE_FILES['smc_uncertainty_x2_trajectories'] + '.png')}")

if __name__ == "__main__":
    main()
