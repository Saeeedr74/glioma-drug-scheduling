"""observer.py - state-observer study for the glioma chemotherapy controllers (mu, PID, SMC).

  Step 1  schedules : load the three final dose schedules of this repository and check them
                      (completeness, values {50, 200}, nominal replay, published dose statistics).
  Step 2  model     : nominal model checks (x_pre, Jacobian, observability at x_pre, old pole rule).
  Step 3  runs      : every scenario (3 controllers x 6 plants x 2 initial errors x
                      measurement modes x observers), in parallel, fixed seeds.
  Step 4  outputs   : metrics_all.csv, summaries, comparison table, time series, figures.

Observers (all on x_obs = [x1, x2], measurement y = x2, nominal model only):
  lsd  : corrected adaptive Luenberger observer, sampled-data form (comparison baseline in the paper).
  lzoh : the same observer in the continuous-time form with y held over each day (comparison, M1).
  ekf  : discrete extended Kalman filter, same model, same sampling (main observer in the paper).

Needs results/runs/pid_simulation.csv and smc_simulation.csv (run the controller scripts first)
and controllers/mu_synthesis/u_schedule_mu.csv.
Usage:  python observer/observer.py            (all steps, about 5 min on 11 worker processes)
        python observer/observer.py --step1    (step 1 only)

Part of the code for: S. Rahimi, H. Moradi, "Robust drug scheduling for glioma chemotherapy based on a
multivariable dynamic model" (Sharif University of Technology). MIT License, see LICENSE.
"""
import gzip
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm

# =============================================================================
# SETTINGS (every value is explained in README.md)
# =============================================================================
DELTA = 0.2            # initial relative error of x1_hat: x1_hat(0) = (1 +/- DELTA) * x1(0)
SIGMA = 0.1            # multiplicative measurement noise, y = x2 * (1 + SIGMA * n), n ~ N(0, 1)
TM_LIST = (7, 28)      # sampling periods of the sparse-imaging mode M3 [day]
N_SEEDS = 20           # noise realisations per noisy scenario
SEED_BASE = 20261115   # noise of seed s: numpy SeedSequence([SEED_BASE, s]); shared by all runs
CONV_THR = 0.01        # convergence: first day after which e1 stays below CONV_THR
TRIGGER = 0.005        # relinearisation trigger: relative change 0.5 % (as in the paper)
K_OBS = 2.0            # pole rule: p_i = -max(K_OBS * |Re(lambda_i(A))|, P_MIN)
P_MIN = 0.01           # [1/day]; slowest admissible observer pole (time constant 100 d)
POLE_SEP = 1.2         # if the two poles are closer than 20 %, the faster one is set to 1.2 x slower
FD_REL = 1e-6          # relative central-difference step of the linearisation in x1, x2
FD_X4_REL = 1e-2       # relative step in x4 (f is affine in x4: exact up to rounding)
EKF_Q_REL = 1e-3       # EKF process noise: Q = diag((q x1_hat)^2, (q x2_hat)^2) per day
EKF_R_REL_M1 = 1e-3    # EKF measurement noise used in the noise-free mode M1 (0.1 %)
N_SEEDS_TIMESERIES = 5 # seeds whose full time series are written for M2/M3 (all seeds in metrics)
N_WORKERS = max(1, min(12, (os.cpu_count() or 2) - 1))

# =============================================================================
# FIXED PROBLEM DATA
# =============================================================================
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCHED_DIR = HERE / "schedules"
RES_DIR = HERE / "results"
TS_DIR = RES_DIR / "timeseries"
FIG_DIR = RES_DIR / "figures"

DT = 1.0
T_PRE = 600
N_DAYS = 2000                       # states on days 0..2000, doses on days 0..1999
Y0_PRE = np.array([1.0, 1e-3, 1.0, 0.0])
X_PRE_EXPECTED = np.array([0.670934, 0.313731, 0.517817, 0.0])
A_REF, B_REF, T_REF = 7.5, 0.0, 2000.0
U_LEVELS = (50.0, 200.0)
X2_FLOOR = 1e-12

NOMINAL = dict(Om1=0.0068, Om2=0.012, b1=1.8e-2, b2=1.8e-3, p1=4.7e-8, p2=4.7e-5, p3=4.7e-8,
               a1=1.0, a2=1.0, a3=1.0, alpha=2.0, zeta=0.2)
GROUPS = {"nominal": {},
          "G1": {"Om1": -0.20, "Om2": 0.20},
          "G2": {"b1": 0.50, "b2": -0.50},
          "G3": {"p1": 0.15, "p2": -0.15},
          "G4": {"a1": 0.20, "a2": 0.20},
          "G5": {"zeta": 0.20}}
GROUP_LABEL = {"nominal": "Nominal", "G1": r"G1: $\Omega_1$ $-$20%, $\Omega_2$ +20%",
               "G2": r"G2: $\beta_1$ +50%, $\beta_2$ $-$50%", "G3": r"G3: $p_1$ +15%, $p_2$ $-$15%",
               "G4": r"G4: $a_1$, $a_2$ +20%", "G5": r"G5: $\zeta$ +20%"}
CTRLS = ("mu", "pid", "smc")
CTRL_LABEL = {"mu": r"$\mu$-synthesis", "pid": "PID", "smc": "SMC"}
MODES = [("M1", 1, False)] + [("M2", 1, True)] + [(f"M3_{tm}", tm, True) for tm in TM_LIST]
C_OBS = np.array([0.0, 1.0])

SRC = {
    "mu": dict(csv=ROOT / "controllers" / "mu_synthesis" / "u_schedule_mu.csv"),
    "pid": dict(csv=ROOT / "results" / "runs" / "pid_simulation.csv"),
    "smc": dict(csv=ROOT / "results" / "runs" / "smc_simulation.csv"),
}
# Values the final PID/SMC code printed (cumulative dose: printed by the v2 tuning run,
# sum of the 2000 daily doses).
PRINTED = {"pid": dict(first_x2=1166, cum_dose=131350, n_hi=71, hi_max=4),
           "smc": dict(first_x2=1165, cum_dose=140050, n_hi=71, hi_max=4)}
MU_EXPECTED = dict(first_x2=1167, cum_dose=142150, n_hi=65, hi_max=5)


# =============================================================================
# I/O HELPERS (cloud-synced folders: write in one go, retry on locking errors)
# =============================================================================
class Tee:
    def __init__(self):
        self.buf = io.StringIO()

    def write(self, s):
        sys.__stdout__.write(s)
        self.buf.write(s)

    def flush(self):
        sys.__stdout__.flush()


def write_bytes(path, data, tries=20):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for i in range(tries):
        try:
            tmp = path.with_name(path.name + ".tmp")
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
            return
        except OSError as exc:
            print(f"  [retry {i + 1}/{tries}] writing {path.name}: {exc}")
            time.sleep(1.0 + i)
    raise OSError(f"could not write {path}")


def write_text(path, text):
    write_bytes(path, text.encode("utf-8"))


def write_csv(path, df, gz=False, float_format=None):
    data = df.to_csv(index=False, float_format=float_format, lineterminator="\n").encode("utf-8")
    write_bytes(path, gzip.compress(data, compresslevel=6, mtime=0) if gz else data)


def copy_file(src, dst, tries=20):
    for i in range(tries):
        try:
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return
        except OSError as exc:
            print(f"  [retry {i + 1}/{tries}] copying {Path(src).name}: {exc}")
            time.sleep(1.0 + i)
    raise OSError(f"could not copy {src}")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mtime(path):
    return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")


def rel(path):
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


# =============================================================================
# MODEL
# =============================================================================
def params(group):
    p = dict(NOMINAL)
    for k, frac in GROUPS[group].items():
        p[k] = NOMINAL[k] * (1.0 + frac)
    return p


def f4(x, u, p):
    x1, x2, x3, x4 = x
    d1 = p["Om1"]*x1*(1-x1) - p["b1"]*x1*x2 - p["p1"]*x1*x4/(p["a1"]+x1)
    d2 = p["Om2"]*x2*(1-x2) - p["b2"]*x1*x2 - p["p2"]*x2*x4/(p["a2"]+x2)
    heav = 1.0 if d1 < 0 else (0.5 if d1 == 0 else 0.0)          # H(-dx1), H(0) = 0.5
    d3 = p["alpha"]*d1*heav*x3 - p["p3"]*x3*x4/(p["a3"]+x3)
    return np.array([d1, d2, d3, u - p["zeta"]*x4])


def rk4_plant(x, u, p):
    k1 = f4(x, u, p)
    k2 = f4(x + DT/2*k1, u, p)
    k3 = f4(x + DT/2*k2, u, p)
    k4 = f4(x + DT*k3, u, p)
    xn = x + DT/6*(k1 + 2*k2 + 2*k3 + k4)
    xn[1] = max(xn[1], 0.0)
    return xn


def precondition():
    x = Y0_PRE.copy()
    for _ in range(T_PRE):
        x = rk4_plant(x, 0.0, NOMINAL)
    return x


def simulate_plant(U, group, x_pre):
    p = params(group)
    X = np.zeros((N_DAYS + 1, 4))
    X[0] = x_pre
    for k in range(N_DAYS):
        X[k + 1] = rk4_plant(X[k], U[k], p)
    return X


def x4_known(U, x4_0=0.0):
    """Drug concentration computed by the observer from the applied dose, NOMINAL zeta."""
    z = NOMINAL["zeta"]
    out = np.zeros(N_DAYS + 1)
    out[0] = x4_0
    for k in range(N_DAYS):
        x, u = out[k], U[k]
        k1 = u - z*x
        k2 = u - z*(x + DT/2*k1)
        k3 = u - z*(x + DT/2*k2)
        k4 = u - z*(x + DT*k3)
        out[k + 1] = x + DT/6*(k1 + 2*k2 + 2*k3 + k4)
    return out


def f2(x1, x2, x4, p=NOMINAL):
    return np.array([p["Om1"]*x1*(1-x1) - p["b1"]*x1*x2 - p["p1"]*x1*x4/(p["a1"]+x1),
                     p["Om2"]*x2*(1-x2) - p["b2"]*x1*x2 - p["p2"]*x2*x4/(p["a2"]+x2)])


def jac2_analytic(x1, x2, x4, p=NOMINAL):
    A = np.array([[p["Om1"]*(1-2*x1) - p["b1"]*x2 - p["p1"]*x4*p["a1"]/(p["a1"]+x1)**2, -p["b1"]*x1],
                  [-p["b2"]*x2, p["Om2"]*(1-2*x2) - p["b2"]*x1 - p["p2"]*x4*p["a2"]/(p["a2"]+x2)**2]])
    B = np.array([-p["p1"]*x1/(p["a1"]+x1), -p["p2"]*x2/(p["a2"]+x2)])
    return A, B


def linearize(x1, x2, x4):
    """Central-difference linearisation of f(x1, x2; x4) at the current estimate.
    Returns A = df/d[x1, x2], B = df/dx4 (x4 is the known input) and f(x_e)."""
    A = np.zeros((2, 2))
    for i, xi in enumerate((x1, x2)):
        h = FD_REL*max(abs(xi), X2_FLOOR)
        xp = [x1, x2]; xm = [x1, x2]
        xp[i] += h; xm[i] -= h
        A[:, i] = (f2(xp[0], xp[1], x4) - f2(xm[0], xm[1], x4)) / (2*h)
    # f is affine in x4, so the x4 step only affects rounding; a larger step keeps the tiny
    # entry df1/dx4 = -p1 x1/(a1 + x1) ~ 2e-8 accurate.
    h = FD_X4_REL*max(abs(x4), 1.0)
    B = (f2(x1, x2, x4 + h) - f2(x1, x2, x4 - h)) / (2*h)
    return A, B, f2(x1, x2, x4)


def obs_matrix(A):
    return np.vstack([C_OBS, C_OBS @ A])


def pole_rule(A):
    """Always stable, distinct, at least K_OBS times faster than each plant mode and never
    slower than P_MIN. Returns the poles sorted from slow to fast and eig(A)."""
    lam = np.linalg.eigvals(A)
    p = np.sort(-np.maximum(K_OBS*np.abs(lam.real), P_MIN))[::-1]
    if p[0] - p[1] < (POLE_SEP - 1.0)*abs(p[0]):
        p[1] = POLE_SEP*p[0]
    return p, lam


def ackermann(G, c, roots):
    """Observer gain K with eig(G - K c) = roots (2 x 2, single output row c)."""
    O = np.vstack([c, c @ G])
    phi = G @ G - (roots[0] + roots[1])*G + roots[0]*roots[1]*np.eye(2)
    return phi @ np.linalg.solve(O, np.array([0.0, 1.0]))


# =============================================================================
# OBSERVERS (one run = one observer on one plant trajectory)
# =============================================================================
def measurements(x2_true, period, noise):
    """y on measurement days (NaN elsewhere). noise: None (M1) or the n-sequence of one seed."""
    y = np.full(N_DAYS + 1, np.nan)
    days = np.arange(0, N_DAYS + 1, period)
    y[days] = x2_true[days] * (1.0 + (SIGMA*noise[days] if noise is not None else 0.0))
    return y


class DesignLog:
    def __init__(self):
        self.n_updates = 0           # triggered relinearisations (initial design not counted)
        self.ranks = []
        self.rank_deficient = []     # (day, x1_hat, x2_hat)
        self.max_re_cont = -np.inf   # max Re(eig(A - L C)) over all designs
        self.max_abs_disc = 0.0      # max |eig((I - K C) Phi)| over all corrections
        self.taus = []               # observer time constants -1/p of every design
        self.tau0 = (np.nan, np.nan)
        self.n_clip = 0

    def design(self, A, day):
        """Rank check, pole rule, continuous gain L and its verification. Returns (ok, p, L)."""
        r = int(np.linalg.matrix_rank(obs_matrix(A)))
        self.ranks.append(r)
        if r < 2:
            return False, None, None
        p, _ = pole_rule(A)
        L = ackermann_cont(A, p)
        ev = np.linalg.eigvals(A - np.outer(L, C_OBS))
        self.max_re_cont = max(self.max_re_cont, float(ev.real.max()))
        tau = -1.0/p
        self.taus.extend(tau.tolist())
        if day == 0:
            self.tau0 = (float(tau[0]), float(tau[1]))
        return True, p, L


def ackermann_cont(A, p):
    """Continuous Luenberger gain: eig(A - L C) = p, via Ackermann with O = [C; CA]."""
    O = obs_matrix(A)
    phi = A @ A - (p[0] + p[1])*A + p[0]*p[1]*np.eye(2)
    return phi @ np.linalg.solve(O, np.array([0.0, 1.0]))


def run_lsd(X, x4h, U, y, period, delta):
    """Sampled-data adaptive Luenberger observer (main form).
    Between samples: x_hat follows the frozen affine model
        dx_hat/dt = A (x_hat - x_e) + B (x4_hat - x4_e) + f(x_e)
    (x4_hat from the dose, nominal zeta). At a sample day j: x_hat += K (y_j - x2_hat), K placed on
    the transition matrix Phi of the linearised error dynamics since the previous sample (product of
    expm(A_seg * 1 d) over the days), with eig((I - K C) Phi) = exp(p_i * period), p_i from the pole
    rule. Relinearisation (A, B, f_e, eig, rank, poles) when x2_hat changed by >= TRIGGER."""
    log = DesignLog()
    xh = np.zeros((N_DAYS + 1, 2))
    xh[0] = [(1 + delta)*X[0, 0], y[0]]
    x = xh[0].copy()
    A, B, fe = linearize(x[0], x[1], x4h[0]); xe = x.copy(); x4e = x4h[0]
    ok, p, _ = log.design(A, 0)
    p_last = p if ok else np.array([-P_MIN, -POLE_SEP*P_MIN])
    Ad = expm(A*DT); Phi = np.eye(2); K = np.zeros(2); x2_last = x[1]
    for k in range(N_DAYS):
        u = U[k]
        z0 = np.array([x[0], x[1], x4h[k]])

        def rhs(z):
            d = A @ (z[:2] - xe) + B*(z[2] - x4e) + fe
            return np.array([d[0], d[1], u - NOMINAL["zeta"]*z[2]])
        k1 = rhs(z0); k2 = rhs(z0 + DT/2*k1); k3 = rhs(z0 + DT/2*k2); k4 = rhs(z0 + DT*k3)
        z = z0 + DT/6*(k1 + 2*k2 + 2*k3 + k4)
        x = z[:2].copy()
        Phi = Ad @ Phi
        j = k + 1
        if j % period == 0:                                    # measurement day: correction
            c = C_OBS @ Phi
            if np.linalg.matrix_rank(np.vstack([c, c @ Phi])) == 2:
                K = ackermann(Phi, c, np.exp(p_last*period*DT))
                ev = np.linalg.eigvals(Phi - np.outer(K, c))
                log.max_abs_disc = max(log.max_abs_disc, float(np.abs(ev).max()))
            else:
                log.rank_deficient.append((j, x[0], x[1]))
            x = x + K*(y[j] - x[1])
            Phi = np.eye(2)
        if x[1] < 0.0:
            x[1] = 0.0; log.n_clip += 1
        if not np.all(np.isfinite(x)) or abs(x[0]) > 1e3:
            xh[j:] = np.nan
            return xh, log, "diverged"
        xh[j] = x
        if abs(x[1] - x2_last) / max(abs(x2_last), 1e-300) >= TRIGGER:   # trigger on x2_hat
            x2_last = x[1]
            A, B, fe = linearize(x[0], x[1], x4h[j]); xe = x.copy(); x4e = x4h[j]
            Ad = expm(A*DT)
            log.n_updates += 1
            ok, p, _ = log.design(A, j)
            if ok:
                p_last = p
            else:
                log.rank_deficient.append((j, x[0], x[1]))
    return xh, log, "ok"


def run_lzoh(X, x4h, U, y, delta):
    """Continuous-time form of the spec (comparison, M1): y held over each day,
        dx_hat/dt = A (x_hat - x_e) + B (x4_hat - x4_e) + f(x_e) + L (y_k - x2_hat),
    RK4, relinearisation when y changed by >= TRIGGER (paper rule)."""
    log = DesignLog()
    xh = np.zeros((N_DAYS + 1, 2))
    xh[0] = [(1 + delta)*X[0, 0], y[0]]
    x = xh[0].copy()
    A, B, fe = linearize(x[0], x[1], x4h[0]); xe = x.copy(); x4e = x4h[0]
    ok, _, L = log.design(A, 0)
    if not ok:
        L = np.zeros(2)
    y_last = y[0]
    for k in range(N_DAYS):
        if k > 0 and abs(y[k] - y_last)/max(abs(y_last), 1e-300) >= TRIGGER:
            y_last = y[k]
            A, B, fe = linearize(x[0], x[1], x4h[k]); xe = x.copy(); x4e = x4h[k]
            log.n_updates += 1
            ok, _, Ln = log.design(A, k)
            if ok:
                L = Ln
            else:
                log.rank_deficient.append((k, x[0], x[1]))
        u, yk = U[k], y[k]
        z0 = np.array([x[0], x[1], x4h[k]])

        def rhs(z):
            d = A @ (z[:2] - xe) + B*(z[2] - x4e) + fe + L*(yk - z[1])
            return np.array([d[0], d[1], u - NOMINAL["zeta"]*z[2]])
        k1 = rhs(z0); k2 = rhs(z0 + DT/2*k1); k3 = rhs(z0 + DT/2*k2); k4 = rhs(z0 + DT*k3)
        x = (z0 + DT/6*(k1 + 2*k2 + 2*k3 + k4))[:2]
        if x[1] < 0.0:
            x[1] = 0.0; log.n_clip += 1
        if not np.all(np.isfinite(x)) or abs(x[0]) > 1e3:
            xh[k + 1:] = np.nan
            return xh, log, "diverged"
        xh[k + 1] = x
    return xh, log, "ok"


def run_ekf(X, x4h, U, y, period, delta, noisy):
    """Discrete EKF on [x1, x2]: daily prediction with the nonlinear nominal model (RK4, x4_hat
    from the dose) and its Jacobian (variational equation integrated with the same RK4);
    update only on measurement days. R = (sigma_eff * x2_prior)^2, Q = diag((q x_hat)^2) per day,
    P0 = diag((DELTA x1_hat(0))^2, (sigma_eff y0)^2)."""
    s_eff = SIGMA if noisy else EKF_R_REL_M1
    log = DesignLog()
    xh = np.zeros((N_DAYS + 1, 2)); sd = np.zeros((N_DAYS + 1, 2))
    x = np.array([(1 + delta)*X[0, 0], y[0]])
    P = np.diag([(DELTA*x[0])**2, (s_eff*y[0])**2])
    xh[0] = x; sd[0] = np.sqrt(np.diag(P))
    n_in2sd = 0
    zeta = NOMINAL["zeta"]
    for k in range(N_DAYS):
        u = U[k]
        s0 = np.concatenate([x, [x4h[k]], np.eye(2).ravel()])

        def rhs(s):
            Aj, _ = jac2_analytic(s[0], s[1], s[2])
            return np.concatenate([f2(s[0], s[1], s[2]), [u - zeta*s[2]],
                                   (Aj @ s[3:].reshape(2, 2)).ravel()])
        k1 = rhs(s0); k2 = rhs(s0 + DT/2*k1); k3 = rhs(s0 + DT/2*k2); k4 = rhs(s0 + DT*k3)
        s = s0 + DT/6*(k1 + 2*k2 + 2*k3 + k4)
        x = s[:2].copy(); F = s[3:].reshape(2, 2)
        Q = np.diag([(EKF_Q_REL*x[0])**2, (EKF_Q_REL*x[1])**2])*DT
        P = F @ P @ F.T + Q
        j = k + 1
        if x[1] < 0.0:
            x[1] = 0.0; log.n_clip += 1
        if j % period == 0:
            Aj, _ = jac2_analytic(x[0], x[1], x4h[j])
            r = int(np.linalg.matrix_rank(obs_matrix(Aj)))
            log.ranks.append(r)
            if r < 2:
                log.rank_deficient.append((j, x[0], x[1]))
            R = (s_eff*max(x[1], X2_FLOOR))**2
            S = P[1, 1] + R
            Kk = P[:, 1]/S
            x = x + Kk*(y[j] - x[1])
            IKH = np.eye(2) - np.outer(Kk, C_OBS)
            P = IKH @ P @ IKH.T + R*np.outer(Kk, Kk)
            log.n_updates += 1
            if x[1] < 0.0:
                x[1] = 0.0; log.n_clip += 1
        if not np.all(np.isfinite(x)) or abs(x[0]) > 1e3 or not np.all(np.isfinite(P)):
            xh[j:] = np.nan; sd[j:] = np.nan
            return xh, sd, log, "diverged"
        xh[j] = x; sd[j] = np.sqrt(np.maximum(np.diag(P), 0.0))
    return xh, sd, log, "ok"


def run_metrics(X, xh):
    x1, x2 = X[:, 0], X[:, 1]
    e1 = np.abs(xh[:, 0] - x1)/x1
    e2 = np.abs(xh[:, 1] - x2)/np.maximum(x2, X2_FLOOR)
    if not np.all(np.isfinite(e1)):
        return dict(e1_final=np.nan, e2_final=np.nan, rms_e1_0_2000=np.nan, rms_e1_100_2000=np.nan,
                    max_e1_100_2000=np.nan, conv_day=np.nan, rms_e2_0_2000=np.nan), e1, e2
    ok = e1 < CONV_THR
    bad = np.where(~ok)[0]
    conv = 0 if bad.size == 0 else (int(bad[-1]) + 1 if bad[-1] < N_DAYS else np.nan)
    return dict(e1_final=e1[N_DAYS], e2_final=e2[N_DAYS],
                rms_e1_0_2000=float(np.sqrt(np.mean(e1**2))),
                rms_e1_100_2000=float(np.sqrt(np.mean(e1[100:]**2))),
                max_e1_100_2000=float(e1[100:].max()), conv_day=conv,
                rms_e2_0_2000=float(np.sqrt(np.mean(e2**2)))), e1, e2


def noise_table():
    return np.array([np.random.default_rng(np.random.SeedSequence([SEED_BASE, s]))
                     .standard_normal(N_DAYS + 1) for s in range(N_SEEDS)])


def worker(task):
    """All runs of one (controller, plant, initial-error sign, observer)."""
    ctrl, plant, sign, obs, X, U, x4h = task
    delta = sign*DELTA
    noise = noise_table()
    out = []
    modes = [MODES[0]] if obs == "lzoh" else MODES
    for mode, period, noisy in modes:
        seeds = range(N_SEEDS) if noisy else [None]
        for s in seeds:
            y = measurements(X[:, 1], period, None if s is None else noise[s])
            sd = None
            if obs == "lsd":
                xh, log, status = run_lsd(X, x4h, U, y, period, delta)
            elif obs == "lzoh":
                xh, log, status = run_lzoh(X, x4h, U, y, delta)
            else:
                xh, sd, log, status = run_ekf(X, x4h, U, y, period, delta, noisy)
            m, e1, e2 = run_metrics(X, xh)
            row = dict(observer=obs, controller=ctrl, plant=plant, delta=delta, mode=mode,
                       period_day=period, sigma=SIGMA if noisy else 0.0,
                       seed=-1 if s is None else s, status=status, **m,
                       n_updates=log.n_updates,
                       rank_min=min(log.ranks) if log.ranks else np.nan,
                       n_rank_deficient=len(log.rank_deficient),
                       max_re_eig_A_LC=log.max_re_cont if obs != "ekf" else np.nan,
                       max_abs_eig_disc=log.max_abs_disc if obs == "lsd" else np.nan,
                       tau1_day0=log.tau0[0], tau2_day0=log.tau0[1],
                       tau_min=min(log.taus) if log.taus else np.nan,
                       tau_median=float(np.median(log.taus)) if log.taus else np.nan,
                       tau_max=max(log.taus) if log.taus else np.nan,
                       n_x2_clip=log.n_clip)
            if sd is not None and np.all(np.isfinite(sd)):
                err1 = np.abs(xh[:, 0] - X[:, 0])
                row["ekf_frac_in_2sd_100_2000"] = float(np.mean(err1[100:] <= 2*sd[100:, 0]))
                row["ekf_sd1_rel_final"] = float(sd[N_DAYS, 0]/abs(xh[N_DAYS, 0]))
            keep_ts = s is None or s < N_SEEDS_TIMESERIES
            out.append(dict(row=row, e1=e1.astype(np.float32),
                            xh=xh if keep_ts else None, sd=sd if keep_ts else None,
                            y=y if keep_ts else None,
                            rankdef=[(row["observer"], ctrl, plant, delta, mode, row["seed"], *r)
                                     for r in log.rank_deficient]))
    return out


# =============================================================================
# STEP 1 - SCHEDULES
# =============================================================================
def runs_of(b):
    d = np.diff(np.r_[0, b.astype(int), 0])
    return np.where(d == -1)[0] - np.where(d == 1)[0]


def dose_stats(U, x2):
    """Single convention for all controllers: doses on days 0..1999."""
    u = U[:N_DAYS]
    rl = runs_of(u >= 0.5*(U_LEVELS[0] + U_LEVELS[1]))
    i = np.where(x2 < 1e-4)[0]
    return dict(first_x2=int(i[0]) if i.size else None, cum_dose=float(np.sum(u)*DT),
                n_hi=int(rl.size), hi_max=int(rl.max()) if rl.size else 0,
                hi_mean=float(rl.mean()) if rl.size else np.nan, x2_2000=float(x2[N_DAYS]))


def reference(x2_0):
    t = np.arange(N_DAYS + 1)*DT
    S = lambda tau: 0.5*(1 - np.tanh(A_REF*(tau - B_REF)))
    return x2_0*(S(t/T_REF) - S(1.0))/(S(0.0) - S(1.0))


def relerr(a, b):
    return float(np.max(np.abs(a - b)/np.maximum(np.abs(b), X2_FLOOR)))


def read_long(path, ctrl):
    df = pd.read_csv(path, dtype={"section": str, "run": str, "name": str, "text": str})
    sub = df[(df.section == "simulation_timeseries") & (df.run == f"{ctrl}_discrete")]
    return {n: g.sort_values("row")["value"].to_numpy(float) for n, g in sub.groupby("name")}


def step1():
    """Load and check the three final dose schedules (the full provenance audit of the original
    working folders is not part of this repository)."""
    print("=" * 78 + "\nSTEP 1 - dose schedules\n" + "=" * 78)
    x_pre = precondition()
    assert np.allclose(x_pre, X_PRE_EXPECTED, atol=1e-6), x_pre
    schedules, summary = {}, {}
    for ctrl in CTRLS:
        s = SRC[ctrl]
        print(f"[{ctrl}] {rel(s['csv'])}")
        if ctrl == "mu":
            dm = pd.read_csv(s["csv"])
            t = dm.time_day.to_numpy(); U = dm.u_mg_per_m2_per_day.to_numpy(float)
            x2_full = dm.x2_tumor.to_numpy(float)
        else:
            tab = read_long(s["csv"], ctrl)
            valid = ~np.isnan(tab["u"])
            t = tab["t_day"][valid].astype(int); U = tab["u"][valid]
            x2_full = tab["x2"][:N_DAYS + 1]
        chk = {}
        exp_len = N_DAYS if ctrl == "smc" else N_DAYS + 1
        chk["a"] = len(U) == exp_len and t[0] == 0 and bool(np.all(np.diff(t) == 1)) and not np.any(np.isnan(U))
        chk["b"] = sorted(set(np.unique(U).tolist())) == list(U_LEVELS)
        Xr = simulate_plant(U, "nominal", x_pre)
        chk["d"] = relerr(Xr[:, 1], x2_full) < 1e-6
        st = dose_stats(U, Xr[:, 1])
        ref_vals = dict(MU_EXPECTED) if ctrl == "mu" else PRINTED[ctrl]
        chk["e"] = all(st[k] == ref_vals[k] for k in ref_vals)
        print(f"  checks: " + ", ".join(f"{k}={'pass' if v else 'FAIL'}" for k, v in chk.items()))
        print(f"  dose stats (days 0..1999): {st}")
        if not all(chk.values()):
            raise SystemExit(f"STOP: a schedule check failed for {ctrl}: {chk}")
        schedules[ctrl] = U[:N_DAYS].copy()
        summary[ctrl] = st
    return x_pre, schedules, summary


# =============================================================================
# STEP 2 - MODEL CHECKS
# =============================================================================
def step2(x_pre):
    print("=" * 78 + "\nSTEP 2 - model checks at x_pre\n" + "=" * 78)
    A, B, _ = linearize(x_pre[0], x_pre[1], x_pre[3])
    Aa, Ba = jac2_analytic(x_pre[0], x_pre[1], x_pre[3])
    jac_err = float(max(np.max(np.abs(A - Aa)/np.maximum(np.abs(Aa), 1e-30)),
                        np.max(np.abs(B - Ba)/np.maximum(np.abs(Ba), 1e-30))))
    assert jac_err < 1e-6, jac_err
    lam = np.linalg.eigvals(A)
    # 3-state observability with y = x2 (x4 known input)
    def f3(x):
        return f4(np.array([x[0], x[1], x[2], x_pre[3]]), 0.0, NOMINAL)[:3]
    A3 = np.zeros((3, 3))
    for i in range(3):
        h = FD_REL*max(abs(x_pre[i]), 1.0)
        xp = x_pre[:3].copy(); xm = x_pre[:3].copy(); xp[i] += h; xm[i] -= h
        A3[:, i] = (f3(xp) - f3(xm))/(2*h)
    C3 = np.array([0.0, 1.0, 0.0])
    O3 = np.vstack([C3, C3 @ A3, C3 @ A3 @ A3])
    rank3 = int(np.linalg.matrix_rank(O3))
    O2 = obs_matrix(A)
    p, _ = pole_rule(A)
    L = ackermann_cont(A, p)
    info = dict(eigA=lam, old_poles=2*lam, rank3=rank3, rank2=int(np.linalg.matrix_rank(O2)),
                cond2=float(np.linalg.cond(O2)), poles=p, tau=-1/p, L=L,
                eig_check=np.linalg.eigvals(A - np.outer(L, C_OBS)), jac_err=jac_err, A=A, A3=A3)
    print(f"  A (2-state) at x_pre, x4 = 0:\n{np.array2string(A, precision=6)}")
    print(f"  central-difference vs analytic Jacobian: max rel. error {jac_err:.1e}")
    print(f"  eig(A) = {np.array2string(lam, precision=6)}  (one positive eigenvalue)")
    print(f"  old pole rule 2*eig(A) = {np.array2string(2*lam, precision=6)}  -> UNSTABLE observer pole")
    print(f"  new pole rule p = {np.array2string(p, precision=6)}, time constants "
          f"{np.array2string(-1/p, precision=1)} d; eig(A - L C) = {np.array2string(info['eig_check'], precision=6)}")
    print(f"  rank [C; CA] (2 states) = {info['rank2']}, cond = {info['cond2']:.3e}")
    print(f"  rank of the 3-state observability matrix [C; CA; CA^2] with y = x2 = {rank3} (x3 unobservable)")
    return info


# =============================================================================
# STEP 3 - ALL RUNS
# =============================================================================
def step3(x_pre, schedules):
    print("=" * 78 + "\nSTEP 3 - observer runs\n" + "=" * 78)
    truths, x4hs, tasks = {}, {}, []
    for ctrl in CTRLS:
        U = schedules[ctrl]
        x4h = x4_known(U, x_pre[3])
        x4hs[ctrl] = x4h
        for plant in GROUPS:
            X = simulate_plant(U, plant, x_pre)
            truths[(ctrl, plant)] = X
            if plant != "G5":
                assert np.max(np.abs(X[:, 3] - x4h)) < 1e-9, "x4_hat must equal x4 when zeta is nominal"
            for sign in (+1, -1):
                for obs in ("lsd", "ekf", "lzoh"):
                    tasks.append((ctrl, plant, sign, obs, X, U, x4h))
    n_runs = sum(1 if t[3] == "lzoh" else 1 + N_SEEDS*(len(MODES) - 1) for t in tasks)
    print(f"  {len(tasks)} tasks, {n_runs} runs, {N_WORKERS} worker processes")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, res in enumerate(ex.map(worker, tasks, chunksize=1)):
            results.extend(res)
            if (i + 1) % 18 == 0 or i + 1 == len(tasks):
                print(f"  {i + 1}/{len(tasks)} tasks done, {time.time() - t0:.0f} s")
    print(f"  all runs finished in {time.time() - t0:.1f} s")
    return truths, x4hs, results


# =============================================================================
# STEP 4 - OUTPUTS
# =============================================================================
OBS_LABEL = {"lsd": "Luenberger (sampled-data)", "ekf": "EKF", "lzoh": "Luenberger (continuous, ZOH)"}
MODE_LABEL = {"M1": "M1: daily, noise-free", "M2": rf"M2: daily, $\sigma$ = {SIGMA:g}"}
MODE_LABEL.update({f"M3_{tm}": rf"M3: every {tm} d, $\sigma$ = {SIGMA:g}" for tm in TM_LIST})


def summarize(df):
    """Median and worst over seeds per scenario (noise-free runs: the single value)."""
    keys = ["observer", "controller", "plant", "delta", "mode"]
    worst_is_max = ["e1_final", "e2_final", "rms_e1_0_2000", "rms_e1_100_2000", "max_e1_100_2000",
                    "conv_day", "max_re_eig_A_LC", "max_abs_eig_disc", "n_rank_deficient"]
    rows = []
    for k, g in df.groupby(keys, sort=False):
        r = dict(zip(keys, k))
        r["n_runs"] = len(g); r["n_diverged"] = int((g.status != "ok").sum())
        r["n_not_converged"] = int(g.conv_day.isna().sum())
        for c in worst_is_max:
            v = g[c]
            r[f"{c}_median"] = float(v.median()) if v.notna().any() else np.nan
            if c == "conv_day":
                r[f"{c}_worst"] = np.nan if v.isna().any() else float(v.max())
            else:
                r[f"{c}_worst"] = float(v.max()) if v.notna().any() else np.nan
        r["n_updates_median"] = float(g.n_updates.median())
        r["rank_min"] = float(g.rank_min.min())
        for c in ("tau1_day0", "tau2_day0", "tau_min", "tau_max"):
            r[c] = float(g[c].median())
        if "ekf_frac_in_2sd_100_2000" in g:
            r["ekf_frac_in_2sd_median"] = float(g.ekf_frac_in_2sd_100_2000.median())
        rows.append(r)
    return pd.DataFrame(rows)


def fmt(v, kind="e"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n.c." if kind == "d" else "-"
    return f"{v:.0f}" if kind == "d" else f"{v:.2e}"


def comparison_table(summ):
    """Markdown/CSV table: both observers side by side, identical metrics."""
    lines = ["# Observer comparison (no winner selected)", "",
             f"Metrics: e1 = |x1_hat - x1|/x1. Final = day 2000. RMS over days 0..2000 and 100..2000. "
             f"Convergence day = first day after which e1 stays below {CONV_THR:g} (n.c. = not "
             f"converged by day 2000). Noisy modes: median / worst over {N_SEEDS} seeds. "
             f"delta = +{DELTA:g} and -{DELTA:g} are shown as two rows.", ""]
    csv_rows = []
    for ctrl in CTRLS:
        lines += [f"## {ctrl}", "",
                  "| plant | mode | delta | observer | e1(2000) med / worst | RMS e1 0..2000 med / worst "
                  "| RMS e1 100..2000 med / worst | conv. day med / worst | updates (med) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        sub = summ[summ.controller == ctrl]
        for plant in GROUPS:
            for mode, _, _ in MODES:
                for delta in (DELTA, -DELTA):
                    for obs in ("lsd", "ekf", "lzoh"):
                        r = sub[(sub.plant == plant) & (sub["mode"] == mode) & (sub.delta == delta)
                                & (sub.observer == obs)]
                        if r.empty:
                            continue
                        r = r.iloc[0]
                        noisy = mode != "M1"
                        two = lambda c, kind="e": (f"{fmt(r[c + '_median'], kind)} / {fmt(r[c + '_worst'], kind)}"
                                                   if noisy else fmt(r[c + '_median'], kind))
                        lines.append(f"| {plant} | {mode} | {delta:+g} | {obs} | {two('e1_final')} | "
                                     f"{two('rms_e1_0_2000')} | {two('rms_e1_100_2000')} | "
                                     f"{two('conv_day', 'd')} | {r.n_updates_median:.0f} |")
                        csv_rows.append(r)
        lines.append("")
    return "\n".join(lines) + "\n", pd.DataFrame(csv_rows)


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.8, "lines.linewidth": 1.1,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 100, "pdf.fonttype": 42})
    return plt


def save_fig(fig, name):
    for ext in ("png", "pdf"):
        b = io.BytesIO()
        fig.savefig(b, format=ext, dpi=300, bbox_inches="tight")
        write_bytes(FIG_DIR / f"{name}.{ext}", b.getvalue())


COL = {"true": "black", "lsd": "#1f5fa8", "ekf": "#d1691d", "lzoh": "#7a7a7a",
       "mu": "#1f5fa8", "pid": "#2a9d4b", "smc": "#c23b3b"}
LW = {"mu": 2.4, "pid": 1.5, "smc": 0.8}   # decreasing widths: coinciding curves stay visible


def figures(results, truths, info):
    plt = setup_mpl()
    t = np.arange(N_DAYS + 1)
    idx = {}
    for i, r in enumerate(results):
        rr = r["row"]
        idx[(rr["observer"], rr["controller"], rr["plant"], rr["delta"], rr["mode"], rr["seed"])] = i
    get = lambda obs, ctrl, plant, d, mode="M1", seed=-1: results[idx[(obs, ctrl, plant, d, mode, seed)]]
    written = []
    # 1) per controller, M1 nominal: x2 & x2_hat, x1 & x1_hat, e1 (log)
    for ctrl in CTRLS:
        X = truths[(ctrl, "nominal")]
        fig, ax = plt.subplots(3, 1, figsize=(6.3, 6.6), sharex=True)
        ax[0].semilogy(t, X[:, 1], color=COL["true"], lw=1.6, label=r"$x_2$ (plant)")
        for obs, ls in (("lsd", "-"), ("ekf", "--")):
            ax[0].semilogy(t, np.maximum(get(obs, ctrl, "nominal", DELTA)["xh"][:, 1], 1e-12),
                           color=COL[obs], ls=ls, label=rf"$\hat x_2$, {OBS_LABEL[obs]}")
        ax[0].set_ylabel(r"Glioma density $x_2$ [-]")
        ax[0].legend(loc="lower left")
        ax[1].plot(t, X[:, 0], color=COL["true"], lw=1.6, label=r"$x_1$ (plant)")
        for obs, ls in (("lsd", "-"), ("ekf", "--")):
            for d, a in ((DELTA, 1.0), (-DELTA, 0.55)):
                ax[1].plot(t, get(obs, ctrl, "nominal", d)["xh"][:, 0], color=COL[obs], ls=ls, alpha=a,
                           label=rf"$\hat x_1$, {OBS_LABEL[obs]}, $\delta$ = {d:+g}")
        ax[1].set_ylabel(r"Glial density $x_1$ [-]")
        ax[1].legend(loc="lower right", ncol=1)
        for obs, ls in (("lsd", "-"), ("ekf", "--"), ("lzoh", ":")):
            for d, a in ((DELTA, 1.0), (-DELTA, 0.55)):
                e1 = get(obs, ctrl, "nominal", d)["e1"]
                ax[2].semilogy(t, np.maximum(e1, 1e-9), color=COL[obs], ls=ls, alpha=a,
                               label=rf"{OBS_LABEL[obs]}, $\delta$ = {d:+g}")
        ax[2].axhline(CONV_THR, color="0.4", lw=0.7, ls="-.")
        ax[2].set_ylabel(r"Relative error $e_1$ [-]")
        ax[2].set_xlabel("Time from start of treatment [days]")
        ax[2].set_ylim(1e-8, 2)
        ax[2].legend(loc="lower left", ncol=2, fontsize=6.5)
        for a in ax:
            a.set_xlim(0, N_DAYS)
        fig.align_ylabels(ax)
        fig.tight_layout()
        save_fig(fig, f"fig_{ctrl}_M1_nominal"); plt.close(fig); written.append(f"fig_{ctrl}_M1_nominal")
    # 2) e1 comparison of the three controllers (M1, nominal, +delta)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for a, obs in zip(ax, ("lsd", "ekf")):
        for ctrl in CTRLS:
            a.semilogy(t, np.maximum(get(obs, ctrl, "nominal", DELTA)["e1"], 1e-9), color=COL[ctrl],
                       lw=LW[ctrl], label=CTRL_LABEL[ctrl])
        a.axhline(CONV_THR, color="0.4", lw=0.7, ls="-.")
        a.set_xlim(0, N_DAYS); a.set_ylim(1e-8, 2)
        a.set_xlabel("Time from start of treatment [days]")
        a.text(0.97, 0.95, OBS_LABEL[obs], transform=a.transAxes, ha="right", va="top", fontsize=8)
    ax[0].set_ylabel(r"Relative error $e_1$ [-]")
    ax[0].legend(loc="lower left")
    fig.tight_layout(); save_fig(fig, "fig_e1_controllers_M1"); plt.close(fig)
    written.append("fig_e1_controllers_M1")
    # 3) five parameter groups (M1, +delta)
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, sharey=True)
    gcol = dict(zip(GROUPS, ["black", "#1f5fa8", "#2a9d4b", "#c23b3b", "#8e44ad", "#d1691d"]))
    for i, obs in enumerate(("lsd", "ekf")):
        for j, ctrl in enumerate(CTRLS):
            a = ax[i, j]
            for plant in GROUPS:
                a.semilogy(t, np.maximum(get(obs, ctrl, plant, DELTA)["e1"], 1e-9), color=gcol[plant],
                           lw=1.0, label=GROUP_LABEL[plant])
            a.axhline(CONV_THR, color="0.4", lw=0.7, ls="-.")
            a.set_xlim(0, N_DAYS); a.set_ylim(1e-7, 30)
            a.text(0.97, 0.04, f"{CTRL_LABEL[ctrl]}, {'EKF' if obs == 'ekf' else 'Luenberger'}",
                   transform=a.transAxes, ha="right", va="bottom", fontsize=7.5)
            if j == 0:
                a.set_ylabel(r"Relative error $e_1$ [-]")
            if i == 1:
                a.set_xlabel("Time [days]")
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); save_fig(fig, "fig_e1_groups_M1"); plt.close(fig)
    written.append("fig_e1_groups_M1")
    # 4) M2/M3: median and 5-95 % band over seeds (nominal plant, +delta)
    noisy = [m for m, _, n in MODES if n]
    fig, ax = plt.subplots(2, len(noisy), figsize=(7.2, 4.4), sharex=True, sharey=True)
    for i, obs in enumerate(("lsd", "ekf")):
        for j, mode in enumerate(noisy):
            a = ax[i, j]
            for ctrl in CTRLS:
                E = np.array([get(obs, ctrl, "nominal", DELTA, mode, s)["e1"] for s in range(N_SEEDS)])
                E = np.where(np.isfinite(E), E, 10.0)
                lo, med, hi = np.percentile(E, [5, 50, 95], axis=0)
                a.fill_between(t, np.maximum(lo, 1e-9), np.maximum(hi, 1e-9), color=COL[ctrl], alpha=0.15, lw=0)
                a.semilogy(t, np.maximum(med, 1e-9), color=COL[ctrl], lw=LW[ctrl], label=CTRL_LABEL[ctrl])
            a.axhline(CONV_THR, color="0.4", lw=0.7, ls="-.")
            a.set_xlim(0, N_DAYS); a.set_ylim(1e-5, 3)
            a.text(0.97, 0.95, f"{'EKF' if obs == 'ekf' else 'Luenberger'}; {MODE_LABEL[mode]}",
                   transform=a.transAxes, ha="right", va="top", fontsize=6.8)
            if j == 0:
                a.set_ylabel(r"Relative error $e_1$ [-]")
            if i == 1:
                a.set_xlabel("Time [days]")
    ax[0, 0].legend(loc="lower left", fontsize=6.5)
    fig.tight_layout(); save_fig(fig, "fig_e1_noise_sparse"); plt.close(fig)
    written.append("fig_e1_noise_sparse")
    # 5) EKF standard deviation vs actual error, and observability measures (nominal, M1 and M2)
    fig, ax = plt.subplots(3, 1, figsize=(6.3, 6.0), sharex=True)
    for ctrl in CTRLS:
        for mode, ls in (("M1", "-"), ("M2", "--")):
            r = get("ekf", ctrl, "nominal", DELTA, mode, -1 if mode == "M1" else 0)
            X = truths[(ctrl, "nominal")]
            ax[0].semilogy(t, np.maximum(r["sd"][:, 0]/X[:, 0], 1e-9), color=COL[ctrl], ls=ls, lw=LW[ctrl],
                           label=rf"{CTRL_LABEL[ctrl]}, {mode}" + (" (seed 0)" if mode == "M2" else ""))
            ax[1].semilogy(t, np.maximum(r["e1"], 1e-9), color=COL[ctrl], ls=ls, lw=LW[ctrl])
    ax[0].set_ylabel(r"EKF $\sqrt{P_{11}}/x_1$ [-]"); ax[0].legend(loc="lower left", ncol=2, fontsize=6.5)
    ax[1].set_ylabel(r"EKF error $e_1$ [-]")
    for a in ax[:2]:
        a.axhline(CONV_THR, color="0.4", lw=0.7, ls="-.")
    for ctrl in CTRLS:
        X = truths[(ctrl, "nominal")]
        b2x2 = NOMINAL["b2"]*X[:, 1]
        ax[2].semilogy(t, b2x2, color=COL[ctrl], label=rf"$|\beta_2 x_2|$, {CTRL_LABEL[ctrl]}")
    ax[2].set_ylabel(r"$|\beta_2 x_2|$ [day$^{-1}$]")
    ax2 = ax[2].twinx()
    for ctrl in CTRLS:
        X = truths[(ctrl, "nominal")]
        cond = [np.linalg.cond(obs_matrix(jac2_analytic(X[k, 0], X[k, 1], X[k, 3])[0])) for k in range(N_DAYS + 1)]
        ax2.semilogy(t, cond, color=COL[ctrl], ls=":", lw=1.0)
    ax2.set_ylabel(r"cond$([C;\,CA])$ (dotted) [-]")
    ax[2].legend(loc="center left", fontsize=6.5)
    ax[2].set_xlabel("Time from start of treatment [days]")
    for a in ax:
        a.set_xlim(0, N_DAYS)
    fig.align_ylabels(ax); fig.tight_layout(); save_fig(fig, "fig_ekf_sd_observability"); plt.close(fig)
    written.append("fig_ekf_sd_observability")
    print("  figures: " + ", ".join(written) + " (png 300 dpi + pdf)")


def step4(results, truths, x4hs, schedules, info, summary_sched):
    print("=" * 78 + "\nSTEP 4 - outputs\n" + "=" * 78)
    df = pd.DataFrame([r["row"] for r in results])
    df.insert(0, "run_id", np.arange(len(df)))
    write_csv(RES_DIR / "metrics_all.csv", df, float_format="%.6g")
    summ = summarize(df)
    write_csv(RES_DIR / "summary_by_scenario.csv", summ, float_format="%.6g")
    md, ctab = comparison_table(summ)
    write_text(RES_DIR / "comparison_table.md", md)
    write_csv(RES_DIR / "comparison_table.csv", ctab, float_format="%.6g")
    rd = [x for r in results for x in r["rankdef"]]
    write_csv(RES_DIR / "rank_deficient_updates.csv",
              pd.DataFrame(rd, columns=["observer", "controller", "plant", "delta", "mode", "seed",
                                        "day", "x1_hat", "x2_hat"]))
    print(f"  metrics_all.csv ({len(df)} runs), summary_by_scenario.csv, comparison_table.md/.csv, "
          f"rank_deficient_updates.csv ({len(rd)} rows)")
    # time series: truth (x, u, x4_hat, observability) and estimates (x_hat, y [, sd])
    t = np.arange(N_DAYS + 1)
    for ctrl in CTRLS:
        parts = []
        for plant in GROUPS:
            X = truths[(ctrl, plant)]
            u = np.r_[schedules[ctrl], np.nan]
            parts.append(pd.DataFrame({"plant": plant, "day": t, "x1": X[:, 0], "x2": X[:, 1], "x3": X[:, 2],
                                       "x4": X[:, 3], "x4_hat_nominal_zeta": x4hs[ctrl], "u": u,
                                       "b2_x2": NOMINAL["b2"]*X[:, 1]}))
        write_csv(TS_DIR / f"truth_{ctrl}.csv.gz", pd.concat(parts), gz=True, float_format="%.9e")
    groups = {}
    for i, r in enumerate(results):
        if r["xh"] is None:
            continue
        rr = r["row"]
        d = {"run_id": i, "day": t, "x1_hat": r["xh"][:, 0], "x2_hat": r["xh"][:, 1], "y": r["y"]}
        if r["sd"] is not None:
            d["sd_x1"] = r["sd"][:, 0]; d["sd_x2"] = r["sd"][:, 1]
        groups.setdefault((rr["observer"], rr["controller"], rr["mode"]), []).append(pd.DataFrame(d))
    for (obs, ctrl, mode), parts in groups.items():
        write_csv(TS_DIR / f"est_{obs}_{ctrl}_{mode}.csv.gz", pd.concat(parts), gz=True, float_format="%.9e")
    print(f"  timeseries: truth_<ctrl>.csv.gz and {len(groups)} est_<observer>_<ctrl>_<mode>.csv.gz files")
    figures(results, truths, info)
    return df, summ


# =============================================================================
# MAIN
# =============================================================================
def main():
    tee = Tee()
    sys.stdout = tee
    t0 = time.time()
    print(f"observer.py - {datetime.now():%Y-%m-%d %H:%M:%S} - Python {sys.version.split()[0]}, "
          f"numpy {np.__version__}, pandas {pd.__version__}")
    print("SETTINGS: " + ", ".join(f"{k}={globals()[k]}" for k in (
        "DELTA", "SIGMA", "TM_LIST", "N_SEEDS", "SEED_BASE", "CONV_THR", "TRIGGER", "K_OBS", "P_MIN",
        "POLE_SEP", "FD_REL", "FD_X4_REL", "EKF_Q_REL", "EKF_R_REL_M1", "N_SEEDS_TIMESERIES", "N_WORKERS")))
    try:
        x_pre, schedules, summary_sched = step1()
        write_text(HERE / "run_log.txt", tee.buf.getvalue())
        if "--step1" in sys.argv:
            return
        info = step2(x_pre)
        truths, x4hs, results = step3(x_pre, schedules)
        step4(results, truths, x4hs, schedules, info, summary_sched)
        print(f"done in {time.time() - t0:.0f} s")
    finally:
        sys.stdout = sys.__stdout__
        write_text(HERE / "run_log.txt", tee.buf.getvalue())


if __name__ == "__main__":
    main()
