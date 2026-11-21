"""
Lyapunov function W along simulated runs (needs numpy, pandas, matplotlib).

2026 revision of the authors' verify_proof_plots.py:
  * SMC parameters set to the final v2 values (SMC+PID+General/README.md), reference a = 7.5;
  * the closed loop is the final discrete SMC of smc_simulation: true-state feedback, no stop
    rule (the old in-loop observer and the x2 <= 1e-5 stop of the old notebook are not in the
    final code and were removed here);
  * added: W along the replay of the nominal PID, SMC and mu schedules on the nominal plant
    and on the five uncertainty groups (the Table 6 protocol, E1).
W, eta, c and the exact derivative are unchanged. Run:  python verify_proof_plots.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "results", "figure_data")

NOM = dict(Om1=0.0068, Om2=0.012, b1=1.8e-2, b2=1.8e-3, p1=4.7e-8, p2=4.7e-5,
           p3=4.7e-8, a1=1.0, a2=1.0, a3=1.0, alpha=2.0, zeta=0.2)


def make_plant(P):
    def plant(x, u):
        x1, x2, x3, x4 = x
        dx1 = P['Om1']*x1*(1-x1) - P['b1']*x1*x2 - P['p1']*x1*x4/(P['a1']+x1)
        dx2 = P['Om2']*x2*(1-x2) - P['b2']*x1*x2 - P['p2']*x2*x4/(P['a2']+x2)
        heav = np.heaviside(-dx1, 0.5)
        dx3 = P['alpha']*heav*dx1*x3 - P['p3']*x3*x4/(P['a3']+x3)
        dx4 = u - P['zeta']*x4
        return np.array([dx1, dx2, dx3, dx4])
    return plant


plant_nominal = make_plant(NOM)


def rk4_step(f, x, u, dt):
    k1 = f(x, u); k2 = f(x+dt*k1/2, u); k3 = f(x+dt*k2/2, u); k4 = f(x+dt*k3, u)
    return x + dt*(k1+2*k2+2*k3+k4)/6


def fg(x1, x2, x3, x4, P=NOM):
    """Lie derivatives of y = x2 used by the controller (always with NOMINAL P)."""
    F1 = P['Om1']*x1*(1-x1) - P['b1']*x1*x2 - P['p1']*x1*x4/(P['a1']+x1)
    F2 = P['Om2']*x2*(1-x2) - P['b2']*x1*x2 - P['p2']*x2*x4/(P['a2']+x2)
    d1 = -P['b2']*x2
    d2 = P['Om2']*(1-2*x2) - P['b2']*x1 - P['p2']*x4*P['a2']/(P['a2']+x2)**2
    d4 = -P['p2']*x2/(P['a2']+x2)
    f = d1*F1 + d2*F2 + d4*(-P['zeta']*x4)
    g = d4
    return f, g


def generate_smc_references(dt, T_ref, a, b, x_burn_val, delta_sharp, delta_off, delta_init, delta_final):
    N = int(T_ref/dt); tt = np.arange(N+1)*dt
    S0 = 0.5*(1-np.tanh(a*(0.0-b))); S1 = 0.5*(1-np.tanh(a*(1.0-b)))
    S = 0.5*(1-np.tanh(a*(tt/T_ref-b))); r = x_burn_val*((S-S1)/(S0-S1))
    z = tt/T_ref-b; sech2 = 1/(np.cosh(a*z)**2)
    dr = r[0]*((-0.5*a*sech2)/(S0-S1))/T_ref
    ddr = r[0]*((a*a*sech2*np.tanh(a*z))/(S0-S1))/(T_ref**2)
    Sd0 = 0.5*(1-np.tanh(delta_sharp*(0.0-delta_off))); Sd1 = 0.5*(1-np.tanh(delta_sharp*(1.0-delta_off)))
    Sd = 0.5*(1-np.tanh(delta_sharp*(tt/T_ref-delta_off)))
    d = delta_final + ((Sd-Sd1)/(Sd0-Sd1))*(delta_init-delta_final)
    d = np.minimum.accumulate(d)
    return tt, r, dr, ddr, d


# final v2 SMC parameters (SMC+PID+General/README.md; smc_simulation.ipynb)
DT, T_START, T_REF = 1.0, 600.0, 2000.0
UMIN, UMAX = 50.0, 200.0
LAMBDA_P = 1.8813613583755686       # coefficient of e
LAMBDA_D = 9.619198702305367        # coefficient of de/dt
LAMBDA_I = 2.9470610386036095e-05   # coefficient of int(e)
A_SMC, B_SMC = 7.5, 0.0
DELTA_INIT, DELTA_FINAL = 0.013664689577712264, 2.676680232434659e-07
A_DELTA, B_DELTA = 7.824136642893967, 0.10364584462506671


def preconditioned_state():
    x = np.array([1.0, 1e-3, 1.0, 0.0])
    for _ in range(int(T_START/DT)):
        x = rk4_step(plant_nominal, x, 0.0, DT); x[1] = max(x[1], 0.0)
    return x


def run_closed_loop(true_params=None):
    """Final discrete SMC (smc_simulation, v2): true-state feedback, no stop rule. true_params
    perturbs ONLY the simulated plant; the controller keeps the nominal model. Returns T (0..2000),
    X (states) and U (U[k] = dose applied on [k, k+1), k = 0..1999)."""
    plant_true = make_plant(true_params) if true_params is not None else plant_nominal
    x = preconditioned_state()
    _, x2_ref, drdt_arr, ddrdt_arr, delta_arr = generate_smc_references(
        DT, T_REF, A_SMC, B_SMC, x[1], A_DELTA, B_DELTA, DELTA_INIT, DELTA_FINAL)
    t = 0.0; u_prev = UMAX; int_e = 0.0; idx_max = len(x2_ref)-1; delta_prev = None
    Tl = [0.0]; Xl = [x.copy()]; Ul = []; Sl = []
    while t < T_REF:
        idx = min(int(np.floor(t/DT)), idx_max)
        r, drdt, ddrdt = x2_ref[idx], drdt_arr[idx], ddrdt_arr[idx]
        e = x[1]-r; de = plant_nominal(x, 0.0)[1]-drdt; int_e += e*DT
        fv, gval = fg(*x)
        s = (fv+gval*u_prev-ddrdt) + LAMBDA_D*de + LAMBDA_P*e + LAMBDA_I*int_e
        delta_k = delta_arr[idx]
        if delta_prev is not None: delta_k = min(delta_prev, delta_k)
        delta_prev = delta_k
        if gval >= 0.0:
            u = UMIN if s > delta_k else (UMAX if s < -delta_k else u_prev)
        else:
            u = UMAX if s > delta_k else (UMIN if s < -delta_k else u_prev)
        x = rk4_step(plant_true, x, u, DT); x[1] = max(x[1], 0.0)
        t += DT; u_prev = u
        Tl.append(t); Xl.append(x.copy()); Ul.append(u); Sl.append(s)
    return np.array(Tl), np.array(Xl), np.array(Ul), np.array(Sl)


def replay(u, x0, P):
    f = make_plant(P)
    x = np.array(x0, dtype=float); X = [x.copy()]
    for uk in u:
        x = rk4_step(f, x, uk, DT); x[1] = max(x[1], 0.0); X.append(x.copy())
    return np.array(X)


# ----------------------------------------------------------------------
# Lyapunov function W and its EXACT analytic derivative Wdot (unchanged)
# ----------------------------------------------------------------------
c = 3.0
x4bar = 50.0/0.2          # = Umin/zeta, nominal zeta used in the certificate's design


def W_of(X, eta, P=NOM):
    x1, x2, x4 = X[:, 0], X[:, 1], X[:, 3]
    return (x1-1-np.log(x1)) + c*x2 + eta*np.maximum(0.0, x4bar-x4)


def Wdot_exact(X, U, eta, P):
    """Exact analytic dW/dt along the TRUE plant (with its own P), for each logged sample."""
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    dx1 = P['Om1']*x1*(1-x1) - P['b1']*x1*x2 - P['p1']*x1*x4/(P['a1']+x1)
    dx2 = P['Om2']*x2*(1-x2) - P['b2']*x1*x2 - P['p2']*x2*x4/(P['a2']+x2)
    dx4 = U - P['zeta']*x4
    dW1 = -(1-x1)/x1 * dx1
    dW2 = c*dx2
    dW4 = np.where(x4 < x4bar, -eta*dx4, 0.0)   # d/dt[eta*max(0,x4bar-x4)], subgradient at the kink
    return dW1 + dW2 + dW4


x10, x20 = 0.6709337312035838, 0.3137305348300217
V0 = (x10-1-np.log(x10)) + c*x20
W0_fp = V0
for _ in range(200):
    x2bar = min(W0_fp/c, 1.0)
    eta = c*NOM['p2']*x2bar/(NOM['zeta']*(NOM['a2']+x2bar))
    W0_fp = V0 + eta*x4bar

GROUPS = {"nominal": {}, "G1": {'Om1': 0.8, 'Om2': 1.2}, "G2": {'b1': 1.5, 'b2': 0.5}, "G3": {'p1': 1.15, 'p2': 0.85},
          "G4": {'a1': 1.2, 'a2': 1.2}, "G5": {'zeta': 1.2}}
TOL = 1e-10


def params(g):
    P = dict(NOM)
    for k, v in GROUPS[g].items():
        P[k] = NOM[k]*v
    return P


def analyse(X, U, P):
    Ua = np.concatenate([U, U[-1:]])          # dose on day k for the sample of day k (last one repeated)
    W = W_of(X, eta, P); Wd = Wdot_exact(X, Ua, eta, P)
    bad = Wd > TOL
    m_ = P['Om2'] - P['p2']*(UMIN/P['zeta'])/P['a2']
    return W, Wd, dict(W0=float(W[0]), WT=float(W[-1]), x2T=float(X[-1, 1]), n_days_wdot_pos=int(bad.sum()),
                       first_day_wdot_pos=(int(np.flatnonzero(bad)[0]) if bad.any() else None),
                       max_wdot=float(Wd.max()), m=float(m_), beta2=float(P['b2']), condition_m_lt_beta2=bool(m_ < P['b2']),
                       monotonicity_ok=bool(P['p2']*(UMIN/P['zeta']) <= P['Om2']*P['a2']**2))


rows, series = [], []
x_pre = preconditioned_state()

# ---- 1) closed-loop SMC (controller in the loop) on the nominal plant and on the five groups
sched = pd.read_csv(os.path.join(DATA, "schedules.csv"))
T, X, U, S = run_closed_loop(None)
same_u = bool(np.array_equal(U, sched["u_smc"].to_numpy()))
nd = pd.read_csv(os.path.join(DATA, "nominal_discrete.csv"))
Xfin = nd[nd.controller == "SMC"][["x1", "x2", "x3", "x4"]].to_numpy()
dev = float(np.max(np.abs(X - Xfin)))
print(f"closed-loop SMC (v2) reproduces the final SMC run: same doses {same_u}, max abs state deviation {dev:.2e}")
Wn, Wdn, _ = analyse(X, U, NOM)
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
axs[0].plot(T, X[:, 0], label='$x_1$ (glial)'); axs[0].plot(T, X[:, 1], label='$x_2$ (glioma)'); axs[0].plot(T, X[:, 2], label='$x_3$ (neuron)')
axs[0].set_yscale('log'); axs[0].legend(); axs[0].set_title('States (nominal parameters, discrete SMC v2)'); axs[0].grid(alpha=0.3)
axs[1].plot(T, Wn, color='tab:purple'); axs[1].set_yscale('log'); axs[1].set_title('Lyapunov function $W(t)$ (log scale)'); axs[1].grid(alpha=0.3)
axs[2].plot(T, Wdn, color='tab:red', label='exact analytic $\\dot W$')
axs[2].plot(T, np.gradient(Wn, T), '--', color='gray', alpha=0.6, label='numerical derivative of $W$')
axs[2].axhline(0, color='k', linewidth=0.8); axs[2].legend(); axs[2].grid(alpha=0.3); axs[2].set_xlabel('time [days]')
plt.tight_layout(); plt.savefig(os.path.join(HERE, 'fig1_nominal.png'), dpi=110); plt.close(fig)

fig, axs = plt.subplots(3, 2, figsize=(13, 11)); axs = axs.flatten()
for ax, g in zip(axs, GROUPS):
    P = params(g)
    Tg, Xg, Ug, _ = run_closed_loop(P if GROUPS[g] else None)
    W, Wd, info = analyse(Xg, Ug, P)
    rows.append(dict(protocol="closed_loop", controller="SMC", group=g, **info))
    series.append(pd.DataFrame(dict(protocol="closed_loop", controller="SMC", group=g, time_day=Tg, W=W, Wdot=Wd)))
    ax.plot(Tg, W, color='tab:purple'); ax.set_yscale('log'); ax.set_title(g, fontsize=10); ax.grid(alpha=0.3); ax.set_xlabel('time [days]')
plt.tight_layout(); plt.savefig(os.path.join(HERE, 'fig2_uncertainty.png'), dpi=110); plt.close(fig)

# ---- 2) E1: nominal schedules of the three controllers replayed on each plant (Table 6 protocol)
for ctl, col in (("PID", "u_pid"), ("SMC", "u_smc"), ("mu", "u_mu")):
    u = sched[col].to_numpy()
    for g in GROUPS:
        P = params(g)
        Xg = replay(u, x_pre, P)
        W, Wd, info = analyse(Xg, u, P)
        rows.append(dict(protocol="E1_replay", controller=ctl, group=g, **info))
        series.append(pd.DataFrame(dict(protocol="E1_replay", controller=ctl, group=g, time_day=np.arange(len(W)), W=W, Wdot=Wd)))

# ---- 3) constant doses on the nominal plant (the two bounds of the admissible set)
for name, uval in (("u=50", 50.0), ("u=200", 200.0)):
    u = np.full(2000, uval)
    Xg = replay(u, x_pre, NOM)
    W, Wd, info = analyse(Xg, u, NOM)
    rows.append(dict(protocol="constant_dose", controller=name, group="nominal", **info))

df = pd.DataFrame(rows)
p = os.path.join(DATA, "lyapunov_wdot_summary.csv"); df.to_csv(p + ".tmp", index=False); os.replace(p + ".tmp", p)
ts = pd.concat(series, ignore_index=True)
p = os.path.join(DATA, "lyapunov_W_timeseries.csv"); ts.to_csv(p + ".tmp", index=False, float_format="%.10g"); os.replace(p + ".tmp", p)
out = dict(c=c, eta=float(eta), x4bar=x4bar, tol=TOL, closed_loop_reproduces_final_smc=dict(same_doses=same_u, max_abs_state_dev=dev),
           sup_abs_s_closed_loop=float(np.max(np.abs(S))), rows=rows)
p = os.path.join(HERE, "results_plots.json")
with open(p + ".tmp", "w") as f:
    json.dump(out, f, indent=1)
os.replace(p + ".tmp", p)

print(f"eta = {eta:.6e}, c = {c}; tolerance for Wdot > 0: {TOL:g}")
print(f"{'protocol':14s} {'ctrl':6s} {'group':8s} {'W(0)':>8s} {'W(T)':>10s} {'x2(T)':>10s} {'#days Wdot>0':>13s} {'first':>6s} {'m<b2':>5s}")
for r in rows:
    print(f"{r['protocol']:14s} {r['controller']:6s} {r['group']:8s} {r['W0']:8.4f} {r['WT']:10.3e} {r['x2T']:10.3e} "
          f"{r['n_days_wdot_pos']:13d} {str(r['first_day_wdot_pos']):>6s} {str(r['condition_m_lt_beta2']):>5s}")
