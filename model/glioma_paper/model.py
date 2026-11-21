"""Normalized glioma model (Iarosz et al.), reference trajectory and uncertainty groups.

States: x1 glial cells, x2 glioma cells, x3 neurons, x4 drug concentration; input u is the
infusion rate [mg m^-2 day^-1]. Integration: classical RK4, dt = 1 day, input held over the
step, x2 clipped at 0 after each step. Time t = 0 is the start of treatment (simulation day 600).
"""
import math

import numpy as np

NOMINAL = dict(Om1=0.0068, Om2=0.012, b1=1.8e-2, b2=1.8e-3,
               p1=4.7e-8, p2=4.7e-5, p3=4.7e-8,
               a1=1.0, a2=1.0, a3=1.0, alpha=2.0, zeta=0.2)

DT = 1.0            # RK4 step and controller update period [day]
T_START = 600       # pre-conditioning with u = 0 [day]
T_REF = 2000        # treatment horizon [day]
Y0 = (1.0, 1e-3, 1.0, 0.0)
U_MIN, U_MAX = 50.0, 200.0
A_REF, B_REF = 7.5, 0.0
X2_THRESHOLD = 1e-4
X1_RECOVERY = 0.99
X_PRE_EXPECTED = (0.670934, 0.313731, 0.517817, 0.0)   # asserted to 1e-6

# Uncertainty groups: plant only, nominal value multiplied by (1 + fraction).
GROUPS = {
    "nominal": {},
    "G1": {"Om1": -0.20, "Om2": +0.20},
    "G2": {"b1": +0.50, "b2": -0.50},
    "G3": {"p1": +0.15, "p2": -0.15},
    "G4": {"a1": +0.20, "a2": +0.20},
    "G5": {"zeta": +0.20},
}
GROUP_LABELS = {
    "nominal": "Nominal",
    "G1": "Omega1 -20%, Omega2 +20%",
    "G2": "beta1 +50%, beta2 -50%",
    "G3": "p1 +15%, p2 -15%",
    "G4": "a1 +20%, a2 +20%",
    "G5": "zeta +20%",
}


def perturbed(group, base=None):
    """Parameter set of one uncertainty group."""
    p = dict(NOMINAL if base is None else base)
    for k, frac in GROUPS[group].items():
        p[k] = p[k] * (1 + frac)
    return p


def rhs(x, u, p):
    """Right-hand side of the normalized model; H(0) = 0.5."""
    x1, x2, x3, x4 = x
    dx1 = p["Om1"]*x1*(1-x1) - p["b1"]*x1*x2 - p["p1"]*x1*x4/(p["a1"]+x1)
    dx2 = p["Om2"]*x2*(1-x2) - p["b2"]*x1*x2 - p["p2"]*x2*x4/(p["a2"]+x2)
    if dx1 < 0:
        heav = 1.0
    elif dx1 == 0:
        heav = 0.5
    else:
        heav = 0.0
    dx3 = p["alpha"]*heav*dx1*x3 - p["p3"]*x3*x4/(p["a3"]+x3)
    dx4 = u - p["zeta"]*x4
    return (dx1, dx2, dx3, dx4)


def rk4_step(x, u, p, dt=DT):
    """One RK4 step with the input held constant, then x2 clipped at 0."""
    h = dt/2
    k1 = rhs(x, u, p)
    k2 = rhs(tuple(xi + h*ki for xi, ki in zip(x, k1)), u, p)
    k3 = rhs(tuple(xi + h*ki for xi, ki in zip(x, k2)), u, p)
    k4 = rhs(tuple(xi + dt*ki for xi, ki in zip(x, k3)), u, p)
    c = dt/6
    xn = [xi + c*(a + 2*b + 2*cc + d) for xi, a, b, cc, d in zip(x, k1, k2, k3, k4)]
    if xn[1] < 0.0:
        xn[1] = 0.0
    return tuple(xn)


def precondition(p=None, check=True):
    """600 days with u = 0 from Y0. Returns (t, X) with X of shape (601, 4)."""
    p = NOMINAL if p is None else p
    x = Y0
    X = [x]
    for _ in range(int(T_START/DT)):
        x = rk4_step(x, 0.0, p)
        X.append(x)
    X = np.array(X)
    if check:
        assert np.max(np.abs(X[-1] - np.array(X_PRE_EXPECTED))) < 1e-6, X[-1]
    return np.arange(len(X))*DT, X


def x_pre():
    """Nominal state at the start of treatment."""
    return tuple(float(v) for v in precondition()[1][-1])


def s_curve(tau, a, b):
    return 0.5*(1 - np.tanh(a*(tau - b)))


def reference(t, x2_pre, a=A_REF, b=B_REF, T=T_REF):
    """r(t) = x2_pre*(S(t/T) - S(1))/(S(0) - S(1)), S(tau) = 0.5*(1 - tanh(a*(tau - b)))."""
    t = np.asarray(t, dtype=float)
    Sc = s_curve(t/T, a, b)
    S0 = 0.5*(1 - math.tanh(a*(0.0 - b)))
    S1 = 0.5*(1 - math.tanh(a*(1.0 - b)))
    return x2_pre*(Sc - S1)/(S0 - S1)
