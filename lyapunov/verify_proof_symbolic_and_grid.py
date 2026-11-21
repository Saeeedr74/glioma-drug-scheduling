"""
Lyapunov analysis of the nominal glioma model under any dose signal in [50, 200].
Needs: numpy, sympy. Run:  python verify_proof_symbolic_and_grid.py
(No controller parameter enters this file: the result holds for any admissible dose signal.)

PART A — symbolic re-derivation of the Lyapunov inequality
PART B — rigorous grid check (exact worst-case corner bound per cell, not sampling)
PART C — sanity check against the actual closed-loop trajectory
"""

import numpy as np
import sympy as sp

print("="*70)
print("PART A: symbolic re-derivation of the bound on dW/dt")
print("="*70)

# symbols
x1, x2, x4, w = sp.symbols('x1 x2 x4 w', positive=True)
Om1, Om2, b1, b2, p1, p2, a1, a2, zeta = sp.symbols(
    'Omega1 Omega2 beta1 beta2 p1 p2 a1 a2 zeta', positive=True)
U = sp.symbols('U', positive=True)   # generic dose, U in [Umin,Umax]

# ---- term 1: d/dt( x1 - 1 - ln x1 ) = (1 - 1/x1) * dx1/dt = -((1-x1)/x1) * dx1 ----
dx1 = Om1*x1*(1-x1) - b1*x1*x2 - p1*x1*x4/(a1+x1)
dV1 = sp.simplify(-(1-x1)/x1 * dx1)     # = d/dt(x1 - 1 - ln x1)   [d/dx1(x1-1-ln x1)=1-1/x1=-w/x1]
dV1_w = sp.simplify(dV1.subs(x1, 1-w))
print("\nd/dt(x1-1-ln x1)  in terms of w=1-x1 :")
sp.pprint(sp.expand(dV1_w))

# claimed bound: -Om1*w^2 + b1*w*x2 + w*p1*x4/(a1+x1), with x4 <= Umax/zeta as the worst case for +p1 term
# check: is  dV1_w - ( -Om1*w^2 + b1*w*x2 + p1*w*x4/(a1+1-w) )  identically zero?
claimed1 = -Om1*w**2 + b1*w*x2 + p1*w*x4/(a1 + (1-w))
diff1 = sp.simplify(dV1_w - claimed1)
print("\nCheck 1 -- (d/dt term1) minus (claimed exact expression), should be exactly 0:")
sp.pprint(diff1)
assert diff1 == 0, "MISMATCH in term 1 -- stop and recheck the derivation"
print(">>> PASS: term-1 algebra is exact.")

# ---- term 2: d/dt( c*x2 ) = c * dx2/dt ----
c = sp.symbols('c', positive=True)
dx2 = Om2*x2*(1-x2) - b2*x1*x2 - p2*x2*x4/(a2+x2)
dV2 = sp.expand(c*dx2)
print("\nd/dt(c*x2) =")
sp.pprint(dV2)
# rewrite x1 = 1-w and factor x2
dV2_w = sp.expand(dV2.subs(x1, 1-w))
dV2_w_over_x2 = sp.simplify(dV2_w / x2)
print("\n(d/dt(c*x2)) / x2 , should equal  c*[Om2*(1-x2) - b2*(1-w) - p2*x4/(a2+x2)] :")
sp.pprint(sp.expand(dV2_w_over_x2))
claimed2_over_x2 = c*(Om2*(1-x2) - b2*(1-w) - p2*x4/(a2+x2))
diff2 = sp.simplify(dV2_w_over_x2 - claimed2_over_x2)
print("\nCheck 2 -- difference, should be exactly 0:")
sp.pprint(diff2)
assert diff2 == 0, "MISMATCH in term 2 -- stop and recheck"
print(">>> PASS: term-2 algebra is exact.")

print("\nConclusion of PART A: the inequality")
print("   dW/dt <= -Om1*w^2 + eps*w + x2*( b1*w + c*( psi(x2) - b2*(1-w) ) )")
print("with  eps = p1*Umax/(zeta*a1)  [worst case of p1*w*x4/(a1+x1) using x1<=1, x4<=Umax/zeta]")
print("and   psi(x2) = Om2*(1-x2) - p2*x4bar/(a2+x2),  x4bar = Umin/zeta  [worst case using x4>=x4bar]")
print("is algebraically exact, not an approximation.  (The only inequalities used are")
print(" x1<=1, x4<=Umax/zeta in term 1, and x4>=Umin/zeta -- i.e. the xi-cancellation -- in term 2.)")


print("\n" + "="*70)
print("PART B: rigorous grid check via exact worst-case corners (no sampling)")
print("="*70)

# Nominal parameters (Table 2 / FACTS.md)
P = dict(Om1=0.0068, Om2=0.012, b1=1.8e-2, b2=1.8e-3,
         p1=4.7e-8, p2=4.7e-5, a1=1.0, a2=1.0, zeta=0.2)
Umin, Umax = 50.0, 200.0
x4bar = Umin/P['zeta']           # 250
eps = P['p1']*Umax/(P['zeta']*P['a1'])
m = P['Om2'] - P['p2']*x4bar/P['a2']     # psi(0)
print(f"eps = {eps:.6e}   m = psi(0) = {m:.6e}   beta2 = {P['b2']:.3e}")
print(f"monotonicity needed (p2*x4bar <= Om2*a2^2): "
      f"{P['p2']*x4bar:.4e} <= {P['Om2']*P['a2']**2:.4e}  -> {P['p2']*x4bar <= P['Om2']*P['a2']**2}")
print(f"m < beta2 required for the conclusion:  {m:.4e} < {P['b2']:.4e}  -> {m < P['b2']}")

c = 3.0
# level set W<=W0, with W0 from the day-600 state (x4=0 there)
x10, x20 = 0.6709337312035838, 0.3137305348300217
V0 = (x10 - 1 - np.log(x10)) + c*x20
W0 = V0   # x4=0 at t=T_start -> the xi-term is eta*x4bar, handled via fixed point below
def psi(x2): return P['Om2']*(1-x2) - P['p2']*x4bar/(P['a2']+x2)

# fixed point for eta (as in lyap_cert.py): eta = c*p2*x2bar/(zeta*(a2+x2bar)), x2bar=min(W0/c,1)
W0_fp = V0
for _ in range(200):
    x2bar = min(W0_fp/c, 1.0)
    eta = c*P['p2']*x2bar/(P['zeta']*(P['a2']+x2bar))
    W0_fp = V0 + eta*x4bar
W0 = W0_fp; x2bar = min(W0/c, 1.0)
print(f"c={c}  eta={eta:.6e}  W0={W0:.6f}  x2bar={x2bar:.6f}")

# smallest x1 on {V<=W0}: solve x1-1-ln(x1)=W0 for x1<1
lo, hi = 1e-9, 1.0
for _ in range(300):
    mid = 0.5*(lo+hi)
    if mid - 1 - np.log(mid) > W0: lo = mid
    else: hi = mid
x1min = lo
print(f"x1min on the level set = {x1min:.6f}")

# EXACT worst-case-corner check (rigorous, not float sampling):
# Q(x1,x2) = -Om1*w^2 + eps*w + x2*(b1*w + c*(psi(x2)-b2*x1))   is, on a cell [x1a,x1b]x[x2a,x2b]:
#   - increasing in w=1-x1  except for the -Om1*w^2 term (concave) -> upper bound of the w-dependent
#     piece on a w-interval is attained at an ENDPOINT of the interval (checked both, take max)
#   - psi(x2) is decreasing (given monotonicity holds) -> for upper bound use x2 = x2a (left edge)
#   - the x2*(...) factor: since the bracket can be + or -, evaluate at both x2 endpoints and take max
# This is an exact interval bound: Q(x1,x2) <= max over the 4 corners of the cell, PROVABLY
# (not approximately), given psi monotone decreasing and the w-quadratic's concavity handled by
# checking both endpoints of w.
assert P['p2']*x4bar <= P['Om2']*P['a2']**2, "monotonicity of psi FAILED -- grid method below is invalid"

def worst_Q_on_grid(x1_lo, x1_hi, n1, x2_lo, x2_hi, n2):
    """Vectorized exact worst-case-corner upper bound of Q on every cell of the grid."""
    e1 = np.linspace(x1_lo, x1_hi, n1+1); e2 = np.linspace(x2_lo, x2_hi, n2+1)
    x1a = e1[:-1][:, None]; x1b = e1[1:][:, None]          # x1a < x1b
    x2a = e2[:-1][None, :]; x2b = e2[1:][None, :]          # x2a < x2b
    wa, wb = 1-x1b, 1-x1a                                   # wa < wb  (w = 1-x1)
    vtx = eps/(2*P['Om1'])
    f_w = np.maximum(-P['Om1']*wa**2+eps*wa, -P['Om1']*wb**2+eps*wb)
    in_range = (wa <= vtx) & (vtx <= wb)
    f_w = np.where(in_range, np.maximum(f_w, -P['Om1']*vtx**2+eps*vtx), f_w)
    psi_max = psi(x2a)                                      # psi decreasing -> largest at x2a
    bracket_max = P['b1']*wb + c*(psi_max - P['b2']*(1-wb))  # affine increasing in w -> max at wb
    Q_upper = f_w + x2b*np.maximum(0.0, bracket_max) + x2a*np.minimum(0.0, bracket_max)
    i, j = np.unravel_index(np.argmax(Q_upper), Q_upper.shape)
    return Q_upper.max(), (x1a[i,0], x1b[i,0], x2a[0,j], x2b[0,j])

worst, worst_cell = worst_Q_on_grid(x1min, 1.0, 4000, 0.0, x2bar, 4000)
print(f"\nRigorous worst-case Q upper bound over the whole level set: {worst:.6e}")
print(f"attained in cell x1 in [{worst_cell[0]:.6f},{worst_cell[1]:.6f}], "
      f"x2 in [{worst_cell[2]:.3e},{worst_cell[3]:.3e}]")
if worst < 0:
    print(">>> PASS: Q<0 EVERYWHERE on the level set -- even the full set, no residual needed at this grid size.")
else:
    print(">>> Q>=0 only very close to (1,0); refine the grid there (expected -- see below).")

# refine near (1,0) with a much finer local grid to pin down the true residual region
worst_f, worst_f_cell = worst_Q_on_grid(1-0.02, 1.0, 6000, 0.0, 5e-5, 6000)
print(f"\nFine local grid near (1,0), box w<=0.02, x2<=5e-5: worst Q = {worst_f:.6e}")
print(f"attained in cell x1 in [{worst_f_cell[0]:.8f},{worst_f_cell[1]:.8f}], "
      f"x2 in [{worst_f_cell[2]:.3e},{worst_f_cell[3]:.3e}]")

print("\n" + "="*70)
print("PART C: sanity check -- does W actually decrease along a real simulated trajectory?")
print("="*70)

def plant(x, u, P=P):
    x1, x2, x3, x4 = x
    dx1 = P['Om1']*x1*(1-x1) - P['b1']*x1*x2 - P['p1']*x1*x4/(P['a1']+x1)
    dx2 = P['Om2']*x2*(1-x2) - P['b2']*x1*x2 - P['p2']*x2*x4/(P['a2']+x2)
    dx3 = 0.0
    dx4 = u - P['zeta']*x4
    return np.array([dx1, dx2, dx3, dx4])

def rk4(x, u, dt=1.0):
    k1 = plant(x, u); k2 = plant(x+dt/2*k1, u); k3 = plant(x+dt/2*k2, u); k4 = plant(x+dt*k3, u)
    return x + dt/6*(k1+2*k2+2*k3+k4)

def W_of(x):
    return (x[0]-1-np.log(x[0])) + c*x[1] + eta*max(0.0, x4bar-x[3])

def V_of(x):
    return (x[0]-1-np.log(x[0])) + c*x[1]

# the residual region found above is the cell x1 in [worst_f_cell[0],worst_f_cell[1]],
# x2 in [worst_f_cell[2],worst_f_cell[3]]; its largest V-value is the correct level lstar
# such that Q<0 is guaranteed everywhere V>lstar (within the searched box).
lstar = (worst_f_cell[0]-1-np.log(worst_f_cell[0])) + c*worst_f_cell[3]
print(f"\nlstar (V-level enclosing the residual cell) = {lstar:.6e}")

rng = np.random.default_rng(0)
x = np.array([x10, x20, 0.5, 0.0])
Ws = [W_of(x)]; Vs = [V_of(x)]
us = rng.choice([Umin, Umax], size=4000)   # arbitrary admissible dosing, worst-case style
for u in us:
    x = rk4(x, u); x[1] = max(x[1], 0.0); Ws.append(W_of(x)); Vs.append(V_of(x))
Ws = np.array(Ws); Vs = np.array(Vs)
viol = np.sum(np.diff(Ws)[Vs[:-1] > lstar] > 0)
print(f"random admissible dosing, 4000 days: W0={Ws[0]:.4f} -> W_final={Ws[-1]:.3e}, V_final={Vs[-1]:.3e}")
print(f"number of day-steps where W increased while V > lstar: {viol}  (must be 0 if the proof is correct)")


# ======================================================================
# ADDED IN THE 2026 REVISION: residual set, x1 floor, decay rate, JSON
# The code above is the authors' script, unchanged apart from the header.
# ======================================================================
import json
import os

print("\n" + "="*70)
print("PART D (added): residual set by successive refinement, x1 floor, decay rate")
print("="*70)


def failing_box(x1_lo, x1_hi, n1, x2_lo, x2_hi, n2):
    """Same exact cell bound as worst_Q_on_grid; returns the bounding box of the cells whose
    bound is >= 0 (None if every cell is negative) and the number of such cells."""
    e1 = np.linspace(x1_lo, x1_hi, n1+1); e2 = np.linspace(x2_lo, x2_hi, n2+1)
    x1a = e1[:-1][:, None]; x1b = e1[1:][:, None]
    x2a = e2[:-1][None, :]; x2b = e2[1:][None, :]
    wa, wb = 1-x1b, 1-x1a
    vtx = eps/(2*P['Om1'])
    f_w = np.maximum(-P['Om1']*wa**2+eps*wa, -P['Om1']*wb**2+eps*wb)
    in_range = (wa <= vtx) & (vtx <= wb)
    f_w = np.where(in_range, np.maximum(f_w, -P['Om1']*vtx**2+eps*vtx), f_w)
    psi_max = psi(x2a)
    bracket_max = P['b1']*wb + c*(psi_max - P['b2']*(1-wb))
    bad = (f_w + x2b*np.maximum(0.0, bracket_max) + x2a*np.minimum(0.0, bracket_max)) >= 0
    if not bad.any():
        return None, 0
    i = np.where(bad.any(axis=1))[0]; j = np.where(bad.any(axis=0))[0]
    return (e1[i[0]], e1[i[-1]+1], e2[j[0]], e2[j[-1]+1]), int(bad.sum())


box = (x1min, 1.0, 0.0, x2bar)
levels = []
for lev in range(14):
    n = 4000 if lev == 0 else 3000
    nb, nbad = failing_box(box[0], box[1], n, box[2], box[3], n)
    if nb is None:
        print(f"level {lev}: no cell with bound >= 0"); break
    shrink = ((nb[1]-nb[0])*(nb[3]-nb[2])) / ((box[1]-box[0])*(box[3]-box[2]))
    levels.append(dict(level=lev, n=n, w_max=1-nb[0], x2_max=nb[3], x1_hi=nb[1], x2_lo=nb[2], n_cells=nbad, area_ratio=shrink))
    print(f"level {lev}: cells with bound >= 0 lie in  w <= {1-nb[0]:.6e}, x2 <= {nb[3]:.6e}  (x1_hi={nb[1]:.6f}, x2_lo={nb[2]:.1e}, {nbad} cells)")
    box = nb
    if shrink > 0.90:
        break
w_res, x2_res = 1-box[0], box[3]
touches_corner = (box[1] == 1.0) and (box[2] == 0.0)
print(f"residual set contained in  w <= {w_res:.6e}  (eps/Om1 = {eps/P['Om1']:.6e}),  x2 <= {x2_res:.6e};  box touches (x1,x2)=(1,0): {touches_corner}")
lstar_box = ((1-w_res) - 1 - np.log(1-w_res)) + c*x2_res          # largest V on the residual box
lo, hi = 1e-9, 1.0
for _ in range(300):
    mid = 0.5*(lo+hi)
    if mid - 1 - np.log(mid) > lstar_box: lo = mid
    else: hi = mid
x1_floor = hi
mu_rate = P['b2']*x1_floor - m
print(f"V level enclosing the residual box  l* = {lstar_box:.6e}")
print(f"x1 floor (x1 - 1 - ln x1 = l*)      = {x1_floor:.6f}")
print(f"decay-rate bound  beta2*x1_floor - m = {mu_rate:.6e} per day")

out = dict(c=c, Umin=Umin, Umax=Umax, x4bar=x4bar, eps=eps, m=m, beta2=P['b2'],
           monotonicity_ok=bool(P['p2']*x4bar <= P['Om2']*P['a2']**2), m_lt_beta2=bool(m < P['b2']),
           eta=eta, V0=V0, W0=W0, x2bar=x2bar, x1min=x1min,
           worst_Q_coarse_4000=float(worst), worst_Q_fine_local_6000=float(worst_f),
           residual_w_max=float(w_res), residual_x2_max=float(x2_res), eps_over_Om1=eps/P['Om1'],
           residual_touches_tumor_free_corner=bool(touches_corner), refinement_levels=levels,
           lstar_box=float(lstar_box), x1_floor=float(x1_floor), mu_rate=float(mu_rate),
           partC_lstar=float(lstar), partC_W_final=float(Ws[-1]), partC_V_final=float(Vs[-1]), partC_violations=int(viol))
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_grid.json")
with open(_p + ".tmp", "w") as _f:
    json.dump(out, _f, indent=1)
os.replace(_p + ".tmp", _p)
print("wrote results_grid.json")
