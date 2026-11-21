"""
Tracking-error block of the stability section. RE-IMPLEMENTED in the 2026 revision from
stability_section.tex and NOTES.md: the original block was not in the two scripts.

With z = int_0^t e dtau, the sliding variable is  s = z''' + lam_d z'' + lam_p z' + lam_i z,
so  e = z'  is the output of  G(sigma) = sigma / (sigma^3 + lam_d sigma^2 + lam_p sigma + lam_i)
driven by s, plus the free response from z''(0) = de/dt(0) (z(0) = 0, e(0) = 0), which is
g(t) * de/dt(0) with g the impulse response of G. Hence, on 0 <= t <= T,
    |e(t)| <= L1 * sup|s| + g_max * |de/dt(0)|,   L1 = int_0^T |g|,  g_max = max_[0,T] |g|.
(Definitions checked against the two numbers of NOTES.md for the OLD parameters: 0.749 and 0.296.)
sup|s|, de/dt(0) and the observed error come from the final SMC run (data/, made by unify.py).
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "results", "figure_data")
T_REF = 2000.0


def gains(lam_d, lam_p, lam_i, T=T_REF):
    roots = np.roots([1.0, lam_d, lam_p, lam_i])
    res = roots/(3*roots**2 + 2*lam_d*roots + lam_p)        # residues of sigma/p(sigma)
    # dense grid: fine near 0 (fast modes), then uniform
    t = np.unique(np.concatenate([np.linspace(0, 50, 500001), np.linspace(50, T, 1950001)]))
    g = np.real(sum(r*np.exp(s*t) for r, s in zip(res, roots)))
    L1 = float(np.trapezoid(np.abs(g), t))
    i = int(np.argmax(np.abs(g)))
    return roots, res, L1, float(abs(g[i])), float(t[i])


# check of the definitions with the old parameters (NOTES.md: 0.749 and 0.296)
_, _, L1_old, gmax_old, _ = gains(2.5, 1.5, 1e-4)

# final v2 parameters
LAMBDA_P, LAMBDA_D, LAMBDA_I = 1.8813613583755686, 9.619198702305367, 2.9470610386036095e-05
roots, res, L1, gmax, t_gmax = gains(LAMBDA_D, LAMBDA_P, LAMBDA_I)
hurwitz = bool(np.all(roots.real < 0)) and (LAMBDA_D*LAMBDA_P > LAMBDA_I)

sb = pd.read_csv(os.path.join(DATA, "smc_sliding_band.csv"))
nd = pd.read_csv(os.path.join(DATA, "nominal_discrete.csv"))
smc = nd[nd.controller == "SMC"].sort_values("time_day")
e = (smc["x2"] - smc["x2_reference"]).to_numpy()
sup_s = float(np.max(np.abs(sb["s"])))
day_sup = int(np.argmax(np.abs(sb["s"].to_numpy())))
# de/dt(0) exactly as the controller computes it: nominal dx2/dt at x_pre (x4 = 0) minus dr/dt(0)
x1, x2 = float(smc["x1"].iloc[0]), float(smc["x2"].iloc[0])
dx2 = 0.012*x2*(1-x2) - 1.8e-3*x1*x2
a = 7.5
S0, S1 = 0.5*(1-np.tanh(0.0)), 0.5*(1-np.tanh(a))
drdt0 = x2*((-0.5*a)/(S0-S1))/T_REF
de0 = dx2 - drdt0
bound = L1*sup_s + gmax*abs(de0)
obs = float(np.max(np.abs(e)))
out = dict(lam_p=LAMBDA_P, lam_d=LAMBDA_D, lam_i=LAMBDA_I, roots=[complex(r).real for r in roots],
           roots_imag_max=float(np.max(np.abs(roots.imag))), hurwitz=hurwitz,
           time_constants_days=[float(-1/r.real) for r in roots],
           L1_gain_0_2000=L1, free_response_gain=gmax, t_of_free_response_peak=t_gmax,
           sup_abs_s=sup_s, day_of_sup_abs_s=day_sup, share_days_abs_s_le_delta=float(np.mean(np.abs(sb["s"]) <= sb["delta"])),
           de_dt_0=float(de0), bound_abs_e=float(bound), observed_max_abs_e=obs, day_observed_max=int(np.argmax(np.abs(e))),
           bound_holds=bool(obs <= bound), observed_max_abs_e_over_x2pre=obs/x2,
           check_old_params_L1=L1_old, check_old_params_gmax=gmax_old)
p = os.path.join(HERE, "results_error_dynamics.json")
with open(p + ".tmp", "w") as f:
    json.dump(out, f, indent=1)
os.replace(p + ".tmp", p)
print("definition check, old parameters (NOTES.md 0.749, 0.296): L1 = %.3f, g_max = %.3f" % (L1_old, gmax_old))
print("roots (1/day): " + ", ".join("%.6g" % r.real for r in roots) + "  (max |imag| %.1e); Hurwitz: %s" % (out["roots_imag_max"], hurwitz))
print("L1 gain on [0,2000] = %.6f ; free-response gain = %.6f (peak at t = %.3f d)" % (L1, gmax, t_gmax))
print("sup|s| = %.6e (day %d); |s| <= delta on %.1f%% of days; de/dt(0) = %.6e" % (sup_s, day_sup, 100*out["share_days_abs_s_le_delta"], de0))
print("bound |e| <= %.6e ; observed max |e| = %.6e (day %d) ; bound holds: %s" % (bound, obs, out["day_observed_max"], out["bound_holds"]))
