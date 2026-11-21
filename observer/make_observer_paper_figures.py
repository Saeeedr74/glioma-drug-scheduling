"""Paper-style observer figures from observer.py functions and stored results (fixed seeds)."""
import sys, io, contextlib
sys.path.insert(0, sys.argv[1]); OUT = sys.argv[2]
import numpy as np, pandas as pd, observer as ob
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.1, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "legend.frameon": False, "pdf.fonttype": 42})
C_EKF, C_LSD, C_TRUE = "#E69F00", "#CC79A7", "black"
W = 190/25.4
t = np.arange(ob.N_DAYS + 1)
def tag(ax, s): ax.text(0.025, 0.97, s, transform=ax.transAxes, va="top", ha="left", fontsize=8)

with contextlib.redirect_stdout(io.StringIO()):
    x_pre, sched, _ = ob.step1()
U = sched["mu"]; x4h = ob.x4_known(U, x_pre[3])
MET = pd.read_csv(ob.RES_DIR/"metrics_all.csv")
TR = pd.read_csv(ob.TS_DIR/"truth_mu.csv.gz")
def truth(plant): return TR[TR.plant == plant].sort_values("day")[["x1","x2","x3","x4"]].to_numpy()
def est(obs, plant, d, mode="M1"):
    r = MET[(MET.observer==obs)&(MET.controller=="mu")&(MET.plant==plant)&np.isclose(MET.delta,d)&(MET["mode"]==mode)&(MET.seed==-1)].iloc[0]
    E = pd.read_csv(ob.TS_DIR/f"est_{obs}_mu_{mode}.csv.gz"); E = E[E.run_id==r.run_id].sort_values("day")
    return E.x1_hat.to_numpy(), E.x2_hat.to_numpy()

# ---------- Fig. 1: M1, nominal, mu schedule
X = truth("nominal"); assert np.allclose(X, ob.simulate_plant(U, "nominal", x_pre), rtol=1e-8)
fig, ax = plt.subplots(1, 3, figsize=(W, 2.45))
ax[0].semilogy(t, X[:,1], color=C_TRUE, lw=1.2, label=r"True $x_2$")
for obs, c, lab in (("ekf", C_EKF, "EKF"), ("lsd", C_LSD, "Luenberger")):
    x1h, x2h = est(obs, "nominal", 0.2)
    ax[0].semilogy(t, np.maximum(x2h,1e-12), color=c, lw=1.0, ls="--", label=rf"$\hat x_2$, {lab}")
ax[0].set_ylabel(r"Glioma cells $x_2$, $\hat x_2$"); ax[0].legend(loc="lower left")
ax[1].plot(t, X[:,0], color=C_TRUE, lw=1.2, label=r"True $x_1$")
for obs, c, lab in (("ekf", C_EKF, "EKF"), ("lsd", C_LSD, "Luenberger")):
    for d, ls in ((0.2, "-"), (-0.2, "--")):
        x1h, _ = est(obs, "nominal", d)
        ax[1].plot(t, x1h, color=c, lw=1.0, ls=ls, label=rf"{lab}, {d*100:+.0f}%")
ax[1].set_ylabel(r"Glial cells $x_1$, $\hat x_1$"); ax[1].legend(loc="lower right", fontsize=6.5)
for obs, c, lab in (("ekf", C_EKF, "EKF"), ("lsd", C_LSD, "Luenberger")):
    for d, ls in ((0.2, "-"), (-0.2, "--")):
        x1h, _ = est(obs, "nominal", d)
        ax[2].semilogy(t, np.maximum(np.abs(x1h - X[:,0])/X[:,0], 1e-10), color=c, lw=1.0, ls=ls)
ax[2].axhline(ob.CONV_THR, color="0.3", lw=0.7, ls=":"); ax[2].set_ylim(1e-8, 2)
ax[2].set_ylabel(r"$e_1=|\hat x_1-x_1|/x_1$")
for a, s in zip(ax, "abc"):
    a.set_xlim(0, ob.N_DAYS); a.set_xlabel("Time since treatment start (days)"); tag(a, f"({s})")
fig.tight_layout(w_pad=0.8); fig.savefig(f"{OUT}/fig_observer_M1.pdf", bbox_inches="tight"); plt.close(fig)

# ---------- Fig. 2: 20 seeds, M2 / M3, nominal, mu, +20 %
noise = ob.noise_table(); chk = []
modes = [("M2", 1, "Daily"), ("M3_7", 7, "Every 7 days"), ("M3_28", 28, "Every 28 days")]
fig, ax = plt.subplots(1, 3, figsize=(W, 2.35), sharey=True)
for a, (mode, per, lab) in zip(ax, modes):
    for obs, c, olab in (("ekf", C_EKF, "EKF"), ("lsd", C_LSD, "Luenberger")):
        E = []
        for s in range(ob.N_SEEDS):
            y = ob.measurements(X[:,1], per, noise[s])
            xh = ob.run_ekf(X, x4h, U, y, per, 0.2, True)[0] if obs=="ekf" else ob.run_lsd(X, x4h, U, y, per, 0.2)[0]
            m, e1, _ = ob.run_metrics(X, xh)
            r = MET[(MET.observer==obs)&(MET.controller=="mu")&(MET.plant=="nominal")&np.isclose(MET.delta,0.2)&(MET["mode"]==mode)&(MET.seed==s)].iloc[0]
            chk.append(abs(m["e1_final"]-r.e1_final)/r.e1_final)
            E.append(e1)
        E = np.array(E); assert np.all(np.isfinite(E))
        lo, med, hi = np.percentile(E, [5, 50, 95], axis=0)
        a.fill_between(t, np.maximum(lo,1e-9), hi, color=c, alpha=0.25, lw=0, label=f"{olab}, 5–95%")
        a.semilogy(t, med, color=c, lw=1.1, label=f"{olab}, median")
    a.axhline(ob.CONV_THR, color="0.3", lw=0.7, ls=":")
    a.set_xlim(0, ob.N_DAYS); a.set_ylim(1e-5, 2); a.set_xlabel("Time since treatment start (days)")
    tag(a, f"({'abc'[modes.index((mode, per, lab))]}) {lab}, $\\sigma=0.1$")
ax[0].set_ylabel(r"$e_1=|\hat x_1-x_1|/x_1$"); ax[0].legend(loc="lower left", fontsize=6.5, ncol=1)
fig.tight_layout(w_pad=0.6); fig.savefig(f"{OUT}/fig_observer_noise20.pdf", bbox_inches="tight"); plt.close(fig)
print("max rel diff of e1(2000), recomputed vs metrics_all:", max(chk))

# ---------- Fig. 3: mismatch, G5, mu, +20 %
TRp = pd.read_csv(ob.TS_DIR/"truth_pid.csv.gz"); XG = TRp[TRp.plant=="G5"].sort_values("day")[["x1","x2","x3","x4"]].to_numpy()
def est_pid(obs, plant, d):
    r = MET[(MET.observer==obs)&(MET.controller=="pid")&(MET.plant==plant)&np.isclose(MET.delta,d)&(MET["mode"]=="M1")].iloc[0]
    E = pd.read_csv(ob.TS_DIR/f"est_{obs}_pid_M1.csv.gz"); E = E[E.run_id==r.run_id].sort_values("day")
    return E.x1_hat.to_numpy(), E.x2_hat.to_numpy()
fig, ax = plt.subplots(1, 2, figsize=(W, 2.45))
ax[0].plot(t, XG[:,0], color=C_TRUE, lw=1.2, label=r"True $x_1$ (plant G5)")
for obs, c, lab in (("ekf", C_EKF, "EKF"), ("lsd", C_LSD, "Luenberger")):
    x1h, x2h = est_pid(obs, "G5", 0.2)
    ax[0].plot(t, x1h, color=c, lw=1.0, label=rf"$\hat x_1$, {lab}")
    ax[1].semilogy(t, np.maximum(np.abs(x1h-XG[:,0])/XG[:,0],1e-10), color=c, lw=1.0, label=rf"$e_1$, {lab}")
    ax[1].semilogy(t, np.maximum(np.abs(x2h-XG[:,1])/np.maximum(XG[:,1],1e-12),1e-10), color=c, lw=1.0, ls="--", label=rf"$e_2$, {lab}")
ax[0].axhline(0, color="0.3", lw=0.7, ls=":"); ax[0].set_ylim(-3, 1.5)
ax[0].set_ylabel(r"Glial cells $x_1$, $\hat x_1$"); ax[0].legend(loc="lower right", fontsize=6.5)
ax[1].axhline(ob.CONV_THR, color="0.3", lw=0.7, ls=":"); ax[1].set_ylim(1e-6, 30)
ax[1].set_ylabel("Relative error"); ax[1].legend(loc="lower right", fontsize=6.5, ncol=2)
for a, s in zip(ax, "ab"):
    a.set_xlim(0, ob.N_DAYS); a.set_xlabel("Time since treatment start (days)"); tag(a, f"({s})")
fig.tight_layout(w_pad=1.0); fig.savefig(f"{OUT}/fig_observer_mismatch.pdf", bbox_inches="tight"); plt.close(fig)
