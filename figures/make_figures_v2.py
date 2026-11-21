"""Revised and new figures of the paper (version 2): every output file ends in _v2.

  fig_baseline_v2                    maximum-dose baseline (data re-run with the stop rule x2 < 1e-4)
  fig_protocol_comparison_v2         SMC vs the 5/28 protocol at 200 and 400, linear axes
  fig_uncertainty_v2                 replay test on G1..G5, new palette, thinner lines, distinct line styles
  fig_observer_M1_stacked_v2         observers, scenario M1, red and blue
  fig_observer_noise20_stacked_v2    observers, noisy measurements (20 seeds), red and blue
  fig_observer_mismatch_stacked_v2   observers on plant G5, red and blue
  fig_lyapunov_W_dW_v2               W and dW/dt on the nominal model for PID, SMC, mu
  fig_lyapunov_G5_SMC_v2             W and dW/dt for the SMC schedule replayed on plant G5, nominal for reference

Run from the repository root:  python figures/make_figures_v2.py
Reads results/figure_data/ and observer/results/timeseries/ (paths relative to the repository root, or --root);
writes vector PDF to figures/main/, PNG previews to figures/preview_v2/, and, when the directory exists, the PDFs
also to ../01_paper_current/figures/main/ (the manuscript folder; skipped in a published repository).

The style block (rcParams, controller colors, reference and threshold styles, mm-based layout helpers, legend
placement) is copied verbatim from figures/make_figures.py so that all figures of the paper look alike.
Nothing is computed that is not in the stored data, except e1 = |x1_hat - x1|/x1 for the 20-seed noise figure
(from the stored estimates; checked against e1_final of metrics_all.csv) and the percentiles over seeds."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.transforms as mtrans  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.join(HERE, ".."), help="repository root (default: parent of figures/)")
ap.add_argument("--data", default=None, help="figure_data directory (default: <root>/results/figure_data)")
ap.add_argument("--also-copy-to", default=None, help="second PDF folder (default: <root>/../01_paper_current/figures/main if it exists)")
ARGS = ap.parse_args()
ROOT = os.path.abspath(ARGS.root)
D = ARGS.data or os.path.join(ROOT, "results", "figure_data")
TS = os.path.join(ROOT, "observer", "results", "timeseries")
MET_CSV = os.path.join(ROOT, "observer", "results", "metrics_all.csv")
OUT_MAIN = os.path.join(ROOT, "figures", "main")
OUT_PREV = os.path.join(ROOT, "figures", "preview_v2")
OUT_PAPER = ARGS.also_copy_to or os.path.join(ROOT, "..", "01_paper_current", "figures", "main")
if not os.path.isdir(OUT_PAPER):
    OUT_PAPER = None

# ======================= style: verbatim from figures/make_figures.py =======================
MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
    "axes.unicode_minus": True, "legend.handlelength": 2.8, "legend.columnspacing": 1.4, "legend.handletextpad": 0.5,
})

CTRL = {"PID": dict(color="#0072B2", ls="-", lw=1.0, label="PID"),
        "SMC": dict(color="#D55E00", ls="-.", lw=1.0, label="SMC"),
        "mu": dict(color="#009E73", ls="-", lw=1.2, label=r"$\mu$-synthesis")}
REF = dict(color="black", ls="--", lw=0.9, label=r"Reference $r(t)$")
THR = dict(color="0.45", ls=":", lw=0.9, label=r"Threshold $10^{-4}$")
TLAB = "Time since treatment start (days)"
STATE = {"x1": dict(color="black", ls="-", label=r"$x_1$ glial cells"), "x2": dict(color="0.45", ls="--", label=r"$x_2$ glioma cells"),
         "x3": dict(color="#56B4E9", ls="-.", lw=1.1, label=r"$x_3$ neurons")}
ULAB = r"$u$ (mg m$^{-2}$ day$^{-1}$)"
# ===========================================================================================


def kw(st, **extra):
    d = {k: v for k, v in st.items()}
    d.update(extra)
    return d


def grid(w, h, nr, nc, top, bottom=9.0, left=13.0, right=3.0, hgap=5.0, wgap=15.0):
    fig = plt.figure(figsize=(w * MM, h * MM))
    aw = (w - left - right - (nc - 1) * wgap) / nc
    ah = (h - top - bottom - (nr - 1) * hgap) / nr
    axes = []
    for i in range(nr):
        for j in range(nc):
            x0 = left + j * (aw + wgap)
            y0 = h - top - (i + 1) * ah - i * hgap
            axes.append(fig.add_axes([x0 / w, y0 / h, aw / w, ah / h]))
    fig._mm = (w, h, top)
    return fig, axes


def plabel(ax, s):
    off = mtrans.ScaledTranslation(0, 1.5 / 72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, "(%s)" % s, transform=ax.transAxes + off, ha="left", va="bottom", fontsize=8)


def time_axis(ax, label=True, step=250):
    ax.set_xlim(0, 2000)
    ax.set_xticks(np.arange(0, 2001, step))
    if label:
        ax.set_xlabel(TLAB)
    else:
        ax.tick_params(labelbottom=False)


def logy(ax, lo, hi):
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)


def pos(y):
    y = np.asarray(y, float).copy()
    y[y <= 0] = np.nan
    return y


def h_line(st):
    return Line2D([], [], **st)


def fig_legend(fig, handles, ncol, gap_mm=4.2):
    w, h, top = fig._mm
    y = (h - top + gap_mm) / h
    fig._leg = (handles, ncol, gap_mm)
    return fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False,
                      borderaxespad=0.0)


def fit_legend(fig):
    """Reduce the legend columns until it fits the width, then set the figure height so that the legend
    ends 1 mm below the top edge (axes keep their size and position in mm)."""
    if not fig.legends:
        return
    handles, ncol, gap = fig._leg
    r = fig.canvas.get_renderer()
    while True:
        fig.canvas.draw()
        lb = fig.legends[0].get_window_extent(r)
        if (lb.x0 >= 1 and lb.x1 <= fig.bbox.x1 - 1) or ncol == 1:
            break
        fig.legends[0].remove()
        ncol -= 1
        fig_legend(fig, handles, ncol, gap)
    w, h, top = fig._mm
    excess_mm = (lb.y1 - fig.bbox.y1) / fig.dpi * 25.4 + 1.0
    h2 = h + excess_mm
    for ax in fig.axes:
        p = ax.get_position()
        ax.set_position([p.x0, p.y0 * h / h2, p.width, p.height * h / h2])
    fig.set_size_inches(w * MM, h2 * MM)
    fig._mm = (w, h2, top + excess_mm)
    fig.legends[0].remove()
    fig_legend(fig, handles, ncol, gap)


def step(ax, t, u, **st):
    u = np.asarray(u, float)[:2000]
    ax.step(np.arange(2001), np.append(u, u[-1]), where="post", **st)


def rd(name, **k):
    p = os.path.join(D, name + ".csv")
    if not os.path.exists(p):          # larger files are stored gzipped
        p += ".gz"
    return pd.read_csv(p, **k)


def run(df, col, val):
    return df[df[col] == val].sort_values("time_day")


def save(fig, name):
    """Vector PDF into figures/main (and the manuscript folder when present), PNG preview into figures/preview_v2."""
    fit_legend(fig)
    dirs = [OUT_MAIN] + ([OUT_PAPER] if OUT_PAPER else [])
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, name + ".pdf")
        fig.savefig(p + ".tmp.pdf")
        os.replace(p + ".tmp.pdf", p)
    os.makedirs(OUT_PREV, exist_ok=True)
    fig.savefig(os.path.join(OUT_PREV, name + ".png"), dpi=180)
    plt.close(fig)
    print("wrote", name)


# ======================= palettes of the revised figures =======================
# Replay-test groups: no hue shared with the controller colors (blue, vermillion, green) used by the bars of
# panel (d); each group also has its own line style and marker, and neighbouring curves (G5, G4, G3, G1 from top
# to bottom in the figure) alternate between dark and light so that they also differ in grayscale print.
GRP2 = {"nominal": dict(color="black", ls="-", lw=0.9, label="Nominal plant"),
        "G1": dict(color="#E69F00", ls="-", lw=0.8, label="G1"),
        "G2": dict(color="#56B4E9", ls="-", lw=0.8, label="G2"),
        "G3": dict(color="#332288", ls="-", lw=0.8, label="G3"),
        "G4": dict(color="#CC79A7", ls="-", lw=0.8, label="G4"),
        "G5": dict(color="#999933", ls="-", lw=0.8, label="G5")}
MK = dict(ms=2.6, markevery=(0, 1), mew=0.0)
# Observers: blue and red (EKF blue, Luenberger red); luminance differs so that they also separate in grayscale.
BLUE, RED = "#3C7DD9", "#B71C1C"
OBS2 = {"ekf": dict(color=BLUE, label="EKF"), "lsd": dict(color=RED, label="Luenberger")}


def grp_plot(ax, r, g, y):
    """Curve of group g with sparse markers, offset differently for each group."""
    st = dict(GRP2[g])
    if "marker" in st:
        k = list(GRP2).index(g)
        st.update(markevery=(60 * k, 300), ms=2.6, mew=0.0)
    ax.plot(r.time_day, y, **st)


# ======================= figures =======================
def f_baseline():
    B = rd("baseline_max_dose").sort_values("time_day")
    fig, ax = grid(140, 105, 3, 1, top=10, hgap=4.0)
    st = STATE
    for k, s in st.items():
        ax[0].plot(B.time_day, B[k], **s)
    ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("Normalized density")
    ax[1].plot(B.time_day, B.x4, color="0.3", ls="-", label=r"$x_4$")
    ax[1].set_ylabel(r"$x_4$ (mg m$^{-2}$)")
    step(ax[2], B.time_day, B.u, color="0.3", ls="-", lw=0.9)
    ax[2].set_ylabel(ULAB)
    ax[2].set_ylim(-10, 215)
    for i, a in enumerate(ax):
        time_axis(a, label=(i == 2))
        plabel(a, "abc"[i])
        a.yaxis.set_label_coords(-0.075, 0.5)
    fig_legend(fig, [h_line(s) for s in st.values()], 3, gap_mm=4.0)
    save(fig, "fig_baseline_v2")


def f_protocol_4panel():
    SP = rd("standard_protocol")
    ND = rd("nominal_discrete")
    REFS = rd("reference")
    smc = run(ND, "controller", "SMC")
    s2 = dict(color="#CC79A7", ls="-", lw=0.8, label=r"5/28 schedule, 200 mg m$^{-2}$ day$^{-1}$")
    s4 = dict(color="#56B4E9", ls="-", lw=0.8, label=r"5/28 schedule, 400 mg m$^{-2}$ day$^{-1}$")
    sm = kw(CTRL["SMC"], ls="-", lw=0.9, label="SMC")
    fig, ax = grid(190, 112, 2, 2, top=11, hgap=12, wgap=17, left=14)
    # draw the SMC last so that it stays visible on top of the reference and of the 400 protocol
    for r, st in ((run(SP, "run", "standard_5_28_dose200"), s2), (run(SP, "run", "standard_5_28_dose400"), s4), (smc, sm)):
        ax[0].plot(r.time_day, r.x2, **st)
        ax[1].plot(r.time_day, r.x1, **st)
        ax[2].plot(r.time_day, r.x3, **st)
        u = np.asarray(r.u, float)[:2000]
        ax[3].plot(np.arange(2001), np.concatenate([[0], np.cumsum(u)]) / 1e3, **st)
    ax[0].plot(REFS.time_day, REFS.x2_reference, **REF)
    ax[0].set_ylim(0, 0.65)
    ax[0].set_ylabel(r"$x_2$ glioma cells (normalized)")
    ax[1].set_ylabel(r"$x_1$ glial cells (normalized)")
    ax[2].set_ylabel(r"$x_3$ neurons (normalized)")
    ax[3].set_ylabel(r"Cumulative dose (10$^3$ mg m$^{-2}$)")
    for i, a in enumerate(ax):
        time_axis(a)
        plabel(a, "abcd"[i])
    fig_legend(fig, [h_line(sm), h_line(s2), h_line(s4), h_line(REF)], 4)
    save(fig, "fig_protocol_comparison_4panel_v2")


def f_protocol():
    """Single panel, same content as the old fig_Comparison_SMC: x2 (linear) for SMC, 5/28 at 200 and 400, and r(t)."""
    SP = rd("standard_protocol")
    ND = rd("nominal_discrete")
    REFS = rd("reference")
    smc = run(ND, "controller", "SMC")
    s2 = dict(color="#CC79A7", ls="-", lw=0.8, label=r"5/28 schedule, 200 mg m$^{-2}$ day$^{-1}$")
    s4 = dict(color="#56B4E9", ls="-", lw=0.8, label=r"5/28 schedule, 400 mg m$^{-2}$ day$^{-1}$")
    sm = kw(CTRL["SMC"], ls="-", lw=0.9, label="SMC")
    fig, (ax,) = grid(140, 72, 1, 1, top=10, left=15)
    ax.plot(REFS.time_day, REFS.x2_reference, **REF)
    for r, st in ((run(SP, "run", "standard_5_28_dose200"), s2), (run(SP, "run", "standard_5_28_dose400"), s4), (smc, sm)):
        ax.plot(r.time_day, r.x2, **st)       # SMC last: drawn over the reference it tracks
    ax.set_ylim(0, 0.65)
    ax.set_ylabel(r"$x_2$ glioma cells (normalized)")
    time_axis(ax)
    fig_legend(fig, [h_line(sm), h_line(s2), h_line(s4), h_line(REF)], 2, gap_mm=4.0)
    save(fig, "fig_protocol_comparison_v2")


def f_uncertainty():
    E = rd("e1_uncertainty")
    REFS = rd("reference")
    fig, ax = grid(190, 140, 2, 2, top=15, hgap=12, wgap=17, left=14)
    groups = ["nominal", "G1", "G2", "G3", "G4", "G5"]
    for i, c in enumerate(("PID", "SMC", "mu")):
        for g in groups[::-1]:           # nominal drawn last (on top)
            r = E[(E.controller == c) & (E.group == g)].sort_values("time_day")
            grp_plot(ax[i], r, g, pos(r.x2))
        ax[i].plot(REFS.time_day, REFS.x2_reference, **kw(REF, lw=0.8))
        ax[i].axhline(1e-4, **kw(THR, lw=0.8))
        logy(ax[i], 1e-7, 1)
        ax[i].set_ylabel(("PID", "SMC", r"$\mu$-synthesis")[i] + r" schedule: $x_2$")
        time_axis(ax[i])
    xT = E[E.time_day == 2000]
    wbar = 0.26
    hatch = {"PID": "", "SMC": "////", "mu": "...."}
    for j, c in enumerate(("PID", "SMC", "mu")):
        v = [float(xT[(xT.controller == c) & (xT.group == g)].x2.iloc[0]) for g in groups]
        ax[3].bar(np.arange(6) + (j - 1) * wbar, v, width=wbar, color=CTRL[c]["color"], edgecolor="black", linewidth=0.4,
                  hatch=hatch[c], label=CTRL[c]["label"])
    ax[3].set_yscale("log")
    ax[3].set_ylim(1e-6, 1e-2)
    ax[3].axhline(1e-4, **THR)
    ax[3].set_xticks(np.arange(6))
    ax[3].set_xticklabels(["Nominal", "G1", "G2", "G3", "G4", "G5"])
    ax[3].set_xlim(-0.6, 5.6)
    ax[3].set_ylabel(r"$x_2(T)$, $T = 2000$ days")
    ax[3].set_xlabel("Plant")
    ax[3].tick_params(axis="x", which="both", top=False)
    for i, a in enumerate(ax):
        plabel(a, "abcd"[i])
    gh = []
    for g in groups:
        st = {k: v for k, v in GRP2[g].items()}
        gh.append(Line2D([], [], **st))
    hs = gh + [h_line(kw(REF, lw=0.8)), h_line(kw(THR, lw=0.8))] + \
         [Patch(facecolor=CTRL[c]["color"], edgecolor="black", linewidth=0.4, hatch=hatch[c], label=CTRL[c]["label"]) for c in CTRL]
    fig_legend(fig, hs, 6)
    save(fig, "fig_uncertainty_v2")


# ---------------- observers ----------------
def tag(ax, s, y=0.97):
    ax.text(0.015, y, s, transform=ax.transAxes, va="top", ha="left", fontsize=8)


def obs_axes(h):
    fig, ax = grid(140, h, 2 if h < 120 else 3, 1, top=2.0, bottom=10.0, left=13.0, right=2.0, hgap=4.0)
    return fig, ax


def f_obs_M1():
    O = rd("observer_M1_nominal_timeseries")
    O["delta"] = O["delta"].astype(float)
    fig, ax = obs_axes(98)
    sel = lambda ob, d: O[(O.controller == "mu") & (O.observer == ob) & (O.delta == d)].sort_values("time_day")
    r0 = sel("ekf", 0.2)
    ax[0].plot(r0.time_day, r0.x1, color="black", ls="-", lw=1.2, label=r"True $x_1$")
    for ob in ("ekf", "lsd"):
        for d, ls in ((0.2, "-"), (-0.2, "--")):
            r = sel(ob, d)
            lab = "%s, %+.0f%%" % (OBS2[ob]["label"], d * 100)
            ax[0].plot(r.time_day, r.x1_hat, color=OBS2[ob]["color"], lw=1.0, ls=ls, label=lab)
            e1 = np.maximum(np.abs(r.x1_hat.values - r.x1.values) / r.x1.values, 1e-10)
            ax[1].plot(r.time_day, e1, color=OBS2[ob]["color"], lw=1.0, ls=ls)
    ax[0].set_ylabel(r"Glial cells $x_1$, $\hat{x}_1$")
    ax[0].legend(loc="lower right", ncol=2, frameon=False, fontsize=7, handlelength=2.4)
    ax[1].axhline(0.01, color="0.3", lw=0.7, ls=":")
    logy(ax[1], 1e-8, 2)
    ax[1].set_ylabel(r"$e_1 = |\hat{x}_1 - x_1|/x_1$")
    for i, a in enumerate(ax):
        a.set_xlim(0, 2000)
        a.set_xticks(np.arange(0, 2001, 250))
        a.tick_params(labelbottom=(i == 1))
        tag(a, "(%s)" % "ab"[i])
        a.yaxis.set_label_coords(-0.075, 0.5)
    ax[1].set_xlabel(TLAB)
    save(fig, "fig_observer_M1_stacked_v2")


def f_obs_noise():
    """Scenarios M2 / M3, nominal plant, mu schedule, +20 %: the 20 noise sequences are re-run with the observer
    functions of observer/observer.py, imported as a module (nothing is written; this is what
    observer/make_observer_paper_figures.py does). The result is checked against e1_final of metrics_all.csv."""
    import sys
    sys.dont_write_bytecode = True
    sys.path.insert(0, os.path.join(ROOT, "observer"))
    import observer as ob
    MET = pd.read_csv(MET_CSV)
    TR = pd.read_csv(os.path.join(TS, "truth_mu.csv.gz"))
    X = TR[TR.plant == "nominal"].sort_values("day")[["x1", "x2", "x3", "x4"]].to_numpy()
    U = rd("schedules")["u_mu"].to_numpy()[:2000]
    x4h = ob.x4_known(U, 0.0)
    noise = ob.noise_table()
    fig, ax = obs_axes(136)
    modes = [("M2", 1, "Daily"), ("M3_7", 7, "Every 7 days"), ("M3_28", 28, "Every 28 days")]
    chk = 0.0
    for k, (mode, per, lab) in enumerate(modes):
        for obn in ("ekf", "lsd"):
            E = []
            for s in range(ob.N_SEEDS):
                y = ob.measurements(X[:, 1], per, noise[s])
                xh = ob.run_ekf(X, x4h, U, y, per, 0.2, True)[0] if obn == "ekf" else ob.run_lsd(X, x4h, U, y, per, 0.2)[0]
                e1 = np.abs(xh[:, 0] - X[:, 0]) / X[:, 0]
                r = MET[(MET.observer == obn) & (MET.controller == "mu") & (MET.plant == "nominal") & np.isclose(MET.delta, 0.2)
                        & (MET["mode"] == mode) & (MET.seed == s)].iloc[0]
                chk = max(chk, abs(e1[-1] - r.e1_final) / r.e1_final)
                E.append(e1)
            E = np.array(E)
            assert np.all(np.isfinite(E))
            lo, med, hi = np.percentile(E, [5, 50, 95], axis=0)
            t = np.arange(E.shape[1])
            c = OBS2[obn]["color"]
            ax[k].fill_between(t, np.maximum(lo, 1e-9), hi, color=c, alpha=0.25, lw=0, label="%s, 5-95%%" % OBS2[obn]["label"])
            ax[k].plot(t, med, color=c, lw=1.1, label="%s, median" % OBS2[obn]["label"])
        ax[k].axhline(0.01, color="0.3", lw=0.7, ls=":")
        logy(ax[k], 1e-5, 2)
        ax[k].set_xlim(0, 2000)
        ax[k].set_xticks(np.arange(0, 2001, 250))
        ax[k].tick_params(labelbottom=(k == 2))
        ax[k].set_ylabel(r"$e_1$")
        tag(ax[k], r"(%s) %s, $\sigma=0.1$" % ("abc"[k], lab))
    ax[2].set_xlabel(TLAB)
    h, l = ax[0].get_legend_handles_labels()
    order = [0, 2, 1, 3]    # EKF band, Luenberger band | EKF median, Luenberger median
    leg = ax[0].legend([h[i] for i in order], [l[i] for i in order], loc="lower left", ncol=2, frameon=True, fontsize=7,
                       columnspacing=1.2, facecolor="white", framealpha=0.9, edgecolor="none", borderpad=0.3)
    print("  noise figure: max relative difference of e1(2000), recomputed vs metrics_all.csv: %.2e" % chk)
    save(fig, "fig_observer_noise20_stacked_v2")


def f_obs_mismatch():
    G = rd("observer_M1_groups_timeseries")
    G["delta"] = G["delta"].astype(float)
    fig, ax = obs_axes(98)
    sel = lambda ob: G[(G.controller == "pid") & (G.plant == "G5") & (G.observer == ob) & (G.delta == 0.2)].sort_values("time_day")
    r0 = sel("ekf")
    ax[0].plot(r0.time_day, r0.x1, color="black", lw=1.2, label=r"True $x_1$ (plant G5)")
    for ob in ("ekf", "lsd"):
        r = sel(ob)
        c = OBS2[ob]["color"]
        ax[0].plot(r.time_day, r.x1_hat, color=c, lw=1.0, label=r"$\hat{x}_1$, %s" % OBS2[ob]["label"])
        x1, x2 = r.x1.values, r.x2.values
        e1 = np.maximum(np.abs(r.x1_hat.values - x1) / x1, 1e-10)
        e2 = np.maximum(np.abs(r.x2_hat.values - x2) / np.maximum(x2, 1e-12), 1e-10)
        ax[1].plot(r.time_day, e1, color=c, lw=1.0, label=r"$e_1$, %s" % OBS2[ob]["label"])
        ax[1].plot(r.time_day, e2, color=c, lw=1.0, ls="--", label=r"$e_2$, %s" % OBS2[ob]["label"])
    ax[0].axhline(0, color="0.3", lw=0.7, ls=":")
    ax[0].set_ylim(-3, 1.5)
    ax[0].set_ylabel(r"Glial cells $x_1$, $\hat{x}_1$")
    ax[0].legend(loc="lower right", ncol=3, frameon=False, fontsize=7, columnspacing=1.0)
    ax[1].axhline(0.01, color="0.3", lw=0.7, ls=":")
    logy(ax[1], 1e-6, 30)
    ax[1].set_ylabel("Relative error")
    h, l = ax[1].get_legend_handles_labels()
    order = [0, 1, 2, 3]
    ax[1].legend([h[i] for i in order], [l[i] for i in order], loc="lower right", ncol=2, frameon=False, fontsize=7,
                 columnspacing=1.0)
    for i, a in enumerate(ax):
        a.set_xlim(0, 2000)
        a.set_xticks(np.arange(0, 2001, 250))
        a.tick_params(labelbottom=(i == 1))
        tag(a, "(%s)" % "ab"[i])
        a.yaxis.set_label_coords(-0.075, 0.5)
    ax[1].set_xlabel(TLAB)
    save(fig, "fig_observer_mismatch_stacked_v2")


# ---------------- Lyapunov ----------------
def f_lyap_nominal():
    L = rd("lyapunov_W_timeseries")
    E = L[L.protocol == "E1_replay"]
    fig, ax = grid(140, 88, 2, 1, top=10, hgap=4.5, left=15)
    for c in ("PID", "SMC", "mu"):
        r = E[(E.controller == c) & (E.group == "nominal")].sort_values("time_day")
        assert (r.Wdot.values < 0).all(), "dW/dt >= 0 on a nominal run: a log axis of -dW/dt would hide it"
        ax[0].plot(r.time_day, r.W, **kw(CTRL[c], lw=1.0))
        ax[1].plot(r.time_day, -r.Wdot, **kw(CTRL[c], lw=0.7))
    for a in ax:
        a.set_yscale("log")
    ax[0].set_ylim(1e-6, 3)
    ax[0].set_ylabel(r"$W$")
    ax[1].set_ylim(5e-9, 0.1)
    ax[1].set_ylabel(r"$-dW/dt$ (day$^{-1}$)")
    for i, a in enumerate(ax):
        time_axis(a, label=(i == 1))
        plabel(a, "ab"[i])
        a.yaxis.set_label_coords(-0.085, 0.5)
    fig_legend(fig, [h_line(CTRL[c]) for c in CTRL], 3, gap_mm=4.0)
    save(fig, "fig_lyapunov_W_dW_v2")


def f_lyap_G5():
    L = rd("lyapunov_W_timeseries")
    E = L[(L.protocol == "E1_replay") & (L.controller == "SMC")]
    nom = E[E.group == "nominal"].sort_values("time_day")
    g5 = E[E.group == "G5"].sort_values("time_day")
    sn = dict(color="#3C7DD9", ls="-", lw=0.9, label="Nominal plant")
    s5 = dict(color="#E69F00", ls="-", lw=0.9, label="Plant G5")
    fig, ax = grid(140, 108, 2, 1, top=10, hgap=4.5, left=15)
    for r, st in ((g5, s5), (nom, sn)):
        ax[0].plot(r.time_day, r.W, **st)
        ax[1].plot(r.time_day, r.Wdot, **st)
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-6, 3)
    ax[0].set_ylabel(r"$W$")
    ax[1].set_yscale("symlog", linthresh=1e-6, linscale=0.5)     # linear in |dW/dt| < 1e-6, logarithmic outside
    ax[1].set_ylim(-0.05, 4e-3)
    ax[1].set_yticks([-1e-2, -1e-4, 0, 1e-4, 1e-3])
    ax[1].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax[1].axhline(0, color="0.3", lw=0.7, ls=":")
    ax[1].set_ylabel(r"$dW/dt$ (day$^{-1}$)")
    for i, a in enumerate(ax):
        time_axis(a, label=(i == 1))
        plabel(a, "ab"[i])
        a.yaxis.set_label_coords(-0.085, 0.5)
    fig_legend(fig, [h_line(sn), h_line(s5)], 2, gap_mm=4.0)
    save(fig, "fig_lyapunov_G5_SMC_v2")


FIGS = [f_baseline, f_protocol, f_protocol_4panel, f_uncertainty, f_obs_M1, f_obs_noise, f_obs_mismatch, f_lyap_nominal, f_lyap_G5]

if __name__ == "__main__":
    for f in FIGS:
        f()
