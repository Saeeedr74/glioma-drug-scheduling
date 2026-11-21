"""Item 1.7: all print figures of the paper. Reads only data/ (relative to this file, or --data DIR).
Writes vector PDF + 600 dpi PNG to figures/main and figures/appendix, a legend-overlap check
(figures/figure_checks.json) and, with --preview DIR, low-resolution PNGs for a visual check.
Usage: python make_figures.py [--data DIR] [--out DIR] [--preview DIR]"""
import argparse
import json
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
ap.add_argument("--data", default=os.path.join(HERE, "..", "results", "figure_data"))
ap.add_argument("--out", default=HERE)
ap.add_argument("--preview", default=None)
ARGS = ap.parse_args()
D = ARGS.data

MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
    "axes.unicode_minus": True, "legend.handlelength": 2.8, "legend.columnspacing": 1.4, "legend.handletextpad": 0.5,
})

# fixed styles (Okabe-Ito)
CTRL = {"PID": dict(color="#0072B2", ls="-", lw=1.0, label="PID"),
        "SMC": dict(color="#D55E00", ls="-.", lw=1.0, label="SMC"),
        "mu": dict(color="#009E73", ls="-", lw=1.2, label=r"$\mu$-synthesis")}
REF = dict(color="black", ls="--", lw=0.9, label=r"Reference $r(t)$")
THR = dict(color="0.45", ls=":", lw=0.9, label=r"Threshold $10^{-4}$")
GRP = {"nominal": dict(color="black", ls="-", lw=1.2, label="Nominal plant"),
       "G1": dict(color="#E69F00", ls="-", lw=1.0, label="G1"),
       "G2": dict(color="#56B4E9", ls="--", lw=1.0, label="G2"),
       "G3": dict(color="#CC79A7", ls="-.", lw=1.0, label="G3"),
       "G4": dict(color="#555555", ls=":", lw=1.2, label="G4"),
       "G5": dict(color="#A6761D", ls=(0, (6, 1.5, 1, 1.5, 1, 1.5)), lw=1.0, label="G5")}
OBS = {"ekf": dict(color="#E69F00", ls="-", lw=1.1, label="EKF"),
       "lsd": dict(color="#CC79A7", ls="--", lw=1.0, label="Luenberger (sampled-data)"),
       "lzoh": dict(color="#56B4E9", ls="-.", lw=1.0, label="Luenberger (held output)")}
TLAB = "Time since treatment start (days)"
# state colors for the figures without controllers (kept apart from the controller colors)
STATE = {"x1": dict(color="black", ls="-", label=r"$x_1$ glial cells"), "x2": dict(color="0.45", ls="--", label=r"$x_2$ glioma cells"),
         "x3": dict(color="#56B4E9", ls="-.", lw=1.1, label=r"$x_3$ neurons")}
ULAB = r"$u$ (mg m$^{-2}$ day$^{-1}$)"
CHECKS = {}


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


def h_line(st):
    return Line2D([], [], **st)


def check_legends(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []
    fb = fig.bbox
    for leg in fig.legends + [a.get_legend() for a in fig.axes if a.get_legend() is not None]:
        lb = leg.get_window_extent(r)
        inside = lb.x0 >= fb.x0 - 0.5 and lb.x1 <= fb.x1 + 0.5 and lb.y0 >= fb.y0 - 0.5 and lb.y1 <= fb.y1 + 0.5
        n_pts = 0
        ax_overlap = []
        for k, ax in enumerate(fig.axes):
            tb = ax.get_tightbbox(r)
            if tb is not None and lb.overlaps(tb):
                ax_overlap.append(k)
            ab = ax.get_window_extent(r)
            for ln in ax.get_lines():
                xy = np.asarray(ln.get_xydata(), float)
                if xy.size == 0:
                    continue
                with np.errstate(all="ignore"):
                    p = ax.transData.transform(xy)
                ok = np.isfinite(p).all(axis=1)
                p = p[ok]
                inax = (p[:, 0] >= ab.x0) & (p[:, 0] <= ab.x1) & (p[:, 1] >= ab.y0) & (p[:, 1] <= ab.y1)
                p = p[inax]
                n_pts += int(((p[:, 0] >= lb.x0) & (p[:, 0] <= lb.x1) & (p[:, 1] >= lb.y0) & (p[:, 1] <= lb.y1)).sum())
            for c in list(ax.collections) + list(ax.patches):
                eb = c.get_window_extent(r)
                if eb is not None and eb.width > 0 and lb.overlaps(eb) and lb.overlaps(ab):
                    n_pts += 1
        out.append(dict(inside_figure=bool(inside), points_under_legend=n_pts, overlaps_axes_tightbbox=ax_overlap))
    w, h, _ = fig._mm
    CHECKS[name] = dict(width_mm=w, height_mm=h, legends=out,
                        ok=all(o["inside_figure"] and o["points_under_legend"] == 0 and not o["overlaps_axes_tightbbox"] for o in out))


def save(fig, folder, name):
    fit_legend(fig)
    check_legends(fig, name)
    d = os.path.join(ARGS.out, folder)
    os.makedirs(d, exist_ok=True)
    for ext, dpi in (("pdf", None), ("png", 600)):
        p = os.path.join(d, name + "." + ext)
        tmp = os.path.join(d, name + ".tmp." + ext)
        fig.savefig(tmp, dpi=dpi)
        os.replace(tmp, p)
    if ARGS.preview:
        os.makedirs(ARGS.preview, exist_ok=True)
        fig.savefig(os.path.join(ARGS.preview, name + ".png"), dpi=130)
    plt.close(fig)


def step(ax, t, u, **st):
    u = np.asarray(u, float)[:2000]
    ax.step(np.arange(2001), np.append(u, u[-1]), where="post", **st)


def rd(name, **k):
    p = os.path.join(D, name + ".csv")
    if not os.path.exists(p):          # the GitHub package stores the larger files gzipped
        p += ".gz"
    return pd.read_csv(p, **k)


ND = rd("nominal_discrete")
NC = rd("nominal_continuous")
REFS = rd("reference")


def run(df, col, val):
    return df[df[col] == val].sort_values("time_day")


def ref_line(ax):
    ax.plot(REFS.time_day, REFS.x2_reference, **REF)


def thr_line(ax, v=1e-4, st=THR):
    ax.axhline(v, **st)


# ---------------- main text -----------------
def m1_preconditioning():
    P = rd("preconditioning")
    fig, (ax,) = grid(90, 62, 1, 1, top=13, left=12)
    st = STATE
    for k, s in st.items():
        ax.plot(P.sim_day, P[k], **s)
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Simulation time (days), no treatment")
    ax.set_ylabel("Normalized density")
    fig_legend(fig, [h_line(s) for s in st.values()], 3, gap_mm=2.0)
    save(fig, "main", "fig_preconditioning")


def m2_baseline():
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
    save(fig, "main", "fig_baseline")


def m3_schedules():
    fig, ax = grid(190, 82, 3, 1, top=10, hgap=4.5, left=14)
    for i, c in enumerate(("PID", "SMC", "mu")):
        r = run(ND, "controller", c)
        step(ax[i], r.time_day, r.u, **kw(CTRL[c], lw=0.8, ls="-"))
        ax[i].set_ylim(30, 220)
        ax[i].set_yticks([50, 200])
        ax[i].set_ylabel(ULAB if i == 1 else "")
        time_axis(ax[i], label=(i == 2))
        plabel(ax[i], "abc"[i])
    fig_legend(fig, [h_line(kw(CTRL[c], ls="-")) for c in CTRL], 3, gap_mm=4.0)
    save(fig, "main", "fig_schedules")


def m4_states():
    fig, ax = grid(190, 112, 2, 2, top=11, hgap=12, wgap=17, left=14)
    # (a) spans the top row; (b), (c) share the bottom row
    p0, p1 = ax[0].get_position(), ax[1].get_position()
    ax[0].set_position([p0.x0, p0.y0, p1.x1 - p0.x0, p0.height])
    ax[1].remove()
    ax = [ax[0], ax[2], ax[3]]
    for c in ("PID", "SMC", "mu"):
        r = run(ND, "controller", c)
        ax[0].plot(r.time_day, pos(r.x2), **CTRL[c])
        ax[1].plot(r.time_day, r.x1, **CTRL[c])
        ax[2].plot(r.time_day, r.x3, **CTRL[c])
    ref_line(ax[0])
    thr_line(ax[0])
    logy(ax[0], 1e-7, 1)
    ax[0].set_ylabel(r"$x_2$ glioma cells (normalized)")
    ax[1].set_ylabel(r"$x_1$ glial cells (normalized)")
    ax[2].set_ylabel(r"$x_3$ neurons (normalized)")
    for i, a in enumerate(ax):
        time_axis(a)
        plabel(a, "abc"[i])
    fig_legend(fig, [h_line(CTRL[c]) for c in CTRL] + [h_line(REF), h_line(THR)], 5)
    save(fig, "main", "fig_states")


def m4b_x4():
    fig, ax = grid(190, 82, 3, 1, top=10, hgap=4.5, left=14)
    for i, c in enumerate(("PID", "SMC", "mu")):
        r = run(ND, "controller", c)
        ax[i].plot(r.time_day, r.x4, **kw(CTRL[c], ls="-", lw=0.8))
        ax[i].set_ylim(0, 800)
        ax[i].set_yticks([0, 250, 500, 750])
        ax[i].set_ylabel(r"$x_4$ (mg m$^{-2}$)" if i == 1 else "")
        time_axis(ax[i], label=(i == 2))
        plabel(ax[i], "abc"[i])
    fig_legend(fig, [h_line(kw(CTRL[c], ls="-")) for c in CTRL], 3, gap_mm=4.0)
    save(fig, "main", "fig_x4")

def m5_pid_gains():
    G = rd("pid_gain_schedule")
    fig, (ax,) = grid(90, 62, 1, 1, top=10, left=13)
    st = {"Kp": dict(color="#0072B2", ls="-", label=r"$K_p$"), "Ki": dict(color="#0072B2", ls="--", label=r"$K_i$"),
          "Kd": dict(color="#0072B2", ls=":", lw=1.2, label=r"$K_d$")}
    for k, s in st.items():
        ax.plot(G.time_day, G[k], **s)
    ax.set_yscale("log")
    ax.set_ylabel("Gain")
    time_axis(ax, step=500)
    fig_legend(fig, [h_line(s) for s in st.values()], 3, gap_mm=2.0)
    save(fig, "main", "fig_pid_gains")


def m6_smc_band():
    S = rd("smc_sliding_band")
    fig, ax = grid(140, 80, 2, 1, top=10, hgap=4.5, left=15)
    sst = kw(CTRL["SMC"], ls="-", lw=0.8, label=r"Sliding variable $s$")
    dst = dict(color="black", ls="--", lw=0.9, label=r"Band $\pm\delta(t)$")
    ax[0].plot(S.time_day, S.s, **sst)
    ax[0].plot(S.time_day, S.delta, **dst)
    ax[0].plot(S.time_day, -S.delta, **dst)
    ax[0].set_ylabel(r"$s$, $\pm\delta$")
    ax[1].plot(S.time_day, pos(np.abs(S.s)), **kw(sst, label=r"$|s|$"))
    ax[1].plot(S.time_day, S.delta, **dst)
    ax[1].set_yscale("log")
    ax[1].set_ylabel(r"$|s|$, $\delta$")
    for i, a in enumerate(ax):
        time_axis(a, label=(i == 1))
        plabel(a, "ab"[i])
        a.yaxis.set_label_coords(-0.085, 0.5)
    fig_legend(fig, [h_line(sst), h_line(dst)], 2)
    save(fig, "main", "fig_smc_band")


def m7_continuous():
    fig, ax = grid(140, 85, 2, 1, top=14, hgap=4.5, left=15)
    for c in ("SMC", "mu"):
        rc = run(NC, "controller", c)
        rdd = run(ND, "controller", c)
        step(ax[0], rc.time_day, rc.u, **kw(CTRL[c], ls="--", lw=0.9))
        ax[1].plot(rdd.time_day, pos(rdd.x2), **CTRL[c])
        ax[1].plot(rc.time_day, pos(rc.x2), **kw(CTRL[c], ls="--"))
    ax[0].set_ylabel(ULAB)
    ax[0].set_ylim(-10, 215)
    ref_line(ax[1])
    thr_line(ax[1])
    logy(ax[1], 1e-7, 1)
    ax[1].set_ylabel(r"$x_2$ (normalized)")
    for i, a in enumerate(ax):
        time_axis(a, label=(i == 1))
        plabel(a, "ab"[i])
        a.yaxis.set_label_coords(-0.085, 0.5)
    hs = [h_line(kw(CTRL["SMC"], ls="--", label="SMC, continuous")), h_line(kw(CTRL["mu"], ls="--", label=r"$\mu$-synthesis, continuous")),
          h_line(kw(CTRL["SMC"], label="SMC, discrete")), h_line(kw(CTRL["mu"], label=r"$\mu$-synthesis, discrete")), h_line(REF), h_line(THR)]
    fig_legend(fig, hs, 3)
    save(fig, "main", "fig_continuous")


def m8_uncertainty():
    E = rd("e1_uncertainty")
    fig, ax = grid(190, 112, 2, 2, top=15, hgap=12, wgap=17, left=14)
    for i, c in enumerate(("PID", "SMC", "mu")):
        for g in ("nominal", "G1", "G2", "G3", "G4", "G5"):
            r = E[(E.controller == c) & (E.group == g)].sort_values("time_day")
            ax[i].plot(r.time_day, pos(r.x2), **GRP[g])
        ref_line(ax[i])
        thr_line(ax[i])
        logy(ax[i], 1e-7, 1)
        ax[i].set_ylabel(r"$x_2$ (normalized)")
        time_axis(ax[i])
    groups = ["nominal", "G1", "G2", "G3", "G4", "G5"]
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
    hs = [h_line(GRP[g]) for g in groups] + [h_line(REF), h_line(THR)] + \
         [Patch(facecolor=CTRL[c]["color"], edgecolor="black", linewidth=0.4, hatch=hatch[c], label=CTRL[c]["label"]) for c in CTRL]
    fig_legend(fig, hs, 6)
    save(fig, "main", "fig_uncertainty")


def m9_standard():
    SP = rd("standard_protocol")
    smc = run(ND, "controller", "SMC")
    st2 = dict(color="#E69F00", ls="--", lw=1.0, label="5/28 protocol, 200")
    st4 = dict(color="#CC79A7", ls=":", lw=1.3, label="5/28 protocol, 400")
    sm = kw(CTRL["SMC"], label="SMC (discrete)")
    fig, ax = grid(140, 88, 2, 1, top=14, hgap=4.5, left=15)
    for r, st in ((run(SP, "run", "standard_5_28_dose200"), st2), (run(SP, "run", "standard_5_28_dose400"), st4), (smc, sm)):
        ax[0].plot(r.time_day, pos(r.x2), **st)
        u = np.asarray(r.u, float)[:2000]
        ax[1].plot(np.arange(2001), np.concatenate([[0], np.cumsum(u)]) / 1e3, **st)
    ref_line(ax[0])
    thr_line(ax[0])
    logy(ax[0], 1e-7, 1)
    ax[0].set_ylabel(r"$x_2$ (normalized)")
    ax[1].set_ylabel(r"Cumulative dose (10$^3$ mg m$^{-2}$)")
    for i, a in enumerate(ax):
        time_axis(a, label=(i == 1))
        plabel(a, "ab"[i])
        a.yaxis.set_label_coords(-0.085, 0.5)
    fig_legend(fig, [h_line(sm), h_line(st2), h_line(st4), h_line(REF), h_line(THR)], 3)
    save(fig, "main", "fig_standard_protocol")


def m10_observer():
    O = rd("observer_M1_nominal_timeseries")
    O["delta"] = O["delta"].astype(float)
    fig, ax = grid(190, 112, 2, 2, top=16, hgap=12, wgap=17, left=14)
    r0 = O[(O.controller == "mu") & (O.observer == "ekf") & (O.delta == 0.2)].sort_values("time_day")
    ax[0].plot(r0.time_day, r0.x1, color="black", ls="-", lw=1.2, label=r"True $x_1$")
    for ob in ("ekf", "lsd", "lzoh"):
        r = O[(O.controller == "mu") & (O.observer == ob) & (O.delta == 0.2)].sort_values("time_day")
        ax[0].plot(r.time_day, r.x1_hat, **OBS[ob])
    ax[0].set_ylabel(r"$x_1$, $\hat{x}_1$ (normalized)")
    for i, c in zip((1, 2, 3), ("mu", "pid", "smc")):
        for ob in ("ekf", "lsd", "lzoh"):
            for dv, lw in ((0.2, 1.0), (-0.2, 0.6)):
                r = O[(O.controller == c) & (O.observer == ob) & (O.delta == dv)].sort_values("time_day")
                ax[i].plot(r.time_day, pos(r.e1), **kw(OBS[ob], lw=lw if dv < 0 else OBS[ob]["lw"]))
        ax[i].axhline(0.01, color="0.45", ls=":", lw=0.9)
        logy(ax[i], 1e-8, 1)
        ax[i].set_ylabel(r"$e_1 = |\hat{x}_1 - x_1|/x_1$")
    for i, a in enumerate(ax):
        time_axis(a)
        plabel(a, "abcd"[i])
    hs = [Line2D([], [], color="black", lw=1.2, label=r"True $x_1$")] + [h_line(OBS[o]) for o in OBS] + \
         [Line2D([], [], color="0.3", lw=1.0, label=r"$\delta = +0.2$ (thick)"), Line2D([], [], color="0.3", lw=0.6, label=r"$\delta = -0.2$ (thin)"),
          Line2D([], [], color="0.45", ls=":", lw=0.9, label="1% convergence level")]
    fig_legend(fig, hs, 4)
    save(fig, "main", "fig_observer")


# ---------------- appendix -----------------
def a1_tuning():
    T = rd("tuning_convergence")
    T["seed"] = T["seed"].astype(str)
    fig, ax = grid(140, 62, 1, 2, top=10, wgap=17, left=14)
    sty = {"7": "-", "11": "--", "23": ":"}
    for i, c in enumerate(("PID", "SMC")):
        for s, ls in sty.items():
            r = T[(T.controller == c) & (T.seed == s)].sort_values("eval")
            ax[i].plot(r["eval"], r.best_J_so_far, color=CTRL[c]["color"], ls=ls, lw=1.0 if ls != ":" else 1.3)
        ax[i].set_yscale("log")
        ax[i].set_ylim(1e-2, 2)
        ax[i].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax[i].set_xlabel("Objective evaluations")
        ax[i].set_ylabel(r"Best objective $J$")
        ax[i].set_xlim(0, None)
        plabel(ax[i], "ab"[i])
    hs = [Line2D([], [], color="0.3", ls=ls, lw=1.0 if ls != ":" else 1.3, label="seed %s" % s) for s, ls in sty.items()] + \
         [Patch(color=CTRL[c]["color"], label=c) for c in ("PID", "SMC")]
    fig_legend(fig, hs, 5, gap_mm=4.0)
    save(fig, "appendix", "fig_app_tuning")


def a2_mu_search():
    W = rd("mu_weight_search")
    S = rd("mu_strings").set_index("key")["value"]
    R = rd("mu_per_redesign")
    fig, ax = grid(140, 62, 1, 2, top=10, wgap=16, left=13)
    ok = W[W.ok == 1]
    bad = W[W.ok != 1]
    win = W[W.tag == S["search_winner_tag"]]
    ax[0].plot(ok.candidate, ok.J, ls="none", marker="o", ms=3, mfc="white", mec="0.3", mew=0.7, label="Valid candidate")
    yb = np.nanmax(ok.J) * 1.8
    ax[0].plot(bad.candidate, np.full(len(bad), yb), ls="none", marker="x", ms=3.5, color="#D55E00", mew=0.8, label="Rejected candidate")
    ax[0].plot(win.candidate, win.J, ls="none", marker="*", ms=8, color=CTRL["mu"]["color"], mec="black", mew=0.4, label="Selected design")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Candidate index")
    ax[0].set_ylabel(r"Search objective $J$")
    for run_, st in (("discrete_nominal", dict(color=CTRL["mu"]["color"], ls="none", marker="o", ms=2.5, label=r"Discrete, nominal")),
                     ("continuous_nominal", dict(color=CTRL["mu"]["color"], ls="none", marker="s", ms=2.5, mfc="white", mew=0.6, label=r"Continuous, nominal"))):
        r = R[R.run == run_].sort_values("day")
        ax[1].plot(r.day, r.mu_peak, **st)
    ax[1].set_xlim(0, 2000)
    ax[1].set_xticks(np.arange(0, 2001, 500))
    ax[1].set_xlabel(TLAB)
    ax[1].set_ylabel(r"Peak $\mu$ of each redesign")
    for i, a in enumerate(ax):
        plabel(a, "ab"[i])
    hs = [h for a in ax for h in a.get_legend_handles_labels()[0]]
    fig_legend(fig, hs, 5, gap_mm=4.0)
    save(fig, "appendix", "fig_app_mu_search")


def a3_mu_e2_mc():
    E2 = rd("mu_e2_closed_loop")
    B = rd("mu_mc_bands")
    fig, ax = grid(140, 92, 2, 1, top=18, hgap=4.5, left=15)
    r = run(ND, "controller", "mu")
    ax[0].plot(r.time_day, pos(r.x2), **GRP["nominal"])
    for g in ("G1", "G2", "G3", "G4", "G5"):
        rr = E2[E2.group == g].sort_values("time_day")
        ax[0].plot(rr.time_day, pos(rr.x2), **GRP[g])
    ax[1].fill_between(B.time_day, pos(B.replay_p5), pos(B.replay_p95), color="0.80", lw=0, label="Replay, 5-95% (300 runs)")
    ax[1].plot(B.time_day, pos(B.replay_p50), color="0.35", ls="--", lw=1.0, label="Replay, median")
    ax[1].fill_between(B.time_day, pos(B.cl_p5), pos(B.cl_p95), color=CTRL["mu"]["color"], alpha=0.35, lw=0, label="Closed loop, 5-95% (20 runs)")
    ax[1].plot(B.time_day, pos(B.cl_p50), color=CTRL["mu"]["color"], ls="-", lw=1.1, label="Closed loop, median")
    for a in ax:
        ref_line(a)
        thr_line(a)
        logy(a, 1e-7, 1)
        a.set_ylabel(r"$x_2$ (normalized)")
    for i, a in enumerate(ax):
        time_axis(a, label=(i == 1))
        plabel(a, "ab"[i])
    hs = [h_line(GRP[g]) for g in GRP] + [h_line(REF), h_line(THR)] + ax[1].get_legend_handles_labels()[0][:4]
    fig_legend(fig, hs, 4)
    save(fig, "appendix", "fig_app_mu_E2_mc")


def a4_mu_rules():
    R = rd("mu_redesign_rules")
    st = {"cap 40": dict(color="#E69F00", ls="--", lw=1.0, label="At most 40 redesigns"),
          "stop at x2 < 0.001": dict(color="#56B4E9", ls="-.", lw=1.0, label=r"Stop at $x_2 < 10^{-3}$"),
          "stop at x2 < 0.0001": dict(color=CTRL["mu"]["color"], ls="-", lw=1.2, label=r"Stop at $x_2 < 10^{-4}$ (used)"),
          "no stop, no cap": dict(color="#CC79A7", ls=":", lw=1.3, label="No stop, no cap")}
    fig, ax = grid(140, 88, 2, 1, top=14, hgap=4.5, left=15)
    for k, s in st.items():
        r = run(R, "rule", k)
        ax[0].plot(r.time_day, pos(r.x2), **s)
        u = np.asarray(r.u, float)[:2000]
        ax[1].plot(np.arange(2001), np.concatenate([[0], np.cumsum(u)]) / 1e3, **s)
    ref_line(ax[0])
    thr_line(ax[0])
    logy(ax[0], 1e-7, 1)
    ax[0].set_ylabel(r"$x_2$ (normalized)")
    ax[1].set_ylabel(r"Cumulative dose (10$^3$ mg m$^{-2}$)")
    for i, a in enumerate(ax):
        time_axis(a, label=(i == 1))
        plabel(a, "ab"[i])
        a.yaxis.set_label_coords(-0.085, 0.5)
    fig_legend(fig, [h_line(s) for s in st.values()] + [h_line(REF), h_line(THR)], 3)
    save(fig, "appendix", "fig_app_mu_rules")


def a5_observer_noise():
    N = rd("observer_noise_e1_stored_seeds")
    N["delta"] = N["delta"].astype(float)
    fig, ax = grid(190, 62, 1, 3, top=10, wgap=15, left=14)
    for i, m in enumerate(("M2", "M3_7", "M3_28")):
        for ob in ("ekf", "lsd"):
            r = N[(N.controller == "mu") & (N.observer == ob) & (N["mode"] == m) & (N.delta == 0.2)].sort_values("time_day")
            ax[i].fill_between(r.time_day, pos(r.e1_min), pos(r.e1_max), color=OBS[ob]["color"], alpha=0.25, lw=0)
            ax[i].plot(r.time_day, pos(r.e1_median), **OBS[ob])
        ax[i].axhline(0.01, color="0.45", ls=":", lw=0.9)
        logy(ax[i], 1e-5, 1)
        ax[i].set_ylabel(r"$e_1$" if i == 0 else "")
        time_axis(ax[i], step=500)
        plabel(ax[i], "abc"[i])
    hs = [h_line(kw(OBS[o], label=OBS[o]["label"] + ", median")) for o in ("ekf", "lsd")] + \
         [Patch(color=OBS[o]["color"], alpha=0.25, label=OBS[o]["label"] + ", range of 5 seeds") for o in ("ekf", "lsd")] + \
         [Line2D([], [], color="0.45", ls=":", lw=0.9, label="1% level")]
    fig_legend(fig, hs, 3, gap_mm=4.0)
    save(fig, "appendix", "fig_app_observer_noise")


def a6_observer_groups():
    O = rd("observer_M1_groups_timeseries")
    O["delta"] = O["delta"].astype(float)
    fig, ax = grid(140, 62, 1, 2, top=11, wgap=16, left=13)
    for i, ob in enumerate(("ekf", "lsd")):
        for g in ("G1", "G2", "G3", "G4", "G5"):
            r = O[(O.controller == "mu") & (O.observer == ob) & (O.plant == g) & (O.delta == 0.2)].sort_values("time_day")
            ax[i].plot(r.time_day, pos(r.e1), **GRP[g])
        sub = O[(O.controller == "mu") & (O.observer == ob) & (O.delta == 0.2)].e1.replace([np.inf, -np.inf], np.nan)
        CHECKS.setdefault("_notes", {})["observer_groups_%s_max_e1" % ob] = float(np.nanmax(sub))
        logy(ax[i], 1e-3, 10 ** np.ceil(np.log10(max(10.0, np.nanmax(sub) * 1.5))))
        ax[i].set_ylabel(r"$e_1$")
        time_axis(ax[i], step=500)
        plabel(ax[i], "ab"[i])
    fig_legend(fig, [h_line(GRP[g]) for g in ("G1", "G2", "G3", "G4", "G5")], 5, gap_mm=4.0)
    save(fig, "appendix", "fig_app_observer_groups")


def a7_lyapunov():
    L = rd("lyapunov_W_timeseries")
    E = L[L.protocol == "E1_replay"]
    fig, ax = grid(140, 88, 2, 1, top=14, hgap=4.5, left=15)
    for c in ("PID", "SMC", "mu"):
        r = E[(E.controller == c) & (E.group == "nominal")].sort_values("time_day")
        ax[0].plot(r.time_day, pos(r.W), **CTRL[c])
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"$W$ (nominal plant)")
    for g in ("nominal", "G1", "G2", "G3", "G4", "G5"):
        r = E[(E.controller == "mu") & (E.group == g)].sort_values("time_day")
        ax[1].plot(r.time_day, np.cumsum(r.Wdot.values > 1e-10), **GRP[g])
    ax[1].set_ylabel(r"Days with $\dot W > 10^{-10}$")
    for i, a in enumerate(ax):
        time_axis(a, label=(i == 1))
        plabel(a, "ab"[i])
        a.yaxis.set_label_coords(-0.085, 0.5)
    fig_legend(fig, [h_line(CTRL[c]) for c in CTRL] + [h_line(GRP[g]) for g in GRP], 5)
    save(fig, "appendix", "fig_app_lyapunov")


FIGS = [m1_preconditioning, m2_baseline, m3_schedules, m4_states, m4b_x4, m5_pid_gains, m6_smc_band, m7_continuous, m8_uncertainty,
        m9_standard, m10_observer, a1_tuning, a2_mu_search, a3_mu_e2_mc, a4_mu_rules, a5_observer_noise, a6_observer_groups, a7_lyapunov]

if __name__ == "__main__":
    for f in FIGS:
        f()
    p = os.path.join(ARGS.out, "figure_checks.json")
    with open(p + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(CHECKS, fh, indent=1)
    os.replace(p + ".tmp", p)
    figs = {k: v for k, v in CHECKS.items() if not k.startswith("_")}
    bad = [k for k, v in figs.items() if not v["ok"]]
    print("figures: %d; legend check pass: %d; fail: %s" % (len(figs), len(figs) - len(bad), bad or "none"))
    print("notes:", CHECKS.get("_notes"))
    for k in bad:
        print("  ", k, CHECKS[k]["legends"])
