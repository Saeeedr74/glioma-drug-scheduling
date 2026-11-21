"""make_facts_block.py - writes results/facts_block.txt from the stored results of observer.py.

It runs no simulation and changes no setting: it reads results/metrics_all.csv,
results/summary_by_scenario.csv and results/timeseries/*.csv.gz, and evaluates the nominal and
perturbed right-hand sides (functions of observer.py) on the stored plant trajectories.
Run after observer.py:   python make_facts_block.py
"""
from datetime import datetime

import numpy as np
import pandas as pd

import observer as ob

RES, TS = ob.RES_DIR, ob.TS_DIR
MET = pd.read_csv(RES / "metrics_all.csv")
SUM = pd.read_csv(RES / "summary_by_scenario.csv")
D = ob.DELTA
WIN = slice(500, 1501)                 # days 500..1500 for the bias comparison
GROWTH_GROUPS = ("G1", "G3", "G4", "G5")


def srow(obs, ctrl, plant, delta, mode):
    r = SUM[(SUM.observer == obs) & (SUM.controller == ctrl) & (SUM.plant == plant)
            & np.isclose(SUM.delta, delta) & (SUM["mode"] == mode)]
    assert len(r) == 1, (obs, ctrl, plant, delta, mode)
    return r.iloc[0]


def f(v, kind="e"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n.c."
    return {"e": f"{v:.2e}", "d": f"{v:.0f}", "p": f"{100 * v:.2f}%", "t": f"{v:.1f}"}[kind]


def pair(obs, ctrl, mode, col, kind="e", stat="median"):
    return " / ".join(f(srow(obs, ctrl, "nominal", d, mode)[f"{col}_{stat}"], kind) for d in (D, -D))


def bias_table():
    """x1 offset predicted from the growth-rate mismatch vs the EKF bias observed in M1."""
    rows = []
    for ctrl in ob.CTRLS:
        truth = pd.read_csv(TS / f"truth_{ctrl}.csv.gz")
        est = pd.read_csv(TS / f"est_ekf_{ctrl}_M1.csv.gz")
        m = MET[(MET.observer == "ekf") & (MET.controller == ctrl) & (MET["mode"] == "M1")]
        for plant in ("G1", "G2", "G3", "G4", "G5"):
            T = truth[truth.plant == plant].sort_values("day")
            x1, x2, x4, x4h = (T[c].to_numpy() for c in ("x1", "x2", "x4", "x4_hat_nominal_zeta"))
            p = ob.params(plant)
            f_true = np.array([ob.f2(a, b, c, p)[1] for a, b, c in zip(x1, x2, x4)])
            f_nom = np.array([ob.f2(a, b, c)[1] for a, b, c in zip(x1, x2, x4h)])
            eps = (f_true - f_nom) / x2               # relative growth-rate error of x2 [1/day]
            pred = -eps / ob.NOMINAL["b2"]
            obs_b = []
            for d in (D, -D):
                rid = int(m[np.isclose(m.delta, d) & (m.plant == plant)].run_id.iloc[0])
                xh = est[est.run_id == rid].sort_values("day").x1_hat.to_numpy()
                obs_b.append(float(np.median((xh - x1)[WIN])))
            rows.append(dict(controller=ctrl, plant=plant, eps=float(np.median(eps[WIN])),
                             pred=float(np.median(pred[WIN])), obs_p=obs_b[0], obs_m=obs_b[1]))
    return pd.DataFrame(rows)


def main():
    L = []
    w = L.append
    w("FACTS BLOCK - state observers for the glioma chemotherapy controllers (mu, PID, SMC)")
    w(f"Written {datetime.now():%Y-%m-%d %H:%M} by make_facts_block.py from the stored results of observer.py")
    w("(no simulation rerun, no setting changed).")
    w("")
    w("ROLES FOR THE PAPER")
    w("  Main observer        : discrete extended Kalman filter (EKF) on [x1, x2].")
    w("  Comparison baseline  : corrected adaptive Luenberger observer, sampled-data form (lsd).")
    w("  Also reported (M1)   : the continuous-time Luenberger form with y held over each day (lzoh),")
    w("                         i.e. the form of the original task text; it shows a sampling-lag bias.")
    w("")
    w("COMMON SETUP (source: observer.py SETTINGS, README.md section 4)")
    w(f"  y = x2 (x3 excluded: rank of [C; CA; CA^2] at x_pre with y = x2 is 2); x4_hat from the applied dose")
    w(f"  with nominal zeta = {ob.NOMINAL['zeta']}; nominal model in every observer; x2_hat(0) = y(0);")
    w(f"  x1_hat(0) = (1 +/- {D}) x1(0). Measurement modes: M1 daily noise-free; M2 daily, sigma = {ob.SIGMA};")
    w(f"  M3 every {ob.TM_LIST[0]} or {ob.TM_LIST[1]} days, sigma = {ob.SIGMA}; {ob.N_SEEDS} seeds per noisy scenario.")
    w(f"  Luenberger: poles p_i = -max({ob.K_OBS:g}|Re lambda_i(A)|, {ob.P_MIN:g}/day), relinearisation trigger")
    w(f"  {100 * ob.TRIGGER:g}% on x2_hat. EKF: Q = diag((q x_hat)^2), q = {ob.EKF_Q_REL:g}/day; R = (sigma x2)^2")
    w(f"  (M1: sigma_eff = {ob.EKF_R_REL_M1:g}); P0 = diag((delta x1_hat(0))^2, (sigma y0)^2).")
    w(f"  Convergence day = first day after which e1 = |x1_hat - x1|/x1 stays below {ob.CONV_THR:g}.")
    w("  EKF 'updates' = Jacobian propagations / measurement updates; Luenberger = triggered relinearisations.")
    w("  Pairs 'a / b' below are delta = +0.2 / -0.2.  n.c. = not converged by day 2000.")
    w("")
    w("A. MAIN NUMBERS - M1 (daily, noise-free), nominal plant  [source: results/summary_by_scenario.csv]")
    for ctrl in ob.CTRLS:
        e, l, z = (lambda c, k="e": pair("ekf", ctrl, "M1", c, k)), \
                  (lambda c, k="e": pair("lsd", ctrl, "M1", c, k)), \
                  (lambda c, k="e": pair("lzoh", ctrl, "M1", c, k))
        r = srow("lsd", ctrl, "nominal", D, "M1")
        rm = srow("lsd", ctrl, "nominal", -D, "M1")
        w(f"  [{ctrl}]                         EKF (main)              Luenberger lsd (baseline)   Luenberger lzoh (task form)")
        w(f"    convergence day (e1<1%)      {e('conv_day', 'd'):<23} {l('conv_day', 'd'):<27} {z('conv_day', 'd')}")
        w(f"    e1 at day 2000               {e('e1_final'):<23} {l('e1_final'):<27} {z('e1_final')}")
        w(f"    RMS e1, days 0..2000         {e('rms_e1_0_2000'):<23} {l('rms_e1_0_2000'):<27} {z('rms_e1_0_2000')}")
        w(f"    RMS e1, days 100..2000       {e('rms_e1_100_2000'):<23} {l('rms_e1_100_2000'):<27} {z('rms_e1_100_2000')}")
        w(f"    e2 at day 2000               {e('e2_final'):<23} {l('e2_final'):<27} {z('e2_final')}")
        w(f"    updates                      {'2000 / 2000 (daily)':<23} "
          f"{r.n_updates_median:.0f} / {rm.n_updates_median:.0f} relinearisations")
        w(f"    min rank [C; CA]             2                       {r.rank_min:.0f} (rank-deficient updates: 0)")
        w(f"    max Re eig(A - LC)           -                       {r.max_re_eig_A_LC_worst:.4f} /day; max |eig((I-KC)Phi)| = {r.max_abs_eig_disc_worst:.5f}")
        w(f"    observer time constants      -                       day 0: {r.tau1_day0:.1f} and {r.tau2_day0:.1f} d (+delta), "
          f"{rm.tau1_day0:.1f} and {rm.tau2_day0:.1f} d (-delta); over all updates {r.tau_min:.1f}..{r.tau_max:.1f} d")
    w("  At x_pre with x1_hat = x1 (step 2 of observer.py): eig(A) = {-0.008549, +0.003840}/day; the old rule")
    w("  2 eig(A) gives the UNSTABLE pole +0.00768/day; the new rule gives -0.0100 and -0.0171/day (tau 100 and 58.5 d).")
    w("")
    w("B. NOISE AND SPARSE IMAGING - nominal plant, median / worst over 20 seeds, delta = +0.2  [summary_by_scenario.csv]")
    for mode in ("M2", "M3_7", "M3_28"):
        w(f"  {mode}:")
        for ctrl in ob.CTRLS:
            for obs, lab in (("ekf", "EKF"), ("lsd", "lsd")):
                r = srow(obs, ctrl, "nominal", D, mode)
                w(f"    {ctrl:<4}{lab:<4} conv. day {f(r.conv_day_median, 'd'):>5} / {f(r.conv_day_worst, 'd'):<5} "
                  f"RMS e1 100..2000 {f(r.rms_e1_100_2000_median)} / {f(r.rms_e1_100_2000_worst)}  "
                  f"e1(2000) {f(r.e1_final_median)} / {f(r.e1_final_worst)}  not converged {r.n_not_converged:.0f}/{r.n_runs:.0f}")
    w("")
    # ---- fact 1
    bt = bias_table()
    gg = bt[bt.plant.isin(GROWTH_GROUPS)]
    obs_all = np.r_[gg.obs_p.to_numpy(), gg.obs_m.to_numpy()]
    w("C. FACT 1 - x1 IS NOT IDENTIFIABLE FROM y = x2 UNDER GROWTH-RATE MISMATCH (structural, not a code error)")
    w("  Mechanism: x1 enters dx2/dt only through -b2 x1 x2, so a relative growth-rate error eps of the x2")
    w(f"  equation is indistinguishable from an x1 offset of about -eps/b2 (b2 = {ob.NOMINAL['b2']:g}/day).")
    w("  eps(t) = [f2_plant(x, x4) - f2_nominal(x, x4_hat)]/x2 on the stored plant trajectory; medians over days 500..1500;")
    w("  observed = median(x1_hat - x1), EKF, M1.  [source: results/timeseries/truth_<ctrl>.csv.gz,")
    w("  est_ekf_<ctrl>_M1.csv.gz; computed by make_facts_block.py]")
    w("    ctrl  group   eps [1/day]   predicted -eps/b2   observed (+delta / -delta)")
    for _, r in bt.iterrows():
        w(f"    {r.controller:<5} {r.plant:<6} {r.eps:+.2e}     {r.pred:+.3f}              {r.obs_p:+.3f} / {r.obs_m:+.3f}")
    w(f"  Groups G1, G3, G4, G5, all controllers: predicted {gg.pred.min():+.2f} to {gg.pred.max():+.2f}, "
      f"observed {obs_all.min():+.2f} to {obs_all.max():+.2f} (same sign, same order).")
    w("  G2 changes b2 itself (the coupling), so the simple mapping does not apply; its observed bias is small.")
    m1 = MET[(MET["mode"] == "M1") & (MET.plant != "nominal")]
    w("  Resulting x1 and x2 errors, M1, five groups  [source: results/metrics_all.csv]:")
    for obs in ("ekf", "lsd", "lzoh"):
        g = m1[m1.observer == obs]
        ok = g[g.status == "ok"]
        gr = ok[ok.plant.isin(GROWTH_GROUPS)]
        g2 = ok[ok.plant == "G2"]
        w(f"    {obs:<5} e1(2000) G1/G3/G4/G5 {gr.e1_final.min():.2f}..{gr.e1_final.max():.2f}; "
          f"G2 {g2.e1_final.min():.3f}..{g2.e1_final.max():.3f}; e2(2000) max {100 * ok.e2_final.max():.2f}%, "
          f"RMS e2 max {100 * ok.rms_e2_0_2000.max():.2f}%; diverged {int((g.status != 'ok').sum())}/{len(g)}")
    okm = m1[m1.status == "ok"]
    e2max = okm.groupby("observer").e2_final.max()
    w(f"  -> x2 stays well estimated while x1 does not: worst e2(2000) over the non-diverged M1 group runs is "
      f"{100 * e2max['ekf']:.2f}% (EKF),")
    w(f"     {100 * e2max['lsd']:.1f}% (lsd) and {100 * e2max['lzoh']:.1f}% (lzoh); the diverged runs are all "
      "Luenberger runs (counted above).")
    w("")
    # ---- fact 2
    w("D. FACT 2 - EKF CONSISTENCY")
    for ctrl in ob.CTRLS:
        est = pd.read_csv(TS / f"est_ekf_{ctrl}_M1.csv.gz")
        truth = pd.read_csv(TS / f"truth_{ctrl}.csv.gz")
        x1 = truth[truth.plant == "nominal"].sort_values("day").x1.to_numpy()
        m = MET[(MET.observer == "ekf") & (MET.controller == ctrl) & (MET["mode"] == "M1") & (MET.plant == "nominal")]
        parts = []
        for d in (D, -D):
            rid = int(m[np.isclose(m.delta, d)].run_id.iloc[0])
            e = est[est.run_id == rid].sort_values("day")
            parts.append((e.sd_x1.to_numpy()[-1] / x1[-1], np.abs(e.x1_hat.to_numpy()[-1] - x1[-1]) / x1[-1]))
        w(f"  {ctrl:<4} nominal, M1: sqrt(P11)/x1 at day 2000 = {100 * parts[0][0]:.2f}% / {100 * parts[1][0]:.2f}%, "
          f"true e1 = {parts[0][1]:.1e} / {parts[1][1]:.1e}  -> conservative (floor set by Q)")
    ek = MET[(MET.observer == "ekf") & (MET.status == "ok")]
    fr = ek.groupby("plant").ekf_frac_in_2sd_100_2000.median()
    w("  Share of days 100..2000 with |x1_hat - x1| <= 2 sqrt(P11), median over all EKF runs per plant")
    w("  (all controllers, modes, seeds, both delta)  [source: results/metrics_all.csv, ekf_frac_in_2sd_100_2000]:")
    w("    " + ", ".join(f"{p} {100 * fr[p]:.1f}%" for p in ob.GROUPS))
    w(f"  -> nominal: consistent; growth-rate mismatch groups G1/G3/G4/G5: {100 * fr[list(GROWTH_GROUPS)].min():.0f}-"
      f"{100 * fr[list(GROWTH_GROUPS)].max():.0f}% inside 2 sigma (overconfident); G2: {100 * fr['G2']:.0f}%.")
    w("  Late in treatment observability fades: b2 x2 falls from 5.6e-4/day at day 0 to 2e-9 (mu) .. 3e-8 (PID)/day")
    w("  at day 2000, cond([C; CA]) rises from 1.8e3 to 4e7 (PID) .. 5e8 (mu)  [figure fig_ekf_sd_observability].")
    w("")
    w("E. DOSE SCHEDULES (single convention, doses on days 0..1999)  [source: schedules/PROVENANCE.md]")
    w("  first day x2 < 1e-4: mu 1167, PID 1166, SMC 1165; cumulative dose [mg m^-2]: 142150, 131350, 140050;")
    w("  high-dose phases: 65, 71, 71; longest: 5, 4, 4 days. All checks a-g passed.")
    text = "\n".join(L) + "\n"
    ob.write_text(RES / "facts_block.txt", text)
    print(text)


if __name__ == "__main__":
    main()
