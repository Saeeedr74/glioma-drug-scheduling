# FACTS_FINAL — Stage 1 (every number Stages 2-4 may use)

> Note: Section 3 below gives the maximum-dose baseline for the stop criterion x2 <= 1e-5. The paper uses the
> rule 'first day with x2 < 1e-4'; its values are in `results/tables/baseline_metrics_v2.json` (stop on day 236,
> cumulative dose 47200 mg m^-2, x3(T) = 0.476068, glial recovery on day 664).

Written by `make_facts.py` from files on disk only. Metric conventions: horizon days 0..2000 of treatment (day 0 = simulation day 600); threshold 1e-4; cumulative dose = sum of daily doses on days 0..1999 (DECISIONS D-07); maintenance phases = complete phases between two high-dose phases (D-08). 1e-5 appears only as a simulation stop criterion (baseline, continuous SMC). Old-paper numbers are not used. Source notation `file [set/controller/group] column` refers to rows of `data/metrics_all.csv`.

## 1. Model, reference, uncertainty groups

| quantity | value | source |
|---|---|---|
| Om1 | 0.0068 | glioma_paper/model.py NOMINAL |
| Om2 | 0.012 | glioma_paper/model.py NOMINAL |
| b1 | 0.018 | glioma_paper/model.py NOMINAL |
| b2 | 0.0018 | glioma_paper/model.py NOMINAL |
| p1 | 4.7000e-08 | glioma_paper/model.py NOMINAL |
| p2 | 4.7000e-05 | glioma_paper/model.py NOMINAL |
| p3 | 4.7000e-08 | glioma_paper/model.py NOMINAL |
| a1 | 1 | glioma_paper/model.py NOMINAL |
| a2 | 1 | glioma_paper/model.py NOMINAL |
| a3 | 1 | glioma_paper/model.py NOMINAL |
| alpha | 2 | glioma_paper/model.py NOMINAL |
| zeta | 0.2 | glioma_paper/model.py NOMINAL |
| dt (day) | 1 | glioma_paper/model.py DT |
| pre-conditioning length (d) | 600 | model.py T_START |
| treatment horizon T (d) | 2000 | model.py T_REF |
| reference a | 7.5 | model.py A_REF |
| reference b | 0 | model.py B_REF |
| dose levels (mg m^-2 day^-1) | 50 (maintenance), 200 (high-dose) | tables/T_design_pid.csv U_MIN/U_MAX |
| group nominal |  | model.py GROUPS |
| group G1 | Om1 -20%, Om2 +20% | model.py GROUPS |
| group G2 | b1 +50%, b2 -50% | model.py GROUPS |
| group G3 | p1 +15%, p2 -15% | model.py GROUPS |
| group G4 | a1 +20%, a2 +20% | model.py GROUPS |
| group G5 | zeta +20% | model.py GROUPS |

Integrator RK4, x2 clipped at 0 after each step; H(0) = 0.5; y0 = [1, 1e-3, 1, 0].

## 2. Pre-conditioning

| quantity | value | source |
|---|---|---|
| x_pre = x(600), x1 | 0.670934 | data/unify_crosschecks.json x_pre[0]; data/preconditioning.csv sim_day 600 |
| x_pre = x(600), x2 | 0.313731 | data/unify_crosschecks.json x_pre[1]; data/preconditioning.csv sim_day 600 |
| x_pre = x(600), x3 | 0.517817 | data/unify_crosschecks.json x_pre[2]; data/preconditioning.csv sim_day 600 |
| x_pre = x(600), x4 | 0 | data/unify_crosschecks.json x_pre[3]; data/preconditioning.csv sim_day 600 |
| max deviation from [0.670934, 0.313731, 0.517817, 0] | 4.6517e-07 | unify_crosschecks.json |

## 3. Maximum-dose baseline (u = 200 until the x2 <= 1e-5 stop criterion, then u = 0)

| quantity | value | source |
|---|---|---|
| first day x2 < 1e-4 (d) | 236 | data/metrics_all.csv [baseline/max_dose/nominal] t_x2_first |
| day from which x2 < 1e-4 to day 2000 (d) | 236 | data/metrics_all.csv [baseline/max_dose/nominal] t_x2_perm |
| day from which x1 >= 0.99 to day 2000 (d) | 666 | data/metrics_all.csv [baseline/max_dose/nominal] t_x1_perm |
| x1(T) | 0.999999 | data/metrics_all.csv [baseline/max_dose/nominal] x1_T |
| x2(T) | 0 | data/metrics_all.csv [baseline/max_dose/nominal] x2_T |
| x3(T) | 0.475113 | data/metrics_all.csv [baseline/max_dose/nominal] x3_T |
| min x3 | 0.475113 | data/metrics_all.csv [baseline/max_dose/nominal] x3_min |
| max x4 | 1000 | data/metrics_all.csv [baseline/max_dose/nominal] x4_max |
| cumulative dose, sum days 0..1999 (mg m^-2) | 59800 | data/metrics_all.csv [baseline/max_dose/nominal] cum_dose |
| mean daily dose (mg m^-2 day^-1) | 29.9 | data/metrics_all.csv [baseline/max_dose/nominal] mean_dose |
| min u | 0 | data/metrics_all.csv [baseline/max_dose/nominal] u_min |
| max u | 200 | data/metrics_all.csv [baseline/max_dose/nominal] u_max |
| longest high-dose phase (d) | 299 | data/metrics_all.csv [baseline/max_dose/nominal] hi_max |

x2(T) = 0 is set by the stop rule of the code (D-06).

## 4. Gain-scheduled PID, nominal discrete

| quantity | value | source |
|---|---|---|
| first day x2 < 1e-4 (d) | 1166 | data/metrics_all.csv [nominal_discrete/PID/nominal] t_x2_first |
| day from which x2 < 1e-4 to day 2000 (d) | 1166 | data/metrics_all.csv [nominal_discrete/PID/nominal] t_x2_perm |
| day from which x1 >= 0.99 to day 2000 (d) | 1042 | data/metrics_all.csv [nominal_discrete/PID/nominal] t_x1_perm |
| x1(T) | 0.999065 | data/metrics_all.csv [nominal_discrete/PID/nominal] x1_T |
| x2(T) | 1.5839e-05 | data/metrics_all.csv [nominal_discrete/PID/nominal] x2_T |
| x3(T) | 0.398091 | data/metrics_all.csv [nominal_discrete/PID/nominal] x3_T |
| min x3 | 0.398091 | data/metrics_all.csv [nominal_discrete/PID/nominal] x3_min |
| max x4 | 609.381 | data/metrics_all.csv [nominal_discrete/PID/nominal] x4_max |
| cumulative dose, sum days 0..1999 (mg m^-2) | 131350 | data/metrics_all.csv [nominal_discrete/PID/nominal] cum_dose |
| mean daily dose (mg m^-2 day^-1) | 65.675 | data/metrics_all.csv [nominal_discrete/PID/nominal] mean_dose |
| min u | 50 | data/metrics_all.csv [nominal_discrete/PID/nominal] u_min |
| max u | 200 | data/metrics_all.csv [nominal_discrete/PID/nominal] u_max |
| high-dose phases | 71 | data/metrics_all.csv [nominal_discrete/PID/nominal] n_hi |
| mean high-dose phase (d) | 2.94366 | data/metrics_all.csv [nominal_discrete/PID/nominal] hi_mean |
| longest high-dose phase (d) | 4 | data/metrics_all.csv [nominal_discrete/PID/nominal] hi_max |
| high-dose phases > 5 d | 0 | data/metrics_all.csv [nominal_discrete/PID/nominal] n_hi_gt5 |
| days at u = 200 | 209 | data/metrics_all.csv [nominal_discrete/PID/nominal] hi_days |
| last high-dose day | 1266 | data/metrics_all.csv [nominal_discrete/PID/nominal] last_hi_day |
| complete maintenance phases | 70 | data/metrics_all.csv [nominal_discrete/PID/nominal] n_lo_between |
| mean complete maintenance phase (d) | 15.1143 | data/metrics_all.csv [nominal_discrete/PID/nominal] lo_between_mean |
| shortest complete maintenance phase (d) | 8 | data/metrics_all.csv [nominal_discrete/PID/nominal] lo_between_min |
| longest complete maintenance phase (d) | 35 | data/metrics_all.csv [nominal_discrete/PID/nominal] lo_between_max |
| RMSE(x2 - r)/x2_pre | 0.00322649 | data/metrics_all.csv [nominal_discrete/PID/nominal] rmse_over_x2pre |

## 5. SMC, nominal discrete

| quantity | value | source |
|---|---|---|
| first day x2 < 1e-4 (d) | 1165 | data/metrics_all.csv [nominal_discrete/SMC/nominal] t_x2_first |
| day from which x2 < 1e-4 to day 2000 (d) | 1165 | data/metrics_all.csv [nominal_discrete/SMC/nominal] t_x2_perm |
| day from which x1 >= 0.99 to day 2000 (d) | 1042 | data/metrics_all.csv [nominal_discrete/SMC/nominal] t_x1_perm |
| x1(T) | 0.999065 | data/metrics_all.csv [nominal_discrete/SMC/nominal] x1_T |
| x2(T) | 2.0502e-06 | data/metrics_all.csv [nominal_discrete/SMC/nominal] x2_T |
| x3(T) | 0.3979 | data/metrics_all.csv [nominal_discrete/SMC/nominal] x3_T |
| min x3 | 0.3979 | data/metrics_all.csv [nominal_discrete/SMC/nominal] x3_min |
| max x4 | 670.782 | data/metrics_all.csv [nominal_discrete/SMC/nominal] x4_max |
| cumulative dose, sum days 0..1999 (mg m^-2) | 140050 | data/metrics_all.csv [nominal_discrete/SMC/nominal] cum_dose |
| mean daily dose (mg m^-2 day^-1) | 70.025 | data/metrics_all.csv [nominal_discrete/SMC/nominal] mean_dose |
| min u | 50 | data/metrics_all.csv [nominal_discrete/SMC/nominal] u_min |
| max u | 200 | data/metrics_all.csv [nominal_discrete/SMC/nominal] u_max |
| high-dose phases | 71 | data/metrics_all.csv [nominal_discrete/SMC/nominal] n_hi |
| mean high-dose phase (d) | 3.76056 | data/metrics_all.csv [nominal_discrete/SMC/nominal] hi_mean |
| longest high-dose phase (d) | 4 | data/metrics_all.csv [nominal_discrete/SMC/nominal] hi_max |
| high-dose phases > 5 d | 0 | data/metrics_all.csv [nominal_discrete/SMC/nominal] n_hi_gt5 |
| days at u = 200 | 267 | data/metrics_all.csv [nominal_discrete/SMC/nominal] hi_days |
| last high-dose day | 1740 | data/metrics_all.csv [nominal_discrete/SMC/nominal] last_hi_day |
| complete maintenance phases | 70 | data/metrics_all.csv [nominal_discrete/SMC/nominal] n_lo_between |
| mean complete maintenance phase (d) | 21.0571 | data/metrics_all.csv [nominal_discrete/SMC/nominal] lo_between_mean |
| shortest complete maintenance phase (d) | 15 | data/metrics_all.csv [nominal_discrete/SMC/nominal] lo_between_min |
| longest complete maintenance phase (d) | 69 | data/metrics_all.csv [nominal_discrete/SMC/nominal] lo_between_max |
| RMSE(x2 - r)/x2_pre | 0.00254078 | data/metrics_all.csv [nominal_discrete/SMC/nominal] rmse_over_x2pre |

## 6. mu-synthesis, nominal discrete

| quantity | value | source |
|---|---|---|
| first day x2 < 1e-4 (d) | 1167 | data/metrics_all.csv [nominal_discrete/mu/nominal] t_x2_first |
| day from which x2 < 1e-4 to day 2000 (d) | 1167 | data/metrics_all.csv [nominal_discrete/mu/nominal] t_x2_perm |
| day from which x1 >= 0.99 to day 2000 (d) | 1046 | data/metrics_all.csv [nominal_discrete/mu/nominal] t_x1_perm |
| x1(T) | 0.999028 | data/metrics_all.csv [nominal_discrete/mu/nominal] x1_T |
| x2(T) | 1.2517e-06 | data/metrics_all.csv [nominal_discrete/mu/nominal] x2_T |
| x3(T) | 0.393602 | data/metrics_all.csv [nominal_discrete/mu/nominal] x3_T |
| min x3 | 0.393602 | data/metrics_all.csv [nominal_discrete/mu/nominal] x3_min |
| max x4 | 727.527 | data/metrics_all.csv [nominal_discrete/mu/nominal] x4_max |
| cumulative dose, sum days 0..1999 (mg m^-2) | 142150 | data/metrics_all.csv [nominal_discrete/mu/nominal] cum_dose |
| mean daily dose (mg m^-2 day^-1) | 71.075 | data/metrics_all.csv [nominal_discrete/mu/nominal] mean_dose |
| min u | 50 | data/metrics_all.csv [nominal_discrete/mu/nominal] u_min |
| max u | 200 | data/metrics_all.csv [nominal_discrete/mu/nominal] u_max |
| high-dose phases | 65 | data/metrics_all.csv [nominal_discrete/mu/nominal] n_hi |
| mean high-dose phase (d) | 4.32308 | data/metrics_all.csv [nominal_discrete/mu/nominal] hi_mean |
| longest high-dose phase (d) | 5 | data/metrics_all.csv [nominal_discrete/mu/nominal] hi_max |
| high-dose phases > 5 d | 0 | data/metrics_all.csv [nominal_discrete/mu/nominal] n_hi_gt5 |
| days at u = 200 | 281 | data/metrics_all.csv [nominal_discrete/mu/nominal] hi_days |
| last high-dose day | 1836 | data/metrics_all.csv [nominal_discrete/mu/nominal] last_hi_day |
| complete maintenance phases | 64 | data/metrics_all.csv [nominal_discrete/mu/nominal] n_lo_between |
| mean complete maintenance phase (d) | 24.2031 | data/metrics_all.csv [nominal_discrete/mu/nominal] lo_between_mean |
| shortest complete maintenance phase (d) | 18 | data/metrics_all.csv [nominal_discrete/mu/nominal] lo_between_min |
| longest complete maintenance phase (d) | 54 | data/metrics_all.csv [nominal_discrete/mu/nominal] lo_between_max |
| RMSE(x2 - r)/x2_pre | 0.00612324 | data/metrics_all.csv [nominal_discrete/mu/nominal] rmse_over_x2pre |

## 6a. Design parameters (PID, SMC, mu)

| quantity | value | source |
|---|---|---|
| PID_A_REF | 7.5 | tables/T_design_pid.csv PID_A_REF |
| PID_B_REF | 0 | tables/T_design_pid.csv PID_B_REF |
| U_MIN | 50 | tables/T_design_pid.csv U_MIN |
| U_MAX | 200 | tables/T_design_pid.csv U_MAX |
| PID_XGAIN | 1.0000e+09 | tables/T_design_pid.csv PID_XGAIN |
| PID_M_P | 7.50764 | tables/T_design_pid.csv PID_M_P |
| PID_M_I | 0.537876 | tables/T_design_pid.csv PID_M_I |
| PID_M_D | 0.740697 | tables/T_design_pid.csv PID_M_D |
| PID_YGAIN | 6.7641e+05 | tables/T_design_pid.csv PID_YGAIN |
| PID_A_GAIN | 14.9047 | tables/T_design_pid.csv PID_A_GAIN |
| PID_B_GAIN | 0.375994 | tables/T_design_pid.csv PID_B_GAIN |
| KP_INIT | 7.5076e+09 | tables/T_design_pid.csv KP_INIT |
| KI_INIT | 5.3788e+08 | tables/T_design_pid.csv KI_INIT |
| KD_INIT | 7.4070e+08 | tables/T_design_pid.csv KD_INIT |
| KP_FINAL | 11099.2 | tables/T_design_pid.csv KP_FINAL |
| KI_FINAL | 795.188 | tables/T_design_pid.csv KI_FINAL |
| KD_FINAL | 1095.04 | tables/T_design_pid.csv KD_FINAL |
| PID_M_P search range | 1 .. 100 (log), hand design 10 | tables/T_design_pid.csv |
| PID_M_I search range | 0.18 .. 18 (log), hand design 1.8 | tables/T_design_pid.csv |
| PID_M_D search range | 0.33 .. 33 (log), hand design 3.3 | tables/T_design_pid.csv |
| PID_YGAIN search range | 200000 .. 2.0000e+07 (log), hand design 2000000 | tables/T_design_pid.csv |
| PID_A_GAIN search range | 2 .. 15 (log), hand design 7 | tables/T_design_pid.csv |
| PID_B_GAIN search range | 0 .. 1 (lin), hand design 0 | tables/T_design_pid.csv |

| quantity | value | source |
|---|---|---|
| SMC_A_REF | 7.5 | tables/T_design_smc.csv SMC_A_REF |
| SMC_B_REF | 0 | tables/T_design_smc.csv SMC_B_REF |
| U_MIN | 50 | tables/T_design_smc.csv U_MIN |
| U_MAX | 200 | tables/T_design_smc.csv U_MAX |
| LAMBDA_P | 1.88136 | tables/T_design_smc.csv LAMBDA_P |
| LAMBDA_D | 9.6192 | tables/T_design_smc.csv LAMBDA_D |
| LAMBDA_I | 2.9471e-05 | tables/T_design_smc.csv LAMBDA_I |
| DELTA_INIT | 0.0136647 | tables/T_design_smc.csv DELTA_INIT |
| DELTA_FINAL | 2.6767e-07 | tables/T_design_smc.csv DELTA_FINAL |
| A_DELTA | 7.82414 | tables/T_design_smc.csv A_DELTA |
| B_DELTA | 0.103646 | tables/T_design_smc.csv B_DELTA |
| GAMMA_1 | 1 | tables/T_design_smc.csv GAMMA_1 |
| GAMMA_2 | 1 | tables/T_design_smc.csv GAMMA_2 |
| LAMBDA_P search range | 0.1 .. 10 (log), hand design 1.5 | tables/T_design_smc.csv |
| LAMBDA_D search range | 0.1 .. 10 (log), hand design 2.5 | tables/T_design_smc.csv |
| LAMBDA_I search range | 1.0000e-06 .. 0.01 (log), hand design 1.0000e-04 | tables/T_design_smc.csv |
| DELTA_INIT search range | 0.001 .. 0.1 (log), hand design 0.01 | tables/T_design_smc.csv |
| DELTA_FINAL search range | 1.0000e-09 .. 0.01 (log), hand design 1.0000e-09 | tables/T_design_smc.csv |
| A_DELTA search range | 2 .. 15 (log), hand design 7.5 | tables/T_design_smc.csv |
| B_DELTA search range | 0 .. 1 (lin), hand design 0 | tables/T_design_smc.csv |

| quantity | value | source |
|---|---|---|
| M | 1.83935 | tables/T_design_mu.csv M (from data/mu_design.csv, mu_scalars.csv) |
| A | 0.00704063 | tables/T_design_mu.csv A (from data/mu_design.csv, mu_scalars.csv) |
| wB | 0.311074 | tables/T_design_mu.csv wB (from data/mu_design.csv, mu_scalars.csv) |
| Wp_gain | 0.594511 | tables/T_design_mu.csv Wp_gain (from data/mu_design.csv, mu_scalars.csv) |
| k | 0.12858 | tables/T_design_mu.csv k (from data/mu_design.csv, mu_scalars.csv) |
| wz | 0.434664 | tables/T_design_mu.csv wz (from data/mu_design.csv, mu_scalars.csv) |
| wp | 6.74273 | tables/T_design_mu.csv wp (from data/mu_design.csv, mu_scalars.csv) |
| a_reference | 7.5 | tables/T_design_mu.csv a_reference (from data/mu_design.csv, mu_scalars.csv) |
| u_min | 50.0 | tables/T_design_mu.csv u_min (from data/mu_design.csv, mu_scalars.csv) |
| u_max | 200.0 | tables/T_design_mu.csv u_max (from data/mu_design.csv, mu_scalars.csv) |
| dose_threshold | 125.0 | tables/T_design_mu.csv dose_threshold (from data/mu_design.csv, mu_scalars.csv) |
| redesign_rel_threshold | 0.1 | tables/T_design_mu.csv redesign_rel_threshold (from data/mu_design.csv, mu_scalars.csv) |
| redesign_min_gap_days | 5.0 | tables/T_design_mu.csv redesign_min_gap_days (from data/mu_design.csv, mu_scalars.csv) |
| redesign_x2_stop | 0.0001 | tables/T_design_mu.csv redesign_x2_stop (from data/mu_design.csv, mu_scalars.csv) |
| cfg.musyn_maxiter | 8.0 | tables/T_design_mu.csv cfg.musyn_maxiter (from data/mu_design.csv, mu_scalars.csv) |
| cfg.mu_max_accept | 10.0 | tables/T_design_mu.csv cfg.mu_max_accept (from data/mu_design.csv, mu_scalars.csv) |
| cfg.aw_beta | 1.0 | tables/T_design_mu.csv cfg.aw_beta (from data/mu_design.csv, mu_scalars.csv) |
| S.xlin_floor | 0.001 | tables/T_design_mu.csv S.xlin_floor (from data/mu_design.csv, mu_scalars.csv) |
| cfg.xo_floor | 1e-06 | tables/T_design_mu.csv cfg.xo_floor (from data/mu_design.csv, mu_scalars.csv) |
| mu_peak_initial | 0.465469648545334 | tables/T_design_mu.csv mu_peak_initial (from data/mu_design.csv, mu_scalars.csv) |
| metrics_discrete.mu_max | 0.4655715730502769 | tables/T_design_mu.csv metrics_discrete.mu_max (from data/mu_design.csv, mu_scalars.csv) |
| robstab_lower | 6.38617796620214 | tables/T_design_mu.csv robstab_lower (from data/mu_design.csv, mu_scalars.csv) |
| robstab_upper | 6.39895032324947 | tables/T_design_mu.csv robstab_upper (from data/mu_design.csv, mu_scalars.csv) |
| Wi_low_freq_percent | 12.858 | tables/T_design_mu.csv Wi_low_freq_percent (from data/mu_design.csv, mu_scalars.csv) |
| Wi_high_freq_percent | 199.459864032908 | tables/T_design_mu.csv Wi_high_freq_percent (from data/mu_design.csv, mu_scalars.csv) |
| metrics_discrete.n_resyn | 60.0 | tables/T_design_mu.csv metrics_discrete.n_resyn (from data/mu_design.csv, mu_scalars.csv) |
| metrics_discrete.n_fail | 0.0 | tables/T_design_mu.csv metrics_discrete.n_fail (from data/mu_design.csv, mu_scalars.csv) |
| search_constraint | wp >= 4*wz & Wp_gain <= 1.2*M | tables/T_design_mu.csv search_constraint (from data/mu_design.csv, mu_scalars.csv) |
| M search range | 1.05 .. 8 (log) | tables/T_design_mu.csv |
| A search range | 5.0000e-05 .. 0.1 (log) | tables/T_design_mu.csv |
| wB search range | 0.001 .. 0.6 (log) | tables/T_design_mu.csv |
| Wp_gain search range | 0.3 .. 5 (log) | tables/T_design_mu.csv |
| k search range | 0.02 .. 1 (log) | tables/T_design_mu.csv |
| wz search range | 0.01 .. 0.5 (log) | tables/T_design_mu.csv |
| wp search range | 0.5 .. 200 (log) | tables/T_design_mu.csv |
| x2_pre (mu file) | 0.313731 | data/mu_design.csv x2_pre |

## 7. Continuous dosing (no continuous PID in the final code, D-04)

| quantity | value | source |
|---|---|---|
| first day x2 < 1e-4 (d) | 1204 | data/metrics_all.csv [nominal_continuous/SMC/nominal] t_x2_first |
| day from which x2 < 1e-4 to day 2000 (d) | 1204 | data/metrics_all.csv [nominal_continuous/SMC/nominal] t_x2_perm |
| day from which x1 >= 0.99 to day 2000 (d) | 1040 | data/metrics_all.csv [nominal_continuous/SMC/nominal] t_x1_perm |
| x1(T) | 0.999968 | data/metrics_all.csv [nominal_continuous/SMC/nominal] x1_T |
| x2(T) | 0 | data/metrics_all.csv [nominal_continuous/SMC/nominal] x2_T |
| x3(T) | 0.399138 | data/metrics_all.csv [nominal_continuous/SMC/nominal] x3_T |
| min x3 | 0.399138 | data/metrics_all.csv [nominal_continuous/SMC/nominal] x3_min |
| max x4 | 1000 | data/metrics_all.csv [nominal_continuous/SMC/nominal] x4_max |
| cumulative dose, sum days 0..1999 (mg m^-2) | 1.0234e+05 | data/metrics_all.csv [nominal_continuous/SMC/nominal] cum_dose |
| cumulative dose, trapz days 0..2000 (mg m^-2) | 1.0233e+05 | data/metrics_all.csv [nominal_continuous/SMC/nominal] cum_dose_trapz_0_2000 |
| mean daily dose (mg m^-2 day^-1) | 51.1696 | data/metrics_all.csv [nominal_continuous/SMC/nominal] mean_dose |
| min u | 0 | data/metrics_all.csv [nominal_continuous/SMC/nominal] u_min |
| max u | 200 | data/metrics_all.csv [nominal_continuous/SMC/nominal] u_max |
| RMSE(x2 - r)/x2_pre | 0.0135741 | data/metrics_all.csv [nominal_continuous/SMC/nominal] rmse_over_x2pre |
| simulation stop day (x2 <= 1e-5 stop criterion) | 1267 | data/metrics_all.csv [nominal_continuous/SMC/nominal] stop_day |

Continuous SMC stops at x2 <= 1e-5 (day 1267), then x2 := 0 and u = 0 (D-05).

| quantity | value | source |
|---|---|---|
| first day x2 < 1e-4 (d) | 1169 | data/metrics_all.csv [nominal_continuous/mu/nominal] t_x2_first |
| day from which x2 < 1e-4 to day 2000 (d) | 1169 | data/metrics_all.csv [nominal_continuous/mu/nominal] t_x2_perm |
| day from which x1 >= 0.99 to day 2000 (d) | 1048 | data/metrics_all.csv [nominal_continuous/mu/nominal] t_x1_perm |
| x1(T) | 0.9989 | data/metrics_all.csv [nominal_continuous/mu/nominal] x1_T |
| x2(T) | 6.4144e-07 | data/metrics_all.csv [nominal_continuous/mu/nominal] x2_T |
| x3(T) | 0.392609 | data/metrics_all.csv [nominal_continuous/mu/nominal] x3_T |
| min x3 | 0.392609 | data/metrics_all.csv [nominal_continuous/mu/nominal] x3_min |
| max x4 | 564.063 | data/metrics_all.csv [nominal_continuous/mu/nominal] x4_max |
| cumulative dose, sum days 0..1999 (mg m^-2) | 1.4503e+05 | data/metrics_all.csv [nominal_continuous/mu/nominal] cum_dose |
| cumulative dose, trapz days 0..2000 (mg m^-2) | 1.4506e+05 | data/metrics_all.csv [nominal_continuous/mu/nominal] cum_dose_trapz_0_2000 |
| mean daily dose (mg m^-2 day^-1) | 72.516 | data/metrics_all.csv [nominal_continuous/mu/nominal] mean_dose |
| min u | 0 | data/metrics_all.csv [nominal_continuous/mu/nominal] u_min |
| max u | 160.814 | data/metrics_all.csv [nominal_continuous/mu/nominal] u_max |
| RMSE(x2 - r)/x2_pre | 0.00618918 | data/metrics_all.csv [nominal_continuous/mu/nominal] rmse_over_x2pre |
| MATLAB cumulative dose (trapz, 0..2000) | 1.4506e+05 | data/mu_scalars.csv metrics_continuous.cum_dose |
| replay check max |dx2|/max(x2,1e-12) | 5.5549e-15 | unify_crosschecks.json mu_continuous |

## 8. Standard 5/28 protocol (dose 200 and 400) vs SMC

| quantity | value | source |
|---|---|---|
| x1(T) | 0.00257403 | data/metrics_all.csv [standard_protocol/standard_5_28_dose200/nominal] x1_T |
| x2(T) | 0.531354 | data/metrics_all.csv [standard_protocol/standard_5_28_dose200/nominal] x2_T |
| x3(T) | 0.134105 | data/metrics_all.csv [standard_protocol/standard_5_28_dose200/nominal] x3_T |
| min x3 | 0.134105 | data/metrics_all.csv [standard_protocol/standard_5_28_dose200/nominal] x3_min |
| max x4 | 634.461 | data/metrics_all.csv [standard_protocol/standard_5_28_dose200/nominal] x4_max |
| cumulative dose, sum days 0..1999 (mg m^-2) | 72000 | data/metrics_all.csv [standard_protocol/standard_5_28_dose200/nominal] cum_dose |
| mean daily dose (mg m^-2 day^-1) | 36 | data/metrics_all.csv [standard_protocol/standard_5_28_dose200/nominal] mean_dose |
| min u | 0 | data/metrics_all.csv [standard_protocol/standard_5_28_dose200/nominal] u_min |
| max u | 200 | data/metrics_all.csv [standard_protocol/standard_5_28_dose200/nominal] u_max |
| high-dose phases | 72 | data/metrics_all.csv [standard_protocol/standard_5_28_dose200/nominal] n_hi |
| longest high-dose phase (d) | 5 | data/metrics_all.csv [standard_protocol/standard_5_28_dose200/nominal] hi_max |
| RMSE(x2 - r)/x2_pre | 1.50613 | data/metrics_all.csv [standard_protocol/standard_5_28_dose200/nominal] rmse_over_x2pre |

| quantity | value | source |
|---|---|---|
| first day x2 < 1e-4 (d) | 1272 | data/metrics_all.csv [standard_protocol/standard_5_28_dose400/nominal] t_x2_first |
| day from which x2 < 1e-4 to day 2000 (d) | 1293 | data/metrics_all.csv [standard_protocol/standard_5_28_dose400/nominal] t_x2_perm |
| day from which x1 >= 0.99 to day 2000 (d) | 1056 | data/metrics_all.csv [standard_protocol/standard_5_28_dose400/nominal] t_x1_perm |
| x1(T) | 0.998693 | data/metrics_all.csv [standard_protocol/standard_5_28_dose400/nominal] x1_T |
| x2(T) | 8.2482e-07 | data/metrics_all.csv [standard_protocol/standard_5_28_dose400/nominal] x2_T |
| x3(T) | 0.423433 | data/metrics_all.csv [standard_protocol/standard_5_28_dose400/nominal] x3_T |
| min x3 | 0.423433 | data/metrics_all.csv [standard_protocol/standard_5_28_dose400/nominal] x3_min |
| max x4 | 1268.92 | data/metrics_all.csv [standard_protocol/standard_5_28_dose400/nominal] x4_max |
| cumulative dose, sum days 0..1999 (mg m^-2) | 144000 | data/metrics_all.csv [standard_protocol/standard_5_28_dose400/nominal] cum_dose |
| mean daily dose (mg m^-2 day^-1) | 72 | data/metrics_all.csv [standard_protocol/standard_5_28_dose400/nominal] mean_dose |
| min u | 0 | data/metrics_all.csv [standard_protocol/standard_5_28_dose400/nominal] u_min |
| max u | 400 | data/metrics_all.csv [standard_protocol/standard_5_28_dose400/nominal] u_max |
| high-dose phases | 72 | data/metrics_all.csv [standard_protocol/standard_5_28_dose400/nominal] n_hi |
| longest high-dose phase (d) | 5 | data/metrics_all.csv [standard_protocol/standard_5_28_dose400/nominal] hi_max |
| RMSE(x2 - r)/x2_pre | 0.0348472 | data/metrics_all.csv [standard_protocol/standard_5_28_dose400/nominal] rmse_over_x2pre |


## 9. Table 6, protocol E1 (nominal discrete schedule replayed on G1..G5 from nominal x_pre)

| quantity | value | source |
|---|---|---|
| PID nominal x2(T) | 1.5839e-05 | data/metrics_all.csv [E1/PID/nominal] x2_T |
| PID nominal first day x2 < 1e-4 | 1166 | data/metrics_all.csv [E1/PID/nominal] t_x2_first |
| PID nominal t_x1_perm | 1042 | data/metrics_all.csv [E1/PID/nominal] t_x1_perm |
| PID nominal x3(T) | 0.398091 | data/metrics_all.csv [E1/PID/nominal] x3_T |
| PID G1 x2(T) | 0.00222812 | data/metrics_all.csv [E1/PID/G1] x2_T |
| PID G1 t_x1_perm | 1560 | data/metrics_all.csv [E1/PID/G1] t_x1_perm |
| PID G1 x3(T) | 0.338295 | data/metrics_all.csv [E1/PID/G1] x3_T |
| PID G2 x2(T) | 8.8382e-05 | data/metrics_all.csv [E1/PID/G2] x2_T |
| PID G2 first day x2 < 1e-4 | 1810 | data/metrics_all.csv [E1/PID/G2] t_x2_first |
| PID G2 t_x1_perm | 1191 | data/metrics_all.csv [E1/PID/G2] t_x1_perm |
| PID G2 x3(T) | 0.305961 | data/metrics_all.csv [E1/PID/G2] x3_T |
| PID G3 x2(T) | 0.00169357 | data/metrics_all.csv [E1/PID/G3] x2_T |
| PID G3 t_x1_perm | 1358 | data/metrics_all.csv [E1/PID/G3] t_x1_perm |
| PID G3 x3(T) | 0.369383 | data/metrics_all.csv [E1/PID/G3] x3_T |
| PID G4 x2(T) | 0.0023542 | data/metrics_all.csv [E1/PID/G4] x2_T |
| PID G4 t_x1_perm | 1380 | data/metrics_all.csv [E1/PID/G4] t_x1_perm |
| PID G4 x3(T) | 0.372545 | data/metrics_all.csv [E1/PID/G4] x3_T |
| PID G5 x2(T) | 0.00282804 | data/metrics_all.csv [E1/PID/G5] x2_T |
| PID G5 t_x1_perm | 1437 | data/metrics_all.csv [E1/PID/G5] t_x1_perm |
| PID G5 x3(T) | 0.368305 | data/metrics_all.csv [E1/PID/G5] x3_T |
| PID sensitivity factor | 116.069 | data/metrics_all.csv [E1_summary/PID/mean_G1_G5] sensitivity_factor |
| PID mean x2(T) G1..G5 | 0.00183846 | data/metrics_all.csv [E1_summary/PID/mean_G1_G5] x2_T_mean_G1_G5 |
| PID worst x2(T) | 0.00282804 | data/metrics_all.csv [E1_summary/PID/mean_G1_G5] x2_T_worst |
| PID worst group | G5 | data/metrics_all.csv [E1_summary/PID/mean_G1_G5] worst_group |
| PID groups reaching 1e-4 | 1 | data/metrics_all.csv [E1_summary/PID/mean_G1_G5] n_groups_reaching_1e-4 |
| SMC nominal x2(T) | 2.0502e-06 | data/metrics_all.csv [E1/SMC/nominal] x2_T |
| SMC nominal first day x2 < 1e-4 | 1165 | data/metrics_all.csv [E1/SMC/nominal] t_x2_first |
| SMC nominal t_x1_perm | 1042 | data/metrics_all.csv [E1/SMC/nominal] t_x1_perm |
| SMC nominal x3(T) | 0.3979 | data/metrics_all.csv [E1/SMC/nominal] x3_T |
| SMC G1 x2(T) | 2.8842e-04 | data/metrics_all.csv [E1/SMC/G1] x2_T |
| SMC G1 t_x1_perm | 1505 | data/metrics_all.csv [E1/SMC/G1] t_x1_perm |
| SMC G1 x3(T) | 0.338604 | data/metrics_all.csv [E1/SMC/G1] x3_T |
| SMC G2 x2(T) | 1.1440e-05 | data/metrics_all.csv [E1/SMC/G2] x2_T |
| SMC G2 first day x2 < 1e-4 | 1324 | data/metrics_all.csv [E1/SMC/G2] t_x2_first |
| SMC G2 t_x1_perm | 1190 | data/metrics_all.csv [E1/SMC/G2] t_x1_perm |
| SMC G2 x3(T) | 0.306151 | data/metrics_all.csv [E1/SMC/G2] x3_T |
| SMC G3 x2(T) | 2.9797e-04 | data/metrics_all.csv [E1/SMC/G3] x2_T |
| SMC G3 t_x1_perm | 1352 | data/metrics_all.csv [E1/SMC/G3] t_x1_perm |
| SMC G3 x3(T) | 0.369567 | data/metrics_all.csv [E1/SMC/G3] x3_T |
| SMC G4 x2(T) | 4.2929e-04 | data/metrics_all.csv [E1/SMC/G4] x2_T |
| SMC G4 t_x1_perm | 1367 | data/metrics_all.csv [E1/SMC/G4] t_x1_perm |
| SMC G4 x3(T) | 0.37284 | data/metrics_all.csv [E1/SMC/G4] x3_T |
| SMC G5 x2(T) | 5.1489e-04 | data/metrics_all.csv [E1/SMC/G5] x2_T |
| SMC G5 t_x1_perm | 1402 | data/metrics_all.csv [E1/SMC/G5] t_x1_perm |
| SMC G5 x3(T) | 0.368817 | data/metrics_all.csv [E1/SMC/G5] x3_T |
| SMC sensitivity factor | 150.425 | data/metrics_all.csv [E1_summary/SMC/mean_G1_G5] sensitivity_factor |
| SMC mean x2(T) G1..G5 | 3.0840e-04 | data/metrics_all.csv [E1_summary/SMC/mean_G1_G5] x2_T_mean_G1_G5 |
| SMC worst x2(T) | 5.1489e-04 | data/metrics_all.csv [E1_summary/SMC/mean_G1_G5] x2_T_worst |
| SMC worst group | G5 | data/metrics_all.csv [E1_summary/SMC/mean_G1_G5] worst_group |
| SMC groups reaching 1e-4 | 1 | data/metrics_all.csv [E1_summary/SMC/mean_G1_G5] n_groups_reaching_1e-4 |
| mu nominal x2(T) | 1.2517e-06 | data/metrics_all.csv [E1/mu/nominal] x2_T |
| mu nominal first day x2 < 1e-4 | 1167 | data/metrics_all.csv [E1/mu/nominal] t_x2_first |
| mu nominal t_x1_perm | 1046 | data/metrics_all.csv [E1/mu/nominal] t_x1_perm |
| mu nominal x3(T) | 0.393602 | data/metrics_all.csv [E1/mu/nominal] x3_T |
| mu G1 x2(T) | 1.7623e-04 | data/metrics_all.csv [E1/mu/G1] x2_T |
| mu G1 t_x1_perm | 1510 | data/metrics_all.csv [E1/mu/G1] t_x1_perm |
| mu G1 x3(T) | 0.334034 | data/metrics_all.csv [E1/mu/G1] x3_T |
| mu G2 x2(T) | 6.9771e-06 | data/metrics_all.csv [E1/mu/G2] x2_T |
| mu G2 first day x2 < 1e-4 | 1326 | data/metrics_all.csv [E1/mu/G2] t_x2_first |
| mu G2 t_x1_perm | 1196 | data/metrics_all.csv [E1/mu/G2] t_x1_perm |
| mu G2 x3(T) | 0.301564 | data/metrics_all.csv [E1/mu/G2] x3_T |
| mu G3 x2(T) | 1.9590e-04 | data/metrics_all.csv [E1/mu/G3] x2_T |
| mu G3 t_x1_perm | 1357 | data/metrics_all.csv [E1/mu/G3] t_x1_perm |
| mu G3 x3(T) | 0.365414 | data/metrics_all.csv [E1/mu/G3] x3_T |
| mu G4 x2(T) | 2.8406e-04 | data/metrics_all.csv [E1/mu/G4] x2_T |
| mu G4 t_x1_perm | 1370 | data/metrics_all.csv [E1/mu/G4] t_x1_perm |
| mu G4 x3(T) | 0.368861 | data/metrics_all.csv [E1/mu/G4] x3_T |
| mu G5 x2(T) | 3.4133e-04 | data/metrics_all.csv [E1/mu/G5] x2_T |
| mu G5 t_x1_perm | 1406 | data/metrics_all.csv [E1/mu/G5] t_x1_perm |
| mu G5 x3(T) | 0.364599 | data/metrics_all.csv [E1/mu/G5] x3_T |
| mu sensitivity factor | 160.495 | data/metrics_all.csv [E1_summary/mu/mean_G1_G5] sensitivity_factor |
| mu mean x2(T) G1..G5 | 2.0090e-04 | data/metrics_all.csv [E1_summary/mu/mean_G1_G5] x2_T_mean_G1_G5 |
| mu worst x2(T) | 3.4133e-04 | data/metrics_all.csv [E1_summary/mu/mean_G1_G5] x2_T_worst |
| mu worst group | G5 | data/metrics_all.csv [E1_summary/mu/mean_G1_G5] worst_group |
| mu groups reaching 1e-4 | 1 | data/metrics_all.csv [E1_summary/mu/mean_G1_G5] n_groups_reaching_1e-4 |
| mu sensitivity factor (MATLAB) | 160.495 | data/mu_scalars.csv sensitivity_factor_E1 |

Ratios to nominal per group: tables/T6_uncertainty.csv ratio_to_nominal. A blank first day means x2 never fell below 1e-4.

## 10. Observer (EKF main, sampled-data Luenberger lsd comparison; not recomputed)

Source: `data/observer_facts_block.txt` (copy of observer\results\facts_block.txt; check f passed). Tables: tables/T_observer.csv, T_app_observer_noise.csv, T_app_observer_groups.csv, T_app_observer_bias.csv. Verbatim:

```
FACTS BLOCK - state observers for the glioma chemotherapy controllers (mu, PID, SMC)
Written 2026-11-16 11:16 by make_facts_block.py from the stored results of observer.py
(no simulation rerun, no setting changed).

ROLES FOR THE PAPER
  Main observer        : discrete extended Kalman filter (EKF) on [x1, x2].
  Comparison baseline  : corrected adaptive Luenberger observer, sampled-data form (lsd).
  Also reported (M1)   : the continuous-time Luenberger form with y held over each day (lzoh),
                         i.e. the form of the original task text; it shows a sampling-lag bias.

COMMON SETUP (source: observer.py SETTINGS, README.md section 4)
  y = x2 (x3 excluded: rank of [C; CA; CA^2] at x_pre with y = x2 is 2); x4_hat from the applied dose
  with nominal zeta = 0.2; nominal model in every observer; x2_hat(0) = y(0);
  x1_hat(0) = (1 +/- 0.2) x1(0). Measurement modes: M1 daily noise-free; M2 daily, sigma = 0.1;
  M3 every 7 or 28 days, sigma = 0.1; 20 seeds per noisy scenario.
  Luenberger: poles p_i = -max(2|Re lambda_i(A)|, 0.01/day), relinearisation trigger
  0.5% on x2_hat. EKF: Q = diag((q x_hat)^2), q = 0.001/day; R = (sigma x2)^2
  (M1: sigma_eff = 0.001); P0 = diag((delta x1_hat(0))^2, (sigma y0)^2).
  Convergence day = first day after which e1 = |x1_hat - x1|/x1 stays below 0.01.
  EKF 'updates' = Jacobian propagations / measurement updates; Luenberger = triggered relinearisations.
  Pairs 'a / b' below are delta = +0.2 / -0.2.  n.c. = not converged by day 2000.

A. MAIN NUMBERS - M1 (daily, noise-free), nominal plant  [source: results/summary_by_scenario.csv]
  [mu]                         EKF (main)              Luenberger lsd (baseline)   Luenberger lzoh (task form)
    convergence day (e1<1%)      247 / 361               271 / 269                   536 / 507
    e1 at day 2000               8.34e-08 / 1.54e-07     8.79e-06 / 8.79e-06         4.21e-04 / 4.21e-04
    RMS e1, days 0..2000         1.65e-02 / 2.28e-02     3.85e-02 / 3.99e-02         3.75e-02 / 4.28e-02
    RMS e1, days 100..2000       5.03e-03 / 9.03e-03     1.75e-02 / 1.78e-02         1.68e-02 / 2.22e-02
    e2 at day 2000               9.35e-11 / 1.72e-10     4.32e-06 / 4.32e-06         1.11e-03 / 1.11e-03
    updates                      2000 / 2000 (daily)     1216 / 1219 relinearisations
    min rank [C; CA]             2                       2 (rank-deficient updates: 0)
    max Re eig(A - LC)           -                       -0.0100 /day; max |eig((I-KC)Phi)| = 0.99005
    observer time constants      -                       day 0: 100.0 and 48.0 d (+delta), 100.0 and 74.8 d (-delta); over all updates 20.9..100.0 d
  [pid]                         EKF (main)              Luenberger lsd (baseline)   Luenberger lzoh (task form)
    convergence day (e1<1%)      244 / 358               277 / 269                   246 / 327
    e1 at day 2000               8.11e-08 / 1.50e-07     1.53e-07 / 1.53e-07         3.25e-06 / 3.25e-06
    RMS e1, days 0..2000         1.65e-02 / 2.27e-02     3.86e-02 / 4.01e-02         3.74e-02 / 4.20e-02
    RMS e1, days 100..2000       4.97e-03 / 8.93e-03     1.76e-02 / 1.80e-02         1.61e-02 / 2.08e-02
    e2 at day 2000               9.09e-11 / 1.67e-10     5.39e-08 / 5.39e-08         7.75e-04 / 7.75e-04
    updates                      2000 / 2000 (daily)     1129 / 1128 relinearisations
    min rank [C; CA]             2                       2 (rank-deficient updates: 0)
    max Re eig(A - LC)           -                       -0.0100 /day; max |eig((I-KC)Phi)| = 0.99005
    observer time constants      -                       day 0: 100.0 and 48.0 d (+delta), 100.0 and 74.8 d (-delta); over all updates 27.2..100.0 d
  [smc]                         EKF (main)              Luenberger lsd (baseline)   Luenberger lzoh (task form)
    convergence day (e1<1%)      244 / 358               274 / 271                   507 / 392
    e1 at day 2000               8.10e-08 / 1.49e-07     7.51e-06 / 7.51e-06         1.97e-04 / 1.97e-04
    RMS e1, days 0..2000         1.65e-02 / 2.27e-02     3.86e-02 / 4.00e-02         3.79e-02 / 4.22e-02
    RMS e1, days 100..2000       4.98e-03 / 8.94e-03     1.76e-02 / 1.79e-02         1.69e-02 / 2.12e-02
    e2 at day 2000               9.08e-11 / 1.67e-10     5.95e-07 / 5.95e-07         9.12e-04 / 9.12e-04
    updates                      2000 / 2000 (daily)     1227 / 1229 relinearisations
    min rank [C; CA]             2                       2 (rank-deficient updates: 0)
    max Re eig(A - LC)           -                       -0.0100 /day; max |eig((I-KC)Phi)| = 0.99005
    observer time constants      -                       day 0: 100.0 and 48.0 d (+delta), 100.0 and 74.8 d (-delta); over all updates 23.6..100.0 d
  At x_pre with x1_hat = x1 (step 2 of observer.py): eig(A) = {-0.008549, +0.003840}/day; the old rule
  2 eig(A) gives the UNSTABLE pole +0.00768/day; the new rule gives -0.0100 and -0.0171/day (tau 100 and 58.5 d).

B. NOISE AND SPARSE IMAGING - nominal plant, median / worst over 20 seeds, delta = +0.2  [summary_by_scenario.csv]
  M2:
    mu  EKF  conv. day   458 / 566   RMS e1 100..2000 1.85e-02 / 3.42e-02  e1(2000) 2.69e-04 / 7.56e-04  not converged 0/20
    mu  lsd  conv. day  1955 / n.c.  RMS e1 100..2000 4.24e-02 / 6.87e-02  e1(2000) 4.67e-03 / 1.38e-02  not converged 3/20
    pid EKF  conv. day   449 / 566   RMS e1 100..2000 1.85e-02 / 3.41e-02  e1(2000) 2.69e-04 / 7.56e-04  not converged 0/20
    pid lsd  conv. day  1946 / 2000  RMS e1 100..2000 3.63e-02 / 5.94e-02  e1(2000) 3.32e-03 / 9.70e-03  not converged 0/20
    smc EKF  conv. day   449 / 566   RMS e1 100..2000 1.85e-02 / 3.41e-02  e1(2000) 2.69e-04 / 7.56e-04  not converged 0/20
    smc lsd  conv. day  1911 / n.c.  RMS e1 100..2000 4.17e-02 / 5.80e-02  e1(2000) 4.68e-03 / 1.01e-02  not converged 1/20
  M3_7:
    mu  EKF  conv. day   532 / 654   RMS e1 100..2000 2.64e-02 / 3.88e-02  e1(2000) 1.44e-04 / 5.18e-04  not converged 0/20
    mu  lsd  conv. day  1974 / n.c.  RMS e1 100..2000 8.38e-02 / 1.06e-01  e1(2000) 8.79e-03 / 3.63e-02  not converged 9/20
    pid EKF  conv. day   532 / 651   RMS e1 100..2000 2.63e-02 / 3.86e-02  e1(2000) 1.44e-04 / 5.18e-04  not converged 0/20
    pid lsd  conv. day  1984 / n.c.  RMS e1 100..2000 6.33e-02 / 8.14e-02  e1(2000) 8.99e-03 / 2.41e-02  not converged 8/20
    smc EKF  conv. day   532 / 651   RMS e1 100..2000 2.63e-02 / 3.86e-02  e1(2000) 1.44e-04 / 5.18e-04  not converged 0/20
    smc lsd  conv. day  1978 / n.c.  RMS e1 100..2000 7.75e-02 / 9.84e-02  e1(2000) 9.30e-03 / 3.03e-02  not converged 10/20
  M3_28:
    mu  EKF  conv. day   560 / 679   RMS e1 100..2000 2.86e-02 / 4.82e-02  e1(2000) 1.26e-04 / 3.64e-04  not converged 0/20
    mu  lsd  conv. day  1988 / n.c.  RMS e1 100..2000 1.20e-01 / 1.65e-01  e1(2000) 1.88e-02 / 3.84e-02  not converged 14/20
    pid EKF  conv. day   560 / 676   RMS e1 100..2000 2.84e-02 / 4.77e-02  e1(2000) 1.26e-04 / 3.64e-04  not converged 0/20
    pid lsd  conv. day  1988 / n.c.  RMS e1 100..2000 9.94e-02 / 1.63e-01  e1(2000) 1.66e-02 / 4.17e-02  not converged 13/20
    smc EKF  conv. day   560 / 675   RMS e1 100..2000 2.84e-02 / 4.77e-02  e1(2000) 1.26e-04 / 3.64e-04  not converged 0/20
    smc lsd  conv. day  1988 / n.c.  RMS e1 100..2000 1.20e-01 / 1.63e-01  e1(2000) 1.60e-02 / 3.84e-02  not converged 14/20

C. FACT 1 - x1 IS NOT IDENTIFIABLE FROM y = x2 UNDER GROWTH-RATE MISMATCH (structural, not a code error)
  Mechanism: x1 enters dx2/dt only through -b2 x1 x2, so a relative growth-rate error eps of the x2
  equation is indistinguishable from an x1 offset of about -eps/b2 (b2 = 0.0018/day).
  eps(t) = [f2_plant(x, x4) - f2_nominal(x, x4_hat)]/x2 on the stored plant trajectory; medians over days 500..1500;
  observed = median(x1_hat - x1), EKF, M1.  [source: results/timeseries/truth_<ctrl>.csv.gz,
  est_ekf_<ctrl>_M1.csv.gz; computed by make_facts_block.py]
    ctrl  group   eps [1/day]   predicted -eps/b2   observed (+delta / -delta)
    mu    G1     +2.39e-03     -1.327              -1.559 / -1.971
    mu    G2     +8.74e-04     -0.485              -0.032 / -0.031
    mu    G3     +2.21e-03     -1.228              -1.765 / -1.797
    mu    G4     +2.43e-03     -1.348              -1.956 / -2.025
    mu    G5     +2.80e-03     -1.555              -2.052 / -2.051
    pid   G1     +2.39e-03     -1.328              -1.959 / -1.959
    pid   G2     +8.75e-04     -0.486              -0.032 / -0.031
    pid   G3     +2.15e-03     -1.195              -1.678 / -1.751
    pid   G4     +2.36e-03     -1.309              -2.005 / -2.023
    pid   G5     +2.66e-03     -1.480              -2.021 / -2.021
    smc   G1     +2.39e-03     -1.328              -1.959 / -1.959
    smc   G2     +8.75e-04     -0.486              -0.032 / -0.031
    smc   G3     +2.24e-03     -1.247              -1.890 / -1.906
    smc   G4     +2.49e-03     -1.382              -1.984 / -1.984
    smc   G5     +2.82e-03     -1.567              -2.037 / -2.037
  Groups G1, G3, G4, G5, all controllers: predicted -1.57 to -1.19, observed -2.05 to -1.56 (same sign, same order).
  G2 changes b2 itself (the coupling), so the simple mapping does not apply; its observed bias is small.
  Resulting x1 and x2 errors, M1, five groups  [source: results/metrics_all.csv]:
    ekf   e1(2000) G1/G3/G4/G5 0.97..2.04; G2 0.017..0.017; e2(2000) max 0.08%, RMS e2 max 0.08%; diverged 0/30
    lsd   e1(2000) G1/G3/G4/G5 1.03..2.55; G2 0.145..0.177; e2(2000) max 3.82%, RMS e2 max 5.43%; diverged 8/30
    lzoh  e1(2000) G1/G3/G4/G5 0.87..2.06; G2 0.144..0.177; e2(2000) max 3.83%, RMS e2 max 3.89%; diverged 9/30
  -> x2 stays well estimated while x1 does not: worst e2(2000) over the non-diverged M1 group runs is 0.08% (EKF),
     3.8% (lsd) and 3.8% (lzoh); the diverged runs are all Luenberger runs (counted above).

D. FACT 2 - EKF CONSISTENCY
  mu   nominal, M1: sqrt(P11)/x1 at day 2000 = 0.85% / 0.85%, true e1 = 8.3e-08 / 1.5e-07  -> conservative (floor set by Q)
  pid  nominal, M1: sqrt(P11)/x1 at day 2000 = 0.85% / 0.85%, true e1 = 8.1e-08 / 1.5e-07  -> conservative (floor set by Q)
  smc  nominal, M1: sqrt(P11)/x1 at day 2000 = 0.85% / 0.85%, true e1 = 8.1e-08 / 1.5e-07  -> conservative (floor set by Q)
  Share of days 100..2000 with |x1_hat - x1| <= 2 sqrt(P11), median over all EKF runs per plant
  (all controllers, modes, seeds, both delta)  [source: results/metrics_all.csv, ekf_frac_in_2sd_100_2000]:
    nominal 100.0%, G1 17.7%, G2 75.4%, G3 12.2%, G4 12.4%, G5 12.3%
  -> nominal: consistent; growth-rate mismatch groups G1/G3/G4/G5: 12-18% inside 2 sigma (overconfident); G2: 75%.
  Late in treatment observability fades: b2 x2 falls from 5.6e-4/day at day 0 to 2e-9 (mu) .. 3e-8 (PID)/day
  at day 2000, cond([C; CA]) rises from 1.8e3 to 4e7 (PID) .. 5e8 (mu)  [figure fig_ekf_sd_observability].

E. DOSE SCHEDULES (single convention, doses on days 0..1999)  [source: schedules/PROVENANCE.md]
  first day x2 < 1e-4: mu 1167, PID 1166, SMC 1165; cumulative dose [mg m^-2]: 142150, 131350, 140050;
  high-dose phases: 65, 71, 71; longest: 5, 4, 4 days. All checks a-g passed.
```

## 11. Lyapunov (nominal model only; v2 SMC parameters)

| quantity | value | source |
|---|---|---|
| c | 3 | lyapunov/lyapunov_results.json certificate.c |
| Umin | 50 | lyapunov/lyapunov_results.json certificate.Umin |
| Umax | 200 | lyapunov/lyapunov_results.json certificate.Umax |
| x4bar | 250 | lyapunov/lyapunov_results.json certificate.x4bar |
| eps | 4.7000e-05 | lyapunov/lyapunov_results.json certificate.eps |
| m | 2.5000e-04 | lyapunov/lyapunov_results.json certificate.m |
| beta2 | 0.0018 | lyapunov/lyapunov_results.json certificate.beta2 |
| monotonicity_ok | True | lyapunov/lyapunov_results.json certificate.monotonicity_ok |
| m_lt_beta2 | True | lyapunov/lyapunov_results.json certificate.m_lt_beta2 |
| eta | 1.8370e-04 | lyapunov/lyapunov_results.json certificate.eta |
| V0 | 1.01121 | lyapunov/lyapunov_results.json certificate.V0 |
| W0 | 1.05713 | lyapunov/lyapunov_results.json certificate.W0 |
| x2bar | 0.352378 | lyapunov/lyapunov_results.json certificate.x2bar |
| x1min | 0.148245 | lyapunov/lyapunov_results.json certificate.x1min |
| residual_w_max | 0.00691221 | lyapunov/lyapunov_results.json certificate.residual_w_max |
| residual_x2_max | 1.7777e-05 | lyapunov/lyapunov_results.json certificate.residual_x2_max |
| eps_over_Om1 | 0.00691176 | lyapunov/lyapunov_results.json certificate.eps_over_Om1 |
| lstar_box | 7.7332e-05 | lyapunov/lyapunov_results.json certificate.lstar_box |
| x1_floor | 0.987615 | lyapunov/lyapunov_results.json certificate.x1_floor |
| mu_rate | 0.00152771 | lyapunov/lyapunov_results.json certificate.mu_rate |
| partC_violations | 0 | lyapunov/lyapunov_results.json certificate.partC_violations |
| partC_lstar (SUPERSEDED by lstar_box) | 6.0131e-06 | lyapunov_results.json certificate.partC_lstar |
| roots (1/day; d) | -9.41947, -0.199715, -1.56658e-05 | lyapunov_results.json error_dynamics.roots (re-implemented) |
| time_constants_days (1/day; d) | 0.106163, 5.00713, 63833.4 | lyapunov_results.json error_dynamics.time_constants_days (re-implemented) |
| hurwitz | True | lyapunov_results.json error_dynamics.hurwitz (re-implemented) |
| L1_gain_0_2000 | 0.54714 | lyapunov_results.json error_dynamics.L1_gain_0_2000 (re-implemented) |
| free_response_gain | 0.0976603 | lyapunov_results.json error_dynamics.free_response_gain (re-implemented) |
| t_of_free_response_peak | 0.418 | lyapunov_results.json error_dynamics.t_of_free_response_peak (re-implemented) |
| sup_abs_s | 0.0302886 | lyapunov_results.json error_dynamics.sup_abs_s (re-implemented) |
| day_of_sup_abs_s | 0 | lyapunov_results.json error_dynamics.day_of_sup_abs_s (re-implemented) |
| share_days_abs_s_le_delta | 0.869 | lyapunov_results.json error_dynamics.share_days_abs_s_le_delta (re-implemented) |
| de_dt_0 | 0.00338125 | lyapunov_results.json error_dynamics.de_dt_0 (re-implemented) |
| bound_abs_e | 0.0169023 | lyapunov_results.json error_dynamics.bound_abs_e (re-implemented) |
| observed_max_abs_e | 0.005223 | lyapunov_results.json error_dynamics.observed_max_abs_e (re-implemented) |
| day_observed_max | 73 | lyapunov_results.json error_dynamics.day_observed_max (re-implemented) |
| bound_holds | True | lyapunov_results.json error_dynamics.bound_holds (re-implemented) |
| observed_max_abs_e_over_x2pre | 0.016648 | lyapunov_results.json error_dynamics.observed_max_abs_e_over_x2pre (re-implemented) |
| check_old_params_L1 | 0.749019 | lyapunov_results.json error_dynamics.check_old_params_L1 (re-implemented) |
| check_old_params_gmax | 0.296295 | lyapunov_results.json error_dynamics.check_old_params_gmax (re-implemented) |
| closed loop reproduces final SMC: same doses / max state dev | True / 2.2737e-13 | lyapunov_results.json closed_loop_reproduces_final_smc |
| dW/dt > 1e-10 days, closed_loop SMC nominal (first day; m < beta2) | 0 (NaN; True) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, closed_loop SMC G1 (first day; m < beta2) | 27 (15; False) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, closed_loop SMC G2 (first day; m < beta2) | 39 (16; True) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, closed_loop SMC G3 (first day; m < beta2) | 455 (17; False) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, closed_loop SMC G4 (first day; m < beta2) | 422 (18; False) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, closed_loop SMC G5 (first day; m < beta2) | 499 (13; False) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, E1_replay PID nominal (first day; m < beta2) | 0 (NaN; True) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, E1_replay PID G1 (first day; m < beta2) | 796 (15; False) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, E1_replay PID G2 (first day; m < beta2) | 58 (16; True) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, E1_replay PID G3 (first day; m < beta2) | 766 (17; False) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, E1_replay PID G4 (first day; m < beta2) | 771 (18; False) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, E1_replay PID G5 (first day; m < beta2) | 1158 (13; False) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, E1_replay SMC nominal (first day; m < beta2) | 0 (NaN; True) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, E1_replay SMC G1 (first day; m < beta2) | 488 (12; False) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, E1_replay SMC G2 (first day; m < beta2) | 51 (13; True) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, E1_replay SMC G3 (first day; m < beta2) | 395 (14; False) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, E1_replay SMC G4 (first day; m < beta2) | 434 (15; False) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, E1_replay SMC G5 (first day; m < beta2) | 1030 (10; False) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, E1_replay mu nominal (first day; m < beta2) | 0 (NaN; True) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, E1_replay mu G1 (first day; m < beta2) | 459 (0; False) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, E1_replay mu G2 (first day; m < beta2) | 70 (0; True) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, E1_replay mu G3 (first day; m < beta2) | 330 (6; False) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, E1_replay mu G4 (first day; m < beta2) | 395 (26; False) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, E1_replay mu G5 (first day; m < beta2) | 1029 (5; False) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, constant_dose u=50 nominal (first day; m < beta2) | 0 (NaN; True) | data/lyapunov_wdot_summary.csv |
| dW/dt > 1e-10 days, constant_dose u=200 nominal (first day; m < beta2) | 0 (NaN; True) | data/lyapunov_wdot_summary.csv |

Holds (nominal): psi decreasing, m < beta2, Q < 0 outside the residual box, dW/dt <= 1e-10 on all nominal runs. Fails (groups): m < beta2 fails for G1, G3, G4, G5; dW/dt > 0 in all five groups. See lyapunov/LYAPUNOV_REPORT.md.

## 12. mu appendix material

| quantity | value | source |
|---|---|---|
| verif.err_weights | 5.0585e-16 | data/mu_scalars.csv verif.err_weights |
| verif.err_interconnect | 9.9715e-16 | data/mu_scalars.csv verif.err_interconnect |
| verif.err_vs_connect | 9.8112e-16 | data/mu_scalars.csv verif.err_vs_connect |
| verif.err_scaling | 7.8756e-09 | data/mu_scalars.csv verif.err_scaling |
| verif.err_zoh | 1.7379e-06 | data/mu_scalars.csv verif.err_zoh |
| verif.nominal_stable | 1 | data/mu_scalars.csv verif.nominal_stable |
| verif.sampled_stable | 1 | data/mu_scalars.csv verif.sampled_stable |
| verif.mu | 0.46547 | data/mu_scalars.csv verif.mu |
| verif.robstab_LB | 6.38618 | data/mu_scalars.csv verif.robstab_LB |
| verif.robstab_UB | 6.39895 | data/mu_scalars.csv verif.robstab_UB |
| mu_initial | 0.46547 | data/mu_scalars.csv mu_initial |
| metrics_discrete.n_resyn | 60 | data/mu_scalars.csv metrics_discrete.n_resyn |
| metrics_discrete.n_fail | 0 | data/mu_scalars.csv metrics_discrete.n_fail |
| metrics_discrete.mu_max | 0.465572 | data/mu_scalars.csv metrics_discrete.mu_max |
| metrics_discrete.mu_first | 0.46547 | data/mu_scalars.csv metrics_discrete.mu_first |
| metrics_discrete.u_mean | 71.0645 | data/mu_scalars.csv metrics_discrete.u_mean |
| metrics_discrete.cum_dose | 142150 | data/mu_scalars.csv metrics_discrete.cum_dose |
| simD_nResyn | 60 | data/mu_scalars.csv simD_nResyn |
| simD_nFail | 0 | data/mu_scalars.csv simD_nFail |
| simD_nBumpFallback | 0 | data/mu_scalars.csv simD_nBumpFallback |
| simD_nMusyn | 60 | data/mu_scalars.csv simD_nMusyn |
| ISE_discrete | 3.7513e-05 | data/mu_scalars.csv ISE_discrete |
| ISE_open | 8.93285 | data/mu_scalars.csv ISE_open |
| search_n_candidates | 33 | data/mu_scalars.csv search_n_candidates |
| search_n_valid | 26 | data/mu_scalars.csv search_n_valid |
| search_n_musyn | 1557 | data/mu_scalars.csv search_n_musyn |
| search_J_winner | 1.5893 | data/mu_scalars.csv search_J_winner |
| search_J_B | 2.4 | data/mu_scalars.csv search_J_B |
| search_winner_index | 2 | data/mu_scalars.csv search_winner_index |
| elapsed_min | 309.464 | data/mu_scalars.csv elapsed_min |
| metrics_discrete.u_mean (SUPERSEDED: mean over 2001 values; use mean_dose 71.075) | 71.0645 | mu_scalars.csv |
| weight search wall_min_total | 295.7 | unify_crosschecks.json mu_search.wall_min_total |
| weight search wall_min_valid_min | 1.4815 | unify_crosschecks.json mu_search.wall_min_valid_min |
| weight search wall_min_valid_median | 7.00591 | unify_crosschecks.json mu_search.wall_min_valid_median |
| weight search wall_min_valid_max | 139.886 | unify_crosschecks.json mu_search.wall_min_valid_max |
| weight search winner_tag | D1 | unify_crosschecks.json mu_search.winner_tag |
| E2 closed loop G1 x2(T) | 1.9355e-06 | data/metrics_all.csv [E2/mu/G1] x2_T |
| E2 G1 first day x2 < 1e-4 | 1178 | data/metrics_all.csv [E2/mu/G1] t_x2_first |
| E2 G1 cumulative dose | 161050 | data/metrics_all.csv [E2/mu/G1] cum_dose |
| E2 G1 high-dose phases | 78 | data/metrics_all.csv [E2/mu/G1] n_hi |
| E2 G1 longest high-dose phase | 6 | data/metrics_all.csv [E2/mu/G1] hi_max |
| E2 G1 phases > 5 d | 32 | data/metrics_all.csv [E2/mu/G1] n_hi_gt5 |
| E2 closed loop G2 x2(T) | 2.2451e-06 | data/metrics_all.csv [E2/mu/G2] x2_T |
| E2 G2 first day x2 < 1e-4 | 1169 | data/metrics_all.csv [E2/mu/G2] t_x2_first |
| E2 G2 cumulative dose | 146950 | data/metrics_all.csv [E2/mu/G2] cum_dose |
| E2 G2 high-dose phases | 67 | data/metrics_all.csv [E2/mu/G2] n_hi |
| E2 G2 longest high-dose phase | 5 | data/metrics_all.csv [E2/mu/G2] hi_max |
| E2 G2 phases > 5 d | 0 | data/metrics_all.csv [E2/mu/G2] n_hi_gt5 |
| E2 closed loop G3 x2(T) | 7.5225e-07 | data/metrics_all.csv [E2/mu/G3] x2_T |
| E2 G3 first day x2 < 1e-4 | 1174 | data/metrics_all.csv [E2/mu/G3] t_x2_first |
| E2 G3 cumulative dose | 169750 | data/metrics_all.csv [E2/mu/G3] cum_dose |
| E2 G3 high-dose phases | 86 | data/metrics_all.csv [E2/mu/G3] n_hi |
| E2 G3 longest high-dose phase | 6 | data/metrics_all.csv [E2/mu/G3] hi_max |
| E2 G3 phases > 5 d | 46 | data/metrics_all.csv [E2/mu/G3] n_hi_gt5 |
| E2 closed loop G4 x2(T) | 2.3523e-06 | data/metrics_all.csv [E2/mu/G4] x2_T |
| E2 G4 first day x2 < 1e-4 | 1172 | data/metrics_all.csv [E2/mu/G4] t_x2_first |
| E2 G4 cumulative dose | 166600 | data/metrics_all.csv [E2/mu/G4] cum_dose |
| E2 G4 high-dose phases | 81 | data/metrics_all.csv [E2/mu/G4] n_hi |
| E2 G4 longest high-dose phase | 7 | data/metrics_all.csv [E2/mu/G4] hi_max |
| E2 G4 phases > 5 d | 50 | data/metrics_all.csv [E2/mu/G4] n_hi_gt5 |
| E2 closed loop G5 x2(T) | 2.4549e-06 | data/metrics_all.csv [E2/mu/G5] x2_T |
| E2 G5 first day x2 < 1e-4 | 1170 | data/metrics_all.csv [E2/mu/G5] t_x2_first |
| E2 G5 cumulative dose | 167050 | data/metrics_all.csv [E2/mu/G5] cum_dose |
| E2 G5 high-dose phases | 92 | data/metrics_all.csv [E2/mu/G5] n_hi |
| E2 G5 longest high-dose phase | 6 | data/metrics_all.csv [E2/mu/G5] hi_max |
| E2 G5 phases > 5 d | 10 | data/metrics_all.csv [E2/mu/G5] n_hi_gt5 |
| rule 'cap 40' redesigns | 40 | data/metrics_all.csv [mu_redesign_rule/mu/cap 40] n_redesigns |
| rule 'cap 40' last redesign day | 691 | data/metrics_all.csv [mu_redesign_rule/mu/cap 40] last_redesign_day |
| rule 'cap 40' first day x2 < 1e-4 | 1721 | data/metrics_all.csv [mu_redesign_rule/mu/cap 40] t_x2_first |
| rule 'cap 40' x2(T) | 6.4870e-05 | data/metrics_all.csv [mu_redesign_rule/mu/cap 40] x2_T |
| rule 'cap 40' cumulative dose | 125350 | data/metrics_all.csv [mu_redesign_rule/mu/cap 40] cum_dose |
| rule 'stop at x2 < 0.001' redesigns | 51 | data/metrics_all.csv [mu_redesign_rule/mu/stop at x2 < 0.001] n_redesigns |
| rule 'stop at x2 < 0.001' last redesign day | 847 | data/metrics_all.csv [mu_redesign_rule/mu/stop at x2 < 0.001] last_redesign_day |
| rule 'stop at x2 < 0.001' first day x2 < 1e-4 | 1184 | data/metrics_all.csv [mu_redesign_rule/mu/stop at x2 < 0.001] t_x2_first |
| rule 'stop at x2 < 0.001' x2(T) | 7.5542e-06 | data/metrics_all.csv [mu_redesign_rule/mu/stop at x2 < 0.001] x2_T |
| rule 'stop at x2 < 0.001' cumulative dose | 134500 | data/metrics_all.csv [mu_redesign_rule/mu/stop at x2 < 0.001] cum_dose |
| rule 'stop at x2 < 0.0001' redesigns | 60 | data/metrics_all.csv [mu_redesign_rule/mu/stop at x2 < 0.0001] n_redesigns |
| rule 'stop at x2 < 0.0001' last redesign day | 1115 | data/metrics_all.csv [mu_redesign_rule/mu/stop at x2 < 0.0001] last_redesign_day |
| rule 'stop at x2 < 0.0001' first day x2 < 1e-4 | 1167 | data/metrics_all.csv [mu_redesign_rule/mu/stop at x2 < 0.0001] t_x2_first |
| rule 'stop at x2 < 0.0001' x2(T) | 1.2517e-06 | data/metrics_all.csv [mu_redesign_rule/mu/stop at x2 < 0.0001] x2_T |
| rule 'stop at x2 < 0.0001' cumulative dose | 142150 | data/metrics_all.csv [mu_redesign_rule/mu/stop at x2 < 0.0001] cum_dose |
| rule 'no stop, no cap' redesigns | 61 | data/metrics_all.csv [mu_redesign_rule/mu/no stop, no cap] n_redesigns |
| rule 'no stop, no cap' last redesign day | 1266 | data/metrics_all.csv [mu_redesign_rule/mu/no stop, no cap] last_redesign_day |
| rule 'no stop, no cap' first day x2 < 1e-4 | 1167 | data/metrics_all.csv [mu_redesign_rule/mu/no stop, no cap] t_x2_first |
| rule 'no stop, no cap' x2(T) | 2.8525e-07 | data/metrics_all.csv [mu_redesign_rule/mu/no stop, no cap] x2_T |
| rule 'no stop, no cap' cumulative dose | 148450 | data/metrics_all.csv [mu_redesign_rule/mu/no stop, no cap] cum_dose |
| Monte Carlo replay, N=300: N | 300 | tables/T_app_mu_monte_carlo.csv |
| Monte Carlo replay, N=300: x2T_median | 7.7047e-07 | tables/T_app_mu_monte_carlo.csv |
| Monte Carlo replay, N=300: x2T_worst | 0.421475 | tables/T_app_mu_monte_carlo.csv |
| Monte Carlo replay, N=300: reach | 225 | tables/T_app_mu_monte_carlo.csv |
| Monte Carlo replay, N=300: reach_fraction | 0.75 | tables/T_app_mu_monte_carlo.csv |
| Monte Carlo replay, N=300: reach_ci95_lo | 0.697006 | tables/T_app_mu_monte_carlo.csv |
| Monte Carlo replay, N=300: reach_ci95_hi | 0.797981 | tables/T_app_mu_monte_carlo.csv |
| Monte Carlo replay, first 20: N | 20 | tables/T_app_mu_monte_carlo.csv |
| Monte Carlo replay, first 20: x2T_median | 2.8839e-05 | tables/T_app_mu_monte_carlo.csv |
| Monte Carlo replay, first 20: x2T_worst | 0.421475 | tables/T_app_mu_monte_carlo.csv |
| Monte Carlo replay, first 20: reach | 13 | tables/T_app_mu_monte_carlo.csv |
| Monte Carlo replay, first 20: reach_fraction | 0.65 | tables/T_app_mu_monte_carlo.csv |
| Monte Carlo replay, first 20: reach_ci95_lo | 0.407811 | tables/T_app_mu_monte_carlo.csv |
| Monte Carlo replay, first 20: reach_ci95_hi | 0.846091 | tables/T_app_mu_monte_carlo.csv |
| Monte Carlo closed loop, first 20: N | 20 | tables/T_app_mu_monte_carlo.csv |
| Monte Carlo closed loop, first 20: x2T_median | 1.4741e-06 | tables/T_app_mu_monte_carlo.csv |
| Monte Carlo closed loop, first 20: x2T_worst | 3.6724e-06 | tables/T_app_mu_monte_carlo.csv |
| Monte Carlo closed loop, first 20: reach | 20 | tables/T_app_mu_monte_carlo.csv |
| Monte Carlo closed loop, first 20: reach_fraction | 1 | tables/T_app_mu_monte_carlo.csv |
| Monte Carlo closed loop, first 20: reach_ci95_lo | 0.831567 | tables/T_app_mu_monte_carlo.csv |
| Monte Carlo closed loop, first 20: reach_ci95_hi | 1 | tables/T_app_mu_monte_carlo.csv |
| Monte Carlo closed loop, first 20: maxHi_worst | 10 | tables/T_app_mu_monte_carlo.csv |
| Monte Carlo closed loop, first 20: n_runs_with_phase_gt5 | 13 | tables/T_app_mu_monte_carlo.csv |

Weight-search history: tables/T_app_mu_weight_search.csv; mu per redesign: data/mu_per_redesign.csv.

## 13. Tuning (PID, SMC) and mu search settings

| quantity | value | source |
|---|---|---|
| A_REF_UNIFIED | 7.5 | tables/tuning_settings.csv A_REF_UNIFIED |
| SEEDS | (7, 11, 23) | tables/tuning_settings.csv SEEDS |
| DE_STRATEGY | 'best1bin' | tables/tuning_settings.csv DE_STRATEGY |
| DE_POLISH | False | tables/tuning_settings.csv DE_POLISH |
| DE_WORKERS | 1 | tables/tuning_settings.csv DE_WORKERS |
| DE_UPDATING | 'immediate' | tables/tuning_settings.csv DE_UPDATING |
| MAXITER_PID | 249 | tables/tuning_settings.csv MAXITER_PID |
| MAXITER_SMC | 148 | tables/tuning_settings.csv MAXITER_SMC |
| W_PROT | 2.0 | tables/tuning_settings.csv W_PROT |
| W_SW | 2.0 | tables/tuning_settings.csv W_SW |
| TRACK_FLOOR | 0.0001 | tables/tuning_settings.csv TRACK_FLOOR |
| PROT_FREE_DAYS | 4 | tables/tuning_settings.csv PROT_FREE_DAYS |
| PROT_CYCLE_DAYS | 28 | tables/tuning_settings.csv PROT_CYCLE_DAYS |
| J_FAIL | 10000.0 | tables/tuning_settings.csv J_FAIL |
| BUDGET_MIN_PER_CONTROLLER | 60.0 | tables/tuning_settings.csv BUDGET_MIN_PER_CONTROLLER |
| PID_SPACE | (('PID_M_P', 1.0, 100.0, 'log'), ('PID_M_I', 0.18, 18.0, 'log'), ('PID_M_D', 0.33, 33.0, 'log'), ('PID_YGAIN', 200000.0, 20000000.0, 'log'), ('PID_A_GAIN', 2.0, 15.0, 'log'), ('PID_B_GAIN', 0.0, 1.0, 'lin')) | tables/tuning_settings.csv PID_SPACE |
| SMC_SPACE | (('LAMBDA_P', 0.1, 10.0, 'log'), ('LAMBDA_D', 0.1, 10.0, 'log'), ('LAMBDA_I', 1e-06, 0.01, 'log'), ('DELTA_INIT', 0.001, 0.1, 'log'), ('DELTA_FINAL', 1e-09, 0.01, 'log'), ('A_DELTA', 2.0, 15.0, 'log'), ('B_DELTA', 0.0, 1.0, 'lin')) | tables/tuning_settings.csv SMC_SPACE |
| PID hand design (start): J / N_high / nfev | 10.324 / 35 / NaN | tables/T_app_tuning_runs.csv |
| PID seed 7: J / N_high / nfev | 0.0610493 / 71 / 22500 | tables/T_app_tuning_runs.csv |
| PID seed 11: J / N_high / nfev | 0.062488 / 71 / 22500 | tables/T_app_tuning_runs.csv |
| PID seed 23: J / N_high / nfev | 0.0616772 / 71 / 22500 | tables/T_app_tuning_runs.csv |
| SMC hand design (start): J / N_high / nfev | 1.028 / 53 / NaN | tables/T_app_tuning_runs.csv |
| SMC seed 7: J / N_high / nfev | 0.0148052 / 71 / 15645 | tables/T_app_tuning_runs.csv |
| SMC seed 11: J / N_high / nfev | 0.0159116 / 71 / 15645 | tables/T_app_tuning_runs.csv |
| SMC seed 23: J / N_high / nfev | 0.0166543 / 71 / 15645 | tables/T_app_tuning_runs.csv |
| tuning_history rows | 114445 | unify_crosschecks.json tuning.rows |
| tuning evaluations PID / SMC | 67500 / 46935 | tuning.search_evals_by_controller |
| tuning max elapsed_s PID / SMC (s) | 5190.9 / 4494.7 | tuning.elapsed_s_by_controller |
| tuning total (min) | 161.427 | sum of the two above |
| cfg.n_random | 5 | data/mu_scalars.csv cfg.n_random |
| cfg.n_bo | 25 | data/mu_scalars.csv cfg.n_bo |
| cfg.search_minutes | 300 | data/mu_scalars.csv cfg.search_minutes |
| cfg.search_seed | 7 | data/mu_scalars.csv cfg.search_seed |
| cfg.mc_replay_N | 300 | data/mu_scalars.csv cfg.mc_replay_N |
| cfg.mc_cl_N | 20 | data/mu_scalars.csv cfg.mc_cl_N |
| cfg.mc_seed | 2024 | data/mu_scalars.csv cfg.mc_seed |
| cfg.musyn_maxiter | 8 | data/mu_scalars.csv cfg.musyn_maxiter |
| cfg.mu_max_accept | 10 | data/mu_scalars.csv cfg.mu_max_accept |
| cfg.resyn_gap | 5 | data/mu_scalars.csv cfg.resyn_gap |
| cfg.max_resyn | Inf | data/mu_scalars.csv cfg.max_resyn |
| cfg.w_prot | 2 | data/mu_scalars.csv cfg.w_prot |
| cfg.aw_beta | 1 | data/mu_scalars.csv cfg.aw_beta |
| S.u_switch | 125 | data/mu_scalars.csv S.u_switch |
| S.rel_thresh | 0.1 | data/mu_scalars.csv S.rel_thresh |
| S.xlin_floor | 0.001 | data/mu_scalars.csv S.xlin_floor |
| S.x2_stop | 1.0000e-04 | data/mu_scalars.csv S.x2_stop |
| mu search objective | J = Jtrack/Jtrack_B + 2*Jprot + max(0,mu_peak-1) + failed_redesign_fraction; Jtrack = mean((log10(max(x2,1e-4)) - log10(max(r,1e-4)))^2); Jprot = sum(max(0,run_days-4)^2)/(T/28); bayesopt minimizes log10(1+J) | data/mu_strings.csv search_objective |

## 14. Software versions

| quantity | value | source |
|---|---|---|
| Python (regeneration runs) | 3.13.0 | this interpreter |
| numpy | 2.2.4 | import |
| scipy | 1.15.2 | import |
| pandas | 3.1.1 | import |
| matplotlib | 3.10.1 | import |
| sympy | 1.14.0 | import |
| MATLAB / toolbox | MATLAB 9.14.0.2306882 (R2023a) Update 4 | regen/mu/matlab_version.txt |
| MATLAB / toolbox | Control System Toolbox / 10.13 / (R2023a) | regen/mu/matlab_version.txt |
| MATLAB / toolbox | Global Optimization Toolbox / 4.8.1 / (R2023a) | regen/mu/matlab_version.txt |
| MATLAB / toolbox | MATLAB / 9.14 / (R2023a) | regen/mu/matlab_version.txt |
| MATLAB / toolbox | MATLAB Coder / 5.6 / (R2023a) | regen/mu/matlab_version.txt |
| MATLAB / toolbox | MATLAB Compiler / 8.6 / (R2023a) | regen/mu/matlab_version.txt |
| MATLAB / toolbox | MATLAB Compiler SDK / 7.2 / (R2023a) | regen/mu/matlab_version.txt |
| MATLAB / toolbox | MATLAB Report Generator / 5.14 / (R2023a) | regen/mu/matlab_version.txt |
| MATLAB / toolbox | MATLAB Test / 1.0 / (R2023a) | regen/mu/matlab_version.txt |
| MATLAB / toolbox | Optimization Toolbox / 9.5 / (R2023a) | regen/mu/matlab_version.txt |
| MATLAB / toolbox | Robust Control Toolbox / 6.11.3 / (R2023a) | regen/mu/matlab_version.txt |
| MATLAB / toolbox | Simulink Design Optimization / 3.13 / (R2023a) | regen/mu/matlab_version.txt |
| MATLAB / toolbox | Statistics and Machine Learning Toolbox / 12.5 / (R2023a) | regen/mu/matlab_version.txt |

The original PID/SMC tuning and the MATLAB weight search were not re-run; their outputs were regenerated byte-identically (item 1.2).

## 15. Superseded values (do not use)

- MATLAB mean dose of the mu discrete run 71.06 (mean over 2001 logged values) -> 71.075 (sum days 0..1999 / 2000).
- mu continuous cumulative dose: 145061 (MATLAB trapz 0..2000) is reported next to 145032 (sum convention); neither is superseded, name the convention.
- SMC mean high-dose phase 3.8 (notebook, includes t = 0 placeholder) -> 3.76.
- Lyapunov Part C l* 6.013e-06 (encloses one cell) -> Part D l* 7.733e-05 (encloses the residual box).
- Every Lyapunov number computed with the old SMC parameters (Liapanov Stability folder) -> section 11.
- All numbers of the Old Paper (recomputed here; differences in STAGE1_REPORT.md, for information only).
