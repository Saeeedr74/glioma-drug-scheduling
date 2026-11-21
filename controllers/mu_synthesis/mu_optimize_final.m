%% mu_optimize_final.m   (version mu_opt_v1)
% ONE script that finds the mu-synthesis weights for the glioma chemotherapy
% manuscript and produces every number, CSV file and figure for the winner.
%
% What "best" means here (the objective) is what the paper reports:
%   J = Jtrack/Jtrack_B + w_prot*Jprot + max(0, mu_peak - 1) + failed_redesign_fraction
%   Jtrack = mean over the horizon of ( log10(max(x2,1e-4)) - log10(max(r,1e-4)) )^2
%     - log scale: an error at x2 ~ 1e-3 counts as much as one at x2 ~ 1e-1, so the
%       late part of the treatment is seen (a linear ISE does not see it);
%     - floor at 1e-4, the suppression threshold of the paper: once both x2 and r are
%       below it, nothing more is counted. Reaching 1e-4 late is penalized, going
%       further below it is not rewarded.
%   Jtrack_B = Jtrack of the reference weights B, so the tracking term is O(1).
%   Jprot = sum over high-dose phases of max(0, length_days - 4)^2 / (T/28): soft
%     penalty from the 5th consecutive high-dose day on; the dose rule has no hard cut.
%   bayesopt minimizes log10(1 + J).
%
% Every candidate is evaluated with the FULL closed loop and the FINAL redesign rule
% (no proxy, no second stage): what is searched is what is reported.
%   dose rule      : u = 200 if the clipped continuous command >= 125, else u = 50
%   redesign rule  : redesign on 10% drift of x2, min. gap 5 days, no redesign once
%                    x2 < 1e-4 (cfg.x2_stop), no cap on the number (cfg.max_resyn)
% Starting points: the weights B, D1, D2 of the earlier searches + cfg.n_random random
% points; then cfg.n_bo Bayesian-optimization evaluations. The winner is the
% candidate with the smallest J. If nothing beats B, B is the winner.
%
% After the search, for the winner only: verification, continuous run, redesign-rule
% check, FACTS.md 6.4 block, CSV files, uncertainty E1/E2/E3a/E3b, figures.
%
% HOW TO RUN: in this folder (a folder outside cloud-synced storage is safer for the caches):  mu_optimize_final
% Resumable: the search history and every full simulation are cached; re-running
% continues where it stopped. The console is copied to mu_opt_console.txt.
% Requires: Robust Control Toolbox, Statistics & Machine Learning Toolbox.
%
% Robustness wording for the paper: the mu guarantee covers each linear design only.
% The redesign logic, the saturation and the {50,200} quantization are outside it,
% so every closed-loop statement from this script is "in these simulations".

clear; clc; close all;
clear functions; clear global MU_H MU_NEW;
diary off; if isfile('mu_opt_console.txt'), delete('mu_opt_console.txt'); end
diary('mu_opt_console.txt');
tAll = tic;
VERSION = 'mu_opt_v1';

%% 0) Settings
cfg = struct();
% ---- redesign rule (used in the search AND in every reported run)
cfg.x2_stop        = 1e-4;     % no redesign once x2 < x2_stop (1e-4 = suppression threshold of the paper)
cfg.max_resyn      = Inf;      % cap on the number of redesigns (Inf = none)
% ---- search
cfg.w_prot         = 2;        % SETTING, not derived: weight of the protocol penalty
cfg.n_random       = 5;        % SETTING, not derived: random starting points (besides B, D1, D2)
cfg.n_bo           = 25;       % SETTING, not derived: Bayesian-optimization evaluations
cfg.search_minutes = 300;      % SETTING: wall-clock cap of the Bayesian-optimization phase
cfg.search_seed    = 7;        % seed of the random starting points
% ---- winner analysis
cfg.rule_check     = true;     % nominal loop of the winner under four redesign rules (diagnostic)
cfg.run_continuous = true;
cfg.run_cl_groups  = true;     % E2
cfg.mc_replay_N    = 300;      % SETTING, not derived
cfg.mc_cl_N        = 20;       % SETTING, not derived: closed-loop Monte Carlo runs (0 = skip)
cfg.mc_seed        = 2024;     % same samples as the earlier scripts
% ---- design safeguards (unchanged from the earlier scripts)
cfg.musyn_maxiter  = 8;
cfg.mu_max_accept  = 10;
cfg.resyn_gap      = 5;
cfg.aw_beta        = 1.0;
cfg.xc_max         = 1e8;
cfg.xo_floor       = 1e-6;
% ---- files
cfg.save_figs      = true;  cfg.fig_dpi = 300;
cfg.hist_file      = 'mu_opt_search_history.mat';
cfg.cache_file     = 'mu_opt_cache.mat';
cfg.rule_cache     = 'mu_opt_rules_cache.mat';
cfg.out_mat        = 'mu_opt_results.mat';
cfg.fig_prefix     = 'mu_opt_fig_';
cfg.csv_prefix     = 'mu_opt_u_';

% Starting points from the earlier searches.
%  B  = winner of mu_discrete_weight_optimization.m (weights of mu_final_compare.m)
%  D1 = winner of mu_final_design.m for w_prot = 0.5 and 2
%  D2 = winner of mu_final_design.m for w_prot = 8 and 32 (4 significant digits, as printed)
W.B  = struct('M',2.2692,'A',0.0025805,'wB',0.29781,'k',0.10699,'Wp_gain',0.65011,'wz',0.12527,'wp',0.86602);
W.D1 = struct('M',1.83935,'A',0.00704063,'wB',0.311074,'k',0.12858,'Wp_gain',0.594511,'wz',0.434664,'wp',6.74273);
W.D2 = struct('M',1.544,'A',0.02205,'wB',0.3159,'k',0.03404,'Wp_gain',0.5477,'wz',0.4228,'wp',5.445);
% Printed by mu_final_design.m for B and D1 (cap 40, no stop); used only for the
% reproduction check in the redesign-rule table.
EXPECT.B  = struct('t_sup',1176,'x2_T',5.31e-06,'cum_dose',136000,'n_hi',54);
EXPECT.D1 = struct('t_sup',1721,'x2_T',6.487e-05,'cum_dose',125350,'n_hi',41);

objdef.track_floor = 1e-4;           % suppression threshold of the paper
objdef.protocol_start_days = 4;      % soft penalty from the 5th consecutive high-dose day
objdef.w_prot = cfg.w_prot;
objdef.w_mu   = 1;                   % SETTING, not derived
objdef.w_fail = 1;                   % SETTING, not derived
objdef.J_fail = 1e4;                 % value of failed/aborted candidates (and cap of J)
objdef.def = sprintf(['J = Jtrack/Jtrack_B + %g*Jprot + max(0,mu_peak-1) + failed_redesign_fraction; ' ...
    'Jtrack = mean((log10(max(x2,1e-4)) - log10(max(r,1e-4)))^2); ' ...
    'Jprot = sum(max(0,run_days-4)^2)/(T/28); bayesopt minimizes log10(1+J)'],cfg.w_prot);

assert(exist('musyn','file')>0 && exist('ultidyn','file')>0 && exist('bayesopt','file')>0, ...
    'Robust Control Toolbox and Statistics & ML Toolbox are required.');
assert(cfg.mc_cl_N <= cfg.mc_replay_N,'cfg.mc_cl_N must not exceed cfg.mc_replay_N.');
if contains(lower(pwd),'onedrive')
    warning('This folder is inside OneDrive. A folder outside OneDrive is safer for the cache files.');
end

%% 1) Model, simulation settings, uncertainty definitions (unchanged from the earlier scripts)
S.dt = 1;  S.T_start = 600;  S.T_ref = 2000;
S.y0 = [1; 1e-3; 1; 0];
S.a = 7.5; S.b = 0;            % reference steepness set by the author
S.u_min = 50; S.u_max = 200;   % maintenance phase (u = 50), high-dose phase (u = 200)
S.u_switch = 125;              % single threshold on the clipped continuous command
S.rel_thresh = 0.10;           % redesign when x2 drifts 10% from the last design point
S.xlin_floor = 1e-3;
S.x2_stop = cfg.x2_stop;       % redesigns only while x2 >= x2_stop
S.discrete = true;

mp.Om1 = 0.0068; mp.Om2 = 0.012;
mp.b1 = 1.8e-2;  mp.b2 = 1.8e-3;
mp.p1 = 4.7e-8;  mp.p2 = 4.7e-5; mp.p3 = 4.7e-8;
mp.a1 = 1; mp.a2 = 1; mp.a3 = 1;
mp.alpha = 2.0; mp.zeta = 0.2;
f = @(x,u) glioma_model(x,u,mp);

unc.groups = { ...
    'Omega1 down, Omega2 up 20%', {'Om1','Om2'}, [-0.20  0.20]; ...
    'beta1 up, beta2 down 50%',   {'b1','b2'},   [ 0.50 -0.50]; ...
    'p1 up, p2 down 15%',         {'p1','p2'},   [ 0.15 -0.15]; ...
    'a1 up, a2 up 20%',           {'a1','a2'},   [ 0.20  0.20]; ...
    'zeta up 20%',                {'zeta'},      0.20};
unc.mc_names = {'Om1','Om2','b1','b2','p1','p2','a1','a2','zeta'};
unc.mc_frac  = [0.20 0.20 0.50 0.50 0.15 0.15 0.20 0.20 0.20];
nG = size(unc.groups,1);
rowNames = [{'nominal'}; unc.groups(:,1)];

%% 2) Preconditioning and reference
x = S.y0;
for ii = 1:round(S.T_start/S.dt), x = rk4_step(f,x,0,S.dt); end
x_pre = x;  xs = x_pre(2);
t   = (0:S.dt:S.T_ref)';
Sc  = 0.5*(1-tanh(S.a*(t/S.T_ref-S.b)));
S0  = 0.5*(1-tanh(S.a*(0-S.b)));
S1  = 0.5*(1-tanh(S.a*(1-S.b)));
ref = xs*(Sc-S1)/(S0-S1);
Xopen = traj_const_u(f,x_pre,numel(t),S.dt,0);
ISE_open = trapz(t,((Xopen(2,:)'-ref)/xs).^2)/t(end);

ruleMain = rule_label(S.x2_stop,cfg.max_resyn);
fprintf('Version %s | a = %g | dose rule: 200 if uclip >= %g, else 50 | redesign rule: %s\n', ...
    VERSION,S.a,S.u_switch,ruleMain);
fprintf('State at T_start: x1=%.6f x2=%.6f x3=%.6f x4=%.6f\n',x_pre);
fprintf('Objective: %s\n',objdef.def);

mopt = musynOptions('Display','off','MaxIter',cfg.musyn_maxiter);
ctx = struct('S',S,'mp',mp,'f',f,'f_plant',f,'x_pre',x_pre,'xs',xs,'t',t,'ref',ref,'cfg',cfg, ...
             'mopt',mopt,'objdef',objdef);

%% 3) WEIGHT SEARCH
spec.names = {'M','A','wB','k','Wp_gain','wz','wp'};
spec.lo    = [1.05  0.00005 1e-3 0.02 0.30 0.01 0.5];   % SETTING (same ranges as mu_final_design.m)
spec.hi    = [8.00  0.100   0.60 1.00 5.00 0.50 200];   % SETTING
spec.constraint = 'wp >= 4*wz & Wp_gain <= 1.2*M';
seedNames = fieldnames(W)';
for nm = seedNames
    pv = pvec(W.(nm{1}));
    assert(all(pv >= spec.lo & pv <= spec.hi) && W.(nm{1}).wp >= 4*W.(nm{1}).wz && W.(nm{1}).Wp_gain <= 1.2*W.(nm{1}).M, ...
        'Starting point %s is outside the search space.',nm{1});
end
sigS = struct('version',VERSION,'S',S,'mp',mp,'spec',spec,'objdef',rmfield(objdef,'def'), ...
    'max_resyn',cfg.max_resyn,'musyn_maxiter',cfg.musyn_maxiter,'mu_max_accept',cfg.mu_max_accept, ...
    'resyn_gap',cfg.resyn_gap,'aw_beta',cfg.aw_beta,'xc_max',cfg.xc_max,'xo_floor',cfg.xo_floor);

global MU_H MU_NEW %#ok<GVMIS>
MU_H = load_compatible(cfg.hist_file,sigS,'H');  MU_NEW = 0;
H = hist_get();
fprintf('\n================ WEIGHT SEARCH (full closed loop, %s) ================\n',ruleMain);
if ~isempty(H), fprintf('Resuming: %d candidates already evaluated.\n',numel(H)); end
fprintf('Columns: Jtr/B = Jtrack over Jtrack of B, t1e-4 = first day x2 < 1e-4, maxHi = longest high-dose phase [d], n>5 = phases longer than 5 d\n');

% (a) starting points B, D1, D2 (B first: it normalizes the tracking term)
for nm = seedNames
    if isempty(H) || ~any(arrayfun(@(h) pv_equal(h.pv,pvec(W.(nm{1}))),H))
        fprintf('evaluating starting point %s ...\n',nm{1});
        rec = eval_candidate(W.(nm{1}),ctx,nm{1});
        fprintf('  done in %.1f min (%d musyn calls)\n',rec.wall/60,rec.nMusyn);
        H = hist_append(cfg.hist_file,sigS,rec);
    end
end
iB = find(strcmp({H.tag},'B'),1);
if ~H(iB).ok, error('The reference weights B failed (%s); the tracking term cannot be normalized.',H(iB).msg); end
Jref = H(iB).T.jtrack;
for nm = seedNames
    i = find(strcmp({H.tag},nm{1}),1);
    print_cand_line(H(i),Jref,scalar_J(H(i).T,H(i).ok,Jref,objdef));
end

% (b) random starting points (log-uniform inside the ranges, constraint respected)
rs0 = RandStream('mt19937ar','Seed',cfg.search_seed);
PR = random_points(rs0,spec,cfg.n_random);
for i = 1:size(PR,1)
    pr = cell2struct(num2cell(PR(i,:))',spec.names',1);
    if any(arrayfun(@(h) pv_equal(h.pv,PR(i,:)),H)), continue; end
    rec = eval_candidate(pr,ctx,'rand');
    H = hist_append(cfg.hist_file,sigS,rec);
    print_cand_line(rec,Jref,scalar_J(rec.T,rec.ok,Jref,objdef));
end

% (c) Bayesian optimization
vc = cell(numel(spec.names),1);
for i = 1:numel(spec.names)
    vc{i} = optimizableVariable(spec.names{i},[spec.lo(i) spec.hi(i)],'Transform','log');
end
vars = vertcat(vc{:});
xcon = @(X) (X.wp >= 4*X.wz) & (X.Wp_gain <= 1.2*X.M);
H = hist_get();
nBoDone = sum(strcmp({H.tag},'bo'));
if nBoDone < cfg.n_bo
    fprintf('--- Bayesian optimization: %d of %d evaluations done ---\n',nBoDone,cfg.n_bo);
    Jh = zeros(numel(H),1);
    for i = 1:numel(H), Jh(i) = log10(1+scalar_J(H(i).T,H(i).ok,Jref,objdef)); end
    Xh = struct2table([H.p]');
    tS = tic;
    objfun = @(q) bo_objective(q,ctx,sigS,Jref);
    outfcn = @(res,state) search_stop(state,tS,cfg.search_minutes*60,cfg.n_bo);
    try
        bayesopt(objfun,vars,'InitialX',Xh,'InitialObjective',Jh, ...
            'IsObjectiveDeterministic',true,'AcquisitionFunctionName','expected-improvement-plus', ...
            'XConstraintFcn',xcon,'MaxObjectiveEvaluations',1e4,'OutputFcn',outfcn, ...
            'Verbose',0,'PlotFcn',[],'UseParallel',false);
    catch ME
        warning('bayesopt stopped with an error (the history is intact): %s',ME.message);
    end
else
    fprintf('Bayesian optimization: %d evaluations already done, skipped.\n',nBoDone);
end

% (d) ranking and winner
H = hist_get();
Jall = arrayfun(@(h) scalar_J(h.T,h.ok,Jref,objdef),H);
okAll = [H.ok];
nMusynSearch = sum([H.nMusyn]);
[~,ord] = sort(Jall);
fprintf('\n================ SEARCH RESULT (%d candidates, %d valid, %d musyn calls) ================\n', ...
    numel(H),sum(okAll),nMusynSearch);
fprintf('%-4s %-5s %8s %7s %7s %6s %10s %5s %4s %7s %6s | %7s %9s %7s %8s %7s %7s %7s\n','rank','tag','J','Jtr/B','Jprot', ...
    't1e-4','x2(T)','maxHi','n>5','cumU','mu','M','A','wB','k','Wp_gain','wz','wp');
showIdx = unique([ord(1:min(10,numel(ord))), find(ismember({H.tag},seedNames))],'stable');
for i = showIdx
    h = H(i); q = h.p; r = find(ord == i,1);
    if ~h.ok
        fprintf('%-4d %-5s %8s  FAILED: %s\n',r,h.tag,'-',h.msg); continue;
    end
    fprintf('%-4d %-5s %8.4g %7.3g %7.3g %6g %10.3e %5g %4d %7.0f %6.3f | %7.4g %9.4g %7.4g %8.4g %7.4g %7.4g %7.4g\n', ...
        r,h.tag,Jall(i),h.T.jtrack/Jref,h.T.Jprot,h.T.t_sup,h.T.x2T,h.T.maxrun,h.T.n_gt5,h.T.cumU,h.mu, ...
        q.M,q.A,q.wB,q.k,q.Wp_gain,q.wz,q.wp);
end
Jsel = Jall; Jsel(~okAll) = Inf;
[Jwin,iWin] = min(Jsel);
if ~isfinite(Jwin), error('No valid candidate.'); end
p = H(iWin).p;
if any(strcmp(H(iWin).tag,seedNames)), selLabel = H(iWin).tag; else, selLabel = 'OPT'; end
fprintf('\nWINNER: %s (candidate %d), J = %.5g   (J of B = %.5g)\n',selLabel,iWin,Jwin,Jall(iB));
fprintf('  vs B: t1e-4 %g vs %g d, x2(T) %.3e vs %.3e, longest high-dose phase %g vs %g d, cumulative dose %.0f vs %.0f\n', ...
    H(iWin).T.t_sup,H(iB).T.t_sup,H(iWin).T.x2T,H(iB).T.x2T,H(iWin).T.maxrun,H(iB).T.maxrun,H(iWin).T.cumU,H(iB).T.cumU);
lpos = (log10(pvec(p))-log10(spec.lo))./(log10(spec.hi)-log10(spec.lo));
if any(lpos < 0.02 | lpos > 0.98)
    fprintf('NOTE: winner within 2%% of a search bound (log scale) in: %s\n',strjoin(spec.names(lpos < 0.02 | lpos > 0.98),', '));
end

fprintf('\nWp = Wp_gain*(s/M + wB)/(s + wB*A);   Wi = k*(s/wz + 1)/(s/wp + 1)\n');
fprintf('Weights %s: M=%.6g A=%.6g wB=%.6g k=%.6g Wp_gain=%.6g wz=%.6g wp=%.6g\n', ...
    selLabel,p.M,p.A,p.wB,p.k,p.Wp_gain,p.wz,p.wp);
fprintf('Input uncertainty covered by Wi: %.1f%% at low frequency, %.1f%% at high frequency.\n', ...
    100*p.k,100*p.k*p.wp/p.wz);

searchInfo = struct('objective',objdef.def,'n_candidates',numel(H),'n_valid',sum(okAll), ...
    'n_musyn',nMusynSearch,'J_winner',Jwin,'J_B',Jall(iB),'winner_index',iWin,'winner_tag',H(iWin).tag);
try
    searchInfo.table = [struct2table([H.p]'), table({H.tag}',okAll',Jall(:),[H.mu]',[H.nResyn]',[H.nMusyn]',[H.wall]', ...
        'VariableNames',{'tag','ok','J','mu_max','nResyn','nMusyn','wall_s'}), struct2table([H.T]')];
catch ME
    warning('Search table could not be built (%s); the history is in %s.',ME.message,cfg.hist_file);
end

sig = struct('version',VERSION,'S',S,'mp',mp,'p',p,'unc',unc,'max_resyn',cfg.max_resyn, ...
             'mc_seed',cfg.mc_seed,'musyn_maxiter',cfg.musyn_maxiter,'mu_max_accept',cfg.mu_max_accept, ...
             'resyn_gap',cfg.resyn_gap,'aw_beta',cfg.aw_beta,'xc_max',cfg.xc_max,'xo_floor',cfg.xo_floor);
E = load_compatible(cfg.cache_file,sig,'E');
if isempty(E), E = struct(); end
if ~isfield(E,'nom'), E.nom = H(iWin).sim; end           % the winner's run from the search is reused

%% 4) Winner: initial design and verification
fprintf('\n---------------- INITIAL DESIGN AND RUNTIME VERIFICATION ----------------\n');
D0 = H(iWin).D0;  dc0 = struct('x_op',x_pre(:),'D',D0);     % the design used in the search
verif = verify_design_math(ctx,p,D0,sprintf('design %s (initial design)',selLabel));

%% 5) Nominal closed loop (discrete; the run from the search, reused)
fprintf('\n---------------- NOMINAL CLOSED LOOP ----------------\n');
[E,simD] = cached_sim(E,'nom',cfg.cache_file,sig,ctx,p,D0,cfg.max_resyn,dc0,'discrete, nominal');
if simD.aborted, error('Nominal discrete run aborted: %s',simD.abortReason); end
metD = full_metrics(ctx,simD);
iseD = ise_of(ctx,simD);
assert(all(simD.U==S.u_min | simD.U==S.u_max),'Discrete dose constraint violated (value other than 50/200).');
fprintf('Dose levels check passed: only {%g,%g} used.\n',S.u_min,S.u_max);
Xrep = replay_schedule(mp,x_pre,simD.U,S.dt);
fprintf('Replay check (schedule replayed on the nominal plant vs closed-loop run): max |dx2| = %.2e\n', ...
    max(abs(Xrep(2,:)-simD.X(2,:))));
if cfg.max_resyn == 40 && S.x2_stop == 0 && isfield(EXPECT,selLabel)
    ex = EXPECT.(selLabel);
    okR = metD.t_sup == ex.t_sup && abs(metD.x2_T/ex.x2_T-1) < 2e-3 && ...
          abs(metD.cum_dose/ex.cum_dose-1) < 1e-4 && metD.n_hi == ex.n_hi;
    fprintf('Reproduction check vs mu_final_design.m: t1e-4 %g (expected %g), x2(T) %.4g (%.4g), cum. dose %.6g (%.6g), high-dose phases %d (%d) -> %s\n', ...
        metD.t_sup,ex.t_sup,metD.x2_T,ex.x2_T,metD.cum_dose,ex.cum_dose,metD.n_hi,ex.n_hi,ternary(okR,'PASS','MISMATCH'));
    if ~okR, warning('Reproduction MISMATCH: check that the model and settings equal those of mu_final_design.m before using any number.'); end
end

%% 6) Continuous mode
simC = []; metC = [];
if cfg.run_continuous
    ctxc = ctx; ctxc.S.discrete = false;
    [E,simC] = cached_sim(E,'cont',cfg.cache_file,sig,ctxc,p,D0,cfg.max_resyn,dc0,'continuous, nominal');
    metC = full_metrics(ctxc,simC);
end

%% 7) Redesign-rule check (diagnostic): nominal discrete loop of the winner under four rules
% Own cache file, independent of cfg.x2_stop / cfg.max_resyn, so changing the main
% rule does not recompute these runs.
RC = struct('label',{},'x2_stop',{},'max_resyn',{},'sim',{},'met',{});
if cfg.rule_check
    fprintf('\n---------------- REDESIGN-RULE CHECK (discrete, nominal) ----------------\n');
    rules = {0, 40; 1e-3, Inf; 1e-4, Inf; 0, Inf};               % {x2_stop, max_resyn}
    sigR = rmfield(sig,{'max_resyn'}); sigR.S = rmfield(S,'x2_stop');
    ER = load_compatible(cfg.rule_cache,sigR,'E');
    if isempty(ER), ER = struct(); end
    for r = 1:size(rules,1)
        ctxr = ctx; ctxr.S.x2_stop = rules{r,1};
        lab = rule_label(rules{r,1},rules{r,2});
        if rules{r,1} == S.x2_stop && isequal(rules{r,2},cfg.max_resyn)
            sr = simD; fprintf('  %-28s same as the main run\n',lab);
        else
            [ER,sr] = cached_sim(ER,sprintf('rule%d',r),cfg.rule_cache,sigR,ctxr,p,D0,rules{r,2},dc0,lab);
        end
        mr = []; if ~sr.aborted, mr = full_metrics(ctxr,sr); end
        RC(end+1) = struct('label',lab,'x2_stop',rules{r,1},'max_resyn',rules{r,2},'sim',sr,'met',mr); %#ok<SAGROW>
    end
    fprintf('\n================ REDESIGN RULES (design %s, discrete, nominal) ================\n',selLabel);
    fprintf('lastR = day of the last redesign, x2@lastR = x2 on that day, lastHi = last high-dose day\n');
    fprintf('%-26s %5s %6s %9s %6s %10s %8s %7s %4s %6s %4s %6s\n','rule','redes','lastR','x2@lastR','t1e-4', ...
        'x2(T)','RMSE%','cumU','nHi','maxHi','n>5','lastHi');
    for r = 1:numel(RC)
        sr = RC(r).sim; mr = RC(r).met;
        if isempty(mr), fprintf('%-26s ABORTED: %s\n',RC(r).label,sr.abortReason); continue; end
        dl = last_or_nan(sr.resynDays); x2l = NaN; if isfinite(dl), x2l = sr.X(2,find(t==dl,1)); end
        iLastHi = find(sr.U >= S.u_switch,1,'last'); lastHi = NaN; if ~isempty(iLastHi), lastHi = t(iLastHi); end
        fprintf('%-26s %5d %6g %9.2e %6g %10.3e %8.4f %7.0f %4d %6g %4d %6g\n',RC(r).label,mr.n_resyn,dl,x2l, ...
            mr.t_sup,mr.x2_T,100*mr.rmse_norm,mr.cum_dose,mr.n_hi,mr.hi_max,mr.n_gt5,lastHi);
    end
    if isfield(EXPECT,selLabel) && ~isempty(RC(1).met)            % rule 1 = cap 40, no stop = mu_final_design.m
        ex = EXPECT.(selLabel); m1 = RC(1).met;
        okR = m1.t_sup == ex.t_sup && abs(m1.x2_T/ex.x2_T-1) < 2e-3 && abs(m1.cum_dose/ex.cum_dose-1) < 1e-4 && m1.n_hi == ex.n_hi;
        fprintf('Reproduction check (cap 40) vs mu_final_design.m: %s\n',ternary(okR,'PASS','MISMATCH'));
    end
    fprintf('Main result below uses: %s\n',ruleMain);
end

%% 8) Report: FACTS block, windows, tail diagnostic
fprintf('\n================ BLOCK FOR FACTS.md 6.4 (design %s, a = %g) ================\n',selLabel,S.a);
fprintf('Weights\n  M        = %.6g\n  A        = %.6g\n  wB       = %.6g\n  k        = %.6g\n  Wp_gain  = %.6g\n  wz       = %.6g\n  wp       = %.6g\n', ...
    p.M,p.A,p.wB,p.k,p.Wp_gain,p.wz,p.wp);
fprintf('Design diagnostics (initial design)\n');
fprintf('  mu (peak)            = %.4g\n',D0.mu);
fprintf('  robstab margin       = [%.4g, %.4g]\n',verif.robstab_LB,verif.robstab_UB);
fprintf('  nominal stable       = %d\n  sampled-data stable  = %d\n',D0.nom_stable,D0.sd_stable);
fprintf('  Wi coverage          = %.1f%% (low freq.) / %.1f%% (high freq.)\n',100*p.k,100*p.k*p.wp/p.wz);
fprintf('  redesign rule        = %s; trigger = %g%% drift of x2, min. gap %g d\n',ruleMain,100*S.rel_thresh,cfg.resyn_gap);
fprintf('  ISE / ISE_open (u=0) = %.4g\n',iseD/ISE_open);
fprintf('Search\n  objective            : %s\n',objdef.def);
fprintf('  redesign rule in the search = %s (full closed loop for every candidate)\n',ruleMain);
fprintf('  candidates           = %d (%d valid): starting points B, D1, D2 + %d random + %d Bayesian optimization\n', ...
    numel(H),sum(okAll),sum(strcmp({H.tag},'rand')),sum(strcmp({H.tag},'bo')));
fprintf('  musyn calls          = %d\n',nMusynSearch);
fprintf('  J (winner %s)        = %.5g ; J (B) = %.5g\n',selLabel,Jwin,Jall(iB));
fprintf('  search ranges        : M [%g,%g], A [%g,%g], wB [%g,%g], k [%g,%g], Wp_gain [%g,%g], wz [%g,%g], wp [%g,%g]; %s\n', ...
    [spec.lo; spec.hi],spec.constraint);
print_facts_block('DISCRETE',metD,true,S.T_ref);
fprintf('  last redesign day          = %g\n',last_or_nan(simD.resynDays));
if ~isempty(metC)
    print_facts_block('CONTINUOUS',metC,false,S.T_ref);
end
print_window_listing(ctx,simD,selLabel);
tail_diagnostic(ctx,simD,ruleMain);

%% 9) CSV files
write_schedule_csv([cfg.csv_prefix selLabel '.csv'],t,simD,ref);
write_schedule_csv('u_schedule_mu.csv',t,simD,ref);             % name read by Miu_Syn+Observer.ipynb
if ~isempty(simC) && ~simC.aborted
    write_schedule_csv([cfg.csv_prefix selLabel '_continuous.csv'],t,simC,ref);
end
nU = numel(t)-1;                                                 % days 0..1999 for 'Drug Schedule.csv'
try
    writetable(table((0:nU-1)',simD.U(1:nU)','VariableNames',{'Day','u_miu'}),[cfg.csv_prefix 'miu_column.csv']);
catch ME
    warning('Could not write the u_miu column file: %s',ME.message);
end
fprintf('\nSaved: u_schedule_mu.csv, %s%s.csv (design %s), %smiu_column.csv\n',cfg.csv_prefix,selLabel,selLabel,cfg.csv_prefix);

%% 10) E1: schedule replay on the five groups (protocol of Uncertainty.ipynb)
x2T_E1 = nan(nG+1,1);  XG = cell(nG+1,1);
for g = 0:nG
    if g == 0, mpg = mp; else, mpg = perturb_params(mp,unc.groups{g,2},unc.groups{g,3}); end
    XG{g+1} = replay_schedule(mpg,x_pre,simD.U,S.dt);
    x2T_E1(g+1) = XG{g+1}(2,end);
end
sensF = mean(x2T_E1(2:end)/x2T_E1(1));

%% 11) E2: the same five groups with the controller in the loop
CLG = cell(nG,1);  MG = cell(nG,1);
if cfg.run_cl_groups
    fprintf('\n---------------- E2 runs (controller in the loop on the perturbed plant) ----------------\n');
    for g = 1:nG
        ctxg = ctx; mpg = perturb_params(mp,unc.groups{g,2},unc.groups{g,3});
        ctxg.f_plant = @(x,u) glioma_model(x,u,mpg);
        [E,CLG{g}] = cached_sim(E,sprintf('clg%d',g),cfg.cache_file,sig,ctxg,p,D0,cfg.max_resyn,dc0,unc.groups{g,1});
        MG{g} = full_metrics(ctx,CLG{g});
    end
end
fprintf('\n================ UNCERTAINTY GROUPS (design %s) ================\n',selLabel);
fprintf('E1 = nominal schedule replayed on the perturbed plant (as in Uncertainty.ipynb).\n');
fprintf('E2 = controller in the loop on the perturbed plant (controller and redesigns keep the nominal model).\n');
fprintf('%-28s %10s %10s %7s %8s %6s %6s %5s %6s\n',' group','x2(T) E1','x2(T) E2','t1e-4','ISE/nom','meanU','maxHi','n>5','redes');
fprintf('%-28s %10.3e %10.3e %7g %8.3g %6.1f %6g %5d %6d\n','nominal',x2T_E1(1),metD.x2_T,metD.t_sup,1, ...
    metD.u_mean,metD.hi_max,metD.n_gt5,metD.n_resyn);
for g = 1:nG
    if isempty(CLG{g})
        fprintf('%-28s %10.3e %10s\n',unc.groups{g,1},x2T_E1(g+1),'-'); continue;
    end
    if CLG{g}.aborted
        fprintf('%-28s %10.3e ABORTED: %s\n',unc.groups{g,1},x2T_E1(g+1),CLG{g}.abortReason); continue;
    end
    mg = MG{g};
    fprintf('%-28s %10.3e %10.3e %7g %8.3g %6.1f %6g %5d %6d\n',unc.groups{g,1},x2T_E1(g+1),mg.x2_T,mg.t_sup, ...
        ise_of(ctx,CLG{g})/iseD,mg.u_mean,mg.hi_max,mg.n_gt5,mg.n_resyn);
end
fprintf('E1 mean sensitivity factor (mean over groups of x2_group(T)/x2_nominal(T)) = %.1f\n',sensF);

%% 12) E3a: Monte Carlo, schedule replay
rs = RandStream('mt19937ar','Seed',cfg.mc_seed);
Z  = (2*rand(rs,numel(unc.mc_names),cfg.mc_replay_N)-1)';     % same samples as mu_final_design.m
MCrep = struct('x2T',nan(cfg.mc_replay_N,1),'reach',false(cfg.mc_replay_N,1),'ise_ratio',nan(cfg.mc_replay_N,1));
X2rep = nan(cfg.mc_replay_N,numel(t));
for s = 1:cfg.mc_replay_N
    mps = perturb_params(mp,unc.mc_names,unc.mc_frac.*Z(s,:));
    Xs = replay_schedule(mps,x_pre,simD.U,S.dt);
    MCrep.x2T(s) = Xs(2,end);  MCrep.reach(s) = any(Xs(2,:) < 1e-4);
    MCrep.ise_ratio(s) = trapz(t,((Xs(2,:)'-ref)/xs).^2)/t(end)/iseD;
    X2rep(s,:) = Xs(2,:);
end
if cfg.mc_replay_N > 0
    fprintf('\n================ E3a: MONTE CARLO, SCHEDULE REPLAY (design %s, N = %d, seed %d) ================\n', ...
        selLabel,cfg.mc_replay_N,cfg.mc_seed);
    fprintf('Nine parameters uniform inside the bounds of the five groups. reach = share of runs with x2 < 1e-4 at some day.\n');
    print_mc_row('replay',MCrep.x2T,MCrep.reach,MCrep.ise_ratio,[]);
end
MCcl = struct('x2T',[],'reach',[],'ise_ratio',[],'maxHi',[],'n_gt5',[],'t_sup',[],'aborted',[],'X2',[]);

%% 13) Save, figures
results = pack_results(VERSION,cfg,S,mp,unc,p,selLabel,verif,D0,metD,metC,RC,iseD,ISE_open, ...
                       x2T_E1,sensF,MG,MCrep,MCcl,toc(tAll)/60);
results.search = searchInfo;
save_all(cfg.out_mat,results,t,ref,x_pre,simD,simC,RC,CLG,XG,MCrep,MCcl,Z);

try
    make_figures(cfg,S,t,ref,simD,simC,RC,XG,CLG,rowNames,selLabel);
    fig_search(H,Jall,cfg);
    fig_montecarlo(t,ref,X2rep,MCcl,selLabel,cfg);
catch ME
    warning('Figure section stopped (%s). All numbers and CSV files above are already saved.',ME.message);
end

%% 14) E3b: Monte Carlo with the controller in the loop (last, longest part)
if cfg.mc_cl_N > 0
    n = cfg.mc_cl_N;
    fprintf('\n---------------- E3b runs (first %d samples of E3a, controller in the loop) ----------------\n',n);
    MCcl.x2T = nan(n,1); MCcl.reach = false(n,1); MCcl.ise_ratio = nan(n,1); MCcl.maxHi = nan(n,1);
    MCcl.n_gt5 = nan(n,1); MCcl.t_sup = nan(n,1); MCcl.aborted = false(n,1); MCcl.X2 = nan(n,numel(t));
    for s = 1:n
        ctxs = ctx; mps = perturb_params(mp,unc.mc_names,unc.mc_frac.*Z(s,:));
        ctxs.f_plant = @(x,u) glioma_model(x,u,mps);
        [E,sm] = cached_sim(E,sprintf('mc%03d',s),cfg.cache_file,sig,ctxs,p,D0,cfg.max_resyn,dc0,sprintf('sample %d/%d',s,n));
        MCcl.aborted(s) = sm.aborted;
        if sm.aborted, continue; end
        ms = full_metrics(ctx,sm);
        MCcl.x2T(s) = sm.X(2,end);  MCcl.reach(s) = any(sm.X(2,:) < 1e-4);
        MCcl.ise_ratio(s) = ise_of(ctx,sm)/iseD;  MCcl.maxHi(s) = ms.hi_max;  MCcl.n_gt5(s) = ms.n_gt5;
        MCcl.t_sup(s) = ms.t_sup;  MCcl.X2(s,:) = sm.X(2,:);
    end
    fprintf('\n================ E3b: MONTE CARLO, CONTROLLER IN THE LOOP (design %s, N = %d) ================\n',selLabel,n);
    fprintf('Same samples as the first %d of E3a. ISE ratios are relative to the nominal closed loop of this design.\n',n);
    okc = ~MCcl.aborted;
    print_mc_row('closed loop',MCcl.x2T(okc),MCcl.reach(okc),MCcl.ise_ratio(okc),MCcl);
    print_mc_row('replay',MCrep.x2T(1:n),MCrep.reach(1:n),MCrep.ise_ratio(1:n),[]);
    fprintf('aborted closed-loop runs: %d of %d\n',sum(MCcl.aborted),n);
    results = pack_results(VERSION,cfg,S,mp,unc,p,selLabel,verif,D0,metD,metC,RC,iseD,ISE_open, ...
                           x2T_E1,sensF,MG,MCrep,MCcl,toc(tAll)/60);
results.search = searchInfo;
    save_all(cfg.out_mat,results,t,ref,x_pre,simD,simC,RC,CLG,XG,MCrep,MCcl,Z);
    try, fig_montecarlo(t,ref,X2rep,MCcl,selLabel,cfg); catch ME, warning('Monte Carlo figure: %s',ME.message); end
end

fprintf('\nElapsed this session: %.1f min.\n',toc(tAll)/60);
diary off;

%% ========================== Local functions ================================

function v = ise_of(ctx,sim)
    v = trapz(ctx.t,((sim.X(2,:)'-ctx.ref)/ctx.xs).^2)/ctx.t(end);
end

function s = rule_label(x2_stop,max_resyn)
    parts = {};
    if x2_stop > 0, parts{end+1} = sprintf('stop at x2 < %g',x2_stop); end
    if isfinite(max_resyn), parts{end+1} = sprintf('cap %g',max_resyn); end
    if isempty(parts), s = 'no stop, no cap'; else, s = strjoin(parts,', '); end
end

function v = last_or_nan(x)
    if isempty(x), v = NaN; else, v = x(end); end
end

function tail_diagnostic(ctx,sim,label)
    % Why x2 stays above the reference late in the run: after the last high-dose day,
    % is the clipped continuous command below the threshold (so only u = 50 is given)?
    S = ctx.S; t = ctx.t(:); U = sim.U(:); uc = sim.Ucmd(:);
    iLast = find(U >= S.u_switch,1,'last');
    fprintf('\nTAIL DIAGNOSTIC (%s): ',label);
    if isempty(iLast) || iLast >= numel(t)
        fprintf('no high-dose day, or the last day is high-dose.\n'); return;
    end
    idx = iLast+1:numel(t);
    e = 100*(sim.X(2,idx)'-ctx.ref(idx))/ctx.xs;
    fprintf('last high-dose day %g, last redesign day %g\n',t(iLast),last_or_nan(sim.resynDays));
    fprintf('  after day %g: clipped command min / mean / max = %.1f / %.1f / %.1f (threshold %g)\n', ...
        t(iLast),min(uc(idx)),mean(uc(idx)),max(uc(idx)),S.u_switch);
    fprintf('  after day %g: error 100*(x2-ref)/x2_pre min / mean / max = %.4f / %.4f / %.4f, share of days with x2 > ref = %.2f\n', ...
        t(iLast),min(e),mean(e),max(e),mean(e > 0));
end

function print_mc_row(name,x2T,reach,iser,MCcl)
    n = numel(x2T);
    k = sum(reach);
    [lo,hi] = cp_interval(k,n);
    fprintf('%-12s N=%3d  x2(T) median %.3e  mean %.3e  worst %.3e | reach %d/%d = %.2f (95%% CI %.2f-%.2f) | ISE ratio mean %.3g worst %.3g', ...
        name,n,median(x2T),mean(x2T),max(x2T),k,n,k/max(n,1),lo,hi,mean(iser),max(iser));
    if ~isempty(MCcl)
        ok = ~MCcl.aborted;
        fprintf(' | longest high-dose phase mean %.1f worst %g | runs with a phase > 5 d: %d | t1e-4 median %g',...
            mean(MCcl.maxHi(ok)),max(MCcl.maxHi(ok)),sum(MCcl.n_gt5(ok) > 0),median(MCcl.t_sup(ok),'omitnan'));
    end
    fprintf('\n');
end

function [lo,hi] = cp_interval(k,n)
    % exact (Clopper-Pearson) 95% interval of a binomial proportion
    if n == 0, lo = NaN; hi = NaN; return; end
    if k == 0, lo = 0; else, lo = betainv(0.025,k,n-k+1); end
    if k == n, hi = 1; else, hi = betainv(0.975,k+1,n-k); end
end

function r = pack_results(VERSION,cfg,S,mp,unc,p,selLabel,verif,D0,metD,metC,RC,iseD,ISE_open,x2T_E1,sensF,MG,MCrep,MCcl,elapsed)
    r = struct('version',VERSION,'cfg',cfg,'S',S,'mp',mp,'unc',unc,'weights',p,'design',selLabel, ...
        'verif',verif,'mu_initial',D0.mu,'nominal_stable',D0.nom_stable,'sampled_stable',D0.sd_stable, ...
        'metrics_discrete',metD,'metrics_continuous',metC,'rule_check',{RC}, ...
        'ISE_discrete',iseD,'ISE_open',ISE_open,'x2T_E1',x2T_E1,'sensitivity_factor',sensF, ...
        'metrics_E2',{MG},'MCrep',MCrep,'MCcl',MCcl,'elapsed_min',elapsed);
end

function save_all(file,results,t,ref,x_pre,simD,simC,RC,CLG,XG,MCrep,MCcl,Z) %#ok<INUSD>
    try
        save(file,'results','t','ref','x_pre','simD','simC','RC','CLG','XG','MCrep','MCcl','Z');
        fprintf('Saved %s\n',file);
    catch ME
        warning('Could not save %s: %s',file,ME.message);
    end
end

function make_figures(cfg,S,t,ref,simD,simC,RC,XG,CLG,rowNames,selLabel)
    pf = cfg.fig_prefix;  nG = numel(CLG);
    % states and tracking (linear + log)
    fg = figure('Color','w','Name','mu_opt: states');
    subplot(3,1,1); hold on; grid on;
    plot(t,simD.X(1,:),t,simD.X(2,:),t,simD.X(3,:),'LineWidth',1.1);
    legend('x_1 glia','x_2 glioma','x_3 neurons','Location','best'); ylabel('Normalized density');
    title(sprintf('Design %s, discrete dosing',selLabel));
    subplot(3,1,2); hold on; grid on;
    plot(t,ref,'k--','LineWidth',1.0); plot(t,simD.X(2,:),'LineWidth',1.1);
    legend('Reference','x_2 glioma','Location','best'); ylabel('x_2');
    subplot(3,1,3);
    semilogy(t,posv(ref),'k--','LineWidth',1.0); hold on; grid on;
    semilogy(t,posv(simD.X(2,:)),'LineWidth',1.1);
    yline(1e-4,':','HandleVisibility','off'); ylim([1e-6 1]);
    xlabel('Time [days]'); ylabel('x_2 (log scale)');
    save_fig(fg,[pf 'states'],cfg);

    fg = figure('Color','w','Name','mu_opt: dose and drug concentration');
    subplot(2,1,1); stairs(t,simD.U,'LineWidth',1.1); grid on; ylim([0 S.u_max+10]);
    ylabel('u [mg m^{-2} day^{-1}]'); title(sprintf('Design %s: maintenance phase 50, high-dose phase 200',selLabel));
    subplot(2,1,2); plot(t,simD.X(4,:),'LineWidth',1.1); grid on;
    xlabel('Time [days]'); ylabel('x_4');
    save_fig(fg,[pf 'dose_x4'],cfg);

    if ~isempty(simC) && ~simC.aborted
        fg = figure('Color','w','Name','mu_opt: continuous mode');
        subplot(2,1,1); hold on; grid on;
        plot(t,ref,'k--','LineWidth',1.0); plot(t,simC.X(2,:),'LineWidth',1.1);
        legend('Reference','x_2 glioma','Location','best'); ylabel('x_2');
        title(sprintf('Design %s, continuous dosing',selLabel));
        subplot(2,1,2); plot(t,simC.U,'LineWidth',1.1); grid on; ylim([0 S.u_max+10]);
        xlabel('Time [days]'); ylabel('u [mg m^{-2} day^{-1}]');
        save_fig(fg,[pf 'continuous'],cfg);
    end

    if ~isempty(RC)
        fg = figure('Color','w','Name','mu_opt: redesign-rule check');
        subplot(2,1,1);
        semilogy(t,posv(ref),'k--','LineWidth',1.0); hold on; grid on; lg = {'Reference'};
        for r = 1:numel(RC)
            if ~RC(r).sim.aborted, semilogy(t,posv(RC(r).sim.X(2,:)),'LineWidth',1.1); lg{end+1} = RC(r).label; end %#ok<AGROW>
        end
        yline(1e-4,':','HandleVisibility','off'); ylim([1e-8 1]);
        legend(lg,'Location','best'); ylabel('x_2 (log scale)');
        title(sprintf('Design %s: redesign rules (diagnostic)',selLabel));
        subplot(2,1,2); hold on; grid on; k = 0; lg = {};
        for r = 1:numel(RC)
            if RC(r).sim.aborted, continue; end
            k = k+1; hi = double(RC(r).sim.U >= S.u_switch);
            stairs(t,k + 0.8*hi,'LineWidth',1.0); lg{end+1} = RC(r).label; %#ok<AGROW>
        end
        set(gca,'YTick',1:k,'YTickLabel',lg); ylim([0.5 k+1]);
        xlabel('Time [days]'); title('High-dose days (raised line = 200)');
        save_fig(fg,[pf 'redesign_rule_check'],cfg);
    end

    fg = figure('Color','w','Name','mu_opt: uncertainty groups, schedule replay');
    semilogy(t,posv(XG{1}(2,:)),'k-','LineWidth',1.3); hold on; grid on;
    for g = 1:nG, semilogy(t,posv(XG{g+1}(2,:)),'LineWidth',1.0); end
    semilogy(t,posv(ref),'k--','LineWidth',1.0); ylim([1e-6 1]);
    legend([rowNames',{'Reference'}],'Location','best');
    xlabel('Time [days]'); ylabel('x_2 (log scale)');
    title(sprintf('Design %s: nominal schedule on the perturbed plant',selLabel));
    save_fig(fg,[pf 'unc_replay'],cfg);

    if ~isempty(CLG{1})
        fg = figure('Color','w','Name','mu_opt: uncertainty groups, controller in the loop');
        semilogy(t,posv(simD.X(2,:)),'k-','LineWidth',1.3); hold on; grid on;
        for g = 1:nG, if ~CLG{g}.aborted, semilogy(t,posv(CLG{g}.X(2,:)),'LineWidth',1.0); end, end
        semilogy(t,posv(ref),'k--','LineWidth',1.0); ylim([1e-6 1]);
        legend([rowNames',{'Reference'}],'Location','best');
        xlabel('Time [days]'); ylabel('x_2 (log scale)');
        title(sprintf('Design %s: controller in the loop on the perturbed plant',selLabel));
        save_fig(fg,[pf 'unc_closedloop'],cfg);
    end

    fg = figure('Color','w','Name','mu_opt: mu per design'); hold on; grid on;
    stairs([0, simD.resynDays(:)'],simD.mu_list(:)','-o','LineWidth',1.1,'MarkerSize',4);
    yline(1,'k--','HandleVisibility','off');
    xlabel('Day of (re)design'); ylabel('\mu peak of the linear design');
    title(sprintf('Design %s: \\mu of the initial design and of each redesign',selLabel));
    save_fig(fg,[pf 'mu'],cfg);
end

function dc = empty_dc()
    dc = struct('x_op',{},'D',{});
end

function [rl,st,en] = run_lengths(hi)
    % lengths (in samples = days) of the runs of true values in a logical vector
    d  = diff([0; double(hi(:)); 0]);
    st = find(d == 1);  en = find(d == -1)-1;
    rl = en-st+1;
end

function [D,dc] = get_design(ctx,p,xop,dc)
    for i = 1:numel(dc)   % reuse an identical earlier design (same weights, same operating point)
        if norm(dc(i).x_op-xop(:)) <= 1e-12*max(1,norm(xop)), D = dc(i).D; return; end
    end
    D = design_mu_controller(ctx,p,xop);
    dc(end+1) = struct('x_op',xop(:),'D',D);
end

function D = new_design(xop)
    D = struct('ok',false,'mu',Inf,'msg','','x_op',xop(:),'xo',NaN,'Ac',[],'Bc',[],'Cc',[],'Dc',[], ...
               'Ad',[],'Bd',[],'Kn',[],'nom_stable',false,'sd_stable',false,'n_musyn',0,'t_design',0);
end

function D = design_mu_controller(ctx,p,xop)
    % UNCHANGED design logic (same as mu_final_compare.m): scaled 3-state plant
    % (x1,x2,x4), performance weight Wp, input multiplicative uncertainty weighted
    % by Wi, musyn, conversion to physical units, exact ZOH discretization.
    % The design always uses the NOMINAL model ctx.f.
    S = ctx.S; cfg = ctx.cfg; D = new_design(xop); tl = tic;
    try
        xo = max(xop(2),cfg.xo_floor); D.xo = xo;
        A = lin_jac(ctx.f,xop); idx = [1 2 4]; Ar = A(idx,idx);   % x3 exactly decoupled
        Gn = ss(Ar,[0;0;S.u_max],[0 1/xo 0],0);
        [Wp,Wi] = make_weights(p);
        Punc = build_generalized_plant(Gn,Wp,Wi);
        assert(isequal(size(Punc),[2 2]),'Unexpected generalized-plant size.');
        D.n_musyn = 1;
        [K,mu] = musyn(Punc,1,1,ctx.mopt);          % measurement = e, control = u
        D.mu = max(mu(:));
        K = ss(K); [Ac,Bc,Cc,Dc] = ssdata(K);
        assert(all(isfinite([Ac(:);Bc(:);Cc(:);Dc(:)])),'Non-finite controller.');
        D.Kn = K;
        D.nom_stable = isstable(feedback(Gn*K,1));
        D.Ac = Ac; D.Bc = Bc/xo; D.Cc = Cc*S.u_max; D.Dc = Dc*S.u_max/xo;   % physical units
        n = size(Ac,1);
        Md = expm([D.Ac D.Bc; zeros(1,n+1)]*S.dt);                           % exact ZOH
        assert(all(isfinite(Md(:))),'Non-finite discretization.');
        D.Ad = Md(1:n,1:n); D.Bd = Md(1:n,n+1);
        Gp = ss(Ar,[0;0;1],[0 1 0],0);
        D.sd_stable = isstable(feedback(c2d(Gp,S.dt,'zoh')*ss(D.Ad,D.Bd,D.Cc,D.Dc,S.dt),1));
        if ~D.nom_stable,      D.msg = 'nominal loop unstable';
        elseif ~D.sd_stable,   D.msg = 'sampled-data loop unstable';
        elseif ~(isfinite(D.mu) && D.mu <= cfg.mu_max_accept), D.msg = 'mu too large';
        else, D.ok = true; D.msg = 'ok'; end
    catch ME
        D.ok = false; D.msg = ME.message;
    end
    D.t_design = toc(tl);
end

function Punc = build_generalized_plant(Gn,Wp,Wi)
    % u_act = (1 + Wi*Delta) u ; e = r - Gn*u_act ; z1 = Wp*e ; [r;u] -> [z1;e]
    Delta = ultidyn('Delta',[1 1]);
    Gunc  = Gn*(1+Delta*Wi);
    Pe    = [ss(1), -Gunc];
    Punc  = [Wp; ss(1)]*Pe;
end

function [Wp,Wi] = make_weights(p)
    Wp = ss(zpk(-p.wB*p.M,-p.wB*p.A,p.Wp_gain/p.M));
    Wi = ss(zpk(-p.wz,-p.wp,p.k*p.wp/p.wz));
end

function [sim,dc] = simulate_closed_loop(ctx,p,D0,maxResyn,dc)
    % Adaptive closed loop. Same mechanics as mu_final_compare.m (redesign on 10%
    % drift of x2, bumpless transfer, anti-windup on the continuous saturation, exact
    % ZOH controller update). Dose rule: single threshold S.u_switch when
    % ctx.S.discrete is true, the clipped command itself otherwise.
    % The plant is stepped with ctx.f_plant (nominal or perturbed); every redesign
    % linearizes the NOMINAL model ctx.f at the current state.
    S = ctx.S; cfg = ctx.cfg; fpl = ctx.f_plant; t = ctx.t; ref = ctx.ref; N = numel(t);
    n0 = numel(dc);
    X = nan(4,N); U = nan(1,N); Ucmd = nan(1,N); X(:,1) = ctx.x_pre;
    ctrl = D0; xc = zeros(size(ctrl.Ac,1),1); xlin = ctx.x_pre(2);
    nResyn = 0; nFail = 0; nBF = 0; lastResyn = -inf; mu_list = D0.mu; resynDays = [];
    aborted = false; why = '';
    for kk = 1:N
        xk = X(:,kk); e = ref(kk)-xk(2);
        if nResyn < maxResyn && xk(2) >= S.x2_stop && (t(kk)-lastResyn) >= cfg.resyn_gap && ...
                abs(xk(2)-xlin) > S.rel_thresh*max(xlin,S.xlin_floor)
            nResyn = nResyn+1; lastResyn = t(kk); xlin = xk(2);
            [Dn,dc] = get_design(ctx,p,xk,dc);
            if Dn.ok
                uold = ctrl.Cc*xc + ctrl.Dc*e;
                if ~isfinite(uold), uold = 0; end
                utar = min(max(uold,0),S.u_max);
                [xnew,okb] = bumpless_state(Dn,utar,e,cfg.xc_max);
                if ~okb, xnew = zeros(size(Dn.Ac,1),1); nBF = nBF+1; end
                ctrl = Dn; xc = xnew; mu_list(end+1) = Dn.mu; resynDays(end+1) = t(kk); %#ok<AGROW>
            else
                nFail = nFail+1;                      % keep the previous controller
            end
        end
        ucmd = ctrl.Cc*xc + ctrl.Dc*e;
        if ~isfinite(ucmd), aborted = true; why = sprintf('non-finite controller output at day %g',t(kk)); break; end
        uclip = min(max(ucmd,0),S.u_max);
        Ucmd(kk) = uclip;                         % logged only (tail diagnostic)
        if cfg.aw_beta > 0 && uclip ~= ucmd
            cc2 = ctrl.Cc*ctrl.Cc';
            if cc2 > 0 && isfinite(cc2), xc = xc + cfg.aw_beta*(ctrl.Cc'*((uclip-ucmd)/cc2)); end
        end
        if S.discrete
            if uclip >= S.u_switch, uapp = S.u_max; else, uapp = S.u_min; end
        else
            uapp = uclip;
        end
        U(kk) = uapp;
        if kk == N, break; end
        xn = rk4_step(fpl,xk,uapp,S.dt);
        if any(~isfinite(xn)) || any(xn < -1e-3) || any(xn > 1e4)
            aborted = true; why = sprintf('nonphysical/unstable state at day %g',t(kk)); U(kk+1:end) = NaN; break;
        end
        X(:,kk+1) = xn;
        xc = ctrl.Ad*xc + ctrl.Bd*e;
        if any(~isfinite(xc)) || norm(xc) > cfg.xc_max
            aborted = true; why = sprintf('controller state blow-up at day %g',t(kk)); X(:,kk+2:end) = NaN; break;
        end
    end
    sim = struct('X',X,'U',U,'Ucmd',Ucmd,'aborted',aborted,'abortReason',why,'nResyn',nResyn,'nFail',nFail, ...
                 'nMusyn',numel(dc)-n0,'mu_list',mu_list,'resynDays',resynDays,'nBumpFallback',nBF);
end

function sim = empty_sim(N,mu,why)
    sim = struct('X',nan(4,N),'U',nan(1,N),'Ucmd',nan(1,N),'aborted',true,'abortReason',why,'nResyn',0,'nFail',0, ...
                 'nMusyn',0,'mu_list',mu,'resynDays',[],'nBumpFallback',0);
end

function [xnew,ok] = bumpless_state(D,utarget,e,xc_max)
    c = D.Cc; c2 = c*c';
    if c2 > 0 && isfinite(c2), xnew = c'*((utarget-D.Dc*e)/c2); else, xnew = zeros(size(D.Ac,1),1); end
    ok = all(isfinite(xnew)) && norm(xnew) <= xc_max;
end

function [E,sim] = cached_sim(E,key,file,sig,ctxX,p,D0,maxResyn,dc0,label)
    % one full closed-loop simulation, cached on disk under E.(key)
    if isfield(E,key)
        sim = E.(key); fprintf('  %-28s cached\n',label); return;
    end
    tl = tic;
    [sim,~] = simulate_closed_loop(ctxX,p,D0,maxResyn,dc0);
    E.(key) = sim;
    try, save_struct(file,'E',E,sig); catch ME, warning('Could not save %s: %s',file,ME.message); end
    fprintf('  %-28s %d redesigns (%d failed), %.1f min%s\n',label,sim.nResyn,sim.nFail,toc(tl)/60, ...
        ternary(sim.aborted,['  ABORTED: ' sim.abortReason],''));
end

function m = full_metrics(ctx,sim)
    % Same definitions as report() in mu_final_compare.m.
    S = ctx.S; t = ctx.t(:); N = numel(t); xs = ctx.xs;
    m = struct('aborted',double(sim.aborted),'t_sup',NaN,'t_rec',NaN,'x1_T',NaN,'x2_T',NaN,'x3_T',NaN, ...
        'rmse_norm',NaN,'u_mean',NaN,'u_max',NaN,'u_min',NaN,'cum_dose',NaN, ...
        'n_resyn',sim.nResyn,'n_fail',sim.nFail,'mu_max',max(sim.mu_list),'mu_first',sim.mu_list(1), ...
        'n_hi',0,'hi_mean',NaN,'hi_min',NaN,'hi_max',0,'n_gt5',0,'n_lo',0,'lo_mean',NaN,'lo_min',NaN,'lo_max',NaN, ...
        'frac_hi',NaN);
    if sim.aborted || any(~isfinite(sim.X(:))) || any(~isfinite(sim.U)), m.aborted = 1; return; end
    X = sim.X; U = sim.U(:);
    i1 = find(X(2,:) < 1e-4,1,'first');
    if ~isempty(i1), m.t_sup = t(i1); end
    i2 = find(X(1,:) < 0.99*S.y0(1),1,'last');
    if isempty(i2), m.t_rec = 0; elseif i2 < N, m.t_rec = t(i2+1); end
    e = X(2,:)'-ctx.ref;
    m.x1_T = X(1,end); m.x2_T = X(2,end); m.x3_T = X(3,end);
    m.rmse_norm = sqrt(mean(e.^2))/xs;
    m.u_mean = mean(U); m.u_max = max(U); m.u_min = min(U); m.cum_dose = trapz(t,U);
    [il,st,en] = run_lengths(U >= S.u_switch);
    il = il*S.dt;
    m.n_hi = numel(il); m.frac_hi = mean(U >= S.u_switch);
    if ~isempty(il)
        m.hi_mean = mean(il); m.hi_min = min(il); m.hi_max = max(il); m.n_gt5 = sum(il > 5);
    end
    if numel(st) >= 2
        ml = (st(2:end)-en(1:end-1)-1)*S.dt;
        m.n_lo = numel(ml); m.lo_mean = mean(ml); m.lo_min = min(ml); m.lo_max = max(ml);
    end
end

function print_facts_block(tag,m,discrete,Tend)
    fprintf('%s mode\n',tag);
    if m.aborted, fprintf('  ABORTED - no numbers\n'); return; end
    fprintf('  time to x2 < 1e-4 [d]      = %g\n',m.t_sup);
    fprintf('  glial recovery 99 %% [d]    = %g\n',m.t_rec);
    fprintf('  x3(T = %g)               = %.4f\n',Tend,m.x3_T);
    fprintf('  x2(T = %g)               = %.4g\n',Tend,m.x2_T);
    fprintf('  tracking RMSE / x2_pre     = %.5f\n',m.rmse_norm);
    fprintf('  dose: mean / max / min     = %.2f / %.1f / %.1f\n',m.u_mean,m.u_max,m.u_min);
    fprintf('  cumulative dose            = %.6g\n',m.cum_dose);
    if discrete
        fprintf('  high-dose phases: count / mean / range [d]   = %d / %.2f / %g-%g\n',m.n_hi,m.hi_mean,m.hi_min,m.hi_max);
        fprintf('  maintenance phases: count / mean / range [d] = %d / %.2f / %g-%g\n',m.n_lo,m.lo_mean,m.lo_min,m.lo_max);
        fprintf('  longest continuous high-dose interval [d]    = %g\n',m.hi_max);
        fprintf('  high-dose phases longer than 5 days          = %d\n',m.n_gt5);
    end
    fprintf('  redesigns (failed)         = %d (%d)\n',m.n_resyn,m.n_fail);
    fprintf('  max mu                     = %.4g\n',m.mu_max);
end

function print_window_listing(ctx,sim,label)
    % Numbers behind the plots of one design, 100-day windows.
    t = ctx.t(:); U = sim.U(:); S = ctx.S;
    e = 100*(sim.X(2,:)'-ctx.ref)/ctx.xs;                 % error in % of x2_pre
    fprintf('\n================ DESIGN %s: ERROR AND DOSE OVER TIME ================\n',label);
    fprintf('err = 100*(x2-ref)/x2_pre; mean/min/max in each 100-day window; hi = share of days at 200; sw = sign changes of err\n');
    fprintf('%-11s %9s %9s %9s %9s %5s %4s\n','window','x2 end','err mean','err min','err max','hi','sw');
    edges = 0:100:t(end);
    for k = 1:numel(edges)-1
        idx = t >= edges(k) & t < edges(k+1);
        if ~any(idx), continue; end
        ee = e(idx); x2w = sim.X(2,idx);
        fprintf('%4d-%-6d %9.2e %9.4f %9.4f %9.4f %5.2f %4d\n',edges(k),edges(k+1),x2w(end),mean(ee),min(ee),max(ee), ...
            mean(U(idx) >= S.u_switch),sum(diff(sign(ee)) ~= 0));
    end
    [rl,st] = run_lengths(U >= S.u_switch);
    fprintf('High-dose phases as start_day:length_days (%d phases):\n',numel(st));
    if ~isempty(st)
        pairs = [t(st)'; rl(:)'];
        fprintf('%g:%g ',pairs(:,1:min(120,size(pairs,2)))); fprintf('\n');
        if size(pairs,2) > 120, fprintf('(first 120 of %d phases listed)\n',size(pairs,2)); end
    end
end

function write_schedule_csv(fname,t,sim,ref)
    try
        writetable(table(t(:),sim.U(:),sim.X(2,:)',ref(:),'VariableNames', ...
            {'time_day','u_mg_per_m2_per_day','x2_tumor','x2_reference'}),fname);
    catch ME
        warning('Could not write %s: %s',fname,ME.message);
    end
end

function mpg = perturb_params(mp,names,fr)
    % multiply the listed parameters by (1 + fraction)
    mpg = mp;
    for i = 1:numel(names), mpg.(names{i}) = mp.(names{i})*(1+fr(i)); end
end

function X = replay_schedule(mp,x0,U,dt)
    % Open-loop replay of a dose schedule, as simulate() in Uncertainty.ipynb
    % (RK4, dt = 1 day, x2 clipped at zero).
    N = numel(U); X = zeros(4,N); X(:,1) = x0;
    for k = 1:N-1
        x = X(:,k); u = U(k);
        k1 = glioma_model(x,u,mp);          k2 = glioma_model(x+dt/2*k1,u,mp);
        k3 = glioma_model(x+dt/2*k2,u,mp);  k4 = glioma_model(x+dt*k3,u,mp);
        xn = x + dt/6*(k1+2*k2+2*k3+k4);
        xn(2) = max(xn(2),0);
        X(:,k+1) = xn;
    end
end

function fig_montecarlo(t,ref,X2sel,MCcl,selLabel,cfg)
    % x2 band of the Monte Carlo runs of the selected design (log axis)
    if isempty(X2sel), return; end
    fg = figure('Color','w','Name','mu_opt: Monte Carlo'); hold on; grid on;
    qrp = pctl_rows(X2sel,[5 50 95]);
    fill([t; flipud(t)],[max(qrp(1,:)',1e-12); flipud(max(qrp(3,:)',1e-12))],[0.80 0.86 1.00],'EdgeColor','none');
    plot(t,posv(qrp(2,:)),'b-','LineWidth',1.1);
    lg = {'Replay, 5-95%','Replay, median'};
    if ~isempty(MCcl.aborted) && any(~MCcl.aborted)
        qc = pctl_rows(MCcl.X2(~MCcl.aborted,:),[5 50 95]);
        plot(t,posv(qc(1,:)),'r:','LineWidth',1.0,'HandleVisibility','off');
        plot(t,posv(qc(3,:)),'r:','LineWidth',1.0);
        plot(t,posv(qc(2,:)),'r-','LineWidth',1.1);
        lg = [lg,{'Closed loop, 5-95%','Closed loop, median'}];
    end
    plot(t,posv(ref),'k--','LineWidth',1.0); lg = [lg,{'Reference'}];
    set(gca,'YScale','log'); ylim([1e-6 1]);
    legend(lg,'Location','best');
    xlabel('Time [days]'); ylabel('x_2 (log scale)');
    title(sprintf('Design %s: Monte Carlo over the model parameters',selLabel));
    save_fig(fg,[cfg.fig_prefix 'montecarlo'],cfg);
end

function v = posv(v)
    % values that cannot be drawn on a log axis become NaN (gaps), not a floor
    v(~(v > 0)) = NaN;
end

function q = pctl_rows(Xm,qs)
    % percentiles over the rows of a matrix (nearest rank), one output row per percentile
    Xs = sort(Xm,1); n = size(Xm,1); q = zeros(numel(qs),size(Xm,2));
    for j = 1:numel(qs)
        idx = max(1,min(n,round(qs(j)/100*(n-1))+1));
        q(j,:) = Xs(idx,:);
    end
end

function save_fig(fg,base,cfg)
    if ~cfg.save_figs, return; end
    try
        exportgraphics(fg,[base '.png'],'Resolution',cfg.fig_dpi);
        exportgraphics(fg,[base '.pdf'],'ContentType','vector');
    catch ME
        try
            print(fg,[base '.png'],'-dpng',sprintf('-r%d',cfg.fig_dpi));
            print(fg,[base '.pdf'],'-dpdf','-bestfit');
        catch ME2
            warning('Could not save figure %s: %s | %s',base,ME.message,ME2.message);
        end
    end
end

function v = verify_design_math(ctx,p,D,label)
    % UNCHANGED from mu_discrete_weight_optimization_v2.m: identity checks of the
    % weights, the generalized plant, the scaling and the ZOH discretization.
    S = ctx.S; w = logspace(-4,1,60);
    fprintf('Verifying %s ...\n',label);
    A = lin_jac(ctx.f,D.x_op); idx = [1 2 4]; Ar = A(idx,idx); xo = D.xo;
    Gn = ss(Ar,[0;0;S.u_max],[0 1/xo 0],0);  Gp = ss(Ar,[0;0;1],[0 1 0],0);
    [Wp,Wi] = make_weights(p);
    relerr = @(a,b) max(abs(a(:)-b(:)))/max(abs(b(:)));
    sj = 1i*w(:);
    Wp_ref = p.Wp_gain*(sj/p.M+p.wB)./(sj+p.wB*p.A);
    Wi_ref = p.k*(sj/p.wz+1)./(sj/p.wp+1);
    Wpj = squeeze(freqresp(Wp,w)); Wij = squeeze(freqresp(Wi,w)); Gj = squeeze(freqresp(Gn,w));
    v.err_weights = max(relerr(Wpj,Wp_ref),relerr(Wij,Wi_ref));
    Punc = build_generalized_plant(Gn,Wp,Wi);
    v.err_interconnect = 0; v.err_vs_connect = 0; usub_ok = true;
    Gc = Gn; Gc.InputName = 'uw'; Gc.OutputName = 'y';
    Wpc = Wp; Wpc.InputName = 'e'; Wpc.OutputName = 'z1';
    Wic = Wi; Wic.InputName = 'u'; Wic.OutputName = 'zd';
    s1 = sumblk('uw = u + wd'); s2 = sumblk('e = r - y');
    for d = [-1 -0.4 0 0.7 1]
        Gd_ = Gj.*(1+Wij*d);
        Pa = cat(3,Wpj,-Wpj.*Gd_, ones(size(Gd_)),-Gd_);
        Dg = ss(d); Dg.InputName = 'zd'; Dg.OutputName = 'wd';
        Pc = connect(Gc,Wpc,Wic,Dg,s1,s2,{'r','u'},{'z1','e'});
        Pcf = freqresp(Pc,w);
        c = [squeeze(Pcf(1,1,:)) squeeze(Pcf(1,2,:)) squeeze(Pcf(2,1,:)) squeeze(Pcf(2,2,:))];
        a = reshape(Pa,numel(w),4);
        v.err_vs_connect = max(v.err_vs_connect,relerr(c,a));
        try
            Pd = ss(usubs(Punc,'Delta',d)); Pdf = freqresp(Pd,w);
            u_ = [squeeze(Pdf(1,1,:)) squeeze(Pdf(1,2,:)) squeeze(Pdf(2,1,:)) squeeze(Pdf(2,2,:))];
            v.err_interconnect = max(v.err_interconnect,relerr(u_,a));
        catch
            usub_ok = false;
        end
    end
    if ~usub_ok, v.err_interconnect = NaN; warning('usubs check on uncertain Punc could not be run.'); end
    Kphys = ss(D.Ac,D.Bc,D.Cc,D.Dc);
    v.err_scaling = relerr(squeeze(freqresp(Gp*Kphys,w)),squeeze(freqresp(Gn*D.Kn,w)));
    Kd = c2d(Kphys,S.dt,'zoh'); [Adc,Bdc] = ssdata(Kd);
    v.err_zoh = max(norm(Adc-D.Ad,'fro')/max(1,norm(Adc,'fro')),norm(Bdc-D.Bd)/max(1,norm(Bdc)));
    v.nominal_stable = D.nom_stable; v.sampled_stable = D.sd_stable; v.mu = D.mu;
    v.robstab_LB = NaN; v.robstab_UB = NaN;
    try
        sm = robstab(lft(Punc,D.Kn)); v.robstab_LB = sm.LowerBound; v.robstab_UB = sm.UpperBound;
    catch ME
        warning('robstab failed: %s',ME.message);
    end
    fprintf('  weights vs formula     : %.2e\n  Punc vs analytic       : %.2e\n  Punc vs connect/sumblk : %.2e\n',v.err_weights,v.err_interconnect,v.err_vs_connect);
    fprintf('  controller scaling     : %.2e\n  ZOH vs c2d             : %.2e\n',v.err_scaling,v.err_zoh);
    fprintf('  nominal stable: %d | sampled-data stable: %d | mu: %.3g | robstab margin [%.3g, %.3g]\n', ...
        v.nominal_stable,v.sampled_stable,v.mu,v.robstab_LB,v.robstab_UB);
    tol     = 1e-6;   % exact-identity checks
    tol_zoh = 1e-4;   % expm vs c2d use different algorithms
    assert(v.err_weights < tol && v.err_vs_connect < tol && v.err_scaling < tol, ...
        'Mathematical verification FAILED for %s - do not trust results.',label);
    assert(v.err_zoh < tol_zoh, ...
        'ZOH discretization check FAILED for %s (err=%.2e, tol=%.2e) - do not trust results.',label,v.err_zoh,tol_zoh);
    assert(isnan(v.err_interconnect) || v.err_interconnect < tol,'Uncertain interconnection check FAILED for %s.',label);
    fprintf('  -> PASS (identity checks)\n');
end

function X = traj_const_u(f,x0,N,dt,u)
    X = zeros(4,N); X(:,1) = x0;
    for k = 1:N-1, X(:,k+1) = rk4_step(f,X(:,k),u,dt); end
end

function save_struct(file,name,data,sig)
    s = struct(); s.(name) = data; s.sig = sig; save(file,'-struct','s');
end

function data = load_compatible(file,sig,name)
    data = [];
    if ~isfile(file), return; end
    try
        L = load(file);
    catch ME
        warning('Could not read %s (%s); starting without it.',file,ME.message); return;
    end
    if isfield(L,'sig') && isequaln(L.sig,sig) && isfield(L,name)
        data = L.(name);
    else
        archive_file(file,'settings/version/ranges differ from the current run');
    end
end

function archive_file(file,reason)
    [d,n,e] = fileparts(file);
    newf = fullfile(d,[n '_archived_' datestr(now,'yyyymmdd_HHMMSS') e]);
    try
        movefile(file,newf);
        warning('Archived %s -> %s (%s). Not reused.',file,newf,reason);
    catch ME
        warning('Could not archive %s (%s).',file,ME.message);
    end
end

function out = ternary(c,a,b)
    if c, out = a; else, out = b; end
end

function A = lin_jac(f,xop)
    n = numel(xop); A = zeros(n);
    for ii = 1:n
        h = 1e-6*max(abs(xop(ii)),1);
        xp = xop; xm = xop; xp(ii) = xp(ii)+h; xm(ii) = xm(ii)-h;
        A(:,ii) = (f(xp,0)-f(xm,0))/(2*h);
    end
end

function dx = glioma_model(x,u,mp)
    x1=x(1); x2=x(2); x3=x(3); x4=x(4);
    dx1 = mp.Om1*x1*(1-x1) - mp.b1*x1*x2 - mp.p1*x1*x4/(mp.a1+x1);
    dx2 = mp.Om2*x2*(1-x2) - mp.b2*x1*x2 - mp.p2*x2*x4/(mp.a2+x2);
    H   = double(-dx1>0) + 0.5*double(dx1==0);
    dx3 = mp.alpha*dx1*H*x3 - mp.p3*x3*x4/(mp.a3+x3);
    dx4 = u - mp.zeta*x4;
    dx  = [dx1;dx2;dx3;dx4];
end

function xn = rk4_step(f,x,u,dt)
    k1 = f(x,u); k2 = f(x+dt/2*k1,u); k3 = f(x+dt/2*k2,u); k4 = f(x+dt*k3,u);
    xn = x + dt/6*(k1+2*k2+2*k3+k4);
end

function rec = eval_candidate(p,ctx,tag)
    % initial design + full closed loop with the final redesign rule
    tl = tic; dc = empty_dc(); sim = []; T = empty_terms(); ok = false; nResyn = 0;
    try
        [D0,dc] = get_design(ctx,p,ctx.x_pre,dc);
        mu = D0.mu; msg = D0.msg;
        if D0.ok
            [sim,dc] = simulate_closed_loop(ctx,p,D0,ctx.cfg.max_resyn,dc);
            [T,ok] = obj_terms(ctx,sim);
            mu = max(sim.mu_list); nResyn = sim.nResyn;
            if sim.aborted, msg = ['aborted: ' sim.abortReason]; elseif ok, msg = 'ok'; end
        end
    catch ME
        D0 = new_design(ctx.x_pre); mu = Inf; msg = ['exception: ' ME.message];
    end
    nM = 0; for jj = 1:numel(dc), nM = nM + dc(jj).D.n_musyn; end
    rec = struct('p',p,'pv',pvec(p),'tag',tag,'T',T,'ok',logical(ok),'mu',mu,'nResyn',nResyn, ...
                 'nMusyn',nM,'wall',toc(tl),'msg',msg,'D0',D0,'sim',sim);
end

function T = empty_terms()
    T = struct('jtrack',NaN,'ise',NaN,'Jprot',NaN,'maxrun',NaN,'n_gt5',NaN,'Jmu',NaN,'Jfail',NaN, ...
               't_sup',NaN,'x2T',NaN,'cumU',NaN);
end

function [T,ok] = obj_terms(ctx,sim)
    % raw terms of the objective for one full simulation
    T = empty_terms(); ok = false;
    if sim.aborted || any(~isfinite(sim.X(:))) || any(~isfinite(sim.U)), return; end
    S = ctx.S; od = ctx.objdef; t = ctx.t; Tend = t(end); x2 = sim.X(2,:)'; U = sim.U(:);
    el = log10(max(x2,od.track_floor)) - log10(max(ctx.ref,od.track_floor));
    T.jtrack = trapz(t,el.^2)/Tend;
    T.ise    = trapz(t,((x2-ctx.ref)/ctx.xs).^2)/Tend;
    rl       = run_lengths(U >= S.u_max-1e-9);
    T.Jprot  = sum(max(rl-od.protocol_start_days,0).^2)/(Tend/28);
    T.maxrun = max([rl; 0]);
    T.n_gt5  = sum(rl > 5);
    T.Jmu    = max(0,max(sim.mu_list)-1);
    T.Jfail  = min((sim.nFail+sim.nBumpFallback)/max(sim.nResyn,1),1);
    i1 = find(x2 < 1e-4,1,'first'); if ~isempty(i1), T.t_sup = t(i1); end
    T.x2T  = x2(end);
    T.cumU = trapz(t,U);
    ok = isfinite(T.jtrack) && isfinite(T.Jprot) && isfinite(T.Jmu);
end

function J = scalar_J(T,ok,Jref,od)
    if ~ok || ~isfinite(T.jtrack), J = od.J_fail; return; end
    J = T.jtrack/Jref + od.w_prot*T.Jprot + od.w_mu*T.Jmu + od.w_fail*T.Jfail;
    if ~isfinite(J), J = od.J_fail; end
    J = min(J,od.J_fail);
end

function print_cand_line(rec,Jref,J)
    q = rec.p;
    fprintf('%-5s M=%-6.3g A=%-8.3g wB=%-7.3g k=%-7.3g g=%-6.3g wz=%-7.3g wp=%-6.3g | Jtr/B %7.3g prot %6.3g t1e-4 %5g x2(T) %9.2e maxHi %3g n>5 %3g mu %5.3g | J %8.4g | %3d musyn %5.1f min | %s\n', ...
        rec.tag,q.M,q.A,q.wB,q.k,q.Wp_gain,q.wz,q.wp,rec.T.jtrack/Jref,rec.T.Jprot,rec.T.t_sup,rec.T.x2T, ...
        rec.T.maxrun,rec.T.n_gt5,rec.mu,J,rec.nMusyn,rec.wall/60,rec.msg);
end

function g = bo_objective(q,ctx,sig,Jref)
    global MU_NEW
    p = table2struct(q); pv = pvec(p);
    H = hist_get();
    if ~isempty(H)
        idx = find(max(abs(vertcat(H.pv)-pv)./abs(pv),[],2) < 1e-6,1);
        if ~isempty(idx), g = log10(1+scalar_J(H(idx).T,H(idx).ok,Jref,ctx.objdef)); return; end
    end
    MU_NEW = MU_NEW + 1;
    rec = eval_candidate(p,ctx,'bo');
    hist_append(ctx.cfg.hist_file,sig,rec);
    J = scalar_J(rec.T,rec.ok,Jref,ctx.objdef);
    print_cand_line(rec,Jref,J);
    g = log10(1+J);
end

function stop = search_stop(state,tS,budget,nbo)
    global MU_NEW
    stop = false;
    if strcmp(state,'iteration')
        H = hist_get();
        stop = toc(tS) >= budget || sum(strcmp({H.tag},'bo')) >= nbo || MU_NEW >= nbo+5;
    end
end

function P = random_points(rs,spec,n)
    % log-uniform points inside the ranges that satisfy wp >= 4*wz and Wp_gain <= 1.2*M
    P = zeros(0,numel(spec.lo)); tries = 0;
    iwp = strcmp(spec.names,'wp'); iwz = strcmp(spec.names,'wz');
    ig = strcmp(spec.names,'Wp_gain'); iM = strcmp(spec.names,'M');
    while size(P,1) < n && tries < 1e5
        tries = tries+1;
        v = spec.lo.*(spec.hi./spec.lo).^rand(rs,1,numel(spec.lo));
        if v(iwp) >= 4*v(iwz) && v(ig) <= 1.2*v(iM), P(end+1,:) = v; end %#ok<AGROW>
    end
end

function H = hist_get()
    global MU_H
    H = MU_H;
end

function H = hist_append(file,sig,rec)
    % history kept in memory; the file is only written (never re-read during a run)
    global MU_H
    if isempty(MU_H), MU_H = rec; else, MU_H(end+1) = rec; end
    H = MU_H;
    for k = 1:10
        try, save_struct(file,'H',H,sig); return; catch, pause(0.5); end
    end
    warning('Could not write %s; the history is kept in memory only.',file);
end

function v = pvec(p)
    v = [p.M p.A p.wB p.k p.Wp_gain p.wz p.wp];
end

function tf_ = pv_equal(a,b)
    tf_ = max(abs(a(:)-b(:))./abs(b(:))) < 1e-9;
end

function fig_search(H,Jall,cfg)
    fg = figure('Color','w','Name','mu_opt: search history');
    n = numel(H); Jp = Jall(:)'; Jp(Jp >= 1e4) = NaN;
    semilogy(1:n,Jp,'o','Color',[0.5 0.5 0.5]); hold on; grid on;
    Jb = Jall(:)'; bestSoFar = Jb; for i = 2:n, bestSoFar(i) = min(bestSoFar(i-1),Jb(i)); end
    semilogy(1:n,bestSoFar,'k-','LineWidth',1.2);
    xlabel('Evaluation'); ylabel('J (log scale)'); legend('candidate','best so far','Location','best');
    title('Weight search: objective of every candidate');
    save_fig(fg,[cfg.fig_prefix 'search'],cfg);
end
