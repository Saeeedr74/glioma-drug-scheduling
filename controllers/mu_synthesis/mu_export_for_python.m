%% mu_export_for_python.m
% Reads mu_opt_results.mat (made by mu_optimize_final.m) and writes plain CSV files
% for the three-controller comparison in Python. Nothing is simulated here, so it
% runs in seconds and needs base MATLAB only (no toolboxes).
%
% Files written (current folder):
%   mu_py_timeseries.csv : one row per day and scenario: x1..x4, u, reference
%   mu_py_metrics.csv    : one row per scenario: first/permanent day of x2<1e-4 and
%                          x1>=0.99, final states, RMSE, dose, high-dose phases
%   mu_py_mc.csv         : Monte Carlo samples (parameter changes and outcomes)
%   mu_py_design.csv     : weights and design diagnostics of the mu design
%
% Scenarios (column "mode"):
%   nominal         : nominal plant, controller in the loop
%   E1_replay       : nominal dose schedule replayed on the perturbed plant (Table 6 protocol)
%   E2_closed_loop  : controller in the loop on the perturbed plant
% Time: time_day = 0 is the start of treatment (day 600 of the simulation).
%
% Run in the folder that holds mu_opt_results.mat.

clear; clc;
matfile = 'mu_opt_results.mat';
assert(isfile(matfile),'%s not found in %s. Run this script in the folder of mu_optimize_final.m.',matfile,pwd);
L = load(matfile);
R = L.results;  t = L.t(:);  ref = L.ref(:);  simD = L.simD;  XG = L.XG;  CLG = L.CLG;
S = R.S;  xs = L.x_pre(2);  nG = numel(CLG);
names = [{'nominal'}; R.unc.groups(:,1)];
ctrl = 'mu';
fprintf('Design %s, version %s, %d days, x2_pre = %.6f\n',R.design,R.version,numel(t),xs);

TS = {};  MT = {};

% ---- nominal
TS{end+1} = ts_table(ctrl,'nominal',names{1},0,t,simD.X,simD.U,ref);
MT{end+1} = metrics_row(ctrl,'nominal',names{1},0,t,simD.X,simD.U,ref,xs,S);

% ---- E1: nominal schedule replayed on the perturbed plants
for g = 1:nG
    X = XG{g+1};
    TS{end+1} = ts_table(ctrl,'E1_replay',names{g+1},g,t,X,simD.U,ref); %#ok<SAGROW>
    MT{end+1} = metrics_row(ctrl,'E1_replay',names{g+1},g,t,X,simD.U,ref,xs,S); %#ok<SAGROW>
end

% ---- E2: controller in the loop on the perturbed plants
for g = 1:nG
    if isempty(CLG{g}) || CLG{g}.aborted
        warning('E2 group %d (%s) is missing or aborted; skipped.',g,names{g+1}); continue;
    end
    TS{end+1} = ts_table(ctrl,'E2_closed_loop',names{g+1},g,t,CLG{g}.X,CLG{g}.U,ref); %#ok<SAGROW>
    MT{end+1} = metrics_row(ctrl,'E2_closed_loop',names{g+1},g,t,CLG{g}.X,CLG{g}.U,ref,xs,S); %#ok<SAGROW>
end

writetable(vertcat(TS{:}),'mu_py_timeseries.csv');
M = vertcat(MT{:});
writetable(M,'mu_py_metrics.csv');

% ---- Monte Carlo (supplementary)
if isfield(L,'Z') && ~isempty(L.Z) && isfield(L,'MCrep') && ~isempty(L.MCrep.x2T)
    Nmc = size(L.Z,1);
    frac = L.Z .* R.unc.mc_frac(:)';                 % fractional change of each parameter
    Tm = array2table(frac,'VariableNames',strcat('frac_',R.unc.mc_names(:)'));
    Tm.sample = (1:Nmc)';
    Tm.x2T_replay = L.MCrep.x2T(:);
    Tm.reach_replay = double(L.MCrep.reach(:));
    cl = L.MCcl;
    Tm.x2T_cl = nan(Nmc,1);  Tm.reach_cl = nan(Nmc,1);  Tm.maxHi_cl = nan(Nmc,1);
    Tm.n_gt5_cl = nan(Nmc,1);  Tm.t_sup_cl = nan(Nmc,1);
    if ~isempty(cl.x2T)
        n = numel(cl.x2T);
        Tm.x2T_cl(1:n) = cl.x2T(:);       Tm.reach_cl(1:n) = double(cl.reach(:));
        Tm.maxHi_cl(1:n) = cl.maxHi(:);   Tm.n_gt5_cl(1:n) = cl.n_gt5(:);
        Tm.t_sup_cl(1:n) = cl.t_sup(:);
    end
    Tm = movevars(Tm,'sample','Before',1);
    writetable(Tm,'mu_py_mc.csv');
else
    warning('No Monte Carlo data in %s; mu_py_mc.csv not written.',matfile);
end

% ---- design info
w = R.weights;
key = {'M';'A';'wB';'k';'Wp_gain';'wz';'wp';'mu_peak_initial';'robstab_lower';'robstab_upper'; ...
       'Wi_low_freq_percent';'Wi_high_freq_percent';'a_reference';'x2_pre';'redesign_x2_stop'; ...
       'redesign_rel_threshold';'redesign_min_gap_days';'dose_threshold';'u_min';'u_max'};
val = [w.M; w.A; w.wB; w.k; w.Wp_gain; w.wz; w.wp; R.mu_initial; R.verif.robstab_LB; R.verif.robstab_UB; ...
       100*w.k; 100*w.k*w.wp/w.wz; S.a; xs; S.x2_stop; S.rel_thresh; R.cfg.resyn_gap; S.u_switch; S.u_min; S.u_max];
writetable(table(key,val,'VariableNames',{'key','value'}),'mu_py_design.csv');

% ---- console check (compare with the run log: 1167, 1046, 1.252e-06, 142150)
fprintf('\nCHECK, nominal row: first day x2<1e-4 = %g | permanent day x2<1e-4 = %g | permanent day x1>=0.99 = %g\n', ...
    M.t_x2_first(1),M.t_x2_perm(1),M.t_x1_perm(1));
fprintf('                    x2(T) = %.4g | cumulative dose = %.0f | high-dose phases = %d (longest %g d)\n', ...
    M.x2_T(1),M.cum_dose(1),M.n_hi(1),M.hi_max(1));
disp(M(:,{'mode','scenario','t_x2_first','t_x2_perm','t_x1_first','t_x1_perm','x2_T','x3_T','hi_max','n_gt5'}));
fprintf('Written: mu_py_timeseries.csv (%d rows), mu_py_metrics.csv, mu_py_mc.csv, mu_py_design.csv\n',sum(cellfun(@height,TS)));

%% ------------------------------ local functions ------------------------------
function T = ts_table(ctrl,mode,scen,gid,t,X,U,ref)
    N = numel(t);
    assert(size(X,2) == N && numel(U) == N,'Length mismatch in scenario %s.',scen);
    T = table(repmat({ctrl},N,1),repmat({mode},N,1),repmat({scen},N,1),repmat(gid,N,1), ...
        t(:),X(1,:)',X(2,:)',X(3,:)',X(4,:)',U(:),ref(:), ...
        'VariableNames',{'controller','mode','scenario','group_id','time_day','x1','x2','x3','x4','u','x2_reference'});
end

function row = metrics_row(ctrl,mode,scen,gid,t,X,U,ref,xs,S)
    t = t(:);  U = U(:);
    x1 = X(1,:)';  x2 = X(2,:)';  x3 = X(3,:)';
    m = struct();
    m.t_x2_first = first_day(t,x2 < 1e-4);          % first day with x2 < 1e-4
    m.t_x2_perm  = perm_day(t,x2 < 1e-4);           % from this day on, x2 stays below 1e-4
    m.t_x1_first = first_day(t,x1 >= 0.99);         % first day with x1 >= 0.99
    m.t_x1_perm  = perm_day(t,x1 >= 0.99);          % from this day on, x1 stays >= 0.99 (glial recovery)
    m.x1_T = x1(end);  m.x2_T = x2(end);  m.x3_T = x3(end);  m.x3_min = min(x3);
    m.rmse_norm = sqrt(mean((x2-ref(:)).^2))/xs;    % tracking RMSE / x2_pre
    m.u_mean = mean(U);  m.cum_dose = trapz(t,U);
    hi = U >= S.u_switch;
    rl = run_lengths(hi);
    m.n_hi = numel(rl);
    m.hi_mean = mean_or_nan(rl);
    m.hi_max = max([rl; 0]);
    m.n_gt5 = sum(rl > 5);
    m.frac_hi = mean(hi);
    row = [table({ctrl},{mode},{scen},gid,'VariableNames',{'controller','mode','scenario','group_id'}), struct2table(m)];
end

function d = first_day(t,ok)
    i = find(ok,1,'first');
    if isempty(i), d = NaN; else, d = t(i); end
end

function d = perm_day(t,ok)
    % first day from which the condition holds until the end; NaN if it does not hold on the last day
    if ~ok(end), d = NaN; return; end
    i = find(~ok,1,'last');
    if isempty(i), d = t(1); else, d = t(i+1); end
end

function rl = run_lengths(hi)
    d = diff([0; double(hi(:)); 0]);
    st = find(d == 1);  en = find(d == -1)-1;
    rl = en-st+1;
end

function v = mean_or_nan(x)
    if isempty(x), v = NaN; else, v = mean(x); end
end
