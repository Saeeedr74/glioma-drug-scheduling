%% mu_export_extras.m
% Reads mu_opt_results.mat (made by mu_optimize_final.m) and writes the material that
% mu_export_for_python.m does not export: full states of the continuous run, the clipped
% controller command, the redesign days and mu of every design, the redesign-rule
% comparison, the weight-search table, the closed-loop Monte Carlo trajectories and the
% scalar diagnostics. Nothing is simulated or designed here (base MATLAB only).
%
% Written for the 2026 revision of the manuscript. Run in the folder that
% holds mu_opt_results.mat; the CSV files are written to the current folder.

clear; clc;
matfile = 'mu_opt_results.mat';
assert(isfile(matfile),'%s not found in %s.',matfile,pwd);
L = load(matfile);
R = L.results;  t = L.t(:);  N = numel(t);  S = R.S;
fprintf('Design %s, version %s\n',R.design,R.version);

% ---- 1) time series of the nominal runs and of the redesign-rule runs
runs = {'discrete_nominal',L.simD};
if ~isempty(L.simC), runs(end+1,:) = {'continuous_nominal',L.simC}; end
ruleKey = cell(numel(L.RC),1);
for r = 1:numel(L.RC)
    ruleKey{r} = sprintf('rule%d',r);
    runs(end+1,:) = {ruleKey{r},L.RC(r).sim}; %#ok<SAGROW>
end
nG = numel(L.CLG);
for g = 1:nG
    if ~isempty(L.CLG{g}), runs(end+1,:) = {sprintf('E2_G%d',g),L.CLG{g}}; end %#ok<SAGROW>
end
TS = {};  RD = {};
for i = 1:size(runs,1)
    s = runs{i,2};  nm = runs{i,1};
    TS{end+1} = table(repmat({nm},N,1),t,s.X(1,:)',s.X(2,:)',s.X(3,:)',s.X(4,:)',s.U(:),s.Ucmd(:), ...
        'VariableNames',{'run','time_day','x1','x2','x3','x4','u','u_command_clipped'}); %#ok<SAGROW>
    days = [0; s.resynDays(:)];  mu = s.mu_list(:);
    assert(numel(days) == numel(mu),'redesign days and mu list differ in length for %s',nm);
    RD{end+1} = table(repmat({nm},numel(mu),1),(0:numel(mu)-1)',days,mu, ...
        'VariableNames',{'run','design_index','day','mu_peak'}); %#ok<SAGROW>
end
writetable(vertcat(TS{:}),'mu_x_timeseries.csv');
writetable(vertcat(RD{:}),'mu_x_redesigns.csv');

% ---- 2) redesign-rule comparison (metrics as computed by mu_optimize_final.m)
RM = {};
for r = 1:numel(L.RC)
    m = L.RC(r).met;  s = L.RC(r).sim;
    lastR = NaN; x2l = NaN;
    if ~isempty(s.resynDays), lastR = s.resynDays(end); x2l = s.X(2,find(t == lastR,1)); end
    iLastHi = find(s.U >= S.u_switch,1,'last'); lastHi = NaN; if ~isempty(iLastHi), lastHi = t(iLastHi); end
    T1 = table({ruleKey{r}},{L.RC(r).label},L.RC(r).x2_stop,L.RC(r).max_resyn,lastR,x2l,lastHi, ...
        'VariableNames',{'run','label','x2_stop','max_resyn','last_redesign_day','x2_at_last_redesign','last_high_dose_day'});
    RM{end+1} = [T1, struct2table(m)]; %#ok<SAGROW>
end
if ~isempty(RM), writetable(vertcat(RM{:}),'mu_x_rule_metrics.csv'); end

% ---- 3) weight-search table
if isfield(R,'search') && isfield(R.search,'table')
    Ts = R.search.table;  Ts.candidate = (1:height(Ts))';  Ts = movevars(Ts,'candidate','Before',1);
    writetable(Ts,'mu_x_search.csv');
    fprintf('search: %d candidates, %d valid, %d musyn calls, winner %s (candidate %d), J = %.6g, J(B) = %.6g\n', ...
        R.search.n_candidates,R.search.n_valid,R.search.n_musyn,R.search.winner_tag,R.search.winner_index,R.search.J_winner,R.search.J_B);
end

% ---- 4) E2 metrics as computed by mu_optimize_final.m
ME = {};
for g = 1:nG
    if isempty(R.metrics_E2{g}), continue; end
    ME{end+1} = [table(g,R.unc.groups(g,1),'VariableNames',{'group_id','scenario'}), struct2table(R.metrics_E2{g})]; %#ok<SAGROW>
end
if ~isempty(ME), writetable(vertcat(ME{:}),'mu_x_e2_metrics.csv'); end

% ---- 5) closed-loop Monte Carlo trajectories (x2) and ISE ratios
if isfield(L,'MCcl') && ~isempty(L.MCcl.X2)
    n = size(L.MCcl.X2,1);
    Tc = array2table([t, L.MCcl.X2'],'VariableNames',[{'time_day'}, arrayfun(@(k) sprintf('s%02d',k),1:n,'UniformOutput',false)]);
    writetable(Tc,'mu_x_mc_cl_x2.csv');
    Tm = table((1:n)',L.MCcl.x2T(:),double(L.MCcl.reach(:)),L.MCcl.ise_ratio(:),L.MCcl.maxHi(:),L.MCcl.n_gt5(:),L.MCcl.t_sup(:),double(L.MCcl.aborted(:)), ...
        'VariableNames',{'sample','x2T_cl','reach_cl','ise_ratio_cl','maxHi_cl','n_gt5_cl','t_sup_cl','aborted_cl'});
    writetable(Tm,'mu_x_mc_cl.csv');
end
Tr = table((1:numel(L.MCrep.x2T))',L.MCrep.x2T(:),double(L.MCrep.reach(:)),L.MCrep.ise_ratio(:), ...
    'VariableNames',{'sample','x2T_replay','reach_replay','ise_ratio_replay'});
writetable(Tr,'mu_x_mc_replay.csv');

% ---- 6) scalars: verification, metrics of the nominal runs, settings
key = {};  val = [];
[key,val] = add_struct(key,val,'verif',R.verif);
[key,val] = add_struct(key,val,'metrics_discrete',R.metrics_discrete);
if ~isempty(R.metrics_continuous), [key,val] = add_struct(key,val,'metrics_continuous',R.metrics_continuous); end
[key,val] = add_struct(key,val,'cfg',R.cfg);
[key,val] = add_struct(key,val,'S',R.S);
[key,val] = add_struct(key,val,'weights',R.weights);
[key,val] = add_struct(key,val,'mp',R.mp);
extra = {'mu_initial',R.mu_initial; 'nominal_stable',double(R.nominal_stable); 'sampled_stable',double(R.sampled_stable); ...
         'ISE_discrete',R.ISE_discrete; 'ISE_open',R.ISE_open; 'sensitivity_factor_E1',R.sensitivity_factor; ...
         'elapsed_min',R.elapsed_min; 'n_days',N; ...
         'simD_nResyn',L.simD.nResyn; 'simD_nFail',L.simD.nFail; 'simD_nBumpFallback',L.simD.nBumpFallback; ...
         'simD_nMusyn',L.simD.nMusyn; 'x_pre_1',L.x_pre(1); 'x_pre_2',L.x_pre(2); 'x_pre_3',L.x_pre(3); 'x_pre_4',L.x_pre(4)};
for g = 0:nG, extra(end+1,:) = {sprintf('x2T_E1_G%d',g),R.x2T_E1(g+1)}; end %#ok<SAGROW>
if isfield(R,'search')
    extra = [extra; {'search_n_candidates',R.search.n_candidates; 'search_n_valid',R.search.n_valid; ...
        'search_n_musyn',R.search.n_musyn; 'search_J_winner',R.search.J_winner; 'search_J_B',R.search.J_B; ...
        'search_winner_index',R.search.winner_index}];
end
key = [key; extra(:,1)];  val = [val; cell2mat(extra(:,2))];
fid = fopen('mu_x_scalars.csv','w');  fprintf(fid,'key,value\n');
for i = 1:numel(key), fprintf(fid,'%s,%.17g\n',key{i},val(i)); end
fclose(fid);
fid = fopen('mu_x_strings.csv','w');  fprintf(fid,'key,value\n');
fprintf(fid,'version,%s\n',R.version);  fprintf(fid,'design,%s\n',R.design);
if isfield(R,'search'), fprintf(fid,'search_winner_tag,%s\n',R.search.winner_tag); fprintf(fid,'search_objective,"%s"\n',R.search.objective); end
for r = 1:numel(L.RC), fprintf(fid,'%s,"%s"\n',ruleKey{r},L.RC(r).label); end
fclose(fid);

v = ver;
fid = fopen('matlab_version.txt','w');
fprintf(fid,'MATLAB %s\n',version);
for i = 1:numel(v), fprintf(fid,'%s | %s | %s\n',v(i).Name,v(i).Version,v(i).Release); end
fclose(fid);
fprintf('Written: mu_x_timeseries.csv, mu_x_redesigns.csv, mu_x_rule_metrics.csv, mu_x_search.csv, mu_x_e2_metrics.csv, mu_x_mc_cl_x2.csv, mu_x_mc_cl.csv, mu_x_mc_replay.csv, mu_x_scalars.csv, mu_x_strings.csv, matlab_version.txt\n');

%% ------------------------------ local functions ------------------------------
function [key,val] = add_struct(key,val,prefix,s)
    % numeric / logical scalar fields of a struct as key-value rows
    f = fieldnames(s);
    for i = 1:numel(f)
        x = s.(f{i});
        if (isnumeric(x) || islogical(x)) && isscalar(x)
            key{end+1,1} = [prefix '.' f{i}]; %#ok<AGROW>
            val(end+1,1) = double(x); %#ok<AGROW>
        end
    end
end
