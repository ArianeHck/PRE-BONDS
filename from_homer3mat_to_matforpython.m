% =========================================================
% Ce code extrait hbo/hbr/time depuis les .mat exportés depuis Homer3
% et les sauvegarde dans un sous-dossier 'exported'
% =========================================================

addpath(genpath('C:\Users\LaPsyDe\Documents\Homer3\Homer3-master'))

% vérification du type des colonnes, regarder directement les noms de colonnes dans l'objet Homer3 avant extraction
%fpath = 'C:\Users\LaPsyDe\Documents\code\données Sara\MATLAB data\play_NSFD2_child.mat';
%data = load(fpath);

%dc = data.output.dc;
%disp(fieldnames(dc))

%ml = dc.measurementList;
%for i = 1:min(9, length(ml))
 %   fprintf('Colonne %d → dataTypeLabel=%s\n', i, ml(i).dataTypeLabel)
%end


BASE_IN  = "C:\Users\LaPsyDe\Documents\bizzego data\dataverse_files\play\mothers\mother_copy\derivatives\homer";
BASE_OUT = "C:\Users\LaPsyDe\Documents\bizzego data\final_analysis\play\exported"; %selon la condition
if ~exist(BASE_OUT, 'dir'), mkdir(BASE_OUT); end

mat_files = dir(fullfile(BASE_IN, '*.mat'));
fprintf('Fichiers trouvés : %d\n', length(mat_files));

n_ok = 0; n_fail = 0;

for f = 1:length(mat_files)
    fname = mat_files(f).name;
    fpath = fullfile(BASE_IN, fname);
    out_path = fullfile(BASE_OUT, fname);

    try
        data = load(fpath);

        % Extraire depuis l'objet ProcResultClass
        ts   = data.output.dc.dataTimeSeries;   % (T × 60)
        t    = data.output.dc.time;              % (T × 1)

        % Colonnes : HbO = 1,4,7,... / HbR = 2,5,8,...
        hbo_homer  = ts(:, 1:3:end);   % (T × 20)
        hbr_homer  = ts(:, 2:3:end);   % (T × 20)
        time_homer = t(:);

        save(out_path, 'hbo_homer', 'hbr_homer', 'time_homer');
        fprintf('  ✓ %s\n', fname);
        n_ok = n_ok + 1;

    catch e
        fprintf('  ✗ %s : %s\n', fname, e.message);
        n_fail = n_fail + 1;
    end
end

fprintf('\n══ %d exportés, %d échoués ══\n', n_ok, n_fail);