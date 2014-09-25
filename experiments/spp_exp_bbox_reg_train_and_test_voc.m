function res = spp_exp_bbox_reg_train_and_test_voc(opts)
% Runs an experiment that trains a bounding box regressor and
% tests it.

if ~exist('opts', 'var')
    % change to point to your VOCdevkit install
    opts.devkit = './datasets/VOCdevkit2007';

    opts.flip                   = false;
    opts.with_selective_search  = true;
    opts.with_edge_box          = false;
    opts.with_hard_samples      = false;
    
    
    opts.imdb_train = { imdb_from_voc(opts.devkit, 'trainval', '2007', opts.flip) };
    opts.roidb_train = cellfun(@(x) x.roidb_func(x, opts.with_hard_samples, opts.with_selective_search, opts.with_edge_box), opts.imdb_train, 'UniformOutput', false);
    opts.feat_cache_train       = {'Zeiler_conv5'};

    opts.imdb_test = imdb_from_voc(opts.devkit, 'test', '2007');
    opts.roidb_test = opts.imdb_test.roidb_func(opts.imdb_test, opts.with_hard_samples, opts.with_selective_search, opts.with_edge_box);
    opts.feat_cache_test        = 'Zeiler_conv5';
    
    opts.cache_name             = 'Zeiler_conv5_ft(5s_flip)_fc7';
end

% load the spp_model trained by spp_exp_train_and_test()
imdbs_name = cell2mat(cellfun(@(x) x.name, opts.imdb_train, 'UniformOutput', false));
conf = spp_config('sub_dir', fullfile(opts.cache_name, imdbs_name));
ld = load([conf.cache_dir 'spp_model']);

timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
diary_file = [conf.cache_dir 'spp_bbox_'  timestamp '.txt'];
diary(diary_file);
fprintf('Logging output in %s\n', diary_file);

% train the bbox regression model
bbox_reg = spp_train_bbox_regressor(opts.imdb_train, opts.roidb_train, ld.spp_model, ...
    'min_overlap', 0.5, ...
    'layer', 5, ...
    'lambda', 4000, ...
    'robust', 0, ...
    'binarize', false);

% load([conf.cache_dir 'bbox_regressor_final']);

% test the bbox regression model
res = spp_test_bbox_regressor(opts.imdb_test, opts.roidb_test, ld.spp_model, bbox_reg, opts.feat_cache_test, 'bbox_reg', true);

diary off;
