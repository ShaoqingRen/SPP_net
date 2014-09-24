function [res_test, spp_model] = spp_exp_train_and_test_voc(opts)
% Runs an experiment that trains a spp model and tests it.

% -------------------- CONFIG --------------------
if ~exist('opts', 'var')
    opts.net_file        = fullfile(pwd, 'data\cnn_model\Zeiler_conv5\Zeiler_conv5');
    opts.net_def_file    = fullfile(pwd, 'data\cnn_model\Zeiler_conv5\Zeiler_spm_scale224_test_conv5.prototxt');
    opts.spp_params_def  = fullfile(pwd, 'data\cnn_model\Zeiler_conv5\spp_config');

    opts.layer                  = 7;
    opts.cache_name             = 'Zeiler_conv5_ft(5s_flip)_fc7';

    % detail setting
    opts.with_selective_search  = true;
    opts.with_edge_box          = false;
    opts.with_hard_samples      = false;

    % change to point to your devkit install
    opts.devkit                 = './datasets/VOCdevkit2007';

    opts.imdb_train             = {  imdb_from_voc(opts.devkit, 'trainval', '2007') };
    opts.roidb_train            = cellfun(@(x) x.roidb_func(x, opts.with_hard_samples, opts.with_selective_search, opts.with_edge_box), opts.imdb_train, 'UniformOutput', false);
    opts.feat_cache_train       = {'Zeiler_conv5'};
    opts.imdb_for_negative_mining = [1];
    opts.neg_ovr_threshs        = {[-1, 0.3]};

    opts.imdb_test              = imdb_from_voc(opts.devkit, 'test', '2007');
    opts.roidb_test             = opts.imdb_test.roidb_func(opts.imdb_test, opts.with_hard_samples, opts.with_selective_search, opts.with_edge_box);
    opts.feat_cache_test        = 'Zeiler_conv5';
    
    opts.gpu_id = 1;
end
% ------------------------------------------------

opts.imdb_train = opts.imdb_train(:)';
opts.roidb_train = opts.roidb_train(:)';
opts.feat_cache_train = opts.feat_cache_train(:)';
% Record a log of the training and test procedure
imdbs_name = cell2mat(cellfun(@(x) x.name, opts.imdb_train, 'UniformOutput', false));
conf = spp_config('sub_dir', fullfile(opts.cache_name, imdbs_name));
timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
diary_file = [conf.cache_dir 'spp_train_'  timestamp '.txt'];
diary(diary_file);
fprintf('Logging output in %s\n', diary_file);

if conf.use_gpu
   g = gpuDevice(opts.gpu_id); 
end

profile clear
profile on
spp_model = ...
    spp_train(opts.imdb_train, opts.roidb_train,...
      'layer',        opts.layer, ...
      'imdb_for_negative_mining', opts.imdb_for_negative_mining,...
      'neg_ovr_threshs', opts.neg_ovr_threshs, ...
      'net_file',     opts.net_file, ...
      'net_def_file', opts.net_def_file, ...
      'spp_params_def', opts.spp_params_def, ...
      'cache_name',   opts.cache_name, ...
      'feat_cache',   opts.feat_cache_train);
  
% load(fullfile(conf.cache_dir, 'spp_model'));
  
res_test = spp_test(spp_model, opts.imdb_test, opts.roidb_test, opts.feat_cache_test, '', true);

if conf.use_gpu
   reset(g);
end

profile viewer
diary off;
