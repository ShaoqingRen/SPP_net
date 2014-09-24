function finetuned_model_path = spp_finetune_voc(opts)
% [] = spp_finetune_voc(opts)
%   finetune fc layers 
% 
% this version read conv feature maps from disk each iteration for compatibility and small
% memory usage. load all conv feature maps into memory will accelerate finetuning
% greatly
%
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Shaoqing Ren
% 
% This file is part of the SPP code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% -------------------- CONFIG --------------------
if ~exist('opts', 'var')
    opts.net_file               = fullfile(pwd, 'data\cnn_model\Zeiler_conv5\Zeiler_conv5');
    opts.finetune_net_def_file  = fullfile(pwd, 'model-defs\pascal_finetune_fc_spm_solver.prototxt');
    opts.spp_params_def         = fullfile(pwd, 'data\cnn_model\Zeiler_conv5\spp_config');

    opts.flip = true;
    opts.finetune_cache_name = 'Zeiler_conv5_ft(5s_flip)_ss';

    opts.with_selective_search = true;
    opts.with_edge_box = false;

    opts.devkit = './datasets/VOCdevkit2007';
    opts.imdb_train   = {   imdb_from_voc(opts.devkit, 'trainval', '2007', opts.flip) };
    opts.feat_cache_train   = { 'Zeiler_conv5' };

    opts.imdb_test    = { imdb_from_voc(opts.devkit, 'test', '2007') };
    opts.feat_cache_test = { 'Zeiler_conv5' };
    
    opts.with_hard_samples = false;  % only for train
else
    if ~iscell(opts.imdb_test)
        opts.imdb_test = {opts.imdb_test};
        opts.feat_cache_test = {opts.feat_cache_test};
    end
end

opts.random_scale = 1;

data_param.img_num_per_iter = 128; % should be same with the prototxt file
data_param.random_scale = opts.random_scale;
data_param.iter_per_batch = 500; % for load data effectively 
data_param.fg_fraction = 0.25;
data_param.fg_threshold = 0.5;
data_param.bg_threshold = [0.1 0.5];

solver_param.test_iter = 500; % better times of data_param.iter_per_batch 
solver_param.test_interval = 2000; % better times of data_param.iter_per_batch 
if isfield(opts, 'max_iter')
    solver_param.max_iter = opts.max_iter;
else
    solver_param.max_iter = 1000000;
end

opts.imdb_train = opts.imdb_train(:)';
opts.feat_cache_train = opts.feat_cache_train(:)';
opts.imdb_test = opts.imdb_test(:)';
opts.feat_cache_test = opts.feat_cache_test(:)';
roidb_train = cellfun(@(x) x.roidb_func(x, opts.with_hard_samples, opts.with_selective_search, opts.with_edge_box), opts.imdb_train, 'UniformOutput', false);
roidb_test = cellfun(@(x) x.roidb_func(x, true, opts.with_selective_search, opts.with_edge_box), opts.imdb_test, 'UniformOutput', false);
% ------------------------------------------------

work_dir = fullfile(pwd, 'finetuning', opts.finetune_cache_name);
timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
log_file = fullfile(work_dir, 'output', [opts.finetune_cache_name, '_' timestamp, '.txt']);
caffe_log_file = fullfile(work_dir, opts.finetune_cache_name);

mkdir_if_missing(work_dir);
mkdir_if_missing(fileparts(log_file));
diary(log_file);

fprintf('data_param:\n');
disp(data_param);
fprintf('solver_param:\n');
disp(solver_param);
fprintf('opts_param:\n');
disp(opts);


%% get all data info
[fg_windows_train, bg_windows_train] = cellfun(@(x) SetupData(x, data_param), roidb_train, 'UniformOutput', false);
[fg_windows_test, bg_windows_test] = cellfun(@(x) SetupData(x, data_param), roidb_test, 'UniformOutput', false);
fg_windows_num_train = cellfun(@(y) cellfun(@(x) size(x, 1), y, 'UniformOutput', true), ...
    fg_windows_train, 'UniformOutput', false);
fg_windows_num_test = cellfun(@(y) cellfun(@(x) size(x, 1), y, 'UniformOutput', true), ...
    fg_windows_test, 'UniformOutput', false);
bg_windows_num_train = cellfun(@(y) cellfun(@(x) size(x, 1), y, 'UniformOutput', true), ...
    bg_windows_train, 'UniformOutput', false);
bg_windows_num_test = cellfun(@(y) cellfun(@(x) size(x, 1), y, 'UniformOutput', true), ...
    bg_windows_test, 'UniformOutput', false);

fg_windows_train = CombineImdbWindows(fg_windows_train);
bg_windows_train = CombineImdbWindows(bg_windows_train);
fg_windows_test = CombineImdbWindows(fg_windows_test);
bg_windows_test = CombineImdbWindows(bg_windows_test);

fg_windows_num_train = cell2mat(fg_windows_num_train');
bg_windows_num_train = cell2mat(bg_windows_num_train');
fg_windows_num_test = cell2mat(fg_windows_num_test');
bg_windows_num_test = cell2mat(bg_windows_num_test');

fprintf('total fg_windows_num_train : %d\n', sum(fg_windows_num_train));
fprintf('total bg_windows_num_train : %d\n', sum(bg_windows_num_train));
fprintf('total fg_windows_num_test : %d\n', sum(fg_windows_num_test));
fprintf('total bg_windows_num_test : %d\n', sum(bg_windows_num_test));
%% init caffe solver
caffe('set_random_seed', 6);
spp_model.spp_pooler = spp_load_pooling_params(opts.spp_params_def);
InitSolver(opts.finetune_net_def_file, opts.net_file, caffe_log_file);

% finetune
caffe('set_mode_gpu');
caffe('set_phase_train');

%% get test sample
% fix the random seed for repeatability
prev_rng = seed_rand();
data_param_test = data_param;
data_param_test.iter_per_batch = solver_param.test_iter;
data_param_test.random_scale = 0; % for test use the best scale
[data_test, label_test] = PrepareBatchData(spp_model.spp_pooler, opts.imdb_test, fg_windows_test, ...
                    fg_windows_num_test, bg_windows_test, bg_windows_num_test, data_param_test, data_param.img_num_per_iter, opts.feat_cache_test);
rng(prev_rng);
                
%% finetune
prev_rng = seed_rand();
data_train = {}; label_train = {};
train_error_sum = 0;
train_error_count = 0;
min_test_error = 1;
iter_ = 1;
while(iter_ < solver_param.max_iter)
    if isempty(data_train)
        [data_train, label_train] = ...
        PrepareBatchData(spp_model.spp_pooler, opts.imdb_train, fg_windows_train, ...
        fg_windows_num_train, bg_windows_train, bg_windows_num_train, data_param, data_param.img_num_per_iter, opts.feat_cache_train);
    end
    rst = caffe('train', {data_train{1}, label_train{1}});
    data_train(1) = []; label_train(1) = [];
    train_error_sum = train_error_sum + rst(1);
    train_error_count = train_error_count + 1;
    
    % test per test_interval iterations
    if ~mod(iter_, solver_param.test_interval) 
        test_error = nan(solver_param.test_iter, 1);
        for it = 1:solver_param.test_iter
            rst = caffe('test', {data_test{it}, label_test{it}});
            test_error(it) = rst(1);
        end
        ShowState(iter_, train_error_sum / train_error_count, mean(test_error));
        if (mean(test_error) < min_test_error)
            min_test_error = mean(test_error);
            finetuned_model_path = fullfile(opts.finetune_rst_dir, sprintf('FT_iter_%d', iter_));
            caffe('snapshot', finetuned_model_path);
            fprintf('best! save as %s\n', finetuned_model_path);
        end
        train_error_sum = 0;
        train_error_count = 0;
        
        diary; diary; % flush diary
    end
    
    iter_ = caffe('get_solver_iter');
end

rng(prev_rng);
caffe('release_solver');

end

function [data, label] = PrepareBatchData(spp_pooler, imdb, fg_windows, fg_windows_num, bg_windows, bg_windows_num, data_param, img_num_per_iter, feat_cache)
    % windows -- (image_id_in_total, image_imdb_idx, image_id_in_imdb,
    % label, ov, l, t, r, b)

    nTimesMoreData = 10;

    fg_num_each = int32(data_param.fg_fraction * img_num_per_iter);
    bg_num_each = img_num_per_iter - fg_num_each;
    fg_num_total = fg_num_each * data_param.iter_per_batch;
    bg_num_total = bg_num_each * data_param.iter_per_batch;
    total_img = size(fg_windows, 1);
    
    % random perm image and get top n image with total more than patch_num
    % windows
    img_idx = randperm(total_img);
    fg_windows_num = fg_windows_num(img_idx);
    fg_windows_cumsum = cumsum(fg_windows_num);
    bg_windows_num = bg_windows_num(img_idx);
    bg_windows_cumsum = cumsum(bg_windows_num);
    img_idx_end = max(find(fg_windows_cumsum > fg_num_total * nTimesMoreData, 1, 'first'), ...
        find(bg_windows_cumsum > bg_num_total * nTimesMoreData, 1, 'first'));
    if isempty( img_idx_end )
        img_idx_end = length(img_idx);
    end
    select_img_idx = img_idx(1:img_idx_end);
    
    fg_window = cell2mat(fg_windows(select_img_idx));
    bg_window = cell2mat(bg_windows(select_img_idx));
    
    % random perm all windows, and drop redundant data
    window_idx = randperm(size(fg_window, 1));
    fg_window = fg_window(window_idx(1:fg_num_total), :);
    window_idx = randperm(size(bg_window, 1));
    bg_window = bg_window(window_idx(1:bg_num_total), :);
    
    fg_data = cell(1, length(select_img_idx)); bg_data = cell(1, length(select_img_idx)); 
    fg_label = cell(length(select_img_idx), 1); bg_label = cell(length(select_img_idx), 1);
    random_scale = data_param.random_scale;
    fg_window_this_img = cell(length(select_img_idx), 1);
    bg_window_this_img = cell(length(select_img_idx), 1);
    for i = 1:length(select_img_idx)
        fg_window_this_img{i} = fg_window(fg_window(:, 1) == select_img_idx(i), :);
        bg_window_this_img{i} = bg_window(bg_window(:, 1) == select_img_idx(i), :);
    end
    assert(sum(cellfun(@(x) size(x, 1), fg_window_this_img, 'UniformOutput', true)) == fg_num_total);
    assert(sum(cellfun(@(x) size(x, 1), bg_window_this_img, 'UniformOutput', true)) == bg_num_total);
    random_seed = randi(10^6, length(select_img_idx));
    parfor i = 1:length(select_img_idx)
        rng(random_seed(i), 'twister')
        window_this_img = [fg_window_this_img{i}; bg_window_this_img{i}];
        if isempty(window_this_img)
            continue;
        end
        image_id_in_total = select_img_idx(i);
        assert(all(window_this_img(:, 1) == image_id_in_total));
        image_imdb_idx = window_this_img(1, 2);
        assert(all(window_this_img(:, 2) == image_imdb_idx));
        image_id_in_imdb = window_this_img(1, 3);
        assert(all(window_this_img(:, 3) == image_id_in_imdb));
        
        [fg_data{i}, bg_data{i}] = LoadCachedPoolXFeature(spp_pooler, imdb{image_imdb_idx}, feat_cache{image_imdb_idx}, ...
            image_id_in_imdb, fg_window_this_img{i}(:, 6:9), bg_window_this_img{i}(:, 6:9), random_scale);
        fg_label{i} = fg_window_this_img{i}(:, 4);
        bg_label{i} = bg_window_this_img{i}(:, 4);
    end
    valid_fg_data = cellfun(@(x) ~isempty(x), fg_label, 'UniformOutput', true);
    valid_bg_data = cellfun(@(x) ~isempty(x), bg_label, 'UniformOutput', true);
    fg_data = cell2mat(fg_data(valid_fg_data));
    bg_data = cell2mat(bg_data(valid_bg_data));
    fg_label = cell2mat(fg_label(valid_fg_data));
    bg_label = cell2mat(bg_label(valid_bg_data));
    
    % random perm all windows
    fg_rnd_idx = randperm(size(fg_label, 1));
    bg_rnd_idx = randperm(size(bg_label, 1));
    fg_data = fg_data(:, fg_rnd_idx);
    fg_label = fg_label(fg_rnd_idx, :);
    bg_data = bg_data(:, bg_rnd_idx);
    bg_label = bg_label(bg_rnd_idx, :);
    
    % split to iter_per_batch 
    fg_split_div = ones(1, data_param.iter_per_batch) * double(fg_num_each);
    bg_split_div = ones(1, data_param.iter_per_batch) * double(bg_num_each);
    fg_data = mat2cell(fg_data, size(fg_data, 1), fg_split_div);
    fg_label = mat2cell(fg_label', size(fg_label, 2), fg_split_div);
    bg_data = mat2cell(bg_data, size(bg_data, 1), bg_split_div);
    bg_label = mat2cell(bg_label', size(bg_label, 2), bg_split_div);
    
    % merge fg and bg_data
    data = cellfun(@(x, y) [x, y], fg_data, bg_data, 'UniformOutput', false);
    label = cellfun(@(x, y) [x, y], fg_label, bg_label, 'UniformOutput', false);
end

function [fg_windows, bg_windows] = SetupData(roidb, data_param)
    fg_windows = cell(length(roidb.rois), 1);
    bg_windows = cell(length(roidb.rois), 1);
    for i = 1:length(roidb.rois)
        roi = roidb.rois(i);
        [ov, label] = max(roi.overlap, [], 2);
        ov = full(ov);
        fg_mask = ov >= data_param.fg_threshold;
        fg_ov = ov(fg_mask);
        fg_label = label(fg_mask);
        fg_img_idx = fg_label * 0 + i;
        fg_window = roi.boxes(fg_mask, :);
        fg_windows{i} = [fg_img_idx, fg_label, fg_ov, fg_window];
        
        bg_mask = ~fg_mask & ov >= data_param.bg_threshold(1) & ov < data_param.bg_threshold(2);
        bg_ov = ov(bg_mask) * 0;
        bg_label = label(bg_mask) * 0;
        bg_img_idx = bg_label * 0 + i;
        bg_window = roi.boxes(bg_mask, :);
        bg_windows{i} = [bg_img_idx, bg_label, bg_ov, bg_window];
    end
end

function [fg_feat, bg_feat] = LoadCachedPoolXFeature(spp_pooler, imdb, feat_cache, img_id, fg_windows, bg_windows, random_scale)
    feat = spp_load_cached_poolX_features(spp_pooler, feat_cache, imdb.name, imdb.image_ids{img_id}, [fg_windows; bg_windows], random_scale);
    fg_feat = feat(:, 1:size(fg_windows, 1));
    bg_feat = feat(:, (size(fg_windows, 1)+1):end);
end

function ShowState(iter, train_error, test_error)
    fprintf('\n------------------------- Iteration %d -------------------------\n', iter);
    fprintf('Error - Training : %.4f - Testing : %.4f\n', train_error, test_error);
end

function windows = CombineImdbWindows(windows_in_cell)
    % windows -- (image_id_in_total, image_imdb_idx, image_id_in_imdb,
    % label, ov, l, t, r, b)

    image_imdb_idx = cellfun(@(x, y) ones(size(y, 1), 1) * x, num2cell(1:length(windows_in_cell)), windows_in_cell, 'UniformOutput', false);
    image_imdb_idx = cat(1, image_imdb_idx{:});
    windows = cat(1, windows_in_cell{:});
    windows = cellfun(@(x, y, z) [ones(size(z, 1), 1)*x, ones(size(z, 1), 1)*y, z], ...
        num2cell(1:size(windows, 1))', num2cell(image_imdb_idx), windows, 'UniformOutput', false);

end

function InitSolver(net_def_file, net_file, log_file)
    cur_dir = pwd;
    cd(fileparts(net_def_file));
    caffe('init_solver', net_def_file, net_file, log_file);
    cd(cur_dir);
end