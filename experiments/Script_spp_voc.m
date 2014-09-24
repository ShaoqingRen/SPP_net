function Script_spp_voc()
% Script_spp_voc()
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

clc;
% -------------------- CONFIG --------------------
% model def
opts.net_file               = fullfile(pwd, 'data\cnn_model\Zeiler_conv5\Zeiler_conv5');
opts.net_def_file           = fullfile(pwd, 'data\cnn_model\Zeiler_conv5\Zeiler_spm_scale224_test_conv5.prototxt');
opts.spp_params_def         = fullfile(pwd, 'data\cnn_model\Zeiler_conv5\spp_config');

% for feature cache
opts.feat_cache_copy_from   = 'Zeiler_conv5';
opts.feat_cache             = 'Zeiler_conv5';
opts.spm_im_size            = [480 576 688 874 1200];

% for finetune
opts.flip_finetune          = true;
opts.finetune_cache_name    = 'Zeiler_conv5_ft(5s_flip)';
opts.finetune_rst_dir       = fullfile(pwd, 'finetuning', opts.finetune_cache_name);
opts.finetune_net_def_file  = fullfile(pwd, 'model-defs\pascal_finetune_fc_spm_solver.prototxt');

% for svm train and test
opts.layer                  = 7;
opts.flip                   = false;
opts.cache_name             = 'Zeiler_conv5_ft(5s_flip)_fc7';

% change to point to your VOCdevkit install
opts.devkit                 = './datasets/VOCdevkit2007';

% detail setting
opts.with_selective_search  = true;
opts.with_edge_box          = false;
opts.with_hard_samples      = false;

% train set
opts                        = perpare_train_data(opts, opts.flip | opts.flip_finetune);
opts.feat_cache_train       = {opts.feat_cache};
opts.imdb_for_negative_mining = [1];
opts.neg_ovr_threshs        = {[-1, 0.3]};

% test set
opts.imdb_test              = imdb_from_voc(opts.devkit, 'test', '2007');
opts.roidb_test             = opts.imdb_test.roidb_func(opts.imdb_test, opts.with_hard_samples, opts.with_selective_search, opts.with_edge_box);
opts.feat_cache_test        = opts.feat_cache;

opts.gpu_id                 = 1;
% ------------------------------------------------

% ------------------------------------------------
g = gpuDevice(opts.gpu_id);
% 
%% extract last conv feature
opts = perpare_train_data(opts, opts.flip | opts.flip_finetune);
spp_exp_cache_features_voc('trainval', opts);
spp_exp_cache_features_voc('test', opts);

%% finetune fc layers and change model to finetuned one
opts = perpare_train_data(opts, opts.flip_finetune);
[~, ~, test_net_file, opts.max_iter] = parse_copy_finetune_prototxt(opts.finetune_net_def_file, opts.finetune_rst_dir);
finetuned_model_path = spp_finetune_voc(opts);                                % finetune
% finetuned_model_path = fullfile(opts.finetune_rst_dir, 'FT_iter_186000');       % load from finetuned file
opts.net_file        = finetuned_model_path;
opts.net_def_file    = fullfile(opts.finetune_rst_dir, test_net_file);
% 

%% svm train and test 
opts = perpare_train_data(opts, opts.flip);
spp_exp_train_and_test_voc(opts);


%% box regression
spp_exp_bbox_reg_train_and_test_voc(opts);

reset(g);

end

function opts = perpare_train_data(opts, flip)
    opts.imdb_train             = {  imdb_from_voc(opts.devkit, 'trainval', '2007', flip) };
    opts.roidb_train            = cellfun(@(x) x.roidb_func(x, opts.with_hard_samples, opts.with_selective_search, opts.with_edge_box), opts.imdb_train, 'UniformOutput', false);
end

% ------------------------------------------------
function [solver_file, train_net_file, test_net_file, max_iter] = parse_copy_finetune_prototxt(solver_file_path, dest_dir)
% copy solver, train_net and test_net to destination folder
% ------------------------------------------------  

    [folder, solver_file, ext] = fileparts(solver_file_path);
    solver_file = [solver_file, ext];
    
    solver_prototxt_text = textread(solver_file_path, '%[^\n]');
    train_net_file_pattern = '(?<=train_net[ :]*")[^"]*(?=")';
    test_net_file_pattern = '(?<=test_net[ :]*")[^"]*(?=")';
    
    train_net_file = cellfun(@(x) regexp(x, train_net_file_pattern, 'match'), solver_prototxt_text, 'UniformOutput', false);
    train_net_file = train_net_file(cellfun(@(x) ~isempty(x), train_net_file, 'UniformOutput', true));
    if isempty(train_net_file)
        error('invalid solver file %s \n', solver_file_path);
    end
    train_net_file = train_net_file{1}{1};
    
    test_net_file = cellfun(@(x) regexp(x, test_net_file_pattern, 'match'), solver_prototxt_text, 'UniformOutput', false);
    test_net_file = test_net_file(cellfun(@(x) ~isempty(x), test_net_file, 'UniformOutput', true));
    if isempty(test_net_file)
        error('invalid solver file %s \n', solver_file_path);
    end
    test_net_file = test_net_file{1}{1};
    
    mkdir_if_missing(dest_dir);
    copyfile(fullfile(folder, solver_file), dest_dir);
    copyfile(fullfile(folder, train_net_file), dest_dir);
    copyfile(fullfile(folder, test_net_file), dest_dir);  
    
    max_iter_pattern = '(?<=max_iter[ :]*)[0-9]*';
    max_iter = cellfun(@(x) regexp(x, max_iter_pattern, 'match'), solver_prototxt_text, 'UniformOutput', false);
    max_iter = max_iter(cellfun(@(x) ~isempty(x), max_iter, 'UniformOutput', true));
    if isempty(max_iter)
        error('invalid solver file %s \n', solver_file_path);
    end
    max_iter = str2double(max_iter{1}{1});
end