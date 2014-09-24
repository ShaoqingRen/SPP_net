function [spp_model] = spp_train(imdb, roidb, varargin)
% [spp_model] = spp_train(imdb, roidb, varargin)
%   Trains a SPP detector for all classes in the imdb.
%   
%   Keys that can be passed in:
%
%   svm_C             SVM regularization parameter
%   bias_mult         Bias feature value (for liblinear)
%   pos_loss_weight   Cost factor on hinge loss for positives
%   layer             Feature layer to use
%   checkpoint        Save the spp_model every checkpoint images
%   imdb_for_negative_mining
%                     Indicate whether imdb is used for negative mining
%   neg_ovr_threshs   threshs for negative sample
%   net_file          Path to the Caffe CNN to use
%   net_def_file      Path to the Caffe CNN to use
%   spp_params_def    Path to spp params defination
%   cache_name        Path to cache dir for result
%   feat_cache        Path to the precomputed feature cache
%
% Adapted from spp code written by Ross Girshick
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Shaoqing Ren
% 
% This file is part of the SPP code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------


ip = inputParser;
ip.addRequired('imdb', @iscell);
ip.addRequired('roidb', @iscell);
ip.addParamValue('svm_C',           10^-3,  @isscalar);
ip.addParamValue('bias_mult',       10,     @isscalar);
ip.addParamValue('pos_loss_weight', 2,      @isscalar);
ip.addParamValue('layer',           7,      @isscalar);
ip.addParamValue('checkpoint',      0,      @isscalar);
ip.addParamValue('imdb_for_negative_mining', ...     
    [], @ismatrix);
ip.addParamValue('neg_ovr_threshs', {}, @iscell);
ip.addParamValue('net_file', ...
    './data/caffe_nets/finetune_voc_2007_trainval_iter_70k', ...
    @isstr);
ip.addParamValue('net_def_file', ...
    './model-defs/spp_batch_256_output_fc7.prototxt', ...
    @isstr);
ip.addParamValue('spp_params_def', ...
    '', @isstr);
ip.addParamValue('cache_name', ...
    'v1_finetune_voc_2007_trainval_iter_70000', @isstr);
ip.addParamValue('feat_cache', ...
    '', @iscell);

ip.parse(imdb, roidb, varargin{:});
opts = ip.Results;

t_start = tic();
imdbs_name = cell2mat(cellfun(@(x) x.name, imdb, 'UniformOutput', false));
conf = spp_config('sub_dir', fullfile(opts.cache_name, imdbs_name));

fprintf('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Training options:\n');
disp(opts);
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');

% check all imdb consistency
assert(all(cellfun(@(x) isequal(imdb{1}.classes, x.classes), imdb, 'UniformOutput', true)));
assert(all(cellfun(@(x) isequal(imdb{1}.class_ids, x.class_ids), imdb, 'UniformOutput', true)));
assert(length(opts.imdb_for_negative_mining) == length(imdb));
assert(length(imdb) == length(opts.feat_cache));
assert(length(opts.neg_ovr_threshs) == length(imdb));

% ------------------------------------------------------------------------
% Create a new spp model
spp_model = spp_create_model(opts.net_def_file, opts.net_file, opts.spp_params_def, opts.cache_name);
spp_model.feat_cache = opts.feat_cache;
spp_model = spp_load_model(spp_model, conf.use_gpu);
[spp_model.cnn.layers, spp_model.cnn.first_fc_idx] = spp_locate_trans_fc(spp_model.cnn.layers);
if opts.layer < spp_model.cnn.first_fc_idx - 1
    error('opts.layer must be fc layers or last conv layer');
end 
if conf.use_gpu
    spp_model.cnn.layers = spp_layers_in_gpu(spp_model.cnn.layers);
end
spp_model.classes = imdb{1}.classes;
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Get the average norm of the features
opts.feat_norm_mean = spp_feature_stats(imdb, roidb, opts.layer, spp_model);
fprintf('average norm = %.3f\n', opts.feat_norm_mean);
spp_model.training_opts = opts;
spp_model.training_opts = rmfield(spp_model.training_opts, 'imdb');
spp_model.training_opts = rmfield(spp_model.training_opts, 'roidb');
spp_model.training_opts = rmfield(spp_model.training_opts, 'feat_cache');
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Get all positive examples
% We cache only the last conv features and convert them on-the-fly to spm
% pool fc as required
save_file = sprintf('%s/gt_pos_layer_%d_cache.mat', ...
    conf.cache_dir, spp_model.cnn.first_fc_idx-1);
try
  load(save_file);
  fprintf('Loaded saved positives from ground truth boxes\n');
catch
  [X_pos, keys_pos] = get_positive_poolX_features(imdb, roidb, spp_model.spp_pooler, opts);
  save(save_file, 'X_pos', 'keys_pos', '-v7.3');
end
% Init training caches
caches = cell(1, length(imdb{1}.class_ids));
for i = imdb{1}.class_ids
  fprintf('%14s has %6d positive instances\n', ...
      imdb{1}.classes{i}, size(X_pos{i},2));
  X_pos{i} = spp_poolX_to_fcX(X_pos{i}, opts.layer, spp_model, conf.use_gpu);
  X_pos{i} = spp_scale_features(X_pos{i}, opts.feat_norm_mean);
  caches{i} = init_cache(X_pos{i}, keys_pos{i});
end
clear X_pos keys_pos
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Train with hard negative mining
first_time = true;
% one pass over the data is enough
max_hard_epochs = 1;

for hard_epoch = 1:max_hard_epochs
  totalImIdx = 0;
  for idb = 1:length(imdb)
      if opts.imdb_for_negative_mining(idb) == 0
          fprintf('%s: hard neg epoch: skip imdb %d/%d \n', procid(), idb, length(imdb));
          continue;
      end
      for i = 1:length(imdb{idb}.image_ids)
        fprintf('%s: hard neg epoch: %d/%d imdb: %d/%d image: %d/%d\n', ...
                procid(), hard_epoch, max_hard_epochs, idb, length(imdb), i, length(imdb{idb}.image_ids));

        % Get hard negatives for all classes at once (avoids loading feature cache
        % more than once)
        [X, keys] = sample_negative_features(first_time, spp_model, caches, ...
            imdb{idb}, roidb{idb}, opts.feat_cache{idb}, i, totalImIdx, conf, idb);

        % Add sampled negatives to each classes training cache, removing
        % duplicates
        for j = imdb{idb}.class_ids
          if ~isempty(keys{j})
            if ~isempty(caches{j}.keys_neg)
              [~, ~, dups] = intersect(caches{j}.keys_neg, keys{j}, 'rows');
              assert(isempty(dups));
            end

            caches{j}.X_neg{end+1} = X{j};
            X{j} = [];
            caches{j}.X_neg_num = caches{j}.X_neg_num + size(keys{j},1);
            caches{j}.keys_neg = cat(1, caches{j}.keys_neg, keys{j});
            caches{j}.num_added = caches{j}.num_added + size(keys{j},1);
          end

          % Update model if
          %  - first time seeing negatives
          %  - more than retrain_limit negatives have been added
          %  - its the final image of the final epoch
          is_last_time = (hard_epoch == max_hard_epochs && i == length(imdb{idb}.image_ids));
          hit_retrain_limit = (caches{j}.num_added > caches{j}.retrain_limit);
          if (first_time || hit_retrain_limit || is_last_time) && ...
              caches{j}.X_neg_num
            caches{j}.X_neg{1} = cell2mat(caches{j}.X_neg);
            caches{j}.X_neg(2:end) = [];
            fprintf('>>> Updating %s detector <<<\n', imdb{idb}.classes{j});
            fprintf('Cache holds %d pos examples %d neg examples\n', ...
                    size(caches{j}.X_pos, 2), size(caches{j}.X_neg{1}, 2));
            [new_w, new_b] = update_model(caches{j}, opts);
            spp_model.detectors.W(j, :) = new_w;
            spp_model.detectors.B(j, :) = new_b;
            caches{j}.num_added = 0;

            z_pos = new_w * caches{j}.X_pos + new_b;
            z_neg = new_w * caches{j}.X_neg{1} + new_b;

            caches{j}.pos_loss(end+1) = opts.svm_C * opts.pos_loss_weight * ...
                                        sum(max(0, 1 - z_pos));
            caches{j}.neg_loss(end+1) = opts.svm_C * sum(max(0, 1 + z_neg));
            caches{j}.reg_loss(end+1) = 0.5 * new_w * new_w'+ ...
                                        0.5 * (new_b / opts.bias_mult)^2;
            caches{j}.tot_loss(end+1) = caches{j}.pos_loss(end) + ...
                                        caches{j}.neg_loss(end) + ...
                                        caches{j}.reg_loss(end);

            for t = 1:length(caches{j}.tot_loss)
              fprintf('    %2d: obj val: %.3f = %.3f (pos) + %.3f (neg) + %.3f (reg)\n', ...
                      t, caches{j}.tot_loss(t), caches{j}.pos_loss(t), ...
                      caches{j}.neg_loss(t), caches{j}.reg_loss(t));
            end

            % store negative support vectors for visualizing later
            SVs_neg = find(z_neg > -1 - eps);
            spp_model.SVs.keys_neg{j} = caches{j}.keys_neg(SVs_neg, :);
            spp_model.SVs.scores_neg{j} = z_neg(SVs_neg);

            % evict easy examples
            easy = find(z_neg < caches{j}.evict_thresh);
            caches{j}.X_neg{1}(:, easy) = [];
            caches{j}.X_neg_num = caches{j}.X_neg_num - length(easy);
            caches{j}.keys_neg(easy,:) = [];

            fprintf('  Pruning easy negatives\n');
            fprintf('  Cache holds %d pos examples %d neg examples\n', ...
                    size(caches{j}.X_pos,2), size(caches{j}.X_neg{1}, 2));
            fprintf('  %d pos support vectors\n', numel(find(z_pos <  1 + eps)));
            fprintf('  %d neg support vectors\n', numel(find(z_neg > -1 - eps)));
          end
        end
        
        first_time = false;

        if opts.checkpoint > 0 && mod(i, opts.checkpoint) == 0
          save_spp_model(spp_model, [conf.cache_dir 'spp_model']);
        end
      end
      totalImIdx = totalImIdx + length(imdb{idb}.image_ids);
      save_spp_model(spp_model, [conf.cache_dir 'spp_model_imdb_' num2str(idb)]);
  end
end
% save the final spp_model
save_spp_model(spp_model, [conf.cache_dir 'spp_model']);

fprintf('spp_train_spm in %f seconds.\n', toc(t_start));
% ------------------------------------------------------------------------


% ------------------------------------------------------------------------
function save_spp_model(spp_model, file_path)
% ------------------------------------------------------------------------
for i = 1:length(spp_model.cnn.layers)
    if isfield(spp_model.cnn.layers(i), 'weights_gpu')
        spp_model.cnn.layers(i).weights_gpu = [];
    end
end
save(file_path, 'spp_model', '-v7.3');

% ------------------------------------------------------------------------
function [X_neg, keys] = sample_negative_features(first_time, spp_model, ...
                                                  caches, imdb, roidb, feat_cache, ind, ind_start, conf, imdb_id)
% ------------------------------------------------------------------------
opts = spp_model.training_opts;

d = roidb.rois(ind);

dedup = true;
[d.feat, unique_box_id] = spp_load_cached_poolX_features(spp_model.spp_pooler, feat_cache, ...
    imdb.name, imdb.image_ids{ind}, d.boxes, false, dedup);

class_ids = imdb.class_ids;

if isempty(d.feat)
  X_neg = cell(max(class_ids), 1);
  keys = cell(max(class_ids), 1);
  return;
end

d.feat = spp_poolX_to_fcX(d.feat, opts.layer, spp_model, conf.use_gpu);
d.feat = spp_scale_features(d.feat, opts.feat_norm_mean);

neg_ovr_thresh = opts.neg_ovr_threshs{imdb_id};

if first_time
  for cls_id = class_ids
    I = intersect(unique_box_id, find((d.overlap(:, cls_id) >= neg_ovr_thresh(1)) & (d.overlap(:, cls_id) < neg_ovr_thresh(2))));
    X_neg{cls_id} = d.feat(:, I);
    keys{cls_id} = [(ind+ind_start)*ones(length(I), 1), I];
  end
else
  zs = bsxfun(@plus, spp_model.detectors.W * d.feat, spp_model.detectors.B)';
  for cls_id = class_ids
    z = zs(:, cls_id);
    I = intersect(unique_box_id, ...
        find((z > caches{cls_id}.hard_thresh) & ...
        (d.overlap(:, cls_id) >= neg_ovr_thresh(1)) & ...
        (d.overlap(:, cls_id) < neg_ovr_thresh(2))  ));

    % Avoid adding duplicate features
    keys_ = [(ind+ind_start)*ones(length(I),1) I];
    if ~isempty(caches{cls_id}.keys_neg) && ~isempty(keys_)
      [~, ~, dups] = intersect(caches{cls_id}.keys_neg, keys_, 'rows');
      keep = setdiff(1:size(keys_,1), dups);
      I = I(keep);
    end

    % Unique hard negatives
    X_neg{cls_id} = d.feat(:, I);
    keys{cls_id} = [(ind+ind_start)*ones(length(I),1) I];
  end
end


% ------------------------------------------------------------------------
function [w, b] = update_model(cache, opts, pos_inds, neg_inds)
% ------------------------------------------------------------------------
solver = 'liblinear';
liblinear_type = 3;  % l2 regularized l1 hinge loss
%liblinear_type = 5; % l1 regularized l2 hinge loss

if ~exist('pos_inds', 'var') || isempty(pos_inds)
  num_pos = size(cache.X_pos, 2);
  pos_inds = 1:num_pos;
else
  num_pos = length(pos_inds);
  fprintf('[subset mode] using %d out of %d total positives\n', ...
      num_pos, size(cache.X_pos,2));
end
if ~exist('neg_inds', 'var') || isempty(neg_inds)
  num_neg = size(cache.X_neg{1}, 2);
  neg_inds = 1:num_neg;
else
  num_neg = length(neg_inds);
  fprintf('[subset mode] using %d out of %d total negatives\n', ...
      num_neg, size(cache.X_neg{1}, 2));
end

switch solver
  case 'liblinear'
    ll_opts = sprintf('-w1 %.5f -c %.5f -s %d -B %.5f', ...
                      opts.pos_loss_weight, opts.svm_C, ...
                      liblinear_type, opts.bias_mult);
    fprintf('liblinear opts: %s\n', ll_opts);
    
    X = [cache.X_pos(:, pos_inds), cache.X_neg{1}(:, neg_inds)];
    y = cat(1, ones(num_pos,1), -ones(num_neg,1));
    llm = liblinear_train(y, X, ll_opts, 'col');
    w = single(llm.w(1:end-1)')';
    b = single(llm.w(end)*opts.bias_mult);
    
  otherwise
    error('unknown solver: %s', solver);
end

% ------------------------------------------------------------------------
function [pos_inds, neg_inds] = get_cache_inds_from_fold(cache, fold)
% ------------------------------------------------------------------------
pos_inds = find(ismember(cache.keys_pos(:,1), fold) == false);
neg_inds = find(ismember(cache.keys_neg(:,1), fold) == false);


% ------------------------------------------------------------------------
function [X_pos, keys] = get_positive_poolX_features(imdb, roidb, spp_pooler, opts)
% ------------------------------------------------------------------------
t_start = tic();

assert(all(cellfun(@(x) isequal(imdb{1}.class_ids, x.class_ids), imdb, 'UniformOutput', true)));

X_pos = {};
keys = {};

nLoadEach = 5000; % load n images in each parfor
totalImIdx = 0;
for idb = 1:length(imdb)
    image_ids = imdb{idb}.image_ids;
    imdb_name = imdb{idb}.name;
    rois = roidb{idb}.rois;
    feat_cache = opts.feat_cache{idb};
    class_ids = imdb{idb}.class_ids;
    
    for iturn = 1:ceil(length(image_ids) / nLoadEach)
        t_turn = tic();
        
        i_start = (iturn - 1) * nLoadEach + 1;
        i_end = min(iturn * nLoadEach, length(image_ids));
        i_num = i_end - i_start + 1;

        X_pos_sub_trans = cell(i_num, 1);
        X_pos_sub_trans = cellfun(@(x) cell(length(class_ids), 1), X_pos_sub_trans, 'UniformOutput', false);
        keys_sub_trans = cell(i_num, 1);
        keys_sub_trans = cellfun(@(x) cell(length(class_ids), 1), keys_sub_trans, 'UniformOutput', false);

        image_ids_sub = image_ids(i_start:i_end);
        i_sub = i_start:i_end;
        
        rois_sub = rois(i_sub);
        parfor i = 1:i_num
%         for i = 1:i_num
          d = rois_sub(i);
   
          pos_boxes = find(d.class ~= 0);
          if isempty(pos_boxes)
              continue;
          end
          
          d.feat = spp_load_cached_poolX_features(spp_pooler, feat_cache, ...
              imdb_name, image_ids_sub{i}, d.boxes(pos_boxes, :));         
          d.class = d.class(pos_boxes);
          %caution only d.feat and d.class is operated

          for j = class_ids
            sel = find(d.class == j);
            if ~isempty(sel)
              X_pos_sub_trans{i}{j} = d.feat(:, sel);
              keys_sub_trans{i}{j} = [(i+totalImIdx)*ones(1, length(sel)); sel'];
            end
          end
        end

        X_pos_sub = cat(2, X_pos_sub_trans{:});
        X_pos_sub = num2cell(X_pos_sub, 2);
        keys_sub = cat(2, keys_sub_trans{:});
        keys_sub = num2cell(keys_sub, 2);

        X_pos{end+1} = cellfun(@(x) cat(2, x{:}), X_pos_sub,  'UniformOutput', false);
        keys{end+1} = cellfun(@(x) cat(2, x{:})', keys_sub,  'UniformOutput', false);
        
        fprintf('get_positive_poolX_features imdb %d image %d/%d in %f seconds. \n', idb, i_end, length(image_ids), toc(t_turn));
        
        totalImIdx = totalImIdx + i_num;
    end
end

X_pos = cat(2, X_pos{:});
X_pos = num2cell(X_pos, 2);
keys = cat(2, keys{:});
keys = num2cell(keys, 2);

X_pos = cellfun(@(x) cat(2, x{:}), X_pos,  'UniformOutput', false);
keys = cellfun(@(x) cat(1, x{:}), keys,  'UniformOutput', false);

fprintf('get_positive_pool5_features in %f seconds. \n', toc(t_start));


% ------------------------------------------------------------------------
function cache = init_cache(X_pos, keys_pos)
% ------------------------------------------------------------------------
cache.X_pos = X_pos;
cache.X_neg = {};
cache.X_neg_num = 0;
cache.keys_neg = [];
cache.keys_pos = keys_pos;
cache.num_added = 0;
cache.retrain_limit = 2000;
cache.evict_thresh = -1.2;
cache.hard_thresh = -1.0001;
cache.pos_loss = [];
cache.neg_loss = [];
cache.reg_loss = [];
cache.tot_loss = [];