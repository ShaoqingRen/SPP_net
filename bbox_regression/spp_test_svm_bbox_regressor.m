function res = spp_test_svm_bbox_regressor(rcnn_model, imdb, roidb, bbox_reg, feat_cache, suffix, fast)
% res = spp_test_svm_bbox_regressor(rcnn_model, imdb, roidb, bbox_reg, feat_cache, suffix, fast)
%   Compute test results using the trained rcnn_model on the
%   image database specified by imdb. Results are saved
%   with an optional suffix.
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

if nargin < 6
    suffix = '';
end

if nargin < 7
    fast = false;
end

t_start = tic();

conf = rcnn_config('sub_dir', fullfile(rcnn_model.cache_name, imdb.name));
image_ids = imdb.image_ids;

% assume they are all the same
feat_opts = rcnn_model.training_opts;
bbox_feat_opts = bbox_reg.training_opts;
feat_opts.feat_cache = feat_cache;
num_classes = length(rcnn_model.classes);
rois = roidb.rois;

if ~exist('suffix', 'var') || isempty(suffix)
  suffix = '';
else
  suffix = ['_' suffix];
end

bbox_reg = Trans_bbox_reg(bbox_reg, conf.use_gpu);

try
  aboxes = cell(num_classes, 1);
  for i = 1:num_classes
    load([conf.cache_dir rcnn_model.classes{i} '_boxes_' imdb.name suffix]);
    aboxes{i} = boxes;
  end
catch
  aboxes = cell(num_classes, 1);
  box_inds = cell(num_classes, 1);
  for i = 1:num_classes
    aboxes{i} = cell(length(image_ids), 1);
    box_inds{i} = cell(length(image_ids), 1);
  end

  % heuristic that yields at most 100k pre-NMS boxes
  % per 2500 images
  max_per_set = ceil(100000/2500)*length(image_ids);
  max_per_set = 100000; % raw is 30000, keep more to combine
  max_per_image = 100;
  top_scores = cell(num_classes, 1);
  thresh = -1.5*ones(num_classes, 1);
  box_counts = zeros(num_classes, 1);

  if ~isfield(rcnn_model, 'folds')
    folds{1} = 1:length(image_ids);
  else
    folds = rcnn_model.folds;
  end

  count = 0;
  for f = 1:length(folds)
    for i = folds{f}
      count = count + 1;
      fprintf('%s: test (%s) %d/%d ', procid(), imdb.name, count, length(image_ids));
      th = tic;
      d = rois(i);
      feat = rcnn_load_cached_poolX_features_spm(feat_opts.feat_cache, ...
          imdb.name, image_ids{i}, d.boxes);
      if isempty(feat)
        continue;
      end
      d.feat = rcnn_poolX_to_fcX(feat, feat_opts.layer, rcnn_model, conf.use_gpu);
      d.feat = rcnn_scale_features(d.feat, feat_opts.feat_norm_mean);
      
      d.feat_bbox = rcnn_poolX_to_fcX(feat, bbox_feat_opts.layer, rcnn_model, conf.use_gpu);
      d.feat_bbox = rcnn_scale_features(d.feat_bbox, bbox_feat_opts.feat_norm_mean);
      
      zs = bsxfun(@plus, rcnn_model.detectors(f).W * d.feat, rcnn_model.detectors(f).B)';
      
      % regression all boxes for all classes, faster though more boxes
      boxes = d.boxes;
      bboxes = rcnn_predict_bbox_regressor_batch_spm(bbox_reg, d.feat_bbox, boxes, conf.use_gpu);
      
      for j = 1:num_classes
        boxes = d.boxes;
        scores = zs(:,j);
        I = find(~d.gt & scores > thresh(j));
        keep = nms(cat(2, single(boxes(I,:)), single(scores(I))), 0.3);
        I = I(keep);
        if ~isempty(I)
          [~, ord] = sort(scores(I), 'descend');
          ord = ord(1:min(length(ord), max_per_image));
          I = I(ord);
          boxes = boxes(I,:);
          scores = scores(I);
          aboxes{j}{i} = cat(2, single(boxes), single(scores));
          box_inds{j}{i} = I;
        else
          aboxes{j}{i} = zeros(0, 5, 'single');
          box_inds{j}{i} = [];
        end
        
        I = box_inds{j}{i};
        boxes = aboxes{j}{i};
        if ~isempty(boxes)
          scores = boxes(:,end);
          boxes = boxes(:,1:4);
          assert(sum(sum(abs(d.boxes(I,:) - boxes))) == 0);
%           boxes = rcnn_predict_bbox_regressor_spm(bbox_reg.models{j}, d.feat_bbox(:,I), boxes, conf.use_gpu);
          boxes = bboxes{j}(I, :);
         
          boxes(:,1) = max(boxes(:,1), 1);
          boxes(:,2) = max(boxes(:,2), 1);
          boxes(:,3) = min(boxes(:,3), imdb.sizes(i,2));
          boxes(:,4) = min(boxes(:,4), imdb.sizes(i,1));
          aboxes{j}{i} = cat(2, single(boxes), single(scores));
        end
        
        if mod(count, 1000) == 0
          [aboxes{j}, box_inds{j}, thresh(j)] = ...
             keep_top_k(aboxes{j}, box_inds{j}, i, max_per_set, thresh(j));
        end
      end
      fprintf(' time: %.3fs\n', toc(th));
      if mod(count, 1000) == 0
        disp(thresh);
      end
    end
  end

  for i = 1:num_classes
    [aboxes{i}, box_inds{i}, thresh(i)] = ...
       keep_top_k(aboxes{i}, box_inds{i}, length(image_ids), ...
          max_per_set, thresh(i));

    save_file = [conf.cache_dir rcnn_model.classes{i} '_boxes_' imdb.name suffix];
    boxes = aboxes{i};
    inds = box_inds{i};
    save(save_file, 'boxes', 'inds');
    clear boxes inds;
  end
end

fprintf('rcnn_test_spm in %f seconds.\n', toc(t_start));

% ------------------------------------------------------------------------
% Peform AP evaluation
% ------------------------------------------------------------------------

if isequal(imdb.eval_func, @imdb_eval_voc)
    if fast
        parfor model_ind = 1:num_classes
          cls = rcnn_model.classes{model_ind};
          res(model_ind) = imdb.eval_func(cls, aboxes{model_ind}, imdb, rcnn_model.cache_name, suffix, fast);
        end
    else
        for model_ind = 1:num_classes
          cls = rcnn_model.classes{model_ind};
          res(model_ind) = imdb.eval_func(cls, aboxes{model_ind}, imdb, rcnn_model.cache_name, suffix, fast);
        end
    end
else
% ilsvrc
    res = imdb.eval_func(aboxes, imdb, rcnn_model.cache_name, suffix, fast);
end

if ~isempty(res)
    fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
    fprintf('Results:\n');
    aps = [res(:).ap]' * 100;
    disp(aps);
    disp(mean(aps));
    fprintf('~~~~~~~~~~~~~~~~~~~~\n');
end

% ------------------------------------------------------------------------
function [boxes, box_inds, thresh] = keep_top_k(boxes, box_inds, end_at, top_k, thresh)
% ------------------------------------------------------------------------
% Keep top K
X = cat(1, boxes{1:end_at});
if isempty(X)
  return;
end
scores = sort(X(:,end), 'descend');
thresh = scores(min(length(scores), top_k));
for image_index = 1:end_at
  bbox = boxes{image_index};
  keep = find(bbox(:,end) >= thresh);
  boxes{image_index} = bbox(keep,:);
  box_inds{image_index} = box_inds{image_index}(keep);
end

