function res = spp_test(spp_model, imdb, roidb, feat_cache, suffix, fast, evaluate)
% res = spp_test(spp_model, imdb, roidb, feat_cache, suffix, fast, evaluate)
%   Compute test results using the trained spp_model on the
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

if nargin < 5
    suffix = '';
end

if nargin < 6
    fast = false;
end

if nargin < 7
    evaluate = true;
end

t_start = tic();

conf = spp_config('sub_dir', fullfile(spp_model.cache_name, imdb.name));
image_ids = imdb.image_ids;

% assume they are all the same
feat_opts = spp_model.training_opts;
feat_opts.feat_cache = feat_cache;
num_classes = length(spp_model.classes);
rois = roidb.rois;

if conf.use_gpu
    spp_model.cnn.layers = spp_layers_in_gpu(spp_model.cnn.layers);
end

if ~exist('suffix', 'var') || isempty(suffix)
  suffix = '';
else
  suffix = ['_' suffix];
end

try
  aboxes = cell(num_classes, 1);
  for i = 1:num_classes
    load([conf.cache_dir spp_model.classes{i} '_boxes_' imdb.name suffix]);
    aboxes{i} = boxes;
  end
catch
  aboxes = cell(num_classes, 1);
  box_inds = cell(num_classes, 1);
  for i = 1:num_classes
    aboxes{i} = cell(length(image_ids), 1);
    box_inds{i} = cell(length(image_ids), 1);
  end

  max_per_set = 5 * length(image_ids);
  max_per_image = 100;
  top_scores = cell(num_classes, 1);
  thresh = -1.5*ones(num_classes, 1);

  if ~isfield(spp_model, 'folds')
    folds{1} = 1:length(image_ids);
  else
    folds = spp_model.folds;
  end

  count = 0;
  for f = 1:length(folds)
    for i = folds{f}
      count = count + 1;
      fprintf('%s: test (%s) %d/%d ', procid(), imdb.name, count, length(image_ids));
      th = tic;
      d = rois(i);
      d.feat = spp_load_cached_poolX_features(spp_model.spp_pooler, feat_opts.feat_cache, ...
          imdb.name, image_ids{i}, d.boxes);
      if isempty(d.feat)
        continue;
      end
      d.feat = spp_poolX_to_fcX(d.feat, feat_opts.layer, spp_model, conf.use_gpu);
      d.feat = spp_scale_features(d.feat, feat_opts.feat_norm_mean);
      zs = bsxfun(@plus, spp_model.detectors(f).W * d.feat, spp_model.detectors(f).B)';

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
      end
      
      if mod(count, 1000) == 0
          [aboxes{j}, box_inds{j}, thresh(j)] = ...
             keep_top_k(aboxes{j}, box_inds{j}, i, max_per_set, thresh(j));
      end
      fprintf(' time: %.3fs\n', toc(th));
      if mod(count, 1000) == 0
        disp(thresh);
      end
      
    end
  end

  for i = 1:num_classes
      
    top_scores{i} = sort(top_scores{i}, 'descend');  
    if (length(top_scores{i}) > max_per_set)
        thresh(i) = top_scores{i}(max_per_set);
    end
      
    % go back through and prune out detections below the found threshold
    for j = 1:length(image_ids)
      if ~isempty(aboxes{i}{j})
        I = find(aboxes{i}{j}(:,end) < thresh(i));
        aboxes{i}{j}(I,:) = [];
        box_inds{i}{j}(I,:) = [];
      end
    end

    save_file = [conf.cache_dir spp_model.classes{i} '_boxes_' imdb.name suffix];
    boxes = aboxes{i};
    inds = box_inds{i};
    save(save_file, 'boxes', 'inds');
    clear boxes inds;
  end
end

fprintf('spp_test_spm in %f seconds.\n', toc(t_start));

% ------------------------------------------------------------------------
% Peform AP evaluation
% ------------------------------------------------------------------------
if ~evaluate
    res = [];
    return;
end

if isequal(imdb.eval_func, @imdb_eval_voc)
    if fast
        classes = spp_model.classes;
        cache_name = spp_model.cache_name;
        parfor model_ind = 1:num_classes
          cls = classes{model_ind};
          res(model_ind) = imdb.eval_func(cls, aboxes{model_ind}, imdb, cache_name, suffix, fast);
        end
    else
        for model_ind = 1:num_classes
          cls = spp_model.classes{model_ind};
          res(model_ind) = imdb.eval_func(cls, aboxes{model_ind}, imdb, spp_model.cache_name, suffix, fast);
        end
    end
else
% ilsvrc
    res = imdb.eval_func(aboxes, imdb, spp_model.cache_name, suffix, fast);
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

