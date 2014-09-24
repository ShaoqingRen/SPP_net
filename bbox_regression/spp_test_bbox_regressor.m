function res = spp_test_bbox_regressor(imdb, roidb, spp_model, bbox_reg, feat_cache_test, suffix, fast)
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
if nargin < 7
    fast = false;
end

conf = spp_config('sub_dir', fullfile(spp_model.cache_name, imdb.name));
image_ids = imdb.image_ids;

% assume they are all the same
feat_opts = bbox_reg.training_opts;
num_classes = length(spp_model.classes);

% translate regression mat for fast compute
bbox_reg = trans_bbox_reg(bbox_reg, conf.use_gpu);

if ~exist('suffix', 'var') || isempty(suffix)
  suffix = '_bbox_reg';
else
  if suffix(1) ~= '_'
    suffix = ['_' suffix];
  end
end

aboxes = cell(num_classes, 1);
box_inds = cell(num_classes, 1);
for i = 1:num_classes
  load([conf.cache_dir spp_model.classes{i} '_boxes_' imdb.name]);
  aboxes{i} = boxes;
  box_inds{i} = inds;
  clear boxes inds;
end

for i = 1:length(image_ids)
  fprintf('%s: bbox reg test (%s) %d/%d', procid(), imdb.name, i, length(image_ids));
  th = tic;
  d = roidb.rois(i);
  d.feat = spp_load_cached_poolX_features(spp_model.spp_pooler, feat_cache_test, ...
      imdb.name, image_ids{i}, d.boxes);
  if isempty(d.feat)
    continue;
  end

  d.feat = spp_poolX_to_fcX(d.feat, feat_opts.layer, spp_model, false);
  d.feat = spp_scale_features(d.feat, feat_opts.feat_norm_mean);

  if feat_opts.binarize
    d.feat = single(d.feat > 0);
  end
  
  for j = 1:num_classes
    I = box_inds{j}{i};
    boxes = aboxes{j}{i};
    if ~isempty(boxes)
      scores = boxes(:,end);
      boxes = boxes(:,1:4);
      assert(nnz(d.boxes(I,:) - boxes) == 0);
      boxes = spp_predict_bbox_regressor(bbox_reg.models{j}, d.feat(:,I), boxes, false);
      boxes(:,1) = max(boxes(:,1), 1);
      boxes(:,2) = max(boxes(:,2), 1);
      boxes(:,3) = min(boxes(:,3), imdb.sizes(i,2));
      boxes(:,4) = min(boxes(:,4), imdb.sizes(i,1));
      aboxes{j}{i} = cat(2, single(boxes), single(scores));

      if 0
        % debugging visualizations
        im = imread(imdb.image_at(i));
        keep = nms(aboxes{j}{i}, 0.3);
        for k = 1:min(10, length(keep))
          if aboxes{j}{i}(keep(k),end) > -0.9
            showboxes(im, aboxes{j}{i}(keep(k),1:4));
            title(sprintf('%s %d score: %.3f\n', spp_model.classes{j}, ...
                k, aboxes{j}{i}(keep(k),end)));
            pause;
          end
        end
      end
    end
  end
  fprintf(' time: %.3fs\n', toc(th));
end

for i = 1:num_classes
  save_file = [conf.cache_dir spp_model.classes{i} '_boxes_' imdb.name suffix];
  boxes = aboxes{i};
  inds = box_inds{i};
  save(save_file, 'boxes', 'inds');
  clear boxes inds;
end

% ------------------------------------------------------------------------
% Peform AP evaluation
% ------------------------------------------------------------------------
if isequal(imdb.eval_func, @imdb_eval_voc)
    if fast
        parfor model_ind = 1:num_classes
          cls = spp_model.classes{model_ind};
          res(model_ind) = imdb.eval_func(cls, aboxes{model_ind}, imdb, spp_model.cache_name, suffix, fast);
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
    fprintf('Results (bbox reg):\n');
    aps = [res(:).ap]'*100;
    disp(aps);
    disp(mean(aps));
    fprintf('~~~~~~~~~~~~~~~~~~~~\n');
end
