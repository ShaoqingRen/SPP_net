function [mean_norm, stdd] = spp_feature_stats(imdb, roidb, layer, spp_model)
% [mean_norm, stdd] = spp_feature_stats(imdb, roidb, layer, spp_model)
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

imdbs_name = cell2mat(cellfun(@(x) x.name, imdb, 'UniformOutput', false));
conf = spp_config('sub_dir', fullfile(spp_model.cache_name, imdbs_name));
save_file = sprintf('%s/feature_stats_layer_%d.mat', ...
                    conf.cache_dir, layer);

t_start = tic();
try
  ld = load(save_file);
  mean_norm = ld.mean_norm;
  stdd = ld.stdd;
  clear ld;
catch
  % fix the random seed for repeatability
  prev_rng = seed_rand();

  image_idx_in_imdb = cell2mat(cellfun(@(x) 1:length(x.image_ids), imdb, 'UniformOutput', false));
  image_imdb_id = cell2mat(cellfun(@(x, y) ones(1, length(x.image_ids))*y, imdb, num2cell(1:length(imdb)), 'UniformOutput', false));
  
  num_images = min(length(image_idx_in_imdb), 200);
  boxes_per_image = 200;

  valid_idx = randperm(length(image_idx_in_imdb), num_images);

  ns = [];
  for i = 1:length(valid_idx)
    image_idx = valid_idx(i);
    tic_toc_print('feature stats: %d/%d\n', i, length(valid_idx));

    imdb_idx = image_imdb_id(image_idx);
    d = roidb{imdb_idx}.rois(image_idx_in_imdb(image_idx));
    d.feat = spp_load_cached_poolX_features(spp_model.spp_pooler, spp_model.feat_cache{imdb_idx}, ...
        imdb{imdb_idx}.name, imdb{imdb_idx}.image_ids{image_idx_in_imdb(image_idx)}, d.boxes);
    if isempty(d.feat)
        continue;
    end
    X = d.feat(:, randperm(size(d.feat,2), min(boxes_per_image, size(d.feat,2))));
    X = spp_poolX_to_fcX(X, layer, spp_model, conf.use_gpu);

    ns = cat(2, ns, sqrt(sum(X.^2, 1)));
  end

  mean_norm = mean(ns);
  stdd = std(ns);
  save(save_file, 'mean_norm', 'stdd');

  % restore previous rng
  rng(prev_rng);
end
fprintf('spp_feature_stats_spm in %f seconds.\n', toc(t_start));
