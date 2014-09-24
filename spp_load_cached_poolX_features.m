function [feat, unique_box_id] = spp_load_cached_poolX_features...
    (spp_pooler, cache_name, imdb_name, id, boxes, random_scale, dedup)
% [feat, unique_box_id] = spp_load_cached_poolX_features...
%    (spp_pooler, cache_name, imdb_name, id, boxes, random_scale, dedup)
%   loads cached last conv features from:
%   feat_cache/[cache_name]/[imdb_name]/[id].mat
%   and online pool 
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
    random_scale = false;
end

if nargin < 7
    dedup = false;
end

file = sprintf('./feat_cache/%s/%s/%s', cache_name, imdb_name, id);

if exist([file '.mat'], 'file')
  d = load(file);
  
  % feat in columns
  if ~isempty(boxes)
      [feat, unique_box_id] = spp_features_convX_to_poolX(spp_pooler, d.feat, boxes, random_scale, dedup);
  else
      unique_box_id = [];
      feat = [];
  end
else
  warning('could not load: %s', file);
  feat = single([]);
end
