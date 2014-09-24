function [feat, unique_box_id] = spp_features_convX_to_poolX(spp_pooler, feat, boxes, random_scale, dedup)
% [feat, unique_box_id] = spp_features_convX_to_poolX(feat, boxes, random_scale, dedup)
%       
%   boxes           in rows with (l, t, r, b)
%   random_scale    find best scale for each box or not
%   dedup           remove duplicate boxes on response map 
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

if nargin < 4
    random_scale = false;
end

if nargin < 5
    dedup = false;
end

assert(isfield(feat, 'rsp') && ~isempty(feat.rsp));

min_img_sz = min(feat.im_height, feat.im_width);

if isempty(boxes)
    feat = [];
    return;
end

feat.scale = feat.scale(:)'; 
if ~random_scale
    expected_scale = spp_pooler.expected_scale(min_img_sz, boxes, spp_pooler);

    [~, best_scale_ids] = min(abs(bsxfun(@minus, feat.scale, expected_scale(:))), [], 2);
else
    best_scale_ids = randi(length(feat.scale), size(boxes, 1), 1);
end

boxes_scales = feat.scale(best_scale_ids(:));
scaled_boxes = bsxfun(@times, (boxes - 1), (boxes_scales(:) - 1)) / (min_img_sz - 1) + 1;

if dedup
    rep_boxes = spp_pooler.response_boxes(scaled_boxes, spp_pooler);
    rsp_keys = [rep_boxes, best_scale_ids];
    [~, ia] = unique(rsp_keys, 'rows');
    unique_box_id = sort(ia);
else
    unique_box_id = 1:size(boxes, 2);
end

feat = spm_pool(feat.rsp, spp_pooler.spm_divs, scaled_boxes', best_scale_ids, ...
    spp_pooler.offset0, spp_pooler.offset, spp_pooler.step_standard); 

end
