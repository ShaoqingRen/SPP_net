function dets = spp_detect(im, spp_model, spm_im_size, use_gpu)
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

% compute selective search candidates
fprintf('Computing candidate regions...');
th = tic();
fast_mode = true;
boxes = selective_search_boxes(im, fast_mode);

% compat: change coordinate order from [y1 x1 y2 x2] to [x1 y1 x2 y2]
boxes = boxes(:, [2 1 4 3]);
fprintf('found %d candidates (in %.3fs).\n', size(boxes,1), toc(th));

if isempty(boxes)
    dets = {};
    return;
end

% extract features from candidates (one row per candidate box)
fprintf('Extracting CNN features from regions...');
th = tic();

% calc_fc_in_matlab = true;
feat = spp_features_convX(im, spm_im_size, [], use_gpu);
feat = spp_features_convX_to_poolX(spp_model.spp_pooler, feat, boxes, false);
feat = spp_poolX_to_fcX(feat, spp_model.training_opts.layer, spp_model, use_gpu);
feat = spp_scale_features(feat, spp_model.training_opts.feat_norm_mean);

fprintf('done (in %.3fs).\n', toc(th));

% compute scores for each candidate [num_boxes x num_classes]
fprintf('Scoring regions with detectors...');
th = tic();
scores = bsxfun(@plus, spp_model.detectors.W*feat, spp_model.detectors.B);
fprintf('done (in %.3fs)\n', toc(th));

fprintf('Applying NMS...');
% apply NMS to each class and return final scored detections
scored_boxes = cat(2, boxes, scores');
keeps = nms_multiclass(scored_boxes, 0.3);
dets = cellfun(@(x, y) [boxes(x, :), scores(y, x)'], keeps, num2cell(1:length(keeps))', 'UniformOutput', false);
fprintf('done (in %.3fs)\n', toc(th));
