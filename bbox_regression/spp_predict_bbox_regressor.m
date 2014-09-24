function pred_boxes = ...
    spp_predict_bbox_regressor(model, feat, ex_boxes, use_gpu)
% pred_boxes = spp_predict_bbox_regressor(model, feat, ex_boxes, use_gpu)

%   Predicts a new bounding box from CNN features computed on input
%   bounding boxes.
%   
%   Inputs
%   model     Bounding box regressor from rcnn_train_bbox_regressor.m
%   feat      Input feature vectors
%   ex_boxes  Input bounding boxes
%
%   Outputs
%   pred_boxes  Modified (hopefully better) ex_boxes
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
if isempty(ex_boxes)
  pred_boxes = [];
  return;
end

% Predict regression targets
if use_gpu
    feat_gpu = gpuArray(feat);
    Y_gpu = bsxfun(@plus, model.Beta_gpu(:, 1:end-1) * feat_gpu, model.Beta_gpu(:, end));
    Y = gather(Y_gpu);
else
    Y = bsxfun(@plus, model.Beta(:, 1:end-1) * feat, model.Beta(:, end));
end
% Invert whitening transformation
Y = bsxfun(@plus, Y'*model.T_inv, model.mu);

% Read out predictions
dst_ctr_x = Y(:,1);
dst_ctr_y = Y(:,2);
dst_scl_x = Y(:,3);
dst_scl_y = Y(:,4);

src_w = ex_boxes(:,3) - ex_boxes(:,1) + eps;
src_h = ex_boxes(:,4) - ex_boxes(:,2) + eps;
src_ctr_x = ex_boxes(:,1) + 0.5*src_w;
src_ctr_y = ex_boxes(:,2) + 0.5*src_h;

pred_ctr_x = (dst_ctr_x .* src_w) + src_ctr_x;
pred_ctr_y = (dst_ctr_y .* src_h) + src_ctr_y;
pred_w = exp(dst_scl_x) .* src_w;
pred_h = exp(dst_scl_y) .* src_h;
pred_boxes = [pred_ctr_x - 0.5*pred_w, pred_ctr_y - 0.5*pred_h, ...
              pred_ctr_x + 0.5*pred_w, pred_ctr_y + 0.5*pred_h];
