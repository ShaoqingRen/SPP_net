function spp_model = spp_create_model(cnn_definition_file, cnn_binary_file, spp_params_def, cache_name)
% spp_model = spp_create_model(cnn_definition_file, cnn_binary_file, spp_params_def, cache_name)
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

if ~exist('cache_name', 'var') || isempty(cache_name)
  cache_name = 'none';
end

% init empty convnet
assert(exist(cnn_binary_file, 'file') ~= 0);
assert(exist(cnn_definition_file, 'file') ~= 0);
cnn.binary_file = cnn_binary_file;
cnn.definition_file = cnn_definition_file;

% init empty detectors
detectors.W = [];
detectors.B = [];
detectors.nms_thresholds = [];

spp_model.cnn = cnn;
spp_model.cache_name = cache_name;
spp_model.detectors = detectors;
spp_model.spp_pooler = spp_load_pooling_params(spp_params_def);
