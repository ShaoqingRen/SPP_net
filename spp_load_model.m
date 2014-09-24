function spp_model = spp_load_model(spp_model_or_file, use_gpu)
% spp_model = spp_load_model(spp_model_or_file, use_gpu)
%   Takes an spp_model structure and loads the associated Caffe
%   CNN into memory. Since this is nasty global state that is carried
%   around, a randomly generated 'key' (or handle) is returned.
%   Before making calls to caffe it's a good idea to check that
%   spp_model.cnn.key is the same as caffe('get_init_key').
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

if ischar(spp_model_or_file)
  assert(exist(spp_model_or_file, 'file') ~= 0);
  ld = load(spp_model_or_file);
  spp_model = ld.spp_model; clear ld;
  return;
else
  spp_model = spp_model_or_file;
end

cur_dir = pwd;
cd(fileparts(spp_model.cnn.definition_file));

spp_model.cnn.init_key = ...
    caffe('init', spp_model.cnn.definition_file, spp_model.cnn.binary_file);
if exist('use_gpu', 'var') && ~use_gpu
  caffe('set_mode_cpu');
else
  caffe('set_mode_gpu');
end
caffe('set_phase_test');
spp_model.cnn.layers = caffe('get_weights');
% caffe('release');

cd(cur_dir);
