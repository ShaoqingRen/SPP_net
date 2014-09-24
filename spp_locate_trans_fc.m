function [layers, first_fc_idx] = spp_locate_trans_fc(layers)
% [layers, first_fc_idx] = spp_locate_trans_fc(layers)
%   locate first fc layer with layer name with 'fcX' ( X usually be 5 or
%   any number larger than 5)
%   the fc layers are set with the proper index in layers
%   fc weights are transposed for memory friendly compute
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

fc_layer_pattern = '(?<=fc)[0-9]*';
[fc_layer_idx] = arrayfun(@(x) regexp(x.layer_names, fc_layer_pattern, 'match'), layers, 'UniformOutput', false);
is_fc_layer = cellfun(@(x) ~isempty(x), fc_layer_idx, 'UniformOutput', true);
first_fc_idx = find(is_fc_layer, 1, 'first');

if ~all(is_fc_layer(first_fc_idx:end))
    error('spp_locate_trans_fc : assert only fc layers after first fc layer');
end

for i = first_fc_idx:length(layers)
   layers(i - first_fc_idx + 1).weights{1} = layers(i).weights{1}';
   layers(i - first_fc_idx + 1).weights{2} = layers(i).weights{2};
   layers(i - first_fc_idx + 1).layer_names = layers(i).layer_names;
end

layers((length(layers)-first_fc_idx+2):end) = [];

first_fc_idx = str2double(fc_layer_idx{first_fc_idx});

fprintf('find first fc layer is fc_%d\n', first_fc_idx);


