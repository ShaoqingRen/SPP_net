function spp_params = spp_load_pooling_params(spp_params_def)
% spp_params = spp_load_pooling_params(spp_params_def)
%   spp_params -- parameters for spatial pyramid pooling for spatial model
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

[~, ~, ext] = fileparts(spp_params_def);
if isempty(ext)
    spp_params_def = [spp_params_def, '.m'];
end
assert(exist(spp_params_def, 'file') ~= 0);

% change folder to avoid too long path for eval()
cur_dir = pwd;
[spp_def_dir, spp_def_file] = fileparts(spp_params_def);

cd(spp_def_dir);
spp_params = eval(spp_def_file);
cd(cur_dir);

spp_params.expected_scale = @spm_expected_scale;
spp_params.response_boxes = @spm_response_boxes;

end