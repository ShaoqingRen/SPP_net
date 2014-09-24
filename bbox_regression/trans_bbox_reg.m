function bbox_reg = trans_bbox_reg(bbox_reg, use_gpu)
% translate bbox_reg model for fast compute
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

    for i = 1:length(bbox_reg.models)
        bbox_reg.models{i}.Beta = bbox_reg.models{i}.Beta';
        if use_gpu
            bbox_reg.models{i}.Beta_gpu = gpuArray(bbox_reg.models{i}.Beta);
        end
    end
    
    bbox_reg.combined_Beta = cell2mat(cellfun(@(x) x.Beta', bbox_reg.models', 'UniformOutput', false))';
    if use_gpu
        bbox_reg.combined_Beta_gpu = gpuArray(bbox_reg.combined_Beta);
    end
    bbox_reg.combined_T_inv = cell2mat(cellfun(@(x) x.T_inv, bbox_reg.models, 'UniformOutput', false));
end