function resp_boxes = spm_response_boxes(boxes, spm_params)
% expected_scale = spm_response_boxes(boxes, spm_params)
%       
%   boxes in rows with (l, t, r, b)
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

resp_boxes = boxes;
resp_boxes(:, [1, 2]) = int32(floor((boxes(:, [1, 2]) - spm_params.offset0 + spm_params.offset) / spm_params.step_standard + 0.5) + 1);
resp_boxes(:, [3, 4]) = int32(ceil((boxes(:, [3, 4]) - spm_params.offset0 - spm_params.offset) / spm_params.step_standard - 0.5) + 1);

end