function expected_scale =  spm_expected_scale(min_img_sz, boxes, spm_params)
% expected_scale =  spm_expected_scale(min_img_sz, boxes, spm_params)
%   
%   min_img_sz      min(size(im, 1), size(im, 2))
%   boxes           in rows with (l, t, r, b)
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

area = (boxes(:, 3) - boxes(:, 1) + 1) .* (boxes(:, 4) - boxes(:, 2) + 1);

expected_scale = spm_params.sz_conv_standard * spm_params.step_standard * min_img_sz ./ sqrt(area);
%     scale_expected = standard_img_size * min_img_sz ./ sqrt(area);
expected_scale = round(expected_scale(:));

end