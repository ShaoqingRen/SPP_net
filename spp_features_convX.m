function feat = spp_features_convX(im, spm_im_size, cache_feat, use_gpu)
% feat = spp_features_convX(im, spm_im_size, cache_feat, use_gpu)
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

if nargin < 3
    cache_feat = [];
end

if nargin < 4
    use_gpu = false;
end

im_width = size(im, 2);
im_height = size(im, 1);

feat.im_height = im_height;
feat.im_width = im_width;
feat.scale = spm_im_size;
feat.rsp = {};
for i = 1:length(spm_im_size)
    scale = spm_im_size(i);
    
    % copy exist scale from cache
    if ~isempty(cache_feat)
        cache_scales = cache_feat.scale;
        cache_scale_idx = find(cache_scales == scale, 1);
        if ~isempty(cache_scale_idx)
            feat.rsp{i} = cache_feat.rsp{cache_scale_idx};
            continue;
        end
    end
    
    % resize min(width, height) to scale
    if (im_width < im_height)
        im_resized_width = scale;
        im_resized_height = im_resized_width * im_height / im_width;
    else
        im_resized_height = scale;
        im_resized_width = im_resized_height * im_width / im_height;
    end

    % We turn off antialiasing to better match OpenCV's bilinear 
    resized_im = imresize(im, [im_resized_height, im_resized_width], 'bilinear', 'antialiasing', false);

%     if ~use_gpu || (numel(resized_im) > 1200*2400*6) % this is max size for k40, should be change for different models
    if ~use_gpu || (numel(resized_im) > 1200*2400*3) % this is max size for gpu, should be change for different models
%     if ~use_gpu || (numel(resized_im) > 1200*2400*1.5) % this is max size for gpu, should be change for different models
        caffe('set_mode_cpu');
        caffe('set_gpu_forbid');
    else
        caffe('set_mode_gpu');
        caffe('set_gpu_available');
    end
    caffe_anysize_test('set_input_size', size(resized_im, 1), ...
            size(resized_im, 2), size(resized_im, 3), size(resized_im, 4));
    response = caffe_anysize_test('forward', resized_im);
    
    feat.rsp{i} = response{1};
end

end
