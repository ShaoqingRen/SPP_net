function feats_pooled = spm_pool(feats, spm_divs, boxes, best_scale_ids, offset0, offset, min_times)
% feats_pooled = spm_pool(feats, spm_divs, boxes, imgheight, imgwidth)
% feats in matlab style (height, width, channel) in cell for different
% scales
% spm_divs [6, 3, 2, 1] and etc.
% boxes in [left1, ... ; top1, ... ; right1, ... ; bottom1, ....] in
% cnn_input_images
% best_scale_ids -- best feat scale for different boxes

%% from caffe
feats = cellfun(@(x) single(x), feats, 'UniformOutput', false);
boxes = double(boxes);
% trans from (height, width, channel) to (channel, width, height)
feats = cellfun(@(x) permute(x, [3, 2, 1]), feats, 'UniformOutput', false);
best_scale_ids = int32(best_scale_ids);

offset0 = double(offset0);
offset = double(offset);
min_times = double(min_times);
spm_divs = double(spm_divs);

feats_pooled = spm_pool_caffe_mex(feats, spm_divs, boxes, best_scale_ids, [offset0, offset, min_times]);

end