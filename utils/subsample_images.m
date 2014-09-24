function [imdbs, roidbs] = subsample_images(imdbs, roidbs, num_per_class, seed, scale_filter)

if nargin < 5
    scale_filter = [];
end

if ~exist('seed', 'var')
  seed = [];
end

class_num = cellfun(@(x) length(x.class_ids), imdbs, 'UniformOutput', true);
assert(length(unique(class_num)) == 1);
class_num = unique(class_num);

rois = cellfun(@(x) x.rois, roidbs, 'UniformOutput', false);

rois_combine = cell2mat(rois(:));
rois_combine_class = arrayfun(@(x) x.class, rois_combine, 'UniformOutput', false);

if ~isempty(scale_filter)
    rois_combine_boxes = arrayfun(@(x) x.boxes, rois_combine, 'UniformOutput', false);
    imdb_combine_size = cellfun(@(x) x.sizes, imdbs, 'UniformOutput', false);
    imdb_combine_size = cell2mat(imdb_combine_size(:));
    imdb_combine_size = num2cell(imdb_combine_size, 2);
    
    scale_expected = cellfun(@windows_scale, rois_combine_boxes, imdb_combine_size, 'UniformOutput', false);
    
    scale_valid_idx = cellfun(@(x) all(x > min(scale_filter) & x < max(scale_filter)), scale_expected, 'UniformOutput', false);
    scale_valid_idx = cell2mat(scale_valid_idx);
else
    scale_valid_idx = true(length(rois_combine_class), 1);
end
% fix the random seed for repeatability
prev_rng = seed_rand(seed);

inds = cell(class_num, 1);
parfor i = 1:class_num
    
    valid_idx = cellfun(@(x) any(x == i), rois_combine_class, 'UniformOutput', false);
    valid_idx = cell2mat(valid_idx);
    
    valid_idx = valid_idx & scale_valid_idx;
    
    valid_num = sum(valid_idx);

    num = min(valid_num, num_per_class);
    inds{i} = 1:length(rois_combine);
    inds{i} = inds{i}(valid_idx);
    inds{i} = inds{i}(randperm(length(inds{i}), num));
end

inds = cell2mat(inds')';
inds = unique(inds);

img_idx_start = 1;
for i = 1:length(imdbs)
    imdb_img_num = length(imdbs{i}.image_ids);
    img_idx_end = img_idx_start + imdb_img_num - 1;
    inds_start = find(inds >= img_idx_start, 1, 'first');
    inds_end = find(inds <= img_idx_end, 1, 'last');
    
    inds_sub = inds(inds_start:inds_end);
    inds_sub = inds_sub - img_idx_start + 1;
    
    imdbs{i}.image_ids = imdbs{i}.image_ids(inds_sub);
    imdbs{i}.sizes = imdbs{i}.sizes(inds_sub, :);
    roidbs{i}.rois = roidbs{i}.rois(inds_sub, :);
    
    img_idx_start = img_idx_start + imdb_img_num;
end

% restore previous rng
rng(prev_rng);

end

function scale_expected = windows_scale(boxes, size)
        offset0 = 15; offset = 6.5; step_standard = 12; spm_divs = -50; sz_conv_standard = 18; standard_img_size = 224;

        if isempty(boxes)
            scale_expected = 0;
            return;
        end
        
        min_img_sz = min(size(1), size(2));

        area = (boxes(:, 3) - boxes(:, 1) + 1) .* (boxes(:, 4) - boxes(:, 2) + 1);

        scale_expected = sz_conv_standard * step_standard * min_img_sz ./ sqrt(area);
    %     scale_expected = standard_img_size * min_img_sz ./ sqrt(area);
        scale_expected = round(scale_expected(:));
%             scale_expected = min(scale_expected, 1200);
end
