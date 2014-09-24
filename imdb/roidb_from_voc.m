function roidb = roidb_from_voc(imdb, with_hard_samples, with_selective_search, with_edge_box, rootDir)
% roidb = roidb_from_voc(imdb, rootDir)
%   Builds an regions of interest database from imdb image
%   database. Uses precomputed selective search boxes available
%   in the R-CNN data package.
%
%   Inspired by Andrea Vedaldi's MKL imdb and roidb code.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

if nargin < 2
    with_hard_samples = true;
end

if nargin < 3
    with_selective_search = true;
end

if nargin < 4
    with_edge_box = false;
end

if nargin < 5
    rootDir = '.';
end

roidb.name = imdb.name;
regions_file_ss = fullfile(rootDir, sprintf('/data/selective_search_data/%s.mat', roidb.name));
regions_file_eb = fullfile(rootDir, sprintf('/data/edge_box_data/%s.mat', roidb.name));

cache_file_ss = [];
cache_file_eb = [];
if with_selective_search 
    cache_file_ss = 'ss_';
    if~exist(regions_file_ss, 'file')
        error('roidb_from_ilsvrc:: cannot find %s', regions_file_ss);
    end
end

if with_edge_box 
    cache_file_eb = 'eb_';
    if ~exist(regions_file_eb, 'file')
        error('roidb_from_ilsvrc:: cannot find %s', regions_file_eb);
    end
end

cache_file = fullfile(rootDir, ['/imdb/cache/roidb_' cache_file_ss cache_file_eb imdb.name ]);
if imdb.flip
    cache_file = [cache_file '_flip'];
end
if ~with_hard_samples
    cache_file = [cache_file '_easy'];
end
try
  load(cache_file);
catch
  VOCopts = imdb.details.VOCopts;

  addpath(fullfile(VOCopts.datadir, 'VOCcode')); 

  roidb.name = imdb.name;

  fprintf('Loading region proposals...');
  regions = [];
  if with_selective_search
        regions = load_proposals(regions_file_ss, regions);
  end
  if with_edge_box
        regions = load_proposals(regions_file_eb, regions);
  end
  fprintf('done\n');

  if ~imdb.flip
      for i = 1:length(imdb.image_ids)
        tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids));
        try
          voc_rec = PASreadrecord(sprintf(VOCopts.annopath, imdb.image_ids{i}));
        catch
          voc_rec = [];
        end
        [~, image_name1] = fileparts(imdb.image_ids{i});
        [~, image_name2] = fileparts(regions.images{i});
        assert(strcmp(image_name1, image_name2));
        roidb.rois(i) = attach_proposals(voc_rec, regions.boxes{i}, imdb.class_to_id, with_hard_samples, false);
      end
  else
      for i = 1:length(imdb.image_ids)/2
        tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids)/2);
        try
          voc_rec = PASreadrecord(sprintf(VOCopts.annopath, imdb.image_ids{i*2-1}));
        catch
          voc_rec = [];
        end
        [~, image_name1] = fileparts(imdb.image_ids{i*2-1});
        [~, image_name2] = fileparts(regions.images{i});
        assert(strcmp(image_name1, image_name2));
        roidb.rois(i*2-1) = attach_proposals(voc_rec, regions.boxes{i}, imdb.class_to_id, with_hard_samples, false);
        assert(imdb.flip_from(i*2) == i*2-1);
        roidb.rois(i*2) = attach_proposals(voc_rec, regions.boxes{i}, imdb.class_to_id, with_hard_samples, true);
      end
  end

  rmpath(fullfile(VOCopts.datadir, 'VOCcode')); 

  fprintf('Saving roidb to cache...');
  save(cache_file, 'roidb', '-v7.3');
  fprintf('done\n');
end


% ------------------------------------------------------------------------
function rec = attach_proposals(voc_rec, boxes, class_to_id, with_hard_samples, flip)
% ------------------------------------------------------------------------

% change selective search order from [y1 x1 y2 x2] to [x1 y1 x2 y2]
boxes = boxes(:, [2 1 4 3]);
if flip
    boxes(:, [1, 3]) = voc_rec.imgsize(1) + 1 - boxes(:, [3, 1]);
end

%           gt: [2108x1 double]
%      overlap: [2108x20 single]
%      dataset: 'voc_2007_trainval'
%        boxes: [2108x4 single]
%         feat: [2108x9216 single]
%        class: [2108x1 uint8]
if isfield(voc_rec, 'objects')
  if with_hard_samples
      valid_objects = ~cat(1, voc_rec.objects(:).difficult);
  else
      valid_objects = 1:length(voc_rec.objects(:));
  end
  gt_boxes = cat(1, voc_rec.objects(valid_objects).bbox);
  if flip
    gt_boxes(:, [1, 3]) = voc_rec.imgsize(1) + 1 - gt_boxes(:, [3, 1]);
  end
  all_boxes = cat(1, gt_boxes, boxes);
  gt_classes = class_to_id.values({voc_rec.objects(valid_objects).class});
  gt_classes = cat(1, gt_classes{:});
  num_gt_boxes = size(gt_boxes, 1);
else
  gt_boxes = [];
  all_boxes = boxes;
  gt_classes = [];
  num_gt_boxes = 0;
end
num_boxes = size(boxes, 1);

rec.gt = cat(1, true(num_gt_boxes, 1), false(num_boxes, 1));
rec.overlap = zeros(num_gt_boxes+num_boxes, class_to_id.Count, 'single');
for i = 1:num_gt_boxes
  rec.overlap(:, gt_classes(i)) = ...
      max(rec.overlap(:, gt_classes(i)), boxoverlap(all_boxes, gt_boxes(i, :)));
end
rec.boxes = single(all_boxes);
rec.feat = [];
rec.class = uint8(cat(1, gt_classes, zeros(num_boxes, 1)));

% ------------------------------------------------------------------------
function regions = load_proposals(proposal_file, regions)
% ------------------------------------------------------------------------
if isempty(regions)
    regions = load(proposal_file);
else
    regions_more = load(proposal_file);
    if ~all(cellfun(@(x, y) strcmp(x, y), regions.images(:), regions_more.images(:), 'UniformOutput', true))
        error('roidb_from_ilsvrc: %s is has different images list with other proposals.\n', proposal_file);
    end
    regions.boxes = cellfun(@(x, y) [double(x); double(y)], regions.boxes(:), regions_more.boxes(:), 'UniformOutput', false);
end
