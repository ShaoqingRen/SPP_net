function spp_cache_convX_features(imdb, varargin)
% spp_cache_convX_features(imdb, varargin)
%   Computes last conv features and saves them to disk. 
%
%   Keys that can be passed in:
%
%   imdb_copy_from    An imdb with partial scales cache which can copy from
%   start             Index of the first image in imdb to process
%   end               Index of the last image in imdb to process
%   spm_im_size       Scales of image resize to
%   spp_params_def    spp pooling def file, only for adaptive scale
%   roidb             only for adaptive scale
%   net_def_file      Path to the Caffe CNN to use
%   net_file          Path to the Caffe CNN to use
%   cache             Path to the precomputed feature cache
%   cache_copy_from   Path to the precomputed feature cache to copy from
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

ip = inputParser;
ip.addRequired('imdb', @isstruct);
ip.addOptional('imdb_copy_from', struct(), @isstruct);
ip.addOptional('start', 1, @isscalar);
ip.addOptional('end', 0, @isscalar);
ip.addOptional('spm_im_size', 688, @ismatrix);
ip.addOptional('spp_params_def', '', @isstr);
ip.addOptional('roidb', '', @isstruct);
ip.addOptional('net_def_file', '', @isstr);
ip.addOptional('net_file', '', @isstr);
ip.addOptional('cache', 'feat_cache', @isstr);
ip.addOptional('cache_copy_from', 'feat_cache', @isstr);

ip.parse(imdb, varargin{:});
opts = ip.Results;

% load caffe model
spp_load_model(spp_create_model(opts.net_def_file, opts.net_file, opts.spp_params_def));

% Where to save feature cache
opts.output_dir = ['./feat_cache/' opts.cache '/' imdb.name '/'];
copy_from_previous_cache = false;
if ~isempty(opts.imdb_copy_from) && isfield(opts.imdb_copy_from, 'name');
    copy_from_previous_cache = true;
    opts.output_dir_copy_from = ['./feat_cache/' opts.cache_copy_from '/' opts.imdb_copy_from.name '/'];
end
mkdir_if_missing(opts.output_dir);
conf = spp_config();

% Log feature extraction
timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
diary_file = [opts.output_dir 'rcnn_train_' opts.cache '_'  timestamp '.txt'];
diary(diary_file);
fprintf('Logging output in %s\n', diary_file);

fprintf('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Feature caching options:\n');
disp(opts);
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');

image_ids = imdb.image_ids;
if opts.end == 0
  opts.end = length(image_ids);
end

total_time = 0;
count = 0;
for i = opts.start:opts.end
  fprintf('%s: cache features: %d/%d\n', procid(), i, opts.end);

  save_file = [opts.output_dir image_ids{i} '.mat'];
  if exist(save_file, 'file') ~= 0
    fprintf(' [already exists]\n');
    
    broken = false;
    if 1 % check all cache files
        try
            d = load(save_file);
            assert(isfield(d, 'feat'));
            assert(~isempty(d.feat));
            if ~isempty(opts.spm_im_size)
                assert(isequal(d.feat.scale, opts.spm_im_size));
                assert(length(d.feat.rsp) == length(opts.spm_im_size));
            end
        catch
            broken = true;
        end
    end
    
    if ~broken
        continue;
    end
  end
  mkdir_if_missing(fileparts(save_file));
  
  feat_cache = [];
  if copy_from_previous_cache
      load_file = [opts.output_dir_copy_from image_ids{i} '.mat'];
      if exist(load_file, 'file') == 0
         fprintf(' [missing]\n');   
      else
         cache = load(load_file); 
         feat_cache = cache.feat;
      end
  end

  tot_th = tic;
  count = count + 1;
  
  d.gt = []; d.overlap = []; d.boxes = []; d.class = []; d.feat = [];
  try
        im = imread(imdb.image_at(i));
        th = tic;

        d.feat = spp_features_convX(im, opts.spm_im_size, feat_cache, conf.use_gpu);
        
        fprintf(' [features: %.3fs]\n', toc(th));
  catch
        fprintf('Warrning: Cannot read %s.\n', imdb.image_at(i));

  end
        
  th = tic;
  save(save_file, '-struct', 'd');
  fprintf(' [saving:   %.3fs]\n', toc(th));

  total_time = total_time + toc(tot_th);
  fprintf(' [avg time: %.3fs (total: %.3fs)]\n', ...
      total_time/count, total_time);
end

