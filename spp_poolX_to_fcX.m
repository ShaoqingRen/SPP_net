function feat = spp_poolX_to_fcX(feat, layer, spp_model, use_gpu)
% feat = spp_poolX_to_fcX(feat, layer, spp_model, use_gpu)
%   On-the-fly conversion of last pool features to fcX
%   using the weights and biases stored in spp_model.cnn.layers.
%   feat is transformed in columns for continuous memory access and fast
%   speed
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

fc_layer_ind = layer - (spp_model.cnn.first_fc_idx - 1);

if fc_layer_ind <= 0
    return;
end

if use_gpu
    maxNumel = 1024*1024*256; % around 4G memory
    if numel(feat) < maxNumel 
        feat_gpu = gpuArray(feat);
        for i = 1:fc_layer_ind
            % weights{1} = matrix of CNN weights [input_dim x output_dim]
            % weights{2} = column vector of biases
            feat_gpu = max(0, bsxfun(@plus, spp_model.cnn.layers(i).weights_gpu{1} * feat_gpu, ...
                              spp_model.cnn.layers(i).weights_gpu{2}));
        end
        feat = gather(feat_gpu);
    else
        nSampleEach = floor(maxNumel / size(feat, 1));
        nSplits = ceil(size(feat, 2) / nSampleEach);
        splits = ones(nSplits, 1) * nSampleEach;
        splits(end) = size(feat, 2) - sum(splits(1:end-1));
        assert(sum(splits) == size(feat, 2));
        feats = mat2cell(feat, size(feat, 1), splits);
        for is = 1:length(feats)
            feat_gpu = gpuArray(feats{is});
            for i = 1:fc_layer_ind
                % weights{1} = matrix of CNN weights [input_dim x output_dim]
                % weights{2} = column vector of biases
                feat_gpu = max(0, bsxfun(@plus, spp_model.cnn.layers(i).weights_gpu{1} * feat_gpu, ...
                                  spp_model.cnn.layers(i).weights_gpu{2}));
            end
            feats{is} = gather(feat_gpu);
        end
        feat = cell2mat(feats);
    end

else
    for i = 1:fc_layer_ind
        % weights{1} = matrix of CNN weights [input_dim x output_dim]
        % weights{2} = column vector of biases
        feat = max(0, bsxfun(@plus, spp_model.cnn.layers(i).weights{1} * feat, ...
                              spp_model.cnn.layers(i).weights{2}));
    end    
end