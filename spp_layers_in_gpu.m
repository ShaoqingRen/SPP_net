function layers = spp_layers_in_gpu(layers)
% layers = spp_layers_in_gpu(layers)
%   create each layers weight array in gpu
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

    for i = 1:length(layers)
       if isempty(layers(i).weights)
          continue;
       end
       layers(i).weights_gpu = layers(i).weights;
       for j = 1:length(layers(i).weights)
            layers(i).weights_gpu{j} = gpuArray(layers(i).weights{j});
       end
    end
end