curdir = fileparts(mfilename('fullpath'));
addpath(fullfile(curdir, 'selective_search'));
if exist(fullfile(curdir,'selective_search/SelectiveSearchCodeIJCV'), 'dir')
  addpath(fullfile(curdir,'selective_search/SelectiveSearchCodeIJCV'));
  addpath(fullfile(curdir,'selective_search/SelectiveSearchCodeIJCV/Dependencies'));
else
  fprintf('Warning: you will need the selective search IJCV code.\n');
  fprintf('Press any key to download it (runs ./selective_search/fetch_selective_search.sh)> ');
  pause;
  if ispc
      fetch_selective_search();
  else
      system('./selective_search/fetch_selective_search.sh');
  end
  addpath(fullfile(curdir,'selective_search/SelectiveSearchCodeIJCV'));
  addpath(fullfile(curdir,'selective_search/SelectiveSearchCodeIJCV/Dependencies'));
end
addpath(fullfile(curdir,'vis'));
addpath(genpath(fullfile(curdir,'utils')));
addpath(fullfile(curdir,'bin'));
addpath(fullfile(curdir,'nms'));
addpath(fullfile(curdir,'finetuning'));
addpath(fullfile(curdir,'bbox_regression'));
if exist(fullfile(curdir,'external/caffe/matlab/caffe'), 'dir')
  addpath(fullfile(curdir,'external/caffe/matlab/caffe'));
else
  warning('Please install Caffe in ./external/caffe');
end
addpath(fullfile(curdir,'experiments'));
addpath(fullfile(curdir,'imdb'));
fprintf('R-CNN startup done\n');
