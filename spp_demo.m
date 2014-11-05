function spp_demo()

spp_model_file = '.\data\spp_model\VOC2007\spp_model.mat';
if ~exist(spp_model_file, 'file')
  error('%s not exist ! \n', spp_model_file);
end
try
    load(spp_model_file);
catch err
    fprintf('load spp_model_file : %s\n', err.message);
end
caffe_net_file     = fullfile(pwd, 'data\cnn_model\Zeiler_conv5\Zeiler_conv5');
caffe_net_def_file = fullfile(pwd, 'data\cnn_model\Zeiler_conv5\Zeiler_spm_scale224_test_conv5.prototxt');

use_gpu = true;
if use_gpu
    clear mex;
    g = gpuDevice(1);
end

caffe('init', caffe_net_def_file, caffe_net_file);
caffe('set_phase_test');
if use_gpu
    spp_model.cnn.layers = spp_layers_in_gpu(spp_model.cnn.layers);
    caffe('set_mode_gpu');
else
    caffe('set_mode_cpu');
end

spm_im_size = [480 576 688 874 1200];
% spm_im_size = [ 688 ];

im = imread('.\datasets\VOCdevkit2007\VOC2007\JPEGImages\000015.jpg');

dets = spp_detect(im, spp_model, spm_im_size, use_gpu);

classes = spp_model.classes;
boxes = cell(length(classes), 1);
thres = -0.5;
for i = 1:length(boxes)
    I = dets{i}(:, 5) >= thres;
    boxes{i} = dets{i}(I, :);
end
showboxes_new(im, boxes, classes);

caffe('release');

if use_gpu
    reset(g);
end
