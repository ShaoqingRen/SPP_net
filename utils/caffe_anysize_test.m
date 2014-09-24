function [result] = caffe_anysize_test(function_handle, varargin)
% conv_anysize_test('init', param_file, model_file)
% conv_anysize_test('release')
% conv_anysize_test('set_input_size', height, width, channel, num)
% is_initialized = conv_anysize_test('is_initialized')
% response = conv_anysize_test('forward', img)

if strcmp(function_handle, 'forward')
    im = varargin{1};
    if size(im, 3) == 1
        im = im(:, :, [1, 1, 1]);
    end
    % convert to single
    im = single(im);
    % permute from RGB to BGR and subtract the data mean (already in BGR)
    data_mean = 128;
    im = im(:, :, [3 2 1]) - data_mean;
    % flip width and height to make width the fastest dimension, height
    % second
    im = permute(im, [2 1 3]);
    varargin{1} = im;
    
    [result] = caffe('forward', varargin);
    
    % result is in (width, height, channel), need trans to (height, width,
    % channel)
    for i = 1:length(result)
        result{i} = permute(result{i}, [2, 1, 3]);
    end
end

if strcmp(function_handle, 'set_input_size')
    varargin = cellfun(@(x) double(x), varargin, 'UniformOutput', false);
    caffe('set_input_size', varargin{:});
end

if strcmp(function_handle, 'init')
    caffe('init', varargin{:});
end

if strcmp(function_handle, 'is_initialized')
    result = caffe('is_initialized');
end

if strcmp(function_handle, 'release')
    caffe('release');
end


