
cur_dir = pwd;
cd(fileparts(mfilename('fullpath')));

fprintf('Downloading SPP_net_release1_data_caffe_mex_cuda5.5...\n');
urlwrite('https://onedrive.live.com/download?resid=4006CBB8476FF777!9458&authkey=!ANmLYSvdEg9OS1k&ithint=file%2czip', ...
    'SPP_net_release1_data_caffe_mex_cuda5.5.zip');

fprintf('Unzipping...\n');
unzip('SPP_net_release1_data_caffe_mex_cuda5.5.zip', '.');

fprintf('Done.\n');
system('del SPP_net_release1_data_caffe_mex_cuda5.5.zip');

cd(cur_dir);