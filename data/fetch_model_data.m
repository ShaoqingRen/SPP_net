
cur_dir = pwd;
cd(fileparts(mfilename('fullpath')));

fprintf('Downloading SPP_net_release1_data...\n');
urlwrite('https://onedrive.live.com/download?resid=D7AF52BADBA8A4BC!115&authkey=!AHbTHCRKXaIV3m8&ithint=file%2czip', ...
    'SPP_net_release1_data.zip');

fprintf('Unzipping...\n');
unzip('SPP_net_release1_data.zip', '../');

fprintf('Done.\n');
system('del SPP_net_release1_data.zip');

cd(cur_dir);