
cur_dir = pwd;
cd(fileparts(mfilename('fullpath')));

fprintf('Downloading Selective Search IJCV code...\n');
urlwrite('http://huppelen.nl/publications/SelectiveSearchCodeIJCV.zip', 'SelectiveSearchCodeIJCV.zip');

fprintf('Unzipping...\n');
unzip('SelectiveSearchCodeIJCV.zip');

fprintf('Done.\n');
system('del SelectiveSearchCodeIJCV.zip');

cd(cur_dir);