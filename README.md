SPP_net: spatial pyramid pooling in deep convolutional networks for visual recognition
========================


Acknowledgements: a huge thanks to Yangqing Jia for creating Caffe and the BVLC team, and to Ross Girshick for creating RCNN

### Introduction

This is a re-implementation of the object detection algorithm described in the ECCV 2014 paper “Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition”. This re-implementation should reproduce the object detection results reported in the paper up to some statistical variance. The models used in the paper are trained/fine-tuned using cuda-convnet, while the model attached with this code is trained/fine-tuned using Caffe, for the ease of code release.

The implementation of image classification training/testing has not been included, but the network configuration files can be found directly in this code.

Please contact sqren@mail.ustc.edu.cn or kahe@microsoft.com if you have any question.

### Citing SPP_net

If you find SPP_net useful in your research, please consider citing:

    @inproceedings{kaiming14ECCV,
        Author = {Kaiming, He and Xiangyu, Zhang and Shaoqing, Ren and Jian Sun},
        Title = {Spatial pyramid pooling in deep convolutional networks for visual recognition},
        Booktitle = {European Conference on Computer Vision},
        Year = {2014}
    }

### License

SPP_net is released under the Simplified BSD License for non-commercial use (refer to the LICENSE file for details).

### Installing SPP_net

0. **Prerequisites**
  0. MATLAB (tested with 2014a on 64-bit Windows)
  0. Caffe's prerequisites (some function is based our modified caffe, so we provied compiled caffe mex and cpp file for mex wapper), run `external\fetch_caffe_mex_5_5.m` to download
  1. News: a caffe version which supports spp mex is provided in https://github.com/ShaoqingRen/caffe, this version is forked from [BVLC/caffe](https://github.com/BVLC/caffe) on Oct. 1, 2014. The Zeiler CNN network with this new version is shared in [OneDrive](https://onedrive.live.com/download?resid=4006CBB8476FF777!9723&authkey=!APTWXLD_P7UN6P0&ithint=file%2czip), also the new prototxt for finetune is updated in `./model-defs`
0. **Install SPP_net**
  0. Get the SPP_net source code by cloning the repository: `git clone https://github.com/ShaoqingRen/SPP_net.git`
  0. Now change into the SPP_net source code directory
  0. SPP_net expects to find Caffe in `external/caffe`
  0. Start MATLAB (make sure you're still in the `spp` directory): `matlab`
  0. You'll be prompted to download the [Selective Search](http://disi.unitn.it/~uijlings/MyHomepage/index.php#page=projects1) code, which we cannot redistribute. Afterwards, you should see the message `SPP_net startup done` followed by the MATLAB prompt `>>`.
  0. Run the build script: `>> spp_build()` (builds [liblinear](http://www.csie.ntu.edu.tw/~cjlin/liblinear/), [Selective Search](http://www.science.uva.nl/research/publications/2013/UijlingsIJCV2013/), spp_pool and nms). Don't worry if you see compiler warnings while building liblinear, this is normal on my system.
  0. Download the model package by run `external\fetch_model_data.m`
 
### Training your own SPP_net detector on PASCAL VOC

Let's use PASCAL VOC 2007 as an example. The basic pipeline is: 

    extract features to disk -> finetune -> train SVMs -> test
    
You'll need about 20GB of disk space free for the feature cache (which is stored in `feat_cache` by default. **It's best if the feature cache is on a fast, local disk.** 

An one click script is `experiments\Script_spp_voc.m`


