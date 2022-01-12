# 3D-Portrait-Stylization

This is the official code for the paper "Exemplar Based 3D Portrait Stylization".

The entire framework consists of four parts, landmark translation, face reconstruction, face deformation, and texture stylization. Codes (or programs) for the last three parts are ready now, and the first part is still under preparation. 

## Landmark Translation

Code under preparation. Dataset can be downloaded [here](https://portland-my.sharepoint.com/:u:/g/personal/fangzhhan2-c_my_cityu_edu_hk/EXdhOdnthWZJgjXxand3E64B9rM-NJUj3iHcoeh_G_sDzw?e=Ar0cnE).

## Face Reconstruction and Deformation

**Environment**

These two parts require Windows with GPU. They also require a simple Python environment with `opencv`, `imageio` and `numpy` for automatic batch file generation and execution.  Python code in the two parts is tested using Pycharm, instead of command lines.

Please download the [`regressor_large.bin`](https://portland-my.sharepoint.com/:u:/g/personal/fangzhhan2-c_my_cityu_edu_hk/EXdQXynEBkdHhvHLZLn1qh0BUmIxR_K5Mhp2fQKel95okQ?e=nVD9r6) and [`tensorMale.bin`](https://portland-my.sharepoint.com/:u:/g/personal/fangzhhan2-c_my_cityu_edu_hk/Ec3NvbEJ2-FNnlVmLMbzskMBnqd1Hs7X_Hxo527AM2r1sw?e=zbWH2l) and put them in `./face_recon_deform/PhotoAvatarLib_exe/Data/`.

**Inputs**

These two parts require inputs in the format given below.

| Path | Description
| :--- | :----------
| dirname_data | Directory of all inputs
| &ensp;&ensp;&boxur;&nbsp; XXX | Directory of one input pair
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; XXX.jpg | Content image
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; XXX.txt | Landmarks of the content image
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; XXX_style.jpg | Style image
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; XXX_style.txt | Landmarks of the style image
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; XXX_translated.txt | Translated landmarks
| &ensp;&ensp;&boxur;&nbsp; YYY | Directory of one input pair
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; ... | ...

Some examples are given in `./data_demo/`. As the code for translation has not been provided, you may use [The Face of Art](https://faculty.idc.ac.il/arik/site/foa/face-of-art.asp) to obtain some results for now.

**Uasge**

Directly run `main_recon_deform.py` is OK, and you can also check the usage from the code. 

In `./face_recon_deform/PhotoAvatarLib_exe/` is a compiled reconstruction program which takes one single image as input, automatically detects the landmarks and fits a 3DMM model towards the detected landmarks. The source code can be downloaded [here](https://portland-my.sharepoint.com/:u:/g/personal/fangzhhan2-c_my_cityu_edu_hk/Ee0QVlheafhCsW3GygBJyawBhZIWpouaK6P0wJygVLg7LQ?e=PTQvVv). 

In `./face_recon_deform/LaplacianDeformerConsole/` is a compiled deformation program which deforms a 3D mesh towards a set of 2D/3D landmark targets. You can find the explanation of the parameters by runing `LaplacianDeformerConsole.exe` without adding options. Please note that it only supports one mesh topology and cannot be used for deforming random meshes. The source code is not able to provide, and some other Laplacian or Laplacian-like deformations can be found in [SoftRas](https://github.com/ShichenLiu/SoftRas) and [libigl](https://libigl.github.io/libigl-python-bindings/tut-chapter3/#biharmonic-deformation).

**Outputs**

Please refer to `./face_recon_deform/readme_output.md`

## Texture Stylization

**Environment**

The environment for this part is built with CUDA 10.0, python 3.7, and PyTorch 1.2.0, using Conda. Create environment by:

```
conda create -n YOUR_ENV_NAME python=3.7
conda activate YOUR_ENV_NAME
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
conda install scikit-image tqdm opencv
```

The code uses neural-renderer, which is already compiled. However, if anything go wrong (perhaps because of the environment difference), you can re-compile it by

```
python setup.py install
mv build/lib.linux-x86_64-3.7-or-something-similar/neural_renderer/cuda/*.so neural_renderer/cuda/
```

Please download [`vgg19_conv.pth`](https://portland-my.sharepoint.com/:u:/g/personal/fangzhhan2-c_my_cityu_edu_hk/EbK8vzgtULNHqhHy93WCHlQBoqHKsCyjAJVyKg0BJFS2_A?e=cNjHMZ) and put it in `./texture_style_transfer/transfer/models/`.

**Inputs**

You can directly use the outputs (and inputs) from the previous parts.

**Usage**

```
cd texture_style_transfer
python transfer/main_texture_transfer.py -dd ../data_demo_or_your_data_dir
```

## Acknowledgements

This code is built based heavliy on [Neural 3D Mesh Renderer](https://github.com/daniilidis-group/neural_renderer) and [STROTSS](https://github.com/human-aimachine-art/pytorch-STROTSS-improved).

## Citation

```
@ARTICLE{9547845,
author={Han, Fangzhou and Ye, Shuquan and He, Mingming and Chai, Menglei and Liao, Jing},  
journal={IEEE Transactions on Visualization and Computer Graphics},   
title={Exemplar-Based 3D Portrait Stylization},   
year={2021},  
doi={10.1109/TVCG.2021.3114308}}
```
