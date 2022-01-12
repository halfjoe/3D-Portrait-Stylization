# 3D-Portrait-Stylization

This is the official code for the paper "Exemplar Based 3D Portrait Stylization".

The entire framework consists of four parts, landmark translation, face reconstruction, face deformation, and texture stylization. Codes (or programs) for the last three parts are ready now, and the first part is still under preparation. 

## Landmark Translation

Code under preparation. Dataset will be uploaded very soon.

## Face Reconstruction and Deformation

**Environment**

These two parts require Windows with GPU. They also require a simple Python environment with `opencv`, `imageio` and `numpy` for automatic batch file generation and execution. 

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

In `./face_recon_deform/LaplacianDeformerConsole/` is a compiled deformation program which deforms a 3D mesh towards a set of 2D/3D landmark targets. You can find the explanation of the parameters by runing `LaplacianDeformerConsole.exe` without adding options. Please note that it only supports one mesh topology and cannot be used for deforming random meshes. The source code is not able to provide, and some other Laplacian or Laplacian-like deformation can be found in [SoftRas](https://github.com/ShichenLiu/SoftRas) and [libigl](https://libigl.github.io/libigl-python-bindings/tut-chapter3/#biharmonic-deformation).

**Outputs**

Please refer to `./face_recon_deform/readme_output.md`

## Texture Stylization

Will add very soon.
