# 3D-Portrait-Stylization

This is the official code for the paper "Exemplar Based 3D Portrait Stylization".

The entire framework consists of four parts, landmark translation, face reconstruction, face deformation, and texture stylization. Codes (or programs) for the last three parts are ready now, and the first part is still under preparation. 

## Landmark Translation

Code under preparation. Dataset will be uploaded very soon.

## Face Reconstruction and Deformation

**Environment**

These two parts require Windows with GPU. They also require a simple Python environment with `opencv`, `imageio` and `numpy` for automatic batch file generation and execution. 

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

Some examples are given in `./data_demo/`. As the code for translation has not been provided, you may use [this work] (https://faculty.idc.ac.il/arik/site/foa/face-of-art.asp) to obtain some results for now.
