**Reconstruction and Deformation Outputs**

| Path | Description
| :--- | :----------
| dirname_data | Directory of all inputs
| &ensp;&ensp;&boxur;&nbsp; XXX | Directory of one input pair
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; XXX.mtl | Mtl file to link the texture file to the mesh
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; XXX_deform.txt | \*Landmarks to input to the deformation program as target
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; XXX\_face.jpg | Reconstructed texture
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; XXX\_face\_fit.obj | \*Reconstructed mesh
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; XXX\_face\_fit\_ortho.obj | \*Reconstructed mesh with projection and translation matrix already applied for easier rendering
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; XXX\_face\_fit\_ortho\_deform.obj | Deformed mesh
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; XXX\_feature.jpg | \*Visualization of the detected landmarks for reconstruction
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; XXX\_feature.txt | \*Detected landmarks for reconstruction
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; XXX\_fit.jpg | \*Visualization of the reconstruction result
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; XXX\_photo.jpg | \*A copy of the input image
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; XXX\_proj.txt | \*Projection matrix parameters
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; XXX\ _trans.txt | \*Translation matrix parameters
| &ensp;&ensp;&boxur;&nbsp; YYY | Directory of one input pair
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; ... | ...

*not necessary for following steps