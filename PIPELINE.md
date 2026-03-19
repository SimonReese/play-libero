# Spatial reference generation pipeline


Key idea
1) Extract obj position in world
2) Extract robot base position in word
3) Compute spatial relations in evironment wrt robot, camera, relative -> can we use robospatial for this?
4) Map a 

# Steps

1) Find datasets with: external + wrist cameras, 3D Observations, Camera extrinsics, Language annotations

2) Filter out useless tasks: we want to insert a spatial annotation, therefore we should focus on environments where a spatial annotation can be added (could be left of another recognizable object)

3) Grounded-SAM: Pass initial image to grounded sam, and generate labels and bbox of objects in the image

4) Code: Using 3d obs, camera calib and bboxes, generate a 3dbbox of the object in the space wrt a particular refenrece frame

5) Robospatial: generate annotations from the 3dbboxes

6) Remap task instructions: insert robospatial configuration elements such that the outpur result is the same


## 1) Extract object position

We can use camera extrinsic/intrinsic.
We can ask grounded sam to segment objects in the image,  bbox positions, get corresponding depth/pc position, map image point to 3d position, produce a 3d bbox scaling 2d bbox size

