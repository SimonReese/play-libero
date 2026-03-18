# Testing VLA on LIBERO Benchmark

This repository contains useful scripts to test LIBERO Benchmark and VLA Models.
Currently testing:

- OpenPI
- OpenVLA

## Issues, notes and other elements

- Flipped obs: according to robosuite [documentation](https://robosuite.ai/docs/modules/environments.html#observations),
images observations provided by obs dict are flipped, since those follow MuJoCo native implemenation (which in turns follows OpenGL buffer layout). 
Therefore, directly viewing or storing those images may lead to flipped results when the opencv convention in reference frame is used (see this [figure](https://amytabb.com/tips/tutorials/2019/06/28/OpenCV-to-OpenGL-tutorial-essentials/#figure3))
It seems possible to change this behaviour in robosuite directly, obtaining an image which should directly match the rendered 
viewpoint (see this [issue](https://github.com/ARISE-Initiative/robosuite/issues/56)), 
by setting this [variable](https://github.com/ARISE-Initiative/robosuite/blob/aaa8b9b214ce8e77e82926d677b4d61d55e577ab/robosuite/macros.py#L28), however both OpenPi and OpenVLA appears to flip both $x$ and $y$ axes, therefore the flipping is done esplicitly

