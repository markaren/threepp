## threepp (Work in progress)

C++ port of the popular Javascript 3D library [three.js](https://github.com/mrdoob/three.js/) [r129](https://github.com/mrdoob/three.js/tree/r129).


#### Current state of the project

Most of the core library has been ported, including basic rendering capabilities, 
however much remains to be done..

##### What works?

* Box, Sphere, Plane and Cylindrical geometries  
* 2D Textures
* Transparency
* OrbitControls
* AmbientLight, DirectionalLight  
* Most materials
* Raycasting against Mesh


### But, but why?

This is mostly a personal exercise. Don't expect much support, although contributions are welcome. 


### How to build

In order to successfully build threepp, you'll need [conan](https://conan.io/).

`pip install conan`

With conan installed, invoke `run_conan_install.sh`.

_note that this command is hardcoded to use the default CLion build folders (cmake-build-\<target>)_

You can now build the project as a regular CMake project using e.g. the command line.