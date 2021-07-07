## threepp (Work in progress)

C++ port of the popular Javascript 3D library [three.js](https://github.com/mrdoob/three.js/) [r129](https://github.com/mrdoob/three.js/tree/r129).


#### Current state of the project

Most of the core library has been ported, including basic rendering capabilities, but some vital parts of the rendering pipeline has yet to be completed.
Thus, threepp is not yet in fully a functional state. 

### But, but why?

This is mostly a personal exercise. Don't expect any support, although contributions are welcome. 


### How to build

In order to successfully build threepp, you'll need [conan](https://conan.io/).

`pip install conan`

With conan installed, invoke `run_conan_install.sh`.

_note that this command is hardcoded to use the default CLion build folders (cmake-build-\<target>)_

You can now build the project as a regular CMake project using e.g. the command line.