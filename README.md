## threepp (Work in progress)

Cross-platform C++17 port of the popular Javascript 3D library [three.js](https://github.com/mrdoob/three.js/) [r129](https://github.com/mrdoob/three.js/tree/r129).


#### Current state of the project

Most of the core library has been ported, including advanced rendering capabilities, 
however much remains to be done..

##### What works?

* Line, Points, Mesh, InstancedMesh
* Geometries [Box, Sphere, Plane, Cylindrical, Capsule, Tube, ++]  
* Lights [Ambient, Directional, Point, Spot, Hemi]
* Raycasting [Mesh, Line, Points]
* 2D/3D Textures, 3D text, Sprites, RenderTarget, CubeMaps
* Transparency, Shadows
* Morphtargets, Bones
* Controls [Orbit, Fly, Drag]
* Water and Sky shaders
* Built-in text rendering and font loading [typeface.json, TTF]
* Loaders [Binary STL, OBJ/MTL, SVG]
* Basic Audio support using [miniaudio](https://miniaud.io/docs/manual/index.html)
* Generic model loader based on [Assimp](https://github.com/assimp/assimp)
* Easy integration with [Dear ImGui](https://github.com/ocornut/imgui)

Builds on Windows, Linux, MacOS, MinGW and with Emscripten.

### But, but why?

Because fun. 


### How to build

`threepp` comes bundled with all required core dependencies. 

Use CMake for project configuration and building.

###### Windows
```shell
cmake . -A x64 -B build -DCMAKE_BUILD_TYPE="Release"
cmake --build build --config "Release"
```

###### Unix
```shell
cmake . -B build -DCMAKE_BUILD_TYPE="Release"
cmake --build build
```

However, some of the examples (and headers) require additional dependencies. 
To make use of all features and to enable/build all examples, the use of [vcpkg](https://vcpkg.io/en/index.html) is encouraged.

#### vcpkg (using manifest mode)

Call CMake with `-DCMAKE_TOOLCHAIN_FILE=[path to vcpkg]/scripts/buildsystems/vcpkg.cmake`

Add optional features by listing them with `-DVCPKG_MANIFEST_FEATURES=feature1;feature2`

See [vcpkg.json](vcpkg.json) for available features.

Note, however, that under MinGW you'll need to specify the vcpkg triplet:
```shell
-DVCPKG_TARGET_TRIPLET=x64-mingw-[static|dynamic]  # choose either `static` or `dynamic`.
-DVCPKG_HOST_TRIPLET=x64-mingw-[static|dynamic]    # <-- needed only if MSVC cannot be found. 
```

##### Building examples with Emscripten

Pass to CMake:
```shell
-DCMAKE_TOOLCHAIN_FILE="[path to emscripten]\emsdk\upstream\emscripten\cmake\Modules\Platform\Emscripten.cmake"
```
When using vcpkg, however, do:
```shell
-DVCPKG_CHAINLOAD_TOOLCHAIN_FILE="[path to emscripten]\emsdk\upstream\emscripten\cmake\Modules\Platform\Emscripten.cmake"
```
This will generate .html versions of a subset of the examples to be loaded in a browser.


##### Optional downstream dependencies

When consuming `threepp` in your own application, 
some headers will require additional dependencies in order to compile.

| **Header**   | **Dependency** | **Description**                               |
|--------------|----------------|-----------------------------------------------|
| AssimpLoader | assimp         | Import a wide variety of different 3D formats |
| ImguiContext | imgui          | ImGUI utility                                 |


### Implementation notes

In general, you'll find that math classes are value types, while `threepp` expect smart pointers for other types. 
For convenience, geometries, materials etc. has a static `::create` function that returns a `std::shared_ptr`.
Thus, you don't necessarily need to handle memory explicitly using `threepp`.
Furthermore, materials, geometries and textures are automatically disposed when they go out of scope.
Yay!

### Example

```cpp
#include "threepp/threepp.hpp"

using namespace threepp;

auto createBox(const Vector3& pos, const Color& color) {
    auto geometry = BoxGeometry::create();
    auto material = MeshPhongMaterial::create();
    material->color = color;
    
    auto box = Mesh::create(geometry, material);
    box->position.copy(pos);
    
    return box;
}

auto createPlane() {
    auto planeGeometry = PlaneGeometry::create(5, 5);
    auto planeMaterial = MeshLambertMaterial::create();
    planeMaterial->color = Color::gray;
    planeMaterial->side = Side::Double;
    
    auto plane = Mesh::create(planeGeometry, planeMaterial);
    plane->position.y = -1;
    plane->rotateX(math::degToRad(90));
    
    return plane;
}

int main() {

    Canvas canvas("Demo");
    GLRenderer renderer{canvas.size()};

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 100);
    camera->position.z = 5;
    
    OrbitControls controls{*camera, canvas};

    auto light = HemisphereLight::create();
    scene->add(light);

    auto plane = createPlane();
    scene->add(plane);
    
    auto group = Group::create();
    group->add(createBox({-1, 0, 0}, Color::green));
    group->add(createBox({1, 0, 0}, Color::red));
    scene->add(group);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });
    
    Clock clock;
    canvas.animate([&]() {
        
        float dt = clock.getDelta();
        group->rotation.y += 1.f * dt;

        renderer.render(*scene, *camera);
    });
}

```

## Consuming threepp

Threepp is available as a CMake package and can be consumed in a number of ways.

#### CMake FetchContent (recommended)

`threepp` is compatible with CMake's `FetchContent`:

```cmake
include(FetchContent)
set(THREEPP_BUILD_TESTS OFF)
set(THREEPP_BUILD_EXAMPLES OFF)
FetchContent_Declare(
        threepp
        GIT_REPOSITORY https://github.com/markaren/threepp.git
        GIT_TAG tag_or_commit_hash
)
FetchContent_MakeAvailable(threepp)
#...
target_link_libraries(main PUBLIC threepp::threepp)
```

This is the preferred approach, as it enables users to update the targeted threepp version at will.

An example is provided [here](tests/threepp_fetchcontent_test).
See also [this demo](https://github.com/markaren/threepp_wxwidgets), which additionally uses WxWidgets as the Window system.

#### vcpkg

`threepp` is available for use with vcpkg through a custom vcpkg-registry.
However, this vcpkg port is cumbersome to maintain and is not so often updated, and support may be discontinued in the future.

An example is provided [here](tests/threepp_vcpkg_test).

### Screenshots
![Fonts](doc/screenshots/fonts.png)
![LeePerrySmith](doc/screenshots/LeePerrySmith.png)
![Shadows](doc/screenshots/Shadows.PNG)
![FlyControls](doc/screenshots/fly.PNG)
![Crane](doc/screenshots/crane.png)
![Physics](doc/screenshots/instanced_physics.PNG)
![Water](doc/screenshots/OlympicOctopus.PNG)
![MotorController](doc/screenshots/motor_controller.PNG)
