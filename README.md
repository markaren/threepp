## threepp (Work in progress)

Cross-platform C++17 port of the popular Javascript 3D library [three.js](https://github.com/mrdoob/three.js/) [r129](https://github.com/mrdoob/three.js/tree/r129).


#### Current state of the project

Most of the core library has been ported, including basic rendering capabilities, 
however much remains to be done..

##### What works?

* Box, Sphere, Plane, Cylindrical and Tube geometries  
* 2D Textures
* Transparency
* OrbitControls
* AmbientLight, DirectionalLight, PointLight  
* Most materials
* RenderTarget
* Raycasting against Mesh
* Binary STL loader
* OBJ/MTL Loader


### But, but why?

Because fun. 


### How to build

`threepp` can be used in conjunction with both `vcpkg` and `conan`.

#### vcpkg (using manifest mode)

Call CMake with `-DCMAKE_TOOLCHAIN_FILE=[path to vcpkg]/scripts/buildsystems/vcpkg.cmake`

Add features by listing them with `-VCPKG_MANIFEST_FEATURES=feature1;feature2`

#### conan

Run something akin to:
`conan install . -s build_type=Debug -if cmake-build-debug -b missing` _before_ calling CMake.
You might also use the supplied `run_conan_XXX.sh` scripts.


### Installing

`threepp` is available through the conan remote ais:

`conan remote add ais https://ais.jfrog.io/artifactory/api/conan/ais-conan-local`

Then add a dependency to:
```bash
threepp/<version>@ais/stable # (stable channel -> releases) </br>
threepp/<version>@ais/testing # (development builds -> master) </br>
threepp/<version>@ais/testing-<branch> # (development builds -> branches)
```

### Implementation notes

In general, you'll find that math classes are value types, while `threepp` expect smart pointers for other types. 
For convenience, geometries, materials etc. has a static `::create` function that returns a `std::shared_ptr`.
There should never be a need to handle memory explicitly using `threepp`. Yay!

### Example

```cpp
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 100);
    camera->position.z = 5;

    GLRenderer renderer(canvas);
    renderer.setSize(canvas.getSize());

    OrbitControls controls{camera, canvas};

    auto light = AmbientLight::create(Color(0xffffff).multiplyScalar(0.75f));
    scene->add(light);

    auto group = Group::create();

    {
        const auto boxGeometry = BoxGeometry::create();
        const auto boxMaterial = MeshPhongMaterial::create();
        boxMaterial->color.setHex(0xff0000);
        auto box = Mesh::create(boxGeometry, boxMaterial);
        box->position.setX(-1);
        group->add(box);
    }

    {
        const auto boxGeometry = BoxGeometry::create();
        const auto boxMaterial = MeshPhongMaterial::create();
        boxMaterial->color.setHex(0x00ff00);
        auto box = Mesh::create(boxGeometry, boxMaterial);
        box->position.setX(1);
        group->add(box);
    }

    scene->add(group);

    const auto planeGeometry = PlaneGeometry::create(5, 5);
    const auto planeMaterial = MeshLambertMaterial::create();
    planeMaterial->color.setHex(Color::gray);
    planeMaterial->side = DoubleSide;
    auto plane = Mesh::create(planeGeometry, planeMaterial);
    plane->position.setY(-1);
    plane->rotateX(math::degToRad(90));
    scene->add(plane);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    group->rotation.setOrder(Euler::YZX);
    canvas.animate([&](float dt) {
        group->rotation.y += 0.5f * dt;

        renderer.render(scene, camera);
    });
}

```
