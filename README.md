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

### Example

```cpp
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 100);
    camera->position.z = 5;

    auto renderer = GLRenderer(canvas);
    renderer.checkShaderErrors = true;
    renderer.setSize(canvas.getSize());

    OrbitControls controls{camera, canvas};

    auto directional = DirectionalLight::create(Color(0xffffff).multiplyScalar(0.75f));
    directional->position.set(100, 100, 100);
    directional->lookAt(0, 0, 0);
    scene->add(directional);

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

    group->rotation.order(Euler::RotationOrders::YZX);
    canvas.animate([&](float dt) {
        group->rotation.y(group->rotation.y() + 0.5f * dt);

        renderer.render(scene, camera);
    });
}

```