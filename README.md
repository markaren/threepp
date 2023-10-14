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
* 2D/3D Textures, 3D text, Sprites, RenderTarget
* Transparency, Shadows
* Morphtargets
* OrbitControls, FlyControls
* Water and Sky shaders
* Loaders [Binary STL, OBJ/MTL, SVG]
* Generic model loader based on [Assimp](https://github.com/assimp/assimp)
* Easy integration with [Dear ImGui](https://github.com/ocornut/imgui)
* Easy integration with [Bullet](https://github.com/bulletphysics/bullet3)
* Easy to use text rendering using [glText](https://github.com/vallentin/glText)

Builds on Windows, Linux, MacOS and MinGW.

### But, but why?

Because fun. 


### How to build

`threepp` is easiest built in conjunction with [vcpkg](https://vcpkg.io/en/index.html).

#### vcpkg (using manifest mode)

Call CMake with `-DCMAKE_TOOLCHAIN_FILE=[path to vcpkg]/scripts/buildsystems/vcpkg.cmake`

Add optional features by listing them with `-DVCPKG_MANIFEST_FEATURES=feature1;feature2`

See [vcpkg.json](vcpkg.json) for available features.

###### Building under MinGW

Under MinGW you'll need to specify the vcpkg triplet:
```shell
-DVCPKG_TARGET_TRIPLET=x64-mingw-[static|dynamic]  # choose either `static` or `dynamic`.
-DVCPKG_HOST_TRIPLET=x64-mingw-[static|dynamic]    # <-- needed only if MSVC cannot be found. 
```

##### Optional downstream dependencies

When consuming `threepp` in your own application, 
some headers will require additional dependencies in order to compile.

| **Header**     | **Dependency** | **Description**                               |
|----------------|----------------|-----------------------------------------------|
| AssimpLoader   | assimp         | Import a wide variety of different 3D formats |
| FontLoader     | nlohmann-json  | Import fonts to be used for 3D text           |
| SVGLoader      | pugixml        | Import SVG files                              |
| ImguiContext   | imgui          | ImGUI utility                                 |
| BulletPhysics  | bullet3        | Bullet utility                                |


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

int main() {

    Canvas canvas("Demo");
    GLRenderer renderer{canvas.size()};

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 100);
    camera->position.z = 5;
    
    OrbitControls controls{*camera, canvas};

    auto light = HemisphereLight::create();
    scene->add(light);

    auto group = Group::create();
    group->add(createBox({-1, 0, 0}, Color::green));
    group->add(createBox({1, 0, 0}, Color::red));
    scene->add(group);

    auto planeGeometry = PlaneGeometry::create(5, 5);
    auto planeMaterial = MeshLambertMaterial::create();
    planeMaterial->color = Color::gray;
    planeMaterial->side = Side::Double;
    auto plane = Mesh::create(planeGeometry, planeMaterial);
    plane->position.y = -1;
    plane->rotateX(math::degToRad(90));
    scene->add(plane);

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

## Consuming threepp using a custom vcpkg-registry

See [here](tests/threepp_vcpkg_test) for an example of how `threepp` may be consumed 
as a library in a separate project using `vcpkg`.

### Screenshots
![Fonts](doc/screenshots/fonts.png)
![LeePerrySmith](doc/screenshots/LeePerrySmith.png)
![Shadows](doc/screenshots/Shadows.PNG)
![Crane](doc/screenshots/crane.png)
![Physics](doc/screenshots/instanced_physics.PNG)
![Water](doc/screenshots/OlympicOctopus.PNG)
