
#include "threepp/objects/LOD.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas;
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(60, canvas.getAspect(), 0.1f, 10);
    camera->position.z = -5;

    OrbitControls controls{camera, canvas};

    auto material = MeshBasicMaterial::create();
    auto geometry = BoxGeometry::create();

    auto mesh = InstancedMesh::create(geometry, material, 1);
    scene->add(mesh);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

//    renderer.enableTextRendering();
//    auto& handle = renderer.textHandle();

    canvas.animate([&](float dt) {

        renderer.render(scene, camera);
    });
}
