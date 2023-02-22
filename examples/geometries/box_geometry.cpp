
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas;
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.z = 3;

    OrbitControls controls{camera, canvas};

    const auto geometry = BoxGeometry::create(0.5f, 0.5f, 0.5f);
    const auto material = MeshBasicMaterial::create();
    material->color.setHex(0xff0000);
    material->wireframe = true;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&]() {
        renderer.render(scene, camera);
    });
}
