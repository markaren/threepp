
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.z = 3;

    GLRenderer renderer(canvas);
    renderer.setSize(canvas.getSize());

    OrbitControls controls{camera, canvas};

    const auto geometry = SphereGeometry::create(0.5f);
    const auto material = MeshBasicMaterial::create();
    material->color.setHex(0xff0000);
    material->wireframe = true;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    canvas.animate([&]() {
        renderer.render(scene, camera);
    });
}
