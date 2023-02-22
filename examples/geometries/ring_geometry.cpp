
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas;
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.set(0, 5, 10);

    OrbitControls controls{camera, canvas};

    const auto material = MeshBasicMaterial::create();
    material->color.setHex(0xff0000);
    material->transparent = true;
    material->opacity = 0.5f;
    material->side = DoubleSide;

    for (int i = 1; i < 5; i++) {
        const auto geometry = RingGeometry::create(0.5f * i, 1 * i, 16, 8);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.y = -i;
        mesh->rotateX(-math::PI / 2);
        scene->add(mesh);
    }

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&]() {
        renderer.render(scene, camera);
    });
}
