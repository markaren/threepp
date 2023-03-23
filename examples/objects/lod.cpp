
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

    auto lod = LOD::create();
    scene->add(lod);

    float radius = 0.5;
    auto material = MeshBasicMaterial::create({{"wireframe", true}});
    for (int z = 0; z <= 5; z++) {
        int detail = 6-z;
        auto obj = Mesh::create(IcosahedronGeometry::create(radius, detail), material);
        lod->addLevel(obj, static_cast<float>(z));
    }

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    renderer.enableTextRendering();
    auto& handle = renderer.textHandle();

    canvas.animate([&](float dt) {

        handle.setText("LOD level: " + std::to_string(lod->getCurrentLevel()));

        renderer.render(scene, camera);
    });
}
