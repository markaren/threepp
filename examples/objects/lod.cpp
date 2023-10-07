
#include "threepp/objects/LOD.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas("LOD");
    GLRenderer renderer(canvas.size());

    Scene scene;
    PerspectiveCamera camera(60, canvas.aspect(), 0.1f, 10);
    camera.position.z = -5;

    OrbitControls controls{camera, canvas};

    LOD lod;
    scene.add(lod);

    float radius = 0.5;
    auto material = MeshBasicMaterial::create({{"wireframe", true}});
    for (int z = 0; z <= 5; z++) {
        int detail = 6 - z;
        auto obj = Mesh::create(IcosahedronGeometry::create(radius, detail), material);
        lod.addLevel(obj, static_cast<float>(z));
    }

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    TextRenderer textRenderer;
    auto& handle = textRenderer.createHandle();

    canvas.animate([&]() {
        handle.setText("LOD level: " + std::to_string(lod.getCurrentLevel()));

        renderer.render(scene, camera);
        renderer.resetState();
        textRenderer.render();
    });
}
