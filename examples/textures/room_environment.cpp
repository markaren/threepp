// Image-based lighting without an HDR file: a RoomEnvironment (a neutral
// studio room) rendered into an environment map with PMREMGenerator, then used
// as scene.environment. There are no lights in the scene at all.
//
// Top row: metalness 0 (plastic), bottom row: metalness 1 (metal); roughness
// goes from 0 on the left to 1 on the right.
//
// GLRenderer only (PMREMGenerator renders through it).

#include "threepp/threepp.hpp"

#include "threepp/extras/environments/RoomEnvironment.hpp"
#include "threepp/renderers/gl/PMREMGenerator.hpp"

using namespace threepp;

int main() {

    Canvas canvas("Room environment");
    GLRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;

    auto scene = Scene::create();
    scene->background = Color(0x444444);

    // The three.js idiom:
    //   scene.environment = pmremGenerator.fromScene( new RoomEnvironment(), 0.04 ).texture;
    {
        PMREMGenerator pmrem(renderer);
        RoomEnvironment room;
        scene->environment = pmrem.fromScene(room, 0.04f);
        room.dispose();
    }

    constexpr int columns = 6;
    auto geometry = SphereGeometry::create(0.4f, 64, 32);
    for (int row = 0; row < 2; ++row) {
        for (int i = 0; i < columns; ++i) {
            auto material = MeshStandardMaterial::create();
            material->color = row == 0 ? Color(0xcc3333) : Color(0xdddddd);
            material->metalness = row == 0 ? 0.f : 1.f;
            material->roughness = static_cast<float>(i) / (columns - 1);

            auto sphere = Mesh::create(geometry, material);
            sphere->position.set((static_cast<float>(i) - (columns - 1) / 2.f) * 1.f, row == 0 ? 0.55f : -0.55f, 0);
            scene->add(sphere);
        }
    }

    PerspectiveCamera camera(40, canvas.aspect(), 0.1f, 100);
    camera.position.set(0, 0, 8);

    OrbitControls controls{camera, canvas};

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&] {
        renderer.render(*scene, camera);
    });
}
