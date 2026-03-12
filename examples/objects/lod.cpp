
#include "threepp/objects/LOD.hpp"
#include "threepp/canvas/Monitor.hpp"
#include "threepp/objects/TextSprite.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas("LOD", {{"aa", 4}});
    GLRenderer renderer(canvas.size());
    renderer.autoClear = false;

    Scene scene;
    PerspectiveCamera camera(60, canvas.aspect(), 0.1f, 10);
    camera.position.z = -5;

    OrbitControls controls{camera, canvas};

    LOD lod1;
    scene.add(lod1);

    auto material = MeshBasicMaterial::create({{"wireframe", true}});
    for (int z = 0; z <= 5; z++) {
        constexpr float radius = 0.5f;
        int detail = 6 - z;
        auto obj = Mesh::create(IcosahedronGeometry::create(radius, detail), material);
        lod1.addLevel(obj, static_cast<float>(z));
    }

    LOD lod2;
    lod2.copy(lod1);
    scene.add(lod2);

    float spacing = 1;
    lod1.position.x = spacing;
    lod2.position.x = -spacing;

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    HUD hud(renderer);
    FontLoader fontLoader;
    const auto font = *fontLoader.load(std::string(DATA_FOLDER) + "/fonts/typeface/gentilis_bold.typeface.json");

    TextSprite handle1(font, 20.f * monitor::contentScale().first);
    handle1.setColor(Color::gray);
    handle1.setVerticalAlignment(TextSprite::VerticalAlignment::Above);
    hud.add(handle1).setNormalizedPosition({0.f, 0.05f});

    TextSprite handle2(font, 20.f * monitor::contentScale().first);
    handle2.setColor(Color::gray);
    handle2.setVerticalAlignment(TextSprite::VerticalAlignment::Above);
    hud.add(handle2);

    canvas.animate([&]() {
        handle1.setText("LOD1 level: " + std::to_string(lod1.getCurrentLevel()));
        handle2.setText("LOD2 level: " + std::to_string(lod2.getCurrentLevel()));

        renderer.clear();
        renderer.render(scene, camera);
        hud.render();
    });
}
