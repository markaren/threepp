
#include "threepp/objects/LOD.hpp"
#include "threepp/threepp.hpp"
#include "utility/FPSCounter.hpp"

#include <cmath>

using namespace threepp;

int main() {

    Canvas canvas("Instancing", {{"aa", 4}, {"vsync", false}});
    GLRenderer renderer(canvas.size());
    renderer.setClearColor(Color::aliceblue);

    int amount = 10;
    int count = static_cast<int>(std::pow(amount, 3));

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 10000);
    camera->position.set(float(amount), float(amount), float(amount));

    OrbitControls controls{*camera, canvas};

    auto light = HemisphereLight::create(0xffffff, 0x888888);
    light->position.set(0, 1, 0);
    scene->add(light);

    auto material = MeshPhongMaterial::create();
    auto geometry = IcosahedronGeometry::create(0.5f, 2);
    auto mesh = InstancedMesh::create(geometry, material, count);

    Matrix4 matrix;
    Color color;
    size_t index = 0;
    float offset = static_cast<float>(amount - 1) / 2;
    for (int x = 0; x < amount; x++) {
        for (int y = 0; y < amount; y++) {
            for (int z = 0; z < amount; z++) {
                matrix.setPosition(offset - float(x), offset - float(y), offset - float(z));
                mesh->setMatrixAt(index, matrix);
                mesh->setColorAt(index, color);
                ++index;
            }
        }
    }

    scene->add(mesh);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Vector2 mouse{-Infinity<float>, -Infinity<float>};
    MouseMoveListener l([&](auto& pos) {
        auto size = canvas.size();
        mouse.x = (pos.x / static_cast<float>(size.width)) * 2 - 1;
        mouse.y = -(pos.y / static_cast<float>(size.height)) * 2 + 1;
    });
    canvas.addMouseListener(&l);

    TextRenderer textRenderer;
    auto& handle = textRenderer.createHandle();

    Clock clock;
    FPSCounter counter;
    Raycaster raycaster;
    std::unordered_map<int, bool> map;
    canvas.animate([&]() {
        raycaster.setFromCamera(mouse, *camera);
        auto intersects = raycaster.intersectObject(*mesh);

        if (!intersects.empty()) {
            auto instanceId = intersects.front().instanceId;
            if (instanceId && !map[*instanceId]) {
                mesh->setColorAt(*instanceId, color.randomize());
                mesh->instanceColor->needsUpdate();
                map[*instanceId] = true;
            }
        }

        counter.update(clock.getElapsedTime());
        handle.setText("FPS: " + std::to_string(counter.fps));

        renderer.render(*scene, *camera);
        renderer.resetState();
        textRenderer.render();
    });
}
