
#include "threepp/objects/LOD.hpp"
#include "threepp/threepp.hpp"
#include "utility/FPSCounter.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/objects/TextSprite.hpp"

#include <cmath>

using namespace threepp;

namespace {

    void setupInstancedMesh(InstancedMesh& mesh, int amount) {

        Matrix4 matrix;
        Color color;
        size_t index = 0;
        float offset = static_cast<float>(amount - 1) / 2;
        for (int x = 0; x < amount; x++) {
            for (int y = 0; y < amount; y++) {
                for (int z = 0; z < amount; z++) {
                    matrix.setPosition(offset - static_cast<float>(x), offset - static_cast<float>(y), offset - static_cast<float>(z));
                    mesh.setMatrixAt(index, matrix);
                    mesh.setColorAt(index, color);
                    ++index;
                }
            }
        }
        mesh.setCount(static_cast<int>(std::pow(amount, 3)));
        mesh.instanceMatrix()->needsUpdate();
        mesh.instanceColor()->needsUpdate();

        mesh.computeBoundingSphere();
    }

}// namespace

int main() {

    int amount = 10;
    constexpr int maxAmount = 25;

    Canvas canvas("Instancing", {{"aa", 4}, {"vsync", false}});
    GLRenderer renderer(canvas.size());
    renderer.autoClear = false;
    renderer.setClearColor(Color::aliceblue);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 10000);
    camera->position.set(static_cast<float>(maxAmount), static_cast<float>(maxAmount), static_cast<float>(maxAmount));

    OrbitControls controls{*camera, canvas};

    auto light = HemisphereLight::create(0xffffff, 0x888888);
    light->position.set(0, 1, 0);
    scene->add(light);

    auto material = MeshPhongMaterial::create();
    auto geometry = IcosahedronGeometry::create(0.5f, 2);
    auto mesh = InstancedMesh::create(geometry, material, static_cast<int>(std::pow(maxAmount, 3)));
    mesh->instanceMatrix()->setUsage(DrawUsage::Dynamic);
    setupInstancedMesh(*mesh, amount);
    scene->add(mesh);

    std::unordered_map<int, bool> colorMap;

    ImguiFunctionalContext ui(canvas, [&] {
        float width = 230 * ui.dpiScale();
        ImGui::SetNextWindowPos({float(canvas.size().width()) - width, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({width, 0}, 0);

        ImGui::Begin("Settings");
        ImGui::SliderInt("Amount", &amount, 2, maxAmount);
        if (ImGui::IsItemEdited()) {
            colorMap.clear();
            setupInstancedMesh(*mesh, amount);
        }

        ImGui::End();
    });

    IOCapture capture{};
    capture.preventMouseEvent = [] {
        return ImGui::GetIO().WantCaptureMouse;
    };
    canvas.setIOCapture(&capture);

    HUD hud(renderer);
    FontLoader fontLoader;
    const auto font = *fontLoader.load(std::string(DATA_FOLDER) + "/fonts/typeface/helvetiker_regular.typeface.json");

    auto handle = TextSprite(font, 20.f);
    handle.setColor(Color::black);
    hud.add(handle).setNormalizedPosition({0, 1});

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Vector2 mouse{-Infinity<float>, -Infinity<float>};
    MouseMoveListener l([&](auto& pos) {
        const auto size = canvas.size();
        mouse.x = (pos.x / static_cast<float>(size.width())) * 2 - 1;
        mouse.y = -(pos.y / static_cast<float>(size.height())) * 2 + 1;
    });
    canvas.addMouseListener(l);


    Clock clock;
    FPSCounter counter;
    Raycaster raycaster;
    long long it{0};
    canvas.animate([&]() {
        raycaster.setFromCamera(mouse, *camera);
        const auto& intersects = raycaster.intersectObject(*mesh);

        if (!intersects.empty()) {
            const auto& instanceId = intersects.front().instanceId;
            if (instanceId && !colorMap[*instanceId]) {
                mesh->setColorAt(*instanceId, Color().randomize());
                mesh->instanceColor()->needsUpdate();
                colorMap[*instanceId] = true;
            }
        }

        counter.update(clock.getElapsedTime());
        if (it++ % 60 == 0) {
            handle.setText("FPS: " + std::to_string(counter.fps));
        }

        renderer.clear();
        renderer.render(*scene, *camera);

        hud.render();
        ui.render();
    });
}
