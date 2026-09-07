
#include "renderer_factory.hpp"

#include "threepp/objects/LOD.hpp"
#include "threepp/threepp.hpp"

#include "threepp/extras/imgui/RendererSettings.hpp"

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
    auto renderer = createRenderer(canvas);

    auto scene = Scene::create();
    scene->background = Color::aliceblue;
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 10000);
    camera->position.set(static_cast<float>(maxAmount), static_cast<float>(maxAmount), static_cast<float>(maxAmount));

    OrbitControls controls{*camera, canvas};

    auto light = HemisphereLight::create(0xffffff, 0x888888, 2.5f);
    light->position.set(0, 1, 0);
    scene->add(light);

    auto material = MeshPhongMaterial::create();
    auto geometry = IcosahedronGeometry::create(0.5f, 2);
    auto mesh = InstancedMesh::create(geometry, material, static_cast<int>(std::pow(maxAmount, 3)));
    mesh->instanceMatrix()->setUsage(DrawUsage::Dynamic);
    setupInstancedMesh(*mesh, amount);
    scene->add(mesh);

    std::unordered_map<int, bool> colorMap;

    RendererSettingsUi ui(canvas, *renderer, [&] {
        ImGui::SliderInt("Amount", &amount, 2, maxAmount);
        if (ImGui::IsItemEdited()) {
            colorMap.clear();
            setupInstancedMesh(*mesh, amount);
        }
    }, "Settings");

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(size);
    });

    Vector2 mouse{-Infinity<float>, -Infinity<float>};
    MouseMoveListener l([&](auto& pos) {
        const auto size = canvas.size();
        mouse.x = (pos.x / static_cast<float>(size.width())) * 2 - 1;
        mouse.y = -(pos.y / static_cast<float>(size.height())) * 2 + 1;
    });
    canvas.addMouseListener(l);


    Raycaster raycaster;
    canvas.animate([&] {
        raycaster.setFromCamera(mouse, *camera);
        const auto intersects = raycaster.intersectObject(*mesh);

        if (!intersects.empty()) {
            const auto& instanceId = intersects.front().instanceId;
            if (instanceId && !colorMap[*instanceId]) {
                mesh->setColorAt(*instanceId, Color().randomize());
                mesh->instanceColor()->needsUpdate();
                colorMap[*instanceId] = true;
            }
        }

        renderer->render(*scene, *camera);
        ui.render();
    });
}
