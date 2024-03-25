
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    auto createBox(const Vector3& pos, const Color& color) {
        auto geometry = BoxGeometry::create();
        auto material = MeshBasicMaterial::create();
        material->color.copy(color);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.copy(pos);

        return mesh;
    }

}// namespace

int main() {

    Canvas canvas("threepp demo", {{"aa", 4}});
    GLRenderer renderer(canvas.size());
    renderer.autoClear = false; // hud

    auto camera = PerspectiveCamera::create(50, canvas.aspect());
    camera->position.z = 5;

    OrbitControls controls{*camera, canvas};

    auto scene = Scene::create();
    scene->background = Color::aliceblue;

    auto group = Group::create();
    group->add(createBox({-1, 0, 0}, Color::green));
    group->add(createBox({1, 0, 0}, Color::blue));
    scene->add(group);

    HUD hud(canvas);
    FontLoader fontLoader;
    const auto font = fontLoader.defaultFont();
    TextGeometry::Options opts(font, 20, 5);
    auto hudText2 = Text2D(opts, "Hello World!");
    hudText2.setColor(Color::gray);
    hud.add(hudText2, HUD::Options()
                              .setNormalizedPosition({1, 1})
                              .setHorizontalAlignment(threepp::HUD::HorizontalAlignment::RIGHT)
                              .setVerticalAlignment(threepp::HUD::VerticalAlignment::TOP));

    std::array<float, 3> posBuf{};
    ImguiFunctionalContext ui(canvas.windowPtr(), [&] {
        ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({230, 0}, 0);

        ImGui::Begin("Demo");
        ImGui::SliderFloat3("position", posBuf.data(), -1.f, 1.f);
        controls.enabled = !ImGui::IsWindowHovered();
        ImGui::End();
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);

        hud.setSize(size);
    });

    Clock clock;
    float rotationSpeed = 1;
    canvas.animate([&] {
        auto dt = clock.getDelta();
        group->rotation.y += rotationSpeed * dt;

        renderer.clear(); //autoClear is false
        renderer.render(*scene, *camera);
        hud.apply(renderer);

        ui.render();
        group->position.fromArray(posBuf);
    });
}
