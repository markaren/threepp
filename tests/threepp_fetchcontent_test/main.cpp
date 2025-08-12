#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    std::shared_ptr<Mesh> createBox(const Vector3 &pos, const Color &color) {
        const auto geometry = BoxGeometry::create();
        const auto material = MeshBasicMaterial::create();
        material->color.copy(color);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.copy(pos);

        return mesh;
    }

    std::unique_ptr<HUD> createHUD(WindowSize size) {
        auto hud = std::make_unique<HUD>(size);
        FontLoader fontLoader;
        const auto font = fontLoader.defaultFont();
        TextGeometry::Options opts(font, 20, 5);
        const auto hudText2 = Text2D::create(opts, "Hello World!");
        hudText2->setColor(Color::gray);
        hud->add(hudText2, HUD::Options()
                                  .setNormalizedPosition({1, 1})
                                  .setHorizontalAlignment(HUD::HorizontalAlignment::RIGHT)
                                  .setVerticalAlignment(HUD::VerticalAlignment::TOP));

        return hud;
    }

}// namespace

int main() {

    Canvas canvas("threepp demo", {{"aa", 4}});
    GLRenderer renderer(canvas.size());
    renderer.autoClear = false;// hud

    auto camera = PerspectiveCamera::create(50, canvas.aspect());
    camera->position.z = 5;

    OrbitControls controls{*camera, canvas};

    auto scene = Scene::create();
    scene->background = Color::aliceblue;

    auto group = Group::create();
    group->add(createBox({-1, 0, 0}, Color::green));
    group->add(createBox({1, 0, 0}, Color::blue));
    scene->add(group);

    const auto hud = createHUD(canvas.size());

    canvas.onWindowResize([&](const WindowSize &size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);

        hud->setSize(size);
    });

    Clock clock;
    float rotationSpeed = 1;
    canvas.animate([&] {
        const auto dt = clock.getDelta();
        group->rotation.y += rotationSpeed * dt;

        renderer.clear();//autoClear is false
        renderer.render(*scene, *camera);
        hud->apply(renderer);
    });
}
