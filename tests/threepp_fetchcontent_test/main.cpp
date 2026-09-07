#include <threepp/objects/TextSprite.hpp>
#include <threepp/threepp.hpp>

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

    // Screen-space "HUD" text, anchored to the top-right corner. Replaces the
    // old HUD/Text2D API with a screenSpace TextSprite (Sprite::screenSpace),
    // which the renderer composites over the 3D scene each frame.
    std::shared_ptr<TextSprite> createHudText(const Font &font) {
        auto text = TextSprite::create(font, 20.f);// 20px
        text->setText("Hello World!");
        text->setColor(Color::gray);
        text->setHorizontalAlignment(TextSprite::HorizontalAlignment::Right);
        text->setVerticalAlignment(TextSprite::VerticalAlignment::Below);
        text->screenSpace = true;
        // Screen-space layout is y-up (origin bottom-left): anchor {1,1} is the
        // top-right corner; the negative offset insets down-left from it, and
        // Right/Below keep the text growing inward (on-screen).
        text->screenAnchor = {1.f, 1.f};
        text->position.set(-15.f, -15.f, 0.f);

        return text;
    }

}// namespace

int main() {

    Canvas canvas("threepp demo", {{"aa", 4}});
    GLRenderer renderer(canvas);

    auto camera = PerspectiveCamera::create(50, canvas.aspect());
    camera->position.z = 5;

    OrbitControls controls{*camera, canvas};

    auto scene = Scene::create();
    scene->background = Color::aliceblue;

    auto group = Group::create();
    group->add(createBox({-1, 0, 0}, Color::green));
    group->add(createBox({1, 0, 0}, Color::blue));
    scene->add(group);

    FontLoader fontLoader;
    const auto font = fontLoader.defaultFont();
    scene->add(createHudText(font));

    canvas.onWindowResize([&](const WindowSize &size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    float rotationSpeed = 1;
    canvas.animate([&] {
        const auto dt = clock.getDelta();
        group->rotation.y += rotationSpeed * dt;

        renderer.render(*scene, *camera);
    });
}
