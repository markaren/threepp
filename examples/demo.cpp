
#include "threepp/canvas/Monitor.hpp"
#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/objects/TextSprite.hpp"
#include "threepp/threepp.hpp"

#include <array>

using namespace threepp;


auto createBox() {

    const auto boxGeometry = BoxGeometry::create();
    const auto boxMaterial = MeshBasicMaterial::create();
    boxMaterial->color.setRGB(1, 0, 0);
    boxMaterial->transparent = true;
    boxMaterial->opacity = 0.1f;
    auto box = Mesh::create(boxGeometry, boxMaterial);

    auto wiredBox = LineSegments::create(WireframeGeometry::create(*boxGeometry));
    wiredBox->materialAs<LineBasicMaterial>()->depthTest = false;
    wiredBox->materialAs<LineBasicMaterial>()->color = Color::gray;
    box->add(wiredBox);

    return box;
}

auto createSphere() {

    const auto sphereGeometry = SphereGeometry::create(0.5f);
    const auto sphereMaterial = MeshBasicMaterial::create();
    sphereMaterial->color.setHex(0x00ff00);
    sphereMaterial->wireframe = true;
    auto sphere = Mesh::create(sphereGeometry, sphereMaterial);
    sphere->position.setX(-1);

    return sphere;
}

auto createPlane() {

    const auto planeGeometry = PlaneGeometry::create(5, 5);
    const auto planeMaterial = MeshBasicMaterial::create();
    planeMaterial->color.setHex(Color::yellow);
    planeMaterial->transparent = true;
    planeMaterial->opacity = 0.5f;
    planeMaterial->side = Side::Double;
    auto plane = Mesh::create(planeGeometry, planeMaterial);
    plane->position.setZ(-2);

    return plane;
}

int main() {

    Canvas canvas("threepp demo", {{"aa", 4}});
    auto renderer = createRenderer(canvas);

    auto scene = Scene::create();
    scene->background = Color::aliceblue;
    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 1000);
    camera->position.z = 5;

    auto box = createBox();
    scene->add(box);

    auto sphere = createSphere();
    box->add(sphere);

    auto plane = createPlane();
    auto planeMaterial = plane->materialAs<MeshBasicMaterial>();
    scene->add(plane);

    FontLoader fontLoader;
    const auto font1 = fontLoader.defaultFont();
    const auto font2 = *fontLoader.load(std::string(DATA_FOLDER) + "/fonts/typeface/gentilis_regular.typeface.json");

    // Screen-space text labels — Sprite::screenSpace + screenAnchor route
    // these through the renderer's ortho overlay automatically, no separate
    // HUD scene / camera / autoClear ritual. Anchor (0,0) = bottom-left of
    // the viewport; (1,1) = top-right. position.xy is the pixel offset from
    // the anchor (negative = "from the opposite edge", CSS-style). Resize
    // is implicit — the renderer samples viewport size each frame.
    auto hudText1 = TextSprite::create(font1, 40 * monitor::contentScale().first);
    hudText1->setText("Hello World!");
    hudText1->setColor(Color::black);
    hudText1->setVerticalAlignment(TextSprite::VerticalAlignment::Above);
    hudText1->screenSpace = true;
    hudText1->screenAnchor.set(0.f, 0.f);
    hudText1->position.set(5.f, 5.f, 0.f);
    scene->add(hudText1);

    auto hudText2 = TextSprite::create(font2, 10 * monitor::contentScale().first);
    hudText2->setColor(Color::red);
    hudText2->setVerticalAlignment(TextSprite::VerticalAlignment::Below);
    hudText2->setHorizontalAlignment(TextSprite::HorizontalAlignment::Right);
    hudText2->screenSpace = true;
    hudText2->screenAnchor.set(1.f, 1.f);
    hudText2->position.set(-5.f, -5.f, 0.f);
    scene->add(hudText2);


    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(size);
    });

    // Plane-transform widgets ride in the shared renderer-settings window;
    // the panel dispatches on the runtime-selected renderer (GL/WGPU/Vulkan)
    // and falls back to tone-map-only controls otherwise.
    std::array<float, 3> posBuf{};
    std::array<float, 3> eulerBuf{};
    std::array<float, 4> colorBuf{planeMaterial->color.r, planeMaterial->color.g,
                                  planeMaterial->color.b, planeMaterial->opacity};
    bool colorChanged = false;
    RendererSettingsUi ui(canvas, *renderer, [&] {
        ImGui::SliderFloat3("position", posBuf.data(), -5.f, 5.f);
        ImGui::SliderFloat3("rotation", eulerBuf.data(), -180.f, 180.f);
        ImGui::ColorEdit4("Color", colorBuf.data());
        colorChanged = ImGui::IsItemEdited();
    }, "Plane transform");


    Clock clock;
    canvas.animate([&] {
        const auto dt = clock.getDelta();

        box->rotation.y += 0.5f * dt;

        hudText2->setText("Delta=" + std::to_string(dt));

        renderer->render(*scene, *camera);
        ui.render();

        plane->position.set(posBuf[0], posBuf[1], posBuf[2]);
        plane->rotation.set(math::DEG2RAD * eulerBuf[0], math::DEG2RAD * eulerBuf[1], math::DEG2RAD * eulerBuf[2]);

        if (colorChanged) {
            planeMaterial->color.fromArray(colorBuf);
            planeMaterial->opacity = colorBuf[3];
            planeMaterial->transparent = colorBuf[3] != 1;
        }
    });
}
