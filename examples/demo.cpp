
#include "threepp/canvas/Monitor.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;


struct MyGui final: ImguiContext {

    bool colorChanged = false;

    explicit MyGui(const Canvas& canvas, const MeshBasicMaterial& m)
        : ImguiContext(canvas) {
        colorBuf_[0] = m.color.r;
        colorBuf_[1] = m.color.g;
        colorBuf_[2] = m.color.b;
        colorBuf_[3] = m.opacity;
    }

    void onRender() override {

        ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({0, 0}, 0);
        ImGui::Begin("Plane transform");
        ImGui::SliderFloat3("position", posBuf_.data(), -5.f, 5.f);
        ImGui::SliderFloat3("rotation", eulerBuf_.data(), -180.f, 180.f);
        ImGui::ColorEdit4("Color", colorBuf_.data());
        colorChanged = ImGui::IsItemEdited();

        ImGui::End();
    }

    const Vector3& position() {
        pos_.fromArray(posBuf_);
        return pos_;
    }

    const Euler& rotation() {
        euler_.set(math::DEG2RAD * eulerBuf_[0], math::DEG2RAD * eulerBuf_[1], math::DEG2RAD * eulerBuf_[2]);
        return euler_;
    }

    [[nodiscard]] const std::array<float, 4>& color() const {
        return colorBuf_;
    }

private:
    Vector3 pos_;
    Euler euler_;

    std::array<float, 3> posBuf_{};
    std::array<float, 3> eulerBuf_{};
    std::array<float, 4> colorBuf_{0, 0, 0, 1};
};

auto createBox() {

    const auto boxGeometry = BoxGeometry::create();
    const auto boxMaterial = MeshBasicMaterial::create();
    boxMaterial->color.setRGB(1, 0, 0);
    boxMaterial->transparent = true;
    boxMaterial->opacity = 0.1f;
    auto box = Mesh::create(boxGeometry, boxMaterial);

    auto wiredBox = LineSegments::create(WireframeGeometry::create(*boxGeometry));
    wiredBox->material()->as<LineBasicMaterial>()->depthTest = false;
    wiredBox->material()->as<LineBasicMaterial>()->color = Color::gray;
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
    GLRenderer renderer(canvas.size());
    renderer.autoClear = false;

    auto scene = Scene::create();
    scene->background = Color::aliceblue;
    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 1000);
    camera->position.z = 5;

    auto box = createBox();
    scene->add(box);

    auto sphere = createSphere();
    box->add(sphere);

    auto plane = createPlane();
    auto planeMaterial = plane->material()->as<MeshBasicMaterial>();
    scene->add(plane);

    HUD hud(canvas.size());
    FontLoader fontLoader;
    const auto font1 = fontLoader.defaultFont();
    const auto font2 = *fontLoader.load(std::string(DATA_FOLDER) + "/fonts/helvetiker_regular.typeface.json");

    TextGeometry::Options opts1(font1, 40 * monitor::contentScale().first);
    auto hudText1 = Text2D(opts1, "Hello World!");
    hudText1.setColor(Color::black);
    hud.add(hudText1, HUD::Options());

    TextGeometry::Options opts2(font2, 10 * monitor::contentScale().first, 1);
    auto hudText2 = Text2D(opts2);
    hudText2.setColor(Color::red);
    hud.add(hudText2, HUD::Options()
                              .setNormalizedPosition({1, 1})
                              .setHorizontalAlignment(HUD::HorizontalAlignment::RIGHT)
                              .setVerticalAlignment(HUD::VerticalAlignment::TOP));


    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);

        hud.setSize(size);
    });

    MyGui ui(canvas, *planeMaterial);


    Clock clock;
    canvas.animate([&]() {
        const auto dt = clock.getDelta();

        box->rotation.y += 0.5f * dt;

        hudText2.setText("Delta=" + std::to_string(dt));
        hud.needsUpdate(hudText2);

        renderer.clear();
        renderer.render(*scene, *camera);
        hud.apply(renderer);

        ui.render();

        plane->position.copy(ui.position());
        plane->rotation.copy(ui.rotation());

        if (ui.colorChanged) {
            const auto& c = ui.color();
            planeMaterial->color.fromArray(c);
            planeMaterial->opacity = c[3];
            planeMaterial->transparent = c[3] != 1;
        }
    });
}
