
#include "threepp/threepp.hpp"
#include "Crane3R.hpp"

#include <thread>

using namespace threepp;

#ifdef HAS_IMGUI
#include "../../imgui_helper.hpp"

struct MyUI : public imggui_helper {

    bool mouseHover = false;
    std::array<float, 3> angles;

    explicit MyUI(const threepp::Canvas &canvas) : imggui_helper(canvas) {}

    void onRender() override {

        ImGui::SetNextWindowPos({}, 0, {});
        ImGui::SetNextWindowSize({}, 0);
        ImGui::Begin("Crane3R");

        ImGui::SliderFloat("j1", &angles[0], -90, 90);
        ImGui::SliderFloat("j2", &angles[1], -90, 0);
        ImGui::SliderFloat("j3", &angles[2], 0, 120);

        mouseHover = ImGui::IsWindowHovered();
        ImGui::End();
    }
};
#endif

int main() {

    Canvas canvas{Canvas::Parameters().size({1280, 720}).antialiasing(8)};
    GLRenderer renderer{canvas};
    renderer.setClearColor(Color::aliceblue);

    auto camera = PerspectiveCamera::create(60, canvas.getAspect());
    camera->position.set(15, 8, 15);

    OrbitControls controls(camera, canvas);

    auto scene = Scene::create();

    auto grid = GridHelper::create(20, 10, Color::yellowgreen);
    scene->add(grid);

    auto light = AmbientLight::create(Color::white);
    scene->add(light);

    std::shared_ptr<Crane3R> crane;
    std::thread t([&] {
        crane = Crane3R::create();
        canvas.invokeLater([&, crane] {
            scene->add(crane);
        });
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

#ifdef HAS_IMGUI
    MyUI ui(canvas);
#endif
    canvas.animate([&] {
        renderer.render(scene, camera);

#ifdef HAS_IMGUI

        if (crane) {
            ui.render();

            auto &angles = ui.angles;
            crane->setAngles(angles[0], angles[1], angles[2]);
        }

        controls.enabled = !ui.mouseHover;

#endif
    });

    t.join();
}
