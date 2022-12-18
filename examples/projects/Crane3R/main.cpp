
#include "Crane3R.hpp"
#include "threepp/threepp.hpp"

#include <thread>

using namespace threepp;

#ifdef HAS_IMGUI
#include "../../imgui_helper.hpp"

struct MyUI : public imggui_helper {

    bool mouseHover = false;
    bool jointMode = false;
    bool posMode = false;
    std::array<float, 3> angles{0, -40, 90};
    Vector3 pos{0, 0, 0};

    explicit MyUI(const threepp::Canvas &canvas) : imggui_helper(canvas) {}

    void onRender() override {

        ImGui::SetNextWindowPos({}, 0, {});
        ImGui::SetNextWindowSize({}, 0);
        ImGui::Begin("Crane3R");

        ImGui::Text("Target angles");
        ImGui::SliderFloat("j1", &angles[0], -90, 90);
        jointMode = ImGui::IsItemEdited();
        ImGui::SliderFloat("j2", &angles[1], -80, 0);
        jointMode = jointMode || ImGui::IsItemEdited();
        ImGui::SliderFloat("j3", &angles[2], 40, 140);
        jointMode = jointMode || ImGui::IsItemEdited();


        ImGui::Text("Target pos");
        ImGui::SliderFloat("px", &pos.x, 0, 10);
        posMode = ImGui::IsItemEdited();
        ImGui::SliderFloat("py", &pos.y, 0, 10);
        posMode = posMode || ImGui::IsItemEdited();
        ImGui::SliderFloat("pz", &pos.z, -10, 10);
        posMode = posMode || ImGui::IsItemEdited();

        mouseHover = ImGui::IsWindowHovered();
        ImGui::End();
    }
};
#endif

int main() {

    Canvas canvas{Canvas::Parameters().size({1280, 720}).antialiasing(8)};
    GLRenderer renderer{canvas};
    renderer.setClearColor(Color::aliceblue);

    auto camera = PerspectiveCamera::create(60, canvas.getAspect(), 0.01, 100);
    camera->position.set(15, 8, 15);

    OrbitControls controls(camera, canvas);

    auto scene = Scene::create();

    auto grid = GridHelper::create(20, 10, Color::yellowgreen);
    scene->add(grid);

    auto endEffectorHelper = AxesHelper::create(1);
    scene->add(endEffectorHelper);

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

    auto targetHelper = AxesHelper::create(2);
    scene->add(targetHelper);

#endif
    canvas.animate([&] {
        renderer.render(scene, camera);

        if (crane) {

            auto endEffectorPosition = Crane3R::calculateEndEffectorPosition(crane->getAngles());
            endEffectorHelper->position.copy(endEffectorPosition);

#ifdef HAS_IMGUI
            ui.render();
            controls.enabled = !ui.mouseHover;

            targetHelper->position.copy(ui.pos);

            if (ui.jointMode) {
                auto &angles = ui.angles;
                ui.pos = Crane3R::calculateEndEffectorPosition(angles);
            }
            if (ui.posMode) {
                auto& pos = ui.pos;
                ui.angles = crane->computeAngles(pos);
            }

            crane->setTargetAngles(ui.angles);
#endif

            crane->update();
        }
    });

    t.join();
}
