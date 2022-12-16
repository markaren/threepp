
#include "threepp/threepp.hpp"

#include <future>

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


using namespace threepp;

std::shared_ptr<Object3D> loadCrane() {

    OBJLoader loader;
    auto part1 = loader.load("data/models/obj/Crane3R/4200/4200.obj");
    auto part2 = loader.load("data/models/obj/Crane3R/7000/7000.obj");
    part2->position.set(0, 0, 4.2);
    part1->add(part2);
    auto part3 = loader.load("data/models/obj/Crane3R/5200/5200.obj");
    part3->position.set(7, 0, 0);
    part2->add(part3);

    part1->rotateX(-math::PI / 2);

    part1->name = "part1";
    part2->name = "part2";
    part3->name = "part3";

    return part1;
}

int main() {

    Canvas canvas{Canvas::Parameters().antialiasing(8)};
    GLRenderer renderer{canvas};
    renderer.setClearColor(Color::aliceblue);

    auto camera = PerspectiveCamera::create();
    camera->position.set(15, 8, 15);

    OrbitControls controls(camera, canvas);

    auto scene = Scene::create();

    auto grid = GridHelper::create(20);
    scene->add(grid);

    auto light = AmbientLight::create(Color::white);
    scene->add(light);

    bool loaded = false;

    Object3D *part1;
    Object3D *part2;
    Object3D *part3;
    std::thread t([&] {
        auto crane = loadCrane();
        part1 = crane->getObjectByName("part1");
        part2 = crane->getObjectByName("part2");
        part3 = crane->getObjectByName("part3");
        canvas.invokeLater([&, crane] {
            scene->add(crane);
            loaded = true;
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

        if (loaded) {
            ui.render();

            auto &angles = ui.angles;
            part1->rotation.z = math::DEG2RAD*angles[0];
            part2->rotation.y = math::DEG2RAD*angles[1];
            part3->rotation.y = math::DEG2RAD*angles[2];
        }

        controls.enabled = !ui.mouseHover;

#endif
    });

    t.join();
}
